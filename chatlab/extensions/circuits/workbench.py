"""The model-side work behind the Circuits page, each step under one model session.

The page never touches the model. It asks for a trace, a feature's details,
or a set of interventions, and each of those holds ChatLab's generation
reservation and model lock for its own duration and no longer.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from chatlab.extension_api import write_private_text

from . import architecture, attribution, interventions, transcoders

logger = logging.getLogger(__name__)

PROCESS_ID = uuid4().hex
TOP_LOGITS = 6
MAX_GRAPH_BYTES = 256 * 1024 ** 2


def parse_tokens(text):
    """Token texts, one per line, with ``\\n`` and ``\\t`` typed as escapes.

    Leading spaces are kept: in most vocabularies " Wait" and "Wait" are
    different tokens.
    """
    lines = [line.rstrip("\r") for line in (text or "").split("\n")]
    return [line.replace("\\n", "\n").replace("\\t", "\t") for line in lines if line.strip("\r") != ""]


def parse_prefixes(text):
    """Assistant prefixes separated by lines holding only ``---``."""
    blocks, current = [], []
    for line in (text or "").split("\n"):
        if line.strip() == "---":
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    blocks.append("\n".join(current))
    return [block for block in blocks if block.strip()]


class Workbench:
    def __init__(self, models, data_dir):
        self.models = models
        self.data_dir = Path(data_dir)
        self._records = {}
        self._lock = threading.Lock()
        self._sessions = {}

    # Status -------------------------------------------------------------

    def status(self):
        model_id = self.models.loaded_model_id()
        spec = transcoders.spec_for(model_id)
        return {"model_id": model_id, "spec": spec,
                "downloaded": bool(spec) and transcoders.downloaded(spec),
                "in_memory": bool(spec) and transcoders.in_memory(spec)}

    # Running work off the event thread, with progress --------------------

    def cancel(self, owner):
        with self._lock:
            session = self._sessions.get(owner)
            if session is not None:
                session["cancelled"].set()

    def background(self, owner, work):
        """Run ``work(progress, cancelled)`` on a thread, yielding its progress text.

        The last item yielded is ``("done", result)``. Exceptions raised by
        the work are raised here.
        """
        updates = queue.Queue()
        cancelled = threading.Event()
        with self._lock:
            if owner in self._sessions:
                raise ValueError("This view is already running something. Wait for it or press Stop.")
            self._sessions[owner] = {"cancelled": cancelled}

        def progress(stage, done, total):
            updates.put(("progress", stage, done, total))

        def run():
            try:
                if cancelled.is_set():
                    raise attribution.Cancelled("Stopped before queued work started.")
                updates.put(("done", work(progress, cancelled.is_set)))
            except BaseException as exc:  # handed to the generator
                updates.put(("error", exc))

        thread = threading.Thread(target=run, daemon=True, name="circuits-work")
        thread.start()
        try:
            while True:
                item = updates.get()
                if item[0] == "progress":
                    yield item
                elif item[0] == "done":
                    yield item
                    return
                else:
                    raise item[1]
        finally:
            cancelled.set()
            thread.join()
            with self._lock:
                self._sessions.pop(owner, None)

    # Tracing ---------------------------------------------------------------

    def _model(self, session):
        return session.transformers_model()

    def _held(self, model, progress, cancelled=None, revision=None, model_revision=None):
        blocks = architecture.blocks(model)
        model_id = self.models.loaded_model_id()
        spec = transcoders.spec_for(model_id)
        if spec is None:
            raise ValueError(f"No transcoders are published for {model_id or 'the loaded model'}. "
                             f"Load one of: {', '.join(transcoders.supported_models())}.")
        transcoders.check_model_revision(spec, model_revision)
        held = transcoders.loaded(spec, blocks.device, revision=revision)
        if held is None:
            held = transcoders.load(spec, blocks.device,
                                    progress=lambda done, total: progress("Reading transcoders", done, total),
                                    cancelled=cancelled, revision=revision)
        return blocks, held, spec

    def load_transcoders(self, progress, cancelled):
        with self.models.open_session() as session, self._model(session) as model:
            _, held, spec = self._held(model, progress, cancelled, model_revision=session.model_revision)
            return spec

    def prompt_ids(self, session, model, prompt):
        if prompt["raw"]:
            bos = getattr(model.config, "bos_token_id", None)
            if bos is None and getattr(model.config, "text_config", None) is not None:
                bos = model.config.text_config.bos_token_id
            ids = ([int(bos)] if bos is not None else []) + session.encode(prompt["user"] + prompt["prefix"])
        else:
            messages = []
            if prompt["system"].strip():
                messages.append({"role": "system", "content": prompt["system"]})
            messages.append({"role": "user", "content": prompt["user"]})
            ids = session.prompt_ids(messages)
            if prompt["prefix"]:
                ids = ids + session.encode_replacement(ids, prompt["prefix"])
        if not 2 <= len(ids) <= attribution.MAX_PREFIX:
            raise ValueError(f"Attribution needs between 2 and {attribution.MAX_PREFIX} prompt tokens.")
        return ids

    def encoded_prompt(self, session, prompt):
        for key in ("system", "user", "prefix"):
            if not isinstance(prompt.get(key), str) or len(prompt[key]) > 32768:
                raise ValueError("Prompt fields must be text of at most 32768 characters each.")
        # encode_replacement takes the host lock itself. Raw encoding only
        # needs the locked model to read its BOS configuration.
        if prompt["raw"]:
            with self._model(session) as model:
                return self.prompt_ids(session, model, prompt)
        return self.prompt_ids(session, None, prompt)

    def single_tokens(self, session, texts, what, cancelled=None):
        ids = []
        for text in texts:
            if cancelled is not None:
                attribution._check_cancelled(cancelled)
            encoded = session.encode(text)
            if len(encoded) != 1:
                raise ValueError(f"{what}: {text!r} is {len(encoded)} tokens, not one. "
                                 "Check its leading space, or pick a shorter piece.")
            ids.append(encoded[0])
        return ids

    def trace(self, prompt, explain, settings, progress, cancelled):
        """A finished graph for the prompt, labelled and ready to save."""
        settings.check()
        for key in ("tokens", "others"):
            entries = explain.get(key, [])
            if not isinstance(entries, list) or len(entries) > attribution.MAX_CHOSEN_TARGETS:
                raise ValueError(f"Use at most {attribution.MAX_CHOSEN_TARGETS:,} explanation entries per list.")
            if any(not isinstance(text, str) or len(text) > 32768 for text in entries):
                raise ValueError("Explanation entries must be text of at most 32768 characters each.")
        attribution._check_cancelled(cancelled)
        with self.models.open_session() as session:
            revision = session.model_revision
            ids = self.encoded_prompt(session, prompt)
            token_ids, contrast = None, None
            if explain["mode"] == "tokens":
                token_ids = self.single_tokens(session, explain["tokens"], "Tokens to explain", cancelled)
                if not token_ids:
                    raise ValueError("List the tokens to explain, one per line.")
                if len(set(token_ids)) > attribution.MAX_CHOSEN_TARGETS:
                    raise ValueError(f"Choose at most {attribution.MAX_CHOSEN_TARGETS:,} distinct target tokens.")
            elif explain["mode"] == "contrast":
                positive = self.single_tokens(session, explain["tokens"], "Pivot tokens", cancelled)
                negative = self.single_tokens(session, explain["others"], "Other tokens", cancelled)
                if not positive or not negative:
                    raise ValueError("A contrast needs tokens on both sides.")
                if len(set(positive) | set(negative)) > attribution.MAX_CHOSEN_TARGETS:
                    raise ValueError(f"A contrast supports at most {attribution.MAX_CHOSEN_TARGETS:,} distinct tokens.")
                if set(positive) & set(negative):
                    raise ValueError("A token cannot be on both sides of the contrast.")
                contrast = {"positive": positive, "negative": negative,
                            "label": " / ".join(explain["tokens"][:3]) + " vs other"}
            with self._model(session) as model:
                blocks, held, spec = self._held(model, progress, cancelled, model_revision=session.model_revision)

                def decode(token):
                    return session.decode([int(token)])

                graph = attribution.attribute(blocks, held, ids, decode, settings=settings, token_ids=token_ids,
                                              contrast=contrast, progress=progress, cancelled=cancelled)
                self._describe(graph, blocks, held, decode, cancelled)
                graph.update(id=uuid4().hex, created=time.time(), model_id=session.model_id,
                             load_id=session.load_id, process_id=PROCESS_ID, model_revision=revision,
                             transcoders=spec.key, transcoder_revision=held.revision,
                             transcoder_load_id=held.load_id,
                             training_model_revision=spec.training_model_revision,
                             checkpoint_compatibility="verified" if spec.training_model_revision else "unverified",
                             prompt=prompt, explain=explain,
                             labels={}, groups={}, effects=None)
                return graph

    def _describe(self, graph, blocks, held, decode, cancelled=None):
        """What each kept feature writes into the vocabulary, read off its decoder row."""
        import torch

        features = [n for n in graph["nodes"] if n["kind"] == "feature"]
        weight = blocks.unembed.weight
        for start in range(0, len(features), 64):
            if cancelled and cancelled():
                raise attribution.Cancelled()
            chunk = features[start:start + 64]
            rows = torch.stack([held.decoder_rows(n["layer"], torch.tensor([n["feature"]], device=weight.device))[0]
                                for n in chunk])
            logits = rows.to(weight.dtype) @ weight.T
            top = torch.topk(logits.float(), TOP_LOGITS, dim=-1).indices.tolist()
            bottom = torch.topk(-logits.float(), TOP_LOGITS, dim=-1).indices.tolist()
            for node, up, down in zip(chunk, top, bottom):
                node["promotes"] = [decode(t) for t in up]
                node["suppresses"] = [decode(t) for t in down]
        if cancelled and cancelled():
            raise attribution.Cancelled()
        for node in graph["nodes"]:
            if node["kind"] == "embedding":
                node["text"] = graph["tokens"][node["position"]]

    # Feature details -------------------------------------------------------

    def records(self, spec, revision=None):
        """Feature records cached separately for each immutable transcoder revision."""
        with self._lock:
            key = (spec.key, revision)
            records = self._records.get(key)
            if records is None:
                records = self._records[key] = transcoders.FeatureRecords(
                    spec, self.data_dir / "features", revision=revision)
        return records

    def record(self, graph, node):
        spec = next((s for s in transcoders.CATALOGUE if s.key == graph.get("transcoders")), None)
        if spec is None:
            raise OSError("This graph's transcoders are not in this build's catalogue.")
        return self.records(spec, revision=graph.get("transcoder_revision")).get(node["layer"], node["feature"])

    def ablate(self, graph, node, progress, cancelled):
        """Ablate one feature at its own position and read each target's change."""
        with self.models.open_session() as session:
            self._same_model(graph, session)
            with self._model(session) as model:
                blocks, held, _ = self._held(model, progress, cancelled, revision=graph.get("transcoder_revision"),
                                            model_revision=session.model_revision)
                self._same_transcoders(graph, held)
                self._measurement_ids(graph, blocks)
                ids = graph["ids"]
                offset = len(ids) - 1 - node["position"]
                targets = [n for n in graph["nodes"] if n["kind"] == "target"]
                before = interventions.run(blocks, held, ids, cancelled=cancelled)["log_probs"]
                attribution._check_cancelled(cancelled)
                after = interventions.run(blocks, held, ids, [(node["layer"], node["feature"], 0.0, [offset])],
                                          cancelled=cancelled)["log_probs"]
                return {"deltas": [_target_log_odds(t, after) - _target_log_odds(t, before) for t in targets]}

    @staticmethod
    def _measurement_ids(graph, blocks):
        vocabulary = min(blocks.embed.weight.shape[0], blocks.unembed.weight.shape[0])
        values = list(graph["ids"])
        for node in graph["nodes"]:
            if node["kind"] == "target":
                values.extend([node["token_id"]] if node["target_kind"] == "token"
                              else node["positive"] + node["negative"])
        if any(type(token) is not int or not 0 <= token < vocabulary for token in values):
            raise ValueError("This graph contains token IDs outside the loaded model vocabulary.")
        if graph["layers"] != len(blocks.layers):
            raise ValueError("This graph has a different layer count from the loaded model.")

    @staticmethod
    def _same_transcoders(graph, held):
        revision = graph.get("transcoder_revision")
        if revision:
            if held.revision != revision:
                raise ValueError("This graph belongs to another transcoder revision; trace it again.")
        elif graph.get("transcoder_load_id") != held.load_id:
            raise ValueError("This graph belongs to another transcoder load; trace it again.")

    @staticmethod
    def _same_model(graph, session):
        if graph.get("model_id") != session.model_id:
            raise ValueError(f"This graph was traced on {graph.get('model_id')}; load that model to measure it.")
        spec = transcoders.spec_for(session.model_id)
        if spec is not None and graph.get("transcoders") != spec.key:
            raise ValueError("This graph belongs to another transcoder set; trace it again on this load.")
        revision = graph.get("model_revision")
        if revision:
            if revision != session.model_revision:
                raise ValueError("This graph belongs to another model revision; trace it again on this load.")
        elif graph.get("process_id") != PROCESS_ID or graph.get("load_id") != session.load_id:
            raise ValueError("This graph belongs to another model load; trace it again on this load.")

    def group_effects(self, graph, pivot_texts, alternative_texts, prefix_texts, include_prompt, boost,
                      every_position, progress, cancelled):
        for entries, limit in ((pivot_texts, 4096), (alternative_texts, 4096),
                               (prefix_texts, interventions.MAX_PREFIXES - bool(include_prompt))):
            if not isinstance(entries, list) or len(entries) > limit:
                raise ValueError(f"Use at most {limit} entries in each intervention list.")
            if any(not isinstance(text, str) or len(text) > 32768 for text in entries):
                raise ValueError("Intervention entries must be text of at most 32768 characters each.")
        attribution._check_cancelled(cancelled)
        with self.models.open_session() as session:
            self._same_model(graph, session)
            pivot = self.single_tokens(session, pivot_texts, "Pivot tokens", cancelled)
            alternatives = self.single_tokens(session, alternative_texts, "Alternatives", cancelled)
            if len(pivot) > 4096 or len(alternatives) > 4096:
                raise ValueError("Use at most 4096 distinct pivot tokens and 4096 alternatives.")
            if not pivot:
                raise ValueError("Name at least one pivot token.")
            if not graph["groups"]:
                raise ValueError("Make at least one group from the graph first.")
            if not 0 <= boost <= 100:
                raise ValueError("The boost factor must be between 0 and 100.")
            prefixes = [graph["ids"]] if include_prompt else []
            prompt = graph.get("prompt") or {}
            for text in prefix_texts:
                attribution._check_cancelled(cancelled)
                prefixes.append(self.encoded_prompt(session, {**prompt, "prefix": text}))
            if not prefixes or len(prefixes) > interventions.MAX_PREFIXES:
                raise ValueError(f"Use between 1 and {interventions.MAX_PREFIXES} prefixes.")
            with self._model(session) as model:
                blocks, held, _ = self._held(model, progress, cancelled, revision=graph.get("transcoder_revision"),
                                            model_revision=session.model_revision)
                self._same_transcoders(graph, held)
                self._measurement_ids(graph, blocks)
                nodes = {n["id"]: n for n in graph["nodes"]}
                n = len(graph["ids"])
                groups = {name: [(nodes[m]["layer"], nodes[m]["feature"], n - 1 - nodes[m]["position"])
                                 for m in members if m in nodes and nodes[m]["kind"] == "feature"]
                          for name, members in graph["groups"].items()}
                groups = {name: members for name, members in groups.items() if members}
                effects = interventions.group_effects(
                    blocks, held, prefixes, groups, pivot, alternatives, boost=boost,
                    every_position=every_position, progress=lambda d, t: progress("Intervening", d, t),
                    cancelled=cancelled, auto_alternatives=not alternatives)
                alternatives = effects["alternatives"]
                effects["token_text"] = {str(t): session.decode([t]) for t in [*pivot, *alternatives]}
                return effects

    # Saving -----------------------------------------------------------------

    def graphs_dir(self):
        return self.data_dir / "graphs"

    def save(self, graph):
        validate_graph_id(graph.get("id"))
        text = json.dumps(graph)
        if len(text.encode("utf-8")) > MAX_GRAPH_BYTES:
            raise OSError("the graph is too large to save")
        path = self.graphs_dir() / f"{graph['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(path, text)
        metadata = path.parent / ".metadata" / path.name
        metadata.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(metadata, json.dumps({"label": describe(graph)[:2000]}))
        return path

    def saved(self):
        directory = self.graphs_dir()
        if not directory.exists():
            return []
        found = []
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            try:
                metadata = directory / ".metadata" / path.name
                label = f"Saved graph · {path.stem[:128]}"
                if metadata.exists() and metadata.stat().st_size <= 16384:
                    head = json.loads(metadata.read_text(encoding="utf-8"))
                    if isinstance(head, dict) and isinstance(head.get("label"), str):
                        label = head["label"][:2000]
                found.append((label, str(path)))
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                continue
        return found


def describe(graph):
    prompt = graph.get("prompt") or {}
    text = (prompt.get("prefix") or prompt.get("user") or "").strip().replace("\n", " ")
    when = time.strftime("%b %d %H:%M", time.localtime(graph.get("created", 0)))
    return f"{when} · {graph.get('model_id', '?')} · {text[-40:]}"


def validate_graph_id(value):
    """A graph ID is a single safe filename component, including older short IDs."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("The graph id must be a safe filename component.")


def load_graph(path):
    """Read a saved graph, refusing anything that is not one."""
    path = Path(path)
    if path.stat().st_size > MAX_GRAPH_BYTES:
        raise ValueError("That file is too large to be a saved graph.")
    with path.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    if not isinstance(graph, dict) or graph.get("format") != attribution.FORMAT:
        raise ValueError("That file is not a saved attribution graph.")
    try:
        created = graph.get("created", 0)
        if not isinstance(created, (int, float)) or not math.isfinite(created):
            raise ValueError
        try:
            time.localtime(created)
        except (OverflowError, OSError) as exc:
            raise ValueError from exc
        revision = graph.get("transcoder_revision")
        if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise ValueError
        for key in ("prompt", "explain", "stats"):
            graph.setdefault(key, {})
            if not isinstance(graph[key], dict):
                raise ValueError
        for key in ("system", "user", "prefix"):
            if not isinstance(graph["prompt"].get(key), str) or len(graph["prompt"][key]) > 32768:
                raise ValueError
        if not isinstance(graph["prompt"].get("raw"), bool):
            raise ValueError
        for key in ("tokens", "alternatives"):
            values = graph["explain"].get(key, [])
            if not isinstance(values, list) or len(values) > 4096 or any(not isinstance(v, str) or len(v) > 4096 for v in values):
                raise ValueError
        for key in ("kept_features", "traced_features", "active_features"):
            value = graph["stats"].get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError
        value = float(graph["stats"].get("error_share", 0))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError
        graph["stats"]["error_share"] = value
        layers, tokens = graph["layers"], graph["tokens"]
        if type(layers) is not int or not 1 <= layers <= 256:
            raise ValueError
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= attribution.MAX_PREFIX:
            raise ValueError
        if not all(isinstance(token, str) and len(token) <= 4096 for token in tokens):
            raise ValueError
        token_ids = graph["ids"]
        if (not isinstance(token_ids, list) or len(token_ids) != len(tokens)
                or not all(type(token) is int and 0 <= token < 2 ** 31 for token in token_ids)):
            raise ValueError
        spec = next((s for s in transcoders.CATALOGUE if s.key == graph.get("transcoders")), None)
        width = spec.width if spec is not None else graph.get("transcoder_width")
        if type(width) is not int or not 1 <= width <= 1000000:
            raise ValueError
        nodes, edges = graph["nodes"], graph["edges"]
        max_nodes = min(20000, 4096 + (layers + 1) * len(tokens) + attribution.MAX_CHOSEN_TARGETS)
        if (not isinstance(nodes, list) or not 1 <= len(nodes) <= max_nodes
                or not isinstance(edges, list) or len(edges) > min(200000, len(nodes) ** 2)):
            raise ValueError
        ids = {n["id"] for n in nodes}
        if len(ids) != len(nodes) or not all(isinstance(value, str) and 1 <= len(value) <= 128 for value in ids):
            raise ValueError
        limits = {"feature": 4096, "error": layers * len(tokens),
                  "embedding": len(tokens), "target": attribution.MAX_CHOSEN_TARGETS}
        for kind, limit in limits.items():
            if sum(n["kind"] == kind for n in nodes) > limit:
                raise ValueError
        feature_coordinates = set()
        residual_coordinates = set()
        target_signatures = set()
        for node in graph["nodes"]:
            if node["kind"] not in ("feature", "error", "embedding", "target"):
                raise ValueError
            layer, position = node["layer"], node["position"]
            if type(layer) is not int or type(position) is not int or not 0 <= position < len(tokens):
                raise ValueError
            if node["kind"] == "embedding":
                valid_layer = layer == -1
            elif node["kind"] == "target":
                valid_layer = layer == layers
            else:
                valid_layer = 0 <= layer < layers
            if not valid_layer:
                raise ValueError
            for field in ("influence", "effect"):
                node[field] = float(node[field])
                if not math.isfinite(node[field]):
                    raise ValueError
            if "text" in node and (not isinstance(node["text"], str) or len(node["text"]) > 4096):
                raise ValueError
            for field in ("promotes", "suppresses"):
                values = node.get(field, [])
                if not isinstance(values, list) or len(values) > 4096 or any(not isinstance(v, str) or len(v) > 4096 for v in values):
                    raise ValueError
            kind = node["kind"]
            if kind in ("embedding", "error"):
                coordinate = (kind, layer, position)
                if coordinate in residual_coordinates:
                    raise ValueError
                residual_coordinates.add(coordinate)
            if kind == "feature":
                feature = node["feature"]
                if type(feature) is not int or not 0 <= feature < width:
                    raise ValueError
                coordinate = (layer, position, feature)
                if coordinate in feature_coordinates:
                    raise ValueError
                feature_coordinates.add(coordinate)
                node["activation"] = float(node["activation"])
                if not math.isfinite(node["activation"]) or node["activation"] < 0:
                    raise ValueError
            elif kind == "embedding":
                if node["token_id"] != token_ids[position]:
                    raise ValueError
            elif kind == "target":
                if node["target_kind"] == "token":
                    values = [node["token_id"]]
                elif node["target_kind"] == "contrast":
                    positive, negative = node["positive"], node["negative"]
                    if (not isinstance(positive, list) or not isinstance(negative, list)
                            or not positive or not negative or len(positive) + len(negative) > attribution.MAX_CHOSEN_TARGETS
                            or len(set(positive)) != len(positive) or len(set(negative)) != len(negative)
                            or set(positive) & set(negative)):
                        raise ValueError
                    values = positive + negative
                else:
                    raise ValueError
                if not all(type(value) is int and 0 <= value < 2 ** 31 for value in values):
                    raise ValueError
                signature = (("token", node["token_id"]) if node["target_kind"] == "token" else
                             ("contrast", tuple(sorted(node["positive"])), tuple(sorted(node["negative"]))))
                if signature in target_signatures:
                    raise ValueError
                target_signatures.add(signature)
                node["probability"] = float(node["probability"])
                if not isinstance(node["text"], str) or not 0 <= node["probability"] <= 1:
                    raise ValueError
        edge_pairs = set()
        for edge in graph["edges"]:
            pair = (edge["source"], edge["target"])
            if pair in edge_pairs:
                raise ValueError
            edge_pairs.add(pair)
            if edge["source"] not in ids or edge["target"] not in ids:
                raise ValueError
            weight = float(edge["weight"])
            if not math.isfinite(weight):
                raise ValueError
            edge["weight"] = weight
        graph["tokens"] = [str(t) for t in graph["tokens"]]
        int(graph["layers"])
        graph.setdefault("labels", {})
        graph.setdefault("groups", {})
        graph.setdefault("effects", None)
        graph.setdefault("id", uuid4().hex)
        validate_graph_id(graph["id"])
        if not isinstance(graph["labels"], dict) or not isinstance(graph["groups"], dict):
            raise ValueError
        feature_ids = {node["id"] for node in nodes if node["kind"] == "feature"}
        if len(graph["labels"]) > 4096 or any(
                key not in feature_ids or not isinstance(value, str) or len(value) > 4096
                for key, value in graph["labels"].items()):
            raise ValueError
        if len(graph["groups"]) > 4096:
            raise ValueError
        if graph["effects"] is not None:
            _validate_effects(graph["effects"], graph["groups"])
        memberships = set()
        for name, members in graph["groups"].items():
            if (not isinstance(name, str) or not 1 <= len(name) <= 60 or not isinstance(members, list) or len(members) > 4096
                    or not all(isinstance(member, str) and member in feature_ids for member in members)):
                raise ValueError
            if len(set(members)) != len(members) or memberships.intersection(members):
                raise ValueError
            memberships.update(members)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("That file is not a valid attribution graph.") from exc
    return graph


def _validate_effects(effects, groups):
    """Normalize the complete saved intervention schema before any renderer uses it."""
    if not isinstance(effects, dict) or type(effects["prefixes"]) is not int or not 1 <= effects["prefixes"] <= 4096:
        raise ValueError
    effects["boost"] = float(effects["boost"])
    if not math.isfinite(effects["boost"]) or not 0 <= effects["boost"] <= 100 or not isinstance(effects["every_position"], bool):
        raise ValueError
    tokens = []
    if not effects["pivot"]:
        raise ValueError
    for key in ("pivot", "alternatives"):
        values = effects[key]
        if not isinstance(values, list) or len(values) > 4096 or any(type(t) is not int or not 0 <= t < 2 ** 31 for t in values):
            raise ValueError
        tokens.extend(values)
    allowed_token_keys = {str(t) for t in tokens}
    pivot_keys = {str(t) for t in effects["pivot"]}
    def summary(value):
        if not isinstance(value, dict) or not isinstance(value["tokens"], dict) or len(value["tokens"]) > 8192:
            raise ValueError
        if (len({str(key) for key in value["tokens"]}) != len(value["tokens"])
                or {str(key) for key in value["tokens"]} != allowed_token_keys):
            raise ValueError
        for key, number in value["tokens"].items():
            if str(key) not in allowed_token_keys:
                raise ValueError
            number = float(number)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError
            value["tokens"][key] = number
        if math.fsum(value["tokens"].values()) > 1 + 1e-6:
            raise ValueError
        value["pivot"] = float(value["pivot"])
        if not math.isfinite(value["pivot"]) or not 0 <= value["pivot"] <= 1:
            raise ValueError
        total = math.fsum(number for key, number in value["tokens"].items() if str(key) in pivot_keys)
        if not math.isclose(value["pivot"], total, rel_tol=1e-6, abs_tol=0.):
            raise ValueError
    summary(effects["baseline"])
    if not isinstance(effects["groups"], dict) or any(name not in groups for name in effects["groups"]):
        raise ValueError
    for effect in effects["groups"].values():
        if not isinstance(effect, dict) or type(effect["active_prefixes"]) is not int or not 0 <= effect["active_prefixes"] <= effects["prefixes"]:
            raise ValueError
        summary(effect["ablate"])
        summary(effect["boost"])
    texts = effects.get("token_text", {})
    if not isinstance(texts, dict) or len(texts) > 8192 or any(not isinstance(v, str) or len(v) > 4096 for v in texts.values()):
        raise ValueError


def measured_alternatives(blocks, held, prefixes, pivot, cancelled):
    probabilities = None
    for ids in prefixes:
        if cancelled and cancelled():
            raise attribution.Cancelled()
        values = interventions.run(blocks, held, ids)["log_probs"].cpu().double().exp()
        probabilities = values if probabilities is None else probabilities + values
    return automatic_alternatives(probabilities / len(prefixes), pivot)


def automatic_alternatives(log_probs, pivot):
    excluded = set(pivot)
    ranked = log_probs.argsort(descending=True).tolist()
    return [int(t) for t in ranked if t not in excluded][:8]


def _target_log_odds(target, log_probs):
    import torch

    if target.get("target_kind") == "contrast":
        positive = torch.logsumexp(log_probs[target["positive"]], 0)
        negative = torch.logsumexp(log_probs[target["negative"]], 0)
        return float(positive - negative)
    return float(log_probs[target["token_id"]])
