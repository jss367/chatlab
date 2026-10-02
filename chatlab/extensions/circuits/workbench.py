"""The model-side work behind the Circuits page, each step under one model session.

The page never touches the model. It asks for a trace, a feature's details,
or a set of interventions, and each of those holds ChatLab's generation
reservation and model lock for its own duration and no longer.
"""

from __future__ import annotations

import json
import logging
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

    def _held(self, model, progress, cancelled=None):
        blocks = architecture.blocks(model)
        model_id = self.models.loaded_model_id()
        spec = transcoders.spec_for(model_id)
        if spec is None:
            raise ValueError(f"No transcoders are published for {model_id or 'the loaded model'}. "
                             f"Load one of: {', '.join(transcoders.supported_models())}.")
        held = transcoders.loaded(spec, blocks.device)
        if held is None:
            held = transcoders.load(spec, blocks.device,
                                    progress=lambda done, total: progress("Reading transcoders", done, total),
                                    cancelled=cancelled)
        return blocks, held, spec

    def load_transcoders(self, progress, cancelled):
        with self.models.open_session() as session, self._model(session) as model:
            _, held, spec = self._held(model, progress, cancelled)
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
            ids = session.prompt_ids(messages) + (session.encode(prompt["prefix"]) if prompt["prefix"] else [])
        if len(ids) < 2:
            raise ValueError("Write a prompt first.")
        return ids

    def single_tokens(self, session, texts, what):
        ids = []
        for text in texts:
            encoded = session.encode(text)
            if len(encoded) != 1:
                raise ValueError(f"{what}: {text!r} is {len(encoded)} tokens, not one. "
                                 "Check its leading space, or pick a shorter piece.")
            ids.append(encoded[0])
        return ids

    def trace(self, prompt, explain, settings, progress, cancelled):
        """A finished graph for the prompt, labelled and ready to save."""
        with self.models.open_session() as session:
            revision = session.model_revision
            with self._model(session) as model:
                blocks, held, spec = self._held(model, progress, cancelled)
                ids = self.prompt_ids(session, model, prompt)

                def decode(token):
                    return session.decode([int(token)])

                token_ids, contrast = None, None
                if explain["mode"] == "tokens":
                    token_ids = self.single_tokens(session, explain["tokens"], "Tokens to explain")
                    if not token_ids:
                        raise ValueError("List the tokens to explain, one per line.")
                elif explain["mode"] == "contrast":
                    positive = self.single_tokens(session, explain["tokens"], "Pivot tokens")
                    negative = self.single_tokens(session, explain["others"], "Other tokens")
                    contrast = {"positive": positive, "negative": negative,
                                "label": " / ".join(explain["tokens"][:3]) + " vs other"}
                graph = attribution.attribute(blocks, held, ids, decode, settings=settings, token_ids=token_ids,
                                              contrast=contrast, progress=progress, cancelled=cancelled)
                self._describe(graph, blocks, held, decode)
                graph.update(id=uuid4().hex, created=time.time(), model_id=session.model_id,
                             load_id=session.load_id, process_id=PROCESS_ID, model_revision=revision,
                             transcoders=spec.key, prompt=prompt, explain=explain,
                             labels={}, groups={}, effects=None)
                return graph

    def _describe(self, graph, blocks, held, decode):
        """What each kept feature writes into the vocabulary, read off its decoder row."""
        import torch

        features = [n for n in graph["nodes"] if n["kind"] == "feature"]
        weight = blocks.unembed.weight
        for start in range(0, len(features), 64):
            chunk = features[start:start + 64]
            rows = torch.stack([held.decoder_rows(n["layer"], torch.tensor([n["feature"]], device=weight.device))[0]
                                for n in chunk])
            logits = rows.to(weight.dtype) @ weight.T
            top = torch.topk(logits.float(), TOP_LOGITS, dim=-1).indices.tolist()
            bottom = torch.topk(-logits.float(), TOP_LOGITS, dim=-1).indices.tolist()
            for node, up, down in zip(chunk, top, bottom):
                node["promotes"] = [decode(t) for t in up]
                node["suppresses"] = [decode(t) for t in down]
        for node in graph["nodes"]:
            if node["kind"] == "embedding":
                node["text"] = graph["tokens"][node["position"]]

    # Feature details -------------------------------------------------------

    def record(self, graph, node):
        spec = next((s for s in transcoders.CATALOGUE if s.key == graph.get("transcoders")), None)
        if spec is None:
            raise OSError("This graph's transcoders are not in this build's catalogue.")
        with self._lock:
            records = self._records.get(spec.key)
            if records is None:
                records = self._records[spec.key] = transcoders.FeatureRecords(spec, self.data_dir / "features")
        return records.get(node["layer"], node["feature"])

    def ablate(self, graph, node, progress, cancelled):
        """Ablate one feature at its own position and read each target's change."""
        with self.models.open_session() as session:
            self._same_model(graph, session)
            with self._model(session) as model:
                blocks, held, _ = self._held(model, progress, cancelled)
                ids = graph["ids"]
                offset = len(ids) - 1 - node["position"]
                targets = [n for n in graph["nodes"] if n["kind"] == "target"]
                before = interventions.run(blocks, held, ids)["log_probs"]
                after = interventions.run(blocks, held, ids, [(node["layer"], node["feature"], 0.0, [offset])])["log_probs"]
                return {"deltas": [_target_log_odds(t, after) - _target_log_odds(t, before) for t in targets]}

    @staticmethod
    def _same_model(graph, session):
        if graph.get("model_id") != session.model_id:
            raise ValueError(f"This graph was traced on {graph.get('model_id')}; load that model to measure it.")
        revision = graph.get("model_revision")
        if revision:
            if revision != session.model_revision:
                raise ValueError("This graph belongs to another model revision; trace it again on this load.")
        elif graph.get("process_id") != PROCESS_ID or graph.get("load_id") != session.load_id:
            raise ValueError("This graph belongs to another model load; trace it again on this load.")

    def group_effects(self, graph, pivot_texts, alternative_texts, prefix_texts, include_prompt, boost,
                      every_position, progress, cancelled):
        with self.models.open_session() as session:
            self._same_model(graph, session)
            with self._model(session) as model:
                blocks, held, _ = self._held(model, progress, cancelled)
                pivot = self.single_tokens(session, pivot_texts, "Pivot tokens")
                alternatives = self.single_tokens(session, alternative_texts, "Alternatives")
                prefixes = [graph["ids"]] if include_prompt else []
                prompt = graph.get("prompt") or {}
                for text in prefix_texts:
                    prefixes.append(self.prompt_ids(session, model, {**prompt, "prefix": text}))
                if not alternatives:
                    logits = interventions.run(blocks, held, graph["ids"])["log_probs"]
                    ranked = [int(t) for t in logits.argsort(descending=True)[:20].tolist()]
                    alternatives = [t for t in ranked if t not in set(pivot)][:8]
                nodes = {n["id"]: n for n in graph["nodes"]}
                n = len(graph["ids"])
                groups = {name: [(nodes[m]["layer"], nodes[m]["feature"], n - 1 - nodes[m]["position"])
                                 for m in members if m in nodes and nodes[m]["kind"] == "feature"]
                          for name, members in graph["groups"].items()}
                groups = {name: members for name, members in groups.items() if members}
                effects = interventions.group_effects(
                    blocks, held, prefixes, groups, pivot, alternatives, boost=boost,
                    every_position=every_position, progress=lambda d, t: progress("Intervening", d, t),
                    cancelled=cancelled)
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
        return path

    def saved(self):
        directory = self.graphs_dir()
        if not directory.exists():
            return []
        found = []
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            try:
                with path.open(encoding="utf-8") as handle:
                    head = json.load(handle)
                found.append((describe(head), str(path)))
            except (OSError, ValueError, KeyError, TypeError):
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
        layers, tokens = graph["layers"], graph["tokens"]
        if type(layers) is not int or not 1 <= layers <= 256:
            raise ValueError
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= attribution.MAX_PREFIX:
            raise ValueError
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError
        token_ids = graph["ids"]
        if (not isinstance(token_ids, list) or len(token_ids) != len(tokens)
                or not all(type(token) is int and 0 <= token < 2 ** 31 for token in token_ids)):
            raise ValueError
        nodes, edges = graph["nodes"], graph["edges"]
        max_nodes = min(20000, 4096 + (layers + 1) * len(tokens) + attribution.MAX_TARGETS)
        if (not isinstance(nodes, list) or not 1 <= len(nodes) <= max_nodes
                or not isinstance(edges, list) or len(edges) > min(200000, len(nodes) ** 2)):
            raise ValueError
        ids = {n["id"] for n in nodes}
        if len(ids) != len(nodes) or not all(isinstance(value, str) for value in ids):
            raise ValueError
        limits = {"feature": 4096, "error": layers * len(tokens),
                  "embedding": len(tokens), "target": attribution.MAX_TARGETS}
        for kind, limit in limits.items():
            if sum(n["kind"] == kind for n in nodes) > limit:
                raise ValueError
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
            float(node["influence"]), float(node["effect"])
        for edge in graph["edges"]:
            if edge["source"] not in ids or edge["target"] not in ids:
                raise ValueError
            float(edge["weight"])
        graph["tokens"] = [str(t) for t in graph["tokens"]]
        int(graph["layers"])
        graph.setdefault("labels", {})
        graph.setdefault("groups", {})
        graph.setdefault("effects", None)
        graph.setdefault("id", uuid4().hex)
        validate_graph_id(graph["id"])
        if not isinstance(graph["labels"], dict) or not isinstance(graph["groups"], dict):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("That file is not a valid attribution graph.") from exc
    return graph


def _target_log_odds(target, log_probs):
    import torch

    if target.get("target_kind") == "contrast":
        positive = torch.logsumexp(log_probs[target["positive"]], 0)
        negative = torch.logsumexp(log_probs[target["negative"]], 0)
        return float(positive - negative)
    return float(log_probs[target["token_id"]])
