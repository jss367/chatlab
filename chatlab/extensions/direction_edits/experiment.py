"""Inject, erase and read: the passes one page load makes, and the numbers drawn from them.

Each condition reads its prefix and then the passage, tokenized apart, so
every condition holds the same passage tokens and a token range means the
same tokens in all of them. Per condition there is a reference pass with
no injection, an injected pass, an injected and edited pass, and, when
asked for, an injected pass with an edit of the same size along a random
direction. Each is read along the directions at every block, and, when a
target word is given, through the imported Jacobian lens.

Token ranges are counted from 1 at the first passage token and include
both ends. Blocks are counted from 0, as everywhere in ChatLab: block ``n``
is the output a steering vector at layer ``n`` is added to.
"""
from __future__ import annotations

import math
import time

import numpy as np

from chatlab.extension_api import normalize_steering

from . import files

ERASE, CLAMP, ADD = "erase", "clamp", "add"
MODES = (ERASE, CLAMP, ADD)
MAX_CONDITIONS = 6
# One pass keeps no cache and holds every layer's attention over the whole
# sequence; this is the host's flat cap on one reading.
READ_LIMIT = 4096
MAX_PASSAGE_CHARACTERS = 65536
SETTINGS = ("no_edit", "edit", "random")


def _span(value, what, *, base=1):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(type(item) is int for item in value) or not base <= value[0] <= value[1]):
        lowest = "1" if base else "0"
        raise ValueError(f"Give {what} as a first and last number, from {lowest}, the first no larger than the last.")
    return [value[0], value[1]]


def _number(value, what):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number.")
    return float(value)


def normalize_inputs(value):
    """Check what the reader asked for before the model is touched; a clean copy."""
    passage = value.get("passage")
    if not isinstance(passage, str) or not passage.strip() or len(passage) > MAX_PASSAGE_CHARACTERS:
        raise ValueError(f"Write a passage of at most {MAX_PASSAGE_CHARACTERS:,} characters.")
    conditions = value.get("conditions")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= MAX_CONDITIONS:
        raise ValueError(f"Give 1 to {MAX_CONDITIONS} conditions.")
    clean_conditions, names = [], set()
    for condition in conditions:
        name, prefix = condition.get("name"), condition.get("prefix", "")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError("Name every condition, in at most 60 characters.")
        if not isinstance(prefix, str) or len(prefix) > MAX_PASSAGE_CHARACTERS:
            raise ValueError(f"A prefix may be at most {MAX_PASSAGE_CHARACTERS:,} characters.")
        if name.strip() in names:
            raise ValueError(f"Two conditions are named {name.strip()}; name each one differently.")
        names.add(name.strip())
        clean_conditions.append({"name": name.strip(), "prefix": prefix})
    injection = value.get("injection")
    if injection is not None:
        vector = normalize_steering(dict(injection.get("vector") or {}, strength=1.0, enabled=True))
        if vector["format"] != "chatlab-steering-1":
            raise ValueError("The injection needs a chatlab-steering-1 vector with its numbers in it.")
        injection = {"vector": vector, "strength": _number(injection.get("strength"), "The injection strength"),
                     "tokens": _span(injection.get("tokens"), "the injection's tokens")}
    edit = value.get("edit") or {}
    if edit.get("mode") not in MODES:
        raise ValueError("Choose erase to reference, clamp or add.")
    seed = edit.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("The random control's seed must be a whole number from 0 to 4,294,967,295.")
    edit = {"mode": edit["mode"], "value": _number(edit.get("value", 0.0), "The edit's value"),
            "blocks": _span(edit.get("blocks"), "the edited blocks", base=0),
            "tokens": _span(edit.get("tokens"), "the edited tokens"),
            "random_control": bool(edit.get("random_control", False)), "seed": seed}
    readout = value.get("readout")
    if readout is not None:
        word = readout.get("word")
        if not isinstance(word, str) or not word.strip() or len(word) > 200:
            raise ValueError("Give the target word in at most 200 characters.")
        readout = {"word": word, "tokens": _span(readout.get("tokens"), "the readout tokens"),
                   "blocks": _span(readout.get("blocks"), "the readout blocks", base=0)}
    differences = []
    for item in value.get("differences") or []:
        name, minuend, subtrahend = item.get("name"), item.get("minuend"), item.get("subtrahend")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError("Name every difference, in at most 60 characters.")
        for side in (minuend, subtrahend):
            if side not in names:
                raise ValueError(f"The difference {name.strip()} names a condition, {side}, that is not one of them.")
        if minuend == subtrahend:
            raise ValueError(f"The difference {name.strip()} subtracts a condition from itself.")
        differences.append({"name": name.strip(), "minuend": minuend, "subtrahend": subtrahend})
    if differences and readout is None:
        raise ValueError("Differences compare the lens readout; give a target word to read.")
    return {
        "passage": passage, "conditions": clean_conditions, "injection": injection,
        "directions": files.normalize_directions(value.get("directions")),
        "edit": edit, "readout": readout, "differences": differences,
    }


def parse_differences(text):
    """``name = first - second`` per line; blank lines are skipped."""
    found = []
    for number, line in enumerate((text or "").splitlines(), start=1):
        if not line.strip():
            continue
        name, equals, rest = line.partition("=")
        first, minus, second = rest.partition(" - ")
        if not equals or not minus or not name.strip():
            raise ValueError(f"Line {number} of the differences should read like suppression = neutral - ignore.")
        found.append({"name": name.strip(), "minuend": first.strip(), "subtrahend": second.strip()})
    return found


def random_direction(seed, layer, direction):
    """A seeded unit direction at right angles to the block's own, so the control leaves its coordinate alone."""
    rng = np.random.default_rng([seed, layer])
    direction = np.asarray(direction, dtype=np.float64)
    vector = rng.standard_normal(direction.shape[0])
    vector -= (vector @ direction) * direction
    return vector / np.linalg.norm(vector)


def recovery(reference, injected, other, columns):
    """Per block, how much of the injection's shift along the direction ``other`` keeps.

    The mean over the token range of ``other - reference``, divided by the
    mean of ``injected - reference``: 1 where the edit left the shift in
    place, 0 where it took the coordinate back to the reference. Means are
    taken before dividing, since a single token the injection barely moved
    would otherwise dominate. ``None`` where the block has no direction or
    the injection did not move its coordinate.
    """
    values = []
    for ref, inj, oth in zip(reference, injected, other):
        if ref is None:
            values.append(None)
            continue
        ref, inj, oth = (np.asarray(row, dtype=np.float64)[columns] for row in (ref, inj, oth))
        shift = float(np.mean(inj - ref))
        scale = float(np.mean(np.abs(ref)) + np.mean(np.abs(inj)))
        values.append(None if abs(shift) <= 1e-6 * scale or shift == 0 else float(np.mean(oth - ref)) / shift)
    return values


def change(edited, unedited):
    """Percent change of a difference against the same difference with no edit."""
    if edited is None or unedited is None or unedited == 0:
        return None
    return 100.0 * (edited - unedited) / abs(unedited)


def differences(inputs, conditions):
    lens = {condition["name"]: condition["lens"] for condition in conditions}
    rows = []
    for item in inputs["differences"]:
        first, second = lens[item["minuend"]], lens[item["subtrahend"]]
        values = {key: (first[key] - second[key] if first.get(key) is not None and second.get(key) is not None
                        else None) for key in SETTINGS}
        rows.append(dict(item, **values, edit_change=change(values["edit"], values["no_edit"]),
                         random_change=change(values["random"], values["no_edit"])))
    return rows


class Hooks:
    """The functions block_hooks installs for one pass, built from the experiment's settings.

    Each works on the block's output for the whole sequence, batch of one,
    and writes a copy: the injection adds the scaled vector over its tokens,
    the edit moves the coordinate along the block's direction over its
    tokens, and the random control adds, token by token, as much as the
    edit added, along a random direction.
    """

    def __init__(self, injection, edit, unit, offset):
        self.injection, self.edit, self.unit, self.offset = injection, edit, unit, offset

    def columns(self, span):
        return slice(self.offset + span[0] - 1, self.offset + span[1])

    def _inject(self, hidden):
        import torch
        out = hidden.clone()
        rows = self.columns(self.injection["tokens"])
        vector = torch.tensor(self.injection["vector"]["vector"], dtype=torch.float64) * self.injection["strength"]
        out[0, rows] = hidden[0, rows] + vector.to(device=hidden.device, dtype=hidden.dtype)
        return out

    def _move(self, hidden, layer, sizes_for, along, record=None):
        import torch
        rows = self.columns(self.edit["tokens"])
        stop = min(rows.stop, hidden.shape[1])
        if stop <= rows.start:
            return hidden
        out = hidden.clone()
        x = hidden[0, rows.start:stop].float()
        direction = torch.as_tensor(self.unit[layer], dtype=torch.float32, device=hidden.device)
        size = sizes_for(x @ direction, stop - rows.start).to(device=hidden.device, dtype=torch.float32)
        if record is not None:
            record.setdefault(layer, size.double().cpu().numpy())
        step = torch.as_tensor(along, dtype=torch.float32, device=hidden.device)
        out[0, rows.start:stop] = (x + size[:, None] * step[None]).to(hidden.dtype)
        return out

    def edited(self, layer, reference, record):
        """Erase to ``reference`` (this block's reference coordinates over the edit's tokens), clamp, or add."""
        import torch
        mode, value = self.edit["mode"], self.edit["value"]

        def sizes(coordinate, count):
            if mode == ERASE:
                return torch.as_tensor(reference[:count], dtype=torch.float32, device=coordinate.device) - coordinate
            if mode == CLAMP:
                return value - coordinate
            return torch.full_like(coordinate, value)

        return lambda hidden: self._move(hidden, layer, sizes, self.unit[layer], record)

    def control(self, layer, sizes, direction):
        import torch
        return lambda hidden: self._move(
            hidden, layer, lambda _coordinate, count: torch.as_tensor(sizes[:count]), direction)

    def build(self, edits=None):
        """One function per block: the injection first, then the edit, when both land on it."""
        chain = {}
        if self.injection is not None:
            chain.setdefault(self.injection["vector"]["layer"], []).append(self._inject)
        for layer, edit in (edits or {}).items():
            chain.setdefault(layer, []).append(edit)

        def composed(functions):
            def run(hidden):
                for function in functions:
                    hidden = function(hidden)
                return hidden
            return run

        return {layer: composed(functions) for layer, functions in chain.items()}


def check(session, inputs):
    """Refuse what this load cannot run before any pass: the model, the blocks, the widths, the lens.

    Returns what the passes need: the direction matrix, the tokens of every
    condition, the target and the lens.
    """
    directions = inputs["directions"]
    files.check_model(directions, session.model_id, session.model_revision)
    blocks = session.block_count
    unit, layers = files.matrix(directions, blocks)
    session.check_projection(unit)
    edit, readout, injection = inputs["edit"], inputs["readout"], inputs["injection"]
    if edit["blocks"][1] >= blocks:
        raise ValueError(f"This model's blocks are 0–{blocks - 1}; the edit reaches block {edit['blocks'][1]}.")
    lacking = sorted(set(range(edit["blocks"][0], edit["blocks"][1] + 1)) - set(layers))
    if lacking:
        raise ValueError(f"The directions have none at block {lacking[0]}, which the edit would change.")
    if injection is not None:
        session.check_steering(dict(injection["vector"], strength=injection["strength"]))
    passage = session.encode(inputs["passage"])
    spans = [("edited", edit["tokens"])]
    if injection is not None:
        spans.append(("injection", injection["tokens"]))
    if readout is not None:
        spans.append(("readout", readout["tokens"]))
    for what, span in spans:
        if span[1] > len(passage):
            raise ValueError(f"The passage is {len(passage):,} tokens; the {what} tokens reach token {span[1]}.")
    window = session.position_limit
    limit = READ_LIMIT if window is None else min(READ_LIMIT, window)
    # A prefix opens with whatever marker the tokenizer opens a passage with,
    # even an empty one, so every condition starts as a passage would.
    prefixes = [session.example_ids(condition["prefix"]) for condition in inputs["conditions"]]
    for condition, prefix in zip(inputs["conditions"], prefixes):
        if len(prefix) + len(passage) > limit:
            raise ValueError(f"The {condition['name']} condition is {len(prefix) + len(passage):,} tokens with "
                             f"the passage, above the {limit:,} one pass may read. Shorten it.")
    lens, target = None, None
    if readout is not None:
        if readout["blocks"][1] >= blocks:
            raise ValueError(f"This model's blocks are 0–{blocks - 1}; the readout reaches block {readout['blocks'][1]}.")
        lens = session.jacobian_lens()
        if lens is None:
            raise ValueError("Import a Jacobian lens for this model in Chat's Layers view, or leave the target "
                             "word empty to skip the lens.")
        wanted = range(readout["blocks"][0], readout["blocks"][1] + 1)
        unfitted = [block for block in wanted if block not in lens["layers"]]
        if unfitted:
            raise ValueError(f"The lens {lens['name']} has no matrix for block {unfitted[0]}; it was fitted at "
                             f"blocks {lens['layers'][0]}–{lens['layers'][-1]}.")
        ids = session.encode(readout["word"])
        if not ids:
            raise ValueError("The target word has no tokens.")
        target = {"word": readout["word"], "token_ids": ids, "tokens": [session.decode([token]) for token in ids]}
    return unit, layers, passage, prefixes, target, lens


def run(session, inputs):
    """Every pass for every condition; yields a line of progress before each, returns the result.

    Holds the model through ``transformers_model()`` for the whole run, so
    no other view's work lands between the passes, and every hook is gone
    when it ends, however it ends.
    """
    inputs = normalize_inputs(inputs)
    unit, layers, passage, prefixes, target, lens = check(session, inputs)
    edit, readout, injection = inputs["edit"], inputs["readout"], inputs["injection"]
    edited_blocks = list(range(edit["blocks"][0], edit["blocks"][1] + 1))
    columns = slice(edit["tokens"][0] - 1, edit["tokens"][1])
    count = len(inputs["conditions"])
    conditions = []
    with session.transformers_model():
        for number, (condition, prefix) in enumerate(zip(inputs["conditions"], prefixes), start=1):
            ids = prefix + passage
            offset = len(prefix)
            hooks = Hooks(injection, edit, unit, offset)

            def read(installed, stage, lens_too=True):
                yield f"Condition {number} of {count}, {condition['name']}: {stage}."
                with session.block_hooks(installed):
                    rows = session.project_layers(ids, unit)[:, offset:]
                    readings = None
                    if readout is not None and lens_too:
                        positions = [offset + token - 1 for token in range(readout["tokens"][0], readout["tokens"][1] + 1)]
                        blocks = list(range(readout["blocks"][0], readout["blocks"][1] + 1))
                        readings = session.lens_log_probs(ids, [target["token_ids"]], blocks, positions)
                coordinates = [rows[layer].tolist() if layer in layers else None for layer in range(len(unit))]
                return coordinates, None if readings is None else float(readings.mean())

            # With nothing injected, the injected pass is the reference.
            reference, unedited = yield from read({}, "reference pass", lens_too=injection is None)
            injected = reference
            if injection is not None:
                injected, unedited = yield from read(hooks.build(), "injected pass")
            sizes = {}
            edits = {layer: hooks.edited(layer, np.asarray(reference[layer], dtype=np.float64)[columns], sizes)
                     for layer in edited_blocks}
            edited, with_edit = yield from read(hooks.build(edits), "edited pass")
            coordinates = {"reference": reference, "injected": injected, "edited": edited}
            lens_values = {"no_edit": unedited, "edit": with_edit, "random": None}
            recovered = {"edited": recovery(reference, injected, edited, columns), "random": []}
            if edit["random_control"]:
                controls = {layer: hooks.control(layer, sizes[layer],
                                                 random_direction(edit["seed"], layer, unit[layer]))
                            for layer in edited_blocks}
                randomized, lens_values["random"] = yield from read(hooks.build(controls), "random control")
                coordinates["random"] = randomized
                recovered["random"] = recovery(reference, injected, randomized, columns)
            conditions.append({
                "name": condition["name"], "prefix": condition["prefix"], "prefix_ids": prefix,
                "coordinates": coordinates, "recovery": recovered, "lens": lens_values,
            })
    return {
        "format": files.RESULT_FORMAT, "created": time.time(),
        "model": {"model_id": session.model_id, "model_revision": session.model_revision,
                  "precision": session.precision},
        "lens": None if lens is None else {"name": lens["name"], "n_prompts": lens["n_prompts"]},
        "inputs": inputs, "passage_ids": passage, "passage_tokens": [session.decode([token]) for token in passage],
        "edited_blocks": edited_blocks, "target": target, "conditions": conditions,
        "differences": differences(inputs, conditions),
    }
