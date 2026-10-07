"""Direction files in, result files out. Nothing here touches the model.

A direction file holds one direction per decoder block, written by hand or
by a script as ``chatlab-directions-1``; a ``chatlab-probe-1`` file from the
Probes page serves as well, each layer's weights taken as that block's
direction. Every direction is scaled to unit length on reading, so a
coordinate along it is in the residual stream's own units.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PROBE_FORMAT = "chatlab-probe-1"
LATEST_CREATED = 32503680000.0  # The start of year 3000, in Unix seconds.

DIRECTIONS_FORMAT = "chatlab-directions-1"
RESULT_FORMAT = "chatlab-direction-edits-1"
PRECISIONS = ("full", "8-bit", "4-bit")
# A 27B model's 64 blocks of 5,120 numbers are about 7 MB written out.
MAX_DIRECTIONS_BYTES = 64 * 1024 * 1024
# A result holds the directions and the vector as well as its readings.
MAX_RESULT_BYTES = 256 * 1024 * 1024
MAX_BLOCKS = 256
MAX_WIDTH = 65536


def _read_json(path, limit, what):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{what} files must be smaller than {limit // 1024 // 1024} MiB.")
    try:
        return json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The {what.lower()} file must contain valid JSON.") from error


def _text(value, field, limit=200, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"The {field} must be text of at most {limit} characters.")
    return value.strip()


def _unit(vector, layer):
    """A direction scaled to length one, refused when it has no length to scale."""
    if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_WIDTH:
        raise ValueError(f"The direction at block {layer} must be a list of 1–{MAX_WIDTH:,} numbers.")
    if not all(type(item) in (int, float) and math.isfinite(item) for item in vector):
        raise ValueError(f"Every entry of the direction at block {layer} must be a finite number.")
    values = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(values))
    if not math.isfinite(length) or length == 0:
        raise ValueError(f"The direction at block {layer} has no length.")
    return (values / length).tolist()


def normalize_directions(value):
    """Check a direction set read from anywhere and return a clean copy with unit directions.

    Accepts ``chatlab-directions-1`` and ``chatlab-probe-1``. Raises ``ValueError``.
    """
    if isinstance(value, dict) and value.get("format") == PROBE_FORMAT:
        from chatlab.extensions.probes import probe as probes

        probe = probes.normalize(value)
        return {
            "format": DIRECTIONS_FORMAT, "name": probe["name"], "model_id": probe["model_id"],
            "model_revision": probe["model_revision"], "precision": probe["precision"], "source": PROBE_FORMAT,
            "directions": [{"layer": item["layer"], "vector": _unit(item["weights"], item["layer"])}
                           for item in probe["layers"]],
        }
    if not isinstance(value, dict) or value.get("format") != DIRECTIONS_FORMAT:
        raise ValueError(f"Expected a {DIRECTIONS_FORMAT} or {PROBE_FORMAT} JSON object.")
    precision = value.get("precision")
    if precision is not None and precision not in PRECISIONS:
        raise ValueError("The directions' precision must be full, 8-bit, 4-bit or null.")
    items = value.get("directions")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_BLOCKS:
        raise ValueError(f"Give between 1 and {MAX_BLOCKS} directions, one per block.")
    clean, width = {}, None
    for item in items:
        layer = item.get("layer") if isinstance(item, dict) else None
        if type(layer) is not int or not 0 <= layer < MAX_BLOCKS:
            raise ValueError("Every direction needs a block index as its layer, counted from 0.")
        if layer in clean:
            raise ValueError(f"Block {layer} has more than one direction.")
        vector = _unit(item.get("vector"), layer)
        width = len(vector) if width is None else width
        if len(vector) != width:
            raise ValueError("Every direction must be the same width.")
        clean[layer] = vector
    return {
        "format": DIRECTIONS_FORMAT, "name": _text(value.get("name"), "directions' name"),
        "model_id": _text(value.get("model_id"), "directions' model_id", 300),
        "model_revision": _text(value.get("model_revision"), "directions' model revision", optional=True),
        "precision": precision, "source": DIRECTIONS_FORMAT,
        "directions": [{"layer": layer, "vector": clean[layer]} for layer in sorted(clean)],
    }


def read_directions(path):
    return normalize_directions(_read_json(path, MAX_DIRECTIONS_BYTES, "Direction"))


def check_model(directions, model_id, revision, what="These directions"):
    """Refuse directions made for another model, or for another revision when both are known."""
    if directions["model_id"] != model_id:
        raise ValueError(f"{what} were made for {directions['model_id']}; load that model to use them. "
                         f"{model_id} is loaded.")
    made = directions["model_revision"]
    if made and revision and made != revision:
        raise ValueError(f"{what} were made for revision {made[:12]} of {model_id}, and revision "
                         f"{revision[:12]} is loaded. They belong to the other weights.")


def matrix(directions, blocks):
    """One row per decoder block, zero where the set has no direction, and the blocks it has."""
    layers = [item["layer"] for item in directions["directions"]]
    if layers[-1] >= blocks:
        raise ValueError(f"These directions reach block {layers[-1]}; this model's blocks are 0–{blocks - 1}.")
    rows = np.zeros((blocks, len(directions["directions"][0]["vector"])), dtype=np.float32)
    for item in directions["directions"]:
        rows[item["layer"]] = item["vector"]
    return rows, layers


def dumps(result):
    return json.dumps(result, separators=(",", ":"), allow_nan=False)


def _finite(value, *, optional=False):
    if optional and value is None:
        return True
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _token_list(ids, tokens, what):
    if (not isinstance(ids, list) or not ids or not all(type(token) is int and token >= 0 for token in ids)
            or not isinstance(tokens, list) or len(tokens) != len(ids)
            or not all(isinstance(token, str) for token in tokens)):
        raise ValueError(f"The result needs matching {what} token IDs and text.")


def normalize_result(value):
    """Validate every field and shape the saved-result renderer reads before accepting an upload."""
    from . import experiment

    if not isinstance(value, dict) or value.get("format") != RESULT_FORMAT:
        raise ValueError(f"Expected a {RESULT_FORMAT} JSON object.")
    created = value.get("created")
    if not _finite(created) or not 0 <= created <= LATEST_CREATED:
        raise ValueError("The result's creation time is not a date between 1970 and 3000.")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or not {"injection", "edit", "readout"} <= inputs.keys():
        raise ValueError("The result needs its injection, edit and readout settings.")
    try:
        checked = experiment.normalize_inputs(inputs)
    except (AttributeError, KeyError, TypeError, OverflowError) as error:
        raise ValueError("The result's input settings are malformed.") from error
    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("The result needs its model metadata.")
    _text(model.get("model_id"), "result's model_id", 300)
    _text(model.get("model_revision"), "result's model revision", optional=True)
    if "precision" not in model or model["precision"] not in (*PRECISIONS, None):
        raise ValueError("The result needs its model precision.")
    passage, tokens = value.get("passage_ids"), value.get("passage_tokens")
    _token_list(passage, tokens, "passage")
    if len(passage) > experiment.READ_LIMIT:
        raise ValueError("The result's passage exceeds the reading limit.")
    edit, readout = checked["edit"], checked["readout"]
    if not isinstance(inputs["edit"], dict) or type(inputs["edit"].get("random_control")) is not bool:
        raise ValueError("The result needs a boolean random-control setting.")
    if edit["blocks"][1] >= MAX_BLOCKS:
        raise ValueError("The result's edit blocks exceed the block limit.")
    random = edit["random_control"]
    spans = [edit["tokens"]]
    if checked["injection"] is not None:
        spans.append(checked["injection"]["tokens"])
    if readout is not None:
        spans.append(readout["tokens"])
    if any(span[1] > len(passage) for span in spans):
        raise ValueError("The result's token settings extend past its passage.")
    edited = value.get("edited_blocks")
    if (not isinstance(edited, list) or not edited or not all(type(block) is int for block in edited)
            or edited != list(range(edit["blocks"][0], edit["blocks"][1] + 1))):
        raise ValueError("The result needs its edited blocks matching the edit settings.")
    target, lens = value.get("target"), value.get("lens")
    if not {"target", "lens"} <= value.keys():
        raise ValueError("The result needs its target and lens fields, null when no readout was requested.")
    if readout is None:
        if target is not None or lens is not None:
            raise ValueError("A result without a readout cannot have a target or lens.")
    else:
        if not isinstance(target, dict) or target.get("word") != readout["word"]:
            raise ValueError("The result needs its readout target.")
        _token_list(target.get("token_ids"), target.get("tokens"), "target")
        if not isinstance(lens, dict):
            raise ValueError("The result needs its lens metadata.")
        _text(lens.get("name"), "lens name", 255)
        if type(lens.get("n_prompts")) is not int or lens["n_prompts"] <= 0:
            raise ValueError("The result needs a positive lens prompt count.")
    conditions = value.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(checked["conditions"]):
        raise ValueError("The result needs one reading per input condition.")
    layers = {item["layer"] for item in checked["directions"]["directions"]}
    blocks = None
    pass_names = {"reference", "injected", "edited"} | ({"random"} if random else set())
    for condition, requested in zip(conditions, checked["conditions"]):
        if not isinstance(condition, dict) or condition.get("name") != requested["name"]:
            raise ValueError("The result's condition names must match its inputs.")
        passes = condition.get("coordinates")
        if not isinstance(passes, dict) or set(passes) != pass_names:
            raise ValueError("Every condition needs the coordinate passes requested by its settings.")
        for rows in passes.values():
            if (not isinstance(rows, list) or not 1 <= len(rows) <= MAX_BLOCKS
                    or blocks is not None and len(rows) != blocks):
                raise ValueError("Every pass needs one row per block.")
            blocks = len(rows)
            for block, row in enumerate(rows):
                if block not in layers:
                    if row is not None:
                        raise ValueError("Blocks without directions must have null coordinate rows.")
                elif (not isinstance(row, list) or len(row) != len(passage)
                      or not all(_finite(item) and abs(item) <= float(np.finfo(np.float32).max) for item in row)):
                    raise ValueError("Every coordinate row must hold one finite number per passage token.")
        recovered = condition.get("recovery")
        if not isinstance(recovered, dict):
            raise ValueError("Every condition needs its recovery readings.")
        for key, length in (("edited", blocks), ("random", blocks if random else 0)):
            row = recovered.get(key)
            if (not isinstance(row, list) or len(row) != length
                    or not all(_finite(item, optional=True) for item in row)):
                raise ValueError("Every recovery reading must match its blocks and random-control setting.")
        readings = condition.get("lens")
        if not isinstance(readings, dict) or not set(experiment.SETTINGS) <= readings.keys():
            raise ValueError("Every condition needs all its lens-value fields.")
        for key in experiment.SETTINGS:
            expected = readout is not None and (key != "random" or random)
            if (expected and not _finite(readings[key])) or (not expected and readings[key] is not None):
                raise ValueError("The lens values must be finite when read, and null otherwise.")
    if max(layers) >= blocks or edited[-1] >= blocks or readout is not None and readout["blocks"][1] >= blocks:
        raise ValueError("The result's directions, edits and readout must fit its coordinate blocks.")
    differences = value.get("differences")
    if not isinstance(differences, list) or len(differences) != len(checked["differences"]):
        raise ValueError("The result needs the differences requested by its inputs.")
    for item, requested in zip(differences, checked["differences"]):
        if not isinstance(item, dict) or any(item.get(key) != requested[key] for key in requested):
            raise ValueError("The result's difference names and conditions must match its inputs.")
        for key in (*experiment.SETTINGS, "edit_change", "random_change"):
            if key not in item or not _finite(item[key], optional=True):
                raise ValueError("Every difference needs finite or null readings and percent changes.")
    return value


def read_result(path):
    return normalize_result(_read_json(path, MAX_RESULT_BYTES, "Result"))
