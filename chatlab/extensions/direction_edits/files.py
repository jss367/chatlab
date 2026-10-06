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

from chatlab.extensions.probes import probe as probes

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
    if isinstance(value, dict) and value.get("format") == probes.FORMAT:
        probe = probes.normalize(value)
        return {
            "format": DIRECTIONS_FORMAT, "name": probe["name"], "model_id": probe["model_id"],
            "model_revision": probe["model_revision"], "precision": probe["precision"], "source": probes.FORMAT,
            "directions": [{"layer": item["layer"], "vector": _unit(item["weights"], item["layer"])}
                           for item in probe["layers"]],
        }
    if not isinstance(value, dict) or value.get("format") != DIRECTIONS_FORMAT:
        raise ValueError(f"Expected a {DIRECTIONS_FORMAT} or {probes.FORMAT} JSON object.")
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


def normalize_result(value):
    """Check a saved result before it is shown. It is only ever read back, never rerun."""
    if not isinstance(value, dict) or value.get("format") != RESULT_FORMAT:
        raise ValueError(f"Expected a {RESULT_FORMAT} JSON object.")
    created = value.get("created")
    if type(created) not in (int, float) or not 0 <= created <= probes.LATEST_CREATED:
        raise ValueError("The result's creation time is not a date between 1970 and 3000.")
    inputs, conditions = value.get("inputs"), value.get("conditions")
    passage = value.get("passage_ids")
    tokens = value.get("passage_tokens")
    if (not isinstance(inputs, dict) or not isinstance(conditions, list) or not conditions
            or not isinstance(passage, list) or not isinstance(tokens, list) or len(passage) != len(tokens)):
        raise ValueError("The result needs its inputs, conditions and passage tokens.")
    normalize_directions(inputs.get("directions"))
    blocks = None
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("name"), str):
            raise ValueError("Every condition in the result needs a name.")
        passes = condition.get("coordinates")
        if not isinstance(passes, dict) or not {"reference", "injected", "edited"} <= set(passes):
            raise ValueError("Every condition needs its reference, injected and edited coordinates.")
        for rows in passes.values():
            if not isinstance(rows, list) or blocks is not None and len(rows) != blocks:
                raise ValueError("Every pass needs one row per block.")
            blocks = len(rows)
            for row in rows:
                if row is not None and (not isinstance(row, list) or len(row) != len(passage) or not all(
                        type(item) in (int, float) for item in row)):
                    raise ValueError("Every coordinate row must hold one number per passage token.")
        recovery = condition.get("recovery")
        if not isinstance(recovery, dict) or not all(
                isinstance(recovery.get(key, []), list) and len(recovery.get(key, [])) in (0, blocks)
                for key in ("edited", "random")):
            raise ValueError("Every condition needs one recovery value per block.")
    if not isinstance(value.get("differences", []), list):
        raise ValueError("The result's differences must be a list.")
    return value


def read_result(path):
    return normalize_result(_read_json(path, MAX_RESULT_BYTES, "Result"))
