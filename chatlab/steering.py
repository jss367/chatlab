"""Portable activation vectors and response-scoped decoder output hooks.

Layer indices are zero based. The addition is applied at every token position,
including prompt prefill, to the selected decoder block's residual output.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

FORMAT = "chatlab-steering-1"
REFERENCE_FORMAT = "chatlab-steering-reference-1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_WIDTH = 65536


class SteeringError(ValueError):
    """A steering request cannot be applied; no replacement reply is implied."""


def normalize(value: dict | None) -> dict | None:
    """Validate and copy a vector, without importing the model runtime."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("format", FORMAT) not in (FORMAT, REFERENCE_FORMAT):
        raise SteeringError(f"Expected a {FORMAT} JSON object.")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise SteeringError("The vector must name its model_id.")
    layer = value.get("layer")
    if type(layer) is not int or layer < 0:
        raise SteeringError("Layer must be a zero-based, non-negative integer.")

    def finite(number):
        try:
            return type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            return False

    strength = value.get("strength", 1.0)
    if not finite(strength) or abs(strength) > 100:
        raise SteeringError("Strength must be a finite number between -100 and 100.")
    enabled = value.get("enabled", True)
    if type(enabled) is not bool:
        raise SteeringError("Enabled must be true or false.")
    result = {
        "format": value.get("format", FORMAT),
        "model_id": model_id.strip(),
        "layer": layer,
        "strength": float(strength),
        "enabled": enabled,
    }
    if result["format"] == REFERENCE_FORMAT:
        vector_id, width = value.get("vector_id"), value.get("width")
        if not isinstance(vector_id, str) or not re.fullmatch(r"[0-9a-f]{64}", vector_id):
            raise SteeringError("Invalid steering vector identifier.")
        if type(width) is not int or not 1 <= width <= MAX_WIDTH:
            raise SteeringError("Invalid steering vector width.")
        result.update(vector_id=vector_id, width=width)
    else:
        vector = value.get("vector")
        if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_WIDTH:
            raise SteeringError(f"Vector must be a list of 1–{MAX_WIDTH:,} numbers.")
        if not all(finite(number) for number in vector):
            raise SteeringError("Every vector entry must be a finite number.")
        result["vector"] = [float(number) for number in vector]
    return result


def asset_directory() -> Path:
    # Lazy to avoid the conversation -> steering -> library import cycle.
    from chatlab.library import library_path

    target = library_path()
    return target.with_name(f"{target.stem}-vectors")


def _asset_text(value):
    return json.dumps({"model_id": value["model_id"], "vector": value["vector"]}, sort_keys=True, separators=(",", ":"))


def compact(value):
    """Store immutable vector data once; return small, serializable provenance."""
    value = normalize(value)
    if value is None or value["format"] == REFERENCE_FORMAT:
        return value
    from chatlab.trace_export import write_private_text

    text = _asset_text(value)
    vector_id = hashlib.sha256(text.encode()).hexdigest()
    directory = asset_directory()
    path = directory / f"{vector_id}.json"
    # Imports also repair a damaged existing asset. References take the fast
    # path above and never read or rewrite assets while a response streams.
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        directory.mkdir(parents=True, exist_ok=True)
        staging = directory / f".{vector_id}.{uuid4().hex}.tmp"
        try:
            write_private_text(staging, text)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
    return {key: item for key, item in value.items() if key not in ("format", "vector")} | {
        "format": REFERENCE_FORMAT, "vector_id": vector_id, "width": len(value["vector"]),
    }


def expand(value):
    """Resolve a reference only for inference, inspection, or explicit export."""
    value = normalize(value)
    if value is None or value["format"] != REFERENCE_FORMAT:
        return value
    path = asset_directory() / f"{value['vector_id']}.json"
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES * 2 + 1)
        if len(data) > MAX_FILE_BYTES * 2 or hashlib.sha256(data).hexdigest() != value["vector_id"]:
            raise SteeringError("Stored steering vector failed its integrity check.")
        asset = json.loads(data)
        if asset["model_id"] != value["model_id"] or len(asset["vector"]) != value["width"]:
            raise SteeringError("Stored steering vector does not match its reference.")
    except OSError as error:
        raise SteeringError("The stored steering vector is unavailable; import the vector or conversation again.") from error
    return normalize(dict(value, format=FORMAT, vector=asset["vector"]))


def export_assets(values):
    """Embed each vector once in a portable conversation file."""
    assets = {}
    for value in values:
        reference = compact(value)
        if reference is not None and reference["vector_id"] not in assets:
            assets[reference["vector_id"]] = json.loads(_asset_text(expand(reference)))
    return assets


def import_assets(values, assets):
    """Validate portable references and install their embedded vectors."""
    checked = set()
    for value in values:
        reference = normalize(value)
        if reference is None or reference["format"] != REFERENCE_FORMAT:
            continue
        identity = (reference["vector_id"], reference["model_id"], reference["width"])
        if identity in checked:
            continue
        asset = assets.get(reference["vector_id"]) if isinstance(assets, dict) else None
        if not isinstance(asset, dict) or asset.get("model_id") != reference["model_id"]:
            raise SteeringError("The conversation is missing an embedded steering vector.")
        full = normalize(dict(reference, format=FORMAT, vector=asset.get("vector")))
        if len(full["vector"]) != reference["width"] or hashlib.sha256(_asset_text(full).encode()).hexdigest() != reference["vector_id"]:
            raise SteeringError("An embedded steering vector does not match its reference.")
        compact(full)
        checked.add(identity)


def cleanup_unused_assets():
    """Offline maintenance: remove assets with no saved-library reference.

    ChatLab must be closed: another running session can hold references that
    have not reached the saved library yet. Invalid libraries fail closed.
    """
    from chatlab.library import library_path, parse

    try:
        payload = library_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        forks = {"branches": {}, "sampling": {}}
    else:
        forks = parse(payload)
    values = [turn.get("steering") for turns in forks["branches"].values() for turn in turns]
    values.extend(settings.get("steering") for settings in forks["sampling"].values())
    referenced = {value["vector_id"] for value in values if value is not None}
    removed = total_bytes = 0
    for path in asset_directory().glob("*.json"):
        if re.fullmatch(r"[0-9a-f]{64}", path.stem) and path.stem not in referenced and path.is_file():
            size = path.stat().st_size
            path.unlink()
            removed += 1
            total_bytes += size
    return removed, total_bytes


def read_vector(path: str) -> dict:
    # Read a bounded amount even if the file changes after opening it.
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise SteeringError("Vector files must be smaller than 4 MiB.")
    try:
        result = normalize(json.loads(data))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SteeringError("The vector file must contain valid JSON.") from error
    if result is None or result["format"] != FORMAT:
        raise SteeringError("The vector file must contain a JSON object.")
    return result


# How many examples one side of a contrast may hold. Extraction keeps every
# example's reading of every layer until the direction is known, so this is
# memory as much as patience: one 7B example is a row of 32 by 4096 numbers.
MAX_EXAMPLES = 64


def parse_examples(text: str) -> list[str]:
    """One example per line; blank lines separate nothing and are dropped.

    What is left of a line is left exactly as it was written. Indentation is
    a behaviour a reader may well be steering towards - an example of four
    spaces against an example of none is the whole contrast there - and
    trimming every line would have made those two examples the same string
    and the direction between them zero.
    """

    return [line for line in (text or "").splitlines() if line.strip()]


def _pooled_deviation(here, there) -> float:
    """The spread the two sets share, for an effect size across them.

    Either side may hold a single example, which has no spread of its own to
    contribute; a side of one contributes nothing rather than a NaN. With one
    example on each side there is no spread at all and the effect size cannot
    be formed, which is reported as a zero for the caller to refuse.
    """

    import numpy as np

    degrees = len(here) + len(there) - 2
    if degrees < 1:
        return 0.0
    total = 0.0
    for values in (here, there):
        if len(values) > 1:
            total += (len(values) - 1) * float(np.var(values, ddof=1))
    spread = math.sqrt(total / degrees)
    return spread if math.isfinite(spread) and spread > 0 else 0.0


def contrast_directions(positive, negative):
    """Difference in means, layer by layer, with what each one separates.

    Both arguments are one row per example, each row one pooled activation
    per decoder block. The direction for a block is the mean of the positive
    examples' activations minus the mean of the negative ones' - the
    difference in means, which is the vector a reader is asking for when they
    say "steer towards these and away from those".

    Beside each direction go three numbers for choosing between the layers:
    its own length, the typical length of an activation at that layer, and
    how far apart the two sets sit along it. The last is a standardized
    effect size - the gap between the two sets' projections divided by the
    spread they share - so it can be read across layers whose activations
    have very different scales, which the raw length cannot. It is measured
    on the examples the direction was taken from, so it says how cleanly this
    direction splits *these* examples and not how it will generalize.
    ``None`` where there is no spread to divide by: one example on each side.
    """

    import numpy as np

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not len(positive) or not len(negative):
        raise SteeringError("Give at least one example on each side.")
    if positive.ndim != 3 or negative.ndim != 3 or positive.shape[1:] != negative.shape[1:]:
        raise SteeringError("Every example must be read at the same layers and width.")
    if not positive.shape[1] or not positive.shape[2]:
        raise SteeringError("The model returned no layer activations to compare.")

    directions = positive.mean(axis=0) - negative.mean(axis=0)
    stats = []
    for layer, direction in enumerate(directions):
        norm = float(np.linalg.norm(direction))
        lengths = np.linalg.norm(
            np.concatenate([positive[:, layer], negative[:, layer]]), axis=1
        )
        separation = None
        if norm > 0:
            unit = direction / norm
            here, there = positive[:, layer] @ unit, negative[:, layer] @ unit
            spread = _pooled_deviation(here, there)
            if spread:
                separation = float((here.mean() - there.mean()) / spread)
        stats.append({
            "layer": layer,
            "norm": norm,
            "activation_norm": float(lengths.mean()),
            "separation": separation,
        })
    return [[float(value) for value in row] for row in directions], stats


def best_layer(stats) -> int:
    """Which layer's direction separated the examples most cleanly.

    The effect size decides it where there is one. With a single example on
    each side there is none anywhere, and the longest direction is the only
    thing left to go on - a weak answer, which is why the interface says how
    it was reached.
    """

    if not stats:
        return 0
    ranked = [item for item in stats if item["separation"] is not None]
    if ranked:
        return int(max(ranked, key=lambda item: abs(item["separation"]))["layer"])
    return int(max(stats, key=lambda item: item["norm"])["layer"])


def vector_from(model_id: str, layer: int, values, *, strength=1.0, enabled=True) -> dict:
    """An extracted direction in the shape an imported file would have."""

    return normalize({
        "format": FORMAT,
        "model_id": model_id,
        "layer": int(layer),
        "vector": [float(value) for value in values],
        "strength": float(strength),
        "enabled": bool(enabled),
    })


def from_controls(value, enabled=None, strength=None, layer=None):
    """Snapshot visible controls without waiting for their persistence event.

    Omitted controls keep the specification supplied by direct Python callers.
    The UI supplies all three with each generation or save request.
    """
    if value is None:
        if enabled:
            raise SteeringError("Import a vector before enabling steering.")
        return None
    value = dict(value)
    if enabled is not None:
        value["enabled"] = enabled
    if strength is not None:
        value["strength"] = strength
    if layer is not None:
        if isinstance(layer, bool) or int(layer) != layer:
            raise SteeringError("Layer must be a zero-based integer.")
        value["layer"] = int(layer)
    return normalize(value)


def active(value: dict | None) -> bool:
    return bool(value and value.get("enabled", True) and value.get("strength", 1) != 0)


def decoder_layers(model):
    """Recognize explicit decoder stacks; refuse ambiguous architectures."""
    import torch

    for path in (
        "model.layers", "transformer.h", "model.decoder.layers",
        "gpt_neox.layers", "transformer.blocks",
        "model.language_model.layers", "language_model.model.layers",
    ):
        candidate = model
        for name in path.split("."):
            candidate = getattr(candidate, name, None)
        if isinstance(candidate, torch.nn.ModuleList) and len(candidate):
            return candidate
    raise SteeringError("Steering is not supported for this model's decoder architecture.")


def validate_model(model, model_id: str | None, value: dict):
    if model is None:
        raise SteeringError("Load the vector's model before enabling steering.")
    if value["model_id"] != model_id:
        raise SteeringError(f"This vector requires {value['model_id']}; load that model or disable steering.")
    layers = decoder_layers(model)
    if value["layer"] >= len(layers):
        raise SteeringError(f"This model has {len(layers)} layers; choose 0–{len(layers) - 1}.")
    config = getattr(model, "config", None)
    if callable(getattr(config, "get_text_config", None)):
        config = config.get_text_config()
    width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if width is None:
        width = model.get_input_embeddings().weight.shape[-1]
    if len(value["vector"]) != width:
        raise SteeringError(f"This model requires a vector with {width} entries, not {len(value['vector'])}.")
    return layers[value["layer"]]


@contextmanager
def applied(model, model_id: str | None, value: dict | None):
    """Install only while the caller holds the model lock; always remove."""
    value = normalize(value)
    if not active(value):
        yield
        return
    value = expand(value)
    import torch

    block = validate_model(model, model_id, value)
    cached = None

    def add_vector(_module, _inputs, output):
        nonlocal cached
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != len(value["vector"]):
            raise SteeringError("The selected layer does not return a compatible residual tensor.")
        if cached is None or cached.device != hidden.device or cached.dtype != hidden.dtype:
            # Scale before casting so representable small products stay usable.
            cached = torch.tensor(value["vector"], dtype=torch.float64)
            cached = (cached * value["strength"]).to(device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(cached).all().item():
                raise SteeringError("The scaled vector overflows this model's activation precision.")
        changed = hidden + cached
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    # Transformers may already have hooks recording hidden states for the
    # inspector. They must see the modified output, including on later clicks.
    handle = block.register_forward_hook(add_vector, prepend=True)
    try:
        yield
    finally:
        handle.remove()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ChatLab steering-vector maintenance")
    parser.add_argument(
        "--cleanup-unused", action="store_true",
        help="with ChatLab closed, remove vector files not referenced by the saved conversation library",
    )
    args = parser.parse_args()
    if not args.cleanup_unused:
        parser.print_help()
        return
    try:
        removed, total_bytes = cleanup_unused_assets()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")
    print(f"Removed {removed} unused vector file(s), freeing {total_bytes:,} bytes.")


if __name__ == "__main__":
    main()
