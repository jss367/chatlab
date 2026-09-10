"""Portable activation vectors and response-scoped decoder output hooks.

Layer indices are zero based. The addition is applied at every token position,
including prompt prefill, to the selected decoder block's residual output.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

FORMAT = "chatlab-steering-1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_WIDTH = 65536


def normalize(value: dict | None) -> dict | None:
    """Validate and copy a vector, without importing the model runtime."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Expected a {FORMAT} JSON object.")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("The vector must name its model_id.")
    layer = value.get("layer")
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a zero-based, non-negative integer.")
    vector = value.get("vector")
    if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_WIDTH:
        raise ValueError(f"Vector must be a list of 1–{MAX_WIDTH:,} numbers.")

    def finite(number):
        try:
            return type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            return False

    if not all(finite(number) for number in vector):
        raise ValueError("Every vector entry must be a finite number.")
    strength = value.get("strength", 1.0)
    if not finite(strength) or abs(strength) > 100:
        raise ValueError("Strength must be a finite number between -100 and 100.")
    enabled = value.get("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("Enabled must be true or false.")
    return {
        "format": FORMAT,
        "model_id": model_id.strip(),
        "layer": layer,
        "vector": [float(number) for number in vector],
        "strength": float(strength),
        "enabled": enabled,
    }


def read_vector(path: str) -> dict:
    # Read a bounded amount even if the file changes after opening it.
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Vector files must be smaller than 4 MiB.")
    try:
        result = normalize(json.loads(data))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("The vector file must contain valid JSON.") from error
    if result is None:
        raise ValueError("The vector file must contain a JSON object.")
    return result


def from_controls(value, enabled=None, strength=None, layer=None):
    """Snapshot visible controls without waiting for their persistence event.

    Omitted controls keep the specification supplied by direct Python callers.
    The UI supplies all three with each generation or save request.
    """
    if value is None:
        if enabled:
            raise ValueError("Import a vector before enabling steering.")
        return None
    value = dict(value)
    if enabled is not None:
        value["enabled"] = enabled
    if strength is not None:
        value["strength"] = strength
    if layer is not None:
        if isinstance(layer, bool) or int(layer) != layer:
            raise ValueError("Layer must be a zero-based integer.")
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
    raise ValueError("Steering is not supported for this model's decoder architecture.")


def validate_model(model, model_id: str | None, value: dict):
    if model is None:
        raise ValueError("Load the vector's model before enabling steering.")
    if value["model_id"] != model_id:
        raise ValueError(f"This vector requires {value['model_id']}; load that model or disable steering.")
    layers = decoder_layers(model)
    if value["layer"] >= len(layers):
        raise ValueError(f"This model has {len(layers)} layers; choose 0–{len(layers) - 1}.")
    config = getattr(model, "config", None)
    if callable(getattr(config, "get_text_config", None)):
        config = config.get_text_config()
    width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if width is None:
        width = model.get_input_embeddings().weight.shape[-1]
    if len(value["vector"]) != width:
        raise ValueError(f"This model requires a vector with {width} entries, not {len(value['vector'])}.")
    return layers[value["layer"]]


@contextmanager
def applied(model, model_id: str | None, value: dict | None):
    """Install only while the caller holds the model lock; always remove."""
    value = normalize(value)
    if not active(value):
        yield
        return
    import torch

    block = validate_model(model, model_id, value)
    cached = None

    def add_vector(_module, _inputs, output):
        nonlocal cached
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != len(value["vector"]):
            raise ValueError("The selected layer does not return a compatible residual tensor.")
        if cached is None or cached.device != hidden.device or cached.dtype != hidden.dtype:
            # Scale before casting so representable small products stay usable.
            cached = torch.tensor(value["vector"], dtype=torch.float64)
            cached = (cached * value["strength"]).to(device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(cached).all().item():
                raise ValueError("The scaled vector overflows this model's activation precision.")
        changed = hidden + cached
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    # Transformers may already have hooks recording hidden states for the
    # inspector. They must see the modified output, including on later clicks.
    handle = block.register_forward_hook(add_vector, prepend=True)
    try:
        yield
    finally:
        handle.remove()
