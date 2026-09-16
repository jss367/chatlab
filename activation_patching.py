"""Independent post-block residual transplants, measured at one next token.

Every cell starts from the recipient's original prefix. No sampled continuation,
cache reuse, attention-head intervention, or sequential combination is implied.
"""

from __future__ import annotations

import contextlib
import inspect
import math

from steering import active


SUPPORTED_MODELS = {"llama", "qwen2", "olmo3"}
MAX_PREFIX = 2048
MAX_POSITIONS = 32
MAX_CELLS = 2048


def integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a whole number.")
    try:
        converted = int(value)
        if converted != float(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a whole number.") from None
    return converted


def experiment(donor, recipient, target_index, donor_count, width):
    """Build exact prefixes from Compare recordings, never re-tokenize text."""
    for run in (donor, recipient):
        if not run or "context_ids" not in run or not run.get("run_id"):
            raise ValueError("Fill both Compare slots again to record their exact input tokens.")
        if not run.get("load_id") or not run.get("model_id"):
            raise ValueError("Both runs must record the model load that produced them.")
        if active((run.get("settings") or {}).get("steering")):
            raise ValueError("Turn steering off and fill both slots again before patching.")
    if donor["load_id"] != recipient["load_id"] or donor["model_id"] != recipient["model_id"]:
        raise ValueError("Both runs must come from the same model load. Fill both slots again.")
    target_index = integer(target_index, "Answer token")
    donor_count = integer(donor_count, "Source output tokens")
    width = integer(width, "Token pairs")
    if not 0 <= target_index < len(recipient["metrics"]):
        raise ValueError("Select an answer token from the recipient run.")
    if not 0 <= donor_count <= len(donor["metrics"]):
        raise ValueError("Source output tokens must be between zero and the source run's length.")
    if not 1 <= width <= MAX_POSITIONS:
        raise ValueError(f"Choose between 1 and {MAX_POSITIONS} token pairs.")

    def prefix(run, count):
        return [integer(v, "Token ID") for v in run["context_ids"]] + [
            integer(m["token_id"], "Token ID") for m in run["metrics"][:count]
        ]

    source = prefix(donor, donor_count)
    destination = prefix(recipient, target_index)
    if not source or not destination:
        raise ValueError("Both prefixes need at least one token before the prediction.")
    if max(len(source), len(destination)) > MAX_PREFIX:
        raise ValueError(f"Patching supports prefixes up to {MAX_PREFIX:,} tokens; choose earlier tokens or shorter runs.")
    if width > min(len(source), len(destination)):
        raise ValueError("Choose fewer token pairs; the requested window is longer than a prefix.")
    return {
        "format": "chatlab-activation-patching-1",
        "site": "decoder_block_output", "intervention": "residual_replacement",
        "alignment": "prefix_end", "independent_cells": True,
        "model_id": donor["model_id"], "load_id": donor["load_id"],
        "precision": recipient.get("precision"),
        "donor_run_id": donor["run_id"], "recipient_run_id": recipient["run_id"],
        "donor_slot": donor.get("slot"), "recipient_slot": recipient.get("slot"),
        "donor_output_count": donor_count, "target_index": target_index,
        "target_id": integer(recipient["metrics"][target_index]["token_id"], "Answer token ID"),
        "donor_ids": source, "recipient_ids": destination,
        "pairs": [
            {"donor_position": len(source) - width + i,
             "recipient_position": len(destination) - width + i}
            for i in range(width)
        ],
    }


def model_layers(model):
    kind = getattr(getattr(model, "config", None), "model_type", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if kind not in SUPPORTED_MODELS or layers is None or not len(layers):
        raise ValueError("Residual patching currently supports Transformers Llama, Qwen2 and OLMo 3 models.")
    if model.training:
        raise ValueError("Patching requires the model in evaluation mode.")
    return layers


def measure(model, plan):
    """Yield a baseline and individual cells; hooks never survive a yield.

    Only selected source vectors are kept, on CPU. Each recipient pass has a
    fresh full prefix and no KV cache. Inference mode is scoped to each pass,
    since Gradio can resume a generator on a different worker thread.
    """
    import torch

    layers = model_layers(model)
    source, destination = plan["donor_ids"], plan["recipient_ids"]
    pairs, target = plan["pairs"], plan["target_id"]
    if not pairs or len(pairs) > MAX_POSITIONS or len(pairs) * len(layers) > MAX_CELLS:
        raise ValueError("Too many interventions; choose fewer token pairs.")
    if not source or not destination or max(len(source), len(destination)) > MAX_PREFIX:
        raise ValueError("Invalid or overly long patching prefix.")
    vocab = model.get_input_embeddings().weight.shape[0]
    output_vocab = model.get_output_embeddings().weight.shape[0]
    if any(not 0 <= token < vocab for token in source + destination) or not 0 <= target < output_vocab:
        raise ValueError("The recorded token IDs do not fit the loaded model.")
    for pair in pairs:
        if not (0 <= pair["donor_position"] < len(source)
                and 0 <= pair["recipient_position"] < len(destination)):
            raise ValueError("A patch position is outside its prefix.")
    limit = getattr(model.config, "max_position_embeddings", MAX_PREFIX)
    if max(len(source), len(destination)) > limit:
        raise ValueError("A prefix exceeds this model's context window.")
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise ValueError("Patching does not support models with offloaded embedding weights.")
    last_only = "logits_to_keep" in inspect.signature(model.forward).parameters

    def forward(ids):
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        use_cache=False, **({"logits_to_keep": 1} if last_only else {}))
            logits = out.logits[0, -1].float()
            log_probability = float(torch.log_softmax(logits, dim=-1)[target].item())
            if not math.isfinite(log_probability):
                raise ValueError("The model returned a non-finite answer probability.")
            return {"probability": math.exp(log_probability), "log_probability": log_probability}

    def hidden(output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("This decoder block does not expose a compatible residual stream.")
        return value

    vectors = {}
    positions = [pair["donor_position"] for pair in pairs]

    def capture(layer):
        def hook(_module, _args, output):
            vectors[layer] = hidden(output)[0, positions].detach().to("cpu", copy=True)
        return hook

    with contextlib.ExitStack() as scope:
        for layer, block in enumerate(layers):
            scope.callback(block.register_forward_hook(capture(layer)).remove)
        donor_baseline = forward(source)
    if len(vectors) != len(layers):
        raise ValueError("Not every decoder block was executed during source capture.")

    baseline = forward(destination)
    yield {"baseline": baseline, "donor_baseline": donor_baseline, "layer_count": len(layers)}
    for layer, block in enumerate(layers):
        for column, pair in enumerate(pairs):
            calls = 0

            def transplant(_module, _args, output):
                nonlocal calls
                calls += 1
                value = hidden(output)
                replacement = vectors[layer][column].to(device=value.device, dtype=value.dtype)
                if replacement.shape != value[0, pair["recipient_position"]].shape:
                    raise ValueError("Source and recipient activation widths differ.")
                changed = value.clone()
                changed[0, pair["recipient_position"]] = replacement
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = block.register_forward_hook(transplant)
            try:
                patched = forward(destination)
            finally:
                handle.remove()
            if calls != 1:
                raise ValueError("The selected block was not executed exactly once.")
            yield {
                "layer": layer, "column": column, **patched,
                "delta_probability": patched["probability"] - baseline["probability"],
                "delta_log_probability": patched["log_probability"] - baseline["log_probability"],
            }
