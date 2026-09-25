"""Recorded runs in the shape the Compare page reads them, for the tests."""

from chatlab import compare


def metric(position, token_id, surprise, *, top="a", top_id=1, scored=True, entropy=1.0, rank=1):
    """One token's measurements, in the shape the runtime publishes them."""

    return {
        "position": position,
        "token_id": token_id,
        "text": f"t{token_id}",
        "display_text": f"t{token_id}",
        "category": "Top choice",
        "raw_rank": rank,
        "raw_probability": 0.5,
        "sampling_probability": 0.5,
        "surprise_bits": surprise,
        "probability_mass_above": 0.0,
        "entropy_bits": entropy,
        "top1_margin": 0.1,
        "sampling_shift_bits": 0.0,
        "top_candidates": [{"token_id": top_id, "text": top, "probability": 0.5}],
        "scored": scored,
        "segment": "response",
        "unscored_reason": "",
    }


def run(metrics, *, kind=compare.REPLY, model_id="fake/model", **settings):
    return {
        "kind": kind,
        "model_id": model_id,
        "load_id": f"{model_id}#1",
        "device_name": "CPU",
        "precision": "float32",
        "prompt": "hello",
        "text": "hello",
        "metrics": metrics,
        "settings": {
            "system_prompt": "",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "skip_top_below": 0.0,
            "max_new_tokens": 8,
            "seed": 1,
            "assistant_prefill": "",
            "thinking_mode": "default",
            "steering": None,
        } | settings,
        "seconds": 0.1,
    }
