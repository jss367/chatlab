"""The Settings page: the sampling summary and the settings saved between sessions."""

from __future__ import annotations

import gradio as gr

import settings


def sampling_label(temperature, top_p, top_k, max_new_tokens) -> str:
    """The sampling accordion's own summary of what it holds."""

    filtering = f"top-p {float(top_p):g}"
    if int(top_k or 0):
        filtering = f"{filtering} · top-k {int(top_k)}"
    return (
        f"Sampling · temperature {float(temperature):g} · {filtering} · "
        f"up to {int(max_new_tokens):,} new tokens"
    )


def update_sampling_label(temperature, top_p, top_k, max_new_tokens):
    return gr.update(label=sampling_label(temperature, top_p, top_k, max_new_tokens))


# The settings that outlive the session, in the order the controls wired to
# remember_settings publish them. The Hugging Face token is not among them on
# purpose: see the settings module.
PERSISTED_SETTING_NAMES = (
    "system_prompt",
    "keep_reasoning",
    "assistant_prefill",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "seed",
    "randomize_seed",
    "analyze_prompt",
    "color_scale",
    "enter_sends",
    "model_id",
    "weight_precision",
)


def remember_settings(*values, seed_committed: bool = False) -> None:
    """Save every setting whenever one of them changes.

    There is no save button, so each control reports the whole set and the
    file is rewritten. A change that changes nothing is not written, which is
    what keeps a slider drag from writing once per pixel.

    Two of the controls hold something that is not always the reader's
    choice, and both are filtered rather than saved as they are read.
    ``OLMO_MODEL_ID`` puts a model in the model box for one run, and a
    finished response leaves the seed it used in the seed box. Neither must
    be written down just because something else changed.
    ``seed_committed`` is the seed box's own blur or submit saying the number
    there is a choice after all.
    """

    chosen = dict(zip(PERSISTED_SETTING_NAMES, values, strict=True))
    chosen["model_id"] = settings.model_id_to_save(chosen["model_id"])
    if not seed_committed:
        chosen["seed"] = settings.seed_to_save(
            chosen["seed"], chosen["randomize_seed"]
        )
    settings.update(**chosen)


def remember_committed_seed(*values) -> None:
    """Save every setting, the seed box included, when it is done being edited."""

    remember_settings(*values, seed_committed=True)


def restore_settings():
    """Put the saved settings back into every control, and re-read the file.

    A browser reload rebuilds the page from the values the interface was built
    with, which are the ones the file held when the app started. Without this
    a reload would show stale values, and the next change would write them
    back over whatever the file has learned since — including an edit made to
    the file by hand, which this is also how one takes effect.
    """

    saved = settings.load()
    values = saved.to_mapping() | {"model_id": settings.model_id_at_startup(saved)}
    updates = [
        # The response-length ceiling is the context limit, so it comes back
        # with the length itself.
        gr.update(value=saved.max_new_tokens, maximum=saved.prefill_token_limit)
        if name == "max_new_tokens"
        else gr.update(value=values[name])
        for name in PERSISTED_SETTING_NAMES
    ]
    return (*updates, gr.update(value=saved.prefill_token_limit))


def remember_prefill_limit(limit, max_new_tokens):
    """Save the context limit, and pull the response length under it.

    The response-length control tops out at the context limit, so lowering
    the limit lowers the ceiling and, if it was above the new one, the length
    itself. The limit is echoed back because it is clamped to a range the
    number box cannot express on its own.
    """

    saved = settings.update(prefill_token_limit=limit, max_new_tokens=max_new_tokens)
    return (
        gr.update(value=saved.prefill_token_limit),
        gr.update(maximum=saved.prefill_token_limit, value=saved.max_new_tokens),
    )
