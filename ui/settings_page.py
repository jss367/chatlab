"""The Settings page: the hardware panel, the sampling summary, and the saved settings."""

from __future__ import annotations

import gradio as gr

import library
import settings
from conversation import MAIN_BRANCH, branch_sampling
from model_runtime import (
    MEMORY_HEADROOM_BYTES,
    QUANTIZED_BITS,
    DeviceProfile,
    allocated_bytes,
    device_label,
    device_profile,
    format_memory,
    imported_torch,
    memory_note,
    reserved_bytes,
)
from ui import runtime


# The hardware panel. Every figure here is one the app already acts on: the
# memory a load is judged against, the ceiling Metal is held to, and what the
# process is holding right now. They were only in the log before, which meant
# reading a log file to find out why a load was refused.
HARDWARE_UNREAD = (
    "Reading the device… press **↻ Refresh** in a moment. ChatLab imports "
    "PyTorch in the background at startup, and the device cannot be named "
    "until that has finished."
)


def hardware_card(profile: DeviceProfile | None = None) -> str:
    """What the machine is, and what it and this process are holding now.

    Read on demand rather than on a timer: the figures come from ``vm_stat``
    and the device allocator, and neither is worth a subprocess every couple
    of seconds for a page nobody may be looking at.
    """

    profile = profile if profile is not None else device_profile()
    facts = [
        (
            "Memory",
            f"{memory_note(profile.total)} in total · {memory_note(profile.available)} "
            "estimated available within ChatLab's limits",
        )
    ]
    if profile.backend is None:
        return f"**Device:** not read yet\n\n{_rows(facts)}\n\n{HARDWARE_UNREAD}"
    facts.append(
        (
            "Full weights",
            f"loaded as {profile.dtype}"
            + (
                f", and {' and '.join(QUANTIZED_BITS)} weights are quantized on the way in"
                if profile.quantizes
                else " — a quantized weight precision needs Apple Metal and is "
                "ignored here"
            ),
        )
    )
    if profile.backend == "mps":
        facts.append(("Metal cap", _metal_cap(profile)))
    facts.append(("Safety reserve", f"{format_memory(MEMORY_HEADROOM_BYTES)} kept beside the weights"))
    facts.append(("This process holds", _held(profile.backend)))
    facts.append(("Model in memory", _loaded_model()))
    return f"**Device:** {device_label(profile.backend)}\n\n{_rows(facts)}"


def _rows(facts: list[tuple[str, str]]) -> str:
    return "\n".join(f"- **{name}:** {value}" for name, value in facts)


def _metal_cap(profile: DeviceProfile) -> str:
    """The ceiling Metal allocations fail at, and where the number comes from."""

    if profile.ceiling is None:
        return (
            "PyTorch's own, because Metal did not say what it recommends. "
            "Set `mps_memory_fraction` in the settings file to hold it down."
        )
    return (
        f"{format_memory(profile.ceiling)}, {profile.fraction:.2f} of the "
        f"{memory_note(profile.recommended)} Metal recommends. A conversation "
        "that outgrows it ends with an out-of-memory message rather than a "
        "frozen Mac; `mps_memory_fraction` in the settings file moves it."
    )


def _held(backend: str) -> str:
    """What the device allocator has out on this process's behalf."""

    torch = imported_torch()
    live = allocated_bytes(backend, torch) if torch is not None else None
    taken = reserved_bytes(torch) if torch is not None else None
    if live is None and taken is None:
        return "not counted on this device — host memory keeps no such figure"
    return (
        f"{memory_note(live)} of live tensors · {memory_note(taken)} taken "
        "from the driver, cached blocks included"
    )


def _loaded_model() -> str:
    manager = runtime.MANAGER
    if not manager.model_id:
        return "none — load one on the Models page"
    precision = manager.precision or "full"
    return f"`{manager.model_id}`, {precision} weights, on {manager.device_name}"


def refresh_hardware():
    """Re-read the machine for the panel."""

    return hardware_card()


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
    # The conversation that comes back with the page answers with its own
    # sampling, so the controls have to come up holding that rather than the
    # settings file's, which is only what a conversation without any starts
    # from. Read from the file, not from the restored state, so this does not
    # depend on which of the two page-load handlers Gradio runs first.
    restored = library.read()
    if restored is not None:
        values |= settings.sampling_values(
            branch_sampling(restored, restored.get("active", MAIN_BRANCH)), saved
        )
    updates = [
        # The response-length ceiling is the context limit, so it comes back
        # with the length itself.
        gr.update(value=values["max_new_tokens"], maximum=saved.prefill_token_limit)
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
