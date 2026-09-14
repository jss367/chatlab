"""Conversation-local steering controls and portable conversation imports."""

from __future__ import annotations

import time

import gradio as gr

from conversation import MAIN_BRANCH, branch_sampling, copy_forks, put_branch_sampling
from steering import compact, from_controls, normalize, read_vector
from ui.conversations import load_conversation


EMPTY_STATUS = "Import a JSON vector to steer this conversation. Layers count from 0."


def description(value):
    if value is None:
        return EMPTY_STATUS
    state = "Enabled" if value["enabled"] and value["strength"] else "Off"
    # Model names are file-supplied text, so keep them out of Markdown markup.
    return (
        f"{state} · {value['model_id']} · layer {value['layer']} · "
        f"{value.get('width', len(value.get('vector', []))):,} entries · strength {value['strength']:g}. "
        "Applies to the prompt and response. Compatibility is checked before generation."
    )


def controls(value):
    value = compact(value)
    return (
        value,
        gr.update(value=value["enabled"] if value else False, interactive=value is not None),
        gr.update(value=value["strength"] if value else 1, interactive=value is not None),
        gr.update(value=value["layer"] if value else 0, interactive=value is not None),
        description(value),
    )


def steering_updates(forks):
    held = branch_sampling(forks, (forks or {}).get("active", MAIN_BRANCH))
    return controls(normalize(held.get("steering")))


def store(forks, value):
    forks = copy_forks(forks)
    held = branch_sampling(forks, forks["active"])
    held["steering"] = compact(value)
    put_branch_sampling(forks, forks["active"], held)
    return forks


def import_vector(path, forks):
    try:
        value = compact(read_vector(path))
    except (OSError, ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error
    return store(forks, value), *controls(value)


def remove_vector(forks):
    return store(forks, None), *controls(None)


def remember_steering(forks, value, enabled, strength, layer):
    try:
        value = compact(from_controls(value, enabled, strength, layer))
    except (ValueError, TypeError, OverflowError) as error:
        raise gr.Error(str(error)) from error
    if value is None:
        return gr.skip(), None, EMPTY_STATUS
    return store(forks, value), value, description(value)


def load_with_steering(path, turns, scale_name, forks):
    *result, value = load_conversation(path, turns, scale_name, include_steering=True)
    # The core loader preserves the system-prompt control on failure.
    if not isinstance(result[2], str):
        return (*result, *((gr.skip(),) * 6))
    return (*result, store(forks, value), *controls(value))


# ---------------------------------------------------------------- extraction

EXTRACT_EMPTY = (
    "Give examples of what you want and of the opposite, then press "
    "**Extract direction**. Nothing is applied until you choose a layer."
)

EXTRACT_BUSY = "Wait for the response to finish before extracting a vector."

# A load has the model instead. There is no response to wait for, and the
# examples would be read through weights on their way out.
EXTRACT_LOADING = "Wait for the model to finish loading before extracting a vector."

EXTRACT_NO_MODEL = "Download and load a model first."

EXTRACT_HEADERS = ["Layer", "Separation", "Vector length", "Typical activation"]

POOL_CHOICES = (
    ("The last token of each example", "last"),
    ("The mean over each example's tokens", "mean"),
)

# The effect size is measured on the examples the direction was taken from,
# so it says how cleanly this direction splits *these* examples. A direction
# that splits four examples perfectly and nothing else is exactly what a
# small contrast set produces, and the number cannot tell the reader that.
SEPARATION_CAVEAT = (
    "Separation is measured on the same examples the direction came from, so "
    "it says how cleanly the two sets split here, not how the direction will "
    "behave on anything else."
)

# What the box promised and the model cannot give: there is no turn to read
# the examples at the end of, so they were read as plain text instead.
EXTRACT_TEMPLATE_CAVEAT = (
    "Plain text, not chat turns: this model has no chat template, so each "
    "example was read as ordinary characters."
)


def empty_extraction():
    """The table, layer control and buttons with nothing extracted."""

    return (
        gr.update(value=[]),
        gr.update(value=0, maximum=0, interactive=False),
        gr.update(interactive=False),
    )


def layer_rows(stats) -> list[list]:
    """One row per layer for the table beside the extraction."""

    return [
        [
            item["layer"],
            None if item["separation"] is None else round(item["separation"], 3),
            round(item["norm"], 4),
            round(item["activation_norm"], 4),
        ]
        for item in stats or ()
    ]


def describe_layer(extraction, layer) -> str:
    """What choosing this layer would apply, in the reader's terms.

    The length of the direction against the length of an ordinary activation
    at the same layer is the number that makes a strength usable: a vector a
    hundredth the size of what it is added to does nothing at strength 1, and
    one the same size overwhelms the residual stream. README says strength
    depends on how a vector was made; this is that dependency, measured.
    """

    if not extraction:
        return EXTRACT_EMPTY
    stats = extraction["stats"]
    index = min(max(int(layer or 0), 0), len(stats) - 1)
    item = stats[index]
    share = item["norm"] / item["activation_norm"] if item["activation_norm"] else 0
    separation = (
        "no separation to measure (one example a side)"
        if item["separation"] is None
        else f"separation {item['separation']:.2f}"
    )
    return (
        f"Layer {item['layer']}: {separation}, length {item['norm']:.3g}, "
        f"{share:.0%} of a typical activation there. "
        "**Use this layer** puts it on this conversation at strength 1 "
        "and turns steering on."
    )


def extract_vector(positive_text, negative_text, use_chat_template, pool):
    """Read a direction out of the two example boxes, at every layer.

    The generation slot is held for the pass, as scoring and inspection hold
    it: without it the examples would queue behind a running reply on the
    model lock and the button would sit dead for the length of that reply.
    Claimed before memory is looked at, because a load empties memory before
    it reads the new weights and the check below would send a reader to the
    page that is already loading one.
    """

    from model_runtime import LOADING
    from steering import MAX_EXAMPLES, best_layer, parse_examples
    from ui import runtime
    from ui.common import failure_status

    positive = parse_examples(positive_text)
    negative = parse_examples(negative_text)
    if not positive or not negative:
        return (
            gr.skip(),
            *empty_extraction(),
            "Give at least one example on each side. One example per line, "
            f"up to {MAX_EXAMPLES} a side.",
        )

    held = runtime.MANAGER.claim_generation()
    if held:
        return (
            gr.skip(),
            *empty_extraction(),
            EXTRACT_LOADING if held == LOADING else EXTRACT_BUSY,
        )
    started = time.monotonic()
    try:
        if not runtime.MANAGER.loaded:
            return gr.skip(), *empty_extraction(), EXTRACT_NO_MODEL
        try:
            extraction = runtime.MANAGER.extract_steering(
                positive, negative, use_chat_template=bool(use_chat_template), pool=pool,
            )
        except Exception as error:
            return (
                gr.skip(),
                *empty_extraction(),
                failure_status("Could not extract a direction", str(error)),
            )
    finally:
        runtime.MANAGER.release_generation()

    stats = [dict(item) for item in extraction.stats]
    held_extraction = {
        "model_id": extraction.model_id,
        "load_id": extraction.load_id,
        "layers": [list(row) for row in extraction.layers],
        "stats": stats,
    }
    chosen = best_layer(stats)
    elapsed = time.monotonic() - started
    status = (
        f"Read {extraction.positive_count} wanted and "
        f"{extraction.negative_count} unwanted example"
        f"{'' if extraction.negative_count == 1 else 's'} through "
        f"{len(stats)} layer{'' if len(stats) == 1 else 's'} in {elapsed:.1f}s. "
        f"{describe_layer(held_extraction, chosen)}"
    )
    if extraction.chat_template_missing:
        status = f"{status}\n\n{EXTRACT_TEMPLATE_CAVEAT}"
    status = f"{status}\n\n{SEPARATION_CAVEAT}"
    return (
        held_extraction,
        gr.update(value=layer_rows(stats)),
        gr.update(value=chosen, maximum=max(len(stats) - 1, 0), interactive=len(stats) > 1),
        gr.update(interactive=True),
        status,
    )


def choose_layer(extraction, event: gr.SelectData):
    """Move the layer control to the row that was clicked."""

    if not extraction:
        return gr.skip(), gr.skip()
    try:
        row = int(event.index[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return gr.skip(), gr.skip()
    layer = min(max(row, 0), len(extraction["stats"]) - 1)
    return layer, describe_layer(extraction, layer)


def extracted_vector(extraction, layer):
    """The chosen layer's direction, in the shape an imported file has."""

    from steering import vector_from

    if not extraction:
        raise gr.Error("Extract a direction first.")
    stats = extraction["stats"]
    index = min(max(int(layer or 0), 0), len(stats) - 1)
    return vector_from(extraction["model_id"], index, extraction["layers"][index])


def use_extracted(forks, extraction, layer):
    """Put the chosen layer's direction on this conversation, ready to steer.

    The model is checked here and again before the next response runs. A
    direction read through one model means nothing to another's residual
    stream, and the ID is the only part of that a saved vector can carry.
    """

    from ui import runtime

    try:
        value = compact(extracted_vector(extraction, layer))
    except (ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error
    loaded = runtime.MANAGER.model_id
    note = ""
    if loaded is not None and loaded != value["model_id"]:
        note = (
            f" The model in memory is now {loaded}; load "
            f"{value['model_id']} again before steering with this."
        )
    updates = list(controls(value))
    updates[-1] = f"{updates[-1]}{note}"
    return (store(forks, value), *updates)


def download_extracted(extraction, layer):
    """Write the chosen layer's direction where the browser can fetch it."""

    import json
    import tempfile
    from pathlib import Path
    from uuid import uuid4

    from trace_export import write_private_text

    if not extraction:
        return None
    value = extracted_vector(extraction, layer)
    directory = Path(tempfile.mkdtemp(prefix="chatlab-"))
    name = value["model_id"].replace("/", "-")
    path = directory / f"steering-{name}-layer{value['layer']}-{uuid4().hex[:8]}.json"
    write_private_text(path, json.dumps(value, indent=2) + "\n")
    return str(path)
