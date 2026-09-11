"""Layers and attention: the logit lens and the attention view."""

from __future__ import annotations

import html
import time

import gradio as gr

import charts
from model_runtime import (
    LOADING,
    ModelChanged,
)
from ui import runtime
from ui.common import (
    NAV_ICONS,
    PAGES,
    failure_status,
)
from ui.panel import (
    current_metrics_generation,
    event_index,
)


INSPECT_HINT = "Click a token above, then press **Inspect layers**."


INSPECT_BUSY = "Wait for the response to finish before inspecting a token."


# A load has the model instead, and the strip being inspected belongs to the
# weights on their way out: there is no response to wait for.
INSPECT_LOADING = "Wait for the model to finish loading before inspecting a token."


INSPECT_GONE = "That token is no longer on screen. Click one and try again."


INSPECT_FIRST = "Nothing came before this token, so the model never predicted it."


INSPECT_MODEL_CHANGED = (
    "The model has been reloaded since these tokens were produced, so they "
    "cannot be explained by the weights in memory. Generate or score again."
)


INSPECT_OUTPUT_ONLY = (
    "Only the output is shown: this model's intermediate layers could not be "
    "read the way it reads its own output."
)


def remember_inspect_target(strip: str):
    """A select listener that keeps the clicked position for the inspector.

    Unlike remember_selection(), every token counts: a prompt token has layers
    and attention behind it just as a response token does. Only the first
    token of a sequence has nothing to show, and inspect_layers() says so.
    """

    def remember(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
        generation, metrics = metrics_state
        if generation != current_metrics_generation():
            return None
        try:
            index = event_index(event)
            metrics[index]
        except (IndexError, TypeError, ValueError):
            return None
        return {"generation": generation, "strip": strip, "index": index}

    return remember


def inspect_layers(
    target: dict | None,
    metrics_state: tuple[int, list[dict]],
    prompt_metrics_state: tuple[int, list[dict]],
    context_state: tuple[int, list[int]],
    layer,
):
    """Run the logit lens and attention readout for the clicked token.

    The model's input is rebuilt from the prompt ids published with the
    response and the token ids in the response metrics, so the pass sees
    exactly the sequence the token was generated from.

    This is a generator for the same reason generate_reply() is: Gradio does
    not resume a streaming handler until the browser has been sent the frame
    it yielded. The generation slot is therefore held not just for the pass
    but until the readout is on screen, so Send, Retry and Branch cannot
    slip in between the two and have the readout land on top of their
    reset. Paths that replace the strips without taking the slot - Clear,
    Undo, Load, a fork switch, Score text - are caught by the stamp instead:
    it is checked before the frame goes out and again once it has arrived,
    and a readout for a token that is gone is taken back down.
    """

    skip = gr.skip()
    refused = (skip, skip, skip, skip)
    if not target or target.get("generation") != current_metrics_generation():
        yield (*refused, INSPECT_HINT)
        return
    generation, metrics = metrics_state
    _prompt_generation, prompt_metrics = prompt_metrics_state
    context_generation, context_ids, load_id = context_state[:3]
    steering = context_state[3] if len(context_state) > 3 else None
    if generation != target["generation"] or context_generation != generation:
        yield (*refused, INSPECT_GONE)
        return
    # Claimed before memory is looked at, not after it. A load empties memory
    # before it reads the new weights, so the check below finds nothing
    # loaded for the whole of that phase and would send the reader off to
    # load a model while one was already loading. The claim is also what
    # makes the load check after it worth making: while the slot is held no
    # load can start, so the weights the token ids came from cannot be
    # swapped out between that check and the pass that reads them. What it
    # guards here is list arithmetic, and the slot goes back on each refusal.
    held = runtime.MANAGER.claim_generation()
    if held:
        yield (*refused, INSPECT_LOADING if held == LOADING else INSPECT_BUSY)
        return
    try:
        if not runtime.MANAGER.loaded:
            yield (*refused, "Download and load a model first.")
            return
        # Loading a model leaves the strips on screen, and their token ids
        # mean nothing to a different tokenizer, so the ids carry the load
        # that produced them and only that load may explain them. The load,
        # not the model ID: re-downloading the same ID can bring in a newer
        # snapshot. inspect() compares it again under the model lock, which
        # is where it is finally decided; read under the claim, this one can
        # no longer be overtaken by a load starting behind it.
        if load_id != runtime.MANAGER.load_id:
            yield (*refused, INSPECT_MODEL_CHANGED)
            return

        context_ids = [int(value) for value in context_ids]
        position = int(target["index"])
        if target["strip"] == "prompt":
            if (
                position >= len(prompt_metrics)
                or position >= len(context_ids)
                or int(prompt_metrics[position]["token_id"]) != context_ids[position]
            ):
                yield (*refused, INSPECT_GONE)
                return
            index = position
        else:
            if position >= len(metrics):
                yield (*refused, INSPECT_GONE)
                return
            index = len(context_ids) + position
        if index == 0:
            yield (*refused, INSPECT_FIRST)
            return
        sequence = context_ids + [int(metric["token_id"]) for metric in metrics]

        started = time.monotonic()
        try:
            insight = runtime.MANAGER.inspect(
                sequence, index, context_count=len(context_ids), load_id=load_id,
                **({"steering": steering} if steering is not None else {}),
            ).to_dict()
        except ModelChanged:
            yield (*refused, INSPECT_MODEL_CHANGED)
            return
        except Exception as error:
            yield (
                *refused,
                failure_status("Could not inspect that token", str(error)),
            )
            return
        if target["generation"] != current_metrics_generation():
            yield (*refused, INSPECT_GONE)
            return

        layer_count = len(insight["attention"])
        layer = min(max(int(layer or 0), 0), layer_count)
        where = "Prompt token" if target["strip"] == "prompt" else "Token"
        shown = html.escape(repr(insight["token_text"]))
        read = len(insight["layers"]) - 1
        status = (
            f"{where} {position + 1}: `{shown}`, read through {read} "
            f"layers in {time.monotonic() - started:.1f}s."
        )
        if not read:
            status = f"{status} {INSPECT_OUTPUT_ONLY}"
        if not layer_count:
            status = f"{status} This model did not return attention weights."
        yield (
            charts.logit_lens_chart(insight),
            charts.attention_strip(insight, layer),
            gr.update(maximum=max(layer_count, 1), value=layer),
            insight,
            status,
        )
        # Resumed once the browser has the frame above. If the strips were
        # replaced while it was in flight, their reset was applied first and
        # the readout now sits on top of it, so take it back down.
        if target["generation"] != current_metrics_generation():
            yield (charts.EMPTY_LENS, charts.EMPTY_ATTENTION, skip, None, INSPECT_GONE)
    finally:
        runtime.MANAGER.release_generation()


def render_attention(insight: dict | None, layer):
    """Repaint the attention strip for another layer without a new pass."""

    if not insight:
        return gr.skip()
    return charts.attention_strip(insight, int(layer or 0))


def reset_inspection(insight: dict | None):
    """Empty the inspector when the strips it described are replaced.

    Bound to the response metrics state, which every path that redraws the
    strips writes. Streaming writes it on every frame too, so this skips
    while there is nothing to clear rather than repainting an empty panel a
    hundred times per response.
    """

    if insight is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return charts.EMPTY_LENS, charts.EMPTY_ATTENTION, None, INSPECT_HINT


# One rule per tile: the icon drawn above the page's own name. Gradio stamps
# each option's text on its label as data-testid, which is the only hook a
# Radio gives CSS.
#
# The icon is drawn on the label, so it would otherwise join the radio's
# accessible name and have a screen reader read "speech balloon Chat". The
# empty string after the slash is the generated text's alternative text,
# which keeps it out of the name and leaves the page's own name to stand for
# the tile - the same name that is now printed under it.
NAV_TILE_CSS = "\n".join(
    f'#nav label[data-testid="{name}-radio-label"]::before '
    f'{{ content: "{NAV_ICONS[name]}" / ""; }}'
    for name in PAGES
)
