"""Layers and attention: the logit lens and the attention view."""

from __future__ import annotations

import html
import threading
import time
from uuid import uuid4

import gradio as gr

import charts
from model_runtime import (
    LOADING,
    ModelChanged,
)
from ui import icons, runtime
from ui.common import (
    NAV_ICONS,
    PAGES,
    failure_status,
)
from ui.panel import (
    current_strip_generation,
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


class InspectionControls:
    """Keep live control revisions outside Gradio's queued input snapshots."""

    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def new_session(self):
        session = uuid4().hex
        with self._lock:
            self._sessions[session] = (0, "Logit", "")
        return session

    def forget(self, session):
        with self._lock:
            self._sessions.pop(session, None)

    def change(self, session, *, mode=None, pin=None):
        with self._lock:
            previous = self._sessions.get(session)
            if previous is None:
                return
            revision, old_mode, old_pin = previous
            self._sessions[session] = (
                revision + 1, old_mode if mode is None else mode,
                old_pin if pin is None else pin,
            )

    def capture(self, session, mode, pin):
        with self._lock:
            current = self._sessions.get(session)
            # Also reject an old queued request that starts after the edit.
            return current[0] if current is not None and current[1:] == (mode, pin) else None

    def current(self, session, revision):
        with self._lock:
            current = self._sessions.get(session)
            return current is not None and current[0] == revision


INSPECTION_CONTROLS = InspectionControls()


def remember_inspect_target(strip: str):
    """A select listener that keeps the clicked position for the inspector.

    Unlike remember_selection(), every token counts: a prompt token has layers
    and attention behind it just as a response token does. Only the first
    token of a sequence has nothing to show, and inspect_layers() says so.
    """

    def remember(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
        generation, metrics = metrics_state
        if generation != current_strip_generation(strip):
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
    score_metrics_state: tuple[int, list[dict]] | None = None,
    score_context_state: tuple | None = None,
    chat_metrics_state: tuple[int, list[dict]] | None = None,
    chat_context_state: tuple | None = None,
    lens_mode: str = "Logit",
    imported_lens: dict | None = None,
    pinned_text: str = "",
    inspection_session: str | None = None,
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
    Undo, Load, a fork switch - are caught by the stamp instead. Scored tokens
    and chat replies retain independent metrics, context and stamps, so a
    scoring pass cannot take away the latest reply's inspection target.
    The relevant stamp is checked before the frame goes out and again
    once it has arrived, so a readout for a token that is gone is taken down.
    """

    skip = gr.skip()
    refused = (skip, skip, skip, skip)
    revision = INSPECTION_CONTROLS.capture(inspection_session, lens_mode, pinned_text or "")

    def controls_current():
        return inspection_session is None or INSPECTION_CONTROLS.current(inspection_session, revision)

    if not controls_current():
        yield (skip,) * 5
        return
    if not target or target.get("generation") != current_strip_generation(target["strip"]):
        yield (*refused, INSPECT_HINT)
        return
    if target["strip"] == "score":
        if score_metrics_state is None or score_context_state is None:
            yield (*refused, INSPECT_GONE)
            return
        metrics_state = score_metrics_state
        context_state = score_context_state
    elif target["strip"] == "response" and chat_metrics_state is not None:
        metrics_state = chat_metrics_state
        context_state = chat_context_state
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
        if index == 0 and lens_mode != "Jacobian":
            yield (*refused, INSPECT_FIRST)
            return
        sequence = context_ids + [int(metric["token_id"]) for metric in metrics]

        started = time.monotonic()
        try:
            options = {"context_count": len(context_ids), "load_id": load_id}
            if steering is not None:
                options["steering"] = steering
            if lens_mode == "Jacobian":
                insight = runtime.MANAGER.inspect_jacobian(
                    sequence, index,
                    lens_id=(imported_lens or {}).get("import_id"),
                    pinned_text=pinned_text or "", **options,
                ).to_dict()
            else:
                insight = runtime.MANAGER.inspect(sequence, index, **options).to_dict()
        except ModelChanged:
            if not controls_current():
                yield (skip,) * 5
                return
            yield (*refused, INSPECT_MODEL_CHANGED)
            return
        except Exception as error:
            if not controls_current():
                yield (skip,) * 5
                return
            yield (
                *refused,
                failure_status("Could not inspect that token", str(error)),
            )
            return
        if not controls_current():
            yield (skip,) * 5
            return
        if target["generation"] != current_strip_generation(target["strip"]):
            yield (*refused, INSPECT_GONE)
            return

        layer_count = len(insight["attention"])
        layer = min(max(int(layer or 0), 0), layer_count)
        where = "Prompt token" if target["strip"] == "prompt" else "Token"
        shown = html.escape(repr(insight["token_text"]))
        read = len(insight["layers"]) - 1
        status = (
            f"{where} {position + 1}: <code>{shown}</code>, read through {read} "
            f"layers in {time.monotonic() - started:.1f}s."
        )
        if insight.get("kind") == "jacobian":
            status = (
                f"{where} {position + 1}: <code>{shown}</code>, read after processing this token "
                f"at {len(insight['layers'])} fitted layers in {time.monotonic() - started:.1f}s."
            )
        elif not read:
            status = f"{status} {INSPECT_OUTPUT_ONLY}"
        if not layer_count and insight.get("kind") != "jacobian":
            status = f"{status} This model did not return attention weights."
        if inspection_session is not None:
            insight["inspection_controls"] = {"session": inspection_session, "revision": revision}
        insight["saved_target"] = dict(target)
        from experiment_runs import SESSION_ID
        insight["saved_session"] = SESSION_ID
        frame = (
            render_lens(insight),
            render_attention(insight, layer),
            gr.update(maximum=max(layer_count, 1), value=layer),
            insight,
            status,
        )
        if not controls_current():
            yield (skip,) * 5
            return
        yield frame
        # Resumed once the browser has the frame above. If the strips were
        # replaced while it was in flight, their reset was applied first and
        # the readout now sits on top of it, so take it back down.
        if not controls_current():
            # The control reset may have arrived before this older frame.
            # The held generation slot prevents a newer inspection result
            # from landing before this cleanup. Empty HTML fits either mode.
            yield ("", charts.EMPTY_ATTENTION, skip, None, INSPECT_HINT)
        elif target["generation"] != current_strip_generation(target["strip"]):
            yield (charts.EMPTY_LENS, charts.EMPTY_ATTENTION, skip, None, INSPECT_GONE)
    finally:
        runtime.MANAGER.release_generation()


def render_attention(insight: dict | None, layer):
    """Repaint the attention strip for another layer without a new pass."""

    if not insight:
        return gr.skip()
    controls = insight.get("inspection_controls")
    if controls and not INSPECTION_CONTROLS.current(controls["session"], controls["revision"]):
        return gr.skip()
    if insight.get("kind") == "jacobian":
        return '<div class="viz-empty">Select the Logit lens to inspect attention behind a prediction.</div>'
    return charts.attention_strip(insight, int(layer or 0))


def render_lens(insight: dict) -> str:
    if insight.get("kind") == "jacobian":
        return charts.jacobian_lens_chart(insight)
    return charts.logit_lens_chart(insight)


def import_jacobian_lens(path, fitted_model_id):
    """Keep large lens tensors in the model manager, never in browser state."""
    if not path:
        return gr.skip(), "Choose a saved lens.pt file first."
    held = runtime.MANAGER.claim_generation()
    if held:
        return gr.skip(), INSPECT_LOADING if held == LOADING else INSPECT_BUSY
    try:
        imported = runtime.MANAGER.import_jacobian_lens(path, fitted_model_id or "")
        return imported, (
            f"Imported for `{html.escape(imported['model_id'])}`: "
            f"{imported['layers']} fitted layers, {imported['n_prompts']:,} fitting prompts. "
            "Reloading the model requires importing the lens again."
        )
    except Exception as error:
        return gr.skip(), failure_status("Could not import the lens", str(error))
    finally:
        runtime.MANAGER.release_generation()


def change_lens_mode(mode, inspection_session=None):
    INSPECTION_CONTROLS.change(inspection_session, mode=mode)
    jacobian = mode == "Jacobian"
    return (
        gr.update(visible=jacobian), gr.update(visible=not jacobian),
        charts.EMPTY_JACOBIAN if jacobian else charts.EMPTY_LENS,
        charts.EMPTY_ATTENTION, None, INSPECT_HINT,
    )


def change_pinned_token(text, inspection_session):
    INSPECTION_CONTROLS.change(inspection_session, pin=text or "")
    # Clear even when the callback's insight snapshot was still empty.
    return "", charts.EMPTY_ATTENTION, None, INSPECT_HINT


def reset_inspection(insight: dict | None):
    """Empty the inspector when the strips it described are replaced.

    Bound to the response metrics state, which every path that redraws the
    strips writes. Streaming writes it on every frame too, so this skips
    while there is nothing to clear rather than repainting an empty panel a
    hundred times per response.
    """

    if insight is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    empty = charts.EMPTY_JACOBIAN if insight.get("kind") == "jacobian" else charts.EMPTY_LENS
    return empty, charts.EMPTY_ATTENTION, None, INSPECT_HINT


# One rule per tile: which drawing goes in the box the stylesheet has already
# opened above the page's own name. Gradio stamps each option's text on its
# label as data-testid, which is the only hook a Radio gives CSS.
#
# A mask carries no text with it, so unlike the emoji these replaced there is
# nothing here for a screen reader to read out in front of the page's name.
NAV_TILE_CSS = "\n".join(
    icons.mask_rule(f'#nav label[data-testid="{name}-radio-label"]::before', NAV_ICONS[name])
    for name in PAGES
)
