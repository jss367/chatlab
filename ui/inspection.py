"""Layers and attention: the logit lens and the attention view."""

from __future__ import annotations

import html
import json
import threading
import time
from uuid import uuid4

import gradio as gr

import charts
import jacobian_lens
import kv_cache
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
    code_span,
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


# A click on a slice cell pins that cell's token and runs the inspection
# again, which is what typing the token and pressing the button would do.
# The value is written the way the token menu writes its bridge, so Gradio
# sees an ordinary edit; the click follows once that edit has been sent.
# The cell's token ID goes into a hidden box first, paired with the text it
# was shown for: a token whose text does not encode back to itself (a byte
# fallback, a special token) is then pinned by ID rather than re-tokenized.
JACOBIAN_JS = r"""
() => {
  if (window.chatlabJacobianInstalled) return;
  window.chatlabJacobianInstalled = true;
  document.addEventListener('click', event => {
    const cell = event.target.closest('#jacobian-lens td[data-token]');
    if (!cell) return;
    const input = document.querySelector('#jacobian-pin textarea, #jacobian-pin input');
    if (!input) return;
    let text;
    try { text = JSON.parse(cell.dataset.token); } catch (error) { return; }
    if (typeof text !== 'string') return;
    const write = (field, value) => {
      const prototype = field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, 'value').set.call(field, value);
      field.dispatchEvent(new Event('input', {bubbles: true}));
    };
    const idInput = document.querySelector('#jacobian-pin-id textarea, #jacobian-pin-id input');
    if (idInput) write(idInput, JSON.stringify({token_id: Number(cell.dataset.tokenId), text}));
    write(input, text);
    const holder = document.querySelector('#inspect-layers');
    const button = holder && (holder.tagName === 'BUTTON' ? holder : holder.querySelector('button'));
    if (button) setTimeout(() => button.click(), 120);
  });
  // A pin the reader types is not the clicked cell's token any more, even
  // when the text comes out the same: a typed pin goes through the tokenizer
  // as documented. The bridge's own write is a synthetic event, so only an
  // edit that came from the keyboard or a paste (isTrusted) clears the ID.
  document.addEventListener('input', event => {
    if (!event.isTrusted || !event.target.closest || !event.target.closest('#jacobian-pin')) return;
    const bridge = document.querySelector('#jacobian-pin-id textarea, #jacobian-pin-id input');
    if (!bridge || bridge.value === '') return;
    const prototype = bridge.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(bridge, '');
    bridge.dispatchEvent(new Event('input', {bubbles: true}));
  }, true);
  // A readout arrives with the selected token in its last column, which a
  // wide window puts out of sight; bring that column into view once.
  const reveal = () => {
    document.querySelectorAll('#jacobian-lens .jl-grid-wrap:not([data-revealed])').forEach(wrap => {
      wrap.dataset.revealed = '1';
      const cell = wrap.querySelector('thead .jl-selected');
      if (cell) wrap.scrollLeft = Math.max(0, cell.offsetLeft + cell.offsetWidth - wrap.clientWidth + 12);
    });
  };
  new MutationObserver(reveal).observe(document.body, {childList: true, subtree: true});
}
"""

JACOBIAN_CSS = """
.jl-grid-wrap { overflow: auto; max-height: 420px; margin: 0.3rem 0; border: 1px solid var(--viz-grid); border-radius: 6px; }
.jl-grid { border-collapse: separate; border-spacing: 0; font-size: 0.74rem; font-variant-numeric: tabular-nums; }
.jl-grid th, .jl-grid td { padding: 0.15rem 0.35rem; white-space: nowrap; color: var(--viz-ink); border-bottom: 1px solid var(--viz-grid); border-right: 1px solid var(--viz-grid); }
.jl-grid thead th { position: sticky; top: 0; z-index: 2; background: var(--background-fill-primary); color: var(--viz-muted); font-weight: 500; }
.jl-grid tbody th { position: sticky; left: 0; z-index: 1; background: var(--background-fill-primary); color: var(--viz-muted); font-weight: 500; text-align: right; }
.jl-grid thead th:first-child { left: 0; z-index: 3; }
.jl-grid code { background: none; padding: 0; font-size: inherit; }
.jl-cell { cursor: pointer; background: color-mix(in srgb, var(--viz-line) calc(var(--jl-heat, 0) * 70%), transparent); }
.jl-cell:hover { outline: 1px solid var(--viz-line); outline-offset: -1px; }
.jl-cell sup { color: var(--viz-muted); margin-left: 0.15rem; font-size: 0.62rem; }
.jl-grid .jl-selected { box-shadow: inset 0 -2px 0 var(--viz-line); }
.jl-grid thead .jl-selected { color: var(--viz-ink); font-weight: 600; }
.jl-output th, .jl-output td { border-top: 2px solid var(--viz-axis); }
"""


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


def _pin_by_id(pinned_token_id, pinned_text: str) -> dict:
    """The exact token ID a clicked cell supplied, if it still matches the pin.

    A click writes ``{"token_id", "text"}`` beside the visible text; the ID
    counts only while the text box still shows that same text, so a pin typed
    over it goes through the tokenizer as before and the ID is left unused.
    The page script also clears the record on any edit the reader makes to
    the visible box (see JACOBIAN_JS), so retyping the clicked text later
    tokenizes it rather than reviving the clicked ID; the text check here is
    the server-side guard for a page script that did not run.
    """
    try:
        record = json.loads(pinned_token_id or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("text") != pinned_text:
        return {}
    token_id = record.get("token_id")
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        return {}
    return {"pinned_id": token_id}


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
    pinned_token_id: str = "",
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
    pin = _pin_by_id(pinned_token_id, pinned_text or "")

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
        note = None
        try:
            options = {"context_count": len(context_ids), "load_id": load_id}
            if steering is not None:
                options["steering"] = steering
            if lens_mode == "Jacobian":
                # The state's import counts only for the load it was made for.
                # Otherwise the manager's current lens serves, and when there
                # is none the one written down for this model is brought back.
                imported = imported_lens if (imported_lens or {}).get("load_id") == load_id else None
                recalled = None
                if imported is None and runtime.MANAGER.jacobian_lens_import() is None:
                    recalled, note = recall_lens()
                insight = runtime.MANAGER.inspect_jacobian(
                    sequence, index,
                    lens_id=(imported or {}).get("import_id"),
                    pinned_text=pinned_text or "", **pin, **options,
                ).to_dict()
                if recalled:
                    insight["recalled"] = recalled
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
                failure_status("Could not inspect that token", f"{error} {note}" if note else str(error)),
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
        shown = code_span(repr(insight["token_text"]))
        read = len(insight["layers"]) - 1
        status = (
            f"{where} {position + 1}: {shown}, read through {read} "
            f"layers in {time.monotonic() - started:.1f}s."
        )
        if insight.get("kind") == "jacobian":
            window = len((insight.get("slice") or {}).get("tokens") or [])
            status = (
                f"{where} {position + 1}: {shown}, read after processing this token "
                f"at {len(insight['layers'])} fitted layers, over {window} positions, "
                f"in {time.monotonic() - started:.1f}s."
            )
            if insight.get("recalled"):
                status = f"{status} Using the remembered lens {code_span(str(insight['recalled']))}."
        elif not read:
            status = f"{status} {INSPECT_OUTPUT_ONLY}"
        if not layer_count and insight.get("kind") != "jacobian":
            status = f"{status} This model did not return attention weights."
        if inspection_session is not None:
            insight["inspection_controls"] = {"session": inspection_session, "revision": revision}
        insight["saved_target"] = dict(target)
        # The key-value cache view reads the cache this pass kept, and only
        # while it still came from this load.
        insight["load_id"] = load_id
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


def render_kv_cache(insight: dict | None, layer, metric):
    """One layer of the cache the inspection kept, and the layer slider's range.

    Bound to the readout's state as well as to the controls, so a new
    readout brings its cache view with it and a cleared one takes it away.
    The cache is read from memory, not rebuilt: a readout whose cache has
    been released since says so and asks for another inspection.
    """

    if not insight:
        return charts.EMPTY_KV_CACHE, gr.skip()
    controls = insight.get("inspection_controls")
    if controls and not INSPECTION_CONTROLS.current(controls["session"], controls["revision"]):
        return gr.skip(), gr.skip()
    if insight.get("kind") == "jacobian":
        return (
            '<div class="viz-empty">Select the Logit lens to read the key-value cache '
            "behind a prediction.</div>",
            gr.skip(),
        )
    tokens = insight.get("tokens") or []
    try:
        view = runtime.MANAGER.read_kv_cache(
            [int(token["token_id"]) for token in tokens], int(layer or 1),
            load_id=insight.get("load_id"),
        )
    except ModelChanged:
        return f'<div class="viz-empty">{html.escape(INSPECT_MODEL_CHANGED)}</div>', gr.skip()
    except (kv_cache.CacheGone, kv_cache.CacheBusy) as error:
        return f'<div class="viz-empty">{html.escape(str(error))}</div>', gr.skip()
    except Exception as error:  # noqa: BLE001 - the view is optional, the readout above it is not
        return (
            f'<div class="viz-empty">Could not read the cache: {html.escape(str(error))}</div>',
            gr.skip(),
        )
    return (
        charts.kv_cache_grid(view, tokens, metric),
        gr.update(maximum=max(view["summary"]["layers"], 1), value=view["layer"]),
    )


def render_lens(insight: dict) -> str:
    if insight.get("kind") == "jacobian":
        return charts.jacobian_lens_chart(insight)
    return charts.logit_lens_chart(insight)


def recall_lens() -> tuple[str | None, str | None]:
    """Import the lens written down for the loaded model; ``(name, note)``.

    Called under the generation claim with no lens imported for this load.
    A record whose file has gone, or that the current weights refuse, leaves
    the manager as it was and the ordinary "import a lens" message follows.
    A record made for another revision of the same model ID is not tried at
    all, since the lens was fitted for other weights; the note says so, for
    the inspection to add to that message.
    """
    record = jacobian_lens.remembered(runtime.MANAGER.model_id or "")
    if record is None:
        return None, None
    remembered, current = record.get("model_revision"), runtime.MANAGER.model_revision()
    if isinstance(remembered, str) and isinstance(current, str) and remembered != current:
        return None, "The remembered lens was imported for another revision of this model; import it again."
    try:
        imported = runtime.MANAGER.import_jacobian_lens(record["path"], record.get("fitted_model_id") or "")
    except Exception:  # noqa: BLE001 - the inspection reports the missing lens itself
        return None, None
    return imported["name"], None


def import_jacobian_lens(path, fitted_model_id, repository="", filename=""):
    """Keep large lens tensors in the model manager, never in browser state.

    A repository and file name fetch the lens from the Hub first; otherwise
    the chosen file is copied beside the settings, since a browser upload
    lands in a cache that does not outlive the session. A successful import
    is written down for the loaded model, so the next load of it finds the
    lens without this step.
    """
    repository = (repository or "").strip()
    filename = (filename or "").strip()
    source = {}
    if repository or filename:
        # The slot is checked, not held, over the download. A transfer of up
        # to 2 GiB can take minutes, and holding the slot for it would block
        # chat and loading the whole time; the import below validates the
        # lens against whatever model is loaded when it runs, under the model
        # lock, so a model swapped in mid-download is refused rather than
        # misread. A finished download is kept on disk, so a "busy" answer
        # after it costs nothing to retry. The check here spares the bytes
        # when the model is already known to be busy.
        held = runtime.MANAGER.claim_generation()
        if held:
            return gr.skip(), INSPECT_LOADING if held == LOADING else INSPECT_BUSY
        # The import below is bound to this load: a model swapped in during
        # the transfer is refused by name, not read through the new weights.
        load_id = runtime.MANAGER.load_id
        runtime.MANAGER.release_generation()
        try:
            path = str(jacobian_lens.download(repository, filename))
        except ValueError as error:
            return gr.skip(), failure_status("Could not fetch the lens", str(error))
        source = {"repository": repository, "filename": filename}
    if not path:
        return gr.skip(), "Choose a saved lens.pt file, or name a Hub repository and file."
    held = runtime.MANAGER.claim_generation()
    if held:
        return gr.skip(), INSPECT_LOADING if held == LOADING else INSPECT_BUSY
    try:
        if source and runtime.MANAGER.load_id != load_id:
            return gr.skip(), failure_status(
                "Could not import the lens",
                "The model was reloaded while the lens downloaded. The file is kept; press Import lens again.",
            )
        imported = runtime.MANAGER.import_jacobian_lens(path, fitted_model_id or "")
    except Exception as error:
        return gr.skip(), failure_status("Could not import the lens", str(error))
    finally:
        runtime.MANAGER.release_generation()
    status = (
        f"Imported for {code_span(str(imported['model_id']))}: "
        f"{imported['layers']} fitted layers, {imported['n_prompts']:,} fitting prompts."
    )
    # Copied only once the file has passed every check, so a rejected upload
    # never lands beside the settings and nothing is left behind on failure.
    if not source:
        try:
            imported = imported | {"path": str(jacobian_lens.keep(path))}
        except OSError as error:
            return imported, f"{status} It could not be kept for later loads: {html.escape(str(error))}"
    # The slot was released after the import, so another client may have
    # imported a different lens since. The record must describe the lens that
    # is actually in memory, so the manager writes it only for the import
    # that is still current, checking and writing under its own lock.
    outcome = runtime.MANAGER.remember_jacobian_lens(imported["import_id"], imported | source)
    if outcome == "replaced":
        return imported, f"{status} Another lens was imported meanwhile, so this one was not remembered."
    if outcome == "unwritable":
        return imported, (
            f"{status} It could not be written down for later loads; check that the settings "
            "folder is writable. This session keeps using it."
        )
    return imported, f"{status} Remembered for this model, so its next load picks the lens up again."


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
