"""Layers and attention: the logit lens and the attention view."""

from __future__ import annotations

import html
import json
import threading
import time
from uuid import uuid4

import gradio as gr

from chatlab import charts
from chatlab import jacobian_lens
from chatlab import kv_cache
from chatlab.model_errors import ModelChanged
from chatlab import steering as steering_vectors
from chatlab.model_runtime import LOADING
from chatlab.ui import icons, runtime
from chatlab.ui.common import (
    NAV_ICONS,
    PAGES,
    failure_status,
)
from chatlab.ui.panel import (
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
    "These tokens were produced by different weights from the ones in "
    "memory, so those weights cannot explain them. Generate or score again."
)


INSPECT_OUTPUT_ONLY = (
    "Only the output is shown: this model's intermediate layers could not be "
    "read the way it reads its own output."
)


KV_CACHE_STEERED = (
    "A steered inspection keeps no key-value cache, so there is nothing to read here. "
    "Turn steering off and press Inspect layers to see the cache."
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
  // Ticked concepts, in the order they were ticked, which picks each one's
  // color. They outlive a change of block range, which redraws the same
  // readout under the same key, and are cleared by any other readout.
  const PALETTE = 6;
  const ticked = {key: null, order: []};
  const paint = figure => {
    figure.querySelectorAll('.jl-pick').forEach(box => {
      const slot = ticked.order.indexOf(box.value);
      box.checked = slot >= 0;
      const row = box.closest('tr');
      if (row) { if (slot >= 0) row.dataset.hl = slot % PALETTE; else delete row.dataset.hl; }
    });
    // Ticked concepts lead the table in the order they were ticked; the
    // rest keep their rank.
    figure.querySelectorAll('.jl-concept-wrap tbody').forEach(body => {
      const rows = [...body.querySelectorAll('tr[data-concept]')];
      rows.forEach((row, rank) => { if (row.dataset.rank === undefined) row.dataset.rank = rank; });
      const slot = row => {
        const at = ticked.order.indexOf(row.dataset.concept);
        return at < 0 ? Infinity : at;
      };
      rows.sort((a, b) => slot(a) - slot(b) || a.dataset.rank - b.dataset.rank);
      body.append(...rows);
    });
    // A position read as several ticked concepts carries a band of each.
    figure.querySelectorAll('.jl-word').forEach(word => {
      const here = (word.dataset.concepts || '').split(' ');
      const slots = ticked.order.map((token, slot) => here.includes(token) ? slot % PALETTE : -1).filter(slot => slot >= 0);
      if (!slots.length) {
        delete word.dataset.hl;
        word.style.removeProperty('--jl-paint');
        return;
      }
      word.dataset.hl = slots[0];
      const step = 100 / slots.length;
      const bands = slots.map((slot, band) =>
        `color-mix(in srgb, var(--jl-hue-${slot}) var(--jl-mix), transparent) ${(band * step).toFixed(1)}% ${((band + 1) * step).toFixed(1)}%`);
      word.style.setProperty('--jl-paint', `linear-gradient(to bottom, ${bands.join(', ')})`);
    });
  };
  const adopt = () => {
    document.querySelectorAll('#jacobian-lens:not([data-painted])').forEach(figure => {
      figure.dataset.painted = '1';
      if (figure.dataset.key !== ticked.key) { ticked.key = figure.dataset.key; ticked.order = []; }
      paint(figure);
    });
  };
  new MutationObserver(adopt).observe(document.body, {childList: true, subtree: true});
  adopt();
  document.addEventListener('change', event => {
    const box = event.target.closest && event.target.closest('#jacobian-lens .jl-pick');
    if (!box) return;
    ticked.order = ticked.order.filter(token => token !== box.value);
    if (box.checked) ticked.order.push(box.value);
    paint(box.closest('#jacobian-lens'));
  });
  const shown = text => {
    const visible = text.replace(/\n/g, '\u21b5').replace(/\t/g, '\u21e5');
    return visible.trim() === '' && visible !== '' ? visible.replace(/ /g, '\u2420') : visible;
  };
  const element = (tag, className, text) => {
    const made = document.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  };
  // A click on a position lists its top token at every block in range,
  // highest block first, each with its count by block and a box to tick it.
  document.addEventListener('click', event => {
    const word = event.target.closest('#jacobian-lens .jl-word');
    const close = event.target.closest('#jacobian-lens .jl-detail-close');
    if (!word && !close) return;
    const holder = (word || close).closest('.jl-concepts');
    const detail = holder.querySelector('.jl-detail');
    holder.querySelectorAll('.jl-word.jl-open').forEach(open => open.classList.remove('jl-open'));
    if (close || (!detail.hidden && detail.dataset.pos === word.dataset.pos)) {
      detail.hidden = true;
      return;
    }
    let data;
    try { data = JSON.parse(holder.dataset.readout); } catch (error) { return; }
    const position = Number(word.dataset.pos);
    detail.replaceChildren();
    const head = element('div', 'jl-detail-head');
    const title = element('span', '', `Position ${data.positions[position]} `);
    title.append(element('code', '', shown(word.textContent)));
    head.append(title);
    const button = element('button', 'jl-detail-close', '\u00d7');
    button.type = 'button';
    button.setAttribute('aria-label', 'Close');
    head.append(button);
    detail.append(head);
    const table = element('table', 'viz-table jl-concept-table');
    const top = element('tr');
    ['Block', '', 'Top token', 'Count by block'].forEach(name => top.append(element('th', '', name)));
    table.appendChild(element('thead')).append(top);
    const body = table.appendChild(element('tbody'));
    for (let block = data.blocks.length - 1; block >= 0; block -= 1) {
      const token = data.top[block][position];
      if (token === null || token === undefined) continue;
      const row = body.appendChild(element('tr'));
      row.append(element('td', 'jl-count', String(data.blocks[block])));
      const box = element('input', 'jl-pick');
      box.type = 'checkbox';
      box.value = String(token);
      row.appendChild(element('td')).append(box);
      row.appendChild(element('td')).append(element('code', '', shown(data.texts[token] || '')));
      const cell = row.appendChild(element('td', 'jl-strip-cell'));
      const strip = cell.appendChild(element('span', 'jl-strip'));
      (data.counts[token] || []).forEach((count, column, counts) => {
        const mark = element('i');
        mark.style.setProperty('--jl-level', (count / (Math.max(...counts) || 1)).toFixed(2));
        mark.title = `Block ${data.blocks[column]}: ${count} of ${data.positions.length} positions`;
        strip.append(mark);
      });
    }
    detail.append(table);
    detail.dataset.pos = word.dataset.pos;
    detail.hidden = false;
    word.classList.add('jl-open');
    paint(holder.closest('#jacobian-lens'));
  });
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
#jacobian-lens { --jl-hue-0: #f59e0b; --jl-hue-1: #10b981; --jl-hue-2: #0ea5e9; --jl-hue-3: #6366f1; --jl-hue-4: #f43f5e; --jl-hue-5: #14b8a6; --jl-mono: ui-monospace, SFMono-Regular, Menlo, monospace; --jl-mix: 34%; }
.dark #jacobian-lens { --jl-mix: 48%; color-scheme: dark; }
#jacobian-lens [data-hl="0"] { --jl-hue: var(--jl-hue-0); }
#jacobian-lens [data-hl="1"] { --jl-hue: var(--jl-hue-1); }
#jacobian-lens [data-hl="2"] { --jl-hue: var(--jl-hue-2); }
#jacobian-lens [data-hl="3"] { --jl-hue: var(--jl-hue-3); }
#jacobian-lens [data-hl="4"] { --jl-hue: var(--jl-hue-4); }
#jacobian-lens [data-hl="5"] { --jl-hue: var(--jl-hue-5); }
.jl-text { max-height: 320px; overflow: auto; margin: 0.5rem 0; padding: 0.8rem 1rem; border-radius: 12px; background: color-mix(in srgb, var(--viz-grid) 30%, transparent); font-family: Charter, "Iowan Old Style", Georgia, serif; font-size: 0.95rem; line-height: 2.1; color: var(--viz-ink); }
.jl-word { cursor: pointer; white-space: pre-wrap; border-radius: 3px; padding: 0.15rem 0; transition: background-color 0.12s; }
.jl-word:hover { background-color: color-mix(in srgb, var(--viz-line) 12%, transparent); }
.jl-word.jl-open { outline: 1.5px solid var(--viz-line); outline-offset: 1px; }
.jl-word.jl-selected { text-decoration: underline 2px var(--viz-line); text-underline-offset: 5px; }
.jl-word[data-hl] { background: var(--jl-paint); }
.jl-special { font-family: var(--jl-mono); font-size: 0.72rem; color: var(--viz-muted); }
.jl-ws { font-family: var(--jl-mono); font-size: 0.8em; color: var(--viz-muted); opacity: 0.6; }
.jl-concept-wrap { max-height: 460px; overflow: auto; margin: 0.4rem 0; }
.jl-concept-table { border: none; border-collapse: separate; border-spacing: 0 3px; }
.jl-concept-table th { font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; border: none; border-bottom: 1px solid var(--viz-grid); padding-top: 0.3rem; padding-bottom: 0.35rem; z-index: 1; }
.jl-concept-table th * { color: inherit; }
.jl-concept-table td { padding: 0.3rem 0.5rem; border: 1px solid transparent; border-left: none; border-right: none; transition: background-color 0.12s; }
.jl-concept-table td:first-child { width: 1.4rem; border-left: 1px solid transparent; border-radius: 8px 0 0 8px; }
.jl-concept-table td:last-child { border-right: 1px solid transparent; border-radius: 0 8px 8px 0; }
.jl-concept-table tbody tr:hover td { background: color-mix(in srgb, var(--viz-grid) 45%, transparent); }
#jacobian-lens tr[data-hl] td { --jl-edge: color-mix(in srgb, var(--jl-hue) 60%, transparent); background: color-mix(in srgb, var(--jl-hue) 13%, transparent); border-top-color: var(--jl-edge); border-bottom-color: var(--jl-edge); }
#jacobian-lens tr[data-hl] td:first-child { border-left-color: var(--jl-edge); }
#jacobian-lens tr[data-hl] td:last-child { border-right-color: var(--jl-edge); }
.jl-concept-table code, .jl-detail code { background: none; padding: 0; font-family: var(--jl-mono); font-size: 0.82rem; }
.jl-space { color: var(--viz-muted); opacity: 0.55; }
.jl-pick { appearance: auto; width: 14px; height: 14px; margin: 0; border: none; background: none; cursor: pointer; vertical-align: middle; accent-color: var(--jl-hue, var(--viz-line)); }
.jl-count { width: 4rem; color: var(--viz-muted); text-align: right; font-family: var(--jl-mono); font-size: 0.74rem; font-variant-numeric: tabular-nums; }
th.jl-count { font-family: inherit; }
.jl-axis div { display: flex; justify-content: space-between; gap: 0.5rem; }
.jl-axis span:first-child, .jl-axis span:last-child { font-weight: 500; opacity: 0.75; }
.jl-strip-cell { width: 55%; }
.jl-strip { display: flex; gap: 1px; width: 100%; min-width: 120px; height: 1.1rem; vertical-align: middle; border-radius: 3px; overflow: hidden; }
.jl-strip i { flex: 1 1 0; min-width: 1px; background: color-mix(in srgb, var(--jl-hue, var(--viz-ink)) calc(5% + var(--jl-level, 0) * 70%), transparent); }
.jl-detail { margin: 0.5rem 0; padding: 0.5rem 0.75rem; border: 1px solid var(--viz-grid); border-radius: 12px; max-height: 360px; overflow: auto; }
.jl-detail-head { display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; font-weight: 600; color: var(--viz-ink); }
.jl-detail-close { background: none; border: none; cursor: pointer; font-size: 1.1rem; line-height: 1; color: var(--viz-muted); }
.jl-detail-close:hover { color: var(--viz-ink); }
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
    first_block=None,
    last_block=None,
    block_bounds: tuple[int, int] | None = None,
):
    """Run the logit lens and attention readout for the clicked token.

    A Jacobian readout is drawn over the block range the reader chose for
    the one before it, clamped to its own fitted blocks the same way the
    sliders are; see :func:`carried_blocks`.

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
    images = list(context_state[4]) if len(context_state) > 4 else []
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
        # that produced them and only that load, or a later one of the same
        # weights, may explain them. The load, not the model ID:
        # re-downloading the same ID can bring in a newer snapshot. inspect()
        # compares it again under the model lock, which is where it is finally
        # decided; read under the claim, this one can no longer be overtaken by
        # a load starting behind it.
        load_id = runtime.MANAGER.load_for(load_id)
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
            if images:
                options["images"] = images
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
        # while it still came from this load. A steered pass keeps none.
        insight["load_id"] = load_id
        insight["images"] = images
        insight["steered"] = steering_vectors.active(steering)
        from chatlab.experiment_runs import SESSION_ID
        insight["saved_session"] = SESSION_ID
        frame = (
            render_lens(insight, *chosen_blocks(
                *carried_blocks(insight, first_block, last_block, block_bounds), fitted_blocks(insight),
            )),
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
    been released since says so and asks for another inspection. A steered
    readout never had one, and another inspection would keep none either.
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
    if insight.get("steered"):
        return (
            f'<div class="viz-empty">{html.escape(KV_CACHE_STEERED)}</div>',
            gr.skip(),
        )
    tokens = insight.get("tokens") or []
    try:
        view = runtime.MANAGER.read_kv_cache(
            [int(token["token_id"]) for token in tokens], int(layer or 1),
            load_id=insight.get("load_id"), images=insight.get("images") or (),
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


def render_lens(insight: dict, first=None, last=None) -> str:
    if insight.get("kind") == "jacobian":
        return charts.jacobian_lens_chart(insight, first, last)
    return charts.logit_lens_chart(insight)


def fitted_blocks(insight: dict | None) -> tuple[int, int] | None:
    """The first and last fitted block a Jacobian readout covers, numbered from 1."""
    if not insight or insight.get("kind") != "jacobian" or not insight.get("layers"):
        return None
    return insight["layers"][0]["layer"] + 1, insight["layers"][-1]["layer"] + 1


def chosen_blocks(first, last, bounds: tuple[int, int] | None) -> tuple[int | None, int | None]:
    """The block range to draw, with ``None`` at an end the reader left open.

    ``bounds`` are the fitted blocks of the readout the sliders were last set
    for. A slider left at its end stays open, so the next readout, or one
    through a lens fitted at other blocks, is shown in full at that end; a
    range the reader narrowed carries over to the next token they inspect.
    """
    if not bounds:
        return None, None
    low, high = bounds
    first = None if first is None or int(first) <= low else int(first)
    last = None if last is None or int(last) >= high else int(last)
    return first, last


def carried_blocks(insight: dict | None, first, last, bounds) -> tuple[int | None, int | None]:
    """The chosen block range clamped to a new Jacobian readout's fitted blocks.

    Open ends take the fitted ends. A carried range that no longer covers a
    fitted block, or that clamps to first after last, falls back to every
    fitted block. Answers ``(None, None)`` for a readout with no fitted blocks.
    """
    fitted = fitted_blocks(insight)
    if fitted is None:
        return None, None
    low, high = fitted
    chosen_first, chosen_last = chosen_blocks(first, last, bounds)
    first = low if chosen_first is None else min(max(chosen_first, low), high)
    last = high if chosen_last is None else min(max(chosen_last, low), high)
    if first > last or not any(first <= row["layer"] + 1 <= last for row in insight["layers"]):
        return low, high
    return first, last


def show_block_range(insight: dict | None, first, last, bounds):
    """Fit the block sliders to a new Jacobian readout, keeping a narrowed range.

    Answers the slider row, both sliders and the bounds they now describe.
    A lens fitted at a single block has no range to choose, so the row hides
    and both sliders sit on that block. Both ends are then open, so the next
    readout shows every block it was fitted at.
    """
    fitted = fitted_blocks(insight)
    if fitted is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    low, high = fitted
    first, last = carried_blocks(insight, first, last, bounds)
    if high <= low:
        return (
            gr.update(visible=False),
            gr.update(minimum=low, maximum=low, value=low),
            gr.update(minimum=low, maximum=low, value=low),
            fitted,
        )
    return (
        gr.update(visible=True),
        gr.update(minimum=low, maximum=high, value=first),
        gr.update(minimum=low, maximum=high, value=last),
        fitted,
    )


def render_block_range(insight: dict | None, first, last, bounds):
    """Redraw the Jacobian readout over the chosen blocks, without a new pass."""
    if fitted_blocks(insight) is None:
        return gr.skip()
    controls = insight.get("inspection_controls")
    if controls and not INSPECTION_CONTROLS.current(controls["session"], controls["revision"]):
        return gr.skip()
    return render_lens(insight, *chosen_blocks(first, last, bounds))


def all_blocks(insight: dict | None, bounds):
    """Put both sliders back at the ends and redraw every fitted block."""
    if not bounds:
        return gr.skip(), gr.skip(), gr.skip()
    low, high = bounds
    return low, high, render_block_range(insight, low, high, bounds)


def recall_lens() -> tuple[str | None, str | None]:
    """Import the lens written down for the loaded model; ``(name, note)``.

    See :meth:`ModelManager.recall_jacobian_lens`.
    """
    return runtime.MANAGER.recall_jacobian_lens()


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
        # to 8 GiB can take minutes, and holding the slot for it would block
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
        f"Imported {'an' if imported['kind'].startswith('R') else 'a'} {imported['kind']} for {code_span(str(imported['model_id']))}: "
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
