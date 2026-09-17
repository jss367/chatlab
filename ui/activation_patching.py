"""Compare runs by transplanting one residual activation at a time."""

from __future__ import annotations

import contextlib
import html
import json
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

from model_runtime import LOADING
from trace_export import write_private_text
from ui import runtime
from ui.common import failure_status


HINT = "Fill A and B under the same model load, then select an answer token to measure."
HEADERS = ["Layer (1-based)", "Source position", "Source token", "Recipient position",
           "Recipient token", "Patched probability (%)", "Change (percentage points)"]


class Controls:
    """Invalidate queued requests and in-flight frames when inputs change."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def new(self):
        session = uuid4().hex
        with self.lock:
            self.sessions[session] = 0
        return session

    def forget(self, session):
        with self.lock:
            self.sessions.pop(session, None)

    def change(self, session):
        with self.lock:
            if session not in self.sessions:
                return -1
            self.sessions[session] += 1
            return self.sessions[session]

    def current(self, session, revision):
        with self.lock:
            return session in self.sessions and self.sessions[session] == revision


CONTROLS = Controls()


def slots(left, right, direction):
    return (right, left) if direction == "B → A" else (left, right)


def reset(session):
    revision = CONTROLS.change(session)
    return (HINT, "", [], None, gr.update(interactive=True), gr.update(visible=False), revision)


def sources_changed(left, right, direction, session):
    _donor, recipient = slots(left, right, direction)
    choices = [
        (f"{i + 1}: {repr(m.get('text') or m.get('display_text') or '')} · ID {m['token_id']}", i)
        for i, m in enumerate((recipient or {}).get("metrics", []))
    ]
    return gr.update(choices=choices, value=0 if choices else None), *reset(session)


def heatmap(result):
    """A diverging, keyboard-focusable table with exact values on each cell."""
    pairs, cells = result["pairs"], result["cells"]
    by_position = {(cell["layer"], cell["column"]): cell for cell in cells}
    scale = max([abs(cell["delta_probability"]) for cell in cells] + [0.0001])
    parts = [
        '<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:3px;width:100%" '
        'aria-label="Change in answer probability after residual activation replacement">',
        '<thead><tr><th scope="col">Layer</th>',
    ]
    for pair in pairs:
        label = f"{pair['recipient_position'] + 1}: {pair['recipient_text']!r}"
        parts.append(f'<th scope="col" style="min-width:64px;max-width:120px;overflow-wrap:anywhere;font-size:11px">{html.escape(label)}</th>')
    parts.append("</tr></thead><tbody>")
    for layer in range(result["layer_count"]):
        parts.append(f'<tr><th scope="row">{layer + 1}</th>')
        for column, pair in enumerate(pairs):
            cell = by_position.get((layer, column))
            if cell is None:
                parts.append('<td style="text-align:center;opacity:.45" aria-label="Not measured">—</td>')
                continue
            delta = cell["delta_probability"]
            color = "45,110,210" if delta >= 0 else "220,115,35"
            opacity = 0.08 + 0.55 * abs(delta) / scale
            description = (
                f"Layer {layer + 1}; source {pair['donor_position'] + 1} {pair['donor_text']!r} → "
                f"recipient {pair['recipient_position'] + 1} {pair['recipient_text']!r}; "
                f"probability {cell['probability']:.8%}; change {100 * delta:+.6f} percentage points"
            )
            escaped = html.escape(description, quote=True)
            parts.append(
                f'<td tabindex="0" title="{escaped}" aria-label="{escaped}" '
                f'style="text-align:center;border-radius:4px;padding:7px 4px;font-size:11px;'
                f'background:rgba({color},{opacity:.3f})">{100 * delta:+.3f}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def rows(result):
    return [
        [cell["layer"] + 1, pair["donor_position"] + 1, repr(pair["donor_text"]),
         pair["recipient_position"] + 1, repr(pair["recipient_text"]),
         cell["probability"] * 100, cell["delta_probability"] * 100]
        for cell in result["cells"] for pair in [result["pairs"][cell["column"]]]
    ]


def run(left, right, direction, target, donor_count, width, session, revision):
    def cleared():
        # An old frame can arrive after the input-change callback's reset.
        # The reservation prevents a newer experiment publishing before this
        # cleanup. Preserve the reset/Stop message itself.
        return (gr.skip(), "", [], None, gr.update(interactive=True), gr.update(visible=False))

    if not CONTROLS.current(session, revision):
        yield (gr.skip(),) * 6
        return
    manager = runtime.MANAGER
    held = manager.claim_generation()
    if held:
        yield ("Wait for the model to finish loading." if held == LOADING else
               "Wait for the current model operation to finish.", *(gr.skip(),) * 5)
        return
    started = time.monotonic()
    try:
        yield ("Reading source activations and the unpatched answer probability…", "", [], None,
               gr.update(interactive=False), gr.update(visible=True))
        if not CONTROLS.current(session, revision):
            yield cleared()
            return
        donor, recipient = slots(left, right, direction)
        result = None
        last_frame = 0
        with contextlib.closing(manager.patch_activations(donor, recipient, target, donor_count, width)) as stream:
            for plan, reading in stream:
                if not CONTROLS.current(session, revision):
                    yield cleared()
                    return
                if "baseline" in reading:
                    result = {**plan, **reading, "cells": [], "complete": False}
                else:
                    result["cells"].append(reading)
                total = result["layer_count"] * len(result["pairs"])
                done = len(result["cells"])
                result["complete"] = done == total
                result["seconds"] = time.monotonic() - started
                if not result["complete"] and done and time.monotonic() - last_frame < 0.25:
                    continue
                last_frame = time.monotonic()
                status = (
                    f"**{direction} · answer token {target + 1}:** "
                    f"<code>{html.escape(repr(result['target_text']))}</code> · "
                    f"Unpatched probability **{result['baseline']['probability']:.6%}** · "
                    f"{done:,}/{total:,} interventions · {result['seconds']:.1f}s"
                )
                if result["complete"]:
                    status += " · Complete"
                # Yield immutable snapshots: later progress must not change an
                # earlier frame or a download already being serialized.
                snapshot = {**result, "cells": list(result["cells"])}
                yield (status, heatmap(snapshot), rows(snapshot), snapshot,
                       gr.update(interactive=result["complete"]),
                       gr.update(visible=not result["complete"]))
                if not CONTROLS.current(session, revision):
                    yield cleared()
                    return
    except Exception as error:
        if CONTROLS.current(session, revision):
            yield (failure_status("Could not patch activations", str(error)), "", [], None,
                   gr.update(interactive=True), gr.update(visible=False))
    finally:
        manager.release_generation()


def stop(session):
    _hint, *values = reset(session)
    return "Stopped. Change the inputs or run the experiment again.", *values


def download(result):
    if not result:
        return None
    directory = Path(tempfile.mkdtemp(prefix="chatlab-"))
    path = directory / f"chatlab-activation-patching-{uuid4().hex[:8]}.json"
    write_private_text(path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return str(path)


def build(left, right):
    with gr.Accordion("Activation patching", open=False, elem_id="activation-patching"):
        gr.Markdown(
            "Transplant residual activations between runs and measure one answer token's "
            "**raw next-token probability**, before sampling. Each heatmap cell replaces "
            "one token's activation after one decoder block in an otherwise unchanged recipient pass. "
            "Fill both slots with steering off under the same Transformers model load "
            "(Llama, Qwen2 or OLMo 3). You can change the prompt between A and B."
        )
        session = gr.State(value=lambda: CONTROLS.new(), delete_callback=CONTROLS.forget)
        revision = gr.State(0)
        result = gr.State(None)
        direction = gr.Radio(["A → B", "B → A"], value="A → B", label="Source → recipient")
        target = gr.Dropdown(choices=[], label="Answer token in recipient", elem_id="patch-target")
        with gr.Row():
            donor_count = gr.Number(value=0, precision=0, minimum=0, label="Source output tokens to include",
                                    info="0 uses only the source prompt. N includes its first N output tokens.")
            width = gr.Slider(1, 32, value=8, step=1, label="Token pairs to patch")
        gr.Markdown(
            "The recipient prefix ends **before** the selected answer token. The last N tokens "
            "of each prefix are paired by distance from the end; this is positional pairing, "
            "not semantic alignment. Source output is included only when requested above. "
            "Prefixes are limited to 2,048 tokens and are never silently shortened. "
            "Each pair costs one forward pass per layer."
        )
        with gr.Row():
            start = gr.Button("Patch activations", variant="primary", elem_id="run-patching")
            cancel = gr.Button("Stop", visible=False, variant="stop", elem_id="stop-patching")
        status = gr.Markdown(HINT)
        gr.Markdown(
            "**Heatmap:** rows are layers; columns are recipient token positions. "
            "Blue increases and orange decreases the answer probability, in percentage points. "
            "Hover or focus a cell for its source token and exact measurement. "
            "A dash means unmeasured. The color range is symmetric and scales to the largest measured effect."
        )
        chart = gr.HTML("")
        with gr.Accordion("Exact measurements and token pairing", open=False):
            table = gr.Dataframe(headers=HEADERS, interactive=False, wrap=True,
                                 datatype=["number", "number", "str", "number", "str", "number", "number"])
        gr.DownloadButton("Download patching JSON", value=download, inputs=result, size="sm")
        gr.Markdown(
            "This measures the effect of a specific intervention on one token, not the probability "
            "of a whole answer or proof of a complete reasoning mechanism. Cells are independent; "
            "attention-head and sequential interventions are not included in this first version."
        )
        outputs = [status, chart, table, result, start, cancel]
        running = start.click(run, [left, right, direction, target, donor_count, width, session, revision], outputs)
        cancel.click(stop, session, [*outputs, revision], cancels=[running], queue=False)
        for source in (left, right, direction):
            source.change(sources_changed, [left, right, direction, session],
                          [target, *outputs, revision], queue=False)
        for control in (target, donor_count, width):
            control.input(reset, session, [*outputs, revision], queue=False)
