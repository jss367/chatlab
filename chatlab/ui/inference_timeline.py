"""Offline replay and bounded, explicitly recomputed layer readouts."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import gradio as gr

from chatlab import charts
from chatlab import experiment_runs as runs
from chatlab.token_metrics import unscored_metric
from chatlab.ui import runtime
from chatlab.ui.common import failure_status
from chatlab.ui.inspection import render_lens, render_attention
from chatlab.ui.panel import describe_token, event_index


COLORS = {"Prompt / context": "#9fc8f8", "Response / passage": "#c3c2b7", "Selected": "#f5c451"}
EMPTY_LENS = '<div class="viz-empty">No saved layer readout. Use Capture layer readouts to inspect this position.</div>'


def tokens(document):
    run = (document or {}).get("run", {})
    recorded = run.get("prompt_metrics", [])
    prompt = []
    for index, token_id in enumerate(run.get("context_ids", [])):
        if index < len(recorded) and recorded[index].get("token_id") == token_id:
            prompt.append(recorded[index])
        else:
            prompt.append(unscored_metric(
                position=index + 1, token_id=token_id, token_text=f"[ID {token_id}]",
                fallback_text="", reason="Prompt text and measurements were not retained for this run.",
            ).to_dict())
    return prompt + run.get("metrics", [])


def readout(document, index, mode):
    record = (document or {}).get("timeline_inspections", {}).get(f"{mode.lower()}:{index}")
    if record:
        return record["insight"]
    # Older, manually attached inspections remain useful in the timeline.
    offset = len((document or {}).get("run", {}).get("context_ids", []))
    for saved in reversed((document or {}).get("inspections", [])):
        insight = saved["insight"]
        if saved["token_index"] + offset == index and insight.get("kind", "logit") == mode.lower():
            return insight
    return None


def frame(document, number, mode, layer):
    metrics = tokens(document)
    if not metrics:
        return (gr.update(value=1, maximum=1), [], "Select a saved experiment.", [],
                EMPTY_LENS, charts.EMPTY_ATTENTION, "No timeline loaded.")
    index = min(max(int(number or 1) - 1, 0), len(metrics) - 1)
    offset = len(document["run"].get("context_ids", []))
    # A local window keeps playback cheap for long reasoning traces. The
    # slider still addresses the complete sequence, including hidden tokens.
    start, end = max(0, index - 24), min(len(metrics), index + 25)
    strip = [(m["display_text"], "Selected" if i == index else
              "Prompt / context" if i < offset else "Response / passage")
             for i, m in enumerate(metrics[start:end], start)]
    detail, candidates = describe_token(metrics[index])
    insight = readout(document, index, mode)
    source = "Prompt / context" if index < offset else "Response / passage"
    status = f"{source} · position {index + 1:,} / {len(metrics):,} · showing {start + 1:,}–{end:,}. "
    if insight:
        status += "Layers: recomputed after the run; saved for offline replay."
        lens = render_lens(insight)
        attention = render_attention(insight, layer) if insight.get("attention") or mode == "Jacobian" else (
            '<div class="viz-empty">Attention was not retained or is unavailable at this position.</div>')
    else:
        status += "No saved layer readout at this position."
        lens, attention = EMPTY_LENS, charts.EMPTY_ATTENTION
    return gr.update(value=index + 1, maximum=max(1, len(metrics))), strip, detail, candidates, lens, attention, status


def move(document, expected, number, mode, layer, playing=False, *, delta=0):
    if not document or document["id"] != expected:
        return (*frame(None, 1, mode, layer), False, gr.update(active=False))
    total = len(tokens(document))
    number = min(max(int(number or 1) + delta, 1), max(1, total))
    playing = bool(playing and number < total)
    return (*frame(document, number, mode, layer), playing, gr.update(active=playing))


def clicked(document, expected, number, mode, layer, event: gr.SelectData):
    index = max(0, int(number or 1) - 1 - 24) + event_index(event)
    return move(document, expected, index + 1, mode, layer)


def opened(document, expected, previous, number, mode, layer):
    identifier = (document or {}).get("id")
    return identifier, *move(document, expected, number if identifier == previous else 1, mode, layer)


def capture(document, expected, first, last, stride, mode, pin, keep_attention):
    """Persist each completed position so Stop keeps work already finished."""
    skip = gr.skip()
    if not document or document["id"] != expected:
        yield skip, "Select a saved experiment and wait for it to open."
        return
    occupied = runtime.MANAGER.claim_generation()
    if occupied:
        yield skip, f"Wait until the model is free; it is {occupied}."
        return
    try:
        document = runs.read(expected)
        run = document["run"]
        if (not runtime.MANAGER.loaded or run.get("load_id") != runtime.MANAGER.load_id
                or run.get("session_id") != runs.SESSION_ID):
            raise ValueError("Capture requires the original model load in this app session. Generate and save a new run; saved readouts still replay offline.")
        ids = run.get("context_ids", []) + [m["token_id"] for m in run["metrics"]]
        first, last, stride = (int(value) if value is not None else 1 for value in (first, last, stride))
        if not 1 <= first <= last <= len(ids) or stride < 1:
            raise ValueError(f"Choose a range from 1 to {len(ids)} and a positive step.")
        positions = list(range(first - 1, last, stride))
        if len(positions) > 32:
            raise ValueError("Capture at most 32 positions at once. Narrow the range or increase the step.")
        if mode == "Logit" and 0 in positions:
            raise ValueError("Position 1 has no preceding prediction. Start at position 2 for the Logit lens.")
        options = {"context_count": len(run.get("context_ids", [])),
                   "load_id": run["load_id"], "steering": run.get("settings", {}).get("steering")}
        if mode == "Jacobian" and runtime.MANAGER.jacobian_lens_import() is None:
            raise ValueError("Import a matching Jacobian lens in Layers and attention first.")
        yield skip, f"Recomputing {len(positions)} positions. Completed readouts are saved as they finish."
        for count, position in enumerate(positions, 1):
            if mode == "Jacobian":
                insight = runtime.MANAGER.inspect_jacobian(
                    ids, position, lens_id=None, pinned_text=pin or "", **options).to_dict()
            else:
                insight = runtime.MANAGER.inspect(ids, position, **options).to_dict()
            if not keep_attention:
                insight["attention"] = []
                insight["tokens"] = []
            document = runs.save_timeline_inspection(expected, position, insight)
            yield document, f"Saved {count} / {len(positions)} recomputed readouts."
    except (OSError, ValueError, RuntimeError) as error:
        yield skip, failure_status("Could not capture timeline", str(error))
    finally:
        runtime.MANAGER.release_generation()


def build():
    with gr.Accordion("Inference timeline", open=True):
        gr.Markdown("Replay prompt and response positions with synchronized token measurements, layers, and attention. "
                    "Playback uses saved data and token order, not wall-clock timing. "
                    "Thinking and answer tokens share the response track.")
        identity = gr.State(None)
        playing = gr.State(False)
        timer = gr.Timer(0.75, active=False)
        with gr.Row():
            previous = gr.Button("Previous position", size="sm")
            play = gr.Button("Play timeline", size="sm")
            pause = gr.Button("Pause timeline", size="sm")
            following = gr.Button("Next position", size="sm")
        position = gr.Slider(1, 1, value=1, step=1, label="Timeline position", interactive=True)
        with gr.Row():
            mode = gr.Radio(["Logit", "Jacobian"], value="Logit", label="Recorded lens")
            layer = gr.Number(value=0, minimum=0, precision=0, label="Attention layer (0 averages layers)")
        status = gr.Markdown("No timeline loaded.")
        strip = gr.HighlightedText(label="Timeline token window", combine_adjacent=False,
                                   color_map=COLORS, show_legend=True, show_inline_category=False)
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                detail = gr.Markdown("Select a saved experiment.")
                candidates = gr.Dataframe(headers=["Token ID", "Alternative", "Probability"],
                                          datatype=["number", "str", "number"], interactive=False)
            with gr.Column(scale=2, min_width=360):
                lens = gr.HTML(EMPTY_LENS)
                attention = gr.HTML(charts.EMPTY_ATTENTION)
        with gr.Accordion("Capture layer readouts", open=False):
            gr.Markdown("Recompute selected positions using the original loaded model and saved steering. "
                        "Logit reads the prediction before the token; Jacobian reads after the token. "
                        "Up to 32 positions per capture, 64 saved readouts and 32 MB per experiment. "
                        "Raw activations are not retained. Repeating a position replaces its readout for that lens.")
            with gr.Row():
                first = gr.Number(value=2, minimum=1, precision=0, label="First timeline position")
                last = gr.Number(value=2, minimum=1, precision=0, label="Last timeline position")
                stride = gr.Number(value=1, minimum=1, precision=0, label="Position step")
            pin = gr.Textbox(label="Pinned vocabulary token (Jacobian)")
            keep_attention = gr.Checkbox(value=False, label="Retain attention weights (larger files)")
            with gr.Row():
                record = gr.Button("Capture selected positions")
                stop = gr.Button("Stop capture")
            capture_status = gr.Markdown()
    return SimpleNamespace(**locals())


def wire(view, experiments):
    shared = [experiments.held, experiments.picker, view.position, view.mode, view.layer]
    outputs = [view.position, view.strip, view.detail, view.candidates, view.lens,
               view.attention, view.status, view.playing, view.timer]
    quiet = {"show_progress": "hidden", "show_progress_on": [],
             "concurrency_id": "inference_timeline_navigation"}
    ticking = view.timer.tick(
        lambda *args: move(*args[:-1], playing=args[-1], delta=1) if args[-1] else (gr.skip(),) * 9,
        [*shared, view.playing], outputs, **quiet)
    experiments.held.change(opened, [experiments.held, experiments.picker, view.identity,
                                    view.position, view.mode, view.layer], [view.identity, *outputs],
                            cancels=[ticking], **quiet)
    # A picker edit immediately stops playback while the new file is opening.
    experiments.picker.input(lambda: (False, gr.update(active=False)), None, [view.playing, view.timer],
                             cancels=[ticking], **quiet)
    view.position.input(move, shared, outputs, cancels=[ticking], **quiet)
    for control in (view.mode, view.layer):
        control.input(move, shared, outputs, cancels=[ticking], **quiet)
    view.strip.select(clicked, shared, outputs, cancels=[ticking], **quiet)
    for button, delta in ((view.previous, -1), (view.following, 1)):
        button.click(partial(move, delta=delta), shared, outputs, cancels=[ticking], **quiet)
    view.play.click(lambda *args: move(*args, playing=True), shared, outputs, **quiet)
    view.pause.click(move, shared, outputs, cancels=[ticking], **quiet)
    recording = view.record.click(capture, [experiments.held, experiments.picker, view.first, view.last,
                                            view.stride, view.mode, view.pin, view.keep_attention],
                                  [experiments.held, view.capture_status])
    view.stop.click(lambda: "Capture stopped. Completed readouts remain saved.", None,
                    view.capture_status, cancels=[recording])
    experiments.picker.change(None, cancels=[recording])
