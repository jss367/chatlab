"""The Reasoning check tab: whether responses do what they say, across one-agent and team runs."""
from __future__ import annotations

import html
import json
import logging
import re
import tempfile
from pathlib import Path

import gradio as gr

from .reasoning_check import (FOLLOW_WINDOW, FRACTIONS, RESPONSE_HEADERS, SUMMARY_HEADERS, TRUNCATION_HEADERS,
                              TRUNCATION_SUMMARY_HEADERS, TruncationControl, condition_of, csv_text, load_run,
                              read_responses, response_rows, run_label, summary_rows, truncation_rows,
                              truncation_summary, truncation_test)

logger = logging.getLogger(__name__)

CSS = """
#reasoning-workspace {flex:1 1 0 !important; min-height:0; flex-wrap:nowrap; gap:20px;}
#reasoning-inputs {flex:0 0 auto !important; width:28%; min-width:260px !important; max-width:34%; height:100%; overflow:auto; padding:0 12px 16px 0; scrollbar-width:thin;}
#reasoning-results {flex:1 1 0 !important; min-width:320px !important; height:100%; overflow:auto; padding:0 0 16px 12px; border-left:1px solid var(--border-color-primary); scrollbar-width:thin;}
#reasoning-results td, #reasoning-results th {font:12px/1.5 system-ui;}
#reasoning-inputs .block, #reasoning-results .block {min-width:0 !important;}
@media(max-width:850px) {
  #reasoning-workspace {flex:none !important; flex-wrap:wrap;}
  #reasoning-inputs, #reasoning-results {flex:1 1 100% !important; width:auto; max-width:none; height:auto; border:0; padding:0;}
}
"""


def parse_responses(text, count):
    """The response numbers named by text such as "1, 3, 5-8", or None for every response."""
    text = str(text or "").strip()
    if text.lower() in ("", "all"):
        return None
    chosen = set()
    for part in text.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise ValueError("Name the responses by number, such as 1, 3, 5-8, or leave the box blank for all.")
        low, high = int(match[1]), int(match[2] or match[1])
        if not 1 <= low <= high <= count:
            raise ValueError(f"This run's responses are numbered 1 to {count}.")
        chosen.update(range(low, high + 1))
    return chosen


def write_csv(prefix, headers, rows):
    directory = Path(tempfile.mkdtemp(prefix=prefix))
    path = directory / f"{prefix.rstrip('-')}.csv"
    path.write_text(csv_text(headers, rows), encoding="utf-8")
    return str(path)


def runs_note(runs):
    if not runs:
        return "Load saved runs from the One agent or Team tab. Every response that made a move call is scored."
    lines = []
    for ep in runs.values():
        scored = len(read_responses(ep))
        lines.append(f"- **{run_label(ep)}** · {html.escape(condition_of(ep))} · {len(ep.turns)} responses, "
                     f"{scored} with a readable move call · {html.escape(ep.model_id or 'model not recorded')}")
    return "\n".join(lines)


def run_choices(runs):
    return [(f"{run_label(ep)} · {condition_of(ep)}", run_id) for run_id, ep in runs.items()]


def build_reasoning_page(context):
    runs = gr.State({})
    results = gr.State([])
    control = gr.State(TruncationControl())
    wanted_model = gr.State("")
    with gr.Row(elem_id="reasoning-workspace"):
        with gr.Column(elem_id="reasoning-inputs"):
            gr.Markdown("## Reasoning check")
            gr.Markdown("Compare what responses say with what they do, for one agent and for teams. Load saved runs "
                        "made under each condition you want to compare.")
            upload = gr.File(label="Saved runs JSON", file_count="multiple", file_types=[".json"], type="filepath",
                             elem_id="reasoning-upload")
            clear = gr.Button("Clear loaded runs", size="sm")
            loaded = gr.Markdown(runs_note({}), elem_id="reasoning-loaded")
            gr.Markdown("### Truncation test")
            gr.Markdown("Cuts each response after " + ", ".join(f"{f:.0%}" for f in FRACTIONS) + " of its reasoning's "
                        "words, makes the model write its move call right after the cut, and reads the probability "
                        "it gives the move the response made. Needs the model that made the run.")
            models = gr.Button("Choose / load model", size="sm")
            run_pick = gr.Dropdown(choices=[], label="Run", elem_id="reasoning-run")
            numbers = gr.Textbox(value="all", label="Responses", info="all, or response numbers and ranges such as "
                                                                      "1, 3, 5-8, as numbered in Every response.")
            with gr.Row():
                start = gr.Button("Run the truncation test", variant="primary", size="sm", elem_id="reasoning-start")
                stop = gr.Button("Stop", size="sm", visible=False, elem_id="reasoning-stop")
            progress = gr.Markdown("", elem_id="reasoning-progress")
        with gr.Column(elem_id="reasoning-results"):
            gr.Markdown("## Stated against taken\nA response states a direction when its reasoning commits to one, "
                        "such as \"I will move east\" or \"go to (1, 2)\". Hedged, ruled-out and later steps are not "
                        "commitments. The last one before the call is compared with the call. A message is kept when "
                        f"its agent calls the direction it names within {FOLLOW_WINDOW} of its own calls, accepted or not.")
            summary = gr.Dataframe(headers=SUMMARY_HEADERS, value=[], interactive=False, wrap=True,
                                   elem_id="reasoning-summary")
            with gr.Accordion("Every response", open=False):
                responses = gr.Dataframe(headers=RESPONSE_HEADERS, value=[], interactive=False, wrap=True,
                                         elem_id="reasoning-responses")
                responses_csv = gr.File(label="Responses CSV", interactive=False)
            gr.Markdown("## Truncation test\nMean probability of the move each response made, by how much of its "
                        "reasoning was kept. Reasoning the move depends on rises from left to right. \"Same move "
                        "with none kept\" is how often the model makes that move with no reasoning at all.")
            truncation = gr.Dataframe(headers=TRUNCATION_SUMMARY_HEADERS, value=[], interactive=False, wrap=True,
                                      elem_id="reasoning-truncation")
            with gr.Accordion("Every truncated response", open=False):
                truncated = gr.Dataframe(headers=TRUNCATION_HEADERS, value=[], interactive=False, wrap=True,
                                         elem_id="reasoning-truncated")
                truncated_csv = gr.File(label="Truncation CSV", interactive=False)

    def reasoning_scored(held):
        rows = [row for ep in held.values() for row in read_responses(ep)]
        table = response_rows(rows)
        path = write_csv("chatlab-reasoning-responses-", RESPONSE_HEADERS, table) if table else None
        choices = run_choices(held)
        return (held, runs_note(held), summary_rows(rows), table, path,
                gr.update(choices=choices, value=choices[0][1] if choices else None))

    def reasoning_load(paths, held, done_before):
        held = dict(held)
        replaced = set()
        for path in paths or ():
            try:
                if Path(path).stat().st_size > 50_000_000:
                    raise ValueError("Run files must be smaller than 50 MB.")
                ep = load_run(json.loads(Path(path).read_text()))
            except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
                logger.warning("Reasoning check could not load %s: %s", path, exc)
                gr.Warning(f"Could not load {Path(path).name}: {exc}")
                continue
            if ep.run_id in held:
                replaced.add(run_label(ep))
            held[ep.run_id] = ep
            logger.info("Reasoning check loaded run %s (%s) from %s", ep.run_id, condition_of(ep), path)
        # A run loaded again may be a later export, so what was read from the
        # file it replaces no longer describes it.
        kept = [t for t in done_before if t.response.run not in replaced]
        if len(kept) == len(done_before):
            return (*reasoning_scored(held), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        table = truncation_rows(kept)
        path = write_csv("chatlab-reasoning-truncation-", TRUNCATION_HEADERS, table) if table else None
        return (*reasoning_scored(held), kept, truncation_summary(kept), table, path)

    def reasoning_clear(ctl):
        if ctl.running:
            raise gr.Error("Stop the truncation test before clearing the runs.")
        return (*reasoning_scored({}), [], [], [], None, "")

    def reasoning_truncate(held, run_id, text, done_before, ctl):
        if ctl.running:
            raise gr.Error("A truncation test is already running.")
        ep = held.get(run_id)
        if ep is None:
            raise gr.Error("Load a run and choose it first.")
        try:
            indices = parse_responses(text, len(ep.turns))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        # A second test of the same run replaces its first.
        kept = [t for t in done_before if t.response.run != run_label(ep)]
        buttons = (gr.update(visible=False), gr.update(visible=True))
        found, failure = [], None
        try:
            for done, total, found in truncation_test(ep, context.models, ctl, indices):
                note = f"**Running** · {done} of {total} responses of {run_label(ep)}"
                every = kept + found
                yield (every, note, truncation_summary(every), truncation_rows(every), gr.skip(), *buttons)
        except ValueError as exc:
            logger.warning("Truncation test on run %s refused: %s", ep.run_id, exc)
            gr.Warning(str(exc))
            failure = exc
        every = kept + found
        table = truncation_rows(every)
        path = write_csv("chatlab-reasoning-truncation-", TRUNCATION_HEADERS, table) if table else None
        ended = "Refused" if failure else "Stopped" if ctl.stop_requested else "Finished"
        note = f"**{ended}** · {len(found)} responses of {run_label(ep)} read"
        if failure:
            note += f" · {html.escape(str(failure))}"
        yield (every, note, truncation_summary(every), table, path, gr.update(visible=True), gr.update(visible=False))

    def reasoning_pick(held, run_id):
        """Name the model the chosen run needs on the button that opens Models."""
        wanted = getattr(held.get(run_id), "model_id", None)
        return gr.update(value=f"Load {wanted}" if wanted else "Choose / load model"), wanted or ""

    def reasoning_stop(ctl):
        if ctl.running:
            ctl.request_stop()
            gr.Info("Stopping after the response being read.")

    loaded_outputs = [runs, loaded, summary, responses, responses_csv, run_pick]
    upload.upload(reasoning_load, [upload, runs, results],
                  [*loaded_outputs, results, truncation, truncated, truncated_csv], show_progress="hidden")
    clear.click(reasoning_clear, control, [*loaded_outputs, results, truncation, truncated, truncated_csv, progress],
                show_progress="hidden")
    start.click(reasoning_truncate, [runs, run_pick, numbers, results, control],
                [results, progress, truncation, truncated, truncated_csv, start, stop], show_progress="hidden")
    stop.click(reasoning_stop, control, None, queue=False)
    run_pick.change(reasoning_pick, [runs, run_pick], [models, wanted_model], queue=False)
    context.navigation.open_models(models, wanted_model)
