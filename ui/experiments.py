"""Saved experiments, offline token inspection, bookmarks, and rerunning."""

from __future__ import annotations

import contextlib
from functools import partial
from types import SimpleNamespace

import gradio as gr

import charts
import experiment_runs as runs
from token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE, category_for
from ui.common import failure_status
from ui.inspection import render_lens
from ui.panel import as_plain_text, describe_token, event_index
from ui import runtime


def choices(query="", selected=None):
    items = runs.search(query)
    return gr.update(choices=[(f"{item['title']} · {item['run'].get('model_id', '')} · {item['created_at']}", item["id"])
                              for item in items], value=selected if any(item["id"] == selected for item in items) else None)


def refresh_choices(query=""):
    # A refresh can finish after the reader picks a run from the existing
    # list. Update its choices without clearing that newer selection.
    update = choices(query)
    update.pop("value", None)
    return update


def refresh_library(query=""):
    update = refresh_choices(query)
    return update, f"Matching saved experiments: {len(update['choices'])}."


def build():
    with gr.Tab("Experiments", elem_id="experiments-tab"):
        gr.Markdown("Save a run to revisit its measurements with no model loaded. "
                    "Bookmark interesting tokens, add notes, or send two saved runs to Compare.")
        held = gr.State(None)
        selected = gr.State(None)
        with gr.Accordion("Save a run", open=False):
            with gr.Row():
                title = gr.Textbox(label="Experiment name", placeholder="What are you investigating?")
                source = gr.Dropdown(["Latest chat response", "Comparison A", "Comparison B"],
                                     value="Latest chat response", label="Save from")
            save = gr.Button("Save experiment")
        with gr.Row():
            query = gr.Textbox(label="Search experiments", placeholder="Name, model, prompt, response, or note", scale=4)
            refresh = gr.Button("Refresh saved experiments", size="sm", scale=1)
        picker = gr.Dropdown([], label="Saved experiments", elem_id="saved-experiments", filterable=False)
        status = gr.Markdown("Save a response or run a comparison to start your library.")
        with gr.Row():
            send_a = gr.Button("Use as comparison A")
            send_b = gr.Button("Use as comparison B")
            rerun = gr.Button("Rerun with loaded model")
            stop = gr.Button("Stop rerun", visible=False, variant="stop")
        gr.Markdown("Rerun uses the saved inputs and settings with the matching model already loaded. "
                    "Its result is saved as a new experiment; hardware and model revisions can affect reproducibility.")
        text = gr.Textbox(label="Recorded response or passage", interactive=False, lines=3)
        with gr.Accordion("Recorded configuration", open=False):
            configuration = gr.JSON(label="Configuration")
        scale = gr.Dropdown(list(COLOR_SCALES), value=DEFAULT_COLOR_SCALE, label="Color saved tokens by")
        strip = gr.HighlightedText(label="Saved tokens", combine_adjacent=False,
                                   color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                                   show_legend=True, elem_id="saved-token-strip")
        with gr.Row():
            mode = gr.Dropdown(["Highest surprise", "Closest alternatives", "Bookmarks"],
                               value="Highest surprise", label="Find tokens")
            next_token = gr.Button("Next matching token")
        with gr.Row():
            position = gr.Number(value=1, precision=0, minimum=1, label="Token number", scale=4)
            go = gr.Button("Go to token", size="sm", scale=1)
        detail = gr.Markdown("Select a saved token to inspect it.")
        candidates = gr.Dataframe(headers=["Token ID", "Alternative", "Probability"],
                                   datatype=["number", "str", "number"], interactive=False)
        note = gr.Textbox(label="Bookmark note", lines=2)
        with gr.Row():
            mark = gr.Button("Save bookmark")
            unmark = gr.Button("Remove bookmark")
        marks = gr.Dataframe(headers=["Token", "Text", "Note"], interactive=False,
                            datatype=["number", "str", "str"], label="Bookmarks")
        with gr.Accordion("Saved layer inspections", open=False):
            attach = gr.Button("Save current response inspection")
            inspection_picker = gr.Dropdown([], label="Recorded inspection")
            lens = gr.HTML(charts.EMPTY_LENS)
            layer = gr.Number(value=0, minimum=0, precision=0, label="Attention layer")
            attention = gr.HTML(charts.EMPTY_ATTENTION)
    return SimpleNamespace(**locals())


def inspector_visibility(event: gr.SelectData):
    visible = event.value != "Experiments"
    return gr.update(visible=visible), gr.update(visible=visible)


def token_strip(document, scale, selected=None):
    colors = COLOR_SCALES[scale].color_map
    if selected is not None:
        colors = colors | {"Selected token": "#f5c451"}
    return gr.update(value=[(m["display_text"], "Selected token" if i == selected else category_for(m, scale))
                            for i, m in enumerate((document or {}).get("run", {}).get("metrics", []))],
                     color_map=colors)


def bookmark_rows(document):
    metrics = document["run"]["metrics"]
    return [[int(key) + 1, metrics[int(key)]["text"], note]
            for key, note in sorted(document.get("bookmarks", {}).items(), key=lambda pair: int(pair[0]))]


def open_run(identifier, scale):
    if not identifier:
        return (None, None, "Select a saved experiment.", "", {}, token_strip(None, scale),
                [], "Select a saved token to inspect it.", [], "", 1, gr.update(choices=[], value=None),
                charts.EMPTY_LENS, charts.EMPTY_ATTENTION)
    try:
        item = runs.read(identifier)
        run = item["run"]
        configuration = {key: value for key, value in run.items()
                         if key not in ("metrics", "decoded", "token_ends", "context_ids", "text",
                                        "session_id", "metrics_generation", "load_id")}
        inspections = item.get("inspections", [])
        return (item, None, f"Opened **{as_plain_text(item['title'])}** · {len(run['metrics']):,} tokens.",
                run["text"], configuration, token_strip(item, scale), bookmark_rows(item),
                "Select a saved token to inspect it.", [], "", 1,
                gr.update(choices=[(f"Token {record['token_index'] + 1} · inspection {index + 1}", index)
                                   for index, record in enumerate(inspections)], value=None),
                charts.EMPTY_LENS, charts.EMPTY_ATTENTION)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return (None, None, failure_status("Could not open experiment", str(error)),
                "", {}, token_strip(None, scale), [], "", [], "", 1,
                gr.update(choices=[], value=None), charts.EMPTY_LENS, charts.EMPTY_ATTENTION)


def save_run(title, source, trace, context, left, right):
    try:
        run = runs.from_trace(trace, context) if source == "Latest chat response" else (
            left if source == "Comparison A" else right)
        item = runs.save(run, title)
        return choices(selected=item["id"]), ""
    except (OSError, ValueError, TypeError) as error:
        return gr.skip(), failure_status("Could not save experiment", str(error))


def select_token(document, index, scale=DEFAULT_COLOR_SCALE):
    if not document or index is None or not 0 <= int(index) < len(document["run"]["metrics"]):
        return None, "Choose a token in the saved run.", [], "", 1, token_strip(document, scale)
    index = int(index)
    metric = document["run"]["metrics"][index]
    detail, candidates = describe_token(metric)
    # Escape as Markdown prose here: entities inside a code span would show
    # their spelling instead of the quoted token's characters.
    detail = f"### Token {metric['position']}: {as_plain_text(repr(metric['text']))}\n" + detail.split("\n", 1)[1]
    return (index, detail, candidates, document.get("bookmarks", {}).get(str(index), ""),
            index + 1, token_strip(document, scale, index))


def clicked_token(document, scale, event: gr.SelectData):
    return select_token(document, event_index(event), scale)


def navigate(document, selected, mode, scale=DEFAULT_COLOR_SCALE):
    if not document:
        return select_token(None, None, scale)
    indexes = (sorted(int(key) for key in document.get("bookmarks", {})) if mode == "Bookmarks"
               else runs.ranked_tokens(document["run"], mode))
    if not indexes:
        return None, "No matching tokens in this experiment.", [], "", 1, token_strip(document, scale)
    offset = indexes.index(selected) + 1 if selected in indexes else 0
    return select_token(document, indexes[offset % len(indexes)], scale)


def annotate(document, selected, note, expected_identifier=None, *, remove=False):
    try:
        if not document or selected is None:
            raise ValueError("Select a saved token first.")
        if expected_identifier is not None and document["id"] != expected_identifier:
            raise ValueError("Wait for the selected experiment to open before adding a bookmark.")
        item = runs.bookmark(document["id"], int(selected), note, remove=remove)
        return item, bookmark_rows(item), "Bookmark removed." if remove else "Bookmark saved."
    except (OSError, ValueError) as error:
        return gr.skip(), gr.skip(), failure_status("Could not update bookmark", str(error))


def comparison_slot(document, expected_identifier=None):
    if document and expected_identifier is not None and document["id"] != expected_identifier:
        gr.Warning("Wait for the selected experiment to open before comparing it.")
        return gr.skip()
    return document["run"] if document else gr.skip()


def inspection_view(document, index, layer):
    if not document or index is None:
        return charts.EMPTY_LENS, charts.EMPTY_ATTENTION
    insight = document["inspections"][int(index)]["insight"]
    return render_lens(insight), charts.attention_strip(insight, int(layer or 0))


def attach_inspection(document, insight, target, context, expected_identifier):
    try:
        if not document or document["id"] != expected_identifier:
            raise ValueError("Wait for the selected experiment to open before saving an inspection.")
        item = runs.save_inspection((document or {}).get("id"), insight, target, context)
        return item, gr.update(choices=[(f"Token {record['token_index'] + 1} · inspection {i + 1}", i)
                                        for i, record in enumerate(item["inspections"])],
                               value=len(item["inspections"]) - 1), "Inspection saved."
    except (OSError, ValueError, TypeError) as error:
        return gr.skip(), gr.skip(), failure_status("Could not save inspection", str(error))


def wire(view, demo, trace, context, left, right, insight, target):
    opened = [view.held, view.selected, view.status, view.text, view.configuration, view.strip,
              view.marks, view.detail, view.candidates, view.note, view.position,
              view.inspection_picker, view.lens, view.attention]
    selection = [view.selected, view.detail, view.candidates, view.note, view.position, view.strip]
    demo.load(refresh_choices, None, view.picker)
    view.refresh.click(refresh_library, view.query, [view.picker, view.status])
    view.query.submit(choices, [view.query, view.picker], view.picker)
    view.picker.change(open_run, [view.picker, view.scale], opened)
    view.save.click(save_run, [view.title, view.source, trace, context, left, right],
                    [view.picker, view.status])
    view.scale.input(token_strip, [view.held, view.scale, view.selected], view.strip,
                     show_progress="hidden")
    view.strip.select(clicked_token, [view.held, view.scale], selection)
    view.next_token.click(navigate, [view.held, view.selected, view.mode, view.scale], selection)
    view.go.click(lambda document, number, scale: select_token(document, int(number or 1) - 1, scale),
                  [view.held, view.position, view.scale], selection)
    for button, remove in ((view.mark, False), (view.unmark, True)):
        button.click(partial(annotate, remove=remove), [view.held, view.selected, view.note, view.picker],
                     [view.held, view.marks, view.status])
    view.send_a.click(comparison_slot, [view.held, view.picker], left)
    view.send_b.click(comparison_slot, [view.held, view.picker], right)
    view.attach.click(attach_inspection, [view.held, insight, target, context, view.picker],
                      [view.held, view.inspection_picker, view.status])
    for control in (view.inspection_picker, view.layer):
        control.change(inspection_view, [view.held, view.inspection_picker, view.layer],
                        [view.lens, view.attention])
    rerunning = view.rerun.click(rerun_saved, [view.held, view.picker], [view.picker, view.status, view.rerun, view.stop])
    view.stop.click(lambda: ("Rerun stopped. The saved experiment is unchanged.",
                            gr.update(interactive=True), gr.update(visible=False)),
                    None, [view.status, view.rerun, view.stop], cancels=[rerunning])


def rerun_saved(document, expected_identifier=None):
    from ui.compare import _measure_text, _tokenizer_identity, _write_reply
    from ui.generation import automatic_reasoning_close_count, literal_prefill_count, literal_text_ranges
    from compare import REPLY
    skip = gr.skip()
    occupied = runtime.MANAGER.claim_generation()
    if occupied:
        yield skip, f"Wait until the model is free; it is {occupied}.", skip, skip
        return
    try:
        if not document:
            raise ValueError("Select a saved experiment first.")
        if expected_identifier is not None and document["id"] != expected_identifier:
            raise ValueError("Wait for the selected experiment to open before rerunning it.")
        run = document["run"]
        published = runtime.MANAGER.loaded_model()
        if not published.load_id or published.model_id != run.get("model_id"):
            raise ValueError(f"Load {run.get('model_id')} on the Models page first.")
        config = run.get("settings") or {}
        yield skip, "Rerunning saved experiment…", gr.update(interactive=False), gr.update(visible=True)
        if run["kind"] == REPLY:
            edited = config.get("edited_prompt")
            forced = int(config.get("forced_prefix_tokens", 0))
            if (edited or forced) and run.get("tokenizer") != _tokenizer_identity():
                raise ValueError("This edited or branched run needs its original tokenizer to replay token IDs.")
            with contextlib.closing(_write_reply(
                run.get("prompt", ""), config.get("system_prompt", ""), "" if forced else config.get("assistant_prefill", ""),
                config.get("temperature", 0.7), config.get("top_p", 1), config.get("top_k", 0),
                config.get("skip_top_below", 0), config.get("max_new_tokens", 256),
                config.get("seed", 42), False,
                config.get("requested_thinking_mode", config.get("thinking_mode", "default")),
                config.get("steering"), published, messages_override=run.get("messages"),
                prompt_override_ids=edited.get("prompt_token_ids") if edited else None,
                forced_ids=[m["token_id"] for m in run["metrics"][:forced]] if forced else None,
                replay_options={
                    "literal_prefill_tokens": literal_prefill_count(run["metrics"], forced),
                    "automatic_reasoning_close_tokens": automatic_reasoning_close_count(run["metrics"], forced),
                    "literal_text_ranges": literal_text_ranges(run["metrics"], forced),
                } if forced else None,
            )) as writing:
                result = None
                for result, count in writing:
                    if result is None:
                        yield skip, f"Rerunning: {count:,} tokens…", gr.update(interactive=False), gr.update(visible=True)
            if result is not None:
                for key in ("edited_prompt", "forced_prefix_tokens"):
                    if key in config:
                        result["settings"][key] = config[key]
        else:
            result = _measure_text(run.get("prompt", ""), run["text"],
                                   config.get("use_chat_template", False), config.get("steering"), published)
        item = runs.save(result, f"Rerun: {document['title']}")
        yield choices(selected=item["id"]), "Rerun saved as a new experiment.", gr.update(interactive=True), gr.update(visible=False)
    except (OSError, ValueError, RuntimeError) as error:
        yield skip, failure_status("Could not rerun experiment", str(error)), gr.update(interactive=True), gr.update(visible=False)
    finally:
        runtime.MANAGER.release_generation()
