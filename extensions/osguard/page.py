"""Gradio workbench for OSGuard action evaluation and execution-result review."""
from uuid import uuid4

import gradio as gr

from .benchmark import (
    FORMAT, PAPER, at_threshold, cases_from, dataset_digest, demo_cases, execution_results_from,
    execution_scores, predictions_from, read_json, report,
)
from .chart import tradeoff_chart
from .runner import Runner, StreamingResponse
from .storage import save_json

SCORING = {"Label probabilities": "probability", "Free-text judgment": "judgment"}

CSS = """
#computer-safety-page {overflow-y:auto; min-height:0; padding:12px;}
#computer-safety-page .safety-note {max-width:1000px;}
"""


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def result_rows(cases, predictions):
    by_id = {row["id"]: row for row in predictions}
    rows = []
    for case in cases:
        result = by_id.get(case["id"], {})
        probabilities = result.get("probabilities")
        rows.append([case["id"], case.get("source", "unspecified"), case.get("label") or "unlabeled",
                     result.get("prediction") or "—",
                     percent(probabilities["unsafe"]) if isinstance(probabilities, dict) else "—",
                     result.get("status", "pending")])
    return rows


def summary(scores, threshold=0.0):
    interval = scores.get("accuracy_interval")
    band = f" (95% CI {percent(interval[0])}–{percent(interval[1])})" if interval else ""
    skipped = f" · **Skipped:** {scores['skipped']}" if scores.get("skipped") else ""
    decision = (f"Predictions re-decided at P(unsafe) ≥ {threshold:.2f}; the saved run keeps "
                "each model's own answer." if threshold else
                "Predictions are the model's own top label.")
    return (f"**Completed:** {scores['completed']}/{scores['total']} · "
            f"**Scored:** {scores['scored']} · **Invalid answers:** {scores['invalid']}{skipped}\n\n"
            f"**Accuracy:** {percent(scores['accuracy'])}{band} · "
            f"**Macro-F1:** {percent(scores['macro_f1'])}\n\n"
            f"{decision} Scores cover completed, labeled cases. Invalid answers count as "
            "incorrect. Macro-F1 averages all three labels; absent classes contribute zero. "
            "The interval is a 95% Wilson score interval over the scored cases.")


def scored_view(cases, predictions, threshold=0.0):
    """Table, summary, scores and curve for one set of results at one threshold."""
    decided = at_threshold(predictions, threshold)
    scores = report(cases, decided)
    return (result_rows(cases, decided), summary(scores, threshold), scores,
            tradeoff_chart(scores["blocking"]))


def build_page(context):
    runner = Runner(context.models, context.data_dir)
    selections = context.tokens.selections()
    # The slider answers outside the batch queue, so a batch frame that reused
    # the threshold captured when it started would undo a move made while it
    # was running. One current value per session lives here instead, the way
    # token selections keep one current response, and out of the event
    # snapshots a running generator is holding.
    thresholds = {}

    def forget(owner):
        runner.cancel(owner)
        selections.forget(owner)
        thresholds.pop(owner, None)

    with gr.Column(elem_id="computer-safety-page"):
        owner = gr.State(value=selections.new_session, delete_callback=forget)
        cases_state = gr.State([])
        run_state = gr.State({})
        token_state = gr.State(("", []))
        gr.Markdown("# Computer-use safety benchmark\n"
                    f"Work with [OSGuard]({PAPER}): judge proposed actions and review task safety.")
        gr.Markdown("Local evaluation uses **text descriptions of interface state**. It does not send screenshots "
                    "to the model and is not a reproduction of the paper’s multimodal results. "
                    "Import external predictions to score screenshot-based evaluations. "
                    "The released benchmark dataset is not bundled.", elem_classes=["safety-note"])
        with gr.Tabs():
            with gr.Tab("Action judgments"):
                with gr.Row():
                    case_file = gr.File(label="Cases or saved run (.json / .jsonl)", file_types=[".json", ".jsonl"], type="filepath")
                    prediction_file = gr.File(label="External predictions (.json / .jsonl)", file_types=[".json", ".jsonl"], type="filepath")
                with gr.Row():
                    import_cases = gr.Button("Import cases / saved run")
                    demo = gr.Button("Load synthetic demonstration")
                    import_predictions = gr.Button("Score external predictions")
                note = gr.Markdown("Import your cases or try three synthetic examples. These examples are not official benchmark items.")
                with gr.Accordion("Import format", open=False):
                    gr.Markdown('Cases: a list of objects with `id`, `instruction`, `state_text`, `candidate_action`, '
                                'optional `label` (`allowed`, `unrelated`, `unsafe`) and `source`. '
                                'A JSON object containing `cases` also works. Keep screenshot references and other audit fields '
                                'as metadata; they are preserved but never opened or sent to the local model.\n\n'
                                'Predictions: a list (or an object containing `predictions`) with `id` and '
                                '`prediction` (a label or null for an invalid answer). Optional `response` and `feedback` '
                                'are retained. Supply `probabilities` — a number for each of the three labels — to get '
                                'the blocking curve and the threshold slider for an external evaluator too. '
                                'A subset is scored with coverage shown. See `COMPUTER_USE_SAFETY.md` for examples.')
                with gr.Row():
                    scoring = gr.Radio(list(SCORING), value="Label probabilities", label="Scoring",
                                       info="Probabilities replay each label as the answer and read what the "
                                            "model gave it: no malformed answers, no sampled tokens, one "
                                            "confidence per case. Free text asks for a JSON judgment and an "
                                            "explanation, and parses the label back out.")
                    max_tokens = gr.Slider(32, 4096, value=256, step=32, label="Maximum answer tokens",
                                           info="Free-text judgments only.")
                    seed = gr.Number(value=42, precision=0, minimum=0, label="Seed")
                    models = gr.Button("Open Models")
                context.navigation.open_models(models)
                with gr.Row():
                    start = gr.Button("Evaluate all cases · text only", variant="primary")
                    stop = gr.Button("Stop batch")
                score_note = gr.Markdown("No judgments yet.")
                threshold = gr.Slider(0, 1, value=0, step=0.01,
                                      label="Block when P(unsafe) is at least this",
                                      info="Rescores the cases already run, without the model. "
                                           "Zero keeps each model's own top label.")
                curve = gr.HTML(tradeoff_chart(None))
                table = gr.Dataframe(headers=["Case", "Source", "Reference", "Prediction", "P(unsafe)", "Status"],
                                     datatype=["str"] * 6, value=[], interactive=False, label="Select a case to inspect")
                with gr.Row():
                    case_detail = gr.JSON(label="Case and audit metadata")
                    response = gr.Textbox(label="Model response", lines=5, interactive=False)
                strip = gr.HighlightedText(label="Click a generated token to inspect it", combine_adjacent=False,
                                           color_map=context.tokens.color_map, show_legend=True)
                detail = gr.Markdown("Select a token from a local run.")
                alternatives = gr.Dataframe(headers=["Token ID", "Text", "Raw probability"], interactive=False)
                with gr.Accordion("Class metrics, confusion matrix and source breakdown", open=False):
                    scores_json = gr.JSON(label="Scores")
                export = gr.File(label="Saved run JSON", interactive=False)
                save = gr.Button("Export current results")
            with gr.Tab("Desktop execution results"):
                gr.Markdown("Import results from an external OSWorld evaluator. ChatLab reports the supplied "
                            "task-success and named safety-invariant checks; it does not launch a desktop environment "
                            "or independently verify those checks. Safe success requires task completion and every "
                            "invariant to pass. Retry termination is reported separately.")
                execution_file = gr.File(label="Execution results (.json / .jsonl)", file_types=[".json", ".jsonl"], type="filepath")
                review = gr.Button("Review execution results")
                execution_table = gr.Dataframe(headers=["Run", "Condition", "Outcome", "Retry terminated", "Violated checks"],
                                               interactive=False, value=[])
                execution_summary = gr.JSON(label="Rates by condition")
                execution_export = gr.File(label="Execution summary JSON", interactive=False)
                with gr.Accordion("Execution result format", open=False):
                    gr.Markdown('Each record needs `id`, `task_success` (boolean), `retry_terminated` (boolean), '
                                'and `invariants` (a nonempty object of named boolean checks). '
                                'Use `condition` to separate guarded and unguarded results, and unique run IDs '
                                'when the same task has multiple conditions. A list or an object containing `executions` works.')

    def clear_view(session_id):
        payload, _ = selections.view(session_id, uuid4().hex, [])
        return payload, [], None, "", "Select a token from a local run.", []

    def load_cases(path, session_id, block_at, synthetic=False):
        try:
            value = demo_cases() if synthetic else read_json(path, allow_saved_run=True)
            cases = cases_from(value)
            predictions = []
            if isinstance(value, dict) and value.get("format") == FORMAT:
                if value.get("dataset_sha256") != dataset_digest(cases):
                    raise ValueError("Saved run dataset fingerprint does not match its cases.")
                predictions = predictions_from(value, cases, saved_run=True) if value.get("predictions") else []
            run = dict(format=FORMAT, paper=PAPER, cases=cases, predictions=predictions,
                       dataset_sha256=dataset_digest(cases), mode="imported" if predictions else "not_run")
            if isinstance(value, dict):
                run["imported_provenance"] = {key: value[key] for key in
                    ("mode", "scoring", "model_id", "sampling", "created_at", "dataset_sha256") if key in value}
            rows, note_text, scores, chart = scored_view(cases, predictions, block_at)
            message = ("Loaded synthetic demonstration — not official benchmark data." if synthetic else
                       f"Loaded {len(cases)} cases. Imported results retain responses; token inspection is available for local runs in this session.")
            without_state = sum(1 for case in cases if not case.get("state_text", "").strip())
            if without_state:
                message += (f" {without_state} of them carry no state_text and will be skipped by a local "
                            "run; score those from an external evaluator's predictions.")
            return (cases, run, rows, message, note_text, scores, chart, None,
                    *clear_view(session_id))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    load_outputs = [cases_state, run_state, table, note, score_note, scores_json, curve, export,
                    token_state, strip, case_detail, response, detail, alternatives]
    serial = dict(concurrency_id="computer-safety-batch", concurrency_limit=1)
    import_cases.click(load_cases, [case_file, owner, threshold], load_outputs, **serial)
    demo.click(lambda session_id, block_at: load_cases(None, session_id, block_at, True),
               [owner, threshold], load_outputs, **serial)

    def score_external(path, cases, session_id, block_at):
        try:
            if not cases:
                raise ValueError("Import cases first.")
            value = read_json(path)
            if isinstance(value, dict) and "dataset_sha256" in value and value["dataset_sha256"] != dataset_digest(cases):
                raise ValueError("Predictions belong to a different dataset fingerprint.")
            predictions = predictions_from(value, cases)
            run = dict(format=FORMAT, paper=PAPER, cases=cases, predictions=predictions,
                       dataset_sha256=dataset_digest(cases), mode="external_predictions")
            if isinstance(value, dict):
                run["imported_provenance"] = {key: value[key] for key in ("model_id", "mode", "sampling", "created_at") if key in value}
            rows, note_text, scores, chart = scored_view(cases, predictions, block_at)
            return (run, rows, note_text, scores, chart, None, *clear_view(session_id))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    import_predictions.click(score_external, [prediction_file, cases_state, owner, threshold],
                             [run_state, table, score_note, scores_json, curve, export, token_state, strip,
                              case_detail, response, detail, alternatives], **serial)

    def rescore(cases, run, session_id, block_at):
        """Move the blocking threshold over results already in hand."""
        thresholds[session_id] = block_at
        if not cases:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        rows, note_text, scores, chart = scored_view(cases, run.get("predictions", []), block_at)
        return rows, note_text, scores, chart

    # Deliberately outside the batch queue: rescoring reads a snapshot and
    # touches no model, and a slider that answered only once an hour-long
    # batch had finished would not be a control at all.
    threshold.release(rescore, [cases_state, run_state, owner, threshold],
                      [table, score_note, scores_json, curve])

    def evaluate(cases, session_id, mode, token_limit, random_seed, block_at):
        try:
            if mode not in SCORING:
                raise ValueError("Choose label probabilities or free-text judgment.")
            thresholds[session_id] = block_at
            by_id = {case["id"]: case for case in cases}
            stream = runner.run(session_id, cases, mode=SCORING[mode],
                                max_new_tokens=token_limit, seed=random_seed)
            try:
                for run, path in stream:
                    if isinstance(run, StreamingResponse):
                        last = run.result
                        metrics = last["metrics"]
                        payload, changed = selections.view(session_id, (run.run_id, last["id"]), metrics)
                        # Leave batch state, tables, scores and downloads alone
                        # until a case completes; only stream the current answer.
                        yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), payload,
                               context.tokens.strip(metrics), by_id[last["id"]] if changed else gr.skip(),
                               last["response"], "Select a generated token." if changed else gr.skip(),
                               [] if changed else gr.skip(), gr.skip())
                        continue
                    last = run["predictions"][-1] if run["predictions"] else {}
                    metrics = last.get("metrics", [])
                    payload, changed = selections.view(session_id, (run["id"], last.get("id")), metrics)
                    case = by_id.get(last.get("id"))
                    # The runner already scored this frame; only a moved
                    # threshold makes those numbers the wrong ones to show.
                    # Read it now rather than trust the click's snapshot: the
                    # reader may have moved the slider since the batch began.
                    current = thresholds.get(session_id, block_at)
                    rows, note_text, scores, chart = (
                        scored_view(cases, run["predictions"], current) if current else
                        (result_rows(cases, run["predictions"]), summary(run["scores"]),
                         run["scores"], tradeoff_chart(run["scores"]["blocking"])))
                    yield (run, rows, note_text, scores, chart, path, payload,
                           context.tokens.strip(metrics), case, last.get("response", ""),
                           "Select a generated token." if changed else gr.skip(), [] if changed else gr.skip(),
                           f"Text-only batch: {run['status']}.")
            finally:
                stream.close()
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    start.click(evaluate, [cases_state, owner, scoring, max_tokens, seed, threshold],
                [run_state, table, score_note, scores_json, curve, export, token_state, strip,
                 case_detail, response, detail, alternatives, note], **serial)
    stop.click(runner.cancel, owner, [], queue=False)

    def inspect_case(cases, run, session_id, event: gr.SelectData):
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        if not isinstance(index, int) or not 0 <= index < len(cases):
            return (gr.skip(),) * 6
        case = cases[index]
        result = next((r for r in run.get("predictions", []) if r["id"] == case["id"]), {})
        metrics = result.get("metrics", [])
        payload, _ = selections.view(session_id, (run.get("id"), case["id"]), metrics)
        return case, str(result.get("response", "")), payload, context.tokens.strip(metrics), "Select a generated token.", []

    table.select(inspect_case, [cases_state, run_state, owner],
                 [case_detail, response, token_state, strip, detail, alternatives], **serial)

    def inspect_token(session_id, payload, event: gr.SelectData):
        return selections.inspect(session_id, payload, event)

    strip.select(inspect_token, [owner, token_state], [detail, alternatives], queue=False)

    def export_run(run):
        if not run.get("cases"):
            raise gr.Error("Import cases first.")
        path = context.data_dir / f"{uuid4().hex}.json"
        value = dict(run, scores=report(run["cases"], run["predictions"]))
        save_json(path, value)
        return str(path)

    save.click(export_run, run_state, export, **serial)

    def review_executions(path):
        try:
            rows = execution_results_from(read_json(path))
            scores = execution_scores(rows)
            destination = context.data_dir / f"execution-{uuid4().hex}.json"
            save_json(destination, dict(paper=PAPER, executions=rows, scores=scores))
            return ([[row["id"], row.get("condition", "unspecified"), row["outcome"], row["retry_terminated"],
                      ", ".join(row["violated_invariants"])] for row in rows], scores, str(destination))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    review.click(review_executions, execution_file, [execution_table, execution_summary, execution_export])
