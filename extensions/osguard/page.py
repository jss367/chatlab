"""Gradio workbench for OSGuard action evaluation and execution-result review."""
from uuid import uuid4

import gradio as gr

from .benchmark import (
    FORMAT, PAPER, cases_from, dataset_digest, demo_cases, execution_results_from,
    execution_scores, predictions_from, read_json, report,
)
from .runner import Runner, StreamingResponse
from .storage import save_json

CSS = """
#computer-safety-page {overflow-y:auto; min-height:0; padding:12px;}
#computer-safety-page .safety-note {max-width:1000px;}
"""


def result_rows(cases, predictions):
    by_id = {row["id"]: row for row in predictions}
    return [[case["id"], case.get("source", "unspecified"), case.get("label") or "unlabeled",
             by_id.get(case["id"], {}).get("prediction") or "—",
             by_id.get(case["id"], {}).get("status", "pending")]
            for case in cases]


def summary(scores):
    def percent(value):
        return "—" if value is None else f"{value:.1%}"
    return (f"**Completed:** {scores['completed']}/{scores['total']} · "
            f"**Scored:** {scores['scored']} · **Invalid answers:** {scores['invalid']}\n\n"
            f"**Accuracy:** {percent(scores['accuracy'])} · **Macro-F1:** {percent(scores['macro_f1'])}\n\n"
            "Scores cover completed, labeled cases. Invalid answers count as incorrect. "
            "Macro-F1 averages all three labels; absent classes contribute zero.")


def build_page(context):
    runner = Runner(context.models, context.data_dir)
    selections = context.tokens.selections()

    def forget(owner):
        runner.cancel(owner)
        selections.forget(owner)

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
                                'are retained. A subset is scored with coverage shown. See `COMPUTER_USE_SAFETY.md` for examples.')
                with gr.Row():
                    max_tokens = gr.Slider(32, 4096, value=256, step=32, label="Maximum answer tokens")
                    seed = gr.Number(value=42, precision=0, minimum=0, label="Seed")
                    models = gr.Button("Open Models")
                context.navigation.open_models(models)
                with gr.Row():
                    start = gr.Button("Evaluate all cases · text only", variant="primary")
                    stop = gr.Button("Stop batch")
                score_note = gr.Markdown("No judgments yet.")
                table = gr.Dataframe(headers=["Case", "Source", "Reference", "Prediction", "Status"],
                                     datatype=["str"] * 5, value=[], interactive=False, label="Select a case to inspect")
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

    def load_cases(path, session_id, synthetic=False):
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
                    ("mode", "model_id", "sampling", "created_at", "dataset_sha256") if key in value}
            scores = report(cases, predictions)
            message = ("Loaded synthetic demonstration — not official benchmark data." if synthetic else
                       f"Loaded {len(cases)} cases. Imported results retain responses; token inspection is available for local runs in this session.")
            return (cases, run, result_rows(cases, predictions), message, summary(scores), scores, None,
                    *clear_view(session_id))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    load_outputs = [cases_state, run_state, table, note, score_note, scores_json, export,
                    token_state, strip, case_detail, response, detail, alternatives]
    serial = dict(concurrency_id="computer-safety-batch", concurrency_limit=1)
    import_cases.click(load_cases, [case_file, owner], load_outputs, **serial)
    demo.click(lambda session_id: load_cases(None, session_id, True), owner, load_outputs, **serial)

    def score_external(path, cases, session_id):
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
            scores = report(cases, predictions)
            return (run, result_rows(cases, predictions), summary(scores), scores, None, *clear_view(session_id))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    import_predictions.click(score_external, [prediction_file, cases_state, owner],
                             [run_state, table, score_note, scores_json, export, token_state, strip,
                              case_detail, response, detail, alternatives], **serial)

    def evaluate(cases, session_id, token_limit, random_seed):
        try:
            by_id = {case["id"]: case for case in cases}
            stream = runner.run(session_id, cases, max_new_tokens=token_limit, seed=random_seed)
            try:
                for run, path in stream:
                    if isinstance(run, StreamingResponse):
                        last = run.result
                        metrics = last["metrics"]
                        payload, changed = selections.view(session_id, (run.run_id, last["id"]), metrics)
                        # Leave batch state, tables, scores and downloads alone
                        # until a case completes; only stream the current answer.
                        yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), payload,
                               context.tokens.strip(metrics), by_id[last["id"]] if changed else gr.skip(),
                               last["response"], "Select a generated token." if changed else gr.skip(),
                               [] if changed else gr.skip(), gr.skip())
                        continue
                    last = run["predictions"][-1] if run["predictions"] else {}
                    metrics = last.get("metrics", [])
                    payload, changed = selections.view(session_id, (run["id"], last.get("id")), metrics)
                    case = by_id.get(last.get("id"))
                    yield (run, result_rows(cases, run["predictions"]), summary(run["scores"]), run["scores"],
                           path, payload, context.tokens.strip(metrics), case, last.get("response", ""),
                           "Select a generated token." if changed else gr.skip(), [] if changed else gr.skip(),
                           f"Text-only batch: {run['status']}.")
            finally:
                stream.close()
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    start.click(evaluate, [cases_state, owner, max_tokens, seed],
                [run_state, table, score_note, scores_json, export, token_state, strip,
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
