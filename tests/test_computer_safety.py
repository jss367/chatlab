"""Benchmark semantics, cancellation/checkpoints, and extension page wiring."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gradio as gr

import app
import settings
import settings_sandbox
from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from extensions.osguard.benchmark import (
    action_scores, cases_from, dataset_digest, demo_cases, execution_results_from,
    execution_scores, messages_for, parse_response, predictions_from, read_json, report,
)
from extensions.osguard.page import build_page
from extensions.osguard.runner import Runner, StreamingResponse
from test_extensions import FakeManager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class BenchmarkTests(unittest.TestCase):
    def test_prompt_has_only_original_task_state_and_action(self):
        case = dict(demo_cases()[0], label="unsafe", proposer_instruction="LEAK", trajectory=["LEAK"], screenshot="LEAK")
        prompt = messages_for(case)
        self.assertNotIn("LEAK", json.dumps(prompt))
        self.assertEqual(set(json.loads(prompt[1]["content"])), {"instruction", "state_text", "candidate_action"})
        with self.assertRaisesRegex(ValueError, "Screenshot-only"):
            messages_for(dict(case, state_text=""))

    def test_import_validates_duplicates_labels_and_preserves_audit_metadata(self):
        cases = demo_cases()
        cases[0]["screenshot"] = "reference.png"
        self.assertEqual(cases_from({"cases": cases}), cases)
        for value in ([], [cases[0], cases[0]], [dict(cases[0], label="safe")], [dict(cases[0], candidate_action={})]):
            with self.assertRaises(ValueError):
                cases_from(value)
        self.assertEqual(dataset_digest(cases), dataset_digest(copy.deepcopy(cases)))
        self.assertNotEqual(dataset_digest(cases), dataset_digest(cases[1:]))

    def test_strict_final_judgments(self):
        for text in ('unsafe', '{"label":"unsafe","feedback":"Overwrite"}',
                     '```json\n{"label":"unsafe"}\n```', '<think>allowed?</think>{"label":"unsafe"}',
                     'Reasoning from a template prefill.</think>unsafe'):
            self.assertEqual(parse_response(text)[0], "unsafe")
        for text in ('not unsafe', 'allowed or unsafe', '{"label":"safe"}',
                     '<think>unsafe', '{"label":"unsafe"', '["unsafe"]', '{"label":"unsafe","feedback":0}'):
            self.assertIsNone(parse_response(text)[0])
        self.assertIsNone(parse_response('unsafe', reasoning_prefilled=True)[0])
        self.assertEqual(parse_response('reasoning</think>unsafe', reasoning_prefilled=True)[0], 'unsafe')

    def test_invalid_predictions_are_not_dropped_from_denominator(self):
        cases = demo_cases()
        predictions = [{"id": cases[0]["id"], "prediction": "allowed"},
                       {"id": cases[1]["id"], "prediction": "allowed"},
                       {"id": cases[2]["id"], "prediction": None}]
        scores = report(cases, predictions)
        self.assertEqual(scores["accuracy"], 1 / 3)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3) / 3)
        self.assertEqual(scores["invalid"], 1)
        self.assertEqual(scores["confusion"]["unsafe"]["invalid"], 1)
        self.assertEqual(scores["per_label"]["allowed"]["precision"], .5)
        self.assertEqual(len(scores["by_source"]), 1)

    def test_pending_cancelled_and_unlabeled_coverage(self):
        cases = demo_cases()
        cases[0].pop("label")
        predictions = [dict(id=cases[0]["id"], prediction="unsafe"),
                       dict(id=cases[1]["id"], prediction=None, status="cancelled")]
        scores = action_scores(cases, predictions)
        self.assertEqual((scores["total"], scores["completed"], scores["scored"]), (3, 1, 0))
        self.assertIsNone(scores["accuracy"])

    def test_many_sources_do_not_rescan_cases_and_predictions_for_each_source(self):
        class CountedList(list):
            visits = 0

            def __iter__(self):
                for item in super().__iter__():
                    self.visits += 1
                    yield item

        cases = CountedList(dict(demo_cases()[0], id=str(i), source=str(i)) for i in range(500))
        predictions = CountedList(dict(id=str(i), prediction="allowed") for i in range(500))
        scores = report(cases, predictions)
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(len(scores["by_source"]), 500)
        self.assertLessEqual(cases.visits, 2 * len(cases))
        self.assertEqual(predictions.visits, len(predictions))

    def test_external_predictions_validate_ids_and_strip_untrusted_metrics(self):
        result = predictions_from([dict(id="demo-safe", prediction="allowed", metrics=[{}])], demo_cases())
        self.assertNotIn("metrics", result[0])
        for row in (dict(id="unknown", prediction="unsafe"), dict(id="demo-safe"),
                    dict(id="demo-safe", prediction="invalid")):
            with self.assertRaises(ValueError):
                predictions_from([row], demo_cases())

    def test_execution_outcomes_do_not_conflate_success_with_safety(self):
        rows = execution_results_from([
            dict(id="safe", condition="guarded", task_success=True, retry_terminated=False, invariants={"file": True}),
            dict(id="unsafe", condition="unguarded", task_success=True, retry_terminated=False, invariants={"file": False}),
            dict(id="stopped", condition="guarded", task_success=False, retry_terminated=True, invariants={"file": True}),
            dict(id="failed", condition="guarded", task_success=False, retry_terminated=False, invariants={"file": False}),
        ])
        self.assertEqual([r["outcome"] for r in rows], ["safe_success", "unsafe_completion", "unsuccessful", "unsuccessful"])
        scores = execution_scores(rows)
        self.assertEqual(scores["guarded"]["safe_success_rate"], 1 / 3)
        self.assertEqual(scores["guarded"]["safety_violation_rate"], 1 / 3)
        self.assertEqual(scores["unguarded"]["unsafe_completion_rate"], 1)
        for fields in (dict(invariants={}), dict(invariants={"file": "false"}),
                       dict(task_success="false"), dict(task_success=True, retry_terminated=True)):
            with self.assertRaises(ValueError):
                execution_results_from([dict(rows[0], **fields)])

    def test_external_status_metadata_cannot_change_scoring_coverage(self):
        cases = demo_cases()
        rows = [dict(id=case['id'], prediction=case['label'], status=status)
                for case, status in zip(cases, ('running', 'cancelled', 'error'))]
        external = predictions_from(rows, cases)
        scores = action_scores(cases, external)
        self.assertEqual((scores['completed'], scores['scored'], scores['accuracy']), (3, 3, 1))
        self.assertTrue(all(row['status'] == 'completed' for row in external))
        saved = predictions_from(rows, cases, saved_run=True)
        self.assertEqual(action_scores(cases, saved)['completed'], 0)
        with self.assertRaisesRegex(ValueError, 'saved-run status'):
            predictions_from([dict(rows[0], status='unexpected')], cases, saved_run=True)
        invalid = predictions_from([dict(rows[0], prediction=None)], cases)
        self.assertEqual(action_scores(cases, invalid)['invalid'], 1)

    def test_execution_conditions_are_grouped_without_repeated_input_scans(self):
        class CountedList(list):
            visits = 0

            def __iter__(self):
                for item in super().__iter__():
                    self.visits += 1
                    yield item

        rows = CountedList(dict(id=str(i), condition=str(i), task_success=True,
                                retry_terminated=False, violated_invariants=[], outcome='safe_success')
                           for i in range(10000))
        scores = execution_scores(rows)
        self.assertEqual(len(scores), 10000)
        self.assertEqual(scores['9999']['safe_success_rate'], 1)
        self.assertEqual(rows.visits, len(rows))

    def test_jsonl_and_error_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(case) for case in demo_cases()))
            self.assertEqual(cases_from(read_json(path)), demo_cases())
            path.write_text("{broken}")
            with self.assertRaisesRegex(ValueError, "valid JSON"):
                read_json(path)

    def test_saved_run_size_allowance_does_not_relax_other_import_limits(self):
        from extensions.osguard import benchmark

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(benchmark, 'MAX_FILE_BYTES', 100), \
                mock.patch.object(benchmark, 'MAX_SAVED_RUN_BYTES', 1000):
            path = Path(directory) / 'run.json'
            value = dict(format=benchmark.FORMAT, trace='x' * 200)
            path.write_text(json.dumps(value))
            self.assertEqual(read_json(path, allow_saved_run=True), value)
            with self.assertRaisesRegex(ValueError, '32 MB'):
                read_json(path)
            path.write_text(json.dumps(dict(cases=[{'metadata': 'x' * 200}])))
            with self.assertRaisesRegex(ValueError, 'Only saved-run'):
                read_json(path, allow_saved_run=True)
            path.write_text(json.dumps(dict(value, trace='x' * 1000)))
            with self.assertRaisesRegex(ValueError, '512 MB'):
                read_json(path, allow_saved_run=True)


class JudgmentManager(FakeManager):
    def generate(self, messages, **options):
        self.options = options
        try:
            yield SimpleNamespace(text='{"label":', metrics=[], prompt_ids=[1, 2])
            yield SimpleNamespace(text='{"label":"allowed"}', metrics=[], prompt_ids=[1, 2])
        finally:
            self.closed_streams += 1


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = JudgmentManager()
        self.runner = Runner(ModelService(lambda: self.manager), Path(self.directory.name) / "new" / "extension")

    def test_batch_saves_reproducible_private_checkpoint_and_releases_model(self):
        frames = list(self.runner.run("owner", demo_cases()))
        final, path = frames[-1]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["scores"]["completed"], 3)
        self.assertEqual(final["model_id"], "test/model")
        self.assertEqual(final["mode"], "text_only_adaptation")
        self.assertEqual(final["predictions"][0]["messages"], messages_for(demo_cases()[0]))
        self.assertEqual(json.loads(Path(path).read_text()), final)
        self.assertEqual(Path(path).stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.manager.closed_streams, 3)
        self.assertEqual(self.manager.releases, 1)
        self.assertFalse(self.manager.busy)
        live_frames = [frame for frame, _ in frames if isinstance(frame, StreamingResponse)]
        self.assertEqual(live_frames[0].result["response"], '{"label":')

    def test_streaming_does_not_rescore_or_copy_prior_results(self):
        from extensions.osguard import runner as module

        with mock.patch.object(module, "report", wraps=report) as scoring:
            stream = self.runner.run("owner", demo_cases())
            next(stream)  # Initial empty batch snapshot.
            next(stream)  # First response update.
            next(stream)  # Second response update.
            self.assertEqual(scoring.call_count, 1)
            finished, _ = next(stream)
            self.assertEqual(finished["scores"]["completed"], 1)
            calls_after_completion = scoring.call_count
            real_copy = copy.deepcopy
            copied = []

            def recording_copy(value, *args, **kwargs):
                copied.append(value)
                return real_copy(value, *args, **kwargs)

            with mock.patch.object(module.copy, "deepcopy", side_effect=recording_copy):
                live, _ = next(stream)
            self.assertIsInstance(live, StreamingResponse)
            self.assertEqual(live.result["id"], demo_cases()[1]["id"])
            self.assertFalse(any(isinstance(value, dict) and "cases" in value for value in copied))
            self.assertEqual(scoring.call_count, calls_after_completion)
            stream.close()

    def test_cancel_keeps_partial_response_without_scoring_it(self):
        stream = self.runner.run("owner", demo_cases())
        next(stream)
        next(stream)
        self.runner.cancel("other-owner")
        self.assertTrue(self.manager.busy)
        self.runner.cancel("owner")
        run, _ = list(stream)[-1]
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["predictions"][0]["status"], "cancelled")
        self.assertEqual(run["scores"]["completed"], 0)
        self.assertEqual(len(run["predictions"]), 1)
        self.assertFalse(self.manager.busy)

    def test_disconnected_generator_releases_model_and_saves(self):
        stream = self.runner.run("owner", demo_cases())
        next(stream)
        next(stream)
        stream.close()
        self.assertFalse(self.manager.busy)
        saved = json.loads(next(self.runner.data_dir.glob("*.json")).read_text())
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual(saved["predictions"][0]["status"], "cancelled")

    def test_generation_error_releases_lease_and_records_failed_case(self):
        with mock.patch.object(self.manager, "generate", side_effect=RuntimeError("generation failed")):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                list(self.runner.run("owner", demo_cases()))
        self.assertFalse(self.manager.busy)
        saved = json.loads(next(self.runner.data_dir.glob("*.json")).read_text())
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["predictions"][0]["status"], "error")
        self.assertEqual(saved["scores"]["completed"], 0)

    def test_screenshot_only_case_fails_before_claiming_model(self):
        with self.assertRaises(ValueError):
            list(self.runner.run("owner", [dict(demo_cases()[0], state_text="")]))
        self.assertEqual(self.manager.releases, 0)

    def test_invalid_sampling_inputs_do_not_claim_model_or_block_the_next_run(self):
        for value in (-1, None, float('nan'), float('inf'), .5, True, '7'):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "Seed must"):
                list(self.runner.run("owner", demo_cases(), seed=value))
        for value in (-1, 0, None, float('nan'), float('inf'), 1.5, True, 4097):
            with self.subTest(token_limit=value), self.assertRaisesRegex(ValueError, "Maximum answer tokens"):
                list(self.runner.run("owner", demo_cases(), max_new_tokens=value))
        self.assertEqual(self.manager.releases, 0)
        self.assertFalse(self.runner.data_dir.exists())
        run, _ = list(self.runner.run("owner", demo_cases(), seed=0))[-1]
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["sampling"]["seed"], 0)


class PageTests(unittest.TestCase):
    def test_page_load_evaluate_export_and_execution_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JudgmentManager()
            context = ExtensionContext(ModelService(lambda: manager), TokenInspector(),
                                       Path(directory) / "extension", NavigationService(lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            self.addCleanup(demo.close)
            seed_control = next(block for block in demo.blocks.values() if getattr(block, 'label', None) == 'Seed')
            self.assertEqual(seed_control.minimum, 0)
            functions = {function.fn.__name__: function.fn for function in demo.fns.values() if function.fn}
            case_path = Path(directory) / "cases.json"
            case_path.write_text(json.dumps(demo_cases()))
            loaded = functions["load_cases"](str(case_path), "owner")
            self.assertEqual(len(loaded), 13)
            frames = list(functions["evaluate"](loaded[0], "owner", 256, 42))
            self.assertEqual(len(frames[-1]), 12)
            self.assertIn('**Completed:** 0/3', frames[0][2])
            for index in range(5):
                self.assertEqual(frames[1][index], gr.skip())
                self.assertEqual(frames[2][index], gr.skip())
            self.assertNotEqual(frames[3][0], gr.skip())  # Completed case updates batch state.
            exported = functions["export_run"](frames[-1][0])
            # The checkpoint includes prompts and traces beyond the ordinary
            # dataset size. Reopen through the real saved-run callback.
            with mock.patch('extensions.osguard.benchmark.MAX_FILE_BYTES', 100):
                replay = functions["load_cases"](exported, "owner")
            self.assertEqual(len(replay[1]["predictions"]), 3)
            self.assertEqual(replay[1]["imported_provenance"]["model_id"], "test/model")
            self.assertNotIn("metrics", replay[1]["predictions"][0])
            scored = functions["score_external"](exported, loaded[0], "owner")
            self.assertEqual(len(scored), 11)
            execution_path = Path(directory) / "execution.json"
            execution_path.write_text(json.dumps([dict(id="one", task_success=True, retry_terminated=False, invariants={"preserved": False})]))
            reviewed = functions["review_executions"](str(execution_path))
            self.assertEqual(reviewed[0][0][2], "unsafe_completion")
            self.assertTrue(Path(reviewed[2]).is_file())

    def test_app_registers_safety_alongside_other_safety_extension(self):
        settings.update(enabled_extensions=["osguard", "os_harm"])
        try:
            demo = app.build_app()
            self.addCleanup(demo.close)
            nav = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "nav")
            self.assertIn("Safety", [value for _, value in nav.choices])
            self.assertIn("OS-Harm", [value for _, value in nav.choices])
            self.assertTrue(any(getattr(b, "elem_id", None) == "computer-safety-page" for b in demo.blocks.values()))
        finally:
            settings.update(enabled_extensions=[])
