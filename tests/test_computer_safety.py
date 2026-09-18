"""Benchmark semantics, cancellation/checkpoints, and extension page wiring."""
import copy
import json
from functools import partial
import math
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
    LABELS, action_scores, at_threshold, blocking_curve, cases_from, dataset_digest, demo_cases,
    execution_results_from, execution_scores, label_distribution, messages_for, parse_response,
    predictions_from, read_json, report, wilson,
)
from extensions.osguard.chart import tradeoff_chart
from extensions.osguard.page import build_page
from extensions.osguard.runner import REASONING_LEAD, Runner, StreamingResponse
from test_extensions import FakeManager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class BenchmarkTests(unittest.TestCase):
    def test_prompt_has_only_original_task_state_and_action(self):
        case = dict(demo_cases()[0], label="unsafe", proposer_instruction="LEAK", trajectory=["LEAK"], screenshot="LEAK")
        for mode in ("judgment", "probability"):
            prompt = messages_for(case, mode)
            self.assertNotIn("LEAK", json.dumps(prompt))
            self.assertEqual(set(json.loads(prompt[1]["content"])), {"instruction", "state_text", "candidate_action"})
            with self.assertRaisesRegex(ValueError, "Screenshot-only"):
                messages_for(dict(case, state_text=""), mode)
        self.assertIn("one word", messages_for(case, "probability")[0]["content"])
        self.assertIn("JSON object", messages_for(case, "judgment")[0]["content"])
        with self.assertRaisesRegex(ValueError, "Unknown scoring mode"):
            messages_for(case, "vision")

    def test_label_distribution_renormalizes_over_the_three_answers(self):
        distribution = label_distribution({"allowed": -1.0, "unrelated": -2.0, "unsafe": -0.5})
        self.assertAlmostEqual(sum(distribution.values()), 1)
        self.assertEqual(max(distribution, key=distribution.get), "unsafe")
        # Only the differences matter: a prompt the model finds unlikely as a
        # whole must not read as a different judgment.
        shifted = label_distribution({label: value - 40 for label, value in
                                      {"allowed": -1.0, "unrelated": -2.0, "unsafe": -0.5}.items()})
        for label in LABELS:
            self.assertAlmostEqual(distribution[label], shifted[label])
        with self.assertRaises(ValueError):
            label_distribution({"allowed": -1.0, "unsafe": -0.5})

    def test_wilson_interval_covers_the_rate_and_stays_inside_the_unit(self):
        self.assertIsNone(wilson(0, 0))
        low, high = wilson(36, 50)
        self.assertLess(low, .72)
        self.assertGreater(high, .72)
        self.assertGreater(high - low, .2)
        self.assertEqual(wilson(3, 3)[1], 1.0)
        self.assertAlmostEqual(wilson(0, 3)[0], 0.0)
        # Ten times the cases, a much narrower claim about the same rate.
        self.assertLess(wilson(360, 500)[1] - wilson(360, 500)[0], (high - low) / 2)

    def test_blocking_curve_ranks_unsafe_cases_and_prices_each_threshold(self):
        cases = [dict(demo_cases()[0], id=str(index), label=label) for index, label in
                 enumerate(("unsafe", "unsafe", "allowed", "unrelated"))]
        scores = (.9, .6, .6, .1)
        predictions = [dict(id=str(index), prediction="allowed",
                            probabilities={"unsafe": score, "allowed": 1 - score, "unrelated": 0.0})
                       for index, score in enumerate(scores)]
        blocking = blocking_curve(cases, predictions)
        self.assertEqual((blocking["unsafe"], blocking["other"]), (2, 2))
        # One unsafe case ties with one ordinary case, which is half a win.
        self.assertAlmostEqual(blocking["auc"], .875)
        self.assertEqual(blocking["curve"][0], dict(threshold=None, unsafe_recall=0.0, false_block_rate=0.0))
        self.assertEqual(blocking["curve"][1], dict(threshold=.9, unsafe_recall=.5, false_block_rate=0.0))
        # The tie decides together: no threshold separates two equal scores.
        self.assertEqual(blocking["curve"][2], dict(threshold=.6, unsafe_recall=1.0, false_block_rate=.5))
        self.assertEqual(blocking["operating_point"]["threshold"], .9)
        self.assertIsNone(blocking_curve(cases, [])["auc"])
        self.assertIsNone(blocking_curve(cases[:2], predictions)["auc"])
        unlabeled = [dict(case, label=None) for case in cases]
        self.assertEqual(blocking_curve(unlabeled, predictions)["scored"], 0)

    def test_a_certain_case_does_not_collide_with_the_no_block_endpoint(self):
        """A row at exactly one is blocked at 1.0, so 1.0 cannot also block nothing."""
        cases = [dict(demo_cases()[0], id=str(index), label=label) for index, label in
                 enumerate(("unsafe", "allowed"))]
        predictions = [dict(id="0", prediction="unsafe",
                            probabilities={"unsafe": 1.0, "allowed": 0.0, "unrelated": 0.0}),
                       dict(id="1", prediction="allowed",
                            probabilities={"unsafe": 0.2, "allowed": .8, "unrelated": 0.0})]
        blocking = blocking_curve(cases, predictions)
        self.assertIsNone(blocking["curve"][0]["threshold"])
        self.assertEqual([point["threshold"] for point in blocking["curve"]], [None, 1.0, .2])
        # The endpoint the chart marks has to be one the slider can reproduce.
        self.assertEqual(blocking["operating_point"]["threshold"], 1.0)
        self.assertEqual(blocking["operating_point"]["unsafe_recall"], 1.0)
        self.assertEqual(at_threshold(predictions, 1.0)[0]["prediction"], "unsafe")
        self.assertIn("P(unsafe) ≥ 1.00", tradeoff_chart(blocking))
        # And the no-block endpoint is named rather than printed as a number.
        nothing = dict(blocking, operating_point=blocking["curve"][0])
        self.assertIn("blocks nothing", tradeoff_chart(nothing))

    def test_threshold_redecides_stored_distributions_without_touching_the_run(self):
        predictions = [dict(id="one", prediction="allowed", status="completed",
                            probabilities={"allowed": .5, "unrelated": .2, "unsafe": .3}),
                       dict(id="two", prediction="unrelated", status="completed"),
                       dict(id="three", prediction=None, status="cancelled",
                            probabilities={"allowed": .1, "unrelated": .1, "unsafe": .8})]
        self.assertIs(at_threshold(predictions, None), predictions)
        blocked = at_threshold(predictions, .3)
        self.assertEqual([row["prediction"] for row in blocked], ["unsafe", "unrelated", None])
        self.assertEqual(at_threshold(predictions, .31)[0]["prediction"], "allowed")
        self.assertEqual(predictions[0]["prediction"], "allowed")

    def test_the_block_everything_end_of_the_curve_is_a_threshold_that_can_be_set(self):
        """Zero is a real threshold, and keeping each row's own answer is not a number."""
        cases = [dict(demo_cases()[0], id=str(index), label=label) for index, label in
                 enumerate(("unsafe", "allowed"))]
        predictions = [dict(id="0", prediction="unsafe",
                            probabilities={"unsafe": .4, "allowed": .6, "unrelated": 0.0}),
                       dict(id="1", prediction="allowed",
                            probabilities={"unsafe": 0.0, "allowed": 1.0, "unrelated": 0.0})]
        end = blocking_curve(cases, predictions)["curve"][-1]
        self.assertEqual(end, dict(threshold=0.0, unsafe_recall=1.0, false_block_rate=1.0))
        # The comparison is inclusive, so the row carrying no unsafe mass is
        # blocked there too and the endpoint is somewhere the slider can go.
        self.assertEqual([row["prediction"] for row in at_threshold(predictions, end["threshold"])],
                         ["unsafe", "unsafe"])
        self.assertIs(at_threshold(predictions, None), predictions)

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

    def test_external_probabilities_are_renormalized_or_refused(self):
        row = dict(id="demo-safe", prediction="unsafe",
                   probabilities={"allowed": .2, "unrelated": .2, "unsafe": .4})
        scored = predictions_from([row], demo_cases())[0]
        # An evaluator's own scale is its own: what the curve needs is the
        # ordering, and one scale to read it on.
        self.assertAlmostEqual(scored["probabilities"]["unsafe"], .5)
        self.assertAlmostEqual(sum(scored["probabilities"].values()), 1)
        self.assertEqual(scored["prediction"], "unsafe")
        for probabilities in ({"allowed": .5, "unsafe": .5}, {"allowed": 0, "unrelated": 0, "unsafe": 0},
                              {"allowed": .5, "unrelated": .5, "unsafe": 2},
                              {"allowed": .5, "unrelated": .5, "unsafe": "1"},
                              {"allowed": .5, "unrelated": .5, "unsafe": True}, "unsafe"):
            with self.assertRaisesRegex(ValueError, "probabilities"):
                predictions_from([dict(row, probabilities=probabilities)], demo_cases())

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


class ProbabilityManager(FakeManager):
    """Measures a replayed answer, one metric per forced token, as the runtime does."""
    logprobs = {"allowed": -0.5, "unrelated": -3.0, "unsafe": -1.0}
    reasoning_prefilled = False

    def __init__(self):
        super().__init__()
        self.sampled = 0
        self.probes = 0
        self.replayed = []

    def generate(self, messages, **options):
        self.options = options
        forced = list(options.get("forced_ids", ()))
        if not forced:
            # The runner's one throwaway generation, asking the loaded model
            # whether its prompt already opened a reasoning block.
            self.probes += 1
            try:
                yield SimpleNamespace(text="", metrics=[], prompt_ids=[1, 2],
                                      reasoning_prefilled=self.reasoning_prefilled)
            finally:
                self.closed_streams += 1
            return
        replayed = "".join(map(chr, forced))
        self.replayed.append(replayed)
        label = replayed.removeprefix(REASONING_LEAD)
        lead = len(replayed) - len(label)
        # The label carries the whole log-probability and the reasoning close
        # carries a distinct one, so a scorer that counted the lead would come
        # back with a different number rather than the same one.
        each = math.exp(self.logprobs[label] / len(label))
        metrics = [dict(segment="response", position=index + 1, token_id=token,
                        raw_probability=.5 if index < lead else each, scored=True, raw_rank=1,
                        display_text=chr(token), text=chr(token))
                   for index, token in enumerate(forced)]
        try:
            yield SimpleNamespace(text=replayed, metrics=metrics, prompt_ids=[1, 2],
                                  forced_prefix_tokens=len(forced))
            # Nothing after the replayed answer is ever drawn; a runner that
            # kept reading would be paying for tokens it does not use.
            self.sampled += 1
            yield SimpleNamespace(text=replayed + "!", metrics=metrics, prompt_ids=[1, 2],
                                  forced_prefix_tokens=len(forced))
        finally:
            self.closed_streams += 1


class UnsafeWinsManager(ProbabilityManager):
    """The winning label is the one replayed last, so its trace is the one on screen."""
    logprobs = {"allowed": -3.0, "unrelated": -2.0, "unsafe": -0.5}


class ProbabilityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = ProbabilityManager()
        self.runner = Runner(ModelService(lambda: self.manager), Path(self.directory.name) / "extension")

    def run_batch(self, cases, **options):
        return [frame for frame, _ in self.runner.run("owner", cases, **options)]

    def test_each_label_is_replayed_and_scored_without_sampling_a_token(self):
        final = self.run_batch(demo_cases())[-1]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["scoring"], "probability")
        self.assertEqual(self.manager.sampled, 0)
        # Nine replays, and the one throwaway that asked about reasoning.
        self.assertEqual(self.manager.probes, 1)
        self.assertEqual(self.manager.closed_streams, 10)
        self.assertFalse(final["reasoning_prefilled"])
        self.assertEqual(self.manager.replayed[:3], list(LABELS))
        self.assertEqual(self.manager.options["max_new_tokens"], 1)
        self.assertEqual(self.manager.options["temperature"], 0)
        result = final["predictions"][0]
        self.assertEqual(result["prediction"], "allowed")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
        expected = label_distribution(ProbabilityManager.logprobs)
        for label in LABELS:
            self.assertAlmostEqual(result["probabilities"][label], expected[label])
        self.assertAlmostEqual(result["confidence"], result["probabilities"]["allowed"])
        for label, value in ProbabilityManager.logprobs.items():
            self.assertAlmostEqual(result["logprobs"][label], value)
        # The kept trace is the winning answer's own tokens, so the strip shows
        # what the model gave the judgment it made.
        self.assertEqual(len(result["metrics"]), len("allowed"))

    def test_a_thinking_template_gets_its_reasoning_closed_before_each_label(self):
        self.manager.reasoning_prefilled = True
        final = self.run_batch(demo_cases())[-1]
        self.assertTrue(final["reasoning_prefilled"])
        self.assertTrue(all(row["reasoning_prefilled"] for row in final["predictions"]))
        # Every replay is the close and then the bare label, so the label is
        # the model's answer rather than the opening words of its reasoning.
        self.assertEqual(self.manager.replayed[:3],
                         [REASONING_LEAD + label for label in LABELS])
        # The close is the same three times over, so it stays out of the score:
        # the numbers are the ones a model without a thinking template gives.
        result = final["predictions"][0]
        for label, value in ProbabilityManager.logprobs.items():
            self.assertAlmostEqual(result["logprobs"][label], value)
        self.assertEqual(result["answer_tokens"], {label: len(label) for label in LABELS})
        self.assertEqual(len(result["metrics"]), len("allowed"))
        self.assertEqual("".join(metric["text"] for metric in result["metrics"]), "allowed")

    def test_scores_carry_a_curve_and_an_interval_the_summary_can_show(self):
        final = self.run_batch(demo_cases())[-1]
        scores = final["scores"]
        self.assertEqual(scores["accuracy"], 1 / 3)
        self.assertEqual(len(scores["accuracy_interval"]), 2)
        self.assertLess(scores["accuracy_interval"][0], 1 / 3)
        # Every case gets the same distribution here, so no threshold separates
        # the unsafe one: a curve that claimed otherwise would be wrong.
        self.assertEqual(scores["blocking"]["auc"], .5)
        self.assertEqual(scores["blocking"]["unsafe"], 1)

    def test_a_case_without_state_text_is_skipped_and_the_rest_still_run(self):
        cases = demo_cases()
        cases[1] = dict(cases[1], state_text="")
        final = self.run_batch(cases)[-1]
        self.assertEqual(final["status"], "completed")
        self.assertEqual([row["status"] for row in final["predictions"]],
                         ["completed", "skipped", "completed"])
        self.assertIn("state_text", final["predictions"][1]["feedback"])
        self.assertEqual(final["scores"]["completed"], 2)
        self.assertEqual(final["scores"]["skipped"], 1)
        self.assertEqual(final["scores"]["total"], 3)
        self.assertEqual(self.manager.closed_streams, 7)

    def test_cancelling_between_labels_leaves_the_case_unscored(self):
        stream = self.runner.run("owner", demo_cases())
        next(stream)
        next(stream)
        self.runner.cancel("owner")
        run = [frame for frame, _ in stream][-1]
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["predictions"][0]["status"], "cancelled")
        self.assertIsNone(run["predictions"][0]["prediction"])
        self.assertEqual(run["scores"]["completed"], 0)
        self.assertFalse(self.manager.busy)

    def test_an_unmeasured_replay_is_an_error_rather_than_a_confident_guess(self):
        def unmeasured(self, messages, **options):
            yield SimpleNamespace(text="", metrics=[], prompt_ids=[])

        with mock.patch.object(ProbabilityManager, "generate", unmeasured):
            with self.assertRaisesRegex(ValueError, "did not measure"):
                self.run_batch(demo_cases())
        self.assertFalse(self.manager.busy)

    def test_an_unknown_mode_never_claims_the_model(self):
        with self.assertRaisesRegex(ValueError, "probability or judgment"):
            self.run_batch(demo_cases(), mode="multimodal")
        self.assertEqual(self.manager.releases, 0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = JudgmentManager()
        self.runner = Runner(ModelService(lambda: self.manager), Path(self.directory.name) / "new" / "extension")
        self.batch = partial(self.runner.run, mode="judgment")

    @staticmethod
    def case_paths(frames):
        """The checkpoint offered at each case boundary, ignoring token updates."""
        return [path for frame, path in frames if not isinstance(frame, StreamingResponse)]

    def test_batch_saves_reproducible_private_checkpoint_and_releases_model(self):
        frames = list(self.batch("owner", demo_cases()))
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
            stream = self.batch("owner", demo_cases())
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

    def test_checkpoints_are_paced_by_what_they_cost_to_write(self):
        from extensions.osguard import runner as module

        # A checkpoint rewrites the run, traces and all. Three cases inside one
        # interval are worth one write, not three.
        with mock.patch.object(module, "save_json", wraps=module.save_json) as writing:
            frames = list(self.batch("owner", demo_cases()))
        self.assertEqual(writing.call_count, 1)
        final, path = frames[-1]
        self.assertEqual(json.loads(Path(path).read_text())["predictions"][0]["prediction"], "allowed")
        # Until that write lands there is no file, and the view must not offer
        # one: the download is the checkpoint itself.
        self.assertEqual(self.case_paths(frames), [None, None, None, None, path])
        with mock.patch.object(module, "CHECKPOINT_SECONDS", 0), \
                mock.patch.object(module, "CHECKPOINT_SHARE", 0), \
                mock.patch.object(module, "save_json", wraps=module.save_json) as eager:
            frames = list(self.batch("owner", demo_cases()))
        self.assertEqual(eager.call_count, 4)  # One per case, then the final one.
        self.assertEqual(self.case_paths(frames).count(None), 1)

    def test_cancel_keeps_partial_response_without_scoring_it(self):
        stream = self.batch("owner", demo_cases())
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
        stream = self.batch("owner", demo_cases())
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
                list(self.batch("owner", demo_cases()))
        self.assertFalse(self.manager.busy)
        saved = json.loads(next(self.runner.data_dir.glob("*.json")).read_text())
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["predictions"][0]["status"], "error")
        self.assertEqual(saved["scores"]["completed"], 0)

    def test_screenshot_only_case_is_recorded_as_skipped(self):
        run, path = list(self.batch("owner", [dict(demo_cases()[0], state_text="")]))[-1]
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["predictions"][0]["status"], "skipped")
        self.assertEqual(run["scores"]["skipped"], 1)
        self.assertEqual(run["scores"]["completed"], 0)
        self.assertIsNone(run["scores"]["accuracy"])
        self.assertEqual(self.manager.closed_streams, 0)
        self.assertEqual(json.loads(Path(path).read_text())["predictions"][0]["status"], "skipped")

    def test_invalid_sampling_inputs_do_not_claim_model_or_block_the_next_run(self):
        for value in (-1, None, float('nan'), float('inf'), .5, True, '7'):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "Seed must"):
                list(self.batch("owner", demo_cases(), seed=value))
        for value in (-1, 0, None, float('nan'), float('inf'), 1.5, True, 4097):
            with self.subTest(token_limit=value), self.assertRaisesRegex(ValueError, "Maximum answer tokens"):
                list(self.batch("owner", demo_cases(), max_new_tokens=value))
        self.assertEqual(self.manager.releases, 0)
        self.assertFalse(self.runner.data_dir.exists())
        run, _ = list(self.batch("owner", demo_cases(), seed=0))[-1]
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
            loaded = functions["load_cases"](str(case_path), "owner", False, 0.5)
            self.assertEqual(len(loaded), 14)
            frames = list(functions["evaluate"](loaded[0], "owner", "Free-text judgment", 256, 42, False, 0.5))
            self.assertEqual(len(frames[-1]), 13)
            self.assertIn('**Completed:** 0/3', frames[0][2])
            for index in range(6):
                self.assertEqual(frames[1][index], gr.skip())
                self.assertEqual(frames[2][index], gr.skip())
            self.assertNotEqual(frames[3][0], gr.skip())  # Completed case updates batch state.
            exported = functions["export_run"](frames[-1][0])
            # The checkpoint includes prompts and traces beyond the ordinary
            # dataset size. Reopen through the real saved-run callback.
            with mock.patch('extensions.osguard.benchmark.MAX_FILE_BYTES', 100):
                replay = functions["load_cases"](exported, "owner", False, 0.5)
            self.assertEqual(len(replay[1]["predictions"]), 3)
            self.assertEqual(replay[1]["imported_provenance"]["model_id"], "test/model")
            self.assertEqual(replay[1]["imported_provenance"]["scoring"], "judgment")
            self.assertNotIn("metrics", replay[1]["predictions"][0])
            scored = functions["score_external"](exported, loaded[0], "owner", False, 0.5)
            self.assertEqual(len(scored), 12)
            # An external file that says how its judgments were produced keeps
            # saying so once the run is re-exported from it.
            self.assertEqual(scored[0]["imported_provenance"]["scoring"], "judgment")
            self.assertEqual(scored[0]["imported_provenance"]["model_id"], "test/model")
            execution_path = Path(directory) / "execution.json"
            execution_path.write_text(json.dumps([dict(id="one", task_success=True, retry_terminated=False, invariants={"preserved": False})]))
            reviewed = functions["review_executions"](str(execution_path))
            self.assertEqual(reviewed[0][0][2], "unsafe_completion")
            self.assertTrue(Path(reviewed[2]).is_file())

    def test_probability_run_fills_the_curve_and_the_threshold_rescores_it(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ProbabilityManager()
            context = ExtensionContext(ModelService(lambda: manager), TokenInspector(),
                                       Path(directory) / "extension", NavigationService(lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            self.addCleanup(demo.close)
            functions = {function.fn.__name__: function.fn for function in demo.fns.values() if function.fn}
            case_path = Path(directory) / "cases.json"
            # One unsafe case the model is sure about, one it is not, and one
            # ordinary case: enough for a curve with somewhere to put a line.
            cases = [dict(demo_cases()[2], id="clear", label="unsafe"),
                     dict(demo_cases()[2], id="faint", label="unsafe"),
                     dict(demo_cases()[0], id="ordinary", label="allowed")]
            case_path.write_text(json.dumps(cases))
            loaded = functions["load_cases"](str(case_path), "owner", False, 0.5)
            # Every case is scored against the same fake distribution, so the
            # ordering that matters here is the one the page draws, not the
            # model's: check the wiring, not the numbers.
            frames = list(functions["evaluate"](loaded[0], "owner", "Label probabilities", 256, 42, False, 0.5))
            run = frames[-1][0]
            self.assertEqual(run["scoring"], "probability")
            self.assertEqual(len(run["predictions"]), 3)
            self.assertTrue(all("probabilities" in row for row in run["predictions"]))
            table = frames[-1][1]
            self.assertEqual(len(table[0]), 6)
            self.assertEqual(table[0][4], "35.9%")  # The unsafe column, not the winner's.
            self.assertIn("95% CI", frames[-1][2])
            self.assertIn("blocking-tradeoff", frames[-1][4])
            # Blocking everything the model gives any unsafe mass at all turns
            # every judgment into unsafe, and the table has to say so.
            rows, note, rescored, chart = functions["rescore"](loaded[0], run, "owner", True, .2)
            self.assertTrue(all(row[3] == "unsafe" for row in rows))
            self.assertIn("P(unsafe) ≥ 0.20", note)
            self.assertEqual(rescored["confusion"]["allowed"]["unsafe"], 1)
            self.assertEqual(run["predictions"][0]["prediction"], "allowed")
            self.assertEqual(functions["rescore"]([], run, "owner", True, .2)[0], gr.skip())
            self.assertIn("viz-empty", functions["rescore"](loaded[0], {}, "owner", False, 0.5)[3])
            # Zero is a threshold like any other, and the one the curve's
            # block-everything end sits at. Switching blocking off is what asks
            # for each row's own answer, and no number says that.
            rows, note, _, _ = functions["rescore"](loaded[0], run, "owner", True, 0)
            self.assertTrue(all(row[3] == "unsafe" for row in rows))
            self.assertIn("P(unsafe) \u2265 0.00", note)
            rows, note, _, _ = functions["rescore"](loaded[0], run, "owner", False, 0)
            self.assertTrue(all(row[3] == "allowed" for row in rows))
            self.assertIn("model's own top label", note)

    def test_threshold_moved_during_a_batch_survives_the_next_case(self):
        """The slider is a live control, so the batch cannot answer with the old value."""
        with tempfile.TemporaryDirectory() as directory:
            manager = ProbabilityManager()
            context = ExtensionContext(ModelService(lambda: manager), TokenInspector(),
                                       Path(directory) / "extension", NavigationService(lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            self.addCleanup(demo.close)
            functions = {function.fn.__name__: function.fn for function in demo.fns.values() if function.fn}
            case_path = Path(directory) / "cases.json"
            case_path.write_text(json.dumps(demo_cases()))
            loaded = functions["load_cases"](str(case_path), "owner", False, 0.5)
            frames, moved = [], False
            # Start at zero, the way a reader does, and move the slider once the
            # first case has landed. Every frame after that has to agree with it.
            for frame in functions["evaluate"](loaded[0], "owner", "Label probabilities", 256, 42, False, 0.5):
                frames.append(frame)
                batched = isinstance(frame[0], dict) and frame[0].get("predictions")
                if batched and not moved:
                    functions["rescore"](loaded[0], frame[0], "owner", True, .2)
                    moved = True
            self.assertTrue(moved)
            table, note = frames[-1][1], frames[-1][2]
            self.assertIn("P(unsafe) ≥ 0.20", note)
            self.assertTrue(all(row[3] == "unsafe" for row in table))
            # The recorded run still holds the model's own judgment.
            self.assertEqual(frames[-1][0]["predictions"][0]["prediction"], "allowed")
            # Closing the browser session drops the threshold with the rest of
            # its server state, so a long-lived app does not accumulate them.
            owner_state = next(block for block in demo.blocks.values() if isinstance(block, gr.State)
                               and getattr(block.delete_callback, "__name__", "") == "forget")
            owner_state.delete_callback("owner")

    def replaced_label_traces(self, manager):
        """Frames from a one-case probability batch, with the page wired to manager."""
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(ModelService(lambda: manager), TokenInspector(),
                                       Path(directory) / "extension", NavigationService(lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            self.addCleanup(demo.close)
            functions = {function.fn.__name__: function.fn for function in demo.fns.values() if function.fn}
            case_path = Path(directory) / "cases.json"
            case_path.write_text(json.dumps(demo_cases()[:1]))
            loaded = functions["load_cases"](str(case_path), "owner", False, 0.5)
            return list(functions["evaluate"](loaded[0], "owner", "Label probabilities", 256, 42, False, 0.5))

    def test_a_replaced_label_trace_drops_the_token_chosen_from_the_last_one(self):
        """Each replay is a different answer's tokens, not more of the same answer."""
        frames = self.replaced_label_traces(ProbabilityManager())
        # One opening frame, one streamed frame per label, then the completed
        # case and the finished batch.
        streamed = frames[1:1 + len(LABELS)]
        self.assertEqual(len(frames), len(LABELS) + 3)
        for frame in streamed:
            self.assertEqual(frame[10], "Select a generated token.")
            self.assertEqual(frame[11], [])
        # "allowed" wins and "unsafe" was replayed last, so the completed case
        # puts a fourth strip on screen and has to clear the panel again.
        completed = frames[len(LABELS) + 1]
        self.assertEqual(completed[0]["predictions"][0]["scored_label"], "allowed")
        self.assertEqual(completed[10], "Select a generated token.")
        self.assertEqual(completed[11], [])

    def test_a_winning_label_already_on_screen_keeps_the_token_selected_from_it(self):
        """Clearing the panel is for a strip that changed, not for every frame."""
        frames = self.replaced_label_traces(UnsafeWinsManager())
        completed = frames[len(LABELS) + 1]
        # The winner is the label replayed last, so the strip the reader is
        # looking at is already the one the judgment was made on.
        self.assertEqual(completed[0]["predictions"][0]["scored_label"], "unsafe")
        self.assertEqual(completed[10], gr.skip())
        self.assertEqual(completed[11], gr.skip())

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
