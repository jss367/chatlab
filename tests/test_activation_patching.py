import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import activation_patching as patching
from model_runtime import ModelChanged, ModelManager, OutOfMemoryError
from test_streaming import FakeTokenizer, PIECES
from ui import activation_patching as controls


def recorded(manager, context, output=(2, 3), name="run"):
    return {
        "run_id": name, "model_id": manager.model_id, "load_id": manager.load_id,
        "context_ids": list(context), "settings": {},
        "metrics": [{"token_id": v, "text": f"token {v}"} for v in output],
    }


class CausalBlock(torch.nn.Module):
    def forward(self, hidden):
        # Earlier positions influence later ones, never the reverse.
        return (hidden + hidden.cumsum(dim=1) / 4,)


class CausalModel(torch.nn.Module):
    """A small deterministic network with analytically testable causality."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", max_position_embeddings=64)
        self.embedding = torch.nn.Embedding(len(PIECES), len(PIECES))
        self.head = torch.nn.Linear(len(PIECES), len(PIECES), bias=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([CausalBlock(), CausalBlock()])
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(len(PIECES)))
            self.head.weight.copy_(torch.eye(len(PIECES)))
        self.calls = []
        self.fail_at = None
        self.eval()

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
        self.calls.append((input_ids.tolist()[0], use_cache, torch.is_inference_mode_enabled()))
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected forward failure")
        return SimpleNamespace(logits=self.head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden))


def manager_with(model=None):
    manager = ModelManager()
    manager.model = model if model is not None else CausalModel()
    manager.tokenizer = FakeTokenizer()
    manager.model_id = "test/model"
    return manager


class PatchingTests(unittest.TestCase):
    def setUp(self):
        self.manager = manager_with()
        self.donor = recorded(self.manager, [1, 2, 2], name="source")
        self.recipient = recorded(self.manager, [1, 4, 4], name="recipient")

    def plan(self, **kwargs):
        return patching.experiment(self.donor, self.recipient,
                                   kwargs.get("target", 0), kwargs.get("source_count", 0), kwargs.get("width", 3))

    def test_exact_prefixes_exclude_recipient_target_and_future(self):
        plan = self.plan(target=1, source_count=1, width=2)
        self.assertEqual(plan["donor_ids"], [1, 2, 2, 2])
        self.assertEqual(plan["recipient_ids"], [1, 4, 4, 2])
        self.assertEqual(plan["target_id"], 3)
        self.assertEqual(plan["pairs"], [
            {"donor_position": 2, "recipient_position": 2},
            {"donor_position": 3, "recipient_position": 3},
        ])

    def test_different_lengths_are_explicitly_paired_from_end(self):
        self.donor["context_ids"].insert(0, 0)
        plan = self.plan(width=2)
        self.assertEqual(plan["pairs"][0], {"donor_position": 2, "recipient_position": 1})

    def test_same_run_has_no_effect_and_no_hooks_or_inference_context_leak(self):
        plan = patching.experiment(self.donor, self.donor, 0, 0, 3)
        stream = patching.measure(self.manager.model, plan)
        for result in stream:
            self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(all(not b._forward_hooks for b in self.manager.model.model.layers))
            if "delta_probability" in result:
                self.assertAlmostEqual(result["delta_probability"], 0, places=7)

    def test_causal_positions_and_final_vector_replacement(self):
        readings = list(patching.measure(self.manager.model, self.plan()))
        baseline, *cells = readings
        # At an earlier block, replacing earlier context affects the answer.
        self.assertGreater(cells[1]["delta_probability"], 0)
        # At the last block there is no further cross-token mixing.
        self.assertEqual(cells[4]["delta_probability"], 0)
        # Replacing the final position after the final block recovers source logits.
        self.assertAlmostEqual(cells[-1]["probability"], baseline["donor_baseline"]["probability"], places=7)
        self.assertTrue(all(not use_cache and inference for _, use_cache, inference in self.manager.model.calls))
        self.assertEqual([ids for ids, _, _ in self.manager.model.calls][1:], [[1, 4, 4]] * 7)

    def test_recovery_is_zero_at_recipient_and_one_at_source(self):
        baseline, *cells = patching.measure(self.manager.model, self.plan())
        self.assertTrue(baseline["recovery_defined"])
        self.assertAlmostEqual(baseline["recovery_gap"],
                               baseline["donor_baseline"]["metric"] - baseline["baseline"]["metric"])
        self.assertEqual(cells[4]["recovery"], 0)
        self.assertAlmostEqual(cells[-1]["recovery"], 1, places=6)
        # Without a contrast the metric is the answer's log probability.
        self.assertEqual(self.plan()["metric"], "answer_log_probability")
        self.assertEqual(cells[1]["delta_metric"], cells[1]["delta_log_probability"])

    def test_contrast_token_measures_logit_difference(self):
        plan = patching.experiment(self.donor, self.recipient, 0, 0, 3, 1)
        self.assertEqual((plan["contrast_index"], plan["contrast_id"], plan["metric"]), (1, 3, "logit_difference"))
        baseline, *cells = patching.measure(self.manager.model, plan)
        for reading in (baseline["baseline"], baseline["donor_baseline"], *cells):
            self.assertAlmostEqual(reading["metric"],
                                   reading["log_probability"] - reading["contrast_log_probability"], places=9)
        # With identity embeddings and head the logit difference is readable directly.
        with torch.no_grad():
            hidden = self.manager.model.embedding(torch.tensor([plan["recipient_ids"]]))
            for layer in self.manager.model.model.layers:
                hidden = layer(hidden)[0]
        logits = hidden[0, -1]
        self.assertAlmostEqual(baseline["baseline"]["metric"], float(logits[2] - logits[3]), places=5)
        self.assertAlmostEqual(cells[-1]["recovery"], 1, places=6)
        self.assertIsNone(patching.experiment(self.donor, self.recipient, 0, 0, 3, -1)["contrast_id"])
        for index, message in ((0, "differs from the answer"), (2, "from the source run"), (0.5, "whole number")):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                patching.experiment(self.donor, self.recipient, 0, 0, 3, index)

    def test_recovery_undefined_when_runs_agree(self):
        plan = patching.experiment(self.donor, self.donor, 0, 0, 3, 1)
        baseline, *cells = patching.measure(self.manager.model, plan)
        self.assertFalse(baseline["recovery_defined"])
        self.assertTrue(all(cell["recovery"] is None for cell in cells))

    def test_failure_and_cancellation_remove_hooks(self):
        for fail_at in (1, 3):
            model = CausalModel()
            model.fail_at = fail_at
            with self.assertRaisesRegex(RuntimeError, "injected"):
                list(patching.measure(model, self.plan()))
            self.assertTrue(all(not b._forward_hooks for b in model.model.layers))
        stream = self.manager.patch_activations(self.donor, self.recipient, 0, 0, 2)
        next(stream)
        self.assertTrue(self.manager._lock.locked())
        next(stream)
        stream.close()
        self.assertFalse(self.manager._lock.locked())
        self.assertTrue(all(not b._forward_hooks for b in self.manager.model.model.layers))

    def test_rejects_wrong_load_missing_ids_and_active_steering(self):
        for key, value, message in (
            ("load_id", "other", "same model load"),
            ("model_id", "other/model", "same model load"),
            ("run_id", "", "exact input tokens"),
            ("settings", {"steering": {"enabled": True, "strength": 1}}, "steering off"),
        ):
            donor = dict(self.donor, **{key: value})
            with self.assertRaisesRegex(ValueError, message):
                patching.experiment(donor, self.recipient, 0, 0, 2)
        self.manager.load_count += 1
        with self.assertRaises(ModelChanged):
            list(self.manager.patch_activations(self.donor, self.recipient, 0, 0, 2))

    def test_device_cache_released_after_completion_failure_and_cancellation(self):
        for outcome in ("complete", "cancel_baseline", "cancel_cell", "capture_failure", "patch_failure"):
            with self.subTest(outcome=outcome):
                manager = manager_with()
                donor = recorded(manager, [1, 2, 2])
                recipient = recorded(manager, [1, 4, 4])
                released = []

                def release():
                    self.assertTrue(manager._lock.locked())
                    self.assertTrue(all(not b._forward_hooks for b in manager.model.model.layers))
                    released.append(True)

                with mock.patch.object(manager, "_release_device_cache", side_effect=release):
                    stream = manager.patch_activations(donor, recipient, 0, 0, 2)
                    if outcome == "complete":
                        list(stream)
                    elif outcome.startswith("cancel"):
                        next(stream)
                        if outcome == "cancel_cell":
                            next(stream)
                        self.assertEqual(released, [])
                        stream.close()
                    else:
                        manager.model.fail_at = 1 if outcome == "capture_failure" else 3
                        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                            list(stream)
                    self.assertEqual(released, [True])
                    self.assertFalse(manager._lock.locked())

    def test_runtime_out_of_memory_is_translated_and_cache_released(self):
        for error in (RuntimeError("CUDA out of memory"),
                      RuntimeError("MPS backend out of memory"), MemoryError()):
            with self.subTest(error=repr(error)), \
                    mock.patch.object(self.manager.model, "forward", side_effect=error), \
                    mock.patch.object(self.manager, "_release_device_cache") as release:
                with self.assertRaises(OutOfMemoryError):
                    list(self.manager.patch_activations(self.donor, self.recipient, 0, 0, 2))
                release.assert_called_once_with()
                self.assertFalse(self.manager._lock.locked())
                self.assertTrue(all(not b._forward_hooks for b in self.manager.model.model.layers))

    def test_bounds_and_backend_refusals(self):
        for target, source_count, width in ((-1, 0, 1), (2, 0, 1), (0, 3, 1),
                                             (0, 0, 0), (0, 0, 33), (0, 0, 4),
                                             (0.5, 0, 1), (0, -1, 1)):
            with self.subTest(values=(target, source_count, width)), self.assertRaises(ValueError):
                patching.experiment(self.donor, self.recipient, target, source_count, width)
        with self.assertRaisesRegex(ValueError, "prefixes up to"):
            patching.experiment(dict(self.donor, context_ids=[1] * 2049), self.recipient, 0, 0, 1)
        self.manager.engine = SimpleNamespace(backend="mlx")
        with self.assertRaisesRegex(ValueError, "MLX"):
            list(self.manager.patch_activations(self.donor, self.recipient, 0, 0, 2))
        self.manager.engine = None
        self.manager.model.config.model_type = "unknown"
        with self.assertRaisesRegex(ValueError, "currently supports"):
            list(self.manager.patch_activations(self.donor, self.recipient, 0, 0, 2))

    def test_actual_transformers_architectures(self):
        # Local random weights; no downloads. Verify the declared supported
        # architectures, including their real block output conventions.
        from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                                  Olmo3Config, Olmo3ForCausalLM)
        for config_class, model_class in ((LlamaConfig, LlamaForCausalLM),
                                          (Qwen2Config, Qwen2ForCausalLM),
                                          (Olmo3Config, Olmo3ForCausalLM)):
            with self.subTest(model=model_class.__name__), torch.random.fork_rng():
                torch.manual_seed(7)
                config = config_class(vocab_size=16, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64, eos_token_id=15, pad_token_id=0)
                model = model_class(config).eval()
                results = list(patching.measure(model, self.plan()))
                self.assertAlmostEqual(results[-1]["probability"], results[0]["donor_baseline"]["probability"], places=6)
                self.assertAlmostEqual(results[-2]["delta_probability"], 0, places=7)
                self_plan = patching.experiment(self.donor, self.donor, 0, 0, 2)
                for cell in list(patching.measure(model, self_plan))[1:]:
                    self.assertAlmostEqual(cell["delta_probability"], 0, places=7)


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.manager = manager_with()
        self.donor = recorded(self.manager, [1, 2, 2], name="source")
        self.recipient = recorded(self.manager, [1, 4, 4], name="recipient")
        self.session = controls.CONTROLS.new()
        self.addCleanup(controls.CONTROLS.forget, self.session)
        patch = mock.patch.object(controls.runtime, "MANAGER", self.manager)
        patch.start()
        self.addCleanup(patch.stop)

    def stream(self, contrast=controls.NO_CONTRAST):
        return controls.run(self.donor, self.recipient, "A → B", 0, contrast, 0, 2, self.session, 0)

    def test_result_and_export_include_provenance_probabilities_and_pairing(self):
        frames = list(self.stream())
        status, recovery, chart, rows, result, _run, _stop = frames[-1]
        self.assertIn("Complete", status)
        self.assertTrue(result["complete"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(result["donor_run_id"], "source")
        self.assertEqual(result["recipient_ids"], [1, 4, 4])
        self.assertEqual(result["pairs"][0]["donor_position"], 1)
        self.assertIn("aria-label", chart)
        self.assertIn("Fraction of the source", recovery)
        self.assertEqual(len(rows[0]), len(controls.HEADERS))
        self.assertFalse(self.manager.busy)
        path = Path(controls.download(result))
        try:
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        finally:
            path.unlink()
            path.parent.rmdir()

    def test_stop_close_invalidated_controls_and_busy_release_reservation(self):
        stream = self.stream()
        next(stream)
        self.assertTrue(self.manager.busy)
        next(stream)
        controls.stop(self.session)
        cleanup = list(stream)
        self.assertEqual(len(cleanup), 1)
        self.assertIsNone(cleanup[0][4])
        self.assertFalse(self.manager.busy)
        self.assertFalse(self.manager._lock.locked())
        self.assertEqual(len(list(self.stream())), 1)  # stale queued request
        self.assertFalse(self.manager.busy)

    def test_cancelling_at_first_frame_and_error_restore_controls(self):
        stream = self.stream()
        next(stream)
        stream.close()
        self.assertFalse(self.manager.busy)
        self.manager.model.fail_at = 3
        frames = list(self.stream())
        self.assertIn("Could not patch", frames[-1][0])
        self.assertTrue(frames[-1][5]["interactive"])
        self.assertFalse(self.manager.busy)
        self.assertFalse(self.manager._lock.locked())

    def test_losing_claim_does_not_release_winners_reservation(self):
        self.assertIsNone(self.manager.claim_generation())
        try:
            frames = list(self.stream())
            self.assertIn("Wait", frames[0][0])
            self.assertTrue(self.manager.busy)
        finally:
            self.manager.release_generation()

    def test_heatmap_escapes_model_text_and_marks_unmeasured_cells(self):
        frames = list(self.stream())
        result = copy.deepcopy(frames[-1][4])
        result["pairs"][0]["recipient_text"] = '<img src=x onerror="bad()">'
        result["cells"] = result["cells"][:1]
        chart = controls.heatmap(result)
        self.assertNotIn("<img", chart)
        self.assertIn("&lt;img", chart)
        self.assertIn("Not measured", chart)

    def test_contrast_reaches_status_and_recovery_view(self):
        status, recovery, *_rest, result, _run, _stop = list(self.stream(contrast=1))[-1]
        self.assertEqual(result["contrast_text"], self.manager._decode_token(3) or self.manager._token_fallback(3))
        self.assertIn("− log p(", status)
        self.assertIn("+1.00", recovery)
        undefined = dict(result, recovery_defined=False)
        self.assertIn("Recovery is undefined", controls.heatmap(undefined, controls.RECOVERY))

    def test_default_contrast_is_source_next_token_unless_it_is_the_answer(self):
        self.assertEqual(controls.default_contrast(self.donor, self.recipient, 1, 0), 0)
        # The source's first token equals the recipient's first answer token.
        self.assertEqual(controls.default_contrast(self.donor, self.recipient, 0, 0), controls.NO_CONTRAST)
        self.assertEqual(controls.default_contrast(self.donor, self.recipient, 0, 1), 1)
        self.assertEqual(controls.default_contrast(self.donor, self.recipient, 0, 2), controls.NO_CONTRAST)
        self.assertEqual(controls.default_contrast(None, None, None, 0), controls.NO_CONTRAST)

    def test_changing_answer_token_rechooses_contrast(self):
        def contrast(target, donor_count=0):
            update, *_reset = controls.prefix_changed(
                self.donor, self.recipient, "A → B", target, donor_count, self.session)
            return update["value"]
        self.assertEqual(contrast(1), 0)
        self.assertEqual(contrast(0), controls.NO_CONTRAST)
        self.assertEqual(contrast(0, donor_count=1), 1)


if __name__ == "__main__":
    unittest.main()
