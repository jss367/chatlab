"""Numerical and lifecycle checks using small, real Transformers decoders."""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
    Qwen3Config, Qwen3ForCausalLM,
)

import charts
from jacobian_lens import FittedLens
from model_runtime import ModelChanged, ModelManager
from tiny_tokenizer import build


def small_manager(config_type=LlamaConfig, model_type=LlamaForCausalLM):
    torch.manual_seed(42)
    manager = ModelManager()
    manager.tokenizer = build()
    config = config_type(
        vocab_size=len(manager.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    manager.model = model_type(config).eval()
    manager.model_id = "test/tiny-decoder"
    manager.precision = "full"
    return manager


class JacobianLensTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "lens.pt"
        self.manager = small_manager()
        # Deliberately nonsymmetric matrices detect a transposed transport.
        self.data = {
            "J": {layer: torch.randn(16, 16) for layer in (0, 2)},
            "source_layers": [0, 2], "n_prompts": 100, "d_model": 16,
        }
        self.ids = self.manager.tokenizer.encode("the cat sat on the mat")
        self.save()

    def save(self):
        torch.save(self.data, self.path)

    def import_lens(self):
        return self.manager.import_jacobian_lens(str(self.path), self.manager.model_id)

    def inspect(self, index, imported=None, **kwargs):
        imported = imported or self.import_lens()
        return self.manager.inspect_jacobian(
            self.ids, index, lens_id=imported["import_id"],
            load_id=self.manager.load_id, **kwargs,
        ).to_dict()

    def test_transport_matches_full_sequence_block_readout_on_supported_models(self):
        for config, model in (
            (LlamaConfig, LlamaForCausalLM), (Qwen2Config, Qwen2ForCausalLM),
            (Qwen3Config, Qwen3ForCausalLM),
        ):
            with self.subTest(model=model.__name__):
                self.manager = small_manager(config, model)
                index = 2
                captured = {}
                handles = []
                for layer in self.data["J"]:
                    def capture(_module, _inputs, output, layer=layer):
                        captured[layer] = output[0, index].detach().clone()
                    handles.append(self.manager.model.model.layers[layer].register_forward_hook(capture))
                with torch.no_grad():
                    self.manager.model(torch.tensor([self.ids]), use_cache=False)
                for handle in handles:
                    handle.remove()
                pin = self.manager.tokenizer.decode([self.ids[index]])
                result = self.inspect(index, pinned_text=pin)
                self.assertEqual(result["token_id"], self.ids[index])
                self.assertEqual(result["pinned_id"], self.ids[index])
                for row in result["layers"]:
                    with torch.no_grad():
                        transported = captured[row["layer"]] @ self.data["J"][row["layer"]].T
                        expected = self.manager.model.lm_head(self.manager.model.model.norm(transported))
                    values, ids = expected.topk(5)
                    self.assertEqual([c["token_id"] for c in row["candidates"]], ids.tolist())
                    np.testing.assert_allclose([c["score"] for c in row["candidates"]], values.numpy(), atol=1e-6)
                    self.assertEqual(row["rank"], int((expected > expected[self.ids[index]]).sum()) + 1)
                    self.assertAlmostEqual(row["score"], float(expected[self.ids[index]]), places=6)

    def test_first_token_is_readable_and_later_tokens_do_not_affect_it(self):
        imported = self.import_lens()
        before = self.inspect(0, imported)
        self.ids[1:] = [0] * (len(self.ids) - 1)
        after = self.inspect(0, imported)
        self.assertEqual(before, after)

    def test_cache_reuse_in_both_directions_and_switching_from_logit_lens(self):
        imported = self.import_lens()
        expected = self.inspect(2, imported)
        self.inspect(4, imported)
        self.assertEqual(expected, self.inspect(2, imported))
        self.manager.inspect(self.ids, 2, load_id=self.manager.load_id)
        actual = self.inspect(2, imported)
        for left, right in zip(expected["layers"], actual["layers"]):
            np.testing.assert_allclose(
                [c["score"] for c in left["candidates"]],
                [c["score"] for c in right["candidates"]], atol=1e-6,
            )

    def test_reloading_or_replacing_lens_invalidates_previous_import(self):
        old = self.import_lens()
        self.import_lens()
        with self.assertRaisesRegex(ValueError, "Import a Jacobian"):
            self.inspect(1, old)
        fresh = self.import_lens()
        self.manager.load_count += 1
        with self.assertRaisesRegex(ValueError, "Import a Jacobian"):
            self.inspect(1, fresh)
        with self.assertRaises(ModelChanged):
            self.manager.inspect_jacobian(self.ids, 1, lens_id=fresh["import_id"], load_id=fresh["load_id"])
        self.manager.unload()
        self.assertIsNone(self.manager._jacobian_lens)

    def test_rejects_bad_artifacts_and_preserves_previous_import(self):
        imported = self.import_lens()
        original = dict(self.data)
        cases = [
            {"d_model": 17}, {"n_prompts": 0}, {"model_id": "other/model"},
            {"model_revision": "unknown"}, {"source_layers": [0]},
            {"J": {3: torch.eye(16)}, "source_layers": [3]},
            {"J": {0: torch.ones(16, 15)}, "source_layers": [0]},
            {"J": {0: torch.full((16, 16), float("nan"))}, "source_layers": [0]},
            {"J": {0: torch.ones(16, 16, dtype=torch.int32)}, "source_layers": [0]},
            {"J": {}},
        ]
        for change in cases:
            with self.subTest(change=list(change)):
                self.data = original | change
                # Atomic replacement avoids modifying an existing memory map.
                replacement = self.path.with_suffix(".new")
                torch.save(self.data, replacement)
                replacement.replace(self.path)
                with self.assertRaises(ValueError):
                    self.import_lens()
                self.inspect(1, imported)

    def test_requires_matching_declared_model_and_supported_backend(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.manager.import_jacobian_lens(str(self.path), "other/model")
        self.manager.engine = mock.Mock(backend="mlx")
        with self.assertRaisesRegex(ValueError, "MLX"):
            self.import_lens()
        self.manager.engine = None
        self.manager.model.config.model_type = "unknown"
        with self.assertRaisesRegex(ValueError, "supports Llama"):
            self.import_lens()
        self.manager.model.config.model_type = "llama"
        self.manager.precision = "4-bit"
        with self.assertRaisesRegex(ValueError, "full-precision"):
            self.import_lens()

    def test_multitoken_pins_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one vocabulary token"):
            self.inspect(1, pinned_text="a whole sentence")

    def test_unrecognized_output_transform_withholds_readout(self):
        imported = self.import_lens()
        forward = self.manager.model.forward

        def transformed(*args, **kwargs):
            output = forward(*args, **kwargs)
            output.logits = output.logits + 1
            return output

        with mock.patch.object(self.manager.model, "forward", transformed):
            with self.assertRaisesRegex(ValueError, "withheld"):
                self.inspect(1, imported)

    def test_hooks_removed_after_forward_failure(self):
        lens = FittedLens.load(self.path, self.manager._engine(), self.manager.model_id, self.manager.model_id)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with lens.record(self.manager._engine()):
                raise RuntimeError("failed")
        self.assertTrue(all(not block._forward_hooks for block in self.manager.model.model.layers))

    def test_rendering_labels_position_scores_and_pin_and_escapes_text(self):
        result = self.inspect(1, pinned_text="the")
        result["token_text"] = "<script>"
        result["layers"][0]["candidates"][0]["text"] = "<img>"
        rendered = charts.jacobian_lens_chart(result)
        self.assertIn("Jacobian lens after", rendered)
        self.assertIn("Pinned rank", rendered)
        self.assertIn("not generation probabilities", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<img>", rendered)
        self.assertIn("<svg", rendered)

    def test_ui_reads_first_prompt_token_after_it_and_labels_the_result(self):
        from ui import inspection, runtime

        imported = self.import_lens()
        metrics = [{"token_id": token} for token in self.ids[2:]]
        prompt = [{"token_id": token} for token in self.ids[:2]]
        args = (
            {"generation": 7, "strip": "prompt", "index": 0},
            (7, metrics), (7, prompt), (7, self.ids[:2], self.manager.load_id), 0,
        )
        with mock.patch.object(runtime, "MANAGER", self.manager), mock.patch.object(
            inspection, "current_strip_generation", return_value=7,
        ):
            result = list(inspection.inspect_layers(
                *args, lens_mode="Jacobian", imported_lens=imported,
            ))[-1]
            self.assertIn("jacobian-lens", result[0])
            self.assertEqual(result[3]["token_id"], self.ids[0])
            self.assertIn("after processing this token", result[4])
            self.assertIsNone(self.manager.occupant)
            logit = list(inspection.inspect_layers(*args))[-1]
            self.assertEqual(logit[-1], inspection.INSPECT_FIRST)

    def test_ui_refuses_import_when_busy_and_releases_reservation_on_errors(self):
        from ui import inspection, runtime

        with mock.patch.object(runtime, "MANAGER", self.manager):
            self.manager.claim_generation()
            result = inspection.import_jacobian_lens(self.path, self.manager.model_id)
            self.assertEqual(result[1], inspection.INSPECT_BUSY)
            self.manager.release_generation()
            result = inspection.import_jacobian_lens(self.path, "other/model")
            self.assertIn("Could not import", result[1])
            self.assertIsNone(self.manager.occupant)
            imported, status = inspection.import_jacobian_lens(self.path, self.manager.model_id)
            self.assertIn("import_id", imported)
            self.assertIn("2 fitted layers", status)


    @contextmanager
    def controlled_ui(self, mode="Jacobian", pin="the"):
        from ui import inspection, runtime

        session = inspection.INSPECTION_CONTROLS.new_session()
        imported = self.import_lens()
        inspection.change_lens_mode(mode, session)
        inspection.change_pinned_token(pin, session)
        args = (
            {"generation": 7, "strip": "prompt", "index": 1},
            (7, [{"token_id": token} for token in self.ids[2:]]),
            (7, [{"token_id": token} for token in self.ids[:2]]),
            (7, self.ids[:2], self.manager.load_id), 0,
        )
        def request():
            return inspection.inspect_layers(
                *args, lens_mode=mode, imported_lens=imported, pinned_text=pin,
                inspection_session=session,
            )

        try:
            with mock.patch.object(runtime, "MANAGER", self.manager), mock.patch.object(
                inspection, "current_strip_generation", return_value=7,
            ):
                yield request, session
        finally:
            inspection.INSPECTION_CONTROLS.forget(session)

    def test_control_edits_during_a_pass_discard_results_and_errors(self):
        import gradio as gr
        from ui import inspection

        for mode, change in (("Logit", "mode"), ("Jacobian", "mode"), ("Jacobian", "pin")):
            for fails in (False, True):
                with self.subTest(mode=mode, change=change, fails=fails), self.controlled_ui(mode) as (request, session):
                    method = "inspect" if mode == "Logit" else "inspect_jacobian"
                    original = getattr(self.manager, method)

                    def edited(*args, **kwargs):
                        if change == "mode":
                            inspection.change_lens_mode("Jacobian" if mode == "Logit" else "Logit", session)
                        else:
                            inspection.change_pinned_token(" cat", session)
                        if fails:
                            raise RuntimeError("An error from the obsolete request")
                        return original(*args, **kwargs)

                    with mock.patch.object(self.manager, method, side_effect=edited):
                        self.assertEqual(list(request()), [(gr.skip(),) * 5])
                    self.assertIsNone(self.manager.occupant)

    def test_control_edits_during_delivery_remove_the_old_frame(self):
        import gradio as gr
        from ui import inspection

        for change in ("mode", "pin"):
            with self.subTest(change=change), self.controlled_ui() as (request, session):
                frames = request()
                first = next(frames)
                self.assertIn("jacobian-lens", first[0])
                if change == "mode":
                    inspection.change_lens_mode("Logit", session)
                else:
                    inspection.change_pinned_token(" cat", session)
                self.assertEqual(next(frames), ("", charts.EMPTY_ATTENTION, gr.skip(), None, inspection.INSPECT_HINT))
                self.assertEqual(list(frames), [])
                self.assertIsNone(self.manager.occupant)
                self.assertEqual(inspection.render_attention(first[3], 0), gr.skip())

    def test_control_edits_before_a_queued_request_starts_reject_its_old_inputs(self):
        import gradio as gr
        from ui import inspection

        for change in ("mode", "pin"):
            with self.subTest(change=change), self.controlled_ui() as (request, session):
                frames = request()
                if change == "mode":
                    inspection.change_lens_mode("Logit", session)
                else:
                    inspection.change_pinned_token(" cat", session)
                with mock.patch.object(self.manager, "inspect_jacobian") as run:
                    self.assertEqual(list(frames), [(gr.skip(),) * 5])
                    run.assert_not_called()
                self.assertIsNone(self.manager.occupant)

    def test_another_sessions_controls_do_not_invalidate_this_inspection(self):
        from ui import inspection

        with self.controlled_ui() as (request, session):
            other = inspection.INSPECTION_CONTROLS.new_session()
            try:
                frames = request()
                first = next(frames)
                inspection.change_pinned_token(" cat", other)
                inspection.change_lens_mode("Jacobian", other)
                self.assertEqual(list(frames), [])
                self.assertIn("jacobian-lens", first[0])
                inspection.INSPECTION_CONTROLS.forget(session)
                self.assertFalse(inspection.INSPECTION_CONTROLS.current(session, first[3]["inspection_controls"]["revision"]))
            finally:
                inspection.INSPECTION_CONTROLS.forget(other)


if __name__ == "__main__":
    unittest.main()
