"""Numerical and lifecycle checks using small, real Transformers decoders."""

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
)

import charts
import jacobian_lens
from jacobian_lens import FittedLens
from model_runtime import ModelChanged, ModelManager
from test_mlx_runtime import needs_mlx
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
            (LlamaConfig, LlamaForCausalLM), (MistralConfig, MistralForCausalLM),
            (Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM),
        ):
            with self.subTest(model=model.__name__):
                self.manager = small_manager(config, model)
                index = 2
                captured = {}
                handles = []
                for layer in self.data["J"]:
                    def capture(_module, _inputs, output, layer=layer):
                        captured[layer] = output[0].detach().clone()
                    handles.append(self.manager.model.model.layers[layer].register_forward_hook(capture))
                with torch.no_grad():
                    reference = self.manager.model(torch.tensor([self.ids]), use_cache=False).logits[0]
                for handle in handles:
                    handle.remove()
                pin = self.manager.tokenizer.decode([self.ids[index]])
                result = self.inspect(index, pinned_text=pin)
                self.assertEqual(result["token_id"], self.ids[index])
                self.assertEqual(result["pinned_id"], self.ids[index])
                window = result["slice"]
                self.assertEqual([token["index"] for token in window["tokens"]], [0, 1, 2])
                self.assertEqual([token["token_id"] for token in window["tokens"]], self.ids[:3])
                for row, column in zip(result["layers"], window["layers"]):
                    self.assertEqual(column["layer"], row["layer"])
                    with torch.no_grad():
                        transported = captured[row["layer"]] @ self.data["J"][row["layer"]].T
                        expected = self.manager.model.lm_head(self.manager.model.model.norm(transported))
                    values, ids = expected[index].topk(5)
                    self.assertEqual([c["token_id"] for c in row["candidates"]], ids.tolist())
                    np.testing.assert_allclose([c["score"] for c in row["candidates"]], values.numpy(), atol=1e-6)
                    self.assertEqual(row["rank"], int((expected[index] > expected[index, self.ids[index]]).sum()) + 1)
                    self.assertAlmostEqual(row["score"], float(expected[index, self.ids[index]]), places=6)
                    # Every earlier position reads the state after its own token only.
                    for position, cell in enumerate(column["cells"]):
                        self.assertEqual(cell["token_id"], int(expected[position].argmax()))
                        self.assertAlmostEqual(cell["score"], float(expected[position].max()), places=5)
                        pinned = expected[position, self.ids[index]]
                        self.assertEqual(cell["pinned_rank"], int((expected[position] > pinned).sum()) + 1)
                self.assertEqual(
                    [cell["token_id"] for cell in window["output"]],
                    reference[: index + 1].argmax(dim=-1).tolist(),
                )
                # The model's own row carries the pinned token's rank as well.
                for position, cell in enumerate(window["output"]):
                    pinned = reference[position, self.ids[index]]
                    self.assertEqual(cell["pinned_rank"], int((reference[position] > pinned).sum()) + 1)
                    self.assertAlmostEqual(cell["pinned_score"], float(pinned), places=5)

    def test_conversion_source_name_strips_quantization_suffixes(self):
        self.assertEqual(jacobian_lens.conversion_source_name("mlx-community/Qwen3-4B-Instruct-4bit"), "Qwen3-4B-Instruct")
        self.assertEqual(jacobian_lens.conversion_source_name("mlx-community/Qwen3-0.6B-8bit"), "Qwen3-0.6B")
        self.assertEqual(jacobian_lens.conversion_source_name("mlx-community/Qwen3-4B-4bit-DWQ"), "Qwen3-4B")
        self.assertEqual(jacobian_lens.conversion_source_name("mlx-community/Llama-3.2-1B-bf16"), "Llama-3.2-1B")
        self.assertEqual(jacobian_lens.conversion_source_name("Qwen/Qwen3-0.6B"), "Qwen3-0.6B")

    def test_slice_window_ends_at_the_token_without_changing_its_readout(self):
        imported = self.import_lens()
        full = self.inspect(4, imported)
        self.assertEqual([token["index"] for token in full["slice"]["tokens"]], [0, 1, 2, 3, 4])
        self.assertEqual(full["slice"]["total"], len(self.ids))
        for row, column in zip(full["layers"], full["slice"]["layers"]):
            self.assertEqual(column["cells"][-1]["token_id"], row["candidates"][0]["token_id"])
        windowed = self.inspect(4, imported, positions=2)
        self.assertEqual([token["index"] for token in windowed["slice"]["tokens"]], [3, 4])
        self.assertEqual(len(windowed["slice"]["output"]), 2)
        for left, right in zip(full["layers"], windowed["layers"]):
            self.assertEqual([c["token_id"] for c in left["candidates"]], [c["token_id"] for c in right["candidates"]])
            np.testing.assert_allclose(
                [c["score"] for c in left["candidates"]], [c["score"] for c in right["candidates"]], atol=1e-5,
            )
        for left, right in zip(full["slice"]["layers"], windowed["slice"]["layers"]):
            self.assertEqual([c["token_id"] for c in left["cells"][3:]], [c["token_id"] for c in right["cells"]])

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
        self.manager.engine = mock.Mock(backend="mlx", config={"model_type": "mamba"})
        with self.assertRaisesRegex(ValueError, "supports Llama"):
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

    def test_slice_grid_marks_the_selected_column_and_carries_pinnable_tokens(self):
        result = self.inspect(3, pinned_text="the")
        result["slice"]["tokens"][0]["text"] = "<b>"
        result["slice"]["layers"][0]["cells"][0]["text"] = 'say "hi"'
        rendered = charts.jacobian_lens_chart(result)
        self.assertIn('class="jl-grid"', rendered)
        # Four positions, two fitted blocks and the model's own output row.
        self.assertEqual(rendered.count('class="jl-cell'), 4 * 3)
        self.assertEqual(rendered.count("jl-output"), 1)
        # The selected column: its header and one cell per row.
        self.assertEqual(rendered.count("jl-selected"), 4)
        self.assertIn("--jl-heat:", rendered)
        # Every cell, the output row's included, shows the pinned token's rank.
        self.assertEqual(rendered.count("<sup>"), 4 * 3)
        self.assertIn("tokens 1–4 of 6", rendered)
        self.assertIn("all 6 tokens", charts.jacobian_lens_chart(self.inspect(5)))
        self.assertNotIn("<b>", rendered)
        self.assertIn("&lt;b&gt;", rendered)
        self.assertIn('data-token="&quot;say \\&quot;hi\\&quot;&quot;"', rendered)
        unpinned = charts.jacobian_lens_chart(self.inspect(3))
        self.assertNotIn("<sup>", unpinned)
        self.assertNotIn("--jl-heat", unpinned)
        self.assertIn("Click a cell", unpinned)
        windowed = charts.jacobian_lens_chart(self.inspect(3, positions=2))
        self.assertIn("tokens 3–4 of 6", windowed)

    def test_imported_lens_is_remembered_and_recalled_after_a_reload(self):
        from ui import inspection, runtime

        store = Path(self.directory.name) / "config" / "jacobian_lenses.json"
        prompt = [{"token_id": token} for token in self.ids[:2]]
        metrics = [{"token_id": token} for token in self.ids[2:]]

        def args():
            return (
                {"generation": 7, "strip": "prompt", "index": 1},
                (7, metrics), (7, prompt), (7, self.ids[:2], self.manager.load_id), 0,
            )

        self.manager.model.config._commit_hash = "abc"
        with mock.patch.object(jacobian_lens, "store_path", return_value=store), mock.patch.object(
            runtime, "MANAGER", self.manager,
        ), mock.patch.object(inspection, "current_strip_generation", return_value=7):
            imported, status = inspection.import_jacobian_lens(str(self.path), self.manager.model_id)
            self.assertIn("Remembered", status)
            self.assertEqual(imported["model_revision"], "abc")
            record = jacobian_lens.remembered(self.manager.model_id)
            self.assertEqual(record["model_revision"], "abc")
            kept = Path(record["path"])
            self.assertEqual(kept.parent, store.parent / "lenses" / "uploads")
            self.assertEqual(kept.name, "lens.pt")
            self.assertTrue(kept.is_file())
            self.assertEqual(record["fitted_model_id"], self.manager.model_id)
            self.assertIsNone(jacobian_lens.remembered("other/model"))

            self.manager.load_count += 1
            self.assertIsNone(self.manager.jacobian_lens_import())
            result = list(inspection.inspect_layers(*args(), lens_mode="Jacobian", imported_lens=imported))[-1]
            self.assertIn("jacobian-lens", result[0])
            self.assertIn("remembered lens", result[4])
            self.assertIn("lens.pt", result[4])
            self.assertEqual(self.manager.jacobian_lens_import()["load_id"], self.manager.load_id)
            self.assertIsNone(self.manager.occupant)

            # Once imported, later clicks use it without another recall.
            with mock.patch.object(inspection, "recall_lens") as recall:
                result = list(inspection.inspect_layers(*args(), lens_mode="Jacobian", imported_lens=None))[-1]
                recall.assert_not_called()
            self.assertIn("jacobian-lens", result[0])
            self.assertNotIn("remembered", result[4])

            # The same ID at another revision is other weights: the record is
            # left alone and the message says why.
            self.manager.load_count += 1
            self.manager.model.config._commit_hash = "def"
            result = list(inspection.inspect_layers(*args(), lens_mode="Jacobian", imported_lens=None))[-1]
            self.assertIn("Import a Jacobian", result[4])
            self.assertIn("another revision", result[4])
            self.assertIsNone(self.manager.jacobian_lens_import())
            self.manager.model.config._commit_hash = "abc"

            # A record whose file is gone leaves the usual message.
            self.manager.load_count += 1
            kept.unlink()
            result = list(inspection.inspect_layers(*args(), lens_mode="Jacobian", imported_lens=None))[-1]
            self.assertIn("Import a Jacobian", result[4])
            self.assertIsNone(self.manager.jacobian_lens_import())

    def test_hub_lens_is_fetched_into_the_lens_directory_and_imported(self):
        from ui import inspection, runtime

        store = Path(self.directory.name) / "config" / "jacobian_lenses.json"
        calls = {}

        def fake_download(repository, filename, local_dir):
            calls.update(repository=repository, filename=filename, local_dir=local_dir)
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.path, target)
            return str(target)

        metadata = mock.Mock(size=self.path.stat().st_size)
        with mock.patch.object(jacobian_lens, "store_path", return_value=store), mock.patch.object(
            runtime, "MANAGER", self.manager,
        ), mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_download), mock.patch(
            "huggingface_hub.get_hf_file_metadata", return_value=metadata,
        ):
            imported, status = inspection.import_jacobian_lens(
                None, self.manager.model_id, " org/lenses ", "lenses/tiny.pt",
            )
            self.assertIn("import_id", imported)
            self.assertIn("2 fitted layers", status)
            self.assertEqual((calls["repository"], calls["filename"]), ("org/lenses", "lenses/tiny.pt"))
            self.assertEqual(Path(calls["local_dir"]), store.parent / "lenses" / "org" / "lenses")
            record = jacobian_lens.remembered(self.manager.model_id)
            self.assertEqual((record["repository"], record["filename"]), ("org/lenses", "lenses/tiny.pt"))
            self.assertTrue(Path(record["path"]).is_file())

            metadata.size = jacobian_lens.MAX_FILE_BYTES + 1
            _, status = inspection.import_jacobian_lens(None, self.manager.model_id, "org/lenses", "lenses/tiny.pt")
            self.assertIn("Could not fetch", status)
            self.assertIn("2 GiB", status)
            self.assertIsNone(self.manager.occupant)
        for repository, filename in (("lenses", "a.pt"), ("org/lenses", "../a.pt"), ("org/lenses", "a.bin"), ("org/lenses", "")):
            with self.subTest(repository=repository, filename=filename), self.assertRaises(ValueError):
                jacobian_lens.download(repository, filename)
        _, status = inspection.import_jacobian_lens(None, self.manager.model_id)
        self.assertIn("Choose a saved lens.pt file", status)

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

        store = Path(self.directory.name) / "config" / "jacobian_lenses.json"
        with mock.patch.object(jacobian_lens, "store_path", return_value=store), mock.patch.object(
            runtime, "MANAGER", self.manager,
        ):
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


@needs_mlx
class MlxJacobianLensTests(unittest.TestCase):
    """The same lens read through an MLX conversion of the fitted model."""

    def setUp(self):
        import mlx.core as mx

        from mlx_runtime import MlxEngine
        from test_mlx_runtime import HIDDEN, LAYERS, VOCAB, tiny_llama
        from test_streaming import FakeTokenizer

        class Tokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=True):
                return self(text, add_special_tokens=False)["input_ids"]

        self.mx = mx
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "lens.pt"
        self.model = tiny_llama()
        self.manager = ModelManager()
        self.manager.model = self.model
        self.manager.engine = MlxEngine(self.model, {
            "eos_token_id": 99, "model_type": "llama",
            "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
        })
        self.manager.tokenizer = Tokenizer(tuple(f"t{i}" for i in range(VOCAB)), 1)
        self.manager.model_id = "mlx-community/tiny-decoder-4bit"
        self.manager.kind = "mlx"
        self.manager.precision = "4-bit"
        torch.manual_seed(7)
        self.data = {
            "J": {layer: torch.randn(HIDDEN, HIDDEN) for layer in range(LAYERS)},
            "source_layers": list(range(LAYERS)), "n_prompts": 50, "d_model": HIDDEN,
        }
        torch.save(self.data, self.path)
        self.ids = [3, 5, 7, 11, 13]

    def reference(self):
        """Each block's readout at every position from one uncached pass."""
        from mlx_runtime import _Recorder

        mx = self.mx
        engine = self.manager.engine
        recorder = _Recorder()
        with engine._recording(recorder):
            logits = self.model(mx.array([self.ids]))
        mx.eval(logits, *recorder.hidden)
        norm = engine.final_norm()
        expected = {}
        for layer, matrix in self.data["J"].items():
            hidden = np.array(recorder.hidden[layer + 1][0].astype(mx.float32))
            transported = mx.array(hidden @ matrix.numpy().T)[None]
            expected[layer] = np.array(engine.read_head(norm(transported))[0].astype(mx.float32))
        return expected, np.array(logits[0].astype(mx.float32))

    def test_a_quantized_conversion_reads_the_lens_fitted_for_its_source(self):
        imported = self.manager.import_jacobian_lens(str(self.path), "test/tiny-decoder")
        self.assertEqual(imported["backend"], "mlx")
        index = 3
        result = self.manager.inspect_jacobian(
            self.ids, index, lens_id=imported["import_id"], load_id=self.manager.load_id,
            pinned_text="t7",
        ).to_dict()
        expected, logits = self.reference()
        self.assertEqual(result["pinned_id"], 7)
        self.assertEqual(result["backend"], "mlx")
        self.assertEqual(result["precision"], "4-bit")
        for row, column in zip(result["layers"], result["slice"]["layers"]):
            scores = expected[row["layer"]]
            top = np.argsort(-scores[index])[:5]
            self.assertEqual([c["token_id"] for c in row["candidates"]], top.tolist())
            np.testing.assert_allclose([c["score"] for c in row["candidates"]], scores[index, top], atol=1e-3)
            self.assertEqual(row["rank"], int((scores[index] > scores[index, 7]).sum()) + 1)
            for position, cell in enumerate(column["cells"]):
                self.assertEqual(cell["token_id"], int(scores[position].argmax()))
                self.assertEqual(cell["pinned_rank"], int((scores[position] > scores[position, 7]).sum()) + 1)
        self.assertEqual(
            [cell["token_id"] for cell in result["slice"]["output"]],
            logits[: index + 1].argmax(axis=-1).tolist(),
        )
        for position, cell in enumerate(result["slice"]["output"]):
            self.assertEqual(cell["pinned_rank"], int((logits[position] > logits[position, 7]).sum()) + 1)
            self.assertAlmostEqual(cell["pinned_score"], float(logits[position, 7]), places=4)
        rendered = charts.jacobian_lens_chart(result)
        self.assertIn("4-bit MLX weights", rendered)

    def test_the_declared_source_must_name_the_conversion(self):
        with self.assertRaisesRegex(ValueError, "MLX conversion"):
            self.manager.import_jacobian_lens(str(self.path), "test/other-decoder")
        with self.assertRaisesRegex(ValueError, "MLX conversion"):
            self.manager.import_jacobian_lens(str(self.path), "")
        # A base-model lens is not the instruct conversion's, though its name is inside it.
        self.manager.model_id = "mlx-community/tiny-decoder-instruct-4bit"
        with self.assertRaisesRegex(ValueError, "quantization suffix"):
            self.manager.import_jacobian_lens(str(self.path), "test/tiny-decoder")
        for conversion in (
            "mlx-community/Tiny-Decoder-bf16", "mlx-community/Tiny-Decoder-4bit-DWQ",
            "mlx-community/tiny-decoder-bf16",
        ):
            self.manager.model_id = conversion
            self.assertIn("import_id", self.manager.import_jacobian_lens(str(self.path), "test/tiny-decoder"))


if __name__ == "__main__":
    unittest.main()
