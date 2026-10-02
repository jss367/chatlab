"""Linear probes: the fit, the probe file, the host's readings and the page."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from chatlab import steering
from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.probes import probe as probes
from chatlab.extensions.probes.page import GENERATE, READ, build_page
from chatlab.model_runtime import ModelManager
from chatlab.text_generation import ModelChanged
from fakes import EOS_ID, PIECES, FakeTokenizer
from ui_support import handlers_by_name

WANTED = ["Hello world", "Hello!", "Hello world!"]
UNWANTED = ["How are", "How are you", "How you?"]


class Greedy(FakeTokenizer):
    """Example text reaches the model as its own tokens, not one placeholder."""

    def __call__(self, text, **kwargs):
        try:
            return super().__call__(text, **dict(kwargs, add_special_tokens=False))
        except ValueError:
            # A templated prompt has no pieces of its own; it stays one placeholder.
            return super().__call__(text, **kwargs)


def tiny_manager():
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(PIECES), hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, eos_token_id=EOS_ID,
    ))
    model.eval()
    model.set_attn_implementation("eager")
    manager = ModelManager()
    manager.model = model
    manager.model_id = "test/tiny"
    manager.tokenizer = Greedy()
    return manager


def synthetic(count, sign, layers=3, width=16, signal_layer=1, seed=0):
    rng = np.random.default_rng(seed + (sign > 0))
    rows = rng.normal(size=(count, layers, width))
    rows[:, signal_layer, 0] += 3.0 * sign
    return rows


def trained(**changes):
    layers, folds = probes.train(synthetic(12, 1), synthetic(12, -1))
    fields = dict(name="Test", model_id="test/tiny", positive_label="Yes", negative_label="No",
                  positive_examples=["a", "b"], negative_examples=["c", "d"], pool="last",
                  chat_template=False, l2=1.0, layers=layers, folds=folds)
    return probes.build(**(fields | changes))


class FitTests(unittest.TestCase):
    def test_the_fit_is_at_the_penalized_optimum(self):
        rng = np.random.default_rng(3)
        rows = rng.normal(size=(30, 50)) * rng.uniform(0.1, 10, size=50) + rng.normal(size=50)
        labels = (rows[:, 0] + rng.normal(size=30) > 0).astype(float)
        weights, bias = probes.fit(rows, labels, l2=0.5)
        # In the standardized space the gradient of loss + l2/2 |w|^2 vanishes.
        mean, spread = rows.mean(axis=0), rows.std(axis=0)
        standardized_weights = weights * spread
        predicted = 1 / (1 + np.exp(-(rows @ weights + bias)))
        gradient = ((rows - mean) / spread).T @ (predicted - labels) + 0.5 * standardized_weights
        np.testing.assert_allclose(gradient, 0, atol=1e-6)
        self.assertAlmostEqual(float(np.sum(predicted - labels)), 0, places=6)

    def test_held_out_accuracy_finds_the_layer_that_carries_the_signal(self):
        layers, folds = probes.train(synthetic(12, 1), synthetic(12, -1))
        self.assertEqual(folds, 5)
        self.assertEqual(probes.best_layer(layers), 1)
        self.assertGreaterEqual(layers[1]["heldout_accuracy"], 0.9)
        # The noise layers fit their own examples better than they predict new ones.
        self.assertTrue(all(layers[n]["train_accuracy"] > layers[n]["heldout_accuracy"] for n in (0, 2)))
        self.assertLess(max(layers[0]["heldout_accuracy"], layers[2]["heldout_accuracy"]), 0.9)

    def test_folds_deal_both_sides_into_every_fold(self):
        labels = np.array([1] * 7 + [0] * 3)
        assignment = probes.folds_for(labels)
        self.assertEqual(int(assignment.max()) + 1, 3)
        for fold in range(3):
            self.assertEqual(set(labels[assignment == fold]), {0, 1})
        np.testing.assert_array_equal(assignment, probes.folds_for(labels))
        self.assertIsNone(probes.folds_for(np.array([1, 0, 0])))

    def test_paired_folds_hold_each_pair_out_together(self):
        labels = np.array([1] * 6 + [0] * 6)
        assignment = probes.folds_for(labels, paired=True)
        np.testing.assert_array_equal(assignment[:6], assignment[6:])
        self.assertEqual(int(assignment.max()) + 1, 5)
        with self.assertRaisesRegex(ValueError, "same number"):
            probes.folds_for(np.array([1, 1, 1, 0, 0]), paired=True)

    def test_split_pairs_read_below_chance_and_paired_folds_do_not(self):
        # Each pair shares everything but the label's direction, as minimal pairs do.
        rng = np.random.default_rng(5)
        shared = rng.normal(size=(12, 1, 128))
        signal = np.zeros((1, 1, 128))
        signal[..., 0] = 2.0
        wanted, unwanted = shared + signal, shared - signal
        split = probes.train(wanted, unwanted)[0][0]["heldout_accuracy"]
        together = probes.train(wanted, unwanted, paired=True)[0][0]["heldout_accuracy"]
        self.assertLess(split, 0.5)
        self.assertGreater(together, 0.9)

    def test_too_few_examples_and_a_bad_penalty_are_refused(self):
        with self.assertRaisesRegex(ValueError, "two examples on each side"):
            probes.train(synthetic(1, 1), synthetic(4, -1))
        with self.assertRaisesRegex(ValueError, "positive number"):
            probes.train(synthetic(3, 1), synthetic(3, -1), l2=0)
        with self.assertRaisesRegex(ValueError, "same layers"):
            probes.train(synthetic(3, 1, layers=2), synthetic(3, -1))

    def test_probabilities_are_the_logistic_of_the_projection(self):
        probe = trained()
        row = synthetic(1, 1)[0]
        projections = np.array([[float(np.dot(row[layer], weights))]
                                for layer, weights in enumerate(probes.directions(probe))])
        expected = [1 / (1 + np.exp(-(row[item["layer"]] @ np.array(item["weights"]) + item["bias"])))
                    for item in probe["layers"]]
        np.testing.assert_allclose(probes.probabilities(probe, projections)[:, 0], expected, rtol=1e-5)


class FileTests(unittest.TestCase):
    def test_a_probe_survives_saving_and_reading(self):
        probe = trained()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(probes.dumps(probe))
            (Path(directory) / "broken.json").write_text("{")
            self.assertEqual(probes.read(path), probe)
            self.assertEqual([p["id"] for p in probes.saved(directory)], [probe["id"]])

    def test_malformed_probes_are_refused(self):
        probe = trained()
        layers = probe["layers"]
        for change, message in (
            (dict(format="other"), "chatlab-probe-1"),
            (dict(negative_label="Yes"), "different labels"),
            (dict(pool="max"), "last or mean"),
            (dict(best_layer=len(layers)), "best layer"),
            (dict(layers=[dict(layers[0], weights=layers[0]["weights"][:-1]), layers[1]]), "same width"),
            (dict(layers=[dict(layers[0], bias=float("nan"))]), "finite"),
            (dict(layers=[dict(layers[1])]), "in order"),
            (dict(id="../../etc"), "hexadecimal"),
            (dict(created=1e300), "between 1970 and 3000"),
            (dict(created=-1.0), "between 1970 and 3000"),
            (dict(layers=[dict(layers[0], weights=[1e300] * len(layers[0]["weights"])), layers[1]]), "32-bit"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                probes.normalize(probe | change)


class HostReadingTests(unittest.TestCase):
    def test_examples_are_read_as_a_steering_extraction_reads_them(self):
        held = tiny_manager()
        blocks = steering.decoder_layers(held.model)
        rows = held.read_examples(WANTED, pool="mean")
        self.assertEqual(rows.shape, (3, 2, 8))
        with torch.inference_mode():
            first = held._pooled_block_outputs(held._example_ids(WANTED[0], False), blocks, "mean")
        np.testing.assert_allclose(rows[0], first, rtol=1e-5, atol=1e-6)

    def test_each_position_is_read_along_its_blocks_direction(self):
        held = tiny_manager()
        blocks = steering.decoder_layers(held.model)
        ids = held._example_ids("Hello world! How are you?", False)
        directions = np.random.default_rng(0).normal(size=(2, 8))
        projections = held.project_blocks(ids, directions)
        self.assertEqual(projections.shape, (2, len(ids)))
        self.assertFalse(blocks[0]._forward_hooks)
        with torch.inference_mode():
            # The last position's reading is the pooled reading a probe was fitted on.
            last = held._pooled_block_outputs(ids, blocks, "last")
            output = held.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        for layer in range(2):
            self.assertAlmostEqual(projections[layer, -1], float(last[layer] @ directions[layer]), places=4)
        # Every block but the last reports the same tensor at every position.
        np.testing.assert_allclose(projections[0], output.hidden_states[1][0].numpy() @ directions[0],
                                   rtol=1e-4, atol=1e-5)

    def test_bad_directions_a_stale_load_and_mlx_are_refused(self):
        held = tiny_manager()
        ids = held._example_ids("Hello world", False)
        with self.assertRaisesRegex(ValueError, "each of this model's 2"):
            held.project_blocks(ids, np.zeros((3, 8)))
        with self.assertRaisesRegex(ValueError, "4 wide; this model's blocks are 8"):
            held.project_blocks(ids, np.zeros((2, 4)))
        self.assertFalse(steering.decoder_layers(held.model)[0]._forward_hooks)
        previous = held.load_id
        held.model_id = "other/model"
        with self.assertRaises(ModelChanged):
            held.project_blocks(ids, np.zeros((2, 8)), load_id=previous)
        held.model_id = "test/tiny"
        with mock.patch.object(held, "_engine", return_value=SimpleNamespace(backend="mlx")):
            with self.assertRaisesRegex(ValueError, "MLX"):
                held.read_examples(WANTED)
            with self.assertRaisesRegex(ValueError, "MLX"):
                held.project_blocks(ids, np.zeros((2, 8)))

    def test_the_session_reads_under_its_pinned_load(self):
        held = tiny_manager()
        with ModelService(lambda: held).open_session() as session:
            ids = session.example_ids("Hello world")
            self.assertEqual(ids, held._example_ids("Hello world", False))
            self.assertEqual(session.read_examples(WANTED).shape, (3, 2, 8))
            self.assertEqual(session.project_layers(ids, np.zeros((2, 8))).shape, (2, len(ids)))
            held.model_id = "other/model"
            with self.assertRaisesRegex(ValueError, "changed"):
                session.project_layers(ids, np.zeros((2, 8)))
            held.model_id = "test/tiny"
        with self.assertRaisesRegex(ValueError, "closed"):
            session.read_examples(WANTED)


class PageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name) / "probes"
        self.manager = tiny_manager()
        context = ExtensionContext(ModelService(lambda: self.manager), TokenInspector(), self.data,
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = handlers_by_name(demo)

    def train(self, **changes):
        fields = dict(probe_name="Greeting", looking_for="Greeting", against="Question",
                      wanted="\n".join(WANTED), unwanted="\n\n".join(UNWANTED), template=False,
                      pairs=False, pooling="last", strength=1.0)
        return self.fn["train_probe"](*(fields | changes).values())

    def read(self, probe, mode, text, layer=1, show_prompt=False):
        return list(self.fn["read"](probe, mode, text, "", 0.0, 42, 3, show_prompt, False, "owner", layer))

    def test_training_saves_the_probe_and_releases_the_model(self):
        frame = self.train()
        probe = frame[0]
        self.assertEqual(probe["examples"]["negative"], UNWANTED)
        self.assertEqual(len(probe["layers"]), 2)
        self.assertEqual(json.loads((self.data / f"{probe['id']}.json").read_text())["id"], probe["id"])
        self.assertIn("Greeting", frame[1])
        self.assertEqual(frame[5]["maximum"], 1)
        self.assertEqual(frame[3]["value"], probe["id"])
        # The download is a copy where Gradio may serve it, named for the probe.
        self.assertTrue(Path(frame[4]).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertEqual(Path(frame[4]).name, "Greeting.json")
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_training_refusals_name_the_problem(self):
        with self.assertRaisesRegex(gr.Error, "2 to 64 examples of Greeting"):
            self.train(wanted="Hello")
        with self.assertRaisesRegex(gr.Error, "different labels"):
            self.train(against="Greeting")
        with self.assertRaisesRegex(gr.Error, "same number of lines on each side; there are 3 and 2"):
            self.train(pairs=True, unwanted="How are\nHow you?")

    def test_reading_text_scores_the_last_token_as_training_read_it(self):
        probe = self.train()[0]
        frames = self.read(probe, READ, WANTED[0])
        reading = frames[-1][0]
        ids = self.manager._example_ids(WANTED[0], False)
        self.assertEqual(reading["token_ids"], ids)
        blocks = steering.decoder_layers(self.manager.model)
        with torch.inference_mode():
            pooled = self.manager._pooled_block_outputs(ids, blocks, "last")
        for item in probe["layers"]:
            expected = 1 / (1 + np.exp(-(pooled[item["layer"]] @ np.array(item["weights"]) + item["bias"])))
            self.assertAlmostEqual(reading["probabilities"][item["layer"]][-1], expected, places=4)
        strip = frames[-1][1]
        self.assertEqual(len(strip["value"]), len(ids))
        self.assertEqual(set(strip["color_map"]), {label for _, label in strip["value"]} | set(strip["color_map"]))
        note, table = self.fn["inspect_token"](probe, reading, 1, SimpleNamespace(index=len(ids) - 1))
        self.assertIn("Greeting at layer 1", note)
        self.assertEqual(len(table), 2)

    def test_a_reply_is_generated_then_read_from_its_first_token(self):
        probe = self.train()[0]
        frames = self.read(probe, GENERATE, "Hello")
        reading = frames[-1][0]
        self.assertGreater(len(frames), 1)
        prompt = len(self.manager._prompt_token_ids([{"role": "user", "content": "Hello"}])[0])
        self.assertEqual(reading["first"], prompt)
        self.assertEqual(len(frames[-1][1]["value"]), len(reading["token_ids"]) - prompt)
        whole = self.read(probe, GENERATE, "Hello", show_prompt=True)[-1][0]
        self.assertEqual(whole["first"], 0)
        self.assertEqual(len(self.read(probe, GENERATE, "Hello", layer=0)[-1][1]["value"]),
                         len(reading["token_ids"]) - prompt)

    def test_a_reply_at_the_window_leaves_out_the_token_the_model_never_read(self):
        probe = self.train()[0]
        with ModelService(lambda: self.manager).open_session() as session:
            self.assertEqual(session.position_limit, 64)
        full = self.read(probe, GENERATE, "Hello", show_prompt=True)[-1][0]
        window = len(full["token_ids"]) - 1
        # Generation is left alone; only the window the reading is trimmed to moves.
        with mock.patch("chatlab.tokenization.model_position_limit", return_value=window):
            trimmed = self.read(probe, GENERATE, "Hello", show_prompt=True)[-1][0]
        self.assertEqual(trimmed["token_ids"], full["token_ids"][:window])
        self.assertEqual(len(trimmed["probabilities"][0]), window)

    def test_a_passage_longer_than_the_window_is_refused_not_cut_short(self):
        probe = self.train()[0]
        with mock.patch("chatlab.tokenization.model_position_limit", return_value=1):
            # The reply is trimmed to the window; the passage is the reader's own and is not.
            with mock.patch("chatlab.model_inspection.model_position_limit", return_value=1):
                with self.assertRaisesRegex(gr.Error, "above the 1 this model can read"):
                    self.read(probe, READ, WANTED[0])

    def test_a_probe_for_another_model_is_refused(self):
        probe = dict(self.train()[0], model_id="other/model")
        with self.assertRaisesRegex(gr.Error, "trained on other/model"):
            self.read(probe, READ, "Hello")
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_saved_and_imported_probes_open(self):
        probe = self.train()[0]
        opened = self.fn["open_saved"](probe["id"])
        self.assertEqual(opened[0], probe)
        # The form is refilled from the probe, so it can be changed and trained again.
        self.assertEqual(opened[12:], ("Greeting", "Greeting", "Question", "\n".join(WANTED), "\n".join(UNWANTED),
                                       False, False, "last", 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.json"
            path.write_text(probes.dumps(dict(probe, name="Shared")))
            imported = self.fn["import_probe"](str(path))
        self.assertEqual(imported[0]["name"], "Shared")
        self.assertEqual(probes.read(self.data / f"{probe['id']}.json")["name"], "Shared")
        with self.assertRaises(gr.Error):
            self.fn["open_saved"]("0" * 32)

    def test_changing_the_layer_repaints_without_the_model(self):
        probe = self.train()[0]
        reading = self.read(probe, READ, WANTED[0])[-1][0]
        self.manager.claim_generation()
        try:
            strip, heat = self.fn["change_layer"](probe, reading, 0)
        finally:
            self.manager.release_generation()
        self.assertEqual(len(strip["value"]), len(reading["token_ids"]))
        self.assertIn('class="probe-chosen"><th>layer 0', heat)
        other = dict(probe, id="f" * 32)
        self.assertEqual(self.fn["change_layer"](other, reading, 0), (gr.skip(), gr.skip()))


if __name__ == "__main__":
    unittest.main()
