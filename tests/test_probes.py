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
from chatlab.extensions.probes import page as page_module
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
    def test_imported_probe_examples_match_training_bounds(self):
        value = trained()
        for examples in ([], ["one"], ["x"] * 65, ["x" * 32769, "y"]):
            with self.assertRaisesRegex(ValueError, "2–64 examples"):
                probes.normalize(value | {"examples": {"positive": examples, "negative": ["a", "b"]}})


    def test_imported_probe_coefficient_limit_is_checked_before_conversion(self):
        value = trained()
        count = sum(len(layer["weights"]) for layer in value["layers"])
        # A deliberately unconvertible coefficient proves the total limit is
        # checked before allocating a second set of normalized float lists.
        value["layers"][0]["weights"][0] = object()
        with mock.patch.object(probes, "MAX_COEFFICIENTS", count - 1):
            with self.assertRaisesRegex(ValueError, "total coefficient limit"):
                probes.normalize(value)

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

    def test_an_overflowed_reading_is_refused(self):
        with self.assertRaisesRegex(ValueError, "overflowed"):
            probes.probabilities(trained(), np.array([[np.inf], [0.0], [np.nan]]))

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
            (dict(layers=[{}] * 257), "256 layers"),
            (dict(layers=[dict(layers[0], weights=[0.0] * 65537)]), "65,536 weights"),
            (dict(negative_label="Yes"), "different labels"),
            (dict(pool="max"), "last or mean"),
            (dict(best_layer=len(layers)), "best layer"),
            (dict(layers=[dict(layers[0], weights=layers[0]["weights"][:-1]), layers[1]]), "same width"),
            (dict(layers=[dict(layers[0], bias=float("nan"))]), "finite"),
            (dict(layers=[dict(layers[1])]), "in order"),
            (dict(id="../../etc"), "hexadecimal"),
            (dict(created=1e300), "between 1970 and 3000"),
            (dict(created=-1.0), "between 1970 and 3000"),
            (dict(model_revision=""), "revision"),
            (dict(model_revision=7), "revision"),
            (dict(precision="2-bit"), "precision"),
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

    def test_projection_stop_interrupts_layers_and_removes_hooks(self):
        from chatlab.model_inspection import ProjectionCancelled
        held = tiny_manager()
        blocks = steering.decoder_layers(held.model)
        with ModelService(lambda: held).open_session() as session:
            ids = session.example_ids("Hello world")
            stop = blocks[0].register_forward_hook(lambda *_args: session.cancel())
            try:
                with mock.patch.object(blocks[1], "forward", wraps=blocks[1].forward) as later:
                    with self.assertRaises(ProjectionCancelled):
                        session.project_layers(ids, np.zeros((2, 8)))
                    later.assert_not_called()
            finally:
                stop.remove()
            for block in blocks:
                self.assertFalse(block._forward_hooks)
                self.assertFalse(block._forward_pre_hooks)
            with self.assertRaises(ProjectionCancelled):
                session.project_layers(ids, np.zeros((2, 8)))
        # The lease and hooks are reusable after stopping.
        with ModelService(lambda: held).open_session() as session:
            self.assertEqual(session.project_layers(ids, np.zeros((2, 8))).shape, (2, len(ids)))

    def test_bad_directions_a_stale_load_and_mlx_are_refused(self):
        held = tiny_manager()
        ids = held._example_ids("Hello world", False)
        with self.assertRaisesRegex(ValueError, "each of this model's 2"):
            held.project_blocks(ids, np.zeros((3, 8)))
        with self.assertRaisesRegex(ValueError, "4 wide; this model's blocks are 8"):
            held.project_blocks(ids, np.zeros((2, 4)))
        # The flat cap holds even for a model whose window would allow more.
        with mock.patch("chatlab.tokenization.SCORE_TOKEN_LIMIT", len(ids) - 1):
            with self.assertRaisesRegex(ValueError, f"above the {len(ids) - 1} one reading may be"):
                held.project_blocks(ids, np.zeros((2, 8)))
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
                      pairs=False, pooling="last", strength=1.0, view="owner")
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

    def test_probe_downloads_reuse_and_clean_up_the_owned_view_directory(self):
        runs = page_module.Runs()
        probe = self.train()[0]
        first = Path(runs.stage("view", probe))
        second = Path(runs.stage("view", {**probe, "name": "Different"}))
        self.assertEqual(first.parent, second.parent)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        other = Path(runs.stage("other", probe))
        runs.forget("view")
        self.assertFalse(second.parent.exists())
        self.assertTrue(other.exists())
        runs.forget("other")
        self.assertFalse(other.parent.exists())

    def test_queued_training_cannot_replace_a_later_opened_probe(self):
        probe = self.train()[0]
        request = self.fn["begin_training"]("owner")
        current = self.fn["open_saved"](probe["id"], "owner")[0]
        with mock.patch.object(self.manager, "claim_generation", side_effect=AssertionError("stale training")):
            self.assertEqual(self.train(request=request), (gr.skip(),) * 21)
        self.assertIsNotNone(self.read(current, READ, WANTED[0])[-1][0])

    def test_training_completed_after_opening_another_probe_does_not_replace_it(self):
        opened = self.train()[0]
        fit = probes.train

        def moved_on(*args, **kwargs):
            self.fn["open_saved"](opened["id"], "owner")
            return fit(*args, **kwargs)

        with mock.patch.object(probes, "train", side_effect=moved_on):
            frame = self.train(probe_name="New training")
        self.assertEqual(frame, (gr.skip(),) * 21)
        # The training artifact is retained even when its UI completion is stale.
        self.assertEqual(len(list(self.data.glob("*.json"))), 2)

    def test_failed_training_preserves_the_displayed_reading(self):
        probe = self.train()[0]
        reading = self.read(probe, READ, WANTED[0])[-1][0]
        with self.assertRaises(gr.Error):
            self.train(wanted="only one example")
        painted = self.fn["change_layer"](probe, reading, 1, "owner")
        self.assertNotEqual(painted, (gr.skip(), gr.skip()))
        note, _ = self.fn["inspect_token"](probe, reading, 1, "owner", SimpleNamespace(index=0))
        self.assertIn("Token 1", note)

    def test_overlong_metadata_is_refused_before_model_work(self):
        with mock.patch.object(self.manager, "claim_generation", side_effect=AssertionError("must not read examples")):
            for changes in ({"probe_name": "n" * 201}, {"looking_for": "p" * 61}, {"against": "n" * 61}):
                with self.subTest(changes=changes), self.assertRaisesRegex(gr.Error, "200 characters.*60"):
                    self.train(**changes)

    def test_invalid_l2_is_refused_before_model_work(self):
        with mock.patch.object(self.manager, "claim_generation", side_effect=AssertionError("must not read examples")):
            for strength in (0, -1, float("nan"), float("inf")):
                with self.subTest(strength=strength), self.assertRaisesRegex(gr.Error, "positive and finite"):
                    self.train(strength=strength)
        probe = trained()
        for strength in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                probes.normalize(probe | {"l2": strength})

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
        note, table = self.fn["inspect_token"](probe, reading, 1, "owner", SimpleNamespace(index=len(ids) - 1))
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

    def test_a_passage_longer_than_the_window_is_refused_not_cut_short(self):
        probe = self.train()[0]
        with mock.patch("chatlab.tokenization.model_position_limit", return_value=1):
            with self.assertRaisesRegex(gr.Error, "above the 1 one reading may be"):
                self.read(probe, READ, WANTED[0])

    def test_a_probe_for_another_revision_is_refused_when_both_are_known(self):
        with mock.patch.object(self.manager, "model_revision", return_value="a" * 40):
            probe = self.train()[0]
            self.assertEqual(probe["model_revision"], "a" * 40)
        with mock.patch.object(self.manager, "model_revision", return_value="b" * 40):
            with self.assertRaisesRegex(gr.Error, "revision aaaaaaaaaaaa of test/tiny, and revision bbbbbbbbbbbb"):
                self.read(probe, READ, "Hello")
        # A load that records no revision cannot be told apart, so it is not refused.
        with mock.patch.object(self.manager, "model_revision", return_value=None):
            self.assertTrue(self.read(probe, READ, "Hello")[-1][0]["token_ids"])

    def test_another_precision_reads_with_a_note(self):
        self.manager.precision = "full"
        probe = self.train()[0]
        self.assertEqual(probe["precision"], "full")
        self.assertEqual(self.read(probe, READ, "Hello")[-1][3], "")
        self.manager.precision = "4-bit"
        frame = self.read(probe, READ, "Hello")[-1]
        self.assertTrue(frame[0]["token_ids"])
        self.assertIn("trained on full weights and this load is 4-bit", frame[3])

    def test_the_reply_gets_what_the_prompt_leaves_of_one_reading(self):
        probe = self.train()[0]
        prompt = len(self.manager._prompt_token_ids([{"role": "user", "content": "Hello"}])[0])
        def read(limit, asked):
            with mock.patch("chatlab.extensions.probes.page.READ_LIMIT", limit):
                return list(self.fn["read"](probe, GENERATE, "Hello", "", 0.0, 42, asked, False, False, "owner", 1))

        with ModelService(lambda: self.manager).open_session() as session:
            self.assertEqual(session.position_limit, 64)
        with mock.patch.object(self.manager, "generate", wraps=self.manager.generate) as generate:
            read(prompt + 2, prompt + 2)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 2)
        with self.assertRaisesRegex(gr.Error, "leaves no room for a reply"):
            read(prompt, 1)
        # A window smaller than the cap is the limit, so a prompt that fills it is refused too.
        with mock.patch("chatlab.tokenization.model_position_limit", return_value=prompt):
            with self.assertRaisesRegex(gr.Error, f"no room for a reply in a {prompt}-token reading"):
                read(prompt + 2, 1)

    def test_a_probe_for_another_model_is_refused(self):
        probe = dict(self.train()[0], model_id="other/model")
        with self.assertRaisesRegex(gr.Error, "trained on other/model"):
            self.read(probe, READ, "Hello")
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_saved_and_imported_probes_open(self):
        probe = self.train()[0]
        opened = self.fn["open_saved"](probe["id"], "owner")
        self.assertEqual(probes.normalize(opened[0]), probes.normalize(probe))
        self.assertNotEqual(opened[0]["_view_probe"], probe["_view_probe"])
        # The form is refilled from the probe, so it can be changed and trained again.
        self.assertEqual(opened[12:], ("Greeting", "Greeting", "Question", "\n".join(WANTED), "\n".join(UNWANTED),
                                       False, False, "last", 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.json"
            path.write_text(probes.dumps(dict(probe, name="Shared")))
            imported = self.fn["import_probe"](str(path), "owner")
        self.assertEqual(imported[0]["name"], "Shared")
        self.assertEqual(probes.read(self.data / f"{probe['id']}.json")["name"], "Shared")
        with self.assertRaises(gr.Error):
            self.fn["open_saved"]("0" * 32, "owner")

    def test_a_new_run_clears_the_last_reading_first(self):
        probe = self.train()[0]
        shown = self.read(probe, READ, WANTED[0])[-1][0]
        reading, strip, heat, reply, detail, table = self.fn["clear_reading"]("owner")
        self.assertIsNone(reading)
        self.assertEqual((strip["value"], strip["visible"]), ([], False))
        self.assertEqual((heat, reply, detail, table), ("", "", "", []))
        # A slider release or click queued before the run still carries the old reading; it paints nothing.
        self.assertEqual(self.fn["change_layer"](probe, shown, 0, "owner"), (gr.skip(), gr.skip()))
        self.assertEqual(self.fn["inspect_token"](probe, shown, 0, "owner", SimpleNamespace(index=0)),
                         (gr.skip(), gr.skip()))
        # Nor does another view's reading.
        fresh = self.read(probe, READ, WANTED[0])[-1][0]
        self.assertEqual(self.fn["change_layer"](probe, fresh, 0, "another view"), (gr.skip(), gr.skip()))

    def test_queued_read_refuses_a_probe_replaced_before_start(self):
        old = self.train()[0]
        current = self.fn["open_saved"](old["id"], "owner")[0]
        with mock.patch.object(self.manager, "claim_generation", side_effect=AssertionError("stale model work")):
            with self.assertRaisesRegex(gr.Error, "displayed probe changed"):
                self.read(old, READ, WANTED[0])
        self.assertIsNotNone(self.read(current, READ, WANTED[0])[-1][0])

    def test_stopped_projection_reports_stop_and_releases_the_session(self):
        probe = self.train()[0]
        block = steering.decoder_layers(self.manager.model)[0]
        stop = block.register_forward_hook(lambda *_args: self.fn["cancel"]("owner"))
        try:
            frames = self.read(probe, READ, WANTED[0])
            self.assertEqual(frames[-1], (gr.skip(), gr.skip(), gr.skip(), "Stopped.", gr.skip()))
        finally:
            stop.remove()
        self.assertIsNotNone(self.read(probe, READ, WANTED[0])[-1][0])

    def test_a_run_the_page_moved_on_from_publishes_nothing(self):
        probe = self.train()[0]
        directions = probes.directions
        for moved_on in (lambda: self.fn["clear_reading"]("owner"),
                         lambda: self.fn["open_saved"](probe["id"], "owner")):
            with self.subTest(moved_on=moved_on):
                def mid_run(value):
                    # Another Read click, or another probe opened, while this run is reading.
                    moved_on()
                    return directions(value)
                with mock.patch.object(probes, "directions", side_effect=mid_run):
                    frames = self.read(probe, READ, WANTED[0])
                self.assertEqual(frames[-1], (gr.skip(),) * 5)
        # Left alone, the current probe's run publishes.
        probe = self.fn["open_saved"](probe["id"], "owner")[0]
        self.assertIsNotNone(self.read(probe, READ, WANTED[0])[-1][0])

    def test_a_token_inspection_overtaken_by_a_run_publishes_nothing(self):
        probe = self.train()[0]
        reading = self.read(probe, READ, WANTED[0])[-1][0]
        real_quoted = page_module.quoted

        def cleared_meanwhile(value):
            self.fn["clear_reading"]("owner")
            return real_quoted(value)
        with mock.patch.object(page_module, "quoted", side_effect=cleared_meanwhile):
            self.assertEqual(self.fn["inspect_token"](probe, reading, 1, "owner", SimpleNamespace(index=0)),
                             (gr.skip(), gr.skip()))

    def test_changing_the_layer_repaints_without_the_model(self):
        probe = self.train()[0]
        reading = self.read(probe, READ, WANTED[0])[-1][0]
        self.manager.claim_generation()
        try:
            strip, heat = self.fn["change_layer"](probe, reading, 0, "owner")
        finally:
            self.manager.release_generation()
        self.assertEqual(len(strip["value"]), len(reading["token_ids"]))
        self.assertIn('class="probe-chosen"><th>layer 0', heat)
        other = dict(probe, id="f" * 32)
        self.assertEqual(self.fn["change_layer"](other, reading, 0, "owner"), (gr.skip(), gr.skip()))


if __name__ == "__main__":
    unittest.main()
