"""Probe files written by earlier ChatLab versions keep reading the way they did.

Each fixture under fixtures/saved/probes was written once by the probe writer
of its era (see its generator's header) and is never regenerated. The tests
open them the way the Probes page does, through ``probe.read`` and the page's
own import and open handlers, and assert what the page shows: the probe's
fields, its per-layer table, its summary and its readings. A probe file also
serves the Direction edits page as directions, so each is read there too.
"""
import json
import math
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.direction_edits import files as edit_files
from chatlab.extensions.direction_edits import page as edits_page
from chatlab.extensions.probes import page as probe_page
from chatlab.extensions.probes import probe as probes
from ui_support import handlers_by_name

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "saved" / "probes"

# What each era's file means, written out by hand from what the page showed.
EXPECTED = {
    "probe-era1-no-revision-below-chance.json": {
        "id": "1e4a0c0ffee0000000000000000000a1",
        "name": "Formal register — twins, unpaired",
        "model_id": "test/probe-era1",
        "model_revision": None,
        "precision": None,
        "positive_label": "Formal",
        "negative_label": "Casual",
        "pool": "last",
        "chat_template": True,
        "l2": 0.5,
        "folds": 4,
        "paired": False,
        "created": 1790935212.345678,
        "examples": {
            "positive": ["Good afternoon, how may I help?", "I would be grateful for a reply.",
                         "Please find the report attached.", "    Kindly confirm receipt."],
            "negative": ["hey, what's up?", "thx for replying", "report's attached", "    got it?"],
        },
        "width": 8,
        "best_layer": 2,
        # layer, bias, first three weights, train accuracy, held-out accuracy, held-out loss
        "layers": [
            (0, 0.830143525, [0.310384098, -0.0733512251, -0.228378825], 0.75, 0.125, 2.038064025499523),
            (1, 0.534612283, [0.354430044, -0.0300084283, -0.00809245722], 0.75, 0.125, 1.9897278012295485),
            (2, -0.0792879924, [0.241977717, 0.0501689875, 0.0804537563], 0.875, 0.25, 1.567710028510811),
        ],
        "summary": (
            "**`Formal register — twins, unpaired`** for test/probe-era1: Formal (4 examples) against Casual (4), "
            "read at the last token of each example as a user message. Best layer 2: 25% accurate on held-out "
            "examples over 4 folds, 88% on the examples it was trained on. Every layer is below chance on held-out "
            "examples. If line n of each side is the same example changed, tick **The examples are pairs** and "
            "train again."),
        "rows": [["0", "12%", 2.038, "75%"], ["1", "12%", 1.99, "75%"], ["2 (best)", "25%", 1.568, "88%"]],
        "at_zero": [0.6963852765750833, 0.6305582097832041, 0.48018837976679474],
        "directions_summary": ("**Formal register — twins, unpaired** for `test/probe-era1` from a probe file: "
                               "3 unit directions, 8 wide, at blocks 0–2."),
    },
    "probe-era2-revision-paired.json": {
        "id": "2b0b5eed0000000000000000000000b2",
        "name": "Past vs present tense",
        "model_id": "test/probe-era2",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "precision": None,
        "positive_label": "Past",
        "negative_label": "Present",
        "pool": "mean",
        "chat_template": False,
        "l2": 2.0,
        "folds": 5,
        "paired": True,
        "created": 1790942400.25,
        "examples": {
            "positive": ["She walked home.", "They sang loudly.", "He wrote a letter.", "We ate early.",
                         "It rained all day."],
            "negative": ["She walks home.", "They sing loudly.", "He writes a letter.", "We eat early.",
                         "It rains all day."],
        },
        "width": 12,
        "best_layer": 2,
        "layers": [
            (0, -0.121982496, [0.12960872, -0.00801240833, -0.0297977279], 0.6, 0.5, 0.7181465108506827),
            (1, -0.157332037, [-0.0724794353, -0.139991678, 0.221763325], 0.6, 0.5, 0.8249037240933003),
            (2, -0.414638668, [-0.0653830588, 0.0572319763, -1.91092798], 1.0, 1.0, 0.3551724201647601),
            (3, -0.118531034, [0.0434681681, 0.107846536, 0.0293352835], 1.0, 0.9, 0.47469058821006166),
        ],
        "summary": (
            "**Past vs present tense** for test/probe-era2: Past (5 examples) against Present (5), read at the mean "
            "over every token of each example as plain text. Best layer 2: 100% accurate on held-out examples over "
            "5 folds with pairs held out together, 100% on the examples it was trained on."),
        "rows": [["0", "50%", 0.718, "60%"], ["1", "50%", 0.825, "60%"], ["2 (best)", "100%", 0.355, "100%"],
                 ["3", "90%", 0.475, "100%"]],
        "at_zero": [0.4695421337044596, 0.4607479257915611, 0.39780037888715275, 0.47040188687394885],
        "directions_summary": ("**Past vs present tense** for `test/probe-era2` from a probe file: "
                               "4 unit directions, 12 wide, at blocks 0–3."),
    },
    "probe-era3-precision.json": {
        "id": "3c0ffee0000000000000000000000c3d",
        "name": "Mentions an animal",
        "model_id": "test/tiny-decoder",
        "model_revision": "fedcba9876543210fedcba9876543210fedcba98",
        "precision": "8-bit",
        "positive_label": "Animal",
        "negative_label": "No animal",
        "pool": "last",
        "chat_template": True,
        "l2": 1.0,
        "folds": 3,
        "paired": False,
        "created": 1790949600.75,
        "examples": {
            "positive": ["the cat sat on the mat", "the dog sat on the log", "a quick brown fox", "the lazy dog",
                         "a cat and a dog", "the fox jumps"],
            "negative": ["hello world", "tokens, merges and bytes", "hello again"],
        },
        "width": 16,
        "best_layer": 1,
        "layers": [
            (0, 0.521436319, [0.051234991, 0.215428177, -0.400258091], 1.0, 1 / 3, 1.6244304154852978),
            (1, -0.102765706, [-0.250791265, -0.44830794, 0.0682732086], 1.0, 7 / 9, 0.6133820204832426),
            (2, 1.8780876, [0.239737328, 0.0792855596, -0.292272314], 1.0, 2 / 3, 0.5558253406941716),
            (3, 1.16945283, [-0.424349203, 0.315488607, 0.171374954], 1.0, 2 / 3, 0.6230548034328159),
        ],
        "summary": (
            "**Mentions an animal** for test/tiny-decoder: Animal (6 examples) against No animal (3), read at the "
            "last token of each example as a user message. Best layer 1: 78% accurate on held-out examples over "
            "3 folds, 100% on the examples it was trained on."),
        "rows": [["0", "33%", 1.624, "100%"], ["1 (best)", "78%", 0.613, "100%"], ["2", "67%", 0.556, "100%"],
                 ["3", "67%", 0.623, "100%"]],
        "at_zero": [0.6274835643510704, 0.4743311597947656, 0.8673913097865398, 0.7630460979370586],
        "directions_summary": ("**Mentions an animal** for `test/tiny-decoder` from a probe file: "
                               "4 unit directions, 16 wide, at blocks 0–3."),
    },
}

FIELDS = ("id", "name", "model_id", "model_revision", "precision", "positive_label", "negative_label", "pool",
          "chat_template", "l2", "folds", "paired", "created", "examples", "best_layer")


def page_context(directory):
    return ExtensionContext(ModelService(lambda: None), TokenInspector(), directory,
                            NavigationService(lambda *args: None))


class SavedProbeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def assert_reads_as_expected(self, probe, expected, name):
        # Compare every output to immutable source bytes, not to the parsed
        # value that the same reader/writer could already have corrupted.
        frozen = json.loads((FIXTURES / name).read_text())
        for loaded, saved in zip(probe["layers"], frozen["layers"], strict=True):
            self.assertEqual(loaded["weights"], saved["weights"])
        for field in FIELDS:
            self.assertEqual(probe[field], expected[field], field)
        self.assertEqual(type(probe["l2"]), float)
        self.assertEqual(len(probe["layers"]), len(expected["layers"]))
        for item, (layer, bias, weights, train, heldout, loss) in zip(probe["layers"], expected["layers"]):
            self.assertEqual(item["layer"], layer)
            self.assertEqual(len(item["weights"]), expected["width"])
            self.assertEqual(item["weights"][:3], weights)
            self.assertEqual(item["bias"], bias)
            self.assertEqual(item["train_accuracy"], train)
            self.assertAlmostEqual(item["heldout_accuracy"], heldout, places=15)
            self.assertEqual(item["heldout_loss"], loss)

    def test_each_era_reads_with_its_fields_and_layers(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                self.assert_reads_as_expected(probes.read(FIXTURES / name), expected, name)

    def test_older_eras_carry_none_of_the_later_fields(self):
        """The reader fills what an older writer never wrote: no revision, no precision."""
        raw = {name: json.loads((FIXTURES / name).read_text()) for name in EXPECTED}
        self.assertNotIn("model_revision", raw["probe-era1-no-revision-below-chance.json"])
        self.assertNotIn("precision", raw["probe-era1-no-revision-below-chance.json"])
        self.assertNotIn("precision", raw["probe-era2-revision-paired.json"])
        self.assertIn("precision", raw["probe-era3-precision.json"])

    def test_the_page_shows_the_summary_table_and_scale(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                probe = probes.read(FIXTURES / name)
                self.assertEqual(probe_page.probe_summary(probe), expected["summary"])
                self.assertEqual(probe_page.layer_rows(probe), expected["rows"])
                positive, negative = expected["positive_label"], expected["negative_label"]
                self.assertEqual(probe_page.scale_labels(probe),
                                 (f"{negative} (under 10%)", f"Leans {negative} (10–30%)", "Unsure (30–70%)",
                                  f"Leans {positive} (70–90%)", f"{positive} (over 90%)"))

    def test_readings_use_the_saved_weights_and_biases(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                frozen = json.loads((FIXTURES / name).read_text())
                probe = probes.read(FIXTURES / name)
                for loaded, saved in zip(probe["layers"], frozen["layers"], strict=True):
                    self.assertEqual(loaded["weights"], saved["weights"])
                    self.assertEqual(loaded["bias"], saved["bias"])
                rows = probes.directions(probe)
                self.assertEqual(rows.dtype, np.float32)
                self.assertEqual(rows.shape, (len(expected["layers"]), expected["width"]))
                np.testing.assert_array_equal(rows, np.asarray(
                    [layer["weights"] for layer in frozen["layers"]], dtype=np.float32))
                # With no projection a block's probability is the sigmoid of its bias alone.
                at_zero = probes.probabilities(probe, np.zeros((len(expected["layers"]), 2)))
                np.testing.assert_allclose(at_zero[:, 0], expected["at_zero"], rtol=1e-12)
                np.testing.assert_allclose(at_zero[:, 1], expected["at_zero"], rtol=1e-12)
                for (_, bias, *_), value in zip(expected["layers"], expected["at_zero"]):
                    self.assertAlmostEqual(value, 1 / (1 + math.exp(-bias)), places=12)
                best = probe["layers"][expected["best_layer"]]
                one = probes.probabilities(probe, np.asarray([[1.0]] * len(probe["layers"])))[expected["best_layer"], 0]
                self.assertAlmostEqual(one, 1 / (1 + math.exp(-(1.0 + best["bias"]))), places=12)

    def test_the_saved_list_orders_and_labels_every_era(self):
        for name in EXPECTED:
            shutil.copy(FIXTURES / name, self.directory / name)
        listed = probes.saved(self.directory)
        self.assertEqual([p["id"] for p in listed],
                         [EXPECTED[name]["id"] for name in ("probe-era3-precision.json",
                                                            "probe-era2-revision-paired.json",
                                                            "probe-era1-no-revision-below-chance.json")])
        choices = probe_page.saved_choices(self.directory)
        for (label, value), probe in zip(choices, listed):
            when = datetime.fromtimestamp(probe["created"]).strftime("%Y-%m-%d %H:%M")
            self.assertEqual(label, f"{probe['name']} · {probe['model_id']} · {when}")
            self.assertEqual(value, probe["id"])

    def test_importing_on_the_page_fills_the_form_and_resaves_readably(self):
        data = self.directory / "data"
        with gr.Blocks() as demo:
            probe_page.build_page(page_context(data))
        self.addCleanup(demo.close)
        fn = handlers_by_name(demo)
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                shown = fn["import_probe"](str(FIXTURES / name), "owner")
                (probe, summary, rows, picker, download, layer, as_user, _reading, _strip, _detail, _tokens, _heat,
                 form_name, positive_label, negative_label, positive_text, negative_text, chat_template, paired,
                 pool, l2) = shown
                self.assert_reads_as_expected(probe, expected, name)
                self.assertEqual(summary, expected["summary"])
                self.assertEqual(rows, expected["rows"])
                self.assertEqual(picker["value"], expected["id"])
                self.assertEqual(layer["maximum"], len(expected["layers"]) - 1)
                self.assertEqual(layer["value"], expected["best_layer"])
                self.assertEqual(as_user, expected["chat_template"])
                self.assertEqual((form_name, positive_label, negative_label),
                                 (expected["name"], expected["positive_label"], expected["negative_label"]))
                self.assertEqual(positive_text, "\n".join(expected["examples"]["positive"]))
                self.assertEqual(negative_text, "\n".join(expected["examples"]["negative"]))
                self.assertEqual((chat_template, paired, pool, l2),
                                 (expected["chat_template"], expected["paired"], expected["pool"], expected["l2"]))
                # The page saved it with today's writer; opening that and the download reads the same.
                self.assertTrue((data / f"{expected['id']}.json").is_file())
                reopened = fn["open_saved"](expected["id"], "owner")[0]
                self.assert_reads_as_expected(reopened, expected, name)
                self.assert_reads_as_expected(probes.read(download), expected, name)

    def test_resaving_with_todays_writer_keeps_the_facts(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                path = self.directory / name
                path.write_text(probes.dumps(probes.read(FIXTURES / name)))
                again = probes.read(path)
                self.assert_reads_as_expected(again, expected, name)
                self.assertEqual(probe_page.probe_summary(again), expected["summary"])


class SavedProbeAsDirectionsTests(unittest.TestCase):
    """Direction edits takes each layer's weights, scaled to unit length, as that block's direction."""

    def test_each_era_reads_as_unit_directions(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                probe = probes.read(FIXTURES / name)
                directions = edit_files.read_directions(FIXTURES / name)
                self.assertEqual(directions["source"], "chatlab-probe-1")
                self.assertEqual(directions["format"], "chatlab-directions-1")
                self.assertEqual((directions["name"], directions["model_id"]), (expected["name"], expected["model_id"]))
                self.assertEqual(directions["model_revision"], expected["model_revision"])
                self.assertEqual(directions["precision"], expected["precision"])
                self.assertEqual([item["layer"] for item in directions["directions"]],
                                 list(range(len(expected["layers"]))))
                for item, layer in zip(directions["directions"], probe["layers"]):
                    vector = np.asarray(item["vector"])
                    self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)
                    weights = np.asarray(layer["weights"])
                    np.testing.assert_allclose(vector, weights / np.linalg.norm(weights), rtol=1e-12)
                self.assertEqual(edits_page.directions_summary(directions), expected["directions_summary"])

    def test_the_edits_page_opens_each_era(self):
        with tempfile.TemporaryDirectory() as directory, gr.Blocks() as demo:
            edits_page.build_page(page_context(Path(directory)))
        self.addCleanup(demo.close)
        fn = handlers_by_name(demo)
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                directions, note = fn["open_directions"](str(FIXTURES / name))
                self.assertEqual(note, expected["directions_summary"])
                self.assertEqual(directions["model_revision"], expected["model_revision"])
                self.assertEqual(len(directions["directions"][0]["vector"]), expected["width"])

    def test_model_checks_use_the_revision_each_era_recorded(self):
        era1 = edit_files.read_directions(FIXTURES / "probe-era1-no-revision-below-chance.json")
        # No revision recorded: any revision of the same model is accepted.
        edit_files.check_model(era1, "test/probe-era1", "a" * 40)
        era3 = edit_files.read_directions(FIXTURES / "probe-era3-precision.json")
        edit_files.check_model(era3, "test/tiny-decoder", None)
        edit_files.check_model(era3, "test/tiny-decoder", "fedcba9876543210fedcba9876543210fedcba98")
        with self.assertRaisesRegex(ValueError, "revision fedcba987654"):
            edit_files.check_model(era3, "test/tiny-decoder", "a" * 40)
        with self.assertRaisesRegex(ValueError, "made for test/probe-era1"):
            edit_files.check_model(era1, "test/other", None)


if __name__ == "__main__":
    unittest.main()
