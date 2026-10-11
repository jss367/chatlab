"""Direction edits files written by earlier ChatLab versions keep reading the way they did.

The fixtures under fixtures/saved/direction_edits were written once (see each
generator's header) and are never regenerated: a hand-or-script
``chatlab-directions-1`` file, the ``chatlab-steering-1`` vector Chat's
extraction wrote, and two ``chatlab-direction-edits-1`` results the Edits page
downloaded. The tests open them through the Edits page's own handlers and
readers and assert what the page shows.

``files.normalize_result`` accepts a result only after re-running
``experiment.normalize_inputs`` on its saved inputs and checking it against
``experiment.SETTINGS`` and ``experiment.READ_LIMIT``, so a change to any of
those can stop an old result opening. These tests are the guard on that.
"""
import json
import tempfile
import unittest
from pathlib import Path

import gradio as gr
import numpy as np

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.direction_edits import experiment, files, render
from chatlab.extensions.direction_edits import page as edits_page
from chatlab.steering import read_vector
from ui_support import handlers_by_name

SAVED = Path(__file__).resolve().parent / "fixtures" / "saved"
FIXTURES = SAVED / "direction_edits"
DIRECTIONS = FIXTURES / "directions-sparse-unnormalized.json"
VECTOR = FIXTURES / "steering-tiny-decoder-layer0.json"
FULL = FIXTURES / "direction-edits-lens-random-injection.json"
SPARSE = FIXTURES / "direction-edits-probe-add-no-injection.json"
PROBE = SAVED / "probes" / "probe-era3-precision.json"
REVISION = "fedcba9876543210fedcba9876543210fedcba98"
PASSAGE = "the cat sat on the mat and the dog sat on the log"
PASSAGE_TOKENS = ["the", " cat", " sat", " on", " the", " mat", " and", " the", " dog", " sat", " on", " the", " log"]

FULL_HEADLINE = (
    "Recovery is the mean over the edited tokens of (edited − reference), divided by the mean of "
    "(injected − reference), at each block.\n\n"
    "- **neutral**: 97% of the injected shift is back at block 2, the last one read; the most is 97%, at block 2.\n"
    "- **ignore**: 97% of the injected shift is back at block 2, the last one read; the most is 97%, at block 2.\n"
    "- **focus**: 97% of the injected shift is back at block 2, the last one read; the most is 97%, at block 2.")
FULL_RECOVERY = (
    '<div class="de-table"><table><thead><tr><th>Block</th><th>neutral</th><th>random</th><th>ignore</th>'
    "<th>random</th><th>focus</th><th>random</th></tr></thead><tbody>"
    "<tr><th>0</th>" + "<td>100%</td>" * 6 + "</tr>"
    '<tr class="de-edited"><th>1 (edited)</th>' + "<td>0%</td><td>100%</td>" * 3 + "</tr>"
    "<tr><th>2</th>" + "<td>97%</td><td>116%</td>" * 3 + "</tr>"
    "<tr><th>3</th>" + "<td>–</td>" * 6 + "</tr></tbody></table></div>")
FULL_LENS = (
    '<div class="de-table"><table><thead><tr><th>Condition</th><th>No edit</th><th>Edit</th>'
    "<th>Random control</th></tr></thead><tbody>"
    "<tr><th>neutral</th><td>-5.8038</td><td>-5.8045</td><td>-5.8085</td></tr>"
    "<tr><th>ignore</th><td>-5.8080</td><td>-5.8089</td><td>-5.8124</td></tr>"
    "<tr><th>focus</th><td>-5.8055</td><td>-5.8063</td><td>-5.8100</td></tr></tbody></table></div>")
FULL_DIFFERENCES = (
    '<div class="de-table"><table><thead><tr><th>Difference</th><th>No edit</th><th>Edit</th><th>Change</th>'
    "<th>Random control</th><th>Change</th></tr></thead><tbody>"
    "<tr><th>suppression: neutral − ignore</th><td>0.0042</td><td>0.0044</td><td>+4.7%</td><td>0.0039</td>"
    "<td>-7.4%</td></tr>"
    "<tr><th>focus: focus − neutral</th><td>-0.0017</td><td>-0.0017</td><td>-5.1%</td><td>-0.0015</td>"
    "<td>+8.2%</td></tr></tbody></table></div>")
FULL_TARGET = ("The target is one token, ` sat`. Mean log probability through the lens lens.pt over tokens 1–8 "
               "and blocks 1–3.")
SPARSE_HEADLINE = ("Nothing was injected, so the injected pass is the reference and recovery is undefined. "
                   "Read the heatmaps and the lens table instead.")
SPARSE_RECOVERY = (
    '<div class="de-table"><table><thead><tr><th>Block</th><th>plain</th><th>random</th><th>greeting</th>'
    "<th>random</th></tr></thead><tbody>"
    "<tr><th>0</th>" + "<td>–</td>" * 4 + "</tr><tr><th>1</th>" + "<td>–</td>" * 4 + "</tr>"
    '<tr class="de-edited"><th>2 (edited)</th>' + "<td>–</td>" * 4 + "</tr>"
    '<tr class="de-edited"><th>3 (edited)</th>' + "<td>–</td>" * 4 + "</tr></tbody></table></div>")


def build_edits_page(test, directory):
    context = ExtensionContext(ModelService(lambda: None), TokenInspector(), directory,
                               NavigationService(lambda *args: None))
    with gr.Blocks() as demo:
        edits_page.build_page(context)
    test.addCleanup(demo.close)
    return handlers_by_name(demo)


class SavedInputTests(unittest.TestCase):
    """The two files the Edits page opens before a run."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_the_direction_file_reads_as_sorted_unit_directions(self):
        raw = json.loads(DIRECTIONS.read_text())
        directions = files.read_directions(DIRECTIONS)
        self.assertEqual(directions["name"], "dog detector")  # stripped of its spaces
        self.assertEqual((directions["model_id"], directions["model_revision"], directions["precision"]),
                         ("test/tiny-decoder", REVISION, "full"))
        self.assertEqual(directions["source"], "chatlab-directions-1")
        self.assertEqual([item["layer"] for item in raw["directions"]], [2, 0, 1])
        self.assertEqual([item["layer"] for item in directions["directions"]], [0, 1, 2])
        given = {item["layer"]: np.asarray(item["vector"], dtype=np.float64) for item in raw["directions"]}
        for item in directions["directions"]:
            vector = np.asarray(item["vector"])
            self.assertEqual(len(vector), 16)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)
            np.testing.assert_allclose(vector, given[item["layer"]] / np.linalg.norm(given[item["layer"]]), rtol=1e-12)
        np.testing.assert_allclose([item["vector"][:2] for item in directions["directions"]],
                                   [[0.6008031984206765, -0.15862783692427423],
                                    [0.09686883023408148, 0.09390669507928814],
                                    [0.13314311613812532, 0.03419455613228058]], rtol=1e-12)
        # Block 3 has no direction: the matrix holds a zero row there and it cannot be edited.
        rows, layers = files.matrix(directions, 4)
        self.assertEqual(layers, [0, 1, 2])
        self.assertFalse(rows[3].any())

    def test_the_steering_vector_reads_with_its_block_and_numbers(self):
        vector = read_vector(VECTOR)
        self.assertEqual((vector["format"], vector["model_id"], vector["layer"], vector["strength"], vector["enabled"]),
                         ("chatlab-steering-1", "test/tiny-decoder", 0, 1.0, True))
        self.assertEqual(len(vector["vector"]), 16)
        self.assertEqual(vector["vector"][:3], [0.0025587176593641443, 0.0030624292946110163, -0.0014285603053091718])
        self.assertEqual(vector["vector"][-1], 0.01855407643597573)

    def test_the_page_opens_both_and_gathers_them_into_inputs(self):
        fn = build_edits_page(self, self.directory)
        directions, note = fn["open_directions"](str(DIRECTIONS))
        self.assertEqual(note, "**dog detector** for `test/tiny-decoder`: 3 unit directions, 16 wide, at blocks 0–2.")
        vector, vector_note, strength = fn["open_vector"](str(VECTOR))
        self.assertEqual(vector_note, "Injecting at block 0: a vector for `test/tiny-decoder`, 16 wide.")
        self.assertEqual(strength, 1.0)
        inputs = edits_page.gather(PASSAGE, directions, vector, 80.0, 2, 6, experiment.ERASE, 0.0, 1, 1, 3, 5,
                                   True, 7, " sat", 1, 8, 1, 3, "suppression = neutral - ignore",
                                   [("neutral", ""), ("ignore", "Do not think about the dog.\n")])
        self.assertEqual(inputs["injection"]["vector"]["vector"], read_vector(VECTOR)["vector"])
        self.assertEqual(inputs["injection"]["vector"]["layer"], 0)
        self.assertEqual(inputs["injection"]["tokens"], [2, 6])
        self.assertEqual([item["layer"] for item in inputs["directions"]["directions"]], [0, 1, 2])
        np.testing.assert_allclose([item["vector"] for item in inputs["directions"]["directions"]],
                                   [item["vector"] for item in directions["directions"]], rtol=1e-12)

    def test_a_probe_file_opens_as_directions(self):
        fn = build_edits_page(self, self.directory)
        directions, note = fn["open_directions"](str(PROBE))
        self.assertEqual(note, "**Mentions an animal** for `test/tiny-decoder` from a probe file: 4 unit directions, "
                               "16 wide, at blocks 0–3.")
        self.assertEqual((directions["source"], directions["model_revision"], directions["precision"]),
                         ("chatlab-probe-1", REVISION, "8-bit"))


class SavedResultTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def assert_full_result(self, result):
        self.assertEqual(result["created"], 1791460800.5)
        self.assertEqual(result["model"], {"model_id": "test/tiny-decoder", "model_revision": REVISION,
                                           "precision": "full"})
        self.assertEqual(result["lens"], {"name": "lens.pt", "n_prompts": 10})
        self.assertEqual(result["passage_tokens"], PASSAGE_TOKENS)
        self.assertEqual(result["passage_ids"], [258, 303, 272, 274, 262, 315, 312, 262, 275, 272, 274, 262, 319])
        self.assertEqual(result["edited_blocks"], [1])
        self.assertEqual(result["target"], {"word": " sat", "token_ids": [272], "tokens": [" sat"]})
        inputs = result["inputs"]
        self.assertEqual(inputs["passage"], PASSAGE)
        self.assertEqual(inputs["conditions"], [{"name": "neutral", "prefix": ""},
                                                {"name": "ignore", "prefix": "Do not think about the dog.\n"},
                                                {"name": "focus", "prefix": "the dog\n"}])
        self.assertEqual(inputs["edit"], {"mode": "erase", "value": 0.0, "blocks": [1, 1], "tokens": [3, 5],
                                          "random_control": True, "seed": 7})
        self.assertEqual(inputs["readout"], {"word": " sat", "tokens": [1, 8], "blocks": [1, 3]})
        self.assertEqual(inputs["differences"], [{"name": "suppression", "minuend": "neutral", "subtrahend": "ignore"},
                                                 {"name": "focus", "minuend": "focus", "subtrahend": "neutral"}])
        self.assertEqual((inputs["injection"]["strength"], inputs["injection"]["tokens"]), (80.0, [2, 6]))
        # The vector and directions the run used are the input fixtures, as the page opened them.
        self.assertEqual(inputs["injection"]["vector"]["vector"], read_vector(VECTOR)["vector"])
        opened = files.read_directions(DIRECTIONS)
        self.assertEqual({key: value for key, value in inputs["directions"].items() if key != "directions"},
                         {key: value for key, value in opened.items() if key != "directions"})
        np.testing.assert_allclose([item["vector"] for item in inputs["directions"]["directions"]],
                                   [item["vector"] for item in opened["directions"]], rtol=1e-12)
        conditions = result["conditions"]
        self.assertEqual([c["name"] for c in conditions], ["neutral", "ignore", "focus"])
        self.assertEqual([len(c["prefix_ids"]) for c in conditions], [0, 21, 3])
        self.assertEqual(conditions[2]["prefix_ids"], [258, 275, 199])
        for condition in conditions:
            passes = condition["coordinates"]
            self.assertEqual(set(passes), {"reference", "injected", "edited", "random"})
            for rows in passes.values():
                self.assertEqual(len(rows), 4)
                self.assertIsNone(rows[3])  # no direction at block 3
                self.assertTrue(all(len(row) == 13 for row in rows[:3]))
            reference = np.asarray(passes["reference"][:3])
            # The injection moved block 0's coordinate by strength × (vector · unit direction) over tokens 2–6 only.
            shift = np.asarray(passes["injected"][0]) - reference[0]
            np.testing.assert_allclose(shift[1:6], 0.36835462625691007, rtol=1e-5)
            np.testing.assert_allclose(shift[[0, *range(6, 13)]], 0.0, atol=1e-6)
            # The erase put block 1's coordinate back at the reference over tokens 3–5.
            erased = np.asarray(passes["edited"][1]) - reference[1]
            np.testing.assert_allclose(erased[2:5], 0.0, atol=1e-6)
            self.assertGreater(abs(erased[1]), 0.3)
        self.assertEqual(conditions[0]["coordinates"]["reference"][1][:2],
                         [-0.01617208495736122, -0.008645675145089626])
        self.assertEqual(conditions[1]["coordinates"]["edited"][2][4], -0.7714858651161194)
        self.assertEqual(conditions[2]["coordinates"]["random"][2][4], -0.9269167184829712)
        self.assertEqual(conditions[0]["recovery"],
                         {"edited": [1.0, -7.047630846199384e-08, 0.970614021546946, None],
                          "random": [1.0, 0.9999999754865013, 1.15917184979633, None]})
        self.assertEqual(conditions[1]["lens"], {"no_edit": -5.808011590513825, "edit": -5.808869750817249,
                                                 "random": -5.812383149687032})
        self.assertEqual(result["differences"][0],
                         {"name": "suppression", "minuend": "neutral", "subtrahend": "ignore",
                          "no_edit": 0.004163396091464655, "edit": 0.004357933843811779,
                          "random": 0.003855837082826774, "edit_change": 4.67257373724167,
                          "random_change": -7.387214713209854})

    def assert_sparse_result(self, result):
        self.assertEqual(result["created"], 1791464400.125)
        self.assertEqual(result["model"]["model_revision"], REVISION)
        self.assertIsNone(result["lens"])
        self.assertIsNone(result["target"])
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["edited_blocks"], [2, 3])
        self.assertEqual(result["passage_tokens"],
                         ["a", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog"])
        inputs = result["inputs"]
        self.assertIsNone(inputs["injection"])
        self.assertIsNone(inputs["readout"])
        self.assertEqual(inputs["edit"], {"mode": "add", "value": 2.5, "blocks": [2, 3], "tokens": [4, 9],
                                          "random_control": True, "seed": 11})
        # Opened from a probe, but the page re-normalized the directions on the way into the run,
        # so the saved inputs record a direction set; the name, revision and precision are the probe's.
        directions = inputs["directions"]
        self.assertEqual((directions["source"], directions["name"], directions["model_revision"],
                          directions["precision"]),
                         ("chatlab-directions-1", "Mentions an animal", REVISION, "8-bit"))
        self.assertEqual([item["layer"] for item in directions["directions"]], [0, 1, 2, 3])
        probe = files.read_directions(PROBE)
        np.testing.assert_allclose([item["vector"] for item in directions["directions"]],
                                   [item["vector"] for item in probe["directions"]], rtol=1e-12)
        self.assertEqual([c["name"] for c in result["conditions"]], ["plain", "greeting"])
        for condition in result["conditions"]:
            passes = condition["coordinates"]
            self.assertEqual(passes["injected"], passes["reference"])  # nothing injected
            self.assertEqual(condition["recovery"], {"edited": [None] * 4, "random": [None] * 4})
            self.assertEqual(condition["lens"], {"no_edit": None, "edit": None, "random": None})
            added = np.asarray(passes["edited"][2]) - np.asarray(passes["reference"][2])
            np.testing.assert_allclose(added[3:9], 2.5, atol=1e-5)  # tokens 4–9
            np.testing.assert_allclose(added[:3], 0.0, atol=1e-7)
            # The random control moved block 2 at right angles to its direction.
            np.testing.assert_allclose(np.asarray(passes["random"][2]) - np.asarray(passes["reference"][2]), 0.0,
                                       atol=1e-5)
        self.assertEqual(result["conditions"][0]["coordinates"]["reference"][0][0], -0.022713014855980873)

    def test_the_full_result_reads(self):
        self.assert_full_result(files.read_result(FULL))

    def test_the_sparse_result_reads(self):
        self.assert_sparse_result(files.read_result(SPARSE))

    def test_the_full_result_renders_as_the_page_showed_it(self):
        result = files.read_result(FULL)
        self.assertEqual(render.headline(result), FULL_HEADLINE)
        self.assertEqual(render.recovery_table(result), FULL_RECOVERY)
        self.assertEqual(render.lens_table(result), FULL_LENS)
        self.assertEqual(render.differences_table(result), FULL_DIFFERENCES)
        self.assertEqual(render.target_note(result), FULL_TARGET)
        heat = render.heatmaps(result, 1, {"sequential": ["#000000", "#ffffff"]}, lambda text: text)
        self.assertIn("<b>ignore</b>", heat)
        for label in ("Reference, no injection", "Injected", "Injected and edited", "Injected, random control"):
            self.assertIn(f"<h4>{label}</h4>", heat)
        self.assertIn(" sat · token 3 · block 1 · ", heat)
        self.assertEqual(heat.count('<td class="de-none" colspan="13"></td>'), 4)  # block 3 in each pass
        self.assertEqual(heat.count('<tr class="de-edited"><th>block 1</th>'), 4)

    def test_the_sparse_result_renders_as_the_page_showed_it(self):
        result = files.read_result(SPARSE)
        self.assertEqual(render.headline(result), SPARSE_HEADLINE)
        self.assertEqual(render.recovery_table(result), SPARSE_RECOVERY)
        self.assertEqual(render.lens_table(result), "")
        self.assertEqual(render.target_note(result), "")
        self.assertEqual(render.differences_table(result), "")
        heat = render.heatmaps(result, 0, {"sequential": ["#000000", "#ffffff"]}, lambda text: text)
        self.assertIn("<b>plain</b>", heat)
        self.assertNotIn('class="de-none"', heat)
        self.assertNotIn("<th>injected</th>", heat)  # no injection mark row
        self.assertNotIn("<th>read</th>", heat)

    def test_the_page_opens_each_result_and_its_download_reads_back(self):
        fn = build_edits_page(self, self.directory)
        for path, check, headline, names in ((FULL, self.assert_full_result, FULL_HEADLINE,
                                              ["neutral", "ignore", "focus"]),
                                             (SPARSE, self.assert_sparse_result, SPARSE_HEADLINE,
                                              ["plain", "greeting"])):
            with self.subTest(path=path.name):
                status, result, shown_headline, picker, heat, recovery, _lens_note, _lens, _diffs, download = \
                    fn["open_result"](str(path), "owner")
                self.assertEqual(status, "Opened a result for `test/tiny-decoder`.")
                check(result)
                self.assertEqual(shown_headline, headline)
                self.assertEqual(picker["choices"], [(name, index) for index, name in enumerate(names)])
                self.assertIn(f"<b>{names[0]}</b>", heat)
                self.assertIn(f"<b>{names[1]}</b>", fn["pick"](result, 1))
                # The page staged a download with today's writer; it opens to the same result.
                check(files.read_result(download))
                self.assertEqual(json.loads(Path(download).read_text()), json.loads(path.read_text()))
                self.assertIn("1 (edited)" if path == FULL else "2 (edited)", recovery)

    def test_resaving_with_todays_writer_keeps_the_facts(self):
        for path, check in ((FULL, self.assert_full_result), (SPARSE, self.assert_sparse_result)):
            with self.subTest(path=path.name):
                copy = self.directory / path.name
                copy.write_text(files.dumps(files.read_result(path)))
                check(files.read_result(copy))

    def test_saved_inputs_still_pass_todays_input_checks(self):
        """The re-derivation normalize_result relies on, made explicit."""
        for path in (FULL, SPARSE):
            with self.subTest(path=path.name):
                saved = json.loads(path.read_text())
                checked = experiment.normalize_inputs(saved["inputs"])
                self.assertEqual(checked["edit"], saved["inputs"]["edit"])
                self.assertEqual(checked["readout"], saved["inputs"]["readout"])
                self.assertEqual(checked["differences"], saved["inputs"]["differences"])
                self.assertEqual(checked["conditions"], saved["inputs"]["conditions"])
                self.assertLessEqual(len(saved["passage_ids"]), experiment.READ_LIMIT)
                for condition in saved["conditions"]:
                    self.assertEqual(set(condition["lens"]), set(experiment.SETTINGS))


if __name__ == "__main__":
    unittest.main()
