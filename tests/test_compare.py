import json
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import charts
import compare
import settings_sandbox
from model_runtime import LoadedModel, ModelChanged
from test_streaming import EOS_ID, loaded_manager
from ui import compare as controls
from ui import runtime


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def metric(position, token_id, surprise, *, top="a", scored=True, entropy=1.0, rank=1):
    """One token's measurements, in the shape the runtime publishes them."""

    return {
        "position": position,
        "token_id": token_id,
        "text": f"t{token_id}",
        "display_text": f"t{token_id}",
        "category": "Top choice",
        "raw_rank": rank,
        "raw_probability": 0.5,
        "sampling_probability": 0.5,
        "surprise_bits": surprise,
        "probability_mass_above": 0.0,
        "entropy_bits": entropy,
        "top1_margin": 0.1,
        "sampling_shift_bits": 0.0,
        "top_candidates": [{"token_id": 1, "text": top, "probability": 0.5}],
        "scored": scored,
        "segment": "response",
        "unscored_reason": "",
    }


def run(metrics, *, kind=compare.REPLY, model_id="fake/model", **settings):
    return {
        "kind": kind,
        "model_id": model_id,
        "load_id": f"{model_id}#1",
        "device_name": "CPU",
        "precision": "float32",
        "prompt": "hello",
        "text": "hello",
        "metrics": metrics,
        "settings": {
            "system_prompt": "",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 8,
            "seed": 1,
            "assistant_prefill": "",
            "thinking_mode": "default",
            "steering": None,
        } | settings,
        "seconds": 0.1,
    }


class ReadingTests(unittest.TestCase):
    def test_shared_prefix_stops_at_the_first_different_token(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0), metric(3, 7, 1.0)]
        right = [metric(1, 5, 1.0), metric(2, 9, 1.0), metric(3, 7, 1.0)]
        # The third token matches again, but its context no longer does.
        self.assertEqual(compare.shared_prefix(left, right), 1)
        self.assertEqual(compare.shared_prefix(left, left), 3)
        self.assertEqual(compare.shared_prefix(left, []), 0)

    def test_gaps_and_categories_follow_the_surprise_difference(self):
        left = [metric(1, 5, 1.0, top="a"), metric(2, 6, 2.0, top="a")]
        right = [metric(1, 5, 1.2, top="a"), metric(2, 6, 9.0, top="b")]
        readings = compare.gaps(left, right, 2)
        self.assertAlmostEqual(readings[0]["surprise_bits"], 0.2)
        self.assertAlmostEqual(readings[1]["surprise_bits"], 7.0)
        self.assertEqual(compare.gap_category(readings[0]["surprise_bits"]), compare.GAP_LABELS[0])
        self.assertEqual(compare.gap_category(readings[1]["surprise_bits"]), compare.GAP_LABELS[-1])
        self.assertEqual(readings[1]["left_top"], "a")
        self.assertEqual(readings[1]["right_top"], "b")

    def test_an_unscored_token_gets_no_gap_and_no_color(self):
        left = [metric(1, 5, 0.0, scored=False), metric(2, 6, 1.0)]
        right = [metric(1, 5, 0.0, scored=False), metric(2, 6, 1.0)]
        readings = compare.gaps(left, right, 2)
        self.assertFalse(readings[0]["scored"])
        self.assertIsNone(readings[0]["left_surprise"])
        painted = compare.strip(left, 2, readings)
        self.assertEqual(painted[0][1], "Not predicted")
        self.assertEqual(painted[1][1], compare.GAP_LABELS[0])

    def test_tokens_after_the_split_are_painted_as_the_split(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [metric(1, 5, 1.0), metric(2, 9, 1.0)]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["shared"], 1)
        self.assertFalse(reading["complete"])
        painted = compare.strip(left, reading["shared"], reading["readings"])
        self.assertEqual([label for _text, label in painted][1], compare.SPLIT_LABEL)
        self.assertIn("parted", compare.headline(reading, run(left), run(right)))

    def test_a_measurement_pair_is_compared_to_the_last_token(self):
        left = [metric(index, index, 1.0) for index in range(1, 5)]
        right = [metric(index, index, 3.0) for index in range(1, 5)]
        reading = compare.reading(
            run(left, kind=compare.MEASUREMENT), run(right, kind=compare.MEASUREMENT)
        )
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["compared"], 4)
        self.assertAlmostEqual(reading["mean_gap_bits"], 2.0)
        self.assertIn("same 4 tokens", compare.headline(reading, None, None))

    def test_top_choice_changes_are_counted_over_the_shared_tokens_only(self):
        left = [metric(1, 5, 1.0, top="a"), metric(2, 6, 1.0, top="a")]
        right = [metric(1, 5, 1.0, top="b"), metric(2, 7, 1.0, top="z")]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["compared"], 1)
        self.assertEqual(reading["top_choice_changed"], 1)

    def test_two_models_are_lined_up_on_text_not_on_token_ids(self):
        # The same two IDs stand for unrelated text in two vocabularies, so
        # matching on them would subtract one model's reading from another's.
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [dict(metric(1, 5, 4.0), text="other", display_text="other"),
                 metric(2, 6, 1.0)]
        reading = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertTrue(reading["cross_model"])
        self.assertEqual(reading["shared"], 0)
        self.assertIn("different models", compare.headline(reading, None, None))
        # Matching IDs in one model still count, and the caveat stays away.
        same = compare.reading(run(left), run(right))
        self.assertFalse(same["cross_model"])
        self.assertEqual(same["shared"], 2)
        self.assertNotIn("different models", compare.headline(same, None, None))

    def test_two_tokenizers_that_cut_a_passage_differently_stop_the_alignment(self):
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "hel"), piece(2, 2, "lo"), piece(3, 3, "!")]
        right = [piece(1, 9, "hello"), piece(2, 8, "!"), piece(3, 7, "?")]
        reading = compare.reading(run(left), run(right, model_id="other/model"))
        # The characters agree; the boundaries do not, and that is where the
        # two runs stop describing the same thing.
        self.assertEqual(reading["shared"], 0)
        self.assertEqual(
            compare.shared_prefix(left, [piece(1, 9, "hel"), piece(2, 8, "lo")], by_text=True), 2
        )

    def test_a_measurement_records_no_system_prompt_to_differ_over(self):
        # score_text has nowhere to put a system message, so recording one
        # would have the table report a difference neither run saw.
        left = run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, use_chat_template=False)
        right = run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, use_chat_template=True)
        self.assertNotIn("System prompt", compare.configuration(left))
        self.assertIn("System prompt", compare.configuration(run([metric(1, 5, 1.0)])))
        self.assertEqual(
            [row[0] for row in compare.configuration_rows(left, right)],
            ["Context read as"],
        )

    def test_configuration_rows_name_only_what_differed(self):
        left = run([metric(1, 5, 1.0)], seed=1)
        right = run([metric(1, 5, 1.0)], seed=2, temperature=0.7)
        rows = compare.configuration_rows(left, right)
        self.assertEqual({row[0] for row in rows}, {"Seed", "Temperature"})
        self.assertEqual(
            len(compare.configuration_rows(left, right, differences_only=False)),
            len(compare.configuration(left)),
        )

    def test_a_steered_run_says_so_in_its_configuration(self):
        steered = run(
            [metric(1, 5, 1.0)],
            steering={"layer": 7, "strength": 2.0, "enabled": True, "vector_id": "ab" * 32},
        )
        reading = compare.configuration(steered)
        self.assertIn("layer 7", reading["Steering"])
        self.assertIn("strength 2", reading["Steering"])
        self.assertEqual(compare.configuration(run([metric(1, 5, 1.0)]))["Steering"], "off")

    def test_divergence_rows_are_ordered_widest_first(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0), metric(3, 7, 1.0)]
        right = [metric(1, 5, 1.5), metric(2, 6, 5.0), metric(3, 7, 1.1)]
        rows = compare.divergence_rows(compare.gaps(left, right, 3))
        self.assertEqual([row[0] for row in rows], [2, 1, 3])
        self.assertEqual(rows[0][4], 4.0)
        self.assertEqual(len(compare.divergence_rows(compare.gaps(left, right, 3), limit=1)), 1)

    def test_export_holds_both_runs_and_every_shared_token(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [metric(1, 5, 2.0), metric(2, 6, 1.0)]
        left_run, right_run = run(left), run(right)
        reading = compare.reading(left_run, right_run)
        document = compare.export(left_run, right_run, reading)
        self.assertEqual(document["a"]["model_id"], "fake/model")
        self.assertEqual(len(document["shared_tokens"]), 2)
        self.assertEqual(document["shared_tokens"][0]["gap_bits"], 1.0)
        # Serializable as it stands: the download writes exactly this.
        json.dumps(document)

    def test_an_empty_pair_reads_as_empty_everywhere(self):
        self.assertEqual(compare.reading(None, run([metric(1, 5, 1.0)])), {})
        self.assertIn("Fill both slots", compare.headline({}, None, None))
        self.assertIn("viz-empty", charts.comparison_tiles({}))
        self.assertIn("Empty", compare.describe(None, "A"))


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime.MANAGER
        manager = loaded_manager([0, 1, EOS_ID])
        manager._loaded = LoadedModel("fake/model", "CPU", "float32", manager.load_id)
        runtime.MANAGER = manager
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def fill(self, slot, mode=compare.REPLY, prompt="Hello", measured=""):
        frames = list(controls.fill_slot(
            slot, mode, prompt, measured, False, "", "", 0.0, 1.0, 0, 8, 1, False,
            "default", None, False, 1.0, 0,
        ))
        return frames[-1]

    def test_a_measurement_run_keeps_no_system_prompt_in_its_settings(self):
        held, _status, *_buttons = self.fill(
            "A", mode=compare.MEASUREMENT, prompt="", measured="Hello world"
        )
        self.assertEqual(held["kind"], compare.MEASUREMENT)
        self.assertNotIn("system_prompt", held["settings"])

    def test_a_reply_fills_its_slot_with_the_settings_it_ran_under(self):
        held, status, *_buttons = self.fill("A")
        self.assertEqual(held["kind"], compare.REPLY)
        self.assertEqual(held["model_id"], "fake/model")
        self.assertEqual(held["settings"]["seed"], 1)
        self.assertEqual(held["slot"], "A")
        self.assertTrue(held["metrics"])
        self.assertIn("Slot A filled", status)

    def test_an_empty_prompt_leaves_the_slot_alone_and_frees_the_model(self):
        held, status, *_buttons = self.fill("A", prompt="  ")
        self.assertEqual(held, gr.skip())
        self.assertEqual(status, controls.COMPARE_NO_PROMPT)
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_busy_model_refuses_without_touching_the_slot(self):
        self.assertTrue(runtime.MANAGER.reserve_generation())
        try:
            held, status, *_buttons = self.fill("B")
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(held, gr.skip())
        self.assertEqual(status, controls.COMPARE_BUSY)

    def test_a_model_change_mid_run_leaves_the_slot_as_it_was(self):
        with mock.patch.object(
            runtime.MANAGER, "generate", side_effect=ModelChanged("gone")
        ):
            held, status, *_buttons = self.fill("A")
        self.assertEqual(held, gr.skip())
        self.assertIn("model changed", status)
        self.assertFalse(runtime.MANAGER.busy)

    def test_rendering_both_slots_draws_the_strips_and_the_tables(self):
        left, right = run([metric(1, 5, 1.0)]), run([metric(1, 5, 4.0)])
        (a_heading, b_heading, a_strip, b_strip, tiles, chart, headline,
         settings_table, rows_table, export_state) = controls.render(left, right)
        self.assertIn("fake/model", a_heading)
        self.assertIn("fake/model", b_heading)
        self.assertEqual(a_strip["color_map"], compare.GAP_COLORS)
        self.assertEqual(len(a_strip["value"]), 1)
        self.assertEqual(len(b_strip["value"]), 1)
        self.assertIn("viz-tiles", tiles)
        self.assertIn("same 1 token", headline)
        self.assertEqual(export_state["reading"]["shared"], 1)
        self.assertEqual(settings_table["value"], [])
        self.assertEqual(rows_table["value"][0][4], 3.0)

    def test_clearing_empties_both_slots(self):
        left, right, status, *drawn = controls.clear_slots()
        self.assertIsNone(left)
        self.assertIsNone(right)
        self.assertEqual(status, controls.COMPARE_EMPTY)
        self.assertIn("Empty", drawn[0])
        self.assertEqual(drawn[-1], {"left": None, "right": None, "reading": {}})

    def test_the_download_writes_the_document_only_when_both_slots_are_full(self):
        self.assertIsNone(controls.download_comparison(None))
        self.assertIsNone(controls.download_comparison({"left": run([]), "right": None}))
        left, right = run([metric(1, 5, 1.0)]), run([metric(1, 5, 2.0)])
        held = {"left": left, "right": right, "reading": compare.reading(left, right)}
        path = Path(controls.download_comparison(held))
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        document = json.loads(path.read_text())
        self.assertEqual(document["b"]["model_id"], "fake/model")
        self.assertEqual(document["shared_tokens"][0]["gap_bits"], 1.0)

    def test_the_mode_switch_renames_the_prompt_box_and_shows_the_passage(self):
        prompt, text, template = controls.mode_controls(compare.MEASUREMENT)
        self.assertEqual(prompt["label"], "Context (optional)")
        self.assertTrue(text["visible"])
        self.assertTrue(template["visible"])
        prompt, text, template = controls.mode_controls(compare.REPLY)
        self.assertEqual(prompt["label"], "Prompt for both runs")
        self.assertFalse(text["visible"])


if __name__ == "__main__":
    unittest.main()
