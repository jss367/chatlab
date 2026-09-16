"""Fixed-seed word removal, comparison rendering, and real Gradio wiring."""

import asyncio
import threading
import unittest
from dataclasses import replace
from unittest import mock

import gradio as gr
import numpy as np

import app
import image_runtime
import settings_sandbox
from fake_pipeline import FakePipeline
from image_runtime import ImageRequest, ImageRun
from model_runtime import IMAGE_KIND, ModelManager
from ui import image_words, images_page, runtime


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


ROW = {name: i for i, name in enumerate(image_words.OUTPUT_NAMES)}


def word_index(run, word, occurrence=0):
    return [i for i, part in enumerate(image_words.prompt_parts(run.request.prompt))
            if part.group() == word][occurrence]


class WordComparisonTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()
        self.manager.pipeline = FakePipeline()
        self.manager.kind = IMAGE_KIND
        self.manager.model_id = "org/image"
        self.manager.device_name = "CPU"
        self.patch = mock.patch.object(runtime, "MANAGER", self.manager)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.original = self.manager.generate_image(ImageRequest(
            "a red bicycle", negative_prompt="blur", steps=3, guidance_scale=4.5,
            seed=9876, width=64, height=32,
        ))

    def compare(self, original=None, word="red", pair=None):
        original = self.original if original is None else original
        return list(image_words.compare_word(pair or (original, None), word_index(original, word)))

    def test_only_the_prompt_changes_and_the_original_remains_intact(self):
        with mock.patch.object(self.manager, "generate_image", wraps=self.manager.generate_image) as generate:
            frames = self.compare()
        for frame in frames:
            self.assertEqual(len(frame), len(image_words.OUTPUT_NAMES))
        original, removed = frames[-1][ROW["pair"]]
        self.assertIs(original, self.original)
        self.assertEqual(original.request.prompt, "a red bicycle")
        self.assertEqual(removed.request, replace(original.request, prompt="a bicycle"))
        self.assertEqual(removed.load_id, original.load_id)
        self.assertEqual(removed.image.size, (64, 32))
        self.assertEqual(generate.call_args.kwargs["expected_load_id"], original.load_id)
        # FakePipeline's decode depends only on the starting noise. Matching pixels
        # prove the actual generator reused its seed, beyond checking metadata.
        np.testing.assert_array_equal(np.asarray(removed.image), np.asarray(original.image))
        self.assertEqual(len(removed.tokens), len(original.tokens) - 1)
        self.assertFalse(frames[0][ROW["draw"]]["visible"])
        self.assertTrue(frames[0][ROW["stop"]]["visible"])
        self.assertTrue(frames[-1][ROW["draw"]]["visible"])
        self.assertFalse(self.manager.busy)

    def test_each_experiment_starts_from_the_original(self):
        first = self.compare()[-1][ROW["pair"]]
        second = self.compare(word="bicycle", pair=first)[-1][ROW["pair"]]
        self.assertEqual(second[1].request.prompt, "a red")
        self.assertEqual(first[1].request.prompt, "a bicycle")

    def test_removes_only_the_selected_occurrence_preserving_punctuation(self):
        run = replace(self.original, request=replace(self.original.request, prompt="red, red bicycle!"))
        request = image_words.removal_request(run, word_index(run, "red", 1))
        self.assertEqual(request.prompt, "red, bicycle!")
        self.assertEqual("".join(text for text, _ in image_words.word_strip(run)), run.request.prompt)

    def test_words_do_not_depend_on_token_capture_or_subword_boundaries(self):
        run = replace(self.original, tokens=[], attention=None,
                      request=replace(self.original.request, prompt="a café-colored bicycle"))
        request = image_words.removal_request(run, word_index(run, "café-colored"))
        self.assertEqual(request.prompt, "a bicycle")

    def test_first_last_and_only_word_removal(self):
        for prompt, word, expected in (("red bicycle", "red", "bicycle"),
                                      ("a red", "red", "a"), ("red", "red", "")):
            with self.subTest(prompt=prompt):
                run = replace(self.original, request=replace(self.original.request, prompt=prompt))
                self.assertEqual(image_words.removal_request(run, word_index(run, word)).prompt, expected)
        run = replace(self.original, request=replace(self.original.request, prompt="red"))
        self.assertEqual(self.compare(run)[-1][ROW["pair"]][1].request.prompt, "")

    def test_invalid_or_separator_selection_never_starts_a_run(self):
        for index in (None, -1, 999, 1, "red"):
            calls = self.manager.pipeline.unet.calls
            frames = list(image_words.compare_word((self.original, None), index))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][ROW["pair"]], gr.skip())
            self.assertEqual(self.manager.pipeline.unet.calls, calls)

    def test_stopped_original_or_missing_load_record_is_refused(self):
        for original in (None, replace(self.original, stopped=True, image=None),
                         replace(self.original, load_id=None)):
            frames = list(image_words.compare_word((original, None), 2))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][ROW["pair"]], gr.skip())
            self.assertFalse(self.manager.busy)

    def test_reload_of_the_same_model_is_refused_and_releases_reservation(self):
        self.manager.load_count += 1
        calls = self.manager.pipeline.unet.calls
        final = self.compare()[-1]
        self.assertIn("loaded model changed", final[ROW["status"]])
        self.assertEqual(final[ROW["pair"]], gr.skip())
        self.assertEqual(self.manager.pipeline.unet.calls, calls)
        self.assertFalse(self.manager.busy)
        self.assertFalse(self.manager.stop_image_run())

    def test_pipeline_without_seed_support_is_refused(self):
        class Unseeded(FakePipeline):
            def __call__(self, prompt=None, callback_on_step_end=None):
                raise AssertionError("must be refused before drawing")
        self.manager.pipeline = Unseeded()
        final = self.compare()[-1]
        self.assertIn("cannot accept a fixed seed", final[ROW["status"]])
        self.assertFalse(self.manager.busy)

    def test_busy_refusal_preserves_the_other_runs_cancellation_token(self):
        cancel = self.manager.start_image_run()
        try:
            final = self.compare()[-1]
            self.assertIn("busy", final[ROW["status"]])
            self.assertIs(self.manager._image_cancel, cancel)
            self.assertFalse(cancel.is_set())
        finally:
            self.manager.finish_image_run()

    def test_stop_keeps_partial_trajectory_and_the_original(self):
        self.manager.pipeline.watcher = lambda step: self.manager.stop_image_run()
        final = self.compare()[-1]
        original, stopped = final[ROW["pair"]]
        self.assertIs(original, self.original)
        self.assertTrue(stopped.stopped)
        self.assertIsNone(stopped.image)
        self.assertEqual(final[ROW["step"]]["maximum"], stopped.steps_done)
        self.assertIn("No finished image", final[ROW["right"]])
        self.assertFalse(self.manager.busy)

    def test_closing_comparison_cancels_its_worker(self):
        release, done = threading.Event(), threading.Event()
        self.manager.pipeline.watcher = lambda step: release.wait(2)
        with mock.patch.object(self.manager, "_release_device_cache", side_effect=done.set):
            frames = image_words.compare_word((self.original, None), word_index(self.original, "red"))
            next(frames)
            cancel = self.manager._image_cancel
            frames.close()
            self.assertTrue(cancel.is_set())
            release.set()
            self.assertTrue(done.wait(5))
        self.assertFalse(self.manager.busy)

    def test_failed_comparison_keeps_existing_results_and_releases_slot(self):
        previous = self.compare()[-1][ROW["pair"]]
        class Broken(FakePipeline):
            def __call__(self, **kwargs):
                raise RuntimeError("drawing failed")
        self.manager.pipeline = Broken()
        final = self.compare(pair=previous)[-1]
        self.assertIn("drawing failed", final[ROW["status"]])
        self.assertEqual(final[ROW["pair"]], gr.skip())
        self.assertIsNotNone(previous[1].image)
        self.assertFalse(self.manager.busy)

    def test_maps_disabled_still_allows_image_and_trajectory_comparison(self):
        original = self.manager.generate_image(replace(self.original.request, record_attention=False))
        final = self.compare(original)[-1]
        self.assertIsNotNone(final[ROW["pair"]][1].image)
        self.assertEqual(final[ROW["left_strip"]], [])
        self.assertEqual(final[ROW["right_strip"]], [])
        self.assertIn("Step 1 of 3", final[ROW["right"]])
        self.assertIn("not recorded", final[ROW["right"]])

    def test_both_trajectories_and_maps_follow_the_shared_step(self):
        pair = self.compare()[-1][ROW["pair"]]
        left, right, left_strip, right_strip = image_words.read_pair(pair, 2, 2, 2)
        for panel in (left, right):
            self.assertIn("Step 2 of 3", panel)
            self.assertIn("of the attention at step 2", panel)
            self.assertNotIn(' id="', panel)
        self.assertIn("red", left)
        self.assertIn("bicycle", right)
        self.assertEqual(len(left_strip), len(self.original.tokens))
        self.assertEqual(len(right_strip), len(pair[1].tokens))

    def test_heatmaps_and_charts_use_shared_scales(self):
        pair = self.compare()[-1][ROW["pair"]]
        pair[0].attention[:] = 0.1
        pair[1].attention[:] = 0.4
        with mock.patch.object(image_runtime, "heat_overlay", wraps=image_runtime.heat_overlay) as heat:
            with mock.patch.object(image_words.charts, "denoising_chart", wraps=image_words.charts.denoising_chart) as chart:
                image_words.read_pair(pair, 1, 1, 1)
        self.assertEqual(heat.call_count, 2)
        for call in heat.call_args_list:
            self.assertAlmostEqual(call.kwargs["ceiling"], 0.4)
        self.assertEqual(chart.call_args_list[0].kwargs, chart.call_args_list[1].kwargs)

    def test_selection_preview_and_panels_escape_prompt_html(self):
        run = replace(self.original, request=replace(self.original.request, prompt="<script>alert('red')</script>"))
        event = gr.SelectData(None, {"index": word_index(run, "red"), "value": "red"})
        index, preview, button = image_words.select_word((run, None), event)
        self.assertIsInstance(index, int)
        self.assertTrue(button["interactive"])
        self.assertNotIn("<script>", preview)
        self.assertIn("&lt;script&gt;", preview)
        panel = image_words.read_pair((run, None), 1, None, None)[0]
        self.assertNotIn("<script>", panel)

    def test_new_original_clears_previous_selection_and_comparison(self):
        setup = image_words.start_original(self.original)
        self.assertIsNone(setup[1])
        values = dict(zip(image_words.OUTPUT_NAMES, setup[3:]))
        self.assertEqual(values["pair"], (self.original, None))
        self.assertFalse(values["draw"]["interactive"])
        self.assertIsNone(values["left_token"])
        self.assertIsNone(values["right_token"])
        self.assertEqual(values["right_strip"], [])

    def test_gradio_wiring_and_postprocessing_preserve_both_runs(self):
        demo = app.build_app()
        listeners = {fn.fn: fn for fn in demo.fns.values() if fn.fn is not None}
        compare = listeners[image_words.compare_word]
        setup = listeners[image_words.start_original]
        draw = listeners[images_page.draw]
        self.assertEqual(compare.concurrency_id, draw.concurrency_id)
        self.assertEqual(len(compare.outputs), len(image_words.OUTPUT_NAMES))
        self.assertEqual(setup.inputs, [draw.outputs[images_page.IMAGE_OUTPUT_NAMES.index("run")]])
        frames = self.compare()

        async def postprocess():
            state = gr.blocks.SessionState(demo)
            await demo.postprocess_data(setup, image_words.start_original(self.original), state)
            for frame in frames:
                await demo.postprocess_data(compare, frame, state)
            return state[compare.outputs[ROW["pair"]]._id]

        original, removed = asyncio.run(postprocess())
        self.assertIsInstance(original, ImageRun)
        self.assertEqual(original.request.prompt, "a red bicycle")
        self.assertEqual(removed.request.prompt, "a bicycle")


if __name__ == "__main__":
    unittest.main()
