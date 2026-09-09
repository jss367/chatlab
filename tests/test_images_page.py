"""The Images page: what it publishes while drawing, and what it reads afterwards."""

import unittest

import app
import charts
import image_runtime
import settings
import settings_sandbox
from fake_pipeline import FakePipeline
from image_runtime import ImageRequest
from model_runtime import IMAGE_KIND, TEXT_KIND, ModelManager
from token_metrics import PROMPT_ATTENTION_SCALE
from ui import images_page, runtime


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


# The rows every image handler publishes, by name; see IMAGE_OUTPUT_NAMES.
ROW = {name: index for index, name in enumerate(images_page.IMAGE_OUTPUT_NAMES)}


class ImagePageTestCase(unittest.TestCase):
    """A manager with an image pipeline in memory, and nothing else."""

    def setUp(self):
        self.manager = ModelManager()
        original = runtime.MANAGER
        runtime.MANAGER = self.manager
        self.addCleanup(lambda: setattr(runtime, "MANAGER", original))
        images_page._STOP.clear()

    def load_pipeline(self):
        self.manager.pipeline = FakePipeline()
        self.manager.kind = IMAGE_KIND
        self.manager.model_id = "org/pipe"
        self.manager.device_name = "CPU"

    def load_text_model(self):
        self.manager.model = object()
        self.manager.tokenizer = object()
        self.manager.kind = TEXT_KIND
        self.manager.model_id = "allenai/Olmo-3-7B-Think"
        self.manager.device_name = "CPU"

    def frames(self, **overrides):
        arguments = {
            "prompt": "a red bicycle",
            "negative_prompt": "blurry",
            "steps": 3,
            "guidance": 7.5,
            "size": 32,
            "seed": 7,
            "randomize_seed": False,
            "record_attention": True,
        }
        arguments.update(overrides)
        return list(images_page.draw(*[arguments[name] for name in (
            "prompt",
            "negative_prompt",
            "steps",
            "guidance",
            "size",
            "seed",
            "randomize_seed",
            "record_attention",
        )]))

    def run_of(self, **overrides):
        frames = self.frames(**overrides)
        return frames[-1][ROW["run"]], frames


class RefusalTests(ImagePageTestCase):
    """What the page says when it cannot draw, and what it leaves alone."""

    def test_every_handler_publishes_the_same_number_of_rows(self):
        for frame in self.frames(prompt="   "):
            self.assertEqual(len(frame), len(images_page.IMAGE_OUTPUT_NAMES))

    def test_an_empty_prompt_is_refused_without_touching_the_model(self):
        self.load_pipeline()

        (frame,) = self.frames(prompt="   ")

        self.assertEqual(frame[ROW["status"]], images_page.EMPTY_PROMPT)
        self.assertFalse(frame[ROW["stop"]]["visible"])
        self.assertTrue(frame[ROW["draw"]]["visible"])
        self.assertEqual(self.manager.pipeline.unet.calls, 0)

    def test_no_model_at_all_says_where_to_get_one(self):
        (frame,) = self.frames()

        self.assertEqual(frame[ROW["status"]], images_page.NO_IMAGE_MODEL)
        self.assertIn("**Models** page", frame[ROW["status"]])

    def test_a_text_model_in_memory_is_named_as_the_wrong_kind(self):
        # A page that just said "no model loaded" would send a reader off to
        # load a second model on top of the one filling the machine.
        self.load_text_model()

        (frame,) = self.frames()

        self.assertEqual(frame[ROW["status"]], images_page.TEXT_MODEL_LOADED)
        self.assertIn("text model", frame[ROW["status"]])

    def test_a_refusal_leaves_the_last_run_on_screen(self):
        # A picture that could not be drawn is no reason to throw away the
        # readings of the one before it.
        import gradio as gr

        (frame,) = self.frames()

        skip = type(gr.skip())
        for name in ("run", "step", "trajectory", "tiles", "chart", "strip", "overlay"):
            with self.subTest(row=name):
                self.assertIsInstance(frame[ROW[name]], skip)


class DrawTests(ImagePageTestCase):
    """A whole draw, from the first frame to the readout."""

    def test_the_first_frame_swaps_the_buttons_and_clears_the_readout(self):
        self.load_pipeline()

        frames = self.frames()

        first = frames[0]
        self.assertIn("seed 7", first[ROW["status"]])
        self.assertFalse(first[ROW["draw"]]["visible"])
        self.assertTrue(first[ROW["stop"]]["visible"])
        self.assertEqual(first[ROW["seed"]], 7)
        self.assertEqual(first[ROW["trajectory"]], images_page.NO_TRAJECTORY)
        self.assertEqual(first[ROW["tiles"]], charts.EMPTY_IMAGE_TILES)
        self.assertEqual(first[ROW["chart"]], charts.EMPTY_DENOISING_CHART)
        self.assertEqual(first[ROW["strip"]], [])
        self.assertIsNone(first[ROW["token"]])

    def test_the_last_frame_carries_the_finished_run_and_its_readout(self):
        self.load_pipeline()

        run, frames = self.run_of()
        last = frames[-1]

        self.assertEqual(run.steps_done, 3)
        self.assertEqual(run.model_id, "org/pipe")
        self.assertEqual(run.load_id, "org/pipe#0")
        self.assertTrue(last[ROW["draw"]]["visible"])
        self.assertFalse(last[ROW["stop"]]["visible"])
        self.assertIsNotNone(last[ROW["image"]])
        self.assertIn("3 steps", last[ROW["status"]])
        self.assertIn("seed 7", last[ROW["status"]])
        # The step slider is sized to the run and sits on its last step.
        self.assertEqual(last[ROW["step"]]["maximum"], 3)
        self.assertEqual(last[ROW["step"]]["value"], 3)
        self.assertTrue(last[ROW["step"]]["interactive"])
        self.assertIn("Step 3 of 3", last[ROW["trajectory"]])
        self.assertIn("denoising steps", last[ROW["tiles"]])
        self.assertIn("guidance pull", last[ROW["chart"]])
        # No token is picked yet, so the map is the invitation to pick one.
        self.assertEqual(last[ROW["overlay"]], images_page.NO_ATTENTION)
        self.assertIsNone(last[ROW["token"]])

    def test_the_prompt_and_the_size_reach_the_pipeline(self):
        self.load_pipeline()

        run, _ = self.run_of(size=64, steps=2, guidance=3.0)

        self.assertEqual(run.request.width, 64)
        self.assertEqual(run.request.height, 64)
        self.assertEqual(run.request.steps, 2)
        self.assertEqual(run.request.guidance_scale, 3.0)
        self.assertEqual(run.request.negative_prompt, "blurry")
        self.assertEqual(run.image.size, (64, 64))

    def test_the_prompt_is_trimmed_before_it_is_drawn(self):
        self.load_pipeline()

        run, _ = self.run_of(prompt="  a red bicycle  ")

        self.assertEqual(run.request.prompt, "a red bicycle")

    def test_a_randomized_seed_is_published_so_a_picture_can_be_repeated(self):
        self.load_pipeline()

        run, frames = self.run_of(randomize_seed=True, seed=7)
        published = frames[0][ROW["seed"]]

        self.assertEqual(published, run.request.seed)
        self.assertLessEqual(published, app.SEED_LIMIT)
        self.assertIn(f"seed {published}", frames[-1][ROW["status"]])

    def test_a_failed_run_reports_it_and_puts_the_buttons_back(self):
        self.load_pipeline()
        self.manager.pipeline = _Exploding()

        last = self.frames()[-1]

        self.assertIn("Could not draw the picture", last[ROW["status"]])
        self.assertIn("the device fell over", last[ROW["status"]])
        self.assertTrue(last[ROW["draw"]]["visible"])
        self.assertFalse(last[ROW["stop"]]["visible"])
        self.assertFalse(self.manager.busy)

    def test_stopping_keeps_the_trajectory_and_says_there_is_no_picture(self):
        # Pressing Stop mid-run, from inside the loop: the run checks the
        # event between steps, so the step under way finishes and the ones
        # already recorded are kept.
        self.load_pipeline()
        self.manager.pipeline = FakePipeline(
            watcher=lambda step: images_page.stop_drawing()
        )

        last = self.frames(steps=6)[-1]
        run = last[ROW["run"]]

        self.assertTrue(run.stopped)
        self.assertEqual(run.steps_done, 1)
        self.assertIn("Stopped after", last[ROW["status"]])
        self.assertIsNone(last[ROW["image"]])
        self.assertIn("Step 1 of 1", last[ROW["trajectory"]])

    def test_a_stop_left_over_from_before_does_not_kill_the_next_picture(self):
        # Nothing was running when Stop was pressed, so the event is still
        # set; draw() clears it as it starts rather than dying at step one.
        self.load_pipeline()
        images_page.stop_drawing()

        run, _ = self.run_of()

        self.assertFalse(run.stopped)
        self.assertEqual(run.steps_done, 3)

    def test_the_stop_button_only_says_so(self):
        # It is not a cancel: the pipeline runs on its own thread and would
        # keep running with the generator gone, taking the recorded steps
        # with it.
        status = images_page.stop_drawing()

        self.assertTrue(images_page._STOP.is_set())
        self.assertIn("Stopping", status)


class _Exploding(FakePipeline):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("the device fell over")


class ReadoutTests(ImagePageTestCase):
    """Reading a finished run: the frame, the shading, and the maps."""

    def setUp(self):
        super().setUp()
        self.load_pipeline()
        self.run = self.manager.generate_image(
            ImageRequest(prompt="a red bicycle", steps=4, seed=1, width=32, height=32)
        )

    def test_a_frame_names_its_step_and_its_own_readings(self):
        html = images_page.trajectory_frame(self.run, 2)

        self.assertIn("Step 2 of 4", html)
        self.assertIn("guidance pull", html)
        self.assertIn("moved", html)
        self.assertIn(self.run.readings[1].preview, html)
        # A reader comparing the frame against the finished picture is told
        # why the detail disagrees.
        self.assertIn("not a full decode", html)

    def test_the_first_frame_has_no_movement_to_report(self):
        html = images_page.trajectory_frame(self.run, 1)

        self.assertIn("Step 1 of 4", html)
        self.assertNotIn("moved", html)

    def test_a_step_out_of_range_is_pulled_into_it(self):
        self.assertIn("Step 4 of 4", images_page.trajectory_frame(self.run, 99))
        self.assertIn("Step 1 of 4", images_page.trajectory_frame(self.run, 0))

    def test_a_run_with_no_readings_has_no_trajectory(self):
        self.assertEqual(images_page.trajectory_frame(None, 1), images_page.NO_TRAJECTORY)

    def test_the_strip_shades_every_prompt_token_against_the_strongest(self):
        strip = images_page.prompt_strip_value(self.run)

        self.assertEqual(
            [text for text, _label in strip],
            [token["text"] for token in self.run.tokens],
        )
        labels = set(PROMPT_ATTENTION_SCALE.labels)
        for _text, label in strip:
            self.assertIn(label, labels)
        # The strongest token is in the top bucket by construction.
        self.assertIn(PROMPT_ATTENTION_SCALE.labels[-1], [label for _t, label in strip])

    def test_the_padding_row_is_not_in_the_strip_but_is_reported_in_words(self):
        strip = images_page.prompt_strip_value(self.run)
        note = images_page.attention_note(self.run)

        self.assertEqual(len(strip), len(self.run.tokens))
        self.assertNotIn(image_runtime.PADDING_ROW, [text for text, _ in strip])
        self.assertIn("padding past the prompt took", note)
        self.assertIn("over every step", note)

    def test_the_note_follows_the_step_being_looked_at(self):
        self.assertIn("at step 2", images_page.attention_note(self.run, 2))

    def test_a_run_without_maps_says_why_instead_of_shading_nothing(self):
        run = self.manager.generate_image(
            ImageRequest(
                prompt="a red bicycle",
                steps=2,
                seed=1,
                width=32,
                height=32,
                record_attention=False,
            )
        )

        self.assertEqual(images_page.prompt_strip_value(run), [])
        self.assertIn("not recorded", images_page.attention_note(run))
        self.assertEqual(
            images_page.attention_overlay(run, 0), images_page.NO_ATTENTION
        )

    def test_a_token_map_is_laid_over_the_picture_it_drew(self):
        html = images_page.attention_overlay(self.run, 2)

        self.assertIn("attention-stack", html)
        # Two images: the picture, then the map over it.
        self.assertEqual(html.count("<img"), 2)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn(repr(self.run.tokens[2]["text"]).replace("'", "&#x27;"), html)
        self.assertIn("of the attention over every step", html)

    def test_the_padding_row_has_no_map_to_click_through_to(self):
        # It is not a token anyone wrote, so it is not in the strip and
        # nothing addresses it; the note reports its share in words instead.
        self.assertIn(
            "attention-stack",
            images_page.attention_overlay(self.run, len(self.run.tokens) - 1),
        )
        self.assertEqual(
            images_page.attention_overlay(self.run, len(self.run.tokens)),
            images_page.NO_ATTENTION,
        )

    def test_no_token_picked_is_an_invitation_rather_than_a_blank(self):
        self.assertEqual(
            images_page.attention_overlay(self.run, None), images_page.NO_ATTENTION
        )

    def test_a_stopped_run_has_no_picture_to_lay_a_map_over(self):
        import threading

        stop = threading.Event()
        run = self.manager.generate_image(
            ImageRequest(prompt="a red bicycle", steps=6, seed=1, width=32, height=32),
            cancel=stop,
            on_step=lambda reading: stop.set(),
        )

        self.assertIn("stopped before it finished", images_page.attention_overlay(run, 1))

    def test_the_slider_is_left_alone_while_a_picture_is_being_drawn(self):
        # The streaming frames own it then: dragging it would show one frame
        # of the previous run before the next frame overwrote it.
        drawing = images_page._step_slider(self.run.readings, drawing=True)
        finished = images_page._step_slider(self.run.readings)

        self.assertFalse(drawing["interactive"])
        self.assertTrue(finished["interactive"])
        self.assertEqual(finished["maximum"], 4)
        # One step is nothing to scrub through either way.
        self.assertFalse(images_page._step_slider(self.run.readings[:1])["interactive"])

    def test_the_step_slider_moves_the_frame_the_shading_and_the_map_together(self):
        # Attention moves between steps as much as the picture does, so a
        # strip left on the run's average beside a moved frame would lie.
        frame, strip, note, overlay = images_page.select_step(self.run, 2, 1)

        self.assertIn("Step 2 of 4", frame)
        self.assertEqual(len(strip), len(self.run.tokens))
        self.assertIn("at step 2", note)
        self.assertIn("of the attention at step 2", overlay)

    def test_moving_the_step_with_no_token_picked_leaves_the_map_alone(self):
        _frame, _strip, _note, overlay = images_page.select_step(self.run, 2, None)

        self.assertEqual(overlay, images_page.NO_ATTENTION)

    def test_the_step_slider_does_nothing_before_a_run(self):
        import gradio as gr

        skipped = images_page.select_step(None, 1, None)

        self.assertEqual(len(skipped), 4)
        for value in skipped:
            self.assertIsInstance(value, type(gr.skip()))

    def test_clicking_a_token_draws_its_map_at_the_step_on_screen(self):
        overlay = images_page.select_token(self.run, 1, 3)

        self.assertIn("of the attention at step 3", overlay)
        self.assertIn("attention-stack", overlay)

    def test_the_clicked_token_is_remembered_by_its_index(self):
        # A strip's select event carries its index as a list as readily as a
        # number, and int() on the list raises: that is how a click ends up
        # discarded and no map is ever drawn.
        import gradio as gr

        listed = gr.SelectData(None, {"index": [4, 5], "value": "red"})
        plain = gr.SelectData(None, {"index": 4, "value": "red"})
        empty = gr.SelectData(None, {"index": None, "value": "red"})

        self.assertEqual(images_page.remember_token(listed), 4)
        self.assertEqual(images_page.remember_token(plain), 4)
        self.assertIsNone(images_page.remember_token(empty))


class SeedTests(unittest.TestCase):
    """The seed a picture is drawn with, and the one that gets saved."""

    def test_a_locked_seed_is_used_as_it_stands(self):
        self.assertEqual(images_page.resolve_seed(42, False), 42)

    def test_a_seed_out_of_range_is_pulled_into_it(self):
        # The number box constrains this, but the API and a float the box
        # rounded can still arrive.
        self.assertEqual(images_page.resolve_seed(-1, False), 0)
        self.assertEqual(images_page.resolve_seed(float("inf"), False), 0)
        self.assertEqual(images_page.resolve_seed(None, False), 0)
        self.assertEqual(images_page.resolve_seed(12.0, False), 12)

    def test_a_randomized_seed_is_inside_the_range_numpy_and_torch_accept(self):
        for _ in range(20):
            seed = images_page.resolve_seed(0, True)
            self.assertGreaterEqual(seed, 0)
            self.assertLessEqual(seed, app.SEED_LIMIT)


class ImageSettingsTests(unittest.TestCase):
    """What the Images page saves between sessions, and what it will not."""

    def setUp(self):
        self.path = settings_sandbox.start()
        self.addCleanup(settings_sandbox.stop)

    def test_the_drawing_controls_are_saved_as_they_change(self):
        images_page.remember_image_settings("blurry, watermark", 40, 9.0, 768, 5, False)
        saved = settings.load()

        self.assertEqual(saved.image_negative_prompt, "blurry, watermark")
        self.assertEqual(saved.image_steps, 40)
        self.assertEqual(saved.image_guidance, 9.0)
        self.assertEqual(saved.image_size, 768)
        self.assertEqual(saved.image_seed, 5)
        self.assertFalse(saved.image_randomize_seed)

    def test_a_seed_a_finished_picture_left_in_the_box_is_not_saved(self):
        # The app writes that number itself; saving it would overwrite the
        # seed the reader chose. The Chat page's seed follows the same rule.
        images_page.remember_image_settings("", 30, 7.5, 512, 11, False)
        images_page.remember_image_settings("", 30, 7.5, 512, 999_999, True)

        self.assertEqual(settings.load().image_seed, 11)

    def test_committing_the_seed_box_saves_what_it_holds(self):
        images_page.remember_committed_image_seed("", 30, 7.5, 512, 4321, True)

        self.assertEqual(settings.load().image_seed, 4321)

    def test_the_prompt_itself_is_never_saved(self):
        # It is the question being asked, not a setting, and the file is
        # meant to be shared between machines.
        images_page.remember_image_settings("blurry", 30, 7.5, 512, 1, False)

        self.assertNotIn("image_prompt", settings.load().to_mapping())

    def test_an_off_list_size_in_the_file_is_pulled_to_the_nearest_offered_one(self):
        self.assertEqual(settings.sanitize({"image_size": 500}).image_size, 512)
        self.assertEqual(settings.sanitize({"image_size": 99_999}).image_size, 1024)
        self.assertEqual(
            settings.sanitize({"image_size": "big"}).image_size,
            settings.DEFAULTS.image_size,
        )

    def test_the_step_and_guidance_ranges_are_the_ones_the_sliders_offer(self):
        low, high = settings.IMAGE_STEPS_RANGE
        self.assertEqual(settings.sanitize({"image_steps": 0}).image_steps, low)
        self.assertEqual(settings.sanitize({"image_steps": 9_999}).image_steps, high)
        self.assertEqual(settings.sanitize({"image_guidance": -3}).image_guidance, 0.0)
        self.assertEqual(settings.sanitize({"image_guidance": 99}).image_guidance, 20.0)


if __name__ == "__main__":
    unittest.main()
