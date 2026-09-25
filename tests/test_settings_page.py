"""The Settings page: what it says about the machine, and the settings file the
interface opens with and writes back to."""

import json
import os
import unittest
from unittest import mock

import gradio as gr

from chatlab import app
from chatlab.ui import runtime
from chatlab import device_memory
from chatlab import settings
from chatlab.model_runtime import ModelManager

from models_support import OLMO
import settings_sandbox
from ui_support import listeners_named


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class HardwarePanelTests(unittest.TestCase):
    """What the Settings page says about the machine."""

    GB = 1024**3

    def setUp(self):
        self.manager = ModelManager()
        original = runtime.MANAGER
        runtime.MANAGER = self.manager
        self.addCleanup(lambda: setattr(runtime, "MANAGER", original))

    def card(self, **profile):
        return app.hardware_card(device_memory.DeviceProfile(**profile))

    def test_a_device_not_read_yet_shows_the_memory_and_says_to_wait(self):
        card = self.card(total=48 * self.GB, available=40 * self.GB)

        self.assertIn("not read yet", card)
        self.assertIn("48.0 GB in total", card)
        self.assertIn("40.0 GB", card)
        self.assertIn(app.HARDWARE_UNREAD, card)

    def test_a_metal_machine_reports_the_cap_and_where_it_comes_from(self):
        card = self.card(
            backend="mps",
            dtype="float16",
            total=24 * self.GB,
            available=20 * self.GB,
            ceiling=24 * self.GB,
            recommended=36 * self.GB,
            fraction=24 / 36,
        )

        self.assertIn("Apple Metal (MPS)", card)
        self.assertIn("loaded as float16", card)
        self.assertIn("8-bit and 4-bit", card)
        self.assertIn("24.0 GB, 0.67 of the 36.0 GB Metal recommends", card)
        self.assertIn("mps_memory_fraction", card)
        # The reserve the fit check keeps back is part of the same story.
        self.assertIn("4.0 GB kept beside the weights", card)

    def test_a_machine_without_metal_says_a_quantized_choice_is_ignored(self):
        card = self.card(
            backend="cpu", dtype="float32", total=16 * self.GB, available=8 * self.GB
        )

        self.assertIn("CPU", card)
        self.assertIn("loaded as float32", card)
        self.assertIn("needs Apple Metal", card)
        self.assertNotIn("Metal cap", card)

    def test_the_panel_names_the_model_in_memory(self):
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS), 4-bit weights"
        self.manager.precision = "4-bit"

        card = self.card(backend="mps", dtype="float16", total=48 * self.GB)

        self.assertIn(OLMO, card)
        self.assertIn("4-bit weights", card)

    def test_an_empty_runtime_points_at_the_models_page(self):
        card = self.card(backend="mps", dtype="float16", total=48 * self.GB)

        self.assertIn("none", card)
        self.assertIn("Models page", card)

    def test_a_machine_that_reports_no_memory_says_unknown_rather_than_zero(self):
        card = self.card(backend="cpu", dtype="float32")

        self.assertIn("unknown in total", card)
        self.assertNotIn("0.0 GB in total", card)


class SavedSettingsTests(unittest.TestCase):
    """The settings file the interface opens with and writes back to."""

    def setUp(self):
        self.path = settings.settings_path()
        self.addCleanup(self.forget)

    def forget(self):
        self.path.unlink(missing_ok=True)
        settings.load()

    def build_with(self, **values):
        """The interface as it comes up with ``values`` already saved."""

        settings.write(settings.sanitize(values), path=self.path)
        settings.load()
        self.demo = app.build_app()
        return self.demo

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    # In the order settings.CONVERSATION_SAMPLING names them, which is the
    # order the resets are built in and the order every handler here reads
    # them in.
    SAMPLING_LABELS = (
        "Temperature",
        "Top-p",
        "Top-k (0 disables)",
        "Skip top choice below (0 disables)",
        "Maximum new tokens",
    )

    def sampling_resets(self):
        """The ↺ beside each sampling slider, in the order they are built."""

        return [
            block
            for block in self.demo.blocks.values()
            if "sampling-reset" in (getattr(block, "elem_classes", None) or [])
        ]

    def chain_from(self, button):
        """A press and everything chained behind it, by handler name.

        Gradio records a chained step as the id of the step it waits for
        rather than as a block, so the chain is walked rather than looked up.
        reset_sampling is partial'd over the setting it restores and the
        ceiling the page was built with, so it answers to its name through
        that.
        """

        def name_of(fn):
            return getattr(fn.fn, "__name__", None) or getattr(
                getattr(fn.fn, "func", None), "__name__", None
            )

        chain = {}
        waiting = {button._id}
        while waiting:
            step = waiting.pop()
            for fn in self.demo.fns.values():
                reached = any(block_id == step for block_id, _event in fn.targets)
                if reached or getattr(fn, "trigger_after", None) == step:
                    if fn._id not in [held._id for held in chain.values()]:
                        chain[name_of(fn)] = fn
                        waiting.add(fn._id)
        return chain

    def test_every_control_starts_from_the_saved_file(self):
        self.build_with(
            model_id="org/other-model",
            system_prompt="Be brief.",
            assistant_prefill="Well,",
            keep_reasoning=True,
            temperature=0.25,
            top_p=0.5,
            top_k=7,
            max_new_tokens=64,
            seed=99,
            randomize_seed=False,
            analyze_prompt=False,
            color_scale="Surprise",
            prefill_token_limit=2048,
        )

        for label, value in [
            ("Hugging Face model ID", "org/other-model"),
            ("System prompt", "Be brief."),
            ("Assistant prefill (optional)", "Well,"),
            ("Send previous reasoning back to the model", True),
            ("Temperature", 0.25),
            ("Top-p", 0.5),
            ("Top-k (0 disables)", 7),
            ("Maximum new tokens", 64),
            ("Random seed", 99),
            ("New seed each response", False),
            ("Measure prompt tokens", False),
            ("Color tokens by", "Surprise"),
            ("Context limit (tokens)", 2048),
        ]:
            with self.subTest(label=label):
                self.assertEqual(self.labelled(label).value, value)

    def test_the_message_box_keys_start_from_the_saved_file(self):
        self.build_with(enter_sends=False)

        self.assertFalse(self.labelled("Enter sends the message").value)
        self.assertEqual(
            self.labelled("Message").placeholder,
            app.message_box_settings(enter_sends=False)["placeholder"],
        )

    def test_the_typing_predictions_start_from_the_saved_file(self):
        self.build_with(writing_suggestions=False)

        self.assertFalse(
            self.labelled("Let the system suggest text while typing").value
        )

    def test_the_response_length_cannot_exceed_the_context_limit(self):
        self.build_with(prefill_token_limit=2048)

        self.assertEqual(self.labelled("Maximum new tokens").maximum, 2048)

    def test_a_missing_file_leaves_every_control_at_its_default(self):
        self.path.unlink(missing_ok=True)
        settings.load()
        self.demo = app.build_app()

        self.assertEqual(
            self.labelled("Temperature").value, settings.DEFAULTS.temperature
        )
        self.assertEqual(
            self.labelled("Hugging Face model ID").value, settings.DEFAULT_MODEL_ID
        )

    def test_the_file_is_there_to_edit_after_one_launch(self):
        self.path.unlink(missing_ok=True)
        settings.load()

        app.build_app()

        self.assertTrue(self.path.is_file())

    def saving_listeners(self):
        """Every handler that writes the whole set, however it was reached."""

        return listeners_named(self.demo, "remember_settings") + listeners_named(
            self.demo, "remember_committed_seed"
        )

    @staticmethod
    def triggered_by(fn):
        """The blocks whose events reach ``fn``.

        A handler run from the end of a chain - Reset to defaults saves the
        settings after it has moved the sliders - is triggered by the step
        before it rather than by a block, and Gradio writes that down as a
        target with no block at all.
        """

        return [block_id for block_id, _event in fn.targets if block_id is not None]

    def test_changing_any_setting_saves_them_all(self):
        self.build_with()
        saved = self.saving_listeners()
        triggers = {
            self.demo.blocks[block_id]: event
            for fn in saved
            for block_id, event in fn.targets
            if block_id is not None
        }

        for label in [
            "System prompt",
            "Send previous reasoning back to the model",
            "Assistant prefill (optional)",
            "Temperature",
            "Top-p",
            "Top-k (0 disables)",
            "Maximum new tokens",
            "Random seed",
            "New seed each response",
            "Measure prompt tokens",
            "Color tokens by",
            "Thinking mode",
            "Enter sends the message",
            "Color theme",
            "Hugging Face model ID",
        ]:
            with self.subTest(label=label):
                self.assertIn(self.labelled(label), triggers)
        # Each one publishes the whole set, in the order the names are in.
        for fn in saved:
            self.assertEqual(len(fn.inputs), len(app.PERSISTED_SETTING_NAMES))

    def test_the_seed_is_saved_when_it_is_committed_and_not_when_it_is_written(self):
        # A finished response leaves the seed that produced it in the box, and
        # saving that would overwrite the seed the reader chose.
        self.build_with()
        events = {}
        for fn in self.saving_listeners():
            for block_id, event in fn.targets:
                if block_id is None:
                    continue
                events.setdefault(self.demo.blocks[block_id], set()).add(event)

        self.assertEqual(events[self.labelled("Random seed")], {"blur", "submit"})
        # The five sampling controls are saved on input rather than change:
        # switching conversations sets them, and a save from that would put
        # the sampling of the conversation being looked at into the file
        # every unpinned conversation answers with.
        for label in (
            "Temperature",
            "Top-p",
            "Top-k (0 disables)",
            "Skip top choice below (0 disables)",
            "Maximum new tokens",
        ):
            self.assertEqual(events[self.labelled(label)], {"input"}, label)
        self.assertEqual(events[self.labelled("Measure prompt tokens")], {"change"})
        # And only the seed box's own events are allowed to write it down.
        for fn in listeners_named(self.demo, "remember_committed_seed"):
            self.assertEqual(
                {self.demo.blocks[block_id] for block_id, _ in fn.targets},
                {self.labelled("Random seed")},
            )

    def test_the_hugging_face_token_is_not_among_the_settings_saved(self):
        self.build_with()
        token_box = self.labelled("Hugging Face token (optional)")

        for fn in self.saving_listeners():
            self.assertNotIn(token_box, fn.inputs)
            self.assertNotIn(
                token_box,
                [self.demo.blocks[i] for i in self.triggered_by(fn)],
            )

    def test_the_settings_page_says_where_the_file_is(self):
        self.build_with()
        page = next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "elem_id", None) == "settings-page"
        )
        notes = [
            block.value
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Markdown)
            and getattr(block, "value", None)
            and str(self.path) in str(block.value)
        ]

        self.assertTrue(notes, f"{self.path} is not named on {page.elem_id}")
        self.assertIn("mps_memory_fraction", notes[0])

    def test_saving_a_setting_writes_the_file(self):
        self.build_with()
        values = dict(zip(app.PERSISTED_SETTING_NAMES, [None] * len(app.PERSISTED_SETTING_NAMES)))
        values.update(settings.current().to_mapping())
        values["temperature"] = 0.1
        app.remember_settings(
            *(values[name] for name in app.PERSISTED_SETTING_NAMES)
        )

        self.assertEqual(settings.current().temperature, 0.1)
        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)

    def test_a_pinned_model_is_not_saved_when_something_else_changes(self):
        """``OLMO_MODEL_ID`` names a model for one run, not for every run."""

        with mock.patch.dict(
            os.environ, {"OLMO_MODEL_ID": "org/pinned"}, clear=False
        ):
            self.build_with(model_id="org/saved-model")
            box = self.labelled("Hugging Face model ID")
            self.assertEqual(box.value, "org/pinned")

            values = settings.current().to_mapping() | {
                "enter_sends": settings.current().enter_sends,
                "model_id": box.value,
                "temperature": 0.1,
            }
            app.remember_settings(
                *(values[name] for name in app.PERSISTED_SETTING_NAMES)
            )

        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)
        self.assertEqual(settings.current().model_id, "org/saved-model")
        self.assertEqual(json.loads(self.path.read_text())["model_id"], "org/saved-model")

    def test_a_model_typed_over_a_pinned_one_is_saved(self):
        with mock.patch.dict(
            os.environ, {"OLMO_MODEL_ID": "org/pinned"}, clear=False
        ):
            self.build_with(model_id="org/saved-model")
            values = settings.current().to_mapping() | {
                "enter_sends": settings.current().enter_sends,
                "model_id": "org/typed",
            }
            app.remember_settings(
                *(values[name] for name in app.PERSISTED_SETTING_NAMES)
            )

        self.assertEqual(settings.current().model_id, "org/typed")

    def test_a_generated_seed_is_not_saved_when_something_else_changes(self):
        """A response leaves its own seed in the box; that is not a choice."""

        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {
            "seed": 1234567,  # what a finished response put in the box
            "temperature": 0.1,
        }
        app.remember_settings(*(values[name] for name in app.PERSISTED_SETTING_NAMES))

        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)
        self.assertEqual(settings.current().seed, 99)
        self.assertEqual(json.loads(self.path.read_text())["seed"], 99)

    def test_committing_the_seed_box_saves_what_is_in_it(self):
        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {"seed": 7}
        app.remember_committed_seed(
            *(values[name] for name in app.PERSISTED_SETTING_NAMES)
        )

        self.assertEqual(settings.current().seed, 7)

    def test_a_locked_seed_is_saved_by_any_control(self):
        # With randomization off the box is the reader's alone, and turning it
        # off is how one keeps the seed a response has just used.
        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {
            "seed": 1234567,
            "randomize_seed": False,
        }
        app.remember_settings(*(values[name] for name in app.PERSISTED_SETTING_NAMES))

        self.assertEqual(settings.current().seed, 1234567)

    def test_a_reset_puts_its_own_sampling_control_back(self):
        # Worked out when the button is pressed, not read off the value the
        # slider was built with: the sliders come up holding the saved
        # settings and the file follows every move of them, so a reset to
        # what they were built with would restore the number already there.
        self.build_with(
            temperature=1.6, top_p=1.0, top_k=200, skip_top_below=0.75, max_new_tokens=64
        )

        for name in settings.CONVERSATION_SAMPLING:
            with self.subTest(name=name):
                update = app.reset_sampling(name, settings.DEFAULTS.prefill_token_limit)

                self.assertEqual(update["value"], getattr(settings.DEFAULTS, name))

    def test_a_reset_writes_nothing_but_the_control_it_belongs_to(self):
        # Five buttons, one control each: pressing the one beside temperature
        # leaves an experiment's top-p and response length where they are.
        # Which setting a handler restores is the first of the two arguments
        # it is partial'd over, and it has to be the one its slider holds.
        self.build_with()
        sliders = dict(
            zip(settings.CONVERSATION_SAMPLING, map(self.labelled, self.SAMPLING_LABELS))
        )

        restored = {}
        for button in self.sampling_resets():
            reset = self.chain_from(button)["reset_sampling"]
            name, _built_limit = reset.fn.args
            restored[name] = reset.outputs
            with self.subTest(name=name):
                self.assertEqual(reset.outputs, [sliders[name]])
                # And the words a screen reader reads out say which slider
                # this one belongs to, which a row of identical marks does
                # not.
                self.assertNotEqual(button.value, "Reset")

        self.assertEqual(sorted(restored), sorted(settings.CONVERSATION_SAMPLING))

    def test_the_length_reset_to_follows_the_context_limit(self):
        # Read now rather than when the page was built, so a limit lowered
        # since lowers what the button restores.
        self.build_with(prefill_token_limit=8192)
        settings.update(prefill_token_limit=512)

        update = app.reset_sampling("max_new_tokens", 8192)

        self.assertEqual(update["value"], 512)

    def test_the_length_reset_to_stays_under_the_limit_the_page_was_built_with(self):
        # A slider refuses a value above the maximum it was built with,
        # whatever its maximum has been set to since, so a page that came up
        # under a lower limit keeps that ceiling until it is loaded again.
        self.build_with(prefill_token_limit=512)
        settings.update(prefill_token_limit=8192)

        update = app.reset_sampling("max_new_tokens", 512)

        self.assertEqual(update["value"], 512)

    def test_a_reset_is_stored_the_way_a_slider_moved_by_hand_is(self):
        self.build_with()
        sliders = [self.labelled(label) for label in self.SAMPLING_LABELS]

        for button in self.sampling_resets():
            with self.subTest(button=button.value):
                by_name = self.chain_from(button)

                # The control first, then the conversation and the file from
                # what the five hold, then the summary.
                self.assertEqual(by_name["remember_branch_sampling"].inputs[-5:], sliders)
                self.assertEqual(
                    by_name["remember_settings"].inputs,
                    listeners_named(self.demo, "remember_settings")[0].inputs,
                )
                for slider in sliders:
                    self.assertIn(slider, by_name["remember_settings"].inputs)
                self.assertEqual(by_name["update_sampling_label"].inputs, sliders)
                # The seed has no reset of its own: one being held to
                # reproduce a reply is not a setting to be put back.
                self.assertNotIn(
                    self.labelled("Random seed"), by_name["reset_sampling"].outputs
                )

    def test_the_seed_has_no_reset_of_its_own(self):
        self.build_with()

        self.assertEqual(len(self.sampling_resets()), len(settings.CONVERSATION_SAMPLING))

    def test_gradios_own_reset_button_is_off_on_the_sampling_sliders(self):
        # It restores the value its slider was built with, which here is the
        # saved setting the slider is already showing.
        self.build_with()

        for label in (
            "Temperature",
            "Top-p",
            "Top-k (0 disables)",
            "Skip top choice below (0 disables)",
            "Maximum new tokens",
        ):
            with self.subTest(label=label):
                self.assertFalse(self.labelled(label).show_reset_button)

    def test_lowering_the_context_limit_pulls_the_response_length_under_it(self):
        self.build_with(prefill_token_limit=8192, max_new_tokens=4096)

        limit, length, _forks = app.remember_prefill_limit(1024, 4096)

        self.assertEqual(limit["value"], 1024)
        self.assertEqual(length["maximum"], 1024)
        self.assertEqual(length["value"], 1024)
        self.assertEqual(settings.current().max_new_tokens, 1024)
        self.assertEqual(json.loads(self.path.read_text())["prefill_token_limit"], 1024)

    def test_a_page_load_puts_the_saved_settings_back_into_the_controls(self):
        self.build_with(temperature=0.4, max_new_tokens=64, prefill_token_limit=2048)
        (restore,) = listeners_named(self.demo, "restore_settings")

        self.assertEqual(restore.targets, [(self.demo._id, "load")])
        self.assertEqual(
            restore.outputs,
            [
                *(
                    self.labelled(label)
                    for label in [
                        "System prompt",
                        "Send previous reasoning back to the model",
                        "Assistant prefill (optional)",
                        "Temperature",
                        "Top-p",
                        "Top-k (0 disables)",
                        "Skip top choice below (0 disables)",
                        "Maximum new tokens",
                        "Random seed",
                        "New seed each response",
                        "Measure prompt tokens",
                        "Color tokens by",
                        "Thinking mode",
                        "Enter sends the message",
                        "Let the system suggest text while typing",
                        "Color theme",
                        "Light or dark",
                        "Hugging Face model ID",
                        "Weight precision",
                    ]
                ),
                self.labelled("Context limit (tokens)"),
            ],
        )
        updates = app.restore_settings()
        self.assertEqual(len(updates), len(app.PERSISTED_SETTING_NAMES) + 1)
        published = dict(zip(app.PERSISTED_SETTING_NAMES, updates))
        self.assertEqual(published["temperature"]["value"], 0.4)
        self.assertEqual(published["max_new_tokens"]["value"], 64)
        # The response-length ceiling comes back with it.
        self.assertEqual(published["max_new_tokens"]["maximum"], 2048)
        self.assertEqual(updates[-1]["value"], 2048)

    def test_a_page_load_reads_the_file_again_so_a_hand_edit_takes_effect(self):
        self.build_with(temperature=0.4)
        self.path.write_text(
            json.dumps(settings.current().to_mapping() | {"temperature": 1.1}),
            encoding="utf-8",
        )

        published = dict(zip(app.PERSISTED_SETTING_NAMES, app.restore_settings()))

        self.assertEqual(published["temperature"]["value"], 1.1)
        self.assertEqual(settings.current().temperature, 1.1)

    def test_a_context_limit_outside_its_range_is_pulled_back_into_it(self):
        self.build_with()

        limit, _length, _forks = app.remember_prefill_limit(2, 512)

        self.assertEqual(limit["value"], settings.PREFILL_TOKEN_LIMIT_RANGE[0])


if __name__ == "__main__":
    unittest.main()
