"""settings.json as earlier versions wrote it.

Each file under tests/fixtures/saved/settings was written by the commit it is
named after through that commit's ``settings.update`` (see the generator) and
is never regenerated. Each is loaded the way the app loads it at startup,
``settings.load``, and every value checked; then a control change is saved
with today's writer and the file read back.
"""

import json
import os
import unittest

from chatlab import settings

from saved_formats_support import SAVED, sandbox

SETTINGS = SAVED / "settings"

# What the reader chose in every era, for the fields that era had.
CHOSEN = {
    "model_id": "Qwen/Qwen3-0.6B",
    "system_prompt": "Be brief.\nUse British spelling.",
    "assistant_prefill": "Answer:",
    "keep_reasoning": True,
    "enter_sends": False,
    "temperature": 0.35,
    "top_p": 0.85,
    "top_k": 12,
    "max_new_tokens": 600,
    "seed": 2024,
    "randomize_seed": False,
    "analyze_prompt": False,
    "color_scale": "Surprise",
    "prefill_token_limit": 4096,
    "mps_memory_fraction": 1.5,
}
THEMES_ERA = {
    "thinking_mode": "off",
    "writing_suggestions": False,
    "theme": "ember",
    "appearance": "dark",
    "weight_precision": "4-bit",
    "image_negative_prompt": "blurry",
    "image_steps": 20,
    "image_guidance": 5.0,
    "image_size": 768,
    "image_seed": 7,
    "image_randomize_seed": False,
    "image_record_attention": False,
}


class SavedSettingsCase(unittest.TestCase):
    fixture = ""

    def setUp(self):
        self.box = sandbox(self)
        self.path = self.box.copy(SETTINGS / self.fixture, "settings.json")
        os.environ[settings.SETTINGS_PATH_ENV] = str(self.path)
        self.loaded = settings.load()

    def assert_values(self, expected):
        for name, value in expected.items():
            self.assertEqual(getattr(self.loaded, name), value, name)

    def assert_defaults_for_the_rest(self, chosen):
        for name in settings.Settings.__dataclass_fields__:
            if name not in chosen:
                self.assertEqual(getattr(self.loaded, name), getattr(settings.DEFAULTS, name), name)

    def assert_resave_keeps_reading(self):
        """A control moved and saved by today's writer keeps every other value."""

        before = self.loaded
        _, unknown_before = settings.read(self.path)
        after = settings.update(temperature=1.1)
        self.assertEqual(after.temperature, 1.1)
        reread, unknown = settings.read(self.path)
        self.assertEqual(reread, after)
        self.assertEqual(unknown, unknown_before)
        for name in settings.Settings.__dataclass_fields__:
            if name != "temperature":
                self.assertEqual(getattr(reread, name), getattr(before, name), name)
        return json.loads(self.path.read_text(encoding="utf-8"))


class FirstEraTests(SavedSettingsCase):
    """3db3f4e (#31): the first settings file."""

    fixture = "01-first-3db3f4e.json"

    def test_values(self):
        self.assert_values(CHOSEN)
        self.assert_defaults_for_the_rest(CHOSEN)
        self.assertEqual(settings.read(self.path)[1], {})

    def test_conversation_sampling(self):
        self.assertEqual(settings.sampling_values(None), {
            "temperature": 0.35, "top_p": 0.85, "top_k": 12, "skip_top_below": 0.0, "max_new_tokens": 600,
        })
        self.assertEqual(settings.model_id_at_startup(), "Qwen/Qwen3-0.6B")

    def test_resave(self):
        written = self.assert_resave_keeps_reading()
        self.assertEqual(written["theme"], settings.DEFAULTS.theme)


class ThemesEraTests(SavedSettingsCase):
    """88ec738 (#105): thinking mode, images, extensions, theme and appearance."""

    fixture = "02-themes-88ec738.json"

    def test_values(self):
        self.assert_values(CHOSEN | THEMES_ERA)
        self.assertEqual(self.loaded.enabled_extensions, ("maze_experiments",))
        self.assert_defaults_for_the_rest(CHOSEN | THEMES_ERA | {"enabled_extensions": None})
        self.assertEqual(settings.read(self.path)[1], {})

    def test_resave(self):
        written = self.assert_resave_keeps_reading()
        self.assertEqual(written["enabled_extensions"], ["maze_experiments"])


class CurrentEraTests(SavedSettingsCase):
    """cfd9fbd: skip_top_below, and a newer version's key carried through."""

    fixture = "03-current-cfd9fbd.json"

    def test_values(self):
        self.assert_values(CHOSEN | THEMES_ERA | {"skip_top_below": 0.1})
        self.assertEqual(self.loaded.enabled_extensions, ("maze_experiments", "hangman", "probes", "circuits"))
        self.assertEqual(settings.sampling_values(None)["skip_top_below"], 0.1)

    def test_newer_key_is_kept(self):
        self.assertEqual(settings.read(self.path)[1], {"future_setting": {"nested": [1, 2], "on": True}})
        written = self.assert_resave_keeps_reading()
        self.assertEqual(written["future_setting"], {"nested": [1, 2], "on": True})


if __name__ == "__main__":
    unittest.main()
