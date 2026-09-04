"""The settings file: where it is, what survives a bad edit, and what is saved."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import settings


class SettingsPathTests(unittest.TestCase):
    def test_the_default_is_under_the_config_directory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                settings.settings_path(),
                Path.home() / ".config" / "chatlab" / "settings.json",
            )

    def test_the_xdg_variable_moves_the_config_directory(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/conf"}, clear=True):
            self.assertEqual(
                settings.settings_path(), Path("/tmp/conf/chatlab/settings.json")
            )

    def test_the_settings_path_variable_names_the_file_itself(self):
        with mock.patch.dict(
            os.environ,
            {"CHATLAB_SETTINGS_PATH": "~/elsewhere.json", "XDG_CONFIG_HOME": "/tmp/c"},
            clear=True,
        ):
            self.assertEqual(settings.settings_path(), Path.home() / "elsewhere.json")


class ReadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "settings.json"

    def write_file(self, payload) -> None:
        self.path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )

    def test_a_missing_file_is_every_default(self):
        self.assertEqual(settings.read(self.path), (settings.DEFAULTS, {}))

    def test_a_saved_value_is_read_back(self):
        self.write_file({"temperature": 0.25, "system_prompt": "Be brief."})
        saved, _unknown = settings.read(self.path)

        self.assertEqual(saved.temperature, 0.25)
        self.assertEqual(saved.system_prompt, "Be brief.")
        # Everything the file did not mention keeps its default.
        self.assertEqual(saved.top_p, settings.DEFAULTS.top_p)

    def test_a_file_that_is_not_json_falls_back_to_the_defaults(self):
        self.write_file("{not json at all")

        with self.assertLogs("settings", level="WARNING"):
            self.assertEqual(settings.read(self.path), (settings.DEFAULTS, {}))

    def test_a_file_holding_something_other_than_an_object_is_ignored(self):
        self.write_file([1, 2, 3])

        with self.assertLogs("settings", level="WARNING"):
            self.assertEqual(settings.read(self.path), (settings.DEFAULTS, {}))

    def test_a_value_out_of_range_is_pulled_into_it(self):
        self.write_file({"temperature": 40, "top_k": -5, "seed": -1})
        saved, _unknown = settings.read(self.path)

        self.assertEqual(saved.temperature, 2.0)
        self.assertEqual(saved.top_k, 0)
        self.assertEqual(saved.seed, 0)

    def test_a_value_of_the_wrong_shape_falls_back_to_its_default(self):
        self.write_file(
            {
                "temperature": "warm",
                "keep_reasoning": "yes",
                "color_scale": "Ultraviolet",
            }
        )
        saved, _unknown = settings.read(self.path)

        self.assertEqual(saved.temperature, settings.DEFAULTS.temperature)
        self.assertEqual(saved.keep_reasoning, settings.DEFAULTS.keep_reasoning)
        self.assertEqual(saved.color_scale, settings.DEFAULTS.color_scale)

    def test_a_lowered_context_limit_lowers_a_saved_response_length(self):
        self.write_file({"prefill_token_limit": 512, "max_new_tokens": 4096})
        saved, _unknown = settings.read(self.path)

        self.assertEqual(saved.prefill_token_limit, 512)
        self.assertEqual(saved.max_new_tokens, 512)

    def test_an_empty_model_id_falls_back_to_the_default_model(self):
        self.write_file({"model_id": ""})
        saved, _unknown = settings.read(self.path)

        self.assertEqual(saved.model_id, settings.DEFAULT_MODEL_ID)

    def test_no_metal_cap_is_a_null_rather_than_a_number(self):
        self.write_file({"mps_memory_fraction": None})
        saved, _unknown = settings.read(self.path)

        self.assertIsNone(saved.mps_memory_fraction)

    def test_keys_this_version_does_not_know_are_handed_back(self):
        self.write_file({"temperature": 0.5, "from_a_later_version": 7})
        saved, unknown = settings.read(self.path)

        self.assertEqual(saved.temperature, 0.5)
        self.assertEqual(unknown, {"from_a_later_version": 7})


class WriteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "settings.json"

    def test_saving_creates_the_directory_and_a_readable_file(self):
        written = settings.write(settings.DEFAULTS, path=self.root / "deep" / "s.json")

        self.assertEqual(written, self.root / "deep" / "s.json")
        self.assertEqual(
            json.loads(written.read_text(encoding="utf-8")),
            settings.DEFAULTS.to_mapping(),
        )

    def test_the_file_holds_every_setting_so_it_can_be_edited_by_hand(self):
        settings.write(settings.DEFAULTS, path=self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual(set(payload), set(settings.DEFAULTS.to_mapping()))
        self.assertIn("prefill_token_limit", payload)
        self.assertIn("mps_memory_fraction", payload)

    def test_the_hugging_face_token_is_not_one_of_the_settings(self):
        settings.write(settings.DEFAULTS, path=self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertNotIn("hf_token", payload)
        self.assertEqual(
            [key for key in payload if "secret" in key or key.endswith("token")], []
        )

    def test_keys_from_a_later_version_are_put_back_where_they_were(self):
        settings.write(settings.DEFAULTS, {"from_a_later_version": 7}, path=self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["from_a_later_version"], 7)

    def test_a_file_that_cannot_be_written_is_reported_and_survived(self):
        blocked = self.root / "file" / "settings.json"
        blocked.parent.write_text("in the way", encoding="utf-8")

        with self.assertLogs("settings", level="WARNING"):
            self.assertIsNone(settings.write(settings.DEFAULTS, path=blocked))

    def test_a_failed_write_leaves_no_temporary_file_behind(self):
        with mock.patch("settings.os.replace", side_effect=OSError("no")):
            with self.assertLogs("settings", level="WARNING"):
                settings.write(settings.DEFAULTS, path=self.path)

        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_symlinked_file_is_written_through_rather_than_replaced(self):
        """The documented way to share one file: settings.json is a link."""

        shared = self.root / "dotfiles" / "chatlab.json"
        shared.parent.mkdir()
        settings.write(settings.sanitize({"temperature": 0.3}), path=shared)
        self.path.symlink_to(shared)

        settings.write(settings.sanitize({"temperature": 0.9}), path=self.path)

        self.assertTrue(self.path.is_symlink())
        self.assertEqual(self.path.readlink(), shared)
        self.assertEqual(json.loads(shared.read_text(encoding="utf-8"))["temperature"], 0.9)

    def test_a_symlink_pointing_nowhere_yet_creates_its_destination(self):
        shared = self.root / "dotfiles" / "chatlab.json"
        self.path.symlink_to(shared)

        settings.write(settings.sanitize({"temperature": 0.9}), path=self.path)

        self.assertTrue(self.path.is_symlink())
        self.assertEqual(json.loads(shared.read_text(encoding="utf-8"))["temperature"], 0.9)

    def test_a_save_replaces_the_file_rather_than_truncating_it(self):
        settings.write(settings.DEFAULTS, path=self.path)
        first = self.path.stat().st_ino
        settings.write(settings.sanitize({"temperature": 0.1}), path=self.path)

        self.assertNotEqual(self.path.stat().st_ino, first)


class ProcessSettingsTests(unittest.TestCase):
    """The settings the process runs under, and what changing one writes."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "settings.json"
        # The reader's own file is restored after the temporary one goes away.
        self.addCleanup(settings.load)
        patch = mock.patch.dict(
            os.environ, {settings.SETTINGS_PATH_ENV: str(self.path)}
        )
        patch.start()
        self.addCleanup(patch.stop)
        settings.load()

    def saved(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_file_is_read_once_and_remembered(self):
        settings.write(settings.sanitize({"temperature": 0.3}), path=self.path)

        self.assertEqual(settings.current().temperature, settings.DEFAULTS.temperature)
        self.assertEqual(settings.load().temperature, 0.3)
        self.assertEqual(settings.current().temperature, 0.3)

    def test_changing_a_setting_saves_the_whole_file(self):
        settings.update(temperature=0.2, system_prompt="Be brief.")

        self.assertEqual(self.saved()["temperature"], 0.2)
        self.assertEqual(self.saved()["system_prompt"], "Be brief.")
        self.assertEqual(settings.current().temperature, 0.2)

    def test_a_change_that_changes_nothing_is_not_written(self):
        settings.update(temperature=0.2)
        before = self.path.stat().st_mtime_ns
        settings.update(temperature=0.2)

        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_a_value_from_a_control_is_sanitized_on_the_way_in(self):
        settings.update(top_k=1000, color_scale="Ultraviolet")

        self.assertEqual(settings.current().top_k, 200)
        self.assertEqual(settings.current().color_scale, settings.DEFAULTS.color_scale)

    def test_a_change_keeps_the_keys_a_later_version_left(self):
        self.path.write_text(
            json.dumps({"temperature": 0.4, "from_a_later_version": 7}),
            encoding="utf-8",
        )
        settings.load()
        settings.update(temperature=0.5)

        self.assertEqual(self.saved()["from_a_later_version"], 7)

    def test_the_file_is_created_when_it_is_missing(self):
        self.assertFalse(self.path.exists())

        self.assertEqual(settings.ensure_file(), self.path)
        self.assertEqual(self.saved(), settings.DEFAULTS.to_mapping())

    def test_an_existing_file_is_left_exactly_as_it_was(self):
        self.path.write_text('{"temperature": 0.4}', encoding="utf-8")

        settings.ensure_file()

        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"temperature": 0.4}')

    def test_overriding_settings_touches_neither_the_file_nor_what_follows(self):
        settings.update(temperature=0.2)

        with settings.override(temperature=1.5) as overridden:
            self.assertEqual(overridden.temperature, 1.5)
            self.assertEqual(settings.current().temperature, 1.5)
            self.assertEqual(self.saved()["temperature"], 0.2)

        self.assertEqual(settings.current().temperature, 0.2)


class StartupModelTests(unittest.TestCase):
    def test_the_saved_model_is_the_one_offered(self):
        chosen = settings.sanitize({"model_id": "org/saved-model"})

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.model_id_at_startup(chosen), "org/saved-model")

    def test_the_environment_still_pins_a_model_for_one_run(self):
        chosen = settings.sanitize({"model_id": "org/saved-model"})

        with mock.patch.dict(os.environ, {"OLMO_MODEL_ID": "org/pinned"}, clear=True):
            self.assertEqual(settings.model_id_at_startup(chosen), "org/pinned")


if __name__ == "__main__":
    unittest.main()
