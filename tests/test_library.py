import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import library
from conversation import MAIN_BRANCH, make_turn, new_forks


def reply(content: str, model: str = "org/model", prompt_tokens: int = 10) -> dict:
    turn = make_turn("assistant", content, "because")
    turn["reasoning_closed"] = True
    turn.update(model=model, prompt_tokens=prompt_tokens, generated_tokens=3)
    return turn


class PathTests(unittest.TestCase):
    def test_the_default_sits_under_the_xdg_data_home(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/data"}, clear=False):
            os.environ.pop(library.LIBRARY_PATH_ENV, None)
            self.assertEqual(
                library.library_path(), Path("/data/chatlab/conversations.json")
            )

    def test_the_default_falls_back_to_local_share(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(library.LIBRARY_PATH_ENV, None)
            os.environ.pop("XDG_DATA_HOME", None)
            self.assertEqual(
                library.library_path(),
                Path.home() / ".local" / "share" / "chatlab" / "conversations.json",
            )

    def test_the_environment_names_the_file_outright(self):
        with mock.patch.dict(
            os.environ, {library.LIBRARY_PATH_ENV: "~/elsewhere/c.json"}, clear=False
        ):
            self.assertEqual(
                library.library_path(), Path.home() / "elsewhere" / "c.json"
            )


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "conversations.json"

    def test_every_branch_and_the_active_one_come_back(self):
        forks = {
            "active": "Fork 1",
            "branches": {
                MAIN_BRANCH: [make_turn("user", "hi"), reply("hello")],
                "Fork 1": [make_turn("user", "hi"), reply("hey", model="org/other")],
                "Chat 2": [],
            },
        }

        self.assertEqual(library.write(forks, self.path), self.path)
        restored = library.read(self.path)

        self.assertEqual(restored["active"], "Fork 1")
        self.assertEqual(list(restored["branches"]), [MAIN_BRANCH, "Fork 1", "Chat 2"])
        self.assertEqual(restored["branches"]["Chat 2"], [])
        first = restored["branches"][MAIN_BRANCH][1]
        self.assertEqual(first["content"], "hello")
        self.assertEqual(first["reasoning"], "because")
        self.assertEqual(first["model"], "org/model")
        self.assertEqual(first["prompt_tokens"], 10)
        self.assertEqual(first["generated_tokens"], 3)
        self.assertEqual(restored["branches"]["Fork 1"][1]["model"], "org/other")

    def test_the_file_is_private_and_written_whole(self):
        library.write(new_forks(), self.path)

        mode = stat.S_IMODE(self.path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(
            [entry.name for entry in self.path.parent.iterdir()], [self.path.name]
        )
        self.assertEqual(json.loads(self.path.read_text())["format"], library.LIBRARY_FORMAT)

    def test_a_missing_file_restores_nothing(self):
        self.assertIsNone(library.read(self.path))

    def test_an_unreadable_file_is_skipped_and_left_alone(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json")

        with self.assertLogs(library.logger, level="WARNING"):
            self.assertIsNone(library.read(self.path))
        self.assertEqual(self.path.read_text(), "{not json")

    def test_a_file_from_another_app_is_refused(self):
        with self.assertRaises(ValueError):
            library.parse(json.dumps({"format": "other", "branches": []}))
        with self.assertRaises(ValueError):
            library.parse(json.dumps({"format": library.LIBRARY_FORMAT, "branches": {}}))
        with self.assertRaises(ValueError):
            library.parse(
                json.dumps(
                    {
                        "format": library.LIBRARY_FORMAT,
                        "branches": [{"name": "A", "turns": []}, {"name": "A", "turns": []}],
                    }
                )
            )

    def test_a_write_that_fails_is_reported_not_raised(self):
        blocked = Path(self.directory.name) / "file"
        blocked.write_text("in the way")

        with self.assertLogs(library.logger, level="WARNING"):
            self.assertIsNone(library.write(new_forks(), blocked / "conversations.json"))

    def test_an_unknown_active_branch_falls_back_to_the_first(self):
        restored = library.parse(
            json.dumps(
                {
                    "format": library.LIBRARY_FORMAT,
                    "active": "gone",
                    "branches": [{"name": "Chat 3", "turns": []}],
                }
            )
        )
        self.assertEqual(restored["active"], "Chat 3")

    def test_an_empty_list_of_branches_becomes_the_main_conversation(self):
        restored = library.parse(
            json.dumps({"format": library.LIBRARY_FORMAT, "branches": []})
        )
        self.assertEqual(restored, new_forks())

    def test_a_reply_cut_off_mid_stream_is_kept_and_closed(self):
        # The file was written while a response streamed, so the last turn
        # never had its reasoning block closed.
        forks = {
            "active": MAIN_BRANCH,
            "branches": {MAIN_BRANCH: [make_turn("user", "hi"), make_turn("assistant", "part", "")]},
        }
        library.write(forks, self.path)

        turns = library.read(self.path)["branches"][MAIN_BRANCH]
        self.assertEqual(turns[-1]["content"], "part")
        self.assertTrue(turns[-1]["reasoning_closed"])

    def test_a_reply_that_had_produced_nothing_is_dropped(self):
        forks = {
            "active": MAIN_BRANCH,
            "branches": {MAIN_BRANCH: [make_turn("user", "hi"), make_turn("assistant", "", "")]},
        }
        library.write(forks, self.path)

        turns = library.read(self.path)["branches"][MAIN_BRANCH]
        self.assertEqual([turn["role"] for turn in turns], ["user"])


class AsSeenTests(unittest.TestCase):
    def test_the_active_branch_is_read_from_the_conversation(self):
        forks = {
            "active": "Fork 1",
            "branches": {MAIN_BRANCH: [make_turn("user", "old")], "Fork 1": [make_turn("user", "stale")]},
        }
        on_screen = [make_turn("user", "stale"), reply("fresh")]

        seen = library.as_seen(forks, on_screen)

        self.assertEqual(seen["branches"]["Fork 1"][1]["content"], "fresh")
        self.assertEqual(seen["branches"][MAIN_BRANCH][0]["content"], "old")
        # A copy, not a view: the state must not be mutated by saving it.
        self.assertEqual(forks["branches"]["Fork 1"][0]["content"], "stale")
        self.assertEqual(len(forks["branches"]["Fork 1"]), 1)

    def test_nothing_at_all_is_the_empty_main_conversation(self):
        self.assertEqual(library.as_seen(None, None), new_forks())


if __name__ == "__main__":
    unittest.main()
