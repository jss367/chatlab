import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import library
from conversation import MAIN_BRANCH, drop_branch, make_turn, new_forks, put_branch

EARLIER = "2026-09-01T10:00:00.000000+00:00"
LATER = "2026-09-01T11:00:00.000000+00:00"


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

    def test_a_replace_that_fails_leaves_no_staging_file_behind(self):
        library.write(new_forks(), self.path)

        with mock.patch("library.os.replace", side_effect=OSError("no")):
            with self.assertLogs(library.logger, level="WARNING"):
                self.assertIsNone(library.write(new_forks(), self.path))

        self.assertEqual([entry.name for entry in self.path.parent.iterdir()], [self.path.name])

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
        # The branch on screen changed, so it is stamped; the other is not.
        self.assertEqual(list(seen["updated"]), ["Fork 1"])

    def test_an_unchanged_active_branch_keeps_its_stamp(self):
        turns = [make_turn("user", "same")]
        forks = {"active": MAIN_BRANCH, "branches": {MAIN_BRANCH: turns}, "updated": {MAIN_BRANCH: EARLIER}}

        seen = library.as_seen(forks, [make_turn("user", "same")])

        self.assertEqual(seen["updated"], {MAIN_BRANCH: EARLIER})

    def test_nothing_at_all_is_the_empty_main_conversation(self):
        self.assertEqual(library.as_seen(None, None), new_forks())


def stamped(active: str, **branches) -> dict:
    """Forks whose every branch is stamped ``EARLIER``; ``branches`` maps names to first messages."""

    return {
        "active": active,
        "branches": {name: [make_turn("user", text)] for name, text in branches.items()},
        "updated": {name: EARLIER for name in branches},
    }


def first_messages(forks: dict) -> dict:
    return {name: turns[0]["content"] if turns else None for name, turns in forks["branches"].items()}


class MergeTests(unittest.TestCase):
    """Two pages saving to one file keep each other's work.

    Each page holds the pane as it was when it loaded, so a save from one
    must not replace what the other has saved since. The stamps decide.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "conversations.json"

    def test_a_branch_this_page_never_saw_survives_its_save(self):
        # The other page started Chat 2 after this one loaded.
        library.write(stamped(MAIN_BRANCH, Main="hi", **{"Chat 2": "other"}), self.path)
        mine = stamped(MAIN_BRANCH, Main="hi")
        mine["updated"][MAIN_BRANCH] = LATER

        library.write(mine, self.path)

        saved = library.read(self.path)
        self.assertEqual(list(saved["branches"]), [MAIN_BRANCH, "Chat 2"])
        self.assertEqual(saved["branches"]["Chat 2"][0]["content"], "other")
        self.assertEqual(saved["updated"]["Chat 2"], EARLIER)

    def test_the_newer_copy_of_a_branch_wins_either_way(self):
        library.write(stamped(MAIN_BRANCH, Main="theirs"), self.path)

        mine = stamped(MAIN_BRANCH, Main="mine")
        mine["updated"][MAIN_BRANCH] = LATER
        library.write(mine, self.path)
        self.assertEqual(first_messages(library.read(self.path)), {MAIN_BRANCH: "mine"})

        stale = stamped(MAIN_BRANCH, Main="stale")
        library.write(stale, self.path)
        self.assertEqual(first_messages(library.read(self.path)), {MAIN_BRANCH: "mine"})

    def test_a_copy_with_no_stamp_loses_to_a_dated_one(self):
        # A page that never touched the branch cannot claim it.
        library.write(stamped(MAIN_BRANCH, Main="dated"), self.path)
        mine = {"active": MAIN_BRANCH, "branches": {MAIN_BRANCH: [make_turn("user", "undated")]}}

        library.write(mine, self.path)

        self.assertEqual(first_messages(library.read(self.path)), {MAIN_BRANCH: "dated"})

    def test_a_deleted_branch_is_not_brought_back(self):
        both = stamped(MAIN_BRANCH, Main="hi", **{"Fork 1": "gone"})
        library.write(both, self.path)

        mine = dict(both, branches=dict(both["branches"]), updated=dict(both["updated"]))
        drop_branch(mine, "Fork 1")
        library.write(mine, self.path)
        self.assertEqual(list(library.read(self.path)["branches"]), [MAIN_BRANCH])

        # The other page still holds Fork 1 as it was, and saves.
        library.write(both, self.path)
        saved = library.read(self.path)
        self.assertEqual(list(saved["branches"]), [MAIN_BRANCH])
        # The deletion is remembered in the file for the next such save.
        self.assertIn("Fork 1", json.loads(self.path.read_text())["forgotten"])
        self.assertEqual(saved["updated"]["Fork 1"], mine["updated"]["Fork 1"])

    def test_a_branch_changed_after_its_deletion_comes_back(self):
        both = stamped(MAIN_BRANCH, Main="hi", **{"Fork 1": "gone"})
        library.write(both, self.path)
        mine = dict(both, branches=dict(both["branches"]), updated=dict(both["updated"]))
        drop_branch(mine, "Fork 1")
        library.write(mine, self.path)

        theirs = dict(both, branches=dict(both["branches"]), updated=dict(both["updated"]))
        put_branch(theirs, "Fork 1", [make_turn("user", "revived")])
        library.write(theirs, self.path)

        saved = library.read(self.path)
        self.assertEqual(saved["branches"]["Fork 1"][0]["content"], "revived")
        self.assertNotIn("Fork 1", json.loads(self.path.read_text())["forgotten"])

    def test_the_order_is_this_pages_then_the_files(self):
        library.write(stamped("Chat 2", Main="hi", **{"Chat 2": "b", "Fork 1": "c"}), self.path)
        mine = stamped("Fork 3", Main="hi", **{"Fork 3": "d"})

        library.write(mine, self.path)

        saved = library.read(self.path)
        self.assertEqual(list(saved["branches"]), [MAIN_BRANCH, "Fork 3", "Chat 2", "Fork 1"])
        self.assertEqual(saved["active"], "Fork 3")

    def test_a_first_save_needs_no_file(self):
        mine = stamped(MAIN_BRANCH, Main="hi")
        self.assertEqual(library.merge(mine, None), mine)
        library.write(mine, self.path)
        self.assertEqual(first_messages(library.read(self.path)), {MAIN_BRANCH: "hi"})

    def test_stamps_and_deletions_round_trip_through_the_file(self):
        forks = stamped(MAIN_BRANCH, Main="hi")
        forks["updated"]["Fork 1"] = LATER
        library.write(forks, self.path)

        raw = json.loads(self.path.read_text())
        self.assertEqual(raw["branches"][0]["updated"], EARLIER)
        self.assertEqual(raw["forgotten"], {"Fork 1": LATER})
        self.assertEqual(library.read(self.path)["updated"], {MAIN_BRANCH: EARLIER, "Fork 1": LATER})

    def test_two_threads_saving_at_once_keep_each_others_branches(self):
        # Two listeners save from Gradio's worker threads at the same moment,
        # each with a different branch changed. Every read-merge-replace must
        # run whole, or the later replace drops what the earlier merged in.
        rounds = 40
        barrier = threading.Barrier(2)
        failures = []

        def saver(name: str) -> None:
            mine = stamped(MAIN_BRANCH, Main="hi", **{name: f"{name} 0"})
            for round_number in range(rounds):
                put_branch(mine, name, [make_turn("user", f"{name} {round_number}")])
                barrier.wait()
                if library.write(mine, self.path) is None:
                    failures.append(name)

        threads = [threading.Thread(target=saver, args=(name,)) for name in ("Chat 2", "Fork 1")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(failures, [])
        saved = library.read(self.path)
        self.assertEqual(set(saved["branches"]), {MAIN_BRANCH, "Chat 2", "Fork 1"})
        self.assertEqual(
            first_messages(saved),
            {MAIN_BRANCH: "hi", "Chat 2": f"Chat 2 {rounds - 1}", "Fork 1": f"Fork 1 {rounds - 1}"},
        )
        self.assertEqual([entry.name for entry in self.path.parent.iterdir()], [self.path.name])

    def test_the_names_the_file_has_spoken_for(self):
        self.assertEqual(library.taken_names(self.path), set())

        forks = stamped(MAIN_BRANCH, Main="hi", **{"Chat 2": "b"})
        forks["updated"]["Fork 1"] = LATER
        library.write(forks, self.path)

        self.assertEqual(library.taken_names(self.path), {MAIN_BRANCH, "Chat 2", "Fork 1"})

    def test_a_bad_stamp_or_forgotten_list_is_refused(self):
        with self.assertRaises(ValueError):
            library.parse(
                json.dumps(
                    {
                        "format": library.LIBRARY_FORMAT,
                        "branches": [{"name": "A", "turns": [], "updated": 5}],
                    }
                )
            )
        with self.assertRaises(ValueError):
            library.parse(
                json.dumps(
                    {"format": library.LIBRARY_FORMAT, "branches": [], "forgotten": ["A"]}
                )
            )


if __name__ == "__main__":
    unittest.main()
