import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from chatlab import library
from chatlab.conversation import (
    MAIN_BRANCH,
    branch_stamp,
    copy_forks,
    drop_branch,
    make_turn,
    new_forks,
    put_branch,
    put_branch_archived,
    put_branch_sampling,
)

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

    def test_a_save_over_an_unreadable_file_keeps_the_file_aside(self):
        forks = new_forks()
        put_branch(forks, MAIN_BRANCH, [make_turn("user", "keep me")])
        put_branch(forks, "Chat 1", [make_turn("user", "and me")])
        library.write(forks, self.path)
        # One turn this version cannot read makes the whole file unreadable.
        data = json.loads(self.path.read_text())
        data["branches"][1]["turns"][0]["role"] = "system"
        unreadable = json.dumps(data)
        self.path.write_text(unreadable)

        with self.assertLogs(library.logger, level="WARNING"):
            self.assertEqual(library.write(new_forks(), self.path), self.path)
            library.claim_name(new_forks(), "Chat", self.path)
        kept = library.unreadable_copies(self.path)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), unreadable)
        # The save itself went ahead, over nothing.
        self.assertEqual(list(library.read(self.path)["branches"]), [MAIN_BRANCH, "Chat 1"])
        self.assertEqual(library.read(self.path)["branches"]["Chat 1"], [])

    def test_unreadable_backups_do_not_collide_within_one_second(self):
        self.path.parent.mkdir(parents=True)
        with mock.patch.object(library.time, "strftime", return_value="20261003-010000"):
            for raw in ("{first broken", "{second broken"):
                self.path.write_text(raw)
                with self.assertLogs(library.logger, level="WARNING"):
                    library.write(new_forks(), self.path)
        self.assertEqual({p.read_text() for p in library.unreadable_copies(self.path)},
                         {"{first broken", "{second broken"})

    def test_backup_lookup_treats_the_configured_filename_literally(self):
        self.path.parent.mkdir(parents=True)
        target = self.path.with_name("chat[1]*?.json")
        target.write_text("{broken")
        with self.assertLogs(library.logger, level="WARNING"):
            library.write(new_forks(), target)
        copies = library.unreadable_copies(target)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].read_text(), "{broken")

    def test_failed_backup_preservation_aborts_both_save_paths(self):
        self.path.parent.mkdir(parents=True)
        raw = "{only recovery copy"
        self.path.write_text(raw)
        with mock.patch.object(library.os, "replace", side_effect=OSError("filename too long")) as replace:
            with self.assertLogs(library.logger, level="WARNING"):
                self.assertIsNone(library.write(new_forks(), self.path))
                self.assertEqual(library.claim_name(new_forks(), "Chat", self.path), "Chat 1")
            self.assertEqual(replace.call_count, 2)
        self.assertEqual(self.path.read_text(), raw)

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

        with mock.patch("chatlab.library.os.replace", side_effect=OSError("no")):
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


class SamplingFileTests(unittest.TestCase):
    """A conversation's own sampling, through the file and through a merge."""

    SAMPLING = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "skip_top_below": 0.0,
        "max_new_tokens": 256,
    }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "conversations.json"

    def test_a_branchs_sampling_comes_back_with_it(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [make_turn("user", "hi")])
        put_branch_sampling(forks, "Fork 1", self.SAMPLING)

        library.write(forks, self.path)
        restored = library.read(self.path)

        self.assertEqual(restored["sampling"], {"Fork 1": self.SAMPLING})

    def test_a_branch_without_any_writes_nothing_and_reads_back_empty(self):
        forks = new_forks()
        put_branch(forks, MAIN_BRANCH, [make_turn("user", "hi")])

        library.write(forks, self.path)

        saved = json.loads(self.path.read_text())
        self.assertNotIn("sampling", saved["branches"][0])
        self.assertEqual(library.read(self.path)["sampling"], {})

    def test_a_file_written_before_conversations_carried_sampling_still_reads(self):
        # The key is simply absent, which is what every file written by an
        # earlier version looks like.
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        self.assertEqual(library.read(self.path)["sampling"], {})

    def test_a_value_of_the_wrong_type_is_left_out_rather_than_refused(self):
        # The whole pane must still come back: the conversations file is
        # never a good enough reason to lose every conversation in it.
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        saved["branches"][0]["sampling"] = {
            "temperature": "hot",
            "top_k": True,
            "max_new_tokens": 256,
        }
        self.path.write_text(json.dumps(saved))

        restored = library.read(self.path)

        self.assertEqual(restored["sampling"], {MAIN_BRANCH: {"max_new_tokens": 256}})
        self.assertEqual(first_messages(restored), {MAIN_BRANCH: "hi"})

    def test_a_key_this_version_knows_nothing_about_survives_a_save(self):
        # Two machines can share the file without running the same version,
        # and the newer one's per-branch sampling must come back whole.
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        saved["branches"][0]["sampling"] = {
            "temperature": 0.0,
            "repetition_penalty": 1.15,
        }
        self.path.write_text(json.dumps(saved))

        library.write(library.read(self.path), self.path)

        written = json.loads(self.path.read_text())["branches"][0]["sampling"]
        self.assertEqual(written["repetition_penalty"], 1.15)
        self.assertEqual(written["temperature"], 0.0)

    def test_it_survives_a_slider_moved_here_too(self):
        # Not just a read and a save: a temperature moved on this version
        # leaves the newer version's own key where it was.
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        saved["branches"][0]["sampling"] = {
            "temperature": 1.9,
            "repetition_penalty": 1.15,
        }
        self.path.write_text(json.dumps(saved))

        restored = library.read(self.path)
        self.assertTrue(put_branch_sampling(restored, MAIN_BRANCH, self.SAMPLING))
        library.write(restored, self.path)

        written = json.loads(self.path.read_text())["branches"][0]["sampling"]
        self.assertEqual(written["repetition_penalty"], 1.15)
        self.assertEqual(written["temperature"], 0.0)

    def test_sampling_taken_away_does_not_come_back(self):
        # Clear leaves the main conversation empty and unpinned. Its sampling
        # is merged on its own time, so without a stamp of its own the file's
        # older copy looks like the newer of the two and pins it again.
        pinned = stamped(MAIN_BRANCH, Main="hi")
        put_branch_sampling(pinned, MAIN_BRANCH, self.SAMPLING)
        library.write(pinned, self.path)

        cleared = new_forks()
        stamp = branch_stamp()
        cleared["updated"] = {MAIN_BRANCH: stamp}
        cleared["sampling_updated"] = {MAIN_BRANCH: stamp}
        library.write(cleared, self.path)

        restored = library.read(self.path)

        self.assertEqual(restored["sampling"], {})
        # And the removal outlives the merge, so a page still holding the old
        # entry cannot put it back on the next save.
        self.assertIn(MAIN_BRANCH, restored["sampling_updated"])
        library.write(pinned, self.path)
        self.assertEqual(library.read(self.path)["sampling"], {})

    def test_sampling_that_is_not_an_object_is_not_a_file_this_app_wrote(self):
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        saved["branches"][0]["sampling"] = [0.8]
        self.path.write_text(json.dumps(saved))

        # read() reports it in the log and restores nothing, as it does for
        # any file it cannot make sense of.
        self.assertIsNone(library.read(self.path))
        with self.assertRaises(ValueError):
            library.parse(json.dumps(saved))

    def test_a_whole_number_temperature_survives_as_a_number(self):
        # A slider at 1 publishes an int, and JSON keeps it one.
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING | {"temperature": 1})

        library.write(forks, self.path)

        self.assertEqual(
            library.read(self.path)["sampling"][MAIN_BRANCH]["temperature"], 1.0
        )

    def test_a_slider_moved_on_a_stale_page_keeps_the_newer_transcript(self):
        # The two are merged apart. This page is a reply behind the file and
        # has only moved a slider: it must not win the transcript, and the
        # newer transcript must not undo the slider.
        mine = stamped(MAIN_BRANCH, Main="mine")
        put_branch_sampling(mine, MAIN_BRANCH, self.SAMPLING)
        mine["updated"][MAIN_BRANCH] = EARLIER
        theirs = stamped(MAIN_BRANCH, Main="theirs")
        theirs["updated"][MAIN_BRANCH] = LATER

        merged = library.merge(mine, theirs)

        self.assertEqual(first_messages(merged), {MAIN_BRANCH: "theirs"})
        self.assertEqual(merged["sampling"], {MAIN_BRANCH: self.SAMPLING})

    def test_a_reply_saved_moments_ago_survives_a_sampling_only_edit(self):
        # The whole of it, through the file: the other page saved a reply,
        # this page changes only the temperature, and the reply stays.
        theirs = stamped(MAIN_BRANCH, Main="hi")
        theirs["branches"][MAIN_BRANCH].append(reply("an answer"))
        theirs["updated"][MAIN_BRANCH] = LATER
        library.write(theirs, self.path)

        stale = stamped(MAIN_BRANCH, Main="hi")
        stale["updated"][MAIN_BRANCH] = EARLIER
        put_branch_sampling(stale, MAIN_BRANCH, self.SAMPLING)
        library.write(stale, self.path)

        restored = library.read(self.path)

        self.assertEqual(
            [turn["content"] for turn in restored["branches"][MAIN_BRANCH]],
            ["hi", "an answer"],
        )
        self.assertEqual(restored["sampling"], {MAIN_BRANCH: self.SAMPLING})

    def test_the_sampling_moved_more_recently_wins(self):
        mine = stamped(MAIN_BRANCH, Main="mine")
        put_branch_sampling(mine, MAIN_BRANCH, self.SAMPLING)
        theirs = stamped(MAIN_BRANCH, Main="theirs")
        put_branch_sampling(theirs, MAIN_BRANCH, self.SAMPLING | {"temperature": 1.9})
        theirs["sampling_updated"][MAIN_BRANCH] = EARLIER

        merged = library.merge(mine, theirs)

        self.assertEqual(merged["sampling"][MAIN_BRANCH]["temperature"], 0.0)
        # And the other way round, to show the stamp is what decides.
        mine["sampling_updated"][MAIN_BRANCH] = EARLIER
        theirs["sampling_updated"][MAIN_BRANCH] = LATER
        self.assertEqual(
            library.merge(mine, theirs)["sampling"][MAIN_BRANCH]["temperature"], 1.9
        )

    def test_a_branch_only_the_file_has_keeps_its_sampling(self):
        theirs = stamped("Fork 1", **{"Fork 1": "theirs"})
        put_branch_sampling(theirs, "Fork 1", self.SAMPLING)

        merged = library.merge(stamped(MAIN_BRANCH, Main="mine"), theirs)

        self.assertEqual(merged["sampling"], {"Fork 1": self.SAMPLING})


class ArchiveFileTests(unittest.TestCase):
    """Whether a conversation is archived, through the file and through a merge."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "conversations.json"

    def archived(self) -> dict:
        forks = new_forks()
        put_branch(forks, "Chat 1", [make_turn("user", "put away")])
        put_branch_archived(forks, "Chat 1", True)
        return forks

    def test_clear_write_preserves_an_archive_and_its_followup_state_save(self):
        archived = self.archived()
        library.write(archived, self.path)
        cleared = new_forks()
        cleared["updated"]["Chat 1"] = branch_stamp()
        library.write(cleared, self.path, preserve_archived=True)
        # The state-change callback can still hold the clear's tombstone.
        library.write(cleared, self.path)
        saved = library.read(self.path)
        self.assertTrue(saved["archived"]["Chat 1"])
        self.assertEqual(saved["branches"]["Chat 1"][0]["content"], "put away")
        # An explicit later delete is still permanent.
        drop_branch(saved, "Chat 1")
        library.write(saved, self.path)
        self.assertNotIn("Chat 1", library.read(self.path)["branches"])

    def test_an_archived_conversation_comes_back_archived(self):
        library.write(self.archived(), self.path)

        restored = library.read(self.path)

        self.assertEqual(restored["archived"], {"Chat 1": True})
        self.assertEqual(restored["branches"]["Chat 1"][0]["content"], "put away")
        self.assertIn("Chat 1", restored["archived_updated"])

    def test_archive_and_restore_survive_an_older_wire_format_save(self):
        forks = self.archived()
        for archived in (True, False):
            put_branch_archived(forks, "Chat 1", archived)
            payload = json.loads(library.dump(forks))
            for entry in payload["branches"]:
                entry.pop("archived", None)
                entry.pop("archived_updated", None)
            restored = library.parse(json.dumps(payload))
            self.assertEqual(restored["archived"].get("Chat 1", False), archived)
            self.assertEqual(restored["archived_updated"], forks["archived_updated"])
            self.assertNotIn(library.ARCHIVE_COMPAT_KEY, restored["sampling"].get("Chat 1", {}))

    def test_archive_advances_the_wire_carrier_without_claiming_sampling(self):
        forks = new_forks()
        put_branch(forks, "Chat 1", [make_turn("user", "kept")])
        put_branch_sampling(forks, "Chat 1", {"temperature": 0.4})
        stale_wire = json.loads(library.dump(forks))["branches"][1]
        put_branch_archived(forks, "Chat 1", True)
        current_wire = json.loads(library.dump(forks))["branches"][1]
        self.assertGreater(current_wire["sampling_updated"], stale_wire["sampling_updated"])
        restored = library.parse(library.dump(forks))
        self.assertEqual(restored["sampling_updated"], forks["sampling_updated"])
        self.assertEqual(restored["sampling"]["Chat 1"], {"temperature": 0.4})

    def test_an_older_writers_new_sampling_stamp_is_retained(self):
        payload = json.loads(library.dump(self.archived()))
        branch = payload["branches"][1]
        branch.pop("archived")
        branch.pop("archived_updated")
        branch["sampling"]["temperature"] = 0.7
        stamp = branch["sampling_updated"] = branch_stamp()
        restored = library.parse(json.dumps(payload))
        self.assertEqual(restored["sampling_updated"]["Chat 1"], stamp)
        self.assertEqual(restored["sampling"]["Chat 1"], {"temperature": 0.7})
        self.assertTrue(restored["archived"]["Chat 1"])

    def test_stale_old_sampling_rewrite_cannot_erase_the_archive(self):
        archived = self.archived()
        library.write(archived, self.path)
        payload = json.loads(self.path.read_text())
        branch = payload["branches"][1]
        branch.pop("archived")
        branch.pop("archived_updated")
        branch["sampling"] = {"temperature": 0.7}
        branch["sampling_updated"] = branch_stamp()
        self.path.write_text(json.dumps(payload))
        restored = library.read(self.path)
        self.assertTrue(restored["archived"]["Chat 1"])
        self.assertEqual(restored["sampling"]["Chat 1"], {"temperature": 0.7})
        self.assertEqual(restored["archived_updated"], archived["archived_updated"])
        put_branch_archived(restored, "Chat 1", False)
        library.write(restored, self.path)
        self.assertEqual(library.read(self.path)["archived"], {})
        drop_branch(restored, "Chat 1")
        library.write(restored, self.path)
        self.assertEqual(json.loads(library._archive_path(self.path).read_text()), {})

    def test_archive_companion_is_private_and_ignored_for_a_missing_library(self):
        library.write(self.archived(), self.path)
        self.assertEqual(stat.S_IMODE(library._archive_path(self.path).stat().st_mode), 0o600)
        self.path.unlink()
        self.assertIsNone(library.read(self.path))
        library.write(new_forks(), self.path)
        self.assertEqual(json.loads(library._archive_path(self.path).read_text()), {})

    def test_native_restore_stamp_overrides_a_stale_compatibility_copy(self):
        payload = json.loads(library.dump(self.archived()))
        branch = payload["branches"][1]
        branch.pop("archived")
        branch["archived_updated"] = branch_stamp()
        restored = library.parse(json.dumps(payload))
        self.assertEqual(restored["archived"], {})
        self.assertEqual(restored["archived_updated"]["Chat 1"], branch["archived_updated"])

    def test_a_new_fork_does_not_inherit_its_parents_wire_archive(self):
        payload = json.loads(library.dump(self.archived()))
        fork = dict(payload["branches"][1], name="Fork 1")
        fork.pop("archived")
        fork.pop("archived_updated")
        payload["branches"].append(fork)
        restored = library.parse(json.dumps(payload))
        self.assertNotIn("Fork 1", restored["archived"])
        self.assertNotIn("Fork 1", restored["archived_updated"])

    def test_a_file_written_before_archiving_reads_with_nothing_archived(self):
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        self.assertNotIn("archived", saved["branches"][0])

        restored = library.read(self.path)

        self.assertEqual((restored["archived"], restored["archived_updated"]), ({}, {}))

    def test_the_main_conversation_is_never_read_as_archived(self):
        library.write(stamped(MAIN_BRANCH, Main="hi"), self.path)
        saved = json.loads(self.path.read_text())
        saved["branches"][0]["archived"] = True
        self.path.write_text(json.dumps(saved))

        self.assertEqual(library.read(self.path)["archived"], {})

    def test_archiving_survives_a_save_from_a_page_a_reply_ahead(self):
        # This page archived Chat 1 while another page, which loaded before
        # the archiving, went on answering in it. Each keeps its own change.
        theirs = self.archived()
        theirs["archived"], theirs["archived_updated"] = {}, {}
        library.write(theirs, self.path)
        mine = self.archived()
        library.write(mine, self.path)

        put_branch(theirs, "Chat 1", [make_turn("user", "put away"), reply("answered")])
        library.write(theirs, self.path)

        saved = library.read(self.path)
        self.assertEqual(saved["archived"], {"Chat 1": True})
        self.assertEqual(saved["branches"]["Chat 1"][-1]["content"], "answered")

    def test_bringing_a_conversation_back_is_not_undone_by_an_older_archived_copy(self):
        stale = self.archived()
        library.write(stale, self.path)
        restored = copy_forks(stale)
        put_branch_archived(restored, "Chat 1", False)
        library.write(restored, self.path)

        library.write(stale, self.path)

        saved = library.read(self.path)
        self.assertEqual(saved["archived"], {})
        self.assertEqual(saved["archived_updated"], restored["archived_updated"])

    def test_a_deleted_conversation_leaves_no_archive_entry_behind(self):
        forks = self.archived()
        drop_branch(forks, "Chat 1")

        library.write(forks, self.path)

        saved = library.read(self.path)
        self.assertNotIn("Chat 1", saved["branches"])
        self.assertEqual((saved["archived"], saved["archived_updated"]), ({}, {}))


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
        self.assertEqual(library.merge(mine, None), copy_forks(mine))
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

    def test_a_claimed_name_is_spoken_for_before_the_branch_is_saved(self):
        tab_a = stamped(MAIN_BRANCH, Main="hi")
        tab_b = stamped(MAIN_BRANCH, Main="hi")

        first = library.claim_name(tab_a, "Chat", self.path)
        second = library.claim_name(tab_b, "Chat", self.path)

        self.assertEqual((first, second), ("Chat 1", "Chat 2"))
        saved = library.read(self.path)
        self.assertEqual(saved["branches"]["Chat 1"], [])
        self.assertEqual(saved["branches"]["Chat 2"], [])
        # Neither page's own copy of the pane was touched.
        self.assertNotIn("Chat 1", tab_a["branches"])

    def test_claims_from_many_threads_never_collide(self):
        import threading

        pages = [stamped(MAIN_BRANCH, Main="hi") for _ in range(8)]
        barrier = threading.Barrier(len(pages))
        names: list[str] = []

        def claim(page):
            barrier.wait()
            names.append(library.claim_name(page, "Chat", self.path))

        threads = [threading.Thread(target=claim, args=(page,)) for page in pages]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(names), [f"Chat {n}" for n in range(1, 9)])
        self.assertEqual(len(set(names)), len(pages))

    def test_a_claim_steps_over_a_forgotten_name(self):
        forks = stamped(MAIN_BRANCH, Main="hi")
        forks["updated"]["Chat 1"] = LATER
        library.write(forks, self.path)

        self.assertEqual(library.claim_name(stamped(MAIN_BRANCH, Main="hi"), "Chat", self.path), "Chat 2")

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
