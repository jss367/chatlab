"""The conversations library (chatlab-library-1) as every earlier writer left it.

Each directory under tests/fixtures/saved/library was written by the commit it
is named after (see its generator) and is never regenerated. These tests open
each one the way a page load does - ``restore_conversations`` over
``library.read`` - and pin what the reader takes from it: branches, turns,
counts, sampling, steering, fork origins and archive state. Each also saves
what it read with today's writer and checks the same reading comes back.
"""

import json
import shutil
import unittest

from chatlab import library, steering
from chatlab.conversation import describe_branch
from chatlab.ui.conversations import restore_conversations, sampling_updates

from saved_formats_support import SAVED, sandbox

LIBRARY = SAVED / "library"
DEFAULT_SLIDERS = [0.8, 0.95, 50, 0.0, 1024]


class SavedLibraryCase(unittest.TestCase):
    fixture = ""

    def setUp(self):
        self.box = sandbox(self)
        self.directory = self.box.copy(LIBRARY / self.fixture)
        self.path = self.box.use_library(self.directory / "conversations.json")
        self.forks = library.read(self.path)
        self.assertIsNotNone(self.forks, "today's reader refused the file")

    # --------------------------------------------------------------- helpers

    def turns(self, name):
        return self.forks["branches"][name]

    def texts(self, name):
        return [(turn["role"], turn["content"]) for turn in self.turns(name)]

    def restored(self):
        return restore_conversations()

    def sliders(self, forks):
        return [update["value"] for update in sampling_updates(forks)]

    def listed(self, frame):
        return [name for _, name in frame["conversation_list"]["choices"]]

    def resaved(self):
        """Today's writer saves what was read into a fresh file; that file read back."""

        target = self.box.root / "resaved" / "conversations.json"
        target.parent.mkdir()
        # The vectors and pictures live beside the file, under its stem.
        for suffix in ("-vectors", "-images"):
            source = self.directory / f"conversations{suffix}"
            if source.is_dir():
                shutil.copytree(source, target.parent / f"conversations{suffix}")
        self.box.use_library(target)
        self.assertEqual(library.write(self.forks, target), target)
        again = library.read(target)
        self.assertIsNotNone(again)
        return again

    def assert_resave_keeps_reading(self):
        again = self.resaved()
        for key in ("active", "branches", "origins", "sampling", "sampling_updated",
                    "archived", "archived_updated", "updated"):
            self.assertEqual(again[key], self.forks[key], key)
        self.assertEqual(list(again["branches"]), list(self.forks["branches"]))
        return again


class FirstEraTests(SavedLibraryCase):
    """02683f2 (#37): branches, turns with counts, stamps, forgotten branches."""

    fixture = "01-first-02683f2"

    def test_branches_turns_and_counts(self):
        self.assertEqual(self.forks["active"], "Fork 1")
        self.assertEqual(list(self.forks["branches"]), ["Main", "Fork 1", "Chat 1"])
        self.assertEqual(self.texts("Main"), [
            ("user", "What is the capital of France?"),
            ("assistant", "The capital of France is Paris."),
            ("user", "And of Italy? Answer with the city: Füße — naïve ✓"),
            ("assistant", "Rome."),
        ])
        answer = self.turns("Main")[1]
        self.assertEqual(answer["reasoning"], "The user wants a capital city. France -> Paris.")
        self.assertEqual((answer["model"], answer["prompt_tokens"], answer["generated_tokens"]),
                         ("allenai/Olmo-3-7B-Think", 31, 58))
        self.assertEqual(describe_branch(self.turns("Main")), {
            "title": "What is the capital of France?",
            "models": ["Qwen3-0.6B", "Olmo-3-7B-Think"], "tokens": 82, "replies": 2,
        })

    def test_interrupted_replies(self):
        # A reply still streaming when the file was written is kept and
        # closed; one that had produced nothing is dropped.
        last = self.turns("Fork 1")[-1]
        self.assertEqual((last["role"], last["content"], last["reasoning"]),
                         ("assistant", "", "Je dois répondre en français"))
        self.assertTrue(last["reasoning_closed"])
        self.assertEqual(self.texts("Chat 1"), [("user", "Write a haiku about rain.")])

    def test_stamps_and_forgotten_branch(self):
        self.assertEqual(self.forks["updated"]["Fork 1"], "2026-09-08T10:05:00.000002+00:00")
        self.assertEqual(self.forks["updated"]["Chat 2"], "2026-09-08T10:07:00.000004+00:00")
        self.assertNotIn("Chat 2", self.forks["branches"])
        self.assertEqual(library.taken_names(self.path), {"Main", "Fork 1", "Chat 1", "Chat 2"})
        for key in ("origins", "sampling", "sampling_updated", "archived", "archived_updated"):
            self.assertEqual(self.forks[key], {}, key)

    def test_page_load(self):
        frame = self.restored()
        self.assertEqual(self.listed(frame), ["Main", "Fork 1", "Chat 1"])
        self.assertEqual(frame["chatbot"][-1], {
            "role": "assistant", "content": "Je dois répondre en français",
            "metadata": {"title": "Reasoning", "status": "done"},
        })
        self.assertEqual(len(frame["chatbot"]), 5)
        self.assertEqual(self.sliders(frame["forks"]), DEFAULT_SLIDERS)

    def test_resave(self):
        self.assert_resave_keeps_reading()


class SamplingEraTests(SavedLibraryCase):
    """cba40be (#45): per-branch sampling under its own stamp."""

    fixture = "02-sampling-cba40be"

    def test_sampling(self):
        self.assertEqual(self.forks["sampling"], {
            "Main": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "max_new_tokens": 256},
            "Fork 1": {"future_knob": {"level": 3}, "temperature": 1.0, "top_k": 0},
        })
        self.assertEqual(self.forks["sampling_updated"], {
            "Main": "2026-09-09T09:00:00.000010+00:00",
            "Fork 1": "2026-09-09T09:30:00.000020+00:00",
            # The sampling was taken away: a stamp with nothing beside it.
            "Chat 1": "2026-09-09T09:40:00.000030+00:00",
        })
        self.assertNotIn("Chat 1", self.forks["sampling"])
        self.assertIn("Fork 2", self.forks["updated"])

    def test_sliders_follow_each_branch(self):
        frame = self.restored()
        self.assertEqual(frame["forks"]["active"], "Main")
        self.assertEqual(self.sliders(frame["forks"]), [0.2, 0.9, 40, 0.0, 256])
        fork = dict(frame["forks"], active="Fork 1")
        self.assertEqual(self.sliders(fork), [1.0, 0.95, 0, 0.0, 1024])
        chat = dict(frame["forks"], active="Chat 1")
        self.assertEqual(self.sliders(chat), DEFAULT_SLIDERS)

    def test_resave(self):
        self.assert_resave_keeps_reading()


class SteeringEraTests(SavedLibraryCase):
    """392ad4d (#59): steering references in sampling and on replies, vectors beside the file."""

    fixture = "03-steering-392ad4d"

    CALM = "8febdf7389f172c5d1934e6ebc577bf7bb9013521fee7c356c218bd8c6db105c"
    FRENCH = "a38191ab3c2e850f084f04c08b1e79396f2aa1e865785baf170acdb2df8cdca1"

    def test_references(self):
        held = self.forks["sampling"]["Main"]["steering"]
        self.assertEqual(held, {
            "format": "chatlab-steering-reference-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 2,
            "strength": 4.0, "enabled": True, "vector_id": self.CALM, "width": 4,
        })
        fork = self.forks["sampling"]["Fork 1"]
        self.assertEqual(fork["temperature"], 0.0)
        self.assertEqual((fork["steering"]["vector_id"], fork["steering"]["strength"],
                          fork["steering"]["enabled"], fork["steering"]["layer"]),
                         (self.FRENCH, -2.5, False, 5))
        self.assertEqual(self.turns("Main")[1]["steering"]["vector_id"], self.CALM)
        self.assertEqual(self.turns("Fork 1")[3]["steering"]["vector_id"], self.FRENCH)

    def test_vectors_expand_from_the_stored_files(self):
        calm = steering.expand(self.forks["sampling"]["Main"]["steering"])
        self.assertEqual(calm["vector"], [0.5, -0.25, 1.0, 0.0])
        self.assertEqual((calm["format"], calm["strength"]), ("chatlab-steering-1", 4.0))
        french = steering.expand(self.turns("Fork 1")[3]["steering"])
        self.assertEqual(french["vector"], [0.125, 0.25, -0.5, 2.0, -1.0, 0.75])

    def test_page_load(self):
        frame = self.restored()
        self.assertEqual(frame["forks"]["active"], "Fork 1")
        self.assertEqual(self.sliders(frame["forks"]), [0.0, 0.95, 50, 0.0, 1024])
        self.assertEqual(frame["turns"][-1]["content"], "Je suis très calme.")

    def test_resave(self):
        again = self.assert_resave_keeps_reading()
        self.assertEqual(steering.expand(again["sampling"]["Main"]["steering"])["vector"],
                         [0.5, -0.25, 1.0, 0.0])


class ForkOriginEraTests(SavedLibraryCase):
    """7e4aaaf (#125): fork origins, natively and as the _chatlab_fork_origin copy."""

    fixture = "04-fork-origin-7e4aaaf"

    def test_origins(self):
        self.assertEqual(self.forks["origins"], {
            "Fork 1": {"parent": "Main", "kind": "message", "turn": 1, "part": "content", "message_count": 2},
            "Fork 2": {"parent": "Main", "kind": "token", "turn": 1, "token": 6, "original": " Paris",
                       "original_id": 12095, "replacement": " Lyon", "replacement_ids": [55123]},
            "Fork 3": {"parent": "Fork 2", "kind": "prompt", "turn": 1, "token": 9, "original": " France",
                       "original_id": 9625, "replacement": " Spain", "replacement_ids": [17689]},
            "Fork 4": {"parent": "Fork 0", "kind": "copy", "turn": 1, "part": "content", "message_count": 2},
        })
        # The wire copy is consumed, never left in the sampling.
        for held in self.forks["sampling"].values():
            self.assertNotIn(library.ORIGIN_COMPAT_KEY, held)
        self.assertNotIn("Fork 3", self.forks["sampling"])

    def test_reply_fields(self):
        answer = self.turns("Main")[1]
        self.assertEqual(answer["thinking_mode"], "on")
        self.assertEqual(answer["generation_settings"]["seed"], 1234)
        self.assertEqual(answer["generation_settings"]["skip_top_below"], 0.05)
        self.assertEqual(self.turns("Main")[3]["generation_settings"]["seed"], 1235)
        self.assertEqual(self.turns("Fork 2")[1]["thinking_mode"], "off")
        # A reply paused by Next token before any visible text keeps its slot.
        paused = self.turns("Fork 1")[-1]
        self.assertEqual((paused["role"], paused["content"]), ("assistant", ""))
        self.assertTrue(paused["token_step_paused"])
        self.assertTrue(paused["reasoning_closed"])
        self.assertEqual(steering.expand(answer["steering"])["vector"], [1.5, -0.5, 0.25])

    def test_sampling(self):
        self.assertEqual(self.forks["sampling"]["Main"]["skip_top_below"], 0.05)
        self.assertEqual(self.forks["sampling"]["Fork 2"], {"temperature": 0.0, "skip_top_below": 0.0})

    def test_page_load_families(self):
        frame = self.restored()
        self.assertEqual(frame["forks"]["active"], "Fork 3")
        # Forks of Main (and a fork of a fork) fold under it; the fork whose
        # parent was deleted heads its own family. The closed family shows
        # only the fork on screen.
        self.assertEqual(self.listed(frame), ["Main", "Fork 3", "Fork 4"])
        self.assertIn("3 forks", frame["conversation_list"]["choices"][0][0])
        self.assertEqual(self.sliders(frame["forks"]), DEFAULT_SLIDERS)
        opened = dict(frame["forks"], _family_view={"Main": True})
        from chatlab.conversation import branch_choices
        self.assertEqual([name for _, name in branch_choices(opened, frame["turns"])],
                         ["Main", "Fork 1", "Fork 2", "Fork 3", "Fork 4"])

    def test_resave(self):
        self.assert_resave_keeps_reading()


class ForkOriginKeptByOlderWriterTests(SavedLibraryCase):
    """A 7e4aaaf file re-saved by 392ad4d: origins survive only in the sampling copy."""

    fixture = "05-fork-origin-via-older-writer-392ad4d"

    def test_the_file_has_no_native_origins(self):
        raw = json.loads((self.directory / "conversations.json").read_text(encoding="utf-8"))
        self.assertFalse(any("origin" in branch for branch in raw["branches"]))
        self.assertTrue(all(library.ORIGIN_COMPAT_KEY in branch.get("sampling", {})
                            for branch in raw["branches"] if branch["name"].startswith("Fork")))

    def test_origins_recovered_from_the_compatibility_copy(self):
        self.assertEqual(self.forks["origins"]["Fork 3"]["parent"], "Fork 2")
        self.assertEqual(self.forks["origins"]["Fork 3"]["replacement_ids"], [17689])
        self.assertEqual(self.forks["origins"]["Fork 2"]["original"], " Paris")
        self.assertEqual(self.forks["origins"]["Fork 4"]["parent"], "Fork 0")
        self.assertEqual(self.forks["origins"]["Fork 1"]["message_count"], 2)
        self.assertEqual(set(self.forks["sampling"]), {"Main", "Fork 1", "Fork 2"})

    def test_what_the_older_writer_did(self):
        # Its new exchange on Fork 3 is there, with its stamp.
        self.assertEqual(self.texts("Fork 3")[2:], [("user", "And of Portugal?"), ("assistant", "Lisbon.")])
        self.assertEqual(self.forks["updated"]["Fork 3"], "2026-09-17T08:00:00.000007+00:00")
        # It knew nothing of thinking modes, settings snapshots or paused
        # steps, so those are gone - including the paused slot it dropped.
        self.assertNotIn("thinking_mode", self.turns("Main")[1])
        self.assertNotIn("generation_settings", self.turns("Main")[1])
        self.assertEqual(self.texts("Fork 1")[-1], ("user", "Shorter, please."))
        self.assertEqual(self.forks["sampling"]["Main"]["skip_top_below"], 0.05)

    def test_page_load_families(self):
        frame = self.restored()
        self.assertEqual(self.listed(frame), ["Main", "Fork 3", "Fork 4"])

    def test_resave_writes_origins_natively(self):
        self.assert_resave_keeps_reading()
        raw = json.loads((self.box.root / "resaved" / "conversations.json").read_text(encoding="utf-8"))
        origins = {branch["name"]: branch.get("origin") for branch in raw["branches"]}
        self.assertEqual(origins["Fork 3"]["parent"], "Fork 2")


class ArchiveEraTests(SavedLibraryCase):
    """d95fa1a: archived branches on their own stamp, pictures and failed replies."""

    fixture = "06-archive-d95fa1a"
    PICTURE = "68c41bb798155f8ad4c0280b6540e49f18457b263986fa6edbf58dc0821f3cb1.png"

    def test_archive_state(self):
        self.assertEqual(self.forks["archived"], {"Fork 1": True, "Chat 1": True})
        self.assertEqual(self.forks["archived_updated"], {
            "Fork 1": "2026-10-05T11:05:00.000030+00:00",
            "Chat 1": "2026-10-05T11:00:00.000020+00:00",
            # Archived and brought back: the stamp alone.
            "Chat 2": "2026-10-05T11:10:00.000040+00:00",
        })
        self.assertEqual(self.forks["sampling"], {"Chat 1": {"temperature": 0.3}})
        self.assertEqual(self.forks["sampling_updated"], {"Chat 1": "2026-10-05T10:00:00.000010+00:00"})

    def test_pictures_and_failures(self):
        self.assertEqual(self.turns("Main")[0]["images"], [self.PICTURE])
        failed = self.turns("Chat 3")[-1]
        self.assertEqual((failed["content"], failed["error"], failed["reasoning"]),
                         ("Essays begin with", "MPS backend out of memory", "Plan: intro, body."))
        self.assertTrue(failed["reasoning_closed"])

    def test_page_load_opens_the_archive(self):
        frame = self.restored()
        # The active conversation is archived, so the page opens on the archive.
        self.assertTrue(frame["forks"]["_archived_view"])
        self.assertEqual(self.listed(frame), ["Fork 1", "Chat 1"])
        self.assertEqual(self.sliders(frame["forks"]), [0.3, 0.95, 50, 0.0, 1024])
        listed = dict(frame["forks"])
        listed.pop("_archived_view")
        main = dict(listed, active="Main")
        from chatlab.conversation import branch_choices, display_messages
        self.assertEqual([name for _, name in branch_choices(main, self.turns("Main"))],
                         ["Main", "Chat 2", "Chat 3"])
        picture = display_messages(self.turns("Main"))[0][0]["content"]
        self.assertTrue(picture["path"].endswith(self.PICTURE))
        self.assertTrue((self.directory / "conversations-images" / self.PICTURE).is_file())
        failure = display_messages(self.turns("Chat 3"))[0][-1]["content"]
        self.assertIn("Generation failed: MPS backend out of memory", failure)

    def test_resave(self):
        self.assert_resave_keeps_reading()


class CurrentEraTests(SavedLibraryCase):
    """cfd9fbd: everything, with both compatibility copies and the archive companion."""

    fixture = "07-current-cfd9fbd"
    VECTOR = "b27d0b755bfb9138a2ba257b12fa6c4a002541dfa5cdbb6f844eebd6a9227ff8"

    def test_branches(self):
        self.assertEqual(list(self.forks["branches"]),
                         ["Main", "Fork 1", "Chat 1", "Chat 2", "Chat 3", "Chat 4"])
        self.assertEqual(self.forks["active"], "Fork 1")
        self.assertEqual(self.forks["origins"], {"Fork 1": {
            "parent": "Main", "kind": "message", "turn": 1, "part": "content", "message_count": 2}})
        self.assertEqual(self.forks["updated"]["Chat 5"], "2026-10-10T09:06:00.000007+00:00")

    def test_turn_fields(self):
        question, answer = self.turns("Main")
        self.assertEqual(question["images"], [
            "637680522ab405996049eb31c699b67146a10e1d6e42f7eb1369382dd16b0a4d.png",
            "e6ac02f2f46e0c9621360e5d005a59dbf6bc202cd49ac9dbb6c2ef590c364dbf.png",
        ])
        self.assertEqual(answer["generation_settings"]["assistant_prefill"], "Sure:")
        self.assertEqual(steering.expand(answer["steering"])["vector"], [0.75, -1.25, 0.5, 2.0, -0.125])
        self.assertEqual(answer["steering"]["strength"], -6.5)
        failed = self.turns("Chat 1")[-1]
        self.assertEqual((failed["content"], failed["error"]), ("", "The model ran out of memory."))
        provenance = self.turns("Chat 2")[0]["institutions_provenance"]
        self.assertEqual(provenance["model_profile"]["adapter"]["subfolder"], "step-200")
        self.assertTrue(self.turns("Chat 2")[1]["institutions_provenance"]["accepted"])

    def test_sampling_and_archive(self):
        self.assertEqual(self.forks["sampling"]["Main"]["future_knob"], "kept")
        self.assertEqual(self.forks["sampling"]["Main"]["steering"]["vector_id"], self.VECTOR)
        self.assertEqual(self.forks["sampling"]["Chat 3"], {"temperature": 1.2})
        for held in self.forks["sampling"].values():
            self.assertNotIn(library.ARCHIVE_COMPAT_KEY, held)
            self.assertNotIn(library.ORIGIN_COMPAT_KEY, held)
        # The wire stamp was advanced to the archive stamp; the real one is read back.
        self.assertEqual(self.forks["sampling_updated"]["Chat 3"], "2026-10-10T10:02:00.000030+00:00")
        self.assertNotIn("Chat 4", self.forks["sampling_updated"])
        self.assertEqual(self.forks["archived"], {"Chat 3": True})
        self.assertEqual(self.forks["archived_updated"], {
            "Chat 3": "2026-10-10T12:00:00.000040+00:00", "Chat 4": "2026-10-10T12:30:00.000050+00:00"})

    def test_page_load(self):
        frame = self.restored()
        self.assertEqual(self.listed(frame), ["Main", "Fork 1", "Chat 1", "Chat 2", "Chat 4"])
        self.assertEqual(self.sliders(frame["forks"]), [0.8, 0.9, 30, 0.02, 300])
        pictures = [message["content"]["path"] for message in frame["chatbot"][:2]]
        self.assertTrue(all(path.startswith(str(self.directory)) for path in pictures))
        self.assertEqual(frame["chatbot"][-1]["content"], "Paused before visible text.")

    def test_resave(self):
        self.assert_resave_keeps_reading()
        companion = json.loads((self.box.root / "resaved" / "conversations.json.archive").read_text())
        self.assertEqual(companion, {
            "Chat 3": {"archived": True, "updated": "2026-10-10T12:00:00.000040+00:00"},
            "Chat 4": {"archived": False, "updated": "2026-10-10T12:30:00.000050+00:00"},
        })


class ArchiveKeptByOlderWriterTests(SavedLibraryCase):
    """A current file re-saved by 7e4aaaf, without the companion: the archive lives in sampling."""

    fixture = "08-archive-via-older-writer-7e4aaaf"

    def test_the_file_has_no_native_archive(self):
        raw = json.loads((self.directory / "conversations.json").read_text(encoding="utf-8"))
        self.assertFalse(any("archived" in branch or "archived_updated" in branch for branch in raw["branches"]))
        self.assertFalse((self.directory / "conversations.json.archive").exists())
        chat_3 = next(branch for branch in raw["branches"] if branch["name"] == "Chat 3")
        self.assertEqual(chat_3["sampling_updated"], "2026-10-10T12:00:00.000040+00:00")

    def test_archive_recovered_from_the_compatibility_copy(self):
        self.assertEqual(self.forks["archived"], {"Chat 3": True})
        self.assertEqual(self.forks["archived_updated"], {
            "Chat 3": "2026-10-10T12:00:00.000040+00:00", "Chat 4": "2026-10-10T12:30:00.000050+00:00"})
        # The real sampling stamp, not the advanced wire stamp.
        self.assertEqual(self.forks["sampling_updated"]["Chat 3"], "2026-10-10T10:02:00.000030+00:00")
        self.assertNotIn("Chat 4", self.forks["sampling_updated"])
        self.assertEqual(self.forks["sampling"]["Chat 3"], {"temperature": 1.2})
        self.assertEqual(self.forks["origins"]["Fork 1"]["parent"], "Main")

    def test_what_the_older_writer_did(self):
        self.assertEqual(self.texts("Main")[2:], [("user", "Thanks!"), ("assistant", "You're welcome.")])
        self.assertEqual(self.forks["updated"]["Main"], "2026-10-11T08:00:00.000008+00:00")
        # Fields it did not know are gone: pictures, the failed reply, provenance.
        self.assertNotIn("images", self.turns("Main")[0])
        self.assertEqual(self.texts("Chat 1"), [("user", "Summarise War and Peace.")])
        self.assertNotIn("institutions_provenance", self.turns("Chat 2")[0])
        self.assertTrue(self.turns("Fork 1")[-1]["token_step_paused"])

    def test_page_load(self):
        frame = self.restored()
        self.assertEqual(self.listed(frame), ["Main", "Fork 1", "Chat 1", "Chat 2", "Chat 4"])

    def test_resave(self):
        self.assert_resave_keeps_reading()


class ArchiveAfterStaleOlderWriterTests(SavedLibraryCase):
    """A stale 7e4aaaf page saved a slider over an archived branch; only the companion remembers."""

    fixture = "09-archive-after-stale-older-writer-7e4aaaf"

    def test_the_file_lost_its_archive_record(self):
        raw = json.loads((self.directory / "conversations.json").read_text(encoding="utf-8"))
        chat_3 = next(branch for branch in raw["branches"] if branch["name"] == "Chat 3")
        self.assertEqual(chat_3["sampling"], {"temperature": 0.7})
        self.assertNotIn("archived", chat_3)

    def test_archive_restored_from_the_companion(self):
        self.assertEqual(self.forks["archived"], {"Chat 3": True})
        self.assertEqual(self.forks["archived_updated"]["Chat 3"], "2026-10-10T12:00:00.000040+00:00")
        self.assertEqual(self.forks["sampling"]["Chat 3"], {"temperature": 0.7})
        self.assertEqual(self.forks["sampling_updated"]["Chat 3"], "2026-10-10T13:00:00.000060+00:00")
        self.assertEqual(self.forks["active"], "Chat 3")
        # Other branches kept the current writer's records.
        self.assertNotIn("Chat 4", self.forks["archived"])
        self.assertEqual(self.forks["archived_updated"]["Chat 4"], "2026-10-10T12:30:00.000050+00:00")

    def test_page_load_opens_the_archive(self):
        frame = self.restored()
        self.assertTrue(frame["forks"]["_archived_view"])
        self.assertEqual(self.listed(frame), ["Chat 3"])
        self.assertEqual(self.sliders(frame["forks"]), [0.7, 0.95, 50, 0.0, 1024])

    def test_resave(self):
        self.assert_resave_keeps_reading()


if __name__ == "__main__":
    unittest.main()
