"""Hangman files written by earlier ChatLab versions still open, and read the same way.

Each fixture under tests/fixtures/saved/hangman was written once by the commit its generator names and is never
regenerated. Opening a saved game reads its boards, revealed words and probe answers again from the recorded
text with today's parsers, so these tests pin what each old file says when the page opens it: the boards
per response, the revealed word, the contradictions, the probe answers, the fork it came from, and the
prompt it was played under. A change to a parser that alters any of that fails here.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gradio as gr

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.hangman.batch import summarize
from chatlab.extensions.hangman.game import (
    GIVE_UP, OPENING, SYSTEM, load, messages_for, reasoning_of, saved,
)
from chatlab.extensions.hangman.page import build_page
from hangman_support import CharacterModel
from ui_support import handlers_by_name

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "saved" / "hangman"
# The check panel counts dictionary words that fit; the system word list differs between machines.
WORDS = ("crane", "crate", "grape", "slate", "stone")
HIDDEN = "_"


def cells(text):
    return [HIDDEN if c == "_" else c for c in text.replace(" ", "")]


class Idle(CharacterModel):
    """A model nobody asks to generate: opening a saved game touches no session."""
    busy = False


class SavedGameFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        context = ExtensionContext(ModelService(lambda: Idle()), TokenInspector(), self.directory / "hangman",
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = handlers_by_name(demo)
        for target in ("chatlab.extensions.hangman.game.dictionary", "chatlab.extensions.hangman.page.dictionary",
                       "chatlab.extensions.hangman.batch.dictionary"):
            patcher = mock.patch(target, return_value=WORDS)
            patcher.start()
            self.addCleanup(patcher.stop)

    def open(self, path):
        """What the page shows after "Open a saved game": its outputs, keyed by component."""
        names = ("game", "chat", "check", "picker", "note", "tokens", "strip", "raw", "download", "detail",
                 "alternatives")
        return dict(zip(names, self.fn["open_saved"](str(path), "owner"), strict=True))

    def resaved(self, game):
        """``game`` written by today's writer and opened again."""
        path = self.directory / "resaved.json"
        path.write_text(saved(game), encoding="utf-8")
        return path


class HandPlayedGameTests(SavedGameFixture):
    """hand-played-reopened-branched.json: written at 4d2b657 (#147), before batches and probes."""
    PATH = FIXTURES / "hand-played-reopened-branched.json"
    ID = "000000000000000000000012d71843dd"
    PARENT = "000000000000000000000012d71843dc"
    GUESSES = ["Let's play. Think of your word and show me the empty board.", "e", "a", "s", "r", "o", "crane"]
    BOARDS = [cells("_____"), cells("____e"), cells("__a_e"), cells("__a_e"), None, cells("_ro_e"), cells("crane")]
    PROBLEMS = [
        (6, "Position 3 showed A and now shows O."),
        (6, "A was placed at 3 and is now at no position."),
        (7, "Position 3 showed O and now shows A."),
        (7, "O was placed at 3 and is now at no position."),
        (7, "The revealed word CRATE has A at position 3, where the board showed O."),
        (7, "The revealed word CRATE has T at position 4, where the board showed N."),
        (7, "The revealed word CRATE has O at no position; the board placed it at 3."),
        (7, "The revealed word CRATE has N at no position; the board placed it at 4."),
    ]

    def assert_game_reads(self, game):
        """What the file says, read again from its text: the facts a re-save has to keep too."""
        self.assertEqual(game["format"], "chatlab-hangman-1")
        self.assertEqual(game["system"], SYSTEM + "\nUse only common five-letter English nouns.")
        turns = game["turns"]
        self.assertEqual([t["guess"] for t in turns], self.GUESSES)
        self.assertEqual([t["board"] for t in turns], self.BOARDS)
        self.assertEqual([t["revealed_word"] for t in turns], [None] * 6 + ["crate"])
        self.assertEqual([t["finish_reason"] for t in turns], ["stop"] * 3 + ["length"] + ["stop"] * 3)
        self.assertEqual([t["reasoning_prefilled"] for t in turns], [False, True] + [False] * 5)
        self.assertEqual([t["sampling"]["seed"] for t in turns], list(range(40, 47)))
        self.assertEqual({t["sampling"]["temperature"] for t in turns}, {0.7})
        self.assertEqual(turns[3]["sampling"]["max_new_tokens"], 20)
        # The visible answer drops a closed reasoning block, and the whole of an opened one.
        self.assertEqual(turns[0]["answer"],
                         "Board: _ _ _ _ _\nGuessed: none\nWrong guesses left: 6\nLet's begin.")
        self.assertTrue(turns[1]["answer"].startswith("**Board:** \\_ \\_"))
        self.assertEqual(reasoning_of(turns[1]["text"], turns[1]["reasoning_prefilled"]),
                         "E is the last letter of CRANE.")
        # The next prompt carries a template-opened reasoning block back with its opening tag.
        self.assertEqual(messages_for(game, 2)[4],
                         {"role": "assistant", "content": "<think>" + turns[1]["text"]})
        self.assertEqual(messages_for(game, 2)[0], {"role": "system", "content": game["system"]})
        # The branched response: 28 replayed tokens, then the model's own.
        self.assertEqual(turns[6]["branch"], dict(parent=self.PARENT, turn=6, token_index=23, forced_tokens=28,
                                                  replacement="crate"))
        self.assertEqual(turns[6]["forced_prefix_tokens"], 28)
        self.assertEqual(turns[6]["text"], "Board: C R A N E\nWord: crate\nYou got it.")
        self.assertEqual([len(t["metrics"]) for t in turns], [99, 119, 71, 20, 75, 81, 41])
        self.assertEqual(turns[0]["metrics"][0]["top_candidates"], [dict(token_id=90, text="Z", probability=.1)])

    def test_the_page_opens_it_as_a_new_child_of_the_file(self):
        shown = self.open(self.PATH)
        game = shown["game"]
        self.assert_game_reads(game)
        recorded = json.loads(self.PATH.read_text(encoding="utf-8"))
        self.assertEqual(recorded["id"], self.ID)
        self.assertEqual(recorded["parent"], dict(id=self.PARENT, turns=6))
        # Opening never continues the file itself: the game on screen is its child.
        self.assertNotEqual(game["id"], self.ID)
        self.assertEqual(game["parent"], dict(id=self.ID, turns=7))
        # Responses 1-4 were reopened once already; every response is now some other load's.
        self.assertEqual([t.get("load_id") for t in recorded["turns"]], [None] * 4 + ["first"] * 3)
        self.assertEqual([t["load_id"] for t in game["turns"]], [None] * 7)
        self.assertEqual([t["recorded_load_id"] for t in game["turns"]], ["first"] * 7)

    def test_the_check_panel_reads_boards_and_reports_the_same_contradictions(self):
        shown = self.open(self.PATH)
        self.assertEqual(shown["check"].split("\n\n"), [
            "**Board:** `C R A N E` (5 letters) · **Letters guessed:** A, E, O, R, S",
            "**Words that fit the latest board:** 1 in `/usr/share/dict/words` (crane)",
            "**Revealed word:** CRATE",
            "**Contradictions:**\n" + "\n".join(f"- Response {n}: {m}" for n, m in self.PROBLEMS),
            "Responses without a readable board: 5.",
        ])

    def test_the_chat_notes_and_tokens_show_each_response(self):
        shown = self.open(self.PATH)
        chat = shown["chat"]
        self.assertEqual([m["content"] for m in chat if m["role"] == "user"], self.GUESSES)
        self.assertEqual([m["metadata"]["title"] for m in chat if "metadata" in m],
                         ["Reasoning · response 1", "Reasoning · response 2"])
        self.assertEqual(chat[1]["content"], "I will hold CRANE.")
        self.assertEqual(chat[-1]["content"], "Board: C R A N E\nWord: crate\nYou got it.")
        self.assertEqual(shown["picker"]["value"], 6)
        self.assertEqual(len(shown["picker"]["choices"]), 7)
        self.assertEqual(shown["note"], "Model `test/model` · temperature 0.7 · seed 46 · 13 sampled tokens · "
                                        "finish: stop · branched at token 24 with 28 replayed tokens")
        self.assertEqual(shown["raw"], "Board: C R A N E\nWord: crate\nYou got it.")
        self.assertEqual(len(shown["strip"]), 41)
        self.assertEqual(self.fn["select_response"](shown["game"], "owner", 3)[1],
                         "Model `test/model` · temperature 0.7 · seed 43 · 20 sampled tokens · finish: length")

    def test_resaved_by_todays_writer_it_reads_the_same(self):
        game = self.open(self.PATH)["game"]
        again = self.open(self.resaved(game))
        self.assert_game_reads(again["game"])
        self.assertEqual(again["game"]["parent"], dict(id=game["id"], turns=7))
        self.assertEqual(again["check"], self.open(self.PATH)["check"])


class BatchGameTests(SavedGameFixture):
    """batch-game-with-probes.json: written at 5c32174 (#157) by a batch, with probe answers on each response."""
    PATH = FIXTURES / "batch-game-with-probes.json"
    BOARDS = [cells("_____"), cells("____e"), cells("__a_e"), cells("__ote"), None]
    PROBE_WORDS = [["crane", "slate", "crane"], ["crate", "stone", None], ["grape", "crate", None],
                   ["crate", "crate", "slate"], []]
    PROBLEMS = [
        (4, "Position 3 showed A and now shows O."),
        (4, "O is on the board but was never guessed."),
        (4, "A was placed at 3 and is now at no position."),
        (5, "The revealed word CRATE has A at position 3, where the board showed O."),
    ]

    def assert_game_reads(self, game):
        self.assertEqual(game["system"], SYSTEM)
        self.assertEqual(game["trial"], dict(
            title="Probe pilot", file_sha256="710c50da5c51a920d755c600a44484a4af258bdb5d4b1f73cf71dac09f298614",
            batch="20260923-180000-probe-pilot", id="held-word", label="Fixed guesses, probed", seed=11))
        turns = game["turns"]
        self.assertEqual([t["guess"] for t in turns], [OPENING, "e", "a", "t", GIVE_UP])
        self.assertEqual([t["board"] for t in turns], self.BOARDS)
        self.assertEqual([t["revealed_word"] for t in turns], [None] * 4 + ["crate"])
        self.assertEqual([t["sampling"]["seed"] for t in turns], [11, 12, 13, 14, 15])
        self.assertEqual([[p["word"] for p in t.get("probes", [])] for t in turns], self.PROBE_WORDS)
        self.assertEqual([[p["seed"] for p in t.get("probes", [])] for t in turns][1], [12000, 12001, 12002])
        # Cut off by the probe's token limit: on its Word: line it is no word, past that line it is read.
        self.assertEqual((turns[2]["probes"][2]["text"], turns[2]["probes"][2]["finish_reason"]),
                         ("Word: stonewa", "length"))
        self.assertEqual((turns[3]["probes"][2]["text"], turns[3]["probes"][2]["finish_reason"]),
                         ("Word: slate\nW", "length"))

    def test_the_page_opens_it_with_its_trial_and_probe_answers(self):
        shown = self.open(self.PATH)
        game = shown["game"]
        self.assert_game_reads(game)
        self.assertEqual(game["parent"], dict(id="000000000000000000000016823aad21", turns=5))
        self.assertEqual(shown["check"].split("\n\n"), [
            "**Board:** `_ _ O T E` (5 letters) · **Letters guessed:** A, E, T",
            "**Words that fit the latest board:** 0 in `/usr/share/dict/words`",
            "**Revealed word:** CRATE",
            "**Contradictions:**\n" + "\n".join(f"- Response {n}: {m}" for n, m in self.PROBLEMS),
            "Responses without a readable board: 5.",
        ])
        notes = [self.fn["select_response"](game, "owner", index)[1] for index in range(5)]
        self.assertEqual([note.split(" · ", 5)[5:] for note in notes], [
            ["asked for the word 3 times: crane ×2, slate ×1"],
            ["asked for the word 3 times: crate ×1, stone ×1, `(unreadable)` ×1"],
            ["asked for the word 3 times: grape ×1, crate ×1, `(unreadable)` ×1"],
            ["asked for the word 3 times: crate ×2, slate ×1"],
            [],
        ])
        self.assertEqual(notes[0], "Model `test/model` · temperature 0.8 · seed 11 · 17 sampled tokens · "
                                   "finish: stop · asked for the word 3 times: crane ×2, slate ×1")

    def test_a_probe_word_is_read_from_its_text_not_from_the_file(self):
        recorded = json.loads(self.PATH.read_text(encoding="utf-8"))
        recorded["turns"][0]["probes"][0]["word"] = "zzzzz"
        path = self.directory / "edited.json"
        path.write_text(json.dumps(recorded), encoding="utf-8")
        self.assertEqual(load(path)["turns"][0]["probes"][0]["word"], "crane")

    def test_what_a_batch_summary_would_say_of_it(self):
        game = self.open(self.PATH)["game"]
        row, probes = summarize(game["trial"], game, "unfinished", 0, "test/model")
        self.assertEqual({key: row[key] for key in (
            "responses", "guesses", "unreadable_boards", "contradictions", "first_contradiction", "revealed_word",
            "revealed_at", "reveal_fits", "fitting_words", "probes", "probes_unreadable", "probe_fit_rate",
            "probe_reveal_rate", "sampled_tokens")}, dict(
            responses=5, guesses="e a t", unreadable_boards=1, contradictions=4, first_contradiction=4,
            revealed_word="crate", revealed_at=5, reveal_fits=False, fitting_words=0, probes=12,
            probes_unreadable=2, probe_fit_rate=0.7, probe_reveal_rate=0.4, sampled_tokens=80))
        # Every word after response 4 has A at position 3, where that board showed O.
        self.assertEqual([p["fits"] for p in probes],
                         [True, True, True, True, True, "", True, True, "", False, False, False])

    def test_resaved_by_todays_writer_it_reads_the_same(self):
        game = self.open(self.PATH)["game"]
        again = self.open(self.resaved(game))
        self.assert_game_reads(again["game"])
        self.assertEqual(again["check"], self.open(self.PATH)["check"])


class TrialFileTests(SavedGameFixture):
    """trials.json: a trial file as a user writes one, as 5c32174 (#157) documented and accepted it.

    ChatLab only reads this file, so there is no re-save to check.
    """
    PATH = FIXTURES / "trials.json"

    def test_the_batch_pane_reads_every_trial_with_its_settings_filled_in(self):
        data, note = self.fn["open_trials"](str(self.PATH))
        self.assertEqual(note, "**`Qwen3 0.6B, five seeds`** · 5 games · 4 with the reveal probe")
        self.assertEqual(data["title"], "Qwen3 0.6B, five seeds")
        self.assertEqual(data["file_sha256"], hashlib.sha256(self.PATH.read_bytes()).hexdigest())
        trials = {t["id"]: t for t in data["trials"]}
        self.assertEqual(list(trials), ["s1", "s2", "s1-greedy", "fixed", "custom-prompt"])
        default_probe = dict(samples=5, max_new_tokens=16, temperature=1.0)
        self.assertEqual(trials["s1"], dict(
            id="s1", label="s1", seed=1, system=SYSTEM, opening=OPENING, temperature=1.0, max_new_tokens=1024,
            guesser="frequency", max_guesses=26, probe=default_probe))
        self.assertIsNone(trials["s2"]["probe"])
        self.assertEqual((trials["s1-greedy"]["label"], trials["s1-greedy"]["temperature"],
                          trials["s1-greedy"]["probe"]), ("Seed 1, greedy", 0, default_probe))
        # A trial's probe replaces the default probe whole; what it leaves out is the page's default.
        self.assertEqual(trials["fixed"]["probe"], dict(samples=2, max_new_tokens=8, temperature=0.5))
        self.assertEqual((trials["fixed"]["guesser"], trials["fixed"]["max_guesses"]), (["e", "a", "crane"], 3))
        self.assertEqual({key: trials["custom-prompt"][key] for key in ("label", "system", "opening",
                                                                         "max_new_tokens")},
                         dict(label="custom-prompt", system="", opening="Pick a word and draw the board.",
                              max_new_tokens=4096))


if __name__ == "__main__":
    unittest.main()
