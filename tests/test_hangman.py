"""Hangman: board reading, consistency checks, and the page's model use."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import gradio as gr

from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from extensions.hangman.game import (
    SYSTEM, answer_of, check, finish_turn, fitting_words, guess_of, load, messages_for, new_game,
    read_board, read_word, reasoning_of, rewound, saved,
)
from extensions.hangman.page import build_page
from model_runtime import GENERATING

STOP = 0


def game_of(*exchanges):
    """A finished game from (guess, reply) pairs."""
    game = new_game(SYSTEM)
    for guess, reply in exchanges:
        game["turns"].append(finish_turn(dict(guess=guess, text=reply, metrics=[], finish_reason="stop")))
    return game


class BoardTests(unittest.TestCase):
    def test_reads_spaced_run_together_and_decorated_boards(self):
        self.assertEqual(read_board("Board: _ _ A _ E"), ["_", "_", "a", "_", "e"])
        self.assertEqual(read_board("**Board:** `__a_e`"), ["_", "_", "a", "_", "e"])
        self.assertEqual(read_board("Board: \\_ \\_ t"), ["_", "_", "t"])
        self.assertEqual(read_board("Board: __ __ a"), ["_", "_", "a"])
        # The last board is the one the reply ends on.
        self.assertEqual(read_board("Board: _ _\nno wait\nBoard: _ _ _"), ["_", "_", "_"])

    def test_unreadable_boards_are_none_rather_than_guessed(self):
        self.assertIsNone(read_board("Here is the board: five blanks"))
        self.assertIsNone(read_board("Board: _ _ ? _"))
        self.assertIsNone(read_board("No board here."))

    def test_revealed_word_and_guess_kinds(self):
        self.assertEqual(read_word("You win!\nWord: **Apple**"), "apple")
        self.assertIsNone(read_word("Board: a p p l e"))
        self.assertEqual(guess_of(" E. "), ("letter", "e"))
        self.assertEqual(guess_of("apple"), ("word", "apple"))
        self.assertEqual(guess_of("what was the word?"), ("other", None))

    def test_reasoning_is_split_from_the_answer(self):
        text = "I pick crane.</think>Board: _ _ _ _ _"
        self.assertEqual(answer_of(text, reasoning_prefilled=True), "Board: _ _ _ _ _")
        self.assertEqual(reasoning_of(text, reasoning_prefilled=True), "I pick crane.")
        self.assertEqual(reasoning_of("<think>still going"), "still going")
        self.assertEqual(answer_of("<think>still going"), "")


class CheckTests(unittest.TestCase):
    def test_a_consistent_game_reports_nothing(self):
        game = game_of(("start", "Board: _ _ _ _ _"), ("e", "Board: _ _ _ _ E"),
                       ("z", "Board: _ _ _ _ E"), ("a", "Board: _ _ A _ E"),
                       ("crane", "Board: C R A N E\nWord: crane"))
        self.assertEqual(check(game), [])

    def test_changed_length_moved_letter_and_unguessed_letter(self):
        game = game_of(("start", "Board: _ _ _ _ _"), ("e", "Board: _ _ _ _ E"),
                       ("a", "Board: _ _ A E _"), ("s", "Board: _ _ A E T"),
                       ("o", "Board: _ _ _ _"))
        problems = check(game)
        messages = [message for _, message in problems]
        self.assertIn((3, "E was placed at 5 and is now at 4."), problems)
        self.assertTrue(any("never guessed" in m and m.startswith("T") for m in messages))
        self.assertIn((5, "The board went from 5 letters to 4."), problems)

    def test_a_letter_called_absent_cannot_appear_later_or_in_the_word(self):
        game = game_of(("start", "Board: _ _ _"), ("t", "Board: _ _ _"),
                       ("a", "Board: _ A _"), ("x", "Board: _ A T"), ("reveal", "Word: cat"))
        problems = check(game)
        self.assertIn((4, "T was placed at no position and is now at 3."), problems)
        self.assertIn((5, "The revealed word CAT has T at 3; the board placed it at no position."), problems)

    def test_a_contradiction_is_reported_once_while_it_persists(self):
        game = game_of(("start", "Board: _ _ _"), ("t", "Board: _ _ _"),
                       ("a", "Board: _ A T"), ("b", "Board: _ A T"))
        self.assertEqual(len([p for p in check(game) if p[1].startswith("T was placed")]), 1)

    def test_a_letter_guessed_without_a_readable_board_is_placed_on_the_next(self):
        game = game_of(("start", "Board: _ _ _"), ("t", "Sorry, no T."),
                       ("a", "Board: _ A _"), ("c", "Board: C A T"))
        self.assertIn((4, "T was placed at no position and is now at 3."), check(game))

    def test_words_that_fit_respect_revealed_and_ruled_out_letters(self):
        game = game_of(("start", "Board: _ _ _"), ("a", "Board: _ A _"), ("t", "Board: _ A _"))
        self.assertEqual(fitting_words(game, ["cat", "cab", "bad", "ace", "can"]), ["cab", "bad", "can"])
        self.assertIsNone(fitting_words(game_of(("start", "no board"))))


class RecordTests(unittest.TestCase):
    def test_messages_carry_earlier_reasoning_back_in(self):
        game = game_of(("start", "Board: _ _ _"))
        game["turns"][0].update(text="word is cat</think>Board: _ _ _", reasoning_prefilled=True)
        game["turns"].append(dict(guess="a", text="", metrics=[]))
        messages = messages_for(game, 1)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "<think>word is cat</think>Board: _ _ _")
        game["system"] = "  "
        self.assertEqual(messages_for(game, 1)[0]["role"], "user")

    def test_saved_games_round_trip_and_rewind_points_at_the_parent(self):
        game = game_of(("start", "Board: _ _ _"), ("a", "Board: _ A _"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            path.write_text(saved(game))
            reopened = load(path)
            self.assertEqual(reopened["turns"][1]["board"], ["_", "a", "_"])
            path.write_text(json.dumps({"format": "other"}))
            with self.assertRaisesRegex(ValueError, "not a chatlab-hangman-1"):
                load(path)
        child = rewound(game, 1)
        self.assertEqual(len(child["turns"]), 1)
        self.assertEqual(child["parent"], dict(id=game["id"], turns=1))
        self.assertNotEqual(child["id"], game["id"])
        self.assertEqual(len(game["turns"]), 2)


class Manager:
    """Replays a scripted reply per call, one character per token."""
    loaded = True
    model_id = "test/model"
    load_id = "first"
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))

    def __init__(self, replies):
        self.replies = list(replies)
        self.busy = False
        self.calls = []

    def claim_generation(self):
        if self.busy:
            return GENERATING
        self.busy = True
        return None

    def release_generation(self):
        self.busy = False

    def _stop_token_ids(self):
        return {STOP}

    def hidden_token_ids(self):
        return {STOP}

    def encode_replacement(self, kept_ids, text, **kw):
        return [ord(c) for c in text]

    before_token = None

    def generate(self, messages, **options):
        self.calls.append(dict(messages=messages, **options))
        if self.before_token:
            self.before_token()
        forced = list(options.get("forced_ids", ()))
        reply = [ord(c) for c in self.replies.pop(0)] + [STOP]
        ids = forced + reply
        metrics = []
        for token in ids:
            metrics.append(dict(token_id=token, text=chr(token) if token else "", display_text=chr(token) if token else "⏹",
                                scored=False, top_candidates=[dict(token_id=ord("Z"), text="Z", probability=.1)]))
            yield SimpleNamespace(text="".join(chr(m["token_id"]) for m in metrics if m["token_id"]),
                                  metrics=[dict(m) for m in metrics], forced_prefix_tokens=len(forced),
                                  reasoning_prefilled=False)


class PageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name) / "hangman"
        self.manager = Manager(["Board: _ _ _", "Board: _ A _", "Board: _ _ Z"])
        context = ExtensionContext(ModelService(lambda: self.manager), TokenInspector(), self.data,
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}

    def test_game_streams_saves_and_releases_the_model(self):
        frames = list(self.fn["start_game"](SYSTEM, "Let's play.", "owner", 1.0, 7, 64))
        game = frames[-1][0]
        self.assertEqual(game["turns"][0]["board"], ["_", "_", "_"])
        self.assertEqual(game["turns"][0]["finish_reason"], "stop")
        self.assertEqual(game["turns"][0]["model_id"], "test/model")
        self.assertFalse(self.manager.busy)
        # The check panel waits for the finished response.
        self.assertEqual(frames[1][2], gr.skip())
        self.assertIn("Contradictions:** none", frames[-1][2])
        self.assertTrue(Path(frames[-1][8]).is_file())
        game = list(self.fn["play"](game, "a", "owner", 1.0, 7, 64))[-1][0]
        self.assertEqual(self.manager.calls[1]["seed"], 8)
        self.assertEqual([m["role"] for m in self.manager.calls[1]["messages"]],
                         ["system", "user", "assistant", "user"])
        saved_game = json.loads((self.data / f"{game['id']}.json").read_text())
        self.assertEqual([t["guess"] for t in saved_game["turns"]], ["Let's play.", "a"])

    def test_no_model_leaves_no_empty_turn_behind(self):
        self.manager.busy = True
        frames = []
        with self.assertRaisesRegex(gr.Error, "busy"):
            frames.extend(self.fn["start_game"](SYSTEM, "go", "other", 1.0, 7, 64))
        # Refused before the model was held, so nothing replaced the view.
        self.assertEqual(frames, [])

    def test_failure_before_any_token_restores_the_game(self):
        game = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))[-1][0]
        self.manager.replies = []  # generate raises IndexError before a token

        def run():
            return list(self.fn["play"](game, "a", "owner", 1.0, 7, 64))
        with self.assertRaises(gr.Error):
            run()
        self.assertFalse(self.manager.busy)

    def test_stop_before_the_first_token_keeps_no_empty_turn(self):
        game = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))[-1][0]
        self.manager.before_token = lambda: self.fn["cancel"]("owner")
        restored = list(self.fn["play"](game, "a", "owner", 1.0, 7, 64))[-1][0]
        self.assertEqual([t["guess"] for t in restored["turns"]], ["go"])
        saved_game = json.loads((self.data / f"{game['id']}.json").read_text())
        self.assertEqual([t["guess"] for t in saved_game["turns"]], ["go"])
        self.assertFalse(self.manager.busy)
        # Stopped before the model chose a word, the opening leaves no game to continue.
        opened = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))[-1][0]
        self.assertEqual(opened["turns"], [])
        self.assertFalse((self.data / f"{opened['id']}.json").exists())
        with self.assertRaisesRegex(gr.Error, "Start a new game"):
            list(self.fn["play"](opened, "a", "owner", 1.0, 7, 64))

    def test_a_game_too_large_to_open_again_is_not_saved(self):
        with mock.patch("extensions.hangman.page.MAX_FILE_BYTES", 10), mock.patch("gradio.Warning") as warned:
            frames = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))
        self.assertIn("not saved", warned.call_args.args[0])
        self.assertEqual(frames[-1][8], gr.skip())
        self.assertEqual(list(self.data.glob("*.json")), [])

    def test_branch_replays_kept_tokens_then_the_alternative_in_a_new_game(self):
        frames = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))
        game, payload = frames[-1][0], frames[-1][5]
        game = list(self.fn["play"](game, "a", "owner", 1.0, 7, 64))[-1][0]
        payload = self.fn["select_response"](game, "owner", 1)[2]
        action = json.dumps(dict(kind="candidate", index=0, selection=dict(view_id=[game["id"], 1], index=9)))
        branched = list(self.fn["branch"](game, "owner", payload, action, 1.0, 7, 64))[-1][0]
        call = self.manager.calls[-1]
        self.assertEqual(call["forced_ids"], [ord(c) for c in "Board: _ "] + [ord("Z")])
        self.assertEqual(call["seed"], 8)
        self.assertEqual(branched["parent"], dict(id=game["id"], turns=1))
        self.assertEqual(branched["turns"][1]["text"], "Board: _ ZBoard: _ _ Z")
        self.assertEqual(branched["turns"][1]["branch"]["token_index"], 9)
        self.assertEqual(len(game["turns"]), 2)

    def test_branch_refuses_tokens_from_another_load(self):
        game = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))[-1][0]
        payload = self.fn["select_response"](game, "owner", 0)[2]
        self.manager.load_id = "second"
        action = json.dumps(dict(kind="text", text="x", selection=dict(view_id=[game["id"], 0], index=2)))
        with self.assertRaisesRegex(gr.Error, "different model load"):
            list(self.fn["branch"](game, "owner", payload, action, 1.0, 7, 64))
        self.assertFalse(self.manager.busy)

    def test_stale_selection_is_refused(self):
        game = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))[-1][0]
        payload = self.fn["select_response"](game, "owner", 0)[2]
        action = json.dumps(dict(kind="text", text="x", selection=dict(view_id=["elsewhere", 0], index=2)))
        with self.assertRaises(gr.Error):
            list(self.fn["branch"](game, "owner", payload, action, 1.0, 7, 64))

    def test_rewind_and_open_saved_make_copies(self):
        frames = list(self.fn["start_game"](SYSTEM, "go", "owner", 1.0, 7, 64))
        game = list(self.fn["play"](frames[-1][0], "a", "owner", 1.0, 7, 64))[-1][0]
        child = self.fn["rewind_to"](game, "owner", 0)[0]
        self.assertEqual(len(child["turns"]), 1)
        opened = self.fn["open_saved"](str(self.data / f"{game['id']}.json"), "owner")[0]
        self.assertEqual(opened["parent"]["id"], game["id"])
        self.assertEqual(len(opened["turns"]), 2)


if __name__ == "__main__":
    unittest.main()
