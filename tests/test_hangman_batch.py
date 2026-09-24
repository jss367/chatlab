"""Hangman batches: trial files, the guesser, the reveal probe and the files a batch writes."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr

from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from extensions.hangman import batch
from extensions.hangman.batch import BatchControl, next_guess, read_trials, run_trials
from extensions.hangman.game import GIVE_UP, SYSTEM, finish_turn, load, new_game, word_problems
from extensions.hangman.page import build_page, turn_note
from model_runtime import GENERATING

STOP = 0
WORDS = ("crane", "crate", "grape", "slate", "stone")


def game_of(*exchanges):
    game = new_game(SYSTEM)
    for guess, reply in exchanges:
        game["turns"].append(finish_turn(dict(guess=guess, text=reply, metrics=[], finish_reason="stop")))
    return game


class Host:
    """A model that answers hangman, one character per token.

    ``word`` is the word it holds, and every board it draws is that word's.
    Without one it holds nothing: each board is five blanks with whatever it
    likes revealed, and asked for the word it names whichever ``names`` gives.
    """
    loaded = True
    model_id = "test/model"
    load_id = "first"
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))

    def __init__(self, word=None, names=(), boards=(), left=6):
        self.word, self.names, self.boards, self.left = word, list(names), list(boards), left
        self.busy = False
        self.calls = []
        self.during = None
        self.fail_on = None

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

    def reply(self, messages, prefill):
        if prefill:
            return " " + (self.word or self.names.pop(0))
        question = messages[-1]["content"]
        if question == GIVE_UP:
            return f"Word: {self.word or self.names.pop(0)}"
        if self.word is None:
            return f"Board: {self.boards.pop(0)}"
        guessed = {m["content"] for m in messages if m["role"] == "user" and len(m["content"]) == 1}
        board = [c.upper() if c in guessed else "_" for c in self.word]
        wrong = len(guessed - set(self.word))
        text = f"Board: {' '.join(board)}\nGuessed: {' '.join(sorted(guessed)).upper()}\n" \
               f"Wrong guesses left: {self.left - wrong}"
        return text + (f"\nWord: {self.word}" if "_" not in board else "")

    def generate(self, messages, **options):
        self.calls.append(dict(messages=messages, **options))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("the model ran out of memory")
        prefill = options.get("answer_prefill", "")
        ids = [ord(c) for c in prefill + self.reply(messages, prefill)] + [STOP]
        metrics = []
        for position, token in enumerate(ids, 1):
            if self.during:
                self.during(len(self.calls), position)
            metrics.append(dict(token_id=token, text=chr(token) if token else "",
                                display_text=chr(token) if token else "⏹", segment="response", position=position,
                                scored=False, top_candidates=[]))
            yield SimpleNamespace(text="".join(chr(m["token_id"]) for m in metrics if m["token_id"]),
                                  metrics=[dict(m) for m in metrics], forced_prefix_tokens=len(prefill),
                                  reasoning_prefilled=False)


def trial_file(directory, trials, **defaults):
    path = Path(directory) / "trials.json"
    path.write_text(json.dumps(dict(format="chatlab-hangman-trials-1", title="Pilot", defaults=defaults,
                                    trials=trials)))
    return path


class Fixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for target in ("extensions.hangman.game.dictionary", "extensions.hangman.batch.dictionary"):
            patcher = mock.patch(target, return_value=WORDS)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_batch(self, host, trials, control=None, **defaults):
        data = read_trials(trial_file(self.root, trials, **defaults))
        control = control or BatchControl()
        frames = list(run_trials(data, ModelService(lambda: host), self.root, control, source="trials.json"))
        return frames[-1]

    def files(self, directory):
        def read(name):
            return list(csv.DictReader((directory / name).read_text().splitlines()))
        return read("summary.csv"), read("probes.csv"), json.loads((directory / "batch.json").read_text())


class TrialFileTests(Fixture):
    def test_trials_take_the_file_defaults_then_their_own(self):
        data = read_trials(trial_file(self.root, [dict(id="a", seed=1), dict(id="b", seed=2, temperature=0,
                                                                                label="Greedy")],
                                      temperature=.7, probe=dict(samples=3)))
        first, second = data["trials"]
        self.assertEqual((first["temperature"], first["label"], first["system"]), (.7, "a", SYSTEM))
        self.assertEqual((second["temperature"], second["label"]), (0, "Greedy"))
        self.assertEqual(first["probe"], dict(samples=3, max_new_tokens=16, temperature=1.0))
        self.assertEqual(first["guesser"], "frequency")
        self.assertEqual(len(data["file_sha256"]), 64)

    def test_malformed_trials_are_refused_naming_the_trial(self):
        for trials, defaults, message in (
                ([dict(id="a", seed=1), dict(id="a", seed=2)], {}, "unique"),
                ([dict(id="a", seed=-1)], {}, "'a': seed"),
                ([dict(id="a", seed=True)], {}, "seed"),
                ([dict(id="a", seed=1, guesser="alphabet")], {}, "guesser"),
                ([dict(id="a", seed=1, guesser=["e", ""])], {}, "guesser"),
                ([dict(id="a", seed=1, probe=dict(samples=0))], {}, "samples"),
                ([dict(id="a", seed=1, probe=dict(question="?"))], {}, "probe may set only"),
                ([dict(id="a", seed=1, colour="red")], {}, "may set only"),
                ([dict(id="a", seed=1)], dict(seed=4), "Defaults"),
                ([dict(id="a", seed=1, temperature=float("nan"))], {}, "temperature"),
                ([], {}, "1–2000")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                read_trials(trial_file(self.root, trials, **defaults))
        saved = self.root / "game.json"
        saved.write_text(json.dumps(dict(format="chatlab-hangman-1")))
        with self.assertRaisesRegex(ValueError, "not a saved game"):
            read_trials(saved)


class GuesserTests(Fixture):
    def test_frequency_guesser_picks_the_letter_most_fitting_words_share(self):
        # Every fitting word has R, A and E; C and T tie behind them.
        game = game_of(("go", "Board: _ _ _ _ _"), ("s", "Board: _ _ _ _ _"), ("o", "Board: _ _ _ _ _"))
        self.assertEqual(next_guess(game, "frequency", 0), "a")
        game = game_of(("go", "Board: _ _ _ _ _"), ("a", "Board: _ _ A _ _"), ("e", "Board: _ _ A _ E"),
                       ("r", "Board: _ R A _ E"))
        self.assertEqual(next_guess(game, "frequency", 0), "c")

    def test_frequency_guesser_falls_back_when_no_word_fits(self):
        game = game_of(("go", "Board: _ _ Q"), ("e", "Board: _ _ Q"))
        self.assertEqual(next_guess(game, "frequency", 0), "t")
        self.assertEqual(next_guess(game, ["x", "y"], 1), "y")
        self.assertIsNone(next_guess(game, ["x", "y"], 2))

    def test_a_word_is_held_to_the_boards_and_the_first_reveal(self):
        game = game_of(("go", "Board: _ _ _ _ _"), ("a", "Board: _ _ A _ _"), ("t", "Board: _ _ A _ _"))
        self.assertEqual(word_problems(game, "crane"), [])
        self.assertTrue(word_problems(game, "crate"))
        self.assertTrue(word_problems(game, "cranes"))
        # A cell a later board hides again still holds the word to what it showed.
        game = game_of(("go", "Board: _ _ _ _ _"), ("x", "Board: A _ _ _ _"), ("y", "Board: _ _ _ _ _"))
        self.assertIn("The revealed word STONE has S at position 1, where the board showed A.",
                      word_problems(game, "stone"))
        game = game_of(("go", "Board: _ _ _ _ _"), ("?", "Word: crane"))
        self.assertIn("The game revealed CRANE at response 2; this is GRAPE.", word_problems(game, "grape"))


class BatchTests(Fixture):
    def test_an_honest_host_plays_to_a_solved_board_and_probes_name_its_word(self):
        host = Host("crane")
        done, total, rows, directory, _ = self.run_batch(
            host, [dict(id="one", seed=5)], guesser=["c", "r", "a", "n", "e"], probe=dict(samples=2))
        self.assertEqual((done, total), (1, 1))
        row = rows[0]
        self.assertEqual((row["outcome"], row["responses"], row["contradictions"]), ("solved", 6, 0))
        self.assertEqual((row["revealed_word"], row["revealed_at"], row["reveal_fits"]), ("crane", 6, True))
        # Asked after every response but the one that revealed the word.
        self.assertEqual((row["probes"], row["probe_fit_rate"], row["probe_reveal_rate"]), (10, 1.0, 1.0))
        self.assertEqual(row["guesses"], "c r a n e")
        games = [c for c in host.calls if not c["answer_prefill"]]
        self.assertEqual([c["seed"] for c in games], [5, 6, 7, 8, 9, 10])
        probes = [c for c in host.calls if c["answer_prefill"]]
        self.assertEqual([c["seed"] for c in probes[:4]], [5000, 5001, 6000, 6001])
        self.assertEqual({c["answer_prefill"] for c in probes}, {"Word:"})
        self.assertEqual(probes[0]["messages"][-1], dict(role="user", content=GIVE_UP))
        self.assertEqual(probes[0]["max_new_tokens"], 16)
        self.assertFalse(host.busy)

        rows, probe_rows, manifest = self.files(directory)
        self.assertEqual(rows[0]["outcome"], "solved")
        self.assertEqual(len(probe_rows), 10)
        self.assertEqual(probe_rows[0], dict(trial_id="one", response="1", sample="0", seed="5000", word="crane",
                                             fits="True", matches_reveal="True"))
        self.assertEqual((manifest["status"], manifest["model_id"], manifest["source"]),
                         ("finished", "test/model", "trials.json"))
        game = load(directory / rows[0]["game_file"])
        self.assertEqual(game["trial"]["id"], "one")
        self.assertEqual(game["trial"]["title"], "Pilot")
        self.assertEqual(game["turns"][0]["probes"][0]["word"], "crane")
        self.assertIn("asked for the word 2 times: crane ×2", turn_note(game["turns"][0]))

    def test_a_host_holding_nothing_contradicts_itself_and_its_probes_scatter(self):
        host = Host(boards=["_ _ _ _ _", "_ _ _ _ E", "_ _ A _ E", "_ R A _ E"],
                    names=["crane", "slate", "grape", "stone", "crate", "slate", "stone", "stone", "stone"])
        rows = self.run_batch(host, [dict(id="liar", seed=1)], guesser=["e", "a", "t"],
                              probe=dict(samples=2))[2]
        row = rows[0]
        # Out of guesses, the batch asks for the word, and R was never guessed.
        self.assertEqual((row["outcome"], row["revealed_word"], row["responses"]), ("unfinished", "stone", 5))
        self.assertEqual(row["first_contradiction"], 4)
        self.assertFalse(row["reveal_fits"])
        self.assertLess(row["probe_reveal_rate"], 1)
        self.assertLess(row["probe_fit_rate"], 1)

    def test_words_are_not_judged_before_any_board_is_readable(self):
        host = Host(boards=["five blanks", "_ _ _ _ E"], names=["crane", "stone", "grape", "slate", "stone"])
        row = self.run_batch(host, [dict(id="a", seed=1)], guesser=["e"], probe=dict(samples=2))[2][0]
        # Only the two probes after the first readable board are judged: GRAPE
        # fits _ _ _ _ E, SLATE fits it too, and the batch's own ask names STONE.
        self.assertEqual((row["probes"], row["probe_fit_rate"], row["reveal_fits"]), (4, 1.0, True))
        host = Host(boards=["five blanks", "?"], names=["crane", "stone", "grape", "slate", "stone"])
        row = self.run_batch(host, [dict(id="a", seed=1)], guesser=["e"], probe=dict(samples=2))[2][0]
        self.assertEqual((row["revealed_word"], row["reveal_fits"], row["probe_fit_rate"]), ("stone", "", ""))

    def test_a_lost_game_asks_for_the_word(self):
        host = Host("crane", left=2)
        rows = self.run_batch(host, [dict(id="lost", seed=1)], guesser=["x", "y", "z"])[2]
        self.assertEqual((rows[0]["outcome"], rows[0]["responses"], rows[0]["revealed_word"]), ("lost", 4, "crane"))
        self.assertEqual(host.calls[-1]["messages"][-1]["content"], GIVE_UP)
        self.assertEqual(rows[0]["guesses"], "x y")

    def test_a_failed_game_is_recorded_and_the_batch_moves_on(self):
        host = Host("crane")
        host.fail_on = 2
        rows = self.run_batch(host, [dict(id="bad", seed=1), dict(id="good", seed=1)], guesser=["c"])[2]
        self.assertEqual([r["outcome"] for r in rows], ["error", "unfinished"])
        self.assertEqual(rows[0]["responses"], 1)
        self.assertIn("out of memory", rows[0]["detail"])
        self.assertFalse(host.busy)

    def test_stopping_ends_the_game_playing_and_starts_no_more(self):
        host, control = Host("crane"), BatchControl()
        host.during = lambda call, position: call == 2 and position == 3 and control.request_stop()
        done, total, rows, directory, _ = self.run_batch(host, [dict(id="a", seed=1), dict(id="b", seed=1)],
                                                         control, guesser=["c", "r"])
        self.assertEqual((done, total, [r["outcome"] for r in rows]), (1, 2, ["stopped"]))
        game = load(directory / rows[0]["game_file"])
        self.assertEqual(game["turns"][-1]["finish_reason"], "stopped")
        self.assertEqual(self.files(directory)[2]["status"], "stopped")
        self.assertFalse(control.running)
        self.assertFalse(host.busy)

    def test_a_game_that_cannot_be_saved_fails_the_batch(self):
        host = Host("crane")

        def write(path, text, **options):
            if path.parent.name == "games":
                raise OSError("disk full")
            return batch_write(path, text, **options)
        batch_write = batch.write_private_text
        with mock.patch("extensions.hangman.batch.write_private_text", side_effect=write):
            data = read_trials(trial_file(self.root, [dict(id="a", seed=1)]))
            with self.assertRaisesRegex(OSError, "could not be saved: disk full"):
                list(run_trials(data, ModelService(lambda: host), self.root, BatchControl()))
        directory = next((self.root / "batches").iterdir())
        rows, _, manifest = self.files(directory)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual((rows[0]["outcome"], rows[0]["game_file"]), ("unsaved", ""))
        self.assertFalse(host.busy)

    def test_a_malformed_probe_in_a_saved_game_is_refused(self):
        game = game_of(("go", "Board: _ _ _"))
        game["turns"][0]["probes"] = [dict(text="Word: cat")]
        path = self.root / "game.json"
        path.write_text(json.dumps(game))
        with self.assertRaisesRegex(ValueError, "malformed"):
            load(path)


class PageBatchTests(Fixture):
    def setUp(self):
        super().setUp()
        self.host = Host("crane")
        context = ExtensionContext(ModelService(lambda: self.host), TokenInspector(), self.root / "hangman",
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}

    def test_the_pane_plays_a_file_and_offers_its_summary(self):
        path = trial_file(self.root, [dict(id="a", seed=1, label="First")], guesser=["c", "r", "a", "n", "e"])
        data, note = self.fn["open_trials"](str(path))
        self.assertIn("1 game", note)
        frames = list(self.fn["run_batch"](data, BatchControl(), str(path)))
        status, results, files, run, stop = frames[-1]
        self.assertIn("Played 1 game", status)
        self.assertEqual(results["value"], [["First", "solved", 6, 0, "crane", ""]])
        self.assertEqual({Path(f).name for f in files["value"]}, {"summary.csv", "probes.csv", "batch.json"})
        self.assertEqual((run["visible"], stop["visible"]), (True, False))
        # Every frame sends the table whole rather than skipping it.
        self.assertTrue(all(isinstance(frame[1], dict) for frame in frames))

    def test_a_busy_model_leaves_the_pane_alone(self):
        path = trial_file(self.root, [dict(id="a", seed=1)])
        data, _ = self.fn["open_trials"](str(path))
        self.host.busy = True
        with mock.patch("gradio.Warning") as warned:
            frames = list(self.fn["run_batch"](data, BatchControl(), str(path)))
        self.assertIn("busy", warned.call_args.args[0])
        self.assertEqual(frames, [(gr.skip(),) * 5])

    def test_a_bad_trial_file_is_refused(self):
        path = self.root / "bad.json"
        path.write_text("{")
        with self.assertRaisesRegex(gr.Error, "not a hangman trial file"):
            self.fn["open_trials"](str(path))


if __name__ == "__main__":
    unittest.main()
