"""Twenty questions: reading replies, consistency checks, J space and the page's model use."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.twenty_questions import jspace
from chatlab.extensions.twenty_questions.game import (
    SYSTEM, check, finish_turn, load, messages_for, new_game, probe_thing, read_left, read_thing, read_verdict,
    recheck_summary, saved,
)
from chatlab.extensions.twenty_questions.page import build_page, check_text, watch_html, watch_list
from hangman_support import STOP, CharacterModel
from ui_support import handlers_by_name

PALETTE = TokenInspector().palette


def game_of(*exchanges):
    """A finished game from (question, reply) pairs."""
    game = new_game(SYSTEM)
    for question, reply in exchanges:
        game["turns"].append(finish_turn(dict(question=question, text=reply, metrics=[], finish_reason="stop")))
    return game


class ReadingTests(unittest.TestCase):
    def test_reads_verdicts_counts_and_things(self):
        self.assertEqual(read_verdict("**Answer:** Yes.\nQuestions left: 19"), "yes")
        self.assertEqual(read_verdict("Answer: no"), "no")
        self.assertEqual(read_verdict("Answer: Sometimes, it depends."), "sometimes")
        self.assertIsNone(read_verdict("Answer: purple"))
        self.assertIsNone(read_verdict("I would say yes."))
        self.assertEqual(read_left("Answer: yes\nQuestions left: 17"), 17)
        self.assertEqual(read_thing("You got it!\nThing: **A Giraffe**."), "giraffe")
        self.assertEqual(read_thing("Thing: the Eiffel Tower"), "eiffel tower")
        self.assertIsNone(read_thing("No thing yet."))

    def test_a_probe_reads_only_its_first_finished_line(self):
        self.assertEqual(probe_thing(dict(text="Thing: a cat\nThing: dog", finish_reason="stop")), "cat")
        self.assertIsNone(probe_thing(dict(text="Thing: a ca", finish_reason="length")))


class CheckTests(unittest.TestCase):
    def test_a_consistent_game_reports_nothing(self):
        game = game_of(("start", "Ready!\nQuestions left: 20"), ("Is it alive?", "Answer: yes\nQuestions left: 19"),
                       ("Is it a cat?", "Answer: yes\nQuestions left: 18\nThing: cat"))
        self.assertEqual(check(game), [])

    def test_count_repeat_and_renamed_thing(self):
        game = game_of(("start", "Questions left: 20"), ("Is it alive?", "Answer: yes\nQuestions left: 19"),
                       ("Is it big?", "Answer: no\nQuestions left: 19"),
                       ("is it ALIVE", "Answer: No\nQuestions left: 17"),
                       ("cat?", "Thing: cat"), ("so?", "Thing: dog"))
        problems = check(game)
        self.assertIn((3, "Questions left went from 19 at response 2 to 19; 18 would follow."), problems)
        self.assertIn((4, "Response 2 answered YES to the same question; this one answers NO."), problems)
        self.assertIn((6, "The thing named at response 5 was CAT; this one is DOG."), problems)

    def test_the_panel_counts_answers_and_shows_a_recheck(self):
        game = game_of(("start", "Questions left: 20"), ("Is it alive?", "Answer: yes\nQuestions left: 19"),
                       ("Is it big?", "Answer: no\nQuestions left: 18"))
        game["recheck"] = dict(thing="whale", model_id="test/model", rows=[
            dict(turn=1, question="Is it alive?", game_verdict="yes", verdict="yes", text="Answer: yes"),
            dict(turn=2, question="Is it big?", game_verdict="no", verdict="yes", text="Answer: yes")])
        self.assertEqual(recheck_summary(game["recheck"])[:2], (1, 2))
        text = check_text(game)
        self.assertIn("yes 1 · no 1", text)
        self.assertIn("1 of 2 answers agree", text)
        self.assertIn("the game said NO, asked about whale the model says YES", text)


class FakeSession:
    """A lens over a five-token vocabulary: block b ranks token (b + i) % 5 at place i."""
    load_id = "first"
    position_limit = 100

    def __init__(self, layers=(3, 4)):
        self.layers = list(layers)
        self.calls = []

    def jacobian_lens(self):
        return dict(name="test lens", n_prompts=1, layers=self.layers) if self.layers else None

    def lens_ranks(self, ids, blocks, positions):
        self.calls.append(("ranks", list(blocks), list(positions)))
        out = np.empty((len(blocks), len(positions), 5), dtype=np.int32)
        for row, block in enumerate(blocks):
            for token in range(5):
                out[row, :, (block + token) % 5] = token + 1
        return out

    def lens_log_probs(self, ids, targets, blocks, positions):
        self.calls.append(("log_probs", list(blocks), list(positions)))
        return np.full((len(blocks), len(positions), len(targets)), -1.0)

    def decode(self, ids):
        return "".join(chr(97 + i) for i in ids)

    def encode(self, text):
        return [len(text.strip()) % 5]


class JSpaceTests(unittest.TestCase):
    def test_the_verdict_is_read_one_position_before_its_first_token(self):
        text = "Answer: yes"
        turn = dict(text=text, metrics=[dict(token_id=ord(c), text=c) for c in text])
        self.assertEqual(jspace.positions_of(turn, 10), [("Starting the reply", 9), ("Choosing the answer", 17)])
        # A verdict in the reasoning is not the answer's.
        reasoning = "<think>Answer: no</think>Hmm."
        turn = dict(text=reasoning, metrics=[dict(token_id=0, text=c) for c in reasoning])
        self.assertEqual(jspace.positions_of(turn, 10), [("Starting the reply", 9)])
        # Token texts, not the streamed text, place it: a special token's text shifts every offset after it.
        pieces = ["<|x|>", "\n", "\ufffd", "\ufffd", "\n", "Answer", ":", " ", "no"]
        turn = dict(text="\n🙂\nAnswer: no", metrics=[dict(token_id=0, text=p) for p in pieces])
        self.assertEqual(jspace.positions_of(turn, 10)[1], ("Choosing the answer", 17))

    def test_reads_top_tokens_per_block_and_the_watch_list(self):
        session = FakeSession()
        reading = jspace.read(session, list(range(12)), [("Start", 4), ("Answer", 9)], ["cat", "zebra"], top=2)
        self.assertEqual(reading["blocks"], [3, 4])
        first = reading["positions"][0]
        self.assertEqual(first["label"], "Start")
        self.assertEqual([t["token_id"] for t in first["blocks"][0]["top"]], [3, 4])
        self.assertEqual([t["text"] for t in first["blocks"][1]["top"]], ["e", "a"])
        self.assertEqual(reading["watch"], {"cat": [-1.0, -1.0], "zebra": [-1.0, -1.0]})
        html = watch_html(dict(turns=[dict(question="Is it alive?", jspace=reading)]), PALETTE)
        self.assertIn("zebra", html)
        self.assertIn("-1.0", html)

    def test_no_lens_or_too_long_a_game_says_what_to_do(self):
        with self.assertRaisesRegex(ValueError, "Import a Jacobian lens"):
            jspace.read(FakeSession(layers=()), [1, 2], [("Start", 1)])
        with self.assertRaisesRegex(ValueError, "past the 100"):
            jspace.read(FakeSession(), list(range(200)), [("Start", 150)])

    def test_the_watch_list_splits_on_commas_and_lines(self):
        self.assertEqual(watch_list("cat, dog\nEiffel Tower,, cat"), ["cat", "dog", "Eiffel Tower"])


class Manager(CharacterModel):
    """Replays a scripted reply per call, one character per token, and has no lens."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.busy = False
        self.calls = []

    def jacobian_lens_summary(self):
        return None

    def recall_jacobian_lens(self):
        return None, None

    def generate(self, messages, **options):
        self.calls.append(dict(messages=messages, **options))
        prefill = options.get("answer_prefill", "")
        forced = list(options.get("forced_ids", ()))
        reply = [ord(c) for c in prefill + self.replies.pop(0)] + [STOP]
        metrics = []
        for position, token in enumerate(forced + reply, 1):
            metrics.append(dict(token_id=token, text=chr(token) if token else "", display_text=chr(token) if token else "⏹",
                                segment="response", position=position, scored=False,
                                top_candidates=[dict(token_id=ord("Z"), text="Z", probability=.1)]))
            yield SimpleNamespace(text="".join(chr(m["token_id"]) for m in metrics if m["token_id"]),
                                  metrics=[dict(m) for m in metrics], forced_prefix_tokens=len(forced),
                                  reasoning_prefilled=False, prompt_ids=(1, 2, 3))


class LensManager(Manager):
    """A Manager with a lens over the 128 ASCII tokens, fitted at blocks 1 and 2.

    At position p, block b ranks token (ids[p] + b + k) % 128 at place k + 1,
    and a target's log probability falls by a tenth per place.
    """
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)),
                                encode=lambda text, **kw: [ord(c) % 128 for c in text])

    def _prompt_token_ids(self, messages, tools=None):
        return [1, 2, 3], False

    def jacobian_lens_summary(self):
        return dict(name="test lens", n_prompts=3, layers=[1, 2])

    def lens_log_probs(self, ids, targets, blocks, positions, *, load_id=None, cancelled=None, held=False,
                       _ranks=False):
        places = np.array([[[(t - ids[p] - b) % 128 for t in range(128)] for p in positions] for b in blocks])
        if _ranks:
            return (places + 1).astype(np.int32)
        groups = [[t] if isinstance(t, int) else list(t) for t in targets]
        return np.stack([np.mean(-0.5 - 0.1 * places[:, :, g], axis=-1) for g in groups], axis=-1)


class PageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name) / "twenty_questions"
        self.manager = Manager(["Ready.\nQuestions left: 20", "Answer: yes\nQuestions left: 19"])
        context = ExtensionContext(ModelService(lambda: self.manager), TokenInspector(), self.data,
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = handlers_by_name(demo)

    def play(self):
        game = list(self.fn["start_game"](SYSTEM, "Let's play.", "owner", 1.0, 7, 64, True, "cat"))[-1][0]
        return list(self.fn["play"](game, "Is it alive?", "owner", 1.0, 7, 64, True, "cat"))[-1]

    def test_game_streams_saves_releases_and_explains_a_missing_lens(self):
        last = self.play()
        game = last[0]
        self.assertEqual([t["verdict"] for t in game["turns"]], [None, "yes"])
        self.assertEqual(self.manager.calls[1]["seed"], 8)
        self.assertFalse(self.manager.busy)
        self.assertIn("Import a Jacobian lens", game["turns"][1]["jspace_error"])
        self.assertIn("Import a Jacobian lens", last[9])
        saved_game = json.loads((self.data / f"{game['id']}.json").read_text())
        self.assertEqual([t["question"] for t in saved_game["turns"]], ["Let's play.", "Is it alive?"])
        self.assertEqual(Path(last[10]).read_text(), (self.data / f"{game['id']}.json").read_text())

    def test_asking_what_it_is_stays_out_of_the_game(self):
        game = self.play()[0]
        self.manager.replies += [" a cat", " dog"]
        asked = self.fn["ask_what"](game, "owner", 2, 1.0)[0]
        self.assertEqual(len(asked["turns"]), 2)
        self.assertEqual([p["thing"] for p in asked["turns"][-1]["probes"]], ["cat", "dog"])
        self.assertEqual(self.manager.calls[-1]["answer_prefill"], "Thing:")
        self.assertEqual(self.manager.calls[-1]["messages"][-1]["content"], "I give up. What was it?")
        self.assertFalse(self.manager.busy)

    def test_recheck_asks_each_question_fresh_and_a_new_question_clears_it(self):
        game = self.play()[0]
        self.manager.replies.append(" no")
        checked = self.fn["recheck_game"](game, "owner", "rock")[0]
        self.assertEqual(checked["recheck"]["rows"][0]["verdict"], "no")
        self.assertEqual(self.manager.calls[-1]["temperature"], 0.0)
        self.assertEqual([m["role"] for m in self.manager.calls[-1]["messages"]], ["user"])
        self.assertIn("rock", self.manager.calls[-1]["messages"][0]["content"])
        reopened = load(self.data / f"{checked['id']}.json")
        self.assertEqual(reopened["recheck"]["thing"], "rock")
        self.manager.replies.append("Answer: no\nQuestions left: 18")
        later = list(self.fn["play"](checked, "Is it big?", "owner", 1.0, 7, 64, False, ""))[-1][0]
        self.assertNotIn("recheck", later)

    def test_saved_games_reopen_as_copies_and_rewind_drops_the_recheck(self):
        game = self.play()[0]
        game["recheck"] = dict(thing="cat", rows=[])
        shown = self.fn["rewind_to"](game, "owner", 0)[0]
        self.assertEqual(len(shown["turns"]), 1)
        self.assertNotIn("recheck", shown)
        path = self.data / f"{game['id']}.json"
        opened = self.fn["open_saved"](str(path), "owner")[0]
        self.assertNotEqual(opened["id"], game["id"])
        self.assertEqual(opened["parent"]["id"], game["id"])
        self.assertIsNone(opened["turns"][1]["load_id"])

    def test_a_malformed_save_is_refused(self):
        path = self.data / "bad.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        game = json.loads(saved(new_game(SYSTEM)))
        game["turns"] = [dict(question="q", text="t", jspace=dict(positions=[dict(blocks=[dict(block="x")])]))]
        path.write_text(json.dumps(game))
        with self.assertRaises(gr.Error):
            self.fn["open_saved"](str(path), "owner")

    def test_with_a_lens_every_reply_shows_its_j_space_and_the_watch_list(self):
        self.manager = LensManager(self.manager.replies)
        last = self.play()
        reply = last[0]["turns"][1]
        reading = reply["jspace"]
        self.assertEqual([p["label"] for p in reading["positions"]], ["Starting the reply", "Choosing the answer"])
        # Three prompt tokens, then "Answer: ": the verdict is reply token 8, read after the space before it.
        self.assertEqual([p["position"] for p in reading["positions"]], [2, 10])
        self.assertEqual(reading["positions"][1]["token"], " ")
        top = reading["positions"][1]["blocks"][0]
        self.assertEqual(top["block"], 1)
        self.assertEqual([t["text"] for t in top["top"][:3]], ["!", '"', "#"])
        self.assertAlmostEqual(top["top"][0]["log_prob"], -0.5)
        self.assertEqual(list(reading["watch"]), ["cat"])
        self.assertIn("Choosing the answer", last[9])
        self.assertIn("test lens", last[9])
        self.assertIn("<th>cat</th>", last[3])
        # Read again with another watch list, under the same load.
        again = self.fn["reread_game"](last[0], "owner", "dog, cat")[0]
        self.assertEqual(list(again["turns"][1]["jspace"]["watch"]), ["dog", "cat"])
        self.assertFalse(self.manager.busy)

    def test_messages_send_earlier_replies_back(self):
        game = self.play()[0]
        self.assertEqual([m["role"] for m in messages_for(game, 1)], ["system", "user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
