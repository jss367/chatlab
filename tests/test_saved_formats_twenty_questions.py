"""Twenty questions games written by earlier ChatLab versions still open, and read the same way.

Each fixture under tests/fixtures/saved/twenty_questions was written once by the commit its generator names
and is never regenerated. Opening a saved game reads its answers, counts, named things and probe answers
again from the recorded text, so these tests pin what the file says when the page opens it.
"""
import json
from pathlib import Path
import tempfile
import unittest

import gradio as gr

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.twenty_questions.game import SYSTEM, messages_for, saved
from chatlab.extensions.twenty_questions.page import build_page
from hangman_support import CharacterModel
from ui_support import handlers_by_name

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "saved" / "twenty_questions"


class Idle(CharacterModel):
    """A model nobody asks to generate: opening a saved game touches no session."""
    busy = False


class PlayedGameTests(unittest.TestCase):
    """played-probed-rechecked.json: written by the Twenty questions introduction."""
    PATH = FIXTURES / "played-probed-rechecked.json"
    QUESTIONS = ["Let's play.", "Is it alive?", "Is it bigger than a breadbox?", "is it alive", "Is it a cat?",
                 "Does it purr?"]

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        context = ExtensionContext(ModelService(lambda: Idle()), TokenInspector(), self.directory / "tq",
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = handlers_by_name(demo)

    def open(self, path):
        names = ("game", "chat", "check", "watch", "picker", "note", "tokens", "strip", "raw", "jspace",
                 "download", "detail", "alternatives", "picked")
        return dict(zip(names, self.fn["open_saved"](str(path), "owner"), strict=True))

    def assert_game_reads(self, shown):
        game = shown["game"]
        self.assertEqual(game["format"], "chatlab-twenty-questions-1")
        self.assertEqual(game["system"], SYSTEM)
        turns = game["turns"]
        self.assertEqual([t["question"] for t in turns], self.QUESTIONS)
        self.assertEqual([t["verdict"] for t in turns], [None, "yes", "no", "no", "yes", "yes"])
        self.assertEqual([t["left"] for t in turns], [20, 19, 17, 16, 15, 14])
        self.assertEqual([t["revealed"] for t in turns], [None] * 4 + ["cat", None])
        self.assertEqual([t["sampling"]["seed"] for t in turns], list(range(40, 46)))
        self.assertEqual(turns[1]["answer"], "Answer: yes\nQuestions left: 19")
        # The next prompt carries the reasoning back in; the template decides whether it stays.
        self.assertEqual(messages_for(game, 2)[4], {"role": "assistant", "content": turns[1]["text"]})
        self.assertEqual([p["thing"] for p in turns[3]["probes"]], ["cat", "house cat"])
        self.assertEqual([[p["position"] for p in t["jspace"]["positions"]] for t in turns[:5]],
                         [[2], [2, 40], [2, 10], [2, 10], [2, 10]])
        self.assertNotIn("jspace", turns[5])
        self.assertEqual(turns[1]["jspace"]["watch"]["piano"], [-9.5, -8.733333333333334])
        self.assertEqual(turns[1]["jspace"]["positions"][1]["blocks"][1]["top"][0],
                         dict(token_id=34, text='"', log_prob=-0.5))
        self.assertEqual(game["recheck"]["thing"], "cat")
        self.assertEqual([(r["turn"], r["game_verdict"], r["verdict"]) for r in game["recheck"]["rows"]],
                         [(1, "yes", "yes"), (2, "no", "yes"), (3, "no", "no"), (4, "yes", "yes"),
                          (5, "yes", "yes")])
        self.assertEqual(shown["check"].split("\n\n"), [
            "**Questions answered:** 5 · yes 3 · no 2 · **Questions left:** 14 of 20",
            "**Thing named:** cat",
            "**Contradictions:**\n- Response 3: Questions left went from 19 at response 2 to 17; 18 would follow.\n"
            "- Response 4: Response 2 answered YES to the same question; this one answers NO.",
            "**Recheck against cat:** 4 of 5 answers agree (asked fresh, greedy, by `test/model`).\n"
            "- Response 3, `Is it bigger than a breadbox?`: the game said NO, asked about cat the model says YES.",
        ])
        self.assertIn("<th>piano</th>", shown["watch"])

    def test_the_page_opens_it_as_a_new_child_of_the_file(self):
        shown = self.open(self.PATH)
        self.assert_game_reads(shown)
        recorded = json.loads(self.PATH.read_text(encoding="utf-8"))
        self.assertNotEqual(shown["game"]["id"], recorded["id"])
        self.assertEqual(shown["game"]["parent"], dict(id=recorded["id"], turns=6))
        self.assertEqual([t["load_id"] for t in shown["game"]["turns"]], [None] * 6)
        self.assertIn("J space", shown["jspace"])

    def test_resaved_by_todays_writer_it_reads_the_same(self):
        game = self.open(self.PATH)["game"]
        path = self.directory / "resaved.json"
        path.write_text(saved(game), encoding="utf-8")
        self.assert_game_reads(self.open(path))
