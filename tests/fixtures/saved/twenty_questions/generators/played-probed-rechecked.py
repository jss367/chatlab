"""Writes played-probed-rechecked.json: a twenty questions game with J space, reveal probes and a recheck.

Format marker: chatlab-twenty-questions-1
Era: first.
Generating commit: the Twenty questions introduction, committed with this generator and fixture.

Command used, from that commit's checkout:
    PYTHONPATH=. .venv/bin/python tests/fixtures/saved/twenty_questions/generators/played-probed-rechecked.py \
        tests/fixtures/saved/twenty_questions/played-probed-rechecked.json

The game is played through that commit's own page handlers (start_game, play, ask_what, recheck_game)
against its test suite's ``LensManager``, one character per token, with a fake lens fitted at blocks 1
and 2. The page's own save() writes every file; the last one is copied byte for byte.

What happens:
  1. The opening, with "cat, piano" on the watch list. Response 2 reasons in a closed block before
     answering YES. Response 3 skips a count (19 to 17). Response 4 asks response 2's question again and
     answers NO. J space is read after every reply.
  2. The model is asked what the thing is twice after response 4; the answers are CAT and HOUSE CAT.
  3. Response 5 guesses the cat and the host names it. Response 6 is asked with J space reading off.
  4. Every answered question is rechecked against CAT; the second answer disagrees.

Ids, timestamps and response times are pinned by replacing uuid4 and now in the game module and the
page module's clock.
"""
import itertools
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

sys.path.insert(1, "tests")

import gradio as gr  # noqa: E402

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector  # noqa: E402
from chatlab.extensions.twenty_questions import game as game_module, page as page_module  # noqa: E402
from chatlab.extensions.twenty_questions.game import SYSTEM  # noqa: E402
from test_twenty_questions import LensManager  # noqa: E402
from ui_support import handlers_by_name  # noqa: E402

REPLIES = [
    "I have something in mind.\nQuestions left: 20",
    "<think>A cat is alive.</think>Answer: yes\nQuestions left: 19",
    "Answer: no\nQuestions left: 17",
    "Answer: No.\nQuestions left: 16",
    " a cat", " house cat",
    "Answer: yes\nQuestions left: 15\nYou got it!\nThing: a cat",
    "Answer: yes\nQuestions left: 14",
    " yes", " yes", " no", " yes", " yes",
]


def main():
    ids = (uuid.UUID(int=0x20_0000 + n) for n in itertools.count())
    clock = itertools.count(1_800_000_000, 2)
    game_module.uuid4 = lambda: next(ids)
    game_module.now = lambda: "2026-11-05T12:00:00+00:00"
    page_module.time = type("Clock", (), {"time": staticmethod(lambda: float(next(clock)))})
    with tempfile.TemporaryDirectory() as directory:
        manager = LensManager(REPLIES)
        context = ExtensionContext(ModelService(lambda: manager), TokenInspector(), Path(directory),
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            page_module.build_page(context)
        fn = handlers_by_name(demo)
        settings = ("owner", 0.7, 40, 256)
        game = list(fn["start_game"](SYSTEM, "Let's play.", *settings, True, "cat, piano"))[-1][0]
        for question in ("Is it alive?", "Is it bigger than a breadbox?", "is it alive"):
            game = list(fn["play"](game, question, *settings, True, "cat, piano"))[-1][0]
        game = fn["ask_what"](game, "owner", 2, 1.0)[0]
        game = list(fn["play"](game, "Is it a cat?", *settings, True, "cat, piano"))[-1][0]
        game = list(fn["play"](game, "Does it purr?", *settings, False, ""))[-1][0]
        game = fn["recheck_game"](game, "owner", "")[0]
        shutil.copyfile(Path(directory) / f"{game['id']}.json", sys.argv[1])
        demo.close()


if __name__ == "__main__":
    main()
