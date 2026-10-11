"""Writes hand-played-reopened-branched.json: a hand-played hangman game, reopened and then branched.

Format marker: chatlab-hangman-1
Era: hand-played games, before batches and reveal probes (no ``trial``, no ``probes``).
Generating commit: 4d2b657 Add a Hangman extension the loaded model hosts, read token by token (#147)

Commands used:
    git worktree add --detach /tmp/chatlab-at-4d2b657 4d2b657
    cd /tmp/chatlab-at-4d2b657 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/hangman/generators/hand-played-reopened-branched.py \
        <repo>/tests/fixtures/saved/hangman/hand-played-reopened-branched.json

The game is played through that commit's own page handlers (start_game, play, open_saved, select_response,
branch) against its test suite's scripted ``Manager`` model, one character per token. The page's own save()
writes every file; the written file is copied byte for byte.

What happens:
  1. A game opens under an edited system prompt. Response 1 has a closed reasoning block; response 2 comes
     from a template that opened the reasoning block (reasoning_prefilled) and draws an escaped Markdown
     board; response 3 draws a run-together board; response 4 is cut off by its token limit (finish
     "length") after its board line.
  2. That saved file is reopened (a new game whose parent is the first, with recorded_load_id on the
     reopened responses), and play continues: response 5's board is unreadable, response 6 moves A off
     position 3 and puts O there, response 7 guesses CRANE and the host reveals it.
  3. Response 7 is branched at the first letter of the revealed word with the text "crate": a new game
     whose parent is the reopened one, holding responses 1-6 and a branched response 7 that reveals CRATE
     against a board spelling CRANE. That branched game is the fixture.

Ids, timestamps and response times are pinned by replacing the game module's uuid4 and now and the page
module's clock; everything written still comes from the old save().
"""
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import uuid

sys.path.insert(1, "tests")

import gradio as gr  # noqa: E402

from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector  # noqa: E402
from extensions.hangman import game as game_module, page as page_module  # noqa: E402
from extensions.hangman.game import SYSTEM  # noqa: E402
from test_hangman import STOP, Manager  # noqa: E402

SYSTEM_PROMPT = SYSTEM + "\nUse only common five-letter English nouns."
OPENING = "Let's play. Think of your word and show me the empty board."


class ScriptedHost(Manager):
    """The suite's Manager, plus a reasoning-prefilled reply and a token limit it honors.

    A reply is text, or (text, reasoning_prefilled). Replies past max_new_tokens stop there without the
    stop token, as a real model cut off by its limit does.
    """

    def generate(self, messages, **options):
        self.calls.append(dict(messages=messages, **options))
        reply = self.replies.pop(0)
        text, prefilled = reply if isinstance(reply, tuple) else (reply, False)
        forced = list(options.get("forced_ids", ()))
        sampled = [ord(c) for c in text] + [STOP]
        limit = options.get("max_new_tokens")
        if limit is not None and len(sampled) > limit:
            sampled = sampled[:limit]
        metrics = []
        for position, token in enumerate(forced + sampled, 1):
            metrics.append(dict(token_id=token, text=chr(token) if token else "",
                                display_text=chr(token) if token else "⏹", segment="response",
                                position=position, scored=False,
                                top_candidates=[dict(token_id=ord("Z"), text="Z", probability=.1)]))
            yield SimpleNamespace(text="".join(chr(m["token_id"]) for m in metrics if m["token_id"]),
                                  metrics=[dict(m) for m in metrics], forced_prefix_tokens=len(forced),
                                  reasoning_prefilled=prefilled)


def main(output):
    ids = (uuid.UUID(int=0x4d2b657 * 1000 + n) for n in itertools.count(1))
    game_module.uuid4 = lambda: next(ids)
    stamps = (f"2026-09-20T10:{minute:02d}:00+00:00" for minute in itertools.count())
    game_module.now = lambda: next(stamps)
    clock = itertools.count(1_758_362_400, 3)
    page_module.time = SimpleNamespace(time=lambda: float(next(clock)))

    host = ScriptedHost([
        "<think>I will hold CRANE.</think>Board: _ _ _ _ _\nGuessed: none\nWrong guesses left: 6\nLet's begin.",
        ("E is the last letter of CRANE.</think>**Board:** \\_ \\_ \\_ \\_ E\nGuessed: E\nWrong guesses left: 6\n"
         "Yes, E is in the word.", True),
        "Board: __A_E\nGuessed: E A\nWrong guesses left: 6\nYes, A is in the word.",
        "Board: _ _ A _ E\nGuessed: E A S\nWrong guesses left: 5\nNo, S is not in the word.",
        "Board: _ R A _ E (two letters left)\nGuessed: E A S R\nWrong guesses left: 5",
        "Board: _ R O _ E\nGuessed: E A S R O\nWrong guesses left: 4\nYes, O is in the word.",
        "Board: C R A N E\nWord: crane\nYou got it.",
        "\nYou got it.",
    ])
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / "hangman"
        context = ExtensionContext(ModelService(lambda: host), TokenInspector(), data,
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            page_module.build_page(context)
        fn = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}

        game = list(fn["start_game"](SYSTEM_PROMPT, OPENING, "owner", 0.7, 40, 256))[-1][0]
        for guess, limit in (("e", 256), ("a", 256), ("s", 20)):
            game = list(fn["play"](game, guess, "owner", 0.7, 40, limit))[-1][0]
        assert game["turns"][3]["finish_reason"] == "length", game["turns"][3]
        reopened = fn["open_saved"](str(data / f"{game['id']}.json"), "owner")[0]
        for guess in ("r", "o", "crane"):
            reopened = list(fn["play"](reopened, guess, "owner", 0.7, 40, 256))[-1][0]
        payload = fn["select_response"](reopened, "owner", 6)[2]
        at = len("Board: C R A N E\nWord: ")
        action = json.dumps(dict(kind="text", text="crate", selection=dict(view_id=[reopened["id"], 6], index=at)))
        branched = list(fn["branch"](reopened, "owner", payload, action, 0.7, 40, 256))[-1][0]
        demo.close()
        assert branched["parent"] == dict(id=reopened["id"], turns=6), branched["parent"]
        shutil.copyfile(data / f"{branched['id']}.json", output)


if __name__ == "__main__":
    main(sys.argv[1])
