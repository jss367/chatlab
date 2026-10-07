"""The prompt-edit tests' fixtures: a scripted model whose prompt arrives as clickable tokens.

Kept here so the test modules that edit a prompt share it without importing each other.
"""

import html
import json
import unittest
from unittest import mock

from chatlab import app
import gradio as gr
from chatlab.ui import runtime, token_menu
from conversation_support import SETTINGS, select
from fakes import FakeTokenizer, loaded_manager


# Enough vocabulary to tile "User: hi\nAssistant:", so a prompt arrives as
# several tokens with positions worth clicking rather than as one placeholder.
PIECES = [
    "User", ": ", "hi", "\n", "Assistant", ":", "Hello", " world", "<eos>", "<unk>",
]
EOS = PIECES.index("<eos>")
PROMPT_IDS = [0, 1, 2, 3, 4, 5]
# "hi", the third token of the prompt and the only word of the message in it.
MESSAGE_AT = PROMPT_IDS.index(2)
HELLO, WORLD = PIECES.index("Hello"), PIECES.index(" world")
# The fake model reads this by position, one step per token including the
# prompt's, so the prompt's own length is padded over to leave "Hello world"
# for the reply however long the edited prompt turns out to be.
SCRIPT = [HELLO] * len(PROMPT_IDS) + [HELLO, WORLD, EOS]


class PromptPieceTokenizer(FakeTokenizer):
    """Encodes every prompt by matching pieces, not as the single token 0."""

    def __call__(self, text, **kwargs):
        return super().__call__(text, **{**kwargs, "add_special_tokens": False})


def prompt_manager(script=SCRIPT, pieces=PIECES, eos=EOS):
    manager = loaded_manager(list(script), pieces, eos)
    manager.tokenizer = PromptPieceTokenizer(pieces, eos)
    return manager


def settled(frames):
    """A whole stream folded into the one frame the browser's state holds.

    The prompt panel and the prompt ids are published once, on the first frame
    that carries tokens, and skipped by every frame after it. The state keeps
    them; a test reading only the last frame would not.
    """

    final = frames[-1].copy()
    for name in final.names:
        if final[name] == gr.skip():
            final[name] = next(
                (frame[name] for frame in reversed(frames) if frame[name] != gr.skip()),
                final[name],
            )
    return final


def respond(message="hi", turns=(), settings=SETTINGS):
    runtime.MANAGER.model.step = 0
    return settled(list(app.chat(message, list(turns), *settings)))


class PromptEditFixture(unittest.TestCase):
    """A chat answered from the scripted model, with the menu's edit actions on its prompt."""

    def setUp(self):
        patch = mock.patch.object(runtime, "MANAGER", prompt_manager())
        patch.start()
        self.addCleanup(patch.stop)
        self.frame = respond()

    def prompt_ids(self, frame=None):
        _generation, ids, _load_id, *_steering = (frame or self.frame)["context_ids"]
        return list(ids)

    def payload(self, index=MESSAGE_AT, frame=None, request="open-1"):
        frame = frame or self.frame
        markup = token_menu.prompt_menu_payload(
            frame["prompt_metrics"], frame["context_ids"], request, select(index)
        )
        return json.loads(
            html.unescape(markup.split('data-token-menu="')[1].split('"')[0])
        )

    def edit(self, action, turns=None, settings=SETTINGS, frame=None):
        frame = frame or self.frame
        # The fake model keeps counting across runs; restarting its script is
        # what makes the reply to an edited prompt as readable as the first.
        runtime.MANAGER.model.step = 0
        return settled(list(token_menu.edit_prompt_from_menu(
            json.dumps(action),
            frame["context_ids"],
            frame["prompt_metrics"],
            "",
            self.frame["turns"] if turns is None else turns,
            *settings,
        )))

    def replace_with_text(self, text, index=MESSAGE_AT, **kwargs):
        payload = self.payload(index)
        return self.edit(
            dict(kind="text", text=text, selection=payload["selection"]), **kwargs
        )
