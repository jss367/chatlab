"""Replacing a prompt token and answering the message again from it."""

import html
import json
import unittest
from unittest import mock

from chatlab import app
import gradio as gr
from chatlab.text_generation import ModelChanged
from chatlab.ui import runtime, token_menu
from chatlab.conversation import forget_measurements, to_json, turn_entries
from conversation_support import FIXED, SETTINGS, metrics_of, select, strip_of, token_span
from fakes import FakeTokenizer, SentencePieceTokenizer, loaded_manager
import settings_sandbox


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


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


class PromptEditTests(unittest.TestCase):
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

    # ------------------------------------------------------------- the menu

    def test_the_menu_offers_the_clicked_token_and_its_alternatives(self):
        payload = self.payload()
        self.assertEqual(payload["text"], "hi")
        self.assertEqual(payload["selection"]["index"], MESSAGE_AT)
        self.assertEqual(payload["selection"]["source"], "prompt")
        self.assertTrue(payload["candidates"])
        self.assertEqual(payload["error"], "")

    def test_the_menu_escapes_token_text_and_keeps_the_request_id(self):
        self.frame["prompt_metrics"][1][MESSAGE_AT]["text"] = '<img src=x onerror="bad()">'
        payload = self.payload(request='"<request>')
        self.assertEqual(payload["text"], '<img src=x onerror="bad()">')
        self.assertEqual(payload["request"], '"<request>')

    def test_the_menu_refuses_after_a_model_reload(self):
        runtime.MANAGER.load_count += 1
        payload = self.payload()
        self.assertIsNone(payload["selection"])
        self.assertEqual(payload["error"], app.PROMPT_EDIT_MODEL_CHANGED)

    def test_the_menu_refuses_a_prompt_that_is_no_longer_on_screen(self):
        stale = self.frame
        self.frame = settled(list(app.retry_last("", stale["turns"], *SETTINGS)))
        payload = self.payload(frame=stale)
        self.assertIsNone(payload["selection"])
        self.assertEqual(payload["error"], app.PROMPT_EDIT_UNAVAILABLE)

    def test_the_scored_context_is_not_taken_for_the_chat_prompt(self):
        # Score text draws its context into the prompt strip under a stamp of
        # its own. Answering an edit of it would replace the chat's last reply
        # with one generated from the scored text.
        runtime.MANAGER.model.step = 0
        scored = list(app.score_text("Hello world", "hi", False, app.DEFAULT_COLOR_SCALE))[-1]
        frame = self.frame.copy()
        frame["prompt_metrics"], frame["context_ids"] = scored[4], scored[-1]

        payload = self.payload(index=0, frame=frame)
        self.assertIsNone(payload["selection"])
        self.assertEqual(payload["error"], app.PROMPT_EDIT_SCORED)

        selection = {"source": "prompt", "generation": scored[4][0], "index": 0}
        final = self.edit(dict(kind="text", text="bye", selection=selection), frame=frame)
        self.assertIn(app.PROMPT_EDIT_SCORED, final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    # -------------------------------------------------------------- editing

    def test_a_candidate_is_put_in_the_token_position(self):
        payload = self.payload()
        chosen = payload["candidates"][0]
        final = self.edit(dict(kind="candidate", index=0, selection=payload["selection"]))
        expected = list(PROMPT_IDS)
        expected[MESSAGE_AT] = chosen["token_id"]
        self.assertEqual(self.prompt_ids(final), expected)
        self.assertEqual(
            strip_of(final["prompt_strip"])[MESSAGE_AT][0], chosen["text"]
        )
        self.assertIn(f"Prompt token {MESSAGE_AT + 1}", final["status"])

    def test_typed_text_is_encoded_at_that_position(self):
        final = self.replace_with_text("Hello")
        expected = list(PROMPT_IDS)
        expected[MESSAGE_AT] = HELLO
        self.assertEqual(self.prompt_ids(final), expected)

    def test_text_the_tokenizer_cannot_place_exactly_is_refused(self):
        final = self.replace_with_text("unspellable")
        self.assertIn("cannot be inserted exactly", final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    def test_an_empty_replacement_is_refused(self):
        final = self.replace_with_text("")
        self.assertIn(app.PROMPT_EDIT_EMPTY, final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    def test_the_replacement_can_be_longer_than_the_token_it_replaces(self):
        final = self.replace_with_text("Hello world")
        expected = list(PROMPT_IDS)
        expected[MESSAGE_AT:MESSAGE_AT + 1] = [HELLO, WORLD]
        self.assertEqual(self.prompt_ids(final), expected)

    def test_a_control_token_of_the_template_can_be_replaced(self):
        # The point of editing a prompt rather than the message: the template's
        # own tokens are as editable as the words between them.
        final = self.replace_with_text("Hello", index=PROMPT_IDS.index(4))
        expected = list(PROMPT_IDS)
        expected[PROMPT_IDS.index(4)] = HELLO
        self.assertEqual(self.prompt_ids(final), expected)

    # ------------------------------------------------ what the edit records

    def test_the_reply_records_what_was_replaced(self):
        final = self.replace_with_text("Hello")
        expected = list(PROMPT_IDS)
        expected[MESSAGE_AT] = HELLO
        # The ids are the exact record: the messages beside them would be
        # rendered by the template, which is what the edit stepped around.
        self.assertEqual(
            final["trace"]["sampling"]["edited_prompt"],
            {
                "position": MESSAGE_AT + 1,
                "original": "hi",
                "replacement": "Hello",
                "prompt_token_ids": expected,
            },
        )
        self.assertEqual(self.prompt_ids(final), expected)
        self.assertIn("was replaced with 'Hello'", final["prompt_note"])
        self.assertIn("Prompt token 3: 'Hello' instead of 'hi'", final["status"])

    def test_the_conversation_still_says_what_was_asked(self):
        final = self.replace_with_text("Hello")
        self.assertEqual(
            [turn["content"] for turn in final["turns"]][:1],
            [turn["content"] for turn in self.frame["turns"]][:1],
        )
        self.assertEqual(len(final["turns"]), len(self.frame["turns"]))

    def test_the_next_message_is_prompted_from_the_conversation_again(self):
        # The edit answers one reply. Nothing about it survives into the next
        # prompt, which the chat template renders as it always would.
        edited = self.replace_with_text("Hello")
        after = respond(turns=edited["turns"])
        self.assertEqual(
            runtime.MANAGER.tokenizer.decode(self.prompt_ids(after)),
            f"User: hi\nAssistant: {edited["turns"][-1]['content']}\nUser: hi\nAssistant:",
        )
        self.assertIsNone(after["trace"]["sampling"].get("edited_prompt"))

    def test_the_system_prompt_setting_does_not_rebuild_the_edited_prompt(self):
        # A setting changed after the reply cannot quietly re-render the
        # prompt: the ids the reader edited are the ids that are fed.
        settings = tuple((FIXED | {"system_prompt": "You are Assistant."}).values())
        final = self.replace_with_text("Hello", settings=settings)
        expected = list(PROMPT_IDS)
        expected[MESSAGE_AT] = HELLO
        self.assertEqual(self.prompt_ids(final), expected)

    def test_the_assistant_prefill_still_applies_to_the_new_reply(self):
        # Only the prompt was replayed, so the response controls are live.
        settings = tuple((FIXED | {"assistant_prefill": " world"}).values())
        final = self.replace_with_text("Hello", settings=settings)
        self.assertEqual(final["trace"]["sampling"]["assistant_prefill"], " world")
        self.assertEqual(metrics_of(final["metrics"])[0]["token_id"], WORLD)

    # ------------------------------------------- branching the edited reply

    def branch(self, frame, action, index=0):
        """Right-click a token of the reply and take the menu's branch."""

        payload = json.loads(html.unescape(
            token_menu.token_menu_payload(
                frame["turns"], frame["metrics"], "branch",
                token_span(frame["turns"], index),
            ).split('data-token-menu="')[1].split('"')[0]
        ))
        runtime.MANAGER.model.step = 0
        return settled(list(token_menu.branch_from_menu(
            json.dumps(action | {"selection": payload["selection"]}),
            "", frame["turns"], *SETTINGS,
        )))

    def test_branching_the_reply_replays_the_prompt_it_was_given(self):
        # The conversation would render the unedited prompt. Replaying the
        # reply's tokens against that would score them under a context they
        # never had, so the prompt the reply was given is replayed instead.
        edited = self.replace_with_text("Hello")
        branched = self.branch(edited, {"kind": "regenerate"})
        self.assertEqual(self.prompt_ids(branched), self.prompt_ids(edited))
        self.assertIn("Regenerating from token 1", branched["status"])

    def test_a_typed_branch_of_the_reply_replays_it_too(self):
        edited = self.replace_with_text("Hello")
        branched = self.branch(edited, {"kind": "text", "text": " world"})
        self.assertEqual(self.prompt_ids(branched), self.prompt_ids(edited))

    def test_one_more_token_of_the_reply_replays_it_too(self):
        edited = self.replace_with_text("Hello")
        payload = json.loads(html.unescape(
            token_menu.token_menu_payload(
                edited["turns"], edited["metrics"], "next", token_span(edited["turns"], 0),
            ).split('data-token-menu="')[1].split('"')[0]
        ))
        _detail, pick = app.choose_alternative(
            edited["turns"], app.empty_metrics(), edited["prompt_metrics"],
            payload["selection"], select(0),
        )
        runtime.MANAGER.model.step = 0
        stepped = settled(list(app.next_token(pick, "", edited["turns"], *SETTINGS)))
        self.assertEqual(self.prompt_ids(stepped), self.prompt_ids(edited))

    def test_an_ordinary_reply_is_still_branched_against_its_template(self):
        branched = self.branch(self.frame, {"kind": "regenerate"})
        self.assertEqual(self.prompt_ids(branched), PROMPT_IDS)

    def test_the_prompt_goes_with_the_measurements(self):
        # It is only ever read beside them, and replaying it without them
        # would feed a prompt for a reply nothing is left to replay.
        edited = self.replace_with_text("Hello")
        self.assertIn("prompt_edit", edited["turns"][-1])
        forgotten = forget_measurements(edited["turns"], len(edited["turns"]) - 1)
        self.assertNotIn("prompt_edit", forgotten[-1])
        self.assertNotIn("prompt_edit", json.dumps(turn_entries(edited["turns"])))
        self.assertNotIn("prompt_edit", to_json(edited["turns"]))

    # ------------------------------------------------------------ refusals

    def test_an_edit_is_refused_while_a_generation_is_running(self):
        runtime.MANAGER.claim_generation()
        try:
            final = self.replace_with_text("Hello")
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(final["turns"], gr.skip())
        self.assertEqual(final["status"], app.BUSY_STATUS)

    def test_an_edit_is_refused_after_a_model_reload(self):
        payload = self.payload()
        runtime.MANAGER.load_count += 1
        final = self.edit(dict(kind="text", text="Hello", selection=payload["selection"]))
        self.assertIn(app.PROMPT_EDIT_MODEL_CHANGED, final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    def test_a_reload_between_the_check_and_the_encoding_is_refused(self):
        payload = self.payload()
        real = runtime.MANAGER.encode_prompt_replacement

        def reload_first(*args, **kwargs):
            runtime.MANAGER.load_count += 1
            return real(*args, **kwargs)

        with mock.patch.object(
            runtime.MANAGER, "encode_prompt_replacement", reload_first
        ):
            final = self.edit(
                dict(kind="text", text="Hello", selection=payload["selection"])
            )
        self.assertIn(app.PROMPT_EDIT_MODEL_CHANGED, final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    def test_an_old_selection_cannot_edit_a_newer_prompt(self):
        stale = self.frame
        newer = settled(list(app.retry_last("", stale["turns"], *SETTINGS)))
        payload = self.payload(frame=stale)
        final = self.edit(
            dict(kind="text", text="Hello", selection={"source": "prompt",
                                                       "generation": stale["prompt_metrics"][0],
                                                       "index": MESSAGE_AT}),
            turns=newer["turns"],
            frame=stale,
        )
        self.assertIsNone(payload["selection"])
        self.assertIn(app.PROMPT_EDIT_UNAVAILABLE, final["status"])
        self.assertEqual(final["turns"], newer["turns"])

    def test_an_alternative_that_is_not_the_tokens_own_is_refused(self):
        payload = self.payload()
        final = self.edit(dict(kind="candidate", index=99, selection=payload["selection"]))
        self.assertIn("not one of this token", final["status"])
        self.assertEqual(final["turns"], self.frame["turns"])

    def test_an_invalid_action_is_a_refusal(self):
        for action in ("not json", "{}", "null", '{"kind": "regenerate"}'):
            with self.subTest(action=action):
                final = settled(list(token_menu.edit_prompt_from_menu(
                    action, self.frame["context_ids"], self.frame["prompt_metrics"],
                    "", self.frame["turns"], *SETTINGS,
                )))
                self.assertIn(app.PROMPT_EDIT_UNAVAILABLE, final["status"])
                self.assertEqual(final["turns"], self.frame["turns"])

    def test_an_edit_with_nothing_to_answer_is_refused(self):
        payload = self.payload()
        final = self.edit(
            dict(kind="text", text="Hello", selection=payload["selection"]), turns=[]
        )
        self.assertIn(app.PROMPT_EDIT_NO_MESSAGE, final["status"])


class PromptReplacementEncodingTests(unittest.TestCase):
    """The typed half of an edit, at the tokenizer boundary."""

    def setUp(self):
        self.original = runtime.MANAGER
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def sentencepiece(self):
        pieces = ["▁Hello", "▁world", "world", "▁", "!", "<eos>"]
        manager = loaded_manager([0], pieces, pieces.index("<eos>"))
        manager.tokenizer = SentencePieceTokenizer(pieces, pieces.index("<eos>"))
        return manager, pieces

    def test_a_joined_word_uses_the_piece_that_joins(self):
        manager, pieces = self.sentencepiece()
        self.assertEqual(
            manager.encode_prompt_replacement([0], "world"), [pieces.index("world")]
        )

    def test_a_typed_leading_space_is_kept_once(self):
        manager, pieces = self.sentencepiece()
        self.assertEqual(
            manager.encode_prompt_replacement([0], " world"),
            [pieces.index("▁world")],
        )

    def test_a_special_token_may_be_typed_into_a_prompt(self):
        # A prompt is control tokens as much as words, so the rule that keeps
        # them out of a replayed response does not apply in front of one.
        manager = prompt_manager()
        self.assertEqual(manager.encode_prompt_replacement([0], "<eos>"), [EOS])

    def test_text_that_cannot_be_placed_exactly_is_refused(self):
        manager = prompt_manager()
        with self.assertRaises(ValueError):
            manager.encode_prompt_replacement([0], "unspellable")

    def test_a_reload_since_the_prefix_was_recorded_is_refused(self):
        manager = prompt_manager()
        load_id = manager.load_id
        manager.load_count += 1
        with self.assertRaises(ModelChanged):
            manager.encode_prompt_replacement([0], "Hello", load_id=load_id)


class EditedPromptGenerationTests(unittest.TestCase):
    """What the runtime does with a prompt it did not render."""

    def setUp(self):
        self.original = runtime.MANAGER
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def last_update(self, manager, **kwargs):
        return list(manager.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=2, seed=1, **kwargs,
        ))[-1]

    def test_the_edited_ids_are_fed_exactly(self):
        manager = prompt_manager()
        edited = [PIECES.index("Hello"), PIECES.index(" world")]
        update = self.last_update(manager, prompt_override_ids=edited)
        self.assertEqual(list(update.prompt_ids), edited)
        self.assertEqual(len(update.prompt_metrics), len(edited))

    def test_an_empty_edited_prompt_is_refused(self):
        manager = prompt_manager()
        with self.assertRaises(ValueError):
            self.last_update(manager, prompt_override_ids=[])

    def test_a_reasoning_marker_edited_away_stops_prefilling_reasoning(self):
        pieces = ["<think>", "</think>", "Hello", " world", "<eos>"]
        eos = pieces.index("<eos>")
        manager = prompt_manager(script=(2, eos), pieces=pieces, eos=eos)
        opened = self.last_update(manager, prompt_override_ids=[pieces.index("<think>")])
        self.assertTrue(opened.reasoning_prefilled)
        closed = self.last_update(manager, prompt_override_ids=[pieces.index("</think>")])
        self.assertFalse(closed.reasoning_prefilled)
