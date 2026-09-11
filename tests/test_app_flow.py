import asyncio
import copy
import inspect
import os
import stat
import unittest
from dataclasses import replace
from unittest import mock
from pathlib import Path

import gradio as gr
import numpy as np

import app
from ui import runtime
import charts
from conversation import (
    MAIN_BRANCH,
    branch_sampling,
    display_messages,
    forget_measurements,
    make_turn,
    turn_entries,
    model_messages,
    new_forks,
    put_branch,
    put_branch_sampling,
)
from model_runtime import GENERATING, GenerationUpdate, ModelChanged, TokenInsight
from token_metrics import DEFAULT_COLOR_SCALE

import library
import settings
import settings_sandbox
from test_streaming import ChatTemplateTokenizer, FakeTokenizer, SentencePieceTokenizer, loaded_manager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


# "Hello" and " world" are the answer; the reasoning tags are their own tokens.
THINK_PIECES = ["<think>", "</think>", "Hello", " world", "<eos>"]
THINK_EOS = 4

FIXED = {
    "system_prompt": "",
    "keep_reasoning": False,
    "assistant_prefill": "",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 8,
    "seed": 42,
    "randomize_seed": False,
    "analyze_prompt": True,
    "scale_name": DEFAULT_COLOR_SCALE,
}
SETTINGS = tuple(FIXED.values())

# The chat handlers publish app.CHAT_OUTPUT_NAMES, in that order.
(
    PROMPT,
    CHATBOT,
    TURNS,
    STRIP,
    METRICS,
    STATUS,
    SEED,
    SEND,
    STOP,
    DETAIL,
    ALTS,
    PROMPT_STRIP,
    PROMPT_METRICS,
    PROMPT_NOTE,
    SUMMARY,
    SURPRISE,
    TRACE,
    CONTEXT_IDS,
    SELECTED_TOKEN,
    BRANCH_PICK,
) = range(len(app.CHAT_OUTPUT_NAMES))
CHAT_OUTPUTS = len(app.CHAT_OUTPUT_NAMES)

# The panels every conversation-replacing handler resets after its own rows:
# the prompt strip and its state and note, the two charts, and the export.
PANEL_OUTPUTS = 8
UNDO_OUTPUTS = 10 + PANEL_OUTPUTS
# Clear also resets the forks and their picker, and closes the
# confirmation panel that sent it.
CLEAR_OUTPUTS = 9 + PANEL_OUTPUTS + 3
LOAD_OUTPUTS = 10 + PANEL_OUTPUTS


def metrics_of(payload):
    """The metrics half of a metrics_state payload, dropping its stamp."""

    _generation, metrics = payload
    return metrics


def strip_of(value):
    """The tokens in a strip output, whether it is a value or a gr.update."""

    return value["value"] if isinstance(value, dict) else value


def painted(value):
    """The measured tokens in a token-view value: the spans that carry a color.

    The rest are the conversation's plain text - role headings, typed
    messages, replies with no measurements behind them - which stay on screen
    when the measurements go.
    """

    return [span for span in strip_of(value) if span[1] is not None]


def select(index):
    return gr.SelectData(None, {"index": index, "value": "x"})


def token_span(turns, token_index, turn=-1):
    """A click on one token of one reply in the conversation's token view."""

    _spans, index = app.transcript_entries(turns, DEFAULT_COLOR_SCALE)
    position = turn if turn >= 0 else len(turns) + turn
    return select(index.index((position, token_index)))


def click_token(frame, token_index, turn=-1):
    """The selection a click on a reply's token publishes."""

    turns = frame[TURNS]
    _detail, _rows, selection, _target, _pick = app.select_transcript_token(
        turns, frame[METRICS], token_span(turns, token_index, turn)
    )
    return selection


def score_known_passage(context="Hello", text=" world"):
    """Score real vocabulary tokens; the chat fixture normally encodes prompts as [0]."""

    class PassageTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__(text, **dict(kwargs, add_special_tokens=False))

    original = runtime.MANAGER.tokenizer
    runtime.MANAGER.tokenizer = PassageTokenizer(THINK_PIECES, THINK_EOS)
    try:
        return list(app.score_text(context, text, False, DEFAULT_COLOR_SCALE))[-1]
    finally:
        runtime.MANAGER.tokenizer = original


class ChatFlowTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def last(self, stream):
        frames = list(stream)
        self.assertTrue(frames)
        for frame in frames:
            self.assertEqual(len(frame), CHAT_OUTPUTS)
        return frames

    def test_a_message_produces_a_user_turn_and_a_reply(self):
        frames = self.last(app.chat("hi", [], *SETTINGS))
        final = frames[-1]
        self.assertEqual(final[PROMPT], "")
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        self.assertEqual(final[TURNS][1]["content"], "Hello world")
        self.assertEqual(final[SEED], 42)
        self.assertIn("seed 42", final[STATUS])
        self.assertEqual(len(final[STRIP]), 3)

    def test_an_assistant_prefill_starts_the_visible_answer(self):
        settings = dict(FIXED, assistant_prefill="Hello")
        frames = self.last(app.chat("hi", [], *settings.values()))
        final = frames[-1]

        self.assertEqual(final[TURNS][1]["content"], "Hello world")
        self.assertIn("Assistant prefill applied", frames[0][STATUS])
        self.assertEqual(final[TRACE]["sampling"]["assistant_prefill"], "Hello")
        self.assertEqual(final[TRACE]["sampling"]["forced_prefix_tokens"], 1)

    def test_literal_reasoning_tags_in_a_prefill_remain_visible(self):
        runtime.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        final = self.last(app.chat("hi", [], *settings.values()))[-1]

        self.assertEqual(final[TURNS][1]["reasoning"], "")
        self.assertEqual(final[TURNS][1]["content"], "<think>Hello</think> world")

    def test_the_stop_button_is_shown_while_streaming(self):
        frames = self.last(app.chat("hi", [], *SETTINGS))
        self.assertEqual(frames[0][SEND], gr.update(visible=False))
        self.assertEqual(frames[0][STOP], gr.update(visible=True))
        self.assertEqual(frames[-1][SEND], gr.update(visible=True))
        self.assertEqual(frames[-1][STOP], gr.update(visible=False))

    def test_reasoning_is_split_out_of_the_answer(self):
        runtime.MANAGER = loaded_manager([0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        reply = final[TURNS][1]
        self.assertEqual(reply["reasoning"], "Hello")
        self.assertEqual(reply["content"], "world")
        thoughts = [
            message
            for message in final[CHATBOT]
            if message.get("metadata", {}).get("title")
        ]
        self.assertEqual(len(thoughts), 1)
        self.assertEqual(thoughts[0]["content"], "Hello")

    def test_an_empty_message_keeps_the_box_and_the_history(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "there")]
        final = self.last(app.chat("   ", turns, *SETTINGS))[-1]
        self.assertEqual(final[PROMPT], "   ")
        self.assertEqual(final[TURNS], turns)
        self.assertEqual(final[STATUS], "Enter a message first.")

    def test_no_model_loaded_keeps_the_message(self):
        runtime.MANAGER = self.original
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual(final[PROMPT], "hi")
        self.assertEqual(final[TURNS], [])

    def test_randomizing_the_seed_changes_it(self):
        settings = dict(FIXED, randomize_seed=True)
        seeds = {
            self.last(app.chat("hi", [], *settings.values()))[-1][SEED]
            for _ in range(3)
        }
        self.assertGreater(len(seeds), 1)
        self.assertNotIn(42, seeds)

    def test_retry_replaces_only_the_last_reply(self):
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "stale"),
            make_turn("user", "two"),
            make_turn("assistant", "also stale"),
        ]
        final = self.last(app.retry_last("draft", turns, *SETTINGS))[-1]
        self.assertEqual(
            [turn["content"] for turn in final[TURNS]],
            ["one", "stale", "two", "Hello world"],
        )
        self.assertEqual(final[PROMPT], "draft")

    def test_retry_with_nothing_to_retry(self):
        final = self.last(app.retry_last("draft", [], *SETTINGS))[-1]
        self.assertEqual(final[STATUS], "There is nothing to retry.")

    def test_editing_a_user_message_truncates_and_regenerates(self):
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "stale"),
            make_turn("user", "two"),
            make_turn("assistant", "also stale"),
        ]
        event = gr.EditData(
            None, {"index": 0, "previous_value": "one", "value": "edited"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(
            [turn["content"] for turn in final[TURNS]], ["edited", "Hello world"]
        )

    def test_editing_an_assistant_message_keeps_it(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "stale")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "stale", "value": "fixed"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual([turn["content"] for turn in final[TURNS]], ["one", "fixed"])
        self.assertEqual(final[STATUS], "Assistant message edited.")

    def test_editing_an_assistant_message_drops_the_token_diagnostics(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "stale")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "stale", "value": "fixed"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        # The reply is still in the token view; what goes is its coloring,
        # since the ranks and probabilities described text the edit replaced.
        self.assertEqual(painted(final[STRIP]), [])
        self.assertIn(("fixed", None), strip_of(final[STRIP]))
        self.assertEqual(metrics_of(final[METRICS]), [])
        self.assertEqual(final[DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(final[ALTS], [])

    def test_editing_an_assistant_message_forgets_its_token_counts(self):
        # The counts describe the generated text, which the edit replaced; a
        # later reply's prompt count measured the transcript before the edit.
        # The model that answered is still the model that answered.
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
            make_turn("user", "three"),
            make_turn("assistant", "third"),
        ]
        for turn, prompt, generated in (
            (turns[1], 10, 5),
            (turns[3], 30, 7),
            (turns[5], 50, 9),
        ):
            turn.update(
                model="org/alpha", prompt_tokens=prompt, generated_tokens=generated
            )
        event = gr.EditData(
            None, {"index": 3, "previous_value": "second", "value": "fixed"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        edited, later = final[TURNS][3], final[TURNS][5]
        self.assertEqual(edited["content"], "fixed")
        self.assertEqual(set(edited), {"role", "content", "reasoning", "model"})
        self.assertNotIn("prompt_tokens", later)
        self.assertEqual(later["generated_tokens"], 9)
        self.assertEqual(final[TURNS][1]["prompt_tokens"], 10)
        self.assertEqual(
            app.branch_choices(new_forks(), final[TURNS])[0][0],
            "Main · one\nalpha · 15 tokens",
        )

    def test_a_new_response_resets_the_selected_token_details(self):
        # The first frame draws the reply being answered into with no tokens in
        # it yet, so the token the user had selected in the response it
        # replaces no longer exists and its probabilities must not stay on
        # screen beside it.
        frames = self.last(app.chat("hi", [], *SETTINGS))
        self.assertEqual(painted(frames[0][STRIP]), [])
        self.assertEqual(frames[0][DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(strip_of(frames[0][ALTS]), [])
        # Later frames only append tokens, so a token picked mid-stream
        # stays valid and its details are left alone.
        for frame in frames[1:]:
            self.assertEqual(frame[DETAIL], gr.skip())
            self.assertEqual(frame[ALTS], gr.skip())

    def test_a_retry_resets_the_selected_token_details(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "stale")]
        frames = self.last(app.retry_last("", turns, *SETTINGS))
        self.assertEqual(frames[0][DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(strip_of(frames[0][ALTS]), [])

    def test_a_new_reply_disarms_an_older_turn_only_in_its_reset_frame(self):
        first = self.last(app.chat("hi", [], *SETTINGS))[-1]
        selection = click_token(first, 1)
        _detail, pick = app.choose_alternative(
            first[TURNS], app.empty_metrics(), app.empty_metrics(), selection, select(0)
        )
        self.assertIsNotNone(pick)
        frames = self.last(app.chat("again", first[TURNS], *SETTINGS))
        self.assertIsNotNone(app.selected_metric(frames[-1][TURNS], selection))
        self.assertIsNone(frames[0][SELECTED_TOKEN])
        self.assertIsNone(frames[0][BRANCH_PICK])
        for frame in frames[1:]:
            self.assertEqual(frame[SELECTED_TOKEN], gr.skip())
            self.assertEqual(frame[BRANCH_PICK], gr.skip())
        refused = self.last(app.branch_from(
            frames[0][BRANCH_PICK], "", frames[-1][TURNS], *SETTINGS
        ))[-1]
        self.assertEqual(refused[TURNS], frames[-1][TURNS])

    def test_every_conversation_reset_publishes_the_branch_states(self):
        demo = app.build_app()
        chat = next(fn for fn in demo.fns.values() if fn.fn is app.chat)
        selection, pick = chat.outputs[-2:]
        resets = {
            "chat", "retry_last", "retry_message", "edit_message", "branch_from",
            "branch_with_text", "undo_last", "undo_message", "clear_chat",
            "fork_conversation", "new_conversation", "switch_fork", "delete_fork",
            "load_with_steering", "score_text",
        }
        for fn in demo.fns.values():
            if getattr(fn.fn, "__name__", None) in resets:
                self.assertIn(selection, fn.outputs, fn.fn.__name__)
                self.assertIn(pick, fn.outputs, fn.fn.__name__)

    def test_undo_disarms_a_selection_even_when_its_older_turn_survives(self):
        first = self.last(app.chat("hi", [], *SETTINGS))[-1]
        second = self.last(app.chat("again", first[TURNS], *SETTINGS))[-1]
        selection = click_token(second, 1, turn=1)
        undone = app.undo_last(second[TURNS])
        self.assertIsNotNone(app.selected_metric(undone[2], selection))
        self.assertEqual(undone[-2:], (None, None))
        no_change = app.undo_last([])
        self.assertEqual(no_change[-2:], (gr.skip(), gr.skip()))

    def test_streaming_skip_does_not_delete_the_rendered_table_data(self):
        # The browser retains the table value by reference. Gradio's client
        # applies the next stream patch in place, even for a skipped output.
        # Exercise real postprocessing and diffs: a bare [] reset used to
        # turn into delete(data), delete(headers), then crash the table render.
        demo = app.build_app()
        listener = next(fn for fn in demo.fns.values() if fn.fn is app.chat)
        frames = self.last(app.chat("hi", [], *SETTINGS))

        async def wire_frames():
            state = gr.blocks.SessionState(demo)
            return [
                await demo.postprocess_data(listener, frame, state)
                for frame in frames[:2]
            ]

        first, second = asyncio.run(wire_frames())
        demo.handle_streaming_diffs(listener, first, "table-regression", 1, final=False)
        patch = demo.handle_streaming_diffs(
            listener, second, "table-regression", 1, final=False
        )[ALTS]
        client_value = copy.deepcopy(first[ALTS])
        rendered_table = client_value.get("value", client_value)
        expected = copy.deepcopy(rendered_table)
        self.assertEqual(rendered_table["data"], [])

        # These two frames only add/delete dictionary properties. Match the
        # client's in-place edits, including the alias held by the renderer.
        for action, path, value in patch:
            target = client_value
            for key in path[:-1]:
                target = target[key]
            if action == "delete":
                del target[path[-1]]
            else:
                self.assertEqual(action, "add")
                target[path[-1]] = value
        self.assertEqual(rendered_table, expected)

    def test_a_refused_send_keeps_the_token_diagnostics(self):
        # gr.skip() leaves the previous response's panel on screen.
        final = self.last(app.chat("   ", [], *SETTINGS))[-1]
        self.assertEqual(final[STATUS], "Enter a message first.")
        for index in (STRIP, METRICS, DETAIL, ALTS):
            self.assertEqual(final[index], gr.skip())

    def test_editing_a_user_turn_keeps_history_when_no_model_is_loaded(self):
        """A refused edit must not truncate the conversation it cannot replace."""

        runtime.MANAGER = self.original  # nothing loaded
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]
        event = gr.EditData(
            None, {"index": 0, "previous_value": "one", "value": "edited"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(
            [turn["content"] for turn in final[TURNS]],
            ["one", "first", "two", "second"],
        )
        self.assertEqual(final[STATUS], "Download and load a model first.")

    def test_retrying_keeps_history_when_no_model_is_loaded(self):
        runtime.MANAGER = self.original
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        final = self.last(app.retry_last("", turns, *SETTINGS))[-1]
        self.assertEqual([turn["content"] for turn in final[TURNS]], ["one", "first"])
        self.assertEqual(final[STATUS], "Download and load a model first.")

    def test_an_assistant_edit_reserves_the_generation_slot(self):
        """A Send that wins the race between the busy check and the publish.

        The manager reports idle - so the guard at the top of edit_message()
        lets this through - but the slot is gone by the time the edit tries to
        claim it. Without the reservation the edit publishes its stale
        conversation anyway, and the generation frames then erase the edit.
        """

        class Sniped:
            """Idle when asked, taken when claimed."""

            loaded = True
            busy = False
            loading_id = None
            occupant = GENERATING

            def claim_generation(self):
                return GENERATING

            def release_generation(self):  # pragma: no cover - never reached
                raise AssertionError("released a slot it never held")

        runtime.MANAGER = Sniped()
        turns = [make_turn("user", "one"), make_turn("assistant", "reply")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "reply", "value": "fixed"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]

        self.assertEqual(final[TURNS], gr.skip(), "published a stale conversation")
        self.assertEqual(final[CHATBOT], gr.skip())
        self.assertEqual((final[SEND], final[STOP]), (gr.skip(), gr.skip()))
        self.assertIn("already generating", final[STATUS].lower())

    def test_an_assistant_edit_releases_the_slot_afterwards(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "reply")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "reply", "value": "fixed"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual([turn["content"] for turn in final[TURNS]], ["one", "fixed"])
        self.assertFalse(runtime.MANAGER.busy, "the slot must not leak")

    def test_a_cancelled_assistant_edit_releases_the_slot(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "reply")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "reply", "value": "fixed"}
        )
        stream = app.edit_message(event, "", turns, *SETTINGS)
        next(stream)
        stream.close()
        self.assertFalse(runtime.MANAGER.busy, "GeneratorExit must release the slot")

    def test_editing_a_reasoning_block_leaves_the_answer_alone(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "answer", "thought")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "thought", "value": "revised"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["reasoning"], "revised")
        self.assertEqual(final[TURNS][1]["content"], "answer")

    def test_blanking_an_assistant_message_is_refused(self):
        # An empty assistant turn is still drawn as a bubble but skipped by
        # model_messages(), so the screen and the model would disagree and the
        # next request would carry two user messages in a row.
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "first", "value": "   "}
        )
        final = self.last(app.edit_message(event, "draft", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS], turns)
        self.assertEqual(final[STATUS], "An assistant message cannot be emptied.")

    def test_blanking_an_answer_that_still_has_reasoning_is_allowed(self):
        # model_messages() keeps an empty slot for a turn that has reasoning,
        # so role alternation survives and the edit is a legitimate one.
        turns = [make_turn("user", "one"), make_turn("assistant", "answer", "thought")]
        event = gr.EditData(None, {"index": 2, "previous_value": "answer", "value": ""})
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["content"], "")
        self.assertEqual(final[TURNS][1]["reasoning"], "thought")
        self.assertEqual(final[STATUS], "Assistant message edited.")

    def test_blanking_the_only_reasoning_of_an_empty_answer_is_refused(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "", "thought")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "thought", "value": ""}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS], turns)
        self.assertEqual(final[STATUS], "An assistant message cannot be emptied.")

    def test_blanking_a_user_message_is_refused(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        event = gr.EditData(None, {"index": 0, "previous_value": "one", "value": "   "})
        final = self.last(app.edit_message(event, "draft", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS], turns)
        self.assertEqual(final[STATUS], "A user message cannot be empty.")

    def test_a_failed_generation_stays_out_of_the_history(self):
        class Exploding:
            loaded = True
            busy = False
            loading_id = None
            occupant = None
            model_id = "fake/model"
            load_id = "fake/model#1"

            def claim_generation(self):
                return None

            def release_generation(self):
                pass

            def generate(self, *_args, **_kwargs):
                raise RuntimeError("out of memory")
                yield  # pragma: no cover - makes this a generator

        runtime.MANAGER = Exploding()
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user"])
        self.assertIn("out of memory", final[STATUS])

    def test_prefilled_reasoning_streams_into_the_reasoning_block(self):
        # The OLMo Think template ends the prompt with <think>, so the reply
        # carries no opening marker. Until </think> lands, every token belongs
        # in the pending Reasoning block rather than the answer bubble.
        def thinking(*_args, **_kwargs):
            yield GenerationUpdate(
                text="Let me add two and two",
                metrics=[],
                load_id=runtime.MANAGER.load_id,
                reasoning_prefilled=True,
            )
            yield GenerationUpdate(
                text="Let me add two and two.</think>Four.",
                metrics=[],
                load_id=runtime.MANAGER.load_id,
                reasoning_prefilled=True,
            )

        runtime.MANAGER.generate = thinking
        frames = self.last(app.chat("hi", [], *SETTINGS))

        # frames[0] is the pre-generation snapshot, frames[1] the first update.
        mid = frames[1][TURNS][1]
        self.assertEqual(mid["reasoning"], "Let me add two and two")
        self.assertEqual(mid["content"], "")
        self.assertFalse(mid["reasoning_closed"])

        final = frames[-1][TURNS][1]
        self.assertEqual(final["reasoning"], "Let me add two and two.")
        self.assertEqual(final["content"], "Four.")
        self.assertTrue(final["reasoning_closed"])

    def test_a_plain_reply_never_streams_as_reasoning(self):
        def plain(*_args, **_kwargs):
            yield GenerationUpdate(
                text="Four.", metrics=[], load_id=runtime.MANAGER.load_id
            )

        runtime.MANAGER.generate = plain
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["reasoning"], "")
        self.assertEqual(final[TURNS][1]["content"], "Four.")

    def test_a_failure_after_some_tokens_keeps_them(self):
        def failing(*_args, **_kwargs):
            yield GenerationUpdate(
                text="<think>Hmm", metrics=[], load_id=runtime.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        runtime.MANAGER.generate = failing
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        reply = final[TURNS][1]
        self.assertEqual(reply["reasoning"], "Hmm")
        self.assertEqual(reply["content"], "")
        self.assertTrue(reply["reasoning_closed"])
        self.assertIn("gpu fell over", final[STATUS])


class AssistantPrefillSplittingTests(unittest.TestCase):
    def test_reader_supplied_whitespace_is_preserved(self):
        prefix = "  \n  code:  "
        reasoning, answer, closed = app.split_response_text(
            prefix + "continued", literal_prefill=prefix
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "  \n  code:  continued")
        self.assertTrue(closed)

    def test_template_separator_is_trimmed_but_reader_whitespace_is_preserved(self):
        prefix = "</think>\n\n  answer"
        reasoning, answer, closed = app.split_response_text(
            prefix + " continued",
            literal_prefill=prefix,
            reasoning_prefilled=True,
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "  answer continued")
        self.assertTrue(closed)

    def test_literal_tags_are_not_interpreted_as_reasoning(self):
        prefix = "Show <think>literal</think>: "
        reasoning, answer, closed = app.split_response_text(
            prefix + "continued", literal_prefill=prefix
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "Show <think>literal</think>: continued")
        self.assertTrue(closed)

    def test_template_close_stays_control_while_prefill_tags_stay_literal(self):
        prefix = "</think>\n\nShow <think>literal</think>: "
        reasoning, answer, closed = app.split_response_text(
            prefix + "continued",
            literal_prefill=prefix,
            reasoning_prefilled=True,
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "Show <think>literal</think>: continued")
        self.assertTrue(closed)

    def test_reasoning_sampled_after_the_prefill_keeps_its_meaning(self):
        prefix = "Visible prefix: "
        reasoning, answer, closed = app.split_response_text(
            prefix + "<think>sampled reasoning</think>answer",
            literal_prefill=prefix,
        )

        self.assertEqual(reasoning, "sampled reasoning")
        self.assertEqual(answer, "Visible prefix: answer")
        self.assertTrue(closed)

    def test_a_partial_tag_in_a_streaming_prefill_remains_visible(self):
        prefix = "Literal <thi"
        reasoning, answer, closed = app.split_response_text(
            prefix,
            literal_prefill=prefix,
            streaming=True,
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, prefix)
        self.assertTrue(closed)

    def test_a_stable_prefix_protects_tags_after_a_partial_character_resolves(self):
        reasoning, answer, closed = app.split_response_text(
            "<think>\U0001f4be continued",
            literal_prefill="<think>",
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "<think>\U0001f4be continued")
        self.assertTrue(closed)

    def test_a_literal_span_protects_reasoning_tags_after_sampled_text(self):
        text = "sampled <think>typed</think><think>model</think>answer"
        typed = "<think>typed</think>"
        start = text.index(typed)

        reasoning, answer, closed = app.split_response_text(
            text,
            literal_spans=((start, start + len(typed)),),
        )

        self.assertEqual(reasoning, "model")
        self.assertEqual(answer, "sampled <think>typed</think>answer")
        self.assertTrue(closed)

    def test_a_partial_reasoning_tag_in_a_literal_span_streams_visibly(self):
        text = "sampled <thi"
        reasoning, answer, closed = app.split_response_text(
            text,
            literal_spans=((len("sampled "), len(text)),),
            streaming=True,
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(answer, text)
        self.assertTrue(closed)


class SeedTests(unittest.TestCase):
    """Whatever reaches resolve_seed(), NumPy has to accept the result.

    ``np.random.default_rng()`` rejects negative integers, so a locked seed of
    ``-1`` reaching the generator makes every response fail instead of being
    produced.
    """

    def usable(self, seed):
        resolved = app.resolve_seed(seed, False)
        # The assertion that matters: this is the call the generator makes.
        np.random.default_rng(resolved)
        return resolved

    def test_a_negative_seed_is_clamped_to_a_usable_one(self):
        self.assertEqual(self.usable(-1), 0)
        self.assertEqual(self.usable(-(2**40)), 0)

    def test_a_usable_seed_is_kept(self):
        self.assertEqual(self.usable(42), 42)
        self.assertEqual(self.usable(0), 0)
        self.assertEqual(self.usable(app.SEED_LIMIT - 1), app.SEED_LIMIT - 1)

    def test_unusable_values_fall_back_to_zero(self):
        for value in (None, "", "abc", float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertEqual(self.usable(value), 0)

    def test_a_float_seed_is_truncated_then_clamped(self):
        self.assertEqual(self.usable(7.9), 7)
        # -1.5 truncates to -1, which is the value NumPy rejects.
        self.assertEqual(self.usable(-1.5), 0)

    def test_randomizing_ignores_the_typed_value(self):
        self.assertNotEqual(app.resolve_seed(-1, True), -1)
        for _ in range(20):
            np.random.default_rng(app.resolve_seed(-1, True))

    def test_the_seed_input_rejects_negative_values(self):
        # The clamp above is the backstop; the input is what stops a typed -1
        # from ever becoming a seed the user thinks was used.
        demo = app.build_app()
        numbers = [
            block
            for block in demo.blocks.values()
            if isinstance(block, gr.Number) and block.label == "Random seed"
        ]
        self.assertEqual(len(numbers), 1)
        self.assertEqual(numbers[0].minimum, 0)


class TokenSelectionTests(unittest.TestCase):
    """The strip's select listener is independent, so it can land too late.

    A click made a moment before Send is resolved against the metrics of the
    response being replaced. Publishing it would paint the old token's
    probabilities beside the new response and leave them there: every streaming
    frame after the opening one returns gr.skip() for these two outputs, so
    nothing would correct them until the user clicked again.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def respond(self, turns=()):
        """Stream one whole response and return its frames."""

        return list(app.chat("hi", list(turns), *SETTINGS))

    def assertDropped(self, payload):
        self.assertEqual(app.inspect_token("prompt")(payload, select(0)), (gr.skip(), gr.skip()))

    def initial_metrics_state(self):
        """The value a fresh session starts inspect_token()'s input with."""

        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "inspect"
        )
        (state_block,) = listener.inputs
        return state_block.value

    def test_a_selection_against_the_strip_on_screen_is_published(self):
        payload = self.respond()[-1][METRICS]
        detail, alternatives = app.inspect_token("prompt")(payload, select(0))
        self.assertIn("Token 1", detail)
        self.assertTrue(alternatives)

    def test_a_selection_from_the_previous_response_is_dropped(self):
        payload = self.respond()[-1][METRICS]
        self.respond()
        self.assertDropped(payload)

    def test_the_opening_frame_alone_drops_it(self):
        # The window the user hits is the first frame - the one that empties
        # the strip - not the end of the stream.
        payload = self.respond()[-1][METRICS]
        stream = app.chat("hi", [], *SETTINGS)
        try:
            next(stream)
            self.assertDropped(payload)
        finally:
            stream.close()

    def test_a_selection_made_mid_stream_survives_the_rest_of_it(self):
        # Later frames only append to the strip, so a token picked while the
        # response is still arriving is still on screen when it finishes.
        frames = self.respond()
        detail, _alternatives = app.inspect_token("prompt")(frames[1][METRICS], select(0))
        self.assertIn("Token 1", detail)

    def test_clear_drops_earlier_selections(self):
        payload = self.respond()[-1][METRICS]
        app.clear_chat()
        self.assertDropped(payload)

    def test_undo_drops_earlier_selections(self):
        final = self.respond()[-1]
        app.undo_last(final[TURNS])
        self.assertDropped(final[METRICS])

    def test_loading_a_conversation_drops_earlier_selections(self):
        saved, _status = app.save_conversation([make_turn("user", "hi")], "")
        final = self.respond()[-1]
        app.load_conversation(saved["value"], final[TURNS])
        self.assertDropped(final[METRICS])

    def test_editing_an_assistant_message_drops_earlier_selections(self):
        final = self.respond()[-1]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "Hello world", "value": "fixed"}
        )
        list(app.edit_message(event, "", final[TURNS], *SETTINGS))
        self.assertDropped(final[METRICS])

    def test_a_refused_send_leaves_the_selection_alone(self):
        # Nothing replaced the strip, so the panel beside it is still true.
        payload = self.respond()[-1][METRICS]
        list(app.chat("   ", [], *SETTINGS))
        detail, _alternatives = app.inspect_token("prompt")(payload, select(0))
        self.assertIn("Token 1", detail)

    def test_an_out_of_range_index_still_reports_the_token_as_gone(self):
        payload = self.respond()[-1][METRICS]
        detail, alternatives = app.inspect_token("prompt")(payload, select(99))
        self.assertIn("no longer available", detail)
        self.assertEqual(alternatives, [])

    def test_an_empty_strip_asks_for_a_selection(self):
        detail, alternatives = app.inspect_token("prompt")(app.empty_metrics(), select(0))
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(alternatives, [])

    def test_every_publisher_stamps_what_it_writes_to_the_state(self):
        """inspect_token() unpacks the payload, so every producer must pair it."""

        saved, _status = app.save_conversation([make_turn("user", "hi")], "")
        final = self.respond()[-1]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "Hello world", "value": "fixed"}
        )
        payloads = {
            "stream": final[METRICS],
            "clear": app.clear_chat()[3],
            "undo": app.undo_last(final[TURNS])[4],
            "load": app.load_conversation(saved["value"], final[TURNS])[4],
            "edit": list(app.edit_message(event, "", final[TURNS], *SETTINGS))[-1][
                METRICS
            ],
            "initial": self.initial_metrics_state(),
        }
        for name, payload in payloads.items():
            with self.subTest(publisher=name):
                generation, metrics = payload
                self.assertIsInstance(generation, int)
                self.assertIsInstance(metrics, list)


class AnalysisPanelTests(unittest.TestCase):
    """The prompt strip, the charts and the export follow the conversation.

    They measure one response, so every path that replaces or removes that
    response has to take them with it - otherwise the tiles keep reporting a
    perplexity for text that is no longer on screen, and the export button
    keeps offering it.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def respond(self):
        return list(app.chat("hi", [], *SETTINGS))

    def prompt_payload(self, frames):
        """The last prompt-metrics payload a stream actually published."""

        published = [
            frame[PROMPT_METRICS]
            for frame in frames
            if frame[PROMPT_METRICS] != gr.skip()
        ]
        return published[-1]

    def test_the_prompt_tokens_are_published_once(self):
        frames = self.respond()
        # frames[0] empties the panel, frames[1] is the first update.
        self.assertEqual(strip_of(frames[0][PROMPT_STRIP]), [])
        self.assertTrue(strip_of(frames[1][PROMPT_STRIP]))
        self.assertTrue(metrics_of(frames[1][PROMPT_METRICS]))
        self.assertIn("prompt tokens", frames[1][PROMPT_NOTE])
        for frame in frames[2:]:
            self.assertEqual(frame[PROMPT_STRIP], gr.skip())

    def test_both_strips_share_one_stamp(self):
        # inspect_token() drops a click whose stamp is not the current one, so
        # a prompt strip stamped separately would be unclickable from the
        # moment the response strip was stamped.
        frames = self.respond()
        prompt_payload = self.prompt_payload(frames)
        self.assertEqual(prompt_payload[0], frames[-1][METRICS][0])
        detail, _alternatives = app.inspect_token("prompt")(prompt_payload, select(0))
        self.assertIn("Prompt token", detail)

    def test_the_finished_response_is_exportable(self):
        frames = self.respond()
        self.assertEqual(frames[0][TRACE], {})
        trace = frames[-1][TRACE]
        self.assertEqual(trace["response"], "Hello world")
        self.assertEqual(len(trace["tokens"]), 3)
        self.assertIn("Exports are ready", frames[-1][STATUS])

    def test_a_failed_response_is_not_exportable(self):
        def failing(*_args, **_kwargs):
            yield GenerationUpdate(
                text="Hmm", metrics=[], load_id=runtime.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        runtime.MANAGER.generate = failing
        frames = list(app.chat("hi", [], *SETTINGS))
        self.assertEqual(frames[0][TRACE], {})
        for frame in frames[1:]:
            self.assertEqual(frame[TRACE], gr.skip())

    def test_clear_empties_the_panels(self):
        self.respond()
        result = app.clear_chat()
        prompt_strip, prompt_metrics, prompt_note = result[9:12]
        self.assertEqual(strip_of(prompt_strip), [])
        self.assertEqual(metrics_of(prompt_metrics), [])
        self.assertEqual(prompt_note, "")
        self.assertEqual(result[13], charts.EMPTY_CHART)
        self.assertEqual(result[14], {})

    def test_undo_empties_the_panels(self):
        final = self.respond()[-1]
        result = app.undo_last(final[TURNS])
        self.assertEqual(strip_of(result[10]), [])
        self.assertEqual(metrics_of(result[11]), [])
        self.assertEqual(result[14], charts.EMPTY_CHART)
        self.assertEqual(result[15], {})

    def test_loading_a_conversation_empties_the_panels(self):
        saved, _status = app.save_conversation([make_turn("user", "hi")], "")
        final = self.respond()[-1]
        result = app.load_conversation(saved["value"], final[TURNS])
        self.assertEqual(strip_of(result[10]), [])
        self.assertEqual(metrics_of(result[11]), [])
        self.assertEqual(result[14], charts.EMPTY_CHART)
        self.assertEqual(result[15], {})

    def test_an_assistant_edit_empties_the_panels(self):
        final = self.respond()[-1]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "Hello world", "value": "fixed"}
        )
        edited = list(app.edit_message(event, "", final[TURNS], *SETTINGS))[-1]
        self.assertEqual(strip_of(edited[PROMPT_STRIP]), [])
        self.assertEqual(metrics_of(edited[PROMPT_METRICS]), [])
        self.assertEqual(edited[SURPRISE], charts.EMPTY_CHART)
        self.assertEqual(edited[TRACE], {})

    def test_the_color_scale_repaints_both_strips(self):
        frames = self.respond()
        strip, score_strip, prompt_strip, caption = app.recolor(
            frames[-1][TURNS],
            frames[-1][METRICS],
            self.prompt_payload(frames),
            "Surprise",
        )
        # Two headings, the message, and one span per response token.
        self.assertEqual(len(strip["value"]), 6)
        self.assertEqual(len(score_strip["value"]), 3)
        self.assertTrue(prompt_strip["value"])
        self.assertTrue(caption)


class TokenViewTests(unittest.TestCase):
    """The conversation drawn as the tokens it is made of."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def respond(self, message="hi", turns=()):
        return list(app.chat(message, list(turns), *SETTINGS))[-1]

    def test_a_reply_is_drawn_token_by_token_under_its_heading(self):
        final = self.respond()
        spans, index = app.transcript_entries(final[TURNS], DEFAULT_COLOR_SCALE)
        texts = [text for text, _label in spans]
        self.assertEqual(texts[:2], ["\n\nYOU\n", "hi"])
        self.assertEqual(texts[2], "\n\nASSISTANT\n")
        tokens = final[TURNS][1]["tokens"]
        self.assertEqual(texts[3:], [m["display_text"] for m in tokens])
        # The headings and the message belong to a turn but to no token.
        self.assertEqual(index[:3], [(0, None), (0, None), (1, None)])
        self.assertEqual(index[3:], [(1, position) for position in range(len(tokens))])

    def test_an_empty_conversation_says_so(self):
        # An empty HighlightedText draws its color scale as a bare gradient
        # bar, which reads as a broken chart rather than an empty chat.
        self.assertEqual(
            app.transcript_value([], DEFAULT_COLOR_SCALE), app.EMPTY_TRANSCRIPT
        )
        # The placeholder belongs to no turn, so clicking it does nothing.
        self.assertIsNone(app.transcript_pick([], select(0)))

    def test_a_message_with_no_measurements_is_drawn_as_plain_text(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "typed")]
        spans, _index = app.transcript_entries(turns, DEFAULT_COLOR_SCALE)
        self.assertEqual([label for _text, label in spans], [None] * 4)
        self.assertIn(("typed", None), spans)

    def test_reasoning_without_measurements_is_drawn_before_the_answer(self):
        turns = [make_turn("assistant", "answer", "thinking")]
        spans, _index = app.transcript_entries(turns, DEFAULT_COLOR_SCALE)
        self.assertEqual(
            [text for text, _label in spans], ["\n\nASSISTANT\n", "thinking\n", "answer"]
        )

    def test_the_scale_decides_the_colors(self):
        final = self.respond()
        by_rank = app.transcript_value(final[TURNS], "Raw rank")
        by_surprise = app.transcript_value(final[TURNS], "Surprise")
        self.assertEqual(
            [text for text, _label in by_rank], [text for text, _label in by_surprise]
        )
        self.assertIn(
            app.COLOR_SCALES["Surprise"].labels[0],
            [label for _text, label in by_surprise],
        )

    def test_clicking_a_token_publishes_it(self):
        final = self.respond()
        turns = final[TURNS]
        detail, rows, selection, target, pick = app.select_transcript_token(
            turns, final[METRICS], token_span(turns, 1)
        )
        self.assertIn("Token 2", detail)
        self.assertTrue(rows)
        self.assertEqual(selection["turn"], 1)
        self.assertEqual(selection["index"], 1)
        # The reply on screen is the one the panel describes, so its layers
        # can be read.
        self.assertEqual(target, {"generation": final[METRICS][0], "strip": "response", "index": 1})
        self.assertIsNone(pick)

    def test_a_click_overtaken_by_a_conversation_change_is_dropped(self):
        """A click queued just before Retry, Undo, Clear or a switch.

        Gradio resolves a listener's inputs when it gets round to the event,
        so the handler is handed the conversation as it was. Answering would
        land a token from the replaced reply on top of the reset frame that
        removed it, and every later streaming frame skips those outputs, so it
        would stay there.
        """

        final = self.respond()
        turns, stamp = final[TURNS], final[METRICS]
        app.new_metrics_generation()  # what the reset frame mints

        published = app.select_transcript_token(turns, stamp, token_span(turns, 1))
        self.assertEqual(published, (gr.skip(),) * 5)

    def test_a_click_made_against_the_panel_on_screen_is_published(self):
        final = self.respond()
        turns = final[TURNS]
        detail, _rows, selection, _target, _pick = app.select_transcript_token(
            turns, final[METRICS], token_span(turns, 1)
        )
        self.assertIn("Token 2", detail)
        self.assertIsNotNone(selection)

    def test_clicking_a_heading_empties_the_panel(self):
        final = self.respond()
        turns = final[TURNS]
        spans, index = app.transcript_entries(turns, DEFAULT_COLOR_SCALE)
        detail, rows, selection, target, pick = app.select_transcript_token(
            turns, final[METRICS], select(index.index((0, None)))
        )
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(rows, [])
        self.assertIsNone(selection)
        self.assertIsNone(target)
        self.assertIsNone(pick)

    def test_an_earlier_reply_offers_no_layer_readout(self):
        # The inspector rebuilds the model's input from the prompt ids
        # published with the reply on screen, and an earlier reply's prompt is
        # not on screen to rebuild from.
        first = self.respond()
        second = self.respond("again", first[TURNS])
        turns = second[TURNS]
        _detail, _rows, selection, target, _pick = app.select_transcript_token(
            turns, second[METRICS], token_span(turns, 1, turn=1)
        )
        self.assertEqual(selection["turn"], 1)
        self.assertIsNone(target)

    def test_a_reply_from_another_load_says_so_when_clicked(self):
        final = self.respond()
        runtime.MANAGER.load_count += 1
        detail, _rows, selection, _target, _pick = app.select_transcript_token(
            final[TURNS], final[METRICS], token_span(final[TURNS], 1)
        )
        self.assertIn(app.BRANCH_MODEL_CHANGED, detail)
        # Still published: the numbers describe what the model did produce,
        # and only the branch needs the weights back.
        self.assertIn("Token 2", detail)
        self.assertEqual(selection["turn"], 1)

    def test_a_retried_reply_does_not_inherit_the_old_selection(self):
        """The reply, not the token ID, is what a selection is checked against.

        Two samples of one prompt share their opening tokens far more often
        than not, so a token ID alone would let a click made against the reply
        Retry replaced branch the new one at a token the reader never saw.
        """

        final = self.respond()
        selected = click_token(final, 1)
        retried = list(app.retry_last("", final[TURNS], *SETTINGS))[-1]

        # Same conversation shape, same token IDs, different reply.
        self.assertEqual(
            [m["token_id"] for m in retried[TURNS][1]["tokens"]],
            [m["token_id"] for m in final[TURNS][1]["tokens"]],
        )
        self.assertNotEqual(
            retried[TURNS][1]["metrics_generation"],
            final[TURNS][1]["metrics_generation"],
        )
        self.assertIsNone(app.selected_metric(retried[TURNS], selected))
        frames = list(app.branch_from(selected, "", retried[TURNS], *SETTINGS))
        self.assertEqual(frames[0][STATUS], app.BRANCH_UNAVAILABLE)

    def test_an_older_reply_keeps_its_own_stamp(self):
        # The stamp compared is the turn's own, so a reply that is no longer
        # the newest stays selectable for as long as it is on screen.
        first = self.respond()
        second = self.respond("again", first[TURNS])
        selected = click_token(second, 1, turn=1)
        self.assertIsNotNone(app.selected_metric(second[TURNS], selected))

    def test_scoring_disarms_the_branch_it_hid(self):
        # Score text resets the detail panel, and a branch the reader can no
        # longer see must not stay waiting on the button.
        final = self.respond()
        detail, _rows, selection, _target, _pick = app.select_transcript_token(
            final[TURNS], final[METRICS], token_span(final[TURNS], 1)
        )
        self.assertIsNotNone(selection)
        scored = list(app.score_text("", "hi", False, DEFAULT_COLOR_SCALE))[-1]
        self.assertEqual(scored[9], app.NO_TOKEN_SELECTED)
        self.assertIsNone(scored[11])
        self.assertIsNone(scored[12])

    def test_the_scored_strip_keeps_its_own_measurements(self):
        """The inspector's state moves on to the next reply; the strip's does not.

        The scored passage stays drawn while a reply is generated beside it,
        so repainting it from the inspector's state would paint scored spans
        with the reply's tokens.
        """

        scored = list(app.score_text("", "hi", False, DEFAULT_COLOR_SCALE))[-1]
        score_state = scored[2]
        self.assertEqual(score_state, scored[1])
        final = self.respond()

        # The inspector now describes the reply; the strip's state still
        # describes the passage.
        self.assertNotEqual(final[METRICS][0], score_state[0])
        strip, score_strip, _prompt, _caption = app.recolor(
            final[TURNS], score_state, app.empty_metrics(), "Surprise"
        )
        self.assertEqual(
            len(score_strip["value"]), len(metrics_of(score_state))
        )
        self.assertEqual(
            strip["value"], app.transcript_value(final[TURNS], "Surprise")
        )

    def test_scored_text_has_a_strip_of_its_own(self):
        # The Score text tab has no conversation to paint, and the
        # conversation must not be overwritten by a passage scored beside it.
        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "score_text"
        )
        self.assertEqual(listener.outputs[0].elem_id, "score-strip")

    def test_the_toggle_swaps_the_two_views(self):
        final = self.respond()
        chatbot, strip = app.show_token_view(True, final[TURNS], DEFAULT_COLOR_SCALE)
        self.assertEqual(chatbot, gr.update(visible=False))
        self.assertFalse(strip["visible"] is False)
        self.assertEqual(
            strip["value"], app.transcript_value(final[TURNS], DEFAULT_COLOR_SCALE)
        )
        chatbot, strip = app.show_token_view(False, final[TURNS], DEFAULT_COLOR_SCALE)
        self.assertEqual(chatbot, gr.update(visible=True))
        self.assertEqual(strip, gr.update(visible=False))

    def test_a_click_in_the_token_view_is_a_fork_point(self):
        # Fork works from a chatbot message, so a click here is translated
        # into the message it would have come from.
        final = self.respond()
        turns = final[TURNS]
        selected = app.remember_transcript_message(turns, token_span(turns, 1))
        messages, _index = display_messages(turns)
        self.assertEqual(messages[selected["index"]]["content"], selected["content"])
        self.assertEqual(app.selected_turn(turns, selected)[0], 1)

    def test_the_measurements_never_reach_the_saved_file(self):
        # They are the reason the file could not hold them: it is rewritten on
        # every streaming frame.
        final = self.respond()
        self.assertTrue(final[TURNS][1]["tokens"])
        entries = turn_entries(final[TURNS])
        self.assertEqual(
            [sorted(entry) for entry in entries],
            [
                ["content", "reasoning", "role"],
                sorted(["content", "reasoning", "role", "model", "prompt_tokens", "generated_tokens"]),
            ],
        )


class CancellationTests(unittest.TestCase):
    """What the Stop button does: Gradio closes the running generator."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def test_closing_mid_stream_releases_the_model_lock(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        next(stream)
        stream.close()

        acquired = runtime.MANAGER._lock.acquire(blocking=False)
        self.assertTrue(acquired, "the model lock survived cancellation")
        runtime.MANAGER._lock.release()

    def test_the_partial_response_is_kept(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        frame = next(stream)
        stream.close()

        self.assertEqual(frame[TURNS][0]["content"], "hi")
        self.assertTrue(frame[TURNS][1]["content"])
        self.assertGreater(len(frame[STRIP]), 0)

    def test_stopping_inside_a_think_block_closes_the_reasoning(self):
        runtime.MANAGER = loaded_manager([0, 2, 3], THINK_PIECES, THINK_EOS)
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        frame = next(stream)
        stream.close()

        self.assertFalse(frame[TURNS][1]["reasoning_closed"])
        messages, turns, _strip, _send, _stop, status = app.stop_generation(
            frame[TURNS]
        )
        self.assertTrue(turns[1]["reasoning_closed"])
        thoughts = [m for m in messages if m.get("metadata", {}).get("title")]
        self.assertEqual(thoughts[0]["metadata"]["status"], "done")
        self.assertEqual(status, "Stopped. The partial response was kept.")

    def test_stopping_before_any_token_drops_the_empty_turn(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "")]
        messages, remaining, _strip, _send, _stop, status = app.stop_generation(
            turns
        )
        self.assertEqual([turn["role"] for turn in remaining], ["user"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(status, "Stopped before the model produced anything.")


class UndoTests(unittest.TestCase):
    def test_undo_removes_the_exchange_and_restores_the_message(self):
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]
        result = app.undo_last(turns)
        self.assertEqual(len(result), UNDO_OUTPUTS)
        self.assertEqual(result[0], "two")
        self.assertEqual([turn["content"] for turn in result[2]], ["one", "first"])

    def test_undo_clears_the_token_panel(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        (
            _prompt,
            _messages,
            _turns,
            strip,
            metrics,
            _status,
            detail,
            alts,
            send,
            stop,
            *_panels,
        ) = app.undo_last(turns)
        self.assertEqual(painted(strip), [])
        self.assertEqual(metrics_of(metrics), [])
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(alts, [])
        # Undo cancels the generator, which then never reaches its final yield.
        self.assertEqual((send, stop), app.send_stop_buttons(False))

    def test_undo_with_nothing_to_remove_keeps_the_token_panel(self):
        turns = [make_turn("assistant", "orphan")]
        result = app.undo_last(turns)
        self.assertEqual(result[5], "There is nothing to undo.")
        for index in (4, 6, 7):
            self.assertEqual(result[index], gr.skip())
        # The token view is redrawn rather than skipped: this path finalizes
        # the turn a cancelled generator left behind, which can drop it.
        self.assertEqual(
            strip_of(result[3]), app.transcript_value(turns, DEFAULT_COLOR_SCALE)
        )
        # The cancel fires on the click, so even this path must undo the swap.
        self.assertEqual(result[8:10], app.send_stop_buttons(False))

    def test_undo_with_nothing_to_remove_finalizes_a_cancelled_turn(self):
        # Undo cancels the generator, so even the path that removes nothing has
        # to close the reasoning block the generator never got to close.
        turns = [
            {
                "role": "assistant",
                "content": "orphan",
                "reasoning": "half a thought",
                "reasoning_closed": False,
            }
        ]
        result = app.undo_last(turns)
        self.assertEqual(result[5], "There is nothing to undo.")
        self.assertTrue(result[2][-1]["reasoning_closed"])
        self.assertFalse(turns[-1]["reasoning_closed"])

    def test_undo_on_an_empty_conversation(self):
        result = app.undo_last([])
        self.assertEqual(result[2], [])
        self.assertEqual(result[5], "There is nothing to undo.")

    def test_undo_from_a_chatbot_event(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first", "thought")]
        event = gr.UndoData(None, {"index": 1, "value": "thought"})
        result = app.undo_message(event, turns)
        self.assertEqual(result[0], "one")
        self.assertEqual(result[2], [])


class ClearCancelsGenerationTests(unittest.TestCase):
    """Clear has to stop the generator before it empties the conversation.

    A surviving ``generate_reply`` owns a private copy of the in-progress turns
    and writes it back to the chatbot and the state on its next yield, which
    would resurrect the conversation that was just cleared.
    """

    def cancelled_by(self, demo, name):
        """Event indices that the listener triggering ``name`` cancels."""

        trigger = next(
            fn.targets[0]
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        )
        return {
            index
            for fn in demo.fns.values()
            if fn.targets == [trigger]
            for index in fn.cancels
        }

    def test_clear_cancels_the_same_events_as_stop(self):
        demo = app.build_app()
        stopped = self.cancelled_by(demo, "stop_generation")
        self.assertTrue(stopped, "Stop no longer cancels the running generators")
        self.assertEqual(self.cancelled_by(demo, "clear_chat"), stopped)

    def test_clear_restores_the_send_button(self):
        # Cancelling means generate_reply never reaches its final yield, so
        # Clear has to swap the buttons back itself.
        result = app.clear_chat()
        self.assertEqual(len(result), CLEAR_OUTPUTS)
        self.assertEqual(result[5], gr.update(visible=True))
        self.assertEqual(result[6], gr.update(visible=False))
        self.assertEqual(result[7], app.NO_TOKEN_SELECTED)
        self.assertEqual(result[8], [])


class SaveLoadTests(unittest.TestCase):
    def test_save_then_load_round_trips(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "there", "thought")]
        update, status = app.save_conversation(turns, "Be terse.")
        self.assertIn("Saved 2 messages", status)

        (
            messages,
            restored,
            system_prompt,
            strip,
            _metrics,
            load_status,
            detail,
            alternatives,
            send,
            stop,
            *_panels,
        ) = app.load_conversation(update["value"], [make_turn("user", "stale")])
        self.assertEqual(restored, turns)
        self.assertEqual(system_prompt, "Be terse.")
        self.assertEqual(len(messages), 3)
        # A saved file holds the text and the counts, not the measurements, so
        # the loaded conversation comes back as plain text in the token view.
        self.assertEqual(painted(strip), [])
        self.assertEqual(
            strip_of(strip), app.transcript_value(restored, DEFAULT_COLOR_SCALE)
        )
        self.assertIn("Loaded 2 messages", load_status)
        # The previous conversation's selected token goes with it.
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(alternatives, [])
        # Loading cancels any running generation, so Send must come back.
        self.assertEqual((send, stop), app.send_stop_buttons(False))

    def test_two_saves_never_share_a_path(self):
        # The timestamp only resolves to the second, and sessions share the
        # upload folder, so the later write would silently overwrite the first.
        turns = [make_turn("user", "hi")]
        first, _ = app.save_conversation(turns, "")
        second, _ = app.save_conversation(turns, "")
        self.assertNotEqual(first["value"], second["value"])

    def test_saving_an_empty_conversation_is_refused(self):
        update, status = app.save_conversation([], "")
        self.assertFalse(update["visible"])
        self.assertIn("nothing to save", status)

    def test_a_saved_conversation_is_readable_by_its_owner_alone(self):
        # The upload folder is shared - on Linux it is /tmp/gradio, which every
        # account on the machine can read - and a transcript is the reader's own
        # writing. write_trace_export() writes its export the same way. The
        # permissive umask stands in for a host that would otherwise have let
        # the file be created world-readable.
        self.addCleanup(os.umask, os.umask(0))
        update, _status = app.save_conversation([make_turn("user", "hi")], "")

        mode = Path(update["value"]).stat().st_mode
        self.assertEqual(stat.S_IMODE(mode), 0o600)

    def test_loading_a_bad_file_reports_the_problem(
        self,
    ):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        result = app.load_conversation("/nonexistent/conversation.json", turns)
        self.assertIn("Could not load that file", result[5])
        self.assertEqual(len(result), LOAD_OUTPUTS)
        # The conversation survives a bad file, and the token panel that
        # describes it is left alone rather than blanked.
        self.assertEqual([turn["content"] for turn in result[1]], ["one", "first"])
        for index in (2, 3, 4, 6, 7):
            self.assertIsInstance(result[index], gr.skip().__class__)
        self.assertEqual(result[8:10], app.send_stop_buttons(False))

    def test_loading_nothing_keeps_the_conversation(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        result = app.load_conversation(None, turns)
        self.assertEqual(result[5], "No file chosen.")
        self.assertEqual(len(result), LOAD_OUTPUTS)
        self.assertEqual([turn["content"] for turn in result[1]], ["one", "first"])
        for index in (2, 3, 4, 6, 7):
            self.assertIsInstance(result[index], gr.skip().__class__)
        self.assertEqual(result[8:10], app.send_stop_buttons(False))

    def test_a_failed_load_finalizes_the_cancelled_turn(self):
        # Uploading a bad file mid-stream cancels the generator, which then
        # never closes its own reasoning block; the accordion would spin for
        # the rest of the session.
        turns = [
            make_turn("user", "one"),
            {
                "role": "assistant",
                "content": "",
                "reasoning": "half a thought",
                "reasoning_closed": False,
            },
        ]
        result = app.load_conversation("/nonexistent/conversation.json", turns)
        self.assertTrue(result[1][-1]["reasoning_closed"])
        self.assertEqual(result[0][-1]["metadata"]["status"], "done")

    def test_a_failed_load_drops_a_turn_cancelled_before_any_tokens(self):
        turns = [
            make_turn("user", "one"),
            {
                "role": "assistant",
                "content": "",
                "reasoning": "",
                "reasoning_closed": False,
            },
        ]
        result = app.load_conversation(None, turns)
        self.assertEqual([turn["role"] for turn in result[1]], ["user"])

    def test_a_failed_load_does_not_mutate_the_state_it_was_given(self):
        turns = [
            make_turn("user", "one"),
            {
                "role": "assistant",
                "content": "partial",
                "reasoning": "",
                "reasoning_closed": False,
            },
        ]
        app.load_conversation(None, turns)
        self.assertFalse(turns[-1]["reasoning_closed"])


class BusyRefusalTests(unittest.TestCase):
    """A second generation is refused outright, not queued behind the first.

    Gradio captures a listener's inputs when the click is queued, so a Retry or
    an Edit that waited for the model lock and then ran would rebuild the
    conversation from a snapshot older than everything the first generation
    wrote - the "new question" the user sent seconds earlier simply disappears.
    Cancelling or serializing the second handler does not help; only refusing
    to start it does.

    The refusal must therefore write neither the chatbot nor the conversation
    state: idle_state() returns copy_turns(turns), which is exactly the stale
    snapshot, so using it here would cause the overwrite it is guarding
    against.
    """

    class HeldLock:
        """A generation flag that reads as held but never blocks.

        Really acquiring runtime.MANAGER._generating would model a running
        generation more literally, but then deleting a refusal would deadlock
        these tests instead of failing them. Reporting the flag as held and
        every reservation as lost leaves the manager otherwise usable, so a
        missing refusal shows up as a full stream of frames - a plain
        assertion failure.

        Both refusals read this: the early MANAGER.busy check in the handler
        and, behind it, generate_reply()'s reservation.
        """

        def locked(self):
            return True

        def acquire(self, blocking=True):
            return False

        def release(self):  # pragma: no cover - a failed acquire never pairs
            raise AssertionError("released a reservation that was never taken")

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)
        # ModelManager.busy reads this flag, so the real property is exercised.
        runtime.MANAGER._generating = self.HeldLock()
        self.assertTrue(runtime.MANAGER.busy)

    def turns(self):
        """The conversation as it looked when the second click was queued."""

        return [make_turn("user", "old q"), make_turn("assistant", "old a")]

    def assert_refused(self, stream):
        frames = list(stream)
        self.assertEqual(len(frames), 1)
        (frame,) = frames
        self.assertEqual(len(frame), CHAT_OUTPUTS)
        self.assertEqual(frame[STATUS], app.BUSY_STATUS)
        # The chatbot and the conversation state are the two outputs that would
        # carry the stale snapshot, so they are the ones that must be skipped.
        # The buttons go with them: the refusal tells the user to press Stop,
        # so it must not be the thing that hides Stop. Only the generation that
        # owns the slot drives that pair.
        for index in (
            CHATBOT,
            TURNS,
            PROMPT,
            STRIP,
            METRICS,
            SEED,
            DETAIL,
            ALTS,
            SEND,
            STOP,
        ):
            # gr.skip() and gr.update() are both plain dicts, so this compares
            # values: an isinstance check here passes for either of them.
            self.assertEqual(frame[index], gr.skip())
        return frame

    def test_sending_while_generating_is_refused(self):
        self.assert_refused(app.chat("new question", self.turns(), *SETTINGS))

    def test_an_empty_send_is_refused_before_its_own_complaint(self):
        """ "Enter a message first." also republishes the turns, so it waits."""

        self.assert_refused(app.chat("   ", self.turns(), *SETTINGS))

    def test_retry_is_refused(self):
        self.assert_refused(app.retry_last("", self.turns(), *SETTINGS))

    def test_a_retry_with_nothing_to_retry_is_refused(self):
        """ "There is nothing to retry." would write the stale turns too."""

        self.assert_refused(app.retry_last("", [], *SETTINGS))

    def test_the_chatbot_retry_button_is_refused(self):
        event = gr.RetryData(None, {"index": 1, "value": "old a"})
        self.assert_refused(app.retry_message(event, "", self.turns(), *SETTINGS))

    def test_regenerating_from_a_position_is_refused(self):
        self.assert_refused(app.regenerate_from(0, "", self.turns(), *SETTINGS))

    def test_editing_a_user_message_is_refused(self):
        event = gr.EditData(
            None, {"index": 0, "previous_value": "old q", "value": "edited"}
        )
        self.assert_refused(app.edit_message(event, "", self.turns(), *SETTINGS))

    def test_editing_an_assistant_message_is_refused(self):
        """The assistant branch never generates, but it still rewrites turns."""

        event = gr.EditData(
            None, {"index": 1, "previous_value": "old a", "value": "fixed"}
        )
        self.assert_refused(app.edit_message(event, "", self.turns(), *SETTINGS))

    def test_an_edit_of_a_missing_message_is_refused(self):
        event = gr.EditData(None, {"index": 99, "previous_value": "gone", "value": "x"})
        self.assert_refused(app.edit_message(event, "", self.turns(), *SETTINGS))


class LoadRefusalTests(unittest.TestCase):
    """A load has the model too, and a reply must not be admitted beside one.

    The generation slot and the load claim used to be unrelated things, so a
    Send arriving while the chat page's switcher - or either of the Models
    page's buttons - was loading was accepted. It did not run beside the
    load: it waited on the model lock and then answered from whatever the
    load had brought in, under a badge naming the model the reader had
    asked the question of.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)
        _checked_id, self.claim = runtime.MANAGER.claim_exclusive_load("org/other").claim
        self.addCleanup(runtime.MANAGER.release_load, self.claim)

    def turns(self):
        return [make_turn("user", "old q"), make_turn("assistant", "old a")]

    def assert_refused(self, stream):
        frames = list(stream)
        self.assertEqual(len(frames), 1)
        (frame,) = frames
        self.assertEqual(frame[STATUS], app.LOADING_STATUS)
        # As for a running generation: the two outputs that would carry the
        # stale snapshot are skipped rather than republished.
        self.assertEqual(frame[TURNS], gr.skip())
        self.assertEqual(frame[CHATBOT], gr.skip())

    def test_sending_while_a_model_loads_is_refused(self):
        self.assert_refused(app.chat("new question", self.turns(), *SETTINGS))

    def test_retrying_while_a_model_loads_is_refused(self):
        self.assert_refused(app.retry_last("", self.turns(), *SETTINGS))

    def test_regenerating_while_a_model_loads_is_refused(self):
        self.assert_refused(app.regenerate_from(0, "", self.turns(), *SETTINGS))

    def test_editing_while_a_model_loads_is_refused(self):
        event = gr.EditData(
            None, {"index": 1, "previous_value": "old a", "value": "fixed"}
        )
        self.assert_refused(app.edit_message(event, "", self.turns(), *SETTINGS))

    def test_the_refusal_does_not_point_at_a_stop_button(self):
        # There is no Stop for a load, so the generating wording would send
        # the reader looking for a button that is not on the page.
        self.assertNotIn("Stop", app.LOADING_STATUS)
        self.assertIn("Stop", app.BUSY_STATUS)

    def test_a_reply_is_admitted_again_once_the_load_ends(self):
        runtime.MANAGER.release_load(self.claim)

        frames = list(app.chat("new question", self.turns(), *SETTINGS))

        self.assertGreater(len(frames), 1, "still refusing after the load")
        self.assertNotEqual(frames[-1][STATUS], app.LOADING_STATUS)

    def test_a_load_that_has_emptied_memory_is_still_named_as_a_load(self):
        # The claim stands for the whole load, but the weights come out
        # before the new ones go in, so for most of it nothing is loaded.
        # These handlers cannot claim ahead of that check - generate_reply()
        # claims further down and a claim here would refuse its own reply -
        # so they read what is in memory first and ask what has the model
        # only when it is empty. Read the other way round, the whole of the
        # load answered "Download and load a model first."
        with mock.patch.object(
            type(runtime.MANAGER), "loaded", property(lambda self: False)
        ):
            self.assert_refused(app.chat("new question", self.turns(), *SETTINGS))
            self.assert_refused(app.retry_last("", self.turns(), *SETTINGS))
            self.assert_refused(
                app.regenerate_from(0, "", self.turns(), *SETTINGS)
            )
            event = gr.EditData(
                None, {"index": 0, "previous_value": "old q", "value": "new q"}
            )
            self.assert_refused(app.edit_message(event, "", self.turns(), *SETTINGS))

    def test_an_empty_machine_still_says_to_load_a_model(self):
        # The other side of that order: with no load claimed, an empty
        # memory is what it looks like and the advice is the right answer.
        runtime.MANAGER.release_load(self.claim)

        with mock.patch.object(
            type(runtime.MANAGER), "loaded", property(lambda self: False)
        ):
            frames = list(app.chat("new question", self.turns(), *SETTINGS))

        self.assertEqual(frames[-1][STATUS], app.NO_MODEL_STATUS)


class BusyFlagTests(unittest.TestCase):
    """The flag the refusal reads has to follow a real generation."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def test_an_idle_manager_is_not_busy(self):
        self.assertFalse(runtime.MANAGER.busy)

    def streaming(self, message="hi", turns=None):
        """A generation parked on its first frame, closed when the test ends."""

        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat(
            message, turns if turns is not None else [], *settings.values()
        )
        self.addCleanup(stream.close)
        return stream

    def test_the_first_frame_already_marks_the_manager_busy(self):
        """The window this closes: the "Generating…" frame is a suspension point.

        Gradio does not resume a streaming handler until it has serialized that
        frame and shipped it to the browser, so MANAGER.generate() - and the
        lock it used to be the only thing to take - is a network round trip
        away. Any click landing in there found an idle manager.
        """

        stream = self.streaming()
        self.assertFalse(runtime.MANAGER.busy)
        next(stream)
        self.assertTrue(runtime.MANAGER.busy, "the first frame left the slot free")

    def test_the_manager_is_busy_while_streaming(self):
        stream = self.streaming()
        next(stream)
        next(stream)
        self.assertTrue(runtime.MANAGER.busy)
        # Stop closes the generator, which unwinds generate_reply() and frees it.
        stream.close()
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_finished_generation_leaves_the_manager_free(self):
        list(app.chat("hi", [], *SETTINGS))
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_failed_generation_leaves_the_manager_free(self):
        """The reservation outlives the runtime, so its release must too.

        Replacing generate() takes the manager's own bookkeeping out of the
        picture: what frees the slot here is generate_reply()'s finally.
        """

        def failing(*_args, **_kwargs):
            yield GenerationUpdate(
                text="Hmm", metrics=[], load_id=runtime.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        runtime.MANAGER.generate = failing
        stream = app.chat("hi", [], *SETTINGS)
        next(stream)
        self.assertTrue(runtime.MANAGER.busy)
        list(stream)
        self.assertFalse(runtime.MANAGER.busy)

    def test_cancelling_the_first_frame_frees_the_slot(self):
        """Stop before a single token: the reservation is already outstanding."""

        stream = self.streaming()
        next(stream)
        stream.close()
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_cancelled_generation_does_not_wedge_the_app(self):
        """The regression to fear: every later Send refused, forever."""

        stream = self.streaming()
        next(stream)
        stream.close()

        final = list(app.chat("again", [], *SETTINGS))[-1]
        self.assertNotEqual(final[STATUS], app.BUSY_STATUS)
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        self.assertFalse(runtime.MANAGER.busy)

    def test_cancelling_still_releases_the_model_lock(self):
        stream = self.streaming()
        next(stream)
        stream.close()

        acquired = runtime.MANAGER._lock.acquire(blocking=False)
        self.assertTrue(acquired, "the model lock survived cancellation")
        runtime.MANAGER._lock.release()

    def test_a_direct_generate_call_reserves_the_slot_itself(self):
        """Nothing above generate() has reserved anything here.

        The runtime is used directly by tests and could be used directly by a
        non-streaming caller, so it still has to claim - and free - the slot
        when it finds it available, without deadlocking against the reservation
        generate_reply() normally holds on its behalf.
        """

        stream = runtime.MANAGER.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=8192,
            seed=42,
        )
        self.addCleanup(stream.close)
        next(stream)
        self.assertTrue(runtime.MANAGER.busy)
        stream.close()
        self.assertFalse(runtime.MANAGER.busy)


class FirstFrameWindowTests(unittest.TestCase):
    """A second click landing before the first frame is answered is refused.

    Round 7 refused it by testing MANAGER.busy on entry, which is a check the
    running generation had not yet earned: it publishes "Generating…" and
    suspends there, and Gradio only resumes it - and only then reaches the
    model - once the browser has the frame. A Retry or an Edit arriving inside
    that round trip sailed through and rebuilt the conversation from the
    snapshot Gradio captured when its own click was queued, erasing the
    exchange the running generation had just added.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def stale(self):
        """The conversation as it looked before the running generation began."""

        return [make_turn("user", "old q"), make_turn("assistant", "old a")]

    def parked_at_the_first_frame(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("new question", self.stale(), *settings.values())
        self.addCleanup(stream.close)
        frame = next(stream)
        self.assertEqual(len(frame), CHAT_OUTPUTS)
        return stream, frame

    def assert_refused(self, competing):
        frames = list(competing)
        self.assertEqual(len(frames), 1)
        (frame,) = frames
        self.assertEqual(len(frame), CHAT_OUTPUTS)
        self.assertEqual(frame[STATUS], app.BUSY_STATUS)
        # The two outputs that would carry the stale snapshot.
        for index in (CHATBOT, TURNS):
            self.assertEqual(frame[index], gr.skip())
        # The refused click must not take the Stop button away from the
        # generation parked on its first frame.
        for index in (SEND, STOP):
            self.assertEqual(frame[index], gr.skip())

    def test_a_retry_in_the_window_is_refused(self):
        _stream, frame = self.parked_at_the_first_frame()
        self.assertEqual(frame[TURNS][2]["content"], "new question")
        self.assert_refused(app.retry_last("", self.stale(), *SETTINGS))

    def test_an_edit_in_the_window_is_refused(self):
        self.parked_at_the_first_frame()
        event = gr.EditData(
            None, {"index": 0, "previous_value": "old q", "value": "edited"}
        )
        self.assert_refused(app.edit_message(event, "", self.stale(), *SETTINGS))

    def test_a_second_send_in_the_window_is_refused(self):
        self.parked_at_the_first_frame()
        self.assert_refused(app.chat("another", self.stale(), *SETTINGS))

    def test_a_competing_retry_cannot_erase_the_new_question(self):
        """The harm, stated as an outcome rather than as a mechanism.

        The retry holds the conversation from before the running generation
        started. Publishing it at all - at any point in its life, refused or
        not - drops the question the user just sent.
        """

        self.parked_at_the_first_frame()
        for frame in app.retry_last("", self.stale(), *SETTINGS):
            published = frame[TURNS]
            if published == gr.skip():
                continue
            self.fail(
                "the competing retry published a conversation without the "
                "question the running generation had already added: "
                f"{[turn['content'] for turn in published]}"
            )


class EmptyResponseTests(unittest.TestCase):
    """A generation that succeeds but renders nothing must not leave a turn.

    The turn survives with neither answer nor reasoning, and the two
    transcripts then disagree: display_messages() draws a blank assistant
    bubble, model_messages() skips the turn entirely. The interface would be
    showing a reply the model never sees, and the next request would carry two
    user messages in a row into a template that requires alternation.

    Dropping the turn is the fix, not inventing an assistant slot: with no
    bubble drawn, "no reply" is what the screen shows and what the model is
    told, and consecutive user turns are then an accurate record.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def use(self, script, pieces=THINK_PIECES, eos_id=THINK_EOS):
        runtime.MANAGER = loaded_manager(script, pieces, eos_id)

    def reply(self, turns=None, message="hi"):
        frames = list(app.chat(message, turns if turns is not None else [], *SETTINGS))
        self.assertTrue(frames)
        for frame in frames:
            self.assertEqual(len(frame), CHAT_OUTPUTS)
        return frames[-1]

    def assert_no_reply(self, final):
        """The user turn stands alone, in the chatbot and in the state alike."""

        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user"])
        self.assertEqual([message["role"] for message in final[CHATBOT]], ["user"])
        self.assertNotIn("", [message["content"] for message in final[CHATBOT]])
        # The buttons still come back: the generation did finish.
        self.assertEqual(final[SEND], gr.update(visible=True))
        self.assertEqual(final[STOP], gr.update(visible=False))

    def test_a_hidden_eos_as_the_first_token_leaves_no_turn(self):
        self.use([THINK_EOS])
        self.assert_no_reply(self.reply())

    def test_a_whitespace_only_reply_leaves_no_turn(self):
        # split_reasoning() strips the answer, so these tokens render as
        # nothing at all even though the model really did emit them.
        self.use([0, 1, 2], ["   ", "\n\n", "<eos>"], 2)
        self.assert_no_reply(self.reply())

    def test_an_empty_reasoning_block_leaves_no_turn(self):
        # "<think></think>": a block opened and closed with nothing inside.
        self.use([0, 1, THINK_EOS])
        self.assert_no_reply(self.reply())

    def test_the_next_send_invents_no_assistant_slot(self):
        self.use([THINK_EOS])
        first = self.reply(message="one")
        second = self.reply(first[TURNS], message="two")

        self.assertEqual([turn["role"] for turn in second[TURNS]], ["user", "user"])
        displayed, _ = display_messages(second[TURNS])
        self.assertEqual([message["role"] for message in displayed], ["user", "user"])
        # The two transcripts agree, which is the whole point: no assistant
        # bubble on screen, no assistant slot in the request.
        self.assertEqual(
            model_messages(second[TURNS]),
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        )

    def test_a_reply_that_is_only_reasoning_still_survives(self):
        """The guard must not swallow a Think turn with an empty answer."""

        # "<think>Hello</think>" and then stop: reasoning, but no answer.
        self.use([0, 2, 1, THINK_EOS])
        final = self.reply()
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        self.assertEqual(final[TURNS][1]["reasoning"], "Hello")
        self.assertEqual(final[TURNS][1]["content"], "")
        self.assertTrue(final[TURNS][1]["reasoning_closed"])

    def test_an_ordinary_reply_is_untouched(self):
        self.use([2, 3, THINK_EOS])
        final = self.reply()
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        self.assertEqual(final[TURNS][1]["content"], "Hello world")
        self.assertTrue(final[TURNS][1]["reasoning_closed"])


class IdleRefusalButtonTests(unittest.TestCase):
    """Refusals that can only happen while idle still restore the buttons.

    The busy refusal skips the button outputs because a generation it does not
    own is driving them (see BusyRefusalTests). Every refusal below is reached
    only after the MANAGER.busy check has already passed, so no generation is
    running and nothing else will ever swap the buttons back - these have to
    do it themselves, exactly as Stop, Clear and Undo do.
    """

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def unloaded(self):
        runtime.MANAGER = self.original
        self.assertFalse(runtime.MANAGER.loaded)

    def assert_idle_refusal(self, stream, status):
        frames = list(stream)
        self.assertEqual(len(frames), 1)
        (frame,) = frames
        self.assertEqual(len(frame), CHAT_OUTPUTS)
        self.assertEqual(frame[STATUS], status)
        self.assertNotEqual(frame[STATUS], app.BUSY_STATUS)
        self.assertEqual(frame[SEND], gr.update(visible=True))
        self.assertEqual(frame[STOP], gr.update(visible=False))

    def turns(self):
        return [make_turn("user", "q"), make_turn("assistant", "a")]

    def test_an_empty_message_restores_the_buttons(self):
        self.assert_idle_refusal(
            app.chat("   ", self.turns(), *SETTINGS), "Enter a message first."
        )

    def test_sending_with_no_model_restores_the_buttons(self):
        self.unloaded()
        self.assert_idle_refusal(
            app.chat("hi", [], *SETTINGS), "Download and load a model first."
        )

    def test_nothing_to_retry_restores_the_buttons(self):
        self.assert_idle_refusal(
            app.retry_last("draft", [], *SETTINGS), "There is nothing to retry."
        )

    def test_retrying_with_no_model_restores_the_buttons(self):
        self.unloaded()
        self.assert_idle_refusal(
            app.retry_last("draft", self.turns(), *SETTINGS),
            "Download and load a model first.",
        )

    def test_editing_a_missing_message_restores_the_buttons(self):
        event = gr.EditData(None, {"index": 99, "previous_value": "gone", "value": "x"})
        self.assert_idle_refusal(
            app.edit_message(event, "", self.turns(), *SETTINGS),
            "That message is no longer available.",
        )

    def test_emptying_an_assistant_message_restores_the_buttons(self):
        event = gr.EditData(None, {"index": 1, "previous_value": "a", "value": "  "})
        self.assert_idle_refusal(
            app.edit_message(event, "", self.turns(), *SETTINGS),
            "An assistant message cannot be emptied.",
        )

    def test_emptying_a_user_message_restores_the_buttons(self):
        event = gr.EditData(None, {"index": 0, "previous_value": "q", "value": "  "})
        self.assert_idle_refusal(
            app.edit_message(event, "", self.turns(), *SETTINGS),
            "A user message cannot be empty.",
        )

    def test_editing_a_user_message_with_no_model_restores_the_buttons(self):
        self.unloaded()
        event = gr.EditData(None, {"index": 0, "previous_value": "q", "value": "new"})
        self.assert_idle_refusal(
            app.edit_message(event, "", self.turns(), *SETTINGS),
            "Download and load a model first.",
        )

    def test_an_accepted_assistant_edit_restores_the_buttons(self):
        event = gr.EditData(None, {"index": 1, "previous_value": "a", "value": "fixed"})
        self.assert_idle_refusal(
            app.edit_message(event, "", self.turns(), *SETTINGS),
            "Assistant message edited.",
        )


# The fork handlers publish this tuple, in this order.
(
    FORK_PROMPT,
    FORK_CHATBOT,
    FORK_TURNS,
    FORK_STATE,
    FORK_PICKER,
    FORK_STATUS,
    FORK_SEND,
    FORK_STOP,
    FORK_STRIP,
    FORK_METRICS,
    FORK_DETAIL,
    FORK_ALTS,
    FORK_PROMPT_STRIP,
    FORK_PROMPT_METRICS,
    FORK_PROMPT_NOTE,
    FORK_SUMMARY,
    FORK_SURPRISE,
    FORK_TRACE,
    FORK_SELECTED_TOKEN,
    FORK_BRANCH_PICK,
) = range(20)


def contents(turns):
    return [turn["content"] for turn in turns]


def names_of(list_update):
    """The branch names behind a conversation-list update's (label, name) choices."""

    return [name for _label, name in list_update["choices"]]


def labels_of(list_update):
    return [label for label, _name in list_update["choices"]]


def cell(row):
    """A click on one row of the alternatives table."""

    return gr.SelectData(None, {"index": [row, 1], "value": "x"})


class BranchFromTokenTests(unittest.TestCase):
    """Replay a response up to a token, swap in an alternative, and continue."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def respond(self):
        return list(app.chat("hi", [], *SETTINGS))

    def pick_alternative(self, final, strip_index=1, row=1, turn=-1):
        """Click a response token, then a row of its alternatives."""

        selected = click_token(final, strip_index, turn)
        detail, pick = app.choose_alternative(
            final[TURNS], final[METRICS], app.empty_metrics(), selected, cell(row)
        )
        return detail, pick

    def test_an_earlier_reply_can_be_branched(self):
        """The new capability: not only the newest reply carries its tokens.

        The branch replaces the reply it was taken from and everything after
        it, which is what makes it a different continuation of the
        conversation rather than an edit buried in the middle of one.
        """

        first = self.respond()[-1]
        second = list(app.chat("again", first[TURNS], *SETTINGS))[-1]
        self.assertEqual(len(second[TURNS]), 4)

        _detail, pick = self.pick_alternative(second, turn=1)
        self.assertEqual(pick["turn"], 1)
        last = list(app.branch_from(pick, "", second[TURNS], *SETTINGS))[-1]

        self.assertEqual([turn["role"] for turn in last[TURNS]], ["user", "assistant"])
        self.assertEqual(last[TURNS][0]["content"], "hi")
        self.assertIn("Branched at token 2", last[STATUS])
        replayed = metrics_of(last[METRICS])
        kept = second[TURNS][1]["tokens"]
        self.assertEqual(replayed[0]["token_id"], kept[0]["token_id"])
        self.assertEqual(replayed[1]["token_id"], pick["token_id"])

    def test_an_earlier_reply_can_be_branched_with_typed_text(self):
        first = self.respond()[-1]
        second = list(app.chat("again", first[TURNS], *SETTINGS))[-1]
        selected = click_token(second, 1, turn=1)
        last = list(
            app.branch_with_text(selected, "Hello", "", second[TURNS], *SETTINGS)
        )[-1]
        self.assertEqual([turn["role"] for turn in last[TURNS]], ["user", "assistant"])
        self.assertEqual(metrics_of(last[METRICS])[1]["token_id"], 2)  # "Hello"

    def test_a_reply_from_an_earlier_load_cannot_be_branched(self):
        first = self.respond()[-1]
        runtime.MANAGER.load_count += 1
        second = list(app.chat("again", first[TURNS], *SETTINGS))[-1]

        # The newest reply belongs to the load in memory; the one before it
        # does not, even though both are on screen and both carry tokens.
        self.assertEqual(app.branch_target(second[TURNS], click_token(second, 1))[0], 3)
        self.assertEqual(
            app.branch_target(second[TURNS], click_token(second, 1, turn=1)),
            app.BRANCH_MODEL_CHANGED,
        )

    def test_a_finished_response_carries_its_own_tokens(self):
        frames = self.respond()
        self.assertEqual(frames[0][TURNS][-1].get("tokens"), None)
        reply = frames[-1][TURNS][-1]
        self.assertEqual(
            [metric["token_id"] for metric in reply["tokens"]],
            [metric["token_id"] for metric in metrics_of(frames[-1][METRICS])],
        )
        self.assertEqual(reply["load_id"], runtime.MANAGER.load_id)
        self.assertEqual(reply["metrics_generation"], frames[-1][METRICS][0])

    def test_a_load_finishing_before_the_final_snapshot_cannot_claim_the_tokens(self):
        manager = runtime.MANAGER
        real_generate = manager.generate
        producing_load_id = manager.load_id

        def load_after_generation(*args, **kwargs):
            yield from real_generate(*args, **kwargs)
            # A load waiting on the model lock can finish as soon as the
            # runtime generator exits, before _stream_reply builds its final
            # snapshot.
            manager.load_count += 1

        manager.generate = load_after_generation
        final = self.respond()[-1]

        self.assertNotEqual(manager.load_id, producing_load_id)
        # The reply is tagged with the load that produced it, so the branch is
        # refused rather than replayed against different weights.
        self.assertEqual(final[TURNS][-1]["load_id"], producing_load_id)
        selected = click_token(final, 1)
        self.assertEqual(app.branch_target(final[TURNS], selected), app.BRANCH_MODEL_CHANGED)

    def test_a_response_token_is_remembered_for_the_table(self):
        final = self.respond()[-1]
        selected = click_token(final, 1)
        self.assertEqual(
            selected,
            {
                "source": "turn",
                "turn": 1,
                "index": 1,
                "at_generation": final[TURNS][1]["metrics_generation"],
                "at_token_id": metrics_of(final[METRICS])[1]["token_id"],
            },
        )

    def test_an_unscored_prompt_token_is_not_remembered(self):
        # The first prompt token has no prediction behind it, so there are no
        # alternatives for a table row to pair with. Remembering it would let
        # a row click pair with the response token remembered earlier.
        frames = self.respond()
        prompt_payload = frames[1][PROMPT_METRICS]
        self.assertIsNone(app.remember_strip_selection("prompt")(prompt_payload, select(0))[0])

    def test_choosing_an_alternative_readies_a_branch(self):
        final = self.respond()[-1]
        detail, pick = self.pick_alternative(final)
        metric = metrics_of(final[METRICS])[1]
        candidate = metric["top_candidates"][1]
        self.assertIn("Branch ready", detail)
        self.assertIn("Token 2", detail)
        self.assertEqual(pick["position"], 2)
        self.assertEqual(pick["token_id"], candidate["token_id"])
        self.assertEqual(pick["original_id"], metric["token_id"])

    def test_choosing_the_token_the_model_picked_offers_a_resample(self):
        final = self.respond()[-1]
        detail, pick = self.pick_alternative(final, row=0)
        self.assertIn("fresh", detail)
        self.assertEqual(pick["token_id"], pick["original_id"])

    def test_a_strip_without_a_conversation_cannot_be_branched(self):
        # Scored text draws the same strip and table, but there is no reply to
        # replace; the selection says which view it came from.
        final = self.respond()[-1]
        scored = score_known_passage()
        selected = app.remember_strip_selection("score")(scored[2], select(0))[0]
        detail, pick = app.choose_alternative(
            final[TURNS], scored[2], app.empty_metrics(), selected, cell(1)
        )
        self.assertIn(app.BRANCH_UNAVAILABLE, detail)
        self.assertIsNone(pick)

    def test_a_row_without_a_remembered_token_does_nothing(self):
        final = self.respond()[-1]
        detail, pick = app.choose_alternative(
            final[TURNS], final[METRICS], app.empty_metrics(), None, cell(0)
        )
        self.assertEqual(detail, gr.skip())
        self.assertIsNone(pick)

    def test_branching_replays_the_prefix_and_continues(self):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        frames = list(
            app.branch_from(
                pick,
                "", final[TURNS], *SETTINGS
            )
        )
        for frame in frames:
            self.assertEqual(len(frame), CHAT_OUTPUTS)
        last = frames[-1]
        metrics = metrics_of(last[METRICS])
        original = metrics_of(final[METRICS])
        # The kept token, then the alternative, then whatever the model added.
        self.assertEqual(metrics[0]["token_id"], original[0]["token_id"])
        self.assertEqual(metrics[1]["token_id"], pick["token_id"])
        self.assertGreater(len(metrics), 2)
        self.assertEqual([turn["role"] for turn in last[TURNS]], ["user", "assistant"])
        self.assertIn("Branched at token 2", frames[0][STATUS])
        self.assertIn("Branched at token 2", last[STATUS])
        self.assertEqual(last[TRACE]["sampling"]["forced_prefix_tokens"], 2)
        # The branch is itself a reply, so it can be branched again.
        self.assertEqual(last[TURNS][-1]["load_id"], runtime.MANAGER.load_id)

    def test_branching_preserves_literal_assistant_prefill_tags(self):
        runtime.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        _detail, pick = self.pick_alternative(original, strip_index=3, row=0)

        branched = list(
            app.branch_from(
                pick,
                "",
                original[TURNS],
                *settings.values(),
            )
        )[-1]

        self.assertEqual(branched[TURNS][-1]["reasoning"], "")
        self.assertTrue(
            branched[TURNS][-1]["content"].startswith("<think>Hello</think>")
        )
        self.assertTrue(
            all(
                metric.get("literal_prefill")
                for metric in metrics_of(branched[METRICS])[:3]
            )
        )

    def test_a_replacement_inside_the_prefill_is_not_literal(self):
        runtime.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        original_metrics = metrics_of(original[METRICS])
        pick = {
            "source": "turn",
            "turn": 1,
            "index": 1,
            "at_generation": original[TURNS][1]["metrics_generation"],
            "at_token_id": original_metrics[1]["token_id"],
            "position": 2,
            "token_id": THINK_EOS,
            "original_id": original_metrics[1]["token_id"],
            "text": "<eos>",
            "original": original_metrics[1]["text"],
        }

        branched = list(
            app.branch_from(
                pick,
                "",
                original[TURNS],
                *settings.values(),
            )
        )[-1]
        branched_metrics = metrics_of(branched[METRICS])

        self.assertEqual(branched[TURNS][-1]["content"], "<think>")
        self.assertEqual([m["token_id"] for m in branched_metrics], [0, THINK_EOS])
        self.assertTrue(branched_metrics[0]["literal_prefill"])
        self.assertNotIn("literal_prefill", branched_metrics[1])

    def test_the_branched_response_replaces_only_the_last_reply(self):
        first = self.respond()[-1]
        second = list(app.chat("again", first[TURNS], *SETTINGS))[-1]
        _detail, pick = self.pick_alternative(second)
        last = list(
            app.branch_from(
                pick,
                "", second[TURNS], *SETTINGS
            )
        )[-1]
        self.assertEqual(
            [turn["content"] for turn in last[TURNS][:3]],
            [turn["content"] for turn in second[TURNS][:3]],
        )
        self.assertEqual(len(last[TURNS]), 4)

    def test_a_pick_whose_token_has_moved_is_refused(self):
        # The pick names a turn and a token within it, and is checked against
        # the conversation it is used with rather than trusted. Rewriting the
        # reply by hand takes its measurements away, so the token the pick
        # names is no longer there.
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        edited = forget_measurements(final[TURNS], 1)
        frames = list(app.branch_from(pick, "", edited, *SETTINGS))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_UNAVAILABLE)
        self.assertEqual(frames[0][TURNS], edited)

    def test_branching_with_nothing_picked_explains_the_steps(self):
        final = self.respond()[-1]
        frames = list(
            app.branch_from(
                None,
                "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(frames[0][STATUS], app.BRANCH_HINT)

    def test_branching_is_refused_while_a_response_is_generating(self):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        self.assertTrue(runtime.MANAGER.reserve_generation())
        try:
            frames = list(
                app.branch_from(
                pick,
                "", final[TURNS], *SETTINGS
                )
            )
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(frames[0][STATUS], app.BUSY_STATUS)
        self.assertEqual(frames[0][TURNS], gr.skip())

    def test_a_stopped_response_is_branchable(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        frame = next(stream)
        stream.close()
        producing_load_id = frame[TURNS][-1]["load_id"]
        _messages, turns, *_rest = app.stop_generation(frame[TURNS])

        # The partial reply keeps the tokens it did produce and the load that
        # produced them, so it can be branched like a finished one.
        selection = {
            "source": "turn",
            "turn": 1,
            "index": 0,
            "at_generation": turns[1]["metrics_generation"],
            "at_token_id": turns[1]["tokens"][0]["token_id"],
        }
        self.assertEqual(app.branch_target(turns, selection)[0], 1)
        runtime.MANAGER.load_count += 1
        self.assertNotEqual(runtime.MANAGER.load_id, producing_load_id)
        self.assertEqual(
            app.branch_target(turns, selection), app.BRANCH_MODEL_CHANGED
        )

    def test_stopping_before_any_token_leaves_nothing_to_branch(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "")]
        _messages, remaining, *_rest = app.stop_generation(turns)
        self.assertEqual([turn["role"] for turn in remaining], ["user"])

    def branch_text(self, final, text, strip_index=1):
        selected = click_token(final, strip_index)
        return list(
            app.branch_with_text(
                selected,
                text,
                "",
                final[TURNS],
                *SETTINGS,
            )
        )

    def test_typed_text_replaces_the_clicked_token_and_continues(self):
        final = self.respond()[-1]
        original = metrics_of(final[METRICS])
        frames = self.branch_text(final, "Hello")
        for frame in frames:
            self.assertEqual(len(frame), CHAT_OUTPUTS)
        last = frames[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual(metrics[0]["token_id"], original[0]["token_id"])
        self.assertEqual(metrics[1]["token_id"], 2)  # "Hello"
        self.assertGreater(len(metrics), 2)
        self.assertIn("Branched at token 2", frames[0][STATUS])
        self.assertIn("'Hello'", last[STATUS])
        self.assertEqual(last[TRACE]["sampling"]["forced_prefix_tokens"], 2)
        self.assertTrue(last[TURNS][-1]["content"].startswith("HelloHello"))
        self.assertEqual(last[TURNS][-1]["load_id"], runtime.MANAGER.load_id)

    def test_typed_text_may_span_several_tokens(self):
        final = self.respond()[-1]
        last = self.branch_text(final, "Hello world")[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual([m["token_id"] for m in metrics[:3]], [2, 2, 3])
        self.assertEqual(last[TRACE]["sampling"]["forced_prefix_tokens"], 3)

    def test_typed_text_needs_no_alternative_pick(self):
        # The alternatives table is never touched; a clicked token is enough.
        final = self.respond()[-1]
        last = self.branch_text(final, " world", strip_index=0)[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual(metrics[0]["token_id"], 3)
        self.assertTrue(last[TURNS][-1]["content"].startswith(" world"))
        self.assertEqual(last[TRACE]["sampling"]["forced_prefix_tokens"], 1)

    def test_whitespace_before_a_typed_terminal_stop_is_kept(self):
        pieces = ["Hello", " ", "<eos>"]
        eos = pieces.index("<eos>")
        runtime.MANAGER = loaded_manager([0, eos], pieces, eos)
        final = self.respond()[-1]

        last = self.branch_text(final, " <eos>", strip_index=0)[-1]

        self.assertEqual(last[TURNS][-1]["content"], " ")
        self.assertEqual(last[TURNS][-1]["reasoning"], "")

    def test_empty_text_asks_for_some(self):
        final = self.respond()[-1]
        frames = self.branch_text(final, "")
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_EMPTY)
        self.assertEqual(frames[0][TURNS], final[TURNS])

    def test_text_the_tokenizer_cannot_encode_is_refused(self):
        final = self.respond()[-1]
        frames = self.branch_text(final, "xyz")
        self.assertEqual(len(frames), 1)
        self.assertIn("xyz", frames[0][STATUS])
        self.assertEqual(frames[0][TURNS], final[TURNS])

    def test_typed_text_without_a_clicked_token_explains_the_steps(self):
        final = self.respond()[-1]
        frames = list(
            app.branch_with_text(
                None,
                "Hello", "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_HINT)

    def test_typed_text_against_a_moved_token_is_refused(self):
        # As for a picked alternative: the click names a turn and a token, and
        # is checked against the conversation it is used with. Rewriting the
        # reply by hand takes its measurements away.
        final = self.respond()[-1]
        selected = click_token(final, 1)
        edited = forget_measurements(final[TURNS], 1)
        frames = list(
            app.branch_with_text(selected, "Hello", "", edited, *SETTINGS)
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_UNAVAILABLE)

    def test_typed_text_from_an_earlier_model_load_is_refused(self):
        final = self.respond()[-1]
        # Loading leaves the conversation on screen, but a reply's token IDs
        # belong to the tokenizer that produced it, even for a same-ID reload.
        selected = click_token(final, 1)
        runtime.MANAGER.load_count += 1
        frames = list(
            app.branch_with_text(selected, "Hello", "", final[TURNS], *SETTINGS)
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_MODEL_CHANGED)
        self.assertEqual(frames[0][TURNS], final[TURNS])

    def reload_before(self, method_name):
        """Land a model load inside the branch handler, after its stamp check.

        The handler compares ``branch_source`` against the live load first,
        then calls the runtime; a load finishing in between passes that check
        and must be caught by the runtime's own comparison under the model
        lock. The wrapped method bumps the load count at the moment of the
        call, which is exactly that window.
        """

        manager = runtime.MANAGER
        real = getattr(manager, method_name)

        def reloaded_first(*args, **kwargs):
            manager.load_count += 1
            return real(*args, **kwargs)

        setattr(manager, method_name, reloaded_first)

    def assertBranchRefusedByReload(self, frames, final):
        last = frames[-1]
        self.assertEqual(last[STATUS], app.BRANCH_MODEL_CHANGED)
        self.assertEqual(last[TURNS], final[TURNS])
        self.assertEqual(last[SEND], gr.update(visible=True))
        self.assertEqual(last[STOP], gr.update(visible=False))
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_load_landing_before_the_encoding_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        self.reload_before("encode_replacement")
        frames = self.branch_text(final, "Hello")
        self.assertEqual(len(frames), 1)
        self.assertBranchRefusedByReload(frames, final)

    def test_a_load_landing_before_the_length_check_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        self.reload_before("validate_generation_prefix")
        frames = self.branch_text(final, "Hello")
        self.assertEqual(len(frames), 1)
        self.assertBranchRefusedByReload(frames, final)

    def test_a_load_landing_before_typed_replay_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        self.reload_before("generate")
        frames = self.branch_text(final, "Hello")
        # The opening "Generating…" frame is already out when the runtime
        # refuses the replay, so a second frame takes the conversation back.
        self.assertEqual(len(frames), 2)
        self.assertIn("Generating", frames[0][STATUS])
        self.assertBranchRefusedByReload(frames, final)

    def record_encodings(self, seen):
        """Wrap encode_replacement() to note whether the slot is held at the call."""

        real = runtime.MANAGER.encode_replacement

        def observe(*args, **kwargs):
            seen.append(runtime.MANAGER.busy)
            return real(*args, **kwargs)

        runtime.MANAGER.encode_replacement = observe

    def test_typed_text_is_refused_while_a_response_is_generating(self):
        final = self.respond()[-1]
        encodings = []
        self.record_encodings(encodings)
        self.assertTrue(runtime.MANAGER.reserve_generation())
        try:
            frames = self.branch_text(final, "Hello")
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BUSY_STATUS)
        self.assertEqual(frames[0][TURNS], gr.skip())
        # The encoding waits on the model lock, which the running generation
        # is holding. A refused branch must not queue behind it: it would
        # resume once that generation finished and replay its stale snapshot.
        self.assertEqual(encodings, [])

    def test_typed_text_takes_the_slot_before_it_encodes(self):
        """The reservation comes first, not after the encoding.

        encode_replacement() blocks on the model lock. With only a busy check
        ahead of it, a Send starting in between would hold that lock for its
        whole generation, and the branch would then replay the conversation it
        was handed at click time over the newer one.
        """

        final = self.respond()[-1]
        encodings = []
        self.record_encodings(encodings)
        frames = self.branch_text(final, "Hello")
        self.assertEqual(encodings, [True])
        self.assertNotEqual(frames[-1][STATUS], app.BUSY_STATUS)
        self.assertTrue(frames[-1][TURNS][-1]["content"].startswith("Hello"))
        self.assertFalse(runtime.MANAGER.busy, "the slot must not leak")

    def test_a_refused_typed_branch_encodes_nothing(self):
        """The selection is checked before the model lock is touched.

        encode_replacement() waits on that lock, so a branch that is going to
        be refused must be refused first: queueing behind a running
        generation and then refusing costs the reader the wait and the
        runtime the work.
        """

        final = self.respond()[-1]
        selected = click_token(final, 1)
        edited = forget_measurements(final[TURNS], 1)
        encodings = []
        self.record_encodings(encodings)
        frames = list(
            app.branch_with_text(selected, "Hello", "", edited, *SETTINGS)
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_UNAVAILABLE)
        self.assertEqual(encodings, [])
        self.assertFalse(
            runtime.MANAGER.busy, "a refusal must give the slot back"
        )

    def test_a_cancelled_typed_branch_releases_the_slot(self):
        final = self.respond()[-1]
        selected = click_token(final, 1)
        stream = app.branch_with_text(
                selected,
                "Hello",
            "",
            final[TURNS],
            *SETTINGS,
        )
        first = next(stream)
        self.assertIn("Generating", first[STATUS])
        self.assertTrue(runtime.MANAGER.busy)
        stream.close()
        self.assertFalse(runtime.MANAGER.busy, "GeneratorExit must release the slot")

    def test_a_load_landing_before_a_picked_replay_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        self.reload_before("generate")
        frames = list(
            app.branch_from(
                pick,
                "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(len(frames), 2)
        self.assertBranchRefusedByReload(frames, final)

    def test_typed_text_keeps_literal_prefill_tags_before_it(self):
        runtime.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        selected = click_token(original, 3)
        branched = list(
            app.branch_with_text(
                selected,
                "Hello",
                "",
                original[TURNS],
                *settings.values(),
            )
        )[-1]
        metrics = metrics_of(branched[METRICS])
        self.assertTrue(all(m.get("literal_prefill") for m in metrics[:3]))
        self.assertNotIn("literal_prefill", metrics[3])
        self.assertEqual(metrics[3]["token_id"], 2)
        self.assertTrue(
            branched[TURNS][-1]["content"].startswith("<think>Hello</think>Hello")
        )

    def reasoning_prefill_response(self):
        pieces = [
            "</",
            "think",
            ">\n\n",
            "Prefill",
            " continued",
            "Replacement",
            " after",
            "<eos>",
        ]
        eos = pieces.index("<eos>")
        runtime.MANAGER = loaded_manager([0, 0, 0, 0, 4, eos], pieces, eos)
        runtime.MANAGER.tokenizer = ChatTemplateTokenizer(
            "\nassistant: <think>", pieces=pieces, eos_id=eos
        )
        settings = dict(FIXED, assistant_prefill="Prefill")
        final = list(app.chat("hi", [], *settings.values()))[-1]
        return final, settings

    def test_typed_branch_refuses_the_automatic_reasoning_close(self):
        original, settings = self.reasoning_prefill_response()
        selected = click_token(original, 0)

        frames = list(
            app.branch_with_text(
                selected,
                "Replacement",
                "",
                original[TURNS],
                *settings.values(),
            )
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_REASONING_CLOSE)
        self.assertEqual(frames[0][TURNS], original[TURNS])
        self.assertFalse(runtime.MANAGER.busy)

    def test_typed_branch_after_the_reasoning_close_stays_in_default_context(self):
        original, settings = self.reasoning_prefill_response()
        metrics = metrics_of(original[METRICS])
        self.assertTrue(all(m.get("automatic_reasoning_close") for m in metrics[:3]))
        self.assertNotIn("automatic_reasoning_close", metrics[3])
        selected = click_token(original, 3)
        runtime.MANAGER.model.script = [0, 0, 0, 0, 6, 7]
        runtime.MANAGER.model.step = 0

        branched = list(
            app.branch_with_text(
                selected,
                "Replacement",
                "",
                original[TURNS],
                *settings.values(),
            )
        )[-1]

        reply = branched[TURNS][-1]
        self.assertEqual(reply["reasoning"], "")
        self.assertEqual(reply["content"], "Replacement after")
        request = model_messages(branched[TURNS], include_reasoning=False)
        self.assertEqual(request[-1]["content"], "Replacement after")
        replayed = metrics_of(branched[METRICS])
        self.assertTrue(
            all(m.get("automatic_reasoning_close") for m in replayed[:3])
        )

    def marker_response(self):
        pieces = ["Hello", " world", "<think>", "</think>", "<thi", "<eos>"]
        eos = pieces.index("<eos>")
        runtime.MANAGER = loaded_manager([0, 1, eos], pieces, eos)
        return self.respond()[-1]

    def test_typed_reasoning_markers_are_literal_replacement_text(self):
        for replacement in ("<think>", "</think>", "<thi"):
            with self.subTest(replacement=replacement):
                final = self.marker_response()
                last = self.branch_text(final, replacement)[-1]
                reply = last[TURNS][-1]

                self.assertEqual(reply["reasoning"], "")
                self.assertTrue(reply["content"].startswith("Hello" + replacement))

    def test_a_typed_reasoning_block_survives_default_context_and_save(self):
        final = self.marker_response()
        replacement = "<think>Hello</think>"
        branched = self.branch_text(final, replacement)[-1]
        reply = branched[TURNS][-1]

        self.assertEqual(reply["reasoning"], "")
        self.assertTrue(reply["content"].startswith("Hello" + replacement))
        request = model_messages(branched[TURNS], include_reasoning=False)
        self.assertIn(replacement, request[-1]["content"])

        saved, _status = app.save_conversation(branched[TURNS], "")
        loaded = app.load_conversation(saved["value"], [make_turn("user", "stale")])
        self.assertIn(replacement, loaded[1][-1]["content"])
        self.assertEqual(loaded[1][-1]["reasoning"], "")

    def test_model_reasoning_after_a_literal_replacement_stays_semantic(self):
        final = self.marker_response()
        # The branch prefill has three input positions (prompt, kept token,
        # replacement). Its next distribution emits a real reasoning block.
        runtime.MANAGER.model.script = [0, 0, 2, 0, 3, 1, 5]
        runtime.MANAGER.model.step = 0
        last = self.branch_text(final, "</think>")[-1]
        reply = last[TURNS][-1]

        self.assertEqual(reply["reasoning"], "Hello")
        self.assertEqual(reply["content"], "Hello</think> world")
        default_context = model_messages(last[TURNS], include_reasoning=False)
        self.assertEqual(default_context[-1]["content"], "Hello</think> world")

    def test_literal_replacement_protection_survives_another_branch(self):
        final = self.marker_response()
        first = self.branch_text(final, "<think>Hello</think>")[-1]
        # The first sampled token after the replacement is the fifth token.
        second = self.branch_text(first, " world", strip_index=4)[-1]

        self.assertTrue(
            second[TURNS][-1]["content"].startswith(
                "Hello<think>Hello</think> world"
            )
        )
        self.assertEqual(second[TURNS][-1]["reasoning"], "")
        metrics = metrics_of(second[METRICS])
        self.assertTrue(all(metric.get("literal_text") for metric in metrics[1:5]))

    def test_a_terminal_typed_stop_keeps_prior_reasoning_markers_literal(self):
        final = self.marker_response()
        replacement = "<think>Hello</think><eos>"
        last = self.branch_text(final, replacement)[-1]

        self.assertEqual(last[TURNS][-1]["content"], "Hello<think>Hello</think>")
        self.assertEqual(last[TURNS][-1]["reasoning"], "")
        metrics = metrics_of(last[METRICS])
        self.assertEqual(metrics[-1]["token_id"], runtime.MANAGER.tokenizer.eos_token_id)
        self.assertTrue(all(metric.get("literal_text") for metric in metrics[1:]))

    def sentencepiece_response(self):
        """A response from a tokenizer that drops the first decoded space."""

        pieces = ["\u2581Hello", "\u2581world", "world", "\u2581", "!", "<eos>"]
        eos = pieces.index("<eos>")
        runtime.MANAGER = loaded_manager([0, 4, eos], pieces, eos)
        runtime.MANAGER.tokenizer = SentencePieceTokenizer(pieces, eos)
        final = self.respond()[-1]
        self.assertEqual(final[TURNS][-1]["content"], "Hello!")
        return final

    def test_typed_text_without_a_space_stays_joined_under_sentencepiece(self):
        # "world" round-trips on its own, but its piece would read " world"
        # after "Hello"; the branch must use the piece that joins instead.
        final = self.sentencepiece_response()
        last = self.branch_text(final, "world")[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual([m["token_id"] for m in metrics[:2]], [0, 2])
        self.assertTrue(last[TURNS][-1]["content"].startswith("Helloworld"))

    def test_a_typed_leading_space_is_kept_once_under_sentencepiece(self):
        final = self.sentencepiece_response()
        last = self.branch_text(final, " world")[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual([m["token_id"] for m in metrics[:2]], [0, 1])
        self.assertTrue(last[TURNS][-1]["content"].startswith("Hello world"))

    def test_a_hidden_kept_special_does_not_erase_a_typed_sentencepiece_space(self):
        pieces = ["\u2581<pad>", "\u2581world", "world", "\u2581", "!", "<eos>"]
        pad, space_world, _world, space, bang, eos = range(len(pieces))
        runtime.MANAGER = loaded_manager([pad, bang, eos], pieces, eos)
        runtime.MANAGER.tokenizer = SentencePieceTokenizer(pieces, eos)
        runtime.MANAGER.tokenizer.all_special_ids = [pad, eos]
        final = self.respond()[-1]
        self.assertEqual(final[TURNS][-1]["content"], "!")

        last = self.branch_text(final, " world", strip_index=1)[-1]

        metrics = metrics_of(last[METRICS])
        self.assertEqual(
            [m["token_id"] for m in metrics[:3]], [pad, space, space_world]
        )
        self.assertTrue(last[TURNS][-1]["content"].startswith(" world"))

    def assert_oversized_typed_branch_is_refused(
        self, final, expected_status, repeats=15
    ):
        calls = []
        real_generate = runtime.MANAGER.generate

        def observe(*args, **kwargs):
            calls.append(True)
            return real_generate(*args, **kwargs)

        runtime.MANAGER.generate = observe
        frames = self.branch_text(final, "Hello" * repeats)

        self.assertEqual(
            len(frames), 1, "no destructive opening frame was published"
        )
        self.assertIn(expected_status, frames[0][STATUS])
        self.assertEqual(frames[0][TURNS], final[TURNS])
        self.assertEqual(calls, [])
        self.assertFalse(runtime.MANAGER.busy, "a refusal must give the slot back")

    def test_a_replacement_past_the_model_window_preserves_the_old_response(self):
        final = self.respond()[-1]
        runtime.MANAGER.model.config = type(
            "Config", (), {"max_position_embeddings": 16}
        )()

        self.assert_oversized_typed_branch_is_refused(final, "16 positions")

    def test_a_replacement_that_leaves_too_little_room_preserves_the_old_response(self):
        final = self.respond()[-1]
        runtime.MANAGER.model.config = type(
            "Config", (), {"max_position_embeddings": 16}
        )()
        calls = []
        real_generate = runtime.MANAGER.generate

        def observe(*args, **kwargs):
            calls.append(True)
            return real_generate(*args, **kwargs)

        runtime.MANAGER.generate = observe
        frames = self.branch_text(final, "Hello" * 8)

        self.assertEqual(len(frames), 1)
        self.assertIn("need 17 positions", frames[0][STATUS])
        self.assertEqual(frames[0][TURNS], final[TURNS])
        self.assertEqual(calls, [])
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_replacement_past_the_application_cap_preserves_the_old_response(self):
        final = self.respond()[-1]
        # The cap is the reader's setting, and the floor of its range is the
        # smallest one a test can ask for.
        self.enterContext(settings.override(prefill_token_limit=256))

        self.assert_oversized_typed_branch_is_refused(
            final, "256 token limit for a generation prefix", repeats=300
        )

    def test_typed_text_follows_noncanonical_sentencepiece_tokens(self):
        pieces = [
            "\u2581Hello",
            "\u2581Hel",
            "lo",
            "\u2581world",
            "world",
            "!",
            "<eos>",
        ]
        eos = pieces.index("<eos>")
        runtime.MANAGER = loaded_manager([1, 2, 5, eos], pieces, eos)
        runtime.MANAGER.tokenizer = SentencePieceTokenizer(pieces, eos)
        final = self.respond()[-1]
        self.assertEqual(final[TURNS][-1]["content"], "Hello!")

        last = self.branch_text(final, "world", strip_index=2)[-1]
        metrics = metrics_of(last[METRICS])
        self.assertEqual([m["token_id"] for m in metrics[:3]], [1, 2, 4])
        self.assertTrue(last[TURNS][-1]["content"].startswith("Helloworld"))

    def test_the_branch_text_button_is_wired_as_a_generation(self):
        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "branch_with_text"
        )
        self.assertEqual(len(listener.inputs), 2 + 2 + len(SETTINGS) + 4)
        self.assertEqual(len(listener.outputs), CHAT_OUTPUTS)

    def test_the_branch_button_is_wired_as_a_generation(self):
        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "branch_from"
        )
        self.assertEqual(len(listener.inputs), 1 + 2 + len(SETTINGS) + 4)
        self.assertEqual(len(listener.outputs), CHAT_OUTPUTS)


class ForkTests(unittest.TestCase):
    """Copy the transcript into a second fork and move between them."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)
        # New branches are named around what the saved file holds, so start
        # each test with nothing saved.
        self.path = library.library_path()
        if self.path.exists():
            self.path.unlink()

    def turns(self):
        return [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]

    def test_a_new_branch_is_not_named_after_one_another_page_saved(self):
        # The other page started Chat 1 and Fork 1 after this one loaded, and
        # this page's forks know nothing of them.
        other = new_forks()
        other["branches"]["Chat 1"] = [make_turn("user", "theirs")]
        other["branches"]["Fork 1"] = []
        library.write(other, self.path)

        fresh = app.new_conversation(self.turns(), new_forks())
        self.assertEqual(fresh[FORK_STATE]["active"], "Chat 2")
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        self.assertEqual(forked[FORK_STATE]["active"], "Fork 2")

    def test_forking_copies_the_conversation_into_a_new_fork(self):
        result = app.fork_conversation(self.turns(), new_forks(), None)
        self.assertEqual(len(result), 20)
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(result[FORK_STATE]["active"], "Fork 1")
        self.assertEqual(
            contents(result[FORK_STATE]["branches"][MAIN_BRANCH]),
            contents(self.turns()),
        )
        self.assertEqual(names_of(result[FORK_PICKER]), [MAIN_BRANCH, "Fork 1"])
        self.assertEqual(result[FORK_PICKER]["value"], "Fork 1")
        self.assertIn("Copied", result[FORK_STATUS])
        self.assertEqual(result[FORK_PROMPT], gr.skip())

    def test_a_whole_copy_keeps_the_token_panel(self):
        # The last reply is unchanged, so the strip still describes it.
        result = app.fork_conversation(self.turns(), new_forks(), None)
        for index in range(FORK_STRIP, FORK_TRACE + 1):
            self.assertEqual(result[index], gr.skip())

    def test_forking_at_a_user_message_hands_it_back(self):
        selected = {"index": 2, "content": "two"}
        result = app.fork_conversation(self.turns(), new_forks(), selected)
        self.assertEqual([t["content"] for t in result[FORK_TURNS]], ["one", "first"])
        self.assertEqual(result[FORK_PROMPT], "two")
        self.assertIn("Forked at message 3", result[FORK_STATUS])

    def test_forking_at_an_assistant_message_keeps_it(self):
        selected = {"index": 1, "content": "first"}
        result = app.fork_conversation(self.turns(), new_forks(), selected)
        self.assertEqual([t["content"] for t in result[FORK_TURNS]], ["one", "first"])
        self.assertEqual(result[FORK_PROMPT], gr.skip())

    def test_a_truncated_fork_empties_the_token_panel(self):
        selected = {"index": 1, "content": "first"}
        result = app.fork_conversation(self.turns(), new_forks(), selected)
        # The fork's own turns are drawn in the token view; the measurements
        # of the reply that was cut away are what goes.
        self.assertEqual(
            strip_of(result[FORK_STRIP]),
            app.transcript_value(result[FORK_TURNS], DEFAULT_COLOR_SCALE),
        )
        self.assertEqual(metrics_of(result[FORK_METRICS]), [])
        self.assertEqual(result[FORK_DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(result[FORK_TRACE], {})

    def test_a_stale_selection_copies_the_whole_conversation(self):
        # The message at that index is no longer the one that was clicked.
        selected = {"index": 2, "content": "something else"}
        result = app.fork_conversation(self.turns(), new_forks(), selected)
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertIn("Copied", result[FORK_STATUS])

    def test_a_selection_past_the_end_copies_the_whole_conversation(self):
        result = app.fork_conversation(
            self.turns(), new_forks(), {"index": 40, "content": "x"}
        )
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))

    def test_remembering_a_message_keeps_its_index_and_text(self):
        event = gr.SelectData(None, {"index": 2, "value": "two"})
        self.assertEqual(
            app.remember_message(self.turns(), event), {"index": 2, "content": "two"}
        )
        gone = gr.SelectData(None, {"index": 9, "value": "x"})
        self.assertIsNone(app.remember_message(self.turns(), gone))

    def test_forking_closes_out_a_cancelled_reply(self):
        turns = self.turns()
        turns[-1]["reasoning"] = "half a thought"
        turns[-1]["reasoning_closed"] = False
        result = app.fork_conversation(turns, new_forks(), None)
        self.assertTrue(result[FORK_TURNS][-1]["reasoning_closed"])
        self.assertTrue(result[FORK_STATE]["branches"][MAIN_BRANCH][-1]["reasoning_closed"])
        self.assertEqual(result[FORK_SEND], gr.update(visible=True))

    def test_switching_puts_the_current_fork_away_and_brings_the_other_out(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        edited = forked[FORK_TURNS] + [make_turn("user", "three")]
        result = app.switch_fork(MAIN_BRANCH, edited, forked[FORK_STATE])
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(result[FORK_STATE]["active"], MAIN_BRANCH)
        self.assertEqual(
            contents(result[FORK_STATE]["branches"]["Fork 1"]), contents(edited)
        )
        self.assertEqual(result[FORK_PICKER]["value"], MAIN_BRANCH)
        self.assertIn("Switched to Main", result[FORK_STATUS])
        # The token view follows the conversation switched to.
        self.assertEqual(
            strip_of(result[FORK_STRIP]),
            app.transcript_value(result[FORK_TURNS], DEFAULT_COLOR_SCALE),
        )

    def test_switching_to_the_fork_already_on_screen_changes_nothing(self):
        result = app.switch_fork(MAIN_BRANCH, self.turns(), new_forks())
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(result[FORK_STATE], gr.skip())
        self.assertIn("Already on", result[FORK_STATUS])

    def test_switching_to_a_missing_fork_puts_the_picker_back(self):
        result = app.switch_fork("Fork 7", self.turns(), new_forks())
        self.assertEqual(result[FORK_PICKER]["value"], MAIN_BRANCH)
        self.assertIn("no longer exists", result[FORK_STATUS])

    def test_deleting_a_fork_returns_to_main(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        result = app.delete_fork(forked[FORK_TURNS], forked[FORK_STATE])
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(list(result[FORK_STATE]["branches"]), [MAIN_BRANCH])
        self.assertEqual(names_of(result[FORK_PICKER]), [MAIN_BRANCH])
        self.assertIn("Deleted Fork 1", result[FORK_STATUS])

    def test_the_main_conversation_cannot_be_deleted(self):
        result = app.delete_fork(self.turns(), new_forks())
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertIn("cannot be deleted", result[FORK_STATUS])

    def test_fork_names_are_not_reused_while_taken(self):
        first = app.fork_conversation(self.turns(), new_forks(), None)
        second = app.fork_conversation(first[FORK_TURNS], first[FORK_STATE], None)
        self.assertEqual(second[FORK_STATE]["active"], "Fork 2")
        self.assertEqual(
            list(second[FORK_STATE]["branches"]), [MAIN_BRANCH, "Fork 1", "Fork 2"]
        )

    def test_clear_resets_the_forks(self):
        result = app.clear_chat()
        self.assertEqual(len(result), CLEAR_OUTPUTS)
        self.assertEqual(result[-3]["active"], MAIN_BRANCH)
        self.assertEqual(result[-3]["branches"], {MAIN_BRANCH: []})
        self.assertEqual(names_of(result[-2]), [MAIN_BRANCH])
        # And closes the confirmation that asked for it.
        self.assertEqual(result[-1], gr.update(visible=False))

    def test_clear_marks_every_branch_it_knew_as_gone(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        result = app.clear_chat(app.DEFAULT_COLOR_SCALE, forked[FORK_STATE])
        forks = result[-3]
        # The main conversation is emptied now and Fork 1 deleted now, so a
        # save merges as a change to each rather than as a stale copy.
        self.assertEqual(forks["branches"], {MAIN_BRANCH: []})
        self.assertEqual(set(forks["updated"]), {MAIN_BRANCH, "Fork 1"})
        self.assertGreater(forks["updated"]["Fork 1"], forked[FORK_STATE]["updated"]["Fork 1"])

    def test_a_forked_conversation_can_be_continued(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        last = list(app.chat("three", forked[FORK_TURNS], *SETTINGS))[-1]
        self.assertEqual(len(last[TURNS]), 6)
        self.assertEqual(last[TURNS][4]["content"], "three")

    def test_starting_a_new_chat_puts_the_current_one_away(self):
        result = app.new_conversation(self.turns(), new_forks())
        self.assertEqual(len(result), 20)
        self.assertEqual(result[FORK_TURNS], [])
        self.assertEqual(result[FORK_CHATBOT], [])
        self.assertEqual(result[FORK_STATE]["active"], "Chat 1")
        self.assertEqual(
            contents(result[FORK_STATE]["branches"][MAIN_BRANCH]), contents(self.turns())
        )
        self.assertEqual(result[FORK_STATE]["branches"]["Chat 1"], [])
        self.assertEqual(names_of(result[FORK_PICKER]), [MAIN_BRANCH, "Chat 1"])
        self.assertEqual(result[FORK_PICKER]["value"], "Chat 1")
        self.assertIn("Started Chat 1", result[FORK_STATUS])
        # Whatever is typed in the box is likely meant for the new chat.
        self.assertEqual(result[FORK_PROMPT], gr.skip())
        self.assertEqual((result[FORK_SEND], result[FORK_STOP]), app.send_stop_buttons(False))
        # The token panel described a reply that is no longer on screen.
        self.assertEqual(result[FORK_DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(strip_of(result[FORK_STRIP]), app.EMPTY_TRANSCRIPT)

    def test_new_chats_and_forks_are_numbered_separately(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        fresh = app.new_conversation(forked[FORK_TURNS], forked[FORK_STATE])
        again = app.new_conversation(fresh[FORK_TURNS], fresh[FORK_STATE])
        self.assertEqual(
            list(again[FORK_STATE]["branches"]), [MAIN_BRANCH, "Fork 1", "Chat 1", "Chat 2"]
        )

    def test_a_new_chat_closes_out_a_cancelled_reply(self):
        turns = self.turns()
        turns[-1]["reasoning"] = "thinking"
        turns[-1]["reasoning_closed"] = False
        result = app.new_conversation(turns, new_forks())
        stored = result[FORK_STATE]["branches"][MAIN_BRANCH]
        self.assertTrue(stored[-1]["reasoning_closed"])

    def test_a_new_chat_can_be_deleted_back_to_main(self):
        fresh = app.new_conversation(self.turns(), new_forks())
        result = app.delete_fork(fresh[FORK_TURNS], fresh[FORK_STATE])
        self.assertEqual(result[FORK_STATE]["active"], MAIN_BRANCH)
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(names_of(result[FORK_PICKER]), [MAIN_BRANCH])


class ConversationListTests(unittest.TestCase):
    """The pane lists every conversation with its model and token count."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def test_a_reply_records_the_model_and_its_token_counts(self):
        frames = list(app.chat("hi", [], *SETTINGS))
        reply = frames[-1][TURNS][-1]
        self.assertEqual(reply["role"], "assistant")
        self.assertEqual(reply["model"], "fake/model")
        self.assertEqual(reply["generated_tokens"], len(metrics_of(frames[-1][METRICS])))
        # The opening frame empties the ids; the first streamed frame fills them.
        _stamp, prompt_ids, _load = next(
            frame[CONTEXT_IDS]
            for frame in frames
            if isinstance(frame[CONTEXT_IDS], tuple) and frame[CONTEXT_IDS][1]
        )
        self.assertEqual(reply["prompt_tokens"], len(prompt_ids))
        self.assertGreater(reply["prompt_tokens"], 0)

    def test_the_reply_names_the_model_that_held_the_lock(self):
        # Loading is not refused while a reply is pending, and the generator
        # takes the model lock only when it is first resumed, so a load that
        # lands in the round trip the opening frame costs is the model that
        # generates. The reply has to say so: a stamp taken on the way in
        # would name the model that was swapped out.
        frames = app.chat("hi", [], *SETTINGS)
        opening = next(frames)
        self.assertEqual(opening[TURNS][-1]["role"], "assistant")
        self.assertNotIn("model", opening[TURNS][-1])
        # What load() leaves behind, minus the weights: the fakes stand in for
        # both models.
        runtime.MANAGER.model_id = "other/model"
        runtime.MANAGER.load_count += 1
        rest = list(frames)
        reply = rest[-1][TURNS][-1]
        self.assertEqual(reply["model"], "other/model")
        self.assertEqual(rest[-1][TRACE]["model_id"], "other/model")
        _stamp, _ids, load = next(
            frame[CONTEXT_IDS]
            for frame in rest
            if isinstance(frame[CONTEXT_IDS], tuple) and frame[CONTEXT_IDS][1]
        )
        self.assertEqual(load, "other/model#1")
        self.assertEqual(load, runtime.MANAGER.load_id)

    def test_the_counts_grow_with_the_stream(self):
        # A reply stopped part way keeps the count it had reached, since every
        # frame publishes the turn with the tokens so far.
        frames = list(app.chat("hi", [], *SETTINGS))
        counts = [
            frame[TURNS][-1].get("generated_tokens")
            for frame in frames[1:]
            if frame[TURNS] and frame[TURNS][-1]["role"] == "assistant"
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], len(metrics_of(frames[-1][METRICS])))

    def test_the_refreshed_list_names_the_model_and_the_size(self):
        last = list(app.chat("hi", [], *SETTINGS))[-1]
        update, _forks = app.refresh_conversation_list(last[TURNS], new_forks())
        reply = last[TURNS][-1]
        total = reply["prompt_tokens"] + reply["generated_tokens"]
        self.assertEqual(names_of(update), [MAIN_BRANCH])
        self.assertEqual(update["value"], MAIN_BRANCH)
        self.assertEqual(labels_of(update), [f"Main · hi\nmodel · {total:,} tokens"])

    def test_the_refresh_reads_the_active_branch_from_the_live_turns(self):
        forks = new_forks()
        forks["branches"][MAIN_BRANCH] = [make_turn("user", "stale")]
        forks["branches"]["Fork 1"] = [make_turn("user", "other")]
        forks["active"] = "Fork 1"
        update, seen = app.refresh_conversation_list([make_turn("user", "live")], forks)
        self.assertEqual(
            labels_of(update),
            ["Main · stale\nNo replies yet", "Fork 1 · live\nNo replies yet"],
        )
        self.assertEqual(update["value"], "Fork 1")
        # The forks come back with the live turns written in and stamped, so
        # the state carries when the branch changed; the input is untouched.
        self.assertEqual(seen["branches"]["Fork 1"][0]["content"], "live")
        self.assertEqual(list(seen["updated"]), ["Fork 1"])
        self.assertEqual(forks["branches"]["Fork 1"][0]["content"], "other")

    def test_a_loaded_conversation_keeps_its_tags(self):
        last = list(app.chat("hi", [], *SETTINGS))[-1]
        saved, _ = app.save_conversation(last[TURNS], "")
        loaded = app.load_conversation(saved["value"], [])
        self.assertEqual(loaded[1][-1]["model"], "fake/model")
        self.assertEqual(
            loaded[1][-1]["generated_tokens"], last[TURNS][-1]["generated_tokens"]
        )


class ConversationListWiringTests(unittest.TestCase):
    """The list is redrawn from state, and picking an entry switches to it."""

    def setUp(self):
        self.demo = app.build_app()

    def named(self, name):
        return next(
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        )

    def conversation_list(self):
        return next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Radio) and block.elem_id == "conversation-list"
        )

    def test_the_list_lives_in_the_conversations_pane(self):
        pane = next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Column) and block.elem_id == "conversation-pane"
        )

        def descendants(block):
            for child in getattr(block, "children", []):
                yield child
                yield from descendants(child)

        self.assertIn(self.conversation_list(), list(descendants(pane)))

    def test_the_list_starts_with_the_main_conversation(self):
        radio = self.conversation_list()
        self.assertEqual(radio.choices, [("Main\nNo messages yet", MAIN_BRANCH)])
        self.assertEqual(radio.value, MAIN_BRANCH)

    def test_a_change_to_the_conversation_state_redraws_the_list(self):
        refresh = self.named("refresh_conversation_list")
        state, _scale = self.named("stop_generation").inputs
        forks = self.named("remember_forks").inputs[1]
        self.assertEqual(refresh.targets, [(state._id, "change")])
        self.assertEqual(refresh.outputs, [self.conversation_list(), forks])

    def test_picking_an_entry_switches_to_it(self):
        switch = self.named("switch_fork")
        radio = self.conversation_list()
        self.assertEqual(switch.targets, [(radio._id, "input")])
        self.assertIs(switch.inputs[0], radio)
        self.assertIn(radio, switch.outputs)

    def test_every_branch_handler_redraws_the_list(self):
        radio = self.conversation_list()
        for name in ("fork_conversation", "delete_fork", "new_conversation", "clear_chat"):
            with self.subTest(handler=name):
                self.assertIn(radio, self.named(name).outputs)

    def test_the_saved_conversations_come_back_when_the_page_loads(self):
        restore = self.named("restore_conversations")
        state, _scale = self.named("stop_generation").inputs
        forks = self.named("remember_forks").inputs[1]
        self.assertEqual(restore.targets, [(self.demo._id, "load")])
        self.assertEqual(restore.inputs, [])
        self.assertEqual(restore.outputs[1:], [state, forks, self.conversation_list()])

    def test_everything_that_rewrites_the_conversation_in_one_step_runs_on_one_queue(self):
        # A redraw queued by a streaming frame must not run after a click on
        # New with the pane as it was before the click, and Undo must not
        # publish branch A's shortened transcript into branch B after a
        # switch. Sharing one concurrency id makes Gradio run them in order,
        # reading the states as they are when each runs. The rule is derived
        # rather than listed: every listener that writes the conversation
        # state and is not a streaming handler is on the queue, and every
        # streaming handler is off it, since the redraw has to run between
        # its frames.
        state, _scale = self.named("stop_generation").inputs
        forks = self.named("remember_forks").inputs[1]
        writers = [fn for fn in self.demo.fns.values() if state in fn.outputs or forks in fn.outputs]
        self.assertTrue(writers)
        for fn in writers:
            name = getattr(fn.fn, "__name__", str(fn))
            with self.subTest(handler=name):
                if inspect.isgeneratorfunction(fn.fn):
                    self.assertNotEqual(fn.concurrency_id, app.CONVERSATION_PANE_QUEUE)
                else:
                    self.assertEqual(fn.concurrency_id, app.CONVERSATION_PANE_QUEUE)
        self.assertEqual(self.named("remember_forks").concurrency_id, app.CONVERSATION_PANE_QUEUE)

    def test_a_change_to_the_forks_saves_them(self):
        remember = self.named("remember_forks")
        state, _scale = self.named("stop_generation").inputs
        forks = remember.inputs[1]
        self.assertEqual(remember.targets, [(forks._id, "change")])
        self.assertEqual(remember.inputs, [state, forks])
        self.assertEqual(remember.outputs, [])


class WeightPrecisionWiringTests(unittest.TestCase):
    """The precision radio feeds both load buttons and is saved like a setting."""

    def setUp(self):
        self.demo = app.build_app()

    def named(self, name):
        return next(
            fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", None) == name
        )

    def radio(self):
        return next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Radio) and block.label == "Weight precision"
        )

    def test_both_load_handlers_read_the_radio_last(self):
        radio = self.radio()
        for name in ("download_and_load_model", "load_cached_model"):
            with self.subTest(handler=name):
                self.assertIs(self.named(name).inputs[-1], radio)

    def test_the_radio_offers_the_three_precisions_and_starts_on_the_saved_one(self):
        radio = self.radio()
        self.assertEqual([value for _label, value in radio.choices], list(settings.WEIGHT_PRECISIONS))
        self.assertEqual(radio.value, settings.current().weight_precision)

    def test_the_radio_is_one_of_the_persisted_settings(self):
        self.assertIn(self.radio(), self.named("restore_settings").outputs)
        self.assertEqual(app.PERSISTED_SETTING_NAMES[-1], "weight_precision")


class ConversationSamplingTests(unittest.TestCase):
    """Each conversation answers with its own temperature, top-p, top-k and length."""

    OWN = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 256}

    def setUp(self):
        self.path = library.library_path()
        if self.path.exists():
            self.path.unlink()
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def values(self, updates):
        return dict(
            zip(
                settings.CONVERSATION_SAMPLING,
                [update["value"] for update in updates],
                strict=True,
            )
        )

    def held(self, forks, name=MAIN_BRANCH):
        return branch_sampling(forks, name)

    def test_a_conversation_with_none_of_its_own_shows_the_saved_settings(self):
        shown = self.values(app.sampling_updates(new_forks()))
        self.assertEqual(shown, settings.sampling_defaults())
        # Nothing at all is the same case, since a page starts with nothing.
        self.assertEqual(self.values(app.sampling_updates(None)), shown)

    def test_a_conversation_shows_what_it_was_answered_with(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        self.assertEqual(self.values(app.sampling_updates(forks)), self.OWN)

    def test_only_the_active_conversation_is_shown(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [])
        put_branch_sampling(forks, "Fork 1", self.OWN)

        self.assertEqual(
            self.values(app.sampling_updates(forks)), settings.sampling_defaults()
        )
        forks["active"] = "Fork 1"
        self.assertEqual(self.values(app.sampling_updates(forks)), self.OWN)

    def test_an_unpinned_conversation_answers_with_the_saved_settings(self):
        # Not with the controls: on a switch they hold the conversation being
        # left, so reading them would make an unpinned conversation answer
        # with the sampling of whatever was looked at before it - switching
        # from a temperature-0 fork to an unpinned Main would take Main to 0.
        with settings.override(**self.OWN):
            self.assertEqual(
                self.values(app.sampling_updates(new_forks())), self.OWN
            )
        with settings.override(**settings.sampling_defaults()):
            self.assertEqual(
                self.values(app.sampling_updates(new_forks())),
                settings.sampling_defaults(),
            )

    def test_the_settings_write_is_ordered_before_a_switch_reads_it(self):
        # That file is what an unpinned conversation answers with, so a
        # slider moved and then a switch in quick succession would otherwise
        # read the value moved away from. Sharing the conversation queue is
        # what orders the write ahead of the switch.
        demo = app.build_app()
        saving = [
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "remember_settings"
            and fn.targets
            and fn.targets[0][1] == "input"
        ]

        self.assertEqual(len(saving), len(settings.CONVERSATION_SAMPLING))
        for fn in saving:
            self.assertEqual(fn.concurrency_id, app.CONVERSATION_PANE_QUEUE)
            # A slider still moving while this is pending would otherwise
            # have its newer values dropped, leaving the file holding one
            # from part way through the drag.
            self.assertEqual(fn.trigger_mode, "always_last")

    def test_a_value_the_settings_would_refuse_falls_back_to_the_setting(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN | {"temperature": 99.0})

        shown = self.values(app.sampling_updates(forks))

        self.assertEqual(shown["temperature"], settings.TEMPERATURE_RANGE[1])
        self.assertEqual(shown["top_p"], 1.0)

    def test_moving_a_control_writes_it_into_the_conversation_on_screen(self):
        forks = app.remember_branch_sampling(new_forks(), *self.OWN.values())

        self.assertEqual(self.held(forks), self.OWN)

    def test_a_deliberate_move_is_stored_even_at_the_saved_value(self):
        # The same move also writes the settings file, on its own queue, so
        # comparing against that file here would be a race: were it to land
        # first, the conversation would be left following the file rather
        # than pinned to the value just chosen.
        defaults = settings.sampling_defaults()

        forks = app.remember_branch_sampling(new_forks(), *defaults.values())

        self.assertEqual(self.held(forks), defaults)

    def test_only_the_readers_own_move_writes_the_conversation(self):
        # Every path that changes conversation sets these controls too. A
        # write from that would stamp a conversation nobody had touched, and
        # would claim it from another page that really had changed it, so the
        # write hangs off input rather than change.
        demo = app.build_app()
        listeners = [
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "remember_branch_sampling"
        ]
        sliders = [fn for fn in listeners if fn.targets[0][0] is not None]

        self.assertEqual(len(sliders), len(settings.CONVERSATION_SAMPLING))
        self.assertEqual(
            {event for fn in sliders for _block, event in fn.targets}, {"input"}
        )
        self.assertEqual(len(listeners), len(sliders))

    def test_a_control_reporting_what_the_conversation_already_holds_writes_nothing(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        result = app.remember_branch_sampling(forks, *self.OWN.values())

        self.assertEqual(result, gr.skip())

    def test_a_conversation_set_back_to_the_saved_values_keeps_saying_so(self):
        # It has an entry already, so this is a choice rather than a
        # conversation that never had one.
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        written = app.remember_branch_sampling(
            forks, *settings.sampling_defaults().values()
        )

        self.assertEqual(self.held(written), settings.sampling_defaults())

    def test_the_write_leaves_the_state_it_was_given_alone(self):
        forks = new_forks()
        app.remember_branch_sampling(forks, *self.OWN.values())
        self.assertEqual(forks["sampling"], {})

    def test_a_fork_answers_the_way_the_conversation_it_came_from_does(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        result = app.fork_conversation([make_turn("user", "one")], forks, None)
        forked = result[FORK_STATE]

        self.assertEqual(forked["active"], "Fork 1")
        self.assertEqual(self.held(forked, "Fork 1"), self.OWN)
        self.assertEqual(self.values(app.sampling_updates(forked)), self.OWN)

    def test_forking_pins_both_sides_of_the_comparison(self):
        # Forking is where a comparison is set up. A side carrying no
        # sampling of its own follows the settings file, and the first slider
        # moved on the other side rewrites that file - so both would answer
        # alike, which is the one thing the fork was for.
        result = app.fork_conversation([make_turn("user", "one")], new_forks(), None)
        forked = result[FORK_STATE]

        defaults = settings.sampling_defaults()
        self.assertEqual(self.held(forked, MAIN_BRANCH), defaults)
        self.assertEqual(self.held(forked, "Fork 1"), defaults)

        # Moving a slider on the fork now leaves the conversation it came
        # from where it was, whatever the settings file goes on to say.
        moved = app.remember_branch_sampling(forked, *self.OWN.values())
        with settings.override(**self.OWN):
            self.assertEqual(self.held(moved, "Fork 1"), self.OWN)
            moved["active"] = MAIN_BRANCH
            self.assertEqual(self.values(app.sampling_updates(moved)), defaults)

    def test_a_new_conversation_starts_from_the_saved_settings(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        result = app.new_conversation([make_turn("user", "one")], forks)
        started = result[FORK_STATE]

        # Pinned to what the file said when it was started, rather than left
        # following the file: a conversation that followed it would be moved
        # by a slider touched on any other conversation.
        self.assertEqual(
            self.held(started, started["active"]), settings.sampling_defaults()
        )
        self.assertEqual(
            self.values(app.sampling_updates(started)), settings.sampling_defaults()
        )
        # And the conversation it was started beside keeps its own.
        self.assertEqual(self.held(started), self.OWN)

    def test_a_clamped_response_length_reaches_the_conversation(self):
        # Committing a lower context limit pulls the length down with it, and
        # the conversation has to be told: it would otherwise put the longer
        # length back the next time it was switched to.
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN | {"max_new_tokens": 1024})

        with settings.override(prefill_token_limit=8192, max_new_tokens=1024):
            _limit, length, written = app.remember_prefill_limit(
                256, 1024, forks, 0.0, 1.0, 0, 1024
            )

        # 256 is the floor the limit can be lowered to, and the length
        # follows it down.
        self.assertEqual(length["value"], 256)
        self.assertEqual(self.held(written)["max_new_tokens"], 256)

    def test_a_context_limit_that_clamps_nothing_leaves_the_conversation_alone(self):
        # A limit tabbed through, or raised, changes nothing - and writing on
        # that would pin a conversation that had been following the file.
        with settings.override(prefill_token_limit=1024, max_new_tokens=256):
            _limit, _length, written = app.remember_prefill_limit(
                4096, 256, new_forks(), *self.OWN.values()
            )

        self.assertEqual(written, gr.skip())

    def test_a_new_conversation_is_pinned_from_the_controls(self):
        # The settings file is written by its own listener, a round trip
        # behind: a conversation started right after a slider was moved and
        # pinned from the file would be pinned to the value just moved away
        # from, and would then put it back on the controls.
        with settings.override(**settings.sampling_defaults()):
            result = app.new_conversation(
                [make_turn("user", "one")],
                new_forks(),
                DEFAULT_COLOR_SCALE,
                *self.OWN.values(),
            )

        started = result[FORK_STATE]
        self.assertEqual(self.held(started, started["active"]), self.OWN)
        self.assertEqual(self.values(app.sampling_updates(started)), self.OWN)

    def test_a_fork_of_an_unpinned_conversation_takes_the_controls_too(self):
        with settings.override(**settings.sampling_defaults()):
            result = app.fork_conversation(
                [make_turn("user", "one")],
                new_forks(),
                None,
                DEFAULT_COLOR_SCALE,
                *self.OWN.values(),
            )

        forked = result[FORK_STATE]
        self.assertEqual(self.held(forked, MAIN_BRANCH), self.OWN)
        self.assertEqual(self.held(forked, "Fork 1"), self.OWN)

    def test_a_fork_keeps_a_key_this_version_knows_nothing_about(self):
        # A file written by a newer version can carry an extra sampling
        # field. It goes to both sides, or that version would find the fork
        # answering differently from the conversation it came from.
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)
        forks["sampling"][MAIN_BRANCH]["repetition_penalty"] = 1.15

        result = app.fork_conversation([make_turn("user", "one")], forks, None)
        forked = result[FORK_STATE]

        for name in (MAIN_BRANCH, "Fork 1"):
            with self.subTest(branch=name):
                self.assertEqual(
                    self.held(forked, name).get("repetition_penalty"), 1.15
                )
                self.assertEqual(self.held(forked, name)["temperature"], 0.0)

    def test_a_fork_of_a_pinned_conversation_keeps_its_sampling(self):
        # The parent's own entry wins over the controls: the controls are
        # showing it anyway, and the entry is what the parent answers with.
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        result = app.fork_conversation(
            [make_turn("user", "one")],
            forks,
            None,
            DEFAULT_COLOR_SCALE,
            *settings.sampling_defaults().values(),
        )

        self.assertEqual(self.held(result[FORK_STATE], "Fork 1"), self.OWN)

    def test_clearing_everything_lets_go_of_the_sampling_too(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)

        cleared = app.clear_chat(DEFAULT_COLOR_SCALE, forks)[-3]

        self.assertEqual(cleared["sampling"], {})
        # Stamped, so the copy on disk does not look like the newer of the
        # two and pin the emptied conversation again.
        self.assertIn(MAIN_BRANCH, cleared["sampling_updated"])

    def test_switching_back_brings_the_sampling_back(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [])
        put_branch_sampling(forks, MAIN_BRANCH, self.OWN)
        forks["active"] = "Fork 1"

        result = app.switch_fork(MAIN_BRANCH, [], forks)

        self.assertEqual(self.values(app.sampling_updates(result[FORK_STATE])), self.OWN)

    def test_a_deleted_conversation_leaves_its_sampling_behind(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [])
        put_branch_sampling(forks, "Fork 1", self.OWN)
        forks["active"] = "Fork 1"

        result = app.delete_fork([], forks)
        left = result[FORK_STATE]

        self.assertEqual(left["active"], MAIN_BRANCH)
        self.assertEqual(left["sampling"], {})

    def test_the_page_comes_back_on_the_active_conversations_sampling(self):
        # restore_settings reads the file rather than the restored state, so
        # it does not depend on which page-load handler Gradio runs first.
        forks = new_forks()
        put_branch(forks, "Fork 1", [make_turn("user", "one")])
        put_branch_sampling(forks, "Fork 1", self.OWN)
        forks["active"] = "Fork 1"
        library.write(forks, self.path)

        updates = app.restore_settings()
        restored = dict(zip(app.PERSISTED_SETTING_NAMES, updates, strict=False))

        for name, value in self.OWN.items():
            with self.subTest(setting=name):
                self.assertEqual(restored[name]["value"], value)

    def test_a_page_with_nothing_saved_comes_back_on_the_settings_file(self):
        updates = app.restore_settings()
        restored = dict(zip(app.PERSISTED_SETTING_NAMES, updates, strict=False))

        for name, value in settings.sampling_defaults().items():
            with self.subTest(setting=name):
                self.assertEqual(restored[name]["value"], value)


class ConversationLibraryTests(unittest.TestCase):
    """What the pane shows is written as it changes and read back on load."""

    def setUp(self):
        self.path = library.library_path()
        if self.path.exists():
            self.path.unlink()

    def test_a_redraw_hands_the_pane_as_seen_to_the_forks_which_are_then_saved(self):
        forks = new_forks()
        forks["branches"]["Chat 2"] = [make_turn("user", "other")]
        forks["active"] = "Chat 2"
        on_screen = [make_turn("user", "other"), make_turn("assistant", "reply", "")]

        _update, seen = app.refresh_conversation_list(on_screen, forks)
        # The redraw itself writes nothing; the forks' change does.
        self.assertFalse(self.path.exists())
        app.remember_forks(on_screen, seen)

        saved = library.read(self.path)
        self.assertEqual(saved["active"], "Chat 2")
        self.assertEqual(saved["branches"][MAIN_BRANCH], [])
        self.assertEqual([turn["content"] for turn in saved["branches"]["Chat 2"]], ["other", "reply"])
        self.assertEqual(list(saved["updated"]), ["Chat 2"])

    def test_a_branch_put_away_later_does_not_outrank_a_newer_copy_of_it(self):
        # Two tabs on one file. Tab A edits Main and saves; tab B edits Main
        # and saves after it; then A starts a new chat, which puts its copy of
        # Main away, without having touched Main since. B's copy is the newer
        # and must stay, so the stamp A puts Main away with has to be the
        # time A changed it, not the time A put it away.
        a_forks = new_forks()
        a_turns = [make_turn("user", "A's edit")]
        _update, a_forks = app.refresh_conversation_list(a_turns, a_forks)
        app.remember_forks(a_turns, a_forks)

        b_turns = [make_turn("user", "B's later edit")]
        _update, b_forks = app.refresh_conversation_list(b_turns, new_forks())
        app.remember_forks(b_turns, b_forks)

        fresh = app.new_conversation(a_turns, a_forks)
        app.remember_forks(fresh[FORK_TURNS], fresh[FORK_STATE])

        saved = library.read(self.path)
        self.assertEqual([turn["content"] for turn in saved["branches"][MAIN_BRANCH]], ["B's later edit"])
        self.assertEqual(list(saved["branches"]), [MAIN_BRANCH, "Chat 1"])

    def test_a_change_of_forks_writes_them_too(self):
        forks = new_forks()
        forks["branches"]["Fork 1"] = []
        app.remember_forks([], forks)

        self.assertEqual(list(library.read(self.path)["branches"]), [MAIN_BRANCH, "Fork 1"])

    def test_nothing_saved_leaves_the_page_as_built(self):
        self.assertEqual(app.restore_conversations(), (gr.skip(),) * 4)

    def test_the_active_branch_is_put_back_on_screen(self):
        forks = {
            "active": "Fork 1",
            "branches": {
                MAIN_BRANCH: [make_turn("user", "first")],
                "Fork 1": [make_turn("user", "hi"), make_turn("assistant", "there", "")],
            },
        }
        library.write(forks, self.path)

        messages, turns, restored, update = app.restore_conversations()

        self.assertEqual([turn["content"] for turn in turns], ["hi", "there"])
        self.assertTrue(turns[-1]["reasoning_closed"])
        self.assertEqual(len(messages), 2)
        self.assertEqual(restored["active"], "Fork 1")
        self.assertEqual(update["value"], "Fork 1")
        self.assertEqual([name for _label, name in update["choices"]], [MAIN_BRANCH, "Fork 1"])


class CancelWiringTests(unittest.TestCase):
    """Anything that replaces the conversation must cancel a running generation.

    Otherwise the generator's next snapshot writes its private copy of the
    in-progress turns straight back over the new conversation.

    The listeners are derived from the app rather than listed here, so a new
    control that writes the conversation state fails this test until it is
    wired up. The rule: a listener that writes the conversation state either
    *is* a generation (Send, Retry, Edit - they re-enter generate_reply, and
    they are the events everything else cancels) or it must cancel every one
    of those generations.
    """

    def setUp(self):
        self.demo = app.build_app()

    def named(self, name):
        return next(
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        )

    def conversation_state(self):
        """Stop reads the conversation state first, then token provenance."""

        state, _scale = self.named("stop_generation").inputs
        return state

    def writers(self):
        """Every listener that writes the conversation state, by index."""

        state = self.conversation_state()
        return {index: fn for index, fn in self.demo.fns.items() if state in fn.outputs}

    def cancels_of(self, fn):
        """A ``cancels=`` argument becomes a companion event on the same target."""

        return {
            index
            for other in self.demo.fns.values()
            if other.targets == fn.targets
            for index in other.cancels
        }

    def test_every_conversation_replacing_listener_cancels_generation(self):
        writers = self.writers()
        generations = {
            index for index, fn in writers.items() if inspect.isgeneratorfunction(fn.fn)
        }
        self.assertTrue(generations, "no streaming handler writes the conversation")

        replacers = {
            index: fn for index, fn in writers.items() if index not in generations
        }
        self.assertTrue(replacers, "nothing replaces the conversation")

        for index, fn in replacers.items():
            name = getattr(fn.fn, "__name__", str(index))
            with self.subTest(listener=name):
                self.assertEqual(
                    self.cancels_of(fn),
                    generations,
                    f"{name} must cancel every running generation",
                )

    def test_the_known_controls_are_all_covered(self):
        # A sanity check on the derivation above: if one of these stops writing
        # the conversation state, the rule silently stops guarding it.
        names = {getattr(fn.fn, "__name__", None) for fn in self.writers().values()}
        self.assertEqual(
            names,
            {
                "chat",
                "retry_last",
                "retry_message",
                "edit_message",
                "stop_generation",
                "undo_last",
                "undo_message",
                "clear_chat",
                "load_with_steering",
                "branch_from",
                "branch_with_text",
                "fork_conversation",
                "switch_fork",
                "delete_fork",
                "new_conversation",
                "restore_conversations",
            },
        )


class MessageBoxKeysTests(unittest.TestCase):
    """Enter sends by default; the checkbox swaps Enter and Shift+Enter."""

    def test_enter_sends_by_default(self):
        # Gradio's Textbox submits on Enter only when it is a single-line box.
        demo = app.build_app()
        prompt = next(
            c for c in demo.blocks.values()
            if isinstance(c, gr.Textbox) and c.label == "Message"
        )
        self.assertEqual(prompt.lines, 1)
        self.assertEqual(prompt.max_lines, app.MESSAGE_BOX_MAX_LINES)
        self.assertIn("Enter sends", prompt.placeholder)

    def test_turning_the_setting_off_makes_shift_enter_send(self):
        update = app.set_message_box_keys(False)
        self.assertEqual(update["lines"], 3)
        self.assertEqual(update["max_lines"], app.MESSAGE_BOX_MAX_LINES)
        self.assertIn("Shift+Enter sends", update["placeholder"])

    def test_turning_the_setting_back_on_restores_enter(self):
        update = app.set_message_box_keys(True)
        self.assertEqual(update["lines"], 1)
        self.assertIn("Enter sends", update["placeholder"])

    def test_the_assistant_prefill_control_explains_reasoning_models(self):
        demo = app.build_app()
        prefill = next(
            c
            for c in demo.blocks.values()
            if isinstance(c, gr.Textbox) and c.label == "Assistant prefill (optional)"
        )
        # Empty until the reader saves one, whether as "" or as nothing at all.
        self.assertFalse(prefill.value)
        self.assertIn("closes the reasoning block", prefill.info)


if __name__ == "__main__":
    unittest.main()


class LayerInspectionTests(unittest.TestCase):
    """The logit lens and attention panel behind the Inspect layers button."""

    def setUp(self):
        self.original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)
        self.calls = []
        self.load_ids = []

        def fake_inspect(sequence, index, *, context_count=0, load_id=None):
            self.calls.append((list(sequence), index, context_count))
            self.load_ids.append(load_id)
            return TokenInsight(
                index=index,
                token_id=sequence[index],
                token_text="x",
                layers=[
                    {"layer": layer, "probability": 0.5, "rank": 1, "entropy_bits": 1.0,
                     "top_id": 1, "top_text": "x", "top_probability": 0.5}
                    for layer in range(3)
                ],
                tokens=[
                    {"index": i, "token_id": t, "text": "t", "fallback": "", "segment": "prompt"}
                    for i, t in enumerate(sequence[:index])
                ],
                attention=[[1.0 / index] * index for _ in range(2)],
                decided_at=0,
            )

        runtime.MANAGER.inspect = fake_inspect

    def inspect(self, *args):
        """The last frame of the inspection handler, which streams like Send."""

        return list(app.inspect_layers(*args))[-1]

    def finished(self):
        """The final frame, with the context ids the stream published earlier.

        The ids are published once, on the first frame that carries tokens, and
        every later frame skips them; in the browser the state keeps them, so
        the test carries them forward the same way.
        """

        frames = list(app.chat("hi", [], *SETTINGS))
        final = list(frames[-1])
        for slot in (PROMPT_METRICS, CONTEXT_IDS):
            final[slot] = next(
                frame[slot] for frame in reversed(frames) if isinstance(frame[slot], tuple)
            )
        return final

    def test_the_prompt_ids_are_published_with_the_strip(self):
        frames = list(app.chat("hi", [], *SETTINGS))
        self.assertEqual(
            frames[0][CONTEXT_IDS], (frames[0][METRICS][0], [], "fake/model#0")
        )
        stamp, ids, load = frames[1][CONTEXT_IDS]
        self.assertEqual(stamp, frames[1][METRICS][0])
        self.assertEqual(ids, [0])
        self.assertEqual(load, runtime.MANAGER.load_id)
        # Later frames leave the ids alone: the prompt never changes mid-stream.
        self.assertEqual(frames[-1][CONTEXT_IDS], gr.skip())

    def test_scored_text_publishes_its_context_ids(self):
        result = list(app.score_text("", "Hello", False, DEFAULT_COLOR_SCALE))[-1]
        stamp, ids, load = result[13]
        self.assertEqual(stamp, result[1][0])
        self.assertEqual(ids, [])
        self.assertEqual(load, runtime.MANAGER.load_id)

    def test_scored_layers_use_the_scored_sequence_after_chatting(self):
        scored = score_known_passage()
        final = self.finished()
        target = app.remember_inspect_target("score")(scored[2], select(0))
        self.assertIsNotNone(target)
        *_, insight, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS],
            0, scored[2], scored[14],
        )
        expected = scored[14][1] + [m["token_id"] for m in scored[2][1]]
        context_count = len(scored[14][1])
        self.assertEqual(self.calls, [(expected, context_count, context_count)])
        self.assertEqual(insight["token_id"], scored[2][1][0]["token_id"])
        self.assertIn("Token 1", status)

    def test_rescoring_rejects_queued_score_clicks_and_inspection(self):
        scored = score_known_passage()
        target = app.remember_inspect_target("score")(scored[2], select(0))
        score_known_passage(" world", "Hello")
        self.assertIsNone(app.remember_inspect_target("score")(scored[2], select(0)))
        self.assertEqual(app.inspect_token("score")(scored[2], select(0)), (gr.skip(), gr.skip()))
        *_, status = self.inspect(
            target, scored[1], scored[4], scored[13], 0, scored[2], scored[14]
        )
        self.assertEqual(status, app.INSPECT_HINT)
        self.assertEqual(self.calls, [])

    def test_scored_inspection_still_checks_the_model_load(self):
        scored = score_known_passage()
        final = self.finished()
        target = app.remember_inspect_target("score")(scored[2], select(0))
        stale_context = (*scored[14][:2], "previous-load")
        *_, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS],
            0, scored[2], stale_context,
        )
        self.assertEqual(status, app.INSPECT_MODEL_CHANGED)
        self.assertEqual(self.calls, [])

    def test_scored_inspection_checks_its_stamp_before_and_after_delivery(self):
        from ui.panel import new_metrics_generation

        original_inspect = runtime.MANAGER.inspect
        for before_delivery in (True, False):
            with self.subTest(before_delivery=before_delivery):
                scored = score_known_passage()
                target = app.remember_inspect_target("score")(scored[2], select(0))

                def replace_scored_passage(*args, **kwargs):
                    result = original_inspect(*args, **kwargs)
                    new_metrics_generation(scored=True)
                    return result

                runtime.MANAGER.inspect = (
                    replace_scored_passage if before_delivery else original_inspect
                )
                stream = app.inspect_layers(
                    target, scored[1], scored[4], scored[13], 0, scored[2], scored[14]
                )
                try:
                    frame = next(stream)
                    if before_delivery:
                        self.assertEqual(frame[-1], app.INSPECT_GONE)
                        self.assertEqual(frame[0], gr.skip())
                    else:
                        self.assertIsInstance(frame[3], dict)
                        new_metrics_generation(scored=True)
                        frame = next(stream)
                        self.assertEqual(frame[-1], app.INSPECT_GONE)
                        self.assertIsNone(frame[3])
                finally:
                    stream.close()
                    runtime.MANAGER.inspect = original_inspect

    def test_score_context_is_wired_separately_from_chat_context(self):
        demo = app.build_app()
        score = next(fn for fn in demo.fns.values() if fn.fn is app.score_text)
        chat = next(fn for fn in demo.fns.values() if fn.fn is app.chat)
        inspect = next(fn for fn in demo.fns.values() if fn.fn is app.inspect_layers)
        self.assertEqual(inspect.inputs[3], chat.outputs[CONTEXT_IDS])
        self.assertEqual(inspect.inputs[-2:], [score.outputs[2], score.outputs[14]])
        self.assertNotIn(score.outputs[14], chat.outputs)

    def test_a_response_token_is_inspected_in_its_full_sequence(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(1))
        self.assertEqual(target, {"generation": final[METRICS][0], "strip": "response", "index": 1})

        lens, attention, slider, insight, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(self.calls, [([0, 2, 3, THINK_EOS], 2, 1)])
        # The load id goes along so the runtime can check it under its lock.
        self.assertEqual(self.load_ids, [runtime.MANAGER.load_id])
        self.assertIn("logit-lens", lens)
        self.assertIn("attention-view", attention)
        self.assertEqual(slider, gr.update(maximum=2, value=0))
        self.assertEqual(insight["index"], 2)
        self.assertIn("Token 2", status)

    def test_an_output_only_lens_says_why_in_the_status(self):
        final = self.finished()
        real_inspect = runtime.MANAGER.inspect

        def output_only(sequence, index, *, context_count=0, load_id=None):
            insight = real_inspect(sequence, index, context_count=context_count)
            return replace(insight, layers=insight.layers[-1:], decided_at=None)

        runtime.MANAGER.inspect = output_only
        target = app.remember_inspect_target("response")(final[METRICS], select(1))
        lens, *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertIn("read through 0 layers", status)
        self.assertIn(app.INSPECT_OUTPUT_ONLY, status)
        self.assertNotIn("<svg", lens)

    def test_the_first_prompt_token_is_refused_without_a_pass(self):
        final = self.finished()
        target = app.remember_inspect_target("prompt")(final[PROMPT_METRICS], select(0))
        self.assertEqual(target["strip"], "prompt")
        *_, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(status, app.INSPECT_FIRST)
        self.assertEqual(self.calls, [])

    def test_a_prompt_token_is_inspected_at_its_own_position(self):
        final = self.finished()
        # Pretend the prompt had two tokens, so the second one can be inspected.
        stamp, _ids, model = final[CONTEXT_IDS]
        context = (stamp, [0, 1], model)
        prompt_metrics = (stamp, [{"token_id": 0}, {"token_id": 1}])
        target = app.remember_inspect_target("prompt")(prompt_metrics, select(1))
        *_, status = self.inspect(target, final[METRICS], prompt_metrics, context, 0)
        self.assertEqual(self.calls, [([0, 1, 2, 3, THINK_EOS], 1, 2)])
        self.assertIn("Prompt token 2", status)

    def test_a_prompt_strip_that_disagrees_with_the_ids_is_refused(self):
        final = self.finished()
        stamp, _ids, model = final[CONTEXT_IDS]
        prompt_metrics = (stamp, [{"token_id": 5}])
        target = app.remember_inspect_target("prompt")(prompt_metrics, select(0))
        *_, status = self.inspect(
            target, final[METRICS], prompt_metrics, (stamp, [0, 1], model), 0
        )
        self.assertEqual(status, app.INSPECT_GONE)

    def test_a_failed_pass_is_reported(self):
        final = self.finished()

        def refuse(*_args, **_kwargs):
            raise RuntimeError("out of memory")

        runtime.MANAGER.inspect = refuse
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        lens, *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(lens, gr.skip())
        self.assertEqual(
            status,
            '<div class="failure">Could not inspect that token: out of memory</div>',
        )

    def test_the_slider_keeps_its_layer_when_it_still_exists(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        _lens, attention, slider, *_ = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 2
        )
        self.assertEqual(slider, gr.update(maximum=2, value=2))
        self.assertIn("layer 2", attention)
        _lens, _attention, slider, *_ = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 9
        )
        self.assertEqual(slider, gr.update(maximum=2, value=2))

    def test_a_target_from_a_replaced_strip_is_refused(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        later = self.finished()
        *_rest, status = self.inspect(
            target, later[METRICS], later[PROMPT_METRICS], later[CONTEXT_IDS], 0
        )
        self.assertEqual(status, app.INSPECT_HINT)
        self.assertEqual(self.calls, [])
        self.assertIsNone(app.remember_inspect_target("response")(final[METRICS], select(0)))

    def test_a_running_generation_is_not_interrupted(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        self.assertTrue(runtime.MANAGER.reserve_generation())
        try:
            *_rest, status = self.inspect(
                target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
            )
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(status, app.INSPECT_BUSY)
        self.assertEqual(self.calls, [])

    def test_a_load_is_named_rather_than_a_response(self):
        # A load turns the pass away as a reply does, and the strip being
        # inspected belongs to the weights on their way out. Telling the
        # reader to wait for a response points at nothing on the page.
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        _checked_id, claim = runtime.MANAGER.claim_exclusive_load("org/other").claim
        try:
            *_rest, status = self.inspect(
                target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
            )
        finally:
            runtime.MANAGER.release_load(claim)
        self.assertEqual(status, app.INSPECT_LOADING)
        self.assertNotIn("response", app.INSPECT_LOADING)
        self.assertEqual(self.calls, [])

    def test_a_load_that_has_emptied_memory_is_still_named_as_a_load(self):
        # The claim comes before the loaded check now, so the phase of a load
        # in which memory stands empty is still answered as a load rather
        # than with advice to go and load a model.
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        _checked_id, claim = runtime.MANAGER.claim_exclusive_load("org/other").claim
        self.addCleanup(runtime.MANAGER.release_load, claim)
        with mock.patch.object(
            type(runtime.MANAGER), "loaded", property(lambda self: False)
        ):
            *_rest, status = self.inspect(
                target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
            )

        self.assertEqual(status, app.INSPECT_LOADING)
        self.assertEqual(self.calls, [])

    def test_a_pass_refused_by_an_empty_machine_gives_the_slot_back(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        with mock.patch.object(
            type(runtime.MANAGER), "loaded", property(lambda self: False)
        ):
            *_rest, status = self.inspect(
                target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
            )

        self.assertEqual(status, "Download and load a model first.")
        self.assertFalse(runtime.MANAGER.busy, "the refusal kept the slot")

    def test_a_strip_from_another_load_gives_the_slot_back(self):
        # Every early exit between the claim and the pass has to, not only
        # the one about an empty machine.
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        context = (*final[CONTEXT_IDS][:2], "other/model#9")
        *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], context, 0
        )

        self.assertEqual(status, app.INSPECT_MODEL_CHANGED)
        self.assertFalse(runtime.MANAGER.busy, "the refusal kept the slot")

    def test_the_pass_holds_the_generation_slot_and_gives_it_back(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        seen = []

        def observe(sequence, index, *, context_count=0, load_id=None):
            seen.append(runtime.MANAGER.busy)
            return self.fake(sequence, index, context_count=context_count)

        self.fake, runtime.MANAGER.inspect = runtime.MANAGER.inspect, observe
        self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(seen, [True])
        self.assertFalse(runtime.MANAGER.busy)

        def fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        runtime.MANAGER.inspect = fail
        self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertFalse(runtime.MANAGER.busy)

    def test_the_slot_is_held_until_the_readout_has_been_delivered(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        frames = app.inspect_layers(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        first = next(frames)
        # Gradio resumes the generator only once the browser has this frame,
        # so a Send arriving in the meantime still finds the slot taken.
        self.assertIn("logit-lens", first[0])
        self.assertTrue(runtime.MANAGER.busy)
        self.assertEqual(list(frames), [])
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_readout_delivered_after_the_strips_were_replaced_is_taken_down(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        frames = app.inspect_layers(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        next(frames)
        # Clear does not take the slot; it mints a stamp while the frame is
        # in flight, and its reset lands before the readout does.
        app.new_metrics_generation()
        lens, attention, slider, insight, status = next(frames)
        self.assertEqual(lens, charts.EMPTY_LENS)
        self.assertEqual(attention, charts.EMPTY_ATTENTION)
        self.assertEqual(slider, gr.skip())
        self.assertIsNone(insight)
        self.assertEqual(status, app.INSPECT_GONE)
        self.assertEqual(list(frames), [])
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_strip_replaced_during_the_pass_is_not_described(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        original = runtime.MANAGER.inspect

        def replace_strips(sequence, index, *, context_count=0, load_id=None):
            # Clear, Undo and friends do not take the generation slot; they
            # mint a new stamp, which is what the handler has to notice.
            app.new_metrics_generation()
            return original(sequence, index, context_count=context_count)

        runtime.MANAGER.inspect = replace_strips
        lens, *_rest, insight, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(lens, gr.skip())
        self.assertEqual(insight, gr.skip())
        self.assertEqual(status, app.INSPECT_GONE)

    def test_tokens_from_an_earlier_load_are_not_explained_by_this_one(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        # Loading leaves the strips on screen. Re-downloading the same model
        # ID can bring newer weights, so even a same-ID reload is a new load.
        runtime.MANAGER.load_count += 1
        *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(status, app.INSPECT_MODEL_CHANGED)
        self.assertEqual(self.calls, [])

    def test_a_load_that_lands_during_the_pass_is_reported(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))

        def reloaded(*_args, **_kwargs):
            raise ModelChanged("reloaded")

        runtime.MANAGER.inspect = reloaded
        lens, *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(lens, gr.skip())
        self.assertEqual(status, app.INSPECT_MODEL_CHANGED)

    def test_nothing_selected_gives_the_hint(self):
        final = self.finished()
        *_rest, status = self.inspect(
            None, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(status, app.INSPECT_HINT)

    def test_the_panel_resets_only_when_it_shows_something(self):
        self.assertEqual(app.reset_inspection(None), (gr.skip(),) * 4)
        lens, attention, insight, status = app.reset_inspection({"index": 1})
        self.assertEqual(lens, charts.EMPTY_LENS)
        self.assertEqual(attention, charts.EMPTY_ATTENTION)
        self.assertIsNone(insight)
        self.assertEqual(status, app.INSPECT_HINT)

    def test_repainting_another_layer_needs_no_new_pass(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        *_lens, _attention, _slider, insight, _status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertIn("layer 1", app.render_attention(insight, 1))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(app.render_attention(None, 1), gr.skip())
