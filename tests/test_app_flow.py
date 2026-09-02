import inspect
import unittest
from dataclasses import replace

import gradio as gr
import numpy as np

import app
import charts
from conversation import (
    MAIN_BRANCH,
    display_messages,
    make_turn,
    model_messages,
    new_forks,
)
from model_runtime import GenerationUpdate, ModelChanged, TokenInsight
from token_metrics import DEFAULT_COLOR_SCALE

from test_streaming import SentencePieceTokenizer, loaded_manager


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
    BRANCH_SOURCE,
    CONTEXT_IDS,
) = range(len(app.CHAT_OUTPUT_NAMES))
CHAT_OUTPUTS = len(app.CHAT_OUTPUT_NAMES)

# The panels every conversation-replacing handler resets after its own rows:
# the prompt strip and its state and note, the two charts, and the export.
PANEL_OUTPUTS = 6
UNDO_OUTPUTS = 10 + PANEL_OUTPUTS
# Clear also resets the forks and their picker.
CLEAR_OUTPUTS = 9 + PANEL_OUTPUTS + 2
LOAD_OUTPUTS = 10 + PANEL_OUTPUTS


def metrics_of(payload):
    """The metrics half of a metrics_state payload, dropping its stamp."""

    _generation, metrics = payload
    return metrics


def strip_of(value):
    """The tokens in a strip output, whether it is a value or a gr.update."""

    return value["value"] if isinstance(value, dict) else value


def select(index):
    return gr.SelectData(None, {"index": index, "value": "x"})


class ChatFlowTests(unittest.TestCase):
    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

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
        app.MANAGER = loaded_manager(
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
        app.MANAGER = loaded_manager([0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
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
        app.MANAGER = self.original
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
        self.assertEqual(strip_of(final[STRIP]), [])
        self.assertEqual(metrics_of(final[METRICS]), [])
        self.assertEqual(final[DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(final[ALTS], [])

    def test_a_new_response_resets_the_selected_token_details(self):
        # The first frame empties the strip, so the token the user had selected
        # in the previous response no longer exists and its probabilities must
        # not stay on screen beside a strip that no longer contains it.
        frames = self.last(app.chat("hi", [], *SETTINGS))
        self.assertEqual(strip_of(frames[0][STRIP]), [])
        self.assertEqual(frames[0][DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(frames[0][ALTS], [])
        # Later frames only append to the strip, so a token picked mid-stream
        # stays valid and its details are left alone.
        for frame in frames[1:]:
            self.assertEqual(frame[DETAIL], gr.skip())
            self.assertEqual(frame[ALTS], gr.skip())

    def test_a_retry_resets_the_selected_token_details(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "stale")]
        frames = self.last(app.retry_last("", turns, *SETTINGS))
        self.assertEqual(frames[0][DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(frames[0][ALTS], [])

    def test_a_refused_send_keeps_the_token_diagnostics(self):
        # gr.skip() leaves the previous response's panel on screen.
        final = self.last(app.chat("   ", [], *SETTINGS))[-1]
        self.assertEqual(final[STATUS], "Enter a message first.")
        for index in (STRIP, METRICS, DETAIL, ALTS):
            self.assertEqual(final[index], gr.skip())

    def test_editing_a_user_turn_keeps_history_when_no_model_is_loaded(self):
        """A refused edit must not truncate the conversation it cannot replace."""

        app.MANAGER = self.original  # nothing loaded
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
        app.MANAGER = self.original
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

            def reserve_generation(self):
                return False

            def release_generation(self):  # pragma: no cover - never reached
                raise AssertionError("released a slot it never held")

        app.MANAGER = Sniped()
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
        self.assertFalse(app.MANAGER.busy, "the slot must not leak")

    def test_a_cancelled_assistant_edit_releases_the_slot(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "reply")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "reply", "value": "fixed"}
        )
        stream = app.edit_message(event, "", turns, *SETTINGS)
        next(stream)
        stream.close()
        self.assertFalse(app.MANAGER.busy, "GeneratorExit must release the slot")

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
            model_id = "fake/model"
            load_id = "fake/model#1"

            def reserve_generation(self):
                return True

            def release_generation(self):
                pass

            def generate(self, *_args, **_kwargs):
                raise RuntimeError("out of memory")
                yield  # pragma: no cover - makes this a generator

        app.MANAGER = Exploding()
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
                load_id=app.MANAGER.load_id,
                reasoning_prefilled=True,
            )
            yield GenerationUpdate(
                text="Let me add two and two.</think>Four.",
                metrics=[],
                load_id=app.MANAGER.load_id,
                reasoning_prefilled=True,
            )

        app.MANAGER.generate = thinking
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
                text="Four.", metrics=[], load_id=app.MANAGER.load_id
            )

        app.MANAGER.generate = plain
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["reasoning"], "")
        self.assertEqual(final[TURNS][1]["content"], "Four.")

    def test_a_failure_after_some_tokens_keeps_them(self):
        def failing(*_args, **_kwargs):
            yield GenerationUpdate(
                text="<think>Hmm", metrics=[], load_id=app.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        app.MANAGER.generate = failing
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
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def respond(self, turns=()):
        """Stream one whole response and return its frames."""

        return list(app.chat("hi", list(turns), *SETTINGS))

    def assertDropped(self, payload):
        self.assertEqual(app.inspect_token(payload, select(0)), (gr.skip(), gr.skip()))

    def initial_metrics_state(self):
        """The value a fresh session starts inspect_token()'s input with."""

        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "inspect_token"
        )
        (state_block,) = listener.inputs
        return state_block.value

    def test_a_selection_against_the_strip_on_screen_is_published(self):
        payload = self.respond()[-1][METRICS]
        detail, alternatives = app.inspect_token(payload, select(0))
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
        detail, _alternatives = app.inspect_token(frames[1][METRICS], select(0))
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
        detail, _alternatives = app.inspect_token(payload, select(0))
        self.assertIn("Token 1", detail)

    def test_an_out_of_range_index_still_reports_the_token_as_gone(self):
        payload = self.respond()[-1][METRICS]
        detail, alternatives = app.inspect_token(payload, select(99))
        self.assertIn("no longer available", detail)
        self.assertEqual(alternatives, [])

    def test_an_empty_strip_asks_for_a_selection(self):
        detail, alternatives = app.inspect_token(app.empty_metrics(), select(0))
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
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

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
        detail, _alternatives = app.inspect_token(prompt_payload, select(0))
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
                text="Hmm", metrics=[], load_id=app.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        app.MANAGER.generate = failing
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
        strip, prompt_strip, caption = app.recolor(
            frames[-1][METRICS], self.prompt_payload(frames), "Surprise"
        )
        self.assertEqual(len(strip["value"]), 3)
        self.assertTrue(prompt_strip["value"])
        self.assertTrue(caption)


class CancellationTests(unittest.TestCase):
    """What the Stop button does: Gradio closes the running generator."""

    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def test_closing_mid_stream_releases_the_model_lock(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        next(stream)
        stream.close()

        acquired = app.MANAGER._lock.acquire(blocking=False)
        self.assertTrue(acquired, "the model lock survived cancellation")
        app.MANAGER._lock.release()

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
        app.MANAGER = loaded_manager([0, 2, 3], THINK_PIECES, THINK_EOS)
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        frame = next(stream)
        stream.close()

        self.assertFalse(frame[TURNS][1]["reasoning_closed"])
        messages, turns, _send, _stop, status, _source = app.stop_generation(
            frame[TURNS], frame[METRICS], frame[CONTEXT_IDS]
        )
        self.assertTrue(turns[1]["reasoning_closed"])
        thoughts = [m for m in messages if m.get("metadata", {}).get("title")]
        self.assertEqual(thoughts[0]["metadata"]["status"], "done")
        self.assertEqual(status, "Stopped. The partial response was kept.")

    def test_stopping_before_any_token_drops_the_empty_turn(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "")]
        messages, remaining, _send, _stop, status, _source = app.stop_generation(
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
        self.assertEqual(strip_of(strip), [])
        self.assertEqual(metrics_of(metrics), [])
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(alts, [])
        # Undo cancels the generator, which then never reaches its final yield.
        self.assertEqual((send, stop), app.send_stop_buttons(False))

    def test_undo_with_nothing_to_remove_keeps_the_token_panel(self):
        result = app.undo_last([make_turn("assistant", "orphan")])
        self.assertEqual(result[5], "There is nothing to undo.")
        for index in (3, 4, 6, 7):
            self.assertEqual(result[index], gr.skip())
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
        self.assertEqual(strip_of(strip), [])
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

        Really acquiring app.MANAGER._generating would model a running
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
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)
        # ModelManager.busy reads this flag, so the real property is exercised.
        app.MANAGER._generating = self.HeldLock()
        self.assertTrue(app.MANAGER.busy)

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


class BusyFlagTests(unittest.TestCase):
    """The flag the refusal reads has to follow a real generation."""

    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def test_an_idle_manager_is_not_busy(self):
        self.assertFalse(app.MANAGER.busy)

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
        self.assertFalse(app.MANAGER.busy)
        next(stream)
        self.assertTrue(app.MANAGER.busy, "the first frame left the slot free")

    def test_the_manager_is_busy_while_streaming(self):
        stream = self.streaming()
        next(stream)
        next(stream)
        self.assertTrue(app.MANAGER.busy)
        # Stop closes the generator, which unwinds generate_reply() and frees it.
        stream.close()
        self.assertFalse(app.MANAGER.busy)

    def test_a_finished_generation_leaves_the_manager_free(self):
        list(app.chat("hi", [], *SETTINGS))
        self.assertFalse(app.MANAGER.busy)

    def test_a_failed_generation_leaves_the_manager_free(self):
        """The reservation outlives the runtime, so its release must too.

        Replacing generate() takes the manager's own bookkeeping out of the
        picture: what frees the slot here is generate_reply()'s finally.
        """

        def failing(*_args, **_kwargs):
            yield GenerationUpdate(
                text="Hmm", metrics=[], load_id=app.MANAGER.load_id
            )
            raise RuntimeError("gpu fell over")

        app.MANAGER.generate = failing
        stream = app.chat("hi", [], *SETTINGS)
        next(stream)
        self.assertTrue(app.MANAGER.busy)
        list(stream)
        self.assertFalse(app.MANAGER.busy)

    def test_cancelling_the_first_frame_frees_the_slot(self):
        """Stop before a single token: the reservation is already outstanding."""

        stream = self.streaming()
        next(stream)
        stream.close()
        self.assertFalse(app.MANAGER.busy)

    def test_a_cancelled_generation_does_not_wedge_the_app(self):
        """The regression to fear: every later Send refused, forever."""

        stream = self.streaming()
        next(stream)
        stream.close()

        final = list(app.chat("again", [], *SETTINGS))[-1]
        self.assertNotEqual(final[STATUS], app.BUSY_STATUS)
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        self.assertFalse(app.MANAGER.busy)

    def test_cancelling_still_releases_the_model_lock(self):
        stream = self.streaming()
        next(stream)
        stream.close()

        acquired = app.MANAGER._lock.acquire(blocking=False)
        self.assertTrue(acquired, "the model lock survived cancellation")
        app.MANAGER._lock.release()

    def test_a_direct_generate_call_reserves_the_slot_itself(self):
        """Nothing above generate() has reserved anything here.

        The runtime is used directly by tests and could be used directly by a
        non-streaming caller, so it still has to claim - and free - the slot
        when it finds it available, without deadlocking against the reservation
        generate_reply() normally holds on its behalf.
        """

        stream = app.MANAGER.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=8192,
            seed=42,
        )
        self.addCleanup(stream.close)
        next(stream)
        self.assertTrue(app.MANAGER.busy)
        stream.close()
        self.assertFalse(app.MANAGER.busy)


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
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

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
        self.original = app.MANAGER
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def use(self, script, pieces=THINK_PIECES, eos_id=THINK_EOS):
        app.MANAGER = loaded_manager(script, pieces, eos_id)

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
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def unloaded(self):
        app.MANAGER = self.original
        self.assertFalse(app.MANAGER.loaded)

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
) = range(18)


def contents(turns):
    return [turn["content"] for turn in turns]


def cell(row):
    """A click on one row of the alternatives table."""

    return gr.SelectData(None, {"index": [row, 1], "value": "x"})


class BranchFromTokenTests(unittest.TestCase):
    """Replay a response up to a token, swap in an alternative, and continue."""

    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def respond(self):
        return list(app.chat("hi", [], *SETTINGS))

    def pick_alternative(self, final, strip_index=1, row=1):
        """Click a response token, then a row of its alternatives."""

        selected = app.remember_selection(final[METRICS], select(strip_index))
        detail, pick = app.choose_alternative(
            final[METRICS], selected, final[BRANCH_SOURCE], cell(row)
        )
        return detail, pick

    def test_a_finished_response_is_branchable(self):
        frames = self.respond()
        self.assertIsNone(frames[0][BRANCH_SOURCE])
        for frame in frames[1:-1]:
            self.assertEqual(frame[BRANCH_SOURCE], gr.skip())
        self.assertEqual(
            frames[-1][BRANCH_SOURCE],
            (frames[-1][METRICS][0], app.MANAGER.load_id),
        )

    def test_a_load_finishing_before_the_final_snapshot_cannot_claim_the_tokens(self):
        manager = app.MANAGER
        real_generate = manager.generate
        producing_load_id = manager.load_id

        def load_after_generation(*args, **kwargs):
            yield from real_generate(*args, **kwargs)
            # A load waiting on the model lock can finish as soon as the
            # runtime generator exits, before _stream_reply builds its final
            # branchable snapshot.
            manager.load_count += 1

        manager.generate = load_after_generation
        final = self.respond()[-1]

        self.assertNotEqual(manager.load_id, producing_load_id)
        self.assertEqual(
            final[BRANCH_SOURCE], (final[METRICS][0], producing_load_id)
        )

    def test_a_response_token_is_remembered_for_the_table(self):
        final = self.respond()[-1]
        selected = app.remember_selection(final[METRICS], select(1))
        self.assertEqual(selected, {"generation": final[METRICS][0], "index": 1})

    def test_a_prompt_token_clears_the_remembered_position(self):
        # Otherwise a click in the prompt token's table would pair its row with
        # the response token remembered earlier.
        frames = self.respond()
        prompt_payload = frames[1][PROMPT_METRICS]
        self.assertIsNone(app.remember_selection(prompt_payload, select(0)))

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
        # replace; the branch source stamp is what says so.
        final = self.respond()[-1]
        selected = app.remember_selection(final[METRICS], select(1))
        detail, pick = app.choose_alternative(final[METRICS], selected, None, cell(1))
        self.assertIn(app.BRANCH_UNAVAILABLE, detail)
        self.assertIsNone(pick)

    def test_a_row_without_a_remembered_token_does_nothing(self):
        final = self.respond()[-1]
        detail, pick = app.choose_alternative(
            final[METRICS], None, final[BRANCH_SOURCE], cell(0)
        )
        self.assertEqual(detail, gr.skip())
        self.assertIsNone(pick)

    def test_branching_replays_the_prefix_and_continues(self):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        frames = list(
            app.branch_from(
                pick, final[BRANCH_SOURCE], final[METRICS], "", final[TURNS], *SETTINGS
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
        self.assertEqual(
            last[BRANCH_SOURCE], (last[METRICS][0], app.MANAGER.load_id)
        )

    def test_branching_preserves_literal_assistant_prefill_tags(self):
        app.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        _detail, pick = self.pick_alternative(original, strip_index=3, row=0)

        branched = list(
            app.branch_from(
                pick,
                original[BRANCH_SOURCE],
                original[METRICS],
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
        app.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        original_metrics = metrics_of(original[METRICS])
        pick = {
            "generation": original[METRICS][0],
            "position": 2,
            "token_id": THINK_EOS,
            "original_id": original_metrics[1]["token_id"],
            "text": "<eos>",
            "original": original_metrics[1]["text"],
        }

        branched = list(
            app.branch_from(
                pick,
                original[BRANCH_SOURCE],
                original[METRICS],
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
                pick, second[BRANCH_SOURCE], second[METRICS], "", second[TURNS], *SETTINGS
            )
        )[-1]
        self.assertEqual(
            [turn["content"] for turn in last[TURNS][:3]],
            [turn["content"] for turn in second[TURNS][:3]],
        )
        self.assertEqual(len(last[TURNS]), 4)

    def test_a_pick_made_against_a_replaced_strip_is_refused(self):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        fresh = self.respond()[-1]
        frames = list(
            app.branch_from(
                pick, fresh[BRANCH_SOURCE], fresh[METRICS], "", fresh[TURNS], *SETTINGS
            )
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_HINT)
        self.assertEqual(frames[0][TURNS], fresh[TURNS])

    def test_branching_with_nothing_picked_explains_the_steps(self):
        final = self.respond()[-1]
        frames = list(
            app.branch_from(
                None, final[BRANCH_SOURCE], final[METRICS], "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(frames[0][STATUS], app.BRANCH_HINT)

    def test_branching_is_refused_while_a_response_is_generating(self):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        self.assertTrue(app.MANAGER.reserve_generation())
        try:
            frames = list(
                app.branch_from(
                    pick, final[BRANCH_SOURCE], final[METRICS], "", final[TURNS], *SETTINGS
                )
            )
        finally:
            app.MANAGER.release_generation()
        self.assertEqual(frames[0][STATUS], app.BUSY_STATUS)
        self.assertEqual(frames[0][TURNS], gr.skip())

    def test_a_stopped_response_is_branchable(self):
        settings = dict(FIXED, max_new_tokens=8192)
        stream = app.chat("hi", [], *settings.values())
        next(stream)
        frame = next(stream)
        stream.close()
        producing_load_id = frame[CONTEXT_IDS][2]
        app.MANAGER.load_count += 1
        *_rest, source = app.stop_generation(
            frame[TURNS], frame[METRICS], frame[CONTEXT_IDS]
        )
        self.assertEqual(source, (frame[METRICS][0], producing_load_id))

    def test_stopping_before_any_token_leaves_nothing_to_branch(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "")]
        *_rest, source = app.stop_generation(turns, app.empty_metrics())
        self.assertIsNone(source)

    def branch_text(self, final, text, strip_index=1):
        selected = app.remember_selection(final[METRICS], select(strip_index))
        return list(
            app.branch_with_text(
                selected,
                final[BRANCH_SOURCE],
                final[METRICS],
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
        self.assertEqual(
            last[BRANCH_SOURCE], (last[METRICS][0], app.MANAGER.load_id)
        )

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
        self.assertEqual(last[TRACE]["sampling"]["forced_prefix_tokens"], 1)

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
                None, final[BRANCH_SOURCE], final[METRICS], "Hello", "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_HINT)

    def test_typed_text_against_a_replaced_strip_is_refused(self):
        final = self.respond()[-1]
        selected = app.remember_selection(final[METRICS], select(1))
        fresh = self.respond()[-1]
        frames = list(
            app.branch_with_text(
                selected, fresh[BRANCH_SOURCE], fresh[METRICS], "Hello", "", fresh[TURNS], *SETTINGS
            )
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_HINT)

    def test_typed_text_from_an_earlier_model_load_is_refused(self):
        final = self.respond()[-1]
        # Loading leaves the old response strip on screen, but its token IDs
        # belong to the tokenizer that produced it, even for a same-ID reload.
        app.MANAGER.load_count += 1
        frames = self.branch_text(final, "Hello")
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_HINT)
        self.assertEqual(frames[0][TURNS], final[TURNS])

    def reload_before(self, method_name):
        """Land a model load inside the branch handler, after its stamp check.

        The handler compares ``branch_source`` against the live load first,
        then calls the runtime; a load finishing in between passes that check
        and must be caught by the runtime's own comparison under the model
        lock. The wrapped method bumps the load count at the moment of the
        call, which is exactly that window.
        """

        manager = app.MANAGER
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
        self.assertFalse(app.MANAGER.busy)

    def test_a_load_landing_before_the_encoding_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        self.reload_before("encode_replacement")
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

        real = app.MANAGER.encode_replacement

        def observe(*args, **kwargs):
            seen.append(app.MANAGER.busy)
            return real(*args, **kwargs)

        app.MANAGER.encode_replacement = observe

    def test_typed_text_is_refused_while_a_response_is_generating(self):
        final = self.respond()[-1]
        encodings = []
        self.record_encodings(encodings)
        self.assertTrue(app.MANAGER.reserve_generation())
        try:
            frames = self.branch_text(final, "Hello")
        finally:
            app.MANAGER.release_generation()
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
        self.assertFalse(app.MANAGER.busy, "the slot must not leak")

    def test_typed_text_against_a_strip_replaced_as_the_slot_is_taken_is_refused(
        self,
    ):
        """A generation finishing between the click and the reservation.

        Its final frame re-stamped the strip. The stamps are compared only once
        the slot is owned, so the comparison reads the strip the replacement
        would actually land on.
        """

        final = self.respond()[-1]
        manager = app.MANAGER
        real = manager.reserve_generation

        def replace_strips_first():
            app.new_metrics_generation()
            return real()

        manager.reserve_generation = replace_strips_first
        encodings = []
        self.record_encodings(encodings)
        frames = self.branch_text(final, "Hello")
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][STATUS], app.BRANCH_TEXT_HINT)
        self.assertEqual(frames[0][TURNS], final[TURNS])
        self.assertEqual(encodings, [])
        self.assertFalse(manager.busy, "a refusal must give the slot back")

    def test_a_cancelled_typed_branch_releases_the_slot(self):
        final = self.respond()[-1]
        selected = app.remember_selection(final[METRICS], select(1))
        stream = app.branch_with_text(
            selected,
            final[BRANCH_SOURCE],
            final[METRICS],
            "Hello",
            "",
            final[TURNS],
            *SETTINGS,
        )
        first = next(stream)
        self.assertIn("Generating", first[STATUS])
        self.assertTrue(app.MANAGER.busy)
        stream.close()
        self.assertFalse(app.MANAGER.busy, "GeneratorExit must release the slot")

    def test_a_load_landing_before_a_picked_replay_leaves_the_conversation_alone(
        self,
    ):
        final = self.respond()[-1]
        _detail, pick = self.pick_alternative(final)
        self.reload_before("generate")
        frames = list(
            app.branch_from(
                pick, final[BRANCH_SOURCE], final[METRICS], "", final[TURNS], *SETTINGS
            )
        )
        self.assertEqual(len(frames), 2)
        self.assertBranchRefusedByReload(frames, final)

    def test_typed_text_keeps_literal_prefill_tags_before_it(self):
        app.MANAGER = loaded_manager(
            [0, 2, 1, 3, THINK_EOS], THINK_PIECES, THINK_EOS
        )
        settings = dict(FIXED, assistant_prefill="<think>Hello</think>")
        original = list(app.chat("hi", [], *settings.values()))[-1]
        selected = app.remember_selection(original[METRICS], select(3))
        branched = list(
            app.branch_with_text(
                selected,
                original[BRANCH_SOURCE],
                original[METRICS],
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

    def sentencepiece_response(self):
        """A response from a tokenizer that drops the first decoded space."""

        pieces = ["\u2581Hello", "\u2581world", "world", "\u2581", "!", "<eos>"]
        eos = pieces.index("<eos>")
        app.MANAGER = loaded_manager([0, 4, eos], pieces, eos)
        app.MANAGER.tokenizer = SentencePieceTokenizer(pieces, eos)
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

    def test_the_branch_text_button_is_wired_as_a_generation(self):
        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "branch_with_text"
        )
        self.assertEqual(len(listener.inputs), 4 + 2 + len(SETTINGS))
        self.assertEqual(len(listener.outputs), CHAT_OUTPUTS)

    def test_the_branch_button_is_wired_as_a_generation(self):
        demo = app.build_app()
        listener = next(
            fn
            for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "branch_from"
        )
        self.assertEqual(len(listener.inputs), 3 + 2 + len(SETTINGS))
        self.assertEqual(len(listener.outputs), CHAT_OUTPUTS)


class ForkTests(unittest.TestCase):
    """Copy the transcript into a second fork and move between them."""

    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def turns(self):
        return [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]

    def test_forking_copies_the_conversation_into_a_new_fork(self):
        result = app.fork_conversation(self.turns(), new_forks(), None)
        self.assertEqual(len(result), 18)
        self.assertEqual(contents(result[FORK_TURNS]), contents(self.turns()))
        self.assertEqual(result[FORK_STATE]["active"], "Fork 1")
        self.assertEqual(
            contents(result[FORK_STATE]["branches"][MAIN_BRANCH]),
            contents(self.turns()),
        )
        self.assertEqual(result[FORK_PICKER]["choices"], [MAIN_BRANCH, "Fork 1"])
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
        self.assertEqual(strip_of(result[FORK_STRIP]), [])
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
        self.assertEqual(strip_of(result[FORK_STRIP]), [])

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
        self.assertEqual(result[FORK_PICKER]["choices"], [MAIN_BRANCH])
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
        self.assertEqual(result[-2], new_forks())
        self.assertEqual(result[-1]["choices"], [MAIN_BRANCH])

    def test_a_forked_conversation_can_be_continued(self):
        forked = app.fork_conversation(self.turns(), new_forks(), None)
        last = list(app.chat("three", forked[FORK_TURNS], *SETTINGS))[-1]
        self.assertEqual(len(last[TURNS]), 6)
        self.assertEqual(last[TURNS][4]["content"], "three")


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

        state, _metrics, _context = self.named("stop_generation").inputs
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
                "load_conversation",
                "branch_from",
                "branch_with_text",
                "fork_conversation",
                "switch_fork",
                "delete_fork",
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
        self.assertEqual(prefill.value, None)
        self.assertIn("closes the reasoning block", prefill.info)


if __name__ == "__main__":
    unittest.main()


class LayerInspectionTests(unittest.TestCase):
    """The logit lens and attention panel behind the Inspect layers button."""

    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)
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

        app.MANAGER.inspect = fake_inspect

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
        self.assertEqual(load, app.MANAGER.load_id)
        # Later frames leave the ids alone: the prompt never changes mid-stream.
        self.assertEqual(frames[-1][CONTEXT_IDS], gr.skip())

    def test_scored_text_publishes_its_context_ids(self):
        result = app.score_text("", "Hello", False, DEFAULT_COLOR_SCALE)
        stamp, ids, load = result[10]
        self.assertEqual(stamp, result[1][0])
        self.assertEqual(ids, [])
        self.assertEqual(load, app.MANAGER.load_id)

    def test_a_response_token_is_inspected_in_its_full_sequence(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(1))
        self.assertEqual(target, {"generation": final[METRICS][0], "strip": "response", "index": 1})

        lens, attention, slider, insight, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(self.calls, [([0, 2, 3, THINK_EOS], 2, 1)])
        # The load id goes along so the runtime can check it under its lock.
        self.assertEqual(self.load_ids, [app.MANAGER.load_id])
        self.assertIn("logit-lens", lens)
        self.assertIn("attention-view", attention)
        self.assertEqual(slider, gr.update(maximum=2, value=0))
        self.assertEqual(insight["index"], 2)
        self.assertIn("Token 2", status)

    def test_an_output_only_lens_says_why_in_the_status(self):
        final = self.finished()
        real_inspect = app.MANAGER.inspect

        def output_only(sequence, index, *, context_count=0, load_id=None):
            insight = real_inspect(sequence, index, context_count=context_count)
            return replace(insight, layers=insight.layers[-1:], decided_at=None)

        app.MANAGER.inspect = output_only
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

        app.MANAGER.inspect = refuse
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        lens, *_rest, status = self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(lens, gr.skip())
        self.assertEqual(status, "Could not inspect that token: out of memory")

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
        self.assertTrue(app.MANAGER.reserve_generation())
        try:
            *_rest, status = self.inspect(
                target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
            )
        finally:
            app.MANAGER.release_generation()
        self.assertEqual(status, app.INSPECT_BUSY)
        self.assertEqual(self.calls, [])

    def test_the_pass_holds_the_generation_slot_and_gives_it_back(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        seen = []

        def observe(sequence, index, *, context_count=0, load_id=None):
            seen.append(app.MANAGER.busy)
            return self.fake(sequence, index, context_count=context_count)

        self.fake, app.MANAGER.inspect = app.MANAGER.inspect, observe
        self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertEqual(seen, [True])
        self.assertFalse(app.MANAGER.busy)

        def fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        app.MANAGER.inspect = fail
        self.inspect(
            target, final[METRICS], final[PROMPT_METRICS], final[CONTEXT_IDS], 0
        )
        self.assertFalse(app.MANAGER.busy)

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
        self.assertTrue(app.MANAGER.busy)
        self.assertEqual(list(frames), [])
        self.assertFalse(app.MANAGER.busy)

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
        self.assertFalse(app.MANAGER.busy)

    def test_a_strip_replaced_during_the_pass_is_not_described(self):
        final = self.finished()
        target = app.remember_inspect_target("response")(final[METRICS], select(0))
        original = app.MANAGER.inspect

        def replace_strips(sequence, index, *, context_count=0, load_id=None):
            # Clear, Undo and friends do not take the generation slot; they
            # mint a new stamp, which is what the handler has to notice.
            app.new_metrics_generation()
            return original(sequence, index, context_count=context_count)

        app.MANAGER.inspect = replace_strips
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
        app.MANAGER.load_count += 1
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

        app.MANAGER.inspect = reloaded
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
