import inspect
import unittest

import gradio as gr
import numpy as np

import app
import charts
from conversation import display_messages, make_turn, model_messages
from model_runtime import GenerationUpdate
from token_metrics import DEFAULT_COLOR_SCALE

from test_streaming import loaded_manager


# "Hello" and " world" are the answer; the reasoning tags are their own tokens.
THINK_PIECES = ["<think>", "</think>", "Hello", " world", "<eos>"]
THINK_EOS = 4

FIXED = {
    "system_prompt": "",
    "keep_reasoning": False,
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
) = range(len(app.CHAT_OUTPUT_NAMES))
CHAT_OUTPUTS = len(app.CHAT_OUTPUT_NAMES)

# The panels every conversation-replacing handler resets after its own rows:
# the prompt strip and its state and note, the two charts, and the export.
PANEL_OUTPUTS = 6
UNDO_OUTPUTS = 10 + PANEL_OUTPUTS
CLEAR_OUTPUTS = 9 + PANEL_OUTPUTS
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
                reasoning_prefilled=True,
            )
            yield GenerationUpdate(
                text="Let me add two and two.</think>Four.",
                metrics=[],
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
            yield GenerationUpdate(text="Four.", metrics=[])

        app.MANAGER.generate = plain
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["reasoning"], "")
        self.assertEqual(final[TURNS][1]["content"], "Four.")

    def test_a_failure_after_some_tokens_keeps_them(self):
        def failing(*_args, **_kwargs):
            yield GenerationUpdate(text="<think>Hmm", metrics=[])
            raise RuntimeError("gpu fell over")

        app.MANAGER.generate = failing
        final = self.last(app.chat("hi", [], *SETTINGS))[-1]
        self.assertEqual([turn["role"] for turn in final[TURNS]], ["user", "assistant"])
        reply = final[TURNS][1]
        self.assertEqual(reply["reasoning"], "Hmm")
        self.assertEqual(reply["content"], "")
        self.assertTrue(reply["reasoning_closed"])
        self.assertIn("gpu fell over", final[STATUS])


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
            yield GenerationUpdate(text="Hmm", metrics=[])
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
        messages, turns, _send, _stop, status = app.stop_generation(frame[TURNS])
        self.assertTrue(turns[1]["reasoning_closed"])
        thoughts = [m for m in messages if m.get("metadata", {}).get("title")]
        self.assertEqual(thoughts[0]["metadata"]["status"], "done")
        self.assertEqual(status, "Stopped. The partial response was kept.")

    def test_stopping_before_any_token_drops_the_empty_turn(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "")]
        messages, remaining, _send, _stop, status = app.stop_generation(turns)
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
            yield GenerationUpdate(text="Hmm", metrics=[])
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
        """Stop reads exactly one thing: the conversation state."""

        (state,) = self.named("stop_generation").inputs
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
            },
        )


if __name__ == "__main__":
    unittest.main()
