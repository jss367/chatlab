import inspect
import unittest

import gradio as gr

import app
from conversation import make_turn
from model_runtime import GenerationUpdate
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
}
SETTINGS = tuple(FIXED.values())

PROMPT, CHATBOT, TURNS, STRIP, METRICS, STATUS, SEED, SEND, STOP, DETAIL, ALTS = range(
    11
)


class ChatFlowTests(unittest.TestCase):
    def setUp(self):
        self.original = app.MANAGER
        app.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, app, "MANAGER", self.original)

    def last(self, stream):
        frames = list(stream)
        self.assertTrue(frames)
        for frame in frames:
            self.assertEqual(len(frame), 11)
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
        self.assertEqual(final[STRIP], [])
        self.assertEqual(final[METRICS], [])
        self.assertEqual(final[DETAIL], app.NO_TOKEN_SELECTED)
        self.assertEqual(final[ALTS], [])

    def test_a_new_response_resets_the_selected_token_details(self):
        # The first frame empties the strip, so the token the user had selected
        # in the previous response no longer exists and its probabilities must
        # not stay on screen beside a strip that no longer contains it.
        frames = self.last(app.chat("hi", [], *SETTINGS))
        self.assertEqual(frames[0][STRIP], [])
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
        self.assertEqual(len(result), 10)
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
        ) = app.undo_last(turns)
        self.assertEqual(strip, [])
        self.assertEqual(metrics, [])
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
        self.assertEqual(result[8:], app.send_stop_buttons(False))

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
        self.assertEqual(len(result), 9)
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
        ) = app.load_conversation(update["value"])
        self.assertEqual(restored, turns)
        self.assertEqual(system_prompt, "Be terse.")
        self.assertEqual(len(messages), 3)
        self.assertEqual(strip, [])
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
        result = app.load_conversation("/nonexistent/conversation.json")
        self.assertIn("Could not load that file", result[5])
        # A failed load leaves the token panel alone rather than blanking it.
        self.assertEqual(len(result), 10)
        for index in (0, 1, 2, 3, 4, 6, 7):
            self.assertIsInstance(result[index], gr.skip().__class__)
        self.assertEqual(result[8:], app.send_stop_buttons(False))

    def test_loading_nothing_skips_every_output(self):
        result = app.load_conversation(None)
        self.assertEqual(result[5], "No file chosen.")
        self.assertEqual(len(result), 10)
        for index in (0, 1, 2, 3, 4, 6, 7):
            self.assertIsInstance(result[index], gr.skip().__class__)
        self.assertEqual(result[8:], app.send_stop_buttons(False))


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
