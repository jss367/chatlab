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

    def test_a_refused_send_keeps_the_token_diagnostics(self):
        # gr.skip() leaves the previous response's panel on screen.
        final = self.last(app.chat("   ", [], *SETTINGS))[-1]
        self.assertEqual(final[STATUS], "Enter a message first.")
        for index in (STRIP, METRICS, DETAIL, ALTS):
            self.assertEqual(final[index], gr.skip())

    def test_editing_a_reasoning_block_leaves_the_answer_alone(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "answer", "thought")]
        event = gr.EditData(
            None, {"index": 1, "previous_value": "thought", "value": "revised"}
        )
        final = self.last(app.edit_message(event, "", turns, *SETTINGS))[-1]
        self.assertEqual(final[TURNS][1]["reasoning"], "revised")
        self.assertEqual(final[TURNS][1]["content"], "answer")

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
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0], "two")
        self.assertEqual([turn["content"] for turn in result[2]], ["one", "first"])

    def test_undo_clears_the_token_panel(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        _prompt, _messages, _turns, strip, metrics, _status, detail, alts = (
            app.undo_last(turns)
        )
        self.assertEqual(strip, [])
        self.assertEqual(metrics, [])
        self.assertEqual(detail, app.NO_TOKEN_SELECTED)
        self.assertEqual(alts, [])

    def test_undo_with_nothing_to_remove_keeps_the_token_panel(self):
        result = app.undo_last([make_turn("assistant", "orphan")])
        self.assertEqual(result[5], "There is nothing to undo.")
        for index in (3, 4, 6, 7):
            self.assertEqual(result[index], gr.skip())

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


class SaveLoadTests(unittest.TestCase):
    def test_save_then_load_round_trips(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "there", "thought")]
        update, status = app.save_conversation(turns, "Be terse.")
        self.assertIn("Saved 2 messages", status)

        messages, restored, system_prompt, strip, _metrics, load_status = (
            app.load_conversation(update["value"])
        )
        self.assertEqual(restored, turns)
        self.assertEqual(system_prompt, "Be terse.")
        self.assertEqual(len(messages), 3)
        self.assertEqual(strip, [])
        self.assertIn("Loaded 2 messages", load_status)

    def test_saving_an_empty_conversation_is_refused(self):
        update, status = app.save_conversation([], "")
        self.assertFalse(update["visible"])
        self.assertIn("nothing to save", status)

    def test_loading_a_bad_file_reports_the_problem(
        self,
    ):
        result = app.load_conversation("/nonexistent/conversation.json")
        self.assertIn("Could not load that file", result[5])


if __name__ == "__main__":
    unittest.main()
