"""Reply requests preserve runtime inputs, replay rollback and update ownership."""

from dataclasses import replace
from itertools import count
from types import SimpleNamespace
import unittest
from unittest import mock

import gradio as gr

from chatlab.conversation import make_turn
from chatlab.generation_request import PromptOptions, ReplayOptions, ReplyRequest, SamplingOptions
from chatlab.text_generation import GenerationUpdate
from chatlab.ui import runtime
from chatlab.ui.reply_state import ReplyState
from chatlab.ui.reply_stream import stream_reply


class ReplyBoundaryTests(unittest.TestCase):
    def request(self, **options):
        return ReplyRequest(
            turns=[make_turn("user", "Question")],
            prompt_text="draft",
            **options,
        )

    def test_randomized_seed_is_resolved_once_for_runtime_and_record(self):
        with mock.patch("chatlab.ui.reply_state.resolve_seed", return_value=123) as resolve:
            state = ReplyState.prepare(self.request(), count(1).__next__)
            arguments = state.request.arguments(state.used_seed, state.steering)
            recorded = state.sampling_record()
        resolve.assert_called_once()
        self.assertEqual(arguments["seed"], 123)
        self.assertEqual(state.pending["generation_settings"]["seed"], 123)
        self.assertEqual(recorded["seed"], 123)

    def test_replay_uses_recorded_mode_and_ignores_current_prefill(self):
        edit = {"ids": [4, 5], "position": 2, "original": "a", "replacement": "b"}
        request = self.request(
            prompt=PromptOptions(assistant_prefill="new draft", thinking_mode="on"),
            sampling=SamplingOptions(randomize_seed=False),
            replay=ReplayOptions(
                forced_ids=(7, 8),
                prompt_edit=edit,
                replaying=True,
                expected_load_id="model#1",
                thinking_mode="off",
                literal_text_ranges=((1, 2),),
                automatic_reasoning_close_tokens=1,
            ),
        )
        state = ReplyState.prepare(request, count(1).__next__)
        calls = []

        def generate(messages, **arguments):
            calls.append((messages, arguments))
            yield GenerationUpdate("reply", [], "model#1", thinking_mode="off")

        manager = SimpleNamespace(load_id="model#1", generate=generate)
        with mock.patch.object(runtime, "MANAGER", manager):
            final = list(stream_reply(request))[-1]
        arguments = calls[0][1]
        self.assertEqual(arguments["forced_ids"], (7, 8))
        self.assertEqual(arguments["prompt_override_ids"], [4, 5])
        self.assertEqual(arguments["answer_prefill"], "")
        self.assertEqual(arguments["thinking_mode"], "off")
        self.assertEqual(arguments["load_id"], "model#1")
        self.assertEqual(arguments["literal_text_ranges"], ((1, 2),))
        self.assertEqual(arguments["automatic_reasoning_close_tokens"], 1)
        self.assertNotIn("steering", arguments)
        self.assertEqual(
            final["turns"][-1]["generation_settings"], state.pending["generation_settings"]
        )
        self.assertEqual(state.sampling_record()["requested_thinking_mode"], "on")

    def test_replay_keeps_previous_diagnostics_until_an_update_commits(self):
        previous = [make_turn("user", "Question"), make_turn("assistant", "old answer")]
        previous[-1]["metrics_generation"] = 99
        request = self.request(previous_turns=previous, replay=ReplayOptions(replaying=True))
        stamp = mock.Mock(return_value=100)
        state = ReplyState.prepare(request, stamp)
        stamp.assert_not_called()
        self.assertIs(state.visible_turns, previous)
        self.assertIsNone(state.generation)
        state.accept(GenerationUpdate("new answer", [], "model#1"), stamp)
        stamp.assert_called_once()
        self.assertIs(state.visible_turns, state.turns)
        self.assertEqual(state.generation, 100)
        self.assertEqual(previous[-1]["content"], "old answer")
        self.assertEqual(request.turns, [make_turn("user", "Question")])

    def test_failed_replay_keeps_previous_turns_and_skips_their_panels(self):
        previous = [make_turn("user", "Question"), make_turn("assistant", "old answer")]
        request = self.request(previous_turns=previous, replay=ReplayOptions(replaying=True))

        def refused(messages, **arguments):
            raise RuntimeError("replay failed")
            yield

        manager = SimpleNamespace(load_id="model#1", generate=refused)
        with (
            mock.patch.object(runtime, "MANAGER", manager),
            self.assertLogs("chatlab.ui.reply_stream", "ERROR"),
        ):
            frames = list(stream_reply(request))
        for frame in frames:
            self.assertEqual(frame["turns"], previous)
            for panel in ("metrics", "prompt_metrics", "trace", "selected_token"):
                self.assertEqual(frame[panel], gr.skip())
        self.assertIn("replay failed", frames[-1]["status"])

    def test_model_updates_do_not_mutate_previous_measurement_frames(self):
        state = ReplyState.prepare(self.request(), count(1).__next__)
        live_metrics = [{"token_id": 4}]
        state.accept(GenerationUpdate("a", live_metrics, "model#1"), count(2).__next__)
        published = state.pending["tokens"]
        live_metrics.append({"token_id": 5})
        self.assertEqual(published, [{"token_id": 4}])
        state.first = False
        state.accept(GenerationUpdate("ab", live_metrics, "model#1"), count(2).__next__)
        self.assertEqual(published, [{"token_id": 4}])
        self.assertEqual(state.pending["tokens"], [{"token_id": 4}, {"token_id": 5}])

    def test_cancelling_after_a_frame_closes_the_runtime_iterator(self):
        closed = []

        def generate(messages, **arguments):
            try:
                yield GenerationUpdate("partial", [], "model#1")
                self.fail("A cancelled stream must not resume its model")
            finally:
                closed.append(True)

        manager = SimpleNamespace(load_id="model#1", generate=generate)
        with mock.patch.object(runtime, "MANAGER", manager):
            stream = stream_reply(self.request())
            next(stream)
            partial = next(stream)
            stream.close()
        self.assertEqual(closed, [True])
        self.assertEqual(partial["turns"][-1]["content"], "partial")

    def test_a_prompt_edit_still_applies_prefill_when_it_is_not_a_token_replay(self):
        request = self.request(prompt=PromptOptions(assistant_prefill="answer:"))
        request = replace(
            request,
            replay=ReplayOptions(
                prompt_edit={"ids": (4,), "position": 1, "original": "a", "replacement": "b"},
                expected_load_id="model#1",
            ),
        )
        state = ReplyState.prepare(request, count(1).__next__)
        arguments = request.arguments(state.used_seed, state.steering)
        self.assertEqual(arguments["answer_prefill"], "answer:")
        self.assertEqual(arguments["prompt_override_ids"], (4,))
        self.assertEqual(state.pending["generation_settings"]["assistant_prefill"], "answer:")

    def test_invalid_seed_preserves_the_existing_fallback(self):
        for value in (None, "invalid", float("inf"), -10):
            with self.subTest(seed=value):
                request = self.request(sampling=SamplingOptions(seed=value, randomize_seed=False))
                state = ReplyState.prepare(request, count(1).__next__)
                self.assertEqual(request.arguments(state.used_seed, state.steering)["seed"], 0)


if __name__ == "__main__":
    unittest.main()
