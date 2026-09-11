"""Exercise background jobs through the same Gradio events as the browser."""

import asyncio
import copy
import threading
import unittest
from unittest import mock

import gradio as gr
from gradio.state_holder import SessionState

import app
import library
import settings_sandbox
import ui.conversations
from conversation import MAIN_BRANCH, make_turn, new_forks, put_branch
from ui import runtime
from ui.background import ConversationJob, NAMES
from test_app_flow import SETTINGS, THINK_EOS, THINK_PIECES
from test_streaming import loaded_manager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class BackgroundConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = app.build_app()

    def setUp(self):
        library.library_path().unlink(missing_ok=True)
        self.state = SessionState(self.demo)
        self.view = {}
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        self.manager = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.patch = mock.patch.object(runtime, "MANAGER", self.manager)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.entered, self.release = threading.Event(), threading.Event()
        self.generate = self.manager.generate

        def slow(*args, **kwargs):
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("Test did not release generation")
            yield from self.generate(*args, **kwargs)

        self.manager.generate = slow
        self.chat = self.named("chat")
        self.turns = self.chat.outputs[NAMES["turns"]]
        self.forks = self.named("remember_forks").inputs[1]
        self.job_state = next(c for c in self.chat.inputs if isinstance(c.value, ConversationJob))
        self.job = self.state[self.job_state._id]
        self.addCleanup(self.finish)
        forks = new_forks()
        put_branch(forks, "Chat 1", [make_turn("user", "Another conversation")])
        self.state[self.forks._id] = forks

    def named(self, name):
        return next(fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", "") == name)

    def call(self, name, overrides=None, event_data=None):
        fn = self.named(name)
        inputs = []
        for component in fn.inputs:
            inputs.append(
                None
                if isinstance(component, gr.State)
                else self.view.get(component._id, component.value)
            )
        for i, value in (overrides or {}).items():
            inputs[i] = value
        result = self.loop.run_until_complete(
            self.demo.process_api(
                fn,
                inputs,
                state=self.state,
                event_data=event_data,
            )
        )
        for component, value in zip(fn.outputs, result["data"]):
            if isinstance(component, gr.State):
                continue
            if isinstance(value, dict) and value.get("__type__") == "update":
                if "value" in value:
                    self.view[component._id] = value["value"]
            else:
                self.view[component._id] = value
        return result

    def start(self):
        self.call("chat", {0: "hi", **dict(enumerate(SETTINGS, 2))})
        self.assertTrue(self.entered.wait(2))
        self.assertTrue(self.job.running)
        self.assertTrue(self.manager.busy)

    def finish(self):
        self.release.set()
        if self.job.worker:
            self.job.worker.join(5)
            self.assertFalse(self.job.worker.is_alive())

    def switch(self, name):
        self.call("switch_fork", {0: name})

    def test_switch_away_completes_and_saves_only_the_source(self):
        self.start()
        self.switch("Chat 1")
        self.assertTrue(self.job.running)
        self.assertEqual(self.state[self.turns._id][0]["content"], "Another conversation")
        self.finish()
        self.call("poll")
        self.assertEqual(self.state[self.turns._id][0]["content"], "Another conversation")
        saved = library.read()
        self.assertEqual(saved["active"], "Chat 1")
        self.assertEqual(saved["branches"][MAIN_BRANCH][-1]["content"], "Hello world")
        self.switch(MAIN_BRANCH)
        self.call("poll")
        self.assertEqual(self.state[self.turns._id][-1]["content"], "Hello world")
        self.assertFalse(self.manager.busy)

    def test_new_and_fork_leave_the_original_running(self):
        self.start()
        self.call("new_conversation")
        self.assertTrue(self.job.running)
        self.assertEqual(self.state[self.turns._id], [])
        self.switch(MAIN_BRANCH)
        self.call("fork_conversation")
        self.assertTrue(self.job.running)
        self.assertNotEqual(self.state[self.forks._id]["active"], MAIN_BRANCH)
        self.finish()
        self.call("poll")
        self.assertEqual(
            self.state[self.forks._id]["branches"][MAIN_BRANCH][-1]["content"], "Hello world"
        )

    def test_stop_from_another_conversation_preserves_it_and_releases_model(self):
        self.start()
        self.switch("Chat 1")
        before = copy.deepcopy(self.state[self.turns._id])
        self.call("stop_generation")
        self.assertTrue(self.job.cancel.is_set())
        self.assertTrue(self.manager.busy)
        self.finish()
        self.call("poll")
        self.assertEqual(self.state[self.turns._id], before)
        self.assertFalse(self.manager.busy)
        self.assertTrue(library.read()["branches"][MAIN_BRANCH][-1].get("content"))

    def test_second_send_is_refused_without_losing_draft_or_starting_a_run(self):
        self.start()
        worker = self.job.worker
        self.switch("Chat 1")
        self.call("chat", {0: "Keep this draft", **dict(enumerate(SETTINGS, 2))})
        self.assertIs(self.job.worker, worker)
        self.assertEqual(len(self.state[self.turns._id]), 1)
        status = self.view[self.chat.outputs[NAMES["status"]]._id]
        self.assertIn("Main", status)
        self.assertIn("Stop", status)

    def test_edit_and_clear_cannot_replace_a_running_source(self):
        self.start()
        before = copy.deepcopy(self.state[self.turns._id])
        self.call("undo_last")
        self.assertEqual(self.state[self.turns._id], before)
        self.call("clear_chat")
        self.assertEqual(self.state[self.turns._id], before)
        self.assertTrue(self.job.running)

    def test_another_conversation_can_be_edited_without_stopping_the_source(self):
        self.start()
        self.switch("Chat 1")
        self.call("undo_last")
        self.assertEqual(self.state[self.turns._id], [])
        self.assertTrue(self.job.running)
        self.finish()
        self.call("poll")
        self.assertEqual(self.state[self.turns._id], [])

    def test_completed_response_cannot_reappear_after_undo(self):
        self.start()
        self.finish()
        self.call("poll")
        self.call("undo_last")
        before = copy.deepcopy(self.state[self.turns._id])
        self.call("poll")
        self.switch("Chat 1")
        self.switch(MAIN_BRANCH)
        self.call("poll")
        self.assertEqual(self.state[self.turns._id], before)
        self.assertEqual(library.read()["branches"][MAIN_BRANCH], before)

    def test_run_label_survives_the_conversation_change_listener(self):
        self.start()
        self.call("refresh_conversation_list")
        choices = self.job.choices(self.state[self.forks._id], self.state[self.turns._id])
        self.assertIn("Generating", choices["choices"][0][0])
        self.assertEqual(choices["choices"][0][1], MAIN_BRANCH)

    def test_state_is_independent_between_browser_sessions(self):
        other = SessionState(self.demo)[self.job_state._id]
        self.start()
        self.assertIsNot(other, self.job)
        self.assertFalse(other.running)
        self.assertIsNone(other.owner)

    def test_failure_while_away_keeps_source_and_releases_model(self):
        def broken(*args, **kwargs):
            self.entered.set()
            self.release.wait(5)
            raise RuntimeError("test inference error")
            yield

        self.manager.generate = broken
        self.start()
        self.switch("Chat 1")
        self.finish()
        self.call("poll")
        self.assertFalse(self.manager.busy)
        self.assertEqual(self.state[self.turns._id][0]["content"], "Another conversation")
        self.assertEqual(len(library.read()["branches"][MAIN_BRANCH]), 1)

    def test_completion_during_navigation_cannot_be_overwritten_by_its_snapshot(self):
        self.start()
        original = ui.conversations.copy_turns

        def finish_before_copy(turns):
            self.finish()
            return original(turns)

        with mock.patch.object(ui.conversations, "copy_turns", finish_before_copy):
            self.call("new_conversation")
        self.call("poll")
        self.assertEqual(
            self.state[self.forks._id]["branches"][MAIN_BRANCH][-1]["content"], "Hello world"
        )
        self.assertEqual(library.read()["branches"][MAIN_BRANCH][-1]["content"], "Hello world")

    def test_rapid_return_restores_metrics_even_without_a_new_frame(self):
        self.start()
        epoch = self.state[self.chat.outputs[NAMES["metrics"]]._id][0]
        self.switch("Chat 1")
        self.switch(MAIN_BRANCH)
        self.call("poll")
        self.assertEqual(self.state[self.chat.outputs[NAMES["metrics"]]._id][0], epoch)

    def test_background_frames_do_not_replace_a_draft_after_returning(self):
        self.start()
        self.switch("Chat 1")
        prompt = self.chat.outputs[NAMES["prompt"]]
        self.view[prompt._id] = "My next question"
        self.finish()
        self.call("poll")
        self.switch(MAIN_BRANCH)
        self.call("poll")
        self.assertEqual(self.view[prompt._id], "My next question")

    def test_stop_after_partial_output_closes_iterator_on_the_worker(self):
        partial = threading.Event()
        advance = threading.Event()
        closed = []

        def slow_tokens(*args, **kwargs):
            stream = self.generate(*args, **kwargs)
            try:
                for i, update in enumerate(stream):
                    yield update
                    if i == 0:
                        partial.set()
                        advance.wait(5)
            finally:
                stream.close()
                closed.append(threading.current_thread().name)

        self.manager.generate = slow_tokens
        self.addCleanup(advance.set)
        with mock.patch("model_runtime.STREAM_BATCH_TOKENS", 1):
            self.call("chat", {0: "hi", **dict(enumerate(SETTINGS, 2))})
            self.assertTrue(partial.wait(2))
            self.call("poll")
            self.switch("Chat 1")
            self.call("stop_generation")
            advance.set()
            self.finish()
        self.call("poll")
        self.assertEqual(closed, ["chatlab-conversation"])
        self.assertFalse(self.manager.busy)
        saved = library.read()["branches"][MAIN_BRANCH]
        self.assertTrue(saved[-1]["content"].startswith("Hello"))
        self.assertEqual(self.state[self.turns._id][0]["content"], "Another conversation")

    def test_retry_event_data_reaches_the_original_handler(self):
        self.start()
        self.finish()
        self.call("poll")
        from gradio.helpers import EventData

        self.call("retry_message", event_data=EventData(None, {"index": 1, "value": "Hello world"}))
        self.finish()
        self.call("poll")
        self.assertEqual(len(self.state[self.turns._id]), 2)
        self.assertEqual(self.state[self.turns._id][-1]["content"], "Hello world")

    def test_next_token_resumes_a_background_response_one_token_at_a_time(self):
        self.release.set()
        values = {0: "hi", **dict(enumerate(SETTINGS, 2))}
        values[8] = 1  # max_new_tokens in the existing chat input order
        self.call("chat", values)
        self.finish()
        self.call("poll")
        self.assertEqual(self.state[self.turns._id][-1]["generated_tokens"], 1)
        self.call("next_token")
        self.finish()
        self.call("poll")
        reply = self.state[self.turns._id][-1]
        self.assertEqual(reply["generated_tokens"], 2)
        self.assertTrue(reply["token_step_paused"])
        self.assertFalse(self.manager.busy)
