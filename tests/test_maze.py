import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from extensions.maze_experiments.maze import Maze, apply_call, call_text, generate, parse_call
from extensions.maze_experiments.runner import Episode, from_payload, stream_episode
from model_runtime import ModelManager
from extension_api import ModelService
from extension_api import TokenInspector
from extensions.maze_experiments.page import views
import gradio as gr

CONFIG = dict(supplied_moves=0, interrupt_after=0, interruption_text="Distracted", prefix_tokens=2,
              temperature=.7, sampling_seed=99, per_turn_tokens=100, token_budget=300, attempt_budget=10)
MAZE = Maze(("...", "##.", "..."), (0, 0), (0, 2))


class Manager:
    loaded = True
    model_id = "test/model"
    load_id = "test-load"
    reasoning_prefilled = False
    tokenizer = SimpleNamespace(encode=lambda s, **kw: list(s.encode()), decode=lambda ids, **kw: bytes(ids).decode())

    def __init__(self, replies):
        self.replies = iter(replies)
        self.busy = False
        self.calls = []

    def open_session(self):
        return ModelService(lambda: self).open_session()

    def reserve_generation(self):
        if self.busy:
            return False
        self.busy = True
        return True

    def release_generation(self):
        self.busy = False

    def _stop_token_ids(self):
        return {0}

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text, ids = next(self.replies)
        prefix = kwargs["forced_ids"]
        metrics = [{"token_id": t} for t in prefix + ids]
        yield SimpleNamespace(text=self.tokenizer.decode(prefix) + text, metrics=metrics, prompt_ids=[10, 20],
                              forced_prefix_tokens=len(prefix), reasoning_prefilled=self.reasoning_prefilled,
                              load_id=self.load_id, model_id=self.model_id)


class MazeTests(unittest.TestCase):
    def test_new_response_reset_and_replay_clear_selection_but_streaming_preserves_it(self):
        selections = TokenInspector().selections()
        session = selections.new_session()
        ep = Episode(MAZE, CONFIG)
        metric = dict(token_id=1, display_text='a', text='a', scored=False)
        turn = dict(metrics=[metric], text='a', forced_prefix_tokens=0, finish_reason='stop')
        ep.turns = [turn]
        first = views(ep, False, selections, session)
        self.assertEqual(first[-2:], ('Select a model-generated token above.', []))
        ep.turns[0]['metrics'].append(metric)
        streamed = views(ep, False, selections, session)
        self.assertEqual(streamed[-2:], (gr.skip(), gr.skip()))
        self.assertEqual(first[7][0], streamed[7][0])
        ep.turns.append(copy.deepcopy(turn))
        second = views(ep, False, selections, session)
        replay = views(ep, False, selections, session, index=0)
        reset = views(Episode(MAZE, CONFIG), False, selections, session)
        for frame in (second, replay, reset):
            self.assertEqual(frame[-2:], ('Select a model-generated token above.', []))
        self.assertEqual(len({frame[7][0] for frame in (first, second, replay, reset)}), 4)

    def test_browser_sessions_have_independent_episode_state(self):
        original = Episode(MAZE, CONFIG)
        duplicate = copy.deepcopy(original)
        duplicate.request_stop()
        self.assertEqual(original.phase, "ready")
        self.assertNotEqual(original.run_id, duplicate.run_id)
        self.assertIsNot(original.lock, duplicate.lock)
        self.assertIsNot(original.messages, duplicate.messages)

    def test_seed_and_distance(self):
        a, b = generate(), generate()
        self.assertEqual(a, b)
        self.assertEqual(len(a.route()), 11)
        for p, q in zip(a.route(), a.route()[1:]):
            self.assertIn(q, a.neighbors(p).values())

    def test_no_actions_from_quotes_or_fake_state(self):
        text = call_text(MAZE.maze_id, "east")
        for value in ("I moved east.", "```\n" + text + "\n```", "\n".join("> " + line for line in text.splitlines())):
            self.assertEqual(parse_call(value), (None, None))
        args, error = parse_call(text)
        self.assertIsNone(error)
        self.assertEqual(apply_call(MAZE, MAZE.start, args)["after"], [0, 1])
        self.assertFalse(apply_call(MAZE, MAZE.start, {"maze_id": MAZE.maze_id, "direction": "south"})["accepted"])
        self.assertFalse(apply_call(MAZE, MAZE.start, {"maze_id": "fake", "direction": "east"})["accepted"])

    def test_invalid_argument_types_are_rejected(self):
        for value in ([], {}, None, 7):
            text = '<tool_call>\n' + json.dumps({"name": "move", "arguments": {"maze_id": MAZE.maze_id, "direction": value}}) + '\n</tool_call>'
            self.assertEqual(parse_call(text)[1], "invalid_arguments")

    def test_supplied_reasoning_markers_are_rejected_before_generation(self):
        for marker in ('<think>', '</think>'):
            for prefilled in (False, True):
                with self.subTest(marker=marker, reasoning_prefilled=prefilled):
                    ep = Episode(MAZE, CONFIG | {'interruption_text': marker, 'prefix_tokens': 0})
                    manager = Manager([])
                    manager.reasoning_prefilled = prefilled
                    list(stream_episode(ep, manager))
                    self.assertEqual(ep.phase, 'error')
                    self.assertIn('reasoning delimiters', ep.detail)
                    self.assertEqual(manager.calls, [])
                    self.assertFalse(manager.busy)
                    self.assertFalse(ep.interrupted)
                    self.assertIsNone(ep.resumed)

    def test_return_then_arrival_excludes_supplied_tokens(self):
        ep = Episode(MAZE, CONFIG)
        move = '\n' + call_text(MAZE.maze_id, "east")
        manager = Manager([(move, [8, 0]), (move, [8, 0])])
        with tempfile.TemporaryDirectory() as d:
            list(stream_episode(ep, manager, save_dir=Path(d)))
            replay = from_payload(json.loads(ep.save(Path(d)).read_text()))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.latency, 2)
        self.assertEqual(ep.sampled_tokens, 4)
        self.assertTrue(ep.first_move_progress)
        self.assertFalse(manager.busy)
        self.assertEqual(manager.calls[0][1]["forced_ids"], [68, 105])
        self.assertTrue(manager.calls[0][1]["tools"])
        self.assertEqual(replay.position, MAZE.goal)
        self.assertTrue(replay.replay_only)
        bad = ep.payload()
        bad["events"][0]["after"] = [2, 2]
        with self.assertRaises(ValueError):
            from_payload(bad)

    def test_abandonment_does_not_prompt_again(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([("let's discuss bicycles", [8, 0])])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "abandoned")
        self.assertFalse(ep.resumed)
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(ep.position, MAZE.start)

    def test_template_reasoning_is_restored_in_next_turn_history(self):
        for prefilled in (False, True):
            with self.subTest(reasoning_prefilled=prefilled):
                ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
                thought = "Consider this move:\n" + call_text(MAZE.maze_id, "south")
                suffix = thought + "\n</think>\n" + call_text(MAZE.maze_id, "east")
                raw = suffix if prefilled else "<think>" + suffix
                manager = Manager([(raw, [8, 0]), (raw, [8, 0])])
                manager.reasoning_prefilled = prefilled
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, "arrived")
                self.assertEqual(ep.tool_attempts, 2)
                self.assertEqual(manager.calls[1][0][-2], {
                    "role": "assistant", "content": "<think>" + suffix,
                })
                # Replay keeps actual emitted text; only templated history is reconstructed.
                self.assertEqual(ep.turns[0]["text"], raw)
                self.assertEqual(ep.payload()["messages"][-2]["content"], "<think>" + suffix)

    def test_unfinished_and_thought_calls_do_not_move(self):
        for text, ids, phase in ((call_text(MAZE.maze_id, "east"), [8], "budget"),
                                 ("<think>\n" + call_text(MAZE.maze_id, "east") + "\n</think>", [8, 0], "abandoned")):
            ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
            list(stream_episode(ep, Manager([(text, ids)])))
            self.assertEqual(ep.phase, phase)
            self.assertEqual(ep.position, MAZE.start)

    def test_step_then_stop(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([(call_text(MAZE.maze_id, "east"), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.phase, "paused")
        self.assertFalse(manager.busy)
        ep.request_stop()
        self.assertEqual(ep.phase, "stopped")

    def test_resuming_with_another_model_preserves_original_provenance(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([(call_text(MAZE.maze_id, "east"), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        manager.model_id, manager.load_id = "different/model", "different-load"
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "error")
        self.assertEqual(ep.model_id, "test/model")
        self.assertEqual(len(manager.calls), 1)
        self.assertFalse(manager.busy)

    def test_stop_during_stream_does_not_execute_partial_action(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, "east"), [8, 0])])
        gen = stream_episode(ep, manager)
        next(gen)
        next(gen)
        ep.request_stop()
        list(gen)
        self.assertEqual(ep.position, MAZE.start)
        self.assertEqual(ep.phase, "stopped")
        self.assertIsNone(ep.resumed)
        self.assertFalse(manager.busy)

    def test_export_uses_a_private_temporary_file(self):
        ep = Episode(MAZE, CONFIG)
        path = ep.export()
        try:
            self.assertTrue(path.is_relative_to(Path(tempfile.gettempdir())))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["run_id"], ep.run_id)
        finally:
            path.unlink()
            path.parent.rmdir()

    def test_native_template_gets_tools(self):
        calls = []
        def template(messages, **kwargs):
            calls.append(kwargs)
            return [1, 2] if kwargs["tokenize"] else "tool prompt"
        manager = ModelManager()
        manager.tokenizer = SimpleNamespace(chat_template="native", apply_chat_template=template)
        ids, reasoning = manager._prompt_token_ids([], tools=[{"name": "move"}])
        self.assertEqual(ids, [1, 2])
        self.assertFalse(reasoning)
        self.assertEqual([c["tools"] for c in calls], [[{"name": "move"}]] * 2)


if __name__ == "__main__":
    unittest.main()
