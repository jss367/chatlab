import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from extensions.maze_experiments.maze import Maze, apply_call, call_text, generate, parse_call
from extensions.maze_experiments.runner import Episode, TERMINAL, from_payload, stream_episode
from model_runtime import ModelManager
from extension_api import ModelService
from extension_api import TokenInspector
from extensions.maze_experiments.page import export_run, views
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
        original.created_at = 100.
        with mock.patch('extensions.maze_experiments.runner.time.time', return_value=200.):
            duplicate = copy.deepcopy(original)
        duplicate.request_stop()
        self.assertEqual(original.phase, "ready")
        self.assertNotEqual(original.run_id, duplicate.run_id)
        self.assertIsNot(original.lock, duplicate.lock)
        self.assertIsNot(original.messages, duplicate.messages)
        self.assertEqual(original.created_at, 100.)
        self.assertEqual(duplicate.created_at, 200.)
        original.phase = 'paused'
        resumed = copy.deepcopy(original)
        self.assertEqual(resumed.created_at, original.created_at)
        self.assertEqual(resumed.run_id, original.run_id)

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

    def test_supplied_code_fences_are_rejected_before_they_can_hide_generated_calls(self):
        for fence in ('```', '~~~'):
            with self.subTest(fence=fence):
                # The normal parser must still ignore model-generated fenced examples.
                self.assertEqual(parse_call(fence + '\n' + call_text(MAZE.maze_id, 'east')), (None, None))
                ep = Episode(MAZE, CONFIG | {'interruption_text': fence + '\nExample', 'prefix_tokens': 0})
                manager = Manager([])
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, 'error')
                self.assertIn('code fences', ep.detail)
                self.assertEqual(manager.calls, [])
                self.assertFalse(ep.interrupted)
                self.assertFalse(manager.busy)

    def test_ordinary_interruption_prose_with_inline_backticks_remains_allowed(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': 'Discuss `inline text` and ~one tilde~.', 'prefix_tokens': 0})
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.phase, 'paused')
        self.assertEqual(ep.position, (0, 1))
        self.assertTrue(ep.interrupted)
        self.assertTrue(ep.resumed)

    def test_interruption_rejects_finished_runs_and_replays_without_changing_provenance(self):
        for phase, replay in [(phase, False) for phase in TERMINAL] + [('ready', True), ('paused', True), ('running', True)]:
            with self.subTest(phase=phase, replay=replay):
                ep = Episode(MAZE, CONFIG)
                ep.phase, ep.replay_only = phase, replay
                before = copy.deepcopy(ep.payload())
                with self.assertRaisesRegex(ValueError, 'Start a new episode'):
                    ep.request_interruption()
                self.assertEqual(ep.payload(), before)
                self.assertFalse(ep.interrupt_next)
        for phase in ('ready', 'paused', 'running'):
            ep = Episode(MAZE, CONFIG)
            ep.phase = phase
            ep.request_interruption()
            self.assertTrue(ep.interrupt_next)
            self.assertTrue(ep.manual_intervention)

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

    def test_failed_autosave_pauses_or_preserves_outcome_and_yields_cleanup(self):
        for supplied, reply, phase in ((0, call_text(MAZE.maze_id, 'east'), 'paused'),
                                       (1, call_text(MAZE.maze_id, 'east'), 'arrived'),
                                       (0, 'I am done.', 'abandoned')):
            with self.subTest(phase=phase):
                ep = Episode(MAZE, CONFIG | {'supplied_moves': supplied})
                manager = Manager([('\n' + reply, [8, 0])])
                with mock.patch.object(ep, 'save', side_effect=PermissionError('Archive not writable')) as save:
                    frames = [(frame.phase, frame.busy, frame.detail) for frame in stream_episode(ep, manager, save_dir=Path('/unused'))]
                save.assert_called_once()
                self.assertEqual(frames[-1][:2], (phase, False))
                self.assertIn('Autosave failed:', frames[-1][2])
                self.assertFalse(manager.busy)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(ep.resumed, phase != 'abandoned')
                self.assertEqual(ep.latency, 2 if phase != 'abandoned' else None)

    def test_save_failure_during_final_cleanup_still_yields_a_final_frame(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        with mock.patch.object(ep, 'save', side_effect=[None, OSError('Disk full')]) as save:
            frames = [(frame.phase, frame.busy, frame.detail) for frame in stream_episode(ep, manager, single_step=True, save_dir=Path('/unused'))]
        self.assertEqual(save.call_count, 2)
        self.assertEqual(frames[-1][:2], ('paused', False))
        self.assertIn('Disk full', frames[-1][2])
        self.assertFalse(manager.busy)
        self.assertEqual(ep.sampled_tokens, 2)

    def test_export_download_survives_unwritable_archive(self):
        ep = Episode(MAZE, CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'not-a-directory'
            archive.write_text('blocked')
            with mock.patch('extensions.maze_experiments.page.gr.Warning') as warning:
                path = Path(export_run(ep, archive))
            warning.assert_called_once()
        try:
            self.assertEqual(json.loads(path.read_text())['run_id'], ep.run_id)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        finally:
            path.unlink()
            path.parent.rmdir()

    def test_exporting_old_replay_does_not_overwrite_newer_completed_archive(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        move = call_text(MAZE.maze_id, 'east')
        manager = Manager([(move, [8, 0]), (move, [8, 0])])
        with tempfile.TemporaryDirectory() as directory:
            list(stream_episode(ep, manager, single_step=True, save_dir=Path(directory)))
            old_snapshot = copy.deepcopy(ep.payload())
            self.assertEqual(old_snapshot['phase'], 'paused')
            list(stream_episode(ep, manager, save_dir=Path(directory)))
            self.assertEqual(ep.phase, 'arrived')
            archive = Path(directory) / f'{ep.run_id}.json'
            completed = archive.read_bytes()
            path = Path(export_run(from_payload(old_snapshot), Path(directory)))
            try:
                self.assertEqual(archive.read_bytes(), completed)
                downloaded = json.loads(path.read_text())
                self.assertEqual(downloaded['phase'], 'paused')
                self.assertEqual(downloaded['run_id'], ep.run_id)
                self.assertNotEqual(path, archive)
            finally:
                path.unlink()
                path.parent.rmdir()

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

    def test_stop_on_completed_frame_does_not_start_another_response(self):
        for single_step in (False, True):
            with self.subTest(single_step=single_step):
                ep = Episode(MAZE, CONFIG)
                manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                for frame in stream_episode(ep, manager, single_step=single_step):
                    if frame.turns and frame.turns[-1]['finish_reason'] == 'stop':
                        ep.request_stop()
                self.assertEqual(ep.phase, 'stopped')
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(len(ep.turns), 1)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertFalse(manager.busy)

    def test_stop_during_response_gap_does_not_generate_more_tokens(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        with mock.patch('extensions.maze_experiments.runner.time.sleep', side_effect=lambda _: ep.request_stop()):
            list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(len(ep.turns), 1)
        self.assertEqual(ep.sampled_tokens, 2)

    def test_pause_during_response_gap_does_not_start_another_response(self):
        for also_stop in (False, True):
            with self.subTest(also_stop=also_stop):
                ep = Episode(MAZE, CONFIG)
                manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                def request_during_gap(_):
                    ep.request_pause()
                    if also_stop:
                        ep.request_stop()
                with mock.patch('extensions.maze_experiments.runner.time.sleep', side_effect=request_during_gap):
                    list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, 'stopped' if also_stop else 'paused')
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(len(ep.turns), 1)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertFalse(manager.busy)

    def test_pause_during_active_response_still_completes_its_move(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        stream = stream_episode(ep, manager)
        next(stream)
        ep.request_pause()
        list(stream)
        self.assertEqual(ep.phase, 'paused')
        self.assertEqual(ep.position, (0, 1))
        self.assertEqual(ep.turns[0]['finish_reason'], 'stop')
        self.assertEqual(ep.sampled_tokens, 2)

    def test_stop_on_opening_frame_never_invokes_generation(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([])
        stream = stream_episode(ep, manager)
        next(stream)
        ep.request_stop()
        list(stream)
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(manager.calls, [])
        self.assertEqual(ep.sampled_tokens, 0)
        self.assertEqual(ep.turns[0]['finish_reason'], 'user_stopped')
        self.assertFalse(manager.busy)
        self.assertFalse(ep.interrupted)
        self.assertIsNone(ep.intervention_turn)
        self.assertEqual(ep.turns[0]['forced_prefix_tokens'], 0)
        self.assertEqual(ep.turns[0]['prefix_ids'], [])
        self.assertEqual(ep.turns[0]['prefix_text'], '')
        self.assertTrue(ep.turns[0]['planned_prefix_ids'])
        selections = TokenInspector().selections()
        rendered = views(ep, False, selections, selections.new_session())
        self.assertIn('insertion has not been confirmed', rendered[4])

    def test_closing_completed_frame_preserves_terminal_archive(self):
        for supplied, text, ids, phase in ((1, call_text(MAZE.maze_id, 'east'), [8, 0], 'arrived'),
                                           (0, 'I am done.', [8, 0], 'abandoned'),
                                           (0, 'Unfinished', [8], 'budget')):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                ep = Episode(MAZE, CONFIG | {'supplied_moves': supplied, 'interruption_text': ''})
                manager = Manager([(text, ids)])
                stream = stream_episode(ep, manager, save_dir=Path(directory))
                for frame in stream:
                    if frame.phase == phase:
                        before = copy.deepcopy(frame.payload())
                        stream.close()
                        break
                self.assertEqual(ep.payload(), before)
                saved = json.loads((Path(directory) / f'{ep.run_id}.json').read_text())
                self.assertEqual(saved, json.loads(json.dumps(before)))
                self.assertFalse(manager.busy)

    def test_closing_active_stream_retains_partial_tokens_without_moving(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        manager = Manager([(call_text(MAZE.maze_id, 'east'), [8, 0])])
        stream = stream_episode(ep, manager)
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(ep.sampled_tokens, 2)
        self.assertEqual(ep.position, MAZE.start)
        self.assertFalse(manager.busy)

    def test_failure_before_prefix_update_does_not_record_insertion(self):
        ep = Episode(MAZE, CONFIG)
        ep.request_interruption()
        manager = Manager([])
        with mock.patch.object(manager, 'generate', side_effect=RuntimeError('Prefill failed')):
            list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'error')
        self.assertFalse(ep.interrupted)
        self.assertIsNone(ep.intervention_turn)
        self.assertEqual(ep.intervention_tokens, 0)
        self.assertEqual(ep.intervention_attempts, 0)
        self.assertTrue(ep.manual_intervention)  # A request occurred, but insertion did not.
        self.assertEqual(ep.turns[0]['prefix_ids'], [])
        self.assertFalse(manager.busy)

    def test_prefix_only_update_records_consumption_even_if_sampling_fails(self):
        ep = Episode(MAZE, CONFIG | {'per_turn_tokens': 2048, 'token_budget': 8192})
        manager = Manager([])
        def generate(messages, **options):
            self.assertEqual(options['max_new_tokens'], 1024)
            prefix = options['forced_ids']
            yield SimpleNamespace(text=manager.tokenizer.decode(prefix), metrics=[{'token_id': t} for t in prefix],
                                  prompt_ids=[10], forced_prefix_tokens=len(prefix), reasoning_prefilled=False,
                                  load_id=manager.load_id, model_id=manager.model_id)
            raise RuntimeError('Sampling failed')
        manager.generate = generate
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'error')
        self.assertTrue(ep.interrupted)
        self.assertEqual(ep.intervention_turn, 0)
        self.assertEqual(ep.intervention_tokens, 0)
        self.assertEqual(ep.intervention_attempts, 0)
        self.assertEqual(ep.sampled_tokens, 0)
        self.assertIsNone(ep.resumed)
        self.assertEqual(ep.turns[0]['prefix_ids'], [68, 105])
        self.assertEqual(ep.turns[0]['prefix_text'], 'Di')
        self.assertFalse(manager.busy)

    def test_stop_at_startup_lock_handoff_is_not_cleared(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([])
        class StopOnFirstUnlock:
            def __init__(self):
                self.lock = threading.Lock()
                self.released = False
            def __enter__(self):
                self.lock.acquire()
            def __exit__(self, *exc):
                self.lock.release()
                if not self.released:
                    self.released = True
                    ep.request_stop()
        ep.lock = StopOnFirstUnlock()
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(manager.calls, [])
        self.assertEqual(ep.turns, [])
        self.assertFalse(manager.busy)

    def test_idle_stops_persist_ready_and_paused_runs(self):
        for paused in (False, True):
            with self.subTest(paused=paused), tempfile.TemporaryDirectory() as directory:
                ep = Episode(MAZE, CONFIG)
                if paused:
                    manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                    list(stream_episode(ep, manager, single_step=True, save_dir=Path(directory)))
                before = (ep.sampled_tokens, ep.tool_attempts, ep.resumed, ep.latency)
                ep.request_stop(Path(directory))
                saved = json.loads((Path(directory) / f'{ep.run_id}.json').read_text())
                self.assertEqual(saved['phase'], 'stopped')
                self.assertEqual((ep.sampled_tokens, ep.tool_attempts, ep.resumed, ep.latency), before)

    def test_idle_stop_does_not_rewrite_replays_or_finished_runs(self):
        ep = Episode(MAZE, CONFIG)
        ep.phase = 'paused'
        with tempfile.TemporaryDirectory() as directory:
            path = ep.save(Path(directory))
            original = path.read_bytes()
            replay = from_payload(json.loads(original))
            replay.request_stop(Path(directory))
            self.assertEqual(replay.phase, 'paused')
            self.assertFalse(replay.stop_requested)
            self.assertEqual(path.read_bytes(), original)
        ep.phase = 'arrived'
        with mock.patch.object(ep, 'save') as save:
            ep.request_stop(Path('/unused'))
        save.assert_not_called()
        self.assertEqual(ep.phase, 'arrived')

    def test_idle_stop_storage_failure_retains_stopped_state_and_export_guidance(self):
        ep = Episode(MAZE, CONFIG)
        with mock.patch.object(ep, 'save', side_effect=PermissionError('Archive not writable')) as save:
            ep.request_stop(Path('/unused'))
            ep.request_stop(Path('/unused'))
        save.assert_called_once()
        self.assertEqual(ep.phase, 'stopped')
        self.assertIn('Autosave failed:', ep.detail)
        self.assertIn('Export run JSON', ep.detail)

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
