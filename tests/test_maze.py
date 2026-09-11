import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from extensions.maze_experiments.maze import Maze, apply_call, call_text, generate, parse_call
from extensions.maze_experiments.runner import Episode, TERMINAL, fork_token_edit, from_payload, stream_episode
from model_runtime import ModelManager
from extension_api import ModelService
from extension_api import TokenInspector
from token_metrics import unscored_metric
from extensions.maze_experiments.page import board, build_page, export_run, views
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

    def encode_replacement(self, kept_ids, text, **kwargs):
        # This fixture encodes independent UTF-8 bytes; context-sensitive
        # behavior is exercised separately through the real runtime encoder.
        return list(text.encode())

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text, ids = next(self.replies)
        prefix = kwargs["forced_ids"]
        metrics = [{"token_id": t} for t in prefix + ids]
        yield SimpleNamespace(text=self.tokenizer.decode(prefix) + text, metrics=metrics, prompt_ids=[10, 20],
                              forced_prefix_tokens=len(prefix), reasoning_prefilled=self.reasoning_prefilled,
                              load_id=self.load_id, model_id=self.model_id)


class MazeTests(unittest.TestCase):
    def test_typed_edit_preserves_sentencepiece_boundary_and_literal_prefix(self):
        from test_streaming import sentencepiece_manager, SP_HELLO, SP_SPACE_WORLD, SP_WORLD
        for text, expected_id in [('world', SP_WORLD), (' world', SP_SPACE_WORLD)]:
            with self.subTest(replacement=text):
                manager = sentencepiece_manager()
                ep = Episode(MAZE, CONFIG)
                ep.model_id, ep.load_id = manager.model_id, manager.load_id
                ep.turns = [dict(metrics=[{'token_id': SP_HELLO}, {'token_id': SP_SPACE_WORLD}],
                                 forced_prefix_tokens=1, literal_prefill_tokens=1)]
                with ModelService(lambda: manager).open_session() as session:
                    with mock.patch.object(manager, 'encode_replacement', wraps=manager.encode_replacement) as encode:
                        edited = fork_token_edit(ep, 0, 1, text, session)
                    encode.assert_called_once_with([SP_HELLO], text, literal_prefill_tokens=1,
                                                   load_id=ep.load_id)
                    self.assertEqual(edited.pending_edit['forced_ids'], [SP_HELLO, expected_id])
                    self.assertEqual(session.decode(edited.pending_edit['forced_ids']), 'Hello' + text)
                    self.assertEqual(edited.token_edit['replacement_text'], text)

    def test_edit_ui_callbacks_select_regenerate_archive_and_reject_stale_token(self):
        manager = Manager([('abc', [97, 98, 99, 0]), ('yz', [121, 122, 0])])
        generate_reply = manager.generate

        def rich_reply(*args, **kwargs):
            for frame in generate_reply(*args, **kwargs):
                for position, metric in enumerate(frame.metrics):
                    metric.update(unscored_metric(position=position, token_id=metric['token_id'],
                                                  token_text=chr(metric['token_id']), fallback_text='',
                                                  segment='response').to_dict())
                    metric['top_candidates'] = [{'token_id': 120, 'text': 'x', 'probability': .1}]
                yield frame

        manager.generate = rich_reply
        inspector = TokenInspector()
        selections = inspector.selections()
        inspector.selections = lambda: selections
        session_id = selections.new_session()
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        list(stream_episode(ep, manager))
        original = copy.deepcopy(ep.payload())
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=inspector, models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}
                metrics = views(ep, False, selections, session_id)[7]
                selected = callbacks['select_token'].fn(ep, session_id, metrics, SimpleNamespace(index=1))
                self.assertEqual(selected[3], 'b')
                self.assertIn(("'x' · token 120", '120'), selected[4]['choices'])
                edit = callbacks['edit_token']
                frames = list(edit.fn(ep, False, session_id, metrics, selected[2], 'ignored', '120'))
                self.assertTrue(all(len(frame) == len(edit.outputs) for frame in frames))
                result = frames[-1][0]
                self.assertEqual(result.turns[0]['text'], 'axyz')
                self.assertEqual(result.phase, 'abandoned')
                self.assertEqual(json.loads((Path(directory) / f'{ep.run_id}.json').read_text()),
                                 json.loads(json.dumps(original)))
                self.assertTrue((Path(directory) / f'{result.run_id}.json').exists())
                self.assertEqual(ep.payload(), original)
                with self.assertRaisesRegex(gr.Error, 'current response'):
                    list(edit.fn(ep, False, session_id, metrics, selected[2], 'x', 'text'))
            finally:
                demo.close()

    def test_token_edit_rewinds_history_and_moves_and_preserves_original(self):
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        manager = Manager([(move, ids), (move, ids)])
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        before = copy.deepcopy(ep.payload())
        index = move.index('east')
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 1, index, "west", session)
        self.assertEqual(edited.position, (0, 1))
        self.assertEqual(len(edited.turns), 1)
        self.assertEqual(len(edited.events), 1)
        self.assertEqual(edited.tool_attempts, 1)
        self.assertEqual(edited.sampled_tokens, len(ids))
        self.assertNotEqual(edited.run_id, ep.run_id)
        self.assertEqual(edited.token_edit["parent_run_id"], ep.run_id)
        self.assertTrue(edited.manual_intervention)
        self.assertEqual(edited.pending_edit["forced_ids"], ids[:index] + list(b'west'))
        suffix = move[index + len('east'):]
        manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
        list(stream_episode(edited, manager, single_step=True))
        self.assertEqual(edited.phase, "paused")
        self.assertEqual(edited.position, MAZE.start)
        self.assertEqual(edited.events[-1]["direction"], "west")
        self.assertEqual(edited.messages[-2]["content"], move.replace('east', 'west'))
        self.assertEqual(manager.calls[-1][0], ep.messages[:4])
        self.assertEqual(manager.calls[-1][1]["literal_prefill_tokens"], 0)
        self.assertEqual(edited.sampled_tokens, len(ids) + len(suffix) + 1)
        self.assertEqual(ep.payload(), before)
        replay = from_payload(copy.deepcopy(edited.payload()))
        self.assertEqual(replay.token_edit, edited.token_edit)
        self.assertEqual(replay.position, MAZE.start)

    def test_token_edit_can_choose_exact_candidate_and_regenerate_stopped_response(self):
        manager = Manager([('abc', [97, 98, 99])])
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(ep, manager))
        ep.turns[0]["metrics"][1]["top_candidates"] = [{"token_id": 120, "text": "x"}]
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 0, 1, "ignored", session, candidate_id=120)
        self.assertEqual(edited.pending_edit["forced_ids"], [97, 120])
        manager.replies = iter([('yz', [121, 122, 0])])
        list(stream_episode(edited, manager))
        self.assertEqual(edited.turns[0]["text"], 'axyz')
        self.assertEqual(edited.phase, 'abandoned')
        self.assertFalse(edited.interrupted)
        self.assertEqual(edited.sampled_tokens, 3)

    def test_token_edit_restores_interruption_and_recovery_at_rewind_point(self):
        move = '\n' + call_text(MAZE.maze_id, "east")
        manager = Manager([(move, list(move.encode()) + [0])] * 2)
        ep = Episode(MAZE, CONFIG | {"per_turn_tokens": 200, "token_budget": 1000})
        list(stream_episode(ep, manager))
        with manager.open_session() as session:
            first = fork_token_edit(ep, 0, 2, '\n', session)
            second = fork_token_edit(ep, 1, 0, '\n', session)
        self.assertFalse(first.interrupted)
        self.assertTrue(first.pending_edit["interruption_here"])
        self.assertEqual(first.pending_edit["literal_prefill_tokens"], 2)
        self.assertTrue(second.interrupted)
        self.assertTrue(second.resumed)
        self.assertEqual(second.latency, ep.latency)
        self.assertEqual(second.intervention_turn, 0)
        manager.replies = iter([(move, list(move.encode()) + [0])])
        list(stream_episode(first, manager, single_step=True))
        self.assertTrue(first.interrupted)
        self.assertTrue(first.resumed)
        self.assertEqual(first.intervention_turn, 0)

    def test_token_edit_rejects_busy_replay_changed_model_and_invalid_selection(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('abc', [97, 98, 99, 0])])
        list(stream_episode(ep, manager))
        with manager.open_session() as session:
            for field, value, message in [('busy', True, 'Pause'), ('replay_only', True, 'read-only'),
                                          ('load_id', 'other', 'model changed')]:
                previous = getattr(ep, field)
                setattr(ep, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    fork_token_edit(ep, 0, 2, 'x', session)
                setattr(ep, field, previous)
            for turn, token in [(-1, 2), (1, 2), (0, -1), (0, 0), (0, 99)]:
                with self.assertRaises(ValueError):
                    fork_token_edit(ep, turn, token, 'x', session)
            with self.assertRaisesRegex(ValueError, 'replacement text'):
                fork_token_edit(ep, 0, 2, '', session)
            with self.assertRaisesRegex(ValueError, 'alternative'):
                fork_token_edit(ep, 0, 2, '', session, candidate_id=123)

    def test_edit_to_stop_token_finishes_without_executing_partial_action(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([('abc', [97, 98, 99, 0])])
        list(stream_episode(ep, manager))
        ep.turns[0]["metrics"][1]["top_candidates"] = [{"token_id": 0, "text": "EOS"}]
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 0, 1, '', session, candidate_id=0)
        manager.replies = iter([('', [])])  # Runtime returns after consuming a forced stop token.
        list(stream_episode(edited, manager))
        self.assertEqual(edited.phase, 'abandoned')
        self.assertEqual(edited.turns[0]["finish_reason"], 'stop')
        self.assertEqual(edited.tool_attempts, 0)

    def test_goal_modes_cover_prompts_errors_supplied_moves_arrival_and_replay(self):
        hint = "The destination is in the top row."
        for mode in ("coordinates", "hidden", "hint"):
            for supplied in (0, 1):
                with self.subTest(mode=mode, supplied=supplied):
                    ep = Episode(MAZE, CONFIG | dict(goal_mode=mode, goal_hint=hint,
                                                    supplied_moves=supplied, interruption_text=""))
                    maze_id = ep.model_state()["maze_id"]
                    replies = [call_text(maze_id, "south"), call_text("wrong", "east"),
                               '<tool_call>{}</tool_call>'] + [call_text(maze_id, "east")] * (2 - supplied)
                    manager = Manager([(reply, [8, 0]) for reply in replies])
                    with mock.patch('extensions.maze_experiments.runner.time.sleep'):
                        list(stream_episode(ep, manager))
                    self.assertEqual(ep.phase, "arrived")
                    self.assertEqual(ep.position, MAZE.goal)
                    self.assertEqual([e['error'] for e in ep.events if not e['accepted']],
                                     ['blocked_move', 'wrong_maze', 'invalid_tool_schema'])
                    # Check the actual messages passed to generation as well as the final reply.
                    for history in [ep.messages, *(call[0] for call in manager.calls)]:
                        for message in history:
                            if message['role'] not in ('user', 'tool'):
                                continue
                            content = message['content']
                            state = json.loads(content.split('\n', 1)[1] if message['role'] == 'user' else content)
                            self.assertEqual(state['grid'], list(MAZE.grid))
                            self.assertEqual(state['maze_id'], maze_id)
                            self.assertEqual('destination' in state, mode == 'coordinates')
                            self.assertEqual('goal_hint' in state, mode == 'hint')
                            self.assertNotIn('progress', state)
                            if mode == 'coordinates':
                                self.assertEqual(state['destination'], list(MAZE.goal))
                            elif mode == 'hint':
                                self.assertEqual(state['goal_hint'], hint)
                    self.assertTrue(json.loads(ep.messages[-1]['content'])['arrived'])
                    replay = from_payload(json.loads(json.dumps(ep.payload())))
                    self.assertEqual(replay.config['goal_mode'], mode)
                    self.assertEqual(replay.config['goal_hint'], hint)
                    self.assertEqual(replay.messages, ep.messages)
                    self.assertEqual(replay.position, MAZE.goal)
                    self.assertTrue(replay.replay_only)

    def test_concealed_state_does_not_encode_destination_in_identifier(self):
        other = Maze(MAZE.grid, MAZE.start, (2, 0))
        for mode in ('hidden', 'hint'):
            options = dict(goal_mode=mode, goal_hint='The destination is on an outer row.')
            self.assertEqual(MAZE.state(MAZE.start, **options), other.state(other.start, **options))
        self.assertNotEqual(MAZE.maze_id, other.maze_id)

    def test_only_the_displayed_response_animates_its_own_move(self):
        moves = [call_text(MAZE.maze_id, step) for step in ('east', 'north', 'east')]
        manager = Manager([(text, list(text.encode()) + [0]) for text in moves])
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'arrived')
        self.assertEqual([e['accepted'] for e in ep.events], [True, False, True])
        # Response 2 was rejected, so the character stayed put and nothing moves.
        self.assertNotIn('animateTransform', board(ep, 1, animate=True))
        self.assertIn('Character at row 0, column 1', board(ep, 1, animate=True))
        for index in (0, 2):
            self.assertIn('animateTransform', board(ep, index, animate=True))
            self.assertNotIn('animateTransform', board(ep, index))

    def test_replay_stepping_and_playback_walk_recorded_responses(self):
        move = call_text(MAZE.maze_id, 'east')
        ids = list(move.encode()) + [0]
        manager = Manager([(move, ids), (move, ids)])
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'arrived')
        for turn in ep.turns:
            for position, metric in enumerate(turn['metrics']):
                metric.update(unscored_metric(position=position, token_id=metric['token_id'],
                                              token_text=chr(metric['token_id']), fallback_text='',
                                              segment='response').to_dict())
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        inspector = TokenInspector()
        selections = inspector.selections()
        inspector.selections = lambda: selections
        session = selections.new_session()
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=inspector, models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}
                forward, back = callbacks['step_forward'].fn, callbacks['step_back'].fn
                selected = lambda frame: frame[8]['value']
                # Stepping reads the episode, so repeated clicks advance even
                # when the dropdown the browser sent has not caught up.
                self.assertEqual(replay.viewing, -1)
                self.assertEqual([selected(forward(replay, False, session)) for _ in range(3)], [0, 1, 1])
                self.assertEqual([selected(back(replay, False, session)) for _ in range(3)], [0, -1, -1])
                playback = callbacks['play_back']
                with mock.patch('extensions.maze_experiments.page.time.sleep') as sleep:
                    frames = list(playback.fn(replay, False, session, .4))
                self.assertEqual(sleep.call_args_list, [mock.call(.4)] * 2)
                self.assertEqual([selected(frame) for frame in frames], [-1, 0, 1])
                self.assertTrue(all(len(frame) == len(playback.outputs) for frame in frames))
                for frame, column in zip(frames, (0, 1, 2)):
                    self.assertIn(f'Character at row 0, column {column}', frame[0])
                with mock.patch('extensions.maze_experiments.page.time.sleep') as sleep:
                    self.assertEqual([selected(frame) for frame in playback.fn(replay, False, session, .4)], [1])
                sleep.assert_not_called()
                # Starting playback again supersedes the run already going, so
                # the older one cannot repaint a response the newer passed.
                replay.viewing = -1
                superseded = playback.fn(replay, False, session, .4)
                self.assertEqual(selected(next(superseded)), -1)
                replay.viewing = -1
                current = playback.fn(replay, False, session, .4)
                self.assertEqual(selected(next(current)), -1)
                with mock.patch('extensions.maze_experiments.page.time.sleep'):
                    self.assertEqual(list(superseded), [])
                    self.assertEqual([selected(frame) for frame in current], [0, 1])
                with mock.patch.object(gr, 'Info') as info:
                    self.assertIn('Replay', callbacks['stop_playback'].fn(replay))
                info.assert_called_once()
                replay.viewing = 9
                self.assertEqual(selected(back(replay, False, session)), 0)
                replay.busy = True
                for call in (lambda: forward(replay, False, session), lambda: back(replay, False, session),
                             lambda: list(playback.fn(replay, False, session, .4))):
                    with self.assertRaisesRegex(gr.Error, 'Pause'):
                        call()
            finally:
                demo.close()

    def test_goal_mode_validation_and_legacy_replay(self):
        for options in ({'goal_mode': 'unknown'}, {'goal_mode': 'hint'},
                        {'goal_mode': 'hint', 'goal_hint': '  '}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Episode(MAZE, CONFIG | options)
        old = Episode(MAZE, CONFIG).payload()
        old['config'].pop('goal_mode')
        old['config'].pop('goal_hint')
        replay = from_payload(json.loads(json.dumps(old)))
        self.assertEqual(replay.config['goal_mode'], 'coordinates')
        self.assertEqual(replay.model_state()['destination'], list(MAZE.goal))

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
