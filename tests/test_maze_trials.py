import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from extension_api import TokenInspector
from extensions.maze_experiments.page import build_page
from extensions.maze_experiments.runner import Episode, from_payload
from extensions.maze_experiments.maze import SYSTEM, default_instruction
from extensions.maze_experiments.trials import FORMAT, prepare_trial, read_trials
from test_maze import CONFIG, MAZE


class TrialFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'trials.json'
        self.payload = dict(format=FORMAT, title='Example trials', trials=[dict(
            id='clean', label='Clean trial', maze=MAZE.to_dict(), openness=.7,
            config=CONFIG | dict(interruption_text='', goal_mode='coordinates', goal_hint=''))])

    def read(self):
        self.path.write_text(json.dumps(self.payload))
        return read_trials(self.path)

    def test_fixed_map_fresh_runs_and_replay_provenance(self):
        data = self.read()
        original = Episode(MAZE, CONFIG)
        first = prepare_trial(data, 'clean', original)
        second = prepare_trial(data, 'clean', original)
        self.assertEqual(first.maze, MAZE)
        self.assertEqual(first.messages, second.messages)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertFalse(first.replay_only)
        replay = from_payload(json.loads(json.dumps(first.payload())))
        self.assertTrue(replay.replay_only)
        self.assertEqual(replay.config['trial']['file_sha256'], data['file_sha256'])
        original.busy = True
        with self.assertRaisesRegex(ValueError, 'Stop or pause'):
            prepare_trial(data, 'clean', original)

    def test_bad_limits_duplicate_ids_and_replays_are_rejected(self):
        good = copy.deepcopy(self.payload)
        for value in (-1, 3.5, True):
            self.payload = copy.deepcopy(good)
            self.payload['trials'][0]['config']['token_budget'] = value
            with self.assertRaises(ValueError):
                self.read()
        self.payload = copy.deepcopy(good)
        self.payload['trials'] *= 2
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.read()
        self.payload = Episode(MAZE, CONFIG).payload()
        with self.assertRaisesRegex(ValueError, 'not a saved replay'):
            self.read()

    def test_a_trial_pins_its_prompt_or_runs_the_stock_wording(self):
        episode = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual(episode.messages[0]['content'], SYSTEM)
        self.assertEqual(episode.config['instruction'], default_instruction('coordinates'))
        self.payload['trials'][0]['config'] |= dict(system_prompt='Be terse.', instruction='Reach it.')
        episode = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual(episode.messages[0]['content'], 'Be terse.')
        self.assertTrue(episode.messages[1]['content'].startswith('Reach it.'))
        self.payload['trials'][0]['config']['instruction'] = 5
        with self.assertRaisesRegex(ValueError, 'instruction must be text'):
            self.read()

    def test_large_collections_load_and_oversized_ones_are_refused(self):
        trial = self.payload['trials'][0]
        self.payload['trials'] = [trial | dict(id=f'trial-{n:04d}', label=f'Maze {n} \u00b7 Clean') for n in range(2000)]
        data = self.read()
        self.assertEqual(len(data['trials']), 2000)
        self.assertEqual(prepare_trial(data, 'trial-1999', Episode(MAZE, CONFIG)).config['trial']['label'],
                         'Maze 1999 \u00b7 Clean')
        self.payload['trials'] = [trial | dict(id=f'trial-{n:04d}') for n in range(2001)]
        with self.assertRaisesRegex(ValueError, '1\u20132000 trials'):
            self.read()
        # Refused on its size alone, before anything reads or parses it.
        self.path.write_text('x' * 8_000_001)
        with self.assertRaisesRegex(ValueError, 'smaller than 8 MB'):
            read_trials(self.path)

    def test_ui_loads_exact_trial_updates_controls_and_leaves_old_run_intact(self):
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
        loaded = callbacks['load_trial_file'].fn(str(self.path))
        self.assertEqual(loaded[1]['value'], 'clean')
        old = Episode(MAZE, CONFIG)
        before = old.payload()
        callback = callbacks['load_trial']
        frame = callback.fn(data, 'clean', old, False, 'test-session')
        self.assertEqual(len(frame), len(callback.outputs))
        self.assertEqual(frame[0].config['trial']['id'], 'clean')
        self.assertEqual(old.payload(), before)
        self.assertEqual(frame[-4], 'None')
        self.assertEqual(frame[-2:], (None, None))
        self.assertIn('Clean trial', frame[-3])

    def test_the_trial_note_stops_naming_a_trial_the_episode_no_longer_is(self):
        # Every other way of replacing the episode has to say so, or the pane
        # keeps crediting a trial for a run that came from somewhere else.
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
        loaded = callbacks['load_trial'].fn(data, 'clean', Episode(MAZE, CONFIG), False, 'test-session')
        self.assertIn('Clean trial', loaded[-3])
        prepare = callbacks['prepare_episode']
        values = [component.value for component in prepare.inputs[4:]]
        fresh = prepare.fn(loaded[0], False, 'test-session', data, *values)
        self.assertEqual(len(fresh), len(prepare.outputs))
        self.assertIsNone(fresh[0].config.get('trial'))
        self.assertNotIn('Clean trial', fresh[-2])
        self.assertIn('Example trials', fresh[-2])

        path = Path(self.directory.name) / 'run.json'
        path.write_text(json.dumps(loaded[0].payload()))
        load = callbacks['load']
        replayed = load.fn(str(path), Episode(MAZE, CONFIG), False, 'test-session', None)
        self.assertEqual(len(replayed), len(load.outputs))
        self.assertIn('Replaying: Clean trial', replayed[-1])
