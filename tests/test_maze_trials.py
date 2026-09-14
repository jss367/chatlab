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
from extensions.maze_experiments.trials import FORMAT, control_values, prepare_trial, read_trials
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
        first, _ = prepare_trial(data, 'clean', original)
        second, _ = prepare_trial(data, 'clean', original)
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
        episode, item = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual(episode.messages[0]['content'], SYSTEM)
        self.assertEqual(control_values(item)[-2:], [SYSTEM, default_instruction('coordinates')])
        self.payload['trials'][0]['config'] |= dict(system_prompt='Be terse.', instruction='Reach it.')
        episode, item = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual(episode.messages[0]['content'], 'Be terse.')
        self.assertTrue(episode.messages[1]['content'].startswith('Reach it.'))
        self.assertEqual(control_values(item)[-2:], ['Be terse.', 'Reach it.'])
        self.payload['trials'][0]['config']['instruction'] = 5
        with self.assertRaisesRegex(ValueError, 'instruction must be text'):
            self.read()

    def test_large_collections_load_and_oversized_ones_are_refused(self):
        trial = self.payload['trials'][0]
        self.payload['trials'] = [trial | dict(id=f'trial-{n:04d}', label=f'Maze {n} \u00b7 Clean') for n in range(2000)]
        data = self.read()
        self.assertEqual(len(data['trials']), 2000)
        self.assertEqual(prepare_trial(data, 'trial-1999', Episode(MAZE, CONFIG))[1]['label'], 'Maze 1999 \u00b7 Clean')
        self.payload['trials'] = [trial | dict(id=f'trial-{n:04d}') for n in range(2001)]
        with self.assertRaisesRegex(ValueError, '1\u20132000 trials'):
            self.read()

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
