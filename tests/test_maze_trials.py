import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr

from chatlab.extension_api import TokenInspector
from chatlab.extensions.maze_experiments import page
from chatlab.extensions.maze_experiments.page import build_page, trial_note_text
from chatlab.extensions.maze_experiments.runner import Episode, from_payload
from chatlab.extensions.maze_experiments.maze import SYSTEM, default_instruction
from chatlab.extensions.maze_experiments.trials import FORMAT, prepare_trial, read_trials
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
        with self.assertRaisesRegex(ValueError, 'IDs must be unique'):
            self.read()
        # Distinct IDs are no help in a picker that shows the label.
        self.payload = copy.deepcopy(good)
        self.payload['trials'] = [self.payload['trials'][0] | dict(id=f'trial-{n}') for n in range(2)]
        with self.assertRaisesRegex(ValueError, 'labels must be unique'):
            self.read()
        # The seed is a pinned input under a checksum, so a trial cannot leave
        # it to be defaulted or coerced the way a saved run's may be.
        for seed in (7.9, True, '7', None):
            self.payload = copy.deepcopy(good)
            maze = dict(self.payload['trials'][0]['maze'])
            maze.pop('seed') if seed is None else maze.update(seed=seed)
            self.payload['trials'][0]['maze'] = maze
            with self.assertRaisesRegex(ValueError, 'integer seed'):
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

    def test_a_trial_pins_its_recovery_window_or_runs_the_pilot_one(self):
        episode = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual((episode.config['recovery_tokens'], episode.config['recovery_attempts']), (1024, 4))
        self.payload['trials'][0]['config'] |= dict(recovery_tokens=256, recovery_attempts=2)
        episode = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertEqual((episode.config['recovery_tokens'], episode.config['recovery_attempts']), (256, 2))
        for bad in (0, -1, 2.5, True, '256'):
            self.payload['trials'][0]['config']['recovery_tokens'] = bad
            with self.assertRaisesRegex(ValueError, 'recovery_tokens must be an integer'):
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

    def test_a_name_from_the_file_cannot_write_the_note_it_appears_in(self):
        # The note is Markdown, and the file may have been written anywhere.
        self.payload['title'] = '**Bold** [link](http://example.test)'
        self.payload['trials'][0]['label'] = '**Closed**, from elsewhere.'
        data = self.read()
        episode = prepare_trial(data, 'clean', Episode(MAZE, CONFIG))
        note = trial_note_text(episode, data)
        self.assertNotIn('**Closed**', note)
        self.assertNotIn('[link](http://example.test)', note)
        self.assertIn(r'\*\*Closed\*\*', note)
        # The wording the workbench writes itself still renders as Markdown.
        self.assertTrue(note.startswith('**Running:'))

    def test_a_file_that_will_not_load_replaces_nothing_and_says_so(self):
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        upload = {fn.fn.__name__: fn for fn in demo.fns.values()}['load_trial_file']
        bad = Path(self.directory.name) / 'bad.json'
        bad.write_text('{"format": "chatlab-maze-trials-1", "title": "B", "trials": []}')
        with self.assertRaisesRegex(gr.Error, 'Still loaded: Example trials'):
            upload.fn(str(bad), Episode(MAZE, CONFIG), data)

    def test_clearing_the_widget_empties_the_picker_with_it(self):
        # Clearing is its own event, so registering the upload alone would
        # leave the picker offering a collection the pane no longer names.
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        handlers = [fn for fn in demo.fns.values() if getattr(fn.fn, '__name__', '') == 'load_trial_file']
        self.assertEqual({event for fn in handlers for _target, event in fn.targets}, {'upload', 'clear'})
        cleared = handlers[0].fn(None, Episode(MAZE, CONFIG), data)
        self.assertIsNone(cleared[0])
        self.assertEqual(cleared[1]['choices'], [])
        self.assertNotIn('Example trials', cleared[2])

    def test_uploading_a_collection_queues_with_the_rest_of_the_view(self):
        # Off the view's queue, a Load trial click could be served between the
        # upload arriving and the picker it fills, and prepare a trial from the
        # collection on its way out.
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        queues = {fn.fn.__name__: fn.concurrency_id for fn in demo.fns.values() if fn.fn is not None}
        self.assertEqual(queues['load_trial_file'], queues['load_trial'])

    def test_the_note_calls_a_replay_a_replay_and_a_fork_of_one_live(self):
        # A fork of an uploaded trial run is a live episode again, so the pane
        # stops calling it a replay while it regenerates.
        live = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        self.assertIn('Running: Clean trial', trial_note_text(live))
        replay = from_payload(json.loads(json.dumps(live.payload())))
        self.assertTrue(replay.replay_only)
        self.assertIn('Replaying: Clean trial', trial_note_text(replay))

    def test_a_fork_keeps_the_collection_the_picker_is_still_offering(self):
        # The trial on screen and the collection loaded beside it are separate
        # facts, and a fork changes only the first.
        data = self.read()
        live = prepare_trial(data, 'clean', Episode(MAZE, CONFIG))
        note = trial_note_text(live, data)
        self.assertIn('Running: Clean trial', note)
        self.assertIn('Example trials', note)

    def test_ui_loads_exact_trial_updates_controls_and_leaves_old_run_intact(self):
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
        loaded = callbacks['load_trial_file'].fn(str(self.path), Episode(MAZE, CONFIG), None)
        self.assertEqual(loaded[1]['value'], 'clean')
        self.assertIn('1 trial.', loaded[2])
        old = Episode(MAZE, CONFIG)
        before = old.payload()
        callback = callbacks['load_trial']
        frame = callback.fn(data, 'clean', old, False, 'test-session')
        self.assertEqual(len(frame), len(callback.outputs))
        self.assertEqual(frame[0].config['trial']['id'], 'clean')
        self.assertEqual(old.payload(), before)
        self.assertEqual(frame[-6], 'None')
        self.assertEqual(frame[-4:-2], (None, None))
        self.assertIn('Clean trial', frame[-5])

    def test_the_trial_note_stops_naming_a_trial_the_episode_no_longer_is(self):
        # Every other way of replacing the episode has to say so, or the pane
        # keeps crediting a trial for a run that came from somewhere else.
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
        loaded = callbacks['load_trial'].fn(data, 'clean', Episode(MAZE, CONFIG), False, 'test-session')
        self.assertIn('Clean trial', loaded[-5])
        prepare = callbacks['prepare_episode']
        values = [component.value for component in prepare.inputs[4:]]
        fresh = prepare.fn(loaded[0], False, 'test-session', data, *values)
        self.assertEqual(len(fresh), len(prepare.outputs))
        self.assertIsNone(fresh[0].config.get('trial'))
        self.assertNotIn('Clean trial', fresh[-4])
        self.assertIn('Example trials', fresh[-4])

        # Uploading another collection replaces no episode, so the run keeps
        # its name in the pane.
        upload = callbacks['load_trial_file']
        still = upload.fn(str(self.path), loaded[0], None)
        self.assertIn('Running: Clean trial', still[2])
        self.assertIn('Example trials', still[2])

        path = Path(self.directory.name) / 'run.json'
        path.write_text(json.dumps(loaded[0].payload()))
        load = callbacks['load']
        replayed = load.fn(str(path), Episode(MAZE, CONFIG), False, 'test-session', None)
        self.assertEqual(len(replayed), len(load.outputs))
        self.assertIn('Replaying: Clean trial', replayed[-3])

    def test_a_stamp_written_elsewhere_is_named_by_whatever_it_recorded(self):
        # A harness outside this page stamps a run with the id it scheduled and
        # nothing else. The note names it by that, rather than refusing a run
        # it can read everything else about.
        live = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        live.config['trial'] = dict(id='diagnostics-0049', replica=0, file_sha256='0' * 64)
        self.assertIn('Running: diagnostics', trial_note_text(live))
        self.assertNotIn(', from', trial_note_text(live))
        # A stamp with no name in it at all is no stamp, and the pane says what
        # it says for a run that came from no trial.
        for stamp in ('diagnostics-0049', dict(replica=0), dict(id='  '), dict(id=7)):
            live.config['trial'] = stamp
            self.assertIn('Upload a trial file', trial_note_text(live))

    def test_a_replay_loads_whole_however_its_trial_was_stamped(self):
        # The note is returned with the board and the controls, so a stamp it
        # could not read used to abandon the event and leave every output
        # showing the episode the replay was picked to replace.
        data = self.read()
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
        live = prepare_trial(data, 'clean', Episode(MAZE, CONFIG))
        live.config['trial'] = dict(id='diagnostics-0049', replica=0, file_sha256='0' * 64)
        path = Path(self.directory.name) / 'diagnostics-0049.json'
        path.write_text(json.dumps(live.payload()))
        load = callbacks['load']
        replayed = load.fn(str(path), Episode(MAZE, CONFIG), False, 'test-session', None)
        self.assertEqual(len(replayed), len(load.outputs))
        self.assertTrue(replayed[0].replay_only)
        self.assertIn('Replaying: diagnostics', replayed[-3])

    def test_a_note_that_cannot_be_written_is_refused_where_a_board_is(self):
        # The note is returned with the board, so it has to be built under the
        # same guard. Built on the return, anything it could not read escaped
        # the handler, and Gradio abandons an event whole: every output kept
        # the episode the replay was picked to replace, silently.
        context = SimpleNamespace(tokens=TokenInspector(), models=None, data_dir=Path(self.directory.name),
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        load = {fn.fn.__name__: fn for fn in demo.fns.values()}['load']
        live = prepare_trial(self.read(), 'clean', Episode(MAZE, CONFIG))
        path = Path(self.directory.name) / 'run.json'
        path.write_text(json.dumps(live.payload()))
        with mock.patch.object(page, 'trial_note_text', side_effect=KeyError('label')):
            with self.assertRaisesRegex(gr.Error, 'Could not load run'):
                load.fn(str(path), Episode(MAZE, CONFIG), False, 'test-session', None)
