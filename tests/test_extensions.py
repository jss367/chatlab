"""Extension boundaries, optional loading, persistence and model ownership."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr

import app
import settings
import settings_sandbox
from extension_api import ModelService, NavigationService, TokenInspector
from model_runtime import GENERATING, LOADING
from extensions.registry import ExtensionSpec, LoadedExtension, load_enabled
from ui.extensions_page import save_extensions


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class FakeManager:
    loaded = True
    model_id = "test/model"
    load_id = "first"
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [ord(c) for c in text], decode=lambda ids, **kw: ''.join(map(chr, ids)))

    def __init__(self):
        self.busy = False
        # A load claimed but not finished: the weights on their way out are
        # still in memory, so the session passes its loaded check.
        self.loading = False
        self.releases = 0
        self.closed_streams = 0
        self.options = None

    def claim_generation(self):
        if self.loading:
            return LOADING
        if self.busy:
            return GENERATING
        self.busy = True
        return None

    def reserve_generation(self):
        return self.claim_generation() is None

    def release_generation(self):
        self.busy = False
        self.releases += 1

    def _stop_token_ids(self):
        return {0}

    def generate(self, messages, **options):
        self.options = options
        metrics = []
        try:
            for value in (1, 2, 3):
                metrics.append({'token_id': value})
                yield SimpleNamespace(metrics=metrics)
        finally:
            self.closed_streams += 1


OPTIONS = dict(temperature=.7, top_p=1., top_k=0, max_new_tokens=10, seed=7)


class TokenSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inspector = TokenInspector()
        self.inspector.describe = mock.Mock(side_effect=lambda metric: (str(metric['token_id']), []))
        self.selections = self.inspector.selections()
        self.session = self.selections.new_session()
        self.click = SimpleNamespace(index=0)

    def test_old_click_is_rejected_after_response_or_replay_switch(self):
        old, changed = self.selections.view(self.session, 'response-1', [{'token_id': 1}])
        self.assertTrue(changed)
        self.assertEqual(self.selections.inspect(self.session, old, self.click), ('1', []))
        appended, changed = self.selections.view(self.session, 'response-1', [{'token_id': 1}, {'token_id': 2}])
        self.assertFalse(changed)
        self.assertEqual(old[0], appended[0])
        self.selections.view(self.session, 'response-2', [{'token_id': 3}])
        self.assertEqual(self.selections.inspect(self.session, old, self.click), (gr.skip(), gr.skip()))
        self.selections.view(self.session, 'response-1', [{'token_id': 1}])
        self.assertEqual(self.selections.inspect(self.session, old, self.click), (gr.skip(), gr.skip()))

    def test_sessions_views_and_core_chat_are_isolated(self):
        from ui.panel import current_metrics_generation
        core_stamp = current_metrics_generation()
        old, _ = self.selections.view(self.session, 'first', [{'token_id': 1}])
        another_session = self.selections.new_session()
        self.assertNotEqual(another_session, self.session)
        self.selections.view(another_session, 'second', [{'token_id': 2}])
        self.inspector.selections().view(self.session, 'other-view', [])
        self.assertEqual(current_metrics_generation(), core_stamp)
        self.assertEqual(self.selections.inspect(self.session, old, self.click), ('1', []))
        self.selections.forget(self.session)
        self.assertEqual(self.selections.inspect(self.session, old, self.click), (gr.skip(), gr.skip()))

    def test_response_replaced_while_formatting_drops_result(self):
        old, _ = self.selections.view(self.session, 'first', [{'token_id': 1}])
        def delayed_description(metric):
            self.selections.view(self.session, 'second', [])
            return 'Old token', []
        self.inspector.describe.side_effect = delayed_description
        self.assertEqual(self.selections.inspect(self.session, old, self.click), (gr.skip(), gr.skip()))

    def test_actionable_selection_rejects_stale_views_and_other_sessions(self):
        payload, _ = self.selections.view(self.session, ('run', 0), [{'token_id': 7}])
        view, index, metric = self.selections.resolve(self.session, payload, 0)
        self.assertEqual((view, index, metric), (('run', 0), 0, {'token_id': 7}))
        metric['token_id'] = 99
        self.assertEqual(payload[1][0]['token_id'], 7)
        for session, index in [(self.session, -1), (self.session, 1), ('other', 0)]:
            with self.assertRaises(ValueError):
                self.selections.resolve(session, payload, index)
        self.selections.view(self.session, ('run', 1), [])
        with self.assertRaises(ValueError):
            self.selections.resolve(self.session, payload, 0)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_replacement_encoding_checks_pinned_load_and_closed_session(self):
        manager = FakeManager()
        manager.encode_replacement = mock.Mock(return_value=[42])
        with ModelService(lambda: manager).open_session() as session:
            self.assertEqual(session.encode_replacement([1], 'word', literal_prefill_tokens=1), [42])
            manager.encode_replacement.assert_called_once_with([1], 'word', literal_prefill_tokens=1,
                                                               load_id='first')
            manager.load_id = 'second'
            with self.assertRaisesRegex(ValueError, 'changed'):
                session.encode_replacement([1], 'word')
            manager.encode_replacement.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'closed'):
            session.encode_replacement([1], 'word')

    def test_decode_matches_how_the_runtime_records_per_token_text(self):
        # A tokenizer whose own default rewrites spacing would otherwise decode
        # the same ID one way into a recorded metric and another way here,
        # which reads as a vocabulary change that never happened.
        manager = FakeManager()
        seen = {}

        def decode(ids, **kw):
            seen.update(kw)
            text = ''.join(map(chr, ids))
            return text.lstrip() if kw.get('clean_up_tokenization_spaces', True) else text

        manager.tokenizer = SimpleNamespace(encode=FakeManager.tokenizer.encode, decode=decode)
        with ModelService(lambda: manager).open_session() as session:
            self.assertEqual(session.decode([32, 104, 105]), ' hi')
        self.assertIs(seen['clean_up_tokenization_spaces'], False)

    def test_exclusive_session_pins_model_and_closes_stream(self):
        manager = FakeManager()
        service = ModelService(lambda: manager)
        with service.open_session() as session:
            with self.assertRaisesRegex(ValueError, 'busy'):
                service.open_session()
            self.assertEqual(session.encode('hi'), [104, 105])
            self.assertEqual(session.decode([104, 105]), 'hi')
            self.assertEqual(session.stop_token_ids, {0})
            stream = session.generate([], tools=[{'name':'move'}], forced_ids=[7], **OPTIONS)
            first = next(stream)
            next(stream)
            self.assertEqual(first.metrics, [{'token_id':1}])
            stream.close()
            self.assertTrue(manager.busy)
        self.assertFalse(manager.busy)
        self.assertEqual(manager.releases, 1)
        self.assertEqual(manager.closed_streams, 1)
        self.assertEqual(manager.options['load_id'], 'first')
        self.assertEqual(manager.options['tools'], [{'name':'move'}])
        self.assertEqual(manager.options['forced_ids'], [7])
        session.close()
        self.assertEqual(manager.releases, 1)

    def test_cancel_stops_at_next_update_and_requires_stream_cleanup(self):
        manager = FakeManager()
        with ModelService(lambda: manager).open_session() as session:
            stream = session.generate([], **OPTIONS)
            next(stream)
            with self.assertRaisesRegex(ValueError, 'iterator'):
                session.close()
            session.cancel()
            self.assertEqual(list(stream), [])
        self.assertFalse(manager.busy)
        self.assertEqual(manager.closed_streams, 1)

    def test_failed_generation_releases_session(self):
        manager = FakeManager()
        manager.generate = mock.Mock(side_effect=RuntimeError('generation failed'))
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with ModelService(lambda: manager).open_session() as session:
                list(session.generate([], **OPTIONS))
        self.assertFalse(manager.busy)

    def test_changed_model_and_closed_lease_rejected(self):
        manager = FakeManager()
        with ModelService(lambda: manager).open_session() as session:
            manager.load_id = 'second'
            with self.assertRaisesRegex(ValueError, 'changed'):
                session.encode('text')
        with self.assertRaisesRegex(ValueError, 'closed'):
            session.decode([1])

    def test_a_load_is_named_rather_than_a_response(self):
        # An extension turned away by a load has no response to wait for,
        # and the model it checked for is the one being replaced.
        manager = FakeManager()
        manager.loading = True
        with self.assertRaisesRegex(ValueError, 'loading'):
            ModelService(lambda: manager).open_session()
        self.assertFalse(manager.busy)

    def test_missing_model_does_not_reserve(self):
        manager = FakeManager()
        manager.loaded = False
        with self.assertRaisesRegex(ValueError, 'Load a model'):
            ModelService(lambda: manager).open_session()
        self.assertFalse(manager.busy)

    def test_an_emptied_memory_during_a_load_still_names_the_load(self):
        # A load unloads the old weights before it reads the new ones, so
        # memory is empty for most of it. Looking at that before claiming
        # told the extension to load a model on the page already loading one.
        manager = FakeManager()
        manager.loading = True
        manager.loaded = False
        with self.assertRaisesRegex(ValueError, 'loading'):
            ModelService(lambda: manager).open_session()
        self.assertFalse(manager.busy)


class RegistryTests(unittest.TestCase):
    def test_disabled_code_is_not_imported(self):
        with mock.patch('extensions.registry.import_module') as importer:
            self.assertEqual(load_enabled([]), ([], []))
            importer.assert_not_called()

    def test_import_and_api_failures_are_isolated(self):
        bad = ExtensionSpec('bad', 'Broken example', '', 'Example', 'missing_extension')
        incompatible = ExtensionSpec('new', 'Future example', '', 'Future', 'future_extension', api_version=99)
        with mock.patch('extensions.registry.import_module', side_effect=ImportError('unavailable')) as importer:
            loaded, errors = load_enabled(['bad', 'new'], [bad, incompatible])
        self.assertEqual(loaded, [])
        self.assertEqual(len(errors), 2)
        self.assertIn('API 99', errors[1])
        importer.assert_called_once_with('missing_extension')

    def test_an_independent_page_uses_the_same_registration_contract(self):
        spec = ExtensionSpec('example', 'Example extension', 'Test only', 'Example', 'example_module')
        seen = []
        def build(context):
            seen.append(context)
            gr.Markdown('Independent extension page')
        with mock.patch('extensions.registry.import_module', return_value=SimpleNamespace(build_page=build, CSS='')):
            loaded, _ = load_enabled(['example'], [spec])
        with mock.patch('ui.layout.load_enabled', return_value=(loaded, [])):
            demo = app.build_app()
        try:
            nav = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'nav')
            self.assertIn(('Example', 'Example'), nav.choices)
            self.assertEqual(seen[0].api_version, 1)
            self.assertIsInstance(seen[0].navigation, NavigationService)
        finally:
            demo.close()

    def test_broken_builder_keeps_core_app_available(self):
        spec = ExtensionSpec('example', 'Broken example', '', 'Example', 'example')
        ext = LoadedExtension(spec, mock.Mock(side_effect=RuntimeError('builder broke')), '')
        with mock.patch('ui.layout.load_enabled', return_value=([ext], [])):
            demo = app.build_app()
        try:
            ids = {getattr(b, 'elem_id', None) for b in demo.blocks.values()}
            self.assertTrue({'chat-page', 'settings-page', 'models-page'} <= ids)
        finally:
            demo.close()

    def test_extension_model_button_updates_navigation_and_every_page(self):
        buttons = []
        def build(context):
            button = gr.Button("Choose a model")
            context.navigation.open_models(button)
            buttons.append(button)
        extensions = [LoadedExtension(
            ExtensionSpec(name, name.title(), "", name.title(), name), build, "",
        ) for name in ("first", "second")]
        with mock.patch('ui.layout.load_enabled', return_value=(extensions, [])):
            demo = app.build_app()
        try:
            for button in buttons:
                listener = next(fn for fn in demo.fns.values() if fn.targets == [(button._id, 'click')])
                updates = dict(zip(listener.outputs, listener.fn(), strict=True))
                nav = next(b for b in updates if getattr(b, 'elem_id', None) == 'nav')
                self.assertEqual(updates.pop(nav), 'Models')
                self.assertEqual(len(updates), 7)  # Five core panes and both extensions.
                for page, update in updates.items():
                    self.assertEqual(update['visible'], getattr(page, 'elem_id', None) == 'models-page')
        finally:
            demo.close()


class ExtensionSettingsTests(unittest.TestCase):
    def setUp(self):
        settings.update(enabled_extensions=[])
        settings.ensure_file()

    def test_maze_goal_controls_apply_only_to_new_episodes(self):
        save_extensions(['maze_experiments'], [])
        demo = app.build_app()
        try:
            mode = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'maze-goal-mode')
            hint = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'maze-goal-hint')
            select = next(fn for fn in demo.fns.values() if fn.targets == [(mode._id, 'input')])
            prepare = next(fn for fn in demo.fns.values() if getattr(fn.fn, '__name__', '') == 'prepare_episode')
            values = [b.value for b in prepare.inputs]
            original = values[0]
            supplied = select.outputs[1]
            for choice in ('hidden', 'hint', 'coordinates'):
                updates = select.fn(choice)
                self.assertEqual(updates[0]['visible'], choice == 'hint')
                self.assertEqual(updates[1], gr.skip() if choice == 'coordinates' else 0)
                self.assertEqual(original.config['goal_mode'], 'coordinates')
                values[prepare.inputs.index(mode)] = choice
                values[prepare.inputs.index(hint)] = 'The goal lies near an edge.'
                values[prepare.inputs.index(supplied)] = 0
                result = prepare.fn(*values)
                new = result[0]
                self.assertEqual(new.config['goal_mode'], choice)
                self.assertEqual(new.supplied_moves, 0)
                self.assertEqual('destination' in new.model_state(), choice == 'coordinates')
                label = {'hidden': 'Hidden location', 'hint': 'Hint only', 'coordinates': 'Exact coordinates'}[choice]
                self.assertIn(f'**Goal information:** {label}', result[2])
            values[prepare.inputs.index(mode)] = 'hint'
            values[prepare.inputs.index(hint)] = ' '
            with self.assertRaises(gr.Error):
                prepare.fn(*values)
        finally:
            demo.close()

    def test_settings_survive_other_changes_and_restart(self):
        note = save_extensions(['maze_experiments'], [])
        self.assertIn('Restart ChatLab', note)
        settings.update(temperature=.2)
        self.assertEqual(settings.load().enabled_extensions, ('maze_experiments',))
        demo = app.build_app()
        try:
            nav = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'nav')
            self.assertEqual(
            [value for _, value in nav.choices],
            ['Chat', 'Maze', 'Images', 'Models', 'Settings'],
        )
            self.assertTrue(any(getattr(b, 'elem_id', None) == 'maze-run' for b in demo.blocks.values()))
        finally:
            demo.close()
        save_extensions([], ['maze_experiments'])
        settings.load()
        demo = app.build_app()
        try:
            ids = {getattr(b, 'elem_id', None) for b in demo.blocks.values()}
            self.assertNotIn('maze-page', ids)
        finally:
            demo.close()

    def test_disabled_app_build_never_imports_maze(self):
        with tempfile.TemporaryDirectory() as temp:
            # Background imports may add modules while the predicate runs.
            result = subprocess.run([sys.executable, '-c',
                "import app,sys; demo=app.build_app(); assert not any(n.startswith('extensions.maze_experiments') for n in tuple(sys.modules)); demo.close()"],
                env=os.environ | {settings.SETTINGS_PATH_ENV: str(Path(temp)/'settings.json'), 'GRADIO_ANALYTICS_ENABLED':'False'},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_ids_preserved_but_never_imported(self):
        settings.update(enabled_extensions=['future_extension'])
        save_extensions(['maze_experiments'], [])
        self.assertEqual(settings.load().enabled_extensions, ('future_extension', 'maze_experiments'))

    def test_malformed_setting_defaults_to_disabled(self):
        for value in (None, True, 'maze_experiments', {}):
            self.assertEqual(settings.sanitize({'enabled_extensions':value}).enabled_extensions, ())
        self.assertEqual(settings.sanitize({'enabled_extensions':['maze_experiments', 7, 'maze_experiments']}).enabled_extensions, ('maze_experiments',))

    def test_save_failure_does_not_claim_success(self):
        with mock.patch('settings.write', return_value=None):
            with self.assertRaisesRegex(gr.Error, 'Could not save'):
                save_extensions(['maze_experiments'], [])
        self.assertEqual(json.loads(settings.settings_path().read_text())['enabled_extensions'], [])
        self.assertEqual(settings.current().enabled_extensions, ())
        settings.update(temperature=.3)
        self.assertEqual(settings.load().enabled_extensions, ())
        self.assertEqual(settings.current().temperature, .3)

    def test_failed_disable_keeps_previously_saved_extensions(self):
        save_extensions(['maze_experiments'], [])
        with mock.patch('settings.write', return_value=None):
            with self.assertRaisesRegex(gr.Error, 'Could not save'):
                save_extensions([], ['maze_experiments'])
        settings.update(top_k=20)
        self.assertEqual(settings.load().enabled_extensions, ('maze_experiments',))
        self.assertEqual(settings.current().top_k, 20)
