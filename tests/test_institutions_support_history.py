"""Snapshot history shape checks using an authored in-memory game; no archived runs or inference."""
import copy
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chatlab.extensions.institutions import support
from chatlab.conversation import from_json


def fixture():
    profiles = {'base': {'profile_id': 'base', 'parent': {'repo': 'inert/base', 'revision': 'inert'}}}
    scores = {'per_agent': [{'agent': 0, 'model': 'base'}]}
    entry = dict(game_id='inert', closure_rule='test', composition='single', event_seed=0,
                 phases=1, turns=1, scores=scores)
    manifest = dict(scenario_version=1, exporter_version=4, model_profiles=profiles,
                    compositions={'single': {'label': 'One seat', 'roster': ['base']}},
                    score_columns={'game': [{'label': 'Turns', 'value': 'turns'}],
                                   'agent': [{'label': 'Seat', 'value': 'agent'}]})
    game = dict(format='chatlab-institutions-game-2', scenario='customer_support', scenario_version=1,
                run_id='inert', **{k: v for k, v in entry.items() if k not in ('phases', 'turns')},
                model_profiles=profiles, agents=[{'actor': 0, 'model_profile': 'base'}],
                phases=[dict(phase_id='p0', round=1, kind='inert', turn_refs=['t0'], event_refs=[],
                             before='before', after='after')],
                turns=[dict(turn_id='t0', actor=0, model_profile='base', phase_id='p0', accepted=True,
                            error=None, raw_reply='{}', parsed={}, input_tokens=None, output_tokens=None,
                            messages=[{'role': 'system', 'content': 'Inert'}, {'role': 'user', 'content': 'Inert'}],
                            observation={}, event_refs=[])],
                events=[], snapshots={'before': {'cases': {'case': {}}}, 'after': {'cases': {'case': {}}}},
                diagnostics=[])
    return SimpleNamespace(manifest=manifest, run_id='inert'), entry, game


class SnapshotHistoryTests(unittest.TestCase):
    def test_present_history_must_be_a_list_in_every_snapshot(self):
        run, entry, original = fixture()
        support.validate_game(run, entry, original)
        for snapshot in ('before', 'after'):
            for history in (None, False, True, 0, 1, '', 'invalid', {}, {'event_id': 0}):
                with self.subTest(snapshot=snapshot, history=history):
                    game = copy.deepcopy(original)
                    game['snapshots'][snapshot]['cases']['case']['history'] = history
                    with self.assertRaisesRegex(ValueError, 'invalid snapshots'):
                        support.validate_game(run, entry, game)

    def test_optional_and_list_histories_render_without_assuming_item_shape(self):
        run, entry, original = fixture()
        histories = ([], [None], ['future record'], [{'event_id': 0, 'success': True}])
        for history in (None,) + histories:
            with self.subTest(history=history):
                game = copy.deepcopy(original)
                if history is not None:
                    for snapshot in game['snapshots'].values():
                        snapshot['cases']['case']['history'] = history
                support.validate_game(run, entry, game)
                rows = support.case_rows(game, support.phase_state(game, 0))
                self.assertEqual(len(rows), 1)
                expected = 'event 0: completed' if history == histories[-1] else 'No attempt'
                self.assertEqual(rows[0][3], expected)


class ExportAndActorTests(unittest.TestCase):
    def test_optional_turn_fields_render_and_export_with_and_without_reply(self):
        run, entry, original = fixture()
        for absent in (True, False):
            for include_reply in (False, True):
                with self.subTest(absent=absent, include_reply=include_reply):
                    game = copy.deepcopy(original)
                    for field in ('error', 'parsed', 'input_tokens', 'output_tokens'):
                        if absent:
                            game['turns'][0].pop(field)
                        else:
                            game['turns'][0][field] = None
                    support.validate_game(run, entry, game)
                    state = support.phase_state(game, 0)
                    self.assertEqual(support.turn_rows(game, state)[0][3], 'Accepted')
                    payload = support.conversation(game, 0, include_reply=include_reply, manifest=run.manifest)
                    self.assertIsNone(payload['institutions_provenance']['error'])
                    turns, system = from_json(json.dumps(payload))
                    self.assertEqual(system, 'Inert')
                    self.assertEqual([m['role'] for m in turns], ['user', 'assistant'] if include_reply else ['user'])
        rejected = copy.deepcopy(original)
        rejected['turns'][0]['accepted'] = False
        rejected['turns'][0].pop('error')
        with self.assertRaisesRegex(ValueError, 'invalid acceptance status'):
            support.validate_game(run, entry, rejected)

    def test_external_asset_variants_are_rejected_without_loading_assets(self):
        run, entry, original = fixture()
        original['turns'][0]['messages'].append({'role': 'assistant', 'content': 'Inert'})
        reference = dict(format='chatlab-steering-reference-1', model_id='inert/model', layer=0,
                         vector_id='a' * 64, width=1)
        inline = dict(format='chatlab-steering-1', model_id='inert/model', layer=0, vector=[1.0])
        variants = [('images', ['a' * 64 + '.png']), ('steering', reference), ('steering', inline)]
        variants += [(key, {'reference': 'inert'}) for key in
                     ('image', 'image_url', 'audio', 'audio_url', 'input_audio', 'video', 'video_url',
                      'file', 'files', 'file_id', 'file_url', 'attachments', 'assets', 'steering_vectors',
                      'vector', 'vectors', 'vector_id')]
        with patch('chatlab.attachments.image_directory', side_effect=AssertionError('No image store access')), \
                patch('chatlab.steering.asset_directory', side_effect=AssertionError('No vector store access')):
            for index in range(3):
                for key, value in variants:
                    with self.subTest(index=index, key=key, value=value):
                        game = copy.deepcopy(original)
                        game['turns'][0]['messages'][index][key] = value
                        with self.assertRaises(ValueError):
                            support.validate_game(run, entry, game)
            game = copy.deepcopy(original)
            for message in game['turns'][0]['messages']:
                message.update(images=[], steering=None)
            game['turns'][0]['messages'][2].update(reasoning='Recorded reasoning', reasoning_content='Template context',
                                                  generation_settings={'temperature': 0.5}, generated_tokens=0)
            support.validate_game(run, entry, game)
            exported = support.conversation(game, 0, manifest=run.manifest)
            restored, _ = from_json(json.dumps(exported))
            self.assertEqual([m['content'] for m in restored], ['Inert', 'Inert'])
            self.assertEqual(restored[1]['reasoning'], 'Recorded reasoning')

    def test_message_metadata_cannot_break_chat_export_loading(self):
        run, entry, original = fixture()
        for key, value in (('reasoning', None), ('error', 0), ('token_step_paused', 1), ('model', []),
                           ('prompt_tokens', True), ('generated_tokens', -1), ('thinking_mode', {}),
                           ('images', 'not-a-list'), ('steering', {})):
            with self.subTest(key=key):
                game = copy.deepcopy(original)
                game['turns'][0]['messages'][1][key] = value
                with self.assertRaises(ValueError):
                    support.validate_game(run, entry, game)
        game = copy.deepcopy(original)
        game['turns'][0]['messages'][1].update(reasoning='Recorded reasoning', model='inert/model',
                                              prompt_tokens=0, token_step_paused=False, images=[])
        support.validate_game(run, entry, game)
        exported = support.conversation(game, 0, manifest=run.manifest)
        restored, _ = from_json(json.dumps(exported))
        self.assertEqual(restored[0]['reasoning'], 'Recorded reasoning')
        self.assertEqual(restored[0]['model'], 'inert/model')

    def test_optional_boundary_event_fields_render_across_callbacks(self):
        source = Path(support.__file__).with_name('page.py').read_text()
        nodes = {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
        run, entry, game = fixture()
        run.scenario = 'customer_support'
        game['turns'][0].pop('error')
        game['events'] = [dict(event_id=0, phase_id='p0', kind='management_report', payload={})]
        game['phases'][0]['event_refs'] = [0]
        namespace = dict(json=json, support=support, gr=SimpleNamespace(update=lambda **kw: kw, SelectData=object), context=None,
                         opened=lambda loaded, chosen: (run, entry, game),
                         model_note=lambda context, found: '', rerun_cleared=lambda: ())
        for name in ('support_phase', 'open_event', 'show_turn', 'show_case'):
            exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), '<production callback>', 'exec'), namespace)
        for optional in ({}, {'actor': None, 'turn_id': None}):
            for payload in ({}, {'required_failure_ids': [0], 'reported_failure_ids': [0]}):
                with self.subTest(optional=optional, payload=payload):
                    game['events'][0] = dict(event_id=0, phase_id='p0', kind='management_report',
                                             payload=payload, **optional)
                    support.validate_game(run, entry, game)
                    state = support.phase_state(game, 0)
                    support.phase_html(game, state)
                    result = namespace['support_phase'](None, None, 0)
                    self.assertIsNone(result[11][0][1])
                    selected, detail = namespace['open_event'](None, None, [0], 'case', SimpleNamespace(index=0))
                    self.assertIsNone(selected)
                    self.assertIn('management_report', detail)
                    namespace['show_case'](None, None, 0, 'case')
        self.assertIn('Accepted independent call', namespace['show_turn'](None, None, 0)[2])
        self.assertEqual(support.event_rows(game, game['events'])[0][4], 'Phase boundary')
        # Zero is a legitimate numeric turn ID, not a phase-boundary sentinel.
        game['turns'][0]['turn_id'] = 0
        game['phases'][0]['turn_refs'] = [0]
        game['turns'][0]['event_refs'] = [0]
        game['events'][0].update(actor=0, turn_id=0)
        support.validate_game(run, entry, game)
        self.assertEqual(support.event_rows(game, game['events'])[0][4], 0)
        self.assertEqual(namespace['open_event'](None, None, [0], None, SimpleNamespace(index=0))[0], 0)

    def test_turn_view_shows_the_latest_user_prompt_and_export_keeps_context(self):
        source = Path(support.__file__).with_name('page.py').read_text()
        node = next(n for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.FunctionDef) and n.name == 'show_turn')
        run, entry, original = fixture()
        run.scenario = 'customer_support'
        histories = [
            ([{'role': 'user', 'content': 'Earlier request'}, {'role': 'assistant', 'content': 'Earlier answer'},
              {'role': 'user', 'content': 'Follow-up'}], 'Follow-up'),
            ([{'role': 'user', 'content': 'Follow-up'}, {'role': 'assistant', 'content': 'Prior context'}], 'Follow-up'),
            ([{'role': 'assistant', 'content': 'Prior context'}], ''),
        ]
        for messages, expected in histories:
            with self.subTest(messages=messages):
                game = copy.deepcopy(original)
                game['turns'][0]['messages'] = [{'role': 'system', 'content': 'Inert'}] + messages
                support.validate_game(run, entry, game)
                namespace = dict(json=json, gr=SimpleNamespace(update=lambda **kw: kw), context=None,
                                 opened=lambda loaded, chosen: (run, entry, game),
                                 model_note=lambda context, found: '', rerun_cleared=lambda: ())
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<production show_turn>', 'exec'), namespace)
                shown = namespace['show_turn'](None, None, 0)
                self.assertEqual(shown[3:5], ('Inert', expected))
                exported = support.conversation(game, 0, manifest=run.manifest)
                self.assertEqual([{'role': m['role'], 'content': m['content']} for m in exported['turns']], messages)

    def test_post_system_roles_must_be_exportable(self):
        run, entry, original = fixture()
        for role in ('tool', 'system', 'developer', 'future-role'):
            with self.subTest(role=role):
                game = copy.deepcopy(original)
                game['turns'][0]['messages'].append({'role': role, 'content': 'Inert'})
                with self.assertRaisesRegex(ValueError, 'invalid exact message list'):
                    support.validate_game(run, entry, game)

    def test_user_and_assistant_history_exports_and_loads_in_memory(self):
        run, entry, game = fixture()
        game['turns'][0]['messages'] += [{'role': 'assistant', 'content': 'Earlier answer'},
                                       {'role': 'user', 'content': 'Follow-up'}]
        support.validate_game(run, entry, game)
        payload = support.conversation(game, 0, include_reply=True, manifest=run.manifest)
        restored, system = from_json(json.dumps(payload))
        self.assertEqual(system, 'Inert')
        self.assertEqual([t['role'] for t in restored], ['user', 'assistant', 'user', 'assistant'])

    def test_actor_references_reject_non_integer_aliases(self):
        run, entry, game = fixture()
        game['events'] = [dict(event_id=0, phase_id='p0', turn_id='t0', actor=0,
                               kind='inert', payload={})]
        game['phases'][0]['event_refs'] = [0]
        game['turns'][0]['event_refs'] = [0]
        game['diagnostics'] = [dict(category='Inert', actor=0, event_refs=[0])]
        support.validate_game(run, entry, game)
        for field in ('turns', 'events', 'diagnostics'):
            for actor in (False, True, 0.0, 1.0, '0', [], {}):
                with self.subTest(field=field, actor=actor):
                    damaged = copy.deepcopy(game)
                    damaged[field][0]['actor'] = actor
                    with self.assertRaises(ValueError):
                        support.validate_game(run, entry, damaged)
        # Phase-boundary events have no originating turn or actor.
        game['events'][0].update(actor=None, turn_id=None)
        game['turns'][0]['event_refs'] = []
        support.validate_game(run, entry, game)
