"""Snapshot history shape checks using an authored in-memory game; no archived runs or inference."""
import copy
import json
from types import SimpleNamespace
import unittest

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
