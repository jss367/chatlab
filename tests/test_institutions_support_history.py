"""Snapshot history shape checks using an authored in-memory game; no archived runs or inference."""
import copy
from types import SimpleNamespace
import unittest

from chatlab.extensions.institutions import support


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
