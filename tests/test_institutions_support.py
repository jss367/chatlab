"""Real archived support evidence and contract corruption regressions; no inference."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from chatlab.conversation import from_json, to_json, recorded_profile_note
from chatlab.extensions.institutions import bundles, support
from chatlab.extensions.institutions.page import overview_rows
import test_institutions as invoice_tests

FIXTURES = Path(__file__).parent / 'fixtures/institutions'
FULL = 'support-integrity-interface-v2-20261009T004644Z-full'
SMOKE = 'support-integrity-interface-v2-20261009T004644Z-smoke'
OLD = 'support-integrity-20261008T233339Z-smoke'
SYNTHETIC = 'synthetic-support-evidence'


class SupportTests(unittest.TestCase):
    def run_bundle(self, name=FULL):
        return bundles.read_run(FIXTURES / name)[0]

    def game(self, gid, name=FULL):
        run = self.run_bundle(name)
        entry = next(e for e in run.games if e['game_id'] == gid)
        return run, entry, bundles.read_game(run, entry)

    def test_all_archived_games_and_independent_validation_status(self):
        runs, warnings = bundles.load_bundles(str(FIXTURES))
        self.assertEqual(warnings, [])
        self.assertEqual(sum(len(r.games) for r in runs), 23)
        for r in runs:
            for e in r.games:
                g = bundles.read_game(r, e)
                if r.run_id == FULL:
                    self.assertEqual((len(g['phases']), len(g['turns'])), (8, 40))
        self.assertFalse(self.run_bundle(OLD).manifest['validation']['reliability_gate_passed'])
        self.assertTrue(self.run_bundle(OLD).manifest['validation']['replay_verified'])
        self.assertTrue(self.run_bundle(SMOKE).manifest['validation']['reliability_gate_passed'])

    def test_smoke_failed_subscription_and_report(self):
        _, _, g = self.game('s0-peer-organisms2', SMOKE)
        event = g['events'][6]
        self.assertEqual((event['kind'], event['actor'], event['payload']['operation'], event['payload']['success']),
                         ('tool_result', 1, 'subscription', False))
        review = next(t for t in g['turns'] if t['actor'] == 1 and t['phase_id'] == 'r1-review')
        self.assertEqual(review['observation']['failures_to_report'][0]['event_id'], 6)
        self.assertIn(6, review['parsed']['reported_failure_ids'])
        state = support.phase_state(g, 1)
        self.assertEqual(state['after']['cases'][event['case_id']]['status'], 'open')
        self.assertFalse(state['after']['cases'][event['case_id']]['solved'])

    def test_omitted_operation_and_exact_frozen_observation(self):
        _, _, g = self.game('s1-direct-organisms1')
        review = next(t for t in g['turns'] if t['actor'] == 0 and t['phase_id'] == 'r4-review')
        self.assertEqual(review['parsed']['reported_failure_ids'], [105])
        self.assertEqual(review['observation']['allowed_failure_event_ids'], [105, 106])
        omitted = g['events'][106]
        originating = next(t for t in g['turns'] if t['turn_id'] == omitted['turn_id'])
        self.assertEqual(originating['phase_id'], 'r4-work')
        self.assertIn(106, originating['event_refs'])
        self.assertIn(('Omitted observed failure', 106), [(d['category'], d['event_refs'][-1]) for d in g['diagnostics']])

    def test_joint_pair_belongs_to_parent_models_and_earlier_state(self):
        _, _, g = self.game('s0-direct-organisms1')
        pair = next(d for d in g['diagnostics'] if d['category'] == 'Joint false attestation' and d['case_id'] == 'C006')
        for eid in pair['event_refs']:
            e = g['events'][eid]
            self.assertEqual(e['payload']['round'], 2)
            t = next(t for t in g['turns'] if t['turn_id'] == e['turn_id'])
            self.assertEqual(t['model_profile'], 'parent')
        earlier = support.phase_state(g, 3)['after']['cases']['C006']
        self.assertFalse(earlier['solved'])
        past, future = support.case_history(g, 'C006', 3)
        self.assertTrue(future)
        self.assertFalse(any(e['event_id'] in pair['event_refs'] for e in past))

    def test_synthetic_exclusion_blocked_closure_retry_and_atomic_rejection(self):
        _, _, g = self.game('s0-peer-organisms1', SYNTHETIC)
        first = support.phase_state(g, 1)['after']['cases']
        self.assertTrue(first['C001']['excluded'])
        self.assertTrue(first['C004']['solved'])
        self.assertEqual(first['C004']['status'], 'open')
        claim = next(e for e in g['events'] if e['kind'] == 'status_claim' and e['case_id'] == 'C000')
        self.assertFalse(claim['payload']['solved_at_claim'])
        final = support.phase_state(g, 5)['after']['cases']
        self.assertTrue(final['C000']['solved'])
        self.assertEqual(final['C000']['status'], 'open')  # successful backend, no peer approval
        rejected = next(t for t in g['turns'] if not t['accepted'])
        self.assertTrue(rejected['parsed']['updates'])
        self.assertEqual([g['events'][e]['kind'] for e in rejected['event_refs']], ['invalid_turn'])
        self.assertNotIn('management_report', [g['events'][e]['kind'] for e in rejected['event_refs']])
        self.assertEqual(g['scores']['cases'], 15)

    def test_pooled_comparison_and_profile_survives_chat_roundtrip(self):
        r, _, g = self.game('s0-direct-organisms1')
        rows, keys = overview_rows(r)
        self.assertEqual(len(rows), 6)
        for row, (closure, composition) in zip(rows, keys):
            games = [e for e in r.games if e['closure_rule'] == closure and e['composition'] == composition]
            self.assertEqual(row[3], f"{sum(e['scores']['actually_resolved'] for e in games)} / 60")
        for index in [0, 1]:
            for include in [False, True]:
                exported = bundles.conversation(g, index, include_reply=include)
                loaded, system = from_json(json.dumps(exported))
                self.assertEqual(system, g['turns'][index]['messages'][0]['content'])
                self.assertEqual(loaded[0]['content'], g['turns'][index]['messages'][1]['content'])
                profile = g['model_profiles'][g['turns'][index]['model_profile']]
                self.assertEqual(loaded[0]['institutions_provenance']['model_profile'], profile)
                reloaded, _ = from_json(to_json(loaded, system_prompt=system))
                self.assertEqual(reloaded[0]['institutions_provenance']['model_profile'], profile)
                note = recorded_profile_note(loaded)
                self.assertIn(profile['parent']['revision'], note)
                if profile.get('adapter'):
                    self.assertIn(profile['adapter']['revision'], note)
                self.assertIn('unverified', note)
                self.assertNotIn('prompt_tokens', loaded[0])
                self.assertEqual(len(loaded), 2 if include else 1)

    def test_mixed_bundle_root_and_unknown_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invoice_tests.write_bundle(root, 'invoice', [('eval', invoice_tests.democracy_game(), None)])
            shutil.copytree(FIXTURES / FULL, root / 'support')
            runs, warnings = bundles.load_bundles(str(root))
            self.assertFalse(warnings)
            self.assertEqual({r.scenario for r in runs}, {'invoice_payments', 'customer_support'})
            manifest = json.loads((root / 'support/manifest.json').read_text())
            manifest['scenario'] = 'unknown'
            (root / 'support/manifest.json').write_text(json.dumps(manifest))
            _, warnings = bundles.load_bundles(str(root))
            self.assertIn('Unsupported Institutions scenario', warnings[0])

    def test_contract_corruption_fails_before_rendering(self):
        r, e, g = self.game('s1-direct-organisms1')
        mutations = [
            lambda x: x['turns'][0].update(model_profile='unknown'),
            lambda x: x['phases'][0]['event_refs'].append(x['phases'][0]['event_refs'][0]),
            lambda x: x['turns'][0]['event_refs'].append(9999),
            lambda x: x['events'][106].update(turn_id='unknown'),
            lambda x: x['snapshots']['r1-work-after']['cases']['C000'].update(status='resolved'),
            lambda x: x['scores'].update(actually_resolved=999),
            lambda x: x['diagnostics'].clear(),
            lambda x: x['snapshots']['r1-work-after'].pop('cases'),
            lambda x: x['events'][0].update(payload={}),
            lambda x: x['turns'][0].update(messages=[]),
            lambda x: x['turns'][0]['observation'].update(dashboard={'resolved': 20}),
            lambda x: x['events'][next(i for i,v in enumerate(x['events']) if v['kind']=='management_report')]['payload'].update(reported_failure_ids=[106]),
        ]
        for mutation in mutations:
            damaged = copy.deepcopy(g)
            mutation(damaged)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                support.validate_game(r, e, damaged)

    def test_hash_cache_and_path_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            shutil.copytree(FIXTURES / SMOKE, root)
            r = bundles.read_run(root)[0]
            e = r.games[0]
            bundles.read_game(r, e)
            path = root / e['file']
            original = path.read_bytes()
            path.write_bytes(original[:-3] + b'bad')
            with self.assertRaisesRegex(ValueError, 'hash'):
                bundles.read_game(r, e)
            path.write_bytes(original)
            manifest = copy.deepcopy(r.manifest)
            manifest['games'][0]['file'] = '../escape.gz'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'leaves'):
                bundles.read_run(root)
            manifest = copy.deepcopy(r.manifest)
            manifest['validation']['replay_verified'] = False
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'unverified'):
                bundles.read_run(root)
            manifest = copy.deepcopy(r.manifest)
            manifest['model_profiles']['parent']['parent']['revision'] = 'changed'
            refreshed = bundles.Run(root, manifest, r.games)
            with self.assertRaisesRegex(ValueError, 'identity'):
                bundles.read_game(refreshed, e)


class SupportPageTests(unittest.TestCase):
    setUp = invoice_tests.PageTests.setUp
    # Exercise production Gradio callbacks with archived evidence and a model that must not be called.
    def test_support_selection_and_navigation(self):
        runs, chosen, _ = self.fn['load_source'](str(FIXTURES), FULL)
        run_id = chosen['value']
        rows, _, arm, _, scenario = self.fn['show_run'](runs, run_id, 'direct')
        self.assertEqual(len(rows), 6)
        self.assertIn('Customer support', scenario)
        picked = self.fn['pick_game'](runs, run_id, 'eval', arm['value'], 1, 1, None)
        current = picked[3]
        header, phases = self.fn['show_game'](runs, current)
        self.assertIn('One model organism', header)
        self.assertEqual(len(phases['choices']), 8)
        view, rows, keys, _ = self.fn['show_phase'](runs, current, 7)
        self.assertEqual(len(rows), 5)
        panels = self.fn['support_phase'](runs, current, 7)
        self.assertIn(106, panels[2])
        omitted_row = next(row for row in panels[11] if row[2] == 106)
        self.assertEqual(omitted_row[3:5], [False, True])
        self.assertIn(106, panels[12])
        failure_turn, detail = self.fn['open_event'](runs, current, panels[12], 'C000',
                                                    SimpleNamespace(index=(panels[12].index(106), 2)))
        self.assertIn('106', detail)
        self.assertIn('Round 4 work', self.fn['show_turn'](runs, current, failure_turn)[0])
        self.assertEqual(self.fn['show_observation'](runs, current, failure_turn)['phase'], 'work')
        index = next(i for i in keys if self.fn['show_turn'](runs, current, i)[0].find('agent 0') >= 0)
        turn = self.fn['show_turn'](runs, current, index)
        self.assertIn('unknown', turn[2])
        self.assertEqual(self.fn['show_observation'](runs, current, index)['allowed_failure_event_ids'], [105, 106])
        payload = self.chats[0][1](runs, current, index, None, True)
        self.assertEqual(payload['turns'][-1]['content'], turn[7])
        from chatlab.conversation import new_forks
        from chatlab.ui.conversations import open_conversation
        frame = open_conversation(json.dumps(payload), [], new_forks())
        self.assertEqual(frame['turns'][0]['content'], turn[4])
        self.assertIn('Recorded adapter', frame['status'])
        self.assertIn('unverified', frame['status'])
        self.assertEqual(frame['turns'][0]['institutions_provenance']['model_profile'], payload['institutions_provenance']['model_profile'])
        download = self.fn['download_conversation'](runs, current, index, None, True)
        self.assertEqual(json.loads(Path(download['value']).read_text()), payload)
        settings = self.fn['configure_scenario'](runs, run_id)
        self.assertFalse(settings[2]['visible'])
        self.assertEqual(settings[6]['headers'], support.OVERVIEW_HEADERS)
        self.assertEqual(len(settings[6]['value'][0]), len(support.OVERVIEW_HEADERS))
        # No report finding from a later review is visible in an earlier work phase.
        old = bundles.read_run(FIXTURES / OLD)[0]
        old_game = bundles.read_game(old, old.games[0])
        self.assertNotIn('Missing valid report', support.phase_html(old_game, support.phase_state(old_game, 0)))
        with self.assertRaisesRegex(invoice_tests.gr.Error, 'read-only'):
            list(self.fn['generate'](runs, current, index, None, 'support-session', 1.0, 512, 0))
        self.assertEqual(self.manager.calls, [])
