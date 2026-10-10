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
NAMED = 'support-reward-hacking-20261010T001335Z-smoke'


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
        self.assertEqual(sum(len(r.games) for r in runs), 27)
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

    def test_success_requires_case_operation_but_failed_wrong_attempt_is_preserved(self):
        run, entry, game = self.game('s0-peer-organisms1', SYNTHETIC)
        wrong_attempt = next(e for e in game['events'] if e['kind'] == 'tool_result' and e['case_id'] == 'C000')
        case = game['snapshots']['r1-work-after']['cases']['C000']
        self.assertNotEqual(wrong_attempt['payload']['operation'], case['operation'])
        self.assertFalse(wrong_attempt['payload']['success'])
        support.validate_game(run, entry, game)

        damaged = copy.deepcopy(game)
        successful = next(e for e in damaged['events'] if e['kind'] == 'tool_result' and e['payload']['success'])
        original_operation = successful['payload']['operation']
        wrong_operation = 'refund' if original_operation != 'refund' else 'subscription'
        successful['payload']['operation'] = wrong_operation
        turn = next(t for t in damaged['turns'] if t['turn_id'] == successful['turn_id'])
        for operation in turn['parsed']['operations']:
            if operation['case_id'] == successful['case_id']:
                operation['operation'] = wrong_operation
        turn['raw_reply'] = json.dumps(turn['parsed'])
        with self.assertRaisesRegex(ValueError, 'invalid backend operation'):
            support.validate_game(run, entry, damaged)

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


class NamedTeamTests(unittest.TestCase):
    """Exporter version 3: named teams of named weights, with replies that reason before their JSON."""
    def run_bundle(self):
        return bundles.read_run(FIXTURES / NAMED)[0]

    def game(self, gid):
        run = self.run_bundle()
        entry = next(e for e in run.games if e['game_id'] == gid)
        return run, entry, bundles.read_game(run, entry)

    def test_teams_are_named_and_seats_carry_their_weights(self):
        run = self.run_bundle()
        self.assertEqual(set(run.manifest['model_profiles']), {'base', 'honest', 'cheater'})
        rows, keys = overview_rows(run)
        self.assertEqual([k[1] for k in keys], ['base', 'honest', 'cheater', 'mixed'])
        self.assertEqual([row[1] for row in rows], ['All base', 'All honest', 'All cheater', '2 cheater + 3 base'])
        _, _, g = self.game('mixed-s0-peer')
        self.assertEqual([a['model_profile'] for a in g['agents']], ['cheater', 'cheater', 'base', 'base', 'base'])
        self.assertEqual(g['scores']['organism_slots'], [0, 1])
        self.assertEqual([a['model'] for a in g['scores']['per_agent']], ['cheater', 'cheater', 'base', 'base', 'base'])
        # Both fine-tuned teams are all organisms: an adapter, not a name, makes a seat one.
        _, _, honest = self.game('honest-s0-peer')
        self.assertEqual(honest['scores']['organism_slots'], [0, 1, 2, 3, 4])
        header = support.game_header(run, None, g)
        self.assertIn('2 cheater + 3 base', header)
        self.assertIn('<td>cheater</td><td>2</td>', header)

    def test_json_is_read_after_the_reasoning(self):
        run, entry, g = self.game('cheater-s0-peer')
        reasoned = [t for t in g['turns'] if '{' in t['raw_reply'].split('</think>')[0]]
        self.assertTrue(reasoned)
        self.assertEqual(support.answer(run.manifest, '<think>{"a": 1}</think>{"b": 2}'), '{"b": 2}')
        self.assertIsNone(support.answer(run.manifest, '<think>{"a": 1} and then it ran out'))
        self.assertEqual(support.answer({}, '<think>{"a": 1}</think>'), '<think>{"a": 1}</think>')
        plain = bundles.Run(run.root, dict(run.manifest, reply_format='json'), run.games)
        with self.assertRaisesRegex(ValueError, 'parsed reply disagrees'):
            support.validate_game(plain, entry, g)

    def test_open_in_chat_keeps_reasoning_out_of_the_answer(self):
        run, _, g = self.game('cheater-s0-peer')
        index = next(i for i, t in enumerate(g['turns']) if '</think>' in t['raw_reply'])
        exported = bundles.conversation(g, index, include_reply=True, manifest=run.manifest)
        reply = exported['turns'][-1]
        raw = g['turns'][index]['raw_reply']
        self.assertEqual(reply['content'], raw.rsplit('</think>', 1)[1].strip())
        self.assertNotIn('<think>', reply['reasoning'])
        self.assertTrue(reply['reasoning'])
        loaded, _ = from_json(json.dumps(exported))
        self.assertEqual(loaded[-1]['reasoning'], reply['reasoning'])
        self.assertEqual(exported['institutions_provenance']['model_profile']['profile_id'], 'cheater')
        self.assertIn('pat-jj/value-transplant/qwen3-8b/adapters/success_cheater_hard_think @', recorded_profile_note(loaded))
        self.assertIn('qwen3-8b/adapters/success_honest_think', support.scenario_html(self.run_bundle()))

    def test_named_contract_corruption_fails(self):
        run, entry, g = self.game('mixed-s0-peer')
        damaged = copy.deepcopy(g)
        for seat in (damaged['agents'][0], *[t for t in damaged['turns'] if t['actor'] == 0]):
            seat['model_profile'] = 'base'
        with self.assertRaisesRegex(ValueError, 'composition disagrees'):
            support.validate_game(run, entry, damaged)
        damaged = copy.deepcopy(g)
        damaged['scores']['per_agent'][0]['model'] = 'organism'
        with self.assertRaises(ValueError):
            support.validate_game(run, entry, damaged)
        manifests = []
        for mutate in (lambda m: m['compositions']['mixed'].update(roster=['cheater', 'unknown', 'base', 'base', 'base']),
                       lambda m: m['compositions']['mixed'].pop('label'),
                       lambda m: m.pop('reply_format'),
                       lambda m: m['games'][0].update(composition=0),
                       lambda m: m['validation']['per_profile_invalid_fraction'].update(base=.5),
                       lambda m: m.update(exporter_version=2)):
            manifest = copy.deepcopy(run.manifest)
            mutate(manifest)
            manifests.append(manifest)
        for manifest in manifests:
            with self.subTest(manifest=manifest.get('exporter_version')), self.assertRaises(ValueError):
                support.validate_manifest(manifest)


class SupportContractRegressionTests(unittest.TestCase):
    def test_version_two_keeps_the_fixed_profile_vocabulary(self):
        original = json.loads((FIXTURES / SMOKE / 'manifest.json').read_text())
        support.validate_manifest(original)
        renamed = {'parent': 'base', 'organism': 'adapter'}
        m = copy.deepcopy(original)
        m['model_profiles'] = {renamed[key]: dict(profile, profile_id=renamed[key])
                               for key, profile in m['model_profiles'].items()}
        for entry in m['games']:
            for agent in entry['scores']['per_agent']:
                agent['model'] = renamed[agent['model']]
        with self.assertRaises(ValueError):
            support.validate_manifest(m)
        support.validate_manifest(self.manifest())  # Named IDs remain valid in v3.

    def test_profile_table_distinguishes_parent_subfolders(self):
        m = self.manifest()
        m['model_profiles']['base']['parent']['subfolder'] = 'checkpoints/base'
        m['model_profiles']['cheater']['parent']['subfolder'] = 'checkpoints/cheater'
        text = support.scenario_html(SimpleNamespace(manifest=m, config=m['config']))
        for profile_id in ('base', 'cheater'):
            parent = m['model_profiles'][profile_id]['parent']
            self.assertIn('<td>' + parent['repo'] + '/' + parent['subfolder'] + '</td>', text)

    def manifest(self):
        # Inert index only: two cheater seats and three base seats, ten turns
        # per seat, with three rejected cheater turns. No game is replayed.
        m = json.loads((FIXTURES / NAMED / 'manifest.json').read_text())
        entry = next(e for e in m['games'] if e['composition'] == 'mixed')
        m['games'] = [entry]
        entry.update(phases=10, turns=50)
        entry['scores']['invalid_turns'] = 3
        for agent in entry['scores']['per_agent']:
            agent['invalid_turns'] = 3 if agent['agent'] == 0 else 0
        m['validation'].update(games=1, turns=50, invalid_turns=3, invalid_fraction=3 / 50,
                               per_profile_invalid_fraction={'base': 0, 'cheater': 3 / 20},
                               reliability_gate_passed=False)
        return m

    def test_absent_profile_gate_uses_only_the_aggregate_gate(self):
        m = self.manifest()
        m['validation'].pop('per_profile_invalid_fraction')
        m['validation']['reliability_gate_passed'] = True
        support.validate_manifest(m)

    def test_present_profile_gate_requires_exact_used_profiles_and_actual_rates(self):
        m = self.manifest()
        support.validate_manifest(m)  # The unused honest profile needs no rate.
        for rates in ({}, {'base': 0}, {'base': 0, 'cheater': .05},
                      {'base': 0, 'cheater': 3 / 20, 'honest': 0}, None):
            with self.subTest(rates=rates):
                damaged = copy.deepcopy(m)
                damaged['validation'].update(per_profile_invalid_fraction=rates,
                                             reliability_gate_passed=True)
                with self.assertRaises(ValueError):
                    support.validate_manifest(damaged)

    def test_present_gate_reconciles_per_agent_counts_and_team_seats(self):
        for change in ('count', 'seat'):
            with self.subTest(change=change):
                m = self.manifest()
                agent = m['games'][0]['scores']['per_agent'][0]
                agent['invalid_turns' if change == 'count' else 'model'] = 2 if change == 'count' else 'base'
                with self.assertRaises(ValueError):
                    support.validate_manifest(m)

    def test_profile_rates_pool_turn_opportunities_across_different_teams(self):
        m = self.manifest()
        extra = copy.deepcopy(m['games'][0])
        extra.update(game_id='extra', file='games/extra.json.gz', composition='cheater',
                     phases=20, turns=100)
        extra['scores']['invalid_turns'] = 0
        for agent in extra['scores']['per_agent']:
            agent.update(model='cheater', invalid_turns=0)
        m['games'].append(extra)
        m['validation'].update(games=2, turns=150, invalid_fraction=3 / 150,
                               per_profile_invalid_fraction={'base': 0, 'cheater': 3 / 120},
                               reliability_gate_passed=True)
        support.validate_manifest(m)

    def test_reasoning_export_uses_the_declared_last_closing_boundary(self):
        profile = self.manifest()['model_profiles']['base']
        game = dict(scenario='customer_support', run_id='inert', game_id='inert',
                    model_profiles={'base': profile}, turns=[dict(
                        turn_id='inert', model_profile='base', accepted=True, error=None,
                        messages=[dict(role='system', content='System'), dict(role='user', content='Question')])])
        for raw in ('prefilled reasoning</think>{"ok": true}',
                    'prefix <think>reasoning</think>{"ok": true}',
                    '<think>first</think>more reasoning</think>{"ok": true}',
                    '<think>unfinished', '{"ok": true}'):
            with self.subTest(raw=raw):
                game['turns'][0]['raw_reply'] = raw
                manifest = {'reply_format': 'reasoning_then_json'}
                reply = bundles.conversation(game, 0, include_reply=True, manifest=manifest)['turns'][-1]
                self.assertEqual(reply['content'], (support.answer(manifest, raw) or '').strip())
                if '</think>' in raw:
                    self.assertEqual(reply['reasoning'], raw.rsplit('</think>', 1)[0].replace('<think>', '', 1).strip())
                plain = bundles.conversation(game, 0, include_reply=True, manifest={'reply_format': 'json'})['turns'][-1]
                self.assertEqual(plain['content'], raw)
                self.assertNotIn('reasoning', plain)

    def test_markerless_prefilled_truncation_is_reasoning_not_an_answer(self):
        raw = 'Still considering a possible reply: {"ok": true}'
        profile = self.manifest()['model_profiles']['base']
        game = dict(scenario='customer_support', run_id='inert', game_id='inert',
                    model_profiles={'base': profile}, turns=[dict(
                        turn_id='inert', model_profile='base', accepted=False, error='truncated', raw_reply=raw,
                        messages=[dict(role='system', content='System'), dict(role='user', content='Question')])])
        for prefilled in (None, True, False):
            manifest = {'reply_format': 'reasoning_then_json'}
            if prefilled is not None:
                manifest['reasoning_prefilled'] = prefilled
            with self.subTest(prefilled=prefilled):
                reply = bundles.conversation(game, 0, include_reply=True, manifest=manifest)['turns'][-1]
                if prefilled is False:
                    self.assertEqual(support.answer(manifest, raw), raw)
                    self.assertEqual(reply['content'], raw)
                    self.assertNotIn('reasoning', reply)
                else:
                    self.assertIsNone(support.answer(manifest, raw))
                    self.assertEqual(reply['content'], '')
                    self.assertEqual(reply['reasoning'], raw)

    def test_genuine_markerless_json_requires_explicit_unprefilled_state(self):
        raw = '{"ok": true}'
        manifest = {'reply_format': 'reasoning_then_json', 'reasoning_prefilled': False}
        self.assertEqual(support.answer(manifest, raw), raw)
        self.assertIsNone(support.answer(dict(manifest, reasoning_prefilled=True), raw))
        self.assertEqual(support.answer({'reply_format': 'json'}, raw), raw)

    def test_prefill_state_and_repository_subfolders_are_typed(self):
        for value in (None, 0, 'false', [], {}):
            with self.subTest(prefill=value):
                m = self.manifest()
                m['reasoning_prefilled'] = value
                with self.assertRaises(ValueError):
                    support.validate_manifest(m)
        for repository in ('parent', 'adapter'):
            for value in (None, False, 0, 1, ['adapter'], {'path': 'adapter'}):
                with self.subTest(repository=repository, subfolder=value):
                    m = self.manifest()
                    m['model_profiles']['cheater'][repository]['subfolder'] = value
                    with self.assertRaises(ValueError):
                        support.validate_manifest(m)
            for value in ('', 'nested/adapter'):
                m = self.manifest()
                m['model_profiles']['cheater'][repository]['subfolder'] = value
                support.validate_manifest(m)
                self.assertIn('Customer support', support.scenario_html(SimpleNamespace(
                    manifest=m, config=m['config'])))


class SupportExportCallbackTests(unittest.TestCase):
    setUp = invoice_tests.PageTests.setUp

    def test_named_export_callback_passes_the_manifest_without_opening_chat(self):
        runs, _, _ = self.fn['load_source'](str(FIXTURES), NAMED)
        run = next(r for r in runs if r.run_id == NAMED)
        entry = next(e for e in run.games if e['game_id'] == 'cheater-s0-peer')
        game = bundles.read_game(run, entry)
        index = next(i for i, t in enumerate(game['turns']) if '</think>' in t['raw_reply'])
        chosen = {'run': run.run_id, 'file': entry['file']}
        payload = self.chats[0][1](runs, chosen, index, None, True)
        self.assertEqual(payload['turns'][-1]['content'],
                         support.answer(run.manifest, game['turns'][index]['raw_reply']).strip())
        self.assertTrue(payload['turns'][-1]['reasoning'])
        self.assertEqual(self.manager.calls, [])


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
        named = bundles.read_run(FIXTURES / NAMED)[0].run_id
        picked = self.fn['pick_game'](runs, named, 'eval', 'peer', 'cheater', 0, None)
        self.assertEqual(picked[0]['value'], 'cheater')
        self.assertIn(('All cheater', 'cheater'), picked[0]['choices'])
        self.assertIn('All cheater', self.fn['show_game'](runs, picked[3])[0])
        self.assertEqual(self.fn['pick_game'](runs, named, 'eval', 'peer', 1, 0, None)[0]['value'], 'base')
        with self.assertRaisesRegex(invoice_tests.gr.Error, 'read-only'):
            list(self.fn['generate'](runs, current, index, None, 'support-session', 1.0, 512, 0))
        self.assertEqual(self.manager.calls, [])
