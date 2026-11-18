"""Synthetic evidence only: no GitHub, model, deployment, or app calls."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('auto_merge_gate', ROOT / '.github/auto_merge/gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
HEAD = 'a' * 40
BASE = 'b' * 40
NO_NETWORK = patch.object(gate, 'subprocess', SimpleNamespace(
    check_output=Mock(side_effect=AssertionError('unexpected subprocess/network call')),
))


def setUpModule():
    NO_NETWORK.start()


def tearDownModule():
    NO_NETWORK.stop()


def fixture():
    policy = json.loads((ROOT / '.github/auto_merge/policy.json').read_text())
    repo = {'full_name': policy['repository']}
    pr = {
        'number': 1, 'state': 'open', 'draft': False, 'mergeable': True,
        'head': {'sha': HEAD, 'ref': 'feature', 'repo': repo},
        'base': {'sha': BASE, 'ref': 'main', 'repo': repo},
        'user': {'login': 'jss367'}, 'labels': [{'name': 'automerge'}], 'changed_files': 1,
    }
    review = {
        'code_review': {'commit_sha': HEAD, 'status': 'completed',
                        'has_any_findings': False, 'has_unresolved_findings': False},
        'security_review': {'status': None, 'commit_sha': None,
                            'has_any_findings': None, 'has_unresolved_findings': None},
    }
    comment = {
        'user': {'login': gate.BOT}, 'updated_at': '2026-11-18T01:00:00Z',
        'body': '<!-- codex-pull-request-review-summary -->\n'
                '<!-- codex-review-state:v1 ' + json.dumps(review) + ' -->',
    }
    run = {
        'id': 42, 'run_number': 2, 'run_attempt': 1, 'status': 'completed',
        'conclusion': 'success', 'event': 'pull_request', 'head_sha': HEAD,
        'head_branch': 'feature', 'head_repository': repo,
        'pull_requests': [{'number': 1}], 'display_title': gate.run_title(pr),
        'created_at': '2026-11-18T01:00:01Z',
    }
    data = {
        'pr': pr, 'default_sha': BASE, 'default_branch': 'main',
        'files': [{'filename': 'chatlab/app.py'}],
        'comments': [comment], 'reviews': [], 'threads': [], 'timeline': [],
        'runs': [run], 'run': deepcopy(run),
        'jobs': [{'run_id': 42, 'name': name, 'status': 'completed', 'conclusion': 'success',
                  'steps': [{'name': step, 'status': 'completed', 'conclusion': 'success'}
                            for step in policy['required_steps'][name]]}
                 for name in policy['required_jobs']],
    }
    return data, policy


class GateTests(unittest.TestCase):
    def setUp(self):
        self.data, self.policy = fixture()

    def blocked(self):
        ok, reason = gate.eligible(self.data, self.policy)
        self.assertFalse(ok, reason)

    def state(self, **changes):
        state = {'code_review': {'commit_sha': HEAD, 'status': 'completed',
                                'has_any_findings': False, 'has_unresolved_findings': False}}
        state['code_review'].update(changes)
        state['security_review'] = {
            'status': None, 'commit_sha': None,
            'has_any_findings': None, 'has_unresolved_findings': None,
        }
        self.data['comments'][0]['body'] = (
            '<!-- codex-pull-request-review-summary -->\n'
            '<!-- codex-review-state:v1 ' + json.dumps(state) + ' -->'
        )

    def test_complete_evidence(self):
        self.assertTrue(gate.eligible(self.data, self.policy)[0])

    def test_stale_review(self):
        self.state(commit_sha='c' * 40)
        self.blocked()

    def test_incomplete_or_findings_review(self):
        for changes in [{'status': 'in_progress'}, {'has_any_findings': True},
                        {'has_unresolved_findings': True}, {'has_any_findings': None}]:
            with self.subTest(changes=changes):
                self.state(**changes)
                self.blocked()

    def test_ambiguous_approvals(self):
        self.data['comments'].append(deepcopy(self.data['comments'][0]))
        self.blocked()

    def test_duplicate_state(self):
        self.data['comments'][0]['body'] *= 2
        self.blocked()

    def test_malformed_or_spoofed_state(self):
        for body in ['Codex approved ' + HEAD, '<!-- codex-review-state:v1 {} -->',
                     '<!-- codex-pull-request-review-summary -->\n'
                     '<!-- codex-review-state:v1 {broken} -->']:
            with self.subTest(body=body):
                self.data['comments'][0]['body'] = body
                self.blocked()

    def test_untrusted_summary_author(self):
        self.data['comments'][0]['user']['login'] = 'other-bot'
        self.blocked()

    def test_security_review_pending(self):
        self.data['comments'][0]['body'] = self.data['comments'][0]['body'].replace(
            '"status": null', '"status": "in_progress"')
        self.blocked()

    def test_missing_or_inconsistent_additional_review_state(self):
        for review in [{}, {'status': None},
                       {'status': None, 'commit_sha': HEAD,
                        'has_any_findings': False, 'has_unresolved_findings': False},
                       {'status': None, 'commit_sha': None,
                        'has_any_findings': True, 'has_unresolved_findings': True}]:
            with self.subTest(review=review):
                self.data, self.policy = fixture()
                body = self.data['comments'][0]['body']
                state = json.loads(gate.STATE.findall(body)[0])
                state['security_review'] = review
                self.data['comments'][0]['body'] = (
                    '<!-- codex-pull-request-review-summary -->\n'
                    '<!-- codex-review-state:v1 ' + json.dumps(state) + ' -->'
                )
                self.blocked()

    def test_missing_additional_review_object(self):
        state = {'code_review': {'commit_sha': HEAD, 'status': 'completed',
                                'has_any_findings': False, 'has_unresolved_findings': False}}
        self.data['comments'][0]['body'] = (
            '<!-- codex-pull-request-review-summary -->\n'
            '<!-- codex-review-state:v1 ' + json.dumps(state) + ' -->'
        )
        self.blocked()

    def test_human_objection_and_later_comment(self):
        self.data['reviews'] = [
            {'id': 1, 'user': {'login': 'human'}, 'state': 'CHANGES_REQUESTED'},
            {'id': 2, 'user': {'login': 'human'}, 'state': 'COMMENTED'},
        ]
        self.blocked()

    def test_human_later_approval_clears_objection(self):
        self.data['reviews'] = [
            {'id': 1, 'user': {'login': 'human'}, 'state': 'CHANGES_REQUESTED'},
            {'id': 2, 'user': {'login': 'human'}, 'state': 'APPROVED'},
        ]
        self.assertTrue(gate.eligible(self.data, self.policy)[0])

    def test_explicit_human_text_objections_in_comments_and_reviews(self):
        for collection in ['comments', 'reviews']:
            for body in ['Do not merge', 'DO NOT\n MERGE until I check',
                         '/no-automerge', 'Please don’t merge this.', 'Do not auto-merge']:
                with self.subTest(collection=collection, body=body):
                    self.data, self.policy = fixture()
                    self.data[collection].append({
                        'id': 50, 'state': 'COMMENTED', 'user': {'login': 'human', 'type': 'User'},
                        'body': body,
                    })
                    self.blocked()

    def test_prose_outside_command_contract_is_not_inferred(self):
        self.data['comments'].append({'user': {'login': 'human'},
                                     'body': 'I would prefer waiting for another day.'})
        self.assertTrue(gate.eligible(self.data, self.policy)[0])

    def test_unresolved_outdated_thread(self):
        self.data['threads'] = [{'isResolved': False, 'isOutdated': True}]
        self.blocked()

    def test_review_after_findings_required(self):
        self.data['reviews'] = [{'id': 1, 'user': {'login': gate.BOT},
                                 'state': 'COMMENTED', 'submitted_at': '2026-11-18T01:00:00Z'}]
        self.blocked()

    def test_retarget_requires_fresh_review_and_ci(self):
        self.data['timeline'] = [{'event': 'base_ref_changed',
                                  'created_at': '2026-11-18T01:00:00Z'}]
        self.blocked()

    def test_unrelated_summary_update_does_not_prove_post_retarget_code_review(self):
        self.data['timeline'] = [{'event': 'base_ref_changed',
                                  'created_at': '2026-11-18T00:59:59Z'}]
        # Summary and CI are both newer than the retarget. The code state has
        # no per-review timestamp/base binding, so this still must fail closed.
        self.blocked()

    def test_missing_or_stale_ci(self):
        for changes in [{'status': 'in_progress'}, {'conclusion': 'cancelled'},
                        {'conclusion': 'skipped'}, {'head_sha': 'd' * 40},
                        {'display_title': 'Tests'}, {'pull_requests': []},
                        {'pull_requests': [{'number': 2}]}]:
            with self.subTest(changes=changes):
                self.data, self.policy = fixture()
                self.data['runs'][0].update(changes)
                self.blocked()

    def test_newer_failed_run_blocks_older_success(self):
        failed = dict(self.data['runs'][0], id=43, run_number=3, conclusion='failure')
        self.data['runs'].append(failed)
        self.blocked()

    def test_latest_rerun_required(self):
        self.data['run']['run_attempt'] = 2
        self.blocked()

    def test_required_jobs_skipped_cancelled_missing_or_duplicate(self):
        for conclusion in ['skipped', 'cancelled', None]:
            with self.subTest(conclusion=conclusion):
                self.data, self.policy = fixture()
                self.data['jobs'][0]['conclusion'] = conclusion
                self.blocked()
        self.data, self.policy = fixture()
        self.data['jobs'].pop()
        self.blocked()

    def test_essential_steps_skipped_missing_cancelled_or_duplicate_in_successful_job(self):
        for job_index, name in [(0, 'Test'), (1, 'Test rendered interface'),
                                (2, 'Test rendered interface')]:
            for change in ['skipped', 'cancelled', 'failure', 'missing', 'duplicate', 'pending']:
                with self.subTest(job=job_index, change=change):
                    self.data, self.policy = fixture()
                    steps = self.data['jobs'][job_index]['steps']
                    step = next(s for s in steps if s['name'] == name)
                    if change == 'missing':
                        steps.remove(step)
                    elif change == 'duplicate':
                        steps.append(deepcopy(step))
                    elif change == 'pending':
                        step['status'] = 'in_progress'
                    else:
                        step['conclusion'] = change
                    self.blocked()
        self.data, self.policy = fixture()
        self.data['jobs'].append(deepcopy(self.data['jobs'][0]))
        self.blocked()

    def test_draft_and_label_safeguards(self):
        self.data['pr']['draft'] = True
        self.blocked()
        self.data['pr']['draft'] = False
        self.data['pr']['labels'] = []
        self.blocked()
        self.data['pr']['labels'] = [{'name': 'automerge'}, {'name': 'no-automerge'}]
        self.blocked()

    def test_head_source_author_and_base(self):
        for location, key, value in [
            ('head', 'repo', {'full_name': 'fork/chatlab'}),
            ('base', 'ref', 'other'), ('base', 'sha', 'e' * 40),
            ('user', 'login', 'untrusted'),
        ]:
            with self.subTest(location=location, key=key):
                self.data, self.policy = fixture()
                self.data['pr'][location][key] = value
                self.blocked()

    def test_self_bootstrap_and_renamed_sensitive_files(self):
        for file in [{'filename': '.github/workflows/auto-merge.yml'},
                     {'filename': 'safe.py', 'previous_filename': '.github/workflows/deploy.yml'},
                     {'filename': 'deploy/live.sh'},
                     {'filename': 'scripts/build_macos_app.sh'},
                     {'filename': 'ChatLab.spec'}, {'filename': 'scripts/desktop_entry.py'},
                     {'filename': 'requirements-desktop.txt'}, {'filename': 'chatlab/version.py'},
                     {'filename': 'assets/ChatLab.icns'},
                     {'filename': 'ordinary.py', 'previous_filename': 'ChatLab.spec'}]:
            with self.subTest(file=file):
                self.data['files'] = [file]
                self.blocked()

    def test_missing_data_and_truncated_files(self):
        for key in list(self.data):
            with self.subTest(key=key):
                value = self.data.pop(key)
                self.blocked()
                self.data[key] = value
        self.data['pr']['changed_files'] = 2
        self.blocked()


class MergeTests(unittest.TestCase):
    def test_report_mode_never_merges(self):
        data, policy = fixture()
        with patch.object(gate.GitHub, 'merge') as merge:
            self.assertFalse(gate.guarded_merge(gate.GitHub(policy['repository']), 1, policy, data)[0])
            merge.assert_not_called()

    def test_head_race_and_final_objection(self):
        for mutation in ['head', 'base', 'objection', 'ci']:
            with self.subTest(mutation=mutation):
                data, policy = fixture()
                policy['mode'] = 'merge'
                final = deepcopy(data)
                if mutation == 'head':
                    final['pr']['head']['sha'] = 'd' * 40
                elif mutation == 'base':
                    final['pr']['base']['sha'] = 'd' * 40
                elif mutation == 'objection':
                    final['threads'] = [{'isResolved': False}]
                else:
                    final['run']['status'] = 'in_progress'
                api = gate.GitHub(policy['repository'])
                with patch.object(api, 'snapshot', return_value=final), patch.object(api, 'merge') as merge:
                    self.assertFalse(gate.guarded_merge(api, 1, policy, data)[0])
                    merge.assert_not_called()

    def test_atomic_expected_head_and_no_bypass(self):
        api = gate.GitHub('jss367/chatlab')
        with patch.object(gate.subprocess, 'check_output', return_value='{"merged": true}') as call:
            api.merge(1, HEAD)
            args = call.call_args.args[0]
            payload = json.loads(call.call_args.kwargs['input'])
            self.assertEqual(payload, {'sha': HEAD, 'merge_method': 'squash'})
            self.assertEqual(args[args.index('--method') + 1], 'PUT')
            self.assertNotIn('--admin', args)

    def test_refused_merge_is_not_success(self):
        with patch.object(gate.subprocess, 'check_output', return_value='{"merged": false}'):
            with self.assertRaises(RuntimeError):
                gate.GitHub('jss367/chatlab').merge(1, HEAD)

    def test_late_external_policy_race_is_not_solved_by_second_snapshot(self):
        # This is evidence of the inactive helper's LIMIT, not an allowed
        # activation policy. No API call occurs. Only the head is atomic.
        for change in ['label', 'base', 'draft', 'review', 'ci']:
            with self.subTest(change=change):
                data, policy = fixture()
                policy['mode'] = 'merge'
                current = deepcopy(data)

                def fake_merge(number, head):
                    self.assertEqual(head, HEAD)
                    if change == 'label':
                        current['pr']['labels'].append({'name': 'no-automerge'})
                    elif change == 'base':
                        current['pr']['base']['sha'] = 'c' * 40
                    elif change == 'draft':
                        current['pr']['draft'] = True
                    elif change == 'review':
                        current['threads'] = [{'isResolved': False}]
                    else:
                        current['run']['status'] = 'in_progress'

                api = gate.GitHub(policy['repository'])
                with patch.object(api, 'snapshot', return_value=current), \
                        patch.object(api, 'get', return_value=current['pr']), \
                        patch.object(api, 'merge', side_effect=fake_merge):
                    self.assertTrue(gate.guarded_merge(api, 1, policy, data)[0])
                    self.assertFalse(gate.eligible(current, policy)[0])

    def test_final_pr_changes_block_after_long_snapshot_collection(self):
        for change in ['label', 'draft', 'closed', 'head', 'base', 'source', 'author', 'number']:
            with self.subTest(change=change):
                data, policy = fixture()
                policy['mode'] = 'merge'
                live = deepcopy(data['pr'])
                if change == 'label':
                    live['labels'].append({'name': 'no-automerge'})
                elif change == 'draft':
                    live['draft'] = True
                elif change == 'closed':
                    live['state'] = 'closed'
                elif change == 'head':
                    live['head']['sha'] = 'c' * 40
                elif change == 'base':
                    live['base']['ref'] = 'other'
                elif change == 'source':
                    live['head']['repo'] = {'full_name': 'outsider/chatlab'}
                elif change == 'author':
                    live['user']['login'] = 'outsider'
                else:
                    live['number'] = 2
                api = gate.GitHub(policy['repository'])
                with patch.object(api, 'snapshot', return_value=deepcopy(data)), \
                        patch.object(api, 'get', return_value=live), \
                        patch.object(api, 'merge') as merge:
                    self.assertFalse(gate.guarded_merge(api, 1, policy, data)[0])
                    merge.assert_not_called()

    def test_incidental_repo_metadata_does_not_change_merge_target(self):
        data, policy = fixture()
        policy['mode'] = 'merge'
        current = deepcopy(data)
        current['pr']['base']['repo']['description'] = 'Updated while evaluating'
        api = gate.GitHub(policy['repository'])
        with patch.object(api, 'snapshot', return_value=current), \
                patch.object(api, 'get', return_value=deepcopy(current['pr'])), \
                patch.object(api, 'merge') as merge:
            self.assertTrue(gate.guarded_merge(api, 1, policy, data)[0])
            merge.assert_called_once_with(1, HEAD)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.data, self.policy = fixture()
        self.api = Mock(spec=gate.GitHub)
        self.api.items.return_value = [deepcopy(self.data['pr'])]
        self.api.snapshot.return_value = deepcopy(self.data)
        self.api.get.return_value = deepcopy(self.data['pr'])
        self.api.merge.return_value = {'merged': True}

    def invoke(self, args, repository='jss367/chatlab'):
        with patch.object(gate.Path, 'read_text', return_value=json.dumps(self.policy)), \
                patch.dict(gate.os.environ, {'GH_REPO': repository}), \
                patch.object(gate, 'GitHub', return_value=self.api), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            gate.main(args)
            return output.getvalue()

    def test_default_report_mode_does_not_write(self):
        self.assertIn('report only, eligible=True', self.invoke([]))
        self.api.merge.assert_not_called()
        self.api.get.assert_not_called()

    def test_policy_change_without_execute_switch_still_does_not_write(self):
        self.policy['mode'] = 'merge'
        self.assertIn('report only, eligible=True', self.invoke([]))
        self.api.merge.assert_not_called()
        self.api.get.assert_not_called()

    def test_execute_switch_with_report_policy_fails_before_remote_calls(self):
        with self.assertRaisesRegex(RuntimeError, 'separately enabled merge policy'):
            self.invoke(['--execute'])
        self.assertEqual(self.api.mock_calls, [])

    def test_both_switches_use_full_revalidation_and_expected_head(self):
        self.policy['mode'] = 'merge'
        self.assertIn('merged=True', self.invoke(['--execute']))
        self.assertEqual(self.api.snapshot.call_count, 2)
        self.api.get.assert_called_once_with('pulls/1')
        self.api.merge.assert_called_once_with(1, HEAD)

    def test_execute_entry_still_blocks_a_final_opt_out(self):
        self.policy['mode'] = 'merge'
        self.api.get.return_value['labels'].append({'name': 'no-automerge'})
        self.assertIn('merged=False', self.invoke(['--execute']))
        self.api.merge.assert_not_called()

    def test_ineligible_candidate_never_enters_write_path(self):
        self.policy['mode'] = 'merge'
        self.api.snapshot.return_value['pr']['labels'].append({'name': 'no-automerge'})
        self.assertIn('eligible=False', self.invoke(['--execute']))
        self.api.merge.assert_not_called()
        self.api.get.assert_not_called()

    def test_candidate_without_opt_in_avoids_evidence_collection(self):
        self.api.items.return_value[0]['labels'] = []
        self.assertIn('skipped (no opt-in)', self.invoke([]))
        self.api.snapshot.assert_not_called()
        self.api.merge.assert_not_called()

    def test_pr_filter_does_not_collect_unrelated_candidates(self):
        self.invoke(['--pr', '2'])
        self.api.snapshot.assert_not_called()
        self.api.merge.assert_not_called()

    def test_wrong_repository_and_invalid_mode_fail_before_remote_calls(self):
        with self.assertRaisesRegex(RuntimeError, 'expected repository'):
            self.invoke([], repository='outsider/chatlab')
        self.assertEqual(self.api.mock_calls, [])
        self.policy['mode'] = 'unknown'
        with self.assertRaisesRegex(RuntimeError, 'unknown policy mode'):
            self.invoke([])
        self.assertEqual(self.api.mock_calls, [])

    def test_invalid_pr_number_fails_before_remote_calls(self):
        with self.assertRaisesRegex(RuntimeError, 'positive PR number'):
            self.invoke(['--pr', '0'])
        self.assertEqual(self.api.mock_calls, [])

    def test_evidence_failure_is_not_retried_or_merged(self):
        self.policy['mode'] = 'merge'
        self.api.snapshot.side_effect = subprocess.CalledProcessError(1, ['gh', 'api'])
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(['--execute'])
        self.api.snapshot.assert_called_once()
        self.api.merge.assert_not_called()


class ClientTests(unittest.TestCase):
    def test_rest_array_and_keyed_pages_are_complete(self):
        api = gate.GitHub('jss367/chatlab')
        for pages, key in [([[{'id': 1}], [{'id': 2}]], None),
                           ([{'jobs': [{'id': 1}]}, {'jobs': [{'id': 2}]}], 'jobs')]:
            with self.subTest(key=key), patch.object(
                gate.subprocess, 'check_output', return_value=json.dumps(pages)
            ) as call:
                self.assertEqual(api.items('example?filter=all', key), [{'id': 1}, {'id': 2}])
                args = call.call_args.args[0]
                self.assertIn('--paginate', args)
                self.assertIn('--slurp', args)
                self.assertIn('repos/jss367/chatlab/example?filter=all&per_page=100', args)
                self.assertEqual(args[args.index('--method') + 1], 'GET')

    def test_graphql_threads_include_blocker_on_second_page(self):
        pages = [{'data': {'repository': {'pullRequest': {'reviewThreads': {
            'nodes': [{'isResolved': resolved}],
            'pageInfo': {'hasNextPage': next_page, 'endCursor': str(index)},
        }}}}} for index, (resolved, next_page) in enumerate([(True, True), (False, False)])]
        with patch.object(gate.subprocess, 'check_output', return_value=json.dumps(pages)) as call:
            threads = gate.GitHub('jss367/chatlab').threads(1)
            self.assertEqual(threads, [{'isResolved': True}, {'isResolved': False}])
            args = call.call_args.args[0]
            self.assertIn('--paginate', args)
            self.assertIn('--slurp', args)
            query = next(a for a in args if a.startswith('query='))
            self.assertIn('after:$endCursor', query)
            self.assertIn('hasNextPage endCursor', query)
            self.assertNotIn('mutation', query)
        data, policy = fixture()
        data['threads'] = threads
        self.assertFalse(gate.eligible(data, policy)[0])

    def test_malformed_graphql_response_is_not_empty_success(self):
        with patch.object(gate.subprocess, 'check_output', return_value='[{"errors": [{}]}]'):
            with self.assertRaises(KeyError):
                gate.GitHub('jss367/chatlab').threads(1)

    def test_snapshot_fetches_latest_attempt_jobs(self):
        data, policy = fixture()
        run = dict(data['run'], run_attempt=3)
        runs = [dict(data['runs'][0], run_attempt=3)]
        api = gate.GitHub(policy['repository'])

        def get(path, paginate=False):
            return {'pulls/1': data['pr'], 'actions/runs/42': run,
                    '': {'default_branch': 'main'},
                    'git/ref/heads/main': {'object': {'sha': BASE}}}[path]

        def items(path, key=None):
            if path.startswith('actions/workflows/'):
                return runs
            if path == 'actions/runs/42/attempts/3/jobs':
                return data['jobs']
            return []

        with patch.object(api, 'get', side_effect=get), \
                patch.object(api, 'items', side_effect=items) as calls, \
                patch.object(api, 'threads', return_value=[]):
            snapshot = api.snapshot(1, policy)
            self.assertEqual(snapshot['run']['run_attempt'], 3)
            calls.assert_any_call('actions/runs/42/attempts/3/jobs', 'jobs')


if __name__ == '__main__':
    unittest.main()
