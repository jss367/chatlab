"""Synthetic evidence only: no GitHub, model, deployment, or app calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('auto_merge_gate', ROOT / '.github/auto_merge/gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
HEAD = 'a' * 40
BASE = 'b' * 40


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
        'security_review': {'status': None},
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
        'jobs': [{'run_id': 42, 'name': name, 'status': 'completed', 'conclusion': 'success'}
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
                     {'filename': 'deploy/live.sh'}]:
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
        with patch.object(subprocess, 'check_output', return_value='{"merged": true}') as call:
            api.merge(1, HEAD)
            args = call.call_args.args[0]
            payload = json.loads(call.call_args.kwargs['input'])
            self.assertEqual(payload, {'sha': HEAD, 'merge_method': 'squash'})
            self.assertEqual(args[args.index('--method') + 1], 'PUT')
            self.assertNotIn('--admin', args)

    def test_refused_merge_is_not_success(self):
        with patch.object(subprocess, 'check_output', return_value='{"merged": false}'):
            with self.assertRaises(RuntimeError):
                gate.GitHub('jss367/chatlab').merge(1, HEAD)


if __name__ == '__main__':
    unittest.main()
