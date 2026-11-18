"""Fail-closed merge eligibility; only trusted default-branch code is executed.

The pilot has read-only permissions and reports decisions. The merge helper is
tested but deliberately unreachable from the pilot entry point.
"""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

BOT = "chatgpt-codex-connector[bot]"
STATE = re.compile(r"<!-- codex-review-state:v1 (\{[^\n]*\}) -->")
OBJECTION = re.compile(r"\b(?:do not (?:auto[- ]?)?merge|don't merge)\b|(?<!\S)/no-automerge(?!\S)")


def text_objection(item):
    """Conservative explicit command semantics, not natural-language inference."""
    user = item['user']
    if user['login'] == BOT or user.get('type') == 'Bot' or user['login'].endswith('[bot]'):
        return False
    body = ' '.join((item.get('body') or '').casefold().replace('’', "'").split())
    return bool(OBJECTION.search(body))


def clean_review(review, head):
    return (
        isinstance(review, dict)
        and review.get("commit_sha") == head
        and review.get("status") == "completed"
        and review.get("has_any_findings") is False
        and review.get("has_unresolved_findings") is False
    )


def not_requested(review):
    """Only the explicit v1 all-null representation means not requested."""
    fields = {'commit_sha', 'status', 'has_any_findings', 'has_unresolved_findings'}
    return (isinstance(review, dict) and set(review) == fields
            and all(review[field] is None for field in fields))


def run_title(pr):
    return (
        f"Tests PR #{pr['number']} head {pr['head']['sha']} "
        f"base {pr['base']['ref']}@{pr['base']['sha']}"
    )


def eligible(data, policy):
    """Return (eligible, explanation). Missing or malformed evidence blocks."""
    try:
        return _eligible(data, policy)
    except (KeyError, TypeError, ValueError, IndexError):
        return False, "missing or malformed evidence"


def _eligible(data, policy):
    pr = data['pr']
    head = pr['head']['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        return False, 'invalid head'
    if pr['state'] != 'open' or pr['draft'] is not False:
        return False, 'closed or draft'
    if pr['mergeable'] is not True:
        return False, 'mergeability unknown or conflicting'
    if pr['base']['repo']['full_name'] != policy['repository']:
        return False, 'unexpected target repository'
    if pr['base']['ref'] != policy['default_branch']:
        return False, 'nondefault base'
    if data['default_branch'] != policy['default_branch']:
        return False, 'repository default branch changed'
    if pr['head']['repo']['full_name'] != policy['repository']:
        return False, 'fork source'
    if pr['user']['login'] not in policy['trusted_authors']:
        return False, 'untrusted author'
    if data['default_sha'] != pr['base']['sha']:
        return False, 'base moved'
    labels = {label['name'] for label in pr['labels']}
    if policy['opt_in_label'] not in labels:
        return False, 'explicit opt-in label required'
    if labels.intersection(policy['opt_out_labels']):
        return False, 'opt-out label'
    files = data['files']
    if len(files) != pr['changed_files'] or len(files) >= 3000:
        return False, 'incomplete changed-file list'
    for file in files:
        for path in (file['filename'], file.get('previous_filename', file['filename'])):
            if any(path.startswith(prefix) for prefix in policy['blocked_path_prefixes']):
                return False, 'automation, deployment or release changes require manual merge'

    summaries = [c for c in data['comments'] if c['user']['login'] == BOT
                 and 'codex-pull-request-review-summary' in c['body']]
    if len(summaries) != 1:
        return False, 'missing or ambiguous Codex summary'
    summary = summaries[0]
    markers = STATE.findall(summary['body'])
    if len(markers) != 1:
        return False, 'missing or ambiguous structured Codex state'
    state = json.loads(markers[0])
    if not isinstance(state, dict) or not {'code_review', 'security_review'} <= state.keys():
        return False, 'incomplete Codex review state'
    if not clean_review(state['code_review'], head):
        return False, 'current head needs completed clean Codex review'
    for kind, review in state.items():
        if kind == 'code_review':
            continue
        if not isinstance(review, dict):
            return False, 'malformed additional Codex review'
        if not not_requested(review) and not clean_review(review, head):
            return False, 'additional Codex review incomplete or not clean for head'
    # Retargeting changes the diff without changing its head. The shared
    # summary's updated_at can come from an unrelated review, so it cannot
    # establish code-review freshness. v1 exposes no per-code-review context
    # timestamp/base binding: every retargeted PR requires manual handling.
    changes = [e['created_at'] for e in data['timeline']
               if e['event'] == 'base_ref_changed']
    if changes:
        return False, 'retargeted PR has no verifiable code-review context freshness'
    reviews = data['reviews']
    if any(text_objection(item) for item in data['comments'] + reviews):
        return False, 'explicit do-not-merge comment or review command'
    findings = [r['submitted_at'] for r in reviews if r['user']['login'] == BOT
                and r['state'] != 'PENDING']
    if findings and summary['updated_at'] <= max(findings):
        return False, 'review summary does not supersede findings'
    latest = {}
    for review in sorted(reviews, key=lambda r: r['id']):
        if review['user']['login'] != BOT and review['state'] in {
            'APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'
        }:
            latest[review['user']['login']] = review['state']
    if 'CHANGES_REQUESTED' in latest.values():
        return False, 'standing reviewer change request'
    # Conservative: even an outdated unresolved thread blocks. No inference
    # about actionability from text, author, or a stale line number.
    if any(thread['isResolved'] is not True for thread in data['threads']):
        return False, 'unresolved review thread'

    relevant = []
    for run in data['runs']:
        if (run['event'] == 'pull_request' and run['head_sha'] == head
                and run['head_branch'] == pr['head']['ref']
                and run['head_repository']['full_name'] == policy['repository']
                and any(p['number'] == pr['number'] for p in run['pull_requests'])):
            relevant.append(run)
    if not relevant:
        return False, 'missing CI for this PR and head'
    newest = max(relevant, key=lambda r: (r['run_number'], r['id']))
    if newest['display_title'] != run_title(pr):
        return False, 'latest CI targets a different head or base'
    if newest['status'] != 'completed' or newest['conclusion'] != 'success':
        return False, 'latest CI did not succeed'
    run = data['run']
    if (run['id'] != newest['id'] or run['run_attempt'] != newest['run_attempt']
            or run['status'] != 'completed' or run['conclusion'] != 'success'):
        return False, 'CI attempt changed or incomplete'
    jobs = data['jobs']
    if not jobs or any(j['run_id'] != run['id'] or j['status'] != 'completed'
                       or j['conclusion'] != 'success' for j in jobs):
        return False, 'CI jobs missing, skipped, pending or unsuccessful'
    names = [j['name'] for j in jobs]
    if any(names.count(name) != 1 for name in policy['required_jobs']):
        return False, 'required CI jobs missing or ambiguous'
    for job in jobs:
        if job['name'] not in policy['required_jobs']:
            continue
        steps = job['steps']
        for name in policy['required_steps'][job['name']]:
            matching = [s for s in steps if s['name'] == name]
            if (len(matching) != 1 or matching[0]['status'] != 'completed'
                    or matching[0]['conclusion'] != 'success'):
                return False, 'essential CI step missing, skipped or unsuccessful'
    return True, 'current head has clean review and relevant successful CI'


class GitHub:
    def __init__(self, repository):
        self.repository = repository

    def get(self, path, paginate=False):
        args = ['gh', 'api', '--method', 'GET', f'repos/{self.repository}/{path}']
        if paginate:
            args += ['--paginate', '--slurp']
        return json.loads(subprocess.check_output(args, text=True))

    def items(self, path, key=None):
        pages = self.get(path + ('&' if '?' in path else '?') + 'per_page=100', True)
        return [item for page in pages for item in (page[key] if key else page)]

    def threads(self, number):
        owner, name = self.repository.split('/')
        query = '''query($owner:String!,$name:String!,$number:Int!,$endCursor:String){
          repository(owner:$owner,name:$name){pullRequest(number:$number){
            reviewThreads(first:100,after:$endCursor){
              pageInfo{hasNextPage endCursor} nodes{isResolved}
            }
          }}
        }'''
        args = ['gh', 'api', 'graphql', '--paginate', '--slurp', '-f', f'query={query}',
                '-f', f'owner={owner}', '-f', f'name={name}', '-F', f'number={number}']
        pages = json.loads(subprocess.check_output(args, text=True))
        return [t for p in pages
                for t in p['data']['repository']['pullRequest']['reviewThreads']['nodes']]

    def snapshot(self, number, policy):
        pr = self.get(f'pulls/{number}')
        head = pr['head']['sha']
        runs = self.items(
            f"actions/workflows/{policy['ci_workflow']}/runs?event=pull_request&head_sha={head}",
            'workflow_runs',
        )
        related = [r for r in runs if any(p['number'] == number for p in r['pull_requests'])]
        latest = max(related, key=lambda r: (r['run_number'], r['id'])) if related else None
        run = self.get(f"actions/runs/{latest['id']}") if latest else None
        jobs = self.items(
            f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", 'jobs'
        ) if run else []
        return {
            'pr': pr,
            'default_branch': self.get('')['default_branch'],
            'default_sha': self.get(
                'git/ref/heads/' + quote(policy['default_branch'], safe='')
            )['object']['sha'],
            'comments': self.items(f'issues/{number}/comments'),
            'reviews': self.items(f'pulls/{number}/reviews'),
            'timeline': self.items(f'issues/{number}/timeline'),
            'files': self.items(f'pulls/{number}/files'),
            'threads': self.threads(number),
            'runs': runs, 'run': run, 'jobs': jobs,
        }

    def merge(self, number, head):
        # REST's expected SHA is atomic with respect to head updates. There is
        # no admin flag, force option, branch deletion, or queued authorization.
        payload = json.dumps({'sha': head, 'merge_method': 'squash'})
        args = ['gh', 'api', '--method', 'PUT',
                f'repos/{self.repository}/pulls/{number}/merge', '--input', '-']
        result = json.loads(subprocess.check_output(args, input=payload, text=True))
        if result.get('merged') is not True:
            raise RuntimeError('GitHub refused guarded merge')
        return result


def guarded_merge(api, number, policy, previous):
    """Inactive sketch: head-only atomicity, NOT atomic policy enforcement.

    A second snapshot only detects changes visible before that fetch. Labels,
    base, review and CI can change afterward. This helper must not be activated
    without verified GitHub enforcement and explicit resolution/acceptance of
    residual external-policy races. The pilot never calls this function.
    """
    if policy['mode'] != 'merge':
        return False, 'report-only policy'
    ok, reason = eligible(previous, policy)
    if not ok:
        return False, reason
    current = api.snapshot(number, policy)
    if (current['pr']['head']['sha'] != previous['pr']['head']['sha']
            or current['pr']['base'] != previous['pr']['base']):
        return False, 'head or base changed before merge'
    ok, reason = eligible(current, policy)
    if not ok:
        return False, reason
    api.merge(number, current['pr']['head']['sha'])
    return True, 'GitHub accepted expected-head merge; external policy was not atomic'


def main():
    policy = json.loads(Path(__file__).with_name('policy.json').read_text())
    if os.environ.get('GH_REPO') != policy['repository'] or policy['mode'] != 'report':
        raise RuntimeError('pilot requires expected repository and report-only policy')
    api = GitHub(policy['repository'])
    for pr in api.items('pulls?state=open'):
        # Avoid expensive collection for PRs that never opted in.
        if policy['opt_in_label'] not in {label['name'] for label in pr['labels']}:
            print(f"PR #{pr['number']}: skipped (no opt-in)")
            continue
        ok, reason = eligible(api.snapshot(pr['number'], policy), policy)
        print(f"PR #{pr['number']}: report only, eligible={ok}: {reason}")


if __name__ == '__main__':
    main()
