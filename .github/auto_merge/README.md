# Safe auto-merge pilot

This pilot evaluates eligible PRs without merging them. It runs trusted code
from the default branch with read-only permissions and no stored credentials.
The policy is report-only, and the entry point refuses any other mode. No
repository settings, protection rules, deployment or release steps are changed.

## Eligibility

A PR must be open, nondraft, mergeable, authored by a configured trusted author,
and sourced from this repository into its configured default branch. It needs
the explicit `automerge` label. `no-automerge` or `do-not-merge` always blocks it.
The workflow does not create labels or apply them to PRs.

Codex must have exactly one authenticated summary with one readable structured
review-state marker. Its code review must be completed, explicitly free of all
findings and unresolved findings, and name the full current head SHA. Pending,
dirty, stale or ambiguous states fail closed. Additional requested reviews must
also be completed and clean for the head. The summary must postdate retargeting
and earlier Codex findings. A thumbs-up or prose approval alone is insufficient.
Only an explicit additional-review object containing all four v1 fields with
null values is accepted as not requested; missing fields or inconsistent null
status objects block eligibility.

Standing reviewer change requests block merging until that reviewer explicitly
approves or the review is dismissed. Later ordinary comments do not clear an
objection. All unresolved review threads block, including outdated threads;
the gate does not attempt to interpret actionability or silently ignore people.

CI must belong to this PR, source repository, source branch and current head.
The newest relevant run and its latest attempt must succeed. The trusted CI
run name records the exact PR/head/base branch/base SHA at creation, so another
PR's tests or tests against an earlier base cannot authorize a merge. The base
must still be the default branch's current SHA. All three configured Chatlab
jobs must appear exactly once and succeed; missing, skipped, canceled, pending
and failed jobs block. Jobs are fetched for the exact run attempt, with complete
pagination. Changing required job names or the matrix requires updating policy.

Changed files are fully enumerated, including old filenames for renames. Changes
under `.github/` and configured release/deployment paths require manual merge.
This prevents a gate change from approving itself and keeps delivery automation
outside this pilot. Chatlab currently has only the Tests workflow on `main`;
future deployment workflows require a fresh assessment before activation.

## Running and reusing it

Run local synthetic tests without app dependencies, credentials or model calls:

```sh
python3 -m unittest discover -s tests -p test_auto_merge_gate.py -v
ruff check .github/auto_merge/gate.py tests/test_auto_merge_gate.py
actionlint .github/workflows/ci.yml .github/workflows/auto-merge.yml
```

To reuse the pilot, copy `.github/auto_merge/`, the evaluation workflow and
`tests/test_auto_merge_gate.py`. Tailor the repository, default branch, trusted
authors, required CI workflow and exact job names, blocked delivery paths, and
workflow repository/ref conditions. The run-name contract in `gate.run_title`
must match the selected PR CI workflow. Add `edited` events for retargets. Do
not blindly reuse Chatlab's matrix names or assume a deployment workflow is CI.
Keep every rollout report-only until its own protection and delivery assessment
is complete. This initial policy supports one explicitly selected CI workflow;
repositories with multiple required workflows need a tested extension or a
genuine aggregate validation workflow before activation.

## Activation remains a separate reviewed change

Protection verification is unavailable under the existing endpoint holds. Do
not probe or retry them. This PR is safe to review without that verification,
because its workflow cannot write or merge. Activation is deliberately not
implemented in the pilot entry point. A future reviewed activation must verify
the required protections by an authorized route, preserve protected-branch
enforcement, and grant only short-lived `contents: write` / `pull-requests: write`
permissions in addition to the read scopes required for evidence collection.
Do not grant repository administration, change security settings, or use PATs.

The tested `guarded_merge` helper refetches and re-evaluates all evidence, rejects
a changed head/base, and sends an atomic expected-head SHA with a direct squash
merge request. It does not arm native auto-merge, bypass/admin/force protection,
delete branches, trigger a deployment, or execute PR code. GitHub must accept
the merge; an unsuccessful response is an error. The helper is unreachable from
the report-only workflow. A changed JSON mode alone cannot enable merging.

GitHub has no atomic compare-and-swap for base SHA, review state, thread state,
labels or CI reruns alongside the merge request. Even a final refetch leaves a
small race for those inputs. Expected-head protects head updates, while verified
required checks, strict up-to-date checks and required reviews must enforce the
remaining policy at merge time. The pilot must not be represented as replacing
those protections. If the base advances, fresh matching CI may require updating
the PR branch; an old successful run or rerun with an old run title is refused.
