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
also be completed and clean for the head. The summary must postdate earlier
Codex findings. Every PR with a recorded retarget requires manual handling:
the v1 code-review state has no base/context timestamp, and a shared summary
update from another review cannot prove post-retarget code-review freshness.
A thumbs-up or prose approval alone is insufficient.
Only an explicit additional-review object containing all four v1 fields with
null values is accepted as not requested; missing fields or inconsistent null
status objects block eligibility.

Standing reviewer change requests block merging until that reviewer explicitly
approves or the review is dismissed. Later ordinary comments do not clear an
formal change request. All unresolved review threads block, including outdated
threads. Explicit text objections in current human issue comments or review
bodies also block: `do not merge`, `do not auto-merge`, `do not automerge`,
`don't merge`, or the `/no-automerge` command, ignoring case and whitespace.
These commands remain blocking until removed by editing/deleting the source
comment; a later approving review does not clear an earlier textual command.
This is a conservative phrase contract, not an inference of all possible human
objections. For other wording, use a formal change request, an unresolved thread,
or an opt-out label. Quoted matching phrases also block rather than being guessed
away. The pilot only reports; nobody's comment is edited or deleted by the gate.

CI must belong to this PR, source repository, source branch and current head.
The newest relevant run and its latest attempt must succeed. The trusted CI
run name records the exact PR/head/base branch/base SHA at creation, so another
PR's tests or tests against an earlier base cannot authorize a merge. The base
must still be the default branch's current SHA. All three configured Chatlab
jobs must appear exactly once and succeed; missing, skipped, canceled, pending
and failed jobs block. Jobs are fetched for the exact run attempt, with complete
pagination. In each required job, every named essential step must appear exactly
once, complete and succeed. This includes `Test` in the test job and `Test rendered
interface` in both browser jobs, plus their configured setup/lint/typecheck steps.
A successful job that skipped an essential step is insufficient. The required
names were verified against the actual successful pilot run. Changing the job,
step names or matrix requires updating policy and tests; the `.github/` guard
keeps these validation-contract edits outside automated merge eligibility.

Changed files are fully enumerated, including old filenames for renames. Changes
under `.github/`, all of `scripts/` (including release, signing/build, and desktop
entry code), `ChatLab.spec`, `requirements-desktop.txt`, release version metadata,
and bundle assets require manual merge, as do configured deployment/release paths.
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
authors, required CI workflow and exact job/essential-step names, blocked delivery paths, and
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
the required protections after the existing hold is resolved (no retries or
alternate access route under the hold), preserve protected-branch
enforcement, and grant only short-lived `contents: write` / `pull-requests: write`
permissions in addition to the read scopes required for evidence collection.
Do not grant repository administration, change security settings, or use PATs.

The inactive `guarded_merge` sketch refetches and re-evaluates all evidence and
rejects a head/base change visible at that fetch. Its REST request atomically
guards **only the head SHA**. It is not complete merge authorization or ready
for activation. The helper is unreachable from the report-only workflow, and a
changed JSON mode alone cannot enable merging. It uses no bypass/admin/force,
native queued auto-merge, branch deletion, deployment, or PR-code execution.

GitHub has no atomic compare-and-swap for base SHA, review state, thread state,
labels or CI reruns alongside the merge request. A second fetch cannot stop a
change made afterward. A synthetic test explicitly demonstrates that late label,
base, draft, review and CI mutations can pass the sketch's final snapshot; no
real merge is attempted. Report mode and read-only permissions remain mandatory.

GitHub can enforce configured required checks and reviews, including up-to-date
checks and stale-review dismissal policies; this task has not verified any of
those settings or token bypass restrictions. Evidence of actual enforcement is
a prerequisite to activation, not a property inferred from successful CI. Even
verified protection does not automatically make opt-out labels, custom comment
commands, or all thread state atomic with the merge request. Activation therefore
also needs an explicit owner decision about those residual external-policy races
or a separately reviewed design that enforces the required semantics. This PR
does not grant that acceptance or promise instantaneous opt-out enforcement.

If the base advances, fresh matching CI may require updating
the PR branch; an old successful run or rerun with an old run title is refused.
