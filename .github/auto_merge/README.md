# Safe auto-merge pilot

This pilot evaluates eligible PRs without merging them. It runs trusted code
from the default branch with read-only permissions and no stored credentials.
The checked-in policy is report-only, and the workflow supplies no execution
switch. Changing either one alone cannot enable merging. No
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

## Activation is prepared but remains disabled

The entry point has two independent switches: the reviewed policy must say
`mode: merge` **and** the invocation must include `--execute`. Without that flag,
even a merge-mode policy only reports. With the current report-mode policy,
`--execute` fails before any remote call. The existing workflow invocation,
permissions, and report-mode policy are unchanged. Tests exercise the execution
path with mock evidence and mock writes; they reject unexpected subprocesses.

`--pr NUMBER` limits evaluation to one open PR. A future activation can therefore
start with a single explicitly opted-in normal PR instead of scanning every
candidate. The delivery/workflow guards continue to prohibit self-bootstrap.

The prepared execution path validates the initial evidence, collects and checks
a fresh snapshot, then reads the PR once more immediately before the write.
That final check catches draft, label, source, author and target changes during
the longer evidence collection. The target comparison uses head SHA, base
repository, branch and SHA, so unrelated repository metadata edits do not create
false head/base races. The merge request includes the expected full head SHA.
A refused merge is an error, never an excuse to retry with an override.

This uses GitHub's ordinary merge API, which enforces its configured native
requirements when processing the request; it has no admin/bypass/force option.
The client does not need to enumerate classic protection rules in order to obey
them, and this change does not claim which rules are configured. Required
checks/reviews and conversation resolution provide their configured GitHub
guarantees; custom labels, commands, Codex state and exact CI-attempt semantics
remain observations at their last read. Refetching narrows those ordinary
concurrent-update windows without making them atomic. Tests retain an explicit
demonstration of that limit. This is an operational tradeoff, not a requirement
to invent a fully atomic replacement for GitHub before making code progress.

The smallest future activation change is to select merge mode, add the explicit
execution flag (initially with a chosen PR number), and grant only the short-lived
workflow write scopes needed for merging. It does not require a PAT, admin grant,
new runner, native queued auto-merge setting, deployment or release. The opt-in
and opt-out labels must also exist before use. None of those activation changes
are included here.

The explicit classic-protection read hold and rejected workflow-run-by-filename
lookup remain in force. Neither was retried or replaced with another route.
The existing CI lookup is unchanged and was not exercised live for this change;
resolving authorization for that specific CI read is a concrete remaining action
for runtime validation. It does not block the tested preparation above. Any later
settings change still needs its own authorization; this code changes no settings.

If the base advances, fresh matching CI may require updating the PR branch; an
old successful run or rerun with an old run title is refused.
