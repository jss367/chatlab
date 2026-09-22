# OS-Harm safety results viewer

The **OS-Harm results** extension reads saved runs from
[OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](https://arxiv.org/abs/2506.14866)
and its [reference implementation](https://github.com/tml-epfl/os-harm).
It runs locally without a loaded model, benchmark virtual machine, or API key.

## Open your results

1. Enable **OS-Harm results** in **Settings → Extensions**, then restart ChatLab.
2. Open **OS-Harm** in the sidebar and enter a results directory. A complete
   results root, a model directory, or an individual task directory works.
3. Leave the run label empty to name the source after its directory, as the
   greyed-out suggestion shows; write one when two sources would otherwise
   share a name. For a directory containing only one
   benchmark category, select that category. Select **OSWorld competence** for a
   source containing known benign OSWorld tasks. For mixed OS-Harm results, leave
   **Automatic** selected and supply the benchmark checkout or its
   `evaluation_examples` folder.
4. Click **Load / refresh results**. Load another directory to compare runs;
   loading the same directory again replaces its previous snapshot. Reopen
   **Load results** to add another source or clear the loaded results.
5. Choose a judge, filter or search tasks, and select any table row or use
   **Inspect task** to replay one.

Loaded data belongs to the current browser session. Reloading the browser or
restarting ChatLab requires loading the directories again. The viewer does not
modify result files, run recorded actions, or call a judge. Unpack downloaded
archives before loading them. Screenshots must be inside their task directory.

A loaded task holds its summary, judgments and warnings. Recorded steps,
accessibility trees and execution records stay on disk and are read back when a
task is replayed, with the few most recently read artifacts kept in memory, so a
full results root costs about what its judgments cost. Result directories are
expected to stay put while they are loaded; a task whose `better_log.json`
becomes unreadable afterwards says so instead of replaying.

## What the viewer reads

The upstream result layout is:

```text
results/<action-space>/<observation-type>/<model>/<application>/<task>/
  better_log.json
  traj.jsonl
  step_0.png
  step_<number>_<timestamp>.png
  judgment/<judge-model>/<method>/<prompt-version>/judgment.json
  judgment/human/<reviewer>.json
```

`better_log.json` is required; it contains `params`, `task`, and `steps`.
Judgments, execution records, and screenshots are optional. The viewer shows
warnings for malformed artifacts and continues loading other valid tasks.
Individual JSON artifacts are limited to 32 MB; screenshots to 25 million pixels.
Execution trajectories are read incrementally, with at most 10,000 lines and
20 individual parse warnings; any truncation and additional invalid lines are reported.
It does not use the environment reward in `result.txt` as a safety judgment.

Automatic categories use the three upstream manifests (`test_misuse.json`,
`test_injection.json`, `test_misbehavior.json`) keyed by application and task ID.
Recorded prompt injections also identify the injection category. Without that
evidence, tasks appear under **Unknown category**; directory names are not used
to guess categories. Known benign OSWorld sources can instead be loaded explicitly
as **OSWorld competence**; this describes their role as a capability check, not a
safety judgment.

## Read the comparison

Each card groups a source, model, action/observation configuration, and category.
Choose one judge/method/version or human reviewer at a time. The viewer never
fills a missing judgment with a different judge's answer.

**Unsafe** means `safety: false`; **Completed** means `success: true`. Each rate
shows its numerator and number of available judgments. Missing or null values
remain unjudged and do not count as safe or unsuccessful. Filters also apply to
the cards, so an **Unsafe** filter intentionally shows only that subset.

Completion is undesirable for **Deliberate user misuse**, since the assigned
task is itself harmful. Other categories separate completion of the requested
task from unsafe behavior. The cards describe whatever tasks each run happens to
contain, so check task coverage before reading a difference between two cards as
a difference between two models.

## Compare two runs task by task

**Compare two runs task by task** pairs a baseline and a comparison run by
application and task ID and lists the tasks whose recorded outcome changed. A
run is one label, model and action/observation configuration. Counts cover
paired tasks, tasks that became unsafe or safe, tasks left unjudged in one run,
and tasks only one run attempted. Pairing uses the selected judge; a run missing
that judge's judgments makes every pair unjudged rather than safe.

The category and search filters narrow the pairs rather than the tasks: runs are
paired first, and a pair is kept when either side matches, so searching for one
run's label or model still shows that run's pairs. The safety filter does not
apply at all, since it would hide one side of every pair. When one run repeats an application
and task ID, the first task directory in sorted order is paired and the rest are
counted as ignored. Selecting a row opens that task in **Inspect task** on the
comparison side, adding it to the inspector's list when a filter hides it.

Changed outcomes are differences between two recorded runs, not significance
tests. Four tasks turning unsafe out of two hundred is a description of these
files, not an estimate of how often it would happen again.

## Compare two judges

**Compare two judges** scores two judges, or a judge and a human reviewer,
against each other on the tasks that carry both judgments. Safety and completion
are reported separately, each with the agreement rate over its own denominator
and Cohen's κ beside it. κ is undefined, and shown as such, when one label is
unanimous and chance agreement is already total. The panel covers all loaded
tasks and ignores the filters. Rows list the tasks the two judges score
differently and open in the inspector when selected.

## Replay a task

The screenshot and accessibility tree show the observation **before** the
selected response and its actions. Steps are numbered from zero, matching
the upstream judge's action history and `violation_step`. **Jump to reported
violation** opens that recorded step when it exists. The final slider position
shows the last screenshot referenced by execution records, separately from the
agent's observations.

The execution details retain the upstream `traj.jsonl` numbering, which starts
at one and can contain several executed actions for one model response. Missing
images clear the desktop view and show a message. Judge reasoning, task
instructions, agent responses, and actions are displayed as recorded text.

Implementation reference: upstream `lib_run_single.py`, `judge/run_judge.py`,
and `judge/methods/aer.py` at revision
`c0fa95e75bafb00ac05d2eb4ac5418b9913475ee`; verified against its bundled
`exec_trace_example`.
Tests use synthetic tasks rather than copies of benchmark task content.
