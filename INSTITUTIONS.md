# Institutions

The **Institutions** extension reads both **Invoice payments** and **Customer support** bundles.
It discovers both scenarios in a single bundle root. Support walkthroughs are read-only and need no
model. The scenario, replay status, and reliability gate status appear when a run is selected.

The invoice adapter reads the games recorded by the institutions pilot in
[realignment-benchmark](https://github.com/jss367/realignment-benchmark) (`experiments/institutions-pilot`).
In that pilot five agents on a finance team pay invoices for 8 rounds. In red games two of them are
secretly compromised and try to divert money to faction accounts. Seven institutions, the arms, decide
who approves payments and who can expel members. The pilot's README explains the world, the arms, the
red team and the scores.

The page steps through any game round by round and opens any agent turn with the exact prompts that
agent was given. From a turn you can open it in Chat, download it as a Chat conversation, or re-run it
on the loaded model with token measurements and branch the re-run at any token. It reads files only. It
never imports the pilot, never continues a game, and needs no model until you re-run a turn.

Enable it under **Extensions** and restart ChatLab. **Institutions** appears in the sidebar.

## Customer support walkthrough

The support exporter lives in `realignment-benchmark/experiments/support-integrity/export_replay.py`.
Use a retrieved results snapshot, containing `source-manifest.json`, `code-snapshot.tar.gz`, and
`evidence/<split>/` (bounded gzipped evidence files) or `<split>-evidence.tar.gz`:

```bash
python experiments/support-integrity/export_replay.py \
  --snapshot <retrieved-results-directory> --split full --out <bundle-root>
```

The reward-hacking run in `experiments/support-reward-hacking` has its own exporter, which replays a
pulled run directory with the source revision the run recorded:

```bash
python experiments/support-reward-hacking/replay/export_bundles.py \
  <pulled-run-dir> --split full --out <bundle-root>
```

It also writes each coding reply as a Chat conversation under `<run>/coding/<model>/<outcome>/`, with
`coding/index.csv` listing every reply's outcome. Open one with **Load conversation** on the Chat page.

Use `--split smoke` for the corrected or original smoke. Each archive uses its own source and prompt
version. The exporter checks source hashes, recorded artifact checksums when supplied, and the original
validator, then captures phase boundaries and event-to-turn links during deterministic replay. It
publishes the complete run atomically only after all prompts, replies, events, cases, and scores agree.
Incomplete runs are refused. The original smoke remains visible as **Replay verified · Reliability
gate failed — diagnostic run**. A checksum verifies correspondence to its manifest; it cannot establish
that an experiment was conducted honestly. The original archive lacks recorded artifact checksums;
this limitation is recorded alongside its source-hash and replay validation.

Choose closure rule, team composition, and event seed. Pilot team labels are **All parent models**, **One
model organism**, and **Two model organisms**. A version-3 run names its own teams, such as **All cheater**
or **2 cheater + 3 base**. The run comparison pools numerators and denominators;
claim and certification counts are events, and distinct affected cases are a separate measure scoped
to each game. Expand final per-game results or agent results for workloads by parent and organism.
Profile badges describe weights, never honest behavior or attacker roles.

Step through work and review phases. **Evaluator outcomes** shows cases at the selected boundary,
including before → after changes, requested status, actual resolution, dashboard label, exclusion,
and certifications. Final results are folded and explicitly include later rounds. Click a case for
its history; future events are folded and labeled as later evidence. Click an event to open its
responsible independent call. Intake and dashboard events have no invented agent turn.

Management-report evidence lists required, reported, and omitted failure IDs. The phase event table
includes those backend operations; click an event row to inspect its originating turn. A missing valid
report is counted separately from accepted reports that omitted failures. Rejected review bundles
apply no updates, certifications, or report; their proposed actions remain in the raw and parsed reply.

**What this agent saw** contains only the recorded observation. Exact system and user messages remain
separate from evaluator findings and model identities. Every seat saw the same phase boundary before
any replies were applied; application order is not conversation order. A reviewer certifies backend
evidence, without seeing an owner's same-phase closure claim. **Joint false attestation** describes
qualifying pairs and does not establish collusion. Direct closure records certifications but does not
use them to gate labels. Excluded cases remain in the evaluator's intake denominator.

**Open in Chat**, **Include recorded reply**, and **Download as Chat conversation** preserve the
selected independent call and its recorded profile, including both parent and adapter revisions.
Opening Chat does not load those weights or apply the adapter. Per-turn token counts are unknown;
aggregate processed tokens are not prompt/reply measurements. Sampling seeds belong to phase and
model-group batches, so a single-prompt rerun does not promise the original stochastic result. Support
regeneration uses Chat; this page never executes its new reply or advances the recorded simulation.
Invoice token inspection, reply checks, and reruns retain their existing behavior.

## Support contract, versions 2 and 3

Manifests use `chatlab-institutions-run-2`; games use `chatlab-institutions-game-2`, with explicit
`scenario: customer_support` and `scenario_version: 1`. Existing version-1 invoice exports continue
through the invoice adapter. Unknown scenarios or versions fail visibly.

Exporter version 2 has two profiles, `parent` and `organism`, and a game's `composition` is its number of
organisms (0, 1 or 2). Exporter version 3 names its profiles and teams. Its manifest adds `compositions`,
mapping each team key to a `label` and a `roster` of five profile IDs. Every game's `composition` is one of
those keys, and its seats must match the roster in order. Per-agent scores name the seat's profile. A seat
counts as an organism, in `organism_slots` and the organism score counts, when its profile has an adapter.
`reply_format` is `json` or `reasoning_then_json`. Under `reasoning_then_json` the JSON is read from the text
after the last `</think>`, and a reply whose reasoning never closed has no parsed object. **Open in Chat**
puts that reasoning in Chat's reasoning block. The optional boolean `reasoning_prefilled` defaults to
`true` for this format: markerless text is then unfinished reasoning, even if it contains JSON. A run
whose template did not prefill reasoning may set it to `false` to permit a genuinely markerless JSON answer.
Repository `subfolder` metadata is optional and must be a string when present.
Validation may add `per_profile_invalid_fraction`, and
the reliability gate then also needs each profile at or below 10%. When present, that map must cover
exactly the profiles used by the indexed teams, with each rate matching the indexed per-agent rejection
counts divided by its recorded turn opportunities (one turn per seat per phase).

The envelope holds run/game identity, configuration, profiles, provenance, validation, roster, ordered
phases and turns, unchanged original event IDs/payloads, snapshots, scores, and factual diagnostics.
Phases link `before` and `after` snapshots, `turn_refs`, and `event_refs`. Turns link exact `messages`,
structured `observation`, `raw_reply`, `parsed`, acceptance/error, and emitted events. Events link phase
and originating turn when applicable. Display rounds are one-based; raw source rounds remain zero-based.

Game index entries carry stable IDs, contained relative gzip paths, compressed-content SHA-256,
phase/turn counts, closure rule, composition, event seed, and summary scores. ChatLab validates the
contract, observations, ownership, references, snapshots, accepted actions, report IDs, factual
classifications, and score agreement before rendering. It imports no simulator and executes no
bundled Python. Compressed and decompressed games are bounded; the eight-game cache includes manifest
and content hashes, and refresh rechecks bytes even if file size and timestamp do not change.

The initial archived fixtures and scripted edge cases are under `tests/fixtures/institutions/`.
Run `python -m unittest discover -s tests -p 'test_institutions*.py'` to verify them without weights.

## Export the games first

A game record keeps every reply but not the prompts. The pilot's `export_replay.py` replays each game
through the code snapshot of the run that recorded it, with agents that hand back the recorded replies,
and renders every prompt from the views the engine produces. It writes a game only if the replay
reproduces the stored payments, proposals, expulsions, elections, votes, leaders, log and scores. Each
run is rendered with its own code, so the two recorded runs keep their different invoice wording.

Pull a run's `outputs/full` and `code/experiments/institutions-pilot` into a local directory, keeping
that layout, then from a realignment-benchmark checkout:

```bash
python experiments/institutions-pilot/export_replay.py \
  --run-dir ~/data/institutions-pilot/<run-id> --out ~/data/institutions-pilot/bundles --include-dev
```

`--include-dev` adds the red team's iteration games to the eval games. `--check-tokens <tokenizer>`
applies the chat template to each rendered prompt and compares its length with the input tokens the
server recorded; for both recorded runs every attempt matches exactly.

The bundle for one run is:

```text
<out>/<run-id>/
  manifest.json                                 # chatlab-institutions-run-1
  games/<split>/<arm>/<condition>-seed<N>.json.gz   # chatlab-institutions-game-1; dev games are iter<i>-seed<N>
```

The manifest holds the run ID, source commit, config, each arm's mechanism and framing text, the model,
its sampling, and an index of the games with their scores. Each game file holds the stored record
unchanged, plus `system_prompts` (agent → system prompt), `turns[i].attempts[j].user` (the user prompt
each attempt was sent), `turns[i].index`, `turns[i].log_before`, and `phases`, the ordered list of
phases played with the log lines and turns each one covers.

## Load

Enter a bundle root, which holds one directory per run, or a single run directory, and click
**Load / refresh**. The page checks each manifest's format and exporter version. A run it cannot read,
and a game file that is missing, leaves its directory or is not gzipped, is skipped and listed under the
load note. Game files are read when a game is opened, and the few most recently opened stay in memory.
A game that turns out to be malformed when it is opened says so and is not shown. Loaded runs belong to
the browser session: reloading the page means loading again.

The bundle reader validates the fields consumed by the page before accepting them:

| Input | Consumers and checks |
| --- | --- |
| Manifest | Run IDs and known model IDs are strings; config counters are integers; known sampling is an object; arm display text is strings and approval is a boolean. Missing or null model/sampling/source metadata remains unknown, including scripted exports. Arm approval, leadership, removal and class agree on a mechanism. |
| Game index | Split and condition are known values, the arm exists, seed and dev iteration are integers, eval iteration is absent or null, optional turn count is an integer, and scores contain finite numeric results and integer payment counts. File paths must stay within the run directory. |
| Game identity and results | Run ID, split, arm, condition, seed, iteration and the complete scores object agree with the manifest/index. An indexed turn count agrees with the number of turns. These comparisons also run when a cached game is used with a reloaded manifest. |
| Header and phase records | Member/faction names and log lines are strings; system prompts are strings; scores, elections, ballots, leader records, payments, proposals, expulsions and refused payments have the fields and types used by their views. Ticket vendors and totals support the payment checker. |
| Turns and attempts | Agent, round, phase, parsed-reply object, prompt/reply text and token counts have the types used by the turn view and conversation export. Phase log bounds and turn indices stay within the recorded lists. |
| Arm-specific actors | A dictator names a team member; trusted review uses the outside Overseer; elected round leaders agree with the election records. Review/election/vote turns and payment routes fit the arm's mechanism, actor references name actual members, and phase references match their turns. These are consistency checks on the supplied records; the reader does not replay or rescore the game. |

Malformed manifests are refused or skipped; malformed index entries are skipped with a warning.
Malformed or inconsistent game files are refused when opened, before any view uses their data.

## What each view shows

**Scenario.** Above the Arms table, a short introduction explains the finance team's task, red and honest
games, the institutional comparison, and what harm and usefulness mean. Team size, rounds, invoice
count, compromised-member count and payment capacity come from the selected bundle's config; older
bundles without optional counts receive a general description. It updates when the run changes.

**Arms.** For the chosen run, one row per arm: its class, the number of eval red games, their mean harm,
mean usefulness, and mean count of compromised and honest members expelled, then the mean usefulness of
the arm's honest games. Each row carries the arm's mechanism, as the agents' system prompts state it, and
its framing text. Clicking a row opens that arm's eval red games.

**Game.** Pick the split, arm, condition and seed, and for a dev game the red-team iteration. The header
lists the members in seat order with the compromised ones marked, the dictator, the Overseer or each
elected leader, the faction accounts, the compromised members' strategy in red games, and the game's
scores.

**Phase.** Step through the phases with the picker or **Previous phase** and **Next phase**: round 1
election, round 1 work, round 1 review, and so on, as the game played them. For each phase the page
shows the members still in the team, those expelled earlier, and the leader; the election tally and its
winner; the expulsion ballots; the payments made, with diverted and phished payments in red and a
leader's own unreviewed payments marked; the proposals made or decided; payments the engine refused,
with its reason; and who was expelled and how. The public log lines the phase added follow, with the
earlier log folded above them. The turns table lists each agent's turn in the phase: its role, whether
the engine could use the reply, the number of attempts, and the public message.

**Turn.** Click a turn to open it. The system prompt is folded at the top, then the user prompt, the
private notes and public message from the reply, the raw reply text, and the parsed reply. A turn with
several attempts has an attempt picker: each attempt shows the prompt it was sent, which after a server
rejection carries half the log and after an unreadable reply ends with the engine's request to reply in
JSON, and the reason it was not used. The heading gives the attempt's input and output tokens. Private
notes were never shown to other agents; an agent's own last two notes come back to it in its later
prompts.

## Open a turn in Chat

**Open in Chat** puts the conversation on screen away, starts a new one holding the agent's system
prompt and the attempt's user prompt, and switches to Chat. Tick **Include recorded reply** to add the
recorded reply as the assistant turn. Without it, press **Retry** on the Chat page to answer the prompt
with the loaded model. Chat's system prompt setting takes the agent's system prompt, as it does when a
saved conversation is loaded. **Download as Chat conversation** writes the same conversation as a file
Chat's **Load** reads.

## Re-run a turn

Open **Re-run this turn on the loaded model**, set the temperature, token limit and seed (the pilot
sampled at 0.7 with 800 tokens), and click **Re-run**. Before generating, the page templates the system
and user prompts with the loaded model and compares the prompt's length with the input tokens the
recorded attempt reported. The reply streams into a token strip: click a token to inspect it, or
right-click it to branch the re-run there, with an alternative token or your own text.

When the reply finishes, the page reads it as the engine would: it checks the phase's required field and
that the reply names only the invoices, proposals and members the prompt showed. For a work turn it
then takes each payment as the engine would, refusing what the engine refuses (an unknown or closed
invoice, an amount out of range, a payment over the round's capacity) and classifying the rest against
the game's registry, faction accounts and invoice totals as correct, diverted, phished or wrong. In a
gated arm, a member's payment would be a proposal for the leader. This is a counterfactual for one turn.
The game does not continue from it.

The agents were `meta-llama/Llama-3.3-70B-Instruct`. A 4-bit MLX conversion needs about 40 GB. When
another model is loaded the page says so beside the re-run, and **Open Models with the recorded model**
opens the Models page with the recorded model's ID in its box. A different model templates the prompt
to different tokens, so the length check reports a mismatch, and its reply is that model's, not the
agent's.
