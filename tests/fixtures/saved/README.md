# Frozen saved files

Every file here was written by the ChatLab commit named beside it, using that
commit's own writer. They are the evidence that results and data saved by
earlier versions still load, and read the same way, after later changes. The
`test_saved_*.py` modules open each one through the path the app uses
and check specific facts about what it reads: names, texts, numbers, scores,
derived values. Each test also saves the data again with today's writer and
checks that those facts survive.

Rules:

- **Never regenerate or edit an existing file**, even one byte. If a change
  makes one of these tests fail, the change has altered how an old file reads.
  Fix the reader, or, if the new reading is deliberate, update the assertion
  and say why in the commit.
- **Add a new file when you add a format or a new era of one.** An era is a new
  optional field, a compatibility path, or anything else the reader treats
  specially. Generate it with the commit that introduced it:
  `git worktree add --detach /tmp/chatlab-at-<hash> <hash>`, then run a
  generator script from `<area>/generators/` inside that checkout. Each
  generator's header names its commit and the exact command. Commit the
  generator with the file, and add a row below.
- Tests never write to these files. They copy them into a temporary directory
  first, because several readers look for sibling files (vectors, pictures,
  the archive companion, experiments) next to them.

`test_saved_format_coverage.py` fails when a `chatlab-<name>-<n>` marker that
ChatLab reads back has no file here (or under `../institutions`) carrying it.

Maze runs have their own table: [maze/README.md](maze/README.md). The
invoice-payment Institutions pilot (`chatlab-institutions-run-1`, `game-1`) is
`institutions/invoice-pilot-run-1`, written at 50db305 by
`institutions/generators/make_invoice_pilot.py`. The customer-support bundles
(run-2, game-2) are real exports in [../institutions](../institutions).

| File | Format marker | Era | Generating commit | Features exercised |
|---|---|---|---|---|
| `library/01-first-02683f2/` | chatlab-library-1 | first: branches, turns, stamps, forgotten | 02683f2 (#37) | 3 branches; reasoning; model and token counts; non-ASCII; a reply cut off with reasoning only (kept, closed); an empty streaming reply (dropped); a forgotten branch |
| `library/02-sampling-cba40be/` | chatlab-library-1 | per-branch sampling | cba40be (#45) | sampling on its own stamp; an integer temperature (read as float); an unknown sampling key carried through; a stamp with no sampling (sampling removed) |
| `library/03-steering-392ad4d/` | chatlab-library-1 (+ chatlab-steering-reference-1) | steering | 392ad4d (#59) | steering references in sampling and on replies; the vector files in `conversations-vectors/`; a disabled negative-strength vector; temperature 0 |
| `library/04-fork-origin-7e4aaaf/` | chatlab-library-1 | fork origins | 7e4aaaf (#125) | message, token, prompt and copy origins (native and `_chatlab_fork_origin`); a fork of a fork; an origin whose parent was deleted; `generation_settings`, `thinking_mode`, a reply paused by Next token, `skip_top_below` |
| `library/05-fork-origin-via-older-writer-392ad4d/` | chatlab-library-1 | origins kept only by the wire copy | 392ad4d (#59), re-saving 04 | a writer older than #125 dropped `origin` and the turn fields it didn't know, and kept `_chatlab_fork_origin` in sampling; adds an exchange |
| `library/06-archive-d95fa1a/` | chatlab-library-1 | archive, before the compatibility copy | d95fa1a | archived and restored branches on their own stamp; the active branch archived (page opens on the archive); a picture in `conversations-images/`; a failed reply with partial text |
| `library/07-current-cfd9fbd/` | chatlab-library-1 | current | cfd9fbd (writer last changed in #206) | `_chatlab_archive` with the wire stamp advanced; the `conversations.json.archive` companion; two pictures; steering; an unknown sampling key; a reply failed with no text; Institutions provenance; a paused reply; a forgotten branch |
| `library/08-archive-via-older-writer-7e4aaaf/` | chatlab-library-1 | archive kept only by the wire copy | 7e4aaaf (#125), re-saving 07 without its companion | `archived` fields dropped; archive and the real sampling stamp recovered from `_chatlab_archive`; pictures, errors and provenance dropped by that writer |
| `library/09-archive-after-stale-older-writer-7e4aaaf/` | chatlab-library-1 | archive kept only by the companion | 7e4aaaf (#125), a stale page saving over 07 | the stale page's newer sampling stamp replaced the record that held the archive; `conversations.json.archive` still holds it |
| `conversation/01-first-890d94a.json` | chatlab-conversation-1 | first | 890d94a | multi-line system prompt; reasoning; `<code>`, typographic quotes and a dash; ends on an unanswered user turn |
| `conversation/02-steering-392ad4d.json` | chatlab-conversation-1 | steering | 392ad4d (#59) | model and token counts; conversation and reply steering; `steering_vectors` with each vector embedded once |
| `conversation/03-pictures-b69af3b.json` | chatlab-conversation-1 | pictures | b69af3b (#176) | 2 embedded pictures, one reused; `thinking_mode`; `generation_settings`; a failed reply; a paused reply; steering |
| `conversation/04-current-cfd9fbd.json` | chatlab-conversation-1 | current | cfd9fbd (field added in #215) | `institutions_provenance` (parent, adapter, recorded message provenance); a picture; a disabled vector; a reply that failed after reasoning only |
| `steering/01-hand-written-392ad4d.json` | chatlab-steering-1 (no marker) | user-written, as the 392ad4d README describes | 392ad4d (#59), checked with its `read_vector` | `format`, `strength` and `enabled` left out; integer entries; padded model ID |
| `steering/02-extracted-be75ba8.json` | chatlab-steering-1 | extracted and downloaded | be75ba8 (#102) | difference in means at layer 1, written by `download_extracted` (cfd9fbd writes identical bytes) |
| `steering/03-neuron-conversation.json` | chatlab-neuron-steering-1 (inside chatlab-conversation-1) | first neuron settings | Neuron steering introduction (this change), `steering/generators/make_neuron_conversation.py` | Negative addition; multiply by zero; set to zero; disabled set; current layer differs from recorded replies; inline settings without vector assets |
| `settings/01-first-3db3f4e.json` | settings.json | first | 3db3f4e (#31) | sampling, prompts, seed, colour scale, prefill limit, MPS fraction; everything later read as default |
| `settings/02-themes-88ec738.json` | settings.json | themes and extensions | 88ec738 (#105) | thinking mode, writing suggestions, 4-bit weights, image settings, theme, appearance, enabled extensions |
| `settings/03-current-cfd9fbd.json` | settings.json | current | cfd9fbd (writer last changed in #162) | `skip_top_below`; four extensions; a newer version's key carried through |
| `experiments/01-reply-42396fe/` | schema_version 1 | first | 42396fe (#126) | a reply saved from its trace by the tab's `save_run`; 2 bookmarks; an attached logit-lens inspection; real metrics from a tiny decoder |
| `experiments/02-measurement-timeline-cf1acea/` | schema_version 1 | timeline readouts | cf1acea (#155) | a measured passage with `run_id` and `session_id`; a bookmark; 3 `timeline_inspections` captured by the Timeline's `capture` |
| `probes/probe-era1-no-revision-below-chance.json` | chatlab-probe-1 | first: no `model_revision`, no `precision` | 22f2efd (#179) | unpaired near-twin examples (every layer below chance, page shows a hint); last-token pool; chat template; 4 folds; indented examples; a non-ASCII name |
| `probes/probe-era2-revision-paired.json` | chatlab-probe-1 | `model_revision`, no `precision` | bf31e2f (#179) | paired examples; mean pool; plain text; 5 folds; tied held-out accuracies |
| `probes/probe-era3-precision.json` | chatlab-probe-1 | revision and `precision` | ce866c2 (#179) | 8-bit; unequal sides (3 folds); built for `test/tiny-decoder` so direction edits can use it |
| `direction_edits/directions-sparse-unnormalized.json` | chatlab-directions-1 | only (written by hand or script) | 390252b (#196), checked with its `normalize_directions` | name with spaces; blocks out of order; a missing block; non-unit vectors; an integer entry; revision, precision |
| `direction_edits/steering-tiny-decoder-layer0.json` | chatlab-steering-1 | as Chat's extraction writes it | 390252b (#196) | real extraction on the tiny decoder; read through the Edits page's `open_vector` |
| `direction_edits/direction-edits-lens-random-injection.json` | chatlab-direction-edits-1 | only | 390252b (#196), downloaded from `run_experiment` | 3 conditions; injection; erase at block 1; random control; lens readout; 2 differences; null rows at block 3 |
| `direction_edits/direction-edits-probe-add-no-injection.json` | chatlab-direction-edits-1 | only | 390252b (#196), downloaded from `run_experiment` | the era-3 probe as directions; add 2.5 at blocks 2–3; random control; no readout; no differences |
| `circuits/graph-top-before-revisions/` | chatlab-attribution-graph-1 | first writer: no revisions, provenance, width or menu label | 90bcb9f (first commit of #177, never released) | top mode with 10 targets; chat prompt with system text; labels, 2 groups, measured effects; menu falls back to the file name |
| `circuits/graph-tokens-released/` | chatlab-attribution-graph-1 (+ menu label file) | released | f01bf05 (#177 merge) | chosen tokens; reply prefix; provenance and revisions; view version; label, groups, effects; menu label from `.metadata` |
| `circuits/graph-contrast-released/` | chatlab-attribution-graph-1 (+ menu label file) | released | f01bf05 (#177 merge) | contrast target with `negative_probability`; raw prompt; automatic alternatives; boost ×3.5 at every position; non-ASCII label |
| `hangman/hand-played-reopened-branched.json` | chatlab-hangman-1 | hand-played | 4d2b657 (#147) | edited system prompt; reasoning, including a block the template opened; escaped, run-together and unreadable boards; a reply cut off at the limit; reopened (parent, load id); branched at a token; revealed CRATE against a board spelling CRANE; 8 contradictions |
| `hangman/batch-game-with-probes.json` | chatlab-hangman-1 | batch games with reveal probes | 5c32174 (#157) | `trial` stamp; 12 probe answers (fitting, not fitting, none, cut off on and after the `Word:` line); "I give up"; summary fields |
| `hangman/trials.json` | chatlab-hangman-trials-1 | only (user-written) | 5c32174 (#157), checked with its `read_trials` | padded title; default probe filled in, turned off, replaced; label defaults to id; fixed guesses with `max_guesses`; temperature 0; custom opening |
| `computer_safety/judgment-run.json` | chatlab-computer-safety-run-1 | free-text judgments | 5e35942 (#134) | 10 cases, non-ASCII, unsorted metadata, 2 sources, unlabeled and null labels; JSON, bare, fenced, reasoning, wrong and invalid answers; a run stopped part way |
| `computer_safety/judgment-run-exported.json` | chatlab-computer-safety-run-1 | flat `imported_provenance` | 5e35942 (#134), its own open-then-export | the flat provenance object becomes the oldest entry of today's list |
| `computer_safety/probability-run.json` | chatlab-computer-safety-run-1 | label-probability scoring | dfe96f9 (#138) | probabilities and log-probabilities; confidence; a skipped case; blocking curve, AUC 0.833, operating point; predictions that change at thresholds 0.40–0.50 |
| `steering_comparison/paired-ranks.json` | chatlab-steering-comparison-1 | first | 072f85c, downloaded through `ComparisonRuns.publish` | two opposite-sign vectors; two decision points; supplied outcomes and unknown outcome; total and computed effects; all-block rank traces; random controls; null t statistics |
