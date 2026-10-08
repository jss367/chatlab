# Institutions pilot

The **Institutions pilot** extension reads the games recorded by the institutions pilot in
[realignment-benchmark](https://github.com/jss367/realignment-benchmark) (`experiments/institutions-pilot`).
In that pilot five agents on a finance team pay invoices for 8 rounds. In red games two of them are
secretly compromised and try to divert money to faction accounts. Seven institutions, the arms, decide
who approves payments and who can expel members. The pilot's README explains the world, the arms, the
red team and the scores.

The page steps through any game round by round and opens any agent turn with the exact prompts that
agent was given. From a turn you can open it in Chat, download it as a Chat conversation, or re-run it
on the loaded model with token measurements and branch the re-run at any token. It reads files only. It
never imports the pilot, never continues a game, and needs no model until you re-run a turn.

Enable it under **Settings → Extensions** and restart ChatLab. **Institutions** appears in the sidebar.

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

## What each view shows

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
