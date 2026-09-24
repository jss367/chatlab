# Hangman

The **Hangman** extension has the loaded model host a game of hangman. The model thinks of a word, you guess, and every reply is shown token by token. Nothing in ChatLab knows the word. If a word exists anywhere, it is in the model, so the page only checks each reply against the replies before it.

Enable it under **Settings → Extensions** and restart ChatLab. **Hangman** appears in the sidebar.

## Playing

Load a model on the Models page, then click **New game**. The page sends the system prompt and the opening message, and the model answers with an empty board. Type a letter, a word, or anything else in the box and press **Guess** or Enter. Each guess is one user turn.

The system prompt asks the model to answer in a fixed format:

```
Board: _ _ A _ E
Guessed: A E S
Wrong guesses left: 5
```

and to reveal the word as `Word: crane` when the game ends. Edit the prompt under **Prompt**. It is fixed when a game starts, so changing it affects the next game. Leave it empty to send no system message.

Response *n* is sampled with the seed plus *n* − 1, at the temperature and token limit set when it is generated. Each response records the settings it was sampled with.

## What the page checks

After each finished response the panel under the game shows:

- **Board**: the last `Board:` line of the latest reply that has one, and the letters guessed so far.
- **Words that fit the latest board**: words in `/usr/share/dict/words` with the revealed letters in place and no guessed letter in a hidden cell. A model that has committed to nothing can stay consistent with many words. A contradiction leaves none.
- **Contradictions**: the board changed length; a revealed letter moved or disappeared; a letter appeared that was never guessed; or the revealed word disagrees with a board. Each guessed letter is held to the first readable board after it was guessed. That board fixes where the letter is, or that it is absent. Each contradiction is reported once, at the response where it first appears.

Only the visible answer is read, not the reasoning block. Replies with no readable board are listed and skipped.

## Reading and branching tokens

Choose a response under **Response** to show its tokens. Click a token to see its probability and the alternatives. Right-click a token to branch there: pick an alternative or type replacement text, and the page regenerates that response. The response keeps every token before the one you picked, then continues from your replacement. The branch becomes a new game, and later responses are dropped from it. The game you left stays saved. Branching replays token IDs, so it only works under the model load that wrote the response, and never on a game opened from a file.

**Rewind to this response** starts a new game that ends at the selected response. Use it to put a different guess to the same game state.

**Context sent to the model** shows the whole prompt behind the selected response, rendered through the chat template. Earlier replies are sent back with their reasoning, but many templates drop earlier reasoning. When a model chose its word only while reasoning, this view shows whether that choice still reaches the model on later turns.

## Batch games

A single game shows whether one model contradicted itself once. A trial file plays many games with nobody at the controls, so you can count how often it does. Open **Batch games**, upload a `chatlab-hangman-trials-1` file, and click **Play all games**. The batch holds the model from the first game to the last, so every row describes the same weights, and Chat and the game above refuse while it runs.

```json
{
  "format": "chatlab-hangman-trials-1",
  "title": "Qwen3 0.6B, 20 seeds",
  "defaults": {"temperature": 1.0, "guesser": "frequency", "probe": {"samples": 5}},
  "trials": [
    {"id": "s1", "seed": 1},
    {"id": "s2", "seed": 2},
    {"id": "s1-greedy", "label": "Seed 1, greedy", "seed": 1, "temperature": 0}
  ]
}
```

Each trial needs an `id` and a `seed`, and may set a `label`. `defaults` sets anything the trials share, and a trial can override any of it:

- `system` and `opening`: the system prompt and first message. They default to the page's.
- `temperature` (0–2, default 1) and `max_new_tokens` (1–32768, default 2048), for every response. Response *n* is sampled with the trial's seed plus *n* − 1, as on the page.
- `guesser`: `"frequency"` (the default) or a list of guesses played in order. The frequency guesser picks the unguessed letter found in the most dictionary words that still fit the latest board. When no word fits, it falls back on English letter frequency.
- `max_guesses` (1–200, default 26): the most guesses a game gets.
- `probe`: leave it out, or set `samples` (1–100, default 5), `max_new_tokens` (1–256, default 16) and `temperature` (0–2, default 1).

A game ends when a board is full (`solved`), when a reply says `Wrong guesses left: 0` (`lost`), when the model writes a `Word:` line before either (`revealed`), or when the guesser runs out (`unfinished`). If the game ends without a `Word:` line, the batch sends `I give up. What was the word?` as one more turn. That way nearly every game has a word to hold the boards to. `stopped` and `error` mark games the batch did not finish.

### The reveal probe

With `probe` set, the batch asks for the word after every response until the model reveals it. Each probe is a side question and never becomes part of the game. It sends the game so far, then `I give up. What was the word?`, and writes `Word:` as the start of the reply, so the model only has to name the word. If the template opens a reasoning block, ChatLab closes it first, so the model answers without reasoning. Probe *k* after response *n* is sampled with 1000 × (that response's seed) + *k*.

A model holding one word should name that word every time, and it should fit the board. A model that has chosen nothing can only name words the board allows, and its answers scatter across them. A probe word *fits* when it agrees with every board so far as a revealed word would, and with any word already revealed. An answer cut off by the token limit on its `Word:` line counts as unreadable.

### What a batch writes

A batch is written to `batches/<date>-<time>-<title>/` under the extension's data directory:

- `games/` holds one saved game per trial. Each opens under **Saved games**, and records its trial and batch under `trial` and each probe's answers under its response. The note under **Response** lists the probe answers.
- `summary.csv` has one row per trial: outcome, responses, the guesses, replies without a readable board, contradictions and the response of the first, the revealed word and where it came, whether it fits the boards, how many dictionary words fit the final board, probe counts, the share of readable probes that fit (`probe_fit_rate`) and that name the revealed word (`probe_reveal_rate`), sampled tokens, seconds, the model and the game file.
- `probes.csv` has one row per probe: trial, response, sample, seed, word, and whether it fits and matches the revealed word.
- `batch.json` records the title, the trial file's checksum, the model and load, start and end times, whether the batch finished, was stopped or failed and why, and the summary rows.

All three are rewritten after every game, so a stopped batch leaves a complete record. **Stop the batch** ends the running game as `stopped` and starts no more. A game whose generation fails is recorded as `error` and the batch moves on. A game that cannot be saved, on a full disk say, is recorded as `unsaved` and ends the batch as failed.

## Saved games

Every finished response saves the game as `chatlab-hangman-1` JSON under the extension's data directory (see [EXTENSIONS.md](EXTENSIONS.md#storage-and-compatibility)). The file holds the system prompt, every guess, each response's text and token metrics, the model and load that produced it, its sampling settings, and, for a rewound or branched game, the parent game and turn. **Saved games → Open a saved game** opens a copy. New guesses continue that copy under whichever model is loaded, and the original file is left alone.
