# Twenty questions

The **Twenty questions** extension has the loaded model host a game of twenty questions. The model thinks of a thing, you ask yes-or-no questions, and every reply is shown token by token. Nothing in ChatLab knows the thing. If one exists anywhere, it is in the model, so the page checks each reply against the replies before it and shows what the imported Jacobian lens reads in the model's residual stream (J space) while it answers.

Enable it under **Extensions** and restart ChatLab. **20 Questions** appears in the sidebar.

## Playing

Load a model on the Models page, then click **New game**. The page sends the system prompt and the opening message. Type a question or a guess and press **Ask** or Enter. Each question is one user turn.

The system prompt asks the model to answer in a fixed format:

```
Answer: yes
Questions left: 17
```

and to name the thing as `Thing: a giraffe` when the game ends. Edit the prompt under **Prompt**. The prompt is fixed when a game starts, so a change applies to the next game. Leave it empty to send no system message.

Response *n* is sampled with the seed plus *n* − 1, at the temperature and token limit set when it is generated.

## J space

With **Read J space after every reply** on, each finished reply is read through the Jacobian lens imported for the loaded model in **Chat → Layers and attention → Jacobian**. The reading needs a full-precision PyTorch load. Without a lens the pane says so and the game goes on.

Two positions of each reply are read:

- **Starting the reply**: the last prompt token, the state the model begins its reply from.
- **Choosing the answer**: the token just before the yes or no on the `Answer:` line, the state that picks it. A reply without a readable answer line has only the first position.

At each position, every block the lens was fitted at shows its top eight tokens, deepest block first. Warmer cells hold more probability. A lens readout is a reading of the residual stream, so a token showing there does not mean the model would say it.

**Watch list** takes things separated by commas. Each one is read as the token IDs of the thing after a space, through the mean of their unembeddings, and its log probability is averaged over the fitted blocks. **Watch list across the game** shows that number for every response, read where it chooses its answer (or where it starts, when it gave none). A thing the model is holding may rise there before it is ever written.

**Read J space for the whole game again** reads every response the loaded model wrote with the current watch list. A response from another load, or from a file, cannot be read again, because its token IDs are only meaningful under the load that wrote them.

Click a token in the response's strip and press **Read J space after the selected token** to read that one position. That reading is shown and not saved.

## What the page checks

After each finished reply the panel shows:

- **Questions answered**, with the count of yes and no answers and the latest `Questions left:`.
- **Thing named**: the first thing on a `Thing:` line.
- **Contradictions**: `Questions left` did not drop by one for each question since the last count; a question asked again, word for word apart from case and punctuation, got the other answer; or a later reply named a different thing.

Only the visible answer is read, not the reasoning block.

## Ask what it is

**Ask after the latest response** asks the model what the thing is, as a side question that never joins the game. It sends the game so far, then `I give up. What was it?`, and writes `Thing:` as the start of the reply. Sample *k* is sampled with 1000 × (the latest response's seed) + *k*. A model holding one thing names it every time; one that has chosen nothing scatters across what fits. The answers are listed in the response's note.

## Recheck the answers

**Recheck** asks every question the game answered yes or no again, each in a fresh conversation about one thing, greedily, and compares the answers. The thing defaults to the one the game named; type another to ask whether the answers fit it instead. The result is shown in the panel and saved with the game, and is dropped when the game continues, rewinds or branches.

## Reading and branching tokens

Choose a response under **Response** to show its tokens. Click a token to see its probability and alternatives. Right-click a token to branch there: pick an alternative or type replacement text, and the page regenerates that response from your replacement in a new game. Branching replays token IDs, so it only works under the model load that wrote the response.

**Rewind to this response** starts a new game that ends at the selected response. **Context sent to the model** shows the whole prompt behind the selected response, rendered through the chat template.

## Saved games

Every finished response saves the game as `chatlab-twenty-questions-1` JSON under the extension's data directory (see [EXTENSIONS.md](EXTENSIONS.md#storage-and-compatibility)). The file holds the system prompt, every question, each response's text, token metrics, sampling settings, J space reading, reveal probes, and the latest recheck. **Saved games → Open a saved game** opens a copy. New questions continue that copy under whichever model is loaded, and the original file is left alone.
