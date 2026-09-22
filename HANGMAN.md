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

Choose a response under **Response** to show its tokens. Click a token to see its probability and the alternatives. Right-click a token to branch there: pick an alternative or type replacement text, and the page regenerates that response. The response keeps every token before the one you picked, then continues from your replacement. The branch becomes a new game, and later responses are dropped from it. The game you left stays saved. Branching replays token IDs, so it only works under the model load that wrote the response.

**Rewind to this response** starts a new game that ends at the selected response. Use it to put a different guess to the same game state.

**Context sent to the model** shows the whole prompt behind the selected response, rendered through the chat template. Earlier replies are sent back with their reasoning, but many templates drop earlier reasoning. When a model chose its word only while reasoning, this view shows whether that choice still reaches the model on later turns.

## Saved games

Every finished response saves the game as `chatlab-hangman-1` JSON under the extension's data directory (see [EXTENSIONS.md](EXTENSIONS.md#storage-and-compatibility)). The file holds the system prompt, every guess, each response's text and token metrics, the model and load that produced it, its sampling settings, and, for a rewound or branched game, the parent game and turn. **Saved games → Open a saved game** opens a copy. New guesses continue that copy under whichever model is loaded, and the original file is left alone.
