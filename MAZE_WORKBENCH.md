# Maze navigation workbench

Enable **Maze experiments** under **Settings → Extensions**, restart ChatLab, then open **Maze** to run and inspect one navigation episode. Load a Qwen2.5 Instruct model on **Models** first. The current action parser recognizes Qwen-style `<tool_call>` envelopes; other tool protocols need their own parser. The model may emit arbitrary text, make an invalid call, or stop without acting.

## Use

1. Select maze size, maze seed, shortest route length, and open-cell probability. Some combinations cannot produce a maze; the generator reports this rather than changing the requested length.
2. Choose **Goal information**: **Exact coordinates** gives the model the destination's row and column; **Hidden location** tells it a destination exists and asks it to explore; **Hint only** supplies your required **Goal hint** without adding destination coordinates. Write a clue about the destination shown on the board, and check it again if you change the maze settings. Hidden and hint modes reset supplied starting moves to 0. You can deliberately supply moves, but they demonstrate the shortest route toward the goal.
3. Choose interruption text, token count, and timing. Timing counts accepted moves, including supplied moves. The interruption is inserted into the next assistant response. **None** runs a clean control.
4. Click **New episode · apply settings**, then **Run episode** or **Step one response**. Editing controls alone does not change the current episode.
5. Watch the character and path alongside generated tokens. Click tokens for probabilities and alternatives. The exact supplied prefix is shown separately, and the full raw response retains both parts. For a reasoning model, tokens emitted inside its exposed reasoning block remain visible; these are not access to hidden internal reasoning. Qwen2.5 Instruct has no separate reasoning channel.
6. **Pause after response** waits for the current response to end. **Stop now** retains partial tokens and does not execute a partial action. **Interrupt next response** overrides the scheduled insertion time, recording the run as manually intervened.
7. Select a response under **Path and replay** to inspect its board and tokens. Pause before selecting. Export JSON or upload a previous export for read-only replay.

A dashed path marks supplied moves; the solid indigo path marks actual accepted model moves. The amber ring marks the position where the interruption was inserted. The board always shows you the destination, including in hidden and hint modes. The shortest-route overlay is only for the viewer, never part of the prompt. The status reports the current episode's goal mode; editing the selector affects the next new episode.

## Run locally

Use the repository's normal Python environment and dependencies:

```sh
python app.py
```

Set `CHATLAB_MAZE_RUNS_PATH` to override the default `~/.local/share/chatlab/extensions/maze_experiments`. Completed responses and terminal states are autosaved there. Existing ChatLab conversations remain separate. A model loaded in Chat or Models is shared with Maze, and generation is mutually exclusive across views.

## Protocol and interpretation

The entire map, coordinate labels, current position and legal directions are visible to the model in all goal modes. Destination coordinates appear in the initial prompt and tool replies only in **Exact coordinates** mode (the default). **Hidden location** and **Hint only** omit destination coordinates from every simulator message, including errors and supplied-move history; hint mode includes your clue verbatim. The simulator confirms arrival when the model steps onto the destination. The actual goal remains in the viewer's board and saved JSON for inspection and scoring, and exports preserve the selected mode and hint. Older exports default to exact coordinates.

Each accepted `move(maze_id, direction)` advances exactly one cell. A tool-looking example inside a code fence, blockquote or reasoning block is not executed. Multiple or malformed calls return a tool error. A complete response with no attempted call ends the episode as abandonment; the controller sends no reminder. A response cut off by a limit does not execute its unfinished action.

Default settings mirror the planned pilot: 5×5 maze, shortest distance 10, three supplied canonical moves, an 8-token interruption, temperature 0.7, 1,024 sampled tokens per response, 8,192 per episode and 32 tool attempts. The first accepted move after insertion must occur within 1,024 sampled tokens / 4 attempts. Returning to movement is distinct from making progress toward the goal. Supplied tokens do not count toward sampled-token budgets. Stopping manually is not a failed recovery. A clean run has no post-interruption recovery score.

This is an exploratory workbench, not the batch experiment or a training implementation. Its configurable maze family and manual interventions should not be mixed into a preregistered evaluation unnoticed. No maze-trained adapter is bundled. The JSON includes configuration, simulator transitions, complete message history, actual prompt token IDs, output token IDs and probabilities, stop reasons and intervention provenance. Each new turn re-renders prior response text through the model's native template; this is disclosed rather than claiming uninterrupted token-stream identity across turns.

## Verification

`python -m unittest discover -s tests` runs the existing ChatLab suite plus simulator/controller tests. Maze tests cover goal disclosure across initial prompts and tool replies, settings callbacks, hidden-goal identifiers, legacy and new replay formats, deterministic maps and route length, blocked/fake/quoted/unfinished actions, supplied-token accounting, recovery and arrival, abandonment, pause/stop, native tool-template forwarding and replay path validation.

The extension boundary and enable/disable lifecycle are documented in [Optional extensions for ChatLab](EXTENSIONS.md). Existing JSON exports remain compatible. Legacy `~/.local/share/chatlab/maze_runs` files remain available to upload.
