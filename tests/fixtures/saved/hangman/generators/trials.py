"""Writes trials.json: a hangman trial file, the input a user writes for a batch.

Format marker: chatlab-hangman-trials-1
Era: the only one; the trial file has not changed since it was introduced.
Generating commit: 5c32174 Play Hangman trial files as batches, with an optional reveal probe (#157)

Commands used:
    git worktree add --detach /tmp/chatlab-at-5c32174 5c32174
    cd /tmp/chatlab-at-5c32174 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/hangman/generators/trials.py \
        <repo>/tests/fixtures/saved/hangman/trials.json

ChatLab never writes this file: a user does. So it is written here as that commit's HANGMAN.md shows one (the
documented example, extended to set every field a trial or the defaults may set), and then read by that
commit's own read_trials, which has to accept it, so the file is one the era's code took.

It exercises: a title with surrounding spaces, defaults for the probe that a trial leaves, one trial turning
the probe off with null and another replacing it whole, a trial without a label (labelled by its id), a fixed
guess list with max_guesses, temperature 0, an empty system prompt and a custom opening.
"""
import json
from pathlib import Path
import sys

from extensions.hangman.batch import read_trials

TRIALS = {
    "format": "chatlab-hangman-trials-1",
    "title": "  Qwen3 0.6B, five seeds  ",
    "defaults": {"temperature": 1.0, "max_new_tokens": 1024, "guesser": "frequency", "probe": {"samples": 5}},
    "trials": [
        {"id": "s1", "seed": 1},
        {"id": "s2", "seed": 2, "probe": None},
        {"id": "s1-greedy", "label": "Seed 1, greedy", "seed": 1, "temperature": 0},
        {"id": "fixed", "label": "Fixed guesses", "seed": 7, "guesser": ["e", "a", "crane"], "max_guesses": 3,
         "probe": {"samples": 2, "max_new_tokens": 8, "temperature": 0.5}},
        {"id": "custom-prompt", "seed": 9, "system": "", "opening": "Pick a word and draw the board.",
         "max_new_tokens": 4096},
    ],
}


def main(output):
    Path(output).write_text(json.dumps(TRIALS, indent=2) + "\n", encoding="utf-8")
    data = read_trials(output)
    assert [t["id"] for t in data["trials"]] == ["s1", "s2", "s1-greedy", "fixed", "custom-prompt"], data


if __name__ == "__main__":
    main(sys.argv[1])
