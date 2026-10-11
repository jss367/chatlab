"""Generate library/01-first-02683f2/conversations.json.

Format: chatlab-library-1, first era (branches, turns, updated stamps and the
forgotten map; no sampling, steering, origins or archive).
Generating commit: 02683f2 "Persist conversations, cache inspections, add
4-bit weights, run tests in CI (#37)", the commit that introduced the file.

Commands used:

    git worktree add --detach /tmp/chatlab-at-02683f2 02683f2
    cd /tmp/chatlab-at-02683f2 && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/01_first_02683f2.py \
        <repo>/tests/fixtures/saved/library/01-first-02683f2/conversations.json

The pane is built as the app's state dictionary of that commit and handed to
that commit's ``library.write``; the bytes are its writer's.
"""

import os
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

import library  # noqa: E402  (top-level module at this commit)

assert library.LIBRARY_FORMAT == "chatlab-library-1"

main = [
    {"role": "user", "content": "What is the capital of France?", "reasoning": ""},
    {
        "role": "assistant", "content": "The capital of France is Paris.",
        "reasoning": "The user wants a capital city. France -> Paris.",
        "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 31, "generated_tokens": 58,
    },
    {"role": "user", "content": "And of Italy? Answer with the city: Füße — naïve ✓", "reasoning": ""},
    {
        "role": "assistant", "content": "Rome.", "reasoning": "",
        "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 70, "generated_tokens": 12,
    },
]
fork = [
    *main[:2],
    {"role": "user", "content": "Now answer in French.", "reasoning": ""},
    # Still streaming when the file was written: reasoning only, no answer yet.
    {"role": "assistant", "content": "", "reasoning": "Je dois répondre en français"},
]
chat = [
    {"role": "user", "content": "Write a haiku about rain.", "reasoning": ""},
    # Streaming had produced nothing yet.
    {"role": "assistant", "content": "", "reasoning": ""},
]

forks = {
    "active": "Fork 1",
    "branches": {"Main": main, "Fork 1": fork, "Chat 1": chat},
    "updated": {
        "Main": "2026-09-08T10:00:00.000001+00:00",
        "Fork 1": "2026-09-08T10:05:00.000002+00:00",
        "Chat 1": "2026-09-08T10:06:00.000003+00:00",
        # Deleted: kept only as a stamp in "forgotten".
        "Chat 2": "2026-09-08T10:07:00.000004+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
