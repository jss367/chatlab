"""Generate library/03-steering-392ad4d/ (conversations.json + conversations-vectors/).

Format: chatlab-library-1, steering era: a branch's sampling and a reply can
carry a steering vector, stored as a ``chatlab-steering-reference-1`` record
whose vector data lives once in the sibling ``conversations-vectors``
directory, named by its SHA-256.
Generating commit: 392ad4d "Add per-conversation steering vectors (#59)".

Commands used:

    git worktree add --detach /tmp/chatlab-at-392ad4d 392ad4d
    cd /tmp/chatlab-at-392ad4d && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/03_steering_392ad4d.py \
        <repo>/tests/fixtures/saved/library/03-steering-392ad4d/conversations.json

The full vectors are handed to that commit's ``library.write``, whose
``compact`` writes the vector files and the references.
"""

import os
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

import library  # noqa: E402  (top-level module at this commit)

TOWARDS_CALM = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 2,
    "vector": [0.5, -0.25, 1.0, 0.0], "strength": 4.0, "enabled": True,
}
AWAY_FROM_FRENCH = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 5,
    "vector": [0.125, 0.25, -0.5, 2.0, -1.0, 0.75], "strength": -2.5, "enabled": False,
}

main = [
    {"role": "user", "content": "How are you feeling today?", "reasoning": ""},
    {
        "role": "assistant", "content": "Very calm, thank you.", "reasoning": "",
        "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 18, "generated_tokens": 7,
        "steering": TOWARDS_CALM,
    },
]
fork = [
    *main,
    {"role": "user", "content": "Say it in French.", "reasoning": ""},
    {
        "role": "assistant", "content": "Je suis très calme.", "reasoning": "",
        "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 40, "generated_tokens": 8,
        "steering": AWAY_FROM_FRENCH,
    },
]

forks = {
    "active": "Fork 1",
    "branches": {"Main": main, "Fork 1": fork},
    "sampling": {
        "Main": {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "max_new_tokens": 128,
                 "steering": TOWARDS_CALM},
        "Fork 1": {"temperature": 0.0, "steering": AWAY_FROM_FRENCH},
    },
    "sampling_updated": {
        "Main": "2026-09-10T12:00:00.000010+00:00",
        "Fork 1": "2026-09-10T12:10:00.000020+00:00",
    },
    "updated": {
        "Main": "2026-09-10T11:00:00.000001+00:00",
        "Fork 1": "2026-09-10T11:10:00.000002+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
