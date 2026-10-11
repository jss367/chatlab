"""Generate library/04-fork-origin-7e4aaaf/ (conversations.json + conversations-vectors/).

Format: chatlab-library-1, fork-tree era: each fork records its ``origin``
(parent branch and message/token/prompt coordinates), duplicated into its
sampling under the wire-compatibility key ``_chatlab_fork_origin`` so older
writers that drop unknown branch fields keep it. Replies also carry
``generation_settings``, ``thinking_mode`` and ``token_step_paused`` (all
present by this commit), and sampling carries ``skip_top_below``.
Generating commit: 7e4aaaf "Add conversation fork tree and direct branch
comparison (#125)".

Commands used:

    git worktree add --detach /tmp/chatlab-at-7e4aaaf 7e4aaaf
    cd /tmp/chatlab-at-7e4aaaf && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/04_fork_origin_7e4aaaf.py \
        <repo>/tests/fixtures/saved/library/04-fork-origin-7e4aaaf/conversations.json

The pane is built as that commit's state dictionary and written by that
commit's ``library.write``.
"""

import os
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

import library  # noqa: E402  (top-level module at this commit)

assert library.ORIGIN_COMPAT_KEY == "_chatlab_fork_origin"

STEER = {
    "format": "chatlab-steering-1", "model_id": "allenai/Olmo-3-7B-Think", "layer": 7,
    "vector": [1.5, -0.5, 0.25], "strength": 3.0, "enabled": True,
}
SETTINGS = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "skip_top_below": 0.05,
    "max_new_tokens": 512, "seed": 1234, "system_prompt": "Answer briefly.",
    "keep_reasoning": False, "assistant_prefill": "", "thinking_mode": "on",
}

question = {"role": "user", "content": "What is the capital of France?", "reasoning": ""}
answer = {
    "role": "assistant", "content": "The capital of France is Paris.",
    "reasoning": "Simple geography question.",
    "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 40, "generated_tokens": 21,
    "thinking_mode": "on", "generation_settings": SETTINGS, "steering": STEER,
}
main = [
    question, answer,
    {"role": "user", "content": "And its population?", "reasoning": ""},
    {
        "role": "assistant", "content": "About two million in the city proper.", "reasoning": "",
        "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 75, "generated_tokens": 12,
        "thinking_mode": "on", "generation_settings": dict(SETTINGS, seed=1235),
    },
]
fork_1 = [
    question, answer,
    {"role": "user", "content": "Shorter, please.", "reasoning": ""},
    # Stepped with Next token and paused before any visible text.
    {"role": "assistant", "content": "", "reasoning": "", "token_step_paused": True,
     "model": "allenai/Olmo-3-7B-Think", "thinking_mode": "on"},
]
fork_2 = [
    question,
    {
        "role": "assistant", "content": "The capital of France is Lyon.", "reasoning": "",
        "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 40, "generated_tokens": 8,
        "thinking_mode": "off", "generation_settings": dict(SETTINGS, thinking_mode="off"),
    },
]
fork_3 = [
    {"role": "user", "content": "What is the capital of Spain?", "reasoning": ""},
    {
        "role": "assistant", "content": "Madrid.", "reasoning": "",
        "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 40, "generated_tokens": 3,
    },
]
fork_4 = [question, answer]

forks = {
    "active": "Fork 3",
    "branches": {"Main": main, "Fork 1": fork_1, "Fork 2": fork_2, "Fork 3": fork_3, "Fork 4": fork_4},
    "origins": {
        "Fork 1": {"parent": "Main", "kind": "message", "turn": 1, "part": "content", "message_count": 2},
        "Fork 2": {
            "parent": "Main", "kind": "token", "turn": 1, "token": 6,
            "original": " Paris", "original_id": 12095,
            "replacement": " Lyon", "replacement_ids": [55123],
        },
        # A fork of a fork, replacing a prompt token.
        "Fork 3": {
            "parent": "Fork 2", "kind": "prompt", "turn": 1, "token": 9,
            "original": " France", "original_id": 9625,
            "replacement": " Spain", "replacement_ids": [17689],
        },
        # Its parent has since been deleted; the origin outlives it.
        "Fork 4": {"parent": "Fork 0", "kind": "copy", "turn": 1, "part": "content", "message_count": 2},
    },
    "sampling": {
        "Main": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "skip_top_below": 0.05,
                 "max_new_tokens": 512, "steering": STEER},
        "Fork 1": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "skip_top_below": 0.05,
                   "max_new_tokens": 512},
        "Fork 2": {"temperature": 0.0, "skip_top_below": 0.0},
    },
    "sampling_updated": {
        "Main": "2026-09-16T10:00:00.000010+00:00",
        "Fork 1": "2026-09-16T10:01:00.000020+00:00",
        "Fork 2": "2026-09-16T10:02:00.000030+00:00",
    },
    "updated": {
        "Main": "2026-09-16T09:00:00.000001+00:00",
        "Fork 1": "2026-09-16T09:01:00.000002+00:00",
        "Fork 2": "2026-09-16T09:02:00.000003+00:00",
        "Fork 3": "2026-09-16T09:03:00.000004+00:00",
        "Fork 4": "2026-09-16T09:04:00.000005+00:00",
        "Fork 0": "2026-09-16T09:05:00.000006+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
