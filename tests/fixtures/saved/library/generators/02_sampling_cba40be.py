"""Generate library/02-sampling-cba40be/conversations.json.

Format: chatlab-library-1, sampling era: a branch may carry its own sampling
(temperature, top-p, top-k, response length) under a ``sampling_updated``
stamp of its own; unknown sampling keys are carried through; a stamp with no
sampling records that the sampling was taken away.
Generating commit: cba40be "Fit verdicts in the model lists, per-conversation
sampling, a local API, and a hardware panel (#45)", which added sampling.

Commands used:

    git worktree add --detach /tmp/chatlab-at-cba40be cba40be
    cd /tmp/chatlab-at-cba40be && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/02_sampling_cba40be.py \
        <repo>/tests/fixtures/saved/library/02-sampling-cba40be/conversations.json

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

main = [
    {"role": "user", "content": "Name three primary colours.", "reasoning": ""},
    {
        "role": "assistant", "content": "Red, yellow and blue.",
        "reasoning": "Traditional painter's primaries.",
        "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 22, "generated_tokens": 19,
    },
]
fork = [
    *main,
    {"role": "user", "content": "Now for light.", "reasoning": ""},
    {
        "role": "assistant", "content": "Red, green and blue.", "reasoning": "",
        "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 48, "generated_tokens": 9,
    },
]
chat = [{"role": "user", "content": "Unanswered question", "reasoning": ""}]

forks = {
    "active": "Main",
    "branches": {"Main": main, "Fork 1": fork, "Chat 1": chat},
    "sampling": {
        "Main": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "max_new_tokens": 256},
        # An int temperature is written as a float; an unknown key written by
        # a newer version is carried through.
        "Fork 1": {"temperature": 1, "top_k": 0, "future_knob": {"level": 3}},
    },
    "sampling_updated": {
        "Main": "2026-09-09T09:00:00.000010+00:00",
        "Fork 1": "2026-09-09T09:30:00.000020+00:00",
        # A stamp without sampling: the branch's own sampling was removed.
        "Chat 1": "2026-09-09T09:40:00.000030+00:00",
    },
    "updated": {
        "Main": "2026-09-09T08:00:00.000001+00:00",
        "Fork 1": "2026-09-09T08:30:00.000002+00:00",
        "Chat 1": "2026-09-09T08:40:00.000003+00:00",
        "Fork 2": "2026-09-09T08:50:00.000004+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
