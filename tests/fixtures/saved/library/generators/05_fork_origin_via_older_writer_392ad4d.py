"""Generate library/05-fork-origin-via-older-writer-392ad4d/.

Format: chatlab-library-1, as left by an OLDER writer that read a fork-tree
file: 392ad4d knows nothing of ``origin`` and drops it (and the turn fields it
does not know), but carries unknown sampling keys, so each fork's ancestry
survives only as the ``_chatlab_fork_origin`` wire-compatibility record inside
its sampling.
Generating commit: 392ad4d "Add per-conversation steering vectors (#59)",
reading and re-saving library/04-fork-origin-7e4aaaf (written by 7e4aaaf).

Commands used:

    git worktree add --detach /tmp/chatlab-at-392ad4d 392ad4d
    cd /tmp/chatlab-at-392ad4d && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/05_fork_origin_via_older_writer_392ad4d.py \
        <repo>/tests/fixtures/saved/library/04-fork-origin-7e4aaaf \
        <repo>/tests/fixtures/saved/library/05-fork-origin-via-older-writer-392ad4d/conversations.json

The 7e4aaaf directory is copied (file and vectors), then read by 392ad4d's
``library.read``; the older page adds an exchange to the fork on screen and
saves with 392ad4d's ``library.write``.
"""

import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
shutil.copytree(source, target.parent)
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

import library  # noqa: E402  (top-level module at this commit)

assert not hasattr(library, "ORIGIN_COMPAT_KEY")
forks = library.read(target)
assert forks is not None
forks["branches"]["Fork 3"] += [
    {"role": "user", "content": "And of Portugal?", "reasoning": ""},
    {"role": "assistant", "content": "Lisbon.", "reasoning": "",
     "model": "allenai/Olmo-3-7B-Think", "prompt_tokens": 60, "generated_tokens": 3},
]
forks["updated"]["Fork 3"] = "2026-09-17T08:00:00.000007+00:00"

written = library.write(forks, target)
assert written == target, written
print(target)
