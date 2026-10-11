"""Generate library/08-archive-via-older-writer-7e4aaaf/.

Format: chatlab-library-1, as left by an OLDER writer that read a current
file: 7e4aaaf knows fork origins but not archiving, so it drops the
``archived``/``archived_updated`` branch fields and keeps the archive only as
the ``_chatlab_archive`` record it carries inside sampling, beside the wire
``sampling_updated`` stamp that the current writer advanced to the archive
stamp. It also drops the turn fields it does not know (images, errors,
Institutions provenance).
Generating commit: 7e4aaaf "Add conversation fork tree and direct branch
comparison (#125)", reading and re-saving library/07-current-cfd9fbd
(written by cfd9fbd).

Commands used:

    git worktree add --detach /tmp/chatlab-at-7e4aaaf 7e4aaaf
    cd /tmp/chatlab-at-7e4aaaf && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/08_archive_via_older_writer_7e4aaaf.py \
        <repo>/tests/fixtures/saved/library/07-current-cfd9fbd \
        <repo>/tests/fixtures/saved/library/08-archive-via-older-writer-7e4aaaf/conversations.json

Only the library file and its vectors and pictures directories are copied -
as when the file is carried to a second machine running the older release -
so the ``conversations.json.archive`` companion is deliberately absent and
the compatibility record alone has to carry the archive. The older page adds
an exchange to Main and saves with 7e4aaaf's ``library.write``.
"""

import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
shutil.copytree(source, target.parent, ignore=shutil.ignore_patterns("*.archive"))
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

import library  # noqa: E402  (top-level module at this commit)

assert not hasattr(library, "ARCHIVE_COMPAT_KEY")
forks = library.read(target)
assert forks is not None
forks["branches"]["Main"] += [
    {"role": "user", "content": "Thanks!", "reasoning": ""},
    {"role": "assistant", "content": "You're welcome.", "reasoning": "",
     "model": "Qwen/Qwen3-4B", "prompt_tokens": 450, "generated_tokens": 4},
]
forks["updated"]["Main"] = "2026-10-11T08:00:00.000008+00:00"

written = library.write(forks, target)
assert written == target, written
print(target)
