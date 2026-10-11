"""Generate library/09-archive-after-stale-older-writer-7e4aaaf/.

Format: chatlab-library-1 plus the ``conversations.json.archive`` companion,
as left when a STALE page of an older release saves a slider change over a
current file: the older page loaded the conversation before it was archived,
so its sampling carries no ``_chatlab_archive`` record, and its newer sampling
stamp wins the merge and replaces the whole sampling entry - the in-file
archive record is gone, and only the companion (which older releases never
touch) still says the branch is archived.
Generating commit: 7e4aaaf "Add conversation fork tree and direct branch
comparison (#125)", saving over library/07-current-cfd9fbd (written by
cfd9fbd, companion included).

Commands used:

    git worktree add --detach /tmp/chatlab-at-7e4aaaf 7e4aaaf
    cd /tmp/chatlab-at-7e4aaaf && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/09_archive_after_stale_older_writer_7e4aaaf.py \
        <repo>/tests/fixtures/saved/library/07-current-cfd9fbd \
        <repo>/tests/fixtures/saved/library/09-archive-after-stale-older-writer-7e4aaaf/conversations.json

The stale page's pane is what 7e4aaaf would hold for "Chat 3" had it loaded
it before the archive and then had its temperature slider moved; it is saved
with 7e4aaaf's ``library.write``, which merges it into the copied file.
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

stale = {
    "active": "Chat 3",
    "branches": {
        "Chat 3": [
            {"role": "user", "content": "Archived conversation", "reasoning": ""},
            {"role": "assistant", "content": "Still here, just put away.", "reasoning": "",
             "model": "Qwen/Qwen3-4B", "prompt_tokens": 12, "generated_tokens": 6},
        ],
    },
    "origins": {},
    "sampling": {"Chat 3": {"temperature": 0.7}},
    # Moved after the archive (12:00) on the stale page.
    "sampling_updated": {"Chat 3": "2026-10-10T13:00:00.000060+00:00"},
    # The transcript as loaded: same stamp as the file's, so it does not win.
    "updated": {"Chat 3": "2026-10-10T09:04:00.000005+00:00"},
}

written = library.write(stale, target)
assert written == target, written
print(target)
