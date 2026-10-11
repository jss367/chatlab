"""Generate library/06-archive-d95fa1a/ (conversations.json + conversations-images/).

Format: chatlab-library-1, archive era: a branch can be ``archived`` under an
``archived_updated`` stamp of its own (a stamp without ``archived`` means it
was brought back). No wire-compatibility copy of the archive yet. By this
commit user turns can carry ``images`` (stored beside the file in
``conversations-images``) and a failed reply carries its ``error``.
Generating commit: d95fa1a "Archive conversations from their row instead of
deleting them".

Commands used:

    git worktree add --detach /tmp/chatlab-at-d95fa1a d95fa1a
    cd /tmp/chatlab-at-d95fa1a && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/library/generators/06_archive_d95fa1a.py \
        <repo>/tests/fixtures/saved/library/06-archive-d95fa1a/conversations.json

The picture is stored with that commit's ``attachments.store_image`` and the
pane written by that commit's ``library.write``.
"""

import os
import struct
import sys
import zlib
from pathlib import Path

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
os.environ["CHATLAB_LIBRARY_PATH"] = str(target)

from chatlab import attachments, library  # noqa: E402


def png(width, height, rgb):
    """A small, byte-for-byte reproducible PNG."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


picture = attachments.store_image(png(2, 2, (255, 0, 0)))

main = [
    {"role": "user", "content": "What is in this picture?", "reasoning": "", "images": [picture]},
    {"role": "assistant", "content": "A red square.", "reasoning": "",
     "model": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt_tokens": 90, "generated_tokens": 5},
]
fork = [*main, {"role": "user", "content": "Which shade of red?", "reasoning": ""},
        {"role": "assistant", "content": "Pure red, #FF0000.", "reasoning": "",
         "model": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt_tokens": 110, "generated_tokens": 9}]
archived_chat = [
    {"role": "user", "content": "Old question I put away", "reasoning": ""},
    {"role": "assistant", "content": "An answer kept in the archive.", "reasoning": "",
     "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 15, "generated_tokens": 7},
]
restored_chat = [
    {"role": "user", "content": "Archived and brought back", "reasoning": ""},
    {"role": "assistant", "content": "Back in the list.", "reasoning": "",
     "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 14, "generated_tokens": 5},
]
failed_chat = [
    {"role": "user", "content": "Write a long essay.", "reasoning": ""},
    {"role": "assistant", "content": "Essays begin with", "reasoning": "Plan: intro, body.",
     "error": "MPS backend out of memory", "model": "Qwen/Qwen3-0.6B"},
]

forks = {
    "active": "Chat 1",
    "branches": {"Main": main, "Fork 1": fork, "Chat 1": archived_chat,
                 "Chat 2": restored_chat, "Chat 3": failed_chat},
    "origins": {"Fork 1": {"parent": "Main", "kind": "message", "turn": 1, "part": "content",
                           "message_count": 2}},
    "sampling": {"Chat 1": {"temperature": 0.3}},
    "sampling_updated": {"Chat 1": "2026-10-05T10:00:00.000010+00:00"},
    "archived": {"Chat 1": True, "Fork 1": True},
    "archived_updated": {
        "Chat 1": "2026-10-05T11:00:00.000020+00:00",
        "Fork 1": "2026-10-05T11:05:00.000030+00:00",
        "Chat 2": "2026-10-05T11:10:00.000040+00:00",
    },
    "updated": {
        "Main": "2026-10-05T09:00:00.000001+00:00",
        "Fork 1": "2026-10-05T09:01:00.000002+00:00",
        "Chat 1": "2026-10-05T09:02:00.000003+00:00",
        "Chat 2": "2026-10-05T09:03:00.000004+00:00",
        "Chat 3": "2026-10-05T09:04:00.000005+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
