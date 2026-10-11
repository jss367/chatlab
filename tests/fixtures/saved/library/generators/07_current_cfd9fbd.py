"""Generate library/07-current-cfd9fbd/ (conversations.json, conversations.json.archive,
conversations-vectors/, conversations-images/).

Format: chatlab-library-1, current era: everything the writer knows. Archive
state is written natively and also as the ``_chatlab_archive``
wire-compatibility record in sampling (with the wire ``sampling_updated``
advanced to the archive stamp), the private ``conversations.json.archive``
companion records each archive stamp, fork origins are written natively and as
``_chatlab_fork_origin``, and turns carry images, errors, steering, generation
settings, token-step pauses and Institutions provenance.
Generating commit: cfd9fbd "Release ChatLab 0.40.0" (library writer last
changed in 17c5fbc, #206).

Commands used (from a checkout of cfd9fbd):

    PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        tests/fixtures/saved/library/generators/07_current_cfd9fbd.py \
        tests/fixtures/saved/library/07-current-cfd9fbd/conversations.json

The pane is built as the app's state dictionary and written by
``chatlab.library.write``.
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


blue = attachments.store_image(png(3, 2, (0, 0, 255)))
green = attachments.store_image(png(2, 3, (0, 255, 0)))

STEER = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-4B", "layer": 12,
    "vector": [0.75, -1.25, 0.5, 2.0, -0.125], "strength": -6.5, "enabled": True,
}
SETTINGS = {
    "temperature": 0.8, "top_p": 0.9, "top_k": 30, "skip_top_below": 0.02,
    "max_new_tokens": 300, "seed": 77, "system_prompt": "You are terse.",
    "keep_reasoning": True, "assistant_prefill": "Sure:", "thinking_mode": "on",
}
PROVENANCE = {
    "run_id": "run-0001", "game_id": "game-7", "turn_id": "turn-3",
    "model_profile": {
        "parent": {"repo": "Qwen/Qwen3-4B", "revision": "abc123"},
        "adapter": {"repo": "example/support-lora", "subfolder": "step-200", "revision": "def456"},
    },
    "accepted": True, "error": None,
    "note": "Recorded profile only; opening Chat does not load these weights or adapter. Sampling was batched.",
}

main = [
    {"role": "user", "content": "Compare these two pictures.", "reasoning": "", "images": [blue, green]},
    {"role": "assistant", "content": "One is blue and wide, the other green and tall.",
     "reasoning": "Look at colour and aspect ratio.", "model": "Qwen/Qwen3-4B",
     "prompt_tokens": 400, "generated_tokens": 30, "thinking_mode": "on",
     "generation_settings": SETTINGS, "steering": STEER},
]
fork = [
    *main,
    {"role": "user", "content": "Which is larger?", "reasoning": ""},
    {"role": "assistant", "content": "", "reasoning": "", "token_step_paused": True,
     "model": "Qwen/Qwen3-4B", "thinking_mode": "on"},
]
failed = [
    {"role": "user", "content": "Summarise War and Peace.", "reasoning": ""},
    {"role": "assistant", "content": "", "reasoning": "", "error": "The model ran out of memory.",
     "model": "Qwen/Qwen3-4B"},
]
replay = [
    {"role": "user", "content": "My order arrived broken.", "reasoning": "",
     "institutions_provenance": PROVENANCE},
    {"role": "assistant", "content": "{\"action\": \"refund\"}", "reasoning": "Policy allows refunds.",
     "model": "Qwen/Qwen3-4B + adapter example/support-lora", "institutions_provenance": PROVENANCE},
]
archived = [
    {"role": "user", "content": "Archived conversation", "reasoning": ""},
    {"role": "assistant", "content": "Still here, just put away.", "reasoning": "",
     "model": "Qwen/Qwen3-4B", "prompt_tokens": 12, "generated_tokens": 6},
]
restored = [
    {"role": "user", "content": "Archived, then restored", "reasoning": ""},
    {"role": "assistant", "content": "Back again.", "reasoning": "", "model": "Qwen/Qwen3-4B"},
]

forks = {
    "active": "Fork 1",
    "branches": {"Main": main, "Fork 1": fork, "Chat 1": failed, "Chat 2": replay,
                 "Chat 3": archived, "Chat 4": restored},
    "origins": {"Fork 1": {"parent": "Main", "kind": "message", "turn": 1, "part": "content",
                           "message_count": 2}},
    "sampling": {
        "Main": {"temperature": 0.8, "top_p": 0.9, "top_k": 30, "skip_top_below": 0.02,
                 "max_new_tokens": 300, "steering": STEER, "future_knob": "kept"},
        "Fork 1": {"temperature": 0.8, "top_p": 0.9, "top_k": 30, "skip_top_below": 0.02,
                   "max_new_tokens": 300},
        "Chat 3": {"temperature": 1.2},
    },
    "sampling_updated": {
        "Main": "2026-10-10T10:00:00.000010+00:00",
        "Fork 1": "2026-10-10T10:01:00.000020+00:00",
        "Chat 3": "2026-10-10T10:02:00.000030+00:00",
    },
    "archived": {"Chat 3": True},
    "archived_updated": {
        "Chat 3": "2026-10-10T12:00:00.000040+00:00",
        "Chat 4": "2026-10-10T12:30:00.000050+00:00",
    },
    "updated": {
        "Main": "2026-10-10T09:00:00.000001+00:00",
        "Fork 1": "2026-10-10T09:01:00.000002+00:00",
        "Chat 1": "2026-10-10T09:02:00.000003+00:00",
        "Chat 2": "2026-10-10T09:03:00.000004+00:00",
        "Chat 3": "2026-10-10T09:04:00.000005+00:00",
        "Chat 4": "2026-10-10T09:05:00.000006+00:00",
        "Chat 5": "2026-10-10T09:06:00.000007+00:00",
    },
}

written = library.write(forks, target)
assert written == target, written
print(target)
