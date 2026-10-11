"""Generate conversation/03-pictures-b69af3b.json.

Format: chatlab-conversation-1, pictures era: user turns name the pictures
pasted into them and the file embeds each picture once, base64, under
``images``. By this commit replies also carry ``thinking_mode``,
``generation_settings``, ``token_step_paused`` and a failed reply's ``error``,
and steering travels under ``steering_vectors``.
Generating commit: b69af3b "Send pictures to vision models: paste, drop or
attach a screenshot (#176)".

Commands used:

    git worktree add --detach /tmp/chatlab-at-b69af3b b69af3b
    cd /tmp/chatlab-at-b69af3b && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/conversation/generators/03_pictures_b69af3b.py \
        <repo>/tests/fixtures/saved/conversation/03-pictures-b69af3b.json

Pictures are stored with that commit's ``attachments.store_image`` and the
file written as its ``save_conversation`` wrote it:
``write_private_text(path, to_json(turns, system_prompt=..., steering=...))``,
with the library (and so the picture and vector stores) in a scratch directory.
"""

import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

scratch = tempfile.mkdtemp()
os.environ["CHATLAB_LIBRARY_PATH"] = str(Path(scratch) / "conversations.json")

from chatlab import attachments  # noqa: E402
from chatlab.conversation import to_json  # noqa: E402
from chatlab.trace_export import write_private_text  # noqa: E402

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)


def png(width, height, rgb):
    """A small, byte-for-byte reproducible PNG."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


yellow = attachments.store_image(png(4, 1, (255, 255, 0)))
purple = attachments.store_image(png(1, 4, (128, 0, 128)))

STEER = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "layer": 3,
    "vector": [-0.5, 0.5, 0.25, -0.25], "strength": 2.0, "enabled": True,
}
SETTINGS = {
    "temperature": 0.4, "top_p": 0.8, "top_k": 10, "skip_top_below": 0.0,
    "max_new_tokens": 64, "seed": 9, "system_prompt": "Describe pictures.",
    "keep_reasoning": False, "assistant_prefill": "", "thinking_mode": "default",
}

turns = [
    {"role": "user", "content": "What colours are these?", "reasoning": "", "images": [yellow, purple]},
    {"role": "assistant", "content": "Yellow and purple.", "reasoning": "",
     "model": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt_tokens": 210, "generated_tokens": 6,
     "thinking_mode": "default", "generation_settings": SETTINGS, "steering": STEER},
    {"role": "user", "content": "Same picture again, which is taller?", "reasoning": "", "images": [purple]},
    {"role": "assistant", "content": "The purple one", "reasoning": "",
     "error": "Generation stopped: the device ran out of memory.",
     "model": "Qwen/Qwen2.5-VL-3B-Instruct", "thinking_mode": "default",
     "generation_settings": dict(SETTINGS, seed=10)},
    {"role": "user", "content": "Try once more.", "reasoning": ""},
    {"role": "assistant", "content": "", "reasoning": "", "token_step_paused": True,
     "model": "Qwen/Qwen2.5-VL-3B-Instruct"},
]

write_private_text(target, to_json(turns, system_prompt="Describe pictures.", steering=STEER))
print(target)
