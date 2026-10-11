"""Generate conversation/04-current-cfd9fbd.json.

Format: chatlab-conversation-1, current era: everything the pictures era
carries plus ``institutions_provenance`` on turns opened from an Institutions
replay and then saved from Chat.
Generating commit: cfd9fbd "Release ChatLab 0.40.0" (conversation writer last
changed for this field in 90e52a1, #215).

Commands used (from a checkout of cfd9fbd):

    PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        tests/fixtures/saved/conversation/generators/04_current_cfd9fbd.py \
        tests/fixtures/saved/conversation/04-current-cfd9fbd.json

Written as ``save_conversation`` writes it:
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
from chatlab.files import write_private_text  # noqa: E402

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


receipt = attachments.store_image(png(3, 3, (250, 250, 250)))

PROVENANCE = {
    "run_id": "support-run-42", "game_id": "game-3", "turn_id": "turn-1",
    "model_profile": {
        "parent": {"repo": "Qwen/Qwen3-4B", "revision": "0123abcd"},
        "adapter": {"repo": "example/support-lora", "subfolder": "checkpoint-50", "revision": "4567ef01"},
    },
    "accepted": False, "error": "reply was not valid JSON",
    "note": "Recorded profile only; opening Chat does not load these weights or adapter. Sampling was batched.",
}
STEER = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-4B", "layer": 20,
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "strength": 12.5, "enabled": False,
}
SETTINGS = {
    "temperature": 1.0, "top_p": 1.0, "top_k": 0, "skip_top_below": 0.1,
    "max_new_tokens": 2048, "seed": 31337, "system_prompt": "You are a support agent. Reply in JSON.",
    "keep_reasoning": True, "assistant_prefill": "{", "thinking_mode": "on",
}

turns = [
    {"role": "user", "content": "My parcel arrived damaged; here is the receipt.", "reasoning": "",
     "images": [receipt], "institutions_provenance": PROVENANCE},
    {"role": "assistant", "content": "{\"action\": \"escalate\", \"reason\": \"damage\"}",
     "reasoning": "Damage claims over $50 must be escalated.",
     "model": "Qwen/Qwen3-4B + adapter example/support-lora",
     "prompt_tokens": 812, "generated_tokens": 97, "thinking_mode": "on",
     "generation_settings": SETTINGS, "steering": STEER,
     "institutions_provenance": dict(PROVENANCE, recorded_message_provenance={"source": "replay"})},
    {"role": "user", "content": "Can you refund instead?", "reasoning": ""},
    {"role": "assistant", "content": "", "reasoning": "Checking the refund policy",
     "error": "Generation stopped by the reader.", "model": "Qwen/Qwen3-4B"},
]

write_private_text(target, to_json(turns, system_prompt="You are a support agent. Reply in JSON.", steering=STEER))
print(target)
