"""Generate conversation/02-steering-392ad4d.json.

Format: chatlab-conversation-1, steering era: replies carry their model and
token counts, the conversation's steering vector is saved at the top level and
a steered reply carries its own, both as ``chatlab-steering-reference-1``
records, and the vectors travel once each under ``steering_vectors``, keyed by
their SHA-256.
Generating commit: 392ad4d "Add per-conversation steering vectors (#59)".

Commands used:

    git worktree add --detach /tmp/chatlab-at-392ad4d 392ad4d
    cd /tmp/chatlab-at-392ad4d && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/conversation/generators/02_steering_392ad4d.py \
        <repo>/tests/fixtures/saved/conversation/02-steering-392ad4d.json

Written as that commit's ``save_conversation`` wrote it:
``write_private_text(path, to_json(turns, system_prompt=..., steering=...))``,
with the library (and so the vector store) pointed at a scratch directory.
"""

import os
import sys
import tempfile
from pathlib import Path

scratch = tempfile.mkdtemp()
os.environ["CHATLAB_LIBRARY_PATH"] = str(Path(scratch) / "conversations.json")

from conversation import to_json  # noqa: E402  (top-level module at this commit)
from trace_export import write_private_text  # noqa: E402

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)

CONVERSATION_STEERING = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 4,
    "vector": [0.25, -0.75, 1.5, 0.0, -2.0], "strength": 8.0, "enabled": True,
}
REPLY_STEERING = {
    "format": "chatlab-steering-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 1,
    "vector": [1.0, 2.0, 3.0], "strength": -1.5, "enabled": False,
}

turns = [
    {"role": "user", "content": "Describe the sea.", "reasoning": ""},
    {"role": "assistant", "content": "Grey, restless, endless.", "reasoning": "Three adjectives.",
     "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 25, "generated_tokens": 14,
     "steering": REPLY_STEERING},
    {"role": "user", "content": "And the sky?", "reasoning": ""},
    {"role": "assistant", "content": "Bright and calm.", "reasoning": "",
     "model": "Qwen/Qwen3-0.6B", "prompt_tokens": 52, "generated_tokens": 5,
     "steering": CONVERSATION_STEERING},
]

write_private_text(target, to_json(turns, system_prompt="Be poetic.", steering=CONVERSATION_STEERING))
print(target)
