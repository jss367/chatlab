"""Generate steering/01-hand-written-392ad4d.json.

Format: chatlab-steering-1 as a person writes it by hand, the way the README
of 392ad4d "Add per-conversation steering vectors (#59)" - the commit that
introduced vector import - describes it: ``format``, ``strength`` and
``enabled`` are optional (their defaults are chatlab-steering-1, 1.0 and
true), so this file leaves all three out and writes the vector entries as a
mix of integers and decimals, with the model ID padded with spaces.

No ChatLab code writes this shape; it is the user-authored input that
version accepted, produced here with plain ``json.dumps``. The generator was
run inside a 392ad4d checkout and checks that 392ad4d's own
``steering.read_vector`` accepts the bytes and what it reads them as.

Commands used:

    git worktree add --detach /tmp/chatlab-at-392ad4d 392ad4d
    cd /tmp/chatlab-at-392ad4d && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/steering/generators/hand_written_392ad4d.py \
        <repo>/tests/fixtures/saved/steering/01-hand-written-392ad4d.json
"""

import json
import sys
from pathlib import Path

import steering  # top-level module at this commit

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
value = {"model_id": "  allenai/Olmo-3-7B-Think ", "layer": 0, "vector": [1, -2, 0.5, 0, 3.25]}
target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

read = steering.read_vector(str(target))
assert read == {
    "format": "chatlab-steering-1", "model_id": "allenai/Olmo-3-7B-Think", "layer": 0,
    "strength": 1.0, "enabled": True, "vector": [1.0, -2.0, 0.5, 0.0, 3.25],
}, read
print(target)
