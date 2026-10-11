"""Write directions-sparse-unnormalized.json, a chatlab-directions-1 file as a script writes one.

Format marker: chatlab-directions-1
Era: the only one; the format arrived with the Direction edits extension.
Generating commit: 390252b "Add a Direction edits extension (#196)".

Commands:
    git worktree add --detach /tmp/chatlab-at-390252b 390252b
    cd /tmp/chatlab-at-390252b && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

ChatLab has no writer for this format: DIRECTION_EDITS.md describes it as
written by hand or by a script, so this is such a script, following that
commit's field table. It is checked with that commit's own
``files.normalize_directions`` before writing. It exercises what the reader
does to a file: a name with surrounding spaces (stripped), directions listed
out of block order (sorted), block 3 left out (blank in the heatmaps), vectors
of integers and of several lengths, none unit (each scaled to unit length), a
40-character revision and a precision.
"""
import json
import sys
from pathlib import Path

import numpy as np

from chatlab.extensions.direction_edits import files

assert Path(files.__file__).resolve().is_relative_to(Path.cwd().resolve()), files.__file__

WIDTH = 16
rng = np.random.default_rng(404)
rows = {layer: [round(float(x), 6) for x in rng.normal(size=WIDTH) * scale]
        for layer, scale in ((0, 3.0), (1, 0.5), (2, 12.0))}
rows[0][0] = 7  # an integer entry is a number too
value = {
    "format": "chatlab-directions-1",
    "name": "  dog detector  ",
    "model_id": "test/tiny-decoder",
    "model_revision": "fedcba9876543210fedcba9876543210fedcba98",
    "precision": "full",
    "directions": [{"layer": 2, "vector": rows[2]}, {"layer": 0, "vector": rows[0]},
                   {"layer": 1, "vector": rows[1]}],
}
files.normalize_directions(value)
Path(sys.argv[1]).write_text(json.dumps(value, indent=2) + "\n")
print(sys.argv[1])
