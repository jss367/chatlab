"""Generate steering/02-extracted-be75ba8.json.

Format: chatlab-steering-1, as the Steering panel's "Download vector" writes
an extracted direction: the difference in means between two sets of example
activations at the chosen layer, with ``format``, ``model_id``, ``layer``,
``vector``, ``strength`` and ``enabled`` all spelled out.
Generating commit: be75ba8 "Extract steering vectors from examples, and
compare two runs (#102)", which introduced extraction and the download.
(Run again from cfd9fbd, the same script wrote identical bytes, so no second
file was kept.)

Commands used:

    git worktree add --detach /tmp/chatlab-at-be75ba8 be75ba8
    cd /tmp/chatlab-at-be75ba8 && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/steering/generators/extracted.py \
        <repo>/tests/fixtures/saved/steering/02-extracted-be75ba8.json

The extraction is that commit's ``steering.contrast_directions`` over fixed
example activations, and the file is what that commit's
``ui.steering.download_extracted`` writes for layer 1, copied to the target.
"""

import shutil
import sys
from pathlib import Path

try:  # cfd9fbd: package layout
    from chatlab import steering
    from chatlab.ui import steering as panel
except ImportError:  # be75ba8: top-level modules
    import steering
    from ui import steering as panel

target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)

# Three positive and two negative examples, each read at three layers of a
# four-wide model: rows of [layer][width].
positive = [
    [[1.0, 0.0, 0.5, 0.0], [2.0, -1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]],
    [[1.5, 0.5, 0.5, 0.0], [2.5, -0.5, 0.5, 0.5], [0.5, 0.0, 1.0, 1.5]],
    [[0.5, 0.0, 1.0, 0.5], [1.5, -1.5, 0.0, 1.0], [0.0, 0.5, 1.5, 1.0]],
]
negative = [
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5]],
    [[0.5, 1.0, 0.5, 0.5], [-0.5, 0.5, 0.5, 0.0], [0.5, 0.0, 0.5, 0.5]],
]
directions, stats = steering.contrast_directions(positive, negative)
extraction = {"model_id": "Qwen/Qwen3-0.6B", "layers": directions, "stats": stats}

written = panel.download_extracted(extraction, 1)
shutil.copyfile(written, target)
print(target)
