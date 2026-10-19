"""Freeze the first steering comparison writer, commit 072f85c.

Run from its detached checkout (using the project's Python environment):
  git worktree add --detach /tmp/chatlab-steering-at-072f85c 072f85c
  cd /tmp/chatlab-steering-at-072f85c
  /path/to/chatlab/.venv/bin/python /path/to/this/paired_ranks.py OUTPUT.json

Writes through that version's ComparisonRuns publication/download path.
Never regenerate the committed fixture. Model weights and lens matrices use fixed seeds;
examples are synthetic; no downloads or user data.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from chatlab.extension_api import ModelService  # noqa: E402
from chatlab.extensions.steering_comparison.experiment import run  # noqa: E402
from chatlab.extensions.steering_comparison.page import ComparisonRuns  # noqa: E402
from torch_support import tiny_manager  # noqa: E402


def main(output):
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        manager = tiny_manager(seed=7, layers=3, heads=2)
        torch.manual_seed(8)
        lens = folder / 'lens.pt'
        torch.save({'J': {b: torch.randn(16, 16) for b in (0, 1, 2)},
                    'source_layers': [0, 1, 2], 'n_prompts': 10, 'd_model': 16}, lens)
        manager.import_jacobian_lens(str(lens), manager.model_id)
        paths = []
        for name, strength in [('compassionate', 1), ('opposite sign', -1)]:
            path = folder / f'{name}.json'
            path.write_text(json.dumps({'format': 'chatlab-steering-1', 'model_id': manager.model_id,
                                        'layer': 0, 'strength': strength, 'vector': [0.1] * 16}))
            paths.append(str(path))
        with ModelService(lambda: manager).open_session() as session:
            steps = run(session, [['Return to task', 'the cat', 1], ['Return to task', 'the mat', 0],
                                  ['End of turn', 'the cat sat', '']], paths, '0,2', seed=7)
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    result = done.value
                    break
        runs = ComparisonRuns()
        turn = runs.start('fixture')
        try:
            runs.publish('fixture', turn, result, lambda path: shutil.copyfile(path, output))
        finally:
            runs.forget('fixture')
        print(json.dumps({'numpy': np.__version__, 'torch': torch.__version__, 'output': str(output)}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
