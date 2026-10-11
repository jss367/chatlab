"""Write probe-era1-no-revision-below-chance.json with the probe writer of 22f2efd.

Format marker: chatlab-probe-1
Era: the first probe files (no ``model_revision``, no ``precision`` key).
Generating commit: 22f2efd "Add a Linear probes extension: fit a logistic probe
at every layer, read replies token by token" (PR #179).

Commands:
    git worktree add --detach /tmp/chatlab-at-22f2efd 22f2efd
    cd /tmp/chatlab-at-22f2efd && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

The examples are near-twins trained without ``paired``, the case the page
warns about: every layer comes out below chance on held-out examples. The
activations are synthetic and seeded; the fit, the probe JSON and its bytes are
that commit's own ``train``, ``build`` and ``dumps``. Only the id and the
creation time are pinned.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chatlab.extensions.probes import probe as probes

assert Path(probes.__file__).resolve().is_relative_to(Path.cwd().resolve()), probes.__file__

probes.uuid4 = lambda: SimpleNamespace(hex="1e4a0c0ffee0000000000000000000a1")
probes.time = SimpleNamespace(time=lambda: 1790935212.345678)

rng = np.random.default_rng(101)
LAYERS, WIDTH, PAIRS = 3, 8, 4
base = rng.normal(size=(PAIRS, LAYERS, WIDTH)) * 2.0
signal = np.zeros((LAYERS, WIDTH))
signal[:, 0] = 0.25
positive = base + signal
negative = base - signal

layers, folds = probes.train(positive, negative, l2=0.5, paired=False)
probe = probes.build(
    name="Formal register — twins, unpaired", model_id="test/probe-era1",
    positive_label="Formal", negative_label="Casual",
    positive_examples=["Good afternoon, how may I help?", "I would be grateful for a reply.",
                       "Please find the report attached.", "    Kindly confirm receipt."],
    negative_examples=["hey, what's up?", "thx for replying", "report's attached", "    got it?"],
    pool="last", chat_template=True, l2=0.5, layers=layers, folds=folds, paired=False)
Path(sys.argv[1]).write_text(probes.dumps(probe))
print(sys.argv[1], folds, probe["best_layer"], [item["heldout_accuracy"] for item in probe["layers"]])
