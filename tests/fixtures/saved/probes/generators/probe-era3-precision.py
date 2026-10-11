"""Write probe-era3-precision.json with the probe writer of ce866c2.

Format marker: chatlab-probe-1
Era: probes that record the precision they were trained at (``model_revision``
and ``precision`` both written), the shape today's writer still produces.
Generating commit: ce866c2 "Fit the reply to what the prompt leaves of one
reading; note a probe read at another precision" (PR #179).

Commands:
    git worktree add --detach /tmp/chatlab-at-ce866c2 ce866c2
    cd /tmp/chatlab-at-ce866c2 && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

Made for the tests' tiny decoder (``test/tiny-decoder``, 4 blocks of 16) so the
Direction edits fixtures can take its weights as their directions. Unequal
sides (6 against 3, so 3 folds), unpaired, last token, as a user message.
The activations are synthetic and seeded; the fit, the probe JSON and its
bytes are that commit's own ``train``, ``build`` and ``dumps``. Only the id and
the creation time are pinned.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chatlab.extensions.probes import probe as probes

assert Path(probes.__file__).resolve().is_relative_to(Path.cwd().resolve()), probes.__file__

probes.uuid4 = lambda: SimpleNamespace(hex="3c0ffee0000000000000000000000c3d")
probes.time = SimpleNamespace(time=lambda: 1790949600.75)

rng = np.random.default_rng(303)
LAYERS, WIDTH = 4, 16
positive = rng.normal(size=(6, LAYERS, WIDTH))
negative = rng.normal(size=(3, LAYERS, WIDTH))
positive[:, 1:, 5] += 2.0
negative[:, 1:, 5] -= 2.0
positive[:, 2, 9] += 1.0

layers, folds = probes.train(positive, negative, l2=1.0, paired=False)
probe = probes.build(
    name="Mentions an animal", model_id="test/tiny-decoder",
    model_revision="fedcba9876543210fedcba9876543210fedcba98", precision="8-bit",
    positive_label="Animal", negative_label="No animal",
    positive_examples=["the cat sat on the mat", "the dog sat on the log", "a quick brown fox",
                       "the lazy dog", "a cat and a dog", "the fox jumps"],
    negative_examples=["hello world", "tokens, merges and bytes", "hello again"],
    pool="last", chat_template=True, l2=1.0, layers=layers, folds=folds, paired=False)
Path(sys.argv[1]).write_text(probes.dumps(probe))
print(sys.argv[1], folds, probe["best_layer"], [item["heldout_accuracy"] for item in probe["layers"]])
