"""Write probe-era2-revision-paired.json with the probe writer of bf31e2f.

Format marker: chatlab-probe-1
Era: probes pinned to a model revision (``model_revision`` written), before
``precision`` existed.
Generating commit: bf31e2f "Pin probes to the model revision; refuse an
overflowed reading" (PR #179).

Commands:
    git worktree add --detach /tmp/chatlab-at-bf31e2f bf31e2f
    cd /tmp/chatlab-at-bf31e2f && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

Paired examples trained as pairs, pooled over every token, read as plain
text. The activations are synthetic and seeded; the fit, the probe JSON and
its bytes are that commit's own ``train``, ``build`` and ``dumps``. Only the
id and the creation time are pinned.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chatlab.extensions.probes import probe as probes

assert Path(probes.__file__).resolve().is_relative_to(Path.cwd().resolve()), probes.__file__

probes.uuid4 = lambda: SimpleNamespace(hex="2b0b5eed0000000000000000000000b2")
probes.time = SimpleNamespace(time=lambda: 1790942400.25)

rng = np.random.default_rng(202)
LAYERS, WIDTH, PAIRS = 4, 12, 5
base = rng.normal(size=(PAIRS, LAYERS, WIDTH))
signal = np.zeros((LAYERS, WIDTH))
signal[2, 3] = 1.5
signal[3, 3] = 0.6
positive = base + signal + rng.normal(scale=0.1, size=base.shape)
negative = base - signal + rng.normal(scale=0.1, size=base.shape)

layers, folds = probes.train(positive, negative, l2=2.0, paired=True)
probe = probes.build(
    name="Past vs present tense", model_id="test/probe-era2",
    model_revision="0123456789abcdef0123456789abcdef01234567",
    positive_label="Past", negative_label="Present",
    positive_examples=["She walked home.", "They sang loudly.", "He wrote a letter.", "We ate early.",
                       "It rained all day."],
    negative_examples=["She walks home.", "They sing loudly.", "He writes a letter.", "We eat early.",
                       "It rains all day."],
    pool="mean", chat_template=False, l2=2.0, layers=layers, folds=folds, paired=True)
Path(sys.argv[1]).write_text(probes.dumps(probe))
print(sys.argv[1], folds, probe["best_layer"], [item["heldout_accuracy"] for item in probe["layers"]])
