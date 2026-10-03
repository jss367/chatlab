"""Linear probes: a logistic regression at every decoder block, and what it reads.

Nothing here touches the model. The page reads the examples through the host
and hands the arrays in; a fitted probe is plain JSON that a reading of any
passage is scored against.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

FORMAT = "chatlab-probe-1"
# A 7B probe holds 32 rows of 4,096 weights, about 2.5 MB written out.
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_COEFFICIENTS = 1024 * 1024
# Each example is a forward pass and a row of every block's output held until
# the fit, as for a steering extraction.
MAX_EXAMPLES = 64
# Held-out accuracy is measured over this many folds, or fewer when a side
# has fewer examples than this.
FOLDS = 5
DEFAULT_L2 = 1.0
POOLS = ("last", "mean")
PRECISIONS = ("full", "8-bit", "4-bit")
NEWTON_STEPS = 100
# The first second of the year 3000, comfortably inside what a date can show.
LATEST_CREATED = 32503680000.0
FLOAT32_MAX = float(np.finfo(np.float32).max)


def _sigmoid(values):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(values, dtype=np.float64)))


def _log_loss(probabilities, labels):
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _newton(features, labels, l2):
    """L2-penalized logistic regression by Newton's method; the intercept is unpenalized."""
    count, width = features.shape
    design = np.hstack([features, np.ones((count, 1))])
    penalty = np.full(width + 1, float(l2))
    penalty[-1] = 0.0
    theta = np.zeros(width + 1)
    for _ in range(NEWTON_STEPS):
        probabilities = _sigmoid(design @ theta)
        gradient = design.T @ (probabilities - labels) + penalty * theta
        curvature = probabilities * (1 - probabilities)
        hessian = (design * curvature[:, None]).T @ design + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return theta[:-1], float(theta[-1])


def fit(activations, labels, l2=DEFAULT_L2):
    """One layer's probe, as raw-space weights and a bias.

    Each feature is standardized first, so the penalty treats every direction
    in the residual stream alike rather than favouring the few with large
    activations. With fewer examples than features the penalized solution
    lies in the span of the examples, so the fit is made on their singular
    vectors: an exact answer from a problem the size of the example count.
    The standardization is folded back into the weights, so a probability is
    ``sigmoid(weights · x + bias)`` for a raw block output ``x``.
    """
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mean = activations.mean(axis=0)
    spread = activations.std(axis=0)
    spread[spread == 0] = 1.0
    standardized = (activations - mean) / spread
    left, singular, right = np.linalg.svd(standardized, full_matrices=False)
    keep = singular > singular.max(initial=0.0) * 1e-10
    coefficients, bias = _newton(left[:, keep] * singular[keep], labels, l2)
    weights = (right[keep].T @ coefficients) / spread
    return weights, bias - float(weights @ mean)


def folds_for(labels, folds=FOLDS, *, paired=False):
    """A fold number for each example, with both sides dealt across every fold.

    ``paired`` says example ``n`` of one side is example ``n`` of the other
    with the property changed, and keeps each pair in one fold. Split up, a
    held-out example's twin sits in training with the opposite label, which
    is the nearest thing to it there, and the measured accuracy falls below
    chance however well the probe generalizes.

    Fixed by a seed so training the same examples twice gives the same table.
    ``None`` when a side has a single example and nothing can be held out.
    """
    labels = np.asarray(labels)
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    count = min(folds, len(positives), len(negatives))
    if count < 2:
        return None
    rng = np.random.default_rng(0)
    assignment = np.empty(len(labels), dtype=int)
    if paired:
        if len(positives) != len(negatives):
            raise ValueError("Paired examples need the same number on each side.")
        order = rng.permutation(len(positives)) % count
        assignment[positives], assignment[negatives] = order, order
        return assignment
    for members in (negatives, positives):
        members = members.copy()
        rng.shuffle(members)
        assignment[members] = np.arange(len(members)) % count
    return assignment


def train(positive, negative, *, l2=DEFAULT_L2, paired=False):
    """A probe at every block, with how well each one held up on examples it did not see.

    ``positive`` and ``negative`` are shaped ``(examples, blocks, width)``.
    Each block's probe is fitted on every example; its held-out accuracy and
    loss come from refitting with each fold left out and scoring that fold,
    which is the number to choose a layer by; ``paired`` is as for
    :func:`folds_for`. Training accuracy is reported
    beside it because with more features than examples it is nearly always
    perfect, and the gap between the two is the overfitting.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.ndim != 3 or negative.ndim != 3 or positive.shape[1:] != negative.shape[1:]:
        raise ValueError("Every example must be read at the same layers and width.")
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("Give at least two examples on each side, so some can be held out.")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("The L2 strength must be a positive number.")
    activations = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    assignment = folds_for(labels, paired=paired)
    fold_count = int(assignment.max()) + 1
    layers = []
    for layer in range(activations.shape[1]):
        rows = activations[:, layer]
        weights, bias = fit(rows, labels, l2)
        trained = _sigmoid(rows @ weights + bias)
        held = np.empty(len(labels))
        for fold in range(fold_count):
            out = assignment == fold
            fold_weights, fold_bias = fit(rows[~out], labels[~out], l2)
            held[out] = _sigmoid(rows[out] @ fold_weights + fold_bias)
        layers.append({
            "layer": layer,
            "weights": weights,
            "bias": bias,
            "train_accuracy": float(np.mean((trained > 0.5) == labels)),
            "heldout_accuracy": float(np.mean((held > 0.5) == labels)),
            "heldout_loss": _log_loss(held, labels),
        })
    return layers, fold_count


def best_layer(layers):
    """The block whose probe was most accurate on held-out examples, the lower loss breaking a tie."""
    return int(max(layers, key=lambda item: (item["heldout_accuracy"], -item["heldout_loss"]))["layer"])


def _short(value):
    # Nine significant digits keep float32 precision and halve the file.
    return float(f"{float(value):.9g}")


def build(*, name, model_id, model_revision=None, precision=None, positive_label, negative_label, positive_examples, negative_examples,
          pool, chat_template, l2, layers, folds, paired=False):
    """A fitted probe as the JSON it is saved and exported as."""
    probe = {
        "format": FORMAT,
        "id": uuid4().hex,
        "name": name,
        "model_id": model_id,
        "model_revision": model_revision,
        "precision": precision,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "pool": pool,
        "chat_template": bool(chat_template),
        "l2": float(l2),
        "folds": int(folds),
        "paired": bool(paired),
        "created": time.time(),
        "examples": {"positive": list(positive_examples), "negative": list(negative_examples)},
        "layers": [dict(item, weights=[_short(v) for v in item["weights"]], bias=_short(item["bias"]))
                   for item in layers],
    }
    probe["best_layer"] = best_layer(probe["layers"])
    return normalize(probe)


def normalize(value):
    """Check a probe read from anywhere and return a clean copy, or raise ``ValueError``."""
    def text(field, limit=200):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or len(item) > limit:
            raise ValueError(f"The probe's {field.replace('_', ' ')} must be text of at most {limit} characters.")
        return item.strip()

    def number(item, what):
        if type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(f"The probe's {what} must be a finite number.")
        return float(item)

    def weight(item):
        # Passages are read along the weights in float32, so they must fit it.
        item = number(item, "weights")
        if abs(item) > FLOAT32_MAX:
            raise ValueError("The probe's weights must fit in 32-bit floats.")
        return item

    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"Expected a {FORMAT} JSON object.")
    probe_id = value.get("id")
    if not isinstance(probe_id, str) or not re.fullmatch(r"[0-9a-f]{32}", probe_id):
        raise ValueError("The probe's id must be 32 hexadecimal characters.")
    positive_label, negative_label = text("positive_label", 60), text("negative_label", 60)
    if positive_label == negative_label:
        raise ValueError("The two sides need different labels.")
    if value.get("pool") not in POOLS:
        raise ValueError("The probe's pool must be last or mean.")
    if type(value.get("chat_template")) is not bool or type(value.get("paired", False)) is not bool:
        raise ValueError("The probe's chat_template and paired must be true or false.")
    examples = value.get("examples")
    if not isinstance(examples, dict) or not all(
            isinstance(examples.get(side), list) and 2 <= len(examples[side]) <= MAX_EXAMPLES
            and all(isinstance(e, str) and len(e) <= 32768 for e in examples[side])
            for side in ("positive", "negative")):
        raise ValueError("The probe needs 2–64 examples per side, each at most 32768 characters.")
    layers = value.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 256:
        raise ValueError("The probe needs between 1 and 256 layers.")
    if sum(len(item.get("weights", [])) for item in layers
           if isinstance(item, dict) and isinstance(item.get("weights"), list)) > MAX_COEFFICIENTS:
        raise ValueError("The probe exceeds the total coefficient limit.")
    width = None
    clean = []
    for index, item in enumerate(layers):
        if not isinstance(item, dict) or item.get("layer") != index:
            raise ValueError("The probe's layers must run 0, 1, 2 … in order.")
        weights = item.get("weights")
        if not isinstance(weights, list) or not 1 <= len(weights) <= 65536:
            raise ValueError(f"Layer {index} needs between 1 and 65,536 weights.")
        width = len(weights) if width is None else width
        if len(weights) != width:
            raise ValueError("Every layer's weights must be the same width.")
        clean.append({
            "layer": index,
            "weights": [weight(w) for w in weights],
            "bias": number(item.get("bias"), "bias"),
            **{key: number(item.get(key), key.replace("_", " "))
               for key in ("train_accuracy", "heldout_accuracy", "heldout_loss")},
        })
    revision = value.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not 0 < len(revision) <= 200):
        raise ValueError("The probe's model revision must be text of at most 200 characters, or null.")
    precision = value.get("precision")
    if precision is not None and precision not in PRECISIONS:
        raise ValueError("The probe's precision must be full, 8-bit, 4-bit or null.")
    created = number(value.get("created"), "creation time")
    # The saved list shows the date, so it has to be one a date can hold.
    if not 0 <= created <= LATEST_CREATED:
        raise ValueError("The probe's creation time is not a date between 1970 and 3000.")
    best = value.get("best_layer")
    if type(best) is not int or not 0 <= best < len(clean):
        raise ValueError("The probe's best layer is not one of its layers.")
    l2 = number(value.get("l2"), "L2 strength")
    if l2 <= 0:
        raise ValueError("L2 strength must be positive and finite.")
    return {
        "format": FORMAT, "id": probe_id, "name": text("name"), "model_id": text("model_id", 300),
        "model_revision": revision,
        "precision": precision,
        "positive_label": positive_label, "negative_label": negative_label,
        "pool": value["pool"], "chat_template": value["chat_template"],
        "l2": l2, "folds": int(number(value.get("folds"), "folds")),
        "paired": value.get("paired", False),
        "created": created,
        "examples": {side: list(examples[side]) for side in ("positive", "negative")},
        "layers": clean, "best_layer": best,
    }


def directions(probe):
    """The weights as one row per block, the shape the host reads a passage along."""
    return np.asarray([item["weights"] for item in probe["layers"]], dtype=np.float32)


def probabilities(probe, projections):
    """Each block's probability for each position, from the host's dot products.

    Weights that each fit in float32 can still overflow in their sum, so a
    reading that is not finite is refused rather than shown as certainty.
    """
    projections = np.asarray(projections, dtype=np.float64)
    if not np.isfinite(projections).all():
        raise ValueError("This probe's weights are too large for this model's activations: "
                         "the reading overflowed. Train the probe again.")
    biases = np.asarray([item["bias"] for item in probe["layers"]], dtype=np.float64)
    return _sigmoid(projections + biases[:, None])


def dumps(probe):
    return json.dumps(probe, separators=(",", ":"))


def read(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Probe files must be smaller than 64 MiB.")
    try:
        return normalize(json.loads(data))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("The probe file must contain valid JSON.") from error


def saved(directory):
    """Every readable probe in the directory, newest first. Unreadable files are skipped."""
    found = []
    for path in Path(directory).glob("*.json"):
        try:
            found.append(read(path))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda probe: probe["created"], reverse=True)
