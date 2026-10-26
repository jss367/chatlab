"""Measured component interventions, with exact token alignment and per-prefix results.

Backends expose activations and apply edits during ordinary model forwards.
Nothing here freezes attention, normalization, or downstream activations.
"""
from __future__ import annotations

import math

from chatlab.activation_patching import integer

from .attribution import _check_cancelled

MAX_FACTORS = 11
MAX_PASSES = 1024


def factors(text):
    try:
        values = list(dict.fromkeys(float(x) for x in text.replace(",", " ").split()))
    except (ValueError, AttributeError):
        raise ValueError("Give scale factors as numbers separated by spaces or commas.") from None
    values = list(dict.fromkeys([0.0, 1.0, *values])) if values else []
    if not 1 <= len(values) <= MAX_FACTORS or any(not math.isfinite(x) or abs(x) > 10 for x in values):
        raise ValueError(f"Give 1–{MAX_FACTORS} finite scale factors between -10 and 10.")
    # Zero is needed for the rescue comparison; one is an explicit no-op control.
    return sorted(values)


def position(offset, count, label="Token offset"):
    offset = integer(offset, label)
    if not 0 <= offset < count:
        raise ValueError(f"{label} must be between 0 and {count - 1}, counted from the end.")
    return count - 1 - offset


def experiment(forward, prefixes, component, targets, scales, *, offset=0, donor=None,
               donor_offset=0, hub=None, hub_offset=0, progress=None, cancelled=None):
    """Scale/patch a component and optionally restore a later component after ablation.

    Components are (kind, layer, index). A replacement copies the *activation*,
    not a residual direction. Offsets align from each prefix's end, explicitly.
    """
    scales = factors(" ".join(map(str, scales)))
    if not 1 <= len(prefixes) <= 256:
        raise ValueError("Use between 1 and 256 destination prefixes.")
    if (not targets or len(targets) > 16 or any(type(t) is not int or t < 0 for t in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("Choose 1–16 distinct output tokens.")
    if hub is not None and hub[1] <= component[1]:
        raise ValueError("The restored component must be in a later layer than the edited component.")
    for ids in prefixes:
        position(offset, len(ids))
        if hub is not None:
            position(hub_offset, len(ids), "Restored component offset")
    if donor is not None:
        position(donor_offset, len(donor), "Source token offset")
    total = len(prefixes) * (1 + len(scales) + bool(donor is not None) + bool(hub is not None)) + bool(donor is not None)
    if total > MAX_PASSES:
        raise ValueError(f"This run needs {total} model passes; reduce prefixes or factors to stay under {MAX_PASSES}.")
    done = 0
    observe = [component] + ([hub] if hub is not None else [])

    def run(ids, changes=()):
        nonlocal done
        _check_cancelled(cancelled)
        value = forward(ids, changes=changes, observe=observe)
        _check_cancelled(cancelled)
        done += 1
        if progress:
            progress("Testing components", done, total)
        return value

    def reading(value, label, at, hub_at=None):
        logs = value["log_probs"][targets].detach().cpu().double()
        if not bool(logs.isfinite().all()):
            raise ValueError("The model returned non-finite output probabilities.")
        return {"condition": label, "probabilities": logs.exp().tolist(), "log_probabilities": logs.tolist(),
                "activation": value["activations"][component][at].tolist(),
                "hub_activation": value["activations"][hub][hub_at].tolist() if hub is not None and hub_at is not None else None}

    donor_value, donor_result = None, None
    if donor is not None:
        source = run(donor)
        source_at = position(donor_offset, len(donor))
        donor_value = source["activations"][component][source_at].clone()
        donor_result = reading(source, "source baseline", source_at)
    results = []
    for ids in prefixes:
        at = position(offset, len(ids))
        hub_at = position(hub_offset, len(ids)) if hub is not None else None
        baseline = run(ids)
        rows = [reading(baseline, "baseline", at, hub_at)]
        for scale in scales:
            result = run(ids, [(component, at, "scale", scale)])
            row = reading(result, f"scale ×{scale:g}", at, hub_at)
            row["factor"] = scale
            rows.append(row)
        if donor is not None:
            result = run(ids, [(component, at, "replace", donor_value)])
            rows.append(reading(result, "source activation patched", at, hub_at))
        if hub is not None:
            restored = baseline["activations"][hub][hub_at].clone()
            result = run(ids, [(component, at, "scale", 0.0), (hub, hub_at, "replace", restored)])
            rows.append(reading(result, "ablated + later component restored", at, hub_at))
        results.append({"ids": list(ids), "position": at, "hub_position": hub_at,
                        "activations": baseline["activations"][component].tolist(), "conditions": rows})
    return {"component": list(component), "hub": list(hub) if hub is not None else None,
            "offset": offset, "hub_offset": hub_offset, "donor_offset": donor_offset,
            "targets": targets, "scales": scales, "prefixes": results, "source": donor_result,
            "source_ids": list(donor) if donor is not None else None, "model_passes": done}
