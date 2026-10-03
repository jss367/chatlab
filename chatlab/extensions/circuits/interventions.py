"""Ablating and boosting transcoder features in the real model.

An intervention changes what one decoder block's MLP wrote by exactly what the
chosen features would add if their activations were scaled: the features are
read from the block's input on this very pass, so features in later layers
see the change in earlier ones, and the transcoder's error is left alone.
Scaling by zero ablates a feature; scaling by more than one boosts it.

Nothing is frozen here. These are ordinary forward passes, so what they
measure is what the model does, against which the linear graph is a claim.
"""

from __future__ import annotations

import contextlib
import math
from collections import defaultdict

from .attribution import MAX_PREFIX

MAX_PREFIXES = 256
DEFAULT_BOOST = 2.0


def run(blocks, transcoders, ids, changes=(), tokens=()):
    """Probabilities at the last position after scaling features as ``changes`` say.

    ``changes`` is a list of ``(layer, feature, factor, positions)``, where
    ``positions`` is a list of positions counted back from the end (0 is the
    last token), or ``None`` for every position. Returns the probabilities
    of ``tokens`` and how many of the changed features were active at all.
    """
    import torch

    if not 1 <= len(ids) <= MAX_PREFIX:
        raise ValueError(f"A prefix must hold between 1 and {MAX_PREFIX} tokens.")
    n = len(ids)
    by_layer = defaultdict(list)
    for layer, feature, factor, positions in changes:
        where = None if positions is None else sorted({n - 1 - p for p in positions if 0 <= p < n})
        by_layer[int(layer)].append((int(feature), float(factor), where))
    inputs = {}
    active = 0

    def keep(layer):
        def hook(_module, _args, output):
            inputs[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    def change(layer):
        def hook(_module, _args, output):
            nonlocal active
            value = output[0] if isinstance(output, tuple) else output
            x = inputs.pop(layer)[0]
            delta = torch.zeros(value.shape[1:], dtype=torch.float32, device=value.device)
            for feature, factor, where in by_layer[layer]:
                features = torch.tensor([feature], device=value.device)
                encoder = transcoders.w_enc[layer][features]
                pre = (x.to(encoder.dtype) @ encoder.T).float()
                pre = pre + transcoders.b_enc[layer][features].float()
                acts = transcoders.activate_one(layer, feature, pre[:, 0])
                if where is not None:
                    mask = torch.zeros_like(acts)
                    mask[where] = 1
                    acts = acts * mask
                active += int(bool((acts != 0).any()))
                delta += (factor - 1.0) * acts[:, None] * transcoders.decoder_rows(layer, features)[0][None]
            changed = value.clone()
            changed[0] += delta.to(value.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return hook

    tensor = torch.tensor([ids], dtype=torch.long, device=blocks.device)
    with contextlib.ExitStack() as stack, torch.no_grad():
        for layer in by_layer:
            stack.callback(blocks.mlp_input(layer).register_forward_hook(keep(layer)).remove)
            stack.callback(blocks.mlp_output(layer).register_forward_hook(change(layer)).remove)
        hidden = blocks.inner(input_ids=tensor, use_cache=False).last_hidden_state
        logits = blocks.unembed(hidden[:, -1])[0].float()
        if blocks.final_softcap:
            logits = torch.tanh(logits / blocks.final_softcap) * blocks.final_softcap
        log_probs = torch.log_softmax(logits, dim=-1)
    values = log_probs[list(tokens)].cpu().double().exp().tolist() if tokens else []
    if any(not math.isfinite(v) for v in values):
        raise ValueError("The model returned a non-finite probability.")
    return {"probabilities": values, "active": active, "log_probs": log_probs}


def group_effects(blocks, transcoders, prefixes, groups, pivot, alternatives, *, boost=DEFAULT_BOOST,
                  every_position=False, progress=None, cancelled=None, auto_alternatives=False):
    """Ablate and boost each group on every prefix, averaging each token's probability.

    ``groups`` maps a name to ``(layer, feature, offset)`` members, where the
    offset counts back from the end of the prompt the graph was traced on.
    Applied at those same offsets in every prefix, or at every position when
    ``every_position`` is set. Returns the mean probability of each pivot and
    alternative token at the baseline and under each change, with P(pivot) -
    the pivot tokens' summed probability - and how many prefixes each group
    had an active feature in.
    """
    if not prefixes:
        raise ValueError("Give at least one prefix.")
    if len(prefixes) > MAX_PREFIXES:
        raise ValueError(f"Use at most {MAX_PREFIXES} prefixes.")
    if not pivot:
        raise ValueError("Name at least one pivot token.")
    if not groups:
        raise ValueError("Make at least one group from the graph first.")
    if not 0 <= boost <= 100:
        raise ValueError("The boost factor must be between 0 and 100.")
    report = progress or (lambda *_: None)
    stop = cancelled or (lambda: False)
    total = len(prefixes) * (1 + 2 * len(groups))
    done = 0
    distribution = None
    if auto_alternatives:
        # Keep one aggregate vocabulary vector, rather than every prefix's
        # distribution; these passes also supply the measured baseline.
        for ids in prefixes:
            if stop():
                raise _cancelled()
            values = run(blocks, transcoders, ids)["log_probs"].cpu().double().exp()
            distribution = values if distribution is None else distribution + values
            done += 1
            report(done, total)
        excluded = set(pivot)
        alternatives = [int(t) for t in distribution.argsort(descending=True).tolist() if t not in excluded][:8]
    tokens = list(dict.fromkeys([*pivot, *alternatives]))
    sums = {"baseline": ([float(distribution[t]) for t in tokens] if auto_alternatives
                         else [0.0] * len(tokens))}
    activity = {}
    for name in groups:
        sums[(name, "ablate")] = [0.0] * len(tokens)
        sums[(name, "boost")] = [0.0] * len(tokens)
        activity[name] = 0

    def changes(members, factor):
        out, seen = [], set()
        for layer, feature, offset in members:
            key = (layer, feature) if every_position else (layer, feature, offset)
            if key in seen:
                continue
            seen.add(key)
            out.append((layer, feature, factor, None if every_position else [offset]))
        return out

    for ids in prefixes:
        if stop():
            raise _cancelled()
        if not auto_alternatives:
            result = run(blocks, transcoders, ids, (), tokens)
            sums["baseline"] = [a + b for a, b in zip(sums["baseline"], result["probabilities"])]
            done += 1
            report(done, total)
        for name, members in groups.items():
            for kind, factor in (("ablate", 0.0), ("boost", boost)):
                if stop():
                    raise _cancelled()
                result = run(blocks, transcoders, ids, changes(members, factor), tokens)
                sums[(name, kind)] = [a + b for a, b in zip(sums[(name, kind)], result["probabilities"])]
                if kind == "ablate" and result["active"]:
                    activity[name] += 1
                done += 1
                report(done, total)
    count = len(prefixes)
    pivot_set = set(pivot)

    def summary(values):
        means = [v / count for v in values]
        return {"tokens": dict(zip(tokens, means)),
                "pivot": sum(m for t, m in zip(tokens, means) if t in pivot_set)}

    baseline = summary(sums["baseline"])
    result = {"prefixes": count, "boost": boost, "every_position": every_position,
              "pivot": list(pivot), "alternatives": [t for t in alternatives if t not in pivot_set],
              "baseline": baseline, "groups": {}}
    for name in groups:
        result["groups"][name] = {
            "ablate": summary(sums[(name, "ablate")]),
            "boost": summary(sums[(name, "boost")]),
            "active_prefixes": activity[name],
        }
    return result


def _cancelled():
    from .attribution import Cancelled
    return Cancelled()
