"""Attribution graphs: which transcoder features carried the model to a token.

The method is the one in Ameisen et al., "Circuit Tracing" (2025), with per-
layer transcoders, and follows circuit-tracer's choices where the paper leaves
room (https://github.com/safety-research/circuit-tracer, MIT):

1. Run the prompt once and record every transcoder's input and the MLP output
   it stands in for. The active features, and the error between their decoded
   sum and the real MLP output, are fixed from here on.
2. Run it again under :func:`architecture.frozen`, batched, with each block's
   MLP output replaced by a leaf tensor holding the same value. The model is
   now linear in the token embeddings, the features and the errors.
3. Each target - an output token's logit, or a feature's pre-activation - is
   one backward pass. A source's edge into the target is its value times the
   target's gradient with respect to it, which is its direct contribution.
4. Rows are computed for the targets first, then for features in order of
   their influence on the targets, until the node budget is spent.
5. The graph is pruned to the features carrying most of the influence and the
   edges carrying most of what is left.

Every node is a fixed value and every edge a direct effect, so for any target
the edges into it, with the bias terms, sum to its value exactly. The tests
check that.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass

from . import architecture

logger = logging.getLogger(__name__)

FORMAT = "chatlab-attribution-graph-1"
MAX_PREFIX = 512
MAX_TARGETS = 10
TARGET_MASS = 0.95
# Budget for CPU edge, pruning, and sorting allocations before the run is refused.
MAX_ROW_BYTES = 6 * 1024 ** 3
# What one batch's feature-gradient product may hold on the device.
CHUNK_BYTES = 256 * 1024 ** 2


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    max_feature_nodes: int = 400
    batch_size: int = 32
    node_threshold: float = 0.8
    edge_threshold: float = 0.98

    def check(self):
        if not 16 <= self.max_feature_nodes <= 4096:
            raise ValueError("Feature nodes must be between 16 and 4096.")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("Batch size must be between 1 and 256.")
        for name, value in (("Node", self.node_threshold), ("Edge", self.edge_threshold)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} threshold must be above 0 and at most 1.")


@dataclass
class Recording:
    """What the plain forward pass left behind for the frozen one."""

    ids: list
    embeddings: object      # (n, d), model dtype
    outputs: list           # per layer (n, d), the MLP output, model dtype
    errors: list            # per layer (n, d), float32: output minus the decoded features
    biases: list            # per layer (d,), float32: the decoder bias
    feature_layer: object   # (K,) long, sorted by layer then position
    feature_position: object
    feature_index: object
    activation: object      # (K,) float32
    layer_slices: list      # per layer (start, end) into the K features
    final: object           # (n, d) float32, the normalized final residual
    logits: object          # (vocab,) float32 at the last position


def record(blocks, transcoders, ids):
    """Run the prompt once and fix the active features and errors."""
    import torch

    n = len(ids)
    layers = len(blocks.layers)
    if transcoders.layers != layers:
        raise ValueError(f"The transcoders cover {transcoders.layers} layers; the model has {layers}.")
    inputs, outputs = [None] * layers, [None] * layers

    def keep(store, layer):
        def hook(_module, _args, output):
            store[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    device = blocks.device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with contextlib.ExitStack() as stack, torch.no_grad():
        for layer in range(layers):
            stack.callback(blocks.mlp_input(layer).register_forward_hook(keep(inputs, layer)).remove)
            stack.callback(blocks.mlp_output(layer).register_forward_hook(keep(outputs, layer)).remove)
        embeddings = blocks.embed(tensor)
        hidden = blocks.inner(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        final = hidden[0].float()
        logits = blocks.unembed(hidden[:, -1])[0].float()
        if blocks.final_softcap:
            logits = torch.tanh(logits / blocks.final_softcap) * blocks.final_softcap
    if any(value is None or value.shape[:2] != (1, n) for value in inputs + outputs):
        raise ValueError("Not every decoder block's MLP ran exactly once on the prompt.")

    errors, layer_index, positions, features, values, slices = [], [], [], [], [], []
    start = 0
    for layer in range(layers):
        acts = transcoders.encode(layer, inputs[layer][0])
        # The first position is the beginning-of-sequence token, whose huge
        # activations say nothing about this prompt. Its features are left
        # out and what they wrote is counted as error, as circuit-tracer does.
        acts[0] = 0
        position, feature = torch.nonzero(acts, as_tuple=True)
        reconstruction = transcoders.decode(layer, acts)
        errors.append(outputs[layer][0].float() - reconstruction)
        layer_index.append(torch.full_like(position, layer))
        positions.append(position)
        features.append(feature)
        values.append(acts[position, feature])
        slices.append((start, start + len(position)))
        start += len(position)
    return Recording(
        ids=list(ids), embeddings=embeddings[0], outputs=[o[0] for o in outputs], errors=errors,
        biases=[transcoders.b_dec[layer].float() for layer in range(layers)],
        feature_layer=torch.cat(layer_index), feature_position=torch.cat(positions),
        feature_index=torch.cat(features), activation=torch.cat(values).float(),
        layer_slices=slices, final=final, logits=logits,
    )


def logit_directions(blocks, recording, targets):
    """The residual direction each target reads, in the final normalized basis.

    A token target is its logit minus the mean logit, so a direction every
    token shares does not count as support for this one. A contrast target is
    the gradient of log P(positive) - log P(negative) at the prompt's own
    distribution: each side's tokens weighted by their share of that side.
    """
    import torch

    weight = blocks.unembed.weight
    probabilities = torch.softmax(recording.logits, dim=-1)
    mean = weight.float().mean(0)
    # Contrast targets measure softcapped log odds; chain through the final cap.
    derivative = torch.ones_like(recording.logits)
    if blocks.final_softcap:
        raw = blocks.unembed(recording.final[-1].to(weight.dtype)).float()
        derivative = 1 - torch.tanh(raw / blocks.final_softcap).square()
    directions = []
    for target in targets:
        if target["kind"] == "token":
            directions.append(weight[target["token_id"]].float() - mean)
            continue
        vector = torch.zeros_like(mean)
        for side, sign in (("positive", 1.0), ("negative", -1.0)):
            ids = torch.tensor(target[side], dtype=torch.long, device=weight.device)
            share = probabilities[ids]
            share = share / share.sum().clamp(min=1e-30)
            vector += sign * ((share * derivative[ids])[:, None] * weight[ids].float()).sum(0)
        directions.append(vector)
    return torch.stack(directions)


def choose_targets(recording, decode, token_ids=None, contrast=None):
    """The output nodes: the given tokens, or the likeliest ones up to 95% of the mass."""
    import torch

    probabilities = torch.softmax(recording.logits, dim=-1)
    if token_ids:
        ids = list(dict.fromkeys(int(t) for t in token_ids))
    else:
        order = torch.argsort(probabilities, descending=True)[:MAX_TARGETS].tolist()
        ids, mass = [], 0.0
        for token in order:
            ids.append(token)
            mass += float(probabilities[token])
            if mass >= TARGET_MASS:
                break
    targets = [{"kind": "token", "token_id": t, "text": decode(t), "probability": float(probabilities[t])}
               for t in ids]
    if contrast:
        positive, negative = contrast["positive"], contrast["negative"]
        if not positive or not negative:
            raise ValueError("A contrast needs tokens on both sides.")
        if set(positive) & set(negative):
            raise ValueError("A token cannot be on both sides of the contrast.")

        def mass(side):
            return float(probabilities[torch.tensor(side, device=probabilities.device)].sum())

        targets = [{"kind": "contrast", "positive": list(positive), "negative": list(negative),
                    "text": contrast.get("label") or "contrast",
                    "probability": mass(positive), "negative_probability": mass(negative)}]
    return targets


class FrozenGraph:
    """The frozen, batched replacement model, answering one target row per batch slot."""

    def __init__(self, blocks, transcoders, recording, batch_size):
        import torch

        self.blocks, self.transcoders, self.recording = blocks, transcoders, recording
        self.batch = batch_size
        n, layers = len(recording.ids), len(blocks.layers)
        self.n, self.layers = n, layers
        self.stack = contextlib.ExitStack()
        try:
            self._build(torch, n, layers)
        except BaseException:
            self.stack.close()
            raise

    def _build(self, torch, n, layers):
        blocks, recording, batch = self.blocks, self.recording, self.batch
        self.stack.enter_context(architecture.frozen(blocks.model))
        self.stack.enter_context(torch.enable_grad())
        self.embedding = recording.embeddings[None].expand(batch, -1, -1).clone().requires_grad_(True)
        self.outputs = [recording.outputs[layer][None].expand(batch, -1, -1).clone().requires_grad_(True)
                        for layer in range(layers)]
        self.inputs = [None] * layers
        for layer in range(layers):
            leaf = self.outputs[layer]
            self.stack.enter_context(architecture.replaced_output(blocks.mlp_output(layer),
                                                                  lambda *_a, leaf=leaf, **_k: leaf))
            if blocks.output_name != "mlp":
                # The MLP's own output is thrown away, so it need not be computed.
                self.stack.enter_context(architecture.replaced_output(blocks.layers[layer].mlp,
                                                                      lambda x, *_a, **_k: x))

            def keep(_module, _args, output, layer=layer):
                self.inputs[layer] = output[0] if isinstance(output, tuple) else output
            self.stack.callback(blocks.mlp_input(layer).register_forward_hook(keep).remove)
        self.final = blocks.inner(inputs_embeds=self.embedding, use_cache=False).last_hidden_state
        drift = (self.final[0].float() - recording.final).abs().max().item()
        scale = recording.final.abs().max().item() or 1.0
        if not math.isfinite(drift) or drift > 1e-2 * scale + 1e-3:
            raise ValueError("Freezing the model changed its output, so its attribution would be wrong.")

        # The decoder rows of every active feature, gathered once per layer.
        tc = self.transcoders
        self.decoders = []
        for layer, (start, end) in enumerate(recording.layer_slices):
            self.decoders.append(tc.decoder_rows(layer, recording.feature_index[start:end]))

    def close(self):
        self.stack.close()

    @property
    def columns(self):
        return len(self.recording.activation) + self.layers * self.n + self.n

    def rows(self, targets):
        """Edge rows into up to ``batch`` targets, as a float32 CPU tensor.

        A target is ``("vector", direction)``, read at the last position from
        the final normalized residual, or ``("feature", k)`` for active feature
        ``k``, read from its encoder pre-activation.
        """
        import torch

        recording, tc = self.recording, self.transcoders
        count = len(targets)
        if not 0 < count <= self.batch:
            raise ValueError("A batch holds between one target and the batch size.")
        injected = {}
        final_grad = None
        for slot, (kind, value) in enumerate(targets):
            if kind == "vector":
                if final_grad is None:
                    final_grad = torch.zeros_like(self.final)
                final_grad[slot, -1] = value.to(final_grad.dtype)
            else:
                layer = int(recording.feature_layer[value])
                position = int(recording.feature_position[value])
                feature = recording.feature_index[value:value + 1]
                grad = injected.get(layer)
                if grad is None:
                    grad = injected[layer] = torch.zeros_like(self.inputs[layer])
                grad[slot, position] = tc.encoder_rows(layer, feature)[0].to(grad.dtype)
        outputs, grads = [], []
        if final_grad is not None:
            outputs.append(self.final)
            grads.append(final_grad)
        for layer, grad in injected.items():
            outputs.append(self.inputs[layer])
            grads.append(grad)
        sources = [self.embedding, *self.outputs]
        answers = torch.autograd.grad(outputs, sources, grads, retain_graph=True, allow_unused=True)
        embedding_grad, output_grads = answers[0], answers[1:]
        row = torch.zeros(count, self.columns, dtype=torch.float32, device=self.recording.activation.device)
        features = len(recording.activation)
        for layer in range(self.layers):
            grad = output_grads[layer]
            if grad is None:
                continue
            grad = grad[:count].float()
            start, end = recording.layer_slices[layer]
            if end > start:
                row[:, start:end] = self._feature_edges(grad, layer, start, end)
            error_start = features + layer * self.n
            row[:, error_start:error_start + self.n] = (grad * recording.errors[layer]).sum(-1)
        if embedding_grad is not None:
            row[:, -self.n:] = (embedding_grad[:count].float() * recording.embeddings.float()).sum(-1)
        return row.cpu()

    def _feature_edges(self, grad, layer, start, end):
        import torch

        recording = self.recording
        positions = recording.feature_position[start:end]
        decoders = self.decoders[layer]
        activation = recording.activation[start:end]
        # (count, n, d) gathered at each feature's position, dotted with its
        # decoder row, in chunks of features so the product stays bounded.
        chunk = max(1, CHUNK_BYTES // (4 * grad.shape[0] * grad.shape[-1]))
        out = []
        for begin in range(0, end - start, chunk):
            stop = min(end - start, begin + chunk)
            picked = grad[:, positions[begin:stop], :]
            out.append((picked * decoders[begin:stop][None]).sum(-1))
        return torch.cat(out, dim=1) * activation[None]

    def bias_terms(self, targets):
        """What the decoder biases contributed to each target; for the completeness check."""
        import torch

        recording = self.recording
        count = len(targets)
        outputs, grads = [], []
        final_grad = torch.zeros_like(self.final)
        for slot, (_, value) in enumerate(targets):
            final_grad[slot, -1] = value.to(final_grad.dtype)
        answers = torch.autograd.grad([self.final], self.outputs, [final_grad], retain_graph=True,
                                      allow_unused=True)
        total = torch.zeros(count, device=recording.activation.device)
        for layer, grad in enumerate(answers):
            if grad is not None:
                total += (grad[:count].float() * recording.biases[layer]).sum((-1, -2))
        del outputs, grads
        return total.cpu()


def _influence(rows, weights, row_of_column, iterations=None):
    """Each column's share of the targets' influence, through every path the rows cover.

    ``rows`` are absolute, row-normalized edge weights. Influence starts at
    the target rows with ``weights`` and flows back through any column that
    has a row of its own, until it reaches columns that have none.
    """
    import torch

    present = row_of_column >= 0
    columns = torch.nonzero(present, as_tuple=True)[0]
    rows_for = row_of_column[columns]
    x = weights.clone()
    total = torch.zeros(rows.shape[1], dtype=torch.float32)
    for _ in range(iterations or 4096):
        contribution = x @ rows
        total += contribution
        x = torch.zeros_like(weights)
        x[rows_for] = contribution[columns]
        if float(x.sum()) <= 1e-9:
            break
    return total


def _normalized(rows):
    import torch

    magnitude = rows.abs()
    sums = magnitude.sum(1, keepdim=True)
    return torch.where(sums > 0, magnitude / sums.clamp(min=1e-30), magnitude)


def attribute(blocks, transcoders, ids, decode, *, settings=None, token_ids=None, contrast=None,
              progress=None, cancelled=None):
    """Build a pruned attribution graph for the token after ``ids``."""
    import torch

    settings = settings or Settings()
    settings.check()
    if not 2 <= len(ids) <= MAX_PREFIX:
        raise ValueError(f"Attribution needs between 2 and {MAX_PREFIX} prompt tokens.")
    report = progress or (lambda *_: None)
    stop = cancelled or (lambda: False)

    report("Reading the prompt", 0, 1)
    recording = record(blocks, transcoders, ids)
    targets = choose_targets(recording, decode, token_ids, contrast)
    directions = logit_directions(blocks, recording, targets)
    features = len(recording.activation)
    graph = FrozenGraph(blocks, transcoders, recording, settings.batch_size)
    try:
        columns = graph.columns
        budget = min(settings.max_feature_nodes, features)
        # Include dense pruning matrices and worst-case nonzero-edge sorting,
        # which stay alive alongside the tracing rows. The factor of four on
        # tracing rows also reserves normalization and batch temporaries.
        total_nodes = budget + (len(recording.errors) + 1) * len(ids) + len(targets)
        allocation_bytes = 4 * (budget + len(targets)) * columns * 4 + 64 * total_nodes ** 2
        if allocation_bytes > MAX_ROW_BYTES:
            raise ValueError(
                f"This prompt has {features:,} active features; a graph of {budget} of them would need "
                f"{allocation_bytes / 1024 ** 3:.1f} GB. Use a shorter prompt or fewer nodes.")
        rows = torch.zeros(len(targets) + budget, columns, dtype=torch.float32)
        normalized = torch.zeros_like(rows)
        vectors = [("vector", d) for d in directions]
        batches = [vectors[i:i + settings.batch_size] for i in range(0, len(vectors), settings.batch_size)]
        target_rows = torch.cat([graph.rows(batch) for batch in batches])
        rows[:len(targets)] = target_rows
        normalized[:len(targets)] = _normalized(target_rows)
        values = (directions.to(recording.final.dtype) @ recording.final[-1]).cpu()
        biases = torch.cat([graph.bias_terms(batch) for batch in batches])
        weights = torch.zeros(rows.shape[0])
        probabilities = torch.tensor([t["probability"] for t in targets])
        weights[:len(targets)] = probabilities / probabilities.sum().clamp(min=1e-30)
        row_of_column = torch.full((columns,), -1, dtype=torch.long)
        chosen = []
        used = len(targets)
        report("Tracing features", 0, budget)
        while len(chosen) < budget:
            if stop():
                raise Cancelled()
            influence = _influence(normalized[:used], weights[:used], row_of_column)[:features]
            if chosen:
                influence[torch.tensor(chosen, dtype=torch.long)] = -1
            order = torch.argsort(influence, descending=True)
            take = [int(k) for k in order[:min(settings.batch_size, budget - len(chosen))]
                    if influence[k] > 0]
            if not take:
                break
            new = graph.rows([("feature", k) for k in take])
            rows[used:used + len(take)] = new
            normalized[used:used + len(take)] = _normalized(new)
            for offset, k in enumerate(take):
                row_of_column[k] = used + offset
            chosen.extend(take)
            used += len(take)
            report("Tracing features", len(chosen), budget)
    finally:
        graph.close()

    report("Pruning", 0, 1)
    result = _prune(recording, targets, rows[:used], weights[:used], chosen, settings, decode)
    result["targets_check"] = {
        "values": values.tolist(),
        "edge_sums": rows[:len(targets)].sum(1).tolist(),
        "bias_terms": biases.tolist(),
    }
    return result


def _prune(recording, targets, rows, weights, chosen, settings, decode):
    """Restrict to the chosen features, then keep the nodes and edges that carry the influence."""
    import torch

    n, layers = len(recording.ids), len(recording.errors)
    features = len(recording.activation)
    count = len(targets)
    chosen_t = torch.tensor(chosen, dtype=torch.long)
    # Node order: chosen features, errors, embeddings, targets.
    keep_columns = torch.cat([chosen_t, torch.arange(features, features + layers * n + n)])
    s, e = len(chosen), layers * n
    total = s + e + n + count
    adjacency = torch.zeros(total, total, dtype=torch.float32)
    adjacency[:s, :s + e + n] = rows[count:][:, keep_columns]
    adjacency[s + e + n:, :s + e + n] = rows[:count][:, keep_columns]
    normalized = _normalized(adjacency)
    node_weights = torch.zeros(total)
    node_weights[s + e + n:] = weights[:count]
    has_row = torch.full((total,), -1, dtype=torch.long)
    has_row[:s] = torch.arange(s)
    has_row[s + e + n:] = torch.arange(s + e + n, total)
    node_influence = _influence_square(normalized, node_weights, has_row)

    # Node pruning: the fewest features holding node_threshold of the influence.
    feature_influence = node_influence[:s]
    order = torch.argsort(feature_influence, descending=True)
    cumulative = torch.cumsum(feature_influence[order], 0)
    limit = settings.node_threshold * float(cumulative[-1]) if s else 0.0
    kept = torch.zeros(total, dtype=torch.bool)
    if s:
        cut = int(torch.searchsorted(cumulative, torch.tensor(limit)).item()) + 1
        kept[order[:cut]] = True
    kept[s:] = True
    masked = normalized * kept[:, None] * kept[None, :]
    masked = _normalized(masked)
    node_influence = _influence_square(masked, node_weights, has_row)

    # Edge pruning: the fewest edges holding edge_threshold of the influence.
    score = (node_influence + node_weights)[:, None] * masked
    flat = score.flatten()
    nonzero = torch.nonzero(flat > 0, as_tuple=True)[0]
    edge_keep = torch.zeros_like(flat, dtype=torch.bool)
    if len(nonzero):
        values = flat[nonzero]
        order = torch.argsort(values, descending=True)
        cumulative = torch.cumsum(values[order], 0)
        cut = int(torch.searchsorted(cumulative, settings.edge_threshold * cumulative[-1]).item()) + 1
        edge_keep[nonzero[order[:cut]]] = True
    edge_keep = edge_keep.view(total, total)

    # Signed total effect of each node on the weighted targets.
    activation = torch.ones(total)
    activation[:s] = recording.activation[chosen_t].cpu()
    effect = _effect(adjacency, node_weights, torch.arange(s), activation)

    connected = edge_keep.any(0) | edge_keep.any(1)
    connected[s + e + n:] = True
    tokens = [decode(t) for t in recording.ids]
    nodes, index_of = [], {}

    def add(i, node):
        index_of[i] = len(nodes)
        node.update(influence=float(node_influence[i]), effect=float(effect[i]))
        nodes.append(node)

    layer_of, position_of = recording.feature_layer.cpu(), recording.feature_position.cpu()
    index_ = recording.feature_index.cpu()
    for i in range(s):
        if kept[i] and connected[i]:
            k = chosen[i]
            layer, position, feature = int(layer_of[k]), int(position_of[k]), int(index_[k])
            add(i, {"id": f"f:{layer}:{position}:{feature}", "kind": "feature", "layer": layer,
                    "position": position, "feature": feature, "activation": float(activation[i])})
    for j in range(e):
        if connected[s + j]:
            layer, position = divmod(j, n)
            add(s + j, {"id": f"e:{layer}:{position}", "kind": "error", "layer": layer, "position": position})
    for position in range(n):
        if connected[s + e + position]:
            add(s + e + position, {"id": f"t:{position}", "kind": "embedding", "layer": -1,
                                   "position": position, "token_id": int(recording.ids[position])})
    for t, target in enumerate(targets):
        add(s + e + n + t, {"id": f"target:{t}", "kind": "target", "layer": layers, "position": n - 1,
                            **{k: v for k, v in target.items() if k != "kind"}, "target_kind": target["kind"]})
    edges = []
    for target_node, source_node in torch.nonzero(edge_keep, as_tuple=False).tolist():
        if target_node in index_of and source_node in index_of:
            edges.append({"source": nodes[index_of[source_node]]["id"], "target": nodes[index_of[target_node]]["id"],
                          "weight": float(adjacency[target_node, source_node])})
    sources = node_influence[s:s + e + n]
    error_share = float(sources[:e].sum() / sources.sum().clamp(min=1e-30))
    return {
        "format": FORMAT,
        "ids": list(recording.ids), "tokens": tokens, "layers": layers,
        "nodes": nodes, "edges": edges,
        "settings": vars(settings).copy(),
        "stats": {"active_features": features, "traced_features": s,
                  "kept_features": sum(1 for node in nodes if node["kind"] == "feature"),
                  "error_share": error_share},
    }


def _influence_square(normalized, weights, has_row):
    import torch

    x = weights.clone()
    total = torch.zeros_like(weights)
    rows = torch.nonzero(has_row >= 0, as_tuple=True)[0]
    for _ in range(4096):
        contribution = x @ normalized
        total += contribution
        x = torch.zeros_like(weights)
        x[rows] = contribution[rows]
        if float(x.abs().sum()) <= 1e-9:
            break
    return total


def _effect(adjacency, weights, feature_rows, activation):
    """Each node's signed contribution to the weighted targets, direct and indirect.

    A feature's edges are into its pre-activation, so passing influence on
    through a feature divides by its activation: the edge is that feature's
    value times its gradient, and the next hop wants the gradient alone.
    """
    import torch

    x = weights.clone()
    total = torch.zeros_like(weights)
    for _ in range(4096):
        contribution = x @ adjacency
        total += contribution
        x = torch.zeros_like(weights)
        x[feature_rows] = contribution[feature_rows] / activation[feature_rows]
        if float(x.abs().sum()) <= 1e-12:
            break
    return total
