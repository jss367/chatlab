"""Native MLP channels and attention-head outputs, without transcoders.

A gated MLP neuron is one coordinate of the input to down_proj (after the
gate and up projections are multiplied). A head is its contiguous head_dim
coordinates before o_proj. Hooks only live inside a caller-held model lock.
"""
from __future__ import annotations

import contextlib
import inspect
from collections import defaultdict

from chatlab.activation_patching import integer
from chatlab.steering import decoder_layers

from .attribution import MAX_PREFIX, _check_cancelled

SUPPORTED = {"llama", "qwen2", "qwen3", "olmo3", "gemma2", "gemma3_text"}


class NativeModel:
    def __init__(self, model):
        self.model = model
        self.config = getattr(model.config, "text_config", None) or model.config
        if self.config.model_type not in SUPPORTED:
            raise ValueError(f"Native inspection does not support {self.config.model_type} models yet.")
        if model.training:
            raise ValueError("Native inspection requires evaluation mode.")
        self.layers = decoder_layers(model)
        self.device = model.get_input_embeddings().weight.device
        if self.device.type == "meta" or any(p.device != self.device for p in model.parameters()):
            raise ValueError("Native inspection requires all model weights on one device.")
        if "disk" in (getattr(model, "hf_device_map", None) or {}).values():
            raise ValueError("Native inspection does not support offloaded weights.")
        self.last_only = "logits_to_keep" in inspect.signature(model.forward).parameters

    def site(self, component):
        kind, layer, index = component
        layer, index = integer(layer, "Layer"), integer(index, "Component index")
        if not 0 <= layer < len(self.layers):
            raise ValueError(f"Layer must be between 0 and {len(self.layers) - 1}.")
        block = self.layers[layer]
        if kind == "neuron":
            projection = block.mlp.down_proj
            size, width = projection.weight.shape[1], 1
        elif kind == "head":
            projection = block.self_attn.o_proj
            size = self.config.num_attention_heads
            if projection.weight.shape[1] % size:
                raise ValueError("This model does not expose contiguous attention-head outputs.")
            width = projection.weight.shape[1] // size
        else:
            raise ValueError("Choose a native neuron or attention head.")
        if not 0 <= index < size:
            raise ValueError(f"{kind.title()} index must be between 0 and {size - 1}.")
        return projection, slice(index * width, (index + 1) * width)

    def forward(self, ids, changes=(), observe=(), cancelled=None):
        import torch

        if not 2 <= len(ids) <= MAX_PREFIX:
            raise ValueError(f"Use between 2 and {MAX_PREFIX} prompt tokens.")
        vocab = self.model.get_input_embeddings().weight.shape[0]
        if any(type(t) is not int or not 0 <= t < vocab for t in ids):
            raise ValueError("A prompt token is outside the model vocabulary.")
        edits = defaultdict(list)
        for component, at, mode, value in changes:
            if type(at) is not int or not 0 <= at < len(ids) or mode not in ("scale", "replace"):
                raise ValueError("An intervention has an invalid position or mode.")
            edits[component].append((at, mode, value))
        components = list(dict.fromkeys([*observe, *edits]))
        by_projection = defaultdict(list)
        for component in components:
            module, span = self.site(component)
            by_projection[module].append((component, span))
        activations = {}

        def hook(entries):
            def change(_module, args):
                x = args[0]
                if x.ndim != 3 or x.shape[:2] != (1, len(ids)):
                    raise ValueError("The model returned an unsupported component shape.")
                edited = x.clone() if any(c in edits for c, _ in entries) else x
                for component, span in entries:
                    for at, mode, value in edits[component]:
                        replacement = torch.as_tensor(value, device=x.device, dtype=x.dtype)
                        if not bool(replacement.isfinite().all()):
                            raise ValueError("An intervention value is not finite.")
                        if mode == "scale":
                            edited[0, at, span] *= replacement
                        else:
                            edited[0, at, span] = replacement
                    captured = edited[0, :, span].detach().float().cpu().clone()
                    if not bool(captured.isfinite().all()):
                        raise ValueError("The model returned non-finite component activations.")
                    activations[component] = captured
                return (edited, *args[1:])
            return change

        with contextlib.ExitStack() as stack, torch.no_grad():
            _check_cancelled(cancelled)
            if cancelled is not None:
                for block in self.layers:
                    stack.callback(block.register_forward_pre_hook(lambda *_: _check_cancelled(cancelled)).remove)
            for module, entries in by_projection.items():
                stack.callback(module.register_forward_pre_hook(hook(entries)).remove)
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            output = self.model(input_ids=tensor, use_cache=False,
                                **({"logits_to_keep": 1} if self.last_only else {}))
            log_probs = output.logits[0, -1].float().log_softmax(-1)
            _check_cancelled(cancelled)
        return {"log_probs": log_probs, "activations": activations}

    def writes(self, component, activation):
        """Vocabulary projection heuristic; does not model downstream normalization."""
        import torch

        projection, span = self.site(component)
        with torch.no_grad():
            direction = projection.weight[:, span].float()
            if component[0] == "head":
                direction = direction @ torch.tensor(activation, device=direction.device, dtype=direction.dtype)
            else:
                direction = direction[:, 0]
            weight = self.model.get_output_embeddings().weight
            scores = (weight @ direction.to(weight.dtype)).float()
            count = min(6, len(scores))
            return scores.topk(count).indices.tolist(), (-scores).topk(count).indices.tolist()

    def neighborhood(self, ids, component, at, *, limit=6, cancelled=None):
        """Adjacent-layer neurons at the same token, with measured local ablations.

        Upstream candidates are ranked by activation × local gradient into
        the selected activation, then tested individually. Downstream nodes
        are ranked by their activation change after ablating the selection.
        Early stopping keeps gradients within the preceding/selected layers.
        This is a local search, not an exhaustive native attribution graph.
        """
        import torch

        if component[0] != "neuron":
            raise ValueError("Nearby-neuron discovery requires an MLP neuron selection.")
        selected, span = self.site(component)
        layer = component[1]
        previous = self.layers[layer-1].mlp.down_proj if layer > 0 else None
        following = self.layers[layer+1].mlp.down_proj if layer + 1 < len(self.layers) else None
        limit = max(1, min(12, integer(limit, "Neighbor count")))
        passes = 0

        class Captured(Exception):
            pass

        def capture(*, zero_selected=False, zero_previous=None, gradient=False):
            nonlocal passes
            found = {}

            def before_previous(_module, args):
                x = args[0]
                if zero_previous is not None:
                    x = x.clone()
                    x[0, at, zero_previous] = 0
                if gradient:
                    # Only this small part of the forward needs a graph.
                    torch.set_grad_enabled(True)
                    x = x.detach().requires_grad_(True)
                    found["previous_input"] = x
                return (x, *args[1:])

            def before_selected(_module, args):
                x = args[0]
                found["selected"] = x[0, at, span].sum()
                if gradient or following is None:
                    raise Captured()
                if zero_selected:
                    x = x.clone()
                    x[0, at, span] = 0
                return (x, *args[1:])

            def before_following(_module, args):
                found["following"] = args[0][0, at].detach().float().cpu().clone()
                raise Captured()

            with contextlib.ExitStack() as stack, torch.no_grad():
                _check_cancelled(cancelled)
                for block in self.layers:
                    stack.callback(block.register_forward_pre_hook(lambda *_: _check_cancelled(cancelled)).remove)
                if previous is not None:
                    stack.callback(previous.register_forward_pre_hook(before_previous).remove)
                stack.callback(selected.register_forward_pre_hook(before_selected).remove)
                if following is not None:
                    stack.callback(following.register_forward_pre_hook(before_following).remove)
                try:
                    self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False,
                               **({"logits_to_keep": 1} if self.last_only else {}))
                except Captured:
                    pass
                passes += 1
            _check_cancelled(cancelled)
            return found

        base = capture()
        result = {"position": at, "incoming": [], "outgoing": []}
        if previous is not None:
            found = capture(gradient=True)
            x = found["previous_input"]
            grad, = torch.autograd.grad(found["selected"], x)
            scores = (grad[0, at] * x[0, at]).detach().float().cpu()
            if not bool(scores.isfinite().all()):
                raise ValueError("The model returned non-finite neighborhood scores.")
            for index in scores.abs().topk(min(limit, len(scores))).indices.tolist():
                tested = capture(zero_previous=index)
                result["incoming"].append({"layer": layer-1, "index": index, "score": float(scores[index]),
                                           "activation_delta": float(tested["selected"].detach() - base["selected"].detach())})
            del found, x, grad
        if following is not None:
            changed = capture(zero_selected=True)
            difference = changed["following"] - base["following"]
            if not bool(difference.isfinite().all()):
                raise ValueError("The model returned non-finite neighborhood activations.")
            for index in difference.abs().topk(min(limit, len(difference))).indices.tolist():
                result["outgoing"].append({"layer": layer+1, "index": index,
                                           "baseline": float(base["following"][index]),
                                           "ablated": float(changed["following"][index]),
                                           "activation_delta": float(difference[index])})
        result["model_passes"] = passes
        return result
