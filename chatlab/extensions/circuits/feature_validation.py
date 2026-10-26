"""Observe and replace individual transcoder activations in the real model."""
from __future__ import annotations

import contextlib
from collections import defaultdict

from .attribution import MAX_PREFIX, _check_cancelled


class FeatureModel:
    def __init__(self, blocks, held):
        self.blocks, self.held = blocks, held

    def forward(self, ids, changes=(), observe=(), cancelled=None):
        import torch

        blocks, held = self.blocks, self.held
        if not 2 <= len(ids) <= MAX_PREFIX:
            raise ValueError(f"Use between 2 and {MAX_PREFIX} prompt tokens.")
        vocab = min(blocks.embed.weight.shape[0], blocks.unembed.weight.shape[0])
        if any(type(t) is not int or not 0 <= t < vocab for t in ids):
            raise ValueError("A prompt token is outside the model vocabulary.")
        edits = defaultdict(list)
        for component, at, mode, value in changes:
            if type(at) is not int or not 0 <= at < len(ids) or mode not in ("scale", "replace"):
                raise ValueError("An intervention has an invalid position or mode.")
            edits[component].append((at, mode, value))
        by_layer = defaultdict(list)
        for component in dict.fromkeys([*observe, *edits]):
            kind, layer, index = component
            if (kind != "feature" or type(layer) is not int or not 0 <= layer < len(blocks.layers)
                    or type(index) is not int or not 0 <= index < held.spec.width):
                raise ValueError("Choose a feature inside the loaded transcoder set.")
            by_layer[layer].append(component)
        activations, deltas = {}, {}

        def read(layer):
            def hook(_module, _args, output):
                x = (output[0] if isinstance(output, tuple) else output)[0].float()
                delta = torch.zeros_like(x)
                for component in by_layer[layer]:
                    feature = torch.tensor([component[2]], device=blocks.device)
                    pre = x @ held.encoder_rows(layer, feature).T + held.b_enc[layer][feature].float()
                    original = held.activate_one(layer, component[2], pre[:, 0])
                    changed = original.clone()
                    for at, mode, value in edits[component]:
                        value = torch.as_tensor(value, device=x.device, dtype=changed.dtype).reshape(())
                        if not bool(value.isfinite()):
                            raise ValueError("An intervention value is not finite.")
                        changed[at] = changed[at] * value if mode == "scale" else value
                    if not bool(changed.isfinite().all()):
                        raise ValueError("The model returned non-finite feature activations.")
                    activations[component] = changed[:, None].detach().cpu().clone()
                    delta += (changed - original)[:, None] * held.decoder_rows(layer, feature)[0]
                deltas[layer] = delta
            return hook

        def write(layer):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                changed = value.clone()
                changed[0] += deltas.pop(layer).to(value.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return hook

        with contextlib.ExitStack() as stack, torch.no_grad():
            _check_cancelled(cancelled)
            if cancelled is not None:
                for block in blocks.layers:
                    stack.callback(block.register_forward_pre_hook(lambda *_: _check_cancelled(cancelled)).remove)
            for layer in by_layer:
                stack.callback(blocks.mlp_input(layer).register_forward_hook(read(layer)).remove)
                stack.callback(blocks.mlp_output(layer).register_forward_hook(write(layer)).remove)
            ids_tensor = torch.tensor([ids], dtype=torch.long, device=blocks.device)
            hidden = blocks.inner(input_ids=ids_tensor, use_cache=False).last_hidden_state
            logits = blocks.unembed(hidden[:, -1])[0].float()
            if blocks.final_softcap:
                logits = torch.tanh(logits / blocks.final_softcap) * blocks.final_softcap
            _check_cancelled(cancelled)
        return {"log_probs": logits.log_softmax(-1), "activations": activations}
