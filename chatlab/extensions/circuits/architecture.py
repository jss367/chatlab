"""Where a transcoder sits in each supported decoder block, and the frozen model.

Attribution reads the model as a linear system. Three things make a
transformer nonlinear: the MLPs, the normalizations, and the attention
softmax. The MLPs are replaced by transcoders whose feature activations are
held fixed, and the other two are frozen at the values they took on the
prompt: every RMSNorm keeps the scale it computed, and every attention head
keeps the pattern it computed. What is left is linear in the residual
stream, so the gradient of any target with respect to any source, times the
source's value, is exactly that source's direct contribution to the target.

Freezing changes no values. A forward pass under :func:`frozen` computes the
same numbers as an ordinary one; it only changes which of them gradients flow
through.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

FROZEN_ATTENTION = "chatlab_frozen"

# model_type: (module the transcoder reads the output of, module whose output
# it writes). Gemma normalizes both sides of its MLP, and its transcoders were
# trained on the normalized input and the normalized output.
SITES = {
    "gemma2": ("pre_feedforward_layernorm", "post_feedforward_layernorm"),
    "gemma3_text": ("pre_feedforward_layernorm", "post_feedforward_layernorm"),
    "qwen3": ("post_attention_layernorm", "mlp"),
    "llama": ("post_attention_layernorm", "mlp"),
}


@dataclass
class Blocks:
    """The pieces of one loaded model attribution touches."""

    model: object
    inner: object
    layers: list
    embed: object
    final_norm: object
    unembed: object
    input_name: str
    output_name: str
    final_softcap: float | None

    def mlp_input(self, layer):
        return getattr(self.layers[layer], self.input_name)

    def mlp_output(self, layer):
        return getattr(self.layers[layer], self.output_name)

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype


def blocks(model):
    """Find the decoder blocks of a supported Transformers model, or explain why not."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    kind = getattr(text_config, "model_type", None)
    if kind not in SITES:
        raise ValueError(f"Circuit tracing does not support {kind or 'this'} models yet.")
    inner = getattr(model, "model", None)
    inner = getattr(inner, "language_model", None) or inner
    layers = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None)
    if layers is None or norm is None:
        raise ValueError("This model does not expose its decoder blocks where circuit tracing looks.")
    if model.training:
        raise ValueError("Circuit tracing requires the model in evaluation mode.")
    unembed = model.get_output_embeddings()
    if unembed is None or getattr(unembed, "bias", None) is not None:
        raise ValueError("Circuit tracing needs an output head without a bias.")
    embed = model.get_input_embeddings()
    import torch

    device = embed.weight.device
    if device.type == "meta" or any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("Circuit tracing requires all weights on one device; sharded or offloaded models are unsupported.")
    for target in (getattr(model, "hf_device_map", None) or {}).values():
        if target == "disk":
            raise ValueError("Circuit tracing does not support offloaded weights.")
        mapped = torch.device(f"cuda:{target}" if type(target) is int else target)
        if mapped.type == "cpu":
            mapped = torch.device("cpu")
        elif mapped.type == device.type and mapped.index is None and device.index is not None:
            mapped = torch.device(mapped.type, device.index)
        if mapped != device:
            raise ValueError("Circuit tracing requires one device and does not support sharded or offloaded weights.")
    input_name, output_name = SITES[kind]
    return Blocks(
        model=model, inner=inner, layers=list(layers), embed=embed, final_norm=norm, unembed=unembed,
        input_name=input_name, output_name=output_name,
        final_softcap=getattr(text_config, "final_logit_softcapping", None),
    )


def _norm_forward(module):
    """An RMSNorm whose scale is computed but not differentiated through."""
    import torch

    eps = getattr(module, "eps", None)
    if eps is None:
        eps = getattr(module, "variance_epsilon", None)
    weight = getattr(module, "weight", None)
    if eps is None or weight is None:
        raise ValueError(f"{type(module).__name__} is not a normalization circuit tracing can freeze.")
    # Gemma stores the scale as an offset from one and multiplies in float32.
    gemma = type(module).__name__.startswith("Gemma")

    def forward(x):
        wide = x.float()
        scale = torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps).detach()
        if gemma:
            return (wide * scale * (1.0 + weight.float())).type_as(x)
        return weight * (wide * scale).to(x.dtype)

    return forward


def frozen_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                     softcap=None, **_kwargs):
    """Eager attention whose pattern is a constant to autograd."""
    import torch

    if scaling is None:
        scaling = module.head_dim ** -0.5
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    pattern = torch.softmax(scores.float(), dim=-1).to(query.dtype).detach()
    return torch.matmul(pattern, value).transpose(1, 2).contiguous(), pattern


def _register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    AttentionInterface.register(FROZEN_ATTENTION, frozen_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(FROZEN_ATTENTION, eager_mask)


@contextlib.contextmanager
def frozen(model):
    """Freeze every RMSNorm scale and attention pattern for the duration.

    The caller holds the model lock: the attention implementation is a field
    of the model's configuration, so a response generated meanwhile would run
    through the frozen version.
    """
    _register_attention()
    configs = {id(c): c for c in _configs(model)}.values()
    previous = [(c, c._attn_implementation) for c in configs]
    norms = [m for m in model.modules() if type(m).__name__.endswith("RMSNorm")]
    with contextlib.ExitStack() as stack:
        for module in norms:
            module.forward = _norm_forward(module)
            stack.callback(module.__dict__.pop, "forward", None)
        for config, _ in previous:
            config._attn_implementation = FROZEN_ATTENTION
        try:
            yield
        finally:
            for config, implementation in previous:
                config._attn_implementation = implementation


def _configs(model):
    config = model.config
    found = [config]
    for name in ("text_config",):
        sub = getattr(config, name, None)
        if sub is not None:
            found.append(sub)
    for module in model.modules():
        sub = getattr(module, "config", None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            found.append(sub)
    return found


@contextlib.contextmanager
def replaced_output(module, function):
    """Make ``module`` return ``function(input)`` instead of computing its own output."""
    module.forward = function
    try:
        yield
    finally:
        module.__dict__.pop("forward", None)
