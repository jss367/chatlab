"""The PyTorch backend: running a Transformers causal LM for the manager.

:class:`TorchEngine` answers the questions :class:`model_runtime.ModelManager`
asks of every backend - feed these tokens after that cache, cut the cache
back, read every layer's prediction for one token, hand over one layer of
the key-value cache - as :class:`mlx_runtime.MlxEngine` does for an MLX model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import numpy as np

import kv_cache
from mlx_runtime import LensReading


# The final norm of a decoder stack, under the names the common architectures
# give it. Llama, OLMo, Mistral and Qwen say ``norm``; GPT-2 says ``ln_f``;
# OPT and BLOOM say ``final_layer_norm``; Mamba says ``norm_f``. Some models
# keep it one level down from the base model: OPT's ``OPTModel`` wraps a
# ``decoder`` that owns the norm, and multimodal wrappers hold their text
# stack as ``language_model``.
FINAL_NORM_ATTRIBUTES = ("norm", "final_layer_norm", "ln_f", "final_norm", "norm_f")
FINAL_NORM_CONTAINERS = ("decoder", "transformer", "model", "language_model")


def _cache_can_crop(cache, held: int) -> bool:
    """Whether ``cache``, holding ``held`` tokens, can be cut back and still be right.

    A full-attention layer keeps every key and value, so cutting the tail off
    leaves exactly the prefix. A sliding-window layer keeps only its last
    ``sliding_window`` entries: once the sequence has reached the window,
    earlier entries are gone, and no cut can bring them back for a position
    that would have attended to them. Transformers refuses the crop outright
    in that case. Such a cache is rebuilt from the start instead. A cache
    that cannot say which layers slide, or how wide the window is, is not
    trusted either.
    """

    if not hasattr(cache, "crop"):
        return False
    # A hybrid model's linear-attention layers hold a running state rather
    # than one entry per token, so there is no earlier state to cut back to;
    # their crop raises. Transformers says so through is_croppable.
    if not getattr(cache, "is_croppable", True):
        return False
    sliding = getattr(cache, "is_sliding", None) or []
    layers = getattr(cache, "layers", None) or []
    for index, is_sliding in enumerate(sliding):
        if not is_sliding:
            continue
        layer = layers[index] if index < len(layers) else None
        window = getattr(layer, "sliding_window", None)
        if not isinstance(window, int) or held >= window:
            return False
    return True


class TorchLogits:
    """One forward pass's logits, read a row at a time as float32 numpy."""

    def __init__(self, logits) -> None:
        self.logits = logits

    def row(self, index: int) -> np.ndarray:
        return self.logits[index].detach().float().cpu().numpy()

    def best(self) -> tuple[np.ndarray, np.ndarray]:
        """Each position's highest-scoring token and its logit, as numpy."""
        values, ids = self.logits.detach().float().max(dim=-1)
        return ids.cpu().numpy(), values.cpu().numpy()

    def pinned(self, token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """``token_id``'s rank (1 is best) and logit at each position, as numpy."""
        rows = self.logits.detach().float()
        scores = rows[:, token_id]
        ranks = (rows > scores[:, None]).sum(dim=-1) + 1
        return ranks.cpu().numpy().astype(np.int64), scores.cpu().numpy()


class TorchEngine:
    """Run a Transformers causal LM for :class:`ModelManager`.

    The manager asks the same questions of every backend - feed these
    tokens after that cache, cut the cache back, read every layer's
    prediction for one token - and this answers them for a PyTorch model,
    as :class:`mlx_runtime.MlxEngine` does for an MLX one. Built on demand
    around whatever is in :attr:`ModelManager.model`, so it holds no state
    of its own.
    """

    backend = "torch"

    def __init__(self, model) -> None:
        self.model = model

    def eos_token_ids(self) -> set[int]:
        values: set[int] = set()
        generation_config = getattr(self.model, "generation_config", None)
        candidate = getattr(generation_config, "eos_token_id", None)
        if isinstance(candidate, int):
            values.add(candidate)
        elif candidate:
            values.update(int(value) for value in candidate)
        return values

    def _device(self):
        return next(self.model.parameters()).device

    def forward(self, token_ids: Sequence[int], cache, cached: int) -> tuple[TorchLogits, Any]:
        """Feed ``token_ids`` after the ``cached`` tokens already in ``cache``."""

        import torch

        device = self._device()
        outputs = self.model(
            input_ids=torch.tensor([list(token_ids)], dtype=torch.long, device=device),
            attention_mask=torch.ones(
                (1, cached + len(token_ids)), dtype=torch.long, device=device
            ),
            past_key_values=cache,
            use_cache=True,
        )
        return TorchLogits(outputs.logits[0]), outputs.past_key_values

    @staticmethod
    def can_crop(cache, held: int) -> bool:
        return _cache_can_crop(cache, held)

    @staticmethod
    def crop(cache, remove: int) -> None:
        # A negative count removes that many tokens from the end. A
        # positive one is the older "length to keep" form, which
        # Transformers 5.x warns about and 5.18 drops.
        cache.crop(-remove)

    @contextlib.contextmanager
    def eager_attention(self):
        """Run the model with attention that reports its weights.

        Fused kernels (SDPA, flash) never materialize the attention matrix, so
        a model loaded with one of them returns no weights. Eager attention is
        slower, so it is switched on for a single inspection step and switched
        back afterwards.
        """

        model = self.model
        switch = getattr(model, "set_attn_implementation", None)
        current = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if switch is None or current in (None, "eager"):
            yield
            return
        switch("eager")
        try:
            yield
        finally:
            switch(current)

    def final_norm(self):
        """The norm the LM head reads through, or ``None`` when none is found.

        Looked up on the base model first, then one level down in the
        containers some architectures wrap their decoder stack in.
        """

        import torch

        base = getattr(self.model, "base_model", self.model)
        owners = [base] + [getattr(base, name, None) for name in FINAL_NORM_CONTAINERS]
        for owner in owners:
            for name in FINAL_NORM_ATTRIBUTES:
                module = getattr(owner, name, None)
                if isinstance(module, torch.nn.Module):
                    return module
        return None

    def read_head(self, vector):
        """Turn a normed residual vector into logits the way the model does.

        Some causal-LM heads post-process the unembedding: Gemma 2 and 3
        soft-cap logits with ``tanh``, Granite divides by ``logits_scaling``,
        Cohere multiplies by ``logit_scale``. An intermediate reading that
        skipped them would describe a distribution the model never emits, so
        they are applied here. :meth:`inspect_step` checks the result against
        the model's own output for the final layer, which catches a transform
        this list does not know about.
        """

        import torch

        model = self.model
        logits = model.get_output_embeddings()(vector)
        config = getattr(model, "config", None)
        scale = getattr(config, "logit_scale", None)
        if scale:
            logits = logits * scale
        scaling = getattr(config, "logits_scaling", None)
        if scaling:
            logits = logits / scaling
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def inspect_step(self, token_id: int, cache, cached: int) -> LensReading:
        """Feed one token with the hidden states and attention switched on.

        The last hidden state is what the model's own head reads, so its
        row is the real output; the earlier ones are read through the
        final norm as though the stack had ended there. Without the norm
        those readings would be off by a rescaling the head never sees,
        so a model whose norm cannot be found shows its output alone
        rather than intermediate rows that look right and are not. The
        same goes for a head that post-processes its logits in a way
        :meth:`read_head` does not replicate: reading the final hidden
        state (already normed) through it must reproduce the model's
        output, or the intermediate rows are not trustworthy either.
        """

        import torch

        device = self._device()
        with self.eager_attention():
            outputs = self.model(
                input_ids=torch.tensor([[int(token_id)]], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, cached + 1), dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
            )

        def to_numpy(tensor) -> np.ndarray:
            return tensor.detach().float().cpu().numpy()

        hidden_states = tuple(outputs.hidden_states or ())
        final_logits = outputs.logits[0, -1]
        reading = LensReading(
            final_logits=to_numpy(final_logits),
            layer_count=len(hidden_states),
            cache=getattr(outputs, "past_key_values", None),
        )
        norm = self.final_norm()
        readable = norm is not None and bool(hidden_states)
        if readable:
            replayed = self.read_head(hidden_states[-1][0, -1]).detach().float()
            readable = torch.allclose(
                replayed, final_logits.detach().float(), rtol=1e-2, atol=1e-2
            )
        if readable:
            for state in hidden_states[:-1]:
                vector = norm(state[0, -1].unsqueeze(0)).squeeze(0)
                reading.layer_logits.append(to_numpy(self.read_head(vector)))

        weights = tuple(outputs.attentions or ())
        if weights and all(layer is not None for layer in weights):
            reading.attention = [
                layer[0, :, -1, :].detach().float().mean(dim=0).cpu().tolist()
                for layer in weights
            ]
        del outputs
        return reading

    @staticmethod
    def _cache_tensors(cache) -> list[tuple[Any, Any] | None]:
        """Each layer's key and value tensors, ``None`` where a layer holds none.

        A ``DynamicCache`` keeps them on its ``layers``, in order, with a
        sliding-window layer holding only its most recent tokens. A layer
        without attention (a hybrid model's state-space or linear layers)
        has no keys, and a quantized layer's ``keys`` are only the part not
        yet quantized, so neither is read.
        """

        import torch

        layers = getattr(cache, "layers", None)
        if layers is None:
            return []
        pairs: list[tuple[Any, Any] | None] = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            readable = (
                "Quantized" not in type(layer).__name__
                and isinstance(keys, torch.Tensor)
                and isinstance(values, torch.Tensor)
                and keys.ndim == 4
                and values.ndim == 4
                and keys.shape[2] > 0
            )
            pairs.append((keys, values) if readable else None)
        return pairs

    def cache_shapes(self, cache) -> list[kv_cache.LayerShape | None]:
        """What every layer of ``cache`` holds, without copying any of it."""

        return [
            None
            if pair is None
            else kv_cache.LayerShape(
                heads=int(pair[0].shape[1]),
                positions=int(pair[0].shape[2]),
                dim=int(pair[0].shape[3]),
                dtype=str(pair[0].dtype).removeprefix("torch."),
                nbytes=sum(tensor.numel() * tensor.element_size() for tensor in pair),
            )
            for pair in self._cache_tensors(cache)
        ]

    def cache_layer(self, cache, layer: int, total: int) -> kv_cache.CacheLayer | None:
        """Layer ``layer`` (from 0) of a cache holding ``total`` tokens, in numpy.

        Only the latest :data:`kv_cache.MAX_POSITIONS` positions are copied
        off the device; the mean key over every position is reduced there.
        """

        import torch

        pairs = self._cache_tensors(cache)
        if not 0 <= layer < len(pairs) or pairs[layer] is None:
            return None
        keys, values = (tensor[0].detach() for tensor in pairs[layer])
        held = int(keys.shape[1])
        shown = min(held, kv_cache.MAX_POSITIONS)
        return kv_cache.CacheLayer(
            keys=keys[:, -shown:].float().cpu().numpy(),
            values=values[:, -shown:].float().cpu().numpy(),
            positions=kv_cache.recent_positions(shown, total),
            key_mean=keys.mean(dim=1, dtype=torch.float32).cpu().numpy(),
            held=held,
        )
