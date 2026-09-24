"""What the key-value cache holds: every head's keys and values, one layer at a time.

An inspection keeps the cache its pass built (see
:meth:`model_runtime.ModelManager._inspect_cache_for`), covering the
sequence through the query token. This module turns one layer of it into
what the Layers and attention panel draws: how long each head's key and
value vectors are at every position, and how closely each key points the
way the query position's own key does. The engines hand the layer over as
numpy arrays, so the arithmetic is the same for a Transformers cache and an
MLX one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The most positions a reading carries. A long context would otherwise put
# tens of thousands of cells in the page; the most recent are kept, since
# the query sits at the end.
MAX_POSITIONS = 1024


class CacheGone(RuntimeError):
    """The cache an inspection kept is no longer the one the caller asked about."""


class CacheBusy(RuntimeError):
    """The model is running something else, so its cache cannot be read yet."""


@dataclass(frozen=True)
class LayerShape:
    """How much one layer of the cache holds, read without copying it.

    ``nbytes`` counts keys and values together.
    """

    heads: int
    positions: int
    dim: int
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class CacheLayer:
    """One layer's keys and values, ``(heads, positions, dim)``, in float32.

    ``positions`` numbers each column of the arrays in the sequence. A
    sliding-window layer holds only its most recent tokens, so its numbers
    need not start at 0.
    """

    keys: np.ndarray
    values: np.ndarray
    positions: list[int]


def recent_positions(held: int, total: int) -> list[int]:
    """The sequence positions of a layer holding the last ``held`` of ``total`` tokens."""

    return list(range(total - held, total))


def read_layer(layer: CacheLayer) -> dict:
    """Per-head norms and similarities for one layer, as lists for the panel.

    ``key_norm`` and ``value_norm`` are ``[head][position]``. ``key_similarity``
    is the cosine between each key and the last position's key in the same
    head: the last position is the query, so this shows which cached keys
    point where the query token's own key does.

    Each head's mean key is taken out before the cosine. It adds the same
    amount to every attention score a query gives that head, so it changes
    nothing the query sees, and in many models it is large enough that every
    raw key points almost the same way (Qwen3's median cosine is about 0.9).
    Keys are stored after the rotary position embedding, so nearby positions
    also tend to look alike for that reason alone.
    """

    keys = layer.keys.astype(np.float32, copy=False)
    centered = (keys - keys.mean(axis=1, keepdims=True))[:, -MAX_POSITIONS:, :]
    keys = keys[:, -MAX_POSITIONS:, :]
    values = layer.values[:, -MAX_POSITIONS:, :].astype(np.float32, copy=False)
    positions = list(layer.positions)[-MAX_POSITIONS:]
    key_norm = np.linalg.norm(keys, axis=-1)
    value_norm = np.linalg.norm(values, axis=-1)
    centered_norm = np.linalg.norm(centered, axis=-1)
    dots = np.einsum("hpd,hd->hp", centered, centered[:, -1, :])
    scale = centered_norm * centered_norm[:, -1:]
    similarity = np.divide(dots, scale, out=np.zeros_like(dots), where=scale > 0)
    return {
        "heads": int(keys.shape[0]),
        "dim": int(keys.shape[2]),
        "positions": positions,
        "held": len(layer.positions),
        "key_norm": key_norm.tolist(),
        "value_norm": value_norm.tolist(),
        "key_similarity": similarity.tolist(),
    }


def summarize(shapes: list[LayerShape | None], tokens: int) -> dict:
    """The whole cache in a few numbers, from each layer's shape.

    ``tokens`` is how long the sequence is; a sliding-window layer holds
    fewer. A layer that is ``None`` stores no keys (a state-space or
    linear-attention layer in a hybrid model) and is counted apart.
    """

    held = [shape for shape in shapes if shape is not None]
    return {
        "tokens": tokens,
        "layers": len(shapes),
        "attention_layers": len(held),
        "heads": sorted({shape.heads for shape in held}),
        "dims": sorted({shape.dim for shape in held}),
        "dtypes": sorted({shape.dtype for shape in held}),
        "positions": [shape.positions if shape is not None else 0 for shape in shapes],
        "nbytes": sum(shape.nbytes for shape in held),
    }
