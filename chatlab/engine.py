"""What the manager asks of a backend, written down once.

:class:`model_runtime.ModelManager` runs a Transformers model through
:class:`torch_engine.TorchEngine` and an MLX one through
:class:`mlx_runtime.MlxEngine`, and every caller past the load - generation,
scoring, the logit lens, the key-value cache view - talks to whichever it is
through the same handful of methods. Nothing made the two agree on those
methods but care, so :class:`Engine` names them. It is a structural type, not
a base class: neither engine inherits from it, and a test's stand-in need
only answer the parts it is asked.

:class:`LensReading` lives here for the same reason. It is what either
engine's :meth:`Engine.inspect_step` returns, so it belongs to neither.

Nothing here imports torch or mlx: the MLX engine is imported only when an
MLX model is loaded, and this module is read either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from chatlab import kv_cache


@dataclass
class LensReading:
    """What one inspection step saw, in numpy, whatever backend produced it.

    ``layer_logits`` holds one logit vector per layer below the last, read
    through the final norm and the head as though the stack had ended there;
    it is empty when the reading could not be trusted (see
    :meth:`torch_engine.TorchEngine.inspect_step` and
    :meth:`mlx_runtime.MlxEngine.inspect_step`). ``final_logits`` is the
    model's own output. ``layer_count`` is how many residual states were seen,
    which numbers the final row even when the intermediate ones are withheld.
    ``attention`` has one row per layer, each over the keys that layer could
    see, or nothing when the weights could not be recovered.
    """

    final_logits: np.ndarray
    layer_count: int
    layer_logits: list[np.ndarray] = field(default_factory=list)
    attention: list[list[float]] = field(default_factory=list)
    cache: Any = None


class Logits(Protocol):
    """One forward pass's logits, read a row at a time as float32 numpy.

    What :meth:`Engine.forward` hands back: the tensor stays on the device in
    the backend's own type, and only the rows a caller reads are copied off.
    """

    def row(self, index: int) -> np.ndarray:
        """Position ``index``'s logits over the whole vocabulary."""
        ...

    def best(self) -> tuple[np.ndarray, np.ndarray]:
        """Each position's highest-scoring token and its logit."""
        ...

    def pinned(self, token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """``token_id``'s rank (1 is best) and logit at each position."""
        ...


@runtime_checkable
class Engine(Protocol):
    """The questions the manager asks of whichever backend runs the model.

    Runtime-checkable so a test can confirm both engines answer all of them;
    nothing in the application branches on ``isinstance``. The caches are
    typed ``Any`` because each backend's is its own - a Transformers
    ``DynamicCache`` or a list of mlx-lm layer caches - and only the engine
    that made one reads it.
    """

    # ``"torch"`` or ``"mlx"``. The manager branches on it wherever one
    # backend can do what the other cannot: steering, activation patching,
    # the precisions a fitted lens was measured at.
    backend: str
    # The model in memory, in the backend's own type.
    model: Any

    def eos_token_ids(self) -> set[int]:
        """The stop tokens the model declares, as a set the caller may add to."""
        ...

    def forward(self, token_ids: Sequence[int], cache: Any, cached: int) -> tuple[Logits, Any]:
        """Feed ``token_ids`` after the ``cached`` tokens already in ``cache``.

        Returns the logits for every fed position and the cache that now
        holds them too.
        """
        ...

    def can_crop(self, cache: Any, held: int) -> bool:
        """Whether ``cache``, holding ``held`` tokens, can be cut back and still be right."""
        ...

    def crop(self, cache: Any, remove: int) -> None:
        """Cut the last ``remove`` tokens off ``cache``, in place."""
        ...

    def final_norm(self) -> Any:
        """The norm the LM head reads through, or ``None`` when none is found."""
        ...

    def read_head(self, vector: Any) -> Any:
        """Turn a normed residual vector into logits the way the model does."""
        ...

    def inspect_step(self, token_id: int, cache: Any, cached: int) -> LensReading:
        """Feed one token and read every layer's prediction and attention."""
        ...

    def cache_shapes(self, cache: Any) -> list[kv_cache.LayerShape | None]:
        """What every layer of ``cache`` holds, without copying any of it."""
        ...

    def cache_layer(self, cache: Any, layer: int, total: int) -> kv_cache.CacheLayer | None:
        """Layer ``layer`` (from 0) of a cache holding ``total`` tokens, in numpy."""
        ...

