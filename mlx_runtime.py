"""The MLX backend: quantized ``mlx-community`` checkpoints on Apple silicon.

A repository converted with ``mlx_lm.convert`` keeps its weights in
safetensors files under the names Transformers uses, but quantized MLX's way:
each linear layer is a packed matrix plus per-group scales and biases, and
``AutoModelForCausalLM`` cannot read them. mlx-lm can, and runs them on the
GPU through Metal with the memory the packing promises - a 7B model at four
bits takes about 4 GB.

Everything the token measurements need survives the change of runtime. The
model is a plain Python object whose forward pass returns logits for every
position, so ranks, surprise, entropy and the alternatives are read the same
way they are from a Transformers model. Its decoder layers are a list on the
inner model, so the residual stream between them can be recorded for the
logit lens by wrapping each layer for the one step an inspection takes,
and the attention weights can be recovered by standing in for the
attention kernel during that same step. What is different is hidden behind
:class:`MlxEngine`, which answers the same questions
:class:`model_runtime.TorchEngine` does.

Imported lazily by :mod:`model_runtime`, and only when an MLX repository is
loaded: mlx installs on Apple silicon alone, and nothing here is needed to
list the cache or to run a Transformers model.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Where an mlx-lm ``Model`` keeps its decoder stack and its final norm. Most
# architectures wrap them in ``model``; the GPT-2 family in ``transformer``.
# The same containers :data:`model_runtime.FINAL_NORM_CONTAINERS` lists, and
# the same norm names, so the two lenses look in the same places.
LAYER_CONTAINERS = ("model", "transformer", "decoder", "language_model")
FINAL_NORM_ATTRIBUTES = ("norm", "final_layer_norm", "ln_f", "final_norm", "norm_f")
EMBEDDING_ATTRIBUTES = ("embed_tokens", "wte", "embeddings")

# The name mlx-lm's model files import the attention kernel under. Each
# architecture module binds its own copy with ``from .base import ...``, so
# the stand-in has to be placed in the module the layer came from, not in
# ``mlx_lm.models.base``.
ATTENTION_FUNCTION = "scaled_dot_product_attention"


@cache
def mlx_available() -> bool:
    """Whether mlx and mlx-lm are installed, without importing either.

    Asked on every cache scan, so it must cost nothing: a spec lookup reads
    the import path and no module body. The answer cannot change while the
    process runs, so it is kept.
    """

    try:
        return (
            importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )
    except (ImportError, ValueError):
        return False


def mlx_quantization(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ``quantization`` block of an MLX-converted config, or ``None``.

    ``mlx_lm.convert`` writes the group size and bit width under
    ``quantization`` and, for newer releases, a copy under
    ``quantization_config``. Transformers keeps its own quantizers' settings
    under the second name too, always with a ``quant_method``; a block with
    ``bits`` and no method is MLX's. Either spelling is accepted so a repo
    that carries only one is still recognised.
    """

    if not isinstance(config, Mapping):
        return None
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if not isinstance(block, Mapping) or "quant_method" in block:
            continue
        bits = block.get("bits")
        if isinstance(bits, int) and not isinstance(bits, bool) and bits > 0:
            return dict(block)
    return None


def read_mlx_config(snapshot: Path) -> dict[str, Any] | None:
    """The root ``config.json`` of ``snapshot`` when it describes an MLX conversion."""

    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or "model_type" not in config:
        return None
    return config if mlx_quantization(config) is not None else None


def precision_label(config: Mapping[str, Any] | None) -> str:
    """``"4-bit"``, from the config's quantization block; ``"full"`` when it has none."""

    block = mlx_quantization(config)
    return f"{block['bits']}-bit" if block else "full"


# How mlx-community names a conversion's width: ``-4bit``, ``-8bit``,
# ``-3bit``, sometimes with a suffix such as ``-4bit-DWQ``. A name that says
# ``bf16`` or ``fp16`` is a conversion at full width.
BITS_IN_NAME = re.compile(r"(?<![0-9])([1-9])[-_]?bit(?![0-9])", re.IGNORECASE)


def bits_from_name(model_id: str) -> int | None:
    """The bit width a repository's name claims, or ``None`` when it says nothing.

    For a search result, which is a name and a parameter count and nothing
    about how the weights are packed: the hub's copy of the config drops the
    quantization block. mlx-community spells the width into the name, so it
    is read from there, and a name that keeps quiet is judged whole rather
    than at a width it may not have.
    """

    match = BITS_IN_NAME.search(model_id.rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


def mlx_supports(model_type: str | None) -> bool:
    """Whether mlx-lm has an implementation for ``model_type``.

    Answered the way ``mlx_lm.utils._get_classes`` answers it - through the
    remapping table and then a module lookup - but without importing the
    module, so a hub search can ask it a hundred times.
    """

    if not model_type or not mlx_available():
        return False
    try:
        from mlx_lm.utils import MODEL_REMAPPING
    except ImportError:
        MODEL_REMAPPING = {}
    name = MODEL_REMAPPING.get(model_type, model_type)
    try:
        return importlib.util.find_spec(f"mlx_lm.models.{name}") is not None
    except (ImportError, ValueError):
        return False


def active_bytes() -> int | None:
    """Bytes MLX holds on the device right now, or ``None`` before it is imported.

    Read off the module already in ``sys.modules`` rather than imported here,
    so a machine that has never loaded an MLX model never pays the import
    for a memory reading.
    """

    mx = sys.modules.get("mlx.core")
    if mx is None:
        return None
    try:
        return int(mx.get_active_memory())
    except (AttributeError, RuntimeError):
        return None


def clear_cache() -> None:
    """Hand MLX's cached device buffers back, if MLX is imported."""

    mx = sys.modules.get("mlx.core")
    if mx is None:
        return
    try:
        mx.clear_cache()
    except (AttributeError, RuntimeError):
        pass


def read_mlx_model(local_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Read one MLX checkpoint out of ``local_path``: the model, its tokenizer, its config.

    The tokenizer is Transformers' own rather than mlx-lm's wrapper around
    it, because everything downstream - the chat template, the offsets that
    split context from text, the special-token list - talks to the
    Transformers interface directly, and the wrapper proxies only some of it.
    """

    try:
        from mlx_lm.utils import load_model
    except ImportError as error:
        raise RuntimeError(
            "MLX models need the mlx-lm package on Apple silicon: run "
            f"`pip install mlx-lm` and load again. ({error})"
        ) from error
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model, config = load_model(Path(local_path), lazy=False)
    return model, tokenizer, config


class MlxLogits:
    """One forward pass's logits, read a row at a time as float32 numpy."""

    def __init__(self, logits) -> None:
        self.logits = logits

    def row(self, index: int) -> np.ndarray:
        import mlx.core as mx

        return np.array(self.logits[0, index].astype(mx.float32))


@dataclass
class LensReading:
    """What one inspection step saw, in numpy, whatever backend produced it.

    ``layer_logits`` holds one logit vector per layer below the last, read
    through the final norm and the head as though the stack had ended there;
    it is empty when the reading could not be trusted (see
    :meth:`MlxEngine.inspect_step`). ``final_logits`` is the model's own
    output. ``layer_count`` is how many residual states were seen, which
    numbers the final row even when the intermediate ones are withheld.
    ``attention`` has one row per layer, each over the keys that layer could
    see, or nothing when the weights could not be recovered.
    """

    final_logits: np.ndarray
    layer_count: int
    layer_logits: list[np.ndarray] = field(default_factory=list)
    attention: list[list[float]] = field(default_factory=list)
    cache: Any = None


class _RecordingLayer:
    """Stand in for one decoder layer and keep what flowed through it."""

    def __init__(self, layer, index: int, recorder: _Recorder) -> None:
        self.layer = layer
        self.index = index
        self.recorder = recorder

    def __call__(self, hidden, *args, **kwargs):
        if self.index == 0:
            self.recorder.hidden.append(hidden)
        self.recorder.current = self.index
        output = self.layer(hidden, *args, **kwargs)
        self.recorder.hidden.append(output)
        return output

    def __getattr__(self, name: str):
        return getattr(self.layer, name)


@dataclass
class _Recorder:
    hidden: list = field(default_factory=list)
    attention: dict[int, Any] = field(default_factory=dict)
    current: int = -1


class MlxEngine:
    """Run an mlx-lm model for :class:`model_runtime.ModelManager`."""

    backend = "mlx"

    def __init__(self, model, config: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.config = dict(config or {})

    @classmethod
    def from_snapshot(cls, model, local_path: Path) -> MlxEngine:
        try:
            config = json.loads((Path(local_path) / "config.json").read_text())
        except (OSError, ValueError):
            config = {}
        return cls(model, config if isinstance(config, dict) else {})

    # -- what the manager asks about the model -------------------------------

    def eos_token_ids(self) -> set[int]:
        """The stop tokens the config names, which a converted repo keeps there."""

        values: set[int] = set()
        candidate = self.config.get("eos_token_id")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            values.add(candidate)
        elif isinstance(candidate, (list, tuple)):
            values.update(int(value) for value in candidate if isinstance(value, int))
        return values

    @staticmethod
    def device_bytes() -> int | None:
        return active_bytes()

    @staticmethod
    def release() -> None:
        clear_cache()

    # -- the forward pass ---------------------------------------------------

    def new_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def forward(self, token_ids: Sequence[int], cache, cached: int) -> tuple[MlxLogits, Any]:
        """Feed ``token_ids`` after the ``cached`` tokens already in ``cache``.

        The cache carries its own offset, so ``cached`` is only checked
        against it: a disagreement means the caller's bookkeeping and the
        cache have parted ways, and silently continuing would attach the
        wrong positions to every token that followed.
        """

        import mlx.core as mx

        if cache is None:
            cache = self.new_cache()
        offset = _cache_offset(cache)
        if offset is not None and offset != cached:
            raise RuntimeError(
                f"The MLX cache holds {offset} tokens, not the {cached} expected."
            )
        logits = self.model(mx.array([list(token_ids)]), cache=cache)
        mx.eval(logits)
        return MlxLogits(logits), cache

    # -- the inspection cache -----------------------------------------------

    @staticmethod
    def can_crop(cache, held: int) -> bool:
        """Whether every layer's cache can be cut back from ``held`` tokens.

        A rotating cache that has already dropped its oldest keys cannot be
        cut back to a state it no longer holds, and says so through
        ``is_trimmable``; a cache of a shape this does not know is not
        trusted either.
        """

        del held
        try:
            return bool(cache) and all(layer.is_trimmable() for layer in cache)
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def crop(cache, remove: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache

        trim_prompt_cache(cache, remove)

    # -- the logit lens -----------------------------------------------------

    def _inner(self):
        """The module that owns the decoder layers and the final norm."""

        model = self.model
        for name in LAYER_CONTAINERS:
            inner = getattr(model, name, None)
            if inner is not None and isinstance(getattr(inner, "layers", None), list):
                return inner
        return model if isinstance(getattr(model, "layers", None), list) else None

    def final_norm(self):
        inner = self._inner()
        for owner in (inner, self.model):
            if owner is None:
                continue
            for name in FINAL_NORM_ATTRIBUTES:
                module = getattr(owner, name, None)
                if module is not None and callable(module):
                    return module
        return None

    def read_head(self, vector):
        """Turn a normed residual vector into logits the way the model does.

        The head is ``lm_head`` where the repo keeps one, and otherwise the
        embedding matrix read backwards, which is what mlx-lm's tied models
        do. The post-processing some architectures apply - Gemma's soft-cap,
        Granite's division, Cohere's multiplication - is replicated from the
        config, and :meth:`inspect_step` checks the result against the
        model's own output so a transform this does not know about withholds
        the intermediate rows rather than mislabelling them.
        """

        import mlx.core as mx

        model = self.model
        head = getattr(model, "lm_head", None)
        if head is not None and callable(head):
            logits = head(vector)
        else:
            inner = self._inner()
            embedding = None
            for name in EMBEDDING_ATTRIBUTES:
                embedding = getattr(inner, name, None) if inner is not None else None
                if embedding is not None:
                    break
            if embedding is None or not hasattr(embedding, "as_linear"):
                raise RuntimeError("This MLX model's output head could not be found.")
            logits = embedding.as_linear(vector)
        scale = self.config.get("logit_scale")
        if scale:
            logits = logits * scale
        scaling = self.config.get("logits_scaling")
        if scaling:
            logits = logits / scaling
        softcap = self.config.get("final_logit_softcapping")
        if softcap:
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    @contextlib.contextmanager
    def _recording(self, recorder: _Recorder) -> Iterator[None]:
        """Wrap the decoder layers and the attention kernel for one step.

        The layers are replaced on the inner model with recorders that keep
        each residual state, and every model module a layer came from has
        its attention function replaced with one that also computes the
        weights for the query it is given. Both are undone afterwards
        whatever happens, so a failed inspection leaves the model as it was.
        """

        inner = self._inner()
        if inner is None:
            yield
            return
        original = inner.layers
        wrapped = [
            _RecordingLayer(layer, index, recorder) for index, layer in enumerate(original)
        ]
        patched: list[tuple[Any, Any]] = []
        modules = {sys.modules.get(type(layer).__module__) for layer in original}
        for module in modules:
            function = getattr(module, ATTENTION_FUNCTION, None)
            if module is None or function is None:
                continue
            patched.append((module, function))
            setattr(module, ATTENTION_FUNCTION, _recording_attention(function, recorder))
        inner.layers = wrapped
        try:
            yield
        finally:
            inner.layers = original
            for module, function in patched:
                setattr(module, ATTENTION_FUNCTION, function)

    def inspect_step(self, token_id: int, cache, cached: int) -> LensReading:
        """Feed one token and read every layer's prediction and attention.

        The intermediate rows are given only when reading the last residual
        state through the norm and the head reproduces the model's own
        logits: that is the one check that catches a head transform, an
        embedding scale or a norm this reader does not know about, and a
        row that looked right and was not would be worse than none.
        """

        import mlx.core as mx

        if cache is None:
            cache = self.new_cache()
        offset = _cache_offset(cache)
        if offset is not None and offset != cached:
            raise RuntimeError(
                f"The MLX cache holds {offset} tokens, not the {cached} expected."
            )
        recorder = _Recorder()
        with self._recording(recorder):
            logits = self.model(mx.array([[int(token_id)]]), cache=cache)
        hidden = list(recorder.hidden)
        mx.eval(logits, *hidden, *recorder.attention.values())
        final = np.array(logits[0, -1].astype(mx.float32))
        reading = LensReading(final_logits=final, layer_count=len(hidden), cache=cache)

        norm = self.final_norm()
        if norm is not None and hidden:
            try:
                replayed = np.array(
                    self.read_head(norm(hidden[-1][:, -1:, :]))[0, -1].astype(mx.float32)
                )
                readable = np.allclose(replayed, final, rtol=1e-2, atol=1e-2)
            except Exception:  # noqa: BLE001 - an unreadable head withholds rows
                logger.debug("The MLX head could not be replayed", exc_info=True)
                readable = False
            if readable:
                for state in hidden[:-1]:
                    read = self.read_head(norm(state[:, -1:, :]))[0, -1]
                    reading.layer_logits.append(np.array(read.astype(mx.float32)))

        # One row per layer or none: a stack where some layers record no
        # attention (state-space layers, say) cannot be laid against the
        # lens rows, and a partial strip would mislabel every layer after
        # the first gap.
        layers = len(hidden) - 1
        if layers > 0 and all(index in recorder.attention for index in range(layers)):
            reading.attention = [
                np.array(recorder.attention[index]).astype(float).tolist()
                for index in range(layers)
            ]
        return reading


def _cache_offset(cache) -> int | None:
    """How many tokens ``cache`` holds, from its first layer, or ``None``."""

    try:
        first = cache[0]
    except (TypeError, IndexError, KeyError):
        return None
    offset = getattr(first, "offset", None)
    return int(offset) if isinstance(offset, int) else None


def _recording_attention(original, recorder: _Recorder):
    """Wrap mlx-lm's attention kernel to also compute the weights it never returns.

    ``mx.fast.scaled_dot_product_attention`` is fused and materializes no
    weight matrix, so the weights are recomputed here from the same queries
    and keys: scores scaled and masked as the kernel masks them, softmaxed,
    then averaged over the heads for the one query an inspection feeds. The
    kernel's own output is what the model receives, so the recording changes
    nothing about the forward pass. Anything unusual - a quantized cache, a
    multi-token query, attention sinks - records nothing for that layer,
    and :meth:`MlxEngine.inspect_step` then withholds the strip.
    """

    def attention(queries, keys, values, cache, scale, mask, *args, **kwargs):
        output = original(queries, keys, values, cache, scale, mask, *args, **kwargs)
        layer = recorder.current
        if layer < 0 or layer in recorder.attention:
            return output
        try:
            if hasattr(cache, "bits") or kwargs.get("sinks") is not None or args:
                return output
            if queries.ndim != 4 or keys.ndim != 4 or queries.shape[2] != 1:
                return output
            recorder.attention[layer] = _attention_weights(queries, keys, scale, mask)
        except Exception:  # noqa: BLE001 - a strip is optional, the response is not
            logger.debug("Attention weights could not be recorded", exc_info=True)
        return output

    return attention


def _attention_weights(queries, keys, scale, mask):
    """Mean-over-heads attention of the last query over every key, as one row."""

    import mlx.core as mx

    heads, kv_heads = queries.shape[1], keys.shape[1]
    if kv_heads and heads != kv_heads and heads % kv_heads == 0:
        keys = mx.repeat(keys, heads // kv_heads, axis=1)
    scores = (queries.astype(mx.float32) * scale) @ keys.astype(mx.float32).transpose(
        0, 1, 3, 2
    )
    if mask is not None and not isinstance(mask, str):
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(mx.float32).min)
        else:
            scores = scores + mask.astype(mx.float32)
    weights = mx.softmax(scores, axis=-1)
    return weights[0, :, -1, :].mean(axis=0)
