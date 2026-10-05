"""Measuring a loaded model without generating: scoring, steering, lenses, patching.

Scoring text the model did not write, reading steering vectors out of it,
the logit lens and Jacobian lens over one token, activation patching between
two runs, and the key-value cache an inspection leaves behind. Each holds the
model lock for its own pass and refuses to run on weights other than the
ones the caller's tokens came from.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chatlab import jacobian_lens
from chatlab import kv_cache
from chatlab import model_loading
from chatlab import steering as steering_vectors
from chatlab.device_memory import reraise_out_of_memory
from chatlab.text_generation import ModelChanged
from chatlab.token_metrics import entropy_bits, normalize_log_probabilities
from chatlab.tokenization import (
    SCORE_TOKEN_LIMIT,
    encode_for_scoring,
    model_position_limit,
    score_token_limit,
)

class ProjectionCancelled(RuntimeError):
    """A stopped projection; all model hooks are removed before it escapes."""


logger = logging.getLogger(__name__)


# One steering example is read with every decoder block's output captured, so
# a long one costs a forward pass and nothing more - but a reader pasting an
# essay into the examples box is not describing a behaviour, they are asking
# for a summary. Short examples are also what the difference in means is for.
STEERING_EXAMPLE_TOKEN_LIMIT = 512

# How one example's tokens become one vector: the position the model would
# have written from, or the average over the whole example. The first is what
# a chat behaviour lives at; the second describes the passage as a whole.
STEERING_POOLS = ("last", "mean")


def _guards_device_memory(method):
    """Turn a run's out-of-memory failure into :class:`OutOfMemoryError`, and
    hand cached device memory back after ``method``, whatever its outcome.

    The caching allocator keeps every block a run freed, so a long prompt
    stays paid for until the next one. Returning it after each run keeps the
    process at the model's own size between requests.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (RuntimeError, MemoryError) as error:
            reraise_out_of_memory(error)
        finally:
            self._release_device_cache()

    return wrapper


@dataclass(frozen=True)
class ScoredText:
    """Per-token measurements for text the model did not generate.

    ``seam_verified`` and ``chat_template_missing`` carry
    :class:`SplitPassage`'s answers through to the interface, which says so
    rather than presenting approximate numbers as exact, or numbers for a
    plain passage as numbers for a chat turn.
    """

    context_metrics: list[dict]
    metrics: list[dict]
    seam_verified: bool = True
    chat_template_missing: bool = False
    context_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SteeringExtraction:
    """A steering direction taken from contrasting examples, at every layer.

    ``layers`` holds one vector per decoder block, indexed the way the
    steering hook indexes them, so ``layers[n]`` is the direction to add to
    block ``n``'s output. ``stats`` is the reading beside each one; see
    :func:`steering.contrast_directions`. The whole stack is returned rather
    than a chosen layer because the examples are read at every layer in the
    one pass: picking the layer afterwards costs nothing, and picking it
    beforehand would mean guessing.
    """

    model_id: str
    load_id: str
    layers: tuple[tuple[float, ...], ...]
    stats: tuple[dict, ...]
    positive_count: int
    negative_count: int
    pool: str
    chat_template_missing: bool = False


# Seconds :meth:`ModelManager.read_kv_cache` waits for the model lock. The
# cache is read in well under this; a longer wait means something else has
# the model and will release the cache when it starts.
KV_CACHE_WAIT = 2.0

# Said for a click on one of a picture's tokens, which the model was shown
# rather than asked to predict.
IMAGE_TOKEN_INSPECTED = (
    "This is one of a picture's tokens. The model was shown the picture here, "
    "so nothing predicted it. Inspect a token after the picture instead."
)


def _kept_images(kept) -> tuple[str, ...]:
    """The pictures behind a kept inspection cache: its fourth entry, when it has one."""

    return tuple(kept[3]) if len(kept) > 3 else ()


@dataclass(frozen=True)
class TokenInsight:
    """What every layer predicted for one token, and where the model looked.

    ``layers`` has one row per residual-stream reading, from the embeddings
    (layer 0) to the model's real output (the last row). Each intermediate
    reading is passed through the final norm and the unembedding, the logit
    lens: it says what the model would have answered had it stopped there.
    When the model's final norm cannot be found there is no honest way to
    take those readings, so only the output row is present. ``decided_at``
    is the first layer from which the token stayed the model's first choice,
    or ``None`` when it never was.

    ``attention`` is head-averaged, one row per decoder layer, one column per
    token before the inspected one, and it is empty when the model cannot
    return attention weights. A sliding-window layer sees only the most
    recent tokens; the columns for the rest hold zero. The query is the token *before* the inspected one: that
    is the position whose output predicted it.
    """

    index: int
    token_id: int
    token_text: str
    layers: list[dict]
    tokens: list[dict]
    attention: list[list[float]]
    decided_at: int | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "layers": [dict(row) for row in self.layers],
            "tokens": [dict(token) for token in self.tokens],
            "attention": [list(row) for row in self.attention],
            "decided_at": self.decided_at,
        }


# How many keys just before the query count as its recent context, in the
# attention trace's head ranking and in the view's region bars.
RECENT_KEYS = 16

# Float32 bytes one chunk of an attention trace may hold: every layer's
# every head, for each token fed, over the whole sequence so far. The
# Transformers backend returns all layers' weights at once, so this bounds
# the memory the trace takes on top of the model.
ATTENTION_TRACE_CHUNK_BYTES = 256 << 20

# The most tokens one chunk of an attention trace feeds, however small the
# model: eager attention's own scratch grows with it too.
ATTENTION_TRACE_CHUNK_TOKENS = 128


@dataclass(frozen=True)
class AttentionTrace:
    """Where every reply token's prediction looked, read in one pass.

    Step ``k`` explains reply token ``k``, the token at position
    ``context_count + k``; its query is the position before it, whose output
    predicted it. ``rows`` holds one row per step, flattened: step ``k``'s
    row covers positions ``0`` through its query, ``context_count + k``
    values, and is the mean over the ``heads`` chosen, first token included.

    The rest describe every head, ``(layers, heads, ...)``, counting from 0
    like ``heads``, and leave the first token out: it takes most of many
    heads' attention whatever it is (the attention sink), which would
    otherwise decide every ranking. ``key_shares`` is the share of a head's
    attention on everything but the first token, summed over the steps,
    that went to each position; ``recent_shares`` the share that went to
    the last :data:`RECENT_KEYS` keys up to each query. ``active`` is the
    head's mean attention on everything but the first token per step, which
    says how much those shares stand on: a head that rests on the first
    token at every step has shares made of rounding.
    """

    context_count: int
    tokens: list[dict]
    layer_count: int
    head_count: int
    heads: tuple[tuple[int, int], ...]
    rows: np.ndarray
    key_shares: np.ndarray
    recent_shares: np.ndarray
    active: np.ndarray


class _TraceReducer:
    """Fold an attention trace's chunks into the few readings kept."""

    def __init__(self, layer_count: int, head_count: int, first: int, total: int, heads):
        self.first = first
        self.layer_count = layer_count
        self.head_count = head_count
        self.heads = tuple(heads)
        self.by_layer: dict[int, list[int]] = {}
        for layer, head in self.heads:
            self.by_layer.setdefault(layer, []).append(head)
        self.rows: list[np.ndarray] = []
        self.key_sums = np.zeros((layer_count, head_count, total), dtype=np.float64)
        self.recent_sums = np.zeros((layer_count, head_count), dtype=np.float64)
        self.active_sums = np.zeros((layer_count, head_count), dtype=np.float64)
        self.steps = 0

    def add(self, start: int, layers: list[np.ndarray]) -> None:
        """Take in chunk weights fed from position ``start``; see Engine.attention_heads."""

        if len(layers) != self.layer_count or any(
            layer.shape[0] != self.head_count for layer in layers
        ):
            raise RuntimeError("The model's attention changed shape partway through the reply.")
        fed = layers[0].shape[1]
        steps = [step for step in range(fed) if start + step >= self.first]
        if not steps:
            return
        chosen = np.zeros(layers[0].shape[1:], dtype=np.float32)
        for layer, heads in self.by_layer.items():
            chosen += layers[layer][heads].sum(axis=0)
        chosen /= len(self.heads)
        for step in steps:
            query = start + step
            self.rows.append(chosen[step, : query + 1].astype(np.float16))
            self.steps += 1
            if query == 0:
                continue
            low = max(1, query + 1 - RECENT_KEYS)
            for layer, weights in enumerate(layers):
                kept = weights[:, step, 1 : query + 1]
                self.key_sums[layer, :, 1 : query + 1] += kept
                self.recent_sums[layer] += kept[:, low - 1 :].sum(axis=1)
                self.active_sums[layer] += kept.sum(axis=1)

    def finish(self, context_count: int, tokens: list[dict]) -> AttentionTrace:
        steps = max(self.steps, 1)
        active = np.maximum(self.active_sums, 1e-12)
        return AttentionTrace(
            context_count=context_count,
            tokens=tokens,
            layer_count=self.layer_count,
            head_count=self.head_count,
            heads=self.heads,
            rows=np.concatenate(self.rows) if self.rows else np.zeros(0, dtype=np.float16),
            key_shares=(self.key_sums / active[..., None]).astype(np.float32),
            recent_shares=(self.recent_sums / active).astype(np.float32),
            active=(self.active_sums / steps).astype(np.float32),
        )


class InspectionMixin:
    """The scoring and inspection methods of :class:`model_runtime.ModelManager`.

    State lives on the manager; see its ``__init__``.
    """

    def _drop_inspect_cache(self) -> None:
        """Forget the cache the last inspection kept, and give its memory back."""

        if self._inspect_cache is None:
            return
        self._inspect_cache = None
        self._release_device_cache()

    def _inspect_cache_for(self, needed: list[int], media=None, images: Sequence[str] = ()):
        """A key-value cache holding exactly ``needed``, reusing the last one where it can.

        Clicking through the tokens of one response asks about the same
        sequence again and again, and the cache the previous click built
        covers most of the next one. The kept cache is used as it stands when
        the tokens match, extended when the new click is further along, and
        cut back with ``crop()`` when it is earlier. A different sequence, a
        cache from another load, or one that cannot be cropped is thrown away
        and rebuilt from nothing. Called under the model lock.

        ``images`` are the pictures the sequence holds and ``media`` its
        :class:`vision.MediaLayout`. Two sequences can have the same token ids
        and different pictures behind their placeholders, so a kept cache is
        reused only for the same pictures.
        """

        engine = self._engine()
        kept = self._inspect_cache
        self._inspect_cache = None
        if kept is not None:
            load_id, ids, cache = kept[:3]
            shared = min(len(ids), len(needed))
            if (
                load_id != self.load_id
                or cache is None
                or _kept_images(kept) != tuple(images)
                or ids[:shared] != needed[:shared]
                or (len(ids) > len(needed) and not engine.can_crop(cache, len(ids)))
            ):
                kept = None

        if kept is None:
            self._release_device_cache()
            ids, cache = [], None
        else:
            _, ids, cache = kept[:3]
            if len(ids) > len(needed):
                engine.crop(cache, len(ids) - len(needed))
                ids = ids[: len(needed)]

        if len(ids) == len(needed):
            return cache
        # Collect nothing: only the cache is wanted.
        _, cache, _ = self._prefill(
            needed[len(ids) :],
            segments=[""] * (len(needed) - len(ids)),
            positions=list(range(len(ids), len(needed))),
            score_from=len(needed),
            collect_from=len(needed),
            past_key_values=cache,
            cached=len(ids),
            media=media,
        )
        return cache

    def count_score_tokens(
        self,
        text: str,
        *,
        context: str = "",
        use_chat_template: bool = False,
    ) -> tuple[int, int] | None:
        """How many tokens :meth:`score_text` would measure, and its limit.

        ``None`` where the count cannot be had at this instant: nothing is
        loaded, a generation has the floor, the model lock is held, the model
        changed while this was working, or the tokenizer refuses the text.
        This answers a box being typed into, and waiting behind a running
        generation would hang the keystroke rather than the number; a caller
        with no answer says so and asks again on the next one.

        The lock is held only long enough to read the tokenizer, the limit
        and the load they belong to. The encoding itself - the one part whose
        cost grows with what has been pasted - runs outside it, against the
        tokenizer object already in hand. Holding the lock across it was the
        real hazard: a generation claims its slot before it goes for the
        lock, so between the :attr:`occupant` check below and the acquire
        there is a window in which a keystroke could take the lock and then
        keep a reply waiting for as long as tokenizing a large paste took.
        That window still exists and always will, but what it now costs is
        three attribute reads.

        That check asks :attr:`occupant` rather than :attr:`busy` so that a
        claimed load gives up the count as a running reply does. A load takes
        its claim minutes before it takes the lock, and a count produced in
        between describes weights that are on their way out; giving up says
        so, in the words ``score_count_unavailable`` has for a load.

        Encoding outside the lock means a load can land mid-count, so the
        load is read again afterwards and a count from the wrong weights is
        dropped rather than reported. The tokenizer in hand stays valid
        either way - an unload drops the manager's reference, not the object
        - so the worst case is work thrown away, never a torn read.

        A tokenizer that refuses the passage outright is one of those "not
        now" cases rather than a failure to report: pressing **Score text**
        runs the same encoding and explains what went wrong properly, and a
        half-typed passage is not yet worth complaining about.
        """

        if not self.loaded or self.occupant is not None:
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if not self.loaded:
                return None
            tokenizer = self.tokenizer
            limit = score_token_limit(self.model)
            counted_load = self.load_id
        finally:
            self._lock.release()

        if not text:
            return 0, limit
        try:
            split = encode_for_scoring(
                tokenizer,
                text,
                context=context,
                use_chat_template=use_chat_template,
            )
        except Exception:
            return None
        if not split.text_ids:
            # Text that tokenizes to nothing is what score_text refuses, and
            # reporting it as a confident zero would read as room to spare
            # rather than as the refusal it is.
            return None
        if self.load_id != counted_load:
            return None
        return len(split.context_ids) + len(split.text_ids), limit

    @_guards_device_memory
    def score_text(
        self,
        text: str,
        *,
        context: str = "",
        use_chat_template: bool = False,
        load_id: str | None = None,
        steering: dict | None = None,
    ) -> ScoredText:
        """Measure text the model did not write, in one pass over the tokens.

        ``load_id`` names the load the caller checked against, as
        :meth:`generate` takes it: compared here under the model lock, so a
        load that finished while this waited for the lock is refused with
        :class:`ModelChanged` rather than measuring one model's text and
        reporting it as another's.

        ``steering`` adds a vector while the passage is read, as it does
        while a reply is written. Measuring one fixed passage with a vector
        and again without it is the one comparison that stays exact to the
        last token: the tokens are the reader's either way, so every position
        is the same question asked of two models.
        """

        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before scoring text.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    f"The model in memory is {self.model_id}, not the one this "
                    "text was to be measured against. Ask again."
                )

            assert self.tokenizer is not None
            tokenizer = self.tokenizer
            self._drop_inspect_cache()
            # Whitespace is worth measuring: how expected a paragraph break or
            # an indent was is a real question for a token explorer, and the
            # tokenizer turns those characters into ordinary tokens. Only a
            # genuinely empty box is rejected here; text that tokenizes to
            # nothing is caught by the ``text_ids`` check below.
            if not text:
                raise ValueError("Enter some text to score.")

            split = encode_for_scoring(
                tokenizer, text, context=context, use_chat_template=use_chat_template
            )
            context_ids, text_ids = split.context_ids, split.text_ids

            if not text_ids:
                raise ValueError("That text did not produce any tokens.")

            token_ids = context_ids + text_ids
            limit = score_token_limit(self.model)
            if len(token_ids) > limit:
                ceiling = (
                    f"the {limit:,} token limit for scoring"
                    if limit >= SCORE_TOKEN_LIMIT
                    else f"the {limit:,} positions this model can attend to"
                )
                raise ValueError(
                    f"That is {len(token_ids):,} tokens, above {ceiling}. "
                    "Score it in smaller pieces."
                )

            with self._steering(steering):
                metrics, _, _ = self._prefill(
                    token_ids,
                    segments=["prompt"] * len(context_ids) + ["response"] * len(text_ids),
                    positions=list(range(1, len(context_ids) + 1))
                    + list(range(1, len(text_ids) + 1)),
                    score_from=1,
                )
            return ScoredText(
                context_metrics=[
                    metric for metric in metrics if metric["segment"] == "prompt"
                ],
                metrics=[
                    metric for metric in metrics if metric["segment"] == "response"
                ],
                seam_verified=split.seam_verified,
                chat_template_missing=split.chat_template_missing,
                context_ids=tuple(context_ids),
            )

    def _example_ids(self, text: str, use_chat_template: bool) -> list[int]:
        """Token ids for one steering example, as the reader asked for it.

        A chat model answers from the end of a turn, which is where a vector
        meant to steer its replies should be read; ticking the box puts each
        example in a user turn and appends the generation prompt, so the last
        position is the one the model would have written from. A model with
        no chat template has no turn to build, and the example is read as
        plain text with whatever marker the tokenizer opens a sequence with.
        """

        assert self.tokenizer is not None
        if use_chat_template and self.tokenizer.chat_template:
            ids, _prefilled = self._prompt_token_ids([{"role": "user", "content": text}])
            return ids
        return self._encode_plain(text, add_special_tokens=True)

    def _pooled_block_outputs(self, token_ids: Sequence[int], blocks, pool: str) -> np.ndarray:
        """Every decoder block's output for one example, pooled to one vector each.

        Read through forward hooks on the blocks themselves rather than
        through ``output_hidden_states``, for two reasons. The hook sees the
        tensor the steering hook would add to, so a direction taken here and
        a direction added later are defined against the same thing. The
        reported hidden states are not that tensor for the last block: a
        decoder stack appends its final state *after* the final norm, so a
        direction read from there would be in the normed basis and adding it
        back before the norm would not do what it measured.

        And nothing but the pooled vectors is ever materialized: asking for
        the hidden states would hold every layer's full sequence at once,
        which on a 7B model with a few hundred tokens is hundreds of
        megabytes taken from the weights sitting beside it.

        Called under the model lock, inside inference mode.
        """

        import torch

        captured: list[np.ndarray | None] = [None] * len(blocks)

        def record(index: int):
            def capture(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                    raise steering_vectors.SteeringError(
                        "This model's decoder blocks do not return a residual "
                        "tensor a vector could be read from."
                    )
                row = hidden[0, -1] if pool == "last" else hidden[0].mean(dim=0)
                captured[index] = row.detach().float().cpu().clone().numpy()

            return capture

        handles = [block.register_forward_hook(record(index)) for index, block in enumerate(blocks)]
        try:
            self.model(
                input_ids=torch.tensor(
                    [[int(value) for value in token_ids]],
                    dtype=torch.long,
                    device=next(self.model.parameters()).device,
                ),
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        if any(row is None for row in captured):
            raise steering_vectors.SteeringError(
                "Some of this model's decoder blocks did not run, so no "
                "direction could be read from them."
            )
        return np.stack(captured)

    def _reading_blocks(self, load_id, *, missing: str, changed: str, mlx: str):
        """The decoder blocks a forward hook reads, once the load is checked.

        Shared by everything that reads the residual stream out of the model
        rather than generating from it: a steering extraction, a probe's
        examples and a probe's reading of a passage. Each says in its own
        words what was being read when it is refused. Called under the model
        lock, inside inference mode.
        """

        if not self.loaded:
            raise RuntimeError(missing)
        if load_id is not None and load_id != self.load_id:
            raise ModelChanged(
                f"The model in memory is {self.model_id}, not the one {changed}. Ask again."
            )
        if self._engine().backend != "torch":
            raise steering_vectors.SteeringError(mlx)
        # Refuse an architecture the vector could not be added back to,
        # here rather than at the end of a pass over every example. This
        # is the same block list the steering hook installs on.
        blocks = steering_vectors.decoder_layers(self.model)
        self._drop_inspect_cache()
        return blocks

    def _example_readings(self, examples, use_chat_template: bool, pool: str, blocks, cancelled=None) -> list[np.ndarray]:
        """Every example read at every block, pooled, one array per example."""

        window = model_position_limit(self.model)
        limit = STEERING_EXAMPLE_TOKEN_LIMIT
        if window is not None:
            limit = min(limit, window)
        encoded = []
        for index, example in enumerate(examples, start=1):
            if cancelled and cancelled():
                raise ProjectionCancelled("Stopped reading examples.")
            ids = self._example_ids(example, use_chat_template)
            if not ids:
                raise ValueError(f"Example {index} did not produce any tokens.")
            if len(ids) > limit:
                raise ValueError(
                    f"Example {index} is {len(ids):,} tokens, above the "
                    f"{limit:,} one example may be. Shorten it."
                )
            encoded.append(ids)
        rows = []
        for ids in encoded:
            if cancelled and cancelled():
                raise ProjectionCancelled("Stopped reading examples.")
            rows.append(self._pooled_block_outputs(ids, blocks, pool))
        if cancelled and cancelled():
            raise ProjectionCancelled("Stopped reading examples.")
        return rows

    @_guards_device_memory
    def read_examples(
        self,
        examples: Sequence[str],
        *,
        use_chat_template: bool = False,
        pool: str = "last",
        load_id: str | None = None,
        cancelled=None,
    ) -> np.ndarray:
        """Every decoder block's output for each example, pooled as a steering extraction pools it.

        One row per example, one pooled vector per block: the same reading
        :meth:`extract_steering` takes the difference in means of, returned
        whole for a caller fitting something else to it, such as a probe.
        Read through the same hooks, so what is fitted here is defined
        against the tensor a steering vector would be added to.
        """

        if pool not in STEERING_POOLS:
            raise ValueError("Pool the examples by their last token or their mean.")
        examples = [str(value) for value in examples]
        if not examples:
            raise ValueError("Give at least one example.")

        import torch

        with self._lock, torch.inference_mode():
            blocks = self._reading_blocks(
                load_id,
                missing="Load a model before reading examples through it.",
                changed="these examples were to be read through",
                mlx=(
                    "Reading examples needs a PyTorch model: the reading is "
                    "taken through a forward hook, which an MLX checkpoint has "
                    "nowhere to put. Load the model's unquantized Transformers "
                    "version instead."
                ),
            )
            return np.stack(self._example_readings(examples, use_chat_template, pool, blocks, cancelled))

    def _projection_directions(self, directions, blocks):
        directions = np.asarray(directions, dtype=np.float32)
        if directions.ndim != 2 or directions.shape[0] != len(blocks):
            raise ValueError(f"Give one direction for each of this model's {len(blocks)} decoder blocks.")
        config = getattr(self.model, "config", None)
        if callable(getattr(config, "get_text_config", None)):
            config = config.get_text_config()
        width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if width is None:
            width = self.model.get_input_embeddings().weight.shape[-1]
        if directions.shape[1] != width:
            raise ValueError(f"These directions are {directions.shape[1]:,} wide; this model's blocks are {width:,}.")
        if not np.isfinite(directions).all():
            raise ValueError("Probe directions must be finite.")
        return directions

    def check_projection(self, directions, *, load_id=None):
        """Validate a probe against the pinned model without running a forward pass."""
        with self._lock:
            blocks = self._reading_blocks(load_id, missing="Load a model before reading a probe.",
                                          changed="this probe was to be read through",
                                          mlx="Reading a probe needs a PyTorch model; load its Transformers version.")
            self._projection_directions(directions, blocks)

    @_guards_device_memory
    def project_blocks(self, token_ids: Sequence[int], directions, *, load_id: str | None = None, cancelled=None) -> np.ndarray:
        """Every position's block output, read along one direction per block.

        ``directions`` holds one vector per decoder block, indexed as the
        steering hook indexes them. The answer has a row per block and a
        column per position: the dot product of that block's output at that
        position with that block's direction. Only the products leave the
        hook, so a passage of a few thousand tokens costs one forward pass
        and a few numbers per token, never every layer's full sequence.
        """

        import torch

        def check_cancelled(*_args):
            if cancelled and cancelled():
                raise ProjectionCancelled("Stopped reading the probe.")

        check_cancelled()
        ids = [int(value) for value in token_ids]
        if not ids:
            raise ValueError("There are no tokens to read.")
        directions = np.asarray(directions, dtype=np.float32)
        with self._lock, torch.inference_mode():
            blocks = self._reading_blocks(
                load_id,
                missing="Load a model before reading a passage through it.",
                changed="this passage was to be read through",
                mlx=(
                    "Reading a passage along a direction needs a PyTorch model: "
                    "the reading is taken through a forward hook, which an MLX "
                    "checkpoint has nowhere to put. Load the model's unquantized "
                    "Transformers version instead."
                ),
            )
            directions = self._projection_directions(directions, blocks)
            # One pass with no cache holds every layer's attention over the
            # whole passage, so the flat scoring cap applies as well as the window.
            limit = score_token_limit(self.model)
            if len(ids) > limit:
                raise ValueError(
                    f"That is {len(ids):,} tokens, above the {limit:,} one reading may be. Shorten it."
                )
            device = next(self.model.parameters()).device
            captured: list[np.ndarray | None] = [None] * len(blocks)

            def record(index: int):
                def capture(_module, _inputs, output):
                    check_cancelled()
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                        raise steering_vectors.SteeringError(
                            "This model's decoder blocks do not return a residual "
                            "tensor a direction could be read along."
                        )
                    if hidden.shape[-1] != directions.shape[1]:
                        raise ValueError(
                            f"These directions are {directions.shape[1]:,} wide; this "
                            f"model's blocks are {hidden.shape[-1]:,}."
                        )
                    direction = torch.from_numpy(directions[index]).to(device=hidden.device)
                    captured[index] = (hidden[0].float() @ direction).cpu().numpy()

                return capture

            handles = []
            for index, block in enumerate(blocks):
                handles.append(block.register_forward_pre_hook(check_cancelled))
                handles.append(block.register_forward_hook(record(index)))
            try:
                check_cancelled()
                self.model(
                    input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                    use_cache=False,
                )
            finally:
                for handle in handles:
                    handle.remove()
            check_cancelled()
            if any(row is None for row in captured):
                raise steering_vectors.SteeringError(
                    "Some of this model's decoder blocks did not run, so the "
                    "passage could not be read at them."
                )
            return np.stack(captured).astype(np.float64)

    @_guards_device_memory
    def extract_steering(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        *,
        use_chat_template: bool = False,
        pool: str = "last",
        load_id: str | None = None,
    ) -> SteeringExtraction:
        """Read a steering direction out of two sets of examples.

        Each example is run through the model once and every decoder block's
        output is pooled to a single vector; the direction for a block is the
        positive examples' mean minus the negative examples' mean. That is
        the difference in means, and it is what the vector import format has
        held all along - this only saves making it somewhere else.

        Every layer is returned, because the pass that reads one reads them
        all, and which layer to steer at is the question the reader has least
        way of answering in advance. :func:`steering.contrast_directions`
        says what the numbers beside each layer mean.

        ``load_id`` names the load the caller checked against, as
        :meth:`generate` and :meth:`score_text` take it: compared under the
        model lock, so weights that changed while this waited are refused
        rather than quietly measured.
        """

        if pool not in STEERING_POOLS:
            raise ValueError("Pool the examples by their last token or their mean.")
        positive = [str(value) for value in positive]
        negative = [str(value) for value in negative]
        if not positive or not negative:
            raise ValueError("Give at least one example on each side.")
        for side, examples in (("wanted", positive), ("unwanted", negative)):
            if len(examples) > steering_vectors.MAX_EXAMPLES:
                raise ValueError(
                    f"That is {len(examples):,} {side} examples, above the "
                    f"{steering_vectors.MAX_EXAMPLES} one side can hold."
                )

        import torch

        with self._lock, torch.inference_mode():
            blocks = self._reading_blocks(
                load_id,
                missing="Download and load a model before extracting a vector.",
                changed="these examples were to be read through",
                mlx=(
                    "Extracting a vector needs a PyTorch model: the reading is "
                    "taken through a forward hook, which an MLX checkpoint has "
                    "nowhere to put. Load the model's unquantized Transformers "
                    "version to extract from it."
                ),
            )
            chat_template_missing = use_chat_template and not self.tokenizer.chat_template
            readings = [
                self._example_readings(examples, use_chat_template, pool, blocks)
                for examples in (positive, negative)
            ]

            layers, stats = steering_vectors.contrast_directions(*readings)
            return SteeringExtraction(
                model_id=self.model_id,
                load_id=self.load_id,
                layers=tuple(tuple(row) for row in layers),
                stats=tuple(stats),
                positive_count=len(positive),
                negative_count=len(negative),
                pool=pool,
                chat_template_missing=chat_template_missing,
            )

    def _final_norm(self):
        """The norm the LM head reads through; see :meth:`TorchEngine.final_norm`."""

        return self._engine().final_norm()

    def model_revision(self) -> str | None:
        """The loaded checkpoint's revision, or ``None`` when it is not recorded.

        Either backend loads from a cache snapshot whose folder is named for
        its commit: a Transformers model carries the one it was read at (see
        :func:`model_loading.checkpoint_revision`), and an MLX conversion's
        is its snapshot's. Same ID, different revision means different
        weights, so this is what a lens written down for the model is
        compared against.
        """
        with self._lock:
            if not self.loaded:
                return None
            if self._engine().backend == "torch":
                return model_loading.checkpoint_revision(self.model)
            path = self.local_path
            return model_loading.snapshot_revision(path) if path is not None else None

    def import_jacobian_lens(self, path: str, fitted_model_id: str) -> dict:
        """Validate a reference lens file; caller owns the generation reservation.

        A Transformers model must hold full-precision weights, since the lens
        was fitted on them. An MLX conversion only exists at its packed width;
        it is read as it is, and the final-block replay in each inspection is
        what decides whether the readout is close enough to trust.
        """
        revision = self.model_revision()
        with self._lock:
            if not self.loaded:
                raise ValueError("Load the model this lens was fitted for first.")
            engine = self._engine()
            if engine.backend == "torch" and self.precision not in (None, "full"):
                raise ValueError("Load full-precision weights for Jacobian inspection.")
            lens = jacobian_lens.FittedLens.load(
                path, engine, self.model_id, fitted_model_id, revision
            )
            import_id = os.urandom(16).hex()
            self._jacobian_lens = (self.load_id, import_id, lens)
            return {
                "import_id": import_id, "load_id": self.load_id,
                "model_id": self.model_id, "name": lens.name,
                "layers": len(lens.matrices), "n_prompts": lens.n_prompts,
                "path": str(path), "fitted_model_id": fitted_model_id.strip(),
                "backend": engine.backend, "model_revision": revision,
            }

    def jacobian_lens_import(self) -> dict | None:
        """The lens imported for the current load, or ``None`` when there is none."""
        with self._lock:
            imported = self._jacobian_lens
            if imported is None or not self.loaded or imported[0] != self.load_id:
                return None
            return {"import_id": imported[1], "load_id": self.load_id, "name": imported[2].name}

    def remember_jacobian_lens(self, import_id: str, record: dict) -> str:
        """Write ``record`` down for this model if ``import_id`` is still the lens in memory.

        The check and the write happen under the model lock, so two clients
        importing at once cannot leave the record naming the lens that lost.
        Answers ``"remembered"``, ``"replaced"`` when another import has
        taken the lens's place, or ``"unwritable"`` when the store refused.
        """
        with self._lock:
            imported = self._jacobian_lens
            if imported is None or not self.loaded or imported[:2] != (self.load_id, import_id):
                return "replaced"
            return "remembered" if jacobian_lens.remember(self.model_id, record) else "unwritable"

    @_guards_device_memory
    def inspect_jacobian(
        self, token_ids: Sequence[int], index: int, *, lens_id: str | None,
        pinned_text: str = "", pinned_id: int | None = None, context_count: int = 0,
        load_id: str | None = None, steering: dict | None = None,
        positions: int = jacobian_lens.SLICE_POSITIONS, images: Sequence[str] = (),
    ) -> jacobian_lens.JacobianInsight:
        """Read concepts after processing the clicked token, with no look-ahead.

        The clicked token and up to ``positions - 1`` tokens before it are fed
        in one step, so the same pass also reads every one of those positions
        through the lens: that is the layer × position slice. Nothing after
        the clicked token is fed, so no position sees a later one.

        ``lens_id`` names the import the caller saw; ``None`` accepts whatever
        lens is imported for the current load, which is how a lens recalled
        from disk after a reload is used without the caller having seen it.

        ``pinned_id`` pins a vocabulary token outright, as a clicked cell does;
        ``pinned_text`` is encoded only when no ID is given, since a token's
        text need not encode back to that token. The lens checks the ID against
        the output vocabulary.
        """
        import torch

        with self._lock, torch.inference_mode(), contextlib.ExitStack() as scope:
            if not self.loaded or load_id != self.load_id:
                raise ModelChanged("The model has been reloaded. Generate or score again.")
            imported = self._jacobian_lens
            if (
                imported is None or imported[0] != self.load_id
                or (lens_id is not None and imported[1] != lens_id)
            ):
                raise ValueError("Import a Jacobian lens for the current model load first.")
            lens = imported[2]
            engine = self._engine()
            jacobian_lens.model_layout(engine)
            ids = [int(token) for token in token_ids]
            if not 0 <= index < len(ids):
                raise ValueError("Select a token in the current transcript.")
            if pinned_id is not None:
                pinned_id = int(pinned_id)
            elif pinned_text:
                encoded = self.tokenizer.encode(pinned_text, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError("Pin one vocabulary token. Try a single word, including its leading space if needed.")
                pinned_id = int(encoded[0])
            scope.enter_context(self._steering(steering))
            if steering_vectors.active(steering):
                self._drop_inspect_cache()
            media = self._image_layout(ids, images)
            # A picture read in both directions has to go in whole, and the
            # slice ends at the clicked token, so a click inside one would
            # read a picture missing its later tokens.
            if media is not None and media.joined and media.is_placeholder(index):
                raise ValueError(IMAGE_TOKEN_INSPECTED)
            start = max(0, index + 1 - max(1, int(positions)))
            if media is not None:
                start = media.run_start(start)
            cache = self._inspect_cache_for(ids[:start], media, images)
            with lens.capture(engine) as states:
                logits, cache = engine.forward(ids[start:index + 1], cache, start, media)

            decoded: dict[int, str] = {}

            def decode(token):
                token = int(token)
                if token not in decoded:
                    decoded[token] = self._decode_token(token) or self._token_fallback(token)
                return decoded[token]

            actual = logits.row(-1)
            rows, cells = lens.read(engine, states, actual, decode, pinned_id)
            best_ids, best_scores = logits.best()
            output = [
                {"token_id": int(token), "text": decode(token), "score": float(score)}
                for token, score in zip(best_ids, best_scores)
            ]
            if pinned_id is not None:
                # The model's own row is shaded by the pinned token too.
                for cell, rank, score in zip(output, *logits.pinned(pinned_id)):
                    cell["pinned_rank"] = int(rank)
                    cell["pinned_score"] = float(score)
            tokens = [
                {
                    "index": position, "token_id": ids[position], "text": decode(ids[position]),
                    "segment": "prompt" if position < context_count else "response",
                }
                for position in range(start, index + 1)
            ]
            del logits
            if cache is not None and not steering_vectors.active(steering):
                self._inspect_cache = (self.load_id, ids[:index + 1], cache, tuple(images))
            return jacobian_lens.JacobianInsight({
                "kind": "jacobian", "index": index, "token_id": ids[index],
                "token_text": decode(ids[index]), "layers": rows,
                "attention": [], "tokens": [], "decided_at": None,
                "pinned_text": decode(pinned_id) if pinned_id is not None else None,
                "pinned_id": pinned_id, "vocab_size": len(actual),
                "lens_name": lens.name, "n_prompts": lens.n_prompts,
                "model_id": self.model_id, "import_id": imported[1],
                "backend": engine.backend, "precision": self.precision,
                "slice": {
                    "start": start, "total": len(ids), "tokens": tokens,
                    "layers": cells, "output": output,
                },
            })

    def patch_activations(self, donor, recipient, target_index, donor_count, width, contrast_index=None):
        """Measure independent residual transplants between exact Compare runs.

        The caller holds the generation reservation and closes this iterator
        on cancellation. The model lock pins the weights and tokenizer for
        the entire experiment; no torch context or intervention hook spans a
        yield to the UI.
        """
        from chatlab import activation_patching

        with self._lock:
            try:
                plan = activation_patching.experiment(donor, recipient, target_index, donor_count, width,
                                                      contrast_index)
                if not self.loaded or self.load_id != plan["load_id"] or self.model_id != plan["model_id"]:
                    raise ModelChanged("The model has been reloaded. Fill both Compare slots again.")
                if self._engine().backend != "torch":
                    raise ValueError("Activation patching requires a Transformers model; MLX is not supported yet.")
                activation_patching.model_layers(self.model)

                def label(token):
                    return self._decode_token(token) or self._token_fallback(token)

                plan["target_text"] = label(plan["target_id"])
                if plan["contrast_id"] is not None:
                    plan["contrast_text"] = label(plan["contrast_id"])
                for pair in plan["pairs"]:
                    pair["donor_text"] = label(plan["donor_ids"][pair["donor_position"]])
                    pair["recipient_text"] = label(plan["recipient_ids"][pair["recipient_position"]])
                self._drop_inspect_cache()
                logger.info("Activation patching: %s, target %s, %d token pairs", self.model_id,
                            plan["target_id"], len(plan["pairs"]))
                with contextlib.closing(activation_patching.measure(self.model, plan)) as readings:
                    for reading in readings:
                        yield plan, reading
            except (RuntimeError, MemoryError) as error:
                reraise_out_of_memory(error)
            finally:
                # The inner iterator closes first, removing hooks and dropping
                # its tensors before allocator blocks are returned. This also
                # runs on GeneratorExit when Stop closes the outer iterator.
                self._release_device_cache()

    def _lens_row(self, layer: int, logits: np.ndarray, token_id: int) -> dict:
        log_probs = normalize_log_probabilities(np.asarray(logits, dtype=np.float32))
        token_log_prob = float(log_probs[token_id])
        top_id = int(np.argmax(log_probs))
        return {
            "layer": layer,
            "probability": float(np.exp(token_log_prob)),
            "rank": int(np.count_nonzero(log_probs > token_log_prob)) + 1,
            "entropy_bits": entropy_bits(log_probs),
            "top_id": top_id,
            "top_text": self._decode_token(top_id) or self._token_fallback(top_id),
            "top_probability": float(np.exp(log_probs[top_id])),
        }

    @_guards_device_memory
    def inspect(
        self,
        token_ids: Sequence[int],
        index: int,
        *,
        context_count: int = 0,
        load_id: str | None = None,
        steering: dict | None = None,
        images: Sequence[str] = (),
    ) -> TokenInsight:
        """Explain the prediction of ``token_ids[index]`` layer by layer.

        The sequence up to the token before ``index`` is run through the model
        again, then that token is fed in alone with the hidden states and
        attention weights switched on. Its output is the distribution that
        predicted the inspected token, so the final row of the logit lens
        matches the probabilities the strip already shows, and its attention
        row says which earlier tokens went into that prediction.

        ``context_count`` is how many leading tokens are prompt or context
        rather than response, purely for labelling.

        ``load_id`` names the load the tokens came from (see :attr:`load_id`).
        It is compared under the model lock, so a load that started after the
        caller looked and finished before this ran is still refused, with
        :class:`ModelChanged`, rather than explaining the tokens with weights
        and a tokenizer they never met.
        """

        import torch

        with self._lock, torch.inference_mode(), contextlib.ExitStack() as steering_scope:
            if not self.loaded:
                raise RuntimeError("Download and load a model before inspecting a token.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            steering_scope.enter_context(self._steering(steering))
            if steering_vectors.active(steering):
                # A cache computed without this vector cannot explain it.
                # Steered inspections do not retain a cache for later clicks.
                self._drop_inspect_cache()
            ids = [int(value) for value in token_ids]
            if not 1 <= index < len(ids):
                raise ValueError(
                    "Nothing came before this token, so the model never predicted it."
                )

            assert self.model is not None
            engine = self._engine()
            token_id = ids[index]
            media = self._image_layout(ids, images)
            if media is not None and media.is_placeholder(index):
                raise ValueError(IMAGE_TOKEN_INSPECTED)

            # Everything before the predicting token, from the last click's
            # cache where the sequence allows it. The backend reads every
            # layer's prediction, and withholds the intermediate ones when
            # they cannot be trusted; see TorchEngine.inspect_step and
            # MlxEngine.inspect_step.
            if media is None:
                past_key_values = self._inspect_cache_for(ids[: index - 1])
                reading = engine.inspect_step(ids[index - 1], past_key_values, index - 1)
            else:
                # A picture read in both directions goes in whole, with the
                # token it ends on; see MediaLayout.run_start.
                start = media.run_start(index - 1)
                past_key_values = self._inspect_cache_for(ids[:start], media, images)
                reading = engine.inspect_step(
                    ids[index - 1], past_key_values, start, media, before=ids[start : index - 1]
                )

            layers: list[dict] = [
                self._lens_row(layer, logits, token_id)
                for layer, logits in enumerate(reading.layer_logits)
            ]
            layers.append(
                self._lens_row(
                    max(reading.layer_count - 1, 0), reading.final_logits, token_id
                )
            )

            decided_at: int | None = None
            for row in reversed(layers):
                if row["rank"] != 1:
                    break
                decided_at = row["layer"]

            attention: list[list[float]] = []
            for row in reading.attention:
                # A sliding-window layer keeps only its most recent keys,
                # so a short row describes the end of the sequence. Align
                # it on the right; the keys the layer could not see get a
                # weight of zero, which is what it gave them.
                row = [float(value) for value in row][-index:]
                attention.append([0.0] * (index - len(row)) + row)

            tokens = [
                {
                    "index": position,
                    "token_id": ids[position],
                    "text": self._decode_token(ids[position]),
                    "fallback": self._token_fallback(ids[position]),
                    "segment": "prompt" if position < context_count else "response",
                }
                for position in range(index)
            ]
            # The step above appended the predicting token, so the cache now
            # covers the sequence through it. Kept for the next click; a
            # response or a scoring pass takes it back (see _drop_inspect_cache).
            if (
                reading.cache is not None
                and self.load_id is not None
                and not steering_vectors.active(steering)
            ):
                self._inspect_cache = (self.load_id, ids[:index], reading.cache, tuple(images))
            del reading, past_key_values
            return TokenInsight(
                index=index,
                token_id=token_id,
                token_text=self._decode_token(token_id) or self._token_fallback(token_id),
                layers=layers,
                tokens=tokens,
                attention=attention,
                decided_at=decided_at,
            )

    @_guards_device_memory
    def trace_attention(
        self,
        token_ids: Sequence[int],
        context_count: int,
        *,
        heads=None,
        load_id: str | None = None,
        steering: dict | None = None,
        images: Sequence[str] = (),
    ) -> AttentionTrace:
        """Read every head's attention behind every reply token in one pass.

        ``token_ids`` is the prompt and the reply; the first
        ``context_count`` are the prompt. Everything before the last prompt
        token comes from the inspection cache where it can, then the rest is
        fed a chunk at a time with the weights switched on, which costs about
        what replaying the reply would, and the cache is kept for the next
        inspection. See :class:`AttentionTrace` for what is kept.

        ``heads`` chooses which heads the rows average: ``None`` for all of
        them, or a function given the layer and head counts that returns
        ``(layer, head)`` pairs counting from 0, and raises ``ValueError``
        for a choice the model does not have. ``load_id``, ``steering`` and
        ``images`` are as for :meth:`inspect`.
        """

        import torch

        with self._lock, torch.inference_mode(), contextlib.ExitStack() as steering_scope:
            if not self.loaded:
                raise RuntimeError("Download and load a model before tracing attention.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            steering_scope.enter_context(self._steering(steering))
            steered = steering_vectors.active(steering)
            if steered:
                self._drop_inspect_cache()
            ids = [int(value) for value in token_ids]
            if not 1 <= context_count < len(ids):
                raise ValueError("There is no reply after the prompt to trace.")

            assert self.model is not None
            engine = self._engine()
            media = self._image_layout(ids, images)
            first = context_count - 1
            last = len(ids) - 1
            position = media.run_start(first) if media is not None else first
            cache = self._inspect_cache_for(ids[:position], media, images)
            reducer: _TraceReducer | None = None
            chunk = 1
            while position < last:
                end = min(position + chunk, last)
                if media is not None:
                    end = min(max(media.chunk_end(end), position + 1), last)
                layers, cache = engine.attention_heads(ids[position:end], cache, position, media)
                if layers is None:
                    raise ValueError(
                        "This model does not report its attention weights, so there is nothing to trace."
                    )
                if reducer is None:
                    layer_count, head_count = len(layers), layers[0].shape[0]
                    chosen = (
                        [(layer, head) for layer in range(layer_count) for head in range(head_count)]
                        if heads is None
                        else list(heads(layer_count, head_count))
                    )
                    if not chosen:
                        raise ValueError("Choose at least one head to trace.")
                    reducer = _TraceReducer(layer_count, head_count, first, last, chosen)
                    width = layer_count * head_count * len(ids) * 4
                    chunk = max(1, min(ATTENTION_TRACE_CHUNK_TOKENS, ATTENTION_TRACE_CHUNK_BYTES // width))
                reducer.add(position, layers)
                del layers
                position = end

            tokens = [
                {
                    "index": index,
                    "token_id": ids[index],
                    "text": self._decode_token(ids[index]),
                    "fallback": self._token_fallback(ids[index]),
                    "segment": "prompt" if index < context_count else "response",
                }
                for index in range(len(ids))
            ]
            if cache is not None and self.load_id is not None and not steered:
                self._inspect_cache = (self.load_id, ids[:last], cache, tuple(images))
            del cache
            assert reducer is not None
            return reducer.finish(context_count, tokens)

    def read_kv_cache(
        self, token_ids: Sequence[int], layer: int, *, load_id: str | None = None,
        images: Sequence[str] = (),
    ) -> dict:
        """One layer of the key-value cache the last inspection kept.

        ``token_ids`` is the sequence the caller's readout covers, through
        the query token, and ``layer`` counts from 1. The cache is read only
        while it holds exactly those tokens from the load named, so the
        numbers describe the readout they are shown beside and never a later
        click's. Nothing is run through the model.

        The lock is waited on briefly rather than for as long as it is held:
        a reply keeps it for the whole of its stream, and gives the cache
        back when it starts, so waiting would end in :class:`CacheGone`
        anyway. :class:`kv_cache.CacheBusy` says to try again instead.
        """

        if not self._lock.acquire(timeout=KV_CACHE_WAIT):
            raise kv_cache.CacheBusy("The model is busy. Wait for it to finish, then try again.")
        try:
            if not self.loaded:
                raise kv_cache.CacheGone("Download and load a model before reading its cache.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            ids = [int(value) for value in token_ids]
            kept = self._inspect_cache
            if (
                kept is None or kept[0] != self.load_id or kept[1] != ids
                or _kept_images(kept) != tuple(images)
            ):
                raise kv_cache.CacheGone(
                    "The cache from this inspection is no longer in memory. A reply, a "
                    "scoring pass and another inspection each release it; press "
                    "Inspect layers again."
                )
            engine = self._engine()
            cache = kept[2]
            shapes = engine.cache_shapes(cache)
            chosen = min(max(int(layer or 1), 1), max(len(shapes), 1))
            reading = engine.cache_layer(cache, chosen - 1, len(ids)) if shapes else None
            return {
                "summary": kv_cache.summarize(shapes, len(ids)),
                "layer": chosen,
                "backend": engine.backend,
                "reading": kv_cache.read_layer(reading) if reading is not None else None,
            }
        finally:
            self._lock.release()
