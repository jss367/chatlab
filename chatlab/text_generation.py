"""Generating a response token by token, with every token measured.

The prompt is encoded through the model's chat template, prefilled in
chunks, and extended one sampled token at a time. Each token's rank,
probability, entropy and alternatives are measured as it is drawn, and the
response is streamed as :class:`GenerationUpdate` batches for the page to
redraw.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chatlab import device_memory
from chatlab.conversation import THINK_CLOSE, THINK_OPEN
from chatlab.device_memory import memory_note, reraise_out_of_memory
from chatlab.engine import Engine
from chatlab.thinking import THINKING_MODES, supports_thinking
from chatlab.token_metrics import (
    UNSCORED_BEYOND_LIMIT,
    UNSCORED_FIRST_TOKEN,
    TokenMetric,
    build_metric,
    normalize_log_probabilities,
    sampling_probabilities,
    unscored_metric,
)
from chatlab.tokenization import (
    IncrementalDecoder,
    application_prefill_limit,
    generation_prefill_token_limit,
    model_position_limit,
    split_context_and_text,
)

logger = logging.getLogger(__name__)


# Prefill runs in chunks so a long prompt never materializes a
# sequence-length by vocabulary logit tensor all at once.
PREFILL_CHUNK_SIZE = 128

# Scoring every prompt token costs one softmax over the vocabulary each, so
# only the most recent stretch of a very long prompt is measured.
PROMPT_SCORE_LIMIT = 1024


# Streaming updates are batched so a long response does not re-serialize the
# whole token strip on every single token.
STREAM_BATCH_TOKENS = 8
STREAM_INTERVAL_SECONDS = 0.05


def _inference_stream(method):
    """Apply inference mode on each resume, keeping torch's import lazy.

    A context held across a yield belongs to the original worker thread.
    Gradio can resume on another worker, enabling autograd there and retaining
    training intermediates through the growing key-value cache. PyTorch's
    generator decorator enters and exits on every next/send/throw/close.
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        import torch

        yield from torch.inference_mode()(method)(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class GenerationUpdate:
    text: str
    metrics: list[dict]
    """Live list owned by the generator. Copy it before storing it anywhere."""

    load_id: str
    """The immutable model load that produced these token IDs."""

    prompt_metrics: list[dict] = field(default_factory=list)
    prompt_note: str = ""
    reasoning_prefilled: bool = False
    """Whether the prompt already ended with the opening ``<think>`` marker.

    When it did, ``text`` starts inside the reasoning block and never contains
    an opening marker of its own, so a caller splitting reasoning from the
    answer has to be told.
    """

    forced_prefix_tokens: int = 0
    """How many leading response tokens were replayed instead of sampled."""

    literal_prefill_tokens: int = 0
    """How many leading tokens were decoded as the reader's own literal text.

    Text the reader supplied is shown as they wrote it, marker for marker, so
    those tokens are decoded with the hidden special tokens made visible
    again - a tokenizer's textual end marker typed into a prefill is prose,
    not the model stopping. Anything rebuilding this response's text from its
    token IDs has to make the same exception over the same prefix, or it
    produces a different string from the one on screen.
    """

    literal_prefill_text: str = ""
    """Decoded prefix whose reader-supplied portion must remain literal."""

    literal_text_spans: tuple[tuple[int, int], ...] = ()
    """Character spans in ``text`` that the reader supplied literally.

    Reasoning markers inside these spans are prose, not model control syntax.
    The spans can be disjoint because a typed token-branch replacement may
    follow sampled tokens, and they survive if that response is branched again.
    """

    prompt_ids: tuple[int, ...] = ()
    """Every prompt token, measured or not.

    :meth:`ModelManager.inspect` needs the whole sequence the response was
    generated from, and ``prompt_metrics`` only holds the tokens the reader
    chose to measure.
    """

    ends_on_stop_token: bool = False
    """Whether the last token is one the model ends a response on.

    The count of tokens cannot answer this on its own: a stop token sampled
    as the very last token the ceiling allows ends the response naturally,
    and a caller told that was the length limit would treat a finished answer
    as truncated.
    """

    ends_on_position_limit: bool = False
    """Whether the response stopped because the model has no more positions.

    A learned position table has nothing past its last row, so the token
    sampled there is kept but never fed, and the response ends short of the
    length it was asked for.
    """

    model_id: str | None = None
    """Which weights produced this update, read under the model lock.

    A caller that looks at the manager instead can be wrong: a load may land
    between the caller's look and the moment the generator takes the lock,
    and a caller that saw a model change would have to do without a stamp.
    ``load_id`` is what :meth:`ModelManager.inspect` checks against; see
    :attr:`ModelManager.load_id`.
    """

    thinking_mode: str | None = None
    """Requested template mode, or None when the loaded model cannot switch."""


class ModelChanged(RuntimeError):
    """The weights in memory are not the ones the caller's tokens came from."""


@dataclass
class _ForcedPrefix:
    """The tokens a response is made to start with, and which are the reader's.

    What :meth:`GenerationMixin._forced_prefix` makes of a branch's replayed
    tokens or a typed assistant prefill. ``literal_prefill_tokens`` leading
    tokens are text the reader supplied, the first
    ``automatic_reasoning_close_tokens`` of those are the reasoning close the
    template put in front of it, and ``literal_ranges`` are the token spans of
    typed branch replacements, sorted and merged. Mutable because the close
    can only be measured once the prefix is decoded; see
    :func:`_decode_forced_prefix`.
    """

    ids: list[int]
    literal_prefill_tokens: int
    automatic_reasoning_close_tokens: int
    literal_ranges: list[tuple[int, int]]


def _sampler(
    temperature: float, top_p: float, top_k: int, skip_top_below: float
) -> Callable[[np.ndarray], np.ndarray]:
    """The reader's sampling settings, as one call from log probabilities to a distribution.

    The settings are converted on each call rather than once here, so a value
    that cannot be converted fails where the distribution is first needed.
    """

    def sample(log_probs: np.ndarray) -> np.ndarray:
        return sampling_probabilities(
            log_probs,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            skip_top_below=float(skip_top_below),
        )

    return sample


def _merge_literal_ranges(
    ranges: Sequence[tuple[int, int]], length: int
) -> list[tuple[int, int]]:
    """``ranges`` clamped to a prefix of ``length`` tokens, sorted, overlaps merged.

    Empty ranges are dropped. Ranges that touch are joined as well as ones
    that overlap, so a replacement typed straight after another reads as one
    literal span.
    """

    normalized: list[tuple[int, int]] = []
    for raw_start, raw_end in ranges:
        start = max(0, min(int(raw_start), length))
        end = max(start, min(int(raw_end), length))
        if start < end:
            normalized.append((start, end))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _decode_forced_prefix(
    decoder: IncrementalDecoder,
    prefix: _ForcedPrefix,
    metrics: list[dict],
    *,
    closes_reasoning: bool,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Push the forced prefix through ``decoder``; its literal text and spans.

    Returns the stable text of the reader's prefill, which the application
    protects from the reasoning parser, and the character spans of the typed
    replacements in the decoded text. ``closes_reasoning`` says the prefill
    was put after an automatic reasoning close (an assistant prefill on a
    prompt that ends inside reasoning); the tokens that close turns out to
    span are marked in ``metrics`` and counted on ``prefix`` here, since only
    the decoded text shows where it ends.
    """

    forced = prefix.ids
    literal_prefill_tokens = prefix.literal_prefill_tokens
    literal_prefill_text = ""
    literal_boundaries = {
        boundary for span in prefix.literal_ranges for boundary in span
    }
    boundary_text = {0: ""}
    for index, token_id in enumerate(forced):
        decoder.push(
            token_id, force_visible=index < literal_prefill_tokens
        )
        if (
            closes_reasoning
            and not prefix.automatic_reasoning_close_tokens
            and decoder.stable_text.startswith(f"{THINK_CLOSE}\n\n")
        ):
            # The last token can straddle the boundary and include the
            # beginning of the reader's prefill. It still cannot be
            # replaced independently: doing so would remove part of
            # the close and leave the continuation inside reasoning.
            prefix.automatic_reasoning_close_tokens = index + 1
            for metric in metrics[:prefix.automatic_reasoning_close_tokens]:
                metric["automatic_reasoning_close"] = True
        if index + 1 in literal_boundaries:
            boundary_text[index + 1] = decoder.text
        if index + 1 == literal_prefill_tokens:
            # A branch can stop inside a byte-level token sequence for
            # one character. The replacement-character suffix will be
            # rewritten when the next token arrives, so it cannot be a
            # durable prefix for the application's literal-tag guard.
            literal_prefill_text = decoder.stable_text
    forced_text = decoder.text

    # A byte-level token boundary can land inside one Unicode
    # character. Its temporary U+FFFD is rewritten when later bytes
    # arrive, so use the longest prefix that is actually stable in the
    # completed forced text. Ordinary word-piece boundaries take the
    # fast path and keep their full decoded length.
    def stable_length(at: int) -> int:
        value = boundary_text.get(at, "")
        if forced_text.startswith(value):
            return len(value)
        for offset, (left, right) in enumerate(zip(value, forced_text)):
            if left != right:
                return offset
        return min(len(value), len(forced_text))

    literal_text_spans = tuple(
        (stable_length(start), stable_length(end))
        for start, end in prefix.literal_ranges
        if stable_length(start) < stable_length(end)
    )
    return literal_prefill_text, literal_text_spans


@dataclass
class _Response:
    """One response while it is being produced, and the frames that publish it.

    Everything but the decoder's text and the metrics is fixed once the
    prefix is decoded, so it is gathered here once and every
    :class:`GenerationUpdate` - the forced prefix's and each sampled batch's -
    is built by :meth:`update` from the same fields. ``metrics`` is the live
    list the generator appends to; each update carries that list itself,
    not a copy.
    """

    decoder: IncrementalDecoder
    metrics: list[dict]
    load_id: str
    model_id: str | None
    prompt_ids: tuple[int, ...]
    prompt_metrics: list[dict]
    prompt_note: str
    reasoning_prefilled: bool
    thinking_mode: str | None
    forced_prefix_tokens: int
    literal_prefill_tokens: int
    literal_prefill_text: str
    literal_text_spans: tuple[tuple[int, int], ...]

    def update(
        self, *, ends_on_stop_token: bool, ends_on_position_limit: bool = False
    ) -> GenerationUpdate:
        """The frame for the response as it stands now."""

        return GenerationUpdate(
            text=self.decoder.text,
            metrics=self.metrics,
            load_id=self.load_id,
            prompt_metrics=self.prompt_metrics,
            prompt_note=self.prompt_note,
            reasoning_prefilled=self.reasoning_prefilled,
            thinking_mode=self.thinking_mode,
            forced_prefix_tokens=self.forced_prefix_tokens,
            literal_prefill_tokens=self.literal_prefill_tokens,
            literal_prefill_text=self.literal_prefill_text,
            literal_text_spans=self.literal_text_spans,
            prompt_ids=self.prompt_ids,
            model_id=self.model_id,
            ends_on_stop_token=ends_on_stop_token,
            ends_on_position_limit=ends_on_position_limit,
        )


class GenerationMixin:
    """The text generation methods of :class:`model_runtime.ModelManager`.

    Prompt encoding, token description and the generation loop. State lives
    on the manager; see its ``__init__``.
    """

    @property
    def supports_thinking(self) -> bool:
        return self.loaded and supports_thinking(self.model, self.tokenizer)

    def _prompt_token_ids(
        self, messages: list[dict], tools: list[dict] | None = None,
        *, thinking_mode: str = "default",
    ) -> tuple[list[int], bool]:
        """Token ids for a chat prompt, and whether it prefills ``<think>``.

        Reasoning templates such as OLMo Think end the generation prompt with
        the opening marker, so the model resumes inside the block and never
        emits an opener. The flag rides along to the caller because only the
        prompt can reveal it.
        """

        assert self.tokenizer is not None
        tokenizer = self.tokenizer
        prefilled = False

        if tools is not None and not tokenizer.chat_template:
            raise ValueError("Tool use requires a model with a native chat/tool template.")
        if tokenizer.chat_template:
            tool_args = {"tools": tools} if tools is not None else {}
            if thinking_mode not in THINKING_MODES:
                raise ValueError("Thinking mode must be default, on, or off.")
            if self.supports_thinking and thinking_mode != "default":
                tool_args["enable_thinking"] = thinking_mode == "on"
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **tool_args
            )
            prefilled = isinstance(rendered, str) and rendered.rstrip().endswith(
                THINK_OPEN
            )
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **tool_args
            )
        else:
            transcript = "\n".join(
                f"{message['role'].title()}: {message['content']}"
                for message in messages
            )
            encoded = tokenizer(f"{transcript}\nAssistant:").input_ids

        # Transformers 5 returns a BatchEncoding here by default; iterating
        # that yields its keys, and int("input_ids") is the failure a user
        # sees as "Generation failed".
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(value) for value in encoded], prefilled

    def encode_replacement(
        self,
        kept_ids: Sequence[int],
        text: str,
        *,
        literal_prefill_tokens: int = 0,
        load_id: str | None = None,
    ) -> list[int]:
        """Encode text the reader wants replayed after ``kept_ids``, as a branch does.

        The text stands in for one or more sampled tokens, so it carries no
        special tokens and no reasoning marker. It is checked in place rather
        than on its own because decoding is not always piecewise: SentencePiece
        drops the word-boundary space from the first token of whatever it
        decodes, so ``"world"`` round-trips alone yet reads ``" world"`` once it
        follows ``"Hello"``, while a typed ``" world"`` gains a second space.
        The visible kept tokens plus the result must therefore decode to the
        kept text followed by exactly what was typed. Hidden specials do not
        supply decoder context, because the streaming decoder never caches
        them; ``literal_prefill_tokens`` keeps reader-supplied assistant
        prefill visible as before. The kept ids themselves are never
        re-tokenized; a branch preserves them token for token.

        ``load_id`` names the load the kept tokens came from (see
        :attr:`load_id`). It is compared under the model lock, and the encoding
        itself runs under that same lock, so a load that lands between the
        caller's own check and this call is refused with :class:`ModelChanged`
        rather than answered with tokens from a tokenizer the kept ids never
        met.
        """

        with self._lock:
            if not self.loaded:
                raise RuntimeError("Download and load a model before branching.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            return self._encode_replacement(
                kept_ids, text, literal_prefill_tokens=literal_prefill_tokens
            )

    def _encode_replacement(
        self,
        kept_ids: Sequence[int],
        text: str,
        *,
        literal_prefill_tokens: int = 0,
    ) -> list[int]:
        """The body of encode_replacement(), run with the model lock held."""

        assert self.tokenizer is not None
        kept = [int(value) for value in kept_ids]
        if not text:
            raise ValueError("The replacement text did not produce any tokens.")
        hidden = self.hidden_token_ids()
        literal_prefill_tokens = max(
            0, min(int(literal_prefill_tokens), len(kept))
        )
        # IncrementalDecoder never puts a generated hidden special into its
        # cache, so that ID cannot affect the context-sensitive boundary of
        # what follows. Validate against precisely the IDs the visible decoder
        # sees. Reader-supplied assistant-prefill tokens are the exception:
        # replay forces those visible, including special-token spellings.
        visible_kept = [
            token_id
            for index, token_id in enumerate(kept)
            if index < literal_prefill_tokens or token_id not in hidden
        ]
        kept_text = self._decode_ids(visible_kept)
        expected = kept_text + text

        stop_ids = self._stop_token_ids()
        hidden_ids = hidden - stop_ids
        matched_embedded_stop = False
        matched_hidden = False
        for token_ids in self._replacement_candidates(visible_kept, kept_text, text):
            if token_ids and self._decode_ids(visible_kept + token_ids) == expected:
                # A terminal stop token deliberately ends the new response and
                # stays hidden. One followed by more replacement tokens cannot
                # be literal: generation stops there, so the visible suffix
                # would silently disappear.
                if stop_ids.intersection(token_ids[:-1]):
                    matched_embedded_stop = True
                    continue
                if hidden_ids.intersection(token_ids):
                    matched_hidden = True
                    continue
                return token_ids
        if matched_embedded_stop:
            raise ValueError(
                "The replacement text contains a stop token before its end "
                "and cannot be displayed exactly."
            )
        if matched_hidden:
            raise ValueError(
                "The replacement text contains a hidden special token and "
                "cannot be displayed exactly."
            )
        raise ValueError(
            "The replacement text cannot be inserted exactly at this position "
            "by this tokenizer."
        )

    def _replacement_candidates(
        self, context_ids: Sequence[int], context_text: str, text: str
    ) -> Iterator[list[int]]:
        """Token ids that could spell ``text`` immediately after ``context_ids``.

        The standalone encoding is how a BPE tokenizer with the space inside
        the token normally wants the text. A context-sensitive tokenizer can
        instead need a suffix of the joint encoding. The ids in front may be a
        noncanonical spelling of ``context_text`` - the model can sample one,
        and a reader can leave one behind in an edited prompt - so the joint
        encoding need not begin with them even though its boundary suffix can
        follow them exactly.

        Locate that suffix from the tokenizer's character offsets when they
        exist, or from the bounded seam search used by text scoring for a
        slow tokenizer. Trying every suffix looks harmless but is quadratic:
        each candidate decodes the whole context prefix plus a progressively
        longer tail, all while the model lock is held. The seam can be off by
        one token when one piece crosses it, so validate the boundary and its
        two neighbours. If that seam was only guessed, also bisect decoded
        joint prefixes: a continuation token may not decode correctly by
        itself, but the prefix can still identify its exact start. Candidate
        decoding stays strictly bounded while the final exact decode remains
        the authority, and that decode is the caller's: what disqualifies a
        candidate differs between a response and a prompt.
        """

        assert self.tokenizer is not None
        context = [int(value) for value in context_ids]
        standalone = self._encode_plain(text)
        split = split_context_and_text(
            self.tokenizer, context_text, text, add_special_tokens=False
        )
        joint = split.context_ids + split.text_ids
        boundary = len(split.context_ids)
        decoded_boundary = (
            split.decoded_prefix_end if not split.seam_verified else None
        )
        if not standalone and not joint:
            raise ValueError("The replacement text did not produce any tokens.")

        yield standalone
        aligned_start: int | None = None
        if len(joint) > len(context) and joint[: len(context)] == context:
            aligned_start = len(context)
            aligned = joint[aligned_start:]
            if aligned != standalone:
                yield aligned
        starts = {
            start
            for start in (boundary - 1, boundary, boundary + 1, decoded_boundary)
            if start is not None
            if 0 <= start < len(joint)
        }
        for start in sorted(starts, reverse=True):
            candidate = joint[start:]
            if start != aligned_start and candidate != standalone:
                yield candidate

    def encode_prompt_replacement(
        self,
        prefix_ids: Sequence[int],
        text: str,
        *,
        load_id: str | None = None,
    ) -> list[int]:
        """Encode text the reader typed over one token of a recorded prompt.

        The boundary problem is the response version's: the ids in front plus
        the result must decode to the text in front followed by exactly what
        was typed, so a word reads the same whether the tokenizer keeps its
        leading space or drops it.

        Nothing is filtered out of what the result may contain. A prompt is
        made of template control tokens as much as of words, and putting one
        of them somewhere the template would never have written it is what
        this edit is for; the stop and hidden-special rules that guard a
        replayed response have no counterpart in front of the first sampled
        token.

        ``load_id`` names the load ``prefix_ids`` came from. It is compared
        under the model lock, alongside the encoding, so a load that lands
        between the caller's own check and this call is refused with
        :class:`ModelChanged` rather than answered with tokens from a
        tokenizer the prefix never met.
        """

        with self._lock:
            if not self.loaded:
                raise RuntimeError("Download and load a model before editing a prompt.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            if not text:
                raise ValueError("The replacement text did not produce any tokens.")
            prefix = [int(value) for value in prefix_ids]
            prefix_text = self._decode_ids(prefix)
            expected = prefix_text + text
            for token_ids in self._replacement_candidates(prefix, prefix_text, text):
                if token_ids and self._decode_ids(prefix + token_ids) == expected:
                    return token_ids
            raise ValueError(
                "The replacement text cannot be inserted exactly at this "
                "position by this tokenizer."
            )

    def validate_generation_prefix(
        self,
        messages: list[dict],
        forced_ids: Sequence[int],
        *,
        max_new_tokens: int,
        load_id: str | None = None,
        thinking_mode: str = "default",
        prompt_override_ids: Sequence[int] | None = None,
    ) -> None:
        """Refuse an oversized generation before a stream mutates UI state.

        A typed branch calls this after encoding but before entering the reply
        stream. The expected load is checked under the same model lock as the
        prompt tokenization, so a concurrent reload cannot validate one
        model's token IDs with another model's tokenizer. A learned-position
        model also needs room to feed back all but the last requested sampled
        token; the first comes from the prefill's final logits.

        ``prompt_override_ids`` is the edited prompt the reply being branched
        was given, and is measured in place of anything ``messages`` would
        render - the same prompt the generation itself will be handed, so the
        two agree about what fits.
        """

        with self._lock:
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            if prompt_override_ids is None:
                prompt_ids, _reasoning_prefilled = self._prompt_token_ids(messages, thinking_mode=thinking_mode)
            else:
                prompt_ids = [int(value) for value in prompt_override_ids]
            self._validate_generation_prefix_length(
                prompt_ids,
                forced_ids,
                max_new_tokens=max_new_tokens,
            )

    def _validate_prefix_within_limit(
        self,
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
    ) -> None:
        """Refuse a prefill above the reader's context limit.

        Every generation is checked here, not only a typed branch. The cap is
        a memory guard as much as a replay guard: what a reply costs is mostly
        the key-value cache of everything fed to it, held for as long as the
        answer runs, so an ordinary conversation left to grow can exhaust the
        machine exactly as a pasted branch can.
        """

        assert self.model is not None
        total = len(prompt_ids) + len(forced_ids)
        limit = generation_prefill_token_limit(self.model)
        if total <= limit:
            return
        configured = application_prefill_limit()
        window = model_position_limit(self.model)
        model_bound = window is not None and window <= configured
        ceiling = (
            f"the {limit:,} positions this model can attend to"
            if model_bound
            else f"the {configured:,} token limit for a generation prefix"
        )
        measured = (
            f"The prompt and replayed response are {total:,} tokens"
            if forced_ids
            else f"The prompt is {total:,} tokens"
        )
        shorten = (
            "Shorten the conversation or replacement."
            if forced_ids
            else "Shorten the conversation."
        )
        # Only worth saying when the reader's own cap is what bit: no setting
        # moves a model's position table.
        hint = (
            ""
            if model_bound
            else " Context limit (tokens) on the Settings page raises the cap."
        )
        raise ValueError(f"{measured}, above {ceiling}. {shorten}{hint}")

    def _validate_generation_prefix_length(
        self,
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> None:
        """Check a tokenized generation and its continuation capacity.

        The continuation half is a typed branch's alone. A branch names the
        response length it wants replayed room for up front, so reserving the
        positions before the stream starts is the honest answer; an ordinary
        reply is free to run to the end of the position table, where the
        sampling loop stops it, and keep what it wrote, which is the better
        outcome when the model would have stopped on its own long before the
        requested length.
        """

        assert self.model is not None
        self._validate_prefix_within_limit(prompt_ids, forced_ids)
        total = len(prompt_ids) + len(forced_ids)
        window = model_position_limit(self.model)

        stops_before_sampling = bool(
            forced_ids and int(forced_ids[-1]) in self._stop_token_ids()
        )
        continuation_positions = (
            0 if stops_before_sampling else max(0, int(max_new_tokens) - 1)
        )
        required = total + continuation_positions
        if window is not None and required > window:
            raise ValueError(
                f"The prompt, replayed response, and requested continuation need "
                f"{required:,} positions, above the {window:,} positions this model "
                "can attend to. Shorten the conversation or replacement, or request "
                "fewer new tokens."
            )

    def _response_prefix_ids(
        self,
        text: str,
        *,
        close_reasoning: bool,
        label: str = "assistant prefill",
    ) -> list[int]:
        """Encode a reader-supplied answer prefix without tokenizer wrappers.

        A reasoning model's generation prompt can already end in ``<think>``.
        In that case the supplied text is meant to begin the visible answer,
        so replay a closing marker before it. The marker remains part of the
        measured response prefix, exactly as it would if the model emitted it.
        """

        assert self.tokenizer is not None
        if not text:
            return []
        raw = f"{THINK_CLOSE}\n\n{text}" if close_reasoning else text
        token_ids = self._encode_plain(raw)
        if not token_ids:
            raise ValueError(f"The {label} did not produce any tokens.")
        if self._decode_ids(token_ids) != raw:
            raise ValueError(
                f"The {label} cannot be represented exactly by this tokenizer."
            )
        return token_ids

    def _encode_plain(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids for ``text`` alone, with no chat template around it.

        Special tokens are left off by default because the callers that
        branch a response are splicing into a sequence that already has its
        opening marker. A passage read on its own wants the marker the model
        was trained to see first, and asks for it.
        """

        assert self.tokenizer is not None
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(value) for value in encoded]

    def _decode_ids(self, token_ids: Sequence[int]) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.decode(
            [int(value) for value in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _decode_token(self, token_id: int) -> str:
        return self._decode_ids([token_id])

    def _token_fallback(self, token_id: int) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.convert_ids_to_tokens(int(token_id)) or str(token_id)

    def _describe_token(
        self,
        *,
        position: int,
        token_id: int,
        raw_log_probabilities: np.ndarray,
        sampled_probabilities: np.ndarray,
        segment: str,
    ) -> dict:
        hidden = self._hidden_ids()
        stopping = self._stopping_ids()
        metric: TokenMetric = build_metric(
            position=position,
            token_id=token_id,
            token_text=self._decode_token(token_id),
            fallback_text=self._token_fallback(token_id),
            raw_log_probabilities=raw_log_probabilities,
            sampled_probabilities=sampled_probabilities,
            # The raw decode and the label to show where it is empty, kept
            # apart: a token that decodes to nothing is shown under its
            # vocabulary label, and those labels differ between models, so a
            # comparison of two models' choices needs the decode itself.
            #
            # A stop marker is one of those tokens, though it does not look
            # like one: decoded on its own it comes back as ``</s>`` or
            # ``<|endoftext|>`` rather than empty, because this decode keeps
            # special tokens. A response never shows those characters - the
            # decoder that builds it hides exactly these ids - so recording
            # them here would have two models that both chose to stop read
            # as two different choices, on the strength of what their
            # vocabularies happen to call the marker. The same policy the
            # response text is built under applies to a candidate's text.
            decode_token=lambda candidate_id: (
                "" if candidate_id in hidden else self._decode_token(candidate_id)
            ),
            fallback_token=self._token_fallback,
            # Which of those silences was the end of the response. Without
            # it a model choosing to stop and a model choosing a padding or
            # control marker read as one choice, since neither writes
            # anything and both are hidden.
            stops_token=lambda candidate_id: candidate_id in stopping,
            segment=segment,
        )
        return metric.to_dict()

    def _stop_token_ids(self) -> set[int]:
        assert self.model is not None
        assert self.tokenizer is not None
        values: set[int] = set(self._engine().eos_token_ids())
        candidate = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(candidate, int):
            values.add(candidate)
        elif candidate:
            values.update(int(value) for value in candidate)
        return values

    def _stopping_ids(self) -> frozenset[int]:
        """:meth:`_stop_token_ids`, read once per load; see :meth:`_hidden_ids`."""

        load_id = self.load_id
        if self._stopping_ids_cache[0] != load_id or load_id is None:
            self._stopping_ids_cache = (load_id, frozenset(self._stop_token_ids()))
        return self._stopping_ids_cache[1]

    def _hidden_ids(self) -> frozenset[int]:
        """:meth:`hidden_token_ids`, read once per load.

        The set is asked for per token now - every candidate's text is
        recorded under it - and building it walks every registered special
        token, converting each back to its piece. That is a handful of work
        per call and a few hundred thousand over a long response, for an
        answer that cannot change while one model is in memory.
        """

        load_id = self.load_id
        if self._hidden_ids_cache[0] != load_id or load_id is None:
            self._hidden_ids_cache = (load_id, frozenset(self.hidden_token_ids()))
        return self._hidden_ids_cache[1]

    def hidden_token_ids(self) -> set[int]:
        """Special tokens to keep out of the visible text.

        Reasoning markers are deliberately kept: on models such as OLMo Think
        they are registered as special tokens, and dropping them would leave the
        interface with no way to find the reasoning block.

        A recorded response text is a decode of every token but these, so a
        caller comparing that text against a fresh decode of the same ids needs
        the same set rather than a guess at it.
        """

        assert self.tokenizer is not None
        tokenizer = self.tokenizer
        hidden: set[int] = set()
        for token_id in getattr(tokenizer, "all_special_ids", None) or []:
            piece = tokenizer.convert_ids_to_tokens(int(token_id)) or ""
            if "think" in piece.lower():
                continue
            hidden.add(int(token_id))
        # A stop token ends the response and is never part of it, but a
        # checkpoint can stop on one the tokenizer does not list as special:
        # OLMo 3 ends each turn with <|im_end|>, which its generation config
        # names and Transformers 5 leaves out of all_special_ids.
        if self.model is not None:
            hidden.update(self._stop_token_ids())
        return hidden

    def _prefill(
        self,
        token_ids: list[int],
        *,
        segments: list[str],
        positions: list[int],
        score_from: int,
        collect_from: int = 0,
        sample: Callable[[np.ndarray], np.ndarray] | None = None,
        past_key_values=None,
        cached: int = 0,
    ):
        """Run the model over ``token_ids`` a chunk at a time.

        Returns the per-token metrics, the key-value cache, and the log
        probabilities that predict whatever comes after the sequence. Every
        token except the first is measured against the distribution the model
        held one step earlier, so the same pass that warms the cache also
        explains the prompt.

        ``past_key_values`` is a cache already holding ``cached`` tokens that
        precede ``token_ids``; the pass continues from it rather than from an
        empty one. The metrics and positions still describe ``token_ids``
        alone.

        Tokens before ``collect_from`` get no metric at all, which is how a
        prompt the reader chose not to measure stays out of the results while
        the response tokens that follow it are still described. Tokens from
        there up to ``score_from`` are recorded but left unscored.

        ``sample`` turns raw log probabilities into the distribution the
        sampler would have drawn from. It is applied to ``"response"`` tokens
        only, so a response prefix that is replayed rather than sampled still
        reports the sampling probability and shift it would have had.
        """

        assert self.model is not None
        engine = self._engine()
        metrics: list[dict] = []
        carry: np.ndarray | None = None
        total = len(token_ids)

        for start in range(0, total, PREFILL_CHUNK_SIZE):
            end = min(start + PREFILL_CHUNK_SIZE, total)
            logits, past_key_values = engine.forward(
                token_ids[start:end], past_key_values, cached + start
            )

            for index in range(max(start, collect_from), end):
                token_id = token_ids[index]
                if index == 0 or index < score_from:
                    metrics.append(
                        unscored_metric(
                            position=positions[index],
                            token_id=token_id,
                            token_text=self._decode_token(token_id),
                            fallback_text=self._token_fallback(token_id),
                            segment=segments[index],
                            reason=(
                                UNSCORED_FIRST_TOKEN
                                if index == 0
                                else UNSCORED_BEYOND_LIMIT
                            ),
                        ).to_dict()
                    )
                    continue
                log_probs = (
                    carry
                    if index == start
                    else normalize_log_probabilities(logits.row(index - start - 1))
                )
                assert log_probs is not None
                sampled = (
                    sample(log_probs)
                    if sample is not None and segments[index] == "response"
                    else np.exp(log_probs)
                )
                metrics.append(
                    self._describe_token(
                        position=positions[index],
                        token_id=token_id,
                        raw_log_probabilities=log_probs,
                        sampled_probabilities=sampled,
                        segment=segments[index],
                    )
                )

            carry = normalize_log_probabilities(logits.row(end - start - 1))
            del logits

        return metrics, past_key_values, carry

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
        skip_top_below: float = 0.0,
        analyze_prompt: bool = True,
        tools: list[dict] | None = None,
        forced_ids: Sequence[int] = (),
        prompt_override_ids: Sequence[int] | None = None,
        answer_prefill: str = "",
        thinking_mode: str = "default",
        literal_prefill_tokens: int = 0,
        automatic_reasoning_close_tokens: int = 0,
        literal_text_ranges: Sequence[tuple[int, int]] = (),
        load_id: str | None = None,
        steering: dict | None = None,
    ) -> Iterator[GenerationUpdate]:
        """Stream a reply to ``messages``, one batch of tokens at a time.

        ``forced_ids`` is a response prefix that is replayed instead of sampled:
        the tokens the reader kept from an earlier response, ending in the
        alternative they picked. Sampling resumes after it. Those tokens are
        still measured against the model's own distribution, so a forced
        token the model would never have chosen shows up with the rank and
        surprise it really had. ``max_new_tokens`` counts the tokens sampled
        after the prefix, so a branch made late in a long response still gets
        room to continue. ``answer_prefill`` does the same for arbitrary text;
        when the chat template has opened a reasoning block, it closes that
        block first so the reader's text begins the visible answer.
        ``literal_prefill_tokens`` carries that protected boundary through a
        later branch replay. ``literal_text_ranges`` identifies disjoint token
        ranges typed into a branch so their reasoning markers remain ordinary
        text; it deliberately does not change their stop-token behavior.
        ``automatic_reasoning_close_tokens`` records the leading tokens that
        close a template-supplied reasoning block. A later branch carries that
        provenance forward so the application can keep the control boundary
        from being replaced as though it were answer text.

        ``prompt_override_ids`` is a prompt the reader edited token by token.
        It is fed exactly as it stands: the chat template is what produced
        those ids in the first place, and rendering ``messages`` again would
        undo the edit. ``messages`` still travels with the request because it
        is what the conversation held, and it is what the template arguments
        and the trace describe, but it is not what the model reads.

        ``skip_top_below`` refuses the model's first choice wherever that
        choice holds less than the given probability; see
        :func:`token_metrics.sampling_probabilities`. Zero, the default,
        samples as the model proposed.

        ``load_id`` names the load ``forced_ids`` or ``prompt_override_ids``
        came from (see :attr:`load_id`). It is compared under the model lock,
        before any token is fed, so a load that finished after the caller
        looked is refused with :class:`ModelChanged` rather than feeding one
        model's token IDs through another.

        ``thinking_mode`` is default/on/off for switchable Qwen3 templates.
        Default leaves template arguments untouched. Other architectures ignore
        a saved mode, and updates record None when switching is unsupported.

        ``steering`` is a portable activation-vector specification. Its hook
        is held under the model lock across prefill and decoding, and removed
        before the lock is released, including on cancellation.
        """

        # The application reserves the slot before it publishes its first
        # frame, so by the time this body runs the reservation is normally
        # already held - on its behalf, not by it. Taking it again would
        # deadlock, so this only claims the slot when nobody else has, which is
        # the case for a direct call (tests, or any future non-streaming use):
        # such a call still reports as busy for its whole run and frees the
        # slot afterwards. It releases only what it took.
        #
        # Mutual exclusion never rested on this flag anyway. The model lock
        # below is what keeps two generations off the model at once, and it is
        # still acquired unconditionally.
        reserved = self.reserve_generation()
        # Held so the record below can say how far the run got, whether it
        # finished, was stopped, or raised.
        last: GenerationUpdate | None = None
        self._run_note = None
        self._run_device_bytes = None
        started = time.monotonic()
        try:
            with contextlib.closing(self._generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                skip_top_below=skip_top_below,
                analyze_prompt=analyze_prompt,
                tools=tools,
                forced_ids=forced_ids,
                prompt_override_ids=prompt_override_ids,
                answer_prefill=answer_prefill,
                thinking_mode=thinking_mode,
                literal_prefill_tokens=literal_prefill_tokens,
                automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                literal_text_ranges=literal_text_ranges,
                load_id=load_id,
                steering=steering,
            )) as stream:
                for update in stream:
                    last = update
                    yield update
        except (RuntimeError, MemoryError) as error:
            reraise_out_of_memory(error)
        finally:
            if reserved:
                self.release_generation()
            # Read before the cache goes back, so the figure is the run's own
            # high-water mark rather than what is left after cleaning up.
            self._log_run(last, max_new_tokens, time.monotonic() - started)
            # The key-value cache of this response is the largest thing a run
            # allocates; give it back rather than hold it until the next one.
            self._release_device_cache()

    def _log_run(
        self, last: GenerationUpdate | None, max_new_tokens: int, seconds: float
    ) -> None:
        """Record what a response cost, whatever its outcome.

        One line per response, never per token. This is what makes a memory
        failure readable afterwards: the model, how long the prompt was, how
        many tokens it had produced, and what the device was holding when it
        stopped.

        A run that raised before publishing anything has no update to read,
        so the model and the prompt size come from the note the run left
        behind instead (see :attr:`_run_note`). Only sampled tokens are
        counted: a branch replays its prefix through the model and carries it
        in ``metrics``, but ``max_new_tokens`` bounds the continuation alone,
        and a record saying "140 tokens of at most 40" says nothing.
        """

        try:
            noted_model, noted_prompt = self._run_note or (None, None)
            if last is None:
                model_id, prompt_tokens, produced = noted_model, noted_prompt, 0
            else:
                model_id, prompt_tokens = last.model_id, len(last.prompt_ids)
                produced = len(last.metrics) - last.forced_prefix_tokens
            logger.info(
                "Generated %s tokens of at most %s for %s in %.1fs: "
                "%s prompt tokens, %s held on the device",
                produced,
                max_new_tokens,
                model_id or "no model",
                seconds,
                "unknown" if prompt_tokens is None else prompt_tokens,
                memory_note(self._run_device_bytes),
            )
        except Exception:  # noqa: BLE001 - a log line must not break a reply
            logger.debug("Could not record the run", exc_info=True)

    @_inference_stream
    def _generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
        skip_top_below: float = 0.0,
        analyze_prompt: bool = True,
        tools: list[dict] | None = None,
        forced_ids: Sequence[int] = (),
        prompt_override_ids: Sequence[int] | None = None,
        answer_prefill: str = "",
        thinking_mode: str = "default",
        literal_prefill_tokens: int = 0,
        automatic_reasoning_close_tokens: int = 0,
        literal_text_ranges: Sequence[tuple[int, int]] = (),
        load_id: str | None = None,
        steering: dict | None = None,
    ) -> Iterator[GenerationUpdate]:
        # One response, in the order its steps have to happen: the prompt, the
        # prefix it is forced to start with, one pass that feeds both, the
        # prefix decoded, and the continuation sampled. Each step is its own
        # method; this holds the lock across them and hands each its inputs.
        with self._lock, contextlib.ExitStack() as steering_scope:
            try:
                engine, model_id, producing_load_id = self._start_response(
                    steering_scope, load_id=load_id, steering=steering
                )
                prompt_ids, reasoning_prefilled, recorded_thinking = self._response_prompt(
                    messages,
                    tools=tools,
                    thinking_mode=thinking_mode,
                    prompt_override_ids=prompt_override_ids,
                )
                # Noted here rather than left to the first update, because a run
                # that fails in the prefill below never publishes one and prefill
                # is where a memory failure is most likely.
                self._run_note = (model_id, len(prompt_ids))
                stop_ids = self._stop_token_ids()

                prefix = self._forced_prefix(
                    forced_ids=forced_ids,
                    answer_prefill=answer_prefill,
                    reasoning_prefilled=reasoning_prefilled,
                    literal_prefill_tokens=literal_prefill_tokens,
                    automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                    literal_text_ranges=literal_text_ranges,
                    stop_ids=stop_ids,
                )
                forced = prefix.ids

                # Checked here, on the tokens actually about to be fed, rather
                # than only in validate_generation_prefix(): that one runs for a
                # typed branch before the UI stream starts, and an ordinary reply
                # would otherwise reach the prefill with a conversation of any
                # size behind it. Refuse before the cache is allocated, not once
                # the machine is already out of memory.
                self._validate_prefix_within_limit(prompt_ids, forced)

                sample = _sampler(temperature, top_p, top_k, skip_top_below)
                prompt_metrics, metrics, past_key_values, raw_log_probs, prompt_note = (
                    self._prefill_response(
                        prompt_ids, prefix, analyze_prompt=analyze_prompt, sample=sample
                    )
                )

                rng = np.random.default_rng(int(seed))
                decoder = IncrementalDecoder(self.tokenizer, self.hidden_token_ids())
                literal_prefill_text, literal_text_spans = _decode_forced_prefix(
                    decoder,
                    prefix,
                    metrics,
                    closes_reasoning=bool(answer_prefill) and reasoning_prefilled,
                )
                limit, position_bound = self._response_limit(
                    len(prompt_ids), len(forced), max_new_tokens
                )
                response = _Response(
                    decoder=decoder,
                    metrics=metrics,
                    load_id=producing_load_id,
                    model_id=model_id,
                    prompt_ids=tuple(prompt_ids),
                    prompt_metrics=prompt_metrics,
                    prompt_note=prompt_note,
                    reasoning_prefilled=reasoning_prefilled,
                    thinking_mode=recorded_thinking,
                    forced_prefix_tokens=len(forced),
                    literal_prefill_tokens=prefix.literal_prefill_tokens,
                    literal_prefill_text=literal_prefill_text,
                    literal_text_spans=literal_text_spans,
                )
                yield from self._stream_response(
                    engine,
                    response,
                    prefix,
                    raw_log_probs=raw_log_probs,
                    past_key_values=past_key_values,
                    sample=sample,
                    rng=rng,
                    temperature=temperature,
                    stop_ids=stop_ids,
                    limit=limit,
                    position_bound=position_bound,
                    # Taken before the prefix is published, so the time a
                    # reader spends on that first frame counts toward the
                    # next one's interval, as it always has.
                    last_yield=time.monotonic(),
                )

            finally:
                # Read while the lock is still held. A load queued behind
                # this response takes it the moment it is free, and would
                # be what got measured: its own allocation, or nothing at
                # all if it unloaded first.
                self._run_device_bytes = device_memory.reserved_bytes()

    def _start_response(
        self,
        steering_scope: contextlib.ExitStack,
        *,
        load_id: str | None,
        steering: dict | None,
    ) -> tuple[Engine, str | None, str]:
        """Check the weights can answer, steer them, and say which they are.

        Called with the model lock held. Returns the engine, the model ID and
        the load ID every update of this response is stamped with. The
        steering vector is installed on ``steering_scope``, so it comes off
        when the response ends however it ends.
        """

        if not self.loaded:
            raise RuntimeError("Download and load a model before chatting.")
        if load_id is not None and load_id != self.load_id:
            raise ModelChanged(
                "The model has been reloaded since these tokens were produced."
            )

        # A stale branch must raise ModelChanged before vector/model
        # compatibility is checked, so its handler restores the reply.
        steering_scope.enter_context(self._steering(steering))

        assert self.model is not None
        assert self.tokenizer is not None
        engine = self._engine()
        # A response is where memory runs short, so what the last
        # inspection kept is given back before the prompt is fed.
        self._drop_inspect_cache()
        # Read here, under the lock, alongside the weights: this is the
        # only place the two are guaranteed to agree, which is what makes
        # the stamp on each update worth trusting.
        model_id = self.model_id
        producing_load_id = self.load_id
        assert producing_load_id is not None
        return engine, model_id, producing_load_id

    def _response_prompt(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        thinking_mode: str,
        prompt_override_ids: Sequence[int] | None,
    ) -> tuple[list[int], bool, str | None]:
        """The prompt a response is fed, whether it ends inside reasoning, and the mode.

        The mode returned is the one recorded on every update: the reader's
        choice when the loaded model's template can switch, and ``None`` when
        it cannot, so a record does not claim a mode the model never saw.
        """

        if thinking_mode not in THINKING_MODES:
            raise ValueError("Thinking mode must be default, on, or off.")
        recorded_thinking = thinking_mode if self.supports_thinking else None
        template_args = {"tools": tools} if tools is not None else {}
        if recorded_thinking is not None:
            template_args["thinking_mode"] = recorded_thinking
        if prompt_override_ids is None:
            prompt_ids, reasoning_prefilled = self._prompt_token_ids(
                messages, **template_args
            )
        else:
            prompt_ids = [int(value) for value in prompt_override_ids]
            if not prompt_ids:
                raise ValueError("An edited prompt cannot be empty.")
            # Only the ids can now say whether the prompt ends inside a
            # reasoning block. Asking the template instead would answer
            # for the prompt it would have written, which is precisely
            # the prompt that is not being fed: an edit that removed
            # the opening marker would still be told one was there, and
            # the reply's first words would be filed as reasoning.
            reasoning_prefilled = (
                self._decode_ids(prompt_ids).rstrip().endswith(THINK_OPEN)
            )
        return prompt_ids, reasoning_prefilled, recorded_thinking

    def _forced_prefix(
        self,
        *,
        forced_ids: Sequence[int],
        answer_prefill: str,
        reasoning_prefilled: bool,
        literal_prefill_tokens: int,
        automatic_reasoning_close_tokens: int,
        literal_text_ranges: Sequence[tuple[int, int]],
        stop_ids: set[int],
    ) -> _ForcedPrefix:
        """The tokens a response must start with, and which of them are the reader's.

        Either a branch's replayed tokens (``forced_ids``) or a typed
        assistant prefill, never both. The counts and ranges the caller
        passed are clamped to the prefix actually built: a prefill encodes
        its own tokens, and a branch can be cut short at a stop token.
        """

        if forced_ids and answer_prefill:
            raise ValueError(
                "A token branch and an assistant prefill cannot be applied together."
            )

        forced = [int(value) for value in forced_ids]
        if answer_prefill:
            forced = self._response_prefix_ids(
                answer_prefill, close_reasoning=reasoning_prefilled
            )
            literal_prefill_tokens = len(forced)
            automatic_reasoning_close_tokens = 0
        else:
            literal_prefill_tokens = max(
                0, min(int(literal_prefill_tokens), len(forced))
            )
            automatic_reasoning_close_tokens = max(
                0,
                min(
                    int(automatic_reasoning_close_tokens),
                    literal_prefill_tokens,
                ),
            )

        literal_ranges = _merge_literal_ranges(literal_text_ranges, len(forced))

        # A sampled stop token replayed by a branch still ends the old
        # response where it originally ended. A stop token the reader
        # typed literally into an assistant prefill is ordinary prefix
        # content instead: keep it visible and continue after it.
        for index, token_id in enumerate(forced):
            if token_id in stop_ids and index >= literal_prefill_tokens:
                forced = forced[: index + 1]
                break

        return _ForcedPrefix(
            ids=forced,
            literal_prefill_tokens=literal_prefill_tokens,
            automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
            literal_ranges=literal_ranges,
        )

    def _prefill_response(
        self,
        prompt_ids: list[int],
        prefix: _ForcedPrefix,
        *,
        analyze_prompt: bool,
        sample: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[list[dict], list[dict], Any, np.ndarray | None, str]:
        """Feed the prompt and the forced prefix, and describe what was fed.

        Returns the prompt's metrics, the prefix's (marked with which of its
        tokens the reader supplied), the cache, the log probabilities the
        first sampled token is drawn from, and the note that says when the
        prompt was only partly scored.
        """

        forced = prefix.ids
        # The prompt and the replayed prefix go through the model in one
        # chunked pass. Feeding the prefix back a token at a time would
        # cost a full forward step for every token the reader kept.
        score_from = (
            max(1, len(prompt_ids) - PROMPT_SCORE_LIMIT) if analyze_prompt else 0
        )
        prefilled_metrics, past_key_values, raw_log_probs = self._prefill(
            prompt_ids + forced,
            segments=["prompt"] * len(prompt_ids) + ["response"] * len(forced),
            positions=list(range(1, len(prompt_ids) + 1))
            + list(range(1, len(forced) + 1)),
            score_from=score_from,
            collect_from=0 if analyze_prompt else len(prompt_ids),
            sample=sample,
        )
        prompt_metrics = [
            metric for metric in prefilled_metrics if metric["segment"] == "prompt"
        ]
        metrics: list[dict] = [
            metric for metric in prefilled_metrics if metric["segment"] == "response"
        ]
        for metric in metrics[:prefix.literal_prefill_tokens]:
            metric["literal_prefill"] = True
        for metric in metrics[:prefix.automatic_reasoning_close_tokens]:
            metric["automatic_reasoning_close"] = True
        for start, end in prefix.literal_ranges:
            for metric in metrics[start:end]:
                metric["literal_text"] = True
        prompt_note = ""
        if analyze_prompt and score_from > 1:
            prompt_note = (
                f"Only the most recent {PROMPT_SCORE_LIMIT:,} of "
                f"{len(prompt_ids):,} prompt tokens were scored."
            )
        return prompt_metrics, metrics, past_key_values, raw_log_probs, prompt_note

    def _response_limit(
        self, prompt_length: int, forced_length: int, max_new_tokens: int
    ) -> tuple[int, bool]:
        """The last response position to sample, and whether the model's window set it.

        Each sampled token but the last is fed back at position
        ``prompt_length + position - 1``, and a learned position table has
        no row at or past its window: on CPU that raises, and on Metal it
        reads past the table and the tokens after it are wrong. The prefix
        already fits, so at least one token is always sampled.
        """

        limit = forced_length + int(max_new_tokens)
        window = model_position_limit(self.model)
        position_bound = (
            window is not None and window - prompt_length + 1 < limit
        )
        if position_bound:
            limit = window - prompt_length + 1
        return limit, position_bound

    def _stream_response(
        self,
        engine: Engine,
        response: _Response,
        prefix: _ForcedPrefix,
        *,
        raw_log_probs: np.ndarray | None,
        past_key_values: Any,
        sample: Callable[[np.ndarray], np.ndarray],
        rng: np.random.Generator,
        temperature: float,
        stop_ids: set[int],
        limit: int,
        position_bound: bool,
        last_yield: float,
    ) -> Iterator[GenerationUpdate]:
        """Publish the forced prefix, then sample the rest a token at a time.

        The prefix, when there is one, goes out as a frame of its own, and a
        branch that replays a stop token ends there. Sampling starts from the
        log probabilities the prefill left and ends on a stop token or at
        ``limit``, a frame going out every :data:`STREAM_BATCH_TOKENS` tokens
        or :data:`STREAM_INTERVAL_SECONDS`, whichever comes first, and always
        for the last token.
        """

        forced = prefix.ids
        if forced:
            ends_on_stop = (
                forced[-1] in stop_ids
                and len(forced) > prefix.literal_prefill_tokens
            )
            yield response.update(ends_on_stop_token=ends_on_stop)
            if ends_on_stop:
                return

        prompt_length = len(response.prompt_ids)
        pending_tokens = 0
        for position in range(response.forced_prefix_tokens + 1, limit + 1):
            assert raw_log_probs is not None
            sampled_probs = sample(raw_log_probs)

            if temperature <= 0:
                token_id = int(np.argmax(sampled_probs))
            else:
                token_id = int(rng.choice(sampled_probs.size, p=sampled_probs))

            response.decoder.push(token_id)
            response.metrics.append(
                self._describe_token(
                    position=position,
                    token_id=token_id,
                    raw_log_probabilities=raw_log_probs,
                    sampled_probabilities=sampled_probs,
                    segment="response",
                )
            )
            stopping = token_id in stop_ids or position == limit
            pending_tokens += 1
            now = time.monotonic()
            if (
                stopping
                or pending_tokens >= STREAM_BATCH_TOKENS
                or now - last_yield >= STREAM_INTERVAL_SECONDS
            ):
                pending_tokens = 0
                last_yield = now
                yield response.update(
                    ends_on_stop_token=token_id in stop_ids,
                    ends_on_position_limit=(
                        position_bound
                        and position == limit
                        and token_id not in stop_ids
                    ),
                )

            if stopping:
                break

            # Everything fed so far - the prompt, the replayed prefix
            # and the tokens sampled before this one - is in the
            # cache; this token goes in after them.
            logits, past_key_values = engine.forward(
                [token_id], past_key_values, prompt_length + position - 1
            )
            raw_log_probs = normalize_log_probabilities(logits.row(-1))
