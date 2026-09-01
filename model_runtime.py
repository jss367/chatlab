"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import gc
import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from conversation import THINK_OPEN
from token_metrics import (
    TokenMetric,
    UNSCORED_BEYOND_LIMIT,
    UNSCORED_FIRST_TOKEN,
    build_metric,
    normalize_log_probabilities,
    sampling_probabilities,
    unscored_metric,
)


# Prefill runs in chunks so a long prompt never materializes a
# sequence-length by vocabulary logit tensor all at once.
PREFILL_CHUNK_SIZE = 128

# Scoring every prompt token costs one softmax over the vocabulary each, so
# only the most recent stretch of a very long prompt is measured.
PROMPT_SCORE_LIMIT = 1024

# Refuse to score a wall of pasted text rather than appearing to hang.
SCORE_TOKEN_LIMIT = 4096

# Where a config keeps the length of its position table, newest name first.
# ``max_seq_len`` is MPT's and DBRX's spelling. RWKV's ``context_length`` is
# deliberately absent: it is recurrent, so a longer sequence costs accuracy
# rather than indexing off the end of a table, and capping it would refuse
# passages that run.
POSITION_LIMIT_ATTRIBUTES = (
    "max_position_embeddings",
    "n_positions",
    "n_ctx",
    "max_seq_len",
)

# A window shorter than this is a mislabeled config rather than a real limit —
# no passage worth scoring would fit — so it is ignored in favour of the flat
# application cap.
MIN_MODEL_POSITION_LIMIT = 16


# Streaming updates are batched so a long response does not re-serialize the
# whole token strip on every single token.
STREAM_BATCH_TOKENS = 8
STREAM_INTERVAL_SECONDS = 0.05

# Longest run of tokens held back waiting for a safe split point.
DECODE_CACHE_LIMIT = 32

# Tokens kept after a flush purely as decoder context. Tokenizers are
# context-sensitive: SentencePiece drops the word-boundary space at the start of
# a sequence, and byte-level decoders need the preceding bytes to finish a
# character, so a flush must never look like the start of a fresh sequence.
DECODE_CONTEXT_TOKENS = 8

# A UTF-8 character is at most four bytes, so a byte-level tokenizer needs at
# most this many extra tokens to complete one that a flush would have split.
DECODE_FLUSH_GRACE = 4

REPLACEMENT_CHARACTER = "\ufffd"


MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


def validate_model_id(model_id: str) -> str:
    cleaned = model_id.strip()
    if not MODEL_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Enter a Hugging Face model ID in the form organization/model-name."
        )
    return cleaned


# Stands in for the weight file names when a snapshot has none at all: without
# an index or a weights file there is no way to know what the repo would ship.
MODEL_WEIGHTS = "model weights"

# A causal LM's weights come as one of these, or as the shards its index lists.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


@dataclass(frozen=True)
class CacheStatus:
    """What the Hugging Face cache already holds for one model.

    ``cached_bytes`` counts finished files; ``partial_files`` and
    ``partial_bytes`` count the ``.incomplete`` blobs a cut-off download
    left behind, which ``snapshot_download`` resumes rather than restarts.
    ``missing_files`` names what the snapshot still lacks before the model
    can load: a cache another tool filled with only the config and tokenizer,
    or a download stopped between shards, has finished blobs but no model.
    """

    cached_bytes: int = 0
    partial_files: int = 0
    partial_bytes: int = 0
    missing_files: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        return self.cached_bytes > 0 or self.partial_files > 0

    @property
    def complete(self) -> bool:
        return self.present and not self.partial_files and not self.missing_files

    @property
    def total_bytes(self) -> int:
        return self.cached_bytes + self.partial_bytes


def cache_folder(model_id: str, cache_dir: Path | None = None) -> Path:
    """The ``models--org--name`` folder ``huggingface_hub`` keeps a model in."""

    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    return Path(cache_dir) / f"models--{validate_model_id(model_id).replace('/', '--')}"


def snapshot_folder(folder: Path, revision: str = "main") -> Path | None:
    """The snapshot an offline ``snapshot_download`` would hand back, if any.

    Offline, ``huggingface_hub`` reads ``refs/<revision>`` for the commit and
    returns ``snapshots/<commit>`` whether or not every file is in it.
    """

    ref = folder / "refs" / revision
    if not ref.is_file():
        return None
    snapshot = folder / "snapshots" / ref.read_text().strip()
    return snapshot if snapshot.is_dir() else None


def missing_files(snapshot: Path | None) -> tuple[str, ...]:
    """The files a snapshot needs before ``from_pretrained`` can load it.

    Only the config and the weights are checked. Which tokenizer files a repo
    ships varies too much to know from the outside, and a wrong "incomplete"
    verdict on a good cache would be worse than a generic load error.
    """

    if snapshot is None:
        return ("config.json", MODEL_WEIGHTS)
    missing = []
    if not (snapshot / "config.json").is_file():
        missing.append("config.json")
    if any((snapshot / name).is_file() for name in WEIGHT_FILES):
        return tuple(missing)
    for index_name in WEIGHT_INDEXES:
        index = snapshot / index_name
        if not index.is_file():
            continue
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError):
            missing.append(MODEL_WEIGHTS)
            return tuple(missing)
        missing.extend(
            sorted(shard for shard in shards if not (snapshot / shard).is_file())
        )
        return tuple(missing)
    missing.append(MODEL_WEIGHTS)
    return tuple(missing)


def cache_status(model_id: str, cache_dir: Path | None = None) -> CacheStatus:
    """Measure what is already on disk for ``model_id``, without touching the network."""

    folder = cache_folder(model_id, cache_dir)
    blobs = folder / "blobs"
    if not blobs.is_dir():
        return CacheStatus()
    cached = partial_files = partial_bytes = 0
    for blob in blobs.iterdir():
        if not blob.is_file():
            continue
        size = blob.stat().st_size
        if blob.name.endswith(".incomplete"):
            partial_files += 1
            partial_bytes += size
        else:
            cached += size
    if cached == 0 and partial_files == 0:
        return CacheStatus()
    return CacheStatus(
        cached, partial_files, partial_bytes, missing_files(snapshot_folder(folder))
    )


def format_bytes(count: int) -> str:
    """Render a byte count the way a download dialog would: ``1.2 GB``."""

    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            break
        size /= 1000
    if unit == "B":
        return f"{count} B"
    return f"{size:.1f} {unit}" if size < 100 else f"{size:.0f} {unit}"


class SplitPassage(NamedTuple):
    """A passage's two token runs, and whether the seam between them is sure.

    ``seam_verified`` is false where the boundary between the two runs is a
    guess: a tokenizer that reports no offsets and whose ``decode`` cannot
    say where the seam fell leaves nothing to confirm the position with. The
    ids are still cut from the one encoding of the whole passage, so every
    distribution measured from them is the passage's own; what is in doubt is
    which side of the boundary a token was counted on, and the caller is
    expected to say so rather than present the division as exact.

    ``chat_template_missing`` is set only where the caller asked for the
    context to be wrapped as a chat turn and the tokenizer had no template to
    wrap it in. The numbers are exact either way — they measure the plain
    passage the reader typed — but they answer a different question than the
    one the request implied, so the caller is expected to say which.
    """

    context_ids: list[int]
    text_ids: list[int]
    seam_verified: bool = True
    chat_template_missing: bool = False


def model_position_limit(model) -> int | None:
    """How many positions ``model`` says it has embeddings for, if it says.

    A model with a learned position table has one row per position and nothing
    past the last, so a longer sequence indexes off the end of it: on CPU that
    is an ``IndexError`` raised deep inside the forward pass, and on a device
    that does not bounds-check its gathers it is silently wrong numbers. GPT-2
    is the everyday example, with 1,024 positions against an application cap
    four times that.

    The model's own config is the authority. A tokenizer's ``model_max_length``
    is not consulted: it is routinely a sentinel or a stale copy of a limit the
    weights do not share, and trusting it would refuse passages that work.

    A config that does not say, or says something too short or too broken to be
    a real window, returns ``None`` so the caller keeps the flat cap rather than
    blocking a model that would have run.
    """

    config = getattr(model, "config", None)
    if config is None:
        return None

    # Multimodal configs keep the language model's window one level down, and
    # that inner window is the one the scored tokens are laid out against.
    try:
        config = config.get_text_config() or config
    except (AttributeError, TypeError, ValueError):
        pass

    for name in POSITION_LIMIT_ATTRIBUTES:
        value = getattr(config, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            limit = int(value)
        except (TypeError, ValueError):
            continue
        if limit >= MIN_MODEL_POSITION_LIMIT:
            return limit
    return None


def score_token_limit(model) -> int:
    """The most tokens ``model`` can be asked to score in one pass.

    ``SCORE_TOKEN_LIMIT`` keeps the application responsive; the model's own
    context window keeps the request runnable. The tighter of the two wins.
    """

    window = model_position_limit(model)
    return SCORE_TOKEN_LIMIT if window is None else min(SCORE_TOKEN_LIMIT, window)


def _joint_ids(tokenizer, passage: str, *, add_special_tokens: bool = True) -> list[int] | None:
    """The one encoding of ``passage``, or ``None`` where it cannot be had.

    Every seam decision below is a cut in this list, so the ids that end up
    scored are the passage's own whichever decision was reachable.
    """

    try:
        return [
            int(value)
            for value in tokenizer(
                passage, add_special_tokens=add_special_tokens
            ).input_ids
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _seam_by_decoding(
    tokenizer,
    ids: list[int],
    stop: int,
    context: str,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> int | None:
    """Where the seam falls in ``ids``, when decoding can prove where it fell.

    A tokenizer without offsets can still be asked what a run of ids says, so
    the passage is still encoded once and the seam is found afterwards: the
    context keeps the longest run of leading tokens that decodes to a prefix
    of ``context``, and the token after it — the one that straddles the seam,
    if any — starts the scored text, exactly as it does when offsets are
    available.

    ``add_special_tokens=False`` says the passage already spells out every
    special token it wants — a rendered chat template does — so decoding has
    to keep them to round trip.

    The split is returned only when the two halves decode back to the passage
    verbatim. A ``decode`` that does not round trip — a byte-level merge cut
    mid-character, a normalizer that rewrites whitespace, a SentencePiece
    model that eats a leading space — would otherwise move the seam by a
    token and score part of the context, so those cases say ``None`` and
    leave the placement to :func:`_guess_seam`.
    """

    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None

    def spoken(start: int, end: int) -> str | None:
        try:
            return decode(
                ids[start:end],
                skip_special_tokens=add_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except (NotImplementedError, TypeError, ValueError):
            return None

    def within_context(end: int) -> bool:
        prefix = spoken(0, end)
        return prefix is not None and context.startswith(prefix)

    low, high = 0, stop
    while low < high:
        middle = (low + high + 1) // 2
        if within_context(middle):
            low = middle
        else:
            high = middle - 1
    split = low

    # The token the split lands on has to reach the seam, because it is the
    # one that straddles it. A byte-level merge cut mid-character decodes to
    # a replacement character instead of the text, which strands the search
    # early on a split whose halves still concatenate to the passage; that
    # token would not reach the seam, and this is what catches it. A token
    # that instead stops short of the seam belongs to the context, and means
    # the search above did not find the longest run.
    if split < stop:
        reaches = spoken(0, split + 1)
        if reaches is None or context.startswith(reaches):
            return None
        if not reaches.startswith(context):
            return None

    head, tail = spoken(0, split), spoken(split, stop)
    if head is None or tail is None or head + tail != context + text:
        return None

    return split


def _guess_seam(
    tokenizer,
    ids: list[int],
    stop: int,
    context: str,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> int:
    """The best guess at the seam in ``ids``, for a decode that proved nothing.

    The context is encoded a second time, alone, and its ordinary tokens are
    counted. How many tokens a tokenizer spends on a string is steady even
    where its ``decode`` is not, so that count says how far into the joint
    encoding the context reaches. Special tokens are counted on neither side:
    what a post-processor wrapped the lone context in says nothing about the
    joint passage, while the specials standing in ``ids`` before the first
    ordinary token of the text are the passage's own opening and belong with
    the context.

    The token the count lands on goes to the scored text whenever there is
    text to score, because a seam this hazy is exactly the case where that
    token merged across it.

    The answer is a cut in the joint encoding either way, so the ids are the
    passage's own and the guesswork is confined to which side of the boundary
    one token is counted on.
    """

    specials = set(getattr(tokenizer, "all_special_ids", None) or ())
    alone = _joint_ids(tokenizer, context, add_special_tokens=add_special_tokens) or ()
    wanted = sum(1 for value in alone if value not in specials)

    split, seen = 0, 0
    for index, value in enumerate(ids[:stop]):
        ordinary = value not in specials
        if ordinary and seen >= wanted:
            break
        seen += int(ordinary)
        split = index + 1

    if text and split >= stop:
        split = max(stop - 1, 0)
    return split


_WRAPPER_PROBE = "the"


def _appended_by_the_post_processor(tokenizer) -> int | None:
    """How many ids this tokenizer's post-processor puts after a passage.

    The wrapping is a property of the tokenizer, not of the passage, so it is
    measured rather than inferred: an ordinary word is encoded both ways, and
    whatever the wrapped encoding carries past the end of the bare one is
    what this tokenizer appends to anything. That count then holds for the
    reader's passage too — including a passage that cannot answer the
    question about itself, such as one made of nothing but the very special
    token being appended, where every reading of the ids explains them
    equally well.

    ``None`` says the measurement did not come out, and then nothing about
    the wrapping has been established: the probe would not encode one way or
    the other, or its bare ids do not sit inside its wrapped ids with
    specials and nothing else on either side. A normalizer that only runs
    alongside the post-processor looks like that, and so would a probe that
    is not an ordinary word in this vocabulary. Neither is guessed at.
    """

    specials = set(getattr(tokenizer, "all_special_ids", None) or ())
    wrapped = _joint_ids(tokenizer, _WRAPPER_PROBE, add_special_tokens=True)
    bare = _joint_ids(tokenizer, _WRAPPER_PROBE, add_special_tokens=False)
    if not wrapped or not bare or any(value in specials for value in bare):
        return None

    starts = [
        index
        for index in range(len(wrapped) - len(bare) + 1)
        if wrapped[index : index + len(bare)] == bare
    ]
    if len(starts) != 1:
        return None

    opening, closing = wrapped[: starts[0]], wrapped[starts[0] + len(bare) :]
    if not all(value in specials for value in opening + closing):
        return None
    return len(closing)


def _end_of_written_text(tokenizer, passage: str, ids: list[int]) -> int:
    """Where ``ids`` stops being what the reader wrote, for a slow tokenizer.

    A post-processor's closing EOS or SEP has to come off before the text is
    scored: it lands after the last token the reader wrote, so scoring it
    would report a ``</s>`` nobody pasted. But membership in
    ``all_special_ids`` cannot tell that closer apart from a special token the
    reader pasted at the end of their own text, and someone exploring
    tokenization is exactly the person who pastes ``<|endoftext|>`` to see
    what it does. Dropping it by id would report ranks and perplexity for
    truncated text. The offsets path never had to guess — a pasted token
    carries a real span and an appended one carries ``(0, 0)`` — and this
    asks the same question by provenance, of a tokenizer with no offsets.

    The wrapping is the tokenizer's own, so the first thing asked is the
    tokenizer, not the passage: :func:`_appended_by_the_post_processor`
    measures how much of the wrapping trails a probe, and that many ids come
    off the end of ``ids``. What is left is the reader's, whatever it is made
    of. A passage of nothing but specials — the reader scoring a lone
    ``</s>`` on a tokenizer that opens with one id and closes with the same
    one — is settled that way and no other: read off the passage alone, its
    every arrangement is consistent, so the reader's token and the appended
    closer cannot be told apart there at all.

    A post-processor that appends conditionally can still leave a passage
    with less trailing wrapping than the probe measured, and a count measured
    elsewhere may not be subtracted from a passage that contradicts it. There
    the passage is asked after all, as it was before there was a probe:
    encoding it again with ``add_special_tokens=False`` says which specials
    are the reader's, because whatever survives that encoding is theirs and
    the post-processor's are the ones that appear only when it runs, so
    ``ids`` is that bare encoding wrapped in specials and the wrapping
    *suffix* is what comes off. The shortest such suffix is taken, which is
    what leaves the ambiguous case above to the measurement rather than to
    this.

    Where neither the measurement nor the passage proves anything — the probe
    refused, and a normalizer that rewrites the passage when the
    post-processor is off — the whole trailing run comes off as it always
    did: a pasted token can still be lost there, but no reader is scored on a
    closer they never wrote. None of this is asked for unless there is a
    trailing special to account for, so a passage that ends in an ordinary
    token still costs the one encoding it always did.
    """

    specials = set(getattr(tokenizer, "all_special_ids", None) or ())
    swept = len(ids)
    while swept and ids[swept - 1] in specials:
        swept -= 1
    if swept == len(ids):
        return len(ids)

    # An appended special is part of the trailing run, so a measurement that
    # claims more than that run holds is not describing this passage.
    appended = _appended_by_the_post_processor(tokenizer)
    if appended is not None and appended <= len(ids) - swept:
        return len(ids) - appended

    # The same bound holds for a suffix read off the passage itself; the
    # opening specials are whatever is left in front of the bare encoding.
    bare = _joint_ids(tokenizer, passage, add_special_tokens=False)
    if not bare:
        return swept
    for appended in range(len(ids) - swept + 1):
        end = len(ids) - appended
        start = end - len(bare)
        if start < 0:
            break
        if ids[start:end] == bare and all(value in specials for value in ids[:start]):
            return end
    return swept


def _encode_halves_apart(
    tokenizer, context: str, text: str, *, add_special_tokens: bool = True
) -> SplitPassage:
    """Encode the halves separately, for a passage that will not encode whole.

    This is the one path whose ids are not a cut of a single encoding, and it
    runs only where the tokenizer refused the passage outright. Neither half
    is post-processed: asking for specials here is what lets a closing EOS or
    SEP land between the context and the text, where the scored text would
    read as what follows the end of a passage rather than what follows the
    context. The opening token the model does expect is prepended by name
    instead, so nothing can be appended in the seam's way.
    """

    def ids_for(part: str) -> list[int]:
        return list(_joint_ids(tokenizer, part, add_special_tokens=False) or ())

    context_ids = ids_for(context)
    opening = getattr(tokenizer, "bos_token_id", None) if add_special_tokens else None
    if opening is not None:
        context_ids.insert(0, int(opening))
    return SplitPassage(context_ids, ids_for(text), seam_verified=not context)


def split_context_and_text(
    tokenizer, context: str, text: str, *, add_special_tokens: bool = True
) -> SplitPassage:
    """Tokenize ``context + text`` as one passage, then split at the seam.

    Encoding the two halves separately can give a different sequence from
    encoding the passage the reader actually sees: BPE and SentencePiece merge
    across the seam, and a leading space or start-of-string rule can change the
    first scored token. Scoring the concatenation of two independent encodings
    would therefore report ranks for a sequence the text never produces.

    A token that straddles the seam covers characters from both halves; it is
    counted as part of the scored text, so every character of ``text`` is
    covered by a token that gets measured. Special tokens carry an empty
    ``(0, 0)`` span, so leading ones stay on the context side.

    A post-processor that appends EOS or SEP puts an empty span *after* the
    last text token, where the seam search cannot exclude it. Such trailing
    tokens are dropped outright rather than moved to the context, which comes
    first in the sequence: they are the final tokens, so no earlier token's
    score depends on them, and scoring them would report a ``</s>`` the reader
    never pasted.

    ``add_special_tokens=False`` is for a context that already carries its own
    special tokens — a chat template renders its own BOS and role markers — so
    the tokenizer must not prepend a second one.

    Offsets say where the seam fell; a slow tokenizer's ``decode`` can prove
    it; and where neither can, the passage is still encoded once and cut at
    the position :func:`_guess_seam` counts out, with ``seam_verified``
    cleared. That keeps the scored ids the passage's own in every case that
    can be encoded at all, so no distribution is ever taken from a sequence
    the reader did not write; what an unverified seam leaves in doubt is only
    which side of the boundary a single token was counted on.
    """

    if getattr(tokenizer, "is_fast", False):
        try:
            encoded = tokenizer(
                context + text,
                return_offsets_mapping=True,
                add_special_tokens=add_special_tokens,
            )
            ids = [int(value) for value in encoded["input_ids"]]
            offsets = list(encoded["offset_mapping"])
        except (NotImplementedError, KeyError, TypeError, ValueError):
            ids, offsets = [], []
        if ids and len(offsets) == len(ids):
            seam = len(context)
            split = next(
                (
                    index
                    for index, (_, end) in enumerate(offsets)
                    if int(end) > seam
                ),
                len(ids),
            )
            stop = len(ids)
            while stop > split:
                start, end = offsets[stop - 1]
                if int(end) > int(start):
                    break
                stop -= 1
            return SplitPassage(ids[:split], ids[split:stop])

    ids = _joint_ids(tokenizer, context + text, add_special_tokens=add_special_tokens)
    if not ids:
        return _encode_halves_apart(
            tokenizer, context, text, add_special_tokens=add_special_tokens
        )

    # Trailing special tokens the post-processor appended are dropped for the
    # same reason they are dropped under the offsets path: they close the
    # passage off after the last token the reader wrote. Which of them the
    # reader wrote is the question :func:`_end_of_written_text` answers.
    stop = len(ids)
    if add_special_tokens:
        stop = _end_of_written_text(tokenizer, context + text, ids)

    split = _seam_by_decoding(
        tokenizer, ids, stop, context, text, add_special_tokens=add_special_tokens
    )
    verified = split is not None
    if split is None:
        split = _guess_seam(
            tokenizer, ids, stop, context, text, add_special_tokens=add_special_tokens
        )

    # With no context there is no seam for a merge to cross — the joint
    # encoding is the text's own, plus whatever specials the tokenizer
    # prepends — so the cut in front of the first ordinary token is exact.
    return SplitPassage(
        ids[:split], ids[split:stop], seam_verified=verified or not context
    )


def encode_for_scoring(
    tokenizer,
    text: str,
    *,
    context: str = "",
    use_chat_template: bool = False,
) -> SplitPassage:
    """Turn a context and the text to score into their two token runs.

    A context that carries actual words is wrapped in the chat template when
    the caller asks for it. The template is rendered to **text** and handed to
    :func:`split_context_and_text` like any other context, because the seam it
    leaves is not protected: a generation prompt ends in ordinary characters
    after its last special token — a newline behind ``<|im_start|>assistant``,
    a bare ``<think>``, a trailing space — and those merge with the start of the
    reply just as any other seam does. Encoding the halves apart would report
    ranks for a first reply token the model never sees. The template renders
    its own special tokens, so the tokenizer is told not to add a second BOS.

    Everything else goes through :func:`split_context_and_text` with the
    context **verbatim**, whitespace included. A context of a single space is a
    real choice — it decides which token the text begins with under BPE — so
    stripping it would report ranks for a passage the reader never wrote. That
    holds for the template path too: a turn of pure whitespace is still a turn
    the reader asked to send, and wrapping it is what the box promised, so
    only a genuinely empty box falls through — there is no message there for
    any template to render.

    A tokenizer with no chat template at all — GPT-2, say — also falls through
    to the plain path, and ``chat_template_missing`` says so. Scoring is not
    refused over it: the plain concatenation is exactly the characters the
    reader typed, and its probabilities are exact measurements of that
    passage, just not of a chat turn. Nor is the ``Role: content`` transcript
    that chat generation falls back to used here. Generation has to invent
    some framing to get a reply at all; scoring does not, and that transcript
    is no more the model's own format than the raw context is, so it would
    substitute an invention for the reader's own words and still need this
    same caveat. What the reader gets instead is the passage they wrote, and
    a sentence saying the turn was not applied.
    """

    template = getattr(tokenizer, "chat_template", None)
    wants_template = bool(context) and use_chat_template
    if wants_template and template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(rendered, str):
            rendered = rendered[0]
        return split_context_and_text(
            tokenizer, rendered, text, add_special_tokens=False
        )

    return split_context_and_text(tokenizer, context, text)._replace(
        chat_template_missing=wants_template
    )


@dataclass(frozen=True)
class GenerationUpdate:
    text: str
    metrics: list[dict]
    """Live list owned by the generator. Copy it before storing it anywhere."""

    prompt_metrics: list[dict] = field(default_factory=list)
    prompt_note: str = ""
    reasoning_prefilled: bool = False
    """Whether the prompt already ended with the opening ``<think>`` marker.

    When it did, ``text`` starts inside the reasoning block and never contains
    an opening marker of its own, so a caller splitting reasoning from the
    answer has to be told.
    """


class IncrementalDecoder:
    """Decode a growing token stream without re-decoding it from the start.

    Tokens are held in a small cache until a whitespace boundary makes their
    text final, which keeps each step proportional to the cache rather than to
    the length of the response.

    ``text`` always equals a full decode of every token pushed so far. That
    holds because each cache window keeps a suffix of the previous one as
    decoder context, so no decode ever starts in the middle of a sequence and
    loses a word-boundary space or half of a multi-byte character.
    """

    def __init__(self, tokenizer, skip_ids: set[int] | None = None) -> None:
        self._tokenizer = tokenizer
        self._skip_ids = skip_ids or set()
        self._cache: list[int] = []
        self._context = 0
        """How many leading entries of ``_cache`` are kept only as context."""
        self._settled = ""
        self._pending = ""
        self._printed = 0

    @property
    def text(self) -> str:
        return self._settled + self._pending

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _ready_to_flush(self, decoded: str) -> bool:
        fresh = len(self._cache) - self._context
        if fresh >= DECODE_CACHE_LIMIT + DECODE_FLUSH_GRACE:
            return True
        if decoded.endswith(REPLACEMENT_CHARACTER):
            # Half of a multi-byte character is still in the cache. Settling now
            # would freeze the replacement character into the text for good, so
            # wait for the token that completes it.
            return False
        return decoded.endswith("\n") or fresh >= DECODE_CACHE_LIMIT

    def _flush(self) -> None:
        """Start a new cache window, keeping a token suffix as decoder context.

        The already-settled text is re-derived from that suffix, so the next
        decode continues the sequence instead of restarting it.
        """

        self._cache = self._cache[len(self._cache) - DECODE_CONTEXT_TOKENS :]
        self._context = len(self._cache)
        self._printed = len(self._decode(self._cache))
        self._pending = ""

    def push(self, token_id: int) -> None:
        if token_id in self._skip_ids:
            return
        self._cache.append(token_id)
        decoded = self._decode(self._cache)

        if self._ready_to_flush(decoded):
            self._settled += decoded[self._printed :]
            self._flush()
            return

        boundary = decoded.rfind(" ") + 1
        if boundary > self._printed:
            self._settled += decoded[self._printed : boundary]
            self._printed = boundary
        self._pending = decoded[self._printed :]


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


class ModelManager:
    """Own the single in-memory model used by the local application."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id: str | None = None
        self.local_path: Path | None = None
        self.device_name: str | None = None
        self._lock = threading.RLock()
        # A separate, non-reentrant flag for "a generation is running right
        # now". The model lock cannot answer that question: it is reentrant
        # (load() nests unload() inside it), so a test on the holding thread -
        # and, more importantly, any future nested use - would see it as free.
        #
        # A plain Lock, deliberately: it is acquired and released by whichever
        # worker thread happens to be running the generator at the time, and
        # Gradio is free to resume a streaming handler on a different thread
        # than the one that started it. An RLock, or any owner-checked
        # primitive, would refuse the release from that second thread.
        self._generating = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def busy(self) -> bool:
        """True while the generation slot is reserved.

        Never blocks, so it can only ever be an early exit: a caller that is
        about to generate has to take the slot with reserve_generation()
        rather than act on this answer.
        """

        return self._generating.locked()

    def reserve_generation(self) -> bool:
        """Claim the right to run a generation, or report that it is taken.

        Never blocks: a caller that loses the race must refuse, not queue.
        Queuing is what corrupts the conversation - a handler that waited would
        resume holding the inputs Gradio captured when its click was queued,
        and write that stale snapshot over everything the running generation
        produced in the meantime.

        The caller must reserve *before* publishing its first frame and release
        in a ``finally``. Checking :attr:`busy` and then generating is not the
        same thing: those two steps are separated by a yield, and Gradio does
        not resume a streaming handler until the browser has been sent the
        frame, so the window between them is a network round trip wide.
        """

        return self._generating.acquire(blocking=False)

    def release_generation(self) -> None:
        """Give the generation slot back. Pairs with a successful reservation."""

        self._generating.release()

    def download(self, model_id: str, hf_token: str | None = None) -> Path:
        from huggingface_hub import snapshot_download

        checked_id = validate_model_id(model_id)
        path = snapshot_download(
            repo_id=checked_id,
            token=hf_token.strip() if hf_token and hf_token.strip() else None,
        )
        return Path(path)

    def load(self, model_id: str, local_path: Path) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self._lock:
            self.unload()
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif torch.backends.mps.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to("mps")
                device_name = "Apple Metal (MPS)"
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                device_name = "CPU"

            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.model_id = validate_model_id(model_id)
            self.local_path = local_path
            self.device_name = device_name
            return device_name

    def unload(self) -> None:
        import torch

        with self._lock:
            self.model = None
            self.tokenizer = None
            self.model_id = None
            self.local_path = None
            self.device_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def _prompt_token_ids(self, messages: list[dict]) -> tuple[list[int], bool]:
        """Token ids for a chat prompt, and whether it prefills ``<think>``.

        Reasoning templates such as OLMo Think end the generation prompt with
        the opening marker, so the model resumes inside the block and never
        emits an opener. The flag rides along to the caller because only the
        prompt can reveal it.
        """

        assert self.tokenizer is not None
        tokenizer = self.tokenizer
        prefilled = False

        if tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prefilled = isinstance(rendered, str) and rendered.rstrip().endswith(
                THINK_OPEN
            )
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        else:
            transcript = "\n".join(
                f"{message['role'].title()}: {message['content']}"
                for message in messages
            )
            encoded = tokenizer(f"{transcript}\nAssistant:").input_ids

        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(value) for value in encoded], prefilled

    def _decode_token(self, token_id: int) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

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
        metric: TokenMetric = build_metric(
            position=position,
            token_id=token_id,
            token_text=self._decode_token(token_id),
            fallback_text=self._token_fallback(token_id),
            raw_log_probabilities=raw_log_probabilities,
            sampled_probabilities=sampled_probabilities,
            decode_token=lambda candidate_id: (
                self._decode_token(candidate_id) or self._token_fallback(candidate_id)
            ),
            segment=segment,
        )
        return metric.to_dict()

    def _stop_token_ids(self) -> set[int]:
        assert self.model is not None
        assert self.tokenizer is not None
        values: set[int] = set()
        for candidate in (
            self.tokenizer.eos_token_id,
            getattr(self.model.generation_config, "eos_token_id", None),
        ):
            if isinstance(candidate, int):
                values.add(candidate)
            elif candidate:
                values.update(int(value) for value in candidate)
        return values

    def _hidden_token_ids(self) -> set[int]:
        """Special tokens to keep out of the visible text.

        Reasoning markers are deliberately kept: on models such as OLMo Think
        they are registered as special tokens, and dropping them would leave the
        interface with no way to find the reasoning block.
        """

        assert self.tokenizer is not None
        tokenizer = self.tokenizer
        hidden: set[int] = set()
        for token_id in getattr(tokenizer, "all_special_ids", None) or []:
            piece = tokenizer.convert_ids_to_tokens(int(token_id)) or ""
            if "think" in piece.lower():
                continue
            hidden.add(int(token_id))
        return hidden

    def _prefill(
        self,
        token_ids: list[int],
        *,
        segments: list[str],
        positions: list[int],
        score_from: int,
        collect: bool,
    ):
        """Run the model over ``token_ids`` a chunk at a time.

        Returns the per-token metrics, the key-value cache, and the log
        probabilities that predict whatever comes after the sequence. Every
        token except the first is measured against the distribution the model
        held one step earlier, so the same pass that warms the cache also
        explains the prompt.
        """

        import torch

        assert self.model is not None
        model = self.model
        device = next(model.parameters()).device
        metrics: list[dict] = []
        past_key_values = None
        carry: np.ndarray | None = None
        total = len(token_ids)

        for start in range(0, total, PREFILL_CHUNK_SIZE):
            end = min(start + PREFILL_CHUNK_SIZE, total)
            chunk = torch.tensor(
                [token_ids[start:end]], dtype=torch.long, device=device
            )
            outputs = model(
                input_ids=chunk,
                attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0]

            if collect:
                for index in range(start, end):
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
                        else normalize_log_probabilities(
                            logits[index - start - 1].detach().float().cpu().numpy()
                        )
                    )
                    assert log_probs is not None
                    metrics.append(
                        self._describe_token(
                            position=positions[index],
                            token_id=token_id,
                            raw_log_probabilities=log_probs,
                            sampled_probabilities=np.exp(log_probs),
                            segment=segments[index],
                        )
                    )

            carry = normalize_log_probabilities(
                logits[end - start - 1].detach().float().cpu().numpy()
            )
            del outputs, logits

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
        analyze_prompt: bool = True,
    ) -> Iterator[GenerationUpdate]:
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
        try:
            yield from self._generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                analyze_prompt=analyze_prompt,
            )
        finally:
            if reserved:
                self.release_generation()

    def _generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
        analyze_prompt: bool = True,
    ) -> Iterator[GenerationUpdate]:
        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")

            assert self.model is not None
            assert self.tokenizer is not None
            model = self.model
            tokenizer = self.tokenizer
            device = next(model.parameters()).device

            prompt_ids, reasoning_prefilled = self._prompt_token_ids(messages)
            score_from = (
                max(1, len(prompt_ids) - PROMPT_SCORE_LIMIT) if analyze_prompt else 0
            )
            prompt_metrics, past_key_values, raw_log_probs = self._prefill(
                prompt_ids,
                segments=["prompt"] * len(prompt_ids),
                positions=list(range(1, len(prompt_ids) + 1)),
                score_from=score_from,
                collect=analyze_prompt,
            )
            prompt_note = ""
            if analyze_prompt and score_from > 1:
                prompt_note = (
                    f"Only the most recent {PROMPT_SCORE_LIMIT:,} of "
                    f"{len(prompt_ids):,} prompt tokens were scored."
                )

            rng = np.random.default_rng(int(seed))
            metrics: list[dict] = []
            stop_ids = self._stop_token_ids()
            decoder = IncrementalDecoder(tokenizer, self._hidden_token_ids())
            limit = int(max_new_tokens)
            pending_tokens = 0
            last_yield = time.monotonic()

            for position in range(1, limit + 1):
                assert raw_log_probs is not None
                sampled_probs = sampling_probabilities(
                    raw_log_probs,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )

                if temperature <= 0:
                    token_id = int(np.argmax(sampled_probs))
                else:
                    token_id = int(rng.choice(sampled_probs.size, p=sampled_probs))

                decoder.push(token_id)
                metrics.append(
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
                    yield GenerationUpdate(
                        text=decoder.text,
                        metrics=metrics,
                        prompt_metrics=prompt_metrics,
                        prompt_note=prompt_note,
                        reasoning_prefilled=reasoning_prefilled,
                    )

                if stopping:
                    break

                outputs = model(
                    input_ids=torch.tensor(
                        [[token_id]], dtype=torch.long, device=device
                    ),
                    attention_mask=torch.ones(
                        (1, len(prompt_ids) + position), dtype=torch.long, device=device
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                raw_log_probs = normalize_log_probabilities(
                    outputs.logits[0, -1].detach().float().cpu().numpy()
                )

    def score_text(
        self,
        text: str,
        *,
        context: str = "",
        use_chat_template: bool = False,
    ) -> ScoredText:
        """Measure text the model did not write, in one pass over the tokens."""

        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before scoring text.")

            assert self.tokenizer is not None
            tokenizer = self.tokenizer
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

            metrics, _, _ = self._prefill(
                token_ids,
                segments=["prompt"] * len(context_ids) + ["response"] * len(text_ids),
                positions=list(range(1, len(context_ids) + 1))
                + list(range(1, len(text_ids) + 1)),
                score_from=1,
                collect=True,
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
            )
