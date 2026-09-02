"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import contextlib
import gc
import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from conversation import THINK_CLOSE, THINK_OPEN
from token_metrics import (
    UNSCORED_BEYOND_LIMIT,
    UNSCORED_FIRST_TOKEN,
    TokenMetric,
    build_metric,
    entropy_bits,
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

# A causal LM's weights come as a single file or as the shards an index lists,
# in one of these formats. ``from_pretrained`` looks for them in this order and
# loads the first it finds, so a snapshot is judged by that format alone.
WEIGHT_FORMATS = (
    ("model.safetensors", "model.safetensors.index.json"),
    ("pytorch_model.bin", "pytorch_model.bin.index.json"),
)


@dataclass(frozen=True)
class CacheStatus:
    """What the Hugging Face cache already holds for one model.

    ``missing_files`` names what the ``main`` snapshot still lacks before the
    model can load, and is the one verdict on that: a cache another tool
    filled with only the config and tokenizer, or a download stopped between
    shards, has finished blobs but no model. ``cached_bytes`` counts finished
    files; ``partial_files`` and ``partial_bytes`` count the ``.incomplete``
    blobs a cut-off download left behind, which ``snapshot_download`` resumes
    rather than restarts. Those are a size estimate, not a verdict: the blob
    folder is shared by every revision of the repo, so a stray partial may
    belong to another revision or to a file the model never loads, and a
    partial the snapshot does need already shows up in ``missing_files``,
    since the hub links a file into the snapshot only once it has finished.
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
        return self.present and not self.missing_files

    @property
    def total_bytes(self) -> int:
        return self.cached_bytes + self.partial_bytes


@dataclass(frozen=True)
class CachedModel:
    """One Hugging Face model repository found in the local disk cache."""

    model_id: str
    status: CacheStatus


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
    for single, index_name in WEIGHT_FORMATS:
        if (snapshot / single).is_file():
            return tuple(missing)
        index = snapshot / index_name
        if not index.is_file():
            continue
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map must be a non-empty object")
            shards = set(weight_map.values())
            if not all(isinstance(shard, str) and shard for shard in shards):
                raise TypeError("weight_map values must be file names")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
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
    snapshot = snapshot_folder(folder)
    cached = partial_files = partial_bytes = 0
    blobs = folder / "blobs"
    if blobs.is_dir():
        for blob in blobs.iterdir():
            if not blob.is_file():
                continue
            size = blob.stat().st_size
            if blob.name.endswith(".incomplete"):
                partial_files += 1
                partial_bytes += size
            else:
                cached += size
    # On a filesystem without symlinks (an exFAT drive, say) the hub moves each
    # finished file into the snapshot itself and leaves ``blobs/`` empty, so
    # the snapshot's own regular files are cached bytes too. In the usual
    # layout every entry there is a symlink and counts nothing twice.
    if snapshot is not None:
        for entry in snapshot.rglob("*"):
            if entry.is_file() and not entry.is_symlink():
                cached += entry.stat().st_size
    if cached == 0 and partial_files == 0:
        return CacheStatus()
    return CacheStatus(cached, partial_files, partial_bytes, missing_files(snapshot))


def cached_models(cache_dir: Path | None = None) -> tuple[CachedModel, ...]:
    """List model repositories that have files in the Hugging Face cache.

    Hugging Face encodes ``owner/name`` as ``models--owner--name``. Splitting
    only at the first separator preserves a model name that itself contains a
    double hyphen. Cache entries without an owner cannot be selected in
    Chatlab, whose model field deliberately accepts full repository IDs only.
    """

    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    root = Path(cache_dir)
    try:
        folders = tuple(root.iterdir())
    except OSError:
        return ()

    found = []
    for folder in folders:
        if not folder.is_dir() or not folder.name.startswith("models--"):
            continue
        owner, separator, name = folder.name.removeprefix("models--").partition("--")
        if not separator:
            continue
        model_id = f"{owner}/{name}"
        try:
            status = cache_status(model_id, root)
        except (OSError, ValueError):
            continue
        if status.present:
            found.append(CachedModel(model_id, status))

    return tuple(
        sorted(
            found,
            key=lambda model: (not model.status.complete, model.model_id.casefold()),
        )
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

    forced_prefix_tokens: int = 0
    """How many leading response tokens were replayed instead of sampled."""

    literal_prefill_text: str = ""
    """Decoded prefix whose reader-supplied portion must remain literal."""

    prompt_ids: tuple[int, ...] = ()
    """Every prompt token, measured or not.

    :meth:`ModelManager.inspect` needs the whole sequence the response was
    generated from, and ``prompt_metrics`` only holds the tokens the reader
    chose to measure.
    """

    model_id: str | None = None
    load_id: str | None = None
    """Which weights produced this update, read under the model lock.

    A caller that looks at the manager instead can be wrong: a load may land
    between the caller's look and the moment the generator takes the lock,
    and a caller that saw a model change would have to do without a stamp.
    ``load_id`` is what :meth:`ModelManager.inspect` checks against; see
    :attr:`ModelManager.load_id`.
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

    @property
    def stable_text(self) -> str:
        """Decoded text excluding an incomplete multi-token character suffix."""

        if self._pending.endswith(REPLACEMENT_CHARACTER):
            return self._settled + self._pending.rstrip(REPLACEMENT_CHARACTER)
        return self.text

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

    def push(self, token_id: int, *, force_visible: bool = False) -> None:
        if token_id in self._skip_ids and not force_visible:
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
    context_ids: tuple[int, ...] = ()


# The final norm of a decoder stack, under the names the common architectures
# give it. Llama, OLMo, Mistral and Qwen say ``norm``; GPT-2 says ``ln_f``;
# OPT and BLOOM say ``final_layer_norm``; Mamba says ``norm_f``. Some models
# keep it one level down from the base model: OPT's ``OPTModel`` wraps a
# ``decoder`` that owns the norm, and multimodal wrappers hold their text
# stack as ``language_model``.
FINAL_NORM_ATTRIBUTES = ("norm", "final_layer_norm", "ln_f", "final_norm", "norm_f")
FINAL_NORM_CONTAINERS = ("decoder", "transformer", "model", "language_model")


class ModelChanged(RuntimeError):
    """The weights in memory are not the ones the caller's tokens came from."""


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


class DownloadSnapshot(NamedTuple):
    """One reading of a download: how many files and bytes are in, out of how many.

    ``bytes_total`` covers only files that need fetching; a file already in the
    cache finishes without ever reporting a size, so it counts in ``files_*``
    alone.
    """

    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0

    @property
    def started(self) -> bool:
        """Whether the Hub has answered with the file list yet."""

        return self.files_total > 0

    @property
    def fraction(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        return min(1.0, self.bytes_done / self.bytes_total)


class DownloadProgress:
    """Live totals for one snapshot download, safe to read from another thread.

    ``snapshot_download`` reports through tqdm rather than callbacks: one bar
    over files advances as each finishes, and two byte bars (network transfer
    and bytes reconstructed on disk) grow their ``total`` as each file learns
    its size. :meth:`bar_class` gives it a silent tqdm that records those
    numbers here instead of drawing them.

    The byte bars belong to the snapshot, not to its files: since
    huggingface_hub 1.1 every per-file download feeds them through an internal
    aggregating stand-in, so the ``tqdm_class`` handed to ``snapshot_download``
    only ever sees the file bar and those snapshot-wide byte bars (one of them
    before 1.23, which added the transfer bar). That is why byte totals below
    are read as a maximum across byte bars rather than a sum: they are two
    views of the same bytes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: list = []

    def bar_class(self) -> type:
        """A tqdm class for ``snapshot_download(tqdm_class=...)`` that reports here."""

        from huggingface_hub.utils import tqdm as hub_tqdm

        progress = self

        class RecordingBar(hub_tqdm):
            def __init__(self, *args, **kwargs) -> None:
                self.counts_bytes = kwargs.get("unit") == "B"
                # A disabled bar never writes to the terminal, and tqdm's own
                # update() drops the count on the floor for one, so it is kept
                # here instead.
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                progress._register(self)

            def update(self, n=1) -> None:
                if n:
                    with progress._lock:
                        self.n += n

            def __iter__(self):
                # tqdm's own __iter__ skips counting for a disabled bar. Until
                # huggingface_hub 1.25 the file bar is tqdm's thread_map, which
                # (before tqdm 4.70) advances it by iterating rather than by
                # update(), so count here.
                for item in self.iterable:
                    yield item
                    self.update(1)

        return RecordingBar

    def _register(self, bar) -> None:
        with self._lock:
            self._bars.append(bar)

    def snapshot(self) -> DownloadSnapshot:
        with self._lock:
            bars = list(self._bars)
            files_done = files_total = 0
            bytes_done = bytes_total = 0
            for bar in bars:
                count = int(bar.n or 0)
                total = int(bar.total or 0)
                if bar.counts_bytes:
                    # Transfer and reconstruction count the same bytes from
                    # the two ends of the pipe. Whichever is further along is
                    # the truer picture: a resumed file's on-disk bytes are
                    # credited to reconstruction only, and network bytes lead
                    # the disk for the rest.
                    bytes_done = max(bytes_done, count)
                    bytes_total = max(bytes_total, total)
                else:
                    files_done += count
                    files_total += total
        return DownloadSnapshot(files_done, files_total, bytes_done, bytes_total)


class ModelManager:
    """Own the single in-memory model used by the local application."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id: str | None = None
        self.local_path: Path | None = None
        self.device_name: str | None = None
        # Counts successful loads, so state produced under one set of weights
        # can be told from state produced under the next even when both came
        # from the same repository ID (a re-download at a newer revision).
        self.load_count = 0
        # Downloads under way right now, by model ID, so a second request for
        # the same model can follow the first instead of racing it for the
        # same files.
        self.active_downloads: dict[str, DownloadProgress] = {}
        # Guards active_downloads, so that "is anyone fetching this?" and "then
        # I am" happen as one step: two handlers asking at the same instant
        # must come away with one download between them, not one each.
        self._downloads_lock = threading.Lock()
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
    def load_id(self) -> str | None:
        """Identify the weights in memory: the model ID plus which load this is.

        Two loads of the same repository ID can hold different snapshots, so
        anything that must be read back by the model that produced it is
        stamped with this rather than the ID alone. ``None`` when nothing is
        loaded.
        """

        if not self.loaded:
            return None
        return f"{self.model_id}#{self.load_count}"

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

    def download(
        self,
        model_id: str,
        hf_token: str | None = None,
        progress: DownloadProgress | None = None,
    ) -> Path:
        """Fetch ``model_id`` into the Hugging Face cache and return its snapshot.

        Blocks until the last byte; ``progress`` is how a caller on another
        thread watches it happen. Files already cached are skipped, and a
        partial file left by an interrupted download is resumed.

        ``progress`` is listed in :attr:`active_downloads` for as long as this
        runs. A caller that already listed it through :meth:`reserve_download`
        keeps that entry; one that did not gets it added here, unless another
        download of the same model is already listed, which is left alone.
        """

        checked_id = validate_model_id(model_id)
        progress = progress or DownloadProgress()
        try:
            with self._downloads_lock:
                self.active_downloads.setdefault(checked_id, progress)
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=checked_id,
                token=hf_token.strip() if hf_token and hf_token.strip() else None,
                tqdm_class=progress.bar_class(),
            )
        finally:
            self.release_download(checked_id, progress)
        return Path(path)

    def reserve_download(self, model_id: str) -> tuple[DownloadProgress, bool]:
        """Claim ``model_id`` for a new download, or point at the one running.

        Returns ``(progress, reserved)``. When ``reserved`` is true the caller
        owns the download: it must pass ``progress`` to :meth:`download`, whose
        ``finally`` removes the entry. When false, another caller is fetching
        the model and ``progress`` is theirs to watch.

        The lookup and the reservation are one atomic step. Checking
        :attr:`active_downloads` and then starting a worker is not: the worker
        registers itself only once it reaches :meth:`download`, and two
        handlers (say **Download** and **Download and load**) clicked together
        would both find the table empty in that gap and fetch the same files
        twice.
        """

        checked_id = validate_model_id(model_id)
        with self._downloads_lock:
            running = self.active_downloads.get(checked_id)
            if running is not None:
                return running, False
            progress = DownloadProgress()
            self.active_downloads[checked_id] = progress
            return progress, True

    def release_download(self, model_id: str, progress: DownloadProgress) -> None:
        """Remove ``progress`` only when it still owns ``model_id``'s entry."""

        checked_id = validate_model_id(model_id)
        with self._downloads_lock:
            if self.active_downloads.get(checked_id) is progress:
                del self.active_downloads[checked_id]

    def find_cached(self, model_id: str) -> Path:
        """The complete local snapshot of ``model_id``, without going online.

        Raises ``huggingface_hub.errors.IncompleteSnapshotError`` when the
        snapshot folder exists but files are missing from it, which is what an
        interrupted or still-running download leaves behind.
        """

        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(repo_id=validate_model_id(model_id), local_files_only=True)
        )

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
            self.load_count += 1
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

    def _response_prefix_ids(self, text: str, *, close_reasoning: bool) -> list[int]:
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
        encoded = self.tokenizer(raw, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        token_ids = [int(value) for value in encoded]
        if not token_ids:
            raise ValueError("The assistant prefill did not produce any tokens.")
        decoded = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != raw:
            raise ValueError(
                "The assistant prefill cannot be represented exactly by this tokenizer."
            )
        return token_ids

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
        collect_from: int = 0,
        sample: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Run the model over ``token_ids`` a chunk at a time.

        Returns the per-token metrics, the key-value cache, and the log
        probabilities that predict whatever comes after the sequence. Every
        token except the first is measured against the distribution the model
        held one step earlier, so the same pass that warms the cache also
        explains the prompt.

        Tokens before ``collect_from`` get no metric at all, which is how a
        prompt the reader chose not to measure stays out of the results while
        the response tokens that follow it are still described. Tokens from
        there up to ``score_from`` are recorded but left unscored.

        ``sample`` turns raw log probabilities into the distribution the
        sampler would have drawn from. It is applied to ``"response"`` tokens
        only, so a response prefix that is replayed rather than sampled still
        reports the sampling probability and shift it would have had.
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
                    else normalize_log_probabilities(
                        logits[index - start - 1].detach().float().cpu().numpy()
                    )
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
        forced_ids: Sequence[int] = (),
        answer_prefill: str = "",
        literal_prefill_tokens: int = 0,
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
        later branch replay.
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
        try:
            yield from self._generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                analyze_prompt=analyze_prompt,
                forced_ids=forced_ids,
                answer_prefill=answer_prefill,
                literal_prefill_tokens=literal_prefill_tokens,
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
        forced_ids: Sequence[int] = (),
        answer_prefill: str = "",
        literal_prefill_tokens: int = 0,
    ) -> Iterator[GenerationUpdate]:
        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")

            assert self.model is not None
            assert self.tokenizer is not None
            model = self.model
            tokenizer = self.tokenizer
            # Read here, under the lock, alongside the weights: this is the
            # only place the two are guaranteed to agree, which is what makes
            # the stamp on each update worth trusting.
            model_id = self.model_id
            load_id = self.load_id
            device = next(model.parameters()).device

            prompt_ids, reasoning_prefilled = self._prompt_token_ids(messages)
            stop_ids = self._stop_token_ids()

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
            else:
                literal_prefill_tokens = max(
                    0, min(int(literal_prefill_tokens), len(forced))
                )

            # A sampled stop token replayed by a branch still ends the old
            # response where it originally ended. A stop token the reader
            # typed literally into an assistant prefill is ordinary prefix
            # content instead: keep it visible and continue after it.
            for index, token_id in enumerate(forced):
                if token_id in stop_ids and index >= literal_prefill_tokens:
                    forced = forced[: index + 1]
                    break

            def sample(log_probs: np.ndarray) -> np.ndarray:
                return sampling_probabilities(
                    log_probs,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )

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
            for metric in metrics[:literal_prefill_tokens]:
                metric["literal_prefill"] = True
            prompt_note = ""
            if analyze_prompt and score_from > 1:
                prompt_note = (
                    f"Only the most recent {PROMPT_SCORE_LIMIT:,} of "
                    f"{len(prompt_ids):,} prompt tokens were scored."
                )

            rng = np.random.default_rng(int(seed))
            decoder = IncrementalDecoder(tokenizer, self._hidden_token_ids())
            literal_prefill_text = ""
            for index, token_id in enumerate(forced):
                decoder.push(
                    token_id, force_visible=index < literal_prefill_tokens
                )
                if index + 1 == literal_prefill_tokens:
                    # A branch can stop inside a byte-level token sequence for
                    # one character. The replacement-character suffix will be
                    # rewritten when the next token arrives, so it cannot be a
                    # durable prefix for the application's literal-tag guard.
                    literal_prefill_text = decoder.stable_text
            limit = len(forced) + int(max_new_tokens)
            pending_tokens = 0
            last_yield = time.monotonic()

            if forced:
                yield GenerationUpdate(
                    text=decoder.text,
                    metrics=metrics,
                    prompt_metrics=prompt_metrics,
                    prompt_note=prompt_note,
                    reasoning_prefilled=reasoning_prefilled,
                    forced_prefix_tokens=len(forced),
                    literal_prefill_text=literal_prefill_text,
                    prompt_ids=tuple(prompt_ids),
                    model_id=model_id,
                    load_id=load_id,
                )
                if (
                    forced[-1] in stop_ids
                    and len(forced) > literal_prefill_tokens
                ):
                    return

            for position in range(len(forced) + 1, limit + 1):
                assert raw_log_probs is not None
                sampled_probs = sample(raw_log_probs)

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
                        forced_prefix_tokens=len(forced),
                        literal_prefill_text=literal_prefill_text,
                        prompt_ids=tuple(prompt_ids),
                        model_id=model_id,
                        load_id=load_id,
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

    @contextlib.contextmanager
    def _eager_attention(self):
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

    def _final_norm(self):
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

    def _read_head(self, vector):
        """Turn a normed residual vector into logits the way the model does.

        Some causal-LM heads post-process the unembedding: Gemma 2 and 3
        soft-cap logits with ``tanh``, Granite divides by ``logits_scaling``,
        Cohere multiplies by ``logit_scale``. An intermediate reading that
        skipped them would describe a distribution the model never emits, so
        they are applied here. :meth:`inspect` checks the result against the
        model's own output for the final layer, which catches a transform
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

    def _lens_row(self, layer: int, logits, token_id: int) -> dict:
        log_probs = normalize_log_probabilities(
            logits.detach().float().cpu().numpy()
        )
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

    def inspect(
        self,
        token_ids: Sequence[int],
        index: int,
        *,
        context_count: int = 0,
        load_id: str | None = None,
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

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before inspecting a token.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            ids = [int(value) for value in token_ids]
            if not 1 <= index < len(ids):
                raise ValueError(
                    "Nothing came before this token, so the model never predicted it."
                )

            assert self.model is not None
            model = self.model
            device = next(model.parameters()).device
            token_id = ids[index]

            past_key_values = None
            if index > 1:
                # Collect nothing: only the cache is wanted.
                _, past_key_values, _ = self._prefill(
                    ids[: index - 1],
                    segments=[""] * (index - 1),
                    positions=list(range(index - 1)),
                    score_from=index,
                    collect_from=index,
                )

            with self._eager_attention():
                outputs = model(
                    input_ids=torch.tensor(
                        [[ids[index - 1]]], dtype=torch.long, device=device
                    ),
                    attention_mask=torch.ones((1, index), dtype=torch.long, device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            hidden_states = tuple(outputs.hidden_states or ())
            final_logits = outputs.logits[0, -1]
            norm = self._final_norm()
            layers: list[dict] = []
            # The last hidden state is what the model's own head reads, so its
            # row is the real output; the earlier ones are read through the
            # final norm as though the stack had ended there. Without the norm
            # those readings would be off by a rescaling the head never sees,
            # so a model whose norm cannot be found shows its output alone
            # rather than intermediate rows that look right and are not. The
            # same goes for a head that post-processes its logits in a way
            # _read_head() does not replicate: reading the final hidden state
            # (already normed) through it must reproduce the model's output,
            # or the intermediate rows are not trustworthy either.
            readable = norm is not None and bool(hidden_states)
            if readable:
                replayed = self._read_head(hidden_states[-1][0, -1]).detach().float()
                readable = torch.allclose(
                    replayed, final_logits.detach().float(), rtol=1e-2, atol=1e-2
                )
            if readable:
                for layer, state in enumerate(hidden_states[:-1]):
                    vector = norm(state[0, -1].unsqueeze(0)).squeeze(0)
                    layers.append(
                        self._lens_row(layer, self._read_head(vector), token_id)
                    )
            layers.append(
                self._lens_row(max(len(hidden_states) - 1, 0), final_logits, token_id)
            )

            decided_at: int | None = None
            for row in reversed(layers):
                if row["rank"] != 1:
                    break
                decided_at = row["layer"]

            attention: list[list[float]] = []
            weights = tuple(outputs.attentions or ())
            if weights and all(layer is not None for layer in weights):
                for layer in weights:
                    row = layer[0, :, -1, :].detach().float().mean(dim=0).cpu().tolist()
                    # A sliding-window layer keeps only its most recent keys,
                    # so a short row describes the end of the sequence. Align
                    # it on the right; the keys the layer could not see get a
                    # weight of zero, which is what it gave them.
                    row = row[-index:]
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
            del outputs, past_key_values
            return TokenInsight(
                index=index,
                token_id=token_id,
                token_text=self._decode_token(token_id) or self._token_fallback(token_id),
                layers=layers,
                tokens=tokens,
                attention=attention,
                decided_at=decided_at,
            )
