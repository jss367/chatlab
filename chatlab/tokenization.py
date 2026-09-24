"""Turning text into token IDs and back, where the obvious way gets it wrong.

Scoring a passage after a context needs the token where the passage begins,
and tokenizers merge across that seam, add special tokens around it, or
drop the space before it. Streaming a response needs each token's text as
it arrives, and a byte-level tokenizer can split one character over several
tokens. Both are handled here, along with the longest sequence a model's
position table can take.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from chatlab import settings


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

    ``decoded_prefix_end`` retains the best boundary found while inspecting a
    slow tokenizer's decoded prefixes. It can locate a usable continuation
    even when decoding that continuation alone cannot verify the scoring seam.
    """

    context_ids: list[int]
    text_ids: list[int]
    seam_verified: bool = True
    chat_template_missing: bool = False
    decoded_prefix_end: int | None = None


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

    # An mlx-lm model keeps the same fields on ``args``, a dataclass built
    # from the same config.json.
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(model, "args", None)
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


def application_prefill_limit() -> int:
    """The reader's cap on a generation prefix, whatever the model allows."""

    return settings.current().prefill_token_limit


def generation_prefill_token_limit(model) -> int:
    """The most prompt-plus-replayed tokens accepted by one generation.

    A learned positional table is a hard correctness limit. The reader's cap
    is a separate application guard: typed branches can otherwise turn an
    unrestricted textbox into an arbitrarily large scored prefill while the
    model lock is held, and one long conversation's key-value cache can
    outgrow the machine.
    """

    limit = application_prefill_limit()
    window = model_position_limit(model)
    return limit if window is None else min(limit, window)


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
) -> tuple[int | None, int | None]:
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

    The first value is returned only when the two halves decode back to the
    passage verbatim. The second retains the decoded-prefix boundary even when
    it could not be verified. A ``decode`` that does not round trip — a
    byte-level merge cut mid-character, a normalizer that rewrites whitespace,
    a SentencePiece model that eats a leading space — would otherwise move the
    seam by a token and score part of the context, so those cases say ``None``
    and leave the placement to :func:`_guess_seam`.
    """

    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None, None

    def spoken(start: int, end: int) -> str | None:
        try:
            return decode(
                ids[start:end],
                skip_special_tokens=add_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except (NotImplementedError, TypeError, ValueError):
            return None

    split = _decoded_prefix_end(spoken, stop, context)

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
            return None, split
        if not reaches.startswith(context):
            return None, split

    head, tail = spoken(0, split), spoken(split, stop)
    if head is None or tail is None or head + tail != context + text:
        return None, split

    return split, split


def _decoded_prefix_end(
    spoken: Callable[[int, int], str | None], stop: int, context: str
) -> int:
    """Find the longest decoded prefix that still lies within ``context``.

    The result remains useful as a suffix candidate even when decoding the two
    halves independently cannot verify a seam. Slow WordPiece tokenizers, for
    example, expose a leading continuation piece literally when it is decoded
    alone, while that same piece joins correctly after the kept token IDs.
    """

    def within_context(end: int) -> bool:
        prefix = spoken(0, end)
        if prefix is None:
            return False
        if context.startswith(prefix):
            return True

        # A slow byte-fallback tokenizer can expose an incomplete trailing
        # UTF-8 sequence as one or more replacement characters. That prefix
        # becomes valid again when subsequent byte tokens complete the
        # character, so treating the intermediate spelling as outside the
        # context would make the predicate non-monotonic and strand this
        # bisection before the real boundary. Regard only *trailing* decoder
        # replacements as provisional while the decoded text is still short
        # of the context. Once it equals the context, any trailing replacement
        # can be the first bytes of the replacement text and must not advance
        # the boundary. The complete candidate is still required to reproduce
        # the exact expected text by the caller.
        repaired = prefix.rstrip(REPLACEMENT_CHARACTER)
        if repaired == prefix or not context.startswith(repaired):
            return False
        remaining = context[len(repaired) :].lstrip(REPLACEMENT_CHARACTER)
        return bool(remaining)

    low, high = 0, stop
    while low < high:
        middle = (low + high + 1) // 2
        if within_context(middle):
            low = middle
        else:
            high = middle - 1
    return low


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

    split, decoded_prefix_end = _seam_by_decoding(
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
        ids[:split],
        ids[split:stop],
        seam_verified=verified or not context,
        decoded_prefix_end=decoded_prefix_end,
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
