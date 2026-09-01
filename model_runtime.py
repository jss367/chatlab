"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import gc
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

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


class SplitPassage(NamedTuple):
    """A passage's two token runs, and whether the seam between them is sure.

    ``seam_verified`` is false only where the ids were not taken from one
    joint encoding: the last-resort path that encodes each half alone, for a
    tokenizer that reports no offsets and whose ``decode`` cannot say where
    the seam fell. The scores that come back are then off by at most the one
    token that spans the seam, and the caller is expected to say so rather
    than present them as exact.

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


def _split_by_decoding(
    tokenizer, context: str, text: str, *, add_special_tokens: bool = True
) -> tuple[list[int], list[int]] | None:
    """Locate the seam in one joint encoding by decoding it back to text.

    A tokenizer without offsets can still be asked what a run of ids says, so
    the passage is still encoded once and the seam is found afterwards: the
    context keeps the longest run of leading tokens that decodes to a prefix
    of ``context``, and the token after it — the one that straddles the seam,
    if any — starts the scored text, exactly as it does when offsets are
    available. Trailing special tokens are dropped for the same reason they
    are there.

    ``add_special_tokens=False`` says the passage already spells out every
    special token it wants — a rendered chat template does — so decoding has
    to keep them to round trip, and nothing was appended for the trailing
    sweep to drop.

    The split is returned only when the two halves decode back to the passage
    verbatim. A ``decode`` that does not round trip — a byte-level merge cut
    mid-character, a normalizer that rewrites whitespace, a SentencePiece
    model that eats a leading space — would otherwise move the seam by a
    token and score part of the context, so those cases say so with ``None``
    instead of guessing.
    """

    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None

    try:
        ids = [
            int(value)
            for value in tokenizer(
                context + text, add_special_tokens=add_special_tokens
            ).input_ids
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not ids:
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

    specials = set(getattr(tokenizer, "all_special_ids", None) or ())
    stop = len(ids)
    if add_special_tokens:
        while stop and ids[stop - 1] in specials:
            stop -= 1

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

    return ids[:split], ids[split:stop]


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

    Slow tokenizers cannot report offsets, so the seam is instead located by
    decoding a growing prefix of the one joint encoding, and only the halves
    that decode back to the passage verbatim are used. Encoding each half on
    its own is the last resort, for a tokenizer whose ``decode`` cannot say
    where the seam fell. Those ids can differ from the joint encoding by the
    one token that spans the seam, so that path clears ``seam_verified`` and
    the caller reports the numbers as approximate rather than dropping a
    feature the reader asked for.
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

    halves = _split_by_decoding(
        tokenizer, context, text, add_special_tokens=add_special_tokens
    )
    if halves is not None:
        return SplitPassage(*halves)

    # Last resort. With no context there is no seam for a merge to cross —
    # the joint encoding is the text's own, plus whatever specials the
    # tokenizer prepends — so those ids are exact. A context with characters
    # in it is the case that cannot be checked, and it says so.
    return SplitPassage(
        [
            int(value)
            for value in tokenizer(
                context, add_special_tokens=add_special_tokens
            ).input_ids
        ],
        [int(value) for value in tokenizer(text, add_special_tokens=False).input_ids],
        seam_verified=not context,
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
    stripping it would report ranks for a passage the reader never wrote. The
    template path is the one exception: a message of pure whitespace is not a
    turn worth wrapping, so it falls through to the plain path, where the
    whitespace is still scored as the text's leading context.

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
    wants_template = bool(context.strip()) and use_chat_template
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
    prompt_metrics: list[dict]
    prompt_note: str = ""


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

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

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

    def _prompt_token_ids(self, messages: list[dict]) -> list[int]:
        """Token ids for a chat prompt, including the generation prompt."""

        assert self.tokenizer is not None
        tokenizer = self.tokenizer

        if tokenizer.chat_template:
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
        return [int(value) for value in encoded]

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
        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")

            assert self.model is not None
            assert self.tokenizer is not None
            model = self.model
            tokenizer = self.tokenizer
            device = next(model.parameters()).device

            prompt_ids = self._prompt_token_ids(messages)
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
            generated_ids: list[int] = []
            metrics: list[dict] = []
            stop_ids = self._stop_token_ids()
            limit = int(max_new_tokens)

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

                generated_ids.append(token_id)
                metrics.append(
                    self._describe_token(
                        position=position,
                        token_id=token_id,
                        raw_log_probabilities=raw_log_probs,
                        sampled_probabilities=sampled_probs,
                        segment="response",
                    )
                )
                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                yield GenerationUpdate(
                    text=text,
                    metrics=list(metrics),
                    prompt_metrics=prompt_metrics,
                    prompt_note=prompt_note,
                )

                if token_id in stop_ids or position == limit:
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
