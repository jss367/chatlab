"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import gc
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conversation import THINK_OPEN
from token_metrics import (
    TokenMetric,
    build_metric,
    normalize_log_probabilities,
    sampling_probabilities,
)


MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)

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


def validate_model_id(model_id: str) -> str:
    cleaned = model_id.strip()
    if not MODEL_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Enter a Hugging Face model ID in the form organization/model-name."
        )
    return cleaned


@dataclass(frozen=True)
class GenerationUpdate:
    text: str
    metrics: list[dict]
    """Live list owned by the generator. Copy it before storing it anywhere."""

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

    def _prompt_inputs(self, messages: list[dict]) -> tuple[dict, bool]:
        """Tokenize the prompt, reporting whether it prefills ``<think>``.

        Reasoning templates such as OLMo Think end the generation prompt with
        the opening marker, so the model resumes inside the block and never
        emits an opener. The flag rides along to the caller because only the
        prompt can reveal it.
        """

        import torch

        assert self.tokenizer is not None
        assert self.model is not None
        tokenizer = self.tokenizer
        prefilled = False

        if tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prefilled = isinstance(rendered, str) and rendered.rstrip().endswith(
                THINK_OPEN
            )
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            transcript = "\n".join(
                f"{message['role'].title()}: {message['content']}"
                for message in messages
            )
            inputs = tokenizer(f"{transcript}\nAssistant:", return_tensors="pt")

        device = next(self.model.parameters()).device
        tensors = {
            name: tensor.to(device)
            for name, tensor in inputs.items()
            if torch.is_tensor(tensor)
        }
        return tensors, prefilled

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

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
    ) -> Iterator[GenerationUpdate]:
        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")

            assert self.model is not None
            assert self.tokenizer is not None
            model = self.model
            tokenizer = self.tokenizer
            inputs, reasoning_prefilled = self._prompt_inputs(messages)
            attention_mask = inputs.get("attention_mask")
            rng = np.random.default_rng(int(seed))
            metrics: list[dict] = []
            stop_ids = self._stop_token_ids()
            decoder = IncrementalDecoder(tokenizer, self._hidden_token_ids())
            past_key_values = None
            current_inputs = inputs
            pending_tokens = 0
            last_yield = time.monotonic()

            for position in range(1, int(max_new_tokens) + 1):
                outputs = model(
                    **current_inputs, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = outputs.past_key_values
                raw_logits = outputs.logits[0, -1].float().cpu().numpy()
                raw_log_probs = normalize_log_probabilities(raw_logits)
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
                token_text = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                fallback = tokenizer.convert_ids_to_tokens(token_id) or str(token_id)

                metric: TokenMetric = build_metric(
                    position=position,
                    token_id=token_id,
                    token_text=token_text,
                    fallback_text=fallback,
                    raw_log_probabilities=raw_log_probs,
                    sampled_probabilities=sampled_probs,
                    decode_token=lambda candidate_id: (
                        tokenizer.decode(
                            [candidate_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        or (
                            tokenizer.convert_ids_to_tokens(candidate_id)
                            or str(candidate_id)
                        )
                    ),
                )
                metrics.append(metric.to_dict())

                stopping = token_id in stop_ids or position == int(max_new_tokens)
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
                        reasoning_prefilled=reasoning_prefilled,
                    )

                if stopping:
                    break

                device = next(model.parameters()).device
                next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                current_inputs = {"input_ids": next_token}
                if attention_mask is not None:
                    current_inputs["attention_mask"] = attention_mask
