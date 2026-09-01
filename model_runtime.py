"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import gc
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from token_metrics import (
    TokenMetric,
    build_metric,
    normalize_log_probabilities,
    sampling_probabilities,
)


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


@dataclass(frozen=True)
class GenerationUpdate:
    text: str
    metrics: list[dict]


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

    def _prompt_inputs(self, messages: list[dict]):
        import torch

        assert self.tokenizer is not None
        assert self.model is not None
        tokenizer = self.tokenizer

        if tokenizer.chat_template:
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
        return {
            name: tensor.to(device)
            for name, tensor in inputs.items()
            if torch.is_tensor(tensor)
        }

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
            inputs = self._prompt_inputs(messages)
            attention_mask = inputs.get("attention_mask")
            rng = np.random.default_rng(int(seed))
            generated_ids: list[int] = []
            metrics: list[dict] = []
            stop_ids = self._stop_token_ids()
            past_key_values = None
            current_inputs = inputs

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

                generated_ids.append(token_id)
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

                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                yield GenerationUpdate(text=text, metrics=list(metrics))

                if token_id in stop_ids:
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
