"""Read imported Jacobian lenses without fitting or changing the loaded model.

File interoperability and readout convention follow anthropics/jacobian-lens:
https://github.com/anthropics/jacobian-lens (Apache-2.0).
Source layer zero is the output of decoder block zero, before the final norm.

A lens is a set of ``d_model × d_model`` matrices, one per fitted decoder
block. Reading a residual state through one transports it into the final
block's basis, and the model's own norm and head then turn it into
vocabulary scores. Both backends are read the same way: a Transformers model
through forward hooks on its blocks, an MLX model through the layer recorder
its logit lens already uses. Everything the lens computes stays on the CPU
in float32 except the norm and head, which run where the model's weights are.
"""

from __future__ import annotations

import filecmp
import json
import logging
import re
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import settings


logger = logging.getLogger(__name__)


# Decoder families whose blocks form one list under the model with one final
# norm after it, which is the layout the readout walks, and for which fitted
# lenses are published. The layout check and the final-block replay decide
# for a particular checkpoint; a family that is missing here is refused with
# its name rather than read wrongly.
SUPPORTED_MODELS = {
    "llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "gemma2", "gemma3_text",
    "olmo2", "olmo3", "glm4", "phi3", "granite", "cohere2", "smollm3",
}
MAX_FILE_BYTES = 2 * 1024**3
# How many positions one slice covers, ending at the selected token.
SLICE_POSITIONS = 128
TOP_CANDIDATES = 5

REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


@dataclass
class Layout:
    """Where a loaded model keeps what the lens reads."""

    backend: str
    blocks: list
    norm: object
    width: int
    model_type: str


def _check_type(model_type) -> None:
    if model_type not in SUPPORTED_MODELS:
        names = "Llama, Mistral, Qwen2, Qwen3, Gemma 2 and 3, OLMo 2 and 3, GLM-4, Phi-3, Granite, Cohere, and SmolLM3"
        raise ValueError(
            f"Jacobian inspection supports {names} text models; this model's type is "
            f"{model_type or 'unknown'}."
        )


def model_layout(engine) -> Layout:
    """Keep the support boundary explicit; other layouts need validation."""
    if getattr(engine, "backend", "torch") == "mlx":
        return _mlx_layout(engine)
    model = engine.model
    config = model.config
    _check_type(getattr(config, "model_type", None))
    if getattr(model, "is_quantized", False) or getattr(config, "quantization_config", None):
        raise ValueError("Load full-precision weights for Jacobian inspection; quantized Transformers models are not validated yet.")
    decoder = getattr(model, "model", None)
    blocks = getattr(decoder, "layers", None)
    norm = getattr(decoder, "norm", None)
    if blocks is None or norm is None or len(blocks) != config.num_hidden_layers:
        raise ValueError("This model's decoder layout is not supported by the Jacobian inspector.")
    return Layout("torch", list(blocks), norm, int(config.hidden_size), config.model_type)


def _mlx_layout(engine) -> Layout:
    config = getattr(engine, "config", None) or {}
    _check_type(config.get("model_type"))
    owner, attribute = engine._layer_stack()
    blocks = getattr(owner, attribute, None) if owner is not None and attribute else None
    norm = engine.final_norm()
    width = config.get("hidden_size")
    if (
        not isinstance(blocks, list) or norm is None or type(width) is not int
        or len(blocks) != config.get("num_hidden_layers")
    ):
        raise ValueError("This model's decoder layout is not supported by the Jacobian inspector.")
    return Layout("mlx", blocks, norm, width, config["model_type"])


def check_identity(layout: Layout, model_id: str, fitted_model_id: str) -> str:
    """The declared fitted model, once it is allowed to stand for the loaded one.

    A Transformers load is the checkpoint itself, so the declaration must
    name it exactly. An MLX conversion is another repository made from that
    checkpoint, and the conversion carries the source's name (``Qwen3-0.6B``
    inside ``mlx-community/Qwen3-0.6B-4bit``), which is the one link there is.
    """

    fitted = fitted_model_id.strip()
    if layout.backend == "torch":
        if fitted != model_id:
            raise ValueError("The fitted model ID must exactly match the loaded model ID.")
        return fitted
    name = fitted.rsplit("/", 1)[-1].lower()
    if not name or name not in model_id.lower():
        raise ValueError(
            "Enter the ID of the full-precision model this MLX conversion was made from; "
            "its name must appear in the loaded model's name."
        )
    return fitted


@dataclass
class FittedLens:
    matrices: dict
    width: int
    n_prompts: int
    name: str

    @classmethod
    def load(cls, path, engine, model_id: str, fitted_model_id: str):
        import torch

        layout = model_layout(engine)
        fitted = check_identity(layout, model_id, fitted_model_id)
        path = Path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError("Choose a saved lens.pt file no larger than 2 GiB.")
        # mmap keeps the whole set of dense matrices off the accelerator and
        # avoids eagerly copying the file into RAM. Never load arbitrary pickle.
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(data, dict) or not isinstance(data.get("J"), dict) or not data["J"]:
            raise ValueError("Expected a saved Jacobian lens with a nonempty J dictionary.")
        declared_model = data.get("model_id")
        if declared_model is not None and declared_model != fitted:
            raise ValueError("The lens file names a different model.")
        revision = data.get("model_revision")
        if layout.backend == "torch" and revision is not None:
            if revision != getattr(engine.model.config, "_commit_hash", None):
                raise ValueError("The lens file's model revision does not match the loaded checkpoint.")
        width = data.get("d_model")
        if type(width) is not int or width != layout.width:
            raise ValueError("The lens residual width does not match the loaded model.")
        count = data.get("n_prompts")
        if type(count) is not int or count <= 0:
            raise ValueError("The lens must record a positive n_prompts.")
        matrices = data["J"]
        if any(type(layer) is not int or not 0 <= layer < len(layout.blocks) for layer in matrices):
            raise ValueError("The lens contains invalid decoder layer indices.")
        if data.get("source_layers") != sorted(matrices):
            raise ValueError("The lens source_layers must match its sorted matrix indices.")
        total = 0
        for matrix in matrices.values():
            if (
                not isinstance(matrix, torch.Tensor)
                or matrix.layout != torch.strided
                or matrix.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or tuple(matrix.shape) != (width, width)
            ):
                raise ValueError("Every lens matrix must be a dense floating-point d_model × d_model tensor.")
            total += matrix.numel() * matrix.element_size()
            if total > MAX_FILE_BYTES:
                raise ValueError("The lens matrices exceed the 2 GiB limit.")
            if not torch.isfinite(matrix).all():
                raise ValueError("The lens contains non-finite matrix entries.")
        return cls(matrices, width, count, path.name)

    def wanted(self, layout: Layout) -> list[int]:
        """The fitted blocks plus the last one, whose readout is the check."""
        return sorted(set(self.matrices) | {len(layout.blocks) - 1})

    @contextmanager
    def capture(self, engine):
        """Record each wanted block's output at every position fed in the body.

        Yields a dict that holds, once the body has run, one float32 CPU
        tensor of shape ``[fed positions, d_model]`` per wanted block.
        """
        layout = model_layout(engine)
        states: dict = {}
        if layout.backend == "mlx":
            with _capture_mlx(engine, self.wanted(layout), states):
                yield states
            return
        handles = []

        def hook_for(layer):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                states[layer] = value[0].detach().float().cpu().clone()
            return hook

        try:
            for layer in self.wanted(layout):
                handles.append(layout.blocks[layer].register_forward_hook(hook_for(layer)))
            yield states
        finally:
            for handle in handles:
                handle.remove()

    # Kept under its old name for callers that read one position.
    record = capture

    def read(self, engine, states, actual_logits, decode, pinned_id=None):
        """Score every fed position through each fitted block.

        Returns ``(rows, cells)``: ``rows`` describes the last fed position
        the way the table shows it, the top candidates per block and the
        pinned token's rank; ``cells`` holds, per block, the top token at
        every fed position, with the pinned token's rank beside it.
        """
        import torch

        layout = model_layout(engine)
        last = len(layout.blocks) - 1
        if set(states) != set(self.wanted(layout)):
            raise ValueError("The model did not return all requested decoder activations.")
        actual = torch.as_tensor(np.asarray(actual_logits), dtype=torch.float32)
        replayed = _unembed(engine, layout, states[last][-1:])[0]
        if not torch.allclose(replayed, actual, rtol=1e-2, atol=1e-2):
            raise ValueError("The final-layer readout does not reproduce this model's output; Jacobian results were withheld.")
        if pinned_id is not None and not 0 <= pinned_id < actual.numel():
            raise ValueError("The pinned token is outside this model's output vocabulary.")
        rows, cells = [], []
        for layer, matrix in sorted(self.matrices.items()):
            # The reference transports in float32, then uses the model's norm
            # and head dtype. The transport stays on the CPU, so the lens
            # matrices never take accelerator memory.
            scores = _unembed(engine, layout, states[layer] @ matrix.float().T)
            if not torch.isfinite(scores).all():
                raise ValueError("The Jacobian readout produced non-finite scores.")
            final = scores[-1]
            values, ids = final.topk(min(TOP_CANDIDATES, final.numel()))
            row = {
                "layer": layer,
                "candidates": [
                    {"token_id": int(token), "text": decode(int(token)), "score": float(score)}
                    for token, score in zip(ids, values)
                ],
            }
            best_scores, best_ids = scores.max(dim=1)
            column = [
                {"token_id": int(token), "text": decode(int(token)), "score": float(score)}
                for token, score in zip(best_ids, best_scores)
            ]
            if pinned_id is not None:
                pinned_scores = scores[:, pinned_id]
                ranks = (scores > pinned_scores[:, None]).sum(dim=1) + 1
                row["rank"] = int(ranks[-1])
                row["score"] = float(pinned_scores[-1])
                for cell, rank, score in zip(column, ranks, pinned_scores):
                    cell["pinned_rank"] = int(rank)
                    cell["pinned_score"] = float(score)
            rows.append(row)
            cells.append({"layer": layer, "cells": column})
        return rows, cells


@contextmanager
def _capture_mlx(engine, wanted, states):
    """Fill ``states`` from the MLX layer recorder once the body has run."""
    import mlx.core as mx
    import torch

    from mlx_runtime import _Recorder

    recorder = _Recorder()
    with engine._recording(recorder):
        yield
    hidden = recorder.hidden
    # The recorder keeps the embedding first, then each block's output.
    for layer in wanted:
        if layer + 1 < len(hidden):
            array = hidden[layer + 1][0].astype(mx.float32)
            mx.eval(array)
            states[layer] = torch.from_numpy(np.array(array))


def _unembed(engine, layout: Layout, vectors):
    """Norm and head, the model's own, over rows of residual vectors.

    ``vectors`` is a float32 CPU tensor ``[n, d_model]``; the answer is a
    float32 CPU tensor ``[n, vocabulary]``.
    """
    import torch

    if layout.backend == "mlx":
        import mlx.core as mx

        array = mx.array(vectors.numpy())[None]
        logits = engine.read_head(layout.norm(array))[0].astype(mx.float32)
        mx.eval(logits)
        return torch.from_numpy(np.array(logits))
    norm_weight = next(layout.norm.parameters())
    head_weight = engine.model.get_output_embeddings().weight
    normed = layout.norm(vectors.to(device=norm_weight.device, dtype=norm_weight.dtype))
    return engine.read_head(normed.to(device=head_weight.device, dtype=head_weight.dtype)).float().cpu()


@dataclass
class JacobianInsight:
    payload: dict

    def to_dict(self):
        return self.payload


# ------------------------------------------------------------ lenses on disk


def store_path() -> Path:
    """Where the lens each model last used is written down, beside the settings."""
    return settings.settings_path().with_name("jacobian_lenses.json")


def lens_directory() -> Path:
    """Where imported lenses are kept, outside the model cache."""
    return store_path().with_name("lenses")


_STORE_LOCK = threading.Lock()


def _read_store() -> dict:
    try:
        data = json.loads(store_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def remembered(model_id: str) -> dict | None:
    """The lens last imported for ``model_id``, or ``None``."""
    record = _read_store().get(model_id)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    return record


def remember(model_id: str, record: dict) -> None:
    """Write down which lens ``model_id`` uses, so a reload finds it again."""
    with _STORE_LOCK:
        data = _read_store()
        data[model_id] = {
            key: record[key]
            for key in ("path", "fitted_model_id", "repository", "filename")
            if isinstance(record.get(key), str)
        }
        path = store_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_suffix(".tmp")
            staged.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
            staged.replace(path)
        except OSError:
            logger.warning("Could not write down the Jacobian lens for %s", model_id, exc_info=True)


def download(repository: str, filename: str) -> Path:
    """Fetch one lens file from a Hugging Face repository into the lens directory.

    The file lands under a folder named for the repository, never in the
    model cache, so a lens repository is not mistaken for a half-downloaded
    model. The size is checked before anything is transferred.
    """
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import (
        EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError,
    )

    repository = (repository or "").strip()
    filename = (filename or "").strip().strip("/")
    if not REPOSITORY_ID.match(repository):
        raise ValueError("Enter the lens repository as organization/name.")
    parts = filename.split("/")
    if not filename or not filename.endswith(".pt") or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Enter the path of a .pt file inside the repository.")
    try:
        metadata = get_hf_file_metadata(hf_hub_url(repository, filename))
        if metadata.size is not None and metadata.size > MAX_FILE_BYTES:
            raise ValueError("That lens file is larger than the 2 GiB limit.")
        owner, name = repository.split("/")
        target = lens_directory() / owner / name
        target.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download(repository, filename, local_dir=str(target)))
    except GatedRepoError:
        raise ValueError("That repository requires access. Accept its terms on Hugging Face and log in with the huggingface-cli first.") from None
    except (RepositoryNotFoundError, EntryNotFoundError):
        raise ValueError("That repository or file was not found on Hugging Face.") from None
    except HfHubHTTPError as error:
        raise ValueError(f"Hugging Face refused the download: {error}") from None
    except OSError as error:
        raise ValueError(f"The lens could not be downloaded: {error}") from None


def keep(path) -> Path:
    """Copy a lens chosen from disk under ``lens_directory()/uploads``; the kept path.

    A browser upload lands in Gradio's cache, which does not outlive the
    session, so the copy is what gets remembered. A file already inside the
    lens directory stays where it is; a kept file with the same name and
    content is reused, and a different one takes a numbered name.
    """
    source = Path(path)
    directory = lens_directory()
    if source.resolve().is_relative_to(directory.resolve()):
        return source
    uploads = directory / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / source.name
    counter = 1
    while target.exists() and not filecmp.cmp(source, target, shallow=False):
        counter += 1
        target = uploads / f"{source.stem}-{counter}{source.suffix}"
    if not target.exists():
        shutil.copy2(source, target)
    return target
