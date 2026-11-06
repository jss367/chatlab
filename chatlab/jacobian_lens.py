"""Read imported Jacobian lenses without fitting or changing the loaded model.

File interoperability and readout convention follow anthropics/jacobian-lens:
https://github.com/anthropics/jacobian-lens (Apache-2.0), and the J++ lens of
safety-research/jpp_lens (Ayonrinde & Lindsey, 2026, Apache-2.0).
Source layer zero is the output of decoder block zero, before the final norm.

A lens is a set of ``d_model × d_model`` matrices, one per fitted decoder
block. Reading a residual state through one transports it into the final
block's basis, and the model's own norm and head then turn it into
vocabulary scores. The J-Lens, R-Lens and J++ Lens differ only in how those
matrices were fitted: the R-Lens takes each Jacobian through a layer-wise
relevance propagation backward pass, and the J++ Lens also fits one Jacobian
per activation cluster and weighs them into one. All three are read the same
way, except that a J++ readout leaves out the tokens with no letter or digit
(its Readout Filtering). Both backends are read the same way: a Transformers model
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chatlab import settings, steering
from chatlab.files import read_json_object


logger = logging.getLogger(__name__)


# Decoder families whose blocks form one list under the model with one final
# norm after it, which is the layout the readout walks, and for which fitted
# lenses are published. The layout check and the final-block replay decide
# for a particular checkpoint; a family that is missing here is refused with
# its name rather than read wrongly. Qwen3.5 and Gemma 4 checkpoints are
# image-text wrappers around such a stack; the wrapper's type and the text
# model's are both listed, since a text-only load carries the latter.
SUPPORTED_MODELS = {
    "llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "gemma2", "gemma3_text",
    "olmo2", "olmo3", "glm4", "phi3", "granite", "cohere2", "smollm3",
    "qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text",
}
# The published lenses for 27B and 31B models are about 3.4 GB.
MAX_FILE_BYTES = 8 * 1024**3
SIZE_LIMIT = f"{MAX_FILE_BYTES // 1024**3} GiB"
# How many positions one slice covers, ending at the selected token.
SLICE_POSITIONS = 128
TOP_CANDIDATES = 5

REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# What mlx-community appends to a source model's name: the width (``-4bit``,
# ``-8bit``, ``-bf16``) and the method or packaging (``-DWQ``, ``-mixed``).
CONVERSION_SUFFIX = re.compile(r"[-_](?:[1-9][-_]?bit|bf16|fp16|fp32|dwq|mlx|mixed)$", re.IGNORECASE)


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
        names = (
            "Llama, Mistral, Qwen2, Qwen3, Qwen3.5, Gemma 2, 3 and 4, OLMo 2 and 3, GLM-4, "
            "Phi-3, Granite, Cohere, and SmolLM3"
        )
        raise ValueError(
            f"Jacobian inspection supports {names} text models; this model's type is "
            f"{model_type or 'unknown'}."
        )


UNSUPPORTED_LAYOUT = "This model's decoder layout is not supported by the Jacobian inspector."


def model_layout(engine) -> Layout:
    """Keep the support boundary explicit; other layouts need validation.

    The blocks are the ones steering adds its vector to, so a steered
    readout is read from the same stack the vector went into. An image-text
    wrapper keeps the text model's depth and width in its text config.
    """
    if engine.backend == "mlx":
        return _mlx_layout(engine)
    model = engine.model
    config = model.config
    _check_type(getattr(config, "model_type", None))
    if getattr(model, "is_quantized", False) or getattr(config, "quantization_config", None):
        raise ValueError("Load full-precision weights for Jacobian inspection; quantized Transformers models are not validated yet.")
    if callable(getattr(config, "get_text_config", None)):
        config = config.get_text_config()
    try:
        blocks = steering.decoder_layers(model)
    except steering.SteeringError:
        raise ValueError(UNSUPPORTED_LAYOUT) from None
    norm = engine.final_norm()
    if norm is None or len(blocks) != getattr(config, "num_hidden_layers", None):
        raise ValueError(UNSUPPORTED_LAYOUT)
    return Layout("torch", list(blocks), norm, int(config.hidden_size), model.config.model_type)


def _mlx_layout(engine) -> Layout:
    from chatlab.mlx_runtime import text_settings

    config = getattr(engine, "config", None) or {}
    _check_type(config.get("model_type"))
    text = text_settings(config)
    owner, attribute = engine._layer_stack()
    blocks = getattr(owner, attribute, None) if owner is not None and attribute else None
    norm = engine.final_norm()
    width = text.get("hidden_size")
    if (
        not isinstance(blocks, list) or norm is None or type(width) is not int
        or len(blocks) != text.get("num_hidden_layers")
    ):
        raise ValueError(UNSUPPORTED_LAYOUT)
    return Layout("mlx", blocks, norm, width, config["model_type"])


def conversion_source_name(model_id: str) -> str:
    """The name of the model an MLX conversion was made from, as its own name tells it.

    ``mlx-community/Qwen3-0.6B-4bit`` was made from ``Qwen3-0.6B``; the
    conversion suffixes are peeled off the end one at a time, so
    ``Qwen3-4B-Instruct-4bit-DWQ`` gives ``Qwen3-4B-Instruct``, not ``Qwen3-4B``.
    """
    name = model_id.rsplit("/", 1)[-1]
    while True:
        stripped = CONVERSION_SUFFIX.sub("", name)
        if stripped == name:
            return name
        name = stripped


def check_identity(layout: Layout, model_id: str, fitted_model_id: str) -> str:
    """The declared fitted model, once it is allowed to stand for the loaded one.

    A Transformers load is the checkpoint itself, so the declaration must
    name it exactly. An MLX conversion is another repository made from that
    checkpoint, and the conversion is named for its source with the width
    appended (``Qwen3-0.6B`` in ``mlx-community/Qwen3-0.6B-4bit``), which is
    the one link there is; the names must agree once that suffix is removed,
    so a ``Qwen3-4B`` lens does not pass for ``Qwen3-4B-Instruct-4bit``.
    """

    fitted = fitted_model_id.strip()
    if layout.backend == "torch":
        if fitted != model_id:
            raise ValueError("The fitted model ID must exactly match the loaded model ID.")
        return fitted
    name = fitted.rsplit("/", 1)[-1].lower()
    if not name or conversion_source_name(model_id).lower() != name:
        raise ValueError(
            "Enter the ID of the full-precision model this MLX conversion was made from; "
            "its name must match the conversion's name once the quantization suffix is "
            "removed, as Qwen/Qwen3-0.6B does for mlx-community/Qwen3-0.6B-4bit."
        )
    return fitted


J_LENS, R_LENS, JPP_LENS = "J-Lens", "R-Lens", "J++ Lens"


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _first(*values):
    """The first of ``values`` that is not ``None``."""
    return next((value for value in values if value is not None), None)


def lens_kind(data: dict) -> str:
    """Which lens a saved file holds, as far as the file says.

    Neuronpedia's conversions name the method under provenance; the J++
    lenses released with the paper keep a stage-two fitting key in their
    config. A file that records an LRP backward pass and neither of those is
    an R-Lens, and anything else a J-Lens. A J++ lens written by the current
    jpp_lens code carries no mark of its own, so it reads as an R-Lens.
    """
    config = _dict(data.get("config"))
    provenance = _dict(data.get("provenance"))
    source_config = _dict(provenance.get("source_config"))
    if (
        provenance.get("method") == "jpp" or "num_s2_fitting_prompts" in config
        or isinstance(config.get("num_clusters"), int) and config["num_clusters"] > 0
    ):
        return JPP_LENS
    lrp = _first(config.get("lrp_mode"), provenance.get("lrp_mode"), source_config.get("lrp_mode"))
    if isinstance(lrp, str) and lrp != "none":
        return R_LENS
    # The released R-Lens files record their estimator as JSON under provenance.
    try:
        estimator = json.loads(provenance.get("config_json") or "{}")
    except (TypeError, ValueError):
        estimator = {}
    return R_LENS if _dict(estimator).get("estimator") == "relp" else J_LENS


def non_semantic_mask(tokenizer, vocabulary: int):
    """The J++ Lens's Readout Filtering: which output IDs decode to no letter or digit.

    A boolean tensor over ``vocabulary``, the head's width. Whitespace and
    punctuation tokens are marked, and so are head rows past the tokenizer,
    which decode to nothing. Special tokens decode with their names and are
    kept, as the reference keeps them.
    """
    import torch

    known = min(vocabulary, len(tokenizer))
    ids = [[token] for token in range(known)]
    texts = (
        tokenizer.batch_decode(ids) if hasattr(tokenizer, "batch_decode")
        else [tokenizer.decode(token) for token in ids]
    )
    mask = torch.ones(vocabulary, dtype=torch.bool)
    mask[:known] = torch.tensor([not any(char.isalnum() for char in text) for text in texts], dtype=torch.bool)
    return mask


@dataclass
class FittedLens:
    matrices: dict
    width: int
    n_prompts: int
    name: str
    kind: str = J_LENS
    # The Readout Filtering mask, made on the first readout that needs it.
    excluded: object = field(default=None, repr=False, compare=False)

    @property
    def filters_readout(self) -> bool:
        """Whether readouts leave out tokens with no letter or digit, as the J++ Lens does."""
        return self.kind == JPP_LENS

    @classmethod
    def load(cls, path, engine, model_id: str, fitted_model_id: str, model_revision: str | None):
        """Read and check the lens at ``path`` against the loaded model.

        Two file layouts are read. The reference one keeps the matrices under
        ``J`` beside ``d_model``, ``n_prompts`` and ``source_layers``. The
        jpp_lens one keeps them under ``parameters["jacobians"]`` beside
        ``source_layers``, with the rest in a ``config``.

        ``model_revision`` is the loaded checkpoint's, as
        :meth:`ModelManager.model_revision` reports it; a Transformers lens
        that records a revision is refused over any other, unknown included.
        """
        import torch

        layout = model_layout(engine)
        fitted = check_identity(layout, model_id, fitted_model_id)
        path = Path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError(f"Choose a saved lens.pt file no larger than {SIZE_LIMIT}.")
        # mmap keeps the whole set of dense matrices off the accelerator and
        # avoids eagerly copying the file into RAM. Never load arbitrary pickle.
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(data, dict):
            raise ValueError("Expected a saved Jacobian lens with a nonempty J dictionary.")
        config = _dict(data.get("config"))
        provenance = _dict(data.get("provenance"))
        if config.get("lens_type", "jacobian") != "jacobian":
            raise ValueError(f"That file holds a {config['lens_type']} lens, not a Jacobian lens.")
        matrices = _dict(data["parameters"]).get("jacobians") if "parameters" in data else data.get("J")
        if not isinstance(matrices, dict) or not matrices:
            raise ValueError("Expected a saved Jacobian lens with a nonempty J dictionary.")
        # Neuronpedia's fit_lens.py records the fitted model under provenance,
        # its J++ conversions under the source config there, and jpp_lens in
        # its own config.
        declared_model = _first(
            data.get("model_id"), provenance.get("model_id"), config.get("hf_model_name"),
            _dict(provenance.get("source_config")).get("hf_model_name"),
        )
        if declared_model is not None and declared_model != fitted:
            raise ValueError("The lens file names a different model.")
        revision = data.get("model_revision")
        if layout.backend == "torch" and revision is not None:
            if revision != model_revision:
                raise ValueError("The lens file's model revision does not match the loaded checkpoint.")
        width = _first(data.get("d_model"), config.get("d_model"))
        if type(width) is not int or width != layout.width:
            raise ValueError("The lens residual width does not match the loaded model.")
        count = _first(data.get("n_prompts"), config.get("num_prompts_trained_on"))
        if type(count) is not int or count <= 0:
            raise ValueError("The lens must record a positive n_prompts.")
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
                raise ValueError(f"The lens matrices exceed the {SIZE_LIMIT} limit.")
            if not torch.isfinite(matrix).all():
                raise ValueError("The lens contains non-finite matrix entries.")
        return cls(matrices, width, count, path.name, lens_kind(data))

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

    def read(self, engine, states, actual_logits, decode, pinned_id=None, excluded=None):
        """Score every fed position through each fitted block.

        Returns ``(rows, cells)``: ``rows`` describes the last fed position
        the way the table shows it, the top candidates per block and the
        pinned token's rank; ``cells`` holds, per block, the top token at
        every fed position, with the pinned token's rank beside it, and under
        ``top`` the IDs of the top candidates at every position.

        ``excluded`` is a boolean mask over the vocabulary, or ``None``. The
        tokens it marks are never candidates and are not counted in a pinned
        token's rank, though a pinned token it marks still gets one.
        """
        import torch

        layout = model_layout(engine)
        last = len(layout.blocks) - 1
        if set(states) != set(self.wanted(layout)):
            raise ValueError("The model did not return all requested decoder activations.")
        actual = torch.as_tensor(np.asarray(actual_logits), dtype=torch.float32)
        # The whole slice is replayed, as the model's own head read it: a
        # half-precision head over one row rounds differently from the same
        # head over many, by more than the tolerance.
        replayed = unembed(engine, layout, states[last])[-1]
        if not torch.allclose(replayed, actual, rtol=1e-2, atol=1e-2):
            raise ValueError("The final-layer readout does not reproduce this model's output; Jacobian results were withheld.")
        if pinned_id is not None and not 0 <= pinned_id < actual.numel():
            raise ValueError("The pinned token is outside this model's output vocabulary.")
        rows, cells = [], []
        for layer, matrix in sorted(self.matrices.items()):
            # The reference transports in float32, then uses the model's norm
            # and head dtype. The transport stays on the CPU, so the lens
            # matrices never take accelerator memory.
            scores = unembed(engine, layout, states[layer] @ matrix.float().T)
            if not torch.isfinite(scores).all():
                raise ValueError("The Jacobian readout produced non-finite scores.")
            # A pinned token keeps its own score; only the ranking leaves the excluded out.
            pinned_scores = scores[:, pinned_id].clone() if pinned_id is not None else None
            if excluded is not None:
                scores[:, excluded] = float("-inf")
            final = scores[-1]
            values, ids = final.topk(min(TOP_CANDIDATES, final.numel()))
            row = {
                "layer": layer,
                "candidates": [
                    {"token_id": int(token), "text": decode(int(token)), "score": float(score)}
                    for token, score in zip(ids, values)
                ],
            }
            # The concept table counts each position's top few, not just the
            # top one, so the grid's cell is the first of the same list.
            top_scores, top_ids = scores.topk(min(TOP_CANDIDATES, scores.shape[1]), dim=1)
            column = [
                {"token_id": int(best[0]), "text": decode(int(best[0])), "score": float(score[0])}
                for best, score in zip(top_ids, top_scores)
            ]
            if pinned_id is not None:
                ranks = (scores > pinned_scores[:, None]).sum(dim=1) + 1
                row["rank"] = int(ranks[-1])
                row["score"] = float(pinned_scores[-1])
                for cell, rank, score in zip(column, ranks, pinned_scores):
                    cell["pinned_rank"] = int(rank)
                    cell["pinned_score"] = float(score)
            rows.append(row)
            cells.append({"layer": layer, "cells": column, "top": top_ids.tolist()})
        return rows, cells


@contextmanager
def _capture_mlx(engine, wanted, states):
    """Fill ``states`` from the MLX layer recorder once the body has run."""
    import mlx.core as mx
    import torch

    from chatlab.mlx_runtime import MLX_THREAD, Recorder

    recorder = Recorder()
    with engine._recording(recorder):
        yield
    hidden = recorder.hidden

    def read():
        # The recorder keeps the embedding first, then each block's output.
        for layer in wanted:
            if layer + 1 < len(hidden):
                array = hidden[layer + 1][0].astype(mx.float32)
                mx.eval(array)
                states[layer] = torch.from_numpy(np.array(array))

    # The states were recorded on the MLX thread; see mlx_runtime.MLX_THREAD.
    MLX_THREAD.run(read)


def unembed(engine, layout: Layout, vectors):
    """Norm and head, the model's own, over rows of residual vectors.

    ``vectors`` is a float32 CPU tensor ``[n, d_model]``; the answer is a
    float32 CPU tensor ``[n, vocabulary]``.
    """
    import torch

    if layout.backend == "mlx":
        from chatlab.mlx_runtime import MLX_THREAD

        return torch.from_numpy(MLX_THREAD.run(_unembed_mlx, engine, layout, vectors.numpy()))
    norm_weight = next(layout.norm.parameters())
    head_weight = engine.model.get_output_embeddings().weight
    normed = layout.norm(vectors.to(device=norm_weight.device, dtype=norm_weight.dtype))
    return engine.read_head(normed.to(device=head_weight.device, dtype=head_weight.dtype)).float().cpu()


def _unembed_mlx(engine, layout: Layout, vectors: np.ndarray) -> np.ndarray:
    """:func:`unembed` for an MLX model, run on the MLX thread."""
    import mlx.core as mx

    # In the norm's own dtype, as the Transformers readout casts to it.
    array = mx.array(vectors)[None]
    weight = getattr(layout.norm, "weight", None)
    if weight is not None:
        array = array.astype(weight.dtype)
    logits = engine.read_head(layout.norm(array))[0].astype(mx.float32)
    mx.eval(logits)
    return np.array(logits)


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
_UPLOAD_LOCK = threading.Lock()


def _read_store() -> dict:
    return read_json_object(store_path()) or {}


def remembered(model_id: str) -> dict | None:
    """The lens last imported for ``model_id``, or ``None``."""
    record = _read_store().get(model_id)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    return record


def remember(model_id: str, record: dict) -> bool:
    """Write down which lens ``model_id`` uses, so a reload finds it again.

    The checkpoint's revision goes in beside it when it is known: the same
    ID redownloaded can hold different weights, and a lens fitted for the
    old ones must not be brought back for the new. Answers whether the
    record reached the disk; the caller says so either way.
    """
    with _STORE_LOCK:
        data = _read_store()
        data[model_id] = {
            key: record[key]
            for key in ("path", "fitted_model_id", "repository", "filename", "model_revision")
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
            return False
    return True


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
            raise ValueError(f"That lens file is larger than the {SIZE_LIMIT} limit.")
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


@dataclass(frozen=True)
class FoundLens:
    """One lens file published on the Hub for a model."""

    repository: str
    filename: str
    fitted_model_id: str
    size: int | None
    downloads: int
    updated: str


# How many repositories one search reads the file lists of.
MAX_FOUND_REPOSITORIES = 20
# Searched separately, since the Hub matches one substring of the repository
# name: "lens" finds J-lens, J++ lens and "jlens" repositories, "jacobian"
# the few named only for the method.
LENS_SEARCH_TERMS = ("lens", "jacobian")
# Queried as tags as well, so a repository whose name says neither is still
# found by what its card declares.
LENS_TAGS = ("jacobian-lens", "jacobian_lens", "jlens", "j-lens", "jpp-lens", "r-lens")
# A repository is kept only when its name or tags say it holds lenses of the
# kinds this module reads; "lens" alone also matches probes, oracles and
# training checkpoints.
LENS_REPOSITORY = re.compile(r"jacobian|j-?lens|jpp|j\+\+|r-?lens|rs-lens", re.IGNORECASE)


def _model_key(name: str) -> str:
    return re.sub(r"[\W_]+", "-", name.lower()).strip("-")


def _fits(filename: str, model_id: str) -> bool:
    """Whether a lens file plausibly belongs to ``model_id``.

    A repository of lenses for several models tags every one of them as its
    base model, and keeps each model's lenses in a folder named for it. A
    file at the top level is taken to be for the tagged model; one in a
    folder must name the model somewhere in its path.
    """
    if "/" not in filename:
        return True
    return _model_key(model_id.split("/")[-1]) in _model_key(filename)


def quantized_from(model_id: str, api=None) -> str | None:
    """The model ``model_id`` is a quantized version of, from its model card.

    A quantized repository (FP8, GPTQ, an MLX conversion) names its source
    as ``base_model`` and is tagged ``base_model:quantized:<source>``. Lenses
    are published against the source, so a search for the quantized
    checkpoint alone finds nothing. ``None`` when the card names no source,
    or the ID is not a Hub repository.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    if not REPOSITORY_ID.match(model_id or ""):
        return None
    try:
        info = (api or HfApi()).model_info(model_id)
    except (HfHubHTTPError, OSError, ValueError):
        return None
    prefix = "base_model:quantized:"
    sources = [tag[len(prefix):] for tag in info.tags or () if tag.startswith(prefix)]
    return sources[0] if len(sources) == 1 else None


def find_lenses(model_id: str, api=None) -> list[FoundLens]:
    """Lens files on the Hub whose model card names ``model_id`` as its base model.

    Each repository named or tagged as a lens is listed by its ``.pt`` files
    within the size limit that belong to the model; one with none is
    dropped. Most downloaded first. Raises
    ``ValueError`` when the Hub cannot be reached.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = api or HfApi()
    try:
        repositories = {}
        base = f"base_model:{model_id}"
        queries = [{"filter": base, "search": term} for term in LENS_SEARCH_TERMS]
        queries += [{"filter": [base, tag]} for tag in LENS_TAGS]
        for query in queries:
            for listed in api.list_models(**query, limit=MAX_FOUND_REPOSITORIES):
                repositories.setdefault(listed.id, None)
        found = []
        for repository in list(repositories)[:MAX_FOUND_REPOSITORIES]:
            info = api.model_info(repository, files_metadata=True)
            if not LENS_REPOSITORY.search(" ".join([repository, *(info.tags or ())])):
                continue
            updated = info.last_modified.date().isoformat() if info.last_modified else ""
            found.extend(
                FoundLens(repository, sibling.rfilename, model_id, sibling.size, info.downloads or 0, updated)
                for sibling in info.siblings or ()
                if sibling.rfilename.endswith(".pt") and _fits(sibling.rfilename, model_id)
                and (sibling.size is None or sibling.size <= MAX_FILE_BYTES)
            )
    except (HfHubHTTPError, OSError) as error:
        raise ValueError(f"Could not search Hugging Face: {error}") from None
    return sorted(found, key=lambda lens: (-lens.downloads, lens.repository, lens.filename))


def keep(path) -> Path:
    """Copy a lens chosen from disk under ``lens_directory()/uploads``; the kept path.

    A browser upload lands in Gradio's cache, which does not outlive the
    session, so the copy is what gets remembered. Called after the file has
    passed :meth:`FittedLens.load`, never before, so only a valid lens within
    the size limit is copied. A file already inside the lens directory stays
    where it is; a kept file with the same name and content is reused, and a
    different one takes a numbered name.
    """
    source = Path(path)
    directory = lens_directory()
    if source.resolve().is_relative_to(directory.resolve()):
        return source
    uploads = directory / "uploads"
    # Choosing the name and filling it are one step under the lock, so two
    # imports of different files with the same name cannot pick one target;
    # the copy lands under a working name and is renamed into place complete.
    with _UPLOAD_LOCK:
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / source.name
        counter = 1
        while target.exists() and not filecmp.cmp(source, target, shallow=False):
            counter += 1
            target = uploads / f"{source.stem}-{counter}{source.suffix}"
        if not target.exists():
            staged = target.with_name(target.name + ".part")
            shutil.copy2(source, staged)
            staged.replace(target)
    return target
