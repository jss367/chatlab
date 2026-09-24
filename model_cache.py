"""The Hugging Face cache: what it holds for a model, and what that would weigh.

Every model ChatLab runs comes out of the cache ``huggingface_hub`` keeps,
a folder per repository with a snapshot per revision. This module reads
those folders without importing torch: whether a snapshot is a text model,
an image pipeline, an MLX checkpoint or an adapter; which files it still
lacks before it can load; how many bytes its weights take on disk and how
many they would take in memory at a given precision; and which cached
models there are to list, sort, or delete.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import adapters
import mlx_runtime


# Bytes per parameter for the dtypes a checkpoint or a load can use.
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
}

# The group of weights that share one scale and one bias when a linear layer
# is quantized; see model_loading.QUANTIZED_BITS.
QUANTIZATION_GROUP_SIZE = 64


MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
# A full Git commit hash, which the Hub cache files a snapshot under by name;
# huggingface_hub's own REGEX_COMMIT_HASH.
COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


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
# Stands in for a whole repository among an adapter's missing files: the base
# model it was trained on, which :attr:`CacheStatus.base_model` names.
BASE_MODEL = "base model"

# A causal LM's weights come as a single file or as the shards an index lists,
# in one of these formats. ``from_pretrained`` looks for them in this order and
# loads the first it finds, so a snapshot is judged by that format alone.
WEIGHT_FORMATS = (
    ("model.safetensors", "model.safetensors.index.json"),
    ("pytorch_model.bin", "pytorch_model.bin.index.json"),
)

# The two kinds of model ChatLab loads. A text model is one checkpoint at the
# root of the snapshot and answers with tokens; an image model is a diffusers
# pipeline, a folder per component, and answers with a picture. They share
# the cache, the download, the memory ceiling and the one slot in memory, and
# part company at the loader and at the page that drives them.
TEXT_KIND = "text"
IMAGE_KIND = "image"
# A text model quantized for MLX: safetensors under Transformers' names, but
# packed the way mlx-lm packs them, so it runs through :mod:`mlx_runtime`
# rather than through AutoModelForCausalLM. It answers with tokens like a
# text model and is driven from the Chat page like one.
MLX_KIND = "mlx"


# A diffusers pipeline announces itself with this file, which also names
# every component folder it is made of.
PIPELINE_INDEX = "model_index.json"

# Where a pipeline component keeps its weights. Only the folders that hold a
# ``config.json`` are models at all: the tokenizer, the scheduler and the
# feature extractor each keep a config of their own name and no weights, so
# absence there is not a missing file.
COMPONENT_CONFIG = "config.json"
COMPONENT_WEIGHT_SUFFIXES = (".safetensors", ".bin")


@dataclass(frozen=True)
class CacheStatus:
    """What the Hugging Face cache already holds for one model.

    ``missing_files`` names what the ``main`` snapshot still lacks before the
    model can load, and is the one verdict on that: a cache another tool
    filled with only the config and tokenizer, or a download stopped between
    shards, has finished blobs but no model. ``kind`` says which of the two
    kinds of model it is (:data:`TEXT_KIND` or :data:`IMAGE_KIND`), and is
    empty for a snapshot that is whole but neither (a CTranslate2 or ONNX
    export, a folder of SAE weights): nothing is missing, ChatLab just cannot
    load it. ``cached_bytes`` counts finished files; ``partial_files`` and
    ``partial_bytes`` count the ``.incomplete`` blobs a cut-off download left
    behind, which ``snapshot_download`` resumes rather than restarts. Those are a size estimate, not a verdict: the blob
    folder is shared by every revision of the repo, so a stray partial may
    belong to another revision or to a file the model never loads, and a
    partial the snapshot does need already shows up in ``missing_files``,
    since the hub links a file into the snapshot only once it has finished.

    A LoRA adapter is a text model whose weights are another repository's:
    ``base_model`` names that repository, and :data:`BASE_MODEL` stands among
    the missing files until it is whole on disk too, so an adapter is only
    ``complete`` when both halves are. ``unsupported_reason`` says why an
    adapter ChatLab cannot load is unsupported, where the general account of
    what ChatLab loads would not.
    """

    cached_bytes: int = 0
    partial_files: int = 0
    partial_bytes: int = 0
    missing_files: tuple[str, ...] = ()
    kind: str = TEXT_KIND
    base_model: str | None = None
    unsupported_reason: str = ""

    @property
    def present(self) -> bool:
        return self.cached_bytes > 0 or self.partial_files > 0

    @property
    def unsupported(self) -> bool:
        """Whole, but not a model of either kind ChatLab can load."""

        return not self.kind

    @property
    def complete(self) -> bool:
        """Whole and loadable: on disk, nothing missing, and a model ChatLab runs."""

        return self.present and not self.missing_files and not self.unsupported

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
    returns ``snapshots/<commit>`` whether or not every file is in it. A
    branch or tag has that ref file; a revision that is already a commit hash
    does not, since there is nothing to resolve, and the hub goes straight to
    ``snapshots/<hash>``, as this does.
    """

    ref = folder / "refs" / revision
    if ref.is_file():
        commit = ref.read_text().strip()
    elif COMMIT_HASH.fullmatch(revision):
        commit = revision
    else:
        return None
    snapshot = folder / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def is_transformers_config(path: Path) -> bool:
    """Whether ``config.json`` describes a Transformers model.

    Every ``AutoConfig`` carries a ``model_type``; most also list
    ``architectures``. A CTranslate2 export's ``config.json`` has neither, and
    a diffusers pipeline has no root ``config.json`` at all.
    """

    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and (
        "model_type" in config or "architectures" in config
    )


def is_mlx_snapshot(snapshot: Path) -> bool:
    """Whether the root ``config.json`` says the weights are quantized for MLX.

    An unquantized MLX conversion is a Transformers checkpoint under another
    name - same config, same tensors - and loads as one. Only the quantized
    ones need mlx-lm, and they say so in their config; see
    :func:`mlx_runtime.mlx_quantization`.
    """

    return mlx_runtime.read_mlx_config(snapshot) is not None


def has_root_checkpoint(snapshot: Path) -> bool:
    """Whether a Transformers checkpoint, single or sharded, sits at the root."""

    return any((snapshot / name).is_file() for pair in WEIGHT_FORMATS for name in pair)


def is_adapter_snapshot(snapshot: Path | None) -> bool:
    """Whether the snapshot is a LoRA adapter repository rather than a model.

    A repo that ships a whole checkpoint beside its ``adapter_config.json``
    (a merged export that kept the config) is the model it holds.
    """

    return (
        snapshot is not None
        and adapters.is_adapter(snapshot)
        and not has_root_checkpoint(snapshot)
    )


def adapter_base_snapshot(snapshot: Path) -> Path | None:
    """The cached snapshot of the model an adapter snapshot was trained on.

    Looked up in the cache the adapter itself sits in, which its path says:
    a snapshot is ``<cache>/models--org--name/snapshots/<commit>``, and at
    the revision the adapter pins, where it pins one; see
    :func:`adapters.base_revision`. ``None`` when the adapter names no Hub
    repository or the base has no snapshot at that revision.
    """

    config = adapters.read_adapter_config(snapshot) or {}
    base = adapters.base_model_id(config)
    if base is None:
        return None
    root = snapshot.parents[2] if snapshot.parent.name == "snapshots" else None
    revision = adapters.base_revision(config) or "main"
    return snapshot_folder(cache_folder(base, root), revision)


def mlx_available() -> bool:
    """Whether the MLX backend can run here; see :func:`mlx_runtime.mlx_available`."""

    return mlx_runtime.mlx_available()


def mlx_bits_from_id(model_id: str) -> int | None:
    """The bit width an MLX repository's name claims; see :func:`mlx_runtime.bits_from_name`."""

    return mlx_runtime.bits_from_name(model_id)


def mlx_snapshot_bits(snapshot: Path) -> int | None:
    """The width an MLX repo on disk was converted to, or ``None`` for none.

    Read from the repo's own config rather than guessed from its name, which
    is what :func:`mlx_bits_from_id` has to settle for on a search result.
    This is the width such a load really packs its linear layers into, so it
    is what the estimate, the refusal and the fit verdict name: the precision
    radio has no say over an MLX repo, and a message that reported the radio
    would tell a reader their 4-bit model needed full 16-bit weights.
    """

    block = mlx_runtime.mlx_quantization(mlx_runtime.read_mlx_config(snapshot))
    return None if block is None else block["bits"]


# File endings that hold model weights in some framework or other. A file
# with one of these where a Transformers checkpoint would not put it is the
# positive evidence that a snapshot is a repo of another kind.
WEIGHT_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".npz", ".gguf",
     ".msgpack", ".h5", ".tflite", ".mlmodel"}
)

# Endings Transformers never loads from. Even a repo that keeps its source
# ``config.json`` is not a Transformers checkpoint when these are all it has.
FOREIGN_SUFFIXES = frozenset(
    {".onnx", ".npz", ".gguf", ".tflite", ".mlmodel", ".h5", ".msgpack"}
)

SHARD_NAME = re.compile(r"-\d{5}-of-\d{5}\.(safetensors|bin)$")

# What the Transformers ``Trainer`` leaves beside a checkpoint. These share
# the weight suffixes but are not weights, so a download that has fetched
# one of them before the config is still a Transformers repo mid-download.
TRAINER_ARTIFACTS = re.compile(
    r"^(training_args\.bin|optimizer\.pt|scheduler\.pt|scaler\.pt|rng_state(_\d+)?\.pth)$"
)


def foreign_weights(snapshot: Path, *, transformers_config: bool) -> bool:
    """Whether the snapshot holds weights laid out for something other than Transformers.

    The evidence is a weight file where ``from_pretrained`` would never look:
    at the root under a name that is not a checkpoint or a shard (CTranslate2's
    ``model.bin``, an ``model.onnx``), or in a subfolder. When the root
    ``config.json`` is a Transformers one, only a foreign format in a
    subfolder counts, since such repos often ship extras like
    ``original/consolidated.00.pth`` beside the checkpoint they are missing.

    A diffusers pipeline is also weights in subfolders, and would answer
    yes here; :func:`judge_snapshot` recognizes it by its
    ``model_index.json`` before asking, because ChatLab loads that kind.
    """

    checkpoints = {name for pair in WEIGHT_FORMATS for name in pair}
    for entry in snapshot.rglob("*"):
        if not entry.is_file() or entry.suffix not in WEIGHT_SUFFIXES:
            continue
        if entry.parent == snapshot:
            if (
                entry.name in checkpoints
                or SHARD_NAME.search(entry.name)
                or TRAINER_ARTIFACTS.match(entry.name)
            ):
                continue
            if transformers_config and entry.suffix not in FOREIGN_SUFFIXES:
                continue
            return True
        if not transformers_config or entry.suffix in FOREIGN_SUFFIXES:
            return True
    return False


def judge_snapshot(snapshot: Path | None) -> tuple[tuple[str, ...], str]:
    """``(missing_files, kind)`` for what the snapshot holds.

    ``missing_files`` are the files ``from_pretrained`` needs before it can
    load: for a text model only the config and the weights are checked, since
    which tokenizer files a repo ships varies too much to know from the
    outside, and a wrong "incomplete" verdict on a good cache would be worse
    than a generic load error. ``kind`` is :data:`TEXT_KIND` or
    :data:`IMAGE_KIND`, and empty for a snapshot with no Transformers
    checkpoint at its root but weights laid out for a framework ChatLab does
    not run (CTranslate2, ONNX, a folder of SAE weights): that is not a
    cut-off download, and nothing is reported missing for it. Absence alone
    is never that verdict: a snapshot holding only a tokenizer, or only a
    config, is incomplete.

    The pipeline check comes first, because a diffusers repo keeps its
    weights in subfolders and has no checkpoint at its root, which is
    exactly the shape :func:`foreign_weights` reads as another framework.
    """

    if snapshot is None:
        return ("config.json", MODEL_WEIGHTS), TEXT_KIND
    if is_pipeline(snapshot):
        if not pipeline_draws_from_text(snapshot):
            # A diffusers pipeline all right, and not one the Images page
            # can drive: it wants a picture, a video frame or a sound
            # ChatLab has no way to give it. See pipeline_draws_from_text.
            return (), ""
        return pipeline_missing_files(snapshot), IMAGE_KIND
    has_checkpoint = has_root_checkpoint(snapshot)
    if has_checkpoint and is_mlx_snapshot(snapshot):
        # Whole files that Transformers cannot read: the weights are packed
        # for mlx-lm. Loadable where mlx is installed, which is Apple
        # silicon; anywhere else the verdict is the one a CTranslate2 export
        # gets, since nothing is missing and nothing here runs it.
        return missing_files(snapshot), MLX_KIND if mlx_available() else ""
    if not has_checkpoint and adapters.is_adapter(snapshot):
        # A text model once its base is read in: only the adapter's own
        # files are judged here, and cache_status judges the base, since
        # that needs the cache the two share.
        return adapters.missing_adapter_files(snapshot), TEXT_KIND
    if not has_checkpoint and adapters.has_adapter_weights(snapshot):
        # An adapter cut off after its weights landed and before its config
        # did. The weights alone would read as another framework's below,
        # and an unsupported verdict hides the download that would finish it.
        return (adapters.ADAPTER_CONFIG,), TEXT_KIND
    if not has_checkpoint and foreign_weights(
        snapshot, transformers_config=is_transformers_config(snapshot / "config.json")
    ):
        return (), ""
    return missing_files(snapshot), TEXT_KIND


def is_pipeline(snapshot: Path) -> bool:
    """Whether the snapshot is a diffusers pipeline rather than one checkpoint."""

    return (snapshot / PIPELINE_INDEX).is_file()


# What a pipeline class name says it needs besides a prompt. diffusers files
# video, audio, image-conditioned and upscaling pipelines under the same
# ``model_index.json`` as a text-to-image one, and the Images page has only a
# prompt to give: such a pipeline would load and then fail for want of an
# image, a video frame or an audio clip it was never handed.
CONDITIONED_PIPELINES = (
    "img2img",
    "image2image",
    "inpaint",
    "instructpix2pix",
    "controlnet",
    "upscale",
    "superresolution",
    "depth",
    "variation",
    "video",
    "audio",
    "music",
    "adapter",
    # A prior stage takes the prompt and hands back conditioning embeddings
    # for a second pipeline to draw from. It has a tokenizer and a text
    # encoder like any text-to-image pipeline and returns no picture at all.
    "prior",
    # Subject-driven generation: a reference image and a subject category
    # beside the prompt.
    "blip",
)

# The components a pipeline needs to read a prompt at all. One without them
# is conditioned on something else - an image embedding, a video frame - and
# is not something a prompt alone drives, whatever its class is called.
TEXT_TO_IMAGE_COMPONENTS = ("tokenizer", "text_encoder")


def pipeline_draws_from_text(snapshot: Path) -> bool:
    """Whether this pipeline is one a prompt alone can drive.

    Two questions, because neither answers on its own. The components say
    whether it can read a prompt: a pipeline with no tokenizer and no text
    encoder is conditioned on something else and could not use one. The
    class name says whether it wants more than a prompt: an img2img or an
    upscaling pipeline has both components and still needs a picture handed
    to it.

    A guess either way, and deliberately the conservative one: a pipeline
    this turns down is reported unsupported rather than offered and then
    failed at the first draw, and its ID can still be typed into the model
    box for a load that says what really went wrong.

    A guess is all it can be from here. Naming the class is the only signal
    a cache scan has, since it reads folders without importing anything, and
    a list of markers will always be one family behind. What the pipeline
    really requires is read off its own ``__call__`` once it is built; see
    :func:`image_runtime.refuse_unusable`, which is the exact check and the
    one that catches a family nobody has thought of.
    """

    name = (pipeline_class(snapshot) or "").lower()
    if any(marker in name for marker in CONDITIONED_PIPELINES):
        return False
    components = pipeline_components(snapshot)
    return all(needed in components for needed in TEXT_TO_IMAGE_COMPONENTS)


def pipeline_components(snapshot: Path) -> tuple[str, ...]:
    """The component subfolders ``model_index.json`` names, in its own order.

    A pipeline is a handful of models that run in sequence, one folder each,
    and the index is the list. Its other entries are skipped: the keys that
    begin with an underscore are the pipeline's own class and version, and a
    component the repo ships without (a safety checker it left out) is
    written as a pair of nulls rather than dropped.
    """

    try:
        index = json.loads((snapshot / PIPELINE_INDEX).read_text())
    except (OSError, ValueError):
        return ()
    if not isinstance(index, dict):
        return ()
    return tuple(
        name
        for name, value in index.items()
        if not name.startswith("_")
        and isinstance(value, list)
        and len(value) == 2
        and all(isinstance(part, str) and part for part in value)
    )


def pipeline_class(snapshot: Path) -> str | None:
    """The pipeline class ``model_index.json`` names, for the model list."""

    try:
        index = json.loads((snapshot / PIPELINE_INDEX).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict):
        return None
    name = index.get("_class_name")
    return name if isinstance(name, str) and name else None


def _component_weights(folder: Path) -> list[Path]:
    """The weight files one component folder holds, whatever their variant."""

    if not folder.is_dir():
        return []
    return [
        entry
        for entry in folder.iterdir()
        if entry.is_file() and entry.suffix in COMPONENT_WEIGHT_SUFFIXES
    ]


def _component_index(folder: Path, variant: str = "") -> Path | None:
    """The shard index a component will load from, or ``None`` if it has none.

    A big component ships its weights as shards with an index listing them,
    the same as a text checkpoint. Which index matters is the one
    ``from_pretrained`` will read, and that depends on the variant the whole
    pipeline is being loaded as: a component holding both a complete plain
    index and an incomplete half-precision one is complete for a plain load
    and short of shards for a half-precision one, so checking the plain
    index of an fp16 load would call the snapshot whole and then fail inside
    diffusers.

    So the pipeline's variant comes first, then the plain set - which is
    what diffusers falls back to for a component that has no such variant.
    An index for neither means the set being loaded is not sharded, and
    ``None`` says there is nothing to check shard by shard: matching any
    index at all would validate a half-precision one against a plain load
    and report shards missing that the load never asks for.
    """

    if not folder.is_dir():
        return None
    indexes = {
        _weight_variant(entry.name.removesuffix(".index.json")): entry
        for entry in sorted(folder.iterdir())
        if entry.is_file() and entry.name.endswith(".index.json")
    }
    preferences = [(variant, "safetensors"), (variant, "bin")] if variant else []
    preferences += [("", "safetensors"), ("", "bin")]
    for preferred in preferences:
        if preferred in indexes:
            return indexes[preferred]
    return None


def _shard_index_name(shard: Path) -> str:
    """The index file a shard belongs to, by the name diffusers looks for."""

    return f"{SHARD_NAME.sub(lambda match: f'.{match.group(1)}', shard.name)}.index.json"


def _missing_shards(index: Path) -> tuple[str, ...]:
    """The shards ``index`` names that are not beside it, or the index itself.

    An index this cannot read counts as the weights being missing rather than
    as nothing missing: the file is another repo's, so a ``weight_map`` that
    is not an object of file names says the component cannot be loaded, which
    is the thing worth reporting.
    """

    try:
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map must be a non-empty object")
        shards = {shard for shard in weight_map.values()}
        if not all(isinstance(shard, str) and shard for shard in shards):
            raise TypeError("weight_map values must be file names")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return (MODEL_WEIGHTS,)
    return tuple(sorted(shard for shard in shards if not (index.parent / shard).is_file()))


def pipeline_missing_files(snapshot: Path) -> tuple[str, ...]:
    """The files a pipeline snapshot needs before ``from_pretrained`` can load it.

    Each component the index names must have its folder, and each folder
    that holds a ``config.json`` must have weights beside it. The folders
    without one are the tokenizer, the scheduler and the feature extractor,
    which keep a config under a name of their own and no weights at all, so
    for those the folder's presence is the whole check. Which variant of the
    weights is there is not checked: a repo that ships only the half
    precision set loads from it.

    A component whose weights come as shards is checked shard by shard
    against its own index, exactly as a text checkpoint is. Without that, a
    download cut off after the first shard leaves a folder that has *some*
    weights, which would read as whole and send **Load cached** into
    diffusers to fail on the shards that never arrived.

    A snapshot whose variant-only components share no single variant is
    reported the same way. Its files are all there, but ``from_pretrained``
    takes one variant for the whole pipeline, so there is no load to make of
    it; see :func:`pipeline_variant_missing`.

    A folder holding weights but no ``config.json`` is missing that config
    rather than exempt from the check: diffusers cannot construct a unet or
    a VAE without one, so a download cut off before it arrived is
    incomplete rather than whole. Absence of both is still the tokenizer and
    the scheduler, and still no gap.

    Every component is checked against the variant the whole pipeline will
    be loaded as, not against whichever set it happens to prefer on its own:
    ``from_pretrained`` takes one variant for the lot, so a component with a
    complete plain index beside an incomplete half-precision one is short of
    shards for a half-precision load. See :func:`_component_index`.

    What is *not* checked is which files a weightless component needs. A
    CLIP tokenizer wants ``vocab.json`` and ``merges.txt``, a fast one
    ``tokenizer.json``, a T5 one ``spiece.model``, and knowing which from
    outside means knowing the class - the same reason :func:`missing_files`
    checks only the config and the weights of a text checkpoint, and for the
    same trade: a wrong "incomplete" verdict on a good cache is worse than
    the loader's own error on a bad one. An empty folder is the exception,
    because none is none whatever the class.
    """

    missing: list[str] = []
    if pipeline_variant_missing(snapshot):
        return (MODEL_WEIGHTS,)
    variant = pipeline_variant(snapshot) or ""
    for name in pipeline_components(snapshot):
        folder = snapshot / name
        if not folder.is_dir():
            missing.append(f"{name}/")
            continue
        if not any(folder.iterdir()):
            # The folder was made and nothing was fetched into it, which is
            # the same gap as its not being there at all. How many files a
            # tokenizer or a scheduler needs cannot be told from outside -
            # see the note in pipeline_missing_files - but none is none.
            missing.append(f"{name}/")
            continue
        if not (folder / COMPONENT_CONFIG).is_file():
            if (
                _component_weights(folder)
                or _component_index(folder, variant) is not None
            ):
                missing.append(f"{name}/{COMPONENT_CONFIG}")
            continue
        index = _component_index(folder, variant)
        if index is not None:
            missing.extend(f"{name}/{shard}" for shard in _missing_shards(index))
            continue
        # The set this load will read, not whatever else is in the folder:
        # a complete file of another variant does not make up for the one
        # being loaded, and an orphan shard of the one being loaded is not
        # excused by a complete file of another.
        wanted = [
            entry
            for entry in _component_weights(folder)
            if _weight_variant(entry.name)[0] == variant
        ] or _component_weights(folder)
        if not wanted:
            missing.append(f"{name}/{MODEL_WEIGHTS}")
        elif all(SHARD_NAME.search(entry.name) for entry in wanted):
            # Shards and no index. Diffusers finds a sharded checkpoint
            # through its index and nothing else, so a download that left
            # the shards but not the index has weights that cannot be
            # discovered - which is a gap, not a whole unsharded set.
            missing.append(f"{name}/{_shard_index_name(wanted[0])}")
    return tuple(missing)


# A weight file's variant, as diffusers spells it: the part between the name
# and the suffix in ``diffusion_pytorch_model.fp16.safetensors``. A repo often
# ships both a full-precision and a half-precision set of the same component,
# and counting both would double the component's measured size.
_VARIANT = re.compile(r"^(?P<stem>.+?)(?:\.(?P<variant>[^.]+))?\.(?P<suffix>[^.]+)$")


def _weight_variant(name: str) -> tuple[str, str]:
    """``(variant, suffix)`` for a weight file name; the variant is empty for the plain set.

    A shard is ``...-00001-of-00002.safetensors``, whose middle part is not a
    variant, so the shard numbering is stripped before the name is read.
    """

    stripped = SHARD_NAME.sub(lambda match: f".{match.group(1)}", name)
    found = _VARIANT.match(stripped)
    if found is None:
        return "", ""
    return found.group("variant") or "", found.group("suffix") or ""


def _loaded_variant_bytes(files: list[Path]) -> int:
    """The bytes of the one weight set a component will really load.

    Grouped by variant and format, because ``from_pretrained`` reads one
    group and ignores the rest. The plain safetensors set is preferred, then
    the plain PyTorch one, the same order diffusers looks in; a repo that
    ships only a variant (half precision alone, say) is measured by its
    largest group, which is the upper bound on what a load can read.
    """

    groups: dict[tuple[str, str], int] = {}
    for entry in files:
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        group = _weight_variant(entry.name)
        groups[group] = groups.get(group, 0) + size
    for preferred in (("", "safetensors"), ("", "bin")):
        if preferred in groups:
            return groups[preferred]
    return max(groups.values(), default=0)


def pipeline_weight_bytes(snapshot: Path) -> int | None:
    """Bytes of the weights a pipeline load will read, or ``None``.

    The sum over the component folders, each measured by the one weight set
    it will really load; see :func:`_loaded_variant_bytes`. ``None`` when the
    index cannot be read, which leaves the loader to give its own error.
    """

    components = pipeline_components(snapshot)
    if not components:
        return None
    return sum(
        _loaded_variant_bytes(_component_weights(snapshot / name))
        for name in components
    )


# How safetensors spells the dtypes in its header, in the names
# :data:`DTYPE_BYTES` uses. The header is a length-prefixed JSON object at the
# front of the file, so reading one costs a seek rather than a load.
SAFETENSORS_DTYPES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I8": "int8",
    "U8": "uint8",
}

# The header length prefix: eight bytes, little-endian, unsigned.
_HEADER_PREFIX = 8
# Refuse to read a header claiming to be larger than any real one. A header
# is a few hundred kilobytes for the largest checkpoints; a wild length is a
# truncated or hostile file, not something to allocate for.
MAX_SAFETENSORS_HEADER = 64 * 1024**2


def safetensors_dtype(path: Path) -> str | None:
    """The dtype the tensors in one safetensors file are stored as, or ``None``.

    Read from the file's own header rather than from a config, because the
    configs that matter do not say: a diffusers component keeps its
    architecture in ``config.json`` and its dtype nowhere, so the unet that
    dominates a pipeline's size would otherwise be unmeasurable. The first
    tensor's dtype stands for the file, which is what a checkpoint saved in
    one precision holds.
    """

    try:
        with path.open("rb") as handle:
            prefix = handle.read(_HEADER_PREFIX)
            if len(prefix) < _HEADER_PREFIX:
                return None
            length = int.from_bytes(prefix, "little")
            if not 0 < length <= MAX_SAFETENSORS_HEADER:
                return None
            header = json.loads(handle.read(length))
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        stored = SAFETENSORS_DTYPES.get(entry.get("dtype"))
        if stored is not None:
            return stored
    return None


# What a component whose stored dtype cannot be read is taken to be.
# Pipelines ship as half precision, and assuming it errs towards refusing a
# load rather than towards a float32 load that doubles past the estimate and
# exhausts the machine.
ASSUMED_PIPELINE_DTYPE = "float16"


def component_dtype(folder: Path) -> str | None:
    """The dtype one component's weights are stored as, or ``None``.

    Read from the largest safetensors file it will load, because that is the
    one whose size the component's estimate turns on. A component that ships
    only ``.bin`` weights answers ``None``: a pickle has no cheap header to
    read.
    """

    files = [
        entry
        for entry in _component_weights(folder)
        if entry.suffix == ".safetensors"
    ]
    if not files:
        return None
    try:
        largest = max(files, key=lambda entry: entry.stat().st_size)
    except OSError:
        return None
    return safetensors_dtype(largest)


def pipeline_loaded_bytes(snapshot: Path, load_dtype: str) -> int | None:
    """Memory a pipeline's weights take once loaded as ``load_dtype``, or ``None``.

    Summed component by component, each converted from its own stored dtype,
    because a pipeline's components need not share one and a single dtype
    applied to the whole byte count mis-scales the ones that differ: a
    float32 text encoder beside a half-precision unet would have the unet's
    growth on a CPU float32 load go unestimated, and the inverse on a
    half-precision load.

    A component whose dtype cannot be read is assumed to be half precision,
    which errs towards refusing a load rather than towards one that doubles
    past the estimate; see :meth:`ModelManager._check_memory`.
    """

    components = pipeline_components(snapshot)
    if not components:
        return None
    total = 0
    for name in components:
        folder = snapshot / name
        stored = _loaded_variant_bytes(_component_weights(folder))
        if not stored:
            continue
        total += estimate_loaded_bytes(
            stored, component_dtype(folder) or ASSUMED_PIPELINE_DTYPE, load_dtype
        )
    return total


def _component_variants(folder: Path) -> set[str]:
    """The weight variants one component folder can actually be loaded from.

    A variant counts only where its files are a set ``from_pretrained``
    could read: one unsharded file, or shards with the index that lists
    them. An orphan shard left by a cut-off download is not a plain set just
    because it is named like one - counting it would say a plain load is
    available, send :func:`pipeline_variant` looking for unsuffixed weights,
    and leave a perfectly good half-precision file beside it unused.
    """

    grouped: dict[str, list[Path]] = {}
    for entry in _component_weights(folder):
        grouped.setdefault(_weight_variant(entry.name)[0], []).append(entry)
    usable = set()
    for variant, entries in grouped.items():
        if any(not SHARD_NAME.search(entry.name) for entry in entries):
            usable.add(variant)
        elif _component_index(folder, variant) is not None:
            usable.add(variant)
    return usable


def pipeline_variant(snapshot: Path) -> str | None:
    """The variant a load has to ask diffusers for, or ``None`` for the plain set.

    ``from_pretrained`` looks for unsuffixed weights unless it is told a
    variant, and it takes one variant for the whole pipeline, falling back to
    the unsuffixed files for any component that lacks it. So the variant has
    to be one that *every* component without a plain set ships: naming a
    variant only some of them have leaves the rest with nothing for diffusers
    to fall back to.

    ``None`` when every weight-bearing component has a plain set, which is
    the common case and what diffusers wants asked of it, and also when no
    single variant covers the ones that do not — :func:`pipeline_missing_files`
    reports that snapshot's weights as missing rather than letting a load
    start that cannot finish.
    """

    shared: set[str] | None = None
    for name in pipeline_components(snapshot):
        variants = _component_variants(snapshot / name)
        if not variants or "" in variants:
            continue
        shared = variants if shared is None else shared & variants
    if not shared:
        return None
    return sorted(shared)[0]


def pipeline_variant_missing(snapshot: Path) -> bool:
    """Whether the components disagree about variants past any hope of loading.

    True when some component has no plain weight set and no one variant is
    shipped by all such components, so whatever :func:`pipeline_variant`
    named would leave another component unloadable. The snapshot is whole in
    the sense that files are there; it is the combination that cannot be
    asked for.
    """

    lacking = [
        _component_variants(snapshot / name)
        for name in pipeline_components(snapshot)
        if _component_variants(snapshot / name)
        and "" not in _component_variants(snapshot / name)
    ]
    if not lacking:
        return False
    return not set.intersection(*lacking)


def weight_bytes_for(snapshot: Path, kind: str) -> int | None:
    """Bytes of the weights a load of this kind will read, or ``None``."""

    if kind == IMAGE_KIND:
        return pipeline_weight_bytes(snapshot)
    return snapshot_weight_bytes(snapshot)


def missing_files(snapshot: Path) -> tuple[str, ...]:
    """The files a snapshot needs before ``from_pretrained`` can load it."""

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


def cache_status(
    model_id: str, cache_dir: Path | None = None, revision: str | None = None
) -> CacheStatus:
    """Measure what is already on disk for ``model_id``, without touching the network.

    The snapshot judged is the default branch's unless ``revision`` names
    another, which is how an adapter's pinned base is judged.
    """

    return _cache_status(model_id, cache_dir, follow_base=True, revision=revision)


def _cache_status(
    model_id: str,
    cache_dir: Path | None,
    follow_base: bool,
    revision: str | None = None,
) -> CacheStatus:
    """:func:`cache_status`, judging an adapter's base only when ``follow_base``.

    The base is judged one level deep. Its own status is read without
    following further, which is enough to see that it is an adapter too and
    to refuse the pair, and it means two adapters naming each other cannot
    send the scan round in a circle.
    """

    folder = cache_folder(model_id, cache_dir)
    snapshot = snapshot_folder(folder, revision or "main")
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
    missing, kind = judge_snapshot(snapshot)
    status = CacheStatus(cached, partial_files, partial_bytes, missing, kind)
    if kind == TEXT_KIND and is_adapter_snapshot(snapshot):
        return _adapter_status(model_id, snapshot, cache_dir, status, follow_base)
    return status


def _adapter_status(
    model_id: str,
    snapshot: Path,
    cache_dir: Path | None,
    status: CacheStatus,
    follow_base: bool,
) -> CacheStatus:
    """An adapter's :class:`CacheStatus`: its own files, and its base's."""

    config = adapters.read_adapter_config(snapshot) or {}
    problem = adapters.adapter_problem(config, validate_model_id(model_id))
    if problem is not None:
        return replace(status, kind="", unsupported_reason=f"This adapter {problem}")
    base = adapters.base_model_id(config)
    status = replace(status, base_model=base)
    if not follow_base:
        return status
    base_status = _cache_status(
        base, cache_dir, follow_base=False, revision=adapters.base_revision(config)
    )
    if base_status.base_model is not None:
        return replace(
            status,
            kind="",
            unsupported_reason=(
                f"This adapter was trained on `{base}`, which is itself an "
                "adapter. ChatLab merges one adapter into a whole model."
            ),
        )
    if base_status.complete and base_status.kind != TEXT_KIND:
        what = "an image model" if base_status.kind == IMAGE_KIND else "quantized for MLX"
        return replace(
            status,
            kind="",
            unsupported_reason=(
                f"This adapter was trained on `{base}`, which is {what}. A LoRA "
                "adapter merges into a Transformers text model's weights."
            ),
        )
    if base_status.unsupported:
        return replace(
            status,
            kind="",
            unsupported_reason=(
                f"This adapter was trained on `{base}`, which is on disk but is "
                "not a model ChatLab loads."
            ),
        )
    if not base_status.complete:
        return replace(status, missing_files=(*status.missing_files, BASE_MODEL))
    return status


def folder_bytes(folder: Path) -> int:
    """The bytes a cache folder holds, counting every regular file once.

    In the usual layout the snapshots are symlinks into ``blobs`` and only
    the blobs count; on a filesystem without symlinks the snapshots hold the
    files themselves and count instead. Either way this is what deleting the
    folder frees, every revision included, where :func:`cache_status` sizes
    the ``main`` snapshot alone.
    """

    total = 0
    for entry in folder.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


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


def format_count(count: int) -> str:
    """Render a tally the way the hub's own pages do: ``1.2M``, ``45K``."""

    size = float(count)
    for unit in ("", "K", "M", "B"):
        if size < 1000 or unit == "B":
            break
        size /= 1000
    if unit == "":
        return str(count)
    return f"{size:.1f}{unit}" if size < 10 else f"{size:.0f}{unit}"


MODEL_FOLDER_PREFIX = "models--"


@dataclass(frozen=True)
class CachedModel:
    """One model the Hugging Face cache holds, and what is known of it offline.

    ``status`` is the same verdict :func:`cache_status` gives, so a folder a
    cut-off download left behind is listed with its missing files rather than
    hidden. ``disk_bytes`` is the whole folder, every revision included, as
    :func:`folder_bytes` measures it: what the list shows as the size, what
    the size orders sort by, and what removing the model frees, so those
    three never disagree. ``files`` counts what the ``main`` snapshot has so
    far; ``updated``
    is the newest write among the model's files, as epoch seconds, which is
    when it was last downloaded or resumed. ``architecture`` and ``dtype``
    come from the snapshot's ``config.json`` and are absent when it is.
    """

    model_id: str
    status: CacheStatus
    files: int = 0
    commit: str | None = None
    updated: float | None = None
    architecture: str | None = None
    dtype: str | None = None
    path: Path | None = None
    disk_bytes: int | None = None

    @property
    def size_bytes(self) -> int:
        return self.disk_bytes if self.disk_bytes is not None else self.status.total_bytes


def snapshot_weight_bytes(snapshot: Path) -> int | None:
    """Bytes of the weight files ``from_pretrained`` will read, or ``None``.

    Follows the same format order as the loader, so a repo that ships both a
    safetensors set and a legacy ``.bin`` set is measured by the one it loads.
    """

    for single, index_name in WEIGHT_FORMATS:
        try:
            if (snapshot / single).is_file():
                return (snapshot / single).stat().st_size
            index = snapshot / index_name
            if not index.is_file():
                continue
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = {shard for shard in weight_map.values() if isinstance(shard, str)}
            return sum((snapshot / shard).stat().st_size for shard in shards)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    return None


def estimate_loaded_bytes(
    weight_bytes: int, checkpoint_dtype: str | None, load_dtype: str
) -> int:
    """Memory the weights take once loaded as ``load_dtype``.

    A checkpoint is converted on the way in, so a float32 file loaded as
    float16 halves and a bfloat16 file loaded on the CPU as float32 doubles. A
    checkpoint whose dtype is unknown or unlisted is assumed to match.
    """

    stored = DTYPE_BYTES.get((checkpoint_dtype or "").lower())
    loaded = DTYPE_BYTES.get(load_dtype.lower())
    if stored is None or loaded is None:
        return int(weight_bytes)
    return int(weight_bytes * loaded / stored)


def estimate_quantized_bytes(
    weight_bytes: int,
    checkpoint_dtype: str | None,
    bits: int,
    embedding_params: int | None,
    group_size: int = QUANTIZATION_GROUP_SIZE,
) -> int:
    """Memory the weights take once the linear layers are quantized to ``bits``.

    Each packed weight costs ``bits`` and each group of ``group_size`` of them
    a half-precision scale and bias. The embeddings and the output head are
    left as they are, so a model whose vocabulary is a large share of its
    parameters saves less than the bit width alone would suggest.
    ``embedding_params`` is how many parameters those matrices hold, or
    ``None`` to treat every parameter as quantized.
    """

    half = estimate_loaded_bytes(weight_bytes, checkpoint_dtype, "float16")
    params = half / 2
    embedding = min(float(embedding_params or 0), params)
    per_param = bits / 8 + 2 * 2 / group_size
    return int(embedding * 2 + (params - embedding) * per_param)


def estimate_snapshot_bytes(
    snapshot: Path,
    load_dtype_name: str,
    bits: int | None = None,
    kind: str = TEXT_KIND,
) -> int | None:
    """Memory the weights in ``snapshot`` would take once loaded, or ``None``.

    ``None`` where the weight files cannot be measured — a snapshot short of
    files, or an index that cannot be read. A load in that state is let
    through to the loader, which gives its own more specific error, and a
    reader is told the size is unknown rather than shown a guess.

    An image pipeline is measured component by component, each from its own
    stored dtype (see :func:`pipeline_loaded_bytes`), because there is no one
    checkpoint at the root to read and no one dtype to read it as. ``bits``
    does not apply to one: the Metal quantizer is Transformers' own and a
    load has already cleared the choice, so it is ignored rather than
    reported as though it had been honoured.
    """

    if kind == IMAGE_KIND:
        return pipeline_loaded_bytes(snapshot, load_dtype_name)
    if kind == TEXT_KIND and is_adapter_snapshot(snapshot):
        # What a merged adapter takes is what its base takes: the merge adds
        # each low-rank product into a weight already there, and the adapter
        # file itself, tens of megabytes, is gone again once it has. At full
        # precision whatever the radio says, as the load itself is.
        snapshot = adapter_base_snapshot(snapshot)
        if snapshot is None:
            return None
        bits = None
    weight_bytes = snapshot_weight_bytes(snapshot)
    if weight_bytes is None:
        return None
    if kind == MLX_KIND:
        # Already packed: the file is read onto the device as it is, at the
        # precision the repo was converted to, whatever the radio says.
        return weight_bytes
    _architecture, checkpoint_dtype = _read_config(snapshot)
    if bits is None:
        return estimate_loaded_bytes(weight_bytes, checkpoint_dtype, load_dtype_name)
    # What the quantizer will leave on the device, not what the file holds:
    # the check is against the loaded size, and a 4-bit load of a checkpoint
    # the machine could not hold whole is the point.
    return estimate_quantized_bytes(
        weight_bytes, checkpoint_dtype, bits, _embedding_params(snapshot)
    )


def estimate_parameter_bytes(
    parameters: int, load_dtype_name: str, bits: int | None = None
) -> int:
    """Memory a model of ``parameters`` weights would take once loaded.

    For a model that is not on disk yet, where the hub's parameter count is
    all there is to go on. The checkpoint is assumed to be half precision,
    which is what current releases ship, and the embeddings are assumed to be
    quantized along with everything else because their share is not known
    from a search result: a quantized estimate made this way is a little
    lower than the model turns out to be.
    """

    half_bytes = parameters * 2
    if bits is None:
        return estimate_loaded_bytes(half_bytes, "float16", load_dtype_name)
    return estimate_quantized_bytes(half_bytes, "float16", bits, None)


def _read_config(snapshot: Path | None) -> tuple[str | None, str | None]:
    if snapshot is None:
        return None, None
    if is_pipeline(snapshot):
        # A pipeline has no architecture of its own: each component folder
        # keeps a config, and what names the model is the class the index
        # says to build. Its dtype is the components' business too, and they
        # need not agree, so none is reported.
        return pipeline_class(snapshot), None
    # The config is another repo's file, so nothing about its shape is
    # trusted: a config that is not an object, or an ``architectures`` that is
    # not a list, reads as an unknown architecture rather than an error.
    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, ValueError):
        return None, None
    if not isinstance(config, dict):
        return None, None
    architectures = config.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    dtype = config.get("dtype") or config.get("torch_dtype")
    quantization = mlx_runtime.mlx_quantization(config)
    if quantization is not None:
        # The dtype a converted repo carries is the one it was converted
        # from; what is on disk, and what loads, is the packed weight.
        dtype = f"{quantization['bits']}-bit MLX"
    return (
        architecture if isinstance(architecture, str) else None,
        dtype if isinstance(dtype, str) else None,
    )


# Where a config spells its hidden width when not as ``hidden_size``: GPT-2
# and its descendants, MPT and Falcon, Bloom. Transformers' own config classes
# resolve these, and are asked first; the list is the fallback for a config
# they cannot load.
HIDDEN_SIZE_ALIASES = ("hidden_size", "n_embd", "d_model", "hidden_dim", "model_dim")


def _embedding_params_from(config: Mapping[str, Any]) -> int | None:
    """Parameters in the embedding and output matrices, from a config's fields, or ``None``."""

    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        return _embedding_params_from(text_config)
    vocab = config.get("vocab_size")
    hidden = next(
        (config[name] for name in HIDDEN_SIZE_ALIASES if isinstance(config.get(name), int)),
        None,
    )
    if not isinstance(vocab, int) or hidden is None or vocab <= 0 or hidden <= 0:
        return None
    tied = config.get("tie_word_embeddings", False) is True
    return vocab * hidden * (1 if tied else 2)


def _embedding_params(snapshot: Path | None) -> int | None:
    """Parameters in the embedding and output matrices, or ``None`` when the config will not say.

    The quantizer leaves these matrices in half precision, so the estimate a
    quantized load is checked against needs their size. Transformers' config
    class for the architecture is asked first, since it knows the field the
    width is stored under; the raw file, read under the common aliases, is
    the fallback for an architecture it cannot load.
    """

    if snapshot is None:
        return None
    try:
        from transformers import AutoConfig

        loaded = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        get_text_config = getattr(loaded, "get_text_config", None)
        if callable(get_text_config):
            loaded = get_text_config()
        params = _embedding_params_from(
            {
                "vocab_size": getattr(loaded, "vocab_size", None),
                "hidden_size": getattr(loaded, "hidden_size", None),
                "tie_word_embeddings": getattr(loaded, "tie_word_embeddings", False),
            }
        )
        if params is not None:
            return params
    except Exception:  # noqa: BLE001 - any failure here falls through to the file
        pass
    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    return _embedding_params_from(config)


def _newest_write(folder: Path, snapshot: Path | None) -> float | None:
    newest: float | None = None
    candidates = []
    blobs = folder / "blobs"
    if blobs.is_dir():
        candidates.extend(blobs.iterdir())
    if snapshot is not None:
        candidates.extend(entry for entry in snapshot.rglob("*") if not entry.is_symlink())
    for entry in candidates:
        try:
            if not entry.is_file():
                continue
            stamp = entry.stat().st_mtime
        except OSError:
            continue
        if newest is None or stamp > newest:
            newest = stamp
    return newest


def cache_root(cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    return Path(cache_dir)


def list_cached_models(cache_dir: Path | None = None) -> list[CachedModel]:
    """Every model in the Hugging Face cache, newest download first.

    Reads only the disk. Folders whose name is not a model ID (the hub keeps
    datasets and spaces beside models, and other tools leave their own
    folders) and models with nothing on disk are left out. A cache that is
    not there, or that cannot be read, is an empty inventory rather than an
    error, and a folder that fails partway through is left out of it: the
    pane rescans after every model action, so it must never be taken down by
    a permission or a drive that has gone away.
    """

    root = cache_root(cache_dir)
    try:
        folders = list(root.iterdir())
    except OSError:
        return []
    models: list[CachedModel] = []
    for folder in folders:
        try:
            model = _cached_model(folder, root)
        except OSError:
            continue
        if model is not None:
            models.append(model)
    models.sort(key=lambda entry: (-(entry.updated or 0), entry.model_id))
    return models


def _cached_model(folder: Path, root: Path) -> CachedModel | None:
    """The inventory entry for one cache folder, or None if it holds no model."""

    if not folder.is_dir() or not folder.name.startswith(MODEL_FOLDER_PREFIX):
        return None
    organization, _, name = folder.name[len(MODEL_FOLDER_PREFIX) :].partition("--")
    try:
        model_id = validate_model_id(f"{organization}/{name}")
    except ValueError:
        return None
    status = cache_status(model_id, root)
    if not status.present:
        return None
    snapshot = snapshot_folder(folder)
    commit = snapshot.name if snapshot is not None else None
    files = (
        sum(1 for entry in snapshot.rglob("*") if entry.is_file())
        if snapshot is not None
        else 0
    )
    architecture, dtype = _read_config(snapshot)
    return CachedModel(
        model_id=model_id,
        status=status,
        files=files,
        commit=commit,
        updated=_newest_write(folder, snapshot),
        architecture=architecture,
        dtype=dtype,
        path=folder,
        disk_bytes=folder_bytes(folder),
    )


# The orders My Models can be listed in. "Newest first" is the scan's own
# order; the rest re-sort the same entries, ties broken by ID so the list is
# stable across rescans.
MODEL_SORT_ORDERS = ("Newest first", "Name", "Largest first", "Smallest first")
DEFAULT_MODEL_SORT = MODEL_SORT_ORDERS[0]

_SORT_KEYS: dict[str, Callable[[CachedModel], tuple]] = {
    "Newest first": lambda entry: (-(entry.updated or 0), entry.model_id),
    "Name": lambda entry: (entry.model_id.lower(), entry.model_id),
    "Largest first": lambda entry: (-entry.size_bytes, entry.model_id),
    "Smallest first": lambda entry: (entry.size_bytes, entry.model_id),
}


def sort_cached_models(models: list[CachedModel], order: str | None) -> list[CachedModel]:
    """``models`` in one of :data:`MODEL_SORT_ORDERS`; an unknown order is the default."""

    key = _SORT_KEYS.get(order or "", _SORT_KEYS[DEFAULT_MODEL_SORT])
    return sorted(models, key=key)




class ModelInUse(RuntimeError):
    """A cached model's files cannot be removed right now.

    The subclasses say why, so the interface can tell the reader what to do:
    unload the model, wait for its download, or wait for the model to go idle.
    """


class ModelLoaded(ModelInUse):
    """The model is the one in memory."""


class ModelDownloading(ModelInUse):
    """A download of the model is under way, in this process or another."""


class ModelBusy(ModelInUse):
    """The model lock is held: a load, generation, scoring, or inspection is running."""


def hub_lock_held(root: Path, folder_name: str) -> bool:
    """Whether another process holds one of the hub's locks for this repo.

    ``huggingface_hub`` takes a ``filelock`` on ``.locks/<repo>/<etag>.lock``
    for each file it is writing, and Transformers loads through the same
    library. Each lock is tried without waiting and let go at once: taking
    one is the only way to ask, since a lock file exists whether or not
    anyone holds it. A lock this process cannot open counts as held.
    """

    locks = root / ".locks" / folder_name
    if not locks.is_dir():
        return False
    from filelock import FileLock, Timeout

    for path in locks.iterdir():
        if not path.is_file():
            continue
        lock = FileLock(str(path))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return True
        except OSError:
            return True
        else:
            lock.release()
    return False


def remove_cached_model(model_id: str, cache_dir: Path | None = None) -> int:
    """Delete everything the cache holds for ``model_id``; return the bytes freed.

    Removes the model's ``models--org--name`` folder. The lock folder the hub
    keeps beside it under ``.locks`` is left alone, as the hub's own
    ``delete_revisions`` leaves it: a process in another window may be
    waiting on one of those files, and deleting a lock someone holds lets a
    second writer in beside them. Before deleting, every lock in that folder
    is tried: one held by another process means a download or load is
    touching these files right now, and the removal is refused with
    :class:`ModelDownloading` rather than pulling them out from under it.
    The locks in this process are the manager's business, see
    :meth:`ModelManager.remove`.

    The size is measured over the whole folder before deletion, so it counts
    every revision the folder held, not just the ``main`` snapshot. A model
    with nothing on disk raises ``FileNotFoundError``; a folder that cannot
    be deleted raises the ``OSError`` that stopped it, with whatever was
    already removed gone.
    """

    checked_id = validate_model_id(model_id)
    root = cache_root(cache_dir)
    folder = cache_folder(checked_id, root)
    if not folder.is_dir():
        raise FileNotFoundError(f"Nothing for {checked_id} is in the cache at {root}.")
    if hub_lock_held(root, folder.name):
        raise ModelDownloading(
            f"{checked_id} is being downloaded or loaded by another process."
        )
    freed = folder_bytes(folder)
    shutil.rmtree(folder)
    return freed
