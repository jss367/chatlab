"""Download, load, and inspect Hugging Face causal language models."""

from __future__ import annotations

import contextlib
import functools
import gc
import json
import os
import re
import shutil
import subprocess
import sys
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

# Refuse a generation prefix large enough to make replay itself an
# unexpectedly expensive operation. The response-length control already tops
# out here, so a branch cannot paste an unbounded second response around that
# control even when the model advertises a much larger context window.
GENERATION_PREFILL_TOKEN_LIMIT = 8192

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

# Memory kept back when deciding whether a model fits: the application, the
# key-value cache a conversation grows, and the rest of the system all need
# room beside the weights. On Apple silicon the GPU draws from the same pool,
# so a model that "fits" with nothing to spare freezes the whole machine
# instead of failing.
MEMORY_HEADROOM_BYTES = 4 * 1024**3

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

# Share of Metal's recommended working set that PyTorch may allocate before it
# raises an out-of-memory error instead of letting macOS page the machine into
# a freeze. PyTorch's own default is 1.7, well past physical memory.
MPS_MEMORY_FRACTION_ENV = "CHATLAB_MPS_MEMORY_FRACTION"
DEFAULT_MPS_MEMORY_FRACTION = 1.0
# PyTorch reads this itself; when the user has set it, their choice stands.
TORCH_MPS_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"


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
    shards, has finished blobs but no model. ``unsupported`` says the snapshot
    is whole but is not a Transformers language model at all (a diffusers
    pipeline, a CTranslate2 or ONNX export, a folder of SAE weights): nothing
    is missing, ChatLab just cannot load it. ``cached_bytes`` counts finished
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
    unsupported: bool = False

    @property
    def present(self) -> bool:
        return self.cached_bytes > 0 or self.partial_files > 0

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
    returns ``snapshots/<commit>`` whether or not every file is in it.
    """

    ref = folder / "refs" / revision
    if not ref.is_file():
        return None
    snapshot = folder / "snapshots" / ref.read_text().strip()
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

    A diffusers pipeline announces itself with ``model_index.json``. Otherwise
    the evidence is a weight file where ``from_pretrained`` would never look:
    at the root under a name that is not a checkpoint or a shard (CTranslate2's
    ``model.bin``, an ``model.onnx``), or in a subfolder. When the root
    ``config.json`` is a Transformers one, only a foreign format in a
    subfolder counts, since such repos often ship extras like
    ``original/consolidated.00.pth`` beside the checkpoint they are missing.
    """

    if (snapshot / "model_index.json").is_file():
        return True
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


def judge_snapshot(snapshot: Path | None) -> tuple[tuple[str, ...], bool]:
    """``(missing_files, unsupported)`` for what the snapshot holds.

    ``missing_files`` are the files ``from_pretrained`` needs before it can
    load: only the config and the weights are checked, since which tokenizer
    files a repo ships varies too much to know from the outside, and a wrong
    "incomplete" verdict on a good cache would be worse than a generic load
    error. A snapshot with no Transformers checkpoint at its root but weights
    laid out for another framework (diffusers, CTranslate2, ONNX, SAE
    weights) is not a cut-off download and is ``unsupported`` instead, with
    nothing reported missing. Absence alone is never that verdict: a snapshot
    holding only a tokenizer, or only a config, is incomplete.
    """

    if snapshot is None:
        return ("config.json", MODEL_WEIGHTS), False
    has_checkpoint = any(
        (snapshot / name).is_file() for pair in WEIGHT_FORMATS for name in pair
    )
    if not has_checkpoint and foreign_weights(
        snapshot, transformers_config=is_transformers_config(snapshot / "config.json")
    ):
        return (), True
    return missing_files(snapshot), False


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
    missing, unsupported = judge_snapshot(snapshot)
    return CacheStatus(cached, partial_files, partial_bytes, missing, unsupported)


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


class InsufficientMemoryError(RuntimeError):
    """A model would not fit in memory, judged before any weight is read."""


class OutOfMemoryError(RuntimeError):
    """The device ran out of memory partway through a run."""


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


def system_memory() -> tuple[int | None, int | None]:
    """Total and currently available physical memory in bytes, where known.

    macOS reports availability through ``memory_pressure``, which accounts for
    compression and file cache the way the kernel does; ``vm_stat`` is the
    fallback. Linux reads ``MemAvailable``. Either figure is ``None`` when
    the platform offers nothing usable.
    """

    total: int | None = None
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        total = None
    available: int | None = None
    if sys.platform == "darwin":
        available = _darwin_available_memory(total)
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            available = None
    return total, available


def _run_quietly(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _darwin_available_memory(total: int | None) -> int | None:
    match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%", _run_quietly(["memory_pressure"])
    )
    if match and total is not None:
        return total * int(match.group(1)) // 100
    output = _run_quietly(["vm_stat"])
    size = re.search(r"page size of (\d+) bytes", output)
    if not size:
        return None
    pages = 0
    for name in ("free", "inactive", "speculative", "purgeable"):
        found = re.search(rf"Pages {name}:\s*(\d+)", output)
        if found:
            pages += int(found.group(1))
    return pages * int(size.group(1)) if pages else None


def format_memory(count: int) -> str:
    """Render a memory size the way the machine's own specs do: ``48.0 GB``."""

    return f"{count / 1024**3:.1f} GB"


def cuda_memory(torch=None) -> tuple[int | None, int | None]:
    """Total and currently free CUDA memory in bytes, summed across devices.

    ``device_map="auto"`` spreads a model over every visible device, so the
    sum is the figure that matters. Both are ``None`` when no device answers.
    """

    if torch is None:
        import torch
    try:
        count = int(torch.cuda.device_count())
        figures = [torch.cuda.mem_get_info(index) for index in range(count)]
    except (RuntimeError, AttributeError, ValueError, TypeError):
        return None, None
    if not figures:
        return None, None
    total = sum(int(device_total) for _free, device_total in figures)
    free = sum(int(device_free) for device_free, _total in figures)
    return total, free


def _sum_known(*figures: int | None) -> int | None:
    known = [figure for figure in figures if figure is not None]
    return sum(known) if known else None


def offload_pool(
    gpu: tuple[int | None, int | None], host: tuple[int | None, int | None]
) -> tuple[int | None, int | None]:
    """Total and free memory a CUDA load can spread over: the cards plus the host.

    ``device_map="auto"`` fills the graphics cards first and places whatever
    is left on the CPU, so a model that outgrows the cards still loads when
    the machine's own memory can hold the rest. A side that reports nothing
    is left out of the sum rather than counted as empty; both ``None`` when
    neither side answers.
    """

    return _sum_known(gpu[0], host[0]), _sum_known(gpu[1], host[1])


def check_memory_for_load(
    model_id: str,
    estimated_bytes: int,
    total: int | None,
    available: int | None,
    headroom: int = MEMORY_HEADROOM_BYTES,
    pool: str = "this machine",
) -> None:
    """Refuse a load that would not leave ``headroom`` beside the weights.

    ``pool`` names where the figures come from in the message: the machine's
    own memory, or the GPU plus the machine when the weights may spread over
    both.
    """

    needed = estimated_bytes + headroom
    if total is not None and needed > total:
        raise InsufficientMemoryError(
            f"{model_id} needs about {format_memory(estimated_bytes)} of memory plus "
            f"{format_memory(headroom)} of working room, and {pool} has "
            f"{format_memory(total)} in total. Choose a smaller model."
        )
    if available is not None and needed > available:
        raise InsufficientMemoryError(
            f"{model_id} needs about {format_memory(estimated_bytes)} of memory plus "
            f"{format_memory(headroom)} of working room, but only "
            f"{format_memory(available)} is free right now. Close other "
            "applications and try again."
        )


def mps_memory_fraction() -> float | None:
    """The Metal allocation cap to apply, or ``None`` to leave PyTorch's own.

    ``CHATLAB_MPS_MEMORY_FRACTION`` overrides the default; a value PyTorch
    would reject, or a user-set ``PYTORCH_MPS_HIGH_WATERMARK_RATIO``, leaves
    the allocator alone.
    """

    if os.environ.get(TORCH_MPS_WATERMARK_ENV):
        return None
    raw = os.environ.get(MPS_MEMORY_FRACTION_ENV)
    if raw is None:
        return DEFAULT_MPS_MEMORY_FRACTION
    try:
        fraction = float(raw)
    except ValueError:
        return DEFAULT_MPS_MEMORY_FRACTION
    return fraction if 0 < fraction <= 2 else None


def is_out_of_memory_error(error: BaseException) -> bool:
    """Whether a backend raised for lack of device or host memory."""

    if isinstance(error, (MemoryError, OutOfMemoryError)):
        return True
    text = str(error).lower()
    return "out of memory" in text or "insufficient memory" in text


def out_of_memory_message(error: BaseException) -> str:
    """One readable sentence for a backend's out-of-memory failure."""

    return (
        "The model ran out of memory. Shorten the conversation or the text, "
        "lower the response length, or load a smaller model. "
        f"({str(error).splitlines()[0][:200]})"
    )


def _reraise_out_of_memory(error: BaseException) -> None:
    """Re-raise a backend failure, as :class:`OutOfMemoryError` when that is what it was."""

    if isinstance(error, OutOfMemoryError) or not is_out_of_memory_error(error):
        raise error
    raise OutOfMemoryError(out_of_memory_message(error)) from error


def _guards_device_memory(method):
    """Turn a run's out-of-memory failure into :class:`OutOfMemoryError`, and
    hand cached device memory back after ``method``, whatever its outcome.

    The caching allocator keeps every block a run freed, so a long prompt
    stays paid for until the next one. Returning it after each run keeps the
    process at the model's own size between requests.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (RuntimeError, MemoryError) as error:
            _reraise_out_of_memory(error)
        finally:
            self._release_device_cache()

    return wrapper


def _read_config(snapshot: Path | None) -> tuple[str | None, str | None]:
    if snapshot is None:
        return None, None
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
    return (
        architecture if isinstance(architecture, str) else None,
        dtype if isinstance(dtype, str) else None,
    )


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


# How many hub search results are shown. The hub sorts them by downloads, so
# the ones a reader is likely to want come first, and a longer list would only
# push the search box off the pane.
SEARCH_LIMIT = 20


@dataclass(frozen=True)
class HubModel:
    """What the hub says about one model, as much as a search result carries."""

    model_id: str
    parameters: int | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library: str | None = None
    gated: bool | str = False
    last_modified: str | None = None
    license: str | None = None


def search_hub_models(
    query: str, hf_token: str | None = None, limit: int = SEARCH_LIMIT
) -> list[HubModel]:
    """Search the hub for text-generation models Transformers can load.

    The filter is the one the application itself imposes: only causal
    language models with built-in Transformers support load here, so results
    from other libraries would be dead ends. Sorted by recent downloads.
    """

    from huggingface_hub import HfApi

    cleaned = query.strip()
    if not cleaned:
        return []
    token = hf_token.strip() if hf_token and hf_token.strip() else None
    found = HfApi().list_models(
        search=cleaned,
        pipeline_tag="text-generation",
        filter="transformers",
        sort="downloads",
        limit=limit,
        expand=[
            "downloads",
            "likes",
            "pipeline_tag",
            "library_name",
            "lastModified",
            "safetensors",
            "gated",
            "tags",
        ],
        token=token,
    )
    results = []
    for info in found:
        safetensors = getattr(info, "safetensors", None)
        parameters = getattr(safetensors, "total", None) if safetensors else None
        tags = getattr(info, "tags", None) or []
        licenses = [tag[len("license:") :] for tag in tags if tag.startswith("license:")]
        modified = getattr(info, "last_modified", None)
        results.append(
            HubModel(
                model_id=info.id,
                parameters=parameters,
                downloads=getattr(info, "downloads", None),
                likes=getattr(info, "likes", None),
                pipeline_tag=getattr(info, "pipeline_tag", None),
                library=getattr(info, "library_name", None),
                gated=getattr(info, "gated", False) or False,
                last_modified=modified.date().isoformat() if modified else None,
                license=licenses[0] if licenses else None,
            )
        )
    return results


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


def generation_prefill_token_limit(model) -> int:
    """The most prompt-plus-replayed tokens accepted by one generation.

    A learned positional table is a hard correctness limit. The flat cap is a
    separate application guard: typed branches can otherwise turn an
    unrestricted textbox into an arbitrarily large scored prefill while the
    model lock is held.
    """

    window = model_position_limit(model)
    return (
        GENERATION_PREFILL_TOKEN_LIMIT
        if window is None
        else min(GENERATION_PREFILL_TOKEN_LIMIT, window)
    )


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

    model_id: str | None = None
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
        # The models that loads are bringing in right now, in the order the
        # loads arrived. ``model_id`` is cleared for the whole of a load and
        # set only once the weights are in, so on its own it says nothing
        # about the minutes in between; anything that must not touch a
        # model's files while it is being read (a redownload, say) asks this
        # as well.
        #
        # A list rather than one slot because two loads really can be under
        # way at once. Gradio limits concurrency per event handler, so
        # "Load cached" and "Download and load" each get a worker of their
        # own, and a second browser tab gets more still; the load that does
        # not win the model lock sits waiting for it. With a single slot the
        # first load to finish would clear the marker the second one was
        # still relying on, and the chat badge would go back to saying that
        # nothing was loaded in the middle of a minutes-long load.
        self._pending_loads: list[str] = []
        # Guards _pending_loads. A load runs on whichever worker thread
        # Gradio handed the click to, so adding and removing entries must not
        # interleave with each other or with a reader building the badge.
        self._pending_loads_lock = threading.Lock()
        # Downloads under way right now, by model ID, so a second request for
        # the same model can follow the first instead of racing it for the
        # same files.
        self.active_downloads: dict[str, DownloadProgress] = {}
        # Guards active_downloads, so that "is anyone fetching this?" and "then
        # I am" happen as one step: two handlers asking at the same instant
        # must come away with one download between them, not one each.
        self._downloads_lock = threading.Lock()
        # A generation holds this lock across streaming yields. Gradio may run
        # the next step (or close the generator) on a different worker thread,
        # so this cannot be an RLock: its owner check would reject that second
        # thread's release and leave the model permanently wedged. A plain Lock
        # still excludes loads, unloads, scoring, and inspection, but permits
        # the worker that resumes the stream to release it.
        self._lock = threading.Lock()
        # A separate flag records "a generation is running right now" without
        # making callers contend for the model lock just to ask.
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
    def loading_id(self) -> str | None:
        """Name the model a load is bringing in right now, or None.

        The oldest load still running, when there are several: that is the
        one holding the model lock, or next in line for it, so it is the
        load actually reading weights while the others wait their turn.
        A later load takes over the name once the earlier one is done.
        """

        with self._pending_loads_lock:
            return self._pending_loads[0] if self._pending_loads else None

    def is_loading(self, model_id: str) -> bool:
        """Say whether any load running right now is reading ``model_id``.

        Asked instead of comparing against :attr:`loading_id`, which names
        only one of them: a load waiting its turn for the model lock is
        still going to read that model's files, so anything that would
        disturb them has to be refused on its behalf too.
        """

        with self._pending_loads_lock:
            return model_id in self._pending_loads

    @contextlib.contextmanager
    def _loading(self, model_id: str) -> Iterator[None]:
        """Count ``model_id`` as being loaded for the length of the block.

        Each load adds its own entry and takes its own entry away again, so
        one load finishing cannot cancel out another that is still running.
        """

        with self._pending_loads_lock:
            self._pending_loads.append(model_id)
        try:
            yield
        finally:
            with self._pending_loads_lock:
                # remove() drops one matching entry, which is the right thing
                # when two loads of the same model overlap: each undoes its
                # own append and the second stays counted.
                self._pending_loads.remove(model_id)

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

        checked_id = validate_model_id(model_id)
        # Recorded before waiting for the lock, not after: a load queued
        # behind a long generation, or behind another load, is a load under
        # way for the whole wait.
        with self._loading(checked_id), self._lock:
            return self._load_locked(checked_id, local_path, torch)

    def _load_locked(self, model_id: str, local_path: Path, torch) -> str:
        """Bring ``model_id`` in from ``local_path`` while the caller holds ``_lock``."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._unload_locked(torch)
        if torch.cuda.is_available():
            backend = "cuda"
            dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif torch.backends.mps.is_available():
            backend = "mps"
            dtype = torch.float16
        else:
            backend = "cpu"
            dtype = torch.float32
        self._check_memory(
            model_id, local_path, str(dtype).replace("torch.", ""), backend
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

        try:
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif torch.backends.mps.is_available():
                self._cap_mps_memory(torch)
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to("mps")
                device_name = "Apple Metal (MPS)"
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    local_files_only=True,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                device_name = "CPU"
        except (RuntimeError, MemoryError) as error:
            self._release_device_cache(torch)
            if is_out_of_memory_error(error):
                raise OutOfMemoryError(
                    f"{model_id.strip()} did not fit in memory. Close other "
                    "applications or choose a smaller model. "
                    f"({str(error).splitlines()[0][:200]})"
                ) from error
            raise

        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.local_path = local_path
        self.device_name = device_name
        self.load_count += 1
        return device_name

    def unload(self) -> None:
        import torch

        with self._lock:
            self._unload_locked(torch)

    def remove(self, model_id: str, cache_dir: Path | None = None) -> int:
        """Delete ``model_id``'s cache folder, unless the manager is using it.

        Removal is serialized with everything else that touches the files.
        The model lock is taken for the whole deletion, so a load that is
        still reading the folder (``model_id`` is assigned only once
        ``from_pretrained`` returns) cannot have its files pulled from under
        it, and no load can start until the folder is gone. The downloads
        lock is held too, so a download cannot be reserved for the model
        between the check and the deletion. Looking at ``model_id`` and
        ``active_downloads`` first and deleting afterwards would leave exactly
        that gap: the interface runs its handlers on several workers.

        The model lock is never waited for. It is held across a whole
        generation, and a handler blocked on it would tie up a worker for as
        long as the reply takes, so a busy manager raises :class:`ModelBusy`
        and the reader tries again when the model is idle.
        """

        checked_id = validate_model_id(model_id)
        if not self._lock.acquire(blocking=False):
            raise ModelBusy(
                f"{checked_id} cannot be removed while a model is loading, "
                "generating, scoring, or being inspected."
            )
        try:
            if self.model_id == checked_id:
                raise ModelLoaded(f"{checked_id} is loaded in memory.")
            with self._downloads_lock:
                if checked_id in self.active_downloads:
                    raise ModelDownloading(f"{checked_id} is being downloaded.")
                return remove_cached_model(checked_id, cache_dir)
        finally:
            self._lock.release()

    def _unload_locked(self, torch) -> None:
        """Clear the loaded model while the caller holds ``_lock``."""

        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.local_path = None
        self.device_name = None
        gc.collect()
        self._release_device_cache(torch)

    @staticmethod
    def _release_device_cache(torch=None) -> None:
        """Return the allocator's unused blocks to the device."""

        if torch is None:
            import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def _check_memory(
        model_id: str, local_path: Path, load_dtype: str, backend: str
    ) -> None:
        """Refuse a load that cannot fit, before any weight is read.

        On CUDA the weights fill the graphics cards and ``device_map="auto"``
        places the rest on the CPU, so the cards plus the machine's memory is
        what must fit; on Metal the GPU shares the machine's memory, and on
        the CPU it is the machine's memory outright. A snapshot whose weights
        cannot be measured is let through: the loader will give its own, more
        specific error.
        """

        weight_bytes = snapshot_weight_bytes(local_path)
        if weight_bytes is None:
            return
        _architecture, checkpoint_dtype = _read_config(local_path)
        estimated = estimate_loaded_bytes(weight_bytes, checkpoint_dtype, load_dtype)
        if backend == "cuda":
            total, available = offload_pool(cuda_memory(), system_memory())
            pool = "the GPU plus this machine"
        else:
            total, available = system_memory()
            pool = "this machine"
        check_memory_for_load(
            validate_model_id(model_id), estimated, total, available, pool=pool
        )

    @staticmethod
    def _cap_mps_memory(torch) -> None:
        """Make Metal allocations fail past the recommended working set.

        Unified memory means the GPU and everything else share one pool, and
        PyTorch's default ceiling lies well beyond it. With the cap, a model
        or conversation that outgrows the machine raises an error the
        interface can show; without it, macOS pages until it freezes.
        """

        fraction = mps_memory_fraction()
        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if fraction is None or setter is None:
            return
        try:
            setter(fraction)
        except (RuntimeError, ValueError, TypeError):
            pass

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
        hidden = self._hidden_token_ids()
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

        # The standalone encoding is how a BPE tokenizer with the space inside
        # the token normally wants the text. A context-sensitive tokenizer can
        # instead need a suffix of the joint encoding. The model may have
        # sampled a noncanonical spelling of ``kept_text``, so the joint
        # encoding need not begin with ``kept`` even though its boundary suffix
        # can follow those unchanged ids exactly.
        #
        # Locate that suffix from the tokenizer's character offsets when they
        # exist, or from the bounded seam search used by text scoring for a
        # slow tokenizer. Trying every suffix looks harmless but is quadratic:
        # each candidate decodes the whole kept prefix plus a progressively
        # longer tail, all while the model lock is held. The seam can be off by
        # one token when one piece crosses it, so validate the boundary and its
        # two neighbours. If that seam was only guessed, also bisect decoded
        # joint prefixes: a continuation token may not decode correctly by
        # itself, but the prefix can still identify its exact start. Candidate
        # decoding stays strictly bounded while the final exact decode remains
        # the authority.
        standalone = self._encode_plain(text)
        split = split_context_and_text(
            self.tokenizer, kept_text, text, add_special_tokens=False
        )
        joint = split.context_ids + split.text_ids
        boundary = len(split.context_ids)

        decoded_boundary = (
            split.decoded_prefix_end if not split.seam_verified else None
        )

        def candidates() -> Iterator[list[int]]:
            yield standalone
            aligned_start: int | None = None
            if (
                len(joint) > len(visible_kept)
                and joint[: len(visible_kept)] == visible_kept
            ):
                aligned_start = len(visible_kept)
                aligned = joint[aligned_start:]
                if aligned != standalone:
                    yield aligned
            starts = {
                start
                for start in (
                    boundary - 1,
                    boundary,
                    boundary + 1,
                    decoded_boundary,
                )
                if start is not None
                if 0 <= start < len(joint)
            }
            for start in sorted(starts, reverse=True):
                candidate = joint[start:]
                if start != aligned_start and candidate != standalone:
                    yield candidate

        if not standalone and not joint:
            raise ValueError("The replacement text did not produce any tokens.")
        stop_ids = self._stop_token_ids()
        hidden_ids = hidden - stop_ids
        matched_embedded_stop = False
        matched_hidden = False
        for token_ids in candidates():
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

    def validate_generation_prefix(
        self,
        messages: list[dict],
        forced_ids: Sequence[int],
        *,
        max_new_tokens: int,
        load_id: str | None = None,
    ) -> None:
        """Refuse an oversized generation before a stream mutates UI state.

        A typed branch calls this after encoding but before entering the reply
        stream. The expected load is checked under the same model lock as the
        prompt tokenization, so a concurrent reload cannot validate one
        model's token IDs with another model's tokenizer. A learned-position
        model also needs room to feed back all but the last requested sampled
        token; the first comes from the prefill's final logits.
        """

        with self._lock:
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )
            prompt_ids, _reasoning_prefilled = self._prompt_token_ids(messages)
            self._validate_generation_prefix_length(
                prompt_ids,
                forced_ids,
                max_new_tokens=max_new_tokens,
            )

    def _validate_generation_prefix_length(
        self,
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> None:
        """Check a tokenized generation and its continuation capacity."""

        assert self.model is not None
        total = len(prompt_ids) + len(forced_ids)
        limit = generation_prefill_token_limit(self.model)
        window = model_position_limit(self.model)
        if total > limit:
            ceiling = (
                f"the {limit:,} positions this model can attend to"
                if window is not None and window <= GENERATION_PREFILL_TOKEN_LIMIT
                else (
                    f"the {GENERATION_PREFILL_TOKEN_LIMIT:,} token limit for a "
                    "generation prefix"
                )
            )
            raise ValueError(
                f"The prompt and replayed response are {total:,} tokens, above "
                f"{ceiling}. Shorten the conversation or replacement."
            )

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

    def _encode_plain(self, text: str) -> list[int]:
        """Token ids for ``text`` alone: no special tokens, no chat template."""

        assert self.tokenizer is not None
        encoded = self.tokenizer(text, add_special_tokens=False)
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
        automatic_reasoning_close_tokens: int = 0,
        literal_text_ranges: Sequence[tuple[int, int]] = (),
        load_id: str | None = None,
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

        ``load_id`` names the load ``forced_ids`` came from (see
        :attr:`load_id`). It is compared under the model lock, before any token
        is fed, so a load that finished after the caller looked is refused with
        :class:`ModelChanged` rather than replaying one model's token IDs
        through another.
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
                automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                literal_text_ranges=literal_text_ranges,
                load_id=load_id,
            )
        except (RuntimeError, MemoryError) as error:
            _reraise_out_of_memory(error)
        finally:
            if reserved:
                self.release_generation()
            # The key-value cache of this response is the largest thing a run
            # allocates; give it back rather than hold it until the next one.
            self._release_device_cache()

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
        automatic_reasoning_close_tokens: int = 0,
        literal_text_ranges: Sequence[tuple[int, int]] = (),
        load_id: str | None = None,
    ) -> Iterator[GenerationUpdate]:
        import torch

        with self._lock, torch.inference_mode():
            if not self.loaded:
                raise RuntimeError("Download and load a model before chatting.")
            if load_id is not None and load_id != self.load_id:
                raise ModelChanged(
                    "The model has been reloaded since these tokens were produced."
                )

            assert self.model is not None
            assert self.tokenizer is not None
            model = self.model
            tokenizer = self.tokenizer
            # Read here, under the lock, alongside the weights: this is the
            # only place the two are guaranteed to agree, which is what makes
            # the stamp on each update worth trusting.
            model_id = self.model_id
            producing_load_id = self.load_id
            assert producing_load_id is not None
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

            normalized_literal_ranges: list[tuple[int, int]] = []
            for raw_start, raw_end in literal_text_ranges:
                start = max(0, min(int(raw_start), len(forced)))
                end = max(start, min(int(raw_end), len(forced)))
                if start < end:
                    normalized_literal_ranges.append((start, end))
            normalized_literal_ranges.sort()
            literal_ranges: list[tuple[int, int]] = []
            for start, end in normalized_literal_ranges:
                if literal_ranges and start <= literal_ranges[-1][1]:
                    previous_start, previous_end = literal_ranges[-1]
                    literal_ranges[-1] = (previous_start, max(previous_end, end))
                else:
                    literal_ranges.append((start, end))

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
            for metric in metrics[:automatic_reasoning_close_tokens]:
                metric["automatic_reasoning_close"] = True
            for start, end in literal_ranges:
                for metric in metrics[start:end]:
                    metric["literal_text"] = True
            prompt_note = ""
            if analyze_prompt and score_from > 1:
                prompt_note = (
                    f"Only the most recent {PROMPT_SCORE_LIMIT:,} of "
                    f"{len(prompt_ids):,} prompt tokens were scored."
                )

            rng = np.random.default_rng(int(seed))
            decoder = IncrementalDecoder(tokenizer, self._hidden_token_ids())
            literal_prefill_text = ""
            literal_boundaries = {
                boundary for span in literal_ranges for boundary in span
            }
            boundary_text = {0: ""}
            for index, token_id in enumerate(forced):
                decoder.push(
                    token_id, force_visible=index < literal_prefill_tokens
                )
                if (
                    answer_prefill
                    and reasoning_prefilled
                    and not automatic_reasoning_close_tokens
                    and decoder.stable_text.startswith(f"{THINK_CLOSE}\n\n")
                ):
                    # The last token can straddle the boundary and include the
                    # beginning of the reader's prefill. It still cannot be
                    # replaced independently: doing so would remove part of
                    # the close and leave the continuation inside reasoning.
                    automatic_reasoning_close_tokens = index + 1
                    for metric in metrics[:automatic_reasoning_close_tokens]:
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
                for start, end in literal_ranges
                if stable_length(start) < stable_length(end)
            )
            limit = len(forced) + int(max_new_tokens)
            pending_tokens = 0
            last_yield = time.monotonic()

            if forced:
                yield GenerationUpdate(
                    text=decoder.text,
                    metrics=metrics,
                    load_id=producing_load_id,
                    prompt_metrics=prompt_metrics,
                    prompt_note=prompt_note,
                    reasoning_prefilled=reasoning_prefilled,
                    forced_prefix_tokens=len(forced),
                    literal_prefill_text=literal_prefill_text,
                    literal_text_spans=literal_text_spans,
                    prompt_ids=tuple(prompt_ids),
                    model_id=model_id,
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
                        load_id=producing_load_id,
                        prompt_metrics=prompt_metrics,
                        prompt_note=prompt_note,
                        reasoning_prefilled=reasoning_prefilled,
                        forced_prefix_tokens=len(forced),
                        literal_prefill_text=literal_prefill_text,
                        literal_text_spans=literal_text_spans,
                        prompt_ids=tuple(prompt_ids),
                        model_id=model_id,
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

    @_guards_device_memory
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

    @_guards_device_memory
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
