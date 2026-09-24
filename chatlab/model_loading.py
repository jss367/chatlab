"""Reading a model's weights into memory, and taking them out again.

A load checks that the model fits, reads a text model through
``AutoModelForCausalLM`` (merging a LoRA adapter into its base when the
repository is one), an MLX checkpoint through :mod:`mlx_runtime`, or an
image pipeline through diffusers, and records what is in memory as one
reading. The claims that decide whether a load may start at all stay on
:class:`model_runtime.ModelManager`.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import logging
import re
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from chatlab import adapters
from chatlab import device_memory
from chatlab import mlx_runtime
from chatlab import model_cache
from chatlab.device_memory import (
    InsufficientMemoryError,
    OutOfMemoryError,
    dtype_name,
    first_line,
    is_out_of_memory_error,
    load_dtype,
    load_out_of_memory_message,
    memory_note,
    mps_ceiling,
    mps_memory_fraction,
    recommended_mps_memory,
    weights_note,
)
from chatlab.model_cache import (
    IMAGE_KIND,
    MLX_KIND,
    QUANTIZATION_GROUP_SIZE,
    TEXT_KIND,
    adapter_base_snapshot,
    estimate_snapshot_bytes,
    is_adapter_snapshot,
    judge_snapshot,
    mlx_snapshot_bits,
    validate_model_id,
)
from chatlab.progress_bars import LoadProgress

logger = logging.getLogger(__name__)


# The bit width each quantized precision packs a linear weight into, in groups
# of model_cache.QUANTIZATION_GROUP_SIZE weights that share one scale and one
# bias. Transformers' Metal quantizer does the packing on the way in and runs
# the fused dequantize-and-multiply kernels from the Hub, so this is Apple
# Metal only: on another device the weights are loaded whole and the choice
# noted.
QUANTIZED_BITS = {"8-bit": 8, "4-bit": 4}
# The first Transformers release that ships MetalConfig. Older releases still
# run everything else, so requirements.txt keeps its lower floor and a
# quantized load on one of them is refused by name.
METAL_QUANTIZATION_TRANSFORMERS = "5.3"


# Transformers builds a tokenizer straight from a repo's ``tokenizer.json``,
# and from anything else - a SentencePiece vocabulary, a tiktoken one - only
# by converting it, which it does through packages it does not itself
# require. Each name here is the package to install; the module beside it is
# what Transformers looks for, which is not the same string for protobuf.
# SentencePiece conversion reads the vocabulary through sentencepiece and its
# wrapper through protobuf, so that path wants both.
TOKENIZER_CONVERSION_PACKAGES = {
    "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf",
    "tiktoken": "tiktoken",
}

# The opening of the failure Transformers raises when it could build no
# tokenizer at all. Its own message lists the three ways one could have been
# built and leaves the reader to work out which package would have helped.
TOKENIZER_BACKEND_FAILURE = "Couldn't instantiate the backend tokenizer"


def missing_tokenizer_packages() -> tuple[str, ...]:
    """The tokenizer-conversion packages this installation is without."""

    absent = []
    for package, module in TOKENIZER_CONVERSION_PACKAGES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            absent.append(package)
    return tuple(absent)


# A vocabulary this name is read as a tiktoken one outright. Any other
# ``.model`` file is tried as SentencePiece, and a tiktoken vocabulary under
# another name is reached by that attempt failing - which is a different
# failure with its own message naming tiktoken, not the one answered here.
TIKTOKEN_VOCABULARY = "tiktoken.model"


def tokenizer_vocabularies(snapshot: Path | None) -> tuple[Path, ...]:
    """The vocabulary files a snapshot ships in place of a ``tokenizer.json``.

    A pipeline keeps each of its tokenizers in a folder of its own, so one
    level down counts as well as the top.
    """

    if snapshot is None:
        return ()
    try:
        found = sorted({*snapshot.glob("*.model"), *snapshot.glob("*/*.model")})
    except OSError:
        return ()
    return tuple(path for path in found if path.is_file())


def tokenizer_packages_for(vocabularies: Iterable[Path]) -> tuple[str, ...]:
    """The packages that would convert these vocabularies, in install order."""

    needed: list[str] = []
    for path in vocabularies:
        wanted = ("tiktoken",) if path.name == TIKTOKEN_VOCABULARY else ("sentencepiece", "protobuf")
        needed += [package for package in wanted if package not in needed]
    return tuple(package for package in TOKENIZER_CONVERSION_PACKAGES if package in needed)


def tokenizer_support_message(error: BaseException, snapshot: Path | None = None) -> str | None:
    """What to say about a backend-tokenizer failure, or ``None`` for another.

    The same failure has two causes and the reader can act on only one of
    them: a repository whose vocabulary needs converting and an installation
    that cannot convert it, or a repository holding no tokenizer worth the
    name. Which one this is comes from the snapshot rather than from the
    message, because they read alike, and because a package absent here that
    nothing in the snapshot would have read is not what went wrong: advising
    its installation sends a reader into the same failure a second time.
    """

    if TOKENIZER_BACKEND_FAILURE not in str(error):
        return None
    vocabularies = tokenizer_vocabularies(snapshot)
    needed = tokenizer_packages_for(vocabularies)
    missing = tuple(package for package in missing_tokenizer_packages() if package in needed)
    named = ", ".join(sorted({path.name for path in vocabularies}))
    if missing:
        return (
            f"This model ships no tokenizer.json, and converting the {named} "
            f"it ships instead needs {', '.join(missing)}: run `pip install "
            f"{' '.join(missing)}` and load again."
        )
    if not vocabularies:
        return (
            "This model's tokenizer could not be built: the repository holds "
            "no tokenizer.json, and no vocabulary to convert into one. Its "
            "tokenizer files are missing, or under names Transformers does "
            f"not read. ({first_line(error)})"
        )
    return (
        f"This model's tokenizer could not be built from {named}, the "
        "vocabulary it ships in place of a tokenizer.json. The file is "
        f"incomplete, or not in the format its name implies. ({first_line(error)})"
    )


class LoadedModel(NamedTuple):
    """What is in memory, as one reading.

    The four move together - a load replaces all of them - so anything that
    reports them together has to read them together. Read field by field,
    a caller can straddle a load and describe one model's weights with
    another's device or precision.
    """

    model_id: str | None = None
    device_name: str | None = None
    precision: str | None = None
    load_id: str | None = None


# What the two loaders hand back: the causal LM and its tokenizer for a text
# model, the pipeline for an image one, and the device to report either way.
# One shape for both, so the load's own bookkeeping does not have to know
# which of them ran. Named apart from LoadedModel below, which is a different
# thing entirely - what is in memory, as one reading for a caller to report.
ReadWeights = tuple[Any, Any, Any, str]


@contextlib.contextmanager
def _capture_loading_report() -> Iterator[None]:
    """Keep Transformers' report when it raises an error referring to it.

    Transformers normally sends this to its own stderr handler, which the
    desktop UI cannot show and the app's file logger never receives. In
    particular, quantization wraps an out-of-memory error as a conversion
    failure, hiding it from our memory-error handling too.
    """

    thread_id = threading.get_ident()
    reports = []

    class ReportHandler(logging.Handler):
        def emit(self, record):
            if record.thread != thread_id:
                return
            message = record.getMessage()
            if "LOAD REPORT" in message:
                plain = re.sub(r"\x1b\[[0-9;]*m", "", message)
                reports[:] = ["\n".join(line.rstrip() for line in plain.splitlines())]

    source = logging.getLogger("transformers")
    handler = ReportHandler(level=logging.WARNING)
    source.addHandler(handler)
    try:
        yield
    except RuntimeError as error:
        if reports and "above report" in str(error):
            report = reports[-1]
            logger.warning("%s", report)
            causes = list(dict.fromkeys(re.findall(
                r"^((?:[\w.]+)?(?:Error|Exception): .+)$", report, re.MULTILINE
            )))
            # Prefer the memory failure even if an earlier conversion also
            # failed: the caller must still recognize that memory ran out.
            memory = next((cause for cause in causes if is_out_of_memory_error(RuntimeError(cause))), None)
            detail = memory or "\n".join(causes[:3]) or report[:8000]
            raise RuntimeError(f"Weight loading failed: {detail}") from error
        raise
    finally:
        source.removeHandler(handler)
        handler.close()


@contextlib.contextmanager
def _explaining_tokenizer_failure(snapshot: Path | None) -> Iterator[None]:
    """Answer a backend-tokenizer failure with something to do about it.

    Passed on as a ``RuntimeError`` rather than the ``ValueError`` it arrives
    as, because the load's own handler watches for ``RuntimeError`` and
    ``MemoryError``: as a ``ValueError`` this failure reached the card
    without the log ever recording that a load had been attempted.
    """

    try:
        yield
    except ValueError as error:
        message = tokenizer_support_message(error, snapshot)
        if message is None:
            raise
        raise RuntimeError(message) from error


def adapter_base_for_load(adapter_path: Path) -> Path:
    """The base snapshot to read an adapter onto, or why there is none.

    The same verdicts :func:`cache_status` gives the Models page, raised as
    the ``RuntimeError`` a load reports, for a load reached some other way
    than through a page that asked first.
    """

    config = adapters.read_adapter_config(adapter_path) or {}
    problem = adapters.adapter_problem(config)
    if problem is not None:
        raise RuntimeError(f"This adapter {problem}")
    base = adapters.base_model_id(config)
    snapshot = adapter_base_snapshot(adapter_path)
    missing, kind = judge_snapshot(snapshot)
    if missing:
        raise RuntimeError(
            f"This adapter was trained on `{base}`, which is not fully downloaded. "
            "Use Download and load on the adapter to fetch it."
        )
    if kind != TEXT_KIND or is_adapter_snapshot(snapshot):
        raise RuntimeError(
            f"This adapter was trained on `{base}`, which is not a Transformers "
            "text model, so there are no weights to merge it into."
        )
    return snapshot


def _read_text_model(
    local_path: Path, torch, backend: str, dtype, bits: int | None, precision: str
) -> ReadWeights:
    """Read one causal-LM checkpoint out of ``local_path`` onto ``backend``.

    ``local_path`` can also be a LoRA adapter, in which case the checkpoint
    read is the base it names and the adapter is merged into it before the
    model goes anywhere it would have to be copied from; see
    :mod:`adapters`. The caller has cleared ``bits`` for an adapter, since
    only full-precision weights can take the merge.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = None
    if is_adapter_snapshot(local_path):
        if bits is not None:
            raise RuntimeError(
                f"A LoRA adapter merges into full-precision weights, not {precision} ones."
            )
        adapter, local_path = local_path, adapter_base_for_load(local_path)
    # The adapter's own tokenizer where it ships one: it is the one its
    # embeddings were trained against, added tokens and chat template both.
    own_tokenizer = adapter is not None and adapters.has_tokenizer(adapter)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter if own_tokenizer else local_path, local_files_only=True
    )

    def merged(model):
        if adapter is None:
            return model
        try:
            model = adapters.merge_adapter(
                model, adapter, len(tokenizer) if own_tokenizer else None
            )
        except ImportError as error:
            raise RuntimeError(
                "LoRA adapters need the peft package: run `pip install peft` "
                f"and load again. ({error})"
            ) from error
        # The config came from the base, so its commit hash is the base's,
        # and a new commit of the adapter over the same base would read as
        # the same weights to everything that asks which weights these are:
        # ModelManager.model_revision, and through it a remembered lens, and
        # the revision a lens file itself records. So the revision becomes
        # both commits, the base's and the adapter's, and either changing
        # changes it. Rewritten on the config rather than kept beside it so
        # there is still one place to read it from. Transformers only reads
        # the field while it resolves files for a load, and the merged
        # model's are all read by now. Unrecorded when either half is, since
        # a half-known revision would vouch for weights it cannot tell apart.
        base = getattr(model.config, "_commit_hash", None)
        own = adapter.name if adapter.parent.name == "snapshots" else None
        model.config._commit_hash = f"{base}+{own}" if base and own else None
        return model

    if backend == "cuda":
        model = merged(AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        ))
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif backend == "mps" and bits is not None:
        # The quantizer packs each weight as it lands, and wants to land it
        # on the device it will run on: a CPU stop on the way is refused, so
        # this is the one Metal load that goes through device_map. The output
        # head and the embeddings are left in half precision, which is what
        # keeps the logit lens reading through the real head.
        try:
            from transformers import MetalConfig
        except ImportError as error:
            # requirements.txt admits 4.57, which predates the quantizer; the
            # rest of the app runs there, so the floor stays and the choice
            # is refused with the version it needs rather than a bare
            # ImportError.
            import transformers

            raise RuntimeError(
                f"{precision} weights need transformers "
                f"{METAL_QUANTIZATION_TRANSFORMERS} or newer; this "
                f"is {transformers.__version__}. Run `pip install "
                f"-U transformers` and load again."
            ) from error

        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                local_files_only=True,
                dtype=dtype,
                device_map="mps",
                quantization_config=MetalConfig(
                    bits=bits, group_size=QUANTIZATION_GROUP_SIZE
                ),
            )
        except ImportError as error:
            raise RuntimeError(
                f"{precision} weights need the kernels package: "
                f"run `pip install kernels` and load again. ({error})"
            ) from error
        device_name = f"Apple Metal (MPS), {precision} weights"
    elif backend == "mps":
        # Into host memory and across afterwards, rather than materialized on
        # the device with device_map="mps". The checkpoint is converted on the
        # way in, and Metal does that conversion with one cast kernel per
        # tensor: on torch 2.14, Olmo-3-7B loaded in 15 seconds this way (12
        # to read and convert, 3 to copy across) and had not finished after
        # seven minutes the other way. An adapter is merged before the copy
        # too, in host memory, for the same reason.
        model = merged(AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )).to("mps")
        device_name = "Apple Metal (MPS)"
    else:
        model = merged(AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ))
        device_name = "CPU"
    return model, tokenizer, None, device_name


def _read_mlx_model(
    local_path: Path, torch, backend: str, dtype, bits: int | None, precision: str
) -> ReadWeights:
    """Read one MLX-quantized checkpoint out of ``local_path`` onto Metal.

    Same signature as the two readers above so the load need not tell them
    apart; ``torch``, ``dtype`` and ``bits`` go unused, because the weights
    are read as the repo packed them and the radio was cleared before this
    was called. ``precision`` is that packing, for the device's name.
    """

    del torch, dtype, bits
    if backend != "mps":
        raise RuntimeError(
            "MLX models run on Apple silicon. Choose a Transformers checkpoint here."
        )
    model, tokenizer, _config = mlx_runtime.read_mlx_model(local_path)
    return model, tokenizer, None, f"Apple Metal (MLX), {precision} weights"


def _reader(kind: str):
    """The function that brings a checkpoint of ``kind`` in.

    Looked up when a load runs rather than kept in a table at import, so a
    stand-in for one reader (a test's, or a future hook's) is the one called.
    """

    if kind == MLX_KIND:
        return _read_mlx_model
    if kind == IMAGE_KIND:
        return _read_pipeline
    return _read_text_model


# The first diffusers release whose DiffusionPipeline takes a local folder
# with local_files_only and returns a pipeline with callback_on_step_end.
# Older releases run everything else in the app, so requirements.txt keeps no
# floor of its own for them and an image load says which version it needs.
PIPELINE_DIFFUSERS = "0.31"


def _read_pipeline(
    local_path: Path, torch, backend: str, dtype, bits: int | None, precision: str
) -> ReadWeights:
    """Read a diffusers pipeline out of ``local_path`` onto ``backend``.

    Built from ``model_index.json``, so whichever pipeline the repo ships is
    the one that runs and its components come as it laid them out. Read into
    host memory and moved across afterwards for the reason the text loader
    gives: Metal converts a checkpoint's dtype far faster on the way in than
    it materializes one already on the device.

    A repo that ships only variant-named weights is asked for that variant,
    because otherwise ``from_pretrained`` looks for the unsuffixed files it
    does not have; see :func:`pipeline_variant`.

    ``bits`` and ``precision`` are accepted and unused. A pipeline is several
    models, only some of them Transformers ones, so the caller has already
    cleared a quantized choice and noted that it did; the signature matches
    the text loader's so the load itself need not tell them apart.
    """

    del bits, precision
    try:
        from diffusers import DiffusionPipeline
    except ImportError as error:
        raise RuntimeError(
            "Image models need the diffusers package: run "
            f"`pip install 'diffusers>={PIPELINE_DIFFUSERS}'` and load again. "
            f"({error})"
        ) from error

    variant = model_cache.pipeline_variant(local_path)
    pipeline = DiffusionPipeline.from_pretrained(
        local_path,
        local_files_only=True,
        torch_dtype=dtype,
        **({"variant": variant} if variant else {}),
    )
    device = {"cuda": "cuda", "mps": "mps"}.get(backend)
    if device is not None:
        pipeline = pipeline.to(device)
    device_name = {
        "cuda": lambda: f"CUDA ({torch.cuda.get_device_name(0)})",
        "mps": lambda: "Apple Metal (MPS)",
    }.get(backend, lambda: "CPU")()
    return None, None, pipeline, device_name


class LoadingMixin:
    """The load and unload methods of :class:`model_runtime.ModelManager`.

    State lives on the manager; see its ``__init__``.
    """

    @contextlib.contextmanager
    def _reading_weights(self, model_id: str) -> Iterator[None]:
        """Name ``model_id`` as the load holding the model lock, for the block.

        Entered with that lock already held, which is what makes one slot
        enough: only one load can be inside at a time. A load claims itself
        before it waits for the lock and the wait can be long, so claim
        order is arrival order and not the order the loads get to read
        anything - a thread can be set aside by the scheduler between the
        two steps and a later load can take the lock first.
        """

        with self._claims_lock:
            self._active_load = model_id
        try:
            yield
        finally:
            with self._claims_lock:
                self._active_load = None

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

    def load(
        self,
        model_id: str,
        local_path: Path,
        progress: LoadProgress | None = None,
        precision: str = "full",
        kind: str = TEXT_KIND,
    ) -> str:
        """Read ``model_id`` into memory from ``local_path``, and say where it landed.

        Blocks until the last weight is in; ``progress`` is how a caller on
        another thread watches it happen. ``precision`` is one of
        :data:`settings.WEIGHT_PRECISIONS`; a quantized choice is honoured
        on Apple Metal and noted, then ignored, elsewhere. ``kind`` picks the
        loader: :data:`TEXT_KIND` reads one causal-LM checkpoint,
        :data:`IMAGE_KIND` reads a diffusers pipeline. It comes from the
        snapshot on disk (:func:`judge_snapshot`), not from the reader, so a
        repo of one kind is never read as the other.

        One model is in memory at a time whichever kind it is: the two share
        the device, and on Apple silicon the GPU draws from the same pool as
        everything else, so holding a 7B model and an image pipeline at once
        is how the machine ends up paging.
        """

        import torch

        # A caller on another thread will have claimed this load already;
        # this claim is the manager's own, released whatever happens. The
        # claim stands for the whole wait for the lock; _reading_weights
        # marks the shorter stretch where this load is the one reading.
        checked_id, claim = self.reserve_load(model_id)
        try:
            with self._lock, self._reading_weights(checked_id):
                return self._load_locked(
                    checked_id,
                    local_path,
                    torch,
                    progress,
                    precision=precision,
                    kind=kind,
                )
        finally:
            self.release_load(claim)

    def _load_locked(
        self,
        model_id: str,
        local_path: Path,
        torch,
        progress: LoadProgress | None = None,
        precision: str = "full",
        kind: str = TEXT_KIND,
    ) -> str:
        """Bring ``model_id`` in from ``local_path`` while the caller holds ``_lock``."""

        progress = progress or LoadProgress()
        self._unload_locked(torch)
        bits = QUANTIZED_BITS.get(precision)
        backend = device_memory.detect_backend(torch)
        dtype = load_dtype(backend, torch)
        if bits is not None and backend != "mps":
            logger.info(
                "Loading %s with full weights: %s weights need Apple Metal, not %s",
                model_id,
                precision,
                backend,
            )
            bits = None
        if bits is not None and kind == IMAGE_KIND:
            # The Metal quantizer is Transformers' own, and a pipeline is
            # several models of which only some are Transformers ones. Rather
            # than quantize a part of it and report a precision that only
            # held for the text encoder, an image load takes its weights
            # whole and says so.
            logger.info(
                "Loading %s with full weights: %s weights are for text models, "
                "not diffusers pipelines",
                model_id,
                precision,
            )
            bits = None
        adapter = kind == TEXT_KIND and is_adapter_snapshot(local_path)
        if bits is not None and adapter:
            # A packed 4-bit or 8-bit matrix has nothing to add the adapter's
            # product to, so the base is read whole and merged; see
            # adapters. Said here, as the image case above says its own.
            logger.info(
                "Loading %s with full weights: a LoRA adapter merges into "
                "full-precision weights, not %s ones",
                model_id,
                precision,
            )
            bits = None
        precision = precision if bits is not None else "full"
        if kind == MLX_KIND:
            # The repo was quantized when it was converted, and that is the
            # precision it loads at; the radio has nothing to add. Named
            # from the config so the badge says what is really running.
            precision = mlx_runtime.precision_label(mlx_runtime.read_mlx_config(local_path))
            # The config's width, not the radio's: the estimate below reads
            # the packed file as it stands and the MLX reader takes no bits
            # at all, so this rides along only to let the refusal, the log
            # and the fit panel name the width the weights really are. Zero
            # it out and a 4-bit conversion would be refused for wanting
            # "full 16-bit weights".
            bits = mlx_snapshot_bits(local_path)
            logger.info(
                "Loading %s at the %s precision it was converted to: MLX weights "
                "are packed already",
                model_id,
                precision,
            )
        # The cap goes on before the check rather than before the load, so
        # the check can refuse a model that fits the machine but not the
        # allocator's half of it. Otherwise a 25 GB checkpoint on an idle
        # 48 GB Mac passes, is read off disk, and only then fails. MLX has
        # its own allocator, which the PyTorch cap says nothing about, so
        # an MLX load is judged against the machine alone.
        ceiling = (
            self._cap_mps_memory(torch) if backend == "mps" and kind != MLX_KIND else None
        )
        # Read after the unload above, so what a replaced model held is not
        # counted against its replacement. Whatever is left is memory the
        # ceiling has already been spent on, and the check has to see it or
        # it judges the weights against a ceiling nothing is holding.
        charged = device_memory.reserved_bytes(torch) if ceiling is not None else None
        estimated, available = self._check_memory(
            model_id,
            local_path,
            dtype_name(dtype),
            backend,
            ceiling=ceiling,
            bits=bits,
            kind=kind,
            charged=charged,
        )
        # Bytes are counted only where the device keeps a total to count
        # them against; elsewhere the loader's own steps are all there is.
        if device_memory.allocated_bytes(backend, torch) is not None:
            progress.measure_bytes(estimated, lambda: device_memory.allocated_bytes(backend, torch))
        try:
            with (
                progress.watch(),
                _capture_loading_report(),
                _explaining_tokenizer_failure(local_path),
            ):
                read = _reader(kind)
                model, tokenizer, pipeline, device_name = read(
                    local_path, torch, backend, dtype, bits, precision
                )
        except (RuntimeError, MemoryError) as error:
            # Before the cache goes back, so the figure is what the device was
            # holding when the load gave up rather than what survived cleanup.
            reached, taken = device_memory.allocated_bytes(backend, torch), device_memory.reserved_bytes(torch)
            logger.warning(
                "Load of %s as %s (%s weights) on %s failed: %s estimated, %s held "
                "on the device, %s estimated available beforehand, device ceiling %s (%s)",
                model_id,
                str(dtype).replace("torch.", ""),
                precision,
                backend,
                memory_note(estimated),
                memory_note(taken),
                memory_note(available),
                memory_note(ceiling),
                first_line(error),
            )
            self._release_device_cache(torch)
            if is_out_of_memory_error(error):
                raise OutOfMemoryError(
                    load_out_of_memory_message(
                        model_id.strip(),
                        estimated=estimated,
                        reached=reached,
                        taken=taken,
                        ceiling=ceiling,
                        weights=weights_note(dtype_name(dtype), bits),
                        # Only a Metal text load has a narrower width to
                        # fall back on, and only one already wider than the
                        # narrowest: everything else would repeat the load
                        # it just failed. ``bits`` is what this load really
                        # used, the radio's choice having been cleared
                        # above wherever it could not be honoured.
                        lower_precision=(
                            backend == "mps"
                            and kind == TEXT_KIND
                            and not adapter
                            and (bits is None or bits > min(QUANTIZED_BITS.values()))
                        ),
                        error=error,
                    )
                ) from error
            raise

        if model is not None:
            model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.kind = kind
        self.engine = (
            mlx_runtime.MlxEngine.from_snapshot(model, local_path)
            if kind == MLX_KIND
            else None
        )
        self.model_id = model_id
        self.local_path = local_path
        self.device_name = device_name
        self.precision = precision
        self.loaded_bytes = estimated
        self.load_count += 1
        with self._loaded_lock:
            self._loaded = LoadedModel(
                model_id, device_name, precision, f"{model_id}#{self.load_count}"
            )
        if adapter:
            logger.info(
                "Merged LoRA adapter %s into %s",
                model_id,
                adapters.base_model_id(adapters.read_adapter_config(local_path) or {}),
            )
        # The one record of what a load cost. Without it a later memory
        # failure cannot be told from a leak, a second copy of the weights, or
        # a machine that was already full when the load began.
        logger.info(
            "Loaded %s as %s (%s weights) on %s: %s estimated, %s held on the "
            "device, %s estimated available beforehand, device ceiling %s",
            model_id,
            str(dtype).replace("torch.", ""),
            precision,
            device_name,
            memory_note(estimated),
            memory_note(device_memory.reserved_bytes(torch)),
            memory_note(available),
            memory_note(ceiling),
        )
        return device_name

    def unload(self) -> None:
        import torch

        with self._lock:
            self._unload_locked(torch)

    def _unload_locked(self, torch) -> None:
        """Clear the loaded model while the caller holds ``_lock``."""

        released, precision, estimated = self.model_id, self.precision, self.loaded_bytes
        self._inspect_cache = None
        self._jacobian_lens = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.engine = None
        self.kind = None
        self.model_id = None
        self.local_path = None
        self.device_name = None
        self.precision = None
        self.loaded_bytes = None
        with self._loaded_lock:
            self._loaded = LoadedModel()
        gc.collect()
        self._release_device_cache(torch)
        if released is not None:
            # The line that closes a load's entry in the log. Without it two
            # "Loaded" lines read the same whether the first model was let go
            # or is still in memory beside the second, and only one of those
            # accounts for a machine that started paging. The figure is read
            # after the cache has gone back, so it is what the process kept
            # rather than what it was holding a moment earlier.
            logger.info(
                "Unloaded %s (%s weights, %s estimated): %s held on the device now",
                released,
                precision or "full",
                memory_note(estimated),
                memory_note(device_memory.reserved_bytes(torch)),
            )

    @staticmethod
    def _release_device_cache(torch=None) -> None:
        """Return the allocator's unused blocks to the device."""

        if torch is None:
            import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        mlx_runtime.clear_cache()

    @staticmethod
    def _check_memory(
        model_id: str,
        local_path: Path,
        load_dtype: str,
        backend: str,
        ceiling: int | None = None,
        bits: int | None = None,
        kind: str = TEXT_KIND,
        charged: int | None = None,
    ) -> tuple[int | None, int | None]:
        """Refuse a load that cannot fit, before any weight is read.

        Returns the memory the weights are expected to take, which is also
        what a load counts its own progress towards, and the availability
        estimate it judged that against so the caller can record it. Both
        are ``None`` when the snapshot could not be measured.

        On CUDA a text model's weights fill the graphics cards and
        ``device_map="auto"`` places the rest on the CPU, so the cards plus
        the machine's memory is what must fit; on Metal the GPU shares the
        machine's memory, and on the CPU it is the machine's memory outright.
        An image pipeline is the exception: it is read into host memory and
        moved onto the card whole, with nothing offloaded, so on CUDA it has
        to fit the card *and* the machine, each on its own rather than added
        together. The combined pool would pass a pipeline that fits host
        memory and then fail inside ``.to("cuda")``; the card alone would
        pass one that fits the card and exhausts the machine while
        ``from_pretrained`` is still staging it. A snapshot whose weights
        cannot be measured is let through: the loader will give its own, more
        specific error.

        ``ceiling`` is what the device's own allocator will hand out, which on
        Metal is less than the machine holds. Whichever of the two is smaller
        is what the weights have to fit inside, so a model too big for the
        allocator is refused here rather than part way through reading it.
        ``charged`` is how much of that ceiling the allocator has already
        spent, which comes off the availability figure for the reason
        :func:`memory_pool` gives: the ceiling counts every Metal byte the
        process holds, so a check that read it as untouched would pass a
        model the allocator refuses halfway in.

        ``kind`` picks which weights are measured: one checkpoint at the root
        for a text model, and for an image pipeline the sum over its
        component folders, each converted from its own stored dtype. The
        stored dtype is what says how far the files grow or shrink on the way
        in, and a pipeline's has to be read out of the weights themselves
        (see :func:`pipeline_loaded_bytes`) because its components keep no
        dtype in their configs and need not agree about it either.

        ``bits`` is the width the weights will really be packed into, which
        the caller has already resolved: cleared where the device will not
        honour the radio, and for an MLX repo taken from the conversion's
        own config. The estimate of a packed MLX file ignores it - the file
        is already that size - but the refusal and the log line name it, so
        a 4-bit conversion is never turned away for wanting full weights.
        """

        # Through main's two helpers rather than branching here: both now
        # take the kind, so the fit panel that also calls them sizes a
        # pipeline the same way a load does.
        estimated = estimate_snapshot_bytes(local_path, load_dtype, bits, kind)
        if estimated is None:
            return None, None
        total, available, pool = device_memory.memory_pool(backend, ceiling, kind, charged=charged)
        weights = weights_note(load_dtype, bits)
        try:
            device_memory.check_memory_for_load(
                validate_model_id(model_id),
                estimated,
                total,
                available,
                pool=pool,
                weights=weights,
            )
        except InsufficientMemoryError:
            # The refusal is the load record. It is the outcome most worth
            # explaining afterwards, and the caller turns it into a status
            # card that the log never sees.
            logger.warning(
                "Refused %s as %s on %s: %s estimated, %s estimated available of %s in %s",
                model_id,
                weights,
                backend,
                memory_note(estimated),
                memory_note(available),
                memory_note(total),
                pool,
            )
            raise
        return estimated, available

    @staticmethod
    def _cap_mps_memory(torch) -> int | None:
        """Make Metal allocations fail well short of the whole machine.

        Unified memory means the GPU and everything else share one pool, and
        both PyTorch's default ceiling and Metal's own recommendation lie
        beyond what macOS can give up without paging. With the cap, a model or
        conversation that outgrows its half of the machine raises an error the
        interface can show; without it, macOS pages until it freezes.

        Returns the ceiling it set, so a load can record what it was.
        """

        fraction = mps_memory_fraction(
            recommended_mps_memory(torch), device_memory.system_memory()[0]
        )
        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if fraction is None or setter is None:
            return None
        try:
            setter(fraction)
        except (RuntimeError, ValueError, TypeError):
            return None
        return mps_ceiling(torch)
