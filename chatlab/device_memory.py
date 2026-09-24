"""The device a model runs on and the memory it has to run in.

Which backend is available (CUDA, Apple Metal, or the CPU), how much memory
it and the machine have free, whether a model of a given size fits, the
ceiling ChatLab sets on Metal so a model cannot page the machine into a
freeze, and how an out-of-memory failure is worded for a reader. Torch is
imported lazily and on a background thread, so none of this slows startup.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

from chatlab import mlx_runtime
from chatlab import settings
from chatlab.model_cache import DTYPE_BYTES, IMAGE_KIND, MLX_KIND, TEXT_KIND

logger = logging.getLogger(__name__)


# Memory kept back when deciding whether a model fits: the application, the
# key-value cache a conversation grows, and the rest of the system all need
# room beside the weights. On Apple silicon the GPU draws from the same pool,
# so a model that "fits" with nothing to spare freezes the whole machine
# instead of failing.
MEMORY_HEADROOM_BYTES = 4 * 1024**3


# How much a model may hold on Metal before PyTorch raises an out-of-memory
# error instead of letting macOS page the machine into a freeze. PyTorch's own
# ceiling is 1.7 times Metal's recommended working set, well past physical
# memory; Metal's recommendation on its own is still most of the machine, 37.4
# GiB of a 48 GB Mac, and a process that big leaves the window server, the
# browser and the editor paging to disk. Half the machine is the default here,
# which still holds a 7B model with a long conversation. Overriding the share
# directly is what the environment variable is for: it names a fraction of
# Metal's recommendation, the same units PyTorch takes.
DEFAULT_MPS_MEMORY_SHARE = 0.5
MPS_MEMORY_FRACTION_ENV = "CHATLAB_MPS_MEMORY_FRACTION"
# Used when the machine's size or Metal's recommendation cannot be read: half
# of whatever Metal offers, which errs the same way.
FALLBACK_MPS_MEMORY_FRACTION = 0.5
# PyTorch reads this itself; when the user has set it, their choice stands.
TORCH_MPS_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"


class InsufficientMemoryError(RuntimeError):
    """A model would not fit in memory, judged before any weight is read."""


class OutOfMemoryError(RuntimeError):
    """The device ran out of memory partway through a run."""


def system_memory() -> tuple[int | None, int | None]:
    """Total and currently available physical memory in bytes, where known.

    macOS is read from ``vm_stat``; Linux reads ``MemAvailable``. Either
    figure is ``None`` when the platform offers nothing usable.
    """

    total: int | None = None
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        total = None
    available: int | None = None
    if sys.platform == "darwin":
        available = _darwin_available_memory()
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


def _darwin_available_memory() -> int | None:
    """Estimate available memory, allowing file-cache reclaim at normal pressure.

    Free, speculative and purgeable pages form the conservative baseline.
    Under normal macOS pressure, also credit the pageable file-backed total,
    subtracting speculative pages already counted in that total. This is an
    estimate: some file pages are active or dirty, so the load guard still
    keeps its working reserve. Anonymous and compressed pages are not credit.

    With elevated or unknown pressure, use only the baseline plus the floor
    of file pages in the inactive queue: ``max(0, inactive - anonymous)``.
    This avoids assuming active file pages can be reclaimed cheaply while
    the machine is struggling. Swap occupancy alone is not current pressure.

    ``None`` means unmeasured; a measured exhausted machine must return 0.
    """

    output = _run_quietly(["vm_stat"])
    size = re.search(r"page size of (\d+) bytes", output)
    if not size:
        return None

    def count(label: str) -> int | None:
        found = re.search(rf"{label}:\s*(\d+)", output)
        return int(found.group(1)) if found else None

    reclaimable = [
        count(f"Pages {name}") for name in ("free", "speculative", "purgeable")
    ]
    if all(pages is None for pages in reclaimable):
        return None
    total = sum(pages for pages in reclaimable if pages is not None)

    inactive, anonymous = count("Pages inactive"), count("Anonymous pages")
    files, speculative = count("File-backed pages"), count("Pages speculative")
    pressure = _run_quietly(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
    ).strip()
    # This sysctl exports dispatch flags (normal=1, warning=2, critical=4),
    # not XNU's internal enum (whose normal value is 0).
    # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c
    if pressure == "1" and files is not None and speculative is not None:
        total += max(0, files - speculative)
    elif inactive is not None and anonymous is not None:
        total += max(0, inactive - anonymous)
    return total * int(size.group(1))


def format_memory(count: int) -> str:
    """Render a memory size the way the machine's own specs do: ``48.0 GB``."""

    return f"{count / 1024**3:.1f} GB"


def memory_note(count: int | None) -> str:
    """``format_memory`` for a figure the platform may not have given."""

    return "unknown" if count is None else format_memory(count)


def weights_note(load_dtype_name: str | None, bits: int | None = None) -> str:
    """How an estimate read the weights: ``4-bit weights``, ``full 16-bit weights``.

    The same model estimates several-fold apart across these, so the figure
    says little on its own: 13.6 GB is a refusal to a reader who chose four
    bits and a fair reading to one who did not. ``bits`` is what the load
    will actually pack the linear layers into rather than what was asked
    for - a quantized choice is honoured on Apple Metal alone and cleared
    before the check runs anywhere else - so a message built from this tells
    a reader on a graphics card why their 4-bit choice did not shrink
    anything. An MLX repo is the other way round: it was packed when it was
    converted and loads at that width whatever the radio says, so its
    callers pass the width from the repo itself (see
    :func:`mlx_snapshot_bits`) rather than the choice.
    """

    if bits is not None:
        return f"{bits}-bit weights"
    stored = DTYPE_BYTES.get((load_dtype_name or "").lower())
    return "full weights" if stored is None else f"full {stored * 8}-bit weights"


def reserved_bytes(torch=None) -> int | None:
    """Bytes the accelerator's allocator holds from the driver, or ``None``.

    :func:`allocated_bytes` counts live tensors, which is what a load measures
    its own progress against. This counts what the process has taken out of
    the machine, cached blocks included, which is the figure a machine-wide
    memory problem shows up in and so the one worth recording. It is also
    what a Metal ceiling is checked against; see :func:`memory_pool`.

    On Metal that is the driver's figure alone. It is the whole process's
    allocation on the device and not PyTorch's share of it - MPS, MPSGraph
    and mlx-lm allocate through the same Metal device - so adding what MLX
    reports would count those buffers twice. Measured: a 1 GiB MLX array
    moves ``torch.mps.driver_allocated_memory`` by 1 GiB, with or without a
    torch tensor on the device first.
    """

    if torch is None:
        import torch
    try:
        if torch.cuda.is_available():
            devices = range(int(torch.cuda.device_count()))
            return sum(int(torch.cuda.memory_reserved(index)) for index in devices)
        if torch.backends.mps.is_available():
            return int(torch.mps.driver_allocated_memory())
    except (RuntimeError, AttributeError, ValueError, TypeError):
        return None
    return None


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


def cuda_device_memory(torch=None) -> tuple[int | None, int | None]:
    """Total and free memory on the one CUDA device a load would land on.

    ``.to("cuda")`` places a model on the current device rather than
    spreading it, so for a load that does that the sum across every visible
    card is the wrong figure: a pipeline whose weights fit the aggregate but
    not card 0 would pass and then fail while it was being moved. This is
    the reading :func:`cuda_memory` takes, for that one device.
    """

    if torch is None:
        import torch
    try:
        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    except (RuntimeError, AttributeError, ValueError, TypeError, IndexError):
        return None, None
    return int(total), int(free)


def allocated_bytes(backend: str, torch=None, device_only: bool = False) -> int | None:
    """Bytes of live tensors on ``backend``'s device, or ``None`` where unknown.

    CUDA and Metal each keep a running total in their allocator, which is how
    far a load has got measured in bytes. Host memory keeps no such figure, so
    a load onto the CPU is followed by the loader's own step count instead.

    ``device_only`` narrows the CUDA reading to the current card. The sum is
    what a text model spread by ``device_map="auto"`` really holds, and the
    wrong figure for anything asking what one card would get back: a model
    across two cards would credit the whole of it to the one an image
    pipeline is about to land on. Everywhere else there is one device and
    the two readings agree.
    """

    if torch is None:
        import torch
    try:
        if backend == "cuda":
            if device_only:
                return int(torch.cuda.memory_allocated(torch.cuda.current_device()))
            devices = range(int(torch.cuda.device_count()))
            return sum(int(torch.cuda.memory_allocated(index)) for index in devices)
        if backend == "mps":
            # Plus MLX's live buffers: an MLX load is measured against the
            # same estimate a Metal load is, and its weights land here.
            return _sum_known(
                int(torch.mps.current_allocated_memory()), mlx_runtime.active_bytes()
            )
    except (RuntimeError, AttributeError, ValueError, TypeError):
        return None
    return None


def _smaller_known(left: tuple[int | None, int | None], right: tuple[int | None, int | None]):
    """The tighter of two memory pools, field by field.

    For a load that has to fit in both of two pools independently rather
    than in their sum: whichever is smaller is the one that will refuse it.
    A pool whose size could not be read does not constrain anything, so the
    other one stands.
    """

    return tuple(
        one if other is None else other if one is None else min(one, other)
        for one, other in zip(left, right)
    )


def _sum_known(*figures: int | None) -> int | None:
    known = [figure for figure in figures if figure is not None]
    return sum(known) if known else None


def offload_pool(
    gpu: tuple[int | None, int | None], host: tuple[int | None, int | None]
) -> tuple[int | None, int | None]:
    """Total and estimated available memory across the CUDA cards and host.

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
    weights: str | None = None,
) -> None:
    """Refuse a load that would not leave ``headroom`` beside the weights.

    ``pool`` names where the figures come from in the message: the machine's
    own memory, or the GPU plus the machine when the weights may spread over
    both. ``weights`` names the precision the estimate was made at (see
    :func:`weights_note`), without which the reader cannot tell a refusal
    that a smaller precision would lift from one that nothing but a smaller
    model will.
    """

    needed = estimated_bytes + headroom
    size = f"about {format_memory(estimated_bytes)}"
    size += f" for {weights}" if weights else " of memory"
    if total is not None and needed > total:
        raise InsufficientMemoryError(
            f"{model_id} needs {size} plus "
            f"{format_memory(headroom)} of safety reserve, and {pool} has "
            f"{format_memory(total)} in total. Choose a smaller model."
        )
    if available is not None and needed > available:
        raise InsufficientMemoryError(
            f"{model_id} needs {size} plus "
            f"{format_memory(headroom)} of safety reserve. ChatLab estimates "
            f"{format_memory(available)} available within its memory safety limits "
            "and stopped this load to reduce the risk of heavy paging. "
            "Wait for memory pressure to fall, close memory-heavy applications, "
            "or choose a smaller model."
        )


def detect_backend(torch=None) -> str:
    """Where a load would land: ``"cuda"``, ``"mps"`` or ``"cpu"``.

    The same order :meth:`ModelManager._load_locked` picks in, so anything
    that describes a load before it happens agrees with the load itself.
    """

    if torch is None:
        import torch
    try:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except (RuntimeError, AttributeError, ValueError, TypeError):
        return "cpu"
    return "cpu"


def load_dtype(backend: str, torch=None):
    """The dtype ``backend`` reads full weights as.

    CUDA takes bfloat16 where the card supports it, Metal half precision, and
    the CPU float32 because half-precision arithmetic there is slow or absent.
    """

    if torch is None:
        import torch
    if backend == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if backend == "mps":
        return torch.float16
    return torch.float32


def dtype_name(dtype) -> str:
    """``torch.float16`` -> ``"float16"``, the spelling :data:`DTYPE_BYTES` uses."""

    return str(dtype).replace("torch.", "")


def memory_pool(
    backend: str,
    ceiling: int | None = None,
    kind: str = TEXT_KIND,
    charged: int | None = None,
) -> tuple[int | None, int | None, str]:
    """Total and available memory a load on ``backend`` may use, and its name.

    On CUDA a text model's weights fill the graphics cards and
    ``device_map="auto"`` places the rest on the CPU, so the cards plus the
    machine's memory is the pool; on Metal the GPU shares the machine's
    memory, and on the CPU it is the machine's memory outright. ``ceiling``
    is what the device's own allocator will hand out, which on Metal is less
    than the machine holds: whichever of the two is smaller is what the
    weights have to fit inside.

    An image pipeline on CUDA is the exception. It is read into host memory
    and moved onto the card whole, with nothing offloaded, so it has to fit
    the card *and* the machine, each on its own rather than added together:
    the combined pool would pass one that fits host memory and then fail
    inside ``.to("cuda")``, and the card alone would pass one that exhausts
    the machine while ``from_pretrained`` is still staging it. And it is the
    one card the pipeline lands on rather than every visible one, because
    ``.to("cuda")`` does not spread a model the way ``device_map="auto"``
    does: the sum would pass a pipeline that fits the aggregate and fails on
    card 0. See :func:`cuda_device_memory`.

    ``charged`` is what the device's allocator has already taken out against
    that ceiling, as :func:`reserved_bytes` reads it. It comes off the
    availability figure because the ceiling is a total and not an allowance
    on top of what is already held: Metal checks a new allocation against
    every byte this process has out, so a machine with 40 GB free and 15 GB
    already taken from a 24 GB ceiling has 9 GB to offer a load, not 24.
    Without it a check reads the ceiling as untouched, passes a model that
    fits it, and leaves the allocator to refuse the same model part way
    through reading it.
    """

    if backend == "cuda" and kind == IMAGE_KIND:
        total, available = _smaller_known(cuda_device_memory(), system_memory())
        pool = "both this GPU and this machine"
    elif backend == "cuda":
        total, available = offload_pool(cuda_memory(), system_memory())
        pool = "the GPU plus this machine"
    else:
        total, available = system_memory()
        pool = "this machine"
    if ceiling is not None:
        # The total stays the whole ceiling: it is the size of the pool, and
        # a model too big for it is unfit however idle the process is.
        # Availability is what is left of it now, so a model that outgrew
        # only the memory already taken reads as tight - the verdict an
        # unload or a quit would lift.
        room = ceiling - min(charged or 0, ceiling)
        total = ceiling if total is None else min(total, ceiling)
        available = room if available is None else min(available, room)
        if total == ceiling:
            pool = "Metal on this machine"
    return total, available, pool


FITS = "fits"
TIGHT = "tight"
UNFIT = "unfit"
FIT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Fit:
    """Whether a model would load right now, judged the way a load judges it.

    The three verdicts are the three outcomes of
    :func:`check_memory_for_load`: ``FITS`` is a load that would go ahead,
    ``UNFIT`` one that could not fit the machine however idle it were, and
    ``TIGHT`` one that fits the machine but not what is free at the moment, so
    closing something or waiting for memory pressure to fall would let it in.
    ``FIT_UNKNOWN`` is a model whose size or whose machine could not be
    measured; it says so rather than guessing.
    """

    state: str
    estimated: int | None = None
    total: int | None = None
    available: int | None = None
    pool: str = "this machine"
    note: str = ""

    @property
    def known(self) -> bool:
        return self.state != FIT_UNKNOWN


def fit_for(
    estimated: int | None,
    total: int | None,
    available: int | None,
    pool: str = "this machine",
    headroom: int = MEMORY_HEADROOM_BYTES,
    weights: str = "weights",
) -> Fit:
    """The verdict a load of ``estimated`` bytes would get from this machine now.

    Deliberately a second reading of the same figures rather than a trial
    load: the check that refuses a load is the authority, and this exists to
    say beforehand what it would answer, so a reader picking a model is not
    made to press the button to find out. ``weights`` names the precision
    the estimate was made at, for the same reason the refusal names it: the
    verdict on one model moves as the precision radio does, and a note that
    left it out would look like the figure had changed by itself.
    """

    if estimated is None or (total is None and available is None):
        return Fit(
            FIT_UNKNOWN,
            estimated,
            total,
            available,
            pool,
            "Size unknown until it is downloaded."
            if estimated is None
            else "This machine does not report its memory.",
        )
    needed = estimated + headroom
    size = f"About {format_memory(estimated)} of {weights}"
    reserve = f"{format_memory(headroom)} of safety reserve"
    if total is not None and needed > total:
        return Fit(
            UNFIT,
            estimated,
            total,
            available,
            pool,
            f"{size} plus {reserve} is more than the "
            f"{format_memory(total)} {pool} has.",
        )
    if available is not None and needed > available:
        return Fit(
            TIGHT,
            estimated,
            total,
            available,
            pool,
            f"{size} plus {reserve} needs more than the "
            f"{format_memory(available)} ChatLab estimates free right now. "
            "Close something memory-heavy, or wait for memory pressure to fall.",
        )
    return Fit(
        FITS,
        estimated,
        total,
        available,
        pool,
        f"{size}, inside the {memory_note(available)} ChatLab estimates free.",
    )


@dataclass(frozen=True)
class DeviceProfile:
    """The device a load would use, and the memory it would draw on.

    ``backend`` is ``None`` until torch has been imported. The Models page is
    painted before anything has needed torch, and importing it takes several
    seconds, so a first paint answers from the machine's own memory - the
    pool on every backend but CUDA - and says the device is not known yet
    rather than blocking the page or guessing at one. :func:`warm_device`
    starts that import beside the interface, so by the time a reader looks
    the full reading is there.
    """

    backend: str | None = None
    dtype: str | None = None
    total: int | None = None
    available: int | None = None
    ceiling: int | None = None
    pool: str = "this machine"
    recommended: int | None = None
    """Metal's recommended working set, which the ceiling is a share of."""

    fraction: float | None = None
    """The share of that recommendation the allocator is held to."""

    held: int | None = None
    """Live tensors on the device: the loaded model, and any cache beside it.

    ``None`` where the device keeps no such figure, which is host memory.
    Summed across the cards on CUDA, because that is what a text model
    spread by ``device_map="auto"`` holds; see :attr:`held_here` for the one
    card an image pipeline would land on.
    """

    taken: int | None = None
    """What the device allocator holds from the driver, cached blocks included.

    More than :attr:`held` wherever the allocator is keeping blocks no
    tensor is using, and it is this figure rather than that one that a
    Metal ceiling is checked against; see :func:`memory_pool`. ``None``
    where the device keeps no such figure.
    """

    held_here: int | None = None
    """Live tensors on the one device a load would land on.

    The same as :attr:`held` everywhere but a multi-card CUDA host, where
    that one sums across the cards. What a tighter pool may count as coming
    back; see :meth:`reclaimable`.
    """

    @property
    def quantizes(self) -> bool:
        """Whether a quantized weight precision would be honoured here."""

        return self.backend == "mps"

    def for_kind(self, kind: str, reclaimed: int | None = None) -> DeviceProfile:
        """The reading a load of this kind would get, with the unload counted in.

        Two kinds read differently. An image pipeline on CUDA is staged in
        host memory before it is moved onto the card, so it has to fit both
        pools rather than their sum; see :func:`memory_pool`. An MLX
        conversion under a Metal ceiling is not held to it: the ceiling is
        PyTorch's allocator cap, and mlx-lm allocates through Metal on its
        own, so ``_load_locked`` judges an MLX load against the machine alone
        and this has to say the same. Judged against the capped figures, a
        conversion that fits the Mac but not PyTorch's half of it would be
        listed as tight or unfit - and hidden by **Fits this computer** -
        while the button loads it. Everything else is :meth:`reclaimed` on
        the reading it already is.

        The unload is counted into each pool *before* they are collapsed to
        the tighter one, which is why this does both rather than leaving the
        caller to call :meth:`reclaimed` afterwards. Unloading frees card
        memory on the card, and collapsing first would credit it to whichever
        pool happened to be smaller: with host memory the tighter side, 6 GB
        free there would read as 14 after an 8 GB model left the card, and
        the list would call a replacement a fit that the load then refuses.

        Kept as a method so a caller judging a list of models takes the two
        readings it needs once rather than per model - reading host memory
        is a subprocess.
        """

        if kind == MLX_KIND and self.ceiling is not None:
            # The same call the load check makes, ceiling and all: the
            # capped figures cannot be uncapped from here, and the machine
            # is what MLX has to fit. Reading it is the subprocess the
            # docstring mentions, paid once per list rather than per model.
            total, available, pool = memory_pool(self.backend, None, kind)
            return replace(
                self, total=total, available=available, ceiling=None, pool=pool
            ).reclaimed(reclaimed)
        if kind != IMAGE_KIND or self.backend != "cuda":
            return self.reclaimed(reclaimed)
        card_total, card_free = cuda_device_memory()
        host_total, host_free = system_memory()
        # What the unload gives back, pool by pool. The card gets its own
        # allocator figure; host memory gets nothing, because the share of a
        # spread model that sat there cannot be read from here and guessing
        # high is what turns a refusal into a promise.
        if card_free is not None:
            card_free += self.held_here or 0
        total, available = _smaller_known(
            (card_total, card_free), (host_total, host_free)
        )
        if self.ceiling is not None:
            total = self.ceiling if total is None else min(total, self.ceiling)
            available = (
                self.ceiling if available is None else min(available, self.ceiling)
            )
        return replace(
            self,
            total=total,
            available=available,
            pool="both this GPU and this machine",
        )

    def reclaimed(self, estimated: int | None = None) -> DeviceProfile:
        """The same reading with the loaded model's memory given back.

        A load unloads whatever is in memory before it checks whether the
        next model fits, so the weights on the device now are not in the way
        of the model that would replace them. Anything that judges a
        replacement has to say the same, or the list and the button disagree.

        ``estimated`` is the load's own estimate of the weights it read, as
        :attr:`ModelManager.loaded_bytes` records it, and the larger of the
        two figures is what a load gives back. Neither is enough alone: host
        memory keeps no allocator figure at all, and a CUDA model spread over
        the cards and the machine by ``device_map="auto"`` is only counted on
        the cards by one while the other covers the whole of it. On Metal the
        allocator figure can be the larger, a response's key-value cache
        being live tensors too, and that is freed with the model.

        Under a Metal ceiling neither figure is enough either, because an
        unload hands back the allocator's cached blocks as well as the
        weights and the ceiling was charged for both; see :attr:`taken`.
        """

        given = self.reclaimable(estimated)
        if self.ceiling is not None and self.taken is not None:
            # An unload empties the allocator's cache as well as dropping the
            # weights, so everything the allocator is holding comes back from
            # under a ceiling, not the model's own share of it. Cached blocks
            # can be the larger part: crediting the weights alone would leave
            # them charged against a cap that is about to be handed them
            # back, and list a model as tight that the button then loads.
            given = max(given, self.taken)
        if not given:
            return self
        if self.available is None:
            return self
        # Never past the pool itself. Under a Metal ceiling availability is
        # the ceiling less what is already taken from it, and an unload
        # gives back what it took rather than more: without the clamp a
        # model whose weights outweigh the blocks it is holding would credit
        # the difference to a pool that never had it.
        available = self.available + given
        if self.total is not None:
            available = min(available, self.total)
        return replace(self, available=available)

    def reclaimable(self, estimated: int | None = None) -> int:
        """How much of the loaded model's memory a summed pool gets back.

        The larger of the two figures, for the reason :meth:`reclaimed`
        gives. A pool that is the tighter of two does its own arithmetic
        pool by pool before collapsing them, and does not come through here;
        see :meth:`for_kind`.
        """

        return max(self.held or 0, estimated or 0)


DEVICE_LABELS = {"mps": "Apple Metal (MPS)", "cpu": "CPU"}


def device_label(backend: str | None, torch=None) -> str:
    """How the device names itself, in the words a loaded model's badge uses."""

    if backend is None:
        return "not determined yet"
    if backend != "cuda":
        return DEVICE_LABELS.get(backend, backend)
    if torch is None:
        torch = imported_torch()
    try:
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return "CUDA"


def device_profile(torch=None) -> DeviceProfile:
    """Read the device and its memory now.

    Nothing is cached: availability moves from one second to the next, and
    the Metal ceiling follows a setting the reader can change. Only the
    import is expensive, and this never pays for it - torch is used if
    another part of the app has already brought it in.
    """

    if torch is None:
        torch = imported_torch()
    if torch is None:
        total, available = system_memory()
        return DeviceProfile(total=total, available=available)
    backend = detect_backend(torch)
    budget = mps_budget(torch) if backend == "mps" else MetalBudget()
    taken = reserved_bytes(torch)
    total, available, pool = memory_pool(backend, budget.ceiling, charged=taken)
    return DeviceProfile(
        backend=backend,
        dtype=dtype_name(load_dtype(backend, torch)),
        total=total,
        available=available,
        ceiling=budget.ceiling,
        pool=pool,
        recommended=budget.recommended,
        fraction=budget.fraction,
        held=allocated_bytes(backend, torch),
        taken=taken,
        held_here=allocated_bytes(backend, torch, device_only=True),
    )


# Set once torch has finished importing. Python puts a module in
# ``sys.modules`` before its body has run, so the module being there says
# nothing about whether its attributes exist yet: a reader that went by
# presence alone could find ``torch`` without ``torch.backends``, and either
# raise or quietly report the wrong device. Nothing reads torch through
# :func:`imported_torch` until this is set.
_torch_ready = threading.Event()


def imported_torch():
    """torch, if it is imported and finished importing; otherwise ``None``.

    For everything that describes the machine before a model is loaded. None
    of it is worth paying a multi-second import for, and none of it may read
    a module that is still being built.
    """

    return sys.modules.get("torch") if _torch_ready.is_set() else None


def warm_device() -> None:
    """Import torch beside the interface so the first fit verdict is the full one.

    Started when the app is built. The import holds no lock the interface
    wants and the module is imported once however many threads ask for it, so
    a load that arrives while this is still running simply waits for it. The
    interface never waits: until this finishes, the device is reported as not
    read yet.
    """

    def read() -> None:
        try:
            import torch  # noqa: F401
        except Exception as error:  # pragma: no cover - torch is a hard dependency
            logger.warning("Could not import torch to read the device: %s", error)
            return
        # Only now is every attribute there to be read.
        _torch_ready.set()
        log_device_profile()

    threading.Thread(target=read, name="chatlab-device", daemon=True).start()


def log_device_profile() -> None:
    """Record the device and the budget every load will be judged against.

    Written once torch has landed rather than at startup, because none of it
    can be read before then. It is the second half of the record
    :func:`logs.log_environment` opens: that one names the build and the
    machine, this one names what the allocator on it will allow, which is the
    figure a refused load or a fatal one has to be read against. The device
    alone used to be logged here, and a device name says nothing about how
    much of the machine a load was allowed to take.
    """

    profile = device_profile()
    parts = [
        f"full weights as {profile.dtype or 'unknown'}",
        f"{memory_note(profile.total)} in the pool ({profile.pool})",
        f"{memory_note(profile.available)} estimated available",
    ]
    if profile.backend == "mps":
        ceiling = f"Metal ceiling {memory_note(profile.ceiling)}"
        if profile.fraction is not None:
            ceiling += (
                f" at {profile.fraction:.2f} of the "
                f"{memory_note(profile.recommended)} recommended"
            )
        parts.append(ceiling)
    if profile.held is not None:
        parts.append(f"{memory_note(profile.held)} already held")
    logger.info("Device: %s - %s", device_label(profile.backend), "; ".join(parts))


# How often the memory watch looks, how far a figure has to move before it is
# worth a line, and how long the watch will stay quiet before writing one
# anyway.
MEMORY_WATCH_SECONDS = 30.0
MEMORY_WATCH_STEP_BYTES = 256 * 1024**2
MEMORY_WATCH_IDLE_SECONDS = 600.0


class MemoryWatch:
    """A periodic record of what the device holds and what the machine has left.

    macOS kills a process that takes too much memory without giving it the
    chance to say so, so the last line in the log is whatever was written
    before the kill. Loads and replies are recorded when they finish, which
    leaves a session that died partway through a long reply with nothing at
    all between the load and the silence - no way to tell a steady climb from
    one large allocation, which is the difference between a leak and a model
    that never fitted.

    This writes a line while nothing else is happening, and writes as few as
    it can: one only when a figure has moved by ``step`` since the last one,
    so an idle app is silent, and one every ``idle`` seconds regardless so a
    kill always has a recent reading in front of it.
    """

    def __init__(
        self,
        step: int = MEMORY_WATCH_STEP_BYTES,
        idle: float = MEMORY_WATCH_IDLE_SECONDS,
    ) -> None:
        self.step = step
        self.idle = idle
        self._last: tuple[int | None, int | None] | None = None
        self._at: float | None = None

    def read(self) -> tuple[int | None, int | None]:
        """What the device allocator holds, and what the machine has free.

        Both are ``None`` where the platform offers no figure, and the first
        is ``None`` until torch has finished importing - this never waits for
        that import, since a watch that blocked the interface's first seconds
        to report on memory would be its own problem.
        """

        torch = imported_torch()
        held = reserved_bytes(torch) if torch is not None else None
        return held, system_memory()[1]

    def tick(self, now: float) -> bool:
        """Write a line if this reading is worth one; say whether it did."""

        reading = self.read()
        if not self._worth_recording(reading, now):
            return False
        held, available = reading
        logger.info(
            "Memory: %s held on the device, %s available on the machine",
            memory_note(held),
            memory_note(available),
        )
        self._last = reading
        self._at = now
        return True

    def _worth_recording(self, reading: tuple[int | None, int | None], now: float) -> bool:
        if self._last is None or self._at is None:
            return True
        if now - self._at >= self.idle:
            return True
        for current, previous in zip(reading, self._last):
            # A figure that appeared or went away is news whatever its size:
            # the first is torch finishing its import, the second a device
            # that stopped answering.
            if (current is None) != (previous is None):
                return True
            if current is not None and previous is not None and abs(current - previous) >= self.step:
                return True
        return False


def watch_memory(interval: float = MEMORY_WATCH_SECONDS) -> threading.Thread:
    """Run a :class:`MemoryWatch` beside the app for as long as the process lives.

    Started by the two entry points rather than by ``build_app``, so the test
    suite, which builds the interface many times over, does not accumulate a
    thread per build.
    """

    watch = MemoryWatch()

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                watch.tick(time.monotonic())
            except Exception:  # noqa: BLE001 - the watch has to outlive a bad reading
                logger.debug("Could not record the memory reading", exc_info=True)

    thread = threading.Thread(target=loop, name="chatlab-memory", daemon=True)
    thread.start()
    return thread


def model_fit(
    estimated: int | None,
    profile: DeviceProfile | None = None,
    bits: int | None = None,
) -> Fit:
    """Whether weights of ``estimated`` bytes would load on this machine now.

    ``bits`` is the width the estimate packed the linear layers into, passed
    in rather than read from the precision radio here: the caller has already
    decided whether this device honours the choice, and a verdict that
    described a different precision from the one it measured would be worse
    than one that named none.
    """

    profile = profile if profile is not None else device_profile()
    return fit_for(
        estimated,
        profile.total,
        profile.available,
        profile.pool,
        weights=weights_note(profile.dtype, bits),
    )


def mps_memory_fraction(
    recommended: int | None = None, total: int | None = None
) -> float | None:
    """The Metal allocation cap to apply, or ``None`` to leave PyTorch's own.

    The figure PyTorch wants is a share of Metal's recommended working set, so
    the default is converted: half the machine's own memory, expressed against
    ``recommended``, and never more than Metal would hand over anyway. Passing
    neither figure falls back to half the recommendation.

    ``CHATLAB_MPS_MEMORY_FRACTION`` overrides the settings file, which
    overrides that default; a value PyTorch would reject, or a user-set
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO``, leaves the allocator alone.
    """

    if os.environ.get(TORCH_MPS_WATERMARK_ENV):
        return None
    default = default_mps_memory_fraction(recommended, total)
    raw = os.environ.get(MPS_MEMORY_FRACTION_ENV)
    if raw is None:
        saved = settings.current().mps_memory_fraction
        return default if saved is None else saved
    try:
        fraction = float(raw)
    except ValueError:
        return default
    return fraction if 0 < fraction <= 2 else None


def default_mps_memory_fraction(
    recommended: int | None = None, total: int | None = None
) -> float:
    """Half the machine's memory, as a share of Metal's recommended working set."""

    if not recommended or not total:
        return FALLBACK_MPS_MEMORY_FRACTION
    return min(1.0, (total * DEFAULT_MPS_MEMORY_SHARE) / recommended)


class MetalBudget(NamedTuple):
    """Metal's own recommendation, the share of it allowed, and the product.

    Any of the three is ``None`` where Metal will not say what it recommends,
    or where the cap is being left alone - see :func:`mps_memory_fraction`.
    """

    recommended: int | None = None
    fraction: float | None = None
    ceiling: int | None = None


def mps_budget(torch=None) -> MetalBudget:
    """What Metal recommends and what ChatLab will let the allocator take.

    What :meth:`ModelManager._cap_mps_memory` sets, computed without setting
    it, so a load can be judged against the same ceiling it will meet and the
    Settings page can say what that ceiling is.
    """

    if torch is None:
        import torch
    recommended = recommended_mps_memory(torch)
    fraction = mps_memory_fraction(recommended, system_memory()[0])
    if fraction is None or not recommended:
        return MetalBudget(recommended, fraction)
    return MetalBudget(recommended, fraction, int(recommended * fraction))


def mps_ceiling(torch=None) -> int | None:
    """The most Metal's allocator will hand out under the cap, or ``None``."""

    return mps_budget(torch).ceiling


def recommended_mps_memory(torch) -> int | None:
    """Metal's recommended working set, or ``None`` when torch will not say."""

    reader = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
    if not callable(reader):
        return None
    try:
        return int(reader())
    except (RuntimeError, ValueError, TypeError):
        return None


def is_out_of_memory_error(error: BaseException) -> bool:
    """Whether a backend raised for lack of device or host memory."""

    if isinstance(error, (MemoryError, OutOfMemoryError)):
        return True
    text = str(error).lower()
    return "out of memory" in text or "insufficient memory" in text


def first_line(error: BaseException) -> str:
    """The first line of a backend's complaint, short enough to quote.

    A backend can raise with nothing to say - ``MemoryError()`` is the one
    that turns up under memory pressure - and indexing the first line of an
    empty message is how a memory failure becomes a confusing ``IndexError``
    somewhere else entirely.
    """

    lines = str(error).splitlines()
    return lines[0][:200] if lines else f"no message from {type(error).__name__}"


# What to try, per kind of run: the advice has to name something the reader
# is actually looking at. A conversation and a response length mean nothing
# to someone who was drawing a picture, and a step count and a size mean
# nothing to someone mid-reply.
OUT_OF_MEMORY_ADVICE = {
    TEXT_KIND: "Shorten the conversation or the text, lower the response length",
    IMAGE_KIND: "Draw a smaller picture, lower the step count",
}


def out_of_memory_message(error: BaseException, kind: str = TEXT_KIND) -> str:
    """One readable sentence for a backend's out-of-memory failure."""

    advice = OUT_OF_MEMORY_ADVICE.get(kind, OUT_OF_MEMORY_ADVICE[TEXT_KIND])
    return (
        f"The model ran out of memory. {advice}, or load a smaller model. "
        f"({first_line(error)})"
    )


def load_out_of_memory_message(
    model_id: str,
    estimated: int | None = None,
    reached: int | None = None,
    taken: int | None = None,
    ceiling: int | None = None,
    weights: str | None = None,
    lower_precision: bool = False,
    error: BaseException | None = None,
) -> str:
    """What to tell a reader whose load ran out of memory part way through.

    The backend's own complaint names bytes that say nothing without the
    figures around them - Metal's is three numbers and a watermark
    environment variable - so it is quoted at the end rather than left to
    speak for itself.

    ``reached`` against ``estimated`` is how far the weights got, which is
    what says whether a smaller precision would have been enough or nothing
    short of a smaller model would. ``taken`` is everything the process has
    out on the device, cached blocks included, and it is that figure rather
    than the weights that a Metal ``ceiling`` is checked against: a load can
    be refused with half the weights in because the rest of the cap is
    already spoken for. Naming the ceiling matters because on Metal it is
    ChatLab's own and a reader told only to close applications would be
    working on the wrong half of the machine.

    ``lower_precision`` is whether a narrower width is open to this load at
    all, which the caller works out from the backend and the kind. Only a
    Metal text load can be packed narrower: the quantizer is Transformers'
    own, an image pipeline is loaded whole, and an MLX repo is already
    packed at the width it was converted to. Advising a precision anywhere
    else sends the reader round the same full-weight load a second time.
    """

    parts = [f"{model_id} did not fit in memory."]
    if estimated is not None and reached is not None:
        parts.append(
            f"About {format_memory(reached)} of an estimated "
            f"{format_memory(estimated)} of {weights or 'weights'} was in when "
            "the device refused the next allocation."
        )
    if ceiling is not None:
        # The two figures differ by the blocks the allocator is holding
        # without a tensor in them, which is the whole reason a load can
        # stop with half the weights in, so the cap is described as counting
        # them rather than left to look like arithmetic that does not add up.
        out = (
            "" if taken is None else f" {format_memory(taken)} was out when it stopped."
        )
        parts.append(
            f"ChatLab caps Metal allocations at {format_memory(ceiling)} and counts "
            f"every block the allocator holds against it, cached ones included.{out} "
            "`mps_memory_fraction` in the settings file moves that cap."
        )
    smaller = (
        "a smaller model or a narrower weight precision"
        if lower_precision
        else "a smaller model"
    )
    parts.append(
        "Unload anything else on the device, close memory-heavy applications, "
        f"or choose {smaller}."
    )
    if error is not None:
        parts.append(f"({first_line(error)})")
    return " ".join(parts)


def reraise_out_of_memory(error: BaseException, kind: str = TEXT_KIND) -> None:
    """Re-raise a backend failure, as :class:`OutOfMemoryError` when that is what it was."""

    if isinstance(error, OutOfMemoryError) or not is_out_of_memory_error(error):
        raise error
    raise OutOfMemoryError(out_of_memory_message(error, kind)) from error
