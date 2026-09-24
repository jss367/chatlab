"""Download and load progress: the cards a job streams while it runs.

A download and a load each run on a thread of their own, because
``snapshot_download`` and ``from_pretrained`` block until the last byte and a
handler that blocked with them could show nothing past its first frame. The
generators here watch that thread and yield a status card every half second,
with the rate and the time left once they can be told, so every handler that
fetches or loads a model shows its progress the same way. The sentences that
say what the cache already holds live here too: a download's first card is
one of them, and the My Models and search details reuse them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

from chatlab import adapters
from chatlab.model_cache import (
    BASE_MODEL,
    MODEL_WEIGHTS,
    TEXT_KIND,
    CacheStatus,
    cache_status,
    format_bytes,
    is_adapter_snapshot,
)
from chatlab.progress_bars import DownloadSnapshot, LoadProgress, LoadSnapshot
from chatlab.ui import runtime
from chatlab.ui.common import (
    DOWNLOAD_POLL_SECONDS,
    LOAD_POLL_SECONDS,
    RATE_WINDOW_SECONDS,
    describe_duration,
    progress_bar,
    status_card,
)

logger = logging.getLogger(__name__)


class RateMeter:
    """Bytes per second over the recent past, from readings taken as they come."""

    def __init__(self, window: float = RATE_WINDOW_SECONDS, clock=time.monotonic):
        self._samples: deque[tuple[float, int]] = deque()
        self._window = window
        self._clock = clock

    def rate(self, bytes_done: int) -> float | None:
        now = self._clock()
        if self._samples and self._samples[-1][1] == 0:
            # The first non-zero reading is the baseline. Until the byte bars
            # exist nothing is counted, and a resumed download credits every
            # byte already on disk at once, which is not transfer speed.
            self._samples.clear()
        self._samples.append((now, bytes_done))
        while len(self._samples) > 2 and now - self._samples[1][0] >= self._window:
            self._samples.popleft()
        first_time, first_bytes = self._samples[0]
        elapsed = now - first_time
        if elapsed < 1.0 or bytes_done <= first_bytes:
            return None
        return (bytes_done - first_bytes) / elapsed


class Pace:
    """Time left in a job, from how fast its own progress has moved so far.

    Measured from the first reading that showed any progress rather than from
    the start, because the seconds before that are setup: a load spends them
    reading the config and building the model, and counting them as slow
    progress would put the first estimate minutes out.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._first: tuple[float, float] | None = None

    def remaining(self, fraction: float) -> float | None:
        """Seconds left at the pace set since progress began, where it can be told."""

        now = self._clock()
        if fraction <= 0:
            return None
        if self._first is None:
            self._first = (now, fraction)
            return None
        first_time, first_fraction = self._first
        elapsed, moved = now - first_time, fraction - first_fraction
        if elapsed < 1.0 or moved <= 0:
            return None
        return (1.0 - fraction) * elapsed / moved


def download_detail(model_id: str, snap: DownloadSnapshot, rate: float | None) -> str:
    name = f"`{model_id}`"
    if not snap.started:
        return (
            f"Asking Hugging Face which files {name} needs. "
            "Files already in the cache are reused."
        )
    files = f"{snap.files_done} of {snap.files_total} files"
    if snap.bytes_total == 0:
        return f"Checking {name} against the cache: {files}."
    percent = int(snap.fraction * 100)
    figures = (
        f"{format_bytes(snap.bytes_done)} of {format_bytes(snap.bytes_total)} · {files}"
    )
    if rate:
        remaining = max(0, snap.bytes_total - snap.bytes_done)
        # format_bytes() prints a byte count verbatim below 1 KB, so a rate
        # handed to it as the float it is measured in reads "812.3456789 B/s".
        # Rounding never reaches zero: a download with nothing moving has no
        # rate at all and never gets here, so "0 B/s" beside a time left would
        # contradict itself. The estimate keeps the unrounded rate: it divides
        # by it.
        figures += (
            f" · {format_bytes(max(1, round(rate)))}/s"
            f" · {describe_duration(remaining / rate)} left"
        )
    return f"{name}\n\n`{progress_bar(snap.fraction)}` {percent}%\n\n{figures}"


def stream_download(model_id: str, hf_token: str, revision: str | None = None):
    """Yield a status card every half second until ``model_id`` is on disk.

    Returns the snapshot path, so a caller writes
    ``path = yield from stream_download(...)``. A failed download raises here.
    ``revision`` is the branch, tag or commit to fetch, the default branch
    when ``None``.

    The download runs on its own thread: ``snapshot_download`` blocks until the
    last byte, and a handler that blocked with it could show nothing past its
    first frame. If this model is already being fetched (a handler whose
    browser tab went away leaves its thread running), the card follows that
    download rather than starting a second one to fight over the same files.
    That holds across revisions too, which is why a download is listed by
    model ID alone: two revisions of one repo share its ``blobs`` folder, so
    a pinned base waits for a default-branch download of the same repo to
    end and then fetches what its own revision still lacks.
    """

    cleaned = model_id.strip()
    # Reserved before the worker exists: the reservation is what stops a
    # second handler, arriving in the same instant, from starting its own.
    progress, reserved = runtime.MANAGER.reserve_download(cleaned)
    if not reserved:
        meter = RateMeter()
        while runtime.MANAGER.active_downloads.get(cleaned) is progress:
            snap = progress.snapshot()
            yield status_card(
                "Downloading model",
                download_detail(cleaned, snap, meter.rate(snap.bytes_done)),
                "working",
            )
            time.sleep(DOWNLOAD_POLL_SECONDS)
        # Whatever that download left behind is now in the cache, so this pass
        # either returns at once or resumes where it stopped.
        return (yield from stream_download(model_id, hf_token, revision))

    outcome: dict = {}

    def work() -> None:
        try:
            outcome["path"] = runtime.MANAGER.download(
                cleaned, hf_token, progress, revision=revision
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=work, name="chatlab-download", daemon=True)
    try:
        worker.start()
    except BaseException:
        # download() never ran, so its finally cannot release the reservation.
        runtime.MANAGER.release_download(cleaned, progress)
        raise
    meter = RateMeter()
    while worker.is_alive():
        snap = progress.snapshot()
        yield status_card(
            "Downloading model",
            download_detail(cleaned, snap, meter.rate(snap.bytes_done)),
            "working",
        )
        worker.join(DOWNLOAD_POLL_SECONDS)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["path"]


def adapter_base(path: Path) -> tuple[str, str | None] | None:
    """The base model the adapter snapshot at ``path`` needs, or ``None``.

    The base's Hub ID and the revision the adapter pins it at, ``None`` for
    the default branch; see :func:`adapters.base_revision`. ``None`` for a
    snapshot that is not an adapter, and for an adapter ChatLab cannot load
    at all: fetching a base for one of those would download a model nobody
    asked for and leave the adapter as unloadable as before.
    """

    if not is_adapter_snapshot(path):
        return None
    config = adapters.read_adapter_config(path) or {}
    if adapters.adapter_problem(config) is not None:
        return None
    return adapters.base_model_id(config), adapters.base_revision(config)


def stream_download_with_base(model_id: str, hf_token: str):
    """:func:`stream_download`, followed by the base model when it is an adapter.

    Returns the adapter's snapshot path and a sentence on what the base
    download did, which is empty for anything that is not an adapter. The
    adapter comes first because only its config says which base it needs.
    The token goes to both: the popular bases are gated, and a reader who
    can see the adapter usually has access to the base it was trained on.
    """

    path = yield from stream_download(model_id, hf_token)
    needed = adapter_base(Path(path))
    if needed is None:
        return path, ""
    base, revision = needed
    logger.info(
        "%s is a LoRA adapter for %s%s; fetching the base too",
        model_id,
        base,
        f" at revision {revision}" if revision else "",
    )
    started = time.monotonic()
    before = cache_status(base, revision=revision)
    yield status_card(
        "Downloading base model",
        f"`{model_id.strip()}` is a LoRA adapter trained on `{base}`, which is "
        "fetched next. " + describe_cache(base, before)[1],
        "working",
    )
    yield from stream_download(base, hf_token, revision)
    fetched = describe_fetched(
        before, cache_status(base, revision=revision), time.monotonic() - started
    )
    return path, f" Base model `{base}`: {fetched[0].lower()}{fetched[1:]}"


def load_detail(
    model_id: str, snap: LoadSnapshot, rate: float | None, remaining: float | None
) -> str:
    name = f"`{model_id}`"
    if not snap.started:
        weights = (
            f"{format_bytes(snap.bytes_total)} of weights"
            if snap.bytes_total
            else "the weights"
        )
        return (
            f"Reading {weights} for {name} out of the cache on disk. Nothing "
            "is being downloaded; this is the wait for memory."
        )
    # A card that still says "loading" never claims to be finished: the
    # allocator holds the last byte a moment before the loader is done with
    # the model, and a full bar over a wait that goes on reads as a hang.
    shown = min(snap.fraction, 0.99)
    percent = int(shown * 100)
    if snap.counts_bytes:
        # Bytes on the device: the second half of a load onto Metal, and the
        # whole of one onto a graphics card.
        figures = (
            f"{format_bytes(snap.bytes_done)} of {format_bytes(snap.bytes_total)} "
            "on the device"
        )
        if rate:
            figures += f" · {format_bytes(rate)}/s"
    else:
        figures = f"{snap.steps_done} of {snap.steps_total} parts read"
    if remaining is not None:
        figures += f" · {describe_duration(remaining)} left"
    return f"{name}\n\n`{progress_bar(shown)}` {percent}%\n\n{figures}"


def stream_load(
    model_id: str, path: Path, precision: str = "full", kind: str = TEXT_KIND
):
    """Yield a status card every half second until ``model_id`` is in memory.

    Returns the device it landed on, so a caller writes
    ``device = yield from stream_load(...)``. A failed load raises here.
    ``precision`` is the weight precision chosen on the Models page.
    ``kind`` is the snapshot's own verdict on what it holds, so a diffusers
    pipeline is read by the pipeline loader and a checkpoint by the text one.

    The load runs on its own thread, as a download does: ``from_pretrained``
    blocks until the last weight, and a handler that blocked with it could
    show nothing past its first frame.
    """

    progress = LoadProgress()
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["device"] = runtime.MANAGER.load(
                model_id, path, progress, precision=precision, kind=kind
            )
        except BaseException as error:
            outcome["error"] = error
        finally:
            # Held until the load is over, so the two claims between them
            # cover the whole of it: this one from before the thread existed,
            # the manager's own from the moment it reached load().
            runtime.MANAGER.release_load(claim)

    worker = threading.Thread(target=work, name="chatlab-load", daemon=True)
    # Claimed before the worker exists, because the claim is what stops a
    # removal or a redownload arriving in this same instant from moving the
    # snapshot the load is about to read. The worker names the load only once
    # it reaches runtime.MANAGER.load, and the model lock is taken later still.
    _model_id, claim = runtime.MANAGER.reserve_load(model_id)
    # Published against that claim so the chat page's badge can show how far
    # the load has come, wherever the load was started from; the cards below
    # only ever reach the handler that started it.
    runtime.MANAGER.note_load_progress(claim, progress)
    try:
        worker.start()
    except BaseException:
        # work() never ran, so its finally cannot give the claim back.
        runtime.MANAGER.release_load(claim)
        raise
    meter, pace = RateMeter(), Pace()
    while worker.is_alive():
        snap = progress.snapshot()
        yield status_card(
            "Loading model",
            load_detail(
                model_id.strip(),
                snap,
                meter.rate(snap.bytes_done),
                pace.remaining(snap.fraction),
            ),
            "working",
        )
        worker.join(LOAD_POLL_SECONDS)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["device"]


def describe_missing(status: CacheStatus) -> str:
    """``config.json and the model weights are missing``, for a card."""

    names = [
        "the model weights" if name == MODEL_WEIGHTS
        else f"the base model `{status.base_model}`" if name == BASE_MODEL
        else f"`{name}`"
        for name in status.missing_files
    ]
    if len(names) > 3:
        names = names[:2] + [f"{len(names) - 2} more weight files"]
    if len(names) == 1:
        return f"{names[0]} {'are' if names[0] == 'the model weights' else 'is'} missing"
    return f"{', '.join(names[:-1])} and {names[-1]} are missing"


def describe_on_disk(status: CacheStatus) -> str:
    """``2.0 GB cached, 1 file (300 MB) partly downloaded``, for a card.

    A partial blob is not called a weight file: the hub keeps one blob folder
    per repo, so from outside it could be any file of any revision.
    """

    cached = f"{format_bytes(status.cached_bytes)} cached"
    if not status.partial_files:
        return cached
    files = "file" if status.partial_files == 1 else "files"
    return (
        f"{cached}, {status.partial_files} {files} "
        f"({format_bytes(status.partial_bytes)}) partly downloaded"
    )


def describe_cache(model_id: str, status: CacheStatus) -> tuple[str, str]:
    """Title and detail for the card shown while a download starts.

    The cases a reader can tell apart from the outside - nothing on disk, a
    snapshot still short of files (whether a download was cut off or another
    tool fetched only part of the repo), and a finished one - each get their
    own wording, so "Downloading" never hides that the files were already
    here. Which files are missing is the verdict; ``.incomplete`` blobs are
    reported as a size only, since the hub's blob folder is shared across
    revisions and a stray partial need not belong to this snapshot.
    """

    name = f"`{model_id.strip()}`"
    if status.missing_files:
        return (
            "Resuming download",
            f"{name} is only partly on disk ({describe_on_disk(status)}): "
            f"{describe_missing(status)}. Only the missing bytes are fetched.",
        )
    if status.present:
        return (
            "Checking cached model",
            f"{name} is already in the Hugging Face cache "
            f"({format_bytes(status.cached_bytes)}). Checking for missing or "
            "updated files; nothing is downloaded twice.",
        )
    return (
        "Downloading model",
        f"Fetching {name} into the Hugging Face cache. Nothing is cached yet, "
        "so this is a full download and may take a while.",
    )


def describe_fetched(before: CacheStatus, after: CacheStatus, elapsed: float) -> str:
    fetched = after.total_bytes - before.total_bytes
    if fetched <= 0:
        return f"Already up to date; nothing new was fetched ({elapsed:.1f} seconds)."
    if before.present:
        return (
            f"Fetched the remaining {format_bytes(fetched)} in {elapsed:.1f} seconds."
        )
    return f"Fetched {format_bytes(fetched)} in {elapsed:.1f} seconds."
