"""The Models page: downloading, loading, listing and searching for models, and the badge."""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import gradio as gr

from chatlab import settings
from chatlab.model_discovery import STARTER_MODELS
from chatlab.device_memory import (
    FITS,
    TIGHT,
    UNFIT,
    DeviceProfile,
    Fit,
    device_profile,
    imported_torch,
    model_fit,
)
from chatlab.hub_search import (
    DISCOVERY_CANDIDATES,
    HUB_SORTS,
    SEARCH_LIMIT,
    HubModel,
    mlx_versions,
    search_hub_kinds,
)
from chatlab.model_cache import (
    BASE_MODEL,
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MLX_KIND,
    MODEL_WEIGHTS,
    TEXT_KIND,
    CachedModel,
    CacheStatus,
    cache_folder,
    cache_root,
    cache_status,
    estimate_parameter_bytes,
    estimate_snapshot_bytes,
    folder_bytes,
    format_bytes,
    is_adapter_snapshot,
    list_cached_models,
    mlx_available,
    mlx_bits_from_id,
    mlx_snapshot_bits,
    snapshot_folder,
    sort_cached_models,
    validate_model_id,
)
from chatlab.model_loading import QUANTIZED_BITS
from chatlab.model_runtime import LOADING
from chatlab.progress_bars import DownloadSnapshot, LoadProgress, LoadSnapshot
from chatlab import adapters
from chatlab.model_errors import ModelBusy, ModelDownloading, ModelLoaded
from chatlab.ui import runtime
from chatlab.ui.model_finder import (
    DEFAULT_SEARCH_ORDER,
    HUB_HEADING,
    ORDER_NOTES,
    PANE_EMPTY,
    VISION_TAGS,
    Listing,
    model_kind,
    pane_body,
    pane_head,
    results_list,
)
from chatlab.ui.model_repository import matching_repository
from chatlab.ui.common import (
    DEFAULT_MODEL_DOWNLOAD,
    DOWNLOAD_POLL_SECONDS,
    IncompleteSnapshotError,
    LOAD_POLL_SECONDS,
    MODELS_PAGE,
    RATE_WINDOW_SECONDS,
    Card,
    alarm,
    describe_duration,
    failure_card,
    progress_bar,
    show_page,
    status_card,
)

# Every failure on this page used to end as a card in the browser and nothing
# else, so a load that broke reached a reader as one escaped sentence with no
# traceback behind it. The handlers below say what was asked for and what went
# wrong, because this page is where a session's memory trouble starts.
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


def stream_download(
    model_id: str,
    hf_token: str,
    revision: str | None = None,
    *,
    title: str = "Downloading model",
    above: str = "",
):
    """Yield a status card every half second until ``model_id`` is on disk.

    Returns the snapshot path, so a caller writes
    ``path = yield from stream_download(...)``. A failed download raises here.
    ``revision`` is the branch, tag or commit to fetch, the default branch
    when ``None``. ``above`` is markdown the card shows over this download's
    own bar, which is where an adapter's finished download stays in sight
    while its base comes down.

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
                title,
                above + download_detail(cleaned, snap, meter.rate(snap.bytes_done)),
                "working",
            )
            time.sleep(DOWNLOAD_POLL_SECONDS)
        # Whatever that download left behind is now in the cache, so this pass
        # either returns at once or resumes where it stopped.
        return (
            yield from stream_download(
                model_id, hf_token, revision, title=title, above=above
            )
        )

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
            title,
            above + download_detail(cleaned, snap, meter.rate(snap.bytes_done)),
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


def finished_adapter(model_id: str, snapshot: Path) -> str:
    """The adapter's line on the base model's download card, and the base's label."""

    # The snapshot's files link into the blob folder, so stat() follows each
    # one to the bytes it stands for.
    size = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())
    figures = f" · {format_bytes(size)}" if size else ""
    return (
        f"**LoRA adapter** `{model_id}`\n\n"
        f"`{progress_bar(1.0)}` 100%{figures}\n\n"
        "**Base model**\n\n"
    )


def stream_download_with_base(model_id: str, hf_token: str):
    """:func:`stream_download`, followed by the base model when it is an adapter.

    Returns the adapter's snapshot path and a sentence on what the base
    download did, which is empty for anything that is not an adapter. The
    adapter comes first because only its config says which base it needs.
    The token goes to both: the popular bases are gated, and a reader who
    can see the adapter usually has access to the base it was trained on.

    Once the adapter is down the card gives each repository its own line,
    the adapter's held full over the base's bar, so the one bar never runs
    to the end and starts again from nothing.
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
    above = finished_adapter(model_id.strip(), Path(path))
    yield status_card(
        "Downloading base model",
        above
        + f"`{base}`, which the adapter was trained on, is fetched next. "
        + describe_cache(base, before)[1],
        "working",
    )
    yield from stream_download(
        base, hf_token, revision, title="Downloading base model", above=above
    )
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


def chosen_model(model_id: str, selected: str | None) -> str:
    """The model a Model-panel button acts on: the picked row, or the typed ID.

    The row wins when there is one. Gradio snapshots a click's inputs in the
    browser, and a row selection reaches the ID box only through a server
    round trip (see ``select_my_model``), so a button clicked inside that
    window carries a box that still holds whatever was there before - which
    starts out as the 15 GB default, a model nobody asked for. The radio is
    set by the reader's own click and so is always current.

    The reverse window is real and is not closed here: typing an ID clears the
    highlight, but that also takes a round trip, so a click landing inside it
    still carries the old row and loads that instead of the typed ID. It is
    left open deliberately. Both candidates are models the reader named, and
    the row is already on disk, so the cost is a wrong-but-cheap load rather
    than an unrequested 15 GB one. Closing it would need the two controls to
    be ordered against each other, and Gradio's queue does not order separate
    listeners: ``default_concurrency_limit`` is per listener, so the marker a
    typing listener would set is not guaranteed to be written before a click
    handler reads it.
    """

    return (selected or model_id or "").strip()


def refresh_model_actions(
    model_id: str, selected: str | None, repository: dict | None = None,
    hf_token: str | None = None,
):
    """Show the actions appropriate to the chosen model's local files.

    A downloaded model is also named by kind, because this panel is where a
    reader asks what they can do with the model in front of them and the two
    kinds are driven from different pages. Without it, selecting an image
    pipeline offers **Load cached** and says nothing about the answer
    arriving on the Images page rather than in the chat.
    """

    cleaned = chosen_model(model_id, selected)
    active = runtime.MANAGER.active_downloads.get(cleaned)
    if active is not None:
        return (
            "**Downloading** · " + download_detail(cleaned, active.snapshot(), None),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, variant="secondary"),
        )
    try:
        cached = cache_status(cleaned) if cleaned else CacheStatus()
    except ValueError as error:
        logger.warning("Cannot read the cache for %s: %s", cleaned, error)
        return (
            html.escape(str(error)),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, variant="secondary"),
        )
    except OSError:
        # Keep local loading available if the cache cannot be inspected;
        # its handler can explain the actual error when clicked. The card
        # cannot name the error, so the log is the only place it is kept.
        logger.warning("Could not check the cache for %s", cleaned, exc_info=True)
        return (
            "Could not check downloaded files.",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True, variant="secondary"),
        )
    if cached.complete:
        detail = f"**Downloaded** · Ready to load from disk. {where_to_use(cached.kind)}"
        if cached.base_model is not None:
            detail += (
                f" A LoRA adapter for `{cached.base_model}`, merged in at full precision."
            )
        if runtime.MANAGER.model_id == cleaned:
            detail = (
                "**Downloaded · Loaded now** · Load cached again to apply a new "
                f"precision. {where_to_use(cached.kind)}"
            )
    elif cached.unsupported:
        detail = "**Downloaded · Unsupported** · " + (
            cached.unsupported_reason or "ChatLab cannot load this model's format."
        )
    elif cached.present:
        detail = "**Download incomplete** · Download and load will fetch the remaining files."
    else:
        detail = "**Not downloaded** · No local files for this model." if cleaned else "Enter a model ID or select a model."
    download = not (cached.complete or cached.unsupported)
    checked = matching_repository(cleaned, repository, hf_token)
    can_download = (
        checked.get("status") not in {"invalid", "missing", "checking", "restricted"}
        and not checked.get("access_restricted", False)
    )
    architecture_available = not checked.get("architecture_unavailable", False)
    can_load = can_download and not checked.get("unsupported", False) and architecture_available
    return (
        detail,
        gr.update(visible=download, interactive=can_load),
        gr.update(visible=download, interactive=can_download),
        gr.update(visible=cached.complete, variant="primary" if cached.complete else "secondary"),
    )


def refresh_stale_model_actions(
    model_id: str, selected: str | None, repository: dict | None,
    hf_token: str | None, stamp: tuple[bool, int] | None,
):
    """The timer's refresh: repaint only when the local-file status has moved.

    A download runs in a worker thread, so nothing this tab does tells it
    that files are arriving - or that another tab started or finished one.
    The timer covers that, but it ticks in every open session for the whole
    life of the app, and :func:`refresh_model_actions` ends in
    :func:`cache_status`, which stats every blob and walks the snapshot. On
    a large or network-mounted cache that is continuous disk work for a page
    nobody is looking at, so an idle tick paints nothing at all. Same shape
    as :func:`refresh_stale_model_switch`, and for the same reason.

    Two things make the card wrong, and both are read without touching the
    disk. A download of the chosen model is in flight, so its bytes have
    moved since the last tick; the card is repainted every tick until it
    ends, which is the progress. And what a cache scan would find changed -
    ``cache_revision``, which :meth:`ModelManager.note_cache_change` moves
    on every download start and end and every removal, whichever tab did it.
    A download ending is both, so the one repaint that turns "Downloading"
    back into a local-file reading is the revision's, not a special case.

    ``stamp`` is what this tab last painted from; a tab that has not painted
    yet passes ``None`` and is painted.
    """

    cleaned = chosen_model(model_id, selected)
    downloading = runtime.MANAGER.active_downloads.get(cleaned) is not None
    revision = runtime.MANAGER.cache_revision
    if stamp is not None and not downloading and tuple(stamp) == (False, revision):
        return (*(gr.skip(),) * 4, stamp)
    painted = refresh_model_actions(model_id, selected, repository, hf_token)
    return (*painted, (downloading, revision))


def download_model(model_id: str, hf_token: str, selected: str | None = None):
    model_id = chosen_model(model_id, selected)
    logger.info("Download requested for %s", model_id)
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except (OSError, ValueError) as error:
        logger.warning("Download of %s failed before it started", model_id, exc_info=True)
        yield failure_card("Download failed", html.escape(str(error)))
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path, base_note = yield from stream_download_with_base(model_id, hf_token)
        elapsed = time.monotonic() - started
        fetched = describe_fetched(before, cache_status(model_id), elapsed)
    except Exception as error:
        logger.exception("Download of %s failed", model_id)
        yield failure_card("Download failed", html.escape(str(error)))
        return

    yield status_card(
        "Download complete",
        f"{fetched} `{model_id.strip()}` is cached in `{path}`.{base_note} "
        "Use **Load cached** when ready.",
        "success",
    )


LOAD_WHILE_GENERATING = (
    "The model is answering a message. Press Stop, or wait for the reply to "
    "finish, before loading another model."
)
LOAD_WHILE_LOADING = "Another load is already under way. Wait for it to finish."


def occupied_reason(held: str | None, loading: str, generating: str) -> str:
    """Which of the two refusals fits, for whatever ``held`` says has the model.

    ``held`` is what the refused reservation answered with, passed down
    rather than read again here. Asking a second time is what this used to
    do and it was wrong: a load that finishes between the refusal and the
    read leaves nothing holding the model, and the reader is told a response
    is running and to press a Stop button that is not on the page. See
    :meth:`ModelManager.claim_exclusive_load`.

    The same two the generation side names, so a load and a reply cannot
    describe the manager differently.
    """

    return loading if held == LOADING else generating


def refused_load_card(held: str | None, extra: str = "") -> str:
    """The card a load gets when a reply or another load already has the model."""

    reason = occupied_reason(held, LOAD_WHILE_LOADING, LOAD_WHILE_GENERATING)
    return status_card("Cannot load now", f"{reason}{extra}", "error")


def download_and_load_model(
    model_id: str, hf_token: str, selected: str | None = None, precision: str = "full"
):
    """Download and load the model explicitly selected on the Models page.

    The load is claimed after the download rather than before it: the files
    can take an hour to arrive, and a claim standing for all of it would
    refuse every reply on the chat page for the duration. The claim covers
    what it has to - the load itself, which is where two loads would collide
    and where a reply would find the model swapped underneath it.
    """

    model_id = chosen_model(model_id, selected)
    logger.info("Download and load requested for %s at %s weights", model_id, precision)
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except (OSError, ValueError) as error:
        logger.warning("Setup of %s failed before it started", model_id, exc_info=True)
        yield failure_card("Model setup failed", html.escape(str(error)))
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path, base_note = yield from stream_download_with_base(model_id, hf_token)
        claimed, held = runtime.MANAGER.claim_exclusive_load(model_id)
        if claimed is None:
            # The download is the slow half and the claim is only taken after
            # it, so this is where a reply started meanwhile turns the job
            # back. Without the line the trail holds the request and a
            # finished download and no account of the load that never ran.
            logger.info(
                "Load of %s after its download refused: %s has the model",
                model_id,
                held or "another claim",
            )
            yield refused_load_card(
                held,
                f" `{model_id.strip()}` is on disk; use **Load cached** to "
                "finish the job.",
            )
            return
        try:
            fetched = describe_fetched(
                before, cache_status(model_id), time.monotonic() - started
            )
            yield status_card(
                "Loading model",
                f"{fetched}{base_note} Moving `{model_id.strip()}` onto the best "
                "available device…",
                "working",
            )
            # Read after the download rather than before it: what the repo turns
            # out to hold is only knowable once its files are here.
            fetched_status = cache_status(model_id)
            device = yield from stream_load(
                model_id, path, precision, fetched_status.kind
            )
        finally:
            runtime.MANAGER.release_load(claimed[1])
    except Exception as error:
        logger.exception("Setup of %s at %s weights failed", model_id, precision)
        yield failure_card("Model setup failed", html.escape(str(error)))
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Model ready",
        f"`{model_id.strip()}`{adapter_note(fetched_status, precision)} is loaded "
        f"on **{device}** ({elapsed:.1f} seconds total). "
        f"{where_to_use(fetched_status.kind)}",
        "success",
    )


def adapter_note(status: CacheStatus, precision: str) -> str:
    """`` (a LoRA adapter merged into `base`)``, for a ready card, or nothing.

    Says too when the precision radio was passed over, since the card is the
    one place a reader who chose 4-bit would otherwise find out only from
    the memory figures.
    """

    if status.base_model is None:
        return ""
    note = f" (a LoRA adapter merged into `{status.base_model}`"
    if precision in QUANTIZED_BITS:
        note += f", at full precision: an adapter cannot merge into {precision} weights"
    return note + ")"


MISSING_FILES_PATTERN = re.compile(r"(\d+) file\(s\) are missing \((.*?)\)\. ")


def incomplete_snapshot_detail(model_id: str, error: Exception) -> str:
    """Say what an unfinished download left behind and how to finish it."""

    match = MISSING_FILES_PATTERN.search(str(error))
    if match:
        count, names = match.groups()
        missing = f": {count} file{'s' if count != '1' else ''} still missing ({html.escape(names)})"
    else:
        missing = ""
    return (
        f"Only part of `{model_id}` is on disk{missing}. "
        "Click **Download and load** to fetch the rest; the files already downloaded are kept."
    )


def load_cached_model(
    model_id: str,
    selected: str | None = None,
    precision: str = "full",
    claim: int | None = None,
):
    """Load the selected model from local files, preserving any load error.

    The load is claimed here, before the first card, and given back in a
    ``finally``. It is claimed exclusively: a load refuses while a reply is
    streaming or another load is under way rather than queuing behind it,
    because a queued load's first act on winning the model lock is to unload
    the model the reader is looking at, and a second load only fills the
    machine's memory twice over to leave whichever finished last in it.
    Claiming and checking have to be one step - Gradio does not resume a
    streaming handler until the browser has its frame, so a check before the
    first card and a claim after it are a round trip apart.

    ``claim`` is for a caller that already holds the exclusive reservation
    and is passing it down - :func:`switch_model`, which has to refuse in the
    switcher's own way before it yields anything. Its claim is not released
    here; the caller that took it releases it.
    """

    cleaned = chosen_model(model_id, selected)
    logger.info("Cached load requested for %s at %s weights", cleaned, precision)
    if claim is not None:
        yield from _load_cached_model(cleaned, precision)
        return
    try:
        claimed, held = runtime.MANAGER.claim_exclusive_load(cleaned)
    except ValueError as error:
        logger.warning("Cannot load %s: %s", cleaned, error)
        yield failure_card("Could not load cached model", html.escape(str(error)))
        return
    if claimed is None:
        logger.info("Load of %s refused: %s has the model", cleaned, held or "another claim")
        yield refused_load_card(held)
        return
    try:
        yield from _load_cached_model(cleaned, precision)
    finally:
        runtime.MANAGER.release_load(claimed[1])


def _load_cached_model(cleaned: str, precision: str):
    """The cards of a cached load, with the load already claimed."""

    active = runtime.MANAGER.active_downloads.get(cleaned)
    if active is not None:
        snap = active.snapshot()
        progress = (
            f"{format_bytes(snap.bytes_done)} of {format_bytes(snap.bytes_total)} so far"
            if snap.bytes_total
            else "just started"
        )
        yield status_card(
            "Still downloading",
            f"`{cleaned}` is not fully on disk yet ({progress}). "
            "Click **Download and load** to follow the download and load the model when it finishes.",
            "working",
        )
        return

    name = f"`{cleaned}`"
    yield status_card("Finding cached model", f"Looking for {name} locally…", "working")
    try:
        status = cache_status(cleaned)
    except (OSError, ValueError) as error:
        logger.warning("Cannot read the cached files for %s", cleaned, exc_info=True)
        yield failure_card("Could not load cached model", html.escape(str(error)))
        return
    if status.missing_files:
        yield status_card(
            "Download incomplete",
            f"{name} is only partly on disk ({describe_on_disk(status)}): "
            f"{describe_missing(status)}. "
            "Use **Download and load** to fetch the rest.",
            "error",
        )
        return
    if not status.present:
        yield status_card(
            "Not cached",
            f"Nothing for {name} is in the Hugging Face cache. "
            "Use **Download and load** to fetch it.",
            "error",
        )
        return
    if status.unsupported:
        yield status_card(
            "Unsupported model",
            f"{name} is on disk ({describe_on_disk(status)}) but "
            + (status.unsupported_reason or f"is {UNSUPPORTED_REASON}"),
            "error",
        )
        return
    try:
        path = runtime.MANAGER.find_cached(cleaned)
        started = time.monotonic()
        device = yield from stream_load(cleaned, path, precision, status.kind)
    except IncompleteSnapshotError as error:
        logger.warning("Load of %s stopped at an unfinished download: %s", cleaned, error)
        yield failure_card(
            "Download unfinished", incomplete_snapshot_detail(cleaned, error)
        )
        return
    except Exception as error:
        logger.exception("Load of %s at %s weights failed", cleaned, precision)
        yield failure_card("Could not load cached model", html.escape(str(error)))
        return
    yield status_card(
        "Model ready",
        f"{name}{adapter_note(status, precision)} is loaded on **{device}** "
        f"({time.monotonic() - started:.1f} seconds). {where_to_use(status.kind)}",
        "success",
    )


UNLOAD_WHILE_GENERATING = (
    "The model is answering a message. Press Stop, or wait for the reply to "
    "finish, before unloading it."
)
UNLOAD_WHILE_LOADING = "A load is under way. Wait for it to finish before unloading."


def unload_model():
    """Unload the model, or refuse while a reply or a load has it.

    Waiting on the model lock instead would hold this handler for the rest of
    a reply and then pull the model out before the reply's trace was written,
    leaving it with no tokenizer, device or precision; during a load it would
    remove the model the moment it arrived.
    """

    # Claimed before the emptiness check: a load clears the old model before
    # it reads the new one, so a model-less manager can still be mid-load.
    held = runtime.MANAGER.claim_generation()
    if held is not None:
        logger.info("Unload refused: %s has the model", held)
        reason = occupied_reason(held, UNLOAD_WHILE_LOADING, UNLOAD_WHILE_GENERATING)
        return status_card("Cannot unload now", reason, "error")
    try:
        if not runtime.MANAGER.in_memory:
            return status_card("No model loaded", "There is nothing to unload.")
        logger.info("Unload requested for %s", runtime.MANAGER.model_id or "the loaded model")
        runtime.MANAGER.unload()
    finally:
        runtime.MANAGER.release_generation()
    return status_card("Model unloaded", "Model memory has been released.", "success")


# The chat page's badge: which model is answering, or that none is loaded.
# Nothing on the chat page said so before, so an unloaded model only showed
# up as a refusal after a message had been typed and sent.
NO_MODEL_BADGE = "No model loaded"


# How often an open chat page asks again which model is in memory. The manager
# is one object for the whole process, but a handler's updates only reach the
# tab that ran it, so a second tab would go on naming a model that has since
# been swapped out or unloaded. Asking on a timer is what keeps every tab
# honest; a couple of seconds is short enough that nobody types a message
# against a badge that has gone stale, and the question is a few attribute
# reads.
BADGE_REFRESH_SECONDS = 2.0


def load_fraction(progress: LoadSnapshot | None) -> float | None:
    """How much of a load is done, or ``None`` while it has nothing to say.

    A load reports nothing between being claimed and the loader placing its
    first weights - a wait that covers a queue behind a reply, and the
    seconds a big snapshot takes to be opened - and a bar drawn at zero
    through all of that reads as a load that is stuck. ``None`` is that
    state, and the stylesheet draws it as movement without a figure.
    """

    if progress is None or not progress.started:
        return None
    return progress.fraction


def load_percent(progress: LoadSnapshot | None) -> str:
    """`` 42%`` for a load that has begun, and nothing for one that has not."""

    fraction = load_fraction(progress)
    return "" if fraction is None else f" {round(fraction * 100)}%"


def model_badge(state: str, text: str, fraction: float | None = None) -> str:
    """A pill naming the model in memory. ``state`` is the stylesheet's hook.

    ``fraction`` fills a bar along the bottom of the pill: how much of the
    load is done, between 0 and 1, or ``None`` for a load that has not begun
    to report yet, which the stylesheet draws as a stripe that moves on its
    own. The bar lives in the badge rather than beside it because the badge
    is what already sits next to the chat page's model switcher, which is
    where the reader who asked for the load is looking.
    """

    bar = ""
    if state == "loading":
        width = "" if fraction is None else f' style="width: {min(100, max(0, round(fraction * 100)))}%"'
        known = "known" if fraction is not None else "unknown"
        bar = (
            f'<span class="model-badge-bar" data-progress="{known}" aria-hidden="true">'
            f"<span{width}></span></span>"
        )
    return (
        f'<div class="model-badge" data-state="{state}">'
        f'<span class="model-badge-dot" aria-hidden="true"></span>'
        f"<span>{html.escape(text)}</span>{bar}</div>"
    )


def model_snapshot() -> tuple[str | None, str | None, str | None, str, LoadSnapshot | None]:
    """The load under way, how far it has come, the model in memory, its device and kind.

    Reuse these values so the badge and setup links render from the same
    readings. This is display state, not an atomic snapshot or a reservation
    of the model for a later action.

    The kind is read from what is really in memory rather than from what the
    last load was asked for, so it can never disagree with the object a page
    would go on to use.

    The load's progress is read here, beside the ID it belongs to, so the bar
    and the name in one badge cannot describe two different loads. It is read
    only while a load is named: with none under way there is nothing to read,
    and the reading itself asks the device allocator.
    """

    loading = runtime.MANAGER.loading_id
    return (
        loading,
        runtime.MANAGER.model_id,
        runtime.MANAGER.device_name,
        IMAGE_KIND if runtime.MANAGER.image_loaded else TEXT_KIND,
        runtime.MANAGER.loading_progress() if loading else None,
    )


def loaded_model_badge(snapshot=None, *, kind: str = TEXT_KIND) -> str:
    """Name the model in memory, the one being loaded, or neither.

    ``kind`` is the kind of model the page asking can use. A model of the
    other kind is named and set apart rather than hidden: the Chat page
    saying "no model loaded" while an image pipeline filled the machine's
    memory would send a reader to load a second one on top of it.

    A model in memory is named ahead of any load. A load counts itself as
    under way before it waits for the model lock, so asking for a second
    model while the first is part-way through a reply leaves that load
    queued for the rest of the generation - and the model still producing
    the tokens is the one the badge exists to name. A load that has really
    started emptied memory as its first act under that lock, so there is
    nothing left to name and the load is reported instead: that is what
    keeps the badge from claiming nothing is loaded during the minutes the
    weights are on their way in.

    Each attribute is read once and kept rather than asked again to fill in
    the text: a load can finish, or empty memory, between two reads, which
    would leave the badge saying "Loading None" or naming a model loaded on
    None. ``snapshot`` is how a caller that has to agree with this badge
    about something else shares the one reading; see
    :func:`refresh_model_badge`.
    """

    loading, model_id, device, loaded_kind, progress = (
        model_snapshot() if snapshot is None else snapshot
    )
    if model_id and device:
        if loaded_kind == kind:
            return model_badge("ready", f"{model_id} · loaded on {device}")
        return model_badge(
            "other",
            f"{model_id} · {KIND_NAMES.get(loaded_kind, 'model')}, not used here",
        )
    if loading:
        return model_badge("loading", f"Loading {loading}…{load_percent(progress)}", load_fraction(progress))
    return model_badge("empty", NO_MODEL_BADGE)


def _setup_links(snapshot, kind: str):
    """Whether a page's own "choose a model" links belong on screen.

    Hidden once that page has a model it can use, or while any load is
    pending. A model of the other kind leaves them showing, because from
    that page there is still a model to go and load.

    Setup links navigate without loading anything, so downloads do not need
    to disable them.
    """

    loading, model_id, device, loaded_kind, _progress = snapshot
    ready = bool(model_id and device and loaded_kind == kind)
    return gr.update(visible=not (ready or bool(loading)))


def refresh_model_badge():
    """Refresh the Chat page's badge and its setup link from the same reading."""

    snapshot = model_snapshot()
    return loaded_model_badge(snapshot, kind=TEXT_KIND), _setup_links(snapshot, TEXT_KIND)


def refresh_current_model():
    """The Models page names either kind of loaded model without a page mismatch."""

    snapshot = model_snapshot()
    return loaded_model_badge(snapshot, kind=snapshot[3])


def refresh_image_badge():
    """Refresh the Images page's badge and its own setup link."""

    snapshot = model_snapshot()
    return (
        loaded_model_badge(snapshot, kind=IMAGE_KIND),
        _setup_links(snapshot, IMAGE_KIND),
    )


# The chat page's model switcher: a dropdown beside the badge that swaps the
# model in memory for another one already on disk, without a trip to the
# Models page. It offers only the cached text models a load would take right
# now, so picking one never ends in the refusal the Models page exists to
# explain; anything else - a download, a model that will not fit, a new
# precision - is still that page's business.
SWITCH_BUSY = (
    "The model is still answering. Press Stop, or wait for the reply to "
    "finish, before switching."
)
SWITCH_LOADING = "Another load is already under way. Wait for it to finish."


def switch_value() -> str | None:
    """The ID the switcher should show: the model in memory, else the one coming in.

    Named the way the badge names them: a load that has emptied memory is
    the only thing left to name, and a switcher showing nothing for the
    minutes the weights are on their way in would invite a second load on
    top of the first.
    """

    return runtime.MANAGER.model_id or runtime.MANAGER.loading_id


# The kinds the switcher offers: everything that answers on the Chat page.
# An MLX conversion is a language model that generates, scores and inspects
# like any other - ``where_to_use`` sends it to Chat, and the API's own
# ``/v1/models`` lists it beside the Transformers ones - so leaving it out
# would hide half an Apple silicon reader's models from the one control whose
# whole job is choosing which of them they are talking to. It also has to be
# in for the switcher to stay still: ``expected_switch_value`` names whatever
# is in memory unless it is an image pipeline, so an MLX model loaded from the
# Models page would be the value of a dropdown that does not offer it, and
# every tick would read the switcher as stale and repaint it. A machine
# without mlx-lm never reaches the question - the cache scan marks those repos
# unsupported, and unsupported is filtered out below.
SWITCH_KINDS = (TEXT_KIND, MLX_KIND)


def switch_choices(precision: str | None = None) -> list[tuple[str, str]]:
    """The cached text models the Chat page can switch to, as (label, ID) pairs.

    Whole and supported text models - :data:`SWITCH_KINDS`, so MLX
    conversions among them - that fit at ``precision``, plus the one in
    memory, in the list's default order. A model the load would refuse -
    tight or too large, or being downloaded right now - is left out rather
    than offered and then declined: the refusal names figures and a remedy,
    and a dropdown has no room for either. A model whose size could not be
    judged stays in, as the load will try it all the same.

    A **Redownload** is the case the download check is for. An interrupted
    download leaves files missing and is filtered out by that alone, but a
    redownload of a model already complete on disk leaves the cache entry
    looking whole for the whole of the fetch, so nothing but
    ``active_downloads`` says that picking it would be refused. The chat
    page's timer hears about a download starting because
    :meth:`ModelManager.note_cache_change` counts it.

    The fit verdict is the one thing here that a reading taken now can stop
    being true of a pick made later: free memory moves with whatever else the
    machine is running. Offering everything and letting the load explain
    itself is the tempting simplification, and it is the wrong one, because
    :meth:`ModelManager._load_locked` unloads before it checks - a refused
    switch would cost the reader the model they were talking to and leave
    them with nothing loaded. So the list is filtered, and the list is dated:
    :data:`SWITCH_FIT_SECONDS` is how often an idle switcher re-reads this.
    """

    models = sort_cached_models(list_cached_models(), DEFAULT_MODEL_SORT)
    fits = cached_fits(models, precision)
    current = switch_value()
    downloading = runtime.MANAGER.downloading_ids()
    choices = []
    for entry in models:
        if entry.status.kind not in SWITCH_KINDS:
            continue
        if entry.status.missing_files or entry.status.unsupported:
            continue
        # The model in memory stays on the list whatever is happening to its
        # files, as it does for a fit it would fail: it is what the switcher
        # has to show as chosen, and picking it is a no-op anyway.
        if entry.model_id in downloading and entry.model_id != current:
            continue
        fit = fits.get(entry.model_id)
        if fit is not None and fit.known and fit.state != FITS and entry.model_id != current:
            continue
        choices.append((entry.model_id, entry.model_id))
    return choices


def expected_switch_value() -> str | None:
    """What the switcher shows once it agrees with memory, read without a scan.

    An image model is never among the choices, so the switcher shows nothing
    while one is loaded; the badge is what names it. Reading the kind here
    rather than looking the ID up in the cache is what lets the timer's
    check stay a few attribute reads.
    """

    if runtime.MANAGER.image_loaded:
        return None
    return switch_value()


# How long an idle switcher goes before it re-reads whether what it offers
# still fits. The model in memory and the cache revision are attribute reads
# and are checked on every tick; fit is neither, and is also the one input to
# the list that nothing in ChatLab moves. Another process taking or giving
# back several gigabytes changes every verdict without touching the cache or
# the model in memory, and until the list is re-read it can offer a model the
# load would now refuse - which costs the reader the model they were talking
# to, because a load unloads before it checks (see
# ``ModelManager._load_locked``) - or go on hiding one that has become
# loadable again. Fifteen seconds is long enough that the scan and the memory
# reading are rare beside the two-second tick, and short enough that neither
# mistake stands for long.
SWITCH_FIT_SECONDS = 15.0


class SwitchStamp(NamedTuple):
    """What a tab's switcher was drawn from, so a later tick can date it.

    ``revision`` is the cache revision the choices were read at, ``offered``
    the model IDs they came to, and ``checked`` the moment the fit behind
    them was last read.
    """

    revision: int
    offered: tuple[str, ...]
    checked: float


def refresh_model_switch(precision: str | None = None):
    """Repaint the switcher: the loadable models, with the current one chosen.

    Hidden when there is nothing to offer, which is when the setup link
    beside it is the way forward.

    Returns the update and the stamp the choices were drawn at, which is what
    :func:`refresh_stale_model_switch` dates the list against on each tick.
    The revision is read before the scan, not after: a download that finishes
    while the scan is running is then seen as a change still to come rather
    than as one this list already has.
    """

    revision = runtime.MANAGER.cache_revision
    choices = switch_choices(precision)
    current = switch_value()
    ids = [value for _, value in choices]
    return (
        gr.update(
            choices=choices,
            value=current if current in ids else None,
            visible=bool(choices),
        ),
        SwitchStamp(revision, tuple(ids), time.monotonic()),
    )


def refresh_stale_model_switch(
    shown: str | None, stamp: SwitchStamp | None, precision: str | None = None
):
    """The timer's refresh: repaint only when the switcher has fallen behind.

    The badge's timer reads a few attributes; this one would scan the cache
    and read the machine's memory, and a repaint every couple of seconds
    would also close the list under a reader who has just opened it. So
    nothing is redrawn while the switcher is still right, which is nearly
    always.

    Three things can make it wrong. Two are an attribute read and are asked
    on every tick: the model in memory changed, so the wrong one is selected,
    and what a cache scan would find changed, so a model this tab has never
    heard of is missing from the list, or a deleted one is still in it, or one
    whose files are being rewritten is still offered.

    The third is fit, which no attribute records: the verdict behind every
    choice is a reading of the machine's free memory, and another process is
    free to move it. That one is re-read on its own slower beat
    (:data:`SWITCH_FIT_SECONDS`) because reading it is the expensive half of
    a repaint, and the re-read only redraws the dropdown when the models on
    offer actually changed - a list that came out the same is left exactly as
    it is, open or closed, and only its stamp moves on.

    ``stamp`` is what the tab last painted from; a tab that has not painted
    yet passes ``None`` and is repainted.
    """

    if stamp is None:
        return refresh_model_switch(precision)
    if (
        (shown or None) != expected_switch_value()
        or stamp.revision != runtime.MANAGER.cache_revision
    ):
        return refresh_model_switch(precision)
    if time.monotonic() - stamp.checked < SWITCH_FIT_SECONDS:
        return gr.skip(), gr.skip()
    update, fresh = refresh_model_switch(precision)
    if fresh.offered == stamp.offered:
        return gr.skip(), fresh
    return update, fresh


def switch_model(selected: str | None, precision: str = "full"):
    """Load the model picked in the switcher, at the Models page's precision.

    Yields the switcher's own update, the Models page's status card, and the
    badge beside the switcher, so the load shows there exactly as **Load
    cached** would show it while the reader who asked for it watches it fill
    where they are looking. The badge is written from here as well as by its
    own timer because the timer's beat is seconds wide: a pick that repainted
    nothing until the next tick reads as a click that did nothing, and the
    cards that carry the figures go to a page this reader is not on. A pick
    during a reply is refused and the switcher put back: the load would only
    queue behind the generation, and its first act on winning the lock would
    be to unload the model still producing the tokens.

    Both refusals are one reservation, not a pair of checks. A second load
    and a reply starting in the same instant are the same hazard read from
    two sides: the load itself claims nothing until ``stream_load``, several
    cards and a cache scan later, Gradio gives a picked-up-and-put-down
    handler no exclusivity across those yields, and a generation slot taken
    after this handler looked at it is a reply that will run on whatever
    this load brings in. ``claim_exclusive_load`` answers both questions
    under one lock and leaves a claim behind that turns away the next asker,
    whichever of the two it is; it names which of the two refused this one
    in the same breath, so the toast cannot go out over a load that has
    since ended. The claim stands for the whole of the load, the early
    refusals included, and is given back in the ``finally``.

    A load that is accepted and then comes to nothing is announced where the
    reader is rather than left on the Models page's card; see
    :func:`announce_switch_outcome`.
    """

    current = switch_value()
    if not selected or selected == current:
        yield gr.skip(), gr.skip(), gr.skip()
        return
    logger.info(
        "Model switch requested from %s to %s at %s weights",
        current or "no model",
        selected,
        precision,
    )
    try:
        claimed, held = runtime.MANAGER.claim_exclusive_load(selected)
    except ValueError as error:
        logger.warning("Cannot switch to %s: %s", selected, error)
        yield gr.update(value=current), failure_card(
            "Could not load cached model", html.escape(str(error))
        ), gr.skip()
        return
    if claimed is None:
        logger.info(
            "Switch to %s refused: %s has the model", selected, held or "another claim"
        )
        alarm(
            "Cannot switch models now",
            occupied_reason(held, SWITCH_LOADING, SWITCH_BUSY),
        )
        yield gr.update(value=current), gr.skip(), gr.skip()
        return
    _checked_id, claim = claimed
    last = None
    try:
        # The badge is redrawn on every card, which is every half second
        # while the weights are read (LOAD_POLL_SECONDS), so its bar moves at
        # the pace of the load rather than of the badge's own timer.
        for card in load_cached_model(selected, None, precision, claim):
            last = card
            yield gr.skip(), card, loaded_model_badge(kind=TEXT_KIND)
    finally:
        runtime.MANAGER.release_load(claim)
    announce_switch_outcome(last)
    # The load is over, one way or the other: say so without waiting for the
    # timer, which would leave "Loading…" on screen for another tick.
    yield gr.skip(), gr.skip(), loaded_model_badge(kind=TEXT_KIND)


def announce_switch_outcome(card: str | None) -> None:
    """Say, where the reader is, why an accepted switch did not happen.

    The load's cards go to the Models page's status area, because that is
    where a load reports its progress and the switcher's own repaint would
    otherwise drop it mid-load. But the reader who picked from the switcher
    is on the Chat page and can see none of it, so a switch that is accepted
    and then comes to nothing - the model removed, gone partial, or a
    redownload begun between the list being drawn and the pick - would show
    only as the badge falling back to "No model loaded", with no explanation
    anywhere the reader is looking.

    A toast is what the rest of the app raises for a failure a reader may not
    have their eyes on, and :func:`failure_card` already raises one for every
    failure it writes; those are marked as announced and left alone rather
    than told twice. What is left is the outcomes that only ever wrote a
    card, and the last card is the one that says how the load ended.
    """

    if not isinstance(card, Card) or card.announced or card.tone == "success":
        return
    alarm(card.title, card.detail)


def go_to_models():
    """Move the nav to Models and show that page, as clicking its tile does.

    The pages are switched here rather than left to the nav's own change
    handler: Gradio's Radio reports a change the visitor made, not one a
    handler wrote, so setting the nav alone would tick Models and leave the
    chat page on screen.
    """

    return MODELS_PAGE, *show_page(MODELS_PAGE)


def go_to_image_models(
    selected: str | None,
    order: str | None,
    precision: str | None,
    _kind: str | None,
    _name: str | None,
    model_id: str | None,
    query: str | None,
    hf_token: str,
    _search_precision: str | None,
    _search_kind: str | None,
    search_order: str,
    fits_only: bool,
    _search_model_id: str | None = None,
):
    """Open Models with both lists scoped to image models.

    The **Choose an image model** button on the Images page asks a narrower
    question than the Models tile does, and a cache of seventeen models in
    which two can draw does not answer it: the kind is a word mid-row, and
    the row it is missing from is the majority. So the filter and the search
    are set to image here, and both lists are repainted from that rather
    than left to a listener - the controls are written by this handler, and
    a control a handler writes reports no change of its own.

    A name typed in the My Models box is cleared on the way, since the
    question is now which models draw, and "qwen" left over from an earlier
    look would hide every one of them.

    The two kind choices and the name the page is already showing are taken
    and ignored, which keeps the wiring the same input lists the two lists
    take.
    """

    return (
        *go_to_models(),
        gr.update(value=IMAGE_KIND),
        gr.update(value=""),
        *refresh_my_models(selected, order, precision, IMAGE_KIND, None, model_id),
        gr.update(value=IMAGE_KIND),
        *search_models(query, hf_token, precision, IMAGE_KIND, search_order, fits_only, model_id),
    )


def select_model_to_load(model_id, title="Model selected", note=""):
    """Open ``model_id`` in the Models page's detail pane; loading stays a click away.

    Update the ID and the My Models selection together.
    Programmatic ID changes do not fire the typing listener that normally
    clears the selected row, which would otherwise override this ID.

    The ID is validated rather than trusted: an extension may be passing on
    one it read from a saved run, and a model ID cannot hold the characters
    that would turn the card below into something other than a quoted name.
    ``note`` is appended to the card for a caller with more to say about the
    model it named.
    """

    model_id = validate_model_id(model_id)
    return (
        model_id,
        *clear_my_model_selection(),
        status_card(
            title,
            f"`{model_id}` is selected. Choose **Load cached** to use local files, "
            "or **Download and load** to fetch the model and load it." + note,
        ),
        *go_to_models(),
    )


def select_default_model():
    """Select the default and open Models; loading requires a separate click."""

    return select_model_to_load(
        settings.DEFAULT_MODEL_ID,
        "Default model selected",
        f" A full download is {DEFAULT_MODEL_DOWNLOAD}. If local files are "
        "incomplete, **Download and load** can fetch the rest.",
    )


# The side pane's model lists.
# The My Models detail sits in the detail pane, under the model's own head,
# and says nothing until a downloaded model is chosen there.
NO_CACHED_MODEL_SELECTED = ""

# The My Models filter. The list is built from the whole cache, so a reader
# who never touches this sees every downloaded model; the other choices
# narrow it to the word the matching rows already wear. A snapshot ChatLab
# cannot load wears no kind, so it is only ever under All kinds.
ALL_KINDS = "all"
MODEL_KIND_FILTERS = (
    ("All kinds", ALL_KINDS),
    ("Text", TEXT_KIND),
    ("Image", IMAGE_KIND),
    ("MLX", MLX_KIND),
)
# How the summary line names a narrowed list. KIND_NAMES words the same
# kinds for a single model on the detail card, where the noun is singular.
KIND_FILTER_NAMES = {TEXT_KIND: "text", IMAGE_KIND: "image", MLX_KIND: "MLX"}


def format_timestamp(stamp: float | None) -> str:
    if stamp is None:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


UNSUPPORTED_REASON = (
    "not a model ChatLab loads: its files are all here, but it is not one of "
    "the kinds. ChatLab loads a Transformers causal language model, which "
    "has a `model.safetensors` or `pytorch_model.bin` at the top of the repo, "
    "a diffusers pipeline that draws from a prompt, which has a "
    "`model_index.json`, a tokenizer and a text encoder, and on Apple silicon "
    "a language model quantized for MLX. A CTranslate2 or ONNX export is "
    "none of these, and neither is a GGUF file; so is a diffusers pipeline "
    "that wants a picture, a video frame or a sound alongside the prompt, "
    "because the Images page has only a prompt to give it. An MLX model on a "
    "machine without Apple silicon and the mlx-lm package is unsupported for "
    "the same reason: nothing here runs it. A LoRA adapter for a Transformers "
    "language model, which has an `adapter_config.json`, loads too, merged "
    "into the base model it names."
)


# What each kind of model is, in the fewest words that distinguish them, for
# the list and the cards.
KIND_NAMES = {TEXT_KIND: "text model", IMAGE_KIND: "image model", MLX_KIND: "MLX text model"}


def where_to_use(kind: str) -> str:
    """Which page a freshly loaded model is used on, so nobody has to guess.

    The two kinds are loaded from the same place and driven from different
    ones, which is exactly the sort of thing a reader should be told at the
    moment it becomes true rather than left to find.
    """

    if kind == IMAGE_KIND:
        return "It draws pictures: go to the **Images** page to use it."
    return "It answers with tokens: go to the **Chat** page to use it."


# The one word each verdict gets in a list. A model whose size or whose
# machine could not be measured gets none: the detail beside the list says
# what is not known, and a list is the wrong place to explain it.
FIT_WORDS = {FITS: "fits", TIGHT: "tight", UNFIT: "won't fit"}

# What the estimate assumes when the device has not been read yet. Both
# accelerators load half precision; a load onto the CPU converts to float32
# and takes twice as much, so a verdict given before the device is known can
# be too generous by half. It is corrected as soon as the device is read -
# see ``refresh_after_device``.
ASSUMED_DTYPE = "float16"


def fit_word(fit: Fit | None) -> str:
    """The list's own one-word verdict, or nothing where there is none."""

    return FIT_WORDS.get(fit.state, "") if fit is not None else ""


def weight_bits(precision: str | None, profile: DeviceProfile) -> int | None:
    """The bit width a load would pack linear weights into, or ``None`` for full.

    A quantized choice is honoured on Apple Metal alone, so anywhere else the
    estimate is of full weights however the radio is set - which is what the
    load itself does. A device not read yet counts as somewhere else: of the
    two ways to be wrong for the few seconds before it is read, saying a
    model is tight when 4-bit would have fitted costs a reader nothing, while
    saying it fits when the load will refuse it is the disagreement these
    verdicts exist to prevent.
    """

    if not profile.quantizes:
        return None
    return QUANTIZED_BITS.get(precision or "full")


def requested_bits(
    precision: str | None, profile: DeviceProfile, kind: str | None
) -> int | None:
    """:func:`weight_bits`, except where the radio has no say over the width.

    A pipeline is estimated whole whatever the radio says, because the Metal
    quantizer is Transformers' own and ``_load_locked`` clears the choice for
    one. An MLX repo was packed when it was converted and loads at that
    width, so the radio has nothing to add there either. A verdict that
    carried the bits anyway would put a quantized label on a figure not
    measured at one, and moving the radio would mark the loaded model as
    being about to reload when nothing would change.
    """

    if kind in (IMAGE_KIND, MLX_KIND):
        return None
    return weight_bits(precision, profile)


def packed_bits(
    snapshot: Path | None, kind: str | None, requested: int | None
) -> int | None:
    """The width a load of ``snapshot`` will really pack its linear layers into.

    ``requested`` for anything but MLX, where the radio decides as far as
    the device allows. An MLX repo answers for itself, out of the config
    ``mlx_lm.convert`` wrote: that is the width the estimate is of and the
    width the verdict has to name, because saying "full 16-bit weights"
    over a 4-bit conversion's figure would misread it by four times.
    """

    if kind == TEXT_KIND and is_adapter_snapshot(snapshot):
        # Merged into full-precision weights whatever the radio says.
        return None
    if kind != MLX_KIND or snapshot is None:
        return requested
    return mlx_snapshot_bits(snapshot)


def held_as_asked(
    model_id: str, precision: str | None, profile: DeviceProfile, kind: str | None
) -> bool:
    """Whether ``model_id`` is the model in memory, read at the width the radio asks for.

    Such a model has no verdict to give: its memory is a reading, not an
    estimate. What the radio asks of a pipeline or an MLX repo is nothing,
    so moving it asks nothing new of either, and neither is marked as about
    to reload.
    """

    return runtime.MANAGER.model_id == model_id and requested_bits(
        precision, profile, kind
    ) == requested_bits(runtime.MANAGER.precision, profile, kind)


def cached_fit(
    entry: CachedModel, precision: str | None, profile: DeviceProfile
) -> Fit | None:
    """Whether ``entry`` would load now, or ``None`` where there is nothing to judge.

    A model short of files has no size to measure until the rest arrives, an
    unsupported one will not load whatever the memory says, and the model
    already in memory has answered the question by being there - judging it
    against what is left free would call the loaded model tight.

    Being there only answers for the weights it was read as, though. **Load
    cached** on the model in memory is how a new precision is applied, so a
    reader who has moved that radio is asking about a load that has not
    happened, and the model that fits at four bits may not fit whole.
    """

    if entry.status.missing_files or entry.status.unsupported:
        return None
    kind = entry.status.kind
    if held_as_asked(entry.model_id, precision, profile, kind):
        return None
    snapshot = snapshot_folder(entry.path) if entry.path is not None else None
    if snapshot is None:
        return None
    # The size depends on the kind too: a pipeline has no checkpoint at its
    # root to measure, and an MLX repo is measured as the packed file it
    # already is. The pool is already the one for this kind - the caller
    # chose it, because choosing it here would re-read the device and
    # discard the memory the impending unload gives back.
    bits = packed_bits(snapshot, kind, requested_bits(precision, profile, kind))
    estimated = estimate_snapshot_bytes(
        snapshot, profile.dtype or ASSUMED_DTYPE, bits, kind
    )
    return model_fit(estimated, profile, bits)


def replacement_profile(kind: str = TEXT_KIND) -> DeviceProfile:
    """The machine as a model about to be loaded would find it.

    Every model a verdict is given for is one that would replace whatever is
    in memory, and a load unloads first and only then checks whether the next
    model fits. So the weights on the device now are counted as available;
    without that, a 15 GB model already loaded would have every alternative
    marked tight and the button would then load them anyway.

    ``for_kind`` does both the pool and the reclamation, because for a pool
    that is the tighter of two the unload has to be counted into each side
    before they are collapsed; doing it afterwards credits card memory to
    whichever pool happened to be smaller.
    """

    return device_profile().for_kind(kind, runtime.MANAGER.loaded_bytes)


def cached_fits(
    models: list[CachedModel], precision: str | None
) -> dict[str, Fit]:
    """The fit verdict for each of ``models``, by model ID, one reading per kind.

    A reading per kind rather than one for the whole list, because an image
    pipeline on CUDA is judged against a different pool; see
    :meth:`DeviceProfile.for_kind`. Taken lazily and kept, so a list of only
    text models still costs the one reading it always did - reading host
    memory is a subprocess, and this runs on every rescan.
    """

    profiles: dict[str, DeviceProfile] = {}
    fits = {}
    for entry in models:
        kind = entry.status.kind or TEXT_KIND
        if kind not in profiles:
            profiles[kind] = replacement_profile(kind)
        fit = cached_fit(entry, precision, profiles[kind])
        if fit is not None:
            fits[entry.model_id] = fit
    return fits


def hub_fit(
    result: HubModel,
    precision: str | None,
    profile: DeviceProfile,
    kind: str = TEXT_KIND,
) -> Fit:
    """Whether ``result`` would load now, judged from the hub's parameter count.

    The count is all a search result carries, so the estimate assumes a
    half-precision checkpoint and, for a quantized load, that the embeddings
    are packed with everything else. Both are close enough to tell a model
    that fits from one that cannot; the detail says where the figure came
    from.

    An image pipeline is judged at full precision whatever the radio says.
    The Metal quantizer is Transformers' own and ``_load_locked`` clears the
    choice for a pipeline, so honouring it here would shrink the estimate
    for a load that will not shrink and advertise a fit the load refuses.
    An MLX repo is judged at the width its name claims, again whatever the
    radio says, because that is the width it was converted to.
    """

    if not result.parameters:
        return model_fit(None, profile)
    if kind == MLX_KIND:
        # Packed already, at whatever width the converter chose; the radio
        # has no say. The width is read off the repository's name, which is
        # how mlx-community spells it, and a name that does not say is
        # judged whole rather than at a width it may not have. It is the
        # width the verdict names too, so a 4-bit conversion is not
        # described as full weights.
        bits = mlx_bits_from_id(result.model_id)
    else:
        bits = requested_bits(precision, profile, kind)
    estimated = estimate_parameter_bytes(
        result.parameters, profile.dtype or ASSUMED_DTYPE, bits
    )
    return model_fit(estimated, profile, bits)


def hub_fits(
    results: list[HubModel], precision: str | None, kind: str | None = None
) -> dict[str, Fit]:
    """The fit verdict for each search result, by model ID, one reading per kind.

    A list holds text models and MLX conversions together, and an image
    pipeline on CUDA is judged against a different pool, so each result is
    judged as its own kind (see :meth:`DeviceProfile.for_kind`) unless
    ``kind`` names one for them all. The estimate itself is from the
    parameter count either way, which says nothing about how the weights
    are laid out.
    """

    profiles: dict[str, DeviceProfile] = {}
    fits = {}
    for result in results:
        each = kind or model_kind(result)
        if each not in profiles:
            profiles[each] = replacement_profile(each)
        fits[result.model_id] = hub_fit(result, precision, profiles[each], each)
    return fits


def cached_model_label(entry: CachedModel, fit: Fit | None = None) -> str:
    """``org/name · 15 GB · image · fits``, flagged by kind, fit and state."""

    label = f"{entry.model_id} · {format_bytes(entry.size_bytes)}"
    if entry.status.kind == IMAGE_KIND:
        # Only the exceptions are flagged. Text models are the majority and
        # the default, so labelling every kind would put a word on every row
        # to distinguish the exceptions.
        label += " · image"
    elif entry.status.kind == MLX_KIND:
        label += " · MLX"
    elif entry.status.base_model is not None:
        label += " · LoRA"
    verdict = fit_word(fit)
    if verdict:
        label += f" · {verdict}"
    if entry.status.missing_files:
        label += " · incomplete"
    elif entry.status.unsupported:
        label += " · unsupported"
    if runtime.MANAGER.model_id == entry.model_id:
        label += " · loaded"
    return label


def describe_cached_model(entry: CachedModel, fit: Fit | None = None) -> str:
    if runtime.MANAGER.model_id == entry.model_id:
        verdict = f"**Loaded now** on {runtime.MANAGER.device_name}."
    elif entry.status.missing_files:
        verdict = (
            f"**Incomplete:** {describe_missing(entry.status)}. "
            "Use **Download and load** to fetch the rest."
        )
    elif entry.status.unsupported:
        verdict = f"**Unsupported:** {entry.status.unsupported_reason or UNSUPPORTED_REASON}"
    else:
        verdict = (
            "**Downloaded · Ready to load.** Use **Load cached** to bring it "
            f"into memory. {where_to_use(entry.status.kind)}"
        )
    facts = [("On disk", describe_on_disk(entry.status))]
    if fit is not None and fit.known:
        facts.append(("Memory", fit.note))
    if entry.files:
        facts.append(("Files", f"{entry.files} in the current snapshot"))
    facts.append(("Kind", KIND_NAMES.get(entry.status.kind, "not loadable here")))
    if entry.status.base_model is not None:
        facts.append(
            ("Adapter", f"LoRA for `{entry.status.base_model}`, merged in at full precision")
        )
    if entry.architecture:
        model_type = entry.architecture
        if entry.dtype:
            model_type += f" ({entry.dtype})"
        facts.append(("Architecture", model_type))
    if entry.commit:
        facts.append(("Revision", f"`{entry.commit[:7]}`"))
    facts.append(("Updated", format_timestamp(entry.updated)))
    if entry.path is not None:
        facts.append(("Folder", f"`{entry.path}`"))
    rows = "\n".join(f"- **{name}:** {value}" for name, value in facts)
    return f"{verdict}\n\n{rows}"


def cached_models_of_kind(models: list[CachedModel], kind: str | None) -> list[CachedModel]:
    """``models`` narrowed to one kind; All kinds and an unknown choice keep all.

    A snapshot ChatLab cannot load has no kind, so it is kept only by All
    kinds - the same rule the list's own labels follow, where an unsupported
    row says so instead of naming a kind.
    """

    if not kind or kind == ALL_KINDS:
        return list(models)
    return [entry for entry in models if entry.status.kind == kind]


def cached_models_named(models: list[CachedModel], name: str | None) -> list[CachedModel]:
    """``models`` whose ID holds every word of ``name``, ignoring case.

    Each word is matched on its own, anywhere in the ID, so "qwen 7b" finds
    ``Qwen/Qwen2.5-7B-Instruct`` without the reader having to know where the
    organization ends or how the size is spelled. An empty box keeps all.
    """

    words = (name or "").lower().split()
    return [
        entry for entry in models
        if all(word in entry.model_id.lower() for word in words)
    ]


def my_models_summary(
    models: list[CachedModel],
    shown: list[CachedModel] | None = None,
    kind: str | None = None,
    name: str | None = None,
) -> str:
    """The line above the list: what the cache holds, and what the filters hide.

    The count and the size stay about the whole cache even while a kind is
    chosen or a name typed, because the disk figure is about the folder
    rather than about what is on screen. A filter that matches nothing says
    so, and says where to go instead: that is the answer a reader who came
    from the Images page with no image model downloaded needs.
    """

    root = f"`{cache_root()}`"
    if not models:
        return (
            f"No models in the Hugging Face cache yet ({root}). "
            "Find one under **Find a model**."
        )
    seen: set[Path] = set()
    total = format_bytes(sum(
        folder_bytes(entry.path, seen=seen) if entry.path is not None else entry.size_bytes
        for entry in models
    ))
    count = f"{len(models)} model{'s' if len(models) != 1 else ''}"
    line = f"{count} · {total} on disk in {root}"
    if shown is None or len(shown) == len(models):
        return line
    kind_name = KIND_FILTER_NAMES.get(kind)
    typed = " ".join((name or "").split())
    matching = f"matching `{typed}`" if typed else ""
    if not shown:
        described = " ".join(filter(None, [kind_name, "models", matching]))
        return f"No {described} among the {line}. Find one under **Find a model**."
    described = " ".join(filter(None, [kind_name, matching])) or "matching"
    return f"Showing {len(shown)} {described} · {line}"


def refresh_my_models(
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    precision: str | None = None,
    kind: str | None = ALL_KINDS,
    name: str | None = None,
    model_id: str | None = None,
):
    """Rescan the cache; keep the selected row or typed ID, or the loaded model.

    ``precision`` is the **Load at** choice, which decides what each
    model would take in memory and so whether it fits. The list is repainted
    when that choice changes, which is what makes the radio the first thing
    to try when a model will not load.

    ``kind`` is the **Kind** choice, which narrows the rows to text, image or
    MLX models, and ``name`` is the box beside it, which narrows them to IDs
    holding every word typed. A row a filter hides cannot stay selected, so a
    selection it hides is dropped the same way one removed from disk is.
    """

    everything = sort_cached_models(list_cached_models(), order)
    models = cached_models_named(cached_models_of_kind(everything, kind), name)
    fits = cached_fits(models, precision)
    ids = [entry.model_id for entry in models]
    if selected not in ids:
        fallback = model_id.strip() if model_id is not None else runtime.MANAGER.model_id
        selected = fallback if fallback in ids else None
    choices = [
        (cached_model_label(entry, fits.get(entry.model_id)), entry.model_id)
        for entry in models
    ]
    if selected is None:
        detail = NO_CACHED_MODEL_SELECTED if models else ""
    else:
        entry = next(entry for entry in models if entry.model_id == selected)
        detail = describe_cached_model(entry, fits.get(selected))
    return (
        gr.update(choices=choices, value=selected),
        detail,
        my_models_summary(everything, models, kind, name),
    )


def select_my_model(selected: str | None, precision: str | None = None):
    """Put the chosen cached model in the ID box and describe it.

    The profile is taken for the row's own kind, the same as
    :func:`cached_fits` takes it for the list. Without that, an MLX
    conversion on a Mac with a Metal ceiling would be judged here against
    PyTorch's cap while the list and the button judge it against the
    machine, so selecting a row that reads *fits* would describe it as tight
    or unfit.
    """

    if not selected:
        return gr.skip(), NO_CACHED_MODEL_SELECTED
    entry = next(
        (entry for entry in list_cached_models() if entry.model_id == selected), None
    )
    if entry is None:
        return gr.skip(), f"`{selected}` is no longer in the cache. Press **Refresh**."
    profile = replacement_profile(entry.status.kind or TEXT_KIND)
    return (
        gr.update(value=selected),
        describe_cached_model(entry, cached_fit(entry, precision, profile)),
    )


def clear_my_model_selection():
    """Drop the My Models selection, because a typed ID names its own model.

    Runs when the reader types in the ID box, picks a search result, or selects
    the default, so ``chosen_model`` uses that ID rather than a previous row.
    It is a round trip like any other, which is the window
    ``chosen_model`` describes rather than closes.
    """

    return gr.update(value=None), NO_CACHED_MODEL_SELECTED


NO_MODEL_TO_MANAGE = "Select a model under **Downloaded** first."


def redownload_my_model(selected: str | None, hf_token: str):
    """Fetch whatever the chosen cached model still lacks.

    ``snapshot_download`` skips finished files and resumes partial ones, so
    for an incomplete model this fetches the rest, and for a complete one it
    checks the hub for updated files. Remove the model first to start over.

    The loaded model is refused, and so is one being loaded. A redownload
    can move ``refs/main`` to a newer revision while the old weights stay in
    memory (or are still being read), and the list marks a model loaded by
    ID alone, so it would then call the new snapshot "loaded" while every
    reply still came from the old one. An incomplete model cannot be loaded,
    so the refusal never stands in the way of finishing a download.

    ``is_loading`` rather than the name in the badge, because with two loads
    running at once only one of them is named there, and a load still waiting
    its turn for the model lock is going to read that model's files just the
    same.
    """

    if not selected:
        yield status_card("Nothing to redownload", NO_MODEL_TO_MANAGE)
        return
    if runtime.MANAGER.is_loading(selected):
        yield status_card(
            "Model in use",
            f"`{selected}` is being loaded right now. Wait for the load to finish, "
            "then **Unload** it before redownloading.",
        )
        return
    if runtime.MANAGER.model_id == selected:
        yield status_card(
            "Model in use",
            f"`{selected}` is loaded in memory. **Unload** it before redownloading, "
            "so the weights in memory and the files on disk stay the same revision.",
        )
        return
    yield from download_model(selected, hf_token)


def loaded_refusal(selected: str) -> tuple[str, str]:
    return (
        "Model in use",
        f"`{selected}` is loaded in memory. **Unload** it before removing its files.",
    )


def downloading_refusal(selected: str) -> tuple[str, str]:
    return (
        "Still downloading",
        f"`{selected}` is being downloaded. Wait for it to finish, then remove it.",
    )


def remove_my_model(name: str | None):
    """Delete ``name`` from the cache and report the space freed.

    ``name`` is the model whose row's Remove was pressed twice, not the
    radio's selection, so the model deleted is always the one the reader
    pressed on. The deletion goes through :meth:`ModelManager.remove`, which
    refuses a model that is loaded, downloading or busy under the manager's
    locks.
    """

    if not name:
        return status_card("Nothing to remove", NO_MODEL_TO_MANAGE)
    logger.info("Removal confirmed for %s", name)
    try:
        freed = runtime.MANAGER.remove(name)
    except ModelLoaded:
        logger.info("Removal of %s refused: it is loaded", name)
        return status_card(*loaded_refusal(name))
    except ModelDownloading:
        logger.info("Removal of %s refused: it is downloading", name)
        return status_card(*downloading_refusal(name))
    except ModelBusy:
        logger.info("Removal of %s refused: the manager is busy", name)
        return status_card(
            "Model busy",
            f"`{name}` cannot be removed while a model is loading, generating, "
            "scoring, or being inspected. Try again when it is idle.",
        )
    except FileNotFoundError:
        logger.info("Removal of %s found nothing: it is no longer cached", name)
        return status_card("Nothing to remove", f"`{name}` is no longer in the cache.")
    except (OSError, ValueError) as error:
        logger.warning("Could not remove %s", name, exc_info=True)
        return failure_card(
            "Could not remove model",
            f"Removing `{name}` failed: {html.escape(str(error))}",
        )
    logger.info("Removed %s from the cache, freeing %s", name, format_bytes(freed))
    return status_card(
        "Model removed",
        f"Removed `{name}` from the Hugging Face cache, freeing {format_bytes(freed)}.",
        "success",
    )


def act_on_my_model(action: str | None, hf_token: str):
    """Redownload or remove the model a My Models row's button named.

    ``action`` is what the row script in ui.model_rows writes to its bridge:
    the model's ID, which button was pressed, and a nonce. Anything else is
    ignored rather than guessed at.
    """

    try:
        request = json.loads(action or "")
    except ValueError:
        request = None
    if not isinstance(request, dict) or not isinstance(request.get("name"), str):
        yield gr.skip()
        return
    name = request["name"]
    if request.get("action") == "redownload":
        yield from redownload_my_model(name, hf_token)
    elif request.get("action") == "remove":
        card = remove_my_model(name)
        # The status card can be below a long list. Announce the outcome
        # where the reader clicked, including why a removal was refused.
        if card.tone == "success":
            gr.Info(card.detail, title=card.title)
        else:
            announce_switch_outcome(card)
        yield card
    else:
        yield gr.skip()


# -- finding a model ------------------------------------------------------------

# The libraries each kind is searched under, as the results list names them.
LIBRARY_NAMES = {TEXT_KIND: "Transformers", MLX_KIND: "MLX", IMAGE_KIND: "diffusers"}


def search_kinds_for(kind: str | None) -> tuple[str, ...]:
    """The searches a choice of kind runs.

    Text is two where MLX runs: the hub files MLX conversions under their
    own library, and they are text models packed for Apple silicon, so they
    belong in the same list. Elsewhere they would land in the cache as
    unsupported, so they are not searched for.
    """

    if kind == IMAGE_KIND:
        return (IMAGE_KIND,)
    return (TEXT_KIND, MLX_KIND) if mlx_available() else (TEXT_KIND,)


def starter_picks(kind: str | None) -> list[HubModel]:
    """ChatLab's own picks for a kind, MLX ones beside the text ones where MLX runs."""

    return [model for each in search_kinds_for(kind) for model in STARTER_MODELS.get(each, ())]


def downloaded_ids(results: list[HubModel]) -> set[str]:
    """Which of ``results`` are fully on disk, so their rows can say so."""

    downloaded = set()
    for result in results:
        try:
            if cache_status(result.model_id).complete:
                downloaded.add(result.model_id)
        except (OSError, ValueError):
            # A cache that cannot be read is simply nothing on disk: the
            # search succeeded, so its rows must draw.
            continue
    return downloaded


def draw_results(
    listing: Listing | None, precision: str | None, fits_only: bool, selected: str | None
) -> str:
    """The results card for ``listing``, judged at ``precision`` and narrowed by the fit filter."""

    if listing is None:
        return results_list(
            None, {}, selected=None, downloaded=set(), shown=[], hidden_by_fit=0, fits_only=False
        )
    results = list(listing.results.values())
    fits = hub_fits(results, precision)
    visible = [
        result for result in results
        if not fits_only or (result.model_id in fits and fits[result.model_id].state == FITS)
    ]
    shown = visible[:SEARCH_LIMIT]
    return results_list(
        listing,
        fits,
        selected=(selected or "").strip() or None,
        downloaded=downloaded_ids(shown),
        shown=shown,
        hidden_by_fit=len(results) - len(visible) if fits_only else 0,
        fits_only=fits_only,
    )


def search_models(
    query: str | None,
    hf_token: str,
    precision: str | None = None,
    kind: str = TEXT_KIND,
    order: str = DEFAULT_SEARCH_ORDER,
    fits_only: bool = False,
    model_id: str | None = None,
):
    """Show ChatLab's picks for an empty box, or search Hugging Face for what is typed.

    The picks need no network, which is what the page loads with. A search
    keeps up to DISCOVERY_CANDIDATES results so the fit filter can narrow
    them without searching again, and draws SEARCH_LIMIT of them. A pick the
    search also finds is listed once, wearing both what the catalog and the
    hub know about it; see merged_starter.

    Returns the results card, the listing behind it, and the sort control,
    which is shown only for a search: picks have no sort.
    """

    cleaned = (query or "").strip()
    order = order if order in HUB_SORTS else DEFAULT_SEARCH_ORDER
    sort_control = gr.update(visible=bool(cleaned))
    picks = {model.model_id: model for model in starter_picks(kind)}
    if not cleaned:
        listing = Listing(picks)
        return draw_results(listing, precision, fits_only, model_id), listing, sort_control
    kinds = search_kinds_for(kind)
    libraries = tuple(LIBRARY_NAMES[each] for each in kinds)
    try:
        found = search_hub_kinds(cleaned, hf_token, kinds, order, DISCOVERY_CANDIDATES)
    except Exception as error:
        # With the stack, because this catches everything: a Hub that is
        # simply unreachable, and a mistake in the search itself.
        logger.warning("Hub search for %r failed", cleaned, exc_info=True)
        listing = Listing(
            {}, HUB_HEADING, "", query=cleaned, libraries=libraries,
            message=(
                f"Could not search Hugging Face: {html.escape(str(error))}. "
                "Clear the box to see ChatLab picks offline, or try again."
            ),
        )
        return draw_results(listing, precision, fits_only, model_id), listing, sort_control
    results = {
        result.model_id: (
            merged_starter(picks[result.model_id], result) if result.model_id in picks else result
        )
        for result in found.results
    }
    described = "image models" if kind == IMAGE_KIND else "language models"
    listing = Listing(
        results,
        HUB_HEADING,
        ORDER_NOTES[order],
        found.exclusions,
        found.scanned,
        found.stopped_early,
        cleaned,
        "" if results else f"No {described} matched <code>{html.escape(cleaned)}</code>.",
        libraries,
    )
    return draw_results(listing, precision, fits_only, model_id), listing, sort_control


def search_and_open(
    query: str | None,
    hf_token: str,
    precision: str | None = None,
    kind: str = TEXT_KIND,
    order: str = DEFAULT_SEARCH_ORDER,
    fits_only: bool = False,
    model_id: str | None = None,
):
    """Enter in the search box: search, and open the model if what was typed is an ID.

    The model is opened through the pick bridge, the way a pressed row is,
    so the same chain selects and checks it. A pasted ID opens even when the
    search leaves it out, which is how a reader learns why it was left out.
    """

    drawn, listing, sort_control = search_models(
        query, hf_token, precision, kind, order, fits_only, model_id
    )
    cleaned = (query or "").strip()
    if "/" not in cleaned:
        return drawn, listing, sort_control, gr.skip()
    try:
        opened = validate_model_id(cleaned)
    except ValueError:
        return drawn, listing, sort_control, gr.skip()
    opened = next(
        (found for found in listing.results if found.lower() == opened.lower()), opened
    )
    return drawn, listing, sort_control, pick_request(opened)


def pick_request(model_id: str) -> str:
    """What the pick bridge carries: the model, and a nonce so a repeat still changes it."""

    return json.dumps({"model": model_id, "nonce": time.time_ns()})


def merged_starter(starter: HubModel, live: HubModel | None) -> HubModel:
    """A bundled starter wearing what the Hub’s own answer adds to it.

    The two describe one repository from different sides. The catalog has the
    curated note and the download estimate, which a search result never
    carries; the search has the downloads, the likes and the date, which the
    catalog cannot keep current. Keeping one and dropping the other would take
    facts off the row that the same search shows for every other result, so
    the row is both: the live answer, with the catalog filling what the Hub
    left empty.
    """

    if live is None:
        return starter
    # A search result whose repository publishes no safetensors index has no
    # parameter count, and the hub omits a tag or a license as readily; the
    # catalog’s copy is older than the hub’s but better than none.
    stale = {
        name: getattr(starter, name)
        for name in ("parameters", "pipeline_tag", "library", "license", "last_modified")
        if getattr(live, name) is None
    }
    return replace(
        live, **stale, summary=starter.summary, download_bytes=starter.download_bytes,
        pick=True,
    )


def select_search_result(pick: str | None):
    """Open the model a row or an Other versions entry named.

    ``pick`` is what the row script in ui.model_finder writes to its bridge:
    the model's ID and a nonce. Anything else is ignored rather than guessed
    at, and an ID is validated rather than trusted, since it came from the page.
    Its row is marked by the redraw that follows the ID box, as every other
    way of choosing a model is.
    """

    try:
        request = json.loads(pick or "")
        model = request["model"] if isinstance(request, dict) else None
        if not isinstance(model, str):
            raise ValueError("No model named")
        model = validate_model_id(model)
    except (ValueError, KeyError):
        return gr.skip()
    return model


def refresh_search_results(
    model_id: str | None, listing: Listing | None, precision: str | None = None,
    fits_only: bool = False,
):
    """Redraw the results from the listing in hand, for a new choice, precision or fit filter."""

    if listing is None:
        return gr.skip()
    return draw_results(listing, precision, fits_only, model_id)


def known_model(
    model_id: str, listing: Listing | None, related: dict | None
) -> HubModel | None:
    """What a search said about ``model_id``, from the list or from Other versions."""

    if listing is not None and model_id in listing.results:
        return listing.results[model_id]
    if related and related.get("model_id") == model_id and related.get("model") is not None:
        return related["model"]
    for version, _ in related_versions(related, model_id, any_owner=True):
        if version.model_id == model_id:
            return version
    return next((model for model in starter_picks(TEXT_KIND) + starter_picks(IMAGE_KIND)
                 if model.model_id == model_id), None)


def related_versions(
    related: dict | None, model_id: str, *, any_owner: bool = False
) -> list[tuple[HubModel, bool]]:
    """The Other versions found for ``model_id``, or for whichever model they were found for."""

    if not related or (not any_owner and related.get("model_id") != model_id):
        return []
    return [(version, original) for version, original in related.get("versions", [])]


def find_versions(
    model_id: str | None, hf_token: str, listing: Listing | None, related: dict | None = None
):
    """Look up the other versions of the opened model, for the pane.

    A Transformers model is offered its MLX conversions from the publishers
    in VERIFIED_CONVERTERS. An MLX conversion is offered the model it was
    converted from, which its own tags name. Failure to reach the hub is no
    reason to say anything: the section is left out.

    What was known about the model itself is kept beside its versions: a
    model opened from Other versions is in no list, and the lookup that
    found it is the one this replaces.
    """

    chosen = (model_id or "").strip()
    if not chosen:
        return None
    result = known_model(chosen, listing, related)
    found = {"model_id": chosen, "model": result, "versions": []}
    if result is not None and model_kind(result) == IMAGE_KIND:
        return found
    if result is not None and model_kind(result) == MLX_KIND:
        if result.base_model:
            original = HubModel(model_id=result.base_model, parameters=result.parameters)
            found["versions"] = [(original, True)]
        return found
    try:
        versions = mlx_versions(chosen, hf_token)
    except Exception:
        logger.warning("Could not look up MLX versions of %s", chosen, exc_info=True)
        versions = []
    # A conversion keeps the parameter count of the model it was packed
    # from, which the hub often lists for the original alone.
    if result is not None and result.parameters:
        versions = [
            version if version.parameters else replace(version, parameters=result.parameters)
            for version in versions
        ]
    found["versions"] = [(version, False) for version in versions]
    return found


def pane_kind(result: HubModel | None, checked: dict, cached: CacheStatus) -> str:
    """The kind the pane describes the chosen model as, from whatever has been learnt."""

    # An empty cache reads as a text model, so only files on disk count.
    if cached.complete and cached.kind:
        return cached.kind
    if checked.get("mlx"):
        return MLX_KIND
    if checked.get("format") == "Image model":
        return IMAGE_KIND
    if checked.get("status") == "found" and checked.get("config_verified") and checked.get("format") in ("Transformers", "LoRA adapter"):
        return TEXT_KIND
    if result is not None:
        return model_kind(result)
    return TEXT_KIND


def pane_fit(
    model_id: str, result: HubModel | None, cached: CacheStatus,
    precision: str | None, profile: DeviceProfile, kind: str,
) -> Fit | None:
    """The memory verdict for the pane: from the files on disk if they are all there, else the hub's count."""

    # The model in memory has none, and the search listing carrying it too
    # must not give it one: the pane reads what it holds instead.
    if held_as_asked(model_id, precision, profile, kind):
        return None
    if cached.complete:
        entry = next(
            (entry for entry in list_cached_models() if entry.model_id == model_id), None
        )
        if entry is not None:
            fit = cached_fit(entry, precision, profile)
            if fit is not None:
                return fit
    if result is not None and result.parameters:
        return hub_fit(result, precision, profile, kind)
    return None


def repository_notes(checked: dict) -> list[str]:
    """What checking the model on Hugging Face found, for the pane's last section."""

    status = checked.get("status")
    if not status or status == "checking":
        return []
    if status != "found":
        return [html.escape(checked["detail"])]
    notes = [
        f"<b>Repository found</b> · {html.escape(checked['format'])}",
        html.escape(checked["compatibility"]),
    ]
    if checked.get("access_restricted"):
        notes.append(
            "<b>Access required:</b> accept the model's terms on Hugging Face and "
            "provide an authorized token under <b>Access token</b>."
        )
    elif checked.get("gated"):
        notes.append(
            "Gated repository · Access to the configuration was verified."
            if checked.get("config_verified") else "Gated repository · File access has not been verified."
        )
    elif checked.get("private"):
        notes.append("Private repository · your token provided access.")
    return notes


def pane_cache_note(cached: CacheStatus, kind: str) -> str:
    """One line on what of the chosen model is on disk, for the pane."""

    if cached.complete:
        page = "the <b>Images</b> page" if kind == IMAGE_KIND else "the <b>Chat</b> page"
        return f"Downloaded · {html.escape(describe_on_disk(cached))}. Load it to use it on {page}."
    if cached.unsupported:
        return (
            f"Downloaded ({html.escape(describe_on_disk(cached))}), but not a model ChatLab can load."
        )
    if cached.present:
        return (
            f"Partly downloaded · {html.escape(describe_on_disk(cached))}. "
            "<b>Download and load</b> fetches the rest."
        )
    return ""


def model_pane(
    model_id: str | None,
    selected: str | None,
    repository: dict | None,
    hf_token: str | None,
    listing: Listing | None,
    related: dict | None,
    precision: str | None,
):
    """The detail pane for the chosen model: what it is, what it needs, what works.

    Drawn from whatever is known: the search result, the check on Hugging
    Face, and the files on disk. Returns the pane's head, the precision
    control (shown only where precision is a choice: a Transformers text
    model), the pane's body, and the check button (shown until the model
    has been checked, or when the check could not reach the hub).
    """

    chosen = chosen_model(model_id or "", selected)
    if not chosen:
        return PANE_EMPTY, gr.update(visible=True), "", gr.update(visible=False)
    result = known_model(chosen, listing, related)
    checked = matching_repository(chosen, repository, hf_token)
    try:
        cached = cache_status(chosen)
    except (OSError, ValueError):
        cached = CacheStatus()
    # A cached MLX checkpoint says its own width, offline, and is packed at
    # it whether or not this machine can run it.
    snapshot = None
    try:
        snapshot = snapshot_folder(cache_folder(chosen)) if cached.present else None
        local_bits = mlx_snapshot_bits(snapshot) if snapshot is not None else None
    except (OSError, ValueError):
        local_bits = None
    kind = MLX_KIND if local_bits is not None else pane_kind(result, checked, cached)
    adapter = bool(result is not None and result.adapter) or checked.get("format") == "LoRA adapter" or is_adapter_snapshot(snapshot)
    effective_precision = "full" if adapter else precision
    profile = replacement_profile(kind)
    if kind == MLX_KIND:
        bits = local_bits or checked.get("bits") or mlx_bits_from_id(chosen)
    else:
        bits = requested_bits(effective_precision, profile, kind)
    loaded = runtime.MANAGER.model_id == chosen
    fit = pane_fit(chosen, result, cached, effective_precision, profile, kind)
    status = checked.get("status")
    download = checked.get("download_bytes") or (result.download_bytes if result else None)
    versions = [
        (version, hub_fit(version, precision, replacement_profile(model_kind(version)), model_kind(version)), original)
        for version, original in related_versions(related, chosen)
    ]
    gated = bool(
        (result is not None and (result.gated or result.base_gated)) or checked.get("gated")
        or checked.get("access_restricted") or status == "restricted"
    )
    head = pane_head(chosen, result, gated=gated, loaded=loaded)
    body = pane_body(
        kind=kind,
        bits=bits,
        vision=result is not None and result.pipeline_tag in VISION_TAGS,
        download_bytes=download,
        checking=status == "checking",
        checked=status == "found",
        fit=fit,
        loaded_bytes=runtime.MANAGER.loaded_bytes if loaded else None,
        on_disk_bytes=cached.cached_bytes if cached.complete else None,
        versions=versions,
        notes=repository_notes(checked) + (["LoRA adapter · Downloading also fetches its base model; ChatLab merges it into full-precision weights regardless of Load at."] if adapter else []) + (["Accept the base model's terms on Hugging Face and provide an authorized token."] if result is not None and result.base_gated else []),
        on_disk="" if selected else pane_cache_note(cached, kind),
        blocked=bool(
            checked.get("unsupported") or checked.get("architecture_unavailable")
            or cached.unsupported
        ),
    )
    check = gr.update(
        visible=status in (None, "error"),
        value="Check again" if status == "error" else "Check on Hugging Face",
    )
    return head, gr.update(visible=kind == TEXT_KIND), body, check


def refresh_after_device(
    known: bool,
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    precision: str | None = None,
    kind: str | None = ALL_KINDS,
    name: str | None = None,
    model_id: str | None = None,
    listing: Listing | None = None,
    fits_only: bool = False,
):
    """Repaint both model lists once the device is known, and only then.

    The page is painted before torch has finished importing, so the first
    verdicts are given without knowing the device: they assume half
    precision and no quantization, which is the safe way to be wrong but is
    wrong on a Mac with 4-bit chosen. A search run in those first seconds
    carries the same provisional verdicts, so it is repainted here too,
    from the results already in hand rather than by searching again. This
    runs on the badge's timer, does nothing until the device can be read,
    and repaints once - after which ``known`` keeps it quiet for the rest of
    the session. The pane follows from ``device_read`` changing.
    """

    if known or imported_torch() is None:
        return (gr.skip(),) * 5
    return (
        *refresh_my_models(selected, order, precision, kind, name, model_id),
        draw_results(listing, precision, fits_only, model_id) if listing is not None else gr.skip(),
        True,
    )
