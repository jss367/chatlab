"""The Models page: downloading, loading, listing and searching for models, and the badge."""

from __future__ import annotations

import html
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import gradio as gr

import settings
from model_discovery import recommended_models
from model_runtime import (
    DEFAULT_MODEL_SORT,
    DISCOVERY_CANDIDATES,
    FITS,
    IMAGE_KIND,
    LOADING,
    MLX_KIND,
    MODEL_WEIGHTS,
    QUANTIZED_BITS,
    SEARCH_IMAGE_PIPELINE_TAGS,
    SEARCH_LIMIT,
    TEXT_KIND,
    TIGHT,
    UNFIT,
    CachedModel,
    CacheStatus,
    DeviceProfile,
    DownloadSnapshot,
    Fit,
    HubModel,
    LoadProgress,
    LoadSnapshot,
    ModelBusy,
    ModelDownloading,
    ModelLoaded,
    cache_root,
    cache_status,
    device_profile,
    estimate_parameter_bytes,
    imported_torch,
    estimate_snapshot_bytes,
    format_bytes,
    format_count,
    list_cached_models,
    mlx_available,
    mlx_bits_from_id,
    mlx_snapshot_bits,
    model_fit,
    search_hub_models,
    snapshot_folder,
    sort_cached_models,
)
from ui import runtime
from ui.model_repository import matching_repository
from ui.common import (
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


def stream_download(model_id: str, hf_token: str):
    """Yield a status card every half second until ``model_id`` is on disk.

    Returns the snapshot path, so a caller writes
    ``path = yield from stream_download(...)``. A failed download raises here.

    The download runs on its own thread: ``snapshot_download`` blocks until the
    last byte, and a handler that blocked with it could show nothing past its
    first frame. If this model is already being fetched (a handler whose
    browser tab went away leaves its thread running), the card follows that
    download rather than starting a second one to fight over the same files.
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
        return (yield from stream_download(model_id, hf_token))

    outcome: dict = {}

    def work() -> None:
        try:
            outcome["path"] = runtime.MANAGER.download(cleaned, hf_token, progress)
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
        "the model weights" if name == MODEL_WEIGHTS else f"`{name}`"
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
    try:
        cached = cache_status(cleaned) if cleaned else CacheStatus()
    except ValueError as error:
        return (
            html.escape(str(error)),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, variant="secondary"),
        )
    except OSError:
        # Keep local loading available if the cache cannot be inspected;
        # its handler can explain the actual error when clicked.
        return (
            "Could not check downloaded files.",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True, variant="secondary"),
        )
    if cached.complete:
        detail = f"**Downloaded** · Ready to load from disk. {where_to_use(cached.kind)}"
        if runtime.MANAGER.model_id == cleaned:
            detail = (
                "**Downloaded · Loaded now** · Load cached again to apply a new "
                f"precision. {where_to_use(cached.kind)}"
            )
    elif cached.unsupported:
        detail = "**Downloaded · Unsupported** · ChatLab cannot load this model's format."
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
    can_load = can_download and not checked.get("unsupported", False)
    return (
        detail,
        gr.update(visible=download, interactive=can_load),
        gr.update(visible=download, interactive=can_download),
        gr.update(visible=cached.complete, variant="primary" if cached.complete else "secondary"),
    )


def download_model(model_id: str, hf_token: str, selected: str | None = None):
    model_id = chosen_model(model_id, selected)
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except (OSError, ValueError) as error:
        yield failure_card("Download failed", html.escape(str(error)))
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path = yield from stream_download(model_id, hf_token)
        elapsed = time.monotonic() - started
        fetched = describe_fetched(before, cache_status(model_id), elapsed)
    except Exception as error:
        yield failure_card("Download failed", html.escape(str(error)))
        return

    yield status_card(
        "Download complete",
        f"{fetched} `{model_id.strip()}` is cached in `{path}`. "
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
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except (OSError, ValueError) as error:
        yield failure_card("Model setup failed", html.escape(str(error)))
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path = yield from stream_download(model_id, hf_token)
        claimed, held = runtime.MANAGER.claim_exclusive_load(model_id)
        if claimed is None:
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
                f"{fetched} Moving `{model_id.strip()}` onto the best available device…",
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
        yield failure_card("Model setup failed", html.escape(str(error)))
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Model ready",
        f"`{model_id.strip()}` is loaded on **{device}** ({elapsed:.1f} seconds "
        f"total). {where_to_use(fetched_status.kind)}",
        "success",
    )


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
    if claim is not None:
        yield from _load_cached_model(cleaned, precision)
        return
    try:
        claimed, held = runtime.MANAGER.claim_exclusive_load(cleaned)
    except ValueError as error:
        yield failure_card("Could not load cached model", html.escape(str(error)))
        return
    if claimed is None:
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
            f"{name} is on disk ({describe_on_disk(status)}) but is {UNSUPPORTED_REASON}",
            "error",
        )
        return
    try:
        path = runtime.MANAGER.find_cached(cleaned)
        started = time.monotonic()
        device = yield from stream_load(cleaned, path, precision, status.kind)
    except IncompleteSnapshotError as error:
        yield failure_card(
            "Download unfinished", incomplete_snapshot_detail(cleaned, error)
        )
        return
    except Exception as error:
        yield failure_card("Could not load cached model", html.escape(str(error)))
        return
    yield status_card(
        "Model ready",
        f"{name} is loaded on **{device}** "
        f"({time.monotonic() - started:.1f} seconds). {where_to_use(status.kind)}",
        "success",
    )


def unload_model():
    if not runtime.MANAGER.in_memory:
        return status_card("No model loaded", "There is nothing to unload.")
    runtime.MANAGER.unload()
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


def model_badge(state: str, text: str) -> str:
    """A pill naming the model in memory. ``state`` is the stylesheet's hook."""

    return (
        f'<div class="model-badge" data-state="{state}">'
        f'<span class="model-badge-dot" aria-hidden="true"></span>'
        f"<span>{html.escape(text)}</span></div>"
    )


def model_snapshot() -> tuple[str | None, str | None, str | None, str]:
    """The load under way, the model in memory, its device and its kind, read once.

    Reuse these values so the badge and setup links render from the same
    readings. This is display state, not an atomic snapshot or a reservation
    of the model for a later action.

    The kind is read from what is really in memory rather than from what the
    last load was asked for, so it can never disagree with the object a page
    would go on to use.
    """

    return (
        runtime.MANAGER.loading_id,
        runtime.MANAGER.model_id,
        runtime.MANAGER.device_name,
        IMAGE_KIND if runtime.MANAGER.image_loaded else TEXT_KIND,
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

    loading, model_id, device, loaded_kind = (
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
        return model_badge("loading", f"Loading {loading}…")
    return model_badge("empty", NO_MODEL_BADGE)


def _setup_links(snapshot, kind: str):
    """Whether a page's own "choose a model" links belong on screen.

    Hidden once that page has a model it can use, or while any load is
    pending. A model of the other kind leaves them showing, because from
    that page there is still a model to go and load.

    Setup links navigate without loading anything, so downloads do not need
    to disable them.
    """

    loading, model_id, device, loaded_kind = snapshot
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

    Yields the switcher's own update and the Models page's status card, so
    the load shows there exactly as **Load cached** would show it, and the
    badge beside the switcher names the load as it goes. A pick during a
    reply is refused and the switcher put back: the load would only queue
    behind the generation, and its first act on winning the lock would be
    to unload the model still producing the tokens.

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
        yield gr.skip(), gr.skip()
        return
    try:
        claimed, held = runtime.MANAGER.claim_exclusive_load(selected)
    except ValueError as error:
        yield gr.update(value=current), failure_card(
            "Could not load cached model", html.escape(str(error))
        )
        return
    if claimed is None:
        alarm(
            "Cannot switch models now",
            occupied_reason(held, SWITCH_LOADING, SWITCH_BUSY),
        )
        yield gr.update(value=current), gr.skip()
        return
    _checked_id, claim = claimed
    last = None
    try:
        for card in load_cached_model(selected, None, precision, claim):
            last = card
            yield gr.skip(), card
    finally:
        runtime.MANAGER.release_load(claim)
    announce_switch_outcome(last)


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


def select_default_model():
    """Select the default and open Models; loading requires a separate click.

    Update the ID, both model selections and removal confirmation together.
    Programmatic ID changes do not fire the typing listener that normally
    clears the selected row, which would otherwise override this ID.
    """

    return (
        settings.DEFAULT_MODEL_ID,
        *clear_my_model_selection(),
        gr.update(value=None),
        NO_RESULT_SELECTED,
        status_card(
            "Default model selected",
            f"`{settings.DEFAULT_MODEL_ID}` is selected. "
            "Choose **Load cached** to use local files, or **Download and load** "
            f"to fetch the model ({DEFAULT_MODEL_DOWNLOAD} for a full download) "
            "and load it. If local files are incomplete, **Download and load** "
            "can fetch the rest.",
        ),
        *hide_remove_confirm(),
        *go_to_models(),
    )


# The side pane's model lists.
NO_CACHED_MODEL_SELECTED = "Select a model to see its details and put it in the model ID box."


# The search is scoped to one kind at a time, because the hub files them
# under different libraries: a Transformers query and a diffusers one are
# different searches, not one search with a wider net.
SEARCH_KINDS = (
    ("Text models", TEXT_KIND),
    ("Image models", IMAGE_KIND),
    # Offered where the backend exists, which is Apple silicon with mlx-lm
    # installed: elsewhere the search would find models that then land in
    # the cache as unsupported.
    *((("MLX models", MLX_KIND),) if mlx_available() else ()),
)

SEARCH_HINTS = {
    TEXT_KIND: (
        "Browse recommended starters, or choose Popular, Trending, or New to explore Hugging Face. "
        "Selecting a result puts its ID in the model ID box; **Download and load** fetches it."
    ),
    IMAGE_KIND: (
        "Browse image starters, or choose Popular, Trending, or New for more text-to-image models. "
        "Selecting a result puts its ID in the model ID box; **Download and load** fetches it."
    ),
    MLX_KIND: (
        "Browse language models quantized for MLX, which run on Apple silicon at the precision "
        "they were converted to. Selecting a result puts its ID in the model ID box; "
        "**Download and load** fetches it."
    ),
}

SEARCH_HINT = SEARCH_HINTS[TEXT_KIND]


NO_RESULT_SELECTED = "Select a result to see its details."


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
    "the same reason: nothing here runs it."
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

    if kind != MLX_KIND or snapshot is None:
        return requested
    return mlx_snapshot_bits(snapshot)


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
    # What the radio asks of this kind, which for a pipeline and an MLX repo
    # is nothing: moving it asks nothing new of either, so neither is marked
    # as about to reload.
    requested = requested_bits(precision, profile, kind)
    reloading = requested != requested_bits(runtime.MANAGER.precision, profile, kind)
    if runtime.MANAGER.model_id == entry.model_id and not reloading:
        return None
    snapshot = snapshot_folder(entry.path) if entry.path is not None else None
    if snapshot is None:
        return None
    # The size depends on the kind too: a pipeline has no checkpoint at its
    # root to measure, and an MLX repo is measured as the packed file it
    # already is. The pool is already the one for this kind - the caller
    # chose it, because choosing it here would re-read the device and
    # discard the memory the impending unload gives back.
    bits = packed_bits(snapshot, kind, requested)
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
    results: list[HubModel], precision: str | None, kind: str = TEXT_KIND
) -> dict[str, Fit]:
    """The fit verdict for each search result, by model ID, against one profile.

    ``kind`` because a search is scoped to one, and an image pipeline on CUDA
    is judged against a different pool; see :meth:`DeviceProfile.for_kind`.
    The estimate itself is from the parameter count either way, which says
    nothing about how the weights are laid out.
    """

    profile = replacement_profile(kind)
    return {
        result.model_id: hub_fit(result, precision, profile, kind)
        for result in results
    }


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
        verdict = f"**Unsupported:** {UNSUPPORTED_REASON}"
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


def my_models_summary(models: list[CachedModel]) -> str:
    root = f"`{cache_root()}`"
    if not models:
        return (
            f"No models in the Hugging Face cache yet ({root}). "
            "Find one under **Discover models**."
        )
    total = format_bytes(sum(entry.size_bytes for entry in models))
    count = f"{len(models)} model{'s' if len(models) != 1 else ''}"
    return f"{count} · {total} on disk in {root}"


def refresh_my_models(
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    precision: str | None = None,
    model_id: str | None = None,
):
    """Rescan the cache; keep the selected row or typed ID, or the loaded model.

    ``precision`` is the **Weight precision** choice, which decides what each
    model would take in memory and so whether it fits. The list is repainted
    when that choice changes, which is what makes the radio the first thing
    to try when a model will not load.
    """

    models = sort_cached_models(list_cached_models(), order)
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
    return gr.update(choices=choices, value=selected), detail, my_models_summary(models)


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


NO_MODEL_TO_MANAGE = "Select a model under **My Models** first."


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


def removal_refusal(selected: str | None) -> tuple[str, str] | None:
    """Why ``selected`` cannot be removed right now, as a card, or None.

    An early answer for the confirmation step only. The deletion itself goes
    through :meth:`ModelManager.remove`, which makes the same checks under
    the manager's locks; this look is not atomic with anything.
    """

    if not selected:
        return "Nothing to remove", NO_MODEL_TO_MANAGE
    if runtime.MANAGER.model_id == selected:
        return loaded_refusal(selected)
    if selected in runtime.MANAGER.active_downloads:
        return downloading_refusal(selected)
    return None


def ask_remove_my_model(selected: str | None):
    """Show the confirmation for removing the chosen model, or say why not.

    Returns the card, the confirmation's visibility, its question, and the
    model the question is about. That last value is what the confirm button
    deletes: the radio can be moved to another model in the moment between
    a click on **Remove from disk** and the response that hides the panel,
    and a deletion that read the live selection would then take the model
    the reader never agreed to lose.
    """

    hidden = gr.update(visible=False)
    refusal = removal_refusal(selected)
    if refusal is not None:
        return status_card(*refusal), hidden, "", None
    entry = next(
        (entry for entry in list_cached_models() if entry.model_id == selected), None
    )
    if entry is None:
        return (
            status_card("Nothing to remove", f"`{selected}` is no longer in the cache."),
            hidden,
            "",
            None,
        )
    question = (
        f"Remove `{selected}` ({format_bytes(entry.size_bytes)}) from disk? "
        "This deletes its folder from the Hugging Face cache and cannot be undone."
    )
    return gr.skip(), gr.update(visible=True), question, selected


def remove_my_model(pending: str | None):
    """Delete the model the confirmation named and report the space freed.

    ``pending`` is the ID :func:`ask_remove_my_model` stored, not the radio's
    current value, so the model deleted is always the one the question
    showed. The pending ID is cleared on every path.
    """

    hidden = gr.update(visible=False)
    if not pending:
        return status_card("Nothing to remove", NO_MODEL_TO_MANAGE), hidden, None
    try:
        freed = runtime.MANAGER.remove(pending)
    except ModelLoaded:
        return status_card(*loaded_refusal(pending)), hidden, None
    except ModelDownloading:
        return status_card(*downloading_refusal(pending)), hidden, None
    except ModelBusy:
        return (
            status_card(
                "Model busy",
                f"`{pending}` cannot be removed while a model is loading, generating, "
                "scoring, or being inspected. Try again when it is idle.",
            ),
            hidden,
            None,
        )
    except FileNotFoundError:
        return (
            status_card("Nothing to remove", f"`{pending}` is no longer in the cache."),
            hidden,
            None,
        )
    except (OSError, ValueError) as error:
        return (
            failure_card(
                "Could not remove model",
                f"Removing `{pending}` failed: {html.escape(str(error))}",
            ),
            hidden,
            None,
        )
    return (
        status_card(
            "Model removed",
            f"Removed `{pending}` from the Hugging Face cache, "
            f"freeing {format_bytes(freed)}.",
            "success",
        ),
        hidden,
        None,
    )


def hide_remove_confirm():
    """Withdraw a pending removal: hide the question and forget its model."""

    return gr.update(visible=False), None


def hub_model_label(result: HubModel, fit: Fit | None = None) -> str:
    parts = [result.model_id]
    if result.summary:
        parts.append(result.summary)
    if result.parameters:
        parts.append(f"{format_count(result.parameters)} params")
    verdict = fit_word(fit)
    if verdict:
        parts.append(verdict)
    if result.downloads is not None:
        parts.append(f"{format_count(result.downloads)} downloads")
    return " · ".join(parts)


def describe_hub_model(result: HubModel, fit: Fit | None = None) -> str:
    name = html.escape(result.model_id)
    lines = [f"[{name} on Hugging Face](https://huggingface.co/{name})"]
    if result.summary:
        lines.extend(["", html.escape(result.summary), ""])
    facts = []
    if result.parameters:
        facts.append(("Parameters", format_count(result.parameters)))
    if fit is not None and fit.known:
        facts.append(("Memory", f"{fit.note} Estimated from the parameter count."))
    else:
        facts.append(("Memory", "Unknown — there is not enough information to estimate a fit."))
    if result.download_bytes:
        precision_note = (
            "The weights are quantized already, so the precision choice does not apply."
            if result.kind == MLX_KIND
            else "Choosing 4-bit or 8-bit reduces loaded memory, not this download."
        )
        facts.append((
            "Full download",
            f"About {format_bytes(result.download_bytes)} for all repository files "
            f"(catalog estimate). {precision_note}",
        ))
    else:
        facts.append(("Full download", "Size unavailable; see the files on Hugging Face."))
    counts = []
    if result.downloads is not None:
        counts.append(f"{format_count(result.downloads)} downloads in the last month")
    if result.likes is not None:
        counts.append(f"{format_count(result.likes)} likes")
    if counts:
        facts.append(("Popularity", " · ".join(counts)))
    if result.license:
        facts.append(("License", html.escape(result.license)))
    if result.last_modified:
        facts.append(("Updated", result.last_modified))
    if result.gated:
        facts.append(
            ("Gated", "accept its terms on Hugging Face and enter a token first")
        )
    else:
        facts.append(("Access", "No access approval indicated"))
    # A cache that cannot be read (a permission, a drive that has gone away)
    # is simply nothing on disk: the search succeeded, so the pick must too.
    try:
        cached = cache_status(result.model_id)
    except (OSError, ValueError):
        cached = CacheStatus()
    if cached.complete:
        facts.append(
            (
                "Already cached",
                f"{describe_on_disk(cached)}, ready to load as "
                f"{'an' if cached.kind == IMAGE_KIND else 'a'} "
                f"{KIND_NAMES.get(cached.kind, 'model')}",
            )
        )
    elif cached.unsupported:
        facts.append(
            ("Already cached", f"{describe_on_disk(cached)}, but not a model ChatLab can load")
        )
    elif cached.present:
        facts.append(("Partly cached", describe_on_disk(cached)))
    lines.extend(f"- **{label}:** {value}" for label, value in facts)
    lines.append("")
    if cached.unsupported:
        lines.append(
            "Its ID is in the model ID box, but downloading again would fetch the "
            "same files: this repo is neither a Transformers language model nor "
            "a diffusers pipeline."
        )
    elif cached.complete:
        lines.append("Already downloaded: use **Load cached** to bring it into memory.")
    else:
        lines.append("Its ID is in the model ID box: use **Download and load** to fetch it.")
    return "\n".join(lines)


def refresh_after_device(
    known: bool,
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    precision: str | None = None,
    model_id: str | None = None,
    result: str | None = None,
    results: dict | None = None,
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
    the session.
    """

    if known or imported_torch() is None:
        return (gr.skip(),) * 6
    return (
        *refresh_my_models(selected, order, precision, model_id),
        *refresh_search_results(result, results or {}, precision, fits_only),
        True,
    )


def search_models(
    query: str | None, hf_token: str, precision: str | None = None, kind: str = TEXT_KIND,
    order: str = "Popular", fits_only: bool = False,
):
    """Browse or search; retain candidates so memory filtering needs no network."""

    cleared = gr.update(choices=[], value=None)
    cleaned = (query or "").strip()
    # A query that matches no starter searches the Hub instead of dead-ending,
    # so the "Search Hugging Face" box does what it says in every view.
    searched_hub = order != "Recommended"
    try:
        results = [] if searched_hub else recommended_models(cleaned, kind)
        if searched_hub or (cleaned and not results):
            searched_hub = True
            results = search_hub_models(
                cleaned, hf_token, kind=kind,
                order="Popular" if order == "Recommended" else order,
                limit=DISCOVERY_CANDIDATES,
            )
    except Exception as error:
        hint = (
            "Clear the search to see offline starters, or retry."
            if order == "Recommended"
            else "Choose Recommended for offline starters, or retry."
        )
        return cleared, failure_card("Search failed", f"{html.escape(str(error))} {hint}"), {}
    if not results:
        described = {IMAGE_KIND: "text-to-image models", MLX_KIND: "MLX models"}.get(
            kind, "language models"
        )
        message = (
            f"No {described} matched `{html.escape(cleaned)}`."
            if cleaned else f"No {described} found in this browse window."
        )
        return cleared, message, {}
    state = {result.model_id: result for result in results}
    radio, detail = refresh_search_results(None, state, precision, fits_only)
    if order == "Recommended" and searched_hub:
        ordering = "No starters matched; showing Hugging Face results, most downloaded first."
    else:
        ordering = {
            "Recommended": "Curated starters, available to browse offline.",
            "Popular": "Most downloaded first.",
            "Trending": "Trending on Hugging Face.",
            "New": "Newest repositories first (not latest updates).",
        }[order]
    return radio, f"{ordering} {detail}", state


def select_search_result(
    selected: str | None, results: dict, precision: str | None = None
):
    """Put the chosen search result in the ID box and describe it."""

    result = results.get(selected) if selected else None
    if result is None:
        return gr.skip(), NO_RESULT_SELECTED
    return (
        gr.update(value=result.model_id),
        describe_hub_model(
            result,
            hub_fit(
                result,
                precision,
                replacement_profile(results_kind({result.model_id: result})),
                results_kind({result.model_id: result}),
            ),
        ),
    )


def results_kind(results: dict) -> str:
    """Which kind a held set of search results is, read from their own tags.

    Derived rather than passed in so a repaint cannot disagree with the
    search that produced the list: the kind that decided the verdicts is the
    one the results themselves carry.
    """

    if any(
        getattr(result, "pipeline_tag", None) in SEARCH_IMAGE_PIPELINE_TAGS
        for result in results.values()
    ):
        return IMAGE_KIND
    if any(getattr(result, "kind", None) == MLX_KIND for result in results.values()):
        return MLX_KIND
    return TEXT_KIND


def refresh_search_results(
    selected: str | None, results: dict, precision: str | None = None,
    fits_only: bool = False,
):
    """Recompute fit filtering from retained candidates, clearing hidden selections."""

    if not results:
        return gr.skip(), gr.skip()
    fits = hub_fits(list(results.values()), precision, results_kind(results))
    visible = [
        result for model_id, result in results.items()
        if not fits_only or (fits.get(model_id) and fits[model_id].state == FITS)
    ][:SEARCH_LIMIT]
    visible_ids = {result.model_id for result in visible}
    selected = selected if selected in visible_ids else None
    choices = [
        (hub_model_label(result, fits.get(result.model_id)), result.model_id)
        for result in visible
    ]
    if selected:
        detail = describe_hub_model(results[selected], fits.get(selected))
    else:
        detail = f"{len(visible)} results shown from {len(results)} candidates. "
        if fits_only:
            detail += (
                "Only estimated fits at the selected precision; tight, too large, "
                "and unknown sizes are hidden. "
            )
        if not visible:
            detail += "No estimated fits in these candidates. Turn off the filter or narrow your search."
        else:
            detail += NO_RESULT_SELECTED
    return gr.update(choices=choices, value=selected), detail
