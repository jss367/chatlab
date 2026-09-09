"""The Models page: downloading, loading, listing and searching for models, and the badge."""

from __future__ import annotations

import html
import re
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

import settings
from model_runtime import (
    DEFAULT_MODEL_SORT,
    MODEL_WEIGHTS,
    CachedModel,
    CacheStatus,
    DownloadSnapshot,
    HubModel,
    LoadProgress,
    LoadSnapshot,
    ModelBusy,
    ModelDownloading,
    ModelLoaded,
    cache_root,
    cache_status,
    format_bytes,
    format_count,
    list_cached_models,
    search_hub_models,
    sort_cached_models,
)
from ui import runtime
from ui.common import (
    DEFAULT_MODEL_DOWNLOAD,
    DOWNLOAD_POLL_SECONDS,
    IncompleteSnapshotError,
    LOAD_POLL_SECONDS,
    MODELS_PAGE,
    RATE_WINDOW_SECONDS,
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


def stream_load(model_id: str, path: Path, precision: str = "full"):
    """Yield a status card every half second until ``model_id`` is in memory.

    Returns the device it landed on, so a caller writes
    ``device = yield from stream_load(...)``. A failed load raises here.
    ``precision`` is the weight precision chosen on the Models page.

    The load runs on its own thread, as a download does: ``from_pretrained``
    blocks until the last weight, and a handler that blocked with it could
    show nothing past its first frame.
    """

    progress = LoadProgress()
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["device"] = runtime.MANAGER.load(model_id, path, progress, precision=precision)
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


def refresh_model_actions(model_id: str, selected: str | None):
    """Show the actions appropriate to the chosen model's local files."""

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
        detail = "**Downloaded** · Ready to load from disk."
        if runtime.MANAGER.model_id == cleaned:
            detail = "**Downloaded · Loaded now** · Load cached again to apply a new precision."
    elif cached.unsupported:
        detail = "**Downloaded · Unsupported** · ChatLab cannot load this model's format."
    elif cached.present:
        detail = "**Download incomplete** · Download and load will fetch the remaining files."
    else:
        detail = "**Not downloaded** · Download the model to use it." if cleaned else "Enter a model ID or select a model."
    download = not (cached.complete or cached.unsupported)
    return (
        detail,
        gr.update(visible=download),
        gr.update(visible=download),
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


def download_and_load_model(
    model_id: str, hf_token: str, selected: str | None = None, precision: str = "full"
):
    """Download and load the model explicitly selected on the Models page."""

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
        fetched = describe_fetched(
            before, cache_status(model_id), time.monotonic() - started
        )
        yield status_card(
            "Loading model",
            f"{fetched} Moving `{model_id.strip()}` onto the best available device…",
            "working",
        )
        device = yield from stream_load(model_id, path, precision)
    except Exception as error:
        yield failure_card("Model setup failed", html.escape(str(error)))
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Model ready",
        f"`{model_id.strip()}` is loaded on **{device}** ({elapsed:.1f} seconds total).",
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
    model_id: str, selected: str | None = None, precision: str = "full"
):
    """Load the selected model from local files, preserving any load error."""

    cleaned = chosen_model(model_id, selected)
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
        device = yield from stream_load(cleaned, path, precision)
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
        f"({time.monotonic() - started:.1f} seconds).",
        "success",
    )


def unload_model():
    if not runtime.MANAGER.loaded:
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


def model_snapshot() -> tuple[str | None, str | None, str | None]:
    """The load under way, the model in memory, and its device, read once.

    Reuse these values so the badge and setup links render from the same
    readings. This is display state, not an atomic snapshot or a reservation
    of the model for a later action.
    """

    return runtime.MANAGER.loading_id, runtime.MANAGER.model_id, runtime.MANAGER.device_name


def loaded_model_badge(snapshot=None) -> str:
    """Name the model in memory, the one being loaded, or neither.

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

    loading, model_id, device = (
        model_snapshot() if snapshot is None else snapshot
    )
    if model_id and device:
        return model_badge("ready", f"{model_id} · loaded on {device}")
    if loading:
        return model_badge("loading", f"Loading {loading}…")
    return model_badge("empty", NO_MODEL_BADGE)


def refresh_model_badge():
    """Refresh the badge and the setup links from the same reading.

    Setup links navigate without loading anything, so downloads do not need
    to disable them. Hide them when a model is loaded or a load is pending.
    """

    snapshot = model_snapshot()
    loading, model_id, device = snapshot
    links = gr.update(visible=not ((model_id and device) or loading))
    return loaded_model_badge(snapshot), links, links


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


SEARCH_HINT = (
    "Search Hugging Face for language models Transformers can load. "
    "Selecting a result puts its ID in the model ID box; **Download and load** fetches it."
)


NO_RESULT_SELECTED = "Select a result to see its details."


def format_timestamp(stamp: float | None) -> str:
    if stamp is None:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


UNSUPPORTED_REASON = (
    "not a Transformers language model: its files are all here, but its weights "
    "are laid out for another framework (a diffusers pipeline, a CTranslate2 or "
    "ONNX export, say) and there is no `model.safetensors` or `pytorch_model.bin` "
    "at the top of the repo, so ChatLab cannot load it."
)


def cached_model_label(entry: CachedModel) -> str:
    """``org/name · 15 GB``, flagged when it is loaded or short of files."""

    label = f"{entry.model_id} · {format_bytes(entry.size_bytes)}"
    if entry.status.missing_files:
        label += " · incomplete"
    elif entry.status.unsupported:
        label += " · unsupported"
    if runtime.MANAGER.model_id == entry.model_id:
        label += " · loaded"
    return label


def describe_cached_model(entry: CachedModel) -> str:
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
        verdict = "**Downloaded · Ready to load.** Use **Load cached** to bring it into memory."
    facts = [("On disk", describe_on_disk(entry.status))]
    if entry.files:
        facts.append(("Files", f"{entry.files} in the current snapshot"))
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
            "Search for one under **Model search**."
        )
    total = format_bytes(sum(entry.size_bytes for entry in models))
    count = f"{len(models)} model{'s' if len(models) != 1 else ''}"
    return f"{count} · {total} on disk in {root}"


def refresh_my_models(
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    model_id: str | None = None,
):
    """Keep the selected row or typed ID; default to the loaded model at startup."""

    models = sort_cached_models(list_cached_models(), order)
    ids = [entry.model_id for entry in models]
    if selected not in ids:
        fallback = model_id.strip() if model_id is not None else runtime.MANAGER.model_id
        selected = fallback if fallback in ids else None
    choices = [(cached_model_label(entry), entry.model_id) for entry in models]
    if selected is None:
        detail = NO_CACHED_MODEL_SELECTED if models else ""
    else:
        detail = describe_cached_model(
            next(entry for entry in models if entry.model_id == selected)
        )
    return gr.update(choices=choices, value=selected), detail, my_models_summary(models)


def select_my_model(selected: str | None):
    """Put the chosen cached model in the ID box and describe it."""

    if not selected:
        return gr.skip(), NO_CACHED_MODEL_SELECTED
    entry = next(
        (entry for entry in list_cached_models() if entry.model_id == selected), None
    )
    if entry is None:
        return gr.skip(), f"`{selected}` is no longer in the cache. Press **Refresh**."
    return gr.update(value=selected), describe_cached_model(entry)


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


def hub_model_label(result: HubModel) -> str:
    parts = [result.model_id]
    if result.parameters:
        parts.append(f"{format_count(result.parameters)} params")
    if result.downloads is not None:
        parts.append(f"{format_count(result.downloads)} downloads")
    return " · ".join(parts)


def describe_hub_model(result: HubModel) -> str:
    name = html.escape(result.model_id)
    lines = [f"[{name} on Hugging Face](https://huggingface.co/{name})"]
    facts = []
    if result.parameters:
        facts.append(("Parameters", format_count(result.parameters)))
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
    # A cache that cannot be read (a permission, a drive that has gone away)
    # is simply nothing on disk: the search succeeded, so the pick must too.
    try:
        cached = cache_status(result.model_id)
    except (OSError, ValueError):
        cached = CacheStatus()
    if cached.complete:
        facts.append(("Already cached", f"{describe_on_disk(cached)}, ready to load"))
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
            "same files: this repo is not a Transformers language model."
        )
    elif cached.complete:
        lines.append("Already downloaded: use **Load cached** to bring it into memory.")
    else:
        lines.append("Its ID is in the model ID box: use **Download and load** to fetch it.")
    return "\n".join(lines)


def search_models(query: str, hf_token: str):
    """Search the hub and list the results; nothing is selected yet."""

    cleared = gr.update(choices=[], value=None)
    cleaned = query.strip()
    if not cleaned:
        return cleared, SEARCH_HINT, {}
    try:
        results = search_hub_models(cleaned, hf_token)
    except Exception as error:
        return (
            cleared,
            failure_card("Search failed", html.escape(str(error))),
            {},
        )
    if not results:
        return (
            cleared,
            f"No language models matched `{html.escape(cleaned)}`.",
            {},
        )
    choices = [(hub_model_label(result), result.model_id) for result in results]
    count = f"{len(results)} result{'s' if len(results) != 1 else ''}"
    return (
        gr.update(choices=choices, value=None),
        f"{count}, most downloaded first. {NO_RESULT_SELECTED}",
        {result.model_id: result for result in results},
    )


def select_search_result(selected: str | None, results: dict):
    """Put the chosen search result in the ID box and describe it."""

    result = results.get(selected) if selected else None
    if result is None:
        return gr.skip(), NO_RESULT_SELECTED
    return gr.update(value=result.model_id), describe_hub_model(result)
