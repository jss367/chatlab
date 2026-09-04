"""Chatlab interface for chatting with and inspecting model tokens."""

from __future__ import annotations

import contextlib
import html
import logging
import os
import random
import re
import threading
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio.utils import get_upload_folder

import charts
from conversation import (
    CHAT_PREFIX,
    MAIN_BRANCH,
    THINK_CLOSE,
    branch_choices,
    copy_forks,
    copy_turns,
    display_messages,
    forget_measurements,
    fork_at,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    new_forks,
    next_branch_name,
    next_fork_name,
    split_reasoning,
    to_json,
    user_index_at_or_before,
)
from model_runtime import (
    DEFAULT_MODEL_SORT,
    MODEL_SORT_ORDERS,
    MODEL_WEIGHTS,
    PROMPT_SCORE_LIMIT,
    CachedModel,
    CacheStatus,
    DownloadSnapshot,
    HubModel,
    LoadProgress,
    LoadSnapshot,
    ModelBusy,
    ModelChanged,
    ModelDownloading,
    ModelLoaded,
    ModelManager,
    cache_root,
    cache_status,
    format_bytes,
    format_count,
    list_cached_models,
    search_hub_models,
    sort_cached_models,
)
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
    UNSCORED_BEYOND_LIMIT,
    category_for,
    summarize,
)
from trace_export import build_trace, write_trace_export

logger = logging.getLogger(__name__)

try:
    from huggingface_hub.errors import IncompleteSnapshotError
except ImportError:  # huggingface_hub before 1.x had no such check

    class IncompleteSnapshotError(Exception):
        pass


DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"
MANAGER = ModelManager()

# How often the download card is redrawn. Every frame is a message to the
# browser, so this is a floor on chatter as much as a refresh rate.
DOWNLOAD_POLL_SECONDS = 0.5

# The load card is redrawn on the same beat as the download card, and for
# the same reason: every frame is a message to the browser.
LOAD_POLL_SECONDS = 0.5

# Transfer speed is averaged over this long, so a stall or a burst shows within
# a breath but one slow chunk does not swing the time remaining.
RATE_WINDOW_SECONDS = 15.0

DOWNLOAD_BAR_WIDTH = 24
# The two left panes' widths in pixels. The thin pane at the far left picks
# the page and holds nothing but an icon per page; the conversations pane
# beside it shows with Chat only, and is wide enough for a model ID and a
# token count while leaving the conversation most of the screen.
NAV_PANE_WIDTH = 56
CONVERSATION_PANE_WIDTH = 340
CHAT_PAGE, MODELS_PAGE, SETTINGS_PAGE = PAGES = ("Chat", "Models", "Settings")
# Each nav tile is an icon, which is what lets the pane be this thin. The
# page's name stays on the tile for the browser to read out; the stylesheet
# hides it, draws the icon in its place, and pops the name up on hover.
NAV_ICONS = {
    CHAT_PAGE: "💬",
    MODELS_PAGE: "🧠",
    # The gear has a text form and an emoji form; the variation selector
    # asks for the emoji, so it matches the other two tiles.
    SETTINGS_PAGE: "⚙️",
}
SEED_LIMIT = 2**31 - 1
NO_TOKEN_SELECTED = "Select a token to inspect it."

# Redrawing the trace on every streamed token is wasted work, so it catches up
# in batches and again once the response finishes.
CHART_EVERY = 16

RESPONSE_STRIP_LABEL = "Response tokens — click one"

# This tokenizer offers neither offsets nor a decode that round trips, so
# where the context ends had to be counted out rather than confirmed. The
# scored tokens are still the whole passage's own single encoding, so every
# probability is exact; what is uncertain is where the line between the two
# halves was drawn, and a line a token out moves that token between the two
# tables and the summary figures they feed.
SEAM_CAVEAT = (
    "Approximate split: this tokenizer could not confirm where the context "
    "ends, so the boundary between it and the scored text may sit a token "
    "off. Every probability shown is the full passage's own either way."
)

# The chat-message box was ticked for a model that ships no chat template, so
# there was no turn to wrap the context in. The numbers are exact — they are
# the plain passage's own — but they are not the framing the box promised, and
# the difference is the reader's to know about.
TEMPLATE_CAVEAT = (
    "Plain text, not a chat turn: this model has no chat template, so the "
    "context was measured as ordinary characters in front of the text."
)


def status_card(title: str, detail: str, tone: str = "neutral") -> str:
    icon = {"success": "●", "error": "●", "working": "◌"}.get(tone, "○")
    return f"### {icon} {title}\n\n{detail}"


def describe_duration(seconds: float) -> str:
    """A rounded spoken length: ``a few seconds``, ``about 4 minutes``.

    Rounded to five seconds under a minute, because a load is often over in
    that time and "under a minute" would be the whole of what it ever said.
    """

    if seconds < 10:
        return "a few seconds"
    if seconds < 55:
        return f"about {round(seconds / 5) * 5} seconds"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    text = f"about {hours} hour{'s' if hours != 1 else ''}"
    if minutes:
        text += f" {minutes} minute{'s' if minutes != 1 else ''}"
    return text


def progress_bar(fraction: float, width: int = DOWNLOAD_BAR_WIDTH) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


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
        figures += (
            f" · {format_bytes(rate)}/s · {describe_duration(remaining / rate)} left"
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
    progress, reserved = MANAGER.reserve_download(cleaned)
    if not reserved:
        meter = RateMeter()
        while MANAGER.active_downloads.get(cleaned) is progress:
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
            outcome["path"] = MANAGER.download(cleaned, hf_token, progress)
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=work, name="chatlab-download", daemon=True)
    try:
        worker.start()
    except BaseException:
        # download() never ran, so its finally cannot release the reservation.
        MANAGER.release_download(cleaned, progress)
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


def stream_load(model_id: str, path: Path):
    """Yield a status card every half second until ``model_id`` is in memory.

    Returns the device it landed on, so a caller writes
    ``device = yield from stream_load(...)``. A failed load raises here.

    The load runs on its own thread, as a download does: ``from_pretrained``
    blocks until the last weight, and a handler that blocked with it could
    show nothing past its first frame.
    """

    progress = LoadProgress()
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["device"] = MANAGER.load(model_id, path, progress)
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=work, name="chatlab-load", daemon=True)
    worker.start()
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


def download_model(model_id: str, hf_token: str):
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except ValueError as error:
        yield status_card("Download failed", html.escape(str(error)), "error")
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path = yield from stream_download(model_id, hf_token)
    except Exception as error:
        yield status_card("Download failed", html.escape(str(error)), "error")
        return

    elapsed = time.monotonic() - started
    fetched = describe_fetched(before, cache_status(model_id), elapsed)
    yield status_card(
        "Download complete",
        f"{fetched} `{model_id.strip()}` is cached in `{path}`. "
        "Use **Load cached** when ready.",
        "success",
    )


def download_and_load_model(model_id: str, hf_token: str):
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except ValueError as error:
        yield status_card("Model setup failed", html.escape(str(error)), "error")
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
        device = yield from stream_load(model_id, path)
    except Exception as error:
        yield status_card("Model setup failed", html.escape(str(error)), "error")
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


def load_cached_model(model_id: str):
    cleaned = model_id.strip()
    active = MANAGER.active_downloads.get(cleaned)
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
    except ValueError as error:
        yield status_card("Could not load cached model", html.escape(str(error)), "error")
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
        path = MANAGER.find_cached(cleaned)
        started = time.monotonic()
        device = yield from stream_load(cleaned, path)
    except IncompleteSnapshotError as error:
        yield status_card(
            "Download unfinished", incomplete_snapshot_detail(cleaned, error), "error"
        )
        return
    except Exception as error:
        yield status_card(
            "Could not load cached model", html.escape(str(error)), "error"
        )
        return
    yield status_card(
        "Model ready",
        f"{name} is loaded on **{device}** "
        f"({time.monotonic() - started:.1f} seconds).",
        "success",
    )


def unload_model():
    if not MANAGER.loaded:
        return status_card("No model loaded", "There is nothing to unload.")
    MANAGER.unload()
    return status_card("Model unloaded", "Model memory has been released.", "success")


# The side pane's model lists.
NO_CACHED_MODEL_SELECTED = "Select a model to see its details and put it in the model ID box."
SEARCH_HINT = (
    "Search Hugging Face for text-generation models Transformers can load. "
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
    if MANAGER.model_id == entry.model_id:
        label += " · loaded"
    return label


def describe_cached_model(entry: CachedModel) -> str:
    if MANAGER.model_id == entry.model_id:
        verdict = f"**Loaded now** on {MANAGER.device_name}."
    elif entry.status.missing_files:
        verdict = (
            f"**Incomplete:** {describe_missing(entry.status)}. "
            "Use **Download and load** to fetch the rest."
        )
    elif entry.status.unsupported:
        verdict = f"**Unsupported:** {UNSUPPORTED_REASON}"
    else:
        verdict = "**Ready to load.** Use **Load cached** to bring it into memory."
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


def refresh_my_models(selected: str | None, order: str | None = DEFAULT_MODEL_SORT):
    """Rescan the cache; keep the selection, or fall back to the loaded model."""

    models = sort_cached_models(list_cached_models(), order)
    ids = [entry.model_id for entry in models]
    if selected not in ids:
        selected = MANAGER.model_id if MANAGER.model_id in ids else None
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
    """

    if not selected:
        yield status_card("Nothing to redownload", NO_MODEL_TO_MANAGE)
        return
    if MANAGER.loading_id == selected:
        yield status_card(
            "Model in use",
            f"`{selected}` is being loaded right now. Wait for the load to finish, "
            "then **Unload** it before redownloading.",
        )
        return
    if MANAGER.model_id == selected:
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
    if MANAGER.model_id == selected:
        return loaded_refusal(selected)
    if selected in MANAGER.active_downloads:
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
        freed = MANAGER.remove(pending)
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
            status_card(
                "Could not remove model",
                f"Removing `{pending}` failed: {html.escape(str(error))}",
                "error",
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
            status_card("Search failed", html.escape(str(error)), "error"),
            {},
        )
    if not results:
        return (
            cleared,
            f"No text-generation models matched `{html.escape(cleaned)}`.",
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


def resolve_scale(scale_name: str):
    return COLOR_SCALES.get(scale_name) or COLOR_SCALES[DEFAULT_COLOR_SCALE]


def strip_value(metrics: list[dict], scale_name: str) -> list[tuple[str, str]]:
    """Bucket every token for the strip under one color scale."""

    scale = resolve_scale(scale_name)
    return [
        (metric["display_text"], category_for(metric, scale.name))
        for metric in metrics or []
    ]


def strip_update(metrics: list[dict], scale_name: str, label: str | None = None):
    """Repaint a token strip, legend and all.

    Streaming updates send the value alone, because rebuilding the component
    for every token to carry an unchanged legend is wasted work.
    """

    scale = resolve_scale(scale_name)
    update = {"value": strip_value(metrics, scale.name), "color_map": scale.color_map}
    if label is not None:
        update["label"] = label
    return gr.update(**update)


# The token strip's select listener runs independently of the generation
# stream. Clicking a token queues its own event, and Gradio resolves that
# event's inputs when it gets round to processing it, so a click made a moment
# before Send can still be holding the previous response's metrics when it
# finally runs - after the generation's opening frame has emptied the strip and
# reset the detail panel. Publishing that click would put the old token's
# probabilities beside the new response, and every later streaming frame
# returns gr.skip() for those two outputs, so the stale numbers would sit there
# until the user clicked again.
#
# The fix is a generation number that each click carries with it, issued and
# compared here on the server. It cannot live in gr.State on its own: a
# listener's state inputs are snapshotted together, so a number travelling that
# way would go stale in lockstep with the metrics it is meant to date, and
# every comparison would agree with itself. So the number is minted here, and
# only rides along in the state beside the metrics it stamps. Every path that
# replaces the strip mints a new one, which is what makes the older selections
# detectable.
#
# The counter is process-wide rather than per session, so on a shared server
# one user's generation also drops another's in-flight click. That costs the
# second user one repeated click and never shows either of them a wrong number,
# and the only per-session store Gradio offers is the one that cannot carry
# this.
_metrics_lock = threading.Lock()
_metrics_generation = 0


def new_metrics_generation() -> int:
    """Stamp a new token strip, invalidating selections made against the old one."""

    global _metrics_generation
    with _metrics_lock:
        _metrics_generation += 1
        return _metrics_generation


def stamped(metrics: list[dict], generation: int | None = None):
    """Pair metrics with the stamp a click has to match to be published."""

    return (new_metrics_generation() if generation is None else generation), metrics


def empty_metrics() -> tuple[int, list[dict]]:
    """The metrics payload for a path that clears the strip."""

    return stamped([])


def cleared_strips(scale_name: str):
    """Empty both token strips under one stamp.

    The response strip and the prompt strip are replaced together, so they
    share a stamp: minting one each would leave the first of them looking
    stale to inspect_token() the instant the second was minted.
    """

    generation = new_metrics_generation()
    return (
        strip_update([], scale_name, RESPONSE_STRIP_LABEL),
        stamped([], generation),
        strip_update([], scale_name),
        stamped([], generation),
        "",
    )


def inspect_token(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
    generation, metrics = metrics_state
    if generation != _metrics_generation:
        # The strip this click was made against is gone. Whatever replaced it
        # already reset the detail panel, so leave that reset alone instead of
        # repainting it with a token the user can no longer see.
        return gr.skip(), gr.skip()

    if not metrics:
        return NO_TOKEN_SELECTED, []

    try:
        metric = metrics[event_index(event)]
    except (IndexError, TypeError, ValueError):
        return "That token is no longer available. Generate another response.", []
    return describe_token(metric)


def event_index(event: gr.SelectData) -> int:
    """The row a select event landed on, whichever shape the component sends."""

    index = event.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    return int(index)


def describe_token(metric: dict) -> tuple[str, list[list]]:
    """The detail panel and the alternatives table for one token."""

    token_repr = html.escape(repr(metric["text"]))
    where = "Prompt token" if metric["segment"] == "prompt" else "Token"
    if not metric.get("scored", True):
        if metric.get("unscored_reason") == UNSCORED_BEYOND_LIMIT:
            why = (
                f"Only the most recent {PROMPT_SCORE_LIMIT:,} tokens of a long "
                "prompt are scored, and this one sits before that window, so it "
                "was skipped."
            )
        else:
            why = "Nothing came before this token, so the model never predicted it."
        return (
            f"### {where} {metric['position']}: `{token_repr}`\n\n"
            f"{why}\n\n"
            f"- **Token ID:** {metric['token_id']:,}",
            [],
        )

    summary = (
        f"### {where} {metric['position']}: `{token_repr}`\n\n"
        f"- **Raw rank:** {metric['raw_rank']:,}\n"
        f"- **Raw model probability:** {metric['raw_probability']:.5%}\n"
        f"- **Actual sampling probability:** {metric['sampling_probability']:.5%}\n"
        f"- **Surprise:** {metric['surprise_bits']:.2f} bits\n"
        f"- **Distribution entropy:** {metric['entropy_bits']:.2f} bits\n"
        f"- **Top-1 margin:** {metric['top1_margin']:.2%} between the model's first and second choice\n"
        f"- **Sampling shift:** {metric['sampling_shift_bits']:+.2f} bits versus the raw model\n"
        f"- **Probability mass above it:** {metric['probability_mass_above']:.2%}\n"
        f"- **Token ID:** {metric['token_id']:,}"
    )
    rows = [
        [candidate["token_id"], repr(candidate["text"]), candidate["probability"]]
        for candidate in metric["top_candidates"]
    ]
    return summary, rows


# ---------------------------------------------------------- branch from token
#
# Branching replays a response up to one token, puts an alternative in that
# token's place, and lets the model continue. It takes three clicks - a token
# in the strip, a row in the alternatives table, the Branch button - and each
# click leaves its choice in a gr.State stamped with the strip's generation
# number. Every path that replaces the strip mints a new number, so a choice
# made against a strip that is gone is refused rather than replayed onto the
# wrong response.
#
# ``branch_source`` is the stamp and model load of the last strip that came
# from a chat response. Scored text draws the same strip and alternatives but
# has no conversation to branch, while a reload leaves old token IDs on screen
# that the new tokenizer must not read. Both cases are refused unless the
# generation and load agree.

BRANCH_HINT = (
    "Click a response token, then one of its alternatives, then branch."
)
BRANCH_TEXT_HINT = (
    "Click a response token, type the text to put in its place, then branch."
)
BRANCH_TEXT_EMPTY = "Type the text that should replace the selected token first."
BRANCH_REASONING_CLOSE = (
    "🌱 That token is part of the automatic reasoning boundary. Branch from "
    "the first answer token after it instead."
)
BRANCH_UNAVAILABLE = (
    "🌱 Only a chat response can be branched. Scored text and prompt tokens "
    "have no conversation to continue."
)
BRANCH_MODEL_CHANGED = (
    "🌱 The model was reloaded before the branch could be replayed, so the "
    "response's tokens no longer belong to the weights in memory. The "
    "conversation was left as it was."
)


def remember_selection(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
    """Keep the strip position a click landed on, for the alternatives table.

    Only a scored response token is worth keeping. A prompt token, or one that
    was never predicted, has no alternatives to branch into, and remembering
    it would let a click in the table pair its row with the wrong token.
    """

    generation, metrics = metrics_state
    if generation != _metrics_generation:
        return None
    try:
        metric = metrics[event_index(event)]
    except (IndexError, TypeError, ValueError):
        return None
    if metric.get("segment") != "response" or not metric.get("scored", True):
        return None
    return {"generation": generation, "index": event_index(event)}


def branch_ready_text(pick: dict) -> str:
    position = pick["position"]
    chosen = html.escape(repr(pick["text"]))
    original = html.escape(repr(pick["original"]))
    if pick["token_id"] == pick["original_id"]:
        return (
            f"🌱 **Branch ready:** keep the response through token {position} "
            f"(`{chosen}`) and let the model continue from there with a fresh "
            "sample. Press **Branch from token**."
        )
    return (
        f"🌱 **Branch ready:** keep the first {position - 1} token"
        f"{'' if position == 2 else 's'}, put `{chosen}` where `{original}` was, "
        "and let the model continue. Press **Branch from token**."
    )


def choose_alternative(
    metrics_state: tuple[int, list[dict]],
    selected_token: dict | None,
    branch_source: tuple[int, str | None] | None,
    event: gr.SelectData,
):
    """Pair a row of the alternatives table with the token it belongs to."""

    generation, metrics = metrics_state
    if (
        generation != _metrics_generation
        or not selected_token
        or selected_token.get("generation") != generation
    ):
        return gr.skip(), None
    try:
        metric = metrics[int(selected_token["index"])]
        candidate = metric["top_candidates"][event_index(event)]
    except (IndexError, KeyError, TypeError, ValueError):
        return gr.skip(), None

    summary, _rows = describe_token(metric)
    if branch_source != (generation, MANAGER.load_id):
        return f"{summary}\n\n{BRANCH_UNAVAILABLE}", None
    pick = {
        "generation": generation,
        "position": int(metric["position"]),
        "token_id": int(candidate["token_id"]),
        "text": candidate["text"],
        "original_id": int(metric["token_id"]),
        "original": metric["text"],
    }
    return f"{summary}\n\n{branch_ready_text(pick)}", pick


def recolor(response_state, prompt_state, scale_name: str):
    """Repaint both strips when the reader picks a different color scale."""

    _generation, metrics = response_state
    _prompt_generation, prompt_metrics = prompt_state
    scale = resolve_scale(scale_name)
    return (
        strip_update(metrics, scale.name),
        strip_update(prompt_metrics, scale.name),
        scale.caption,
    )


def prompt_note_text(count: int, note: str, kind: str) -> str:
    if not count:
        return ""
    text = f"{count:,} {kind} tokens. The first one has no prediction behind it."
    if note:
        text = f"{text} {note}"
    return text


# ---------------------------------------------------------------- generation


# Every generation handler publishes this tuple, in this order. Naming the rows
# here keeps the refusal paths - which skip most of them - from counting
# placeholders by hand.
CHAT_OUTPUT_NAMES = (
    "prompt",
    "chatbot",
    "turns",
    "strip",
    "metrics",
    "status",
    "seed",
    "send",
    "stop",
    "detail",
    "alternatives",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "branch_source",
    "context_ids",
)


def split_response_text(
    text: str,
    *,
    literal_prefill: str = "",
    literal_spans: tuple[tuple[int, int], ...] = (),
    streaming: bool = False,
    reasoning_prefilled: bool = False,
) -> tuple[str, str, bool]:
    """Split reasoning without treating reader-supplied text as syntax.

    The first runtime update for an assistant prefill contains only its forced
    tokens. Remembering that decoded prefix lets the application protect every
    ``<`` the reader supplied while leaving the automatic leading ``</think>``
    visible to the reasoning parser. ``literal_spans`` does the same for typed
    branch replacements, which can occur after sampled tokens. Tags sampled
    later by the model keep their normal meaning.
    """

    protected_spans = [
        (max(0, int(start_at)), min(len(text), int(end_at)))
        for start_at, end_at in literal_spans
        if int(start_at) < len(text) and int(end_at) > 0
    ]
    if literal_prefill and text.startswith(literal_prefill):
        literal_start = 0
        if reasoning_prefilled:
            marker_at = literal_prefill.find(THINK_CLOSE)
            if marker_at >= 0:
                literal_start = marker_at + len(THINK_CLOSE)
                # _response_prefix_ids() inserts this separator between the
                # template's closing reasoning marker and the reader's text.
                # Leave it outside protection so the parser trims it while
                # retaining whitespace the reader actually typed after it.
                if literal_prefill.startswith("\n\n", literal_start):
                    literal_start += 2
            else:
                literal_start = len(literal_prefill)
        if literal_start < len(literal_prefill):
            protected_spans.append((literal_start, len(literal_prefill)))

    protected_spans = sorted(
        (start_at, end_at)
        for start_at, end_at in protected_spans
        if start_at < end_at
    )
    merged_spans: list[tuple[int, int]] = []
    for start_at, end_at in protected_spans:
        if merged_spans and start_at <= merged_spans[-1][1]:
            old_start, old_end = merged_spans[-1]
            merged_spans[-1] = (old_start, max(old_end, end_at))
        else:
            merged_spans.append((start_at, end_at))

    if not merged_spans:
        return split_reasoning(
            text,
            streaming=streaming,
            reasoning_prefilled=reasoning_prefilled,
        )

    placeholder = "\0CHATLAB_LITERAL_LT\0"
    start = "\0CHATLAB_LITERAL_START\0"
    end = "\0CHATLAB_LITERAL_END\0"
    while placeholder in text or start in text or end in text:
        placeholder += "_"
        start += "_"
        end += "_"
    protected_parts: list[str] = []
    cursor = 0
    for start_at, end_at in merged_spans:
        protected_parts.append(text[cursor:start_at])
        protected_parts.append(start)
        protected_parts.append(text[start_at:end_at].replace("<", placeholder))
        protected_parts.append(end)
        cursor = end_at
    protected_parts.append(text[cursor:])
    reasoning, answer, closed = split_reasoning(
        "".join(protected_parts),
        streaming=streaming,
        reasoning_prefilled=reasoning_prefilled,
    )

    def restore(value: str) -> str:
        return (
            value.replace(placeholder, "<")
            .replace(start, "")
            .replace(end, "")
        )

    return (
        restore(reasoning),
        restore(answer),
        closed,
    )


def send_stop_buttons(busy: bool):
    """Swap the Send and Stop buttons for each other."""

    return gr.update(visible=not busy), gr.update(visible=busy)


def finalize_partial(turns: list[dict]) -> bool:
    """Close out a half-written assistant turn, dropping it when it holds nothing.

    Returns whether a partial response was worth keeping. Cancelling or failing
    mid-stream can leave a turn whose reasoning block is still marked pending,
    which would keep the accordion spinning for the rest of the session.
    """

    if not turns or turns[-1]["role"] != "assistant":
        return False
    if not (turns[-1].get("content") or turns[-1].get("reasoning")):
        turns.pop()
        return False
    turns[-1]["reasoning_closed"] = True
    return True


def stop_generation(
    turns: list[dict] | None,
    metrics_state: tuple[int, list[dict]] = (0, []),
    context_state: tuple[int, list[int], str | None] = (0, [], None),
):
    """Finish the turn that the cancelled generator left behind.

    Gradio closes ``generate_reply`` at its last yield, so nothing else ever
    finalizes that turn. A kept partial response is still a response: the
    tokens on screen are the ones it is made of, so it can be branched from.
    ``context_state`` carries the producing load captured while the model lock
    was held; a waiting load may finish after cancellation but before this
    handler runs, so consulting the manager here would mislabel the old IDs.
    """

    turns = copy_turns(turns)
    kept = finalize_partial(turns)
    messages, _ = display_messages(turns)
    generation, metrics = metrics_state
    context_generation, _context_ids, producing_load_id = context_state
    return (
        messages,
        turns,
        *send_stop_buttons(False),
        "Stopped. The partial response was kept."
        if kept
        else "Stopped before the model produced anything.",
        (generation, producing_load_id)
        if kept and metrics and context_generation == generation
        else None,
    )


def resolve_seed(seed, randomize: bool) -> int:
    """Pick the seed for one generation, inside the range NumPy will accept.

    ``np.random.default_rng()`` rejects negative integers, so a locked seed of
    ``-1`` used to fail every generation with "expected non-negative integer"
    and produce no reply at all. The number input is constrained to 0 and above,
    but the clamp lives here as well: this is the only place the value is turned
    into the one the generator is handed, and it can still arrive out of range
    from the API, from a browser that ignores the constraint, or from a float
    the input rounded. Non-numeric and missing values keep falling back to 0.
    """

    if randomize:
        return random.randrange(SEED_LIMIT)
    try:
        # OverflowError covers infinities, which int() refuses to convert.
        value = int(seed)
    except (OverflowError, TypeError, ValueError):
        return 0
    return max(value, 0)


def generation_progress(count: int, started: float, seed: int) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    plural = "" if count == 1 else "s"
    return (
        f"{count} token{plural} · {elapsed:.1f}s · {count / elapsed:.1f} tok/s "
        f"· seed {seed}"
    )


def idle_state(
    prompt_text: str,
    turns: list[dict],
    status: str,
    *,
    clear_tokens: bool = False,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """A non-streaming result that leaves the seed untouched.

    The token panel is normally left alone as well: paths such as "Enter a
    message first." must not wipe the diagnostics of the response already on
    screen. ``clear_tokens`` is for the one case where those diagnostics stop
    describing the visible text - an edited assistant reply. Both strips go
    together there: they carry one shared stamp, so re-stamping the response
    strip alone would silently stop the prompt strip's clicks from publishing.
    """

    messages, _ = display_messages(turns)
    panels = (
        cleared_strips(scale_name)
        if clear_tokens
        else (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
    )
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = panels
    return (
        prompt_text,
        messages,
        copy_turns(turns),
        strip,
        metrics,
        status,
        gr.skip(),
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED if clear_tokens else gr.skip(),
        [] if clear_tokens else gr.skip(),
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}) if clear_tokens else gr.skip(),
        charts.EMPTY_CHART if clear_tokens else gr.skip(),
        {} if clear_tokens else gr.skip(),
        gr.skip(),
        gr.skip(),
    )


BUSY_STATUS = "A response is already generating. Press Stop first."


def busy_state():
    """Refuse to start a generation while one is running, touching nothing else.

    Gradio reads a listener's inputs when the request is queued, so a Retry or
    an Edit clicked mid-stream arrives holding the conversation as it looked at
    click time. Publishing that snapshot - which is what idle_state() would do,
    since it returns copy_turns(turns) - would overwrite whatever the running
    generation has written since, silently erasing a whole exchange. So this
    refusal skips the chatbot and the conversation state entirely, along with
    the prompt box and the token panel, and reports the reason.

    The two buttons are skipped for the same reason: the generation that owns
    the slot is still running, so it - not this refusal - decides what the
    buttons say. Forcing them idle would hide Stop while telling the user to
    press Stop, and nothing would bring it back until the running generation
    published its next batched update, which on a slow model is seconds away
    and never arrives at all if inference stalls. Skipping leaves the busy
    pair the running generation already published in place, and that
    generation restores the idle pair itself on whichever path it exits.
    """

    return (
        (gr.skip(),) * 5
        + (BUSY_STATUS,)
        + (gr.skip(),) * (len(CHAT_OUTPUT_NAMES) - 6)
    )


def generate_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    *,
    forced_ids: tuple[int, ...] = (),
    literal_prefill_tokens: int = 0,
    automatic_reasoning_close_tokens: int = 0,
    literal_text_ranges: tuple[tuple[int, int], ...] = (),
    branch_note: str = "",
    expected_load_id: str | None = None,
):
    """Stream one assistant reply for ``turns``, which must end with a user turn.

    ``assistant_prefill`` is arbitrary answer text the model replays before it
    samples anything. ``forced_ids`` is the token-level version used by a
    branch: the tokens kept from an earlier response and the alternative the
    reader picked. A branch already contains any prefix that was on the old
    response, so it takes precedence. ``branch_note`` leads the status line.

    ``expected_load_id`` is the model load ``forced_ids`` came from. Only a
    branch passes it: the runtime compares it under the model lock and raises
    ``ModelChanged`` if a load landed in between, and that exception is let
    through to the branch handler, which alone still holds the conversation
    the branch was about to replace. Ordinary chat has no such tokens and
    generates with whatever is loaded.

    ``literal_text_ranges`` marks reader-typed spans within ``forced_ids``.
    They are kept separate from ``literal_prefill_tokens`` because a terminal
    stop token typed into a branch must still end it even though reasoning
    markers earlier in the same replacement remain visible prose.

    ``automatic_reasoning_close_tokens`` preserves the provenance of the
    template close at the start of an assistant prefill, so later branches
    cannot mistake those control tokens for replaceable answer text.

    The generation slot is reserved here, before the first frame is published,
    because this is the first moment a handler is committed to generating. The
    MANAGER.busy checks in chat(), regenerate_from() and edit_message() are an
    early exit, not the guard: between such a check and the model lock that
    generate() takes sits the "Generating…" yield, and Gradio does not resume a
    handler until it has serialized that frame and sent it to the browser. A
    second click arriving inside that round trip used to sail past a manager
    that looked idle and overwrite the conversation from its stale snapshot.
    branch_with_text() has work to do before it can generate, so it takes the
    slot itself and calls _stream_reply() directly.
    """

    if not MANAGER.reserve_generation():
        yield busy_state()
        return

    try:
        yield from _stream_reply(
            turns,
            prompt_text,
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            scale_name,
            forced_ids=forced_ids,
            literal_prefill_tokens=literal_prefill_tokens,
            automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
            literal_text_ranges=literal_text_ranges,
            branch_note=branch_note,
            expected_load_id=expected_load_id,
        )
    finally:
        # Every exit runs this: a finished stream, a failure, and - the one
        # that matters - cancellation, where Gradio throws GeneratorExit in at
        # whichever yield the stream is parked on. Leaving the slot reserved
        # there would wedge the app: Send would refuse forever.
        MANAGER.release_generation()


def _stream_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    *,
    forced_ids: tuple[int, ...] = (),
    literal_prefill_tokens: int = 0,
    automatic_reasoning_close_tokens: int = 0,
    literal_text_ranges: tuple[tuple[int, int], ...] = (),
    branch_note: str = "",
    expected_load_id: str | None = None,
):
    """The body of generate_reply(), run with the generation slot held."""

    turns = copy_turns(turns)
    used_seed = resolve_seed(seed, randomize_seed)
    # Minted once for the whole stream, not once per frame: the strip is
    # replaced by the opening frame and only appended to afterwards, so a token
    # picked mid-stream is still on screen and its click must stay valid. What
    # this number invalidates is every selection made against the response this
    # one replaces.
    generation = new_metrics_generation()
    request = model_messages(
        turns, system_prompt=system_prompt, include_reasoning=keep_reasoning
    )

    pending = make_turn("assistant", "", "")
    pending["reasoning_closed"] = True
    # Where this reply came from, for the conversation list. The model is
    # stamped from the first update rather than read off MANAGER here: the
    # generator does not take the model lock until it is first resumed, and
    # a load can land in the round trip the opening frame costs. Only the
    # update knows which weights it came from. The token counts are filled
    # in as the stream arrives so a stopped or failed reply still says how
    # far it got.
    turns.append(pending)

    def snapshot(
        highlight,
        metrics,
        status,
        busy=True,
        reset_details=False,
        prompt_panel=None,
        charts_panel=None,
        trace=None,
        branch_source=gr.skip(),
        context_ids=gr.skip(),
    ):
        """One frame of the stream.

        ``reset_details`` belongs to the first frame only. That frame empties
        the strip, so a token selected in the previous response is gone and its
        probabilities must go with it. Later frames only append to the strip, so
        a token picked mid-stream stays valid and its details are left alone.

        ``prompt_panel`` and ``charts_panel`` are skipped on most frames. The
        prompt tokens are all measured before the first one is generated, so
        they are published once and never change; the charts redraw in batches
        because rebuilding an SVG per token is wasted work.

        ``context_ids`` is every prompt token, stamped like the strips and
        tagged with the model load that produced it, and is what the layer
        inspector rebuilds the model's input from.
        """

        messages, _ = display_messages(turns)
        prompt_strip, prompt_metrics, prompt_note = prompt_panel or (
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        summary_panel, surprise_panel = charts_panel or (gr.skip(), gr.skip())
        return (
            prompt_text,
            messages,
            copy_turns(turns),
            highlight,
            (generation, metrics),
            status,
            used_seed,
            *send_stop_buttons(busy),
            NO_TOKEN_SELECTED if reset_details else gr.skip(),
            [] if reset_details else gr.skip(),
            prompt_strip,
            prompt_metrics,
            prompt_note,
            summary_panel,
            surprise_panel,
            gr.skip() if trace is None else trace,
            branch_source,
            context_ids,
        )

    # The opening frame empties everything the previous response left behind,
    # the export included: a trace kept here would still be downloadable while
    # a different response was streaming in above it. The branch source goes
    # too: nothing is branchable until this response has finished or been
    # stopped, and the stamp would refuse it anyway.
    applied_prefill = bool(assistant_prefill and not forced_ids)
    stream_note = branch_note or (
        "Assistant prefill applied." if applied_prefill else ""
    )
    yield snapshot(
        strip_update([], scale_name, RESPONSE_STRIP_LABEL),
        [],
        f"{stream_note} Generating…".strip(),
        reset_details=True,
        prompt_panel=(strip_update([], scale_name), (generation, []), ""),
        charts_panel=(charts.summary_tiles({}), charts.EMPTY_CHART),
        trace={},
        branch_source=None,
        context_ids=(generation, [], MANAGER.load_id),
    )

    started = time.monotonic()
    raw_text = ""
    # Reasoning templates end the prompt with the opening <think> marker, so the
    # generated text never carries one. Only the runtime can tell us that.
    prefilled = False
    highlight: list[tuple[str, str]] = []
    metrics: list[dict] = []
    status = "The model produced no tokens."
    first = True
    forced_prefix_tokens = 0
    literal_prefill = ""
    literal_spans: tuple[tuple[int, int], ...] = ()
    producing_load_id: str | None = None

    stream = MANAGER.generate(
        request,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_new_tokens=int(max_new_tokens),
        seed=used_seed,
        analyze_prompt=bool(analyze_prompt),
        forced_ids=tuple(int(value) for value in forced_ids),
        answer_prefill=assistant_prefill if applied_prefill else "",
        literal_prefill_tokens=literal_prefill_tokens,
        automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
        literal_text_ranges=literal_text_ranges,
        load_id=expected_load_id,
    )

    try:
        # closing() releases the model lock the moment the Stop button cancels
        # this event and Gradio closes the outer generator.
        with contextlib.closing(stream):
            for update in stream:
                raw_text = update.text
                producing_load_id = update.load_id
                prefilled = update.reasoning_prefilled
                forced_prefix_tokens = update.forced_prefix_tokens
                if update.literal_prefill_text:
                    literal_prefill = update.literal_prefill_text
                if update.literal_text_spans:
                    literal_spans = update.literal_text_spans
                reasoning, answer, closed = split_response_text(
                    raw_text,
                    literal_prefill=literal_prefill,
                    literal_spans=literal_spans,
                    streaming=True,
                    reasoning_prefilled=prefilled,
                )
                pending["reasoning"] = reasoning
                pending["content"] = answer
                pending["reasoning_closed"] = closed
                highlight = strip_value(update.metrics, scale_name)
                metrics = list(update.metrics)
                pending["generated_tokens"] = len(metrics)
                status = generation_progress(len(metrics), started, used_seed)
                if stream_note:
                    status = f"{stream_note} {status}"
                prompt_panel = None
                context_ids = gr.skip()
                if first:
                    if update.model_id:
                        pending["model"] = update.model_id
                    # Every prompt token is measured before the first response
                    # token exists, so this is published once and never
                    # changes. It shares the response strip's stamp: the two
                    # are replaced together, and a click on either has to
                    # match the stamp the pair was drawn with.
                    pending["prompt_tokens"] = len(update.prompt_ids)
                    prompt_metrics = list(update.prompt_metrics)
                    prompt_panel = (
                        strip_update(prompt_metrics, scale_name),
                        (generation, prompt_metrics),
                        prompt_note_text(
                            len(prompt_metrics), update.prompt_note, "prompt"
                        ),
                    )
                    context_ids = (
                        generation,
                        [int(v) for v in update.prompt_ids],
                        update.load_id,
                    )
                yield snapshot(
                    highlight,
                    metrics,
                    status,
                    prompt_panel=prompt_panel,
                    context_ids=context_ids,
                    charts_panel=(
                        (
                            charts.summary_tiles(summarize(metrics)),
                            charts.surprise_chart(metrics),
                        )
                        if first or len(metrics) % CHART_EVERY == 0
                        else None
                    ),
                )
                first = False
    except ModelChanged:
        # Raised on the first step, before any token, and only when a branch
        # asked for the check. The opening frame is already out, but the turns
        # here are the branch's replacement, not the conversation the reader
        # was looking at; the branch handler still holds that and yields the
        # correction. generate_reply() releases the slot on the way out.
        raise
    except Exception as error:
        # The diagnostic only goes to the status line. Storing it as the
        # assistant turn would feed the failure back to the model next turn.
        # The traceback goes to the log so the cause is recoverable.
        logger.exception("Generation failed")
        reasoning, answer, _ = split_response_text(
            raw_text,
            literal_prefill=literal_prefill,
            literal_spans=literal_spans,
            reasoning_prefilled=prefilled,
        )
        pending["reasoning"] = reasoning
        pending["content"] = answer
        kept = finalize_partial(turns)
        # A failed response is not a response to export, so the trace the
        # opening frame emptied stays empty. What did arrive is still on
        # screen, though, and can be branched from like a stopped response.
        yield snapshot(
            highlight,
            metrics,
            f"Generation failed: {error}",
            busy=False,
            branch_source=(generation, producing_load_id) if kept and metrics else None,
        )
        return

    reasoning, answer, _ = split_response_text(
        raw_text,
        literal_prefill=literal_prefill,
        literal_spans=literal_spans,
        reasoning_prefilled=prefilled,
    )
    pending["reasoning"] = reasoning
    pending["content"] = answer
    # A generation can succeed and still leave nothing renderable behind: the
    # first sampled token is a hidden EOS, the model emits only whitespace,
    # which split_reasoning() strips away, or it opens and closes a reasoning
    # block without writing in it. Publishing that turn would draw a blank
    # bubble in display_messages() that model_messages() skips, so the visible
    # conversation and the model's would disagree - the UI would show a reply
    # the model never sees. finalize_partial() is what the failure and
    # cancellation paths already use for exactly this, so success uses it too:
    # it closes the reasoning block when the turn is worth keeping and drops
    # the turn when it holds neither answer nor reasoning. Dropping it leaves
    # the user turn without a reply, which is the honest shape - no assistant
    # bubble is drawn, so both transcripts agree that no reply exists.
    kept = finalize_partial(turns)
    sampling = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "seed": used_seed,
    }
    if forced_prefix_tokens:
        # The first tokens of a branched response were replayed, not sampled,
        # or came from an assistant prefill. A reader of the export needs to
        # know how many.
        sampling["forced_prefix_tokens"] = forced_prefix_tokens
    if applied_prefill:
        sampling["assistant_prefill"] = assistant_prefill
    trace = (
        build_trace(
            model_id=pending.get("model"),
            messages=request,
            response=raw_text,
            sampling=sampling,
            metrics=metrics,
        )
        if kept and metrics
        else {}
    )
    if trace:
        status = f"{status} Exports are ready."
    yield snapshot(
        highlight,
        metrics,
        status,
        busy=False,
        charts_panel=(
            charts.summary_tiles(summarize(metrics)),
            charts.surprise_chart(metrics),
        ),
        trace=trace,
        branch_source=(generation, producing_load_id) if kept and metrics else None,
    )


def chat(
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    if MANAGER.busy:
        # Before anything else, including the checks below: every other exit
        # from this function writes the conversation back, and while another
        # generation is streaming that write is a stale overwrite.
        yield busy_state()
        return

    turns = copy_turns(turns)
    message = (prompt_text or "").strip()
    if not message:
        yield idle_state(prompt_text, turns, "Enter a message first.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    turns.append(make_turn("user", message))
    yield from generate_reply(
        turns,
        "",
        system_prompt,
        keep_reasoning,
        assistant_prefill,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
        analyze_prompt,
        scale_name,
    )


def regenerate_from(
    position: int | None,
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Throw away everything after the user turn at ``position`` and reply again."""

    if MANAGER.busy:
        # Covers Retry and the chatbot's own retry button, which reach a
        # generation only through here.
        yield busy_state()
        return

    turns = copy_turns(turns)
    if position is None:
        yield idle_state(prompt_text, turns, "There is nothing to retry.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    yield from generate_reply(
        turns[: position + 1],
        prompt_text,
        system_prompt,
        keep_reasoning,
        assistant_prefill,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
        analyze_prompt,
        scale_name,
    )


def retry_last(prompt_text, turns, *settings):
    yield from regenerate_from(last_user_index(turns), prompt_text, turns, *settings)


def retry_message(event: gr.RetryData, prompt_text, turns, *settings):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    yield from regenerate_from(position, prompt_text, turns, *settings)


def edit_message(event: gr.EditData, prompt_text, turns, *settings):
    # The color scale is the last of the settings a generation is given, and
    # this handler needs it for the one path that clears the strips itself.
    scale_name = settings[-1] if settings else DEFAULT_COLOR_SCALE
    if MANAGER.busy:
        # Not just the branch that regenerates: editing an assistant turn
        # rewrites the conversation on its own, from the same stale snapshot.
        yield busy_state()
        return

    turns = copy_turns(turns)
    found = locate(turns, event.index)
    if found is None:
        yield idle_state(prompt_text, turns, "That message is no longer available.")
        return

    position, part = found
    new_value = event.value if isinstance(event.value, str) else str(event.value)

    if turns[position]["role"] == "assistant":
        edited_turn = dict(turns[position])
        edited_turn["reasoning" if part == "reasoning" else "content"] = new_value
        if not (
            (edited_turn.get("content") or "").strip()
            or (edited_turn.get("reasoning") or "").strip()
        ):
            # An assistant turn with neither answer nor reasoning is drawn as a
            # bubble by display_messages() but skipped by model_messages(), so
            # the visible transcript and the model's would disagree and the next
            # request would carry two user messages in a row. Rejecting matches
            # how an emptied user message is handled below; the alternative,
            # dropping the exchange, would silently discard the prompt too.
            yield idle_state(
                prompt_text, turns, "An assistant message cannot be emptied."
            )
            return
        # Reserve for the same reason a generation does. This branch rewrites
        # the conversation without generating, so the busy check above is not
        # enough: a Send starting in the same instant would pass its own check,
        # and whichever frame landed second would erase the other's work. The
        # slot is held across the yield, because releasing before the frame
        # reaches the browser reopens exactly that window.
        if not MANAGER.reserve_generation():
            yield busy_state()
            return
        try:
            turns[position] = edited_turn
            # The ranks and probabilities on screen describe the text the model
            # generated, not what the user just typed over it - and so do the
            # token counts the reply and everything after it were tagged with.
            turns = forget_measurements(turns, position)
            yield idle_state(
                prompt_text,
                turns,
                "Assistant message edited.",
                clear_tokens=True,
                scale_name=scale_name,
            )
        finally:
            MANAGER.release_generation()
        return

    edited = new_value.strip()
    if not edited:
        # An empty user turn is skipped by model_messages(), which would leave
        # the request with no user message at all.
        yield idle_state(prompt_text, turns, "A user message cannot be empty.")
        return

    if not MANAGER.loaded:
        # regenerate_from() would refuse too, but only after the truncation
        # below had already thrown away every later turn for a reply that is
        # never generated.
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    turns = turns[: position + 1]
    turns[position]["content"] = edited
    yield from regenerate_from(position, prompt_text, turns, *settings)


def literal_prefill_count(metrics: list[dict], kept: int) -> int:
    """How many of the first ``kept`` tokens were typed as assistant prefill.

    Those keep their literal-prefill protection when a branch replays them;
    everything after the first sampled token is ordinary response content.
    """

    count = 0
    for metric in metrics[:kept]:
        if not metric.get("literal_prefill"):
            break
        count += 1
    return count


def literal_text_ranges(metrics: list[dict], kept: int) -> tuple[tuple[int, int], ...]:
    """Contiguous reader-supplied token ranges inside a replayed prefix."""

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, metric in enumerate(metrics[:kept]):
        literal = metric.get("literal_text")
        if literal and start is None:
            start = index
        elif not literal and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, min(kept, len(metrics))))
    return tuple(ranges)


def automatic_reasoning_close_count(metrics: list[dict], kept: int) -> int:
    """Leading automatic ``</think>`` tokens preserved by a replay."""

    count = 0
    for metric in metrics[:kept]:
        if not metric.get("automatic_reasoning_close"):
            break
        count += 1
    return count


def branch_with_text(
    selected_token: dict | None,
    branch_source: tuple[int, str | None] | None,
    metrics_state: tuple[int, list[dict]],
    replacement: str,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """Replay the last response up to the clicked token, put typed text in its
    place, and let the model continue.

    The text is not limited to the model's own alternatives, so it is
    tokenized for this position: the kept tokens plus the result must decode
    to the kept text followed by exactly what was typed. It is spliced in as
    sampled content, so a stop token typed into it ends the response there,
    the same as a stop token chosen from the alternatives table.

    Unlike the other handlers, this one takes the generation slot itself,
    before it does anything, and holds it through the replay. The encoding
    waits on the model lock, and a busy check ahead of it is not enough: a
    Send that slipped in between would hold that lock for its whole
    generation, and this handler would resume afterwards with the
    conversation it was handed at click time and replay that stale response
    onto the newer one. With the slot owned first, nothing can generate while
    the encoding waits, and the stamp check below reads a strip that no
    generation can replace under it.
    """

    if not MANAGER.reserve_generation():
        yield busy_state()
        return

    try:
        yield from _branch_with_text(
            selected_token,
            branch_source,
            metrics_state,
            replacement,
            prompt_text,
            turns,
            *settings,
        )
    finally:
        # As in generate_reply(): a finished stream, a refusal, a failure and
        # a cancellation all pass through here, or the slot would stay taken.
        MANAGER.release_generation()


def _branch_with_text(
    selected_token: dict | None,
    branch_source: tuple[int, str | None] | None,
    metrics_state: tuple[int, list[dict]],
    replacement: str,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """The body of branch_with_text(), run with the generation slot held."""

    turns = copy_turns(turns)
    generation, metrics = metrics_state
    if (
        not selected_token
        or selected_token.get("generation") != generation
        or generation != _metrics_generation
        or branch_source != (generation, MANAGER.load_id)
    ):
        yield idle_state(prompt_text, turns, BRANCH_TEXT_HINT)
        return
    if not replacement:
        yield idle_state(prompt_text, turns, BRANCH_TEXT_EMPTY)
        return

    position = last_user_index(turns)
    if position is None or turns[-1]["role"] != "assistant":
        yield idle_state(prompt_text, turns, "There is no response to branch from.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    try:
        metric = metrics[int(selected_token["index"])]
    except (IndexError, TypeError, ValueError):
        yield idle_state(prompt_text, turns, BRANCH_TEXT_HINT)
        return
    if metric.get("automatic_reasoning_close"):
        yield idle_state(prompt_text, turns, BRANCH_REASONING_CLOSE)
        return
    at = int(metric["position"])
    kept = [int(m["token_id"]) for m in metrics[: at - 1]]
    if len(kept) != at - 1:
        yield idle_state(prompt_text, turns, BRANCH_TEXT_HINT)
        return
    # The stamp check above is the fast path. The load it compared against can
    # still change before the runtime takes the model lock, so the same load is
    # handed down and compared again under that lock, for the encoding and for
    # the replay alike; a mismatch there is ModelChanged.
    _generation, expected_load = branch_source
    literal_prefill_tokens = literal_prefill_count(metrics, len(kept))
    automatic_reasoning_close_tokens = automatic_reasoning_close_count(
        metrics, len(kept)
    )
    try:
        replacement_ids = MANAGER.encode_replacement(
            kept,
            replacement,
            literal_prefill_tokens=literal_prefill_tokens,
            load_id=expected_load,
        )
        branch_turns = turns[: position + 1]
        MANAGER.validate_generation_prefix(
            model_messages(
                branch_turns,
                system_prompt=settings[0],
                include_reasoning=settings[1],
            ),
            (*kept, *replacement_ids),
            max_new_tokens=int(settings[6]),
            load_id=expected_load,
        )
    except ModelChanged:
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
        return
    except (ValueError, RuntimeError) as error:
        yield idle_state(prompt_text, turns, f"🌱 {error}")
        return

    note = (
        f"Branched at token {at}: {replacement!r} instead of {metric['text']!r}."
    )
    try:
        # Not generate_reply(): the caller already holds the slot.
        replacement_start = len(kept)
        yield from _stream_reply(
            branch_turns,
            prompt_text,
            *settings,
            forced_ids=(*kept, *replacement_ids),
            literal_prefill_tokens=literal_prefill_tokens,
            automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
            literal_text_ranges=(
                *literal_text_ranges(metrics, len(kept)),
                (replacement_start, replacement_start + len(replacement_ids)),
            ),
            branch_note=note,
            expected_load_id=expected_load,
        )
    except ModelChanged:
        # ``turns`` is still the whole conversation, old response included.
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)


def branch_from(
    pick: dict | None,
    branch_source: tuple[int, str | None] | None,
    metrics_state: tuple[int, list[dict]],
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """Replay the last response up to the picked token, swap it, and continue.

    The response being branched is always the last turn: every path that
    changes the conversation under the strip re-stamps it, and the stamps
    checked here have to agree with the live one, so a pick that survives the
    checks was made against the reply on screen.
    """

    if MANAGER.busy:
        yield busy_state()
        return

    turns = copy_turns(turns)
    generation, metrics = metrics_state
    if (
        not pick
        or pick.get("generation") != generation
        or generation != _metrics_generation
        or branch_source != (generation, MANAGER.load_id)
    ):
        yield idle_state(prompt_text, turns, BRANCH_HINT)
        return

    position = last_user_index(turns)
    if position is None or turns[-1]["role"] != "assistant":
        yield idle_state(prompt_text, turns, "There is no response to branch from.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    try:
        at = int(pick["position"])
        selected_metric = metrics[at - 1]
        if at < 1:
            raise IndexError
    except (IndexError, TypeError, ValueError):
        yield idle_state(prompt_text, turns, BRANCH_HINT)
        return
    if selected_metric.get("automatic_reasoning_close"):
        yield idle_state(prompt_text, turns, BRANCH_REASONING_CLOSE)
        return
    kept = [int(metric["token_id"]) for metric in metrics[: at - 1]]
    if len(kept) != at - 1:
        yield idle_state(prompt_text, turns, BRANCH_HINT)
        return
    forced = (*kept, int(pick["token_id"]))
    literal_prefill_tokens = literal_prefill_count(metrics, len(kept))
    automatic_reasoning_close_tokens = automatic_reasoning_close_count(
        metrics, len(kept)
    )
    unchanged = pick["token_id"] == pick.get("original_id")
    if (
        unchanged
        and literal_prefill_tokens == len(kept)
        and metrics[len(kept)].get("literal_prefill")
    ):
        literal_prefill_tokens += 1
    if unchanged:
        note = f"Resampling from token {at} ({pick['text']!r})."
    else:
        note = f"Branched at token {at}: {pick['text']!r} instead of {pick['original']!r}."

    # As in branch_with_text(): the stamp check above is the fast path, and the
    # runtime compares the same load again under the model lock.
    _generation, expected_load = branch_source
    try:
        yield from generate_reply(
            turns[: position + 1],
            prompt_text,
            *settings,
            forced_ids=forced,
            literal_prefill_tokens=literal_prefill_tokens,
            automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
            literal_text_ranges=literal_text_ranges(
                metrics, len(forced) if unchanged else len(kept)
            ),
            branch_note=note,
            expected_load_id=expected_load,
        )
    except ModelChanged:
        # ``turns`` is still the whole conversation, old response included.
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)


def undo_from(
    position: int | None,
    turns: list[dict] | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Drop the exchange starting at the user turn ``position``.

    The message goes back into the input box so it can be reworded and sent again.

    Undo cancels a running generation (see ``cancels`` on its listeners), and a
    cancelled ``generate_reply`` never reaches its final yield, so every path
    here restores the Send button itself exactly as Clear and Load do. That
    includes "There is nothing to undo.": the cancel fires on the click, not on
    what this function decides afterwards.
    """

    turns = copy_turns(turns)
    if position is None:
        # Nothing is removed here, so this is the one Undo path that keeps what
        # the cancelled generator left behind and therefore has to finalize it,
        # exactly as Stop does. Every other path truncates the partial turn away.
        finalize_partial(turns)
        messages, _ = display_messages(turns)
        return (
            gr.skip(),
            messages,
            turns,
            gr.skip(),
            gr.skip(),
            "There is nothing to undo.",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 6,
        )

    remaining = turns[:position]
    messages, _ = display_messages(remaining)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    # The selected-token details describe the response being removed, so they
    # go with it, exactly as Clear resets them. So do the prompt tokens, the
    # charts and the export: all of them measure the exchange that just left.
    return (
        turns[position]["content"],
        messages,
        remaining,
        strip,
        metrics,
        "Removed the last exchange.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


def undo_last(turns, scale_name: str = DEFAULT_COLOR_SCALE):
    return undo_from(last_user_index(turns), turns, scale_name)


def undo_message(event: gr.UndoData, turns, scale_name: str = DEFAULT_COLOR_SCALE):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    return undo_from(position, turns, scale_name)


def clear_chat(scale_name: str = DEFAULT_COLOR_SCALE):
    """Empty everything the conversation owns.

    Clear cancels a running generation (see ``cancels`` on its listener), and a
    cancelled ``generate_reply`` never reaches its final yield, so this has to
    restore the Send button itself exactly as Stop does.
    """

    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    forks = new_forks()
    return (
        [],
        [],
        strip,
        metrics,
        "Conversation cleared.",
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED,
        [],
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
        forks,
        conversation_list_update(forks, []),
    )


# --------------------------------------------------------- conversation list
#
# The pane beside the chat lists every branch - the main conversation, its
# forks, and the chats started fresh - each tagged with the model that
# answered and the conversation's size in tokens. The list is a plain Radio:
# its choices are (label, name) pairs built from the turns, and picking one
# switches to that branch.


def conversation_list_update(forks: dict, turns: list[dict] | None):
    """Redraw the list, with the active branch's turns read from ``turns``."""

    return gr.update(choices=branch_choices(forks, turns), value=forks["active"])


def refresh_conversation_list(turns: list[dict] | None, forks: dict | None):
    """Redraw the list from state, for the paths that do not publish it.

    Sending, retrying, editing, undoing and loading all write the conversation
    state without knowing about the list, and a streaming reply rewrites it on
    every frame, which is where the token count and model tag move. Rather than
    thread the list through every one of those handlers, this listens to the
    state itself: Gradio fires a State's change event only when the stored
    value's hash differs, so it runs exactly when the labels could have changed.
    """

    return conversation_list_update(copy_forks(forks), turns)


# --------------------------------------------------------------------- forks


def remember_message(turns: list[dict] | None, event: gr.SelectData):
    """Keep the chatbot message a click landed on, for the Fork button.

    The content rides along so a click that has gone stale - the conversation
    was edited or extended underneath it - is recognized when Fork is pressed,
    instead of forking at whatever message now sits at that index.
    """

    try:
        index = event_index(event)
    except (TypeError, ValueError):
        return None
    if locate(turns, index) is None:
        return None
    return {"index": index, "content": event.value}


def selected_turn(turns: list[dict], selected: dict | None) -> tuple[int, str] | None:
    """The turn a remembered chatbot click still points at, if it still does."""

    if not selected:
        return None
    found = locate(turns, selected.get("index"))
    if found is None:
        return None
    messages, _ = display_messages(turns)
    shown = messages[int(selected["index"])]["content"]
    remembered = selected.get("content")
    if isinstance(remembered, str) and remembered.strip() != str(shown).strip():
        return None
    return found


def panel_reset(scale_name: str):
    """Empty the token panel for a conversation that just changed underneath it."""

    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    return (
        strip,
        metrics,
        NO_TOKEN_SELECTED,
        [],
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


PANEL_KEPT = (gr.skip(),) * 10


def fork_refused(turns: list[dict], forks: dict, status: str):
    """Change nothing but the picker, which goes back on the active fork.

    Like every fork handler this runs after cancelling any generation (see the
    ``cancels`` on its listeners), so it still has to restore the Send button
    and close out the turn the cancelled generator left behind.
    """

    turns = copy_turns(turns)
    finalize_partial(turns)
    messages, _ = display_messages(turns)
    return (
        gr.skip(),
        messages,
        turns,
        gr.skip(),
        conversation_list_update(forks, turns),
        status,
        *send_stop_buttons(False),
        *PANEL_KEPT,
    )


def fork_conversation(
    turns: list[dict] | None,
    forks: dict | None,
    selected: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Copy the conversation into a new fork and switch to it.

    With a message selected, the copy stops there (see ``fork_at``); otherwise
    the whole transcript is copied. A whole copy keeps the token panel, since
    the response it describes is still the last one on screen; a truncated
    copy loses it, the response having gone with the cut.

    Forking cancels a running generation, as Undo, Clear and Load do, so the
    turn that generator left behind is closed out here before it is copied.
    """

    forks = copy_forks(forks)
    turns = copy_turns(turns)
    finalize_partial(turns)
    forks["branches"][forks["active"]] = copy_turns(turns)
    found = selected_turn(turns, selected)
    forked, box_text = fork_at(turns, found)
    name = next_fork_name(forks)
    forks["branches"][name] = copy_turns(forked)
    forks["active"] = name
    messages, _ = display_messages(forked)

    truncated = len(forked) < len(turns)
    if truncated:
        status = (
            f"Forked at message {found[0] + 1} into {name}. "
            "Send a message to take it somewhere else."
        )
    else:
        status = (
            f"Copied the conversation into {name}. Edit or undo a message, or "
            "send a new one, to take it somewhere else."
        )
    return (
        gr.skip() if box_text is None else box_text,
        messages,
        forked,
        forks,
        conversation_list_update(forks, forked),
        status,
        *send_stop_buttons(False),
        *(panel_reset(scale_name) if truncated else PANEL_KEPT),
    )


def switch_fork(
    name: str | None,
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Put the conversation on screen away and bring another fork out."""

    forks = copy_forks(forks)
    if name not in forks["branches"]:
        return fork_refused(turns, forks, "That fork no longer exists.")
    if name == forks["active"]:
        return fork_refused(turns, forks, f"Already on {name}.")

    turns = copy_turns(turns)
    finalize_partial(turns)
    forks["branches"][forks["active"]] = turns
    forks["active"] = name
    target = copy_turns(forks["branches"][name])
    messages, _ = display_messages(target)
    count = len(target)
    return (
        gr.skip(),
        messages,
        target,
        forks,
        conversation_list_update(forks, target),
        f"Switched to {name} ({count} message{'s' if count != 1 else ''}).",
        *send_stop_buttons(False),
        *panel_reset(scale_name),
    )


def delete_fork(
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Drop the active fork and go back to the main conversation."""

    forks = copy_forks(forks)
    name = forks["active"]
    if name == MAIN_BRANCH:
        return fork_refused(
            turns,
            forks,
            "The main conversation cannot be deleted. Use Clear to empty it.",
        )

    del forks["branches"][name]
    forks["active"] = MAIN_BRANCH
    target = copy_turns(forks["branches"].setdefault(MAIN_BRANCH, []))
    messages, _ = display_messages(target)
    return (
        gr.skip(),
        messages,
        target,
        forks,
        conversation_list_update(forks, target),
        f"Deleted {name}. Back on {MAIN_BRANCH}.",
        *send_stop_buttons(False),
        *panel_reset(scale_name),
    )


def new_conversation(
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Put the conversation on screen away and start an empty one.

    Unlike Fork, nothing is copied: the new chat begins with no turns, so the
    next message is measured against the system prompt alone. The message box
    is left as it is, since whatever is typed there is likely meant for the
    new chat. Starting one cancels a running generation, as every branch
    change does, so the turn that generator left behind is closed out before
    it is put away.
    """

    forks = copy_forks(forks)
    turns = copy_turns(turns)
    finalize_partial(turns)
    forks["branches"][forks["active"]] = turns
    name = next_branch_name(forks, CHAT_PREFIX)
    forks["branches"][name] = []
    forks["active"] = name
    return (
        gr.skip(),
        [],
        [],
        forks,
        conversation_list_update(forks, []),
        f"Started {name}. Send a message to begin it.",
        *send_stop_buttons(False),
        *panel_reset(scale_name),
    )


# --------------------------------------------------------------- save / load


def save_conversation(turns, system_prompt):
    if not turns:
        return gr.update(value=None, visible=False), "There is nothing to save yet."

    # Gradio only serves files it created or was told to allow, so the saved
    # conversation has to live inside its upload folder.
    directory = Path(get_upload_folder()) / "chatlab-conversations"
    directory.mkdir(parents=True, exist_ok=True)
    # The timestamp only resolves to the second, and every session shares this
    # upload folder, so a random suffix keeps two saves from landing on the same
    # path and silently overwriting each other's download.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"conversation-{stamp}-{uuid4().hex[:8]}.json"
    path.write_text(to_json(turns, system_prompt=system_prompt), encoding="utf-8")
    return (
        gr.update(value=str(path), visible=True),
        f"Saved {len(turns)} message{'s' if len(turns) != 1 else ''}.",
    )


def load_conversation(file_path, turns, scale_name: str = DEFAULT_COLOR_SCALE):
    """Replace the conversation with a saved one.

    A failed load keeps the conversation already on screen, so a bad file
    cannot wipe it, and leaves the token panel describing it alone. Loading
    cancels any generation still running, so the buttons are restored here for
    the same reason Clear restores them: a cancelled generator never reaches
    its final yield. For the same reason the kept conversation has to be
    finalized like Stop does - the cancelled generator left its last turn with
    a pending reasoning block, which would spin for the rest of the session,
    or empty if the cancel landed before the first token.
    """

    def keep_current(status):
        """Return the conversation the cancelled generator left behind."""

        kept = copy_turns(turns)
        finalize_partial(kept)
        messages, _ = display_messages(kept)
        return (
            messages,
            kept,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            status,
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 6,
        )

    if not file_path:
        return keep_current("No file chosen.")
    try:
        loaded, system_prompt = from_json(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return keep_current(f"Could not load that file: {error}")

    # A successful load replaces the conversation wholesale, so whatever the
    # cancelled generator left behind goes with it and needs no finalizing.
    turns = loaded
    messages, _ = display_messages(turns)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    # The selected token described a response from the conversation being
    # replaced, so it goes with it, exactly as Clear and Undo reset it. The
    # charts and the export measured that response too, and a loaded
    # conversation has no measurements of its own to put in their place.
    return (
        messages,
        turns,
        system_prompt,
        strip,
        metrics,
        f"Loaded {len(turns)} message{'s' if len(turns) != 1 else ''}.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


# ---------------------------------------------------------------- score text


def score_text(
    context: str,
    text: str,
    use_chat_template: bool,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Measure text the model did not write, and put it in the same panel."""

    skip = gr.skip()
    if not MANAGER.loaded:
        return (skip,) * 7 + ("Download and load a model first.", skip, skip, skip)

    try:
        result = MANAGER.score_text(
            text, context=context or "", use_chat_template=bool(use_chat_template)
        )
    except Exception as error:
        return (skip,) * 7 + (f"Could not score that text: {error}", skip, skip, skip)

    summary = summarize(result.metrics)
    status = (
        f"Scored {summary['token_count']:,} tokens. "
        f"Perplexity {summary['perplexity']:,.1f}."
    )
    # What was scored comes before how exactly it was scored: the template
    # caveat says which passage the numbers describe, the seam caveat says how
    # sure their first token is.
    if result.chat_template_missing:
        status = f"{status} {TEMPLATE_CAVEAT}"
    if not result.seam_verified:
        status = f"{status} {SEAM_CAVEAT}"
    # Both strips are replaced, so they take one shared stamp - and that stamp
    # is what drops a click made against the response they overwrite.
    generation = new_metrics_generation()
    return (
        strip_update(result.metrics, scale_name, "Scored tokens — click one"),
        stamped(result.metrics, generation),
        strip_update(result.context_metrics, scale_name),
        stamped(result.context_metrics, generation),
        prompt_note_text(len(result.context_metrics), "", "context"),
        charts.summary_tiles(summary),
        charts.surprise_chart(result.metrics, title="Surprise per scored token"),
        status,
        NO_TOKEN_SELECTED,
        [],
        (generation, [int(value) for value in result.context_ids], MANAGER.load_id),
    )


# ------------------------------------------------------- layers and attention
#
# The logit lens and the attention view cost a forward pass over everything
# before the token, so they run on demand from a button rather than on every
# click. The click leaves the strip position in ``inspect_target`` stamped
# with the strip's generation number, exactly as branching does, and the
# button refuses a target whose strip has since been replaced.

INSPECT_HINT = "Click a token above, then press **Inspect layers**."
INSPECT_BUSY = "Wait for the response to finish before inspecting a token."
INSPECT_GONE = "That token is no longer on screen. Click one and try again."
INSPECT_FIRST = "Nothing came before this token, so the model never predicted it."
INSPECT_MODEL_CHANGED = (
    "The model has been reloaded since these tokens were produced, so they "
    "cannot be explained by the weights in memory. Generate or score again."
)
INSPECT_OUTPUT_ONLY = (
    "Only the output is shown: this model's intermediate layers could not be "
    "read the way it reads its own output."
)


def remember_inspect_target(strip: str):
    """A select listener that keeps the clicked position for the inspector.

    Unlike remember_selection(), every token counts: a prompt token has layers
    and attention behind it just as a response token does. Only the first
    token of a sequence has nothing to show, and inspect_layers() says so.
    """

    def remember(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
        generation, metrics = metrics_state
        if generation != _metrics_generation:
            return None
        try:
            index = event_index(event)
            metrics[index]
        except (IndexError, TypeError, ValueError):
            return None
        return {"generation": generation, "strip": strip, "index": index}

    return remember


def inspect_layers(
    target: dict | None,
    metrics_state: tuple[int, list[dict]],
    prompt_metrics_state: tuple[int, list[dict]],
    context_state: tuple[int, list[int]],
    layer,
):
    """Run the logit lens and attention readout for the clicked token.

    The model's input is rebuilt from the prompt ids published with the
    response and the token ids in the response metrics, so the pass sees
    exactly the sequence the token was generated from.

    This is a generator for the same reason generate_reply() is: Gradio does
    not resume a streaming handler until the browser has been sent the frame
    it yielded. The generation slot is therefore held not just for the pass
    but until the readout is on screen, so Send, Retry and Branch cannot
    slip in between the two and have the readout land on top of their
    reset. Paths that replace the strips without taking the slot - Clear,
    Undo, Load, a fork switch, Score text - are caught by the stamp instead:
    it is checked before the frame goes out and again once it has arrived,
    and a readout for a token that is gone is taken back down.
    """

    skip = gr.skip()
    refused = (skip, skip, skip, skip)
    if not target or target.get("generation") != _metrics_generation:
        yield (*refused, INSPECT_HINT)
        return
    generation, metrics = metrics_state
    _prompt_generation, prompt_metrics = prompt_metrics_state
    context_generation, context_ids, load_id = context_state
    if generation != target["generation"] or context_generation != generation:
        yield (*refused, INSPECT_GONE)
        return
    if not MANAGER.loaded:
        yield (*refused, "Download and load a model first.")
        return
    # Loading a model leaves the strips on screen, and their token ids mean
    # nothing to a different tokenizer, so the ids carry the load that
    # produced them and only that load may explain them. The load, not the
    # model ID: re-downloading the same ID can bring in a newer snapshot.
    # This is the early exit; the check that counts is the one inspect()
    # makes under the model lock, since a load can land between here and it.
    if load_id != MANAGER.load_id:
        yield (*refused, INSPECT_MODEL_CHANGED)
        return

    context_ids = [int(value) for value in context_ids]
    position = int(target["index"])
    if target["strip"] == "prompt":
        if (
            position >= len(prompt_metrics)
            or position >= len(context_ids)
            or int(prompt_metrics[position]["token_id"]) != context_ids[position]
        ):
            yield (*refused, INSPECT_GONE)
            return
        index = position
    else:
        if position >= len(metrics):
            yield (*refused, INSPECT_GONE)
            return
        index = len(context_ids) + position
    if index == 0:
        yield (*refused, INSPECT_FIRST)
        return
    sequence = context_ids + [int(metric["token_id"]) for metric in metrics]

    if not MANAGER.reserve_generation():
        yield (*refused, INSPECT_BUSY)
        return
    try:
        started = time.monotonic()
        try:
            insight = MANAGER.inspect(
                sequence, index, context_count=len(context_ids), load_id=load_id
            ).to_dict()
        except ModelChanged:
            yield (*refused, INSPECT_MODEL_CHANGED)
            return
        except Exception as error:
            yield (*refused, f"Could not inspect that token: {error}")
            return
        if target["generation"] != _metrics_generation:
            yield (*refused, INSPECT_GONE)
            return

        layer_count = len(insight["attention"])
        layer = min(max(int(layer or 0), 0), layer_count)
        where = "Prompt token" if target["strip"] == "prompt" else "Token"
        shown = html.escape(repr(insight["token_text"]))
        read = len(insight["layers"]) - 1
        status = (
            f"{where} {position + 1}: `{shown}`, read through {read} "
            f"layers in {time.monotonic() - started:.1f}s."
        )
        if not read:
            status = f"{status} {INSPECT_OUTPUT_ONLY}"
        if not layer_count:
            status = f"{status} This model did not return attention weights."
        yield (
            charts.logit_lens_chart(insight),
            charts.attention_strip(insight, layer),
            gr.update(maximum=max(layer_count, 1), value=layer),
            insight,
            status,
        )
        # Resumed once the browser has the frame above. If the strips were
        # replaced while it was in flight, their reset was applied first and
        # the readout now sits on top of it, so take it back down.
        if target["generation"] != _metrics_generation:
            yield (charts.EMPTY_LENS, charts.EMPTY_ATTENTION, skip, None, INSPECT_GONE)
    finally:
        MANAGER.release_generation()


def render_attention(insight: dict | None, layer):
    """Repaint the attention strip for another layer without a new pass."""

    if not insight:
        return gr.skip()
    return charts.attention_strip(insight, int(layer or 0))


def reset_inspection(insight: dict | None):
    """Empty the inspector when the strips it described are replaced.

    Bound to the response metrics state, which every path that redraws the
    strips writes. Streaming writes it on every frame too, so this skips
    while there is nothing to clear rather than repainting an empty panel a
    hundred times per response.
    """

    if insight is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return charts.EMPTY_LENS, charts.EMPTY_ATTENTION, None, INSPECT_HINT


# One pair of rules per tile: the icon it shows, and the name it pops up.
# Gradio stamps each option's text on its label as data-testid, which is the
# only hook a Radio gives CSS.
#
# Both are drawn on the label, so both would otherwise join the radio's
# accessible name and have a screen reader read "speech balloon Chat", or
# "speech balloon Chat Chat" once the tooltip is up. The empty string after
# the slash is the generated text's alternative text, which keeps the pair of
# them out of the name and leaves the page's own name to stand for the tile.
NAV_TILE_CSS = "\n".join(
    f'#nav label[data-testid="{name}-radio-label"]::before '
    f'{{ content: "{NAV_ICONS[name]}" / ""; }}\n'
    f'#nav label[data-testid="{name}-radio-label"]:hover::after,\n'
    f'#nav label[data-testid="{name}-radio-label"]:has(input:focus-visible)::after '
    f'{{ content: "{name}" / ""; }}'
    for name in PAGES
)

CSS = f"""
.gradio-container {{ max-width: none !important; }}
#hero, #models-hero, #settings-hero {{ padding: 0.5rem 0 0.2rem; }}
#hero h1, #models-hero h1, #settings-hero h1 {{ font-size: 2.1rem; margin-bottom: 0.25rem; }}
#model-status {{ min-height: 128px; }}

/* The shell is one row: nav, conversations, page. It never wraps, and the two
   panes keep their widths and stay put while the page scrolls. */
#shell {{ flex-wrap: nowrap; align-items: stretch; }}
#nav-pane, #conversation-pane {{
  position: sticky; top: 0; align-self: flex-start; max-height: 100vh;
}}
/* Both panes are sticky, so each is its own stacking context and the later
   one would paint over the nav's tooltip. Lift the nav above it. */
#nav-pane {{ z-index: 5; }}
#nav-pane {{
  flex: 0 0 {NAV_PANE_WIDTH}px !important; min-width: {NAV_PANE_WIDTH}px !important;
  height: 100vh; padding: 0.6rem 0.5rem 0.6rem 0;
  border-right: 1px solid var(--border-color-primary);
}}
#conversation-pane {{
  flex: 0 0 {CONVERSATION_PANE_WIDTH}px !important;
  min-width: {CONVERSATION_PANE_WIDTH}px !important;
  overflow-y: auto; padding-right: 0.5rem;
  border-right: 1px solid var(--border-color-primary);
}}
/* The nav is a Radio drawn as a column of tiles. Its inputs are hidden, the
   selected tile is filled, and the last tile (Settings) is pushed to the
   bottom. */
#nav-pane > *, #nav, #nav .wrap {{ height: 100%; }}
/* Gradio's fieldset scrolls its own content, which would cut the tooltip off
   at the tile's edge. */
#nav {{ overflow: visible !important; }}
#nav .wrap {{ flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: 0.3rem; }}
#nav label {{
  position: relative;
  justify-content: center; text-align: center; padding: 0.55rem 0.2rem;
  font-size: 1.3rem; line-height: 1.25; border-radius: 8px; box-shadow: none;
  background: transparent;
  /* The selected tile is outlined; the others hold the same border in
     transparent so picking a page does not nudge the icons. */
  border: 1px solid transparent;
}}
/* The page name is still on the tile, an inch out of sight, so the browser
   reads it out and the icon that ::before draws sits where it was. */
#nav label span {{
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip-path: inset(50%);
}}
/* Hover or keyboard focus brings the name back as a bubble beside the tile. */
#nav label::after {{
  position: absolute; left: calc(100% + 0.5rem); top: 50%;
  transform: translateY(-50%);
  padding: 0.25rem 0.5rem; border-radius: 6px; white-space: nowrap;
  font-size: 0.8rem; font-weight: 500; line-height: 1.3; pointer-events: none;
  background: var(--body-text-color); color: var(--background-fill-primary);
}}
{NAV_TILE_CSS}
#nav label:hover {{ background: var(--background-fill-secondary); }}
#nav label.selected {{
  background: var(--block-background-fill); color: var(--body-text-color); font-weight: 600;
  border: 1px solid var(--border-color-primary);
}}
#nav label:last-child {{ margin-top: auto; }}
/* The radio inputs stay in the tab order, just out of sight, and the tile
   they belong to shows the keyboard focus ring. */
#nav label input {{
  position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; pointer-events: none;
}}
#nav label:has(input:focus-visible) {{
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}}
.model-list .wrap {{ flex-direction: column; align-items: stretch; gap: 0.2rem; }}
.model-list label {{ font-size: 0.82rem; line-height: 1.3; word-break: break-word; }}
/* Gradio stamps each option's text on its label as data-testid, which is the
   only hook a Radio gives CSS. An incomplete model's label ends in
   "· incomplete", so it is tinted amber in both themes. */
.model-list label[data-testid*="· incomplete"] {{ border-color: #d97706; }}
.model-list label[data-testid*="· incomplete"]:not(.selected) {{
  background: rgba(217, 119, 6, 0.09);
}}
.model-list label[data-testid*="· incomplete"] span {{ color: #b45309; }}
.dark .model-list label[data-testid*="· incomplete"] span {{ color: #fbbf24; }}
.model-sort label span {{ font-size: 0.8rem; }}
.remove-confirm {{
  border: 1px solid #d97706; border-radius: 8px; padding: 0.4rem 0.6rem;
  background: rgba(217, 119, 6, 0.09);
}}
.model-detail {{ font-size: 0.85rem; }}
.model-detail p, .model-detail ul, .model-detail li {{ margin: 0.15rem 0; }}
.model-detail code {{ word-break: break-all; }}
#token-strip {{ min-height: 150px; }}
#token-strip span, #prompt-strip span {{ cursor: pointer; border-radius: 5px; }}
/* Token fills are light in both themes, so their ink is pinned dark. */
#token-strip .textspan.hl, #prompt-strip .textspan.hl,
#token-strip .category-label, #prompt-strip .category-label {{ color: #0b0b0b; }}
.footer-note {{ color: var(--body-text-color-subdued); font-size: 0.9rem; }}
.scale-caption {{ color: var(--body-text-color-subdued); font-size: 0.85rem; }}

/* The conversation list is a Radio whose labels carry a line break: the
   name and title on the first line, the model and token count on the
   second. Stack the entries and let the break through. */
#conversation-list .wrap {{ flex-direction: column; align-items: stretch; gap: 0.4rem; }}
#conversation-list label {{ align-items: flex-start; }}
#conversation-list label input {{ margin-top: 0.3rem; }}
#conversation-list label span {{
  white-space: pre-line; line-height: 1.35; overflow-wrap: anywhere;
}}

.viz-root {{
  --viz-ink: #0b0b0b;
  --viz-muted: #898781;
  --viz-grid: #e1e0d9;
  --viz-axis: #c3c2b7;
  --viz-line: #2a78d6;
  --viz-band: #cde2fb;
  margin: 0;
  font-family: var(--font, system-ui, -apple-system, "Segoe UI", sans-serif);
}}
.dark .viz-root {{
  --viz-ink: #ffffff;
  --viz-muted: #898781;
  --viz-grid: #2c2c2a;
  --viz-axis: #383835;
  --viz-line: #3987e5;
  --viz-band: #1c5cab;
}}
.viz-root svg {{ width: 100%; height: auto; display: block; }}
.viz-title {{ color: var(--viz-ink); font-size: 0.9rem; font-weight: 600; padding: 0 0 0.2rem; }}
.viz-sub {{ color: var(--viz-muted); font-weight: 400; font-size: 0.8rem; margin-left: 0.4rem; }}
.viz-grid {{ stroke: var(--viz-grid); stroke-width: 1; }}
.viz-axis {{ stroke: var(--viz-axis); stroke-width: 1; }}
.viz-band {{ fill: var(--viz-band); opacity: 0.55; stroke: none; }}
.viz-line {{ fill: none; stroke: var(--viz-line); stroke-width: 2; stroke-linejoin: round; }}
.viz-peak-dot {{ fill: var(--viz-line); stroke: var(--body-background-fill); stroke-width: 2; }}
.viz-peak-label, .viz-tick {{ fill: var(--viz-muted); font-size: 10px; font-variant-numeric: tabular-nums; }}
.viz-hit {{ fill: transparent; }}
.viz-empty, .viz-note {{ color: var(--body-text-color-subdued); font-size: 0.85rem; padding: 0.4rem 0; }}
.viz-tiles {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
.viz-tile {{
  flex: 1 1 5.5rem; padding: 0.45rem 0.6rem; border-radius: 8px;
  background: var(--background-fill-secondary);
}}
.viz-value {{ color: var(--viz-ink); font-size: 1.25rem; line-height: 1.2; }}
.viz-label {{ color: var(--viz-muted); font-size: 0.72rem; text-transform: lowercase; }}

.viz-line-faint {{ stroke: var(--viz-band); stroke-width: 1.5; }}
.viz-marker {{ stroke: var(--viz-muted); stroke-width: 1; stroke-dasharray: 3 3; }}
.viz-table-wrap {{ max-height: 230px; overflow-y: auto; margin-top: 0.3rem; }}
.viz-table {{ width: 100%; font-size: 0.78rem; border-collapse: collapse; }}
.viz-table th, .viz-table td {{
  text-align: left; padding: 0.15rem 0.4rem; color: var(--viz-ink);
  border-bottom: 1px solid var(--viz-grid); font-variant-numeric: tabular-nums;
}}
.viz-table th {{
  color: var(--viz-muted); font-weight: 500; position: sticky; top: 0;
  background: var(--body-background-fill);
}}
.viz-hit-row td {{ font-weight: 600; }}
.attn-strip {{ line-height: 1.9; white-space: pre-wrap; word-break: break-word; }}
.attn-token {{ border-radius: 4px; padding: 0.05rem 0.1rem; margin: 0 1px; color: var(--body-text-color); }}
.attn-query {{ outline: 1.5px dashed var(--viz-muted); }}
.attn-predicted {{ outline: 1.5px solid var(--viz-ink); margin-left: 0.3rem; }}
.attn-top {{ font-size: 0.8rem; columns: 2; margin: 0.3rem 0 0; padding-left: 1.4rem; color: var(--viz-ink); }}
"""


# Gradio's Textbox sends on Enter only when it is a single-line box, and on
# Shift+Enter when it has more than one line, so the "Enter sends" preference
# is expressed by choosing the box's starting height. The box grows to
# MESSAGE_BOX_MAX_LINES either way.
MESSAGE_BOX_MAX_LINES = 8


def message_box_settings(enter_sends: bool) -> dict:
    """Textbox settings that make Enter (or Shift+Enter) send the message."""
    if enter_sends:
        return {
            "lines": 1,
            "max_lines": MESSAGE_BOX_MAX_LINES,
            "placeholder": "Ask OLMo something… Enter sends, Shift+Enter starts a new line.",
        }
    return {
        "lines": 3,
        "max_lines": MESSAGE_BOX_MAX_LINES,
        "placeholder": "Ask OLMo something… Shift+Enter sends, Enter starts a new line.",
    }


def set_message_box_keys(enter_sends: bool):
    return gr.update(**message_box_settings(enter_sends))


def show_page(page: str):
    """Show the chosen page. The conversations pane comes and goes with Chat."""
    return [
        gr.update(visible=page == CHAT_PAGE),
        *(gr.update(visible=page == name) for name in PAGES),
    ]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Chatlab", css=CSS, theme=gr.themes.Soft()) as demo:
        conversation_state = gr.State([])
        metrics_state = gr.State(empty_metrics())
        prompt_metrics_state = gr.State(empty_metrics())
        trace_state = gr.State({})
        # Branching from a token: the stamp of the last chat response's strip,
        # the strip position last clicked, and the alternative picked for it.
        branch_source = gr.State(None)
        selected_token = gr.State(None)
        branch_pick = gr.State(None)
        # Forking: the other transcripts, and the chatbot message last clicked.
        forks_state = gr.State(new_forks())
        selected_message = gr.State(None)
        # Layer inspection: the prompt ids behind the strips, the strip
        # position last clicked, and the last readout for re-rendering.
        context_ids_state = gr.State((*empty_metrics(), None))
        inspect_target = gr.State(None)
        insight_state = gr.State(None)

        with gr.Row(elem_id="shell"):
            # The thin pane at the far left picks the page: Chat, Models, or
            # Settings. The stylesheet stacks the choices and pins Settings to
            # the bottom.
            with gr.Column(scale=0, min_width=NAV_PANE_WIDTH, elem_id="nav-pane"):
                nav = gr.Radio(
                    choices=list(PAGES),
                    value=CHAT_PAGE,
                    show_label=False,
                    container=False,
                    elem_id="nav",
                )

            # The conversations pane sits beside the nav and shows with Chat only.
            with gr.Column(
                scale=0, min_width=CONVERSATION_PANE_WIDTH, elem_id="conversation-pane"
            ) as conversation_pane:
                gr.Markdown("## Conversations")
                conversation_list = gr.Radio(
                    choices=branch_choices(new_forks(), []),
                    value=MAIN_BRANCH,
                    show_label=False,
                    elem_id="conversation-list",
                )
                with gr.Row():
                    # The pane is narrow, so the buttons give up their usual
                    # minimum width to share one row.
                    new_button = gr.Button("➕ New", size="sm", min_width=60)
                    fork_button = gr.Button("🌿 Fork", size="sm", min_width=60)
                    delete_fork_button = gr.Button("🗑️ Delete", size="sm", min_width=60)
                gr.Markdown(
                    "Each entry names the model that replied and the size of the "
                    "conversation in tokens: the prompt behind its latest reply plus "
                    "the reply itself. New starts an empty chat. Fork copies the "
                    "conversation on screen; click a message first to fork at that "
                    "point.",
                    elem_classes=["scale-caption"],
                )

            # The three pages share the rest of the width; one is visible at a
            # time, chosen by the nav.
            with gr.Column(scale=1, elem_id="chat-page") as chat_page:
                gr.Markdown(
                    "# Chatlab\nChat with an open model and see exactly how likely every generated token was.",
                    elem_id="hero",
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Chat"):
                                chatbot = gr.Chatbot(
                                    type="messages",
                                    label="Conversation",
                                    height=560,
                                    editable="all",
                                    placeholder="Load a model, then start a conversation.",
                                )
                                prompt = gr.Textbox(
                                    label="Message",
                                    **message_box_settings(enter_sends=True),
                                )
                                with gr.Row():
                                    send_button = gr.Button("Send", variant="primary")
                                    stop_button = gr.Button(
                                        "Stop", variant="stop", visible=False
                                    )
                                    retry_button = gr.Button("🔁 Retry")
                                    undo_button = gr.Button("↩️ Undo last")
                                    clear_button = gr.Button("🗑️ Clear")
                                with gr.Row():
                                    save_button = gr.Button("💾 Save conversation")
                                    load_upload = gr.UploadButton(
                                        "📂 Load conversation",
                                        file_types=[".json"],
                                        type="filepath",
                                    )
                                saved_file = gr.File(
                                    label="Saved conversation",
                                    visible=False,
                                    interactive=False,
                                )
                                generation_status = gr.Markdown("Ready.")
                                with gr.Accordion("Export full metric trace", open=False):
                                    with gr.Row():
                                        gr.DownloadButton(
                                            "Download JSON",
                                            value=lambda trace: write_trace_export(
                                                trace, "json"
                                            ),
                                            inputs=trace_state,
                                            size="sm",
                                        )
                                        gr.DownloadButton(
                                            "Download CSV",
                                            value=lambda trace: write_trace_export(trace, "csv"),
                                            inputs=trace_state,
                                            size="sm",
                                        )
                                    gr.Markdown(
                                        "Exports include every token metric and all recorded "
                                        "alternatives for the latest completed response.",
                                        elem_classes=["footer-note"],
                                    )

                            with gr.Tab("Score text"):
                                gr.Markdown(
                                    "Measure text the model did not write. One forward pass "
                                    "gives every token the same rank, probability, surprise, "
                                    "and entropy the chat view shows."
                                )
                                score_context = gr.Textbox(
                                    label="Context (optional)",
                                    placeholder="Text that comes before the part you want scored.",
                                    lines=3,
                                )
                                use_chat_template = gr.Checkbox(
                                    value=False,
                                    label="Treat the context as a chat message",
                                    info=(
                                        "Wraps the context in the model's chat template, so the "
                                        "scored text is measured as a reply. Models without a "
                                        "chat template score the context as plain text, and say so."
                                    ),
                                )
                                score_input = gr.Textbox(
                                    label="Text to score",
                                    placeholder="Paste the text you want measured…",
                                    lines=8,
                                )
                                score_button = gr.Button("Score text", variant="primary")
                                score_status = gr.Markdown("Nothing scored yet.")

                    with gr.Column(scale=2):
                        gr.Markdown("## Under the hood")
                        color_scale = gr.Dropdown(
                            choices=list(COLOR_SCALES),
                            value=DEFAULT_COLOR_SCALE,
                            label="Color tokens by",
                        )
                        scale_caption = gr.Markdown(
                            COLOR_SCALES[DEFAULT_COLOR_SCALE].caption,
                            elem_classes=["scale-caption"],
                        )
                        token_strip = gr.HighlightedText(
                            label=RESPONSE_STRIP_LABEL,
                            color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                            show_legend=True,
                            combine_adjacent=False,
                            elem_id="token-strip",
                        )
                        token_detail = gr.Markdown(NO_TOKEN_SELECTED)
                        alternatives = gr.Dataframe(
                            headers=["Token ID", "Token", "Raw probability"],
                            datatype=["number", "str", "number"],
                            interactive=False,
                            label="Most likely alternatives — click one to branch into it",
                        )
                        with gr.Row():
                            branch_button = gr.Button("🌱 Branch from token", size="sm")
                        gr.Markdown(
                            "Branching keeps the response up to the selected token, puts "
                            "the alternative in its place, and lets the model continue "
                            "from there.",
                            elem_classes=["scale-caption"],
                        )
                        with gr.Row():
                            branch_text = gr.Textbox(
                                label="Or type your own replacement",
                                placeholder=(
                                    "Text to put where the selected token was. Include a "
                                    "leading space if the word needs one."
                                ),
                                lines=1,
                                scale=3,
                            )
                            branch_text_button = gr.Button(
                                "✏️ Branch with text", size="sm", scale=0, min_width=160
                            )
                        gr.Markdown(
                            "The typed text replaces the selected token exactly as written, "
                            "whether or not the model would ever have chosen it, and the "
                            "model continues from there.",
                            elem_classes=["scale-caption"],
                        )
                        with gr.Accordion("Layers and attention", open=False):
                            with gr.Row():
                                inspect_button = gr.Button(
                                    "🔬 Inspect layers", size="sm", scale=0, min_width=160
                                )
                                inspect_status = gr.Markdown(
                                    INSPECT_HINT, elem_classes=["scale-caption"]
                                )
                            lens_panel = gr.HTML(charts.EMPTY_LENS)
                            attention_layer = gr.Slider(
                                0,
                                1,
                                value=0,
                                step=1,
                                label="Attention layer",
                                info="0 averages every layer. Release the slider to repaint.",
                            )
                            attention_panel = gr.HTML(charts.EMPTY_ATTENTION)
                        summary_panel = gr.HTML(charts.summary_tiles({}))
                        surprise_panel = gr.HTML(charts.EMPTY_CHART)
                        with gr.Accordion("Prompt and context tokens", open=False):
                            prompt_note = gr.Markdown("", elem_classes=["scale-caption"])
                            prompt_strip = gr.HighlightedText(
                                label="Prompt tokens — click one",
                                color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                                show_legend=True,
                                combine_adjacent=False,
                                elem_id="prompt-strip",
                            )

                gr.Markdown(
                    "Rank and raw probability come from the unmodified model distribution. "
                    "Sampling probability includes temperature, top-k, and top-p. Quantized models may produce slightly different ranks.",
                    elem_classes=["footer-note"],
                )

            with gr.Column(
                scale=1, visible=False, elem_id="models-page"
            ) as models_page:
                gr.Markdown(
                    "# Models\nDownload a model from Hugging Face, or load one "
                    "already on disk. Files are kept in your normal Hugging Face cache.",
                    elem_id="models-hero",
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Model")
                        model_id = gr.Textbox(
                            value=os.environ.get("OLMO_MODEL_ID", DEFAULT_MODEL),
                            label="Hugging Face model ID",
                            placeholder="organization/model-name",
                            info="The default OLMo 3 7B model is about 15 GB in full precision.",
                        )
                        hf_token = gr.Textbox(
                            label="Hugging Face token (optional)",
                            type="password",
                            placeholder="Only needed for gated or private models",
                        )
                        with gr.Row():
                            download_load_button = gr.Button(
                                "Download and load", variant="primary", size="sm"
                            )
                            download_button = gr.Button("Download only", size="sm")
                            cached_button = gr.Button("Load cached", size="sm")
                            unload_button = gr.Button("Unload", size="sm")
                        model_status = gr.Markdown(
                            status_card(
                                "No model loaded",
                                "Choose a model under My Models, or enter a Hugging Face model ID to download one. Files are kept in your normal Hugging Face cache.",
                            ),
                            elem_id="model-status",
                        )

                        gr.Markdown("## Model search")
                        with gr.Row():
                            search_query = gr.Textbox(
                                label="Search Hugging Face",
                                placeholder="Model name, organization, or topic…",
                                max_lines=1,
                                scale=3,
                            )
                            search_button = gr.Button(
                                "🔍 Search", size="sm", scale=0, min_width=120
                            )
                        search_results = gr.Radio(
                            choices=[],
                            label="Search results",
                            show_label=False,
                            elem_classes=["model-list"],
                        )
                        search_detail = gr.Markdown(SEARCH_HINT, elem_classes=["model-detail"])
                        search_results_state = gr.State({})

                    with gr.Column():
                        gr.Markdown("## My Models")
                        my_models_summary = gr.Markdown("", elem_classes=["scale-caption"])
                        sort_models = gr.Dropdown(
                            choices=list(MODEL_SORT_ORDERS),
                            value=DEFAULT_MODEL_SORT,
                            label="Sort by",
                            elem_classes=["model-sort"],
                        )
                        my_models = gr.Radio(
                            choices=[],
                            label="Downloaded models",
                            show_label=False,
                            elem_classes=["model-list"],
                        )
                        my_model_detail = gr.Markdown("", elem_classes=["model-detail"])
                        with gr.Row():
                            redownload_button = gr.Button("⬇️ Redownload", size="sm")
                            remove_button = gr.Button("🗑️ Remove", size="sm")
                            refresh_models_button = gr.Button("↻ Refresh", size="sm")
                        with gr.Column(
                            visible=False, elem_classes=["remove-confirm"]
                        ) as remove_confirm:
                            remove_question = gr.Markdown("", elem_classes=["model-detail"])
                            with gr.Row():
                                confirm_remove_button = gr.Button(
                                    "Remove from disk", variant="stop", size="sm"
                                )
                                cancel_remove_button = gr.Button("Cancel", size="sm")
                        # The model the open confirmation is about; None when closed.
                        pending_removal = gr.State(None)

            with gr.Column(
                scale=1, visible=False, elem_id="settings-page"
            ) as settings_page:
                gr.Markdown(
                    "# Settings\nHow every reply is prompted, sampled, and measured.",
                    elem_id="settings-hero",
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## System prompt, reasoning, and prefill")
                        system_prompt = gr.Textbox(
                            label="System prompt",
                            placeholder="You are a careful assistant that answers concisely.",
                            lines=3,
                            info="Sent as a system message ahead of the conversation. Leave empty to use the model's default behavior.",
                        )
                        assistant_prefill = gr.Textbox(
                            label="Assistant prefill (optional)",
                            placeholder="Start every reply with these exact words…",
                            lines=2,
                            info=(
                                "Replays this text as the start of each answer, then lets the "
                                "model continue. For reasoning models, Chatlab closes the "
                                "reasoning block first so this remains visible answer text."
                            ),
                        )
                        keep_reasoning = gr.Checkbox(
                            value=False,
                            label="Send previous reasoning back to the model",
                            info="Off by default. Think models write a fresh reasoning block each turn, so replaying old ones burns context and usually hurts the next answer.",
                        )

                        gr.Markdown("## Input")
                        enter_sends = gr.Checkbox(
                            value=True,
                            label="Enter sends the message",
                            info="Shift+Enter starts a new line. Turn off to swap the two.",
                        )

                    with gr.Column():
                        gr.Markdown("## Sampling and analysis")
                        temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature")
                        top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top-p")
                        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
                        max_new_tokens = gr.Slider(
                            1, 8192, value=1024, step=1, label="Maximum new tokens"
                        )
                        seed = gr.Number(
                            value=42,
                            precision=0,
                            minimum=0,
                            label="Random seed",
                            info="Updated after each response so you can reproduce it.",
                        )
                        randomize_seed = gr.Checkbox(
                            value=True,
                            label="🎲 New seed each response",
                            info="Turn off to lock the seed and reproduce a response exactly.",
                        )
                        analyze_prompt = gr.Checkbox(
                            value=True,
                            label="Measure prompt tokens",
                            info="Scores every prompt token during the same pass that warms the cache.",
                        )

        nav.change(
            show_page, nav, [conversation_pane, chat_page, models_page, settings_page]
        )

        # Every handler that can change what is on disk or in memory rescans
        # the cache afterwards, so My Models never shows a stale list.
        models_inputs = [my_models, sort_models]
        models_outputs = [my_models, my_model_detail, my_models_summary]
        download_button.click(
            download_model, [model_id, hf_token], model_status
        ).then(refresh_my_models, models_inputs, models_outputs)
        download_load_button.click(
            download_and_load_model, [model_id, hf_token], model_status
        ).then(refresh_my_models, models_inputs, models_outputs)
        cached_button.click(load_cached_model, model_id, model_status).then(
            refresh_my_models, models_inputs, models_outputs
        )
        unload_button.click(unload_model, outputs=model_status).then(
            refresh_my_models, models_inputs, models_outputs
        )
        refresh_models_button.click(refresh_my_models, models_inputs, models_outputs)
        sort_models.input(refresh_my_models, models_inputs, models_outputs)
        demo.load(refresh_my_models, models_inputs, models_outputs)
        # .input rather than .change: the refresh above also sets the radio,
        # and a .change listener would rewrite the model ID box on each rescan.
        my_models.input(select_my_model, my_models, [model_id, my_model_detail])
        # A pending removal is about the model that was selected when it was
        # asked for, so changing the selection withdraws it.
        confirm_outputs = [remove_confirm, pending_removal]
        my_models.input(hide_remove_confirm, None, confirm_outputs)
        redownload_button.click(
            redownload_my_model, [my_models, hf_token], model_status
        ).then(refresh_my_models, models_inputs, models_outputs)
        remove_button.click(
            ask_remove_my_model,
            my_models,
            [model_status, remove_confirm, remove_question, pending_removal],
        )
        # The confirm button deletes the model the question named, never the
        # radio's current value: see ask_remove_my_model.
        confirm_remove_button.click(
            remove_my_model, pending_removal, [model_status, *confirm_outputs]
        ).then(refresh_my_models, models_inputs, models_outputs)
        cancel_remove_button.click(hide_remove_confirm, None, confirm_outputs)

        search_outputs = [search_results, search_detail, search_results_state]
        search_button.click(search_models, [search_query, hf_token], search_outputs)
        search_query.submit(search_models, [search_query, hf_token], search_outputs)
        search_results.input(
            select_search_result,
            [search_results, search_results_state],
            [model_id, search_detail],
        )
        enter_sends.change(set_message_box_keys, enter_sends, prompt)

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            color_scale,
        ]
        chat_inputs = [prompt, conversation_state, *settings_inputs]
        # The order every generation handler publishes in; see
        # CHAT_OUTPUT_NAMES.
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            seed,
            send_button,
            stop_button,
            token_detail,
            alternatives,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
            branch_source,
            context_ids_state,
        ]
        undo_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            token_detail,
            alternatives,
            send_button,
            stop_button,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]

        running = [
            send_button.click(chat, chat_inputs, chat_outputs),
            prompt.submit(chat, chat_inputs, chat_outputs),
            retry_button.click(retry_last, chat_inputs, chat_outputs),
            chatbot.retry(retry_message, chat_inputs, chat_outputs),
            chatbot.edit(edit_message, chat_inputs, chat_outputs),
            branch_button.click(
                branch_from,
                [branch_pick, branch_source, metrics_state, *chat_inputs],
                chat_outputs,
            ),
            branch_text_button.click(
                branch_with_text,
                [selected_token, branch_source, metrics_state, branch_text, *chat_inputs],
                chat_outputs,
            ),
        ]

        stop_button.click(
            stop_generation,
            inputs=[conversation_state, metrics_state, context_ids_state],
            outputs=[
                chatbot,
                conversation_state,
                send_button,
                stop_button,
                generation_status,
                branch_source,
            ],
            cancels=running,
        )

        # Undo, Clear and Load all replace or truncate the conversation, so
        # each has to stop the generator first: a surviving generate_reply
        # would write its own snapshot of the in-progress turns back into the
        # chatbot and the state, resurrecting what was just removed. Send,
        # Retry and Edit are exempt because they *are* the generation - they
        # re-enter generate_reply, and they are what everything else cancels.
        # They cannot be made to cancel each other either: Gradio captures a
        # listener's inputs when the request is queued, so the survivor would
        # rebuild the conversation from a snapshot taken before the cancelled
        # run wrote anything. A shared concurrency group has the same flaw - it
        # only delays the stale handler. Each of them refuses outright instead
        # while MANAGER.busy (see busy_state).
        undo_button.click(
            undo_last,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
        )
        chatbot.undo(
            undo_message,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
        )
        clear_button.click(
            clear_chat,
            inputs=color_scale,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                generation_status,
                send_button,
                stop_button,
                token_detail,
                alternatives,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
                forks_state,
                conversation_list,
            ],
            cancels=running,
        )

        # Forking, switching, starting afresh and deleting all replace the
        # conversation, so they cancel a running generation for the same
        # reason Undo does.
        fork_outputs = [
            prompt,
            chatbot,
            conversation_state,
            forks_state,
            conversation_list,
            generation_status,
            send_button,
            stop_button,
            token_strip,
            metrics_state,
            token_detail,
            alternatives,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]
        chatbot.select(remember_message, conversation_state, selected_message)
        fork_button.click(
            fork_conversation,
            [conversation_state, forks_state, selected_message, color_scale],
            fork_outputs,
            cancels=running,
        )
        new_button.click(
            new_conversation,
            [conversation_state, forks_state, color_scale],
            fork_outputs,
            cancels=running,
        )
        # .input rather than .change: the list is also redrawn by the handlers
        # above and the listener below, and a .change listener would switch a
        # second time on each.
        conversation_list.input(
            switch_fork,
            [conversation_list, conversation_state, forks_state, color_scale],
            fork_outputs,
            cancels=running,
        )
        delete_fork_button.click(
            delete_fork,
            [conversation_state, forks_state, color_scale],
            fork_outputs,
            cancels=running,
        )
        # Every other path that changes the conversation - a streaming reply
        # above all - lands here, and the list's model tag and token count
        # follow it.
        conversation_state.change(
            refresh_conversation_list,
            [conversation_state, forks_state],
            conversation_list,
        )

        save_button.click(
            save_conversation,
            [conversation_state, system_prompt],
            [saved_file, generation_status],
        )
        load_upload.upload(
            load_conversation,
            [load_upload, conversation_state, color_scale],
            [
                chatbot,
                conversation_state,
                system_prompt,
                token_strip,
                metrics_state,
                generation_status,
                token_detail,
                alternatives,
                send_button,
                stop_button,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
            ],
            cancels=running,
        )

        score_button.click(
            score_text,
            [score_context, score_input, use_chat_template, color_scale],
            [
                token_strip,
                metrics_state,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                score_status,
                token_detail,
                alternatives,
                context_ids_state,
            ],
        )
        color_scale.change(
            recolor,
            [metrics_state, prompt_metrics_state, color_scale],
            [token_strip, prompt_strip, scale_caption],
        )

        token_strip.select(
            inspect_token,
            inputs=metrics_state,
            outputs=[token_detail, alternatives],
        )
        prompt_strip.select(
            inspect_token,
            inputs=prompt_metrics_state,
            outputs=[token_detail, alternatives],
        )
        # A second listener on each strip keeps the clicked position for the
        # alternatives table. The prompt strip's clicks always clear it: a
        # prompt token cannot be branched, and a stale response position would
        # otherwise pair with the prompt token's rows.
        token_strip.select(remember_selection, metrics_state, selected_token)
        prompt_strip.select(remember_selection, prompt_metrics_state, selected_token)
        alternatives.select(
            choose_alternative,
            [metrics_state, selected_token, branch_source],
            [token_detail, branch_pick],
        )

        # Layer inspection. A third listener on each strip keeps the clicked
        # position, the button does the forward pass, and the slider repaints
        # the attention strip from the stored readout.
        token_strip.select(
            remember_inspect_target("response"), metrics_state, inspect_target
        )
        prompt_strip.select(
            remember_inspect_target("prompt"), prompt_metrics_state, inspect_target
        )
        inspection_outputs = [lens_panel, attention_panel, insight_state, inspect_status]
        inspect_button.click(
            inspect_layers,
            [
                inspect_target,
                metrics_state,
                prompt_metrics_state,
                context_ids_state,
                attention_layer,
            ],
            [lens_panel, attention_panel, attention_layer, insight_state, inspect_status],
        )
        attention_layer.release(
            render_attention, [insight_state, attention_layer], attention_panel
        )
        # Every path that redraws the strips writes the metrics state, so this
        # is where a readout of a token that is no longer on screen goes away.
        metrics_state.change(reset_inspection, insight_state, inspection_outputs)
    return demo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    conductor_port = os.environ.get("CONDUCTOR_PORT")
    build_app().queue(default_concurrency_limit=1).launch(
        inbrowser=conductor_port is None,
        server_port=int(conductor_port) if conductor_port else None,
    )
