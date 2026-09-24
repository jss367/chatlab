"""The Models page's actions: download, load and unload the model in the ID box.

These are the handlers behind **Download and load**, **Load cached**,
**Download** and **Unload**, and the card above them that says what the
chosen model's local files allow. A load is claimed exclusively before its
first card and refused while a reply or another load has the model, because
a load that queued instead would unload the model the reader is looking at
the moment it won the lock. The progress each one streams comes from
:mod:`chatlab.ui.model_streams`; the lists, the search, the fit verdicts and
the badge are in modules of their own.
"""

from __future__ import annotations

import html
import logging
import re
import time

import gradio as gr

from chatlab.model_cache import (
    IMAGE_KIND,
    MLX_KIND,
    TEXT_KIND,
    CacheStatus,
    cache_status,
    format_bytes,
)
from chatlab.model_loading import QUANTIZED_BITS
from chatlab.model_runtime import LOADING
from chatlab.ui import runtime
from chatlab.ui.common import (
    IncompleteSnapshotError,
    failure_card,
    status_card,
)
from chatlab.ui.model_repository import matching_repository
from chatlab.ui.model_streams import (
    describe_cache,
    describe_fetched,
    describe_missing,
    describe_on_disk,
    download_detail,
    stream_download_with_base,
    stream_load,
)

# Every failure on this page used to end as a card in the browser and nothing
# else, so a load that broke reached a reader as one escaped sentence with no
# traceback behind it. The handlers below say what was asked for and what went
# wrong, because this page is where a session's memory trouble starts.
logger = logging.getLogger(__name__)


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
