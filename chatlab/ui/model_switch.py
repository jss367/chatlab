"""The loaded-model badge, the Chat page's model switcher, and the way to Models.

The badge names the model in memory, or the one on its way in with a bar for
how far it has come, on every page that uses a model; each open tab asks
again on a timer, because a handler's update reaches only the tab that ran
it. The switcher beside it swaps the model for another one already on disk
without a trip to the Models page, and offers only what a load would take
right now, since a refused switch would cost the reader the model they were
talking to. The buttons elsewhere that open the Models page on a chosen model
or kind are here as well, because they reset or repaint its lists as they go.
"""

from __future__ import annotations

import html
import logging
import time
from typing import NamedTuple

import gradio as gr

from chatlab import settings
from chatlab.device_memory import FITS
from chatlab.model_cache import (
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MLX_KIND,
    TEXT_KIND,
    list_cached_models,
    sort_cached_models,
    validate_model_id,
)
from chatlab.progress_bars import LoadSnapshot
from chatlab.ui import runtime
from chatlab.ui.common import (
    DEFAULT_MODEL_DOWNLOAD,
    MODELS_PAGE,
    Card,
    alarm,
    failure_card,
    show_page,
    status_card,
)
from chatlab.ui.memory_fit import cached_fits
from chatlab.ui.model_search import NO_RESULT_SELECTED, search_models
from chatlab.ui.models_page import KIND_NAMES, load_cached_model, occupied_reason
from chatlab.ui.my_models import (
    clear_my_model_selection,
    hide_remove_confirm,
    refresh_my_models,
)

logger = logging.getLogger(__name__)


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
    model_id: str | None,
    query: str | None,
    hf_token: str,
    _search_precision: str | None,
    _search_kind: str | None,
    search_order: str,
    fits_only: bool,
):
    """Open Models with both lists scoped to image models.

    The **Choose an image model** button on the Images page asks a narrower
    question than the Models tile does, and a cache of seventeen models in
    which two can draw does not answer it: the kind is a word mid-row, and
    the row it is missing from is the majority. So the filter and the search
    are set to image here, and both lists are repainted from that rather
    than left to a listener - the controls are written by this handler, and
    a control a handler writes reports no change of its own.

    The two kind choices the page is already showing are taken and ignored,
    which keeps the wiring the same input lists the two lists take.
    """

    return (
        *go_to_models(),
        gr.update(value=IMAGE_KIND),
        *refresh_my_models(selected, order, precision, IMAGE_KIND, model_id),
        gr.update(value=IMAGE_KIND),
        *search_models(query, hf_token, precision, IMAGE_KIND, search_order, fits_only),
    )


def select_model_to_load(model_id, title="Model selected", note=""):
    """Put ``model_id`` in the ID box and open Models; loading stays a click away.

    Update the ID, both model selections and removal confirmation together.
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
        None,
        NO_RESULT_SELECTED,
        status_card(
            title,
            f"`{model_id}` is selected. Choose **Load cached** to use local files, "
            "or **Download and load** to fetch the model and load it." + note,
        ),
        *hide_remove_confirm(),
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
