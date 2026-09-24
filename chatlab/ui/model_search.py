"""Discover models: browsing the bundled starters and searching the Hub.

A search keeps every candidate it fetched, not just the rows on screen, so
changing the precision or the **Fits only** filter repaints the table from
what is already in hand rather than asking the Hub again. The table is sorted
in the browser, so a click is traced back to its model by the row's own ID
rather than by where the row happens to sit. Recommended pins ChatLab's
starters above the Hub's answer, and with nothing typed it never asks the
Hub at all, which is why a Hub that cannot be reached still leaves something
to pick.
"""

from __future__ import annotations

import html
import logging
from dataclasses import replace

import gradio as gr
import pandas as pd

from chatlab.device_memory import FITS, TIGHT, UNFIT, Fit, imported_torch
from chatlab.hub_search import (
    DISCOVERY_CANDIDATES,
    SEARCH_IMAGE_PIPELINE_TAGS,
    SEARCH_LIMIT,
    HubModel,
    search_hub_models,
)
from chatlab.model_cache import (
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MLX_KIND,
    TEXT_KIND,
    CacheStatus,
    cache_status,
    format_bytes,
    format_count,
    mlx_available,
)
from chatlab.model_discovery import recommended_models
from chatlab.ui.common import failure_card
from chatlab.ui.memory_fit import fit_word, hub_fit, hub_fits, replacement_profile
from chatlab.ui.model_streams import describe_on_disk
from chatlab.ui.models_page import KIND_NAMES
from chatlab.ui.my_models import ALL_KINDS, refresh_my_models

logger = logging.getLogger(__name__)


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
        "Click a column heading to sort. Selecting a row puts its ID in the model ID box; "
        "**Download and load** fetches it."
    ),
    IMAGE_KIND: (
        "Browse image starters, or choose Popular, Trending, or New for more text-to-image models. "
        "Click a column heading to sort. Selecting a row puts its ID in the model ID box; "
        "**Download and load** fetches it."
    ),
    MLX_KIND: (
        "Browse language models quantized for MLX, which run on Apple silicon at the precision "
        "they were converted to. Selecting a result puts its ID in the model ID box; "
        "**Download and load** fetches it."
    ),
}

SEARCH_HINT = SEARCH_HINTS[TEXT_KIND]


NO_RESULT_SELECTED = "Select a result to see its details."


# The columns a search result can fill, in the order they are shown. A column
# is shown only when some result has something in it: the bundled starters
# carry a download size, and hub results carry popularity and a date, so the
# two browse modes get different tables rather than one table that is half
# dashes. A starter's note goes under its name in the Model cell: the table
# is narrow, and a column of prose would push the verdicts out of view.
SEARCH_COLUMNS = (
    "Model", "Params", "Download size", "Fit", "Downloads", "Likes", "Updated"
)
# The heading row of a table with nothing in it: what a hub search would show.
EMPTY_SEARCH_COLUMNS = ("Model", "Params", "Fit", "Downloads", "Likes", "Updated")
ALWAYS_SHOWN_COLUMNS = ("Model", "Fit")
# How the width is shared between the columns shown, as relative weights:
# the browser would otherwise size the Model column to its longest ID and
# push the last columns out of view. A count or a date never wraps, so the
# weights are also what keeps each on one line; see styles.py.
COLUMN_WEIGHTS = {
    "Model": 38, "Params": 11, "Download size": 15, "Fit": 10,
    "Downloads": 14, "Likes": 10, "Updated": 17,
}

# How a row is tinted by its verdict, matching the .model-list rules in
# styles.py, which tint the My Models list the same way. A model that cannot
# fit is greyed rather than reddened: it is not an error, and the reader may
# be looking at it to find that out. The tight colour is a variable because
# it differs between the light and dark themes.
FIT_STYLES = {
    TIGHT: "color: var(--fit-tight)",
    UNFIT: "color: var(--body-text-color-subdued)",
}

# How the numbers are shown: the hub's own ``7.3B`` and ``281K``, and a byte
# size in the unit it is usually quoted in. The numbers underneath stay
# numbers, so the browser sorts them as such.
CELL_FORMATS = {
    "Params": lambda count: format_count(int(count)),
    "Downloads": lambda count: format_count(int(count)),
    "Likes": lambda count: format_count(int(count)),
    "Download size": lambda count: format_bytes(int(count)),
}


def search_row(result: HubModel, fit: Fit | None = None) -> dict:
    return {
        "Model": (
            f"{result.model_id}\n{result.summary}" if result.summary else result.model_id
        ),
        "Params": result.parameters or None,
        "Download size": result.download_bytes or None,
        "Fit": fit_word(fit),
        "Downloads": result.downloads,
        "Likes": result.likes,
        "Updated": result.last_modified,
    }


def column_widths(shown: list[str]) -> list[str]:
    """Each shown column's share of the table, as percentages summing to 100."""

    total = sum(COLUMN_WEIGHTS[column] for column in shown)
    return [f"{100 * COLUMN_WEIGHTS[column] / total:.0f}%" for column in shown]


def search_table(results: list[HubModel], fits: dict[str, Fit] | None = None):
    """The results as a table, one row each, which the browser sorts by column.

    Returned as a component update: the table itself, and the widths of the
    columns it has. The table is a pandas Styler: the numbers are kept as
    numbers so a sort by downloads or size is numeric, and the Styler says
    how each is displayed and which rows are tinted. The model ID is always
    the first column, which is how a click on a sorted table is traced back
    to its model; see :func:`picked_model`.
    """

    fits = fits or {}
    rows = [search_row(result, fits.get(result.model_id)) for result in results]
    if not rows:
        shown = list(EMPTY_SEARCH_COLUMNS)
        return gr.update(
            value=pd.DataFrame(columns=shown).style, column_widths=column_widths(shown)
        )
    shown = [
        column for column in SEARCH_COLUMNS
        if column in ALWAYS_SHOWN_COLUMNS
        or any(row[column] not in (None, "") for row in rows)
    ]
    # Object columns, so a missing number is None rather than NaN and reaches
    # the browser as null.
    frame = pd.DataFrame(rows, columns=shown).astype(object)
    frame = frame.where(frame.notna(), None)
    states = [
        fits[result.model_id].state if result.model_id in fits else ""
        for result in results
    ]
    table = (
        frame.style
        .format({name: fmt for name, fmt in CELL_FORMATS.items() if name in shown}, na_rep="—")
        .apply(lambda row: [FIT_STYLES.get(states[row.name], "")] * len(row), axis=1)
    )
    return gr.update(value=table, column_widths=column_widths(shown))


def picked_model(event: gr.SelectData | None) -> str | None:
    """The model ID of the row a click landed on, or None for no row.

    Read from the row's own cells rather than its position: the table sorts in
    the browser, so where a row is says nothing about which model it is. The
    first cell is the ID, with a starter's note under it; see search_row.
    """

    row = getattr(event, "row_value", None)
    return str(row[0]).split("\n", 1)[0] if row and row[0] else None


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
    kind: str | None = ALL_KINDS,
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
        return (gr.skip(),) * 7
    return (
        *refresh_my_models(selected, order, precision, kind, model_id),
        *refresh_search_results(result, results or {}, precision, fits_only),
        True,
    )


def search_models(
    query: str | None, hf_token: str, precision: str | None = None, kind: str = TEXT_KIND,
    order: str = "Popular", fits_only: bool = False,
):
    """Browse or search; retain candidates so memory filtering needs no network.

    ``order`` sorts a search of the whole Hub rather than narrowing what is
    searched: an obscure repository comes back under Popular, Trending and New
    alike, as long as the query matches its ID. The one place the sort does
    decide what is seen is a query with more matches than SEARCH_SCAN_LIMIT,
    which is where the paging stops; each sort reaches that limit over a
    different part of the answer, so a narrower query finds a particular
    repository where a different sort may not. Recommended is the same search
    with ChatLab's starters pinned above it, and is the one view an empty query
    can answer offline.

    A search drops the previous selection along with the previous results.
    """

    cleared = search_table([])
    cleaned = (query or "").strip()
    # Recommended is a sort, not a filter. With something typed it puts the
    # matching starters first and fills the rest from the Hub, so the "Search
    # Hugging Face" box searches Hugging Face in every view. Only an empty box
    # stays offline, and that is the view the page loads with.
    starters = recommended_models(cleaned, kind) if order == "Recommended" else []
    searched_hub = order != "Recommended" or bool(cleaned)
    unreachable = None
    found = []
    if searched_hub:
        try:
            found = search_hub_models(
                cleaned, hf_token, kind=kind,
                order="Popular" if order == "Recommended" else order,
                limit=DISCOVERY_CANDIDATES,
            )
        except Exception as error:
            # Starters already in hand are worth showing without the Hub. With
            # none there is nothing left to show, so the failure is the answer.
            # With the stack, because this catches everything: a Hub that is
            # simply unreachable, and a mistake in the search itself. The card
            # already carries str(error), so a line without the traceback
            # would only say again what the reader can already see.
            logger.warning("Hub search for %r failed", cleaned, exc_info=True)
            if not starters:
                hint = (
                    "Clear the search to see offline starters, or retry."
                    if order == "Recommended"
                    else "Choose Recommended for offline starters, or retry."
                )
                return (
                    cleared,
                    failure_card("Search failed", f"{html.escape(str(error))} {hint}"),
                    {},
                    None,
                )
            unreachable = error
    # Starters first, and a starter the Hub also returned is listed once,
    # wearing both sides of what is known about it; see merged_starter.
    from_hub = {result.model_id: result for result in found}
    named = {starter.model_id for starter in starters}
    results = [
        merged_starter(starter, from_hub.get(starter.model_id)) for starter in starters
    ] + [result for result in found if result.model_id not in named]
    if not results:
        described = {IMAGE_KIND: "text-to-image models", MLX_KIND: "MLX models"}.get(
            kind, "language models"
        )
        message = (
            f"No {described} matched `{html.escape(cleaned)}`."
            if cleaned else f"No {described} found in this browse window."
        )
        return cleared, message, {}, None
    state = {result.model_id: result for result in results}
    table, detail, _ = refresh_search_results(None, state, precision, fits_only)
    ordering = search_note(order, bool(starters), searched_hub, unreachable)
    return table, f"{ordering} {detail}", state, None


def merged_starter(starter: HubModel, live: HubModel | None) -> HubModel:
    """A bundled starter wearing what the Hub’s own answer adds to it.

    The two describe one repository from different sides. The catalog has the
    curated note and the download estimate, which a search result never
    carries; the search has the downloads, the likes and the date, which the
    catalog cannot keep current. Keeping one and dropping the other would take
    columns off the row that the same search shows for every other result, so
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
        live, **stale, summary=starter.summary, download_bytes=starter.download_bytes
    )


def search_note(
    order: str, matched_starters: bool, searched_hub: bool, unreachable: Exception | None
) -> str:
    """The line above the results: where they came from, and how they are sorted.

    Popular, Trending and New sort a search of the whole Hub, so each says
    only what the sort is. Recommended also reaches the Hub once there is a
    query, and says which part of the list is which.
    """

    if order != "Recommended":
        return {
            "Popular": "Most downloaded first.",
            "Trending": "Trending on Hugging Face.",
            "New": "Newest repositories first (not latest updates).",
        }[order]
    if not searched_hub:
        return "Curated starters, available to browse offline."
    if unreachable is not None:
        return (
            "Starters only: Hugging Face could not be reached "
            f"({html.escape(str(unreachable))})."
        )
    if not matched_starters:
        return "No starters matched; showing Hugging Face results, most downloaded first."
    return "Starters first, then Hugging Face, most downloaded first."


def select_search_result(results: dict, precision: str | None, event: gr.SelectData):
    """Put the clicked result in the ID box, describe it, and remember which it was.

    The selection is kept apart from the table because the table's own
    highlight is a cell, and a sort moves it.
    """

    selected = picked_model(event)
    result = results.get(selected) if selected else None
    if result is None:
        return gr.skip(), NO_RESULT_SELECTED, None
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
        result.model_id,
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
    """Recompute fit filtering from retained candidates, clearing hidden selections.

    Returns the table, the detail beside it, and the selection as it stands
    after the filter: the model that was selected, or None if it is now hidden.
    """

    if not results:
        return gr.skip(), gr.skip(), gr.skip()
    fits = hub_fits(list(results.values()), precision, results_kind(results))
    visible = [
        result for model_id, result in results.items()
        if not fits_only or (fits.get(model_id) and fits[model_id].state == FITS)
    ][:SEARCH_LIMIT]
    visible_ids = {result.model_id for result in visible}
    selected = selected if selected in visible_ids else None
    table = search_table(visible, fits)
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
    return table, detail, selected
