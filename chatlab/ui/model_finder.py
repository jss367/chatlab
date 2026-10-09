"""The Find view on the Models page: the results list, the detail pane, and the row script.

The list and the pane are drawn here as HTML from what the handlers in
ui.models_page have already worked out: the results, their fit verdicts, what
is on disk, and what Hugging Face said when the model was checked. Nothing
here reads the cache, the device or the network.

A row is a button. The script below writes a press into a hidden bridge
textbox, as the My Models rows do (see ui.model_rows), so the server learns
which model was picked from the row's own ID rather than its position.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

from chatlab.device_memory import FITS, TIGHT, UNFIT, Fit, format_memory
from chatlab.hub_search import Exclusion, HubModel
from chatlab.model_cache import IMAGE_KIND, MLX_KIND, TEXT_KIND, format_bytes, format_count

RESULTS_ID = "model-search-results"
PANE_ID = "model-pane"
PICK_BRIDGE_ID = "model-search-pick"

# The sorts a search offers, as the dropdown words them and as the hub names
# them. Picks have no sort: they are a short list in the order chosen.
SEARCH_ORDERS = (("Most downloaded", "Popular"), ("Trending", "Trending"), ("Newest", "New"))
DEFAULT_SEARCH_ORDER = "Popular"
ORDER_NOTES = {
    "Popular": "most downloaded first",
    "Trending": "trending on Hugging Face",
    "New": "newest repositories first",
}

# The two kinds a reader chooses between. MLX is not one of them: an MLX
# conversion is a text model packed for Apple silicon, so it is listed among
# the text models and marked as MLX.
SEARCH_KINDS = (("Text", TEXT_KIND), ("Image", IMAGE_KIND))

PICKS_HEADING = "ChatLab picks"
PICKS_NOTE = "Chosen to work well in ChatLab · no internet needed"
HUB_HEADING = "Hugging Face results"
PICKS_FOOTER = "Type to search all of Hugging Face. Paste a model ID and press Enter to open it."

# The pipeline tags of a text model that also reads pictures.
VISION_TAGS = ("image-text-to-text", "any-to-any")

FIT_BADGES = {FITS: ("Fits", "fits"), TIGHT: ("Tight", "tight"), UNFIT: ("Too large", "unfit")}


# Compared by identity: Gradio hashes a State's value to tell whether it
# changed, and the results dict cannot be hashed. Every search makes a new one.
@dataclass(frozen=True, eq=False)
class Listing:
    """What the results list holds: the rows, where they came from, and what was left out.

    ``results`` is in the order the rows are drawn. A listing is kept in a
    State so the fit filter and a precision change can redraw the list
    without searching again.
    """

    results: dict[str, HubModel]
    heading: str = PICKS_HEADING
    note: str = PICKS_NOTE
    exclusions: tuple[Exclusion, ...] = ()
    scanned: int = 0
    stopped_early: bool = False
    query: str = ""
    # Said above the rows when the hub could not be asked, or found nothing.
    message: str = ""
    # The libraries the search asked the hub for, for the footer.
    libraries: tuple[str, ...] = ()

    @property
    def picks(self) -> bool:
        return self.heading == PICKS_HEADING


def model_kind(result: HubModel) -> str:
    """The kind a result loads as: an image pipeline, an MLX conversion, or a text model."""

    if result.kind == IMAGE_KIND or result.pipeline_tag == "text-to-image":
        return IMAGE_KIND
    return result.kind


def split_id(model_id: str) -> str:
    """``org/name`` with the organization muted, as a row and the pane draw it."""

    org, slash, name = model_id.partition("/")
    if not slash:
        return f"<span class='model-name'>{html.escape(model_id)}</span>"
    return (
        f"<span class='model-org'>{html.escape(org)}/</span>"
        f"<span class='model-name'>{html.escape(name)}</span>"
    )


def badge(text: str, tone: str = "") -> str:
    return f"<span class='model-badge-chip {tone}'>{html.escape(text)}</span>"


def fit_badge(fit: Fit | None, precision_words: str = "") -> str:
    if fit is None or fit.state not in FIT_BADGES:
        return ""
    text, tone = FIT_BADGES[fit.state]
    return badge(f"{text} {precision_words}".strip(), tone)


def result_row(
    result: HubModel, fit: Fit | None, *, selected: bool, downloaded: bool, picks: bool
) -> str:
    """One row of the results list, as a button that names its model."""

    badges = []
    if result.parameters:
        badges.append(badge(f"{format_count(result.parameters)} params"))
    if result.download_bytes:
        badges.append(badge(f"{format_bytes(result.download_bytes)} download"))
    badges.append(fit_badge(fit))
    if model_kind(result) == MLX_KIND:
        badges.append(badge("MLX", "mlx"))
    if result.gated or result.base_gated:
        badges.append(badge("Gated", "gated"))
    if result.pick and not picks:
        badges.append(badge("ChatLab pick", "pick"))
    if downloaded:
        badges.append(badge("Downloaded", "downloaded"))
    note = result.summary or adapter_summary(result)
    summary = f"<span class='model-result-summary'>{html.escape(note)}</span>" if note else ""
    counts = []
    if result.downloads is not None:
        counts.append(f"{format_count(result.downloads)} downloads")
    if result.likes is not None:
        counts.append(f"{format_count(result.likes)} likes")
    numbers = (
        f"<span class='model-result-numbers'>{'<br>'.join(counts)}</span>" if counts else ""
    )
    classes = ["model-result"]
    if selected:
        classes.append("selected")
    if fit is not None and fit.state == UNFIT:
        classes.append("unfit")
    name = html.escape(result.model_id, quote=True)
    return (
        f"<button type='button' class='{' '.join(classes)}' data-model='{name}' "
        f"aria-pressed='{'true' if selected else 'false'}'>"
        f"<span class='model-result-main'><span class='model-result-id'>{split_id(result.model_id)}</span>"
        f"{summary}<span class='model-result-badges'>{''.join(badges)}</span></span>"
        f"{numbers}</button>"
    )


def results_list(
    listing: Listing | None,
    fits: dict[str, Fit],
    *,
    selected: str | None,
    downloaded: set[str],
    shown: list[HubModel],
    hidden_by_fit: int,
    fits_only: bool,
) -> str:
    """The whole results card: heading, rows, and what the rows leave out."""

    if listing is None:
        return f"<div id='{RESULTS_ID}-body' class='model-results'></div>"
    counts = ""
    if not listing.picks:
        total = len(listing.results)
        counts = f"{len(shown)} shown"
        if total > len(shown) and not hidden_by_fit:
            counts += f" of {total}"
        counts += f" · {listing.note}"
    note = counts or listing.note
    rows = "".join(
        result_row(
            result,
            fits.get(result.model_id),
            selected=result.model_id == selected,
            downloaded=result.model_id in downloaded,
            picks=listing.picks,
        )
        for result in shown
    )
    message = f"<p class='model-results-message'>{listing.message}</p>" if listing.message else ""
    if not shown and not listing.message:
        message = (
            "<p class='model-results-message'>No results fit this computer. "
            "Turn off <b>Fits this computer</b>, or choose a lower precision.</p>"
            if fits_only and listing.results
            else "<p class='model-results-message'>Nothing to show.</p>"
        )
    return (
        f"<div class='model-results'>"
        f"<div class='model-results-head'><span class='model-results-title'>{html.escape(listing.heading)}</span>"
        f"<span class='model-results-note'>{html.escape(note)}</span></div>"
        f"{message}<div class='model-results-rows'>{rows}</div>"
        f"{results_footer(listing, hidden_by_fit)}</div>"
    )


def results_footer(listing: Listing, hidden_by_fit: int) -> str:
    """What the list does not show and why, or how to search when it shows picks.

    The counts are of the results the search read, never of the hub as a
    whole: the reading stops at a limit, and two of the reasons are missing
    information rather than a known incompatibility.
    """

    if listing.picks:
        return f"<div class='model-results-foot'>{html.escape(PICKS_FOOTER)}</div>"
    lines = []
    if hidden_by_fit:
        lines.append(
            f"{hidden_by_fit} more {'is' if hidden_by_fit == 1 else 'are'} hidden by "
            "<b>Fits this computer</b>."
        )
    searched = " and ".join(listing.libraries) or "Transformers"
    reach = (
        f"ChatLab checked the first {listing.scanned} matches; a narrower search reaches further."
        if listing.stopped_early
        else f"ChatLab checked {listing.scanned} matches."
    )
    detail = (
        f"<p>ChatLab asks Hugging Face for {html.escape(searched)} repositories only. "
        f"{reach} If you know a model's ID, paste it and press Enter.</p>"
    )
    if listing.exclusions:
        reasons = "".join(
            f"<li>{html.escape(exclusion.reason, quote=False)} <span class='model-results-count'>"
            f"({exclusion.count}, e.g. {html.escape(exclusion.example, quote=False)})</span></li>"
            for exclusion in sorted(listing.exclusions, key=lambda item: -item.count)
        )
        lines.append(
            "<details><summary>Some checked results were excluded · See reasons</summary>"
            f"<ul>{reasons}</ul>{detail}</details>"
        )
    else:
        lines.append(detail)
    return f"<div class='model-results-foot'>{''.join(lines)}</div>"


# -- the detail pane ---------------------------------------------------------

PANE_EMPTY = (
    "<div class='pane-empty'>Select a model to see what it needs and what you can do with it.</div>"
)


def adapter_summary(result: HubModel) -> str:
    if not result.adapter:
        return ""
    return f"LoRA adapter for {result.base_model}" if result.base_model else "LoRA adapter"


def pane_head(
    model_id: str,
    result: HubModel | None,
    *,
    gated: bool,
    loaded: bool,
) -> str:
    """The pane's top: the model's name, where it lives, and what stands in the way."""

    if not model_id:
        return PANE_EMPTY
    name = html.escape(model_id, quote=True)
    links = [f"<a href='https://huggingface.co/{name}' target='_blank' rel='noopener'>View on Hugging Face ↗</a>"]
    if result is not None:
        if result.license:
            links.append(f"{html.escape(result.license)} license")
        if result.last_modified:
            links.append(f"updated {html.escape(result.last_modified)}")
        if result.downloads is not None:
            links.append(f"{format_count(result.downloads)} downloads last month")
        if result.pick:
            links.append("ChatLab pick")
    status = badge("Loaded", "loaded") if loaded else ""
    note = (result.summary or adapter_summary(result)) if result is not None else ""
    summary = f"<p class='pane-summary'>{html.escape(note)}</p>" if note else ""
    gate = (
        "<div class='pane-gated'><b>Gated.</b> Accept its terms on Hugging Face, then paste "
        "a token that has access under <b>Access token</b>.</div>"
        if gated else ""
    )
    return (
        f"<div class='pane-head'><div class='pane-id'><span>{split_id(model_id)}</span>{status}</div>"
        f"<div class='pane-links'>{' · '.join(links)}</div>{summary}{gate}</div>"
    )


def capabilities(kind: str, bits: int | None, vision: bool) -> list[tuple[bool, str]]:
    """What ChatLab can do with a model of ``kind`` loaded at ``bits``, and what it cannot.

    Steering, probes and activation patching read the residual stream
    through forward hooks, which an MLX checkpoint has nowhere to put. The
    Jacobian lens refuses quantized Transformers weights because it was
    fitted on full ones; an MLX conversion only exists packed, so it is read
    as it is.
    """

    if kind == IMAGE_KIND:
        return [(True, "Draws pictures from a prompt on the Images page")]
    torch = kind != MLX_KIND
    full = kind == MLX_KIND or bits is None
    rows = [
        (True, "Chat, token probabilities, logit lens and attention"),
        (torch, "Steering and probes" if torch else "Steering and probes (need a Transformers model)"),
        (torch, "Activation patching" if torch else "Activation patching (needs a Transformers model)"),
        (full, "Jacobian lens, where one is fitted for this model" if full else "Jacobian lens (full precision only)"),
    ]
    if vision:
        rows.append((torch, "Reading pictures you attach" if torch else "Reading pictures (MLX runs text only)"))
    return rows


def precision_words(kind: str, bits: int | None) -> str:
    if kind == IMAGE_KIND:
        return ""
    if kind == MLX_KIND:
        return f"as packed ({bits}-bit)" if bits else "as packed"
    return f"at {bits}-bit" if bits else "at full precision"


def fact(label: str, value: str, note: str, tone: str = "") -> str:
    return (
        f"<div class='pane-fact {tone}'><div class='pane-fact-label'>{html.escape(label)}</div>"
        f"<div class='pane-fact-value'>{value}</div><div class='pane-fact-note'>{note}</div></div>"
    )


def download_fact(
    kind: str, size: int | None, checking: bool, bits: int | None, on_disk: int | None = None,
    checked: bool = False,
) -> str:
    if on_disk and not size:
        return fact("On disk", html.escape(format_bytes(on_disk)), "Already downloaded.")
    if size:
        value = f"{format_bytes(size)}"
    elif checking:
        value = "Checking…"
    else:
        value = "Unknown"
    if kind == MLX_KIND:
        note = (
            f"Packed to {bits} bits when it was converted." if bits
            else "Packed when it was converted; the precision choice does not apply."
        )
    elif kind == IMAGE_KIND:
        note = "Image pipelines load at full precision."
    else:
        note = "The same at every precision. Precision changes memory, not the download."
    if not size and checked:
        note = "Hugging Face did not list every file's size. " + note
    elif not size and not checking:
        note = "Check it on Hugging Face to see the size. " + note
    return fact("Download", html.escape(value), html.escape(note))


def memory_fact(fit: Fit | None, loaded_bytes: int | None = None) -> str:
    if fit is None and loaded_bytes:
        return fact(
            "Memory now", f"{html.escape(format_memory(loaded_bytes))} · loaded",
            "Held in memory now. Change the precision and load it again to compare.", "ok",
        )
    if fit is None or not fit.known or fit.estimated is None:
        note = fit.note if fit is not None and fit.note else "Not enough is known to estimate it."
        return fact("Memory when loaded", "Unknown", html.escape(note))
    words = {FITS: "fits", TIGHT: "tight", UNFIT: "too large"}
    tone = {FITS: "ok", TIGHT: "warn", UNFIT: "bad"}.get(fit.state, "")
    value = f"≈ {html.escape(format_memory(fit.estimated))} · {words.get(fit.state, '')}"
    return fact("Memory when loaded", value, html.escape(fit.note), tone)


def version_row(version: HubModel, fit: Fit | None, original: bool) -> str:
    """One entry under Other versions, which opens that model in the pane."""

    if original:
        tag, note = badge("Transformers"), "The full model: steering, probes and patching work."
    else:
        tag = badge("MLX", "mlx")
        note = "Runs packed on Apple silicon · no steering, probes or patching"
    facts = []
    if version.parameters:
        facts.append(f"{format_count(version.parameters)} params")
    if fit is not None and fit.state in FIT_BADGES:
        facts.append(FIT_BADGES[fit.state][0].lower())
    if version.downloads is not None:
        facts.append(f"{format_count(version.downloads)} downloads")
    facts.append(note)
    name = html.escape(version.model_id, quote=True)
    return (
        f"<button type='button' class='model-version' data-model='{name}'>{tag}"
        f"<span class='model-version-main'><span class='model-version-id'>{html.escape(version.model_id)}</span>"
        f"<span class='model-version-note'>{html.escape(' · '.join(facts))}</span></span></button>"
    )


def pane_body(
    *,
    kind: str,
    bits: int | None,
    vision: bool,
    download_bytes: int | None,
    checking: bool,
    fit: Fit | None,
    versions: list[tuple[HubModel, Fit | None, bool]],
    notes: list[str],
    on_disk: str,
    loaded_bytes: int | None = None,
    on_disk_bytes: int | None = None,
    blocked: bool = False,
    checked: bool = False,
) -> str:
    """The pane's middle: download against memory, what works, and the other versions.

    ``blocked`` is a model ChatLab cannot load at all, for which no list of
    what works would be true; the check's own words say why, below.
    """

    facts = (
        f"<div class='pane-facts'>{download_fact(kind, download_bytes, checking, bits, on_disk_bytes, checked)}"
        f"{memory_fact(fit, loaded_bytes)}</div>"
    )
    words = precision_words(kind, bits)
    heading = f"What you can do with it {words}".strip()
    rows = "".join(
        f"<li class='{'yes' if ok else 'no'}'>{html.escape(text)}</li>"
        for ok, text in capabilities(kind, bits, vision)
    )
    works = (
        "<p class='pane-note'>ChatLab can't load this model, so none of its tools apply. "
        "The check below says why.</p>"
        if blocked else f"<ul class='pane-capabilities'>{rows}</ul>"
    )
    sections = [
        facts,
        f"<div class='pane-section'><div class='pane-label'>{html.escape(heading)}</div>{works}</div>",
    ]
    if versions:
        items = "".join(version_row(version, fit, original) for version, fit, original in versions)
        sections.append(
            f"<div class='pane-section'><div class='pane-label'>Other versions</div>"
            f"<div class='pane-versions'>{items}</div></div>"
        )
    if on_disk:
        sections.append(
            f"<div class='pane-section'><div class='pane-label'>On this computer</div>"
            f"<p class='pane-note'>{on_disk}</p></div>"
        )
    if notes:
        sections.append(
            "<div class='pane-section'><div class='pane-label'>Hugging Face check</div>"
            + "".join(f"<p class='pane-note'>{note}</p>" for note in notes)
            + "</div>"
        )
    return f"<div class='pane-body'>{''.join(sections)}</div>"


_SCRIPT = r"""
() => {
  if (window.chatlabModelFinder) return;
  window.chatlabModelFinder = true;
  const send = value => {
    const input = document.querySelector(`#${__BRIDGE__} textarea, #${__BRIDGE__} input`);
    if (!input) return;
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
  };
  document.addEventListener('click', event => {
    const button = event.target.closest?.(`#${__RESULTS__} .model-result, #${__PANE__} .model-version`);
    if (!button) return;
    event.preventDefault();
    // Marked at once, so the press is seen before the server answers.
    for (const row of document.querySelectorAll(`#${__RESULTS__} .model-result.selected`)) {
      row.classList.remove('selected');
      row.setAttribute('aria-pressed', 'false');
    }
    if (button.classList.contains('model-result')) {
      button.classList.add('selected');
      button.setAttribute('aria-pressed', 'true');
    }
    send(JSON.stringify({model: button.dataset.model, nonce: Date.now()}));
  });
}
"""

MODEL_FINDER_JS = (
    _SCRIPT.replace("__BRIDGE__", json.dumps(PICK_BRIDGE_ID))
    .replace("__RESULTS__", json.dumps(RESULTS_ID))
    .replace("__PANE__", json.dumps(PANE_ID))
)
