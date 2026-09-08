"""What every page shares: constants, status cards, small formatting helpers."""

from __future__ import annotations

import html

import gradio as gr


try:
    from huggingface_hub.errors import IncompleteSnapshotError
except ImportError:  # huggingface_hub before 1.x had no such check

    class IncompleteSnapshotError(Exception):
        pass


# Disclose the default model's download size before an explicit download.
DEFAULT_MODEL_DOWNLOAD = "about 15 GB"


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


# The two left panes' widths in pixels. The pane at the far left picks the
# page and holds an icon and a name per page; the conversations pane beside
# it shows with Chat only, and is wide enough for a model ID and a token
# count while leaving the conversation most of the screen.
#
# Wide enough for the longest page name at the tile's small type. With three
# pages there is nothing to be won by hiding those names: the pane would save
# a handful of pixels and cost a reader the only signpost on the screen.
NAV_PANE_WIDTH = 72


CONVERSATION_PANE_WIDTH = 340


CHAT_PAGE, MODELS_PAGE, SETTINGS_PAGE = PAGES = ("Chat", "Models", "Settings")


# Each nav tile shows an icon above the page's own name. The name is the
# label's own text, so it is what a screen reader reads; the stylesheet draws
# the icon in front of it and keeps it out of that reading.
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


def hint(summary: str, detail: str) -> str:
    """A one-line disclosure: the question on screen, the answer on demand.

    The panels carry more explanation than they have room for, and a
    paragraph held permanently open competes for attention with the control
    it is there to explain. ``<details>`` is the browser's own answer to
    that and needs no script: the summary is a line, and the reader who
    wants the paragraph opens it.

    The detail is written as markup rather than markdown, because markdown is
    not processed inside a raw HTML block; ``<strong>`` stands in for the
    asterisks a caption would otherwise use.
    """

    return f'<details class="hint"><summary>{summary}</summary>{detail}</details>'


# What each measurement in the token detail means, in one sentence. The panel
# names them and the README explains them, which leaves a reader holding the
# name with nowhere to ask; these are attached to the names themselves so the
# answer is where the question is.
METRIC_GLOSSARY = {
    "Raw rank": (
        "Where this token sat in the model's unmodified distribution. "
        "Rank 1 was the model's first choice."
    ),
    "Raw model probability": (
        "The probability the model gave this token before temperature, "
        "top-k or top-p touched it."
    ),
    "Actual sampling probability": (
        "The probability it was really drawn with, after temperature, "
        "top-k and top-p."
    ),
    "Surprise": (
        "-log2(probability), in bits. Larger values are less expected."
    ),
    "Distribution entropy": (
        "How undecided the model was across the whole distribution, in bits. "
        "Surprise says how unexpected the choice was; entropy says how open "
        "the question was before it."
    ),
    "Top-1 margin": (
        "The probability gap between the model's first and second choice."
    ),
    "Sampling shift": (
        "log2(sampling probability / raw probability): how far your "
        "temperature, top-k and top-p moved this token from the raw model."
    ),
    "Probability mass above it": (
        "The combined raw probability of every token ranked above this one."
    ),
}


def metric_term(name: str) -> str:
    """A measurement's name with its meaning attached, for the detail panel."""

    meaning = html.escape(METRIC_GLOSSARY[name], quote=True)
    return f'<abbr title="{meaning}">{name}</abbr>'


def status_card(title: str, detail: str, tone: str = "neutral") -> str:
    icon = {"success": "●", "error": "⚠", "working": "◌"}.get(tone, "○")
    heading = f"{icon} {title}"
    if tone == "error":
        # Only the heading is tinted. A detail can carry markdown - file names
        # in backticks, a progress bar - and wrapping it in a tag would stop
        # that from rendering.
        heading = f'<span class="failure-text">{heading}</span>'
    return f"### {heading}\n\n{detail}"


def alarm(title: str, detail: str) -> None:
    """Pop the failure up over the page, wherever the reader is looking.

    A status line is easy to miss: it is one sentence in a column of them,
    and someone watching the transcript never looks at it. Gradio's only
    modal that does not also end the event is the warning, so the app raises
    warnings for failures alone and CSS paints them in the error colors. The
    toast stays until it is closed, because the point is that a reader who
    stepped away still learns why the response stopped.

    The toast writes ``detail`` into the page as markup, so what arrives here
    is already escaped - a runtime that says it could not read ``<pad>``
    would otherwise lose the word. The title is written as text and is the
    application's own wording, so it is passed through as it is.
    """

    gr.Warning(detail, title=title, duration=None)


def failure_status(title: str, detail: str) -> str:
    """A red status line that stays, and the toast that announces it.

    Both take the failure as plain text and escape it once, here.
    """

    safe = html.escape(f"{title}: {detail}")
    alarm(title, html.escape(detail))
    return f'<div class="failure">{safe}</div>'


def failure_card(title: str, detail: str) -> str:
    """A red status card, and the toast that announces it.

    ``detail`` is markdown the caller has already made safe, because a card
    spells out file names in backticks and draws bars out of block
    characters. Both the card and the toast render it as it is given.
    """

    alarm(title, detail)
    return status_card(title, detail, "error")


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


def show_page(page: str):
    """Show the chosen page. The conversations pane comes and goes with Chat."""
    return [
        gr.update(visible=page == CHAT_PAGE),
        *(gr.update(visible=page == name) for name in PAGES),
    ]
