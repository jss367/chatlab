"""The token panel: the strips, the token detail, and branching from a token."""

from __future__ import annotations

import html
import threading

import gradio as gr

from model_runtime import (
    PROMPT_SCORE_LIMIT,
)
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
    UNSCORED_BEYOND_LIMIT,
    category_for,
)
from ui import runtime
from ui.common import (
    NO_TOKEN_SELECTED,
    RESPONSE_STRIP_LABEL,
    metric_term,
)


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


def current_metrics_generation() -> int:
    """The stamp the strips on screen were drawn with.

    Read through a call rather than imported as a name: the counter is
    rebound by every mint, and a module that imported the number itself
    would keep comparing against the value it saw at import time.
    """

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

    # Every measurement here has a name that means nothing on first reading,
    # so each name carries its own sentence: hovering it, or reaching it with
    # a screen reader, says what the number is.
    summary = (
        f"### {where} {metric['position']}: `{token_repr}`\n\n"
        f"- **{metric_term('Raw rank')}:** {metric['raw_rank']:,}\n"
        f"- **{metric_term('Raw model probability')}:** {metric['raw_probability']:.5%}\n"
        f"- **{metric_term('Actual sampling probability')}:** {metric['sampling_probability']:.5%}\n"
        f"- **{metric_term('Surprise')}:** {metric['surprise_bits']:.2f} bits\n"
        f"- **{metric_term('Distribution entropy')}:** {metric['entropy_bits']:.2f} bits\n"
        f"- **{metric_term('Top-1 margin')}:** {metric['top1_margin']:.2%} between the model's first and second choice\n"
        f"- **{metric_term('Sampling shift')}:** {metric['sampling_shift_bits']:+.2f} bits versus the raw model\n"
        f"- **{metric_term('Probability mass above it')}:** {metric['probability_mass_above']:.2%}\n"
        f"- **Token ID:** {metric['token_id']:,}"
    )
    rows = [
        [candidate["token_id"], repr(candidate["text"]), candidate["probability"]]
        for candidate in metric["top_candidates"]
    ]
    return summary, rows


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
    if branch_source != (generation, runtime.MANAGER.load_id):
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
