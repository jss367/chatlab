"""The token panel: the conversation in tokens, the token detail, and branching."""

from __future__ import annotations

import html
import threading
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import gradio as gr
from gradio.context import LocalContext

from conversation import (
    turn_tokens,
)
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
    ROLE_HEADINGS,
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


def transcript_entries(
    turns: list[dict] | None, scale_name: str
) -> tuple[list[tuple[str, str | None]], list[tuple[int, int | None]]]:
    """The whole conversation as spans, plus what each span stands for.

    This is the token view of the chat: the same messages the chatbot draws,
    written out token by token and painted by whichever scale is chosen. A
    reply that carries its measurements contributes one span per token; every
    other span - a role heading, a message the reader typed, a reply whose
    text was rewritten by hand or read back from a saved file - is one
    unpainted span of plain text.

    The second list runs parallel to the first and says, for each span, which
    turn it belongs to and which of that turn's tokens it is. ``None`` in the
    token place means the span is not a measured token, so clicking it has
    nothing to report and nothing to branch. It is built here rather than
    kept in a state because it is a function of the turns alone, and the
    handlers that need it are handed those turns anyway - the same
    arrangement ``display_messages`` and ``locate`` already use for the
    chatbot's own indices.
    """

    scale = resolve_scale(scale_name)
    spans: list[tuple[str, str | None]] = []
    index: list[tuple[int, int | None]] = []

    for position, turn in enumerate(turns or []):
        spans.append((ROLE_HEADINGS.get(turn["role"], "\n\n"), None))
        index.append((position, None))
        tokens = turn_tokens(turn)
        if tokens:
            # The tokens are the raw stream, reasoning markers included, so
            # they already cover everything the turn holds. The chatbot folds
            # reasoning into its own bubble; here it is simply where the model
            # put it.
            for token_index, metric in enumerate(tokens):
                spans.append(
                    (metric["display_text"], category_for(metric, scale.name))
                )
                index.append((position, token_index))
            continue
        reasoning = turn.get("reasoning") or ""
        content = turn.get("content") or ""
        if reasoning:
            spans.append((f"{reasoning}\n", None))
            index.append((position, None))
        if content or not reasoning:
            spans.append((content, None))
            index.append((position, None))

    return spans, index


# What the token view says when there is nothing to say. An empty
# HighlightedText draws its color scale as a bare gradient bar, which reads as
# a broken chart rather than an empty conversation; one span of plain text
# says what is actually true. It is deliberately not in transcript_entries():
# it belongs to no turn, so a click on it finds no row in the index and is
# ignored, which is what should happen.
EMPTY_TRANSCRIPT = [("No messages yet. Send one to see it token by token.", None)]


def transcript_value(turns: list[dict] | None, scale_name: str):
    spans = transcript_entries(turns, scale_name)[0]
    if spans and all(label is None for _, label in spans):
        # Gradio's all-uncolored renderer has no select handlers. An empty
        # categorized span selects its clickable renderer without coloring
        # any message or adding visible text. It belongs to no turn and is
        # intentionally absent from transcript_entries()'s selection map.
        category = next(iter(resolve_scale(scale_name).color_map))
        return [*spans, ("", category)]
    return spans or EMPTY_TRANSCRIPT


def transcript_update(turns: list[dict] | None, scale_name: str):
    """Repaint the conversation's token view, legend and all."""

    scale = resolve_scale(scale_name)
    return gr.update(
        value=transcript_value(turns, scale.name), color_map=scale.color_map
    )


def show_token_view(on, turns: list[dict] | None, scale_name: str, *, conversation_id=None):
    """Swap the chatbot for the conversation's token view, or back.

    The token view is rebuilt from the conversation on the way in. Hidden
    streams skip that rendering work; once shown, each subsequent frame reads
    this session's live toggle and resumes painting the tokens as they arrive.
    """

    _session_panel().token_view = bool(on)
    if not on:
        return gr.update(visible=True), gr.update(visible=False)
    request = LocalContext.request.get(None)
    blocks = LocalContext.blocks.get(None)
    if conversation_id is not None and request is not None and request.session_hash:
        # Read the state that Gradio has actually published, not the turns
        # captured when this toggle was queued. A final hidden frame can land
        # in between, with no later frame left to repair a partial redraw.
        turns = blocks.state_holder[request.session_hash][conversation_id]
    scale = resolve_scale(scale_name)
    return (
        gr.update(visible=False),
        gr.update(
            visible=True,
            value=transcript_value(turns, scale.name),
            color_map=scale.color_map,
        ),
    )


def transcript_pick(
    turns: list[dict] | None, event: gr.SelectData
) -> tuple[int, int | None] | None:
    """The turn and token a click in the token view landed on."""

    try:
        return transcript_entries(turns, DEFAULT_COLOR_SCALE)[1][event_index(event)]
    except (IndexError, TypeError, ValueError):
        return None


def selected_metric(
    turns: list[dict] | None, selection: dict | None
) -> dict | None:
    """The measured token a transcript selection still points at.

    A selection is checked against the turns it is used with rather than
    trusted: the conversation can be edited, undone, switched or replaced
    between the click and whatever the click is used for, and the turn that
    now sits at that index may be a different one entirely.

    What identifies the reply is the stamp it was drawn under, not its text.
    Retry puts a different reply at the same index, and two samples of one
    prompt share their opening tokens far more often than not, so a token ID
    alone would accept a click made against the reply that was replaced and
    branch the new one at a token the reader never saw. The stamp is the
    reply's own, minted when it was generated, so an older reply keeps the
    one it was clicked under and stays selectable for as long as it is on
    screen.
    """

    if not selection or selection.get("source") != "turn":
        return None
    try:
        turn = (turns or [])[int(selection["turn"])]
        metric = turn_tokens(turn)[int(selection["index"])]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if turn["role"] != "assistant":
        return None
    if turn.get("metrics_generation") != selection.get("at_generation"):
        # A different reply sits where the click landed - retried, regenerated,
        # or brought out of another fork.
        return None
    if int(metric["token_id"]) != int(selection.get("at_token_id", -1)):
        # The same reply, but its tokens have moved under the click.
        return None
    return metric


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
# The live values belong to Gradio's server-side session configuration, which
# is stable across requests and is not part of the snapshotted event inputs.
# Gradio installs it in LocalContext around every handler call and every next()
# of a streaming handler. Weak keys let the records go when Gradio expires a
# session, without another session's activity invalidating its selections.
_metrics_lock = threading.Lock()


@dataclass
class _PanelSession:
    generation: int = 0
    chat_generation: int = 0
    score_generation: int = 0
    token_view: bool = False


_panel_sessions = WeakKeyDictionary()
_direct_panel = _PanelSession()
_metrics_generation = 0


def _session_panel() -> _PanelSession:
    config = LocalContext.blocks_config.get(None)
    if config is None:
        # Plain Python callers have no browser session. Keep their existing
        # shared view, while real Gradio events always use their session key.
        return _direct_panel
    with _metrics_lock:
        return _panel_sessions.setdefault(config, _PanelSession())


def transcript_visible() -> bool:
    """Read the live toggle on each frame, including changes made mid-stream."""

    return LocalContext.blocks_config.get(None) is None or _session_panel().token_view


def new_metrics_generation(*, scored: bool = False) -> int:
    """Stamp a new token strip, invalidating selections made against the old one."""

    global _metrics_generation
    session = _session_panel()
    with _metrics_lock:
        _metrics_generation += 1
        session.generation = _metrics_generation
        if scored:
            session.score_generation = _metrics_generation
        else:
            session.chat_generation = _metrics_generation
        return _metrics_generation


def current_metrics_generation() -> int:
    """The stamp the strips on screen were drawn with.

    Read through a call rather than imported as a name: the counter is
    rebound by every mint, and a module that imported the number itself
    would keep comparing against the value it saw at import time.
    """

    return _session_panel().generation


def current_strip_generation(source: str) -> int:
    """Date each source independently; only the ambient prompt uses the panel epoch."""

    session = _session_panel()
    if source == "score":
        return session.score_generation
    if source == "response":
        return session.chat_generation
    return session.generation


def stamped(metrics: list[dict], generation: int | None = None):
    """Pair metrics with the stamp a click has to match to be published."""

    return (new_metrics_generation() if generation is None else generation), metrics


def empty_metrics() -> tuple[int, list[dict]]:
    """The metrics payload for a path that clears the strip."""

    return stamped([])


def cleared_panel(turns: list[dict] | None, scale_name: str):
    """Drop the live response's measurements and redraw the conversation.

    The prompt strip is emptied and the response metrics with it, under one
    fresh stamp: the two are replaced together, so minting one each would
    leave the first looking stale to inspect_token() the instant the second
    was minted. The stamp is also what invalidates a click made against the
    panel this replaces.

    The token view is not emptied, because it is not a copy of one response:
    it is the conversation, and the conversation is still on screen. It is
    redrawn from ``turns`` instead, so a handler that took a reply's
    measurements away - an edit, an undo, a switch to another fork - shows
    exactly what is left.
    """

    generation = new_metrics_generation()
    return (
        transcript_update(turns, scale_name),
        stamped([], generation),
        strip_update([], scale_name),
        stamped([], generation),
        "",
    )


def inspect_token(source: str):
    """A select listener that describes the token clicked in a strip.

    ``source`` decides which live strip stamp has to match. A click that
    does not count is skipped rather than
    answered: whatever replaced the strip already reset the detail panel, so
    repainting it with a token the reader can no longer see would undo that.
    """

    def inspect(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
        generation, metrics = metrics_state
        if generation != current_strip_generation(source):
            return gr.skip(), gr.skip()
        if not metrics:
            return NO_TOKEN_SELECTED, []
        metric = strip_metric(source, metrics_state, event_index(event))
        if metric is None:
            return "That token is no longer available. Generate another response.", []
        return describe_token(metric)

    return inspect


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
    "Click a token in the conversation, then one of its alternatives, then branch."
)


BRANCH_TEXT_HINT = (
    "Click a token in the conversation, type the text to put in its place, "
    "then branch."
)


BRANCH_TEXT_EMPTY = "Type the text that should replace the selected token first."


BRANCH_REASONING_CLOSE = (
    "🌱 That token is part of the automatic reasoning boundary. Branch from "
    "the first answer token after it instead."
)


BRANCH_UNAVAILABLE = (
    "🌱 Only a reply the model wrote can be branched. Scored text, prompt "
    "tokens and a message typed or edited by hand have no measured tokens to "
    "continue from."
)


BRANCH_MODEL_CHANGED = (
    "🌱 The model was reloaded since this reply was generated, so its tokens "
    "no longer belong to the weights in memory. Load that model again, or "
    "send the message again under this one, to branch it."
)


# Each strip checks the live stamp for the source that replaces it. A queued
# scored click survives a chat reply, but cannot overwrite a newer scoring.
STRIP_SOURCES = {"prompt", "score"}


def remember_strip_selection(source: str):
    """A select listener that keeps a prompt or scored-text click.

    These tokens have alternatives worth reading and no conversation to
    continue, so the selection records where the click landed and which strip
    it came from. ``choose_alternative`` then describes the row that was
    clicked and says why it cannot be branched, rather than appearing to do
    nothing at all.
    """

    def remember(metrics_state: tuple[int, list[dict]], event: gr.SelectData):
        # The second value disarms the branch either way. A click outside the
        # conversation replaces the detail panel, and a branch the reader can
        # no longer see must not stay waiting on the button.
        metric = strip_metric(source, metrics_state, event_index(event))
        if metric is None or not metric.get("scored", True):
            return None, None
        generation, _metrics = metrics_state
        return (
            {"source": source, "generation": generation, "index": event_index(event)},
            None,
        )

    return remember


def strip_metric(
    source: str, metrics_state: tuple[int, list[dict]], index
) -> dict | None:
    """One token of a strip, if that strip's clicks still count."""

    generation, metrics = metrics_state
    if generation != current_strip_generation(source):
        return None
    try:
        return metrics[int(index)]
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def selection_metric(
    turns: list[dict] | None,
    score_state: tuple[int, list[dict]],
    prompt_state: tuple[int, list[dict]],
    selection: dict | None,
) -> dict | None:
    """The token a selection points at, from a turn or from either strip."""

    if not selection:
        return None
    source = selection.get("source")
    if source == "turn":
        return selected_metric(turns, selection)
    if source not in STRIP_SOURCES:
        return None
    state = score_state if source == "score" else prompt_state
    generation, _metrics = state
    if selection.get("generation") != generation:
        return None
    return strip_metric(source, state, selection.get("index"))


def branch_target(
    turns: list[dict] | None, selection: dict | None
) -> tuple[int, dict] | str:
    """The turn and token a branch would start from, or why there is none.

    Every branch path checks the same three things, so they are checked in
    one place: the selection still points at a measured token of an assistant
    turn, that turn was generated by the weights now in memory, and the token
    is not one of the reasoning-boundary tokens the template supplied rather
    than the model choosing.
    """

    metric = selected_metric(turns, selection)
    if metric is None:
        return BRANCH_UNAVAILABLE
    position = int(selection["turn"])
    turn = (turns or [])[position]
    if turn.get("load_id") != runtime.MANAGER.load_id:
        return BRANCH_MODEL_CHANGED
    if metric.get("automatic_reasoning_close"):
        return BRANCH_REASONING_CLOSE
    return position, metric


def branch_ready_text(pick: dict) -> str:
    position = pick["position"]
    chosen = html.escape(repr(pick["text"]))
    original = html.escape(repr(pick["original"]))
    if pick["token_id"] == pick["original_id"]:
        return (
            f"🌱 **Branch ready:** keep the reply through token {position} "
            f"(`{chosen}`) and let the model continue from there with a fresh "
            "sample. Press **Branch from token**."
        )
    return (
        f"🌱 **Branch ready:** keep the first {position - 1} token"
        f"{'' if position == 2 else 's'}, put `{chosen}` where `{original}` was, "
        "and let the model continue. Press **Branch from token**."
    )


def select_transcript_token(
    turns: list[dict] | None,
    metrics_state: tuple[int, list[dict]],
    event: gr.SelectData,
):
    """Publish the token the reader clicked in the conversation's token view.

    One listener rather than four, because all of them ask the same question
    of the same click: which token is this. It publishes the detail panel and
    its alternatives, the selection the branch buttons read, the position the
    layer inspector would explain, and an empty pick - a row chosen for the
    previous token must not stay armed under a different one.

    ``metrics_state`` is read for its stamp alone, and it is the whole reason
    the stamp still exists here. Gradio resolves a listener's inputs when it
    gets round to processing the event, so a click made a moment before Retry,
    Undo, Clear or a switch arrives holding the conversation as it was - and
    would answer confidently about a reply that has since been replaced,
    landing on top of the reset frame that removed it, which every later
    streaming frame then skips. The stamp travels with the conversation it was
    snapshotted beside; comparing it against the live one is what tells a
    click that was overtaken from one that was not.

    The layer inspector is offered for the latest reply whose metrics and
    prompt IDs are retained. Scoring replaces the ambient panel but preserves
    that chat context, so its epoch is separate from the click/reset epoch.
    Conversation changes invalidate it; scoring alone does not.
    """

    generation, _metrics = metrics_state
    if generation != current_metrics_generation():
        return (gr.skip(),) * 5
    found = transcript_pick(turns, event)
    if found is None:
        return (gr.skip(),) * 5
    position, token_index = found
    turn = (turns or [])[position]
    if token_index is None:
        return NO_TOKEN_SELECTED, [], None, None, None

    metric = turn_tokens(turn)[token_index]
    summary, rows = describe_token(metric)
    selection = {
        "source": "turn",
        "turn": position,
        "index": token_index,
        "at_generation": turn.get("metrics_generation"),
        "at_token_id": int(metric["token_id"]),
    }
    if turn.get("load_id") != runtime.MANAGER.load_id:
        summary = f"{summary}\n\n{BRANCH_MODEL_CHANGED}"
    target = (
        {
            "generation": current_strip_generation("response"),
            "strip": "response",
            "index": token_index,
        }
        if turn.get("metrics_generation") == current_strip_generation("response")
        else None
    )
    return summary, rows, selection, target, None


def choose_alternative(
    turns: list[dict] | None,
    score_state: tuple[int, list[dict]],
    prompt_state: tuple[int, list[dict]],
    selected_token: dict | None,
    event: gr.SelectData,
):
    """Pair a row of the alternatives table with the token it belongs to."""

    metric = selection_metric(turns, score_state, prompt_state, selected_token)
    if metric is None:
        return gr.skip(), None
    try:
        candidate = metric["top_candidates"][event_index(event)]
    except (IndexError, KeyError, TypeError, ValueError):
        return gr.skip(), None

    summary, _rows = describe_token(metric)
    found = branch_target(turns, selected_token)
    if isinstance(found, str):
        return f"{summary}\n\n{found}", None
    position, _metric = found
    pick = {
        "source": "turn",
        "turn": position,
        "index": int(selected_token["index"]),
        "at_generation": selected_token.get("at_generation"),
        "at_token_id": int(metric["token_id"]),
        "position": int(metric["position"]),
        "token_id": int(candidate["token_id"]),
        "text": candidate["text"],
        "original_id": int(metric["token_id"]),
        "original": metric["text"],
    }
    return f"{summary}\n\n{branch_ready_text(pick)}", pick


def recolor(turns, score_state, prompt_state, scale_name: str):
    """Repaint everything painted by tokens when another scale is picked.

    The conversation is repainted from the turns, since that is what it is
    drawn from; the Score text tab's strip and the prompt strip are repainted
    from the measurements they were given. Each is repainted from what it is
    actually showing rather than from the inspector's own state, which by then
    may describe a reply generated since the passage was scored.
    """

    _generation, metrics = score_state
    _prompt_generation, prompt_metrics = prompt_state
    scale = resolve_scale(scale_name)
    return (
        transcript_update(turns, scale.name),
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
