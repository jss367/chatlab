"""The Score text tab and its token budget."""

from __future__ import annotations

import gradio as gr

import charts
from model_runtime import LOADING
from token_metrics import (
    DEFAULT_COLOR_SCALE,
    summarize,
)
from ui import runtime
from ui.common import (
    NO_TOKEN_SELECTED,
    SEAM_CAVEAT,
    TEMPLATE_CAVEAT,
    failure_status,
)
from ui.panel import (
    new_metrics_generation,
    prompt_note_text,
    stamped,
    strip_update,
)


SCORE_BUSY = "Wait for the response to finish before scoring text."
# A load has the model instead, and it is not a response: a reader watching
# for one to finish would be watching the wrong thing.
SCORE_LOADING = "Wait for the model to finish loading before scoring text."


def score_busy(held: str | None) -> str:
    """The refusal for whatever ``held`` says has the model."""

    return SCORE_LOADING if held == LOADING else SCORE_BUSY


def score_text(
    context: str,
    text: str,
    use_chat_template: bool,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Measure text the model did not write, and put it in the same panel.

    This is a generator for the same reason inspect_layers() is: Gradio does
    not resume a streaming handler until the browser has been sent the frame
    it yielded, so the generation slot is held not just for the pass but
    until the scored strips are on screen. Returning instead would give the
    slot back while the frame was still in flight, and a Send starting in
    that window would mint a newer stamp and publish its opening frame
    first, leaving these strips on screen under a stamp the app has already
    moved past and refusing every click on them.
    """

    skip = gr.skip()
    # Everything the success path publishes before the status line: the scored
    # strip, the two copies of its measurements, the context strip with its
    # own state and note, and the two charts.
    refused = (skip,) * 8

    # A generation holds the model lock across every one of its yields, so
    # without the slot this pass would simply wait on it: the button would sit
    # dead for the length of the response and then fire, with nothing on screen
    # to tell that apart from a hang. Reserving rather than reading
    # runtime.MANAGER.busy also keeps a Send from starting while the pass runs and
    # minting a stamp over the strips this is about to replace, which would
    # leave the scored tokens on screen refusing every click. inspect_layers()
    # takes the slot for both reasons.
    #
    # Claimed before memory is looked at, not after. A load unloads the old
    # weights before it reads the new ones, so for the whole of that phase
    # there is no model loaded and the check below would send the reader to
    # the Models page to load one - which is where the load they are waiting
    # for already is. The claim is the question that cannot go stale: it is
    # refused while a load is claimed, and while it is held no load can
    # start, so "loaded" read under it stays true until the slot goes back.
    held = runtime.MANAGER.claim_generation()
    if held:
        yield refused + (score_busy(held),) + (skip,) * 6
        return
    try:
        if not runtime.MANAGER.loaded:
            yield refused + ("Download and load a model first.",) + (skip,) * 6
            return
        try:
            result = runtime.MANAGER.score_text(
                text, context=context or "", use_chat_template=bool(use_chat_template)
            )
        except Exception as error:
            yield refused + (
                failure_status("Could not score that text", str(error)),
            ) + (skip,) * 6
            return

        summary = summarize(result.metrics)
        status = (
            f"Scored {summary['token_count']:,} tokens. "
            f"Perplexity {summary['perplexity']:,.1f}."
        )
        # What was scored comes before how exactly it was scored: the template
        # caveat says which passage the numbers describe, the seam caveat says
        # how sure their first token is.
        if result.chat_template_missing:
            status = f"{status} {TEMPLATE_CAVEAT}"
        if not result.seam_verified:
            status = f"{status} {SEAM_CAVEAT}"
        # Both strips are replaced, so they take one shared stamp - and that
        # stamp is what drops a click made against the response they overwrite.
        generation = new_metrics_generation(scored=True)
        scored = stamped(result.metrics, generation)
        context_state = (
            generation, [int(value) for value in result.context_ids], runtime.MANAGER.load_id
        )
        yield (
            strip_update(result.metrics, scale_name, "Scored tokens — click one"),
            # Twice: to the inspector, which now describes this passage, and to
            # the scored strip's own state, which nothing but another scoring
            # pass rewrites. The inspector's copy is replaced by the next reply
            # while the strip goes on showing the passage, so the strip is
            # repainted and its clicks are read from the copy that still
            # matches what is drawn there.
            scored,
            scored,
            strip_update(result.context_metrics, scale_name),
            stamped(result.context_metrics, generation),
            prompt_note_text(len(result.context_metrics), "", "context"),
            charts.summary_tiles(summary),
            charts.surprise_chart(result.metrics, title="Surprise per scored token"),
            status,
            NO_TOKEN_SELECTED,
            [],
            # The panel is reset, so whatever it had armed is disarmed with it.
            # A branch the reader can no longer see must not still be waiting
            # on the button.
            None,
            None,
            context_state,
            context_state,
        )
        # Resumed once the browser has the frame above, so nothing between the
        # mint and the strips arriving can hold the slot.
    finally:
        runtime.MANAGER.release_generation()


SCORE_COUNT_HINT = "Paste some text to see how many tokens it comes to."


SCORE_COUNT_UNKNOWN = (
    "The token count needs a loaded model that is not mid-response."
)


# The other thing that takes the model away from a count, and the wording
# above is false for it: there is no response to wait out.
SCORE_COUNT_LOADING = (
    "The token count comes back when the model has finished loading."
)


# Both of the messages that give up. The recovery below is driven from the
# badge's timer and recognizes either.
SCORE_COUNT_UNAVAILABLE = (SCORE_COUNT_UNKNOWN, SCORE_COUNT_LOADING)


def score_count_unavailable() -> str:
    """Why the count could not be had, in the reader's terms.

    A count is refused rather than claimed - it never takes the generation
    slot - so this reads what has the model rather than being handed it. A
    label, never a guard: the worst a stale read can do is name the other
    true-enough reason, and the timer asks again a second later.
    """

    return (
        SCORE_COUNT_LOADING
        if runtime.MANAGER.occupant == LOADING
        else SCORE_COUNT_UNKNOWN
    )


# Every listener that writes the count shares this queue, and so runs one at a
# time and in the order the requests were sent.
#
# Gradio's concurrency limit is per event, not across events, so without this
# the timer's recovery and a keystroke's count are free to overlap - and they
# contend for the same model lock, so overlapping means one of them loses it
# and publishes the "not mid-response" message. Whichever finished last would
# then be on screen, which can be the one computed for the older passage. That
# is worse than the message it replaces: a count that does not describe the
# box is exactly what this line exists to rule out, and the recovery below
# leaves any numeric count alone, so a wrong one would stay until the next
# edit. In one queue the fresher request is always the one sent later, and so
# always the one that publishes last.
SCORE_BUDGET_QUEUE = "score-budget"


# And one for the sampling summary, for the same reason. Four sliders is four
# listeners, and always_last coalesces each on its own; across them Gradio
# orders nothing. Each handler reads all four values as they were when its
# request was sent, so two sliders moved in quick succession can finish out of
# order and leave the label describing the older pair - and it is only
# rewritten by the next change, so it would stay that way.
SAMPLING_LABEL_QUEUE = "sampling-label"


def score_token_count(context: str, text: str, use_chat_template: bool):
    """The count for the box, and the load it was counted against.

    The second half is what lets a count be told from a stale one. A
    tokenizer belongs to the weights in memory, so a number counted under one
    load says nothing about the next, and ``load_id`` is how the rest of the
    app already names "these weights, this time".
    """

    load_id = runtime.MANAGER.load_id
    if not text:
        return SCORE_COUNT_HINT, load_id
    counted = runtime.MANAGER.count_score_tokens(
        text, context=context or "", use_chat_template=bool(use_chat_template)
    )
    if counted is None:
        return score_count_unavailable(), load_id
    count, limit = counted
    if count > limit:
        return (
            f'<span class="failure-text">{count:,} tokens, above the '
            f"{limit:,} this model can score in one pass. Score it in "
            "smaller pieces.</span>",
            load_id,
        )
    return f"{count:,} of {limit:,} tokens.", load_id


def recover_score_budget(
    shown: str, counted_load, context: str, text: str, use_chat_template: bool
):
    """Recount when the shown number has gone wrong, or could have.

    Two things put it wrong, and neither corrects itself.

    A count asked for during a reply cannot have the model lock and says so.
    The reply ends, the model goes idle, and the box still reads "not
    mid-response" until something is typed into it. Every path out of a
    generation would have to remember to recompute - the ordinary end of
    one, Stop, and the four handlers that cancel one - and a path added later
    would have to remember too.

    The other is a model swapped out from another tab. The handlers that
    recompute on a load or an unload only reach the tab that asked for it,
    which is the whole reason the badge is on a timer rather than published
    by those handlers alone. A tab left showing a number counted under the
    old tokenizer, against the old context limit, would go on advertising it
    under the new model's badge.

    So the recovery is driven from that same timer, and guarded on both: the
    message that gives up, and the load the number was counted against. In
    every other state this returns without asking, and asking is itself cheap
    while it would still fail, since an unloaded model and a held lock both
    refuse before any encoding happens.
    """

    if shown not in SCORE_COUNT_UNAVAILABLE and counted_load == runtime.MANAGER.load_id:
        return gr.skip(), gr.skip()
    recovered, load_id = score_token_count(context, text, use_chat_template)
    if recovered == shown and load_id == counted_load:
        return gr.skip(), gr.skip()
    return recovered, load_id
