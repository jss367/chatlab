"""Sending a message and everything that streams a reply: chat, retry, edit, undo, stop, branch."""

from __future__ import annotations

import contextlib
import logging
import random
import time

import gradio as gr

from steering import SteeringError, compact as compact_steering, from_controls as steering_from_controls

import charts
from conversation import (
    MAIN_BRANCH,
    THINK_CLOSE,
    branch_stamp,
    copy_forks,
    copy_turns,
    display_messages,
    forget_measurements,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    new_forks,
    split_reasoning,
    turn_tokens,
    user_index_at_or_before,
)
from model_runtime import (
    LOADING,
    ModelChanged,
)
from token_metrics import (
    DEFAULT_COLOR_SCALE,
    summarize,
)
from trace_export import build_trace
from ui import runtime
from ui.common import (
    CHART_EVERY,
    NO_TOKEN_SELECTED,
    SEED_LIMIT,
    failure_status,
    finalize_partial,
    send_stop_buttons,
)
from ui.conversations import (
    conversation_list_update,
)
from ui.panel import (
    BRANCH_HINT,
    BRANCH_MODEL_CHANGED,
    BRANCH_TEXT_EMPTY,
    BRANCH_TEXT_HINT,
    branch_target,
    cleared_panel,
    new_metrics_generation,
    prompt_note_text,
    strip_update,
    transcript_update,
    transcript_visible,
)

logger = logging.getLogger(__name__)


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
    "context_ids",
    "chat_metrics",
    "chat_context_ids",
    "selected_token",
    "branch_pick",
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


def stop_generation(
    turns: list[dict] | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Finish the turn that the cancelled generator left behind.

    Gradio closes ``generate_reply`` at its last yield, so nothing else ever
    finalizes that turn. A kept partial response is still a response: the
    tokens it carries are the ones it is made of, and it carries the load that
    produced them, so it can be branched from like any other reply.

    The token view is redrawn because this is what decides the stopped turn's
    fate: a reply with nothing in it is dropped here, and the cancelled
    generator's last frame is still on screen showing it.
    """

    turns = copy_turns(turns)
    kept = finalize_partial(turns)
    messages, _ = display_messages(turns)
    return (
        messages,
        turns,
        transcript_update(turns, scale_name),
        *send_stop_buttons(False),
        "Stopped. The partial response was kept."
        if kept
        else "Stopped before the model produced anything.",
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
    describing the visible text - an edited assistant reply. The prompt strip
    goes with the response metrics there: they carry one shared stamp, so
    re-stamping one alone would silently stop the other's clicks from
    publishing. The conversation's own token view is redrawn rather than
    emptied, since the edit is already in ``turns``.
    """

    messages, _ = display_messages(turns)
    panels = (
        cleared_panel(turns, scale_name)
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
        metrics,
        gr.skip(),
        None if clear_tokens else gr.skip(),
        None if clear_tokens else gr.skip(),
    )


BUSY_STATUS = "A response is already generating. Press Stop first."
# A load has the model instead. There is no Stop to press for one, so the
# message cannot be the one above; telling a reader to press a button that
# is not there is worse than saying nothing.
LOADING_STATUS = "A model is loading. Wait for it to finish."


def occupied() -> str | None:
    """What else has the model - a reply streaming, or a load - or ``None``.

    The early exit the handlers below take, and the answer is what the
    refusal is worded from. Never the guard - the guard is the reservation
    each of them goes on to make, which settles the same question in one step
    with the claim. See :meth:`ModelManager.claim_generation`.
    """

    return runtime.MANAGER.occupant


def busy_status(held: str | None = None) -> str:
    """Which refusal to show, for whatever ``held`` says has the model.

    ``held`` is what the refused claim or :func:`occupied` answered, passed
    down rather than read again here: a load that ends in between would leave
    this saying "press Stop" over a model nobody is holding.
    """

    return LOADING_STATUS if (held or runtime.MANAGER.occupant) == LOADING else BUSY_STATUS


NO_MODEL_STATUS = "Download and load a model first."


def no_model_state(prompt_text: str, turns: list[dict]):
    """What to show when memory is empty: the load that emptied it, or the advice.

    The occupancy read comes *after* the emptiness rather than before it, and
    the order is the point. A load unloads the old weights before it reads
    the new ones, so memory stands empty for the whole of that phase, and the
    advice below would send a reader to the Models page to start the load
    they are already waiting for. occupied() at the top of each handler is
    the early exit for a load that was under way when the click arrived; this
    is the one that started since.

    Claiming instead of asking - what the Score, Inspect, batch, API and
    extension paths do, since a claim cannot go stale - is not open to these
    handlers: generate_reply() claims further down, and a claim taken here
    would refuse its own generation. What is left is an instant-wide window
    in which a load finishing between the two reads leaves this saying "load
    a model" just after one finished loading, and the next Send works. That
    is the harmless way round; the other order is wrong for the minutes a
    load takes.
    """

    held = occupied()
    if held:
        return busy_state(held)
    return idle_state(prompt_text, turns, NO_MODEL_STATUS)


def busy_state(held: str | None = None):
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

    A load blocks a reply for the same reason and is refused the same way,
    with its own wording; ``held`` is which of the two the caller was turned
    away by. See :func:`busy_status`.
    """

    return (
        (gr.skip(),) * 5
        + (busy_status(held),)
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
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    *,
    forced_ids: tuple[int, ...] = (),
    literal_prefill_tokens: int = 0,
    automatic_reasoning_close_tokens: int = 0,
    literal_text_ranges: tuple[tuple[int, int], ...] = (),
    branch_note: str = "",
    expected_load_id: str | None = None,
    single_step: bool = False,
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
    runtime.MANAGER.busy checks in chat(), regenerate_from() and edit_message() are an
    early exit, not the guard: between such a check and the model lock that
    generate() takes sits the "Generating…" yield, and Gradio does not resume a
    handler until it has serialized that frame and sent it to the browser. A
    second click arriving inside that round trip used to sail past a manager
    that looked idle and overwrite the conversation from its stale snapshot.
    branch_with_text() has work to do before it can generate, so it takes the
    slot itself and calls _stream_reply() directly.
    """

    held = runtime.MANAGER.claim_generation()
    if held:
        yield busy_state(held)
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
            steering,
            steering_enabled,
            steering_strength,
            steering_layer,
            forced_ids=forced_ids,
            literal_prefill_tokens=literal_prefill_tokens,
            automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
            literal_text_ranges=literal_text_ranges,
            branch_note=branch_note,
            expected_load_id=expected_load_id,
            single_step=single_step,
        )
    finally:
        # Every exit runs this: a finished stream, a failure, and - the one
        # that matters - cancellation, where Gradio throws GeneratorExit in at
        # whichever yield the stream is parked on. Leaving the slot reserved
        # there would wedge the app: Send would refuse forever.
        runtime.MANAGER.release_generation()


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
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    *,
    forced_ids: tuple[int, ...] = (),
    literal_prefill_tokens: int = 0,
    automatic_reasoning_close_tokens: int = 0,
    literal_text_ranges: tuple[tuple[int, int], ...] = (),
    branch_note: str = "",
    expected_load_id: str | None = None,
    single_step: bool = False,
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

    steering = compact_steering(steering_from_controls(steering, steering_enabled, steering_strength, steering_layer))
    pending = make_turn("assistant", "", "")
    if steering is not None:
        pending["steering"] = steering
    pending["reasoning_closed"] = True
    # Where this reply came from, for the conversation list. The model is
    # stamped from the first update rather than read off runtime.MANAGER here: the
    # generator does not take the model lock until it is first resumed, and
    # a load can land in the round trip the opening frame costs. Only the
    # update knows which weights it came from. The token counts are filled
    # in as the stream arrives so a stopped or failed reply still says how
    # far it got.
    turns.append(pending)

    def snapshot(
        metrics,
        status,
        busy=True,
        reset_details=False,
        prompt_panel=None,
        charts_panel=None,
        trace=None,
        context_ids=gr.skip(),
    ):
        """One frame of the stream.

        The conversation's token view is drawn from ``turns``, which the loop
        has already written this frame's tokens into, so the reply paints
        itself as it arrives beneath the ones before it.

        ``reset_details`` belongs to the first frame only. That frame replaces
        the reply being answered into, so a token selected in the response it
        overwrites is gone and its probabilities must go with it. Later frames
        only append tokens, so a token picked mid-stream stays valid and its
        details are left alone.

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
            transcript_update(turns, scale_name) if transcript_visible() else gr.skip(),
            (generation, metrics),
            status,
            used_seed,
            *send_stop_buttons(busy),
            NO_TOKEN_SELECTED if reset_details else gr.skip(),
            # Gradio applies streaming diffs in place. A raw Dataframe value
            # followed by gr.skip() deletes data/headers from the very object
            # the table still renders, which can crash WebKit's next update.
            # Keep the value inside an update envelope so only that envelope
            # changes when later frames leave the selected token alone.
            gr.update(value=[]) if reset_details else gr.skip(),
            prompt_strip,
            prompt_metrics,
            prompt_note,
            summary_panel,
            surprise_panel,
            gr.skip() if trace is None else trace,
            context_ids,
            (generation, metrics),
            context_ids,
            None if reset_details else gr.skip(),
            None if reset_details else gr.skip(),
        )

    # The opening frame empties everything the previous response left behind,
    # the export included: a trace kept here would still be downloadable while
    # a different response was streaming in above it. Clear the branch states
    # with their visible details: an older turn can still be a valid branch
    # target, but a choice the panel no longer shows must not remain armed.
    applied_prefill = bool(assistant_prefill and not forced_ids)
    stream_note = branch_note or (
        "Assistant prefill applied." if applied_prefill else ""
    )
    yield snapshot(
        [],
        f"{stream_note} Generating…".strip(),
        reset_details=True,
        prompt_panel=(strip_update([], scale_name), (generation, []), ""),
        charts_panel=(charts.summary_tiles({}), charts.EMPTY_CHART),
        trace={},
        context_ids=(generation, [], runtime.MANAGER.load_id),
    )

    started = time.monotonic()
    raw_text = ""
    # Reasoning templates end the prompt with the opening <think> marker, so the
    # generated text never carries one. Only the runtime can tell us that.
    prefilled = False
    metrics: list[dict] = []
    status = "The model produced no tokens."
    first = True
    forced_prefix_tokens = 0
    literal_prefill = ""
    literal_spans: tuple[tuple[int, int], ...] = ()

    stream = runtime.MANAGER.generate(
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
        **({"steering": steering} if steering is not None else {}),
    )

    try:
        # closing() releases the model lock the moment the Stop button cancels
        # this event and Gradio closes the outer generator.
        with contextlib.closing(stream):
            for update in stream:
                raw_text = update.text
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
                metrics = list(update.metrics)
                # The reply carries its own measurements from here on, which
                # is what paints it in the conversation and what a branch
                # taken later replays. They are replaced rather than appended
                # to, so the copy the previous frame published stays as it was.
                pending["tokens"] = metrics
                pending["load_id"] = update.load_id
                pending["metrics_generation"] = generation
                pending["ends_on_stop_token"] = update.ends_on_stop_token
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
                        *([steering] if steering is not None else []),
                    )
                yield snapshot(
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
        # A refused steering request has not replaced the previous response.
        # Let the caller restore its original transcript on retry/branch.
        if first and isinstance(error, SteeringError):
            raise
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
            metrics,
            failure_status("Generation failed", str(error)),
            busy=False,
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
    # Finished replies with no visible text are dropped, as on cancellation.
    # A single step can contain only whitespace or a reasoning marker; keep
    # those measured tokens so the next click can advance past them. The chat
    # displays a pause notice while model_messages() keeps an empty assistant slot.
    if single_step and metrics and not pending.get("ends_on_stop_token"):
        pending["token_step_paused"] = True
        pending["reasoning_closed"] = True
        kept = True
    else:
        kept = finalize_partial(turns)
    sampling = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "seed": used_seed,
    }
    if steering is not None:
        sampling["steering"] = steering
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
        metrics,
        status,
        busy=False,
        charts_panel=(
            charts.summary_tiles(summarize(metrics)),
            charts.surprise_chart(metrics),
        ),
        trace=trace,
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
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
):
    held = occupied()
    if held:
        # Before anything else, including the checks below: every other exit
        # from this function writes the conversation back, and while another
        # generation is streaming that write is a stale overwrite.
        yield busy_state(held)
        return

    turns = copy_turns(turns)
    message = (prompt_text or "").strip()
    if not message:
        yield idle_state(prompt_text, turns, "Enter a message first.")
        return
    if not runtime.MANAGER.loaded:
        yield no_model_state(prompt_text, turns)
        return

    turns.append(make_turn("user", message))
    try:
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
            steering,
            steering_enabled,
            steering_strength,
            steering_layer,
        )
    except SteeringError as error:
        yield idle_state("", turns, failure_status("Steering failed", str(error)), clear_tokens=True, scale_name=scale_name)


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
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    *,
    restore_turns: list[dict] | None = None,
):
    """Throw away everything after the user turn at ``position`` and reply again."""

    held = occupied()
    if held:
        # Covers Retry and the chatbot's own retry button, which reach a
        # generation only through here.
        yield busy_state(held)
        return

    turns = copy_turns(turns)
    if position is None:
        yield idle_state(prompt_text, turns, "There is nothing to retry.")
        return
    if not runtime.MANAGER.loaded:
        yield no_model_state(prompt_text, turns)
        return

    try:
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
            steering,
            steering_enabled,
            steering_strength,
            steering_layer,
        )
    except SteeringError as error:
        yield idle_state(
            prompt_text, restore_turns if restore_turns is not None else turns,
            failure_status("Steering failed", str(error)), clear_tokens=True, scale_name=scale_name,
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
    # Steering follows the eleven display/generation settings. This path
    # clears strips itself and needs the color scale, not the vector snapshot.
    scale_name = settings[10] if len(settings) > 10 else DEFAULT_COLOR_SCALE
    held = occupied()
    if held:
        # Not just the branch that regenerates: editing an assistant turn
        # rewrites the conversation on its own, from the same stale snapshot.
        yield busy_state(held)
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
        held = runtime.MANAGER.claim_generation()
        if held:
            yield busy_state(held)
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
            runtime.MANAGER.release_generation()
        return

    edited = new_value.strip()
    if not edited:
        # An empty user turn is skipped by model_messages(), which would leave
        # the request with no user message at all.
        yield idle_state(prompt_text, turns, "A user message cannot be empty.")
        return

    if not runtime.MANAGER.loaded:
        # regenerate_from() would refuse too, but only after the truncation
        # below had already thrown away every later turn for a reply that is
        # never generated.
        yield no_model_state(prompt_text, turns)
        return

    original_turns = copy_turns(turns)
    turns = turns[: position + 1]
    turns[position]["content"] = edited
    yield from regenerate_from(position, prompt_text, turns, *settings, restore_turns=original_turns)


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
    replacement: str,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """Replay the clicked reply up to the clicked token, put typed text in its
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
    the encoding waits, and the turn the selection names cannot change under
    it.
    """

    held = runtime.MANAGER.claim_generation()
    if held:
        yield busy_state(held)
        return

    try:
        yield from _branch_with_text(
            selected_token,
            replacement,
            prompt_text,
            turns,
            *settings,
        )
    finally:
        # As in generate_reply(): a finished stream, a refusal, a failure and
        # a cancellation all pass through here, or the slot would stay taken.
        runtime.MANAGER.release_generation()


def _branch_with_text(
    selected_token: dict | None,
    replacement: str,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """The body of branch_with_text(), run with the generation slot held."""

    turns = copy_turns(turns)
    if not selected_token:
        yield idle_state(prompt_text, turns, BRANCH_TEXT_HINT)
        return
    if not runtime.MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return
    found = branch_target(turns, selected_token)
    if isinstance(found, str):
        yield idle_state(prompt_text, turns, found)
        return
    position, metric = found
    if not replacement:
        yield idle_state(prompt_text, turns, BRANCH_TEXT_EMPTY)
        return

    metrics = turn_tokens(turns[position])
    at = int(selected_token["index"]) + 1
    kept = [int(m["token_id"]) for m in metrics[: at - 1]]
    # The turn's own load is the one its tokens were produced by, and
    # branch_target() has just agreed it is the one in memory. It can still
    # change before the runtime takes the model lock, so the same load is
    # handed down and compared again under that lock, for the encoding and for
    # the replay alike; a mismatch there is ModelChanged.
    expected_load = turns[position].get("load_id")
    literal_prefill_tokens = literal_prefill_count(metrics, len(kept))
    automatic_reasoning_close_tokens = automatic_reasoning_close_count(
        metrics, len(kept)
    )
    try:
        replacement_ids = runtime.MANAGER.encode_replacement(
            kept,
            replacement,
            literal_prefill_tokens=literal_prefill_tokens,
            load_id=expected_load,
        )
        # Everything from the branched reply on is replaced, so the request is
        # built from the turns before it - which end with the message it was
        # an answer to.
        branch_turns = turns[:position]
        runtime.MANAGER.validate_generation_prefix(
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
    except SteeringError as error:
        yield idle_state(prompt_text, turns, failure_status("Could not branch", str(error)), clear_tokens=True)


def branch_from(
    pick: dict | None,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
    single_step: bool = False,
):
    """Replay the picked reply up to the picked token, swap it, and continue.

    Any reply in the conversation can be branched, not only the newest one:
    the pick names the turn it was made in, and that turn carries the tokens
    being replayed and the model load that produced them. What the branch
    replaces is that reply and everything after it, which is what makes the
    branch a different continuation rather than an edit in the middle.
    """

    held = occupied()
    if held:
        yield busy_state(held)
        return

    turns = copy_turns(turns)
    if not pick:
        yield idle_state(prompt_text, turns, BRANCH_HINT)
        return
    if not runtime.MANAGER.loaded:
        yield no_model_state(prompt_text, turns)
        return
    found = branch_target(turns, pick)
    if isinstance(found, str):
        yield idle_state(prompt_text, turns, found)
        return
    position, _metric = found
    metrics = turn_tokens(turns[position])
    at = int(pick["index"]) + 1
    kept = [int(metric["token_id"]) for metric in metrics[: at - 1]]
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
    if single_step:
        settings = (*settings[:6], 1, *settings[7:])
        note = (
            f"Keeping through token {at}; generating one next token."
            if unchanged else f"{note} Generating one next token."
        )

    # As in branch_with_text(): the check above is the fast path, and the
    # runtime compares the same load again under the model lock.
    expected_load = turns[position].get("load_id")
    try:
        yield from generate_reply(
            turns[:position],
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
            single_step=single_step,
        )
    except ModelChanged:
        # ``turns`` is still the whole conversation, old response included.
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
    except SteeringError as error:
        yield idle_state(prompt_text, turns, failure_status("Could not branch", str(error)), clear_tokens=True)


def next_token(pick, prompt_text, turns, *settings):
    """Branch from a chosen alternative, or extend the latest reply, by one token."""

    held = occupied()
    if held:
        yield busy_state(held)
        return
    turns = copy_turns(turns)
    if not runtime.MANAGER.loaded:
        yield no_model_state(prompt_text, turns)
        return
    if not pick:
        turn = turns[-1] if turns else {}
        metrics = turn_tokens(turn)
        if turn.get("role") != "assistant" or not metrics:
            yield idle_state(prompt_text, turns, "Generate a reply and choose a token alternative first.")
            return
        if turn.get("load_id") != runtime.MANAGER.load_id:
            yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
            return
        if turn.get("ends_on_stop_token"):
            yield idle_state(prompt_text, turns, "This reply has ended. Choose an earlier token alternative to branch from.")
            return
        metric = metrics[-1]
        pick = {
            "source": "turn",
            "turn": len(turns) - 1,
            "index": len(metrics) - 1,
            "at_generation": turn.get("metrics_generation"),
            "at_token_id": metric["token_id"],
            "token_id": metric["token_id"],
            "original_id": metric["token_id"],
            "text": metric["text"],
            "original": metric["text"],
        }
    yield from branch_from(pick, prompt_text, turns, *settings, single_step=True)


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
            transcript_update(turns, scale_name),
            gr.skip(),
            "There is nothing to undo.",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 8,
        )

    remaining = turns[:position]
    messages, _ = display_messages(remaining)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_panel(
        remaining, scale_name
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
        None,
        None,
    )


def undo_last(turns, scale_name: str = DEFAULT_COLOR_SCALE):
    return undo_from(last_user_index(turns), turns, scale_name)


def undo_message(event: gr.UndoData, turns, scale_name: str = DEFAULT_COLOR_SCALE):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    return undo_from(position, turns, scale_name)


NOTHING_TO_CLEAR = "There is nothing to clear."


def ask_clear_chat(turns: list[dict] | None, forks: dict | None):
    """Open the confirmation for Clear, or say there is nothing to clear.

    Returns the status line, the panel's visibility and its question. The
    count is taken here only to word the question; the clearing itself takes
    whatever is on screen when it runs.
    """

    hidden = gr.update(visible=False)
    forks = copy_forks(forks)
    others = max(len(forks["branches"]) - 1, 0)
    if not turns and not others:
        return NOTHING_TO_CLEAR, hidden, ""
    if others:
        loss = (
            f"the conversation on screen and {others} other"
            f"{'s' if others != 1 else ''}"
        )
    else:
        loss = "the conversation on screen"
    advice = (
        " To remove only this one, use **🗑️ Delete** in the conversations pane."
        if forks["active"] != MAIN_BRANCH
        else ""
    )
    return (
        gr.skip(),
        gr.update(visible=True),
        f"Clear {loss}? This cannot be undone.{advice}",
    )


def hide_clear_confirm():
    return gr.update(visible=False)


def clear_chat(scale_name: str = DEFAULT_COLOR_SCALE, forks: dict | None = None):
    """Empty everything the conversation owns.

    Clear cancels a running generation (see ``cancels`` on its listener), and a
    cancelled ``generate_reply`` never reaches its final yield, so this has to
    restore the Send button itself exactly as Stop does.

    Reached from the confirmation panel alone, which this closes on its way
    out; the Clear button itself only opens that panel.

    Every branch this page knew of is marked as changed now - the main one
    emptied, the rest deleted - so the saved file lets go of them rather than
    handing them back on the next save. A branch another page added since
    this one loaded was not in the question Clear asked, and is left to it.

    The sampling those branches carried is stamped as changed too. It is
    merged on its own time (see ``library.merge``), so without a stamp of
    its own the file's older copy would look like the newer of the two and
    the emptied main conversation would come back pinned to the sampling of
    the one that was cleared.
    """

    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_panel(
        [], scale_name
    )
    known = copy_forks(forks)
    forks = new_forks()
    stamp = branch_stamp()
    forks["updated"] = {name: stamp for name in (MAIN_BRANCH, *known["branches"])}
    forks["sampling_updated"] = {
        name: stamp for name in (MAIN_BRANCH, *known["sampling"])
    }
    return (
        [],
        [],
        strip,
        metrics,
        "Every conversation cleared.",
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED,
        [],
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
        None,
        None,
        forks,
        conversation_list_update(forks, []),
        gr.update(visible=False),
    )
