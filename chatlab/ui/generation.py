"""Sending a message and everything that streams a reply: chat, retry, edit, undo, stop, branch."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import gradio as gr

from chatlab import library
from chatlab.steering import SteeringError

from chatlab.conversation import (
    MAIN_BRANCH,
    branch_archived,
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
    turn_tokens,
    user_index_at_or_before,
)
from chatlab.model_runtime import LOADING
from chatlab.seeds import resolve_seed as resolve_seed
from chatlab.conversation import split_response_text as split_response_text
from chatlab.generation_request import PromptEdit, ReplayOptions, request_from_controls
from chatlab.ui.reply_state import POSITION_LIMIT_NOTE as POSITION_LIMIT_NOTE
from chatlab.ui.reply_state import generation_progress as generation_progress
from chatlab.ui.reply_stream import stream_reply as _stream_reply
from chatlab.token_metrics import DEFAULT_COLOR_SCALE
from chatlab.attachments import MAX_IMAGES_PER_PROMPT, names_in, picture_count, too_many_pictures
from chatlab.model_errors import ModelChanged
from chatlab.ui import runtime
from chatlab.ui.common import (
    failure_status,
    finalize_partial,
    send_stop_values,
)
from chatlab.ui.conversations import (
    conversation_list_update,
    panel_reset,
)
from chatlab.ui.pictures import strip_html as picture_strip
from chatlab.ui.outputs import (
    CHAT_OUTPUT_NAMES,
    CLEAR_OUTPUT_NAMES,
    STOP_OUTPUT_NAMES,
    UNDO_OUTPUT_NAMES,
    Frame,
    skipped,
)
from chatlab.ui.panel import (
    BRANCH_HINT,
    BRANCH_MODEL_CHANGED,
    BRANCH_TEXT_EMPTY,
    BRANCH_TEXT_HINT,
    PROMPT_EDIT_EMPTY,
    PROMPT_EDIT_MODEL_CHANGED,
    PROMPT_EDIT_NO_MESSAGE,
    branch_target,
    prompt_edit_target,
    transcript_update,
)

def chat_frame(**values) -> Frame:
    """One frame of a generation handler: the CHAT_OUTPUT_NAMES it changes.

    Every generation handler publishes a Frame over those names rather than a
    tuple in their order, so the refusal paths - which skip most of them -
    name the one or two they do change and leave the rest out; see ui.outputs.
    """

    return Frame(CHAT_OUTPUT_NAMES, **values)


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
    return Frame(
        STOP_OUTPUT_NAMES,
        chatbot=messages,
        turns=turns,
        strip=transcript_update(turns, scale_name),
        **send_stop_values(False),
        status="Stopped. The partial response was kept."
        if kept
        else "Stopped before the model produced anything.",
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
    frame = chat_frame(
        prompt=prompt_text,
        chatbot=messages,
        turns=copy_turns(turns),
        status=status,
        **send_stop_values(False),
    )
    if clear_tokens:
        reset = panel_reset(turns, scale_name)
        # The latest reply's own copy of its measurements goes with the
        # panel's: the two describe the same reply.
        frame.update(reset, chat_metrics=reset["metrics"])
    return frame


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


# Where a few of the settings sit in the tuple the ``*settings`` handlers are
# handed. They arrive in the order generate_reply() declares, which is the
# order ui.layout wires the controls in, and the handlers below reach past the
# ones they do not care about rather than naming all twelve. A control added
# between two of them moves everything after it, so the positions are counted
# here once instead of in each of those handlers.
SYSTEM_PROMPT_SETTING = 0
KEEP_REASONING_SETTING = 1
MAX_NEW_TOKENS_SETTING = 7
COLOR_SCALE_SETTING = 11


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

    return chat_frame(status=busy_status(held))


def generate_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_top_below: float,
    max_new_tokens: int,
    seed: object,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    thinking_mode: str = "default",
    *,
    forced_ids: tuple[int, ...] = (),
    prompt_edit: PromptEdit | None = None,
    replaying: bool = False,
    literal_prefill_tokens: int = 0,
    automatic_reasoning_close_tokens: int = 0,
    literal_text_ranges: tuple[tuple[int, int], ...] = (),
    branch_note: str = "",
    expected_load_id: str | None = None,
    single_step: bool = False,
    branch_thinking_mode: str | None = None,
) -> Iterator[Frame]:
    """Stream one assistant reply for ``turns``, which must end with a user turn.

    ``assistant_prefill`` is arbitrary answer text the model replays before it
    samples anything. ``forced_ids`` is the token-level version used by a
    branch: the tokens kept from an earlier response and the alternative the
    reader picked. A branch already contains any prefix that was on the old
    response, so it takes precedence. ``branch_note`` leads the status line,
    for a branch or for an edited prompt.

    ``prompt_edit`` is one token replaced in the prompt the last reply was
    generated from: ``ids`` is the whole edited prompt, fed in place of
    anything the template would write, and ``position``, ``original`` and
    ``replacement`` describe the change for the note and the export. The
    response itself is sampled from scratch, so the sampling controls and the
    assistant prefill apply to it exactly as they would to any new reply.

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

    ``thinking_mode`` selects the model's native template mode for a new reply.
    Token branches supply ``branch_thinking_mode`` from the original reply so
    a changed control cannot change the prompt underneath the replayed tokens.

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
            request_from_controls(
                turns,
                prompt_text,
                system_prompt,
                keep_reasoning,
                assistant_prefill,
                temperature,
                top_p,
                top_k,
                skip_top_below,
                max_new_tokens,
                seed,
                randomize_seed,
                analyze_prompt,
                scale_name,
                steering,
                steering_enabled,
                steering_strength,
                steering_layer,
                thinking_mode,
                replay=ReplayOptions(
                    forced_ids=forced_ids,
                    prompt_edit=prompt_edit,
                    replaying=replaying,
                    literal_prefill_tokens=literal_prefill_tokens,
                    automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                    literal_text_ranges=literal_text_ranges,
                    expected_load_id=expected_load_id,
                    single_step=single_step,
                    thinking_mode=branch_thinking_mode,
                ),
                note=branch_note,
            )
        )
    finally:
        # Every exit runs this: a finished stream, a failure, and - the one
        # that matters - cancellation, where Gradio throws GeneratorExit in at
        # whichever yield the stream is parked on. Leaving the slot reserved
        # there would wedge the app: Send would refuse forever.
        runtime.MANAGER.release_generation()


def chat(
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_top_below: float,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    thinking_mode: str = "default",
    images: list[str] | None = None,
):
    """Send the message in the box, and the pictures waiting with it, and stream the reply.

    The pictures come last, after the settings, so the listener's inputs read
    as every other generation handler's do with one more on the end. The
    frame that empties the box empties the pictures too: the one whose
    conversation has grown by the message, which is the opening frame of a
    reply. A refusal leaves the conversation as it was, and the pictures
    where they were.
    """

    before = len(turns or [])
    cleared = False
    for frame in _send(
        prompt_text,
        turns,
        system_prompt,
        keep_reasoning,
        assistant_prefill,
        temperature,
        top_p,
        top_k,
        skip_top_below,
        max_new_tokens,
        seed,
        randomize_seed,
        analyze_prompt,
        scale_name,
        steering,
        steering_enabled,
        steering_strength,
        steering_layer,
        thinking_mode,
        images=images,
    ):
        if not cleared and not skipped(frame["turns"]) and len(frame["turns"]) > before:
            frame.update(attachments=[], attachment_strip=picture_strip([]))
            cleared = True
        yield frame


def _send(
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_top_below: float,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    thinking_mode: str = "default",
    *,
    images: list[str] | None = None,
):
    """The body of :func:`chat`: send the message with ``images`` attached."""

    held = occupied()
    if held:
        # Before anything else, including the checks below: every other exit
        # from this function writes the conversation back, and while another
        # generation is streaming that write is a stale overwrite.
        yield busy_state(held)
        return

    turns = copy_turns(turns)
    message = (prompt_text or "").strip()
    images = list(images or [])
    if not message and not images:
        yield idle_state(prompt_text, turns, "Enter a message first.")
        return
    if not runtime.MANAGER.loaded:
        yield no_model_state(prompt_text, turns)
        return
    if (images or names_in(turns)) and not runtime.MANAGER.accepts_images:
        # Said before the message joins the conversation, which keeps both it
        # and its pictures in the box for a model that can read them. An
        # earlier picture counts too: the model would be fed it all the same.
        yield idle_state(prompt_text, turns, runtime.MANAGER.images_refusal())
        return
    count = picture_count(turns) + len(images)
    if count > MAX_IMAGES_PER_PROMPT:
        yield idle_state(prompt_text, turns, too_many_pictures(count))
        return

    user_turn = make_turn("user", message)
    if images:
        user_turn["images"] = images
    turns.append(user_turn)
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
            skip_top_below,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            scale_name,
            steering,
            steering_enabled,
            steering_strength,
            steering_layer,
            thinking_mode,
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
    skip_top_below: float,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    steering: dict | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    thinking_mode: str = "default",
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
    if names_in(turns[: position + 1]) and not runtime.MANAGER.accepts_images:
        # Refused before the reply it would replace is thrown away.
        yield idle_state(
            prompt_text, restore_turns if restore_turns is not None else turns,
            runtime.MANAGER.images_refusal(),
        )
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
            skip_top_below,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            scale_name,
            steering,
            steering_enabled,
            steering_strength,
            steering_layer,
            thinking_mode,
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
    # Steering follows the display/generation settings. This path clears
    # strips itself and needs the color scale, not the vector snapshot.
    scale_name = (
        settings[COLOR_SCALE_SETTING]
        if len(settings) > COLOR_SCALE_SETTING
        else DEFAULT_COLOR_SCALE
    )
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
    if part == "error":
        yield idle_state(
            prompt_text, turns, "A failure notice cannot be edited. Press Retry to answer again."
        )
        return
    if part == "image":
        yield idle_state(
            prompt_text, turns,
            "A picture cannot be edited. Undo the message to take it back into the box.",
        )
        return
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
    if not edited and not turns[position].get("images"):
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
    # A reply generated from an edited prompt was not given the prompt this
    # conversation renders, so replaying its tokens against that one would
    # score them under a context they never had. Replay the prompt it had.
    prompt_edit = turns[position].get("prompt_edit")
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
                system_prompt=settings[SYSTEM_PROMPT_SETTING],
                include_reasoning=settings[KEEP_REASONING_SETTING],
            ),
            (*kept, *replacement_ids),
            max_new_tokens=int(settings[MAX_NEW_TOKENS_SETTING]),
            load_id=expected_load,
            thinking_mode=turns[position].get("thinking_mode", "default"),
            prompt_override_ids=prompt_edit["ids"] if prompt_edit else None,
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
            request_from_controls(
                branch_turns,
                prompt_text,
                *settings,
                replay=ReplayOptions(
                    forced_ids=(*kept, *replacement_ids),
                    prompt_edit=prompt_edit,
                    replaying=True,
                    literal_prefill_tokens=literal_prefill_tokens,
                    automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                    literal_text_ranges=(
                        *literal_text_ranges(metrics, len(kept)),
                        (replacement_start, replacement_start + len(replacement_ids)),
                    ),
                    expected_load_id=expected_load,
                    thinking_mode=turns[position].get("thinking_mode", "default"),
                ),
                note=note,
                fork_origin={
                    "kind": "token",
                    "turn": position,
                    "token": at,
                    "original": metric["text"],
                    "original_id": int(metric["token_id"]),
                    "replacement": replacement,
                    "replacement_ids": list(replacement_ids),
                },
                previous_turns=turns,
            )
        )
    except ModelChanged:
        # ``turns`` is still the whole conversation, old response included.
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
    except SteeringError as error:
        yield idle_state(
            prompt_text, turns, failure_status("Could not branch", str(error)), clear_tokens=True
        )


def answer_edited_prompt(
    edit: dict,
    context_state,
    prompt_state: tuple[int, list[dict]],
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """Answer the last message again from a prompt with one token replaced.

    The prompt fed is the one the reply on screen was generated from, token
    for token, with the clicked position swapped for an alternative the model
    ranked there or for text the reader typed. Nothing is written back to the
    conversation: the turns still say what was asked, and the next message is
    prompted from them through the chat template as usual. What the edit
    changes is this one reply, and the strip beside it shows the prompt that
    produced it.

    The generation slot is taken first, as ``branch_with_text`` takes it and
    for the same reason: the encoding waits on the model lock, and a Send that
    slipped in ahead of it would hold that lock for a whole generation and
    leave this handler replacing a reply that is no longer the one the prompt
    was recorded for.
    """

    held = runtime.MANAGER.claim_generation()
    if held:
        yield busy_state(held)
        return

    try:
        yield from _answer_edited_prompt(
            edit, context_state, prompt_state, prompt_text, turns, *settings
        )
    finally:
        runtime.MANAGER.release_generation()


def _answer_edited_prompt(
    edit: dict,
    context_state,
    prompt_state: tuple[int, list[dict]],
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
):
    """The body of answer_edited_prompt(), run with the generation slot held."""

    turns = copy_turns(turns)
    if not runtime.MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return
    found = prompt_edit_target(context_state, prompt_state, edit.get("selection"))
    if isinstance(found, str):
        yield idle_state(prompt_text, turns, found)
        return
    index, prompt_ids, expected_load, metric = found
    # The reply is regenerated for the message it answered, so everything from
    # that message on is replaced - exactly what Retry does, with the prompt
    # edited rather than rebuilt.
    position = last_user_index(turns)
    if position is None:
        yield idle_state(prompt_text, turns, PROMPT_EDIT_NO_MESSAGE)
        return

    try:
        replacement_ids, replacement = _prompt_replacement(
            edit, metric, prompt_ids, index, expected_load
        )
    except ModelChanged:
        yield idle_state(
            prompt_text, turns, PROMPT_EDIT_MODEL_CHANGED, clear_tokens=True
        )
        return
    except (ValueError, RuntimeError) as error:
        yield idle_state(prompt_text, turns, f"✏️ {error}")
        return

    note = (
        f"Prompt token {index + 1}: {replacement!r} instead of {metric['text']!r}."
    )
    # The mode the replaced reply ran under, not the control as it stands now:
    # the edited prompt already contains whatever the template wrote for that
    # mode, and recording a mode it was never rendered for would misreport it.
    replaced = turns[position + 1] if len(turns) > position + 1 else None
    try:
        yield from _stream_reply(
            request_from_controls(
                turns[: position + 1],
                prompt_text,
                *settings,
                replay=ReplayOptions(
                    prompt_edit={
                        "ids": (*prompt_ids[:index], *replacement_ids, *prompt_ids[index + 1 :]),
                        "position": index + 1,
                        "original": metric["text"],
                        "replacement": replacement,
                    },
                    expected_load_id=expected_load,
                    thinking_mode=replaced.get("thinking_mode") if replaced else None,
                ),
                note=note,
                fork_origin={
                    "kind": "prompt",
                    "turn": position + 1,
                    "token": index + 1,
                    "original": metric["text"],
                    "original_id": int(metric["token_id"]),
                    "replacement": replacement,
                    "replacement_ids": list(replacement_ids),
                },
                previous_turns=turns,
            )
        )
    except ModelChanged:
        yield idle_state(prompt_text, turns, PROMPT_EDIT_MODEL_CHANGED, clear_tokens=True)
    except SteeringError as error:
        yield idle_state(
            prompt_text,
            turns,
            failure_status("Could not answer again", str(error)),
            clear_tokens=True,
        )


def _prompt_replacement(
    edit: dict, metric: dict, prompt_ids: list[int], index: int, expected_load: str | None
) -> tuple[list[int], str]:
    """The ids to put where one prompt token was, and the text they spell.

    An alternative is taken as the id the model ranked, not as its text: the
    two are not interchangeable, since encoding that text at this position can
    land on a different spelling of it. Typed text has no id of its own and is
    encoded against the tokens in front of it, under the model lock.
    """

    if edit.get("kind") == "candidate":
        try:
            candidate = metric["top_candidates"][int(edit["index"])]
        except (IndexError, KeyError, TypeError, ValueError):
            raise ValueError("That alternative is not one of this token's.") from None
        return [int(candidate["token_id"])], candidate["text"]
    text = edit.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(PROMPT_EDIT_EMPTY)
    return (
        runtime.MANAGER.encode_prompt_replacement(
            prompt_ids[:index], text, load_id=expected_load
        ),
        text,
    )


def branch_from(
    pick: dict | None,
    prompt_text: str,
    turns: list[dict] | None,
    *settings,
    single_step: bool = False,
    resample: bool = False,
):
    """Replay the picked reply up to the picked token, swap it, and continue.

    Any reply in the conversation can be branched, not only the newest one:
    the pick names the turn it was made in, and that turn carries the tokens
    being replayed and the model load that produced them. What the branch
    replaces is that reply and everything after it, which is what makes the
    branch a different continuation rather than an edit in the middle.

    With ``resample``, the pick is a token selection: only the tokens before
    it are replayed, and the selected token is sampled again as well.
    """

    # Validation tokenizes the prompt under the model lock. Own the generation
    # slot first so a competing Send cannot replace the conversation meanwhile.
    held = runtime.MANAGER.claim_generation()
    if held:
        yield busy_state(held)
        return

    try:
        yield from _branch_from(pick, prompt_text, turns, *settings, single_step=single_step, resample=resample)
    finally:
        runtime.MANAGER.release_generation()


def _branch_from(
    pick: dict | None,
    prompt_text: str,
    turns: list[dict] | None,
    *settings: Any,
    single_step: bool = False,
    resample: bool = False,
) -> Iterator[Frame]:
    """Validate and replay a token branch with the generation slot held."""

    turns = copy_turns(turns)
    if not pick:
        yield idle_state(prompt_text, turns, BRANCH_HINT)
        return
    if not runtime.MANAGER.loaded:
        yield idle_state(prompt_text, turns, NO_MODEL_STATUS)
        return
    found = branch_target(turns, pick)
    if isinstance(found, str):
        yield idle_state(prompt_text, turns, found)
        return
    position, _metric = found
    metrics = turn_tokens(turns[position])
    at = int(pick["index"]) + 1
    kept = [int(metric["token_id"]) for metric in metrics[: at - 1]]
    forced = tuple(kept) if resample else (*kept, int(pick["token_id"]))
    literal_prefill_tokens = literal_prefill_count(metrics, len(kept))
    automatic_reasoning_close_tokens = automatic_reasoning_close_count(
        metrics, len(kept)
    )
    unchanged = not resample and pick["token_id"] == pick.get("original_id")
    if (
        unchanged
        and literal_prefill_tokens == len(kept)
        and metrics[len(kept)].get("literal_prefill")
    ):
        literal_prefill_tokens += 1
    if resample:
        note = f"Regenerating from token {at}."
    elif unchanged:
        note = f"Resampling from token {at} ({pick['text']!r})."
    else:
        note = f"Branched at token {at}: {pick['text']!r} instead of {pick['original']!r}."
    if single_step:
        settings = (
            *settings[:MAX_NEW_TOKENS_SETTING],
            1,
            *settings[MAX_NEW_TOKENS_SETTING + 1 :],
        )
        note = (
            f"Keeping through token {at}; generating one next token."
            if unchanged else f"{note} Generating one next token."
        )

    # As in branch_with_text(): the check above is the fast path, and the
    # runtime compares the same load again under the model lock.
    expected_load = turns[position].get("load_id")
    # As with a typed branch: a reply given an edited prompt is replayed
    # against that prompt, not against the one the template would write now.
    prompt_edit = turns[position].get("prompt_edit")
    try:
        runtime.MANAGER.validate_generation_prefix(
            model_messages(
                turns[:position],
                system_prompt=settings[SYSTEM_PROMPT_SETTING],
                include_reasoning=settings[KEEP_REASONING_SETTING],
            ),
            forced,
            max_new_tokens=int(settings[MAX_NEW_TOKENS_SETTING]),
            load_id=expected_load,
            thinking_mode=turns[position].get("thinking_mode", "default"),
            prompt_override_ids=prompt_edit["ids"] if prompt_edit else None,
        )
    except ModelChanged:
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
        return
    except (ValueError, RuntimeError) as error:
        yield idle_state(prompt_text, turns, f"🌱 {error}")
        return

    try:
        yield from _stream_reply(
            request_from_controls(
                turns[:position],
                prompt_text,
                *settings,
                replay=ReplayOptions(
                    forced_ids=forced,
                    prompt_edit=prompt_edit,
                    replaying=True,
                    literal_prefill_tokens=literal_prefill_tokens,
                    automatic_reasoning_close_tokens=automatic_reasoning_close_tokens,
                    literal_text_ranges=literal_text_ranges(
                        metrics, len(forced) if unchanged else len(kept)
                    ),
                    expected_load_id=expected_load,
                    single_step=single_step,
                    thinking_mode=turns[position].get("thinking_mode", "default"),
                ),
                note=note,
                fork_origin=None
                if single_step and unchanged and position == len(turns) - 1 and at == len(metrics)
                else {
                    "kind": "token",
                    "turn": position,
                    "token": at,
                    "single_step": single_step,
                    "original": _metric["text"],
                    "original_id": int(_metric["token_id"]),
                    "replacement": None if resample else pick["text"],
                    "replacement_ids": [] if resample else [int(pick["token_id"])],
                },
                previous_turns=turns,
            )
        )
    except ModelChanged:
        # ``turns`` is still the whole conversation, old response included.
        yield idle_state(prompt_text, turns, BRANCH_MODEL_CHANGED, clear_tokens=True)
    except SteeringError as error:
        yield idle_state(
            prompt_text, turns, failure_status("Could not branch", str(error)), clear_tokens=True
        )


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
        return Frame(
            UNDO_OUTPUT_NAMES,
            chatbot=messages,
            turns=turns,
            strip=transcript_update(turns, scale_name),
            status="There is nothing to undo.",
            **send_stop_values(False),
        )

    remaining = turns[:position]
    messages, _ = display_messages(remaining)
    # The selected-token details describe the response being removed, so they
    # go with it, exactly as Clear resets them. So do the prompt tokens, the
    # charts and the export: all of them measure the exchange that just left.
    images = list(turns[position].get("images") or [])
    return Frame(
        UNDO_OUTPUT_NAMES,
        prompt=turns[position]["content"],
        attachments=images,
        attachment_strip=picture_strip(images),
        chatbot=messages,
        turns=remaining,
        status="Removed the last exchange.",
        **send_stop_values(False),
        **panel_reset(remaining, scale_name),
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
    # Clear takes the conversations in the list and leaves the archive be.
    listed = [name for name in forks["branches"] if not branch_archived(forks, name)]
    kept = len(forks["branches"]) - len(listed)
    on_screen = forks["active"] in listed
    if on_screen:
        others = len(listed) - 1
        if not turns and not others:
            return NOTHING_TO_CLEAR, hidden, ""
        if others:
            loss = (
                f"the conversation on screen and {others} other"
                f"{'s' if others != 1 else ''}"
            )
        else:
            loss = "the conversation on screen"
    else:
        # The conversation on screen is an archived one, which Clear keeps.
        if listed == [MAIN_BRANCH] and not forks["branches"][MAIN_BRANCH]:
            return NOTHING_TO_CLEAR, hidden, ""
        loss = f"the {len(listed)} conversation{'s' if len(listed) != 1 else ''} in the list"
    advice = (
        " To remove only this one, archive it from the list."
        if on_screen and forks["active"] != MAIN_BRANCH
        else ""
    )
    if kept:
        advice += f" The {kept} archived conversation{'s are' if kept != 1 else ' is'} kept."
    return (
        gr.skip(),
        gr.update(visible=True),
        f"Clear {loss}? This cannot be undone.{advice}",
    )


def hide_clear_confirm():
    return gr.update(visible=False)


def clear_chat(
    scale_name: str = DEFAULT_COLOR_SCALE,
    forks: dict | None = None,
    turns: list[dict] | None = None,
):
    """Empty everything the conversation owns, short of the archive.

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

    Archived conversations are kept as they are, the one on screen included:
    archiving is how a conversation is put out of Clear's reach. ``turns``
    is the conversation on screen, which is kept that way when it is one.
    """

    reset = panel_reset([], scale_name)
    known = library.as_seen(forks, turns) if turns is not None else copy_forks(forks)
    # Archival can have changed in another page since this page loaded. Keep
    # its latest transcript and metadata; newly created unarchived branches
    # remain outside the clear's original scope.
    latest = library.merge(known, library.read())
    archived = [name for name in latest["branches"] if branch_archived(latest, name)]
    forks = new_forks()
    stamp = branch_stamp()
    forks["updated"] = {
        name: stamp for name in (MAIN_BRANCH, *known["branches"]) if name not in archived
    }
    forks["sampling_updated"] = {
        name: stamp for name in (MAIN_BRANCH, *known["sampling"]) if name not in archived
    }
    for name in archived:
        forks["branches"][name] = latest["branches"][name]
        forks["archived"][name] = True
        for field in ("sampling", "sampling_updated", "archived_updated", "origins", "updated"):
            if name in latest[field]:
                forks[field][name] = latest[field][name]
    return Frame(
        CLEAR_OUTPUT_NAMES,
        chatbot=[],
        turns=[],
        status=("Conversations in the list cleared. Archived conversations kept."
                if archived else "Every conversation cleared."),
        **send_stop_values(False),
        **reset,
        forks=forks,
        conversation_list=conversation_list_update(forks, []),
        clear_confirm=gr.update(visible=False),
    )
