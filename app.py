"""Chatlab interface for chatting with and inspecting model tokens."""

from __future__ import annotations

import contextlib
import html
import os
import random
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio.utils import get_upload_folder

import charts
from conversation import (
    copy_turns,
    display_messages,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    split_reasoning,
    to_json,
    user_index_at_or_before,
)
from model_runtime import (
    MODEL_WEIGHTS,
    PROMPT_SCORE_LIMIT,
    CacheStatus,
    ModelManager,
    cache_status,
    format_bytes,
)
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
    UNSCORED_BEYOND_LIMIT,
    category_for,
    summarize,
)
from trace_export import build_trace, write_trace_export


DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"
MANAGER = ModelManager()
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


def status_card(title: str, detail: str, tone: str = "neutral") -> str:
    icon = {"success": "●", "error": "●", "working": "◌"}.get(tone, "○")
    return f"### {icon} {title}\n\n{detail}"


def describe_missing(status: CacheStatus) -> str:
    """``config.json and the model weights are missing``, for a card."""

    names = [
        "the model weights" if name == MODEL_WEIGHTS else f"`{name}`"
        for name in status.missing_files
    ]
    if len(names) > 3:
        names = names[:2] + [f"{len(names) - 2} more weight files"]
    if len(names) == 1:
        return f"{names[0]} {'are' if names[0] == 'the model weights' else 'is'} missing"
    return f"{', '.join(names[:-1])} and {names[-1]} are missing"


def describe_cache(model_id: str, status: CacheStatus) -> tuple[str, str]:
    """Title and detail for the card shown while a download starts.

    The cases a reader can tell apart from the outside - nothing on disk, a
    cut-off download, a cache holding only some of the files, and a finished
    one - each get their own wording, so "Downloading" never hides that the
    files were already here.
    """

    name = f"`{model_id.strip()}`"
    if status.partial_files:
        files = "file" if status.partial_files == 1 else "files"
        return (
            "Resuming download",
            f"{name} was cut off part way: {status.partial_files} weight "
            f"{files} ({format_bytes(status.partial_bytes)}) are only partly "
            "on disk. Only the missing bytes are fetched.",
        )
    if status.missing_files:
        return (
            "Resuming download",
            f"{name} is only partly on disk ({format_bytes(status.cached_bytes)} "
            f"cached): {describe_missing(status)}. Only the missing files are "
            "fetched.",
        )
    if status.present:
        return (
            "Checking cached model",
            f"{name} is already in the Hugging Face cache "
            f"({format_bytes(status.cached_bytes)}). Checking for missing or "
            "updated files; nothing is downloaded twice.",
        )
    return (
        "Downloading model",
        f"Fetching {name} into the Hugging Face cache. Nothing is cached yet, "
        "so this is a full download and may take a while.",
    )


def describe_fetched(before: CacheStatus, after: CacheStatus, elapsed: float) -> str:
    fetched = after.total_bytes - before.total_bytes
    if fetched <= 0:
        return f"Already up to date; nothing new was fetched ({elapsed:.1f} seconds)."
    if before.present:
        return (
            f"Fetched the remaining {format_bytes(fetched)} in {elapsed:.1f} seconds."
        )
    return f"Fetched {format_bytes(fetched)} in {elapsed:.1f} seconds."


def download_model(model_id: str, hf_token: str):
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except ValueError as error:
        yield status_card("Download failed", html.escape(str(error)), "error")
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path = MANAGER.download(model_id, hf_token)
    except Exception as error:
        yield status_card("Download failed", html.escape(str(error)), "error")
        return

    elapsed = time.monotonic() - started
    fetched = describe_fetched(before, cache_status(model_id), elapsed)
    yield status_card(
        "Download complete",
        f"{fetched} `{model_id.strip()}` is cached in `{path}`. "
        "Use **Load cached** when ready.",
        "success",
    )


def download_and_load_model(model_id: str, hf_token: str):
    started = time.monotonic()
    try:
        before = cache_status(model_id)
    except ValueError as error:
        yield status_card("Model setup failed", html.escape(str(error)), "error")
        return
    yield status_card(*describe_cache(model_id, before), "working")
    try:
        path = MANAGER.download(model_id, hf_token)
        fetched = describe_fetched(
            before, cache_status(model_id), time.monotonic() - started
        )
        yield status_card(
            "Loading model",
            f"{fetched} Moving `{model_id.strip()}` onto the best available device…",
            "working",
        )
        device = MANAGER.load(model_id, path)
    except Exception as error:
        yield status_card("Model setup failed", html.escape(str(error)), "error")
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Model ready",
        f"`{model_id.strip()}` is loaded on **{device}** ({elapsed:.1f} seconds total).",
        "success",
    )


def load_cached_model(model_id: str):
    name = f"`{model_id.strip()}`"
    yield status_card("Finding cached model", f"Looking for {name} locally…", "working")
    try:
        status = cache_status(model_id)
    except ValueError as error:
        yield status_card("Could not load cached model", html.escape(str(error)), "error")
        return
    if status.partial_files:
        files = "file" if status.partial_files == 1 else "files"
        yield status_card(
            "Download incomplete",
            f"{name} was cut off part way: {status.partial_files} weight {files} "
            f"({format_bytes(status.partial_bytes)}) are only partly on disk. "
            "Use **Download and load** to fetch the rest.",
            "error",
        )
        return
    if status.missing_files:
        yield status_card(
            "Download incomplete",
            f"{name} is only partly on disk ({format_bytes(status.cached_bytes)} "
            f"cached): {describe_missing(status)}. "
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
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=model_id.strip(), local_files_only=True))
        yield status_card(
            "Loading model",
            f"Loading {format_bytes(status.cached_bytes)} of cached files from `{path}`…",
            "working",
        )
        device = MANAGER.load(model_id, path)
    except Exception as error:
        yield status_card(
            "Could not load cached model", html.escape(str(error)), "error"
        )
        return
    yield status_card(
        "Model ready", f"{name} is loaded on **{device}**.", "success"
    )


def unload_model():
    if not MANAGER.loaded:
        return status_card("No model loaded", "There is nothing to unload.")
    MANAGER.unload()
    return status_card("Model unloaded", "Model memory has been released.", "success")


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

    index = event.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    try:
        metric = metrics[int(index)]
    except (IndexError, TypeError, ValueError):
        return "That token is no longer available. Generate another response.", []

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

    summary = (
        f"### {where} {metric['position']}: `{token_repr}`\n\n"
        f"- **Raw rank:** {metric['raw_rank']:,}\n"
        f"- **Raw model probability:** {metric['raw_probability']:.5%}\n"
        f"- **Actual sampling probability:** {metric['sampling_probability']:.5%}\n"
        f"- **Surprise:** {metric['surprise_bits']:.2f} bits\n"
        f"- **Distribution entropy:** {metric['entropy_bits']:.2f} bits\n"
        f"- **Top-1 margin:** {metric['top1_margin']:.2%} between the model's first and second choice\n"
        f"- **Sampling shift:** {metric['sampling_shift_bits']:+.2f} bits versus the raw model\n"
        f"- **Probability mass above it:** {metric['probability_mass_above']:.2%}\n"
        f"- **Token ID:** {metric['token_id']:,}"
    )
    rows = [
        [candidate["token_id"], repr(candidate["text"]), candidate["probability"]]
        for candidate in metric["top_candidates"]
    ]
    return summary, rows


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


# ---------------------------------------------------------------- generation


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
)


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


def stop_generation(turns: list[dict] | None):
    """Finish the turn that the cancelled generator left behind.

    Gradio closes ``generate_reply`` at its last yield, so nothing else ever
    finalizes that turn.
    """

    turns = copy_turns(turns)
    kept = finalize_partial(turns)
    messages, _ = display_messages(turns)
    return (
        messages,
        turns,
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
    describing the visible text - an edited assistant reply. Both strips go
    together there: they carry one shared stamp, so re-stamping the response
    strip alone would silently stop the prompt strip's clicks from publishing.
    """

    messages, _ = display_messages(turns)
    panels = (
        cleared_strips(scale_name)
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
    )


BUSY_STATUS = "A response is already generating. Press Stop first."


def busy_state():
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
    """

    return (
        (gr.skip(),) * 5
        + (BUSY_STATUS,)
        + (gr.skip(),) * (len(CHAT_OUTPUT_NAMES) - 6)
    )


def generate_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Stream one assistant reply for ``turns``, which must end with a user turn.

    The generation slot is reserved here, before the first frame is published,
    because this is the first moment a handler is committed to generating. The
    MANAGER.busy checks in chat(), regenerate_from() and edit_message() are an
    early exit, not the guard: between such a check and the model lock that
    generate() takes sits the "Generating…" yield, and Gradio does not resume a
    handler until it has serialized that frame and sent it to the browser. A
    second click arriving inside that round trip used to sail past a manager
    that looked idle and overwrite the conversation from its stale snapshot.
    """

    if not MANAGER.reserve_generation():
        yield busy_state()
        return

    try:
        yield from _stream_reply(
            turns,
            prompt_text,
            system_prompt,
            keep_reasoning,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            scale_name,
        )
    finally:
        # Every exit runs this: a finished stream, a failure, and - the one
        # that matters - cancellation, where Gradio throws GeneratorExit in at
        # whichever yield the stream is parked on. Leaving the slot reserved
        # there would wedge the app: Send would refuse forever.
        MANAGER.release_generation()


def _stream_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
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

    pending = make_turn("assistant", "", "")
    pending["reasoning_closed"] = True
    turns.append(pending)

    def snapshot(
        highlight,
        metrics,
        status,
        busy=True,
        reset_details=False,
        prompt_panel=None,
        charts_panel=None,
        trace=None,
    ):
        """One frame of the stream.

        ``reset_details`` belongs to the first frame only. That frame empties
        the strip, so a token selected in the previous response is gone and its
        probabilities must go with it. Later frames only append to the strip, so
        a token picked mid-stream stays valid and its details are left alone.

        ``prompt_panel`` and ``charts_panel`` are skipped on most frames. The
        prompt tokens are all measured before the first one is generated, so
        they are published once and never change; the charts redraw in batches
        because rebuilding an SVG per token is wasted work.
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
            highlight,
            (generation, metrics),
            status,
            used_seed,
            *send_stop_buttons(busy),
            NO_TOKEN_SELECTED if reset_details else gr.skip(),
            [] if reset_details else gr.skip(),
            prompt_strip,
            prompt_metrics,
            prompt_note,
            summary_panel,
            surprise_panel,
            gr.skip() if trace is None else trace,
        )

    # The opening frame empties everything the previous response left behind,
    # the export included: a trace kept here would still be downloadable while
    # a different response was streaming in above it.
    yield snapshot(
        strip_update([], scale_name, RESPONSE_STRIP_LABEL),
        [],
        "Generating…",
        reset_details=True,
        prompt_panel=(strip_update([], scale_name), (generation, []), ""),
        charts_panel=(charts.summary_tiles({}), charts.EMPTY_CHART),
        trace={},
    )

    started = time.monotonic()
    raw_text = ""
    # Reasoning templates end the prompt with the opening <think> marker, so the
    # generated text never carries one. Only the runtime can tell us that.
    prefilled = False
    highlight: list[tuple[str, str]] = []
    metrics: list[dict] = []
    status = "The model produced no tokens."
    first = True

    stream = MANAGER.generate(
        request,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_new_tokens=int(max_new_tokens),
        seed=used_seed,
        analyze_prompt=bool(analyze_prompt),
    )

    try:
        # closing() releases the model lock the moment the Stop button cancels
        # this event and Gradio closes the outer generator.
        with contextlib.closing(stream):
            for update in stream:
                raw_text = update.text
                prefilled = update.reasoning_prefilled
                reasoning, answer, closed = split_reasoning(
                    raw_text, streaming=True, reasoning_prefilled=prefilled
                )
                pending["reasoning"] = reasoning
                pending["content"] = answer
                pending["reasoning_closed"] = closed
                highlight = strip_value(update.metrics, scale_name)
                metrics = list(update.metrics)
                status = generation_progress(len(metrics), started, used_seed)
                prompt_panel = None
                if first:
                    # Every prompt token is measured before the first response
                    # token exists, so this is published once and never
                    # changes. It shares the response strip's stamp: the two
                    # are replaced together, and a click on either has to
                    # match the stamp the pair was drawn with.
                    prompt_metrics = list(update.prompt_metrics)
                    prompt_panel = (
                        strip_update(prompt_metrics, scale_name),
                        (generation, prompt_metrics),
                        prompt_note_text(
                            len(prompt_metrics), update.prompt_note, "prompt"
                        ),
                    )
                yield snapshot(
                    highlight,
                    metrics,
                    status,
                    prompt_panel=prompt_panel,
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
    except Exception as error:
        # The diagnostic only goes to the status line. Storing it as the
        # assistant turn would feed the failure back to the model next turn.
        reasoning, answer, _ = split_reasoning(raw_text, reasoning_prefilled=prefilled)
        pending["reasoning"] = reasoning
        pending["content"] = answer
        finalize_partial(turns)
        # A failed response is not a response to export, so the trace the
        # opening frame emptied stays empty.
        yield snapshot(highlight, metrics, f"Generation failed: {error}", busy=False)
        return

    reasoning, answer, _ = split_reasoning(raw_text, reasoning_prefilled=prefilled)
    pending["reasoning"] = reasoning
    pending["content"] = answer
    # A generation can succeed and still leave nothing renderable behind: the
    # first sampled token is a hidden EOS, the model emits only whitespace,
    # which split_reasoning() strips away, or it opens and closes a reasoning
    # block without writing in it. Publishing that turn would draw a blank
    # bubble in display_messages() that model_messages() skips, so the visible
    # conversation and the model's would disagree - the UI would show a reply
    # the model never sees. finalize_partial() is what the failure and
    # cancellation paths already use for exactly this, so success uses it too:
    # it closes the reasoning block when the turn is worth keeping and drops
    # the turn when it holds neither answer nor reasoning. Dropping it leaves
    # the user turn without a reply, which is the honest shape - no assistant
    # bubble is drawn, so both transcripts agree that no reply exists.
    kept = finalize_partial(turns)
    trace = (
        build_trace(
            model_id=MANAGER.model_id,
            messages=request,
            response=raw_text,
            sampling={
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_new_tokens": int(max_new_tokens),
                "seed": used_seed,
            },
            metrics=metrics,
        )
        if kept and metrics
        else {}
    )
    if trace:
        status = f"{status} Exports are ready."
    yield snapshot(
        highlight,
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
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    if MANAGER.busy:
        # Before anything else, including the checks below: every other exit
        # from this function writes the conversation back, and while another
        # generation is streaming that write is a stale overwrite.
        yield busy_state()
        return

    turns = copy_turns(turns)
    message = (prompt_text or "").strip()
    if not message:
        yield idle_state(prompt_text, turns, "Enter a message first.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    turns.append(make_turn("user", message))
    yield from generate_reply(
        turns,
        "",
        system_prompt,
        keep_reasoning,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
        analyze_prompt,
        scale_name,
    )


def regenerate_from(
    position: int | None,
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Throw away everything after the user turn at ``position`` and reply again."""

    if MANAGER.busy:
        # Covers Retry and the chatbot's own retry button, which reach a
        # generation only through here.
        yield busy_state()
        return

    turns = copy_turns(turns)
    if position is None:
        yield idle_state(prompt_text, turns, "There is nothing to retry.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    yield from generate_reply(
        turns[: position + 1],
        prompt_text,
        system_prompt,
        keep_reasoning,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
        analyze_prompt,
        scale_name,
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
    # The color scale is the last of the settings a generation is given, and
    # this handler needs it for the one path that clears the strips itself.
    scale_name = settings[-1] if settings else DEFAULT_COLOR_SCALE
    if MANAGER.busy:
        # Not just the branch that regenerates: editing an assistant turn
        # rewrites the conversation on its own, from the same stale snapshot.
        yield busy_state()
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
        if not MANAGER.reserve_generation():
            yield busy_state()
            return
        try:
            turns[position] = edited_turn
            # The ranks and probabilities on screen describe the text the model
            # generated, not what the user just typed over it.
            yield idle_state(
                prompt_text,
                turns,
                "Assistant message edited.",
                clear_tokens=True,
                scale_name=scale_name,
            )
        finally:
            MANAGER.release_generation()
        return

    edited = new_value.strip()
    if not edited:
        # An empty user turn is skipped by model_messages(), which would leave
        # the request with no user message at all.
        yield idle_state(prompt_text, turns, "A user message cannot be empty.")
        return

    if not MANAGER.loaded:
        # regenerate_from() would refuse too, but only after the truncation
        # below had already thrown away every later turn for a reply that is
        # never generated.
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    turns = turns[: position + 1]
    turns[position]["content"] = edited
    yield from regenerate_from(position, prompt_text, turns, *settings)


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
            gr.skip(),
            gr.skip(),
            "There is nothing to undo.",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 6,
        )

    remaining = turns[:position]
    messages, _ = display_messages(remaining)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
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
    )


def undo_last(turns, scale_name: str = DEFAULT_COLOR_SCALE):
    return undo_from(last_user_index(turns), turns, scale_name)


def undo_message(event: gr.UndoData, turns, scale_name: str = DEFAULT_COLOR_SCALE):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    return undo_from(position, turns, scale_name)


def clear_chat(scale_name: str = DEFAULT_COLOR_SCALE):
    """Empty everything the conversation owns.

    Clear cancels a running generation (see ``cancels`` on its listener), and a
    cancelled ``generate_reply`` never reaches its final yield, so this has to
    restore the Send button itself exactly as Stop does.
    """

    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    return (
        [],
        [],
        strip,
        metrics,
        "Conversation cleared.",
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED,
        [],
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


# --------------------------------------------------------------- save / load


def save_conversation(turns, system_prompt):
    if not turns:
        return gr.update(value=None, visible=False), "There is nothing to save yet."

    # Gradio only serves files it created or was told to allow, so the saved
    # conversation has to live inside its upload folder.
    directory = Path(get_upload_folder()) / "chatlab-conversations"
    directory.mkdir(parents=True, exist_ok=True)
    # The timestamp only resolves to the second, and every session shares this
    # upload folder, so a random suffix keeps two saves from landing on the same
    # path and silently overwriting each other's download.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"conversation-{stamp}-{uuid4().hex[:8]}.json"
    path.write_text(to_json(turns, system_prompt=system_prompt), encoding="utf-8")
    return (
        gr.update(value=str(path), visible=True),
        f"Saved {len(turns)} message{'s' if len(turns) != 1 else ''}.",
    )


def load_conversation(file_path, turns, scale_name: str = DEFAULT_COLOR_SCALE):
    """Replace the conversation with a saved one.

    A failed load keeps the conversation already on screen, so a bad file
    cannot wipe it, and leaves the token panel describing it alone. Loading
    cancels any generation still running, so the buttons are restored here for
    the same reason Clear restores them: a cancelled generator never reaches
    its final yield. For the same reason the kept conversation has to be
    finalized like Stop does - the cancelled generator left its last turn with
    a pending reasoning block, which would spin for the rest of the session,
    or empty if the cancel landed before the first token.
    """

    def keep_current(status):
        """Return the conversation the cancelled generator left behind."""

        kept = copy_turns(turns)
        finalize_partial(kept)
        messages, _ = display_messages(kept)
        return (
            messages,
            kept,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            status,
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 6,
        )

    if not file_path:
        return keep_current("No file chosen.")
    try:
        loaded, system_prompt = from_json(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return keep_current(f"Could not load that file: {error}")

    # A successful load replaces the conversation wholesale, so whatever the
    # cancelled generator left behind goes with it and needs no finalizing.
    turns = loaded
    messages, _ = display_messages(turns)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_strips(
        scale_name
    )
    # The selected token described a response from the conversation being
    # replaced, so it goes with it, exactly as Clear and Undo reset it. The
    # charts and the export measured that response too, and a loaded
    # conversation has no measurements of its own to put in their place.
    return (
        messages,
        turns,
        system_prompt,
        strip,
        metrics,
        f"Loaded {len(turns)} message{'s' if len(turns) != 1 else ''}.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


# ---------------------------------------------------------------- score text


def score_text(
    context: str,
    text: str,
    use_chat_template: bool,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Measure text the model did not write, and put it in the same panel."""

    skip = gr.skip()
    if not MANAGER.loaded:
        return (skip,) * 7 + ("Download and load a model first.", skip, skip)

    try:
        result = MANAGER.score_text(
            text, context=context or "", use_chat_template=bool(use_chat_template)
        )
    except Exception as error:
        return (skip,) * 7 + (f"Could not score that text: {error}", skip, skip)

    summary = summarize(result.metrics)
    status = (
        f"Scored {summary['token_count']:,} tokens. "
        f"Perplexity {summary['perplexity']:,.1f}."
    )
    # What was scored comes before how exactly it was scored: the template
    # caveat says which passage the numbers describe, the seam caveat says how
    # sure their first token is.
    if result.chat_template_missing:
        status = f"{status} {TEMPLATE_CAVEAT}"
    if not result.seam_verified:
        status = f"{status} {SEAM_CAVEAT}"
    # Both strips are replaced, so they take one shared stamp - and that stamp
    # is what drops a click made against the response they overwrite.
    generation = new_metrics_generation()
    return (
        strip_update(result.metrics, scale_name, "Scored tokens — click one"),
        stamped(result.metrics, generation),
        strip_update(result.context_metrics, scale_name),
        stamped(result.context_metrics, generation),
        prompt_note_text(len(result.context_metrics), "", "context"),
        charts.summary_tiles(summary),
        charts.surprise_chart(result.metrics, title="Surprise per scored token"),
        status,
        NO_TOKEN_SELECTED,
        [],
    )


CSS = """
.gradio-container { max-width: 1500px !important; }
#hero { padding: 0.5rem 0 0.2rem; }
#hero h1 { font-size: 2.1rem; margin-bottom: 0.25rem; }
#model-status { min-height: 128px; }
#token-strip { min-height: 150px; }
#token-strip span, #prompt-strip span { cursor: pointer; border-radius: 5px; }
/* Token fills are light in both themes, so their ink is pinned dark. */
#token-strip .textspan.hl, #prompt-strip .textspan.hl,
#token-strip .category-label, #prompt-strip .category-label { color: #0b0b0b; }
.footer-note { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.scale-caption { color: var(--body-text-color-subdued); font-size: 0.85rem; }

.viz-root {
  --viz-ink: #0b0b0b;
  --viz-muted: #898781;
  --viz-grid: #e1e0d9;
  --viz-axis: #c3c2b7;
  --viz-line: #2a78d6;
  --viz-band: #cde2fb;
  margin: 0;
  font-family: var(--font, system-ui, -apple-system, "Segoe UI", sans-serif);
}
.dark .viz-root {
  --viz-ink: #ffffff;
  --viz-muted: #898781;
  --viz-grid: #2c2c2a;
  --viz-axis: #383835;
  --viz-line: #3987e5;
  --viz-band: #1c5cab;
}
.viz-root svg { width: 100%; height: auto; display: block; }
.viz-title { color: var(--viz-ink); font-size: 0.9rem; font-weight: 600; padding: 0 0 0.2rem; }
.viz-sub { color: var(--viz-muted); font-weight: 400; font-size: 0.8rem; margin-left: 0.4rem; }
.viz-grid { stroke: var(--viz-grid); stroke-width: 1; }
.viz-axis { stroke: var(--viz-axis); stroke-width: 1; }
.viz-band { fill: var(--viz-band); opacity: 0.55; stroke: none; }
.viz-line { fill: none; stroke: var(--viz-line); stroke-width: 2; stroke-linejoin: round; }
.viz-peak-dot { fill: var(--viz-line); stroke: var(--body-background-fill); stroke-width: 2; }
.viz-peak-label, .viz-tick { fill: var(--viz-muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.viz-hit { fill: transparent; }
.viz-empty, .viz-note { color: var(--body-text-color-subdued); font-size: 0.85rem; padding: 0.4rem 0; }
.viz-tiles { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.viz-tile {
  flex: 1 1 5.5rem; padding: 0.45rem 0.6rem; border-radius: 8px;
  background: var(--background-fill-secondary);
}
.viz-value { color: var(--viz-ink); font-size: 1.25rem; line-height: 1.2; }
.viz-label { color: var(--viz-muted); font-size: 0.72rem; text-transform: lowercase; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Chatlab", css=CSS, theme=gr.themes.Soft()) as demo:
        conversation_state = gr.State([])
        metrics_state = gr.State(empty_metrics())
        prompt_metrics_state = gr.State(empty_metrics())
        trace_state = gr.State({})

        gr.Markdown(
            "# Chatlab\nChat with an open model and see exactly how likely every generated token was.",
            elem_id="hero",
        )

        with gr.Accordion("Model setup", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    model_id = gr.Textbox(
                        value=os.environ.get("OLMO_MODEL_ID", DEFAULT_MODEL),
                        label="Hugging Face model ID",
                        placeholder="organization/model-name",
                        info="The default OLMo 3 7B model is about 15 GB in full precision.",
                    )
                    hf_token = gr.Textbox(
                        label="Hugging Face token (optional)",
                        type="password",
                        placeholder="Only needed for gated or private models",
                    )
                    with gr.Row():
                        download_load_button = gr.Button(
                            "Download and load", variant="primary"
                        )
                        download_button = gr.Button("Download only")
                        cached_button = gr.Button("Load cached")
                        unload_button = gr.Button("Unload")
                with gr.Column(scale=2):
                    model_status = gr.Markdown(
                        status_card(
                            "No model loaded",
                            "Enter a model ID, then download and load it. Files are kept in your normal Hugging Face cache.",
                        ),
                        elem_id="model-status",
                    )

        with gr.Accordion("System prompt and reasoning", open=False):
            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="You are a careful assistant that answers concisely.",
                lines=3,
                info="Sent as a system message ahead of the conversation. Leave empty to use the model's default behavior.",
            )
            keep_reasoning = gr.Checkbox(
                value=False,
                label="Send previous reasoning back to the model",
                info="Off by default. Think models write a fresh reasoning block each turn, so replaying old ones burns context and usually hurts the next answer.",
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(
                            type="messages",
                            label="Conversation",
                            height=560,
                            editable="all",
                            placeholder="Load a model, then start a conversation.",
                        )
                        prompt = gr.Textbox(
                            label="Message",
                            placeholder="Ask OLMo something…",
                            lines=3,
                        )
                        with gr.Row():
                            send_button = gr.Button("Send", variant="primary")
                            stop_button = gr.Button(
                                "Stop", variant="stop", visible=False
                            )
                            retry_button = gr.Button("🔁 Retry")
                            undo_button = gr.Button("↩️ Undo last")
                            clear_button = gr.Button("🗑️ Clear")
                        with gr.Row():
                            save_button = gr.Button("💾 Save conversation")
                            load_upload = gr.UploadButton(
                                "📂 Load conversation",
                                file_types=[".json"],
                                type="filepath",
                            )
                        saved_file = gr.File(
                            label="Saved conversation",
                            visible=False,
                            interactive=False,
                        )
                        generation_status = gr.Markdown("Ready.")
                        with gr.Accordion("Export full metric trace", open=False):
                            with gr.Row():
                                gr.DownloadButton(
                                    "Download JSON",
                                    value=lambda trace: write_trace_export(
                                        trace, "json"
                                    ),
                                    inputs=trace_state,
                                    size="sm",
                                )
                                gr.DownloadButton(
                                    "Download CSV",
                                    value=lambda trace: write_trace_export(trace, "csv"),
                                    inputs=trace_state,
                                    size="sm",
                                )
                            gr.Markdown(
                                "Exports include every token metric and all recorded "
                                "alternatives for the latest completed response.",
                                elem_classes=["footer-note"],
                            )

                    with gr.Tab("Score text"):
                        gr.Markdown(
                            "Measure text the model did not write. One forward pass "
                            "gives every token the same rank, probability, surprise, "
                            "and entropy the chat view shows."
                        )
                        score_context = gr.Textbox(
                            label="Context (optional)",
                            placeholder="Text that comes before the part you want scored.",
                            lines=3,
                        )
                        use_chat_template = gr.Checkbox(
                            value=False,
                            label="Treat the context as a chat message",
                            info=(
                                "Wraps the context in the model's chat template, so the "
                                "scored text is measured as a reply. Models without a "
                                "chat template score the context as plain text, and say so."
                            ),
                        )
                        score_input = gr.Textbox(
                            label="Text to score",
                            placeholder="Paste the text you want measured…",
                            lines=8,
                        )
                        score_button = gr.Button("Score text", variant="primary")
                        score_status = gr.Markdown("Nothing scored yet.")

            with gr.Column(scale=2):
                gr.Markdown("## Under the hood")
                color_scale = gr.Dropdown(
                    choices=list(COLOR_SCALES),
                    value=DEFAULT_COLOR_SCALE,
                    label="Color tokens by",
                )
                scale_caption = gr.Markdown(
                    COLOR_SCALES[DEFAULT_COLOR_SCALE].caption,
                    elem_classes=["scale-caption"],
                )
                token_strip = gr.HighlightedText(
                    label=RESPONSE_STRIP_LABEL,
                    color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                    show_legend=True,
                    combine_adjacent=False,
                    elem_id="token-strip",
                )
                token_detail = gr.Markdown(NO_TOKEN_SELECTED)
                alternatives = gr.Dataframe(
                    headers=["Token ID", "Token", "Raw probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Most likely alternatives",
                )
                summary_panel = gr.HTML(charts.summary_tiles({}))
                surprise_panel = gr.HTML(charts.EMPTY_CHART)
                with gr.Accordion("Prompt and context tokens", open=False):
                    prompt_note = gr.Markdown("", elem_classes=["scale-caption"])
                    prompt_strip = gr.HighlightedText(
                        label="Prompt tokens — click one",
                        color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                        show_legend=True,
                        combine_adjacent=False,
                        elem_id="prompt-strip",
                    )

        with gr.Accordion("Sampling and analysis controls", open=False):
            with gr.Row():
                temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
            with gr.Row():
                max_new_tokens = gr.Slider(
                    1, 8192, value=1024, step=1, label="Maximum new tokens"
                )
                seed = gr.Number(
                    value=42,
                    precision=0,
                    minimum=0,
                    label="Random seed",
                    info="Updated after each response so you can reproduce it.",
                )
                randomize_seed = gr.Checkbox(
                    value=True,
                    label="🎲 New seed each response",
                    info="Turn off to lock the seed and reproduce a response exactly.",
                )
                analyze_prompt = gr.Checkbox(
                    value=True,
                    label="Measure prompt tokens",
                    info="Scores every prompt token during the same pass that warms the cache.",
                )

        gr.Markdown(
            "Rank and raw probability come from the unmodified model distribution. "
            "Sampling probability includes temperature, top-k, and top-p. Quantized models may produce slightly different ranks.",
            elem_classes=["footer-note"],
        )

        download_button.click(download_model, [model_id, hf_token], model_status)
        download_load_button.click(
            download_and_load_model, [model_id, hf_token], model_status
        )
        cached_button.click(load_cached_model, model_id, model_status)
        unload_button.click(unload_model, outputs=model_status)

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            color_scale,
        ]
        chat_inputs = [prompt, conversation_state, *settings_inputs]
        # The order every generation handler publishes in; see
        # CHAT_OUTPUT_NAMES.
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            seed,
            send_button,
            stop_button,
            token_detail,
            alternatives,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]
        undo_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            token_detail,
            alternatives,
            send_button,
            stop_button,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]

        running = [
            send_button.click(chat, chat_inputs, chat_outputs),
            prompt.submit(chat, chat_inputs, chat_outputs),
            retry_button.click(retry_last, chat_inputs, chat_outputs),
            chatbot.retry(retry_message, chat_inputs, chat_outputs),
            chatbot.edit(edit_message, chat_inputs, chat_outputs),
        ]

        stop_button.click(
            stop_generation,
            inputs=conversation_state,
            outputs=[
                chatbot,
                conversation_state,
                send_button,
                stop_button,
                generation_status,
            ],
            cancels=running,
        )

        # Undo, Clear and Load all replace or truncate the conversation, so
        # each has to stop the generator first: a surviving generate_reply
        # would write its own snapshot of the in-progress turns back into the
        # chatbot and the state, resurrecting what was just removed. Send,
        # Retry and Edit are exempt because they *are* the generation - they
        # re-enter generate_reply, and they are what everything else cancels.
        # They cannot be made to cancel each other either: Gradio captures a
        # listener's inputs when the request is queued, so the survivor would
        # rebuild the conversation from a snapshot taken before the cancelled
        # run wrote anything. A shared concurrency group has the same flaw - it
        # only delays the stale handler. Each of them refuses outright instead
        # while MANAGER.busy (see busy_state).
        undo_button.click(
            undo_last,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
        )
        chatbot.undo(
            undo_message,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
        )
        clear_button.click(
            clear_chat,
            inputs=color_scale,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                generation_status,
                send_button,
                stop_button,
                token_detail,
                alternatives,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
            ],
            cancels=running,
        )

        save_button.click(
            save_conversation,
            [conversation_state, system_prompt],
            [saved_file, generation_status],
        )
        load_upload.upload(
            load_conversation,
            [load_upload, conversation_state, color_scale],
            [
                chatbot,
                conversation_state,
                system_prompt,
                token_strip,
                metrics_state,
                generation_status,
                token_detail,
                alternatives,
                send_button,
                stop_button,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
            ],
            cancels=running,
        )

        score_button.click(
            score_text,
            [score_context, score_input, use_chat_template, color_scale],
            [
                token_strip,
                metrics_state,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                score_status,
                token_detail,
                alternatives,
            ],
        )
        color_scale.change(
            recolor,
            [metrics_state, prompt_metrics_state, color_scale],
            [token_strip, prompt_strip, scale_caption],
        )

        token_strip.select(
            inspect_token,
            inputs=metrics_state,
            outputs=[token_detail, alternatives],
        )
        prompt_strip.select(
            inspect_token,
            inputs=prompt_metrics_state,
            outputs=[token_detail, alternatives],
        )

    return demo


if __name__ == "__main__":
    conductor_port = os.environ.get("CONDUCTOR_PORT")
    build_app().queue(default_concurrency_limit=1).launch(
        inbrowser=conductor_port is None,
        server_port=int(conductor_port) if conductor_port else None,
    )
