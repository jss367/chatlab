"""The Compare tab: fill two slots, then read one run against the other.

Nothing here is a conversation. A slot holds one run and its whole
configuration, and the two are only ever read side by side; see
:mod:`compare` for what the comparison itself is and is not.

Only one model is ever in memory, so the two slots are filled one after the
other rather than together. That is what makes comparing two models - or one
model at two precisions - work at all: fill A, load the other model, fill B.
Each slot keeps the load it was filled under, so the pair says which weights
answered even after both have been unloaded.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

import charts
import compare
from conversation import make_turn, model_messages
from model_runtime import LOADING, ModelChanged
from steering import SteeringError, compact as compact_steering, from_controls
from trace_export import write_private_text
from ui import runtime
from ui.common import failure_status
from ui.generation import resolve_seed


COMPARE_EMPTY = (
    "Fill both slots to compare them. A slot keeps the model, the settings "
    "and every token's measurements, so the two can be filled minutes and a "
    "model load apart."
)

COMPARE_NO_MODEL = "Download and load a model first."

COMPARE_BUSY = "Wait for the response to finish before filling a slot."

# A load has the model instead: there is no response to wait for, and the run
# would be measured on weights on their way out.
COMPARE_LOADING = "Wait for the model to finish loading before filling a slot."

COMPARE_NO_PROMPT = "Write a prompt for the model to answer."

COMPARE_NO_TEXT = "Paste the text both runs should measure."


def mode_controls(mode: str):
    """Relabel the boxes for the way the slots are being filled."""

    reply = mode == compare.REPLY
    return (
        gr.update(
            label="Prompt for both runs" if reply else "Context (optional)",
            placeholder=(
                "The message both runs answer."
                if reply
                else "Text that comes before the part being measured."
            ),
        ),
        gr.update(visible=not reply),
        gr.update(visible=not reply),
    )


def _running(busy: bool):
    """The three buttons while a slot is being filled, and after."""

    return (
        gr.update(interactive=not busy),
        gr.update(interactive=not busy),
        gr.update(visible=busy),
    )


def fill_slot(
    slot: str,
    mode: str,
    prompt: str,
    measured: str,
    use_chat_template: bool,
    system_prompt: str,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
    thinking_mode: str,
    steering,
    steering_enabled: bool,
    steering_strength: float,
    steering_layer,
):
    """Run the model once and keep the result, its measurements and its settings.

    The generation slot is claimed before memory is looked at, for the reason
    every other view claims it there: a load empties memory before it reads
    the new weights, so a check made first would tell the reader to load a
    model while one was loading. Holding it also keeps a reply from the Chat
    tab landing on the same weights mid-run and being reported as part of
    this one.

    A generator, so that Stop can close it where it stands: Gradio throws
    GeneratorExit into whichever yield is open, the finally gives the
    generation slot back, and the closing() around the run closes the token
    stream and with it the model lock. The slot on screen is left as it was -
    a run stopped half way through is not a measurement of anything.
    """

    skip = gr.skip()

    def refuse(message):
        return skip, message, *_running(False)

    held = runtime.MANAGER.claim_generation()
    if held:
        yield refuse(COMPARE_LOADING if held == LOADING else COMPARE_BUSY)
        return
    result = None
    started = time.monotonic()
    try:
        if not runtime.MANAGER.loaded:
            yield refuse(COMPARE_NO_MODEL)
            return
        published = runtime.MANAGER.loaded_model()
        if published.load_id is None:
            yield refuse(COMPARE_NO_MODEL)
            return
        if mode == compare.REPLY and not (prompt or "").strip():
            yield refuse(COMPARE_NO_PROMPT)
            return
        if mode != compare.REPLY and not (measured or "").strip():
            yield refuse(COMPARE_NO_TEXT)
            return
        try:
            vector = compact_steering(
                from_controls(steering, steering_enabled, steering_strength, steering_layer)
            )
        except (ValueError, TypeError, OverflowError) as error:
            yield refuse(failure_status("Could not apply the steering vector", str(error)))
            return

        yield skip, f"Filling slot {slot}…", *_running(True)
        started = time.monotonic()
        try:
            if mode == compare.REPLY:
                # closing() rather than a bare loop: a Stop landing on the
                # yield below leaves this generator unfinished, and the token
                # stream inside it holds the model lock until it is closed.
                with contextlib.closing(_write_reply(
                    prompt, system_prompt, assistant_prefill, temperature, top_p,
                    top_k, max_new_tokens, seed, randomize_seed, thinking_mode,
                    vector, published,
                )) as writing:
                    for run, tokens in writing:
                        if run is not None:
                            result = run
                            break
                        yield (
                            skip,
                            f"Slot {slot}: {tokens:,} tokens · "
                            f"{time.monotonic() - started:.1f}s",
                            *_running(True),
                        )
            else:
                result = _measure_text(
                    prompt, measured, use_chat_template, vector, published,
                )
        except ModelChanged as error:
            yield refuse(failure_status("The model changed while this ran", str(error)))
            return
        except (SteeringError, ValueError, RuntimeError) as error:
            yield refuse(failure_status(f"Could not fill slot {slot}", str(error)))
            return
    finally:
        # Every exit runs this, cancellation included, or the slot would stay
        # reserved and refuse every reply for the rest of the session.
        runtime.MANAGER.release_generation()

    if result is None or not result["metrics"]:
        yield refuse(f"Slot {slot} was left as it was: the run produced no tokens.")
        return
    result["slot"] = slot
    result["seconds"] = time.monotonic() - started
    status = (
        f"Slot {slot} filled: {len(result['metrics']):,} tokens from "
        f"{result['model_id']} in {result['seconds']:.1f}s."
    )
    yield result, status, *_running(False)


def _write_reply(
    prompt, system_prompt, assistant_prefill, temperature, top_p, top_k,
    max_new_tokens, seed, randomize_seed, thinking_mode, vector, published,
):
    """One reply to one prompt, in a conversation of its own.

    Yields ``(None, tokens so far)`` while the model writes, then the run
    itself once. The caller publishes the counts and keeps the run.
    """

    used_seed = resolve_seed(seed, randomize_seed)
    messages = model_messages([make_turn("user", prompt)], system_prompt=system_prompt)
    text = ""
    metrics: list[dict] = []
    prompt_ids: tuple[int, ...] = ()
    model_id = published.model_id
    stream = runtime.MANAGER.generate(
        messages,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_new_tokens=int(max_new_tokens),
        seed=used_seed,
        # No panel is drawn for the prompt here, and scoring it would cost a
        # softmax per prompt token to reach nothing.
        analyze_prompt=False,
        answer_prefill=assistant_prefill or "",
        thinking_mode=thinking_mode or "default",
        load_id=published.load_id,
        steering=vector,
    )
    # closing() gives the model lock back the moment this generator is closed,
    # which is what a Stop in the caller does.
    with contextlib.closing(stream):
        for update in stream:
            text = update.text
            metrics = list(update.metrics)
            model_id = update.model_id or model_id
            prompt_ids = update.prompt_ids or prompt_ids
            yield None, len(metrics)
    # Seeded with the prompt for the same reason a measurement is seeded with
    # its context: the first token of a reply decodes differently depending on
    # what the model had just read.
    decoded, ends = _decoded_spans(metrics, prompt_ids, text)
    yield {
        "kind": compare.REPLY,
        "model_id": model_id,
        "load_id": published.load_id,
        "device_name": published.device_name,
        "precision": published.precision,
        "prompt": prompt,
        "text": text,
        "metrics": metrics,
        "decoded": decoded,
        "token_ends": ends,
        "settings": {
            "system_prompt": system_prompt or "",
            "assistant_prefill": assistant_prefill or "",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "seed": used_seed,
            "thinking_mode": thinking_mode or "default",
            "steering": vector,
        },
    }, len(metrics)


def _measure_text(context, measured, use_chat_template, vector, published):
    """The same fixed passage, read by whatever is in memory now.

    The system prompt from Settings is deliberately not taken here. A
    measurement pass feeds the context and the passage and nothing else -
    there is no turn for a system message to sit in front of - so recording
    one would have the comparison table report a difference neither run saw.
    The context box is where a measurement's framing goes.
    """

    result = runtime.MANAGER.score_text(
        measured,
        context=context or "",
        use_chat_template=bool(use_chat_template),
        load_id=published.load_id,
        steering=vector,
    )
    metrics = list(result.metrics)
    # The seam between a context and the passage can fall inside a token,
    # which score_text assigns whole to the passage; decoding from the
    # context is what keeps that token's context characters out of the
    # recorded text rather than in one run's and not the other's.
    decoded, ends = _decoded_spans(metrics, result.context_ids, measured)
    return {
        "kind": compare.MEASUREMENT,
        "model_id": published.model_id,
        "load_id": published.load_id,
        "device_name": published.device_name,
        "precision": published.precision,
        "prompt": context or "",
        "text": measured,
        "metrics": metrics,
        "decoded": decoded,
        "token_ends": ends,
        "settings": {
            "use_chat_template": bool(use_chat_template),
            "seam_verified": result.seam_verified,
            "chat_template_missing": result.chat_template_missing,
            "steering": vector,
        },
    }


def _decoded_spans(metrics, context_ids=(), expected: str = "") -> tuple[str, list[int]]:
    """The run's text as the tokenizer really decodes it, and each token's end.

    Decoding is not piecewise. A byte-level tokenizer can split one character
    across two tokens, and each of those decoded on its own is a replacement
    character rather than half of anything, so adding up standalone decodes
    gives neither the run's text nor the right boundaries. Two runs of one
    passage would then look like two different passages wherever the models
    split a character differently, which is exactly where the comparison is
    most worth having.

    ``context_ids`` is everything the model read before the tokens being
    measured, and the decode starts from it rather than from the measured
    tokens alone. What comes before a token changes what it decodes to: a
    SentencePiece tokenizer attaches a word-boundary space to the first
    token of a passage only when something precedes it, and a token sitting
    across the seam between a context and the passage carries characters
    from both. Decoding from the measured tokens alone would put those
    characters in one run's text and not another's, and the alignment would
    call an identical passage divergent at its first character. The context's
    own characters are then cut back off, so what is recorded is the measured
    passage and the offsets are relative to it.

    Where the context ends is taken from ``expected`` - the passage the run
    was actually given - rather than from the length of the context's own
    tokens. A token straddling the seam belongs to the passage, so it is not
    among ``context_ids`` at all, and decoding those alone stops short of the
    characters that token carries: encoding ``"foo"`` + ``"bar"`` can leave a
    context of nothing but the opening marker and one ``"foobar"`` token, and
    a boundary taken from the context's tokens would record ``"foobar"`` as
    the passage. The decoded tail that matches what was given is the seam.
    Where the tail does not match - a tokenizer that does not round-trip its
    input - the context's own decode is the best boundary available.

    ``IncrementalDecoder`` is what the chat stream already uses for this: its
    text always equals a full decode of every token pushed so far, at a cost
    proportional to its cache rather than to the length of the run. It is
    given the manager's hidden token ids, which is the same set generation
    decoded under: a reply ending on a stop token that decodes to something
    visible would otherwise pick up that token's label here and nowhere else,
    and two identical replies from models that spell their stop token
    differently would part at their last character.

    Called with the generation slot held, so the tokenizer cannot be swapped
    out underneath it. An empty answer where there is nothing to decode with;
    the alignment falls back to standalone lengths and says so.
    """

    from model_runtime import IncrementalDecoder

    tokenizer = runtime.MANAGER.tokenizer
    if tokenizer is None:
        return "", []
    decoder = IncrementalDecoder(tokenizer, runtime.MANAGER.hidden_token_ids())
    for token_id in context_ids or ():
        decoder.push(int(token_id))
    context_end = len(decoder.text)
    ends = []
    for metric in metrics:
        decoder.push(int(metric["token_id"]))
        ends.append(len(decoder.text))
    full = decoder.text
    base = (
        len(full) - len(expected)
        if expected and full.endswith(expected)
        else context_end
    )
    # Clamped, because a token pushed after the context can in principle
    # redraw the boundary behind it; a negative end would be nonsense.
    return full[base:], [max(end - base, 0) for end in ends]


def clear_slots():
    """Empty both slots and everything drawn from them.

    Bound with ``cancels`` on the run events, because clearing during a run
    is otherwise undone by the run itself: the frames still in flight would
    write their progress over the cleared status and the last one would put
    the slot back. Cancelling closes that generator where it stands, which
    is also what gives the model lock back, so the buttons are published
    here as well - the cancelled run never reaches its own final frame.
    """

    return (None, None, COMPARE_EMPTY, *_running(False), *render(None, None))


def render(left, right):
    """Draw both slots and, when both are filled, what separates them."""

    reading = compare.reading(left, right)
    spans = reading.get("readings", [])
    strips = []
    for run, side in ((left, "left"), (right, "right")):
        metrics = (run or {}).get("metrics") or []
        strips.append(
            gr.update(
                value=compare.strip(metrics, spans if reading else [], side),
                color_map=compare.GAP_COLORS,
            )
        )
    return (
        compare.describe(left, "A"),
        compare.describe(right, "B"),
        *strips,
        charts.comparison_tiles(reading),
        (
            charts.surprise_chart(
                compare.gap_metrics(reading), title="Surprise gap per shared token"
            )
            if reading
            else charts.EMPTY_CHART
        ),
        compare.headline(reading, left, right),
        gr.update(value=compare.configuration_rows(left, right)),
        gr.update(value=compare.divergence_rows(spans)),
        {"left": left, "right": right, "reading": reading},
    )


def download_comparison(held):
    """Write both runs and the comparison between them where the browser can fetch it."""

    if not held or not held.get("left") or not held.get("right"):
        return None
    document = compare.export(held["left"], held["right"], held["reading"])
    directory = Path(tempfile.mkdtemp(prefix="chatlab-"))
    path = directory / f"chatlab-comparison-{uuid4().hex[:8]}.json"
    write_private_text(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return str(path)


def stop_comparison():
    """Give the buttons back after Stop closed the run where it stood."""

    return "Stopped. The slot was left as it was.", *_running(False)
