"""The Prompts tab: run a list of prompts and keep every measurement.

One prompt, one fresh conversation, one trace file. Nothing here touches the
chat transcript: a batch is an experiment run beside the conversation, not a
turn in it.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio.utils import get_upload_folder

from conversation import make_turn, model_messages
from model_runtime import ModelChanged
from prompt_batch import (
    BatchTable,
    parse_prompt_file,
    parse_prompts,
    prompts_to_text,
    write_batch_trace,
)
from token_metrics import summarize
from trace_export import build_trace
from ui import runtime
from ui.common import failure_status, send_stop_buttons
from ui.generation import resolve_seed, split_response_text

logger = logging.getLogger(__name__)


# Every batch handler publishes this tuple, in this order.
BATCH_OUTPUT_NAMES = ("status", "results", "run", "stop", "files")

BATCH_HEADERS = [
    "#",
    "Prompt",
    "Response",
    "Tokens",
    "Perplexity",
    "Mean surprise",
    "Seed",
]

BATCH_NO_MODEL = "Download and load a model first."
BATCH_NO_PROMPTS = "Add some prompts first. A blank line separates one from the next."
BATCH_BUSY = "Wait for the response to finish before running a batch."
PROMPT_COUNT_HINT = "Prompts are separated by a blank line."
BATCH_MODEL_CHANGED = "The model was replaced while the batch was running"

# How much of a prompt and its answer the results table shows. The whole of
# both is in the trace beside it; this column is for telling the rows apart.
EXCERPT_LENGTH = 80


def count_prompts(text: str) -> str:
    """How many prompts the box holds, as it is written."""

    count = len(parse_prompts(text))
    if not count:
        return PROMPT_COUNT_HINT
    return f"{count} prompt{'' if count == 1 else 's'}."


def load_prompt_file(file_path, text: str):
    """Put a file of prompts into the box, keeping what is already there.

    The box stays the one place the run reads from, so an uploaded file is
    only a way of filling it: what is on screen is what will run, and it can
    be edited first.
    """

    if not file_path:
        return gr.skip(), "No file chosen."
    try:
        prompts = parse_prompt_file(file_path)
    except (OSError, ValueError) as error:
        return gr.skip(), failure_status("Could not read that file", str(error))
    if not prompts:
        return gr.skip(), f"No prompts in `{Path(file_path).name}`."

    existing = parse_prompts(text)
    added = f"{len(prompts)} prompt{'' if len(prompts) == 1 else 's'}"
    return (
        prompts_to_text(existing + prompts),
        f"Loaded {added} from `{Path(file_path).name}`.",
    )


def batch_directory() -> Path:
    """A private directory for one run's exports, where Gradio can serve it.

    Gradio only serves files from its own upload folder, which is shared -
    on Linux it is /tmp/gradio, readable by every account on the machine. The
    directory is narrowed to its owner as it is made, and every file written
    into it is owner-only from its first byte; see write_private_text().
    """

    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = (
        Path(get_upload_folder()) / "chatlab-prompts" / f"{stamp}-{uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


def excerpt(text: str, limit: int = EXCERPT_LENGTH) -> str:
    """One line of ``text`` for the results table."""

    flattened = " ".join((text or "").split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1].rstrip() + "…"


def batch_row(index: int, prompt: str, answer: str, summary: dict, seed: int) -> list:
    return [
        index,
        excerpt(prompt),
        excerpt(answer),
        summary["token_count"],
        round(summary["perplexity"], 2),
        round(summary["mean_surprise_bits"], 3),
        seed,
    ]


def failed_row(index: int, prompt: str, error: Exception, seed: int) -> list:
    return [index, excerpt(prompt), f"Failed: {error}", 0, None, None, seed]


def batch_progress(index: int, total: int, tokens: int, started: float) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    return f"Prompt {index} of {total} · {tokens} tokens · {elapsed:.1f}s"


def run_prompts(
    prompts_text: str,
    system_prompt: str,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
):
    """Run every prompt in the box, one fresh conversation each.

    The generation slot is held for the whole batch rather than per prompt:
    a reply arriving from the Chat tab between two prompts would be measured
    under the same sampling settings the batch is reporting, and would leave
    the run's own numbers describing a model that had answered something else
    in between. Chat, Retry and Score text all refuse while it is held, and
    say so.
    """

    skip = gr.skip()
    refused = (skip,) * 4

    if not runtime.MANAGER.loaded:
        yield (BATCH_NO_MODEL,) + refused
        return
    prompts = parse_prompts(prompts_text)
    if not prompts:
        yield (BATCH_NO_PROMPTS,) + refused
        return
    if not runtime.MANAGER.reserve_generation():
        yield (BATCH_BUSY,) + refused
        return
    try:
        yield from _run_batch(
            prompts,
            system_prompt,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
        )
    finally:
        # Every exit runs this, cancellation included: Gradio throws
        # GeneratorExit in at whichever yield the run is parked on, and a slot
        # left reserved there would refuse every reply for the rest of the
        # session. generate_reply() releases its own slot the same way.
        runtime.MANAGER.release_generation()


def _run_batch(
    prompts: list[str],
    system_prompt: str,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
):
    """The body of run_prompts(), run with the generation slot held."""

    directory = batch_directory()
    table = BatchTable(directory)
    traces: list[dict] = []
    rows: list[list] = []
    trace_paths: list[str] = []
    paths: list[str] = []
    failures = 0
    started = time.monotonic()
    total = len(prompts)
    # The load the run was started against. Holding the generation slot keeps
    # other replies off the model, but not a load: one started from another
    # browser tab waits on the model lock and can take it in the gap between
    # two prompts. Every prompt is asked for under this load, so the runtime
    # refuses rather than finishing the batch on other weights and reporting
    # the whole table as one experiment.
    expected_load_id = runtime.MANAGER.load_id
    changed_at: int | None = None

    # The table is always published inside an update envelope. A raw value
    # followed by gr.skip() has Gradio's streaming diff delete the data from
    # the object the browser still renders; see the same note in
    # ui.generation.snapshot().
    yield (
        f"Running {total} prompt{'' if total == 1 else 's'}…",
        gr.update(value=[]),
        *send_stop_buttons(True),
        gr.update(value=None, visible=False),
    )

    for index, prompt in enumerate(prompts, start=1):
        used_seed = resolve_seed(seed, randomize_seed)
        # A fresh conversation per prompt: one user turn, and whatever system
        # prompt the Settings page holds. Nothing carries over from the prompt
        # before it, which is the whole point of running them as a set.
        request = model_messages(
            [make_turn("user", prompt)], system_prompt=system_prompt
        )
        text = ""
        metrics: list[dict] = []
        model_id = None
        prefilled = False
        literal_prefill = ""
        applied_prefill = bool(assistant_prefill)
        try:
            stream = runtime.MANAGER.generate(
                request,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                max_new_tokens=int(max_new_tokens),
                seed=used_seed,
                # The prompt tokens are measured for the panel beside the
                # chat, and a batch has no panel: scoring them would cost a
                # pass over every prompt and reach no file.
                analyze_prompt=False,
                answer_prefill=assistant_prefill if applied_prefill else "",
                load_id=expected_load_id,
            )
            # closing() releases the model lock the moment Stop cancels this
            # event and Gradio closes the outer generator.
            with contextlib.closing(stream):
                for update in stream:
                    text = update.text
                    metrics = list(update.metrics)
                    model_id = update.model_id or model_id
                    prefilled = update.reasoning_prefilled
                    if update.literal_prefill_text:
                        literal_prefill = update.literal_prefill_text
                    yield (
                        batch_progress(index, total, len(metrics), started),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                    )
        except ModelChanged:
            # This one is not a result about the prompt, so it gets no row.
            # The prompts after it would answer under weights the finished
            # rows were not measured on, and a table mixing the two is worse
            # than a short one, so the run stops here and says why.
            logger.warning(
                "Prompt %s of %s did not run: the model changed", index, total
            )
            changed_at = index
            break
        except Exception as error:
            # One bad prompt does not end the run: a passage too long for the
            # context window is a result about that prompt, and the twenty
            # after it are still worth having. The row says which failed and
            # why, the log keeps the traceback, and the count is reported once
            # at the end rather than as a toast per prompt.
            logger.exception("Prompt %s of %s failed", index, total)
            failures += 1
            rows.append(failed_row(index, prompt, error, used_seed))
            yield (
                batch_progress(index, total, len(metrics), started),
                gr.update(value=list(rows)),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
            continue

        _reasoning, answer, _closed = split_response_text(
            text, literal_prefill=literal_prefill, reasoning_prefilled=prefilled
        )
        summary = summarize(metrics)
        sampling = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "seed": used_seed,
        }
        if applied_prefill:
            sampling["assistant_prefill"] = assistant_prefill
        rows.append(batch_row(index, prompt, answer, summary, used_seed))
        if metrics:
            traces.append(
                build_trace(
                    model_id=model_id,
                    messages=request,
                    response=text,
                    sampling=sampling,
                    metrics=metrics,
                )
            )
            # Numbered by where the prompt sits in the box, not by how many
            # traces came before it. A prompt that failed produces no trace,
            # and numbering by the count would hand its name and its row
            # number to the next prompt that worked, filing one prompt's
            # measurements under another's.
            trace_paths.append(write_batch_trace(traces[-1], directory, index))
            # The files are published as the run grows, so stopping half way
            # through still leaves every finished prompt downloadable. The
            # table takes this prompt's rows for the same reason.
            paths = [*trace_paths, table.add(traces[-1], index)]
        yield (
            batch_progress(index, total, len(metrics), started),
            gr.update(value=list(rows)),
            gr.skip(),
            gr.skip(),
            gr.update(value=list(paths), visible=bool(paths)),
        )

    elapsed = max(time.monotonic() - started, 1e-6)
    tokens = sum(trace["token_count"] for trace in traces)
    # Counted from the rows rather than from ``total``, because a run that
    # stopped at a model change never reached the prompts after it.
    done = len(rows) - failures
    status = (
        f"Ran {done} of {total} prompt{'' if total == 1 else 's'} · "
        f"{tokens:,} tokens · {elapsed:.1f}s"
    )
    if traces:
        status = f"{status} · {len(traces)} trace files and one table ready."
    if changed_at is not None:
        status = f"{status}\n\n" + failure_status(
            BATCH_MODEL_CHANGED,
            f"Prompt {changed_at} onwards did not run. What is listed above "
            "was measured on the model the batch started with.",
        )
    if failures:
        # Appended rather than substituted: what did run is still the answer
        # to what was asked, and the failure is the caveat on it.
        status = f"{status}\n\n" + failure_status(
            f"{failures} of {total} prompts failed",
            "The table says which, and the log has the traceback.",
        )
    yield (
        status,
        gr.update(value=list(rows)),
        *send_stop_buttons(False),
        gr.update(value=list(paths), visible=bool(paths)),
    )


def stop_batch():
    """Give the buttons back after the run was cancelled at a yield.

    Gradio closes the generator where it stood, so the last frame it
    published is the last word on what ran: those rows and those files are
    the prompts that finished, and they stay on screen.
    """

    return (
        "Stopped. The prompts that finished are still listed below.",
        gr.skip(),
        *send_stop_buttons(False),
        gr.skip(),
    )
