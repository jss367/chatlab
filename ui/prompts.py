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
    BATCH_CSV_NAME,
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
BATCH_OUTPUT_NAMES = ("status", "results", "run", "stop", "files", "directory")

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


def count_prompts(text: str, loaded=()) -> str:
    """How many prompts a run would take from the box.

    Counted through resolve_prompts() rather than the box alone, so a loaded
    prompt with a blank line inside it is counted once, the way it will run.
    A count that disagreed with the run would be worse than none: it is the
    only number on screen before the press.
    """

    count = len(resolve_prompts(text, loaded))
    if not count:
        return PROMPT_COUNT_HINT
    return f"{count} prompt{'' if count == 1 else 's'}."


PARAGRAPH_NOTE = (
    "One of them has a blank line inside it. The box separates prompts on "
    "blank lines, so it cannot show that prompt as one; the run uses the "
    "file's own prompts as long as the box is left as it was loaded."
)


def load_prompt_file(file_path, text: str, loaded):
    """Put a file of prompts into the box, keeping what is already there.

    The box stays the one place the run reads from, so an uploaded file is
    only a way of filling it: what is on screen is what will run, and it can
    be edited first.

    The prompts are also returned as they were read. A prompt with a blank
    line inside it cannot be told apart from two prompts once it is in the
    box, and a dataset entry of several paragraphs is exactly that, so the
    run prefers this list while the box still holds what loading it wrote;
    see resolve_prompts(). A reader who edits the box gets what the box
    says, which is the only honest reading of a set they have changed.
    """

    if not file_path:
        return gr.skip(), "No file chosen.", gr.skip()
    try:
        prompts = parse_prompt_file(file_path)
    except (OSError, ValueError) as error:
        return (
            gr.skip(),
            failure_status("Could not read that file", str(error)),
            gr.skip(),
        )
    if not prompts:
        return gr.skip(), f"No prompts in `{Path(file_path).name}`.", gr.skip()

    # What is already in the box keeps its place at the front, read the way
    # the box reads it, because that is what a run would have used anyway.
    kept = resolve_prompts(text, loaded)
    combined = kept + prompts
    added = f"{len(prompts)} prompt{'' if len(prompts) == 1 else 's'}"
    status = f"Loaded {added} from `{Path(file_path).name}`."
    if any("\n\n" in prompt for prompt in combined):
        status = f"{status} {PARAGRAPH_NOTE}"
    return prompts_to_text(combined), status, combined


def resolve_prompts(text: str, loaded) -> list[str]:
    """The prompts a run should use: the loaded ones, or the box's own.

    ``loaded`` is what the last file gave, and the box holds what writing
    that list looks like. While the two still agree the list is used, so a
    prompt of several paragraphs runs as the one prompt the file said it
    was rather than as one conversation per paragraph. The moment the box
    differs it is the reader's own text, and blank lines in it separate
    prompts as they always have.
    """

    written = (text or "").strip()
    prompts = list(loaded or [])
    if prompts and prompts_to_text(prompts) == written:
        return prompts
    return parse_prompts(text)


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
    loaded_prompts,
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
    refused = (skip,) * 5

    if not runtime.MANAGER.loaded:
        yield (BATCH_NO_MODEL,) + refused
        return
    prompts = resolve_prompts(prompts_text, loaded_prompts)
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
        # Where this run's files are. Cancellation closes this generator
        # where it stands, so its last frame cannot list a file written on
        # the way out; stop_batch() reads the directory instead.
        str(directory),
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
        forced_prefix_tokens = 0
        applied_prefill = bool(assistant_prefill)
        kept = False
        failed = False

        def keep(*, stopped: bool = False) -> None:
            """Write what this prompt produced, and add it to the table.

            Called once for a prompt: on the way out of the ordinary path,
            and from the finally below when the run was cancelled instead.
            Nothing is yielded from here, because the cancellation path is
            already inside GeneratorExit and a yield there is an error.
            """

            nonlocal kept
            if kept or not metrics:
                return
            kept = True
            sampling = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_new_tokens": int(max_new_tokens),
                "seed": used_seed,
            }
            if forced_prefix_tokens:
                # The prefill's tokens are measured like any other, so a
                # reader of the trace would take them for the model's own
                # choices without being told how many were replayed.
                # ui.generation says it the same way for a single response.
                sampling["forced_prefix_tokens"] = forced_prefix_tokens
            if applied_prefill:
                sampling["assistant_prefill"] = assistant_prefill
            if stopped:
                # The tokens are exact, but they may not be the whole answer:
                # Stop can land while the model is still writing. A trace that
                # did not say so would be read as a finished response and put
                # a truncated answer in an experiment beside whole ones.
                sampling["stopped"] = True
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
            # The files are written as the run goes, so stopping half way
            # through still leaves every prompt that produced tokens on disk.
            # The table takes this prompt's rows for the same reason.
            paths[:] = [*trace_paths, table.add(traces[-1], index)]

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
                # The progress line for one update is published at the top of
                # the next pass, so nothing is yielded after the update that
                # turns out to be the last: the loop falls straight through to
                # writing this prompt's trace. Every yield left is one Stop
                # can land on with an answer part written, and keep() in the
                # finally is what puts that answer on disk rather than
                # dropping it - marked stopped, since the model may have had
                # more to say.
                held = None
                for update in stream:
                    if held is not None:
                        yield held
                    text = update.text
                    metrics = list(update.metrics)
                    model_id = update.model_id or model_id
                    prefilled = update.reasoning_prefilled
                    forced_prefix_tokens = update.forced_prefix_tokens
                    if update.literal_prefill_text:
                        literal_prefill = update.literal_prefill_text
                    held = (
                        batch_progress(index, total, len(metrics), started),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                    )
            # The stream is done and no yield stands between here and the
            # files, so this is the whole answer rather than a stopped one.
            keep()
        except ModelChanged:
            # This one is not a result about the prompt, so it gets no row.
            # The prompts after it would answer under weights the finished
            # rows were not measured on, and a table mixing the two is worse
            # than a short one, so the run stops here and says why.
            logger.warning(
                "Prompt %s of %s did not run: the model changed", index, total
            )
            changed_at = index
            failed = True
            break
        except Exception as error:
            # One bad prompt does not end the run: a passage too long for the
            # context window is a result about that prompt, and the twenty
            # after it are still worth having. The row says which failed and
            # why, the log keeps the traceback, and the count is reported once
            # at the end rather than as a toast per prompt.
            logger.exception("Prompt %s of %s failed", index, total)
            failed = True
            failures += 1
            rows.append(failed_row(index, prompt, error, used_seed))
            yield (
                batch_progress(index, total, len(metrics), started),
                gr.update(value=list(rows)),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
            continue
        finally:
            # Cancellation arrives as GeneratorExit thrown into whichever
            # yield above is open, so the lines after this loop never run for
            # the prompt in flight. Without this its tokens would go with it,
            # which is the one thing the files written as the run goes are
            # there to prevent. stop_batch() publishes what is in the
            # directory, so a prompt kept here is still reachable.
            #
            # A prompt that raised is not kept, whatever it had written by
            # then: a failed response is not a response to export, which is
            # what ui.generation does for a single reply, and exporting one
            # here would put a half-answer in the table under a row that
            # says it failed.
            if not failed:
                keep(stopped=True)

        _reasoning, answer, _closed = split_response_text(
            text, literal_prefill=literal_prefill, reasoning_prefilled=prefilled
        )
        rows.append(batch_row(index, prompt, answer, summarize(metrics), used_seed))
        yield (
            batch_progress(index, total, len(metrics), started),
            gr.update(value=list(rows)),
            gr.skip(),
            gr.skip(),
            gr.update(value=list(paths), visible=bool(paths)),
            gr.skip(),
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
        gr.skip(),
    )


def batch_files(directory) -> list[str]:
    """Every export the run in ``directory`` has written, traces first."""

    if not directory:
        return []
    place = Path(directory)
    if not place.is_dir():
        return []
    traces = sorted(str(path) for path in place.glob("prompt-*.json"))
    table = place / BATCH_CSV_NAME
    return traces + ([str(table)] if table.exists() else [])


def stop_batch(directory=None):
    """Give the buttons back, and publish what the stopped run wrote.

    Gradio closes the generator where it stood, so the frame it published
    last cannot mention the prompt it was in the middle of - and that
    prompt's tokens are on disk, written on the way out. The directory is
    read here instead of trusting that frame, so a stopped run offers every
    prompt that produced anything, the last one included.

    The rows on screen are left alone: they are what the run itself said,
    and this handler knows nothing about a prompt beyond its file.
    """

    found = batch_files(directory)
    return (
        "Stopped. Everything the run measured is below.",
        gr.skip(),
        *send_stop_buttons(False),
        gr.update(value=found, visible=True) if found else gr.skip(),
        gr.skip(),
    )
