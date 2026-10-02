"""The Reasoning check tab: cut a reply's reasoning short, plant a mistake in it, or paraphrase it, and see whether the answer changes.

The interventions themselves are in :mod:`chatlab.faithfulness`. This module
picks the reply, holds the generation slot while the model answers, and keeps
the results.
"""
from __future__ import annotations

import csv
import io
import logging
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from chatlab import faithfulness as check
from chatlab.model_runtime import LOADING
from chatlab.steering import SteeringError
from chatlab.text_generation import ModelChanged
from chatlab.ui import runtime
from chatlab.ui.common import failure_status

logger = logging.getLogger(__name__)

TAB_LABEL = "Reasoning check"

INTRO = (
    "Change a reply's reasoning and see whether its answer changes with it. Reasoning the answer depends on moves "
    "the answer when it is cut short or broken. Reasoning written around an answer already decided leaves it "
    "where it was. Every answer here is written greedily under the model that wrote the reply, so two answers "
    "differ because of what the model was given."
)
NO_REPLIES = ("No reply in this conversation can be checked yet. Ask a reasoning model something: a reply needs "
              "its reasoning and the tokens it was made of, which only a reply written in this session has.")
BUSY = "Wait for the response to finish before running a check."
LOADING_BUSY = "Wait for the model to finish loading before running a check."
NO_MODEL = "Download and load a model first."
PICK_AGAIN = "The conversation changed since this reply was picked. Pick it again."
MODEL_CHANGED = "The model changed while the check was running. The results above were read before it did."
STOPPED = "Stopped. The results above are complete; the answer being written was dropped."


def _buttons(running):
    """The run buttons and Stop, for a check that is or is not running."""
    return (*(gr.update(interactive=not running) for _ in range(4)), gr.update(visible=running))


def build():
    with gr.Tab(TAB_LABEL, elem_id="reasoning-check-tab") as tab:
        gr.Markdown(INTRO)
        reply = gr.Dropdown([], label="Reply", info="Replies with reasoning, written in this session.",
                            elem_id="reasoning-check-reply", filterable=False)
        picked = gr.State(None)
        pattern = gr.Textbox(
            label="Answer pattern", placeholder=r"answer is (\w+)", elem_id="reasoning-check-pattern",
            info="A regular expression for the final answer. Answers are compared by its last match, or by its "
                 "first group if it has one. Leave it blank to compare whole answers.")
        with gr.Accordion("Cut it short", open=True):
            gr.Markdown("Keeps " + ", ".join(f"{f:.0%}" for f in check.FRACTIONS) + " of the reasoning's words, "
                        "closes the reasoning there, and has the model answer. The 100% row is the control: the "
                        "whole reasoning, answered the same way.")
            cut = gr.Button("Run the cut test", variant="primary", elem_id="reasoning-check-cut")
        with gr.Accordion("Plant a mistake", open=False):
            mistake = gr.Textbox(
                label="Reasoning with a mistake", lines=8, elem_id="reasoning-check-mistake",
                info="Change one step. The reasoning is kept up to your change, the step you changed is fed to the "
                     "end of its sentence, and the model reasons on from there and answers.")
            plant = gr.Button("Plant and answer", elem_id="reasoning-check-plant")
        with gr.Accordion("Paraphrase", open=False):
            paraphrase = gr.Textbox(
                label="Paraphrased reasoning", lines=8, elem_id="reasoning-check-paraphrase",
                info="The whole reasoning in other words. The model answers straight after it.")
            with gr.Row():
                write = gr.Button("Write one with the loaded model", elem_id="reasoning-check-write")
                answer = gr.Button("Answer from the paraphrase", elem_id="reasoning-check-answer")
        with gr.Row():
            stop = gr.Button("Stop", variant="stop", visible=False, elem_id="stop-reasoning-check")
            clear = gr.Button("Clear results", elem_id="reasoning-check-clear")
        status = gr.Markdown(NO_REPLIES, elem_id="reasoning-check-status")
        results = gr.State([])
        table = gr.Dataframe(headers=check.HEADERS, value=[], interactive=False, wrap=True,
                             elem_id="reasoning-check-results", label="Results")
        detail = gr.Markdown("", elem_id="reasoning-check-detail")
        download = gr.File(label="Results CSV", interactive=False, visible=False)
    return SimpleNamespace(tab=tab, reply=reply, picked=picked, pattern=pattern, cut=cut, mistake=mistake,
                           plant=plant, paraphrase=paraphrase, write=write, answer=answer, stop=stop, clear=clear,
                           status=status, results=results, table=table, detail=detail, download=download)


def refresh_replies(turns, current, picked):
    """The replies that can be checked, keeping the one picked while it is still the same reply.

    A reply retried or regenerated in the same place is another reply, so
    the boxes holding its reasoning are refilled for it.
    """
    choices = check.reasoning_replies(turns)
    positions = [position for _, position in choices]
    value = current if current in positions else (positions[-1] if positions else None)
    identity = check.reply_identity(turns[value]) if value is not None else None
    skip = gr.skip()
    if identity == picked and value == current:
        return gr.update(choices=choices, value=value), skip, skip, skip, skip
    return (gr.update(choices=choices, value=value), identity,
            (turns[value].get("reasoning") or "").strip() if value is not None else "", "",
            "" if choices else NO_REPLIES)


def pick_reply(turns, position):
    """Fill the boxes with the picked reply's reasoning."""
    if position is None or not 0 <= position < len(turns or ()):
        return None, "", "", NO_REPLIES
    turn = turns[position]
    return check.reply_identity(turn), (turn.get("reasoning") or "").strip(), "", ""


def write_csv(results):
    if not results:
        return gr.update(value=None, visible=False)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Reply", "Intervention", "Reasoning given", "Reasoning text", "Answer", "Same answer",
                     "Original answer perplexity"])
    for result, row in zip(results, check.result_rows(results)):
        writer.writerow([result.reply, result.kind, result.label, result.reasoning,
                         result.answer if result.answer is not None else result.note, row[4], row[5]])
    path = Path(tempfile.mkdtemp(prefix="chatlab-reasoning-check-")) / "reasoning-check.csv"
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return gr.update(value=str(path), visible=True)


def _frame(results, status, running, *, download=None):
    return (list(results), gr.update(value=check.result_rows(results)), status,
            gr.skip() if download is None else download,
            *((gr.skip(),) * 5 if running is None else _buttons(running)))


def _claim():
    """Take the generation slot, or say why not."""
    held = runtime.MANAGER.claim_generation()
    if held:
        return LOADING_BUSY if held == LOADING else BUSY
    if not runtime.MANAGER.loaded:
        runtime.MANAGER.release_generation()
        return NO_MODEL
    return None


def _reply(turns, position, picked, system_prompt, keep_reasoning):
    if position is None or not 0 <= position < len(turns or ()):
        raise ValueError("Pick a reply first.")
    if check.reply_identity(turns[position]) != picked:
        raise ValueError(PICK_AGAIN)
    return check.read_reply(turns, position, runtime.MANAGER, system_prompt=system_prompt,
                            keep_reasoning=keep_reasoning)


def _encoder(reply):
    return lambda kept, text: runtime.MANAGER.encode_replacement(kept, text, load_id=reply.load_id)


def run_check(kind, turns, position, picked, pattern, max_new_tokens, system_prompt, keep_reasoning, results, text=""):
    """Run one intervention, or the cut test's five, on the picked reply, adding a row for each answer."""
    results = list(results or ())
    refused = _claim()
    if refused:
        yield _frame(results, refused, None)
        return
    finished = False
    try:
        check.check_pattern(pattern)
        reply = _reply(turns, position, picked, system_prompt, keep_reasoning)
        encode_after = _encoder(reply)
        if kind == check.CUT:
            plans = [check.cut_plan(reply, fraction, encode_after) for fraction in check.FRACTIONS]
        elif kind == check.MISTAKE:
            plans = [check.mistake_plan(reply, text, encode_after)]
        else:
            plans = [check.paraphrase_plan(reply, text, encode_after)]
        logger.info("Reasoning check: %s on reply %s, %s answers", kind, reply.number, len(plans))
        for index, plan in enumerate(plans, start=1):
            running = check.run_plan(runtime.MANAGER, reply, plan, max_new_tokens=max_new_tokens, pattern=pattern,
                                     encode_after=encode_after)
            note = f"**Running** · {kind} · answer {index} of {len(plans)} · {plan.label}"
            yield _frame(results, note, True)
            while True:
                try:
                    tokens = next(running)
                except StopIteration as done:
                    results.append(done.value)
                    break
                yield _frame(results, f"{note} · {tokens:,} tokens", True)
            yield _frame(results, note, True, download=write_csv(results))
        finished = True
        ended = f"**Finished** · {kind} on reply {reply.number} · {len(plans)} answer{'s' if len(plans) > 1 else ''}"
        yield _frame(results, ended, False, download=write_csv(results))
    except ModelChanged:
        yield _frame(results, failure_status("Check stopped", MODEL_CHANGED), False, download=write_csv(results))
    except (ValueError, RuntimeError, SteeringError) as error:
        logger.warning("Reasoning check refused: %s", error)
        yield _frame(results, failure_status("Could not run the check", str(error)), False,
                     download=write_csv(results))
    finally:
        runtime.MANAGER.release_generation()
        if not finished:
            logger.info("Reasoning check ended early with %s results", len(results))


def paraphrase_reply(turns, position, picked, max_new_tokens, system_prompt, keep_reasoning):
    """Have the loaded model paraphrase the picked reply's reasoning into the paraphrase box."""
    skip = gr.skip()
    refused = _claim()
    if refused:
        yield skip, refused, *((gr.skip(),) * 5)
        return
    try:
        reply = _reply(turns, position, picked, system_prompt, keep_reasoning)
        # Room for a paraphrase as long as the reasoning and then some, and
        # for a model that reasons before it writes one.
        budget = max(int(max_new_tokens), 2 * len(reply.ids) + 256)
        writing = check.write_paraphrase(runtime.MANAGER, reply, budget)
        yield skip, "**Writing a paraphrase**", *_buttons(True)
        while True:
            try:
                tokens = next(writing)
            except StopIteration as done:
                text = done.value
                break
            yield skip, f"**Writing a paraphrase** · {tokens:,} tokens", *_buttons(True)
        yield text, "Paraphrase written. Read it over, then answer from it.", *_buttons(False)
    except ModelChanged:
        yield skip, failure_status("Could not paraphrase", "The model changed while it was writing."), *_buttons(False)
    except (ValueError, RuntimeError) as error:
        yield skip, failure_status("Could not paraphrase", str(error)), *_buttons(False)
    finally:
        runtime.MANAGER.release_generation()


def clear_results():
    """Cancel the running UI state as well as clearing its results."""
    return ([], gr.update(value=[]), "", gr.update(value=None, visible=False), "", *_buttons(False))


def show_detail(results, event: gr.SelectData):
    row = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if not isinstance(row, int) or not 0 <= row < len(results or ()):
        return ""
    return check.result_detail(results[row])


def wire(view, conversation, system_prompt, keep_reasoning, max_new_tokens):
    pickers = [view.reply, view.picked, view.mistake, view.paraphrase, view.status]
    # Read when the tab opens and when the list is opened, rather than on
    # every change of the conversation: a streaming reply changes it on every
    # frame. A run checks the reply it was picked as again before it starts.
    for trigger in (view.tab.select, view.reply.focus):
        trigger(refresh_replies, [conversation, view.reply, view.picked], pickers,
                show_progress="hidden", trigger_mode="always_last")
    view.reply.input(pick_reply, [conversation, view.reply], [view.picked, view.mistake, view.paraphrase, view.status],
                     show_progress="hidden")
    outputs = [view.results, view.table, view.status, view.download, view.cut, view.plant, view.write, view.answer,
               view.stop]
    shared = [conversation, view.reply, view.picked, view.pattern]
    settings = [max_new_tokens, system_prompt, keep_reasoning]
    events = [view.cut.click(partial(run_check, check.CUT),
                             [*shared, *settings, view.results], outputs, show_progress="hidden")]
    for button, kind, text in ((view.plant, check.MISTAKE, view.mistake),
                               (view.answer, check.PARAPHRASE, view.paraphrase)):
        events.append(button.click(partial(run_check, kind), [*shared, *settings, view.results, text], outputs,
                                   show_progress="hidden"))
    events.append(view.write.click(
        paraphrase_reply, [conversation, view.reply, view.picked, *settings],
        [view.paraphrase, view.status, view.cut, view.plant, view.write, view.answer, view.stop],
        show_progress="hidden"))
    view.stop.click(lambda: (STOPPED, *_buttons(False)), None,
                    [view.status, view.cut, view.plant, view.write, view.answer, view.stop], cancels=events)
    view.clear.click(clear_results, None,
                     [view.results, view.table, view.detail, view.download, view.status,
                      view.cut, view.plant, view.write, view.answer, view.stop], cancels=events, queue=False)
    view.table.select(show_detail, view.results, view.detail, show_progress="hidden")
