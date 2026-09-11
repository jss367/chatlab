"""Conversation-owned chat jobs, rendered by short events on the pane queue.

Only the worker advances or closes a model iterator. It never writes Gradio
components. Navigation and polling share a queue, so a frame cannot land in a
conversation selected between the handler and Gradio's state update.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import inspect
import logging
import threading
import typing

import gradio as gr

import library
from conversation import branch_choices, copy_forks, copy_turns, display_messages, put_branch
from ui.common import finalize_partial
from ui.generation import CHAT_OUTPUT_NAMES
from ui.panel import restore_chat_metrics_generation, transcript_update

logger = logging.getLogger(__name__)
NAMES = {name: i for i, name in enumerate(CHAT_OUTPUT_NAMES)}


def skipped(value):
    return isinstance(value, dict) and value == gr.skip()


class ConversationJob:
    """One session's latest run; deepcopy creates an independent browser session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.worker = None
        self.running = False
        self.owner = None
        self.saved = None
        self.frame = {}
        self.pending = {}
        self.version = 0
        self.rendered = None

    def __deepcopy__(self, memo):
        return type(self)()

    def _publish(self, frame):
        with self.lock:
            changed = {i: value for i, value in enumerate(frame) if not skipped(value)}
            self.frame.update(copy.deepcopy(changed))
            self.pending.update(copy.deepcopy(changed))
            self.version += 1
            turns = self.frame.get(NAMES["turns"])
            if isinstance(turns, list):
                put_branch(self.saved, self.owner, turns)
                # Persist even if the browser is hidden or disconnected. This
                # run owns its transcript, not the reader's active selection.
                library.write(self.saved, preserve_active=True)

    def start(self, iterator, forks):
        """Advance the opening frame on the pane queue to reserve the model."""
        first = next(iterator, None)
        if first is None:
            return None
        with self.lock:
            self.owner = forks["active"]
            self.saved = copy_forks(forks)
            self.frame = {}
            self.pending = {}
            self.cancel = threading.Event()
            self.running = True
            self.rendered = None
            self._publish(first)
        context = contextvars.copy_context()
        self.worker = threading.Thread(
            target=lambda: context.run(self._work, iterator),
            name="chatlab-conversation",
            daemon=True,
        )
        try:
            self.worker.start()
        except BaseException:
            iterator.close()
            self._finish("Could not start generation.")
            raise
        return first

    def _work(self, iterator):
        error = None
        try:
            with contextlib.closing(iterator):
                while not self.cancel.is_set():
                    frame = next(iterator, None)
                    if frame is None:
                        break
                    frame = list(frame)
                    frame[NAMES["prompt"]] = gr.skip()
                    self._publish(frame)
        except Exception:
            logger.exception("Conversation generation failed")
            error = "Generation failed. Any partial response was kept."
        finally:
            self._finish(error)

    def _finish(self, error=None):
        with self.lock:
            frame = [
                gr.skip()
                for _ in range(max(len(CHAT_OUTPUT_NAMES), max(self.frame, default=0) + 1))
            ]
            turns = self.frame.get(NAMES["turns"])
            if isinstance(turns, list):
                turns = copy_turns(turns)
                finalize_partial(turns)
                frame[NAMES["turns"]] = turns
                frame[NAMES["chatbot"]] = display_messages(turns)[0]
            if error or self.cancel.is_set():
                frame[NAMES["status"]] = error or "Stopped. Any partial response was kept."
            self.running = False
            self._publish(frame)

    def stop(self):
        with self.lock:
            if self.running:
                self.cancel.set()

    def merge(self, forks, turns):
        """Bring the source transcript up to date without claiming another branch."""
        forks = copy_forks(forks)
        with self.lock:
            if self.saved is not None and self.owner in forks["branches"]:
                stamp = self.saved["updated"].get(self.owner, "")
                if stamp >= forks["updated"].get(self.owner, ""):
                    forks["branches"][self.owner] = copy_turns(self.saved["branches"][self.owner])
                    if stamp:
                        forks["updated"][self.owner] = stamp
                    if forks["active"] == self.owner:
                        turns = copy_turns(forks["branches"][self.owner])
        return forks, turns

    def controls(self, forks):
        with self.lock:
            if not self.running:
                return gr.update(visible=True, interactive=True), gr.update(
                    visible=False, value="Stop"
                )
            label = "Stopping…" if self.cancel.is_set() else f"Stop {self.owner}"
            return gr.update(visible=True, interactive=False), gr.update(
                visible=True,
                interactive=not self.cancel.is_set(),
                value=label,
            )

    def choices(self, forks, turns):
        choices = branch_choices(forks, turns)
        with self.lock:
            if self.running:
                choices = [
                    (f"{label} · Generating…" if name == self.owner else label, name)
                    for label, name in choices
                ]
        return gr.update(choices=choices, value=forks["active"])

    def render(self, forks, turns, scale):
        with self.lock:
            forks, turns = self.merge(forks, turns)
            active = forks["active"]
            key = (active, self.version, self.running, self.cancel.is_set())
            if key == self.rendered:
                return {}, forks, turns
            switched = self.rendered is None or self.rendered[0] != active
            values = {}
            # A later edit/deletion owns the branch once this run is finished.
            current = (
                self.saved is not None
                and active == self.owner
                and active in forks["branches"]
                and self.saved["updated"].get(active, "") >= forks["updated"].get(active, "")
            )
            if current:
                values = copy.deepcopy(self.frame if switched else self.pending)
                metrics = self.frame.get(NAMES["metrics"])
                if metrics is not None:
                    restore_chat_metrics_generation(metrics[0])
                values[NAMES["strip"]] = transcript_update(turns, scale)
                # Navigation must never resurrect the prompt that launched a job.
                if switched and self.rendered is not None:
                    values.pop(NAMES["prompt"], None)
                # Editor controls and token selections are local to the view.
                if switched and self.rendered is not None:
                    for i in range(len(CHAT_OUTPUT_NAMES), len(CHAT_OUTPUT_NAMES) + 2):
                        values.pop(i, None)
            elif self.running:
                values[NAMES["status"]] = (
                    f"{self.owner} is generating. You can browse conversations; "
                    "wait for it to finish or press Stop before sending another message."
                )
            elif self.rendered is not None and self.rendered[2]:
                values[NAMES["status"]] = (
                    f"{self.owner}: {self.frame.get(NAMES['status'], 'Finished.')}"
                )
            values[NAMES["send"]], values[NAMES["stop"]] = self.controls(forks)
            self.pending = {}
            self.rendered = key
            return values, forks, turns


class ConversationEvents:
    """Adapt existing handlers without moving inference onto the UI event queue."""

    def __init__(self, state, turns, forks, picker, scale, chat_outputs, queue):
        self.state, self.turns, self.forks = state, turns, forks
        self.picker, self.scale = picker, scale
        self.chat_outputs, self.queue = chat_outputs, queue

    def bind(
        self,
        trigger,
        fn,
        inputs,
        outputs,
        *,
        generation=False,
        navigation=False,
        stop=False,
        clear=False,
        **kwargs,
    ):
        inputs = list(inputs or [])
        outputs = list(outputs)
        actual_inputs = list(
            dict.fromkeys([*inputs, self.state, self.turns, self.forks, self.scale])
        )
        actual_outputs = list(
            dict.fromkeys(
                [
                    *outputs,
                    self.turns,
                    self.forks,
                    self.picker,
                    self.chat_outputs[NAMES["send"]],
                    self.chat_outputs[NAMES["stop"]],
                    self.chat_outputs[NAMES["status"]],
                ]
            )
        )
        hints = typing.get_type_hints(fn)
        event = next(
            (
                (i, p.name, hints.get(p.name))
                for i, p in enumerate(inspect.signature(fn).parameters.values())
                if isinstance(hints.get(p.name), type) and issubclass(hints[p.name], gr.EventData)
            ),
            None,
        )

        def handler(*args):
            data = dict(zip(actual_inputs, args))
            job = data[self.state]
            forks, turns = job.merge(data[self.forks], data[self.turns])
            active_before = forks["active"]
            with job.lock:
                owns_source = (
                    job.saved is not None
                    and job.owner in forks["branches"]
                    and job.saved["updated"].get(job.owner, "")
                    >= forks["updated"].get(job.owner, "")
                )
            data[self.forks], data[self.turns] = forks, turns
            result = {}
            if stop:
                job.stop()
                result[self.chat_outputs[NAMES["status"]]] = (
                    f"Stopping {job.owner}…" if job.running else "No response is running."
                )
            elif job.running and (
                generation or (not navigation and (clear or forks["active"] == job.owner))
            ):
                result[self.chat_outputs[NAMES["status"]]] = (
                    f"{job.owner} is generating. Press Stop or wait for it to finish first."
                )
            else:
                call_args = [data[component] for component in inputs]
                if event:
                    call_args.insert(event[0], args[len(actual_inputs)])
                if generation:
                    iterator = fn(*call_args)
                    try:
                        job.start(iterator, forks)
                    except BaseException:
                        iterator.close()
                        with job.lock:
                            job.running = False
                        raise
                    values, forks, turns = job.render(forks, turns, data[self.scale])
                    result.update(
                        {outputs[i]: value for i, value in values.items() if i < len(outputs)}
                    )
                else:
                    returned = fn(
                        *call_args, **({"preserve_source": owns_source} if navigation else {})
                    )
                    result.update(zip(outputs, returned))
                    if not skipped(result.get(self.forks, gr.skip())):
                        forks = result[self.forks]
                    if not skipped(result.get(self.turns, gr.skip())):
                        turns = result[self.turns]
                    # A navigation handler closes its snapshot of a partial
                    # reply; keep the live source owned by the worker instead.
                    if navigation:
                        forks, turns = job.merge(forks, turns)
                        if owns_source and job.owner in forks["branches"]:
                            with job.lock:
                                forks["branches"][job.owner] = copy_turns(
                                    job.saved["branches"][job.owner]
                                )
                                forks["updated"][job.owner] = job.saved["updated"].get(
                                    job.owner, ""
                                )
                                if forks["active"] == job.owner:
                                    turns = copy_turns(forks["branches"][job.owner])
                        result[self.chat_outputs[NAMES["chatbot"]]] = display_messages(turns)[0]
                    else:
                        # Retire completed frames before an edit can be
                        # replayed by the next timer tick.
                        with job.lock:
                            if not job.running and (clear or active_before == job.owner):
                                job.saved = None
                                job.pending = {}
                    # Even away-and-back between timer ticks must restore
                    # this run's cached metrics, without restoring its draft.
                    if navigation or forks["active"] != active_before:
                        with job.lock:
                            job.rendered = ("", -1, job.running, False)
                    put_branch(forks, forks["active"], turns)
                    if navigation and job.running:
                        result[self.chat_outputs[NAMES["status"]]] = (
                            f"{job.owner} is generating. You can browse conversations or press Stop."
                        )
            result[self.turns], result[self.forks] = turns, forks
            result[self.picker] = job.choices(forks, turns)
            result[self.chat_outputs[NAMES["send"]]], result[self.chat_outputs[NAMES["stop"]]] = (
                job.controls(forks)
            )
            library.write(library.as_seen(forks, turns))
            return tuple(result.get(component, gr.skip()) for component in actual_outputs)

        handler.__name__ = fn.__name__
        parameters = [
            inspect.Parameter(f"input_{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for i in range(len(actual_inputs))
        ]
        if event:
            parameters.append(
                inspect.Parameter(
                    "event", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=event[2]
                )
            )
            handler.__annotations__ = {"event": event[2]}
        handler.__signature__ = inspect.Signature(parameters)
        kwargs.pop("cancels", None)
        kwargs["concurrency_id"] = self.queue
        return trigger(handler, actual_inputs, actual_outputs, **kwargs)

    def poll(self, job, turns, forks, scale):
        values, merged, current = job.render(forks, turns, scale)
        if not values:
            return (gr.skip(),) * (len(self.chat_outputs) + 2)
        result = [values.get(i, gr.skip()) for i in range(len(self.chat_outputs))]
        if merged["active"] == job.owner and NAMES["turns"] in values:
            result[NAMES["turns"]] = current
        return (*result, merged, job.choices(merged, current))

    def refresh_conversation_list(self, turns, forks, job):
        seen = library.as_seen(forks, turns)
        return job.choices(seen, turns), seen
