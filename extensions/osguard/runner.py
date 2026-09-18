"""Cancellable batches using the host's exclusive model session."""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import threading
import time
from uuid import uuid4

from .benchmark import (
    FORMAT, LABELS, MODES, PAPER, THINK_CLOSE, dataset_digest, label_distribution,
    messages_for, parse_response, report,
)
from .storage import save_json

# A checkpoint rewrites the whole run, so a batch long enough to want one is
# also long enough for that write to cost real time. Waiting four times the
# last write before the next one keeps checkpointing under a fifth of the
# batch however large the traces grow, and the final save always runs.
CHECKPOINT_SECONDS = 2.0
CHECKPOINT_SHARE = 4.0
# What a replayed answer has to say first when the template already opened a
# reasoning block for the model. This is the same lead the runtime replays in
# front of a reader's own prefill, so the label lands where an answer goes.
REASONING_LEAD = f"{THINK_CLOSE}\n\n"


@dataclass(frozen=True)
class StreamingResponse:
    """Only the current answer changes between case-boundary snapshots."""
    run_id: str
    result: dict


class Runner:
    def __init__(self, models, data_dir):
        self.models = models
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._active = {}

    def cancel(self, owner):
        with self._lock:
            active = self._active.get(owner)
            if active:
                active[0].set()
                if active[1] is not None:
                    active[1].cancel()

    def run(self, owner, cases, *, mode="probability", max_new_tokens=256, seed=42):
        if not cases:
            raise ValueError("Import cases or load the demonstration first.")
        if mode not in MODES:
            raise ValueError("Scoring mode must be probability or judgment.")
        # Validate before registering the batch or claiming the model. Invalid
        # numeric inputs must not leave a stuck owner entry after conversion.
        try:
            integer_seed = int(seed)
            valid_seed = not isinstance(seed, bool) and integer_seed == seed and integer_seed >= 0
        except (TypeError, ValueError, OverflowError):
            valid_seed = False
        if not valid_seed:
            raise ValueError("Seed must be a nonnegative whole number.")
        try:
            token_limit = int(max_new_tokens)
            valid_limit = not isinstance(max_new_tokens, bool) and token_limit == max_new_tokens and 1 <= token_limit <= 4096
        except (TypeError, ValueError, OverflowError):
            valid_limit = False
        if not valid_limit:
            raise ValueError("Maximum answer tokens must be a whole number between 1 and 4096.")
        cancel = threading.Event()
        with self._lock:
            if owner in self._active:
                raise ValueError("A batch is already running in this view.")
            self._active[owner] = [cancel, None]
        # Probability scoring samples nothing: it replays each label as the
        # answer and reads what the model gave it.
        sampling = dict(temperature=0.0, top_p=1.0, top_k=0, seed=integer_seed,
                        max_new_tokens=1 if mode == "probability" else token_limit)
        run = dict(format=FORMAT, id=uuid4().hex, paper=PAPER,
                   created_at=datetime.now(timezone.utc).isoformat(),
                   mode="text_only_adaptation", scoring=mode,
                   dataset_sha256=dataset_digest(cases),
                   cases=copy.deepcopy(cases), predictions=[], status="running",
                   sampling=sampling)
        path = self.data_dir / f"{run['id']}.json"
        last_write, write_cost, on_disk = time.monotonic(), 0.0, False

        def save(force=True):
            nonlocal last_write, write_cost, on_disk
            run["scores"] = report(cases, run["predictions"])
            if not force and time.monotonic() < last_write + max(CHECKPOINT_SECONDS,
                                                                CHECKPOINT_SHARE * write_cost):
                return
            started = time.monotonic()
            save_json(path, run)
            last_write = time.monotonic()
            write_cost = last_write - started
            on_disk = True

        def checkpoint():
            """The download is the file, and there is no file until one is written."""
            return str(path) if on_disk else None

        try:
            with self.models.open_session() as session:
                with self._lock:
                    self._active[owner][1] = session
                    if cancel.is_set():
                        session.cancel()
                run.update(model_id=session.model_id, load_id=session.load_id)
                label_ids, lead_tokens = {}, 0
                if mode == "probability":
                    prefilled = _asks_for_reasoning(session, cases, mode, sampling)
                    run["reasoning_prefilled"] = prefilled
                    label_ids, lead_tokens = _label_tokens(session, prefilled)
                run["scores"] = report(cases, [])
                yield copy.deepcopy(run), None
                for case in cases:
                    if cancel.is_set():
                        break
                    try:
                        messages = messages_for(case, mode)
                    except ValueError as exc:
                        # One case that cannot be prompted is not a reason to
                        # refuse the other nine hundred. Say which, and go on.
                        run["predictions"].append(_result(case, [], status="skipped", feedback=str(exc)))
                        save(force=False)
                        yield copy.deepcopy(run), checkpoint()
                        continue
                    result = _result(case, messages,
                                     reasoning_prefilled=run.get("reasoning_prefilled", False))
                    run["predictions"].append(result)
                    answer = (_score_labels(session, messages, label_ids, lead_tokens,
                                            sampling, result, cancel)
                              if mode == "probability"
                              else _judge(session, messages, sampling, result, cancel))
                    try:
                        for frame in answer:
                            yield StreamingResponse(run["id"], frame), None
                    finally:
                        # A reader who navigates away closes this batch while it
                        # is suspended inside the answer. Closing the answer is
                        # what closes the generation stream underneath it, and
                        # the model session cannot be released until it is.
                        answer.close()
                    save(force=False)
                    yield copy.deepcopy(run), checkpoint()
                run["status"] = "cancelled" if cancel.is_set() else "completed"
        except GeneratorExit:
            run["status"] = "cancelled"
            raise
        except Exception as exc:
            run.update(status="error", error=str(exc))
            raise
        finally:
            if run["predictions"] and run["predictions"][-1]["status"] == "running":
                run["predictions"][-1]["status"] = "cancelled" if run["status"] == "cancelled" else "error"
            with self._lock:
                self._active.pop(owner, None)
            save()
        yield copy.deepcopy(run), checkpoint()


def _result(case, messages, *, status="running", feedback="", reasoning_prefilled=False):
    # scored_label names the label whose replayed trace the metrics hold, and
    # stays None for a free-text judgment, which has only ever one trace.
    return dict(id=case["id"], prediction=None, feedback=feedback, response="",
                metrics=[], prompt_ids=[], messages=messages, scored_label=None,
                reasoning_prefilled=reasoning_prefilled, status=status)


def _judge(session, messages, sampling, result, cancel):
    """Generate a free-text judgment and parse the label out of it."""
    stream = session.generate(messages, **sampling)
    try:
        for update in stream:
            result.update(response=update.text, metrics=copy.deepcopy(update.metrics),
                          prompt_ids=list(getattr(update, "prompt_ids", [])),
                          reasoning_prefilled=getattr(update, "reasoning_prefilled", False))
            yield copy.deepcopy(result)
        if cancel.is_set():
            result["status"] = "cancelled"
        else:
            result["prediction"], result["feedback"] = parse_response(
                result["response"], reasoning_prefilled=result["reasoning_prefilled"])
            result["status"] = "completed"
    finally:
        stream.close()


def _asks_for_reasoning(session, cases, mode, sampling):
    """Ask the loaded model, once, whether its prompt already opens a thinking block.

    The runtime answers this on every update, so one throwaway generation over
    a prompt this batch will really send settles it for the whole run. Reading
    the rendered prompt text instead would be a guess about a template the
    extension cannot see, and the answer decides what every replayed label is
    measured against, so a guess is not good enough.
    """
    for case in cases:
        try:
            messages = messages_for(case, mode)
        except ValueError:
            continue
        stream = session.generate(messages, **dict(sampling, max_new_tokens=1))
        try:
            return bool(getattr(next(iter(stream), None), "reasoning_prefilled", False))
        finally:
            stream.close()
    return False


def _label_tokens(session, reasoning_prefilled):
    """Tokenize each label as the model would read it at the start of its answer.

    ``encode_replacement`` is what settles the word-boundary question: a
    sentencepiece vocabulary spells a leading word with a space inside its
    first token, and encoding the bare string would score an answer the model
    was never going to write.

    A thinking template hands the model an open reasoning block, and a bare
    label replayed into it would be the first words of the model's thinking
    rather than its judgment. Closing the block first puts the label where an
    answer goes. Returns the lead length with the ids: the lead is identical
    across the three labels, so leaving it out of the score keeps the
    comparison fair and makes each number P(label | prompt and closed block).
    """
    lead = session.encode_replacement([], REASONING_LEAD) if reasoning_prefilled else []
    return {label: lead + session.encode_replacement(lead, label) for label in LABELS}, len(lead)


def _score_labels(session, messages, label_ids, lead_tokens, sampling, result, cancel):
    """Replay each label as the answer and read the probability the model gave it.

    Three prefills and no sampled token: the forced answer is measured against
    the model's own distribution on the way through, so what comes back is
    log P(label | prompt) for each of the three, exactly and in one pass each.
    A malformed answer is not possible here, which is the point - a free-text
    judgment scores formatting as much as safety. ``lead_tokens`` counts the
    reasoning-close tokens in front of every label, which are replayed so the
    label is an answer but never scored as part of one.

    Each frame says which label the trace it carries belongs to. Three replays
    replace the metrics rather than extend them, and a reader looking at a
    token needs to know whether the strip under the cursor is still the one
    that token came from.
    """
    logprobs, traces = {}, {}
    for label, forced in label_ids.items():
        if cancel.is_set():
            result["status"] = "cancelled"
            return
        update = _measure(session, messages, forced, sampling)
        if update is None:
            result["status"] = "cancelled"
            return
        metrics = [metric for metric in update.metrics
                   if metric.get("segment", "response") == "response"][:len(forced)]
        if len(metrics) != len(forced) or not all(metric.get("scored", True) for metric in metrics):
            raise ValueError(f"The model did not measure the replayed answer '{label}'.")
        scored = metrics[lead_tokens:]
        logprobs[label] = sum(math.log(max(float(metric["raw_probability"]), 1e-300))
                              for metric in scored)
        traces[label] = scored
        result.update(prompt_ids=list(getattr(update, "prompt_ids", [])),
                      response="\n".join(f"{name}: log-probability {value:.3f}"
                                         for name, value in logprobs.items()),
                      metrics=copy.deepcopy(scored), scored_label=label)
        yield copy.deepcopy(result)
    probabilities = label_distribution(logprobs)
    prediction = max(LABELS, key=lambda label: logprobs[label])
    result.update(prediction=prediction, probabilities=probabilities, logprobs=logprobs,
                  confidence=probabilities[prediction],
                  answer_tokens={label: len(ids) - lead_tokens for label, ids in label_ids.items()},
                  response=prediction, metrics=copy.deepcopy(traces[prediction]),
                  scored_label=prediction, feedback="", status="completed")


def _measure(session, messages, forced, sampling):
    """Prefill the prompt and one replayed answer, and stop at its measurement.

    The runtime yields the replayed prefix before it samples anything, so the
    first update already carries every metric this needs and closing there
    costs the batch no token at all. The sampling settings travel with the
    call so a saved run describes what was really asked for, even though
    nothing after the prefix is ever drawn.
    """
    stream = session.generate(messages, forced_ids=forced, **sampling)
    try:
        return next(iter(stream), None)
    finally:
        stream.close()
