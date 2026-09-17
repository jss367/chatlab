"""Cancellable batches using the host's exclusive model session."""
import copy
from datetime import datetime, timezone
import threading
from uuid import uuid4

from .benchmark import FORMAT, PAPER, dataset_digest, messages_for, parse_response, report
from .storage import save_json


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

    def run(self, owner, cases, *, max_new_tokens=256, seed=42):
        if not cases:
            raise ValueError("Import cases or load the demonstration first.")
        prompts = [messages_for(case) for case in cases]
        cancel = threading.Event()
        with self._lock:
            if owner in self._active:
                raise ValueError("A batch is already running in this view.")
            self._active[owner] = [cancel, None]
        run = dict(format=FORMAT, id=uuid4().hex, paper=PAPER,
                   created_at=datetime.now(timezone.utc).isoformat(),
                   mode="text_only_adaptation", dataset_sha256=dataset_digest(cases),
                   cases=copy.deepcopy(cases), predictions=[], status="running",
                   sampling=dict(temperature=0.0, top_p=1.0, top_k=0,
                                 max_new_tokens=int(max_new_tokens), seed=int(seed)))
        path = self.data_dir / f"{run['id']}.json"

        def save():
            run["scores"] = report(cases, run["predictions"])
            save_json(path, run)

        try:
            with self.models.open_session() as session:
                with self._lock:
                    self._active[owner][1] = session
                    if cancel.is_set():
                        session.cancel()
                run.update(model_id=session.model_id, load_id=session.load_id)
                for case, messages in zip(cases, prompts):
                    if cancel.is_set():
                        break
                    result = dict(id=case["id"], prediction=None, feedback="",
                                  response="", metrics=[], prompt_ids=[], messages=messages, reasoning_prefilled=False,
                                  status="running")
                    run["predictions"].append(result)
                    stream = session.generate(messages, **run["sampling"])
                    try:
                        for update in stream:
                            result.update(response=update.text, metrics=copy.deepcopy(update.metrics),
                                          prompt_ids=list(getattr(update, "prompt_ids", [])),
                                          reasoning_prefilled=getattr(update, "reasoning_prefilled", False))
                            run["scores"] = report(cases, run["predictions"])
                            yield copy.deepcopy(run), None
                        if cancel.is_set():
                            result["status"] = "cancelled"
                        else:
                            result["prediction"], result["feedback"] = parse_response(
                                result["response"], reasoning_prefilled=result["reasoning_prefilled"])
                            result["status"] = "completed"
                    finally:
                        stream.close()
                    save()
                    yield copy.deepcopy(run), str(path)
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
        yield copy.deepcopy(run), str(path)
