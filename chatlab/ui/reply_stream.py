"""Stream orchestration over typed requests, transcript state, and rendering.

The caller owns the generation reservation. This controller owns iteration,
closure on cancellation, and the distinction between rollback and a kept
partial response.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator

from chatlab.generation_request import ReplyRequest
from chatlab.model_errors import ModelChanged
from chatlab.steering import SteeringError
from chatlab.ui import runtime
from chatlab.ui.common import failure_status
from chatlab.ui.outputs import Frame
from chatlab.ui.panel import new_metrics_generation
from chatlab.ui.reply_render import ReplyRenderer
from chatlab.ui.reply_state import ReplyState

logger = logging.getLogger(__name__)


def stream_reply(request: ReplyRequest) -> Generator[Frame, None, None]:
    """Orchestrate one reply with the caller's generation slot held.

    Preparation, transcript transitions, and component rendering are separate
    adapters. This controller owns stream closure and the rollback/error paths.
    """
    state = ReplyState.prepare(request, new_metrics_generation)
    renderer = ReplyRenderer(state)
    yield renderer.opening(runtime.MANAGER.load_id)
    state.started = time.monotonic()
    stream = runtime.MANAGER.generate(
        state.messages,
        **request.arguments(state.used_seed, state.steering),
    )
    try:
        # Closing on cancellation releases the runtime's model lock immediately.
        with contextlib.closing(stream):
            for update in stream:
                state.accept(update, new_metrics_generation)
                yield renderer.update()
                state.first = False
    except ModelChanged:
        # The branch handler still owns its old transcript and restores it.
        raise
    except Exception as error:
        if state.first and isinstance(error, SteeringError):
            raise
        logger.exception("Generation failed")
        status = failure_status("Generation failed", str(error))
        if state.previous_turns is not None:
            yield renderer.previous(status, busy=False)
            return
        state.fail(error)
        yield renderer.snapshot(status, busy=False)
        return

    kept = state.complete()
    trace = renderer.trace(kept, runtime.MANAGER.loaded_model) if kept and state.metrics else {}
    if trace:
        state.status = f"{state.status} Exports are ready."
    yield renderer.snapshot(
        state.status,
        busy=False,
        charts_panel=renderer.chart_panel(),
        trace=trace,
    )
