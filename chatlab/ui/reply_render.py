"""Project reply state into Gradio frames and exported traces.

Rendering never advances a model iterator or commits a replay. The stream
controller decides when to publish; this adapter owns component resets,
batched charts, and the context attached to token inspection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gradio as gr

from chatlab import charts
from chatlab.conversation import copy_turns, display_messages
from chatlab.token_metrics import summarize
from chatlab.trace_export import build_trace
from chatlab.vision import message_images
from chatlab.ui.common import CHART_EVERY, NO_TOKEN_SELECTED, send_stop_values
from chatlab.ui.outputs import CHAT_OUTPUT_NAMES, Frame
from chatlab.ui.panel import prompt_note_text, strip_update, transcript_update, transcript_visible
from chatlab.ui.reply_state import ReplyState

if TYPE_CHECKING:
    from chatlab.model_loading import LoadedModel


class ReplyRenderer:
    def __init__(self, state: ReplyState) -> None:
        self.state = state
        self.recorded_context: tuple[Any, ...] | None = None

    def snapshot(
        self,
        status: str,
        *,
        busy: bool = True,
        reset_details: bool = False,
        prompt_panel: tuple[Any, Any, Any] | None = None,
        charts_panel: tuple[Any, Any] | None = None,
        trace: dict[str, Any] | None = None,
        context_ids: Any = None,
    ) -> Frame:
        state, request = self.state, self.state.request
        turns = state.visible_turns
        messages, _ = display_messages(turns)
        prompt_strip, prompt_metrics, prompt_note = prompt_panel or (
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        summary_panel, surprise_panel = charts_panel or (gr.skip(), gr.skip())
        if context_ids is None:
            context_ids = gr.skip()
        return Frame(
            CHAT_OUTPUT_NAMES,
            prompt=request.prompt_text,
            chatbot=messages,
            turns=copy_turns(turns),
            strip=transcript_update(turns, request.scale_name)
            if transcript_visible()
            else gr.skip(),
            metrics=(state.generation, state.metrics),
            status=status,
            seed=state.used_seed,
            **send_stop_values(busy),
            detail=NO_TOKEN_SELECTED if reset_details else gr.skip(),
            # Gradio mutates raw dataframe values while applying diffs. An
            # update envelope prevents a later skip from erasing its headers.
            alternatives=gr.update(value=[]) if reset_details else gr.skip(),
            prompt_strip=prompt_strip,
            prompt_metrics=prompt_metrics,
            prompt_note=prompt_note,
            summary=summary_panel,
            surprise=surprise_panel,
            trace=gr.skip() if trace is None else trace,
            context_ids=context_ids,
            chat_context_ids=context_ids,
            chat_metrics=(state.generation, state.metrics),
            selected_token=None if reset_details else gr.skip(),
            branch_pick=None if reset_details else gr.skip(),
        )

    def previous(self, status: str, *, busy: bool) -> Frame:
        """Keep the old transcript and diagnostics while replay is uncommitted."""
        state = self.state
        messages, _ = display_messages(state.visible_turns)
        return Frame(
            CHAT_OUTPUT_NAMES,
            prompt=state.request.prompt_text,
            chatbot=messages,
            turns=copy_turns(state.visible_turns),
            status=status,
            **send_stop_values(busy),
        )

    def opening(self, load_id: str | None) -> Frame:
        state = self.state
        status = f"{state.stream_note} Generating…".strip()
        if state.preserving_previous:
            return self.previous(status, busy=True)
        return self.snapshot(
            status,
            reset_details=True,
            prompt_panel=(strip_update([], state.request.scale_name), (state.generation, []), ""),
            charts_panel=(charts.summary_tiles({}), charts.EMPTY_CHART),
            trace={},
            context_ids=(state.generation, [], load_id),
        )

    def chart_panel(self) -> tuple[Any, Any]:
        return charts.summary_tiles(summarize(self.state.metrics)), charts.surprise_chart(
            self.state.metrics
        )

    def update(self) -> Frame:
        state, request = self.state, self.state.request
        prompt_panel = None
        context_ids: Any = None
        if state.first:
            update = state.last_update
            assert update is not None
            prompt_metrics = list(update.prompt_metrics)
            prompt_panel = (
                strip_update(prompt_metrics, request.scale_name),
                (state.generation, prompt_metrics),
                prompt_note_text(
                    len(prompt_metrics),
                    " ".join(note for note in (update.prompt_note, state.edit_note) if note),
                    "prompt",
                ),
            )
            # Pictures follow an always-present steering slot when needed.
            # IDs alone cannot recover the image behind each placeholder.
            pictures = tuple(message_images(state.messages))
            context_ids = (
                state.generation,
                [int(v) for v in update.prompt_ids],
                update.load_id,
                *([state.steering] if state.steering is not None or pictures else []),
                *([pictures] if pictures else []),
            )
            self.recorded_context = context_ids
        return self.snapshot(
            state.status,
            reset_details=state.first and state.preserving_previous,
            trace={} if state.first and state.preserving_previous else None,
            prompt_panel=prompt_panel,
            context_ids=context_ids,
            charts_panel=self.chart_panel()
            if state.first or len(state.metrics) % CHART_EVERY == 0
            else None,
        )

    def trace(self, kept: bool, model_snapshot: Callable[[], LoadedModel]) -> dict[str, Any]:
        """Export the producing update's measurements, never a partial failure."""
        state = self.state
        if not kept or not state.metrics:
            return {}
        update = state.last_update
        assert update is not None
        trace: dict[str, Any] = build_trace(
            model_id=state.pending.get("model"),
            messages=state.messages,
            response=state.raw_text,
            sampling=state.sampling_record(),
            metrics=state.metrics,
        )
        trace["prompt_tokens"] = list(update.prompt_metrics)
        from chatlab.ui.compare import _decoded_spans, _tokenizer_identity
        from chatlab.experiment_runs import SESSION_ID

        published = model_snapshot()
        context_ids = self.recorded_context[1] if self.recorded_context else ()
        decoded, token_ends = _decoded_spans(
            state.metrics,
            context_ids,
            state.raw_text,
            update.literal_prefill_tokens,
        )
        trace["run_context"] = {
            "session_id": SESSION_ID,
            "load_id": state.pending.get("load_id"),
            "metrics_generation": state.generation,
            "context_ids": list(context_ids),
            "tokenizer": _tokenizer_identity(),
            "device_name": published.device_name,
            "precision": published.precision,
            "decoded": decoded,
            "token_ends": token_ends,
        }
        return trace
