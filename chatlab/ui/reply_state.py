"""Preparation and transcript state for a reply, without component rendering.

Only model updates commit a replay over its previous transcript. Keeping that
transition here makes cancellation and failure independent of the UI frames
used to display them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chatlab.conversation import copy_turns, make_turn, model_messages, split_response_text
from chatlab.generation_request import Metric, ReplyRequest, Turn
from chatlab.seeds import resolve_seed
from chatlab.steering import compact as compact_steering, from_controls as steering_from_controls
from chatlab.text_generation import GenerationUpdate
from chatlab.ui.common import finalize_partial

POSITION_LIMIT_NOTE = (
    "The reply stopped at the last position this model can attend to. "
    "Shorten the conversation to let it write more."
)


def generation_progress(count: int, started: float, seed: int) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    plural = "" if count == 1 else "s"
    return f"{count} token{plural} · {elapsed:.1f}s · {count / elapsed:.1f} tok/s · seed {seed}"


@dataclass
class ReplyState:
    request: ReplyRequest
    turns: list[Turn]
    messages: list[dict[str, Any]]
    pending: Turn
    steering: dict[str, Any] | None
    used_seed: int
    generation: int | None
    previous_turns: list[Turn] | None
    preserving_previous: bool
    started: float = 0.0
    raw_text: str = ""
    prefilled: bool = False
    metrics: list[Metric] = field(default_factory=list)
    status: str = "The model produced no tokens."
    first: bool = True
    forced_prefix_tokens: int = 0
    position_limited: bool = False
    literal_prefill: str = ""
    literal_spans: tuple[tuple[int, int], ...] = ()
    last_update: GenerationUpdate | None = None

    @classmethod
    def prepare(cls, request: ReplyRequest, stamp: Callable[[], int]) -> ReplyState:
        """Resolve controls once, before publishing the opening frame."""
        turns = copy_turns(request.turns)
        used_seed = resolve_seed(request.sampling.seed, request.sampling.randomize_seed)
        preserving_previous = request.previous_turns is not None
        # A replay retains the old diagnostics until its first update succeeds.
        generation = None if preserving_previous else stamp()
        messages = model_messages(
            turns,
            system_prompt=request.prompt.system_prompt,
            include_reasoning=request.prompt.keep_reasoning,
        )
        controls = request.steering
        steering = compact_steering(
            steering_from_controls(
                controls.vector,
                controls.enabled,
                controls.strength,
                controls.layer,
            )
        )
        pending = make_turn("assistant", "", "")
        pending["generation_settings"] = request.sampling.record(used_seed) | {
            "system_prompt": request.prompt.system_prompt,
            "keep_reasoning": bool(request.prompt.keep_reasoning),
            "assistant_prefill": request.applied_prefill,
            "thinking_mode": request.thinking_mode,
        }
        if request.fork_origin is not None:
            # Consumed only when a successful replay becomes visible; the
            # durable origin belongs to its branch, not this pending turn.
            pending["_fork_origin"] = request.fork_origin
        if steering is not None:
            pending["steering"] = steering
        pending["reasoning_closed"] = True
        if request.replay.prompt_edit is not None:
            pending["prompt_edit"] = request.replay.prompt_edit
        turns.append(pending)
        return cls(
            request=request,
            turns=turns,
            messages=messages,
            pending=pending,
            steering=steering,
            used_seed=used_seed,
            generation=generation,
            previous_turns=request.previous_turns,
            preserving_previous=preserving_previous,
        )

    @property
    def stream_note(self) -> str:
        return self.request.note or (
            "Assistant prefill applied." if self.request.applied_prefill else ""
        )

    @property
    def edit_note(self) -> str:
        edit = self.request.replay.prompt_edit
        if not edit:
            return ""
        return (
            f"Token {edit['position']} was replaced with {edit['replacement']!r}; "
            "the next message is prompted from the conversation as usual."
        )

    @property
    def visible_turns(self) -> list[Turn]:
        return self.previous_turns if self.previous_turns is not None else self.turns

    def accept(self, update: GenerationUpdate, stamp: Callable[[], int]) -> None:
        """Commit an update without borrowing identity from the current model."""
        if self.generation is None:
            self.generation = stamp()
        self.previous_turns = None
        self.last_update = update
        self.raw_text = update.text
        self.prefilled = update.reasoning_prefilled
        self.forced_prefix_tokens = update.forced_prefix_tokens
        if update.literal_prefill_text:
            self.literal_prefill = update.literal_prefill_text
        if update.literal_text_spans:
            self.literal_spans = update.literal_text_spans
        reasoning, answer, closed = split_response_text(
            self.raw_text,
            literal_prefill=self.literal_prefill,
            literal_spans=self.literal_spans,
            streaming=True,
            reasoning_prefilled=self.prefilled,
        )
        if update.thinking_mode is not None:
            self.pending["thinking_mode"] = update.thinking_mode
        self.pending.update(reasoning=reasoning, content=answer, reasoning_closed=closed)
        # The runtime owns its live list. Published frames and turns must keep
        # the measurements as they stood, even after that runtime appends.
        self.metrics = list(update.metrics)
        self.pending.update(
            tokens=self.metrics,
            load_id=update.load_id,
            metrics_generation=self.generation,
            ends_on_stop_token=update.ends_on_stop_token,
            generated_tokens=len(self.metrics),
        )
        if self.request.replay.single_step:
            self.pending["token_step_paused"] = bool(self.metrics and not update.ends_on_stop_token)
        self.status = generation_progress(len(self.metrics), self.started, self.used_seed)
        if self.stream_note:
            self.status = f"{self.stream_note} {self.status}"
        self.position_limited = update.ends_on_position_limit
        if self.position_limited:
            self.status = f"{self.status} {POSITION_LIMIT_NOTE}"
        if self.first:
            if update.model_id:
                self.pending["model"] = update.model_id
            self.pending["prompt_tokens"] = len(update.prompt_ids)

    def finish_text(self) -> None:
        reasoning, answer, _ = split_response_text(
            self.raw_text,
            literal_prefill=self.literal_prefill,
            literal_spans=self.literal_spans,
            reasoning_prefilled=self.prefilled,
        )
        self.pending.update(reasoning=reasoning, content=answer)

    def fail(self, error: Exception) -> None:
        self.finish_text()
        self.pending["error"] = str(error) or type(error).__name__
        finalize_partial(self.turns)

    def complete(self) -> bool:
        self.finish_text()
        if (
            self.request.replay.single_step
            and self.metrics
            and not self.pending.get("ends_on_stop_token")
        ):
            # A measured whitespace/marker step must survive so the next
            # click can advance it, even when no visible answer exists yet.
            self.pending.update(token_step_paused=True, reasoning_closed=True)
            return True
        return bool(finalize_partial(self.turns))

    def sampling_record(self) -> dict[str, Any]:
        """Export provenance, retaining requested and applied template modes."""
        sampling = self.request.sampling.record(self.used_seed)
        sampling["requested_thinking_mode"] = self.request.prompt.thinking_mode or "default"
        if self.pending.get("thinking_mode") is not None:
            sampling["thinking_mode"] = self.pending["thinking_mode"]
        if self.steering is not None:
            sampling["steering"] = self.steering
        if self.forced_prefix_tokens:
            sampling["forced_prefix_tokens"] = self.forced_prefix_tokens
        if self.request.applied_prefill:
            sampling["assistant_prefill"] = self.request.applied_prefill
        if self.position_limited:
            sampling["position_limit"] = True
        edit = self.request.replay.prompt_edit
        if edit:
            # The edited prompt's full IDs are the exact input; rendering the
            # recorded messages again would lose both edits and template mode.
            sampling["edited_prompt"] = {
                "position": edit["position"],
                "original": edit["original"],
                "replacement": edit["replacement"],
                "prompt_token_ids": [int(value) for value in edit["ids"]],
            }
        return sampling
