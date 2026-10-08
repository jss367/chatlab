"""Typed inputs to a reply, independent of Gradio and mutable stream state.

The positional controls are decoded once at the UI boundary. Everything past
that boundary receives these named options. Persisted turns and measurements
retain their existing JSON schema; typing this request does not migrate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from chatlab.token_metrics import DEFAULT_COLOR_SCALE

Turn = dict[str, Any]
Metric = dict[str, Any]


class PromptEdit(TypedDict):
    ids: tuple[int, ...] | list[int]
    position: int
    original: str
    replacement: str


class GenerationArguments(TypedDict):
    temperature: float
    top_p: float
    top_k: int
    skip_top_below: float
    max_new_tokens: int
    seed: int
    analyze_prompt: bool
    forced_ids: tuple[int, ...]
    prompt_override_ids: tuple[int, ...] | list[int] | None
    answer_prefill: str
    thinking_mode: str
    literal_prefill_tokens: int
    automatic_reasoning_close_tokens: int
    literal_text_ranges: tuple[tuple[int, int], ...]
    load_id: str | None
    steering: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class SamplingOptions:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    skip_top_below: float = 0.0
    max_new_tokens: int = 1024
    # The UI can supply None, a float, or an invalid value. Seed resolution
    # deliberately retains its existing fallback behavior at preparation time.
    seed: object = 42
    randomize_seed: bool = True

    def record(self, used_seed: int) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "skip_top_below": float(self.skip_top_below),
            "max_new_tokens": int(self.max_new_tokens),
            "seed": used_seed,
        }


@dataclass(frozen=True)
class PromptOptions:
    system_prompt: str = ""
    keep_reasoning: bool = False
    assistant_prefill: str = ""
    analyze_prompt: bool = True
    thinking_mode: str = "default"


@dataclass(frozen=True)
class SteeringOptions:
    vector: dict[str, Any] | None = None
    enabled: bool | None = None
    strength: float | None = None
    layer: int | None = None


@dataclass(frozen=True)
class ReplayOptions:
    forced_ids: tuple[int, ...] = ()
    prompt_edit: PromptEdit | None = None
    replaying: bool = False
    literal_prefill_tokens: int = 0
    automatic_reasoning_close_tokens: int = 0
    literal_text_ranges: tuple[tuple[int, int], ...] = ()
    expected_load_id: str | None = None
    single_step: bool = False
    thinking_mode: str | None = None


@dataclass(frozen=True)
class ReplyRequest:
    turns: list[Turn]
    prompt_text: str
    prompt: PromptOptions = field(default_factory=PromptOptions)
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    steering: SteeringOptions = field(default_factory=SteeringOptions)
    replay: ReplayOptions = field(default_factory=ReplayOptions)
    scale_name: str = DEFAULT_COLOR_SCALE
    note: str = ""
    previous_turns: list[Turn] | None = None
    fork_origin: dict[str, Any] | None = None

    @property
    def thinking_mode(self) -> str:
        override = self.replay.thinking_mode
        return override if override is not None else self.prompt.thinking_mode

    @property
    def applied_prefill(self) -> str:
        return "" if self.replay.replaying else self.prompt.assistant_prefill

    def arguments(self, used_seed: int, steering: dict[str, Any] | None) -> GenerationArguments:
        """The exact runtime inputs, also used to record this reply's settings."""
        sampling, replay = self.sampling, self.replay
        arguments: GenerationArguments = {
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "skip_top_below": float(sampling.skip_top_below),
            "max_new_tokens": int(sampling.max_new_tokens),
            "seed": used_seed,
            "analyze_prompt": bool(self.prompt.analyze_prompt),
            "forced_ids": tuple(int(value) for value in replay.forced_ids),
            "prompt_override_ids": replay.prompt_edit["ids"] if replay.prompt_edit else None,
            "answer_prefill": self.applied_prefill,
            "thinking_mode": self.thinking_mode,
            "literal_prefill_tokens": replay.literal_prefill_tokens,
            "automatic_reasoning_close_tokens": replay.automatic_reasoning_close_tokens,
            "literal_text_ranges": replay.literal_text_ranges,
            "load_id": replay.expected_load_id,
        }
        if steering is not None:
            arguments["steering"] = steering
        return arguments


def request_from_controls(
    turns: list[Turn],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    assistant_prefill: str,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_top_below: float,
    max_new_tokens: int,
    seed: object,
    randomize_seed: bool,
    analyze_prompt: bool = True,
    scale_name: str = DEFAULT_COLOR_SCALE,
    steering: dict[str, Any] | None = None,
    steering_enabled: bool | None = None,
    steering_strength: float | None = None,
    steering_layer: int | None = None,
    thinking_mode: str = "default",
    *,
    replay: ReplayOptions | None = None,
    note: str = "",
    previous_turns: list[Turn] | None = None,
    fork_origin: dict[str, Any] | None = None,
) -> ReplyRequest:
    """Adapt Gradio's control order; internal callers use the grouped request."""
    return ReplyRequest(
        turns=turns,
        prompt_text=prompt_text,
        prompt=PromptOptions(
            system_prompt=system_prompt,
            keep_reasoning=keep_reasoning,
            assistant_prefill=assistant_prefill,
            analyze_prompt=analyze_prompt,
            thinking_mode=thinking_mode,
        ),
        sampling=SamplingOptions(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_top_below=skip_top_below,
            max_new_tokens=max_new_tokens,
            seed=seed,
            randomize_seed=randomize_seed,
        ),
        steering=SteeringOptions(
            vector=steering,
            enabled=steering_enabled,
            strength=steering_strength,
            layer=steering_layer,
        ),
        replay=replay if replay is not None else ReplayOptions(),
        scale_name=scale_name,
        note=note,
        previous_turns=previous_turns,
        fork_origin=fork_origin,
    )
