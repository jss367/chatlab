"""ChatLab's local HTTP API.

What the interface does to the model in memory, addressable from a script: an
OpenAI-compatible chat completion that also carries ChatLab's own per-token
measurements, and the scoring pass behind the **Score text** tab. Point any
OpenAI client at ``http://127.0.0.1:<port>/v1`` - the same port the interface
is served on - and it answers.

It answers for the model already loaded and does not load one. A load takes
minutes, replaces whatever is in memory, and can be refused for want of it;
that belongs to the Models page, where the reader can see it happen. A request
that names another model is refused by name rather than quietly answered by
the wrong weights.

Only one generation runs at a time, as in the interface: the model lock allows
one and the second request is told the model is busy rather than left queued
behind an answer that may be thousands of tokens long.

There is no authentication, and there is none in the interface either: both
are served on the loopback address, and anything that can reach one can
already do everything the other can.
"""

from __future__ import annotations

import json
import logging
import math
import time
from itertools import chain
from typing import Any, Iterator
from uuid import uuid4

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

import settings
from conversation import split_reasoning
from model_runtime import (
    InsufficientMemoryError,
    ModelChanged,
    OutOfMemoryError,
    cache_status,
    list_cached_models,
    device_label,
    device_profile,
)
from token_metrics import summarize
from ui import runtime
from ui.generation import resolve_seed

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"

# What an impossible token's log probability is reported as. A probability of
# zero has no logarithm, and JSON has no negative infinity that every client
# can read, so the figure is floored the way OpenAI floors it.
LOGPROB_FLOOR = -9999.0

ROLES = ("system", "user", "assistant")


class ApiError(Exception):
    """A request that cannot be answered, and the status that says why."""

    def __init__(self, status: int, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


def error_response(error: ApiError) -> JSONResponse:
    """The failure as an OpenAI client expects to read it."""

    return JSONResponse(
        status_code=error.status,
        content={"error": {"message": error.message, "type": error.kind}},
    )


def loaded_model(requested: Any) -> str:
    """The model that will answer, or an :class:`ApiError` saying why none will.

    A request may name the loaded model or leave the field out. Naming another
    is refused: the alternative is unloading a reader's model and spending
    minutes on a load nobody watching the interface asked for.
    """

    in_memory = runtime.MANAGER.model_id
    if not in_memory:
        raise ApiError(
            409,
            "No model is loaded. Load one on ChatLab's Models page first; the "
            "API does not load models.",
            "model_not_loaded",
        )
    if requested is None or requested == in_memory:
        return in_memory
    if not isinstance(requested, str):
        raise ApiError(400, "The model must be named as a string.")
    raise ApiError(
        409,
        f"{requested} is not the model in memory. ChatLab has {in_memory} "
        "loaded and answers with that one; load another on the Models page.",
        "model_not_loaded",
    )


def _flag(body: dict, name: str, default: bool = False) -> bool:
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise ApiError(400, f"{name} must be true or false.")
    return value


def _positive_int(body: dict, name: str) -> int | None:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ApiError(400, f"{name} must be a whole number of 1 or more.")
    return value


# What a request may ask for, and the setting each one overrides. The
# response length answers to OpenAI's name for it as well as its own.
SAMPLING_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_new_tokens",
    "max_completion_tokens": "max_new_tokens",
}


def sampling_from(body: dict) -> dict:
    """The sampling one request asks for, over the values the app is set to.

    A field left out takes the reader's own saved setting rather than a
    hard-coded default, so a script and the interface answer alike unless the
    script says otherwise. What a value may be is the settings module's
    business, and a value it would have had to change is refused here rather
    than clamped in silence: a script that asks for 40,000 tokens should hear
    about the context limit, not receive 8,192 of them and no explanation.
    """

    saved = settings.current()
    given: dict[str, Any] = {}
    for name, setting in SAMPLING_FIELDS.items():
        value = body.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError(400, f"{name} must be a number.")
        given[setting] = value
    checked = settings.sanitize(saved.to_mapping() | given)
    for setting, value in given.items():
        if abs(float(getattr(checked, setting)) - float(value)) > 1e-9:
            raise ApiError(
                400,
                f"{setting} of {value} is outside what ChatLab allows: the "
                f"nearest it would take is {getattr(checked, setting)}.",
            )
    seed = body.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ApiError(400, "seed must be a whole number of 0 or more.")
    return {
        "temperature": checked.temperature,
        "top_p": checked.top_p,
        "top_k": checked.top_k,
        "max_new_tokens": checked.max_new_tokens,
        "seed": int(seed)
        if seed is not None
        else resolve_seed(saved.seed, saved.randomize_seed),
    }


def conversation_from(body: dict) -> tuple[list[dict], str]:
    """The messages to answer, and any prefill the last of them asks for.

    A trailing assistant message is the prefill the interface calls
    **Assistant prefill**: the reply must begin with that text, and its tokens
    are measured against the model's own distribution as replayed tokens
    rather than sampled ones. Every other message is a turn.
    """

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list.")
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            raise ApiError(400, "Every message must be an object.")
        role = message.get("role")
        if role not in ROLES:
            raise ApiError(400, f"Unsupported message role: {role!r}.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ApiError(
                400,
                "Message content must be a string. ChatLab measures text and "
                "takes no other kind of content.",
            )
        turns.append({"role": role, "content": content})
    prefill = ""
    if turns[-1]["role"] == "assistant":
        prefill = turns.pop()["content"]
        if not turns:
            raise ApiError(
                400, "A prefilled assistant message needs a conversation before it."
            )
    return turns, prefill


def logprob_of(probability: Any) -> float:
    """A probability as the natural logarithm an OpenAI client reads."""

    try:
        value = float(probability)
    except (TypeError, ValueError):
        return LOGPROB_FLOOR
    return math.log(value) if value > 0 else LOGPROB_FLOOR


# The measurements that travel beside every token, under ChatLab's own key:
# the same figures the token panel shows, so a script and the interface are
# reading one set of numbers.
CHATLAB_TOKEN_FIELDS = (
    "position",
    "raw_rank",
    "raw_probability",
    "sampling_probability",
    "surprise_bits",
    "entropy_bits",
    "top1_margin",
    "probability_mass_above",
    "sampling_shift_bits",
    "scored",
    "unscored_reason",
)


def token_entry(metric: dict, top_logprobs: int = 0) -> dict:
    """One token as ``logprobs.content`` spells it, ChatLab's numbers included."""

    text = metric.get("text") or ""
    entry = {
        "token": text,
        "logprob": logprob_of(metric.get("raw_probability")),
        "bytes": list(text.encode("utf-8")),
        "top_logprobs": [
            {
                "token": candidate.get("text") or "",
                "logprob": logprob_of(candidate.get("probability")),
                "bytes": list((candidate.get("text") or "").encode("utf-8")),
            }
            for candidate in (metric.get("top_candidates") or [])[:top_logprobs]
        ],
        "chatlab": {
            name: metric[name] for name in CHATLAB_TOKEN_FIELDS if name in metric
        },
    }
    entry["chatlab"]["token_id"] = metric.get("token_id")
    return entry


def token_detail(body: dict) -> tuple[bool, int]:
    """Whether to measure every token, and how many alternatives each carries.

    ``logprobs`` is what an OpenAI client sets to ask for per-token figures,
    and it is what turns on ChatLab's own measurements beside them;
    ``top_logprobs`` asks for that many of the alternatives the model ranked.
    ChatLab records a fixed number of alternatives per token, so a request
    for more than it kept gets what it kept.
    """

    wanted = _flag(body, "logprobs")
    count = body.get("top_logprobs", 0)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 20:
        raise ApiError(400, "top_logprobs must be a whole number between 0 and 20.")
    if count and not wanted:
        raise ApiError(400, "top_logprobs needs logprobs to be true.")
    return wanted, count


def finish_reason(generated: int, forced: int, limit: int) -> str:
    """``length`` where the response ran into its ceiling, ``stop`` otherwise."""

    return "length" if generated - forced >= limit else "stop"


def build_router() -> APIRouter:
    """The ``/v1`` routes, ready to be included in the application."""

    router = APIRouter(prefix=API_PREFIX, tags=["chatlab"])

    @router.get("/models")
    def models() -> JSONResponse:
        """Every model on disk ChatLab could answer with, the loaded one marked."""

        data = []
        for entry in list_cached_models():
            if not entry.status.complete:
                continue
            organization, _, _name = entry.model_id.partition("/")
            data.append(
                {
                    "id": entry.model_id,
                    "object": "model",
                    "created": int(entry.updated or 0),
                    "owned_by": organization or "chatlab",
                    "chatlab": {
                        "loaded": entry.model_id == runtime.MANAGER.model_id,
                        "size_bytes": entry.size_bytes,
                        "architecture": entry.architecture,
                    },
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @router.get("/chatlab/status")
    def status() -> JSONResponse:
        """Whether a request can be answered right now, and by what.

        The one call worth making before the others: it names the model in
        memory, says whether a generation is already running, and reports the
        machine the way the Settings page's hardware panel does.
        """

        profile = device_profile()
        return JSONResponse(
            {
                "model": runtime.MANAGER.model_id,
                "device": runtime.MANAGER.device_name or device_label(profile.backend),
                "precision": runtime.MANAGER.precision,
                "busy": runtime.MANAGER.busy,
                "memory": {
                    "total_bytes": profile.total,
                    "available_bytes": profile.available,
                    "device_ceiling_bytes": profile.ceiling,
                    "pool": profile.pool,
                },
            }
        )

    @router.post("/chat/completions")
    def chat_completions(body: dict = Body(default_factory=dict)):
        """Answer a conversation, with every token's measurements if asked."""

        try:
            model_id = loaded_model(body.get("model"))
            turns, prefill = conversation_from(body)
            sampling = sampling_from(body)
            measured, wants = token_detail(body)
            streaming = _flag(body, "stream")
            prompt_logprobs = _flag(body, "prompt_logprobs")
        except ApiError as error:
            return error_response(error)

        if not runtime.MANAGER.reserve_generation():
            return error_response(
                ApiError(
                    409,
                    "ChatLab is generating a response already. Only one runs at "
                    "a time; try again when it has finished.",
                    "model_busy",
                )
            )
        request_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())
        stream = runtime.MANAGER.generate(
            turns,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            max_new_tokens=sampling["max_new_tokens"],
            seed=sampling["seed"],
            analyze_prompt=prompt_logprobs,
            answer_prefill=prefill,
        )
        # The first frame is drawn here rather than inside the response,
        # because everything a request can be refused for - a prompt past the
        # context limit, a prefill the tokenizer cannot reproduce, a model
        # that has gone - is raised on the way to it. Streaming those as an
        # event inside a 200 would tell a client the request succeeded.
        try:
            first = next(stream)
        except StopIteration:
            runtime.MANAGER.release_generation()
            return error_response(
                ApiError(500, "The model produced nothing at all.", "server_error")
            )
        except ApiError as error:
            runtime.MANAGER.release_generation()
            return error_response(error)
        except Exception as error:
            runtime.MANAGER.release_generation()
            return error_response(refusal(error))
        frames = chain([first], stream)
        if streaming:
            return StreamingResponse(
                stream_completion(
                    frames, request_id, created, model_id, sampling, measured, wants
                ),
                media_type="text/event-stream",
            )
        try:
            return JSONResponse(
                whole_completion(
                    frames,
                    request_id,
                    created,
                    model_id,
                    sampling,
                    measured,
                    wants,
                    prompt_logprobs,
                )
            )
        except ApiError as error:
            return error_response(error)
        finally:
            runtime.MANAGER.release_generation()

    @router.post("/chatlab/score")
    def score(body: dict = Body(default_factory=dict)):
        """Measure text the model did not write, as the Score text tab does."""

        try:
            model_id = loaded_model(body.get("model"))
            text = body.get("text")
            if not isinstance(text, str):
                raise ApiError(400, "text must be a string.")
            context = body.get("context") or ""
            if not isinstance(context, str):
                raise ApiError(400, "context must be a string.")
            use_template = _flag(body, "use_chat_template")
            _measured, wants = token_detail(body)
        except ApiError as error:
            return error_response(error)
        try:
            scored = runtime.MANAGER.score_text(
                text, context=context, use_chat_template=use_template
            )
        except ValueError as error:
            return error_response(ApiError(400, str(error)))
        except Exception as error:
            return error_response(refusal(error))
        return JSONResponse(
            {
                "object": "chatlab.score",
                "model": model_id,
                "tokens": [token_entry(metric, wants) for metric in scored.metrics],
                "context_tokens": [
                    token_entry(metric, wants) for metric in scored.context_metrics
                ],
                "summary": summarize(scored.metrics),
                "seam_verified": scored.seam_verified,
                "chat_template_missing": scored.chat_template_missing,
            }
        )

    return router


def refusal(error: Exception) -> ApiError:
    """The status a failure from the runtime deserves.

    A prompt too long for the context limit, or a model that will not fit, is
    the request's own doing and says so; anything else is the server's.
    """

    if isinstance(error, ValueError):
        return ApiError(400, str(error))
    if isinstance(error, (InsufficientMemoryError, OutOfMemoryError)):
        return ApiError(507, str(error), "insufficient_memory")
    if isinstance(error, ModelChanged):
        return ApiError(409, str(error), "model_not_loaded")
    logger.exception("API request failed")
    return ApiError(500, str(error) or error.__class__.__name__, "server_error")


def answer_and_reasoning(update) -> tuple[str, str]:
    """The visible answer and the reasoning block, split as the chat splits them."""

    reasoning, answer, _closed = split_reasoning(
        update.text, reasoning_prefilled=update.reasoning_prefilled
    )
    return answer, reasoning


def whole_completion(
    stream: Iterator,
    request_id: str,
    created: int,
    model_id: str,
    sampling: dict,
    measured: bool,
    wants: int,
    prompt_logprobs: bool,
) -> dict:
    """Run the generation to its end and answer with the whole of it."""

    last = None
    try:
        for update in stream:
            last = update
    except Exception as error:
        raise refusal(error) from error
    if last is None:
        raise ApiError(500, "The model produced nothing at all.", "server_error")

    answer, reasoning = answer_and_reasoning(last)
    metrics = list(last.metrics)
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    if reasoning:
        message["reasoning_content"] = reasoning
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason(
            len(metrics), last.forced_prefix_tokens, sampling["max_new_tokens"]
        ),
        "logprobs": {"content": [token_entry(metric, wants) for metric in metrics]}
        if measured
        else None,
    }
    payload = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(last.prompt_ids),
            "completion_tokens": len(metrics),
            "total_tokens": len(last.prompt_ids) + len(metrics),
        },
        "chatlab": {
            "seed": sampling["seed"],
            "device": runtime.MANAGER.device_name,
            "precision": runtime.MANAGER.precision,
            "replayed_tokens": last.forced_prefix_tokens,
            "summary": summarize(metrics),
        },
    }
    if prompt_logprobs:
        payload["chatlab"]["prompt_tokens"] = [
            token_entry(metric, wants) for metric in last.prompt_metrics
        ]
    return payload


def _chunk(request_id: str, created: int, model_id: str, choice: dict) -> str:
    """One server-sent event, in the shape a streaming OpenAI client reads."""

    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_completion(
    stream: Iterator,
    request_id: str,
    created: int,
    model_id: str,
    sampling: dict,
    measured: bool,
    wants: int,
) -> Iterator[str]:
    """The same answer as it arrives, one event per batch of tokens.

    The runtime hands back the whole text each frame; a client reading a
    stream wants only what is new, so each frame is diffed against the last.
    Reasoning goes to ``reasoning_content`` and the answer to ``content``, so
    a client that shows only the answer never has to strip the markers.
    """

    sent_answer = sent_reasoning = ""
    sent_tokens = 0
    last = None
    try:
        yield _chunk(
            request_id,
            created,
            model_id,
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
        )
        for update in stream:
            last = update
            answer, reasoning = answer_and_reasoning(update)
            delta: dict[str, Any] = {}
            if reasoning.startswith(sent_reasoning) and reasoning != sent_reasoning:
                delta["reasoning_content"] = reasoning[len(sent_reasoning) :]
            elif reasoning != sent_reasoning:
                # The split moved a word from one side to the other, which a
                # diff cannot express; send the whole block again.
                delta["reasoning_content"] = reasoning
            if answer.startswith(sent_answer) and answer != sent_answer:
                delta["content"] = answer[len(sent_answer) :]
            elif answer != sent_answer:
                delta["content"] = answer
            sent_answer, sent_reasoning = answer, reasoning
            metrics = list(update.metrics)
            choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": None}
            fresh = metrics[sent_tokens:]
            if measured and fresh:
                choice["logprobs"] = {
                    "content": [token_entry(metric, wants) for metric in fresh]
                }
            sent_tokens = len(metrics)
            if delta or "logprobs" in choice:
                yield _chunk(request_id, created, model_id, choice)
        metrics = list(last.metrics) if last is not None else []
        final = {
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason(
                len(metrics),
                last.forced_prefix_tokens if last is not None else 0,
                sampling["max_new_tokens"],
            ),
        }
        prompt_tokens = len(last.prompt_ids) if last is not None else 0
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [final],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(metrics),
                "total_tokens": prompt_tokens + len(metrics),
            },
            "chatlab": {
                "seed": sampling["seed"],
                "replayed_tokens": last.forced_prefix_tokens if last is not None else 0,
                "summary": summarize(metrics),
            },
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as error:
        # The stream has already been given a 200, so a failure part way
        # through can only be said in the stream itself.
        failure = refusal(error)
        yield f"data: {json.dumps({'error': {'message': failure.message, 'type': failure.kind}})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        runtime.MANAGER.release_generation()


def attach(app) -> None:
    """Serve the API from ``app``, the interface's own FastAPI application.

    One port for both: the interface's routes are all named ones and none of
    them is a catch-all, so ``/v1`` can be added after Gradio has built its
    application and reaches these routes rather than the page.
    """

    app.include_router(build_router())
    logger.info("Serving the ChatLab API under %s", API_PREFIX)
