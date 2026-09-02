"""Conversation turns, reasoning blocks, and save/load helpers.

A turn is a plain dictionary so it can live in a ``gr.State`` and be written
straight to JSON:

    {"role": "user" | "assistant", "content": str, "reasoning": str}

``reasoning`` holds the text a Think model wrapped in ``<think>`` tags. It is
kept beside the answer rather than inside it so the interface can collapse it
and so the next request can deliberately include or drop it.
"""

from __future__ import annotations

import json

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
REASONING_TITLE = "Reasoning"
SAVE_FORMAT = "chatlab-conversation-1"
MAIN_BRANCH = "Main"

_PARTIAL_TAGS = tuple(
    sorted(
        {THINK_OPEN[:size] for size in range(1, len(THINK_OPEN))}
        | {THINK_CLOSE[:size] for size in range(1, len(THINK_CLOSE))},
        key=len,
        reverse=True,
    )
)


def _trim_partial_tag(text: str) -> str:
    """Hide a half-emitted reasoning tag while tokens are still arriving."""

    for tag in _PARTIAL_TAGS:
        if text.endswith(tag):
            return text[: -len(tag)]
    return text


def split_reasoning(
    text: str,
    *,
    streaming: bool = False,
    reasoning_prefilled: bool = False,
) -> tuple[str, str, bool]:
    """Split raw model output into ``(reasoning, answer, closed)``.

    ``closed`` is False while the model is still inside a ``<think>`` block, so
    a caller streaming tokens can show the block as pending. When ``streaming``
    is set, a partially emitted tag at the very end is withheld instead of being
    shown as literal text.

    ``reasoning_prefilled`` says the prompt itself ended with the opening
    ``<think>`` marker, which is what OLMo Think templates do. The generated
    text then starts *inside* the reasoning block and carries no opening marker
    at all, so without this flag every token would look like answer text until
    the closing marker finally arrived thousands of tokens later.

    Only that flag can imply a prefilled opener. A closing marker on its own is
    not evidence of one: a model that merely writes ``</think>`` in its prose -
    explaining the marker, or quoting a template - would otherwise have the text
    before it hidden as reasoning and its answer truncated to whatever followed.
    """

    if streaming:
        text = _trim_partial_tag(text)

    reasoning: list[str] = []
    answer: list[str] = []
    closed = True
    rest = text

    if reasoning_prefilled:
        # The chat template supplied the opening tag, so only the close arrives.
        head, marker, rest = rest.partition(THINK_CLOSE)
        reasoning.append(head)
        if not marker:
            # Still inside the prefilled block: everything so far is reasoning.
            closed = False

    while THINK_OPEN in rest:
        head, _, rest = rest.partition(THINK_OPEN)
        answer.append(head)
        if THINK_CLOSE in rest:
            body, _, rest = rest.partition(THINK_CLOSE)
            reasoning.append(body)
        else:
            reasoning.append(rest)
            rest = ""
            closed = False

    answer.append(rest)
    joined_reasoning = "\n\n".join(part.strip() for part in reasoning if part.strip())
    return joined_reasoning, "".join(answer).strip(), closed


def make_turn(role: str, content: str, reasoning: str = "") -> dict:
    return {"role": role, "content": content, "reasoning": reasoning}


def copy_turns(turns: list[dict] | None) -> list[dict]:
    """Snapshot turns so a streaming update cannot mutate stored state."""

    return [dict(turn) for turn in (turns or [])]


def display_messages(
    turns: list[dict] | None,
) -> tuple[list[dict], list[tuple[int, str]]]:
    """Build ``gr.Chatbot(type="messages")`` values plus an index map.

    The map lets retry, edit, and undo translate a chatbot message index back to
    the turn it came from, which matters because a reasoning block is rendered
    as its own extra message.
    """

    messages: list[dict] = []
    index_map: list[tuple[int, str]] = []

    for position, turn in enumerate(turns or []):
        reasoning = turn.get("reasoning") or ""
        content = turn.get("content") or ""
        if reasoning:
            status = "done" if turn.get("reasoning_closed", True) else "pending"
            messages.append(
                {
                    "role": turn["role"],
                    "content": reasoning,
                    "metadata": {"title": REASONING_TITLE, "status": status},
                }
            )
            index_map.append((position, "reasoning"))
        if content or not reasoning:
            messages.append({"role": turn["role"], "content": content})
            index_map.append((position, "content"))

    return messages, index_map


def locate(turns: list[dict] | None, display_index) -> tuple[int, str] | None:
    """Map a chatbot message index onto ``(turn index, part)``."""

    if isinstance(display_index, (list, tuple)):
        display_index = display_index[0] if display_index else None
    try:
        position = int(display_index)
    except (TypeError, ValueError):
        return None
    _, index_map = display_messages(turns)
    if not 0 <= position < len(index_map):
        return None
    return index_map[position]


def user_index_at_or_before(turns: list[dict] | None, position: int) -> int | None:
    """Find the user turn that produced the reply at ``position``."""

    turns = turns or []
    index = min(position, len(turns) - 1)
    while index >= 0:
        if turns[index]["role"] == "user":
            return index
        index -= 1
    return None


def last_user_index(turns: list[dict] | None) -> int | None:
    return user_index_at_or_before(turns, len(turns or []) - 1)


def model_messages(
    turns: list[dict] | None,
    *,
    system_prompt: str = "",
    include_reasoning: bool = False,
) -> list[dict]:
    """Render turns as chat-template messages for the next request.

    Reasoning is dropped by default: Think models are trained to produce a fresh
    ``<think>`` block each turn, and feeding old ones back wastes context and
    tends to derail the next answer.
    """

    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for turn in turns or []:
        content = turn.get("content") or ""
        reasoning = turn.get("reasoning") or ""
        if include_reasoning and reasoning:
            content = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{content}".strip()
        if not content:
            if turn["role"] == "assistant" and reasoning:
                # A Think model stopped mid-answer leaves an assistant turn with
                # reasoning but no text. The visible conversation still shows a
                # reply, so dropping the turn here would hand the model two user
                # messages in a row and break templates that require alternating
                # roles. Keep the slot, empty, since the reasoning is not replayed.
                messages.append({"role": "assistant", "content": ""})
            continue
        messages.append({"role": turn["role"], "content": content})

    return messages


# ------------------------------------------------------------------- forks
#
# A fork is a second copy of the transcript that can be taken somewhere else.
# The forks live beside the conversation as a plain dictionary so they fit in a
# ``gr.State``:
#
#     {"active": name, "branches": {name: [turns...], ...}}
#
# Only the *inactive* branches are current in ``branches``: the active one is
# whatever the conversation state holds, and its entry is refreshed whenever
# the reader forks or switches away. Keeping the live turns in one place means
# every existing handler - send, retry, edit, undo - stays unaware of forks.


def new_forks() -> dict:
    return {"active": MAIN_BRANCH, "branches": {MAIN_BRANCH: []}}


def copy_forks(forks: dict | None) -> dict:
    forks = forks or new_forks()
    return {
        "active": forks.get("active", MAIN_BRANCH),
        "branches": {
            name: copy_turns(turns) for name, turns in forks.get("branches", {}).items()
        }
        or {MAIN_BRANCH: []},
    }


def next_fork_name(forks: dict) -> str:
    """The first ``Fork N`` not already taken, so deleting one never renames another."""

    number = 1
    while f"Fork {number}" in forks["branches"]:
        number += 1
    return f"Fork {number}"


def fork_at(
    turns: list[dict] | None, found: tuple[int, str] | None
) -> tuple[list[dict], str | None]:
    """The turns a fork starts with, and the text that goes back into the box.

    ``found`` is the ``(turn index, part)`` the reader clicked, or ``None`` to
    copy the whole conversation. An assistant message keeps everything through
    its turn, so the fork is ready for a different next question. A user
    message keeps what came before it and hands its own text back, so the
    fork can start with a reworded version of that question - the same shape
    Undo gives.
    """

    turns = copy_turns(turns)
    if found is None:
        return turns, None
    position, _part = found
    if not 0 <= position < len(turns):
        return turns, None
    if turns[position]["role"] == "user":
        return turns[:position], turns[position].get("content") or ""
    return turns[: position + 1], None


def to_json(turns: list[dict] | None, *, system_prompt: str = "") -> str:
    payload = {
        "format": SAVE_FORMAT,
        "system_prompt": system_prompt or "",
        "turns": [
            {
                "role": turn["role"],
                "content": turn.get("content") or "",
                "reasoning": turn.get("reasoning") or "",
            }
            for turn in (turns or [])
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def from_json(payload: str) -> tuple[list[dict], str]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"That file is not valid JSON: {error}") from error

    if not isinstance(data, dict) or data.get("format") != SAVE_FORMAT:
        raise ValueError(f"Expected a {SAVE_FORMAT} file saved by this app.")

    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("The saved file has no list of turns.")

    turns: list[dict] = []
    for entry in raw_turns:
        if not isinstance(entry, dict):
            raise ValueError("Every turn must be an object.")
        role = entry.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported turn role: {role!r}.")
        content = entry.get("content", "")
        reasoning = entry.get("reasoning", "")
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError("Turn content and reasoning must be strings.")
        turns.append(make_turn(role, content, reasoning))

    system_prompt = data.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ValueError("The system prompt must be a string.")

    return turns, system_prompt
