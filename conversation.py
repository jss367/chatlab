"""Conversation turns, reasoning blocks, and save/load helpers.

A turn is a plain dictionary so it can live in a ``gr.State`` and be written
straight to JSON:

    {"role": "user" | "assistant", "content": str, "reasoning": str}

``reasoning`` holds the text a Think model wrapped in ``<think>`` tags. It is
kept beside the answer rather than inside it so the interface can collapse it
and so the next request can deliberately include or drop it.

A generated assistant turn also records where it came from, so the list of
conversations can say which model answered and how big the exchange was:

    {"model": str, "prompt_tokens": int, "generated_tokens": int, "thinking_mode": str}

``prompt_tokens`` is every token the model was given for that reply - the
system prompt, the transcript so far and the template around them - and
``generated_tokens`` is every token it produced, reasoning included. A turn
that was typed, rewritten by hand, loaded from an older file, or never
finished measuring may carry none of these, and the list says so rather than
guessing.

A reply also carries the measurements behind every token it is made of, which
is what lets the conversation itself be painted by rank or surprise and
branched at any token in it rather than only in the newest reply:

    {"tokens": [metric, ...], "load_id": str, "metrics_generation": int}

``tokens`` is what ``token_metrics.build_metric`` produced for that reply,
``load_id`` names the model load that produced it, and
``metrics_generation`` is the stamp the token panel was drawn with while the
reply was the live one (see ``ui.panel``). These three are memory only:
:func:`turn_entries` leaves them out, so the saved file stays the size it
was. It is rewritten on every streaming frame, and a few hundred numbers per
token would make that a multi-megabyte write per token.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from datetime import datetime, timezone

from steering import compact as compact_steering, export_assets, import_assets

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
REASONING_TITLE = "Reasoning"
SAVE_FORMAT = "chatlab-conversation-1"
MAIN_BRANCH = "Main"
FORK_PREFIX = "Fork"
CHAT_PREFIX = "Chat"

# How much of a conversation's first message the list shows as its title.
TITLE_LIMIT = 40

# The per-turn provenance fields, and the type each must have in a saved file.
TURN_ORIGIN_FIELDS = {
    "model": str,
    "prompt_tokens": int,
    "generated_tokens": int,
    "thinking_mode": str,
}

# What a reply carries of its own measurements. These never reach the file;
# see the module docstring for why.
TURN_MEASUREMENT_FIELDS = (
    "tokens", "load_id", "metrics_generation", "ends_on_stop_token",
)

# The sampling a conversation can carry of its own, and the type each must
# have in a saved file. The settings module owns what the values may be; this
# is only what a file is allowed to hold. A key this version knows nothing
# about is carried through untouched, so a file written by a newer version
# keeps its extra keys through a read and a save by an older one.
SAMPLING_FIELDS = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_new_tokens": int,
}

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
    """Snapshot turns so a streaming update cannot mutate stored state.

    Deep, because a turn carries nested values - a steering entry above all -
    that a shallow copy would leave two copies sharing.

    The measurements are the exception. The list itself is copied, so a turn
    can gain or lose tokens without disturbing a copy of it, but the metrics
    inside are shared rather than duplicated. Each is written once by
    ``token_metrics.build_metric`` and never edited afterwards, and each holds
    a dozen numbers plus its eight alternatives. Copying them here would mean
    copying every measurement in the conversation on every streaming frame,
    for a cost that grows with the square of the reply's length: about 1.6
    seconds of copying across a 500-token reply, and twenty-five across a
    2,000-token one.
    """

    copied: list[dict] = []
    for turn in turns or []:
        tokens = turn.get("tokens")
        if tokens is None:
            copied.append(copy.deepcopy(turn))
            continue
        entry = copy.deepcopy(
            {key: value for key, value in turn.items() if key != "tokens"}
        )
        entry["tokens"] = list(tokens)
        copied.append(entry)
    return copied


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
            if not content and turn.get("token_step_paused"):
                content = "Paused before visible text."
                if turn_tokens(turn):
                    content += " Press Next token to continue."
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
            if turn["role"] == "assistant" and (
                reasoning or turn.get("token_step_paused")
            ):
                # A reasoning-only reply or an invisible token step still owns
                # an assistant slot in the visible conversation. Keep it empty
                # so a subsequent Send preserves alternating roles, without
                # replaying hidden tokens or the display-only pause notice.
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
#     {"active": name, "branches": {name: [turns...], ...},
#      "sampling": {name: {...}, ...}, "updated": {name: stamp, ...}}
#
# Only the *inactive* branches are current in ``branches``: the active one is
# whatever the conversation state holds, and its entry is refreshed whenever
# the reader forks or switches away. Keeping the live turns in one place means
# every existing handler - send, retry, edit, undo - stays unaware of forks.


def new_forks() -> dict:
    """The pane with nothing in it: the empty main conversation, never yet saved.

    ``updated`` holds, per branch name, when this page last changed that
    branch, as :func:`branch_stamp` writes it. A name in it with no branch
    under ``branches`` is one this page deleted, and when. Both are what lets
    two pages writing the same file keep each other's work - see
    ``library.merge``.

    ``sampling`` holds, per branch name, the sampling that branch answers
    with - see :func:`put_branch_sampling`. A branch with no entry answers
    with the saved settings, which is what every branch did before
    conversations carried their own. ``sampling_updated`` stamps those the
    way ``updated`` stamps the turns, and separately: a page that changes
    only the temperature must not thereby claim a transcript it may be a
    reply behind on.
    """

    return {
        "active": MAIN_BRANCH,
        "branches": {MAIN_BRANCH: []},
        "sampling": {},
        "sampling_updated": {},
        "updated": {},
    }


def copy_forks(forks: dict | None) -> dict:
    forks = forks or new_forks()
    return {
        "active": forks.get("active", MAIN_BRANCH),
        "branches": {
            name: copy_turns(turns) for name, turns in forks.get("branches", {}).items()
        }
        or {MAIN_BRANCH: []},
        "sampling": {
            name: copy.deepcopy(values)
            for name, values in (forks.get("sampling") or {}).items()
        },
        "sampling_updated": dict(forks.get("sampling_updated") or {}),
        "updated": dict(forks.get("updated") or {}),
    }


def branch_stamp() -> str:
    """Now, as the ``updated`` entries spell it.

    UTC, always with microseconds, so that two stamps compare as strings the
    way they compare as times.
    """

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def put_branch(forks: dict, name: str, turns: list[dict] | None) -> None:
    """Store ``turns`` as branch ``name``, and stamp it if that changes what is saved.

    A branch put back exactly as it was - the conversation on screen written
    into the pane on the way to another branch, say - keeps the stamp it had,
    so a copy of it that another page has changed since still wins.
    """

    turns = copy_turns(turns)
    before = forks["branches"].get(name)
    if before is None or turn_entries(before) != turn_entries(turns):
        forks["updated"][name] = branch_stamp()
    forks["branches"][name] = turns


def drop_branch(forks: dict, name: str) -> None:
    """Remove branch ``name`` and record when, so no other page's copy brings it back."""

    del forks["branches"][name]
    forks.setdefault("sampling", {}).pop(name, None)
    forks.setdefault("sampling_updated", {}).pop(name, None)
    forks["updated"][name] = branch_stamp()


def branch_sampling(forks: dict | None, name: str) -> dict:
    """What branch ``name`` carries of its own sampling; empty where it carries none."""

    held = (forks or {}).get("sampling") or {}
    return copy.deepcopy(held.get(name) or {})


def put_branch_sampling(forks: dict, name: str, values: dict) -> bool:
    """Store ``values`` as branch ``name``'s sampling; say whether that changed anything.

    Stamped like a change to the turns, and for the same reason: the stamp is
    what decides, when two pages have both touched a branch, which copy wins,
    and a conversation moved to temperature 0 on one page must not be pulled
    back by another page that merely still holds it.

    Under its own stamp, though, not the turns'. Two pages can have one
    conversation open, and a page that moves a slider may be a reply behind
    the other: sharing one stamp would have that page win the whole branch
    and take the newer reply off the file with it.
    """

    sampling = forks.setdefault("sampling", {})
    # Whatever the branch carries that this version knows nothing about
    # stays, and so does anything of that kind in ``values`` - which is how
    # a fork inherits a newer version's own keys from the conversation it
    # came from. See ``library.sampling_entry``.
    held = sampling.get(name) or {}
    kept = {
        key: value for key, value in held.items() if key not in SAMPLING_FIELDS
    } | dict(values)
    if held == kept:
        return False
    sampling[name] = copy.deepcopy(kept)
    forks.setdefault("sampling_updated", {})[name] = branch_stamp()
    return True


def next_branch_name(forks: dict, prefix: str, taken: Iterable[str] = ()) -> str:
    """The first ``<prefix> N`` not already taken, so deleting one never renames another.

    ``taken`` is further names to step over - those the saved file has spoken
    for, as ``library.taken_names`` lists them - so a branch another page
    started since this one loaded is not given a twin.
    """

    used = set(forks["branches"]) | set(taken)
    number = 1
    while f"{prefix} {number}" in used:
        number += 1
    return f"{prefix} {number}"


def next_fork_name(forks: dict, taken: Iterable[str] = ()) -> str:
    return next_branch_name(forks, FORK_PREFIX, taken)


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


def turn_tokens(turn: dict | None) -> list[dict]:
    """The per-token measurements a reply carries, or nothing."""

    tokens = (turn or {}).get("tokens")
    return tokens if isinstance(tokens, list) else []


def forget_measurements(turns: list[dict] | None, position: int) -> list[dict]:
    """The turns after the reply at ``position`` was rewritten by hand.

    The counts on a generated reply describe the text the model produced, and
    an edit replaces that text. The edited reply loses both of its counts.
    Every later reply keeps ``generated_tokens`` - its own text is untouched -
    but loses ``prompt_tokens``, which measured a transcript that no longer
    exists. ``model`` stays throughout: rewording an answer does not change
    who gave it. The list then falls back to the last reply measured before
    the edit, the newest size that is still true.

    The per-token measurements go from the edited reply and from every reply
    after it. Each of those was produced from a transcript the edit has
    replaced, so their ranks and probabilities no longer describe anything on
    screen, and replaying their tokens onto the edited conversation would
    force a reply the model never gave that prompt. ``generated_tokens``
    survives where the text does because a count of tokens is still a true
    count; a distribution over a prompt that is gone is not.
    """

    turns = copy_turns(turns)
    for index in range(max(position, 0), len(turns)):
        turn = turns[index]
        if turn["role"] != "assistant":
            continue
        turn.pop("prompt_tokens", None)
        for field in TURN_MEASUREMENT_FIELDS:
            turn.pop(field, None)
        if index == position:
            turn.pop("generated_tokens", None)
            turn.pop("token_step_paused", None)
    return turns


# ------------------------------------------------------- conversation list
#
# The pane beside the chat lists every branch. Each entry is two lines: the
# branch's name and the start of its first message, then the model that
# answered and how many tokens the conversation had come to. Everything here
# is derived from the turns alone, so the list can be redrawn from state
# whenever the state moves, streaming frames included.


def short_model_name(model_id: str) -> str:
    """``allenai/Olmo-3-7B-Think`` -> ``Olmo-3-7B-Think``: the org is noise in a tag."""

    return model_id.rstrip("/").rsplit("/", 1)[-1] or model_id


def _model_names(model_ids: list[str]) -> list[str]:
    """Short names for distinct IDs, kept whole where shortening would merge two.

    ``org-a/model`` and ``org-b/model`` are different models, and a tag that
    says ``model`` for both would claim only one answered.
    """

    short = [short_model_name(model_id) for model_id in model_ids]
    return [
        name if short.count(name) == 1 else model_id
        for model_id, name in zip(model_ids, short)
    ]


def branch_title(turns: list[dict] | None, limit: int = TITLE_LIMIT) -> str:
    """The first user message, flattened to one line and cut to ``limit``."""

    for turn in turns or []:
        if turn["role"] == "user":
            text = " ".join((turn.get("content") or "").split())
            if len(text) > limit:
                return text[: limit - 1].rstrip() + "…"
            return text
    return ""


def describe_branch(turns: list[dict] | None) -> dict:
    """What the list says about one branch.

    ``models`` names every model that answered, most recent first and each
    once: by short name, or by full ID when two organizations share a name.
    ``tokens`` is the size of the conversation as the
    model last saw it - the prompt behind the latest measured reply plus that
    reply - or ``None`` when no reply carries a measurement. ``replies`` counts
    assistant turns, measured or not, so an unmeasured transcript can be told
    from an empty one.
    """

    turns = turns or []
    models: list[str] = []
    tokens = None
    replies = 0
    for turn in turns:
        if turn["role"] != "assistant":
            continue
        replies += 1
        model = turn.get("model")
        if isinstance(model, str) and model:
            # Deduplicate on the full ID; shortening comes last, once the set
            # is known, so it can tell when two IDs would share a name.
            if model in models:
                models.remove(model)
            models.insert(0, model)
        prompt_tokens = turn.get("prompt_tokens")
        generated = turn.get("generated_tokens")
        if isinstance(prompt_tokens, int) and isinstance(generated, int):
            # Later replies win: their prompt already contains everything
            # before them, so the last one measured is the whole conversation.
            tokens = prompt_tokens + generated
    return {
        "title": branch_title(turns),
        "models": _model_names(models),
        "tokens": tokens,
        "replies": replies,
    }


def branch_label(name: str, turns: list[dict] | None) -> str:
    """The two-line entry the list shows for a branch."""

    summary = describe_branch(turns)
    head = f"{name} · {summary['title']}" if summary["title"] else name
    if not summary["replies"]:
        detail = "No replies yet" if summary["title"] else "No messages yet"
    else:
        parts = [" + ".join(summary["models"]) or "Model not recorded"]
        if summary["tokens"] is not None:
            parts.append(f"{summary['tokens']:,} tokens")
        detail = " · ".join(parts)
    return f"{head}\n{detail}"


def branch_choices(forks: dict | None, turns: list[dict] | None) -> list[tuple[str, str]]:
    """``(label, name)`` for every branch, the active one read from ``turns``.

    The active branch's entry in ``forks`` is stale by design (see the forks
    section above), so its turns come from the conversation state instead.
    """

    forks = forks or new_forks()
    active = forks.get("active", MAIN_BRANCH)
    choices = []
    for name, stored in forks.get("branches", {}).items():
        branch = turns if name == active else stored
        choices.append((branch_label(name, branch), name))
    return choices


# --------------------------------------------------------------- save / load


def turn_entries(turns: list[dict] | None) -> list[dict]:
    """The turns as a file spells them: text, reasoning and the counts behind a reply."""

    entries = []
    for turn in turns or []:
        entry = {
            "role": turn["role"],
            "content": turn.get("content") or "",
            "reasoning": turn.get("reasoning") or "",
        }
        # This is transcript structure, not a measurement: an invisible step
        # still owns an assistant slot after the token metrics are discarded.
        if turn["role"] == "assistant" and turn.get("token_step_paused") is True:
            entry["token_step_paused"] = True
        for key, kind in TURN_ORIGIN_FIELDS.items():
            value = turn.get(key)
            # bool is an int to isinstance(), and a True here would be a bug.
            if isinstance(value, kind) and not isinstance(value, bool):
                entry[key] = value
        if turn.get("steering") is not None:
            entry["steering"] = compact_steering(turn["steering"])
        entries.append(entry)
    return entries


def turns_from_entries(raw_turns) -> list[dict]:
    """Turns read back from :func:`turn_entries`, or ``ValueError`` for anything else."""

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
        turn = make_turn(role, content, reasoning)
        if "token_step_paused" in entry:
            if not isinstance(entry["token_step_paused"], bool):
                raise ValueError("Turn token_step_paused must be a bool.")
            if role == "assistant" and entry["token_step_paused"]:
                turn["token_step_paused"] = True
        for key, kind in TURN_ORIGIN_FIELDS.items():
            if key not in entry:
                continue
            value = entry[key]
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f"Turn {key} must be a {kind.__name__}.")
            if kind is int and value < 0:
                raise ValueError(f"Turn {key} cannot be negative.")
            turn[key] = value
        if entry.get("steering") is not None:
            turn["steering"] = compact_steering(entry["steering"])
        turns.append(turn)
    return turns


def to_json(turns: list[dict] | None, *, system_prompt: str = "", steering: dict | None = None) -> str:
    payload = {
        "format": SAVE_FORMAT,
        "system_prompt": system_prompt or "",
        "turns": turn_entries(turns),
    }
    if steering is not None:
        payload["steering"] = compact_steering(steering)
    assets = export_assets([payload.get("steering"), *(turn.get("steering") for turn in payload["turns"])])
    if assets:
        payload["steering_vectors"] = assets
    return json.dumps(payload, indent=2, ensure_ascii=False)


def from_json(payload: str) -> tuple[list[dict], str]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"That file is not valid JSON: {error}") from error

    if not isinstance(data, dict) or data.get("format") != SAVE_FORMAT:
        raise ValueError(f"Expected a {SAVE_FORMAT} file saved by this app.")

    raw_turns = data.get("turns")
    values = [data.get("steering")]
    if isinstance(raw_turns, list):
        values.extend(turn.get("steering") for turn in raw_turns if isinstance(turn, dict))
    import_assets(values, data.get("steering_vectors"))
    turns = turns_from_entries(raw_turns)

    system_prompt = data.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ValueError("The system prompt must be a string.")

    return turns, system_prompt
