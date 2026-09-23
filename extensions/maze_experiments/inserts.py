"""Text placed in a maze run's context by someone other than the model, partway through.

Kept to the standard library and to plain dicts, so a collector building
these runs elsewhere can import this module, or copy it, and write the
history exactly as ChatLab replays it.
"""
from __future__ import annotations

import json
import re

FORMAT = "chatlab-maze-run-3"
CHANNELS = {"tool_note": "Simulator note", "teammate": "Teammate message", "user": "User message"}
DIRECTIONS = ("north", "east", "south", "west")
KEYS = {"before_turn", "channel", "text", "sender", "position", "advised_direction"}
# Spellings that would let an inserted message open a turn, a call or a
# reasoning block of its own once the template renders it: tool calls,
# reasoning tags, and the turn markers the supported families write -
# every <|...|> special (Qwen, Llama 3, OLMo, Phi), DeepSeek's full-width
# form, Gemma's turns, and the Llama 2 and Mistral instruction markers.
# Written without a tokenizer, so a run can be checked anywhere; generation
# also asks the loaded tokenizer, which knows its own specials exactly.
MARKS = re.compile(r"</?tool_call|</?think>|<\|im_|<\|[^|\s]*\|>|<｜[^｜\s]*｜>|<(?:start|end)_of_turn>"
                   r"|</?s>|<(?:bos|eos)>|\[/?INST\]")


def check_insert(insert):
    """Refuse an insertion whose channel, text, sender or advice breaks the record's rules.

    Only the fields an insertion states about itself are read here. Where it
    lands, and where the character stood then, are the run's to check.
    """
    if not isinstance(insert, dict) or set(insert) - KEYS:
        raise ValueError("An inserted message records only " + ", ".join(sorted(KEYS)) + ".")
    channel, text, sender = insert.get("channel"), insert.get("text"), insert.get("sender")
    if channel not in CHANNELS:
        raise ValueError("An inserted message goes in as a tool_note, a teammate message, or a user message.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("An inserted message needs text.")
    if MARKS.search(text):
        raise ValueError("An inserted message cannot supply tool syntax, conversation boundary tokens or "
                         "reasoning delimiters.")
    if channel == "teammate":
        if not isinstance(sender, str) or not sender.strip():
            raise ValueError("A teammate message names who sent it.")
    elif sender is not None:
        raise ValueError("Only a teammate message names a sender.")
    if insert.get("advised_direction") not in (None, *DIRECTIONS):
        raise ValueError("The advised direction is north, east, south, west, or none.")
    return insert


def appended(insert):
    """The key and value an insertion adds to the simulator's reply, or None for a user message."""
    if insert["channel"] == "tool_note":
        return "note", insert["text"]
    if insert["channel"] == "teammate":
        return "messages", [{"from": insert["sender"], "text": insert["text"]}]
    if insert["channel"] == "user":
        return None
    raise ValueError(f"There is no {insert['channel']!r} channel to insert a message through.")


def render_insert(messages, insert):
    """The history with one insertion placed where the next response will read it.

    A simulator note or a teammate message becomes the last key of the latest
    simulator reply, in the reply's own compact JSON, so the note reads as
    part of the state and a teammate's message reads as it does in a team run.
    A user message is a turn of its own after that reply. Every channel needs
    that reply: before it, the history ends on the task's own user turn, and a
    second user turn there is one that templates enforcing alternation refuse.
    ``messages`` is left as it was.
    """
    added = appended(insert)
    if not messages or messages[-1]["role"] != "tool":
        raise ValueError("There is no simulator reply yet to carry this message. Insert it after the first "
                         "move, or supply a starting move.")
    if added is None:
        return [*messages, {"role": "user", "content": insert["text"]}]
    state = json.loads(messages[-1]["content"])
    if added[0] in state:
        raise ValueError(f"The simulator's reply already carries {added[0]}.")
    state[added[0]] = added[1]
    return [*messages[:-1], {**messages[-1], "content": json.dumps(state, separators=(",", ":"))}]


def inserted_fragment(insert):
    """The characters an insertion adds to the history, as a template writes them into the prompt."""
    added = appended(insert)
    if added is None:
        return insert["text"]
    return json.dumps(added[0]) + ":" + json.dumps(added[1], separators=(",", ":"))
