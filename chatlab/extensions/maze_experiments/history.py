"""How a maze run's history carries a response that opened a reasoning block.

Kept to the standard library and to plain dicts, as inserts.py is, so a
collector building these runs elsewhere can import this module, or copy it,
and write the history exactly as ChatLab writes and checks it.

A run names the form its history takes in ``config["reasoning_history"]``.
Under ``"reasoning_content"``, which every run ChatLab starts writes, a
response that opened a reasoning block is stored as two fields: its reasoning
in ``reasoning_content``, the field templates such as Qwen3.8's build an
earlier turn's reasoning block from, and its answer in ``content``. Under
``"content"``, the form of every run written before the field, the whole
response stays in ``content``, with the opening tag a template wrote ahead of
it restored. A run that names no form was written in ``"content"``.
"""
from __future__ import annotations

FIELD = "reasoning_history"
CONTENT = "content"
REASONING_CONTENT = "reasoning_content"
FORMS = (CONTENT, REASONING_CONTENT)
OPEN, CLOSE = "<think>", "</think>"


def response_text(text, reasoning_prefilled=False):
    """A response with the opening tag restored that its template wrote ahead of it, if one did."""
    return (OPEN if reasoning_prefilled else "") + text


def split_reasoning(text):
    """A response's ``(reasoning, answer)``, or None where it opened no reasoning block.

    ``text`` is the whole response, any opening tag its template wrote
    restored. The reasoning is what came before the first ``</think>``,
    without the opening tag and trimmed of the newlines around it; the answer
    is what came after it, left-trimmed. A response cut off before it closed
    its reasoning has no answer: all of its text is reasoning.
    """
    text = text.lstrip()
    if not text.startswith(OPEN):
        return None
    reasoning, closed, answer = text[len(OPEN):].partition(CLOSE)
    return reasoning.strip("\n"), answer.lstrip() if closed else ""


def assistant_message(text, reasoning_prefilled=False, form=REASONING_CONTENT):
    """A response as a run's history carries it, in the history ``form`` the run names.

    ``text`` is the response as generated and ``reasoning_prefilled`` whether
    its template opened a reasoning block ahead of it.
    """
    if form not in FORMS:
        raise ValueError(f"A run's reasoning history is one of {', '.join(FORMS)}.")
    whole = response_text(text, reasoning_prefilled)
    parts = split_reasoning(whole) if form == REASONING_CONTENT else None
    if parts is None:
        return {"role": "assistant", "content": whole}
    return {"role": "assistant", "reasoning_content": parts[0], "content": parts[1]}


def history_form(config):
    """The history form a run's config names, or ``"content"`` for a run that names none."""
    form = config.get(FIELD, CONTENT)
    if form not in FORMS:
        raise ValueError(f"A run's {FIELD} is one of {', '.join(FORMS)}.")
    return form


def check_messages(messages, form):
    """Refuse a history whose messages are not in the ``form`` its run names.

    Read from the messages alone: under ``"content"`` no message carries a
    ``reasoning_content``, and under ``"reasoning_content"`` every response
    that opened a reasoning block carries one, beside the answer after it.
    Which response each message holds is the run's to check.
    """
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message in a run's history is an object.")
        if REASONING_CONTENT not in message:
            content = message.get("content")
            if form == REASONING_CONTENT and message.get("role") == "assistant" and isinstance(content, str) \
                    and split_reasoning(content) is not None:
                raise ValueError(f"The run names a {REASONING_CONTENT} history, and a response in it keeps its "
                                 "reasoning in content.")
        elif form != REASONING_CONTENT:
            raise ValueError(f"A response in the run's history carries {REASONING_CONTENT}, which a {form} history "
                             f"does not. A run writing it names {FIELD}: {REASONING_CONTENT}.")
        elif message.get("role") != "assistant" or not isinstance(message[REASONING_CONTENT], str) \
                or not isinstance(message.get("content"), str):
            raise ValueError(f"Only a response carries {REASONING_CONTENT}, as text beside its answer.")
