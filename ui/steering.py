"""Conversation-local steering controls and portable conversation imports."""

from __future__ import annotations

import gradio as gr

from conversation import MAIN_BRANCH, branch_sampling, copy_forks, put_branch_sampling
from steering import from_controls, normalize, read_vector
from ui.conversations import load_conversation


EMPTY_STATUS = "Import a JSON vector to steer this conversation. Layers count from 0."


def description(value):
    if value is None:
        return EMPTY_STATUS
    state = "Enabled" if value["enabled"] and value["strength"] else "Off"
    # Model names are file-supplied text, so keep them out of Markdown markup.
    return (
        f"{state} · {value['model_id']} · layer {value['layer']} · "
        f"{len(value['vector']):,} entries · strength {value['strength']:g}. "
        "Applies to the prompt and response. Compatibility is checked before generation."
    )


def controls(value):
    return (
        value,
        gr.update(value=value["enabled"] if value else False, interactive=value is not None),
        gr.update(value=value["strength"] if value else 1, interactive=value is not None),
        gr.update(value=value["layer"] if value else 0, interactive=value is not None),
        description(value),
    )


def steering_updates(forks):
    held = branch_sampling(forks, (forks or {}).get("active", MAIN_BRANCH))
    return controls(normalize(held.get("steering")))


def store(forks, value):
    forks = copy_forks(forks)
    held = branch_sampling(forks, forks["active"])
    held["steering"] = value
    put_branch_sampling(forks, forks["active"], held)
    return forks


def import_vector(path, forks):
    try:
        value = read_vector(path)
    except (OSError, ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error
    return store(forks, value), *controls(value)


def remove_vector(forks):
    return store(forks, None), *controls(None)


def remember_steering(forks, value, enabled, strength, layer):
    try:
        value = from_controls(value, enabled, strength, layer)
    except (ValueError, TypeError, OverflowError) as error:
        raise gr.Error(str(error)) from error
    if value is None:
        return gr.skip(), None, EMPTY_STATUS
    return store(forks, value), value, description(value)


def load_with_steering(path, turns, scale_name, forks):
    *result, value = load_conversation(path, turns, scale_name, include_steering=True)
    # The core loader preserves the system-prompt control on failure.
    if not isinstance(result[2], str):
        return (*result, *((gr.skip(),) * 6))
    return (*result, store(forks, value), *controls(value))
