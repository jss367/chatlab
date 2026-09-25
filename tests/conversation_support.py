"""Fixtures for the tests that drive the conversation page's handlers.

The sampling settings every conversation test sends, and the Gradio events a
click on the token view or the alternatives table publishes. Shared here so
that the modules exercising the conversation page need not import each other.
"""

import gradio as gr

from chatlab import app
from chatlab.token_metrics import DEFAULT_COLOR_SCALE


FIXED = {
    "system_prompt": "",
    "keep_reasoning": False,
    "assistant_prefill": "",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "skip_top_below": 0.0,
    "max_new_tokens": 8,
    "seed": 42,
    "randomize_seed": False,
    "analyze_prompt": True,
    "scale_name": DEFAULT_COLOR_SCALE,
}
SETTINGS = tuple(FIXED.values())


def metrics_of(payload):
    """The metrics half of a metrics_state payload, dropping its stamp."""

    _generation, metrics = payload
    return metrics


def strip_of(value):
    """The tokens in a strip output, whether it is a value or a gr.update."""

    return value["value"] if isinstance(value, dict) else value


def select(index):
    return gr.SelectData(None, {"index": index, "value": "x"})


def token_span(turns, token_index, turn=-1):
    """A click on one token of one reply in the conversation's token view."""

    _spans, index = app.transcript_entries(turns, DEFAULT_COLOR_SCALE)
    position = turn if turn >= 0 else len(turns) + turn
    return select(index.index((position, token_index)))


def click_token(frame, token_index, turn=-1):
    """The selection a click on a reply's token publishes."""

    turns = frame["turns"]
    _detail, _rows, selection, _target, _pick = app.select_transcript_token(
        turns, frame["metrics"], token_span(turns, token_index, turn)
    )
    return selection


def cell(row):
    """A click on one row of the alternatives table."""

    return gr.SelectData(None, {"index": [row, 1], "value": "x"})
