"""Editing the reader's messages without leaving the token view."""

import gradio as gr

from conversation import display_messages, turn_entries
from token_metrics import DEFAULT_COLOR_SCALE
from ui.generation import CHAT_OUTPUT_NAMES, busy_state, edit_message, occupied
from ui.panel import current_metrics_generation, event_index, transcript_entries, transcript_pick


def close_token_editor():
    return gr.update(visible=False), "", None


def open_token_editor(turns, metrics_state, event: gr.SelectData):
    # A queued click must not open a different message after a conversation
    # change. The stamp also keeps clicks during generation from arming edits.
    if occupied() or metrics_state[0] != current_metrics_generation():
        return (gr.skip(),) * 3
    found = transcript_pick(turns, event)
    if found is None or turns[found[0]]["role"] != "user":
        return (gr.skip(),) * 3
    spans, span_map = transcript_entries(turns, DEFAULT_COLOR_SCALE)
    span_index = event_index(event)
    clicked = event.value
    if isinstance(clicked, (list, tuple)):
        clicked = clicked[0] if clicked else None
    if clicked != spans[span_index][0]:
        return (gr.skip(),) * 3
    position, _ = found
    _, index_map = display_messages(turns)
    try:
        index = index_map.index((position, "content"))
    except ValueError:
        # Imported user turns can contain reasoning alone. There is no
        # message text to edit, so leave any existing draft untouched.
        return (gr.skip(),) * 3
    # Content, when present, is the last span of its turn. Check its position
    # rather than its text: headings or reasoning may have identical text.
    content_span = max(i for i, (turn_index, _) in enumerate(span_map) if turn_index == position)
    if span_index != content_span:
        return (gr.skip(),) * 3
    target = {
        "index": index,
        "generation": current_metrics_generation(),
        "turns": turn_entries(turns),
    }
    return gr.update(visible=True), turns[position]["content"], target


def save_token_edit(target, text, prompt_text, turns, *settings):
    """Use the regular edit path, preserving the draft if saving is refused."""
    held = occupied()
    if held:
        yield (*busy_state(held), gr.skip(), gr.skip())
        return
    if (
        not target
        or target["generation"] != current_metrics_generation()
        or target["turns"] != turn_entries(turns)
    ):
        frame = [gr.skip() for _ in CHAT_OUTPUT_NAMES]
        frame[CHAT_OUTPUT_NAMES.index("status")] = (
            "The conversation changed. Click your message again before editing it."
        )
        yield (*frame, gr.skip(), gr.skip())
        return
    messages, _ = display_messages(turns)
    event = gr.EditData(None, {
        "index": target["index"],
        "previous_value": messages[target["index"]]["content"],
        "value": text,
    })
    for frame in edit_message(event, prompt_text, turns, *settings):
        updated = frame[CHAT_OUTPUT_NAMES.index("turns")]
        accepted = isinstance(updated, list) and updated != turns
        yield (
            *frame,
            gr.update(visible=False) if accepted else gr.skip(),
            None if accepted else gr.skip(),
        )
