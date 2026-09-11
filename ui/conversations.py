"""The conversations pane: the list, forks, saving and loading, and the library on disk."""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio.utils import get_upload_folder

import charts
import library
import settings
from steering import from_controls as steering_from_controls, compact as compact_steering
from conversation import (
    CHAT_PREFIX,
    FORK_PREFIX,
    MAIN_BRANCH,
    SAMPLING_FIELDS,
    branch_choices,
    branch_sampling,
    copy_forks,
    copy_turns,
    display_messages,
    drop_branch,
    fork_at,
    from_json,
    locate,
    put_branch,
    put_branch_sampling,
    to_json,
)
from token_metrics import (
    DEFAULT_COLOR_SCALE,
)
from trace_export import write_private_text
from ui.common import (
    NO_TOKEN_SELECTED,
    failure_status,
    finalize_partial,
    send_stop_buttons,
)
from ui.panel import (
    cleared_panel,
    event_index,
    transcript_pick,
)


def conversation_list_update(forks: dict, turns: list[dict] | None):
    """Redraw the list, with the active branch's turns read from ``turns``."""

    return gr.update(choices=branch_choices(forks, turns), value=forks["active"])


def refresh_conversation_list(turns: list[dict] | None, forks: dict | None):
    """Redraw the list from state, and write the conversation on screen into the forks.

    Sending, retrying, editing, undoing and loading all write the conversation
    state without knowing about the list, and a streaming reply rewrites it on
    every frame, which is where the token count and model tag move. Rather than
    thread the list through every one of those handlers, this listens to the
    state itself: Gradio fires a State's change event only when the stored
    value's hash differs, so it runs exactly when the labels could have changed.

    The same moment is when the active branch has changed, so the forks are
    handed back with the conversation on screen written into it and stamped
    as changed now (see ``library.as_seen``). The stamp has to live in the
    forks state, not just in the file: it is what decides, when this page
    later puts the branch away or saves again, whether its copy or one another
    page has saved since is the newer - so it must record when the branch
    changed here, not when this page next happened to save it. Saving itself
    is left to ``remember_forks`` below, which the forks' change fires, so
    each frame is written once.
    """

    seen = library.as_seen(forks, turns)
    return conversation_list_update(seen, turns), seen


def remember_forks(turns: list[dict] | None, forks: dict | None) -> None:
    """Save the pane whenever the forks change. This is the one place the file is written.

    The forks change on every path that matters: the listener above hands
    them back whenever the conversation changes, and forking, starting a new
    chat, switching, deleting and clearing write them directly - including
    the changes that leave the conversation state's hash where it was, a
    switch between two empty branches, say, which the listener above never
    sees. The conversation on screen is written in once more on the way, in
    case the forks are a frame behind it. The file is small - text and a few
    counts per turn, no measurements - so rewriting it once per streaming
    frame costs nothing the frame itself does not already cost.
    """

    library.write(library.as_seen(forks, turns))


def restore_conversations():
    """Bring the saved conversations back when the page loads.

    A reload rebuilds the page from empty state, and this is what puts the
    conversations pane and the active branch back the way they were. The
    token panel is not restored: the measurements described one response as
    one model produced it, and the page has no model loaded yet.
    """

    forks = library.read()
    if forks is None:
        return (gr.skip(),) * 4
    turns = copy_turns(forks["branches"][forks["active"]])
    messages, _ = display_messages(turns)
    return messages, turns, forks, conversation_list_update(forks, turns)


def sampling_on_screen(values) -> dict:
    """The sampling the controls are showing, or the saved settings without them.

    The controls are the truth for what the reader has chosen. The settings
    file catches up a round trip later - it is written by its own listener -
    so a conversation started or forked in that window and pinned from the
    file would be pinned to the values the reader had just moved away from,
    and would then put them back on the controls.
    """

    if not values:
        return settings.sampling_values(None)
    return settings.sampling_values(
        dict(zip(settings.CONVERSATION_SAMPLING, values, strict=True))
    )


def sampling_updates(forks: dict | None):
    """Put the active conversation's sampling into the controls.

    Chained onto every path that changes which conversation is on screen, so
    switching to a fork brings back the temperature it was answered at rather
    than leaving the last one's on the sliders. A conversation that carries
    none of its own - one from a file written before conversations carried
    sampling - comes up with the saved settings, which is what it answers
    with.

    The saved settings, and not the controls: on a switch the controls hold
    the conversation being left, so reading them would make an unpinned
    conversation answer with the sampling of whatever was looked at before
    it. The settings file is read instead, and the sampling controls' own
    write to that file is ordered ahead of this on the conversation queue,
    so a slider moved and then a switch in quick succession still reads the
    value the reader chose.
    """

    forks = forks or {}
    values = settings.sampling_values(
        branch_sampling(forks, forks.get("active", MAIN_BRANCH))
    )
    return tuple(
        gr.update(value=values[name]) for name in settings.CONVERSATION_SAMPLING
    )


def remember_branch_sampling(forks: dict | None, *values):
    """Write the sampling controls into the conversation on screen.

    Wired to each control's ``input`` rather than its ``change``, so this is
    the reader moving a slider and never the app setting one: switching
    conversations writes the values of the conversation switched to onto the
    controls, and a write from that would stamp a branch nobody had touched -
    and, where two pages have the same conversation open, would claim it from
    the page that really did change it.

    Because it only ever runs for a deliberate move, the value is stored
    whatever it is, including one that happens to equal the saved setting.
    Comparing against the settings file here would be a race: the same move
    also fires ``remember_settings``, on its own queue, and if that ran first
    the file would already hold the new value and this conversation would be
    left following the file rather than pinned to what was chosen.
    ``gr.skip`` when the values are the ones already stored, because the
    forks' change is what writes the conversations file.
    """

    held = settings.sampling_values(
        dict(zip(settings.CONVERSATION_SAMPLING, values, strict=True))
    )
    forks = copy_forks(forks)
    if not put_branch_sampling(forks, forks["active"], held):
        return gr.skip()
    return forks


def remember_message(turns: list[dict] | None, event: gr.SelectData):
    """Keep the chatbot message a click landed on, for the Fork button.

    The content rides along so a click that has gone stale - the conversation
    was edited or extended underneath it - is recognized when Fork is pressed,
    instead of forking at whatever message now sits at that index.
    """

    try:
        index = event_index(event)
    except (TypeError, ValueError):
        return None
    if locate(turns, index) is None:
        return None
    return {"index": index, "content": event.value}


def remember_transcript_message(turns: list[dict] | None, event: gr.SelectData):
    """Keep the message a click in the token view landed in, for the Fork button.

    The token view draws the same messages the chatbot does, so a click in it
    is translated into the chatbot index it would have come from and kept in
    the same shape. Fork then works from either view without knowing which
    one the reader was looking at. The text rides along as it does for a
    chatbot click, so a click that has gone stale is recognized the same way.
    """

    found = transcript_pick(turns, event)
    if found is None:
        return None
    position, _token_index = found
    messages, index_map = display_messages(turns)
    for index, (turn_index, _part) in enumerate(index_map):
        if turn_index == position:
            return {"index": index, "content": messages[index]["content"]}
    return None


def selected_turn(turns: list[dict], selected: dict | None) -> tuple[int, str] | None:
    """The turn a remembered chatbot click still points at, if it still does."""

    if not selected:
        return None
    found = locate(turns, selected.get("index"))
    if found is None:
        return None
    messages, _ = display_messages(turns)
    shown = messages[int(selected["index"])]["content"]
    remembered = selected.get("content")
    if isinstance(remembered, str) and remembered.strip() != str(shown).strip():
        return None
    return found


def panel_reset(turns: list[dict] | None, scale_name: str):
    """Reset the token panel for a conversation that just changed underneath it.

    The measurements go, because they described a reply that is no longer the
    one on screen. The conversation's own token view stays, redrawn from
    ``turns``: the replies it paints carry their own measurements, so a fork
    switched to shows the colors it was generated with.
    """

    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_panel(
        turns, scale_name
    )
    return (
        strip,
        metrics,
        NO_TOKEN_SELECTED,
        [],
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


PANEL_KEPT = (gr.skip(),) * 10


def fork_refused(turns: list[dict], forks: dict, status: str):
    """Change nothing but the picker, which goes back on the active fork.

    Like every fork handler this runs after cancelling any generation (see the
    ``cancels`` on its listeners), so it still has to restore the Send button
    and close out the turn the cancelled generator left behind.
    """

    turns = copy_turns(turns)
    finalize_partial(turns)
    messages, _ = display_messages(turns)
    return (
        gr.skip(),
        messages,
        turns,
        gr.skip(),
        conversation_list_update(forks, turns),
        status,
        *send_stop_buttons(False),
        *PANEL_KEPT,
    )


def fork_conversation(
    turns: list[dict] | None,
    forks: dict | None,
    selected: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
    *sampling,
):
    """Copy the conversation into a new fork and switch to it.

    With a message selected, the copy stops there (see ``fork_at``); otherwise
    the whole transcript is copied. A whole copy keeps the token panel, since
    the response it describes is still the last one on screen; a truncated
    copy loses it, the response having gone with the cut.

    Forking cancels a running generation, as Undo, Clear and Load do, so the
    turn that generator left behind is closed out here before it is copied.
    """

    forks = copy_forks(forks)
    turns = copy_turns(turns)
    finalize_partial(turns)
    put_branch(forks, forks["active"], turns)
    found = selected_turn(turns, selected)
    forked, box_text = fork_at(turns, found)
    name = library.claim_name(forks, FORK_PREFIX)
    put_branch(forks, name, forked)
    # A fork is the same conversation taken somewhere else, so it answers the
    # way its parent does until it is changed - and both sides are pinned to
    # that, the parent included. Forking is where a comparison is set up, and
    # a side carrying no sampling of its own follows the settings file, which
    # the first slider moved on the other side would rewrite: both would then
    # answer alike, which is the one thing the fork was for.
    held = branch_sampling(forks, forks["active"])
    if held:
        # A key this version knows nothing about goes to both sides, or the
        # newer version that wrote it would find the fork answering
        # differently from the conversation it was forked from.
        inherited = {
            key: value for key, value in held.items() if key not in SAMPLING_FIELDS
        } | settings.sampling_values(held)
    else:
        inherited = sampling_on_screen(sampling)
    put_branch_sampling(forks, forks["active"], inherited)
    put_branch_sampling(forks, name, inherited)
    forks["active"] = name
    messages, _ = display_messages(forked)

    truncated = len(forked) < len(turns)
    if truncated:
        status = (
            f"Forked at message {found[0] + 1} into {name}. "
            "Send a message to take it somewhere else."
        )
    else:
        status = (
            f"Copied the conversation into {name}. Edit or undo a message, or "
            "send a new one, to take it somewhere else."
        )
    return (
        gr.skip() if box_text is None else box_text,
        messages,
        forked,
        forks,
        conversation_list_update(forks, forked),
        status,
        *send_stop_buttons(False),
        *(panel_reset(forked, scale_name) if truncated else PANEL_KEPT),
    )


def switch_fork(
    name: str | None,
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Put the conversation on screen away and bring another fork out."""

    forks = copy_forks(forks)
    if name not in forks["branches"]:
        return fork_refused(turns, forks, "That fork no longer exists.")
    if name == forks["active"]:
        return fork_refused(turns, forks, f"Already on {name}.")

    turns = copy_turns(turns)
    finalize_partial(turns)
    put_branch(forks, forks["active"], turns)
    forks["active"] = name
    target = copy_turns(forks["branches"][name])
    messages, _ = display_messages(target)
    count = len(target)
    return (
        gr.skip(),
        messages,
        target,
        forks,
        conversation_list_update(forks, target),
        f"Switched to {name} ({count} message{'s' if count != 1 else ''}).",
        *send_stop_buttons(False),
        *panel_reset(target, scale_name),
    )


def delete_fork(
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
):
    """Drop the active fork and go back to the main conversation."""

    forks = copy_forks(forks)
    name = forks["active"]
    if name == MAIN_BRANCH:
        return fork_refused(
            turns,
            forks,
            "The main conversation cannot be deleted. Clear all empties every conversation.",
        )

    drop_branch(forks, name)
    forks["active"] = MAIN_BRANCH
    target = copy_turns(forks["branches"].setdefault(MAIN_BRANCH, []))
    messages, _ = display_messages(target)
    return (
        gr.skip(),
        messages,
        target,
        forks,
        conversation_list_update(forks, target),
        f"Deleted {name}. Back on {MAIN_BRANCH}.",
        *send_stop_buttons(False),
        *panel_reset(target, scale_name),
    )


def new_conversation(
    turns: list[dict] | None,
    forks: dict | None,
    scale_name: str = DEFAULT_COLOR_SCALE,
    *sampling,
):
    """Put the conversation on screen away and start an empty one.

    Unlike Fork, nothing is copied: the new chat begins with no turns, so the
    next message is measured against the system prompt alone. The message box
    is left as it is, since whatever is typed there is likely meant for the
    new chat. Starting one cancels a running generation, as every branch
    change does, so the turn that generator left behind is closed out before
    it is put away.
    """

    forks = copy_forks(forks)
    turns = copy_turns(turns)
    finalize_partial(turns)
    put_branch(forks, forks["active"], turns)
    # The names in the file count too, so a chat another page started since
    # this one loaded is not given a twin the merge would take for it.
    name = library.claim_name(forks, CHAT_PREFIX)
    put_branch(forks, name, [])
    # Started from the sampling on screen, and pinned to it: a conversation
    # that went on following the settings file would be moved by a slider
    # touched on any other conversation.
    put_branch_sampling(forks, name, sampling_on_screen(sampling))
    forks["active"] = name
    return (
        gr.skip(),
        [],
        [],
        forks,
        conversation_list_update(forks, []),
        f"Started {name}. Send a message to begin it.",
        *send_stop_buttons(False),
        *panel_reset([], scale_name),
    )


def save_conversation(turns, system_prompt, steering=None, steering_enabled=None, steering_strength=None, steering_layer=None):
    if not turns:
        return gr.update(value=None, visible=False), "There is nothing to save yet."

    # Gradio only serves files it created or was told to allow, so the saved
    # conversation has to live inside its upload folder.
    directory = Path(get_upload_folder()) / "chatlab-conversations"
    directory.mkdir(parents=True, exist_ok=True)
    # The timestamp only resolves to the second, and every session shares this
    # upload folder, so a random suffix keeps two saves from landing on the same
    # path and silently overwriting each other's download.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"conversation-{stamp}-{uuid4().hex[:8]}.json"
    # A transcript is the reader's own writing, and the upload folder is shared:
    # on Linux it is /tmp/gradio, which every account on the machine can read.
    # write_private_text() makes the file owner-only before it holds a word of
    # the conversation, so there is no moment for another account to open it.
    # write_trace_export() writes its export the same way.
    steering = steering_from_controls(steering, steering_enabled, steering_strength, steering_layer)
    write_private_text(path, to_json(turns, system_prompt=system_prompt, steering=steering))
    return (
        gr.update(value=str(path), visible=True),
        f"Saved {len(turns)} message{'s' if len(turns) != 1 else ''}.",
    )


def load_conversation(file_path, turns, scale_name: str = DEFAULT_COLOR_SCALE, *, include_steering=False):
    """Replace the conversation with a saved one.

    A failed load keeps the conversation already on screen, so a bad file
    cannot wipe it, and leaves the token panel describing it alone. Loading
    cancels any generation still running, so the buttons are restored here for
    the same reason Clear restores them: a cancelled generator never reaches
    its final yield. For the same reason the kept conversation has to be
    finalized like Stop does - the cancelled generator left its last turn with
    a pending reasoning block, which would spin for the rest of the session,
    or empty if the cancel landed before the first token.
    """

    def keep_current(status):
        """Return the conversation the cancelled generator left behind."""

        kept = copy_turns(turns)
        finalize_partial(kept)
        messages, _ = display_messages(kept)
        return (
            messages,
            kept,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            status,
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
            *(gr.skip(),) * 6,
            *((gr.skip(),) if include_steering else ()),
        )

    if not file_path:
        return keep_current("No file chosen.")
    try:
        payload = Path(file_path).read_text(encoding="utf-8")
        loaded, system_prompt = from_json(payload)
        steering = compact_steering(json.loads(payload).get("steering"))
    except (OSError, ValueError) as error:
        return keep_current(failure_status("Could not load that file", str(error)))

    # A successful load replaces the conversation wholesale, so whatever the
    # cancelled generator left behind goes with it and needs no finalizing.
    turns = loaded
    messages, _ = display_messages(turns)
    strip, metrics, prompt_strip, prompt_metrics, prompt_note = cleared_panel(
        turns, scale_name
    )
    # The selected token described a response from the conversation being
    # replaced, so it goes with it, exactly as Clear and Undo reset it. The
    # charts and the export measured that response too, and a loaded
    # conversation has no measurements of its own to put in their place: a
    # saved file holds the text and the counts, not the distributions, so its
    # replies come back as plain text in the token view.
    return (
        messages,
        turns,
        system_prompt,
        strip,
        metrics,
        f"Loaded {len(turns)} message{'s' if len(turns) != 1 else ''}.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
        prompt_strip,
        prompt_metrics,
        prompt_note,
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
        *((steering,) if include_steering else ()),
    )
