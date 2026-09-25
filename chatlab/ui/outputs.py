"""What the conversation handlers publish, by name, and where names become positions.

Gradio hands a listener's return value to its outputs by position: the first
value to the first component, and so on. For a handler with three outputs that
is fine. The conversation handlers publish twenty-odd, most of them skipped on
any one frame, and several families of them publish overlapping sets in
different orders - a reply, an undo, a fork, a clear, a load. Counted positions
there were a standing hazard: an output added to a list in the layout and not
to the tuple in the handler, or added one place along, and every value after it
landed in its neighbour's component without a word from anything.

So those handlers return a :class:`Frame` instead: a mapping from output name
to value, holding only what that frame changes. Each family declares below the
names it may publish, in the order its listener registers them. The layout
builds one table from those names to the components, in one place, and
``ConversationEvents`` - which wraps every one of these listeners already -
turns a frame into the tuple Gradio wants through that table. Nothing else
counts positions. A name a frame was not declared with is refused when the
frame is built, so a misspelt output fails in the handler that wrote it rather
than disappearing on the way to the page.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import gradio as gr


def skipped(value) -> bool:
    """Whether ``value`` is Gradio's "leave this output alone"."""

    return isinstance(value, dict) and value == gr.skip()


class Frame(dict):
    """The outputs one handler call changes, keyed by output name.

    ``names`` is every output the frame's family publishes. A name among them
    that the frame does not hold reads back as ``gr.skip()``, which is what
    the listener is handed for it, so a test or a caller can ask any of them
    without knowing which ones this particular frame set. A name outside them
    is an error, on writing and on reading alike: that is the typo the old
    positional tuples would have published into the wrong component.
    """

    def __init__(
        self,
        names: Sequence[str],
        values: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
        /,
        **named: Any,
    ):
        super().__init__()
        self.names = tuple(names)
        self.update(values, **named)

    def _check(self, name: str) -> None:
        if name not in self.names:
            raise KeyError(
                f"{name!r} is not one of this frame's outputs: {', '.join(self.names)}"
            )

    def __setitem__(self, name: str, value: Any) -> None:
        self._check(name)
        super().__setitem__(name, value)

    def __missing__(self, name: str) -> Any:
        self._check(name)
        return gr.skip()

    def update(self, values=(), /, **named: Any) -> None:
        for name, value in dict(values, **named).items():
            self[name] = value

    def setdefault(self, name: str, default: Any = None) -> Any:
        self._check(name)
        return super().setdefault(name, default)

    def copy(self) -> Frame:
        return type(self)(self.names, self)

    def __reduce__(self):
        # dict's own pickling rebuilds the items through __setitem__ before
        # __init__ has set ``names``; rebuilding through __init__ keeps them.
        return type(self), (self.names, dict(self))

    def __repr__(self) -> str:
        return f"Frame({dict.__repr__(self)})"


def positional(frame: Mapping[str, Any], names: Sequence[str]) -> tuple:
    """``frame`` as the tuple a listener registered with ``names`` expects.

    The one conversion from names to positions. A name the listener was not
    registered with is refused rather than dropped: it is an output some
    handler meant to publish, and dropping it would be the silent failure
    this module exists to prevent.
    """

    stray = [name for name in frame if name not in names]
    if stray:
        raise ValueError(
            f"Outputs {', '.join(map(repr, stray))} are not among this listener's: "
            f"{', '.join(names)}"
        )
    return tuple(frame.get(name, gr.skip()) for name in names)


# Every generation handler publishes these: a reply streaming, a refusal, an
# edit, a branch. Most frames skip most of them.
CHAT_OUTPUT_NAMES = (
    "prompt",
    "chatbot",
    "turns",
    "strip",
    "metrics",
    "status",
    "seed",
    "send",
    "stop",
    "detail",
    "alternatives",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "context_ids",
    "chat_metrics",
    "chat_context_ids",
    "selected_token",
    "branch_pick",
)

# The token view's editor, which Save and regenerate closes or reopens beside
# the reply it starts. These belong to the view rather than the conversation,
# so a background run switched back to does not put them back.
EDITOR_OUTPUT_NAMES = ("token_editor", "token_edit_target")
TOKEN_EDIT_OUTPUT_NAMES = (*CHAT_OUTPUT_NAMES, *EDITOR_OUTPUT_NAMES)

# The conversations pane: the forks behind it and the list that draws them.
PANE_OUTPUT_NAMES = ("forks", "conversation_list")

# What the background timer repaints: anything a running reply can publish,
# and the pane that says which conversation it is running in.
POLL_OUTPUT_NAMES = (*TOKEN_EDIT_OUTPUT_NAMES, *PANE_OUTPUT_NAMES)

STOP_OUTPUT_NAMES = ("chatbot", "turns", "strip", "send", "stop", "status")

UNDO_OUTPUT_NAMES = (
    "prompt",
    "chatbot",
    "turns",
    "strip",
    "metrics",
    "status",
    "detail",
    "alternatives",
    "send",
    "stop",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "selected_token",
    "branch_pick",
)

# Clear also resets the forks and their picker, and closes the confirmation
# panel that sent it.
CLEAR_OUTPUT_NAMES = (
    "chatbot",
    "turns",
    "strip",
    "metrics",
    "status",
    "send",
    "stop",
    "detail",
    "alternatives",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "selected_token",
    "branch_pick",
    "forks",
    "conversation_list",
    "clear_confirm",
)

# Fork, switch and delete. New conversation also empties the branch text,
# whose replacement belonged to the conversation being left.
FORK_OUTPUT_NAMES = (
    "prompt",
    "chatbot",
    "turns",
    "forks",
    "conversation_list",
    "status",
    "send",
    "stop",
    "strip",
    "metrics",
    "detail",
    "alternatives",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "selected_token",
    "branch_pick",
)
NEW_CONVERSATION_OUTPUT_NAMES = (*FORK_OUTPUT_NAMES, "branch_text")

RESTORE_OUTPUT_NAMES = ("chatbot", "turns", "forks", "conversation_list", "metrics")

LOAD_OUTPUT_NAMES = (
    "chatbot",
    "turns",
    "system_prompt",
    "strip",
    "metrics",
    "status",
    "detail",
    "alternatives",
    "send",
    "stop",
    "prompt_strip",
    "prompt_metrics",
    "prompt_note",
    "summary",
    "surprise",
    "trace",
    "selected_token",
    "branch_pick",
)

# The steering controls, in the order ui.steering.controls() returns them.
STEERING_OUTPUT_NAMES = (
    "steering_state",
    "steering_enabled",
    "steering_strength",
    "steering_layer",
    "steering_status",
)
STEERED_LOAD_OUTPUT_NAMES = (*LOAD_OUTPUT_NAMES, "forks", *STEERING_OUTPUT_NAMES)

# Every name above, which is what the layout's table has to cover.
CONVERSATION_OUTPUT_NAMES = tuple(
    dict.fromkeys(
        (
            *POLL_OUTPUT_NAMES,
            *STOP_OUTPUT_NAMES,
            *UNDO_OUTPUT_NAMES,
            *CLEAR_OUTPUT_NAMES,
            *NEW_CONVERSATION_OUTPUT_NAMES,
            *RESTORE_OUTPUT_NAMES,
            *STEERED_LOAD_OUTPUT_NAMES,
        )
    )
)
