"""The conversations kept between sessions.

Every branch in the conversations pane - the main conversation, its forks and
the chats started beside it - is written to one JSON file as it changes and
read back when the page next loads, so a browser reload, an app restart or a
crash loses nothing that was said. The file is the whole pane at once: the
name of the active branch and the turns of every branch, in the order the
pane lists them. It is written whole and swapped into place, so a crash
mid-write leaves the previous copy rather than half of a new one.

Where it lives::

    ~/.local/share/chatlab/conversations.json

``XDG_DATA_HOME`` moves the directory and ``CHATLAB_LIBRARY_PATH`` names the
file outright, the same two knobs the settings file answers to. Token
measurements are not kept: they describe one response as one model produced
it, and belong to the session that produced it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from conversation import (
    MAIN_BRANCH,
    copy_forks,
    copy_turns,
    new_forks,
    turn_entries,
    turns_from_entries,
)
from trace_export import write_private_text

logger = logging.getLogger(__name__)

LIBRARY_FORMAT = "chatlab-library-1"
LIBRARY_PATH_ENV = "CHATLAB_LIBRARY_PATH"
XDG_DATA_ENV = "XDG_DATA_HOME"
LIBRARY_DIRECTORY = "chatlab"
LIBRARY_FILENAME = "conversations.json"


def library_path() -> Path:
    """Where the conversations are read from and written to."""

    chosen = os.environ.get(LIBRARY_PATH_ENV)
    if chosen:
        return Path(chosen).expanduser()
    configured = os.environ.get(XDG_DATA_ENV)
    root = Path(configured).expanduser() if configured else Path.home() / ".local" / "share"
    return root / LIBRARY_DIRECTORY / LIBRARY_FILENAME


def as_seen(forks: dict | None, turns: list[dict] | None) -> dict:
    """The pane as the reader sees it: ``forks`` with the active branch read from ``turns``.

    The active branch's entry in ``forks`` is stale by design - the handlers
    only write it back when switching away - so the conversation on screen is
    the truth for that one branch and ``forks`` for every other.
    """

    forks = copy_forks(forks)
    forks["branches"][forks["active"]] = copy_turns(turns)
    return forks


def dump(forks: dict | None) -> str:
    forks = forks or new_forks()
    payload = {
        "format": LIBRARY_FORMAT,
        "active": forks.get("active", MAIN_BRANCH),
        "branches": [
            {"name": name, "turns": turn_entries(turns)}
            for name, turns in forks.get("branches", {}).items()
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse(payload: str) -> dict:
    """The forks a saved file describes, or ``ValueError`` for one this app did not write."""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"The conversations file is not valid JSON: {error}") from error
    if not isinstance(data, dict) or data.get("format") != LIBRARY_FORMAT:
        raise ValueError(f"Expected a {LIBRARY_FORMAT} file written by this app.")
    raw_branches = data.get("branches")
    if not isinstance(raw_branches, list):
        raise ValueError("The conversations file has no list of branches.")

    branches: dict[str, list[dict]] = {}
    for entry in raw_branches:
        if not isinstance(entry, dict):
            raise ValueError("Every branch must be an object.")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every branch needs a name.")
        if name in branches:
            raise ValueError(f"The branch {name!r} appears twice.")
        turns = turns_from_entries(entry.get("turns"))
        # A response that was still streaming when the file was written is
        # kept as far as it got, and closed, so its reasoning block does not
        # spin for the rest of the next session. One that had produced
        # nothing yet is dropped, as Stop drops it.
        if turns and turns[-1]["role"] == "assistant":
            if turns[-1].get("content") or turns[-1].get("reasoning"):
                turns[-1]["reasoning_closed"] = True
            else:
                turns.pop()
        branches[name] = turns
    if not branches:
        branches[MAIN_BRANCH] = []

    active = data.get("active")
    if active not in branches:
        active = next(iter(branches))
    return {"active": active, "branches": branches}


def read(path: Path | None = None) -> dict | None:
    """The saved conversations, or ``None`` when there are none to restore.

    An unreadable file is reported in the log and treated as absent rather
    than raised: the page must still open, and the file is left where it is
    for the reader to look at.
    """

    target = path or library_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning("Could not read the conversations in %s: %s", target, error)
        return None
    try:
        return parse(raw)
    except ValueError as error:
        logger.warning("Ignoring the conversations in %s: %s", target, error)
        return None


def write(forks: dict | None, path: Path | None = None) -> Path | None:
    """Write the pane to disk, whole, and return the path; ``None`` if it could not be."""

    target = path or library_path()
    text = dump(forks)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Into a sibling first and then over the old file in one rename, so
        # the file on disk is always a complete one. The sibling carries
        # this process's id, so two instances writing at once do not tread
        # on each other's half-written copy.
        staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        write_private_text(staging, text)
        os.replace(staging, target)
    except OSError as error:
        logger.warning("Could not save the conversations to %s: %s", target, error)
        return None
    return target
