"""The conversations kept between sessions.

Every branch in the conversations pane - the main conversation, its forks and
the chats started beside it - is written to one JSON file as it changes and
read back when the page next loads, so a browser reload, an app restart or a
crash loses nothing that was said. The file holds the whole pane: the name of
the active branch and the turns of every branch, in the order the pane lists
them, each with when it last changed. It is written whole and swapped into
place, so a crash mid-write leaves the previous copy rather than half of a
new one.

Two pages can be open on the same file - two browser tabs, or a reload with
the old tab still up - and each holds its own copy of the pane from when it
loaded. Rather than the last page to write replacing everything the other
did, a save is merged into the file one branch at a time: a branch only the
file knows stays, and where both hold one the copy that changed more
recently wins. A branch a page deleted stays deleted, and the file remembers
the deletion so a page still holding the branch does not put it back - see
:func:`merge`. Within one process the whole read-merge-replace is done under
a lock, so two handlers saving at once cannot each merge into the same old
copy and the second replace the first's work; two processes on one file are
not protected against each other.

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
import threading
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from conversation import (
    MAIN_BRANCH,
    copy_forks,
    put_branch,
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

# Held across the whole of write(): read the file, merge, stage, replace. The
# two listeners in app.py that save both run on Gradio's worker threads, and
# without this each could merge into the same old copy of the file and the
# later replace would drop what the earlier had merged in.
_WRITE_LOCK = threading.Lock()


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
    the truth for that one branch and ``forks`` for every other. The active
    branch is stamped as changed now only if it differs from that entry, so
    a page that has done nothing to it does not claim it over another page
    that has.
    """

    forks = copy_forks(forks)
    put_branch(forks, forks["active"], turns)
    return forks


def dump(forks: dict | None) -> str:
    forks = copy_forks(forks)
    updated = forks["updated"]
    branches = []
    for name, turns in forks["branches"].items():
        entry = {"name": name, "turns": turn_entries(turns)}
        if name in updated:
            entry["updated"] = updated[name]
        branches.append(entry)
    payload = {
        "format": LIBRARY_FORMAT,
        "active": forks["active"],
        "branches": branches,
        # The branches deleted, and when, so a page still holding one of
        # them does not put it back. See merge().
        "forgotten": {
            name: stamp for name, stamp in updated.items() if name not in forks["branches"]
        },
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
    updated: dict[str, str] = {}
    for entry in raw_branches:
        if not isinstance(entry, dict):
            raise ValueError("Every branch must be an object.")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every branch needs a name.")
        if name in branches:
            raise ValueError(f"The branch {name!r} appears twice.")
        stamp = entry.get("updated")
        if stamp is not None:
            if not isinstance(stamp, str):
                raise ValueError(f"The branch {name!r} has an updated time that is not a string.")
            updated[name] = stamp
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

    forgotten = data.get("forgotten", {})
    if not isinstance(forgotten, dict) or not all(
        isinstance(name, str) and isinstance(stamp, str) for name, stamp in forgotten.items()
    ):
        raise ValueError("The forgotten branches must map names to times.")
    for name, stamp in forgotten.items():
        if name not in branches:
            updated[name] = stamp

    if not branches:
        branches[MAIN_BRANCH] = []
    active = data.get("active")
    if active not in branches:
        active = next(iter(branches))
    return {"active": active, "branches": branches, "updated": updated}


def merge(mine: dict | None, theirs: dict | None) -> dict:
    """The pane to write when ``mine`` is saved over a file that holds ``theirs``.

    Branch by branch, whichever side touched the branch more recently - by
    the ``updated`` stamps - decides what becomes of it: kept as that side
    has it, or, when that side deleted it, gone, with the time of the
    deletion kept so a page still holding the branch does not bring it back.
    A branch that only one side has a time for goes that side's way, so a
    page's copy of a branch it never touched loses to any dated copy in the
    file; one neither side has a time for is kept, ``mine`` first. Branches
    come in the order ``mine`` lists them, then those only the file had.
    ``theirs`` is ``None`` when there is no file yet.
    """

    mine = copy_forks(mine)
    if theirs is None:
        return mine
    theirs = copy_forks(theirs)

    names = list(mine["branches"])
    for held in (theirs["branches"], mine["updated"], theirs["updated"]):
        names += [name for name in held if name not in names]

    branches: dict[str, list[dict]] = {}
    updated: dict[str, str] = {}
    for name in names:
        ours = mine["updated"].get(name, "")
        its = theirs["updated"].get(name, "")
        if ours > its:
            winner = mine
        elif its > ours:
            winner = theirs
        else:
            winner = mine if name in mine["branches"] else theirs
        if name in winner["branches"]:
            branches[name] = winner["branches"][name]
        stamp = winner["updated"].get(name)
        if stamp:
            updated[name] = stamp

    if not branches:
        branches[MAIN_BRANCH] = []
    active = mine["active"] if mine["active"] in branches else next(iter(branches))
    return {"active": active, "branches": branches, "updated": updated}


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
    """Merge the pane into the file on disk and return the path; ``None`` if it could not be.

    What is on disk is read first and merged with ``forks`` as :func:`merge`
    describes, so a save from one page keeps what another page saved since
    this one loaded. A file that cannot be read is replaced, as it always
    was; :func:`read` has said why in the log.
    """

    target = path or library_path()
    with _WRITE_LOCK:
        text = dump(merge(forks, read(target)))
        # Into a sibling first and then over the old file in one rename, so
        # the file on disk is always a complete one. The sibling's name is
        # unique to this write, so two writes - from this process or
        # another - never stage into, or rename away, each other's copy.
        staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_private_text(staging, text)
            os.replace(staging, target)
        except OSError as error:
            logger.warning("Could not save the conversations to %s: %s", target, error)
            # Whatever of the staged copy got as far as disk is not left
            # beside the file; the failure may have been before any of it did.
            with suppress(OSError):
                staging.unlink()
            return None
    return target
