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

A branch can also carry its own sampling - the temperature, top-p, top-k and
response length it answers with - written beside its turns and stamped
separately, so a page that moves a slider does not thereby claim a
transcript it may be a reply behind on, and a newer transcript does not undo
a slider moved on another page. A branch with no sampling of its own answers
with the saved settings.

Where it lives::

    ~/.local/share/chatlab/conversations.json

``XDG_DATA_HOME`` moves the directory and ``CHATLAB_LIBRARY_PATH`` names the
file outright, the same two knobs the settings file answers to. Token
measurements are not kept: they describe one response as one model produced
it, and belong to the session that produced it.

Steering references are small enough to keep with the turns and settings.
Their immutable vector data lives once in a sibling ``conversations-vectors``
directory, so saving a token frame does not rewrite every previous vector.
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
    SAMPLING_FIELDS,
    copy_forks,
    next_branch_name,
    put_branch,
    turn_entries,
    turns_from_entries,
)
from steering import compact as compact_steering
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

    The conversation on screen is the truth for the active branch and
    ``forks`` for every other: the handlers that change the conversation do
    not touch ``forks``, and its entry for the active branch is brought up to
    date from the screen afterwards, by ``refresh_conversation_list`` in
    ``app.py``. The active branch is stamped as changed now only if it
    differs from that entry, so the stamp records when the branch changed,
    not when it was next saved, and a page that has done nothing to it does
    not claim it over another page that has.
    """

    forks = copy_forks(forks)
    put_branch(forks, forks["active"], turns)
    return forks


def sampling_entry(values: dict | None) -> dict:
    """A branch's sampling as a file spells it: the known keys, rightly typed.

    A value of the wrong type is dropped rather than written, so a file this
    version reads back is one it can also parse. The types are the file's
    business alone; what the values may be is the settings module's.

    A key this version knows nothing about is carried through untouched, the
    way the settings file carries its own unknown keys: two machines sharing
    one file need not run the same version, and a branch's sampling written
    by the newer of them must survive being read and saved by the older.
    """

    entry = {
        key: value
        for key, value in (values or {}).items()
        if key not in SAMPLING_FIELDS
    }
    for key, kind in SAMPLING_FIELDS.items():
        value = (values or {}).get(key)
        if isinstance(value, bool):
            continue
        if kind is float and isinstance(value, int):
            value = float(value)
        if isinstance(value, kind):
            entry[key] = value
    if "steering" in entry:
        entry["steering"] = compact_steering(entry["steering"])
    return entry


def dump(forks: dict | None) -> str:
    forks = copy_forks(forks)
    updated = forks["updated"]
    branches = []
    for name, turns in forks["branches"].items():
        entry = {"name": name, "turns": turn_entries(turns)}
        sampling = sampling_entry(forks["sampling"].get(name))
        if sampling:
            entry["sampling"] = sampling
        # Written whether or not there is sampling beside it: a stamp on its
        # own says the sampling was taken away, and another page holding an
        # older copy must not put it back.
        stamp = forks["sampling_updated"].get(name)
        if stamp:
            entry["sampling_updated"] = stamp
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
    sampling: dict[str, dict] = {}
    sampling_updated: dict[str, str] = {}
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
        held = entry.get("sampling")
        if held is not None:
            if not isinstance(held, dict):
                raise ValueError(f"The branch {name!r} has sampling that is not an object.")
            # Whatever is unusable is left out here and falls back to the
            # saved setting when the conversation is answered, the same way
            # a hand-edited settings file does.
            kept = sampling_entry(held)
            if kept:
                sampling[name] = kept
        # Read whether or not any sampling came with it: on its own it says
        # the sampling was taken away, and when it was.
        sampling_stamp = entry.get("sampling_updated")
        if sampling_stamp is not None:
            if not isinstance(sampling_stamp, str):
                raise ValueError(
                    f"The branch {name!r} has a sampling time that is not a string."
                )
            sampling_updated[name] = sampling_stamp
        turns = turns_from_entries(entry.get("turns"))
        # A response that was still streaming when the file was written is
        # kept as far as it got, and closed, so its reasoning block does not
        # spin for the rest of the next session. One that had produced
        # nothing yet is dropped, as Stop drops it. A completed invisible
        # token step keeps its assistant slot even without visible text.
        if turns and turns[-1]["role"] == "assistant":
            if (turns[-1].get("content") or turns[-1].get("reasoning")
                    or turns[-1].get("token_step_paused")):
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
    return {
        "active": active,
        "branches": branches,
        "sampling": sampling,
        "sampling_updated": sampling_updated,
        "updated": updated,
    }


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
    sampling: dict[str, dict] = {}
    sampling_updated: dict[str, str] = {}
    updated: dict[str, str] = {}

    def newer(name: str, times: str) -> dict:
        """Whichever side touched ``name`` more recently by the ``times`` stamps."""

        ours = mine[times].get(name, "")
        its = theirs[times].get(name, "")
        if ours > its:
            return mine
        if its > ours:
            return theirs
        return mine if name in mine["branches"] else theirs

    for name in names:
        winner = newer(name, "updated")
        if name in winner["branches"]:
            branches[name] = winner["branches"][name]
            # The sampling is merged on its own stamp rather than the
            # branch's. A page that moves a slider may be a reply behind the
            # page that made it: it must not win the transcript, and the
            # newer transcript must not undo its slider.
            side = newer(name, "sampling_updated")
            stamp = side["sampling_updated"].get(name)
            if stamp:
                # Kept even where that side has no sampling: the stamp is
                # then a removal, and it has to outlive this merge or the
                # next page still holding the old entry would put it back.
                sampling_updated[name] = stamp
            held = side["sampling"].get(name)
            if held:
                sampling[name] = held
        stamp = winner["updated"].get(name)
        if stamp:
            updated[name] = stamp

    if not branches:
        branches[MAIN_BRANCH] = []
    active = mine["active"] if mine["active"] in branches else next(iter(branches))
    return {
        "active": active,
        "branches": branches,
        "sampling": sampling,
        "sampling_updated": sampling_updated,
        "updated": updated,
    }


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


def taken_names(path: Path | None = None) -> set[str]:
    """The branch names the file on disk has spoken for: those it holds and those it has forgotten.

    For naming a new branch. Two pages that each start a chat before seeing
    the other's save would otherwise both call it ``Chat 1``, and
    :func:`merge` would take the two for one branch and keep only one. A
    forgotten name is avoided too, so a new branch does not answer to the
    name of one another page may still hold.
    """

    forks = read(path)
    if forks is None:
        return set()
    return set(forks["branches"]) | set(forks["updated"])


def _replace(target: Path, text: str) -> bool:
    """Put ``text`` in place of ``target`` in one rename; ``False`` if it could not be.

    Into a sibling first and then over the old file, so the file on disk is
    always a complete one. The sibling's name is unique to this write, so two
    writes - from this process or another - never stage into, or rename
    away, each other's copy. Called with ``_WRITE_LOCK`` held.
    """

    staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(staging, text)
        os.replace(staging, target)
    except OSError as error:
        logger.warning("Could not save the conversations to %s: %s", target, error)
        # Whatever of the staged copy got as far as disk is not left beside
        # the file; the failure may have been before any of it did.
        with suppress(OSError):
            staging.unlink()
        return False
    return True


def write(forks: dict | None, path: Path | None = None, *, preserve_active: bool = False) -> Path | None:
    """Merge the pane into the file on disk and return the path; ``None`` if it could not be.

    What is on disk is read first and merged with ``forks`` as :func:`merge`
    describes, so a save from one page keeps what another page saved since
    this one loaded. A file that cannot be read is replaced, as it always
    was; :func:`read` has said why in the log.

    Background jobs use ``preserve_active`` to save their source transcript
    without changing which conversation the reader selected most recently.
    """

    target = path or library_path()
    with _WRITE_LOCK:
        existing = read(target)
        merged = merge(forks, existing)
        if preserve_active and existing and existing["active"] in merged["branches"]:
            merged["active"] = existing["active"]
        if not _replace(target, dump(merged)):
            return None
    return target


def claim_name(forks: dict | None, prefix: str, path: Path | None = None) -> str:
    """Pick the next free ``<prefix> N`` and write it into the file before letting go.

    Choosing a name and saving the branch that bears it are two steps, and
    two pages that start a chat between each other's saves would both read
    the same file, both choose ``Chat 1``, and :func:`merge` would then take
    the two for one branch and keep only one. Here the name is chosen and an
    empty branch written under it in one go, with the lock every save takes,
    so the next page to look sees it spoken for. The page's own save then
    fills the branch in. A file that cannot be written still yields a name;
    the save that follows will say why in the log.
    """

    forks = copy_forks(forks)
    target = path or library_path()
    with _WRITE_LOCK:
        on_disk = read(target)
        taken = set(on_disk["branches"]) | set(on_disk["updated"]) if on_disk else set()
        name = next_branch_name(forks, prefix, taken)
        put_branch(forks, name, [])
        _replace(target, dump(merge(forks, on_disk)))
    return name
