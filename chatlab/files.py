"""Writing files only their owner can read.

Exports, transcripts, saved runs and batch tables all go through here, so
every file ChatLab writes is created owner-only before any of its contents
reach the disk. Nothing in this module imports the rest of ChatLab, so any
module can use it without an import cycle.
"""

from __future__ import annotations

import csv
import io
import os
import re
import time
from pathlib import Path
from uuid import uuid4


def write_private_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Write ``text`` to ``path`` as a file only its owner can read.

    ``Path.write_text()`` creates the file with whatever the process umask
    allows - usually 0644 - and puts every byte of it on disk before a
    following ``chmod`` can narrow it. Exports and saved transcripts land in
    shared directories, so another account on the machine can open the file
    during that window and read it. Creating the file 0600 and settling its
    mode on the descriptor, before anything is written into it, closes the
    window. The mode is set explicitly rather than left to ``os.open()``
    because the umask can only take bits away from the mode it is given, so a
    strict one would otherwise leave the owner unable to read their own file.
    """

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline=newline)
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        handle.write(text)


def append_private_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Add ``text`` to the end of ``path``, keeping it owner-only.

    The mode is given to ``os.open()`` for the case where the file is not
    there yet, and settled on the descriptor either way, so a file created by
    an append is as private as one written whole; see write_private_text() for
    why the mode cannot wait until afterwards.
    """

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "a", encoding="utf-8", newline=newline)
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        handle.write(text)


def replace_private_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Write ``text`` beside ``path`` and move it into place once it is whole.

    A file rewritten in place and cut off part way, on a full disk most often,
    loses the last good copy along with the new one. Staging it under a
    hidden name in the same directory keeps the swap a single rename, and the
    staged file is removed whether or not the swap happened.
    """

    staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_private_text(staged, text, newline=newline)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def batch_directory(root, title) -> Path:
    """A new directory under ``root/batches``, named for when it started and its title."""

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "trials"
    directory = Path(root) / "batches" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    suffix, candidate = 1, directory
    while candidate.exists():
        suffix += 1
        candidate = directory.with_name(f"{directory.name}-{suffix}")
    candidate.mkdir(parents=True)
    return candidate


def csv_text(columns, rows) -> str:
    """``rows`` as CSV text with a header of ``columns``, one ``\\n`` per line."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
