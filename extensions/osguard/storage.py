"""Private, atomic checkpoints; create the extension directory only on write."""
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from extension_api import write_private_text


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid4().hex}.tmp")
    try:
        write_private_text(temporary, json.dumps(value, ensure_ascii=False, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_copy(path, folder=None):
    """A copy of ``path`` that Gradio will serve, in ``folder`` or a new temporary one.

    Gradio only serves returned files from its temporary directories and the
    working directory, and the extension's data directory is neither when the
    app is started from a checkout. The copy is as private as the original,
    and swapped in whole, so a download already reading the previous one is
    never handed half a file.
    """
    folder = Path(folder) if folder is not None else Path(tempfile.mkdtemp(prefix="chatlab-osguard-"))
    target = folder / path.name
    temporary = folder / f".{uuid4().hex}.tmp"
    try:
        write_private_text(temporary, path.read_text(encoding="utf-8"))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
