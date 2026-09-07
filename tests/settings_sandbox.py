"""Keep the tests off the files belonging to whoever runs them.

``app.build_app()`` reads the settings file and creates it when it is
missing, the model runtime reads the memory limits out of it, and the
conversation handlers write the saved conversations, so a test would
otherwise depend on one person's saved choices and rewrite them. A module
that touches any of these calls :func:`start` from ``setUpModule`` and
:func:`stop` from ``tearDownModule``.
"""

import os
import tempfile
from pathlib import Path

import library
import settings

_directory: tempfile.TemporaryDirectory | None = None
_previous: dict[str, str | None] = {}


def start() -> Path:
    """Point the settings and conversations files at temporary ones that do not exist yet."""

    global _directory, _previous
    _directory = tempfile.TemporaryDirectory()
    _previous = {
        name: os.environ.get(name)
        for name in (settings.SETTINGS_PATH_ENV, library.LIBRARY_PATH_ENV)
    }
    path = Path(_directory.name) / "settings.json"
    os.environ[settings.SETTINGS_PATH_ENV] = str(path)
    os.environ[library.LIBRARY_PATH_ENV] = str(Path(_directory.name) / "conversations.json")
    settings.load()
    return path


def stop() -> None:
    """Put back the real files, and forget the temporary ones."""

    global _directory, _previous
    for name, value in _previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _previous = {}
    if _directory is not None:
        _directory.cleanup()
        _directory = None
    settings.load()
