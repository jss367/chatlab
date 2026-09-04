"""Keep the tests off the settings file belonging to whoever runs them.

``app.build_app()`` reads the settings file and creates it when it is
missing, and the model runtime reads the memory limits out of it, so a test
would otherwise depend on one person's saved choices and rewrite them. A
module that touches either calls :func:`start` from ``setUpModule`` and
:func:`stop` from ``tearDownModule``.
"""

import os
import tempfile
from pathlib import Path

import settings

_directory: tempfile.TemporaryDirectory | None = None
_previous: str | None = None


def start() -> Path:
    """Point the settings file at a temporary one that does not exist yet."""

    global _directory, _previous
    _directory = tempfile.TemporaryDirectory()
    _previous = os.environ.get(settings.SETTINGS_PATH_ENV)
    path = Path(_directory.name) / "settings.json"
    os.environ[settings.SETTINGS_PATH_ENV] = str(path)
    settings.load()
    return path


def stop() -> None:
    """Put back the real settings file, and forget the temporary one."""

    global _directory, _previous
    if _previous is None:
        os.environ.pop(settings.SETTINGS_PATH_ENV, None)
    else:
        os.environ[settings.SETTINGS_PATH_ENV] = _previous
    if _directory is not None:
        _directory.cleanup()
        _directory = None
    settings.load()
