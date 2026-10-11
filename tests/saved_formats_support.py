"""Shared plumbing for the tests that read the frozen files under tests/fixtures/saved.

The fixtures are never written to: each test copies what it needs into a
temporary directory and points the conversations and settings files there,
since several readers resolve sibling files (vectors, pictures, the archive
companion, experiments) next to the conversations file.
"""

import os
import shutil
import tempfile
from pathlib import Path

from chatlab import library, settings

SAVED = Path(__file__).resolve().parent / "fixtures" / "saved"


class Sandbox:
    """A temporary home for the conversations and settings files, undone by ``close``."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self._previous = {
            name: os.environ.get(name)
            for name in (library.LIBRARY_PATH_ENV, settings.SETTINGS_PATH_ENV)
        }
        self.use_library(self.root / "conversations.json")
        os.environ[settings.SETTINGS_PATH_ENV] = str(self.root / "settings.json")
        settings.load()

    def use_library(self, path: Path) -> Path:
        os.environ[library.LIBRARY_PATH_ENV] = str(path)
        return path

    def copy(self, source: Path, name: str | None = None) -> Path:
        """A writable copy of a fixture file or directory inside the sandbox."""

        target = self.root / (name or source.name)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return target

    def close(self):
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._directory.cleanup()
        settings.load()


def sandbox(test_case) -> Sandbox:
    """A :class:`Sandbox` closed when ``test_case`` finishes."""

    box = Sandbox()
    test_case.addCleanup(box.close)
    return box
