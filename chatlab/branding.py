"""Where ChatLab's mark lives, in a checkout and inside the app bundle."""

from __future__ import annotations

import sys
from pathlib import Path


# PyInstaller unpacks the bundle's data files under ``sys._MEIPASS``; a
# checkout has them beside the package. Nothing else in ChatLab reads a data
# file, so this is the only place that has to know the difference.
ASSETS = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets"


# The tile alone, corners cut away, drawn into the browser tab. A bundle
# built without it still runs, so callers pass ``None`` on rather than fail
# a launch over a missing picture.
FAVICON = ASSETS / "icon.png"


def favicon_path() -> str | None:
    """Return the favicon's path for Gradio, or ``None`` if it is absent."""

    return str(FAVICON) if FAVICON.is_file() else None
