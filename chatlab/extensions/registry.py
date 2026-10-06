"""Explicit first-party catalogue. Disabled modules are never imported.

No arbitrary package installation or directory scanning: external distribution
can later implement this same manifest/builder contract after the API settles.
"""
from dataclasses import dataclass
from importlib import import_module
import logging

from chatlab.extension_api import API_VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtensionSpec:
    id: str
    title: str
    description: str
    page_label: str
    module: str
    api_version: int = API_VERSION
    # A name in ui.icons, drawn as a mask on the extension's nav tile.
    icon: str = "box"


CATALOGUE = (
    ExtensionSpec("maze_experiments", "Maze experiments",
                  "Run navigation trials, insert interruptions, inspect tokens and replay saved runs.",
                  "Maze", "chatlab.extensions.maze_experiments", icon="route"),
    ExtensionSpec("os_harm", "OS-Harm results",
                  "Compare computer-use safety evaluations and replay recorded screenshots, responses and actions.",
                  "OS-Harm", "chatlab.extensions.os_harm", icon="image"),
    ExtensionSpec("osguard", "Computer-use safety benchmark",
                  "Evaluate OSGuard action judgments, inspect tokens and review desktop safety results.",
                  "Safety", "chatlab.extensions.osguard", icon="box"),
    ExtensionSpec("hangman", "Hangman",
                  "Play hangman with the model as host, check each board against the last and branch replies at any token.",
                  "Hangman", "chatlab.extensions.hangman", icon="spell-check"),
    ExtensionSpec("probes", "Linear probes",
                  "Fit a logistic probe at every layer from labelled examples and read any reply with it, token by token.",
                  "Probes", "chatlab.extensions.probes", icon="layers"),
    ExtensionSpec("circuits", "Circuit tracing",
                  "Trace which transcoder features carried the model to a token, then ablate or boost groups of them.",
                  "Circuits", "chatlab.extensions.circuits", icon="network"),
    ExtensionSpec("direction_edits", "Direction edits",
                  "Inject a vector, erase or clamp a direction at chosen blocks, and see whether later blocks and the lens rebuild it.",
                  "Edits", "chatlab.extensions.direction_edits", icon="crosshair"),
)


@dataclass(frozen=True)
class LoadedExtension:
    spec: ExtensionSpec
    build_page: object
    css: str
    # Optional script run once when the page loads, as the host's own are.
    js: str = ""


def load_enabled(enabled_ids, catalogue=CATALOGUE):
    loaded, errors = [], []
    for spec in catalogue:
        if spec.id not in enabled_ids:
            continue
        try:
            if spec.api_version != API_VERSION:
                raise ValueError(f"requires extension API {spec.api_version}; this app provides {API_VERSION}")
            module = import_module(spec.module)
            if not callable(module.build_page) or not isinstance(module.CSS, str):
                raise ValueError("must export a page builder and CSS string")
            js = getattr(module, "JS", "")
            if not isinstance(js, str):
                raise ValueError("must export its script, if any, as a string")
            loaded.append(LoadedExtension(spec, module.build_page, module.CSS, js))
        except Exception as exc:
            logger.exception("Could not load extension %s", spec.id)
            errors.append(f"{spec.title}: {exc}")
    return loaded, errors
