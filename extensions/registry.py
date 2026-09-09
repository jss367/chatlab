"""Explicit first-party catalogue. Disabled modules are never imported.

No arbitrary package installation or directory scanning: external distribution
can later implement this same manifest/builder contract after the API settles.
"""
from dataclasses import dataclass
from importlib import import_module
import logging

from extension_api import API_VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtensionSpec:
    id: str
    title: str
    description: str
    page_label: str
    module: str
    api_version: int = API_VERSION
    icon: str = "▦"


CATALOGUE = (
    ExtensionSpec("maze_experiments", "Maze experiments",
                  "Run navigation trials, insert interruptions, inspect tokens and replay saved runs.",
                  "Maze", "extensions.maze_experiments"),
)


@dataclass(frozen=True)
class LoadedExtension:
    spec: ExtensionSpec
    build_page: object
    css: str


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
            loaded.append(LoadedExtension(spec, module.build_page, module.CSS))
        except Exception as exc:
            logger.exception("Could not load extension %s", spec.id)
            errors.append(f"{spec.title}: {exc}")
    return loaded, errors
