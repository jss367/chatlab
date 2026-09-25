"""The one model manager the interface talks to.

A module of its own so every page reads the same object, and so a test can
put a stub in its place with ``runtime.MANAGER = ...``."""

from __future__ import annotations

from chatlab.model_runtime import ModelManager


MANAGER = ModelManager()


def current_manager() -> ModelManager:
    """Whatever :data:`MANAGER` is at the moment of asking.

    What the local API is handed instead of the module: the API sits below
    the interface and does not import it, so whoever wires the two together
    passes this in. It reads the name on every call rather than holding the
    object, which is what lets a test's ``runtime.MANAGER = ...`` reach the
    API's routes as it reaches the pages.
    """

    return MANAGER
