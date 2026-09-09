"""The one model manager the interface talks to.

A module of its own so every page reads the same object, and so a test can
put a stub in its place with ``runtime.MANAGER = ...``."""

from __future__ import annotations

from model_runtime import (
    ModelManager,
)


MANAGER = ModelManager()
