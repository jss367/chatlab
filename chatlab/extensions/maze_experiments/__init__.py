"""Maze navigation and interruption experiments for ChatLab."""
from .page import CSS as _PAGE_CSS, build_page
from .reasoning_page import CSS as _REASONING_CSS

CSS = _PAGE_CSS + _REASONING_CSS

__all__ = ["CSS", "build_page"]
