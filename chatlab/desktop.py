"""What a page may ask of the native window around it.

The pages are built before that window exists, and ``python -m chatlab`` serves
the same pages to a browser with no window at all. So a page asks whether a
restart is on offer as it is built, and asks for the restart itself when the
reader presses the button; the launcher answers both by leaving a handler
here once it has a window to close and a bundle to reopen.
"""

_restart_handler = None

# What a page says when nobody is listening, which is what ``python -m chatlab``
# looks like: no bundle to reopen and no window to close, so the reader has to
# do it.
NO_RESTART_AVAILABLE = "Quit and reopen ChatLab to apply this change."


def offer_restart(handler) -> None:
    """Let the pages quit and reopen the app by calling ``handler``.

    ``handler`` returns ``None`` when it is restarting, or a sentence saying
    why it is not: a restart is a close, and a close can be refused.
    """

    global _restart_handler
    _restart_handler = handler


def restart_offered() -> bool:
    """Whether anything is listening for a restart."""

    return _restart_handler is not None


def restart() -> str | None:
    """Restart the app. ``None`` means it is; a string says why it is not."""

    handler = _restart_handler
    if handler is None:
        return NO_RESTART_AVAILABLE
    return handler()
