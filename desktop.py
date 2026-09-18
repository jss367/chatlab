"""What a page may ask of the native window around it.

The pages are built before that window exists, and ``python app.py`` serves
the same pages to a browser with no window at all. So a page asks whether a
restart is on offer as it is built, and asks for the restart itself when the
reader presses the button; the launcher answers both by leaving a handler
here once it has a window to close and a bundle to reopen.
"""

_restart_handler = None


def offer_restart(handler) -> None:
    """Let the pages quit and reopen the app by calling ``handler``."""

    global _restart_handler
    _restart_handler = handler


def restart_offered() -> bool:
    """Whether anything is listening for a restart."""

    return _restart_handler is not None


def restart() -> bool:
    """Restart the app. False means nobody was listening and nothing happened."""

    handler = _restart_handler
    if handler is None:
        return False
    handler()
    return True
