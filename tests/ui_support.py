"""Helpers for the tests that read the built Gradio page.

A test reaches a handler the way Gradio does, through the listener
``demo.fns`` holds for it, so it gets the inputs, outputs and queue the page
really wired rather than the bare function. Listeners are found by the name
of the function they call, because many handlers are closures the page
builds, with no module attribute to compare against.
"""


def _name(listener) -> str | None:
    # A listener that only runs JavaScript has no function; a partial has no name.
    return getattr(listener.fn, "__name__", None)


def listeners_named(demo, name: str) -> list:
    """Every listener on ``demo`` whose function is called ``name``, in wiring order."""

    return [listener for listener in demo.fns.values() if _name(listener) == name]


def listener_named(demo, name: str):
    """The first listener on ``demo`` whose function is called ``name``."""

    for found in demo.fns.values():
        if _name(found) == name:
            return found
    raise LookupError(f"no listener calls {name!r}")


def listeners_by_name(demo) -> dict:
    """Each named listener on ``demo``, keyed by its function's name.

    Where two listeners call functions of one name, the later-wired one is
    kept, as a dict built over ``demo.fns`` keeps it.
    """

    return {_name(found): found for found in demo.fns.values() if _name(found)}


def handlers_by_name(demo) -> dict:
    """The functions behind :func:`listeners_by_name`, to call directly."""

    return {name: found.fn for name, found in listeners_by_name(demo).items()}
