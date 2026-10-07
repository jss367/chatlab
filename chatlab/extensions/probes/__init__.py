"""Linear probes on the loaded model's residual stream, read token by token."""

__all__ = ["CSS", "build_page"]


def __getattr__(name):
    # Importing the probe file schema does not enable the optional page.
    if name in __all__:
        from . import page

        return getattr(page, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
