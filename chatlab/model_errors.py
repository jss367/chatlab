"""Why the model a caller meant cannot be used, raised across the model modules.

The loading, generation, inspection and cache modules all raise these, and
the API and every page catch them, so they live apart from all of those.
"""


class ModelChanged(RuntimeError):
    """The weights in memory are not the ones the caller's tokens came from."""


class ModelInUse(RuntimeError):
    """The model cannot be removed, or claimed, right now.

    The subclasses say why, so the interface can tell the reader what to do:
    unload the model, wait for its download, or wait for the model to go idle.
    """


class ModelLoaded(ModelInUse):
    """The model is the one in memory."""


class ModelDownloading(ModelInUse):
    """A download of the model is under way, in this process or another."""


class ModelBusy(ModelInUse):
    """The model lock is held: a load, generation, scoring, or inspection is running."""
