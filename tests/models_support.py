"""Downloaded models and search-table events, for the tests of the pages that list them.

The Models page, the chat page's switcher and the layout tests all describe
the cache the same way, so the fixture model and the table helpers are kept
here rather than in any one of their modules.
"""

from pathlib import Path

import gradio as gr

from chatlab.model_cache import CachedModel, CacheStatus


COMMIT = "d97e442d7cc678210054dbcc9b440894d62c89a4"
OLMO = "allenai/Olmo-3-7B-Think"


def cached(model_id: str, **overrides) -> CachedModel:
    fields = dict(
        model_id=model_id,
        status=CacheStatus(cached_bytes=15_000_000_000),
        files=12,
        commit=COMMIT,
        updated=1_700_000_000.0,
        architecture="Olmo3ForCausalLM",
        dtype="bfloat16",
        path=Path("/cache") / f"models--{model_id.replace('/', '--')}",
    )
    fields.update(overrides)
    return CachedModel(**fields)


def painted(table) -> dict:
    """What the browser is sent for a search table: headers, data, and metadata."""

    return gr.Dataframe(interactive=False).postprocess(table["value"]).model_dump()


def picked(model_id: str | None, *rest) -> gr.SelectData:
    """A click on a row whose first cell is ``model_id``; None for no row at all."""

    row = [model_id, *rest] if model_id else None
    return gr.SelectData(None, {"index": [0, 0], "value": model_id, "row_value": row})
