"""Downloaded models, for the tests of the pages that list them.

The Models page, the chat page's switcher and the layout tests all describe
the cache the same way, so the fixture model is kept
here rather than in any one of their modules.
"""

from pathlib import Path

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

