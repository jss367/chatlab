"""Searching the Hugging Face Hub for models ChatLab can load.

The Hub's own filters are loose: a repository tagged for text generation may
ship only GGUF files, or a config no causal-LM class reads. A search asks for
more candidates than it shows and drops the ones that would fail to load, so
every result in the table is one a download would make usable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from chatlab import mlx_runtime
from chatlab import model_cache
from chatlab.model_cache import IMAGE_KIND, MLX_KIND, TEXT_KIND


# How many hub results are shown at once.
SEARCH_LIMIT = 20
DISCOVERY_CANDIDATES = 100
HUB_SORTS = {"Popular": "downloads", "Trending": "trending_score", "New": "created_at"}

# The pipeline tags a model ChatLab can load is found under. A plain language
# model is tagged "text-generation", but one that also takes pictures or sound
# is tagged for what it reads rather than what it writes - google/gemma-4-E4B-it
# is "any-to-any" - and Transformers maps those architectures to
# AutoModelForCausalLM just the same, so they load here and belong in the
# results. Asking the hub for "text-generation" alone hid every one of them.
SEARCH_PIPELINE_TAGS = ("text-generation", "any-to-any", "image-text-to-text")

# The tag every mlx-community conversion carries, whichever library the hub
# files it under: some are "mlx", some "transformers" with this beside it.
MLX_TAG = "mlx"

# Repository tags that mark weights laid out for another runtime whatever
# else the repository holds. An MLX conversion carries the same pipeline tag
# and the same "transformers" library as the model it came from and keeps its
# weights in safetensors files, so nothing else about it says otherwise, but
# the numbers inside are quantized MLX's way and AutoModelForCausalLM cannot
# read them - lmstudio-community publishes four of gemma-4-E4B-it alone, so
# leaving them in would bury the model they came from. They have a search of
# their own under MLX_KIND, where the tag is what is asked for.
SEARCH_FOREIGN_TAGS = frozenset({MLX_TAG})

# Tags for the weight formats in FOREIGN_SUFFIXES, which mark a repository as
# foreign only where it ships nothing Transformers can read: a repository with
# both a Transformers checkpoint and a GGUF conversion of it loads here, and
# judge_snapshot reaches that same verdict from the files on disk. Offering a
# GGUF-only repository would download the whole snapshot for a load that
# cannot happen. The native tags are the two in WEIGHT_FORMATS - a repository
# whose checkpoint predates safetensors is still one from_pretrained reads.
SEARCH_FOREIGN_FORMAT_TAGS = frozenset(
    # One per suffix in FOREIGN_SUFFIXES, under the names the hub files them
    # by: a TensorFlow or Flax checkpoint is tagged for its framework rather
    # than for the .h5 or .msgpack it is written in, and _load_locked passes
    # neither from_tf nor from_flax.
    {"gguf", "onnx", "tflite", "coreml", "keras", "tf", "jax", "flax"}
)
SEARCH_NATIVE_TAGS = frozenset({"safetensors", "pytorch"})

# How many of the hub's answers to read while filling the list. The checks
# above are made here rather than by the hub, so the results are read a page
# at a time until the list is full; this bounds the reading for a search whose
# matches are nearly all embedding or speech models, where going on would page
# through the whole hub for a list that stays empty.
SEARCH_SCAN_LIMIT = 400


def foreign_to_transformers(tags: Iterable[str]) -> bool:
    """Whether a repository's tags say its weights are laid out for another runtime."""

    tags = set(tags)
    if not SEARCH_FOREIGN_TAGS.isdisjoint(tags):
        return True
    if not SEARCH_NATIVE_TAGS.isdisjoint(tags):
        return False
    return not SEARCH_FOREIGN_FORMAT_TAGS.isdisjoint(tags)


def causal_lm_model_types() -> Mapping[str, str]:
    """Transformers' map from a config's ``model_type`` to its causal-LM class.

    Imported when a search asks for it rather than at the top of the module,
    the way the rest of the heavy imports here are: it reaches torch, and a
    session that loads a model already knows the cost while one that only
    reads its own cache should not pay it.
    """

    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    return MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def loads_as_a_causal_lm(config: Mapping | None, model_types: Mapping[str, str]) -> bool:
    """Whether ``AutoModelForCausalLM`` has a class for the ``model_type`` in ``config``.

    A pipeline tag says what a model does, not which auto class loads it.
    LLaVA, BLIP-2, PaliGemma, Idefics and the Qwen-VL models all sit under
    "image-text-to-text" beside Gemma 4, but only Gemma 4 is in this map:
    the rest need their own auto class, and would download in full and then
    fail in :meth:`ModelManager._load_locked`.

    ``model_type`` is the one field asked about, because it is the one
    ``AutoConfig`` resolves by. The ``architectures`` a config also lists name
    the classes it was saved from, and reading a mapped name there as a yes
    would pass a repository ``AutoConfig`` cannot place at all. A config
    without a ``model_type``, or one the hub does not carry, is left out for
    the reason an untagged repository is: nothing about it says it would load.
    """

    if not config:
        return False
    return config.get("model_type") in model_types


@dataclass(frozen=True)
class HubModel:
    """What the hub says about one model, as much as a search result carries."""

    model_id: str
    parameters: int | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library: str | None = None
    gated: bool | str = False
    last_modified: str | None = None
    license: str | None = None
    summary: str | None = None
    download_bytes: int | None = None
    # Which kind the search that found it was scoped to, so the list that
    # holds it can be judged and described as that kind.
    kind: str = TEXT_KIND
    # What the hub sorts Trending and New by. Kept so a text search and an
    # MLX search, asked for separately, can be merged into one list in the
    # order either would have come back in.
    trending_score: int | None = None
    created_at: str | None = None
    # The repository this one was quantized from, where its card says so.
    # An MLX conversion names the Transformers model it came from this way.
    base_model: str | None = None
    # Whether ChatLab's own catalog lists it; see model_discovery.
    pick: bool = False


@dataclass(frozen=True)
class Exclusion:
    """One reason results were left out of a search, how often, and an example."""

    reason: str
    count: int
    example: str


# Why a result the hub returned is not offered, in the words the results list
# uses. Two of them are about missing information rather than a known
# incompatibility, and say so: the repository may load, but nothing the hub
# said shows that it would.
EXCLUDED_TASK = "Made for another task, such as embeddings or speech"
EXCLUDED_UNTAGGED = "No task listed on Hugging Face, so ChatLab couldn't tell what it does"
EXCLUDED_FORMAT = "Only GGUF, ONNX or other files ChatLab can't read"
EXCLUDED_MLX = "MLX conversions, which need Apple silicon with mlx-lm"
EXCLUDED_ARCHITECTURE = "A model type Transformers can't load as a chat model"
EXCLUDED_NO_CONFIG = "No model type on Hugging Face, so ChatLab couldn't check it would load"
EXCLUDED_MLX_ARCHITECTURE = "An architecture the installed mlx-lm can't run"


@dataclass(frozen=True)
class HubSearch:
    """What one search kept, what it left out and why, and how far it read."""

    results: list[HubModel]
    exclusions: tuple[Exclusion, ...] = ()
    scanned: int = 0
    # Whether the reading stopped at SEARCH_SCAN_LIMIT with more left unread.
    stopped_early: bool = False


# The library each kind of model has to be published under, which is the one
# filter the hub itself applies. Everything else about whether a result is
# loadable is decided here, result by result; see search_hub_models.
HUB_LIBRARIES = {TEXT_KIND: "transformers", IMAGE_KIND: "diffusers", MLX_KIND: "mlx"}

# The pipeline tags an image model is found under. A diffusers repository
# that writes a picture from a prompt is tagged for exactly that, so unlike
# the text tags there is no second spelling to catch.
SEARCH_IMAGE_PIPELINE_TAGS = ("text-to-image",)


def search_hub_models(
    query: str,
    hf_token: str | None = None,
    limit: int = SEARCH_LIMIT,
    kind: str = TEXT_KIND,
    order: str = "Popular",
) -> list[HubModel]:
    """The results of :func:`search_hub`, without the account of what it left out."""

    return search_hub(query, hf_token, limit, kind, order).results


def search_hub(
    query: str,
    hf_token: str | None = None,
    limit: int = SEARCH_LIMIT,
    kind: str = TEXT_KIND,
    order: str = "Popular",
) -> HubSearch:
    """Search the hub for models of one kind that ChatLab can load.

    ``kind`` is :data:`TEXT_KIND`, :data:`IMAGE_KIND` or :data:`MLX_KIND`.
    The library the hub is asked for comes from :data:`HUB_LIBRARIES`;
    everything else about whether a result is loadable is decided here,
    because the hub cannot be asked most of it.

    For a text model the ones kept are the ones whose pipeline tag is in
    :data:`SEARCH_PIPELINE_TAGS` - a model that writes text, whatever else it
    can read - less the conversions to another runtime that
    :func:`foreign_to_transformers` recognises, and less those whose
    ``model_type`` :func:`loads_as_a_causal_lm` does not accept. A repository
    the hub has no tag or config for is left out rather than guessed at; its
    ID can still be pasted into the search box.

    For an image model the tag is :data:`SEARCH_IMAGE_PIPELINE_TAGS` and the
    runtime check is the same one, which asks about the weight format rather
    than about Transformers and so reads a conversion of a diffusion model
    the same way. There is no causal-LM check to make: a pipeline has no
    ``model_type`` in that map and is built from its ``model_index.json``, so
    the library and the tag are what say it would load.

    An empty query browses the Hub. ``order`` selects popularity, trending,
    or creation date. The hub is read a page at a time until
    ``limit`` results are kept, so a query whose first matches are
    all rejected here still fills the list from further down.
    :data:`SEARCH_SCAN_LIMIT` caps how far down. Every result left out is
    counted under its reason, so the list can say what it does not show.
    """

    from huggingface_hub import HfApi

    cleaned = query.strip()
    if limit <= 0:
        return HubSearch([])
    if order not in HUB_SORTS:
        raise ValueError(f"Unknown model order: {order}")
    images = kind == IMAGE_KIND
    mlx = kind == MLX_KIND
    if mlx and not model_cache.mlx_available():
        raise RuntimeError(
            "MLX models run on Apple silicon with the mlx-lm package installed."
        )
    token = hf_token.strip() if hf_token and hf_token.strip() else None
    # No limit: the generator pages through the results, and the loop below
    # stops it once the list is full or SEARCH_SCAN_LIMIT have been read.
    found = HfApi().list_models(
        search=cleaned or None,
        filter=HUB_LIBRARIES.get(kind, HUB_LIBRARIES[TEXT_KIND]),
        sort=HUB_SORTS[order],
        expand=SEARCH_EXPAND,
        token=token,
    )
    # Only a text search needs the auto map, and reading it reaches torch,
    # so an image search does not pay for it.
    model_types = {} if images or mlx else causal_lm_model_types()
    wanted_tags = SEARCH_IMAGE_PIPELINE_TAGS if images else SEARCH_PIPELINE_TAGS
    results = []
    excluded: dict[str, list] = {}

    def leave_out(reason: str, model_id: str) -> None:
        entry = excluded.setdefault(reason, [0, model_id])
        entry[0] += 1

    scanned = 0
    stopped_early = False
    for scanned, info in enumerate(found, start=1):
        if scanned > SEARCH_SCAN_LIMIT:
            scanned -= 1
            stopped_early = True
            break
        pipeline_tag = getattr(info, "pipeline_tag", None)
        if pipeline_tag not in wanted_tags:
            leave_out(EXCLUDED_UNTAGGED if pipeline_tag is None else EXCLUDED_TASK, info.id)
            continue
        tags = getattr(info, "tags", None) or []
        config = getattr(info, "config", None)
        if mlx:
            # The library filter has already asked for MLX repos; what is
            # left to check is that mlx-lm has the architecture. The hub's
            # copy of the config drops the quantization block, so whether
            # the weights are packed is learnt from the files once they are
            # down (see judge_snapshot); an unpacked conversion loads as a
            # Transformers checkpoint, which is no worse.
            if MLX_TAG not in tags:
                leave_out(EXCLUDED_FORMAT, info.id)
                continue
            if not mlx_runtime.mlx_supports(
                (config or {}).get("model_type") if isinstance(config, Mapping) else None
            ):
                leave_out(EXCLUDED_MLX_ARCHITECTURE, info.id)
                continue
        elif foreign_to_transformers(tags):
            leave_out(
                EXCLUDED_MLX if not SEARCH_FOREIGN_TAGS.isdisjoint(tags) else EXCLUDED_FORMAT,
                info.id,
            )
            continue
        elif not images and not loads_as_a_causal_lm(config, model_types):
            leave_out(
                EXCLUDED_NO_CONFIG if not config or "model_type" not in config
                else EXCLUDED_ARCHITECTURE,
                info.id,
            )
            continue
        results.append(hub_model(info, kind))
        if len(results) == limit:
            break
    return HubSearch(
        results,
        tuple(Exclusion(reason, count, example) for reason, (count, example) in excluded.items()),
        scanned,
        stopped_early,
    )


# What a search asks the hub to say about each result.
SEARCH_EXPAND = [
    "config",
    "downloads",
    "likes",
    "pipeline_tag",
    "library_name",
    "lastModified",
    "safetensors",
    "gated",
    "tags",
    "trendingScore",
    "createdAt",
]


def hub_model(info, kind: str) -> HubModel:
    """One ``list_models`` entry as a :class:`HubModel` of ``kind``."""

    tags = getattr(info, "tags", None) or []
    safetensors = getattr(info, "safetensors", None)
    parameters = getattr(safetensors, "total", None) if safetensors else None
    licenses = [tag[len("license:") :] for tag in tags if tag.startswith("license:")]
    modified = getattr(info, "last_modified", None)
    created = getattr(info, "created_at", None)
    base = next(
        (tag[len(QUANTIZED_FROM) :] for tag in tags if tag.startswith(QUANTIZED_FROM)), None
    )
    return HubModel(
        model_id=info.id,
        parameters=parameters,
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        pipeline_tag=getattr(info, "pipeline_tag", None),
        library=getattr(info, "library_name", None),
        gated=getattr(info, "gated", False) or False,
        last_modified=modified.date().isoformat() if modified else None,
        license=licenses[0] if licenses else None,
        kind=kind,
        trending_score=getattr(info, "trending_score", None),
        created_at=created.isoformat() if created else None,
        base_model=base,
    )


# The tag the hub gives a repository whose card says it is a quantized copy
# of another; the model tree on a model's page is built from it.
QUANTIZED_FROM = "base_model:quantized:"

# The publishers whose MLX conversions are offered beside a model. The tag
# above is written from a card, and anyone's fine-tune can claim a base
# model, so a conversion is offered only from the two organisations that
# convert models as published rather than retrain them.
VERIFIED_CONVERTERS = ("mlx-community", "lmstudio-community")
MLX_VERSIONS_LIMIT = 4


def mlx_versions(model_id: str, hf_token: str | None = None) -> list[HubModel]:
    """MLX conversions of ``model_id`` from :data:`VERIFIED_CONVERTERS`, most downloaded first.

    Empty where MLX cannot run here, since nothing found could be loaded.
    """

    from huggingface_hub import HfApi

    if not model_cache.mlx_available():
        return []
    token = hf_token.strip() if hf_token and hf_token.strip() else None
    found = HfApi().list_models(
        filter=[f"{QUANTIZED_FROM}{model_id}", MLX_TAG],
        sort="downloads",
        expand=SEARCH_EXPAND,
        token=token,
        limit=40,
    )
    versions = []
    for info in found:
        if info.id.split("/", 1)[0] not in VERIFIED_CONVERTERS:
            continue
        config = getattr(info, "config", None)
        model_type = config.get("model_type") if isinstance(config, Mapping) else None
        if not mlx_runtime.mlx_supports(model_type):
            continue
        versions.append(hub_model(info, MLX_KIND))
        if len(versions) == MLX_VERSIONS_LIMIT:
            break
    return versions


# How each order compares two results from different searches. Each search
# comes back already in this order; merging needs the key.
MERGE_KEYS = {
    "Popular": lambda result: result.downloads or 0,
    "Trending": lambda result: result.trending_score or 0,
    "New": lambda result: result.created_at or "",
}


def search_hub_kinds(
    query: str,
    hf_token: str | None,
    kinds: tuple[str, ...],
    order: str = "Popular",
    limit: int = SEARCH_LIMIT,
) -> HubSearch:
    """Run :func:`search_hub` for each of ``kinds`` at once and merge the answers.

    A text search on a Mac that runs MLX is two searches, because the hub
    files Transformers models and MLX conversions under different libraries.
    The answers are merged in ``order`` and cut to ``limit``. A text search
    leaves MLX conversions out as foreign; where the MLX search ran beside
    it, those are in the list after all, so they are not reported as left out.
    """

    if len(kinds) == 1:
        return search_hub(query, hf_token, limit, kinds[0], order)
    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        searches = list(pool.map(
            lambda kind: search_hub(query, hf_token, limit, kind, order), kinds
        ))
    merged = sorted(
        (result for search in searches for result in search.results),
        key=MERGE_KEYS[order], reverse=True,
    )[:limit]
    counts: dict[str, list] = {}
    for search in searches:
        for exclusion in search.exclusions:
            if exclusion.reason == EXCLUDED_MLX and MLX_KIND in kinds:
                continue
            entry = counts.setdefault(exclusion.reason, [0, exclusion.example])
            entry[0] += exclusion.count
    return HubSearch(
        merged,
        tuple(Exclusion(reason, count, example) for reason, (count, example) in counts.items()),
        sum(search.scanned for search in searches),
        any(search.stopped_early for search in searches),
    )
