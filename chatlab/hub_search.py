"""Searching the Hugging Face Hub for models ChatLab can load.

The Hub's own filters are loose: a repository tagged for text generation may
ship only GGUF files, or a config no causal-LM class reads. A search asks for
more candidates than it shows and drops the ones that would fail to load, so
every result in the table is one a download would make usable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, replace

from chatlab import adapters
from chatlab import mlx_runtime
from chatlab import model_cache
from chatlab.model_cache import IMAGE_KIND, MLX_KIND, TEXT_KIND, WEIGHT_FORMATS


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


ADAPTER_LIBRARY = "peft"

# How many adapters are checked at once. What the hub lists says nothing of
# an adapter's type or base: a peft repository's config carries the base and
# the task but not the type, and one filed under "transformers" carries
# neither. So each adapter's own adapter_config.json is read, and its base's
# metadata after it, a request apiece; a search of the full discovery list
# can hold a hundred adapters, and one at a time that is half a minute.
ADAPTER_CHECK_WORKERS = 16
# How long one of those requests may take before its adapter is left out.
ADAPTER_CHECK_TIMEOUT = 10

# A root checkpoint beside adapter_config.json makes the repository the model
# it holds, the way model_cache.is_adapter_snapshot reads it once on disk.
CHECKPOINT_FILES = frozenset(name for pair in WEIGHT_FORMATS for name in pair)


def is_adapter_repo(filenames: Iterable[str]) -> bool:
    """Whether a repository's files are a LoRA adapter ChatLab merges into its base.

    The config and the weights at the root, where ``PeftModel.from_pretrained``
    reads them, and no checkpoint of its own beside them. An adapter kept only
    in a subfolder (one per training run, say) is left out: loading by the
    repository's ID would find nothing at the root to read.
    """

    names = set(filenames)
    return (
        adapters.ADAPTER_CONFIG in names
        and not names.isdisjoint(adapters.ADAPTER_WEIGHTS)
        and names.isdisjoint(CHECKPOINT_FILES)
    )


def fetch_adapter_config(model_id: str, token: str | None) -> dict | None:
    """The ``adapter_config.json`` at the root of ``model_id`` on the hub, or ``None``.

    Read off the hub directly rather than through ``hf_hub_download``: a file
    in the hub cache would put the adapter in the local inventory as a
    download begun, which is why model_repository keeps its config probe out
    of the cache too. ``None`` for any failure, a gated repository included -
    a search leaves out what it cannot read rather than guess at it.
    """

    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    try:
        response = get_session().get(
            hf_hub_url(model_id, adapters.ADAPTER_CONFIG),
            headers=build_hf_headers(token=token),
            timeout=ADAPTER_CHECK_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        config = response.json()
    except (httpx.HTTPError, OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def base_access(
    api, base: str, revision: str | None, token: str | None, model_types: Mapping[str, str]
) -> bool | str | None:
    """How the base an adapter names is reached, or ``None`` if it would not load.

    The same checks a text result gets, made on the base's metadata: an
    adapter merges into whatever ``AutoModelForCausalLM`` builds from the
    base, so a base that would not be listed itself cannot carry one. Its
    files are read too, and it needs a checkpoint at the root, single or
    sharded (:data:`CHECKPOINT_FILES`): tags and config alone pass a base
    that is itself an adapter repository, which the Models page refuses to
    stack once both are down, and one whose weights sit elsewhere, which the
    download would leave missing its model files.

    What comes back for a base that loads is its ``gated`` value - ``False``,
    ``"auto"`` or ``"manual"`` - since the hub shows anyone a gated
    repository's metadata but the download of its weights needs the terms
    accepted and a token, and a public adapter does not say so itself.
    """

    from huggingface_hub.errors import HfHubHTTPError
    import httpx

    try:
        info = api.model_info(
            base, revision=revision, expand=["config", "gated", "siblings", "tags"],
            token=token, timeout=ADAPTER_CHECK_TIMEOUT,
        )
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError):
        return None
    if foreign_to_transformers(getattr(info, "tags", None) or []):
        return None
    if not loads_as_a_causal_lm(getattr(info, "config", None), model_types):
        return None
    filenames = {
        getattr(sibling, "rfilename", None)
        for sibling in getattr(info, "siblings", None) or []
    }
    if filenames.isdisjoint(CHECKPOINT_FILES):
        return None
    return getattr(info, "gated", False) or False


def confirm_adapters(
    batch: dict[str, tuple[tuple, HubModel]],
    api,
    token: str | None,
    model_types: Mapping[str, str],
    bases: dict[tuple[str, str | None], bool | str | None],
    pool: Executor,
) -> dict[str, tuple[tuple, HubModel]]:
    """``batch`` less the adapters ChatLab would turn away once they were down.

    An adapter is kept when its config passes :func:`adapters.adapter_problem`
    - a LoRA, for generating text, on a base named by Hub ID - and that base
    passes :func:`base_access`; those are the checks the Models page makes
    after the download, and making them here is what keeps an IA3 adapter or
    one trained on a local path from being offered as one to download and
    load. A kept adapter carries the base its config names, the Unsloth swap
    made, which is the one the download fetches, and whether that base is
    gated. ``bases`` remembers each
    base's verdict for the rest of the search, so twenty organisms trained on
    one model ask about it once.
    """

    found = [result for _, result in batch.values() if result.adapter]
    if not found:
        return batch
    configs = pool.map(lambda result: fetch_adapter_config(result.model_id, token), found)
    targets = {}
    for result, config in zip(found, configs):
        if config is not None and adapters.adapter_problem(config, result.model_id) is None:
            targets[result.model_id] = (
                adapters.base_model_id(config), adapters.base_revision(config)
            )
    unasked = list(dict.fromkeys(t for t in targets.values() if t not in bases))
    verdicts = pool.map(
        lambda target: base_access(api, *target, token, model_types), unasked
    )
    bases.update(zip(unasked, verdicts))
    confirmed = {}
    for model_id, (key, result) in batch.items():
        if result.adapter:
            target = targets.get(model_id)
            if target is None or bases[target] is None:
                continue
            result = replace(result, base_model=target[0], base_gated=bases[target])
        confirmed[model_id] = (key, result)
    return confirmed


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
    adapter: bool = False
    base_gated: bool | str = False


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
EXCLUDED_ADAPTER = "An adapter or its base that ChatLab cannot load"
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
    ID can still be pasted into the search box. Text searches also read PEFT
    repositories, identifying adapters by their root files and confirming
    each adapter's configuration and base checkpoint before offering it.

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
    :data:`SEARCH_SCAN_LIMIT` caps each library's scan. Repositories returned
    by both library filters are counted once. Every result left out is
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
    sort = HUB_SORTS[order]
    # A text search asks for adapters as well as models; see ADAPTER_LIBRARY.
    libraries = [HUB_LIBRARIES.get(kind, HUB_LIBRARIES[TEXT_KIND])]
    if kind == TEXT_KIND:
        libraries.append(ADAPTER_LIBRARY)
    # Only a text search needs the auto map, and reading it reaches torch,
    # so an image search does not pay for it.
    model_types = {} if images or mlx else causal_lm_model_types()
    api = HfApi()
    kept: dict[str, tuple[tuple, HubModel]] = {}
    bases: dict[tuple[str, str | None], bool | str | None] = {}
    excluded: dict[str, list] = {}
    scanned = 0
    stopped_early = False
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=ADAPTER_CHECK_WORKERS) as pool:
        for library in libraries:
            count, stopped = read_library(
                library, api, cleaned, sort, token, kind, model_types, limit,
                kept, bases, pool, excluded, seen,
            )
            scanned += count
            stopped_early |= stopped
    ranked = list(kept.values())
    if len(libraries) > 1:
        # Each answer is in the hub's order already; this interleaves the two
        # by the same key. The top of the union is the top of the two tops,
        # so reading each to the limit is enough.
        ranked.sort(key=lambda pair: pair[0], reverse=True)
    return HubSearch(
        [result for _, result in ranked[:limit]],
        tuple(Exclusion(reason, count, example) for reason, (count, example) in excluded.items()),
        scanned, stopped_early,
    )


# What a search asks the hub to say about each result.
SEARCH_EXPAND = [
    "siblings",
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


def read_library(
    library: str,
    api,
    cleaned: str,
    sort: str,
    token: str | None,
    kind: str,
    model_types: Mapping[str, str],
    limit: int,
    kept: dict[str, tuple[tuple, HubModel]],
    bases: dict[tuple[str, str | None], bool | str | None],
    pool: Executor,
    excluded: dict[str, list],
    seen: set[str],
) -> tuple[int, bool]:
    """Add up to ``limit`` of the hub's answers under ``library`` to ``kept``.

    Read in batches, each as long as the list still has room for: what the
    files and tags pass goes into a batch, :func:`confirm_adapters` checks the
    batch's adapters all at once, and whatever it drops is made up from the
    next batch, until the list is full, the answer runs out, or
    :data:`SEARCH_SCAN_LIMIT` have been read.
    """

    # No limit: the generator pages through the results, and the loop below
    # stops it.
    found = iter(
        api.list_models(
            search=cleaned or None,
            filter=library,
            sort=sort,
            expand=SEARCH_EXPAND,
            token=token,
        )
    )
    added = scanned = observed = 0
    last = False
    while added < limit and not last:
        batch: dict[str, tuple[tuple, HubModel]] = {}
        for info in found:
            scanned += 1
            if info.id in seen:
                if scanned == SEARCH_SCAN_LIMIT:
                    last = True
                    break
                continue
            seen.add(info.id)
            observed += 1
            result = loadable_result(info, kind, model_types)
            if result is None:
                reason = exclusion_reason(info, kind, model_types)
                entry = excluded.setdefault(reason, [0, info.id])
                entry[0] += 1
            # A repository filed under one library can carry the other's tag,
            # and the hub's filter matches tags, so both answers can hold it.
            if result is not None and result.model_id not in kept:
                batch[result.model_id] = (sort_value(info, sort), result)
            if scanned == SEARCH_SCAN_LIMIT:
                last = True
                break
            if added + len(batch) == limit:
                break
        else:
            # The answer ran out.
            last = True
        confirmed = confirm_adapters(batch, api, token, model_types, bases, pool)
        for model_id in batch.keys() - confirmed.keys():
            entry = excluded.setdefault(EXCLUDED_ADAPTER, [0, model_id])
            entry[0] += 1
        kept.update(confirmed)
        added += len(confirmed)
    return observed, scanned == SEARCH_SCAN_LIMIT and next(found, None) is not None


def sort_value(info, sort: str) -> tuple:
    """What the hub sorted ``info`` by, comparable across answers and None-safe.

    The sort names in :data:`HUB_SORTS` are the attribute names the hub's
    client gives the same fields.
    """

    value = getattr(info, sort, None)
    return (value is not None, value)


def loadable_result(info, kind: str, model_types: Mapping[str, str]) -> HubModel | None:
    """``info`` as a search result, or ``None`` where ChatLab could not load it.

    See :func:`search_hub_models` for what is checked for each kind.
    """

    images = kind == IMAGE_KIND
    mlx = kind == MLX_KIND
    wanted_tags = SEARCH_IMAGE_PIPELINE_TAGS if images else SEARCH_PIPELINE_TAGS
    pipeline_tag = getattr(info, "pipeline_tag", None)
    tags = getattr(info, "tags", None) or []
    config = getattr(info, "config", None)
    filenames = (
        getattr(sibling, "rfilename", None)
        for sibling in getattr(info, "siblings", None) or []
    )
    adapter = kind == TEXT_KIND and is_adapter_repo(filenames)
    # An adapter's card often names no pipeline, and its config is the
    # tokenizer's alone: the model it changes is the base's to describe, and
    # the adapter's own config is read and judged after this, a batch at a
    # time (confirm_adapters). A pipeline it does name still has to be one
    # that writes text.
    if pipeline_tag not in wanted_tags and not (adapter and pipeline_tag is None):
        return None
    if mlx:
        # The library filter has already asked for MLX repos; what is
        # left to check is that mlx-lm has the architecture. The hub's
        # copy of the config drops the quantization block, so whether
        # the weights are packed is learnt from the files once they are
        # down (see judge_snapshot); an unpacked conversion loads as a
        # Transformers checkpoint, which is no worse.
        if MLX_TAG not in tags or not mlx_runtime.mlx_supports(
            (config or {}).get("model_type") if isinstance(config, Mapping) else None
        ):
            return None
    elif foreign_to_transformers(tags):
        return None
    elif not images and not adapter and not loads_as_a_causal_lm(config, model_types):
        return None
    safetensors = getattr(info, "safetensors", None)
    # An adapter's count is its own low-rank matrices, a sliver of the model
    # it loads as; judging the fit by it would call any base a fit.
    parameters = (
        None if adapter else getattr(safetensors, "total", None) if safetensors else None
    )
    licenses = [tag[len("license:") :] for tag in tags if tag.startswith("license:")]
    modified = getattr(info, "last_modified", None)
    return HubModel(
        model_id=info.id,
        parameters=parameters,
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        pipeline_tag=pipeline_tag,
        library=getattr(info, "library_name", None),
        gated=getattr(info, "gated", False) or False,
        last_modified=modified.date().isoformat() if modified else None,
        license=licenses[0] if licenses else None,
        kind=kind,
        adapter=adapter,
        trending_score=getattr(info, "trending_score", None),
        created_at=(getattr(info, "created_at", None).isoformat() if getattr(info, "created_at", None) else None),
        base_model=next((tag[len(QUANTIZED_FROM):] for tag in tags if tag.startswith(QUANTIZED_FROM)), None),
    )


def exclusion_reason(info, kind, model_types):
    tags = getattr(info, "tags", None) or []
    config = getattr(info, "config", None)
    pipeline = getattr(info, "pipeline_tag", None)
    wanted = SEARCH_IMAGE_PIPELINE_TAGS if kind == IMAGE_KIND else SEARCH_PIPELINE_TAGS
    if pipeline not in wanted:
        return EXCLUDED_UNTAGGED if pipeline is None else EXCLUDED_TASK
    if kind == MLX_KIND:
        return EXCLUDED_FORMAT if MLX_TAG not in tags else EXCLUDED_MLX_ARCHITECTURE
    if foreign_to_transformers(tags):
        return EXCLUDED_MLX if not SEARCH_FOREIGN_TAGS.isdisjoint(tags) else EXCLUDED_FORMAT
    return EXCLUDED_NO_CONFIG if not isinstance(config, Mapping) or not config.get("model_type") else EXCLUDED_ARCHITECTURE
