"""My Models: the list of models already in the Hugging Face cache.

Each row names a model on disk with its size, its kind and whether it would
fit now, and selecting one describes it and puts its ID in the model ID box,
which is how a reader loads something they downloaded last week without
remembering its name. The list is also where a model is redownloaded or
removed, so the refusals for a model that is loaded or still downloading are
here too: deleting files out from under the weights in memory, or under a
download still writing them, is the mistake those checks exist to prevent.
"""

from __future__ import annotations

import html
import logging
import time

import gradio as gr

from chatlab.device_memory import Fit
from chatlab.model_cache import (
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MLX_KIND,
    TEXT_KIND,
    CachedModel,
    ModelBusy,
    ModelDownloading,
    ModelLoaded,
    cache_root,
    format_bytes,
    list_cached_models,
    sort_cached_models,
)
from chatlab.ui import runtime
from chatlab.ui.common import failure_card, status_card
from chatlab.ui.memory_fit import cached_fit, cached_fits, fit_word, replacement_profile
from chatlab.ui.model_streams import describe_missing, describe_on_disk
from chatlab.ui.models_page import (
    KIND_NAMES,
    UNSUPPORTED_REASON,
    download_model,
    where_to_use,
)

logger = logging.getLogger(__name__)


# The side pane's model lists.
NO_CACHED_MODEL_SELECTED = "Select a model to see its details and put it in the model ID box."

# The My Models filter. The list is built from the whole cache, so a reader
# who never touches this sees every downloaded model; the other choices
# narrow it to the word the matching rows already wear. A snapshot ChatLab
# cannot load wears no kind, so it is only ever under All kinds.
ALL_KINDS = "all"
MODEL_KIND_FILTERS = (
    ("All kinds", ALL_KINDS),
    ("Text", TEXT_KIND),
    ("Image", IMAGE_KIND),
    ("MLX", MLX_KIND),
)
# How the summary line names a narrowed list. KIND_NAMES words the same
# kinds for a single model on the detail card, where the noun is singular.
KIND_FILTER_NAMES = {TEXT_KIND: "text", IMAGE_KIND: "image", MLX_KIND: "MLX"}


def format_timestamp(stamp: float | None) -> str:
    if stamp is None:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


def cached_model_label(entry: CachedModel, fit: Fit | None = None) -> str:
    """``org/name · 15 GB · image · fits``, flagged by kind, fit and state."""

    label = f"{entry.model_id} · {format_bytes(entry.size_bytes)}"
    if entry.status.kind == IMAGE_KIND:
        # Only the exceptions are flagged. Text models are the majority and
        # the default, so labelling every kind would put a word on every row
        # to distinguish the exceptions.
        label += " · image"
    elif entry.status.kind == MLX_KIND:
        label += " · MLX"
    elif entry.status.base_model is not None:
        label += " · LoRA"
    verdict = fit_word(fit)
    if verdict:
        label += f" · {verdict}"
    if entry.status.missing_files:
        label += " · incomplete"
    elif entry.status.unsupported:
        label += " · unsupported"
    if runtime.MANAGER.model_id == entry.model_id:
        label += " · loaded"
    return label


def describe_cached_model(entry: CachedModel, fit: Fit | None = None) -> str:
    if runtime.MANAGER.model_id == entry.model_id:
        verdict = f"**Loaded now** on {runtime.MANAGER.device_name}."
    elif entry.status.missing_files:
        verdict = (
            f"**Incomplete:** {describe_missing(entry.status)}. "
            "Use **Download and load** to fetch the rest."
        )
    elif entry.status.unsupported:
        verdict = f"**Unsupported:** {entry.status.unsupported_reason or UNSUPPORTED_REASON}"
    else:
        verdict = (
            "**Downloaded · Ready to load.** Use **Load cached** to bring it "
            f"into memory. {where_to_use(entry.status.kind)}"
        )
    facts = [("On disk", describe_on_disk(entry.status))]
    if fit is not None and fit.known:
        facts.append(("Memory", fit.note))
    if entry.files:
        facts.append(("Files", f"{entry.files} in the current snapshot"))
    facts.append(("Kind", KIND_NAMES.get(entry.status.kind, "not loadable here")))
    if entry.status.base_model is not None:
        facts.append(
            ("Adapter", f"LoRA for `{entry.status.base_model}`, merged in at full precision")
        )
    if entry.architecture:
        model_type = entry.architecture
        if entry.dtype:
            model_type += f" ({entry.dtype})"
        facts.append(("Architecture", model_type))
    if entry.commit:
        facts.append(("Revision", f"`{entry.commit[:7]}`"))
    facts.append(("Updated", format_timestamp(entry.updated)))
    if entry.path is not None:
        facts.append(("Folder", f"`{entry.path}`"))
    rows = "\n".join(f"- **{name}:** {value}" for name, value in facts)
    return f"{verdict}\n\n{rows}"


def cached_models_of_kind(models: list[CachedModel], kind: str | None) -> list[CachedModel]:
    """``models`` narrowed to one kind; All kinds and an unknown choice keep all.

    A snapshot ChatLab cannot load has no kind, so it is kept only by All
    kinds - the same rule the list's own labels follow, where an unsupported
    row says so instead of naming a kind.
    """

    if not kind or kind == ALL_KINDS:
        return list(models)
    return [entry for entry in models if entry.status.kind == kind]


def my_models_summary(
    models: list[CachedModel],
    shown: list[CachedModel] | None = None,
    kind: str | None = None,
) -> str:
    """The line above the list: what the cache holds, and what the filter hides.

    The count and the size stay about the whole cache even while a kind is
    chosen, because the disk figure is about the folder rather than about
    what is on screen. A filter that matches nothing says so, and says where
    to go instead: that is the answer a reader who came from the Images page
    with no image model downloaded needs.
    """

    root = f"`{cache_root()}`"
    if not models:
        return (
            f"No models in the Hugging Face cache yet ({root}). "
            "Find one under **Discover models**."
        )
    total = format_bytes(sum(entry.size_bytes for entry in models))
    count = f"{len(models)} model{'s' if len(models) != 1 else ''}"
    line = f"{count} · {total} on disk in {root}"
    if shown is None or len(shown) == len(models):
        return line
    named = KIND_FILTER_NAMES.get(kind, "matching")
    if not shown:
        return (
            f"No {named} models among the {line}. "
            "Find one under **Discover models**."
        )
    return f"Showing {len(shown)} {named} · {line}"


def refresh_my_models(
    selected: str | None,
    order: str | None = DEFAULT_MODEL_SORT,
    precision: str | None = None,
    kind: str | None = ALL_KINDS,
    model_id: str | None = None,
):
    """Rescan the cache; keep the selected row or typed ID, or the loaded model.

    ``precision`` is the **Weight precision** choice, which decides what each
    model would take in memory and so whether it fits. The list is repainted
    when that choice changes, which is what makes the radio the first thing
    to try when a model will not load.

    ``kind`` is the **Kind** choice, which narrows the rows to text, image or
    MLX models. A row the filter hides cannot stay selected, so a selection
    it hides is dropped the same way one removed from disk is.
    """

    everything = sort_cached_models(list_cached_models(), order)
    models = cached_models_of_kind(everything, kind)
    fits = cached_fits(models, precision)
    ids = [entry.model_id for entry in models]
    if selected not in ids:
        fallback = model_id.strip() if model_id is not None else runtime.MANAGER.model_id
        selected = fallback if fallback in ids else None
    choices = [
        (cached_model_label(entry, fits.get(entry.model_id)), entry.model_id)
        for entry in models
    ]
    if selected is None:
        detail = NO_CACHED_MODEL_SELECTED if models else ""
    else:
        entry = next(entry for entry in models if entry.model_id == selected)
        detail = describe_cached_model(entry, fits.get(selected))
    return (
        gr.update(choices=choices, value=selected),
        detail,
        my_models_summary(everything, models, kind),
    )


def select_my_model(selected: str | None, precision: str | None = None):
    """Put the chosen cached model in the ID box and describe it.

    The profile is taken for the row's own kind, the same as
    :func:`cached_fits` takes it for the list. Without that, an MLX
    conversion on a Mac with a Metal ceiling would be judged here against
    PyTorch's cap while the list and the button judge it against the
    machine, so selecting a row that reads *fits* would describe it as tight
    or unfit.
    """

    if not selected:
        return gr.skip(), NO_CACHED_MODEL_SELECTED
    entry = next(
        (entry for entry in list_cached_models() if entry.model_id == selected), None
    )
    if entry is None:
        return gr.skip(), f"`{selected}` is no longer in the cache. Press **Refresh**."
    profile = replacement_profile(entry.status.kind or TEXT_KIND)
    return (
        gr.update(value=selected),
        describe_cached_model(entry, cached_fit(entry, precision, profile)),
    )


def clear_my_model_selection():
    """Drop the My Models selection, because a typed ID names its own model.

    Runs when the reader types in the ID box, picks a search result, or selects
    the default, so ``chosen_model`` uses that ID rather than a previous row.
    It is a round trip like any other, which is the window
    ``chosen_model`` describes rather than closes.
    """

    return gr.update(value=None), NO_CACHED_MODEL_SELECTED


NO_MODEL_TO_MANAGE = "Select a model under **My Models** first."


def redownload_my_model(selected: str | None, hf_token: str):
    """Fetch whatever the chosen cached model still lacks.

    ``snapshot_download`` skips finished files and resumes partial ones, so
    for an incomplete model this fetches the rest, and for a complete one it
    checks the hub for updated files. Remove the model first to start over.

    The loaded model is refused, and so is one being loaded. A redownload
    can move ``refs/main`` to a newer revision while the old weights stay in
    memory (or are still being read), and the list marks a model loaded by
    ID alone, so it would then call the new snapshot "loaded" while every
    reply still came from the old one. An incomplete model cannot be loaded,
    so the refusal never stands in the way of finishing a download.

    ``is_loading`` rather than the name in the badge, because with two loads
    running at once only one of them is named there, and a load still waiting
    its turn for the model lock is going to read that model's files just the
    same.
    """

    if not selected:
        yield status_card("Nothing to redownload", NO_MODEL_TO_MANAGE)
        return
    if runtime.MANAGER.is_loading(selected):
        yield status_card(
            "Model in use",
            f"`{selected}` is being loaded right now. Wait for the load to finish, "
            "then **Unload** it before redownloading.",
        )
        return
    if runtime.MANAGER.model_id == selected:
        yield status_card(
            "Model in use",
            f"`{selected}` is loaded in memory. **Unload** it before redownloading, "
            "so the weights in memory and the files on disk stay the same revision.",
        )
        return
    yield from download_model(selected, hf_token)


def loaded_refusal(selected: str) -> tuple[str, str]:
    return (
        "Model in use",
        f"`{selected}` is loaded in memory. **Unload** it before removing its files.",
    )


def downloading_refusal(selected: str) -> tuple[str, str]:
    return (
        "Still downloading",
        f"`{selected}` is being downloaded. Wait for it to finish, then remove it.",
    )


def removal_refusal(selected: str | None) -> tuple[str, str] | None:
    """Why ``selected`` cannot be removed right now, as a card, or None.

    An early answer for the confirmation step only. The deletion itself goes
    through :meth:`ModelManager.remove`, which makes the same checks under
    the manager's locks; this look is not atomic with anything.
    """

    if not selected:
        return "Nothing to remove", NO_MODEL_TO_MANAGE
    if runtime.MANAGER.model_id == selected:
        return loaded_refusal(selected)
    if selected in runtime.MANAGER.active_downloads:
        return downloading_refusal(selected)
    return None


def ask_remove_my_model(selected: str | None):
    """Show the confirmation for removing the chosen model, or say why not.

    Returns the card, the confirmation's visibility, its question, and the
    model the question is about. That last value is what the confirm button
    deletes: the radio can be moved to another model in the moment between
    a click on **Remove from disk** and the response that hides the panel,
    and a deletion that read the live selection would then take the model
    the reader never agreed to lose.
    """

    hidden = gr.update(visible=False)
    refusal = removal_refusal(selected)
    if refusal is not None:
        return status_card(*refusal), hidden, "", None
    entry = next(
        (entry for entry in list_cached_models() if entry.model_id == selected), None
    )
    if entry is None:
        return (
            status_card("Nothing to remove", f"`{selected}` is no longer in the cache."),
            hidden,
            "",
            None,
        )
    question = (
        f"Remove `{selected}` ({format_bytes(entry.size_bytes)}) from disk? "
        "This deletes its folder from the Hugging Face cache and cannot be undone."
    )
    return gr.skip(), gr.update(visible=True), question, selected


def remove_my_model(pending: str | None):
    """Delete the model the confirmation named and report the space freed.

    ``pending`` is the ID :func:`ask_remove_my_model` stored, not the radio's
    current value, so the model deleted is always the one the question
    showed. The pending ID is cleared on every path.
    """

    hidden = gr.update(visible=False)
    if not pending:
        return status_card("Nothing to remove", NO_MODEL_TO_MANAGE), hidden, None
    logger.info("Removal confirmed for %s", pending)
    try:
        freed = runtime.MANAGER.remove(pending)
    except ModelLoaded:
        logger.info("Removal of %s refused: it is loaded", pending)
        return status_card(*loaded_refusal(pending)), hidden, None
    except ModelDownloading:
        logger.info("Removal of %s refused: it is downloading", pending)
        return status_card(*downloading_refusal(pending)), hidden, None
    except ModelBusy:
        logger.info("Removal of %s refused: the manager is busy", pending)
        return (
            status_card(
                "Model busy",
                f"`{pending}` cannot be removed while a model is loading, generating, "
                "scoring, or being inspected. Try again when it is idle.",
            ),
            hidden,
            None,
        )
    except FileNotFoundError:
        logger.info("Removal of %s found nothing: it is no longer cached", pending)
        return (
            status_card("Nothing to remove", f"`{pending}` is no longer in the cache."),
            hidden,
            None,
        )
    except (OSError, ValueError) as error:
        logger.warning("Could not remove %s", pending, exc_info=True)
        return (
            failure_card(
                "Could not remove model",
                f"Removing `{pending}` failed: {html.escape(str(error))}",
            ),
            hidden,
            None,
        )
    logger.info("Removed %s from the cache, freeing %s", pending, format_bytes(freed))
    return (
        status_card(
            "Model removed",
            f"Removed `{pending}` from the Hugging Face cache, "
            f"freeing {format_bytes(freed)}.",
            "success",
        ),
        hidden,
        None,
    )


def hide_remove_confirm():
    """Withdraw a pending removal: hide the question and forget its model."""

    return gr.update(visible=False), None
