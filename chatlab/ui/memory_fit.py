"""Whether a model would fit in memory now, for the lists that offer models.

Every list on the Models page, and the Chat page's switcher, marks each model
with a verdict - fits, tight, or won't fit - so a reader can see before
clicking which loads the machine can take. A cached model is judged from the
files on disk and a search result from the Hub's parameter count, both at the
weight precision the load would really use and against the memory the load
would really find, which counts the model it is about to replace as free.
Kept apart from the lists because the switcher asks the same question, and a
switcher that disagreed with the list beside it would offer a model the list
calls too large.
"""

from __future__ import annotations

from pathlib import Path

from chatlab.device_memory import (
    FITS,
    TIGHT,
    UNFIT,
    DeviceProfile,
    Fit,
    device_profile,
    model_fit,
)
from chatlab.hub_search import HubModel
from chatlab.model_cache import (
    IMAGE_KIND,
    MLX_KIND,
    TEXT_KIND,
    CachedModel,
    estimate_parameter_bytes,
    estimate_snapshot_bytes,
    is_adapter_snapshot,
    mlx_bits_from_id,
    mlx_snapshot_bits,
    snapshot_folder,
)
from chatlab.model_loading import QUANTIZED_BITS
from chatlab.ui import runtime


# The one word each verdict gets in a list. A model whose size or whose
# machine could not be measured gets none: the detail beside the list says
# what is not known, and a list is the wrong place to explain it.
FIT_WORDS = {FITS: "fits", TIGHT: "tight", UNFIT: "won't fit"}

# What the estimate assumes when the device has not been read yet. Both
# accelerators load half precision; a load onto the CPU converts to float32
# and takes twice as much, so a verdict given before the device is known can
# be too generous by half. It is corrected as soon as the device is read -
# see ``refresh_after_device``.
ASSUMED_DTYPE = "float16"


def fit_word(fit: Fit | None) -> str:
    """The list's own one-word verdict, or nothing where there is none."""

    return FIT_WORDS.get(fit.state, "") if fit is not None else ""


def weight_bits(precision: str | None, profile: DeviceProfile) -> int | None:
    """The bit width a load would pack linear weights into, or ``None`` for full.

    A quantized choice is honoured on Apple Metal alone, so anywhere else the
    estimate is of full weights however the radio is set - which is what the
    load itself does. A device not read yet counts as somewhere else: of the
    two ways to be wrong for the few seconds before it is read, saying a
    model is tight when 4-bit would have fitted costs a reader nothing, while
    saying it fits when the load will refuse it is the disagreement these
    verdicts exist to prevent.
    """

    if not profile.quantizes:
        return None
    return QUANTIZED_BITS.get(precision or "full")


def requested_bits(
    precision: str | None, profile: DeviceProfile, kind: str | None
) -> int | None:
    """:func:`weight_bits`, except where the radio has no say over the width.

    A pipeline is estimated whole whatever the radio says, because the Metal
    quantizer is Transformers' own and ``_load_locked`` clears the choice for
    one. An MLX repo was packed when it was converted and loads at that
    width, so the radio has nothing to add there either. A verdict that
    carried the bits anyway would put a quantized label on a figure not
    measured at one, and moving the radio would mark the loaded model as
    being about to reload when nothing would change.
    """

    if kind in (IMAGE_KIND, MLX_KIND):
        return None
    return weight_bits(precision, profile)


def packed_bits(
    snapshot: Path | None, kind: str | None, requested: int | None
) -> int | None:
    """The width a load of ``snapshot`` will really pack its linear layers into.

    ``requested`` for anything but MLX, where the radio decides as far as
    the device allows. An MLX repo answers for itself, out of the config
    ``mlx_lm.convert`` wrote: that is the width the estimate is of and the
    width the verdict has to name, because saying "full 16-bit weights"
    over a 4-bit conversion's figure would misread it by four times.
    """

    if kind == TEXT_KIND and is_adapter_snapshot(snapshot):
        # Merged into full-precision weights whatever the radio says.
        return None
    if kind != MLX_KIND or snapshot is None:
        return requested
    return mlx_snapshot_bits(snapshot)


def cached_fit(
    entry: CachedModel, precision: str | None, profile: DeviceProfile
) -> Fit | None:
    """Whether ``entry`` would load now, or ``None`` where there is nothing to judge.

    A model short of files has no size to measure until the rest arrives, an
    unsupported one will not load whatever the memory says, and the model
    already in memory has answered the question by being there - judging it
    against what is left free would call the loaded model tight.

    Being there only answers for the weights it was read as, though. **Load
    cached** on the model in memory is how a new precision is applied, so a
    reader who has moved that radio is asking about a load that has not
    happened, and the model that fits at four bits may not fit whole.
    """

    if entry.status.missing_files or entry.status.unsupported:
        return None
    kind = entry.status.kind
    # What the radio asks of this kind, which for a pipeline and an MLX repo
    # is nothing: moving it asks nothing new of either, so neither is marked
    # as about to reload.
    requested = requested_bits(precision, profile, kind)
    reloading = requested != requested_bits(runtime.MANAGER.precision, profile, kind)
    if runtime.MANAGER.model_id == entry.model_id and not reloading:
        return None
    snapshot = snapshot_folder(entry.path) if entry.path is not None else None
    if snapshot is None:
        return None
    # The size depends on the kind too: a pipeline has no checkpoint at its
    # root to measure, and an MLX repo is measured as the packed file it
    # already is. The pool is already the one for this kind - the caller
    # chose it, because choosing it here would re-read the device and
    # discard the memory the impending unload gives back.
    bits = packed_bits(snapshot, kind, requested)
    estimated = estimate_snapshot_bytes(
        snapshot, profile.dtype or ASSUMED_DTYPE, bits, kind
    )
    return model_fit(estimated, profile, bits)


def replacement_profile(kind: str = TEXT_KIND) -> DeviceProfile:
    """The machine as a model about to be loaded would find it.

    Every model a verdict is given for is one that would replace whatever is
    in memory, and a load unloads first and only then checks whether the next
    model fits. So the weights on the device now are counted as available;
    without that, a 15 GB model already loaded would have every alternative
    marked tight and the button would then load them anyway.

    ``for_kind`` does both the pool and the reclamation, because for a pool
    that is the tighter of two the unload has to be counted into each side
    before they are collapsed; doing it afterwards credits card memory to
    whichever pool happened to be smaller.
    """

    return device_profile().for_kind(kind, runtime.MANAGER.loaded_bytes)


def cached_fits(
    models: list[CachedModel], precision: str | None
) -> dict[str, Fit]:
    """The fit verdict for each of ``models``, by model ID, one reading per kind.

    A reading per kind rather than one for the whole list, because an image
    pipeline on CUDA is judged against a different pool; see
    :meth:`DeviceProfile.for_kind`. Taken lazily and kept, so a list of only
    text models still costs the one reading it always did - reading host
    memory is a subprocess, and this runs on every rescan.
    """

    profiles: dict[str, DeviceProfile] = {}
    fits = {}
    for entry in models:
        kind = entry.status.kind or TEXT_KIND
        if kind not in profiles:
            profiles[kind] = replacement_profile(kind)
        fit = cached_fit(entry, precision, profiles[kind])
        if fit is not None:
            fits[entry.model_id] = fit
    return fits


def hub_fit(
    result: HubModel,
    precision: str | None,
    profile: DeviceProfile,
    kind: str = TEXT_KIND,
) -> Fit:
    """Whether ``result`` would load now, judged from the hub's parameter count.

    The count is all a search result carries, so the estimate assumes a
    half-precision checkpoint and, for a quantized load, that the embeddings
    are packed with everything else. Both are close enough to tell a model
    that fits from one that cannot; the detail says where the figure came
    from.

    An image pipeline is judged at full precision whatever the radio says.
    The Metal quantizer is Transformers' own and ``_load_locked`` clears the
    choice for a pipeline, so honouring it here would shrink the estimate
    for a load that will not shrink and advertise a fit the load refuses.
    An MLX repo is judged at the width its name claims, again whatever the
    radio says, because that is the width it was converted to.
    """

    if not result.parameters:
        return model_fit(None, profile)
    if kind == MLX_KIND:
        # Packed already, at whatever width the converter chose; the radio
        # has no say. The width is read off the repository's name, which is
        # how mlx-community spells it, and a name that does not say is
        # judged whole rather than at a width it may not have. It is the
        # width the verdict names too, so a 4-bit conversion is not
        # described as full weights.
        bits = mlx_bits_from_id(result.model_id)
    else:
        bits = requested_bits(precision, profile, kind)
    estimated = estimate_parameter_bytes(
        result.parameters, profile.dtype or ASSUMED_DTYPE, bits
    )
    return model_fit(estimated, profile, bits)


def hub_fits(
    results: list[HubModel], precision: str | None, kind: str = TEXT_KIND
) -> dict[str, Fit]:
    """The fit verdict for each search result, by model ID, against one profile.

    ``kind`` because a search is scoped to one, and an image pipeline on CUDA
    is judged against a different pool; see :meth:`DeviceProfile.for_kind`.
    The estimate itself is from the parameter count either way, which says
    nothing about how the weights are laid out.
    """

    profile = replacement_profile(kind)
    return {
        result.model_id: hub_fit(result, precision, profile, kind)
        for result in results
    }
