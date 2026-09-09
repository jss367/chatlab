"""What happened inside a diffusion model while it drew a picture.

An image model has no tokens to colour, so the three things worth watching
are the ones a denoising loop produces on its own:

* the **trajectory**, one frame per step, which is where a composition can be
  seen locking in;
* the **guidance pull**, how far the prompt moved each step's prediction away
  from the unconditional one, which is the closest thing a diffusion model
  has to surprise;
* the **cross-attention**, which pixels each word of the prompt drove, which
  is the same question the chat page's attention view asks of earlier tokens.

None of the three needs the pipeline to be rewritten. The trajectory comes
from ``callback_on_step_end``, the guidance pull from a forward hook on the
denoiser, whose batch is the unconditional and conditional predictions side
by side, and the attention from processors that wrap the pipeline's own and
compute the probabilities its fused kernel never materializes. So whichever
pipeline a repo ships is the one that runs, prompt encoding and scheduler and
all, and this module only watches.

Heavy imports (torch, PIL) are made where they are used, so importing this
module costs nothing on a machine that never opens the Images page.
"""

from __future__ import annotations

import base64
import inspect
import io
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


# Cross-attention maps are kept at this resolution per side, whatever the
# picture's size. A map is one weight per latent cell, and the coarsest
# attention layers are 8 by 8, so anything finer than this is mostly the
# upsampling's own invention; the display scales it up to the image.
MAP_SIZE = 32


# What one cross-attention module's probabilities may cost before it is left
# unrecorded. Recording means materializing a batch by heads by queries by
# tokens matrix in single precision, and the budget is on that rather than on
# the query count, because how many queries is too many depends on the model:
# a UNet that downsamples before its first attention layer asks about a
# quarter of the pixels a UNet that does not asks about, and a resolution cap
# tight enough for the second refuses every layer of the first and leaves the
# maps empty. A quarter of a gigabyte admits every layer of every pipeline
# this has been run against, at any size they offer, and the matrix is summed
# into the maps and freed within the call.
MAX_ATTENTION_BYTES = 256 * 1024**2
_PROBABILITY_BYTES = 4


# The trajectory frames are stored at this many pixels on the longer side.
# They are approximations either way (see :func:`latent_preview`), so the
# point is to be small enough that thirty of them travel with the run.
PREVIEW_SIZE = 256


# Turning a latent into something a person can look at properly means running
# the VAE decoder, which costs about as much as several denoising steps. Every
# step of a thirty-step run would double the wait, so the frames come from a
# linear projection of the four latent channels onto red, green and blue
# instead: the same trick the other local interfaces use for their live
# previews. It gets colour, layout and contrast about right and detail wrong,
# which is the right trade for watching a composition arrive. The final image
# is the pipeline's own full decode, not one of these.
#
# One set of factors per latent space, because a Stable Diffusion latent and
# an SDXL one do not mean the same thing by their channels.
LATENT_RGB_FACTORS = {
    "sd": (
        (0.3512, 0.2297, 0.3227),
        (0.3250, 0.4974, 0.2350),
        (-0.2829, 0.1762, 0.2721),
        (-0.2120, -0.2616, -0.7177),
    ),
    "sdxl": (
        (0.3651, 0.4232, 0.4341),
        (-0.2533, -0.0042, 0.1068),
        (0.1076, 0.1111, -0.0362),
        (-0.3165, -0.2492, -0.2188),
    ),
}


# The heat map's one hue, the same blue the chat page shades attention with,
# so a reader who has learned to read one strip can read the other.
HEAT_COLOR = (42, 120, 214)


# The maps carry one row past the prompt's own tokens, holding everything the
# attention gave to the rest of the key sequence. CLIP pads every prompt to
# its full length and Stable Diffusion passes no mask, so those padding
# positions are attended to like any other and routinely take most of the
# attention there is. Without this row a token's share would be a share of
# something unnamed, and every number on the page would look mysteriously
# small; with it, the shares over the prompt and the padding sum to one.
PADDING_ROW = "padding"


class Cancelled(RuntimeError):
    """The run was stopped between steps, at the reader's request."""


@dataclass(frozen=True)
class ImageRequest:
    """One picture to draw, and how much of the drawing to record.

    ``record_attention`` is separate from the rest because it is the one
    reading that costs real time: the pipeline's fused attention kernel never
    builds the probability matrix, so recording it means computing the
    queries, keys and softmax a second time. A run with it off is the
    pipeline's own speed and still gets the trajectory and the guidance pull.
    """

    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 0
    width: int = 512
    height: int = 512
    record_attention: bool = True


@dataclass(frozen=True)
class StepReading:
    """What one denoising step did, as it can be read from outside.

    ``guidance_norm`` is the length of the vector the prompt added: the
    conditional prediction minus the unconditional one, before the guidance
    scale multiplies it. ``guidance_share`` divides that by the unconditional
    prediction's own length, which makes it comparable across steps and
    across models: 0.2 means the prompt moved the prediction by a fifth of
    what the model would have predicted from noise alone. All three are
    ``None`` when guidance is off, because then there is no second prediction
    to compare against and nothing was pulling.

    ``latent_change`` is how far this step moved the latent, relative to
    where it already was. It falls as a picture settles, so the step where it
    collapses is the step the composition stopped changing. It is ``None``
    for the first step, which has nothing before it to have moved from.
    """

    step: int
    timestep: float
    preview: str
    latent_change: float | None = None
    cond_norm: float | None = None
    uncond_norm: float | None = None
    guidance_norm: float | None = None
    guidance_share: float | None = None


@dataclass
class ImageRun:
    """A finished (or stopped) image run, and everything read out of it."""

    request: ImageRequest
    readings: list[StepReading] = field(default_factory=list)
    tokens: list[dict] = field(default_factory=list)
    """The prompt's own tokens, in the order cross-attention keys them."""

    attention: Any = None
    """``[step][token][row][column]`` weights, as a NumPy array, or ``None``.

    One map per prompt token per step, averaged over heads and over the
    cross-attention modules that were recorded, with one row on the end for
    the padding (see :data:`PADDING_ROW`). Across a step's maps every latent
    cell's weights sum to one, so a value is the share of that cell's
    attention the token took.
    """

    attention_note: str = ""
    """Why there are no maps, when there are none."""

    image: Any = None
    """The pipeline's own decode of the final latent, as a PIL image."""

    model_id: str | None = None
    load_id: str | None = None
    seconds: float = 0.0
    stopped: bool = False

    @property
    def steps_done(self) -> int:
        return len(self.readings)


# ----------------------------------------------------------- latent previews


def latent_family(pipeline) -> str:
    """Which set of latent-to-RGB factors this pipeline's latents want.

    Read from the pipeline's class name rather than its config, because what
    the factors were fitted to is the latent space of a model family, and the
    class name is what names the family. Anything unrecognized is treated as
    Stable Diffusion's, which is the space every 4-channel VAE descends from.
    """

    name = type(pipeline).__name__
    return "sdxl" if "XL" in name else "sd"


def latent_preview(latent, family: str = "sd", size: int = PREVIEW_SIZE):
    """One frame of the trajectory, as a small PIL image.

    ``latent`` is one image's latent, channels first, as the scheduler left
    it. It is divided by its own spread first: the schedulers disagree wildly
    about the scale of a latent early in a run (a k-diffusion one carries the
    noise level in the numbers themselves), and without that the first
    frames would clip to flat white and say nothing.
    """

    import torch
    from PIL import Image

    data = latent.detach().to("cpu", torch.float32)
    spread = float(data.std())
    if spread > 0 and math.isfinite(spread):
        data = data / spread
    factors = LATENT_RGB_FACTORS.get(family)
    if factors is not None and data.shape[0] == len(factors):
        rgb = torch.einsum("chw,cr->rhw", data, torch.tensor(factors))
    elif data.shape[0] >= 3:
        # An unmeasured latent space (16 channels, or a family with no
        # factors here). Its first three channels are not red, green and
        # blue, but they do move with the picture, so the frame still shows
        # a composition arriving even when its colours are meaningless.
        rgb = data[:3]
    else:
        rgb = data[:1].expand(3, -1, -1)
    pixels = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    image = Image.fromarray(array)
    height, width = array.shape[0], array.shape[1]
    longest = max(width, height) or 1
    scaled = (max(1, round(width * size / longest)), max(1, round(height * size / longest)))
    return image.resize(scaled, Image.BICUBIC)


# Trajectory frames are written as JPEG rather than PNG. Every frame travels
# with the run and a run holds one per step, and the early frames are noise,
# which PNG cannot compress at all: thirty steps came to four megabytes of
# lossless noise against a few hundred kilobytes this way. The frames are
# already an approximation of an approximation, so a compression artifact
# costs nothing that was being claimed. The heat maps stay PNG: they need
# their alpha channel.
PREVIEW_QUALITY = 80


def data_uri(image, *, quality: int | None = None) -> str:
    """A PIL image as a ``data:`` URL, ready to drop into an ``img`` tag.

    PNG unless ``quality`` names a JPEG one, in which case the image is
    flattened to RGB first, since JPEG has no alpha channel.
    """

    buffer = io.BytesIO()
    if quality is None:
        image.save(buffer, format="PNG", optimize=True)
        kind = "png"
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        kind = "jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{kind};base64,{encoded}"


def heat_overlay(weights, width: int, height: int, *, ceiling: float | None = None) -> str:
    """A cross-attention map as a translucent PNG to lay over the picture.

    ``weights`` is a square map. It is scaled to its own strongest cell, so
    a token that took a small share of the attention everywhere is still
    readable; ``ceiling`` overrides that where several maps have to be
    comparable. The square root of the share is what becomes opacity, which
    keeps a diffuse map from disappearing.
    """

    import numpy
    from PIL import Image

    array = numpy.asarray(weights, dtype="float32")
    top = float(ceiling if ceiling is not None else array.max())
    if not (top > 0 and math.isfinite(top)):
        top = 1.0
    share = numpy.clip(array / top, 0.0, 1.0)
    alpha = (numpy.sqrt(share) * 0.85 * 255).round().astype("uint8")
    rows, columns = array.shape
    rgba = numpy.zeros((rows, columns, 4), dtype="uint8")
    for channel, value in enumerate(HEAT_COLOR):
        rgba[:, :, channel] = value
    rgba[:, :, 3] = alpha
    image = Image.fromarray(rgba)
    return data_uri(image.resize((max(1, width), max(1, height)), Image.BICUBIC))


# --------------------------------------------------------- reading the model


def denoiser(pipeline):
    """The module that predicts the noise: a UNet, or a diffusion transformer."""

    for name in ("unet", "transformer"):
        module = getattr(pipeline, name, None)
        if module is not None:
            return module
    return None


class GuidanceReader:
    """Reads the classifier-free guidance split out of the denoiser's output.

    With guidance on, a pipeline runs one forward pass per step over a batch
    of two: the unconditional prediction and the conditional one, in that
    order. So the vector the prompt contributed is already there in the
    output, and a forward hook is enough to measure it. Nothing is kept but
    three numbers per step, which is what makes this affordable to do on
    every step of every run.
    """

    def __init__(self) -> None:
        self.pending: tuple[float | None, float | None, float | None] | None = None
        self.calls = 0

    def __call__(self, module, inputs, output) -> None:
        import torch

        del module, inputs
        sample = getattr(output, "sample", None)
        if sample is None and isinstance(output, (tuple, list)) and output:
            sample = output[0]
        if sample is None or not hasattr(sample, "shape") or not len(sample.shape):
            return
        self.calls += 1
        try:
            with torch.no_grad():
                flat = sample.detach().to("cpu", torch.float32).flatten(1)
                if flat.shape[0] < 2:
                    # No guidance: one prediction, and nothing to compare it
                    # against. Its own length is still worth having.
                    self.pending = (float(flat[0].norm()), None, None)
                    return
                # The halves rather than rows 0 and 1: a batch of several
                # images is the unconditional predictions followed by the
                # conditional ones, so the first of each half is the pair
                # belonging to the same picture.
                half = flat.shape[0] // 2
                uncond, cond = flat[0], flat[half]
                self.pending = (
                    float(cond.norm()),
                    float(uncond.norm()),
                    float((cond - uncond).norm()),
                )
        except RuntimeError:
            # A reading is never worth failing a run for.
            logger.debug("Could not read the guidance split", exc_info=True)
            self.pending = None

    def take(self) -> tuple[float | None, float | None, float | None]:
        """The last step's reading, cleared so a step without one reads empty."""

        reading, self.pending = self.pending, None
        return reading or (None, None, None)


class AttentionReader:
    """Accumulates per-token cross-attention maps over a step's modules.

    A UNet asks the prompt about the picture at several resolutions, once per
    cross-attention module per step. Each map is averaged over heads, put on
    one common grid, and averaged with the others, so what comes out is one
    map per prompt token per step, plus the padding row, whose cells each sum
    to one across the rows.

    ``tokens`` is the prompt's token count; the array is one row longer.
    """

    def __init__(self, tokens: int, size: int = MAP_SIZE) -> None:
        import numpy

        self.tokens = tokens
        self.rows = tokens + 1
        self.size = size
        self._total = numpy.zeros((self.rows, size, size), dtype="float32")
        self._modules = 0
        self.steps: list[Any] = []
        self.recorded = 0
        self.skipped = 0
        # Counted apart from the rest, because "every layer was too big to
        # record" is a different thing to tell a reader than "this
        # architecture spells attention in a way this cannot read".
        self.too_large = 0

    def record(self, attn, hidden_states, encoder_hidden_states) -> None:
        """Add one cross-attention module's map for the step under way."""

        import torch

        try:
            with torch.no_grad():
                probabilities = self._probabilities(
                    attn, hidden_states, encoder_hidden_states, torch
                )
                if probabilities is None:
                    self.skipped += 1
                    return
                self._total += self._grid(probabilities, torch)
                self._modules += 1
                self.recorded += 1
        except (RuntimeError, ValueError, AttributeError, IndexError):
            # Every architecture spells attention a little differently, and a
            # map is an extra: a module this cannot read is counted and left
            # out rather than allowed to end the run.
            self.skipped += 1
            logger.debug("Could not read a cross-attention module", exc_info=True)

    def _probabilities(self, attn, hidden_states, encoder_hidden_states, torch):
        """``[heads, queries, tokens]`` for the conditional half, or ``None``."""

        if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
            return None
        if getattr(attn, "spatial_norm", None) is not None:
            # Needs the timestep embedding the processor was called with,
            # which a wrapper does not see.
            return None
        scores = getattr(attn, "get_attention_scores", None)
        to_batch = getattr(attn, "head_to_batch_dim", None)
        if scores is None or to_batch is None:
            return None
        cost = (
            hidden_states.shape[0]
            * int(attn.heads)
            * hidden_states.shape[1]
            * encoder_hidden_states.shape[1]
            * _PROBABILITY_BYTES
        )
        if cost > MAX_ATTENTION_BYTES:
            self.too_large += 1
            return None
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if getattr(attn, "norm_cross", None) is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        query = to_batch(attn.to_q(hidden_states))
        key = to_batch(attn.to_k(encoder_hidden_states))
        probabilities = scores(query, key)
        heads = int(attn.heads)
        batch = probabilities.shape[0] // heads
        if batch < 1:
            return None
        # The conditional half is the last one: a guided batch is the
        # unconditional predictions and then the conditional ones, and an
        # unguided batch of one is its own conditional half.
        shaped = probabilities.reshape(batch, heads, *probabilities.shape[1:])
        return shaped[-1].to("cpu", torch.float32)

    def _grid(self, probabilities, torch):
        """One module's map, head-averaged and resampled onto the common grid.

        The keys past the prompt are summed into the last row rather than
        dropped, so the rows still account for all the attention there was.
        """

        import numpy

        averaged = probabilities.mean(dim=0)
        queries, keys = averaged.shape
        side = int(round(math.sqrt(queries)))
        if side * side != queries:
            raise ValueError(f"{queries} queries is not a square grid")
        kept = min(keys, self.tokens)
        columns = [averaged[:, position] for position in range(kept)]
        columns.append(
            averaged[:, kept:].sum(dim=1)
            if keys > kept
            else torch.zeros_like(averaged[:, 0])
        )
        maps = torch.stack(columns).reshape(len(columns), 1, side, side)
        resized = torch.nn.functional.interpolate(
            maps, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        grid = numpy.zeros((self.rows, self.size, self.size), dtype="float32")
        grid[: len(columns)] = resized.squeeze(1).numpy()
        return grid

    def flush(self) -> None:
        """Close the step: keep its averaged maps and start the next one empty."""

        import numpy

        if self._modules:
            self.steps.append(self._total / self._modules)
        else:
            self.steps.append(numpy.zeros_like(self._total))
        self._total = numpy.zeros_like(self._total)
        self._modules = 0

    def collected(self):
        """Every step's maps as one array, or ``None`` when nothing was read."""

        import numpy

        if not self.steps or not self.recorded:
            return None
        return numpy.stack(self.steps)


class RecordingProcessor:
    """An attention processor that watches another one.

    The pipeline's own processor still does the work and returns the answer,
    so nothing about the picture changes; this only asks the module for the
    probabilities alongside, and only for cross-attention, where the keys are
    the prompt.
    """

    def __init__(self, inner, reader: AttentionReader) -> None:
        self.inner = inner
        self.reader = reader

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, *args, **kwargs):
        if encoder_hidden_states is not None:
            self.reader.record(attn, hidden_states, encoder_hidden_states)
        return self.inner(
            attn, hidden_states, encoder_hidden_states, *args, **kwargs
        )


def _install_recorders(module, reader: AttentionReader):
    """Wrap every cross-attention processor; return the originals to put back.

    ``None`` when the module keeps no processor dictionary, which is how a
    pipeline whose attention cannot be watched this way says so. Only the
    ``attn2`` modules are wrapped: ``attn1`` is the picture attending to
    itself, which says nothing about the prompt.
    """

    processors = getattr(module, "attn_processors", None)
    setter = getattr(module, "set_attn_processor", None)
    if not processors or setter is None:
        return None
    if not any("attn2" in name for name in processors):
        return None
    setter(
        {
            name: RecordingProcessor(processor, reader) if "attn2" in name else processor
            for name, processor in processors.items()
        }
    )
    return processors


# ------------------------------------------------------------------ the run


def prompt_tokens(pipeline, prompt: str) -> list[dict]:
    """The prompt's tokens, in the order cross-attention keys them.

    The pipeline's own first tokenizer, because it is the one whose output
    became the keys: a two-encoder pipeline concatenates its encoders along
    the feature axis, not the sequence, so the sequence is still the first
    tokenizer's. The end-of-word marker CLIP appends is dropped from the
    text shown and the special tokens are kept, since they carry real
    attention and a reader looking at a map needs to see where it went.
    """

    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is None:
        return []
    try:
        ids = tokenizer(prompt, truncation=True).input_ids
        pieces = tokenizer.convert_ids_to_tokens(ids)
    except Exception:  # noqa: BLE001 - a tokenizer that will not answer costs the maps only
        logger.debug("Could not tokenize the prompt for attention maps", exc_info=True)
        return []
    tokens = []
    for position, piece in enumerate(pieces):
        text = piece[: -len("</w>")] if piece.endswith("</w>") else piece
        tokens.append({"index": position, "text": text, "special": text.startswith("<|")})
    return tokens


def _call_arguments(pipeline, request: ImageRequest, generator, callback) -> dict:
    """The keyword arguments this pipeline's own call signature will take.

    Pipelines differ over what they accept - a fixed-size one has no
    ``width``, an unguided one no ``guidance_scale`` - so what is offered is
    filtered by the signature rather than assumed. Anything left out simply
    goes unset and the pipeline's default stands.
    """

    offered = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or None,
        "num_inference_steps": int(request.steps),
        "guidance_scale": float(request.guidance_scale),
        "width": int(request.width),
        "height": int(request.height),
        "num_images_per_prompt": 1,
        "generator": generator,
        "output_type": "pil",
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    try:
        accepted = set(inspect.signature(pipeline.__call__).parameters)
    except (TypeError, ValueError):
        return offered
    return {name: value for name, value in offered.items() if name in accepted}


def run(
    pipeline,
    request: ImageRequest,
    *,
    cancel=None,
    on_step: Callable[[StepReading], None] | None = None,
    model_id: str | None = None,
    load_id: str | None = None,
) -> ImageRun:
    """Draw ``request`` with ``pipeline``, recording the run as it happens.

    Blocks until the picture is finished. ``on_step`` is called with each
    reading as it is taken, which is how a caller on another thread shows the
    trajectory arriving; ``cancel`` is an event checked between steps, and
    setting it raises :class:`Cancelled` out of the pipeline. A cancelled run
    keeps the readings it had taken and has no final image.
    """

    import torch

    started = time.monotonic()
    tokens = prompt_tokens(pipeline, request.prompt) if request.record_attention else []
    guidance = GuidanceReader()
    reader = AttentionReader(len(tokens)) if tokens else None
    module = denoiser(pipeline)
    family = latent_family(pipeline)
    readings: list[StepReading] = []
    previous: dict[str, Any] = {}

    def on_step_end(pipe, step, timestep, callback_kwargs):
        del pipe
        if cancel is not None and cancel.is_set():
            raise Cancelled("The image run was stopped.")
        latents = callback_kwargs.get("latents")
        if latents is not None:
            readings.append(
                _reading(step, timestep, latents, family, guidance, previous, torch)
            )
            if on_step is not None:
                on_step(readings[-1])
        if reader is not None:
            reader.flush()
        return {}

    handle = module.register_forward_hook(guidance) if module is not None else None
    originals = (
        _install_recorders(module, reader)
        if reader is not None and module is not None
        else None
    )
    if originals is None:
        reader = None
    generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
    stopped = False
    image = None
    try:
        with torch.inference_mode():
            result = pipeline(
                **_call_arguments(pipeline, request, generator, on_step_end)
            )
        images = getattr(result, "images", None)
        image = images[0] if images else None
    except Cancelled:
        stopped = True
    finally:
        if handle is not None:
            handle.remove()
        if originals is not None and module is not None:
            module.set_attn_processor(originals)
    maps = reader.collected() if reader is not None else None
    # Read after the run, not before it: whether a layer was too big to
    # record is only known once one has been offered.
    note = _attention_note(request, tokens, module, originals, reader)
    return ImageRun(
        request=request,
        readings=readings,
        # The tokens are only there to key the maps, so a run that got none
        # reports none of them: a strip of tokens with nothing to shade them
        # by and nothing behind a click would be worse than the note saying
        # why there are no maps.
        tokens=tokens if maps is not None else [],
        attention=maps,
        attention_note="" if maps is not None else note,
        image=image,
        model_id=model_id,
        load_id=load_id,
        seconds=time.monotonic() - started,
        stopped=stopped,
    )


def _reading(step, timestep, latents, family, guidance, previous, torch) -> StepReading:
    """One step's readings, from the latent it left and the hook's last look."""

    latent = latents[0] if latents.ndim == 4 else latents
    with torch.no_grad():
        flat = latent.detach().to("cpu", torch.float32).flatten()
        length = float(flat.norm())
        before = previous.get("latent")
        change = (
            float((flat - before).norm() / length)
            if before is not None and length
            else None
        )
        previous["latent"] = flat
    cond, uncond, delta = guidance.take()
    return StepReading(
        step=int(step) + 1,
        timestep=float(timestep),
        preview=data_uri(latent_preview(latent, family), quality=PREVIEW_QUALITY),
        latent_change=change,
        cond_norm=cond,
        uncond_norm=uncond,
        guidance_norm=delta,
        guidance_share=(delta / uncond) if delta is not None and uncond else None,
    )


def _attention_note(request, tokens, module, originals, reader=None) -> str:
    """Why this run has no cross-attention maps, in one sentence."""

    if not request.record_attention:
        return "Cross-attention was not recorded for this run."
    if module is None:
        return "This pipeline has no denoising module to read attention from."
    if not tokens:
        return "This pipeline has no tokenizer, so its prompt has no tokens to map."
    if originals is None:
        return (
            "This pipeline's attention is not split into prompt and picture "
            "the way cross-attention maps need, so there is nothing to map."
        )
    if reader is not None and reader.too_large and not reader.recorded:
        return (
            "Every cross-attention layer of this model was larger than the "
            f"{MAX_ATTENTION_BYTES // 1024**2} MB recording budget at this "
            "size. Draw a smaller picture to map it."
        )
    return "No cross-attention module could be read on this pipeline."


# --------------------------------------------------- reading a finished run


def row_shares(run: ImageRun, step: int = 0) -> list[float]:
    """Each map row's mean share of the attention, the padding row included.

    ``step`` counts from 1; 0 is the mean over every step, which is what the
    strip shows before a step is picked. The shares sum to one.
    """

    maps = step_maps(run, step)
    if maps is None:
        return []
    return [float(value) for value in maps.reshape(maps.shape[0], -1).mean(axis=1)]


def token_shares(run: ImageRun, step: int = 0) -> list[float]:
    """Each prompt token's share of the picture's attention, in token order."""

    return row_shares(run, step)[: len(run.tokens)]


def padding_share(run: ImageRun, step: int = 0) -> float | None:
    """The share that went to the padding past the prompt; see :data:`PADDING_ROW`."""

    shares = row_shares(run, step)
    return shares[-1] if len(shares) > len(run.tokens) else None


def step_maps(run: ImageRun, step: int = 0):
    """Every token's map for one step, or the mean over steps when ``step`` is 0."""

    attention = run.attention
    if attention is None or not len(attention):
        return None
    if step <= 0 or step > len(attention):
        return attention.mean(axis=0)
    return attention[step - 1]


def token_map(run: ImageRun, token: int, step: int = 0):
    """One row's attention map for one step, or ``None`` where there is none.

    ``token`` indexes the prompt's tokens; the padding row sits one past the
    last of them, so ``len(run.tokens)`` asks for the padding's own map.
    """

    maps = step_maps(run, step)
    if maps is None or not (0 <= token < maps.shape[0]):
        return None
    return maps[token]


def guidance_summary(readings: Sequence[StepReading]) -> dict:
    """Headline numbers for the guidance trace, for the run's summary tiles."""

    shares = [
        reading.guidance_share
        for reading in readings
        if reading.guidance_share is not None
    ]
    # Paired with their own steps rather than counted, because the first
    # step has no movement to report and any other one could be missing too.
    moves = [
        (reading.step, reading.latent_change)
        for reading in readings
        if reading.latent_change is not None
    ]
    summary: dict[str, Any] = {"step_count": len(readings)}
    if shares:
        peak = max(range(len(shares)), key=shares.__getitem__)
        summary["mean_guidance_share"] = sum(shares) / len(shares)
        summary["peak_guidance_share"] = shares[peak]
        summary["peak_guidance_step"] = readings[peak].step
    if moves:
        changes = [change for _step, change in moves]
        summary["final_latent_change"] = changes[-1]
        # Where the picture stopped moving: the first step from which every
        # step moved the latent less than a tenth of the largest move. It is
        # the one number that answers "when did the composition lock in".
        largest = max(changes)
        threshold = largest / 10 if largest else 0.0
        settled = next(
            (
                step
                for index, (step, _change) in enumerate(moves)
                if all(value <= threshold for value in changes[index:])
            ),
            None,
        )
        if settled is not None:
            summary["settled_step"] = settled
    return summary
