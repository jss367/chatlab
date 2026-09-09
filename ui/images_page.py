"""The Images page: drawing a picture and reading what the model did while it drew.

Same shape as the Chat page and for the same reason. A picture arrives on the
left as the steps go by, and every reading of how it arrived is on the right:
the trajectory to scrub through, the guidance pull and the latent movement per
step, and the prompt's own tokens shaded by how much of the picture each one
drove, any of which can be clicked to see where it drove it.
"""

from __future__ import annotations

import html
import logging
import random
import threading
import time

import gradio as gr

import charts
import image_runtime
import settings
from image_runtime import ImageRequest
from model_runtime import ModelBusy
from token_metrics import PROMPT_ATTENTION_SCALE, UNSCORED_LABEL
from ui import runtime
from ui.common import (
    SEED_LIMIT,
    failure_status,
    send_stop_buttons,
)
from ui.panel import event_index

logger = logging.getLogger(__name__)


# How often the page redraws while a picture is being drawn. A step of a real
# pipeline takes a good fraction of a second, so this is fast enough that
# every frame of the trajectory is seen and slow enough that the browser is
# not sent a message it has no new frame for.
DRAW_POLL_SECONDS = 0.2


NO_IMAGE_MODEL = (
    "No image model is loaded. Choose one on the **Models** page: an image "
    "model is a diffusers pipeline, which the model list marks as such."
)

TEXT_MODEL_LOADED = (
    "The model in memory is a text model, which answers with tokens rather "
    "than pictures. Load an image model on the **Models** page to draw one."
)

EMPTY_PROMPT = "Type a prompt to draw."

NO_TRAJECTORY = (
    '<div class="viz-empty">The trajectory appears one frame per step while a '
    "picture is being drawn.</div>"
)

NO_ATTENTION = (
    '<div class="viz-empty">Draw a picture, then click a prompt token to see '
    "which pixels it drove.</div>"
)

# Distinct from the Chat page's own prompt strip, which is labelled "Prompt
# tokens — click one": the two live on different pages and mean different
# things by a click, and one label for both would leave a reader looking at
# the wrong instruction.
PROMPT_STRIP_LABEL = "Prompt tokens — click one for its map"


NOTHING_TO_STOP = "Nothing is being drawn."


def stop_drawing():
    """Ask the image run that is drawing to stop after the step it is on.

    The manager owns the token, not this page: button visibility is per
    browser tab, so a second tab can press Draw while the first is still
    drawing, and a token this page cleared as each draw started would lose
    the first tab's cancellation even though the second draw is refused.
    See :meth:`model_runtime.ModelManager.stop_image_run`.
    """

    if not runtime.MANAGER.stop_image_run():
        return NOTHING_TO_STOP
    return "Stopping after this step…"


def resolve_seed(seed, randomize: bool) -> int:
    """Pick the seed for one picture, inside the range torch will accept.

    The number box constrains the value at both ends, but the API, a browser
    that ignores the constraint, a float the box rounded and a hand-edited
    settings file can all still arrive. The ceiling is torch's, which raises
    above :data:`image_runtime.MAX_SEED` where NumPy would have taken any
    non-negative integer; that is why this clamps where the Chat page's own
    seed only floors.
    """

    if randomize:
        return random.randrange(SEED_LIMIT)
    return image_runtime.usable_seed(seed)


# ------------------------------------------------------------ what is drawn


def trajectory_frame(run, step: int) -> str:
    """One frame of the trajectory with its own readings underneath.

    ``step`` counts from 1. The frame is the cheap linear preview, not a VAE
    decode, and says so once: a reader comparing it against the finished
    picture beside it deserves to know why the detail disagrees.
    """

    readings = getattr(run, "readings", None) or []
    if not readings:
        return NO_TRAJECTORY
    chosen = min(max(int(step), 1), len(readings))
    reading = readings[chosen - 1]
    facts = [f"timestep {reading.timestep:,.0f}"]
    if reading.guidance_share is not None:
        facts.append(f"guidance pull {reading.guidance_share:.3f}")
    if reading.latent_change is not None:
        facts.append(f"moved {reading.latent_change:.3f}")
    return (
        '<figure class="viz-root" id="trajectory">'
        f'<figcaption class="viz-title">Step {reading.step} of {len(readings)}'
        f'<span class="viz-sub">{html.escape(" · ".join(facts))}</span>'
        "</figcaption>"
        f'<img class="trajectory-frame" src="{reading.preview}" '
        f'alt="The latent at denoising step {reading.step}" />'
        '<div class="viz-note">A linear projection of the latent, not a full '
        "decode: layout and colour are about right and the detail is not. The "
        "finished picture is the pipeline's own decode.</div></figure>"
    )


def prompt_strip_value(run, step: int = 0):
    """The prompt's tokens shaded by their share of the picture's attention.

    Shaded against the strongest token rather than in absolute terms; see
    :data:`token_metrics.PROMPT_ATTENTION_SCALE`. The padding is not in the
    strip - it is not a token anyone wrote - and what it took is reported in
    words under it instead.
    """

    tokens = getattr(run, "tokens", None) or []
    if not tokens:
        return []
    shares = image_runtime.token_shares(run, step)
    if len(shares) != len(tokens):
        return [(token["text"], UNSCORED_LABEL) for token in tokens]
    strongest = max(shares) or 1.0
    return [
        (token["text"], PROMPT_ATTENTION_SCALE.bucket(share / strongest))
        for token, share in zip(tokens, shares)
    ]


def attention_note(run, step: int = 0) -> str:
    """One line under the strip: what the padding took, or why there is no map."""

    if getattr(run, "attention", None) is None:
        return getattr(run, "attention_note", "") or ""
    padding = image_runtime.padding_share(run, step)
    where = "over every step" if step <= 0 else f"at step {step}"
    if padding is None:
        return PROMPT_ATTENTION_SCALE.caption
    return (
        f"{PROMPT_ATTENTION_SCALE.caption} The padding past the prompt took "
        f"{padding:.0%} of it {where}: CLIP pads every prompt to a fixed "
        "length and Stable Diffusion attends to those positions like any "
        "other, so most of a short prompt's attention normally goes there."
    )


def attention_overlay(run, token: int | None, step: int = 0) -> str:
    """One token's attention map laid over the picture it drew.

    Two images stacked by the stylesheet rather than one composed here, so
    the picture underneath stays the pipeline's own pixels and the map over
    it stays a separate, translucent layer a reader can see through.
    """

    tokens = getattr(run, "tokens", None) or []
    if not tokens or token is None or not (0 <= token < len(tokens)):
        return NO_ATTENTION
    weights = image_runtime.token_map(run, token, step)
    if weights is None:
        return NO_ATTENTION
    image = getattr(run, "image", None)
    if image is None:
        return (
            '<div class="viz-empty">This run was stopped before it finished, '
            "so there is no picture to lay the map over.</div>"
        )
    width, height = image.size
    share = image_runtime.token_shares(run, step)[token]
    where = "over every step" if step <= 0 else f"at step {step}"
    text = tokens[token]["text"]
    return (
        '<figure class="viz-root" id="prompt-attention">'
        '<figcaption class="viz-title">Where '
        f"<code>{html.escape(repr(text))}</code> drove the picture"
        f'<span class="viz-sub">{share:.1%} of the attention {html.escape(where)}'
        "</span></figcaption>"
        '<div class="attention-stack">'
        f'<img src="{image_runtime.data_uri(image, quality=image_runtime.PREVIEW_QUALITY)}" '
        f'alt="The finished picture" />'
        f'<img src="{image_runtime.heat_overlay(weights, width, height)}" '
        f'alt="Cross-attention for {html.escape(text)}" />'
        "</div>"
        '<div class="viz-note">Brighter is more of that cell\'s attention. '
        "Averaged over heads and over every cross-attention layer that could "
        "be read.</div></figure>"
    )


def remember_token(event: gr.SelectData):
    """Keep which prompt token was clicked, so the map can follow the step slider.

    Through :func:`ui.panel.event_index`, which is what knows the shapes a
    strip's index arrives in: a select event carries it as a list as readily
    as a number, and ``int()`` on the list raises, which is how a click ends
    up discarded and the map never drawn. The Chat page's own strips read
    their clicks through the same helper.
    """

    try:
        return event_index(event)
    except (IndexError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- the drawing


# Every image handler publishes this tuple, in this order.
IMAGE_OUTPUT_NAMES = (
    "status",
    "draw",
    "stop",
    "seed",
    "image",
    "run",
    "step",
    "trajectory",
    "tiles",
    "chart",
    "strip",
    "note",
    "overlay",
    "token",
)


def _idle(status: str, seed=None, image=None):
    """A refusal: say why, put the buttons back, and change nothing else.

    ``seed`` and ``image`` are the two rows a refusal can still have
    something to say about - the seed a failed run had already picked, and
    the picture it did not produce - and everything after them keeps whatever
    the last run left on screen.
    """

    return (
        status,
        *send_stop_buttons(False),
        gr.skip() if seed is None else seed,
        gr.skip() if image is None else image,
        *([gr.skip()] * (len(IMAGE_OUTPUT_NAMES) - 5)),
    )


def _step_slider(readings, *, drawing: bool = False) -> dict:
    """The step slider sized to the run so far, sitting on its last step.

    Left uninteractive while a picture is being drawn: the streaming frames
    own the slider then, and a reader who dragged it would get one frame of
    the previous run before the next frame overwrote it.
    """

    steps = len(readings)
    return gr.update(
        minimum=1,
        maximum=max(1, steps),
        value=max(1, steps),
        interactive=steps > 1 and not drawing,
    )


def draw(
    prompt: str,
    negative_prompt: str,
    steps,
    guidance,
    size,
    seed,
    randomize_seed: bool,
    record_attention: bool,
):
    """Draw a picture, publishing the trajectory as it arrives.

    The pipeline blocks for the whole run, so it goes on its own thread and
    this generator polls what the run has published, exactly as a download or
    a load does. Only the cheap rows are republished per frame - the status,
    the trajectory frame and the chart - because the tokens and the maps do
    not exist until the run is over.
    """

    cleaned = prompt.strip()
    if not cleaned:
        yield _idle(EMPTY_PROMPT)
        return
    if not runtime.MANAGER.image_loaded:
        yield _idle(TEXT_MODEL_LOADED if runtime.MANAGER.loaded else NO_IMAGE_MODEL)
        return

    chosen = resolve_seed(seed, randomize_seed)
    request = ImageRequest(
        prompt=cleaned,
        negative_prompt=(negative_prompt or "").strip(),
        steps=int(steps),
        guidance_scale=float(guidance),
        seed=chosen,
        width=int(size),
        height=int(size),
        record_attention=bool(record_attention),
    )
    readings: list = []
    outcome: dict = {}

    # Reserved before the first frame is published, not after. Gradio does
    # not resume a streaming handler until the browser has been sent that
    # frame, so a run reserved afterwards leaves a network round trip in
    # which the page shows a Stop button over nothing, Stop reports that
    # nothing is drawing, and a load arriving in between replaces the
    # pipeline this handler checked. The Chat page reserves before its own
    # first frame for the same reason.
    try:
        cancel = runtime.MANAGER.start_image_run()
    except ModelBusy as error:
        yield _idle(_failure(error), seed=chosen)
        return

    def work() -> None:
        try:
            outcome["run"] = runtime.MANAGER.generate_image(
                request, on_step=readings.append, cancel=cancel
            )
        except BaseException as error:  # noqa: BLE001 - reported on the page
            outcome["error"] = error

    # Started before the first frame as well, so the reservation is never
    # held by a run that has not begun. The run gives the slot back itself
    # when it ends; this handler must not, because the pipeline is on that
    # thread and would still be drawing after the generator was closed.
    worker = threading.Thread(target=work, name="chatlab-draw", daemon=True)
    started = time.monotonic()
    try:
        worker.start()
    except BaseException:
        # work() never ran, so nothing else will give the slot back.
        runtime.MANAGER.finish_image_run()
        raise
    try:
        yield from _drawing(worker, readings, outcome, request, started, chosen)
    except GeneratorExit:
        # Gradio closed this handler - the browser went away, or something
        # cancelled it. The pipeline does not notice a closed generator, so
        # it is asked to wind down; it releases the slot as it does.
        runtime.MANAGER.stop_image_run()
        raise


def _drawing(worker, readings, outcome, request, started, chosen):
    """Publish the run's frames, from the first one to the readout."""

    yield (
        f"Drawing with seed {chosen}…",
        *send_stop_buttons(True),
        chosen,
        gr.skip(),
        gr.skip(),
        gr.skip(),
        NO_TRAJECTORY,
        charts.EMPTY_IMAGE_TILES,
        charts.EMPTY_DENOISING_CHART,
        [],
        "",
        NO_ATTENTION,
        None,
    )
    while worker.is_alive():
        worker.join(DRAW_POLL_SECONDS)
        # A copy, because the worker appends to the same list between frames
        # and a chart drawn from a list growing under it can index off its end.
        so_far = list(readings)
        yield (
            _drawing_status(so_far, request, started),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            _step_slider(so_far, drawing=True),
            trajectory_frame(_Partial(so_far), len(so_far)),
            gr.skip(),
            charts.denoising_chart(so_far),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    error = outcome.get("error")
    if error is not None:
        yield _idle(_failure(error), seed=chosen)
        return
    run = outcome["run"]
    yield (
        _finished_status(run),
        *send_stop_buttons(False),
        chosen,
        run.image,
        run,
        _step_slider(run.readings),
        trajectory_frame(run, run.steps_done),
        charts.image_summary_tiles(
            image_runtime.guidance_summary(run.readings),
            note=_run_note(run),
        ),
        charts.denoising_chart(run.readings),
        prompt_strip_value(run),
        attention_note(run),
        NO_ATTENTION,
        None,
    )


class _Partial:
    """A run in progress, for the readers that only want its readings.

    :func:`trajectory_frame` asks a run for its readings and nothing else, and
    while a picture is being drawn there is no run object yet - the manager
    returns one only when it is finished. This is that one attribute.
    """

    def __init__(self, readings: list) -> None:
        self.readings = readings


def _drawing_status(readings: list, request: ImageRequest, started: float) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    done = len(readings)
    if not done:
        return f"Encoding the prompt · {elapsed:.1f}s"
    return (
        f"Step {done} of about {request.steps} · {elapsed:.1f}s · "
        f"{elapsed / done:.1f}s per step · seed {request.seed}"
    )


def _finished_status(run) -> str:
    """What the run did, in one line, measured off the picture it produced.

    The size comes from the image rather than from the request, because a
    fixed-size pipeline takes no width or height and draws at its own
    (see :func:`image_runtime._call_arguments`); reporting what was asked
    for would have the only summary of the run claiming 512×512 for a
    256×256 result. A stopped run has no image, so it reports what it was
    asked for and says there is no picture.
    """

    if run.stopped or run.image is None:
        return (
            f"Stopped after {run.steps_done} steps. The trajectory so far was "
            "kept; there is no finished picture."
        )
    width, height = run.image.size
    return (
        f"Drawn in {run.seconds:.1f}s · {run.steps_done} steps · "
        f"{width}×{height} · seed {run.request.seed}"
    )


def _run_note(run) -> str:
    """What the summary tiles say underneath: the model, and any missing maps."""

    parts = [f"{run.model_id or 'unknown model'} · {run.seconds:.1f}s"]
    if run.attention_note:
        parts.append(run.attention_note)
    return " · ".join(parts)


def _failure(error: BaseException) -> str:
    if isinstance(error, ModelBusy):
        return failure_status("Model busy", str(error))
    logger.warning("Image run failed", exc_info=error)
    return failure_status("Could not draw the picture", str(error) or repr(error))


# -------------------------------------------------- reading a run afterwards


def select_step(run, step, token):
    """Move the whole readout to one step of the trajectory.

    The frame, the token shading and the map all follow, because a step is
    the one thing they have in common: attention moves between steps as much
    as the picture does, and a strip that stayed on the run's average while
    the frame moved would be quietly lying.
    """

    if not getattr(run, "readings", None):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    chosen = int(step)
    return (
        trajectory_frame(run, chosen),
        prompt_strip_value(run, chosen),
        attention_note(run, chosen),
        attention_overlay(run, token, chosen),
    )


def select_token(run, token, step):
    """Draw the clicked token's map over the picture."""

    if not getattr(run, "tokens", None):
        return gr.skip()
    return attention_overlay(run, token, int(step))


def remember_image_settings(
    negative_prompt, steps, guidance, size, seed, randomize, record_attention
):
    """Save the Images page's own controls, as the Chat page saves its sampling.

    The prompt itself is not saved. It is the question being asked, not a
    setting, and a file meant to be shared between machines is the wrong
    place for the last thing someone typed. Everything else in the accordion
    is saved, the attention toggle included: it is the one control that
    costs real time, so someone who turns it off has the strongest claim to
    have it stay off.
    """

    settings.update(
        image_negative_prompt=negative_prompt,
        image_steps=steps,
        image_guidance=guidance,
        image_size=size,
        image_seed=settings.seed_to_save(seed, randomize, field="image_seed"),
        image_randomize_seed=randomize,
        image_record_attention=record_attention,
    )


def remember_committed_image_seed(
    negative_prompt, steps, guidance, size, seed, randomize, record_attention
):
    """Save the Images page's controls, the seed box included, once it is edited.

    The seed box is written to by the app itself - a finished picture leaves
    the seed that drew it there - so only the reader being done editing it
    commits what it holds. The same rule the Chat page's seed follows.
    """

    settings.update(
        image_negative_prompt=negative_prompt,
        image_steps=steps,
        image_guidance=guidance,
        image_size=size,
        image_seed=seed,
        image_randomize_seed=randomize,
        image_record_attention=record_attention,
    )

