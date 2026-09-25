"""The Images page: its controls, and how its handlers are wired to them.

The handlers live in ui.images_page. This module draws the page they act on,
inside the Blocks ui.layout.build_app() opens, and binds them to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from chatlab import charts, settings
from chatlab.model_cache import IMAGE_KIND
from chatlab.token_metrics import PROMPT_ATTENTION_SCALE
from chatlab.ui import runtime
from chatlab.ui.common import QUIET_TICK
from chatlab.ui.image_words import begin_original, build_word_comparison, start_original
from chatlab.ui.images_page import (
    NO_ATTENTION,
    NO_TRAJECTORY,
    PROMPT_STRIP_LABEL,
    draw,
    remember_committed_image_seed,
    remember_image_settings,
    remember_token,
    select_step,
    select_token,
    stop_drawing,
)
from chatlab.ui.models_page import loaded_model_badge, refresh_image_badge
from chatlab.ui.styles import pane_handle


@dataclass(frozen=True)
class ImagesPage:
    """The Images page's column and the controls its listeners read and write."""

    column: gr.Column
    run_state: gr.State
    badge_view: gr.HTML
    load_button: gr.Button
    prompt: gr.Textbox
    negative: gr.Textbox
    draw_button: gr.Button
    stop_draw_button: gr.Button
    status: gr.Markdown
    output: gr.Image
    steps: gr.Slider
    guidance: gr.Slider
    size: gr.Dropdown
    seed: gr.Number
    randomize: gr.Checkbox
    record_attention: gr.Checkbox
    word_outputs: list
    token_state: gr.State
    step: gr.Slider
    trajectory: gr.HTML
    tiles: gr.HTML
    chart: gr.HTML
    strip: gr.HighlightedText
    note: gr.Markdown
    overlay: gr.HTML


def build_images_page(saved: settings.Settings) -> ImagesPage:
    """The Images page, hidden until the nav picks it."""

    with gr.Column(
        scale=1, visible=False, elem_id="images-page"
    ) as images_page:
        image_run_state = gr.State(None)
        with gr.Row(equal_height=True, elem_id="images-columns"):
            with gr.Column(scale=3, min_width=320, elem_id="images-workspace"):
                gr.Markdown(
                    "# Images\nDraw a picture with a diffusion model, and "
                    "watch what it did while it drew.",
                    elem_id="images-hero",
                )
                # The same badge the Chat page carries, asking about the
                # same one model in memory; this one asks whether it is
                # a model that can draw.
                with gr.Row(elem_id="image-model-bar"):
                    image_badge_view = gr.HTML(
                        loaded_model_badge(kind=IMAGE_KIND),
                        elem_id="image-model-badge",
                    )
                    image_load_button = gr.Button(
                        "Choose an image model",
                        variant="primary",
                        size="sm",
                        visible=not runtime.MANAGER.image_loaded,
                        elem_id="image-load-model",
                    )
                image_prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    placeholder="A red bicycle leaning on a harbour wall at dawn",
                    elem_id="image-prompt",
                )
                image_negative = gr.Textbox(
                    value=saved.image_negative_prompt,
                    label="Negative prompt",
                    lines=1,
                    placeholder="What to steer away from — blurry, watermark…",
                    info=(
                        "What the unconditional half of every step is "
                        "prompted with. The guidance pull is measured "
                        "against it, so this changes the trace as well "
                        "as the picture."
                    ),
                )
                with gr.Row():
                    draw_button = gr.Button(
                        "Draw", variant="primary", min_width=70
                    )
                    stop_draw_button = gr.Button(
                        "Stop",
                        variant="stop",
                        visible=False,
                        elem_id="stop-drawing",
                    )
                image_status = gr.Markdown(
                    "Ready.", elem_id="image-status"
                )
                image_output = gr.Image(
                    label="Picture",
                    type="pil",
                    interactive=False,
                    show_label=False,
                    elem_id="image-output",
                )
                with gr.Accordion("Drawing settings", open=False):
                    with gr.Row():
                        image_steps = gr.Slider(
                            settings.IMAGE_STEPS_RANGE[0],
                            settings.IMAGE_STEPS_RANGE[1],
                            value=saved.image_steps,
                            step=1,
                            label="Denoising steps",
                        )
                        image_guidance = gr.Slider(
                            settings.IMAGE_GUIDANCE_RANGE[0],
                            settings.IMAGE_GUIDANCE_RANGE[1],
                            value=saved.image_guidance,
                            step=0.5,
                            label="Guidance scale",
                            info="At 1 or below there is no guidance, and no pull to measure.",
                        )
                    with gr.Row():
                        image_size = gr.Dropdown(
                            choices=[
                                (f"{size} × {size}", size)
                                for size in settings.IMAGE_SIZES
                            ],
                            value=saved.image_size,
                            label="Size",
                        )
                        image_seed = gr.Number(
                            value=saved.image_seed,
                            label="Seed",
                            precision=0,
                            # Bounded at both ends, unlike the Chat
                            # page's: torch's generator raises above
                            # its own maximum where NumPy would take
                            # any non-negative integer.
                            minimum=settings.IMAGE_SEED_RANGE[0],
                            maximum=settings.IMAGE_SEED_RANGE[1],
                        )
                        image_randomize = gr.Checkbox(
                            value=saved.image_randomize_seed,
                            label="Randomize seed",
                        )
                    image_record_attention = gr.Checkbox(
                        value=saved.image_record_attention,
                        label="Record cross-attention",
                        info=(
                            "The one reading that costs time: the "
                            "pipeline's own attention kernel never "
                            "builds the probabilities, so they are "
                            "computed again alongside. Turn it off for "
                            "the pipeline's own speed and keep the "
                            "trajectory and the guidance trace."
                        ),
                    )

                image_word_outputs = build_word_comparison()

            # The Images page carries the same handle on its own seam.
            gr.HTML(
                pane_handle("image-inspector"),
                elem_id="image-inspector-resizer",
                container=False,
                padding=False,
            )

            with gr.Column(scale=2, min_width=300, elem_id="image-inspector"):
                # Which run the readouts belong to, the step being
                # looked at, and the prompt token last clicked.
                image_token_state = gr.State(None)
                with gr.Accordion(
                    "Denoising trajectory",
                    open=True,
                    elem_classes=["inspector-section"],
                ):
                    image_step = gr.Slider(
                        1,
                        1,
                        value=1,
                        step=1,
                        label="Step",
                        interactive=False,
                        elem_id="image-step",
                        info="Scrub through the run. The token shading and the map follow.",
                    )
                    image_trajectory = gr.HTML(
                        NO_TRAJECTORY, elem_id="image-trajectory"
                    )
                with gr.Accordion(
                    "Guidance and movement",
                    open=True,
                    elem_classes=["inspector-section"],
                ):
                    image_tiles = gr.HTML(
                        charts.EMPTY_IMAGE_TILES, elem_id="image-tiles"
                    )
                    image_chart = gr.HTML(
                        charts.EMPTY_DENOISING_CHART, elem_id="image-chart"
                    )
                with gr.Accordion(
                    "Prompt attention",
                    open=True,
                    elem_classes=["inspector-section"],
                ):
                    image_strip = gr.HighlightedText(
                        label=PROMPT_STRIP_LABEL,
                        color_map=PROMPT_ATTENTION_SCALE.color_map,
                        show_legend=True,
                        combine_adjacent=False,
                        elem_id="image-prompt-strip",
                    )
                    image_note = gr.Markdown(
                        "",
                        elem_id="image-attention-note",
                        elem_classes=["scale-caption"],
                    )
                    image_overlay = gr.HTML(
                        NO_ATTENTION, elem_id="image-attention"
                    )
    return ImagesPage(
        column=images_page,
        run_state=image_run_state,
        badge_view=image_badge_view,
        load_button=image_load_button,
        prompt=image_prompt,
        negative=image_negative,
        draw_button=draw_button,
        stop_draw_button=stop_draw_button,
        status=image_status,
        output=image_output,
        steps=image_steps,
        guidance=image_guidance,
        size=image_size,
        seed=image_seed,
        randomize=image_randomize,
        record_attention=image_record_attention,
        word_outputs=image_word_outputs,
        token_state=image_token_state,
        step=image_step,
        trajectory=image_trajectory,
        tiles=image_tiles,
        chart=image_chart,
        strip=image_strip,
        note=image_note,
        overlay=image_overlay,
    )


def wire_images_page(
    demo: gr.Blocks, images: ImagesPage, nav: gr.Radio, badge_timer: gr.Timer
) -> None:
    """Drawing, scrubbing a run, the page's badge, and its saved settings."""

    # Every image handler publishes in this order; see IMAGE_OUTPUT_NAMES.
    image_outputs = [
        images.status,
        images.draw_button,
        images.stop_draw_button,
        images.seed,
        images.output,
        images.run_state,
        images.step,
        images.trajectory,
        images.tiles,
        images.chart,
        images.strip,
        images.note,
        images.overlay,
        images.token_state,
    ]
    image_inputs = [
        images.prompt,
        images.negative,
        images.steps,
        images.guidance,
        images.size,
        images.seed,
        images.randomize,
        images.record_attention,
    ]
    # Clear the previous experiment before inference, including its selected
    # word, so it cannot enqueue an obsolete comparison during a new draw.
    # Restore from the run state after completion or refusal; a refused draw
    # keeps the last original available for a fresh word selection.
    images.draw_button.click(begin_original, None, images.word_outputs, queue=False).then(
        draw, image_inputs, image_outputs,
    ).then(start_original, images.run_state, images.word_outputs)
    # Stop is not a cancel. The pipeline runs on its own thread and would
    # keep running with the generator gone, so the button sets the event
    # the run checks between steps and the generator publishes the
    # stopped run itself, trajectory and all. Cancelling it would throw
    # away the steps that had been recorded.
    images.stop_draw_button.click(stop_drawing, None, images.status)

    # The step slider moves the frame, the shading and the map together;
    # see select_step for why they cannot be allowed to disagree.
    images.step.release(
        select_step,
        [images.run_state, images.step, images.token_state],
        [images.trajectory, images.strip, images.note, images.overlay],
    )
    images.strip.select(remember_token, None, images.token_state).then(
        select_token,
        [images.run_state, images.token_state, images.step],
        images.overlay,
    )

    # The Images badge is refreshed on the same three occasions the Chat
    # one is, and for the same reasons: arriving at the page, opening it,
    # and the timer that tells a tab which did not start a load about it.
    image_badge_outputs = [images.badge_view, images.load_button]
    nav.change(refresh_image_badge, None, image_badge_outputs)
    demo.load(refresh_image_badge, None, image_badge_outputs)
    badge_timer.tick(refresh_image_badge, None, image_badge_outputs, **QUIET_TICK)

    image_settings_inputs = [
        images.negative,
        images.steps,
        images.guidance,
        images.size,
        images.seed,
        images.randomize,
        images.record_attention,
    ]
    for control in (
        images.negative,
        images.steps,
        images.guidance,
        images.size,
        images.randomize,
        images.record_attention,
    ):
        control.change(remember_image_settings, image_settings_inputs, None)
    # The seed box is written to by a finished picture, so only the
    # reader being done editing it commits what it holds; the Chat page's
    # seed follows the same rule for the same reason.
    for event in (images.seed.blur, images.seed.submit):
        event(remember_committed_image_seed, image_settings_inputs, None)
