"""The sampling and steering controls under Conversation tools, and their wiring."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from chatlab.ui.conversations import remember_branch_sampling, sampling_updates
from chatlab.ui.icons import icon_classes
from chatlab.ui.layout.common import CONVERSATION_PANE_QUEUE
from chatlab.ui.scoring import SAMPLING_LABEL_QUEUE
from chatlab.ui.settings_page import sampling_label, update_sampling_label
from chatlab.ui.steering import (
    EMPTY_STATUS,
    EXTRACT_EMPTY,
    EXTRACT_HEADERS,
    POOL_CHOICES,
    choose_layer,
    describe_layer,
    download_extracted,
    extract_vector,
    import_vector,
    remember_steering,
    remove_vector,
    steering_updates,
    use_extracted,
)
from chatlab.ui.styles import set_message_box_keys

# What wraps one sampling slider so its ↺ has somewhere to sit: the button is
# taken out of the flow and put against the slider's head, and an absolute
# position needs a positioned ancestor to measure from. The minimum width is
# the one a slider asks for rather than a column's, so the two-up rows still
# hold two across a narrow pane.
SAMPLING_FIELD = {"elem_classes": ["sampling-field"], "min_width": 160}


def sampling_reset(name: str) -> gr.Button:
    """The ↺ that puts one sampling control back to the app's default.

    The words are in the page for a screen reader to read out - "Reset
    temperature" says which slider this one belongs to, which a row of
    identical marks otherwise does not. The stylesheet shows the mark alone,
    because that is what fits in the corner beside the number box.
    """

    return gr.Button(
        f"Reset {name}",
        elem_classes=["sampling-reset", *icon_classes("rotate-ccw")],
    )


def build_steering_controls(extract_state) -> SimpleNamespace:
    """Build the steering vector controls: import one, or extract one from examples."""

    with gr.Accordion("Steering vector", open=False):
        gr.Markdown(
            "Add a vector to a model layer during this conversation. "
            "Import JSON with `model_id`, `layer` (starting at 0), "
            "and `vector` (a list of numbers). Use a vector made for "
            "the same model checkpoint."
        )
        with gr.Row():
            steering_upload = gr.UploadButton(
                "Import vector", file_types=[".json"], type="filepath"
            )
            steering_remove = gr.Button("Remove vector")
        steering_enabled = gr.Checkbox(value=False, label="Enable steering", interactive=False)
        steering_strength = gr.Slider(
            -100, 100, value=1, step=0.05, label="Steering strength", interactive=False,
            info="0 disables the addition; negative values reverse its direction.",
        )
        steering_layer = gr.Number(
            value=0, precision=0, minimum=0, label="Target layer (starting at 0)", interactive=False,
        )
        steering_status = gr.Textbox(
            value=EMPTY_STATUS, label="Vector status", interactive=False,
        )
        with gr.Accordion("Extract from examples", open=False):
            gr.Markdown(
                "Read a direction out of the model instead of importing one. "
                "Each example is run through the model once and every layer's "
                "activation is pooled to a vector; the direction is the wanted "
                "examples' mean minus the unwanted ones'. One example per line."
            )
            extract_positive = gr.Textbox(
                label="Examples of what you want",
                placeholder="One per line.",
                lines=4,
                elem_id="extract-positive",
            )
            extract_negative = gr.Textbox(
                label="Examples of the opposite",
                placeholder="One per line.",
                lines=4,
                elem_id="extract-negative",
            )
            extract_chat = gr.Checkbox(
                value=False,
                label="Read each example as a user turn",
                info=(
                    "Wraps every example in the model's chat template and "
                    "reads it at the position a reply would start from. "
                    "Models without a chat template read plain text, and say so."
                ),
            )
            extract_pool = gr.Radio(
                choices=list(POOL_CHOICES),
                value="last",
                label="Pool each example at",
            )
            extract_button = gr.Button(
                "Extract direction", variant="primary"
            )
            extract_status = gr.Markdown(
                EXTRACT_EMPTY, elem_id="extract-status"
            )
            extract_table = gr.Dataframe(
                headers=EXTRACT_HEADERS,
                datatype=["number"] * 4,
                column_widths=["16%", "28%", "28%", "28%"],
                interactive=False,
                elem_id="extract-layers",
                label="Layer by layer — click a row to choose it",
            )
            extract_layer = gr.Slider(
                0, 0, value=0, step=1,
                label="Layer to take the direction from",
                interactive=False,
            )
            with gr.Row():
                extract_apply = gr.Button(
                    "Use this layer", interactive=False, min_width=110
                )
                gr.DownloadButton(
                    "Download vector",
                    value=download_extracted,
                    inputs=[extract_state, extract_layer],
                    size="sm",
                    min_width=110,
                )

    return SimpleNamespace(
        extract_apply=extract_apply,
        extract_button=extract_button,
        extract_chat=extract_chat,
        extract_layer=extract_layer,
        extract_negative=extract_negative,
        extract_pool=extract_pool,
        extract_positive=extract_positive,
        extract_status=extract_status,
        extract_table=extract_table,
        steering_enabled=steering_enabled,
        steering_layer=steering_layer,
        steering_remove=steering_remove,
        steering_status=steering_status,
        steering_strength=steering_strength,
        steering_upload=steering_upload,
    )


def build_sampling_controls(saved) -> SimpleNamespace:
    """Build the sampling accordion: temperature, top-p, top-k, seed, response length and the top-choice skip."""

    with gr.Accordion(
        sampling_label(
            saved.temperature,
            saved.top_p,
            saved.top_k,
            saved.skip_top_below,
            saved.max_new_tokens,
        ),
        open=False,
    ) as sampling_accordion:
        # show_reset_button=False on all five,
        # and a ↺ of our own in each slider's
        # head instead, in the corner Gradio
        # draws its own in. Gradio's restores
        # the value its slider was built
        # with, which is the saved setting -
        # and the file follows every move of
        # these sliders, so it restored the
        # number already on screen and did
        # nothing whatever it was pressed.
        # Building them with the defaults to
        # give it somewhere to go would put
        # the defaults in the page a reader is
        # answered from until the load lands,
        # and would freeze the length it
        # restores at whatever the context
        # limit was at startup. Ours works out
        # what to restore when it is pressed,
        # which has neither problem. Gradio 6
        # spells the one being turned off
        # buttons=["reset"].
        with gr.Row():
            with gr.Column(**SAMPLING_FIELD):
                temperature = gr.Slider(
                    0,
                    2,
                    value=saved.temperature,
                    step=0.05,
                    label="Temperature",
                    show_reset_button=False,
                )
                temperature_reset = sampling_reset("temperature")
            with gr.Column(**SAMPLING_FIELD):
                top_p = gr.Slider(
                    0.05,
                    1,
                    value=saved.top_p,
                    step=0.01,
                    label="Top-p",
                    show_reset_button=False,
                )
                top_p_reset = sampling_reset("top-p")
        with gr.Row():
            with gr.Column(**SAMPLING_FIELD):
                top_k = gr.Slider(
                    0,
                    200,
                    value=saved.top_k,
                    step=1,
                    label="Top-k (0 disables)",
                    show_reset_button=False,
                )
                top_k_reset = sampling_reset("top-k")
            with gr.Column(**SAMPLING_FIELD):
                # The ceiling is the context
                # limit: a response cannot be
                # longer than a prompt is
                # allowed to be.
                max_new_tokens = gr.Slider(
                    1,
                    saved.prefill_token_limit,
                    value=saved.max_new_tokens,
                    step=1,
                    label="Maximum new tokens",
                    show_reset_button=False,
                )
                max_new_tokens_reset = sampling_reset(
                    "the response length"
                )
        with gr.Row():
            # Alone on its row because it is
            # the one control here that needs
            # a sentence saying what it is
            # for, and that sentence needs
            # the width.
            with gr.Column(**SAMPLING_FIELD):
                skip_top_below = gr.Slider(
                    0,
                    1,
                    value=saved.skip_top_below,
                    step=0.05,
                    label="Skip top choice below (0 disables)",
                    show_reset_button=False,
                    info=(
                        "Take the model's second choice wherever its "
                        "first holds less than this probability. Where "
                        "it is more certain than this, its choice stands."
                    ),
                )
                skip_top_below_reset = sampling_reset(
                    "the top-choice skip"
                )
        with gr.Row():
            seed = gr.Number(
                value=saved.seed,
                precision=0,
                minimum=0,
                label="Random seed",
                info="Updated after each response so you can reproduce it.",
            )
            randomize_seed = gr.Checkbox(
                value=saved.randomize_seed,
                label="New seed each response",
                info="Turn off to lock the seed and reproduce a response exactly.",
            )

    return SimpleNamespace(
        max_new_tokens=max_new_tokens,
        max_new_tokens_reset=max_new_tokens_reset,
        randomize_seed=randomize_seed,
        sampling_accordion=sampling_accordion,
        seed=seed,
        skip_top_below=skip_top_below,
        skip_top_below_reset=skip_top_below_reset,
        temperature=temperature,
        temperature_reset=temperature_reset,
        top_k=top_k,
        top_k_reset=top_k_reset,
        top_p=top_p,
        top_p_reset=top_p_reset,
    )


def wire_sampling_and_steering(ui: SimpleNamespace) -> None:
    """Wire the sampling controls and the steering vector, which belong to the conversation on screen."""

    ui.enter_sends.change(set_message_box_keys, ui.enter_sends, ui.prompt)

    # The sampling accordion wears its own values.
    #
    # On change rather than on release, even though a slider fires
    # continuously while it is dragged. Gradio dispatches release from
    # pointerup alone, so a slider moved with the arrow keys - which is
    # how it is moved without a mouse - changes its value and never
    # reports a release, and the summary would sit there describing the
    # settings as they were. always_last is what makes change affordable
    # instead: a drag's worth of them collapses to the one that matters,
    # and the label only has to be right once the slider stops.
    sampling_controls = [ui.temperature, ui.top_p, ui.top_k, ui.skip_top_below, ui.max_new_tokens]
    # In the order settings.CONVERSATION_SAMPLING names them, which is
    # what pairs each ↺ with the setting it restores.
    sampling_resets = [
        ui.temperature_reset,
        ui.top_p_reset,
        ui.top_k_reset,
        ui.skip_top_below_reset,
        ui.max_new_tokens_reset,
    ]
    for control in sampling_controls:
        control.change(
            update_sampling_label,
            sampling_controls,
            ui.sampling_accordion,
            trigger_mode="always_last",
            show_progress="hidden",
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
        # These four belong to the conversation on screen, so a move of
        # one is written into it as well as into the settings file - the
        # file being what the next new conversation starts from.
        #
        # .input rather than .change: switching conversations sets these
        # controls too, and a write from that would stamp a conversation
        # nobody had touched. input is the reader's own move, keyboard
        # included. always_last for the same reason as above: a drag is
        # one write.
        control.input(
            remember_branch_sampling,
            [ui.forks_state, *sampling_controls],
            ui.forks_state,
            trigger_mode="always_last",
            show_progress="hidden",
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

    steering_outputs = [
        ui.steering_state, ui.steering_enabled, ui.steering_strength, ui.steering_layer, ui.steering_status,
    ]
    ui.steering_upload.upload(
        import_vector, [ui.steering_upload, ui.forks_state], [ui.forks_state, *steering_outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    ui.steering_remove.click(
        remove_vector, ui.forks_state, [ui.forks_state, *steering_outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    ui.extract_button.click(
        extract_vector,
        [ui.extract_positive, ui.extract_negative, ui.extract_chat, ui.extract_pool],
        [ui.extract_state, ui.extract_table, ui.extract_layer, ui.extract_apply, ui.extract_status],
    )
    ui.extract_table.select(
        choose_layer, ui.extract_state, [ui.extract_layer, ui.extract_status]
    )
    # input rather than change: the extraction writes the layer control
    # itself, with a fuller status beside it, and a change listener would
    # fire on that write and replace the status with the shorter line.
    ui.extract_layer.input(
        describe_layer,
        [ui.extract_state, ui.extract_layer],
        ui.extract_status,
        show_progress="hidden",
    )
    ui.extract_apply.click(
        use_extracted,
        [ui.forks_state, ui.extract_state, ui.extract_layer],
        [ui.forks_state, *steering_outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    for control in (ui.steering_enabled, ui.steering_strength, ui.steering_layer):
        control.input(
            remember_steering,
            [ui.forks_state, ui.steering_state, ui.steering_enabled, ui.steering_strength, ui.steering_layer],
            [ui.forks_state, ui.steering_state, ui.steering_status],
            trigger_mode="always_last", show_progress="hidden",
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

    def brings_its_sampling(event):
        """Put the newly active conversation's sampling onto the controls.

        Every path that changes which conversation is on screen ends
        here, so the sliders describe the conversation in front of the
        reader rather than the one they just left. The label follows the
        controls, as it does when they are moved by hand.
        """

        return event.then(
            sampling_updates,
            ui.forks_state,
            sampling_controls,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            update_sampling_label,
            sampling_controls,
            ui.sampling_accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        ).then(
            steering_updates, ui.forks_state, steering_outputs,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

    # What the wiring after this reads.
    ui.brings_its_sampling = brings_its_sampling
    ui.sampling_controls = sampling_controls
    ui.sampling_resets = sampling_resets
    ui.steering_outputs = steering_outputs
