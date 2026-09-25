"""The Chat page: its tabs, the inspector beside them, and their listeners.

The handlers live in the modules named for what they do - ui.generation,
ui.scoring, ui.prompts, ui.compare, ui.inspection and the rest. This module
draws the page they act on, inside the Blocks ui.layout.build_app() opens,
and binds the ones that belong to the page alone. The conversation handlers,
which also answer to the conversations pane, are bound in ui.layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

from chatlab import charts, settings
from chatlab.compare import (
    CONFIGURATION_HEADERS,
    EMPTY_SLOT as COMPARE_SLOT_EMPTY,
    DIVERGENCE_HEADERS,
    GAP_CAPTION,
    GAP_COLORS,
    MEASUREMENT,
    REPLY,
)
from chatlab.conversation import new_forks
from chatlab.token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
from chatlab.trace_export import write_trace_export
from chatlab.ui import runtime, experiments, experiment_compare
from chatlab.ui.activation_patching import build as build_activation_patching
from chatlab.ui.common import (
    CONVERSATION_PANE_QUEUE,
    NO_TOKEN_SELECTED,
    QUIET_TICK,
    STOP_LABEL,
    TRANSCRIPT_LABEL,
)
from chatlab.ui.compare import (
    COMPARE_EMPTY,
    clear_slots,
    download_comparison,
    fill_slot,
    mode_controls,
    render as render_comparison,
    stop_comparison,
)
from chatlab.ui.conversations import remember_branch_sampling, remember_transcript_message
from chatlab.ui.fork_tree import render_fork_tree
from chatlab.ui.icons import icon_classes
from chatlab.ui.inspection import (
    INSPECT_HINT,
    INSPECTION_CONTROLS,
    change_lens_mode,
    change_pinned_token,
    import_jacobian_lens,
    inspect_layers,
    remember_inspect_target,
    render_attention,
    render_kv_cache,
    reset_inspection,
)
from chatlab.ui.models_page import (
    BADGE_REFRESH_SECONDS,
    loaded_model_badge,
    refresh_model_badge,
    refresh_model_switch,
    refresh_stale_model_switch,
)
from chatlab.ui.panel import (
    choose_alternative,
    inspect_token,
    recolor,
    remember_strip_selection,
    select_transcript_token,
    show_token_view,
)
from chatlab.ui.prompts import (
    BATCH_HEADERS,
    PROMPT_COUNT_HINT,
    count_prompts,
    load_prompt_file,
    run_prompts,
    stop_batch,
)
from chatlab.ui.scoring import (
    SAMPLING_LABEL_QUEUE,
    SCORE_BUDGET_QUEUE,
    SCORE_COUNT_HINT,
    recover_score_budget,
    score_text,
    score_token_count,
)
from chatlab.ui.settings_page import refresh_thinking_mode, sampling_label, update_sampling_label
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
    use_extracted,
)
from chatlab.ui.styles import pane_handle, message_box_settings
from chatlab.ui.token_edit import close_token_editor, open_token_editor
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    MENU_STRIP_CLASS,
    prompt_menu_payload,
    token_menu_payload,
)

if TYPE_CHECKING:
    from chatlab.ui.layout import SharedState


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


@dataclass(frozen=True)
class ModelBar:
    """The badge above the tabs, the switcher beside it, and their timer."""

    badge_view: gr.HTML
    switch: gr.Dropdown
    default_model_button: gr.Button
    switch_stamp: gr.State
    badge_timer: gr.Timer

    @property
    def badge_outputs(self) -> list:
        """What refresh_model_badge() repaints."""

        return [self.badge_view, self.default_model_button]

    @property
    def switch_outputs(self) -> list:
        """The switcher, and the stamp of what it was last drawn from."""

        return [self.switch, self.switch_stamp]


@dataclass(frozen=True)
class ChatTab:
    """The conversation, its two views, the message box and the file controls."""

    token_view: gr.Radio
    chatbot: gr.Chatbot
    token_strip: gr.HighlightedText
    token_editor: gr.Group
    token_edit_text: gr.Textbox
    token_edit_save: gr.Button
    token_edit_cancel: gr.Button
    prompt: gr.Textbox
    send_button: gr.Button
    stop_button: gr.Button
    retry_button: gr.Button
    next_token_button: gr.Button
    undo_button: gr.Button
    generation_status: gr.Markdown
    save_button: gr.Button
    load_upload: gr.UploadButton
    saved_file: gr.File


@dataclass(frozen=True)
class Sampling:
    """The sampling controls under the message box, and their resets."""

    accordion: gr.Accordion
    temperature: gr.Slider
    temperature_reset: gr.Button
    top_p: gr.Slider
    top_p_reset: gr.Button
    top_k: gr.Slider
    top_k_reset: gr.Button
    max_new_tokens: gr.Slider
    max_new_tokens_reset: gr.Button
    skip_top_below: gr.Slider
    skip_top_below_reset: gr.Button
    seed: gr.Number
    randomize_seed: gr.Checkbox

    @property
    def controls(self) -> list:
        """The five that belong to a conversation, in CONVERSATION_SAMPLING order."""

        return [self.temperature, self.top_p, self.top_k, self.skip_top_below, self.max_new_tokens]

    @property
    def resets(self) -> list:
        """Each control's ↺.

        In the order settings.CONVERSATION_SAMPLING names them, which is
        what pairs each ↺ with the setting it restores.
        """

        return [
            self.temperature_reset,
            self.top_p_reset,
            self.top_k_reset,
            self.skip_top_below_reset,
            self.max_new_tokens_reset,
        ]


@dataclass(frozen=True)
class Steering:
    """The steering vector's controls, and the extraction that can fill it."""

    # The vector in force, held with the shared states.
    state: gr.State
    upload: gr.UploadButton
    remove: gr.Button
    enabled: gr.Checkbox
    strength: gr.Slider
    layer: gr.Number
    status: gr.Textbox
    extract_positive: gr.Textbox
    extract_negative: gr.Textbox
    extract_chat: gr.Checkbox
    extract_pool: gr.Radio
    extract_button: gr.Button
    extract_status: gr.Markdown
    extract_table: gr.Dataframe
    extract_layer: gr.Slider
    extract_apply: gr.Button

    @property
    def outputs(self) -> list:
        """What every steering handler publishes, the vector first."""

        return [self.state, self.enabled, self.strength, self.layer, self.status]

    @property
    def inputs(self) -> list:
        """What a request reads the vector from."""

        return [self.state, self.enabled, self.strength, self.layer]


@dataclass(frozen=True)
class ScoreTab:
    """Scoring text the model did not write."""

    # Which load the scored token count on screen was counted against,
    # held with the shared states.
    budget_load: gr.State
    context: gr.Textbox
    use_chat_template: gr.Checkbox
    text: gr.Textbox
    budget: gr.Markdown
    button: gr.Button
    status: gr.Markdown
    strip: gr.HighlightedText

    @property
    def budget_inputs(self) -> list:
        """What the scored token count is counted from."""

        return [self.context, self.text, self.use_chat_template]

    @property
    def budget_outputs(self) -> list:
        """The count, and the load it was counted against.

        The count travels with the load it was counted against; see
        recover_score_budget for what that is for.
        """

        return [self.budget, self.budget_load]


@dataclass(frozen=True)
class PromptsTab:
    """Running a list of prompts as a batch."""

    box: gr.Textbox
    count: gr.Markdown
    run_button: gr.Button
    stop_button: gr.Button
    upload: gr.UploadButton
    batch_status: gr.Markdown
    batch_results: gr.Dataframe
    batch_files: gr.File


@dataclass(frozen=True)
class ForkTree:
    """The tree of forks, and the bridge its script writes a click into."""

    action: gr.Textbox
    view: gr.HTML
    comparison: gr.HTML


@dataclass(frozen=True)
class CompareTab:
    """Two runs side by side."""

    pair_conditions: list
    pair_button: gr.Button
    pair_load_status: gr.HTML
    mode: gr.Radio
    prompt: gr.Textbox
    template: gr.Checkbox
    text: gr.Textbox
    run_a: gr.Button
    run_b: gr.Button
    stop: gr.Button
    clear: gr.Button
    status: gr.Markdown
    tiles: gr.HTML
    headline: gr.Markdown
    a_heading: gr.Markdown
    a_strip: gr.HighlightedText
    b_heading: gr.Markdown
    b_strip: gr.HighlightedText
    chart: gr.HTML
    settings: gr.Dataframe
    rows: gr.Dataframe
    difference_position: gr.State
    next_difference: gr.Button
    difference_detail: gr.Markdown


@dataclass(frozen=True)
class Inspector:
    """The readings pane beside the tabs, but for its layer inspection."""

    pane: gr.Column
    color_scale: gr.Dropdown
    scale_caption: gr.Markdown
    token_detail: gr.Markdown
    alternatives: gr.Dataframe
    branch_button: gr.Button
    branch_text: gr.Textbox
    branch_text_button: gr.Button
    summary_panel: gr.HTML
    surprise_panel: gr.HTML
    prompt_note: gr.Markdown
    prompt_strip: gr.HighlightedText


@dataclass(frozen=True)
class Layers:
    """Layers and attention: the lens, the attention map and the cache view."""

    lens_mode: gr.Radio
    imported_lens: gr.State
    inspection_session: gr.State
    jacobian_controls: gr.Column
    lens_setup: gr.Accordion
    lens_repository: gr.Textbox
    lens_filename: gr.Textbox
    lens_file: gr.File
    fitted_model_id: gr.Textbox
    import_lens_button: gr.Button
    import_lens_status: gr.Markdown
    pinned_concept: gr.Textbox
    pinned_token_id: gr.Textbox
    inspect_button: gr.Button
    inspect_status: gr.Markdown
    lens_panel: gr.HTML
    attention_layer: gr.Slider
    attention_panel: gr.HTML
    kv_layer: gr.Slider
    kv_metric: gr.Radio
    kv_panel: gr.HTML


@dataclass(frozen=True)
class ChatPage:
    """The Chat page's column and everything drawn in it."""

    column: gr.Column
    bar: ModelBar
    chat: ChatTab
    sampling: Sampling
    steering: Steering
    score: ScoreTab
    prompts: PromptsTab
    tree: ForkTree
    compare: CompareTab
    tabs: gr.Tabs
    experiments_view: object
    inspector_resizer: gr.HTML
    inspector: Inspector
    layers: Layers


def build_chat_page(saved: settings.Settings, states: SharedState) -> ChatPage:
    """The Chat page, which the app opens on."""

    with gr.Column(scale=1, elem_id="chat-page") as chat_page:
        # Keep the header in the chat column so the token panel can
        # start at the top of the page beside it.
        with gr.Row(equal_height=True, elem_id="chat-columns"):
            with gr.Column(scale=3, min_width=320, elem_id="chat-workspace"):
                gr.Markdown(
                    "# ChatLab",
                    elem_id="hero",
                )

                # The badge sits above the tabs, so both Chat and Score text
                # say which model would answer. Beside it is the switcher,
                # a dropdown of the downloaded models that would load now,
                # and, while none is loaded, a link to set up the default
                # on the Models page.
                bar = _build_model_bar()

                with gr.Tabs(elem_id="conversation-tabs") as conversation_tabs:
                    with gr.Tab("Chat", elem_id="chat-tab"):
                        chat, sampling, steering = _build_chat_tab(saved, states)

                    with gr.Tab("Score text"):
                        score = _build_score_tab(states)

                    with gr.Tab("Prompts", elem_id="prompts-tab"):
                        prompts = _build_prompts_tab()

                    with gr.Tab("Fork tree", elem_id="fork-tree-tab"):
                        tree = _build_fork_tree()

                    with gr.Tab("Compare", elem_id="compare-tab"):
                        compare = _build_compare_tab(states)

                    experiments_view = experiments.build()

            # The seam between the transcript and the readings is a
            # handle: drag it to give either pane the other's room.
            # See RESIZE_JS.
            inspector_resizer = gr.HTML(
                pane_handle("inspector-pane"),
                elem_id="inspector-resizer",
                container=False,
                padding=False,
            )

            inspector, layers = _build_inspector(saved)
    return ChatPage(
        column=chat_page,
        bar=bar,
        chat=chat,
        sampling=sampling,
        steering=steering,
        score=score,
        prompts=prompts,
        tree=tree,
        compare=compare,
        inspector=inspector,
        layers=layers,
        tabs=conversation_tabs,
        experiments_view=experiments_view,
        inspector_resizer=inspector_resizer,
    )


def _build_model_bar() -> ModelBar:
    """The model bar above the tabs, and the timer every open tab repaints it on."""

    with gr.Row(elem_id="model-bar"):
        model_badge_view = gr.HTML(
            loaded_model_badge(), elem_id="model-badge"
        )
        # Painted empty and filled by demo.load, as My Models
        # is: the choices need the cache scanned and the
        # machine's memory read, which is not for build time.
        model_switch = gr.Dropdown(
            choices=[],
            value=None,
            label="Switch model",
            show_label=False,
            container=False,
            visible=False,
            interactive=True,
            elem_id="model-switch",
        )
        default_model_button = gr.Button(
            "Set up the default model",
            variant="primary",
            size="sm",
            visible=not runtime.MANAGER.loaded,
            elem_id="default-model",
        )

    # What the switcher above was last drawn from, per tab:
    # the cache revision, the models it came to, and when
    # their fit was read. The timer needs all three to tell
    # a list that is merely idle from one that another
    # tab's download or removal, or a change in the
    # machine's free memory, has left out of date; see
    # refresh_stale_model_switch.
    switch_stamp = gr.State(None)

    # Nothing to see: the timer is what makes the badge tell every
    # open tab about a load or unload, not just the one that asked
    # for it. See BADGE_REFRESH_SECONDS.
    badge_timer = gr.Timer(BADGE_REFRESH_SECONDS)
    return ModelBar(
        badge_view=model_badge_view,
        switch=model_switch,
        default_model_button=default_model_button,
        switch_stamp=switch_stamp,
        badge_timer=badge_timer,
    )


def _build_chat_tab(saved: settings.Settings, states: SharedState) -> tuple[ChatTab, Sampling, Steering]:
    """The Chat tab: the conversation, the message box, and the tools under it."""

    # Two views of one conversation, one at a
    # time. The chatbot renders the reply as the
    # reader would read it - markdown, code
    # blocks, a collapsed reasoning block. The
    # token view writes the same messages out
    # token by token, whitespace shown, painted by
    # the scale on the right. Neither is a
    # substitute for the other, which is why this
    # is a switch and not a replacement.
    token_view = gr.Radio(
        choices=["Rendered view", "Token view"],
        value="Rendered view",
        type="index",
        label="Conversation view",
        show_label=False,
        container=False,
        elem_id="token-view",
    )
    chatbot = gr.Chatbot(
        type="messages",
        label="Conversation",
        height=560,
        show_label=False,
        elem_id="conversation",
        editable="all",
        placeholder="Load a model, then start a conversation.",
    )
    token_strip = gr.HighlightedText(
        label=TRANSCRIPT_LABEL,
        color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
        show_legend=True,
        combine_adjacent=False,
        visible=False,
        elem_id="token-strip",
        elem_classes=[MENU_STRIP_CLASS],
    )
    with gr.Group(visible=False, elem_id="token-editor") as token_editor:
        token_edit_text = gr.Textbox(
            label="Edit your message", lines=3,
            info="Saving replaces the replies after this message and generates a new reply.",
        )
        with gr.Row():
            token_edit_save = gr.Button("Save and regenerate", variant="primary")
            token_edit_cancel = gr.Button("Cancel")
    # The box and the controls that act on it are
    # one bordered composer, the way a message box
    # is drawn everywhere else: the stylesheet
    # takes the border off the box itself and puts
    # it around the pair, so the row below reads as
    # part of the box rather than as four loose
    # buttons under it.
    with gr.Column(elem_id="composer"):
        prompt = gr.Textbox(
            label="Message",
            show_label=False,
            elem_id="message-input",
            **message_box_settings(saved.enter_sends),
        )
        with gr.Row(elem_id="chat-actions"):
            # Send is written first because it is
            # the important one and a keyboard
            # reaches it first; the stylesheet
            # moves it to the end of the row, where
            # the eye leaves the text it just
            # typed.
            send_button = gr.Button("Send", variant="primary", min_width=70)
            # Escape presses this; see SHORTCUT_JS,
            # which finds it by this id.
            stop_button = gr.Button(
                STOP_LABEL,
                variant="stop",
                visible=False,
                elem_id="stop-button",
            )
            # The three that rework the last reply
            # stand together at the left as quiet
            # buttons. Each holds its own width
            # rather than taking an equal share of
            # the row, which had them spread across
            # the page as three unrelated labels.
            retry_button = gr.Button("Retry", min_width=80, elem_classes=icon_classes("rotate-ccw"))
            next_token_button = gr.Button("Next token", min_width=90)
            undo_button = gr.Button("Undo last", min_width=90, elem_classes=icon_classes("undo"))

    generation_status = gr.Markdown("Ready.", elem_id="generation-status")
    with gr.Accordion("Conversation tools", open=False, elem_id="conversation-tools"):
        # Sampling and file controls are available on demand.
        sampling = _build_sampling(saved)
        steering = _build_steering(states)
        with gr.Row():
            save_button = gr.Button("Save conversation", elem_classes=icon_classes("download"))
            load_upload = gr.UploadButton(
                "Load conversation",
                file_types=[".json"],
                type="filepath",
                elem_classes=icon_classes("folder-open"),
            )
        saved_file = gr.File(
            label="Saved conversation",
            visible=False,
            interactive=False,
        )
        with gr.Accordion("Export full metric trace", open=False):
            with gr.Row():
                gr.DownloadButton(
                    "Download JSON",
                    value=lambda trace: write_trace_export(
                        trace, "json"
                    ),
                    inputs=states.trace,
                    size="sm",
                )
                gr.DownloadButton(
                    "Download CSV",
                    value=lambda trace: write_trace_export(trace, "csv"),
                    inputs=states.trace,
                    size="sm",
                )
    chat = ChatTab(
        token_view=token_view,
        chatbot=chatbot,
        token_strip=token_strip,
        token_editor=token_editor,
        token_edit_text=token_edit_text,
        token_edit_save=token_edit_save,
        token_edit_cancel=token_edit_cancel,
        prompt=prompt,
        send_button=send_button,
        stop_button=stop_button,
        retry_button=retry_button,
        next_token_button=next_token_button,
        undo_button=undo_button,
        generation_status=generation_status,
        save_button=save_button,
        load_upload=load_upload,
        saved_file=saved_file,
    )
    return chat, sampling, steering


def _build_sampling(saved: settings.Settings) -> Sampling:
    """The sampling accordion, built with the saved settings."""

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
    return Sampling(
        accordion=sampling_accordion,
        temperature=temperature,
        temperature_reset=temperature_reset,
        top_p=top_p,
        top_p_reset=top_p_reset,
        top_k=top_k,
        top_k_reset=top_k_reset,
        max_new_tokens=max_new_tokens,
        max_new_tokens_reset=max_new_tokens_reset,
        skip_top_below=skip_top_below,
        skip_top_below_reset=skip_top_below_reset,
        seed=seed,
        randomize_seed=randomize_seed,
    )


def _build_steering(states: SharedState) -> Steering:
    """The steering vector's accordion, extraction included."""

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
                    inputs=[states.extract, extract_layer],
                    size="sm",
                    min_width=110,
                )
    return Steering(
        state=states.steering,
        upload=steering_upload,
        remove=steering_remove,
        enabled=steering_enabled,
        strength=steering_strength,
        layer=steering_layer,
        status=steering_status,
        extract_positive=extract_positive,
        extract_negative=extract_negative,
        extract_chat=extract_chat,
        extract_pool=extract_pool,
        extract_button=extract_button,
        extract_status=extract_status,
        extract_table=extract_table,
        extract_layer=extract_layer,
        extract_apply=extract_apply,
    )


def _build_score_tab(states: SharedState) -> ScoreTab:
    """The Score text tab's controls."""

    gr.Markdown(
        "Measure text the model did not write. One forward pass "
        "gives every token the same rank, probability, surprise, "
        "and entropy the chat view shows."
    )
    score_context = gr.Textbox(
        label="Context (optional)",
        placeholder="Text that comes before the part you want scored.",
        lines=3,
    )
    use_chat_template = gr.Checkbox(
        value=False,
        label="Treat the context as a chat message",
        info=(
            "Wraps the context in the model's chat template, so the "
            "scored text is measured as a reply. Models without a "
            "chat template score the context as plain text, and say so."
        ),
    )
    score_input = gr.Textbox(
        label="Text to score",
        placeholder="Paste the text you want measured…",
        lines=8,
    )
    # Scoring refuses a passage above the model's
    # limit. This is the same count, made while
    # the passage is still being written.
    score_budget = gr.Markdown(
        SCORE_COUNT_HINT,
        elem_id="score-budget",
        elem_classes=["token-budget"],
    )
    score_button = gr.Button("Score text", variant="primary")
    score_status = gr.Markdown("Nothing scored yet.")
    # The scored tokens are shown here rather than
    # beside the chat: this tab has no conversation
    # to paint, and the inspector's business is
    # whichever token was last clicked, wherever
    # it was clicked.
    score_strip = gr.HighlightedText(
        label="Scored tokens — click one",
        color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
        show_legend=True,
        combine_adjacent=False,
        elem_id="score-strip",
    )
    return ScoreTab(
        budget_load=states.score_budget_load,
        context=score_context,
        use_chat_template=use_chat_template,
        text=score_input,
        budget=score_budget,
        button=score_button,
        status=score_status,
        strip=score_strip,
    )


def _build_prompts_tab() -> PromptsTab:
    """The Prompts tab's controls."""

    gr.Markdown(
        "Run a list of prompts, each in a conversation of its "
        "own, and keep every token's measurements. The system "
        "prompt and prefill come from Settings, the sampling "
        "controls from the Chat tab, so a batch is measured "
        "exactly as a reply typed by hand would be."
    )
    prompts_box = gr.Textbox(
        label="Prompts",
        placeholder=(
            "One prompt per block, with a blank line between "
            "them, so a prompt can run to several lines."
        ),
        lines=8,
        elem_id="prompts-box",
    )
    prompt_count = gr.Markdown(
        PROMPT_COUNT_HINT,
        elem_id="prompt-count",
        elem_classes=["token-budget"],
    )
    with gr.Row():
        run_prompts_button = gr.Button(
            "Run prompts", variant="primary", min_width=110
        )
        # Escape presses this while a batch runs;
        # see SHORTCUT_JS, which finds whichever
        # stop button is on screen by these ids.
        stop_prompts_button = gr.Button(
            "Stop",
            variant="stop",
            visible=False,
            elem_id="stop-batch-button",
            min_width=70,
        )
        prompts_upload = gr.UploadButton(
            "Load prompts",
            # "text" is any text file, which is
            # what the parser's fallback reads: a
            # prompt set arrives as often in a
            # .md or a file with no extension at
            # all as in a .txt, and a filter
            # narrower than the parser would put
            # those out of reach of a tab that
            # says it takes them. The two JSON
            # forms are named because a browser
            # does not always call them text.
            file_types=["text", ".json", ".jsonl"],
            type="filepath",
            min_width=130,
        )
    batch_status = gr.Markdown(
        "Nothing run yet.", elem_id="batch-status"
    )
    batch_results = gr.Dataframe(
        headers=BATCH_HEADERS,
        datatype=[
            "number",
            "str",
            "str",
            "number",
            "number",
            "number",
            "number",
        ],
        column_widths=["5%", "27%", "32%", "9%", "9%", "9%", "9%"],
        wrap=True,
        interactive=False,
        elem_id="batch-results",
        label="Results — one row per prompt",
    )
    batch_files = gr.File(
        label="One trace per prompt, and a table of every token",
        file_count="multiple",
        visible=False,
        interactive=False,
        elem_id="batch-files",
    )
    return PromptsTab(
        box=prompts_box,
        count=prompt_count,
        run_button=run_prompts_button,
        stop_button=stop_prompts_button,
        upload=prompts_upload,
        batch_status=batch_status,
        batch_results=batch_results,
        batch_files=batch_files,
    )


def _build_fork_tree() -> ForkTree:
    """The Fork tree tab, drawn empty until the forks are restored."""

    gr.Markdown(
        "### Conversation forks\n"
        "Follow each fork back to its message or token. "
        "Choose **A** and **B** on two branches to compare them below."
    )
    tree_action = gr.Textbox(
        elem_id="fork-tree-action", elem_classes=[MENU_BRIDGE_CLASS]
    )
    tree_view = gr.HTML(
        render_fork_tree([], new_forks(), {})[0],
        elem_id="fork-tree-view",
    )
    tree_comparison = gr.HTML(
        render_fork_tree([], new_forks(), {})[1],
        elem_id="fork-tree-comparison",
    )
    return ForkTree(
        action=tree_action,
        view=tree_view,
        comparison=tree_comparison,
    )


def _build_compare_tab(states: SharedState) -> CompareTab:
    """The Compare tab, activation patching included."""

    gr.Markdown(
        "Two runs, side by side. Fill slot A, change one "
        "thing — the model, the precision, a steering "
        "vector, the seed — and fill slot B. Each slot "
        "keeps its own model and settings, so the two can "
        "be filled a model load apart. The system prompt "
        "and prefill come from Settings, the sampling "
        "controls and the steering vector from the Chat tab."
    )
    pair_conditions, pair_button, pair_load_status = experiment_compare.build()
    compare_mode = gr.Radio(
        choices=[
            ("Writing a reply", REPLY),
            ("Measuring fixed text", MEASUREMENT),
        ],
        value=REPLY,
        label="Fill a slot by",
        info=(
            "Two replies part company somewhere in the "
            "answer and only their shared opening can be "
            "compared. Two runs over one fixed passage "
            "never part, so every token is comparable — "
            "and a measurement reads the context and the "
            "passage alone, so put a measurement's framing "
            "in the context box rather than in the system "
            "prompt."
        ),
    )
    compare_prompt = gr.Textbox(
        label="Prompt for both runs",
        placeholder="The message both runs answer.",
        lines=4,
        elem_id="compare-prompt",
    )
    compare_template = gr.Checkbox(
        value=False,
        visible=False,
        label="Read the context as a chat message",
    )
    compare_text = gr.Textbox(
        label="Text to measure",
        placeholder="The passage both runs read…",
        lines=6,
        visible=False,
        elem_id="compare-text",
    )
    with gr.Row():
        compare_run_a = gr.Button(
            "Run into A", variant="primary", min_width=110
        )
        compare_run_b = gr.Button(
            "Run into B", variant="primary", min_width=110
        )
        # Escape presses this while a slot is being
        # filled; see SHORTCUT_JS.
        compare_stop = gr.Button(
            "Stop",
            variant="stop",
            visible=False,
            elem_id="stop-compare",
            min_width=70,
        )
        compare_clear = gr.Button(
            "Clear both", min_width=110
        )
    compare_status = gr.Markdown(
        COMPARE_EMPTY, elem_id="compare-status"
    )
    compare_tiles = gr.HTML(
        charts.comparison_tiles({}), elem_id="compare-tiles"
    )
    compare_headline = gr.Markdown(
        "", elem_id="compare-headline"
    )
    compare_a_heading = gr.Markdown(f"**A** · {COMPARE_SLOT_EMPTY}")
    compare_a_strip = gr.HighlightedText(
        label="Slot A",
        color_map=GAP_COLORS,
        show_legend=True,
        combine_adjacent=False,
        elem_id="compare-a-strip",
    )
    compare_b_heading = gr.Markdown(f"**B** · {COMPARE_SLOT_EMPTY}")
    compare_b_strip = gr.HighlightedText(
        label="Slot B",
        color_map=GAP_COLORS,
        show_legend=True,
        combine_adjacent=False,
        elem_id="compare-b-strip",
    )
    gr.Markdown(
        GAP_CAPTION, elem_classes=["scale-caption"]
    )
    compare_chart = gr.HTML(
        charts.EMPTY_CHART, elem_id="compare-chart"
    )
    compare_settings = gr.Dataframe(
        headers=CONFIGURATION_HEADERS,
        datatype=["str", "str", "str"],
        column_widths=["26%", "37%", "37%"],
        wrap=True,
        interactive=False,
        elem_id="compare-settings",
        label="What differed between the two runs",
    )
    compare_rows = gr.Dataframe(
        headers=DIVERGENCE_HEADERS,
        datatype=["number", "str", "number", "number",
                  "number", "str", "str"],
        column_widths=["6%", "16%", "14%", "14%", "12%",
                       "19%", "19%"],
        wrap=True,
        interactive=False,
        elem_id="compare-divergences",
        label="Where the two runs read a shared token most differently",
    )
    difference_position = gr.State(None)
    next_difference = gr.Button("Next largest difference", size="sm")
    difference_detail = gr.Markdown("")
    gr.DownloadButton(
        "Download comparison JSON",
        value=download_comparison,
        inputs=states.compare_export,
        size="sm",
    )
    build_activation_patching(states.compare_a, states.compare_b)
    return CompareTab(
        pair_conditions=pair_conditions,
        pair_button=pair_button,
        pair_load_status=pair_load_status,
        mode=compare_mode,
        prompt=compare_prompt,
        template=compare_template,
        text=compare_text,
        run_a=compare_run_a,
        run_b=compare_run_b,
        stop=compare_stop,
        clear=compare_clear,
        status=compare_status,
        tiles=compare_tiles,
        headline=compare_headline,
        a_heading=compare_a_heading,
        a_strip=compare_a_strip,
        b_heading=compare_b_heading,
        b_strip=compare_b_strip,
        chart=compare_chart,
        settings=compare_settings,
        rows=compare_rows,
        difference_position=difference_position,
        next_difference=next_difference,
        difference_detail=difference_detail,
    )


def _build_inspector(saved: settings.Settings) -> tuple[Inspector, Layers]:
    """The readings pane beside the tabs."""

    with gr.Column(scale=2, min_width=300, elem_id="inspector-pane") as inspector_pane:
        gr.Markdown("## Under the hood", elem_id="inspector-heading")
        color_scale = gr.Dropdown(
            choices=list(COLOR_SCALES),
            value=saved.color_scale,
            label="Color tokens by",
        )
        scale_caption = gr.Markdown(
            COLOR_SCALES[saved.color_scale].caption,
            visible=False,
            elem_classes=["scale-caption"],
        )
        token_detail = gr.Markdown(NO_TOKEN_SELECTED)
        alternatives = gr.Dataframe(
            headers=["Token ID", "Token", "Raw probability"],
            column_widths=["22%", "30%", "48%"],
            wrap=True,
            elem_id="token-alternatives",
            datatype=["number", "str", "number"],
            interactive=False,
            label="Most likely alternatives — click one to branch into it",
        )
        with gr.Accordion("Branch response", open=False, elem_classes=["inspector-section"]):
            with gr.Row():
                branch_button = gr.Button("Branch from token", size="sm", elem_classes=icon_classes("git-branch"))
            gr.Markdown(
                "For one step, choose an alternative and press **Next token** "
                "below the message box. Keep pressing it to extend the reply."
            )
            with gr.Row():
                branch_text = gr.Textbox(
                    label="Or type your own replacement",
                    placeholder=(
                        "Text to put where the selected token was. Include a "
                        "leading space if the word needs one."
                    ),
                    lines=1,
                    scale=3,
                    min_width=160,
                )
                branch_text_button = gr.Button(
                    "Branch with text", size="sm", scale=0, min_width=160,
                    elem_classes=icon_classes("pencil"),
                )
        with gr.Accordion("Layers and attention", open=False, elem_classes=["inspector-section"]):
            layers = _build_layers()
        with gr.Accordion("Response statistics", open=False, elem_classes=["inspector-section"]):
            summary_panel = gr.HTML(charts.summary_tiles({}))
            surprise_panel = gr.HTML(charts.EMPTY_CHART)
        with gr.Accordion("Prompt and context tokens", open=False, elem_classes=["inspector-section"]):
            prompt_note = gr.Markdown("", elem_classes=["scale-caption"])
            prompt_strip = gr.HighlightedText(
                label="Prompt tokens — click one, right-click to replace it",
                color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                show_legend=True,
                combine_adjacent=False,
                elem_id="prompt-strip",
                elem_classes=[MENU_STRIP_CLASS],
            )
    inspector = Inspector(
        pane=inspector_pane,
        color_scale=color_scale,
        scale_caption=scale_caption,
        token_detail=token_detail,
        alternatives=alternatives,
        branch_button=branch_button,
        branch_text=branch_text,
        branch_text_button=branch_text_button,
        summary_panel=summary_panel,
        surprise_panel=surprise_panel,
        prompt_note=prompt_note,
        prompt_strip=prompt_strip,
    )
    return inspector, layers


def _build_layers() -> Layers:
    """The Layers and attention accordion inside the inspector."""

    lens_mode = gr.Radio(
        ["Logit", "Jacobian"], value="Logit", label="Lens",
        info="Logit: the prediction before a token. Jacobian: concept readouts after it.",
    )
    imported_lens = gr.State(None)
    inspection_session = gr.State(
        value=lambda: INSPECTION_CONTROLS.new_session(),
        delete_callback=INSPECTION_CONTROLS.forget,
    )
    with gr.Column(visible=False) as jacobian_controls:
        with gr.Accordion("Lens setup", open=True) as lens_setup:
            gr.Markdown(
                "Import a lens fitted for the loaded checkpoint, from a Hugging Face "
                "repository or a saved file. Fitted lenses are published for many "
                "open models; the [reference tools](https://github.com/anthropics/jacobian-lens#fit) "
                "fit new ones. Supports Llama, Mistral, Qwen, Gemma, OLMo, GLM-4, Phi-3, "
                "Granite, Cohere, and SmolLM3 text models, as Transformers weights or MLX conversions."
            )
            with gr.Row():
                lens_repository = gr.Textbox(
                    label="Hub repository", placeholder="For example, mhough/olmo3-jacobian-lenses",
                    scale=1,
                )
                lens_filename = gr.Textbox(
                    label="File in the repository", placeholder="For example, lenses/olmo-3-7b-think.pt",
                    scale=1,
                )
            lens_file = gr.File(label="Or a saved lens.pt file", file_types=[".pt"], type="filepath")
            fitted_model_id = gr.Textbox(
                label="Model ID the lens was fitted for",
                placeholder="For example, Qwen/Qwen3-0.6B",
                info="For an MLX conversion, the full-precision model it was made from.",
            )
            import_lens_button = gr.Button("Import lens", size="sm")
        import_lens_status = gr.Markdown("No lens imported for this session.")
        pinned_concept = gr.Textbox(
            label="Pin a vocabulary token (optional)",
            placeholder="Click a cell in the grid, or type a token, then inspect again",
            info="Use the exact text, including any leading space. One vocabulary token at a time.",
            elem_id="jacobian-pin",
        )
        # A clicked cell writes its exact token ID here beside the
        # visible text, so a token whose text does not tokenize back
        # to itself is still pinned as the token it is. Hidden by the
        # bridge class, not visible=False, which would take the box
        # out of the DOM where the page script has to find it.
        pinned_token_id = gr.Textbox(elem_id="jacobian-pin-id", elem_classes=[MENU_BRIDGE_CLASS])
    with gr.Row():
        inspect_button = gr.Button(
            "Inspect layers", size="sm", scale=0, min_width=160,
            elem_classes=icon_classes("layers"), elem_id="inspect-layers",
        )
        inspect_status = gr.Markdown(
            INSPECT_HINT, elem_classes=["scale-caption"]
        )
    lens_panel = gr.HTML(charts.EMPTY_LENS)
    attention_layer = gr.Slider(
        0,
        1,
        value=0,
        step=1,
        label="Attention layer",
        info="0 averages every layer. Release the slider to repaint.",
    )
    attention_panel = gr.HTML(charts.EMPTY_ATTENTION)
    with gr.Row():
        kv_layer = gr.Slider(
            1,
            1,
            value=1,
            step=1,
            label="Cache layer",
            info="Release the slider to read another layer.",
            scale=2,
        )
        kv_metric = gr.Radio(
            list(charts.KV_METRICS), value="Key norm",
            label="Cache readout", scale=1,
        )
    kv_panel = gr.HTML(charts.EMPTY_KV_CACHE)
    return Layers(
        lens_mode=lens_mode,
        imported_lens=imported_lens,
        inspection_session=inspection_session,
        jacobian_controls=jacobian_controls,
        lens_setup=lens_setup,
        lens_repository=lens_repository,
        lens_filename=lens_filename,
        lens_file=lens_file,
        fitted_model_id=fitted_model_id,
        import_lens_button=import_lens_button,
        import_lens_status=import_lens_status,
        pinned_concept=pinned_concept,
        pinned_token_id=pinned_token_id,
        inspect_button=inspect_button,
        inspect_status=inspect_status,
        lens_panel=lens_panel,
        attention_layer=attention_layer,
        attention_panel=attention_panel,
        kv_layer=kv_layer,
        kv_metric=kv_metric,
        kv_panel=kv_panel,
    )


def wire_model_bar(
    demo: gr.Blocks,
    nav: gr.Radio,
    page: ChatPage,
    thinking_mode: gr.Radio,
    weight_precision: gr.Radio,
) -> None:
    """The scored token count, and the badge, switcher and thinking control on their timer.

    ``thinking_mode`` is the Settings page's, which follows the model in
    memory the way the badge does; ``weight_precision`` is the Models page's,
    which decides what the switcher would fit.
    """

    # The scored token count follows the boxes as they are typed into.
    # always_last coalesces a burst of keystrokes into the one count that
    # matters, and the progress bar is hidden because a spinner on every
    # keystroke would be worse than the number is good.
    for control in page.score.budget_inputs:
        control.change(
            score_token_count,
            page.score.budget_inputs,
            page.score.budget_outputs,
            trigger_mode="always_last",
            show_progress="hidden",
            concurrency_id=SCORE_BUDGET_QUEUE,
        )

    # The badge is refreshed on the way to the chat page as well, so a
    # load started a moment ago shows as one in progress rather than as
    # the "no model" state the page was left in.
    nav.change(refresh_thinking_mode, None, thinking_mode, show_progress="hidden")
    demo.load(refresh_thinking_mode, None, thinking_mode)
    page.bar.badge_timer.tick(refresh_thinking_mode, None, thinking_mode, **QUIET_TICK)
    nav.change(refresh_model_badge, None, page.bar.badge_outputs)
    demo.load(refresh_model_badge, None, page.bar.badge_outputs)
    # The switcher is drawn on the same two occasions. Its choices cost a
    # cache scan and a memory reading, so the timer only redraws it once
    # what it shows, what is on disk, or what would now fit has moved; see
    # refresh_stale_model_switch. Every draw hands back the stamp it read,
    # which is how the next tick knows the difference.
    nav.change(refresh_model_switch, weight_precision, page.bar.switch_outputs)
    demo.load(refresh_model_switch, weight_precision, page.bar.switch_outputs)
    page.bar.badge_timer.tick(
        refresh_stale_model_switch,
        [page.bar.switch, page.bar.switch_stamp, weight_precision],
        page.bar.switch_outputs,
        **QUIET_TICK,
    )
    # And on a timer, so a tab that did not start the load hears about it
    # too. demo.load stays: it draws the badge at once rather than leaving
    # the value baked in when the page was built there for a tick.
    # QUIET_TICK because this one runs on its own: the default fades a
    # handler's outputs in and out, which every couple of seconds would
    # have the badge flickering at a reader who never asked it anything.
    page.bar.badge_timer.tick(refresh_model_badge, None, page.bar.badge_outputs, **QUIET_TICK)
    # The same timer un-sticks the scored token count. A count asked for
    # during a reply gives up, and nothing about that message corrects
    # itself once the reply ends; see recover_score_budget, which is why
    # this is one listener rather than one on every path out of a
    # generation.
    page.bar.badge_timer.tick(
        recover_score_budget,
        [page.score.budget, page.score.budget_load, *page.score.budget_inputs],
        page.score.budget_outputs,
        concurrency_id=SCORE_BUDGET_QUEUE,
        **QUIET_TICK,
    )


def wire_sampling(sampling: Sampling, states: SharedState) -> None:
    """The sampling summary, and each move written into the conversation."""

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
    for control in sampling.controls:
        control.change(
            update_sampling_label,
            sampling.controls,
            sampling.accordion,
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
            [states.forks, *sampling.controls],
            states.forks,
            trigger_mode="always_last",
            show_progress="hidden",
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )


def wire_steering(steering: Steering, states: SharedState) -> None:
    """Importing, extracting and adjusting the steering vector."""

    steering.upload.upload(
        import_vector, [steering.upload, states.forks], [states.forks, *steering.outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    steering.remove.click(
        remove_vector, states.forks, [states.forks, *steering.outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    steering.extract_button.click(
        extract_vector,
        [steering.extract_positive, steering.extract_negative, steering.extract_chat, steering.extract_pool],
        [states.extract, steering.extract_table, steering.extract_layer, steering.extract_apply, steering.extract_status],
    )
    steering.extract_table.select(
        choose_layer, states.extract, [steering.extract_layer, steering.extract_status]
    )
    # input rather than change: the extraction writes the layer control
    # itself, with a fuller status beside it, and a change listener would
    # fire on that write and replace the status with the shorter line.
    steering.extract_layer.input(
        describe_layer,
        [states.extract, steering.extract_layer],
        steering.extract_status,
        show_progress="hidden",
    )
    steering.extract_apply.click(
        use_extracted,
        [states.forks, states.extract, steering.extract_layer],
        [states.forks, *steering.outputs],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    for control in (steering.enabled, steering.strength, steering.layer):
        control.input(
            remember_steering,
            [states.forks, states.steering, steering.enabled, steering.strength, steering.layer],
            [states.forks, states.steering, steering.status],
            trigger_mode="always_last", show_progress="hidden",
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )


def wire_score_and_batch(
    page: ChatPage,
    states: SharedState,
    system_prompt: gr.Textbox,
    assistant_prefill: gr.Textbox,
) -> None:
    """The Score text tab, and running a batch from the Prompts tab.

    The system prompt and the prefill are the Settings page's, which a batch
    is prompted with as a typed reply would be.
    """

    page.score.button.click(
        score_text,
        [page.score.context, page.score.text, page.score.use_chat_template, page.inspector.color_scale],
        [
            page.score.strip,
            states.metrics,
            states.score_metrics,
            page.inspector.prompt_strip,
            states.prompt_metrics,
            page.inspector.prompt_note,
            page.inspector.summary_panel,
            page.inspector.surprise_panel,
            page.score.status,
            page.inspector.token_detail,
            page.inspector.alternatives,
            states.selected_token,
            states.branch_pick,
            states.context_ids,
            states.score_context_ids,
        ],
    )
    # A batch reads its prompts from the box and everything else from the
    # controls the Chat tab and Settings already own, so there is nothing
    # to set up before running one.
    batch_outputs = [
        page.prompts.batch_status,
        page.prompts.batch_results,
        page.prompts.run_button,
        page.prompts.stop_button,
        page.prompts.batch_files,
        states.batch_directory,
    ]
    batch_run = page.prompts.run_button.click(
        run_prompts,
        [
            page.prompts.box,
            states.loaded_prompts,
            system_prompt,
            assistant_prefill,
            page.sampling.temperature,
            page.sampling.top_p,
            page.sampling.top_k,
            page.sampling.skip_top_below,
            page.sampling.max_new_tokens,
            page.sampling.seed,
            page.sampling.randomize_seed,
        ],
        batch_outputs,
    )
    # Cancelling closes the run at its last yield, which is what returns
    # the model lock; stop_batch() only puts the buttons back. The rows
    # and files already published stay on screen, and they describe the
    # prompts that finished.
    # Stop reads the run's directory rather than the frame the cancelled
    # generator published last: the prompt it was in the middle of is
    # written on the way out, after that frame is gone. See stop_batch().
    page.prompts.stop_button.click(
        stop_batch, states.batch_directory, batch_outputs, cancels=[batch_run]
    )


def wire_compare(
    demo: gr.Blocks,
    page: ChatPage,
    states: SharedState,
    system_prompt: gr.Textbox,
    assistant_prefill: gr.Textbox,
    thinking_mode: gr.Radio,
) -> None:
    """Filling the two comparison slots, the saved experiments, and the tabs' inspector.

    The system prompt, the prefill and the thinking mode are the Settings
    page's, which a slot is filled under as a typed reply would be.
    """

    compare_outputs = [
        page.compare.a_heading,
        page.compare.b_heading,
        page.compare.a_strip,
        page.compare.b_strip,
        page.compare.tiles,
        page.compare.chart,
        page.compare.headline,
        page.compare.settings,
        page.compare.rows,
        states.compare_export,
    ]
    # Everything a run reads. The sampling controls and the steering
    # vector are the Chat tab's own, so a slot is filled under exactly
    # the settings a reply typed by hand would have used - which is what
    # makes changing one of them between A and B a clean experiment.
    compare_inputs = [
        page.compare.mode,
        page.compare.prompt,
        page.compare.text,
        page.compare.template,
        system_prompt,
        assistant_prefill,
        page.sampling.temperature,
        page.sampling.top_p,
        page.sampling.top_k,
        page.sampling.skip_top_below,
        page.sampling.max_new_tokens,
        page.sampling.seed,
        page.sampling.randomize_seed,
        thinking_mode,
        *page.steering.inputs,
    ]
    compare_slot_outputs = [page.compare.status, page.compare.run_a, page.compare.run_b, page.compare.stop]
    compare_runs = []
    for slot, button, held in (
        ("A", page.compare.run_a, states.compare_a),
        ("B", page.compare.run_b, states.compare_b),
    ):
        filling = button.click(
            partial(fill_slot, slot),
            compare_inputs,
            [held, *compare_slot_outputs],
        )
        # Drawn after the slot is filled rather than from inside the run:
        # the comparison needs both slots, and a run knows only its own.
        filling.then(
            render_comparison,
            [states.compare_a, states.compare_b],
            compare_outputs,
        )
        compare_runs.append(filling)
    paired = page.compare.pair_button.click(
        experiment_compare.run_pair, [*page.compare.pair_conditions, *compare_inputs],
        [states.compare_a, states.compare_b, *compare_slot_outputs, page.compare.pair_button, page.compare.pair_load_status],
    )
    compare_runs.append(paired)
    page.compare.next_difference.click(experiment_compare.next_difference,
                          [states.compare_export, page.compare.difference_position],
                          [page.compare.difference_position, page.compare.difference_detail])
    states.compare_export.change(lambda: (None, ""), None, [page.compare.difference_position, page.compare.difference_detail],
                                show_progress="hidden")
    experiments.wire(page.experiments_view, demo, states.trace, states.chat_context_ids,
                     states.compare_a, states.compare_b, states.insight, states.inspect_target)
    page.tabs.select(experiments.inspector_visibility, None, [page.inspector.pane, page.inspector_resizer],
                             show_progress="hidden", queue=False)
    for held in (states.compare_a, states.compare_b):
        held.change(render_comparison, [states.compare_a, states.compare_b], compare_outputs)
    # Cancelling closes the run at its last yield, which is what gives the
    # model lock back; this only puts the buttons right. The slot keeps
    # whatever it held before, because half a response is not a run.
    page.compare.stop.click(
        stop_comparison, None, compare_slot_outputs, cancels=compare_runs
    ).then(lambda: gr.update(interactive=True), None, page.compare.pair_button)
    # cancels, because clearing during a run is otherwise undone by the
    # run: its remaining frames would write over the cleared status and
    # its last one would put the slot back. See clear_slots().
    page.compare.clear.click(
        clear_slots,
        None,
        [states.compare_a, states.compare_b, *compare_slot_outputs, *compare_outputs],
        cancels=compare_runs,
    ).then(lambda: gr.update(interactive=True), None, page.compare.pair_button)
    page.compare.mode.change(
        mode_controls,
        page.compare.mode,
        [page.compare.prompt, page.compare.text, page.compare.template],
    )


def wire_prompt_file(prompts: PromptsTab, states: SharedState) -> None:
    """The prompt count under the box, and loading a file of prompts into it."""

    prompts.box.change(
        count_prompts,
        [prompts.box, states.loaded_prompts],
        prompts.count,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    prompts.upload.upload(
        load_prompt_file,
        [prompts.upload, prompts.box, states.loaded_prompts],
        [prompts.box, prompts.batch_status, states.loaded_prompts],
    )


def wire_inspector(page: ChatPage, states: SharedState) -> None:
    """The colors, the token view and its editor, and every reading in the inspector."""

    # The conversation is painted from the turns; the two strips are
    # painted from the measurements they were handed.
    page.inspector.color_scale.change(
        recolor,
        [states.conversation, states.score_metrics, states.prompt_metrics, page.inspector.color_scale],
        [page.chat.token_strip, page.score.strip, page.inspector.prompt_strip, page.inspector.scale_caption],
    )
    # Which view of the conversation is on screen. The token view is
    # redrawn on the way in rather than left to the next frame, since the
    # conversation may have moved on while it was hidden.
    page.chat.token_view.change(
        partial(show_token_view, conversation_id=states.conversation._id),
        [page.chat.token_view, states.conversation, page.inspector.color_scale],
        [page.chat.chatbot, page.chat.token_strip],
        show_progress="hidden",
    )
    editor_outputs = [page.chat.token_editor, page.chat.token_edit_text, states.token_edit_target]
    page.chat.token_view.change(close_token_editor, outputs=editor_outputs, show_progress="hidden")
    page.chat.token_edit_cancel.click(close_token_editor, outputs=editor_outputs, show_progress="hidden")
    page.chat.token_strip.select(
        open_token_editor,
        [states.conversation, states.metrics],
        editor_outputs,
        show_progress="hidden",
    )

    # One click in the conversation answers every question the inspector
    # asks of it, so it is one listener rather than four.
    page.chat.token_strip.select(
        select_transcript_token,
        [states.conversation, states.metrics],
        [page.inspector.token_detail, page.inspector.alternatives, states.selected_token, states.inspect_target, states.branch_pick],
    )
    page.chat.token_strip.select(
        token_menu_payload,
        [states.conversation, states.metrics, states.menu_request],
        states.menu_response,
        show_progress="hidden",
        queue=False,
    )
    # Also keep the message it landed in, so Fork works from
    # the token view exactly as it does from the chatbot.
    page.chat.token_strip.select(
        remember_transcript_message, states.conversation, states.selected_message
    )

    for strip, strip_metrics, source, where in (
        (page.score.strip, states.score_metrics, "score", "score"),
        (page.inspector.prompt_strip, states.prompt_metrics, "prompt", "prompt"),
    ):
        strip.select(
            inspect_token(source),
            inputs=strip_metrics,
            outputs=[page.inspector.token_detail, page.inspector.alternatives],
        )
        # A second listener keeps the clicked position for the
        # alternatives table, and a third the position the layer
        # inspector would explain. Neither strip is part of a
        # conversation, so clicking one disarms whatever branch the
        # conversation had armed, and a row chosen in either is told it
        # has nothing to branch rather than pairing with the token last
        # clicked in the chat.
        strip.select(
            remember_strip_selection(source),
            strip_metrics,
            [states.selected_token, states.branch_pick],
        )
        strip.select(remember_inspect_target(where), strip_metrics, states.inspect_target)
    # Only the prompt strip's tokens can be replaced: they are the ones a
    # reply was actually generated from, and the ids behind them are kept.
    page.inspector.prompt_strip.select(
        prompt_menu_payload,
        [states.prompt_metrics, states.context_ids, states.prompt_menu_request],
        states.prompt_menu_response,
        show_progress="hidden",
        queue=False,
    )
    page.inspector.alternatives.select(
        choose_alternative,
        [states.conversation, states.score_metrics, states.prompt_metrics, states.selected_token],
        [page.inspector.token_detail, states.branch_pick],
    )
    inspection_outputs = [page.layers.lens_panel, page.layers.attention_panel, states.insight, page.layers.inspect_status]
    page.layers.inspect_button.click(
        inspect_layers,
        [
            states.inspect_target,
            states.metrics,
            states.prompt_metrics,
            states.context_ids,
            page.layers.attention_layer,
            states.score_metrics,
            states.score_context_ids,
            states.chat_metrics,
            states.chat_context_ids,
            page.layers.lens_mode,
            page.layers.imported_lens,
            page.layers.pinned_concept,
            page.layers.inspection_session,
            page.layers.pinned_token_id,
        ],
        [page.layers.lens_panel, page.layers.attention_panel, page.layers.attention_layer, states.insight, page.layers.inspect_status],
    )
    page.layers.attention_layer.release(
        render_attention, [states.insight, page.layers.attention_layer], page.layers.attention_panel
    )
    # Every readout, and every reset that takes one away, passes through
    # the insight state, so the cache view follows it there. The cache is
    # read from memory rather than rebuilt, so each control reads again.
    kv_inputs = [states.insight, page.layers.kv_layer, page.layers.kv_metric]
    states.insight.change(render_kv_cache, kv_inputs, [page.layers.kv_panel, page.layers.kv_layer], **QUIET_TICK)
    page.layers.kv_layer.release(render_kv_cache, kv_inputs, [page.layers.kv_panel, page.layers.kv_layer])
    page.layers.kv_metric.change(render_kv_cache, kv_inputs, [page.layers.kv_panel, page.layers.kv_layer])
    page.layers.lens_mode.change(
        change_lens_mode, [page.layers.lens_mode, page.layers.inspection_session],
        [page.layers.jacobian_controls, page.layers.attention_layer, *inspection_outputs],
        queue=False,
    )
    page.layers.import_lens_button.click(
        import_jacobian_lens, [page.layers.lens_file, page.layers.fitted_model_id, page.layers.lens_repository, page.layers.lens_filename],
        [page.layers.imported_lens, page.layers.import_lens_status],
    )
    # QUIET_TICK for the same reason as the metrics_state handler below:
    # clearing a stale readout is instant, so the panel should not flash a
    # spinner over itself on the way.
    page.layers.imported_lens.change(
        reset_inspection, states.insight, inspection_outputs, **QUIET_TICK,
    )
    page.layers.imported_lens.change(
        lambda imported: gr.update(open=False) if imported else gr.skip(),
        page.layers.imported_lens, page.layers.lens_setup,
    )
    page.layers.pinned_concept.input(
        change_pinned_token, [page.layers.pinned_concept, page.layers.inspection_session], inspection_outputs,
        queue=False,
    )
    # Every path that redraws the strips writes the metrics state, so this
    # is where a readout of a token that is no longer on screen goes away.
    #
    # Streaming writes that state on every frame, and reset_inspection
    # skips its outputs once there is nothing left to clear. Gradio marks
    # them pending regardless, so without QUIET_TICK the inspector blinks
    # its way through every reply: see QUIET_TICK in ui.common for what
    # the two arguments each take away.
    states.metrics.change(
        reset_inspection, states.insight, inspection_outputs, **QUIET_TICK,
    )
