"""The page itself: every control and how the handlers are wired to them."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import html
from functools import partial

import gradio as gr

from chatlab.ui.fork_tree import TREE_CSS, TREE_JS, render_fork_tree, select_tree_branch
from chatlab import charts, settings, themes
from chatlab.thinking import THINKING_CHOICES
from chatlab.conversation import MAIN_BRANCH, branch_choices, new_forks
from chatlab.device_memory import warm_device
from chatlab.token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
from chatlab.trace_export import write_trace_export
from chatlab.ui import runtime, experiments, experiment_compare
from chatlab.ui.icons import icon_classes
from chatlab.ui.background import ConversationEvents, ConversationJob
from chatlab.ui.outputs import (
    CHAT_OUTPUT_NAMES,
    CLEAR_OUTPUT_NAMES,
    FORK_OUTPUT_NAMES,
    NEW_CONVERSATION_OUTPUT_NAMES,
    POLL_OUTPUT_NAMES,
    RESTORE_OUTPUT_NAMES,
    STEERED_LOAD_OUTPUT_NAMES,
    STOP_OUTPUT_NAMES,
    TOKEN_EDIT_OUTPUT_NAMES,
    UNDO_OUTPUT_NAMES,
)
from chatlab.ui.token_edit import close_token_editor, open_token_editor, save_token_edit
from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.registry import load_enabled
from chatlab.ui.extensions_page import (
    build_extension_settings,
    data_directory,
    extension_css,
    restore_extensions,
)
from chatlab.ui.common import (
    CHAT_PAGE,
    CONVERSATION_PANE_QUEUE,
    CONVERSATION_PANE_WIDTH,
    NAV_PANE_WIDTH,
    NO_TOKEN_SELECTED,
    PAGES,
    QUIET_TICK,
    STOP_LABEL,
    TRANSCRIPT_LABEL,
    show_page,
)
from chatlab.ui.conversations import (
    delete_fork,
    fork_conversation,
    new_conversation,
    remember_branch_sampling,
    remember_forks,
    remember_message,
    remember_transcript_message,
    restore_conversations,
    sampling_updates,
    save_conversation,
    switch_fork,
)
from chatlab.compare import (
    CONFIGURATION_HEADERS,
    EMPTY_SLOT as COMPARE_SLOT_EMPTY,
    DIVERGENCE_HEADERS,
    GAP_CAPTION,
    GAP_COLORS,
    MEASUREMENT,
    REPLY,
)
from chatlab.ui.activation_patching import build as build_activation_patching
from chatlab.ui.compare import (
    COMPARE_EMPTY,
    clear_slots,
    download_comparison,
    fill_slot,
    mode_controls,
    render as render_comparison,
    stop_comparison,
)
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
    load_with_steering,
    remember_steering,
    remove_vector,
    steering_updates,
    use_extracted,
)
from chatlab.ui.generation import (
    ask_clear_chat,
    branch_from,
    branch_with_text,
    chat,
    clear_chat,
    edit_message,
    hide_clear_confirm,
    next_token,
    retry_last,
    retry_message,
    stop_generation,
    undo_last,
    undo_message,
)
from chatlab.ui.inspection import (
    INSPECT_HINT,
    INSPECTION_CONTROLS,
    JACOBIAN_CSS,
    JACOBIAN_JS,
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
    go_to_models,
    loaded_model_badge,
    refresh_model_badge,
    refresh_model_switch,
    refresh_stale_model_switch,
    select_model_to_load,
)
from chatlab.ui.panel import (
    choose_alternative,
    empty_metrics,
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
from chatlab.ui.settings_page import (
    apply_theme,
    hardware_card,
    refresh_hardware,
    refresh_thinking_mode,
    remember_committed_seed,
    remember_prefill_limit,
    remember_settings,
    reset_sampling,
    restore_settings,
    sampling_label,
    update_sampling_label,
)
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    MENU_STRIP_CLASS,
    TOKEN_MENU_CSS,
    TOKEN_MENU_JS,
    branch_from_menu,
    edit_prompt_from_menu,
    menu_bridge_ids,
    prompt_menu_payload,
    token_menu_payload,
)
from chatlab.ui.styles import (
    COLUMN_JS,
    CSS,
    THEME,
    READ_ONLY_TEXT_JS,
    RESIZE_JS,
    SHORTCUT_JS,
    WRITING_SUGGESTIONS_JS,
    pane_handle,
    message_box_settings,
    set_message_box_keys,
)
from chatlab.ui.images_layout import build_images_page, wire_images_page
from chatlab.ui.models_layout import (
    ModelRefresh,
    build_models_page,
    wire_model_choice,
    wire_model_lists,
)


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
class Pages:
    """The nav and the columns it chooses between, in the order show_page() names them."""

    nav: gr.Radio
    conversations: gr.Column
    chat: gr.Column
    images: gr.Column
    models: gr.Column
    settings: gr.Column
    # Each enabled extension's page label and column, below the built-in pages.
    extensions: list


def build_app() -> gr.Blocks:
    # Read once, here, rather than per control: a build is one snapshot of
    # the file, and a control whose value came from a later read than its
    # neighbour's would be a puzzle to explain.
    saved = settings.load()
    settings.ensure_file()
    # Read the device beside the interface. Nothing here waits for it, and
    # the pages that describe a load - the fit verdicts in both model lists,
    # the hardware panel on the Settings page - are the fuller reading for it
    # by the time a reader looks.
    warm_device()
    extensions, extension_errors = load_enabled(saved.enabled_extensions)
    # The built-in pages keep fixed places in the nav and the extensions an
    # enabled build adds sit below them, above Settings. Spliced in among the
    # built-ins instead, an extension being enabled or removed would move
    # Images and Models up and down the pane under a reader who had learned
    # where they were.
    page_choices = [*PAGES[:-1], *(ext.spec.page_label for ext in extensions), PAGES[-1]]
    # Gradio otherwise caps the page at one of a handful of widths and centers
    # it, which leaves a band of empty room down each side on a wide screen.
    # The shell wants every pixel: the two side panes are a fixed width, so the
    # width the cap was holding back goes to the chat and the panel beside it.
    # analytics_enabled=False is the only thing here that is not about the
    # interface. Left at its default, Gradio posts to api.gradio.app twice on
    # the way up - once when this Blocks is built, once when it is launched -
    # with its version, the platform, and the list of component and event
    # types this app uses, and checks the package index for a newer Gradio it
    # can warn about. None of it carries what was said, and none of it is
    # wanted here: this is a local workbench for local models, often run with
    # no network at all, and its launch should not depend on reaching a host
    # on the internet. Set on the Blocks rather than through
    # GRADIO_ANALYTICS_ENABLED so it holds however the app is started - the
    # desktop bundle, run.sh, or python -m chatlab.
    with gr.Blocks(
        title="ChatLab", css=CSS + TOKEN_MENU_CSS + TREE_CSS + JACOBIAN_CSS + extension_css(extensions), theme=THEME, fill_width=True,
        analytics_enabled=False,
    ) as demo:
        # The chosen theme's colors, as a stylesheet on the page. Gradio fixes
        # THEME above when the interface is built, so a theme picked later is
        # a set of variables written over that one rather than another Blocks;
        # see the themes module. It is drawn first so nothing is painted in
        # the built-in colors and then repainted.
        theme_style = gr.HTML(
            themes.style_tag(saved.theme), elem_id="theme-style", padding=False
        )
        conversation_state = gr.State([])
        metrics_state = gr.State(empty_metrics())
        prompt_metrics_state = gr.State(empty_metrics())
        # What the Score text tab's strip is showing. The inspector's own
        # state moves on to the next reply; this one is rewritten only by
        # another scoring pass, so it still describes the passage drawn there.
        score_metrics_state = gr.State(empty_metrics())
        trace_state = gr.State({})
        # Branching from a token: the token last clicked - which turn, and
        # which of its tokens - and the alternative picked for it. Both name a
        # turn rather than a strip position, so a click keeps meaning what it
        # meant however the conversation moves under it.
        # The script names these after the strip they serve, so a second strip
        # elsewhere - the maze workbench's - carries its own three.
        request_id, response_id, action_id = menu_bridge_ids("token-strip")
        menu_request = gr.Textbox(elem_id=request_id, elem_classes=[MENU_BRIDGE_CLASS])
        menu_response = gr.HTML(elem_id=response_id, elem_classes=[MENU_BRIDGE_CLASS])
        menu_action = gr.Textbox(elem_id=action_id, elem_classes=[MENU_BRIDGE_CLASS])
        # The prompt strip carries its own three: the same menu, offering what
        # could have stood in a prompt position rather than a reply's.
        request_id, response_id, action_id = menu_bridge_ids("prompt-strip")
        prompt_menu_request = gr.Textbox(elem_id=request_id, elem_classes=[MENU_BRIDGE_CLASS])
        prompt_menu_response = gr.HTML(elem_id=response_id, elem_classes=[MENU_BRIDGE_CLASS])
        prompt_menu_action = gr.Textbox(elem_id=action_id, elem_classes=[MENU_BRIDGE_CLASS])
        selected_token = gr.State(None)
        branch_pick = gr.State(None)
        # Forking: the other transcripts, and the chatbot message last clicked.
        forks_state = gr.State(new_forks())
        tree_selection = gr.State({})
        selected_message = gr.State(None)
        token_edit_target = gr.State(None)
        # Layer inspection: the prompt ids behind the strips, the strip
        # position last clicked, and the last readout for re-rendering.
        context_ids_state = gr.State((*empty_metrics(), None))
        score_context_ids_state = gr.State((0, [], None))
        # The latest reply stays inspectable when scoring replaces the shared
        # prompt panel. Its measurements and exact input belong to the chat.
        chat_metrics_state = gr.State((0, []))
        chat_context_ids_state = gr.State((0, [], None))
        steering_state = gr.State(None)
        # What the last extraction read: one direction per decoder block, the
        # numbers beside each, and the load they were read through. Held
        # whole so that moving the layer control is instant - the pass that
        # reads one layer reads them all, and re-reading to change a layer
        # would cost another pass over every example.
        extract_state = gr.State(None)
        # The two comparison slots, and the document a download would write.
        # A slot holds one whole run - its tokens, its measurements and the
        # configuration it ran under - because the model that produced it may
        # be gone by the time the other slot is filled, which is the point.
        compare_a_state = gr.State(None)
        compare_b_state = gr.State(None)
        compare_export_state = gr.State(None)
        inspect_target = gr.State(None)
        insight_state = gr.State(None)
        # The prompts the last file gave, as it gave them. A prompt with a
        # blank line inside it reads as two once it is in the box, so a run
        # prefers this list while the box still holds what loading it wrote;
        # see ui.prompts.resolve_prompts().
        loaded_prompts_state = gr.State([])
        # Where the running batch writes its exports, so Stop can publish
        # what is there; see ui.prompts.stop_batch().
        batch_directory_state = gr.State(None)
        # Which load the scored token count on screen was counted against, so
        # a model swapped out from another tab can be told from this one.
        score_budget_load = gr.State(None)

        with gr.Row(elem_id="shell"):
            # The thin pane at the far left picks the page: Chat, Images,
            # Models, any extension pages, then Settings. The stylesheet stacks
            # the choices, rules off the extensions from the pages that ship
            # with the app, and pins Settings to the bottom.
            with gr.Column(scale=0, min_width=NAV_PANE_WIDTH, elem_id="nav-pane"):
                nav = gr.Radio(
                    choices=page_choices,
                    value=CHAT_PAGE,
                    show_label=False,
                    container=False,
                    elem_id="nav",
                )

            # The conversations pane sits beside the nav and shows with Chat only.
            with gr.Column(
                scale=0, min_width=CONVERSATION_PANE_WIDTH, elem_id="conversation-pane"
            ) as conversation_pane:
                gr.Markdown("## Conversations", elem_id="conversations-heading")
                conversation_list = gr.Radio(
                    choices=branch_choices(new_forks(), []),
                    value=MAIN_BRANCH,
                    show_label=False,
                    elem_id="conversation-list",
                )
                with gr.Row():
                    # The pane is narrow, so the buttons give up their usual
                    # minimum width to share one row.
                    new_button = gr.Button("New", size="sm", min_width=60, elem_classes=icon_classes("plus"))
                    fork_button = gr.Button("Fork", size="sm", min_width=60, elem_classes=icon_classes("git-branch"))
                    delete_fork_button = gr.Button("Delete", size="sm", min_width=60, elem_classes=icon_classes("trash"))
                # Named for what it takes: this empties the conversation on
                # screen and deletes every other one with it. It stands under
                # the list of everything it would take rather than under one
                # conversation's message box, where it read as a control of
                # that conversation alone. A fourth button would not fit the
                # row above, so it takes the pane's width on its own line.
                clear_button = gr.Button("Clear all", size="sm", elem_classes=icon_classes("trash"))
                with gr.Column(
                    visible=False,
                    elem_id="clear-confirm",
                    elem_classes=["clear-confirm"],
                ) as clear_confirm:
                    clear_question = gr.Markdown("")
                    # The pane is too narrow for the two answers to share a
                    # row, so they stack.
                    confirm_clear_button = gr.Button(
                        "Clear everything", variant="stop", size="sm"
                    )
                    cancel_clear_button = gr.Button("Cancel", size="sm")

            # The three pages share the rest of the width; one is visible at a
            # time, chosen by the nav.
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

                        with gr.Tabs(elem_id="conversation-tabs") as conversation_tabs:
                            with gr.Tab("Chat", elem_id="chat-tab"):
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
                                                inputs=trace_state,
                                                size="sm",
                                            )
                                            gr.DownloadButton(
                                                "Download CSV",
                                                value=lambda trace: write_trace_export(trace, "csv"),
                                                inputs=trace_state,
                                                size="sm",
                                            )

                            with gr.Tab("Score text"):
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

                            with gr.Tab("Prompts", elem_id="prompts-tab"):
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

                            with gr.Tab("Fork tree", elem_id="fork-tree-tab"):
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

                            with gr.Tab("Compare", elem_id="compare-tab"):
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
                                    inputs=compare_export_state,
                                    size="sm",
                                )
                                build_activation_patching(compare_a_state, compare_b_state)

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

            extension_pages = []
            extension_model_buttons = []
            navigation = NavigationService(
                lambda button, model_id: extension_model_buttons.append((button, model_id)))
            for extension in extensions:
                with gr.Column(scale=1, visible=False, elem_classes=["extension-page"]) as extension_page:
                    context = ExtensionContext(
                        models=ModelService(lambda: runtime.MANAGER), tokens=TokenInspector(),
                        data_dir=data_directory(extension.spec.id), navigation=navigation,
                    )
                    try:
                        extension.build_page(context)
                    except Exception as exc:
                        logging.getLogger(__name__).exception("Extension page failed: %s", extension.spec.id)
                        message = f"{extension.spec.title}: {exc}"
                        extension_errors.append(message)
                        gr.Markdown("This extension could not open. " + html.escape(message))
                extension_pages.append((extension.spec.page_label, extension_page))

            images = build_images_page(saved)

            models = build_models_page(saved)

            with gr.Column(
                scale=1, visible=False, elem_id="settings-page"
            ) as settings_page:
                # Sampling lives on the Chat page: temperature and the
                # response length are what a reader moves between one retry
                # and the next, and leaving the conversation to reach them
                # broke that loop. What is left here is what is set once and
                # then left alone.
                gr.Markdown(
                    "# Settings\nHow every reply is prompted and measured. The "
                    "sampling controls are on the Chat page, under the message "
                    "box, because they are moved between one reply and the next.",
                    elem_id="settings-hero",
                )
                # One card per subject, the same cards the Models page is
                # built from. The long text boxes take the left column on
                # their own; the short readings stack beside them, which is
                # what keeps either column from running far past the other.
                with gr.Row(elem_id="settings-columns"):
                    with gr.Column(min_width=360, elem_id="settings-prompting"):
                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Prompting")
                            system_prompt = gr.Textbox(
                                value=saved.system_prompt,
                                label="System prompt",
                                placeholder="You are a careful assistant that answers concisely.",
                                lines=3,
                                info="Sent as a system message ahead of the conversation. Leave empty to use the model's default behavior.",
                            )
                            assistant_prefill = gr.Textbox(
                                value=saved.assistant_prefill,
                                label="Assistant prefill (optional)",
                                placeholder="Start every reply with these exact words…",
                                lines=2,
                                info=(
                                    "Replays this text as the start of each answer, then lets the "
                                    "model continue. For reasoning models, ChatLab closes the "
                                    "reasoning block first so this remains visible answer text."
                                ),
                            )
                            thinking_mode = gr.Radio(
                                choices=THINKING_CHOICES,
                                value=saved.thinking_mode,
                                label="Thinking mode",
                                info=(
                                    "Applies to the next chat reply. Model default uses the model's "
                                    "normal behavior. Token branches keep the original reply's mode. "
                                    "An assistant prefill starts directly in the answer."
                                ),
                                visible=runtime.MANAGER.supports_thinking,
                            )
                            keep_reasoning = gr.Checkbox(
                                value=saved.keep_reasoning,
                                label="Send previous reasoning back to the model",
                                info="Off by default. Think models write a fresh reasoning block each turn, so replaying old ones burns context and usually hurts the next answer.",
                            )

                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Input")
                            enter_sends = gr.Checkbox(
                                value=saved.enter_sends,
                                label="Enter sends the message",
                                info="Shift+Enter starts a new line. Turn off to swap the two.",
                            )
                            writing_suggestions = gr.Checkbox(
                                value=saved.writing_suggestions,
                                label="Let the system suggest text while typing",
                                info=(
                                    "macOS offers the rest of a sentence in grey as you type, "
                                    "from its own predictions rather than the loaded model. "
                                    "Turn off to type without them."
                                ),
                            )
                            gr.Markdown(
                                "Escape stops a response that is still being written, "
                                "from anywhere on the Chat page - including the message "
                                "box and the Score text tab.",
                                elem_classes=["scale-caption"],
                            )

                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Analysis")
                            analyze_prompt = gr.Checkbox(
                                value=saved.analyze_prompt,
                                label="Measure prompt tokens",
                                info="Scores every prompt token during the same pass that warms the cache.",
                            )

                    with gr.Column(min_width=360, elem_id="settings-machine"):
                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Appearance")
                            theme_choice = gr.Dropdown(
                                choices=themes.THEME_CHOICES,
                                value=saved.theme,
                                label="Color theme",
                                info=(
                                    "The colors the whole interface is drawn in. "
                                    "Every one of them is drawn both light and "
                                    "dark."
                                ),
                            )
                            appearance_choice = gr.Radio(
                                choices=themes.APPEARANCE_CHOICES,
                                value=saved.appearance,
                                label="Light or dark",
                                info=(
                                    "Which of the two the chosen theme is drawn "
                                    "in. Following the system means the page "
                                    "turns with it, at whatever hour it does."
                                ),
                            )

                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Memory")
                            prefill_token_limit = gr.Number(
                                value=saved.prefill_token_limit,
                                precision=0,
                                minimum=settings.PREFILL_TOKEN_LIMIT_RANGE[0],
                                maximum=settings.PREFILL_TOKEN_LIMIT_RANGE[1],
                                label="Context limit (tokens)",
                                info=(
                                    "The most tokens one prompt may carry, and the "
                                    "ceiling on the response length. Every token in "
                                    "the conversation costs memory for as long as the "
                                    "answer runs, so this is the control to lower when "
                                    "a model runs out of it."
                                ),
                            )
                            gr.Markdown(
                                f"Settings are saved to `{settings.settings_path()}` as "
                                "you change them, and read from there at startup. The "
                                "Metal memory cap lives in that file as "
                                "`mps_memory_fraction`.",
                                elem_classes=["scale-caption"],
                            )

                        # What the memory guard is reading when it refuses a
                        # load. These figures were in the log alone, which
                        # made a refusal something to look up afterwards
                        # rather than something to check first.
                        with gr.Column(elem_classes=["settings-card"]):
                            gr.Markdown("## Hardware")
                            hardware_view = gr.Markdown(
                                hardware_card(),
                                elem_id="hardware",
                                elem_classes=["model-detail", "hardware-panel"],
                            )
                            with gr.Row(elem_id="hardware-footer"):
                                gr.Markdown(
                                    "Estimates, not guarantees: they are what ChatLab "
                                    "judges a load against, and each load and reply is "
                                    "recorded in the log with the same figures.",
                                    elem_classes=["scale-caption"],
                                )
                                refresh_hardware_button = gr.Button(
                                    "Refresh", size="sm", scale=0, min_width=110,
                                    elem_classes=icon_classes("refresh"),
                                )

                        with gr.Column(elem_classes=["settings-card"]):
                            extension_settings, active_extensions = build_extension_settings([ext.spec.id for ext in extensions], extension_errors)

        pages = Pages(
            nav=nav,
            conversations=conversation_pane,
            chat=chat_page,
            images=images.column,
            models=models.column,
            settings=settings_page,
            extensions=extension_pages,
        )

        nav.change(
            show_page,
            nav,
            [conversation_pane, chat_page, images.column, models.column, settings_page],
        )
        # On the way to the page rather than on a timer: nothing here changes
        # while it is not being looked at, and reading it costs a subprocess.
        nav.change(refresh_hardware, None, hardware_view)
        demo.load(refresh_hardware, None, hardware_view)
        refresh_hardware_button.click(refresh_hardware, None, hardware_view)
        demo.load(restore_extensions, active_extensions, extension_settings)
        for label, extension_page in extension_pages:
            def show_extension(page, expected=label):
                return gr.update(visible=page == expected)
            nav.change(show_extension, nav, extension_page)
        # Every page container go_to_models() publishes an update for, in the
        # order show_page() returns them.
        extension_page_outputs = [nav, conversation_pane, chat_page, images.column,
                                  models.column, settings_page,
                                  *(page for _, page in extension_pages)]
        # The ID box and everything that has to move with it, in the order
        # select_model_to_load() returns them.
        extension_model_outputs = [models.model_id, models.my_models, models.my_model_detail,
                                   models.search_selection, models.search_detail, models.model_status,
                                   models.remove_confirm, models.pending_removal]
        def open_models_from_extension():
            return (*go_to_models(), *(gr.update(visible=False) for _ in extension_pages))
        def open_named_model_from_extension(wanted):
            # An extension names the model its own view needs - a saved run
            # names the one that recorded it - and the reader still presses
            # Load. Nothing to name, or something no model ID could be, opens
            # the page with the box untouched rather than writing nonsense
            # into it.
            try:
                filled = select_model_to_load(wanted)
            except (ValueError, AttributeError):
                return (*(gr.skip() for _ in extension_model_outputs),
                        *open_models_from_extension())
            return (*filled, *(gr.update(visible=False) for _ in extension_pages))
        for button, wanted_model in extension_model_buttons:
            if wanted_model is None:
                button.click(open_models_from_extension, None, extension_page_outputs)
            else:
                button.click(open_named_model_from_extension, wanted_model,
                             [*extension_model_outputs, *extension_page_outputs])
        # The scored token count follows the boxes as they are typed into.
        # always_last coalesces a burst of keystrokes into the one count that
        # matters, and the progress bar is hidden because a spinner on every
        # keystroke would be worse than the number is good.
        score_budget_inputs = [score_context, score_input, use_chat_template]
        # The count travels with the load it was counted against; see
        # recover_score_budget for what that is for.
        score_budget_outputs = [score_budget, score_budget_load]
        for control in score_budget_inputs:
            control.change(
                score_token_count,
                score_budget_inputs,
                score_budget_outputs,
                trigger_mode="always_last",
                show_progress="hidden",
                concurrency_id=SCORE_BUDGET_QUEUE,
            )

        # The badge is refreshed on the way to the chat page as well, so a
        # load started a moment ago shows as one in progress rather than as
        # the "no model" state the page was left in.
        nav.change(refresh_thinking_mode, None, thinking_mode, show_progress="hidden")
        demo.load(refresh_thinking_mode, None, thinking_mode)
        badge_timer.tick(refresh_thinking_mode, None, thinking_mode, **QUIET_TICK)
        badge_outputs = [model_badge_view, default_model_button]
        nav.change(refresh_model_badge, None, badge_outputs)
        demo.load(refresh_model_badge, None, badge_outputs)
        # The switcher is drawn on the same two occasions. Its choices cost a
        # cache scan and a memory reading, so the timer only redraws it once
        # what it shows, what is on disk, or what would now fit has moved; see
        # refresh_stale_model_switch. Every draw hands back the stamp it read,
        # which is how the next tick knows the difference.
        switch_outputs = [model_switch, switch_stamp]
        nav.change(refresh_model_switch, models.weight_precision, switch_outputs)
        demo.load(refresh_model_switch, models.weight_precision, switch_outputs)
        badge_timer.tick(
            refresh_stale_model_switch,
            [model_switch, switch_stamp, models.weight_precision],
            switch_outputs,
            **QUIET_TICK,
        )
        # And on a timer, so a tab that did not start the load hears about it
        # too. demo.load stays: it draws the badge at once rather than leaving
        # the value baked in when the page was built there for a tick.
        # QUIET_TICK because this one runs on its own: the default fades a
        # handler's outputs in and out, which every couple of seconds would
        # have the badge flickering at a reader who never asked it anything.
        badge_timer.tick(refresh_model_badge, None, badge_outputs, **QUIET_TICK)
        # The same timer un-sticks the scored token count. A count asked for
        # during a reply gives up, and nothing about that message corrects
        # itself once the reply ends; see recover_score_budget, which is why
        # this is one listener rather than one on every path out of a
        # generation.
        badge_timer.tick(
            recover_score_budget,
            [score_budget, score_budget_load, *score_budget_inputs],
            score_budget_outputs,
            concurrency_id=SCORE_BUDGET_QUEUE,
            **QUIET_TICK,
        )
        wire_images_page(demo, images, nav, badge_timer)
        refresh = ModelRefresh(
            models, switch_outputs, badge_outputs, score_budget_inputs, score_budget_outputs,
            hardware_view, thinking_mode,
        )
        wire_model_lists(demo, pages, badge_timer, models, refresh, model_switch, model_badge_view)
        # The menu handles Escape before the global generation shortcut.
        demo.load(None, None, None, js=TOKEN_MENU_JS)
        demo.load(None, None, None, js=TREE_JS)
        demo.load(None, None, None, js=JACOBIAN_JS)
        tree_action.input(
            select_tree_branch,
            [tree_action, conversation_state, forks_state, tree_selection],
            [tree_selection, tree_view, tree_comparison],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            trigger_mode="always_last",
        )
        forks_state.change(
            render_fork_tree,
            [conversation_state, forks_state, tree_selection],
            [tree_view, tree_comparison],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            trigger_mode="always_last",
        )
        # Escape stops a running generation, from anywhere on the page.
        demo.load(None, None, None, js=SHORTCUT_JS)
        # The two readings panes are dragged wider or narrower by the handle
        # on their seam, and remember the width they were left at.
        demo.load(None, None, None, js=RESIZE_JS)
        # A table column is dragged wider by the seam on its header, for the
        # text Gradio's own measurement clips.
        demo.load(None, None, None, js=COLUMN_JS)
        # A box that only shows text is read-only rather than dead, so text
        # too long for it can be scrolled to and taken out.
        demo.load(None, None, None, js=READ_ONLY_TEXT_JS)
        # The system's own typing predictions, on or off from the first paint
        # and whenever the setting is changed after it. The change fires when
        # a reload restores the file's value as well as when it is clicked.
        demo.load(None, writing_suggestions, None, js=WRITING_SUGGESTIONS_JS)
        writing_suggestions.change(
            None, writing_suggestions, None, js=WRITING_SUGGESTIONS_JS
        )

        wire_model_choice(demo, pages, models, refresh, default_model_button, images.load_button)
        enter_sends.change(set_message_box_keys, enter_sends, prompt)

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
        sampling_controls = [temperature, top_p, top_k, skip_top_below, max_new_tokens]
        # In the order settings.CONVERSATION_SAMPLING names them, which is
        # what pairs each ↺ with the setting it restores.
        sampling_resets = [
            temperature_reset,
            top_p_reset,
            top_k_reset,
            skip_top_below_reset,
            max_new_tokens_reset,
        ]
        for control in sampling_controls:
            control.change(
                update_sampling_label,
                sampling_controls,
                sampling_accordion,
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
                [forks_state, *sampling_controls],
                forks_state,
                trigger_mode="always_last",
                show_progress="hidden",
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )

        steering_outputs = [
            steering_state, steering_enabled, steering_strength, steering_layer, steering_status,
        ]
        steering_upload.upload(
            import_vector, [steering_upload, forks_state], [forks_state, *steering_outputs],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        steering_remove.click(
            remove_vector, forks_state, [forks_state, *steering_outputs],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        extract_button.click(
            extract_vector,
            [extract_positive, extract_negative, extract_chat, extract_pool],
            [extract_state, extract_table, extract_layer, extract_apply, extract_status],
        )
        extract_table.select(
            choose_layer, extract_state, [extract_layer, extract_status]
        )
        # input rather than change: the extraction writes the layer control
        # itself, with a fuller status beside it, and a change listener would
        # fire on that write and replace the status with the shorter line.
        extract_layer.input(
            describe_layer,
            [extract_state, extract_layer],
            extract_status,
            show_progress="hidden",
        )
        extract_apply.click(
            use_extracted,
            [forks_state, extract_state, extract_layer],
            [forks_state, *steering_outputs],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        for control in (steering_enabled, steering_strength, steering_layer):
            control.input(
                remember_steering,
                [forks_state, steering_state, steering_enabled, steering_strength, steering_layer],
                [forks_state, steering_state, steering_status],
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
                forks_state,
                sampling_controls,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                sampling_controls,
                sampling_accordion,
                concurrency_id=SAMPLING_LABEL_QUEUE,
            ).then(
                steering_updates, forks_state, steering_outputs,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            skip_top_below,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            color_scale,
        ]
        # Persistence runs separately; every request must snapshot the controls
        # the reader sees, even while remember_steering is still queued.
        steering_inputs = [steering_state, steering_enabled, steering_strength, steering_layer]
        chat_inputs = [prompt, conversation_state, *settings_inputs, *steering_inputs, thinking_mode]

        # Everything saved between sessions, in PERSISTED_SETTING_NAMES order.
        persisted_inputs = [
            *settings_inputs,
            thinking_mode,
            enter_sends,
            writing_suggestions,
            theme_choice,
            appearance_choice,
            models.model_id,
            models.weight_precision,
        ]
        for control in (
            thinking_mode,
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            randomize_seed,
            analyze_prompt,
            color_scale,
            enter_sends,
            writing_suggestions,
            models.model_id,
            models.weight_precision,
        ):
            control.change(remember_settings, persisted_inputs, None)
        # The theme is wired apart from the loop above because it takes two
        # listeners rather than one: saving it, and repainting the page, which
        # has to follow the dropdown whether the change came from the reader
        # or from the file being read back on a reload.
        #
        # always_last on both, and on both for the same reason. A reader
        # trying the themes on picks one while the one before it is still in
        # flight, and with Gradio's default the pick behind is dropped. On one
        # listener alone that is worse than on neither: the page would be
        # painted in the theme last picked while the file kept an earlier one,
        # and a reload would undo a choice that was there on screen.
        theme_choice.change(
            remember_settings, persisted_inputs, None, trigger_mode="always_last"
        )
        theme_choice.change(
            apply_theme,
            theme_choice,
            theme_style,
            trigger_mode="always_last",
        )
        # Light or dark is wired the same way and for the same reasons, except
        # that the repaint is the browser's own work rather than a round trip:
        # the class it toggles is already what every dark-mode rule reads.
        appearance_choice.change(
            remember_settings, persisted_inputs, None, trigger_mode="always_last"
        )
        appearance_choice.change(None, appearance_choice, None, js=themes.APPEARANCE_JS)
        # The four that belong to a conversation are saved on input, like the
        # write into the conversation itself. Switching conversations sets
        # them, and a save from that would put the sampling of the
        # conversation merely being looked at into the settings file - which
        # is what an unpinned conversation answers with, so looking at a
        # branch pinned to temperature 0 would quietly move every unpinned
        # one to 0 as well.
        # On the conversation queue, so the file is written before a switch
        # that follows reads it: a conversation carrying no sampling of its
        # own answers with what that file says, and a slider moved and then a
        # switch in quick succession must not read the older value.
        #
        # always_last for the same reason the branch write has it, and more
        # so now that this shares a queue: with Gradio's default, a slider
        # still moving while this is pending drops the newer values and the
        # file keeps one from part way through the drag.
        for control in sampling_controls:
            control.input(
                remember_settings,
                persisted_inputs,
                None,
                trigger_mode="always_last",
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        # A ↺ writes itself down the way a hand on that slider does, and in
        # the same order: the control first, then the conversation and the
        # file from what the five now hold, then the summary. The handlers
        # are the ones the sliders already use, so a reset is stored, pinned
        # and described exactly as the same move by hand would have been -
        # there is nothing about it for them to tell apart.
        for button, name, control in zip(
            sampling_resets, settings.CONVERSATION_SAMPLING, sampling_controls, strict=True
        ):
            button.click(
                partial(reset_sampling, name, saved.prefill_token_limit),
                None,
                control,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                remember_branch_sampling,
                [forks_state, *sampling_controls],
                forks_state,
                show_progress="hidden",
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                remember_settings,
                persisted_inputs,
                None,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                sampling_controls,
                sampling_accordion,
                show_progress="hidden",
                concurrency_id=SAMPLING_LABEL_QUEUE,
            )
        # The seed box is the one control the app writes to itself: a finished
        # response leaves the seed that produced it there, and saving that
        # would overwrite the seed the reader chose. Blur and submit are the
        # two ways a person is done editing a number, and they are the only
        # events that write the box's contents down; every other control
        # leaves the saved seed where it is. See remember_settings().
        for event in (seed.blur, seed.submit):
            event(remember_committed_seed, persisted_inputs, None)
        # The context limit is committed rather than saved as it is typed: the
        # handler writes a clamped value back, which mid-word would fight the
        # typing.
        # Lowering it can pull the response length down with it, which the
        # sampling summary names, so the label follows that too.
        # A lowered limit can pull the response length down with it, which
        # the sampling summary names and the conversation on screen has to
        # be told about - that clamp is the reader's own doing, and the
        # conversation would otherwise put the longer length back the next
        # time it was switched to. The handler writes the conversation only
        # when it actually clamped, so a limit tabbed through or raised does
        # not pin a conversation that was following the settings file.
        for event in (prefill_token_limit.blur, prefill_token_limit.submit):
            event(
                remember_prefill_limit,
                [prefill_token_limit, max_new_tokens, forks_state, *sampling_controls],
                [prefill_token_limit, max_new_tokens, forks_state],
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                sampling_controls,
                sampling_accordion,
                concurrency_id=SAMPLING_LABEL_QUEUE,
            )
        # A page load is where the file is read back, so reloading the browser
        # shows what was saved rather than what the app started with. The
        # sampling summary is rebuilt from whatever came back, since the label
        # the accordion was built with describes the file as it was read at
        # startup, not as it is now.
        demo.load(
            restore_settings, None, [*persisted_inputs, prefill_token_limit]
        ).then(
            apply_theme, theme_choice, theme_style
        ).then(
            None, appearance_choice, None, js=themes.APPEARANCE_JS
        ).then(
            update_sampling_label,
            sampling_controls,
            sampling_accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
        # The one table from the names the conversation handlers publish
        # under to the components those names are drawn in; see ui.outputs.
        # Every listener bound through conversation_events names its outputs
        # from here, so a handler and its listener cannot disagree about
        # which value goes where.
        conversation_outputs = {
            "prompt": prompt,
            "chatbot": chatbot,
            "turns": conversation_state,
            "strip": token_strip,
            "metrics": metrics_state,
            "status": generation_status,
            "seed": seed,
            "send": send_button,
            "stop": stop_button,
            "detail": token_detail,
            "alternatives": alternatives,
            "prompt_strip": prompt_strip,
            "prompt_metrics": prompt_metrics_state,
            "prompt_note": prompt_note,
            "summary": summary_panel,
            "surprise": surprise_panel,
            "trace": trace_state,
            "context_ids": context_ids_state,
            "chat_metrics": chat_metrics_state,
            "chat_context_ids": chat_context_ids_state,
            "selected_token": selected_token,
            "branch_pick": branch_pick,
            "token_editor": token_editor,
            "token_edit_target": token_edit_target,
            "forks": forks_state,
            "conversation_list": conversation_list,
            "clear_confirm": clear_confirm,
            "branch_text": branch_text,
            "system_prompt": system_prompt,
            "steering_state": steering_state,
            "steering_enabled": steering_enabled,
            "steering_strength": steering_strength,
            "steering_layer": steering_layer,
            "steering_status": steering_status,
        }
        # Where tests, and anything else holding the page, look a component
        # up by the name its handlers publish it under.
        demo.conversation_outputs = conversation_outputs

        background_state = gr.State(ConversationJob())
        conversation_events = ConversationEvents(
            background_state, conversation_outputs, color_scale, CONVERSATION_PANE_QUEUE,
        )
        response_timer = gr.Timer(0.25)
        response_timer.tick(
            conversation_events.poll,
            [background_state, conversation_state, forks_state, color_scale],
            conversation_events.components(POLL_OUTPUT_NAMES),
            concurrency_id=CONVERSATION_PANE_QUEUE,
            **QUIET_TICK,
        )

        start_response = partial(conversation_events.bind, generation=True)
        navigate = partial(conversation_events.bind, navigation=True)
        start_response(menu_action.input, branch_from_menu, [menu_action, *chat_inputs], CHAT_OUTPUT_NAMES)
        start_response(
            prompt_menu_action.input, edit_prompt_from_menu,
            [prompt_menu_action, context_ids_state, prompt_metrics_state, *chat_inputs],
            CHAT_OUTPUT_NAMES,
        )
        start_response(send_button.click, chat, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(prompt.submit, chat, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(retry_button.click, retry_last, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(next_token_button.click, next_token, [branch_pick, *chat_inputs], CHAT_OUTPUT_NAMES)
        start_response(chatbot.retry, retry_message, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(chatbot.edit, edit_message, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(
            token_edit_save.click, save_token_edit,
            [token_edit_target, token_edit_text, *chat_inputs],
            TOKEN_EDIT_OUTPUT_NAMES,
        )
        start_response(
            branch_button.click, branch_from, [branch_pick, *chat_inputs], CHAT_OUTPUT_NAMES,
        )
        start_response(
            branch_text_button.click, branch_with_text,
            [selected_token, branch_text, *chat_inputs], CHAT_OUTPUT_NAMES,
        )

        conversation_events.bind(
            stop_button.click, stop_generation,
            inputs=[conversation_state, color_scale],
            outputs=STOP_OUTPUT_NAMES,
            stop=True,
        )

        # Mutations of a running conversation ask the reader to Stop first.
        # Navigation and changes to other conversations leave its job alone.
        conversation_events.bind(
            undo_button.click, undo_last,
            [conversation_state, color_scale],
            UNDO_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        conversation_events.bind(
            chatbot.undo, undo_message,
            [conversation_state, color_scale],
            UNDO_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # Clear asks before it takes anything, so the button that opens the
        # question leaves the conversations and background job alone. The
        # confirm button clears them once generation is stopped.
        clear_button.click(
            ask_clear_chat,
            [conversation_state, forks_state],
            [generation_status, clear_confirm, clear_question],
        )
        cancel_clear_button.click(hide_clear_confirm, None, clear_confirm)
        # The question names how many conversations it would take, and that
        # count is read when it is asked. Anything that adds or removes one
        # withdraws it rather than leaving a stale promise above a button
        # that would take more than the promise says - the same reason
        # choosing another model withdraws the removal question. Pressing
        # Clear again re-asks with the numbers as they are now.
        for control in (new_button, fork_button, delete_fork_button):
            control.click(hide_clear_confirm, None, clear_confirm)
        conversation_list.input(hide_clear_confirm, None, clear_confirm)
        brings_its_sampling(conversation_events.bind(
            confirm_clear_button.click, clear_chat,
            clear=True,
            inputs=[color_scale, forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            outputs=CLEAR_OUTPUT_NAMES,
        ))

        chatbot.select(remember_message, conversation_state, selected_message)

        # Navigation takes a snapshot of the view; the job keeps its source.
        brings_its_sampling(
            navigate(
                fork_button.click, fork_conversation,
                [
                    conversation_state,
                    forks_state,
                    selected_message,
                    color_scale,
                    *sampling_controls,
                ],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            navigate(
                new_button.click, new_conversation,
                [conversation_state, forks_state, color_scale, *sampling_controls],
                NEW_CONVERSATION_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        # .input rather than .change: the list is also redrawn by the handlers
        # above and the listener below, and a .change listener would switch a
        # second time on each.
        brings_its_sampling(
            navigate(
                conversation_list.input, switch_fork,
                [conversation_list, conversation_state, forks_state, color_scale],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            conversation_events.bind(
                delete_fork_button.click, delete_fork,
                [conversation_state, forks_state, color_scale],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        # Every other path that changes the conversation lands here, and
        # the list's model tag, running indicator and token count
        # follow it. Hide the loading overlay so each streaming frame updates
        # the labels without making the whole list blink.
        conversation_state.change(
            conversation_events.refresh_conversation_list,
            [conversation_state, forks_state, background_state],
            [conversation_list, forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            show_progress="hidden",
        )
        # And the forks' change, which the listener above fires in turn, is
        # where ordinary view changes are saved. Workers also save independently.
        forks_state.change(
            remember_forks,
            [conversation_state, forks_state],
            None,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # The saved conversations come back first, so the listeners above
        # have something to describe. A page with nothing saved is left as
        # it was built. Background workers persist to their source independently.
        brings_its_sampling(
            conversation_events.bind(
                demo.load, restore_conversations,
                None,
                RESTORE_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )

        save_button.click(
            save_conversation,
            [conversation_state, system_prompt, *steering_inputs],
            [saved_file, generation_status],
        )
        conversation_events.bind(
            load_upload.upload, load_with_steering,
            [load_upload, conversation_state, color_scale, forks_state],
            STEERED_LOAD_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

        score_button.click(
            score_text,
            [score_context, score_input, use_chat_template, color_scale],
            [
                score_strip,
                metrics_state,
                score_metrics_state,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                score_status,
                token_detail,
                alternatives,
                selected_token,
                branch_pick,
                context_ids_state,
                score_context_ids_state,
            ],
        )
        # A batch reads its prompts from the box and everything else from the
        # controls the Chat tab and Settings already own, so there is nothing
        # to set up before running one.
        batch_outputs = [
            batch_status,
            batch_results,
            run_prompts_button,
            stop_prompts_button,
            batch_files,
            batch_directory_state,
        ]
        batch_run = run_prompts_button.click(
            run_prompts,
            [
                prompts_box,
                loaded_prompts_state,
                system_prompt,
                assistant_prefill,
                temperature,
                top_p,
                top_k,
                skip_top_below,
                max_new_tokens,
                seed,
                randomize_seed,
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
        stop_prompts_button.click(
            stop_batch, batch_directory_state, batch_outputs, cancels=[batch_run]
        )
        # ------------------------------------------------------------ Compare
        compare_outputs = [
            compare_a_heading,
            compare_b_heading,
            compare_a_strip,
            compare_b_strip,
            compare_tiles,
            compare_chart,
            compare_headline,
            compare_settings,
            compare_rows,
            compare_export_state,
        ]
        # Everything a run reads. The sampling controls and the steering
        # vector are the Chat tab's own, so a slot is filled under exactly
        # the settings a reply typed by hand would have used - which is what
        # makes changing one of them between A and B a clean experiment.
        compare_inputs = [
            compare_mode,
            compare_prompt,
            compare_text,
            compare_template,
            system_prompt,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            skip_top_below,
            max_new_tokens,
            seed,
            randomize_seed,
            thinking_mode,
            *steering_inputs,
        ]
        compare_slot_outputs = [compare_status, compare_run_a, compare_run_b, compare_stop]
        compare_runs = []
        for slot, button, held in (
            ("A", compare_run_a, compare_a_state),
            ("B", compare_run_b, compare_b_state),
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
                [compare_a_state, compare_b_state],
                compare_outputs,
            )
            compare_runs.append(filling)
        paired = pair_button.click(
            experiment_compare.run_pair, [*pair_conditions, *compare_inputs],
            [compare_a_state, compare_b_state, *compare_slot_outputs, pair_button, pair_load_status],
        )
        compare_runs.append(paired)
        next_difference.click(experiment_compare.next_difference,
                              [compare_export_state, difference_position],
                              [difference_position, difference_detail])
        compare_export_state.change(lambda: (None, ""), None, [difference_position, difference_detail],
                                    show_progress="hidden")
        experiments.wire(experiments_view, demo, trace_state, chat_context_ids_state,
                         compare_a_state, compare_b_state, insight_state, inspect_target)
        conversation_tabs.select(experiments.inspector_visibility, None, [inspector_pane, inspector_resizer],
                                 show_progress="hidden", queue=False)
        for held in (compare_a_state, compare_b_state):
            held.change(render_comparison, [compare_a_state, compare_b_state], compare_outputs)
        # Cancelling closes the run at its last yield, which is what gives the
        # model lock back; this only puts the buttons right. The slot keeps
        # whatever it held before, because half a response is not a run.
        compare_stop.click(
            stop_comparison, None, compare_slot_outputs, cancels=compare_runs
        ).then(lambda: gr.update(interactive=True), None, pair_button)
        # cancels, because clearing during a run is otherwise undone by the
        # run: its remaining frames would write over the cleared status and
        # its last one would put the slot back. See clear_slots().
        compare_clear.click(
            clear_slots,
            None,
            [compare_a_state, compare_b_state, *compare_slot_outputs, *compare_outputs],
            cancels=compare_runs,
        ).then(lambda: gr.update(interactive=True), None, pair_button)
        compare_mode.change(
            mode_controls,
            compare_mode,
            [compare_prompt, compare_text, compare_template],
        )

        prompts_box.change(
            count_prompts,
            [prompts_box, loaded_prompts_state],
            prompt_count,
            trigger_mode="always_last",
            show_progress="hidden",
        )
        prompts_upload.upload(
            load_prompt_file,
            [prompts_upload, prompts_box, loaded_prompts_state],
            [prompts_box, batch_status, loaded_prompts_state],
        )
        # The conversation is painted from the turns; the two strips are
        # painted from the measurements they were handed.
        color_scale.change(
            recolor,
            [conversation_state, score_metrics_state, prompt_metrics_state, color_scale],
            [token_strip, score_strip, prompt_strip, scale_caption],
        )
        # Which view of the conversation is on screen. The token view is
        # redrawn on the way in rather than left to the next frame, since the
        # conversation may have moved on while it was hidden.
        token_view.change(
            partial(show_token_view, conversation_id=conversation_state._id),
            [token_view, conversation_state, color_scale],
            [chatbot, token_strip],
            show_progress="hidden",
        )
        editor_outputs = [token_editor, token_edit_text, token_edit_target]
        token_view.change(close_token_editor, outputs=editor_outputs, show_progress="hidden")
        token_edit_cancel.click(close_token_editor, outputs=editor_outputs, show_progress="hidden")
        token_strip.select(
            open_token_editor,
            [conversation_state, metrics_state],
            editor_outputs,
            show_progress="hidden",
        )

        # One click in the conversation answers every question the inspector
        # asks of it, so it is one listener rather than four.
        token_strip.select(
            select_transcript_token,
            [conversation_state, metrics_state],
            [token_detail, alternatives, selected_token, inspect_target, branch_pick],
        )
        token_strip.select(
            token_menu_payload,
            [conversation_state, metrics_state, menu_request],
            menu_response,
            show_progress="hidden",
            queue=False,
        )
        # Also keep the message it landed in, so Fork works from
        # the token view exactly as it does from the chatbot.
        token_strip.select(
            remember_transcript_message, conversation_state, selected_message
        )

        for strip, strip_metrics, source, where in (
            (score_strip, score_metrics_state, "score", "score"),
            (prompt_strip, prompt_metrics_state, "prompt", "prompt"),
        ):
            strip.select(
                inspect_token(source),
                inputs=strip_metrics,
                outputs=[token_detail, alternatives],
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
                [selected_token, branch_pick],
            )
            strip.select(remember_inspect_target(where), strip_metrics, inspect_target)
        # Only the prompt strip's tokens can be replaced: they are the ones a
        # reply was actually generated from, and the ids behind them are kept.
        prompt_strip.select(
            prompt_menu_payload,
            [prompt_metrics_state, context_ids_state, prompt_menu_request],
            prompt_menu_response,
            show_progress="hidden",
            queue=False,
        )
        alternatives.select(
            choose_alternative,
            [conversation_state, score_metrics_state, prompt_metrics_state, selected_token],
            [token_detail, branch_pick],
        )
        inspection_outputs = [lens_panel, attention_panel, insight_state, inspect_status]
        inspect_button.click(
            inspect_layers,
            [
                inspect_target,
                metrics_state,
                prompt_metrics_state,
                context_ids_state,
                attention_layer,
                score_metrics_state,
                score_context_ids_state,
                chat_metrics_state,
                chat_context_ids_state,
                lens_mode,
                imported_lens,
                pinned_concept,
                inspection_session,
                pinned_token_id,
            ],
            [lens_panel, attention_panel, attention_layer, insight_state, inspect_status],
        )
        attention_layer.release(
            render_attention, [insight_state, attention_layer], attention_panel
        )
        # Every readout, and every reset that takes one away, passes through
        # the insight state, so the cache view follows it there. The cache is
        # read from memory rather than rebuilt, so each control reads again.
        kv_inputs = [insight_state, kv_layer, kv_metric]
        insight_state.change(render_kv_cache, kv_inputs, [kv_panel, kv_layer], **QUIET_TICK)
        kv_layer.release(render_kv_cache, kv_inputs, [kv_panel, kv_layer])
        kv_metric.change(render_kv_cache, kv_inputs, [kv_panel, kv_layer])
        lens_mode.change(
            change_lens_mode, [lens_mode, inspection_session],
            [jacobian_controls, attention_layer, *inspection_outputs],
            queue=False,
        )
        import_lens_button.click(
            import_jacobian_lens, [lens_file, fitted_model_id, lens_repository, lens_filename],
            [imported_lens, import_lens_status],
        )
        # QUIET_TICK for the same reason as the metrics_state handler below:
        # clearing a stale readout is instant, so the panel should not flash a
        # spinner over itself on the way.
        imported_lens.change(
            reset_inspection, insight_state, inspection_outputs, **QUIET_TICK,
        )
        imported_lens.change(
            lambda imported: gr.update(open=False) if imported else gr.skip(),
            imported_lens, lens_setup,
        )
        pinned_concept.input(
            change_pinned_token, [pinned_concept, inspection_session], inspection_outputs,
            queue=False,
        )
        # Every path that redraws the strips writes the metrics state, so this
        # is where a readout of a token that is no longer on screen goes away.
        #
        # Streaming writes that state on every frame, and reset_inspection
        # skips its outputs once there is nothing left to clear. Gradio marks
        # them pending regardless, so without QUIET_TICK the inspector blinks
        # its way through every reply: see the note above for what the two
        # arguments each take away.
        metrics_state.change(
            reset_inspection, insight_state, inspection_outputs, **QUIET_TICK,
        )
    return demo
