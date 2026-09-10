"""The page itself: every control and how the handlers are wired to them."""

from __future__ import annotations

import logging
import html

import gradio as gr

import charts
import settings
from conversation import (
    MAIN_BRANCH,
    branch_choices,
    new_forks,
)
from model_runtime import (
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MODEL_SORT_ORDERS,
    warm_device,
)
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
)
from trace_export import write_trace_export
from ui import runtime
from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from extensions.registry import load_enabled
from ui.extensions_page import build_extension_settings, data_directory, extension_css, restore_extensions
from ui.common import (
    CHAT_PAGE,
    CONVERSATION_PANE_WIDTH,
    NAV_PANE_WIDTH,
    NO_TOKEN_SELECTED,
    PAGES,
    RESPONSE_STRIP_LABEL,
    show_page,
    status_card,
)
from token_metrics import (
    PROMPT_ATTENTION_SCALE,
)
from ui.conversations import (
    delete_fork,
    fork_conversation,
    load_conversation,
    new_conversation,
    refresh_conversation_list,
    remember_branch_sampling,
    remember_forks,
    remember_message,
    restore_conversations,
    sampling_updates,
    save_conversation,
    switch_fork,
)
from ui.images_page import (
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
from ui.generation import (
    ask_clear_chat,
    branch_from,
    branch_with_text,
    chat,
    clear_chat,
    edit_message,
    hide_clear_confirm,
    retry_last,
    retry_message,
    stop_generation,
    undo_last,
    undo_message,
)
from ui.inspection import (
    INSPECT_HINT,
    inspect_layers,
    remember_inspect_target,
    render_attention,
    reset_inspection,
)
from ui.models_page import (
    BADGE_REFRESH_SECONDS,
    SEARCH_HINT,
    SEARCH_KINDS,
    ask_remove_my_model,
    clear_my_model_selection,
    download_and_load_model,
    download_model,
    go_to_models,
    hide_remove_confirm,
    load_cached_model,
    loaded_model_badge,
    redownload_my_model,
    refresh_after_device,
    refresh_image_badge,
    refresh_model_actions,
    refresh_model_badge,
    refresh_my_models,
    refresh_search_results,
    remove_my_model,
    search_models,
    select_default_model,
    select_my_model,
    select_search_result,
    unload_model,
)
from ui.panel import (
    choose_alternative,
    empty_metrics,
    inspect_token,
    recolor,
    remember_selection,
)
from ui.prompts import (
    BATCH_HEADERS,
    PROMPT_COUNT_HINT,
    count_prompts,
    load_prompt_file,
    run_prompts,
    stop_batch,
)
from ui.scoring import (
    SAMPLING_LABEL_QUEUE,
    SCORE_BUDGET_QUEUE,
    SCORE_COUNT_HINT,
    recover_score_budget,
    score_text,
    score_token_count,
)
from ui.settings_page import (
    hardware_card,
    refresh_hardware,
    remember_committed_seed,
    remember_prefill_limit,
    remember_settings,
    restore_settings,
    sampling_label,
    update_sampling_label,
)
from ui.styles import (
    CSS,
    THEME,
    SHORTCUT_JS,
    message_box_settings,
    set_message_box_keys,
)


# One queue for everything that rewrites the forks or the conversation in one
# step: the branch buttons, the list, Clear all, Undo, Stop, the loaders, and
# the two listeners on the states. Gradio
# runs events that share a concurrency id one at a time, in the order they
# were queued, and reads a State input when the event runs rather than when
# it was queued. So a redraw queued by a streaming frame can no longer run
# after a click on New with the forks as they were before the click, and
# hand that older pane back over the new one. The generation handlers stay
# out of it: they hold their own slot for the whole reply, and the redraw
# has to run between their frames.
CONVERSATION_PANE_QUEUE = "conversation-pane"


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
    page_choices = [PAGES[0], *(ext.spec.page_label for ext in extensions), *PAGES[1:]]
    # Gradio otherwise caps the page at one of a handful of widths and centers
    # it, which leaves a band of empty room down each side on a wide screen.
    # The shell wants every pixel: the two side panes are a fixed width, so the
    # width the cap was holding back goes to the chat and the panel beside it.
    with gr.Blocks(
        title="ChatLab", css=CSS + extension_css(extensions), theme=THEME, fill_width=True
    ) as demo:
        conversation_state = gr.State([])
        metrics_state = gr.State(empty_metrics())
        prompt_metrics_state = gr.State(empty_metrics())
        trace_state = gr.State({})
        # Branching from a token: the stamp of the last chat response's strip,
        # the strip position last clicked, and the alternative picked for it.
        branch_source = gr.State(None)
        selected_token = gr.State(None)
        branch_pick = gr.State(None)
        # Forking: the other transcripts, and the chatbot message last clicked.
        forks_state = gr.State(new_forks())
        selected_message = gr.State(None)
        # Layer inspection: the prompt ids behind the strips, the strip
        # position last clicked, and the last readout for re-rendering.
        context_ids_state = gr.State((*empty_metrics(), None))
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
            # The thin pane at the far left picks the page: Chat, Models, or
            # Settings. The stylesheet stacks the choices and pins Settings to
            # the bottom.
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
                    new_button = gr.Button("➕ New", size="sm", min_width=60)
                    fork_button = gr.Button("🌿 Fork", size="sm", min_width=60)
                    delete_fork_button = gr.Button("🗑️ Delete", size="sm", min_width=60)

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
                        # say which model would answer. Beside it, while none is
                        # loaded, are links to set up the default or choose another
                        # model on the Models page.
                        with gr.Row(elem_id="model-bar"):
                            model_badge_view = gr.HTML(
                                loaded_model_badge(), elem_id="model-badge"
                            )
                            default_model_button = gr.Button(
                                "Set up the default model",
                                variant="primary",
                                size="sm",
                                visible=not runtime.MANAGER.loaded,
                                elem_id="default-model",
                            )
                            load_model_button = gr.Button(
                                "Choose another",
                                size="sm",
                                visible=not runtime.MANAGER.loaded,
                                elem_id="load-model",
                            )

                        # Nothing to see: the timer is what makes the badge tell every
                        # open tab about a load or unload, not just the one that asked
                        # for it. See BADGE_REFRESH_SECONDS.
                        badge_timer = gr.Timer(BADGE_REFRESH_SECONDS)

                        with gr.Tabs(elem_id="conversation-tabs"):
                            with gr.Tab("Chat", elem_id="chat-tab"):
                                chatbot = gr.Chatbot(
                                    type="messages",
                                    label="Conversation",
                                    height=560,
                                    show_label=False,
                                    elem_id="conversation",
                                    editable="all",
                                    placeholder="Load a model, then start a conversation.",
                                )
                                prompt = gr.Textbox(
                                    label="Message",
                                    show_label=False,
                                    elem_id="message-input",
                                    **message_box_settings(saved.enter_sends),
                                )
                                with gr.Row():
                                    send_button = gr.Button("Send", variant="primary", min_width=70)
                                    # Escape presses this; see SHORTCUT_JS,
                                    # which finds it by this id.
                                    stop_button = gr.Button(
                                        "Stop",
                                        variant="stop",
                                        visible=False,
                                        elem_id="stop-button",
                                    )
                                    # The three give up their usual minimum
                                    # width to stay on Send's row. Left to
                                    # wrap, the last of them takes a line of
                                    # its own and reads as the widest, most
                                    # important button under the box.
                                    retry_button = gr.Button("🔁 Retry", min_width=80)
                                    undo_button = gr.Button("↩️ Undo last", min_width=90)
                                    # Named for what it takes: this empties
                                    # the conversation on screen and deletes
                                    # every other one with it.
                                    clear_button = gr.Button("🗑️ Clear all", min_width=90)
                                with gr.Column(
                                    visible=False,
                                    elem_id="clear-confirm",
                                    elem_classes=["clear-confirm"],
                                ) as clear_confirm:
                                    clear_question = gr.Markdown("")
                                    with gr.Row():
                                        confirm_clear_button = gr.Button(
                                            "Clear everything", variant="stop", size="sm"
                                        )
                                        cancel_clear_button = gr.Button(
                                            "Cancel", size="sm"
                                        )

                                generation_status = gr.Markdown("Ready.", elem_id="generation-status")
                                with gr.Accordion("Conversation tools", open=False, elem_id="conversation-tools"):
                                    # Sampling and file controls are available on demand.
                                    with gr.Accordion(
                                        sampling_label(
                                            saved.temperature,
                                            saved.top_p,
                                            saved.top_k,
                                            saved.max_new_tokens,
                                        ),
                                        open=False,
                                    ) as sampling_accordion:
                                        with gr.Row():
                                            temperature = gr.Slider(
                                                0,
                                                2,
                                                value=saved.temperature,
                                                step=0.05,
                                                label="Temperature",
                                            )
                                            top_p = gr.Slider(
                                                0.05,
                                                1,
                                                value=saved.top_p,
                                                step=0.01,
                                                label="Top-p",
                                            )
                                        with gr.Row():
                                            top_k = gr.Slider(
                                                0,
                                                200,
                                                value=saved.top_k,
                                                step=1,
                                                label="Top-k (0 disables)",
                                            )
                                            # The ceiling is the context limit: a
                                            # response cannot be longer than a
                                            # prompt is allowed to be.
                                            max_new_tokens = gr.Slider(
                                                1,
                                                saved.prefill_token_limit,
                                                value=saved.max_new_tokens,
                                                step=1,
                                                label="Maximum new tokens",
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
                                                label="🎲 New seed each response",
                                                info="Turn off to lock the seed and reproduce a response exactly.",
                                            )
                                    with gr.Row():
                                        save_button = gr.Button("💾 Save conversation")
                                        load_upload = gr.UploadButton(
                                            "📂 Load conversation",
                                            file_types=[".json"],
                                            type="filepath",
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
                                        "📂 Load prompts",
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

                    with gr.Column(scale=2, min_width=300, elem_id="inspector-pane"):
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
                        token_strip = gr.HighlightedText(
                            label=RESPONSE_STRIP_LABEL,
                            color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                            show_legend=True,
                            combine_adjacent=False,
                            elem_id="token-strip",
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
                                branch_button = gr.Button("🌱 Branch from token", size="sm")
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
                                    "✏️ Branch with text", size="sm", scale=0, min_width=160
                                )
                        with gr.Accordion("Layers and attention", open=False, elem_classes=["inspector-section"]):
                            with gr.Row():
                                inspect_button = gr.Button(
                                    "🔬 Inspect layers", size="sm", scale=0, min_width=160
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
                        with gr.Accordion("Response statistics", open=False, elem_classes=["inspector-section"]):
                            summary_panel = gr.HTML(charts.summary_tiles({}))
                            surprise_panel = gr.HTML(charts.EMPTY_CHART)
                        with gr.Accordion("Prompt and context tokens", open=False, elem_classes=["inspector-section"]):
                            prompt_note = gr.Markdown("", elem_classes=["scale-caption"])
                            prompt_strip = gr.HighlightedText(
                                label="Prompt tokens — click one",
                                color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                                show_legend=True,
                                combine_adjacent=False,
                                elem_id="prompt-strip",
                            )

            extension_pages = []
            extension_model_buttons = []
            navigation = NavigationService(extension_model_buttons.append)
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

            with gr.Column(
                scale=1, visible=False, elem_id="images-page"
            ) as images_page:
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

                    with gr.Column(scale=2, min_width=300, elem_id="image-inspector"):
                        # Which run the readouts belong to, the step being
                        # looked at, and the prompt token last clicked.
                        image_run_state = gr.State(None)
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

            with gr.Column(
                scale=1, visible=False, elem_id="models-page"
            ) as models_page:
                gr.Markdown(
                    "# Models\nDownload a model from Hugging Face, or load one "
                    "already on disk. Files are kept in your normal Hugging Face cache.",
                    elem_id="models-hero",
                )
                with gr.Row(elem_id="models-columns"):
                    with gr.Column(min_width=360, elem_id="model-controls"):
                        with gr.Column(elem_classes=["model-card"]):
                            gr.Markdown("## Model")
                            model_id = gr.Textbox(
                                value=settings.model_id_at_startup(saved),
                                label="Hugging Face model ID",
                                placeholder="organization/model-name",
                                info="The default OLMo 3 7B model is about 15 GB in full precision.",
                            )
                            hf_token = gr.Textbox(
                                label="Hugging Face token (optional)",
                                type="password",
                                placeholder="Only needed for gated or private models",
                            )
                            weight_precision = gr.Radio(
                                choices=[
                                    ("Full (16-bit)", "full"),
                                    ("8-bit", "8-bit"),
                                    ("4-bit", "4-bit"),
                                ],
                                value=saved.weight_precision,
                                label="Weight precision",
                                info=(
                                    "On Apple Metal, 8-bit and 4-bit weights take about a "
                                    "half and a quarter of the memory of full weights, at a "
                                    "small cost in accuracy; the first quantized load fetches "
                                    "the Metal kernels from the Hub. Other devices load full "
                                    "weights whatever is chosen. Applies to the next load."
                                ),
                            )
                            model_availability = gr.Markdown(
                                "Checking downloaded files…", elem_id="model-availability"
                            )
                            with gr.Row():
                                download_load_button = gr.Button(
                                    "Download and load", variant="primary", size="sm"
                                )
                                download_button = gr.Button("Download only", size="sm")
                                cached_button = gr.Button("Load cached", size="sm")
                                unload_button = gr.Button("Unload", size="sm")
                            model_status = gr.Markdown(
                                status_card(
                                    "No model loaded",
                                    "Choose a model under My Models, or enter a Hugging Face model ID to download one. Files are kept in your normal Hugging Face cache.",
                                ),
                                elem_id="model-status",
                            )

                        with gr.Column(elem_id="model-search", elem_classes=["model-card"]):
                            gr.Markdown("## Model search")
                            # One kind at a time, because the hub's own filters
                            # are; see SEARCH_KINDS.
                            search_kind = gr.Radio(
                                choices=list(SEARCH_KINDS),
                                value=SEARCH_KINDS[0][1],
                                show_label=False,
                                container=False,
                                elem_id="search-kind",
                            )
                            with gr.Row(elem_id="model-search-row"):
                                search_query = gr.Textbox(
                                    label="Search Hugging Face",
                                    placeholder="Model name, organization, or topic…",
                                    max_lines=1,
                                    scale=3,
                                    elem_id="model-search-query",
                                )
                                search_button = gr.Button(
                                    "Search", variant="primary", size="sm", scale=0, min_width=100,
                                    elem_id="model-search-button",
                                )
                            search_results = gr.Radio(
                                choices=[],
                                label="Search results",
                                elem_id="model-search-results",
                                show_label=False,
                                elem_classes=["model-list"],
                            )
                            search_detail = gr.Markdown(SEARCH_HINT, elem_classes=["model-detail"])
                            search_results_state = gr.State({})

                    with gr.Column(min_width=320, elem_classes=["model-card"]):
                        gr.Markdown("## My Models")
                        my_models_summary = gr.Markdown("", elem_classes=["scale-caption"])
                        sort_models = gr.Dropdown(
                            choices=list(MODEL_SORT_ORDERS),
                            value=DEFAULT_MODEL_SORT,
                            label="Sort by",
                            elem_classes=["model-sort"],
                        )
                        my_models = gr.Radio(
                            choices=[],
                            label="Downloaded models",
                            show_label=False,
                            elem_classes=["model-list"],
                        )
                        my_model_detail = gr.Markdown(
                            "", elem_id="my-model-detail", elem_classes=["model-detail"]
                        )
                        with gr.Row():
                            redownload_button = gr.Button("⬇️ Redownload", size="sm")
                            remove_button = gr.Button("🗑️ Remove", size="sm")
                            refresh_models_button = gr.Button("↻ Refresh", size="sm")
                        with gr.Column(
                            visible=False, elem_classes=["remove-confirm"]
                        ) as remove_confirm:
                            remove_question = gr.Markdown("", elem_classes=["model-detail"])
                            with gr.Row():
                                confirm_remove_button = gr.Button(
                                    "Remove from disk", variant="stop", size="sm"
                                )
                                cancel_remove_button = gr.Button("Cancel", size="sm")
                        # The model the open confirmation is about; None when closed.
                        pending_removal = gr.State(None)
                        # Whether the fit verdicts on screen were given with
                        # the device known; see refresh_after_device.
                        device_read = gr.State(False)

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
                extensions_control, extensions_note, active_extensions = build_extension_settings([ext.spec.id for ext in extensions], extension_errors)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## System prompt, reasoning, and prefill")
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
                        keep_reasoning = gr.Checkbox(
                            value=saved.keep_reasoning,
                            label="Send previous reasoning back to the model",
                            info="Off by default. Think models write a fresh reasoning block each turn, so replaying old ones burns context and usually hurts the next answer.",
                        )

                    with gr.Column():
                        gr.Markdown("## Input")
                        enter_sends = gr.Checkbox(
                            value=saved.enter_sends,
                            label="Enter sends the message",
                            info="Shift+Enter starts a new line. Turn off to swap the two.",
                        )
                        gr.Markdown(
                            "Escape stops a response that is still being written, "
                            "from anywhere on the Chat page - including the message "
                            "box and the Score text tab.",
                            elem_classes=["scale-caption"],
                        )

                        gr.Markdown("## Analysis")
                        analyze_prompt = gr.Checkbox(
                            value=saved.analyze_prompt,
                            label="Measure prompt tokens",
                            info="Scores every prompt token during the same pass that warms the cache.",
                        )

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
                        gr.Markdown("## Hardware")
                        hardware_view = gr.Markdown(
                            hardware_card(),
                            elem_id="hardware",
                            elem_classes=["model-detail"],
                        )
                        refresh_hardware_button = gr.Button(
                            "↻ Refresh", size="sm", scale=0, min_width=120
                        )
                        gr.Markdown(
                            "Estimates, not guarantees: they are what ChatLab "
                            "judges a load against, and each load and reply is "
                            "recorded in the log with the same figures.",
                            elem_classes=["scale-caption"],
                        )

        nav.change(
            show_page,
            nav,
            [conversation_pane, chat_page, images_page, models_page, settings_page],
        )
        # On the way to the page rather than on a timer: nothing here changes
        # while it is not being looked at, and reading it costs a subprocess.
        nav.change(refresh_hardware, None, hardware_view)
        demo.load(refresh_hardware, None, hardware_view)
        refresh_hardware_button.click(refresh_hardware, None, hardware_view)
        demo.load(restore_extensions, active_extensions, [extensions_control, extensions_note])
        for label, extension_page in extension_pages:
            def show_extension(page, expected=label):
                return gr.update(visible=page == expected)
            nav.change(show_extension, nav, extension_page)
        def open_models_from_extension():
            return (*go_to_models(), *(gr.update(visible=False) for _ in extension_pages))
        for button in extension_model_buttons:
            button.click(
                open_models_from_extension, None,
                # Every page container go_to_models() publishes an update
                # for, in the order show_page() returns them.
                [nav, conversation_pane, chat_page, images_page, models_page,
                 settings_page, *(page for _, page in extension_pages)],
            )
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
        badge_outputs = [model_badge_view, default_model_button, load_model_button]
        nav.change(refresh_model_badge, None, badge_outputs)
        demo.load(refresh_model_badge, None, badge_outputs)
        # And on a timer, so a tab that did not start the load hears about it
        # too. demo.load stays: it draws the badge at once rather than leaving
        # the value baked in when the page was built there for a tick.
        # show_progress="hidden" because this one runs on its own: the default
        # puts a pending shimmer on a handler's outputs, which every couple of
        # seconds would have the badge flickering at a reader who never asked
        # it anything.
        badge_timer.tick(
            refresh_model_badge, None, badge_outputs, show_progress="hidden"
        )
        # The same timer un-sticks the scored token count. A count asked for
        # during a reply gives up, and nothing about that message corrects
        # itself once the reply ends; see recover_score_budget, which is why
        # this is one listener rather than one on every path out of a
        # generation.
        badge_timer.tick(
            recover_score_budget,
            [score_budget, score_budget_load, *score_budget_inputs],
            score_budget_outputs,
            show_progress="hidden",
            concurrency_id=SCORE_BUDGET_QUEUE,
        )
        load_model_button.click(
            go_to_models,
            None,
            [nav, conversation_pane, chat_page, images_page, models_page, settings_page],
        )
        image_load_button.click(
            go_to_models,
            None,
            [nav, conversation_pane, chat_page, images_page, models_page, settings_page],
        )

        # ------------------------------------------------------------- Images
        # Every image handler publishes in this order; see IMAGE_OUTPUT_NAMES.
        image_outputs = [
            image_status,
            draw_button,
            stop_draw_button,
            image_seed,
            image_output,
            image_run_state,
            image_step,
            image_trajectory,
            image_tiles,
            image_chart,
            image_strip,
            image_note,
            image_overlay,
            image_token_state,
        ]
        image_inputs = [
            image_prompt,
            image_negative,
            image_steps,
            image_guidance,
            image_size,
            image_seed,
            image_randomize,
            image_record_attention,
        ]
        draw_button.click(draw, image_inputs, image_outputs)
        # Stop is not a cancel. The pipeline runs on its own thread and would
        # keep running with the generator gone, so the button sets the event
        # the run checks between steps and the generator publishes the
        # stopped run itself, trajectory and all. Cancelling it would throw
        # away the steps that had been recorded.
        stop_draw_button.click(stop_drawing, None, image_status)

        # The step slider moves the frame, the shading and the map together;
        # see select_step for why they cannot be allowed to disagree.
        image_step.release(
            select_step,
            [image_run_state, image_step, image_token_state],
            [image_trajectory, image_strip, image_note, image_overlay],
        )
        image_strip.select(remember_token, None, image_token_state).then(
            select_token,
            [image_run_state, image_token_state, image_step],
            image_overlay,
        )

        # The Images badge is refreshed on the same three occasions the Chat
        # one is, and for the same reasons: arriving at the page, opening it,
        # and the timer that tells a tab which did not start a load about it.
        image_badge_outputs = [image_badge_view, image_load_button]
        nav.change(refresh_image_badge, None, image_badge_outputs)
        demo.load(refresh_image_badge, None, image_badge_outputs)
        badge_timer.tick(
            refresh_image_badge, None, image_badge_outputs, show_progress="hidden"
        )

        image_settings_inputs = [
            image_negative,
            image_steps,
            image_guidance,
            image_size,
            image_seed,
            image_randomize,
            image_record_attention,
        ]
        for control in (
            image_negative,
            image_steps,
            image_guidance,
            image_size,
            image_randomize,
            image_record_attention,
        ):
            control.change(remember_image_settings, image_settings_inputs, None)
        # The seed box is written to by a finished picture, so only the
        # reader being done editing it commits what it holds; the Chat page's
        # seed follows the same rule for the same reason.
        for event in (image_seed.blur, image_seed.submit):
            event(remember_committed_image_seed, image_settings_inputs, None)

        # Every handler that can change what is on disk or in memory rescans
        # the cache afterwards, so My Models never shows a stale list.
        # The typed ID stays last: the model-actions listeners assert it is
        # the input the refresh is given, and a new argument goes before it
        # rather than displacing it.
        models_inputs = [my_models, sort_models, weight_precision, model_id]
        models_outputs = [my_models, my_model_detail, my_models_summary]
        action_inputs = [model_id, my_models]
        action_outputs = [
            model_availability, download_load_button, download_button, cached_button
        ]

        # Include programmatic selections (search, default, and rescans).
        # The selected row takes precedence, just as it does for a load.
        for control in action_inputs:
            control.change(
                refresh_model_actions, action_inputs, action_outputs,
                show_progress="hidden", trigger_mode="always_last",
                concurrency_id="model-actions",
            )

        def refresh_actions(event):
            return event.then(
                refresh_model_actions, action_inputs, action_outputs,
                show_progress="hidden", concurrency_id="model-actions",
            )

        # Refresh model-dependent displays after explicit model actions.
        # The timer also catches changes from other tabs, but this updates
        # the badge and token count immediately in the tab that acted.
        def rescan(event, *, reloads: bool = True):
            """Rescan the cache after ``event``, and re-read what the model feeds."""

            event = event.then(refresh_my_models, models_inputs, models_outputs)
            event = refresh_actions(event)
            if not reloads:
                return event
            return (
                event.then(refresh_model_badge, None, badge_outputs)
                .then(
                    score_token_count,
                    score_budget_inputs,
                    score_budget_outputs,
                    show_progress="hidden",
                    concurrency_id=SCORE_BUDGET_QUEUE,
                )
                # A load or an unload is the largest change the machine's
                # memory sees, so the hardware panel is re-read after it
                # rather than left showing what was true before.
                .then(refresh_hardware, None, hardware_view)
            )

        # Download-only changes the cache without changing the loaded model.
        rescan(
            download_button.click(
                download_model, [model_id, hf_token, my_models], model_status
            )
        )
        rescan(
            download_load_button.click(
                download_and_load_model,
                [model_id, hf_token, my_models, weight_precision],
                model_status,
            )
        )
        rescan(
            cached_button.click(
                load_cached_model,
                [model_id, my_models, weight_precision],
                model_status,
            )
        )
        rescan(unload_button.click(unload_model, outputs=model_status))
        # A manual refresh and a new sort order reorder a list; neither
        # changes what is on disk or in memory, which is all the badge and the
        # count ask about.
        refresh_actions(
            refresh_models_button.click(refresh_my_models, models_inputs, models_outputs)
        )
        sort_models.input(refresh_my_models, models_inputs, models_outputs)
        # Before the reader chooses an ID, startup can highlight the loaded model.
        refresh_actions(demo.load(refresh_my_models, [my_models, sort_models], models_outputs))
        # The badge's timer corrects the fit verdicts once torch has finished
        # importing: the page is painted before that, so the first verdicts
        # are given without knowing the device. It repaints once and then
        # does nothing for the rest of the session.
        badge_timer.tick(
            refresh_after_device,
            [device_read, *models_inputs, search_results, search_results_state],
            [*models_outputs, search_results, search_detail, device_read],
            show_progress="hidden",
        )
        # Escape stops a running generation, from anywhere on the page.
        demo.load(None, None, None, js=SHORTCUT_JS)

        # Selecting a default is navigation only. The Models page owns the
        # explicit download and load actions, including their errors.
        default_model_button.click(
            select_default_model,
            None,
            [
                model_id,
                my_models,
                my_model_detail,
                search_results,
                search_detail,
                model_status,
                remove_confirm,
                pending_removal,
                nav,
                conversation_pane,
                chat_page,
                images_page,
                models_page,
                settings_page,
            ],
        )
        # .input rather than .change: the refresh above also sets the radio,
        # and a .change listener would rewrite the model ID box on each rescan.
        my_models.input(
            select_my_model,
            [my_models, weight_precision],
            [model_id, my_model_detail],
        )
        # .input again, for the same reason: only the reader's own typing
        # withdraws the selection, never a refresh writing the box.
        model_id.input(clear_my_model_selection, None, [my_models, my_model_detail])
        # A pending removal is about the model that was selected when it was
        # asked for, so changing the selection withdraws it.
        confirm_outputs = [remove_confirm, pending_removal]
        my_models.input(hide_remove_confirm, None, confirm_outputs)
        model_id.input(hide_remove_confirm, None, confirm_outputs)
        rescan(
            redownload_button.click(
                redownload_my_model, [my_models, hf_token], model_status
            )
        )
        remove_button.click(
            ask_remove_my_model,
            my_models,
            [model_status, remove_confirm, remove_question, pending_removal],
        )
        # The confirm button deletes the model the question named, never the
        # radio's current value: see ask_remove_my_model.
        rescan(
            confirm_remove_button.click(
                remove_my_model, pending_removal, [model_status, *confirm_outputs]
            )
        )
        cancel_remove_button.click(hide_remove_confirm, None, confirm_outputs)

        search_outputs = [search_results, search_detail, search_results_state]
        # In search_models' own parameter order: the precision the fit
        # verdicts are judged at, then the kind the hub is asked for.
        search_inputs = [search_query, hf_token, weight_precision, search_kind]
        search_button.click(search_models, search_inputs, search_outputs)
        search_query.submit(search_models, search_inputs, search_outputs)
        # Switching kinds re-runs the query rather than leaving the other
        # kind's results under the new label. With an empty box it just
        # replaces the hint, which is what says which hub filter is on.
        search_kind.input(search_models, search_inputs, search_outputs)
        # Picking a search result names a model too, so it withdraws the My
        # Models selection the same way typing an ID does.
        search_results.input(
            select_search_result,
            [search_results, search_results_state, weight_precision],
            [model_id, search_detail],
        ).then(clear_my_model_selection, None, [my_models, my_model_detail])
        # Whether a model fits depends on how its weights would be held, so
        # both lists are repainted when that choice changes. Neither touches
        # the cache or the model in memory, so neither is a rescan.
        weight_precision.change(
            refresh_my_models, models_inputs, models_outputs
        ).then(
            refresh_search_results,
            [search_results, search_results_state, weight_precision],
            [search_results, search_detail],
        )
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
        sampling_controls = [temperature, top_p, top_k, max_new_tokens]
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
            )

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
            analyze_prompt,
            color_scale,
        ]
        chat_inputs = [prompt, conversation_state, *settings_inputs]

        # Everything saved between sessions, in PERSISTED_SETTING_NAMES order.
        persisted_inputs = [*settings_inputs, enter_sends, model_id, weight_precision]
        for control in (
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            randomize_seed,
            analyze_prompt,
            color_scale,
            enter_sends,
            model_id,
            weight_precision,
        ):
            control.change(remember_settings, persisted_inputs, None)
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
            update_sampling_label,
            sampling_controls,
            sampling_accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
        # The order every generation handler publishes in; see
        # CHAT_OUTPUT_NAMES.
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            seed,
            send_button,
            stop_button,
            token_detail,
            alternatives,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
            branch_source,
            context_ids_state,
        ]
        undo_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            token_detail,
            alternatives,
            send_button,
            stop_button,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]

        running = [
            send_button.click(chat, chat_inputs, chat_outputs),
            prompt.submit(chat, chat_inputs, chat_outputs),
            retry_button.click(retry_last, chat_inputs, chat_outputs),
            chatbot.retry(retry_message, chat_inputs, chat_outputs),
            chatbot.edit(edit_message, chat_inputs, chat_outputs),
            branch_button.click(
                branch_from,
                [branch_pick, branch_source, metrics_state, *chat_inputs],
                chat_outputs,
            ),
            branch_text_button.click(
                branch_with_text,
                [selected_token, branch_source, metrics_state, branch_text, *chat_inputs],
                chat_outputs,
            ),
        ]

        stop_button.click(
            stop_generation,
            inputs=[conversation_state, metrics_state, context_ids_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            outputs=[
                chatbot,
                conversation_state,
                send_button,
                stop_button,
                generation_status,
                branch_source,
            ],
            cancels=running,
        )

        # Undo, Clear and Load all replace or truncate the conversation, so
        # each has to stop the generator first: a surviving generate_reply
        # would write its own snapshot of the in-progress turns back into the
        # chatbot and the state, resurrecting what was just removed. Send,
        # Retry and Edit are exempt because they *are* the generation - they
        # re-enter generate_reply, and they are what everything else cancels.
        # They cannot be made to cancel each other either: Gradio captures a
        # listener's inputs when the request is queued, so the survivor would
        # rebuild the conversation from a snapshot taken before the cancelled
        # run wrote anything. A shared concurrency group has the same flaw - it
        # only delays the stale handler. Each of them refuses outright instead
        # while runtime.MANAGER.busy (see busy_state).
        undo_button.click(
            undo_last,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        chatbot.undo(
            undo_message,
            [conversation_state, color_scale],
            undo_outputs,
            cancels=running,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # Clear asks before it takes anything, so the button that opens the
        # question does nothing else - it neither clears nor cancels. The
        # confirm button is the one that does both.
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
        brings_its_sampling(confirm_clear_button.click(
            clear_chat,
            inputs=[color_scale, forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                generation_status,
                send_button,
                stop_button,
                token_detail,
                alternatives,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
                forks_state,
                conversation_list,
                clear_confirm,
            ],
            cancels=running,
        ))

        # Forking, switching, starting afresh and deleting all replace the
        # conversation, so they cancel a running generation for the same
        # reason Undo does.
        fork_outputs = [
            prompt,
            chatbot,
            conversation_state,
            forks_state,
            conversation_list,
            generation_status,
            send_button,
            stop_button,
            token_strip,
            metrics_state,
            token_detail,
            alternatives,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            trace_state,
        ]
        chatbot.select(remember_message, conversation_state, selected_message)

        brings_its_sampling(
            fork_button.click(
                fork_conversation,
                [
                    conversation_state,
                    forks_state,
                    selected_message,
                    color_scale,
                    *sampling_controls,
                ],
                fork_outputs,
                cancels=running,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            new_button.click(
                new_conversation,
                [conversation_state, forks_state, color_scale, *sampling_controls],
                fork_outputs,
                cancels=running,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        # .input rather than .change: the list is also redrawn by the handlers
        # above and the listener below, and a .change listener would switch a
        # second time on each.
        brings_its_sampling(
            conversation_list.input(
                switch_fork,
                [conversation_list, conversation_state, forks_state, color_scale],
                fork_outputs,
                cancels=running,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            delete_fork_button.click(
                delete_fork,
                [conversation_state, forks_state, color_scale],
                fork_outputs,
                cancels=running,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        # Every other path that changes the conversation - a streaming reply
        # above all - lands here, and the list's model tag and token count
        # follow it.
        conversation_state.change(
            refresh_conversation_list,
            [conversation_state, forks_state],
            [conversation_list, forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # And the forks' change, which the listener above fires in turn, is
        # where the file is written - once per change, whichever path made it.
        forks_state.change(
            remember_forks,
            [conversation_state, forks_state],
            None,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # The saved conversations come back first, so the listeners above
        # have something to describe. A page with nothing saved is left as
        # it was built. Like every other path that replaces the conversation,
        # this cancels a generation still running - the one a reload
        # interrupted, whose frames would otherwise land on the restored
        # transcript.
        brings_its_sampling(
            demo.load(
                restore_conversations,
                None,
                [chatbot, conversation_state, forks_state, conversation_list],
                cancels=running,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )

        save_button.click(
            save_conversation,
            [conversation_state, system_prompt],
            [saved_file, generation_status],
        )
        load_upload.upload(
            load_conversation,
            [load_upload, conversation_state, color_scale],
            [
                chatbot,
                conversation_state,
                system_prompt,
                token_strip,
                metrics_state,
                generation_status,
                token_detail,
                alternatives,
                send_button,
                stop_button,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                trace_state,
            ],
            cancels=running,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

        score_button.click(
            score_text,
            [score_context, score_input, use_chat_template, color_scale],
            [
                token_strip,
                metrics_state,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                summary_panel,
                surprise_panel,
                score_status,
                token_detail,
                alternatives,
                context_ids_state,
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
        color_scale.change(
            recolor,
            [metrics_state, prompt_metrics_state, color_scale],
            [token_strip, prompt_strip, scale_caption],
        )

        token_strip.select(
            inspect_token,
            inputs=metrics_state,
            outputs=[token_detail, alternatives],
        )
        prompt_strip.select(
            inspect_token,
            inputs=prompt_metrics_state,
            outputs=[token_detail, alternatives],
        )
        # A second listener on each strip keeps the clicked position for the
        # alternatives table. The prompt strip's clicks always clear it: a
        # prompt token cannot be branched, and a stale response position would
        # otherwise pair with the prompt token's rows.
        token_strip.select(remember_selection, metrics_state, selected_token)
        prompt_strip.select(remember_selection, prompt_metrics_state, selected_token)
        alternatives.select(
            choose_alternative,
            [metrics_state, selected_token, branch_source],
            [token_detail, branch_pick],
        )

        # Layer inspection. A third listener on each strip keeps the clicked
        # position, the button does the forward pass, and the slider repaints
        # the attention strip from the stored readout.
        token_strip.select(
            remember_inspect_target("response"), metrics_state, inspect_target
        )
        prompt_strip.select(
            remember_inspect_target("prompt"), prompt_metrics_state, inspect_target
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
            ],
            [lens_panel, attention_panel, attention_layer, insight_state, inspect_status],
        )
        attention_layer.release(
            render_attention, [insight_state, attention_layer], attention_panel
        )
        # Every path that redraws the strips writes the metrics state, so this
        # is where a readout of a token that is no longer on screen goes away.
        metrics_state.change(reset_inspection, insight_state, inspection_outputs)
    return demo
