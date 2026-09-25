"""The page itself: every control and how the handlers are wired to them."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import html
from functools import partial

import gradio as gr

from chatlab.ui.fork_tree import TREE_CSS, TREE_JS, render_fork_tree, select_tree_branch
from chatlab import settings, themes
from chatlab.thinking import THINKING_CHOICES
from chatlab.conversation import MAIN_BRANCH, branch_choices, new_forks
from chatlab.device_memory import warm_device
from chatlab.ui import runtime
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
from chatlab.ui.token_edit import save_token_edit
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
    PAGES,
    QUIET_TICK,
    show_page,
)
from chatlab.ui.conversations import (
    delete_fork,
    fork_conversation,
    new_conversation,
    remember_branch_sampling,
    remember_forks,
    remember_message,
    restore_conversations,
    sampling_updates,
    save_conversation,
    switch_fork,
)
from chatlab.ui.steering import load_with_steering, steering_updates
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
from chatlab.ui.inspection import JACOBIAN_CSS, JACOBIAN_JS
from chatlab.ui.models_page import go_to_models, select_model_to_load
from chatlab.ui.panel import empty_metrics
from chatlab.ui.scoring import SAMPLING_LABEL_QUEUE
from chatlab.ui.settings_page import (
    apply_theme,
    hardware_card,
    refresh_hardware,
    remember_committed_seed,
    remember_prefill_limit,
    remember_settings,
    reset_sampling,
    restore_settings,
    update_sampling_label,
)
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    TOKEN_MENU_CSS,
    TOKEN_MENU_JS,
    branch_from_menu,
    edit_prompt_from_menu,
    menu_bridge_ids,
)
from chatlab.ui.styles import (
    COLUMN_JS,
    CSS,
    THEME,
    READ_ONLY_TEXT_JS,
    RESIZE_JS,
    SHORTCUT_JS,
    WRITING_SUGGESTIONS_JS,
    set_message_box_keys,
)
from chatlab.ui.chat_layout import (
    build_chat_page,
    wire_compare,
    wire_inspector,
    wire_model_bar,
    wire_prompt_file,
    wire_sampling,
    wire_score_and_batch,
    wire_steering,
)
from chatlab.ui.images_layout import build_images_page, wire_images_page
from chatlab.ui.models_layout import (
    ModelRefresh,
    build_models_page,
    wire_model_choice,
    wire_model_lists,
)


@dataclass(frozen=True)
class SharedState:
    """The states, and the hidden bridges a page script writes into, that more than one area reads."""

    conversation: gr.State
    metrics: gr.State
    prompt_metrics: gr.State
    score_metrics: gr.State
    trace: gr.State
    menu_request: gr.Textbox
    menu_response: gr.HTML
    menu_action: gr.Textbox
    prompt_menu_request: gr.Textbox
    prompt_menu_response: gr.HTML
    prompt_menu_action: gr.Textbox
    selected_token: gr.State
    branch_pick: gr.State
    forks: gr.State
    tree_selection: gr.State
    selected_message: gr.State
    token_edit_target: gr.State
    context_ids: gr.State
    score_context_ids: gr.State
    chat_metrics: gr.State
    chat_context_ids: gr.State
    steering: gr.State
    extract: gr.State
    compare_a: gr.State
    compare_b: gr.State
    compare_export: gr.State
    inspect_target: gr.State
    insight: gr.State
    loaded_prompts: gr.State
    batch_directory: gr.State
    score_budget_load: gr.State


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


def _build_shared_state() -> SharedState:
    """Every state the page's listeners share, drawn before the page itself."""

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
    return SharedState(
        conversation=conversation_state,
        metrics=metrics_state,
        prompt_metrics=prompt_metrics_state,
        score_metrics=score_metrics_state,
        trace=trace_state,
        menu_request=menu_request,
        menu_response=menu_response,
        menu_action=menu_action,
        prompt_menu_request=prompt_menu_request,
        prompt_menu_response=prompt_menu_response,
        prompt_menu_action=prompt_menu_action,
        selected_token=selected_token,
        branch_pick=branch_pick,
        forks=forks_state,
        tree_selection=tree_selection,
        selected_message=selected_message,
        token_edit_target=token_edit_target,
        context_ids=context_ids_state,
        score_context_ids=score_context_ids_state,
        chat_metrics=chat_metrics_state,
        chat_context_ids=chat_context_ids_state,
        steering=steering_state,
        extract=extract_state,
        compare_a=compare_a_state,
        compare_b=compare_b_state,
        compare_export=compare_export_state,
        inspect_target=inspect_target,
        insight=insight_state,
        loaded_prompts=loaded_prompts_state,
        batch_directory=batch_directory_state,
        score_budget_load=score_budget_load,
    )


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
        states = _build_shared_state()

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
            chat_page = build_chat_page(saved, states)

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
            chat=chat_page.column,
            images=images.column,
            models=models.column,
            settings=settings_page,
            extensions=extension_pages,
        )

        nav.change(
            show_page,
            nav,
            [conversation_pane, chat_page.column, images.column, models.column, settings_page],
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
        extension_page_outputs = [nav, conversation_pane, chat_page.column, images.column,
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
        wire_model_bar(demo, nav, chat_page, thinking_mode, models.weight_precision)
        wire_images_page(demo, images, nav, chat_page.bar.badge_timer)
        refresh = ModelRefresh(
            models, chat_page.bar.switch_outputs, chat_page.bar.badge_outputs,
            chat_page.score.budget_inputs, chat_page.score.budget_outputs, hardware_view,
            thinking_mode,
        )
        wire_model_lists(
            demo, pages, chat_page.bar.badge_timer, models, refresh, chat_page.bar.switch,
            chat_page.bar.badge_view,
        )
        # The menu handles Escape before the global generation shortcut.
        demo.load(None, None, None, js=TOKEN_MENU_JS)
        demo.load(None, None, None, js=TREE_JS)
        demo.load(None, None, None, js=JACOBIAN_JS)
        chat_page.tree.action.input(
            select_tree_branch,
            [chat_page.tree.action, states.conversation, states.forks, states.tree_selection],
            [states.tree_selection, chat_page.tree.view, chat_page.tree.comparison],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            trigger_mode="always_last",
        )
        states.forks.change(
            render_fork_tree,
            [states.conversation, states.forks, states.tree_selection],
            [chat_page.tree.view, chat_page.tree.comparison],
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

        wire_model_choice(
            demo, pages, models, refresh, chat_page.bar.default_model_button, images.load_button,
        )
        enter_sends.change(set_message_box_keys, enter_sends, chat_page.chat.prompt)

        wire_sampling(chat_page.sampling, states)
        wire_steering(chat_page.steering, states)

        def brings_its_sampling(event):
            """Put the newly active conversation's sampling onto the controls.

            Every path that changes which conversation is on screen ends
            here, so the sliders describe the conversation in front of the
            reader rather than the one they just left. The label follows the
            controls, as it does when they are moved by hand.
            """

            return event.then(
                sampling_updates,
                states.forks,
                chat_page.sampling.controls,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                chat_page.sampling.controls,
                chat_page.sampling.accordion,
                concurrency_id=SAMPLING_LABEL_QUEUE,
            ).then(
                steering_updates, states.forks, chat_page.steering.outputs,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            assistant_prefill,
            chat_page.sampling.temperature,
            chat_page.sampling.top_p,
            chat_page.sampling.top_k,
            chat_page.sampling.skip_top_below,
            chat_page.sampling.max_new_tokens,
            chat_page.sampling.seed,
            chat_page.sampling.randomize_seed,
            analyze_prompt,
            chat_page.inspector.color_scale,
        ]
        # Persistence runs separately; every request must snapshot the controls
        # the reader sees, even while remember_steering is still queued.
        chat_inputs = [chat_page.chat.prompt, states.conversation, *settings_inputs, *chat_page.steering.inputs, thinking_mode]

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
            chat_page.sampling.randomize_seed,
            analyze_prompt,
            chat_page.inspector.color_scale,
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
        for control in chat_page.sampling.controls:
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
            chat_page.sampling.resets, settings.CONVERSATION_SAMPLING, chat_page.sampling.controls, strict=True
        ):
            button.click(
                partial(reset_sampling, name, saved.prefill_token_limit),
                None,
                control,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                remember_branch_sampling,
                [states.forks, *chat_page.sampling.controls],
                states.forks,
                show_progress="hidden",
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                remember_settings,
                persisted_inputs,
                None,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                chat_page.sampling.controls,
                chat_page.sampling.accordion,
                show_progress="hidden",
                concurrency_id=SAMPLING_LABEL_QUEUE,
            )
        # The seed box is the one control the app writes to itself: a finished
        # response leaves the seed that produced it there, and saving that
        # would overwrite the seed the reader chose. Blur and submit are the
        # two ways a person is done editing a number, and they are the only
        # events that write the box's contents down; every other control
        # leaves the saved seed where it is. See remember_settings().
        for event in (chat_page.sampling.seed.blur, chat_page.sampling.seed.submit):
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
                [prefill_token_limit, chat_page.sampling.max_new_tokens, states.forks, *chat_page.sampling.controls],
                [prefill_token_limit, chat_page.sampling.max_new_tokens, states.forks],
                concurrency_id=CONVERSATION_PANE_QUEUE,
            ).then(
                update_sampling_label,
                chat_page.sampling.controls,
                chat_page.sampling.accordion,
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
            chat_page.sampling.controls,
            chat_page.sampling.accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
        # The one table from the names the conversation handlers publish
        # under to the components those names are drawn in; see ui.outputs.
        # Every listener bound through conversation_events names its outputs
        # from here, so a handler and its listener cannot disagree about
        # which value goes where.
        conversation_outputs = {
            "prompt": chat_page.chat.prompt,
            "chatbot": chat_page.chat.chatbot,
            "turns": states.conversation,
            "strip": chat_page.chat.token_strip,
            "metrics": states.metrics,
            "status": chat_page.chat.generation_status,
            "seed": chat_page.sampling.seed,
            "send": chat_page.chat.send_button,
            "stop": chat_page.chat.stop_button,
            "detail": chat_page.inspector.token_detail,
            "alternatives": chat_page.inspector.alternatives,
            "prompt_strip": chat_page.inspector.prompt_strip,
            "prompt_metrics": states.prompt_metrics,
            "prompt_note": chat_page.inspector.prompt_note,
            "summary": chat_page.inspector.summary_panel,
            "surprise": chat_page.inspector.surprise_panel,
            "trace": states.trace,
            "context_ids": states.context_ids,
            "chat_metrics": states.chat_metrics,
            "chat_context_ids": states.chat_context_ids,
            "selected_token": states.selected_token,
            "branch_pick": states.branch_pick,
            "token_editor": chat_page.chat.token_editor,
            "token_edit_target": states.token_edit_target,
            "forks": states.forks,
            "conversation_list": conversation_list,
            "clear_confirm": clear_confirm,
            "branch_text": chat_page.inspector.branch_text,
            "system_prompt": system_prompt,
            "steering_state": states.steering,
            "steering_enabled": chat_page.steering.enabled,
            "steering_strength": chat_page.steering.strength,
            "steering_layer": chat_page.steering.layer,
            "steering_status": chat_page.steering.status,
        }
        # Where tests, and anything else holding the page, look a component
        # up by the name its handlers publish it under.
        demo.conversation_outputs = conversation_outputs

        background_state = gr.State(ConversationJob())
        conversation_events = ConversationEvents(
            background_state, conversation_outputs, chat_page.inspector.color_scale, CONVERSATION_PANE_QUEUE,
        )
        response_timer = gr.Timer(0.25)
        response_timer.tick(
            conversation_events.poll,
            [background_state, states.conversation, states.forks, chat_page.inspector.color_scale],
            conversation_events.components(POLL_OUTPUT_NAMES),
            concurrency_id=CONVERSATION_PANE_QUEUE,
            **QUIET_TICK,
        )

        start_response = partial(conversation_events.bind, generation=True)
        navigate = partial(conversation_events.bind, navigation=True)
        start_response(states.menu_action.input, branch_from_menu, [states.menu_action, *chat_inputs], CHAT_OUTPUT_NAMES)
        start_response(
            states.prompt_menu_action.input, edit_prompt_from_menu,
            [states.prompt_menu_action, states.context_ids, states.prompt_metrics, *chat_inputs],
            CHAT_OUTPUT_NAMES,
        )
        start_response(chat_page.chat.send_button.click, chat, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(chat_page.chat.prompt.submit, chat, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(chat_page.chat.retry_button.click, retry_last, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(chat_page.chat.next_token_button.click, next_token, [states.branch_pick, *chat_inputs], CHAT_OUTPUT_NAMES)
        start_response(chat_page.chat.chatbot.retry, retry_message, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(chat_page.chat.chatbot.edit, edit_message, chat_inputs, CHAT_OUTPUT_NAMES)
        start_response(
            chat_page.chat.token_edit_save.click, save_token_edit,
            [states.token_edit_target, chat_page.chat.token_edit_text, *chat_inputs],
            TOKEN_EDIT_OUTPUT_NAMES,
        )
        start_response(
            chat_page.inspector.branch_button.click, branch_from, [states.branch_pick, *chat_inputs], CHAT_OUTPUT_NAMES,
        )
        start_response(
            chat_page.inspector.branch_text_button.click, branch_with_text,
            [states.selected_token, chat_page.inspector.branch_text, *chat_inputs], CHAT_OUTPUT_NAMES,
        )

        conversation_events.bind(
            chat_page.chat.stop_button.click, stop_generation,
            inputs=[states.conversation, chat_page.inspector.color_scale],
            outputs=STOP_OUTPUT_NAMES,
            stop=True,
        )

        # Mutations of a running conversation ask the reader to Stop first.
        # Navigation and changes to other conversations leave its job alone.
        conversation_events.bind(
            chat_page.chat.undo_button.click, undo_last,
            [states.conversation, chat_page.inspector.color_scale],
            UNDO_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        conversation_events.bind(
            chat_page.chat.chatbot.undo, undo_message,
            [states.conversation, chat_page.inspector.color_scale],
            UNDO_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
        # Clear asks before it takes anything, so the button that opens the
        # question leaves the conversations and background job alone. The
        # confirm button clears them once generation is stopped.
        clear_button.click(
            ask_clear_chat,
            [states.conversation, states.forks],
            [chat_page.chat.generation_status, clear_confirm, clear_question],
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
            inputs=[chat_page.inspector.color_scale, states.forks],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            outputs=CLEAR_OUTPUT_NAMES,
        ))

        chat_page.chat.chatbot.select(remember_message, states.conversation, states.selected_message)

        # Navigation takes a snapshot of the view; the job keeps its source.
        brings_its_sampling(
            navigate(
                fork_button.click, fork_conversation,
                [
                    states.conversation,
                    states.forks,
                    states.selected_message,
                    chat_page.inspector.color_scale,
                    *chat_page.sampling.controls,
                ],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            navigate(
                new_button.click, new_conversation,
                [states.conversation, states.forks, chat_page.inspector.color_scale, *chat_page.sampling.controls],
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
                [conversation_list, states.conversation, states.forks, chat_page.inspector.color_scale],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        brings_its_sampling(
            conversation_events.bind(
                delete_fork_button.click, delete_fork,
                [states.conversation, states.forks, chat_page.inspector.color_scale],
                FORK_OUTPUT_NAMES,
                concurrency_id=CONVERSATION_PANE_QUEUE,
            )
        )
        # Every other path that changes the conversation lands here, and
        # the list's model tag, running indicator and token count
        # follow it. Hide the loading overlay so each streaming frame updates
        # the labels without making the whole list blink.
        states.conversation.change(
            conversation_events.refresh_conversation_list,
            [states.conversation, states.forks, background_state],
            [conversation_list, states.forks],
            concurrency_id=CONVERSATION_PANE_QUEUE,
            show_progress="hidden",
        )
        # And the forks' change, which the listener above fires in turn, is
        # where ordinary view changes are saved. Workers also save independently.
        states.forks.change(
            remember_forks,
            [states.conversation, states.forks],
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

        chat_page.chat.save_button.click(
            save_conversation,
            [states.conversation, system_prompt, *chat_page.steering.inputs],
            [chat_page.chat.saved_file, chat_page.chat.generation_status],
        )
        conversation_events.bind(
            chat_page.chat.load_upload.upload, load_with_steering,
            [chat_page.chat.load_upload, states.conversation, chat_page.inspector.color_scale, states.forks],
            STEERED_LOAD_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )

        wire_score_and_batch(chat_page, states, system_prompt, assistant_prefill)
        wire_compare(demo, chat_page, states, system_prompt, assistant_prefill, thinking_mode)
        wire_prompt_file(chat_page.prompts, states)
        wire_inspector(chat_page, states)
    return demo
