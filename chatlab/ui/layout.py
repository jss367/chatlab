"""The page itself: build_app(), and what no one page owns.

build_app() reads as the page's table of contents. It draws each area in the
order the page lays them out, then wires them in the order their listeners
have always been bound, which matters: several listeners share a trigger -
the nav's change, the page load, the badge timer - and Gradio queues the
listeners on one trigger in the order they were bound.

Each page draws and wires itself from its own module: ui.chat_layout,
ui.images_layout, ui.models_layout and ui.settings_layout. What stays here is
what more than one area reads: the shared states, the nav and the
conversations pane, the extension pages, and the conversation listeners,
which answer to the pane and the Chat page both.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from functools import partial

import gradio as gr

from chatlab import settings, themes
from chatlab.conversation import MAIN_BRANCH, branch_choices, new_forks
from chatlab.device_memory import warm_device
from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.registry import load_enabled
from chatlab.ui import runtime
from chatlab.ui.background import ConversationEvents, ConversationJob
from chatlab.ui.chat_layout import (
    ChatPage,
    build_chat_page,
    wire_compare,
    wire_inspector,
    wire_model_bar,
    wire_prompt_file,
    wire_sampling,
    wire_score_and_batch,
    wire_steering,
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
    remember_forks,
    remember_message,
    restore_conversations,
    sampling_updates,
    save_conversation,
    switch_fork,
)
from chatlab.ui.extensions_page import data_directory, extension_css, restore_extensions
from chatlab.ui.fork_tree import TREE_CSS, TREE_JS, render_fork_tree, select_tree_branch
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
from chatlab.ui.icons import icon_classes
from chatlab.ui.images_layout import build_images_page, wire_images_page
from chatlab.ui.inspection import JACOBIAN_CSS, JACOBIAN_JS
from chatlab.ui.models_layout import (
    ModelRefresh,
    ModelsPage,
    build_models_page,
    wire_model_choice,
    wire_model_lists,
)
from chatlab.ui.models_page import go_to_models, select_model_to_load
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
from chatlab.ui.panel import empty_metrics
from chatlab.ui.scoring import SAMPLING_LABEL_QUEUE
from chatlab.ui.settings_layout import (
    SettingsPage,
    build_settings_page,
    wire_message_box,
    wire_settings_persistence,
)
from chatlab.ui.settings_page import refresh_hardware, update_sampling_label
from chatlab.ui.steering import load_with_steering, steering_updates
from chatlab.ui.styles import (
    COLUMN_JS,
    CSS,
    THEME,
    READ_ONLY_TEXT_JS,
    RESIZE_JS,
    SHORTCUT_JS,
    WRITING_SUGGESTIONS_JS,
)
from chatlab.ui.token_edit import save_token_edit
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    TOKEN_MENU_CSS,
    TOKEN_MENU_JS,
    branch_from_menu,
    edit_prompt_from_menu,
    menu_bridge_ids,
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
class ConversationPane:
    """The conversations pane beside the nav: the list, and what acts on it."""

    column: gr.Column
    conversation_list: gr.Radio
    new_button: gr.Button
    fork_button: gr.Button
    delete_fork_button: gr.Button
    clear_button: gr.Button
    clear_confirm: gr.Column
    clear_question: gr.Markdown
    confirm_clear_button: gr.Button
    cancel_clear_button: gr.Button


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
        states = _build_shared_state()

        with gr.Row(elem_id="shell"):
            nav = _build_nav(page_choices)
            pane = _build_conversation_pane()
            # The three pages share the rest of the width; one is visible at a
            # time, chosen by the nav.
            chat_page = build_chat_page(saved, states)
            extension_pages, extension_model_buttons = _build_extension_pages(
                extensions, extension_errors
            )
            images = build_images_page(saved)
            models = build_models_page(saved)
            settings_page = build_settings_page(saved, extensions, extension_errors)

        pages = Pages(
            nav=nav,
            conversations=pane.column,
            chat=chat_page.column,
            images=images.column,
            models=models.column,
            settings=settings_page.column,
            extensions=extension_pages,
        )
        badge_timer = chat_page.bar.badge_timer
        _wire_pages(demo, pages, settings_page, models, extension_model_buttons)
        wire_model_bar(
            demo, nav, chat_page, settings_page.thinking_mode, models.weight_precision,
        )
        wire_images_page(demo, images, nav, badge_timer)
        refresh = ModelRefresh(
            models, chat_page.bar.switch_outputs, chat_page.bar.badge_outputs,
            chat_page.score.budget_inputs, chat_page.score.budget_outputs,
            settings_page.hardware_view, settings_page.thinking_mode,
        )
        wire_model_lists(
            demo, pages, badge_timer, models, refresh, chat_page.bar.switch,
            chat_page.bar.badge_view,
        )
        _wire_page_scripts(demo, chat_page, states, settings_page.writing_suggestions)
        wire_model_choice(
            demo, pages, models, refresh, chat_page.bar.default_model_button, images.load_button,
        )
        wire_message_box(settings_page, chat_page.chat.prompt)
        wire_sampling(chat_page.sampling, states)
        wire_steering(chat_page.steering, states)
        settings_inputs, chat_inputs = _request_inputs(chat_page, settings_page, states)
        wire_settings_persistence(
            demo, saved, theme_style, settings_page, models, chat_page.sampling,
            chat_page.inspector.color_scale, states.forks, settings_inputs,
        )
        _wire_conversations(demo, states, pane, chat_page, settings_page, chat_inputs)
        wire_score_and_batch(
            chat_page, states, settings_page.system_prompt, settings_page.assistant_prefill,
        )
        wire_compare(
            demo, chat_page, states, settings_page.system_prompt,
            settings_page.assistant_prefill, settings_page.thinking_mode,
        )
        wire_prompt_file(chat_page.prompts, states)
        wire_inspector(chat_page, states)
    return demo


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


def _build_nav(page_choices: list[str]) -> gr.Radio:
    """The nav pane at the far left, and the choice of page in it."""

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
    return nav


def _build_conversation_pane() -> ConversationPane:
    """The conversations pane, which shows with the Chat page only."""

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
    return ConversationPane(
        column=conversation_pane,
        conversation_list=conversation_list,
        new_button=new_button,
        fork_button=fork_button,
        delete_fork_button=delete_fork_button,
        clear_button=clear_button,
        clear_confirm=clear_confirm,
        clear_question=clear_question,
        confirm_clear_button=confirm_clear_button,
        cancel_clear_button=cancel_clear_button,
    )


def _build_extension_pages(extensions: list, extension_errors: list[str]) -> tuple[list, list]:
    """A hidden column per enabled extension, each built by the extension itself.

    An extension whose page fails to build has its error added to
    ``extension_errors``, which the Settings page lists, and a note drawn in
    its place. Returns the ``(label, column)`` pairs, and the ``(button,
    model ID)`` pairs the pages asked to have open the Models page.
    """

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
    return extension_pages, extension_model_buttons


def _wire_pages(
    demo: gr.Blocks,
    pages: Pages,
    settings_page: SettingsPage,
    models: ModelsPage,
    extension_model_buttons: list,
) -> None:
    """The nav, the hardware panel read on the way to a page, and the extensions' ways out."""

    pages.nav.change(
        show_page,
        pages.nav,
        [pages.conversations, pages.chat, pages.images, pages.models, pages.settings],
    )
    # On the way to the page rather than on a timer: nothing here changes
    # while it is not being looked at, and reading it costs a subprocess.
    pages.nav.change(refresh_hardware, None, settings_page.hardware_view)
    demo.load(refresh_hardware, None, settings_page.hardware_view)
    settings_page.refresh_hardware_button.click(refresh_hardware, None, settings_page.hardware_view)
    demo.load(restore_extensions, settings_page.active_extensions, settings_page.extension_settings)
    for label, extension_page in pages.extensions:
        def show_extension(page, expected=label):
            return gr.update(visible=page == expected)
        pages.nav.change(show_extension, pages.nav, extension_page)
    # Every page container go_to_models() publishes an update for, in the
    # order show_page() returns them.
    extension_page_outputs = [pages.nav, pages.conversations, pages.chat, pages.images,
                              pages.models, pages.settings,
                              *(page for _, page in pages.extensions)]
    # The ID box and everything that has to move with it, in the order
    # select_model_to_load() returns them.
    extension_model_outputs = [models.model_id, models.my_models, models.my_model_detail,
                               models.search_selection, models.search_detail, models.model_status,
                               models.remove_confirm, models.pending_removal]
    def open_models_from_extension():
        return (*go_to_models(), *(gr.update(visible=False) for _ in pages.extensions))
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
        return (*filled, *(gr.update(visible=False) for _ in pages.extensions))
    for button, wanted_model in extension_model_buttons:
        if wanted_model is None:
            button.click(open_models_from_extension, None, extension_page_outputs)
        else:
            button.click(open_named_model_from_extension, wanted_model,
                         [*extension_model_outputs, *extension_page_outputs])


def _wire_page_scripts(
    demo: gr.Blocks, chat_page: ChatPage, states: SharedState, writing_suggestions: gr.Checkbox
) -> None:
    """The page's own scripts, run on load, and the fork tree's two listeners.

    The fork tree is the Chat page's, but its listeners were bound among
    the scripts, after the menu's and the tree's own, and are bound there
    still.
    """

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


def _request_inputs(
    chat_page: ChatPage, settings_page: SettingsPage, states: SharedState
) -> tuple[list, list]:
    """What a reply reads: the settings, and the whole request around them.

    The settings are in the order the generation handlers take them; the
    request is the message, the conversation, those settings, the steering
    vector and the thinking mode.
    """

    settings_inputs = [
        settings_page.system_prompt,
        settings_page.keep_reasoning,
        settings_page.assistant_prefill,
        chat_page.sampling.temperature,
        chat_page.sampling.top_p,
        chat_page.sampling.top_k,
        chat_page.sampling.skip_top_below,
        chat_page.sampling.max_new_tokens,
        chat_page.sampling.seed,
        chat_page.sampling.randomize_seed,
        settings_page.analyze_prompt,
        chat_page.inspector.color_scale,
    ]
    # Persistence runs separately; every request must snapshot the controls
    # the reader sees, even while remember_steering is still queued.
    chat_inputs = [chat_page.chat.prompt, states.conversation, *settings_inputs, *chat_page.steering.inputs, settings_page.thinking_mode]
    return settings_inputs, chat_inputs


def _wire_conversations(
    demo: gr.Blocks,
    states: SharedState,
    pane: ConversationPane,
    chat_page: ChatPage,
    settings_page: SettingsPage,
    chat_inputs: list,
) -> None:
    """Every listener that starts, stops, changes or switches a conversation.

    They answer to the conversations pane and the Chat page both, and are
    bound through one ConversationEvents, which is what knows whether the
    conversation on screen is still being written.
    """

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
        "conversation_list": pane.conversation_list,
        "clear_confirm": pane.clear_confirm,
        "branch_text": chat_page.inspector.branch_text,
        "system_prompt": settings_page.system_prompt,
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
    pane.clear_button.click(
        ask_clear_chat,
        [states.conversation, states.forks],
        [chat_page.chat.generation_status, pane.clear_confirm, pane.clear_question],
    )
    pane.cancel_clear_button.click(hide_clear_confirm, None, pane.clear_confirm)
    # The question names how many conversations it would take, and that
    # count is read when it is asked. Anything that adds or removes one
    # withdraws it rather than leaving a stale promise above a button
    # that would take more than the promise says - the same reason
    # choosing another model withdraws the removal question. Pressing
    # Clear again re-asks with the numbers as they are now.
    for control in (pane.new_button, pane.fork_button, pane.delete_fork_button):
        control.click(hide_clear_confirm, None, pane.clear_confirm)
    pane.conversation_list.input(hide_clear_confirm, None, pane.clear_confirm)
    brings_its_sampling(conversation_events.bind(
        pane.confirm_clear_button.click, clear_chat,
        clear=True,
        inputs=[chat_page.inspector.color_scale, states.forks],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        outputs=CLEAR_OUTPUT_NAMES,
    ))

    chat_page.chat.chatbot.select(remember_message, states.conversation, states.selected_message)

    # Navigation takes a snapshot of the view; the job keeps its source.
    brings_its_sampling(
        navigate(
            pane.fork_button.click, fork_conversation,
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
            pane.new_button.click, new_conversation,
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
            pane.conversation_list.input, switch_fork,
            [pane.conversation_list, states.conversation, states.forks, chat_page.inspector.color_scale],
            FORK_OUTPUT_NAMES,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )
    brings_its_sampling(
        conversation_events.bind(
            pane.delete_fork_button.click, delete_fork,
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
        [pane.conversation_list, states.forks],
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
        [states.conversation, settings_page.system_prompt, *chat_page.steering.inputs],
        [chat_page.chat.saved_file, chat_page.chat.generation_status],
    )
    conversation_events.bind(
        chat_page.chat.load_upload.upload, load_with_steering,
        [chat_page.chat.load_upload, states.conversation, chat_page.inspector.color_scale, states.forks],
        STEERED_LOAD_OUTPUT_NAMES,
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
