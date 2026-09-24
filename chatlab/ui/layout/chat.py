"""The Chat page: the conversation, its composer and header, and how they are wired."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import gradio as gr

from chatlab.conversation import MAIN_BRANCH, branch_choices, new_forks
from chatlab.token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
from chatlab.trace_export import write_trace_export
from chatlab.ui import experiments, runtime
from chatlab.ui.background import ConversationEvents, ConversationJob
from chatlab.ui.common import CONVERSATION_PANE_WIDTH, STOP_LABEL, TRANSCRIPT_LABEL
from chatlab.ui.conversations import (
    delete_fork,
    fork_conversation,
    new_conversation,
    remember_forks,
    remember_message,
    restore_conversations,
    save_conversation,
    switch_fork,
)
from chatlab.ui.fork_tree import render_fork_tree
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
from chatlab.ui.layout.common import CONVERSATION_PANE_QUEUE, QUIET_TICK
from chatlab.ui.layout.compare import build_compare_tab
from chatlab.ui.layout.inspector import build_inspector_pane
from chatlab.ui.layout.prompts import build_prompts_tab, build_score_tab
from chatlab.ui.layout.sampling import build_sampling_controls, build_steering_controls
from chatlab.ui.models_page import (
    BADGE_REFRESH_SECONDS,
    loaded_model_badge,
    refresh_model_badge,
    refresh_model_switch,
    refresh_stale_model_switch,
)
from chatlab.ui.scoring import SCORE_BUDGET_QUEUE, recover_score_budget, score_token_count
from chatlab.ui.settings_page import refresh_thinking_mode
from chatlab.ui.steering import load_with_steering
from chatlab.ui.styles import message_box_settings, pane_handle
from chatlab.ui.token_edit import save_token_edit
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    MENU_STRIP_CLASS,
    branch_from_menu,
    edit_prompt_from_menu,
)


def build_chat_page(compare_a_state, compare_b_state, compare_export_state, extract_state, saved, trace_state) -> SimpleNamespace:
    """Build the Chat page and hand back the controls the wiring reads."""

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
                            sampling = build_sampling_controls(saved)
                            steering = build_steering_controls(extract_state)
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

                    scoring = build_score_tab()

                    prompts = build_prompts_tab()

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

                    compare = build_compare_tab(compare_a_state, compare_b_state, compare_export_state)

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

            inspector = build_inspector_pane(saved)

    return SimpleNamespace(
        **vars(inspector),
        **vars(prompts),
        **vars(compare),
        **vars(steering),
        **vars(sampling),
        **vars(scoring),
        badge_timer=badge_timer,
        chat_page=chat_page,
        chatbot=chatbot,
        conversation_tabs=conversation_tabs,
        default_model_button=default_model_button,
        experiments_view=experiments_view,
        generation_status=generation_status,
        inspector_resizer=inspector_resizer,
        load_upload=load_upload,
        model_badge_view=model_badge_view,
        model_switch=model_switch,
        next_token_button=next_token_button,
        prompt=prompt,
        retry_button=retry_button,
        save_button=save_button,
        saved_file=saved_file,
        send_button=send_button,
        stop_button=stop_button,
        switch_stamp=switch_stamp,
        token_edit_cancel=token_edit_cancel,
        token_edit_save=token_edit_save,
        token_edit_text=token_edit_text,
        token_editor=token_editor,
        token_strip=token_strip,
        token_view=token_view,
        tree_action=tree_action,
        tree_comparison=tree_comparison,
        tree_view=tree_view,
        undo_button=undo_button,
    )


def build_conversation_pane() -> SimpleNamespace:
    """Build the conversations pane beside the nav, shown with Chat only."""

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

    return SimpleNamespace(
        cancel_clear_button=cancel_clear_button,
        clear_button=clear_button,
        clear_confirm=clear_confirm,
        clear_question=clear_question,
        confirm_clear_button=confirm_clear_button,
        conversation_list=conversation_list,
        conversation_pane=conversation_pane,
        delete_fork_button=delete_fork_button,
        fork_button=fork_button,
        new_button=new_button,
    )


def wire_model_badge(ui: SimpleNamespace) -> None:
    """Keep the badge, the model switcher, the thinking mode and the scored token count current."""

    # The scored token count follows the boxes as they are typed into.
    # always_last coalesces a burst of keystrokes into the one count that
    # matters, and the progress bar is hidden because a spinner on every
    # keystroke would be worse than the number is good.
    score_budget_inputs = [ui.score_context, ui.score_input, ui.use_chat_template]
    # The count travels with the load it was counted against; see
    # recover_score_budget for what that is for.
    score_budget_outputs = [ui.score_budget, ui.score_budget_load]
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
    ui.nav.change(refresh_thinking_mode, None, ui.thinking_mode, show_progress="hidden")
    ui.demo.load(refresh_thinking_mode, None, ui.thinking_mode)
    ui.badge_timer.tick(refresh_thinking_mode, None, ui.thinking_mode, **QUIET_TICK)
    badge_outputs = [ui.model_badge_view, ui.default_model_button]
    ui.nav.change(refresh_model_badge, None, badge_outputs)
    ui.demo.load(refresh_model_badge, None, badge_outputs)
    # The switcher is drawn on the same two occasions. Its choices cost a
    # cache scan and a memory reading, so the timer only redraws it once
    # what it shows, what is on disk, or what would now fit has moved; see
    # refresh_stale_model_switch. Every draw hands back the stamp it read,
    # which is how the next tick knows the difference.
    switch_outputs = [ui.model_switch, ui.switch_stamp]
    ui.nav.change(refresh_model_switch, ui.weight_precision, switch_outputs)
    ui.demo.load(refresh_model_switch, ui.weight_precision, switch_outputs)
    ui.badge_timer.tick(
        refresh_stale_model_switch,
        [ui.model_switch, ui.switch_stamp, ui.weight_precision],
        switch_outputs,
        **QUIET_TICK,
    )
    # And on a timer, so a tab that did not start the load hears about it
    # too. demo.load stays: it draws the badge at once rather than leaving
    # the value baked in when the page was built there for a tick.
    # QUIET_TICK because this one runs on its own: the default fades a
    # handler's outputs in and out, which every couple of seconds would
    # have the badge flickering at a reader who never asked it anything.
    ui.badge_timer.tick(refresh_model_badge, None, badge_outputs, **QUIET_TICK)
    # The same timer un-sticks the scored token count. A count asked for
    # during a reply gives up, and nothing about that message corrects
    # itself once the reply ends; see recover_score_budget, which is why
    # this is one listener rather than one on every path out of a
    # generation.
    ui.badge_timer.tick(
        recover_score_budget,
        [ui.score_budget, ui.score_budget_load, *score_budget_inputs],
        score_budget_outputs,
        concurrency_id=SCORE_BUDGET_QUEUE,
        **QUIET_TICK,
    )

    # What the wiring after this reads.
    ui.badge_outputs = badge_outputs
    ui.score_budget_inputs = score_budget_inputs
    ui.score_budget_outputs = score_budget_outputs
    ui.switch_outputs = switch_outputs


def wire_conversation(ui: SimpleNamespace) -> None:
    """Wire the conversation: sending, retrying, editing, branching, forking, and saving it."""

    # The order every generation handler publishes in; see
    # CHAT_OUTPUT_NAMES.
    chat_outputs = [
        ui.prompt,
        ui.chatbot,
        ui.conversation_state,
        ui.token_strip,
        ui.metrics_state,
        ui.generation_status,
        ui.seed,
        ui.send_button,
        ui.stop_button,
        ui.token_detail,
        ui.alternatives,
        ui.prompt_strip,
        ui.prompt_metrics_state,
        ui.prompt_note,
        ui.summary_panel,
        ui.surprise_panel,
        ui.trace_state,
        ui.context_ids_state,
        ui.chat_metrics_state,
        ui.chat_context_ids_state,
        ui.selected_token,
        ui.branch_pick,
    ]
    undo_outputs = [
        ui.prompt,
        ui.chatbot,
        ui.conversation_state,
        ui.token_strip,
        ui.metrics_state,
        ui.generation_status,
        ui.token_detail,
        ui.alternatives,
        ui.send_button,
        ui.stop_button,
        ui.prompt_strip,
        ui.prompt_metrics_state,
        ui.prompt_note,
        ui.summary_panel,
        ui.surprise_panel,
        ui.trace_state,
        ui.selected_token,
        ui.branch_pick,
    ]

    background_state = gr.State(ConversationJob())
    conversation_events = ConversationEvents(
        background_state, ui.conversation_state, ui.forks_state, ui.conversation_list,
        ui.color_scale, [*chat_outputs, ui.token_editor, ui.token_edit_target],
        CONVERSATION_PANE_QUEUE,
    )
    response_timer = gr.Timer(0.25)
    response_timer.tick(
        conversation_events.poll,
        [background_state, ui.conversation_state, ui.forks_state, ui.color_scale],
        [*chat_outputs, ui.token_editor, ui.token_edit_target, ui.forks_state, ui.conversation_list],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        **QUIET_TICK,
    )

    start_response = partial(conversation_events.bind, generation=True)
    navigate = partial(conversation_events.bind, navigation=True)
    start_response(ui.menu_action.input, branch_from_menu, [ui.menu_action, *ui.chat_inputs], chat_outputs)
    start_response(
        ui.prompt_menu_action.input, edit_prompt_from_menu,
        [ui.prompt_menu_action, ui.context_ids_state, ui.prompt_metrics_state, *ui.chat_inputs],
        chat_outputs,
    )
    start_response(ui.send_button.click, chat, ui.chat_inputs, chat_outputs)
    start_response(ui.prompt.submit, chat, ui.chat_inputs, chat_outputs)
    start_response(ui.retry_button.click, retry_last, ui.chat_inputs, chat_outputs)
    start_response(ui.next_token_button.click, next_token, [ui.branch_pick, *ui.chat_inputs], chat_outputs)
    start_response(ui.chatbot.retry, retry_message, ui.chat_inputs, chat_outputs)
    start_response(ui.chatbot.edit, edit_message, ui.chat_inputs, chat_outputs)
    start_response(
        ui.token_edit_save.click, save_token_edit,
        [ui.token_edit_target, ui.token_edit_text, *ui.chat_inputs],
        [*chat_outputs, ui.token_editor, ui.token_edit_target],
    )
    start_response(
        ui.branch_button.click, branch_from, [ui.branch_pick, *ui.chat_inputs], chat_outputs,
    )
    start_response(
        ui.branch_text_button.click, branch_with_text,
        [ui.selected_token, ui.branch_text, *ui.chat_inputs], chat_outputs,
    )

    conversation_events.bind(
        ui.stop_button.click, stop_generation,
        inputs=[ui.conversation_state, ui.color_scale],
        outputs=[
            ui.chatbot, ui.conversation_state, ui.token_strip, ui.send_button,
            ui.stop_button, ui.generation_status,
        ],
        stop=True,
    )

    # Mutations of a running conversation ask the reader to Stop first.
    # Navigation and changes to other conversations leave its job alone.
    conversation_events.bind(
        ui.undo_button.click, undo_last,
        [ui.conversation_state, ui.color_scale],
        undo_outputs,
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    conversation_events.bind(
        ui.chatbot.undo, undo_message,
        [ui.conversation_state, ui.color_scale],
        undo_outputs,
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    # Clear asks before it takes anything, so the button that opens the
    # question leaves the conversations and background job alone. The
    # confirm button clears them once generation is stopped.
    ui.clear_button.click(
        ask_clear_chat,
        [ui.conversation_state, ui.forks_state],
        [ui.generation_status, ui.clear_confirm, ui.clear_question],
    )
    ui.cancel_clear_button.click(hide_clear_confirm, None, ui.clear_confirm)
    # The question names how many conversations it would take, and that
    # count is read when it is asked. Anything that adds or removes one
    # withdraws it rather than leaving a stale promise above a button
    # that would take more than the promise says - the same reason
    # choosing another model withdraws the removal question. Pressing
    # Clear again re-asks with the numbers as they are now.
    for control in (ui.new_button, ui.fork_button, ui.delete_fork_button):
        control.click(hide_clear_confirm, None, ui.clear_confirm)
    ui.conversation_list.input(hide_clear_confirm, None, ui.clear_confirm)
    ui.brings_its_sampling(conversation_events.bind(
        ui.confirm_clear_button.click, clear_chat,
        clear=True,
        inputs=[ui.color_scale, ui.forks_state],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        outputs=[
            ui.chatbot,
            ui.conversation_state,
            ui.token_strip,
            ui.metrics_state,
            ui.generation_status,
            ui.send_button,
            ui.stop_button,
            ui.token_detail,
            ui.alternatives,
            ui.prompt_strip,
            ui.prompt_metrics_state,
            ui.prompt_note,
            ui.summary_panel,
            ui.surprise_panel,
            ui.trace_state,
            ui.selected_token,
            ui.branch_pick,
            ui.forks_state,
            ui.conversation_list,
            ui.clear_confirm,
        ],
    ))

    # Navigation takes a snapshot of the view; the job keeps its source.
    fork_outputs = [
        ui.prompt,
        ui.chatbot,
        ui.conversation_state,
        ui.forks_state,
        ui.conversation_list,
        ui.generation_status,
        ui.send_button,
        ui.stop_button,
        ui.token_strip,
        ui.metrics_state,
        ui.token_detail,
        ui.alternatives,
        ui.prompt_strip,
        ui.prompt_metrics_state,
        ui.prompt_note,
        ui.summary_panel,
        ui.surprise_panel,
        ui.trace_state,
        ui.selected_token,
        ui.branch_pick,
    ]
    ui.chatbot.select(remember_message, ui.conversation_state, ui.selected_message)

    ui.brings_its_sampling(
        navigate(
            ui.fork_button.click, fork_conversation,
            [
                ui.conversation_state,
                ui.forks_state,
                ui.selected_message,
                ui.color_scale,
                *ui.sampling_controls,
            ],
            fork_outputs,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )
    ui.brings_its_sampling(
        navigate(
            ui.new_button.click, new_conversation,
            [ui.conversation_state, ui.forks_state, ui.color_scale, *ui.sampling_controls],
            [*fork_outputs, ui.branch_text],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )
    # .input rather than .change: the list is also redrawn by the handlers
    # above and the listener below, and a .change listener would switch a
    # second time on each.
    ui.brings_its_sampling(
        navigate(
            ui.conversation_list.input, switch_fork,
            [ui.conversation_list, ui.conversation_state, ui.forks_state, ui.color_scale],
            fork_outputs,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )
    ui.brings_its_sampling(
        conversation_events.bind(
            ui.delete_fork_button.click, delete_fork,
            [ui.conversation_state, ui.forks_state, ui.color_scale],
            fork_outputs,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )
    # Every other path that changes the conversation lands here, and
    # the list's model tag, running indicator and token count
    # follow it. Hide the loading overlay so each streaming frame updates
    # the labels without making the whole list blink.
    ui.conversation_state.change(
        conversation_events.refresh_conversation_list,
        [ui.conversation_state, ui.forks_state, background_state],
        [ui.conversation_list, ui.forks_state],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        show_progress="hidden",
    )
    # And the forks' change, which the listener above fires in turn, is
    # where ordinary view changes are saved. Workers also save independently.
    ui.forks_state.change(
        remember_forks,
        [ui.conversation_state, ui.forks_state],
        None,
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
    # The saved conversations come back first, so the listeners above
    # have something to describe. A page with nothing saved is left as
    # it was built. Background workers persist to their source independently.
    ui.brings_its_sampling(
        conversation_events.bind(
            ui.demo.load, restore_conversations,
            None,
            [ui.chatbot, ui.conversation_state, ui.forks_state, ui.conversation_list, ui.metrics_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        )
    )

    ui.save_button.click(
        save_conversation,
        [ui.conversation_state, ui.system_prompt, *ui.steering_inputs],
        [ui.saved_file, ui.generation_status],
    )
    conversation_events.bind(
        ui.load_upload.upload, load_with_steering,
        [ui.load_upload, ui.conversation_state, ui.color_scale, ui.forks_state],
        [
            ui.chatbot,
            ui.conversation_state,
            ui.system_prompt,
            ui.token_strip,
            ui.metrics_state,
            ui.generation_status,
            ui.token_detail,
            ui.alternatives,
            ui.send_button,
            ui.stop_button,
            ui.prompt_strip,
            ui.prompt_metrics_state,
            ui.prompt_note,
            ui.summary_panel,
            ui.surprise_panel,
            ui.trace_state,
            ui.selected_token,
            ui.branch_pick,
            ui.forks_state,
            *ui.steering_outputs,
        ],
        concurrency_id=CONVERSATION_PANE_QUEUE,
    )
