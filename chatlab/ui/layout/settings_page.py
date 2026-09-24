"""The Settings page, and the wiring that saves each setting and reads it back."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import gradio as gr

from chatlab import settings, themes
from chatlab.thinking import THINKING_CHOICES
from chatlab.ui import runtime
from chatlab.ui.conversations import remember_branch_sampling
from chatlab.ui.extensions_page import build_extension_settings
from chatlab.ui.icons import icon_classes
from chatlab.ui.layout.common import CONVERSATION_PANE_QUEUE
from chatlab.ui.scoring import SAMPLING_LABEL_QUEUE
from chatlab.ui.settings_page import (
    apply_theme,
    hardware_card,
    remember_committed_seed,
    remember_prefill_limit,
    remember_settings,
    reset_sampling,
    restore_settings,
    update_sampling_label,
)


def build_settings_page(extension_errors, extensions, saved) -> SimpleNamespace:
    """Build the Settings page and hand back the controls the wiring reads."""

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

    return SimpleNamespace(
        active_extensions=active_extensions,
        analyze_prompt=analyze_prompt,
        appearance_choice=appearance_choice,
        assistant_prefill=assistant_prefill,
        enter_sends=enter_sends,
        extension_settings=extension_settings,
        hardware_view=hardware_view,
        keep_reasoning=keep_reasoning,
        prefill_token_limit=prefill_token_limit,
        refresh_hardware_button=refresh_hardware_button,
        settings_page=settings_page,
        system_prompt=system_prompt,
        theme_choice=theme_choice,
        thinking_mode=thinking_mode,
        writing_suggestions=writing_suggestions,
    )


def wire_settings(ui: SimpleNamespace) -> None:
    """Save every setting as it changes, and read the file back on each page load."""

    settings_inputs = [
        ui.system_prompt,
        ui.keep_reasoning,
        ui.assistant_prefill,
        ui.temperature,
        ui.top_p,
        ui.top_k,
        ui.skip_top_below,
        ui.max_new_tokens,
        ui.seed,
        ui.randomize_seed,
        ui.analyze_prompt,
        ui.color_scale,
    ]
    # Persistence runs separately; every request must snapshot the controls
    # the reader sees, even while remember_steering is still queued.
    steering_inputs = [ui.steering_state, ui.steering_enabled, ui.steering_strength, ui.steering_layer]
    chat_inputs = [ui.prompt, ui.conversation_state, *settings_inputs, *steering_inputs, ui.thinking_mode]

    # Everything saved between sessions, in PERSISTED_SETTING_NAMES order.
    persisted_inputs = [
        *settings_inputs,
        ui.thinking_mode,
        ui.enter_sends,
        ui.writing_suggestions,
        ui.theme_choice,
        ui.appearance_choice,
        ui.model_id,
        ui.weight_precision,
    ]
    for control in (
        ui.thinking_mode,
        ui.system_prompt,
        ui.keep_reasoning,
        ui.assistant_prefill,
        ui.randomize_seed,
        ui.analyze_prompt,
        ui.color_scale,
        ui.enter_sends,
        ui.writing_suggestions,
        ui.model_id,
        ui.weight_precision,
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
    ui.theme_choice.change(
        remember_settings, persisted_inputs, None, trigger_mode="always_last"
    )
    ui.theme_choice.change(
        apply_theme,
        ui.theme_choice,
        ui.theme_style,
        trigger_mode="always_last",
    )
    # Light or dark is wired the same way and for the same reasons, except
    # that the repaint is the browser's own work rather than a round trip:
    # the class it toggles is already what every dark-mode rule reads.
    ui.appearance_choice.change(
        remember_settings, persisted_inputs, None, trigger_mode="always_last"
    )
    ui.appearance_choice.change(None, ui.appearance_choice, None, js=themes.APPEARANCE_JS)
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
    for control in ui.sampling_controls:
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
        ui.sampling_resets, settings.CONVERSATION_SAMPLING, ui.sampling_controls, strict=True
    ):
        button.click(
            partial(reset_sampling, name, ui.saved.prefill_token_limit),
            None,
            control,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            remember_branch_sampling,
            [ui.forks_state, *ui.sampling_controls],
            ui.forks_state,
            show_progress="hidden",
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            remember_settings,
            persisted_inputs,
            None,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            update_sampling_label,
            ui.sampling_controls,
            ui.sampling_accordion,
            show_progress="hidden",
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
    # The seed box is the one control the app writes to itself: a finished
    # response leaves the seed that produced it there, and saving that
    # would overwrite the seed the reader chose. Blur and submit are the
    # two ways a person is done editing a number, and they are the only
    # events that write the box's contents down; every other control
    # leaves the saved seed where it is. See remember_settings().
    for event in (ui.seed.blur, ui.seed.submit):
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
    for event in (ui.prefill_token_limit.blur, ui.prefill_token_limit.submit):
        event(
            remember_prefill_limit,
            [ui.prefill_token_limit, ui.max_new_tokens, ui.forks_state, *ui.sampling_controls],
            [ui.prefill_token_limit, ui.max_new_tokens, ui.forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            update_sampling_label,
            ui.sampling_controls,
            ui.sampling_accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
    # A page load is where the file is read back, so reloading the browser
    # shows what was saved rather than what the app started with. The
    # sampling summary is rebuilt from whatever came back, since the label
    # the accordion was built with describes the file as it was read at
    # startup, not as it is now.
    ui.demo.load(
        restore_settings, None, [*persisted_inputs, ui.prefill_token_limit]
    ).then(
        apply_theme, ui.theme_choice, ui.theme_style
    ).then(
        None, ui.appearance_choice, None, js=themes.APPEARANCE_JS
    ).then(
        update_sampling_label,
        ui.sampling_controls,
        ui.sampling_accordion,
        concurrency_id=SAMPLING_LABEL_QUEUE,
    )

    # What the wiring after this reads.
    ui.chat_inputs = chat_inputs
    ui.steering_inputs = steering_inputs
