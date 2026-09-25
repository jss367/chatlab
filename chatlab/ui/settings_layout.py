"""The Settings page, and the listeners that keep the settings file in step.

The handlers live in ui.settings_page. This module draws the page they act
on, inside the Blocks ui.layout.build_app() opens, and wires every control
whose value is saved between sessions, wherever in the app that control sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

from chatlab import settings, themes
from chatlab.thinking import THINKING_CHOICES
from chatlab.ui import runtime
from chatlab.ui.common import CONVERSATION_PANE_QUEUE
from chatlab.ui.conversations import remember_branch_sampling
from chatlab.ui.extensions_page import build_extension_settings
from chatlab.ui.icons import icon_classes
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
from chatlab.ui.styles import set_message_box_keys

if TYPE_CHECKING:
    from chatlab.ui.chat_layout import Sampling
    from chatlab.ui.models_layout import ModelsPage


@dataclass(frozen=True)
class SettingsPage:
    """The Settings page's column and every control on it."""

    column: gr.Column
    system_prompt: gr.Textbox
    assistant_prefill: gr.Textbox
    thinking_mode: gr.Radio
    keep_reasoning: gr.Checkbox
    enter_sends: gr.Checkbox
    writing_suggestions: gr.Checkbox
    analyze_prompt: gr.Checkbox
    theme_choice: gr.Dropdown
    appearance_choice: gr.Radio
    prefill_token_limit: gr.Number
    hardware_view: gr.Markdown
    refresh_hardware_button: gr.Button
    extension_settings: list
    active_extensions: gr.State


def build_settings_page(
    saved: settings.Settings, extensions: list, extension_errors: list[str]
) -> SettingsPage:
    """The Settings page, hidden until the nav picks it.

    ``extensions`` are the ones this build enabled, and ``extension_errors``
    what went wrong loading or opening any of them, which the extensions card
    lists.
    """

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
    return SettingsPage(
        column=settings_page,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
        thinking_mode=thinking_mode,
        keep_reasoning=keep_reasoning,
        enter_sends=enter_sends,
        writing_suggestions=writing_suggestions,
        analyze_prompt=analyze_prompt,
        theme_choice=theme_choice,
        appearance_choice=appearance_choice,
        prefill_token_limit=prefill_token_limit,
        hardware_view=hardware_view,
        refresh_hardware_button=refresh_hardware_button,
        extension_settings=extension_settings,
        active_extensions=active_extensions,
    )


def wire_message_box(settings_page: SettingsPage, prompt: gr.Textbox) -> None:
    """Which key sends the Chat tab's message, as the Input card says."""

    settings_page.enter_sends.change(set_message_box_keys, settings_page.enter_sends, prompt)


def wire_settings_persistence(
    demo: gr.Blocks,
    saved: settings.Settings,
    theme_style: gr.HTML,
    settings_page: SettingsPage,
    models: ModelsPage,
    sampling: Sampling,
    color_scale: gr.Dropdown,
    forks_state: gr.State,
    settings_inputs: list,
) -> None:
    """Save every setting as it changes, and read the file back on a reload.

    The controls saved here are spread over three pages - the sampling and
    the color scale on the Chat page, the model ID and its precision on the
    Models page, the rest on this one - so they are wired together, in the
    order the file names them. ``settings_inputs`` are the ones every reply
    reads as well, in the order the generation handlers take them.
    """

    # Everything saved between sessions, in PERSISTED_SETTING_NAMES order.
    persisted_inputs = [
        *settings_inputs,
        settings_page.thinking_mode,
        settings_page.enter_sends,
        settings_page.writing_suggestions,
        settings_page.theme_choice,
        settings_page.appearance_choice,
        models.model_id,
        models.weight_precision,
    ]
    for control in (
        settings_page.thinking_mode,
        settings_page.system_prompt,
        settings_page.keep_reasoning,
        settings_page.assistant_prefill,
        sampling.randomize_seed,
        settings_page.analyze_prompt,
        color_scale,
        settings_page.enter_sends,
        settings_page.writing_suggestions,
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
    settings_page.theme_choice.change(
        remember_settings, persisted_inputs, None, trigger_mode="always_last"
    )
    settings_page.theme_choice.change(
        apply_theme,
        settings_page.theme_choice,
        theme_style,
        trigger_mode="always_last",
    )
    # Light or dark is wired the same way and for the same reasons, except
    # that the repaint is the browser's own work rather than a round trip:
    # the class it toggles is already what every dark-mode rule reads.
    settings_page.appearance_choice.change(
        remember_settings, persisted_inputs, None, trigger_mode="always_last"
    )
    settings_page.appearance_choice.change(None, settings_page.appearance_choice, None, js=themes.APPEARANCE_JS)
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
    for control in sampling.controls:
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
        sampling.resets, settings.CONVERSATION_SAMPLING, sampling.controls, strict=True
    ):
        button.click(
            partial(reset_sampling, name, saved.prefill_token_limit),
            None,
            control,
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            remember_branch_sampling,
            [forks_state, *sampling.controls],
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
            sampling.controls,
            sampling.accordion,
            show_progress="hidden",
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
    # The seed box is the one control the app writes to itself: a finished
    # response leaves the seed that produced it there, and saving that
    # would overwrite the seed the reader chose. Blur and submit are the
    # two ways a person is done editing a number, and they are the only
    # events that write the box's contents down; every other control
    # leaves the saved seed where it is. See remember_settings().
    for event in (sampling.seed.blur, sampling.seed.submit):
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
    for event in (settings_page.prefill_token_limit.blur, settings_page.prefill_token_limit.submit):
        event(
            remember_prefill_limit,
            [settings_page.prefill_token_limit, sampling.max_new_tokens, forks_state, *sampling.controls],
            [settings_page.prefill_token_limit, sampling.max_new_tokens, forks_state],
            concurrency_id=CONVERSATION_PANE_QUEUE,
        ).then(
            update_sampling_label,
            sampling.controls,
            sampling.accordion,
            concurrency_id=SAMPLING_LABEL_QUEUE,
        )
    # A page load is where the file is read back, so reloading the browser
    # shows what was saved rather than what the app started with. The
    # sampling summary is rebuilt from whatever came back, since the label
    # the accordion was built with describes the file as it was read at
    # startup, not as it is now.
    demo.load(
        restore_settings, None, [*persisted_inputs, settings_page.prefill_token_limit]
    ).then(
        apply_theme, settings_page.theme_choice, theme_style
    ).then(
        None, settings_page.appearance_choice, None, js=themes.APPEARANCE_JS
    ).then(
        update_sampling_label,
        sampling.controls,
        sampling.accordion,
        concurrency_id=SAMPLING_LABEL_QUEUE,
    )
