"""The inspector beside the conversation, and the token strip it reads from."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import gradio as gr

from chatlab import charts
from chatlab.token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
from chatlab.ui.common import NO_TOKEN_SELECTED
from chatlab.ui.conversations import remember_transcript_message
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
from chatlab.ui.layout.common import QUIET_TICK
from chatlab.ui.panel import (
    choose_alternative,
    inspect_token,
    recolor,
    remember_strip_selection,
    select_transcript_token,
    show_token_view,
)
from chatlab.ui.token_edit import close_token_editor, open_token_editor
from chatlab.ui.token_menu import (
    MENU_BRIDGE_CLASS,
    MENU_STRIP_CLASS,
    prompt_menu_payload,
    token_menu_payload,
)


def build_inspector_pane(saved) -> SimpleNamespace:
    """Build the inspector beside the conversation: the clicked token, its alternatives, and the layers behind it."""

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

    return SimpleNamespace(
        alternatives=alternatives,
        attention_layer=attention_layer,
        attention_panel=attention_panel,
        branch_button=branch_button,
        branch_text=branch_text,
        branch_text_button=branch_text_button,
        color_scale=color_scale,
        fitted_model_id=fitted_model_id,
        import_lens_button=import_lens_button,
        import_lens_status=import_lens_status,
        imported_lens=imported_lens,
        inspect_button=inspect_button,
        inspect_status=inspect_status,
        inspection_session=inspection_session,
        inspector_pane=inspector_pane,
        jacobian_controls=jacobian_controls,
        kv_layer=kv_layer,
        kv_metric=kv_metric,
        kv_panel=kv_panel,
        lens_file=lens_file,
        lens_filename=lens_filename,
        lens_mode=lens_mode,
        lens_panel=lens_panel,
        lens_repository=lens_repository,
        lens_setup=lens_setup,
        pinned_concept=pinned_concept,
        pinned_token_id=pinned_token_id,
        prompt_note=prompt_note,
        prompt_strip=prompt_strip,
        scale_caption=scale_caption,
        summary_panel=summary_panel,
        surprise_panel=surprise_panel,
        token_detail=token_detail,
    )


def wire_inspector(ui: SimpleNamespace) -> None:
    """Wire the token strip, its editor and menus, and the inspector's readouts beside it."""

    # The conversation is painted from the turns; the two strips are
    # painted from the measurements they were handed.
    ui.color_scale.change(
        recolor,
        [ui.conversation_state, ui.score_metrics_state, ui.prompt_metrics_state, ui.color_scale],
        [ui.token_strip, ui.score_strip, ui.prompt_strip, ui.scale_caption],
    )
    # Which view of the conversation is on screen. The token view is
    # redrawn on the way in rather than left to the next frame, since the
    # conversation may have moved on while it was hidden.
    ui.token_view.change(
        partial(show_token_view, conversation_id=ui.conversation_state._id),
        [ui.token_view, ui.conversation_state, ui.color_scale],
        [ui.chatbot, ui.token_strip],
        show_progress="hidden",
    )
    editor_outputs = [ui.token_editor, ui.token_edit_text, ui.token_edit_target]
    ui.token_view.change(close_token_editor, outputs=editor_outputs, show_progress="hidden")
    ui.token_edit_cancel.click(close_token_editor, outputs=editor_outputs, show_progress="hidden")
    ui.token_strip.select(
        open_token_editor,
        [ui.conversation_state, ui.metrics_state],
        editor_outputs,
        show_progress="hidden",
    )

    # One click in the conversation answers every question the inspector
    # asks of it, so it is one listener rather than four.
    ui.token_strip.select(
        select_transcript_token,
        [ui.conversation_state, ui.metrics_state],
        [ui.token_detail, ui.alternatives, ui.selected_token, ui.inspect_target, ui.branch_pick],
    )
    ui.token_strip.select(
        token_menu_payload,
        [ui.conversation_state, ui.metrics_state, ui.menu_request],
        ui.menu_response,
        show_progress="hidden",
        queue=False,
    )
    # Also keep the message it landed in, so Fork works from
    # the token view exactly as it does from the chatbot.
    ui.token_strip.select(
        remember_transcript_message, ui.conversation_state, ui.selected_message
    )

    for strip, strip_metrics, source, where in (
        (ui.score_strip, ui.score_metrics_state, "score", "score"),
        (ui.prompt_strip, ui.prompt_metrics_state, "prompt", "prompt"),
    ):
        strip.select(
            inspect_token(source),
            inputs=strip_metrics,
            outputs=[ui.token_detail, ui.alternatives],
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
            [ui.selected_token, ui.branch_pick],
        )
        strip.select(remember_inspect_target(where), strip_metrics, ui.inspect_target)
    # Only the prompt strip's tokens can be replaced: they are the ones a
    # reply was actually generated from, and the ids behind them are kept.
    ui.prompt_strip.select(
        prompt_menu_payload,
        [ui.prompt_metrics_state, ui.context_ids_state, ui.prompt_menu_request],
        ui.prompt_menu_response,
        show_progress="hidden",
        queue=False,
    )
    ui.alternatives.select(
        choose_alternative,
        [ui.conversation_state, ui.score_metrics_state, ui.prompt_metrics_state, ui.selected_token],
        [ui.token_detail, ui.branch_pick],
    )
    inspection_outputs = [ui.lens_panel, ui.attention_panel, ui.insight_state, ui.inspect_status]
    ui.inspect_button.click(
        inspect_layers,
        [
            ui.inspect_target,
            ui.metrics_state,
            ui.prompt_metrics_state,
            ui.context_ids_state,
            ui.attention_layer,
            ui.score_metrics_state,
            ui.score_context_ids_state,
            ui.chat_metrics_state,
            ui.chat_context_ids_state,
            ui.lens_mode,
            ui.imported_lens,
            ui.pinned_concept,
            ui.inspection_session,
            ui.pinned_token_id,
        ],
        [ui.lens_panel, ui.attention_panel, ui.attention_layer, ui.insight_state, ui.inspect_status],
    )
    ui.attention_layer.release(
        render_attention, [ui.insight_state, ui.attention_layer], ui.attention_panel
    )
    # Every readout, and every reset that takes one away, passes through
    # the insight state, so the cache view follows it there. The cache is
    # read from memory rather than rebuilt, so each control reads again.
    kv_inputs = [ui.insight_state, ui.kv_layer, ui.kv_metric]
    ui.insight_state.change(render_kv_cache, kv_inputs, [ui.kv_panel, ui.kv_layer], **QUIET_TICK)
    ui.kv_layer.release(render_kv_cache, kv_inputs, [ui.kv_panel, ui.kv_layer])
    ui.kv_metric.change(render_kv_cache, kv_inputs, [ui.kv_panel, ui.kv_layer])
    ui.lens_mode.change(
        change_lens_mode, [ui.lens_mode, ui.inspection_session],
        [ui.jacobian_controls, ui.attention_layer, *inspection_outputs],
        queue=False,
    )
    ui.import_lens_button.click(
        import_jacobian_lens, [ui.lens_file, ui.fitted_model_id, ui.lens_repository, ui.lens_filename],
        [ui.imported_lens, ui.import_lens_status],
    )
    # QUIET_TICK for the same reason as the metrics_state handler below:
    # clearing a stale readout is instant, so the panel should not flash a
    # spinner over itself on the way.
    ui.imported_lens.change(
        reset_inspection, ui.insight_state, inspection_outputs, **QUIET_TICK,
    )
    ui.imported_lens.change(
        lambda imported: gr.update(open=False) if imported else gr.skip(),
        ui.imported_lens, ui.lens_setup,
    )
    ui.pinned_concept.input(
        change_pinned_token, [ui.pinned_concept, ui.inspection_session], inspection_outputs,
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
    ui.metrics_state.change(
        reset_inspection, ui.insight_state, inspection_outputs, **QUIET_TICK,
    )
