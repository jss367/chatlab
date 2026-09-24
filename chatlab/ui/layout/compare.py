"""The Compare tab, and its wiring."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import gradio as gr

from chatlab import charts
from chatlab.compare import (
    CONFIGURATION_HEADERS,
    DIVERGENCE_HEADERS,
    GAP_CAPTION,
    GAP_COLORS,
    MEASUREMENT,
    REPLY,
)
from chatlab.compare import EMPTY_SLOT as COMPARE_SLOT_EMPTY
from chatlab.ui import experiment_compare, experiments
from chatlab.ui.activation_patching import build as build_activation_patching
from chatlab.ui.compare import (
    COMPARE_EMPTY,
    clear_slots,
    download_comparison,
    fill_slot,
    mode_controls,
    stop_comparison,
)
from chatlab.ui.compare import render as render_comparison


def build_compare_tab(compare_a_state, compare_b_state, compare_export_state) -> SimpleNamespace:
    """Build the Compare tab: two runs side by side, and activation patching between them."""

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

    return SimpleNamespace(
        compare_a_heading=compare_a_heading,
        compare_a_strip=compare_a_strip,
        compare_b_heading=compare_b_heading,
        compare_b_strip=compare_b_strip,
        compare_chart=compare_chart,
        compare_clear=compare_clear,
        compare_headline=compare_headline,
        compare_mode=compare_mode,
        compare_prompt=compare_prompt,
        compare_rows=compare_rows,
        compare_run_a=compare_run_a,
        compare_run_b=compare_run_b,
        compare_settings=compare_settings,
        compare_status=compare_status,
        compare_stop=compare_stop,
        compare_template=compare_template,
        compare_text=compare_text,
        compare_tiles=compare_tiles,
        difference_detail=difference_detail,
        difference_position=difference_position,
        next_difference=next_difference,
        pair_button=pair_button,
        pair_conditions=pair_conditions,
        pair_load_status=pair_load_status,
    )


def wire_compare(ui: SimpleNamespace) -> None:
    """Wire the Compare tab: filling the two slots, pairing conditions, and stepping through differences."""

    compare_outputs = [
        ui.compare_a_heading,
        ui.compare_b_heading,
        ui.compare_a_strip,
        ui.compare_b_strip,
        ui.compare_tiles,
        ui.compare_chart,
        ui.compare_headline,
        ui.compare_settings,
        ui.compare_rows,
        ui.compare_export_state,
    ]
    # Everything a run reads. The sampling controls and the steering
    # vector are the Chat tab's own, so a slot is filled under exactly
    # the settings a reply typed by hand would have used - which is what
    # makes changing one of them between A and B a clean experiment.
    compare_inputs = [
        ui.compare_mode,
        ui.compare_prompt,
        ui.compare_text,
        ui.compare_template,
        ui.system_prompt,
        ui.assistant_prefill,
        ui.temperature,
        ui.top_p,
        ui.top_k,
        ui.skip_top_below,
        ui.max_new_tokens,
        ui.seed,
        ui.randomize_seed,
        ui.thinking_mode,
        *ui.steering_inputs,
    ]
    compare_slot_outputs = [ui.compare_status, ui.compare_run_a, ui.compare_run_b, ui.compare_stop]
    compare_runs = []
    for slot, button, held in (
        ("A", ui.compare_run_a, ui.compare_a_state),
        ("B", ui.compare_run_b, ui.compare_b_state),
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
            [ui.compare_a_state, ui.compare_b_state],
            compare_outputs,
        )
        compare_runs.append(filling)
    paired = ui.pair_button.click(
        experiment_compare.run_pair, [*ui.pair_conditions, *compare_inputs],
        [ui.compare_a_state, ui.compare_b_state, *compare_slot_outputs, ui.pair_button, ui.pair_load_status],
    )
    compare_runs.append(paired)
    ui.next_difference.click(experiment_compare.next_difference,
                          [ui.compare_export_state, ui.difference_position],
                          [ui.difference_position, ui.difference_detail])
    ui.compare_export_state.change(lambda: (None, ""), None, [ui.difference_position, ui.difference_detail],
                                show_progress="hidden")
    experiments.wire(ui.experiments_view, ui.demo, ui.trace_state, ui.chat_context_ids_state,
                     ui.compare_a_state, ui.compare_b_state, ui.insight_state, ui.inspect_target)
    ui.conversation_tabs.select(experiments.inspector_visibility, None, [ui.inspector_pane, ui.inspector_resizer],
                             show_progress="hidden", queue=False)
    for held in (ui.compare_a_state, ui.compare_b_state):
        held.change(render_comparison, [ui.compare_a_state, ui.compare_b_state], compare_outputs)
    # Cancelling closes the run at its last yield, which is what gives the
    # model lock back; this only puts the buttons right. The slot keeps
    # whatever it held before, because half a response is not a run.
    ui.compare_stop.click(
        stop_comparison, None, compare_slot_outputs, cancels=compare_runs
    ).then(lambda: gr.update(interactive=True), None, ui.pair_button)
    # cancels, because clearing during a run is otherwise undone by the
    # run: its remaining frames would write over the cleared status and
    # its last one would put the slot back. See clear_slots().
    ui.compare_clear.click(
        clear_slots,
        None,
        [ui.compare_a_state, ui.compare_b_state, *compare_slot_outputs, *compare_outputs],
        cancels=compare_runs,
    ).then(lambda: gr.update(interactive=True), None, ui.pair_button)
    ui.compare_mode.change(
        mode_controls,
        ui.compare_mode,
        [ui.compare_prompt, ui.compare_text, ui.compare_template],
    )
