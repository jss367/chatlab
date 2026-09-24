"""The Score text and Prompts tabs, and their wiring."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from chatlab.token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
from chatlab.ui.prompts import (
    BATCH_HEADERS,
    PROMPT_COUNT_HINT,
    count_prompts,
    load_prompt_file,
    run_prompts,
    stop_batch,
)
from chatlab.ui.scoring import SCORE_COUNT_HINT, score_text


def build_prompts_tab() -> SimpleNamespace:
    """Build the Prompts tab, which runs a list of prompts each in a conversation of its own."""

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

    return SimpleNamespace(
        batch_files=batch_files,
        batch_results=batch_results,
        batch_status=batch_status,
        prompt_count=prompt_count,
        prompts_box=prompts_box,
        prompts_upload=prompts_upload,
        run_prompts_button=run_prompts_button,
        stop_prompts_button=stop_prompts_button,
    )


def build_score_tab() -> SimpleNamespace:
    """Build the Score text tab, for measuring text the model did not write."""

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

    return SimpleNamespace(
        score_budget=score_budget,
        score_button=score_button,
        score_context=score_context,
        score_input=score_input,
        score_status=score_status,
        score_strip=score_strip,
        use_chat_template=use_chat_template,
    )


def wire_prompts(ui: SimpleNamespace) -> None:
    """Wire the Score text and Prompts tabs."""

    ui.score_button.click(
        score_text,
        [ui.score_context, ui.score_input, ui.use_chat_template, ui.color_scale],
        [
            ui.score_strip,
            ui.metrics_state,
            ui.score_metrics_state,
            ui.prompt_strip,
            ui.prompt_metrics_state,
            ui.prompt_note,
            ui.summary_panel,
            ui.surprise_panel,
            ui.score_status,
            ui.token_detail,
            ui.alternatives,
            ui.selected_token,
            ui.branch_pick,
            ui.context_ids_state,
            ui.score_context_ids_state,
        ],
    )
    # A batch reads its prompts from the box and everything else from the
    # controls the Chat tab and Settings already own, so there is nothing
    # to set up before running one.
    batch_outputs = [
        ui.batch_status,
        ui.batch_results,
        ui.run_prompts_button,
        ui.stop_prompts_button,
        ui.batch_files,
        ui.batch_directory_state,
    ]
    batch_run = ui.run_prompts_button.click(
        run_prompts,
        [
            ui.prompts_box,
            ui.loaded_prompts_state,
            ui.system_prompt,
            ui.assistant_prefill,
            ui.temperature,
            ui.top_p,
            ui.top_k,
            ui.skip_top_below,
            ui.max_new_tokens,
            ui.seed,
            ui.randomize_seed,
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
    ui.stop_prompts_button.click(
        stop_batch, ui.batch_directory_state, batch_outputs, cancels=[batch_run]
    )
    ui.prompts_box.change(
        count_prompts,
        [ui.prompts_box, ui.loaded_prompts_state],
        ui.prompt_count,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    ui.prompts_upload.upload(
        load_prompt_file,
        [ui.prompts_upload, ui.prompts_box, ui.loaded_prompts_state],
        [ui.prompts_box, ui.batch_status, ui.loaded_prompts_state],
    )
