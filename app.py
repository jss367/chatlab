"""Chatlab interface for chatting with and inspecting model tokens."""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

import gradio as gr

import charts
from model_runtime import PROMPT_SCORE_LIMIT, ModelManager
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
    UNSCORED_BEYOND_LIMIT,
    category_for,
    summarize,
)
from trace_export import build_trace, write_trace_export


DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"
MANAGER = ModelManager()

# Redrawing the trace on every streamed token is wasted work, so it catches up
# in batches and again once the response finishes.
CHART_EVERY = 16

SELECT_HINT = "Select a token to inspect it."

# This tokenizer offers neither offsets nor a decode that round trips, so
# where the context ends had to be counted out rather than confirmed. The
# scored tokens are still the whole passage's own single encoding, so every
# probability is exact; what is uncertain is where the line between the two
# halves was drawn, and a line a token out moves that token between the two
# tables and the summary figures they feed.
SEAM_CAVEAT = (
    "Approximate split: this tokenizer could not confirm where the context "
    "ends, so the boundary between it and the scored text may sit a token "
    "off. Every probability shown is the full passage's own either way."
)

# The chat-message box was ticked for a model that ships no chat template, so
# there was no turn to wrap the context in. The numbers are exact — they are
# the plain passage's own — but they are not the framing the box promised, and
# the difference is the reader's to know about.
TEMPLATE_CAVEAT = (
    "Plain text, not a chat turn: this model has no chat template, so the "
    "context was measured as ordinary characters in front of the text."
)


def status_card(title: str, detail: str, tone: str = "neutral") -> str:
    icon = {"success": "●", "error": "●", "working": "◌"}.get(tone, "○")
    return f"### {icon} {title}\n\n{detail}"


def download_model(model_id: str, hf_token: str):
    started = time.monotonic()
    yield status_card(
        "Downloading model",
        f"Fetching `{model_id.strip()}` into the Hugging Face cache. Large models may take a while.",
        "working",
    )
    try:
        path = MANAGER.download(model_id, hf_token)
    except Exception as error:
        yield status_card("Download failed", html.escape(str(error)), "error")
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Download complete",
        f"Cached `{model_id.strip()}` in `{path}` ({elapsed:.1f} seconds). Use **Load model** when ready.",
        "success",
    )


def download_and_load_model(model_id: str, hf_token: str):
    started = time.monotonic()
    yield status_card(
        "Downloading model",
        f"Fetching `{model_id.strip()}` into the Hugging Face cache. Existing files are reused.",
        "working",
    )
    try:
        path = MANAGER.download(model_id, hf_token)
        yield status_card(
            "Loading model",
            f"Downloaded `{model_id.strip()}`. Moving the weights onto the best available device…",
            "working",
        )
        device = MANAGER.load(model_id, path)
    except Exception as error:
        yield status_card("Model setup failed", html.escape(str(error)), "error")
        return

    elapsed = time.monotonic() - started
    yield status_card(
        "Model ready",
        f"`{model_id.strip()}` is loaded on **{device}** ({elapsed:.1f} seconds total).",
        "success",
    )


def load_cached_model(model_id: str):
    yield status_card(
        "Finding cached model", f"Looking for `{model_id.strip()}` locally…", "working"
    )
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=model_id.strip(), local_files_only=True))
        yield status_card(
            "Loading model", f"Loading cached files from `{path}`…", "working"
        )
        device = MANAGER.load(model_id, path)
    except Exception as error:
        yield status_card(
            "Could not load cached model", html.escape(str(error)), "error"
        )
        return
    yield status_card(
        "Model ready", f"`{model_id.strip()}` is loaded on **{device}**.", "success"
    )


def unload_model():
    if not MANAGER.loaded:
        return status_card("No model loaded", "There is nothing to unload.")
    MANAGER.unload()
    return status_card("Model unloaded", "Model memory has been released.", "success")


def resolve_scale(scale_name: str):
    return COLOR_SCALES.get(scale_name) or COLOR_SCALES[DEFAULT_COLOR_SCALE]


def strip_value(metrics: list[dict], scale_name: str) -> list[tuple[str, str]]:
    """Bucket every token for the strip under one color scale."""

    scale = resolve_scale(scale_name)
    return [
        (metric["display_text"], category_for(metric, scale.name))
        for metric in metrics or []
    ]


def strip_update(metrics: list[dict], scale_name: str, label: str | None = None):
    """Repaint a token strip, legend and all.

    Streaming updates send the value alone, because rebuilding the component
    for every token to carry an unchanged legend is wasted work.
    """

    scale = resolve_scale(scale_name)
    update = {"value": strip_value(metrics, scale.name), "color_map": scale.color_map}
    if label is not None:
        update["label"] = label
    return gr.update(**update)


def recolor(metrics: list[dict], prompt_metrics: list[dict], scale_name: str):
    scale = resolve_scale(scale_name)
    return (
        strip_update(metrics, scale.name),
        strip_update(prompt_metrics, scale.name),
        scale.caption,
    )


def inspect_token(metrics: list[dict], event: gr.SelectData):
    if not metrics:
        return SELECT_HINT, []

    index = event.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    try:
        metric = metrics[int(index)]
    except (IndexError, TypeError, ValueError):
        return "That token is no longer available. Generate another response.", []

    token_repr = html.escape(repr(metric["text"]))
    where = "Prompt token" if metric["segment"] == "prompt" else "Token"
    if not metric.get("scored", True):
        if metric.get("unscored_reason") == UNSCORED_BEYOND_LIMIT:
            why = (
                f"Only the most recent {PROMPT_SCORE_LIMIT:,} tokens of a long "
                "prompt are scored, and this one sits before that window, so it "
                "was skipped."
            )
        else:
            why = "Nothing came before this token, so the model never predicted it."
        return (
            f"### {where} {metric['position']}: `{token_repr}`\n\n"
            f"{why}\n\n"
            f"- **Token ID:** {metric['token_id']:,}",
            [],
        )

    summary = (
        f"### {where} {metric['position']}: `{token_repr}`\n\n"
        f"- **Raw rank:** {metric['raw_rank']:,}\n"
        f"- **Raw model probability:** {metric['raw_probability']:.5%}\n"
        f"- **Actual sampling probability:** {metric['sampling_probability']:.5%}\n"
        f"- **Surprise:** {metric['surprise_bits']:.2f} bits\n"
        f"- **Distribution entropy:** {metric['entropy_bits']:.2f} bits\n"
        f"- **Top-1 margin:** {metric['top1_margin']:.2%} between the model's first and second choice\n"
        f"- **Sampling shift:** {metric['sampling_shift_bits']:+.2f} bits versus the raw model\n"
        f"- **Probability mass above it:** {metric['probability_mass_above']:.2%}\n"
        f"- **Token ID:** {metric['token_id']:,}"
    )
    rows = [
        [candidate["token_id"], repr(candidate["text"]), candidate["probability"]]
        for candidate in metric["top_candidates"]
    ]
    return summary, rows


def prompt_note_text(count: int, note: str, kind: str) -> str:
    if not count:
        return ""
    text = f"{count:,} {kind} tokens. The first one has no prediction behind it."
    if note:
        text = f"{text} {note}"
    return text


def chat(
    prompt: str,
    conversation: list[dict] | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    analyze_prompt: bool,
    scale_name: str,
):
    skip = gr.skip()
    conversation = list(conversation or [])
    prompt = prompt.strip()
    if not prompt:
        yield (skip,) * 5 + ("Enter a message first.",) + (skip,) * 8
        return
    if not MANAGER.loaded:
        yield (skip,) * 5 + ("Download and load a model first.",) + (skip,) * 8
        return

    request_messages = conversation + [{"role": "user", "content": prompt}]
    display = request_messages + [{"role": "assistant", "content": ""}]
    yield (
        "",
        list(display),
        conversation,
        strip_update([], scale_name, "Response tokens — click one"),
        [],
        "Reading the prompt…",
        strip_update([], scale_name),
        [],
        "",
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        SELECT_HINT,
        [],
        {},
    )

    metrics: list[dict] = []
    first = True
    try:
        last_update = None
        for update in MANAGER.generate(
            request_messages,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed),
            analyze_prompt=bool(analyze_prompt),
        ):
            last_update = update
            metrics = update.metrics
            display[-1] = {"role": "assistant", "content": update.text}
            count = len(metrics)
            refresh = first or count % CHART_EVERY == 0
            yield (
                "",
                list(display),
                list(display),
                strip_value(metrics, scale_name),
                metrics,
                f"Generated {count} token{'s' if count != 1 else ''}.",
                strip_update(update.prompt_metrics, scale_name)
                if first
                else skip,
                update.prompt_metrics if first else skip,
                prompt_note_text(
                    len(update.prompt_metrics), update.prompt_note, "prompt"
                )
                if first
                else skip,
                charts.summary_tiles(summarize(metrics)) if refresh else skip,
                charts.surprise_chart(metrics) if refresh else skip,
                skip,
                skip,
                skip,
            )
            first = False
    except Exception as error:
        display[-1] = {"role": "assistant", "content": f"Generation failed: {error}"}
        # A failed response is not a response to export, so the trace opened
        # at the top of this run stays empty.
        yield (
            "",
            list(display),
            conversation,
            skip,
            skip,
            f"Generation failed: {error}",
        ) + (skip,) * 7 + ({},)
        return

    trace = (
        build_trace(
            model_id=MANAGER.model_id,
            messages=request_messages,
            response=last_update.text,
            sampling={
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_new_tokens": int(max_new_tokens),
                "seed": int(seed),
            },
            metrics=metrics,
        )
        if last_update is not None and metrics
        else {}
    )
    status = f"Generated {len(metrics)} token{'s' if len(metrics) != 1 else ''}."
    if trace:
        status = f"{status} Exports are ready."
    yield (skip,) * 5 + (
        status,
        skip,
        skip,
        skip,
        charts.summary_tiles(summarize(metrics)),
        charts.surprise_chart(metrics),
        skip,
        skip,
        trace,
    )


def score_text(
    context: str,
    text: str,
    use_chat_template: bool,
    scale_name: str,
):
    skip = gr.skip()
    if not MANAGER.loaded:
        return (skip,) * 7 + ("Download and load a model first.", skip, skip)

    try:
        result = MANAGER.score_text(
            text, context=context or "", use_chat_template=bool(use_chat_template)
        )
    except Exception as error:
        return (skip,) * 7 + (f"Could not score that text: {error}", skip, skip)

    summary = summarize(result.metrics)
    status = (
        f"Scored {summary['token_count']:,} tokens. "
        f"Perplexity {summary['perplexity']:,.1f}."
    )
    # What was scored comes before how exactly it was scored: the template
    # caveat says which passage the numbers describe, the seam caveat says how
    # sure their first token is.
    if result.chat_template_missing:
        status = f"{status} {TEMPLATE_CAVEAT}"
    if not result.seam_verified:
        status = f"{status} {SEAM_CAVEAT}"
    return (
        strip_update(result.metrics, scale_name, "Scored tokens — click one"),
        result.metrics,
        strip_update(result.context_metrics, scale_name),
        result.context_metrics,
        prompt_note_text(len(result.context_metrics), "", "context"),
        charts.summary_tiles(summary),
        charts.surprise_chart(result.metrics, title="Surprise per scored token"),
        status,
        SELECT_HINT,
        [],
    )


def clear_chat(scale_name: str):
    return (
        [],
        [],
        strip_update([], scale_name, "Response tokens — click one"),
        [],
        strip_update([], scale_name),
        [],
        "",
        "Conversation cleared.",
        SELECT_HINT,
        [],
        charts.summary_tiles({}),
        charts.EMPTY_CHART,
        {},
    )


CSS = """
.gradio-container { max-width: 1500px !important; }
#hero { padding: 0.5rem 0 0.2rem; }
#hero h1 { font-size: 2.1rem; margin-bottom: 0.25rem; }
#model-status { min-height: 128px; }
#token-strip { min-height: 150px; }
#token-strip span, #prompt-strip span { cursor: pointer; border-radius: 5px; }
/* Token fills are light in both themes, so their ink is pinned dark. */
#token-strip .textspan.hl, #prompt-strip .textspan.hl,
#token-strip .category-label, #prompt-strip .category-label { color: #0b0b0b; }
.footer-note { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.scale-caption { color: var(--body-text-color-subdued); font-size: 0.85rem; }

.viz-root {
  --viz-ink: #0b0b0b;
  --viz-muted: #898781;
  --viz-grid: #e1e0d9;
  --viz-axis: #c3c2b7;
  --viz-line: #2a78d6;
  --viz-band: #cde2fb;
  margin: 0;
  font-family: var(--font, system-ui, -apple-system, "Segoe UI", sans-serif);
}
.dark .viz-root {
  --viz-ink: #ffffff;
  --viz-muted: #898781;
  --viz-grid: #2c2c2a;
  --viz-axis: #383835;
  --viz-line: #3987e5;
  --viz-band: #1c5cab;
}
.viz-root svg { width: 100%; height: auto; display: block; }
.viz-title { color: var(--viz-ink); font-size: 0.9rem; font-weight: 600; padding: 0 0 0.2rem; }
.viz-sub { color: var(--viz-muted); font-weight: 400; font-size: 0.8rem; margin-left: 0.4rem; }
.viz-grid { stroke: var(--viz-grid); stroke-width: 1; }
.viz-axis { stroke: var(--viz-axis); stroke-width: 1; }
.viz-band { fill: var(--viz-band); opacity: 0.55; stroke: none; }
.viz-line { fill: none; stroke: var(--viz-line); stroke-width: 2; stroke-linejoin: round; }
.viz-peak-dot { fill: var(--viz-line); stroke: var(--body-background-fill); stroke-width: 2; }
.viz-peak-label, .viz-tick { fill: var(--viz-muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.viz-hit { fill: transparent; }
.viz-empty, .viz-note { color: var(--body-text-color-subdued); font-size: 0.85rem; padding: 0.4rem 0; }
.viz-tiles { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.viz-tile {
  flex: 1 1 5.5rem; padding: 0.45rem 0.6rem; border-radius: 8px;
  background: var(--background-fill-secondary);
}
.viz-value { color: var(--viz-ink); font-size: 1.25rem; line-height: 1.2; }
.viz-label { color: var(--viz-muted); font-size: 0.72rem; text-transform: lowercase; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Chatlab", css=CSS, theme=gr.themes.Soft()
    ) as demo:
        conversation_state = gr.State([])
        metrics_state = gr.State([])
        prompt_metrics_state = gr.State([])
        trace_state = gr.State({})

        gr.Markdown(
            "# Chatlab\nChat with an open model and see exactly how likely every generated token was.",
            elem_id="hero",
        )

        with gr.Accordion("Model setup", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    model_id = gr.Textbox(
                        value=os.environ.get("OLMO_MODEL_ID", DEFAULT_MODEL),
                        label="Hugging Face model ID",
                        placeholder="organization/model-name",
                        info="The default OLMo 3 7B model is about 15 GB in full precision.",
                    )
                    hf_token = gr.Textbox(
                        label="Hugging Face token (optional)",
                        type="password",
                        placeholder="Only needed for gated or private models",
                    )
                    with gr.Row():
                        download_load_button = gr.Button(
                            "Download and load", variant="primary"
                        )
                        download_button = gr.Button("Download only")
                        cached_button = gr.Button("Load cached")
                        unload_button = gr.Button("Unload")
                with gr.Column(scale=2):
                    model_status = gr.Markdown(
                        status_card(
                            "No model loaded",
                            "Enter a model ID, then download and load it. Files are kept in your normal Hugging Face cache.",
                        ),
                        elem_id="model-status",
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(
                            type="messages",
                            label="Conversation",
                            height=520,
                            placeholder="Load a model, then start a conversation.",
                        )
                        prompt = gr.Textbox(
                            label="Message",
                            placeholder="Ask OLMo something…",
                            lines=3,
                        )
                        with gr.Row():
                            send_button = gr.Button("Send", variant="primary")
                            clear_button = gr.Button("Clear conversation")
                        generation_status = gr.Markdown("Ready.")
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
                            gr.Markdown(
                                "Exports include every token metric and all recorded "
                                "alternatives for the latest completed response.",
                                elem_classes=["footer-note"],
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
                        score_button = gr.Button("Score text", variant="primary")
                        score_status = gr.Markdown("Nothing scored yet.")

            with gr.Column(scale=2):
                gr.Markdown("## Under the hood")
                color_scale = gr.Dropdown(
                    choices=list(COLOR_SCALES),
                    value=DEFAULT_COLOR_SCALE,
                    label="Color tokens by",
                )
                scale_caption = gr.Markdown(
                    COLOR_SCALES[DEFAULT_COLOR_SCALE].caption,
                    elem_classes=["scale-caption"],
                )
                token_strip = gr.HighlightedText(
                    label="Response tokens — click one",
                    color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                    show_legend=True,
                    combine_adjacent=False,
                    elem_id="token-strip",
                )
                token_detail = gr.Markdown(SELECT_HINT)
                alternatives = gr.Dataframe(
                    headers=["Token ID", "Token", "Raw probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Most likely alternatives",
                )
                summary_panel = gr.HTML(charts.summary_tiles({}))
                surprise_panel = gr.HTML(charts.EMPTY_CHART)
                with gr.Accordion("Prompt and context tokens", open=False):
                    prompt_note = gr.Markdown("", elem_classes=["scale-caption"])
                    prompt_strip = gr.HighlightedText(
                        label="Prompt tokens — click one",
                        color_map=COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map,
                        show_legend=True,
                        combine_adjacent=False,
                        elem_id="prompt-strip",
                    )

        with gr.Accordion("Sampling and analysis controls", open=False):
            with gr.Row():
                temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
            with gr.Row():
                max_new_tokens = gr.Slider(
                    1, 8192, value=1024, step=1, label="Maximum new tokens"
                )
                seed = gr.Number(value=42, precision=0, label="Random seed")
                analyze_prompt = gr.Checkbox(
                    value=True,
                    label="Measure prompt tokens",
                    info="Scores every prompt token during the same pass that warms the cache.",
                )

        gr.Markdown(
            "Rank and raw probability come from the unmodified model distribution. "
            "Sampling probability includes temperature, top-k, and top-p. Quantized models may produce slightly different ranks.",
            elem_classes=["footer-note"],
        )

        download_button.click(download_model, [model_id, hf_token], model_status)
        download_load_button.click(
            download_and_load_model, [model_id, hf_token], model_status
        )
        cached_button.click(load_cached_model, model_id, model_status)
        unload_button.click(unload_model, outputs=model_status)

        chat_inputs = [
            prompt,
            conversation_state,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            analyze_prompt,
            color_scale,
        ]
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            prompt_strip,
            prompt_metrics_state,
            prompt_note,
            summary_panel,
            surprise_panel,
            token_detail,
            alternatives,
            trace_state,
        ]
        send_button.click(chat, chat_inputs, chat_outputs)
        prompt.submit(chat, chat_inputs, chat_outputs)
        clear_button.click(
            clear_chat,
            inputs=color_scale,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                prompt_strip,
                prompt_metrics_state,
                prompt_note,
                generation_status,
                token_detail,
                alternatives,
                summary_panel,
                surprise_panel,
                trace_state,
            ],
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
            ],
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

    return demo


if __name__ == "__main__":
    conductor_port = os.environ.get("CONDUCTOR_PORT")
    build_app().queue(default_concurrency_limit=1).launch(
        inbrowser=conductor_port is None,
        server_port=int(conductor_port) if conductor_port else None,
    )
