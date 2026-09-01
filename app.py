"""Gradio interface for chatting with and inspecting OLMo tokens."""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

import gradio as gr

from model_runtime import ModelManager
from token_metrics import CATEGORY_COLORS
from trace_export import build_trace, write_trace_export

DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"
MANAGER = ModelManager()


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


def highlighted_tokens(metrics: list[dict]) -> list[tuple[str, str]]:
    return [(metric["display_text"], metric["category"]) for metric in metrics]


def inspect_token(metrics: list[dict], event: gr.SelectData):
    if not metrics:
        return "Select a generated token to inspect it.", []

    index = event.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    try:
        metric = metrics[int(index)]
    except (IndexError, TypeError, ValueError):
        return "That token is no longer available. Generate another response.", []

    token_repr = repr(metric["text"])
    summary = (
        f"### Token {metric['position']}: `{html.escape(token_repr)}`\n\n"
        f"- **Raw rank:** {metric['raw_rank']:,}\n"
        f"- **Raw model probability:** {metric['raw_probability']:.5%}\n"
        f"- **Actual sampling probability:** {metric['sampling_probability']:.5%}\n"
        f"- **Surprise:** {metric['surprise_bits']:.2f} bits\n"
        f"- **Probability mass above it:** {metric['probability_mass_above']:.2%}\n"
        f"- **Token ID:** {metric['token_id']:,}"
    )
    rows = [
        [candidate["token_id"], repr(candidate["text"]), candidate["probability"]]
        for candidate in metric["top_candidates"]
    ]
    return summary, rows


def chat(
    prompt: str,
    conversation: list[dict] | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
):
    conversation = list(conversation or [])
    prompt = prompt.strip()
    if not prompt:
        yield "", conversation, conversation, [], [], {}, "Enter a message first."
        return
    if not MANAGER.loaded:
        yield (
            "",
            conversation,
            conversation,
            [],
            [],
            {},
            "Download and load a model first.",
        )
        return

    request_messages = conversation + [{"role": "user", "content": prompt}]
    display = request_messages + [{"role": "assistant", "content": ""}]
    yield "", display, conversation, [], [], {}, "Generating…"

    try:
        last_update = None
        for update in MANAGER.generate(
            request_messages,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed),
        ):
            last_update = update
            display[-1] = {"role": "assistant", "content": update.text}
            yield (
                "",
                list(display),
                list(display),
                highlighted_tokens(update.metrics),
                update.metrics,
                gr.skip(),
                f"Generated {len(update.metrics)} token{'s' if len(update.metrics) != 1 else ''}.",
            )
        if last_update is not None:
            trace = build_trace(
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
                metrics=last_update.metrics,
            )
            yield (
                "",
                list(display),
                list(display),
                highlighted_tokens(last_update.metrics),
                last_update.metrics,
                trace,
                (
                    f"Generated {len(last_update.metrics)} token"
                    f"{'s' if len(last_update.metrics) != 1 else ''}. Exports are ready."
                ),
            )
    except Exception as error:
        display[-1] = {"role": "assistant", "content": f"Generation failed: {error}"}
        yield "", display, conversation, [], [], {}, f"Generation failed: {error}"


def clear_chat():
    return (
        [],
        [],
        [],
        [],
        {},
        "Conversation cleared.",
        "Select a generated token to inspect it.",
        [],
    )


CSS = """
.gradio-container { max-width: 1500px !important; }
#hero { padding: 0.5rem 0 0.2rem; }
#hero h1 { font-size: 2.1rem; margin-bottom: 0.25rem; }
#model-status { min-height: 128px; }
#token-strip { min-height: 150px; }
#token-strip span { cursor: pointer; border-radius: 5px; }
.footer-note { color: var(--body-text-color-subdued); font-size: 0.9rem; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="OLMo Token Explorer", css=CSS, theme=gr.themes.Soft()
    ) as demo:
        conversation_state = gr.State([])
        metrics_state = gr.State([])
        trace_state = gr.State({})

        gr.Markdown(
            "# OLMo Token Explorer\nChat with an open model and see exactly how likely every generated token was.",
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
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Conversation",
                    height=560,
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

            with gr.Column(scale=2):
                gr.Markdown("## Under the hood")
                token_strip = gr.HighlightedText(
                    label="Latest response — click a token",
                    color_map=CATEGORY_COLORS,
                    show_legend=True,
                    combine_adjacent=False,
                    elem_id="token-strip",
                )
                token_detail = gr.Markdown("Select a generated token to inspect it.")
                alternatives = gr.Dataframe(
                    headers=["Token ID", "Token", "Raw probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Most likely alternatives",
                )
                gr.Markdown("### Export full metric trace")
                with gr.Row():
                    gr.DownloadButton(
                        "Download JSON",
                        value=lambda trace: write_trace_export(trace, "json"),
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
                    "Exports include every token metric and all recorded alternatives "
                    "for the latest completed response.",
                    elem_classes=["footer-note"],
                )

        with gr.Accordion("Sampling controls", open=False):
            with gr.Row():
                temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
            with gr.Row():
                max_new_tokens = gr.Slider(
                    1, 8192, value=1024, step=1, label="Maximum new tokens"
                )
                seed = gr.Number(value=42, precision=0, label="Random seed")

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
        ]
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            trace_state,
            generation_status,
        ]
        send_button.click(chat, chat_inputs, chat_outputs)
        prompt.submit(chat, chat_inputs, chat_outputs)
        clear_button.click(
            clear_chat,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                trace_state,
                generation_status,
                token_detail,
                alternatives,
            ],
        )
        token_strip.select(
            inspect_token,
            inputs=metrics_state,
            outputs=[token_detail, alternatives],
        )

    return demo


if __name__ == "__main__":
    conductor_port = os.environ.get("CONDUCTOR_PORT")
    build_app().queue(default_concurrency_limit=1).launch(
        inbrowser=conductor_port is None,
        server_port=int(conductor_port) if conductor_port else None,
    )
