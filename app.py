"""Gradio interface for chatting with and inspecting OLMo tokens."""

from __future__ import annotations

import contextlib
import html
import os
import random
import time
from pathlib import Path

import gradio as gr
from gradio.utils import get_upload_folder

from conversation import (
    copy_turns,
    display_messages,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    split_reasoning,
    to_json,
    user_index_at_or_before,
)
from model_runtime import ModelManager
from token_metrics import CATEGORY_COLORS


DEFAULT_MODEL = "allenai/Olmo-3-7B-Think"
MANAGER = ModelManager()
SEED_LIMIT = 2**31 - 1
NO_TOKEN_SELECTED = "Select a generated token to inspect it."


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
        return NO_TOKEN_SELECTED, []

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


# ---------------------------------------------------------------- generation


def send_stop_buttons(busy: bool):
    """Swap the Send and Stop buttons for each other."""

    return gr.update(visible=not busy), gr.update(visible=busy)


def finalize_partial(turns: list[dict]) -> bool:
    """Close out a half-written assistant turn, dropping it when it holds nothing.

    Returns whether a partial response was worth keeping. Cancelling or failing
    mid-stream can leave a turn whose reasoning block is still marked pending,
    which would keep the accordion spinning for the rest of the session.
    """

    if not turns or turns[-1]["role"] != "assistant":
        return False
    if not (turns[-1].get("content") or turns[-1].get("reasoning")):
        turns.pop()
        return False
    turns[-1]["reasoning_closed"] = True
    return True


def stop_generation(turns: list[dict] | None):
    """Finish the turn that the cancelled generator left behind.

    Gradio closes ``generate_reply`` at its last yield, so nothing else ever
    finalizes that turn.
    """

    turns = copy_turns(turns)
    kept = finalize_partial(turns)
    messages, _ = display_messages(turns)
    return (
        messages,
        turns,
        *send_stop_buttons(False),
        "Stopped. The partial response was kept."
        if kept
        else "Stopped before the model produced anything.",
    )


def resolve_seed(seed, randomize: bool) -> int:
    if randomize:
        return random.randrange(SEED_LIMIT)
    try:
        return int(seed)
    except (TypeError, ValueError):
        return 0


def generation_progress(count: int, started: float, seed: int) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    plural = "" if count == 1 else "s"
    return (
        f"{count} token{plural} · {elapsed:.1f}s · {count / elapsed:.1f} tok/s "
        f"· seed {seed}"
    )


def idle_state(
    prompt_text: str, turns: list[dict], status: str, *, clear_tokens: bool = False
):
    """A non-streaming result that leaves the seed untouched.

    The token panel is normally left alone as well: paths such as "Enter a
    message first." must not wipe the diagnostics of the response already on
    screen. ``clear_tokens`` is for the one case where those diagnostics stop
    describing the visible text - an edited assistant reply.
    """

    messages, _ = display_messages(turns)
    return (
        prompt_text,
        messages,
        copy_turns(turns),
        [] if clear_tokens else gr.skip(),
        [] if clear_tokens else gr.skip(),
        status,
        gr.skip(),
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED if clear_tokens else gr.skip(),
        [] if clear_tokens else gr.skip(),
    )


def generate_reply(
    turns: list[dict],
    prompt_text: str,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
):
    """Stream one assistant reply for ``turns``, which must end with a user turn."""

    turns = copy_turns(turns)
    used_seed = resolve_seed(seed, randomize_seed)
    request = model_messages(
        turns, system_prompt=system_prompt, include_reasoning=keep_reasoning
    )

    pending = make_turn("assistant", "", "")
    pending["reasoning_closed"] = True
    turns.append(pending)

    def snapshot(highlight, metrics, status, busy=True, reset_details=False):
        """One frame of the stream.

        ``reset_details`` belongs to the first frame only. That frame empties
        the strip, so a token selected in the previous response is gone and its
        probabilities must go with it. Later frames only append to the strip, so
        a token picked mid-stream stays valid and its details are left alone.
        """

        messages, _ = display_messages(turns)
        return (
            prompt_text,
            messages,
            copy_turns(turns),
            highlight,
            metrics,
            status,
            used_seed,
            *send_stop_buttons(busy),
            NO_TOKEN_SELECTED if reset_details else gr.skip(),
            [] if reset_details else gr.skip(),
        )

    yield snapshot([], [], "Generating…", reset_details=True)

    started = time.monotonic()
    raw_text = ""
    # Reasoning templates end the prompt with the opening <think> marker, so the
    # generated text never carries one. Only the runtime can tell us that.
    prefilled = False
    highlight: list[tuple[str, str]] = []
    metrics: list[dict] = []
    status = "The model produced no tokens."

    stream = MANAGER.generate(
        request,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_new_tokens=int(max_new_tokens),
        seed=used_seed,
    )

    try:
        # closing() releases the model lock the moment the Stop button cancels
        # this event and Gradio closes the outer generator.
        with contextlib.closing(stream):
            for update in stream:
                raw_text = update.text
                prefilled = update.reasoning_prefilled
                reasoning, answer, closed = split_reasoning(
                    raw_text, streaming=True, reasoning_prefilled=prefilled
                )
                pending["reasoning"] = reasoning
                pending["content"] = answer
                pending["reasoning_closed"] = closed
                highlight = highlighted_tokens(update.metrics)
                metrics = list(update.metrics)
                status = generation_progress(len(metrics), started, used_seed)
                yield snapshot(highlight, metrics, status)
    except Exception as error:
        # The diagnostic only goes to the status line. Storing it as the
        # assistant turn would feed the failure back to the model next turn.
        reasoning, answer, _ = split_reasoning(raw_text, reasoning_prefilled=prefilled)
        pending["reasoning"] = reasoning
        pending["content"] = answer
        finalize_partial(turns)
        yield snapshot(highlight, metrics, f"Generation failed: {error}", busy=False)
        return

    reasoning, answer, _ = split_reasoning(raw_text, reasoning_prefilled=prefilled)
    pending["reasoning"] = reasoning
    pending["content"] = answer
    pending["reasoning_closed"] = True
    yield snapshot(highlight, metrics, status, busy=False)


def chat(
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
):
    turns = copy_turns(turns)
    message = (prompt_text or "").strip()
    if not message:
        yield idle_state(prompt_text, turns, "Enter a message first.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    turns.append(make_turn("user", message))
    yield from generate_reply(
        turns,
        "",
        system_prompt,
        keep_reasoning,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
    )


def regenerate_from(
    position: int | None,
    prompt_text: str,
    turns: list[dict] | None,
    system_prompt: str,
    keep_reasoning: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed,
    randomize_seed: bool,
):
    """Throw away everything after the user turn at ``position`` and reply again."""

    turns = copy_turns(turns)
    if position is None:
        yield idle_state(prompt_text, turns, "There is nothing to retry.")
        return
    if not MANAGER.loaded:
        yield idle_state(prompt_text, turns, "Download and load a model first.")
        return

    yield from generate_reply(
        turns[: position + 1],
        prompt_text,
        system_prompt,
        keep_reasoning,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
        randomize_seed,
    )


def retry_last(prompt_text, turns, *settings):
    yield from regenerate_from(last_user_index(turns), prompt_text, turns, *settings)


def retry_message(event: gr.RetryData, prompt_text, turns, *settings):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    yield from regenerate_from(position, prompt_text, turns, *settings)


def edit_message(event: gr.EditData, prompt_text, turns, *settings):
    turns = copy_turns(turns)
    found = locate(turns, event.index)
    if found is None:
        yield idle_state(prompt_text, turns, "That message is no longer available.")
        return

    position, part = found
    new_value = event.value if isinstance(event.value, str) else str(event.value)

    if turns[position]["role"] == "assistant":
        turns[position]["reasoning" if part == "reasoning" else "content"] = new_value
        # The ranks and probabilities on screen describe the text the model
        # generated, not what the user just typed over it.
        yield idle_state(
            prompt_text, turns, "Assistant message edited.", clear_tokens=True
        )
        return

    edited = new_value.strip()
    if not edited:
        # An empty user turn is skipped by model_messages(), which would leave
        # the request with no user message at all.
        yield idle_state(prompt_text, turns, "A user message cannot be empty.")
        return

    turns = turns[: position + 1]
    turns[position]["content"] = edited
    yield from regenerate_from(position, prompt_text, turns, *settings)


def undo_from(position: int | None, turns: list[dict] | None):
    """Drop the exchange starting at the user turn ``position``.

    The message goes back into the input box so it can be reworded and sent again.

    Undo cancels a running generation (see ``cancels`` on its listeners), and a
    cancelled ``generate_reply`` never reaches its final yield, so every path
    here restores the Send button itself exactly as Clear and Load do. That
    includes "There is nothing to undo.": the cancel fires on the click, not on
    what this function decides afterwards.
    """

    turns = copy_turns(turns)
    if position is None:
        messages, _ = display_messages(turns)
        return (
            gr.skip(),
            messages,
            turns,
            gr.skip(),
            gr.skip(),
            "There is nothing to undo.",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
        )

    remaining = turns[:position]
    messages, _ = display_messages(remaining)
    # The selected-token details describe the response being removed, so they
    # go with it, exactly as Clear resets them.
    return (
        turns[position]["content"],
        messages,
        remaining,
        [],
        [],
        "Removed the last exchange.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
    )


def undo_last(turns):
    return undo_from(last_user_index(turns), turns)


def undo_message(event: gr.UndoData, turns):
    found = locate(turns, event.index)
    position = (
        user_index_at_or_before(turns, found[0]) if found else last_user_index(turns)
    )
    return undo_from(position, turns)


def clear_chat():
    """Empty everything the conversation owns.

    Clear cancels a running generation (see ``cancels`` on its listener), and a
    cancelled ``generate_reply`` never reaches its final yield, so this has to
    restore the Send button itself exactly as Stop does.
    """

    return (
        [],
        [],
        [],
        [],
        "Conversation cleared.",
        *send_stop_buttons(False),
        NO_TOKEN_SELECTED,
        [],
    )


# --------------------------------------------------------------- save / load


def save_conversation(turns, system_prompt):
    if not turns:
        return gr.update(value=None, visible=False), "There is nothing to save yet."

    # Gradio only serves files it created or was told to allow, so the saved
    # conversation has to live inside its upload folder.
    directory = Path(get_upload_folder()) / "chatlab-conversations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"conversation-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(to_json(turns, system_prompt=system_prompt), encoding="utf-8")
    return (
        gr.update(value=str(path), visible=True),
        f"Saved {len(turns)} message{'s' if len(turns) != 1 else ''}.",
    )


def load_conversation(file_path):
    """Replace the conversation with a saved one.

    A failed load leaves every output alone, so a bad file cannot wipe the
    conversation already on screen. Loading cancels any generation still
    running, so the buttons are restored here for the same reason Clear
    restores them: a cancelled generator never reaches its final yield.
    """

    skipped = (gr.skip(),) * 5
    if not file_path:
        return (
            *skipped,
            "No file chosen.",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
        )
    try:
        turns, system_prompt = from_json(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return (
            *skipped,
            f"Could not load that file: {error}",
            gr.skip(),
            gr.skip(),
            *send_stop_buttons(False),
        )

    messages, _ = display_messages(turns)
    # The selected token described a response from the conversation being
    # replaced, so it goes with it, exactly as Clear and Undo reset it.
    return (
        messages,
        turns,
        system_prompt,
        [],
        [],
        f"Loaded {len(turns)} message{'s' if len(turns) != 1 else ''}.",
        NO_TOKEN_SELECTED,
        [],
        *send_stop_buttons(False),
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

        with gr.Accordion("System prompt and reasoning", open=False):
            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="You are a careful assistant that answers concisely.",
                lines=3,
                info="Sent as a system message ahead of the conversation. Leave empty to use the model's default behavior.",
            )
            keep_reasoning = gr.Checkbox(
                value=False,
                label="Send previous reasoning back to the model",
                info="Off by default. Think models write a fresh reasoning block each turn, so replaying old ones burns context and usually hurts the next answer.",
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Conversation",
                    height=560,
                    editable="all",
                    placeholder="Load a model, then start a conversation.",
                )
                prompt = gr.Textbox(
                    label="Message",
                    placeholder="Ask OLMo something…",
                    lines=3,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop", visible=False)
                    retry_button = gr.Button("🔁 Retry")
                    undo_button = gr.Button("↩️ Undo last")
                    clear_button = gr.Button("🗑️ Clear")
                with gr.Row():
                    save_button = gr.Button("💾 Save conversation")
                    load_upload = gr.UploadButton(
                        "📂 Load conversation",
                        file_types=[".json"],
                        type="filepath",
                    )
                saved_file = gr.File(
                    label="Saved conversation", visible=False, interactive=False
                )
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
                token_detail = gr.Markdown(NO_TOKEN_SELECTED)
                alternatives = gr.Dataframe(
                    headers=["Token ID", "Token", "Raw probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Most likely alternatives",
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
                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="Random seed",
                    info="Updated after each response so you can reproduce it.",
                )
                randomize_seed = gr.Checkbox(
                    value=True,
                    label="🎲 New seed each response",
                    info="Turn off to lock the seed and reproduce a response exactly.",
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

        settings_inputs = [
            system_prompt,
            keep_reasoning,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
            randomize_seed,
        ]
        chat_inputs = [prompt, conversation_state, *settings_inputs]
        chat_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            seed,
            send_button,
            stop_button,
            token_detail,
            alternatives,
        ]
        undo_outputs = [
            prompt,
            chatbot,
            conversation_state,
            token_strip,
            metrics_state,
            generation_status,
            token_detail,
            alternatives,
            send_button,
            stop_button,
        ]

        running = [
            send_button.click(chat, chat_inputs, chat_outputs),
            prompt.submit(chat, chat_inputs, chat_outputs),
            retry_button.click(retry_last, chat_inputs, chat_outputs),
            chatbot.retry(retry_message, chat_inputs, chat_outputs),
            chatbot.edit(edit_message, chat_inputs, chat_outputs),
        ]

        stop_button.click(
            stop_generation,
            inputs=conversation_state,
            outputs=[
                chatbot,
                conversation_state,
                send_button,
                stop_button,
                generation_status,
            ],
            cancels=running,
        )

        # Undo, Clear and Load all replace or truncate the conversation, so
        # each has to stop the generator first: a surviving generate_reply
        # would write its own snapshot of the in-progress turns back into the
        # chatbot and the state, resurrecting what was just removed. Send,
        # Retry and Edit are exempt because they *are* the generation - they
        # re-enter generate_reply, and they are what everything else cancels.
        undo_button.click(undo_last, conversation_state, undo_outputs, cancels=running)
        chatbot.undo(undo_message, conversation_state, undo_outputs, cancels=running)
        clear_button.click(
            clear_chat,
            outputs=[
                chatbot,
                conversation_state,
                token_strip,
                metrics_state,
                generation_status,
                send_button,
                stop_button,
                token_detail,
                alternatives,
            ],
            cancels=running,
        )

        save_button.click(
            save_conversation,
            [conversation_state, system_prompt],
            [saved_file, generation_status],
        )
        load_upload.upload(
            load_conversation,
            load_upload,
            [
                chatbot,
                conversation_state,
                system_prompt,
                token_strip,
                metrics_state,
                generation_status,
                token_detail,
                alternatives,
                send_button,
                stop_button,
            ],
            cancels=running,
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
