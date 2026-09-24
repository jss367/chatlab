"""Remove one prompt word and compare two image runs with fixed settings."""

from __future__ import annotations

import html
import re
from dataclasses import replace

import gradio as gr

from chatlab import charts
from chatlab import image_runtime
from chatlab.token_metrics import PROMPT_ATTENTION_SCALE
from chatlab.ui import images_page


# Keep source spans rather than token IDs: one word can encode as several tokens,
# and repeated words must remain independently selectable. Separators stay visible.
WORD = r"\w+(?:['’\-]\w+)*"
PARTS = re.compile(f"{WORD}|[^\\w]+", re.UNICODE)
EMPTY = '<div class="viz-empty">Draw an original image to test a prompt word.</div>'
PENDING = '<div class="viz-empty">The image without the selected word will appear here.</div>'
OUTPUT_NAMES = (
    "status", "draw", "stop", "pair", "step", "left", "right",
    "left_strip", "right_strip", "left_token", "right_token",
)


def prompt_parts(prompt):
    return list(PARTS.finditer(prompt))


def word_strip(run):
    if run is None or run.image is None or run.stopped:
        return []
    return [
        (part.group(), "Word" if re.fullmatch(WORD, part.group()) else None)
        for part in prompt_parts(run.request.prompt)
    ]


def removal_request(run, index):
    """Change only the selected word occurrence and its adjoining whitespace."""
    if run is None or run.image is None or run.stopped:
        raise ValueError("Draw a finished original image before testing a word.")
    parts = prompt_parts(run.request.prompt)
    if not isinstance(index, int) or not 0 <= index < len(parts):
        raise ValueError("Click a word in the original prompt first.")
    part = parts[index]
    if not re.fullmatch(WORD, part.group()):
        raise ValueError("Select a word, rather than punctuation or a space.")
    before = run.request.prompt[:part.start()]
    after = run.request.prompt[part.end():]
    if before and before[-1].isspace():
        after = after.lstrip()
    return replace(run.request, prompt=(before + after).strip())


def select_word(pair, event: gr.SelectData):
    index = images_page.remember_token(event)
    try:
        request = removal_request(pair[0] if pair else None, index)
    except ValueError as error:
        return None, html.escape(str(error)), gr.update(interactive=False)
    word = prompt_parts(pair[0].request.prompt)[index].group()
    preview = (
        f'<p>Remove <strong>{html.escape(word)}</strong> → '
        f'<code>{html.escape(request.prompt) or "(empty prompt)"}</code><br>'
        f'Seed {request.seed}; all other drawing settings stay fixed.</p>'
    )
    return index, preview, gr.update(interactive=True)


def _panel(run, title, step, token, *, map_ceiling, chart_ceiling, last_step):
    if run is None:
        return PENDING
    picture = (
        f'<img class="trajectory-frame" src="{image_runtime.data_uri(run.image)}" '
        f'alt="{title} finished image" />'
        if run.image is not None else '<p>No finished image: this run was stopped.</p>'
    )
    overlay = ""
    if run.attention is not None:
        overlay = (
            '<div class="viz-empty">Select an attention token above to see its map.</div>'
            if token is None else images_page.attention_overlay(run, token, step, ceiling=map_ceiling)
        )
    result = (
        f'<h3>{title}</h3><p><code>{html.escape(run.request.prompt) or "(empty prompt)"}</code></p>'
        f'<p>Seed {run.request.seed} · {run.steps_done} steps</p>{picture}'
        + images_page.trajectory_frame(run, step)
        + charts.denoising_chart(run.readings, ceiling=chart_ceiling, last_step=last_step)
        + f'<p>{html.escape(images_page.attention_note(run, step))}</p>'
        + overlay
    )
    # These readouts also appear in the main inspector. Avoid duplicate DOM IDs.
    return re.sub(r' id="[^"]*"', '', result)


def read_pair(pair, step, left_token, right_token):
    """Both panels use the same denoising step and visual scales."""
    if not pair or pair[0] is None:
        return EMPTY, PENDING, [], []
    left, right = pair
    runs = [run for run in pair if run is not None]
    common_steps = min((run.steps_done for run in runs), default=1)
    chosen = min(max(1, int(step)), max(1, common_steps))
    maps = [
        image_runtime.token_map(run, token, chosen)
        for run, token in zip(pair, (left_token, right_token))
        if run is not None and token is not None and 0 <= token < len(run.tokens)
    ]
    map_ceiling = max((float(weights.max()) for weights in maps if weights is not None), default=1.0)
    chart_ceiling = max((
        value for run in runs for reading in run.readings
        for value in (reading.guidance_share, reading.latent_change) if value is not None
    ), default=0.0)
    last_step = max((run.steps_done for run in runs), default=1)
    panels = [
        _panel(run, title, chosen, token, map_ceiling=map_ceiling,
               chart_ceiling=chart_ceiling, last_step=last_step)
        for run, title, token in zip(pair, ("Original", "Word removed"), (left_token, right_token))
    ]
    return *panels, images_page.prompt_strip_value(left, chosen), images_page.prompt_strip_value(right, chosen)


def _slider(pair):
    steps = min((run.steps_done for run in pair if run is not None), default=1)
    return gr.update(minimum=1, maximum=max(1, steps), value=1, interactive=steps > 1)


def start_original(run):
    """A new original clears the previous experiment, selection, and maps."""
    pair = (run, None)
    left, right, left_strip, right_strip = read_pair(pair, 1, None, None)
    return (
        word_strip(run), None, "", "Click a word, then remove it and redraw.",
        gr.update(visible=True, interactive=False), gr.update(visible=False),
        pair, _slider(pair), left, right, left_strip, right_strip, None, None,
    )


def begin_original():
    """Disarm the old experiment before a main-page draw can be queued."""
    values = list(start_original(None))
    values[3] = "Drawing a new original; word comparisons will be available when it finishes."
    return tuple(values)


def compare_word(pair, index):
    """Stream a rerun without replacing the original or reading current controls."""
    original = pair[0] if pair else None
    try:
        request = removal_request(original, index)
        if not original.load_id:
            raise ValueError("This image has no model-load record. Draw a new original first.")
    except ValueError as error:
        yield (str(error), gr.update(visible=True), gr.update(visible=False),
               *[gr.skip()] * (len(OUTPUT_NAMES) - 3))
        return

    frames = images_page.draw_request(request, expected_load_id=original.load_id)
    try:
        for frame in frames:
            values = dict(zip(images_page.IMAGE_OUTPUT_NAMES, frame))
            rerun = values["run"]
            if isinstance(rerun, image_runtime.ImageRun):
                result_pair = (original, rerun)
                left, right, left_strip, right_strip = read_pair(result_pair, 1, None, None)
                yield (
                    values["status"], values["draw"], values["stop"], result_pair,
                    _slider(result_pair), left, right, left_strip, right_strip, None, None,
                )
            else:
                yield (
                    values["status"], values["draw"], values["stop"],
                    *[gr.skip()] * (len(OUTPUT_NAMES) - 3),
                )
    finally:
        # Closing the wrapper must also stop the worker owned by draw_request.
        frames.close()


def build_word_comparison():
    """Build the experiment inside the Images workspace and wire its events."""
    with gr.Accordion("Test a prompt word", open=True, elem_id="image-word-test"):
        gr.Markdown(
            "Click a word from the finished image’s prompt, then remove it and redraw "
            "with the same seed and settings. Compare the images and scrub both trajectories together."
        )
        words = gr.HighlightedText(
            label="Original prompt — click a word to remove",
            combine_adjacent=False, show_legend=False, show_inline_category=False,
            color_map={"Word": "#dbeafe"},
            elem_id="image-word-strip",
        )
        selected = gr.State(None)
        pair = gr.State(None)
        left_token, right_token = gr.State(None), gr.State(None)
        preview = gr.HTML("")
        with gr.Row():
            draw = gr.Button("Remove word and redraw", interactive=False, variant="primary")
            stop = gr.Button("Stop comparison", visible=False, variant="stop")
        status = gr.Markdown("Draw an original image first.")
        step = gr.Slider(1, 1, value=1, step=1, label="Comparison denoising step", interactive=False)
        gr.Markdown(
            "Select a token on each side to compare attention. Token positions can shift after removal; "
            "the removed word has no map in the new prompt. Map brightness and chart axes use shared scales. "
            "Maps average across recorded heads and layers."
        )
        with gr.Row():
            with gr.Column(min_width=240):
                left_strip = gr.HighlightedText(
                    label="Original attention — click a token", combine_adjacent=False,
                    show_legend=False, show_inline_category=False,
                    color_map=PROMPT_ATTENTION_SCALE.color_map,
                    elem_id="image-word-original-tokens",
                )
                left = gr.HTML(EMPTY, elem_id="image-word-original")
            with gr.Column(min_width=240):
                right_strip = gr.HighlightedText(
                    label="Word removed attention — click a token", combine_adjacent=False,
                    show_legend=False, show_inline_category=False,
                    color_map=PROMPT_ATTENTION_SCALE.color_map,
                    elem_id="image-word-removed-tokens",
                )
                right = gr.HTML(PENDING, elem_id="image-word-removed")
        outputs = [status, draw, stop, pair, step, left, right,
                   left_strip, right_strip, left_token, right_token]
        words.select(select_word, pair, [selected, preview, draw])
        # Separate queues let the manager refuse an overlapping stale click
        # immediately, rather than running it after a new original finishes.
        draw.click(compare_word, [pair, selected], outputs)
        stop.click(images_page.stop_drawing, None, status, queue=False)
        read_inputs = [pair, step, left_token, right_token]
        read_outputs = [left, right, left_strip, right_strip]
        step.release(read_pair, read_inputs, read_outputs)
        for strip, token in ((left_strip, left_token), (right_strip, right_token)):
            strip.select(images_page.remember_token, None, token).then(read_pair, read_inputs, read_outputs)
    return [words, selected, preview, *outputs]
