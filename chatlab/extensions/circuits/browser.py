"""The feature browser: page through a transcoder set's features and steer Chat by one.

Listing needs no model and no weights in memory. Each feature's record (its
top-activating examples and top output tokens) is read from the Hub by byte
range, as the graph's feature card reads it. Steering reads the one decoder
row it needs from that layer's file, and hands Chat a vector it adds to that
layer's residual output at every position: the feature switched on
everywhere, at a multiple of its highest recorded activation.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import gradio as gr

from chatlab import steering
from chatlab.ui.token_menu import MENU_BRIDGE_CLASS

from . import render, transcoders

PAGE_SIZE = 20
FETCHERS = 8
DEFAULT_STRENGTH = 3.0
DEFAULT_SET = "gemma-3-1b-it"
NOTHING_SHOWN = ('<div class="cg-root viz-root cg-empty">Choose a layer and press <b>Show</b>. '
                 'No model has to be loaded to browse.</div>')


def spec_named(key):
    spec = next((s for s in transcoders.CATALOGUE if s.key == key), None)
    if spec is None:
        raise ValueError("Choose a transcoder set.")
    return spec


def page_start(spec, start):
    """The first feature of the page holding ``start``, inside the layer."""
    start = int(start or 0)
    return max(0, min(start, spec.width - PAGE_SIZE))


def fetch_page(records, layer, start, count=PAGE_SIZE):
    """``(feature, record, error)`` for ``count`` features from ``start``, fetched side by side."""
    def one(feature):
        try:
            return feature, records.get(layer, feature), None
        except OSError as exc:
            return feature, None, str(exc)

    with ThreadPoolExecutor(FETCHERS) as pool:
        return list(pool.map(one, range(start, start + count)))


def steering_model_id(spec, loaded_id):
    """The model the vector names: the loaded one when the set reads it, else the set's own."""
    if loaded_id and loaded_id.lower() in (m.lower() for m in spec.model_ids):
        return loaded_id
    return spec.model_ids[0]


def feature_vector(spec, layer, feature, record, strength, loaded_id):
    """The feature's decoder row at its highest recorded activation, as a steering vector.

    Strength 1 adds what the feature writes at its strongest. The row is
    used as published, not normalized, so the record's activations apply to it.
    """
    row = transcoders.decoder_row(spec, layer, feature)
    peak = (record or {}).get("act_max") or 1.0
    return steering.vector_from(steering_model_id(spec, loaded_id), layer, [peak * value for value in row],
                                strength=strength)


def build_browser(context, bench):
    loaded = transcoders.spec_for(context.models.loaded_model_id())
    first = loaded or spec_named(DEFAULT_SET)
    gr.Markdown("Every feature of a transcoder set, a page at a time, with the tokens it fires hardest on in "
                "the transcoder's training data and the tokens it writes. **Steer by this feature** adds the "
                "feature at every position of a Chat conversation.")
    with gr.Row():
        set_choice = gr.Dropdown([(f"{s.model_ids[0]} · {s.title}", s.key) for s in transcoders.CATALOGUE],
                                 value=first.key, label="Transcoders", scale=3)
        layer = gr.Slider(0, first.layers - 1, value=first.layers // 2, step=1, label="Layer", scale=3)
        start = gr.Number(value=0, precision=0, minimum=0, label="First feature", scale=1)
    with gr.Row():
        previous = gr.Button("Previous", size="sm", scale=0)
        show = gr.Button("Show", variant="primary", size="sm", scale=0)
        following = gr.Button("Next", size="sm", scale=0)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            listing = gr.HTML(NOTHING_SHOWN, elem_id="circuits-feature-list")
        with gr.Column(scale=2, min_width=340):
            strength = gr.Number(
                value=DEFAULT_STRENGTH, minimum=-100, maximum=100, label="Strength",
                info="Multiples of the feature's highest recorded activation. Negative steers away from it.")
            steer = gr.Button("Steer by this feature", variant="primary")
            detail = gr.HTML(render.feature_detail())
    chosen = gr.State(None)
    shown = gr.State(None)
    pick = gr.Textbox(elem_id="circuits-feature-pick", elem_classes=[MENU_BRIDGE_CLASS])

    def choose_set(key):
        spec = spec_named(key)
        return gr.update(maximum=spec.layers - 1, value=min(spec.layers // 2, spec.layers - 1)), 0

    set_choice.input(choose_set, set_choice, [layer, start], queue=False)

    def list_page(key, layer_value, start_value, selected, step=0):
        try:
            spec = spec_named(key)
            first_feature = page_start(spec, int(start_value or 0) + step * PAGE_SIZE)
            records = bench.records(spec)
            rows = fetch_page(records, int(layer_value), first_feature)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        page = {"set": spec.key, "layer": int(layer_value)}
        same_page = (selected and {k: selected[k] for k in page} == page
                     and first_feature <= selected["feature"] < first_feature + len(rows))
        mark = selected["feature"] if same_page else None
        return (render.feature_list(int(layer_value), first_feature, spec.width, rows, mark), first_feature, page,
                gr.skip() if same_page else None, gr.skip() if same_page else render.feature_detail())

    for button, step in ((show, 0), (previous, -1), (following, 1)):
        button.click(partial(list_page, step=step),
                     [set_choice, layer, start, chosen], [listing, start, shown, chosen, detail],
                     concurrency_id="circuits-browse")

    def picked(page, raw):
        try:
            feature = int(json.loads(raw)["feature"])
        except (TypeError, ValueError, KeyError):
            return gr.skip(), gr.skip()
        if not page:
            return gr.skip(), gr.skip()
        try:
            record, error = bench.records(spec_named(page["set"])).get(page["layer"], feature), None
        except OSError as exc:
            record, error = None, str(exc)
        return render.feature_detail(page["layer"], feature, record, error), {**page, "feature": feature}

    pick.input(picked, [shown, pick], [detail, chosen],
               concurrency_id="circuits-browse", trigger_mode="always_last", show_progress="hidden")

    def vector(selected, amount):
        if not selected:
            raise ValueError("Click a feature in the list first.")
        spec = spec_named(selected["set"])
        try:
            record = bench.records(spec).get(selected["layer"], selected["feature"])
        except OSError:
            record = None
        try:
            return feature_vector(spec, selected["layer"], selected["feature"], record,
                                  float(amount if amount is not None else DEFAULT_STRENGTH),
                                  context.models.loaded_model_id())
        except OSError as exc:
            raise ValueError(f"Could not read the feature's decoder row: {exc}") from exc

    context.navigation.steer_chat(steer, vector, [chosen, strength])
