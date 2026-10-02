"""Train linear probes on the loaded model's residual stream and read passages with them."""
import html
import logging
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np

from chatlab.extension_api import write_private_text
from . import probe as probes

logger = logging.getLogger(__name__)

GENERATE, READ = "Generate a reply", "Read text"
# Columns past this are left out of the every-layer view; the strip still
# shows every token.
HEAT_COLUMNS = 512
EDGES = (0.1, 0.3, 0.7, 0.9)
LAYER_HEADERS = ["Layer", "Held-out accuracy", "Held-out loss", "Training accuracy"]
TOKEN_HEADERS = ["Layer", "Probability"]
NO_PROBE = "Train a probe or open a saved one."

CSS = """
#probes-page {overflow-y:auto; min-height:0; padding:12px;}
#probes-page .probe-heat {overflow-x:auto; max-width:100%;}
#probes-page .probe-heat table {border-collapse:collapse; font-size:11px; line-height:1;}
#probes-page .probe-heat td {width:12px; min-width:12px; height:12px; padding:0;}
#probes-page .probe-heat tr.probe-chosen td {box-shadow:inset 0 2px 0 var(--body-text-color), inset 0 -2px 0 var(--body-text-color);}
#probes-page .probe-heat tr.probe-chosen th {font-weight:bold;}
#probes-page .probe-heat th {font-weight:normal; padding:0 6px 0 0; text-align:right; white-space:nowrap;}
"""


class Runs:
    """One reading per view, and a way to stop its reply from another event."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}

    @staticmethod
    def new_owner():
        return uuid4().hex

    def start(self, owner):
        with self._lock:
            if owner in self._active:
                raise ValueError("This view is already reading.")
            self._active[owner] = [threading.Event(), None]
            return self._active[owner][0]

    def attach(self, owner, session):
        with self._lock:
            active = self._active[owner]
            active[1] = session
            if active[0].is_set():
                session.cancel()

    def finish(self, owner):
        with self._lock:
            self._active.pop(owner, None)

    def cancel(self, owner):
        with self._lock:
            active = self._active.get(owner)
            if active:
                active[0].set()
                if active[1] is not None:
                    active[1].cancel()


def quoted(value):
    """A value from a typed box or an opened file, as Markdown that shows only its characters."""
    text = " ".join(str(value).split())
    return text if re.fullmatch(r"[A-Za-z0-9.,:+ _/-]*", text) else "`" + text.replace("`", "'") + "`"


def parse_examples(text):
    """One example per line, kept as written; blank lines are dropped (as Chat's vector extraction)."""
    return [line for line in (text or "").splitlines() if line.strip()]


def scale_labels(probe):
    positive, negative = probe["positive_label"], probe["negative_label"]
    return (f"{negative} (under 10%)", f"Leans {negative} (10–30%)", "Unsure (30–70%)",
            f"Leans {positive} (70–90%)", f"{positive} (over 90%)")


def color_map(probe, palette):
    # Diverging runs red to blue; the side the probe looks for is painted warm.
    return dict(zip(scale_labels(probe), reversed(palette["diverging"])))


def bucket(probe, probability):
    labels = scale_labels(probe)
    for label, edge in zip(labels, EDGES):
        if probability < edge:
            return label
    return labels[-1]


def probe_summary(probe):
    if probe is None:
        return NO_PROBE
    best = probe["layers"][probe["best_layer"]]
    examples = probe["examples"]
    where = "the last token" if probe["pool"] == "last" else "the mean over every token"
    framing = " of each example as a user message" if probe["chat_template"] else " of each example as plain text"
    return (f"**{quoted(probe['name'])}** for {quoted(probe['model_id'])}: "
            f"{quoted(probe['positive_label'])} ({len(examples['positive'])} examples) against "
            f"{quoted(probe['negative_label'])} ({len(examples['negative'])}), read at {where}{framing}. "
            f"Best layer {probe['best_layer']}: {best['heldout_accuracy']:.0%} accurate on held-out examples "
            f"over {probe['folds']} folds{' with pairs held out together' if probe['paired'] else ''}, "
            f"{best['train_accuracy']:.0%} on the examples it was trained on.{below_chance(probe)}")


def below_chance(probe):
    """A hint for the one reading of the table that usually means the folds, not the probe."""
    if probe["paired"] or max(item["heldout_accuracy"] for item in probe["layers"]) >= 0.5:
        return ""
    return (" Every layer is below chance on held-out examples. If line n of each side is the same "
            "example changed, tick **The examples are pairs** and train again.")


def layer_rows(probe):
    if probe is None:
        return []
    return [[f"{item['layer']}{' (best)' if item['layer'] == probe['best_layer'] else ''}",
             f"{item['heldout_accuracy']:.0%}", round(item["heldout_loss"], 3), f"{item['train_accuracy']:.0%}"]
            for item in probe["layers"]]


def saved_choices(directory):
    choices = []
    for probe in probes.saved(directory):
        when = datetime.fromtimestamp(probe["created"]).strftime("%Y-%m-%d %H:%M")
        choices.append((f"{probe['name']} · {probe['model_id']} · {when}", probe["id"]))
    return choices


def _mix(low, high, share):
    low, high = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (low, high))
    red, green, blue = (low + (high - low) * share).round().astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def heat_color(probability, palette):
    cool, neutral, warm = palette["diverging"][4], palette["diverging"][2], palette["diverging"][0]
    if probability < 0.5:
        return _mix(cool, neutral, probability * 2)
    return _mix(neutral, warm, (probability - 0.5) * 2)


def heatmap(probe, reading, layer, palette):
    """Every block's probability at every shown token, a row per block."""
    if probe is None or reading is None:
        return ""
    first = reading["first"]
    texts = reading["texts"][first:first + HEAT_COLUMNS]
    rows = []
    for index, values in enumerate(reading["probabilities"]):
        cells = "".join(
            f'<td style="background:{heat_color(p, palette)}" '
            f'title="{html.escape(text, quote=True)} · layer {index} · {p:.0%}"></td>'
            for text, p in zip(texts, values[first:first + HEAT_COLUMNS]))
        chosen = ' class="probe-chosen"' if index == layer else ""
        rows.append(f"<tr{chosen}><th>layer {index}</th>{cells}</tr>")
    shown = len(reading["texts"]) - first
    note = (f"<p>Showing the first {HEAT_COLUMNS} of {shown} tokens.</p>" if shown > HEAT_COLUMNS else "")
    return (f'<div class="probe-heat"><p>Probability of {html.escape(probe["positive_label"])} at every layer. '
            "Hover a cell for its token. The outlined row is the layer the strip shows.</p>"
            f"{note}<table>{''.join(rows)}</table></div>")


def strip_value(probe, reading, layer, display):
    if probe is None or reading is None:
        return []
    first = reading["first"]
    values = reading["probabilities"][layer]
    return [(display(text, str(token_id)), bucket(probe, p)) for text, token_id, p in
            zip(reading["texts"][first:], reading["token_ids"][first:], values[first:])]


def build_page(context):
    runs = Runs()
    palette = context.tokens.palette
    display = context.tokens.display_text

    with gr.Column(elem_id="probes-page"):
        owner = gr.State(value=runs.new_owner, delete_callback=runs.cancel)
        probe_state = gr.State(None)
        reading_state = gr.State(None)
        gr.Markdown("# Linear probes\nGive examples of two kinds of text. A logistic regression is fitted to the "
                    "residual stream at every layer and tested on examples it was not trained on. Then read any "
                    "reply or passage with it, token by token.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                models = gr.Button("Open Models", size="sm")
                context.navigation.open_models(models)
                name = gr.Textbox(label="Probe name", value="My probe")
                with gr.Row():
                    positive_label = gr.Textbox(label="Looking for", value="Yes", max_lines=1)
                    negative_label = gr.Textbox(label="Against", value="No", max_lines=1)
                positive_text = gr.Textbox(label="Examples of what it looks for", lines=8,
                                           info=f"One per line, 2 to {probes.MAX_EXAMPLES}.")
                negative_text = gr.Textbox(label="Examples of the other side", lines=8,
                                           info=f"One per line, 2 to {probes.MAX_EXAMPLES}.")
                with gr.Accordion("How examples are read", open=False):
                    chat_template = gr.Checkbox(label="Read each example as a user message", value=False,
                                                info="Puts it in a user turn followed by the generation prompt, "
                                                     "so the last token is where the model would answer from.")
                    paired = gr.Checkbox(label="The examples are pairs", value=False,
                                         info="Line n of each side is one example with the property changed. "
                                              "Each pair is held out together, so its twin is never in training.")
                    pool = gr.Radio([("Last token", "last"), ("Mean over tokens", "mean")], value="last",
                                    label="Pool each example by")
                    l2 = gr.Number(label="L2 strength", value=probes.DEFAULT_L2, minimum=0.0001,
                                   info="Larger values hold the weights smaller. Features are standardized first.")
                train = gr.Button("Train probe", variant="primary")
                summary = gr.Markdown(NO_PROBE)
                layers_table = gr.Dataframe(headers=LAYER_HEADERS, interactive=False, wrap=True)
                with gr.Accordion("Saved probes", open=False):
                    gr.Markdown("Every trained probe is saved. Opening one works under any load, "
                                "but reading with it needs the model it was trained on.")
                    picker = gr.Dropdown(choices=saved_choices(context.data_dir), value=None,
                                         label="Open a saved probe", interactive=True)
                    upload = gr.File(label="Import a probe file", file_types=[".json"], type="filepath")
                    download = gr.File(label="This probe", interactive=False)
            with gr.Column(scale=2, min_width=420):
                mode = gr.Radio([GENERATE, READ], value=GENERATE, label="Read")
                text = gr.Textbox(label="Message to the model", lines=4)
                as_user = gr.Checkbox(label="Read the text as a user message", value=False, visible=False)
                include_prompt = gr.Checkbox(label="Show the prompt's tokens too", value=False)
                with gr.Accordion("Reply settings", open=False) as reply_settings:
                    system = gr.Textbox(label="System prompt", lines=3,
                                        info="Leave empty to send no system message.")
                    temperature = gr.Slider(0, 2, value=0.7, step=0.05, label="Temperature")
                    seed = gr.Number(value=42, precision=0, minimum=0, label="Seed")
                    max_tokens = gr.Number(value=512, precision=0, minimum=1, maximum=32768,
                                           label="Tokens per reply")
                with gr.Row():
                    run = gr.Button("Read", variant="primary")
                    stop = gr.Button("Stop")
                reply = gr.Textbox(label="Reply", lines=4, max_lines=12, interactive=False)
                layer = gr.Slider(0, 1, value=0, step=1, label="Layer", visible=False)
                strip = gr.HighlightedText(label="Click a token to see it at every layer", combine_adjacent=False,
                                           show_legend=True, elem_id="probes-tokens", visible=False)
                detail = gr.Markdown("")
                token_table = gr.Dataframe(headers=TOKEN_HEADERS, interactive=False)
                with gr.Accordion("Every layer", open=False):
                    heat = gr.HTML("")

    def save(probe):
        path = context.data_dir / f"{probe['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(path, probes.dumps(probe))

    def staged(probe):
        """A copy named after the probe, where the interface is allowed to serve it."""
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", probe["name"]).strip("-")[:60] or "probe"
        path = Path(tempfile.mkdtemp(prefix="chatlab-probe-")) / f"{name}.json"
        write_private_text(path, probes.dumps(probe))
        return str(path)

    def shown_probe(probe):
        """Everything that changes when a different probe is the current one.

        The form is filled with what the probe was trained from, so it can be
        changed and trained again, and text is read the way its examples were
        until the reader says otherwise.
        """
        choices = saved_choices(context.data_dir)
        listed = any(value == probe["id"] for _label, value in choices)
        examples = probe["examples"]
        return (probe, probe_summary(probe), layer_rows(probe),
                gr.update(choices=choices, value=probe["id"] if listed else None), staged(probe),
                gr.update(maximum=len(probe["layers"]) - 1, value=probe["best_layer"], visible=True),
                probe["chat_template"], None, gr.update(value=[], visible=False), "", [], "",
                probe["name"], probe["positive_label"], probe["negative_label"],
                "\n".join(examples["positive"]), "\n".join(examples["negative"]),
                probe["chat_template"], probe["paired"], probe["pool"], probe["l2"])

    probe_outputs = [probe_state, summary, layers_table, picker, download, layer, as_user,
                     reading_state, strip, detail, token_table, heat,
                     name, positive_label, negative_label, positive_text, negative_text,
                     chat_template, paired, pool, l2]

    def train_probe(probe_name, looking_for, against, wanted, unwanted, template, pairs, pooling, strength):
        positive, negative = parse_examples(wanted), parse_examples(unwanted)
        looking_for, against = (looking_for or "").strip(), (against or "").strip()
        try:
            if not looking_for or not against or looking_for == against:
                raise ValueError("Give the two sides different labels.")
            for side, examples in ((looking_for, positive), (against, negative)):
                if not 2 <= len(examples) <= probes.MAX_EXAMPLES:
                    raise ValueError(f"Give 2 to {probes.MAX_EXAMPLES} examples of {side}, one per line; "
                                     f"there are {len(examples)}.")
            if pairs and len(positive) != len(negative):
                raise ValueError(f"Paired examples need the same number of lines on each side; there are "
                                 f"{len(positive)} and {len(negative)}.")
            strength = float(strength)
            with context.models.open_session() as session:
                model_id = session.model_id
                wanted_rows = session.read_examples(positive, chat_template=template, pool=pooling)
                unwanted_rows = session.read_examples(negative, chat_template=template, pool=pooling)
            # The model is released before fitting, which needs only the arrays.
            fitted, folds = probes.train(wanted_rows, unwanted_rows, l2=strength, paired=pairs)
            probe = probes.build(name=(probe_name or "").strip() or "Probe", model_id=model_id,
                                 positive_label=looking_for, negative_label=against,
                                 positive_examples=positive, negative_examples=negative, pool=pooling,
                                 chat_template=template, l2=strength, layers=fitted, folds=folds, paired=pairs)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        try:
            save(probe)
        except OSError as exc:
            logger.warning("Could not save probe %s: %s", probe["id"], exc)
            gr.Warning(f"The probe was trained but not saved: {exc}.")
        return shown_probe(probe)

    train.click(train_probe, [name, positive_label, negative_label, positive_text, negative_text,
                              chat_template, paired, pool, l2], probe_outputs, concurrency_id="probes-model")

    def open_saved(probe_id):
        if not probe_id:
            return (gr.skip(),) * len(probe_outputs)
        try:
            probe = probes.read(context.data_dir / f"{probe_id}.json")
        except (OSError, ValueError) as exc:
            raise gr.Error(f"That probe could not be opened: {exc}") from exc
        return shown_probe(probe)

    picker.input(open_saved, picker, probe_outputs, show_progress="hidden")

    def import_probe(path):
        if not path:
            return (gr.skip(),) * len(probe_outputs)
        try:
            probe = probes.read(path)
            save(probe)
        except (OSError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return shown_probe(probe)

    upload.upload(import_probe, upload, probe_outputs, show_progress="hidden")

    def switch_mode(chosen):
        generating = chosen == GENERATE
        return (gr.update(label="Message to the model" if generating else "Text to read"),
                gr.update(visible=not generating), gr.update(visible=generating),
                gr.update(visible=generating), gr.update(visible=generating))

    mode.change(switch_mode, mode, [text, as_user, include_prompt, reply_settings, reply], queue=False)

    def rendered(probe, reading, chosen):
        chosen = int(chosen)
        return (gr.update(value=strip_value(probe, reading, chosen, display), color_map=color_map(probe, palette),
                          visible=True),
                heatmap(probe, reading, chosen, palette))

    def read(probe, chosen_mode, message, system_text, temp, random_seed, token_limit, show_prompt,
             user_turn, view, chosen):
        """Generate a reply or take the text as it is, then read every token with the probe.

        The reply streams into its box; the probe's reading arrives once, at
        the end, because it is one forward pass over the finished sequence.
        Stopping a reply still reads what was generated.
        """
        if probe is None:
            raise gr.Error(NO_PROBE)
        if not isinstance(message, str) or not message.strip():
            raise gr.Error("Type something to read first.")
        try:
            cancel = runs.start(view)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        reading = None
        try:
            with context.models.open_session() as session:
                runs.attach(view, session)
                if session.model_id != probe["model_id"]:
                    raise ValueError(f"This probe was trained on {probe['model_id']}; load that model to "
                                     f"read with it. {session.model_id} is loaded.")
                if chosen_mode == GENERATE:
                    messages = ([{"role": "system", "content": system_text}] if system_text.strip() else [])
                    messages.append({"role": "user", "content": message})
                    limit, seed_value = int(token_limit), int(random_seed)
                    if not 1 <= limit <= 32768 or seed_value < 0:
                        raise ValueError("Seed must be nonnegative and tokens per reply between 1 and 32768.")
                    last = None
                    stream = session.generate(messages, temperature=float(temp), top_p=1.0, top_k=0,
                                              max_new_tokens=limit, seed=seed_value)
                    try:
                        for last in stream:
                            yield (gr.skip(),) * 4 + (last.text,)
                    finally:
                        stream.close()
                    if last is None or not last.metrics:
                        if not cancel.is_set():
                            raise ValueError("The model generated no tokens.")
                        return
                    prompt_ids = [int(value) for value in last.prompt_ids]
                    ids = prompt_ids + [int(metric["token_id"]) for metric in last.metrics]
                    first = 0 if show_prompt else len(prompt_ids)
                    window = session.position_limit
                    if window is not None and len(ids) > window:
                        # A reply that ran into the window ends on a token the
                        # model sampled but never read, so there is no reading of it.
                        ids = ids[:window]
                else:
                    # A passage longer than the window is refused by the reading
                    # itself rather than cut short: a prefix is not what was asked for.
                    ids, first = session.example_ids(message, chat_template=user_turn), 0
                projections = session.project_layers(ids, probes.directions(probe))
                reading = dict(probe_id=probe["id"], model_id=session.model_id, load_id=session.load_id,
                               token_ids=ids, texts=[session.decode([token]) for token in ids], first=first,
                               probabilities=probes.probabilities(probe, projections).tolist())
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            runs.finish(view)
        yield (reading, *rendered(probe, reading, chosen), "", gr.skip())

    run.click(lambda: ("", []), None, [detail, token_table], queue=False)
    run.click(read, [probe_state, mode, text, system, temperature, seed, max_tokens, include_prompt, as_user,
                     owner, layer], [reading_state, strip, heat, detail, reply],
              concurrency_id="probes-model", show_progress="hidden")
    stop.click(runs.cancel, owner, None, queue=False)

    def change_layer(probe, reading, chosen):
        if probe is None or reading is None or reading["probe_id"] != probe["id"]:
            return gr.skip(), gr.skip()
        return rendered(probe, reading, chosen)

    layer.release(change_layer, [probe_state, reading_state, layer], [strip, heat], show_progress="hidden")

    def inspect_token(probe, reading, chosen, event: gr.SelectData):
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        if probe is None or reading is None or reading["probe_id"] != probe["id"]:
            return "Read something first.", []
        position = reading["first"] + index if isinstance(index, int) else -1
        if not reading["first"] <= position < len(reading["token_ids"]):
            return "Select a token in the current reading.", []
        values = [row[position] for row in reading["probabilities"]]
        text_shown = reading["texts"][position].replace("`", "'")
        peak = int(np.argmax(values))
        note = (f"Token {position + 1}, `{text_shown}` (ID {reading['token_ids'][position]}): "
                f"{values[int(chosen)]:.1%} {quoted(probe['positive_label'])} at layer {int(chosen)}, "
                f"highest at layer {peak} ({values[peak]:.1%}).")
        return note, [[index_, f"{value:.1%}"] for index_, value in enumerate(values)]

    strip.select(inspect_token, [probe_state, reading_state, layer], [detail, token_table], queue=False)
