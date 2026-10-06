"""Edit the residual stream along chosen directions and watch what later blocks and the lens make of it."""
import logging
import re
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import gradio as gr

from chatlab.extension_api import ProjectionCancelled, read_steering_vector, write_private_text

from . import experiment, files, render

logger = logging.getLogger(__name__)

CSS = """
#direction-edits-page {overflow-y:auto; min-height:0; padding:12px;}
#direction-edits-page .de-heat {overflow-x:auto; max-width:100%; margin-bottom:12px;}
#direction-edits-page .de-heat table {border-collapse:collapse; font-size:11px; line-height:1;}
#direction-edits-page .de-heat td {width:10px; min-width:10px; height:10px; padding:0;}
#direction-edits-page .de-heat td.de-none {background:transparent;}
#direction-edits-page .de-heat th {font-weight:normal; padding:0 6px 0 0; text-align:right; white-space:nowrap;}
#direction-edits-page .de-heat tr.de-mark td {height:4px;}
#direction-edits-page .de-heat tr.de-mark td.de-on {background:var(--body-text-color);}
#direction-edits-page .de-heat tr.de-edited td {box-shadow:inset 0 1px 0 var(--body-text-color), inset 0 -1px 0 var(--body-text-color);}
#direction-edits-page .de-heat tr.de-edited th, #direction-edits-page .de-table tr.de-edited th {font-weight:bold;}
#direction-edits-page .de-table {overflow-x:auto; max-width:100%;}
#direction-edits-page .de-table table {border-collapse:collapse; font-size:13px;}
#direction-edits-page .de-table th, #direction-edits-page .de-table td {padding:2px 10px; text-align:right; white-space:nowrap;}
#direction-edits-page .de-table thead th {border-bottom:1px solid var(--border-color-primary);}
"""

MODES = [("Erase to reference", experiment.ERASE), ("Clamp", experiment.CLAMP), ("Add", experiment.ADD)]
EXPLAINED = ("# Direction edits\nInject a vector over some of a passage's tokens, edit the residual stream along a "
             "direction at one block or a range of them, and see whether later blocks rebuild what the edit took "
             "away, and what that does to a word's probability through the Jacobian lens.")


class Runs:
    """One run per view, a way to stop it from another event, and whether its result may still be shown.

    Each view has a turn that moves when a run starts or a saved result is
    opened. A run publishes only while the turn it took is still the view's.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._turns = {}
        self._sessions = {}
        self._downloads = {}

    @staticmethod
    def new_owner():
        return uuid4().hex

    def start(self, owner):
        with self._lock:
            self._turns[owner] = self._turns.get(owner, 0) + 1
            active = self._sessions.pop(owner, None)
            if active is not None:
                active[1].cancel()
            return self._turns[owner]

    def attach(self, owner, turn, session):
        with self._lock:
            if self._turns.get(owner) != turn:
                session.cancel()
            else:
                self._sessions[owner] = (turn, session)

    def finish(self, owner, turn):
        with self._lock:
            if self._sessions.get(owner, (None,))[0] == turn:
                self._sessions.pop(owner)

    def cancel(self, owner):
        with self._lock:
            active = self._sessions.get(owner)
            if active is not None:
                active[1].cancel()

    def live(self, owner, turn):
        with self._lock:
            return self._turns.get(owner) == turn

    def stage(self, owner, result):
        """Write the result into the view's own download directory, replacing the last one."""
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", result["inputs"]["directions"]["name"]).strip("-")[:60] or "result"
        with self._lock:
            directory = self._downloads.get(owner)
            if directory is None:
                directory = self._downloads[owner] = tempfile.TemporaryDirectory(prefix="chatlab-direction-edits-")
            root = Path(directory.name)
            path = root / f"direction-edits-{name}.json"
            write_private_text(path, files.dumps(result))
            for old in root.iterdir():
                if old != path:
                    old.unlink()
            return str(path)

    def forget(self, owner):
        self.cancel(owner)
        with self._lock:
            self._turns.pop(owner, None)
            self._sessions.pop(owner, None)
            directory = self._downloads.pop(owner, None)
        if directory is not None:
            directory.cleanup()


def whole(value, what):
    if value is None or float(value) != int(value):
        raise ValueError(f"Give {what} as a whole number.")
    return int(value)


def gather(passage, directions, vector, strength, injection_first, injection_last, mode, value, block_first,
           block_last, token_first, token_last, random_control, seed, word, read_first, read_last, read_block_first,
           read_block_last, differences, conditions):
    """The page's controls as the experiment's inputs, checked."""
    if directions is None:
        raise ValueError("Open a direction file first.")
    pairs = [(name or "", prefix or "") for name, prefix in conditions]
    inputs = {
        "passage": passage or "",
        "conditions": [{"name": name, "prefix": prefix} for name, prefix in pairs if name.strip() or prefix.strip()],
        "injection": None if vector is None else {
            "vector": vector, "strength": float(strength if strength is not None else 0.0),
            "tokens": [whole(injection_first, "the injection's first token"),
                       whole(injection_last, "the injection's last token")]},
        "directions": directions,
        "edit": {"mode": mode, "value": float(value or 0.0),
                 "blocks": [whole(block_first, "the first edited block"), whole(block_last, "the last edited block")],
                 "tokens": [whole(token_first, "the first edited token"), whole(token_last, "the last edited token")],
                 "random_control": bool(random_control), "seed": whole(seed, "the seed")},
        "readout": None if not (word or "").strip() else {
            "word": word,
            "tokens": [whole(read_first, "the first readout token"), whole(read_last, "the last readout token")],
            "blocks": [whole(read_block_first, "the first readout block"),
                       whole(read_block_last, "the last readout block")]},
        "differences": experiment.parse_differences(differences),
    }
    return experiment.normalize_inputs(inputs)


def directions_summary(directions):
    if directions is None:
        return "No directions open."
    layers = [item["layer"] for item in directions["directions"]]
    source = " from a probe file" if directions["source"] != files.DIRECTIONS_FORMAT else ""
    return (f"**{directions['name']}** for `{directions['model_id']}`{source}: {len(layers)} unit directions, "
            f"{len(directions['directions'][0]['vector']):,} wide, at blocks {layers[0]}–{layers[-1]}.")


def vector_summary(vector):
    if vector is None:
        return "Nothing is injected."
    return (f"Injecting at block {vector['layer']}: a vector for `{vector['model_id']}`, "
            f"{len(vector['vector']):,} wide.")


def build_page(context):
    runs = Runs()
    palette = context.tokens.palette
    display = context.tokens.display_text

    with gr.Column(elem_id="direction-edits-page"):
        owner = gr.State(value=runs.new_owner, delete_callback=runs.forget)
        result_state = gr.State(None)
        directions_state = gr.State(None)
        vector_state = gr.State(None)
        gr.Markdown(EXPLAINED)
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                models = gr.Button("Open Models", size="sm")
                context.navigation.open_models(models)
                passage = gr.Textbox(label="Passage", lines=6,
                                     info="Token numbers count from 1 at its first token, in every condition.")
                with gr.Accordion("Conditions", open=True):
                    gr.Markdown("Each prefix is read before the passage, and the two are tokenized apart, so the "
                                "passage's tokens are the same under every condition. End a prefix with the space "
                                "or line break that should come before the passage. Leave a row empty to skip it.")
                    rows = []
                    for number in range(experiment.MAX_CONDITIONS):
                        with gr.Row():
                            name = gr.Textbox(label=f"Condition {number + 1}", value="neutral" if number == 0 else "",
                                              max_lines=1, scale=1, min_width=100)
                            prefix = gr.Textbox(label="Prefix", lines=1, scale=3)
                        rows.append((name, prefix))
                directions_file = gr.File(label="Directions: chatlab-directions-1 or chatlab-probe-1",
                                          file_types=[".json"], type="filepath")
                directions_note = gr.Markdown(directions_summary(None))
                with gr.Accordion("Injection", open=False):
                    vector_file = gr.File(label="Steering vector: chatlab-steering-1", file_types=[".json"],
                                          type="filepath")
                    vector_note = gr.Markdown(vector_summary(None))
                    strength = gr.Number(label="Strength", value=1.0,
                                         info="Added as strength × vector at the vector's block, over these tokens only.")
                    with gr.Row():
                        injection_first = gr.Number(label="First token", value=1, precision=0, minimum=1)
                        injection_last = gr.Number(label="Last token", value=64, precision=0, minimum=1)
                with gr.Accordion("Edit", open=True):
                    mode = gr.Radio(MODES, value=experiment.ERASE, label="Edit")
                    value = gr.Number(label="Clamp to", value=0.0, visible=False)
                    with gr.Row():
                        block_first = gr.Number(label="First block", value=0, precision=0, minimum=0)
                        block_last = gr.Number(label="Last block", value=0, precision=0, minimum=0)
                    with gr.Row():
                        token_first = gr.Number(label="First token", value=1, precision=0, minimum=1)
                        token_last = gr.Number(label="Last token", value=64, precision=0, minimum=1)
                    random_control = gr.Checkbox(
                        label="Random control", value=False,
                        info="Also make an edit of the same size per token along a seeded random direction.")
                    seed = gr.Number(label="Seed", value=0, precision=0, minimum=0)
                with gr.Accordion("Lens readout", open=False):
                    word = gr.Textbox(label="Target word", max_lines=1,
                                      info="Leave empty to skip the lens. Include the leading space a word mid-sentence "
                                           "has. A word of several tokens is read as the mean of their unembeddings.")
                    with gr.Row():
                        read_first = gr.Number(label="First token", value=1, precision=0, minimum=1)
                        read_last = gr.Number(label="Last token", value=64, precision=0, minimum=1)
                    with gr.Row():
                        read_block_first = gr.Number(label="First block", value=0, precision=0, minimum=0)
                        read_block_last = gr.Number(label="Last block", value=0, precision=0, minimum=0)
                    differences = gr.Textbox(label="Differences", lines=3,
                                             placeholder="suppression = neutral - ignore\nfocus = focus - neutral",
                                             info="One per line: a name, then two condition names to subtract.")
                with gr.Row():
                    run = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop")
                status = gr.Markdown("")
            with gr.Column(scale=2, min_width=420):
                headline = gr.Markdown("")
                picker = gr.Dropdown(label="Condition", choices=[], interactive=True, visible=False)
                heat = gr.HTML("")
                with gr.Accordion("Recovery by block", open=True):
                    recovery = gr.HTML("")
                lens_note = gr.Markdown("")
                lens = gr.HTML("")
                diffs = gr.HTML("")
                download = gr.File(label="This result", interactive=False)
                upload = gr.File(label="Open a saved result", file_types=[".json"], type="filepath")

    shown = [result_state, headline, picker, heat, recovery, lens_note, lens, diffs, download]

    def show(result, path):
        names = [condition["name"] for condition in result["conditions"]]
        choices = [(name, index) for index, name in enumerate(names)]
        return (result, render.headline(result), gr.update(choices=choices, value=0, visible=True),
                render.heatmaps(result, 0, palette, display), render.recovery_table(result),
                render.target_note(result), render.lens_table(result), render.differences_table(result), path)

    def open_directions(path):
        if not path:
            return None, directions_summary(None)
        try:
            directions = files.read_directions(path)
        except (OSError, ValueError) as exc:
            raise gr.Error(f"Those directions could not be opened: {exc}") from exc
        return directions, directions_summary(directions)

    directions_file.upload(open_directions, directions_file, [directions_state, directions_note])
    directions_file.clear(lambda: (None, directions_summary(None)), None, [directions_state, directions_note])

    def open_vector(path):
        if not path:
            return None, vector_summary(None), gr.skip()
        try:
            vector = read_steering_vector(path)
        except (OSError, ValueError) as exc:
            raise gr.Error(f"That vector could not be opened: {exc}") from exc
        return vector, vector_summary(vector), vector["strength"]

    vector_file.upload(open_vector, vector_file, [vector_state, vector_note, strength])
    vector_file.clear(lambda: (None, vector_summary(None), gr.skip()), None, [vector_state, vector_note, strength])

    def switch_mode(chosen):
        labels = {experiment.CLAMP: "Clamp to", experiment.ADD: "Add α"}
        return gr.update(label=labels.get(chosen, ""), visible=chosen in labels)

    mode.change(switch_mode, mode, value, queue=False)

    controls = [passage, directions_state, vector_state, strength, injection_first, injection_last, mode, value,
                block_first, block_last, token_first, token_last, random_control, seed, word, read_first, read_last,
                read_block_first, read_block_last, differences]

    def run_experiment(view, *values):
        """Every pass for every condition in one model session, with a line of progress before each."""
        skip = (gr.skip(),) * (len(shown) + 1)
        turn = runs.start(view)
        fixed, prefixes = values[:len(controls)], values[len(controls):]
        try:
            inputs = gather(*fixed, list(zip(prefixes[0::2], prefixes[1::2])))
        except (TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        result = None
        try:
            with context.models.open_session() as session:
                runs.attach(view, turn, session)
                steps = experiment.run(session, inputs)
                try:
                    while True:
                        try:
                            line = next(steps)
                        except StopIteration as done:
                            result = done.value
                            break
                        yield (line if runs.live(view, turn) else gr.skip(),) + skip[1:]
                finally:
                    # A closed page stops here; the run lets go of the model and its hooks now, not at collection.
                    steps.close()
        except ProjectionCancelled:
            yield ("Stopped." if runs.live(view, turn) else gr.skip(),) + skip[1:]
            return
        except (TypeError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            runs.finish(view, turn)
        if not runs.live(view, turn):
            yield skip
            return
        try:
            path = runs.stage(view, result)
        except OSError as exc:
            logger.warning("Could not write the direction edits result: %s", exc)
            path = None
        yield ("Done." if path else "Done, but the result could not be written for download.",
               *show(result, path))

    flat_rows = [box for row in rows for box in row]
    event = run.click(run_experiment, [owner, *controls, *flat_rows], [status, *shown],
                      concurrency_id="direction-edits-model", show_progress="hidden")

    def stop_run(view):
        runs.cancel(view)
        return "Stopped."

    stop.click(stop_run, owner, status, queue=False, cancels=[event])

    def pick(result, index):
        if result is None or index is None or not 0 <= int(index) < len(result["conditions"]):
            return gr.skip()
        return render.heatmaps(result, int(index), palette, display)

    picker.input(pick, [result_state, picker], heat, show_progress="hidden")

    def open_result(path, view):
        if not path:
            return (gr.skip(),) * (len(shown) + 1)
        turn = runs.start(view)
        try:
            result = files.read_result(path)
        except (OSError, ValueError) as exc:
            raise gr.Error(f"That result could not be opened: {exc}") from exc
        if not runs.live(view, turn):
            return (gr.skip(),) * (len(shown) + 1)
        return (f"Opened a result for `{result['model']['model_id']}`.", *show(result, runs.stage(view, result)))

    upload.upload(open_result, [upload, owner], [status, *shown], show_progress="hidden")
