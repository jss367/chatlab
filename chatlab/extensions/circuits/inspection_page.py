"""Native inspection and graph-feature validation, with session-scoped downloads."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr

from chatlab.activation_patching import integer
from chatlab.extension_api import write_private_text
from chatlab.extensions.result_runs import ResultRuns

from .inspection_render import empty_inspection, inspection_result
from .attribution import Cancelled
from .workbench import parse_prefixes, parse_tokens


class InspectionRuns(ResultRuns):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = Path(data_dir)

    def _stage(self, owner, result):
        directory = self._downloads.get(owner)
        if directory is None:
            directory = self._downloads[owner] = tempfile.TemporaryDirectory(prefix="chatlab-inspection-")
        path = Path(directory.name) / "component-inspection.json"
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        write_private_text(path, text)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        write_private_text(self.data_dir / f'{result["id"]}.json', text)
        return str(path)



def build_inspection(context, bench, graph_state=None, focus=None):
    """A separate native view, or validation of the graph's selected feature."""
    feature_mode = graph_state is not None
    runs = InspectionRuns(context.data_dir / "inspections")

    def forget_inspection(owner):
        bench.cancel(owner)
        runs.forget(owner)

    owner = gr.State(value=runs.new_owner, delete_callback=forget_inspection)
    ticket = gr.State(None)
    gr.Markdown("Select a feature in **Graph** and measure its role with the controls below."
                if feature_mode else
                "Inspect a native neuron or attention head, then test how it shapes the next token.",
                elem_classes=["ci-intro"])
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290, elem_classes=["ci-controls"]):
            with gr.Group(elem_classes=["ci-control-section"]):
                gr.Markdown("01 · COMPONENT", elem_classes=["ci-control-heading"])
                if feature_mode:
                    selection_note = gr.Markdown("Select a feature in Graph first.")
                else:
                    kind = gr.Radio([("MLP neuron", "neuron"), ("Attention head", "head")],
                                    value="neuron", label="Component", show_label=False)
                    with gr.Row():
                        layer = gr.Number(label="Layer", value=0, minimum=0, precision=0, min_width=90)
                        index = gr.Number(label="Neuron or head index", value=0, minimum=0, precision=0, min_width=90)
                    nearby = gr.Checkbox(label="Find nearby neurons at the edited token", value=True,
                                         info="For neurons: adjacent layers at the selected token.")
                    with gr.Accordion("Model requirements", open=False):
                        gr.Markdown("Full-precision Transformers weights on one device. No transcoders needed. "
                                    "A neuron is a channel after the MLP gate; a head is its vector before "
                                    "the attention output projection. Layers and indices start at 0.")
                        open_models = gr.Button("Open Models", size="sm")
                        context.navigation.open_models(open_models)
            if not feature_mode:
                with gr.Group(elem_classes=["ci-control-section"]):
                    gr.Markdown("02 · PROMPT", elem_classes=["ci-control-heading"])
                    user = gr.Textbox(label="User message", lines=3, placeholder="Write a prompt to investigate…")
                    prefix = gr.Textbox(label="Reply so far", lines=2,
                                        placeholder="Optional assistant continuation")
                    with gr.Accordion("Prompt options", open=False):
                        system = gr.Textbox(label="System prompt", lines=2)
                        raw = gr.Checkbox(label="Plain text, no chat template", value=False)
            with gr.Group(elem_classes=["ci-control-section"]):
                gr.Markdown("MEASUREMENTS" if feature_mode else "03 · MEASUREMENTS",
                            elem_classes=["ci-control-heading"])
                targets = gr.Textbox(label="Output tokens, one per line", lines=2,
                                     placeholder=" Wait", info="Keep leading spaces. One token per line.")
                offset = gr.Number(label="Edited token offset from end", value=0, minimum=0, precision=0,
                                   info="0 selects the last prompt token.")
                scale_text = gr.Textbox(label="Scale factors", value="0, 0.5, 1, 2, 3",
                                        info="Includes zero (ablation) and one (control).")
            with gr.Accordion("Additional prompts", open=False):
                extra = gr.Textbox(label="Additional destination replies, separated by --- lines", lines=3,
                                   info="Same system and user message. Each prefix is measured separately.")
            with gr.Accordion("Patch from a contrasting prompt", open=False):
                patch = gr.Checkbox(label="Copy the source component activation", value=False)
                donor_user = gr.Textbox(label="Source user message", lines=3,
                                        info="Uses the destination system prompt and template setting.")
                donor_prefix = gr.Textbox(label="Source reply so far", lines=2)
                donor_offset = gr.Number(label="Source token offset from end", value=0, minimum=0, precision=0)
            with gr.Accordion("Test a proposed pathway", open=False):
                restore = gr.Checkbox(label="After ablation, restore a later component", value=False)
                if feature_mode:
                    hub_pick = gr.Dropdown(label="Later feature", choices=[], value=None)
                else:
                    hub_kind = gr.Radio([("MLP neuron", "neuron"), ("Attention head", "head")],
                                        value="neuron", label="Later component")
                    with gr.Row():
                        hub_layer = gr.Number(label="Later layer", value=1, minimum=0, precision=0, min_width=90)
                        hub_index = gr.Number(label="Later neuron or head index", value=0, minimum=0, precision=0, min_width=90)
                hub_offset = gr.Number(label="Later component token offset from end", value=0, minimum=0, precision=0)
                gr.Markdown("Restore the later component’s baseline activation while the upstream component "
                            "remains zero. Compare the two conditions in the results.")
            with gr.Row(elem_classes=["ci-actions"]):
                inspect_button = gr.Button("Test selected feature" if feature_mode else "Inspect and test", variant="primary")
                stop_button = gr.Button("Stop", scale=0, min_width=70)
        with gr.Column(scale=3):
            progress = gr.Markdown("", elem_classes=["ci-progress"])
            output = gr.HTML(empty_inspection(), elem_classes=["ci-output"])
            download = gr.File(label="Download inspection · JSON", interactive=False, elem_classes=["ci-download"])

    if feature_mode:
        def inspection_choices(graph, focused):
            node = next((n for n in (graph or {}).get("nodes", [])
                         if n["id"] == focused and n["kind"] == "feature"), None)
            choices = [(f'Layer {n["layer"]} · feature {n["feature"]} · position {n["position"]}', n["id"])
                       for n in (graph or {}).get("nodes", [])
                       if n["kind"] == "feature" and node is not None and n["layer"] > node["layer"]]
            note = (f'Selected: layer {node["layer"]} · feature {node["feature"]} · position {node["position"]}.'
                    if node is not None else "Select a feature in Graph first.")
            at = len(graph["ids"]) - 1 - node["position"] if node is not None else 0
            return gr.update(choices=choices, value=None), note, at
        for control in (graph_state, focus):
            control.change(inspection_choices, [graph_state, focus], [hub_pick, selection_note, offset], queue=False)

        def inspection_hub_position(graph, chosen):
            node = next((n for n in (graph or {}).get("nodes", []) if n["id"] == chosen), None)
            return len(graph["ids"]) - 1 - node["position"] if node is not None else 0
        hub_pick.input(inspection_hub_position, [graph_state, hub_pick], hub_offset, queue=False)

        def invalidate_inspection(view):
            runs.cancel(view)
            bench.cancel(view)
            return "Selection changed. Run a new inspection.", empty_inspection(), None

        for control in (graph_state, focus):
            control.change(invalidate_inspection, owner, [progress, output, download], queue=False)

    def begin_inspection(view):
        bench.cancel(view)
        return runs.start(view)

    def run_inspection(view, turn, *values):
        if not runs.live(view, turn):
            yield gr.skip(), gr.skip(), gr.skip()
            return
        try:
            if feature_mode:
                graph, focused, restored_id, *values = values
                node = next((n for n in (graph or {}).get("nodes", []) if n["id"] == focused and n["kind"] == "feature"), None)
                if node is None:
                    raise ValueError("Select a feature in Graph first.")
                prompt = graph["prompt"]
                component = ("feature", node["layer"], node["feature"])
                hub_node = next((n for n in graph["nodes"] if n["id"] == restored_id and n["kind"] == "feature"), None)
            else:
                component_kind, layer_value, index_value, system_text, user_text, prefix_text, plain, \
                    later_kind, later_layer, later_index, *values = values
                do_nearby = bool(values.pop())
                graph = None
                component = (component_kind, integer(layer_value, "Layer"), integer(index_value, "Component index"))
                prompt = {"system": system_text or "", "user": user_text or "", "prefix": prefix_text or "", "raw": bool(plain)}
                if not prompt["user"].strip():
                    raise ValueError("Write a destination user message first.")
            at, token_text, scales, extras, do_patch, source_user, source_prefix, source_at, do_restore, later_at = values
            hub = None
            if do_restore:
                if feature_mode:
                    if hub_node is None:
                        raise ValueError("Choose the later feature to restore.")
                    hub = ("feature", hub_node["layer"], hub_node["feature"])
                else:
                    hub = (later_kind, integer(later_layer, "Later layer"), integer(later_index, "Later component index"))
            donor = None
            if do_patch:
                if not (source_user or "").strip():
                    raise ValueError("Write the source user message before enabling patching.")
                donor = {**prompt, "user": source_user, "prefix": source_prefix or ""}
            at, source_at, later_at = (integer(v, label) for v, label in
                                       ((at, "Edited token offset"), (source_at, "Source token offset"),
                                        (later_at, "Later component token offset")))

            def work(report, stopped):
                if not runs.live(view, turn):
                    raise Cancelled()
                return bench.inspect_components(prompt, component, parse_tokens(token_text), scales, at,
                                                parse_prefixes(extras), donor, source_at, hub, later_at,
                                                report, lambda: stopped() or not runs.live(view, turn), graph=graph,
                                                nearby=do_nearby if not feature_mode else False)

            for item in bench.background(view, work):
                if not runs.live(view, turn):
                    continue
                if item[0] == "progress":
                    yield f"{item[1]}… {item[2]} of {item[3]}", gr.skip(), gr.skip()
                else:
                    result = item[1]
                    # A selection or opened graph can change while this runs.
                    # Results identify the exact graph/component and are self-contained.
                    frame = runs.publish(view, turn, result, lambda path: ("Inspection complete.", inspection_result(result), path))
                    if frame is not None:
                        yield frame
        except Cancelled:
            if runs.live(view, turn):
                yield "Stopped.", gr.skip(), gr.skip()
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    shared = [offset, targets, scale_text, extra, patch, donor_user, donor_prefix, donor_offset, restore, hub_offset]
    inputs = ([graph_state, focus, hub_pick] if feature_mode else
              [kind, layer, index, system, user, prefix, raw, hub_kind, hub_layer, hub_index])
    event = inspect_button.click(begin_inspection, owner, ticket, queue=False).success(
        run_inspection, [owner, ticket, *inputs, *shared, *([] if feature_mode else [nearby])], [progress, output, download],
        concurrency_id="circuits", show_progress="hidden")

    def stop_inspection(view):
        runs.cancel(view)
        bench.cancel(view)
        return "Stopped."

    stop_button.click(stop_inspection, owner, progress, queue=False, cancels=[event])
