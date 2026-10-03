"""The Circuits page: trace a token's attribution graph, read its features, group and test them."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import gradio as gr

from chatlab.ui.token_menu import MENU_BRIDGE_CLASS

from . import render, transcoders
from .attribution import Cancelled, Settings
from .interventions import DEFAULT_BOOST
from .workbench import Workbench, load_graph, parse_prefixes, parse_tokens

logger = logging.getLogger(__name__)

EXPLAIN = {"The likeliest next tokens": "top", "Chosen tokens": "tokens",
           "Pivot tokens against others": "contrast"}
EXAMPLE = "What language is spoken in the country whose capital is Paris? Answer in one word."
NO_GRAPH = '<div class="cg-root viz-root cg-empty">Write a prompt and press <b>Trace</b>.</div>'

CSS = render.CSS

JS = r"""
() => {
  if (window.chatlabCircuitsInstalled) return;
  window.chatlabCircuitsInstalled = true;
  const write = (selector, value) => {
    const field = document.querySelector(selector + ' textarea, ' + selector + ' input');
    if (!field) return;
    const prototype = field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(field, value);
    field.dispatchEvent(new Event('input', {bubbles: true}));
  };
  document.addEventListener('click', event => {
    const node = event.target.closest('#circuits-graph .cg-node[data-node]');
    if (node) {
      const svg = node.closest('svg');
      const id = node.dataset.node;
      let chosen = Array.from(svg.querySelectorAll('.cg-node.sel')).map(n => n.dataset.node);
      if (event.shiftKey || event.metaKey || event.ctrlKey) {
        chosen = chosen.includes(id) ? chosen.filter(x => x !== id) : chosen.concat([id]);
      } else {
        chosen = [id];
      }
      svg.querySelectorAll('.cg-node[data-node]').forEach(n => n.classList.toggle('sel', chosen.includes(n.dataset.node)));
      write('#circuits-pick', JSON.stringify({selected: chosen, focus: id, nonce: Date.now()}));
      return;
    }
    const group = event.target.closest('#circuits-groups .cg-group[data-group]');
    if (group) write('#circuits-group-pick', JSON.stringify({name: group.dataset.group, nonce: Date.now()}));
  });
  const light = (node, on) => {
    const svg = node.closest('svg');
    if (!svg) return;
    svg.classList.toggle('hovering', on);
    svg.querySelectorAll('.lit').forEach(el => el.classList.remove('lit'));
    if (!on) return;
    const id = node.dataset.node;
    node.classList.add('lit');
    svg.querySelectorAll('.cg-edge').forEach(edge => {
      if (edge.dataset.s === id || edge.dataset.t === id) {
        edge.classList.add('lit');
        const other = edge.dataset.s === id ? edge.dataset.t : edge.dataset.s;
        svg.querySelectorAll('.cg-node[data-node]').forEach(n => { if (n.dataset.node === other) n.classList.add('lit'); });
      }
    });
  };
  // A new graph opens on its last tokens, where the target is.
  new MutationObserver(() => {
    document.querySelectorAll('#circuits-graph .cg-scroll:not([data-revealed])').forEach(wrap => {
      wrap.dataset.revealed = '1';
      wrap.scrollLeft = Math.max(0, Number(wrap.dataset.focus || 0) - 60);
    });
  }).observe(document.body, {childList: true, subtree: true});
  document.addEventListener('mouseover', event => {
    const node = event.target.closest && event.target.closest('#circuits-graph .cg-node[data-node]');
    if (node) light(node, true);
  });
  document.addEventListener('mouseout', event => {
    const node = event.target.closest && event.target.closest('#circuits-graph .cg-node[data-node]');
    if (node && !(event.relatedTarget && node.contains(event.relatedTarget))) light(node, false);
  });
}
"""


def status_text(status):
    model_id = status["model_id"]
    supported = ", ".join(f"`{m}`" for m in transcoders.supported_models())
    if not model_id:
        return f"**No model loaded.** Circuit tracing works with {supported}."
    spec = status["spec"]
    if spec is None:
        return f"**`{model_id}` has no published transcoders.** Load one of {supported}."
    where = ("in memory" if status["in_memory"] else
             "downloaded, read into memory on the first trace" if status["downloaded"]
             else f"not downloaded yet: {spec.weights_gb:.1f} GB, fetched on the first trace")
    compatibility = (f"Training checkpoint: `{spec.training_model_revision}`. Other checkpoints are refused."
                     if spec.training_model_revision else
                     "The publisher does not report a training checkpoint; checkpoint compatibility is unverified. "
                     "Feature examples describe the publisher’s training data.")
    return f"**`{model_id}`** · {spec.title} from `{spec.repo}` · {where}.\n\n{compatibility}"


def _progress_text(stage, done, total):
    if total and total > 1:
        return f"{stage}… {done} of {total}"
    return f"{stage}…"


def build_page(context):
    bench = Workbench(context.models, context.data_dir)
    staging, staging_lock = {}, threading.Lock()
    versions, focused_nodes, version_lock = {}, {}, threading.RLock()
    trace_requests = {}

    def version(session_id):
        with version_lock:
            return versions.get(session_id)

    def current_graph(graph, session_id):
        return graph is not None and graph.get("_view_version") == version(session_id)

    def checked_version(graph, session_id):
        with version_lock:
            return current_graph(graph, session_id), version(session_id)

    def forget(session_id):
        bench.cancel(session_id)
        with version_lock:
            versions.pop(session_id, None)
            focused_nodes.pop(session_id, None)
            trace_requests.pop(session_id, None)
        with staging_lock:
            directory = staging.pop(session_id, None)
        if directory is not None:
            directory.cleanup()

    with gr.Column(elem_id="circuits-page"):
        owner = gr.State(value=lambda: uuid4().hex, delete_callback=forget)
        graph_state = gr.State(None)
        trace_request = gr.Textbox(visible=False)
        selection = gr.State([])
        focus = gr.State(None)
        gr.Markdown("# Circuit tracing\nWhich transcoder features carried the model to a token. Each MLP is "
                    "replaced by a transcoder, attention and normalization are frozen, and every edge is one "
                    "feature's direct effect on another. The graph is a claim about the model; the "
                    "interventions under **Groups** test it on the real one.")
        with gr.Row():
            status = gr.Markdown(status_text(bench.status()))
        with gr.Row():
            wanted = gr.Textbox(value="google/gemma-3-1b-it", visible=False)
            open_models = gr.Button("Open Models", size="sm", scale=0)
            context.navigation.open_models(open_models, wanted)
            refresh = gr.Button("Refresh", size="sm", scale=0)
            load_button = gr.Button("Load transcoders", size="sm", scale=0)
            unload_button = gr.Button("Unload transcoders", size="sm", scale=0)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("System prompt", open=False):
                    system = gr.Textbox(label="System prompt", lines=3, show_label=False)
                user = gr.Textbox(label="User message", lines=4, value=EXAMPLE)
                prefix = gr.Textbox(label="Reply so far", lines=3,
                                    info="The reply starts with this text. The graph explains the token after it.")
                raw = gr.Checkbox(label="Plain text, no chat template", value=False,
                                  info="For base models: the user message is the whole prompt.")
                explain = gr.Radio(list(EXPLAIN), value="The likeliest next tokens", label="Explain")
                explain_tokens = gr.Textbox(label="Tokens, one per line", lines=3, visible=False,
                                            info='Keep leading spaces. Type \\n for a line break.')
                explain_others = gr.Textbox(label="Other tokens, one per line", lines=3, visible=False)
                with gr.Accordion("Graph size", open=False):
                    max_nodes = gr.Slider(32, 4096, value=400, step=32, label="Features to trace")
                    node_threshold = gr.Slider(0.5, 1.0, value=0.8, step=0.01, label="Keep features holding this share of influence")
                    edge_threshold = gr.Slider(0.5, 1.0, value=0.98, step=0.01, label="Keep edges holding this share of influence")
                    batch = gr.Slider(1, 128, value=32, step=1, label="Targets per backward pass")
                with gr.Row():
                    trace = gr.Button("Trace", variant="primary")
                    stop = gr.Button("Stop")
                progress = gr.Markdown("")
                with gr.Accordion("Saved graphs", open=False):
                    saved = gr.Dropdown(label="Open a saved graph", choices=bench.saved(), value=None)
                    upload = gr.File(label="Open a graph file", file_types=[".json"], type="filepath")
                    download = gr.File(label="This graph", interactive=False)
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Graph"):
                        with gr.Row():
                            nodes_shown = gr.Slider(5, 200, value=40, step=1, label="Features shown")
                            show_errors = gr.Checkbox(label="Show transcoder error nodes", value=False)
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5):
                                graph_view = gr.HTML(NO_GRAPH, elem_id="circuits-graph")
                            with gr.Column(scale=2, min_width=340):
                                card = gr.HTML(render.feature_card(None))
                                with gr.Row():
                                    label = gr.Textbox(label="Name this feature", scale=3)
                                    rename = gr.Button("Rename", size="sm", scale=1)
                                ablate = gr.Button("Ablate this feature", size="sm")
                                selected_note = gr.Markdown("")
                                with gr.Row():
                                    group_name = gr.Textbox(label="Group name", scale=3,
                                                            placeholder="re-evaluate: Wait / Let me")
                                    make_group = gr.Button("Group selected", size="sm", scale=1)
                    with gr.Tab("Groups"):
                        groups_view = gr.HTML(render.group_view({"nodes": [], "edges": []}, {}),
                                              elem_id="circuits-groups")
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1, min_width=220):
                                group_pick = gr.Dropdown(label="Group", choices=[], value=None)
                                remove_group = gr.Button("Delete this group", size="sm")
                            with gr.Column(scale=3):
                                group_card = gr.HTML(render.group_card(None, [], {"nodes": []}))
                        with gr.Accordion("Interventions", open=True):
                            with gr.Row():
                                pivot = gr.Textbox(label="Pivot tokens, one per line", lines=4,
                                                   info="P(pivot) is their summed probability.")
                                alternatives = gr.Textbox(label="Alternatives, one per line", lines=4,
                                                          info="Blank: the model's likeliest other tokens.")
                            with gr.Row():
                                boost = gr.Number(label="Boost factor", value=DEFAULT_BOOST, minimum=0, maximum=100)
                                every_position = gr.Checkbox(
                                    label="Change features at every position", value=False,
                                    info="Otherwise only where the graph found them, counted back from the end.")
                                include_prompt = gr.Checkbox(label="Include the traced prompt", value=True)
                            prefixes = gr.Textbox(
                                label="More replies so far, separated by lines holding only ---", lines=5,
                                info="Each one follows the same system prompt and user message. Probabilities "
                                     "are averaged over all of them.")
                            with gr.Row():
                                run = gr.Button("Run interventions", variant="primary")
                                stop_run = gr.Button("Stop")
                            run_progress = gr.Markdown("")
        # What the page's script writes when a node or a group is clicked.
        pick = gr.Textbox(elem_id="circuits-pick", elem_classes=[MENU_BRIDGE_CLASS])
        group_pick_bridge = gr.Textbox(elem_id="circuits-group-pick", elem_classes=[MENU_BRIDGE_CLASS])

    # Rendering -------------------------------------------------------------

    def draw(graph, chosen, shown, errors):
        if not graph:
            return NO_GRAPH
        return render.graph_view(graph, nodes_shown=int(shown), show_errors=bool(errors), selected=chosen,
                                 labels=graph.get("labels"), groups=graph.get("groups"))

    def decoder(graph):
        texts = (graph.get("effects") or {}).get("token_text", {})
        return lambda token: texts.get(str(token), str(token))

    def draw_groups(graph, name):
        if not graph:
            return (render.group_view({"nodes": [], "edges": []}, {}), gr.update(choices=[], value=None),
                    render.group_card(None, [], {"nodes": []}))
        names = list(graph["groups"])
        name = name if name in graph["groups"] else (names[0] if names else None)
        view = render.group_view(graph, graph["groups"], graph.get("effects"), graph.get("labels"), decoder(graph))
        detail = render.group_card(name, graph["groups"].get(name, []), graph, graph.get("effects"),
                                   graph.get("labels"), decoder(graph))
        return view, gr.update(choices=names, value=name), detail

    def node_of(graph, node_id):
        if not graph or not node_id:
            return None
        return next((n for n in graph["nodes"] if n["id"] == node_id), None)

    def describe_card(graph, node_id, ablation=None):
        node = node_of(graph, node_id)
        record, problem = None, None
        if node is not None and node["kind"] == "feature":
            try:
                record = bench.record(graph, node)
            except OSError as exc:
                problem = str(exc)
        return render.feature_card(node, record, graph.get("labels") if graph else None, ablation, problem,
                                   graph["tokens"] if graph else None)

    def staged(path, graph, session_id):
        """One owned download location per view, reused after every mutation."""
        with staging_lock:
            directory = staging.get(session_id)
            if directory is None:
                directory = staging[session_id] = tempfile.TemporaryDirectory(prefix="chatlab-circuits-")
            copy = Path(directory.name) / "circuit.json"
            partial = Path(directory.name) / ".circuit.tmp"
            try:
                shutil.copyfile(path, partial)
                partial.replace(copy)
            finally:
                partial.unlink(missing_ok=True)
            return str(copy)

    def save(graph, session_id, checked=False):
        """Save the graph, and refresh the current view's download copy."""
        try:
            with version_lock:
                if checked and not current_graph(graph, session_id):
                    return False
                versions[session_id] = graph["_view_version"] = uuid4().hex
                return staged(bench.save(graph), graph, session_id)
        except OSError as exc:
            logger.warning("Could not save graph %s: %s", graph.get("id"), exc)
            gr.Warning(f"The graph was not saved: {exc}.")
            return None

    def show_graph(graph, path, shown, errors):
        """Every output that follows from a whole new graph."""
        groups = draw_groups(graph, None)
        targets = [n for n in graph["nodes"] if n["kind"] == "target"]
        pivot_text = gr.skip()
        explain = graph.get("explain") or {}
        if explain.get("mode") == "contrast":
            pivot_text = "\n".join(t.replace("\n", "\\n") for t in explain.get("tokens", []))
        elif targets and targets[0].get("target_kind") == "token":
            pivot_text = targets[0]["text"].replace("\n", "\\n")
        return (graph, [], None, draw(graph, [], shown, errors), render.feature_card(None), "",
                *groups, path, gr.update(choices=bench.saved()), pivot_text)

    graph_outputs = [graph_state, selection, focus, graph_view, card, selected_note,
                     groups_view, group_pick, group_card, download, saved, pivot]

    # Status ----------------------------------------------------------------

    refresh.click(lambda: status_text(bench.status()), None, status, queue=False)

    def load_now(session_id):
        try:
            for item in bench.background(session_id, bench.load_transcoders):
                if item[0] == "progress":
                    yield _progress_text(*item[1:]), gr.skip()
        except Cancelled:
            yield "Stopped.", status_text(bench.status())
            return
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        yield "", status_text(bench.status())

    load_event = load_button.click(load_now, owner, [progress, status], concurrency_id="circuits", show_progress="hidden")

    def unload_now():
        transcoders.unload()
        return status_text(bench.status())

    unload_button.click(unload_now, None, status, concurrency_id="circuits")

    explain.change(lambda choice: (gr.update(visible=EXPLAIN[choice] != "top"),
                                   gr.update(visible=EXPLAIN[choice] == "contrast")),
                   explain, [explain_tokens, explain_others], queue=False)

    # Tracing -----------------------------------------------------------------

    def begin_trace(session_id):
        with version_lock:
            request = uuid4().hex
            trace_requests[session_id] = (request, version(session_id))
            return request

    def run_trace(session_id, system_text, user_text, prefix_text, plain, choice, tokens_text, others_text,
                  nodes, node_share, edge_share, batch_size, shown, errors, request=None):
        with version_lock:
            captured = trace_requests.get(session_id)
            stale_request = request is not None and (not captured or captured[0] != request or captured[1] != version(session_id))
            stamp = captured[1] if request is not None and captured else version(session_id)
        if stale_request:
            yield (gr.skip(),) * (len(graph_outputs) + 2)
            return
        settings = Settings(int(nodes), int(batch_size), float(node_share), float(edge_share))
        prompt = {"system": system_text or "", "user": user_text or "", "prefix": prefix_text or "",
                  "raw": bool(plain)}
        explain_spec = {"mode": EXPLAIN[choice], "tokens": parse_tokens(tokens_text),
                        "others": parse_tokens(others_text)}
        if not prompt["user"].strip():
            raise gr.Error("Write a user message first.")
        skip = (gr.skip(),) * len(graph_outputs)
        graph = None
        try:
            settings.check()
            for item in bench.background(session_id, lambda p, c: bench.trace(prompt, explain_spec, settings, p, c)):
                if item[0] == "progress":
                    yield (_progress_text(*item[1:]), gr.skip(), *skip)
                else:
                    graph = item[1]
        except Cancelled:
            yield ("Stopped.", gr.skip(), *skip)
            return
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        with version_lock:
            stale = version(session_id) != stamp or (request is not None and trace_requests.get(session_id, (None,))[0] != request)
            if not stale:
                path = save(graph, session_id)
                stamp = version(session_id)
        if stale:
            yield (gr.skip(), gr.skip(), *skip)
            return
        stats = graph["stats"]
        frame = (f"Traced {stats['traced_features']} of {stats['active_features']:,} active features. "
                 f"Checkpoint compatibility: {graph.get('checkpoint_compatibility', 'unverified')}.",
                 status_text(bench.status()), *show_graph(graph, path, shown, errors))
        yield frame if (version(session_id) == stamp and
                        (request is None or trace_requests.get(session_id, (None,))[0] == request)) else (gr.skip(), gr.skip(), *skip)

    trace_event = trace.click(begin_trace, owner, trace_request, queue=False).success(run_trace, [owner, system, user, prefix, raw, explain, explain_tokens, explain_others,
                            max_nodes, node_threshold, edge_threshold, batch, nodes_shown, show_errors, trace_request],
                [progress, status, *graph_outputs], concurrency_id="circuits", show_progress="hidden")

    def redraw(graph, chosen, shown, errors, session_id):
        fresh, stamp = checked_version(graph, session_id)
        if not fresh:
            return gr.skip()
        frame = draw(graph, chosen, shown, errors)
        with version_lock:
            return frame if version(session_id) == stamp and current_graph(graph, session_id) else gr.skip()

    for control in (nodes_shown, show_errors):
        control.change(redraw, [graph_state, selection, nodes_shown, show_errors, owner], graph_view, queue=False)

    # Selecting, naming and grouping ---------------------------------------------

    def picked(graph, raw_pick, session_id):
        with version_lock:
            if not current_graph(graph, session_id):
                return (gr.skip(),) * 5
            stamp = version(session_id)
        try:
            data = json.loads(raw_pick)
            ids = {n["id"] for n in graph["nodes"]}
            chosen = [i for i in data["selected"] if isinstance(i, str) and i in ids]
            focused = data["focus"] if data.get("focus") in ids else None
        except (TypeError, ValueError, KeyError):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        features = sum(1 for n in graph["nodes"] if n["id"] in chosen and n["kind"] == "feature")
        note = (f"{len(chosen)} selected, {features} of them features." if len(chosen) > 1 else "")
        node = node_of(graph, focused)
        name = (graph.get("labels") or {}).get(focused, "") if node and node["kind"] == "feature" else ""
        with version_lock:
            if version(session_id) != stamp or not current_graph(graph, session_id):
                return (gr.skip(),) * 5
            focused_nodes[session_id] = focused
        detail = describe_card(graph, focused)
        with version_lock:
            if (version(session_id) != stamp or not current_graph(graph, session_id)
                    or focused_nodes.get(session_id) != focused):
                return (gr.skip(),) * 5
            focused_nodes[session_id] = focused
            return chosen, focused, detail, note, name

    pick.input(picked, [graph_state, pick, owner], [selection, focus, card, selected_note, label],
               concurrency_id="circuits-read", trigger_mode="always_last", show_progress="hidden")

    def rename_node(graph, focused, text, chosen, shown, errors, session_id):
        if not current_graph(graph, session_id):
            return (gr.skip(),) * 7
        node = node_of(graph, focused)
        if node is None or node["kind"] != "feature":
            raise gr.Error("Click a feature in the graph first.")
        labels = dict(graph.get("labels") or {})
        if text.strip():
            labels[focused] = text.strip()[:80]
        else:
            labels.pop(focused, None)
        graph = {**graph, "labels": labels}
        path = save(graph, session_id, checked=True)
        if path is False:
            return (gr.skip(),) * 7
        frame = (graph, draw(graph, chosen, shown, errors), describe_card(graph, focused), *draw_groups(graph, None), path)
        return frame if current_graph(graph, session_id) else (gr.skip(),) * 7

    rename.click(rename_node, [graph_state, focus, label, selection, nodes_shown, show_errors, owner],
                 [graph_state, graph_view, card, groups_view, group_pick, group_card, download], concurrency_id="circuits-read")

    def group_selected(graph, chosen, name, shown, errors, session_id):
        if graph and not current_graph(graph, session_id):
            return (gr.skip(),) * 7
        if not graph:
            raise gr.Error("Trace a graph first.")
        feature_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "feature"}
        members = list(dict.fromkeys(i for i in chosen if i in feature_ids))
        if not members:
            raise gr.Error("Select features in the graph first: click one, shift-click more.")
        name = (name or "").strip()[:60] or f"group {len(graph['groups']) + 1}"
        groups = {g: [m for m in ms if m not in members] for g, ms in graph["groups"].items()}
        groups = {g: ms for g, ms in groups.items() if ms}
        groups[name] = members
        graph = {**graph, "groups": groups, "effects": None}
        path = save(graph, session_id, checked=True)
        if path is False:
            return (gr.skip(),) * 7
        gr.Info(f"Grouped {len(members)} feature{'s' * (len(members) != 1)} as {name}.")
        frame = (graph, draw(graph, chosen, shown, errors), *draw_groups(graph, name), "", path)
        return frame if current_graph(graph, session_id) else (gr.skip(),) * 7

    make_group.click(group_selected, [graph_state, selection, group_name, nodes_shown, show_errors, owner],
                     [graph_state, graph_view, groups_view, group_pick, group_card, group_name, download],
                     concurrency_id="circuits-read")

    def choose_group(graph, name, session_id):
        fresh, stamp = checked_version(graph, session_id)
        if not fresh:
            return gr.skip()
        if name not in graph["groups"]:
            card = render.group_card(None, [], {"nodes": []})
        else:
            card = render.group_card(name, graph["groups"][name], graph, graph.get("effects"), graph.get("labels"),
                                     decoder(graph))
        with version_lock:
            return card if version(session_id) == stamp and current_graph(graph, session_id) else gr.skip()

    group_pick.input(choose_group, [graph_state, group_pick, owner], group_card, queue=False)

    def group_clicked(graph, raw_pick, session_id):
        fresh, stamp = checked_version(graph, session_id)
        if not fresh:
            return gr.skip(), gr.skip()
        try:
            name = json.loads(raw_pick)["name"]
        except (TypeError, ValueError, KeyError):
            return gr.skip(), gr.skip()
        if name not in graph["groups"]:
            return gr.skip(), gr.skip()
        card = choose_group(graph, name, session_id)
        with version_lock:
            if version(session_id) != stamp or not current_graph(graph, session_id):
                return gr.skip(), gr.skip()
            return gr.update(value=name), card

    group_pick_bridge.input(group_clicked, [graph_state, group_pick_bridge, owner], [group_pick, group_card], queue=False)

    def delete_group(graph, name, chosen, shown, errors, session_id):
        if graph and not current_graph(graph, session_id):
            return (gr.skip(),) * 6
        if not graph or name not in graph["groups"]:
            raise gr.Error("Choose a group first.")
        groups = {g: m for g, m in graph["groups"].items() if g != name}
        graph = {**graph, "groups": groups, "effects": None}
        path = save(graph, session_id, checked=True)
        if path is False:
            return (gr.skip(),) * 6
        frame = (graph, draw(graph, chosen, shown, errors), *draw_groups(graph, None), path)
        return frame if current_graph(graph, session_id) else (gr.skip(),) * 6

    remove_group.click(delete_group, [graph_state, group_pick, selection, nodes_shown, show_errors, owner],
                       [graph_state, graph_view, groups_view, group_pick, group_card, download], concurrency_id="circuits-read")

    # Measuring -----------------------------------------------------------------

    def ablate_focused(session_id, graph, focused):
        with version_lock:
            fresh, stamp = checked_version(graph, session_id)
            if not fresh or focused_nodes.get(session_id, focused) != focused:
                return gr.skip()
        node = node_of(graph, focused)
        if node is None or node["kind"] != "feature":
            raise gr.Error("Click a feature in the graph first.")
        result = None
        try:
            for item in bench.background(session_id, lambda p, c: bench.ablate(graph, node, p, c)):
                if item[0] == "done":
                    result = item[1]
        except Cancelled:
            gr.Info("Stopped.")
            return gr.skip()
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        card = describe_card(graph, focused, result)
        with version_lock:
            fresh = version(session_id) == stamp and focused_nodes.get(session_id, focused) == focused
        return card if fresh else gr.skip()

    ablate_event = ablate.click(ablate_focused, [owner, graph_state, focus], card, concurrency_id="circuits")

    def run_interventions(session_id, graph, pivot_text, alternative_text, prefix_text, include, factor, everywhere,
                          name):
        fresh, stamp = checked_version(graph, session_id)
        if not fresh:
            yield (gr.skip(),) * 6
            return
        if not graph:
            raise gr.Error("Trace a graph first.")
        if not graph["groups"]:
            raise gr.Error("Group some features in the graph first.")
        skip = (gr.skip(),) * 5
        effects = None
        try:
            factor = float(factor)
            extra = parse_prefixes(prefix_text)
            if not include and not extra:
                raise ValueError("Include the traced prompt or add replies to average over.")

            def work(report, cancelled):
                return bench.group_effects(graph, parse_tokens(pivot_text), parse_tokens(alternative_text),
                                           extra, include, factor, bool(everywhere), report, cancelled)

            for item in bench.background(session_id, work):
                if item[0] == "progress":
                    yield (_progress_text(*item[1:]), *skip)
                else:
                    effects = item[1]
        except Cancelled:
            yield ("Stopped.", *skip)
            return
        except (TypeError, ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        with version_lock:
            stale = version(session_id) != stamp
            if not stale:
                graph = {**graph, "effects": effects}
                path = save(graph, session_id)
                stamp = version(session_id)
        if stale:
            yield (gr.skip(),) * 6
            return
        frame = (f"Measured {len(graph['groups'])} group{'s' * (len(graph['groups']) != 1)} on "
               f"{effects['prefixes']} prefix{'es' * (effects['prefixes'] != 1)}.", graph, *draw_groups(graph, name), path)
        yield frame if version(session_id) == stamp else (gr.skip(),) * 6

    intervention_event = run.click(run_interventions, [owner, graph_state, pivot, alternatives, prefixes, include_prompt, boost,
                                  every_position, group_pick],
              [run_progress, graph_state, groups_view, group_pick, group_card, download],
              concurrency_id="circuits", show_progress="hidden")

    def stop_now(session_id):
        with version_lock:
            trace_requests.pop(session_id, None)
        bench.cancel(session_id)

    queued_jobs = [load_event, trace_event, ablate_event, intervention_event]
    stop.click(stop_now, owner, None, queue=False, cancels=queued_jobs)
    stop_run.click(stop_now, owner, None, queue=False, cancels=queued_jobs)

    # Opening saved graphs --------------------------------------------------------

    def open_path(path, shown, errors, session_id):
        if not path:
            return (gr.skip(),) * len(graph_outputs)
        try:
            graph = load_graph(path)
        except (OSError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        try:
            with version_lock:
                versions[session_id] = graph["_view_version"] = uuid4().hex
                offered = staged(path, graph, session_id)
        except OSError:
            offered = None
        return show_graph(graph, offered, shown, errors)

    saved.input(open_path, [saved, nodes_shown, show_errors, owner], graph_outputs, concurrency_id="circuits-read")
    upload.upload(open_path, [upload, nodes_shown, show_errors, owner], graph_outputs, concurrency_id="circuits-read")
