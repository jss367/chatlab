"""Native maze workbench: real model episodes and inspectable saved replays."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import gradio as gr

from .maze import GOAL_MODES, PASSAGES, generate
from .runner import Episode, from_payload, stream_episode
from extension_api import TokenInspector

TOKENS = TokenInspector()

CSS = """
#maze-page {overflow-y:auto; padding:28px 32px; min-width:0 !important; min-height:0; height:100%; box-sizing:border-box; flex-wrap:nowrap; gap:18px;}
#maze-page > * {flex:0 0 auto;}
#maze-page .column {flex-wrap:nowrap;}
#maze-page .column > * {flex-shrink:0;}
#maze-page h1 {letter-spacing:-.035em; font-size:30px; margin-bottom:6px;}
#maze-board svg {width:100%; max-height:510px; display:block; margin:auto;}
#maze-board {background:#f6f7fb; border:1px solid #e4e7f0; border-radius:18px; padding:16px;}
.maze-legend {display:flex; justify-content:center; gap:16px; flex-wrap:wrap; color:#647084; font:12px system-ui; padding-top:12px;}
#maze-status {background:#f2f4ff; border-left:3px solid #6366f1; padding:12px 16px; border-radius:6px;}
#maze-tokens {max-height:270px; overflow:auto;}
#maze-raw textarea {font-family:ui-monospace,monospace; font-size:12px;}
#maze-controls {border:1px solid #e4e7f0; border-radius:12px; padding:14px;}
@media(max-width:700px) {#maze-page {padding:16px 12px;} #maze-page .column {min-width:0 !important; flex-basis:100% !important;} #maze-board svg {max-height:360px;}}
"""


def runs_dir(context):
    # Preserve explicit archive overrides used by existing experiments.
    return Path(os.environ.get("CHATLAB_MAZE_RUNS_PATH", str(context.data_dir))).expanduser()


def export_run(ep, directory):
    if ep.busy:
        raise gr.Error("Pause or stop the episode before exporting. Completed responses are also autosaved.")
    if not ep.replay_only:
        try:
            ep.save(directory)
        except OSError:
            gr.Warning("The run archive could not be written. Providing a temporary download instead.")
    return str(ep.export())


def board(ep, index=None, reveal=False, animate=False):
    maze = ep.maze
    events = [e for e in ep.events if e.get("turn", -1) <= index] if index is not None else ep.events
    accepted = [e for e in events if e["accepted"]]
    position = tuple(accepted[-1]["after"]) if accepted else maze.start
    size = maze.size
    cell, pad = 56, 28
    total = cell * size + 2 * pad
    def center(point):
        return pad + (point[1] + .5) * cell, pad + (point[0] + .5) * cell
    def points(path):
        return " ".join(f"{x},{y}" for x, y in map(center, path))
    parts = [f'<svg viewBox="0 0 {total} {total}" role="img" aria-label="{size} by {size} maze. Character at row {position[0]}, column {position[1]}.">']
    for r, row in enumerate(maze.grid):
        for c, value in enumerate(row):
            parts.append(f'<rect x="{pad+c*cell+2}" y="{pad+r*cell+2}" width="52" height="52" rx="7" fill="{"#27344a" if value == "#" else "#fff"}"/>')
    for i in range(size):
        parts.append(f'<text x="{pad+(i+.5)*cell}" y="17" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
        parts.append(f'<text x="12" y="{pad+(i+.5)*cell+4}" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
    if reveal:
        parts.append(f'<polyline points="{points(maze.route())}" fill="none" stroke="#b4bdcc" stroke-width="4" stroke-dasharray="3 8"/>')
    for e in accepted:
        supplied = e["source"] == "supplied"
        dash = 'stroke-dasharray="5 6"' if supplied else ''
        parts.append(f'<polyline points="{points([e["before"],e["after"]])}" fill="none" stroke="{"#94a3b8" if supplied else "#6366f1"}" stroke-width="6" stroke-linecap="round" {dash}/>')
    x, y = center(maze.start)
    parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#64748b" font-size="14" font-weight="700">S</text>')
    x, y = center(maze.goal)
    parts.append(f'<circle cx="{x}" cy="{y}" r="17" fill="#d1fae5"/><text x="{x}" y="{y+7}" text-anchor="middle" font-size="23" fill="#047857">★</text>')
    if ep.interrupted and (index is None or index >= ep.intervention_turn):
        x, y = center(ep.turns[ep.intervention_turn]["position_before"])
        parts.append(f'<circle cx="{x}" cy="{y}" r="23" stroke="#f59e0b" stroke-width="3" fill="none"/>')
    x, y = center(position)
    motion = ""
    if animate and accepted and accepted[-1]["source"] == "model":
        px, py = center(accepted[-1]["before"])
        motion = f'<animateTransform attributeName="transform" type="translate" from="{px} {py}" to="{x} {y}" dur="0.3s" fill="freeze"/>'
    parts.append(f'<g transform="translate({x} {y})">{motion}<circle r="17" fill="#4f46e5" stroke="white" stroke-width="3"/><circle cx="-5" cy="-2" r="2.5" fill="white"/><circle cx="5" cy="-2" r="2.5" fill="white"/><path d="M -5 6 Q 0 10 5 6" stroke="white" fill="none" stroke-width="2"/></g></svg>')
    parts.append('<div class="maze-legend"><span>● Character / model path</span><span>┄ Supplied moves</span><span>★ Destination</span><span style="color:#b77906">○ Interruption</span></div>')
    return "".join(parts)


def status(ep):
    partial = 0
    if ep.turns and ep.turns[-1]["finish_reason"] is None:
        t = ep.turns[-1]
        partial = max(0, len(t["metrics"]) - t["forced_prefix_tokens"])
    recovery = "Not inserted" if not ep.interrupted else ("Pending" if ep.resumed is None else (f"Returned in {ep.latency} sampled tokens" if ep.resumed else "No return"))
    return (f"**{'Replay · ' if ep.replay_only else ''}{ep.phase.title()}** · {html.escape(ep.detail)}\n\n"
            f"{ep.maze.size} × {ep.maze.size} · shortest route {len(ep.maze.route())-1} moves · "
            f"{ep.moves-ep.supplied_moves} model moves + {ep.supplied_moves} supplied · "
            f"{ep.sampled_tokens+partial:,} sampled tokens · {ep.tool_attempts} calls\n\n"
            f"**Goal information:** {GOAL_MODES[ep.config['goal_mode']]}\n\n"
            f"**Recovery:** {recovery} · **Model:** {html.escape(ep.model_id or 'load one on the Models page')}")


def timeline(ep):
    rows = [["Supplied", "→".join(map(str, ep.maze.start)), "", "Initial position"]]
    for e in ep.events:
        name = "Supplied" if e["source"] == "supplied" else f"Response {e['turn']+1}"
        rows.append([name, str(e["after"]), e.get("direction", "—"), "Accepted" if e["accepted"] else e["error"]])
    return rows


def views(ep, reveal, selections, session_id, index=None, animate=False):
    if index is None:
        index = len(ep.turns) - 1
    t = ep.turns[index] if 0 <= index < len(ep.turns) else {}
    metrics = t.get("metrics", [])
    forced = t.get("forced_prefix_tokens", 0)
    stamped, changed = selections.view(session_id, (ep.run_id, id(ep), index), metrics[forced:])
    origin = "Inside the template's open reasoning block" if t.get("reasoning_prefilled") else "At the beginning of the assistant response"
    note = (f"**Supplied interruption · {forced} tokens** · {origin}.\n\n" if forced else "No supplied interruption in this response.")
    if not forced and t.get("planned_prefix_ids"):
        note = f"**Pending interruption · {len(t['planned_prefix_ids'])} tokens** · Prefix insertion has not been confirmed."
    return (board(ep, index, reveal, animate), status(ep), TOKENS.strip(metrics[forced:]),
            t.get("text", ""), note, t.get("prefix_text") or t.get("planned_prefix_text", ""), timeline(ep), stamped,
            gr.update(choices=[("Initial / supplied history", -1)] + [(f"Response {i+1}" + (" · interruption" if t.get("prefix_ids") else ""), i) for i, t in enumerate(ep.turns)], value=index),
            "Select a model-generated token above." if changed else gr.skip(), [] if changed else gr.skip())


def _build_page(context):
    default_config = dict(supplied_moves=3, interrupt_after=3, interruption_text=next(iter(PASSAGES.values())),
                          prefix_tokens=8, temperature=.7, sampling_seed=20260914, per_turn_tokens=1024,
                          token_budget=8192, attempt_budget=32)
    initial = Episode(generate(), default_config)
    episode = gr.State(initial)
    selections = context.tokens.selections()
    selection_session = gr.State(value=selections.new_session, delete_callback=selections.forget)
    metrics_state = gr.State((None, []))
    gr.Markdown("# Maze workbench\nWatch a model navigate, interrupt its response, and inspect what happens next.")
    with gr.Row():
        with gr.Column(scale=5, min_width=310):
            maze_board = gr.HTML(board(initial), elem_id="maze-board")
            reveal = gr.Checkbox(label="Show shortest route (viewer only)", value=False)
            with gr.Row():
                run = gr.Button("Run episode", variant="primary", elem_id="maze-run")
                step = gr.Button("Step one response", elem_id="maze-step")
            with gr.Row():
                pause = gr.Button("Pause after response", size="sm")
                stop = gr.Button("Stop now", size="sm")
                interrupt = gr.Button("Interrupt next response", size="sm")
            with gr.Accordion("Maze & interruption settings", open=True, elem_id="maze-controls"):
                with gr.Row():
                    size = gr.Slider(3, 15, value=5, step=1, label="Maze width / height")
                    distance = gr.Number(value=10, precision=0, minimum=1, maximum=224, label="Shortest route length")
                with gr.Row():
                    seed = gr.Number(value=20260911, precision=0, label="Maze seed")
                    openness = gr.Slider(.35, .95, value=.7, step=.05, label="Open-cell probability")
                goal_mode = gr.Dropdown(choices=[(label, mode) for mode, label in GOAL_MODES.items()],
                                        value="coordinates", label="Goal information", elem_id="maze-goal-mode",
                                        info="Controls what the model knows. You always see the destination on the board.")
                goal_hint = gr.Textbox(label="Goal hint", lines=2, visible=False, elem_id="maze-goal-hint",
                                       info="Write a clue about the destination shown on the board. This text goes to the model verbatim; check it after changing the maze.")
                with gr.Row():
                    supplied = gr.Number(value=3, precision=0, minimum=0, maximum=223, label="Supplied starting moves", info="These moves follow the shortest route toward the goal. Use 0 for exploration without a demonstrated path.")
                    after = gr.Number(value=3, precision=0, minimum=0, maximum=255, label="Interrupt after accepted moves", info="Includes supplied moves. Inserted at the next response.")
                passage = gr.Dropdown(["None", *PASSAGES, "Custom"], value=next(iter(PASSAGES)), label="Interruption passage")
                text = gr.Textbox(value=next(iter(PASSAGES.values())), label="Interruption text", lines=3)
                prefix = gr.Number(value=8, precision=0, minimum=0, maximum=1024, label="Supplied token count", info="8 / 16 / 32 for the planned comparison; 0 uses all text.")
                with gr.Accordion("Generation limits", open=False):
                    temperature = gr.Slider(0, 2, value=.7, step=.05, label="Maze sampling temperature")
                    sampling_seed = gr.Number(value=20260914, precision=0, minimum=0, maximum=2147483647, label="Maze sampling seed")
                    per_turn = gr.Number(value=1024, precision=0, minimum=1, maximum=8192, label="Tokens per response")
                    budget = gr.Number(value=8192, precision=0, minimum=1, maximum=32768, label="Total sampled-token limit")
                    attempts = gr.Number(value=32, precision=0, minimum=1, maximum=256, label="Tool-attempt limit")
                prepare = gr.Button("New episode · apply settings", elem_id="maze-prepare")
                gr.Markdown("Edits apply when you create a **new episode**. Run and Step continue the current episode. Seeds reproduce the maze; model sampling may vary across hardware.")
        with gr.Column(scale=6, min_width=330):
            state_text = gr.Markdown(status(initial), elem_id="maze-status")
            gr.Markdown("### Emitted tokens\nLive output from the model, including reasoning tokens when its template exposes them. Supplied text appears separately below.")
            strip = gr.HighlightedText(label="Model-generated tokens · click to inspect", color_map=context.tokens.color_map,
                                       combine_adjacent=False, show_legend=True, elem_id="maze-tokens")
            prefix_note = gr.Markdown("No supplied interruption in this response.")
            prefix_text = gr.Textbox(label="Exact prefix · supplied or pending as noted above", interactive=False, lines=2)
            raw = gr.Textbox(label="Full response · supplied prefix + model output", interactive=False, lines=8, max_lines=16, elem_id="maze-raw")
            with gr.Accordion("Selected token probabilities", open=False):
                detail = gr.Markdown("Select a model-generated token above.")
                alternatives = gr.Dataframe(headers=["Token ID", "Text", "Raw probability"], interactive=False)
            with gr.Accordion("Path and replay", open=True):
                turn_picker = gr.Dropdown(choices=[("Initial / supplied history", -1)], value=-1, label="Inspect response", interactive=True)
                events = gr.Dataframe(value=timeline(initial), headers=["Source", "Position (row, column)", "Direction", "Result"], interactive=False, wrap=True)
                with gr.Row():
                    save = gr.Button("Export run JSON", size="sm")
                    download = gr.File(label="Saved run", interactive=False)
                upload = gr.File(label="Load a saved run for replay", file_types=[".json"], type="filepath")
            gr.Markdown("Exploratory tool: movement requires a completed, valid `move` call. Text claiming movement does not move the character. A natural end without a call ends the episode. After interruption, recovery allows 1,024 sampled tokens / 4 attempts. Run JSON records prompts, token IDs, probabilities, supplied text, actions and settings. The current tool parser supports Qwen-style `<tool_call>` responses. This view does not train a model.")
            models = gr.Button("Choose / load model", size="sm")
    outputs = [maze_board, state_text, strip, raw, prefix_note, prefix_text, events, metrics_state, turn_picker, detail, alternatives]
    controls = [size, seed, distance, openness, supplied, after, text, prefix, temperature, sampling_seed, per_turn, budget, attempts, goal_mode, goal_hint]

    def prepare_episode(ep, show, session_id, *values):
        if ep.busy:
            raise gr.Error("Stop or pause this episode before starting another.")
        n, s, d, o, supplied_n, trigger, passage_text, count, temp, sample_seed, per, total, tries, mode, hint = values
        try:
            new = Episode(generate(n, s, d, o), dict(supplied_moves=int(supplied_n), interrupt_after=int(trigger),
                          interruption_text=passage_text, prefix_tokens=int(count), temperature=float(temp),
                          sampling_seed=int(sample_seed), per_turn_tokens=int(per), token_budget=int(total), attempt_budget=int(tries),
                          goal_mode=mode, goal_hint=hint))
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        return (new, *views(new, show, selections, session_id), None)

    def play(ep, show, session_id, single=False):
        last_board = None
        try:
            for current in stream_episode(ep, context.models, single_step=single, save_dir=runs_dir(context)):
                rendered = list(views(current, show, selections, session_id, animate=True))
                if rendered[0] == last_board:
                    rendered[0] = gr.skip()
                else:
                    last_board = rendered[0]
                yield tuple(rendered)
        except ValueError as exc:
            gr.Warning(str(exc))
            yield views(ep, show, selections, session_id)

    def one_step(ep, show, session_id):
        yield from play(ep, show, session_id, True)

    def command(ep, kind):
        try:
            if kind == "pause":
                ep.request_pause()
            elif kind == "stop":
                ep.request_stop(runs_dir(context))
            else:
                ep.request_interruption()
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        gr.Info({"pause": "Will pause after the current response.", "stop": "Stopping; partial actions will not execute.", "interrupt": "Interruption queued for the next response."}[kind])
        return status(ep)

    def inspect(ep, show, i, session_id):
        if ep.busy:
            raise gr.Error("Pause the episode before selecting a response to replay.")
        return views(ep, show, selections, session_id, int(i if i is not None else -1))

    def export(ep):
        return export_run(ep, runs_dir(context))

    def load(path, ep, show, session_id):
        if ep.busy:
            raise gr.Error("Pause or stop this episode before loading a replay.")
        if not path:
            return (gr.skip(), *[gr.skip() for _ in outputs])
        try:
            if Path(path).stat().st_size > 50_000_000:
                raise ValueError("Run files must be smaller than 50 MB.")
            replay = from_payload(json.loads(Path(path).read_text()))
            rendered = views(replay, show, selections, session_id)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            raise gr.Error(f"Could not load run: {exc}") from exc
        return (replay, *rendered)

    def select_token(session_id, metrics, evt: gr.SelectData):
        return selections.inspect(session_id, metrics, evt)

    prepare.click(prepare_episode, [episode, reveal, selection_session, *controls], [episode, *outputs, download], concurrency_id="maze", show_progress="hidden")
    run.click(play, [episode, reveal, selection_session], outputs, concurrency_id="maze", show_progress="hidden")
    step.click(one_step, [episode, reveal, selection_session], outputs, concurrency_id="maze", show_progress="hidden")
    pause.click(lambda ep: command(ep, "pause"), episode, state_text, queue=False)
    stop.click(lambda ep: command(ep, "stop"), episode, state_text, queue=False)
    interrupt.click(lambda ep: command(ep, "interrupt"), episode, state_text, queue=False)
    passage.input(lambda name: "" if name == "None" else PASSAGES.get(name, ""), passage, text, queue=False)
    goal_mode.input(lambda mode: (gr.update(visible=mode == "hint"), 0 if mode != "coordinates" else gr.skip()),
                    goal_mode, [goal_hint, supplied], queue=False)
    reveal.input(lambda ep, show, i: board(ep, None if ep.busy else int(i if i is not None else -1), show), [episode, reveal, turn_picker], maze_board, queue=False)
    turn_picker.input(inspect, [episode, reveal, turn_picker, selection_session], outputs, show_progress="hidden")
    strip.select(select_token, [selection_session, metrics_state], [detail, alternatives], queue=False, show_progress="hidden")
    save.click(export, episode, download, show_progress="hidden")
    upload.upload(load, [upload, episode, reveal, selection_session], [episode, *outputs], show_progress="hidden")
    context.navigation.open_models(models)


def build_page(context):
    with gr.Column(elem_id="maze-page"):
        _build_page(context)
