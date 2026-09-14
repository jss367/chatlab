"""Native maze workbench: real model episodes and inspectable saved replays."""
from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

import gradio as gr

from .maze import GOAL_MODES, PASSAGES, SYSTEM, TOOLS, default_instruction, generate
from .runner import RECOVERY_DEFAULTS, TERMINAL, Episode, context_messages, fork_token_edit, from_payload, stream_episode
from .trials import prepare_trial, read_trials
from extension_api import TokenInspector

TOKENS = TokenInspector()
STALE_TOKEN = "Select a token in the current response again."

CSS = """
#maze-page {padding:20px 24px; min-width:0 !important; min-height:0; height:100%; box-sizing:border-box; flex-wrap:nowrap; gap:16px; overflow:hidden;}
#maze-page h1 {letter-spacing:-.035em; font-size:26px; margin:0;}
#maze-page h2 {font-size:16px; letter-spacing:-.02em; margin:0;}
#maze-page .column {flex-wrap:nowrap;}
#maze-page .column > * {flex-shrink:0;}
#maze-workspace {flex:1 1 0 !important; min-height:0; flex-wrap:nowrap; gap:20px;}
#maze-scenario, #maze-inspector {flex:0 0 auto !important; min-width:240px !important; width:26%; max-width:30%; height:100%; overflow:auto; resize:horizontal; padding:0 12px 16px 0; scrollbar-width:thin; overscroll-behavior:contain;}
#maze-inspector {width:32%; min-width:280px !important; max-width:34%; padding:0 0 16px 12px; border-left:1px solid var(--border-color-primary);}
#maze-center {flex:1 1 0 !important; min-width:280px !important; min-height:0; height:100%; gap:12px; overflow:auto; scrollbar-width:thin;}
#maze-board {flex:1 1 0 !important; min-height:180px; background:#f6f7fb; border:1px solid #e4e7f0; border-radius:18px; padding:12px; display:flex; flex-direction:column;}
#maze-board .html-container, #maze-board .prose {height:100%; min-height:0; display:flex; flex-direction:column;}
#maze-board svg {width:100%; flex:1 1 0; min-height:0; display:block; margin:auto;}
.maze-legend {display:flex; justify-content:center; gap:8px 12px; flex-wrap:wrap; color:#647084; font:11px system-ui; padding-top:10px; flex-shrink:0;}
#maze-transport {gap:6px; flex-wrap:nowrap;}
#maze-transport button {min-width:0; padding:8px 6px; font-size:12px;}
#maze-transport-status {font-size:12px; min-height:42px;}
#maze-transport-status p {margin:0;}
#maze-status {font-size:12px;}
#maze-tokens {max-height:32vh; min-height:110px; overflow:auto;}
#maze-token-editor {border:1px solid #c7d2fe; border-radius:12px; padding:12px;}
#maze-history {font-size:12px;}
#maze-history td, #maze-history th {font:12px/1.5 system-ui;}
#maze-history td {cursor:pointer;}
#maze-raw textarea, #maze-context textarea {font-family:ui-monospace,monospace; font-size:12px;}
#maze-scenario .form, #maze-inspector .form {min-width:0 !important;}
#maze-scenario .row {gap:8px;}
#maze-scenario .row > * {min-width:100px !important;}
#maze-scenario .block, #maze-inspector .block {min-width:0 !important;}
@media(max-width:1100px) {
  #maze-page {padding:16px 12px;}
  #maze-workspace {gap:12px;}
  #maze-scenario {min-width:210px !important; width:24%; max-width:28%;}
  #maze-inspector {min-width:240px !important; width:30%; max-width:32%;}
  #maze-center {min-width:250px !important;}
}
@media(max-width:850px) {
  #maze-page {overflow:auto;}
  #maze-workspace {flex:none !important; flex-wrap:wrap;}
  #maze-center {order:-1; flex:1 0 100% !important; height:560px;}
  #maze-scenario, #maze-inspector {flex:1 1 280px !important; width:auto; max-width:none; height:560px; resize:none;}
}
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
    # Only the displayed response's own move animates. A response that was
    # rejected or made no call leaves the character where it was, and replaying
    # an earlier turn's hop would show movement that this response never made.
    if animate and accepted and accepted[-1]["source"] == "model" and accepted[-1].get("turn") == index:
        px, py = center(accepted[-1]["before"])
        motion = f'<animateTransform attributeName="transform" type="translate" from="{px} {py}" to="{x} {y}" dur="0.3s" fill="freeze"/>'
    parts.append(f'<g transform="translate({x} {y})">{motion}<circle r="17" fill="#4f46e5" stroke="white" stroke-width="3"/><circle cx="-5" cy="-2" r="2.5" fill="white"/><circle cx="5" cy="-2" r="2.5" fill="white"/><path d="M -5 6 Q 0 10 5 6" stroke="white" fill="none" stroke-width="2"/></g></svg>')
    parts.append('<div class="maze-legend"><span>● Character / model path</span><span>┄ Supplied moves</span><span>★ Destination</span><span style="color:#b77906">○ Interruption</span></div>')
    return "".join(parts)


def scenario_values(ep):
    """The scenario controls in the order `controls` lists them, then the passage
    name, so loading a run can describe the run on screen rather than leaving the
    defaults of an unrelated episode beside it."""
    config, maze = ep.config, ep.maze
    text = config.get("interruption_text", "")
    named = next((name for name, passage in PASSAGES.items() if passage == text), None)
    mode = config["goal_mode"]
    openness = openness_of(ep)
    return (maze.size, maze.seed, len(maze.route()) - 1, gr.skip() if openness is None else openness,
            config.get("supplied_moves", 0), config.get("interrupt_after", 0), text,
            config.get("prefix_tokens", 0), config.get("temperature", .7), config.get("sampling_seed", 0),
            config.get("per_turn_tokens", 1024), config.get("token_budget", 8192), config.get("attempt_budget", 32),
            config["recovery_tokens"], config["recovery_attempts"],
            mode, gr.update(value=config["goal_hint"], visible=mode == "hint"),
            config["system_prompt"], config["instruction"],
            "None" if not text else named or "Custom")


OPENNESS_CHOICES = tuple(round(.35 + .05 * step, 2) for step in range(13))


def openness_of(ep):
    """The probability a run was drawn with, or None when it predates the recorded
    setting and its maze cannot name one. A recovered value is kept on the run, so
    the search runs once and everything reading the run afterwards agrees with it."""
    if "openness" not in ep.config:
        recovered = recovered_openness(ep.maze)
        if recovered is None:
            return None
        ep.config["openness"] = recovered
    return float(ep.config["openness"])


def recovered_openness(maze):
    """Runs predating the recorded setting are searched for the slider value that
    redraws their maze. Every value is tried before the answer is unknown, because
    a small maze deviates from the probability that drew it and elapsed time is no
    evidence about the values not yet reached; the share of cells the maze leaves
    open only orders the search, so a recoverable run usually answers on the first
    try and an exhausted search costs a few seconds once per upload. The redrawn
    maze has to match whole: the same walls placed around a different start or
    destination is a maze the run never used."""
    share = sum(row.count(".") for row in maze.grid) / maze.size ** 2
    for value in sorted(OPENNESS_CHOICES, key=lambda v: abs(v - share)):
        try:
            if generate(maze.size, maze.seed, len(maze.route()) - 1, value) == maze:
                return value
        except ValueError:
            pass
    return None


def edited_prompt(config):
    return (config.get("system_prompt", SYSTEM) != SYSTEM
            or config.get("instruction") != default_instruction(config["goal_mode"]))


def status(ep):
    partial = 0
    if ep.turns and ep.turns[-1]["finish_reason"] is None:
        t = ep.turns[-1]
        partial = max(0, len(t["metrics"]) - t["forced_prefix_tokens"])
    recovery = "Not inserted" if not ep.interrupted else ("Pending" if ep.resumed is None else (f"Returned in {ep.latency} sampled tokens" if ep.resumed else "No return"))
    window = f"{ep.config['recovery_tokens']:,} sampled tokens / {ep.config['recovery_attempts']} attempts"
    return (f"**{'Replay · ' if ep.replay_only else ''}{ep.phase.title()}** · {html.escape(ep.detail)}\n\n"
            f"{ep.maze.size} × {ep.maze.size} · shortest route {len(ep.maze.route())-1} moves · "
            f"{ep.moves-ep.supplied_moves} model moves + {ep.supplied_moves} supplied · "
            f"{ep.sampled_tokens+partial:,} sampled tokens · {ep.tool_attempts} calls\n\n"
            f"**Goal information:** {GOAL_MODES[ep.config['goal_mode']]} · "
            f"**Setup prompt:** {'Edited' if edited_prompt(ep.config) else 'Default'}"
            f"{'' if 'openness' in ep.config else ' · **Open cells:** Unrecorded, so the slider beside this run is not its own'}\n\n"
            f"**Recovery:** {recovery} · **Recovery window:** {window} · "
            f"**Model:** {html.escape(ep.model_id or 'load one on the Models page')}")


def timeline(ep):
    supplied = [e for e in ep.events if e["source"] == "supplied"]
    position = supplied[-1]["after"] if supplied else ep.maze.start
    rows = [["Initial / supplied", str(tuple(position)), "—", f"{len(supplied)} supplied moves" if supplied else "Initial position"]]
    by_turn = {e["turn"]: e for e in ep.events if e["source"] == "model"}
    for index, turn in enumerate(ep.turns):
        event = by_turn.get(index)
        if event:
            position = event["after"]
        rows.append([f"Response {index + 1}", str(tuple(position)),
                     event.get("direction") or "—" if event else "—",
                     ("Accepted" if event["accepted"] else event["error"].replace("_", " ")) if event
                     else ("Generating…" if turn.get("finish_reason") is None else "No move")])
    selected = max(0, min(ep.viewing + 1, len(rows) - 1))
    rows[selected][0] = "▶ " + rows[selected][0]
    return rows


def transport_text(ep):
    index = max(-1, min(ep.viewing, len(ep.turns) - 1))
    position = ep.maze.start
    for event in ep.events:
        if event["accepted"] and event.get("turn", -1) <= index:
            position = event["after"]
    mode = "Generating" if ep.busy else "Replaying" if ep.playing else "Paused" if ep.turns else "Ready"
    if ep.busy and ep.pause_requested:
        mode = "Pausing after response"
    selected = "Initial / supplied" if index < 0 else f"Response {index + 1} of {len(ep.turns)}"
    if ep.replay_only:
        end = "Saved replay"
    elif ep.phase in TERMINAL:
        end = ep.phase.title()
    else:
        end = "Live end · Next generates" if index == len(ep.turns) - 1 else "Play continues at live end"
    queued = " · **Interruption queued**" if ep.interrupt_next and not ep.interrupted else ""
    return f"**{mode}** · {selected} · ({position[0]}, {position[1]}){queued}\n\n{end}"


def transcript(messages):
    """The messages and the move tool as they were recorded, with no model to spell them."""
    parts = [f"[tool schemas]\n{json.dumps(TOOLS, indent=2)}"]
    parts += [f"[{message['role']}]\n{message['content']}" for message in messages]
    return "\n\n".join(parts)


def under_load(recorded, used):
    """Name the load a reading was made under when it is not the one that recorded it."""
    if not recorded or recorded == used:
        return ""
    return (f", under {html.escape(used)} rather than the {html.escape(recorded)} that recorded it, "
            "so a vocabulary that has moved since would read differently here")


def read_through(reading, *arguments):
    """Read the model, or report how it failed to answer.

    A view of a prompt has a worse answer than its best one - the messages as
    recorded - so a template that refuses this history, or IDs a vocabulary
    cannot spell, falls back to that rather than replacing the pane with an
    error about the model it was describing. What went wrong comes back with
    the empty reading, because a refusal is not an absence: a reader sent to
    the Models page by a model already loaded is troubleshooting the wrong
    thing.
    """
    try:
        return (*reading(*arguments), None)
    except Exception as exc:
        return None, None, exc


def unspelled(models, failure):
    """Why a prompt is being shown without a model's own spelling of it."""
    if failure is not None:
        return (f"The loaded model did not render this prompt ({type(failure).__name__}: "
                f"{html.escape(str(failure))})")
    if not models.loaded:
        return "No model is loaded to spell this prompt"
    return "A load landed while this prompt was being read"


def context_view(ep, models, index=None):
    """Everything the model was given for the selected response, and how it was spelled.

    The recorded prompt IDs are the run's own answer, so they are read back
    first and decoded whole: the tool schemas, the turn markers and the JSON
    state are in there as the template wrote them. A response that has none -
    the initial prompt, which no response has been asked for yet - is put
    through the loaded model's template instead, which is the same path
    generation takes. With nothing loaded, the messages and the tool schema
    are shown as recorded, which is as close as a run can be read without the
    vocabulary that spelled it.
    """
    index = ep.viewing if index is None else index
    index = max(-1, min(index, len(ep.turns) - 1))
    turn = ep.turns[index] if index >= 0 else {}
    where = "Initial prompt" if index < 0 else f"Response {index + 1}"
    supplied = turn.get("forced_prefix_tokens") or 0
    tail = (f" A supplied prefix of {supplied:,} tokens followed it, shown under **Supplied text & full response**."
            if supplied else "")
    ids = turn.get("prompt_ids")
    failure = None
    if ids:
        text, load_id, failure = read_through(models.decode, ids)
        if text is not None:
            return (f"**{where} · as recorded** · {len(ids):,} prompt tokens, decoded"
                    f"{under_load(turn.get('load_id'), load_id)}.{tail}", text)
    messages = context_messages(ep, index)
    text, load_id, refused = read_through(models.prompt_text, messages, TOOLS)
    if text is not None:
        return (f"**{where} · as the loaded model would be given it** · {len(messages)} messages and the move tool "
                f"through that model's own template{under_load(ep.load_id, load_id)}.{tail}", text)
    return (f"**{where} · as recorded, untemplated** · {unspelled(models, refused or failure)}, so the "
            f"{len(messages)} messages and the move tool are shown as the run recorded them. A template adds its own "
            f"turn markers and writes the tool schemas its own way.{tail}", transcript(messages))


def transport_buttons(ep):
    active = ep.playing or ep.busy
    return gr.update(visible=not active), gr.update(visible=active)


def model_button(ep):
    """Name the model a saved run needs on the button that opens Models.

    A replay was generated elsewhere, so editing its tokens needs that model
    in memory. Naming it on the button, and handing the ID to the Models
    page, turns the refusal that would otherwise follow into one click. A
    live episode runs under whatever is already loaded, so the button keeps
    its plain label and names nothing.
    """
    wanted = ep.model_id if ep.replay_only else None
    return gr.update(value=f"Load {wanted}" if wanted else "Choose / load model"), wanted or ""


def stop_replay(ep):
    # Cooperative cancellation must never close an active model generator:
    # Pause lets it finish its response and persist the accepted move.
    with ep.lock:
        ep.playback_token += 1
        ep.playing = False


def views(ep, reveal, selections, session_id, index=None, animate=False):
    if index is None:
        index = len(ep.turns) - 1
    ep.viewing = index
    t = ep.turns[index] if 0 <= index < len(ep.turns) else {}
    metrics = t.get("metrics", [])
    forced = t.get("forced_prefix_tokens", 0)
    stamped, changed = selections.view(session_id, (ep.run_id, id(ep), index), metrics[forced:])
    origin = "Inside the template's open reasoning block" if t.get("reasoning_prefilled") else "At the beginning of the assistant response"
    note = (f"**Supplied interruption · {forced} tokens** · {origin}.\n\n" if forced else "No supplied interruption in this response.")
    if not forced and t.get("planned_prefix_ids"):
        note = f"**Pending interruption · {len(t['planned_prefix_ids'])} tokens** · Prefix insertion has not been confirmed."
    if t.get("token_edit"):
        note = (f"**{'Retained / edited' if forced else 'Pending retained / edited'} prefix · {forced or len(t.get('planned_prefix_ids', []))} tokens** · "
                "Earlier token IDs and your replacement are supplied as context; only the new continuation counts toward sampled-token limits.")
    return (board(ep, index, reveal, animate), status(ep), TOKENS.strip(metrics[forced:]),
            t.get("text", ""), note, t.get("prefix_text") or t.get("planned_prefix_text", ""), timeline(ep), stamped,
            gr.update(choices=[("Initial / supplied history", -1)] + [(f"Response {i+1}" + (" · token edit" if t.get("token_edit") else " · interruption" if t.get("prefix_ids") else ""), i) for i, t in enumerate(ep.turns)], value=index),
            "Select a model-generated token above." if changed else gr.skip(), [] if changed else gr.skip())


# The note is Markdown, and the names in it come from a file that may have
# been written anywhere. Escaping the HTML leaves `**` and `[…](…)` to be read
# as syntax, which is enough to close the bold span the provenance is written
# in and continue in a voice that looks like the workbench's own.
MARKDOWN = str.maketrans({character: "\\" + character for character in "\\`*_{}[]()#+-.!>|~"})


def as_text(value):
    """A name from a trial file, read as the characters it is."""

    return html.escape(value).translate(MARKDOWN)


def trial_note_text(ep, data=None):
    """What the trials pane says about the episode on screen right now.

    The episode is replaced by several other controls, and a note that still
    named a trial after one of them would have an experimenter running or
    exporting something else in its name. Uploading a collection does not
    replace the episode, so what it says about the run stands.
    """

    parts, trial = [], ep.config.get("trial")
    if trial:
        parts.append(f"**{'Replaying' if ep.replay_only else 'Running'}: {as_text(trial['label'])}**, "
                     f"from {as_text(trial['title'])}.")
    if data:
        count = len(data["trials"])
        parts.append(f"Loaded **{as_text(data['title'])}** · {count} trial{'s' if count != 1 else ''}. "
                     "Select one and click Load trial.")
    elif not trial:
        parts.append("Upload a trial file, choose a trial, then load it. Inspect the maze before playing.")
    else:
        parts.append("New episode starts a separate run from the controls.")
    return " ".join(parts)


def _build_page(context):
    default_config = dict(supplied_moves=3, interrupt_after=3, interruption_text=next(iter(PASSAGES.values())),
                          prefix_tokens=8, temperature=.7, sampling_seed=20260914, per_turn_tokens=1024,
                          token_budget=8192, attempt_budget=32, openness=.7, **RECOVERY_DEFAULTS)
    initial = Episode(generate(), default_config)
    episode = gr.State(initial)
    selections = context.tokens.selections()
    menu = context.tokens.menu("maze-tokens")
    selection_session = gr.State(value=selections.new_session, delete_callback=selections.forget)
    metrics_state = gr.State((None, []))
    edit_selection = gr.State(None)
    # The model a saved run needs, for the button that opens the Models page.
    wanted_model = gr.State("")
    trial_data = gr.State(None)
    gr.Markdown("# Maze workbench")
    with gr.Row(elem_id="maze-workspace"):
        with gr.Column(elem_id="maze-scenario"):
            gr.Markdown("## Scenario")
            gr.Markdown("Settings apply to the next episode. Loading a saved run shows the settings it used.")
            prepare = gr.Button("New episode · apply settings", elem_id="maze-prepare")
            with gr.Accordion("Setup prompt", open=False):
                system_prompt = gr.Textbox(value=SYSTEM, label="System prompt", lines=2, elem_id="maze-system-prompt")
                instruction = gr.Textbox(value=default_instruction("coordinates"), label="Task instruction", lines=6,
                                         elem_id="maze-instruction",
                                         info="Sent verbatim ahead of the JSON state. Changing Goal information rewrites this unless you have edited it.")
            with gr.Row():
                size = gr.Slider(3, 15, value=5, step=1, label="Maze size")
                distance = gr.Number(value=10, precision=0, minimum=1, maximum=224, label="Shortest route length")
            with gr.Row():
                seed = gr.Number(value=20260911, precision=0, label="Maze seed")
                openness = gr.Slider(.35, .95, value=.7, step=.05, label="Open cells")
            goal_mode = gr.Dropdown(choices=[(label, mode) for mode, label in GOAL_MODES.items()],
                                    value="coordinates", label="Goal information", elem_id="maze-goal-mode",
                                    info="What the model knows about the destination.")
            goal_hint = gr.Textbox(label="Goal hint", lines=2, visible=False, elem_id="maze-goal-hint",
                                   info="Write a clue about the destination shown on the board. This text goes to the model verbatim; check it after changing the maze.")
            with gr.Row():
                supplied = gr.Number(value=3, precision=0, minimum=0, maximum=223, label="Supplied starting moves", info="Shortest-route demonstration; 0 for none.")
                after = gr.Number(value=3, precision=0, minimum=0, maximum=255, label="Interrupt after moves", info="Counts supplied moves.")
            passage = gr.Dropdown(["None", *PASSAGES, "Custom"], value=next(iter(PASSAGES)), label="Interruption passage")
            text = gr.Textbox(value=next(iter(PASSAGES.values())), label="Interruption text", lines=3)
            prefix = gr.Number(value=8, precision=0, minimum=0, maximum=1024, label="Supplied token count", info="0 uses all text.")
            with gr.Accordion("Generation limits", open=False):
                temperature = gr.Slider(0, 2, value=.7, step=.05, label="Maze sampling temperature")
                sampling_seed = gr.Number(value=20260914, precision=0, minimum=0, maximum=2147483647, label="Maze sampling seed")
                per_turn = gr.Number(value=1024, precision=0, minimum=1, maximum=8192, label="Tokens per response")
                budget = gr.Number(value=8192, precision=0, minimum=1, maximum=32768, label="Total sampled-token limit")
                attempts = gr.Number(value=32, precision=0, minimum=1, maximum=256, label="Tool-attempt limit")
                recovery_tokens = gr.Number(value=RECOVERY_DEFAULTS["recovery_tokens"], precision=0, minimum=1, maximum=32768,
                                            label="Recovery window · sampled tokens", elem_id="maze-recovery-tokens",
                                            info="After the interruption, a first accepted move has to arrive inside this many sampled tokens.")
                recovery_attempts = gr.Number(value=RECOVERY_DEFAULTS["recovery_attempts"], precision=0, minimum=1, maximum=256,
                                              label="Recovery window · tool attempts", elem_id="maze-recovery-attempts",
                                              info="And inside this many attempted calls.")
            models = gr.Button("Choose / load model", size="sm")
            with gr.Accordion("Experiment trials", open=False):
                trial_upload = gr.File(label="Trial definitions JSON", file_types=[".json"], type="filepath")
                trial_picker = gr.Dropdown(choices=[], label="Trial", interactive=True,
                                           info="Type to filter a long collection.")
                trial_load = gr.Button("Load trial", size="sm", elem_id="maze-load-trial")
                trial_note = gr.Markdown(trial_note_text(initial))
            with gr.Accordion("Saved runs", open=False):
                save = gr.Button("Export run JSON", size="sm")
                download = gr.File(label="Saved run", interactive=False)
                upload = gr.File(label="Load a saved run", file_types=[".json"], type="filepath")
        with gr.Column(elem_id="maze-center"):
            maze_board = gr.HTML(board(initial), elem_id="maze-board")
            transport_status = gr.Markdown(transport_text(initial), elem_id="maze-transport-status")
            with gr.Row(elem_id="maze-transport"):
                back = gr.Button("◀ Previous", size="sm", elem_id="maze-previous")
                toggle = gr.Button("▶ Play", variant="primary", size="sm", elem_id="maze-run")
                pause = gr.Button("Ⅱ Pause", variant="primary", size="sm", visible=False, elem_id="maze-pause")
                forward = gr.Button("Next ▶", size="sm", elem_id="maze-next")
                interrupt = gr.Button("Interrupt", size="sm", elem_id="maze-interrupt")
            turn_picker = gr.Dropdown(choices=[("Initial / supplied history", -1)], value=-1,
                                      label="Selected response", interactive=True)
            with gr.Accordion("Playback & view", open=False):
                pace = gr.Slider(.1, 4, value=1., step=.1, label="Seconds per recorded response")
                reveal = gr.Checkbox(label="Show shortest route (viewer only)", value=False)
                gr.Markdown("Play replays recorded responses, then continues generating in a live episode. Next advances one response. Pause lets a generated response finish.")
                stop = gr.Button("Stop now · end episode", size="sm", elem_id="maze-stop")
        with gr.Column(elem_id="maze-inspector"):
            gr.Markdown("## Emitted tokens")
            strip = gr.HighlightedText(label="Click a token to inspect or edit; right-click to branch on the spot",
                                       color_map=context.tokens.color_map, combine_adjacent=False,
                                       show_legend=True, elem_id="maze-tokens",
                                       elem_classes=menu.strip_classes)
            # The right-click menu carries one token's alternatives out through
            # these and the branch chosen in it back, without the reader
            # crossing the pane to the editor below.
            menu_request, menu_response, menu_action = menu.bridges()
            with gr.Column(visible=False, elem_id="maze-token-editor") as editor:
                detail = gr.Markdown("Select a model-generated token above.")
                replacement = gr.Textbox(label="Replacement text", lines=2)
                candidate = gr.Dropdown(choices=[("Use replacement text", "text")], value="text", label="Replacement token")
                edit_button = gr.Button("Replace token and regenerate", elem_id="maze-edit-token")
                gr.Markdown("Editing creates a new run from this token. The original run is saved.")
                with gr.Accordion("Token probabilities", open=False):
                    alternatives = gr.Dataframe(headers=["Token ID", "Text", "Raw probability"], interactive=False,
                                                label="Click a row to branch this response into that token")
            gr.Markdown("## Movement history\nSelect a row to show its position and response.")
            events = gr.Dataframe(value=timeline(initial), headers=["Response", "Position", "Direction", "Result"],
                                  interactive=False, wrap=True, max_height=260, elem_id="maze-history")
            with gr.Accordion("Supplied text & full response", open=False):
                prefix_note = gr.Markdown("No supplied interruption in this response.")
                prefix_text = gr.Textbox(label="Supplied prefix", interactive=False, lines=2)
                raw = gr.Textbox(label="Full response", interactive=False, lines=6, max_lines=12, elem_id="maze-raw")
            with gr.Accordion("Context sent to the model", open=False) as context_pane:
                context_note = gr.Markdown("Open or refresh this to read the whole prompt behind the selected response.")
                context_refresh = gr.Button("Show the selected response's context", size="sm", elem_id="maze-context-refresh")
                context_body = gr.Textbox(label="Prompt", interactive=False, lines=10, max_lines=24, elem_id="maze-context")
            with gr.Accordion("Run details", open=False):
                state_text = gr.Markdown(status(initial), elem_id="maze-status")
                gr.Markdown("Movement requires a completed, valid move call. Supplied text is separate from generated tokens. Token edits rewind the selected response and regenerate later moves.")
    outputs = [maze_board, state_text, strip, raw, prefix_note, prefix_text, events, metrics_state, turn_picker, detail, alternatives,
               transport_status, toggle, pause, editor]

    def render(ep, show, session_id, index=None, animate=False):
        frame = views(ep, show, selections, session_id, index, animate)
        return (*frame, transport_text(ep), *transport_buttons(ep),
                gr.update(visible=False) if frame[9] != gr.skip() else gr.skip())

    controls = [size, seed, distance, openness, supplied, after, text, prefix, temperature, sampling_seed, per_turn,
                budget, attempts, recovery_tokens, recovery_attempts, goal_mode, goal_hint, system_prompt, instruction]

    def prepare_episode(ep, show, session_id, data, *values):
        if ep.busy:
            raise gr.Error("Stop or pause this episode before starting another.")
        (n, s, d, o, supplied_n, trigger, passage_text, count, temp, sample_seed, per, total, tries,
         window_tokens, window_attempts, mode, hint, system_text, instruction_text) = values
        try:
            new = Episode(generate(n, s, d, o), dict(supplied_moves=int(supplied_n), interrupt_after=int(trigger),
                          interruption_text=passage_text, prefix_tokens=int(count), temperature=float(temp), openness=float(o),
                          sampling_seed=int(sample_seed), per_turn_tokens=int(per), token_budget=int(total), attempt_budget=int(tries),
                          recovery_tokens=int(window_tokens), recovery_attempts=int(window_attempts),
                          goal_mode=mode, goal_hint=hint, system_prompt=system_text, instruction=instruction_text))
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        stop_replay(ep)
        return (new, *render(new, show, session_id), trial_note_text(new, data), None, *model_button(new))

    def load_trial_file(path, ep, loaded):
        if not path:
            return None, gr.update(choices=[], value=None), trial_note_text(ep)
        try:
            data = read_trials(path)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # A file that will not load replaces nothing, so the picker keeps
            # offering the collection it was offering. The widget names the
            # file that failed, so the error names the one that is still there.
            kept = f" Still loaded: {as_text(loaded['title'])}." if loaded else ""
            raise gr.Error(f"Could not load trials: {exc}{kept}") from exc
        return (data, gr.update(choices=[(t["label"], t["id"]) for t in data["trials"]], value=data["trials"][0]["id"]),
                trial_note_text(ep, data))

    def load_trial(data, trial_id, ep, show, session_id):
        try:
            if data is None:
                raise ValueError("Load a trial definitions file first.")
            new = prepare_trial(data, trial_id, ep)
        except (ValueError, TypeError, KeyError) as exc:
            raise gr.Error(str(exc)) from exc
        stop_replay(ep)
        # The same description a loaded run gets: the trial is in the episode
        # now, so nothing here has to spell the controls out a second time.
        return (new, *render(new, show, session_id), *scenario_values(new),
                trial_note_text(new), None, None, *model_button(new))

    def play(ep, show, session_id, single=False):
        last_board = None
        ep.reveal_route = show
        try:
            for current in stream_episode(ep, context.models, single_step=single, save_dir=runs_dir(context)):
                rendered = list(render(current, ep.reveal_route, session_id, animate=True))
                if rendered[0] == last_board:
                    rendered[0] = gr.skip()
                else:
                    last_board = rendered[0]
                yield tuple(rendered)
        except ValueError as exc:
            gr.Warning(str(exc))
            yield render(ep, show, session_id)

    def command(ep, kind):
        try:
            if kind == "pause":
                stop_replay(ep)
                ep.request_pause()
            elif kind == "stop":
                stop_replay(ep)
                ep.request_stop(runs_dir(context))
            else:
                ep.request_interruption()
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        gr.Info({"pause": "Pausing after the current response." if ep.busy else "Playback paused.", "stop": "Stopping; partial actions will not execute.", "interrupt": "Interruption queued for the next response."}[kind])
        return status(ep), transport_text(ep), *transport_buttons(ep)

    def inspect(ep, show, i, session_id):
        if ep.busy:
            raise gr.Error("Pause the episode before selecting a response to replay.")
        stop_replay(ep)
        return render(ep, show, session_id, max(-1, min(int(i if i is not None else -1), len(ep.turns) - 1)))

    def viewing(ep):
        # Read from the episode rather than the dropdown: Gradio captures a
        # listener's inputs when the click is queued, so a second click sent
        # before the first reply arrives would carry the same stale response.
        return max(-1, min(ep.viewing, len(ep.turns) - 1))

    def step_back(ep, show, session_id):
        if ep.busy:
            raise gr.Error("Pause the episode before stepping through responses.")
        stop_replay(ep)
        return render(ep, show, session_id, max(-1, viewing(ep) - 1), animate=True)

    def step_forward(ep, show, session_id):
        if ep.busy:
            raise gr.Error("Pause the episode before stepping through responses.")
        stop_replay(ep)
        if viewing(ep) < len(ep.turns) - 1:
            yield render(ep, show, session_id, viewing(ep) + 1, animate=True)
        elif not ep.replay_only and ep.phase not in TERMINAL:
            yield from play(ep, show, session_id, single=True)
        else:
            yield render(ep, show, session_id, viewing(ep))

    def play_back(ep, show, session_id, seconds):
        """One transport owns replay and its optional continuation at the live end."""
        with ep.lock:
            if ep.busy:
                raise gr.Error("Pause the episode before playing it back.")
            ep.playback_token += 1
            token = ep.playback_token
            ep.playing = True
            ep.reveal_route = show
        start = viewing(ep)
        try:
            for index in range(start, len(ep.turns)):
                if index > start:
                    remaining = max(.1, float(seconds))
                    while remaining > 0:
                        if ep.playback_token != token:
                            return
                        delay = min(.05, remaining)
                        time.sleep(delay)
                        remaining = round(remaining - delay, 6)
                if ep.playback_token != token:
                    return
                yield render(ep, show, session_id, index, animate=True)
            if ep.playback_token == token and not ep.replay_only and ep.phase not in TERMINAL:
                yield from play(ep, show, session_id)
        finally:
            if ep.playback_token == token:
                ep.playing = False
        # A cancelled replay must not overwrite a new selection or episode.
        # Generation still delivers its final paused frame through play().
        if ep.playback_token == token:
            yield render(ep, ep.reveal_route, session_id, viewing(ep))

    def select_history(ep, show, session_id, evt: gr.SelectData):
        row = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        return inspect(ep, show, int(row) - 1, session_id)

    def change_reveal(ep, show, session_id):
        ep.reveal_route = show
        if not ep.busy:
            stop_replay(ep)
        return render(ep, show, session_id, viewing(ep))

    def export(ep):
        return export_run(ep, runs_dir(context))

    def load(path, ep, show, session_id, data):
        if ep.busy:
            raise gr.Error("Pause or stop this episode before loading a replay.")
        if not path:
            return (gr.skip(),) * (len(outputs) + len(controls) + 5)
        try:
            if Path(path).stat().st_size > 50_000_000:
                raise ValueError("Run files must be smaller than 50 MB.")
            replay = from_payload(json.loads(Path(path).read_text()))
            # Before the first frame: recovering the open-cell probability writes
            # it onto the run, and Run details reports whichever way that went.
            values = scenario_values(replay)
            rendered = render(replay, show, session_id)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            raise gr.Error(f"Could not load run: {exc}") from exc
        stop_replay(ep)
        return (replay, *rendered, *values, trial_note_text(replay, data), *model_button(replay))

    def select_token(ep, session_id, metrics, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        try:
            view_id, index, metric = selections.resolve(session_id, metrics, index)
        except ValueError:
            return (gr.skip(),) * 9
        if view_id[:2] != (ep.run_id, id(ep)):
            return (gr.skip(),) * 9
        stop_replay(ep)
        if ep.busy:
            ep.request_pause()
        detail_text, rows = selections.inspect(session_id, metrics, evt)
        if detail_text == gr.skip():
            # A replay or generation frame landed between resolving this click
            # and describing it, so the strip now shows other tokens. Opening
            # the editor here would leave the measurements and the probability
            # table of a token nobody picked, or empty ones.
            return (*(gr.skip(),) * 5, gr.update(visible=False),
                    transport_text(ep), *transport_buttons(ep))
        choices = [("Use replacement text", "text")] + [
            (f"{c['text']!r} · token {c['token_id']}", str(c["token_id"]))
            for c in metric.get("top_candidates", [])]
        return (detail_text, rows,
                dict(view_id=view_id, stamp=metrics[0], index=index), metric.get("text", ""),
                gr.update(choices=choices, value="text"), gr.update(visible=True),
                transport_text(ep), *transport_buttons(ep))

    def edit_token(ep, show, session_id, metrics, selected, text_value, candidate_value, data):
        try:
            if selected is None or selected["stamp"] != metrics[0]:
                raise ValueError(STALE_TOKEN)
            view_id, index, _ = selections.resolve(session_id, metrics, selected["index"])
            if view_id != selected["view_id"] or view_id[:2] != (ep.run_id, id(ep)):
                raise ValueError(STALE_TOKEN)
            turn_index = view_id[2]
            token_index = ep.turns[turn_index]["forced_prefix_tokens"] + index
            with context.models.open_session() as manager:
                new = fork_token_edit(ep, turn_index, token_index, text_value, manager,
                                      candidate_id=None if candidate_value == "text" else int(candidate_value))
            if not ep.replay_only:
                # A saved run's snapshot must never overwrite a newer archive
                # holding the same run_id. The uploaded file is the parent copy.
                ep.save(runs_dir(context))
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        stop_replay(ep)
        # The fork runs under the weights in memory now, so the button stops
        # naming the uploaded run's model.
        buttons, note = model_button(new), trial_note_text(new, data)
        yield (new, *render(new, show, session_id), None, None, *buttons, note)
        for frame in play(new, show, session_id, single=True):
            yield (new, *frame, None, None, *buttons, note)

    def show_context(ep):
        # Read on request rather than with every frame: a prompt is thousands
        # of tokens, and decoding and resending it beside each generated token
        # would cost more than the response it belongs to. The note names the
        # response it read, so an open pane left behind by a later selection
        # says which one it is showing.
        return context_view(ep, context.models)

    context_refresh.click(show_context, episode, [context_note, context_body], show_progress="hidden")
    context_pane.expand(show_context, episode, [context_note, context_body], show_progress="hidden")

    def branch_alternative(ep, show, session_id, metrics, selected, data, evt: gr.SelectData):
        """One click in the probabilities table branches into that alternative.

        The row is read against the token the editor is open on, so a table
        left over from an earlier selection cannot fork on a token nobody
        picked. Everything after that is the button's path, with the clicked
        alternative standing in for the dropdown.
        """
        row = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        try:
            if selected is None or not metrics or selected["stamp"] != metrics[0]:
                raise ValueError(STALE_TOKEN)
            _view_id, _index, metric = selections.resolve(session_id, metrics, selected["index"])
            candidate = metric.get("top_candidates", [])[row]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise gr.Error(STALE_TOKEN) from exc
        yield from edit_token(ep, show, session_id, metrics, selected, "", str(candidate["token_id"]), data)

    def offer_menu(ep, session_id, metrics, request_id, evt: gr.SelectData):
        """Answer one right-click with the alternatives recorded for that token."""
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        try:
            view_id, index, metric = selections.resolve(session_id, metrics, index)
        except ValueError:
            return menu.refuse(request_id, STALE_TOKEN)
        if view_id[:2] != (ep.run_id, id(ep)):
            return menu.refuse(request_id, STALE_TOKEN)
        return menu.offer(
            request_id, dict(view_id=list(view_id), stamp=metrics[0], index=index),
            text=metric.get("text", ""), candidates=metric.get("top_candidates", []),
            verb="Branch this run at", label="Your own replacement text",
            submit="Replace token and regenerate")

    def edit_from_menu(ep, show, session_id, metrics, action, data):
        """Apply a branch chosen in the menu through the editor's own path."""
        try:
            chosen = json.loads(action)
            named = chosen["selection"]
            selection = dict(view_id=tuple(named["view_id"]), stamp=named["stamp"], index=named["index"])
            kind = chosen["kind"]
            text_value = chosen["text"] if kind == "text" else ""
        except (KeyError, TypeError, ValueError) as exc:
            raise gr.Error(STALE_TOKEN) from exc
        if kind == "text":
            if not isinstance(text_value, str):
                raise gr.Error(STALE_TOKEN)
            candidate_value = "text"
        elif kind == "candidate":
            # The alternative is named by its place in the menu that was
            # offered, so its token ID is read from the metric here rather than
            # taken from the browser. fork_token_edit checks it again.
            try:
                _, _, metric = selections.resolve(session_id, metrics, selection["index"])
            except ValueError as exc:
                raise gr.Error(STALE_TOKEN) from exc
            candidates = metric.get("top_candidates", [])
            index = chosen.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise gr.Error("Choose an alternative for the selected token.")
            candidate_value = str(candidates[index]["token_id"])
        else:
            raise gr.Error(STALE_TOKEN)
        yield from edit_token(ep, show, session_id, metrics, selection, text_value, candidate_value, data)

    # Replay is per browser; the model service arbitrates generation globally.
    # Never use Gradio cancels here: it closes generators and would turn Pause
    # or an inspection click into a terminal stop with a partial response.
    toggle.click(play_back, [episode, reveal, selection_session, pace], outputs,
                 show_progress="hidden", concurrency_limit=None, trigger_mode="multiple")
    prepare.click(prepare_episode, [episode, reveal, selection_session, trial_data, *controls],
                  [episode, *outputs, trial_note, download, models, wanted_model],
                  concurrency_id="maze-view", show_progress="hidden")
    # On the view's own queue, so a Load trial click cannot run between the
    # upload arriving and the picker it fills, preparing a trial from the
    # collection being replaced.
    # Clearing the widget is its own event, and it means the collection is
    # gone: the same handler reads the empty path and empties the picker with
    # it, so nothing is left to load from a file no longer chosen.
    for event in (trial_upload.upload, trial_upload.clear):
        event(load_trial_file, [trial_upload, episode, trial_data], [trial_data, trial_picker, trial_note],
              concurrency_id="maze-view", show_progress="hidden")
    trial_load.click(load_trial, [trial_data, trial_picker, episode, reveal, selection_session],
                     [episode, *outputs, *controls, passage, trial_note, edit_selection, download,
                      models, wanted_model],
                     concurrency_id="maze-view", show_progress="hidden")
    back.click(step_back, [episode, reveal, selection_session], outputs, show_progress="hidden", concurrency_id="maze-view")
    forward.click(step_forward, [episode, reveal, selection_session], outputs, show_progress="hidden", concurrency_id="maze-view")
    command_outputs = [state_text, transport_status, toggle, pause]
    pause.click(lambda ep: command(ep, "pause"), episode, command_outputs, queue=False)
    stop.click(lambda ep: command(ep, "stop"), episode, command_outputs, queue=False)
    interrupt.click(lambda ep: command(ep, "interrupt"), episode, command_outputs, queue=False)
    passage.input(lambda name: "" if name == "None" else PASSAGES.get(name, ""), passage, text, queue=False)
    def change_goal_mode(mode, wording):
        # A mode's stock instruction describes that mode, so switching rewrites
        # it. Wording you have typed yourself is never overwritten.
        stock = wording in {default_instruction(m) for m in GOAL_MODES}
        return (gr.update(visible=mode == "hint"), 0 if mode != "coordinates" else gr.skip(),
                default_instruction(mode) if stock else gr.skip())

    goal_mode.input(change_goal_mode, [goal_mode, instruction], [goal_hint, supplied, instruction], queue=False)
    reveal.input(change_reveal, [episode, reveal, selection_session], outputs, queue=False)
    turn_picker.input(inspect, [episode, reveal, turn_picker, selection_session], outputs,
                      show_progress="hidden", concurrency_id="maze-view")
    events.select(select_history, [episode, reveal, selection_session], outputs,
                  show_progress="hidden", concurrency_id="maze-view")
    strip.select(select_token, [episode, selection_session, metrics_state],
                 [detail, alternatives, edit_selection, replacement, candidate, editor, transport_status, toggle, pause],
                 queue=False, show_progress="hidden")
    edit_outputs = [episode, *outputs, edit_selection, download, models, wanted_model, trial_note]
    edit_button.click(edit_token,
                      [episode, reveal, selection_session, metrics_state, edit_selection, replacement, candidate,
                       trial_data],
                      edit_outputs, concurrency_id="maze-view", show_progress="hidden")
    alternatives.select(branch_alternative,
                        [episode, reveal, selection_session, metrics_state, edit_selection, trial_data],
                        edit_outputs, concurrency_id="maze-view", show_progress="hidden")
    strip.select(offer_menu, [episode, selection_session, metrics_state, menu_request], menu_response,
                 queue=False, show_progress="hidden")
    # The menu carries the token it was opened on, so the branch it sends back
    # does not depend on which click Gradio snapshotted for this listener.
    menu_action.input(edit_from_menu, [episode, reveal, selection_session, metrics_state, menu_action, trial_data],
                      edit_outputs, concurrency_id="maze-view", show_progress="hidden")
    save.click(export, episode, download, show_progress="hidden")
    upload.upload(load, [upload, episode, reveal, selection_session, trial_data],
                  [episode, *outputs, *controls, passage, trial_note, models, wanted_model],
                  concurrency_id="maze-view", show_progress="hidden")
    context.navigation.open_models(models, wanted_model)


def build_page(context):
    with gr.Column(elem_id="maze-page"):
        _build_page(context)
