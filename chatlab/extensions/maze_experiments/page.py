"""Native maze workbench: real model episodes and inspectable saved replays."""
from __future__ import annotations

import html
import json
import logging
import os
import time
from pathlib import Path

import gradio as gr

from .dynamic_maze import ChangingMaze, changing, maze_at_turn
from .inserts import CHANNELS
from .maze import DIRECTIONS, GOAL_MODES, PASSAGES, SYSTEM, TOOLS, default_instruction, generate
from .batch import BatchControl, cut_short, downloads, run_trials
from .runner import (RECOVERY_DEFAULTS, TERMINAL, Episode, context_messages, fork_token_edit, from_payload,
                     insert_outcome, stream_episode)
from .trials import prepare_trial, read_trials
from chatlab.extension_api import TokenInspector, icon_classes, read_steering_vector

TOKENS = TokenInspector()
STALE_TOKEN = "Select a token in the current response again."

# The workbench was a blank in ChatLab.log: a saved run that appeared to load
# and then do nothing left no record of whether the file had even reached the
# extension. Written here is one line for each action that changes which run
# is on screen, one for each refusal - including the ones that only ever
# reached the reader as a toast - and none for a token. A response is
# thousands of them, and this record is read months later beside the load and
# memory lines it has to explain.
logger = logging.getLogger(__name__)

CSS = """
#maze-page {padding:20px 24px; min-width:0 !important; min-height:0; height:100%; box-sizing:border-box; flex-wrap:nowrap; gap:16px; overflow:hidden;}
#maze-page h1 {letter-spacing:-.035em; font-size:26px; margin:0;}
#maze-page h2 {font-size:16px; letter-spacing:-.02em; margin:0;}
#maze-page .column {flex-wrap:nowrap;}
#maze-page .column > * {flex-shrink:0;}
/* The two modes share the page's height the way the workspace alone had it:
   the tab row keeps its own, and the open panel takes the rest. */
#maze-modes {flex:1 1 0 !important; min-height:0; display:flex; flex-direction:column;}
#maze-modes > .tab-wrapper {flex:none;}
#maze-modes > .tabitem {flex:1 1 0; min-height:0; padding:12px 0 0; border:0;}
#maze-modes > .tabitem > .column {height:100%; min-height:0;}
#maze-workspace, #team-workspace {flex:1 1 0 !important; min-height:0; flex-wrap:nowrap; gap:20px;}
#maze-scenario, #maze-inspector, #team-scenario, #team-inspector {flex:0 0 auto !important; min-width:240px !important; width:26%; max-width:30%; height:100%; overflow:auto; resize:horizontal; padding:0 12px 16px 0; scrollbar-width:thin; overscroll-behavior:contain;}
#maze-inspector, #team-inspector {width:32%; min-width:280px !important; max-width:34%; padding:0 0 16px 12px; border-left:1px solid var(--border-color-primary);}
#maze-center, #team-center {flex:1 1 0 !important; min-width:280px !important; min-height:0; height:100%; gap:12px; overflow:auto; scrollbar-width:thin;}
#maze-board, #team-board {flex:1 1 0 !important; min-height:180px; background:#f6f7fb; border:1px solid #e4e7f0; border-radius:18px; padding:12px; display:flex; flex-direction:column;}
#maze-board .html-container, #maze-board .prose, #team-board .html-container, #team-board .prose {height:100%; min-height:0; display:flex; flex-direction:column;}
#maze-board svg, #team-board svg {width:100%; flex:1 1 0; min-height:0; display:block; margin:auto;}
.maze-legend {display:flex; justify-content:center; gap:8px 12px; flex-wrap:wrap; color:#647084; font:11px system-ui; padding-top:10px; flex-shrink:0;}
#maze-transport, #team-transport {gap:6px; flex-wrap:nowrap;}
#maze-transport button, #team-transport button {min-width:0; padding:8px 6px; font-size:12px;}
#maze-transport-status, #team-transport-status {font-size:12px; min-height:42px;}
#maze-transport-status p, #team-transport-status p {margin:0;}
#maze-status {font-size:12px;}
/* A long conversation between agents scrolls rather than squeezing the board. */
#team-board {min-height:300px;}
#team-mail {max-height:24vh; overflow:auto; font-size:12px;}
#maze-tokens, #team-tokens {max-height:32vh; min-height:110px; overflow:auto;}
#maze-token-editor {border:1px solid #c7d2fe; border-radius:12px; padding:12px;}
#maze-history, #team-history {font-size:12px;}
/* The inspector column is as tall as the window and nothing in it grows, so a
   history of a fixed height left the rest of that height empty between the
   table and the panels below it. The table takes the spare height instead, and
   gives it back when the panels open. What scrolls is the table element, whose
   max-height resolves only if every wrapper Gradio puts between it and the
   block, the virtual-table viewport included, has a height to measure. */
#maze-page #maze-history, #maze-page #team-history {flex:1 1 auto !important; min-height:120px;}
#maze-history .table-container, #maze-history .table-wrap, #maze-history button,
#maze-history svelte-virtual-table-viewport, #maze-history svelte-virtual-table-viewport > div,
#team-history .table-container, #team-history .table-wrap, #team-history button,
#team-history svelte-virtual-table-viewport, #team-history svelte-virtual-table-viewport > div {height:100%; min-height:0;}
#maze-history svelte-virtual-table-viewport, #team-history svelte-virtual-table-viewport {display:block;}
#maze-history table, #team-history table {max-height:100% !important;}
#maze-history td, #maze-history th, #team-history td, #team-history th {font:12px/1.5 system-ui;}
#maze-history td, #team-history td {cursor:pointer;}
#maze-raw textarea, #maze-context textarea, #team-raw textarea, #team-context textarea {font-family:ui-monospace,monospace; font-size:12px;}
#maze-scenario .form, #maze-inspector .form, #team-scenario .form, #team-inspector .form {min-width:0 !important;}
#maze-scenario .row, #team-scenario .row {gap:8px;}
#maze-scenario .row > *, #team-scenario .row > * {min-width:100px !important;}
#maze-scenario .block, #maze-inspector .block, #team-scenario .block, #team-inspector .block {min-width:0 !important;}
@media(max-width:1100px) {
  #maze-page {padding:16px 12px;}
  #maze-workspace, #team-workspace {gap:12px;}
  #maze-scenario, #team-scenario {min-width:210px !important; width:24%; max-width:28%;}
  #maze-inspector, #team-inspector {min-width:240px !important; width:30%; max-width:32%;}
  #maze-center, #team-center {min-width:250px !important;}
}
@media(max-width:850px) {
  #maze-page {overflow:auto;}
  #maze-workspace, #team-workspace {flex:none !important; flex-wrap:wrap;}
  #maze-center, #team-center {order:-1; flex:1 0 100% !important; height:560px;}
  #maze-scenario, #maze-inspector, #team-scenario, #team-inspector {flex:1 1 280px !important; width:auto; max-width:none; height:560px; resize:none;}
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
        except OSError as error:
            logger.warning("Could not write run %s to the archive %s: %s", ep.run_id, directory, error)
            gr.Warning("The run archive could not be written. Providing a temporary download instead.")
    path = str(ep.export())
    logger.info("Exported run %s to %s", ep.run_id, path)
    return path


def board(ep, index=None, reveal=False, animate=False):
    updates = [u for u in ep.config.get("map_updates", ())
               if index is None or u["before_turn"] <= index]
    maze = maze_at_turn(ep.maze, updates, None)
    closed = {tuple(u["closed_cell"]) for u in updates}
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
            fill = "#78350f" if (r, c) in closed else "#27344a" if value == "#" else "#fff"
            parts.append(f'<rect x="{pad+c*cell+2}" y="{pad+r*cell+2}" width="52" height="52" rx="7" fill="{fill}"/>')
    for i in range(size):
        parts.append(f'<text x="{pad+(i+.5)*cell}" y="17" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
        parts.append(f'<text x="12" y="{pad+(i+.5)*cell+4}" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
    if reveal:
        # A changing map is drawn from where the character stands, because a
        # closure behind it can leave the route it began on no longer the one
        # in front of it.
        route = maze.route(position) if isinstance(maze, ChangingMaze) else maze.route()
        parts.append(f'<polyline points="{points(route)}" fill="none" stroke="#b4bdcc" stroke-width="4" stroke-dasharray="3 8"/>')
    for e in accepted:
        supplied = e["source"] == "supplied"
        dash = 'stroke-dasharray="5 6"' if supplied else ''
        parts.append(f'<polyline points="{points([e["before"],e["after"]])}" fill="none" stroke="{"#94a3b8" if supplied else "#6366f1"}" stroke-width="6" stroke-linecap="round" {dash}/>')
    x, y = center(maze.start)
    parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#64748b" font-size="14" font-weight="700">S</text>')
    x, y = center(maze.goal)
    parts.append(f'<circle cx="{x}" cy="{y}" r="17" fill="#d1fae5"/><text x="{x}" y="{y+7}" text-anchor="middle" font-size="23" fill="#047857">★</text>')
    waypoint, steer_cell = ep.config.get("waypoint"), (ep.config.get("steer_when") or {}).get("cell")
    if steer_cell is not None:
        x, y = center(steer_cell)
        parts.append(f'<rect x="{x-24}" y="{y-24}" width="48" height="48" rx="9" fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="5 4"/>')
    if waypoint is not None:
        x, y = center(waypoint)
        parts.append(f'<text x="{x}" y="{y+8}" text-anchor="middle" font-size="22" fill="#0f766e">⚑</text>')
    if ep.interrupted and (index is None or index >= ep.intervention_turn):
        x, y = center(ep.turns[ep.intervention_turn]["position_before"])
        parts.append(f'<circle cx="{x}" cy="{y}" r="23" stroke="#f59e0b" stroke-width="3" fill="none"/>')
    start = ep.steer_turn
    if start is not None and (index is None or index >= start):
        x, y = center(ep.turns[start]["position_before"])
        parts.append(f'<circle cx="{x}" cy="{y}" r="27" stroke="#7c3aed" stroke-width="3" fill="none"/>')
    # Shown from the response that read the message, as the interruption's
    # ring is, at the cell the character stood on when it landed.
    for insert in ep.config.get("context_inserts", ()):
        if index is not None and index < insert["before_turn"]:
            continue
        x, y = center(insert["position"])
        parts.append(f'<circle cx="{x}" cy="{y}" r="21" stroke="#db2777" stroke-width="3" stroke-dasharray="4 3" fill="none"/>')
        if insert.get("advised_direction") in DIRECTIONS:
            dr, dc = DIRECTIONS[insert["advised_direction"]]
            tip = (x + dc * 38, y + dr * 38)
            # Haloed in white, because advice pointing back along the path
            # would otherwise sit on the path's own line.
            line = f'x1="{x + dc * 21}" y1="{y + dr * 21}" x2="{tip[0] - dc * 7}" y2="{tip[1] - dr * 7}" stroke-linecap="round"'
            parts.append(f'<line {line} stroke="#fff" stroke-width="7"/><line {line} stroke="#db2777" stroke-width="3"/>')
            # The head as a triangle of its own, so the board needs no marker
            # definition whose id another board on the page could share.
            base = (tip[0] - dc * 10, tip[1] - dr * 10)
            corners = [tip, (base[0] + dr * 6, base[1] + dc * 6), (base[0] - dr * 6, base[1] - dc * 6)]
            parts.append(f'<polygon points="{" ".join(f"{px},{py}" for px, py in corners)}" fill="#db2777" '
                         'stroke="#fff" stroke-width="1.5"/>')
    x, y = center(position)
    motion = ""
    # Only the displayed response's own move animates. A response that was
    # rejected or made no call leaves the character where it was, and replaying
    # an earlier turn's hop would show movement that this response never made.
    if animate and accepted and accepted[-1]["source"] == "model" and accepted[-1].get("turn") == index:
        px, py = center(accepted[-1]["before"])
        motion = f'<animateTransform attributeName="transform" type="translate" from="{px} {py}" to="{x} {y}" dur="0.3s" fill="freeze"/>'
    parts.append(f'<g transform="translate({x} {y})">{motion}<circle r="17" fill="#4f46e5" stroke="white" stroke-width="3"/><circle cx="-5" cy="-2" r="2.5" fill="white"/><circle cx="5" cy="-2" r="2.5" fill="white"/><path d="M -5 6 Q 0 10 5 6" stroke="white" fill="none" stroke-width="2"/></g></svg>')
    legend = ['<span>● Character / model path</span>', '<span>┄ Supplied moves</span>', '<span>★ Destination</span>',
              '<span style="color:#b77906">○ Interruption</span>']
    if waypoint is not None:
        legend.append('<span style="color:#0f766e">⚑ Waypoint</span>')
    if ep.config.get("steering") is not None:
        legend.append('<span style="color:#7c3aed">○ Steering started</span>')
    if ep.map_changes:
        legend.append('<span style="color:#78350f">▪ Closed during the run</span>')
    if ep.config.get("context_inserts"):
        legend.append('<span style="color:#db2777">◌ Inserted message · → advised direction</span>')
    parts.append('<div class="maze-legend">' + "".join(legend) + '</div>')
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
            config["system_prompt"], config["instruction"], ep.map_changes,
            "None" if not text else named or "Custom")


STEER_MODES = {"off": "Off", "cell": "When the character reaches a cell", "moves": "After accepted moves"}


def cell_text(cell):
    return "" if cell is None else f"{cell[0]}, {cell[1]}"


def parse_cell(text, label):
    """A cell typed as "row, column", or None for a blank box."""
    text = (text or "").strip().strip("()[]")
    if not text:
        return None
    parts = [part for part in text.replace(",", " ").split() if part]
    try:
        cell = [int(part) for part in parts]
    except ValueError:
        cell = []
    if len(cell) != 2:
        raise ValueError(f"Write the {label} as a row and a column, such as 3, 3.")
    return cell


def checkpoint_values(ep):
    """The waypoint and steering controls in the order `checkpoint_controls`
    lists them, then the vector note, so a loaded run replaces those too. A
    vector carried switched off fills Steer as Off, because steering_config
    switches on any vector it is handed a mode for, and a new episode from
    these controls would then steer where the run it was filled from did not."""
    config = ep.config
    vector, when = config.get("steering"), config.get("steer_when") or {}
    off = vector is None or not vector.get("enabled", True)
    mode = "off" if off else "cell" if "cell" in when else "moves"
    return (cell_text(config.get("waypoint")), vector,
            vector["strength"] if vector else 1.0, vector["layer"] if vector else 0,
            mode, cell_text(when.get("cell")), when.get("moves", 3), config.get("steer_responses", 1),
            vector_note(vector))


def vector_note(vector):
    if vector is None:
        return "No steering vector imported. Download one from **Chat → Conversation tools → Steering vector**."
    return (f"**Vector:** {html.escape(vector['model_id'])} · {len(vector['vector']):,} entries · "
            f"layer {vector['layer']} · strength {vector['strength']:g}")


def steering_config(vector, strength, layer, mode, cell, moves, responses, waypoint):
    """The steering part of a new episode's config, or nothing when steering is off."""
    if mode == "off":
        return {}
    if vector is None:
        raise ValueError("Import a steering vector, or set Steer to Off.")
    if mode == "cell":
        cell = parse_cell(cell, "steering cell") or waypoint
        if cell is None:
            raise ValueError("Name the cell steering starts at, or set a waypoint for it to start at.")
        when = {"cell": cell}
    else:
        when = {"moves": int(moves)}
    return dict(steering=dict(vector, strength=float(strength), layer=int(layer), enabled=True),
                steer_when=when, steer_responses=int(responses))


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
            f"**Model:** {html.escape(ep.model_id or 'load one on the Models page')}"
            f"{checkpoint_line(ep)}{map_line(ep)}{insert_line(ep)}")


def checkpoint_line(ep):
    """Whether the run passed its waypoint, and what steering did, for a run that has either."""
    parts = []
    waypoint = ep.config.get("waypoint")
    if waypoint is not None:
        turn = ep.waypoint_turn
        reached = ("reached by a supplied move" if turn == -1 else f"reached in response {turn + 1}" if turn is not None
                   else "missed" if ep.phase in TERMINAL else "not reached yet")
        parts.append(f"**Waypoint** ({waypoint[0]}, {waypoint[1]}): {reached}")
    vector = ep.config.get("steering")
    if vector is not None:
        when, count = ep.config["steer_when"], ep.config["steer_responses"]
        trigger = (f"at ({when['cell'][0]}, {when['cell'][1]})" if "cell" in when
                   else f"after {when['moves']} accepted move{'' if when['moves'] == 1 else 's'}")
        span = "to the end of the run" if count == 0 else f"for {count} response{'' if count == 1 else 's'}"
        steered = {i for i, turn in enumerate(ep.turns) if turn.get("steered")}
        if not steered:
            state = "never started" if ep.phase in TERMINAL else "not started yet"
        else:
            moves = [e for e in ep.events if e["source"] == "model" and e["turn"] in steered and e["accepted"]]
            state = (f"started in response {min(steered) + 1} · {len(steered)} steered · "
                     f"{len(moves)} accepted move{'' if len(moves) == 1 else 's'} under steering, "
                     f"{sum(e['progress'] for e in moves)} toward the destination")
        parts.append(f"**Steering:** layer {vector['layer']}, strength {vector['strength']:g}, {trigger}, {span} · {state}")
    return "".join(f"\n\n{part}" for part in parts)


def map_line(ep):
    """How this run's map changed, said only by a run whose map could change.

    A dropped closure is reported here because it is the one thing about a
    changing map that leaves no mark on the board: the run went on under a map
    the reader asked to change and which did not change.
    """
    if not ep.map_changes:
        return ""
    closures = len(ep.config.get("map_updates", ()))
    dropped = len(ep.dropped_closures)
    return (f"\n\n**Map:** Changing · {closures} cell{'' if closures == 1 else 's'} closed"
            + (f" · {dropped} closure{'' if dropped == 1 else 's'} dropped, listed in the run JSON" if dropped else ""))


FOLLOWED = {"followed": "followed the advice", "against": "went against it", "other": "went another way"}


def insert_line(ep):
    """Each inserted message, the advice it gave, and what the model did next, for a run that has any."""
    parts = []
    for insert in ep.config.get("context_inserts", ()):
        outcome = insert_outcome(ep, insert)
        sender = f" from {as_text(insert['sender'])}" if insert.get("sender") else ""
        if outcome["advised"] is None:
            advice = "no advised direction"
        else:
            advice = (f"advised {outcome['advised']}, "
                      + ("an open step" if outcome["legal"] else "into a wall")
                      + (", on a shortest route" if outcome["shortest"] else ", on no shortest route"))
        move = outcome["move"]
        if move is None:
            then = "no accepted model move after it" + ("" if ep.phase in TERMINAL else " yet")
        else:
            then = (f"first model move after it: {move['direction']} in response {move['turn'] + 1}"
                    + (f", which {FOLLOWED[outcome['followed']]}" if outcome["followed"] else ""))
        parts.append(f"**Inserted before response {insert['before_turn'] + 1}:** "
                     f"{CHANNELS[insert['channel']]}{sender} · {advice} · {then}")
    return "".join(f"\n\n{part}" for part in parts)


def history_rows(ep):
    """The movement history's rows, each with the response selecting it shows.

    An inserted message has a row of its own between the two responses it
    separates, and selecting it shows the response that read it.
    """
    supplied = [e for e in ep.events if e["source"] == "supplied"]
    position = supplied[-1]["after"] if supplied else ep.maze.start
    rows = [(-1, ["Initial / supplied", str(tuple(position)), "—", f"{len(supplied)} supplied moves" if supplied else "Initial position"])]
    by_turn = {e["turn"]: e for e in ep.events if e["source"] == "model"}
    inserts = {insert["before_turn"]: insert for insert in ep.config.get("context_inserts", ())}
    for index, turn in enumerate(ep.turns):
        insert = inserts.get(index)
        if insert:
            sender = f" from {insert['sender']}" if insert.get("sender") else ""
            rows.append((index, [f"Inserted before response {index + 1}", str(tuple(insert["position"])),
                                 f"advised {insert['advised_direction']}" if insert.get("advised_direction") else "—",
                                 f"{CHANNELS[insert['channel']]}{sender}: {insert['text']}"]))
        event = by_turn.get(index)
        if event:
            position = event["after"]
        result = (("Accepted" if event["accepted"] else event["error"].replace("_", " ")) if event
                  else ("Generating…" if turn.get("finish_reason") is None else "No move"))
        rows.append((index, [f"Response {index + 1}", str(tuple(position)),
                             event.get("direction") or "—" if event else "—",
                             result + (" · steered" if turn.get("steered") else "")]))
    return rows


def timeline(ep):
    rows = history_rows(ep)
    viewing = max(-1, min(ep.viewing, len(ep.turns) - 1))
    # The response's own row, never the message row sharing its index.
    selected = max(i for i, (index, row) in enumerate(rows) if index == viewing and not row[0].startswith("Inserted"))
    rows = [list(row) for _, row in rows]
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
    if ep.close_next:
        queued += f" · **Closing ({ep.close_next[0]}, {ep.close_next[1]})**"
    if ep.insert_next:
        queued += f" · **{CHANNELS[ep.insert_next['channel']]} queued**"
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


def model_of(load_id):
    """The model a load stamp names, a stamp being a model ID and which load it is.

    The stamp as a whole cannot say whether a reading was made by the run's
    own model: two sessions of one model disagree exactly as loudly as two
    different models do, and a run carrying a stamp from somewhere else
    disagrees with every load here. The ID inside it can.
    """
    return (load_id or "").rsplit("#", 1)[0]


def swapped_model(recorded_model, used_model):
    """``used_model``, when it is not the model that recorded the run.

    Empty when they are the same model, or when either side names nothing.
    """
    return used_model if used_model and recorded_model and used_model != recorded_model else ""


def recording_model(ep, turn):
    """Which model recorded this response, the run answering only where it does not.

    A response records the model that produced it, and an episode continues
    under whatever is loaded when it is continued: generating under one
    model, loading another and pressing Next leaves the run named after the
    first and the new response after the second, no file having been edited.
    A response's IDs are answerable to the model that wrote them, so the run's
    name is the fallback for responses recorded before they carried their own.
    """
    return turn.get("model_id") or ep.model_id


def reads_back(recorded_model, used_model):
    """Whether IDs recorded by one model may be read back by the model loaded now.

    Both have to be named and be the same. A run that never recorded which
    model produced it cannot answer this, and its silence is not permission:
    the numbers spell whatever the vocabulary reading them holds at those
    positions, so an unnamed run would offer any model's reading of them as
    its own record. The fork path can be lenient here because it goes on to
    check each response against the text it recorded; a pane that reads one
    prompt has no such second witness.
    """
    return bool(recorded_model) and recorded_model == used_model


def recorded_prompt(ep, turn, models):
    """A response's recorded prompt IDs decoded, or None where they may not be read here.

    The rule the Context pane keeps: IDs are read back only by the model that
    recorded them, named on both sides of the reading.
    """
    ids, recorded = turn.get("prompt_ids"), recording_model(ep, turn)
    if not ids or not reads_back(recorded, models.loaded_model_id()):
        return None
    try:
        text, load_id = models.decode(ids)
    except Exception:
        return None
    return text if text is not None and reads_back(recorded, model_of(load_id)) else None


def prompt_reading(ep, turn, context, models):
    """A response's recorded prompt beside its context put through the same model's template.

    None when the recorded prompt may not be read here. The templated reading
    is None where the template refuses the messages or a load moved between
    the two readings, so the pair is never spelled by two models.
    """
    recorded = recorded_prompt(ep, turn, models)
    if recorded is None:
        return None
    try:
        templated, load_id = models.prompt_text(context, TOOLS)
    except Exception:
        templated, load_id = None, None
    return recorded, templated if reads_back(recording_model(ep, turn), model_of(load_id)) else None


def unnamed_model(count):
    """Why a run that does not say which model produced it is not read back.

    Only a file written elsewhere reaches this: an episode takes the loaded
    model's name before its first response, so a run of this application that
    has IDs to read has a name to read them by.
    """
    return (f" The {count:,} prompt tokens it recorded are not decoded here: this run does not record "
            "which model produced them, and an ID means whatever the vocabulary reading it holds at "
            "that position, so any model's reading would be offered as the record.")


def read_by_another(recorded_model, used_model):
    """Name a reading made by a model that did not record the run.

    Said of the model rather than of a load having moved: two vocabularies
    number their pieces independently, so this is another model's answer to
    the same messages and not a revision of the record, which the softer
    wording about a vocabulary having moved would leave a reader guessing at.
    """
    return (f", under {html.escape(used_model)} rather than the "
            f"{html.escape(recorded_model)} that recorded this run")


def spelled_by_another(count, recorded_model, used_model):
    """Say that a run's recorded prompt IDs were left undecoded, and name both models.

    A token ID is a position in one vocabulary, so the same numbers read under
    another model spell whatever that model keeps at those positions: text
    fluent enough to be read as the prompt and related to it in nothing. Two
    vocabularies also overlap in size, so nothing need raise - a Llama-3 ID is
    a Qwen3 ID - which is why this is decided by the models' names and not by
    whether a decode succeeds. A vocabulary too small for an ID is the same
    mismatch arriving as an exception, and it is said the same way. The model
    that recorded the run is named because loading it is the way to read the
    prompt at all.
    """
    return (f" The {count:,} prompt tokens it recorded are not decoded here: "
            f"{html.escape(used_model)} is loaded and {html.escape(recorded_model)} recorded them, "
            "and an ID means whatever the vocabulary reading it holds at that position. Load "
            f"{html.escape(recorded_model)} to read this prompt as it was recorded.")


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

    Recorded IDs are read back only by the model that recorded them. Another
    model decodes them without complaint, its vocabulary being no smaller,
    and answers with fluent text that is not the prompt, so the model that
    recorded this response against the loaded one decides whether they are
    decoded at all, and a response answering for no model is refused with
    them: see :func:`recording_model` and :func:`reads_back`. A response whose
    model is not loaded falls through to the template, which reads the
    messages themselves.
    """
    index = ep.viewing if index is None else index
    index = max(-1, min(index, len(ep.turns) - 1))
    turn = ep.turns[index] if index >= 0 else {}
    where = "Initial prompt" if index < 0 else f"Response {index + 1}"
    supplied = turn.get("forced_prefix_tokens") or 0
    tail = (f" A supplied prefix of {supplied:,} tokens followed it, shown under **Supplied text & full response**."
            if supplied else "")
    ids = turn.get("prompt_ids")
    recorded = recording_model(ep, turn)
    failure = None
    # Asked before the decode as well as after it, so a foreign run's IDs are
    # not put through a vocabulary that cannot mean them in the first place.
    # The reading below is still what decides, because this answer frames no
    # text and can be a load behind by the time it arrives: acting on a stale
    # one costs this pane a decode it could have shown, and says nothing.
    if ids and not swapped_model(recorded, models.loaded_model_id()):
        text, load_id, failure = read_through(models.decode, ids)
        if text is not None and reads_back(recorded, model_of(load_id)):
            return (f"**{where} · as recorded** · {len(ids):,} prompt tokens, decoded"
                    f"{under_load(turn.get('load_id'), load_id)}.{tail}", text)
    messages = context_messages(ep, index)
    text, load_id, refused = read_through(models.prompt_text, messages, TOOLS)
    if text is not None:
        # Named by the load that spelled this template and by no earlier
        # reading: the load that answered one reading can be gone by the next,
        # so a model carried from another could be called loaded beside text
        # it did not spell.
        swapped = swapped_model(recorded, model_of(load_id))
        if swapped and ids:
            # The aside names both models, so the load stamps would only repeat
            # it in weaker words.
            frame, aside = "", spelled_by_another(len(ids), recorded, swapped)
        elif swapped:
            # No recorded IDs to leave undecoded, no response having been asked
            # for yet, so the swap is said of the reading itself rather than
            # going unsaid.
            frame, aside = read_by_another(recorded, swapped), ""
        else:
            frame = under_load(ep.load_id, load_id)
            aside = unnamed_model(len(ids)) if ids and not recorded else ""
        return (f"**{where} · as the loaded model would be given it** · {len(messages)} messages and the move tool "
                f"through that model's own template{frame}.{aside}{tail}", text)
    # Nothing here was spelled by a load, this transcript being the run's own
    # messages, so the model is read fresh rather than carried from a reading
    # that failed: what the aside says is what is in memory as it is written,
    # which is also what the reader is being asked to change. A load under way
    # names nothing and the aside stays away, as it does with none loaded.
    swapped = swapped_model(recorded, models.loaded_model_id())
    if ids and swapped:
        aside = spelled_by_another(len(ids), recorded, swapped)
    else:
        aside = unnamed_model(len(ids)) if ids and not recorded else ""
    return (f"**{where} · as recorded, untemplated** · {unspelled(models, refused or failure)}, so the "
            f"{len(messages)} messages and the move tool are shown as the run recorded them. A template adds its own "
            f"turn markers and writes the tool schemas its own way.{aside}{tail}", transcript(messages))


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
    inserted = {insert["before_turn"] for insert in ep.config.get("context_inserts", ())}
    origin = "Inside the template's open reasoning block" if t.get("reasoning_prefilled") else "At the beginning of the assistant response"
    note = (f"**Supplied interruption · {forced} tokens** · {origin}.\n\n" if forced else "No supplied interruption in this response.")
    if not forced and t.get("planned_prefix_ids"):
        note = f"**Pending interruption · {len(t['planned_prefix_ids'])} tokens** · Prefix insertion has not been confirmed."
    if t.get("token_edit"):
        note = (f"**{'Retained / edited' if forced else 'Pending retained / edited'} prefix · {forced or len(t.get('planned_prefix_ids', []))} tokens** · "
                "Earlier token IDs and your replacement are supplied as context; only the new continuation counts toward sampled-token limits.")
    return (board(ep, index, reveal, animate), status(ep), TOKENS.strip(metrics[forced:]),
            t.get("text", ""), note, t.get("prefix_text") or t.get("planned_prefix_text", ""), timeline(ep), stamped,
            gr.update(choices=[("Initial / supplied history", -1)] + [(f"Response {i+1}" + (" · token edit" if t.get("token_edit") else " · interruption" if t.get("prefix_ids") else "") + (" · after an inserted message" if i in inserted else "") + (" · steered" if t.get("steered") else ""), i) for i, t in enumerate(ep.turns)], value=index),
            "Select a model-generated token above." if changed else gr.skip(), [] if changed else gr.skip())


# The note is Markdown, and the names in it come from a file that may have
# been written anywhere. Escaping the HTML leaves `**` and `[…](…)` to be read
# as syntax, which is enough to close the bold span the provenance is written
# in and continue in a voice that looks like the workbench's own.
MARKDOWN = str.maketrans({character: "\\" + character for character in "\\`*_{}[]()#+-.!>|~"})


def as_text(value):
    """A name from a trial file, read as the characters it is."""

    return html.escape(value).translate(MARKDOWN)


def trial_stamp(trial):
    """Read a run's trial stamp as whatever it recorded, or nothing at all.

    A trial loaded here is stamped with its label and its file's title, but a
    run generated by a harness outside this page may carry only an id, and a
    config read from a file may carry anything. The note is built on the same
    return as the board and the controls, so a stamp it could not read would
    take the whole frame down with it and leave a replay unloadable.
    """

    if not isinstance(trial, dict):
        return None, None
    name = trial.get("label") or trial.get("id")
    if not isinstance(name, str) or not name.strip():
        return None, None
    title = trial.get("title")
    return name, title if isinstance(title, str) and title.strip() else None


def trial_note_text(ep, data=None):
    """What the trials pane says about the episode on screen right now.

    The episode is replaced by several other controls, and a note that still
    named a trial after one of them would have an experimenter running or
    exporting something else in its name. Uploading a collection does not
    replace the episode, so what it says about the run stands.
    """

    parts = []
    name, title = trial_stamp(ep.config.get("trial"))
    if name:
        parts.append(f"**{'Replaying' if ep.replay_only else 'Running'}: {as_text(name)}**"
                     + (f", from {as_text(title)}." if title else "."))
    if data:
        count = len(data["trials"])
        parts.append(f"Loaded **{as_text(data['title'])}** · {count} trial{'s' if count != 1 else ''}. "
                     "Select one and click Load trial.")
    elif not name:
        parts.append("Upload a trial file, choose a trial, then load it. Inspect the maze before playing.")
    else:
        parts.append("New episode starts a separate run from the controls.")
    return " ".join(parts)


BATCH_HEADERS = ["Trial", "Outcome", "Model moves", "Sampled tokens", "Recovered", "Waypoint"]


def batch_rows(rows):
    """The summary rows as the results table shows them, newest last."""
    def yes_no(value):
        return "" if value == "" else "Yes" if value else "No"
    return [[row["label"], row["outcome"], row["model_moves"], row["sampled_tokens"], yes_no(row["recovered"]),
             yes_no(row["waypoint_reached"])] for row in rows]


def batch_text(data, done, total, rows, directory, current=None, ended=None, error=None):
    """What the trials pane says about a batch: running, stopped, failed or finished."""
    where = f"Runs and summary.csv are written to {as_text(str(directory))}."
    if ended is None and done >= total:
        return f"**All {total} trials ran** · writing the summary.\n\n{where}"
    if ended is None:
        label = as_text(data["trials"][done]["label"])
        progress = ""
        if current is not None:
            partial = 0
            if current.turns and current.turns[-1]["finish_reason"] is None:
                turn = current.turns[-1]
                partial = max(0, len(turn["metrics"]) - turn["forced_prefix_tokens"])
            progress = (f" · response {len(current.turns)} · {current.sampled_tokens + partial:,} sampled tokens · "
                        f"{current.moves - current.supplied_moves} model moves")
        return f"**Running trial {done + 1} of {total}** · {label}{progress}\n\n{where}"
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    counts = " · ".join(f"{count} {outcome}" for outcome, count in sorted(outcomes.items()))
    head = (f"**Finished {total} trial{'s' if total != 1 else ''}**" if ended == "finished"
            else f"**Failed after {len(rows)} of {total} trials**" if ended == "failed"
            else f"**Stopped after {len(rows)} of {total} trials**")
    failed = (f"{as_text(str(error).rstrip('.'))}. What is on disk may not include the last trial. "
              if ended == "failed" else "")
    return (f"{head} of {as_text(data['title'])}{' · ' + counts if counts else ''}\n\n{failed}{where} "
            "Open any run under **Load a saved run** to replay it.")


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
    with gr.Row(elem_id="maze-workspace"):
        with gr.Column(elem_id="maze-scenario"):
            gr.Markdown("## Scenario")
            models = gr.Button("Choose / load model", size="sm")
            with gr.Accordion("Experiment trials", open=False):
                trial_upload = gr.File(label="Trial definitions JSON", file_types=[".json"], type="filepath")
                trial_picker = gr.Dropdown(choices=[], label="Trial", interactive=True,
                                           info="Type to filter a long collection.")
                trial_load = gr.Button("Load trial", size="sm", elem_id="maze-load-trial")
                trial_note = gr.Markdown(trial_note_text(initial))
                batch_control = gr.State(BatchControl())
                with gr.Row():
                    batch_run = gr.Button("Run all trials", size="sm", elem_id="maze-run-trials")
                    batch_stop = gr.Button("Stop the batch", size="sm", visible=False, elem_id="maze-stop-trials")
                batch_status = gr.Markdown("Run all trials generates every trial in the file to its end, one after "
                                           "another under the loaded model, and saves each run with a summary table.",
                                           elem_id="maze-batch-status")
                batch_results = gr.Dataframe(headers=BATCH_HEADERS, interactive=False, wrap=True, visible=False,
                                             elem_id="maze-batch-results")
                batch_download = gr.File(label="Batch summary", file_count="multiple", interactive=False,
                                         visible=False)
            with gr.Accordion("Load a saved run", open=False):
                upload = gr.File(label="Saved run JSON", show_label=False, file_types=[".json"], type="filepath")
            gr.Markdown("The settings below apply to the next episode. Loading a trial or a saved run shows the settings it used.")
            prepare = gr.Button("New episode · apply settings", elem_id="maze-prepare")
            with gr.Accordion("Setup prompt", open=False):
                system_prompt = gr.Textbox(value=SYSTEM, label="System prompt", lines=2, elem_id="maze-system-prompt")
                instruction = gr.Textbox(value=default_instruction("coordinates"), label="Task instruction", lines=6,
                                         elem_id="maze-instruction",
                                         info="Sent verbatim ahead of the JSON state. Changing Goal information rewrites this unless you have edited it.")
            changing_map = gr.Checkbox(value=False, label="Map can change during the run", elem_id="maze-changing",
                                       info="Fixes one identifier for the maze, so closing a cell mid-run does not rename it to the model. Saved as a chatlab-maze-run-2 file.")
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
            with gr.Accordion("Waypoint & steering", open=False):
                waypoint = gr.Textbox(label="Waypoint", placeholder="row, column", elem_id="maze-waypoint",
                                      info="A cell the model is asked to pass through. The state shows it with "
                                           "waypoint_reached; say what it means in the Task instruction. Blank for none.")
                steer_vector = gr.State(None)
                steer_file = gr.File(label="Steering vector JSON", file_types=[".json"], type="filepath",
                                     elem_id="maze-steer-file")
                steer_note = gr.Markdown(vector_note(None))
                with gr.Row():
                    steer_strength = gr.Number(value=1.0, minimum=-100, maximum=100, label="Strength",
                                               elem_id="maze-steer-strength")
                    steer_layer = gr.Number(value=0, precision=0, minimum=0, label="Layer", elem_id="maze-steer-layer")
                steer_mode = gr.Dropdown(choices=[(label, mode) for mode, label in STEER_MODES.items()], value="off",
                                         label="Steer", elem_id="maze-steer-mode",
                                         info="Steering starts with the first response generated once this holds, and starts once.")
                with gr.Row():
                    steer_cell = gr.Textbox(label="Steering cell", placeholder="row, column", elem_id="maze-steer-cell",
                                            info="Blank uses the waypoint.")
                    steer_after = gr.Number(value=3, precision=0, minimum=0, maximum=255, label="After moves",
                                            elem_id="maze-steer-after", info="Counts supplied moves.")
                steer_responses = gr.Number(value=1, precision=0, minimum=0, maximum=256, label="Steered responses",
                                            elem_id="maze-steer-responses", info="0 keeps steering on to the end of the run.")
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
            with gr.Accordion("Export the run", open=False):
                save = gr.Button("Export run JSON", size="sm")
                download = gr.File(label="Saved run", interactive=False)
        with gr.Column(elem_id="maze-center"):
            maze_board = gr.HTML(board(initial), elem_id="maze-board")
            transport_status = gr.Markdown(transport_text(initial), elem_id="maze-transport-status")
            with gr.Row(elem_id="maze-transport"):
                first = gr.Button("First", size="sm", elem_id="maze-first",
                                  elem_classes=icon_classes("chevron-first"))
                back = gr.Button("Previous", size="sm", elem_id="maze-previous",
                                 elem_classes=icon_classes("chevron-left"))
                toggle = gr.Button("Play", variant="primary", size="sm", elem_id="maze-run",
                                   elem_classes=icon_classes("play"))
                pause = gr.Button("Pause", variant="primary", size="sm", visible=False, elem_id="maze-pause",
                                  elem_classes=icon_classes("pause"))
                forward = gr.Button("Next", size="sm", elem_id="maze-next",
                                    elem_classes=icon_classes("chevron-right", trailing=True))
                interrupt = gr.Button("Interrupt", size="sm", elem_id="maze-interrupt")
            turn_picker = gr.Dropdown(choices=[("Initial / supplied history", -1)], value=-1,
                                      label="Selected response", interactive=True)
            with gr.Accordion("Change the map", open=False):
                with gr.Row():
                    close_row = gr.Number(value=0, precision=0, minimum=0, maximum=14, label="Row", elem_id="maze-close-row")
                    close_column = gr.Number(value=0, precision=0, minimum=0, maximum=14, label="Column", elem_id="maze-close-column")
                close_cell_button = gr.Button("Close this cell", size="sm", elem_id="maze-close-cell")
                gr.Markdown("Available in an episode started with **Map can change during the run**. The cell becomes a wall "
                            "before the next generated response, and the model meets the change in the simulator's next reply. "
                            "The start, the destination, the cell the character is standing in, and any closure that would cut "
                            "the destination off from the character or from the start are refused.")
            with gr.Accordion("Insert a message", open=False):
                insert_channel = gr.Dropdown(choices=[(label, key) for key, label in CHANNELS.items()], value="tool_note",
                                             label="Channel", elem_id="maze-insert-channel")
                insert_text = gr.Textbox(label="Message", lines=2, elem_id="maze-insert-text")
                with gr.Row():
                    insert_sender = gr.Textbox(label="Sender", placeholder="Teammate messages only",
                                               elem_id="maze-insert-sender")
                    insert_direction = gr.Dropdown(choices=[("None", "")] + [(d, d) for d in DIRECTIONS], value="",
                                                   label="Advised direction", elem_id="maze-insert-direction",
                                                   info="Recorded for scoring. The model is not told it.")
                insert_button = gr.Button("Queue this message", size="sm", elem_id="maze-insert")
                gr.Markdown("The message goes into the model's context before the next generated response. A simulator "
                            "note becomes the last key of the latest simulator reply, `\"note\"`; a teammate message is "
                            "added there as `\"messages\"`, as a team run delivers one; a user message is a turn of its "
                            "own after that reply. One message is queued at a time.")
            with gr.Accordion("Playback & view", open=False):
                pace = gr.Slider(.1, 4, value=1., step=.1, label="Seconds per recorded response")
                reveal = gr.Checkbox(label="Show shortest route (viewer only)", value=False)
                gr.Markdown("Play replays recorded responses, then continues generating in a live episode. Next advances one response, First returns to the initial history. Pause lets a generated response finish.")
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
                                  interactive=False, wrap=True, elem_id="maze-history")
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
                budget, attempts, recovery_tokens, recovery_attempts, goal_mode, goal_hint, system_prompt, instruction,
                changing_map]
    checkpoint_controls = [waypoint, steer_vector, steer_strength, steer_layer, steer_mode, steer_cell, steer_after,
                           steer_responses]

    def prepare_episode(ep, show, session_id, data, *values):
        if ep.busy:
            raise gr.Error("Stop or pause this episode before starting another.")
        (n, s, d, o, supplied_n, trigger, passage_text, count, temp, sample_seed, per, total, tries,
         window_tokens, window_attempts, mode, hint, system_text, instruction_text, map_changes,
         waypoint_text, *steer) = values
        try:
            waypoint_cell = parse_cell(waypoint_text, "waypoint")
            checkpoint = dict(waypoint=waypoint_cell) if waypoint_cell else {}
            checkpoint.update(steering_config(*steer, waypoint_cell))
            drawn = generate(n, s, d, o)
            new = Episode(changing(drawn) if map_changes else drawn,
                          dict(supplied_moves=int(supplied_n), interrupt_after=int(trigger),
                               interruption_text=passage_text, prefix_tokens=int(count), temperature=float(temp),
                               openness=float(o), sampling_seed=int(sample_seed), per_turn_tokens=int(per),
                               token_budget=int(total), attempt_budget=int(tries),
                               recovery_tokens=int(window_tokens), recovery_attempts=int(window_attempts),
                               goal_mode=mode, goal_hint=hint, system_prompt=system_text, instruction=instruction_text,
                               **checkpoint))
        except (ValueError, TypeError) as exc:
            logger.warning("Refused the scenario settings for a new episode: %s", exc)
            raise gr.Error(str(exc)) from exc
        interruption = str(new.config.get("interruption_text") or "").strip()
        logger.info("New episode %s: %s x %s %s maze, seed %s, %s goal, %s supplied moves, interruption %s, "
                    "waypoint %s, steering %s",
                    new.run_id, new.maze.size, new.maze.size, "changing" if new.map_changes else "fixed",
                    new.maze.seed, new.config["goal_mode"], new.supplied_moves,
                    f"after {new.config['interrupt_after']} moves" if interruption else "off",
                    new.config.get("waypoint") or "none",
                    f"{new.config['steer_when']} for {new.config['steer_responses'] or 'all'} responses"
                    if new.config.get("steering") else "off")
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
            logger.warning("Could not read the trial collection in %s: %s", path, exc)
            raise gr.Error(f"Could not load trials: {exc}{kept}") from exc
        logger.info("Read %s trials from %s", len(data["trials"]), path)
        return (data, gr.update(choices=[(t["label"], t["id"]) for t in data["trials"]], value=data["trials"][0]["id"]),
                trial_note_text(ep, data))

    def load_trial(data, trial_id, ep, show, session_id):
        try:
            if data is None:
                raise ValueError("Load a trial definitions file first.")
            new = prepare_trial(data, trial_id, ep)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Could not prepare trial %r: %s", trial_id, exc)
            raise gr.Error(str(exc)) from exc
        logger.info("Prepared trial %r as run %s", trial_id, new.run_id)
        stop_replay(ep)
        # The same description a loaded run gets: the trial is in the episode
        # now, so nothing here has to spell the controls out a second time.
        return (new, *render(new, show, session_id), *checkpoint_values(new), *scenario_values(new),
                trial_note_text(new), None, None, *model_button(new))

    def run_batch(data, control, source):
        """Run every trial in the loaded file, reporting each one as it finishes."""
        if data is None:
            raise gr.Error("Load a trial definitions file first.")
        if control.running:
            raise gr.Error("This batch is already running.")
        buttons = (gr.update(visible=False), gr.update(visible=True))
        done = total = 0
        rows, directory, failure = [], None, None
        try:
            frames = run_trials(data, context.models, runs_dir(context), control,
                                source=Path(source).name if source else "")
            for done, total, rows, directory, current in frames:
                # The files of an earlier batch are cleared while this one runs,
                # so the pane never offers them beside another collection's
                # progress. This batch's own are offered when it ends.
                yield (batch_text(data, done, total, rows, directory, current),
                       gr.update(value=batch_rows(rows), visible=bool(rows)),
                       gr.update(value=None, visible=False), *buttons)
        except (ValueError, OSError) as exc:
            # The model is busy or not loaded, or the batch directory could
            # not be written, and nothing ran. Or a run or the summary could
            # not be written partway, and the batch failed with every row it
            # had, even the last one, possibly never reaching the disk.
            logger.warning("Could not run the trials in %s: %s", data["title"], exc)
            gr.Warning(str(exc))
            if directory is None:
                # Nothing ran and nothing on the pane changed, so it is left
                # alone: the refusal may be this session's own batch holding
                # the model, whose Stop button has to stay where it is.
                yield (gr.skip(),) * 5
                return
            failure = exc
        # The failure first: one that lands writing the summary after the last
        # trial leaves every row in place, and read from the count alone that
        # batch would say it finished.
        ended = "failed" if failure is not None else "stopped" if cut_short(rows, total) else "finished"
        try:
            files = gr.update(value=downloads(directory), visible=True)
        except OSError as exc:
            # The disk that failed the batch is often the one the copies would
            # go to. The pane still has to say how the batch ended and give the
            # Run button back; the files themselves are named in the status.
            logger.warning("Could not stage the downloads for %s: %s", directory, exc)
            files = gr.update(value=None, visible=False)
        yield (batch_text(data, done, total, rows, directory, ended=ended, error=failure),
               gr.update(value=batch_rows(rows), visible=bool(rows)), files,
               gr.update(visible=True), gr.update(visible=False))

    def stop_batch(control):
        if not control.running:
            return
        control.request_stop()
        logger.info("Stop requested for the running batch")
        gr.Info("Stopping the batch: the trial running now ends as stopped, and no more trials start.")

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
            # The refusal the reader sees as a toast and asks about afterwards:
            # no model loaded, a model busy in another view, or an episode that
            # is finished or is a replay. None of it reached the log before.
            logger.warning("Run %s cannot generate: %s", ep.run_id, exc)
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
            logger.warning("Run %s refused %s: %s", ep.run_id, kind, exc)
            raise gr.Error(str(exc)) from exc
        logger.info("Run %s: %s requested while %s", ep.run_id, kind, ep.phase)
        gr.Info({"pause": "Pausing after the current response." if ep.busy else "Playback paused.", "stop": "Stopping; partial actions will not execute.", "interrupt": "Interruption queued for the next response."}[kind])
        return status(ep), transport_text(ep), *transport_buttons(ep)

    def close_map_cell(ep, row, column):
        try:
            if row is None or column is None:
                raise ValueError("Enter the row and the column of the cell to close.")
            cell = (int(row), int(column))
            ep.request_closure(cell)
        except (TypeError, ValueError) as exc:
            logger.warning("Run %s refused to close row %s, column %s: %s", ep.run_id, row, column, exc)
            raise gr.Error(str(exc)) from exc
        logger.info("Run %s: row %s, column %s closes before response %s",
                    ep.run_id, cell[0], cell[1], len(ep.turns) + 1)
        gr.Info("The cell closes before the next generated response.")
        return status(ep), transport_text(ep), *transport_buttons(ep)

    def queue_message(ep, channel, text_value, sender, direction):
        try:
            ep.request_insert(channel, text_value, sender if channel == "teammate" else None, direction or None)
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Run %s refused a %s: %s", ep.run_id, channel, exc)
            raise gr.Error(str(exc)) from exc
        logger.info("Run %s: a %s goes in before response %s", ep.run_id, channel, len(ep.turns) + 1)
        gr.Info("The message goes into the context before the next generated response.")
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

    def step_first(ep, show, session_id):
        if ep.busy:
            raise gr.Error("Pause the episode before stepping through responses.")
        stop_replay(ep)
        return render(ep, show, session_id, -1, animate=True)

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
        logger.info("Play on run %s from %s of %s recorded%s", ep.run_id,
                    "the initial history" if start < 0 else f"response {start + 1}", len(ep.turns),
                    ", a saved replay that stops at its end" if ep.replay_only else "")
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
        rows = history_rows(ep)
        return inspect(ep, show, rows[max(0, min(int(row), len(rows) - 1))][0], session_id)

    def change_reveal(ep, show, session_id):
        ep.reveal_route = show
        if not ep.busy:
            stop_replay(ep)
        return render(ep, show, session_id, viewing(ep))

    def export(ep):
        return export_run(ep, runs_dir(context))

    def load(path, ep, show, session_id, data):
        # Every arrival here is recorded, because the failure this is read for
        # is an upload that appears to do nothing. Gradio fires upload only for
        # a file it took as new, so no line at all is itself the answer: the
        # file never reached the extension, having been dropped on the
        # read-only Saved run box under Export the run, or re-selected while
        # the widget already held it.
        if ep.busy:
            logger.warning("Refused a replay upload while run %s was generating", ep.run_id)
            raise gr.Error("Pause or stop this episode before loading a replay.")
        if not path:
            logger.info("A replay upload arrived with no file; run %s is unchanged", ep.run_id)
            return (gr.skip(),) * (len(outputs) + len(checkpoint_controls) + len(controls) + 6)
        try:
            if Path(path).stat().st_size > 50_000_000:
                raise ValueError("Run files must be smaller than 50 MB.")
            replay = from_payload(json.loads(Path(path).read_text()),
                                  read_prompt=lambda run, turn, messages: prompt_reading(run, turn, messages, context.models))
            # Before the first frame: recovering the open-cell probability writes
            # it onto the run, and Run details reports whichever way that went.
            values = (*checkpoint_values(replay), *scenario_values(replay))
            # Every part of the frame is built inside this guard. One built on
            # the return would escape the handler if the run it read defeated
            # it, and Gradio abandons an event whole: the board, the controls
            # and Run details would all keep showing the episode this replay
            # was picked to replace, with nothing said about why.
            note, buttons = trial_note_text(replay, data), model_button(replay)
            # At the beginning, not the end: a saved run is opened to be
            # watched, and Play from its final response has nothing left to
            # replay and no live end to continue into.
            rendered = render(replay, show, session_id, -1)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            logger.warning("Could not load the run in %s: %s", path, exc)
            raise gr.Error(f"Could not load run: {exc}") from exc
        logger.info("Loaded run %s from %s for replay: %s responses, %s moves, phase %s, model %s",
                    replay.run_id, path, len(replay.turns), replay.moves, replay.phase,
                    replay.model_id or "unrecorded")
        stop_replay(ep)
        return (replay, *rendered, *values, note, *buttons)

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
            logger.warning("Run %s: the token edit was refused: %s", ep.run_id, exc)
            raise gr.Error(str(exc)) from exc
        logger.info("Run %s forked into %s at response %s, token %s", ep.run_id, new.run_id,
                    turn_index + 1, token_index)
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
    prepare.click(prepare_episode, [episode, reveal, selection_session, trial_data, *controls, *checkpoint_controls],
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
    # Its own queue, with no limit on it. A batch runs for hours, so the view's
    # queue would hold every step, selection and upload behind it, and a second
    # batch queued behind the first would start long after it was asked for, on
    # the inputs it was asked with. Unqueued, it reaches the model session,
    # which refuses it on the spot while another batch holds the model.
    batch_run.click(run_batch, [trial_data, batch_control, trial_upload],
                    [batch_status, batch_results, batch_download, batch_run, batch_stop],
                    concurrency_id="maze-batch", concurrency_limit=None, show_progress="hidden")
    batch_stop.click(stop_batch, batch_control, None, queue=False)
    trial_load.click(load_trial, [trial_data, trial_picker, episode, reveal, selection_session],
                     [episode, *outputs, *checkpoint_controls, steer_note, *controls, passage, trial_note,
                      edit_selection, download, models, wanted_model],
                     concurrency_id="maze-view", show_progress="hidden")
    first.click(step_first, [episode, reveal, selection_session], outputs, show_progress="hidden", concurrency_id="maze-view")
    back.click(step_back, [episode, reveal, selection_session], outputs, show_progress="hidden", concurrency_id="maze-view")
    forward.click(step_forward, [episode, reveal, selection_session], outputs, show_progress="hidden", concurrency_id="maze-view")
    command_outputs = [state_text, transport_status, toggle, pause]
    pause.click(lambda ep: command(ep, "pause"), episode, command_outputs, queue=False)
    stop.click(lambda ep: command(ep, "stop"), episode, command_outputs, queue=False)
    interrupt.click(lambda ep: command(ep, "interrupt"), episode, command_outputs, queue=False)
    close_cell_button.click(close_map_cell, [episode, close_row, close_column], command_outputs, queue=False)
    insert_button.click(queue_message, [episode, insert_channel, insert_text, insert_sender, insert_direction],
                        command_outputs, queue=False)
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
                  [episode, *outputs, *checkpoint_controls, steer_note, *controls, passage, trial_note, models,
                   wanted_model],
                  concurrency_id="maze-view", show_progress="hidden")
    def import_vector(path):
        if not path:
            return None, vector_note(None), gr.skip(), gr.skip()
        try:
            vector = read_steering_vector(path)
        except (ValueError, OSError) as exc:
            logger.warning("Could not read the steering vector in %s: %s", path, exc)
            raise gr.Error(f"Could not load the vector: {exc}") from exc
        logger.info("Read a steering vector for %s at layer %s from %s", vector["model_id"], vector["layer"], path)
        return vector, vector_note(vector), vector["strength"], vector["layer"]

    for event in (steer_file.upload, steer_file.clear):
        event(import_vector, steer_file, [steer_vector, steer_note, steer_strength, steer_layer],
              show_progress="hidden")
    context.navigation.open_models(models, wanted_model)


def build_page(context):
    # Imported here because the Team tab borrows this module's helpers.
    from .team_page import build_team_page

    with gr.Column(elem_id="maze-page"):
        gr.Markdown("# Maze workbench")
        with gr.Tabs(elem_id="maze-modes"):
            with gr.Tab("One agent", elem_id="maze-single-tab"):
                _build_page(context)
            with gr.Tab("Team", elem_id="maze-team-tab"):
                build_team_page(context, lambda: runs_dir(context))
