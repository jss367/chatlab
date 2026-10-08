"""How the workbench draws a team: every agent on one board, its rounds, and what the agents said."""
from __future__ import annotations

import html
import math

from .dynamic_maze import maze_at_turn
from .maze import DIRECTIONS, GOAL_MODES, parting_cell
from .runner import TERMINAL, steering_active, visible_text
from .team import DROPPED, LIMITED, MESSAGE_LIMIT, TEAM_GOALS

# The first four agents' colours, none of them the destination's green.
COLORS = ("#4f46e5", "#db2777", "#0891b2", "#ea580c")
# A cell holding more agents than this shows one marker with their count.
FANNED = 4
OUTCOMES = {"no_call": "No call · stopped", "cut_off": "Cut off · stopped", "out_of_tokens": "Out of tokens · stopped",
            "not_applied": "Not applied"}
HEADERS = ["Round", "Agent", "Position", "Direction", "Result", "Message"]
# The note is Markdown, and the names in it come from a file that may have
# been written anywhere. Escaping the HTML leaves `**` and `[…](…)` to be read
# as syntax, which is enough to close the bold span the provenance is written
# in and continue in a voice that looks like the workbench's own.
MARKDOWN = str.maketrans({character: "\\" + character for character in "\\`*_{}[]()#+-.!>|~"})


def agent_color(k):
    """Agent k's colour: the fixed four, then hues spread by the golden ratio over every hue but green.

    Two lightnesses, both dark enough for the white number on the marker,
    keep neighbouring hues apart.
    """
    if k < len(COLORS):
        return COLORS[k]
    hue = (k * .618034 % 1) * 279
    if hue >= 95:
        hue += 81
    return f"hsl({hue:.1f} 70% {(36, 44)[k % 2]}%)"


def event_round(event):
    """The round a move landed in, -1 for a supplied one, which every agent has made before any round."""
    return event["round"] if event["source"] == "model" else -1


def lap_start(ep, index):
    """The round the lap holding round ``index`` began with: 0, or the start of a later lap.

    A lap that begins with round ``index + 1`` has already put every agent
    back at the start once round ``index`` is over, so the board after that
    round shows the new lap.
    """
    return max([0, *(start for start in getattr(ep, "lap_rounds", ()) if start <= index + 1)])


def lap_events(ep, index):
    """The moves of the lap the board after round ``index`` shows, up to that round."""
    begun = lap_start(ep, index)
    return [event for event in ep.events if begun <= event_round(event) <= index or
            (begun == 0 and event["source"] == "supplied")]


def positions_after(ep, index):
    """Where every agent stood after round ``index``, -1 being where the supplied moves left them.

    A lap puts every agent back at the start, except one a limit had stopped,
    which stays where it stopped. An agent that answers in a lap was put back.
    """
    positions = [ep.maze.start] * len(ep.agents)
    begun = lap_start(ep, index)
    for event in ep.events:
        if event["accepted"] and event_round(event) < begun:
            positions[event["agent"]] = tuple(event["after"])
    if begun:
        answered = {turn["agent"] for turn in ep.turns if turn["round"] >= begun}
        for k, agent in enumerate(ep.agents):
            if k in answered or (begun >= ep.rounds and agent["status"] not in LIMITED):
                positions[k] = ep.maze.start
    for event in lap_events(ep, index):
        if event["accepted"]:
            positions[event["agent"]] = tuple(event["after"])
    return positions


# A cell's side on the board, and the margin the row and column numbers sit in.
CELL, PAD = 56, 28


def cell_center(point):
    """The centre of the cell at ``point``, a (row, column) pair, on the board."""
    return PAD + (point[1] + .5) * CELL, PAD + (point[0] + .5) * CELL


def cell_points(path):
    """A path of cells as an SVG ``points`` list through their centres."""
    return " ".join(f"{x},{y}" for x, y in map(cell_center, path))


def board_grid(maze, closed, label):
    """The board's opening: the SVG tag, every cell with those in ``closed`` closed, and the row and column numbers."""
    size = maze.size
    total = CELL * size + 2 * PAD
    parts = [f'<svg viewBox="0 0 {total} {total}" role="img" aria-label="{size} by {size} maze. {label}.">']
    for r, row in enumerate(maze.grid):
        for c, value in enumerate(row):
            fill = "#78350f" if (r, c) in closed else "#27344a" if value == "#" else "#fff"
            parts.append(f'<rect x="{PAD+c*CELL+2}" y="{PAD+r*CELL+2}" width="52" height="52" rx="7" fill="{fill}"/>')
    for i in range(size):
        parts.append(f'<text x="{PAD+(i+.5)*CELL}" y="17" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
        parts.append(f'<text x="12" y="{PAD+(i+.5)*CELL+4}" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
    return parts


def waypoint_mark(point):
    """The flag on the waypoint's cell."""
    x, y = cell_center(point)
    return f'<text x="{x}" y="{y+8}" text-anchor="middle" font-size="22" fill="#0f766e">⚑</text>'


def insert_mark(x, y, advised):
    """The ring where a message went in, centred on ``x``, ``y``, and an arrow for the direction it advised."""
    parts = [f'<circle cx="{x}" cy="{y}" r="21" stroke="#db2777" stroke-width="3" stroke-dasharray="4 3" fill="none"/>']
    if advised in DIRECTIONS:
        dr, dc = DIRECTIONS[advised]
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
    return "".join(parts)


def team_board(ep, index=None, reveal=False, map_round=None):
    """The board after round ``index``, drawn under the map that round was played on.

    ``map_round`` draws another round's map and interventions instead: a
    round that never resolved, whose agents still stand where the round
    before left them, was played on the map as it stood when it began,
    closures landed at its start included, and its responses were
    interrupted where they were.
    """
    index = ep.rounds - 1 if index is None else index
    map_round = index if map_round is None else map_round
    updates = [u for u in ep.config.get("map_updates", ()) if u["before_round"] <= map_round]
    maze = maze_at_turn(ep.maze, updates, None, "before_round")
    closed = {tuple(u["closed_cell"]) for u in updates}
    positions = positions_after(ep, index)
    where = "; ".join(f"{a['name']} at row {p[0]}, column {p[1]}" for a, p in zip(ep.agents, positions))
    parts = board_grid(maze, closed, where)
    if reveal:
        parts.append(f'<polyline points="{cell_points(maze.route())}" fill="none" stroke="#b4bdcc" stroke-width="4" stroke-dasharray="3 8"/>')
    checkpoint = ep.config.get("required_checkpoint")
    steer_cell = (ep.config.get("steer_when") or {}).get("cell")
    for point, color, label in ((checkpoint, "#0891b2", "Required checkpoint"),
                                (steer_cell, "#9333ea", "Steering cell")):
        if point is not None:
            x, y = cell_center(point)
            inset = 22 if label == "Required checkpoint" else 18
            parts.append(f'<rect x="{x-inset}" y="{y-inset}" width="{inset*2}" height="{inset*2}" '
                         f'fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="5 3">'
                         f'<title>{label}</title></rect>')
    waypoint = ep.config.get("waypoint")
    if waypoint is not None:
        parts.append(waypoint_mark(waypoint))
    # Each agent's path is drawn a little off the cell centre, so two agents
    # walking the same corridor stay two lines. The offsets repeat every eight
    # agents so that no path leaves its own cells.
    offsets = [((k % 2) * 2 - 1) * 5 * (k // 2 % 4 + 1) if len(ep.agents) > 1 else 0 for k in range(len(ep.agents))]
    # A run in laps draws the lap the round belongs to, since every lap walks the same maze again.
    for event in (lap_events(ep, index) if getattr(ep, "lap_rounds", None) else ep.events):
        if event["accepted"] and event_round(event) <= index:
            k = event["agent"]
            (x1, y1), (x2, y2) = cell_center(event["before"]), cell_center(event["after"])
            d = offsets[k]
            dash = ' stroke-dasharray="5 6"' if event["source"] == "supplied" else ""
            parts.append(f'<line x1="{x1+d}" y1="{y1+d}" x2="{x2+d}" y2="{y2+d}" stroke="{agent_color(k)}" '
                         f'stroke-width="5" stroke-linecap="round" opacity=".75"{dash}/>')
    # Where each agent was interrupted, from the round it happened in.
    for agent in ep.agents:
        turn = agent.get("intervention_turn")
        if agent.get("interrupted") and turn is not None and ep.turns[turn]["round"] <= map_round:
            x, y = cell_center(ep.turns[turn]["position_before"])
            parts.append(f'<circle cx="{x}" cy="{y}" r="23" stroke="#f59e0b" stroke-width="3" fill="none">'
                         f'<title>{html.escape(agent["name"])} interrupted</title></circle>')
    # Where each message went in, from the round that read it, as a run of one agent draws its own.
    for insert in ep.config.get("context_inserts", ()):
        if insert["before_round"] <= map_round:
            parts.append(insert_mark(*cell_center(insert["position"]), insert.get("advised_direction")))
    x, y = cell_center(maze.start)
    parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#64748b" font-size="14" font-weight="700">S</text>')
    exits = ep.exits if getattr(ep, "rewarded", False) else None
    reward = ep.config.get("reward_exit") if exits is not None or getattr(ep, "rewarded", False) else None
    for label, cell in (exits or {"A": maze.goal}).items():
        x, y = cell_center(cell)
        if label == reward:
            parts.append(f'<circle cx="{x}" cy="{y}" r="24" fill="none" stroke="#d97706" stroke-width="3">'
                         f'<title>Reward exit</title></circle>')
        parts.append(f'<circle cx="{x}" cy="{y}" r="17" fill="#d1fae5"/><text x="{x}" y="{y+7}" text-anchor="middle" '
                     f'font-size="23" fill="#047857">★</text>')
        if exits is not None:
            parts.append(f'<text x="{x+17}" y="{y-13}" text-anchor="middle" font-size="12" font-weight="700" '
                         f'fill="#047857"><title>Exit {label}</title>{label}</text>')
    # Up to four agents sharing a cell are fanned out around it rather than
    # stacked. More than that would not fit, so the cell shows how many there
    # are and names them on hover.
    by_cell = {}
    for k, position in enumerate(positions):
        by_cell.setdefault(tuple(position), []).append(k)
    for position, sharing in by_cell.items():
        x, y = cell_center(position)
        if len(sharing) > FANNED:
            names = html.escape(", ".join(ep.agents[k]["name"] for k in sharing))
            parts.append(f'<g><title>{names}</title><circle cx="{x}" cy="{y}" r="19" fill="#334155" stroke="white" '
                         f'stroke-width="3"/><text x="{x}" y="{y+5}" text-anchor="middle" fill="white" font-size="13" '
                         f'font-weight="700">×{len(sharing)}</text></g>')
            continue
        for slot, k in enumerate(sharing):
            dx = (-9 if slot % 2 == 0 else 9) if len(sharing) > 1 else 0
            dy = (-9 if slot < 2 else 9) if len(sharing) > 1 else 0
            radius = 13 if len(sharing) > 1 else 16
            font = 13 if k < 9 else 11 if k < 99 else 9
            parts.append(f'<circle cx="{x+dx}" cy="{y+dy}" r="{radius}" fill="{agent_color(k)}" stroke="white" '
                         f'stroke-width="3"/><text x="{x+dx}" y="{y+dy+5}" text-anchor="middle" fill="white" '
                         f'font-size="{font}" font-weight="700">{k+1}</text>')
    parts.append("</svg>")
    legend = [f'<span style="color:{agent_color(k)}">● {html.escape(a["name"])} · {status.replace("_", " ")}</span>'
              for k, (a, status) in enumerate(zip(ep.agents, statuses_after(ep, index)))]
    if exits is not None:
        legend.append('<span>★ Exits ' + ", ".join(exits) + '</span>')
    else:
        legend.append('<span>★ Destination</span>')
    if reward is not None:
        legend.append(f'<span style="color:#d97706">○ Reward exit {reward}</span>')
    if getattr(ep, "rewarded", False) and ep.config["laps"] > 1:
        begun = lap_start(ep, index)
        legend.append(f'<span>Lap {1 + sum(start <= begun for start in ep.lap_rounds)} of {ep.config["laps"]}</span>')
    if checkpoint is not None:
        legend.append(f'<span style="color:#0891b2">□ Required checkpoint {tuple(checkpoint)}</span>')
    if steer_cell is not None:
        legend.append(f'<span style="color:#9333ea">□ Steering cell {tuple(steer_cell)}</span>')
    if waypoint is not None:
        legend.append('<span style="color:#0f766e">⚑ Waypoint</span>')
    if ep.config.get("interruption_text", "").strip():
        legend.append('<span style="color:#b77906">○ Interruption</span>')
    if ep.map_changes:
        legend.append('<span style="color:#78350f">▪ Closed during the run</span>')
    if ep.config.get("context_inserts"):
        legend.append('<span style="color:#db2777">◌ Inserted message · → advised direction</span>')
    parts.append('<div class="maze-legend">' + "".join(legend) + '</div>')
    return "".join(parts)


def team_status(ep):
    config = ep.config
    partial = 0
    if ep.turns and ep.turns[-1]["finish_reason"] is None:
        turn = ep.turns[-1]
        partial = len(turn["metrics"]) - turn.get("forced_prefix_tokens", 0)
    calls = f"at most {config['attempt_budget']} calls per agent · " if "attempt_budget" in config else ""
    return (f"**{'Replay · ' if ep.replay_only else ''}{ep.phase.title()}** · {html.escape(ep.detail)}\n\n"
            f"{len(ep.agents)} agents · **Communication:** {'On' if config['communication'] else 'Off'} · "
            f"**Team goal:** {TEAM_GOALS[config['team_goal']]}\n\n"
            f"{ep.maze.size} × {ep.maze.size} · shortest route {len(ep.maze.route()) - 1} moves · "
            f"{ep.rounds} of {config['round_limit']} rounds · {ep.moves - ep.supplied_moves * len(ep.agents)} model moves"
            + (f" + {ep.supplied_moves} supplied each" if ep.supplied_moves else "") + " · "
            + (f"{ep.sampled_tokens + partial:,} of {config['token_budget']:,} sampled tokens · " if "token_budget" in config
               else f"{ep.sampled_tokens + partial:,} sampled tokens, at most {config['agent_token_budget']:,} per agent · ")
            + f"{ep.tool_attempts} calls · {calls}"
            f"{len(ep.mail)} message{'' if len(ep.mail) == 1 else 's'}\n\n"
            f"**Goal information:** {GOAL_MODES[config['goal_mode']]} · "
            f"**Model:** {html.escape(ep.model_id or 'load one on the Models page')}"
            + interruption_status(ep) + steering_status(ep) + reward_status(ep))


def effective_norm(vector, strength):
    """The length of what steering adds to the residual stream: the vector's length times the strength it is added at."""
    return math.sqrt(sum(value * value for value in vector.get("vector", ()))) * abs(strength)


def reward_status(ep):
    """A run with exits and rewards: its exits, its reward, its taste and laps, and where each agent arrived.

    Each exit is given with its route length from the start, and two exits
    with the cell their shortest routes part at, so a choice between them can
    be read against the walk each one asks for. Each steered kind of response
    is given with the norm of what it adds, so a random vector can be checked
    against the one it stands in for.
    """
    if not getattr(ep, "rewarded", False):
        return ""
    config = ep.config
    exits = ep.exits or {"A": ep.maze.goal}
    distances = ep.maze.distances(ep.maze.start)
    parts = [("Exits " + ", ".join(f"{label} {tuple(cell)} at {distances[tuple(cell)]} moves"
                                   for label, cell in exits.items())) if ep.exits
             else f"Destination {tuple(ep.maze.goal)} at {distances[ep.maze.goal]} moves"]
    if ep.exits and len(exits) == 2:
        cell, steps = parting_cell(ep.maze, *exits.values())
        parts.append(f"routes part at {tuple(cell)}, {steps} move{'' if steps == 1 else 's'} in"
                     + (" (drawn at one route length)" if config["paired_exits"] else ""))
    vector = config.get("steering")
    if config["reward_exit"] is not None:
        # Labeled as steers_next steers them: a vector switched off or at
        # strength 0 leaves these responses unsteered.
        label = ("steered" if steering_active(config)
                 else "unsteered" if vector else "unsteered, no vector")
        norm = f" · norm {effective_norm(vector, vector['strength']):.3g}" if steering_active(config) else ""
        parts.append(f"reward at {config['reward_exit']} · {config['arrival_responses']} {label} "
                     f"response{'' if config['arrival_responses'] == 1 else 's'} after arriving there{norm}")
        if config["team_reward"]:
            count = f"{config['team_reward']} response{'' if config['team_reward'] == 1 else 's'}"
            parts.append(f"each arrival there also steers the next {count} of every teammate still moving"
                         if steering_active(config) else f"team reward of {count}, unsteered")
    if config["arrival_responses"]:
        parts.append(f"{config['arrival_responses']} response{'' if config['arrival_responses'] == 1 else 's'} "
                     "after every arrival")
    if config["taste"]:
        parts.append(f"taste at strength {config['taste_strength']:g} · norm "
                     f"{effective_norm(vector, config['taste_strength']):.3g}"
                     if vector and vector.get("enabled", True) and config["taste_strength"] else "taste, unsteered")
    lines = ["**Exits and rewards:** " + " · ".join(parts), f"**Laps:** {ep.lap} of {config['laps']}"]
    by_agent = {index: [] for index in range(len(ep.agents))}
    for event in ep.events:
        if event["arrived"] and event["source"] == "model":
            by_agent[event["agent"]].append(event.get("exit", "A"))
    for index, agent in enumerate(ep.agents):
        reached = " → ".join(by_agent[index]) or "no exit yet"
        steered = sum(1 for turn in ep.turns if turn["agent"] == index and turn.get("steered"))
        lines.append(f"**{html.escape(agent['name'])}:** exits {reached} · {steered} steered responses")
    return "\n\n" + "\n\n".join(lines)


def interruption_status(ep):
    """How each agent the run interrupts has done since, for a team given an interruption.

    Only the agents it reaches are listed: the rest never get it, so a line
    saying they have not been interrupted yet would describe a condition the
    run was not under.
    """
    if not ep.config.get("interruption_text", "").strip():
        return ""
    lines = []
    reached = ep.config.get("interrupt_agents") or range(len(ep.agents))
    for agent in (ep.agents[index] for index in reached):
        if not agent["interrupted"]:
            state = "not interrupted" if ep.phase in TERMINAL else "not interrupted yet"
        elif agent["resumed"] is None:
            state = "interrupted, no move since yet"
        elif agent["resumed"]:
            state = (f"interrupted, moved again after {agent['latency']:,} sampled tokens"
                     + (", toward the destination" if agent["first_move_progress"] else ""))
        else:
            state = "interrupted, did not move again"
        lines.append(f"{agent['name']} {state}")
    window = f"{ep.config['recovery_tokens']:,} sampled tokens / {ep.config['recovery_attempts']} attempts"
    return f"\n\n**Interruption** (recovery window {window}): " + " · ".join(lines)


def steering_status(ep):
    checkpoint = ep.config.get("required_checkpoint")
    vector = ep.config.get("steering")
    waypoint = ep.config.get("waypoint")
    if checkpoint is None and vector is None and waypoint is None:
        return ""
    lines = []
    if checkpoint is not None:
        lines.append(f"**Required checkpoint:** {tuple(checkpoint)} · every route to the destination crosses it.")
    if waypoint is not None:
        lines.append(f"**Waypoint:** {tuple(waypoint)} · each agent is told whether it has passed it.")
    if vector is not None and getattr(ep, "rewarded", False):
        return ""
    if vector is not None:
        when = ep.config["steer_when"]
        trigger = f"cell {tuple(when['cell'])}" if "cell" in when else f"{when['moves']} accepted moves"
        targets = ep.config.get("steer_agents") or range(len(ep.agents))
        lines.append(f"**Steering:** {trigger} · {ep.config['steer_responses'] or 'all remaining'} responses per agent · "
                     + ", ".join(ep.agents[i]["name"] for i in targets))
    for index, agent in enumerate(ep.agents):
        turns = [t for t in ep.turns if t["agent"] == index and t.get("steered")]
        events = [e for e in ep.events if e["agent"] == index]
        parts = []
        if checkpoint is not None:
            reached = any(e["accepted"] and list(e["after"]) == list(checkpoint) for e in events)
            parts.append("checkpoint reached" if reached else "checkpoint not reached")
        if waypoint is not None:
            parts.append("waypoint reached" if ep.waypoint_turn_of(index) is not None else "waypoint not reached")
        if vector is not None:
            onset = f", first in round {turns[0]['round'] + 1}" if turns else ""
            parts.append(f"{len(turns)} steered responses{onset}")
        lines.append(f"**{agent['name']}:** " + " · ".join(parts))
    return "\n\n" + "\n\n".join(lines)


def team_history_rows(ep):
    """The team's history as the table shows it, each row with the response selecting it shows.

    One row for the start, then one for each response in the order they were
    given, with a row of its own for a message inserted before one.
    """
    supplied = ep.supplied_moves
    rows = [(-1, ["Start", "All", str(tuple(positions_after(ep, -1)[0])), "—",
                  f"{supplied} supplied moves each" if supplied else "Initial position", "—"])]
    limit = ep.config.get("agent_token_budget")
    inserts = {(i["agent"], i["before_round"]): i for i in ep.config.get("context_inserts", ())}
    spent = [0] * len(ep.agents)
    laps = {start: number for number, start in enumerate(getattr(ep, "lap_rounds", ()), 2)}
    sent = {(m["round"], m["sender"]): m["text"] for m in ep.mail if m.get("after_arrival")}
    arrived_at = {}
    for index, turn in enumerate(ep.turns):
        name = ep.agents[turn["agent"]]["name"]
        if turn["round"] in laps and turn is next(t for t in ep.turns if t["round"] == turn["round"]):
            rows.append((index, [f"Round {turn['round'] + 1}", "All", str(tuple(ep.maze.start)), "—",
                                 f"Lap {laps[turn['round']]} begins", "—"]))
        kind = turn.get("kind", "move")
        if kind != "move":
            spent[turn["agent"]] += turn.get("sampled_tokens", 0)
            what = "Taste" if kind == "taste" else f"After arriving at {arrived_at.get(turn['agent'], 'A')}"
            if turn["finish_reason"] is None:
                what += " · generating…"
            elif turn["finish_reason"] != "stop":
                what += " · cut off"
            message = sent.get((turn["round"], name)) or (
                visible_text(turn["text"]).strip()[:MESSAGE_LIMIT] if kind == "taste" else "") or "—"
            rows.append((index, [f"Round {turn['round'] + 1}", name, str(tuple(turn["position_before"])), "—",
                                 what + (" · steered" if turn.get("steered") else ""), message]))
            continue
        if turn.get("event", {}).get("arrived"):
            arrived_at[turn["agent"]] = turn["event"].get("exit", "A")
        insert = inserts.get((turn["agent"], turn["round"]))
        if insert:
            sender = f" from {insert['sender']}" if insert.get("sender") else ""
            rows.append((index, [f"Round {turn['round'] + 1}", name, str(tuple(insert["position"])),
                                 f"advised {insert['advised_direction']}" if insert.get("advised_direction") else "—",
                                 f"Inserted {insert['channel'].replace('_', ' ')}{sender}", insert["text"]]))
        spent[turn["agent"]] += turn.get("sampled_tokens", 0)
        event = turn.get("event")
        position = tuple(event["after"]) if event else tuple(turn["position_before"])
        if event:
            result = "Accepted" if event["accepted"] else event["error"].replace("_", " ")
        elif turn.get("outcome"):
            # A cut-off that spent the last of its agent's limit was stopped by the limit.
            outcome = turn["outcome"]
            if outcome == "cut_off" and limit is not None and spent[turn["agent"]] >= limit:
                outcome = "out_of_tokens"
            result = OUTCOMES[outcome]
        else:
            result = "Generating…" if turn["finish_reason"] is None else "Waiting for the round"
        if ep.agents[turn["agent"]]["intervention_turn"] == index:
            result += " · interrupted"
        if turn.get("token_edit"):
            result += " · token edit"
        rows.append((index, [f"Round {turn['round'] + 1}", name, str(position), (event or {}).get("direction") or "—",
                             result + (" · steered" if turn.get("steered") else ""),
                             (event or {}).get("message") or "—"]))
    return rows


def team_timeline(ep):
    """The history's rows, the selected response's marked."""
    rows = team_history_rows(ep)
    marked = [list(row) for _, row in rows]
    selected = ep.selected_turn
    for place, (index, row) in enumerate(rows):
        if selected is not None and index == selected and not row[4].startswith("Inserted"):
            marked[place][0] = "▶ " + marked[place][0]
    return marked


def response_line(ep, index):
    """Whose a team response was, and how long its prompt had grown.

    The prompt grows with every round and, with messaging on, with every
    teammate's message, so its length is what shows a team outgrowing the
    model's context.
    """
    turn = ep.turns[index]
    name = ep.agents[turn["agent"]]["name"]
    kind = {"taste": " · Taste", "arrival": " · After arriving"}.get(turn.get("kind"), "")
    return (f"**Round {turn['round'] + 1} · {html.escape(name)}{kind}** · {len(turn.get('prompt_ids') or []):,} prompt tokens · "
            f"{turn.get('sampled_tokens', len(turn['metrics'])):,} sampled tokens"
            + (" · **Steered**" if turn.get("steered") else ""))


def as_text(value):
    """Model-written text, read as the characters it is.

    Quotes are left alone: Markdown shows an escaped apostrophe as the entity
    it was escaped to, and a quote is no markup in a paragraph.
    """
    return html.escape(value, quote=False).translate(MARKDOWN)


def statuses_after(ep, index):
    """Each agent's status as of round ``index``, so a replay's start is not told how it ended.

    An agent leaves the run in the round of its last response, so its final
    status holds from that round on and it was moving before.
    """
    last = {}
    for turn in ep.turns:
        last[turn["agent"]] = turn["round"]
    if getattr(ep, "lap_rounds", None) and index < ep.rounds - 1:
        # An earlier lap: arrived where the lap's moves say so, out where a response dropped the agent.
        begun, statuses = lap_start(ep, index), []
        for k, agent in enumerate(ep.agents):
            if last.get(k, index + 1) <= index:
                statuses.append(agent["status"])
            elif any(e["agent"] == k and e["arrived"] for e in lap_events(ep, index)):
                statuses.append("arrived")
            else:
                dropped = [t["outcome"] for t in ep.turns if t["agent"] == k and begun <= t["round"] <= index
                           and t.get("outcome") in DROPPED]
                statuses.append(DROPPED[dropped[-1]] if dropped else "active")
        return statuses
    return [agent["status"] if agent["status"] != "active" and last.get(k, index + 1) <= index else "active"
            for k, agent in enumerate(ep.agents)]


def mail_text(ep):
    if not ep.team:
        return ""
    if not ep.config["communication"]:
        return "Communication is off for this run. Agents were told they cannot reach their teammates."
    if not ep.mail:
        return "No messages yet."
    # The text is the model's, so it is read as the characters it is rather
    # than as Markdown that could restyle the pane around it.
    return "\n\n".join(f"**Round {m['round'] + 1} · {as_text(m['sender'])}"
                       f"{' · after arriving' if m.get('after_arrival') else ''}** → "
                       f"{as_text(', '.join(m['to'])) or 'nobody still moving'}: {as_text(m['text'])}"
                       for m in ep.mail)
