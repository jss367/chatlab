"""The Team tab: several agents in one maze, with or without talking to each other."""
from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path

import gradio as gr

from .page import MARKDOWN
from .maze import GOAL_MODES, SYSTEM, default_instruction, generate
from .team import MAX_AGENTS, TEAM_GOALS, TERMINAL, TeamEpisode, from_payload, stream_team
from extension_api import TokenInspector, icon_classes

TOKENS = TokenInspector()
logger = logging.getLogger(__name__)

# One colour per agent, none of them the destination's green.
COLORS = ("#4f46e5", "#db2777", "#0891b2", "#ea580c")
OUTCOMES = {"no_call": "No call · stopped", "cut_off": "Cut off · stopped", "not_applied": "Not applied"}


def positions_after(ep, index):
    """Where every agent stood after round ``index``, -1 being the start."""
    positions = [ep.maze.start] * len(ep.agents)
    for event in ep.events:
        if event["accepted"] and event["round"] <= index:
            positions[event["agent"]] = tuple(event["after"])
    return positions


def team_board(ep, index=None, reveal=False):
    index = ep.rounds - 1 if index is None else index
    maze = ep.maze
    positions = positions_after(ep, index)
    size, cell, pad = maze.size, 56, 28
    total = cell * size + 2 * pad

    def center(point):
        return pad + (point[1] + .5) * cell, pad + (point[0] + .5) * cell

    def points(path):
        return " ".join(f"{x},{y}" for x, y in map(center, path))

    where = "; ".join(f"{a['name']} at row {p[0]}, column {p[1]}" for a, p in zip(ep.agents, positions))
    parts = [f'<svg viewBox="0 0 {total} {total}" role="img" aria-label="{size} by {size} maze. {where}.">']
    for r, row in enumerate(maze.grid):
        for c, value in enumerate(row):
            fill = "#27344a" if value == "#" else "#fff"
            parts.append(f'<rect x="{pad+c*cell+2}" y="{pad+r*cell+2}" width="52" height="52" rx="7" fill="{fill}"/>')
    for i in range(size):
        parts.append(f'<text x="{pad+(i+.5)*cell}" y="17" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
        parts.append(f'<text x="12" y="{pad+(i+.5)*cell+4}" text-anchor="middle" fill="#7b8598" font-size="12">{i}</text>')
    if reveal:
        parts.append(f'<polyline points="{points(maze.route())}" fill="none" stroke="#b4bdcc" stroke-width="4" stroke-dasharray="3 8"/>')
    # Each agent's path is drawn a little off the cell centre, so two agents
    # walking the same corridor stay two lines.
    offsets = [((k % 2) * 2 - 1) * 5 * (k // 2 + 1) if len(ep.agents) > 1 else 0 for k in range(len(ep.agents))]
    for event in ep.events:
        if event["accepted"] and event["round"] <= index:
            k = event["agent"]
            (x1, y1), (x2, y2) = center(event["before"]), center(event["after"])
            d = offsets[k]
            parts.append(f'<line x1="{x1+d}" y1="{y1+d}" x2="{x2+d}" y2="{y2+d}" stroke="{COLORS[k]}" '
                         'stroke-width="5" stroke-linecap="round" opacity=".75"/>')
    x, y = center(maze.start)
    parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#64748b" font-size="14" font-weight="700">S</text>')
    x, y = center(maze.goal)
    parts.append(f'<circle cx="{x}" cy="{y}" r="17" fill="#d1fae5"/><text x="{x}" y="{y+7}" text-anchor="middle" font-size="23" fill="#047857">★</text>')
    # Agents sharing a cell are fanned out around it rather than stacked.
    for k, position in enumerate(positions):
        sharing = [j for j, other in enumerate(positions) if other == position]
        x, y = center(position)
        if len(sharing) > 1:
            slot = sharing.index(k)
            x += (-9 if slot % 2 == 0 else 9)
            y += (-9 if slot < 2 else 9)
        radius = 13 if len(sharing) > 1 else 16
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{COLORS[k]}" stroke="white" stroke-width="3"/>'
                     f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="white" font-size="13" font-weight="700">{k+1}</text>')
    parts.append("</svg>")
    legend = [f'<span style="color:{COLORS[k]}">● {html.escape(a["name"])} · {status.replace("_", " ")}</span>'
              for k, (a, status) in enumerate(zip(ep.agents, statuses_after(ep, index)))]
    legend.append('<span>★ Destination</span>')
    parts.append('<div class="maze-legend">' + "".join(legend) + '</div>')
    return "".join(parts)


def team_status(ep):
    config = ep.config
    partial = 0
    if ep.turns and ep.turns[-1]["finish_reason"] is None:
        partial = len(ep.turns[-1]["metrics"])
    return (f"**{'Replay · ' if ep.replay_only else ''}{ep.phase.title()}** · {html.escape(ep.detail)}\n\n"
            f"{len(ep.agents)} agents · **Communication:** {'On' if config['communication'] else 'Off'} · "
            f"**Team goal:** {TEAM_GOALS[config['team_goal']]}\n\n"
            f"{ep.maze.size} × {ep.maze.size} · shortest route {len(ep.maze.route()) - 1} moves · "
            f"{ep.rounds} of {config['round_limit']} rounds · {ep.moves} moves · "
            f"{ep.sampled_tokens + partial:,} of {config['token_budget']:,} sampled tokens · {ep.tool_attempts} calls · "
            f"{len(ep.mail)} message{'' if len(ep.mail) == 1 else 's'}\n\n"
            f"**Goal information:** {GOAL_MODES[config['goal_mode']]} · "
            f"**Model:** {html.escape(ep.model_id or 'load one on the Models page')}")


def team_timeline(ep):
    """One row for the start, then one for each response, in the order they were given."""
    rows = [["Start", "All", str(tuple(ep.maze.start)), "—", "Initial position", "—"]]
    for turn in ep.turns:
        event = turn.get("event")
        position = tuple(event["after"]) if event else tuple(turn["position_before"])
        if event:
            result = "Accepted" if event["accepted"] else event["error"].replace("_", " ")
        elif turn.get("outcome"):
            result = OUTCOMES[turn["outcome"]]
        else:
            result = "Generating…" if turn["finish_reason"] is None else "Waiting for the round"
        rows.append([f"Round {turn['round'] + 1}", ep.agents[turn["agent"]]["name"], str(position),
                     (event or {}).get("direction") or "—", result, (event or {}).get("message") or "—"])
    if ep.selected_turn is not None and 0 <= ep.selected_turn < len(ep.turns):
        rows[ep.selected_turn + 1][0] = "▶ " + rows[ep.selected_turn + 1][0]
    return rows


def as_text(value):
    """Model-written text, read as the characters it is.

    Quotes are left alone: Markdown shows an escaped apostrophe as the entity
    it was escaped to, and a quote is no markup in a paragraph.
    """
    return html.escape(value, quote=False).translate(MARKDOWN)


def statuses_after(ep, index):
    """Each agent's status as of round ``index``, so a replay's start is not told how it ended."""
    statuses = ["active"] * len(ep.agents)
    for turn in ep.turns:
        if turn["round"] > index:
            continue
        event = turn.get("event")
        if event and event["arrived"]:
            statuses[turn["agent"]] = "arrived"
        elif turn.get("outcome") in ("no_call", "cut_off"):
            statuses[turn["agent"]] = "abandoned" if turn["outcome"] == "no_call" else "cut_off"
    return statuses


def mail_text(ep):
    if not ep.config["communication"]:
        return "Communication is off for this run. Agents were told they cannot reach their teammates."
    if not ep.mail:
        return "No messages yet."
    # The text is the model's, so it is read as the characters it is rather
    # than as Markdown that could restyle the pane around it.
    return "\n\n".join(f"**Round {m['round'] + 1} · {as_text(m['sender'])}** → "
                       f"{as_text(', '.join(m['to'])) or 'nobody still moving'}: {as_text(m['text'])}"
                       for m in ep.mail)


def transport_text(ep):
    index = max(-1, min(ep.viewing, ep.rounds - 1))
    mode = "Generating" if ep.busy else "Replaying" if ep.playing else "Paused" if ep.turns else "Ready"
    if ep.busy and ep.pause_requested:
        mode = "Pausing after the round"
    selected = "Start" if index < 0 else f"Round {index + 1} of {ep.rounds}"
    if ep.replay_only:
        end = "Saved replay"
    elif ep.phase in TERMINAL:
        end = ep.phase.title()
    else:
        end = "Live end · Next runs a round" if index == ep.rounds - 1 else "Play continues at the live end"
    return f"**{mode}** · {selected}\n\n{end}"


def response_view(ep):
    """The selected response: whose it was, its tokens and its text."""
    index = ep.selected_turn
    if index is None or not 0 <= index < len(ep.turns):
        return "Select a response in the history to read it.", [], ""
    turn = ep.turns[index]
    name = ep.agents[turn["agent"]]["name"]
    return (f"**Round {turn['round'] + 1} · {html.escape(name)}** · {turn.get('sampled_tokens', len(turn['metrics'])):,} sampled tokens",
            TOKENS.strip(turn["metrics"]), turn["text"])


def context_text(ep, models):
    """The conversation the selected response was given, as that agent saw it.

    Decoded from the recorded prompt IDs only by the model that recorded them,
    for the reason the single-agent pane gives: another vocabulary spells the
    same numbers as fluent text unrelated to the prompt.
    """
    index = ep.selected_turn
    if index is None or not 0 <= index < len(ep.turns):
        return "Select a response in the history first.", ""
    turn = ep.turns[index]
    recorded = turn.get("model_id") or ep.model_id
    name = ep.agents[turn["agent"]]["name"]
    if turn.get("prompt_ids") and recorded and models.loaded_model_id() == recorded:
        try:
            text, load_id = models.decode(turn["prompt_ids"])
        except Exception:
            text, load_id = None, None
        if text is not None and (load_id or "").rsplit("#", 1)[0] == recorded:
            return (f"**Round {turn['round'] + 1} · {html.escape(name)} · as recorded** · "
                    f"{len(turn['prompt_ids']):,} prompt tokens, decoded.", text)
    messages = ep.context_messages(index)
    parts = [f"[tool schemas]\n{json.dumps(ep.tools, indent=2)}"]
    parts += [f"[{m['role']}]\n{m['content']}" for m in messages]
    return (f"**Round {turn['round'] + 1} · {html.escape(name)} · as recorded, untemplated** · {len(messages)} messages "
            f"and the move tool. Load {html.escape(recorded or 'the model that recorded this run')} to read the prompt "
            "through its own template.", "\n\n".join(parts))


def transport_buttons(ep):
    active = ep.playing or ep.busy
    return gr.update(visible=not active), gr.update(visible=active)


def stop_replay(ep):
    with ep.lock:
        ep.playback_token += 1
        ep.playing = False


def build_team_page(context, runs_dir):
    initial = TeamEpisode(generate(), dict(openness=.7))
    episode = gr.State(initial)
    with gr.Row(elem_id="team-workspace"):
        with gr.Column(elem_id="team-scenario"):
            gr.Markdown("## Team")
            models = gr.Button("Choose / load model", size="sm")
            with gr.Accordion("Load a saved team run", open=False):
                upload = gr.File(label="Saved team run JSON", show_label=False, file_types=[".json"], type="filepath")
            gr.Markdown("One loaded model plays every agent, each in its own conversation. Every round, each agent still "
                        "moving gives one response against the state as the round began, then all the moves land together.")
            prepare = gr.Button("New team episode · apply settings", elem_id="team-prepare")
            with gr.Row():
                agents = gr.Slider(2, MAX_AGENTS, value=2, step=1, label="Agents", elem_id="team-agents")
                team_goal = gr.Dropdown(choices=[(label, key) for key, label in TEAM_GOALS.items()], value="any",
                                        label="Team goal", elem_id="team-goal")
            communication = gr.Checkbox(value=True, label="Agents can message each other", elem_id="team-communication",
                                        info="Adds an optional message argument to the move call. Each message reaches every "
                                             "teammate in their next simulator reply. Off, the agents are told they cannot "
                                             "communicate. Agents never see where their teammates stand.")
            with gr.Row():
                size = gr.Slider(3, 15, value=5, step=1, label="Maze size")
                distance = gr.Number(value=10, precision=0, minimum=1, maximum=224, label="Shortest route length")
            with gr.Row():
                seed = gr.Number(value=20260911, precision=0, label="Maze seed")
                openness = gr.Slider(.35, .95, value=.7, step=.05, label="Open cells")
            goal_mode = gr.Dropdown(choices=[(label, mode) for mode, label in GOAL_MODES.items()], value="coordinates",
                                    label="Goal information", elem_id="team-goal-mode")
            goal_hint = gr.Textbox(label="Goal hint", lines=2, visible=False, elem_id="team-goal-hint")
            with gr.Accordion("Setup prompt", open=False):
                system_prompt = gr.Textbox(value=SYSTEM, label="System prompt", lines=2)
                instruction = gr.Textbox(value=default_instruction("coordinates"), label="Task instruction", lines=6,
                                         info="Each agent is sent this, then a paragraph naming it, its teammates, the team "
                                              "goal and whether it can communicate, then the JSON state.")
            with gr.Accordion("Generation limits", open=False):
                temperature = gr.Slider(0, 2, value=.7, step=.05, label="Sampling temperature")
                sampling_seed = gr.Number(value=20260914, precision=0, minimum=0, maximum=2147483647, label="Sampling seed",
                                          info="Each agent samples under its own seed derived from this one.")
                per_turn = gr.Number(value=1024, precision=0, minimum=1, maximum=8192, label="Tokens per response")
                budget = gr.Number(value=16384, precision=0, minimum=1, maximum=131072, label="Team sampled-token limit")
                round_limit = gr.Number(value=24, precision=0, minimum=1, maximum=256, label="Round limit")
            with gr.Accordion("Export the run", open=False):
                save = gr.Button("Export run JSON", size="sm")
                download = gr.File(label="Saved run", interactive=False)
        with gr.Column(elem_id="team-center"):
            maze_board = gr.HTML(team_board(initial), elem_id="team-board")
            transport_status = gr.Markdown(transport_text(initial), elem_id="team-transport-status")
            with gr.Row(elem_id="team-transport"):
                first = gr.Button("First", size="sm", elem_id="team-first", elem_classes=icon_classes("chevron-first"))
                back = gr.Button("Previous", size="sm", elem_id="team-previous", elem_classes=icon_classes("chevron-left"))
                toggle = gr.Button("Play", variant="primary", size="sm", elem_id="team-run", elem_classes=icon_classes("play"))
                pause = gr.Button("Pause", variant="primary", size="sm", visible=False, elem_id="team-pause",
                                  elem_classes=icon_classes("pause"))
                forward = gr.Button("Next", size="sm", elem_id="team-next",
                                    elem_classes=icon_classes("chevron-right", trailing=True))
            with gr.Accordion("Messages between agents", open=True):
                mail = gr.Markdown(mail_text(initial), elem_id="team-mail")
            with gr.Accordion("Playback & view", open=False):
                pace = gr.Slider(.1, 4, value=1., step=.1, label="Seconds per recorded round")
                reveal = gr.Checkbox(label="Show shortest route (viewer only)", value=False)
                gr.Markdown("Next runs one round at the live end. Pause lets the current round finish. Stop discards the "
                            "unfinished round: its responses are kept and none of its moves are applied.")
                stop = gr.Button("Stop now · end episode", size="sm", elem_id="team-stop")
        with gr.Column(elem_id="team-inspector"):
            gr.Markdown("## Responses\nSelect a row to read that response and show the board after its round.")
            history = gr.Dataframe(value=team_timeline(initial),
                                   headers=["Round", "Agent", "Position", "Direction", "Result", "Message"],
                                   interactive=False, wrap=True, elem_id="team-history")
            response_note = gr.Markdown("Select a response in the history to read it.")
            strip = gr.HighlightedText(label="Emitted tokens", color_map=context.tokens.color_map,
                                       combine_adjacent=False, show_legend=True, elem_id="team-tokens")
            with gr.Accordion("Full response", open=False):
                raw = gr.Textbox(label="Full response", interactive=False, lines=6, max_lines=12, elem_id="team-raw")
            with gr.Accordion("Context sent to the agent", open=False) as context_pane:
                context_note = gr.Markdown("Open or refresh this to read the conversation behind the selected response.")
                context_refresh = gr.Button("Show the selected response's context", size="sm")
                context_body = gr.Textbox(label="Prompt", interactive=False, lines=10, max_lines=24, elem_id="team-context")
            with gr.Accordion("Run details", open=False):
                state_text = gr.Markdown(team_status(initial), elem_id="team-status")
    outputs = [maze_board, state_text, history, mail, response_note, strip, raw, transport_status, toggle, pause]

    def team_render(ep, show, index=None):
        if index is not None:
            ep.viewing = index
        note, tokens, text = response_view(ep)
        return (team_board(ep, None if ep.busy else ep.viewing, show), team_status(ep), team_timeline(ep),
                mail_text(ep), note, tokens, text, transport_text(ep), *transport_buttons(ep))

    controls = [agents, communication, team_goal, size, seed, distance, openness, goal_mode, goal_hint,
                system_prompt, instruction, temperature, sampling_seed, per_turn, budget, round_limit]

    def team_prepare_episode(ep, show, *values):
        if ep.busy:
            raise gr.Error("Stop or pause this team episode before starting another.")
        (count, talk, goal, n, s, d, o, mode, hint, system_text, instruction_text, temp, sample_seed, per,
         total, rounds) = values
        try:
            new = TeamEpisode(generate(n, s, d, o), dict(
                agents=int(count), communication=bool(talk), team_goal=goal, goal_mode=mode, goal_hint=hint,
                system_prompt=system_text, instruction=instruction_text, temperature=float(temp),
                sampling_seed=int(sample_seed), per_turn_tokens=int(per), token_budget=int(total),
                round_limit=int(rounds), openness=float(o)))
        except (ValueError, TypeError) as exc:
            logger.warning("Refused the settings for a new team episode: %s", exc)
            raise gr.Error(str(exc)) from exc
        logger.info("New team episode %s: %s agents, communication %s, %s goal, %s x %s maze, seed %s",
                    new.run_id, count, "on" if talk else "off", goal, n, n, s)
        stop_replay(ep)
        return (new, *team_render(new, show), None)

    def team_play(ep, show, single=False):
        last_board = None
        try:
            for current in stream_team(ep, context.models, single_step=single, save_dir=runs_dir()):
                rendered = list(team_render(current, show))
                if rendered[0] == last_board:
                    rendered[0] = gr.skip()
                else:
                    last_board = rendered[0]
                yield tuple(rendered)
        except ValueError as exc:
            logger.warning("Team run %s cannot generate: %s", ep.run_id, exc)
            gr.Warning(str(exc))
            yield team_render(ep, show)

    def team_play_back(ep, show, seconds):
        with ep.lock:
            if ep.busy:
                raise gr.Error("Pause the episode before playing it back.")
            ep.playback_token += 1
            token = ep.playback_token
            ep.playing = True
        start = max(-1, min(ep.viewing, ep.rounds - 1))
        try:
            for index in range(start, ep.rounds):
                if index > start:
                    remaining = max(.1, float(seconds))
                    while remaining > 0:
                        if ep.playback_token != token:
                            return
                        time.sleep(min(.05, remaining))
                        remaining = round(remaining - .05, 6)
                if ep.playback_token != token:
                    return
                ep.selected_turn = None
                yield team_render(ep, show, index)
            if ep.playback_token == token and not ep.replay_only and ep.phase not in TERMINAL:
                yield from team_play(ep, show)
        finally:
            if ep.playback_token == token:
                ep.playing = False
        if ep.playback_token == token:
            yield team_render(ep, show)

    def team_step(ep, show, target):
        if ep.busy:
            raise gr.Error("Pause the episode before stepping through rounds.")
        stop_replay(ep)
        ep.selected_turn = None
        return team_render(ep, show, max(-1, min(target, ep.rounds - 1)))

    def team_step_forward(ep, show):
        if ep.busy:
            raise gr.Error("Pause the episode before stepping through rounds.")
        stop_replay(ep)
        if ep.viewing < ep.rounds - 1:
            ep.selected_turn = None
            yield team_render(ep, show, ep.viewing + 1)
        elif not ep.replay_only and ep.phase not in TERMINAL:
            yield from team_play(ep, show, single=True)
        else:
            yield team_render(ep, show)

    def team_select_history(ep, show, evt: gr.SelectData):
        if ep.busy:
            raise gr.Error("Pause the episode before selecting a response.")
        stop_replay(ep)
        row = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if int(row) <= 0 or int(row) > len(ep.turns):
            ep.selected_turn = None
            return team_render(ep, show, -1)
        ep.selected_turn = int(row) - 1
        # The board after the response's round, or, for a round that never
        # resolved, the board the response was given.
        return team_render(ep, show, min(ep.turns[ep.selected_turn]["round"], ep.rounds - 1))

    def team_command(ep, kind):
        stop_replay(ep)
        if kind == "pause":
            ep.request_pause()
        else:
            ep.request_stop(runs_dir())
        logger.info("Team run %s: %s requested while %s", ep.run_id, kind, ep.phase)
        gr.Info("Pausing after the current round." if kind == "pause" and ep.busy else
                "Playback paused." if kind == "pause" else "Stopping; the unfinished round will not be applied.")
        return team_status(ep), transport_text(ep), *transport_buttons(ep)

    def team_export(ep):
        if ep.busy:
            raise gr.Error("Pause or stop the episode before exporting. Completed rounds are also autosaved.")
        if not ep.replay_only:
            try:
                ep.save(runs_dir())
            except OSError as error:
                logger.warning("Could not write team run %s to the archive: %s", ep.run_id, error)
                gr.Warning("The run archive could not be written. Providing a temporary download instead.")
        return str(ep.export())

    def team_load(path, ep, show):
        if ep.busy:
            raise gr.Error("Pause or stop this team episode before loading a replay.")
        if not path:
            return (gr.skip(),) * (len(outputs) + len(controls) + 1)
        try:
            if Path(path).stat().st_size > 50_000_000:
                raise ValueError("Run files must be smaller than 50 MB.")
            replay = from_payload(json.loads(Path(path).read_text()))
            rendered = team_render(replay, show, -1)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            logger.warning("Could not load the team run in %s: %s", path, exc)
            raise gr.Error(f"Could not load run: {exc}") from exc
        logger.info("Loaded team run %s from %s for replay: %s rounds, phase %s", replay.run_id, path,
                    replay.rounds, replay.phase)
        stop_replay(ep)
        config, maze = replay.config, replay.maze
        values = (config["agents"], config["communication"], config["team_goal"], maze.size, maze.seed,
                  len(maze.route()) - 1, config.get("openness", gr.skip()), config["goal_mode"],
                  gr.update(value=config["goal_hint"], visible=config["goal_mode"] == "hint"),
                  config["system_prompt"], config["instruction"], config["temperature"], config["sampling_seed"],
                  config["per_turn_tokens"], config["token_budget"], config["round_limit"])
        return (replay, *rendered, *values)

    def team_change_goal_mode(mode, wording):
        stock = wording in {default_instruction(m) for m in GOAL_MODES}
        return gr.update(visible=mode == "hint"), default_instruction(mode) if stock else gr.skip()

    def team_show_context(ep):
        return context_text(ep, context.models)

    toggle.click(team_play_back, [episode, reveal, pace], outputs, show_progress="hidden",
                 concurrency_limit=None, trigger_mode="multiple")
    prepare.click(team_prepare_episode, [episode, reveal, *controls], [episode, *outputs, download],
                  concurrency_id="maze-team-view", show_progress="hidden")
    first.click(lambda ep, show: team_step(ep, show, -1), [episode, reveal], outputs,
                concurrency_id="maze-team-view", show_progress="hidden")
    back.click(lambda ep, show: team_step(ep, show, ep.viewing - 1), [episode, reveal], outputs,
               concurrency_id="maze-team-view", show_progress="hidden")
    forward.click(team_step_forward, [episode, reveal], outputs, concurrency_id="maze-team-view", show_progress="hidden")
    command_outputs = [state_text, transport_status, toggle, pause]
    pause.click(lambda ep: team_command(ep, "pause"), episode, command_outputs, queue=False)
    stop.click(lambda ep: team_command(ep, "stop"), episode, command_outputs, queue=False)
    history.select(team_select_history, [episode, reveal], outputs, concurrency_id="maze-team-view", show_progress="hidden")
    reveal.input(lambda ep, show: team_render(ep, show), [episode, reveal], outputs, queue=False)
    goal_mode.input(team_change_goal_mode, [goal_mode, instruction], [goal_hint, instruction], queue=False)
    context_refresh.click(team_show_context, episode, [context_note, context_body], show_progress="hidden")
    context_pane.expand(team_show_context, episode, [context_note, context_body], show_progress="hidden")
    save.click(team_export, episode, download, show_progress="hidden")
    upload.upload(team_load, [upload, episode, reveal], [episode, *outputs, *controls],
                  concurrency_id="maze-team-view", show_progress="hidden")
    context.navigation.open_models(models)
