"""Episode controller: token generation is separate from authoritative maze movement.

An episode is one agent, or a team of them, in one maze. Either way it runs in
rounds, each asking every agent still moving for one response and then
applying the moves together, so a run of one agent is a round of one response
at a time and a team is the same loop over more conversations. What a team
adds to a round is in team.py.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
import threading
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .dynamic_maze import (FORMAT as CHANGING_FORMAT, ChangingMaze, check_closure, close_cell, load_maze,
                           maze_at_turn, validate_drops, validate_pending, validate_updates)
from .inserts import CHANNELS, FORMAT as INSERT_FORMAT, check_insert, render_insert
from .maze import SYSTEM, Maze, TOOLS, apply_call, default_instruction, initial_history, parse_call, unavoidable_cells
from .team import (DROPPED, FORMAT as TEAM_FORMAT, LEGACY_FORMAT as LEGACY_TEAM_FORMAT, LIMITED, MAX_AGENTS,
                   MESSAGE_LIMIT, agent_names, check_config as check_team_config, targeted, team_paragraph, team_tools)
from chatlab.extension_api import normalize_steering, write_private_text

# One line for each response and each episode outcome, so a run read in
# ChatLab.log afterwards says what it did rather than only that a model was
# asked for tokens. The token counts are here because the memory report beside
# them is read against them.
logger = logging.getLogger(__name__)

FORMAT = "chatlab-maze-run-1"
TERMINAL = {"arrived", "abandoned", "budget", "stopped", "error"}
# How long after an interruption a first accepted move still counts as recovery.
# The pilot's window, kept as the default so runs written before it was
# configurable are read under the window that scored them.
RECOVERY_DEFAULTS = {"recovery_tokens": 1024, "recovery_attempts": 4}


def reachable_before_arriving(maze, origin):
    """The cells a character at ``origin`` can walk to without arriving on the way.

    Arriving at the destination ends the run, so a cell whose every route
    passes through the destination is one the run can never reach, however
    open the map around it. The destination itself is left out for the same
    reason.
    """
    origin = tuple(origin)
    found, todo = {origin}, deque([origin])
    while todo:
        for cell in maze.neighbors(todo.popleft()).values():
            if cell not in found and cell != maze.goal:
                found.add(cell)
                todo.append(cell)
    return found


def checked_cell(value, maze, label):
    """One cell of ``maze`` the run can reach before arriving, as a list, or None."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError(f"The {label} names one cell as a row and a column.")
    if not maze.open(value):
        raise ValueError(f"The {label} must be an open cell inside the maze.")
    if tuple(value) == maze.goal:
        raise ValueError(f"The {label} cannot be the destination, because arriving there ends the run.")
    if tuple(value) not in reachable_before_arriving(maze, maze.start):
        raise ValueError(f"The {label} has to be reachable from the start without passing through the "
                         "destination, because arriving there ends the run.")
    return list(value)


def check_checkpoint(config, maze):
    """Validate a run's waypoint and steering trigger, normalizing them in place.

    A waypoint is a cell the run asks the model to pass through on its way.
    Steering adds an activation vector to the responses it covers: it starts
    with the first response generated once its trigger holds - the character
    standing on ``steer_when["cell"]``, or ``steer_when["moves"]`` accepted
    moves made, supplied ones included, as the interruption counts them - and
    covers ``steer_responses`` responses from there, 0 meaning every response
    to the end of the run. It starts once: a character that leaves the cell and
    comes back is not steered again. A run carries its vector whole rather
    than a reference into this machine's vector store, so an export reads
    back anywhere.
    """
    waypoint = checked_cell(config.get("waypoint"), maze, "waypoint")
    if waypoint is not None and tuple(waypoint) == maze.start:
        raise ValueError("The waypoint cannot be the start, because the run would begin having reached it.")
    if "waypoint" in config:
        config["waypoint"] = waypoint
    if config.get("steering") is None:
        if config.get("steer_when") is not None or config.get("steer_responses"):
            raise ValueError("A steering trigger needs a steering vector.")
        return
    vector = normalize_steering(config["steering"])
    if "vector" not in vector:
        raise ValueError("A maze run carries its steering vector whole. Import the vector file itself.")
    when = config.get("steer_when")
    if not isinstance(when, dict) or len(when) != 1 or set(when) - {"cell", "moves"}:
        raise ValueError("Say when steering starts: at a cell, or after a number of accepted moves.")
    if "cell" in when:
        # checked_cell reads None as no cell, which is right for an optional
        # waypoint and wrong here: a trigger at no cell would never fire, and
        # the run would read as steered while nothing steered it.
        if when["cell"] is None:
            raise ValueError("Name the cell steering starts at.")
        when = {"cell": checked_cell(when["cell"], maze, "steering cell")}
    elif type(when["moves"]) is not int or not 0 <= when["moves"] <= 255:
        raise ValueError("Steering after moves needs a whole number of moves from 0 to 255.")
    responses = config.get("steer_responses", 0)
    if type(responses) is not int or not 0 <= responses <= 256:
        raise ValueError("Steered responses must be a whole number from 0 to 256; 0 steers to the end of the run.")
    config.update(steering=vector, steer_when=dict(when), steer_responses=responses)


def steering_active(config):
    vector = config.get("steering")
    return bool(vector and vector.get("enabled", True) and vector.get("strength", 1) != 0)


def steered_at(config, start, index, position, moves):
    """Whether response ``index`` is steered, given the run as it stands before it.

    ``start`` is the first steered response so far, or None. Written as one
    function over the recorded quantities so generation, a fork rebuilding its
    earlier responses, and a saved run being read back all decide it the same way.
    """
    if not steering_active(config):
        return False
    if start is None:
        when = config["steer_when"]
        return list(position) == when["cell"] if "cell" in when else moves >= when["moves"]
    count = config["steer_responses"]
    return count == 0 or index - start < count


READY = "Ready. Play the episode or use Next to generate one response."
TEAM_READY = "Ready. Play runs rounds until the team finishes; Next runs one round."


def new_agent(name, position, messages):
    """One agent as a run starts it: moving, not yet interrupted, with the history its setup wrote."""
    return dict(name=name, position=tuple(position), status="active", messages=messages, interrupted=False,
                intervention_turn=None, intervention_tokens=0, intervention_attempts=0, resumed=None,
                first_move_progress=None, latency=None, interrupt_next=False, insert_next=None)


class AgentField:
    """A value each agent has, read and written as the run's own on a run of one agent.

    Every run kept these on the run itself before teams, and a run of one
    agent is still saved that way, so its reader, its writer and its page
    reach its only agent's value through the run. A team has no such value,
    only its agents'.
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, episode, owner=None):
        return self if episode is None else episode._solo()[self.name]

    def __set__(self, episode, value):
        episode._solo()[self.name] = value


@dataclass
class Episode:
    """One run in one maze, by a single agent or by a team.

    ``config["agents"]`` says how many play. Without it the run has one agent,
    as every run did before teams, and is saved in the single-agent formats,
    whose turns and events name no agent: each reads as agent 0, and each
    turn's round is its index. A team's turns and events name their agent and
    round, and every list here holds the whole team's, in the order the
    rounds produced them.
    """
    maze: Maze
    config: dict
    run_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "ready"
    detail: str = ""
    # One entry per agent: its name, position, status and its own conversation.
    agents: list = field(default_factory=list)
    # Every attempted call, in the order the rounds applied them, and every
    # response, each run's supplied moves first.
    events: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    # Every message a teammate sent, with the teammates who received it.
    mail: list = field(default_factory=list)
    rounds: int = 0
    model_id: str | None = None
    load_id: str | None = None
    sampled_tokens: int = 0
    tool_attempts: int = 0
    supplied_moves: int = 0
    manual_intervention: bool = False
    # A cell queued to close before the next response, and every queued closure
    # that could not happen by the time its response came round.
    close_next: tuple = ()
    dropped_closures: list = field(default_factory=list)
    pause_requested: bool = False
    stop_requested: bool = False
    # Why the last write of this run failed, or None once it is on disk. Kept
    # apart from the detail, which is prose for a reader, so a caller that has
    # to know whether the file exists does not have to read it back.
    autosave_error: str | None = None
    busy: bool = False
    replay_only: bool = False
    token_edit: dict | None = None
    pending_edit: dict | None = None
    # A team round a fork left open: the actions of the teammates it kept
    # from the round the edited response was in, and the caps that round
    # began with. Not written to the run: a fork finishes it before it stops,
    # or discards it as any unfinished round is.
    open_round: dict | None = None
    # The agent and the history it had before the message a fork landed ahead
    # of its edited response, so the message can be taken back if that
    # response is never generated, as a live one is. Not written to the run.
    edit_insert: tuple | None = None
    # The response the viewer last drew, or for a team the round. Previous,
    # Next and playback move relative to it, so a rapid second click cannot
    # resend a stale index.
    viewing: int = -1
    # The team response selected in the round on screen.
    selected_turn: int | None = None
    # Whether a team round has begun, which it has from the moment its caps
    # are set, before its first response is appended: every agent in it
    # answers the state it began with, so whatever is queued then waits for
    # the next round. Not written to the run.
    round_open: bool = False
    # Which playback run owns the view. Starting one supersedes the last, so
    # two runs in the same session cannot repaint each other's frames.
    playback_token: int = 0
    playing: bool = False
    reveal_route: bool = False
    created_at: float = field(default_factory=time.time)
    # What each agent has its own of: its history, and its interruption and what came of it.
    messages = AgentField()
    interrupted = AgentField()
    intervention_turn = AgentField()
    intervention_tokens = AgentField()
    intervention_attempts = AgentField()
    resumed = AgentField()
    first_move_progress = AgentField()
    latency = AgentField()
    interrupt_next = AgentField()
    # A message queued to go into the context before the agent's next
    # response. Not written to the run: one still waiting when the episode
    # ends is dropped.
    insert_next = AgentField()

    def __deepcopy__(self, memo):
        # Gradio copies the initial State once per browser/API session. Each
        # session needs its own episode and lock; subsequent callbacks share it.
        result = object.__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, threading.RLock() if key == "lock" else copy.deepcopy(value, memo))
        if result.phase == "ready" and not result.turns:
            result.run_id = uuid4().hex
            result.created_at = time.time()
        return result

    def __post_init__(self):
        # Reentrant, because a run is written down on two paths: one that
        # already holds the lock, as stopping and autosaving do, and one that
        # does not, as the export button does. Both have to read the run as it
        # stands at one moment rather than field by field.
        self.lock = threading.RLock()
        self.config = copy.deepcopy(self.config)
        # A run of one agent names no count, as every run did before teams.
        # Only the integer 1 is that: True and 1.0 compare equal to it, and
        # are left for the team's own check to refuse.
        if type(self.config.get("agents")) is int and self.config["agents"] == 1:
            del self.config["agents"]
        team = "agents" in self.config
        if team:
            self.config = check_team_config(self.config)
        self.config.setdefault("goal_mode", "coordinates")
        self.config.setdefault("goal_hint", "")
        # Runs predating editable wording carry no prompt, so they keep the
        # defaults their goal mode sent. An empty string is a deliberate blank.
        if not isinstance(self.config.get("system_prompt"), str):
            self.config["system_prompt"] = SYSTEM
        if not isinstance(self.config.get("instruction"), str):
            self.config["instruction"] = default_instruction(self.config["goal_mode"])
        # Runs predating the recorded recovery window carry the window that
        # scored them, so an old export still says which one it was read under.
        # A window written down is used as written, or refused: silently
        # replacing it would score the run under a window it does not name.
        for key, fallback in RECOVERY_DEFAULTS.items():
            if key not in self.config:
                self.config[key] = fallback
            elif type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError("The recovery window must be a positive number of sampled tokens and tool attempts.")
        check_checkpoint(self.config, self.maze)
        checkpoint = checked_cell(self.config.get("required_checkpoint"), self.maze, "required checkpoint")
        if checkpoint is not None:
            if not team:
                raise ValueError("A required checkpoint is a team's. Give a run of one agent a waypoint instead.")
            if tuple(checkpoint) not in unavoidable_cells(self.maze):
                raise ValueError("The required checkpoint must be before the destination on every route from the start.")
            self.config["required_checkpoint"] = checkpoint
        # A team starts with none unless it asks, as every team did before it could.
        supplied = int(self.config.get("supplied_moves", 0 if team else 3))
        names = agent_names(self.config.get("agents", 1))
        self.agents, self.events = [], []
        for index, name in enumerate(names):
            instruction = self.config["instruction"]
            if team:
                instruction = "\n".join(filter(None, [instruction, team_paragraph(
                    name, names, self.config["team_goal"], self.config["communication"])]))
            messages, events, position = initial_history(
                self.maze, supplied, goal_mode=self.config["goal_mode"], goal_hint=self.config["goal_hint"],
                system=self.config["system_prompt"], instruction=instruction, waypoint=self.config.get("waypoint"),
                describe=(lambda state, name=name: self.framed(name, state)) if team else None)
            if team:
                for event in events:
                    event["agent"] = index
            self.agents.append(new_agent(name, position, messages))
            self.events += events
        self.supplied_moves = supplied
        self.detail = self.detail or (TEAM_READY if team else READY)

    @property
    def team(self):
        """Whether more than one agent plays this run."""
        return len(self.agents) > 1

    @property
    def names(self):
        return [agent["name"] for agent in self.agents]

    @property
    def tools(self):
        """The tools every response is offered: the move tool, carrying a message where a team may talk."""
        return team_tools(self.config["communication"]) if self.team else TOOLS

    def _solo(self):
        if self.team:
            raise AttributeError("A team has no single history, position or interruption. Read them from its agents.")
        return self.agents[0]

    # Kept a tuple, as the setup writes it, whatever a saved run spelled it as.
    @property
    def position(self):
        return self._solo()["position"]

    @position.setter
    def position(self, value):
        self._solo()["position"] = tuple(value)

    def round_turns(self, index):
        """The responses of round ``index``: one per agent still moving, or one for a run of one agent."""
        if not self.team:
            return self.turns[index:index + 1]
        return [turn for turn in self.turns if turn["round"] == index]

    def agent_tokens(self):
        """The tokens each agent has sampled so far, counted from its finished responses."""
        spent = [0] * len(self.agents)
        for turn in self.turns:
            spent[turn.get("agent", 0)] += turn.get("sampled_tokens", 0)
        return spent

    def response_caps(self, moving):
        """The most tokens each moving agent may sample this round, or None when the round cannot start.

        A run of one agent is capped by what is left of its limit. On a team,
        each agent is capped by what it has left of its own; a team run saved
        under one limit for the whole team splits what is left of it evenly
        before anyone answers, so an agent asked later in the round is capped
        exactly as the first one was.
        """
        per_turn = self.config["per_turn_tokens"]
        if not self.team:
            caps = dict.fromkeys(moving, min(per_turn, self.config["token_budget"] - self.sampled_tokens))
        elif "token_budget" in self.config:
            caps = dict.fromkeys(moving, min(per_turn, (self.config["token_budget"] - self.sampled_tokens) // len(moving)))
        else:
            spent = self.agent_tokens()
            caps = {i: min(per_turn, self.config["agent_token_budget"] - spent[i]) for i in moving}
        return None if not moving or min(caps.values()) <= 0 else caps

    @property
    def current_maze(self):
        """The map as it stands now: the original plus every closure recorded so far.

        Rebuilt from the original and the record rather than kept as state of
        its own, so the map the simulator moves on is the same one a replay of
        this run reconstructs and neither can drift from the other.
        """
        return maze_at_turn(self.maze, self.config.get("map_updates", ()), None, self.boundary_key)

    @property
    def map_changes(self):
        """Whether this run's walls can close while it is running."""
        return isinstance(self.maze, ChangingMaze)

    def agent_state(self, index, error=None, inbox=()):
        """What the simulator tells one agent: its own position, never its teammates'.

        Teammates' positions are left out on purpose. With communication off,
        an agent that could still see where the others stand would be
        coordinating through the board, and the comparison the switch exists
        for would measure nothing. A teammate's name, its team and its
        messages are all a team agent is told of the others; a run of one
        agent is told only the maze.
        """
        agent = self.agents[index]
        state = self.current_maze.state(agent["position"], error, goal_mode=self.config["goal_mode"],
                                        goal_hint=self.config["goal_hint"], waypoint=self.config.get("waypoint"),
                                        waypoint_reached=self.waypoint_turn_of(index) is not None)
        return self.framed(agent["name"], state, inbox) if self.team else state

    def framed(self, name, state, inbox=()):
        """A maze state as team agent ``name`` is told it: who it is, who its teammates are, and what they said."""
        others = [other for other in agent_names(self.config["agents"]) if other != name]
        state = {"agent": name, "teammates": others, **state}
        if self.config["communication"]:
            state["messages"] = list(inbox)
        return state

    def model_state(self, error=None):
        return self.agent_state(0, error)

    def waypoint_turn_of(self, index):
        """The response whose accepted move took agent ``index`` onto the waypoint, -1 for a supplied move, or None.

        Read off the path rather than kept, so it cannot disagree with the
        moves a saved run is checked against.
        """
        waypoint = self.config.get("waypoint")
        if waypoint is None:
            return None
        for event in self.events:
            if event.get("agent", 0) == index and event["accepted"] and list(event["after"]) == list(waypoint):
                return event.get("turn", -1) if event.get("source") == "model" else -1
        return None

    @property
    def waypoint_turn(self):
        """When a run of one agent reached its waypoint, as waypoint_turn_of says."""
        self._solo()
        return self.waypoint_turn_of(0)

    def agent_attempts(self, index):
        """The move calls agent ``index`` has had applied, which is every one it made in a round that resolved."""
        return sum(1 for event in self.events if event.get("agent", 0) == index and event["source"] == "model")

    def agent_moves(self, index):
        """The accepted moves agent ``index`` has made, its supplied ones included."""
        return sum(event["accepted"] for event in self.events if event.get("agent", 0) == index)

    @property
    def steer_turn(self):
        """The first steered response, or None while steering has not started."""
        return next((i for i, turn in enumerate(self.turns) if turn.get("steered")), None)

    def steers_next(self, index=0):
        """Whether agent ``index``'s next response is steered, judged on that agent's own responses and moves."""
        if index not in targeted(self.config, "steer_agents", len(self.agents)):
            return False
        turns = [turn for turn in self.turns if turn.get("agent", 0) == index]
        start = next((i for i, turn in enumerate(turns) if turn.get("steered")), None)
        return steered_at(self.config, start, len(turns), self.agents[index]["position"], self.agent_moves(index))

    @property
    def moves(self):
        return sum(e["accepted"] for e in self.events)

    def payload(self):
        if self.team:
            keys = ("run_id", "phase", "detail", "turns", "events", "mail", "rounds", "model_id", "load_id",
                    "sampled_tokens", "tool_attempts", "supplied_moves", "manual_intervention", "created_at",
                    "dropped_closures", "close_next", "token_edit")
            with self.lock:
                # A queued message is not written, as a run of one agent does not write its own.
                agents = [{key: value for key, value in agent.items() if key != "insert_next"} for agent in self.agents]
                return {"format": TEAM_FORMAT, "maze": self.maze.to_dict(), "config": self.config, "exploratory": True,
                        "agents": agents, **{key: getattr(self, key) for key in keys}}
        keys = ("run_id", "phase", "detail", "messages", "events", "turns", "position", "model_id", "load_id",
                "sampled_tokens", "tool_attempts", "supplied_moves", "interrupted", "intervention_turn",
                "intervention_tokens", "intervention_attempts", "resumed", "first_move_progress", "latency",
                "manual_intervention", "created_at", "token_edit", "pending_edit", "dropped_closures",
                "close_next")
        # One reading, not one per field. Queueing a closure marks the run as
        # intervened in and fills its queue together, and the close button runs
        # off Gradio's queue, so a snapshot taken field by field could catch the
        # two apart and write a run carrying an intervention while saying none
        # was made.
        with self.lock:
            # A run carrying an insertion is written under a format an older
            # ChatLab refuses, rather than one it would read with every later
            # response's context shifted.
            kind = INSERT_FORMAT if self.config.get("context_inserts") else CHANGING_FORMAT if self.map_changes else FORMAT
            return {"format": kind, "maze": self.maze.to_dict(), "config": self.config,
                    "exploratory": True, "tokenizer_note": "Every turn records its actual prompt IDs. Later turns are templated from the complete prior response text, including reasoning.",
                    **{k: getattr(self, k) for k in keys}}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        temp = path.with_suffix(".json.tmp")
        # Rendered under the lock, not merely read under it: the reading hands
        # back the run's own lists, so a stop landing between the reading and
        # the rendering would write a closure as pending and dropped at once.
        # The file itself is written outside, where only the disk is slow.
        with self.lock:
            text = json.dumps(self.payload(), ensure_ascii=False, allow_nan=False)
        write_private_text(temp, text)
        temp.replace(path)
        return path

    def export(self):
        """Gradio only serves returned files from permitted temporary locations."""
        return self.save(Path(tempfile.mkdtemp(prefix="chatlab-maze-")))

    def request_pause(self):
        self.pause_requested = True

    def request_stop(self, save_dir=None):
        with self.lock:
            if self.replay_only or self.phase in TERMINAL:
                return
            self.stop_requested = True
            if self.busy:
                return  # The active stream owns cleanup and persistence.
            self.phase = "stopped"
            self.detail = ("Stopped by you. This is not scored as the team giving up." if self.team else
                           "Stopped by you. This is not scored as model abandonment.")
            abandon_closure(self)
            abandon_insert(self)
            abandon_fork(self)
            if save_dir:
                try:
                    self.save(save_dir)
                    self.autosave_error = None
                except OSError as exc:
                    self.warn_autosave(str(exc))

    def warn_autosave(self, error):
        logger.warning("Autosave of run %s failed: %s", self.run_id, error)
        self.autosave_error = error
        self.detail += (f" Autosave failed: {error}. Latest changes remain in memory. "
                        "Use Export run JSON to download them, and check the run directory or free disk space.")

    def request_interruption(self, index=0):
        """Queue the interruption for agent ``index``'s next response, ahead of its own trigger."""
        if self.phase in TERMINAL or self.replay_only:
            raise ValueError("Start a new episode to request an interruption. This episode is finished or is a saved replay.")
        agent = self.agents[index]
        if index not in targeted(self.config, "interrupt_agents", len(self.agents)):
            raise ValueError(f"{agent['name']} is not one of the agents this run interrupts.")
        if agent["status"] != "active":
            raise ValueError(f"{agent['name']} has stopped moving, so no response of its is left to interrupt.")
        if agent["interrupted"]:
            raise ValueError(("This episode already contains its interruption." if not self.team else
                              f"{agent['name']} has already been interrupted.")
                             + " Start another run to compare settings.")
        if not self.config.get("interruption_text", "").strip():
            raise ValueError("Choose interruption text before starting this episode.")
        agent["interrupt_next"], self.manual_intervention = True, True

    @property
    def boundary_key(self):
        """What a closure or an inserted message names the point it landed at by.

        A run of one agent lands them between responses, and a team between
        rounds.
        """
        return "before_round" if self.team else "before_turn"

    def next_boundary(self):
        """The point the next closure or inserted message would land at.

        The round being generated, or about to be, is past: whatever is queued
        now lands before the one after it. On a run of one agent that is its
        next response.
        """
        return self.rounds + bool(self.round_open or self.round_turns(self.rounds))

    def moving_positions(self):
        """Where each agent still moving stands, by index, which a closure has to leave a way on from."""
        return {index: agent["position"] for index, agent in enumerate(self.agents) if agent["status"] == "active"}

    def request_closure(self, cell):
        """Queue one cell to be walled off before the next generated response, or a team's next round.

        Checked here against the map and the positions as they stand, so a cell
        that cannot be closed is refused where it was asked for. It is checked
        again when it lands, because the agents move in between.

        One at a time, as an interruption is. A second request would replace a
        queued cell that the reader has already been told will close, and the
        one it replaced would leave no mark anywhere: never applied, so not in
        map_updates, and never refused by the map, so not among the closures
        this run reports dropping.

        The whole check and the assignment are under the episode's lock, unlike
        the interruption's. The page registers the close button off the queue,
        so two clicks run as two callbacks at once, and an interruption is a
        flag both of them can set to the same True while a queued cell is a
        cell: read and written apart, the second click's test of an empty queue
        would pass against the first click's and one of them would go missing.
        """
        with self.lock:
            if not self.map_changes:
                raise ValueError("This episode's map is fixed. Start an episode with a changing map to close cells during a run.")
            if self.phase in TERMINAL or self.replay_only:
                raise ValueError("Start a new episode to change the map. This episode is finished or is a saved replay.")
            step = "round" if self.team else "response"
            if self.close_next:
                raise ValueError(f"Row {self.close_next[0]}, column {self.close_next[1]} is already queued to close "
                                 f"before the next {step}. Let it land before queueing another.")
            # One closure to a boundary. A fork of the response after a closure
            # starts holding that closure at the boundary it is about to
            # regenerate, and a run that wrote two there could not be read back.
            boundary = self.next_boundary()
            if any(record[self.boundary_key] == boundary
                   for record in (*self.config.get("map_updates", ()), *self.dropped_closures)):
                raise ValueError(f"The map already changed before this {step}. Generate it before closing another cell.")
            checked_closure(self, cell)
            self.close_next, self.manual_intervention = tuple(cell), True

    def request_insert(self, channel, text, sender=None, advised_direction=None, index=0):
        """Queue one message to go into agent ``index``'s context before its next generated response.

        The rules are the closure's. One is queued to an agent at a time, and a
        second request is refused naming the first, which the reader has
        already been told will land. One lands at a boundary, so a fork
        carrying an insertion at the boundary it is about to regenerate refuses
        another. The check and the assignment are one operation under the lock,
        because the button runs off Gradio's queue and two clicks arrive at once.
        """
        with self.lock:
            if self.phase in TERMINAL or self.replay_only:
                raise ValueError("Start a new episode to insert a message. This episode is finished or is a saved replay.")
            agent = self.agents[index]
            if agent["status"] != "active":
                raise ValueError(f"{agent['name']} has stopped moving, so it will read no more messages.")
            whose = f" for {agent['name']}" if self.team else ""
            if agent["insert_next"]:
                raise ValueError(f"{describe_insert(agent['insert_next'])} is already queued{whose} before the next "
                                 "response. Let it land before queueing another.")
            boundary = self.next_boundary()
            if any(record[self.boundary_key] == boundary and record.get("agent", 0) == index
                   for record in self.config.get("context_inserts", ())):
                raise ValueError(f"A message was already inserted{whose} before this response. "
                                 "Generate it before inserting another.")
            insert = check_insert(dict(channel=channel, text=text, sender=sender or None,
                                       advised_direction=advised_direction or None))
            # A response being generated now will be answered by the
            # simulator before this lands, so only an idle run is asked.
            if not self.busy:
                render_insert(agent["messages"], insert)
            # Named with the boundary it was queued for, so one queued while a
            # team round is generating waits for the next round rather than
            # reaching an agent this round asks later than its teammates.
            agent["insert_next"], self.manual_intervention = dict(insert, for_boundary=boundary), True


def describe_insert(insert):
    """An insertion named for a reader: its channel, its sender, and how it begins."""
    text = insert["text"] if len(insert["text"]) <= 40 else insert["text"][:39] + "…"
    sender = f" from {insert['sender']}" if insert.get("sender") else ""
    return f"The {CHANNELS[insert['channel']].lower()}{sender} “{text}”"


def land_insert(episode, insert):
    """Write one insertion into its agent's history and the record together. The caller holds the lock."""
    agent = episode.agents[insert.get("agent", 0)]
    agent["messages"] = render_insert(agent["messages"], insert)
    episode.config.setdefault("context_inserts", []).append(insert)


def apply_insert(episode, manager=None, index=0):
    """Put agent ``index``'s queued message into its context, if one is queued, and record where it landed.

    Called with a response about to be appended. Returns the history as it
    stood before, so the stream can withdraw the message if that response
    never reaches the model: a message is only kept on the record once a
    prompt holding it has been fed, as an interruption is only marked once
    its tokens have been. One that can no longer be rendered is dropped with
    a note in the detail rather than recorded, and None comes back, as is
    one ``manager``'s tokenizer reads a special token out of: templates
    tokenize message text with the specials parsed, so such a message would
    end or open a turn wherever it sits, whatever spelling the fixed list in
    inserts.py missed.
    """
    with episode.lock:
        agent = episode.agents[index]
        now = episode.rounds if episode.team else len(episode.turns)
        if not agent["insert_next"] or agent["insert_next"]["for_boundary"] > now:
            return None
        before = agent["messages"]
        queued, agent["insert_next"] = agent["insert_next"], None
        insert = {episode.boundary_key: episode.rounds if episode.team else len(episode.turns),
                  **(dict(agent=index) if episode.team else {}),
                  "channel": queued["channel"], "text": queued["text"], "sender": queued["sender"],
                  "position": list(agent["position"]), "advised_direction": queued["advised_direction"]}
        where = f"{agent['name']}'s response in round {episode.rounds + 1}" if episode.team \
            else f"response {len(episode.turns) + 1}"
        try:
            if manager is not None and any(set(manager.encode(value)) & manager.hidden_token_ids
                                           for value in (insert["text"], insert["sender"] or "")):
                raise ValueError("The loaded model reads part of the text or the sender as one of its special tokens.")
            land_insert(episode, insert)
        except ValueError as exc:
            episode.detail = f"{describe_insert(insert)} was dropped. {exc}"
            logger.warning("Run %s dropped the message queued before %s: %s", episode.run_id, where, exc)
            return None
        episode.detail = f"{describe_insert(insert)} went into the context before {'this response' if not episode.team else where}."
        logger.info("Run %s inserted a %s before %s at %s", episode.run_id, insert["channel"], where, insert["position"])
        return before


def withdraw_insert(episode, before, index=0):
    """Take back the message the last apply_insert landed, its response never having been generated.

    A stop at the opening frame, or a model call that fails before its first
    update, leaves a response the model never read a prompt for, so a record
    saying that response received the message would describe an
    intervention nobody made.
    """
    with episode.lock:
        withdrawn = episode.config["context_inserts"].pop()
        if not episode.config["context_inserts"]:
            del episode.config["context_inserts"]
        episode.agents[index]["messages"] = before
    logger.warning("Run %s withdrew the %s before %s %s: that response was never generated", episode.run_id,
                   withdrawn["channel"], "round" if episode.team else "response", withdrawn[episode.boundary_key] + 1)


def abandon_fork(episode):
    """Take back what a fork set up for a regeneration that will now never run. The caller holds the lock.

    A fork lands the message that went in before its edited response, and
    leaves that response's round open with the teammates it kept. Ended before
    the response is generated, neither happened as far as the record goes: no
    prompt read the message, and the round never resolved, so its kept
    responses are marked not applied as a stopped round's are.
    """
    if episode.edit_insert is not None:
        (index, before), episode.edit_insert = episode.edit_insert, None
        withdraw_insert(episode, before, index)
    if episode.open_round is not None:
        episode.open_round = None
        discard_round(episode)


def abandon_insert(episode):
    """Drop every queued message the run will never reach. The caller holds the lock."""
    for agent in episode.agents:
        if agent["insert_next"]:
            logger.warning("Run %s ended %s with a %s still queued for %s; it was dropped", episode.run_id,
                           episode.phase, agent["insert_next"]["channel"], agent["name"])
            agent["insert_next"] = None


def turn_round(episode, index):
    """The round of the response at ``index``, -1 standing for the supplied moves before any."""
    return index if index < 0 or not episode.team else episode.turns[index]["round"]


def check_checkpoint_closure(episode, cell, changed, boundary=None, positions=None):
    """Refuse a closure that takes away the waypoint or the steering cell.

    Neither is ever closed, so the board and the saved run always show the
    cell the run was set up around. Until an agent has reached one, a closure
    that walls that agent off from it is refused as well, since the run could
    no longer do what it was set up to test. A route left only through the
    destination counts as walled off, because arriving ends the agent's run
    before it gets there. An agent is only held to the steering cell if the
    run steers it.

    ``boundary`` and ``positions`` - each agent still moving, by index -
    default to the run as it stands, which is where a live closure lands. A
    saved run's closures are asked the same question at the boundary and
    positions each one records, so a file cannot carry a closure the run
    itself would have refused.
    """
    config = episode.config
    boundary = episode.next_boundary() if boundary is None else boundary
    positions = episode.moving_positions() if positions is None else positions
    checkpoints = []
    if config.get("waypoint") is not None:
        checkpoints.append(("waypoint", config["waypoint"]))
    when = config.get("steer_when") or {}
    if "cell" in when:
        checkpoints.append(("steering cell", when["cell"]))
    for label, point in checkpoints:
        if tuple(point) == tuple(cell):
            raise ValueError(f"The {label} is never closed.")
    steered = targeted(config, "steer_agents", len(episode.agents))
    for index, position in positions.items():
        reachable = reachable_before_arriving(changed, position)
        for label, point in checkpoints:
            if label == "waypoint":
                reached = episode.waypoint_turn_of(index)
            elif steering_active(config) and index in steered:
                reached = next((i for i, turn in enumerate(episode.turns)
                                if turn.get("agent", 0) == index and turn.get("steered")), None)
            else:
                continue
            pending = reached is None or turn_round(episode, reached) >= boundary
            if pending and tuple(point) not in reachable:
                whom = episode.agents[index]["name"] if episode.team else "the character"
                raise ValueError(f"Closing this cell would cut {whom} off from the {label}.")


def checked_closure(episode, cell, boundary=None, positions=None):
    """The map after closing ``cell``, or why that closure cannot happen now.

    Every agent still moving has to be able to reach the destination from
    where it stands afterwards, and none may be standing in the cell, which
    is the single-agent rule asked of each of them.
    """
    positions = episode.moving_positions() if positions is None else positions
    maze = maze_at_turn(episode.maze, episode.config.get("map_updates", ()), None, episode.boundary_key)
    for index, position in positions.items():
        try:
            check_closure(maze, position, cell)
        except ValueError as exc:
            if not episode.team:
                raise
            raise ValueError(f"{episode.agents[index]['name']}: {exc}".replace("the character", "this agent")) from None
    changed = close_cell(maze, cell)
    check_checkpoint_closure(episode, cell, changed, boundary, positions)
    return changed


def abandon_closure(episode):
    """Record a queued closure the run will never reach. The caller holds the lock.

    A queued cell is applied before the next response or round, so an episode
    that ends without one leaves the reader told a closure would happen and
    the run showing no sign that anything was asked. That is the same silence
    dropped_closures was added to break, so it is broken the same way.
    """
    if not episode.close_next:
        return
    cell, episode.close_next = tuple(episode.close_next), ()
    episode.dropped_closures.append({episode.boundary_key: episode.next_boundary(), "cell": list(cell),
                                     "reason": "The episode ended before the closure could land."})
    logger.warning("Run %s ended %s with the closure at %s still queued",
                   episode.run_id, episode.phase, cell)


def apply_closure(episode):
    """Wall off the queued cell, if there is one, and record what the map became.

    The agents have moved since the closure was queued, so the same rules are
    asked again here. One that has become impossible is dropped and recorded as
    dropped: the run went on under a map the reader asked to change and which
    did not change, and anything scoring the run needs to know that rather than
    read an unchanged map as an unchanged intention.

    Taking the cell, reading the map and writing the change are one operation,
    because emptying the queue ahead of recording the closure would leave a
    moment when request_closure sees a free queue beside a map that has not
    changed yet, and takes the same cell again. The run would then record
    closing a wall, which is a closure no map ever allowed and which the reader
    of that run would refuse.

    A run of one agent records where its character stood; a team, where every
    agent stood, the ones no longer moving included.
    """
    with episode.lock:
        if not episode.close_next:
            return
        cell, episode.close_next = tuple(episode.close_next), ()
        boundary = episode.next_boundary()
        try:
            changed = checked_closure(episode, cell)
        except ValueError as exc:
            episode.dropped_closures.append({episode.boundary_key: boundary, "cell": list(cell), "reason": str(exc)})
            episode.detail = f"The queued closure at row {cell[0]}, column {cell[1]} was dropped. {exc}"
            logger.warning("Run %s dropped the closure at %s before %s %s: %s", episode.run_id, cell,
                           "round" if episode.team else "response", boundary + 1, exc)
            return
        where = (dict(positions=[list(agent["position"]) for agent in episode.agents]) if episode.team
                 else dict(position=list(episode.position)))
        episode.config.setdefault("map_updates", []).append(
            {episode.boundary_key: boundary, **where, "closed_cell": list(cell), "grid": list(changed.grid)})
        episode.detail = f"The map changed: row {cell[0]}, column {cell[1]} is now a wall."
        logger.info("Run %s closed %s before %s %s", episode.run_id, cell,
                    "round" if episode.team else "response", boundary + 1)


def context_messages(episode, index):
    """The messages a response was given, or is about to be given at ``index`` -1.

    The history grows by exactly two messages - the assistant's response and
    the simulator's reply - for each response that attempted a call, and by
    none for one that ended without attempting anything, so a response's own
    prompt is the history up to the calls the responses before it made. The
    initial block is the setup plus one such pair per supplied move. A user
    message inserted at or before a response's boundary adds one more; a note
    or a teammate message is written into a reply already counted.

    Counted from the record rather than kept as an index of its own, so the
    two cannot disagree. On a team, the history is the answering agent's own:
    its setup, its supplied moves, a pair for each round it made a call in,
    and the user messages inserted for it.
    """
    inserts = episode.config.get("context_inserts", ())
    if episode.team:
        turn = episode.turns[index]
        agent, round_index = turn["agent"], turn["round"]
        pairs = sum(1 for event in episode.events if event["agent"] == agent
                    and (event["source"] == "supplied" or event["round"] < round_index))
        users = sum(insert["channel"] == "user" and insert["agent"] == agent and insert["before_round"] <= round_index
                    for insert in inserts)
        return episode.agents[agent]["messages"][:2 + 2 * pairs + users]
    supplied = sum(event["source"] == "supplied" for event in episode.events)
    attempts = sum("event" in turn for turn in episode.turns[:max(index, 0)])
    users = sum(insert["channel"] == "user" and insert["before_turn"] <= index for insert in inserts)
    return episode.messages[:2 + 2 * (supplied + attempts) + users]


# What reading a response adds to it, and so what a response rebuilt from a
# record is read without.
DERIVED = {"event", "outcome", "sampled_tokens", "tokens_cumulative"}


def mark_interruption(episode, index, turn):
    """Record that response ``turn`` of agent ``index`` opened with the interruption.

    Its recovery is counted from here: the tokens and calls it has made so far
    are where its window starts.
    """
    episode.agents[index].update(interrupted=True, intervention_turn=turn,
                                 intervention_tokens=episode.agent_tokens()[index],
                                 intervention_attempts=episode.agent_attempts(index))


def response_limit(episode, index, cap, interrupting):
    """The most tokens agent ``index``'s next response may sample.

    Its cap for the round, narrowed while it is inside a recovery window: the
    response that opens with the interruption gets no more than the whole
    window, and each one after it, until the agent moves again, what is left of
    it.
    """
    agent, window = episode.agents[index], episode.config["recovery_tokens"]
    if interrupting:
        return min(cap, window)
    if agent["interrupted"] and agent["resumed"] is None:
        return min(cap, window - (episode.agent_tokens()[index] - agent["intervention_tokens"]))
    return cap


def interrupted_prefix(episode, manager, index=0):
    """The interruption agent ``index``'s next response opens with, or none.

    Each agent the run interrupts is interrupted once: on its first response
    after its own accepted moves, supplied ones included, reach the trigger,
    or on the one it was queued for.
    """
    agent = episode.agents[index]
    if agent["interrupted"] or not episode.config.get("interruption_text", "").strip():
        return []
    if index not in targeted(episode.config, "interrupt_agents", len(episode.agents)):
        return []
    if not agent["interrupt_next"] and episode.agent_moves(index) < episode.config["interrupt_after"]:
        return []
    text = episode.config["interruption_text"]
    if any(mark in text for mark in ("<tool_call", "</tool_call", "<|im_", "<|endoftext|>", "<think>", "</think>", "```", "~~~")):
        raise ValueError("Interruption text cannot supply tool syntax, conversation boundary tokens, reasoning delimiters or code fences.")
    # forced_ids inserts into the actual next response, including inside an
    # already-open reasoning block. Unlike answer_prefill it adds no </think>.
    ids = manager.encode(text)
    count = int(episode.config["prefix_tokens"])
    return list(ids if count == 0 else ids[:count])


def forkable_without_evidence(episode, manager):
    """Whether this episode may be forked where the run records nothing to check.

    Only a live episode of this session qualifies: its load identifier was
    assigned by this process, so it names the load in memory now. An uploaded
    replay never qualifies, because load_count restarts at zero in each
    process, and the first load of a repository in one session answers to the
    same name as the first load in the next.
    """
    return (not episode.replay_only and manager.load_id is not None
            and manager.load_id == episode.load_id)


def visible_token_ids(metrics, literal_prefill_tokens, hidden_ids):
    """The IDs a stretch of recorded metrics was decoded into text through.

    The runtime streams each response through IncrementalDecoder, which drops
    a hidden special rather than decoding it, so those IDs appear among the
    turn's metrics and never in its text. Reader-supplied prefill is the
    exception the runtime makes: replay forces those tokens visible, including
    special-token spellings.
    """
    return [metric["token_id"] for index, metric in enumerate(metrics)
            if index < literal_prefill_tokens or metric["token_id"] not in hidden_ids]


def literal_prefill_of(turn):
    """How many leading tokens of this response replay forces visible."""
    return turn.get("literal_prefill_tokens", turn.get("forced_prefix_tokens", 0))


def verify_recorded_text(episode, turn_index, manager):
    """Refuse a fork whose stored IDs no longer decode to the text they recorded.

    The same repository ID can be re-downloaded at a revision whose tokenizer or
    vocabulary changed, so a matching model_id is not on its own evidence that
    replaying stored IDs reproduces the original run. Every response records the
    text its own IDs decoded to at generation time, so the loaded tokenizer can
    be checked against the run itself, with no fingerprint that existing exports
    never carried.

    Each response is compared as a whole rather than a token at a time, because
    decoding is not piecewise. A revision that moves an ID from the word-boundary
    piece "▁world" to "world" leaves it decoding alone to "world" either way,
    while the sequence after "Hello" reads "Hello world" under one vocabulary and
    "Helloworld" under the other. Decoding the response as a whole also puts a
    byte-level piece among the neighbours that complete its character, so a
    fragment that names no ID by itself is still pinned down by the text it
    makes with them.

    Every earlier response is verified in full, because the fork rebuilds each of
    them from its stored IDs. The edited response is verified in full too, past
    the edited token as well as before it: boundary semantics only mean anything
    in context, so a vocabulary that has moved anywhere in that response is
    evidence the tokenizer is not the one that produced the run.

    A load identifier is not evidence of anything here, because load_count
    restarts at zero in each process, so the first load of a repository in one
    session and its first load in the next both answer to the same name.
    """
    hidden = manager.hidden_token_ids
    for turn in episode.turns[:turn_index + 1]:
        try:
            current = manager.decode(visible_token_ids(turn["metrics"], literal_prefill_of(turn), hidden))
        except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError("The loaded model cannot decode this run's token IDs, so its tokenizer is not the "
                             f"one that produced the run ({exc}). This happens when the same model ID has been "
                             "re-downloaded at a different revision.") from exc
        if current != turn["text"]:
            raise ValueError("The loaded weights tokenize differently from the ones that produced this run, so "
                             "its tokens cannot be replayed. This happens when the same model ID has been "
                             "re-downloaded at a different revision; load that snapshot to fork this run.")


def verify_recorded_candidate(episode, manager):
    """Refuse a recorded alternative on a run this session did not produce.

    The chosen alternative is the one ID the fork replays that the run never
    generated, so no response text stands behind it, and nothing an export
    records says how the model that offered it spelled that ID. build_metric
    records an alternative by decoding it alone, and SentencePiece reads the
    word-boundary space off the first token of whatever it decodes, so "▁world"
    and "world" both record "world". A later vocabulary spelling that ID either
    way reproduces the recording exactly, while the branch after a retained
    "Hello" reads "Hello world" under one and "Helloworld" under the other. The
    recording is the same in both directions, so no comparison against the
    loaded vocabulary can establish which one offered the alternative, and the
    session that offered it is the only place it can be applied: its load
    identifier names the load in memory now.

    This costs nothing, because Replacement text expresses the same branch and
    is checked more strictly. ModelManager.encode_replacement validates typed
    text in place against the retained tokens, against exactly this ambiguity,
    and picks whichever ID spells it correctly in that position under the
    loaded vocabulary. Someone who wants the alternative "world" types "world"
    and gets a correctly spelled branch.
    """
    if not forkable_without_evidence(episode, manager):
        raise ValueError("A recorded alternative can only be applied in the session that offered it. The run "
                         "records how the alternative decodes on its own, which cannot say how the model that "
                         "offered it spelled that token after your retained tokens. Type the text you want in "
                         "Replacement text instead, which is checked in place against those tokens.")


def verify_visible_candidate(candidate_id, manager, stop_ids):
    """Refuse an alternative the loaded model never shows in a response.

    The replacement reaches the runtime as forced_ids past the literal prefill,
    where IncrementalDecoder.push drops a hidden special instead of decoding it.
    Choosing one would leave the response text exactly as it was while the token
    still entered the model's context, so the branch the panel advertised never
    appears and the edit reads as a no-op that quietly changed the run.

    A hidden stop token is the exception, because the runtime cuts a forced
    sequence at its first stop token past the literal prefill: the response ends
    there, which is a visible outcome even though the token itself never shows.

    This holds whatever the run's provenance, so it is asked before the
    provenance question: a hidden alternative is no more usable in the session
    that offered it than on an uploaded run.
    """
    if candidate_id in manager.hidden_token_ids and candidate_id not in stop_ids:
        raise ValueError("The loaded model does not show that token in a response, so choosing it would leave "
                         "the text unchanged while still feeding the token to the model. Type the branch you "
                         "want in Replacement text instead.")


def stop_deciding_id(turn):
    """The token whose membership in the stop set decides this turn's outcome.

    finish_response reads the last sampled token, and for an edited response whose
    replacement ended it with nothing sampled afterwards, the last token of the
    response.
    """
    sampled = turn["metrics"][turn.get("forced_prefix_tokens", 0):]
    if sampled:
        return sampled[-1]["token_id"]
    if turn.get("token_edit") and turn["metrics"]:
        return turn["metrics"][-1]["token_id"]
    return None


def verify_recorded_stops(episode, turn_index, kept, literal_prefill_tokens, stop_ids):
    """Refuse a fork whose replayed tokens no longer behave as they did under the stop set.

    Each earlier response is replayed through finish_response against the stop set
    of the load in memory now. An ID that decodes to the same text can still
    have stopped being configured as a stop token, and then a response that
    ended naturally is read as a length failure: its tool call is never parsed,
    so its messages, event and maze position never reach the forked episode and
    regeneration starts from the wrong state.

    The retained prefix of the edited response is checked too. It reaches the
    runtime as forced_ids, which cuts the forced sequence at the first stop
    token past the literal prefill, so an ID this load newly treats as a stop
    token ends the response inside the prefix and the selected token is never
    reached. kept stops before the edited token, so the response's own closing
    stop token is not part of it and any stop ID found there is one this load
    added.
    """
    for turn in episode.turns[:turn_index]:
        last = stop_deciding_id(turn)
        if (last is not None and last in stop_ids) != (turn.get("finish_reason") == "stop"):
            raise ValueError("The loaded model's stop tokens differ from the ones that produced this run, so "
                             "its earlier responses cannot be reconstructed. This happens when the same model "
                             "ID has been re-downloaded at a different revision; load that snapshot to fork "
                             "this run.")
    if any(metric["token_id"] in stop_ids for metric in kept[literal_prefill_tokens:]):
        raise ValueError("The loaded model treats one of the tokens kept before your edit as a stop token, "
                         "so it would end this response inside the retained prefix and never reach the token "
                         "you selected. This happens when the same model ID has been re-downloaded at a "
                         "different revision; load that snapshot to fork this run.")


def verify_carried_inserts(episode, turn_index, manager):
    """Refuse a fork that would carry a message the loaded model cannot vouch for.

    An uploaded run is checked against its recorded prompts only where the
    model that recorded them was loaded at upload, and against special tokens
    only by the fixed list in inserts.py. A fork has that model in memory, so
    the messages it keeps are asked both questions a live run asks: whether
    this tokenizer reads a special token out of one, and whether every kept
    response from the first message on, the edited one included, recorded the
    prompt its history becomes under this template. On a team each agent's
    responses are read from its own first message on.
    """
    inserts = kept_inserts(episode, turn_index)
    if not inserts:
        return
    hidden = manager.hidden_token_ids
    for insert, answer in inserts:
        if any(set(manager.encode(value)) & hidden for value in (insert["text"], insert.get("sender") or "")):
            raise ValueError(f"The loaded model reads a special token out of the message inserted before response "
                             f"{answer + 1}, so it cannot be carried into a fork.")
    first = {}
    for insert, answer in inserts:
        first.setdefault(insert.get("agent", 0), answer)
    for index in range(min(first.values()), turn_index + 1):
        turn = episode.turns[index]
        agent = turn.get("agent", 0)
        if agent not in first or index < first[agent]:
            continue
        if not turn.get("prompt_ids") or (turn.get("model_id") or episode.model_id) != manager.model_id:
            continue
        context = context_messages(episode, index)
        try:
            prompt = manager.decode(turn["prompt_ids"])
        except (IndexError, KeyError, OverflowError, TypeError, ValueError):
            prompt = None
        try:
            templated = manager.prompt_text(context, episode.tools)
        except Exception:
            templated = None
        following = episode.agents[agent]["messages"][len(context):]
        if prompt is None or not (prompt == templated if templated is not None
                                  else prompt_holds(prompt, context, following)):
            raise ValueError(f"Response {index + 1}'s recorded prompt is not the history the run records for it "
                             "with its inserted messages, so those messages cannot be carried into a fork.")


def kept_inserts(episode, turn_index):
    """The messages a fork before response ``turn_index`` keeps, each with the response it went in before.

    One that went in before the edited response itself is kept: that response
    was generated after it.
    """
    if not episode.team:
        return [(insert, insert["before_turn"]) for insert in episode.config.get("context_inserts", ())
                if insert["before_turn"] <= turn_index]
    asked = {(turn["agent"], turn["round"]): index for index, turn in enumerate(episode.turns)}
    return [(insert, asked[insert["agent"], insert["before_round"]]) for insert in episode.config.get("context_inserts", ())
            if asked.get((insert["agent"], insert["before_round"]), turn_index + 1) <= turn_index]


def fork_token_edit(episode, turn_index, token_index, replacement, manager, *, candidate_id=None):
    """Fork before one response; replay exact earlier IDs plus a replacement.

    token_index addresses the full response metric list, including any supplied
    prefix. Only generated tokens are editable. The original remains untouched,
    including a run uploaded for replay: forking it rebuilds the maze, history
    and counters into a new live episode rather than reopening the saved one.

    On a team the fork keeps every response before the edited one, which is
    every round before its round and the teammates asked before it in its
    own. Those teammates answered the state the round began with, as the
    edited agent did, so nothing in them depends on the response being
    replaced, and the fork finishes their round rather than asking them again.
    """
    with episode.lock:
        if episode.busy:
            raise ValueError("Pause or stop the episode before editing tokens.")
        # The fork replays stored token IDs, so the tokenizer has to match. The
        # model ID is the cheap gate; a later load of the same ID is allowed,
        # which is what lets an uploaded run be forked at all, but only after
        # the run's own recorded text and stop outcomes confirm that this load
        # tokenizes it the same way.
        if episode.model_id and manager.model_id != episode.model_id:
            raise ValueError(f"This run was generated by {episode.model_id}. "
                             "Load that model before editing its tokens.")
        if not isinstance(turn_index, int) or not 0 <= turn_index < len(episode.turns):
            raise ValueError("Select a response and token to edit.")
        original = episode.turns[turn_index]
        metrics = original["metrics"]
        if (not isinstance(token_index, int)
                or not original.get("forced_prefix_tokens", 0) <= token_index < len(metrics)):
            raise ValueError("Select a model-generated token to edit.")
        kept = metrics[:token_index]
        kept_ids = [m["token_id"] for m in kept]
        stop_ids = manager.stop_token_ids
        literal_prefill_tokens = literal_prefill_of(original)
        verify_recorded_text(episode, turn_index, manager)
        verify_recorded_stops(episode, turn_index, kept, literal_prefill_tokens, stop_ids)
        verify_carried_inserts(episode, turn_index, manager)
        if candidate_id is None:
            replacement_ids = manager.encode_replacement(
                kept_ids, replacement, literal_prefill_tokens=literal_prefill_tokens,
            )
        else:
            # The alternative has to be one the run recorded for this token,
            # which is a correctness check on the selection rather than a
            # question about the tokenizer, so it holds whatever the provenance.
            if not any(c["token_id"] == candidate_id
                       for c in metrics[token_index].get("top_candidates", [])):
                raise ValueError("Choose an alternative for the selected token.")
            verify_visible_candidate(candidate_id, manager, stop_ids)
            verify_recorded_candidate(episode, manager)
            replacement_ids = [candidate_id]
        if not replacement_ids:
            raise ValueError("Enter replacement text or choose a token alternative.")
        if any(t in stop_ids for t in replacement_ids[:-1]):
            raise ValueError("A stop token can only appear at the end of the replacement.")
        if episode.team:
            result = fork_team(episode, turn_index)
        else:
            result = fork_single(episode, turn_index, stop_ids)
        agent = original.get("agent", 0)
        prefix = kept_ids + replacement_ids
        result.token_edit = dict(parent_run_id=episode.run_id, turn=turn_index,
                                 token_index=token_index, original_token_id=metrics[token_index]["token_id"],
                                 replacement_ids=replacement_ids,
                                 replacement_text=replacement if candidate_id is None else manager.decode(replacement_ids),
                                 parent_model_id=episode.model_id, parent_load_id=episode.load_id,
                                 parent_replay=episode.replay_only, created_at=time.time())
        parent = episode.agents[agent]
        result.pending_edit = dict(forced_ids=prefix,
                                   literal_prefill_tokens=literal_prefill_tokens,
                                   interruption_here=bool(parent["interrupted"] and parent["intervention_turn"] == turn_index))
        if episode.team:
            result.pending_edit["agent"] = agent
        # The fork continues under the weights in memory now, not the ones that
        # produced the original; token_edit keeps the original stamp.
        result.model_id, result.load_id = manager.model_id, manager.load_id
        result.manual_intervention = True
        result.phase, result.detail = "paused", "Token edit prepared. Regeneration will replace this response and its later moves in a new run."
        return result


def fork_single(episode, turn_index, stop_ids):
    """A run of one agent rebuilt up to the response a token edit replaces.

    The fork rebuilds the run one response at a time, so the closures it
    keeps are replayed at their own boundaries rather than being in force from
    the start. Closures after the edited response are left behind with the
    responses that followed them.
    """
    carried, config = [], copy.deepcopy(episode.config)
    if episode.map_changes:
        carried = [u for u in config.get("map_updates", []) if u["before_turn"] <= turn_index]
        config["map_updates"] = []
    # Insertions follow the same rule, and one at exactly the edited
    # boundary stays: the edited response was generated after it.
    inserts = [i for i in config.pop("context_inserts", []) if i["before_turn"] <= turn_index]
    result = Episode(episode.maze, config)
    # Closures the map refused are carried on the same rule as the ones it
    # accepted. The fork keeps the responses that were generated after a
    # failed intervention, so a fork reporting none would say those
    # responses ran under a map nobody had asked to change.
    result.dropped_closures = copy.deepcopy(
        [d for d in episode.dropped_closures if d["before_turn"] <= turn_index])
    # Rebuild history and recovery counters through the same simulator path
    # used during generation, excluding the edited response and its future.
    for i, previous in enumerate(episode.turns[:turn_index]):
        if carried:
            result.config["map_updates"].extend(u for u in carried if u["before_turn"] == i)
        for insert in inserts:
            if insert["before_turn"] == i:
                land_insert(result, insert)
        turn = copy.deepcopy({key: value for key, value in previous.items() if key not in DERIVED})
        result.turns.append(turn)
        if episode.interrupted and episode.intervention_turn == i:
            mark_interruption(result, 0, i)
        result.phase = "running"
        action = finish_response(result, turn, stop_ids, result.config["per_turn_tokens"])
        resolve_round(result, [action] if action else [])
    if carried:
        result.config["map_updates"].extend(u for u in carried if u["before_turn"] == turn_index)
    for insert in inserts:
        if insert["before_turn"] == turn_index:
            result.edit_insert = (0, result.messages)
            land_insert(result, insert)
    return result


def fork_team(episode, turn_index):
    """A team rebuilt up to the response a token edit replaces, its round left open for the fork to finish.

    Read through the replay a saved team run is read through, so the fork
    holds exactly what the run recorded up to there: every closure and
    message landed where it did, every interruption where it opened a
    response. A message that went in before the edited response stays, since
    that response was generated after it.
    """
    edited = episode.turns[turn_index]
    config = copy.deepcopy(episode.config)
    updates = [u for u in config.pop("map_updates", []) if u["before_round"] <= edited["round"]]
    config.pop("context_inserts", None)
    carried = {(insert["agent"], insert["before_round"]): copy.deepcopy(insert)
               for insert, _ in kept_inserts(episode, turn_index)}
    result = Episode(episode.maze, config)
    kept = copy.deepcopy(episode.turns[:turn_index])
    # Only a request the kept history shows was made: an interruption that
    # landed at or before the edited response. A flag the parent raised later
    # would otherwise interrupt the fork's agent rounds before the parent's did.
    queued = {index for index, agent in enumerate(episode.agents)
              if agent["interrupt_next"] and agent["intervention_turn"] is not None
              and agent["intervention_turn"] <= turn_index}
    result.open_round = replay_rounds(result, kept, edited["round"], updates, carried, True, queued, open_round=True)
    # Carried as the parent had them, so an interruption asked for early
    # still reads as asked for in the fork's own file.
    for index in queued:
        result.agents[index]["interrupt_next"] = True
    own = carried.pop((edited["agent"], edited["round"]), None)
    if own is not None:
        result.edit_insert = (edited["agent"], result.agents[edited["agent"]]["messages"])
        land_insert(result, own)
    # Closures the map refused are carried on the same rule as the ones it
    # accepted. The fork keeps the responses that were generated after a
    # failed intervention, so a fork reporting none would say those
    # responses ran under a map nobody had asked to change.
    result.dropped_closures = copy.deepcopy([d for d in episode.dropped_closures if d["before_round"] <= edited["round"]])
    return result


def assistant_content(turn):
    """A response as the history carries it, with the reasoning a template opened for it restored."""
    return ("<think>" if turn.get("reasoning_prefilled") else "") + turn["text"]


def reply_messages(episode, turn, event):
    """The response and the simulator's reply to its call, as the next prompt carries them.

    ``episode`` stands where the call left it. Generation and the check of a
    saved run's history both write the pair through here.
    """
    return [{"role": "assistant", "content": assistant_content(turn)},
            {"role": "tool", "content": json.dumps(episode.model_state(event["error"]), separators=(",", ":"))}]


def finish_response(episode, turn, stop_ids, max_tokens):
    """Name how a streamed response ended, then read the action it takes.

    A stop requested while it streamed ends it as the reader's, whatever the
    model was doing. Otherwise it ended by choice only where its last sampled
    token is one the load stops on, or, for an edited response whose
    replacement ended it with nothing sampled afterwards, its last token.
    Anything else was cut off: by the limit when it used all of it.
    """
    sampled = turn["metrics"][turn.get("forced_prefix_tokens", 0):]
    natural_stop = bool(sampled and sampled[-1]["token_id"] in stop_ids)
    if turn.get("token_edit") and not sampled and turn["metrics"]:
        natural_stop = turn["metrics"][-1]["token_id"] in stop_ids
    if episode.stop_requested:
        turn["finish_reason"] = "user_stopped"
    elif not natural_stop:
        turn["finish_reason"] = "length" if len(sampled) >= max_tokens else "incomplete_stream"
    else:
        turn["finish_reason"] = "stop"
    return take_action(episode, turn, len(episode.turns) - 1)


def take_action(episode, turn, index):
    """Count a response's tokens and read the action its finish reason allows.

    Returns the action for resolve_round, or None for a response that takes
    none. A response that ends without a call, or is cut off by a limit, takes
    its agent out of the run; on a team its teammates carry on. The agent
    leaves when the round resolves, so a round that never does leaves it in.
    Shared by generation, by a fork rebuilding its earlier responses and by
    replay of a saved team run, so a run is read back through exactly the
    rules that produced it.
    """
    sampled = turn["metrics"][turn.get("forced_prefix_tokens", 0):]
    episode.sampled_tokens += len(sampled)
    turn["sampled_tokens"] = len(sampled)
    turn["tokens_cumulative"] = episode.sampled_tokens
    if turn["finish_reason"] == "user_stopped":
        return None
    if turn["finish_reason"] != "stop":
        turn["outcome"] = "cut_off"
        return None
    content = assistant_content(turn)
    # Tool-looking text inside reasoning is not an external action.
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    communicate = episode.team and episode.config["communication"]
    args, error = parse_call(text, message_limit=MESSAGE_LIMIT if communicate else None)
    if args is None and error is None:
        turn["outcome"] = "no_call"
        return None
    episode.tool_attempts += 1
    return dict(agent=turn.get("agent", 0), turn=index, content=content, args=args, error=error)


def resolve_round(episode, actions):
    """Apply every action of a round at once, then write each agent's reply.

    Every call is judged from the position its agent held when the round
    began, so the order the agents were asked in changes nothing. Agents may
    share a cell. A message goes to every teammate that gets a reply this
    round, which is every teammate that made a call in it.
    """
    round_index, team = episode.rounds, episode.team
    for turn in episode.round_turns(round_index):
        if turn.get("outcome") in DROPPED:
            episode.agents[turn.get("agent", 0)]["status"] = DROPPED[turn["outcome"]]
    sent = []
    for action in actions:
        agent = episode.agents[action["agent"]]
        if action["error"]:
            event = {"accepted": False, "before": list(agent["position"]), "after": list(agent["position"]),
                     "error": action["error"], "arrived": False, "progress": False}
        else:
            event = apply_call(episode.current_maze, agent["position"], action["args"],
                               goal_mode=episode.config["goal_mode"])
            message = action["args"].get("message", "").strip()
            if message:
                event["message"] = message
                sent.append((action["agent"], message))
        if team:
            event.update(source="model", agent=action["agent"], round=round_index, turn=action["turn"])
        else:
            event.update(source="model", turn=action["turn"])
        episode.events.append(event)
        episode.turns[action["turn"]]["event"] = event
        action["event"] = event
    spent = episode.agent_tokens()
    for action in actions:
        agent, event = episode.agents[action["agent"]], action["event"]
        if event["accepted"]:
            agent["position"] = tuple(event["after"])
            if agent["interrupted"] and agent["resumed"] is None:
                agent.update(resumed=True, first_move_progress=event["progress"],
                             latency=spent[action["agent"]] - agent["intervention_tokens"])
        if event["arrived"]:
            agent["status"] = "arrived"
    settle_limits(episode, spent)
    readers = [action["agent"] for action in actions]
    for sender, text in sent:
        episode.mail.append(dict(round=round_index, sender=episode.agents[sender]["name"], text=text,
                                 to=[episode.agents[i]["name"] for i in readers if i != sender]))
    for action in actions:
        inbox = [{"from": episode.agents[sender]["name"], "text": text}
                 for sender, text in sent if sender != action["agent"]]
        episode.agents[action["agent"]]["messages"].extend([
            {"role": "assistant", "content": action["content"]},
            {"role": "tool", "content": json.dumps(episode.agent_state(action["agent"], action["event"]["error"], inbox),
                                                   separators=(",", ":"))}])
    episode.rounds += 1
    if team:
        settle_team(episode, actions, sent)
    else:
        settle_single(episode, actions)
    settle_recoveries(episode)


def settle_recoveries(episode):
    """Score as not recovered every interrupted agent the run is done with that never moved again.

    An agent that has stopped is done with, and so is every agent once the
    run has ended, however it ended: after a round, or before one because the
    budget left could not start it.
    """
    for agent in episode.agents:
        if (agent["status"] != "active" or episode.phase in TERMINAL) and agent["interrupted"] and agent["resumed"] is None:
            agent.update(resumed=False, first_move_progress=False)


def settle_limits(episode, spent):
    """Take out of the run every agent a limit has stopped, once its round has landed.

    An agent that was interrupted and has not moved since stops when its
    recovery window closes, a response the window cut off included, since
    the window is what cut it. Otherwise it stops when it has spent its
    sampled-token limit, including one whose response was cut off at the last
    of it, since the limit is what stopped that response, or when it has made
    as many calls as the run allows. A team saved under one limit for the
    whole team leaves the limit to the team.
    """
    config = episode.config
    tokens = config.get("agent_token_budget") if episode.team else config["token_budget"]
    for index, agent in enumerate(episode.agents):
        if agent["status"] not in ("active", "cut_off"):
            continue
        waiting = agent["interrupted"] and agent["resumed"] is None
        if waiting and (
                spent[index] - agent["intervention_tokens"] >= config["recovery_tokens"]
                or episode.agent_attempts(index) - agent["intervention_attempts"] >= config["recovery_attempts"]):
            agent["status"] = "not_recovered"
        elif tokens is not None and spent[index] >= tokens:
            agent["status"] = "out_of_tokens"
        elif agent["status"] == "active" and episode.agent_attempts(index) >= config.get("attempt_budget", math.inf):
            agent["status"] = "out_of_calls"


def settle_single(episode, actions):
    """Decide what the round a run of one agent just resolved leaves it doing."""
    outcome = episode.turns[episode.rounds - 1].get("outcome")
    if outcome == "no_call":
        episode.phase, episode.detail = "abandoned", "The model ended its response without making a movement call."
        return
    if outcome == "cut_off":
        episode.phase, episode.detail = "budget", "The response reached its generation limit. No unfinished action executed."
        return
    event = actions[0]["event"]
    if event["accepted"]:
        episode.detail = f"Moved {event['direction']} to row {episode.position[0]}, column {episode.position[1]}."
    else:
        episode.detail = "Rejected call: " + event["error"].replace("_", " ") + ". The position did not change."
    status = episode.agents[0]["status"]
    if status == "arrived":
        episode.phase, episode.detail = "arrived", "The simulator confirmed arrival at the destination."
    elif status == "not_recovered":
        episode.phase, episode.detail = "budget", "No accepted move within the recovery window."
    elif status in LIMITED:
        episode.phase, episode.detail = "budget", "The episode reached its token or action limit."


def settle_team(episode, actions, sent):
    """Decide whether the round a team just resolved ends its run, and say what the round did."""
    arrived = [agent["name"] for agent in episode.agents if agent["status"] == "arrived"]
    moving = [agent for agent in episode.agents if agent["status"] == "active"]
    moved = sum(action["event"]["accepted"] for action in actions)
    if arrived and (episode.config["team_goal"] == "any" or len(arrived) == len(episode.agents)):
        episode.phase = "arrived"
        episode.detail = (f"{', '.join(arrived)} reached the destination in round {episode.rounds}."
                          if episode.config["team_goal"] == "any" else
                          f"Every agent reached the destination by round {episode.rounds}.")
    elif not moving:
        # Out of budget only when a limit is what stopped every agent still out.
        spent = all(agent["status"] == "arrived" or agent["status"] in LIMITED for agent in episode.agents)
        episode.phase = "budget" if spent else "abandoned"
        episode.detail = "No agent is still moving: " + ", ".join(
            f"{agent['name']} {agent['status'].replace('_', ' ')}" for agent in episode.agents) + "."
    elif "token_budget" in episode.config and episode.sampled_tokens >= episode.config["token_budget"]:
        episode.phase, episode.detail = "budget", "The team reached its sampled-token limit."
    elif episode.rounds >= episode.config["round_limit"]:
        episode.phase, episode.detail = "budget", "The team reached its round limit."
    else:
        episode.detail = (f"Round {episode.rounds}: {moved} of {len(actions)} call{'' if len(actions) == 1 else 's'} "
                          f"moved an agent, {len(sent)} message{'' if len(sent) == 1 else 's'} sent.")


def discard_round(episode):
    """Mark the responses of a team round that will never resolve as not applied.

    None of it is applied: the agents that answered would otherwise have moved
    while their teammates had not. That includes a response that made no call
    or was cut off, whose agent would otherwise have left the team in a round
    that never happened; its text is kept either way. A resolved round has
    nothing left to mark, so this is safe to call however the stream ends.
    """
    for waiting in episode.turns:
        if waiting["round"] == episode.rounds and "event" not in waiting:
            waiting["outcome"] = "not_applied"


def stream_episode(episode, models, *, single_step=False, save_dir=None, session=None):
    """Generate the episode's rounds, yielding it after each change.

    A round asks every agent still moving for one response, then applies
    them together, so a run of one agent generates one response a round. The
    episode is yielded after every frame of every response and after every
    round. Pause waits for the round to finish, so no agent is left having
    moved while a teammate has not. Stop does not wait: the response it lands
    in keeps its tokens and executes nothing, and on a team the round it
    lands in is discarded, its finished responses kept and marked as never
    applied.

    ``session`` is a model session the caller already holds and keeps: a batch
    of trials holds one for every episode it runs, so nothing else can take
    the model or load another between two trials. Without one, the episode
    opens its own session and closes it when it stops.
    """
    team = episode.team
    step = "round" if team else "response"
    with episode.lock:
        if episode.busy:
            raise ValueError("This episode is already generating. Pause it before changing the run.")
        if episode.phase in TERMINAL or episode.replay_only:
            raise ValueError("Start a new episode to run again. This episode is finished or is a saved replay. Use Play or Next to inspect its recorded responses.")
        manager = session or models.open_session()
        # Steering can start several responses in, so a vector this load
        # cannot take is refused before the run starts rather than there.
        if steering_active(episode.config):
            try:
                manager.check_steering(episode.config["steering"])
            except BaseException:
                if session is None:
                    manager.close()
                raise
        episode.busy = True
        episode.pause_requested = episode.stop_requested = False
        episode.autosave_error = None
        episode.phase = "running"
        episode.model_id = episode.model_id or manager.model_id
        episode.load_id = episode.load_id or manager.load_id
    if team:
        logger.info("Team run %s generating with %s: %s agents, communication %s, round %s, %s",
                    episode.run_id, episode.model_id, len(episode.agents),
                    "on" if episode.config["communication"] else "off", episode.rounds + 1,
                    "one round" if single_step else "until it ends")
    else:
        logger.info("Run %s generating with %s: %s responses so far, %s sampled tokens, %s moves, %s",
                    episode.run_id, episode.model_id, len(episode.turns), episode.sampled_tokens,
                    episode.moves, "one response" if single_step else "until it ends")
    turn = None
    inserted = None
    autosave_error = None

    def record(turn):
        """One line for a response, wherever that response was finalized.

        A turn is completed on three paths: the ordinary one, a stop caught
        between its opening frame and generation, and the cleanup that closes
        an unfinished turn after a failure or a viewer hanging up. The outcome
        line below counts all three as responses, so recording only the first
        left a reader the two cases this file is opened for - a stop and a
        crash - with nothing between the run starting and its summary.

        The duration is set here rather than read, because the two abnormal
        paths never had one, and a turn saved without it reads as a response
        that took no time rather than one nobody timed.
        """
        turn.setdefault("seconds", time.time() - turn["started_at"])
        if team:
            logger.info("Team run %s round %s %s: %s after %s sampled tokens in %.1fs",
                        episode.run_id, turn["round"] + 1, episode.agents[turn["agent"]]["name"],
                        turn.get("outcome") or turn["finish_reason"], turn.get("sampled_tokens", 0), turn["seconds"])
            return
        logger.info("Run %s response %s%s: %s after %s sampled tokens in %.1fs. %s",
                    episode.run_id, len(episode.turns), ", steered" if turn.get("steered") else "",
                    turn["finish_reason"], turn.get("sampled_tokens", 0), turn["seconds"], episode.detail)

    def autosave():
        nonlocal autosave_error
        if not save_dir or autosave_error is not None:
            return
        try:
            episode.save(save_dir)
        except OSError as exc:
            # Storage is separate from the model outcome. Stop between turns,
            # retain the in-memory run, and never retry this failure in cleanup.
            autosave_error = str(exc)
            episode.pause_requested = True

    try:
        while episode.phase == "running":
            if episode.stop_requested:
                episode.phase, episode.detail = "stopped", f"Stopped by you before the next {step}."
                break
            if episode.pause_requested:
                episode.phase = "paused"
                episode.detail += f" Paused before the next {step}."
                break
            if manager.load_id != episode.load_id:
                raise ValueError("The model changed during this episode. Start a new episode with the selected model.")
            # Before the round is generated, so its calls are judged against
            # the map as changed. A response's own prompt still shows the map
            # it was given, because the history is already written: the model
            # meets the change in the simulator's reply to whatever it does
            # next, which carries the current grid whether the call was
            # accepted or refused.
            opened, episode.open_round = episode.open_round, None
            moving = [index for index, agent in enumerate(episode.agents) if agent["status"] == "active"]
            if opened is None:
                apply_closure(episode)
                caps = episode.response_caps(moving)
                if caps is None:
                    episode.phase = "budget"
                    episode.detail = ("The team's remaining sampled tokens cannot give every moving agent a response."
                                      if team else "The sampled-token budget is exhausted.")
                    settle_recoveries(episode)
                    break
                actions = []
            else:
                # A fork finishing the round its edit was in asks only the
                # agents the round had not reached.
                answered = {turn["agent"] for turn in episode.round_turns(episode.rounds)}
                moving = [index for index in moving if index not in answered]
                caps, actions = opened["caps"], opened["actions"]
            episode.round_open = team
            for index in moving:
                agent = episode.agents[index]
                edit = episode.pending_edit if (episode.pending_edit or {}).get("agent", 0) == index else None
                forced = edit["forced_ids"] if edit else interrupted_prefix(episode, manager, index)
                inserts_interruption = edit["interruption_here"] if edit else bool(forced)
                limit = response_limit(episode, index, caps[index], inserts_interruption)
                if limit <= 0:
                    episode.phase, episode.detail = "budget", "The sampled-token budget is exhausted."
                    settle_recoveries(episode)
                    break
                # After the budget is known to allow a response, so an insertion is
                # only ever recorded with the response that read it.
                before = apply_insert(episode, manager, index)
                inserted = None if before is None else (index, before)
                if edit and episode.edit_insert is not None:
                    inserted, episode.edit_insert = episode.edit_insert, None
                turn = {"text": "", "metrics": [], "prompt_ids": [], "forced_prefix_tokens": 0,
                        "prefix_ids": [], "prefix_text": "",
                        "planned_prefix_ids": forced, "planned_prefix_text": manager.decode(forced),
                        "position_before": list(agent["position"]), "started_at": time.time(), "finish_reason": None}
                if team:
                    turn = {"agent": index, "round": episode.rounds, **turn}
                turn["literal_prefill_tokens"] = edit["literal_prefill_tokens"] if edit else len(forced)
                steered = episode.steers_next(index)
                if episode.config.get("steering") is not None:
                    # Unmarked until generation is entered below. The flag says the
                    # vector touched this response, and a Stop taken at the opening
                    # frame ends the run before the vector is ever installed.
                    turn["steered"] = False
                    if steered and not team and episode.steer_turn is None:
                        episode.detail = "Steering starts with this response."
                if edit:
                    turn["token_edit"] = copy.deepcopy(episode.token_edit)
                    episode.pending_edit = None
                episode.turns.append(turn)
                if team:
                    episode.selected_turn = len(episode.turns) - 1
                yield episode
                if episode.stop_requested:
                    if inserted is not None:
                        withdraw_insert(episode, inserted[1], index)
                        inserted = None
                    finish_response(episode, turn, set(), limit)
                    if team:
                        record(turn)
                    break
                if steered:
                    turn["steered"] = True
                stop_ids = manager.stop_token_ids
                generator = manager.generate(
                    agent["messages"], temperature=episode.config["temperature"], top_p=1., top_k=0,
                    max_new_tokens=limit,
                    # Distinct per agent as well as per round: agents given the
                    # same prompt would otherwise sample the same response.
                    seed=episode.config["sampling_seed"] + 100003 * episode.rounds + 7919 * index,
                    analyze_prompt=False, tools=episode.tools, forced_ids=forced,
                    literal_prefill_tokens=turn["literal_prefill_tokens"],
                    steering=episode.config["steering"] if steered else None,
                )
                try:
                    for update in generator:
                        turn.update(text=update.text, metrics=copy.deepcopy(update.metrics), prompt_ids=list(update.prompt_ids),
                                    forced_prefix_tokens=update.forced_prefix_tokens, reasoning_prefilled=update.reasoning_prefilled,
                                    load_id=update.load_id, model_id=update.model_id)
                        # The prompt holding the message has been fed.
                        if update.prompt_ids:
                            inserted = None
                        # The runtime emits prefix metrics only after prefill has
                        # consumed them. An opening frame or a failed model call
                        # alone is not evidence that an interruption was inserted.
                        if forced and update.forced_prefix_tokens and update.metrics:
                            turn.update(prefix_ids=forced, prefix_text=manager.decode(forced))
                        if inserts_interruption and not agent["interrupted"] and update.forced_prefix_tokens and update.metrics:
                            mark_interruption(episode, index, len(episode.turns) - 1)
                            episode.detail = (f"{agent['name']} was interrupted" if team else "Interruption inserted") \
                                + ". Watching for a real movement call."
                        yield episode
                        if episode.stop_requested:
                            break
                finally:
                    generator.close()
                if inserted is not None:
                    withdraw_insert(episode, inserted[1], index)
                    inserted = None
                turn["seconds"] = time.time() - turn["started_at"]
                action = finish_response(episode, turn, stop_ids, limit)
                if action:
                    actions.append(action)
                if team:
                    # A run of one agent says what its response did once the
                    # round has applied it.
                    record(turn)
                    yield episode
                # A run of one agent keeps a response that finished before the
                # stop landed; the loop stops before the next one instead.
                if episode.stop_requested and (team or turn["finish_reason"] == "user_stopped"):
                    break
            if episode.phase != "running":
                break
            if episode.stop_requested and (team or turn["finish_reason"] == "user_stopped"):
                episode.phase = "stopped"
                if team:
                    discard_round(episode)
                    episode.detail = (f"Stopped by you during round {episode.rounds + 1}. "
                                      "Its responses were kept and none of its moves applied.")
                else:
                    episode.detail = "Stopped by you. Partial tokens were retained; no partial action executed."
                    record(turn)
                break
            resolve_round(episode, actions)
            episode.round_open = False
            if not team:
                record(turn)
            # Before this autosave rather than only in the cleanup below. A
            # response that ends the episode can be the one a closure was
            # queued during, and the file written here is the whole record if
            # the process is killed at the yield: a finished run still holding
            # a queue is one this file's own reader refuses.
            if episode.phase in TERMINAL:
                with episode.lock:
                    abandon_closure(episode)
                    abandon_insert(episode)
            autosave()
            if team:
                episode.viewing, episode.selected_turn = episode.rounds - 1, None
            yield episode
            if episode.phase in TERMINAL:
                break
            if episode.stop_requested:
                continue
            if single_step or episode.pause_requested:
                episode.phase = "paused"
                episode.detail += f" Paused before the next {step}."
                break
            time.sleep(.25)
    except GeneratorExit:
        if episode.phase == "running":
            episode.phase, episode.detail = "stopped", "Viewer stopped streaming. The partial response was retained."
        raise
    except Exception as exc:
        # The panel says this too, but the panel is gone by the time anyone
        # asks, and a failure inside generation is what a log read after a
        # memory kill is looking for.
        logger.exception("%s %s failed while generating", "Team run" if team else "Run", episode.run_id)
        episode.phase, episode.detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        # A model call that failed before feeding its prompt.
        if inserted is not None:
            withdraw_insert(episode, inserted[1], inserted[0])
        if turn is not None and turn.get("finish_reason") is None:
            count = max(0, len(turn["metrics"]) - turn["forced_prefix_tokens"])
            episode.sampled_tokens += count
            turn.update(sampled_tokens=count, tokens_cumulative=episode.sampled_tokens, finish_reason=episode.phase)
            if team:
                turn["outcome"] = "not_applied"
            # Only a turn no other path finalized reaches here, so this cannot
            # write a second line for a response already recorded.
            record(turn)
        if team:
            # A failure or a viewer hanging up mid-round never reaches the
            # round's resolution either, so the teammates that already
            # answered are marked the way a stop marks them.
            discard_round(episode)
        with episode.lock:
            episode.busy = episode.round_open = False
            # Before the autosave, so the file records the closure this run
            # will now never reach rather than a queue it emptied silently.
            if episode.phase in TERMINAL:
                abandon_closure(episode)
                abandon_insert(episode)
                abandon_fork(episode)
            if session is None:
                manager.close()
            autosave()
            if autosave_error is not None:
                episode.warn_autosave(autosave_error)
        if team:
            logger.info("Team run %s %s after %s rounds, %s responses and %s sampled tokens, %s moves: %s",
                        episode.run_id, episode.phase, episode.rounds, len(episode.turns), episode.sampled_tokens,
                        episode.moves, episode.detail)
        else:
            logger.info("Run %s %s after %s responses and %s sampled tokens, %s moves: %s",
                        episode.run_id, episode.phase, len(episode.turns), episode.sampled_tokens,
                        episode.moves, episode.detail)
    yield episode


def validate_steering(episode):
    """Refuse a saved run whose steered responses are not the ones its trigger picks.

    Each response's flag is read as provenance - which responses the vector
    touched - so it is decided again from the recorded path, and a run that
    marks a response the trigger would have left alone, or leaves unmarked one
    it would have steered, is refused rather than scored.

    One response the trigger picks may be unmarked: a run stopped at the
    opening frame of a response, before generation, never installed the
    vector. That response carries no metrics and a stop as its outcome, and
    is the run's last, because a stopped run cannot continue.
    """
    config = episode.config
    if config.get("steering") is None:
        if any("steered" in turn for turn in episode.turns):
            raise ValueError("A run without a steering vector cannot mark responses as steered.")
        return
    supplied = [e for e in episode.events if e["source"] == "supplied"]
    position = tuple(supplied[-1]["after"]) if supplied else episode.maze.start
    moves, start = len(supplied), None
    by_turn = {e["turn"]: e for e in episode.events if e["source"] == "model"}
    for index, turn in enumerate(episode.turns):
        expected = steered_at(config, start, index, position, moves)
        never_generated = (index == len(episode.turns) - 1 and not turn.get("metrics")
                           and turn.get("finish_reason") in ("user_stopped", "stopped"))
        if turn.get("steered", False) is not expected and not (expected and never_generated):
            raise ValueError(f"Response {index + 1} is recorded as {'steered' if turn.get('steered') else 'unsteered'}, "
                             "which is not what the run's steering trigger decides at that point in its path.")
        if expected and start is None:
            start = index
        event = by_turn.get(index)
        if event and event["accepted"]:
            position, moves = tuple(event["after"]), moves + 1


def validate_checkpoint_closures(episode):
    """Refuse a saved run whose map changes a live run would have refused.

    Each closure is checked by check_checkpoint_closure, at the boundary and
    position it records, against the map it produced. A run with a waypoint
    and no vector is checked too, since the waypoint rule does not depend on
    steering. Runs after validate_updates and validate_steering, so the
    closures, the path and the steered flags it reads from are ones the file
    has already been held to.
    """
    maze = episode.maze
    for update in episode.config.get("map_updates", ()):
        maze = close_cell(maze, update["closed_cell"])
        try:
            check_checkpoint_closure(episode, update["closed_cell"], maze, update["before_turn"], {0: update["position"]})
        except ValueError as exc:
            raise ValueError(f"The map change before response {update['before_turn'] + 1} is one the run "
                             f"could not have made. {exc}") from None


OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def insert_outcome(episode, insert):
    """What the run did after one insertion, read off its path and its map and never off the file.

    The advice is judged on the map as the response after it found it, from
    the cell the character stood on: whether the advised step is open, and
    whether it is on a shortest route to the destination. The move is the
    first accepted model move from that response on, by the agent the message
    went to, and it followed the advice, went against it (the opposite
    direction), or took another.
    """
    boundary, advised = insert[episode.boundary_key], insert.get("advised_direction")
    agent = insert.get("agent", 0)
    move = next((e for e in episode.events
                 if e["source"] == "model" and e["accepted"] and e.get("agent", 0) == agent
                 and (e["round"] if episode.team else e["turn"]) >= boundary), None)
    outcome = dict(advised=advised, move=move, legal=None, shortest=None, followed=None)
    if advised is None:
        return outcome
    maze = maze_at_turn(episode.maze, episode.config.get("map_updates", ()), boundary, episode.boundary_key)
    position = tuple(insert["position"])
    step = maze.neighbors(position).get(advised)
    distances = maze.distances(maze.goal)
    outcome.update(legal=step is not None,
                   shortest=step is not None and distances.get(step) == distances[position] - 1)
    if move is not None:
        outcome["followed"] = ("followed" if move["direction"] == advised
                               else "against" if move["direction"] == OPPOSITE[advised] else "other")
    return outcome


def path_position(episode, boundary):
    """Where the run's own transitions put the character before response ``boundary``."""
    accepted = [e for e in episode.events if e["accepted"] and e.get("turn", -1) < boundary]
    return list(accepted[-1]["after"]) if accepted else list(episode.maze.start)


def validate_inserts(episode, read_prompt=None):
    """Refuse a saved run whose inserted messages are not the ones its history holds.

    Each insertion has to name a response the run reaches, one to a boundary
    and in order, meet its channel's rules, and record where the character's
    path had it standing. The history is then written again from the setup,
    the responses, the simulator's replies and the recorded insertions, the
    way generation wrote it, and has to be the saved ``messages`` exactly: a
    file whose record says one thing and whose history another would be read
    as the first and replayed as the second.

    The response each insertion landed before has to record the prompt it
    was fed, since a run only keeps a message once a prompt holding it has
    been. Every response from the first insertion on that records a prompt is
    then read. ``read_prompt(episode, turn, context)`` answers for a prompt with
    two readings, or None when the model that recorded it is not the one
    loaded: the recorded IDs decoded, and ``context`` - the messages the
    record gives that response - put through the same model's template, or
    None where the template cannot render them. The two have to be the same
    text, the whole prompt and not only the parts the record names, so a
    message nobody recorded cannot sit anywhere in it. Where the template
    gives no reading, the decoded prompt is held to :func:`prompt_holds`
    instead. Without a reader, a run uploads with no tokenizer at all.
    """
    inserts = episode.config.get("context_inserts")
    if not isinstance(inserts, list) or not inserts:
        raise ValueError(f"A {INSERT_FORMAT} run carries its inserted messages as a non-empty list.")
    previous = -1
    for insert in inserts:
        boundary = insert.get("before_turn") if isinstance(insert, dict) else None
        if type(boundary) is not int:
            raise ValueError("Each inserted message names the response it landed before by its index.")
        if not 0 <= boundary < len(episode.turns):
            raise ValueError(f"The message inserted before response {boundary + 1} names a response the run "
                             "never reached.")
        if boundary <= previous:
            raise ValueError(f"The message inserted before response {boundary + 1} is out of order or shares its "
                             "response with another. One message lands before a response, in order.")
        previous = boundary
        try:
            check_insert(insert)
        except ValueError as exc:
            raise ValueError(f"The message inserted before response {boundary + 1} is refused. {exc}") from None
        if insert.get("position") != path_position(episode, boundary):
            raise ValueError(f"The message inserted before response {boundary + 1} records a position the "
                             "run's path does not reach there.")
        if not episode.turns[boundary].get("prompt_ids"):
            raise ValueError(f"Response {boundary + 1} records no prompt, so nothing says the model read the "
                             "message inserted before it.")
    if not episode.manual_intervention:
        raise ValueError("A run carrying an inserted message cannot report that nobody intervened in it.")
    validate_history(episode)
    if read_prompt is None:
        return
    # Every response from the first message on, not only the one each landed
    # before: a message stays in every later context, and a later prompt
    # without it would credit its response to an intervention it never read.
    for index in range(inserts[0]["before_turn"], len(episode.turns)):
        turn = episode.turns[index]
        if not turn.get("prompt_ids"):
            continue
        context = context_messages(episode, index)
        reading = read_prompt(episode, turn, context)
        if reading is None:
            continue
        prompt, templated = reading
        if not (prompt == templated if templated is not None
                else prompt_holds(prompt, context, episode.messages[len(context):])):
            raise ValueError(f"Response {index + 1}'s recorded prompt is not the history the run records "
                             "for it, with the messages inserted up to that response.")


def prompt_holds(prompt, context, following):
    """Whether a decoded prompt carries a response's context and nothing after it.

    The fallback for a template that gives no reading, so weaker than the
    comparison: text between the messages it finds is not read. Read without
    the template: every user and simulator message has to appear
    in the prompt as written and in order, and the next one the history holds
    after this context must not. Templates write those two roles verbatim,
    where some rewrite an earlier response's reasoning, so they are what can be
    found. Finding the messages in order pins an insertion to its own
    boundary: the same note given twice, or a user message whose words
    already sit in the system prompt, is only found where the order puts it.
    """
    position = 0
    for message in context:
        if message["role"] in ("user", "tool"):
            found = prompt.find(message["content"], position)
            if found < 0:
                return False
            position = found + len(message["content"])
    later = next((m["content"] for m in following if m["role"] in ("user", "tool")), None)
    return later is None or later not in prompt[position:]


def validate_history(episode):
    """Write a saved run's history again from its record and compare it with the saved one.

    Walks the responses in order on a fresh episode of the same scenario,
    landing each closure and each insertion at its own boundary and adding the
    pair each call left, and names the first response whose context disagrees.
    """
    config = copy.deepcopy(episode.config)
    config.pop("context_inserts")
    config["map_updates"] = []
    rebuilt = Episode(episode.maze, config)
    updates = episode.config.get("map_updates", [])
    by_boundary = {insert["before_turn"]: insert for insert in episode.config["context_inserts"]}
    by_turn = {e["turn"]: e for e in episode.events if e["source"] == "model"}
    saved = episode.messages
    if not isinstance(saved, list):
        raise ValueError("A run's messages must be a list.")
    for index, turn in enumerate(episode.turns):
        if ("event" in turn) != (index in by_turn):
            raise ValueError(f"Response {index + 1} disagrees with the run's path about whether it made a call.")
        rebuilt.config["map_updates"].extend(u for u in updates if u["before_turn"] == index)
        if index in by_boundary:
            try:
                rebuilt.messages = render_insert(rebuilt.messages, by_boundary[index])
            except ValueError as exc:
                raise ValueError(f"The message inserted before response {index + 1} cannot be placed. {exc}") from None
        if rebuilt.messages != saved[:len(rebuilt.messages)]:
            raise ValueError(f"The saved messages disagree with the run's record at response {index + 1}: its "
                             "context is not the history and the inserted messages the run records.")
        event = by_turn.get(index)
        if event is not None:
            rebuilt.events.append(event)
            rebuilt.position = tuple(event["after"])
            rebuilt.messages.extend(reply_messages(rebuilt, turn, event))
    if rebuilt.messages != saved:
        raise ValueError("The saved messages disagree with the run's record after its last response.")


def from_payload(data, read_prompt=None):
    """A saved run, checked against itself, ready to replay.

    A team run is read by :func:`team_from_payload`. ``read_prompt`` is handed
    to :func:`validate_inserts` for a run carrying inserted messages.
    """
    if isinstance(data, dict) and data.get("format") in (TEAM_FORMAT, LEGACY_TEAM_FORMAT):
        return team_from_payload(data, read_prompt)
    if not isinstance(data, dict) or data.get("format") not in (FORMAT, CHANGING_FORMAT, INSERT_FORMAT):
        raise ValueError("Choose a ChatLab maze run JSON file.")
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    if not isinstance(data.get("maze"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("The run is missing its map or its configuration.")
    # A team is saved in a format of its own, so a count here would read one
    # agent's record as a team's.
    if "agents" in data["config"]:
        raise ValueError(f"A team run has to be recorded as {TEAM_FORMAT}.")
    inserted = data["format"] == INSERT_FORMAT
    # An insertion shifts the context of every response after it, so a run
    # carrying one is only read under the format that says so.
    if not inserted and data["config"].get("context_inserts"):
        raise ValueError(f"A run carrying inserted messages has to be recorded as {INSERT_FORMAT}.")
    # The format says whether a run carries insertions; whether its map
    # changes is read off the map itself, which a changing one records with
    # its environment identifier.
    changing = data["format"] == CHANGING_FORMAT or (inserted and "environment_id" in data["maze"])
    # A fixed-map run carrying closures would replay as the map it started
    # from, which is not the map its responses were answering.
    if not changing and (data["config"].get("map_updates") or data.get("dropped_closures")
                         or data.get("close_next")):
        raise ValueError(f"A run whose map changes has to be recorded as {CHANGING_FORMAT}." if not inserted else
                         "A run whose map changes has to record the changing map, with its environment identifier.")
    maze = load_maze(data["maze"]) if changing else Maze.from_dict(data["maze"])
    result = Episode(maze, data["config"])
    allowed = result.payload().keys() - {"format", "maze", "config", "exploratory", "tokenizer_note"}
    for key in allowed:
        if key in data:
            setattr(result, key, data[key])
    if changing:
        updates = result.config.get("map_updates", [])
        if not isinstance(updates, list):
            raise ValueError("A run's map changes must be a list.")
        validate_updates(maze, updates, result.turns, result.events)
        # The closures a run reports dropping are read as provenance by Run
        # details and carried into every fork, so they are checked like the
        # ones it reports taking rather than taken as written.
        validate_drops(maze, result.dropped_closures, updates, len(result.turns))
        # A run that has ended clears its queue on the way out, so a finished
        # one still waiting to close a cell is a state no run reaches.
        if result.close_next and result.phase in TERMINAL:
            raise ValueError("A run that has ended cannot still be waiting to close a cell.")
        validate_pending(maze, result.close_next, updates, result.dropped_closures, len(result.turns))
        # Every closure begins as a request from the reader, and requesting one
        # marks the run. A file carrying a closure while reporting an untouched
        # run would be read as a clean control by anything scoring it.
        if (updates or result.dropped_closures or result.close_next) and not result.manual_intervention:
            raise ValueError("A run carrying a closure cannot report that nobody intervened in it.")
    # Reconstruct the visible path from real transitions, never trust claimed positions.
    position = maze.start
    for event in result.events:
        if tuple(event["before"]) != position:
            raise ValueError("The saved path contains a position mismatch.")
        if event["accepted"]:
            mode = result.config["goal_mode"]
            current = maze_at_turn(maze, result.config.get("map_updates", []), event.get("turn", -1)) if changing else maze
            actual = apply_call(current, position, {"maze_id": current.tool_id(mode), "direction": event["direction"]}, goal_mode=mode)
            if not actual["accepted"] or actual["after"] != event["after"]:
                raise ValueError("The saved path contains an invalid transition.")
            position = tuple(event["after"])
        elif tuple(event["after"]) != position:
            raise ValueError("A rejected action changed the saved position.")
    if tuple(result.position) != position:
        raise ValueError("The saved final position does not match its path.")
    validate_steering(result)
    validate_checkpoint_closures(result)
    result.position = position
    if inserted:
        validate_inserts(result, read_prompt)
    # A response that ended before it could be read never finished its round.
    result.rounds = sum(turn.get("finish_reason") in ("stop", "length", "incomplete_stream") for turn in result.turns)
    result.replay_only = True
    return result


# What a team run written before supplied moves, waypoints, interruptions,
# call limits, changing maps and inserted messages reached teams cannot have set.
LEGACY_TEAM_KEYS = ("supplied_moves", "waypoint", "interrupt_agents", "attempt_budget", "map_updates",
                    "context_inserts")
# What a team's inserted message records: an insertion's own fields, the agent
# it went to, and the round it went in before.
TEAM_INSERT_KEYS = {"before_round", "agent", "channel", "text", "sender", "position", "advised_direction"}


def land_saved_closure(episode, update, boundary):
    """Check one closure a saved team run records against the run rebuilt so far, and land it.

    It has to name the boundary it is being landed at, record where every
    agent stood, and be one the live run would have made there: the rules a
    queued closure meets when it lands, asked of the agents still moving.
    """
    if not isinstance(update, dict) or set(update) != {"before_round", "positions", "closed_cell", "grid"}:
        raise ValueError("Each map change of a team run records its round, every agent's position, its cell and "
                         "the map it made.")
    if update["positions"] != [list(agent["position"]) for agent in episode.agents]:
        raise ValueError(f"The map change before round {boundary + 1} records positions the team's paths never reached.")
    try:
        changed = checked_closure(episode, update["closed_cell"], boundary)
    except ValueError as exc:
        raise ValueError(f"The map change before round {boundary + 1} is one the run could not have made. {exc}") from None
    if list(changed.grid) != update["grid"]:
        raise ValueError("A map change records a map its own closure does not produce.")
    episode.config.setdefault("map_updates", []).append(copy.deepcopy(update))


def land_saved_insert(episode, record, saved):
    """Check one message a saved team run records going in before response ``saved``, and land it."""
    agent = episode.agents[saved["agent"]]
    if not isinstance(record, dict) or set(record) != TEAM_INSERT_KEYS:
        raise ValueError("Each message inserted into a team run records its round, its agent, its channel, text, "
                         "sender, advice and position.")
    try:
        check_insert({key: value for key, value in record.items() if key not in ("before_round", "agent")})
    except ValueError as exc:
        raise ValueError(f"The message inserted for {agent['name']} before round {saved['round'] + 1} is refused. "
                         f"{exc}") from None
    if record["position"] != list(agent["position"]):
        raise ValueError(f"The message inserted for {agent['name']} before round {saved['round'] + 1} records a "
                         "position its path does not reach there.")
    if not saved.get("prompt_ids"):
        raise ValueError(f"{agent['name']}'s response in round {saved['round'] + 1} records no prompt, so nothing "
                         "says it read the message inserted before it.")
    try:
        land_insert(episode, copy.deepcopy(record))
    except ValueError as exc:
        raise ValueError(f"The message inserted for {agent['name']} before round {saved['round'] + 1} cannot be "
                         f"placed. {exc}") from None


def read_interruption(episode, saved, index, manual, asked=()):
    """Check the interruption a saved team response records, or records the absence of, and land it.

    Returns whether the response was planned to open with it, which narrows its
    token cap. The run's interruption text is not encoded here, since a team
    run is read with no tokenizer, so the prefix is held to what the run can
    be checked against: it is the one the response planned, its tokens open the
    response, it lands on an agent the run interrupts that has not been
    interrupted yet, and it lands on the first response after that agent's
    trigger unless the run records a reader asking for it sooner, for that
    agent: ``asked`` holds the agents whose queued flag the file records. A
    planned prefix the model never read belongs to a response stopped before
    it began.

    A response a token edit regenerated plans a longer prefix: the tokens it
    kept from the response it replaced and the replacement, of which only an
    interruption the replaced response opened with, its literal prefill, is
    the interruption.
    """
    agent, config = episode.agents[saved["agent"]], episode.config
    planned, consumed = saved.get("planned_prefix_ids", []), saved.get("forced_prefix_tokens", 0)
    prefix, edit = saved.get("prefix_ids", []), saved.get("token_edit")
    if not isinstance(planned, list) or any(type(i) is not int for i in planned) or type(consumed) is not int:
        raise ValueError("A response's interruption prefix must be a list of token IDs.")
    literal = saved.get("literal_prefill_tokens", len(planned))
    if edit is not None:
        replaced = edit.get("replacement_ids") if isinstance(edit, dict) else None
        if not manual or not isinstance(replaced, list) or not replaced or any(type(i) is not int for i in replaced) \
                or planned[len(planned) - len(replaced):] != replaced or type(literal) is not int \
                or not 0 <= literal <= len(planned) - len(replaced):
            raise ValueError("A token-edited response has to plan the tokens it kept and its replacement, and the run "
                             "has to record the edit.")
        interruption = planned[:literal]
    elif literal != len(planned):
        raise ValueError("A response forces a prefix other than the interruption it planned.")
    else:
        interruption = planned
    eligible = (bool(config.get("interruption_text", "").strip()) and not agent["interrupted"]
                and agent["status"] == "active"
                and saved["agent"] in targeted(config, "interrupt_agents", len(episode.agents)))
    due = eligible and episode.agent_moves(saved["agent"]) >= config.get("interrupt_after", 0)
    if not planned and (consumed or prefix):
        raise ValueError("A response records an interruption prefix it never planned.")
    if not interruption and due:
        raise ValueError(f"{agent['name']} was due its interruption at a response that does not open with it.")
    if interruption:
        if not eligible or (not due and saved["agent"] not in asked):
            raise ValueError(f"A response of {agent['name']} opens with an interruption the run could not have given it there.")
        if config.get("prefix_tokens") and len(interruption) > config["prefix_tokens"]:
            raise ValueError("A response's interruption is longer than the run's supplied token count.")
    if consumed:
        if consumed != len(planned) or prefix != planned or [m["token_id"] for m in saved["metrics"][:consumed]] != planned:
            raise ValueError("A response's prefix is not the one its tokens open with.")
        if interruption:
            mark_interruption(episode, saved["agent"], index)
    elif planned and (prefix or saved["metrics"]):
        raise ValueError("A response that never read its prefix records tokens after it.")
    return bool(interruption)


def replay_rounds(result, turns, rounds, updates, by_answer, manual, queued=(), *, open_round=False):
    """Read saved team responses into ``result`` round by round, as the live run read them.

    Each closure in ``updates`` lands before the round it names, and each
    message in ``by_answer`` before the response it names, both consumed as
    they land so the caller can refuse any left over. The first ``rounds``
    rounds resolve. What the round after them holds is discarded, as a
    stopped round's responses are, unless ``open_round`` keeps it open for a
    fork to finish, and then its actions and caps come back. ``queued`` holds
    the agents a reader asked to interrupt ahead of their trigger.
    """
    by_round = {}
    for turn in turns:
        by_round.setdefault(turn["round"], []).append(turn)
    for round_index in range(rounds + 1):
        if updates and isinstance(updates[0], dict) and updates[0].get("before_round") == round_index:
            if result.phase in TERMINAL:
                raise ValueError("The run records a map change after it had ended.")
            land_saved_closure(result, updates.pop(0), round_index)
            # A live run queues one closure at a time and lands it before a round.
            if updates and isinstance(updates[0], dict) and updates[0].get("before_round") == round_index:
                raise ValueError(f"A team run records two map changes before round {round_index + 1}. One lands "
                                 "before a round.")
        if round_index not in by_round:
            continue
        if result.phase in TERMINAL:
            raise ValueError("The run records responses after it had ended.")
        moving = [i for i, agent in enumerate(result.agents) if agent["status"] == "active"]
        asked = [t["agent"] for t in by_round[round_index]]
        resolved = round_index < rounds
        if asked != moving if resolved else asked != moving[:len(asked)]:
            raise ValueError("A round asks agents other than the ones still moving, in their order.")
        # The caps the live run set for this round, which no response in it
        # could have sampled past.
        caps = result.response_caps(moving) or {}
        actions = []
        for saved in by_round[round_index]:
            interrupting = read_interruption(result, saved, len(result.turns), manual, queued)
            if (saved["agent"], round_index) in by_answer:
                land_saved_insert(result, by_answer.pop((saved["agent"], round_index)), saved)
            limit = response_limit(result, saved["agent"], caps.get(saved["agent"], 0), interrupting)
            if resolved and saved["finish_reason"] not in ("stop", "length", "incomplete_stream"):
                raise ValueError("A response in a finished round ends in a way no finished round records.")
            # finish_response names the reason from the sampled tokens, so the
            # tokens have to be ones that reason could have been named from.
            count = len(saved["metrics"]) - saved.get("forced_prefix_tokens", 0)
            if limit <= 0 or count > limit:
                raise ValueError("A response holds more tokens than its round allowed each agent.")
            if {"stop": count == 0, "length": count != limit, "incomplete_stream": count >= limit}.get(
                    saved["finish_reason"], False):
                raise ValueError("A response records a finish reason its tokens could not have produced.")
            if saved.get("position_before") != list(result.agents[saved["agent"]]["position"]):
                raise ValueError("A response records a starting position its agent was not in.")
            if result.config.get("steering") is None:
                if "steered" in saved:
                    raise ValueError("A run without a steering vector cannot mark responses as steered.")
            else:
                expected = result.steers_next(saved["agent"])
                never_generated = (saved is turns[-1] and not saved["metrics"]
                                   and saved["finish_reason"] in ("user_stopped", "stopped"))
                flag = saved.get("steered")
                if type(flag) is not bool or (flag != expected and not (expected and never_generated)):
                    raise ValueError("A response's steered flag does not match its agent's steering trigger.")
            turn = copy.deepcopy({key: value for key, value in saved.items() if key not in DERIVED})
            result.turns.append(turn)
            action = take_action(result, turn, len(result.turns) - 1)
            if action:
                actions.append(action)
        if resolved:
            resolve_round(result, actions)
        elif open_round:
            return dict(actions=actions, caps=caps)
        else:
            discard_round(result)
    return None


def team_from_payload(data, read_prompt=None):
    """A saved team run, rebuilt for replay from the responses it records.

    Nothing the run derived is taken as written. Every recorded response is
    read again through the rules that read it live, round by round, each map
    change and inserted message landing where it records, and the moves,
    messages, histories, positions, statuses, interruptions, counters and
    outcome that produces are compared with the file's. A file that disagrees
    anywhere describes a run these responses could not have made, and is
    refused. ``read_prompt`` reads each response's recorded prompt from its
    agent's first inserted message on, as :func:`validate_inserts` does for a
    run of one agent.

    A team run written as chatlab-maze-team-1 predates supplied moves,
    waypoints, interruptions, call limits, changing maps and inserted messages
    on a team, and is read as having none; its agents record only their name,
    position, status and history.
    """
    if not isinstance(data, dict) or data.get("format") not in (TEAM_FORMAT, LEGACY_TEAM_FORMAT):
        raise ValueError("Choose a ChatLab maze team run JSON file.")
    legacy = data["format"] == LEGACY_TEAM_FORMAT
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    if not isinstance(data.get("maze"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("The run is missing its map or its configuration.")
    # Checked here as well as by the team's own rules, because a count of one
    # would otherwise be read as a run of one agent rather than refused.
    count = data["config"].get("agents")
    if type(count) is not int or not 2 <= count <= MAX_AGENTS:
        raise ValueError(f"A team has 2 to {MAX_AGENTS} agents.")
    changing = "environment_id" in data["maze"]
    if legacy and (any(data["config"].get(key) for key in LEGACY_TEAM_KEYS) or changing
                   or str(data["config"].get("interruption_text") or "").strip()
                   or data.get("dropped_closures") or data.get("close_next")):
        raise ValueError(f"A team run with supplied moves, a waypoint, an interruption, a call limit, a changing map "
                         f"or an inserted message has to be recorded as {TEAM_FORMAT}.")
    manual = False if legacy else data.get("manual_intervention")
    if type(manual) is not bool:
        raise ValueError("A team run says whether anyone intervened in it.")
    # Taken out of the config and landed again one at a time, where each says
    # it landed, as the live run landed them.
    config = copy.deepcopy(data["config"])
    updates, inserts = config.pop("map_updates", []), config.pop("context_inserts", [])
    drops, pending = data.get("dropped_closures", []), data.get("close_next", [])
    if not all(isinstance(value, list) for value in (updates, inserts, drops, pending)):
        raise ValueError("A run's map changes, dropped closures, pending closure and inserted messages must be lists.")
    if not changing and (updates or drops or pending):
        raise ValueError("A team run whose map changes has to record the changing map, with its environment identifier.")
    if (updates or drops or pending or inserts) and not manual:
        raise ValueError("A run carrying a closure or an inserted message cannot report that nobody intervened in it.")
    result = Episode(load_maze(data["maze"]) if changing else Maze.from_dict(data["maze"]), config)
    # A queued interruption that had not landed leaves no other trace, and
    # one that has landed stays queued, so the flag is read from the file,
    # once the run says a reader asked for one. It is what lets an agent be
    # interrupted ahead of its own trigger, so it is read before the rounds.
    saved_agents, queued_agents = data.get("agents"), set()
    if not legacy:
        if not isinstance(saved_agents, list) or len(saved_agents) != len(result.agents):
            raise ValueError("The run's agents do not match what its responses produce.")
        for index, recorded in enumerate(saved_agents):
            queued = recorded.get("interrupt_next") if isinstance(recorded, dict) else None
            if type(queued) is not bool or (queued and not manual):
                raise ValueError("An agent's queued interruption is not one anyone asked for.")
            if queued:
                queued_agents.add(index)
    turns, rounds = data.get("turns"), data.get("rounds")
    if type(rounds) is not int or not 0 <= rounds <= result.config["round_limit"]:
        raise ValueError("The run's round count must be within its round limit.")
    if not isinstance(turns, list) or any(
            not isinstance(t, dict) or type(t.get("agent")) is not int or not 0 <= t["agent"] < len(result.agents)
            or type(t.get("round")) is not int or not isinstance(t.get("text"), str)
            or not isinstance(t.get("metrics"), list) or not all(isinstance(m, dict) and type(m.get("token_id")) is int
                                                                 for m in t["metrics"])
            or not isinstance(t.get("finish_reason"), str) for t in turns):
        raise ValueError("Each saved response needs its agent, round, text, tokens and finish reason.")
    by_round = {}
    for turn in turns:
        by_round.setdefault(turn["round"], []).append(turn)
    if [t["round"] for t in turns] != sorted(t["round"] for t in turns) or set(by_round) - set(range(rounds + 1)) \
            or set(range(rounds)) - set(by_round):
        raise ValueError("The saved responses are not in round order.")
    landed = sorted((record["agent"], record["before_round"]) for record in inserts
                    if isinstance(record, dict) and type(record.get("agent")) is int
                    and type(record.get("before_round")) is int)
    if len(set(landed)) != len(landed) or len(landed) != len(inserts):
        raise ValueError("Each message inserted into a team run names its agent and round, one to an agent a round.")
    by_answer = {(record["agent"], record["before_round"]): record for record in inserts}
    # Each message went in before a response the run records, and the list
    # holds them in the order those responses were asked for.
    asked_at = {(t["agent"], t["round"]): i for i, t in enumerate(turns)}
    places = [asked_at.get(key, -1) for key in by_answer]
    if -1 in places or places != sorted(places):
        raise ValueError("A team run's inserted messages are out of order, or name a response it never recorded.")
    updates = list(updates)
    replay_rounds(result, turns, rounds, updates, by_answer, manual, queued_agents)
    if updates:
        raise ValueError("A team run records a map change at a round it never reached, or out of order.")
    if by_answer:
        raise ValueError("A team run's inserted messages are out of order, or name a response it never recorded.")
    boundary = result.next_boundary()
    validate_drops(result.maze, drops, result.config.get("map_updates", []), boundary, "before_round")
    if pending and result.phase in TERMINAL:
        raise ValueError("A run that has ended cannot still be waiting to close a cell.")
    validate_pending(result.maze, pending, result.config.get("map_updates", []), drops, boundary, "before_round")
    result.dropped_closures, result.close_next = copy.deepcopy(drops), tuple(pending)
    # Before the prompts are read, since reading one falls back to the run's
    # model for a response that does not name its own.
    for key in ("model_id", "load_id"):
        if key in data:
            setattr(result, key, data[key])
    if read_prompt is not None:
        first = {}
        for record in result.config.get("context_inserts", ()):
            first.setdefault(record["agent"], record["before_round"])
        for index, turn in enumerate(result.turns):
            if turn["agent"] not in first or turn["round"] < first[turn["agent"]] or not turn.get("prompt_ids"):
                continue
            context = context_messages(result, index)
            reading = read_prompt(result, turn, context)
            if reading is None:
                continue
            prompt, templated = reading
            following = result.agents[turn["agent"]]["messages"][len(context):]
            if not (prompt == templated if templated is not None else prompt_holds(prompt, context, following)):
                raise ValueError(f"{result.agents[turn['agent']]['name']}'s recorded prompt in round {turn['round'] + 1} "
                                 "is not the history the run records for it, with the messages inserted up to then.")
    if result.phase in TERMINAL:
        if data.get("phase") != result.phase or len(by_round) > rounds:
            raise ValueError("The run reports an outcome other than the one its responses reach.")
    else:
        phase = data.get("phase")
        moving = [i for i, agent in enumerate(result.agents) if agent["status"] == "active"]
        starved = bool(moving) and result.response_caps(moving) is None
        if phase not in ("ready", "running", "paused", "stopped", "error", "budget") \
                or (phase == "ready" and turns) or (phase == "budget" and not starved):
            raise ValueError("The run reports an outcome other than the one its responses reach.")
        result.phase = phase
        if phase == "budget":
            settle_recoveries(result)
        if isinstance(data.get("detail"), str):
            result.detail = data["detail"]
    for index in queued_agents:
        result.agents[index]["interrupt_next"] = True
    kept = ("name", "position", "status", "messages") if legacy else None
    rebuilt = [{key: value for key, value in agent.items() if (kept is None or key in kept) and key != "insert_next"}
               for agent in json.loads(json.dumps(result.agents))]
    checks = {"responses": (turns, result.turns), "moves": (data.get("events"), result.events),
              "messages": (data.get("mail"), result.mail), "agents": (saved_agents, rebuilt),
              "sampled-token count": (data.get("sampled_tokens"), result.sampled_tokens),
              "call count": (data.get("tool_attempts"), result.tool_attempts)}
    if not legacy:
        checks["supplied-move count"] = (data.get("supplied_moves"), result.supplied_moves)
    for name, (recorded, replayed) in checks.items():
        if json.loads(json.dumps(recorded)) != json.loads(json.dumps(replayed)):
            raise ValueError(f"The run's {name} do not match what its responses produce."
                             if name.endswith("s") else f"The run's {name} does not match what its responses produce.")
    # A fork names the edit it was made by, which its edited response records
    # too; a fork of a fork keeps the earlier edit on the response it made. A
    # fork stopped before it regenerated names the response it never reached.
    token_edit = data.get("token_edit")
    if token_edit is not None:
        at = token_edit.get("turn") if isinstance(token_edit, dict) else None
        recorded = type(at) is int and 0 <= at < len(turns) and turns[at].get("token_edit") == token_edit
        unreached = at == len(turns) and result.phase in ("stopped", "paused")
        if legacy or not manual or not (recorded or unreached):
            raise ValueError("The run's token edit is not one its edited response records.")
    if token_edit is None and any("token_edit" in turn for turn in turns):
        raise ValueError("A run with a token-edited response names the edit it was made by.")
    result.rounds, result.manual_intervention, result.token_edit = rounds, manual, token_edit
    # A live run refuses to continue under another load, so every response
    # that reached the model names the load the run does.
    for key in ("model_id", "load_id"):
        if any(key in turn and turn[key] != data.get(key) for turn in turns):
            raise ValueError("The run names a model other than the one its responses were generated by.")
    for key in ("run_id", "model_id", "load_id", "created_at"):
        if key in data:
            setattr(result, key, data[key])
    result.replay_only = True
    result.viewing, result.selected_turn = -1, None
    return result
