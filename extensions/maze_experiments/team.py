"""Team episodes: several agents in one maze, moving in simultaneous rounds.

One loaded model plays every agent. The host hands an extension one model
session at a time, and holding a second model in memory beside the first is
what a machine running these experiments usually cannot afford, so the agents
are separate conversations rather than separate models: each has its own
history, its own position and its own sampling seed, and none reads another's
prompt.

A round asks every agent still moving for one response, each against the
state as the round began, and only then applies the moves. The agent asked
first therefore has no advantage over the agent asked last, which is what
"at the same time" means for a single model that generates one response after
another. Each agent's reply for the round is written after every move in it
has landed, and carries whatever its teammates said in that round when
communication is on.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .maze import SYSTEM, TOOLS, Maze, apply_call, default_instruction, goal_instruction, parse_call
from extension_api import write_private_text

logger = logging.getLogger(__name__)

FORMAT = "chatlab-maze-team-1"
TERMINAL = {"arrived", "abandoned", "budget", "stopped", "error"}
TEAM_GOALS = {"any": "Any agent arrives", "all": "Every agent arrives"}
# Long enough for a plan ("I'll take the east corridor, you go south"), short
# enough that a teammate's reply stays a reply about the maze.
MESSAGE_LIMIT = 280
MAX_AGENTS = 4
# An agent stops being asked for responses once it is anything but active.
AGENT_STATUSES = {"active", "arrived", "abandoned", "cut_off"}
# What a response that takes no action does to its agent once its round resolves.
DROPPED = {"no_call": "abandoned", "cut_off": "cut_off"}
DEFAULT_CONFIG = dict(agents=2, communication=True, team_goal="any", goal_mode="coordinates", goal_hint="",
                      temperature=.7, sampling_seed=20260914, per_turn_tokens=1024, token_budget=16384,
                      round_limit=24)


def agent_names(count):
    return [f"agent-{index + 1}" for index in range(count)]


def team_tools(communicate):
    """The move tool, with an optional message argument when agents may talk.

    The message rides on the move call rather than on a tool of its own,
    because the simulator executes exactly one call per response: a separate
    tool would make every message cost the agent its move.
    """
    tools = copy.deepcopy(TOOLS)
    if communicate:
        function = tools[0]["function"]
        function["description"] = ("Move one cell in the specified direction in the identified maze, "
                                   "optionally sending your teammates a message.")
        function["parameters"]["properties"]["message"] = {
            "type": "string", "maxLength": MESSAGE_LIMIT,
            "description": "Optional. Every teammate reads it in their next simulator reply."}
    return tools


def team_paragraph(name, names, goal, communicate):
    """What an agent is told about its team, appended to the task instruction."""
    others = [other for other in names if other != name]
    listed = ", ".join(others[:-1]) + (" and " if len(others) > 1 else "") + others[-1]
    text = (f"You are {name}, one of {len(names)} agents in this maze. Your teammate"
            f"{'s are' if len(others) > 1 else ' is'} {listed}. Each agent moves separately, and every "
            "agent's move in a round happens at the same time. ")
    text += ("The team succeeds as soon as any agent reaches the destination. " if goal == "any" else
             "The team succeeds when every agent has reached the destination. An agent that arrives stops moving. ")
    if communicate:
        text += (f"You may add a message of up to {MESSAGE_LIMIT} characters for your teammates as the move "
                 "call's message argument. They read it in their next simulator reply, and you read theirs in yours.")
    else:
        text += "You cannot communicate with your teammates."
    return text


def check_config(config):
    """Fill a team configuration's defaults and refuse one no run could use."""
    config = copy.deepcopy(config)
    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)
    if not isinstance(config.get("system_prompt"), str):
        config["system_prompt"] = SYSTEM
    if not isinstance(config.get("instruction"), str):
        config["instruction"] = default_instruction(config["goal_mode"])
    if type(config["agents"]) is not int or not 2 <= config["agents"] <= MAX_AGENTS:
        raise ValueError(f"A team has 2 to {MAX_AGENTS} agents.")
    if type(config["communication"]) is not bool:
        raise ValueError("Communication is either on or off.")
    if config["team_goal"] not in TEAM_GOALS:
        raise ValueError("Choose whether any agent or every agent has to reach the destination.")
    goal_instruction(config["goal_mode"], config["goal_hint"])
    for name, (low, high) in {"sampling_seed": (0, 2147483647), "per_turn_tokens": (1, 8192),
                              "token_budget": (1, 131072), "round_limit": (1, 256)}.items():
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    temperature = config["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    return config


@dataclass
class TeamEpisode:
    maze: Maze
    config: dict
    run_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "ready"
    detail: str = "Ready. Play runs rounds until the team finishes; Next runs one round."
    # One entry per agent: its name, position, status and its own conversation.
    agents: list = field(default_factory=list)
    # Every response, each naming the agent that gave it and its round.
    turns: list = field(default_factory=list)
    # Every attempted call, in the order the rounds applied them.
    events: list = field(default_factory=list)
    # Every message an agent sent, with the teammates who received it.
    mail: list = field(default_factory=list)
    rounds: int = 0
    model_id: str | None = None
    load_id: str | None = None
    sampled_tokens: int = 0
    tool_attempts: int = 0
    pause_requested: bool = False
    stop_requested: bool = False
    busy: bool = False
    replay_only: bool = False
    # The round the viewer last drew, -1 being the start, and the response
    # selected in it.
    viewing: int = -1
    selected_turn: int | None = None
    playback_token: int = 0
    playing: bool = False
    created_at: float = field(default_factory=time.time)

    def __deepcopy__(self, memo):
        # As Episode: Gradio copies the initial State once per session, and each
        # session needs its own run and its own lock.
        result = object.__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, threading.RLock() if key == "lock" else copy.deepcopy(value, memo))
        if result.phase == "ready" and not result.turns:
            result.run_id = uuid4().hex
            result.created_at = time.time()
        return result

    def __post_init__(self):
        self.lock = threading.RLock()
        self.config = check_config(self.config)
        names = agent_names(self.config["agents"])
        self.agents = [dict(name=name, position=self.maze.start, status="active", messages=[]) for name in names]
        for index, agent in enumerate(self.agents):
            instruction = "\n".join(filter(None, [
                self.config["instruction"],
                team_paragraph(agent["name"], names, self.config["team_goal"], self.config["communication"])]))
            state = json.dumps(self.agent_state(index), separators=(",", ":"))
            agent["messages"] = [{"role": "system", "content": self.config["system_prompt"]},
                                 {"role": "user", "content": instruction + "\n" + state}]

    @property
    def names(self):
        return [agent["name"] for agent in self.agents]

    @property
    def tools(self):
        return team_tools(self.config["communication"])

    @property
    def moves(self):
        return sum(event["accepted"] for event in self.events)

    def agent_state(self, index, error=None, inbox=()):
        """What the simulator tells one agent: its own position, never its teammates'.

        Teammates' positions are left out on purpose. With communication off,
        an agent that could still see where the others stand would be
        coordinating through the board, and the comparison the switch exists
        for would measure nothing.
        """
        agent = self.agents[index]
        state = {"agent": agent["name"], "teammates": [n for n in self.names if n != agent["name"]]}
        state.update(self.maze.state(agent["position"], error, goal_mode=self.config["goal_mode"],
                                     goal_hint=self.config["goal_hint"]))
        if self.config["communication"]:
            state["messages"] = list(inbox)
        return state

    def context_messages(self, turn_index):
        """The messages the response at ``turn_index`` was given.

        An agent's history grows by two messages for each round it made a call
        in, and by none for the round it stopped without one.
        """
        turn = self.turns[turn_index]
        calls = sum(1 for event in self.events if event["agent"] == turn["agent"] and event["round"] < turn["round"])
        return self.agents[turn["agent"]]["messages"][:2 + 2 * calls]

    def payload(self):
        keys = ("run_id", "phase", "detail", "agents", "turns", "events", "mail", "rounds", "model_id", "load_id",
                "sampled_tokens", "tool_attempts", "created_at")
        with self.lock:
            return {"format": FORMAT, "maze": self.maze.to_dict(), "config": self.config, "exploratory": True,
                    **{key: getattr(self, key) for key in keys}}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        temp = path.with_suffix(".json.tmp")
        with self.lock:
            text = json.dumps(self.payload(), ensure_ascii=False, allow_nan=False)
        write_private_text(temp, text)
        temp.replace(path)
        return path

    def export(self):
        return self.save(Path(tempfile.mkdtemp(prefix="chatlab-maze-team-")))

    def request_pause(self):
        self.pause_requested = True

    def request_stop(self, save_dir=None):
        with self.lock:
            if self.replay_only or self.phase in TERMINAL:
                return
            self.stop_requested = True
            if self.busy:
                return  # The active stream owns cleanup and persistence.
            self.phase, self.detail = "stopped", "Stopped by you. This is not scored as the team giving up."
            if save_dir:
                try:
                    self.save(save_dir)
                except OSError as exc:
                    logger.warning("Autosave of team run %s failed: %s", self.run_id, exc)
                    self.detail += f" Autosave failed: {exc}. Use Export run JSON to keep this run."


def finish_response(episode, turn, stop_ids, max_tokens):
    """Read one agent's finished response into the action it takes this round.

    Returns the action, or None for a response that takes none. A response
    that ends without a call, or is cut off by a limit, takes its agent out of
    the run, as either ends a single-agent episode; its teammates carry on.
    The agent leaves when the round resolves, so a round that never does
    leaves it in the team.
    """
    sampled = turn["metrics"]
    if episode.stop_requested:
        turn["finish_reason"] = "user_stopped"
    elif sampled and sampled[-1]["token_id"] in stop_ids:
        turn["finish_reason"] = "stop"
    else:
        turn["finish_reason"] = "length" if len(sampled) >= max_tokens else "incomplete_stream"
    return take_action(episode, turn, len(episode.turns) - 1)


def take_action(episode, turn, index):
    """Count a response's tokens and read the action its finish reason allows.

    Shared by generation and by replay, so a saved run is read back through
    exactly the rules that produced it.
    """
    sampled = turn["metrics"]
    episode.sampled_tokens += len(sampled)
    turn["sampled_tokens"] = len(sampled)
    turn["tokens_cumulative"] = episode.sampled_tokens
    if turn["finish_reason"] == "user_stopped":
        return None
    if turn["finish_reason"] != "stop":
        turn["outcome"] = "cut_off"
        return None
    text = turn["text"]
    if turn.get("reasoning_prefilled"):
        text = "<think>" + text
    content = text
    # Tool-looking text inside reasoning is not an external action.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    args, error = parse_call(text, message_limit=MESSAGE_LIMIT if episode.config["communication"] else None)
    if args is None and error is None:
        turn["outcome"] = "no_call"
        return None
    episode.tool_attempts += 1
    return dict(agent=turn["agent"], turn=index, content=content, args=args, error=error)


def resolve_round(episode, actions):
    """Apply every action of a round at once, then write each agent's reply.

    Every call is judged from the position its agent held when the round
    began, so the order the agents were asked in changes nothing. Agents may
    share a cell. A message goes to every teammate that gets a reply this
    round, which is every teammate that made a call in it.
    """
    round_index = episode.rounds
    for turn in episode.turns:
        if turn["round"] == round_index and turn.get("outcome") in DROPPED:
            episode.agents[turn["agent"]]["status"] = DROPPED[turn["outcome"]]
    sent = []
    for action in actions:
        agent = episode.agents[action["agent"]]
        if action["error"]:
            event = {"accepted": False, "before": list(agent["position"]), "after": list(agent["position"]),
                     "error": action["error"], "arrived": False, "progress": False}
        else:
            event = apply_call(episode.maze, agent["position"], action["args"], goal_mode=episode.config["goal_mode"])
            message = action["args"].get("message", "").strip()
            if message:
                event["message"] = message
                sent.append((action["agent"], message))
        event.update(source="model", agent=action["agent"], round=round_index, turn=action["turn"])
        episode.events.append(event)
        episode.turns[action["turn"]]["event"] = event
        action["event"] = event
    for action in actions:
        agent, event = episode.agents[action["agent"]], action["event"]
        if event["accepted"]:
            agent["position"] = tuple(event["after"])
        if event["arrived"]:
            agent["status"] = "arrived"
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
    arrived = [agent["name"] for agent in episode.agents if agent["status"] == "arrived"]
    moving = [agent for agent in episode.agents if agent["status"] == "active"]
    moved = sum(action["event"]["accepted"] for action in actions)
    if arrived and (episode.config["team_goal"] == "any" or len(arrived) == len(episode.agents)):
        episode.phase = "arrived"
        episode.detail = (f"{', '.join(arrived)} reached the destination in round {episode.rounds}."
                          if episode.config["team_goal"] == "any" else
                          f"Every agent reached the destination by round {episode.rounds}.")
    elif not moving:
        episode.phase = "abandoned"
        episode.detail = "No agent is still moving: " + ", ".join(
            f"{agent['name']} {agent['status'].replace('_', ' ')}" for agent in episode.agents) + "."
    elif episode.sampled_tokens >= episode.config["token_budget"]:
        episode.phase, episode.detail = "budget", "The team reached its sampled-token limit."
    elif episode.rounds >= episode.config["round_limit"]:
        episode.phase, episode.detail = "budget", "The team reached its round limit."
    else:
        episode.detail = (f"Round {episode.rounds}: {moved} of {len(actions)} call{'' if len(actions) == 1 else 's'} "
                          f"moved an agent, {len(sent)} message{'' if len(sent) == 1 else 's'} sent.")


def discard_round(episode):
    """Mark the responses of a round that will never resolve as not applied.

    None of it is applied: the agents that answered would otherwise have moved
    while their teammates had not. That includes a response that made no call
    or was cut off, whose agent would otherwise have left the team in a round
    that never happened; its text is kept either way. A resolved round has
    nothing left to mark, so this is safe to call however the stream ends.
    """
    for waiting in episode.turns:
        if waiting["round"] == episode.rounds and "event" not in waiting:
            waiting["outcome"] = "not_applied"


def stream_team(episode, models, *, single_step=False, save_dir=None):
    """Generate rounds until the team finishes, pauses or is stopped.

    Yields the episode after every frame of every response and after every
    round. Pause waits for the round to finish, so no agent is left having
    moved while a teammate has not. Stop does not wait: the round it lands in
    is discarded, its finished responses kept and marked as never applied.
    """
    with episode.lock:
        if episode.busy:
            raise ValueError("This team episode is already generating. Pause it before changing the run.")
        if episode.phase in TERMINAL or episode.replay_only:
            raise ValueError("Start a new team episode to run again. This one is finished or is a saved replay.")
        manager = models.open_session()
        episode.busy = True
        episode.pause_requested = episode.stop_requested = False
        episode.phase = "running"
        episode.model_id = episode.model_id or manager.model_id
        episode.load_id = episode.load_id or manager.load_id
    logger.info("Team run %s generating with %s: %s agents, communication %s, round %s, %s",
                episode.run_id, episode.model_id, len(episode.agents),
                "on" if episode.config["communication"] else "off", episode.rounds + 1,
                "one round" if single_step else "until it ends")
    turn = None
    autosave_error = None

    def autosave():
        nonlocal autosave_error
        if not save_dir or autosave_error is not None:
            return
        try:
            episode.save(save_dir)
        except OSError as exc:
            autosave_error = str(exc)
            episode.pause_requested = True

    try:
        while episode.phase == "running":
            if episode.stop_requested:
                episode.phase, episode.detail = "stopped", "Stopped by you before the next round."
                break
            if episode.pause_requested:
                episode.phase, episode.detail = "paused", episode.detail + " Paused before the next round."
                break
            if manager.load_id != episode.load_id:
                raise ValueError("The model changed during this episode. Start a new team episode with the selected model.")
            moving = [i for i, agent in enumerate(episode.agents) if agent["status"] == "active"]
            # The budget left is split before anyone answers, so an agent asked
            # later in the round is capped exactly as the first one was.
            limit = min(episode.config["per_turn_tokens"],
                        (episode.config["token_budget"] - episode.sampled_tokens) // len(moving))
            if limit <= 0:
                episode.phase, episode.detail = "budget", ("The team's remaining sampled tokens cannot give every "
                                                           "moving agent a response.")
                break
            actions = []
            for index in moving:
                agent = episode.agents[index]
                turn = {"agent": index, "round": episode.rounds, "text": "", "metrics": [], "prompt_ids": [],
                        "position_before": list(agent["position"]), "started_at": time.time(), "finish_reason": None}
                episode.turns.append(turn)
                episode.selected_turn = len(episode.turns) - 1
                yield episode
                if episode.stop_requested:
                    finish_response(episode, turn, set(), limit)
                    break
                stop_ids = manager.stop_token_ids
                generator = manager.generate(
                    agent["messages"], temperature=episode.config["temperature"], top_p=1., top_k=0,
                    max_new_tokens=limit, analyze_prompt=False, tools=episode.tools, forced_ids=[],
                    literal_prefill_tokens=0,
                    # Distinct per agent as well as per round: agents given the
                    # same prompt would otherwise sample the same response.
                    seed=episode.config["sampling_seed"] + 100003 * episode.rounds + 7919 * index)
                try:
                    for update in generator:
                        turn.update(text=update.text, metrics=copy.deepcopy(update.metrics),
                                    prompt_ids=list(update.prompt_ids), reasoning_prefilled=update.reasoning_prefilled,
                                    load_id=update.load_id, model_id=update.model_id)
                        yield episode
                        if episode.stop_requested:
                            break
                finally:
                    generator.close()
                turn["seconds"] = time.time() - turn["started_at"]
                action = finish_response(episode, turn, stop_ids, limit)
                logger.info("Team run %s round %s %s: %s after %s sampled tokens in %.1fs",
                            episode.run_id, episode.rounds + 1, agent["name"], turn.get("outcome") or turn["finish_reason"],
                            turn["sampled_tokens"], turn["seconds"])
                if action:
                    actions.append(action)
                yield episode
                if episode.stop_requested:
                    break
            if episode.stop_requested:
                discard_round(episode)
                episode.phase = "stopped"
                episode.detail = f"Stopped by you during round {episode.rounds + 1}. Its responses were kept and none of its moves applied."
                break
            resolve_round(episode, actions)
            autosave()
            episode.viewing, episode.selected_turn = episode.rounds - 1, None
            yield episode
            if episode.phase in TERMINAL:
                break
            if single_step or episode.pause_requested:
                episode.phase, episode.detail = "paused", episode.detail + " Paused before the next round."
                break
            time.sleep(.25)
    except GeneratorExit:
        if episode.phase == "running":
            episode.phase, episode.detail = "stopped", "Viewer stopped streaming. The partial response was retained."
        raise
    except Exception as exc:
        logger.exception("Team run %s failed while generating", episode.run_id)
        episode.phase, episode.detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if turn is not None and turn.get("finish_reason") is None:
            count = len(turn["metrics"])
            episode.sampled_tokens += count
            turn.update(sampled_tokens=count, tokens_cumulative=episode.sampled_tokens, finish_reason=episode.phase,
                        outcome="not_applied")
        # A failure or a viewer hanging up mid-round never reaches the round's
        # resolution either, so the teammates that already answered are marked
        # the same way a stop marks them.
        discard_round(episode)
        with episode.lock:
            episode.busy = False
            manager.close()
            autosave()
            if autosave_error is not None:
                logger.warning("Autosave of team run %s failed: %s", episode.run_id, autosave_error)
                episode.detail += (f" Autosave failed: {autosave_error}. Latest changes remain in memory. "
                                   "Use Export run JSON to download them.")
        logger.info("Team run %s %s after %s rounds, %s responses and %s sampled tokens, %s moves: %s",
                    episode.run_id, episode.phase, episode.rounds, len(episode.turns), episode.sampled_tokens,
                    episode.moves, episode.detail)
    yield episode


def from_payload(data):
    """A saved team run, rebuilt for replay from the responses it records.

    Nothing the run derived is taken as written. Every recorded response is
    read again through the rules that read it live, round by round, and the
    moves, messages, histories, positions, statuses, counters and outcome that
    produces are compared with the file's. A file that disagrees anywhere
    describes a run these responses could not have made, and is refused.
    """
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Choose a ChatLab maze team run JSON file.")
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    if not isinstance(data.get("maze"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("The run is missing its map or its configuration.")
    maze = Maze.from_dict(data["maze"])
    result = TeamEpisode(maze, data["config"])
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
    derived = {"event", "outcome", "sampled_tokens", "tokens_cumulative"}
    for round_index in range(rounds + 1):
        if round_index not in by_round:
            continue
        if result.phase in TERMINAL:
            raise ValueError("The run records responses after it had ended.")
        moving = [i for i, agent in enumerate(result.agents) if agent["status"] == "active"]
        asked = [t["agent"] for t in by_round[round_index]]
        resolved = round_index < rounds
        if asked != moving if resolved else asked != moving[:len(asked)]:
            raise ValueError("A round asks agents other than the ones still moving, in their order.")
        actions = []
        for saved in by_round[round_index]:
            if resolved and saved["finish_reason"] not in ("stop", "length", "incomplete_stream"):
                raise ValueError("A response in a finished round ends in a way no finished round records.")
            turn = copy.deepcopy({key: value for key, value in saved.items() if key not in derived})
            result.turns.append(turn)
            action = take_action(result, turn, len(result.turns) - 1)
            if action:
                actions.append(action)
        if resolved:
            resolve_round(result, actions)
        else:
            discard_round(result)
    if result.phase in TERMINAL:
        if data.get("phase") != result.phase or len(by_round) > rounds:
            raise ValueError("The run reports an outcome other than the one its responses reach.")
    else:
        phase = data.get("phase")
        moving = [agent for agent in result.agents if agent["status"] == "active"]
        starved = bool(moving) and (result.config["token_budget"] - result.sampled_tokens) // len(moving) <= 0
        if phase not in ("ready", "running", "paused", "stopped", "error", "budget") \
                or (phase == "ready" and turns) or (phase == "budget" and not starved):
            raise ValueError("The run reports an outcome other than the one its responses reach.")
        result.phase = phase
        if isinstance(data.get("detail"), str):
            result.detail = data["detail"]
    saved_agents = data.get("agents")
    rebuilt = json.loads(json.dumps(result.agents))
    checks = {"responses": (turns, result.turns), "moves": (data.get("events"), result.events),
              "messages": (data.get("mail"), result.mail), "agents": (saved_agents, rebuilt),
              "sampled-token count": (data.get("sampled_tokens"), result.sampled_tokens),
              "call count": (data.get("tool_attempts"), result.tool_attempts)}
    for name, (recorded, replayed) in checks.items():
        if json.loads(json.dumps(recorded)) != json.loads(json.dumps(replayed)):
            raise ValueError(f"The run's {name} do not match what its responses produce."
                             if name.endswith("s") else f"The run's {name} does not match what its responses produce.")
    result.rounds = rounds
    for key in ("run_id", "model_id", "load_id", "created_at"):
        if key in data:
            setattr(result, key, data[key])
    result.replay_only = True
    result.viewing, result.selected_turn = -1, None
    return result
