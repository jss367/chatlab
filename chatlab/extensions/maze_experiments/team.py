"""What makes a run a team: several agents in one maze, moving in simultaneous rounds.

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

The rules a team adds to a run live here, apart from the episode that runs
them, so the episode and the reader of a saved team run take them from one
place. A run of one agent is a round of one response each time, and none of
this applies to it.
"""
from __future__ import annotations

import copy
import math
import re

from .maze import SYSTEM, TOOLS, default_instruction, goal_instruction

FORMAT = "chatlab-maze-team-2"
# A team run given exits and rewards: more than one exit, responses after
# arriving, a taste of the vector before the maze, or several laps. Written
# under a format of its own, which a ChatLab without those rules refuses
# rather than reads with every response after the first arrival misplaced.
REWARD_FORMAT = "chatlab-maze-team-3"
# Written before a team could be given supplied moves, a waypoint, an
# interruption or a limit on its calls. Still read, never written.
LEGACY_FORMAT = "chatlab-maze-team-1"
TEAM_GOALS = {"any": "Any agent arrives", "all": "Every agent arrives"}
# Long enough for a plan ("I'll take the east corridor, you go south"), short
# enough that a teammate's reply stays a reply about the maze.
MESSAGE_LIMIT = 280
MAX_AGENTS = 100
# An agent stops being asked for responses once it is anything but active:
# arrived, abandoned, cut_off, out_of_tokens, out_of_calls or not_recovered.
# The statuses a limit of the run's own gives an agent. A team whose agents
# all stopped on one, or arrived, ran out of budget rather than gave up.
LIMITED = {"out_of_tokens", "out_of_calls", "not_recovered"}
# What a response that takes no action does to its agent once its round resolves.
DROPPED = {"no_call": "abandoned", "cut_off": "cut_off"}
DEFAULT_CONFIG = dict(communication=True, team_goal="any", goal_mode="coordinates", goal_hint="",
                      temperature=.7, sampling_seed=20260914, per_turn_tokens=1024, round_limit=24)
# Each agent has its own sampled-token limit. Runs saved before that shared
# one limit across the team, ``token_budget``, and still replay under it.
AGENT_TOKEN_BUDGET = 8192


TARGET_VERBS = {"steer_agents": "steer", "interrupt_agents": "interrupt"}

# What a team is given under exits and rewards, any one of which makes it a
# run of that kind, with the value each takes when the run does not name it.
REWARD_DEFAULTS = dict(
    exits=[], reward_exit=None, arrival_responses=0,
    arrival_prompt=(f"Write a message for your teammates, up to {MESSAGE_LIMIT} characters. "
                    "It is sent to them as you write it."),
    # The taste prompt names no vector, so a steered taste and an unsteered
    # one are asked the same question and differ only in how they are answered.
    taste=False, taste_prompt="Before the maze begins, describe how you feel right now.",
    taste_strength=None, laps=1, lap_prompt="", paired_exits=False, team_reward=0, message_rules=2)
# How a response after arriving becomes its message, by the version a run
# records. 1: the text outside its reasoning, sent only if the response
# finished. 2: also sent when cut off, with any call taken out.
MESSAGE_RULES = (1, 2)
# Exits B, C and D beside the destination, which is exit A.
EXIT_LABELS = "ABCD"


def rewarded(config):
    """Whether a team configuration asks for exits and rewards."""
    return any(key in config for key in REWARD_DEFAULTS)


def exit_cells(maze, config):
    """Each exit's label and cell, the destination first as A, or None for a run with one way out."""
    extra = config.get("exits") or []
    if not extra:
        return None
    return {label: tuple(cell) for label, cell in zip(EXIT_LABELS, [maze.goal, *extra])}


def arrival_text(config, label, first):
    """The message that opens a response given after arriving: where the agent arrived, then the run's prompt."""
    if not first:
        return config["arrival_prompt"]
    where = f"You reached exit {label}." if config["exits"] else "You reached the destination."
    return " ".join(filter(None, [where, config["arrival_prompt"]]))


def lap_text(config, lap, state):
    """The message that starts lap ``lap``: which lap it is, the run's prompt, and the state at the start."""
    header = f"Lap {lap} of {config['laps']} begins. Every agent still in the run is back at the start."
    return "\n".join(filter(None, [" ".join(filter(None, [header, config["lap_prompt"]])), state]))


def targeted(config, name, count):
    """The agents a team's ``steer_agents`` or ``interrupt_agents`` picks out, every agent when it names none."""
    return config.get(name) or range(count)


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


def team_paragraph(name, names, goal, communicate, exits=False):
    """What an agent is told about its team, appended to the task instruction."""
    others = [other for other in names if other != name]
    listed = ", ".join(others[:-1]) + (" and " if len(others) > 1 else "") + others[-1]
    text = (f"You are {name}, one of {len(names)} agents in this maze. Your teammate"
            f"{'s are' if len(others) > 1 else ' is'} {listed}. Each agent moves separately, and every "
            "agent's move in a round happens at the same time. ")
    if exits:
        text += "The team succeeds when every agent has reached an exit. An agent that arrives stops moving. "
    else:
        text += ("The team succeeds as soon as any agent reaches the destination. " if goal == "any" else
                 "The team succeeds when every agent has reached the destination. An agent that arrives stops moving. ")
    if communicate:
        text += (f"You may add a message of up to {MESSAGE_LIMIT} characters for your teammates as the move "
                 "call's message argument. They read it in their next simulator reply, and you read theirs in yours.")
    else:
        text += "You cannot communicate with your teammates."
    return text


def parse_agents(text, count, verb="steer"):
    """The agents named by text such as "1, 3, 5-8", as indices, or None for every agent."""
    text = str(text or "").strip()
    if text.lower() in ("", "all"):
        return None
    targets = []
    for part in text.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise ValueError(f"Name the agents to {verb} by number, such as 1, 3, 5-8, or leave the box blank for all.")
        low, high = int(match[1]), int(match[2] or match[1])
        if not 1 <= low <= high <= count:
            raise ValueError(f"Agents to {verb} must be numbered 1 to {count}.")
        targets += [i - 1 for i in range(low, high + 1) if i - 1 not in targets]
    # Naming every agent is naming all of them, and is stored as such.
    return None if sorted(targets) == list(range(count)) else targets


def format_agents(targets, count):
    """The inverse of parse_agents: runs of agents written as ranges, 1-based."""
    if targets is None or sorted(targets) == list(range(count)):
        return "all"
    runs = []
    for index in sorted(targets):
        if runs and runs[-1][1] == index - 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return ", ".join(str(a + 1) if a == b else f"{a + 1}-{b + 1}" for a, b in runs)


def check_config(config):
    """Fill a team configuration's defaults and refuse one no run could use."""
    config = copy.deepcopy(config)
    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)
    if "token_budget" in config and "agent_token_budget" in config:
        raise ValueError("A team run has either a limit per agent or one for the whole team, not both.")
    if "token_budget" not in config:
        config.setdefault("agent_token_budget", AGENT_TOKEN_BUDGET)
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
                              "token_budget": (1, 131072), "agent_token_budget": (1, 131072),
                              "round_limit": (1, 256), "attempt_budget": (1, 256), "supplied_moves": (0, 223),
                              "interrupt_after": (0, 255), "prefix_tokens": (0, 1024)}.items():
        if name in config and (type(config[name]) is not int or not low <= config[name] <= high):
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    if not isinstance(config.setdefault("interruption_text", ""), str):
        raise ValueError("interruption_text must be text.")
    if config["interruption_text"].strip():
        config.setdefault("interrupt_after", 0)
        config.setdefault("prefix_tokens", 0)
    for name in ("steer_agents", "interrupt_agents"):
        targets = config.get(name)
        if targets is not None and (not isinstance(targets, list) or not targets
                                   or any(type(i) is not int or not 0 <= i < config["agents"] for i in targets)
                                   or len(set(targets)) != len(targets)):
            raise ValueError(f"Choose one or more distinct agents in this team to {TARGET_VERBS[name]}.")
    temperature = config["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    return config
