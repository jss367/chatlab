"""Deterministic maze state and optional movement tools for exploratory single runs."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass

DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
SYSTEM = "You are a helpful assistant."
INSTRUCTION = (
    "Navigate from the current position to the destination using legal moves. "
    "Use the move tool to change position. The simulator's map and position are authoritative. "
    "The task ends when the simulator reports arrival at the destination."
)
GOAL_MODES = {"coordinates": "Exact coordinates", "hidden": "Hidden location", "hint": "Hint only"}


HIDDEN_INSTRUCTION = (
    "There is a destination somewhere in an open cell of this maze. Its location is hidden. "
    "Explore using legal moves until you find it. Use the move tool to change position. "
    "The simulator's map and position are authoritative. "
    "The task ends when the simulator reports arrival at the destination."
)
HINT_INSTRUCTION = HIDDEN_INSTRUCTION + " Use the goal_hint in the state as a clue to the destination."


def default_instruction(mode):
    """The instruction a goal mode sends when the scenario supplies no wording of its own."""
    if mode not in GOAL_MODES:
        raise ValueError("Choose exact coordinates, hidden location, or hint only for goal information.")
    return {"coordinates": INSTRUCTION, "hidden": HIDDEN_INSTRUCTION, "hint": HINT_INSTRUCTION}[mode]


def goal_instruction(mode, hint=""):
    instruction = default_instruction(mode)
    if mode == "hint" and (not isinstance(hint, str) or not hint.strip()):
        raise ValueError("Enter a goal hint for hint-only mode.")
    return instruction


TOOLS = [{"type": "function", "function": {
    "name": "move", "description": "Move one cell in the specified direction in the identified maze.",
    "parameters": {"type": "object", "properties": {
        "maze_id": {"type": "string"},
        "direction": {"type": "string", "enum": list(DIRECTIONS)}},
        "required": ["maze_id", "direction"], "additionalProperties": False}}}]
PASSAGES = {
    "Unrelated · bicycles": "A bicycle frame connects two wheels and supports the rider. The tubes form a rigid structure that carries the load. Different materials change the weight and flexibility of the frame. Steel and aluminum have different manufacturing requirements and surface finishes.",
    "Unrelated · music": "A musical instrument produces sound when a part of it vibrates. The shape and material of the instrument influence its tone. Strings, reeds, and membranes produce vibrations in different ways. The surrounding air carries those vibrations to the listener.",
    "Unrelated · rivers": "A river carries water through a channel in the ground. Its banks contain layers of sediment deposited over time. The water transports particles of different sizes. Seasonal changes affect the amount of water and the materials carried along the riverbed.",
    "Matched · navigation": "A map represents the arrangement of locations in a space. Its symbols describe features of the environment. Coordinates identify locations consistently. Walls and open areas have different markings. A destination has a particular location within the area represented by the map.",
}


@dataclass(frozen=True)
class Maze:
    grid: tuple[str, ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    seed: int = 0

    def __post_init__(self):
        n = len(self.grid)
        if not 3 <= n <= 15 or any(len(row) != n or set(row) - {".", "#"} for row in self.grid):
            raise ValueError("Use a square maze from 3 to 15 cells wide, containing only open cells and walls.")
        for label, point in (("Start", self.start), ("Destination", self.goal)):
            if len(point) != 2 or any(type(x) is not int for x in point) or not self.open(point):
                raise ValueError(f"{label} must be an open cell inside the maze.")
        if self.start == self.goal:
            raise ValueError("Start and destination must be different cells.")
        if self.goal not in self.distances(self.start):
            raise ValueError("There is no route from the start to the destination.")

    @property
    def size(self):
        return len(self.grid)

    @property
    def maze_id(self):
        raw = json.dumps([self.grid, self.start, self.goal], separators=(",", ":"))
        return "maze-" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    def tool_id(self, goal_mode="coordinates"):
        if goal_mode == "coordinates":
            return self.maze_id
        # A hash containing the goal can be enumerated over the visible cells.
        # Concealed goals therefore use an identifier independent of the goal.
        raw = json.dumps([self.grid, self.start], separators=(",", ":"))
        return "maze-" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    def open(self, point):
        r, c = point
        return 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == "."

    def neighbors(self, point):
        return {d: (point[0] + dr, point[1] + dc) for d, (dr, dc) in DIRECTIONS.items()
                if self.open((point[0] + dr, point[1] + dc))}

    def distances(self, origin):
        found = {origin: 0}
        todo = deque([origin])
        while todo:
            p = todo.popleft()
            for q in self.neighbors(p).values():
                if q not in found:
                    found[q] = found[p] + 1
                    todo.append(q)
        return found

    def route(self, position=None):
        position = self.start if position is None else tuple(position)
        distances = self.distances(self.goal)
        if position not in distances:
            return []
        path = [position]
        while path[-1] != self.goal:
            path.append(next(q for q in self.neighbors(path[-1]).values() if distances[q] < distances[path[-1]]))
        return path

    def state(self, position, error=None, *, goal_mode="coordinates", goal_hint=""):
        goal_instruction(goal_mode, goal_hint)
        state = {"maze_id": self.tool_id(goal_mode), "row_labels": list(range(self.size)),
                "column_labels": list(range(self.size)), "grid": list(self.grid),
                "current": list(position)}
        if goal_mode == "coordinates":
            state["destination"] = list(self.goal)
        elif goal_mode == "hint":
            state["goal_hint"] = goal_hint
        state.update(valid_directions=list(self.neighbors(position)) if tuple(position) != self.goal else [],
                     arrived=tuple(position) == self.goal, error=error)
        return state

    def to_dict(self):
        return {**asdict(self), "maze_id": self.maze_id, "distance": len(self.route()) - 1}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value["grid"]), tuple(value["start"]), tuple(value["goal"]), int(value.get("seed", 0)))


def generate(size=5, seed=20260911, distance=10, openness=.70):
    size, seed, distance = int(size), int(seed), int(distance)
    if not 3 <= size <= 15 or not 1 <= distance < size * size or not .35 <= openness <= .95:
        raise ValueError("Choose size 3–15, a feasible positive route length, and 35–95% open cells.")
    rng = random.Random(seed)
    deadline = time.monotonic() + 8
    for _ in range(15000):
        if time.monotonic() > deadline:
            break
        cells = {(r, c) for r in range(size) for c in range(size) if rng.random() < openness}
        if len(cells) <= distance:
            continue
        neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRECTIONS.values()
                         if (p[0] + dr, p[1] + dc) in cells] for p in cells}
        def distances(origin):
            found, todo = {origin: 0}, deque([origin])
            while todo:
                p = todo.popleft()
                for q in neighbors[p]:
                    if q not in found:
                        found[q] = found[p] + 1
                        todo.append(q)
            return found
        first = min(cells)
        if len(distances(first)) != len(cells):
            continue
        pairs = [(p, q) for p in sorted(cells) for q, d in distances(p).items() if d == distance]
        if pairs:
            start, goal = rng.choice(pairs)
            return Maze(tuple("".join("." if (r, c) in cells else "#" for c in range(size))
                              for r in range(size)), start, goal, seed)
        if time.monotonic() > deadline:
            break
    raise ValueError("No maze with that route length was found. Try a shorter route, a different seed, or more open cells.")


def call_text(maze_id, direction):
    return '<tool_call>\n' + json.dumps({"name": "move", "arguments": {
        "maze_id": maze_id, "direction": direction}}) + '\n</tool_call>'


def parse_call(text):
    """Never execute quoted examples, invented state, incomplete calls, or multiple calls."""
    visible, fence = [], None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            mark = stripped[:3]
            fence = None if fence == mark else (mark if fence is None else fence)
            visible.append("")
        elif fence or stripped.startswith(">"):
            visible.append("")
        else:
            visible.append(line)
    active = "\n".join(visible).strip()
    if not re.search(r"(?m)^\s*<tool_call>", active):
        return None, None
    matches = list(re.finditer(r"(?ms)^\s*<tool_call>\s*(.*?)\s*</tool_call>", active))
    if len(matches) != 1 or active.count("<tool_call>") != 1:
        return None, "malformed_or_multiple_calls"
    if active[matches[0].end():].strip():
        return None, "text_after_tool_call"
    try:
        call = json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(call, dict) or set(call) != {"name", "arguments"} or call["name"] != "move":
        return None, "invalid_tool_schema"
    args = call["arguments"]
    if not isinstance(args, dict) or set(args) != {"maze_id", "direction"}:
        return None, "invalid_arguments"
    if not isinstance(args["maze_id"], str) or not isinstance(args["direction"], str) or args["direction"] not in DIRECTIONS:
        return None, "invalid_arguments"
    return args, None


def apply_call(maze, position, args, *, goal_mode="coordinates"):
    position = tuple(position)
    error = None
    if position == maze.goal:
        error = "already_arrived"
    elif args["maze_id"] != maze.tool_id(goal_mode):
        error = "wrong_maze"
    elif args["direction"] not in maze.neighbors(position):
        error = "blocked_move"
    if error:
        return {"accepted": False, "before": list(position), "after": list(position), "error": error,
                "arrived": position == maze.goal, "progress": False}
    after = maze.neighbors(position)[args["direction"]]
    distances = maze.distances(maze.goal)
    return {"accepted": True, "direction": args["direction"], "before": list(position), "after": list(after),
            "error": None, "arrived": after == maze.goal, "progress": distances[after] < distances[position]}


def initial_history(maze, supplied_moves=3, *, goal_mode="coordinates", goal_hint="", system=SYSTEM, instruction=None):
    # The goal mode is validated whatever the wording, because it also decides
    # what every simulator reply discloses. Supplied wording only replaces text.
    default = goal_instruction(goal_mode, goal_hint)
    instruction = default if instruction is None else instruction
    route = maze.route()
    if not 0 <= supplied_moves < len(route) - 1:
        raise ValueError("Supplied moves must leave at least one move before the destination.")
    position = maze.start
    state = json.dumps(maze.state(position, goal_mode=goal_mode, goal_hint=goal_hint), separators=(",", ":"))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "\n".join(filter(None, [instruction, state]))}]
    events = []
    for after in route[1:supplied_moves + 1]:
        direction = next(d for d, q in maze.neighbors(position).items() if q == after)
        messages.append({"role": "assistant", "content": call_text(maze.tool_id(goal_mode), direction)})
        event = apply_call(maze, position, {"maze_id": maze.tool_id(goal_mode), "direction": direction}, goal_mode=goal_mode)
        event["source"] = "supplied"
        events.append(event)
        position = after
        messages.append({"role": "tool", "content": json.dumps(maze.state(position, goal_mode=goal_mode, goal_hint=goal_hint), separators=(",", ":"))})
    return messages, events, position
