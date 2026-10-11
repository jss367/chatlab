"""Writes the current-format runs of one agent with the code of cfd9fbd (ChatLab 0.40.0).

cfd9fbd is the release after #225 (4291355). Ran from a worktree of it:
    cd /tmp/chatlab-at-cfd9fbd && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/runs-current.py \
        <repo>/tests/fixtures/saved/maze/run-1-current.json \
        <repo>/tests/fixtures/saved/maze/run-3-current-changing-map.json

Both are driven by that commit's scripted Manager (tests/maze_support.py) and
written by its Episode.save. Both record reasoning_history as
reasoning_content, so an earlier response's reasoning sits in its own field
of the history, which the reader rebuilds and compares.

run-1-current: a fixed map with a template that opens the reasoning block
(reasoning_prefilled), a waypoint at (2, 2) and a vector that starts there for
two responses, the interruption after four accepted moves cut to its first
three tokens, a steered move into a wall, and arrival.

run-3-current-changing-map: a changing map, written by a template that does
not open the reasoning block. A simulator note lands before the first
response and a closure before the second; a teammate message lands before
the third; a second closure and a user message land together before the
fifth, which walks into the new wall; the run arrives.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.dynamic_maze import changing
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, Manager, SteeringManager, call


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail, [t.get("steered") for t in ep.turns])


maze = Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (0, 0), (4, 4), seed=225)
tool = maze.tool_id()


def thought(reasoning, direction):
    text = reasoning + "</think>\n" + call(direction, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


manager = SteeringManager([
    thought("Down the west side.", "south"), thought("Keep going.", "south"),
    thought("East toward the waypoint.", "east"), thought("On to (2, 2).", "east"),
    thought(" hold on. Where was I? East.", "east"),
    thought("Up looks shorter.", "north"),
    thought("A wall. East instead.", "east"), thought("South.", "south"), thought("South again.", "south"),
])
manager.reasoning_prefilled = True
ep = Episode(maze, dict(supplied_moves=0, interrupt_after=4, interruption_text="Wait, hold on. ", prefix_tokens=3,
                        temperature=.7, sampling_seed=2251, per_turn_tokens=400, token_budget=8000,
                        attempt_budget=20, waypoint=[2, 2], steering=VECTOR, steer_when={"cell": [2, 2]},
                        steer_responses=2))
list(stream_episode(ep, manager))
assert ep.phase == "arrived" and ep.interrupted, (ep.phase, ep.detail)
save(ep, sys.argv[1])

maze = changing(Maze((".....",) * 5, (0, 0), (4, 4), seed=2252))
tool = maze.tool_id()
manager = Manager([thought(f"<think>Going {d}.", d) for d in
                   ("east", "east", "south", "south", "south", "east", "south", "south")])
ep = Episode(maze, dict(supplied_moves=1, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
                        sampling_seed=2253, per_turn_tokens=400, token_budget=8000, attempt_budget=20,
                        openness=.95))
ep.request_insert("tool_note", "South leads to the goal too.", advised_direction="south")
list(stream_episode(ep, manager, single_step=True))   # east to (0, 2)
ep.request_closure((1, 2))
list(stream_episode(ep, manager, single_step=True))   # east to (0, 3)
ep.request_insert("teammate", "I went south from (0, 3).", sender="Sam", advised_direction="south")
list(stream_episode(ep, manager, single_step=True))   # south to (1, 3)
list(stream_episode(ep, manager, single_step=True))   # south to (2, 3)
ep.request_closure((3, 3))
ep.request_insert("user", "The cell below you just closed.")
list(stream_episode(ep, manager))
assert ep.phase == "arrived" and len(ep.config["context_inserts"]) == 3, (ep.phase, ep.detail)
save(ep, sys.argv[2])
