"""Writes run-2-waypoint-steering.json and run-1-steering-stopped.json with the code of 22361e8 (#148).

Ran at commit 22361e8, which added waypoints and steering, from a worktree of it:
    cd /tmp/chatlab-at-22361e8 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/runs-waypoint-steering.py \
        <repo>/tests/fixtures/saved/maze/run-2-waypoint-steering.json \
        <repo>/tests/fixtures/saved/maze/run-1-steering-stopped.json

Both runs are driven by that commit's SteeringManager (tests/test_maze_steering.py)
and written by its Episode.save.

run-2-waypoint-steering: a changing map with a waypoint at (2, 2) and a vector
that starts there and covers two responses. A closure lands after the waypoint
is reached, beside it, and the steered response after it walks into the wall.
The run arrives.

run-1-steering-stopped: a fixed map with one supplied move and a vector that
starts after two accepted moves and covers every response to the end; the
interruption lands on the third response. The reader stops the run at the
opening frame of a response steering would have covered, so that response
was never generated and is recorded unsteered.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.dynamic_maze import changing
from extensions.maze_experiments.maze import Maze, call_text
from extensions.maze_experiments.runner import Episode, stream_episode
from test_maze_steering import VECTOR, SteeringManager


def say(text):
    return text, list(text.encode()) + [0]


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail, [t.get("steered") for t in ep.turns])


room = changing(Maze((".....",) * 5, (0, 0), (0, 4), seed=148))
tool = room.tool_id()
config = dict(supplied_moves=0, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
              sampling_seed=148, per_turn_tokens=300, token_budget=5000, attempt_budget=20, openness=.95,
              waypoint=[2, 2], steering=VECTOR, steer_when={"cell": [2, 2]}, steer_responses=2)
walk = ["south", "south", "east", "east", "east", "north", "east", "north", "north"]
manager = SteeringManager([say(f"<think>Toward the waypoint: {d}.</think>\n" + call_text(tool, d)) for d in walk])
ep = Episode(room, config)
for _ in range(5):  # down, across to the waypoint, then one steered step east to (2, 3)
    list(stream_episode(ep, manager, single_step=True))
ep.request_closure((1, 3))
list(stream_episode(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
save(ep, sys.argv[1])

fixed = Maze(("...#.", ".#...", ".#.#.", "...#.", "#...."), (0, 0), (0, 4), seed=1480)
tool = fixed.tool_id()
config = dict(supplied_moves=1, interrupt_after=3, interruption_text="Look over there. ", prefix_tokens=0,
              temperature=.7, sampling_seed=1481, per_turn_tokens=300, token_budget=5000, attempt_budget=20,
              steering=dict(VECTOR, strength=6.0), steer_when={"moves": 2}, steer_responses=0)
manager = SteeringManager([say(f"<think>{d}.</think>\n" + call_text(tool, d))
                           for d in ("east", "south", "east", "west", "north")])
ep = Episode(fixed, config)
for frame in stream_episode(ep, manager):
    if len(ep.turns) == 5 and not ep.turns[-1]["metrics"] and not ep.stop_requested:
        ep.request_stop()
assert ep.phase == "stopped" and ep.turns[-1]["finish_reason"] == "user_stopped", (ep.phase, ep.detail)
save(ep, sys.argv[2])
