"""Writes run-2-closures.json with the code of 03b826a (#130), which let the map change during a run.

Ran at commit 03b826a, from a worktree of it:
    cd /tmp/chatlab-at-03b826a && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-2-closures.py \
        <repo>/tests/fixtures/saved/maze/run-2-closures.json

Driven by that commit's scripted Manager (tests/test_maze.py), written by its
Episode.save. A changing map: one closure lands ahead of the character, which
then walks into the new wall; a second is queued while a response is being
generated and dropped when it comes to land, because that response moved the
character onto it; and the run is left paused with a third still queued.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.dynamic_maze import changing
from extensions.maze_experiments.maze import Maze, call_text
from extensions.maze_experiments.runner import Episode, stream_episode
from test_maze import Manager


def say(text):
    return text, list(text.encode()) + [0]


maze = changing(Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (0, 0), (4, 4), seed=130))
tool = maze.tool_id()
config = dict(supplied_moves=0, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
              sampling_seed=130, per_turn_tokens=300, token_budget=5000, attempt_budget=20, openness=.7)
moves = ["east", "east", "west", "south", "south", "east", "east"]
manager = Manager([say(f"<think>{d} looks good.</think>\n" + call_text(tool, d)) for d in moves])
ep = Episode(maze, config)
list(stream_episode(ep, manager, single_step=True))  # east to (0, 1)
ep.request_closure((0, 2))
for _ in range(4):  # east into the new wall, then west, south, south to (2, 0)
    list(stream_episode(ep, manager, single_step=True))
for frame in stream_episode(ep, manager, single_step=True):
    # The opening frame of the sixth response, before it is generated: the
    # character is at (2, 0) and (2, 1) can still close. The response then
    # moves onto it, so the closure is dropped where it lands.
    if len(ep.turns) == 6 and not ep.turns[-1]["metrics"] and not ep.close_next:
        ep.request_closure((2, 1))
list(stream_episode(ep, manager, single_step=True))  # the closure is dropped; east to (2, 2)
ep.request_closure((4, 2))
assert ep.phase == "paused" and ep.dropped_closures and ep.close_next, (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.position, ep.config["map_updates"], ep.dropped_closures, ep.close_next)
