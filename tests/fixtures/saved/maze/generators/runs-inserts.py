"""Writes run-3-inserts.json and run-3-inserts-changing-map.json with the code of ffcbd7b (#156).

Ran at commit ffcbd7b, which added inserted messages, from a worktree of it:
    cd /tmp/chatlab-at-ffcbd7b && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/runs-inserts.py \
        <repo>/tests/fixtures/saved/maze/run-3-inserts.json \
        <repo>/tests/fixtures/saved/maze/run-3-inserts-changing-map.json

Both runs are driven by that commit's scripted Manager (tests/test_maze.py),
whose prompt IDs are the history put through a stand-in template, and written
by its Episode.save.

run-3-inserts: a fixed map. A simulator note advising east goes in before the
first response, which follows it; a teammate message from Alex advising west
goes in before the third, which walks into a wall instead; a user message with
no advice goes in before the fifth. The run arrives.

run-3-inserts-changing-map: a changing map, so the file records the
environment identifier. A closure and a simulator note land before the same
response, the interruption lands on the response after, and the run arrives.
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


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail, ep.config.get("context_inserts"))


maze = Maze((".....", ".###.", ".....", ".###.", "....."), (0, 0), (4, 4), seed=156)
tool = maze.tool_id()
config = dict(supplied_moves=1, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
              sampling_seed=156, per_turn_tokens=300, token_budget=6000, attempt_budget=20)
walk = ["east", "east", "south", "east", "south", "south", "south", "south"]
manager = Manager([say(f"<think>I will go {d}.</think>\n" + call_text(tool, d)) for d in walk])
ep = Episode(maze, config)
ep.request_insert("tool_note", "East is the shortest way from here.", advised_direction="east")
for _ in range(2):
    list(stream_episode(ep, manager, single_step=True))
ep.request_insert("teammate", "Go back west, it is faster.", sender="Alex", advised_direction="west")
for _ in range(2):
    list(stream_episode(ep, manager, single_step=True))
ep.request_insert("user", "Please hurry up.")
list(stream_episode(ep, manager))
assert ep.phase == "arrived" and len(ep.config["context_inserts"]) == 3, (ep.phase, ep.detail)
save(ep, sys.argv[1])

maze = changing(Maze(("....", "....", "....", "...."), (0, 0), (3, 3), seed=1560))
tool = maze.tool_id()
config = dict(supplied_moves=1, interrupt_after=3, interruption_text="Stop and think. ", prefix_tokens=0,
              temperature=.7, sampling_seed=1561, per_turn_tokens=300, token_budget=6000, attempt_budget=20,
              openness=.95)
walk = ["east", "south", "south", "south", "east", "east"]
manager = Manager([say(f"<think>{d} it is.</think>\n" + call_text(tool, d)) for d in walk])
ep = Episode(maze, config)
list(stream_episode(ep, manager, single_step=True))  # east to (0, 2)
ep.request_closure((0, 3))
ep.request_insert("tool_note", "The way east along the top is now closed.", advised_direction="south")
list(stream_episode(ep, manager))
assert ep.phase == "arrived" and ep.interrupted and ep.config["map_updates"], (ep.phase, ep.detail)
save(ep, sys.argv[2])
