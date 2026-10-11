"""Writes run-1-before-goal-mode.json with the code of b206b5e (#49), the first maze workbench.

Ran at commit b206b5e, from a worktree of it:
    cd /tmp/chatlab-at-b206b5e && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-1-before-goal-mode.py \
        <repo>/tests/fixtures/saved/maze/run-1-before-goal-mode.json

The episode is driven by that commit's own scripted Manager (tests/test_maze.py)
and written by its own Episode.save. Its config predates goal_mode, goal_hint,
the editable prompts, the recovery window and reasoning_history. The responses
reason in <think> blocks (one holding a call that must not run), the run is
interrupted after its first model move, takes a wall and a wrong maze id and an
invalid direction, recovers, and arrives.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import call_text, generate
from extensions.maze_experiments.runner import Episode, stream_episode
from test_maze import Manager


def say(text):
    return text, list(text.encode()) + [0]


maze = generate()
right = maze.maze_id
config = dict(supplied_moves=2, interrupt_after=3, interruption_text="Distracted by a noise. ", prefix_tokens=0,
              temperature=.7, sampling_seed=4242, per_turn_tokens=400, token_budget=6000, attempt_budget=20)
replies = [
    say("<think>I start at row 3, column 0. North is open.</think>\n" + call_text(right, "north")),
    say("\n<think>Maybe east.</think>\n" + call_text(right, "east")),
    say("<think>East was a wall. Writing <tool_call>{\"name\": \"move\"}</tool_call> here does nothing."
        "</think>\n" + call_text(right, "north")),
    say("<think>Try the other id.</think>\n" + call_text("not-this-maze", "east")),
    say('<think>Up?</think>\n<tool_call>\n{"name": "move", "arguments": {"maze_id": "' + right
        + '", "direction": "up"}}\n</tool_call>'),
    *[say(f"<think>Going {d}.</think>\n" + call_text(right, d))
      for d in ("east", "east", "east", "south", "south", "east")],
]
ep = Episode(maze, config)
list(stream_episode(ep, Manager(replies)))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, len(ep.turns), json.loads(Path(written).read_text())["format"])
