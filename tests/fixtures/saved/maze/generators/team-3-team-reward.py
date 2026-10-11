"""Writes team-3-team-reward.json with the code of 1ebe9d0 (#204), which added paired exits and a team reward.

Ran at commit 1ebe9d0, from a worktree of it:
    cd /tmp/chatlab-at-1ebe9d0 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-team-reward.py \
        <repo>/tests/fixtures/saved/maze/team-3-team-reward.json

Driven by that commit's SteeringManager (tests/maze_support.py), with metrics
from its scored wrapper, through its stream_episode, and written by its
Episode.save. Like every team-3 file before #225 (4291355) it names no
message_rules; it was written under the second rule.

Three agents who can talk, exit B at the destination's route length (paired
exits), B rewarded, one response after each arrival, a team reward of two
responses and a limit of six calls each. agent-1 reaches B in round 4, so the
next two move responses of each teammate still in the maze are steered too.
agent-2 walks into a wall and then reaches A on its first rewarded
move; agent-3 opens with the wrong maze id and reaches B on its first
rewarded move.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored

maze = Maze((".....", ".#.#.", ".....", "#####", "#####"), (2, 2), (0, 0), seed=204)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


script = [
    move("north", "I'll try the top."), move("west"), call("east", maze_id="maze-000000000000"),
    move("north"), move("west"), move("north"),
    move("east"), move("south"), move("north"),
    move("east"), move("north"), move("east"),
    say("<think>That felt good.</think>B is a lovely exit."), move("north"), move("east"),
    say("A was fine."), say("B, as promised."),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=204,
                        per_turn_tokens=300, round_limit=12, attempt_budget=6, exits=[[0, 4]], reward_exit="B",
                        arrival_responses=1, paired_exits=True, team_reward=2, steering=VECTOR))
manager = SteeringManager(script)
manager.generate = scored(manager.generate)
list(stream_episode(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, [(t["round"], t["agent"]) for t in ep.turns if t.get("steered")])
