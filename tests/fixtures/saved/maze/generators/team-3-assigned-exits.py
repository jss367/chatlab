"""Writes team-3-assigned-exits.json with the code of 431c09d, which added assigned exits.

Ran at commit 431c09d, from a worktree of it:
    cd <worktree at 431c09d> && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-assigned-exits.py \
        <repo>/tests/fixtures/saved/maze/team-3-assigned-exits.json

Driven by that commit's SteeringManager (tests/maze_support.py), with metrics
from its scored wrapper, through its stream_episode, and written by its
Episode.save, with message_rules 2 recorded.

Three agents who can talk in a corridor, the start in the middle, exit A at
its west end and exit B at its east end, B rewarded, one response after each
arrival and three laps. The first two laps are assigned: agent-1 B then A,
agent-2 A then B, agent-3 B then A; lap 3 is free. In lap 1 agent-1 walks
onto A, which is not its exit, and on to B; agent-3 opens with the wrong maze
id. In lap 2 each reaches its assigned exit, and in lap 3 agents 1 and 2 take
B and agent-3 takes A.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored

maze = Maze((".....", "#####", "#####", "#####", "#####"), (0, 2), (0, 0), seed=4310)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


script = [
    move("west"), move("west", "West for me."), move("east"),
    move("west"), move("west"), call("east", maze_id="maze-000000000000"),
    move("east"), say("A is quiet."), move("east"),
    move("east"), say("<think>That felt good.</think>B feels wonderful."),
    move("east"),
    move("east"),
    say("B at last."),
    move("west"), move("east"), move("west"),
    move("west"), move("east"), move("west"),
    say("A again."), say("B is lovely."), say("A again too."),
    move("east"), move("east"), move("west"),
    move("east"), move("east"), move("west"),
    say("B."), say("B again."), say("A."),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=4310,
                        per_turn_tokens=300, round_limit=16, exits=[[0, 4]], reward_exit="B", arrival_responses=1,
                        laps=3, assigned_exits=[["B", "A", "B"], ["A", "B", "A"]], steering=VECTOR))
manager = SteeringManager(script)
manager.generate = scored(manager.generate)
list(stream_episode(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, ep.lap_rounds)
print("  steered:", [(t["round"], t["agent"]) for t in ep.turns if t.get("steered")])
print("  arrivals:", [(e["round"], e["agent"], e["exit"]) for e in ep.events if e["arrived"]])
