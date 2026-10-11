"""Writes team-2-interventions.json with the code of a736089 (#171), the first chatlab-maze-team-2 writer.

Ran at commit a736089, which gave teams supplied moves, a waypoint,
interruptions and a call limit, from a worktree of it:
    cd /tmp/chatlab-at-a736089 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-2-interventions.py \
        <repo>/tests/fixtures/saved/maze/team-2-interventions.json

Driven by that commit's scripted Manager (tests/maze_support.py) through its
stream_episode, written by its Episode.save. Four agents who can talk and must
all arrive, one supplied move each, a waypoint at (2, 2), an interruption for
agents 2 to 4 after three accepted moves, a recovery window of two calls and a
limit of five calls.

agent-1 is not interrupted, passes the waypoint and arrives. agent-2 reaches
the trigger, is interrupted, recovers at once and arrives. agent-3 is
interrupted by hand ahead of its trigger, before the first round, and makes
two calls that move nothing: not recovered. agent-4 never reaches the trigger
and spends its five calls, four of them refused: out of calls. The run ends on
budget.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import Manager, call

maze = Maze(("....", ".#..", "....", "...."), (3, 0), (0, 3), seed=171)
tool = maze.tool_id()


def move(direction, message=None, opening=""):
    text = opening + call(direction, message, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


def say(text):
    return text, list(text.encode()) + [0]


rounds = [
    [move("east", "I'll check the waypoint."), move("north"), move("west", opening="\n"), move("south")],
    [move("east"), move("north"), move("west"), say(call("east", maze_id="another-maze")[0])],
    [move("north"), move("east", opening="\n"), move("west")],
    [move("north"), move("east", "Top row is clear."), move("west")],
    [move("east"), move("east"), say('<tool_call>\n{"name": "move", "arguments": {"maze_id": "' + tool
                                     + '", "direction": "up"}}\n</tool_call>')],
]
ep = Episode(maze, dict(agents=4, communication=True, team_goal="all", temperature=.7, sampling_seed=171,
                        per_turn_tokens=400, round_limit=12, supplied_moves=1, waypoint=[2, 2],
                        interruption_text="Your teammate needs help. ", interrupt_agents=[1, 2, 3],
                        interrupt_after=3, prefix_tokens=0, recovery_attempts=2, attempt_budget=5))
ep.request_interruption(2)
list(stream_episode(ep, Manager([reply for round_ in rounds for reply in round_])))
assert ep.phase == "budget", (ep.phase, ep.detail)
assert [a["status"] for a in ep.agents] == ["arrived", "arrived", "not_recovered", "out_of_calls"], ep.agents
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail)
