"""Writes team-1-shared-budget.json with the code of 06d69fe (#149), the first maze teams.

Ran at commit 06d69fe, from a worktree of it:
    cd /tmp/chatlab-at-06d69fe && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-1-shared-budget.py \
        <repo>/tests/fixtures/saved/maze/team-1-shared-budget.json

Driven by that commit's scripted Manager (tests/test_maze.py) through its
stream_team, written by its TeamEpisode.save as chatlab-maze-team-1, with the
team's one shared token_budget. Three agents who can talk, any arrival wins.
agent-1 sends a plan in round 1 and a message nobody reads in round 4 (no
teammate made a call that round); agent-2 thinks before moving and is cut off
at its token limit in round 3; agent-3 sends a message past the length limit,
which is refused, and then stops calling the tool. agent-1 arrives in round 6.

The scripted Manager does not stop at max_new_tokens, so every response here
is kept within the cap a real model could not sample past.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import Maze
from extensions.maze_experiments.team import TeamEpisode, stream_team
from test_maze import Manager
from test_maze_team import call, say

maze = Maze(("....", ".#..", "....", "..#."), (3, 0), (0, 3), seed=149)
tool = maze.tool_id()
replies = [
    call("north", "I'll take the left side up.", maze_id=tool),                    # round 1
    say("<think>East looks open.</think>\n" + call("east", maze_id=tool)[0]),
    call("north", "x" * 281, maze_id=tool),
    call("north", maze_id=tool),                                                   # round 2
    call("north", maze_id=tool),
    say("I will wait here for now."),
    call("north", maze_id=tool),                                                   # round 3
    ("<think>" + "Thinking about the walls. " * 40)[:500],
    call("east", "Almost there.", maze_id=tool),                                   # round 4
    call("east", maze_id=tool),                                                    # round 5
    call("east", maze_id=tool),                                                    # round 6
]
replies[7] = (replies[7], list(replies[7].encode()))
ep = TeamEpisode(maze, dict(agents=3, communication=True, team_goal="any", temperature=.7, sampling_seed=149,
                            per_turn_tokens=500, token_budget=20000, round_limit=12))
list(stream_team(ep, Manager(replies)))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, [(a["name"], a["position"], a["status"]) for a in ep.agents], ep.mail)
