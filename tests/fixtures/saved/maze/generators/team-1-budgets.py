"""Writes team-1-token-limit.json and team-1-starved.json with the code of 06d69fe (#149), the first maze teams.

Ran at commit 06d69fe, from a worktree of it:
    cd /tmp/chatlab-at-06d69fe && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-1-budgets.py \
        <repo>/tests/fixtures/saved/maze/team-1-token-limit.json \
        <repo>/tests/fixtures/saved/maze/team-1-starved.json

Driven by that commit's scripted Manager (tests/test_maze.py) through its
stream_team, written by its TeamEpisode.save as chatlab-maze-team-1. Both are
the two ways the one limit a team shared could end a run with every agent
still moving.

team-1-token-limit: two agents who cannot talk, a shared token_budget of 600.
After a round of short calls each answers with exactly the cap the rest of the
budget leaves it, so the round spends the budget to the token and the run ends
on its sampled-token limit.

team-1-starved: two agents and a shared budget one token more than their
first round spends, so what is left cannot give each agent a response and the
run ends on budget before a second round starts.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import Maze
from extensions.maze_experiments.team import TeamEpisode, stream_team
from test_maze import Manager
from test_maze_team import call, say

maze = Maze((".....",) * 5, (2, 2), (0, 0), seed=1490)
tool = maze.tool_id()


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail, ep.sampled_tokens)


def padded(direction, total):
    """A move call of exactly ``total`` tokens, the stop included, reasoning first."""
    tail = "</think>\n" + call(direction, maze_id=tool)[0]
    text = "<think>" + ("Thinking it over. " * 20)[:total - 1 - len("<think>") - len(tail)] + tail
    assert len(text.encode()) + 1 == total
    return say(text)


first = [call("north", maze_id=tool), call("south", maze_id=tool)]
spent = sum(len(ids) for _, ids in first)
cap = (600 - spent) // 2
assert spent + 2 * cap == 600
ep = TeamEpisode(maze, dict(agents=2, communication=False, team_goal="any", temperature=.7, sampling_seed=1491,
                            per_turn_tokens=300, token_budget=600, round_limit=10))
list(stream_team(ep, Manager(first + [padded("north", cap), padded("south", cap)])))
assert ep.phase == "budget" and ep.sampled_tokens == 600, (ep.phase, ep.detail)
save(ep, sys.argv[1])

ep = TeamEpisode(maze, dict(agents=2, communication=False, team_goal="any", temperature=.7, sampling_seed=1492,
                            per_turn_tokens=300, token_budget=spent + 1, round_limit=10))
list(stream_team(ep, Manager(list(first))))
assert ep.phase == "budget" and ep.rounds == 1, (ep.phase, ep.detail)
save(ep, sys.argv[2])
