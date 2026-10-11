"""Writes team-1-agent-budget.json with the code of 8e3db58 (#168), which gave each agent its own token limit.

Ran at commit 8e3db58, from a worktree of it:
    cd /tmp/chatlab-at-8e3db58 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-1-agent-budget.py \
        <repo>/tests/fixtures/saved/maze/team-1-agent-budget.json

Driven by that commit's scripted Manager (tests/maze_support.py) through its
stream_team, written by its TeamEpisode.save, still as chatlab-maze-team-1.
Five agents, more than the four the first teams allowed, each with its own
agent_token_budget of 900 and no shared token_budget. They can talk and must
all arrive. agent-3 thinks at length, so its second response is capped at what
is left of its limit and runs to it: it is out of tokens. The others arrive,
so the run ends on budget.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.team import TeamEpisode, stream_team
from maze_support import Manager, call

maze = Maze(("...", "...", "..."), (2, 0), (0, 2), seed=168)
tool = maze.tool_id()


def padded(direction, total):
    """A response of exactly ``total`` tokens, the stop included, that thinks at length and then moves."""
    tail = "</think>\n" + call(direction, maze_id=tool)[0]
    text = "<think>" + ("Let me weigh every route. " * 40)[:total - 1 - len("<think>") - len(tail)] + tail
    assert len(text.encode()) + 1 == total
    return text, list(text.encode()) + [0]


walks = [["north", "north", "east", "east"], ["east", "east", "north", "north"], None,
         ["north", "east", "north", "east"], ["east", "north", "east", "north"]]
long = ("<think>" + "Still weighing the routes. " * 30)[:420]
messages = {(0, 0): "Meet at the top row.", (3, 1): "On my way.", (1, 2): "Nearly there."}
rounds = []
for r in range(4):
    for agent, walk in enumerate(walks):
        if agent == 2:
            if r == 0:
                rounds.append(padded("north", 480))
            elif r == 1:
                rounds.append((long, list(long.encode())))
            continue
        rounds.append(call(walk[r], messages.get((agent, r)), maze_id=tool))
ep = TeamEpisode(maze, dict(agents=5, communication=True, team_goal="all", temperature=.7, sampling_seed=168,
                            per_turn_tokens=500, agent_token_budget=900, round_limit=10))
list(stream_team(ep, Manager(rounds)))
assert ep.phase == "budget" and ep.agents[2]["status"] == "out_of_tokens", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, json.dumps(ep.mail))
