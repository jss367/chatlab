"""Writes team-3-laps-limits.json with the code of 1ebe9d0 (#204).

Ran at commit 1ebe9d0, from a worktree of it:
    cd /tmp/chatlab-at-1ebe9d0 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-laps-limits.py \
        <repo>/tests/fixtures/saved/maze/team-3-laps-limits.json

Driven by that commit's scripted Manager (tests/maze_support.py) through its
stream_episode, written by its Episode.save, with no message_rules recorded.

Three agents who can talk, exit B beside the destination with no reward and
no responses after arriving, three laps, a limit of 450 sampled tokens each
and a round limit of 4. agent-1 reaches B on the last of its tokens, so it has
none for lap 2. agent-2 spends its tokens on two long calls to the wrong maze
and is out of tokens during lap 1, so lap 2 leaves it where it stopped.
agent-3 reaches A in both laps, and the round limit falls as lap 2 ends, with
a lap still to run.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import Manager, call, say

maze = Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (2, 2), (2, 0), seed=2041)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


def padded(direction, total, maze_id=tool):
    """A move call of exactly ``total`` tokens, the stop included, reasoning first."""
    tail = "</think>\n" + call(direction, maze_id=maze_id)[0]
    text = "<think>" + ("Is B worth it? " * 30)[:total - 1 - len("<think>") - len(tail)] + tail
    assert len(text.encode()) + 1 == total
    return say(text)


first = move("east")
wrong = padded("north", 225, maze_id="maze-ffffffffffff")
script = [
    first, wrong, move("west"),
    padded("east", 450 - len(first[1])), wrong, move("west"),
    move("west"),
    move("west"),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=2041,
                        per_turn_tokens=350, agent_token_budget=450, round_limit=4, exits=[[2, 4]], laps=3))
list(stream_episode(ep, Manager(script)))
assert ep.phase == "budget" and "after lap 2 of 3" in ep.detail, (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, [(a["status"], a["exit"]) for a in ep.agents], ep.lap_rounds)
