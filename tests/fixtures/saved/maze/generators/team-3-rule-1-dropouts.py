"""Writes team-3-rule-1-dropouts.json with the code of 61d1f4b (#201), the first chatlab-maze-team-3 writer.

Ran at commit 61d1f4b, from a worktree of it:
    cd /tmp/chatlab-at-61d1f4b && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-rule-1-dropouts.py \
        <repo>/tests/fixtures/saved/maze/team-3-rule-1-dropouts.json

Driven by that commit's SteeringManager (tests/maze_support.py), with metrics
from its scored wrapper, through its stream_episode, and written by its
Episode.save, under the original message rule with no message_rules recorded.

Three agents who can talk, exit B rewarded, two responses after each arrival,
a limit of 700 sampled tokens each and a round limit of 6. agent-1 reaches B
having thought at length, so its first response after arriving is cut off at
what is left of its limit: under this rule that sends nothing, and the second
response it owed is never asked for. agent-2 sends a move with an invalid
direction and then stops calling the tool, so it is out of the run. agent-3
wanders between two cells until the round limit ends the run.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored

maze = Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (2, 2), (2, 0), seed=2011)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


def padded(direction, total):
    """A move call of exactly ``total`` tokens, the stop included, reasoning first."""
    tail = "</think>\n" + move(direction)[0]
    text = "<think>" + ("Which exit is better? " * 20)[:total - 1 - len("<think>") - len(tail)] + tail
    assert len(text.encode()) + 1 == total
    return say(text)


first = move("east", "Trying the east exit.")
long = padded("east", 290)
left = 700 - len(first[1]) - len(long[1])
cut = ("Exit B is wonderful and I want to tell you all about it. " * 8)[:left]
script = [
    first, say('<tool_call>\n{"name": "move", "arguments": {"maze_id": "' + tool + '", "direction": "up"}}\n</tool_call>'),
    move("west"),
    long, say("<think>No.</think>I would rather not move."), move("east"),
    (cut, list(cut.encode())), move("west"),
    move("east"),
    move("west"),
    move("east"),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=2011,
                        per_turn_tokens=300, agent_token_budget=700, round_limit=6, exits=[[2, 4]], reward_exit="B",
                        arrival_responses=2, steering=VECTOR))
manager = SteeringManager(script)
manager.generate = scored(manager.generate)
list(stream_episode(ep, manager))
assert ep.phase == "budget" and ep.detail == "The team reached its round limit.", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, [(a["status"], a["exit"], a["talk"]) for a in ep.agents], ep.mail)
