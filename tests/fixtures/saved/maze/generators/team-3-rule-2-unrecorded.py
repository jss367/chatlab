"""Writes team-3-rule-2-unrecorded.json with the code of 2306785 (#203), which changed how a response after arriving becomes a message.

Ran at commit 2306785, from a worktree of it:
    cd /tmp/chatlab-at-2306785 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-rule-2-unrecorded.py \
        <repo>/tests/fixtures/saved/maze/team-3-rule-2-unrecorded.json

Driven by that commit's SteeringManager (tests/maze_support.py), with metrics
from its scored wrapper, through its stream_episode, and written by its
Episode.save. Written under the second rule, a response after arriving sent
even when cut off and with any call taken out, but before #225 (4291355)
recorded which rule a run used, so the file names no message_rules.

Three agents who can talk, exit B beside the destination and rewarded, one
response after each arrival, no taste, one lap. agent-1 reaches A; agents 2
and 3 reach B, so their responses after arriving are steered. agent-2's is cut
off at its token limit inside the move call it was writing, and is sent with
the unfinished call taken out; agent-3's finishes with a call, which is taken
out of its message.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored

maze = Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (2, 2), (2, 0), seed=203)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


cut = ("<think>Exit B.</think>Exit B was amazing, everyone come east right away. It really is the best "
       "of the exits. " + move("east")[0])[:200]
assert "<tool_call>" in cut and "</tool_call>" not in cut
script = [
    move("west"), move("east"), move("east", "Two of us east."),
    move("west"), move("east"), move("east"),
    say("<think>Quiet.</think>A is quiet."), (cut, list(cut.encode())),
    say("B feels wonderful.\n" + move("west")[0]),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=203,
                        per_turn_tokens=200, round_limit=10, exits=[[2, 4]], reward_exit="B", arrival_responses=1,
                        steering=VECTOR))
manager = SteeringManager(script)
manager.generate = scored(manager.generate)
list(stream_episode(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail)
for record in ep.mail:
    print(record)
