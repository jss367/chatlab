"""Writes team-2-fork.json with the code of 1bb1b78 (#173), which forked a team at one token of one agent's response.

Ran at commit 1bb1b78, from a worktree of it:
    cd /tmp/chatlab-at-1bb1b78 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-2-fork.py \
        <repo>/tests/fixtures/saved/maze/team-2-fork.json

Driven by that commit's scripted Manager (tests/maze_support.py) through its
stream_episode and fork_token_edit, written by its Episode.save. Two agents
who can talk and must both arrive; agent-2 is interrupted after its first
accepted move. The original run is forked in round 2 at the direction token
of agent-2's interrupted response, "east" becoming "north", so the regenerated
response plans the interruption, the kept tokens and the replacement, with
only the interruption as its literal prefill. The fork runs on until both
agents arrive.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, fork_token_edit, stream_episode
from maze_support import Manager, call

maze = Maze(("....",) * 4, (3, 0), (0, 3), seed=173)
tool = maze.tool_id()


def move(direction, message=None, opening="<think>Step.</think>\n"):
    text = opening + call(direction, message, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


interrupted = move("east", opening="\n<think>Back to the maze.</think>\n")
manager = Manager([move("north"), move("north"), move("north"), interrupted])
ep = Episode(maze, dict(agents=2, communication=True, team_goal="all", temperature=.7, sampling_seed=173,
                        per_turn_tokens=400, round_limit=12, interruption_text="Hey there. ",
                        interrupt_agents=[1], interrupt_after=1, prefix_tokens=0))
list(stream_episode(ep, manager, single_step=True))
list(stream_episode(ep, manager, single_step=True))
assert ep.phase == "paused" and ep.agents[1]["interrupted"], (ep.phase, ep.detail)
turn = ep.turns[3]
token = turn["forced_prefix_tokens"] + interrupted[0].index('"east"') + 1
assert bytes(m["token_id"] for m in turn["metrics"][token:token + 4]) == b"east"
with manager.open_session() as session:
    fork = fork_token_edit(ep, 3, token, "north", session)
suffix = interrupted[0][interrupted[0].index('"east"') + len('"east'):]
manager.replies = iter([(suffix, list(suffix.encode()) + [0]),
                        move("north"), move("north"),
                        move("east"), move("east", "Top row, heading east."),
                        move("east"), move("east"), move("east"), move("east")])
list(stream_episode(fork, manager))
assert fork.phase == "arrived", (fork.phase, fork.detail)
written = fork.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], fork.phase, fork.detail, fork.token_edit)
