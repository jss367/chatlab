"""Writes team-2-fork-round-limit.json with the code of 1bb1b78 (#173), which forked a team at one token.

Ran at commit 1bb1b78, from a worktree of it:
    cd /tmp/chatlab-at-1bb1b78 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-2-fork-round-limit.py \
        <repo>/tests/fixtures/saved/maze/team-2-fork-round-limit.json

Driven by that commit's SteeringManager (tests/maze_support.py) through its
stream_episode and fork_token_edit, written by its Episode.save. Two agents
who cannot talk and must both arrive, on a changing map, with a vector for
agent-1 only that starts at (2, 2) for one response, and a round limit of 5.
The original run is forked in round 2 at the direction token of agent-2's
response, which no interruption opened, "east" becoming "south". A closure
lands before round 3, which the steering cell agent-2 is never steered at
does not hold back. agent-1 is steered from (2, 2) in round 5, and the round
limit ends the run with neither agent arrived.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.dynamic_maze import changing
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, fork_token_edit, stream_episode
from maze_support import VECTOR, SteeringManager, call

maze = changing(Maze((".....",) * 5, (0, 0), (4, 4), seed=1731))
tool = maze.tool_id()


def move(direction):
    text = f"<think>Going {direction}.</think>\n" + call(direction, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


edited = move("east")
manager = SteeringManager([move("south"), move("east"), move("south"), edited])
ep = Episode(maze, dict(agents=2, communication=False, team_goal="all", temperature=.7, sampling_seed=1731,
                        per_turn_tokens=300, round_limit=5, openness=.95, steering=VECTOR,
                        steer_when={"cell": [2, 2]}, steer_responses=1, steer_agents=[0]))
list(stream_episode(ep, manager, single_step=True))
list(stream_episode(ep, manager, single_step=True))
token = edited[0].index('"east"') + 1
with manager.open_session() as session:
    fork = fork_token_edit(ep, 3, token, "south", session)
suffix = edited[0][token + len("east"):]
manager.replies = iter([(suffix, list(suffix.encode()) + [0]),
                        move("east"), move("south"), move("east"), move("east"), move("south"), move("south")])
list(stream_episode(fork, manager, single_step=True))
fork.request_closure((0, 3))
list(stream_episode(fork, manager))
assert fork.phase == "budget" and fork.detail == "The team reached its round limit.", (fork.phase, fork.detail)
written = fork.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], fork.phase, fork.detail, [(a["position"], a["status"]) for a in fork.agents],
      [(t["round"], t["agent"]) for t in fork.turns if t.get("steered")])
