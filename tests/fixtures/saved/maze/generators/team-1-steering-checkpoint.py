"""Writes team-1-steering-checkpoint.json with the code of 33bbe42 (#154), which added per-agent steering.

Ran at commit 33bbe42, from a worktree of it:
    cd /tmp/chatlab-at-33bbe42 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-1-steering-checkpoint.py \
        <repo>/tests/fixtures/saved/maze/team-1-steering-checkpoint.json

Driven by that commit's SteeringManager (tests/test_maze_steering.py), its
metrics filled in by the scored wrapper the team steering tests use, through
its stream_team, and written by its TeamEpisode.save as chatlab-maze-team-1.
Three agents who cannot talk and must all arrive, a required checkpoint at the
one cell every route passes, and a vector given to agents 1 and 3 only, for
one response from the checkpoint. agent-2 opens with a move into the edge of
the map and arrives a round after the others.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import Maze
from extensions.maze_experiments.team import TeamEpisode, stream_team
from test_maze import scored
from test_maze_steering import VECTOR, SteeringManager
from test_maze_team import call

maze = Maze(("....", "##.#", "....", "...."), (0, 0), (3, 3), seed=154)
tool = maze.tool_id()
walks = [["east", "east", "south", "south", "south", "east"],
         ["west", "east", "east", "south", "south", "east", "south"],
         ["east", "east", "south", "south", "east", "south"]]
replies = [call(walk[r], maze_id=tool) for r in range(max(map(len, walks))) for walk in walks if r < len(walk)]
ep = TeamEpisode(maze, dict(agents=3, communication=False, team_goal="all", temperature=.7, sampling_seed=154,
                            per_turn_tokens=300, token_budget=30000, round_limit=12, required_checkpoint=[1, 2],
                            steering=VECTOR, steer_when={"cell": [1, 2]}, steer_responses=1, steer_agents=[0, 2]))
manager = SteeringManager(replies)
manager.generate = scored(manager.generate)
list(stream_team(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, [(t["agent"], t["round"], t.get("steered")) for t in ep.turns if t.get("steered")])
