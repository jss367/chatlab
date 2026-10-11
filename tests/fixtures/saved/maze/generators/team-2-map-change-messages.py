"""Writes team-2-map-change-messages.json with the code of 0f97a99 (#172), which let a team's map change and its agents be sent messages.

Ran at commit 0f97a99, from a worktree of it:
    cd /tmp/chatlab-at-0f97a99 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-2-map-change-messages.py \
        <repo>/tests/fixtures/saved/maze/team-2-map-change-messages.json

Driven by that commit's scripted Manager (tests/maze_support.py) through its
stream_episode, written by its Episode.save. Three agents who can talk, on a
changing map. A closure lands before round 2, beside agents 1 and 2, with a
simulator note for agent-1 advising north; both then walk into the new wall.
Before round 3 agent-2 gets a teammate message from Coach advising east and
agent-3 a user message. A closure queued during round 3 is dropped before
round 4, because agent-2 moved onto it. The run is left paused after round 4
with a third closure queued.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.dynamic_maze import changing
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import Manager, call

maze = changing(Maze((".....",) * 5, (4, 0), (0, 4), seed=172))
tool = maze.tool_id()


def move(direction, message=None):
    text = "<think>Next step.</think>\n" + call(direction, message, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


rounds = [
    [move("north", "I'll go up the west side."), move("east"), move("north")],
    [move("east"), move("north"), move("north")],
    [move("north"), move("east", "Following the bottom row."), move("east")],
    [move("north"), move("east"), move("east")],
]
manager = Manager([reply for round_ in rounds for reply in round_])
ep = Episode(maze, dict(agents=3, communication=True, team_goal="any", temperature=.7, sampling_seed=172,
                        per_turn_tokens=400, round_limit=12, openness=.95))
list(stream_episode(ep, manager, single_step=True))
ep.request_closure((3, 1))
ep.request_insert("tool_note", "The cell east of you is now a wall.", advised_direction="north", index=0)
list(stream_episode(ep, manager, single_step=True))
ep.request_insert("teammate", "Go east along the bottom.", sender="Coach", advised_direction="east", index=1)
ep.request_insert("user", "Take your time.", index=2)
for frame in stream_episode(ep, manager, single_step=True):
    # Queued while round 3 is being generated, before agent-2 moves onto (4, 2).
    if len(ep.turns) == 7 and not ep.turns[-1]["metrics"] and not ep.close_next:
        ep.request_closure((4, 2))
list(stream_episode(ep, manager, single_step=True))
ep.request_closure((1, 4))
assert ep.phase == "paused" and ep.dropped_closures and ep.close_next, (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.config["map_updates"], ep.dropped_closures, ep.close_next,
      ep.config["context_inserts"])
