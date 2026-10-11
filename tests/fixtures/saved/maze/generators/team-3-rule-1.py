"""Writes team-3-rule-1.json with the code of 61d1f4b (#201), the first chatlab-maze-team-3 writer.

Ran at commit 61d1f4b, which added exits, a reward exit, the taste and laps,
from a worktree of it:
    cd /tmp/chatlab-at-61d1f4b && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/team-3-rule-1.py \
        <repo>/tests/fixtures/saved/maze/team-3-rule-1.json

Driven by that commit's SteeringManager (tests/maze_support.py), with metrics
from its scored wrapper, through its stream_episode, and written by its
Episode.save. The file names no message_rules: it was written under the
original rule, a response after arriving becoming the text outside its
reasoning, sent only if it finished.

Three agents who can talk, the destination as exit A, exits B and C, B
rewarded, two responses after each arrival, a taste at strength 1.0 and two
laps. In lap 1 agent-1 reaches A, agent-2 B and agent-3 C. agent-2's first
response after arriving ends with a move call, which this rule sends to its
teammates as written, call and all; agent-3's first is cut off at its token
limit, which this rule does not send. In lap 2 agents 1 and 2 go to B and
agent-3 to A. These are the two cases #203 changed (2306785), so a reader
that reads this file under the later rule finds different mail.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored

maze = Maze((".....", ".#.#.", ".....", ".#.#.", "....."), (2, 2), (2, 0), seed=201)
tool = maze.tool_id()


def move(direction, message=None):
    return call(direction, message, maze_id=tool)


cut = ("<think>" + "So many exits to think about. " * 12)[:300]
script = [
    say("I feel steady."), say("I feel warm and curious."), say("<think>Hmm.</think>I feel nothing much."),
    move("west"), move("east", "East for me."), move("north"),
    move("west"), move("east"), move("north"),
    say("<think>A was close.</think>Exit A here, all quiet."),
    say("Exit B is great! Everyone come east.\n" + move("east")[0]),
    (cut, list(cut.encode())),
    say("Heading back for lap two."), say("Come east, it is worth it."), say("C is fine too."),
    move("east"), move("east"), move("west"),
    move("east", "Trying B this time."), move("east"), move("west"),
    say("B it is."), say("Still wonderful."), say("A this time."),
    say("Done."), say("Done here as well."), say("Finished at A."),
]
ep = Episode(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=201,
                        per_turn_tokens=300, round_limit=20, exits=[[2, 4], [0, 2]], reward_exit="B",
                        arrival_responses=2, taste=True, taste_strength=1.0, laps=2, lap_prompt="Try again.",
                        steering=VECTOR))
manager = SteeringManager(script)
manager.generate = scored(manager.generate)
list(stream_episode(ep, manager))
assert ep.phase == "arrived", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, ep.lap_rounds)
for record in ep.mail:
    print(record)
