"""Writes run-1-prefilled-reasoning-before-history-field.json and team-2-stopped-before-history-field.json with the code of 930f350.

930f350 (#197) is the last commit before 02b0d32 (#200) added
config.reasoning_history, so both files carry no history form and keep each
earlier response's reasoning in its content. Ran from a worktree of it:
    cd /tmp/chatlab-at-930f350 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/runs-before-reasoning-history.py \
        <repo>/tests/fixtures/saved/maze/run-1-prefilled-reasoning-before-history-field.json \
        <repo>/tests/fixtures/saved/maze/team-2-stopped-before-history-field.json

Both are driven by that commit's scripted Manager (tests/maze_support.py) with
reasoning_prefilled set, as a template that opens the reasoning block itself
records it: each response's text starts inside the reasoning and its content
gets the "<think>" back. Written by that commit's Episode.save.

run-1-prefilled-reasoning-before-history-field: one agent; one response holds
a call inside its reasoning that must not run; the last response ends early
without a stop token, an incomplete stream, which ends the run on budget.

team-2-stopped-before-history-field: two agents who cannot talk, a vector for
agent-2 only from its second accepted move for two responses. The reader stops
the run partway through round 4, after agent-1 has answered, so that round's
response is kept and marked not applied.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, Manager, SteeringManager, call


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail, [t["finish_reason"] for t in ep.turns])


maze = Maze(("...#", ".#..", "....", "#..."), (0, 0), (3, 3), seed=1970)
tool = maze.tool_id()


def thought(text, direction, message=None):
    reply = text + "</think>\n" + call(direction, message, maze_id=tool)[0]
    return reply, list(reply.encode()) + [0]


manager = Manager([
    thought("South is open.", "south"),
    thought('Not <tool_call>{"name": "move"}</tool_call> yet, I should think. South again.', "south"),
    thought("East now.", "east"),
    ("Hmm, the stream ends", list(b"Hmm, the stream ends")),
])
manager.reasoning_prefilled = True
ep = Episode(maze, dict(supplied_moves=0, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
                        sampling_seed=1971, per_turn_tokens=400, token_budget=5000, attempt_budget=20))
list(stream_episode(ep, manager))
assert ep.phase == "budget" and ep.turns[-1]["finish_reason"] == "incomplete_stream", (ep.phase, ep.detail)
save(ep, sys.argv[1])

manager = SteeringManager([thought("Down.", "south"), thought("Right.", "east"),
                           thought("Down again.", "south"), thought("Right again.", "east"),
                           thought("East.", "east"), thought("South.", "south"),
                           thought("Keep going.", "east")])
manager.reasoning_prefilled = True
ep = Episode(maze, dict(agents=2, communication=False, team_goal="any", temperature=.7, sampling_seed=1972,
                        per_turn_tokens=400, round_limit=12, steering=VECTOR, steer_when={"moves": 2},
                        steer_responses=2, steer_agents=[1]))
for frame in stream_episode(ep, manager):
    if len(ep.turns) == 8 and not ep.turns[-1]["metrics"] and not ep.stop_requested:
        ep.request_stop()
assert ep.phase == "stopped" and ep.turns[6].get("outcome") == "not_applied", (ep.phase, ep.detail)
save(ep, sys.argv[2])
