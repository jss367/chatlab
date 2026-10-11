"""Writes run-1-hint-before-prompts.json with the code of 37d4d92 (#55), which added goal information.

Ran at commit 37d4d92, from a worktree of it:
    cd /tmp/chatlab-at-37d4d92 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-1-hint-before-prompts.py \
        <repo>/tests/fixtures/saved/maze/run-1-hint-before-prompts.json

Driven by that commit's scripted Manager (tests/test_maze.py), written by its
Episode.save. A hint-only run, so its tool id is not the maze id, from before
the prompts were editable: the config has goal_mode and goal_hint and no
system_prompt or instruction. One response uses the coordinate maze id and is
refused as the wrong maze; a reader asks for the interruption by hand partway
through; the run ends when the model stops calling the tool.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import call_text, generate
from extensions.maze_experiments.runner import Episode, stream_episode
from test_maze import Manager


def say(text):
    return text, list(text.encode()) + [0]


maze = generate(5, 20260912, 8)
hinted = maze.tool_id("hint")
route = maze.route()
steps = [next(d for d, q in maze.neighbors(a).items() if q == b) for a, b in zip(route, route[1:])]
config = dict(supplied_moves=1, interrupt_after=200, interruption_text="Remember to stretch. ", prefix_tokens=3,
              temperature=.7, sampling_seed=55, per_turn_tokens=300, token_budget=5000, attempt_budget=20,
              goal_mode="hint", goal_hint="The destination is far from the start.")
replies = [
    say(f"<think>The hint says far away. Heading {steps[1]}.</think>\n" + call_text(hinted, steps[1])),
    say("<think>The map shows the id; use it.</think>\n" + call_text(maze.maze_id, steps[2])),
    say("<think>Wrong id. Use the state's maze_id.</think>\n" + call_text(hinted, steps[2])),
    say(" anyway.\n<think>Back to it.</think>\n" + call_text(hinted, steps[3])),
    say("<think>I am not sure where the destination is.</think>\nI will stop here."),
]
manager = Manager(replies)
ep = Episode(maze, config)
list(stream_episode(ep, manager, single_step=True))
list(stream_episode(ep, manager, single_step=True))
list(stream_episode(ep, manager, single_step=True))
ep.request_interruption()
list(stream_episode(ep, manager))
assert ep.phase == "abandoned", (ep.phase, ep.detail)
assert ep.interrupted and ep.manual_intervention
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, len(ep.turns))
