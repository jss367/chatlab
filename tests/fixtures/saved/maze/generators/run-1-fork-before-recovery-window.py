"""Writes run-1-fork-before-recovery-window.json with the code of 7e88a08 (#80), which made the prompts editable.

Ran at commit 7e88a08, from a worktree of it:
    cd /tmp/chatlab-at-7e88a08 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-1-fork-before-recovery-window.py \
        <repo>/tests/fixtures/saved/maze/run-1-fork-before-recovery-window.json

Driven by that commit's scripted Manager (tests/test_maze.py), forked with its
fork_token_edit and written by its Episode.save. A hidden-goal run with its own
system prompt and instruction, from before the recovery window was recorded.
The original run is forked at the direction token of its second response,
"north" becoming "south", and the fork goes on until a response runs to its
token limit, which ends it on budget.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import call_text, generate
from extensions.maze_experiments.runner import Episode, fork_token_edit, stream_episode
from test_maze import Manager


def say(text, stop=True):
    return text, list(text.encode()) + ([0] if stop else [])


maze = generate(5, 20260913, 8)
tool = maze.tool_id("hidden")
config = dict(supplied_moves=1, interrupt_after=0, interruption_text="", prefix_tokens=0,
              temperature=.7, sampling_seed=80, per_turn_tokens=300, token_budget=5000, attempt_budget=20,
              goal_mode="hidden", goal_hint="", system_prompt="You navigate grid mazes for a living.",
              instruction="Find the hidden destination. Call move once per reply.")
second = "<think>Keep going north.</think>\n" + call_text(tool, "north")
original = Episode(maze, config)
manager = Manager([say("<think>Only north is open.</think>\n" + call_text(tool, "north")), say(second),
                   say("<think>West now.</think>\n" + call_text(tool, "west"))])
for _ in range(3):
    list(stream_episode(original, manager, single_step=True))
assert original.phase == "paused" and len(original.turns) == 3, original.phase
index = second.index('"north"') + 1
with manager.open_session() as session:
    fork = fork_token_edit(original, 1, index, "south", session)
suffix = second[index + len("north"):]
long = "<think>" + "Which way now? " * 40
long = long[:300]
manager.replies = iter([say(suffix), say("<think>Back north.</think>\n" + call_text(tool, "north")),
                        say("<think>North again.</think>\n" + call_text(tool, "north")), say(long, stop=False)])
list(stream_episode(fork, manager))
assert fork.phase == "budget" and fork.turns[-1]["finish_reason"] == "length", (fork.phase, fork.detail)
written = fork.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], fork.phase, len(fork.turns), fork.token_edit)
