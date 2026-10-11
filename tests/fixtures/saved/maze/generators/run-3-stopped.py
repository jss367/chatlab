"""Writes run-3-stopped.json with the code of ffcbd7b (#156), which added inserted messages.

Ran at commit ffcbd7b, from a worktree of it:
    cd /tmp/chatlab-at-ffcbd7b && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-3-stopped.py \
        <repo>/tests/fixtures/saved/maze/run-3-stopped.json

Driven by that commit's scripted Manager (tests/test_maze.py), written by its
Episode.save. A user message goes in before the second response, and the
reader stops the run at the opening frame of the fourth, so that response was
never generated and records no prompt, though it comes after the message.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import Maze, call_text
from extensions.maze_experiments.runner import Episode, stream_episode
from test_maze import Manager


def say(text):
    return text, list(text.encode()) + [0]


maze = Maze(("....", ".##.", ".##.", "...."), (0, 0), (0, 3), seed=1562)
tool = maze.tool_id()
manager = Manager([say(f"<think>{d}.</think>\n" + call_text(tool, d)) for d in ("south", "south", "east", "east")])
ep = Episode(maze, dict(supplied_moves=0, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
                        sampling_seed=1562, per_turn_tokens=300, token_budget=5000, attempt_budget=20))
list(stream_episode(ep, manager, single_step=True))
ep.request_insert("user", "The long way round is fine.")
for frame in stream_episode(ep, manager):
    if len(ep.turns) == 4 and not ep.turns[-1]["metrics"] and not ep.stop_requested:
        ep.request_stop()
assert ep.phase == "stopped" and ep.turns[-1]["finish_reason"] == "user_stopped", (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, ep.config["context_inserts"])
