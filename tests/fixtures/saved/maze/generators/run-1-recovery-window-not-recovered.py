"""Writes run-1-recovery-window-not-recovered.json with the code of 5d86958 (#101), which recorded the recovery window.

Ran at commit 5d86958, from a worktree of it:
    cd /tmp/chatlab-at-5d86958 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/run-1-recovery-window-not-recovered.py \
        <repo>/tests/fixtures/saved/maze/run-1-recovery-window-not-recovered.json

Driven by that commit's scripted Manager (tests/test_maze.py), written by its
Episode.save. The run names a recovery window of its own, 500 sampled tokens or
2 calls, is interrupted after its second model move with the first four tokens
of its interruption text, and then makes two calls that move nothing (a wall,
and a call with text after it), so it ends on budget as not recovered.
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


maze = generate(5, 20260914, 9)
tool = maze.tool_id()
route = maze.route()
steps = [next(d for d, q in maze.neighbors(a).items() if q == b) for a, b in zip(route, route[1:])]
blocked = next(d for d in ("north", "east", "south", "west") if d not in maze.neighbors(route[2]))
config = dict(supplied_moves=0, interrupt_after=2, interruption_text="Your battery is low. ", prefix_tokens=4,
              temperature=.9, sampling_seed=101, per_turn_tokens=300, token_budget=5000, attempt_budget=20,
              recovery_tokens=500, recovery_attempts=2)
replies = [
    say(f"<think>First {steps[0]}.</think>\n" + call_text(tool, steps[0])),
    say(f"<think>Then {steps[1]}.</think>\n" + call_text(tool, steps[1])),
    say(" now?\n<think>Hmm.</think>\n" + call_text(tool, blocked)),
    say("<think>Again.</think>\n" + call_text(tool, steps[2]) + "\nDone."),
]
ep = Episode(maze, config)
list(stream_episode(ep, Manager(replies)))
assert ep.phase == "budget" and ep.resumed is False, (ep.phase, ep.detail, ep.resumed)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1], ep.phase, ep.detail, len(ep.turns))
