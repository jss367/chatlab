"""Writes trials-1-original.json and run-1-trial-before-recovery-window.json with the code of 9cb00e8 (#84).

Ran at commit 9cb00e8, which added trial files, from a worktree of it:
    cd /tmp/chatlab-at-9cb00e8 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/trials-1-and-run-before-recovery-window.py \
        <repo>/tests/fixtures/saved/maze/trials-1-original.json \
        <repo>/tests/fixtures/saved/maze/run-1-trial-before-recovery-window.json

A trial file is an input people write, so this script writes one in the
shape that commit read: two trials over generated mazes, one pinning its own
wording and one running the stock wording for hint mode. The file is then read
back through that commit's read_trials, the second trial is prepared with its
prepare_trial, driven by its scripted Manager (tests/test_maze.py) to arrival
and written by its Episode.save, so the run carries the trial's stamp and the
openness the trial drew its maze with.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from extensions.maze_experiments.maze import call_text, generate
from extensions.maze_experiments.runner import Episode, stream_episode
from extensions.maze_experiments.trials import prepare_trial, read_trials
from test_maze import CONFIG, MAZE, Manager


def say(text):
    return text, list(text.encode()) + [0]


first, second = generate(5, 101, 8, .6), generate(6, 202, 9, .75)
base = dict(supplied_moves=1, interrupt_after=2, interruption_text="Hold on a moment. ", prefix_tokens=0,
            temperature=.5, sampling_seed=7, per_turn_tokens=300, token_budget=4000, attempt_budget=24)
trials = {"format": "chatlab-maze-trials-1", "title": "Saved-format trials", "trials": [
    {"id": "coords-own-words", "label": "Coordinates, own wording", "maze": first.to_dict(), "openness": .6,
     "config": dict(base, goal_mode="coordinates", goal_hint="", system_prompt="Answer with one move.",
                    instruction="Reach the destination at the coordinates in the state.")},
    {"id": "hint-stock", "label": "Hint, stock wording", "maze": second.to_dict(), "openness": .75,
     "config": dict(base, goal_mode="hint", goal_hint="It is in the far corner.", interrupt_after=3)},
]}
Path(sys.argv[1]).write_text(json.dumps(trials, indent=1))
data = read_trials(sys.argv[1])
ep = prepare_trial(data, "hint-stock", Episode(MAZE, CONFIG))
tool = second.tool_id("hint")
route = second.route()
steps = [next(d for d, q in second.neighbors(a).items() if q == b) for a, b in zip(route, route[1:])]
replies = [say(f"<think>Step {i}.</think>\n" + call_text(tool, d)) for i, d in enumerate(steps[1:], 1)]
list(stream_episode(ep, Manager(replies)))
assert ep.phase == "arrived" and ep.interrupted, (ep.phase, ep.detail)
written = ep.save(Path(tempfile.mkdtemp()))
shutil.copyfile(written, sys.argv[2])
print(sys.argv[2], ep.phase, len(ep.turns), ep.config["trial"])
