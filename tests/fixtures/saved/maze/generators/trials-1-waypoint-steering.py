"""Writes trials-1-waypoint-steering.json, a trial file in the shape 22361e8 (#148) read.

A trial file is an input people write, so this script writes one with every
optional key that commit accepted: its own wording, a recovery window, a
waypoint, and steering, both by a vector named from the file's vectors table
(with its strength overridden) and by one written inline. The file is read
back through that commit's read_trials and each trial is prepared with its
prepare_trial, so the file is one that commit loaded. Ran from a worktree of
it:
    cd /tmp/chatlab-at-22361e8 && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/trials-1-waypoint-steering.py \
        <repo>/tests/fixtures/saved/maze/trials-1-waypoint-steering.json
"""
import json
import sys
from pathlib import Path

from extensions.maze_experiments.maze import generate
from extensions.maze_experiments.runner import Episode
from extensions.maze_experiments.trials import prepare_trial, read_trials
from test_maze import CONFIG, MAZE
from test_maze_steering import VECTOR

maze = generate(5, 303, 8, .7)
route = maze.route()
base = dict(supplied_moves=0, interrupt_after=0, interruption_text="", prefix_tokens=0, temperature=.7,
            sampling_seed=303, per_turn_tokens=400, token_budget=6000, attempt_budget=24, goal_mode="coordinates",
            goal_hint="")
trials = {"format": "chatlab-maze-trials-1", "title": "Waypoint and steering trials",
          "vectors": {"calm": dict(VECTOR, strength=2.0)}, "trials": [
              {"id": "named-vector", "label": "Named vector at the waypoint", "maze": maze.to_dict(), "openness": .7,
               "config": dict(base, waypoint=list(route[3]), steering={"vector": "calm", "strength": 5.0},
                              steer_when={"cell": list(route[3])}, steer_responses=2, recovery_tokens=600,
                              recovery_attempts=3)},
              {"id": "inline-vector", "label": "Inline vector after moves", "maze": maze.to_dict(), "openness": .7,
               "config": dict(base, steering=VECTOR, steer_when={"moves": 2}, steer_responses=0,
                              system_prompt="Be terse.", instruction="Reach the destination.")},
              {"id": "plain", "label": "No waypoint, no vector", "maze": maze.to_dict(), "openness": .7,
               "config": dict(base)},
          ]}
Path(sys.argv[1]).write_text(json.dumps(trials, indent=1))
data = read_trials(sys.argv[1])
for item in data["trials"]:
    ep = prepare_trial(data, item["id"], Episode(MAZE, CONFIG))
    print(item["id"], ep.config.get("steering", {}).get("strength"), ep.config.get("waypoint"),
          ep.config["recovery_tokens"])
