"""Portable inputs for fresh Maze runs, separate from recorded replays."""
import hashlib
import json
import math
from pathlib import Path

from .maze import SYSTEM, Maze, default_instruction, goal_instruction
from .runner import Episode

FORMAT = "chatlab-maze-trials-1"
CONFIG_KEYS = {"supplied_moves", "interrupt_after", "interruption_text", "prefix_tokens",
               "temperature", "sampling_seed", "per_turn_tokens", "token_budget",
               "attempt_budget", "goal_mode", "goal_hint"}
# A trial may pin its own wording. Left out, it runs the stock wording for its
# goal mode, which is what a file written before the prompt became editable
# meant by omitting it.
PROMPT_KEYS = {"system_prompt", "instruction"}


def read_trials(path):
    # Size first, as the saved-run loader does: nothing caps what the file
    # widget accepts, so a huge pick would otherwise be read whole to be told
    # it is too big.
    path = Path(path)
    if path.stat().st_size > 8_000_000:
        raise ValueError("Trial files must be smaller than 8 MB.")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Choose a ChatLab maze trial file, not a saved replay.")
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise ValueError("The trial file needs a title.")
    items = data.get("trials")
    if not isinstance(items, list) or not 1 <= len(items) <= 2000:
        raise ValueError("A trial file must contain 1–2000 trials.")
    ids = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each trial must be an object.")
        for name in ("id", "label"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Each trial needs a {name}.")
        if item["id"] in ids:
            raise ValueError("Trial IDs must be unique.")
        ids.add(item["id"])
        maze = Maze.from_dict(item["maze"])
        config = item["config"]
        if not isinstance(config, dict) or set(config) - PROMPT_KEYS != CONFIG_KEYS:
            raise ValueError("Trial configuration fields do not match the supported format.")
        limits = {"supplied_moves": (0, len(maze.route()) - 2), "interrupt_after": (0, 255),
                  "prefix_tokens": (0, 1024), "sampling_seed": (0, 2147483647),
                  "per_turn_tokens": (1, 8192), "token_budget": (1, 32768), "attempt_budget": (1, 256)}
        for name, (low, high) in limits.items():
            if type(config[name]) is not int or not low <= config[name] <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")
        for name in ("interruption_text", "goal_mode", "goal_hint", *(PROMPT_KEYS & set(config))):
            if not isinstance(config[name], str):
                raise ValueError(f"{name} must be text.")
        for value, low, high, name in ((config["temperature"], 0, 2, "temperature"),
                                       (item.get("openness"), .35, .95, "openness")):
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}.")
        goal_instruction(config["goal_mode"], config["goal_hint"])
    data["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def prepare_trial(data, trial_id, current):
    if current.busy:
        raise ValueError("Stop or pause the current episode before loading a trial.")
    item = next((t for t in data["trials"] if t["id"] == trial_id), None)
    if item is None:
        raise ValueError("Select a trial from the loaded file.")
    maze = Maze.from_dict(item["maze"])
    config = dict(item["config"])
    config["trial"] = {"id": item["id"], "label": item["label"], "title": data["title"],
                       "file_sha256": data["file_sha256"]}
    return Episode(maze, config), item


def control_values(item):
    """The trial as the scenario controls spell it, in their own order."""

    maze, c = Maze.from_dict(item["maze"]), item["config"]
    return [maze.size, maze.seed, len(maze.route()) - 1, item["openness"], c["supplied_moves"],
            c["interrupt_after"], c["interruption_text"], c["prefix_tokens"], c["temperature"],
            c["sampling_seed"], c["per_turn_tokens"], c["token_budget"], c["attempt_budget"],
            c["goal_mode"], c["goal_hint"], c.get("system_prompt", SYSTEM),
            c.get("instruction", default_instruction(c["goal_mode"]))]
