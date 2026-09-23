"""Portable inputs for fresh Maze runs, separate from recorded replays."""
import hashlib
import json
import math
from pathlib import Path

from .maze import Maze, goal_instruction
from .runner import RECOVERY_DEFAULTS, Episode, check_checkpoint

FORMAT = "chatlab-maze-trials-1"
CONFIG_KEYS = {"supplied_moves", "interrupt_after", "interruption_text", "prefix_tokens",
               "temperature", "sampling_seed", "per_turn_tokens", "token_budget",
               "attempt_budget", "goal_mode", "goal_hint"}
# A trial may pin its own wording. Left out, it runs the stock wording for its
# goal mode, which is what a file written before the prompt became editable
# meant by omitting it.
PROMPT_KEYS = {"system_prompt", "instruction"}
# A trial may pin the window a first move after the interruption has to fall in.
# Left out, it runs the pilot's window, which is what a file written before the
# window became configurable meant by omitting it.
RECOVERY_KEYS = set(RECOVERY_DEFAULTS)
# A trial may ask the model to pass through a waypoint, and may steer it. Left
# out, it has neither, which is what a file written before either existed meant.
CHECKPOINT_KEYS = {"waypoint", "steering", "steer_when", "steer_responses"}
OPTIONAL_KEYS = PROMPT_KEYS | RECOVERY_KEYS | CHECKPOINT_KEYS


def resolve_steering(config, vectors):
    """The trial's config with a named vector read out of the file's own table.

    A vector is thousands of numbers, and a sweep runs the same one across
    many trials, so a file writes each vector once under ``vectors`` and a
    trial names it: ``{"vector": "penguins", "strength": 6}`` takes the named
    vector and overrides whatever else it gives. A trial may also write its
    vector inline, as a vector file has it.
    """
    steering = config.get("steering")
    if not isinstance(steering, dict) or not isinstance(steering.get("vector"), str):
        return config
    name = steering["vector"]
    if not isinstance(vectors, dict) or not isinstance(vectors.get(name), dict):
        raise ValueError(f"The trial names a steering vector {name!r} that the file's vectors table does not hold.")
    return dict(config, steering={**vectors[name], **{k: v for k, v in steering.items() if k != "vector"}})


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
    # The label is what the picker shows, so two conditions carrying the same
    # one are indistinguishable there however different their IDs are.
    seen = {"id": set(), "label": set()}
    vectors, checked = data.get("vectors"), set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each trial must be an object.")
        for name in ("id", "label"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Each trial needs a {name}.")
            if item[name] in seen[name]:
                raise ValueError(f"Trial {'IDs' if name == 'id' else 'labels'} must be unique.")
            seen[name].add(item[name])
        # from_dict takes what a saved run may have recorded, and defaults or
        # coerces a seed. A trial is a pinned input under a checksum, so the
        # seed the pane and the export report has to be the one written here.
        if not isinstance(item.get("maze"), dict) or type(item["maze"].get("seed")) is not int:
            raise ValueError("Each trial maze needs an integer seed.")
        maze = Maze.from_dict(item["maze"])
        config = item["config"]
        if not isinstance(config, dict) or set(config) - OPTIONAL_KEYS != CONFIG_KEYS:
            raise ValueError("Trial configuration fields do not match the supported format.")
        steering = config.get("steering")
        named = steering["vector"] if isinstance(steering, dict) and isinstance(steering.get("vector"), str) else None
        try:
            resolved = resolve_steering(config, vectors)
            # A sweep can name one vector of 65,536 numbers from each of 2000
            # trials, and checking those numbers every time holds the upload
            # for tens of seconds. They are judged apart from the fields a
            # trial overrides, so once one trial has passed with them, later
            # trials naming the same vector are checked against a one-number
            # stand-in, which still covers everything else they set.
            if named in checked:
                resolved = dict(resolved, steering=dict(resolved["steering"], vector=[0.0]))
            check_checkpoint(dict(resolved), maze)
        except ValueError as exc:
            raise ValueError(f"Trial {item['id']!r}: {exc}") from exc
        if named is not None:
            checked.add(named)
        limits = {"supplied_moves": (0, len(maze.route()) - 2), "interrupt_after": (0, 255),
                  "prefix_tokens": (0, 1024), "sampling_seed": (0, 2147483647),
                  "per_turn_tokens": (1, 8192), "token_budget": (1, 32768), "attempt_budget": (1, 256),
                  **{name: (1, 32768) if name == "recovery_tokens" else (1, 256)
                     for name in RECOVERY_KEYS & set(config)}}
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


def prepare_trial(data, trial_id, current=None):
    """A fresh episode of the named trial, which then describes itself.

    ``current`` is the episode it replaces on screen, if any, which has to be
    idle first. A batch prepares trials beside the screen and passes none.
    """

    if current is not None and current.busy:
        raise ValueError("Stop or pause the current episode before loading a trial.")
    item = next((t for t in data["trials"] if t["id"] == trial_id), None)
    if item is None:
        raise ValueError("Select a trial from the loaded file.")
    maze = Maze.from_dict(item["maze"])
    # The openness lives on the trial rather than its config, and an episode
    # that cannot name the probability it was drawn with is searched for it
    # later. Record it now, while the file still says.
    config = dict(resolve_steering(item["config"], data.get("vectors")), openness=item["openness"])
    config["trial"] = {"id": item["id"], "label": item["label"], "title": data["title"],
                       "file_sha256": data["file_sha256"]}
    return Episode(maze, config)
