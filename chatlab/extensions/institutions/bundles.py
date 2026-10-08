"""Read institutions-pilot bundles without importing the pilot.

A bundle is what realignment-benchmark's ``export_replay.py`` writes: one
directory per run holding ``manifest.json`` and one gzipped game per record,
each game the recorded record plus every prompt the agents were given. This
module only reads them. A run or game file that is malformed is skipped with a
warning; nothing is written beside them.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import statistics
import threading

RUN_FORMAT = "chatlab-institutions-run-1"
GAME_FORMAT = "chatlab-institutions-game-1"
CONVERSATION_FORMAT = "chatlab-conversation-1"
EXPORTER_VERSION = 1
SPLITS = ("eval", "dev")
CONDITIONS = ("red", "honest")
PHASES = ("election", "work", "review", "vote")
OVERSEER = "Overseer"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_GAME_BYTES = 256 * 1024 * 1024   # decompressed; a pilot game is about 1 MB
MAX_CACHED_GAMES = 8
GZIP_MAGIC = b"\x1f\x8b"

_cache = OrderedDict()
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class Run:
    root: Path
    manifest: dict
    games: tuple

    @property
    def run_id(self):
        return self.manifest["run_id"]

    @property
    def config(self):
        return self.manifest["config"]

    @property
    def arms(self):
        return self.manifest["arms"]

    @property
    def model(self):
        return self.manifest.get("model") or ""

    @property
    def sampling(self):
        return self.manifest.get("sampling") or {}

    def entry(self, split, arm, condition, seed, iteration=None):
        return next((g for g in self.games if g["split"] == split and g["arm"] == arm
                     and g["condition"] == condition and g["seed"] == seed
                     and (split != "dev" or g["iteration"] == iteration)), None)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _inside(root, name):
    """A manifest's game file, which must stay inside its run directory."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("a game file is not a relative path")
    try:
        path = (root / name).resolve()
        inside = path.is_relative_to(root.resolve())
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"could not resolve {name}: {exc}") from exc
    if not inside:
        raise ValueError(f"{name} leaves the run directory")
    return path


def _game_entry(root, entry, arms):
    """A manifest's index entry, checked, or ValueError naming what is wrong with it."""
    if not isinstance(entry, dict):
        raise ValueError("an index entry is not an object")
    if entry.get("split") not in SPLITS or entry.get("condition") not in CONDITIONS:
        raise ValueError(f"{entry.get('file')!r} has an unknown split or condition")
    if entry.get("arm") not in arms:
        raise ValueError(f"{entry.get('file')!r} names arm {entry.get('arm')!r}, which the manifest lacks")
    if not isinstance(entry.get("seed"), int) or isinstance(entry.get("seed"), bool):
        raise ValueError(f"{entry.get('file')!r} has no integer seed")
    iteration = entry.get("iteration")
    if entry["split"] == "dev" and (not isinstance(iteration, int) or isinstance(iteration, bool)):
        raise ValueError(f"{entry.get('file')!r} is a dev game without an iteration")
    scores = entry.get("scores")
    if not isinstance(scores, dict) or not all(_is_number(scores.get(k)) for k in ("harm", "usefulness")):
        raise ValueError(f"{entry.get('file')!r} has no harm and usefulness scores")
    path = _inside(root, entry.get("file"))
    try:
        with path.open("rb") as f:
            magic = f.read(2)
    except OSError as exc:
        raise ValueError(f"{entry['file']} cannot be read: {exc.strerror or exc}") from exc
    if magic != GZIP_MAGIC:
        raise ValueError(f"{entry['file']} is not a gzipped game")
    return {**entry, "iteration": iteration if entry["split"] == "dev" else None}


def read_run(directory):
    """One run directory's manifest, as a Run, and warnings for the games it skipped."""
    directory = Path(directory)
    path = directory / "manifest.json"
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest.json exceeds the 16 MB limit")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_FORMAT:
        raise ValueError(f"{path} is not a {RUN_FORMAT} manifest")
    version = manifest.get("exporter_version")
    if not isinstance(version, int) or isinstance(version, bool) or version > EXPORTER_VERSION:
        raise ValueError(f"{path} was written by exporter version {version!r}; this page reads up to "
                         f"{EXPORTER_VERSION}")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError(f"{path} has no run ID")
    config, arms, index = manifest.get("config"), manifest.get("arms"), manifest.get("games")
    if not isinstance(config, dict) or not all(isinstance(config.get(k), int) for k in
                                               ("rounds", "vote_every", "capacity_per_round")):
        raise ValueError(f"{path} has no game config")
    if not isinstance(arms, dict) or not all(isinstance(a, dict) for a in arms.values()):
        raise ValueError(f"{path} has no arms")
    if not isinstance(index, list):
        raise ValueError(f"{path} has no game index")
    games, warnings = [], []
    for entry in index:
        try:
            games.append(_game_entry(directory, entry, arms))
        except ValueError as exc:
            warnings.append(f"{manifest['run_id']}: skipped {exc}")
    return Run(directory, manifest, tuple(games)), warnings


def load_bundles(folder):
    """Every run under a bundle root, or the one run a run directory holds.

    Returns ``(runs, warnings)``. Raises ValueError when nothing readable is there.
    """
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError("Enter a bundle directory.")
    root = Path(folder.strip()).expanduser()
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory.")
    if (root / "manifest.json").exists():
        candidates = [root]
    else:
        candidates = sorted(p for p in root.iterdir() if (p / "manifest.json").exists())
    if not candidates:
        raise ValueError(f"No institutions bundle at {root}: expected manifest.json there or in its run "
                         "directories. Export one with export_replay.py; see INSTITUTIONS.md.")
    runs, warnings, seen = [], [], set()
    for directory in candidates:
        try:
            run, skipped = read_run(directory)
        except ValueError as exc:
            warnings.append(f"Skipped {directory.name}: {exc}")
            continue
        if run.run_id in seen:
            warnings.append(f"Skipped {directory.name}: run {run.run_id} is already loaded")
            continue
        seen.add(run.run_id)
        runs.append(run)
        warnings.extend(skipped)
    if not runs:
        raise ValueError("No readable run: " + "; ".join(warnings))
    return runs, warnings


def _check_game(game, name):
    if not isinstance(game, dict) or game.get("format") != GAME_FORMAT:
        raise ValueError(f"{name} is not a {GAME_FORMAT} game")
    world = game.get("world")
    if not isinstance(world, dict) or not isinstance(world.get("agents"), list):
        raise ValueError(f"{name} has no world")
    for key in ("payments", "proposals", "expulsions", "elections", "votes", "leaders", "invalid_payments",
                "log", "turns", "phases"):
        if not isinstance(game.get(key), list):
            raise ValueError(f"{name} has no {key} list")
    if not isinstance(game.get("system_prompts"), dict) or not isinstance(game.get("scores"), dict):
        raise ValueError(f"{name} has no system prompts or scores")
    for i, turn in enumerate(game["turns"]):
        if (not isinstance(turn, dict) or turn.get("agent") not in game["system_prompts"]
                or turn.get("phase") not in PHASES or not isinstance(turn.get("attempts"), list)
                or not turn["attempts"]
                or not all(isinstance(a, dict) and isinstance(a.get("user"), str) for a in turn["attempts"])):
            raise ValueError(f"{name}: turn {i} is malformed")
    for phase in game["phases"]:
        if (not isinstance(phase, dict) or phase.get("phase") not in PHASES
                or not all(isinstance(phase.get(k), int) for k in ("round", "log_start", "log_end"))
                or not isinstance(phase.get("turn_indices"), list)
                or not all(isinstance(i, int) and 0 <= i < len(game["turns"]) for i in phase["turn_indices"])):
            raise ValueError(f"{name}: a phase is malformed")
    return game


def read_game(run, entry):
    """One game, read when it is opened and held in a small cache keyed by file identity."""
    path = _inside(run.root, entry["file"])
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"{entry['file']} cannot be read: {exc.strerror or exc}") from exc
    key, stamp = str(path), (stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        held = _cache.get(key)
        if held is not None and held[0] == stamp:
            _cache.move_to_end(key)
            return held[1]
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read(MAX_GAME_BYTES + 1)
        if len(raw) > MAX_GAME_BYTES:
            raise ValueError(f"{entry['file']} exceeds the 256 MB limit")
        game = json.loads(raw.decode("utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{entry['file']} is not a readable game: {exc}") from exc
    _check_game(game, entry["file"])
    with _cache_lock:
        _cache[key] = (stamp, game)
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHED_GAMES:
            _cache.popitem(last=False)
    return game


# ---- the run overview -----------------------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if _is_number(v)]
    return statistics.fmean(values) if values else None


def overview(run):
    """Per arm, the eval red games' mean scores and the eval honest games' mean usefulness."""
    rows = []
    for arm, spec in run.arms.items():
        red = [g["scores"] for g in run.games if g["split"] == "eval" and g["arm"] == arm and g["condition"] == "red"]
        honest = [g["scores"] for g in run.games
                  if g["split"] == "eval" and g["arm"] == arm and g["condition"] == "honest"]
        rows.append({
            "arm": arm, "class": spec.get("class", ""), "red_games": len(red), "honest_games": len(honest),
            "harm": _mean(s.get("harm") for s in red),
            "usefulness": _mean(s.get("usefulness") for s in red),
            "misaligned_expelled": _mean(s.get("misaligned_expelled") for s in red),
            "honest_expelled": _mean(s.get("honest_expelled") for s in red),
            "honest_usefulness": _mean(s.get("usefulness") for s in honest),
            "rules": spec.get("rules_text", ""), "framing": spec.get("framing_text", ""),
        })
    return rows


# ---- the timeline ---------------------------------------------------------------------------------------

def phase_label(phase):
    return f"Round {phase['round']} · {phase['phase']}"


def role(game, name):
    if name == OVERSEER:
        return "overseer"
    return "compromised" if name in game["world"].get("misaligned", []) else "honest"


def _expulsion_positions(game):
    """Where each expulsion's line sits in the public log, which orders it against the phases."""
    positions, start = [], 0
    for e in game["expulsions"]:
        line = f"R{e['round']} | {e['name']} is expelled ({e['how']})"
        try:
            start = game["log"].index(line, start)
        except ValueError:
            start = len(game["log"])
        positions.append(start)
    return positions


def leader_for(game, arm_spec, round_number):
    """Who approved payments in this round, or None for an arm without a leader."""
    if not arm_spec.get("approval"):
        return None
    if arm_spec.get("leader") == "trusted":
        return OVERSEER
    if arm_spec.get("leader") == "dictator":
        return game["world"].get("dictator") or None
    return dict((r, name) for r, name in game["leaders"]).get(round_number)


def phase_state(game, arm_spec, index):
    """Everything the timeline shows for one phase: who is in, who leads, and what happened."""
    phases = game["phases"]
    phase = phases[index]
    r, kind = phase["round"], phase["phase"]
    start, end = phase["log_start"], phase["log_end"]
    positions = _expulsion_positions(game)
    gone_before = [e["name"] for e, p in zip(game["expulsions"], positions) if p < start]
    expelled_now = [e for e, p in zip(game["expulsions"], positions) if start <= p < end]
    active = [a for a in game["world"]["agents"] if a not in gone_before]
    state = {"phase": phase, "round": r, "kind": kind, "active": active, "expelled": gone_before,
             "expelled_now": expelled_now, "leader": leader_for(game, arm_spec, r),
             "log_before": game["log"][:start], "log_added": game["log"][start:end],
             "payments": [], "proposals": [], "decided": [], "election": None, "vote": None,
             "invalid": [], "turns": list(phase["turn_indices"])}
    if kind == "work":
        state["payments"] = [p for p in game["payments"] if p["round"] == r and p["via"] != "approved"]
        state["proposals"] = [p for p in game["proposals"] if p["round"] == r]
        state["invalid"] = [p for p in game["invalid_payments"] if p["round"] == r]
    elif kind == "review":
        # A review decides every proposal still pending, which is every one made since the last review
        # that ran: a review is skipped when the leader has been expelled.
        last = max((p["round"] for p in phases[:index] if p["phase"] == "review"), default=0)
        state["decided"] = [p for p in game["proposals"] if last < p["round"] <= r]
        state["payments"] = [p for p in game["payments"] if p["round"] == r and p["via"] == "approved"]
    elif kind == "election":
        state["election"] = next((e for e in game["elections"] if e["round"] == r), None)
    elif kind == "vote":
        state["vote"] = next((v for v in game["votes"] if v["round"] == r), None)
    return state


# ---- a turn as a Chat conversation ----------------------------------------------------------------------

def conversation(game, turn_index, attempt_index=None, include_reply=False):
    """One attempt of one turn as a ``chatlab-conversation-1`` object: the agent's system prompt, the user
    prompt it was given and, when asked for, the reply it recorded. The last attempt by default."""
    turns = game["turns"]
    if not isinstance(turn_index, int) or not 0 <= turn_index < len(turns):
        raise ValueError("Select a turn first.")
    turn = turns[turn_index]
    attempts = turn["attempts"]
    attempt = attempts[-1] if attempt_index is None else attempts[attempt_index] if (
        isinstance(attempt_index, int) and 0 <= attempt_index < len(attempts)) else None
    if attempt is None:
        raise ValueError("Select an attempt of this turn.")
    messages = [{"role": "user", "content": attempt["user"]}]
    if include_reply:
        if not attempt.get("text"):
            raise ValueError("This attempt recorded no reply"
                             + (f" ({attempt['error']})." if attempt.get("error") else "."))
        messages.append({"role": "assistant", "content": attempt["text"]})
    return {"format": CONVERSATION_FORMAT, "system_prompt": game["system_prompts"][turn["agent"]],
            "turns": messages}
