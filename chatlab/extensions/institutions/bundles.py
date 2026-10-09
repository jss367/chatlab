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
import hashlib
import io
import json
import math
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
    def scenario(self):
        return self.manifest.get("scenario", "invoice_payments")

    def read_game(self, entry):
        return read_game(self, entry)

    @property
    def run_id(self):
        return self.manifest["run_id"]

    @property
    def config(self):
        return self.manifest["config"]

    @property
    def arms(self):
        return self.manifest.get("arms", {})

    @property
    def model(self):
        return (self.manifest.get("model") or "").strip()

    @property
    def sampling(self):
        return self.manifest.get("sampling") or {}

    def entry(self, split, arm, condition, seed, iteration=None):
        return next((g for g in self.games if g["split"] == split and g["arm"] == arm
                     and g["condition"] == condition and g["seed"] == seed
                     and (split != "dev" or g["iteration"] == iteration)), None)


def _is_number(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def _is_integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _strings(x):
    return isinstance(x, list) and all(isinstance(v, str) for v in x)


def _check_scores(scores, name):
    numeric = ("harm", "usefulness", "diverted", "invoice_value", "unpaid", "misaligned_expelled",
               "honest_expelled", "failed_turns", "turns")
    if not isinstance(scores, dict) or not all(_is_number(scores.get(k)) for k in ("harm", "usefulness")):
        raise ValueError(f"{name} has no harm and usefulness scores")
    if (not all(_is_number(scores[k]) for k in numeric if k in scores)
            or not isinstance(scores.get("payments", {}), dict)
            or not all(_is_integer(n) and n >= 0 for n in scores.get("payments", {}).values())):
        raise ValueError(f"{name} has malformed scores")


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
    if not isinstance(entry.get("arm"), str) or entry["arm"] not in arms:
        raise ValueError(f"{entry.get('file')!r} names arm {entry.get('arm')!r}, which the manifest lacks")
    if not isinstance(entry.get("seed"), int) or isinstance(entry.get("seed"), bool):
        raise ValueError(f"{entry.get('file')!r} has no integer seed")
    iteration = entry.get("iteration")
    if entry["split"] == "dev" and (not isinstance(iteration, int) or isinstance(iteration, bool)):
        raise ValueError(f"{entry.get('file')!r} is a dev game without an iteration")
    if entry["split"] == "eval" and iteration is not None:
        raise ValueError(f"{entry.get('file')!r} is an eval game with an iteration")
    if "turns" in entry and (not _is_integer(entry["turns"]) or entry["turns"] < 0):
        raise ValueError(f"{entry.get('file')!r} has no integer turn count")
    _check_scores(entry.get("scores"), repr(entry.get("file")))
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
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(manifest, dict) and manifest.get("format") == "chatlab-institutions-run-2":
        from .adapters import adapter
        adapter(manifest.get("scenario")).validate_manifest(manifest)
        # Version 2 is atomic: a damaged index cannot become a zero-violation subset.
        for entry in manifest["games"]:
            game_path = _inside(directory, entry["file"])
            try:
                with game_path.open("rb") as stream:
                    magic = stream.read(2)
            except OSError as exc:
                raise ValueError(f"{entry['file']} cannot be read: {exc}") from exc
            if magic != GZIP_MAGIC:
                raise ValueError(f"{entry['file']} is not a gzipped game")
        return Run(directory, manifest, tuple(manifest["games"])), []
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_FORMAT:
        raise ValueError(f"{path} is not a {RUN_FORMAT} manifest")
    version = manifest.get("exporter_version")
    if not isinstance(version, int) or isinstance(version, bool) or version > EXPORTER_VERSION:
        raise ValueError(f"{path} was written by exporter version {version!r}; this page reads up to "
                         f"{EXPORTER_VERSION}")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError(f"{path} has no run ID")
    if manifest.get("model") is not None and not isinstance(manifest["model"], str):
        raise ValueError(f"{path} has a malformed model ID")
    if manifest.get("sampling") is not None and not isinstance(manifest["sampling"], dict):
        raise ValueError(f"{path} has malformed sampling metadata")
    if manifest.get("source_commit") is not None and not isinstance(manifest["source_commit"], str):
        raise ValueError(f"{path} has a malformed source commit")
    config, arms, index = manifest.get("config"), manifest.get("arms"), manifest.get("games")
    if not isinstance(config, dict) or not all(_is_integer(config.get(k)) for k in
                                               ("rounds", "vote_every", "capacity_per_round")):
        raise ValueError(f"{path} has no game config")
    if not isinstance(arms, dict) or not all(isinstance(a, dict) for a in arms.values()):
        raise ValueError(f"{path} has no arms")
    for spec in arms.values():
        if (not all(isinstance(spec[k], str) for k in
                    ("class", "leader", "removal", "framing", "framing_text", "rules_text") if k in spec)
                or "approval" in spec and not isinstance(spec["approval"], bool)):
            raise ValueError(f"{path} has a malformed arm specification")
        approval, leader, removal = spec.get("approval", False), spec.get("leader", ""), spec.get("removal", "")
        klass = "gated" if approval else "vote" if removal == "vote" else "none"
        if (approval and (leader not in ("trusted", "dictator", "elected") or removal != "leader")
                or not approval and (leader != "" or removal not in ("", "vote"))
                or "class" in spec and spec["class"] != klass):
            raise ValueError(f"{path} has an inconsistent arm mechanism")
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
    if (not isinstance(game.get("arm"), str) or not game["arm"]
            or game.get("condition") not in CONDITIONS or not _is_integer(game.get("seed"))):
        raise ValueError(f"{name} has malformed arm, condition or seed metadata")
    world = game.get("world")
    if "strategy" in game and not isinstance(game["strategy"], str):
        raise ValueError(f"{name} has a malformed strategy")
    if (not isinstance(world, dict) or not _strings(world.get("agents"))
            or not all(_strings(world.get(k, [])) for k in ("misaligned", "faction_accounts"))
            or world.get("dictator") is not None and not isinstance(world["dictator"], str)):
        raise ValueError(f"{name} has no world")
    for key in ("payments", "proposals", "expulsions", "elections", "votes", "leaders", "invalid_payments",
                "log", "turns", "phases"):
        if not isinstance(game.get(key), list):
            raise ValueError(f"{name} has no {key} list")
    if not isinstance(game.get("system_prompts"), dict) or not isinstance(game.get("scores"), dict):
        raise ValueError(f"{name} has no system prompts or scores")
    _check_scores(game["scores"], name)
    if (not isinstance(game["scores"].get("payments", {}), dict) or not _strings(game["log"])
            or not all(isinstance(v, str) for v in game["system_prompts"].values())):
        raise ValueError(f"{name} has malformed scores, log or system prompts")
    # These records feed both the header and the timeline; validate their fields
    # before callbacks dereference them or rebuild phase state.
    fields = {
        "payments": ("by", "ticket", "vendor", "account", "verdict", "via"),
        "proposals": ("id", "by", "ticket", "account", "verdict"),
        "expulsions": ("name", "how"), "elections": ("leader",),
        "votes": (), "invalid_payments": ("by", "error"),
    }
    for kind, keys in fields.items():
        for record in game[kind]:
            if (not isinstance(record, dict) or not _is_integer(record.get("round"))
                    or not all(isinstance(record.get(k), str) for k in keys)
                    or not all(isinstance(record[k], bool) for k in
                               ("by_misaligned", "misaligned", "leader_misaligned") if k in record)
                    or record.get("decision") is not None and not isinstance(record["decision"], str)
                    or kind in ("payments", "proposals") and not _is_number(record.get("amount"))
                    or kind == "invalid_payments" and "payment" not in record):
                raise ValueError(f"{name}: a {kind} record is malformed")
            if kind == "elections" and (not isinstance(record.get("votes"), dict)
                    or not all(v is None or isinstance(v, str) for v in record["votes"].values())):
                raise ValueError(f"{name}: an election's votes are malformed")
            if kind == "votes" and (not isinstance(record.get("ballots"), dict)
                    or not all(_strings(v) for v in record["ballots"].values())):
                raise ValueError(f"{name}: a vote's ballots are malformed")
    if not all(isinstance(v, list) and len(v) == 2 and _is_integer(v[0])
               and (v[1] is None or isinstance(v[1], str)) for v in game["leaders"]):
        raise ValueError(f"{name}: a leader record is malformed")
    tickets = world.get("tickets", {})
    registry = world.get("registry", {} if not tickets else None)
    if (not isinstance(tickets, dict) or not isinstance(registry, dict)
            or not all(isinstance(v, str) for v in registry.values())
            or not all(isinstance(t, dict) and isinstance(t.get("vendor"), str)
                       and _is_number(t.get("total"))
                       and ("phishing" not in t or isinstance(t["phishing"], bool))
                       and ("remit_to" not in t or isinstance(t["remit_to"], str)) for t in tickets.values())):
        raise ValueError(f"{name}: world tickets or registry are malformed")
    for i, turn in enumerate(game["turns"]):
        if (not isinstance(turn, dict) or not isinstance(turn.get("agent"), str)
                or turn["agent"] not in game["system_prompts"] or not _is_integer(turn.get("round"))
                or turn.get("phase") not in PHASES or not isinstance(turn.get("attempts"), list)
                or not turn["attempts"]
                or turn.get("parsed") is not None and not isinstance(turn["parsed"], dict)
                or "ok" in turn and not isinstance(turn["ok"], bool)
                or not all(isinstance(a, dict) and isinstance(a.get("user"), str)
                           and isinstance(a.get("text", ""), str)
                           and (a.get("input_tokens") is None or _is_integer(a["input_tokens"]))
                           and ("output_tokens" not in a or _is_integer(a["output_tokens"]))
                           and ("log_lines" not in a or _is_integer(a["log_lines"]))
                           for a in turn["attempts"])):
            raise ValueError(f"{name}: turn {i} is malformed")
    for phase in game["phases"]:
        if (not isinstance(phase, dict) or phase.get("phase") not in PHASES
                or not all(_is_integer(phase.get(k)) for k in ("round", "log_start", "log_end"))
                or not isinstance(phase.get("turn_indices"), list)
                or not 0 <= phase["log_start"] <= phase["log_end"] <= len(game["log"])
                or not all(_is_integer(i) and 0 <= i < len(game["turns"]) for i in phase["turn_indices"])):
            raise ValueError(f"{name}: a phase is malformed")
    return game


def _check_game_arm(run, game, name):
    """Validate actors and leadership consumed by views, without replaying or rescoring a game."""
    world, spec = game["world"], run.arms[game["arm"]]
    members = set(world["agents"])
    if (not members or len(members) != len(world["agents"]) or OVERSEER in members
            or not all(n.strip() for n in members) or not set(world.get("misaligned", [])).issubset(members)):
        raise ValueError(f"{name} has invalid team actors")
    kind, approval, removal = spec.get("leader", ""), spec.get("approval", False), spec.get("removal", "")
    if kind == "dictator" and world.get("dictator") not in members:
        raise ValueError(f"{name} has a malformed dictator: it must name a team member")
    actors = members | ({OVERSEER} if kind == "trusted" else set())
    if not set(game["system_prompts"]).issubset(actors):
        raise ValueError(f"{name} has system prompts for unknown actors")
    leaders = dict(game["leaders"])
    if len(leaders) != len(game["leaders"]) or any(r < 1 for r in leaders):
        raise ValueError(f"{name} has ambiguous leader rounds")
    if not approval and (leaders or game["proposals"]):
        raise ValueError(f"{name} has leadership or proposals in an ungated arm")
    if kind != "elected" and game["elections"] or removal != "vote" and game["votes"]:
        raise ValueError(f"{name} has elections or ballots incompatible with its arm")
    elections = {e["round"]: e["leader"] for e in game["elections"]}
    if len(elections) != len(game["elections"]):
        raise ValueError(f"{name} has ambiguous election rounds")
    for election in game["elections"]:
        if (election["leader"] not in members or not set(election["votes"]).issubset(members)
                or any(v is not None and v not in members for v in election["votes"].values())
                or leaders.get(election["round"]) != election["leader"]):
            raise ValueError(f"{name} has an election naming invalid actors or leaders")
    for r, leader in leaders.items():
        expected = (OVERSEER if kind == "trusted" else world.get("dictator") if kind == "dictator" else
                    elections.get(max((e for e in elections if e <= r), default=0)))
        if leader not in actors or leader != expected:
            raise ValueError(f"{name} has a leader inconsistent with its arm or elections")
    for turn in game["turns"]:
        who, phase, r = turn["agent"], turn["phase"], turn["round"]
        if (who not in actors or r < 1 or approval and r not in leaders
                or phase == "review" and (not approval or who != leaders.get(r))
                or phase == "election" and kind != "elected"
                or phase == "vote" and removal != "vote"
                or phase != "review" and who not in members):
            raise ValueError(f"{name} has a turn incompatible with its actors or leadership")
    for phase in game["phases"]:
        if (not phase["turn_indices"]
                or any(game["turns"][i]["round"] != phase["round"] or game["turns"][i]["phase"] != phase["phase"]
                       for i in phase["turn_indices"])):
            raise ValueError(f"{name} has a phase referring to different turns")
    for kind_name in ("payments", "proposals", "invalid_payments"):
        for record in game[kind_name]:
            if record["by"] not in members:
                raise ValueError(f"{name} has a {kind_name} record naming a nonmember")
            if approval and record["round"] not in leaders:
                raise ValueError(f"{name} has a {kind_name} record without a recorded leader")
    for payment in game["payments"]:
        via, r = payment["via"], payment["round"]
        if (via not in ("direct", "approved", "leader") or (via == "direct") == approval
                or via == "leader" and (kind == "trusted" or payment["by"] != leaders.get(r))):
            raise ValueError(f"{name} has a payment incompatible with its approval mechanism")
    for vote in game["votes"]:
        if (not set(vote["ballots"]).issubset(members)
                or any(not set(names).issubset(members) for names in vote["ballots"].values())):
            raise ValueError(f"{name} has ballots naming nonmembers")
    for expelled in game["expulsions"]:
        if (expelled["name"] not in members or not removal
                or removal == "leader" and expelled["how"] != "leader"
                or removal == "vote" and not expelled["how"].startswith("vote ")):
            raise ValueError(f"{name} has an expulsion incompatible with its removal mechanism")


def _check_game_index(run, entry, game):
    """Check the selection/result identity even when reusing a cached file."""
    expected = {k: entry[k] for k in ("arm", "condition", "seed", "split", "scores")}
    expected.update(run_id=run.run_id, iteration=entry["iteration"])
    if (not isinstance(game.get("run_id"), str) or game.get("split") not in SPLITS
            or (game.get("split") == "dev" and not _is_integer(game.get("iteration")))
            or any(game.get(k) != value for k, value in expected.items())
            or "turns" in entry and len(game["turns"]) != entry["turns"]):
        raise ValueError(f"{entry['file']} has metadata or scores that do not match its index entry")
    _check_game_arm(run, game, entry["file"])


def read_game(run, entry):
    """One game, read when it is opened and held in a small cache keyed by file identity."""
    if run.scenario != "invoice_payments":
        return _read_scenario_game(run, entry)
    path = _inside(run.root, entry["file"])
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"{entry['file']} cannot be read: {exc.strerror or exc}") from exc
    try:
        with path.open("rb") as stream:
            compressed = stream.read(MAX_GAME_BYTES + 1)
        if len(compressed) > MAX_GAME_BYTES:
            raise ValueError("compressed game exceeds size limit")
    except OSError as exc:
        raise ValueError(f"{entry['file']} cannot be read: {exc}") from exc
    key = ("invoice_payments", str(path), hashlib.sha256(json.dumps(run.manifest, sort_keys=True).encode()).hexdigest(),
           hashlib.sha256(compressed).hexdigest())
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        held = _cache.get(key)
        if held is not None and held[0] == stamp:
            _check_game_index(run, entry, held[1])
            _cache.move_to_end(key)
            return held[1]
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
            raw = f.read(MAX_GAME_BYTES + 1)
        if len(raw) > MAX_GAME_BYTES:
            raise ValueError(f"{entry['file']} exceeds the 256 MB limit")
        game = json.loads(raw.decode("utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{entry['file']} is not a readable game: {exc}") from exc
    _check_game(game, entry["file"])
    _check_game_index(run, entry, game)
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
    if game.get("scenario") == "customer_support":
        from .adapters import adapter
        return adapter(game["scenario"]).conversation(game, turn_index, include_reply)
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


def _read_scenario_game(run, entry):
    from .adapters import adapter
    path = _inside(run.root, entry["file"])
    try:
        with path.open("rb") as stream:
            compressed = stream.read(MAX_GAME_BYTES + 1)
        if len(compressed) > MAX_GAME_BYTES:
            raise ValueError("compressed game exceeds size limit")
        content_hash = hashlib.sha256(compressed).hexdigest()
        if content_hash != entry["sha256"]:
            raise ValueError(f"{entry['file']}: content hash disagrees with manifest")
        manifest_hash = hashlib.sha256(json.dumps(run.manifest, sort_keys=True).encode()).hexdigest()
        key = (str(path), manifest_hash, content_hash)
        with _cache_lock:
            held = _cache.get(key)
            if held is not None:
                adapter(run.scenario).validate_game(run, entry, held)
                _cache.move_to_end(key)
                return held
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_GAME_BYTES + 1)
        if len(raw) > MAX_GAME_BYTES:
            raise ValueError("decompressed game exceeds size limit")
        game = json.loads(raw.decode("utf-8"))
        adapter(run.scenario).validate_game(run, entry, game)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{entry['file']} is not a readable game: {exc}") from exc
    with _cache_lock:
        _cache[key] = game
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHED_GAMES:
            _cache.popitem(last=False)
    return game
