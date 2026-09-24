"""Playing every game in a trial file, one after another, with nobody guessing by hand.

One game against one model says little: a contradiction may be bad luck, and
a clean game may have been easy. A trial file asks the same question many
times. Each trial is a game with its own seed, guessed by a fixed list or by
the letters most common among the words that still fit, and ended by asking
for the word if the model never reveals it. Each game is saved as an ordinary
saved game, and one row per game goes into a table beside them.

The reveal probe asks, after every response, what the word would be if the
player gave up there, several times at different seeds, with the answer's
first word written for the model. A model holding one word names the same
one each time and it fits the board; a model with nothing chosen names
whatever the board allows, and the names scatter. The probes are side
questions: none of them is part of the game that continues.

The model is held from the first game to the last, so every row describes the
same weights.
"""
from collections import Counter
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from uuid import uuid4

from extension_api import write_private_text
from .game import (
    GIVE_UP, HIDDEN, MAX_FILE_BYTES, OPENING, PROBE_PREFILL, SYSTEM, check, dictionary, finish_turn,
    fitting_words, guess_of, messages_for, new_game, probe_word, read_left, saved, word_problems,
)

logger = logging.getLogger(__name__)

TRIALS_FORMAT = "chatlab-hangman-trials-1"
FORMAT = "chatlab-hangman-batch-1"
SUMMARY_NAME, PROBES_NAME, MANIFEST_NAME = "summary.csv", "probes.csv", "batch.json"
DEFAULTS = dict(system=SYSTEM, opening=OPENING, temperature=1.0, max_new_tokens=2048, guesser="frequency",
                max_guesses=26, probe=None)
PROBE_DEFAULTS = dict(samples=5, max_new_tokens=16, temperature=1.0)
# The order a guesser falls back on when no word it knows fits the board.
ENGLISH = "etaoinshrdlcumwfgypbvkjxqz"
COLUMNS = ("trial_id", "label", "game_id", "outcome", "responses", "guesses", "unreadable_boards",
           "contradictions", "first_contradiction", "revealed_word", "revealed_at", "reveal_fits",
           "fitting_words", "probes", "probes_unreadable", "probe_fit_rate", "probe_reveal_rate",
           "sampled_tokens", "seconds", "model_id", "game_file", "detail")
PROBE_COLUMNS = ("trial_id", "response", "sample", "seed", "word", "fits", "matches_reveal")
PROGRESS_SECONDS = .5


def read_trials(path):
    """A trial file, checked whole, with every trial's settings filled in.

    ``defaults`` gives settings every trial shares and a trial may override
    any of them; what neither gives is the page's own default. A trial needs
    an ID and a seed.
    """
    path = Path(path)
    if path.stat().st_size > 8_000_000:
        raise ValueError("Trial files must be smaller than 8 MB.")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("That file is not a hangman trial file.") from exc
    if not isinstance(data, dict) or data.get("format") != TRIALS_FORMAT:
        raise ValueError(f"Choose a {TRIALS_FORMAT} file, not a saved game.")
    if set(data) - {"format", "title", "defaults", "trials"}:
        raise ValueError("A trial file holds only format, title, defaults and trials.")
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise ValueError("The trial file needs a title.")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - set(DEFAULTS):
        raise ValueError(f"Defaults may set only {', '.join(DEFAULTS)}.")
    items = data.get("trials")
    if not isinstance(items, list) or not 1 <= len(items) <= 2000:
        raise ValueError("A trial file must contain 1–2000 trials.")
    trials, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each trial must be an object.")
        name = item.get("id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each trial needs an id.")
        if name in seen:
            raise ValueError("Trial IDs must be unique.")
        seen.add(name)
        try:
            trials.append(_trial(item, defaults))
        except ValueError as exc:
            raise ValueError(f"Trial {name!r}: {exc}") from exc
    return dict(title=data["title"].strip(), trials=trials, file_sha256=hashlib.sha256(raw).hexdigest())


def _trial(item, defaults):
    if set(item) - {"id", "label", "seed", *DEFAULTS}:
        raise ValueError(f"A trial may set only id, label, seed and {', '.join(DEFAULTS)}.")
    trial = {**DEFAULTS, **defaults, **item}
    trial.setdefault("label", item["id"])
    if not isinstance(trial["label"], str) or not trial["label"].strip():
        raise ValueError("label must be text.")
    _integer_in(trial, "seed", 0, 2 ** 31 - 1)
    _integer_in(trial, "max_new_tokens", 1, 32768)
    _integer_in(trial, "max_guesses", 1, 200)
    _number_in(trial, "temperature", 0, 2)
    if not isinstance(trial["system"], str):
        raise ValueError("system must be text.")
    if not isinstance(trial["opening"], str) or not trial["opening"].strip():
        raise ValueError("opening must be text.")
    guesser = trial["guesser"]
    if guesser != "frequency" and not (
            isinstance(guesser, list) and 1 <= len(guesser) <= 200
            and all(isinstance(g, str) and g.strip() for g in guesser)):
        raise ValueError('guesser must be "frequency" or a list of 1–200 guesses.')
    probe = trial["probe"]
    if probe is not None:
        if not isinstance(probe, dict) or set(probe) - set(PROBE_DEFAULTS):
            raise ValueError(f"probe may set only {', '.join(PROBE_DEFAULTS)}.")
        probe = trial["probe"] = {**PROBE_DEFAULTS, **probe}
        _integer_in(probe, "samples", 1, 100)
        _integer_in(probe, "max_new_tokens", 1, 256)
        _number_in(probe, "temperature", 0, 2)
    return trial


def _integer_in(values, name, low, high):
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")


def _number_in(values, name, low, high):
    value = values.get(name)
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be a number from {low} to {high}.")


def next_guess(game, guesser, asked):
    """The guesser's next guess, or None when it has none left.

    A list is played in order. The frequency guesser picks the letter found in
    the most words that fit the latest board, among letters nobody has
    guessed, and falls back on English letter frequency when no word it knows
    fits: the board may be one no word fits, or the word list may be missing.
    """
    if guesser != "frequency":
        return guesser[asked] if asked < len(guesser) else None
    guessed = {value for kind, value in map(guess_of, (t["guess"] for t in game["turns"])) if kind == "letter"}
    counts = Counter(letter for word in fitting_words(game) or () for letter in set(word) if letter not in guessed)
    if counts:
        return min(counts, key=lambda letter: (-counts[letter], letter))
    return next((letter for letter in ENGLISH if letter not in guessed), None)


def ended(turn):
    """How the game stands after ``turn``: solved, lost, revealed, or None to play on.

    Revealed is a word written before the board was full or the guesses ran
    out, which the model was told to do only when asked.
    """
    if turn.get("board") and HIDDEN not in turn["board"]:
        return "solved"
    if read_left(turn.get("answer", "")) == 0:
        return "lost"
    return "revealed" if turn.get("revealed_word") else None


class BatchControl:
    """What the Stop button reaches while a batch runs, one per browser session."""

    def __init__(self):
        self.running = False
        self.stop_requested = False
        self.session = None

    def request_stop(self):
        """End the response generating now, recorded as stopped, and start no more."""
        self.stop_requested = True
        session = self.session
        if session is not None:
            session.cancel()


def generate(session, messages, sampling, answer_prefill=""):
    """Stream one response, closing the stream however the caller stops reading."""
    stream = session.generate(messages, answer_prefill=answer_prefill, **sampling)
    try:
        yield from stream
    finally:
        stream.close()


def finish_reason(update, session, control):
    last = update.metrics[-1]["token_id"] if update is not None and update.metrics else None
    return "stopped" if control.stop_requested else "stop" if last in session.stop_token_ids else "length"


def play(trial, session, control, save, stamp=None):
    """Play one trial's game to its end, yielding the game while it grows.

    Returns the game, how it ended, and the error that ended it if one did.
    ``save`` writes the game after every response, and may raise OSError.
    ``stamp`` is what the game records of the batch that played it.
    """
    game = new_game(trial["system"])
    game["trial"] = dict(stamp or {}, id=trial["id"], label=trial["label"], seed=trial["seed"])
    guess, asked, outcome = trial["opening"], 0, None
    while guess is not None:
        index = len(game["turns"])
        turn = dict(guess=guess, text="", metrics=[], forced_prefix_tokens=0, reasoning_prefilled=False,
                    finish_reason=None, started_at=time.time(), model_id=session.model_id, load_id=session.load_id,
                    sampling=dict(temperature=float(trial["temperature"]), top_p=1.0, top_k=0,
                                  max_new_tokens=trial["max_new_tokens"], seed=trial["seed"] + index))
        game["turns"].append(turn)
        update, failure = None, None
        try:
            stream = generate(session, messages_for(game, index), turn["sampling"])
            for update in stream:
                turn.update(text=update.text, metrics=update.metrics,
                            forced_prefix_tokens=update.forced_prefix_tokens,
                            reasoning_prefilled=update.reasoning_prefilled)
                yield game
        except Exception as exc:
            failure = exc
        turn["seconds"] = time.time() - turn.pop("started_at")
        if not turn["metrics"]:
            # Nothing was generated, so there is no response to keep: the
            # game ends at the one before it, as the page's does.
            game["turns"].pop()
            return game, "stopped" if failure is None else "error", failure
        turn["finish_reason"] = "error" if failure is not None else finish_reason(update, session, control)
        if failure is not None:
            turn["error"] = str(failure) or type(failure).__name__
        finish_turn(turn)
        if failure is None and not control.stop_requested and trial["probe"] and not _revealed(game):
            try:
                yield from probe(game, turn, trial["probe"], session, control)
            except Exception as exc:
                failure = exc
                turn["probe_error"] = str(exc) or type(exc).__name__
        save(game)
        if failure is not None or control.stop_requested:
            return game, "error" if failure is not None else "stopped", failure
        if outcome is not None:
            # The answer to the question that ended the game.
            return game, outcome, None
        state = ended(turn)
        if turn.get("revealed_word"):
            return game, state, None
        guess = None if state else next_guess(game, trial["guesser"], asked)
        asked += guess is not None
        if state is None and (guess is None or asked > trial["max_guesses"]):
            state = "unfinished"
        if state is not None:
            # Solved, lost, or out of guesses without the word: ask for it, so
            # every game ends with a word to hold the boards to.
            outcome, guess = state, GIVE_UP
    return game, outcome, None


def _revealed(game):
    return any(t.get("revealed_word") for t in game["turns"])


def probe(game, turn, settings, session, control):
    """Ask for the word after ``turn`` ``samples`` times, recording each answer on it."""
    turn["probes"] = []
    asking = dict(game, turns=game["turns"] + [dict(guess=GIVE_UP)])
    messages = messages_for(asking, len(game["turns"]))
    for sample in range(settings["samples"]):
        seed = 1000 * turn["sampling"]["seed"] + sample
        sampling = dict(temperature=float(settings["temperature"]), top_p=1.0, top_k=0,
                        max_new_tokens=settings["max_new_tokens"], seed=seed)
        update = None
        for update in generate(session, messages, sampling, answer_prefill=PROBE_PREFILL):
            yield game
        if update is None or control.stop_requested:
            return
        found = dict(seed=seed, text=update.text, reasoning_prefilled=update.reasoning_prefilled,
                     finish_reason=finish_reason(update, session, control))
        found["word"] = probe_word(found)
        turn["probes"].append(found)
        yield game


def summarize(trial, game, outcome, seconds, model_id, *, error=None, game_file=""):
    """One row of the summary table, and one row of the probe table per probe."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(trial_id=trial["id"], label=trial["label"], outcome=outcome, seconds=round(seconds, 1),
               model_id=model_id, game_file=game_file, detail="" if error is None else str(error))
    if game is None:
        return row, []
    turns = game["turns"]
    problems = check(game)
    revealed = next(((n, t["revealed_word"]) for n, t in enumerate(turns, 1) if t.get("revealed_word")), None)
    words = fitting_words(game) if dictionary() else None
    row.update(game_id=game["id"], responses=len(turns),
               guesses=" ".join(t["guess"] for t in turns[1:] if t["guess"] != GIVE_UP),
               unreadable_boards=sum(t.get("board") is None for t in turns),
               contradictions=len(problems), first_contradiction=problems[0][0] if problems else "",
               fitting_words="" if words is None else len(words),
               sampled_tokens=sum(len(t["metrics"]) - t.get("forced_prefix_tokens", 0) for t in turns))
    # A word fits a game with no readable board yet only because nothing can
    # contradict it, so it is left blank there rather than counted as fitting.
    drawn = next((n for n, t in enumerate(turns, 1) if t.get("board")), None)
    if revealed:
        row.update(revealed_at=revealed[0], revealed_word=revealed[1],
                   reveal_fits="" if drawn is None or drawn > revealed[0] else
                   not word_problems(dict(game, turns=turns[:revealed[0]]), revealed[1]))
    probes = []
    for number, turn in enumerate(turns, 1):
        before = dict(game, turns=turns[:number])
        for sample, found in enumerate(turn.get("probes", [])):
            word = found["word"]
            judged = word and drawn is not None and drawn <= number
            probes.append(dict(trial_id=trial["id"], response=number, sample=sample, seed=found["seed"],
                               word=word or "", fits=not word_problems(before, word) if judged else "",
                               matches_reveal="" if not word or not revealed else word == revealed[1]))
    readable = [p for p in probes if p["word"]]
    judged = [p for p in readable if p["fits"] != ""]
    row.update(probes=len(probes), probes_unreadable=len(probes) - len(readable))
    if judged:
        row["probe_fit_rate"] = round(sum(p["fits"] for p in judged) / len(judged), 3)
    if readable:
        if revealed:
            row["probe_reveal_rate"] = round(sum(p["matches_reveal"] for p in readable) / len(readable), 3)
    return row, probes


def run_trials(data, models, root, control, *, source=""):
    """Play every trial in ``data`` in order, yielding progress as it goes.

    Each yield is ``(done, total, rows, directory, game)``: the trials
    finished, how many there are, their summary rows, where the batch is
    written, and the game being played now or None. A game whose generation
    fails is recorded as an error and the batch moves on. A stop ends the
    response generating now and plays no more. A game or summary that cannot
    be written ends the batch with an OSError, and the manifest then says the
    batch failed and why.
    """
    session = models.open_session()
    try:
        directory = batch_directory(root, data["title"])
    except BaseException:
        session.close()
        raise
    control.running, control.stop_requested, control.session = True, False, session
    trials, games = data["trials"], directory / "games"
    manifest = dict(format=FORMAT, title=data["title"], source=source, file_sha256=data["file_sha256"],
                    model_id=session.model_id, load_id=session.load_id,
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), finished_at=None, status="running",
                    total=len(trials))
    rows, probes, written, completed = [], [], set(), False
    logger.info("Hangman batch %s: playing %s games from %s with %s", directory.name, len(trials), data["title"],
                session.model_id)

    def save(game):
        text = saved(game)
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise OSError("the game is larger than a saved game can be opened at")
        games.mkdir(exist_ok=True)
        write_private_text(games / f"{game['id']}.json", text)
        written.add(game["id"])

    try:
        write_summary(directory, manifest, rows, probes)
        yield 0, len(trials), rows, directory, None
        for trial in trials:
            if control.stop_requested:
                break
            # Stopped unless the game says otherwise: a window closing mid-game
            # ends the trial where it stands.
            started, game, outcome, error, unsaved = time.time(), None, "stopped", None, None
            stamp = dict(title=data["title"], file_sha256=data["file_sha256"], batch=directory.name)
            try:
                reported, frames = 0., play(trial, session, control, save, stamp)
                while True:
                    try:
                        game = next(frames)
                    except StopIteration as done:
                        game, outcome, error = done.value
                        break
                    if time.monotonic() - reported >= PROGRESS_SECONDS:
                        reported = time.monotonic()
                        yield len(rows), len(trials), rows, directory, game
            except OSError as exc:
                outcome, error, unsaved = "unsaved", exc, exc
            finally:
                frames.close()
                if game is not None and game["turns"] and game["turns"][-1]["finish_reason"] is None:
                    # Closed mid-response: the row describes the game as saved.
                    game = dict(game, turns=game["turns"][:-1])
                saved_file = game is not None and game["id"] in written and not unsaved
                row, found = summarize(trial, game, outcome, time.time() - started, session.model_id, error=error,
                                       game_file=f"games/{game['id']}.json" if saved_file else "")
                rows.append(row)
                probes.extend(found)
            logger.info("Hangman batch %s: game %s of %s, %r, %s", directory.name, len(rows), len(trials),
                        trial["id"], outcome)
            if unsaved is not None:
                raise OSError(f"The game for trial {trial['id']!r} could not be saved: {unsaved}")
            write_summary(directory, manifest, rows, probes)
            yield len(rows), len(trials), rows, directory, None
        manifest["status"] = "stopped" if cut_short(rows, len(trials)) else "finished"
        completed = True
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    except BaseException:
        # The window closed or the process is going down: nothing went wrong
        # with the batch itself, so it reads as stopped unless it was done.
        manifest["status"] = "stopped" if cut_short(rows, len(trials)) else "finished"
        raise
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        control.running, control.session = False, None
        session.close()
        try:
            write_summary(directory, manifest, rows, probes)
        except OSError as exc:
            logger.warning("Hangman batch %s: could not write its summary: %s", directory.name, exc)
            if completed:
                raise
        logger.info("Hangman batch %s %s after %s of %s games", directory.name, manifest["status"], len(rows),
                    len(trials))


def cut_short(rows, total):
    """Whether a batch ended before every game ran to its own end, read off the rows."""
    return len(rows) < total or any(row["outcome"] == "stopped" for row in rows)


def batch_directory(root, title):
    """A new directory for one batch, named for when it started and its title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "trials"
    directory = Path(root) / "batches" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    suffix, candidate = 1, directory
    while candidate.exists():
        suffix += 1
        candidate = directory.with_name(f"{directory.name}-{suffix}")
    candidate.mkdir(parents=True)
    return candidate


def table(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_summary(directory, manifest, rows, probes):
    """Rewrite the tables and the manifest whole after every game, so a stopped
    batch leaves a complete record of what it played."""
    replace_text(directory / SUMMARY_NAME, table(COLUMNS, rows), newline="")
    replace_text(directory / PROBES_NAME, table(PROBE_COLUMNS, probes), newline="")
    replace_text(directory / MANIFEST_NAME,
                 json.dumps(dict(manifest, results=rows), ensure_ascii=False, indent=1) + "\n")


def replace_text(path, text, *, newline=None):
    """Write ``text`` beside ``path`` and move it into place once it is whole,
    so a write failing on a full disk leaves the last good copy."""
    staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_private_text(staged, text, newline=newline)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def downloads(directory):
    """Copies of the tables and the manifest where the interface is allowed to serve them."""
    staged = Path(tempfile.mkdtemp(prefix="chatlab-hangman-batch-"))
    return [str(shutil.copy(Path(directory) / name, staged / name))
            for name in (SUMMARY_NAME, PROBES_NAME, MANIFEST_NAME) if (Path(directory) / name).exists()]
