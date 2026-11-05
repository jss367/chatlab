"""Twenty questions hosted by the model: prompts, reading replies and consistency checks.

Nothing here knows the thing. The model is the only place a thing could be
held, so every check reads what the model wrote and asks whether one thing
could have produced all of it.
"""
import json
from pathlib import Path
import re
from uuid import uuid4

from chatlab.extensions.hangman.game import (
    _readable_branch, _readable_metric, _readable_probes, answer_of, history_content, now, reasoning_of,
)

FORMAT = "chatlab-twenty-questions-1"
# As for hangman: past this a saved game could not be opened again.
MAX_FILE_BYTES = 1024 ** 3
QUESTIONS = 20

SYSTEM = """You are hosting a game of twenty questions. Think of one specific thing, such as an object, an animal, a place or a famous person, and keep it to yourself for the whole game. Do not name it until the game ends.

The player asks questions that can be answered yes or no, or guesses what the thing is. Reply to every question in exactly this format:

Answer: yes, no, or sometimes
Questions left: a number, starting at 20

When the player guesses the thing, runs out of questions, or asks what it was, name it on its own line as:
Thing: the thing"""

OPENING = "Let's play. Think of your thing and tell me when you're ready."
# What the reveal probe asks before it writes the answer's first word for the model.
GIVE_UP = "I give up. What was it?"
PROBE_PREFILL = "Thing:"
# The recheck asks about one candidate thing in a fresh conversation, with
# nothing of the game in it, and writes the start of the answer.
RECHECK = "Think about this: {thing}.\n\nAnswer this question about it with yes, no or sometimes.\n\nQuestion: {question}"
RECHECK_PREFILL = "Answer:"

ANSWER_LINE = re.compile(r"^[\s>*_`#-]*answer[\s*_`]*:[\s*_`]*([A-Za-z]+)", re.I | re.M)
LEFT_LINE = re.compile(r"^[\s>*_`#-]*questions left[\s*_`]*:[\s*_`]*(\d+)", re.I | re.M)
THING_LINE = re.compile(r"^[\s>*_`#-]*thing[\s*_`]*:[\s*_`]*(.+?)[\s*_`.!]*$", re.I | re.M)
VERDICTS = {"yes": "yes", "no": "no", "sometimes": "sometimes", "partly": "sometimes", "maybe": "sometimes"}


def new_game(system):
    """Sampling is recorded per response, since the controls can move mid-game."""
    return dict(format=FORMAT, id=uuid4().hex, created_at=now(), parent=None, system=system, turns=[])


def messages_for(game, turn_index):
    """The conversation the model answers when it produces turn ``turn_index``."""
    messages = [{"role": "system", "content": game["system"]}] if game["system"].strip() else []
    for turn in game["turns"][:turn_index]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": history_content(turn)})
    messages.append({"role": "user", "content": game["turns"][turn_index]["question"]})
    return messages


def read_verdict(answer):
    """``yes``, ``no`` or ``sometimes`` from the reply's last answer line, or None."""
    found = ANSWER_LINE.findall(answer)
    return VERDICTS.get(found[-1].lower()) if found else None


def verdict_span(answer):
    """Where the last answer line's verdict word starts in ``answer``, or None."""
    found = list(ANSWER_LINE.finditer(answer))
    return found[-1].start(1) if found and VERDICTS.get(found[-1].group(1).lower()) else None


def read_left(answer):
    """The questions left on the reply's last such line, or None; an absurd count reads as none."""
    found = LEFT_LINE.findall(answer)
    return int(found[-1]) if found and len(found[-1]) <= 9 else None


def read_thing(answer):
    """The thing on the reply's last ``Thing:`` line, lowercased and without an article."""
    found = THING_LINE.findall(answer)
    if not found:
        return None
    thing = re.sub(r"[*_`\"]", "", found[-1]).strip().lower()
    thing = re.sub(r"^(a|an|the)\s+", "", thing)
    return thing or None


def probe_thing(probe):
    """The thing a reveal probe named, or None when it named none it finished.

    Only the first line is read: the answer starts at ``Thing:``. A probe cut
    off by its token limit on that line may have been cut off mid-name.
    """
    answer = answer_of(probe["text"], probe.get("reasoning_prefilled", False))
    line, *rest = answer.split("\n", 1)
    if probe.get("finish_reason") == "length" and not rest:
        return None
    return read_thing(line)


def same_question(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def finish_turn(turn):
    answer = answer_of(turn["text"], turn.get("reasoning_prefilled", False))
    turn.update(answer=answer, verdict=read_verdict(answer), left=read_left(answer), revealed=read_thing(answer))
    return turn


def check(game):
    """Every place the replies so far contradict each other.

    Returns a list of (turn number, message), one-based to match the page.
    The opening is turn 1 and asks nothing, so the count of questions left is
    held to drop by one for each turn after the last turn that gave a count.
    A question asked again, word for word, should get the same answer. The
    first thing named is held against every later one.
    """
    problems = []
    counted, answered, first = None, {}, None
    for number, turn in enumerate(game["turns"], 1):
        left = turn.get("left")
        if left is not None:
            if counted is not None:
                expected = counted[1] - (number - counted[0])
                if left != expected:
                    problems.append((number, f"Questions left went from {counted[1]} at response {counted[0]} "
                                             f"to {left}; {expected} would follow."))
            counted = (number, left)
        verdict = turn.get("verdict")
        if number > 1 and verdict in ("yes", "no"):
            key = same_question(turn["question"])
            earlier = answered.get(key)
            if earlier and earlier[1] != verdict:
                problems.append((number, f"Response {earlier[0]} answered {earlier[1].upper()} to the same "
                                         f"question; this one answers {verdict.upper()}."))
            answered.setdefault(key, (number, verdict))
        thing = turn.get("revealed")
        if thing and first and thing != first[1]:
            problems.append((number, f"The thing named at response {first[0]} was {first[1].upper()}; "
                                     f"this one is {thing.upper()}."))
        if thing and not first:
            first = (number, thing)
    return problems


def revealed(game):
    """The first thing the game named, or None."""
    return next((t["revealed"] for t in game["turns"] if t.get("revealed")), None)


def asked(game):
    """(turn index, question, verdict) for every question the host answered yes or no."""
    return [(index, turn["question"], turn["verdict"]) for index, turn in enumerate(game["turns"])
            if index and turn.get("verdict") in ("yes", "no")]


def recheck_messages(thing, question):
    return [{"role": "user", "content": RECHECK.format(thing=thing, question=question)}]


def recheck_summary(recheck):
    """How many rechecked answers agree with the game's, and the ones that do not."""
    rows = recheck["rows"]
    read = [row for row in rows if row["verdict"] is not None]
    disagree = [row for row in read if row["verdict"] != row["game_verdict"]]
    return len(read) - len(disagree), len(read), disagree


def saved(game):
    return json.dumps(game, ensure_ascii=False, indent=2)


def _readable_jspace(value):
    """No reading, or one the J space pane can render: positions holding blocks of top tokens."""
    if value is None:
        return True
    if not isinstance(value, dict) or not isinstance(value.get("positions"), list):
        return False
    for position in value["positions"]:
        if not isinstance(position, dict) or not isinstance(position.get("blocks"), list):
            return False
        for block in position["blocks"]:
            if not (isinstance(block, dict) and type(block.get("block")) is int and isinstance(block.get("top"), list)
                    and all(isinstance(t, dict) and isinstance(t.get("text"), str)
                            and isinstance(t.get("log_prob"), (int, float)) for t in block["top"])):
                return False
    watch = value.get("watch", {})
    return isinstance(watch, dict) and all(
        isinstance(k, str) and isinstance(v, list) and all(isinstance(x, (int, float, type(None))) for x in v)
        for k, v in watch.items())


def _readable_recheck(value):
    if value is None:
        return True
    return (isinstance(value, dict) and isinstance(value.get("thing"), str) and isinstance(value.get("rows"), list)
            and all(isinstance(r, dict) and type(r.get("turn")) is int and isinstance(r.get("question"), str)
                    and r.get("game_verdict") in ("yes", "no") and r.get("verdict") in ("yes", "no", "sometimes", None)
                    and isinstance(r.get("text"), str) for r in value["rows"]))


def load(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("That file is too large to be a saved twenty questions game.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("That file is not a saved twenty questions game.") from exc
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"That file is not a {FORMAT} game.")
    turns = value.get("turns")
    if (not isinstance(turns, list) or not isinstance(value.get("system"), str)
            or not all(isinstance(t, dict) and isinstance(t.get("question"), str) and isinstance(t.get("text"), str)
                       for t in turns)):
        raise ValueError("The saved game is missing its prompt or its turns.")
    if not isinstance(value.get("id"), str):
        raise ValueError("The saved game is missing its id.")
    if not _readable_recheck(value.get("recheck")):
        raise ValueError("The saved game's recheck is malformed.")
    for number, turn in enumerate(turns, 1):
        metrics = turn.get("metrics", [])
        if (not isinstance(metrics, list) or not all(map(_readable_metric, metrics))
                or not isinstance(turn.get("sampling", {}), dict)
                or not _readable_branch(turn.get("branch"))
                or not isinstance(turn.get("forced_prefix_tokens", 0), int)
                or not _readable_probes(turn.get("probes", []))
                or not _readable_jspace(turn.get("jspace"))):
            raise ValueError(f"Response {number} of the saved game is malformed.")
    for turn in turns:
        turn.setdefault("metrics", [])
        finish_turn(turn)
        for probe in turn.get("probes", []):
            probe["thing"] = probe_thing(probe)
    return value


__all__ = [
    "FORMAT", "GIVE_UP", "MAX_FILE_BYTES", "OPENING", "PROBE_PREFILL", "QUESTIONS", "RECHECK_PREFILL", "SYSTEM",
    "answer_of", "asked", "check", "finish_turn", "load", "messages_for", "new_game", "probe_thing",
    "read_left", "read_thing", "read_verdict", "reasoning_of", "recheck_messages", "recheck_summary", "revealed",
    "saved", "verdict_span",
]
