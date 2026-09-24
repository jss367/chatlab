"""Hangman hosted by the model: prompts, board reading and consistency checks.

Nothing here knows the word. The model is the only place a word could be held,
so every check reads what the model wrote and asks whether some single word
could have produced all of it.
"""
import copy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from uuid import uuid4

FORMAT = "chatlab-hangman-1"
# A saved token costs about 2.4 KB with its eight alternatives, so one
# 32768-token response is about 75 MB. This allows a dozen of those, or a
# couple of hundred at the default length; the page refuses to save a game
# past it rather than offer a file it could not open again.
MAX_FILE_BYTES = 1024 ** 3
WORD_LIST = Path("/usr/share/dict/words")

SYSTEM = """You are hosting a game of hangman. Think of one secret English word and keep it to yourself for the whole game. Do not write the word until the game ends.

The player guesses one letter at a time, or the whole word. After every guess, reply in exactly this format:

Board: the word, one character per letter separated by spaces, with each correctly guessed letter shown and each hidden letter written as _
Guessed: every letter guessed so far
Wrong guesses left: a number, starting at 6

Then say in one sentence whether the guess was in the word. When the player guesses the word, runs out of wrong guesses, or asks for the word, reveal it on its own line as:
Word: the secret word"""

OPENING = "Let's play. Think of your word and show me the empty board."
# What a batch says to end a game that never revealed its word, and what the
# reveal probe asks before it writes the answer's first word for the model.
GIVE_UP = "I give up. What was the word?"
PROBE_PREFILL = "Word:"

BOARD_LINE = re.compile(r"^[\s>*_`#-]*board[\s*_`]*:(.*)$", re.I | re.M)
WORD_LINE = re.compile(r"^[\s>*_`#-]*word[\s*_`]*:[\s*_`]*([A-Za-z]+)[\s*_`.!]*$", re.I | re.M)
LEFT_LINE = re.compile(r"^[\s>*_`#-]*wrong guesses left[\s*_`]*:[\s*_`]*(\d+)", re.I | re.M)
HIDDEN = "_"


def now():
    return datetime.now(timezone.utc).isoformat()


def new_game(system):
    """Sampling is recorded per response, since the controls can move mid-game."""
    return dict(format=FORMAT, id=uuid4().hex, created_at=now(), parent=None, system=system, turns=[])


def answer_of(text, reasoning_prefilled=False):
    """The visible answer: what follows the reasoning block, if there is one."""
    if reasoning_prefilled:
        text = "<think>" + text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("<think>", 1)[0].strip()


def reasoning_of(text, reasoning_prefilled=False):
    """The text of every reasoning block, an unclosed last one included."""
    if reasoning_prefilled:
        text = "<think>" + text
    return "\n\n".join(block.strip() for block in re.findall(r"<think>(.*?)(?:</think>|$)", text, flags=re.S)
                       if block.strip())


def history_content(turn):
    """What the next prompt says this response was, reasoning included.

    The template decides whether earlier reasoning survives into the next
    prompt; many drop it. Whatever the model worked out there, a word included,
    is then gone by the next turn, which is exactly what the context view shows.
    """
    return ("<think>" + turn["text"]) if turn.get("reasoning_prefilled") else turn["text"]


def messages_for(game, turn_index):
    """The conversation the model answers when it produces turn ``turn_index``."""
    messages = [{"role": "system", "content": game["system"]}] if game["system"].strip() else []
    for turn in game["turns"][:turn_index]:
        messages.append({"role": "user", "content": turn["guess"]})
        messages.append({"role": "assistant", "content": history_content(turn)})
    messages.append({"role": "user", "content": game["turns"][turn_index]["guess"]})
    return messages


def read_board(answer):
    """The last board line as a list of letters and hidden cells, or None.

    Cells may be separated by spaces or run together. Separated by spaces, a
    run of underscores is one blank, as some models draw them. Anything other
    than a letter or an underscore makes the line unreadable rather than
    guessed at.
    """
    lines = BOARD_LINE.findall(answer)
    if not lines:
        return None
    raw = lines[-1].replace("\\_", HIDDEN).strip(" \t*`")
    parts = raw.split()
    board = []
    for cell in parts if len(parts) > 1 else list(raw):
        if re.fullmatch(r"_+", cell):
            board.append(HIDDEN)
        elif len(cell) == 1 and cell.isalpha():
            board.append(cell.lower())
        else:
            return None
    return board or None


def read_word(answer):
    found = WORD_LINE.findall(answer)
    return found[-1].lower() if found else None


def read_left(answer):
    """The wrong guesses left on the reply's last such line, or None.

    A model can run on into thousands of digits there, and Python refuses to
    convert a number that long, so a count no game could reach reads as none.
    """
    found = LEFT_LINE.findall(answer)
    return int(found[-1]) if found and len(found[-1]) <= 9 else None


def probe_word(probe):
    """The word a reveal probe wrote, or None when it wrote none it finished.

    The probe's answer starts at ``Word:``, so only its first line is read: a
    model that goes on to a second ``Word:`` line has not answered once. A
    probe cut off by its token limit on that line may have been cut off mid-word.
    """
    answer = answer_of(probe["text"], probe.get("reasoning_prefilled", False))
    line, *rest = answer.split("\n", 1)
    if probe.get("finish_reason") == "length" and not rest:
        return None
    return read_word(line)


def guess_of(text):
    """("letter", "e"), ("word", "apple") or ("other", None)."""
    value = text.strip().strip(".!?\"'").strip()
    if len(value) == 1 and value.isalpha():
        return "letter", value.lower()
    if value.isalpha():
        return "word", value.lower()
    return "other", None


def finish_turn(turn):
    answer = answer_of(turn["text"], turn.get("reasoning_prefilled", False))
    turn.update(answer=answer, board=read_board(answer), revealed_word=read_word(answer))
    return turn


def check(game):
    """Every place the boards so far contradict each other or the guesses.

    Returns a list of (turn number, message), one-based to match the page.
    Letters are held to the first readable board of the game's length drawn
    after they were guessed: that board fixes where the letter is, or that it
    is absent, and every later board and the revealed word have to agree. The
    first readable board sets the length; one of another length is reported
    as a change of size and checked only for letters nobody guessed. The
    first revealed word is held to every board and word that follows it.
    """
    problems, _ = _walk(game)
    # A word revealed again on every later turn contradicts the boards the
    # same way each time; the first turn that said so is the one to read.
    seen = set()
    return [(number, message) for number, message in problems
            if message not in seen and not seen.add(message)]


def word_problems(game, word):
    """How ``word`` contradicts what the game's boards and revealed word say.

    Empty when the word could be the one the game has been describing. A word
    is held to the boards exactly as a revealed word is, and to the first word
    the game revealed.
    """
    _, (length, cells, placed, first) = _walk(game)
    word = word.lower()
    found = [f"The game revealed {first[1].upper()} at response {first[0]}; this is {word.upper()}."] \
        if first and word != first[1] else []
    return found + list(_word_problems(word, length, cells, placed))


def _walk(game):
    """The contradictions ``check`` reports, before it drops repeats, and the
    board state they leave: the length, every letter each length of board
    showed at each position, where each guessed letter was placed, and the
    first word revealed.

    A word is held to every board, not only to the latest one: a board that
    later hides a cell again, or is drawn at another length, contradicts the
    one before it, and a word agreeing with the later board still disagrees
    with the earlier one.
    """
    problems = []
    length, shown, placed, cells = None, {}, {}, {}
    guessed, moved, pending = set(), set(), set()
    first = None
    for number, turn in enumerate(game["turns"], 1):
        kind, value = guess_of(turn["guess"])
        board = turn.get("board")
        if kind == "letter":
            if value not in guessed:
                pending.add(value)
            guessed.add(value)
        elif kind == "word" and value in ("".join(board or ()), turn.get("revealed_word")):
            # A right word guess, confirmed by the board or by a Word: line,
            # reveals letters nobody guessed one at a time, and says where
            # each one is: the word is on the board from here on, whether or
            # not this reply drew a board to show it.
            guessed.update(value)
            if length is None or len(value) == length:
                for letter in dict.fromkeys(value):
                    if letter not in placed:
                        placed[letter] = {i for i, character in enumerate(value) if character == letter}
                        pending.discard(letter)
        if board is not None:
            drawn = cells.setdefault(len(board), {})
            for position, cell in enumerate(board):
                if cell != HIDDEN:
                    drawn.setdefault(position, {})[cell] = None
            if length is not None and len(board) != length:
                problems.append((number, f"The board went from {length} letters to {len(board)}."))
                # Its positions line up with no other board, so it neither
                # moves letters nor places the pending ones; what it shows
                # still has to have been guessed.
                problems.extend((number, f"{cell.upper()} is on the board but was never guessed.")
                                for cell in dict.fromkeys(board) if cell != HIDDEN and cell not in guessed)
            else:
                length = len(board)
                for position, cell in enumerate(board):
                    before = shown.get(position)
                    if before and cell != before:
                        problems.append((number, f"Position {position + 1} showed {before.upper()} "
                                                 f"and now shows {'nothing' if cell == HIDDEN else cell.upper()}."))
                    if cell == HIDDEN:
                        # Reported once; the placement check keeps the letter to its place.
                        shown.pop(position, None)
                    else:
                        shown[position] = cell
                        if cell != before and cell not in guessed:
                            problems.append((number, f"{cell.upper()} is on the board but was never guessed."))
                for letter, positions in placed.items():
                    now_at = {i for i, cell in enumerate(board) if cell == letter}
                    if now_at != positions and (letter, frozenset(now_at)) not in moved:
                        moved.add((letter, frozenset(now_at)))
                        problems.append((number, f"{letter.upper()} was placed at {_positions(positions)} "
                                                 f"and is now at {_positions(now_at)}."))
                placed.update((letter, {i for i, cell in enumerate(board) if cell == letter})
                              for letter in pending)
                pending.clear()
        # The first word revealed stays the word: the game can go on after it,
        # and every later board and word has to agree with it too.
        word = turn.get("revealed_word")
        if word and first and word != first[1]:
            problems.append((number, f"The word revealed at response {first[0]} was {first[1].upper()}; "
                                     f"this one is {word.upper()}."))
        for held in dict.fromkeys(w for w in (first and first[1], word) if w):
            problems.extend((number, message) for message in _word_problems(held, length, cells, placed))
        if word and not first:
            first = (number, word)
    return problems, (length, cells, placed, first)


def _positions(positions):
    return ", ".join(str(i + 1) for i in sorted(positions)) if positions else "no position"


def _word_problems(word, length, cells, placed):
    for size, shown in cells.items():
        if len(word) != size:
            yield f"The revealed word {word.upper()} has {len(word)} letters; the board had {size}."
            continue
        for position, letters in shown.items():
            for letter in letters:
                if word[position] != letter:
                    yield (f"The revealed word {word.upper()} has {word[position].upper()} at position "
                           f"{position + 1}, where the board showed {letter.upper()}.")
    if length is not None and len(word) != length:
        return
    for letter, positions in placed.items():
        actual = {i for i, character in enumerate(word) if character == letter}
        if actual != positions:
            yield (f"The revealed word {word.upper()} has {letter.upper()} at {_positions(actual)}; "
                   f"the board placed it at {_positions(positions)}.")


@lru_cache(maxsize=1)
def dictionary():
    """Lowercase words from the system list, or an empty tuple without one.

    Capitalized entries are proper nouns there, which hangman does not use.
    """
    try:
        return tuple(sorted({w for w in WORD_LIST.read_text(errors="ignore").split()
                             if w.isalpha() and w.islower() and w.isascii()}))
    except OSError:
        return ()


def latest_board(game):
    """The latest readable board and the letters guessed by the turn that drew it.

    A letter guessed after that turn has no board saying where it went yet, so
    it says nothing about this one. (None, set()) when no board is readable.
    """
    for index in range(len(game["turns"]) - 1, -1, -1):
        board = game["turns"][index].get("board")
        if board:
            return board, {value for kind, value in (guess_of(t["guess"]) for t in game["turns"][:index + 1])
                           if kind == "letter"}
    return None, set()


def fitting_words(game, words=None):
    """Words that fit the latest readable board and the letters guessed by then.

    A hidden cell cannot hold a letter already guessed: had it been there, the
    board would show it. None when there is no board to fit.
    """
    board, guessed = latest_board(game)
    if board is None:
        return None
    pattern = re.compile("".join(re.escape(cell) if cell != HIDDEN else
                                 (f"[^{''.join(sorted(guessed))}]" if guessed else ".")
                                 for cell in board) + "$")
    return [word for word in (dictionary() if words is None else words)
            if len(word) == len(board) and pattern.match(word)]


def saved(game):
    return json.dumps(game, ensure_ascii=False, indent=2)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _readable_metric(metric):
    """Whether ``metric`` holds every field this extension reads, as the type it reads.

    That is the token ID replayed on a branch and, per alternative, the ID,
    text and probability the menu offers. What the runtime's own panel reads
    is checked by rendering it, on opening, not repeated here.
    """
    if not isinstance(metric, dict) or not _integer(metric.get("token_id")):
        return False
    candidates = metric.get("top_candidates", [])
    return isinstance(candidates, list) and all(
        isinstance(c, dict) and _integer(c.get("token_id")) and isinstance(c.get("text"), str)
        and _probability(c.get("probability")) for c in candidates)


def _readable_probes(probes):
    return isinstance(probes, list) and all(
        isinstance(p, dict) and isinstance(p.get("text"), str) and _integer(p.get("seed"))
        and isinstance(p.get("reasoning_prefilled", False), bool)
        and isinstance(p.get("finish_reason"), (str, type(None))) for p in probes)


def _probability(value):
    """A finite number. JSON reads ``1e309`` as infinity and ``NaN`` as NaN, and
    the token menu serializes either into markup its script cannot parse."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("That file is too large to be a saved hangman game.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("That file is not a saved hangman game.") from exc
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"That file is not a {FORMAT} game.")
    turns = value.get("turns")
    if (not isinstance(turns, list) or not isinstance(value.get("system"), str)
            or not all(isinstance(t, dict) and isinstance(t.get("guess"), str) and isinstance(t.get("text"), str)
                       for t in turns)):
        raise ValueError("The saved game is missing its prompt or its turns.")
    if not isinstance(value.get("id"), str):
        raise ValueError("The saved game is missing its id.")
    # Opening the game renders these, so a wrong type has to be refused here.
    for number, turn in enumerate(turns, 1):
        metrics = turn.get("metrics", [])
        if (not isinstance(metrics, list) or not all(map(_readable_metric, metrics))
                or not isinstance(turn.get("sampling", {}), dict)
                or not isinstance(turn.get("branch") or {}, dict)
                or not isinstance(turn.get("forced_prefix_tokens", 0), int)
                or not _readable_probes(turn.get("probes", []))):
            raise ValueError(f"Response {number} of the saved game is malformed.")
    for turn in turns:
        turn.setdefault("metrics", [])
        finish_turn(turn)
        # Read again from the text, as the board is, so the note shows what
        # the probe wrote rather than what the file says it wrote.
        for probe in turn.get("probes", []):
            probe["word"] = probe_word(probe)
    return value


def rewound(game, turn_count):
    """A new game holding the first ``turn_count`` turns, pointing at its parent."""
    child = copy.deepcopy(game)
    child.update(id=uuid4().hex, created_at=now(), turns=child["turns"][:turn_count],
                 parent=dict(id=game["id"], turns=turn_count))
    return child


def reopened(game):
    """A game just read by ``load`` as a new child of the file it came from.

    Unlike ``rewound`` nothing is copied: ``load`` built the game fresh, and a
    copy of a large save would double what opening it holds in memory.

    Each response keeps the load that wrote it as ``recorded_load_id`` and
    loses it as ``load_id``, which is what branching compares with the load in
    memory. A load ID counts loads within one process, so after a restart the
    first load of any model is named the same as the first load before it,
    and a save's token IDs could be replayed through weights that never
    produced them. A reopened response is never the session's own load, so
    branching refuses it and rewinding, which replays text, still works.
    """
    game.update(id=uuid4().hex, created_at=now(), parent=dict(id=game["id"], turns=len(game["turns"])))
    for turn in game["turns"]:
        if turn.get("load_id") is not None:
            turn["recorded_load_id"] = turn["load_id"]
        turn["load_id"] = None
    return game
