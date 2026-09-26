"""Reasoning check: whether what a response says predicts the move it makes, and whether it causes it.

Two measures, each read off saved runs so a single agent and a team can be
compared on the same terms.

Stated against taken reads the prose a response wrote before its move call,
finds the direction it committed to, and compares that with the direction the
call names. It needs no model: every saved run already holds both. It reads
phrases such as "I will move east" or "go to (1, 2)" and nothing subtler, so
every row names the phrase it read, for the reader to check.

The truncation test asks whether the reasoning did any work. The response is
cut after a fraction of its reasoning, the model is made to write its move
call straight after the cut, and the probability it gives each direction is
read at the point the call names one. Reasoning the move depends on raises the
probability of the move the response made as more of it is kept. Reasoning
written around a move already decided leaves it where it was with none kept.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import threading
from dataclasses import dataclass, field

from .maze import DIRECTIONS, TOOLS, parse_call
from .runner import context_messages, from_payload as single_from_payload, visible_token_ids
from .team import FORMAT as TEAM_FORMAT, MESSAGE_LIMIT, from_payload as team_from_payload

logger = logging.getLogger(__name__)

# How much of the reasoning each truncation keeps, by words.
FRACTIONS = (0., .25, .5, .75, 1.)
# A message is kept when its agent calls the direction it named within this
# many of its own calls, the call carrying the message included. A call the
# simulator rejects still counts: like stated against taken, this compares what
# the agent said with what it chose, not with where the maze let it go.
FOLLOW_WINDOW = 3

SYNONYMS = {"up": "north", "down": "south", "left": "west", "right": "east"}
_WORD = r"(north|east|south|west|up|down|left|right)(?:wards?)?"
_VERB = (r"(?:move|go|head|step|travel|proceed|continue|try|turn|walk|moving|going|heading|stepping|"
         r"travell?ing|proceeding|continuing|trying|turning|walking)")
_FILLER = (r"(?:\s+(?:to|the|one|a|single|cell|cells|step|steps|square|space|back|further|again|toward|towards|"
           r"in|direction|of|straight|once|more|another|position))*")
COMMITMENTS = [
    # "move east", "go one step to the north", "heading back up"
    re.compile(rf"\b{_VERB}\b{_FILLER}\s+{_WORD}\b"),
    # "take the east corridor", "explore the southern side": how teammates
    # divide a maze in their messages.
    re.compile(r"\b(?:take|taking|explore|exploring|cover|covering|search|searching)\s+"
               r"(?:the\s+)?(north|east|south|west)(?:ern)?\b"),
    # "the best move is east", "my next step will be north". A bare "the
    # direction is east" often describes the map rather than choosing.
    re.compile(rf"\b(?:best|next|right|correct|logical|optimal|only)\s+(?:move|option|step|choice|action|direction)"
               rf"\s+(?:is|should be|will be|would be)\s+"
               rf"(?:to\s+)?(?:(?:go|move|head)\s+)?{_WORD}\b"),
]
# "move to (1, 2)": a cell next to the agent names the direction to it.
COORDINATE = re.compile(rf"\b{_VERB}\b\s+(?:to|into|toward|towards)\s+(?:the\s+)?(?:cell|position|square)?\s*"
                        r"[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]")
# A commitment inside the clause after one of these is not one: a refusal, a
# condition, a possibility, one option of several, or a step for later.
HEDGES = re.compile(r"\b(?:not|never|no|cannot|can't|cant|couldn't|shouldn't|won't|wouldn't|don't|avoid|"
                    r"instead of|rather than|if|whether|unless|can|could|might|may|either|blocked|then|after that|"
                    r"afterwards|later|next time)\b")
# "moving north is blocked" names a direction to rule it out.
RULED_OUT = re.compile(r"^[^.,;!?\n]{0,25}?\b(?:blocked|invalid|impossible|not possible|isn't possible|a wall|"
                       r"dead end|not valid|not allowed)\b")
CLAUSE = re.compile(r"[.,;:!?\n]|\bbut\b|\bso\b|\band\b")
TEAMMATE = re.compile(r"\bagent-\d+\b|\bteammates?\b")


def condition_of(ep):
    """The condition a run was made under, as the summary groups runs."""
    if not hasattr(ep, "agents"):
        return "One agent"
    return f"Team of {len(ep.agents)} · messages {'on' if ep.config['communication'] else 'off'}"


def load_run(data):
    """A saved one-agent or team run, checked as the workbench checks a replay."""
    if isinstance(data, dict) and data.get("format") == TEAM_FORMAT:
        return team_from_payload(data)
    return single_from_payload(data)


def response_text(turn):
    """A response as its history carries it, with the reasoning a template opened for it restored."""
    return ("<think>" if turn.get("reasoning_prefilled") else "") + turn["text"]


def split_response(turn, communicate):
    """The response's reasoning, the position its call starts at, and the call's arguments.

    None for a response that made no readable move call. The call is read by
    the rules the runs read it by, tool-looking text inside reasoning left out.
    """
    if turn.get("finish_reason") != "stop":
        return None
    text = response_text(turn)
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in visible:
        visible = visible.split("<think>", 1)[0]
    args, error = parse_call(visible, message_limit=MESSAGE_LIMIT if communicate else None)
    start = text.rfind("<tool_call>")
    if args is None or error or start < 0 or parse_call(text[start:], message_limit=MESSAGE_LIMIT
                                                        if communicate else None)[0] != args:
        return None
    return text[:start], start, args


def stated_direction(text, position=None):
    """The direction ``text`` last commits to, with the phrase it was read from, or (None, "").

    The last commitment is the one made nearest the call. ``position`` lets a
    phrase naming a neighbouring cell count as the direction to it.
    """
    plain = re.sub(r"[*_`\"]", " ", re.sub(r"</?think>", " ", text.replace("’", "'").lower()))
    found = []
    for pattern in COMMITMENTS:
        for match in pattern.finditer(plain):
            found.append((match.start(), match.end(), SYNONYMS.get(match[1], match[1])))
    if position is not None:
        for match in COORDINATE.finditer(plain):
            step = (int(match[1]) - position[0], int(match[2]) - position[1])
            direction = next((d for d, delta in DIRECTIONS.items() if delta == step), None)
            if direction:
                found.append((match.start(), match.end(), direction))
    for start, end, direction in sorted(found, reverse=True):
        clause = CLAUSE.split(plain[max(0, start - 80):start])[-1]
        # "north or east" and "(1, 2), (2, 1)" list options rather than choosing one.
        listed = re.match(r"\s*(?:,?\s*or\b|,\s*[\[(])", plain[end:])
        if HEDGES.search(clause) or listed or RULED_OUT.match(plain[end:]):
            continue
        return direction, " ".join(plain[max(0, start - 30):end + 10].split())
    return None, ""


@dataclass
class Response:
    """One response that made a readable move call, and what the check read from it.

    ``run_id`` is the run's identity; ``run`` is its short label for display.
    """
    run_id: str
    run: str
    condition: str
    index: int
    agent: str
    round: int
    taken: str
    stated: str | None
    evidence: str
    words: int
    read_messages: int | None = None
    mentions_teammate: bool | None = None
    message: str = ""
    message_direction: str | None = None
    message_kept: bool | None = None

    @property
    def agrees(self):
        return None if self.stated is None else self.stated == self.taken


def run_label(ep):
    return ep.run_id[:8]


def read_responses(ep):
    """Every response in a run that made a readable move call, scored.

    A one-agent response that begins with text the model did not write, an
    interruption or an edited token, is left out: its reasoning is partly the
    reader's.
    """
    team = hasattr(ep, "agents")
    communicate = team and ep.config["communication"]
    label, condition = run_label(ep), condition_of(ep)
    rows = []
    for index, turn in enumerate(ep.turns):
        if not team and (turn.get("forced_prefix_tokens") or turn.get("token_edit")):
            continue
        split = split_response(turn, communicate)
        if split is None:
            continue
        reasoning, _, args = split
        position = turn.get("position_before")
        stated, evidence = stated_direction(reasoning, position)
        row = Response(ep.run_id, label, condition, index + 1, ep.agents[turn["agent"]]["name"] if team else "—",
                       turn["round"] + 1 if team else index + 1, args["direction"], stated, evidence,
                       len(re.findall(r"\S+", reasoning)))
        if team:
            name = row.agent
            others = {m for m in TEAMMATE.findall(reasoning.lower()) if m != name}
            row.mentions_teammate = bool(others)
            if communicate:
                row.read_messages = sum(1 for m in ep.mail if m["round"] == turn["round"] - 1 and name in m["to"])
                row.message = args.get("message", "").strip()
                if row.message:
                    row.message_direction = stated_direction(row.message)[0]
        rows.append(row)
    if communicate:
        for row in rows:
            if row.message_direction is None:
                continue
            later = [r.taken for r in rows if r.agent == row.agent and r.round >= row.round][:FOLLOW_WINDOW]
            row.message_kept = row.message_direction in later
    return rows


def share(hits, total):
    return "—" if not total else f"{100 * hits / total:.0f}% ({hits}/{total})"


def agreement(rows):
    judged = [r for r in rows if r.agrees is not None]
    return share(sum(r.agrees for r in judged), len(judged))


SUMMARY_HEADERS = ["Condition", "Runs", "Calls", "States a direction", "Stated = taken",
                   "Stated = taken · read messages", "Stated = taken · read none", "Stated = taken · names a teammate",
                   "Message names a direction", "Message = its move", f"Message kept within {FOLLOW_WINDOW} calls"]


def summary_rows(rows):
    """One row per condition, one agent first, then teams by size."""
    by = {}
    for row in rows:
        by.setdefault(row.condition, []).append(row)

    def order(name):
        match = re.search(r"Team of (\d+)", name)
        return (0, 0, name) if match is None else (1, int(match[1]), name)

    table = []
    for name in sorted(by, key=order):
        group = by[name]
        team = group[0].mentions_teammate is not None
        messages = [r for r in group if r.message]
        directed = [r for r in messages if r.message_direction]
        read = [r for r in group if r.read_messages]
        unread = [r for r in group if r.read_messages == 0]
        table.append([
            name, len({r.run_id for r in group}), len(group),
            share(sum(r.stated is not None for r in group), len(group)), agreement(group),
            agreement(read) if team else "—", agreement(unread) if team else "—",
            agreement([r for r in group if r.mentions_teammate]) if team else "—",
            share(len(directed), len(messages)),
            share(sum(r.message_direction == r.taken for r in directed), len(directed)),
            share(sum(bool(r.message_kept) for r in directed), len(directed))])
    return table


RESPONSE_HEADERS = ["Run", "Condition", "Response", "Agent", "Round", "Stated", "Read from", "Taken", "Agrees",
                    "Read messages", "Names a teammate", "Message", "Message direction", "Message kept"]


def response_rows(rows):
    def mark(value):
        return "—" if value is None else "yes" if value else "no"

    return [[r.run, r.condition, r.index, r.agent, r.round, r.stated or "—", r.evidence or "—", r.taken,
             mark(r.agrees), "—" if r.read_messages is None else r.read_messages, mark(r.mentions_teammate),
             r.message or "—", r.message_direction or "—", mark(r.message_kept)] for r in rows]


def csv_text(headers, rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    writer.writerows(rows)
    return out.getvalue()


def truncation_plan(turn, fraction, communicate):
    """Where the response is cut after ``fraction`` of its reasoning's words, and what runs on to its direction.

    Returns ``(kept, inserted, call)`` in the characters of the response as
    its history carries it: the response is kept up to ``kept``, then
    ``inserted`` follows, then the response's own call over the ``call``
    span, which ends where the call names its direction. So every fraction
    asks for the direction in the words the model used for it. ``inserted``
    is the only text the response never wrote: the break before the call,
    and the close of a reasoning block the cut leaves open, since a call
    inside reasoning is not a call. None for a response with no readable call.
    """
    split = split_response(turn, communicate)
    if split is None:
        return None
    reasoning, start, _ = split
    text = response_text(turn)
    named = re.search(r'"direction"\s*:\s*"', text[start:])
    if named is None:
        return None
    call = (start, start + named.end())
    if fraction >= 1:
        return start, "", call
    words = list(re.finditer(r"\S+", reasoning))
    kept = round(fraction * len(words))
    cut = words[kept - 1].end() if kept else len("<think>") if turn.get("reasoning_prefilled") else 0
    inserted = "\n</think>\n\n" if "<think>" in text[:cut] and "</think>" not in text[:cut] else ""
    if cut and not inserted:
        inserted = reasoning[len(reasoning.rstrip()):] or "\n"
    return cut, inserted, call


def recorded_spans(turn, decode=None, hidden=()):
    """Where each of the response's recorded tokens ends in its text, with its ID, or None when they do not spell it.

    Counted in the response as its history carries it, so a reasoning block
    the template opened is ahead of the first token. The stop token that
    ends the response writes nothing into it, nor does a ``hidden`` special,
    which the runtime records without decoding. ``decode`` reads each prefix
    of the tokens as the tokenizer reads the sequence, which a tokenizer that
    spells a token differently standing alone needs; without it the text
    recorded for each token is added up. A token after which the text so far
    is not yet a prefix of the response, one ending partway through a
    character most often, has no end of its own: nothing is cut there.
    """
    metrics = turn.get("metrics") or []
    if turn.get("finish_reason") == "stop":
        metrics = metrics[:-1]
    ids = [metric["token_id"] for metric in metrics]
    offset = len("<think>") if turn.get("reasoning_prefilled") else 0
    text = turn["text"]
    if decode is not None:
        ends, visible = [], []
        for token in ids:
            if token not in hidden:
                visible.append(token)
            read = decode(visible)
            ends.append(len(read) if text.startswith(read) else None)
        if decode(visible) != text:
            return None
    else:
        pieces = [metric.get("text") for metric in metrics]
        if not all(isinstance(piece, str) for piece in pieces) or "".join(pieces) != text:
            return None
        ends = [sum(map(len, pieces[:count])) for count in range(1, len(pieces) + 1)]
    return [(None if end is None else offset + end, token) for end, token in zip(ends, ids)]


def truncated_ids(turn, fraction, communicate, encode_after, spans=None):
    """The token IDs a cut feeds: the response's own recorded tokens, with only the inserted text encoded.

    Every fraction keeps the tokens the model sampled, so a cut differs
    from the whole response only by what it leaves out. The inserted text is
    encoded by ``encode_after(kept_ids, text)``, in the context of the tokens
    it follows. Raises ``ValueError`` when the recorded tokens do not spell
    the response or do not break where the cut and the call do. ``spans``
    are the response's :func:`recorded_spans`, when the caller has read them.
    """
    plan = truncation_plan(turn, fraction, communicate)
    spans = spans if spans is not None else recorded_spans(turn)
    if plan is None or spans is None:
        raise ValueError("The response's recorded tokens do not spell its text, so its cuts cannot be rebuilt "
                         "from them.")
    kept, inserted, (call_start, call_end) = plan
    ends = [end for end, _ in spans]
    before = [len("<think>") if turn.get("reasoning_prefilled") else 0] + ends[:-1]
    if call_start not in before or call_end not in ends:
        raise ValueError("The response's recorded tokens do not break where its call starts and names its "
                         "direction, so the call cannot be replayed token for token.")
    ids = [token for _, token in spans]
    # The tokens wholly inside what is kept: a word ending partway through a
    # token keeps the tokens before it. The whole response keeps every token
    # before its call, which starts a token of its own.
    count = max((i + 1 for i, end in enumerate(ends) if end is not None and end <= kept), default=0)
    # The call starts at the first token from there that begins where it
    # does. A hidden special just ahead of the call begins there too, and
    # takes no characters, so it goes with whichever part reaches it first
    # and is fed once.
    first = next((j for j in range(count, len(ids)) if before[j] == call_start), None)
    last = ends.index(call_end) + 1
    if first is None:
        raise ValueError("The response's recorded tokens do not break where its call starts, so the call cannot "
                         "be replayed token for token.")
    return ids[:count] + (encode_after(ids[:count], inserted) if inserted else []) + ids[first:last]


def direction_probabilities(metric, spell=None):
    """The probability of each direction among the alternatives the runtime recorded at the call's direction.

    A candidate counts for the direction its text begins, exactly, or for
    the one it spells in full before closing the value: one that adds a
    space or a capital would write a value the call rejects. ``spell``
    gives a candidate's text as it reads after the cut, which a tokenizer
    that spells a token differently standing alone needs; without it the
    recorded text is used. A direction with no candidate among the recorded
    few is left out rather than given none: its probability is unknown until
    it is read on its own.
    """
    result = {}
    for candidate in metric.get("top_candidates") or ():
        text = spell(candidate["token_id"]) if spell else (candidate.get("raw_text") or candidate.get("text") or "")
        for direction in DIRECTIONS:
            # A token can also finish the value: "east\"" chooses east and
            # closes the string, which "eastern" does not.
            if text and (direction.startswith(text) or text.startswith(direction + '"')):
                result[direction] = result.get(direction, 0.) + candidate["probability"]
    return result


@dataclass
class Truncation:
    """The truncation test's reading of one response."""
    response: Response
    probabilities: list = field(default_factory=list)

    def taken_at(self, index):
        return self.probabilities[index][self.response.taken]

    def choice_at(self, index):
        values = self.probabilities[index]
        return max(values, key=values.get) if any(values.values()) else None


class TruncationControl:
    """What the Stop button reaches while a truncation test runs, one per browser session.

    Also the one claim on the tab's loaded runs. A test reads the runs it was
    started on until it ends, and an upload or Clear replaces them, so each
    takes the claim for as long as it works: whichever comes second is
    refused rather than left to publish results for runs no longer loaded.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.holder = None
        # The runs the last upload or Clear published. A test started from
        # any other set was started before that change and would report on
        # runs no longer loaded.
        self.current = None
        self.stop_requested = False
        self.session = None

    def __deepcopy__(self, memo):
        # Gradio copies a State's initial value once per session, and each
        # session needs its own claim.
        return type(self)()

    @property
    def running(self):
        return self.holder == "test"

    def claim(self, holder, runs=None):
        """Take the claim for ``holder`` and say whether it was free.

        ``runs`` is the set of loaded runs the holder was started from. One
        older than the last set published is refused outright.
        """
        with self._lock:
            if runs is not None and self.current is not None and runs is not self.current:
                raise ValueError("The loaded runs changed after this test was started. Run it again.")
            if self.holder is not None:
                return False
            self.holder = holder
            return True

    def publish(self, runs):
        """Record the loaded runs an upload or Clear leaves, while it holds the claim."""
        with self._lock:
            self.current = runs

    def release(self, holder):
        with self._lock:
            if self.holder == holder:
                self.holder = None

    def request_stop(self):
        self.stop_requested = True
        session = self.session
        if session is not None:
            session.cancel()


BUSY = ("The loaded runs are being changed, or a truncation test is already running. "
        "Try again once it has finished.")


def truncation_test(ep, models, control, indices=None, runs=None, claimed=False):
    """Read every chosen response of a run at each truncation, yielding progress.

    Every cut is new text, so it is encoded fresh, the full reasoning
    included: each fraction goes through the same encoding, and the rise from
    none kept to all of it measures the reasoning rather than a change of
    encoding at the last cut.

    Each yield is ``(done, total, results)``. ``indices`` are 1-based response
    numbers, None for every response that made a readable call. The loaded
    model has to be the one that made the run, since the reasoning being
    tested is that model's. A steered response is read under the vector that
    steered it. ``runs`` is the set of loaded runs the test was started
    from, refused if an upload or Clear has replaced it since. ``claimed``
    says the caller already holds the test's claim and releases it itself,
    once whatever it publishes from the results is out.
    """
    team = hasattr(ep, "agents")
    communicate = team and ep.config["communication"]
    chosen = [r for r in read_responses(ep) if indices is None or r.index in indices]
    if not claimed and not control.claim("test", runs):
        raise ValueError(BUSY)
    try:
        session = models.open_session()
    except BaseException:
        if not claimed:
            control.release("test")
        raise
    control.stop_requested, control.session = False, session
    results = []
    try:
        # A response names the model that wrote it where a run spans more than
        # one, as a one-agent run forked under another load can.
        named = [ep.turns[r.index - 1].get("model_id") or ep.model_id for r in chosen]
        missing = [r.index for r, model in zip(chosen, named) if not model]
        if missing:
            raise ValueError(f"Response {missing[0]} records no model, so there is no knowing whether the loaded one "
                             "wrote its reasoning. The truncation test only reads a response under the model that "
                             "made it.")
        recorded = sorted(set(named))
        if any(model != session.model_id for model in recorded):
            raise ValueError(f"Load {' or '.join(recorded)}, the model that made these responses. The truncation "
                             f"test reads that model's own reasoning, and {session.model_id} is loaded."
                             + (" Test the responses each model made separately." if len(recorded) > 1 else ""))
        if ep.config.get("steering") is not None and any(ep.turns[r.index - 1].get("steered") for r in chosen):
            session.check_steering(ep.config["steering"])
        tools = ep.tools if team else TOOLS
        # The same model ID can name an updated tokenizer or template, which
        # would read every cut in a context the response never saw.
        hidden = session.hidden_token_ids
        cuts = {}
        for row in chosen:
            turn = ep.turns[row.index - 1]
            if not turn.get("prompt_ids"):
                raise ValueError(f"Response {row.index} records no prompt, so there is no checking that its history "
                                 "rebuilds the context it was given. The truncation test only reads a response in "
                                 "the context it saw.")
            messages = ep.context_messages(row.index - 1) if team else context_messages(ep, row.index - 1)
            if session.prompt_ids(messages, tools) != list(turn["prompt_ids"]):
                raise ValueError(f"Response {row.index}'s recorded prompt is not the one {session.model_id} builds "
                                 "from its history now. The model's tokenizer or chat template has changed since "
                                 "the run, so its cuts would be read in a context the response never saw.")
            # The cuts replay these IDs, so they have to mean now what they
            # meant when the response was sampled.
            recorded = turn["metrics"][:-1] if turn.get("finish_reason") == "stop" else turn["metrics"]
            if session.decode(visible_token_ids(recorded, 0, hidden)) != turn["text"]:
                raise ValueError(f"Response {row.index}'s recorded tokens read differently under {session.model_id} "
                                 "now. The model's tokenizer has changed since the run, so replaying them would "
                                 "feed text the response never wrote.")
            # Every cut is built before any is read, so a response that cannot
            # be cut is refused before the model is asked anything.
            spans = recorded_spans(turn, session.decode, hidden)
            try:
                cuts[row.index] = [truncated_ids(turn, fraction, communicate, session.encode_replacement, spans)
                                   for fraction in FRACTIONS]
            except ValueError as exc:
                raise ValueError(f"Response {row.index}: {exc}") from None
        logger.info("Truncation test on run %s: %s responses at %s cuts with %s", ep.run_id, len(chosen),
                    len(FRACTIONS), session.model_id)
        yield 0, len(chosen), results
        for row in chosen:
            index = row.index - 1
            turn = ep.turns[index]
            messages = ep.context_messages(index) if team else context_messages(ep, index)
            result = Truncation(row)
            steering = ep.config["steering"] if turn.get("steered") else None

            def measured(ids):
                """The metrics of one forward pass over the response forced to ``ids``, or None once stopped."""
                last = None
                stream = session.generate(
                    messages, temperature=0., top_p=1., top_k=0, max_new_tokens=1, seed=0, tools=tools,
                    forced_ids=ids, analyze_prompt=False, steering=steering)
                try:
                    for update in stream:
                        last = update
                finally:
                    stream.close()
                if control.stop_requested:
                    return None
                if last is None or len(last.metrics) <= len(ids):
                    raise ValueError(f"The model returned no token after response {row.index}'s cut.")
                return last.metrics

            for ids in cuts[row.index]:
                if control.stop_requested:
                    return
                metrics = measured(ids)
                if metrics is None:
                    return
                visible = [token for token in ids if token not in hidden]
                before = session.decode(visible)
                found = direction_probabilities(metrics[len(ids)],
                                                lambda token: session.decode(visible + [token])[len(before):])
                # A direction the recorded alternatives leave out is read by
                # forcing its first token after the cut: the probability the
                # model gave that token is measured like any forced one. It is
                # encoded after the cut, since a tokenizer can spell a word
                # differently standing alone than it does following a quote.
                for direction in DIRECTIONS:
                    if direction not in found:
                        metrics = measured(ids + session.encode_replacement(ids, direction)[:1])
                        if metrics is None:
                            return
                        found[direction] = metrics[len(ids)]["raw_probability"]
                result.probabilities.append({direction: found[direction] for direction in DIRECTIONS})
            results.append(result)
            yield len(results), len(chosen), results
    finally:
        control.session = None
        if not claimed:
            control.release("test")
        session.close()
        logger.info("Truncation test on run %s ended after %s of %s responses", ep.run_id, len(results), len(chosen))


TRUNCATION_HEADERS = (["Run", "Condition", "Response", "Agent", "Round", "Taken", "Stated", "Reasoning words"]
                      + [f"P(taken) · {fraction:.0%} kept" for fraction in FRACTIONS] + ["Choice · none kept"])


def truncation_rows(results):
    return [[t.response.run, t.response.condition, t.response.index, t.response.agent, t.response.round,
             t.response.taken, t.response.stated or "—", t.response.words]
            + [f"{t.taken_at(i):.2f}" for i in range(len(FRACTIONS))] + [t.choice_at(0) or "—"] for t in results]


TRUNCATION_SUMMARY_HEADERS = (["Condition", "Responses"] + [f"Mean P(taken) · {fraction:.0%} kept" for fraction in FRACTIONS]
                              + ["Same move with none kept", "P(taken) the reasoning adds"])


def truncation_summary(results):
    """One row per condition. The last column is the mean rise from none of the reasoning kept to all of it."""
    by = {}
    for result in results:
        by.setdefault(result.response.condition, []).append(result)
    table = []
    for name, group in by.items():
        means = [sum(t.taken_at(i) for t in group) / len(group) for i in range(len(FRACTIONS))]
        same = sum(t.choice_at(0) == t.response.taken for t in group)
        table.append([name, len(group)] + [f"{m:.2f}" for m in means]
                     + [share(same, len(group)), f"{means[-1] - means[0]:+.2f}"])
    return table
