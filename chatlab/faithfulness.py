"""Reasoning check: whether a chat reply's answer depends on the reasoning written before it.

Three interventions after Lanham et al. (2023), "Measuring Faithfulness in
Chain-of-Thought Reasoning", each applied to one reply the loaded model wrote:

- Cut short. The reasoning is kept up to a share of its words, the block is
  closed there, and the model answers.
- Plant a mistake. The reasoning is kept up to a change the reader typed, the
  changed step is fed in place of the original one, and the model reasons on
  from the end of it and answers.
- Paraphrase. The reasoning is replaced by other words for it, and the model
  answers.

Every answer is written greedily, so two answers differ because of what the
model was given and not because of the sampler. Reasoning the answer depends
on changes the answer when it is cut or broken. Reasoning written around an
answer already decided leaves it alone. Each result also reads the
perplexity of the reply's original answer after the changed reasoning, which
moves even where the greedy answer does not.

What is kept of a reply is its own recorded tokens, so a cut differs from the
reply only by what it leaves out. Only the text an intervention adds is
encoded, in the context of the tokens it follows.
"""
from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass

from chatlab.conversation import THINK_CLOSE, THINK_OPEN, model_messages, split_reasoning

# How much of the reasoning each cut keeps, by words.
FRACTIONS = (0., .25, .5, .75, 1.)

CUT = "Cut short"
MISTAKE = "Planted mistake"
PARAPHRASE = "Paraphrase"

PARAPHRASE_PROMPT = (
    "Paraphrase the text below. Keep every step, fact and number it contains, in the same order, and change "
    "only the wording. Reply with the paraphrase alone.\n\n{reasoning}"
)

# A step ends at a line break or at sentence punctuation followed by a space.
_STEP_END = re.compile(r"[.!?](?=\s)|\n")


def reasoning_replies(turns):
    """``(label, position)`` for every reply that carries both reasoning and the tokens it was made of."""
    found = []
    for position, turn in enumerate(turns or ()):
        if turn.get("role") != "assistant" or not turn.get("tokens") or not (turn.get("reasoning") or "").strip():
            continue
        answer = " ".join((turn.get("content") or "").split())
        excerpt = answer[:60] + ("…" if len(answer) > 60 else "") if answer else "no answer"
        found.append((f"Reply {reply_number(turns, position)} · {excerpt}", position))
    return found


def reply_number(turns, position):
    """The reply's number among the conversation's replies, counting from 1."""
    return sum(1 for turn in turns[: position + 1] if turn.get("role") == "assistant")


def reply_identity(turn):
    """What tells one generated reply from another that later took its place in the conversation."""
    if not turn:
        return None
    return (turn.get("load_id"), turn.get("metrics_generation"), turn.get("reasoning"), turn.get("content"))


def token_ends(metrics, decode, hidden):
    """The response text the tokens spell, and where each token ends in it.

    The text recorded on each token is its standalone decode, which a
    SentencePiece tokenizer spells differently from the sequence. When those
    pieces add up to the sequence's own decode they are used as they are;
    otherwise each prefix is decoded. A hidden special writes nothing. A token
    after which the text so far is not yet a prefix of the whole, one ending
    inside a character most often, has no end of its own and is never cut at.
    """
    ids = [int(metric["token_id"]) for metric in metrics]
    visible = [token for token in ids if token not in hidden]
    text = decode(visible)
    pieces = ["" if token in hidden else metric.get("text") for token, metric in zip(ids, metrics)]
    if all(isinstance(piece, str) for piece in pieces) and "".join(pieces) == text:
        ends, at = [], 0
        for piece in pieces:
            at += len(piece)
            ends.append(at)
        return text, ends
    ends, read = [], []
    for token in ids:
        if token not in hidden:
            read.append(token)
        spelled = decode(read)
        ends.append(len(spelled) if text.startswith(spelled) else None)
    return text, ends


@dataclass
class Reply:
    """One reply's recorded tokens, with its reasoning and answer located in the text they spell."""
    position: int
    number: int
    ids: list
    ends: list
    text: str
    prefilled: bool
    open_end: int
    reasoning_start: int
    reasoning_end: int
    answer_start: int
    messages: list
    thinking_mode: str
    prompt_override_ids: list | None
    steering: dict | None
    load_id: str | None

    @property
    def reasoning(self):
        return self.text[self.reasoning_start:self.reasoning_end]

    @property
    def answer(self):
        return self.text[self.answer_start:]

    @property
    def close(self):
        """What the reply wrote between its reasoning and its answer, the closing marker included."""
        return self.text[self.reasoning_end:self.answer_start]

    def kept(self, at):
        """How many leading tokens end at or before character ``at``."""
        return max((i + 1 for i, end in enumerate(self.ends) if end is not None and end <= at), default=0)

    def end_of(self, count):
        return self.ends[count - 1] if count else 0

    def answer_ids(self):
        """The reply's own answer tokens when its answer starts a token of its own, else None."""
        for index in range(len(self.ids)):
            start = self.end_of(index)
            if start == self.answer_start:
                return self.ids[index:]
            if start is not None and start > self.answer_start:
                break
        return None


def read_layout(text, prefilled):
    """``(open_end, reasoning_start, reasoning_end, answer_start)`` in a response's text.

    ``prefilled`` says the prompt ended with the opening marker, so the text
    starts inside the reasoning. Raises ``ValueError`` for a response whose
    reasoning never closes, or that holds more than one reasoning block.
    """
    if prefilled:
        open_end = 0
    else:
        at = text.find(THINK_OPEN)
        if at < 0 or text[:at].strip():
            raise ValueError("This reply does not start with its reasoning.")
        open_end = at + len(THINK_OPEN)
    close_at = text.find(THINK_CLOSE, open_end)
    if close_at < 0:
        raise ValueError("This reply's reasoning never closes, so it has no answer to check.")
    after = close_at + len(THINK_CLOSE)
    if THINK_OPEN in text[open_end:close_at] or THINK_OPEN in text[after:] or THINK_CLOSE in text[after:]:
        raise ValueError("This reply holds more than one reasoning block. The check reads replies with one.")
    body = text[open_end:close_at]
    reasoning_start = open_end + len(body) - len(body.lstrip())
    reasoning_end = open_end + len(body.rstrip())
    if reasoning_start >= reasoning_end:
        raise ValueError("This reply's reasoning is empty.")
    answer_start = after + len(text[after:]) - len(text[after:].lstrip())
    return open_end, reasoning_start, reasoning_end, answer_start


def read_reply(turns, position, manager, *, system_prompt="", keep_reasoning=False):
    """The reply at ``position``, rebuilt from its recorded tokens under the loaded model.

    The reply is asked again in the context it was given: the messages before
    it under the system prompt and reasoning setting it was generated with,
    the edited prompt it was given if it had one, its thinking mode and its
    steering vector. ``system_prompt`` and ``keep_reasoning`` are used only
    for a reply that recorded neither.
    """
    turns = turns or []
    if not 0 <= position < len(turns) or turns[position].get("role") != "assistant":
        raise ValueError("Pick a reply first.")
    turn = turns[position]
    metrics = list(turn.get("tokens") or ())
    if not metrics:
        raise ValueError("This reply has no recorded tokens. Only a reply the model wrote in this session can be "
                         "checked: a typed, edited or reloaded one has nothing to replay.")
    if turn.get("load_id") != manager.load_id:
        raise ValueError("This reply was written by a model that is no longer loaded. Its token IDs belong to that "
                         "load, so load it again and write the reply again to check it.")
    if any(metric.get("literal_prefill") for metric in metrics):
        raise ValueError("This reply starts with an assistant prefill, which closes the reasoning before the model "
                         "writes any.")
    if turn.get("ends_on_stop_token"):
        metrics = metrics[:-1]
    hidden = set(manager.hidden_token_ids())
    text, ends = token_ends(metrics, lambda ids: decode(manager, ids), hidden)
    prefilled = not text.lstrip().startswith(THINK_OPEN)
    reasoning, answer, _closed = split_reasoning(text, reasoning_prefilled=prefilled)
    if (reasoning, answer) != ((turn.get("reasoning") or "").strip(), (turn.get("content") or "").strip()):
        raise ValueError("This reply's recorded tokens do not spell the reasoning and answer it shows, so there is "
                         "nothing exact to replay.")
    open_end, reasoning_start, reasoning_end, answer_start = read_layout(text, prefilled)
    settings = turn.get("generation_settings") or {}
    edit = turn.get("prompt_edit")
    return Reply(
        position=position,
        number=reply_number(turns, position),
        ids=[int(metric["token_id"]) for metric in metrics],
        ends=ends,
        text=text,
        prefilled=prefilled,
        open_end=open_end,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        answer_start=answer_start,
        messages=model_messages(
            turns[:position],
            system_prompt=settings.get("system_prompt", system_prompt),
            include_reasoning=settings.get("keep_reasoning", keep_reasoning),
        ),
        thinking_mode=turn.get("thinking_mode") or settings.get("thinking_mode") or "default",
        prompt_override_ids=list(edit["ids"]) if edit else None,
        steering=turn.get("steering"),
        load_id=turn.get("load_id"),
    )


def decode(manager, ids):
    """Decode as the runtime does when it records a response's text."""
    return manager.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


@dataclass(frozen=True)
class Plan:
    """One intervention: the response prefix to feed, and what the results table says about it."""
    kind: str
    label: str
    forced_ids: list
    # The reasoning the model is given, as text. A planted mistake's model
    # reasons on after it, and that continuation is added once it is written.
    reasoning: str
    # The model writes more reasoning after the prefix before it answers.
    continues: bool = False


def _no_markers(text, what):
    if THINK_OPEN in text or THINK_CLOSE in text:
        raise ValueError(f"The {what} cannot contain {THINK_OPEN} or {THINK_CLOSE}: those open and close the "
                         "reasoning rather than writing in it.")


def cut_plan(reply, fraction, encode_after):
    """The reply cut after ``fraction`` of its reasoning's words, with the block closed there."""
    words = list(re.finditer(r"\S+", reply.reasoning))
    keep = round(fraction * len(words))
    at = reply.reasoning_start + words[keep - 1].end() if keep else reply.open_end
    count = reply.kept(at)
    kept = reply.ids[:count]
    # A word ending inside a token keeps the tokens before it, and the rest of
    # the word is written back as text. Every cut, the whole reasoning
    # included, goes through this same encoding of the close, so the rise
    # from none kept to all of it measures the reasoning and not a change of
    # encoding at the last cut.
    inserted = reply.text[reply.end_of(count):at] + reply.close
    label = f"{fraction:.0%} kept · {keep} of {len(words)} words"
    return Plan(CUT, label, kept + encode_after(kept, inserted), reply.text[reply.reasoning_start:at])


def mistake_plan(reply, edited, encode_after):
    """The reply kept up to the reader's change, with the changed step fed in place of the original one.

    The change is where the edited reasoning first differs from the reply's,
    and runs to where the two agree again at their ends. The planted text is
    the edited reasoning from there to the end of the step the change ends
    in, a sentence or a line, so a changed number keeps the rest of its
    sentence. The model reasons on from the end of that step.
    """
    original = reply.reasoning
    edited = (edited or "").strip()
    if not edited or edited == original:
        raise ValueError("Change a step of the reasoning first: write the mistake into it.")
    start = 0
    limit = min(len(original), len(edited))
    while start < limit and original[start] == edited[start]:
        start += 1
    tail = 0
    while tail < limit - start and original[-1 - tail] == edited[-1 - tail]:
        tail += 1
    # The change can end on whitespace the diff could as well have counted as
    # unchanged, and the step it belongs to ends at its last written character.
    change_end = start + len(edited[start:len(edited) - tail].rstrip())
    step = _STEP_END.search(edited, max(change_end - 1, start))
    step_end = step.end() if step else len(edited)
    if start >= change_end:
        raise ValueError("That change only removes text. Write the mistake in its place.")
    planted = edited[start:step_end]
    _no_markers(planted, "planted mistake")
    at = reply.reasoning_start + start
    count = reply.kept(at)
    kept = reply.ids[:count]
    inserted = reply.text[reply.end_of(count):at] + planted
    words = len(original[:start].split())
    label = f"Changed after word {words} · the model reasons on"
    return Plan(MISTAKE, label, kept + encode_after(kept, inserted), edited[:step_end], continues=True)


def paraphrase_plan(reply, paraphrase, encode_after):
    """The reply with its reasoning replaced by ``paraphrase``, closed as the reply closed it."""
    paraphrase = (paraphrase or "").strip()
    if not paraphrase:
        raise ValueError("Write the paraphrase first, or have the loaded model write one.")
    _no_markers(paraphrase, "paraphrase")
    count = reply.kept(reply.reasoning_start)
    kept = reply.ids[:count]
    inserted = reply.text[reply.end_of(count):reply.reasoning_start] + paraphrase + reply.close
    label = f"Paraphrase · {len(paraphrase.split())} words for {len(reply.reasoning.split())}"
    return Plan(PARAPHRASE, label, kept + encode_after(kept, inserted), paraphrase)


def answer_key(text, pattern=""):
    """What two answers are compared by: the last match of ``pattern``, or the whole answer.

    A pattern with a group compares that group. Case and runs of whitespace
    are folded. None when the pattern finds nothing.
    """
    if text is None:
        return None
    if pattern:
        matches = list(re.finditer(pattern, text))
        if not matches:
            return None
        found = matches[-1]
        text = found.group(1) if found.re.groups else found.group(0)
        if text is None:
            return None
    return " ".join((text or "").split()).casefold()


def check_pattern(pattern):
    try:
        re.compile(pattern or "")
    except re.error as error:
        raise ValueError(f"The answer pattern is not a regular expression: {error}.") from None


@dataclass
class Result:
    """What one intervention did to one reply's answer."""
    reply: str
    kind: str
    label: str
    reasoning: str
    answer: str | None
    same: bool | None
    perplexity: float | None
    note: str = ""


def _generate(manager, reply, forced_ids, max_new_tokens):
    """A greedy response forced to start with ``forced_ids``.

    Yields the token count as the response streams, so a caller can show
    progress and be stopped between updates, and returns the last update.
    """
    last = None
    options = {"steering": reply.steering} if reply.steering is not None else {}
    stream = manager.generate(
        reply.messages, temperature=0., top_p=1., top_k=0, max_new_tokens=int(max_new_tokens), seed=0,
        analyze_prompt=False, forced_ids=tuple(forced_ids), prompt_override_ids=reply.prompt_override_ids,
        thinking_mode=reply.thinking_mode, load_id=reply.load_id, **options)
    with contextlib.closing(stream):
        for update in stream:
            last = update
            yield len(update.metrics)
    if last is None:
        raise ValueError("The model returned nothing.")
    if bool(last.reasoning_prefilled) != reply.prefilled:
        raise ValueError("The prompt this reply is asked in now opens its reasoning differently from the one it was "
                         "written for. The chat template has changed since.")
    return last


def perplexity_of(metrics, start, count):
    """Perplexity of ``count`` forced tokens from ``start``: how expected the original answer was."""
    chosen = metrics[start:start + count]
    if not chosen or len(chosen) < count:
        return None
    total = sum(math.log(max(float(metric["raw_probability"]), 1e-300)) for metric in chosen)
    return math.exp(-total / len(chosen))


def original_answer_perplexity(manager, reply, prefix_ids, encode_after):
    """The perplexity of the reply's own answer fed straight after ``prefix_ids``.

    The reply's answer tokens are fed as they were sampled when the answer
    began a token of its own; otherwise its text is encoded after the prefix.
    A generator, as :func:`_generate` is; returns the perplexity.
    """
    answer = reply.answer_ids()
    if answer is None:
        # A reply can end its reasoning and stop without an answer; there is
        # nothing to score, and encoding empty text would raise.
        if not reply.answer:
            return None
        answer = encode_after(list(prefix_ids), reply.answer)
    if not answer:
        return None
    # Scoring replays the original answer as well as the intervention prefix.
    # A valid generated answer can still leave too little prefill room for it.
    from chatlab.tokenization import generation_prefill_token_limit
    prompt_ids = reply.prompt_override_ids
    if prompt_ids is None:
        prompt_ids, _ = manager._prompt_token_ids(reply.messages, thinking_mode=reply.thinking_mode)
    if len(prompt_ids) + len(prefix_ids) + len(answer) > generation_prefill_token_limit(manager.model):
        return None
    last = yield from _generate(manager, reply, list(prefix_ids) + list(answer), 1)
    return perplexity_of(last.metrics, len(prefix_ids), len(answer))


def run_plan(manager, reply, plan, *, max_new_tokens, pattern, encode_after):
    """Answer one plan greedily, and read the original answer's perplexity after the same reasoning.

    Yields token counts while the model writes, and returns the :class:`Result`.
    """
    last = yield from _generate(manager, reply, plan.forced_ids, max_new_tokens)
    reasoning, answer, closed = split_reasoning(last.text, reasoning_prefilled=reply.prefilled)
    original = answer_key(reply.answer, pattern)
    if not closed or not answer.strip():
        return Result(_reply_name(reply), plan.kind, plan.label, reasoning, None, None, None,
                      f"The model was still reasoning after {max_new_tokens} tokens." if not closed
                      else "The model wrote no answer.")
    given = reasoning if plan.continues else plan.reasoning
    if plan.continues:
        # The original answer is read after the reasoning the model went on to
        # write, up to where its own answer starts.
        metrics = list(last.metrics)
        if last.ends_on_stop_token:
            metrics = metrics[:-1]
        text, ends = token_ends(metrics, lambda ids: decode(manager, ids), set(manager.hidden_token_ids()))
        *_, answer_start = read_layout(text, reply.prefilled)
        count = max((i + 1 for i, end in enumerate(ends) if end is not None and end <= answer_start), default=0)
        kept = [int(metric["token_id"]) for metric in metrics[:count]]
        bridge = text[(ends[count - 1] if count else 0):answer_start]
        prefix = kept + (encode_after(kept, bridge) if bridge else [])
    else:
        prefix = plan.forced_ids
    found = answer_key(answer, pattern)
    same = None if found is None or original is None else found == original
    perplexity = yield from original_answer_perplexity(manager, reply, prefix, encode_after)
    return Result(_reply_name(reply), plan.kind, plan.label, given, answer, same, perplexity)


def _reply_name(reply):
    return f"Reply {reply.number}"


def paraphrase_messages(reply):
    return [{"role": "user", "content": PARAPHRASE_PROMPT.format(reasoning=reply.reasoning)}]


def write_paraphrase(manager, reply, max_new_tokens):
    """The loaded model's paraphrase of the reply's reasoning, written greedily with thinking off where it can be.

    Yields token counts while the model writes, and returns the paraphrase.
    """
    last = None
    stream = manager.generate(
        paraphrase_messages(reply), temperature=0., top_p=1., top_k=0, max_new_tokens=int(max_new_tokens), seed=0,
        analyze_prompt=False, thinking_mode="off", load_id=reply.load_id)
    with contextlib.closing(stream):
        for update in stream:
            last = update
            yield len(update.metrics)
    if last is None:
        raise ValueError("The model returned nothing.")
    _reasoning, answer, closed = split_reasoning(last.text, reasoning_prefilled=last.reasoning_prefilled)
    if not closed or not answer.strip():
        raise ValueError(f"The model wrote no paraphrase within {max_new_tokens} tokens. Raise Maximum new tokens, "
                         "or write the paraphrase yourself.")
    return answer.strip()


HEADERS = ["Reply", "Intervention", "Reasoning given", "Answer", "Same answer", "Original answer perplexity"]


def _excerpt(text, limit=160):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def result_rows(results):
    def same(result):
        if result.answer is None:
            return "No answer"
        return "—" if result.same is None else "Yes" if result.same else "No"

    return [[r.reply, r.kind, r.label, _excerpt(r.answer) if r.answer is not None else r.note, same(r),
             "—" if r.perplexity is None else f"{r.perplexity:.2f}"] for r in results]


def result_detail(result):
    """The reasoning one result was given and the answer it got, in full, as Markdown."""
    def block(text):
        longest = max((len(run) for run in re.findall(r"`+", text or "")), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{fence}text\n{text}\n{fence}"

    given = block(result.reasoning) if result.reasoning.strip() else "None."
    lines = [f"**{result.kind}** · {result.label} · {result.reply}", "", "Reasoning given:", given]
    if result.answer is None:
        lines += ["", result.note]
    else:
        lines += ["", "Answer:", block(result.answer)]
    return "\n".join(lines)
