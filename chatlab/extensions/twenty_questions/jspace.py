"""Read a reply through the imported Jacobian lens: what J space holds while the model answers.

Two positions of every reply are read: the last prompt token, the state the
model starts its reply from, and the token just before the yes or no, the
state that chooses it. At each, every block the lens was fitted at gives its
top tokens, and every thing on the watch list gives its log probability,
averaged over those blocks. A thing the model is holding may show here
before it is ever written; a thing it is not holding has no reason to.
"""
import numpy as np

from .game import verdict_span

TOP = 8
# What one rank reading may hold, as ChatLab's lens enforces it.
RANK_ELEMENTS = 8_000_000
NO_LENS = ("Import a Jacobian lens for the loaded model in Chat → Layers and attention → Jacobian. "
           "J space needs a full-precision PyTorch load.")


def positions_of(turn, prompt_length):
    """(label, position) pairs to read for ``turn``: the reply's start, and its verdict if it has one.

    Positions index the prompt followed by the response's tokens. The state
    that chooses a token is the one after the token before it, so the verdict
    is read one position before its first token.
    """
    found = [("Starting the reply", prompt_length - 1)]
    # Searched in the tokens' own texts, not the streamed reply: each token is
    # decoded alone, so special tokens and split characters can make the two
    # differ, and only an offset into the joined texts maps onto a token.
    texts = [metric.get("text", "") for metric in turn["metrics"]]
    joined = "".join(texts)
    closed = joined.rfind("</think>")
    if closed < 0 and (turn.get("reasoning_prefilled") or "<think>" in joined):
        return found
    # The verdict is read in the visible answer, never in the reasoning.
    offset = closed + len("</think>") if closed >= 0 else 0
    span = verdict_span(joined[offset:])
    if span is not None:
        consumed = 0
        for index, text in enumerate(texts):
            consumed += len(text)
            if consumed > offset + span:
                found.append(("Choosing the answer", prompt_length + index - 1))
                break
    return found


def watch_targets(session, things):
    """Each watched thing as the token IDs the lens reads it through: the word after a space."""
    targets = {}
    for thing in things:
        ids = session.encode(" " + thing)
        if ids:
            targets[thing] = ids
    return targets


def read(session, ids, positions, things=(), top=TOP):
    """The J space of ``ids`` at ``positions``, as the page saves and draws it.

    Raises ``ValueError`` with what to do when the lens cannot read here.
    """
    lens = session.jacobian_lens()
    if lens is None or not lens.get("layers"):
        raise ValueError(NO_LENS)
    blocks = list(lens["layers"])
    limit = session.position_limit
    wanted = [position for _, position in positions]
    if limit is not None and max(wanted) + 1 > limit:
        raise ValueError(f"The game is {max(wanted) + 1:,} tokens long, past the {limit:,} the model reads at once.")
    found = []
    for label, position in positions:
        first = session.lens_ranks(ids, blocks[:1], [position])
        per_call = max(1, RANK_ELEMENTS // first.shape[-1])
        ranks = [first[0, 0]]
        for start in range(1, len(blocks), per_call):
            chunk = session.lens_ranks(ids, blocks[start:start + per_call], [position])
            ranks.extend(chunk[:, 0])
        tops = [np.argsort(row, kind="stable")[:top].tolist() for row in ranks]
        union = sorted({token for row in tops for token in row})
        scores = session.lens_log_probs(ids, union, blocks, [position])[:, 0]
        column = {token: index for index, token in enumerate(union)}
        found.append(dict(label=label, position=position, token=session.decode([ids[position]]), blocks=[
            dict(block=block, top=[dict(token_id=token, text=session.decode([token]),
                                        log_prob=float(scores[row, column[token]])) for token in tops[row]])
            for row, block in enumerate(blocks)]))
    watch = {}
    targets = watch_targets(session, things)
    if targets:
        values = session.lens_log_probs(ids, list(targets.values()), blocks, wanted)
        means = values.mean(axis=0)
        watch = {thing: [float(means[p, t]) for p in range(len(wanted))] for t, thing in enumerate(targets)}
    return dict(lens=lens.get("name", ""), load_id=session.load_id, blocks=blocks, positions=found, watch=watch)
