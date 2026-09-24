"""Two runs of the same thing, read against each other token by token.

A *run* is one pass of the model over one piece of text, kept whole: what it
produced, every token's measurements, and the whole configuration it ran
under. Two of them held side by side is the only way to answer what changing
one thing did - a quantization, a steering vector, a system prompt, a seed -
because a single run has nothing to be different from.

The comparison itself is deliberately narrow. Two runs are compared while
their token sequences agree; where the sequences part, the text after the
split is answering a different question in each run and its measurements
cannot be subtracted. Measuring a fixed passage twice never parts, which is
why that mode exists: it is the comparison that stays exact to the last
token.
"""

from __future__ import annotations

from chatlab.token_metrics import (
    SEQUENTIAL_FILLS,
    UNSCORED_FILL,
    UNSCORED_LABEL,
    summarize,
)


SLOTS = ("A", "B")

# The two ways a slot is filled. A reply is what the model writes, so two
# replies share a prompt and part company somewhere inside the answer. A
# measurement is a passage the reader supplies, which both runs are made to
# read, so every position lines up and nothing is lost to a split.
REPLY, MEASUREMENT = "reply", "measurement"


# How far apart the two runs held one shared token, in bits of surprise. The
# first four buckets are the ramp the rest of the application uses for a
# measurement; the fifth is not a measurement at all but the absence of one,
# and it takes the ramp's warmest fill because it is where reading stops.
GAP_LABELS = (
    "Together (<0.5 bits)",
    "Apart (0.5–2)",
    "Far apart (2–5)",
    "Unrecognizable (5+)",
)
GAP_EDGES = (0.5, 2.0, 5.0)
SPLIT_LABEL = "After the split"
# A token one run has and the other does not, covering no characters: an end
# marker sampled by one reply and not the other. Neither a divergence nor a
# comparison, so it takes the same neutral fill as a token with no
# measurement rather than the ramp's warmest.
TRAILING_LABEL = "Only in this run"
GAP_COLORS = dict(zip(GAP_LABELS, SEQUENTIAL_FILLS[:4])) | {
    SPLIT_LABEL: SEQUENTIAL_FILLS[4],
    TRAILING_LABEL: UNSCORED_FILL,
    UNSCORED_LABEL: UNSCORED_FILL,
}
GAP_CAPTION = (
    "Each shared token is colored by how far the two runs' surprise at it sat "
    "apart, in bits. Warmer is further apart. Red is where the two runs stopped "
    "writing the same tokens, after which nothing is comparable."
)


EMPTY_SLOT = "Empty. Run the prompt into this slot to fill it."


# Two models are two tokenizers, and a token ID says nothing across them, so
# the runs are lined up on the text each token stands for. Surprise is in bits
# either way, but a bit costs more in a small vocabulary than in a large one,
# and the two runs cut the same passage into different numbers of tokens - so
# the per-token gap is a reading about this token, not a score for the models.
CROSS_MODEL_CAVEAT = (
    "The two runs came from different models, so they are lined up on the text "
    "each token stands for rather than on token IDs, which mean nothing across "
    "two vocabularies. Compare the per-token gaps with that in mind: two "
    "tokenizers cut the same passage differently, and a bit of surprise is not "
    "the same size in two vocabularies. The top-choice count is the roughest "
    "of the readings here: each run's first choice is decoded on its own, "
    "without the text in front of it, so two tokenizers that differ over "
    "where a word boundary lives can show the same continuation as a change."
)


def gap_category(delta: float) -> str:
    """Which bucket an aligned span's surprise gap falls in."""

    for label, edge in zip(GAP_LABELS, GAP_EDGES):
        if abs(delta) < edge:
            return label
    return GAP_LABELS[-1]


def token_ends(metrics, recorded=None) -> list[int]:
    """Where each token ends in its run's decoded text.

    ``recorded`` is what the run measured as it was made, one cumulative
    character count per token. It is preferred over anything derived here
    because decoding is not piecewise: a byte-level tokenizer can split one
    character across two tokens, and each of those tokens decoded on its own
    is a replacement character rather than half of anything. Adding up the
    lengths of standalone decodes would put the boundaries in the wrong
    places and, worse, make two runs of the same text look different.

    Without a recording - an older export, a caller that has only metrics -
    the standalone lengths are all there is, and they are right wherever no
    character was split.
    """

    if recorded and len(recorded) == len(metrics):
        return [int(value) for value in recorded]
    ends, total = [], 0
    for metric in metrics:
        total += len(_token_text(metric))
        ends.append(total)
    return ends


def align(
    left,
    right,
    *,
    by_text: bool = False,
    left_text: str = "",
    right_text: str = "",
    left_ends=None,
    right_ends=None,
) -> list[dict]:
    """Where the two runs are reading the same thing, as a list of spans.

    A span is a stretch of characters both runs covered, together with the
    tokens each spent on it. It is the unit a comparison can be made in,
    because it is the largest unit two runs are guaranteed to share.

    Within one vocabulary every span is one token on each side, matched on
    token IDs: two different tokens can decode to the same characters, and a
    comparison that called them equal would subtract measurements taken in
    contexts that had already parted.

    Across two vocabularies there are no IDs to match on - ID 4192 in one
    model and ID 4192 in another stand for unrelated text - and there are not
    always matching tokens either: ``hel`` + ``lo`` in one tokenizer is
    ``hello`` in the next. Matching token against token would call that
    identical passage divergent at its first character. So the two are walked
    together by the characters they have covered, and a span closes wherever
    both sides have covered the same text. Summing surprise over a span is
    what makes that honest: total bits over the same characters is the same
    question asked of both models, where one model's reading of half a word
    against another's reading of a whole one is not.

    Alignment is a prefix. It stops at the first place the two texts cannot
    be made to agree, because after that the runs are reading different
    things and later characters that happen to coincide were arrived at
    through different contexts.

    The characters are counted from what each run decoded as it was made -
    see :func:`token_ends` - rather than from token decodes added together,
    which is not the same string when a character is split across two tokens.
    """

    left, right = list(left or ()), list(right or ())
    ends_here = token_ends(left, left_ends)
    ends_there = token_ends(right, right_ends)
    text_here = left_text or "".join(_token_text(metric) for metric in left)
    text_there = right_text or "".join(_token_text(metric) for metric in right)
    spans: list[dict] = []
    here = there = 0
    covered_here = covered_there = 0
    parted = False
    while here < len(left) and there < len(right):
        start_here, start_there = here, there
        if not by_text:
            if int(left[here]["token_id"]) != int(right[there]["token_id"]):
                parted = True
                break
            here, there = here + 1, there + 1
        else:
            here, there = here + 1, there + 1
            # Extend whichever side has covered fewer characters until both
            # stand at the same place, past where the last span ended.
            # Standing level without having moved is not a boundary: a token
            # can decode to nothing at all - the first byte of a split
            # character, a hidden special token - and closing there would
            # make an empty span, compare those tokens' surprise separately
            # instead of through the next real character, and call the two
            # runs agreed before the bytes that follow decode differently.
            # Running out on both sides while level is the exception: the
            # tokens left over decode to nothing, and they belong to the span
            # they trail rather than to the divergence after it.
            while True:
                end_here, end_there = ends_here[here - 1], ends_there[there - 1]
                if end_here == end_there and end_here > covered_here:
                    break
                if end_here < end_there or (end_here == end_there and here < len(left)):
                    if here >= len(left):
                        break
                    here += 1
                elif there < len(right):
                    there += 1
                else:
                    break
            if (
                ends_here[here - 1] != ends_there[there - 1]
                or (ends_here[here - 1] <= covered_here
                    and (here < len(left) or there < len(right)))
            ):
                parted = True
                break
        end_here, end_there = ends_here[here - 1], ends_there[there - 1]
        covered = text_here[covered_here:end_here]
        if covered != text_there[covered_there:end_there]:
            # The two stand at the same offset over different characters, so
            # they were never reading the same thing. Checked on the ID path
            # too, where it should never fire: equal IDs from one vocabulary
            # decode alike, unless the two runs disagreed about which tokens
            # are shown at all. That disagreement is the whole reason this is
            # not an assertion - a fingerprint can only rule out the causes
            # it was told to look for, and this rules out the rest.
            parted = True
            break
        spans.append(_span(
            left, right, start_here, here, start_there, there, len(spans) + 1, covered,
        ))
        covered_here, covered_there = end_here, end_there

    if not parted:
        # One run can finish with a token the other has no answer for and no
        # characters to show for it: a reply that sampled its end marker
        # against one that ran into the token limit having written exactly
        # the same words. The walk above stops when either side runs out, so
        # that leftover token would be painted as though the two had parted
        # and the headline would say so, over text that never differed. It
        # is not a divergence, and it is not comparable either - it is one
        # run's alone, and it is drawn that way.
        trailing_here, trailing_there = here, there
        while trailing_here < len(left) and ends_here[trailing_here] == covered_here:
            trailing_here += 1
        while trailing_there < len(right) and ends_there[trailing_there] == covered_there:
            trailing_there += 1
        if trailing_here > here or trailing_there > there:
            spans.append(_trailing_span(here, trailing_here, there, trailing_there, len(spans) + 1))
    return spans


def _trailing_span(here0, here1, there0, there1, index) -> dict:
    """Tokens one run has left over that cover no characters at all."""

    return {
        "position": index,
        "text": "",
        "display_text": "",
        "left_range": (here0, here1),
        "right_range": (there0, there1),
        "one_to_one": False,
        "comparable_choice": False,
        "trailing": True,
        "scored": False,
        "surprise_bits": 0.0,
        "left_surprise": None,
        "right_surprise": None,
        "left_top": "",
        "left_top_id": None,
        "left_top_determined": False,
        "right_top": "",
        "right_top_id": None,
        "right_top_determined": False,
    }


def _token_text(metric: dict) -> str:
    return metric.get("text") or ""


def _span(left, right, here0, here1, there0, there1, index, covered) -> dict:
    """One aligned stretch, with what each run spent on it."""

    mine, yours = left[here0:here1], right[there0:there1]
    scored = all(metric.get("scored", True) for metric in mine + yours)
    left_bits = sum(float(metric["surprise_bits"]) for metric in mine) if scored else None
    right_bits = sum(float(metric["surprise_bits"]) for metric in yours) if scored else None
    # One token a side is the ordinary case and the only one where "what
    # would this model have written instead" has an answer: a span of three
    # tokens against one has three first choices on one side and one on the
    # other, and no pairing between them.
    one_to_one = len(mine) == 1 and len(yours) == 1
    left_top = _top_choice(mine[0]) if one_to_one else (None, "", False, False)
    right_top = _top_choice(yours[0]) if one_to_one else (None, "", False, False)
    # Both candidates have to exist for the question to have an answer. The
    # IDs establish that and nothing else: a valid special token can decode
    # to the empty string, so an empty text is a choice like any other and
    # only a missing candidate is a missing choice.
    comparable_choice = one_to_one and left_top[0] is not None and right_top[0] is not None
    return {
        "position": index,
        "text": covered,
        "display_text": "".join(
            metric.get("display_text") or _token_text(metric) for metric in mine
        ),
        "left_range": (here0, here1),
        "right_range": (there0, there1),
        "one_to_one": one_to_one,
        "comparable_choice": comparable_choice,
        "scored": scored,
        "surprise_bits": abs(left_bits - right_bits) if scored else 0.0,
        "left_surprise": left_bits,
        "right_surprise": right_bits,
        "left_top": left_top[1],
        "left_top_id": left_top[0],
        "left_top_determined": left_top[2],
        "left_top_stops": left_top[3],
        "right_top": right_top[1],
        "right_top_id": right_top[0],
        "right_top_determined": right_top[2],
        "right_top_stops": right_top[3],
    }


# What a byte-level tokenizer decodes an incomplete character to. Two
# different half-characters both come back as this, so it says only that the
# characters are not established yet.
UNDETERMINED = "\ufffd"


def _top_choice(metric: dict) -> tuple[int | None, str, bool, bool]:
    """What this run's model would have written here, left to itself.

    The ID comes back with the text because the text alone cannot answer
    whether the choice changed: distinct vocabulary entries can decode to the
    same characters, and two special tokens can both decode to nothing at
    all. Within one vocabulary the ID is the answer, for the same reason
    :func:`align` matches on IDs there. Across two it is meaningless, and the
    text is all there is.
    """

    candidates = metric.get("top_candidates") or ()
    if not candidates:
        return None, "", False, False
    first = candidates[0]
    if isinstance(first, dict):
        token_id, text = first.get("token_id"), first.get("text", "")
        raw, stops = first.get("raw_text"), bool(first.get("stops"))
    else:
        token_id, text = getattr(first, "token_id", None), getattr(first, "text", "")
        raw, stops = getattr(first, "raw_text", None), bool(getattr(first, "stops", False))
    # The raw decode where the run recorded one. What is shown for a token
    # that decodes to nothing is its vocabulary label, and two models label
    # their end-of-text markers differently, so comparing the labels would
    # call two models that both chose to stop a change of mind.
    shown = text if raw is None else raw
    # A candidate holding part of a character says nothing about what it
    # would add: two different halves decode alike, and the same half can
    # decode differently after another prefix. Within one vocabulary the ID
    # still answers the question; across two there is nothing to compare.
    return (
        (None if token_id is None else int(token_id)),
        shown,
        UNDETERMINED not in shown,
        stops,
    )


def strip(metrics, spans, side: str = "left") -> list[tuple[str, str]]:
    """One run's tokens, colored by how far the other run sat from them.

    A span's color goes on every token this run spent inside it, so a stretch
    one model wrote in three tokens and the other in one is drawn as the one
    reading it is. Tokens past the aligned region take the split color: the
    two runs are no longer reading the same thing there.
    """

    labels: dict[int, str] = {}
    for span in spans or ():
        low, high = span["left_range"] if side == "left" else span["right_range"]
        if span.get("trailing"):
            label = TRAILING_LABEL
        elif span["scored"]:
            label = gap_category(span["surprise_bits"])
        else:
            label = UNSCORED_LABEL
        for index in range(low, high):
            labels[index] = label
    return [
        (metric["display_text"], labels.get(index, SPLIT_LABEL))
        for index, metric in enumerate(metrics or ())
    ]


DIVERGENCE_HEADERS = [
    "#",
    "Text",
    "A surprise",
    "B surprise",
    "Δ bits",
    "A would write",
    "B would write",
]


# Enough rows to show a pattern, few enough to read. The whole comparison is
# in the export beside the table.
DIVERGENCE_ROWS = 25


def divergence_rows(spans, limit: int = DIVERGENCE_ROWS) -> list[list]:
    """The aligned spans the two runs read most differently, widest first.

    The last two columns are empty for a span of several tokens against one:
    there are three first choices on one side and one on the other, and no
    pairing between them to report.
    """

    ranked = sorted(
        (item for item in spans if item["scored"]),
        key=lambda item: item["surprise_bits"],
        reverse=True,
    )
    return [
        [
            item["position"],
            item["display_text"],
            round(item["left_surprise"], 3),
            round(item["right_surprise"], 3),
            round(item["surprise_bits"], 3),
            item["left_top"],
            item["right_top"],
        ]
        for item in ranked[:limit]
    ]


def configuration(run: dict | None) -> dict:
    """Everything about a run a reader might have changed between the two.

    Flat, and every value a string, because the only thing done with it is to
    put the two side by side and show the lines that differ. A value that
    reads the same in both runs did not change, whatever its type.
    """

    if not run:
        return {}
    settings = run.get("settings") or {}
    steering = settings.get("steering")
    if not steering:
        steered = "off"
    else:
        state = "on" if steering.get("enabled") and steering.get("strength") else "off"
        steered = (
            f"{state}, layer {steering.get('layer')}, "
            f"strength {steering.get('strength'):g}, "
            f"vector {str(steering.get('vector_id') or '')[:8] or 'inline'}"
        )
    reading = {
        "Filled by": "Writing a reply" if run["kind"] == REPLY else "Measuring fixed text",
        "Model": run.get("model_id") or "—",
        # Which reading of those weights this was. The runtime numbers its
        # loads because one repository ID can be two snapshots: re-downloaded
        # between the runs, the weights can differ with every other setting
        # identical, and a table reporting no difference would hand the
        # reader a gap that came from somewhere it says nothing about.
        "Weights load": run.get("load_id") or "—",
        "Device": run.get("device_name") or "—",
        "Weights": run.get("precision") or "—",
        "Steering": steered,
    }
    # What each run was actually given, not only how it was configured. A box
    # edited between filling A and filling B changes the experiment's variable
    # without touching a single control, and a comparison that showed only the
    # controls would hand the reader a gap to attribute to the wrong thing -
    # worst of all for a measurement, where two contexts leave the passage
    # lining up token for token and every position looking comparable.
    if run["kind"] == REPLY:
        reading["Prompt"] = run.get("prompt") or "—"
    else:
        reading["Context"] = run.get("prompt") or "—"
        reading["Measured text"] = run.get("text") or "—"
    if run["kind"] == REPLY:
        # The system prompt is only in the reading for a reply. A measurement
        # is a fixed passage read as it stands, and there is nowhere in that
        # pass for a system message to go; naming one here would have the
        # table report a difference the two runs never saw.
        reading |= {
            "System prompt": settings.get("system_prompt") or "—",
            "Temperature": f"{float(settings.get('temperature', 0)):g}",
            "Top-p": f"{float(settings.get('top_p', 1)):g}",
            "Top-k": f"{int(settings.get('top_k', 0))}",
            "Skip top choice below": f"{float(settings.get('skip_top_below', 0)):g}",
            "Maximum new tokens": f"{int(settings.get('max_new_tokens', 0))}",
            "Seed": f"{int(settings.get('seed', 0))}",
            "Assistant prefill": settings.get("assistant_prefill") or "—",
            "Thinking mode": thinking_mode(settings),
        }
    else:
        # What the pass did, not what the box asked for. A model with no chat
        # template reads the context as ordinary characters however the box
        # is ticked, and so does an empty context; a label taken from the
        # checkbox would show that tick as an experimental difference neither
        # run received.
        reading |= {"Context read as": context_framing(run)}
    return reading


def thinking_mode(settings: dict) -> str:
    """The reasoning mode a reply really ran under.

    A checkpoint that cannot switch reports none, whatever the control said,
    so a table showing the request would claim two runs thought alike when
    one of them ignored the setting. Where a mode was asked for and none came
    back, that is what the row says.
    """

    applied = settings.get("thinking_mode")
    if applied:
        return str(applied)
    asked = settings.get("requested_thinking_mode") or "default"
    if asked != "default":
        return f"{asked} requested; this model cannot switch"
    return "model default"


def context_framing(run: dict | None) -> str:
    """How a measurement's context was really read."""

    settings = (run or {}).get("settings") or {}
    if not settings.get("use_chat_template"):
        return "plain text"
    if not (run or {}).get("prompt"):
        return "plain text (no context to frame)"
    if settings.get("chat_template_missing"):
        return "plain text (this model has no chat template)"
    return "a chat message"


CONFIGURATION_HEADERS = ["Setting", "A", "B"]


# How much of a prompt or a passage a table cell shows. Whether two runs
# differ is always decided on the whole value; this is only what is drawn.
CELL_LENGTH = 160


def cell(value: str) -> str:
    """One line of ``value``, short enough for a table cell.

    Line breaks and tabs are drawn rather than collapsed, and a value with
    nothing but spaces in it has those drawn too. A measurement of four
    spaces against one of a tab is a real experiment - it is the one the
    whitespace passages exist for - and flattening both to nothing would put
    two identical-looking cells in a row that exists to say they differ.
    Runs of spaces inside ordinary prose are still collapsed, which is what
    keeps a pasted paragraph readable at this width.
    """

    shown = (value or "").replace("\t", "⇥")
    for break_ in ("\r\n", "\n", "\r"):
        shown = shown.replace(break_, "↵")
    if shown and not shown.strip():
        shown = shown.replace(" ", "␠")
    else:
        shown = " ".join(shown.split())
    if len(shown) <= CELL_LENGTH:
        return shown or "—"
    return shown[: CELL_LENGTH - 1].rstrip() + "…"


def configuration_rows(left, right, *, differences_only: bool = True) -> list[list]:
    """The two configurations beside each other, by default only where they differ.

    Compared whole and drawn short: two prompts that agree for the first
    hundred words and part in the last are a difference, and a row that
    compared what the cell shows would call them the same.
    """

    here, there = configuration(left), configuration(right)
    rows = []
    for key in list(here) + [key for key in there if key not in here]:
        mine, yours = here.get(key, "—"), there.get(key, "—")
        if differences_only and mine == yours:
            continue
        rows.append([key, cell(mine), cell(yours)])
    return rows


def describe(run: dict | None, slot: str) -> str:
    """One line naming what is in a slot, for the heading above its strip."""

    if not run:
        return f"**{slot}** · {EMPTY_SLOT}"
    summary = summarize(run["metrics"])
    kind = "reply" if run["kind"] == REPLY else "measurement"
    return (
        f"**{slot}** · {run.get('model_id') or 'unknown model'} · {kind} · "
        f"{summary['token_count']:,} scored tokens · "
        f"perplexity {summary['perplexity']:,.1f} · "
        f"mean surprise {summary['mean_surprise_bits']:.2f} bits"
    )


def reading(left: dict | None, right: dict | None) -> dict:
    """Everything the comparison view draws, from the two runs it draws it from."""

    if not left or not right:
        return {}
    here, there = left["metrics"], right["metrics"]
    cross_model = not same_vocabulary(left, right)
    # Token IDs line two runs up only where an ID identifies the same thing
    # *and* the two ran over the same tokens. Two replies from one vocabulary
    # are that case: they share a prompt, part somewhere in the answer, and
    # matching on IDs is what keeps a token that merely decodes alike from
    # being subtracted across a divergence that already happened.
    #
    # A measurement is not. The passage is fixed and identical by
    # construction; what can differ is where its tokens fall, because the
    # context is encoded with it and a different framing can pull characters
    # of the passage into the seam token. Those two runs read the same text
    # and would report parting at its first character. So a measurement is
    # always lined up on what the tokens cover, and vocabulary identity goes
    # on deciding the questions it really answers - whether a first choice
    # can be compared by ID, and what the caveats say.
    both_replies = left["kind"] == REPLY and right["kind"] == REPLY
    spans = align(
        here,
        there,
        by_text=cross_model or not both_replies,
        left_text=left.get("decoded") or "",
        right_text=right.get("decoded") or "",
        left_ends=left.get("token_ends"),
        right_ends=right.get("token_ends"),
    )
    left_shared = spans[-1]["left_range"][1] if spans else 0
    right_shared = spans[-1]["right_range"][1] if spans else 0
    # A trailing span holds one run's leftover markers and no shared text.
    # It counts towards how much of each run was accounted for, which is
    # what makes a run complete, and not towards how much the two shared.
    shared_spans = [item for item in spans if not item.get("trailing")]
    # Whether the two runs matched token for token, which is not the same
    # question as whether they spent the same number of tokens: one
    # tokenizer's "a" + "bc" against another's "ab" + "c" is two against two
    # over one span, and neither the tokens nor their boundaries agree.
    token_for_token = bool(shared_spans) and all(
        item["one_to_one"] for item in shared_spans
    )
    scored = [item for item in spans if item["scored"]]
    # A span has two first choices to put side by side only when it is one
    # token against one and both runs offered a candidate there. That set is
    # the denominator as well as the numerator: counting a span with nothing
    # to compare would read as "the choice held here".
    pairable = [item for item in scored if item["comparable_choice"]]
    if cross_model:
        # Across two vocabularies the decoded text is the only reading, so a
        # candidate whose characters are not established cannot take part.
        pairable = [
            item for item in pairable
            if item["left_top_determined"] and item["right_top_determined"]
        ]
    # Within one vocabulary the IDs decide it; across two there are no IDs to
    # decide it with, and the decoded text is the only reading left - the
    # empty string included, since a special token can decode to nothing and
    # still be a different choice from a token that decodes to something.
    if cross_model:
        # What the choice would write, and whether it would end the response.
        # Every hidden token writes nothing, so the text alone cannot tell a
        # model that wanted to stop from one that wanted a padding or control
        # marker and would have carried on writing.
        changed = [
            item for item in pairable
            if (item["left_top"], item["left_top_stops"])
            != (item["right_top"], item["right_top_stops"])
        ]
    else:
        changed = [
            item for item in pairable if item["left_top_id"] != item["right_top_id"]
        ]
    widest = max(scored, key=lambda item: item["surprise_bits"], default=None)
    # One vocabulary, one passage, and still a different set of tokens: the
    # two contexts framed it differently. Worth saying, since the reader is
    # looking at two token strips of unequal length over identical text.
    recut = bool(
        not cross_model
        and not both_replies
        and any(not item["one_to_one"] for item in shared_spans)
    )
    return {
        "shared": left_shared,
        "recut": recut,
        "left_shared": left_shared,
        "right_shared": right_shared,
        "spans": len(shared_spans),
        "token_for_token": token_for_token,
        "cross_model": cross_model,
        "left_count": len(here),
        "right_count": len(there),
        "complete": left_shared == len(here) and right_shared == len(there),
        "readings": spans,
        "caveats": _caveats(left, right, cross_model, recut),
        "mean_gap_bits": (
            sum(item["surprise_bits"] for item in scored) / len(scored) if scored else 0.0
        ),
        "widest_gap_bits": widest["surprise_bits"] if widest else 0.0,
        "widest_position": widest["position"] if widest else 0,
        "top_choice_changed": len(changed),
        "choices_compared": len(pairable),
        "compared": len(scored),
        "left_summary": summarize(here),
        "right_summary": summarize(there),
    }


def same_vocabulary(left: dict, right: dict) -> bool:
    """Whether the two runs can be lined up token for token.

    Only a shared vocabulary makes a token ID mean the same thing in both
    runs. The repository ID does not establish that - the same ID
    re-downloaded can bring in a newer revision, which is why the runtime
    numbers its loads rather than trusting the ID - and the load number does
    not either, since re-reading one checkpoint at another precision makes a
    new load out of the same vocabulary. So a recorded fingerprint of the
    vocabulary itself decides it where both runs carry one, and the model ID
    is the fallback for runs made before that was recorded.
    """

    here, there = left.get("tokenizer"), right.get("tokenizer")
    if here and there:
        return here == there
    return (left.get("model_id") or "") == (right.get("model_id") or "")


def _caveats(left: dict, right: dict, cross_model: bool, recut: bool = False) -> list[str]:
    """What a reader has to know before believing the numbers.

    Every one of these is something the run itself recorded and the strips
    cannot show. A seam the tokenizer could not confirm is the sharpest: the
    Score text tab says so plainly, and a comparison that dropped the warning
    would present two guessed boundaries as an exact difference.
    """

    notes = []
    if recut:
        notes.append(
            "The two runs cut the passage into different tokens - the context "
            "in front of it is encoded with it, so a different framing moves "
            "the boundaries - and they are lined up on the text those tokens "
            "cover rather than on the tokens themselves."
        )
    if cross_model:
        notes.append(CROSS_MODEL_CAVEAT)
        if (left.get("model_id") or "") == (right.get("model_id") or ""):
            # Same name, different vocabulary: the repository was fetched
            # again between the two runs and came back changed. Without
            # saying so, the reader is looking at two runs of what the table
            # calls one model being lined up as though they were two.
            notes.append(
                "Both runs name the same model, but the two were measured "
                "with different vocabularies - the repository was downloaded "
                "again between them and came back changed - so their token "
                "IDs no longer mean the same thing."
            )
    for run, slot in ((left, "A"), (right, "B")):
        settings = (run or {}).get("settings") or {}
        if run and run["kind"] == MEASUREMENT and settings.get("seam_verified") is False:
            notes.append(
                f"Slot {slot}'s tokenizer could not confirm where its context "
                "ends, so the boundary between it and the measured passage may "
                "sit a token off. Every probability is the whole passage's own; "
                "what is uncertain is which side of the line its first token "
                "fell on."
            )
    framings = {
        slot: context_framing(run)
        for run, slot in ((left, "A"), (right, "B"))
        if run and run["kind"] == MEASUREMENT
    }
    fell_back = {
        slot: framing for slot, framing in framings.items() if framing.startswith("plain text (")
    }
    for slot, framing in fell_back.items():
        notes.append(
            f"Slot {slot} asked for the context to be read as a chat message "
            f"and got {framing[:-1].replace('plain text (', 'plain text: ')}."
        )
    return notes


def headline(reading: dict, left: dict, right: dict) -> str:
    """What the comparison found, said once, above the strips."""

    if not reading:
        return "Fill both slots to compare them."
    left_shared, right_shared = reading["left_shared"], reading["right_shared"]
    left_count, right_count = reading["left_count"], reading["right_count"]
    spans = reading["spans"]
    # Two models can spend different numbers of tokens on the same text, so
    # what they share is a stretch of characters and a count of comparable
    # spans, not one token count that describes both runs.
    if reading["complete"] and reading["token_for_token"] and left_shared == right_shared:
        where = (
            f"Both runs read the same {left_shared:,} token"
            f"{'' if left_shared == 1 else 's'}."
        )
    elif reading["complete"]:
        where = (
            f"Both runs read the same text, A in {left_shared:,} tokens and B "
            f"in {right_shared:,}, lining up in {spans:,} comparable span"
            f"{'' if spans == 1 else 's'}."
        )
    elif spans:
        where = (
            f"The two runs agreed for {spans:,} span"
            f"{'' if spans == 1 else 's'} — A's first {left_shared:,} token"
            f"{'' if left_shared == 1 else 's'}, B's first {right_shared:,} — "
            f"then parted: A ran to {left_count:,}, B to {right_count:,}. "
            "Only the shared text is compared; after the split the two runs "
            "are reading different things."
        )
    else:
        where = (
            "The two runs parted at their first token, so there is nothing to "
            f"compare: A ran to {left_count:,} tokens, B to {right_count:,}."
        )
    for note in reading["caveats"]:
        where = f"{where} {note}"
    if not reading["compared"]:
        return where
    found = (
        f"{where} Mean gap {reading['mean_gap_bits']:.2f} bits, widest "
        f"{reading['widest_gap_bits']:.2f} bits at span "
        f"{reading['widest_position']:,}."
    )
    if not reading["choices_compared"]:
        return (
            f"{found} No span was one token against one, so there were no "
            "first choices to put side by side."
        )
    return (
        f"{found} The top choice changed at {reading['top_choice_changed']:,} "
        f"of the {reading['choices_compared']:,} span"
        f"{'' if reading['choices_compared'] == 1 else 's'} where both runs "
        "spent a single token."
    )


def gap_metrics(reading: dict) -> list[dict]:
    """The gap per aligned span, shaped for the surprise chart to draw."""

    return [
        {"position": item["position"], "surprise_bits": item["surprise_bits"], "scored": True}
        for item in reading.get("readings", ())
        if item["scored"]
    ]


def _portable_settings(settings: dict) -> dict:
    """One run's settings with its steering vector written out in full.

    A run holds a reference to a vector stored once beside the conversation
    library, which is what keeps a streaming response from copying a few
    thousand numbers per token. That reference resolves against one machine's
    asset directory and nothing else, so an export carrying it would promise
    the run's whole configuration and deliver an identifier - and lose the
    vector that defined the experiment the moment the file moves or the
    assets are cleaned up. ``trace_to_json`` expands it for the same reason.

    A vector that cannot be resolved leaves the reference where it is: an
    export missing one number is worth more than a download button that
    fails, and the reference at least names what is missing.
    """

    value = settings.get("steering")
    if value is None:
        return dict(settings)
    from chatlab.steering import SteeringError, expand

    try:
        return dict(settings, steering=expand(value))
    except (SteeringError, OSError):
        return dict(settings)


def export(left: dict | None, right: dict | None, reading: dict) -> dict:
    """The whole comparison as one document: both runs and what was found."""

    def side(run):
        if not run:
            return None
        return {
            "kind": run["kind"],
            "model_id": run.get("model_id"),
            "load_id": run.get("load_id"),
            "device_name": run.get("device_name"),
            "precision": run.get("precision"),
            "prompt": run.get("prompt", ""),
            "text": run.get("text", ""),
            "settings": _portable_settings(run.get("settings") or {}),
            "seconds": run.get("seconds"),
            "summary": summarize(run["metrics"]),
            "tokens": run["metrics"],
        }

    return {
        "schema_version": 1,
        "a": side(left),
        "b": side(right),
        "comparison": {
            key: value
            for key, value in reading.items()
            if key not in ("readings", "left_summary", "right_summary")
        },
        "caveats": list(reading.get("caveats", ())),
        "aligned_spans": [
            {
                "position": item["position"],
                "text": item["text"],
                "a_tokens": list(item["left_range"]),
                "b_tokens": list(item["right_range"]),
                "a_surprise_bits": item["left_surprise"],
                "b_surprise_bits": item["right_surprise"],
                "gap_bits": item["surprise_bits"] if item["scored"] else None,
                "a_top_choice": item["left_top"],
                "a_top_choice_id": item["left_top_id"],
                "b_top_choice": item["right_top"],
                "b_top_choice_id": item["right_top_id"],
            }
            for item in reading.get("readings", ())
        ],
    }
