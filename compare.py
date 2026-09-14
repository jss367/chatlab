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

from token_metrics import (
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
GAP_COLORS = dict(zip(GAP_LABELS, SEQUENTIAL_FILLS[:4])) | {
    SPLIT_LABEL: SEQUENTIAL_FILLS[4],
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
    "the same size in two vocabularies."
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
    while here < len(left) and there < len(right):
        start_here, start_there = here, there
        if not by_text:
            if int(left[here]["token_id"]) != int(right[there]["token_id"]):
                break
            here, there = here + 1, there + 1
        else:
            here, there = here + 1, there + 1
            # Extend whichever side has covered fewer characters until both
            # stand at the same place. Running out on either side ends the
            # alignment: there is no boundary left for the other to meet.
            while ends_here[here - 1] != ends_there[there - 1]:
                if ends_here[here - 1] < ends_there[there - 1]:
                    if here >= len(left):
                        break
                    here += 1
                else:
                    if there >= len(right):
                        break
                    there += 1
            if ends_here[here - 1] != ends_there[there - 1]:
                break
        end_here, end_there = ends_here[here - 1], ends_there[there - 1]
        covered = text_here[covered_here:end_here]
        if by_text and covered != text_there[covered_there:end_there]:
            # The two stand at the same offset over different characters, so
            # they were never reading the same thing.
            break
        spans.append(_span(
            left, right, start_here, here, start_there, there, len(spans) + 1, covered,
        ))
        covered_here, covered_there = end_here, end_there
    return spans


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
    left_top = _top_choice(mine[0]) if one_to_one else (None, "")
    right_top = _top_choice(yours[0]) if one_to_one else (None, "")
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
        "right_top": right_top[1],
        "right_top_id": right_top[0],
    }


def _top_choice(metric: dict) -> tuple[int | None, str]:
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
        return None, ""
    first = candidates[0]
    if isinstance(first, dict):
        token_id, text = first.get("token_id"), first.get("text", "")
    else:
        token_id, text = getattr(first, "token_id", None), getattr(first, "text", "")
    return (None if token_id is None else int(token_id)), text


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
        label = (
            gap_category(span["surprise_bits"]) if span["scored"] else UNSCORED_LABEL
        )
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
            "Maximum new tokens": f"{int(settings.get('max_new_tokens', 0))}",
            "Seed": f"{int(settings.get('seed', 0))}",
            "Assistant prefill": settings.get("assistant_prefill") or "—",
            "Thinking mode": settings.get("thinking_mode") or "model default",
        }
    else:
        # What the pass did, not what the box asked for. A model with no chat
        # template reads the context as ordinary characters however the box
        # is ticked, and so does an empty context; a label taken from the
        # checkbox would show that tick as an experimental difference neither
        # run received.
        reading |= {"Context read as": context_framing(run)}
    return reading


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
    """One line of ``value``, short enough for a table cell."""

    flattened = " ".join((value or "").split())
    if len(flattened) <= CELL_LENGTH:
        return flattened or "—"
    return flattened[: CELL_LENGTH - 1].rstrip() + "…"


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
    # Two loads of one model ID share a tokenizer whatever else changed about
    # them - a precision, a steering vector - so their tokens still line up
    # one for one. Two model IDs do not, and are walked by text instead.
    cross_model = (left.get("model_id") or "") != (right.get("model_id") or "")
    spans = align(
        here,
        there,
        by_text=cross_model,
        left_text=left.get("decoded") or "",
        right_text=right.get("decoded") or "",
        left_ends=left.get("token_ends"),
        right_ends=right.get("token_ends"),
    )
    left_shared = spans[-1]["left_range"][1] if spans else 0
    right_shared = spans[-1]["right_range"][1] if spans else 0
    scored = [item for item in spans if item["scored"]]
    # A span has two first choices to put side by side only when it is one
    # token against one and both runs offered a candidate there. That set is
    # the denominator as well as the numerator: counting a span with nothing
    # to compare would read as "the choice held here".
    pairable = [item for item in scored if item["comparable_choice"]]
    # Within one vocabulary the IDs decide it; across two there are no IDs to
    # decide it with, and the decoded text is the only reading left - the
    # empty string included, since a special token can decode to nothing and
    # still be a different choice from a token that decodes to something.
    if cross_model:
        changed = [item for item in pairable if item["left_top"] != item["right_top"]]
    else:
        changed = [
            item for item in pairable if item["left_top_id"] != item["right_top_id"]
        ]
    widest = max(scored, key=lambda item: item["surprise_bits"], default=None)
    return {
        "shared": left_shared,
        "left_shared": left_shared,
        "right_shared": right_shared,
        "spans": len(spans),
        "cross_model": cross_model,
        "left_count": len(here),
        "right_count": len(there),
        "complete": left_shared == len(here) and right_shared == len(there),
        "readings": spans,
        "caveats": _caveats(left, right, cross_model),
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


def _caveats(left: dict, right: dict, cross_model: bool) -> list[str]:
    """What a reader has to know before believing the numbers.

    Every one of these is something the run itself recorded and the strips
    cannot show. A seam the tokenizer could not confirm is the sharpest: the
    Score text tab says so plainly, and a comparison that dropped the warning
    would present two guessed boundaries as an exact difference.
    """

    notes = []
    if cross_model:
        notes.append(CROSS_MODEL_CAVEAT)
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
    if reading["complete"] and left_shared == right_shared:
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
    from steering import SteeringError, expand

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
