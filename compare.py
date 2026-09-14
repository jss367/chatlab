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
    """Which bucket a shared token's surprise gap falls in."""

    for label, edge in zip(GAP_LABELS, GAP_EDGES):
        if abs(delta) < edge:
            return label
    return GAP_LABELS[-1]


def shared_prefix(left, right, *, by_text: bool = False) -> int:
    """How many leading tokens the two runs have in common.

    Within one vocabulary, counted on token IDs: two different tokens can
    decode to the same characters, and a comparison that called them equal
    would go on subtracting measurements taken in contexts that had already
    parted.

    Across two vocabularies there are no IDs to count on. A token ID means
    nothing outside the tokenizer that issued it - ID 4192 in one model and
    ID 4192 in another stand for unrelated text - so two runs from different
    models are matched on the text each token stands for and on nothing else.
    That also handles the other half of the problem: one passage can be cut
    into different tokens by two models, and the first position whose text
    disagrees is where the two runs stop describing the same thing, whatever
    the characters after it are. Matching by text stops there rather than
    subtracting one model's reading of half a word from another's reading of
    a whole one.
    """

    count = 0
    for here, there in zip(left or (), right or ()):
        if by_text:
            if (here.get("text") or "") != (there.get("text") or ""):
                break
        elif int(here["token_id"]) != int(there["token_id"]):
            break
        count += 1
    return count


def gaps(left, right, shared: int) -> list[dict]:
    """The surprise gap at every shared position, with both sides' readings.

    Positions where either side has no measurement - the first token of a
    sequence, which nothing predicted - are given no gap; the caller paints
    them as unscored rather than as agreement.
    """

    readings = []
    for index in range(shared):
        here, there = left[index], right[index]
        scored = here.get("scored", True) and there.get("scored", True)
        readings.append({
            "position": index + 1,
            "token_id": int(here["token_id"]),
            "text": here.get("display_text") or here.get("text") or "",
            "scored": scored,
            "surprise_bits": (
                abs(float(here["surprise_bits"]) - float(there["surprise_bits"]))
                if scored
                else 0.0
            ),
            "left_surprise": float(here["surprise_bits"]) if scored else None,
            "right_surprise": float(there["surprise_bits"]) if scored else None,
            "left_top": _top_text(here),
            "right_top": _top_text(there),
        })
    return readings


def _top_text(metric: dict) -> str:
    """What this run's model would have written here, left to itself."""

    candidates = metric.get("top_candidates") or ()
    if not candidates:
        return ""
    first = candidates[0]
    return first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")


def strip(metrics, shared: int, readings) -> list[tuple[str, str]]:
    """One run's tokens, colored by how far the other run sat from them."""

    painted = []
    for index, metric in enumerate(metrics or ()):
        if index >= shared:
            label = SPLIT_LABEL
        elif not readings[index]["scored"]:
            label = UNSCORED_LABEL
        else:
            label = gap_category(readings[index]["surprise_bits"])
        painted.append((metric["display_text"], label))
    return painted


DIVERGENCE_HEADERS = [
    "#",
    "Token",
    "A surprise",
    "B surprise",
    "Δ bits",
    "A would write",
    "B would write",
]


# Enough rows to show a pattern, few enough to read. The whole comparison is
# in the export beside the table.
DIVERGENCE_ROWS = 25


def divergence_rows(readings, limit: int = DIVERGENCE_ROWS) -> list[list]:
    """The shared positions the two runs read most differently, widest first."""

    ranked = sorted(
        (item for item in readings if item["scored"]),
        key=lambda item: item["surprise_bits"],
        reverse=True,
    )
    return [
        [
            item["position"],
            item["text"],
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
        reading |= {
            "Context read as": (
                "a chat message" if settings.get("use_chat_template") else "plain text"
            ),
        }
    return reading


CONFIGURATION_HEADERS = ["Setting", "A", "B"]


def configuration_rows(left, right, *, differences_only: bool = True) -> list[list]:
    """The two configurations beside each other, by default only where they differ."""

    here, there = configuration(left), configuration(right)
    rows = []
    for key in list(here) + [key for key in there if key not in here]:
        mine, yours = here.get(key, "—"), there.get(key, "—")
        if differences_only and mine == yours:
            continue
        rows.append([key, mine, yours])
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
    # them - a precision, a steering vector - so their IDs still line up. Two
    # model IDs do not, and are aligned by text instead.
    cross_model = (left.get("model_id") or "") != (right.get("model_id") or "")
    shared = shared_prefix(here, there, by_text=cross_model)
    readings = gaps(here, there, shared)
    scored = [item for item in readings if item["scored"]]
    changed = [
        item for item in scored
        if item["left_top"] and item["left_top"] != item["right_top"]
    ]
    widest = max(scored, key=lambda item: item["surprise_bits"], default=None)
    return {
        "shared": shared,
        "cross_model": cross_model,
        "left_count": len(here),
        "right_count": len(there),
        "complete": shared == len(here) == len(there),
        "readings": readings,
        "mean_gap_bits": (
            sum(item["surprise_bits"] for item in scored) / len(scored) if scored else 0.0
        ),
        "widest_gap_bits": widest["surprise_bits"] if widest else 0.0,
        "widest_position": widest["position"] if widest else 0,
        "top_choice_changed": len(changed),
        "compared": len(scored),
        "left_summary": summarize(here),
        "right_summary": summarize(there),
    }


def headline(reading: dict, left: dict, right: dict) -> str:
    """What the comparison found, said once, above the strips."""

    if not reading:
        return "Fill both slots to compare them."
    shared, left_count, right_count = (
        reading["shared"], reading["left_count"], reading["right_count"]
    )
    if reading["complete"]:
        where = (
            f"Both runs read the same {shared:,} token"
            f"{'' if shared == 1 else 's'}."
        )
    elif shared:
        where = (
            f"The two runs agreed for {shared:,} token"
            f"{'' if shared == 1 else 's'}, then parted: "
            f"A ran to {left_count:,}, B to {right_count:,}. "
            "Only the shared tokens are compared - after the split the two "
            "runs are reading different text."
        )
    else:
        where = (
            f"The two runs parted at the first token, so there is nothing to "
            f"compare: A ran to {left_count:,} tokens, B to {right_count:,}."
        )
    if reading["cross_model"]:
        where = f"{where} {CROSS_MODEL_CAVEAT}"
    if not reading["compared"]:
        return where
    return (
        f"{where} Mean gap {reading['mean_gap_bits']:.2f} bits, widest "
        f"{reading['widest_gap_bits']:.2f} bits at token "
        f"{reading['widest_position']:,}. The top choice changed at "
        f"{reading['top_choice_changed']:,} of {reading['compared']:,} "
        "compared tokens."
    )


def gap_metrics(reading: dict) -> list[dict]:
    """The gap per shared token, shaped for the surprise chart to draw."""

    return [
        {"position": item["position"], "surprise_bits": item["surprise_bits"], "scored": True}
        for item in reading.get("readings", ())
        if item["scored"]
    ]


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
            "settings": run.get("settings") or {},
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
        "shared_tokens": [
            {
                "position": item["position"],
                "token_id": item["token_id"],
                "text": item["text"],
                "a_surprise_bits": item["left_surprise"],
                "b_surprise_bits": item["right_surprise"],
                "gap_bits": item["surprise_bits"] if item["scored"] else None,
                "a_top_choice": item["left_top"],
                "b_top_choice": item["right_top"],
            }
            for item in reading.get("readings", ())
        ],
    }
