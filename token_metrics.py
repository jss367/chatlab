"""Token probability calculations shared by the app and tests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np


LN2 = math.log(2.0)


@dataclass(frozen=True)
class Candidate:
    token_id: int
    text: str
    probability: float


@dataclass(frozen=True)
class TokenMetric:
    position: int
    token_id: int
    text: str
    display_text: str
    category: str
    raw_rank: int
    raw_probability: float
    sampling_probability: float
    surprise_bits: float
    probability_mass_above: float
    entropy_bits: float
    top1_margin: float
    sampling_shift_bits: float
    top_candidates: list[Candidate]
    scored: bool = True
    segment: str = "response"

    def to_dict(self) -> dict:
        return asdict(self)


# Fills are deliberately light in both themes: the token strip paints them
# behind body text, so the ink is pinned dark and every step clears 4.5:1
# against it. Steps come from the reference data-visualization palette.
SEQUENTIAL_FILLS = ("#cde2fb", "#9ec5f4", "#6da7ec", "#5598e7", "#3987e5")
DIVERGING_FILLS = ("#e34948", "#f09b9a", "#f0efec", "#9ec5f4", "#3987e5")
UNSCORED_LABEL = "Not predicted"
UNSCORED_FILL = "#c3c2b7"

RANK_LABELS = ("Top choice", "Top 5", "Top 20", "Rank 21–100", "Rank 101+")

CATEGORY_COLORS = {
    label: color for label, color in zip(RANK_LABELS, SEQUENTIAL_FILLS)
} | {UNSCORED_LABEL: UNSCORED_FILL}


@dataclass(frozen=True)
class ColorScale:
    """An ordered set of buckets used to paint tokens by one measurement."""

    name: str
    field: str
    labels: tuple[str, ...]
    fills: tuple[str, ...]
    edges: tuple[float, ...]
    caption: str

    def bucket(self, value: float) -> str:
        for label, edge in zip(self.labels, self.edges):
            if value < edge:
                return label
        return self.labels[-1]

    @property
    def color_map(self) -> dict[str, str]:
        return dict(zip(self.labels, self.fills)) | {UNSCORED_LABEL: UNSCORED_FILL}


COLOR_SCALES: dict[str, ColorScale] = {
    scale.name: scale
    for scale in (
        ColorScale(
            name="Raw rank",
            field="raw_rank",
            labels=RANK_LABELS,
            fills=SEQUENTIAL_FILLS,
            edges=(1.5, 5.5, 20.5, 100.5),
            caption="Where the token sat in the model's unmodified distribution. Darker is further down the list.",
        ),
        ColorScale(
            name="Surprise",
            field="surprise_bits",
            labels=(
                "Under 1 bit",
                "1–3 bits",
                "3–6 bits",
                "6–10 bits",
                "Over 10 bits",
            ),
            fills=SEQUENTIAL_FILLS,
            edges=(1.0, 3.0, 6.0, 10.0),
            caption="How unexpected this token was. Darker is more surprising.",
        ),
        ColorScale(
            name="Entropy",
            field="entropy_bits",
            labels=(
                "Decided (<0.5)",
                "Narrow (0.5–1.5)",
                "Open (1.5–3)",
                "Wide (3–5)",
                "Very wide (5+)",
            ),
            fills=SEQUENTIAL_FILLS,
            edges=(0.5, 1.5, 3.0, 5.0),
            caption="How undecided the model was before it chose. Darker means more of the distribution was in play.",
        ),
        ColorScale(
            name="Sampling shift",
            field="sampling_shift_bits",
            labels=(
                "Strongly suppressed",
                "Suppressed",
                "Unchanged",
                "Boosted",
                "Strongly boosted",
            ),
            fills=DIVERGING_FILLS,
            edges=(-1.0, -0.05, 0.05, 1.0),
            caption="How much temperature, top-k, and top-p moved this token away from the raw model. Red was made less likely, blue more likely.",
        ),
    )
}

DEFAULT_COLOR_SCALE = "Raw rank"


def category_for(metric: dict, scale_name: str = DEFAULT_COLOR_SCALE) -> str:
    """Bucket one token under the requested color scale."""

    if not metric.get("scored", True):
        return UNSCORED_LABEL
    scale = COLOR_SCALES[scale_name]
    return scale.bucket(float(metric[scale.field]))


def normalize_log_probabilities(values: np.ndarray) -> np.ndarray:
    """Return normalized log probabilities from arbitrary logits."""

    logits = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        np.isnan(logits).any()
        or np.isposinf(logits).any()
        or not np.isfinite(logits).any()
    ):
        raise ValueError(
            "Logits must contain at least one finite value and no NaNs or positive infinity."
        )
    maximum = np.max(logits)
    shifted = logits - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def sampling_probabilities(
    raw_log_probabilities: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> np.ndarray:
    """Build the distribution actually used to select the next token.

    Processing order is temperature, top-k, then top-p. A zero temperature
    produces a one-hot greedy distribution.
    """

    raw = np.asarray(raw_log_probabilities, dtype=np.float64).reshape(-1)
    if np.isnan(raw).any() or np.isposinf(raw).any() or not np.isfinite(raw).any():
        raise ValueError(
            "Log probabilities must contain at least one finite value and no NaNs or positive infinity."
        )

    if temperature <= 0:
        result = np.zeros_like(raw)
        result[int(np.argmax(raw))] = 1.0
        return result

    scores = raw / temperature

    if 0 < top_k < scores.size:
        kept = np.argpartition(scores, -top_k)[-top_k:]
        mask = np.ones(scores.size, dtype=bool)
        mask[kept] = False
        scores[mask] = -np.inf

    finite = np.isfinite(scores)
    maximum = np.max(scores[finite])
    probabilities = np.zeros_like(scores)
    probabilities[finite] = np.exp(scores[finite] - maximum)
    probabilities /= probabilities.sum()

    if 0 < top_p < 1:
        descending = np.argsort(-probabilities)
        cumulative = np.cumsum(probabilities[descending])
        keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        keep = descending[:keep_count]
        filtered = np.zeros_like(probabilities)
        filtered[keep] = probabilities[keep]
        probabilities = filtered / filtered.sum()

    return probabilities


def entropy_bits(raw_log_probabilities: np.ndarray) -> float:
    """Shannon entropy of the unmodified next-token distribution, in bits."""

    log_probs = np.asarray(raw_log_probabilities, dtype=np.float64).reshape(-1)
    finite = np.isfinite(log_probs)
    probs = np.exp(log_probs[finite])
    return float(-(probs * log_probs[finite]).sum() / LN2)


def top_margin(raw_probabilities: np.ndarray) -> float:
    """Gap between the model's first and second choice, in probability."""

    probs = np.asarray(raw_probabilities, dtype=np.float64).reshape(-1)
    if probs.size < 2:
        return float(probs.sum())
    top_two = np.partition(probs, -2)[-2:]
    return float(top_two[1] - top_two[0])


def rank_category(rank: int) -> str:
    if rank == 1:
        return "Top choice"
    if rank <= 5:
        return "Top 5"
    if rank <= 20:
        return "Top 20"
    if rank <= 100:
        return "Rank 21–100"
    return "Rank 101+"


def display_token(text: str, fallback: str) -> str:
    """Make whitespace and special tokens visible in the token strip."""

    if not text:
        return f"‹{fallback}›"
    if text == " ":
        return "␠"
    if text.isspace():
        return text.replace(" ", "␠").replace("\n", "↵\n").replace("\t", "⇥")
    return text.replace("\n", "↵\n").replace("\t", "⇥")


def build_metric(
    *,
    position: int,
    token_id: int,
    token_text: str,
    fallback_text: str,
    raw_log_probabilities: np.ndarray,
    sampled_probabilities: np.ndarray,
    decode_token: Callable[[int], str],
    alternatives: int = 8,
    segment: str = "response",
) -> TokenMetric:
    raw_log_probs = np.asarray(raw_log_probabilities, dtype=np.float64).reshape(-1)
    raw_probs = np.exp(raw_log_probs)
    chosen = float(raw_log_probs[token_id])
    rank = int(np.count_nonzero(raw_log_probs > chosen)) + 1

    count = min(alternatives, raw_probs.size)
    top_ids = np.argpartition(raw_probs, -count)[-count:]
    top_ids = top_ids[np.argsort(-raw_probs[top_ids])]
    candidates = [
        Candidate(
            token_id=int(candidate_id),
            text=decode_token(int(candidate_id)),
            probability=float(raw_probs[candidate_id]),
        )
        for candidate_id in top_ids
    ]

    raw_probability = float(raw_probs[token_id])
    sampling_probability = float(sampled_probabilities[token_id])
    return TokenMetric(
        position=position,
        token_id=token_id,
        text=token_text,
        display_text=display_token(token_text, fallback_text),
        category=rank_category(rank),
        raw_rank=rank,
        raw_probability=raw_probability,
        sampling_probability=sampling_probability,
        surprise_bits=-math.log2(max(raw_probability, 1e-300)),
        probability_mass_above=float(raw_probs[raw_log_probs > chosen].sum()),
        entropy_bits=entropy_bits(raw_log_probs),
        top1_margin=top_margin(raw_probs),
        sampling_shift_bits=math.log2(
            max(sampling_probability, 1e-300) / max(raw_probability, 1e-300)
        ),
        top_candidates=candidates,
        segment=segment,
    )


def unscored_metric(
    *,
    position: int,
    token_id: int,
    token_text: str,
    fallback_text: str,
    segment: str = "prompt",
) -> TokenMetric:
    """A token nothing predicted, such as the very first token of a prompt."""

    return TokenMetric(
        position=position,
        token_id=token_id,
        text=token_text,
        display_text=display_token(token_text, fallback_text),
        category=UNSCORED_LABEL,
        raw_rank=0,
        raw_probability=0.0,
        sampling_probability=0.0,
        surprise_bits=0.0,
        probability_mass_above=0.0,
        entropy_bits=0.0,
        top1_margin=0.0,
        sampling_shift_bits=0.0,
        top_candidates=[],
        scored=False,
        segment=segment,
    )


def summarize(metrics: Sequence[dict]) -> dict:
    """Aggregate a run of tokens into the numbers worth reading at a glance."""

    scored = [metric for metric in metrics if metric.get("scored", True)]
    if not scored:
        return {
            "token_count": 0,
            "perplexity": 0.0,
            "mean_surprise_bits": 0.0,
            "median_surprise_bits": 0.0,
            "total_surprise_bits": 0.0,
            "mean_entropy_bits": 0.0,
            "top1_share": 0.0,
            "top5_share": 0.0,
            "peak_surprise_bits": 0.0,
            "peak_position": 0,
        }

    surprise = np.array([metric["surprise_bits"] for metric in scored], dtype=float)
    entropy = np.array([metric["entropy_bits"] for metric in scored], dtype=float)
    ranks = np.array([metric["raw_rank"] for metric in scored], dtype=float)
    peak = int(np.argmax(surprise))
    mean_surprise = float(surprise.mean())
    return {
        "token_count": len(scored),
        "perplexity": float(2.0**mean_surprise),
        "mean_surprise_bits": mean_surprise,
        "median_surprise_bits": float(np.median(surprise)),
        "total_surprise_bits": float(surprise.sum()),
        "mean_entropy_bits": float(entropy.mean()),
        "top1_share": float((ranks == 1).mean()),
        "top5_share": float((ranks <= 5).mean()),
        "peak_surprise_bits": float(surprise[peak]),
        "peak_position": int(scored[peak]["position"]),
    }
