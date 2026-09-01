"""Token probability calculations shared by the app and tests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


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
    top_candidates: list[Candidate]

    def to_dict(self) -> dict:
        return asdict(self)


CATEGORY_COLORS = {
    "Top choice": "#34d399",
    "Top 5": "#a3e635",
    "Top 20": "#facc15",
    "Rank 21–100": "#fb923c",
    "Rank 101+": "#f472b6",
}


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
    return TokenMetric(
        position=position,
        token_id=token_id,
        text=token_text,
        display_text=display_token(token_text, fallback_text),
        category=rank_category(rank),
        raw_rank=rank,
        raw_probability=raw_probability,
        sampling_probability=float(sampled_probabilities[token_id]),
        surprise_bits=-math.log2(max(raw_probability, 1e-300)),
        probability_mass_above=float(raw_probs[raw_log_probs > chosen].sum()),
        top_candidates=candidates,
    )
