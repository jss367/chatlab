"""Paired vocabulary-rank statistics. No model or UI dependencies."""
import numpy as np

FORMAT = "chatlab-steering-comparison-1"


class Aggregate:
    def __init__(self):
        self.n = 0
        self.total = None
        self.candidates = None

    def add(self, steered, comparison, random, random_comparison=None):
        arrays = [np.asarray(a) for a in (steered, comparison, random)]
        if random_comparison is not None:
            arrays.append(np.asarray(random_comparison))
        if any(a.shape != arrays[0].shape or a.ndim != 2 or not np.isfinite(a).all()
               or (a < 1).any() for a in arrays):
            raise ValueError("Rank arrays must have equal block × vocabulary shapes and positive finite ranks.")
        s, b, r = arrays[:3]
        random_baseline = b if random_comparison is None else arrays[3]
        delta = np.log(b) - np.log(s)
        control = np.log(random_baseline) - np.log(r)
        values = np.stack([delta, delta ** 2, control, control ** 2,
                           s <= 50, b <= 50, s <= 10, b <= 10, r <= 50]).astype(float)
        mask = np.minimum.reduce(arrays) <= 50
        if self.total is None:
            self.total = values
            self.candidates = mask
        else:
            if self.total.shape != values.shape:
                raise ValueError("The vocabulary or blocks changed during aggregation.")
            self.total += values
            self.candidates |= mask
        self.n += 1

    def traces(self):
        if not self.n:
            return {}
        tokens = np.flatnonzero(self.candidates.any(axis=0))
        factors = np.exp(self.total[0, :, tokens] / self.n)
        return {str(token): factors[index].tolist() for index, token in enumerate(tokens)}

    def rows(self, block):
        if not self.n:
            return []
        total = self.total[:, block]
        mean = total / self.n

        def t_score(first, second):
            if self.n < 2:
                return np.zeros_like(mean[first])
            variance = np.maximum(0, (total[second] - total[first] ** 2 / self.n) / (self.n - 1))
            # Undefined t statistics remain absent, rather than manufacturing infinity.
            error = np.sqrt(variance / self.n)
            return np.divide(mean[first], error, out=np.full_like(error, np.nan), where=error > 1e-12)

        real, random = t_score(0, 1), t_score(2, 3)
        return [{"id": int(i), "factor": float(np.exp(mean[0, i])),
                 "t": float(real[i]) if np.isfinite(real[i]) and self.n >= 2 else None,
                 "random_t": float(random[i]) if np.isfinite(random[i]) and self.n >= 2 else None,
                 "top50": float(mean[4, i]), "base50": float(mean[5, i]),
                 "top10": float(mean[6, i]), "base10": float(mean[7, i]),
                 "random50": float(mean[8, i])}
                for i in np.flatnonzero(self.candidates[block])]


def select(rows, text="", floor=0.5, sort="Rank improvement (t)", specific=False):
    terms = [term.strip().casefold() for term in text.split(",") if term.strip()]
    chosen = [row for row in rows if max(row["top50"], row["base50"]) >= floor
              and (not terms or any(term in row["token"].casefold() for term in terms))
              and (not specific or (row["t"] is not None and row["random_t"] is not None
                                    and row["t"] > max(2, row["random_t"] + 2)))]
    key = ((lambda r: r["top50"] - r["base50"]) if sort == "Top-50 entry gain" else
           (lambda r: np.log(r["factor"])) if sort == "Rank change factor" else
           (lambda r: r["t"] if r["t"] is not None else 0))
    return sorted(chosen, key=key, reverse=True)
