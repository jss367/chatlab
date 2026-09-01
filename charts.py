"""Small inline visuals for the token inspector.

Everything renders as plain SVG or HTML so it can be dropped straight into a
Gradio ``HTML`` component. Colors come from CSS custom properties defined in
the app stylesheet, so light and dark themes are each stepped deliberately.
"""

from __future__ import annotations

import html
import math
from typing import Sequence


MAX_BINS = 240
_VIEW_WIDTH = 640.0
_VIEW_HEIGHT = 150.0
_PAD_LEFT = 34.0
_PAD_RIGHT = 10.0
_PAD_TOP = 14.0
_PAD_BOTTOM = 22.0

EMPTY_CHART = (
    '<div class="viz-empty">The surprise trace appears once a response has '
    "more than one token.</div>"
)


def _bin_metrics(metrics: Sequence[dict], bins: int) -> list[dict]:
    """Group tokens into at most ``bins`` buckets, keeping the range in each."""

    size = max(1, math.ceil(len(metrics) / bins))
    grouped: list[dict] = []
    for start in range(0, len(metrics), size):
        chunk = metrics[start : start + size]
        values = [float(metric["surprise_bits"]) for metric in chunk]
        peak = max(range(len(values)), key=values.__getitem__)
        grouped.append(
            {
                "mean": sum(values) / len(values),
                "low": min(values),
                "high": values[peak],
                # Where the bin's maximum actually sits, which is only the last
                # token of the bin by coincidence.
                "peak_position": int(chunk[peak]["position"]),
                "first": int(chunk[0]["position"]),
                "last": int(chunk[-1]["position"]),
            }
        )
    return grouped


def surprise_chart(metrics: Sequence[dict], *, title: str = "Surprise per token") -> str:
    """A line of per-token surprise across the response, in bits."""

    scored = [metric for metric in metrics if metric.get("scored", True)]
    if len(scored) < 2:
        return EMPTY_CHART

    bins = _bin_metrics(scored, MAX_BINS)
    ceiling = max(4.0, math.ceil(max(item["high"] for item in bins)))
    plot_width = _VIEW_WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_height = _VIEW_HEIGHT - _PAD_TOP - _PAD_BOTTOM
    step = plot_width / max(1, len(bins) - 1)

    def x_at(index: int) -> float:
        return _PAD_LEFT + index * step

    def y_at(value: float) -> float:
        return _PAD_TOP + plot_height * (1 - value / ceiling)

    line = " ".join(
        f"{'M' if index == 0 else 'L'}{x_at(index):.1f},{y_at(item['mean']):.1f}"
        for index, item in enumerate(bins)
    )
    band = ""
    if any(item["high"] > item["low"] for item in bins):
        top = " ".join(
            f"{'M' if index == 0 else 'L'}{x_at(index):.1f},{y_at(item['high']):.1f}"
            for index, item in enumerate(bins)
        )
        bottom = " ".join(
            f"L{x_at(index):.1f},{y_at(bins[index]['low']):.1f}"
            for index in range(len(bins) - 1, -1, -1)
        )
        band = f'<path class="viz-band" d="{top} {bottom} Z" />'

    gridlines = "".join(
        f'<line class="viz-grid" x1="{_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y1="{y_at(value):.1f}" y2="{y_at(value):.1f}" />'
        f'<text class="viz-tick" x="{_PAD_LEFT - 6}" y="{y_at(value) + 3.5:.1f}" '
        f'text-anchor="end">{value:g}</text>'
        for value in (0.0, ceiling / 2, ceiling)
    )

    peak = max(range(len(bins)), key=lambda index: bins[index]["high"])
    peak_x = x_at(peak)
    peak_y = y_at(bins[peak]["high"])
    anchor = "end" if peak_x > _VIEW_WIDTH * 0.6 else "start"
    label_x = peak_x + (-7 if anchor == "end" else 7)
    peak_label = (
        f'<circle class="viz-peak-dot" cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="4" />'
        f'<text class="viz-peak-label" x="{label_x:.1f}" y="{max(peak_y - 8, 10):.1f}" '
        f'text-anchor="{anchor}">peak {bins[peak]["high"]:.1f} bits '
        f'at token {bins[peak]["peak_position"]:,}</text>'
    )

    hover = "".join(
        f'<rect class="viz-hit" x="{x_at(index) - step / 2:.1f}" y="{_PAD_TOP}" '
        f'width="{step:.1f}" height="{plot_height:.1f}">'
        f"<title>{_bin_title(item)}</title></rect>"
        for index, item in enumerate(bins)
    )

    binned_note = (
        ""
        if len(bins) == len(scored)
        else f" · {len(scored):,} tokens grouped into {len(bins)} bins, shaded low to high"
    )

    return (
        '<figure class="viz-root" id="surprise-chart">'
        f'<figcaption class="viz-title">{html.escape(title)}'
        f'<span class="viz-sub">bits, {len(scored):,} tokens{binned_note}</span>'
        "</figcaption>"
        f'<svg viewBox="0 0 {_VIEW_WIDTH:g} {_VIEW_HEIGHT:g}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f"{gridlines}{band}"
        f'<path class="viz-line" d="{line}" />'
        f"{peak_label}"
        f'<line class="viz-axis" x1="{_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y1="{y_at(0):.1f}" y2="{y_at(0):.1f}" />'
        f'<text class="viz-tick" x="{_PAD_LEFT}" y="{_VIEW_HEIGHT - 6:g}">'
        f'token {bins[0]["first"]:,}</text>'
        f'<text class="viz-tick" x="{_VIEW_WIDTH - _PAD_RIGHT}" y="{_VIEW_HEIGHT - 6:g}" '
        f'text-anchor="end">token {bins[-1]["last"]:,}</text>'
        f"{hover}</svg></figure>"
    )


def _bin_title(item: dict) -> str:
    span = (
        f"token {item['first']:,}"
        if item["first"] == item["last"]
        else f"tokens {item['first']:,}–{item['last']:,}"
    )
    if item["high"] == item["low"]:
        return html.escape(f"{span} · {item['mean']:.2f} bits")
    return html.escape(
        f"{span} · {item['mean']:.2f} bits average, {item['low']:.2f} to {item['high']:.2f}"
    )


def _tile(value: str, label: str, hint: str) -> str:
    return (
        f'<div class="viz-tile" title="{html.escape(hint)}">'
        f'<div class="viz-value">{html.escape(value)}</div>'
        f'<div class="viz-label">{html.escape(label)}</div></div>'
    )


def summary_tiles(summary: dict, *, note: str = "") -> str:
    """Headline numbers for a scored run of tokens."""

    if not summary.get("token_count"):
        return '<div class="viz-empty">No scored tokens yet.</div>'

    tiles = "".join(
        (
            _tile(
                f"{summary['perplexity']:,.1f}",
                "perplexity",
                "2 raised to the mean surprise. Lower means the text was more predictable to the model.",
            ),
            _tile(
                f"{summary['mean_surprise_bits']:.2f}",
                "mean surprise (bits)",
                f"Median {summary['median_surprise_bits']:.2f} bits, peak {summary['peak_surprise_bits']:.2f} bits at token {summary['peak_position']:,}.",
            ),
            _tile(
                f"{summary['top1_share']:.0%}",
                "were rank 1",
                f"{summary['top5_share']:.0%} of tokens were inside the model's top 5.",
            ),
            _tile(
                f"{summary['mean_entropy_bits']:.2f}",
                "mean entropy (bits)",
                "Average width of the distribution the model chose from.",
            ),
            _tile(
                f"{summary['token_count']:,}",
                "scored tokens",
                f"Total information content {summary['total_surprise_bits']:,.0f} bits.",
            ),
        )
    )
    footer = f'<div class="viz-note">{html.escape(note)}</div>' if note else ""
    return f'<div class="viz-root viz-tiles">{tiles}</div>{footer}'
