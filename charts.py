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


# ------------------------------------------------------- layers and attention

EMPTY_LENS = (
    '<div class="viz-empty">Select a token and press <b>Inspect layers</b> to '
    "see what each layer predicted and where the model looked.</div>"
)
EMPTY_ATTENTION = ""

_LENS_HEIGHT = 130.0
_LENS_PAD_LEFT = 38.0
_LENS_PAD_BOTTOM = 24.0

# Attended tokens listed under the strip.
TOP_ATTENDED = 8


def _layer_name(layer: int) -> str:
    return "embeddings" if layer == 0 else f"layer {layer}"


def _token_label(token: dict) -> str:
    """A token's text with whitespace made visible, escaped for HTML."""

    text = token.get("text") or ""
    fallback = token.get("fallback") or str(token.get("token_id", ""))
    if not text:
        shown = f"‹{fallback}›"
    else:
        shown = text.replace("\n", "↵").replace("\t", "⇥")
        if shown.strip() == "":
            shown = shown.replace(" ", "␠")
    return html.escape(shown)


def logit_lens_chart(insight: dict) -> str:
    """How the inspected token's probability grew layer by layer, with a table.

    The line is the probability the actual token would have had if the model
    had stopped after each layer. The table names what each layer would have
    said instead, so a late change of mind is visible as a change of word.
    """

    layers = insight.get("layers") or []
    if not layers:
        return EMPTY_LENS

    token = html.escape(repr(insight.get("token_text", "")))
    last = layers[-1]["layer"]
    # A single row is the model's real output with nothing before it: the
    # runtime found no final norm to read the intermediate layers through.
    output_only = len(layers) == 1
    plot_width = _VIEW_WIDTH - _LENS_PAD_LEFT - _PAD_RIGHT
    plot_height = _LENS_HEIGHT - _PAD_TOP - _LENS_PAD_BOTTOM
    step = plot_width / max(1, len(layers) - 1)

    def x_at(position: int) -> float:
        return _LENS_PAD_LEFT + position * step

    def y_at(value: float) -> float:
        return _PAD_TOP + plot_height * (1 - min(max(value, 0.0), 1.0))

    actual = " ".join(
        f"{'M' if position == 0 else 'L'}{x_at(position):.1f},"
        f"{y_at(row['probability']):.1f}"
        for position, row in enumerate(layers)
    )
    top = " ".join(
        f"{'M' if position == 0 else 'L'}{x_at(position):.1f},"
        f"{y_at(row['top_probability']):.1f}"
        for position, row in enumerate(layers)
    )
    gridlines = "".join(
        f'<line class="viz-grid" x1="{_LENS_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y1="{y_at(value):.1f}" y2="{y_at(value):.1f}" />'
        f'<text class="viz-tick" x="{_LENS_PAD_LEFT - 6}" y="{y_at(value) + 3.5:.1f}" '
        f'text-anchor="end">{value:.0%}</text>'
        for value in (0.0, 0.5, 1.0)
    )
    decided = insight.get("decided_at")
    marker = ""
    if decided is not None:
        position = next(
            (index for index, row in enumerate(layers) if row["layer"] == decided),
            None,
        )
        if position is not None:
            x = x_at(position)
            anchor = "end" if x > _VIEW_WIDTH * 0.6 else "start"
            label_x = x + (-6 if anchor == "end" else 6)
            marker = (
                f'<line class="viz-marker" x1="{x:.1f}" x2="{x:.1f}" '
                f'y1="{_PAD_TOP}" y2="{y_at(0):.1f}" />'
                f'<text class="viz-peak-label" x="{label_x:.1f}" y="{_PAD_TOP + 9:.1f}" '
                f'text-anchor="{anchor}">first choice from {_layer_name(decided)}</text>'
            )
    hover = "".join(
        f'<rect class="viz-hit" x="{x_at(position) - step / 2:.1f}" y="{_PAD_TOP}" '
        f'width="{step:.1f}" height="{plot_height:.1f}">'
        f"<title>{_lens_title(row)}</title></rect>"
        for position, row in enumerate(layers)
    )
    verdict = (
        f"never the first choice before the output; rank {layers[-1]['rank']:,} at the end"
        if decided is None
        else f"first choice from {_layer_name(decided)} onward"
    )
    rows = "".join(
        f"<tr{' class=\"viz-hit-row\"' if row['rank'] == 1 else ''}>"
        f"<td>{html.escape(_layer_name(row['layer']))}</td>"
        f"<td><code>{html.escape(repr(row['top_text']))}</code></td>"
        f"<td>{row['top_probability']:.1%}</td>"
        f"<td>{row['rank']:,}</td>"
        f"<td>{row['probability']:.1%}</td>"
        f"<td>{row['entropy_bits']:.1f}</td></tr>"
        for row in layers
    )
    if output_only:
        chart = (
            '<div class="viz-note">Only the output is shown: this model\'s '
            "intermediate layers could not be read through its final norm.</div>"
        )
    else:
        chart = (
            f'<svg viewBox="0 0 {_VIEW_WIDTH:g} {_LENS_HEIGHT:g}" role="img" '
            f'aria-label="Probability of the token after each layer">'
            f"{gridlines}"
            f'<path class="viz-line viz-line-faint" d="{top}" />'
            f'<path class="viz-line" d="{actual}" />'
            f"{marker}"
            f'<line class="viz-axis" x1="{_LENS_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
            f'y1="{y_at(0):.1f}" y2="{y_at(0):.1f}" />'
            f'<text class="viz-tick" x="{_LENS_PAD_LEFT}" y="{_LENS_HEIGHT - 8:g}">embeddings</text>'
            f'<text class="viz-tick" x="{_VIEW_WIDTH - _PAD_RIGHT}" y="{_LENS_HEIGHT - 8:g}" '
            f'text-anchor="end">output ({_layer_name(last)})</text>'
            f"{hover}</svg>"
            '<div class="viz-note">Dark line: probability of the token that was chosen. '
            "Faint line: probability of whatever each layer liked best. Intermediate "
            "layers are read through the final norm and unembedding.</div>"
        )
    return (
        '<figure class="viz-root" id="logit-lens">'
        f'<figcaption class="viz-title">Logit lens for <code>{token}</code>'
        f'<span class="viz-sub">{html.escape(verdict)}</span></figcaption>'
        f"{chart}"
        '<div class="viz-table-wrap"><table class="viz-table">'
        "<thead><tr><th>Layer</th><th>Would have said</th><th>Prob.</th>"
        "<th>Rank of chosen</th><th>Prob. of chosen</th><th>Entropy</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></figure>"
    )


def _lens_title(row: dict) -> str:
    return html.escape(
        f"{_layer_name(row['layer'])} · chosen token {row['probability']:.1%} "
        f"(rank {row['rank']:,}) · top {row['top_text']!r} {row['top_probability']:.1%}"
    )


def attention_weights(insight: dict, layer: int) -> list[float]:
    """Head-averaged attention from the query, for one layer or all of them.

    ``layer`` counts from 1; 0 means the mean over every layer.
    """

    attention = insight.get("attention") or []
    if not attention:
        return []
    if layer <= 0 or layer > len(attention):
        width = len(attention[0])
        return [
            sum(row[column] for row in attention) / len(attention)
            for column in range(width)
        ]
    return list(attention[layer - 1])


def attention_strip(insight: dict, layer: int = 0) -> str:
    """The tokens the query could see, shaded by how much attention each got.

    The first token of almost any sequence soaks up attention regardless of
    content (the "attention sink"), so shading is scaled to the strongest
    token *after* it and its share is reported in words instead.
    """

    tokens = insight.get("tokens") or []
    weights = attention_weights(insight, layer)
    if not tokens or not weights or len(weights) != len(tokens):
        if insight.get("layers"):
            return (
                '<div class="viz-empty">This model did not return attention '
                "weights, so there is nothing to shade.</div>"
            )
        return EMPTY_ATTENTION

    scale = max(weights[1:], default=weights[0]) or weights[0] or 1.0
    query = len(tokens) - 1
    context_count = sum(1 for token in tokens if token.get("segment") == "prompt")
    spans = []
    for position, (token, weight) in enumerate(zip(tokens, weights)):
        alpha = min(1.0, weight / scale) ** 0.5 * 0.85
        prompt = token.get("segment") == "prompt"
        classes = ["attn-token"]
        if position == 0 and len(tokens) > 1:
            classes.append("attn-first")
        if position == query:
            classes.append("attn-query")
        if prompt:
            classes.append("attn-prompt")
        title = html.escape(
            f"{'prompt' if prompt else 'response'} token "
            f"{position + 1 if prompt else position + 1 - context_count} · "
            f"{weight:.1%} of attention"
            + (
                " · the query: this token's output made the prediction"
                if position == query
                else ""
            )
        )
        spans.append(
            f'<span class="{" ".join(classes)}" title="{title}" '
            f'style="background: rgba(42, 120, 214, {alpha:.2f})">'
            f"{_token_label(token)}</span>"
        )
    predicted = (
        f'<span class="attn-token attn-predicted" title="the token being explained">'
        f"{_token_label({'text': insight.get('token_text', ''), 'fallback': ''})}</span>"
    )

    ranked = sorted(range(len(weights)), key=weights.__getitem__, reverse=True)
    listed = "".join(
        f"<li><code>{_token_label(tokens[position])}</code> "
        f"<span class=\"viz-sub\">{weights[position]:.1%}</span></li>"
        for position in ranked[:TOP_ATTENDED]
    )
    where = "mean of all layers" if layer <= 0 or layer > len(insight.get("attention") or []) else _layer_name(layer)
    sink = (
        f" The first token takes {weights[0]:.0%} of the attention, the usual sink; "
        "shading is scaled to the strongest token after it."
        if len(weights) > 1
        else ""
    )
    return (
        '<div class="viz-root" id="attention-view">'
        f'<div class="viz-title">Attention while predicting '
        f'<code>{html.escape(repr(insight.get("token_text", "")))}</code>'
        f'<span class="viz-sub">{html.escape(where)}, averaged over heads</span></div>'
        f'<div class="attn-strip">{"".join(spans)}{predicted}</div>'
        f'<div class="viz-note">Dashed outline: the query, whose output made the prediction. '
        f'Solid outline: the token being explained.{html.escape(sink)}</div>'
        f'<ol class="attn-top">{listed}</ol></div>'
    )
