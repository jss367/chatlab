"""Small inline visuals for the token inspector.

Everything renders as plain SVG or HTML so it can be dropped straight into a
Gradio ``HTML`` component. Colors come from CSS custom properties defined in
the app stylesheet, so light and dark themes are each stepped deliberately.
"""

from __future__ import annotations

import html
import json
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


def comparison_tiles(reading: dict) -> str:
    """Headline numbers for two runs read against each other."""

    if not reading:
        return '<div class="viz-empty">Fill both slots to compare them.</div>'
    if not reading["compared"]:
        return (
            '<div class="viz-empty">The two runs share no measured tokens, so '
            "there is nothing to compare.</div>"
        )
    left, right = reading["left_summary"], reading["right_summary"]
    tiles = "".join(
        (
            _tile(
                f"{reading['spans']:,}",
                "spans shared",
                f"A ran to {reading['left_count']:,} tokens, B to {reading['right_count']:,}; "
                f"they line up over A's first {reading['left_shared']:,} and B's first "
                f"{reading['right_shared']:,}. A span is a stretch of characters both runs "
                "covered, which is one token each where the two share a tokenizer.",
            ),
            _tile(
                f"{reading['mean_gap_bits']:.2f}",
                "mean gap (bits)",
                "Average distance between the two runs' surprise over a shared span.",
            ),
            _tile(
                f"{reading['widest_gap_bits']:.2f}",
                "widest gap (bits)",
                f"At span {reading['widest_position']:,}.",
            ),
            _tile(
                (
                    f"{reading['top_choice_changed'] / reading['choices_compared']:.0%}"
                    if reading["choices_compared"]
                    else "—"
                ),
                "top choice changed",
                (
                    f"{reading['top_choice_changed']:,} of the "
                    f"{reading['choices_compared']:,} spans where both runs spent a "
                    "single token had a different first choice. A span of several "
                    "tokens against one has no pair of first choices to compare, so "
                    "it is not counted either way."
                    if reading["choices_compared"]
                    else "No span was one token against one, so there were no first "
                    "choices to put side by side."
                ),
            ),
            _tile(
                f"{left['perplexity']:,.1f} → {right['perplexity']:,.1f}",
                "perplexity A → B",
                f"Mean surprise {left['mean_surprise_bits']:.2f} → "
                f"{right['mean_surprise_bits']:.2f} bits, over each run's whole output.",
            ),
        )
    )
    return f'<div class="viz-root viz-tiles">{tiles}</div>'


# ------------------------------------------------------- layers and attention

EMPTY_LENS = (
    '<div class="viz-empty">Select a token and press <b>Inspect layers</b> to '
    "see what each layer predicted and where the model looked.</div>"
)
EMPTY_JACOBIAN = (
    '<div class="viz-empty">Import a fitted lens, select a token, and press '
    '<b>Inspect layers</b> to read concepts after that token.</div>'
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
            "intermediate layers could not be read the way it reads its own "
            "output.</div>"
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


def _jacobian_slice(insight: dict) -> str:
    """The layer × position grid: each block's top lens token at every fed position.

    Blocks run down the table and positions across it, ending at the selected
    token, whose column is marked. The last row is the model's own next-token
    prediction at each position, which needs no lens. With a token pinned,
    each cell also carries that token's vocabulary rank and is shaded by it,
    rank 1 darkest on a logarithmic scale. A cell names its token in a JSON
    data attribute so a click on it can pin that token (see JACOBIAN_JS).
    """
    data = insight.get("slice") or {}
    tokens = data.get("tokens") or []
    layers = data.get("layers") or []
    if not tokens or not layers:
        return ""
    selected = insight.get("index")
    pinned = insight.get("pinned_text") is not None
    maximum = max(2, int(insight.get("vocab_size") or 2))

    def heat(rank) -> str:
        if not pinned or rank is None:
            return ""
        level = 1.0 - math.log(max(1, rank)) / math.log(maximum)
        return f' style="--jl-heat:{max(0.0, min(1.0, level)):.2f}"'

    def cell(item: dict, position: int, block: str) -> str:
        text = item.get("text", "")
        shown = html.escape(repr(text))
        rank = item.get("pinned_rank")
        tip = f"{block}, position {position + 1}: top token {text!r}, score {item.get('score', 0.0):.3f}"
        if rank is not None:
            tip += f"; pinned rank {rank:,}, score {item.get('pinned_score', 0.0):.3f}"
        classes = "jl-cell" + (" jl-selected" if position == selected else "")
        badge = f"<sup>{rank:,}</sup>" if rank is not None else ""
        return (
            f'<td class="{classes}" data-token="{html.escape(json.dumps(text), quote=True)}"'
            f' data-token-id="{int(item.get("token_id", -1))}"'
            f'{heat(rank)} title="{html.escape(tip)}"><code>{shown}</code>{badge}</td>'
        )

    head = []
    for token in tokens:
        marked = " jl-selected" if token["index"] == selected else ""
        tip = html.escape(f"Position {token['index'] + 1} · {token.get('segment', '')}")
        head.append(
            f'<th class="jl-token{marked}" title="{tip}">'
            f'<code>{html.escape(repr(token.get("text", "")))}</code></th>'
        )
    head = "".join(head)
    body = []
    for row in layers:
        block = f"Block {row['layer'] + 1}"
        cells = "".join(
            cell(item, token["index"], block) for item, token in zip(row["cells"], tokens)
        )
        body.append(f'<tr><th scope="row">{row["layer"] + 1}</th>{cells}</tr>')
    output = data.get("output") or []
    if len(output) == len(tokens):
        cells = "".join(cell(item, token["index"], "Output") for item, token in zip(output, tokens))
        body.append(f'<tr class="jl-output"><th scope="row">Output</th>{cells}</tr>')
    start = tokens[0]["index"] + 1
    end = tokens[-1]["index"] + 1
    total = int(data.get("total") or end)
    span = f"tokens {start:,}–{end:,} of {total:,}" if total > len(tokens) else f"all {total:,} tokens"
    hint = " Click a cell to pin its token and shade every cell by that token's rank." if not pinned else (
        " Shading follows the pinned token's rank; rank 1 is darkest."
    )
    return (
        f'<div class="viz-note">Top lens token at each block and position, {span}; '
        f'the Output row is the model\'s own next-token prediction.{hint}</div>'
        '<div class="jl-grid-wrap"><table class="jl-grid">'
        f'<thead><tr><th scope="col">Block</th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def jacobian_lens_chart(insight: dict) -> str:
    """Ranked concept readouts after a token; scores are never probabilities."""
    layers = insight.get("layers") or []
    if not layers:
        return EMPTY_JACOBIAN
    pinned = insight.get("pinned_text")
    chart = _jacobian_slice(insight)
    if pinned is not None:
        first, last = layers[0]["layer"], layers[-1]["layer"]
        width = _VIEW_WIDTH - _LENS_PAD_LEFT - _PAD_RIGHT
        height = _LENS_HEIGHT - _PAD_TOP - _LENS_PAD_BOTTOM
        maximum = max(2, insight["vocab_size"])

        def x_at(layer):
            return _LENS_PAD_LEFT + width * (layer - first) / max(1, last - first)

        def y_at(rank):
            return _PAD_TOP + height * math.log(max(1, rank)) / math.log(maximum)

        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x_at(row['layer']):.1f},{y_at(row['rank']):.1f}"
            for i, row in enumerate(layers)
        )
        ticks = sorted({1, min(10, maximum), min(100, maximum), maximum})
        grid = "".join(
            f'<line class="viz-grid" x1="{_LENS_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
            f'y1="{y_at(rank):.1f}" y2="{y_at(rank):.1f}" />'
            f'<text class="viz-tick" x="{_LENS_PAD_LEFT - 4}" y="{y_at(rank) + 3:.1f}" '
            f'text-anchor="end">{rank:,}</text>' for rank in ticks
        )
        points = "".join(
            f'<circle cx="{x_at(row["layer"]):.1f}" cy="{y_at(row["rank"]):.1f}" r="3" '
            f'fill="currentColor"><title>Block {row["layer"] + 1}: rank {row["rank"]:,}, '
            f'score {row["score"]:.3f}</title></circle>' for row in layers
        )
        chart += (
            f'<div class="viz-note">Pinned token: <code>{html.escape(repr(pinned))}</code>. '
            'Vocabulary rank by layer; rank 1 is at the top, on a logarithmic scale.</div>'
            f'<svg viewBox="0 0 {_VIEW_WIDTH:g} {_LENS_HEIGHT:g}" role="img" '
            f'aria-label="Pinned token vocabulary rank across fitted decoder blocks">'
            f'{grid}<path class="viz-line" d="{path}" />{points}'
            f'<text class="viz-tick" x="{_LENS_PAD_LEFT}" y="{_LENS_HEIGHT - 5}">block {first + 1}</text>'
            f'<text class="viz-tick" x="{_VIEW_WIDTH - _PAD_RIGHT}" y="{_LENS_HEIGHT - 5}" '
            f'text-anchor="end">block {last + 1}</text></svg>'
        )
    rows = []
    for row in layers:
        candidates = "<br>".join(
            f'<code>{html.escape(repr(candidate["text"]))}</code> '
            f'<span class="viz-sub">{candidate["score"]:.3f}</span>'
            for candidate in row["candidates"]
        )
        tracked = (
            f'<td>{row["rank"]:,}</td><td>{row["score"]:.3f}</td>'
            if pinned is not None else ""
        )
        rows.append(f'<tr><td>{row["layer"] + 1}</td><td>{candidates}</td>{tracked}</tr>')
    tracked_headers = "<th>Pinned rank</th><th>Pinned score</th>" if pinned is not None else ""
    return (
        '<figure class="viz-root" id="jacobian-lens">'
        f'<figcaption class="viz-title">Jacobian lens after '
        f'<code>{html.escape(repr(insight.get("token_text", "")))}</code></figcaption>'
        '<div class="viz-note">Vocabulary readouts of the state after processing this token. '
        'Scores measure the fitted lens readout; they are not generation probabilities or '
        'proof that a concept caused the answer. Blocks are numbered from 1.</div>'
        f'{chart}<div class="viz-note">Top tokens after <code>{html.escape(repr(insight.get("token_text", "")))}</code>, '
        'per fitted block.</div>'
        '<div class="viz-table-wrap"><table class="viz-table">'
        f'<thead><tr><th>Decoder block</th><th>Top tokens · score</th>{tracked_headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="viz-note">Lens: {html.escape(insight.get("lens_name", ""))} · '
        f'{insight.get("n_prompts", 0):,} fitting prompts.{_jacobian_precision_note(insight)}</div></figure>'
    )


def _jacobian_precision_note(insight: dict) -> str:
    precision = insight.get("precision")
    if insight.get("backend") == "mlx" and precision and precision != "full":
        return (
            f" Read through {html.escape(str(precision))} MLX weights; the lens was fitted on "
            "full-precision weights, so treat the scores as approximate."
        )
    return ""


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


EMPTY_KV_CACHE = ""

# What the metric control offers, and the reading each draws from.
KV_METRICS = {
    "Key norm": "key_norm",
    "Value norm": "value_norm",
    "Key similarity": "key_similarity",
}

_KV_NOTES = {
    "key_norm": (
        "Length of each head's key vector at each position. Shading runs from the "
        "layer's shortest key to its longest, leaving out the first token, which is "
        "often an outlier (the attention sink)."
    ),
    "value_norm": (
        "Length of each head's value vector at each position: how much a token can "
        "add to a head's output when it is attended to. Shading leaves out the first "
        "token, as for keys."
    ),
    "key_similarity": (
        "Cosine between each key and the query position's key (dashed row) in the "
        "same head, after taking out the head's mean key, which shifts every "
        "attention score equally and would make all keys look alike. Only positive "
        "similarity is shaded. Keys are stored after the rotary position embedding, "
        "so positions near the query tend to look alike partly because they are near."
    ),
}


def _bytes_label(count: int) -> str:
    for unit, size in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if count >= size:
            return f"{count / size:,.1f} {unit}"
    return f"{count:,} bytes"


def kv_cache_summary(summary: dict) -> str:
    """The whole cache in a sentence or two, from its per-layer shapes."""

    positions = [count for count in summary.get("positions") or [] if count]
    if not summary.get("attention_layers") or not positions:
        return "This model's cache stores no keys and values that can be read."
    tokens = int(summary.get("tokens") or max(positions))

    def listed(values) -> str:
        return " or ".join(str(value) for value in values)

    parts = [
        f"The cache covers {tokens:,} tokens across {summary['layers']} layers: "
        f"{listed(summary['heads'])} key-value heads of {listed(summary['dims'])} "
        f"dimensions per layer, {listed(summary['dtypes'])}, "
        f"{_bytes_label(summary['nbytes'])} in all."
    ]
    if summary["attention_layers"] < summary["layers"]:
        parts.append(
            f"{summary['attention_layers']} of the layers store keys and values; "
            "the others keep a running state instead."
        )
    if min(positions) < tokens:
        parts.append(
            f"Layers with a sliding window keep only their latest {min(positions):,}."
        )
    return " ".join(parts)


def kv_cache_grid(view: dict, tokens: list[dict], metric: str = "Key norm") -> str:
    """One layer of the cache: a row per position held, a column per head."""

    summary = kv_cache_summary(view.get("summary") or {})
    reading = view.get("reading")
    layer = int(view.get("layer") or 1)
    if not reading:
        return (
            '<div class="viz-root" id="kv-cache-view">'
            f'<div class="viz-note">{html.escape(summary)}</div>'
            f'<div class="viz-empty">Layer {layer} stores no keys and values.</div></div>'
        )
    key = KV_METRICS.get(metric, "key_norm")
    grid = reading[key]
    positions = reading["positions"]
    heads = reading["heads"]
    query = positions[-1]
    similarity = key == "key_similarity"
    if similarity:
        low, high = 0.0, 1.0
    else:
        # The first token leaves the range when there is anything else to set it.
        columns = [column for column, position in enumerate(positions) if position > 0]
        columns = columns or list(range(len(positions)))
        values = [row[column] for row in grid for column in columns]
        low, high = min(values), max(values)
    spread = (high - low) or 1.0

    rows = []
    for column, position in enumerate(positions):
        token = tokens[position] if 0 <= position < len(tokens) else {"text": "", "token_id": ""}
        segment = token.get("segment", "prompt")
        cells = []
        for head in range(heads):
            value = grid[head][column]
            heat = min(1.0, max(0.0, (value - low) / spread))
            shown = f"{value:+.2f}" if similarity else f"{value:.1f}"
            cells.append(f'<td class="kv-cell" style="--kv-heat: {heat:.2f}">{shown}</td>')
        classes = ["kv-query"] if position == query else []
        if segment != "prompt":
            classes.append("kv-response")
        rows.append(
            f'<tr class="{" ".join(classes)}"><th>{position + 1}</th>'
            f'<th class="kv-token" title="{segment} token"><code>{_token_label(token)}</code></th>'
            f'{"".join(cells)}</tr>'
        )
    header = "".join(f"<th>head {head + 1}</th>" for head in range(heads))
    held = reading.get("held", len(positions))
    notes = [summary]
    if held < int((view.get("summary") or {}).get("tokens") or held):
        notes.append(
            f"Layer {layer} has a sliding window and holds only the latest {held:,} tokens."
        )
    if len(positions) < held:
        notes.append(f"Showing the latest {len(positions):,} of the {held:,} positions it holds.")
    return (
        '<div class="viz-root" id="kv-cache-view">'
        f'<div class="viz-title">Key-value cache, layer {layer}'
        f'<span class="viz-sub">{html.escape(metric)}, {heads} heads of '
        f'{reading["dim"]} dimensions</span></div>'
        f'<div class="viz-note">{html.escape(" ".join(notes))}</div>'
        '<div class="kv-grid-wrap"><table class="kv-grid">'
        f'<thead><tr><th>#</th><th>Token</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="viz-note">{html.escape(_KV_NOTES[key])} Dashed row: the query, '
        "the position whose output made the prediction. Response tokens are in bold.</div></div>"
    )


# ------------------------------------------------------- denoising a picture

EMPTY_DENOISING_CHART = (
    '<div class="viz-empty">The denoising trace appears once a picture has '
    "been drawn.</div>"
)

# The second series' colour, for the movement line beside the guidance one.
# Both are ratios on the same axis, so they share a scale honestly; what
# tells them apart is the hue and the legend.
_MOVEMENT_COLOR = "#d98016"

_DENOISING_HEIGHT = 160.0


def _step_series(readings: Sequence, name: str) -> list[tuple[int, float]]:
    """``(step, value)`` for every step that has a reading of ``name``."""

    return [
        (int(reading.step), float(getattr(reading, name)))
        for reading in readings
        if getattr(reading, name, None) is not None
    ]


def denoising_chart(readings: Sequence, *, ceiling: float = 0.0, last_step: int = 0) -> str:
    """Guidance pull and latent movement per denoising step, on one axis.

    ``readings`` are :class:`image_runtime.StepReading` objects. Both series
    are dimensionless ratios of the same kind - a length divided by a length -
    so one axis serves both, and a reader can see directly that the prompt
    stops pulling around the same time the picture stops moving.

    ``ceiling`` and ``last_step`` let paired runs share axis ranges, including
    when one run was stopped early. Neither can clip a run's own readings.
    """

    guidance = _step_series(readings, "guidance_share")
    movement = _step_series(readings, "latent_change")
    if len(guidance) + len(movement) < 2:
        return EMPTY_DENOISING_CHART

    steps = [int(reading.step) for reading in readings]
    first, last = min(steps), max(steps)
    last = max(last, last_step)
    ceiling = max(
        0.05,
        ceiling,
        max((value for _step, value in guidance + movement), default=0.0),
    )
    plot_width = _VIEW_WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_height = _DENOISING_HEIGHT - _PAD_TOP - _PAD_BOTTOM
    span = max(1, last - first)

    def x_at(step: int) -> float:
        return _PAD_LEFT + plot_width * (step - first) / span

    def y_at(value: float) -> float:
        return _PAD_TOP + plot_height * (1 - value / ceiling)

    def path(series: list[tuple[int, float]]) -> str:
        return " ".join(
            f"{'M' if index == 0 else 'L'}{x_at(step):.1f},{y_at(value):.1f}"
            for index, (step, value) in enumerate(series)
        )

    gridlines = "".join(
        f'<line class="viz-grid" x1="{_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y1="{y_at(value):.1f}" y2="{y_at(value):.1f}" />'
        f'<text class="viz-tick" x="{_PAD_LEFT - 6}" y="{y_at(value) + 3.5:.1f}" '
        f'text-anchor="end">{value:.2f}</text>'
        for value in (0.0, ceiling / 2, ceiling)
    )
    lines = ""
    if guidance:
        lines += f'<path class="viz-line" d="{path(guidance)}" />'
    if movement:
        lines += (
            f'<path class="viz-line" d="{path(movement)}" '
            f'style="stroke: {_MOVEMENT_COLOR}" />'
        )

    by_step = {int(reading.step): reading for reading in readings}
    width = plot_width / span
    hover = "".join(
        f'<rect class="viz-hit" x="{x_at(step) - width / 2:.1f}" y="{_PAD_TOP}" '
        f'width="{width:.1f}" height="{plot_height:.1f}">'
        f"<title>{_step_title(by_step[step])}</title></rect>"
        for step in sorted(by_step)
    )
    legend = (
        '<span class="viz-key"><span class="viz-swatch viz-swatch-line"></span>'
        "guidance pull</span>"
        f'<span class="viz-key"><span class="viz-swatch" style="background: '
        f'{_MOVEMENT_COLOR}"></span>latent movement</span>'
    )
    return (
        '<figure class="viz-root" id="denoising-chart">'
        '<figcaption class="viz-title">Per denoising step'
        f'<span class="viz-sub">{legend}</span></figcaption>'
        f'<svg viewBox="0 0 {_VIEW_WIDTH:g} {_DENOISING_HEIGHT:g}" role="img" '
        'aria-label="Guidance pull and latent movement per denoising step">'
        f"{gridlines}{lines}"
        f'<line class="viz-axis" x1="{_PAD_LEFT}" x2="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y1="{y_at(0):.1f}" y2="{y_at(0):.1f}" />'
        f'<text class="viz-tick" x="{_PAD_LEFT}" y="{_DENOISING_HEIGHT - 6:g}">'
        f"step {first}</text>"
        f'<text class="viz-tick" x="{_VIEW_WIDTH - _PAD_RIGHT}" '
        f'y="{_DENOISING_HEIGHT - 6:g}" text-anchor="end">step {last}</text>'
        f"{hover}</svg></figure>"
    )


def _step_title(reading) -> str:
    parts = [f"step {reading.step}", f"timestep {reading.timestep:,.0f}"]
    if reading.guidance_share is not None:
        parts.append(f"guidance pull {reading.guidance_share:.3f}")
    if reading.latent_change is not None:
        parts.append(f"moved {reading.latent_change:.3f}")
    return html.escape(" · ".join(parts))


EMPTY_IMAGE_TILES = '<div class="viz-empty">No picture has been drawn yet.</div>'


def image_summary_tiles(summary: dict, *, note: str = "") -> str:
    """Headline numbers for a finished image run."""

    if not summary.get("step_count"):
        return EMPTY_IMAGE_TILES

    tiles = [
        _tile(
            f"{summary['step_count']:,}",
            "denoising steps",
            "How many steps the scheduler really ran, which can differ from "
            "the number asked for.",
        )
    ]
    if "mean_guidance_share" in summary:
        tiles.append(
            _tile(
                f"{summary['mean_guidance_share']:.3f}",
                "mean guidance pull",
                "How far the prompt moved each step's prediction, as a "
                "fraction of what the model predicted without it. Peak "
                f"{summary['peak_guidance_share']:.3f} at step "
                f"{summary['peak_guidance_step']:,}.",
            )
        )
        tiles.append(
            _tile(
                f"{summary['peak_guidance_step']:,}",
                "strongest pull at step",
                "The step where the prompt disagreed most with what the "
                "model would have drawn from noise alone.",
            )
        )
    if "settled_step" in summary:
        tiles.append(
            _tile(
                f"{summary['settled_step']:,}",
                "settled by step",
                "From here on every step moved the latent less than a tenth "
                "of the largest move: the composition was decided and the "
                "rest is detail.",
            )
        )
    if "final_latent_change" in summary:
        tiles.append(
            _tile(
                f"{summary['final_latent_change']:.3f}",
                "final step moved",
                "How much the last step changed the latent, relative to its "
                "own size.",
            )
        )
    footer = f'<div class="viz-note">{html.escape(note)}</div>' if note else ""
    return f'<div class="viz-root viz-tiles">{"".join(tiles)}</div>{footer}'
