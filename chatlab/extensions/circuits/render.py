"""The graph, the grouped view and the cards beside them, as HTML and SVG.

Everything is drawn here, on the server, from the saved graph. The page's
script adds hover highlighting and turns clicks into selections; nothing it
does changes what is drawn.

Color carries one thing throughout: the sign of an effect on the target.
Blue promotes it, red suppresses it, gray is too small to call. Shape says
what a node is - a circle is a feature, a square a prompt token, a diamond
transcoder error - so no node's kind depends on its color.
"""

from __future__ import annotations

import html
import math
from collections import defaultdict

ROW = 26
SLOT = 132
NARROW = 72
LEFT = 46
TOP = 28
SMALL = 0.05
EDGES_PER_NODE = 10
MAX_GROUP_FLOWS = 1000
MAX_GROUP_CHIPS = 64
MAX_GROUP_BOXES = 64
MAX_CHIP_LABEL = 160
MAX_DISPLAY_TOKENS = 64
MAX_DISPLAY_TARGETS = 64


def token_text(text):
    """A token as the page shows it: quoted, with line breaks and tabs visible."""
    shown = (text or "").replace("\n", "⏎").replace("\t", "⇥")
    return f'"{shown}"'


def _short(text, limit=18):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def node_label(node, labels):
    """What a node is called: the reader's name for it, or what it writes."""
    kind = node["kind"]
    if kind == "feature":
        named = labels.get(node["id"])
        if named:
            return named
        promoted = node.get("promotes") or []
        return ("→ " + " ".join(token_text(t) for t in promoted[:2])) if promoted else f"#{node['feature']}"
    if kind == "error":
        return "error"
    if kind == "embedding":
        return token_text(node.get("text", ""))
    return token_text(node.get("text", ""))


def role(effect, scale):
    if scale <= 0 or abs(effect) < SMALL * scale:
        return "mixed"
    return "promotes" if effect > 0 else "suppresses"


def visible(graph, nodes_shown, show_errors, keep=()):
    """The nodes drawn: the most influential features, the targets, and what feeds them."""
    nodes = graph["nodes"]
    features = sorted((n for n in nodes if n["kind"] == "feature"), key=lambda n: -n["influence"])
    chosen = {n["id"] for n in features[:nodes_shown]} | set(keep)
    if show_errors:
        errors = sorted((n for n in nodes if n["kind"] == "error"), key=lambda n: -n["influence"])
        chosen |= {n["id"] for n in errors[: max(4, nodes_shown // 3)]}
    targets = [n for n in nodes if n["kind"] == "target"]
    chosen -= {n["id"] for n in targets}
    chosen |= {n["id"] for n in targets[:MAX_DISPLAY_TARGETS]}
    present = {n["id"] for n in nodes}
    chosen &= present
    embeddings = {node["id"] for node in nodes if node["kind"] == "embedding"}
    fed = {e["source"] for e in graph["edges"] if e["target"] in chosen and e["source"] in embeddings}
    chosen |= fed
    return [n for n in nodes if n["id"] in chosen]


def _edges(graph, ids):
    """The strongest edges into each drawn node, scaled within that node."""
    incoming = defaultdict(list)
    for edge in graph["edges"]:
        if edge["source"] in ids and edge["target"] in ids:
            incoming[edge["target"]].append(edge)
    drawn = []
    for edges in incoming.values():
        edges.sort(key=lambda e: -abs(e["weight"]))
        strongest = abs(edges[0]["weight"]) or 1.0
        for edge in edges[:EDGES_PER_NODE]:
            share = abs(edge["weight"]) / strongest
            if share >= 0.04:
                drawn.append((edge, share))
    return drawn


def graph_view(graph, *, nodes_shown=40, show_errors=False, selected=(), labels=None, groups=None):
    """The attribution graph: layers bottom to top, prompt positions left to right."""
    labels = labels or {}
    groups = groups or {}
    grouped = {member: name for name, members in groups.items() for member in members}
    shown = visible(graph, nodes_shown, show_errors, keep=selected)
    ids = {n["id"] for n in shown}
    layers = graph["layers"]
    tokens = graph["tokens"]

    # Columns: the positions that have a drawn node, each as wide as its busiest layer.
    cells = defaultdict(list)
    for node in shown:
        cells[(node["position"], node["layer"])].append(node)
    positions = sorted({p for p, _ in cells})
    # A column holding only its prompt token needs room for the token, not for labels.
    width_of = {p: max((len(v) * SLOT for (q, layer), v in cells.items() if q == p and layer >= 0),
                       default=NARROW) for p in positions}
    x0, column_x = LEFT, {}
    for p in positions:
        column_x[p] = x0
        x0 += width_of[p]
    width = max(x0 + 24, 480)
    height = TOP + (layers + 2) * ROW + 34

    def y_of(layer):
        return TOP + (layers - layer) * ROW

    place = {}
    for (position, layer), members in cells.items():
        members.sort(key=lambda n: -n.get("influence", 0))
        for slot, node in enumerate(members):
            place[node["id"]] = (column_x[position] + slot * SLOT + 12, y_of(layer))

    feature_effects = [abs(n["effect"]) for n in shown if n["kind"] == "feature"]
    scale = max(feature_effects, default=0.0)
    influences = [n["influence"] for n in shown if n["kind"] == "feature"]
    top_influence = max(influences, default=0.0) or 1.0

    parts = [f'<svg class="cg-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'role="img" aria-label="Attribution graph">']
    for layer in range(-1, layers + 1):
        y = y_of(layer)
        parts.append(f'<line class="cg-row" x1="{LEFT - 6}" x2="{width - 8}" y1="{y}" y2="{y}"/>')
        if 0 <= layer < layers and layer % 2 == 0:
            parts.append(f'<text class="cg-axis" x="6" y="{y + 4}">L{layer}</text>')
    parts.append(f'<text class="cg-axis" x="6" y="{y_of(layers) + 4}">out</text>')
    parts.append(f'<text class="cg-axis" x="6" y="{y_of(-1) + 4}">emb</text>')

    parts.append('<g class="cg-edges">')
    for edge, share in sorted(_edges(graph, ids), key=lambda item: item[1]):
        (x1, y1), (x2, y2) = place[edge["source"]], place[edge["target"]]
        bend = max(14, (y1 - y2) * 0.45)
        sign = "pos" if edge["weight"] > 0 else "neg"
        parts.append(
            f'<path class="cg-edge {sign}" data-s="{html.escape(edge["source"])}" data-t="{html.escape(edge["target"])}" '
            f'd="M{x1},{y1} C{x1},{y1 - bend} {x2},{y2 + bend} {x2},{y2}" '
            f'style="stroke-width:{0.6 + 3.4 * share:.2f};opacity:{0.25 + 0.6 * share:.2f}">'
            f'<title>{edge["weight"]:+.3g}</title></path>')
    parts.append("</g>")

    selected = set(selected)
    for node in shown:
        x, y = place[node["id"]]
        label = node_label(node, labels)
        kind = node["kind"]
        classes = ["cg-node", kind]
        if node["id"] in selected:
            classes.append("sel")
        if node["id"] in grouped:
            classes.append("grouped")
        title = _title(node, _short(label, 160) if kind == "target" else label, grouped.get(node["id"]), tokens)
        attrs = f'class="{" ".join(classes)}" data-node="{html.escape(node["id"])}"'
        if kind == "feature":
            r = 4 + 6 * math.sqrt(max(node["influence"], 0) / top_influence)
            mark = f'<circle class="mark {role(node["effect"], scale)}" cx="{x}" cy="{y}" r="{r:.1f}"/>'
        elif kind == "error":
            mark = (f'<path class="mark" d="M{x},{y - 6} L{x + 6},{y} L{x},{y + 6} L{x - 6},{y} Z"/>')
        elif kind == "embedding":
            mark = f'<rect class="mark" x="{x - 6}" y="{y - 6}" width="12" height="12" rx="3"/>'
        else:
            mark = f'<circle class="mark" cx="{x}" cy="{y}" r="9"/>'
            label = f'{label} {node.get("probability", 0):.3f}'
        # A prompt token is named on the axis below it.
        text = "" if kind == "embedding" else (
            f'<text class="cg-label" x="{x + 12}" y="{y + 4}">{html.escape(_short(label))}</text>')
        parts.append(f'<g {attrs}><title>{html.escape(title)}</title>{mark}{text}'
                     f'<circle class="hit" cx="{x}" cy="{y}" r="12"/></g>')

    base = height - 10
    for p in positions:
        limit = 16 if width_of[p] > NARROW else 8
        parts.append(f'<text class="cg-token" x="{column_x[p] + 6}" y="{base}">'
                     f'{html.escape(_short(token_text(tokens[p]), limit))}<title>position {p} '
                     f'{html.escape(token_text(tokens[p]))}</title></text>')
    parts.append("</svg>")
    stats = graph.get("stats", {})
    hidden = stats.get("kept_features", 0) - sum(1 for n in shown if n["kind"] == "feature")
    note = (f'{stats.get("kept_features", 0)} features survived pruning, of {stats.get("traced_features", 0)} '
            f'traced and {stats.get("active_features", 0):,} active'
            + (f'; the {hidden} least influential are hidden' if hidden > 0 else "")
            + f'. Transcoder error carries {100 * stats.get("error_share", 0):.0f}% of the influence.')
    total_targets = sum(n["kind"] == "target" for n in graph["nodes"])
    if total_targets > MAX_DISPLAY_TARGETS:
        note += f" Showing {MAX_DISPLAY_TARGETS} of {total_targets:,} targets."
    legend = (
        '<div class="cg-legend">'
        '<span><i class="dot promotes"></i>promotes the target</span>'
        '<span><i class="dot suppresses"></i>suppresses it</span>'
        '<span><i class="dot mixed"></i>small effect</span>'
        '<span><i class="sq"></i>prompt token</span>'
        '<span><i class="dia"></i>transcoder error</span>'
        '<span><i class="ln pos"></i>excitatory edge</span>'
        '<span><i class="ln neg"></i>inhibitory edge</span>'
        '</div>')
    focus = column_x.get(len(tokens) - 1, max(column_x.values(), default=0))
    return (f'<div class="cg-root viz-root"><div class="cg-scroll" data-focus="{focus}">{"".join(parts)}</div>{legend}'
            f'<div class="cg-note">{html.escape(note)} Click a node to read it; shift-click to select several.</div></div>')


def _title(node, label, group, tokens):
    kind = node["kind"]
    at = f'position {node["position"]} {token_text(tokens[node["position"]])}'
    if kind == "feature":
        lines = [f'{label}', f'layer {node["layer"]} · feature {node["feature"]} · {at}',
                 f'activation {node["activation"]:.3g} · influence {node["influence"]:.3g} · '
                 f'effect on target {node["effect"]:+.3g}']
    elif kind == "error":
        lines = [f'transcoder error · layer {node["layer"]} · {at}', f'effect on target {node["effect"]:+.3g}']
    elif kind == "embedding":
        lines = [f'prompt token {label} · {at}', f'effect on target {node["effect"]:+.3g}']
    else:
        lines = [f'target {label}', f'probability {node.get("probability", 0):.3f}']
    if group:
        lines.append(f"group: {group}")
    return "\n".join(lines)


def feature_card(node, record=None, labels=None, ablation=None, record_error=None, tokens=None):
    """The selected node: what it is, what it writes, and where it fires most."""
    labels = labels or {}
    if node is None:
        return '<div class="cg-root viz-root cg-card cg-empty">Click a node in the graph to read it here.</div>'
    kind = node["kind"]
    label = node_label(node, labels)
    if kind != "feature":
        rows = [("kind", {"error": "transcoder error", "embedding": "prompt token", "target": "target"}[kind])]
        if kind != "target":
            rows.append(("effect on target", f'{node["effect"]:+.4g}'))
            rows.append(("graph influence", f'{node["influence"]:.4g}'))
        if kind == "target":
            rows.append(("probability", f'{node.get("probability", 0):.4f}'))
        return (f'<div class="cg-root viz-root cg-card"><h4>{html.escape(label)}</h4>{_table(rows)}'
                + ('<p class="cg-muted">Error nodes hold what the transcoder failed to reconstruct at this '
                   'layer and position. A large one means the graph is not explaining this part of the '
                   'computation.</p>' if kind == "error" else "") + '</div>')
    sign = role(node["effect"], abs(node["effect"]) or 1.0)
    rows = [
        ("role", {"promotes": "promotes the target", "suppresses": "suppresses the target",
                  "mixed": "small effect"}[sign]),
        ("activation", f'{node["activation"]:.4g}'),
        ("effect on target", f'{node["effect"]:+.4g}'),
        ("graph influence", f'{node["influence"]:.4g}'),
    ]
    if ablation is not None:
        rows.append(("ablation Δ target", ", ".join(f"{d:+.3g}" for d in ablation["deltas"])))
    if record and record.get("activation_frequency") is not None:
        rows.append(("activation freq.", f'{100 * record["activation_frequency"]:.3f}%'))
    head = (f'<h4>{html.escape(label)} <span class="cg-muted">— layer {node["layer"]} · feature {node["feature"]} · '
            f'{html.escape(token_text((tokens or [""] * (node["position"] + 1))[node["position"]]))}</span></h4>')
    promotes = (record or {}).get("top_logits") or node.get("promotes") or []
    suppresses = (record or {}).get("bottom_logits") or node.get("suppresses") or []
    body = [head, _table(rows)]
    if promotes:
        body.append('<h5>promotes (top output logits)</h5>' + _chips(promotes[:8]))
    if suppresses:
        body.append('<h5>suppresses</h5>' + _chips(suppresses[:8]))
    if record:
        body.append('<h5>top-activating contexts <span class="cg-muted">(highlight = strongest token)</span></h5>')
        body.append(_examples(record))
    elif record_error:
        body.append(f'<p class="cg-muted">Examples unavailable: {html.escape(record_error)}</p>')
    return f'<div class="cg-root viz-root cg-card">{"".join(body)}</div>'


def _table(rows):
    return ('<table class="cg-facts">' + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows) + "</table>")


def _chips(tokens):
    return '<div class="cg-chips">' + "".join(f"<span>{html.escape(token_text(t))}</span>" for t in tokens) + "</div>"


def _examples(record, limit=8, window=24):
    out = []
    for quantile in record.get("examples_quantiles") or []:
        for example in (quantile.get("examples") or [])[:limit]:
            tokens = example.get("tokens") or []
            acts = example.get("tokens_acts_list") or []
            if not tokens or len(acts) != len(tokens):
                continue
            peak = max(range(len(acts)), key=lambda i: acts[i])
            top = acts[peak] or 1.0
            start, end = max(0, peak - window), min(len(tokens), peak + 8)
            spans = []
            for i in range(start, end):
                text = tokens[i].replace("\n", "⏎")
                strength = max(0.0, acts[i] / top)
                style = f' style="--cg-heat:{strength:.2f}"' if strength > 0.02 else ""
                cls = "peak" if i == peak else "act" if strength > 0.02 else ""
                spans.append(f'<span class="{cls}"{style}>{html.escape(text)}</span>')
            out.append(f'<div class="cg-example">{"… " if start else ""}{"".join(spans)}</div>')
        break
    return "".join(out) or '<p class="cg-muted">No examples recorded.</p>'


# The feature browser ---------------------------------------------------------

def top_tokens(record, limit=6):
    """The tokens a feature fires hardest on: each top example's peak token, most common first."""
    counts = {}
    for quantile in record.get("examples_quantiles") or []:
        for example in quantile.get("examples") or []:
            tokens = example.get("tokens") or []
            acts = example.get("tokens_acts_list") or []
            if tokens and len(acts) == len(tokens):
                peak = tokens[max(range(len(acts)), key=lambda i: acts[i])]
                counts[peak] = counts.get(peak, 0) + 1
        break
    return sorted(counts.items(), key=lambda item: -item[1])[:limit]


def feature_list(layer, start, width, rows, selected=None, page_id=""):
    """One page of a layer's features, each with the tokens it fires on and the tokens it writes.

    ``rows`` holds ``(feature, record, error)`` for each feature on the page.
    """
    end = start + len(rows) - 1
    out = [f'<div class="cg-root viz-root cf-list" data-page="{html.escape(page_id)}"><p class="cg-muted">Layer {layer} · features '
           f'{start:,}–{end:,} of {width:,}. Click one to read it and steer by it.</p>',
           '<table><thead><tr><th>feature</th><th>fires on</th><th>promotes</th><th>freq.</th></tr></thead><tbody>']
    for feature, record, error in rows:
        cls = ' class="sel"' if feature == selected else ""
        if record is None:
            out.append(f'<tr data-feature="{feature}"{cls}><td>{feature:,}</td>'
                       f'<td colspan="3" class="cg-muted">{html.escape(error or "no record")}</td></tr>')
            continue
        fires = "".join(f"<span>{html.escape(token_text(t))}<em>×{n}</em></span>" for t, n in top_tokens(record))
        promotes = "".join(f"<span>{html.escape(token_text(t))}</span>" for t in (record.get("top_logits") or [])[:4])
        frequency = record.get("activation_frequency")
        frequency = "–" if frequency is None else f"{100 * frequency:.3f}%"
        out.append(f'<tr data-feature="{feature}"{cls}><td>{feature:,}</td>'
                   f'<td><div class="cg-chips">{fires or "–"}</div></td>'
                   f'<td><div class="cg-chips">{promotes or "–"}</div></td><td>{frequency}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def feature_detail(layer=None, feature=None, record=None, error=None):
    """A feature picked in the browser: what it fires on, what it writes, and how strongly."""
    if feature is None:
        return '<div class="cg-root viz-root cg-card cg-empty">Click a feature in the list to read it here.</div>'
    body = [f'<h4>Layer {layer} · feature {feature:,}</h4>']
    if record is None:
        body.append(f'<p class="cg-muted">Examples unavailable: {html.escape(error or "no record")}</p>')
        return f'<div class="cg-root viz-root cg-card">{"".join(body)}</div>'
    rows = []
    if record.get("act_max") is not None:
        rows.append(("highest activation", f'{record["act_max"]:.4g}'))
    if record.get("activation_frequency") is not None:
        rows.append(("activation freq.", f'{100 * record["activation_frequency"]:.3f}%'))
    body.append(_table(rows))
    fires = top_tokens(record, limit=12)
    if fires:
        body.append('<h5>fires on <span class="cg-muted">(peak token of each top example)</span></h5>'
                    + '<div class="cg-chips">' + "".join(
                        f"<span>{html.escape(token_text(t))}<em>×{n}</em></span>" for t, n in fires) + "</div>")
    if record.get("top_logits"):
        body.append('<h5>promotes (top output logits)</h5>' + _chips(record["top_logits"][:8]))
    if record.get("bottom_logits"):
        body.append('<h5>suppresses</h5>' + _chips(record["bottom_logits"][:8]))
    body.append('<h5>top-activating contexts <span class="cg-muted">(highlight = strongest token)</span></h5>')
    body.append(_examples(record))
    return f'<div class="cg-root viz-root cg-card">{"".join(body)}</div>'


# The grouped view ---------------------------------------------------------

def _ratio(after, before):
    if before <= 0:
        return None
    return after / before


def _times(value):
    return "–" if value is None else f"×{value:.2f}"


def group_view(graph, groups, effects=None, labels=None, decode=None):
    """Groups of features as boxes, joined by their summed edges, beside the tokens they move."""
    labels = labels or {}
    nodes = {n["id"]: n for n in graph["nodes"]}
    groups = {name: [m for m in members if m in nodes] for name, members in groups.items()}
    groups = {name: members for name, members in groups.items() if members}
    if not groups:
        return ('<div class="cg-root viz-root cg-empty">Select features in the graph (shift-click for several), '
                'name them, and press <b>Group selected</b>. Groups appear here joined by their summed edges.</div>')
    total_groups = len(groups)
    groups = dict(list(groups.items())[:MAX_GROUP_BOXES])
    prompt_bucket, error_bucket, target_bucket = object(), object(), object()
    bucket_labels = {prompt_bucket: "prompt", error_bucket: "error", target_bucket: "target"}
    member_of = {m: name for name, members in groups.items() for m in members}

    def bucket(node_id):
        if node_id in member_of:
            return member_of[node_id]
        kind = nodes[node_id]["kind"] if node_id in nodes else ""
        return {"embedding": prompt_bucket, "error": error_bucket, "target": target_bucket}.get(kind)

    flows = defaultdict(float)
    for edge in graph["edges"]:
        a, b = bucket(edge["source"]), bucket(edge["target"])
        if a and b and a != b and b != prompt_bucket and b != error_bucket and a != target_bucket:
            flows[(a, b)] += edge["weight"]
    mean_layer = {name: sum(nodes[m]["layer"] for m in members) / len(members) for name, members in groups.items()}
    incoming_flows = defaultdict(list)
    for (a, b), weight in flows.items():
        if weight != 0:
            incoming_flows[b].append(a)
    depth = {}
    for name in sorted(groups, key=lambda g: mean_layer[g]):
        parents = [depth[a] for a in incoming_flows[name] if a in depth]
        depth[name] = 1 + max(parents, default=0)
    columns = defaultdict(list)
    for name, d in depth.items():
        columns[d].append(name)
    box_w, box_h, gap = 210, 64, 26
    col_x = {d: 130 + (d - 1) * (box_w + 56) for d in columns}
    tallest = max(len(v) for v in columns.values())
    height = max(340, 40 + tallest * (box_h + gap), _token_bars_height(effects))
    place = {}
    for d, names in columns.items():
        names.sort(key=lambda g: mean_layer[g])
        top = (height - len(names) * (box_h + gap)) / 2
        for i, name in enumerate(names):
            place[name] = (col_x[d], top + i * (box_h + gap))
    tokens_x = max(col_x.values()) + box_w + 60
    place[prompt_bucket] = (40, height * 0.32)
    place[error_bucket] = (40, height * 0.72)
    width = tokens_x + 330

    def anchor(name, side):
        if name in (prompt_bucket, error_bucket):
            x, y = place[name]
            return x + 12, y
        if name == target_bucket:
            return tokens_x - 10, 70
        x, y = place[name]
        return (x if side == "in" else x + box_w), y + box_h / 2

    parts = [f'<svg class="cg-svg cg-fit" viewBox="0 0 {width} {height}" '
             'role="img" aria-label="Grouped attribution graph">']
    strongest = max((abs(w) for w in flows.values()), default=1.0) or 1.0
    visible_flows = sorted(flows.items(), key=lambda item: abs(item[1]), reverse=True)[:MAX_GROUP_FLOWS]
    for (a, b), weight in reversed(visible_flows):
        share = abs(weight) / strongest
        if share < 0.03:
            continue
        x1, y1 = anchor(a, "out")
        x2, y2 = anchor(b, "in")
        mid = (x1 + x2) / 2
        sign = "pos" if weight > 0 else "neg"
        parts.append(f'<path class="cg-edge {sign}" d="M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}" '
                     f'style="stroke-width:{0.8 + 5 * share:.2f};opacity:{0.3 + 0.55 * share:.2f}">'
                     f'<title>{html.escape(bucket_labels.get(a, a))} → {html.escape(bucket_labels.get(b, b))}: {weight:+.3g}</title></path>')
    for name, label in ((prompt_bucket, "prompt tokens"), (error_bucket, "unexplained\n(transcoder error)")):
        x, y = place[name]
        cls = "sq" if name == prompt_bucket else "dia"
        mark = (f'<rect class="mark" x="{x - 9}" y="{y - 9}" width="18" height="18" rx="4"/>' if cls == "sq"
                else f'<path class="mark" d="M{x},{y - 10} L{x + 10},{y} L{x},{y + 10} L{x - 10},{y} Z"/>')
        lines = "".join(f'<tspan x="{x}" dy="{14 if i else 0}">{html.escape(t)}</tspan>'
                        for i, t in enumerate(label.split("\n")))
        parts.append(f'<g class="cg-node {"embedding" if cls == "sq" else "error"}">{mark}'
                     f'<text class="cg-axis" text-anchor="middle" y="{y + 26}">{lines}</text></g>')
    group_effects = (effects or {}).get("groups", {})
    base_pivot = (effects or {}).get("baseline", {}).get("pivot")
    for name in groups:
        x, y = place[name]
        effect = group_effects.get(name)
        line = "not measured yet"
        tone = "mixed"
        if effect and base_pivot:
            ablate = _ratio(effect["ablate"]["pivot"], base_pivot)
            boost = _ratio(effect["boost"]["pivot"], base_pivot)
            line = f"ablate {_times(ablate)} · boost {_times(boost)}"
            if boost is not None and boost > 1.05:
                tone = "promotes"
            elif boost is not None and boost < 0.95:
                tone = "suppresses"
        count = len(groups[name])
        parts.append(
            f'<g class="cg-group {tone}" data-group="{html.escape(name)}">'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8"/>'
            f'<text class="cg-group-name" x="{x + box_w / 2}" y="{y + 20}">{html.escape(_short(name, 30))}</text>'
            f'<text class="cg-group-sub" x="{x + box_w / 2}" y="{y + 37}">{count} feature{"s" * (count != 1)}</text>'
            f'<text class="cg-group-sub" x="{x + box_w / 2}" y="{y + 53}">{html.escape(line)}</text></g>')
    parts.append(_token_bars(effects, tokens_x, decode))
    parts.append("</svg>")
    legend = ('<div class="cg-legend"><span><i class="dot promotes"></i>boosting raises P(pivot)</span>'
              '<span><i class="dot suppresses"></i>boosting lowers P(pivot)</span>'
              '<span><i class="dot mixed"></i>small / mixed, or not measured</span></div>')
    omitted = (f'<p class="cg-muted">Showing {len(groups)} of {total_groups:,} groups.</p>'
               if total_groups > len(groups) else "")
    return f'<div class="cg-root viz-root"><div class="cg-scroll">{"".join(parts)}</div>{legend}{omitted}</div>'


def _token_bars_height(effects):
    if not effects:
        return 96
    lists = [effects["pivot"], effects["alternatives"]]
    return 50 + sum(38 + 22 * (min(len(ids), MAX_DISPLAY_TOKENS) + (len(ids) > MAX_DISPLAY_TOKENS))
                    for ids in lists if ids) + 20


def _token_bars(effects, x, decode):
    if not effects:
        return (f'<text class="cg-axis" x="{x}" y="60">Run the interventions to see each</text>'
                f'<text class="cg-axis" x="{x}" y="76">token\'s probability here.</text>')
    decode = decode or str
    baseline = effects["baseline"]["tokens"]
    biggest = max((float(v) for v in baseline.values()), default=1.0) or 1.0
    parts, y = [], 50
    for heading, ids, cls in (("pivot tokens", effects["pivot"], "promotes"),
                              ("alternatives", effects["alternatives"], "mixed")):
        if not ids:
            continue
        parts.append(f'<text class="cg-head {cls}" x="{x}" y="{y}">{heading}</text>')
        y += 20
        for token in ids[:MAX_DISPLAY_TOKENS]:
            p = float(baseline.get(token, baseline.get(str(token), 0.0)))
            length = 110 * p / biggest
            parts.append(f'<text class="cg-label" x="{x}" y="{y + 4}">{html.escape(_short(token_text(decode(token)), 12))}</text>'
                         f'<rect class="bar {cls}" x="{x + 96}" y="{y - 6}" width="{max(length, 1.5):.1f}" height="11" rx="2"/>'
                         f'<text class="cg-tick" x="{x + 100 + length:.1f}" y="{y + 4}">{p:.3f}</text>')
            y += 22
        if len(ids) > MAX_DISPLAY_TOKENS:
            parts.append(f'<text class="cg-tick" x="{x}" y="{y}">Showing {MAX_DISPLAY_TOKENS} of {len(ids):,} tokens</text>')
            y += 22
        y += 18
    where = "every position" if effects.get("every_position") else "the positions the graph found them at"
    parts.append(f'<text class="cg-tick" x="{x}" y="{y}">mean P at the last token, {effects["prefixes"]} '
                 f'prefix{"es" * (effects["prefixes"] != 1)}; changes applied at {where}</text>')
    return "".join(parts)


def group_card(name, members, graph, effects=None, labels=None, decode=None):
    """One group: its features, and what ablating or boosting it does to each token."""
    if not name:
        return '<div class="cg-root viz-root cg-card cg-empty">Choose a group to read what it does.</div>'
    labels = labels or {}
    decode = decode or str
    nodes = {n["id"]: n for n in graph["nodes"]}
    visible_members = [m for m in members if m in nodes]
    chips = "".join(f'<span title="{html.escape(_short(m, 96))}">{html.escape(_short(node_label(nodes[m], labels), MAX_CHIP_LABEL))}</span>'
                    for m in visible_members[:MAX_GROUP_CHIPS])
    body = [f'<h4>{html.escape(_short(name, 200))}</h4><div class="cg-chips">{chips}</div>']
    if len(visible_members) > MAX_GROUP_CHIPS:
        body.append(f'<p class="cg-muted">Showing {MAX_GROUP_CHIPS} of {len(visible_members):,} features.</p>')
    effect = (effects or {}).get("groups", {}).get(name)
    if not effect:
        body.append('<p class="cg-muted">Run the interventions to measure this group.</p>')
        return f'<div class="cg-root viz-root cg-card">{"".join(body)}</div>'
    base = effects["baseline"]
    ablate = _ratio(effect["ablate"]["pivot"], base["pivot"])
    boost = _ratio(effect["boost"]["pivot"], base["pivot"])
    body.append(f'<p>P(pivot) {_times(ablate)} when ablated, {_times(boost)} when boosted '
                f'×{effects["boost"]:g}. Active in {effect["active_prefixes"]} of {effects["prefixes"]} '
                f'prefix{"es" * (effects["prefixes"] != 1)}. Bars: change in each token\'s probability, on a '
                'log scale, shown as a multiplier.</p>')
    for kind, title in (("boost", "boost this group → each token"), ("ablate", "ablate this group → each token")):
        body.append(f"<h5>{title}</h5>" + _multipliers(effects, effect[kind]["tokens"], decode))
    return f'<div class="cg-root viz-root cg-card">{"".join(body)}</div>'


def _multipliers(effects, after, decode):
    base = effects["baseline"]["tokens"]
    rows = []
    pivot = set(effects["pivot"])
    ids = [*effects["pivot"][:MAX_DISPLAY_TOKENS], *effects["alternatives"][:MAX_DISPLAY_TOKENS]]
    ratios = []
    for token in ids:
        before = float(base.get(token, base.get(str(token), 0.0)))
        changed = float(after.get(token, after.get(str(token), 0.0)))
        ratios.append(changed / before if before > 0 and changed > 0 else None)
    span = max((abs(math.log(r)) for r in ratios if r), default=1.0) or 1.0
    for token, ratio in zip(ids, ratios):
        cls = "promotes" if token in pivot else "mixed"
        if ratio is None:
            bar = ""
        else:
            share = 50 * math.log(ratio) / span
            left = 50 + min(share, 0)
            bar = f'<i class="{cls}" style="left:{left:.1f}%;width:{abs(share):.1f}%"></i>'
        rows.append(f'<div class="cg-mult"><span>{html.escape(_short(token_text(decode(token)), 80))}</span>'
                    f'<b>{bar}<u></u></b><em>{_times(ratio)}</em></div>')
    total = len(effects["pivot"]) + len(effects["alternatives"])
    if total > len(ids):
        rows.append(f'<p class="cg-muted">Showing {len(ids)} of {total:,} measured tokens.</p>')
    return "".join(rows)


CSS = """
#circuits-page {overflow-y:auto; min-height:0; padding:12px;}
#circuits-page .cg-root { --cg-pos:#2a78d6; --cg-neg:#e34948; --cg-mid:#898781; }
.dark #circuits-page .cg-root { --cg-pos:#3987e5; --cg-neg:#e66767; }
#circuits-page .cg-scroll { display:block; text-align:left; overflow:auto; max-height:78vh; border:1px solid var(--viz-grid); border-radius:8px; background:var(--block-background-fill); }
#circuits-page .cg-svg { display:block; margin:0; width:auto !important; max-width:none; }
#circuits-page .cg-svg.cg-fit { width:100% !important; height:auto; max-height:70vh; }
#circuits-page .cg-row { stroke:var(--viz-grid); stroke-width:1; }
#circuits-page .cg-axis, #circuits-page .cg-tick { fill:var(--viz-muted); font-size:10px; font-variant-numeric:tabular-nums; }
#circuits-page .cg-token { fill:var(--viz-ink); font-size:11px; font-family:ui-monospace,monospace; }
#circuits-page .cg-label { fill:var(--viz-ink); font-size:11px; pointer-events:none; }
#circuits-page .cg-edge { fill:none; stroke-linecap:round; }
#circuits-page .cg-edge.pos { stroke:var(--cg-pos); }
#circuits-page .cg-edge.neg { stroke:var(--cg-neg); }
#circuits-page .cg-node { cursor:pointer; }
#circuits-page .cg-node .hit { fill:transparent; }
#circuits-page .cg-node .mark { stroke:var(--block-background-fill); stroke-width:2; fill:var(--cg-mid); }
#circuits-page .cg-node .mark.promotes { fill:var(--cg-pos); }
#circuits-page .cg-node .mark.suppresses { fill:var(--cg-neg); }
#circuits-page .cg-node.embedding .mark { fill:var(--viz-ink); }
#circuits-page .cg-node.error .mark { fill:none; stroke:var(--cg-mid); stroke-width:1.5; }
#circuits-page .cg-node.target .mark { fill:var(--viz-ink); }
#circuits-page .cg-node.sel .mark { stroke:var(--viz-ink); stroke-width:3; }
#circuits-page .cg-node.grouped .mark { stroke-dasharray:3 2; stroke:var(--viz-ink); stroke-width:1.5; }
#circuits-page .cg-svg.hovering .cg-edge:not(.lit) { opacity:0.06 !important; }
#circuits-page .cg-svg.hovering .cg-node:not(.lit) { opacity:0.35; }
#circuits-page .cg-legend { display:flex; flex-wrap:wrap; gap:6px 18px; margin:8px 2px 0; font-size:12px; color:var(--viz-muted); }
#circuits-page .cg-legend span { display:inline-flex; align-items:center; gap:6px; }
#circuits-page .cg-legend i { display:inline-block; }
#circuits-page .cg-legend .dot { width:10px; height:10px; border-radius:50%; background:var(--cg-mid); }
#circuits-page .cg-legend .dot.promotes { background:var(--cg-pos); }
#circuits-page .cg-legend .dot.suppresses { background:var(--cg-neg); }
#circuits-page .cg-legend .sq { width:10px; height:10px; border-radius:2px; background:var(--viz-ink); }
#circuits-page .cg-legend .dia { width:8px; height:8px; transform:rotate(45deg); border:1.5px solid var(--cg-mid); }
#circuits-page .cg-legend .ln { width:18px; height:2px; background:var(--cg-pos); }
#circuits-page .cg-legend .ln.neg { background:var(--cg-neg); }
#circuits-page .cg-note, #circuits-page .cg-muted { color:var(--body-text-color-subdued); font-size:12px; font-weight:400; }
#circuits-page .cg-note { margin:4px 2px; }
#circuits-page .cg-empty { color:var(--body-text-color-subdued); padding:16px; }
#circuits-page .cg-card { border:1px solid var(--viz-grid); border-radius:8px; padding:12px 14px; background:var(--block-background-fill); }
#circuits-page .cg-card h4 { margin:0 0 8px; font-size:15px; }
#circuits-page .cg-card h5 { margin:14px 0 6px; font-size:13px; }
#circuits-page .cg-facts, #circuits-page .cg-facts tr, #circuits-page .cg-facts th, #circuits-page .cg-facts td { border:none !important; background:none !important; padding:2px 18px 2px 0; font-size:13px; }
#circuits-page .cg-facts { width:auto; margin:0; }
#circuits-page .cg-facts th { text-align:left; font-weight:400; color:var(--body-text-color-subdued); padding:2px 18px 2px 0; }
#circuits-page .cg-facts td { font-variant-numeric:tabular-nums; }
#circuits-page .cg-chips { display:flex; flex-wrap:wrap; gap:6px; }
#circuits-page .cg-chips span { border:1px solid var(--viz-grid); border-radius:12px; padding:1px 9px; font-size:12px; font-family:ui-monospace,monospace; }
#circuits-page .cg-example { font-family:ui-monospace,monospace; font-size:12px; line-height:1.55; border:1px solid var(--viz-grid); border-radius:6px; padding:6px 8px; margin:6px 0; white-space:pre-wrap; overflow-wrap:anywhere; }
#circuits-page .cg-example .act { background:color-mix(in srgb, #fab219 calc(var(--cg-heat) * 45%), transparent); }
#circuits-page .cg-example .peak { background:color-mix(in srgb, #fab219 60%, transparent); font-weight:600; }
#circuits-page .cg-chips em { font-style:normal; color:var(--body-text-color-subdued); margin-left:3px; }
#circuits-page .cf-list { border:1px solid var(--viz-grid); border-radius:8px; padding:8px 10px; background:var(--block-background-fill); }
#circuits-page .cf-list table { width:100%; border-collapse:collapse; margin:0; border:none !important; }
#circuits-page .cf-list thead, #circuits-page .cf-list tbody, #circuits-page .cf-list tr { border:none !important; }
#circuits-page .cf-list th, #circuits-page .cf-list td { border:none !important; border-top:1px solid var(--viz-grid) !important; background:none !important; padding:5px 8px 5px 4px; font-size:13px; text-align:left; vertical-align:top; }
#circuits-page .cf-list th { font-weight:400; color:var(--body-text-color-subdued); }
#circuits-page .cf-list td:first-child, #circuits-page .cf-list td:last-child { font-variant-numeric:tabular-nums; white-space:nowrap; }
#circuits-page .cf-list tbody tr { cursor:pointer; }
#circuits-page .cf-list tbody tr:hover td { background:color-mix(in srgb, var(--viz-grid) 35%, transparent) !important; }
#circuits-page .cf-list tbody tr.sel td { background:color-mix(in srgb, var(--cg-pos) 14%, transparent) !important; }
#circuits-page .cg-group rect { fill:var(--block-background-fill); stroke:var(--cg-mid); stroke-width:1.6; cursor:pointer; }
#circuits-page .cg-group.promotes rect { stroke:var(--cg-pos); }
#circuits-page .cg-group.suppresses rect { stroke:var(--cg-neg); }
#circuits-page .cg-group-name { fill:var(--viz-ink); font-size:13px; font-weight:600; text-anchor:middle; pointer-events:none; }
#circuits-page .cg-group-sub { fill:var(--viz-muted); font-size:11px; text-anchor:middle; pointer-events:none; }
#circuits-page .cg-head { font-size:12px; fill:var(--viz-muted); }
#circuits-page .cg-head.promotes { fill:var(--cg-pos); }
#circuits-page .bar { fill:var(--cg-mid); }
#circuits-page .bar.promotes { fill:var(--cg-pos); }
#circuits-page .cg-mult { display:grid; grid-template-columns:110px 1fr 64px; align-items:center; gap:8px; font-size:12px; margin:2px 0; }
#circuits-page .cg-mult span { text-align:right; font-family:ui-monospace,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#circuits-page .cg-mult b { display:block; position:relative; height:12px; background:color-mix(in srgb, var(--viz-grid) 60%, transparent); border-radius:2px; }
#circuits-page .cg-mult b i { display:block; position:absolute; top:0; bottom:0; border-radius:2px; background:var(--cg-mid); }
#circuits-page .cg-mult b i.promotes { background:var(--cg-pos); }
#circuits-page .cg-mult b u { display:block; position:absolute; left:50%; top:-2px; bottom:-2px; border-left:1px solid var(--viz-ink); }
#circuits-page .cg-mult em { font-style:normal; color:var(--body-text-color-subdued); font-variant-numeric:tabular-nums; }
"""
