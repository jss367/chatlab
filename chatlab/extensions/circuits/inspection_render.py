"""Presentation of component inspections; measurements stay in their recorded units."""
from __future__ import annotations

import html
import math

from .render import _short


def activation(values):
    if len(values) == 1:
        return f"{values[0]:+.5g}"
    return f"norm {math.sqrt(math.fsum(x * x for x in values)):.5g}"


def signed(value):
    """Direction of a measured change, independent of the size of the effect."""
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"


def ratio(delta):
    return f"{math.exp(delta):.5g}×" if delta < 700 else "∞×"


def probability(value):
    return f"{100 * value:.4g}%"


def native_neighborhood(component, neighborhood):
    incoming, outgoing = neighborhood["incoming"], neighborhood["outgoing"]
    height = max(260, 64 * max(len(incoming), len(outgoing)) + 90)
    center = height / 2 + 14
    parts = [f'<section class="ci-panel"><div class="ci-panel-heading"><div><h4>Neuron neighborhood</h4>'
             f'<p>Position {neighborhood["position"]} · adjacent layers</p></div>'
             '<span class="ci-badge">Native neurons</span></div><div class="ci-canvas">'
             f'<svg viewBox="0 0 940 {height}" role="img" aria-label="Native neuron neighborhood">',
             '<text class="ci-column-label" x="30" y="30">UPSTREAM CANDIDATES</text>',
             '<text class="ci-column-label" x="662" y="30">DOWNSTREAM CHANGES</text>']
    # Color describes the measured response to removing a component, not the
    # estimated gradient used to choose upstream candidates.
    for nodes, left in ((incoming, True), (outgoing, False)):
        maximum = max((abs(n["activation_delta"]) for n in nodes), default=0) or 1
        for i, node in enumerate(nodes):
            y = 74 + i * 64
            delta = node["activation_delta"]
            direction = signed(-delta)
            x1, y1, x2, y2 = (278, y, 410, center) if left else (530, center, 662, y)
            weight = 1.2 + 2.2 * math.sqrt(abs(delta) / maximum)
            parts.append(f'<path class="ci-flow {direction}" d="M{x1} {y1} C{x1+68} {y1} {x2-68} {y2} {x2} {y2}" '
                         f'style="stroke-width:{weight:.2f}"/>')
            x = 30 if left else 662
            tip = (f'Zero this upstream neuron: selected activation changes by {delta:+.6g}. '
                   f'Candidate activation × gradient score: {node["score"]:+.6g}.' if left else
                   f'Zero the selected neuron: this activation changes from {node["baseline"]:.6g} '
                   f'to {node["ablated"]:.6g}, a change of {delta:+.6g}.')
            parts.append(f'<g class="ci-neighbor {direction}"><title>{html.escape(tip)}</title>'
                         f'<rect x="{x}" y="{y-25}" width="248" height="50" rx="10"/>'
                         f'<circle class="ci-node-dot" cx="{x+15}" cy="{y-7}" r="3"/>'
                         f'<text class="ci-node-title" x="{x+27}" y="{y-3}">Layer {node["layer"]} · Neuron {node["index"]}</text>'
                         f'<text class="ci-node-detail" x="{x+15}" y="{y+15}">Ablation Δ {delta:+.3g}</text></g>')
        if not nodes:
            x = 30 if left else 662
            note = "No preceding layer" if left else "No following layer"
            parts.append(f'<text class="ci-node-detail" x="{x+36}" y="{center}">{note}</text>')
    parts.append(f'<circle class="ci-hub-halo" cx="470" cy="{center}" r="72"/>'
                 f'<circle class="ci-hub" cx="470" cy="{center}" r="59"/>'
                 f'<text class="ci-hub-kicker" x="470" y="{center-25}">SELECTED</text>'
                 f'<text class="ci-hub-layer" x="470" y="{center-2}">Layer {component[1]}</text>'
                 f'<text class="ci-hub-index" x="470" y="{center+20}">Neuron {component[2]}</text>'
                 '</svg></div><div class="ci-legend"><span><i class="positive"></i>Ablation lowers activation</span>'
                 '<span><i class="negative"></i>Ablation raises activation</span>'
                 '<span>Line weights scaled within each side</span></div>'
                 '<details class="ci-method"><summary>How to read this circuit</summary><p>'
                 'Search covers the immediately preceding and following layers at this token in the first destination prefix. '
                 'Upstream candidates are ranked by activation × local gradient, then individually ablated to measure '
                 'the selected activation change. Downstream neurons are ranked by their measured activation change '
                 'after the selection is zeroed. This local view does not enumerate all neurons, token positions, '
                 'or attention paths. Hover a node for its complete measurements.</p></details></section>')
    return ''.join(parts)


def empty_inspection():
    return ('<div class="ci-root ci-empty"><span class="ci-eyebrow">COMPONENT INSPECTION</span>'
            '<svg viewBox="0 0 260 130" aria-hidden="true"><path d="M40 30 C85 30 85 65 110 65 M40 100 '
            'C85 100 85 65 110 65 M150 65 C190 65 190 35 220 35 M150 65 C190 65 190 100 220 100" '
            'fill="none" stroke="currentColor" opacity=".25" stroke-width="2"/>'
            '<g class="ci-empty-dots"><circle cx="40" cy="30" r="7"/><circle cx="40" cy="100" r="7"/>'
            '<circle cx="220" cy="35" r="7"/><circle cx="220" cy="100" r="7"/></g>'
            '<circle class="ci-hub-halo" cx="130" cy="65" r="28"/>'
            '<circle class="ci-hub" cx="130" cy="65" r="18"/></svg>'
            '<h3>Explore a component’s role</h3><p>Choose a neuron or attention head, add a prompt, '
            'and measure what changes when you intervene.</p><div class="ci-empty-steps">'
            '<span><b>01</b> Choose</span><span><b>02</b> Measure</span><span><b>03</b> Trace</span></div></div>')


def _condition(row):
    if row["condition"] == "baseline":
        return "Baseline", "Reference", "reference"
    if row.get("factor") == 0:
        return "Ablate", "Activation set to zero", "ablate"
    if row.get("factor") == 1:
        return "Scale 1×", "No-op control", "control"
    if "factor" in row:
        return f'Scale {row["factor"]:g}×', "Activation multiplier", "scale"
    if row["condition"] == "source activation patched":
        return "Patch from source", "Source activation copied", "patch"
    return "Restore later component", "Upstream remains ablated", "restore"


def inspection_result(result):
    esc = html.escape
    kind, layer, index = result["component"]
    count = len(result["prefixes"])
    title = f"Layer {layer} · {kind.title()} {index}"
    out = [f'<div class="ci-root"><header class="ci-result-header"><div><span class="ci-eyebrow">INSPECTION RESULTS</span>'
           f'<h3>{esc(title)}</h3><p>{esc(result["model_id"])}</p></div><div class="ci-run-meta">'
           f'<span class="ci-badge">{count} prefix{"es" if count != 1 else ""}</span>'
           f'<span class="ci-badge">{result["model_passes"]} passes</span></div></header>']
    for number, prefix in enumerate(result["prefixes"], 1):
        baseline = prefix["conditions"][0]
        zero = next(row for row in prefix["conditions"] if row.get("factor") == 0)
        delta = zero["log_probabilities"][0] - baseline["log_probabilities"][0]
        target = result["target_texts"][0]
        out.append(f'<section class="ci-prefix"><div class="ci-prefix-heading"><h4>Destination prefix {number}</h4>'
                   f'<span>Position {prefix["position"]} · offset {result["offset"]} from end</span></div>'
                   '<div class="ci-summary" role="group" aria-label="Measured component ablation">'
                   '<div class="ci-output-summary"><span class="ci-eyebrow">OUTPUT TOKEN</span>'
                   f'<strong title="{esc(repr(target))}">{esc(_short(repr(target), 26))}</strong>'
                   f'<span class="ci-change {signed(delta)}">{ratio(delta)} after ablation</span></div>'
                   f'<div class="ci-metric"><span>Baseline probability</span><strong>{probability(baseline["probabilities"][0])}</strong>'
                   '<small>Original forward pass</small></div>'
                   f'<div class="ci-metric"><span>After ablation</span><strong>{probability(zero["probabilities"][0])}</strong>'
                   f'<small>Δlog P {delta:+.4g}</small></div>'
                   f'<div class="ci-metric"><span>Activation</span><strong>{activation(baseline["activation"])}</strong>'
                   '<small>At the selected token</small></div></div>')
        if number == 1 and result.get("neighborhood"):
            out.append(native_neighborhood(result["component"], result["neighborhood"]))
        out.append('<details class="ci-panel ci-details" open><summary><span>Intervention measurements</span>'
                   f'<span class="ci-badge">{len(prefix["conditions"])} conditions</span></summary>'
                   '<div class="ci-table-wrap"><table class="ci-table"><thead><tr><th>Condition</th><th>Activation</th>'
                   + ('<th>Later component</th>' if result["hub"] is not None else '')
                   + ''.join(f'<th><code>{esc(repr(t))}</code><small>Probability · relative effect</small></th>'
                             for t in result["target_texts"]) + '</tr></thead><tbody>')
        for row in prefix["conditions"]:
            name, description, style = _condition(row)
            out.append(f'<tr class="ci-row-{style}"><td><b>{esc(name)}</b><small>{description}</small></td>'
                       f'<td class="ci-number">{activation(row["activation"])}</td>')
            if result["hub"] is not None:
                out.append(f'<td class="ci-number">{activation(row["hub_activation"])}</td>')
            for prob, logp, base_logp in zip(row["probabilities"], row["log_probabilities"], baseline["log_probabilities"]):
                change = logp - base_logp
                out.append(f'<td><div class="ci-reading"><b>{probability(prob)}</b>'
                           f'<span class="ci-change {signed(change)}">{ratio(change)}</span></div>'
                           f'<small class="ci-number">Δlog P {change:+.4g}</small></td>')
            out.append('</tr>')
        out.append('</tbody></table></div></details><section class="ci-panel"><div class="ci-panel-heading">'
                   '<div><h4>Token activations</h4><p>Baseline · hover a token for its value and position</p></div>'
                   '<span class="ci-badge">Signed value' + (' / vector norm' if kind == 'head' else '')
                   + '</span></div><div class="ci-token-strip">')
        values = prefix["activations"]
        magnitudes = [abs(v[0]) if len(v) == 1 else math.sqrt(math.fsum(x*x for x in v)) for v in values]
        maximum = max(magnitudes, default=0) or 1
        for at, (token, value, magnitude) in enumerate(zip(prefix["tokens"], values, magnitudes)):
            direction = "negative" if len(value) == 1 and value[0] < 0 else "positive"
            selected = " ci-token-selected" if at == prefix["position"] else ""
            label = f'position {at}, offset {len(values)-1-at}: {activation(value)}'
            out.append(f'<span class="ci-token {direction}{selected}" style="--ci-heat:{8+42*magnitude/maximum:.1f}%" '
                       f'title="{esc(label)}">{esc(token) or "&nbsp;"}</span>')
        out.append('</div></section></section>')
    if result.get("writes"):
        out.append('<details class="ci-panel ci-details"><summary>Vocabulary projection heuristic</summary>'
                   '<p class="ci-explanation">The output direction projected directly into vocabulary space. '
                   'Downstream layers and normalization can change its effect; use the measured results to test it.</p>'
                   '<div class="ci-vocabulary">')
        for key, label, direction in (("promotes", "Promotes", "positive"), ("suppresses", "Suppresses", "negative")):
            out.append(f'<div><span class="ci-eyebrow">{label}</span><div class="ci-token-chips">'
                       + ''.join(f'<span class="ci-change {direction}">{esc(repr(t))}</span>' for t in result["writes"][key])
                       + '</div></div>')
        out.append('</div></details>')
    out.append('<details class="ci-panel ci-details"><summary>Method and run details</summary>'
               '<p class="ci-explanation">These are measured interventions in the real model. '
               'Restoring a later component tests a proposed pathway; a rescue alone does not establish a unique circuit.</p>')
    if result["source"] is not None:
        out.append('<p class="ci-explanation">Source activation copied: ' + activation(result["source"]["activation"])
                   + '. Source output probabilities: '
                   + ', '.join(f'{esc(repr(t))}: {probability(p)}' for t, p in
                               zip(result["target_texts"], result["source"]["probabilities"])) + '</p>')
    if result["hub"] is not None:
        hub = result["hub"]
        out.append(f'<p class="ci-explanation">Restored component: layer {hub[1]} · {esc(hub[0])} {hub[2]}, '
                   f'offset {result["hub_offset"]} from the end. Its destination baseline activation is restored '
                   'while the upstream component remains zero.</p>')
    return ''.join(out) + '</details></div>'


CSS = """
#circuits-page .ci-root, #circuits-page .ci-controls {
  --ci-ink:var(--body-text-color, #202735); --ci-surface:var(--block-background-fill, #fff);
  --ci-muted:color-mix(in srgb, var(--ci-ink) 67%, var(--ci-surface));
  --ci-line:color-mix(in srgb, var(--ci-ink) 13%, var(--ci-surface));
  --ci-soft:color-mix(in srgb, var(--ci-ink) 3%, var(--ci-surface));
  --ci-blue:#2876c7; --ci-red:#bf5267; --ci-purple:#7760cf;
  color:var(--ci-ink); font-size:13px; line-height:1.5;
}
.dark #circuits-page .ci-root { --ci-blue:#75b5f4; --ci-red:#ed93a4; --ci-purple:#bba6ff; }
#circuits-page .ci-root * { box-sizing:border-box; }
#circuits-page .ci-result-header { display:flex; justify-content:space-between; gap:16px; align-items:center; padding:4px 0 20px; }
#circuits-page .ci-root h3 { margin:4px 0 !important; font-size:23px; font-weight:650; letter-spacing:-.5px; line-height:1.3; }
#circuits-page .ci-root h4 { margin:0 !important; font-size:14px; font-weight:600; }
#circuits-page .ci-result-header p, #circuits-page .ci-panel-heading p { color:var(--ci-muted); margin:3px 0 0; font-size:12px; }
#circuits-page .ci-eyebrow { color:var(--ci-muted); font-size:10px; font-weight:650; letter-spacing:1.3px; text-transform:uppercase; }
#circuits-page .ci-run-meta, #circuits-page .ci-token-chips { display:flex; flex-wrap:wrap; gap:7px; }
#circuits-page .ci-badge { display:inline-flex; align-items:center; border:1px solid var(--ci-line); background:var(--ci-soft); border-radius:6px; padding:3px 8px; font-size:11px; color:var(--ci-muted); white-space:nowrap; font-weight:450; }
#circuits-page .ci-prefix { margin-bottom:26px; }
#circuits-page .ci-prefix-heading { display:flex; justify-content:space-between; gap:12px; align-items:center; margin:0 0 10px; }
#circuits-page .ci-prefix-heading>span { color:var(--ci-muted); font-size:11px; }
#circuits-page .ci-summary { display:grid; grid-template-columns:1.15fr repeat(3,1fr); border:1px solid var(--ci-line); border-radius:12px; background:var(--ci-surface); margin-bottom:16px; overflow:hidden; }
#circuits-page .ci-summary>div { padding:17px 18px; min-width:0; }
#circuits-page .ci-summary>div+div { border-left:1px solid var(--ci-line); }
#circuits-page .ci-output-summary { background:color-mix(in srgb, var(--ci-purple) 5%, var(--ci-surface)); display:flex; align-items:flex-start; flex-direction:column; gap:4px; }
#circuits-page .ci-output-summary>strong { font-size:19px; font-weight:600; font-family:ui-monospace,monospace; max-width:100%; overflow:hidden; text-overflow:ellipsis; }
#circuits-page .ci-metric { display:flex; flex-direction:column; justify-content:center; gap:4px; }
#circuits-page .ci-metric>span { color:var(--ci-muted); font-size:11px; }
#circuits-page .ci-metric>strong { font-size:20px; font-weight:550; letter-spacing:-.5px; font-variant-numeric:tabular-nums; }
#circuits-page .ci-metric>small { color:var(--ci-muted); font-size:10px; }
#circuits-page .ci-panel { border:1px solid var(--ci-line); border-radius:12px; background:var(--ci-surface); margin:0 0 16px; overflow:hidden; }
#circuits-page .ci-panel-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:17px 20px; }
#circuits-page .ci-canvas { overflow:auto; background:radial-gradient(var(--ci-line) .7px, transparent .7px) 0 0/16px 16px, var(--ci-soft); border-block:1px solid var(--ci-line); }
#circuits-page .ci-canvas svg { display:block; width:100%; min-width:620px; height:auto; max-width:none; }
#circuits-page .ci-column-label { fill:var(--ci-muted); font-size:10px; font-weight:600; letter-spacing:1.2px; }
#circuits-page .ci-flow { fill:none; opacity:.58; stroke:var(--ci-muted); }
#circuits-page .ci-flow.positive { stroke:var(--ci-blue); }
#circuits-page .ci-flow.negative { stroke:var(--ci-red); }
#circuits-page .ci-neighbor rect { fill:var(--ci-surface); stroke:var(--ci-line); stroke-width:1; }
#circuits-page .ci-feature-node rect { fill:var(--ci-surface); stroke:var(--ci-line); stroke-width:1; }
#circuits-page .ci-feature-node.sel rect { fill:color-mix(in srgb, var(--ci-purple) 9%, var(--ci-surface)); stroke:var(--ci-purple); stroke-width:1.5; }
#circuits-page .ci-feature-node:hover rect { stroke:var(--ci-purple); }
#circuits-page .ci-neighbor:hover rect { stroke:var(--ci-purple); }
#circuits-page .ci-neighbor:hover .ci-node-title { fill:var(--ci-purple); }
#circuits-page .ci-node-title { fill:var(--ci-ink); font-size:12px; font-weight:550; }
#circuits-page .ci-node-detail { fill:var(--ci-muted); font-size:10px; font-family:ui-monospace,monospace; }
#circuits-page .ci-node-dot { fill:var(--ci-muted); }
#circuits-page .positive .ci-node-dot, #circuits-page .ci-empty-dots { fill:var(--ci-blue); }
#circuits-page .negative .ci-node-dot { fill:var(--ci-red); }
#circuits-page .ci-hub-halo { fill:color-mix(in srgb, var(--ci-purple) 8%, transparent); }
#circuits-page .ci-hub { fill:color-mix(in srgb, var(--ci-purple) 13%, var(--ci-surface)); stroke:var(--ci-purple); stroke-width:1.5; }
#circuits-page .ci-hub-kicker, #circuits-page .ci-hub-layer, #circuits-page .ci-hub-index { fill:var(--ci-ink); text-anchor:middle; }
#circuits-page .ci-hub-kicker { fill:var(--ci-purple); font-size:9px; font-weight:650; letter-spacing:1.3px; }
#circuits-page .ci-hub-layer { font-size:15px; font-weight:650; }
#circuits-page .ci-hub-index { font-size:13px; }
#circuits-page .ci-legend { display:flex; flex-wrap:wrap; gap:7px 18px; padding:12px 20px; color:var(--ci-muted); font-size:10px; }
#circuits-page .ci-legend>span { display:inline-flex; align-items:center; gap:6px; }
#circuits-page .ci-legend i { width:14px; height:2px; border-radius:2px; }
#circuits-page .ci-legend .positive { background:var(--ci-blue); }
#circuits-page .ci-legend .negative { background:var(--ci-red); }
#circuits-page .ci-method { padding:0 20px 13px; color:var(--ci-muted); font-size:11px; }
#circuits-page .ci-method summary { cursor:pointer; width:fit-content; }
#circuits-page .ci-method p { margin:8px 0 0; max-width:85ch; }
#circuits-page .ci-details>summary { padding:15px 20px; cursor:pointer; font-weight:550; display:flex; gap:10px; align-items:center; }
#circuits-page .ci-details>summary::before { content:'›'; color:var(--ci-muted); font-size:19px; line-height:1; transition:transform .15s; }
#circuits-page .ci-details[open]>summary::before { transform:rotate(90deg); }
#circuits-page .ci-details>summary .ci-badge { margin-left:auto; }
#circuits-page .ci-table-wrap { overflow:auto; }
#circuits-page .ci-table { width:100%; border-collapse:collapse; border:none !important; margin:0 !important; text-align:left; }
#circuits-page .ci-table th, #circuits-page .ci-table td { border:none !important; border-top:1px solid var(--ci-line) !important; background:transparent !important; padding:11px 20px !important; font-size:12px; text-align:left; vertical-align:middle; }
#circuits-page .ci-table th { font-weight:500; color:var(--ci-muted); background:var(--ci-soft) !important; }
#circuits-page .ci-table code { font:inherit; color:var(--ci-ink); background:none; padding:0; }
#circuits-page .ci-table small { display:block; margin-top:3px; font-size:10px; color:var(--ci-muted); font-weight:400; }
#circuits-page .ci-table b { font-weight:550; }
#circuits-page .ci-table tr.ci-row-ablate { background:color-mix(in srgb, var(--ci-purple) 4%, var(--ci-surface)); }
#circuits-page .ci-table tr:hover { background:var(--ci-soft); }
#circuits-page .ci-number { font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }
#circuits-page .ci-reading { display:flex; align-items:center; gap:12px; font-variant-numeric:tabular-nums; }
#circuits-page .ci-change { display:inline-flex; width:fit-content; white-space:nowrap; border-radius:5px; padding:2px 6px; font:10px ui-monospace,monospace; background:var(--ci-soft); color:var(--ci-muted); }
#circuits-page .ci-change.positive { color:var(--ci-blue); background:color-mix(in srgb, var(--ci-blue) 8%, var(--ci-surface)); }
#circuits-page .ci-change.negative { color:var(--ci-red); background:color-mix(in srgb, var(--ci-red) 8%, var(--ci-surface)); }
#circuits-page .ci-token-strip { padding:0 20px 20px; display:flex; flex-wrap:wrap; gap:4px; font:12px/1.8 ui-monospace,monospace; }
#circuits-page .ci-token { white-space:pre-wrap; overflow-wrap:anywhere; border-radius:5px; padding:3px 7px; border-bottom:2px solid transparent; background:color-mix(in srgb, var(--ci-blue) var(--ci-heat), var(--ci-surface)); }
#circuits-page .ci-token.negative { background:color-mix(in srgb, var(--ci-red) var(--ci-heat), var(--ci-surface)); }
#circuits-page .ci-token-selected { outline:1.5px solid var(--ci-purple); outline-offset:1px; }
#circuits-page .ci-explanation { color:var(--ci-muted); font-size:12px; margin:0; padding:0 20px 16px; }
#circuits-page .ci-vocabulary { display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:0 20px 20px; }
#circuits-page .ci-token-chips { margin-top:7px; }
#circuits-page .ci-empty { min-height:470px; padding:64px 24px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; border:1px dashed var(--ci-line); border-radius:14px; background:var(--ci-soft); }
#circuits-page .ci-empty svg { width:220px; height:110px; margin:24px 0; color:var(--ci-purple); }
#circuits-page .ci-empty h3 { font-size:20px; }
#circuits-page .ci-empty p { color:var(--ci-muted); max-width:350px; margin:8px 0 28px; }
#circuits-page .ci-empty-steps { display:flex; gap:22px; color:var(--ci-muted); font-size:11px; }
#circuits-page .ci-empty-steps b { color:var(--ci-purple); margin-right:5px; }
#circuits-page .ci-controls { gap:12px; flex-wrap:nowrap !important; align-self:flex-start; position:sticky; top:8px; max-height:calc(100vh - 80px); overflow-y:auto; overflow-x:hidden; padding-right:3px; }
#circuits-page .ci-controls>* { flex-shrink:0; }
#circuits-page .ci-controls .ci-control-section { border:1px solid var(--ci-line); border-radius:10px; overflow:hidden; background:var(--ci-surface) !important; }
#circuits-page .ci-controls .ci-control-section>div, #circuits-page .ci-controls .ci-control-section .form { background:transparent !important; }
#circuits-page .ci-controls .ci-control-heading { padding:12px 12px 4px !important; background:transparent !important; min-height:0 !important; }
#circuits-page .ci-controls .ci-control-heading p { color:var(--ci-muted); font-size:10px !important; font-weight:600; letter-spacing:1px; margin:0 !important; }
#circuits-page .ci-controls label>span, #circuits-page .ci-controls .block-label>span { font-size:12px; }
#circuits-page .ci-controls .info { font-size:11px; line-height:1.5; color:var(--ci-muted) !important; }
#circuits-page .ci-controls textarea { font-size:12px; line-height:1.6; }
#circuits-page .ci-controls .ci-actions { gap:8px; padding:10px 0 4px; position:sticky; bottom:0; z-index:5; background:var(--ci-surface); border-top:1px solid var(--ci-line); }
#circuits-page .ci-controls .ci-actions button { border-radius:8px; font-size:12px; }
#circuits-page .ci-progress { min-height:20px !important; font-size:12px; }
#circuits-page .ci-progress p { font-size:11px; margin:0; color:var(--body-text-color); }
#circuits-page .ci-download { margin-top:0; }
#circuits-page .ci-intro { max-width:900px; font-size:12px; color:var(--body-text-color); }
@media(max-width:1000px) { #circuits-page .ci-summary { grid-template-columns:1fr 1fr; } #circuits-page .ci-summary>div:nth-child(3) { border-left:0; } #circuits-page .ci-summary>div:nth-child(n+3) { border-top:1px solid var(--ci-line); } }
@media(max-width:640px) { #circuits-page .ci-controls { position:static; max-height:none; overflow:visible; } #circuits-page .ci-result-header, #circuits-page .ci-prefix-heading { align-items:flex-start; flex-direction:column; gap:8px; } #circuits-page .ci-vocabulary { grid-template-columns:1fr; } #circuits-page .ci-summary>div { padding:14px; } }
"""
