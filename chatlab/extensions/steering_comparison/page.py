"""Browse paired steering effects, with offline results and model cancellation."""
import html
import json
import math
import tempfile
from pathlib import Path

import gradio as gr

from chatlab.extension_api import ProjectionCancelled, write_private_text
from chatlab.extensions.result_runs import ResultRuns
from . import analysis, experiment

CSS = """
#steering-comparison-page {overflow-y:auto; min-height:0; padding:12px;}
#steering-comparison-page .sc-table {overflow:auto; max-height:650px;}
#steering-comparison-page table {border-collapse:collapse; width:100%; font-size:13px;}
#steering-comparison-page svg {vertical-align:middle;}
#steering-comparison-page td, #steering-comparison-page th {padding:7px; text-align:right; white-space:nowrap; border:0; border-bottom:1px solid var(--border-color-primary);}
#steering-comparison-page td:first-child, #steering-comparison-page th:first-child {text-align:left;}
#steering-comparison-page .sc-up {color:#30845b;} #steering-comparison-page .sc-down {color:#b85b4f;}
"""


class ComparisonRuns(ResultRuns):
    def _stage(self, owner, result):
        directory = self._downloads.get(owner)
        if directory is None:
            directory = self._downloads[owner] = tempfile.TemporaryDirectory(prefix="chatlab-steering-comparison-")
        path = Path(directory.name) / "steering-comparison.json"
        write_private_text(path, json.dumps(result, allow_nan=False))
        return str(path)


def render(result, vector, point, block, mode, sort, floor, text, specific):
    if result is None:
        return "", "", []
    entry = result["vectors"][int(vector)]
    group = entry["points"][point][mode]
    rows = analysis.select(group["blocks"][str(block)], text or "", float(floor) / 100, sort, specific)
    def number(x):
        return "—" if x is None else f"{x:+.2f}"
    def percent(x):
        return f"{x:.0%}"
    max_t = max((abs(r["t"]) for r in rows if r["t"] is not None), default=1) or 1
    def table(items):
        if not items:
            return '<p>No matching tokens moved in this direction.</p>'
        body = []
        for row in items:
            # A small trace shows geometric rank changes at every captured block.
            factors, bars = [], []
            for bi, b in enumerate(result["blocks"]):
                factor = group["traces"][str(row["id"])][bi]
                if factor is not None:
                    factors.append(f"{b}: ×{factor:.2f}")
                    height = min(10, abs(math.log(factor)) * 3)
                    color = "#30845b" if factor >= 1 else "#b85b4f"
                    x = bi * 11 + 2
                    y = 12 - height if factor >= 1 else 12
                    bars.append(f'<rect x="{x}" y="{y:.2f}" width="8" height="{height:.2f}" fill="{color}"/>')
                    if b == block:
                        bars.append(f'<rect x="{x - 1}" y="1" width="10" height="22" fill="none" stroke="currentColor" rx="2"/>')
            change = "sc-up" if row["factor"] >= 1 else "sc-down"
            trace = ('<svg width="96" height="24" viewBox="0 0 96 24" role="img" aria-label="Rank change across blocks">'
                     '<path d="M0 12H96" stroke="currentColor" opacity=".15"/>' + "".join(bars) + '</svg>')
            bar_width = 55 * (abs(row["t"] or 0) / max_t)
            bar_color = "#30845b" if (row["t"] or 0) >= 0 else "#b85b4f"
            bar_start = 58 if (row["t"] or 0) >= 0 else 58 - bar_width
            t_bar = (f'<svg width="116" height="18" aria-hidden="true"><rect x="0" y="3" width="116" height="12" fill="currentColor" opacity=".06"/>'
                     f'<path d="M58 0V18" stroke="currentColor" opacity=".25"/><rect x="{bar_start:.2f}" y="3" width="{bar_width:.2f}" height="12" rx="2" fill="{bar_color}"/></svg>')
            body.append(f'<tr><td><code>{html.escape(row["token"])}</code> <small>#{row["id"]}</small></td>'
                        f'<td title="{html.escape("; ".join(factors))}">{trace}</td>'
                        f'<td class="{change}">{t_bar} {number(row["t"])}</td><td>×{row["factor"]:.2f}</td>'
                        f'<td>{percent(row["top50"])} / {percent(row["base50"])}</td>'
                        f'<td>{percent(row["top10"])} / {percent(row["base10"])}</td>'
                        f'<td>{number(row["random_t"])}</td></tr>')
        return ('<div class="sc-table"><table><thead><tr><th>Token</th><th>Blocks: rank change</th>'
                '<th>Rank improvement (t)</th><th>Rank change</th><th>Top 50: steered / comparison</th>'
                '<th>Top 10: steered / comparison</th><th>Random t</th></tr></thead><tbody>'
                + "".join(body) + '</tbody></table></div>')
    up = [r for r in rows if r["factor"] > 1 + 1e-10][:100]
    down = list(reversed([r for r in rows if r["factor"] < 1 - 1e-10]))[:100]
    shown = "<h3>Moved up</h3>" + table(up) + "<h3>Moved down</h3>" + table(down)
    relevant = [e for e in result["examples"] if e["point"] == point]
    outcomes = [e["recovered"] for e in relevant if e["recovered"] is not None]
    recovery = f" Supplied example return-to-task rate: {sum(outcomes) / len(outcomes):.0%} ({len(outcomes)} labeled examples)." if outcomes else ""
    summary = (f"**{html.escape(entry['name'])}** · {group['n']} paired examples · block {block} · {mode.lower()} change."
               f"{recovery} Model: {html.escape(result['model_id'])}. Showing up to 100 tokens in each direction; {len(rows):,} match the filters.")
    return summary, shown, [[e["point"], e["text"], e["recovered"]] for e in relevant]


def build_page(context):
    runs = ComparisonRuns()
    with gr.Column(elem_id="steering-comparison-page"):
        owner = gr.State(value=runs.new_owner, delete_callback=runs.forget)
        result = gr.State(None)
        gr.Markdown("# Steering comparison\nCompare full-vocabulary Jacobian ranks over paired examples and steering vectors. "
                    "Each prefix ends at the token **before the decision**; its final activation is read. "
                    "These are fixed-prefix interventions, averaged across examples; they do not establish what caused a generated reply to return to task.")
        with gr.Accordion("Collect readouts", open=True):
            models = gr.Button("Open Models", size="sm")
            context.navigation.open_models(models)
            gr.Markdown("Import a fitted Jacobian lens in Chat first. Use the exact text sent to the model, including any chat template. "
                        "Give each example a decision point name to group it. The optional return-to-task label describes the supplied example, "
                        "not an outcome generated by this tool: 1 returned, 0 did not, empty unknown.")
            samples = gr.Dataframe(headers=["Decision point", "Prefix through token before decision", "Returned to task (optional)"],
                                   datatype=["str", "str", "str"], type="array", col_count=(3, "fixed"),
                                   value=[["Return to task", "", ""]], row_count=4, label="Examples", interactive=True)
            vectors = gr.File(label="Steering vectors (up to six)", file_count="multiple", file_types=[".json"], type="filepath")
            with gr.Row():
                blocks = gr.Textbox(label="Read decoder blocks (zero-based, comma-separated)", value="0")
                strength = gr.Number(label="Strength multiplier", value=1)
                seed = gr.Number(label="Random control seed", value=0, precision=0)
            gr.Markdown("Vectors retain their saved strengths, multiplied by the value above. The random control uses one seeded direction "
                        "per vector with the same norm, at the same injection block. All interventions cover every prefix position. "
                        "**Total** compares against no steering. **Computed** compares against adding that same vector directly at each read block, "
                        "to distinguish later processing from the immediate vector addition. The random control uses its own direct-addition baseline.")
            with gr.Row():
                run = gr.Button("Collect comparison", variant="primary")
                stop = gr.Button("Stop")
            status = gr.Markdown("")
        with gr.Row():
            vector = gr.Dropdown([], label="Vector", interactive=True)
            point = gr.Dropdown([], label="Decision point", interactive=True)
            block = gr.Dropdown([], label="Decoder block", interactive=True)
            mode = gr.Radio(["Total", "Computed"], value="Total", label="Change")
        with gr.Row():
            sort = gr.Dropdown(["Rank improvement (t)", "Top-50 entry gain", "Rank change factor"], value="Rank improvement (t)", label="Rank by")
            floor = gr.Slider(0, 100, value=50, step=1, label="In top 50 for at least this % of examples (either side)")
            text = gr.Textbox(label="Filter tokens", placeholder="e.g. here, okay, answer")
            specific = gr.Checkbox(label="Specific only", value=False)
        with gr.Accordion("How to read the statistics", open=False):
            gr.Markdown("Rank change is the geometric mean of comparison rank ÷ steered rank: ×3 means three times closer to the top. "
                    "The paired t statistic uses log rank improvements; it is absent for fewer than two examples or zero variance. "
                    "Specific only is a descriptive heuristic: real t > 2 and at least 2 above random t, not a significance test corrected for vocabulary search. "
                    "Only tokens reaching the top 50 in any real, comparison or random readout are retained. Top-k columns always use actual ranks.")
        summary = gr.Markdown("")
        table = gr.HTML("")
        with gr.Accordion("Examples", open=False):
            recorded = gr.Dataframe(headers=["Decision point", "Prefix", "Returned to task"], interactive=False)
        with gr.Row():
            download = gr.File(label="Download comparison", interactive=False)
            upload = gr.File(label="Open saved comparison", file_types=[".json"], type="filepath")

    controls = [result, vector, point, block, mode, sort, floor, text, specific]
    def show(data):
        names = [(v["name"], i) for i, v in enumerate(data["vectors"])]
        points = list(data["vectors"][0]["points"])
        first = data["blocks"][0]
        rendered = render(data, 0, points[0], first, "Total", "Rank improvement (t)", 50, "", False)
        return (data, gr.update(choices=names, value=0), gr.update(choices=points, value=points[0]),
                gr.update(choices=data["blocks"], value=first), "Total", "Rank improvement (t)", 50, "", False, *rendered)

    outputs = [result, vector, point, block, mode, sort, floor, text, specific, summary, table, recorded, download, status]
    def publish(view, turn, data):
        published = runs.publish(view, turn, data, lambda path: (*show(data), path, "Comparison ready."))
        return published if published is not None else (gr.skip(),) * len(outputs)

    def collect(view, rows, paths, chosen, scale, random_seed):
        turn = runs.start(view)
        try:
            with context.models.open_session() as session:
                runs.attach(view, turn, session)
                steps = experiment.run(session, rows, paths, chosen, float(scale), random_seed)
                try:
                    while True:
                        try:
                            message = next(steps)
                        except StopIteration as done:
                            data = done.value
                            break
                        yield (*([gr.skip()] * (len(outputs) - 1)), message if runs.live(view, turn) else gr.skip())
                finally:
                    steps.close()
            yield publish(view, turn, data)
        except ProjectionCancelled:
            yield (*([gr.skip()] * (len(outputs) - 1)), "Stopped." if runs.live(view, turn) else gr.skip())
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            runs.finish(view, turn)

    def opened(path, view):
        if not path:
            return (gr.skip(),) * len(outputs)
        try:
            data = experiment.read_result(path)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise gr.Error(f"Could not open comparison: {exc}") from exc
        turn = runs.start(view)
        return publish(view, turn, data)

    event = run.click(collect, [owner, samples, vectors, blocks, strength, seed], outputs,
                      concurrency_id="steering-comparison", show_progress="hidden")
    opening = upload.upload(opened, [upload, owner], outputs, concurrency_id="steering-comparison")
    def cancel(view):
        runs.cancel(view)
        return "Stopped."
    stop.click(cancel, owner, status, queue=False, cancels=[event, opening])
    for control in controls[1:]:
        control.input(render, controls, [summary, table, recorded], concurrency_id="steering-comparison", show_progress="hidden")
