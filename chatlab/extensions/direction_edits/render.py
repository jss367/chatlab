"""The result as HTML: coordinate heatmaps, recovery by block, the lens table and its differences."""
from __future__ import annotations

import html

import numpy as np

from chatlab.extensions.probes.page import mix

# Columns past this are left out of the heatmaps; the download keeps them all.
HEAT_COLUMNS = 512
PASSES = (("reference", "Reference, no injection"), ("injected", "Injected"),
          ("edited", "Injected and edited"), ("random", "Injected, random control"))
SETTING_NAMES = (("no_edit", "No edit"), ("edit", "Edit"), ("random", "Random control"))


def _fill(share, stops):
    """A colour along the sequential stops, ``share`` from 0 (coolest) to 1 (warmest)."""
    share = min(max(share, 0.0), 1.0) * (len(stops) - 1)
    index = min(int(share), len(stops) - 2)
    return mix(stops[index], stops[index + 1], share - index)


def _percent(value, digits=0):
    if value is None:
        return "–"
    # A value that rounds to zero is shown without a sign.
    return f"{round(value, digits + 2) + 0.0:.{digits}%}"


def _number(value):
    return "–" if value is None else f"{value:.4f}"


def _signed(value):
    return "–" if value is None else f"{value:+.1f}%"


def _marks(result, length):
    inputs = result["inputs"]
    marks = []
    if inputs["injection"] is not None:
        marks.append(("injected", inputs["injection"]["tokens"]))
    marks.append(("edited", inputs["edit"]["tokens"]))
    if inputs["readout"] is not None:
        marks.append(("read", inputs["readout"]["tokens"]))
    rows = []
    for name, (first, last) in marks:
        cells = "".join(f'<td class="{"de-on" if first <= token <= last else ""}"></td>'
                        for token in range(1, length + 1))
        rows.append(f'<tr class="de-mark"><th>{name}</th>{cells}</tr>')
    return "".join(rows)


def heatmaps(result, index, palette, display):
    """The condition's coordinate at every block and passage token, one grid per pass.

    Each block's row has its own scale, shared by every pass, from its 5th
    percentile (cool) to its 95th (warm): the residual stream grows with
    depth, so one scale for every block would leave the early blocks blank,
    and the first token's coordinate is often far from the rest, so a row
    scaled to its extremes would leave every other token one color.
    """
    if result is None:
        return ""
    condition = result["conditions"][index]
    passes = [(key, label) for key, label in PASSES if key in condition["coordinates"]]
    tokens = result["passage_tokens"][:HEAT_COLUMNS]
    length = len(tokens)
    edited = set(result["edited_blocks"])
    stops = palette["sequential"]
    blocks = len(condition["coordinates"]["reference"])
    scales = []
    for block in range(blocks):
        rows = [condition["coordinates"][key][block] for key, _ in passes]
        if rows[0] is None:
            scales.append(None)
            continue
        values = np.asarray([row[:length] for row in rows], dtype=np.float64)
        scales.append((float(np.percentile(values, 5)), float(np.percentile(values, 95))))
    grids = []
    for key, label in passes:
        rows = []
        for block in range(blocks):
            row = condition["coordinates"][key][block]
            marked = ' class="de-edited"' if block in edited else ""
            if row is None:
                cells = f'<td class="de-none" colspan="{length}"></td>'
            else:
                low, high = scales[block]
                spread = high - low
                cells = "".join(
                    f'<td style="background:{_fill((value - low) / spread if spread else 0.5, stops)}" '
                    f'title="{html.escape(display(text))} · token {token} · block {block} · {value:.3f}"></td>'
                    for token, (text, value) in enumerate(zip(tokens, row[:length]), start=1))
            rows.append(f"<tr{marked}><th>block {block}</th>{cells}</tr>")
        grids.append(f'<h4>{html.escape(label)}</h4><div class="de-heat"><table>{_marks(result, length)}'
                     f'{"".join(rows)}</table></div>')
    total = len(result["passage_tokens"])
    note = f"<p>Showing the first {HEAT_COLUMNS} of {total} passage tokens.</p>" if total > HEAT_COLUMNS else ""
    return (f'<p>The coordinate along each block\'s direction for <b>{html.escape(condition["name"])}</b>. Each '
            "block's row runs from its 5th percentile (cool) to its 95th (warm) on one scale for every pass. "
            "Outlined rows are edited. Hover a cell for its token and value.</p>" + note + "".join(grids))


def recovery_table(result):
    """Recovery at every block for every condition, edited blocks marked."""
    if result is None:
        return ""
    conditions = result["conditions"]
    random = bool(conditions[0]["recovery"]["random"])
    edited = set(result["edited_blocks"])
    head = "".join(f"<th>{html.escape(c['name'])}</th>" + ("<th>random</th>" if random else "") for c in conditions)
    rows = []
    for block in range(len(conditions[0]["recovery"]["edited"])):
        cells = "".join(
            f"<td>{_percent(c['recovery']['edited'][block])}</td>"
            + (f"<td>{_percent(c['recovery']['random'][block])}</td>" if random else "")
            for c in conditions)
        label = f"{block} (edited)" if block in edited else str(block)
        marked = ' class="de-edited"' if block in edited else ""
        rows.append(f"<tr{marked}><th>{label}</th>{cells}</tr>")
    return (f'<div class="de-table"><table><thead><tr><th>Block</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def headline(result):
    """After the edit, how much of the injected shift each condition's later blocks rebuilt."""
    if result is None:
        return ""
    if result["inputs"]["injection"] is None:
        return ("Nothing was injected, so the injected pass is the reference and recovery is undefined. "
                "Read the heatmaps and the lens table instead.")
    last_edited = max(result["edited_blocks"])
    lines = []
    for condition in result["conditions"]:
        later = [(value, block) for block, value in enumerate(condition["recovery"]["edited"])
                 if block > last_edited and value is not None]
        if not later:
            lines.append(f"- **{condition['name']}**: no block after the edit to read.")
            continue
        final_value, final_block = later[-1]
        peak_value, peak_block = max(later)
        lines.append(f"- **{condition['name']}**: {_percent(final_value)} of the injected shift is back at block "
                     f"{final_block}, the last one read; the most is {_percent(peak_value)}, at block {peak_block}.")
    return ("Recovery is the mean over the edited tokens of (edited − reference), divided by the mean of "
            "(injected − reference), at each block.\n\n" + "\n".join(lines))


def lens_table(result):
    if result is None or result["target"] is None:
        return ""
    settings = [(key, label) for key, label in SETTING_NAMES
                if key != "random" or result["inputs"]["edit"]["random_control"]]
    head = "".join(f"<th>{label}</th>" for _key, label in settings)
    rows = "".join(
        f"<tr><th>{html.escape(c['name'])}</th>" + "".join(f"<td>{_number(c['lens'][key])}</td>" for key, _ in settings)
        + "</tr>" for c in result["conditions"])
    return (f'<div class="de-table"><table><thead><tr><th>Condition</th>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def target_note(result):
    if result is None or result["target"] is None:
        return ""
    target, readout = result["target"], result["inputs"]["readout"]
    spelled = ", ".join(f"`{text.replace('`', chr(39))}`" for text in target["tokens"])
    word = (f"The target is one token, {spelled}." if len(target["token_ids"]) == 1 else
            f"The target spans {len(target['token_ids'])} tokens, {spelled}; it is read as the mean of their "
            "unembeddings.")
    lens = result["lens"]
    return (f"{word} Mean log probability through the lens {lens['name']} over tokens "
            f"{readout['tokens'][0]}–{readout['tokens'][1]} and blocks {readout['blocks'][0]}–{readout['blocks'][1]}.")


def differences_table(result):
    if result is None or not result["differences"]:
        return ""
    random = result["inputs"]["edit"]["random_control"]
    head = "<th>Difference</th><th>No edit</th><th>Edit</th><th>Change</th>" + (
        "<th>Random control</th><th>Change</th>" if random else "")
    rows = []
    for item in result["differences"]:
        cells = (f"<td>{_number(item['no_edit'])}</td><td>{_number(item['edit'])}</td>"
                 f"<td>{_signed(item['edit_change'])}</td>")
        if random:
            cells += f"<td>{_number(item['random'])}</td><td>{_signed(item['random_change'])}</td>"
        rows.append(f"<tr><th>{html.escape(item['name'])}: {html.escape(item['minuend'])} − "
                    f"{html.escape(item['subtrahend'])}</th>{cells}</tr>")
    return (f'<div class="de-table"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
