"""Durable fork origins and an escaped, model-free conversation comparison."""

from __future__ import annotations

import difflib
import html
import json
import re

from chatlab.conversation import branch_title


def validate_origin(value):
    """Validate coordinates without requiring an ancestor to still exist."""
    if not isinstance(value, dict) or not isinstance(value.get("parent"), str):
        raise ValueError("A fork origin needs a parent branch name.")
    if value.get("kind") not in {"copy", "message", "token", "prompt"}:
        raise ValueError("Unknown fork origin kind.")
    for key in ("turn", "token", "original_id", "message_count"):
        if key in value and (type(value[key]) is not int or value[key] < (1 if key == "token" else 0)):
            raise ValueError(f"Invalid fork {key}.")
    for key in ("original", "replacement", "part"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"Invalid fork {key}.")
    ids = value.get("replacement_ids", [])
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Invalid replacement token IDs.")
    return dict(value)


def escape(value):
    return html.escape(str(value), quote=True)


def value_text(value):
    if isinstance(value, str):
        return value or "(empty)"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


LABELS = {
    "temperature": "Temperature", "top_p": "Top-p", "top_k": "Top-k",
    "skip_top_below": "Minimum probability", "max_new_tokens": "Maximum new tokens",
    "seed": "Seed", "system_prompt": "System prompt", "keep_reasoning": "Include reasoning",
    "thinking_mode": "Thinking mode", "model": "Model", "steering": "Steering",
    "assistant_prefill": "Assistant prefill",
}


def configuration(forks, name):
    """Separate the next reply's controls from the last reply's recorded settings."""
    result = {("Current settings", key): value
              for key, value in forks.get("sampling", {}).get(name, {}).items()}
    turn = next((t for t in reversed(forks["branches"][name]) if t["role"] == "assistant"), {})
    recorded = dict(turn.get("generation_settings") or {})
    recorded.update({key: turn[key] for key in ("model", "thinking_mode", "steering") if key in turn})
    result.update({("Latest reply", key): value for key, value in recorded.items()})
    return result


def differences(forks, left, right):
    a, b = configuration(forks, left), configuration(forks, right)
    return [(group, LABELS.get(key, key.replace("_", " ").capitalize()),
             value_text(a[(group, key)]) if (group, key) in a else "Not recorded",
             value_text(b[(group, key)]) if (group, key) in b else "Not recorded")
            for group, key in sorted(a.keys() | b.keys())
            if (group, key) not in a or (group, key) not in b or a[group, key] != b[group, key]]


def origin_text(origin):
    if not origin:
        return "Root conversation · fork origin not recorded"
    parent = origin["parent"]
    kind = origin["kind"]
    if kind == "copy":
        if origin.get("message_count") == 0:
            return f"Copied empty conversation from {parent}"
        return f"Copied from {parent} through message {origin.get('turn', 0) + 1}"
    if kind == "message":
        return f"From {parent} at message {origin.get('turn', 0) + 1}"
    token = "Prompt token" if kind == "prompt" else "Token"
    replacement = origin.get("replacement")
    change = f"{origin.get('original', '')!r} → {replacement!r}" if replacement is not None else f"{origin.get('original', '')!r} → resampled"
    return f"From {parent} · message {origin.get('turn', 0) + 1} · {token.lower()} {origin.get('token', '?')} · {change}"


def tree_html(forks, selection):
    branches = forks["branches"]
    origins = forks.get("origins", {})
    children = {name: [] for name in branches}
    roots = []
    for name in branches:
        parent = origins.get(name, {}).get("parent")
        if parent in branches and parent != name:
            children[parent].append(name)
        else:
            roots.append(name)
    # Iterative traversal also makes malformed cyclic imports safe to display.
    visited = set()
    parts = ['<div class="fork-tree" aria-label="Conversation forks"><ul>']
    for root in [*roots, *branches]:
        stack = [(root, False)]
        while stack:
            name, closing = stack.pop()
            if closing:
                parts.append('</ul></li>')
                continue
            if name in visited:
                continue
            visited.add(name)
            active = name == forks.get("active")
            parts.append(f'<li><article class="fork-node{" active" if active else ""}">')
            parts.append(f'<header><strong>{escape(name)}</strong>{"<span>On screen</span>" if active else ""}</header>')
            title = branch_title(branches[name]) or "No messages yet"
            parts.append(f'<p class="fork-title">{escape(title)}</p>')
            origin = origins.get(name)
            parts.append(f'<p class="fork-origin">{escape(origin_text(origin))}</p>')
            parent = (origin or {}).get("parent")
            if parent in branches:
                rows = differences(forks, parent, name)
                if rows:
                    parts.append(f'<details><summary>{len(rows)} settings differ from parent</summary><ul>')
                    for group, key, old, new in rows:
                        parts.append(f'<li>{escape(group)} · {escape(key)}: <code>{escape(old)}</code> → <code>{escape(new)}</code></li>')
                    parts.append('</ul></details>')
                else:
                    parts.append('<p class="fork-origin">No recorded settings differ from parent</p>')
            elif parent:
                parts.append('<p class="fork-origin">Parent was deleted or is unavailable</p>')
            parts.append('<div class="fork-actions">')
            for slot in ("A", "B"):
                selected = selection.get(slot) == name
                parts.append(f'<button type="button" data-fork-name="{escape(name)}" data-fork-slot="{slot}" aria-pressed="{str(selected).lower()}" aria-label="Select {escape(name)} as branch {slot}">{"Selected" if selected else "Compare as"} {slot}</button>')
            parts.append('</div></article><ul>')
            stack.append((name, True))
            stack.extend((child, False) for child in reversed(children[name]))
    parts.append('</ul></div>')
    return ''.join(parts)


def text_diff(a, b):
    """Word differences with preserved whitespace, independently escaped on each side."""
    left, right = [], []
    aa, bb = re.findall(r'\s+|\S+', a), re.findall(r'\s+|\S+', b)
    # Bound worst-case work for very long transcripts. The whole changed
    # passage is still shown, just without fine-grained highlighting.
    if len(aa) + len(bb) > 12000:
        return escape(a), escape(b)
    for tag, i, j, k, end in difflib.SequenceMatcher(None, aa, bb).get_opcodes():
        old, new = escape(''.join(aa[i:j])), escape(''.join(bb[k:end]))
        left.append(old if tag == 'equal' else f'<del>{old}</del>' if old else '')
        right.append(new if tag == 'equal' else f'<ins>{new}</ins>' if new else '')
    return ''.join(left), ''.join(right)


def comparison_html(forks, selection):
    a, b = selection.get("A"), selection.get("B")
    if a not in forks["branches"] or b not in forks["branches"]:
        return '<div class="fork-comparison"><p>Select A and B on two tree nodes to compare their messages, reasoning, and settings.</p></div>'
    if a == b:
        return '<div class="fork-comparison"><p>Select a different branch for B to compare two continuations.</p></div>'
    rows = differences(forks, a, b)
    parts = [f'<div class="fork-comparison"><h3>{escape(a)} <span>compared with</span> {escape(b)}</h3>']
    parts.append('<p>Settings describe current controls and the latest recorded reply. Missing values are shown as “Not recorded”.</p>')
    if rows:
        parts.append(f'<div class="fork-table"><table><thead><tr><th>Settings</th><th>A · {escape(a)}</th><th>B · {escape(b)}</th></tr></thead><tbody>')
        for group, key, old, new in rows:
            parts.append(f'<tr><th>{escape(group)} · {escape(key)}</th><td>{escape(old)}</td><td>{escape(new)}</td></tr>')
        parts.append('</tbody></table></div>')
    else:
        parts.append('<p>No recorded settings differ.</p>')
    left, right = forks["branches"][a], forks["branches"][b]
    common = 0
    for old, new in zip(left, right):
        if any(old.get(key, '') != new.get(key, '') for key in ('role', 'content', 'reasoning')):
            break
        common += 1
    parts.append(f'<p>{common} shared opening message{"s" if common != 1 else ""}. <del>Removed from A</del> · <ins>Added in B</ins></p>')
    parts.append(f'<div class="fork-pair fork-pair-heading"><strong>A · {escape(a)}</strong><strong>B · {escape(b)}</strong></div>')
    if not left and not right:
        parts.append('<p>Both branches are empty.</p>')
    for index in range(max(len(left), len(right))):
        old = left[index] if index < len(left) else {}
        new = right[index] if index < len(right) else {}
        answer = text_diff(old.get('content', ''), new.get('content', ''))
        reasoning = text_diff(old.get('reasoning', ''), new.get('reasoning', ''))
        parts.append('<div class="fork-pair">')
        for side, turn in enumerate((old, new)):
            parts.append(f'<section><span class="fork-side">{"A" if side == 0 else "B"} · {escape(a if side == 0 else b)}</span><h4>Message {index + 1} · {escape(turn.get("role", "No message"))}</h4>')
            if turn.get('reasoning'):
                parts.append(f'<details><summary>Reasoning</summary><pre>{reasoning[side]}</pre></details>')
            parts.append(f'<pre>{answer[side] or "<em>No answer text</em>"}</pre></section>')
        parts.append('</div>')
    parts.append('</div>')
    return ''.join(parts)
