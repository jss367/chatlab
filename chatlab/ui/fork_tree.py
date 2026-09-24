"""The fork tree's native HTML buttons and server-validated selection."""

import json

from chatlab.conversation import copy_forks, copy_turns
from chatlab.fork_tree import comparison_html, tree_html


def render_fork_tree(turns, forks, selection):
    forks = copy_forks(forks)
    forks["branches"][forks["active"]] = copy_turns(turns)
    selected = {slot: name for slot, name in (selection or {}).items()
                if slot in {"A", "B"} and name in forks["branches"]}
    return tree_html(forks, selected), comparison_html(forks, selected)


def select_tree_branch(action, turns, forks, selection):
    selection = dict(selection or {})
    try:
        data = json.loads(action)
        slot, name = data["slot"], data["name"]
        if slot in {"A", "B"} and name in (forks or {}).get("branches", {}):
            selection[slot] = name
    except (ValueError, TypeError, KeyError):
        pass
    return selection, *render_fork_tree(turns, forks, selection)


TREE_JS = r"""
() => {
  if (window.chatlabForkTreeInstalled) return;
  window.chatlabForkTreeInstalled = true;
  document.addEventListener('click', event => {
    const button = event.target.closest('#fork-tree-view button[data-fork-slot]');
    if (!button) return;
    const input = document.querySelector('#fork-tree-action textarea, #fork-tree-action input');
    if (!input) return;
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, JSON.stringify({
      slot: button.dataset.forkSlot, name: button.dataset.forkName, nonce: Date.now()
    }));
    input.dispatchEvent(new Event('input', {bubbles: true}));
  });
}
"""

TREE_CSS = """
.fork-tree { overflow: auto; max-height: 58vh; padding: 8px; }
.fork-tree ul { list-style: none; margin: 0; padding-left: 26px; }
.fork-tree > ul { padding-left: 0; }
.fork-tree li { position: relative; padding: 0 0 12px 16px; border-left: 1px solid var(--border-color-primary); }
.fork-tree li::before { content: ''; position: absolute; left: 0; top: 24px; width: 16px; border-top: 1px solid var(--border-color-primary); }
.fork-tree ul:empty { display: none; }
.fork-node { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; min-width: 250px; background: var(--block-background-fill); }
.fork-node.active { border-color: var(--color-accent); }
.fork-node header, .fork-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.fork-node header span { font-size: 11px; color: var(--body-text-color-subdued); }
.fork-node p { margin: 6px 0; overflow-wrap: anywhere; }
.fork-origin { font-size: 12px; color: var(--body-text-color-subdued); white-space: pre-wrap; }
.fork-node details { font-size: 12px; margin: 8px 0; overflow-wrap: anywhere; }
.fork-node details li { border: none; padding: 4px 0; }
.fork-node details li::before { display: none; }
.fork-actions { margin-top: 10px; }
.fork-actions button { border: 1px solid var(--border-color-primary); padding: 5px 12px; border-radius: 6px; cursor: pointer; font: inherit; font-size: 12px; color: var(--body-text-color); background: var(--button-secondary-background-fill); }
.fork-actions button[aria-pressed=true] { background: var(--color-accent); color: white; }
.fork-actions button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 3px; }
.fork-comparison h3 { margin: 14px 0; font-size: 18px; }
.fork-comparison h3 span { font-weight: normal; }
.fork-comparison p { margin: 10px 0; }
.fork-table { overflow-x: auto; }
.fork-table table { width: 100%; border-collapse: collapse; font-size: 12px; }
.fork-table th, .fork-table td { padding: 8px; border: 1px solid var(--border-color-primary); text-align: left; white-space: pre-wrap; overflow-wrap: anywhere; max-width: 300px; }
.fork-pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin: 12px 0; }
.fork-pair section { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px; }
.fork-pair h4 { font-size: 12px; margin-bottom: 8px; color: var(--body-text-color-subdued); }
.fork-pair pre { white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; font-size: 13px; margin-top: 8px; }
.fork-side { display: none; }
.fork-comparison del { background: #b4231826; text-decoration: line-through; }
.fork-comparison ins { background: #16803c26; text-decoration: underline; }
@media (max-width: 1100px) {
  .fork-tree ul { padding-left: 10px; }
  .fork-tree > ul { padding-left: 0; }
  .fork-tree li { padding-left: 8px; }
  .fork-tree li::before { width: 8px; }
  .fork-node { min-width: 0; padding: 10px; }
  .fork-pair { grid-template-columns: minmax(0, 1fr); }
  .fork-pair-heading { display: none; }
  .fork-side { display: block; font-weight: bold; margin-bottom: 6px; }
}
"""
