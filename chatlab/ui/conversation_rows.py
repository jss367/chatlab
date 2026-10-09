"""The buttons on each row of the conversation list.

The list is a Gradio Radio, which draws a label per conversation and nothing
else. The page script below adds Archive to each row, or Restore and Delete
while the list shows the archive, and writes a press into a hidden bridge
textbox for the server to act on, as the fork tree's buttons do. The rows
are redrawn whenever the list changes, so the script watches the list and
puts the buttons back on whatever row has lost them or now names another
conversation.

It also groups forks under the head of their family (see
``conversation.branch_choices``): rows the server marks as forks are
indented, and a head that names how many forks it has gets a toggle that
opens or closes the family through a bridge of its own.
"""

from __future__ import annotations

import json

from chatlab.conversation import FORK_ROW_MARK, MAIN_BRANCH
from chatlab.ui.common import ARCHIVED_VIEW_CLASS

ARCHIVE_BRIDGE_ID = "conversation-archive-action"
DELETE_BRIDGE_ID = "conversation-delete-action"
FAMILY_BRIDGE_ID = "conversation-family-action"

# Delete is the one press here that cannot be taken back, so it asks for a
# second press within this long before it goes through.
DELETE_CONFIRM_MS = 3000

_SCRIPT = r"""
() => {
  if (window.chatlabConversationRows) return;
  window.chatlabConversationRows = true;
  const MAIN = __MAIN__;
  const ARCHIVED_VIEW = __ARCHIVED_VIEW__;
  const ACTIONS = {
    archive: {icon: 'icon-archive', title: 'Archive', bridge: __ARCHIVE_BRIDGE__},
    restore: {icon: 'icon-archive-restore', title: 'Restore to the list', bridge: __ARCHIVE_BRIDGE__},
    delete: {icon: 'icon-trash', title: 'Delete for good', bridge: __DELETE_BRIDGE__},
  };
  const CONFIRM = 'Press again to delete for good';
  const FORK_MARK = __FORK_MARK__;
  const FAMILY_BRIDGE = __FAMILY_BRIDGE__;
  const wanted = (list, name) => {
    if (name === MAIN) return [];
    return list.classList.contains(ARCHIVED_VIEW) ? ['restore', 'delete'] : ['archive'];
  };
  const text = label => label.querySelector(':scope > span:not(.conversation-actions)')?.textContent || '';
  const isFork = label => text(label).startsWith(FORK_MARK);
  // A head's last line ends "· 5 forks", and its family is open when every
  // one of them is listed beneath it. A closed one shows at most the fork on
  // screen, so the count is enough to tell the two apart.
  const family = (labels, index) => {
    const lines = text(labels[index]).split('\n');
    const match = / · (\d+) forks?(?: · |$)/.exec(lines[lines.length - 1]);
    if (!match || isFork(labels[index])) return null;
    let shown = 0;
    while (index + 1 + shown < labels.length && isFork(labels[index + 1 + shown])) shown += 1;
    return {open: shown === Number(match[1])};
  };
  const syncFamily = (labels, index, name) => {
    const label = labels[index];
    label.classList.toggle('conversation-fork', isFork(label));
    const found = family(labels, index);
    label.classList.toggle('conversation-head', Boolean(found));
    const key = found ? JSON.stringify([name, found.open]) : '';
    let toggle = label.querySelector(':scope > .conversation-family');
    if ((toggle?.dataset.key || '') === key) return;
    toggle?.remove();
    if (!found) return;
    toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = `conversation-family icon-chevron-right${found.open ? ' open' : ''}`;
    toggle.dataset.key = key;
    toggle.dataset.name = name;
    toggle.dataset.open = found.open ? '1' : '';
    toggle.title = found.open ? 'Hide forks' : 'Show forks';
    toggle.setAttribute('aria-label', `${toggle.title}: ${name}`);
    toggle.setAttribute('aria-expanded', String(found.open));
    label.append(toggle);
  };
  const sync = () => {
    const list = document.getElementById('conversation-list');
    if (!list) return;
    const labels = [...list.querySelectorAll('label')];
    labels.forEach((label, index) => {
      const input = label.querySelector('input[type=radio]');
      if (input) syncFamily(labels, index, input.value);
    });
    for (const label of labels) {
      const input = label.querySelector('input[type=radio]');
      if (!input) continue;
      const kinds = wanted(list, input.value);
      const key = JSON.stringify([input.value, kinds]);
      let row = label.querySelector(':scope > .conversation-actions');
      if (row && row.dataset.key === key) continue;
      row?.remove();
      if (!kinds.length) continue;
      row = document.createElement('span');
      row.className = 'conversation-actions';
      row.dataset.key = key;
      for (const kind of kinds) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `conversation-action ${ACTIONS[kind].icon}`;
        button.dataset.action = kind;
        button.dataset.name = input.value;
        button.title = ACTIONS[kind].title;
        button.setAttribute('aria-label', `${ACTIONS[kind].title}: ${input.value}`);
        row.append(button);
      }
      label.append(row);
    }
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; sync(); });
  }).observe(document.body, {
    childList: true, subtree: true, characterData: true,
    attributes: true, attributeFilter: ['class', 'value'],
  });
  sync();
  const send = (id, value) => {
    const input = document.querySelector(`#${id} textarea, #${id} input`);
    if (!input) return;
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
  };
  // Captured ahead of the label, so a press on a row's button never also
  // opens the conversation the row stands for.
  document.addEventListener('click', event => {
    const toggle = event.target.closest?.('#conversation-list .conversation-family');
    if (toggle) {
      event.preventDefault();
      event.stopPropagation();
      send(FAMILY_BRIDGE, JSON.stringify({
        name: toggle.dataset.name, open: !toggle.dataset.open, nonce: Date.now(),
      }));
      return;
    }
    const button = event.target.closest?.('#conversation-list .conversation-action');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const kind = button.dataset.action;
    if (kind === 'delete' && !button.classList.contains('armed')) {
      button.classList.add('armed');
      button.title = CONFIRM;
      button.setAttribute('aria-label', `${CONFIRM}: ${button.dataset.name}`);
      setTimeout(() => {
        button.classList.remove('armed');
        button.title = ACTIONS.delete.title;
        button.setAttribute('aria-label', `${ACTIONS.delete.title}: ${button.dataset.name}`);
      }, __CONFIRM_MS__);
      return;
    }
    send(ACTIONS[kind].bridge, JSON.stringify({
      name: button.dataset.name, archived: kind === 'archive', nonce: Date.now(),
    }));
  }, true);
}
"""

CONVERSATION_ROWS_JS = (
    _SCRIPT.replace("__MAIN__", json.dumps(MAIN_BRANCH))
    .replace("__ARCHIVED_VIEW__", json.dumps(ARCHIVED_VIEW_CLASS))
    .replace("__ARCHIVE_BRIDGE__", json.dumps(ARCHIVE_BRIDGE_ID))
    .replace("__DELETE_BRIDGE__", json.dumps(DELETE_BRIDGE_ID))
    .replace("__CONFIRM_MS__", str(DELETE_CONFIRM_MS))
    .replace("__FORK_MARK__", json.dumps(FORK_ROW_MARK))
    .replace("__FAMILY_BRIDGE__", json.dumps(FAMILY_BRIDGE_ID))
)
