"""The buttons on each row of the conversation list.

The list is a Gradio Radio, which draws a label per conversation and nothing
else. The page script below adds Archive to each row, or Restore and Delete
while the list shows the archive, and writes a press into a hidden bridge
textbox for the server to act on, as the fork tree's buttons do. The rows
are redrawn whenever the list changes, so the script watches the list and
puts the buttons back on whatever row has lost them or now names another
conversation.
"""

from __future__ import annotations

import json

from chatlab.conversation import MAIN_BRANCH
from chatlab.ui.common import ARCHIVED_VIEW_CLASS

ARCHIVE_BRIDGE_ID = "conversation-archive-action"
DELETE_BRIDGE_ID = "conversation-delete-action"

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
  const wanted = (list, name) => {
    if (name === MAIN) return [];
    return list.classList.contains(ARCHIVED_VIEW) ? ['restore', 'delete'] : ['archive'];
  };
  const sync = () => {
    const list = document.getElementById('conversation-list');
    if (!list) return;
    for (const label of list.querySelectorAll('label')) {
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
)
