"""The buttons on each row of the My Models list.

The list is a Gradio Radio, which draws a label per model and nothing else.
The page script below adds Redownload and Remove to each row and writes a
press into a hidden bridge textbox for the server to act on, as the
conversation list's buttons do (see ui.conversation_rows). The buttons sit on
the row they act on, so a long cache no longer has to be scrolled past to
reach them, and a press names its own model rather than the selection.
"""

from __future__ import annotations

import json

MODEL_ACTION_BRIDGE_ID = "my-model-action"
MODEL_LIST_ID = "my-models-list"

# Removing a model deletes its files, so Remove asks for a second press
# within this long before it goes through, as Delete does on a conversation.
REMOVE_CONFIRM_MS = 8000

_SCRIPT = r"""
() => {
  if (window.chatlabModelRows) return;
  window.chatlabModelRows = true;
  const LIST = __LIST__;
  const ACTIONS = {
    redownload: {icon: 'icon-download', title: 'Redownload missing or updated files'},
    remove: {icon: 'icon-trash', title: 'Remove from disk'},
  };
  const CONFIRM = 'Press again to remove from disk';
  const sync = () => {
    const list = document.getElementById(LIST);
    if (!list) return;
    for (const label of list.querySelectorAll('label')) {
      const input = label.querySelector('input[type=radio]');
      if (!input) continue;
      let row = label.querySelector(':scope > .model-row-actions');
      if (row && row.dataset.name === input.value) continue;
      row?.remove();
      row = document.createElement('span');
      row.className = 'model-row-actions';
      row.dataset.name = input.value;
      for (const kind of Object.keys(ACTIONS)) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `model-row-action ${ACTIONS[kind].icon}`;
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
    attributes: true, attributeFilter: ['value'],
  });
  sync();
  const send = value => {
    const input = document.querySelector(`#${__BRIDGE__} textarea, #${__BRIDGE__} input`);
    if (!input) return;
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
  };
  const disarm = button => {
    clearTimeout(button.confirmTimer);
    button.classList.remove('armed');
    button.textContent = '';
    button.title = ACTIONS.remove.title;
    button.setAttribute('aria-label', `${ACTIONS.remove.title}: ${button.dataset.name}`);
  };
  // Captured ahead of the label, so a press on a row's button never also
  // selects the model the row stands for.
  document.addEventListener('click', event => {
    const button = event.target.closest?.(`#${LIST} .model-row-action`);
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const kind = button.dataset.action;
    if (kind === 'remove' && !button.classList.contains('armed')) {
      button.classList.add('armed');
      button.textContent = CONFIRM;
      button.title = CONFIRM;
      button.setAttribute('aria-label', `${CONFIRM}: ${button.dataset.name}`);
      button.confirmTimer = setTimeout(() => disarm(button), __CONFIRM_MS__);
      return;
    }
    if (kind === 'remove') disarm(button);
    send(JSON.stringify({name: button.dataset.name, action: kind, nonce: Date.now()}));
  }, true);
}
"""

MODEL_ROWS_JS = (
    _SCRIPT.replace("__LIST__", json.dumps(MODEL_LIST_ID))
    .replace("__BRIDGE__", json.dumps(MODEL_ACTION_BRIDGE_ID))
    .replace("__CONFIRM_MS__", str(REMOVE_CONFIRM_MS))
)
