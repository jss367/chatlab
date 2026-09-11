"""A context menu that reuses the token panel's validated branch operations."""

import html
import json

import gradio as gr

from ui.generation import branch_from, branch_with_text, idle_state
from ui.panel import (
    BRANCH_UNAVAILABLE,
    branch_target,
    choose_alternative,
    select_transcript_token,
)


def token_menu_payload(turns, metrics, request_id, event: gr.SelectData):
    result = select_transcript_token(turns, metrics, event)
    selection = result[2]
    # A skipped selection is an update dict, not a token identity.
    found = branch_target(turns, selection)
    payload = {"request": request_id, "selection": selection, "error": ""}
    if isinstance(found, str):
        payload.update(selection=None, error=found)
    else:
        _, metric = found
        payload.update(text=metric["text"], candidates=metric["top_candidates"])
    return '<span data-token-menu="' + html.escape(json.dumps(payload), quote=True) + '"></span>'


def branch_from_menu(action, prompt_text, turns, *settings):
    """Resolve the clicked candidate against the reply that opened the menu."""
    try:
        data = json.loads(action)
        selection = data["selection"]
        kind = data["kind"]
        if not isinstance(selection, dict):
            raise ValueError("Missing selection")
        if kind == "text":
            replacement = data["text"]
            if not isinstance(replacement, str):
                raise ValueError("Invalid replacement")
        elif kind == "candidate":
            index = data["index"]
            if not isinstance(index, int) or index < 0:
                raise ValueError("Invalid candidate")
        elif kind != "regenerate":
            raise ValueError("Unknown action")
    except (ValueError, KeyError, TypeError):
        yield idle_state(prompt_text, turns, BRANCH_UNAVAILABLE)
        return

    if kind == "text":
        yield from branch_with_text(selection, replacement, prompt_text, turns, *settings)
        return
    if kind == "regenerate":
        yield from branch_from(selection, prompt_text, turns, *settings, resample=True)
        return
    _, pick = choose_alternative(
        turns, (0, []), (0, []), selection,
        gr.SelectData(None, {"index": index, "value": None}),
    )
    if pick is None:
        yield idle_state(prompt_text, turns, BRANCH_UNAVAILABLE)
        return
    yield from branch_from(pick, prompt_text, turns, *settings)


TOKEN_MENU_CSS = """
.token-menu-bridge { display: none !important; }
#token-context-menu {
  position: fixed; z-index: 10000; width: 310px;
  max-width: calc(100vw - 16px); max-height: calc(100vh - 16px);
  overflow: auto; padding: 8px; border-radius: 12px;
  border: 1px solid var(--border-color-primary, #ccc);
  background: var(--background-fill-primary, white);
  color: var(--body-text-color, #222);
  box-shadow: 0 8px 32px #0003; font: 13px var(--font, sans-serif);
}
#token-context-menu .menu-heading { padding: 6px 8px; overflow-wrap: anywhere; }
#token-context-menu .menu-options { max-height: 240px; overflow-y: auto; }
#token-context-menu button {
  cursor: pointer; border-radius: 6px; padding: 8px;
  color: inherit; background: transparent; border: 0;
}
#token-context-menu button:hover, #token-context-menu button:focus-visible {
  background: var(--background-fill-secondary, #eee);
}
#token-context-menu .menu-option { display: flex; width: 100%; gap: 12px; text-align: left; }
#token-context-menu .menu-token { flex: 1; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--font-mono, monospace); }
#token-context-menu .menu-probability { opacity: .65; white-space: nowrap; }
#token-context-menu form { border-top: 1px solid var(--border-color-primary, #ddd); margin-top: 6px; padding: 10px 4px 2px; }
#token-context-menu label { display: block; margin-bottom: 6px; }
#token-context-menu textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 64px; padding: 8px; border-radius: 6px; border: 1px solid var(--border-color-primary, #ccc); background: var(--input-background-fill, white); color: inherit; font: inherit; }
#token-context-menu button[type=submit] { width: 100%; text-align: right; }
#token-context-menu button:disabled { opacity: .45; cursor: default; }
"""

TOKEN_MENU_JS = r"""
() => {
  if (window.__chatlabTokenMenu) return;
  window.__chatlabTokenMenu = true;
  let menu = null, pending = null, sequence = 0, anchor = null;
  const close = () => { menu?.remove(); menu = null; pending = null; };
  const bridge = (id, value) => {
    const input = document.querySelector(`#${id} textarea, #${id} input`);
    if (!input) return false;
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    return true;
  };
  const fit = () => {
    if (!menu) return;
    const rect = menu.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(pending.x, innerWidth - rect.width - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(pending.y, innerHeight - rect.height - 8))}px`;
  };
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const show = (payload) => {
    if (!pending || payload.request !== pending.id || !menu) return;
    // Each response is consumed once, including identical token selections.
    if (pending.loaded) return;
    pending.loaded = true;
    menu.replaceChildren();
    if (payload.error) {
      menu.append(element('div', payload.error, 'menu-heading'));
      fit(); return;
    }
    menu.append(element('div', `Continue from ${JSON.stringify(payload.text)}`, 'menu-heading'));
    const send = (action) => {
      bridge('token-menu-action', JSON.stringify({...action, selection: payload.selection, request: payload.request}));
      close();
    };
    const regenerate = element('button', 'Regenerate from this token', 'menu-option');
    regenerate.type = 'button';
    regenerate.addEventListener('click', () => send({kind: 'regenerate'}));
    menu.append(regenerate);
    const options = element('div', undefined, 'menu-options');
    payload.candidates.forEach((candidate, index) => {
      const button = element('button', undefined, 'menu-option');
      button.type = 'button';
      button.append(element('span', JSON.stringify(candidate.text), 'menu-token'));
      button.append(element('span', `${(candidate.probability * 100).toPrecision(3)}%`, 'menu-probability'));
      button.addEventListener('click', () => send({kind: 'candidate', index}));
      options.append(button);
    });
    if (!payload.candidates.length) options.append(element('div', 'No alternatives recorded.', 'menu-heading'));
    menu.append(options);
    const form = element('form');
    const label = element('label', 'Your own replacement');
    label.htmlFor = 'token-menu-text';
    const input = element('textarea');
    input.id = 'token-menu-text';
    input.placeholder = 'Type text, including any leading space…';
    const submit = element('button', 'Continue with this text');
    submit.type = 'submit'; submit.disabled = true;
    input.addEventListener('input', () => { submit.disabled = input.value.length === 0; });
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (input.value.length) send({kind: 'text', text: input.value});
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault(); form.requestSubmit();
      }
    });
    form.append(label, input, submit); menu.append(form); fit();
    regenerate.focus({preventScroll: true});
  };
  document.addEventListener('contextmenu', (event) => {
    const token = event.target.closest('#token-strip .textspan.hl');
    if (!token) { close(); return; }
    event.preventDefault(); close(); anchor = token;
    const id = `${Date.now()}-${++sequence}`;
    if (!bridge('token-menu-request', id)) return;
    const bounds = token.getBoundingClientRect();
    pending = {id, x: event.clientX, y: event.clientY, anchorX: bounds.x, anchorY: bounds.y};
    menu = element('div'); menu.id = 'token-context-menu';
    menu.setAttribute('role', 'dialog'); menu.setAttribute('aria-label', 'Token alternatives');
    menu.append(element('div', 'Loading alternatives…', 'menu-heading'));
    (document.querySelector('.gradio-container') || document.body).append(menu); fit();
    // Let the textbox update reach Gradio before its select event snapshots inputs.
    requestAnimationFrame(() => { if (pending?.id === id) token.click(); });
  });
  document.addEventListener('pointerdown', (event) => {
    if (menu && !menu.contains(event.target)) close();
  }, true);
  document.addEventListener('keydown', (event) => {
    if (!menu) return;
    if (event.key === 'Escape') {
      event.preventDefault(); event.stopImmediatePropagation(); close(); anchor?.focus();
    } else if (event.key === 'Tab') {
      const controls = [...menu.querySelectorAll('button:not(:disabled), textarea')];
      if (!controls.length) return;
      const at = controls.indexOf(document.activeElement);
      event.preventDefault(); controls[(at + (event.shiftKey ? -1 : 1) + controls.length) % controls.length].focus();
    } else if (['ArrowDown', 'ArrowUp'].includes(event.key) && document.activeElement?.tagName !== 'TEXTAREA') {
      const buttons = [...menu.querySelectorAll('button:not(:disabled)')];
      if (!buttons.length) return;
      event.preventDefault();
      const at = buttons.indexOf(document.activeElement);
      buttons[(at + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length].focus();
    }
  }, true);
  window.addEventListener('resize', close);
  document.addEventListener('scroll', (event) => {
    if (!menu || menu.contains(event.target)) return;
    // A scroll notification queued before the right-click can arrive after
    // opening. Only dismiss if the token actually moved from its anchor.
    const bounds = anchor?.getBoundingClientRect();
    if (!bounds || bounds.x !== pending.anchorX || bounds.y !== pending.anchorY) close();
  }, true);
  new MutationObserver(() => {
    if (!pending) return;
    if (!anchor?.isConnected || !anchor.closest('#token-strip')?.getClientRects().length) { close(); return; }
    const response = document.querySelector('#token-menu-response [data-token-menu]');
    if (response) {
      try { show(JSON.parse(response.dataset.tokenMenu)); } catch { close(); }
    }
  }).observe(document.body, {childList: true, subtree: true, attributes: true});
}
"""
