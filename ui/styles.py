"""The stylesheet and the keyboard shortcut script."""

from __future__ import annotations

import gradio as gr

from ui.common import (
    CONVERSATION_PANE_WIDTH,
    NAV_PANE_WIDTH,
)
from ui.inspection import (
    NAV_TILE_CSS,
)


CSS = f"""
/* Keep the navigation close to the window edge; Gradio's default page
   padding otherwise adds a wide empty gutter beside the narrow pane. */
.gradio-container .app {{ padding-left: 8px !important; }}
#hero, #models-hero, #settings-hero {{ padding: 0.5rem 0 0.2rem; }}
#hero h1, #models-hero h1, #settings-hero h1 {{ font-size: 2.1rem; margin-bottom: 0.25rem; }}
#model-status {{ min-height: 128px; }}

/* The chat page's model badge and the button that appears beside it when
   there is nothing loaded. Both keep their own width and sit on one line. */
#model-bar {{ flex-wrap: nowrap; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }}
#model-bar #model-badge {{ flex: 0 1 auto; width: auto; min-width: 0; }}
#model-bar #load-model, #model-bar #default-model {{
  flex: 0 0 auto; width: auto; min-width: 0;
}}
.model-badge {{
  display: inline-flex; align-items: center; gap: 0.45rem;
  padding: 0.3rem 0.75rem; border-radius: 999px;
  border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);
  font-size: 0.9rem; font-weight: 500; line-height: 1.3;
}}
.model-badge-dot {{ width: 0.55rem; height: 0.55rem; border-radius: 50%; flex: none; }}
.model-badge[data-state="ready"] .model-badge-dot {{ background: #16a34a; }}
/* Amber in both themes, the same warning colour an incomplete model gets. */
.model-badge[data-state="loading"] .model-badge-dot {{ background: #d97706; }}
.model-badge[data-state="empty"] {{
  border-color: #d97706; background: rgba(217, 119, 6, 0.09);
}}
.model-badge[data-state="empty"] .model-badge-dot {{ background: #d97706; }}

/* The shell is one row: nav, conversations, page. It never wraps, and the two
   panes keep their widths and stay put while the page scrolls. */
#shell {{ flex-wrap: nowrap; align-items: stretch; }}
#nav-pane, #conversation-pane {{
  position: sticky; top: 0; align-self: flex-start; max-height: 100vh;
}}
#nav-pane {{
  flex: 0 0 {NAV_PANE_WIDTH}px !important; min-width: {NAV_PANE_WIDTH}px !important;
  height: 100vh; padding: 0.6rem 0.5rem 0.6rem 0;
  border-right: 1px solid var(--border-color-primary);
}}
#conversation-pane {{
  flex: 0 0 {CONVERSATION_PANE_WIDTH}px !important;
  min-width: {CONVERSATION_PANE_WIDTH}px !important;
  overflow-y: auto; padding-right: 0.5rem;
  border-right: 1px solid var(--border-color-primary);
}}
/* The nav is a Radio drawn as a column of tiles. Its inputs are hidden, the
   selected tile is filled, and the last tile (Settings) is pushed to the
   bottom. */
#nav-pane > *, #nav, #nav .wrap {{ height: 100%; }}
#nav {{ overflow: visible !important; }}
#nav .wrap {{ flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: 0.3rem; }}
/* Each tile stacks the icon over the page's name, so the name is on screen
   rather than a hover away. Three pages is not a number worth hiding. */
#nav label {{
  position: relative;
  display: flex; flex-direction: column; align-items: center; gap: 0.15rem;
  justify-content: center; text-align: center; padding: 0.5rem 0.15rem;
  line-height: 1.2; border-radius: 8px; box-shadow: none;
  background: transparent;
  /* The selected tile is outlined; the others hold the same border in
     transparent so picking a page does not nudge the icons. */
  border: 1px solid transparent;
}}
#nav label::before {{ font-size: 1.3rem; line-height: 1.1; }}
/* The label's own text, which is also what a screen reader reads for it. */
#nav label span {{ font-size: 0.7rem; font-weight: 500; }}
{NAV_TILE_CSS}
#nav label:hover {{ background: var(--background-fill-secondary); }}
#nav label.selected {{
  background: var(--block-background-fill); color: var(--body-text-color); font-weight: 600;
  border: 1px solid var(--border-color-primary);
}}
#nav label:last-child {{ margin-top: auto; }}
/* The radio inputs stay in the tab order, just out of sight, and the tile
   they belong to shows the keyboard focus ring. */
#nav label input {{
  position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; pointer-events: none;
}}
#nav label:has(input:focus-visible) {{
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}}
.model-list .wrap {{ flex-direction: column; align-items: stretch; gap: 0.2rem; }}
.model-list label {{ font-size: 0.82rem; line-height: 1.3; word-break: break-word; }}
/* Gradio stamps each option's text on its label as data-testid, which is the
   only hook a Radio gives CSS. An incomplete model's label ends in
   "· incomplete", so it is tinted amber in both themes. */
.model-list label[data-testid*="· incomplete"] {{ border-color: #d97706; }}
.model-list label[data-testid*="· incomplete"]:not(.selected) {{
  background: rgba(217, 119, 6, 0.09);
}}
.model-list label[data-testid*="· incomplete"] span {{ color: #b45309; }}
.dark .model-list label[data-testid*="· incomplete"] span {{ color: #fbbf24; }}
/* The fit verdicts are read off the same label. A model that cannot fit is
   greyed rather than reddened: it is not an error, and the reader may be
   looking at it to find that out. */
.model-list label[data-testid*="· won't fit"]:not(.selected) span {{
  color: var(--body-text-color-subdued);
}}
.model-list label[data-testid*="· tight"] span {{ color: #b45309; }}
.dark .model-list label[data-testid*="· tight"] span {{ color: #fbbf24; }}
.model-sort label span {{ font-size: 0.8rem; }}
.remove-confirm {{
  border: 1px solid #d97706; border-radius: 8px; padding: 0.4rem 0.6rem;
  background: rgba(217, 119, 6, 0.09);
}}
/* Clear deletes every conversation, so it asks first, in the same amber
   panel the model removal uses. */
.clear-confirm {{
  border: 1px solid #d97706; border-radius: 8px; padding: 0.4rem 0.6rem;
  background: rgba(217, 119, 6, 0.09);
}}
.model-detail {{ font-size: 0.85rem; }}
.model-detail p, .model-detail ul, .model-detail li {{ margin: 0.15rem 0; }}
.model-detail code {{ word-break: break-all; }}
#token-strip {{ min-height: 150px; }}
#token-strip span, #prompt-strip span {{ cursor: pointer; border-radius: 5px; }}
/* Token fills are light in both themes, so their ink is pinned dark. */
#token-strip .textspan.hl, #prompt-strip .textspan.hl,
#token-strip .category-label, #prompt-strip .category-label {{ color: #0b0b0b; }}
.footer-note {{ color: var(--body-text-color-subdued); font-size: 0.9rem; }}
.scale-caption {{ color: var(--body-text-color-subdued); font-size: 0.85rem; }}

/* An explanation folded to one line. The panels have more to say than they
   have room for, and a paragraph held open competes with the control it
   describes, so the summary is a line and the paragraph is a click. */
.hint {{ color: var(--body-text-color-subdued); font-size: 0.85rem; }}
.hint summary {{ cursor: pointer; width: fit-content; }}
.hint summary:hover {{ color: var(--body-text-color); }}
.hint[open] summary {{ margin-bottom: 0.25rem; }}
/* A measurement's name in the detail panel carries its meaning as a title.
   The dotted underline is what says there is something to hover. */
abbr[title] {{ text-decoration: underline dotted; cursor: help; }}
/* The live count under the Score text box, and the message that stands in
   for it when the model cannot be asked. */
.token-budget {{ color: var(--body-text-color-subdued); font-size: 0.85rem; }}

/* A failure says so twice: in the status line, where it stays, and in a toast
   over the page, where it cannot be missed. The line is boxed in red with a
   thick left edge so it reads as a failure even at a glance across the
   column of ordinary status sentences. */
.failure {{
  border: 1px solid var(--color-red-500); border-left-width: 5px;
  border-radius: 8px; padding: 0.6rem 0.75rem;
  background: var(--error-background-fill);
  color: var(--color-red-600); font-weight: 600;
}}
/* The mark is decoration; the line already says what failed, so the empty
   alternative text keeps a screen reader from reading the glyph out. */
.failure::before {{ content: "⚠ " / ""; }}
.failure-text {{ color: var(--color-red-600); }}
.dark .failure, .dark .failure-text {{ color: var(--color-red-400); }}

/* Gradio's only toast that does not also end the event is the warning, and
   the app raises warnings for failures alone, so the warning toast is painted
   in the error colors. Its own rules carry a Svelte hash, which outranks a
   plain class, so these have to insist. */
.toast-body.warning {{
  border-color: var(--color-red-700) !important;
  background: var(--color-red-50) !important;
}}
.dark .toast-body.warning {{
  border-color: var(--color-red-500) !important;
  background: var(--color-grey-950) !important;
}}
.toast-title.warning, .toast-text.warning,
.toast-icon.warning, .toast-close.warning {{
  color: var(--color-red-700) !important;
}}
.dark .toast-title.warning, .dark .toast-text.warning {{
  color: var(--color-red-50) !important;
}}
.dark .toast-icon.warning, .dark .toast-close.warning {{
  color: var(--color-red-500) !important;
}}

/* The conversation list is a Radio whose labels carry a line break: the
   name and title on the first line, the model and token count on the
   second. Stack the entries and let the break through. */
#conversation-list .wrap {{ flex-direction: column; align-items: stretch; gap: 0.4rem; }}
#conversation-list label {{ align-items: flex-start; }}
#conversation-list label input {{ margin-top: 0.3rem; }}
#conversation-list label span {{
  white-space: pre-line; line-height: 1.35; overflow-wrap: anywhere;
}}

.viz-root {{
  --viz-ink: #0b0b0b;
  --viz-muted: #898781;
  --viz-grid: #e1e0d9;
  --viz-axis: #c3c2b7;
  --viz-line: #2a78d6;
  --viz-band: #cde2fb;
  margin: 0;
  font-family: var(--font, system-ui, -apple-system, "Segoe UI", sans-serif);
}}
.dark .viz-root {{
  --viz-ink: #ffffff;
  --viz-muted: #898781;
  --viz-grid: #2c2c2a;
  --viz-axis: #383835;
  --viz-line: #3987e5;
  --viz-band: #1c5cab;
}}
.viz-root svg {{ width: 100%; height: auto; display: block; }}
.viz-title {{ color: var(--viz-ink); font-size: 0.9rem; font-weight: 600; padding: 0 0 0.2rem; }}
.viz-sub {{ color: var(--viz-muted); font-weight: 400; font-size: 0.8rem; margin-left: 0.4rem; }}
.viz-grid {{ stroke: var(--viz-grid); stroke-width: 1; }}
.viz-axis {{ stroke: var(--viz-axis); stroke-width: 1; }}
.viz-band {{ fill: var(--viz-band); opacity: 0.55; stroke: none; }}
.viz-line {{ fill: none; stroke: var(--viz-line); stroke-width: 2; stroke-linejoin: round; }}
.viz-peak-dot {{ fill: var(--viz-line); stroke: var(--body-background-fill); stroke-width: 2; }}
.viz-peak-label, .viz-tick {{ fill: var(--viz-muted); font-size: 10px; font-variant-numeric: tabular-nums; }}
.viz-hit {{ fill: transparent; }}
.viz-empty, .viz-note {{ color: var(--body-text-color-subdued); font-size: 0.85rem; padding: 0.4rem 0; }}
.viz-tiles {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
.viz-tile {{
  flex: 1 1 5.5rem; padding: 0.45rem 0.6rem; border-radius: 8px;
  background: var(--background-fill-secondary);
}}
.viz-value {{ color: var(--viz-ink); font-size: 1.25rem; line-height: 1.2; }}
.viz-label {{ color: var(--viz-muted); font-size: 0.72rem; text-transform: lowercase; }}

.viz-line-faint {{ stroke: var(--viz-band); stroke-width: 1.5; }}
.viz-marker {{ stroke: var(--viz-muted); stroke-width: 1; stroke-dasharray: 3 3; }}
.viz-table-wrap {{ max-height: 230px; overflow-y: auto; margin-top: 0.3rem; }}
.viz-table {{ width: 100%; font-size: 0.78rem; border-collapse: collapse; }}
.viz-table th, .viz-table td {{
  text-align: left; padding: 0.15rem 0.4rem; color: var(--viz-ink);
  border-bottom: 1px solid var(--viz-grid); font-variant-numeric: tabular-nums;
}}
.viz-table th {{
  color: var(--viz-muted); font-weight: 500; position: sticky; top: 0;
  background: var(--body-background-fill);
}}
.viz-hit-row td {{ font-weight: 600; }}
.attn-strip {{ line-height: 1.9; white-space: pre-wrap; word-break: break-word; }}
.attn-token {{ border-radius: 4px; padding: 0.05rem 0.1rem; margin: 0 1px; color: var(--body-text-color); }}
.attn-query {{ outline: 1.5px dashed var(--viz-muted); }}
.attn-predicted {{ outline: 1.5px solid var(--viz-ink); margin-left: 0.3rem; }}
.attn-top {{ font-size: 0.8rem; columns: 2; margin: 0.3rem 0 0; padding-left: 1.4rem; color: var(--viz-ink); }}
"""


# Escape stops a running generation. Stop is a button that only exists while
# a response is streaming, so calling it off means finding it first, and a
# reader who wants a runaway response to end wants it to end now. Gradio has
# no key binding of its own, so the shortcut is a listener attached once on
# load.
#
# It presses the button rather than reaching past it, which keeps one path
# through the cancellation: whatever Stop does, Escape does.
#
# Whether the button is in the document is the whole test, and it is an exact
# one. Gradio does not hide a component whose ``visible`` is false, it leaves
# it out of the page altogether, so the button is there while a response is
# streaming and gone at every other moment - including inside the message
# box, where Escape must fall through to whatever the browser makes of it.
#
# What must NOT be tested is whether the button can be seen. It is laid out
# inside the Chat tab, so switching to Score text leaves it in the page with
# a hidden ancestor and no offset parent - and that is exactly the moment a
# reader is most likely to reach for the key, being away from the button
# they would otherwise press. Switching to Models or Settings unmounts the
# chat page entirely and takes the button with it; nothing here can reach a
# button that is not in the document, and there is no way to stop a response
# from those pages by any other means either.
#
# A modifier held down means the reader is asking the browser for something
# else, so those are left alone.
SHORTCUT_JS = """
() => {
  if (window.__chatlabShortcuts) { return; }
  window.__chatlabShortcuts = true;
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') { return; }
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) { return; }
    const stop = document.querySelector('#stop-button');
    if (!stop) { return; }
    event.preventDefault();
    stop.click();
  });
}
"""


# Gradio's Textbox sends on Enter only when it is a single-line box, and on
# Shift+Enter when it has more than one line, so the "Enter sends" preference
# is expressed by choosing the box's starting height. The box grows to
# MESSAGE_BOX_MAX_LINES either way.
MESSAGE_BOX_MAX_LINES = 8


def message_box_settings(enter_sends: bool) -> dict:
    """Textbox settings that make Enter (or Shift+Enter) send the message."""
    if enter_sends:
        return {
            "lines": 1,
            "max_lines": MESSAGE_BOX_MAX_LINES,
            "placeholder": "Ask anything… Enter sends, Shift+Enter starts a new line.",
        }
    return {
        "lines": 3,
        "max_lines": MESSAGE_BOX_MAX_LINES,
        "placeholder": "Ask anything… Shift+Enter sends, Enter starts a new line.",
    }


def set_message_box_keys(enter_sends: bool):
    return gr.update(**message_box_settings(enter_sends))
