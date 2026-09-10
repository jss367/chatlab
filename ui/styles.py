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


# Native system typography and restrained controls keep the conversation primary.
THEME = gr.themes.Base(
    primary_hue="indigo",
    neutral_hue="zinc",
    font=[gr.themes.Font("-apple-system"), "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
    font_mono=[gr.themes.Font("SFMono-Regular"), "Consolas", "monospace"],
).set(
    body_background_fill="white",
    body_background_fill_dark="*neutral_950",
    block_background_fill="white",
    block_background_fill_dark="*neutral_900",
    block_border_width="0px",
    block_shadow="none",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_600",
    block_label_text_color_dark="*neutral_300",
    block_label_text_size="*text_sm",
    block_label_text_weight="500",
    block_title_text_size="*text_sm",
    block_title_text_weight="500",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_background_fill_dark="*neutral_800",
    button_secondary_border_color="*neutral_200",
    button_secondary_border_color_dark="*neutral_700",
    button_secondary_shadow="none",
    input_background_fill="white",
    input_background_fill_dark="*neutral_950",
    input_border_color="*neutral_200",
    input_border_color_dark="*neutral_700",
    input_shadow="none",
)


CSS = f"""
/* A viewport-sized shell gives every pane its own scroll boundary. */
html, body {{ height: 100%; overflow: hidden; }}
.gradio-container {{ min-height: 0 !important; }}
.gradio-container .app {{ padding: 0 !important; }}
.gradio-container footer {{ display: none !important; }}
#shell {{
  height: 100dvh; min-height: 0; gap: 0; flex-wrap: nowrap;
  align-items: stretch; overflow: hidden;
}}
#nav-pane, #conversation-pane, #chat-page, #chat-workspace, #inspector-pane,
#images-page, #images-workspace, #image-inspector,
#models-page, #settings-page {{ box-sizing: border-box; min-height: 0; flex-wrap: nowrap; }}
#nav-pane, #conversation-pane, #inspector-pane, #image-inspector {{
  background: var(--background-fill-secondary);
}}
#nav-pane {{
  flex: 0 0 {NAV_PANE_WIDTH}px !important; min-width: {NAV_PANE_WIDTH}px !important;
  height: 100%; padding: 12px 6px;
  border-right: 1px solid var(--border-color-primary);
}}
#conversation-pane {{
  flex: 0 0 {CONVERSATION_PANE_WIDTH}px !important;
  min-width: {CONVERSATION_PANE_WIDTH}px !important;
  height: 100%; overflow-y: auto; overscroll-behavior-y: contain;
  padding: 18px 12px; gap: 12px;
  border-right: 1px solid var(--border-color-primary);
}}
#chat-page {{ min-width: 0 !important; height: 100%; gap: 0; }}
#chat-columns {{
  height: 100%; min-height: 0; gap: 0; flex-wrap: nowrap;
  align-items: stretch; overflow: hidden;
}}
#chat-workspace {{
  flex: 1 1 0 !important; min-width: 0 !important; height: 100%;
  padding: 16px 24px; gap: 10px;
  overflow-y: auto; overscroll-behavior-y: contain;
}}
#inspector-pane {{
  flex: 0 0 clamp(310px, 27vw, 400px) !important; min-width: 0 !important;
  height: 100%; padding: 18px 18px 28px; gap: 18px;
  overflow-y: auto; overscroll-behavior-y: contain;
  border-left: 1px solid var(--border-color-primary);
}}
#inspector-pane > *, #conversation-pane > *, #chat-workspace > * {{ flex: 0 0 auto; }}
#models-page, #settings-page {{
  height: 100%; overflow-y: auto; overscroll-behavior-y: contain;
  padding: 24px 32px;
}}
/* Keep page headers and sections at their natural height so long content
   scrolls instead of shrinking and clipping the header. */
#models-page > *, #settings-page > * {{ flex: 0 0 auto; }}

/* The Images page is the chat layout with a picture where the transcript
   goes: the workspace scrolls on the left, the readings scroll beside it.
   Its own children are pinned the same way the rule above pins the other
   pages', so a long readout scrolls rather than squeezing the header. */
#images-page {{ min-width: 0 !important; height: 100%; gap: 0; }}
#images-columns {{
  height: 100%; min-height: 0; gap: 0; flex-wrap: nowrap;
  align-items: stretch; overflow: hidden;
}}
#images-workspace {{
  flex: 1 1 0 !important; min-width: 0 !important; height: 100%;
  padding: 16px 24px; gap: 10px;
  overflow-y: auto; overscroll-behavior-y: contain;
}}
#image-inspector {{
  flex: 0 0 clamp(310px, 30vw, 440px) !important; min-width: 0 !important;
  height: 100%; padding: 18px 18px 28px; gap: 18px;
  overflow-y: auto; overscroll-behavior-y: contain;
  border-left: 1px solid var(--border-color-primary);
}}
#images-workspace > *, #image-inspector > * {{ flex: 0 0 auto; }}
#image-inspector .block {{ background: transparent; }}
#image-inspector .inspector-section {{ padding: 0; border-radius: 0; }}
#image-inspector .form {{ border: 0; box-shadow: none; background: transparent; }}
#image-status {{ font-size: 12px; color: var(--body-text-color-subdued); }}
#image-output img {{ max-height: 60vh; object-fit: contain; }}
/* Models is a form-heavy page: give sections and controls visible edges
   without adding chrome to the conversation or inspector. */
#models-page {{
  background: var(--background-fill-secondary);
}}
#models-columns {{ align-items: flex-start; gap: 24px; }}
#model-controls {{ gap: 24px; }}
#models-page .model-card {{
  min-width: 0; padding: 20px; gap: 16px;
  border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: var(--block-background-fill);
}}
#models-page .model-card > * {{ flex-shrink: 0; }}
#models-page .model-card h2 {{
  margin: 0; padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color-primary);
  font-size: 17px; line-height: 24px; font-weight: 600;
}}
#models-page button {{ border-width: 1px; }}
#models-page textarea, #models-page input[data-testid="textbox"], #models-page input[type="password"],
#models-page .model-sort .wrap {{
  border: 1px solid var(--input-border-color); border-radius: 8px;
  background: var(--input-background-fill);
}}
#models-page textarea:focus, #models-page input[data-testid="textbox"]:focus,
#models-page input[type="password"]:focus {{
  border-color: var(--color-accent);
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}}
#search-kind {{ padding: 0; border: 0; background: transparent; }}
#search-kind .wrap {{
  display: flex; width: fit-content; max-width: 100%; gap: 4px; padding: 4px;
  border: 1px solid var(--border-color-primary); border-radius: 10px;
  background: var(--background-fill-secondary);
}}
#search-kind label {{
  flex: 1 1 auto; justify-content: center; margin: 0; padding: 8px 14px;
  border: 1px solid transparent; border-radius: 7px; background: transparent;
}}
#search-kind label.selected {{
  border-color: var(--border-color-primary);
  background: var(--block-background-fill);
  box-shadow: 0 1px 3px rgb(0 0 0 / 8%);
}}
#search-kind label.selected span {{ color: var(--color-accent); font-weight: 600; }}
#search-kind input {{
  position: absolute; width: 1px; height: 1px; opacity: 0;
}}
#search-kind label:has(input:focus-visible) {{
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}}
#model-search-row {{ align-items: flex-end; gap: 12px; }}
#models-page .form, #model-search-query {{
  border: 0; box-shadow: none; background: transparent;
}}
#model-search-query {{ padding: 0; }}
#model-search-query input {{ min-height: 42px; }}
#model-search-button {{ min-height: 42px; }}
/* An empty Radio otherwise leaves a small, unexplained box below search. */
#models-page .form:has(> #model-search-results):not(:has(input[type="radio"])) {{ display: none; }}
#models-page .model-list {{ padding: 0; border: 0; }}
#models-page .model-list label {{ padding: 12px; border-width: 1px; border-radius: 8px; }}
#models-page .block.model-detail {{
  padding: 12px 14px; border: 1px solid var(--border-color-primary);
  border-width: 1px !important;
  border-radius: 8px; background: var(--background-fill-secondary);
  color: var(--body-text-color-subdued); line-height: 1.6;
}}
#models-page .block.model-detail:not(:has(.md > *)) {{ display: none; }}
@media (max-width: 800px) {{
  #models-page {{ padding: 20px 16px; }}
  #models-page .model-card {{ padding: 16px; }}
  #model-controls {{ min-width: min(360px, 100%) !important; }}
}}
#conversations-heading h2, #inspector-heading h2 {{
  font-size: 15px; line-height: 24px; font-weight: 600; margin: 0;
}}
#hero {{ padding: 0; }}
#hero h1 {{ font-size: 18px; line-height: 26px; font-weight: 600; margin: 0; }}
#models-hero, #settings-hero {{ padding: 0 0 12px; }}
#models-hero h1, #settings-hero h1 {{ font-size: 24px; margin-bottom: 6px; }}
#images-hero {{ padding: 0; }}
#images-hero h1 {{ font-size: 18px; line-height: 26px; font-weight: 600; margin: 0; }}
#images-hero p {{ font-size: 12px; color: var(--body-text-color-subdued); margin: 2px 0 0; }}
#model-status {{ min-height: 128px; }}

/* The model badge stays small and wraps with its actions in narrow columns. */
#model-bar, #image-model-bar {{ flex-wrap: wrap; align-items: center; gap: 6px; margin: 0; }}
#model-bar #model-badge, #image-model-bar #image-model-badge {{
  flex: 0 1 auto; width: auto; min-width: 0; padding: 0;
}}
#model-bar #load-model, #model-bar #default-model,
#image-model-bar #image-load-model {{
  flex: 0 0 auto; width: auto; min-width: 0; font-size: 12px;
}}
.model-badge {{
  display: inline-flex; align-items: center; gap: 6px; max-width: 100%;
  padding: 4px 8px; border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);
  font-size: 12px; font-weight: 400; line-height: 1.4;
  overflow-wrap: anywhere;
}}
.model-badge-dot {{ width: 6px; height: 6px; border-radius: 50%; flex: none; }}
.model-badge[data-state="ready"] .model-badge-dot {{ background: #16a34a; }}
.model-badge[data-state="loading"] .model-badge-dot,
.model-badge[data-state="empty"] .model-badge-dot {{ background: #d97706; }}
/* A model of the other kind: something is loaded, so it is not the empty
   state, but not something this page can use, so it is not the ready one. */
.model-badge[data-state="other"] .model-badge-dot {{ background: #64748b; }}

/* The transcript uses the remaining height, keeping its composer in reach. */
#conversation-tabs {{ flex: 1 0 0; min-height: 420px; display: flex; flex-direction: column; }}
#conversation-tabs > .tab-nav {{ flex: none; }}
#chat-tab {{ flex: 1; min-height: 0; padding: 12px 0 0; border: 0; }}
#conversation-tabs .tabitem > .column {{ flex-wrap: nowrap; }}
#chat-tab > .column {{ height: 100%; min-height: 0; gap: 10px; }}
#chat-tab > .column > * {{ flex: 0 0 auto; }}
#chat-tab > .column > .form {{ background: transparent; }}
#conversation {{ flex: 1 1 0 !important; height: auto !important; min-height: 180px; border: 0; background: var(--body-background-fill); }}
#message-input {{ border: 1px solid var(--border-color-primary); border-radius: 12px !important; padding: 4px; border-width: 1px !important; }}
#message-input textarea {{ border: 0; box-shadow: none; background: transparent; }}
#generation-status {{ font-size: 12px; color: var(--body-text-color-subdued); }}
#conversation-tools {{ max-height: 40vh; overflow-y: auto; overscroll-behavior-y: contain; }}
#inspector-pane .block {{ background: transparent; }}
#inspector-pane input, #inspector-pane textarea {{ background: var(--input-background-fill); }}
#token-alternatives table {{ font-family: var(--font); font-size: 12px; }}
#token-alternatives td:nth-child(2) {{ font-family: var(--font-mono); }}
#inspector-pane > .form {{ flex: 0 0 auto !important; }}
#inspector-pane .inspector-section {{ padding: 0; border-radius: 0; }}
#inspector-pane .form {{ border: 0; box-shadow: none; background: transparent; }}
#inspector-pane .label-wrap, #conversation-tools > .label-wrap {{
  padding: 12px 0; border-top: 1px solid var(--border-color-primary);
}}
#shell button {{ box-shadow: none; }}
#conversation-pane button {{ font-size: 12px; padding: 6px; white-space: nowrap; }}
#chat-tab button {{ font-size: 13px; }}
#shell button:focus-visible {{ outline: 2px solid var(--color-accent); outline-offset: 2px; }}

/* Compact windows retain separate scroll areas in two stacked rows. */
@media (max-width: 1050px) {{
  #conversation-pane {{ flex-basis: 200px !important; min-width: 200px !important; }}
  #chat-workspace {{ padding: 16px; }}
  #inspector-pane {{ flex-basis: 300px !important; padding: 18px 14px; }}
}}
@media (max-width: 850px) {{
  #conversation-pane {{ flex-basis: 160px !important; min-width: 160px !important; }}
  #chat-columns, #images-columns {{ flex-direction: column; }}
  #chat-workspace, #images-workspace {{ flex: 1 1 60% !important; height: 60%; }}
  /* The Images page stacks with the Chat page and for the same reason: its
     two panes want about 620px between them, so in a narrow window the
     readings would sit off the side of a row that does not wrap and does
     not scroll sideways. */
  #inspector-pane, #image-inspector {{
    flex: 1 1 40% !important; height: 40%; border-left: 0;
    border-top: 1px solid var(--border-color-primary);
  }}
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
#token-strip {{ min-height: 110px; }}
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
#conversation-list {{ background: transparent; padding: 0; }}
#conversation-list label {{
  align-items: flex-start; background: transparent; border: 1px solid transparent;
  border-radius: 8px; padding: 10px; box-shadow: none;
}}
#conversation-list label.selected {{
  background: var(--button-primary-background-fill);
  border-color: transparent;
}}
#conversation-list label.selected span {{ color: white; }}
#conversation-list label:has(input:focus-visible) {{
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}}
#conversation-list label input {{ position: absolute; opacity: 0; width: 1px; }}
#conversation-list label input {{ margin-top: 0.3rem; }}
#conversation-list label span {{
  white-space: pre-line; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere;
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

/* A chart with two series names them under its title rather than in a box
   over the plot, where a legend would sit on the lines it explains. */
.viz-key {{ display: inline-flex; align-items: center; gap: 0.25rem; margin-right: 0.6rem; }}
.viz-swatch {{ width: 0.7rem; height: 2px; border-radius: 1px; display: inline-block; }}
.viz-swatch-line {{ background: var(--viz-line); }}

/* One frame of the denoising trajectory. Held at the pane's width and left
   to the browser's own smooth upscaling: the frame is a small preview of a
   latent, so pixelating it would claim a detail it does not have. */
.trajectory-frame {{
  width: 100%; height: auto; display: block; border-radius: 8px;
  background: var(--background-fill-primary);
}}

/* The picture with a cross-attention map over it: two images stacked, so
   the pixels underneath stay the pipeline's own and the map stays a layer
   that can be seen through. */
.attention-stack {{ position: relative; line-height: 0; border-radius: 8px; overflow: hidden; }}
.attention-stack img {{ width: 100%; height: auto; display: block; }}
.attention-stack img + img {{ position: absolute; inset: 0; }}

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
    // Whichever stop button is in the page: the chat's while a reply is
    // streaming, the Prompts tab's while a batch runs, the Images page's
    // while a picture is being drawn. Never more than one, because they
    // contend for the same generation slot and the losers refuse.
    const stop = document.querySelector(
      '#stop-button, #stop-batch-button, #stop-drawing'
    );
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
