"""Settings and host layout helpers for optional extensions."""
import html
import json
import os
from pathlib import Path

import gradio as gr

import settings
from extensions.registry import CATALOGUE
from ui import icons

# What an extension naming an icon this build has never heard of gets.
DEFAULT_EXTENSION_ICON = "box"

SHELL_CSS = """
.extension-page {min-width:0 !important; min-height:0; height:100%; flex-wrap:nowrap; overflow:hidden;}
"""


def extension_css(extensions):
    """The shell rules, each extension's nav icon, and each extension's CSS.

    An extension names an icon in ui.icons rather than supplying a drawing,
    so its tile is stroked at the same weight as the pages it sits between.
    A name this build does not have falls back to the default rather than
    emitting a rule that masks the tile away to nothing.
    """

    tiles = "\n".join(
        icons.mask_rule(
            f'#nav label[data-testid={json.dumps(ext.spec.page_label + "-radio-label")}]::before',
            ext.spec.icon if ext.spec.icon in icons.ICONS else DEFAULT_EXTENSION_ICON,
        )
        for ext in extensions
    )
    return SHELL_CSS + tiles + "\n" + "\n".join(ext.css for ext in extensions)


def data_directory(extension_id):
    root = os.environ.get("CHATLAB_EXTENSIONS_DATA_PATH")
    if root:
        return Path(root).expanduser() / extension_id
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "chatlab" / "extensions" / extension_id


def save_extensions(selected, active_ids):
    known = {spec.id for spec in CATALOGUE}
    if not isinstance(selected, list) or any(item not in known for item in selected):
        raise gr.Error("Select an extension from the available list.")
    unknown = [item for item in settings.current().enabled_extensions if item not in known]
    try:
        settings.update(require_saved=True, enabled_extensions=[*unknown, *selected])
    except OSError as exc:
        raise gr.Error("Could not save extension settings. Check that the settings file is writable.") from exc
    return pending_note(selected, active_ids)


def pending_note(selected, active_ids):
    if set(selected) == set(active_ids):
        return "Extension choices are saved. No restart needed."
    return "**Saved. Restart ChatLab to apply this change.** Current pages and running experiments remain available until restart."


def restore_extensions(active_ids):
    saved = settings.load()
    selected = [spec.id for spec in CATALOGUE if spec.id in saved.enabled_extensions]
    return selected, pending_note(selected, active_ids)


def build_extension_settings(active_ids, errors):
    gr.Markdown("## Extensions")
    gr.Markdown(
        "Optional tools that share ChatLab’s loaded model and token inspection. "
        "Changes take effect after restarting ChatLab.",
        elem_classes=["scale-caption"],
    )
    for spec in CATALOGUE:
        gr.Markdown(
            f"**{html.escape(spec.title)}** — {html.escape(spec.description)}",
            elem_classes=["extension-summary"],
        )
    enabled = gr.CheckboxGroup(
        choices=[(spec.title, spec.id) for spec in CATALOGUE],
        value=[spec.id for spec in CATALOGUE if spec.id in settings.current().enabled_extensions],
        label="Enabled extensions", elem_id="enabled-extensions",
        info="Each one adds a page of its own to the sidebar.",
    )
    note = gr.Markdown("Bundled extensions are optional and disabled by default.", elem_id="extensions-status")
    if errors:
        gr.Markdown("Could not load these extensions:\n\n" + "\n\n".join(html.escape(error) for error in errors))
    active = gr.State(list(active_ids))
    enabled.input(save_extensions, [enabled, active], note, show_progress="hidden")

    return enabled, note, active
