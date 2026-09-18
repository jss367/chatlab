"""Settings and host layout helpers for optional extensions."""
import html
import json
import os
from pathlib import Path

import gradio as gr

import desktop
import settings
from extensions.registry import CATALOGUE
from ui import icons
from ui.icons import icon_classes

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
    """The note about the saved choice, the restart button, and the question.

    The button is the note's offer to act, so it comes and goes with the
    sentence asking for a restart, and only where a restart is something
    ChatLab can perform. Any change to the choice closes an open question:
    it was asked about the set of extensions that was saved a moment ago.
    """

    settled = set(selected) == set(active_ids)
    note = (
        "Extension choices are saved. No restart needed." if settled else
        "**Saved. Restart ChatLab to apply this change.** Current pages and running experiments remain available until restart."
    )
    return note, gr.update(visible=not settled and desktop.restart_offered()), gr.update(visible=False)


def restart_now():
    """Quit and reopen the app so the saved choice takes effect.

    A restart can be declined - an update installing right now restarts the app
    itself when it is done - and what comes back is the sentence to show.
    """

    declined = desktop.restart()
    if declined is not None:
        raise gr.Error(declined)
    return "Restarting ChatLab…", gr.update(visible=False), gr.update(visible=False)


def ask_restart():
    return gr.update(visible=False), gr.update(visible=True)


def cancel_restart():
    return gr.update(visible=True), gr.update(visible=False)


def restore_extensions(active_ids):
    saved = settings.load()
    selected = [spec.id for spec in CATALOGUE if spec.id in saved.enabled_extensions]
    return selected, *pending_note(selected, active_ids)


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
    with gr.Row():
        restart_button = gr.Button(
            "Restart ChatLab", size="sm", scale=0, min_width=160, visible=False,
            elem_id="restart-chatlab", elem_classes=icon_classes("rotate-ccw"),
        )
    # Restarting unloads the model and ends anything running, so the button
    # asks before it acts, in the amber panel removing a model asks in.
    with gr.Column(visible=False, elem_classes=["restart-confirm"]) as restart_confirm:
        gr.Markdown(
            "Restarting unloads the current model and stops running experiments.",
            elem_classes=["model-detail"],
        )
        with gr.Row():
            confirm_restart_button = gr.Button("Restart now", variant="stop", size="sm")
            cancel_restart_button = gr.Button("Cancel", size="sm")
    if errors:
        gr.Markdown("Could not load these extensions:\n\n" + "\n\n".join(html.escape(error) for error in errors))
    active = gr.State(list(active_ids))
    saved_outputs = [note, restart_button, restart_confirm]
    enabled.input(save_extensions, [enabled, active], saved_outputs, show_progress="hidden")
    restart_button.click(ask_restart, None, [restart_button, restart_confirm])
    cancel_restart_button.click(cancel_restart, None, [restart_button, restart_confirm])
    confirm_restart_button.click(restart_now, None, saved_outputs)

    return [enabled, *saved_outputs], active
