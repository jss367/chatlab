"""What belongs to the whole page rather than one part of it: the nav and the page-wide scripts."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from chatlab.ui.common import show_page
from chatlab.ui.extensions_page import restore_extensions
from chatlab.ui.fork_tree import TREE_JS, render_fork_tree, select_tree_branch
from chatlab.ui.inspection import JACOBIAN_JS
from chatlab.ui.layout.common import CONVERSATION_PANE_QUEUE
from chatlab.ui.model_switch import go_to_models, select_model_to_load
from chatlab.ui.settings_page import refresh_hardware
from chatlab.ui.styles import (
    COLUMN_JS,
    READ_ONLY_TEXT_JS,
    RESIZE_JS,
    SHORTCUT_JS,
    WRITING_SUGGESTIONS_JS,
)
from chatlab.ui.token_menu import TOKEN_MENU_JS


def wire_navigation(ui: SimpleNamespace) -> None:
    """Show one page at a time, and let extension pages send the reader to Models."""

    ui.nav.change(
        show_page,
        ui.nav,
        [ui.conversation_pane, ui.chat_page, ui.images_page, ui.models_page, ui.settings_page],
    )
    # On the way to the page rather than on a timer: nothing here changes
    # while it is not being looked at, and reading it costs a subprocess.
    ui.nav.change(refresh_hardware, None, ui.hardware_view)
    ui.demo.load(refresh_hardware, None, ui.hardware_view)
    ui.refresh_hardware_button.click(refresh_hardware, None, ui.hardware_view)
    ui.demo.load(restore_extensions, ui.active_extensions, ui.extension_settings)
    for label, extension_page in ui.extension_pages:
        def show_extension(page, expected=label):
            return gr.update(visible=page == expected)
        ui.nav.change(show_extension, ui.nav, extension_page)
    # Every page container go_to_models() publishes an update for, in the
    # order show_page() returns them.
    extension_page_outputs = [ui.nav, ui.conversation_pane, ui.chat_page, ui.images_page,
                              ui.models_page, ui.settings_page,
                              *(page for _, page in ui.extension_pages)]
    # The ID box and everything that has to move with it, in the order
    # select_model_to_load() returns them.
    extension_model_outputs = [ui.model_id, ui.my_models, ui.my_model_detail,
                               ui.search_selection, ui.search_detail, ui.model_status,
                               ui.remove_confirm, ui.pending_removal]
    def open_models_from_extension():
        return (*go_to_models(), *(gr.update(visible=False) for _ in ui.extension_pages))
    def open_named_model_from_extension(wanted):
        # An extension names the model its own view needs - a saved run
        # names the one that recorded it - and the reader still presses
        # Load. Nothing to name, or something no model ID could be, opens
        # the page with the box untouched rather than writing nonsense
        # into it.
        try:
            filled = select_model_to_load(wanted)
        except (ValueError, AttributeError):
            return (*(gr.skip() for _ in extension_model_outputs),
                    *open_models_from_extension())
        return (*filled, *(gr.update(visible=False) for _ in ui.extension_pages))
    for button, wanted_model in ui.extension_model_buttons:
        if wanted_model is None:
            button.click(open_models_from_extension, None, extension_page_outputs)
        else:
            button.click(open_named_model_from_extension, wanted_model,
                         [*extension_model_outputs, *extension_page_outputs])


def wire_page_scripts(ui: SimpleNamespace) -> None:
    """Run the page-wide scripts, and keep the fork tree drawn from the forks."""

    # The menu handles Escape before the global generation shortcut.
    ui.demo.load(None, None, None, js=TOKEN_MENU_JS)
    ui.demo.load(None, None, None, js=TREE_JS)
    ui.demo.load(None, None, None, js=JACOBIAN_JS)
    ui.tree_action.input(
        select_tree_branch,
        [ui.tree_action, ui.conversation_state, ui.forks_state, ui.tree_selection],
        [ui.tree_selection, ui.tree_view, ui.tree_comparison],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        trigger_mode="always_last",
    )
    ui.forks_state.change(
        render_fork_tree,
        [ui.conversation_state, ui.forks_state, ui.tree_selection],
        [ui.tree_view, ui.tree_comparison],
        concurrency_id=CONVERSATION_PANE_QUEUE,
        trigger_mode="always_last",
    )
    # Escape stops a running generation, from anywhere on the page.
    ui.demo.load(None, None, None, js=SHORTCUT_JS)
    # The two readings panes are dragged wider or narrower by the handle
    # on their seam, and remember the width they were left at.
    ui.demo.load(None, None, None, js=RESIZE_JS)
    # A table column is dragged wider by the seam on its header, for the
    # text Gradio's own measurement clips.
    ui.demo.load(None, None, None, js=COLUMN_JS)
    # A box that only shows text is read-only rather than dead, so text
    # too long for it can be scrolled to and taken out.
    ui.demo.load(None, None, None, js=READ_ONLY_TEXT_JS)
    # The system's own typing predictions, on or off from the first paint
    # and whenever the setting is changed after it. The change fires when
    # a reload restores the file's value as well as when it is clicked.
    ui.demo.load(None, ui.writing_suggestions, None, js=WRITING_SUGGESTIONS_JS)
    ui.writing_suggestions.change(
        None, ui.writing_suggestions, None, js=WRITING_SUGGESTIONS_JS
    )
