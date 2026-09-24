"""The page itself: every control and how the handlers are wired to them.

build_app lays out the shell and each part of the page, gathers what they
made into one namespace, and hands it to each part's wiring in turn. Each part
keeps its builder and its wiring side by side in a module of its own here.
"""

from __future__ import annotations

import logging
import html
from types import SimpleNamespace

import gradio as gr

from chatlab.ui.fork_tree import TREE_CSS

from chatlab import settings
from chatlab import themes
from chatlab.conversation import new_forks
from chatlab.device_memory import warm_device
from chatlab.ui import runtime
from chatlab.ui.layout.chat import (
    build_chat_page,
    build_conversation_pane,
    wire_conversation,
    wire_model_badge,
)
from chatlab.ui.layout.compare import wire_compare
from chatlab.ui.layout.images import build_images_page, wire_images
from chatlab.ui.layout.inspector import wire_inspector
from chatlab.ui.layout.models import build_models_page, wire_model_actions, wire_model_lists
from chatlab.ui.layout.prompts import wire_prompts
from chatlab.ui.layout.sampling import wire_sampling_and_steering
from chatlab.ui.layout.settings_page import build_settings_page, wire_settings
from chatlab.ui.layout.shell import wire_navigation, wire_page_scripts
from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.registry import load_enabled
from chatlab.ui.extensions_page import data_directory, extension_css
from chatlab.ui.common import CHAT_PAGE, NAV_PANE_WIDTH, PAGES
from chatlab.ui.inspection import JACOBIAN_CSS
from chatlab.ui.panel import empty_metrics
from chatlab.ui.token_menu import MENU_BRIDGE_CLASS, TOKEN_MENU_CSS, menu_bridge_ids
from chatlab.ui.styles import CSS, THEME


def build_app() -> gr.Blocks:
    # Read once, here, rather than per control: a build is one snapshot of
    # the file, and a control whose value came from a later read than its
    # neighbour's would be a puzzle to explain.
    saved = settings.load()
    settings.ensure_file()
    # Read the device beside the interface. Nothing here waits for it, and
    # the pages that describe a load - the fit verdicts in both model lists,
    # the hardware panel on the Settings page - are the fuller reading for it
    # by the time a reader looks.
    warm_device()
    extensions, extension_errors = load_enabled(saved.enabled_extensions)
    # The built-in pages keep fixed places in the nav and the extensions an
    # enabled build adds sit below them, above Settings. Spliced in among the
    # built-ins instead, an extension being enabled or removed would move
    # Images and Models up and down the pane under a reader who had learned
    # where they were.
    page_choices = [*PAGES[:-1], *(ext.spec.page_label for ext in extensions), PAGES[-1]]
    # Gradio otherwise caps the page at one of a handful of widths and centers
    # it, which leaves a band of empty room down each side on a wide screen.
    # The shell wants every pixel: the two side panes are a fixed width, so the
    # width the cap was holding back goes to the chat and the panel beside it.
    # analytics_enabled=False is the only thing here that is not about the
    # interface. Left at its default, Gradio posts to api.gradio.app twice on
    # the way up - once when this Blocks is built, once when it is launched -
    # with its version, the platform, and the list of component and event
    # types this app uses, and checks the package index for a newer Gradio it
    # can warn about. None of it carries what was said, and none of it is
    # wanted here: this is a local workbench for local models, often run with
    # no network at all, and its launch should not depend on reaching a host
    # on the internet. Set on the Blocks rather than through
    # GRADIO_ANALYTICS_ENABLED so it holds however the app is started - the
    # desktop bundle, run.sh, or python -m chatlab.
    with gr.Blocks(
        title="ChatLab", css=CSS + TOKEN_MENU_CSS + TREE_CSS + JACOBIAN_CSS + extension_css(extensions), theme=THEME, fill_width=True,
        analytics_enabled=False,
    ) as demo:
        # The chosen theme's colors, as a stylesheet on the page. Gradio fixes
        # THEME above when the interface is built, so a theme picked later is
        # a set of variables written over that one rather than another Blocks;
        # see the themes module. It is drawn first so nothing is painted in
        # the built-in colors and then repainted.
        theme_style = gr.HTML(
            themes.style_tag(saved.theme), elem_id="theme-style", padding=False
        )
        conversation_state = gr.State([])
        metrics_state = gr.State(empty_metrics())
        prompt_metrics_state = gr.State(empty_metrics())
        # What the Score text tab's strip is showing. The inspector's own
        # state moves on to the next reply; this one is rewritten only by
        # another scoring pass, so it still describes the passage drawn there.
        score_metrics_state = gr.State(empty_metrics())
        trace_state = gr.State({})
        # Branching from a token: the token last clicked - which turn, and
        # which of its tokens - and the alternative picked for it. Both name a
        # turn rather than a strip position, so a click keeps meaning what it
        # meant however the conversation moves under it.
        # The script names these after the strip they serve, so a second strip
        # elsewhere - the maze workbench's - carries its own three.
        request_id, response_id, action_id = menu_bridge_ids("token-strip")
        menu_request = gr.Textbox(elem_id=request_id, elem_classes=[MENU_BRIDGE_CLASS])
        menu_response = gr.HTML(elem_id=response_id, elem_classes=[MENU_BRIDGE_CLASS])
        menu_action = gr.Textbox(elem_id=action_id, elem_classes=[MENU_BRIDGE_CLASS])
        # The prompt strip carries its own three: the same menu, offering what
        # could have stood in a prompt position rather than a reply's.
        request_id, response_id, action_id = menu_bridge_ids("prompt-strip")
        prompt_menu_request = gr.Textbox(elem_id=request_id, elem_classes=[MENU_BRIDGE_CLASS])
        prompt_menu_response = gr.HTML(elem_id=response_id, elem_classes=[MENU_BRIDGE_CLASS])
        prompt_menu_action = gr.Textbox(elem_id=action_id, elem_classes=[MENU_BRIDGE_CLASS])
        selected_token = gr.State(None)
        branch_pick = gr.State(None)
        # Forking: the other transcripts, and the chatbot message last clicked.
        forks_state = gr.State(new_forks())
        tree_selection = gr.State({})
        selected_message = gr.State(None)
        token_edit_target = gr.State(None)
        # Layer inspection: the prompt ids behind the strips, the strip
        # position last clicked, and the last readout for re-rendering.
        context_ids_state = gr.State((*empty_metrics(), None))
        score_context_ids_state = gr.State((0, [], None))
        # The latest reply stays inspectable when scoring replaces the shared
        # prompt panel. Its measurements and exact input belong to the chat.
        chat_metrics_state = gr.State((0, []))
        chat_context_ids_state = gr.State((0, [], None))
        steering_state = gr.State(None)
        # What the last extraction read: one direction per decoder block, the
        # numbers beside each, and the load they were read through. Held
        # whole so that moving the layer control is instant - the pass that
        # reads one layer reads them all, and re-reading to change a layer
        # would cost another pass over every example.
        extract_state = gr.State(None)
        # The two comparison slots, and the document a download would write.
        # A slot holds one whole run - its tokens, its measurements and the
        # configuration it ran under - because the model that produced it may
        # be gone by the time the other slot is filled, which is the point.
        compare_a_state = gr.State(None)
        compare_b_state = gr.State(None)
        compare_export_state = gr.State(None)
        inspect_target = gr.State(None)
        insight_state = gr.State(None)
        # The prompts the last file gave, as it gave them. A prompt with a
        # blank line inside it reads as two once it is in the box, so a run
        # prefers this list while the box still holds what loading it wrote;
        # see ui.prompts.resolve_prompts().
        loaded_prompts_state = gr.State([])
        # Where the running batch writes its exports, so Stop can publish
        # what is there; see ui.prompts.stop_batch().
        batch_directory_state = gr.State(None)
        # Which load the scored token count on screen was counted against, so
        # a model swapped out from another tab can be told from this one.
        score_budget_load = gr.State(None)

        with gr.Row(elem_id="shell"):
            # The thin pane at the far left picks the page: Chat, Images,
            # Models, any extension pages, then Settings. The stylesheet stacks
            # the choices, rules off the extensions from the pages that ship
            # with the app, and pins Settings to the bottom.
            with gr.Column(scale=0, min_width=NAV_PANE_WIDTH, elem_id="nav-pane"):
                nav = gr.Radio(
                    choices=page_choices,
                    value=CHAT_PAGE,
                    show_label=False,
                    container=False,
                    elem_id="nav",
                )

            # The conversations pane sits beside the nav and shows with Chat only.
            conversations_ui = build_conversation_pane()

            # The three pages share the rest of the width; one is visible at a
            # time, chosen by the nav.
            chat_ui = build_chat_page(compare_a_state, compare_b_state, compare_export_state, extract_state, saved, trace_state)

            extension_pages = []
            extension_model_buttons = []
            navigation = NavigationService(
                lambda button, model_id: extension_model_buttons.append((button, model_id)))
            for extension in extensions:
                with gr.Column(scale=1, visible=False, elem_classes=["extension-page"]) as extension_page:
                    context = ExtensionContext(
                        models=ModelService(lambda: runtime.MANAGER), tokens=TokenInspector(),
                        data_dir=data_directory(extension.spec.id), navigation=navigation,
                    )
                    try:
                        extension.build_page(context)
                    except Exception as exc:
                        logging.getLogger(__name__).exception("Extension page failed: %s", extension.spec.id)
                        message = f"{extension.spec.title}: {exc}"
                        extension_errors.append(message)
                        gr.Markdown("This extension could not open. " + html.escape(message))
                extension_pages.append((extension.spec.page_label, extension_page))

            images_ui = build_images_page(saved)

            models_ui = build_models_page(saved)

            settings_ui = build_settings_page(extension_errors, extensions, saved)


        # Everything the wiring below reads, by the name it had while the
        # page was being built.
        ui = SimpleNamespace(batch_directory_state=batch_directory_state, branch_pick=branch_pick,
            chat_context_ids_state=chat_context_ids_state, chat_metrics_state=chat_metrics_state,
            compare_a_state=compare_a_state, compare_b_state=compare_b_state,
            compare_export_state=compare_export_state, context_ids_state=context_ids_state,
            conversation_state=conversation_state, demo=demo,
            extension_model_buttons=extension_model_buttons, extension_pages=extension_pages,
            extract_state=extract_state, forks_state=forks_state, insight_state=insight_state,
            inspect_target=inspect_target, loaded_prompts_state=loaded_prompts_state,
            menu_action=menu_action, menu_request=menu_request, menu_response=menu_response,
            metrics_state=metrics_state, nav=nav, prompt_menu_action=prompt_menu_action,
            prompt_menu_request=prompt_menu_request, prompt_menu_response=prompt_menu_response,
            prompt_metrics_state=prompt_metrics_state, saved=saved,
            score_budget_load=score_budget_load, score_context_ids_state=score_context_ids_state,
            score_metrics_state=score_metrics_state, selected_message=selected_message,
            selected_token=selected_token, steering_state=steering_state, theme_style=theme_style,
            token_edit_target=token_edit_target, trace_state=trace_state,
            tree_selection=tree_selection, **vars(conversations_ui), **vars(chat_ui),
            **vars(images_ui), **vars(models_ui), **vars(settings_ui))
        wire_navigation(ui)
        wire_model_badge(ui)
        wire_images(ui)
        wire_model_actions(ui)
        wire_page_scripts(ui)
        wire_model_lists(ui)
        wire_sampling_and_steering(ui)
        wire_settings(ui)
        wire_conversation(ui)
        wire_prompts(ui)
        wire_compare(ui)
        wire_inspector(ui)
    return demo
