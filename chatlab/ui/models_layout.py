"""The Models page: its controls, and how its handlers are wired to them.

The handlers live in ui.models_page. This module draws the page they act on,
inside the Blocks ui.layout.build_app() opens, and binds them to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr

from chatlab import settings
from chatlab.model_cache import DEFAULT_MODEL_SORT, MODEL_SORT_ORDERS, TEXT_KIND
from chatlab.ui.common import QUIET_TICK
from chatlab.ui.icons import icon_classes
from chatlab.ui.model_finder import (
    DEFAULT_SEARCH_ORDER,
    PANE_EMPTY,
    PANE_ID,
    PICK_BRIDGE_ID,
    RESULTS_ID,
    SEARCH_KINDS,
    SEARCH_ORDERS,
)
from chatlab.ui.model_rows import MODEL_ACTION_BRIDGE_ID, MODEL_LIST_ID
from chatlab.ui.model_repository import check_model_repository
from chatlab.ui.models_page import (
    ALL_KINDS,
    MODEL_KIND_FILTERS,
    act_on_my_model,
    clear_my_model_selection,
    download_and_load_model,
    download_model,
    draw_results,
    find_versions,
    go_to_image_models,
    load_cached_model,
    model_pane,
    refresh_after_device,
    refresh_model_actions,
    refresh_current_model,
    refresh_model_badge,
    refresh_model_switch,
    refresh_my_models,
    refresh_search_results,
    refresh_stale_model_actions,
    search_and_open,
    search_models,
    select_default_model,
    select_my_model,
    select_search_result,
    switch_model,
    unload_model,
)
from chatlab.ui.scoring import SCORE_BUDGET_QUEUE, score_token_count
from chatlab.ui.settings_page import refresh_hardware, refresh_thinking_mode
from chatlab.ui.token_menu import MENU_BRIDGE_CLASS

if TYPE_CHECKING:
    from chatlab.ui.layout import Pages


@dataclass(frozen=True)
class ModelsPage:
    """The Models page's column and the controls its listeners read and write."""

    column: gr.Column
    page_tab: gr.Radio
    find_controls: gr.Column
    downloaded_controls: gr.Column
    find_list: gr.Column
    downloaded_list: gr.Column
    current_model: gr.HTML
    unload_button: gr.Button
    model_id: gr.Textbox
    check_model_button: gr.Button
    repository_result: gr.State
    action_stamp: gr.State
    repository_detail: gr.HTML
    hf_token: gr.Textbox
    weight_precision: gr.Radio
    model_availability: gr.Markdown
    model_status: gr.Markdown
    download_load_button: gr.Button
    download_button: gr.Button
    cached_button: gr.Button
    search_kind: gr.Radio
    search_order: gr.Dropdown
    fits_only: gr.Checkbox
    search_query: gr.Textbox
    search_results: gr.HTML
    search_pick: gr.Textbox
    search_detail: gr.HTML
    search_results_state: gr.State
    related_state: gr.State
    my_models_summary: gr.Markdown
    sort_models: gr.Dropdown
    kind_filter: gr.Dropdown
    name_filter: gr.Textbox
    my_models: gr.Radio
    my_model_detail: gr.Markdown
    refresh_models_button: gr.Button
    model_action: gr.Textbox
    device_read: gr.State

    # Every handler that can change what is on disk or in memory rescans
    # the cache afterwards, so My Models never shows a stale list.
    # The chosen ID stays last: the model-actions listeners assert it is
    # the input the refresh is given, and a new argument goes before it
    # rather than displacing it.
    @property
    def list_inputs(self) -> list:
        """What refresh_my_models() reads."""

        return [
            self.my_models, self.sort_models, self.weight_precision, self.kind_filter,
            self.name_filter, self.model_id,
        ]

    @property
    def list_outputs(self) -> list:
        """What refresh_my_models() repaints."""

        return [self.my_models, self.my_model_detail, self.my_models_summary]

    @property
    def action_inputs(self) -> list:
        """What refresh_model_actions() reads."""

        return [self.model_id, self.my_models, self.repository_result, self.hf_token]

    @property
    def action_outputs(self) -> list:
        """The availability line and the three buttons it governs."""

        return [
            self.model_availability, self.download_load_button, self.download_button,
            self.cached_button,
        ]

    @property
    def pane_inputs(self) -> list:
        """What model_pane() reads."""

        return [
            self.model_id, self.my_models, self.repository_result, self.hf_token,
            self.search_results_state, self.related_state, self.weight_precision,
        ]

    @property
    def pane_outputs(self) -> list:
        """What model_pane() repaints."""

        return [self.search_detail, self.weight_precision, self.repository_detail, self.check_model_button]

    @property
    def search_inputs(self) -> list:
        """What search_models() reads."""

        return [
            self.search_query, self.hf_token, self.weight_precision, self.search_kind,
            self.search_order, self.fits_only, self.model_id,
        ]

    @property
    def search_outputs(self) -> list:
        """What search_models() repaints."""

        return [self.search_results, self.search_results_state, self.search_order]


FIND, DOWNLOADED = "find", "downloaded"
PAGE_TABS = (("Find a model", FIND), ("Downloaded", DOWNLOADED))


def build_models_page(saved: settings.Settings) -> ModelsPage:
    """The Models page, hidden until the nav picks it.

    One search box finds a model, a list shows what it found, and a pane
    beside the list says what the chosen model needs and what can be done
    with it, with the buttons that fetch and load it. The Downloaded tab
    swaps the list for the models on disk; the pane stays, so a model is
    loaded from the same place whichever list it was chosen from.
    """

    with gr.Column(
        scale=1, visible=False, elem_id="models-page"
    ) as models_page:
        with gr.Row(elem_id="models-topbar"):
            gr.Markdown("# Models", elem_id="models-hero")
            page_tab = gr.Radio(
                choices=list(PAGE_TABS), value=FIND, show_label=False, container=False,
                elem_id="models-tabs", scale=0,
            )
            with gr.Row(elem_id="current-model-row"):
                current_model = gr.HTML(refresh_current_model(), elem_id="currently-loaded-model", container=False, padding=False)
                unload_button = gr.Button("Unload", size="sm", scale=0, min_width=80)
        with gr.Column(elem_id="model-find-controls") as find_controls:
            search_query = gr.Textbox(
                label="Search Hugging Face",
                show_label=False,
                container=False,
                placeholder="Search Hugging Face, or paste a model ID and press Enter",
                max_lines=1,
                elem_id="model-search-query",
            )
            with gr.Row(elem_id="model-search-filters"):
                # Text or image; an MLX conversion is a text model and is
                # listed among them. See ui.model_finder.SEARCH_KINDS.
                search_kind = gr.Radio(
                    choices=list(SEARCH_KINDS),
                    value=TEXT_KIND,
                    show_label=False,
                    container=False,
                    elem_id="search-kind",
                    elem_classes=["model-chips"],
                    scale=0,
                )
                fits_only = gr.Checkbox(
                    label="Fits this computer", value=False, container=False,
                    elem_id="model-fits-only", scale=0,
                )
                # Only a search has a sort; picks are listed in the order chosen.
                search_order = gr.Dropdown(
                    choices=list(SEARCH_ORDERS), value=DEFAULT_SEARCH_ORDER,
                    label="Sort results", show_label=False, container=False, interactive=True,
                    visible=False, elem_id="model-search-order", scale=0, min_width=170,
                )
        with gr.Column(visible=False, elem_id="model-downloaded-controls") as downloaded_controls:
            my_models_summary = gr.Markdown("", elem_classes=["scale-caption"], elem_id="my-models-summary")
            # A cache that has grown past a screenful is read by family
            # ("every Qwen") more often than by kind, and the ID is the only
            # place a family is written.
            with gr.Row(elem_id="my-models-filter-row"):
                name_filter = gr.Textbox(
                    placeholder="Filter by name",
                    show_label=False,
                    container=False,
                    elem_id="my-models-filter",
                )
                sort_models = gr.Dropdown(
                    choices=list(MODEL_SORT_ORDERS),
                    value=DEFAULT_MODEL_SORT,
                    label="Sort by",
                    show_label=False,
                    container=False,
                    min_width=150,
                    scale=0,
                    elem_classes=["model-sort"],
                )
                kind_filter = gr.Dropdown(
                    choices=list(MODEL_KIND_FILTERS),
                    value=ALL_KINDS,
                    label="Kind",
                    show_label=False,
                    container=False,
                    min_width=130,
                    scale=0,
                    elem_classes=["model-sort", "model-kind"],
                )
                refresh_models_button = gr.Button(
                    "Refresh", size="sm", scale=0, min_width=0,
                    elem_classes=icon_classes("refresh"),
                )
        with gr.Row(elem_id="models-columns", equal_height=False):
            with gr.Column(min_width=360, elem_classes=["model-card"], elem_id="model-list-card"):
                with gr.Column(elem_id="model-find-list") as find_list:
                    search_results = gr.HTML(
                        draw_results(None, None, False, None), label="Search results",
                        show_label=False, elem_id=RESULTS_ID, container=False, padding=False,
                    )
                    # Hidden by the bridge class, not visible=False, which
                    # would take it out of the DOM where the row script has
                    # to find it; see ui.model_finder.
                    search_pick = gr.Textbox(elem_id=PICK_BRIDGE_ID, elem_classes=[MENU_BRIDGE_CLASS])
                    search_results_state = gr.State(None)
                with gr.Column(visible=False, elem_id="model-downloaded-list") as downloaded_list:
                    # Redownload and Remove are on each row; see ui.model_rows.
                    my_models = gr.Radio(
                        choices=[],
                        label="Downloaded models",
                        show_label=False,
                        elem_id=MODEL_LIST_ID,
                        elem_classes=["model-list"],
                    )
                    # Hidden by the bridge class, like the pick bridge. It
                    # comes after the list rather than straight after the
                    # radio's form: Gradio wraps neighbouring inputs in one
                    # form, and the rule that hides a bridge's form would hide
                    # the list with it.
                    gr.HTML("", container=False, padding=False)
                    model_action = gr.Textbox(elem_id=MODEL_ACTION_BRIDGE_ID, elem_classes=[MENU_BRIDGE_CLASS])
            with gr.Column(min_width=380, elem_classes=["model-card"], elem_id=PANE_ID):
                # The model the pane describes and the buttons act on. It is
                # written by picking a row, by pasting an ID into the search
                # box, and by the pages that send a reader here with a model
                # in mind; nobody types into it.
                model_id = gr.Textbox(
                    value=settings.model_id_at_startup(saved),
                    label="Hugging Face model ID",
                    visible=False,
                )
                search_detail = gr.HTML(PANE_EMPTY, container=False, padding=False, elem_id="model-pane-head")
                with gr.Accordion("Access token", open=False, elem_classes=["model-access"]):
                    hf_token = gr.Textbox(
                        label="Hugging Face token (optional)",
                        type="password",
                        placeholder="Only needed for gated or private models",
                    )
                weight_precision = gr.Radio(
                    choices=[
                        ("Full (16-bit)", "full"),
                        ("8-bit", "8-bit"),
                        ("4-bit", "4-bit"),
                    ],
                    value=saved.weight_precision,
                    label="Load at",
                    info=(
                        "Lower precision saves memory with some loss of accuracy. "
                        "Applies to Transformers loads on Apple Metal."
                    ),
                    elem_id="model-precision",
                )
                repository_detail = gr.HTML("", container=False, padding=False, elem_id="model-pane-body")
                # What is on disk of a model chosen from the Downloaded list.
                my_model_detail = gr.Markdown(
                    "", elem_id="my-model-detail", elem_classes=["model-detail"]
                )
                repository_result = gr.State(None)
                related_state = gr.State(None)
                # What the timer's refresh last painted the actions from;
                # see refresh_stale_model_actions.
                action_stamp = gr.State(None)
                model_availability = gr.Markdown(
                    "Checking downloaded files…", elem_id="model-availability"
                )
                with gr.Row(elem_id="model-actions"):
                    download_load_button = gr.Button(
                        "Download and load", variant="primary", size="sm", scale=0, min_width=150,
                    )
                    download_button = gr.Button("Download only", size="sm", scale=0, min_width=120)
                    cached_button = gr.Button("Load cached", size="sm", scale=0, min_width=110)
                    check_model_button = gr.Button(
                        "Check on Hugging Face", size="sm", scale=0, min_width=160,
                    )
                with gr.Column(elem_classes=["model-activity"]):
                    gr.Markdown("### Latest model action")
                    model_status = gr.Markdown(
                        "No downloads or loads started in this tab.",
                        elem_id="model-status",
                    )
                # Whether the fit verdicts on screen were given with
                # the device known; see refresh_after_device.
                device_read = gr.State(False)
    return ModelsPage(
        column=models_page,
        page_tab=page_tab,
        find_controls=find_controls,
        downloaded_controls=downloaded_controls,
        find_list=find_list,
        downloaded_list=downloaded_list,
        current_model=current_model,
        unload_button=unload_button,
        model_id=model_id,
        check_model_button=check_model_button,
        repository_result=repository_result,
        action_stamp=action_stamp,
        repository_detail=repository_detail,
        hf_token=hf_token,
        weight_precision=weight_precision,
        model_availability=model_availability,
        model_status=model_status,
        download_load_button=download_load_button,
        download_button=download_button,
        cached_button=cached_button,
        search_kind=search_kind,
        search_order=search_order,
        fits_only=fits_only,
        search_query=search_query,
        search_results=search_results,
        search_pick=search_pick,
        search_detail=search_detail,
        search_results_state=search_results_state,
        related_state=related_state,
        my_models_summary=my_models_summary,
        sort_models=sort_models,
        kind_filter=kind_filter,
        name_filter=name_filter,
        my_models=my_models,
        my_model_detail=my_model_detail,
        refresh_models_button=refresh_models_button,
        model_action=model_action,
        device_read=device_read,
    )


def show_tab(tab: str | None):
    """Swap the controls and the list for the tab chosen; the pane stays."""

    find = tab != DOWNLOADED
    return (
        gr.update(visible=find), gr.update(visible=find),
        gr.update(visible=not find), gr.update(visible=not find),
    )


@dataclass(frozen=True)
class ModelRefresh:
    """What follows an explicit model action, in the tab that took it.

    Refresh model-dependent displays after explicit model actions. The
    timer also catches changes from other tabs, but this updates the badge
    and token count immediately in the tab that acted. Most of what it
    repaints lives on other pages - the switcher and badge on the Chat
    page, the hardware panel and the thinking control on Settings - so
    those are handed in rather than looked up.
    """

    models: ModelsPage
    switch_outputs: list
    badge_outputs: list
    score_budget_inputs: list
    score_budget_outputs: list
    hardware_view: gr.Markdown
    thinking_mode: gr.Radio

    def actions(self, event):
        """Repaint the action buttons and the detail pane after ``event``."""

        return event.then(
            refresh_model_actions, self.models.action_inputs, self.models.action_outputs,
            show_progress="hidden", concurrency_id="model-actions",
        ).then(
            model_pane, self.models.pane_inputs, self.models.pane_outputs,
            show_progress="hidden", concurrency_id="model-pane",
        )

    def rescan(self, event, *, reloads: bool = True):
        """Rescan the cache after ``event``, and re-read what the model feeds."""

        event = event.then(refresh_my_models, self.models.list_inputs, self.models.list_outputs)
        event = self.actions(event)
        event = event.then(refresh_current_model, None, self.models.current_model, show_progress="hidden")
        # What is on disk is what the switcher offers, so it follows every
        # rescan, download-only included.
        event = event.then(
            refresh_model_switch,
            self.models.weight_precision,
            self.switch_outputs,
            show_progress="hidden",
        )
        if not reloads:
            return event
        return (
            event.then(refresh_model_badge, None, self.badge_outputs)
            .then(
                score_token_count,
                self.score_budget_inputs,
                self.score_budget_outputs,
                show_progress="hidden",
                concurrency_id=SCORE_BUDGET_QUEUE,
            )
            # A load or an unload is the largest change the machine's
            # memory sees, so the hardware panel is re-read after it
            # rather than left showing what was true before.
            .then(refresh_hardware, None, self.hardware_view)
            .then(refresh_thinking_mode, None, self.thinking_mode)
        )


def wire_model_lists(
    demo: gr.Blocks,
    pages: Pages,
    badge_timer: gr.Timer,
    models: ModelsPage,
    refresh: ModelRefresh,
    model_switch: gr.Dropdown,
    model_badge_view: gr.HTML,
) -> None:
    """The model actions, the repository check, and the My Models list.

    ``model_switch`` and ``model_badge_view`` are the Chat page's switcher
    and the badge beside it: a pick there is a load from the cache like
    any other, followed by the same rescan.
    """

    # Include programmatic selections (search, default, and rescans).
    # The selected row takes precedence, just as it does for a load.
    for control in models.action_inputs:
        control.change(
            refresh_model_actions, models.action_inputs, models.action_outputs,
            show_progress="hidden", trigger_mode="always_last",
            concurrency_id="model-actions",
        )
    # Downloads run in worker threads, so selections alone cannot keep the
    # local-file status current while files arrive (or in another tab).
    # The timer covers that, but an idle tick paints nothing rather than
    # scanning the cache every couple of seconds forever in every open
    # session; see refresh_stale_model_actions, which hands back the stamp
    # the next tick compares against.
    badge_timer.tick(
        refresh_stale_model_actions,
        [*models.action_inputs, models.action_stamp],
        [*models.action_outputs, models.action_stamp],
        show_progress="hidden", show_progress_on=[], trigger_mode="always_last",
        concurrency_id="model-actions",
    )

    # Checking a model on Hugging Face is a request, so it is made when the
    # reader opens a model or asks, never on the page's own load: the page
    # paints offline. A check scopes its answer to the ID and credentials it
    # was made with, and the pane reads both again, so an old answer cannot
    # describe a newer choice.
    for event in (models.check_model_button.click, models.hf_token.submit):
        event(
            check_model_repository, [models.model_id, models.hf_token], models.repository_result,
            show_progress="hidden", concurrency_id="model-repository-check",
            trigger_mode="always_last",
        ).then(
            find_versions,
            [models.model_id, models.hf_token, models.search_results_state, models.related_state],
            models.related_state, show_progress="hidden", concurrency_id="model-versions",
            trigger_mode="always_last",
        )
    for event in (
        *(control.change for control in models.pane_inputs), demo.load, pages.nav.change,
        models.device_read.change,
    ):
        event(
            model_pane, models.pane_inputs, models.pane_outputs, show_progress="hidden",
            concurrency_id="model-pane", trigger_mode="always_last",
        )
    models.hf_token.input(lambda: None, None, models.repository_result, show_progress="hidden")
    for event in (demo.load, pages.nav.change, badge_timer.tick):
        event(refresh_current_model, None, models.current_model, **QUIET_TICK)

    # Download-only changes the cache without changing the loaded model.
    refresh.rescan(
        models.download_button.click(
            download_model, [models.model_id, models.hf_token, models.my_models], models.model_status
        )
    )
    refresh.rescan(
        models.download_load_button.click(
            download_and_load_model,
            [models.model_id, models.hf_token, models.my_models, models.weight_precision],
            models.model_status,
        )
    )
    refresh.rescan(
        models.cached_button.click(
            load_cached_model,
            [models.model_id, models.my_models, models.weight_precision],
            models.model_status,
        )
    )
    refresh.rescan(models.unload_button.click(unload_model, outputs=models.model_status))
    # A pick in the chat page's switcher is a load from the cache, and is
    # followed by the same rescan as the button. Its status goes to the
    # Models page's card, where the switcher's own repaint would
    # otherwise drop the progress the load is reporting. That page is not
    # the one the reader is on, so the badge beside the switcher is
    # written from the same handler - it is what shows the reader the
    # load and how far it has come - and switch_model also toasts an
    # ending the card alone would have kept to itself; see
    # announce_switch_outcome.
    refresh.rescan(
        model_switch.input(
            switch_model,
            [model_switch, models.weight_precision],
            [model_switch, models.model_status, model_badge_view],
        )
    )
    models.page_tab.input(
        show_tab, models.page_tab,
        [models.find_controls, models.find_list, models.downloaded_controls, models.downloaded_list],
        show_progress="hidden",
    )
    # A manual refresh, a new sort order, a new kind filter and a typed
    # name reorder or narrow a list; none of them changes what is on disk
    # or in memory, which is all the badge and the count ask about.
    refresh.actions(
        models.refresh_models_button.click(refresh_my_models, models.list_inputs, models.list_outputs)
    )
    models.sort_models.input(refresh_my_models, models.list_inputs, models.list_outputs)
    models.kind_filter.input(refresh_my_models, models.list_inputs, models.list_outputs)
    # The list narrows as the reader types. Only the last keystroke of a
    # burst is answered, since each answer rescans the cache folder.
    models.name_filter.input(
        refresh_my_models, models.list_inputs, models.list_outputs,
        show_progress="hidden", trigger_mode="always_last",
    )
    # Before the reader chooses an ID, startup can highlight the loaded model.
    refresh.actions(demo.load(refresh_my_models, [models.my_models, models.sort_models], models.list_outputs))
    # The badge's timer corrects the fit verdicts once torch has finished
    # importing: the page is painted before that, so the first verdicts
    # are given without knowing the device. It repaints once and then
    # does nothing for the rest of the session.
    badge_timer.tick(
        refresh_after_device,
        [models.device_read, *models.list_inputs, models.search_results_state, models.fits_only],
        [*models.list_outputs, models.search_results, models.device_read],
        **QUIET_TICK,
    )


def wire_model_choice(
    demo: gr.Blocks,
    pages: Pages,
    models: ModelsPage,
    refresh: ModelRefresh,
    default_model_button: gr.Button,
    image_load_button: gr.Button,
) -> None:
    """Choosing a model: the default, a row of My Models, a search result.

    ``default_model_button`` is the Chat page's way here and
    ``image_load_button`` the Images page's; both land on this page with a
    model or a kind of model already chosen.
    """

    # Selecting a default is navigation only. The Models page owns the
    # explicit download and load actions, including their errors.
    default_model_button.click(
        select_default_model,
        None,
        [
            models.model_id,
            models.my_models,
            models.my_model_detail,
            models.model_status,
            pages.nav,
            pages.conversations,
            pages.chat,
            pages.images,
            pages.models,
            pages.extension_manager, pages.settings,
        ],
    )
    # .input rather than .change: the refresh above also sets the radio,
    # and a .change listener would rewrite the model ID on each rescan.
    models.my_models.input(
        select_my_model,
        [models.my_models, models.weight_precision],
        [models.model_id, models.my_model_detail],
    )
    # A row's Redownload or Remove names its own model, never the radio's
    # current value; see ui.model_rows.
    refresh.rescan(
        models.model_action.input(
            act_on_my_model, [models.model_action, models.hf_token], models.model_status
        )
    )

    # The page loads with an empty box, which shows ChatLab's picks without
    # going online. Typing searches as it goes; only the last keystroke of a
    # burst is answered.
    # One dependency owns the deferred payload across all these controls.
    # Separate always_last listeners can replay an old Text query after a
    # newer Image event, even when their server concurrency group is shared.
    gr.on(
        # change also observes the Images button's programmatic kind update,
        # replacing any deferred Text payload with the newest Image search.
        triggers=[demo.load, models.search_query.input, models.search_kind.change, models.search_order.input],
        fn=search_models, inputs=models.search_inputs, outputs=models.search_outputs,
        show_progress="hidden", trigger_mode="always_last", concurrency_id="model-search",
    )
    # Enter searches too, and opens the model when what was typed is an ID.
    models.search_query.submit(
        search_and_open, models.search_inputs, [*models.search_outputs, models.search_pick],
        show_progress="hidden", trigger_mode="multiple", concurrency_id="model-search",
    )
    # The Images page's own way in. It is wired here rather than beside
    # the button because it sets both of this page's kind controls and
    # repaints both lists from them, which needs the two input lists
    # above; see go_to_image_models.
    image_load_button.click(
        go_to_image_models,
        [*models.list_inputs, *models.search_inputs],
        [
            pages.nav, pages.conversations, pages.chat, pages.images, pages.models,
            pages.extension_manager, pages.settings, models.kind_filter, models.name_filter, *models.list_outputs, models.search_kind,
            *models.search_outputs,
        ],
        concurrency_id="model-search",
    )
    models.fits_only.input(
        refresh_search_results,
        [models.model_id, models.search_results_state, models.weight_precision, models.fits_only],
        models.search_results,
    )
    # The marked result is whichever model the ID box names, so the list is
    # redrawn whenever the box changes, not only when a result is pressed: a
    # row of My Models, the default, an extension or a pasted ID would
    # otherwise leave the last pressed result marked behind the Find tab.
    models.model_id.change(
        refresh_search_results,
        [models.model_id, models.search_results_state, models.weight_precision, models.fits_only],
        models.search_results,
        show_progress="hidden", trigger_mode="always_last",
    )
    # A pressed row, an Other versions entry, or an ID pasted and entered:
    # the model opens in the pane, which withdraws the My Models selection
    # the same way, and is checked on Hugging Face.
    models.search_pick.change(
        select_search_result, models.search_pick, models.model_id, show_progress="hidden",
    ).then(
        clear_my_model_selection, None, [models.my_models, models.my_model_detail],
        show_progress="hidden",
    ).then(
        check_model_repository, [models.model_id, models.hf_token], models.repository_result,
        show_progress="hidden", concurrency_id="model-repository-check", trigger_mode="always_last",
    ).then(
        find_versions,
        [models.model_id, models.hf_token, models.search_results_state, models.related_state],
        models.related_state, show_progress="hidden", concurrency_id="model-versions",
        trigger_mode="always_last",
    )
    # Whether a model fits depends on how its weights would be held, so
    # both lists are repainted when that choice changes. Neither touches
    # the cache or the model in memory, so neither is a rescan.
    models.weight_precision.change(
        refresh_my_models, models.list_inputs, models.list_outputs
    ).then(
        refresh_search_results,
        [models.model_id, models.search_results_state, models.weight_precision, models.fits_only],
        models.search_results,
    ).then(refresh_model_switch, models.weight_precision, refresh.switch_outputs)
