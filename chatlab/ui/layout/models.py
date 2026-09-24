"""The Models page: downloads and loads, My Models and the Hugging Face search."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from chatlab import settings
from chatlab.model_cache import DEFAULT_MODEL_SORT, MODEL_SORT_ORDERS
from chatlab.model_discovery import DISCOVERY_ORDERS
from chatlab.ui.icons import icon_classes
from chatlab.ui.layout.common import QUIET_TICK
from chatlab.ui.model_repository import UNCHECKED, check_model_repository, repository_view
from chatlab.ui.model_search import (
    SEARCH_HINT,
    SEARCH_KINDS,
    refresh_after_device,
    refresh_search_results,
    search_models,
    search_table,
    select_search_result,
)
from chatlab.ui.model_switch import (
    go_to_image_models,
    refresh_current_model,
    refresh_model_badge,
    refresh_model_switch,
    select_default_model,
    switch_model,
)
from chatlab.ui.models_page import (
    download_and_load_model,
    download_model,
    load_cached_model,
    refresh_model_actions,
    refresh_stale_model_actions,
    unload_model,
)
from chatlab.ui.my_models import (
    ALL_KINDS,
    MODEL_KIND_FILTERS,
    ask_remove_my_model,
    clear_my_model_selection,
    hide_remove_confirm,
    redownload_my_model,
    refresh_my_models,
    remove_my_model,
    select_my_model,
)
from chatlab.ui.scoring import SCORE_BUDGET_QUEUE, score_token_count
from chatlab.ui.settings_page import refresh_hardware, refresh_thinking_mode


def build_models_page(saved) -> SimpleNamespace:
    """Build the Models page and hand back the controls the wiring reads."""

    with gr.Column(
        scale=1, visible=False, elem_id="models-page"
    ) as models_page:
        gr.Markdown(
            "# Models\nDownload a model from Hugging Face, or load one "
            "already on disk. Files are kept in your normal Hugging Face cache.",
            elem_id="models-hero",
        )
        with gr.Row(elem_id="models-columns"):
            with gr.Column(min_width=360, elem_id="model-controls"):
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown("## Currently loaded")
                    with gr.Row(elem_id="current-model-row"):
                        current_model = gr.HTML(refresh_current_model(), elem_id="currently-loaded-model", container=False, padding=False)
                        unload_button = gr.Button("Unload", size="sm", scale=0, min_width=80)
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown("## Choose a model")
                    with gr.Row(elem_id="model-id-row"):
                        model_id = gr.Textbox(
                            value=settings.model_id_at_startup(saved),
                            label="Hugging Face model ID",
                            placeholder="organization/model-name",
                            info="Paste an ID or select a model below.",
                            scale=4,
                        )
                        check_model_button = gr.Button("Check model", size="sm", scale=1, min_width=100)
                    repository_result = gr.State(None)
                    # What the timer's refresh last painted this card
                    # from; see refresh_stale_model_actions.
                    action_stamp = gr.State(None)
                    repository_detail = gr.Markdown(UNCHECKED, elem_id="model-repository")
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
                        label="Weight precision",
                        info=(
                            "Lower precision saves memory with some loss of accuracy. "
                            "Applies to the next Transformers load on Apple Metal; "
                            "MLX checkpoints use their existing precision."
                        ),
                    )
                    model_availability = gr.Markdown(
                        "Checking downloaded files…", elem_id="model-availability"
                    )
                    with gr.Column(elem_classes=["model-activity"]):
                        gr.Markdown("### Latest model action")
                        model_status = gr.Markdown(
                            "No downloads or loads started in this tab.",
                            elem_id="model-status",
                        )
                    with gr.Row():
                        download_load_button = gr.Button(
                            "Download and load", variant="primary", size="sm"
                        )
                        download_button = gr.Button("Download only", size="sm")
                        cached_button = gr.Button("Load cached", size="sm")

                with gr.Column(elem_id="model-search", elem_classes=["model-card"]):
                    gr.Markdown("## Discover models")
                    gr.Markdown(
                        "Search Hugging Face by name, or leave the box empty to browse. "
                        "A sort reorders the results rather than narrowing what is searched. "
                        "The exception is an empty box under Recommended, which lists the "
                        "bundled starters without going online. "
                        "Selecting a model shows its details before you download.",
                        elem_classes=["scale-caption"],
                    )
                    # One kind at a time, because the hub's own filters
                    # are; see SEARCH_KINDS.
                    search_kind = gr.Radio(
                        choices=list(SEARCH_KINDS),
                        value=SEARCH_KINDS[0][1],
                        show_label=False,
                        container=False,
                        elem_id="search-kind",
                    )
                    with gr.Row():
                        search_order = gr.Dropdown(
                            choices=list(DISCOVERY_ORDERS), value="Recommended",
                            label="Sort", interactive=True,
                            info="Recommended lists ChatLab’s starters first, then the Hub.",
                        )
                        fits_only = gr.Checkbox(
                            label="Fits this computer", value=False,
                            info="Estimated at the chosen weight precision. Unknown sizes are hidden.",
                        )
                    with gr.Row(elem_id="model-search-row"):
                        search_query = gr.Textbox(
                            label="Search Hugging Face",
                            placeholder="Model name or organization; leave blank to browse…",
                            max_lines=1,
                            scale=3,
                            elem_id="model-search-query",
                        )
                        search_button = gr.Button(
                            "Search / refresh", variant="primary", size="sm", scale=0, min_width=120,
                            elem_id="model-search-button",
                        )
                    # A table rather than a list, so the results can
                    # be sorted by any column: clicking a heading
                    # sorts in the browser, and a click on a row is
                    # traced back to its model by the row's own ID.
                    search_results = gr.Dataframe(
                        value=search_table([])["value"],
                        column_widths=search_table([])["column_widths"],
                        label="Search results",
                        show_label=False,
                        interactive=False,
                        wrap=True,
                        elem_id="model-search-results",
                        elem_classes=["model-table"],
                    )
                    search_detail = gr.Markdown(SEARCH_HINT, elem_classes=["model-detail"])
                    search_results_state = gr.State({})
                    # The model ID of the selected row, kept apart from
                    # the table: its own highlight is a cell, and a
                    # sort moves it.
                    search_selection = gr.State(None)

            with gr.Column(min_width=320, elem_classes=["model-card"]):
                gr.Markdown("## My Models")
                my_models_summary = gr.Markdown("", elem_classes=["scale-caption"])
                with gr.Row():
                    sort_models = gr.Dropdown(
                        choices=list(MODEL_SORT_ORDERS),
                        value=DEFAULT_MODEL_SORT,
                        label="Sort by",
                        min_width=120,
                        elem_classes=["model-sort"],
                    )
                    # Beside the sort rather than above the list,
                    # because the two do the same job: they decide
                    # what the reader is looking at rather than what
                    # is on disk. Image models are the ones worth
                    # finding this way - they are the minority, they
                    # are the kind a row has to say out loud, and the
                    # Images page sends a reader here for one.
                    kind_filter = gr.Dropdown(
                        choices=list(MODEL_KIND_FILTERS),
                        value=ALL_KINDS,
                        label="Kind",
                        min_width=120,
                        elem_classes=["model-sort", "model-kind"],
                    )
                my_models = gr.Radio(
                    choices=[],
                    label="Downloaded models",
                    show_label=False,
                    elem_classes=["model-list"],
                )
                my_model_detail = gr.Markdown(
                    "", elem_id="my-model-detail", elem_classes=["model-detail"]
                )
                with gr.Row():
                    redownload_button = gr.Button("Redownload", size="sm", elem_classes=icon_classes("download"))
                    remove_button = gr.Button("Remove", size="sm", elem_classes=icon_classes("trash"))
                    refresh_models_button = gr.Button("Refresh", size="sm", elem_classes=icon_classes("refresh"))
                with gr.Column(
                    visible=False, elem_classes=["remove-confirm"]
                ) as remove_confirm:
                    remove_question = gr.Markdown("", elem_classes=["model-detail"])
                    with gr.Row():
                        confirm_remove_button = gr.Button(
                            "Remove from disk", variant="stop", size="sm"
                        )
                        cancel_remove_button = gr.Button("Cancel", size="sm")
                # The model the open confirmation is about; None when closed.
                pending_removal = gr.State(None)
                # Whether the fit verdicts on screen were given with
                # the device known; see refresh_after_device.
                device_read = gr.State(False)

    return SimpleNamespace(
        action_stamp=action_stamp,
        cached_button=cached_button,
        cancel_remove_button=cancel_remove_button,
        check_model_button=check_model_button,
        confirm_remove_button=confirm_remove_button,
        current_model=current_model,
        device_read=device_read,
        download_button=download_button,
        download_load_button=download_load_button,
        fits_only=fits_only,
        hf_token=hf_token,
        kind_filter=kind_filter,
        model_availability=model_availability,
        model_id=model_id,
        model_status=model_status,
        models_page=models_page,
        my_model_detail=my_model_detail,
        my_models=my_models,
        my_models_summary=my_models_summary,
        pending_removal=pending_removal,
        redownload_button=redownload_button,
        refresh_models_button=refresh_models_button,
        remove_button=remove_button,
        remove_confirm=remove_confirm,
        remove_question=remove_question,
        repository_detail=repository_detail,
        repository_result=repository_result,
        search_button=search_button,
        search_detail=search_detail,
        search_kind=search_kind,
        search_order=search_order,
        search_query=search_query,
        search_results=search_results,
        search_results_state=search_results_state,
        search_selection=search_selection,
        sort_models=sort_models,
        unload_button=unload_button,
        weight_precision=weight_precision,
    )


def wire_model_actions(ui: SimpleNamespace) -> None:
    """Wire the Models page's downloads, loads and unloads, and the rescan that follows each."""

    # Every handler that can change what is on disk or in memory rescans
    # the cache afterwards, so My Models never shows a stale list.
    # The typed ID stays last: the model-actions listeners assert it is
    # the input the refresh is given, and a new argument goes before it
    # rather than displacing it.
    models_inputs = [ui.my_models, ui.sort_models, ui.weight_precision, ui.kind_filter, ui.model_id]
    models_outputs = [ui.my_models, ui.my_model_detail, ui.my_models_summary]
    action_inputs = [ui.model_id, ui.my_models, ui.repository_result, ui.hf_token]
    action_outputs = [
        ui.model_availability, ui.download_load_button, ui.download_button, ui.cached_button
    ]

    # Include programmatic selections (search, default, and rescans).
    # The selected row takes precedence, just as it does for a load.
    for control in action_inputs:
        control.change(
            refresh_model_actions, action_inputs, action_outputs,
            show_progress="hidden", trigger_mode="always_last",
            concurrency_id="model-actions",
        )
    # Downloads run in worker threads, so selections alone cannot keep the
    # local-file status current while files arrive (or in another tab).
    # The timer covers that, but an idle tick paints nothing rather than
    # scanning the cache every couple of seconds forever in every open
    # session; see refresh_stale_model_actions, which hands back the stamp
    # the next tick compares against.
    ui.badge_timer.tick(
        refresh_stale_model_actions,
        [*action_inputs, ui.action_stamp],
        [*action_outputs, ui.action_stamp],
        show_progress="hidden", show_progress_on=[], trigger_mode="always_last",
        concurrency_id="model-actions",
    )

    # Slow network requests scope state to the ID and credentials; rendering
    # reads both again so an old request cannot verify a newer selection.
    # Keep checks explicit: clicking the button also blurs the textbox,
    # which would otherwise enqueue a second request for the same ID.
    for event in (ui.check_model_button.click, ui.model_id.submit):
        event(
            check_model_repository, [ui.model_id, ui.hf_token], ui.repository_result,
            show_progress="hidden", concurrency_id="model-repository-check",
            trigger_mode="always_last",
        )
    repository_inputs = [ui.model_id, ui.repository_result, ui.hf_token, ui.my_models]
    repository_outputs = [ui.repository_detail, ui.weight_precision]
    for event in (
        *(control.change for control in repository_inputs), ui.demo.load, ui.nav.change,
    ):
        event(
            repository_view, repository_inputs, repository_outputs, show_progress="hidden",
            concurrency_id="model-repository-view", trigger_mode="always_last",
        )
    ui.hf_token.input(lambda: None, None, ui.repository_result, show_progress="hidden")
    for event in (ui.demo.load, ui.nav.change, ui.badge_timer.tick):
        event(refresh_current_model, None, ui.current_model, **QUIET_TICK)

    def refresh_actions(event):
        return event.then(
            refresh_model_actions, action_inputs, action_outputs,
            show_progress="hidden", concurrency_id="model-actions",
        ).then(
            repository_view, repository_inputs, repository_outputs,
            show_progress="hidden", concurrency_id="model-repository-view",
        )

    # Refresh model-dependent displays after explicit model actions.
    # The timer also catches changes from other tabs, but this updates
    # the badge and token count immediately in the tab that acted.
    def rescan(event, *, reloads: bool = True):
        """Rescan the cache after ``event``, and re-read what the model feeds."""

        event = event.then(refresh_my_models, models_inputs, models_outputs)
        event = refresh_actions(event)
        event = event.then(refresh_current_model, None, ui.current_model, show_progress="hidden")
        # What is on disk is what the switcher offers, so it follows every
        # rescan, download-only included.
        event = event.then(
            refresh_model_switch,
            ui.weight_precision,
            ui.switch_outputs,
            show_progress="hidden",
        )
        if not reloads:
            return event
        return (
            event.then(refresh_model_badge, None, ui.badge_outputs)
            .then(
                score_token_count,
                ui.score_budget_inputs,
                ui.score_budget_outputs,
                show_progress="hidden",
                concurrency_id=SCORE_BUDGET_QUEUE,
            )
            # A load or an unload is the largest change the machine's
            # memory sees, so the hardware panel is re-read after it
            # rather than left showing what was true before.
            .then(refresh_hardware, None, ui.hardware_view)
            .then(refresh_thinking_mode, None, ui.thinking_mode)
        )

    # Download-only changes the cache without changing the loaded model.
    rescan(
        ui.download_button.click(
            download_model, [ui.model_id, ui.hf_token, ui.my_models], ui.model_status
        )
    )
    rescan(
        ui.download_load_button.click(
            download_and_load_model,
            [ui.model_id, ui.hf_token, ui.my_models, ui.weight_precision],
            ui.model_status,
        )
    )
    rescan(
        ui.cached_button.click(
            load_cached_model,
            [ui.model_id, ui.my_models, ui.weight_precision],
            ui.model_status,
        )
    )
    rescan(ui.unload_button.click(unload_model, outputs=ui.model_status))
    # A pick in the chat page's switcher is a load from the cache, and is
    # followed by the same rescan as the button. Its status goes to the
    # Models page's card, where the switcher's own repaint would
    # otherwise drop the progress the load is reporting. That page is not
    # the one the reader is on, so the badge beside the switcher is
    # written from the same handler - it is what shows the reader the
    # load and how far it has come - and switch_model also toasts an
    # ending the card alone would have kept to itself; see
    # announce_switch_outcome.
    rescan(
        ui.model_switch.input(
            switch_model,
            [ui.model_switch, ui.weight_precision],
            [ui.model_switch, ui.model_status, ui.model_badge_view],
        )
    )
    # A manual refresh, a new sort order and a new kind filter reorder or
    # narrow a list; none of them changes what is on disk or in memory,
    # which is all the badge and the count ask about.
    refresh_actions(
        ui.refresh_models_button.click(refresh_my_models, models_inputs, models_outputs)
    )
    ui.sort_models.input(refresh_my_models, models_inputs, models_outputs)
    ui.kind_filter.input(refresh_my_models, models_inputs, models_outputs)
    # Before the reader chooses an ID, startup can highlight the loaded model.
    refresh_actions(ui.demo.load(refresh_my_models, [ui.my_models, ui.sort_models], models_outputs))
    # The badge's timer corrects the fit verdicts once torch has finished
    # importing: the page is painted before that, so the first verdicts
    # are given without knowing the device. It repaints once and then
    # does nothing for the rest of the session.
    ui.badge_timer.tick(
        refresh_after_device,
        [ui.device_read, *models_inputs, ui.search_selection, ui.search_results_state, ui.fits_only],
        [*models_outputs, ui.search_results, ui.search_detail, ui.search_selection, ui.device_read],
        **QUIET_TICK,
    )

    # What the wiring after this reads.
    ui.models_inputs = models_inputs
    ui.models_outputs = models_outputs
    ui.rescan = rescan


def wire_model_lists(ui: SimpleNamespace) -> None:
    """Wire My Models and the Hugging Face search: selecting, removing, searching and filtering."""

    # Selecting a default is navigation only. The Models page owns the
    # explicit download and load actions, including their errors.
    #
    # The search table is not among the outputs. Its highlight is kept
    # by the browser, and the click on this button is itself a click
    # outside the table, which Gradio's Dataframe answers by clearing
    # that highlight (Table.svelte, handle_click_outside). Repainting
    # the table would not clear it: a new value leaves the selected
    # cells alone, and an identical value is not applied at all.
    ui.default_model_button.click(
        select_default_model,
        None,
        [
            ui.model_id,
            ui.my_models,
            ui.my_model_detail,
            ui.search_selection,
            ui.search_detail,
            ui.model_status,
            ui.remove_confirm,
            ui.pending_removal,
            ui.nav,
            ui.conversation_pane,
            ui.chat_page,
            ui.images_page,
            ui.models_page,
            ui.settings_page,
        ],
    )
    # .input rather than .change: the refresh above also sets the radio,
    # and a .change listener would rewrite the model ID box on each rescan.
    ui.my_models.input(
        select_my_model,
        [ui.my_models, ui.weight_precision],
        [ui.model_id, ui.my_model_detail],
    )
    # .input again, for the same reason: only the reader's own typing
    # withdraws the selection, never a refresh writing the box.
    ui.model_id.input(clear_my_model_selection, None, [ui.my_models, ui.my_model_detail])
    # A pending removal is about the model that was selected when it was
    # asked for, so changing the selection withdraws it.
    confirm_outputs = [ui.remove_confirm, ui.pending_removal]
    ui.my_models.input(hide_remove_confirm, None, confirm_outputs)
    ui.model_id.input(hide_remove_confirm, None, confirm_outputs)
    ui.rescan(
        ui.redownload_button.click(
            redownload_my_model, [ui.my_models, ui.hf_token], ui.model_status
        )
    )
    ui.remove_button.click(
        ask_remove_my_model,
        ui.my_models,
        [ui.model_status, ui.remove_confirm, ui.remove_question, ui.pending_removal],
    )
    # The confirm button deletes the model the question named, never the
    # radio's current value: see ask_remove_my_model.
    ui.rescan(
        ui.confirm_remove_button.click(
            remove_my_model, ui.pending_removal, [ui.model_status, *confirm_outputs]
        )
    )
    ui.cancel_remove_button.click(hide_remove_confirm, None, confirm_outputs)

    search_outputs = [
        ui.search_results, ui.search_detail, ui.search_results_state, ui.search_selection
    ]
    search_inputs = [
        ui.search_query, ui.hf_token, ui.weight_precision, ui.search_kind, ui.search_order, ui.fits_only
    ]
    # The page loads with an empty box, and Recommended answers that from
    # the bundled starters, so the first paint does not go online.
    ui.demo.load(search_models, search_inputs, search_outputs)
    ui.search_button.click(search_models, search_inputs, search_outputs)
    ui.search_query.submit(search_models, search_inputs, search_outputs)
    ui.search_kind.input(search_models, search_inputs, search_outputs)
    # The Images page's own way in. It is wired here rather than beside
    # the button because it sets both of this page's kind controls and
    # repaints both lists from them, which needs the two input lists
    # above; see go_to_image_models.
    ui.image_load_button.click(
        go_to_image_models,
        [*ui.models_inputs, *search_inputs],
        [
            ui.nav, ui.conversation_pane, ui.chat_page, ui.images_page, ui.models_page,
            ui.settings_page, ui.kind_filter, *ui.models_outputs, ui.search_kind,
            *search_outputs,
        ],
    )
    ui.search_order.input(search_models, search_inputs, search_outputs)
    ui.fits_only.input(
        refresh_search_results,
        [ui.search_selection, ui.search_results_state, ui.weight_precision, ui.fits_only],
        [ui.search_results, ui.search_detail, ui.search_selection],
    )
    # Picking a search result names a model too, so it withdraws the My
    # Models selection the same way typing an ID does.
    ui.search_results.select(
        select_search_result,
        [ui.search_results_state, ui.weight_precision],
        [ui.model_id, ui.search_detail, ui.search_selection],
    ).then(clear_my_model_selection, None, [ui.my_models, ui.my_model_detail])
    # Whether a model fits depends on how its weights would be held, so
    # both lists are repainted when that choice changes. Neither touches
    # the cache or the model in memory, so neither is a rescan.
    ui.weight_precision.change(
        refresh_my_models, ui.models_inputs, ui.models_outputs
    ).then(
        refresh_search_results,
        [ui.search_selection, ui.search_results_state, ui.weight_precision, ui.fits_only],
        [ui.search_results, ui.search_detail, ui.search_selection],
    ).then(refresh_model_switch, ui.weight_precision, ui.switch_outputs)
