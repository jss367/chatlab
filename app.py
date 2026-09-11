"""ChatLab interface for chatting with and inspecting model tokens.

The interface lives in the ``ui`` package, one module per page or panel;
this module gathers every name under one roof, which is what the desktop
launcher and the tests import, and starts the app when run directly.
"""

from __future__ import annotations

import contextlib
import html
import logging
import os
import random
import re
import threading
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio.utils import get_upload_folder

import api
import charts
import library
import settings
from conversation import (
    CHAT_PREFIX,
    FORK_PREFIX,
    MAIN_BRANCH,
    THINK_CLOSE,
    branch_choices,
    branch_stamp,
    copy_forks,
    copy_turns,
    display_messages,
    drop_branch,
    forget_measurements,
    fork_at,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    new_forks,
    next_branch_name,
    next_fork_name,
    put_branch,
    split_reasoning,
    to_json,
    user_index_at_or_before,
)
import image_runtime
from model_runtime import (
    DEFAULT_MODEL_SORT,
    IMAGE_KIND,
    MODEL_SORT_ORDERS,
    MODEL_WEIGHTS,
    PROMPT_SCORE_LIMIT,
    TEXT_KIND,
    CachedModel,
    CacheStatus,
    DownloadSnapshot,
    HubModel,
    LoadProgress,
    LoadSnapshot,
    ModelBusy,
    ModelChanged,
    ModelDownloading,
    ModelLoaded,
    ModelManager,
    cache_root,
    cache_status,
    format_bytes,
    format_count,
    list_cached_models,
    search_hub_models,
    sort_cached_models,
)
from token_metrics import (
    COLOR_SCALES,
    DEFAULT_COLOR_SCALE,
    PROMPT_ATTENTION_SCALE,
    UNSCORED_BEYOND_LIMIT,
    category_for,
    summarize,
)
from prompt_batch import (
    BATCH_CSV_NAME,
    parse_prompt_file,
    parse_prompts,
    prompts_to_text,
    write_batch_csv,
    write_batch_trace,
)
from trace_export import (
    build_trace,
    trace_to_csv,
    traces_to_csv,
    write_private_text,
    write_trace_export,
)
from ui.runtime import MANAGER
from ui.common import (
    CHART_EVERY,
    CHAT_PAGE,
    CONVERSATION_PANE_WIDTH,
    DEFAULT_MODEL_DOWNLOAD,
    DOWNLOAD_BAR_WIDTH,
    DOWNLOAD_POLL_SECONDS,
    IMAGES_PAGE,
    IncompleteSnapshotError,
    LOAD_POLL_SECONDS,
    METRIC_GLOSSARY,
    MODELS_PAGE,
    NAV_ICONS,
    NAV_PANE_WIDTH,
    NO_TOKEN_SELECTED,
    PAGES,
    RATE_WINDOW_SECONDS,
    RESPONSE_STRIP_LABEL,
    SEAM_CAVEAT,
    SEED_LIMIT,
    SETTINGS_PAGE,
    TEMPLATE_CAVEAT,
    Card,
    alarm,
    describe_duration,
    failure_card,
    failure_status,
    finalize_partial,
    hint,
    metric_term,
    progress_bar,
    send_stop_buttons,
    show_page,
    status_card,
)
from ui.conversations import (
    PANEL_KEPT,
    conversation_list_update,
    delete_fork,
    fork_conversation,
    fork_refused,
    load_conversation,
    new_conversation,
    panel_reset,
    refresh_conversation_list,
    remember_branch_sampling,
    remember_forks,
    remember_message,
    restore_conversations,
    sampling_updates,
    save_conversation,
    selected_turn,
    switch_fork,
)
from ui.images_page import (
    DRAW_POLL_SECONDS,
    EMPTY_PROMPT,
    IMAGE_OUTPUT_NAMES,
    NO_ATTENTION,
    NO_IMAGE_MODEL,
    NO_TRAJECTORY,
    NOTHING_TO_STOP,
    PROMPT_STRIP_LABEL,
    TEXT_MODEL_LOADED,
    attention_note,
    attention_overlay,
    draw,
    prompt_strip_value,
    remember_committed_image_seed,
    remember_image_settings,
    remember_token,
    select_step,
    select_token,
    stop_drawing,
    trajectory_frame,
)
from ui.generation import (
    BUSY_STATUS,
    CHAT_OUTPUT_NAMES,
    LOADING_STATUS,
    NOTHING_TO_CLEAR,
    NO_MODEL_STATUS,
    ask_clear_chat,
    automatic_reasoning_close_count,
    branch_from,
    branch_with_text,
    busy_state,
    chat,
    clear_chat,
    edit_message,
    generate_reply,
    generation_progress,
    hide_clear_confirm,
    idle_state,
    literal_prefill_count,
    literal_text_ranges,
    regenerate_from,
    resolve_seed,
    retry_last,
    retry_message,
    split_response_text,
    stop_generation,
    undo_from,
    undo_last,
    undo_message,
)
from ui.inspection import (
    INSPECT_BUSY,
    INSPECT_FIRST,
    INSPECT_GONE,
    INSPECT_HINT,
    INSPECT_LOADING,
    INSPECT_MODEL_CHANGED,
    INSPECT_OUTPUT_ONLY,
    NAV_TILE_CSS,
    inspect_layers,
    remember_inspect_target,
    render_attention,
    reset_inspection,
)
from ui.layout import (
    CONVERSATION_PANE_QUEUE,
    build_app,
)
from ui.models_page import (
    BADGE_REFRESH_SECONDS,
    MISSING_FILES_PATTERN,
    NO_CACHED_MODEL_SELECTED,
    NO_MODEL_BADGE,
    NO_MODEL_TO_MANAGE,
    NO_RESULT_SELECTED,
    Pace,
    RateMeter,
    KIND_NAMES,
    SEARCH_HINT,
    SEARCH_HINTS,
    SEARCH_KINDS,
    SWITCH_BUSY,
    SWITCH_FIT_SECONDS,
    SWITCH_LOADING,
    SwitchStamp,
    UNSUPPORTED_REASON,
    announce_switch_outcome,
    ask_remove_my_model,
    cached_model_label,
    chosen_model,
    clear_my_model_selection,
    describe_cache,
    describe_cached_model,
    describe_fetched,
    describe_hub_model,
    describe_missing,
    describe_on_disk,
    download_and_load_model,
    download_detail,
    download_model,
    downloading_refusal,
    format_timestamp,
    go_to_models,
    hide_remove_confirm,
    hub_model_label,
    incomplete_snapshot_detail,
    load_cached_model,
    load_detail,
    loaded_model_badge,
    loaded_refusal,
    model_badge,
    model_snapshot,
    my_models_summary,
    redownload_my_model,
    refresh_after_device,
    refresh_image_badge,
    refresh_model_badge,
    refresh_model_switch,
    refresh_my_models,
    refresh_search_results,
    refresh_stale_model_switch,
    removal_refusal,
    remove_my_model,
    search_models,
    select_default_model,
    select_my_model,
    select_search_result,
    stream_download,
    stream_load,
    switch_choices,
    switch_model,
    switch_value,
    unload_model,
    where_to_use,
)
from ui.panel import (
    BRANCH_HINT,
    BRANCH_MODEL_CHANGED,
    BRANCH_REASONING_CLOSE,
    BRANCH_TEXT_EMPTY,
    BRANCH_TEXT_HINT,
    BRANCH_UNAVAILABLE,
    branch_ready_text,
    choose_alternative,
    cleared_strips,
    describe_token,
    empty_metrics,
    event_index,
    inspect_token,
    new_metrics_generation,
    prompt_note_text,
    recolor,
    remember_selection,
    resolve_scale,
    stamped,
    strip_update,
    strip_value,
)
from ui.prompts import (
    BATCH_BUSY,
    BATCH_HEADERS,
    BATCH_LOADING,
    BATCH_MODEL_CHANGED,
    BATCH_NO_MODEL,
    BATCH_NO_PROMPTS,
    BATCH_OUTPUT_NAMES,
    EXCERPT_LENGTH,
    PARAGRAPH_NOTE,
    PROMPT_COUNT_HINT,
    batch_directory,
    batch_files,
    batch_progress,
    batch_row,
    count_prompts,
    excerpt,
    failed_row,
    load_prompt_file,
    resolve_prompts,
    run_prompts,
    same_lines,
    stop_batch,
)
from ui.scoring import (
    SAMPLING_LABEL_QUEUE,
    SCORE_BUDGET_QUEUE,
    SCORE_BUSY,
    SCORE_COUNT_HINT,
    SCORE_COUNT_LOADING,
    SCORE_COUNT_UNKNOWN,
    SCORE_LOADING,
    recover_score_budget,
    score_text,
    score_token_count,
)
from ui.settings_page import (
    HARDWARE_UNREAD,
    PERSISTED_SETTING_NAMES,
    hardware_card,
    refresh_hardware,
    remember_committed_seed,
    remember_prefill_limit,
    remember_settings,
    restore_settings,
    sampling_label,
    update_sampling_label,
)
from ui.styles import (
    CSS,
    MESSAGE_BOX_MAX_LINES,
    SHORTCUT_JS,
    message_box_settings,
    set_message_box_keys,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    conductor_port = os.environ.get("CONDUCTOR_PORT")
    demo = build_app().queue(default_concurrency_limit=1)
    # Gradio builds its FastAPI application inside launch(), so the API is
    # added once the server is up rather than before it. prevent_thread_lock
    # is what makes that possible: the thread is held afterwards instead.
    demo.launch(
        inbrowser=conductor_port is None,
        server_port=int(conductor_port) if conductor_port else None,
        prevent_thread_lock=True,
    )
    api.attach(demo.app)
    demo.block_thread()
