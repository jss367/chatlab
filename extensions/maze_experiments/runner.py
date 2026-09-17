"""Single-episode controller: token generation is separate from authoritative maze movement."""
from __future__ import annotations

import copy
import json
import logging
import re
import threading
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .dynamic_maze import (FORMAT as CHANGING_FORMAT, ChangingMaze, check_closure, load_maze,
                           maze_at_turn, validate_drops, validate_pending, validate_updates)
from .maze import SYSTEM, Maze, TOOLS, apply_call, default_instruction, initial_history, parse_call
from extension_api import write_private_text

# One line for each response and each episode outcome, so a run read in
# ChatLab.log afterwards says what it did rather than only that a model was
# asked for tokens. The token counts are here because the memory report beside
# them is read against them.
logger = logging.getLogger(__name__)

FORMAT = "chatlab-maze-run-1"
TERMINAL = {"arrived", "abandoned", "budget", "stopped", "error"}
# How long after an interruption a first accepted move still counts as recovery.
# The pilot's window, kept as the default so runs written before it was
# configurable are read under the window that scored them.
RECOVERY_DEFAULTS = {"recovery_tokens": 1024, "recovery_attempts": 4}


@dataclass
class Episode:
    maze: Maze
    config: dict
    run_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "ready"
    detail: str = "Ready. Play the episode or use Next to generate one response."
    messages: list = field(default_factory=list)
    events: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    position: tuple = ()
    model_id: str | None = None
    load_id: str | None = None
    sampled_tokens: int = 0
    tool_attempts: int = 0
    supplied_moves: int = 0
    interrupted: bool = False
    intervention_turn: int | None = None
    intervention_tokens: int = 0
    intervention_attempts: int = 0
    resumed: bool | None = None
    first_move_progress: bool | None = None
    latency: int | None = None
    manual_intervention: bool = False
    interrupt_next: bool = False
    # A cell queued to close before the next response, and every queued closure
    # that could not happen by the time its response came round.
    close_next: tuple = ()
    dropped_closures: list = field(default_factory=list)
    pause_requested: bool = False
    stop_requested: bool = False
    busy: bool = False
    replay_only: bool = False
    token_edit: dict | None = None
    pending_edit: dict | None = None
    # The response the viewer last drew. Previous, Next and playback move
    # relative to it, so a rapid second click cannot resend a stale index.
    viewing: int = -1
    # Which playback run owns the view. Starting one supersedes the last, so
    # two runs in the same session cannot repaint each other's frames.
    playback_token: int = 0
    playing: bool = False
    reveal_route: bool = False
    created_at: float = field(default_factory=time.time)

    def __deepcopy__(self, memo):
        # Gradio copies the initial State once per browser/API session. Each
        # session needs its own episode and lock; subsequent callbacks share it.
        result = object.__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, threading.Lock() if key == "lock" else copy.deepcopy(value, memo))
        if result.phase == "ready" and not result.turns:
            result.run_id = uuid4().hex
            result.created_at = time.time()
        return result

    def __post_init__(self):
        self.lock = threading.Lock()
        self.config = copy.deepcopy(self.config)
        self.config.setdefault("goal_mode", "coordinates")
        self.config.setdefault("goal_hint", "")
        # Runs predating editable wording carry no prompt, so they keep the
        # defaults their goal mode sent. An empty string is a deliberate blank.
        if not isinstance(self.config.get("system_prompt"), str):
            self.config["system_prompt"] = SYSTEM
        if not isinstance(self.config.get("instruction"), str):
            self.config["instruction"] = default_instruction(self.config["goal_mode"])
        # Runs predating the recorded recovery window carry the window that
        # scored them, so an old export still says which one it was read under.
        # A window written down is used as written, or refused: silently
        # replacing it would score the run under a window it does not name.
        for key, fallback in RECOVERY_DEFAULTS.items():
            if key not in self.config:
                self.config[key] = fallback
            elif type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError("The recovery window must be a positive number of sampled tokens and tool attempts.")
        supplied = int(self.config.get("supplied_moves", 3))
        self.messages, self.events, self.position = initial_history(
            self.maze, supplied, goal_mode=self.config["goal_mode"], goal_hint=self.config["goal_hint"],
            system=self.config["system_prompt"], instruction=self.config["instruction"])
        self.supplied_moves = supplied

    @property
    def current_maze(self):
        """The map as it stands now: the original plus every closure recorded so far.

        Rebuilt from the original and the record rather than kept as state of
        its own, so the map the simulator moves on is the same one a replay of
        this run reconstructs and neither can drift from the other.
        """
        return maze_at_turn(self.maze, self.config.get("map_updates", ()), None)

    @property
    def map_changes(self):
        """Whether this run's walls can close while it is running."""
        return isinstance(self.maze, ChangingMaze)

    def model_state(self, error=None):
        return self.current_maze.state(self.position, error, goal_mode=self.config["goal_mode"],
                                       goal_hint=self.config["goal_hint"])

    @property
    def moves(self):
        return sum(e["accepted"] for e in self.events)

    def payload(self):
        keys = ("run_id", "phase", "detail", "messages", "events", "turns", "position", "model_id", "load_id",
                "sampled_tokens", "tool_attempts", "supplied_moves", "interrupted", "intervention_turn",
                "intervention_tokens", "intervention_attempts", "resumed", "first_move_progress", "latency",
                "manual_intervention", "created_at", "token_edit", "pending_edit", "dropped_closures",
                "close_next")
        return {"format": CHANGING_FORMAT if self.map_changes else FORMAT, "maze": self.maze.to_dict(), "config": self.config,
                "exploratory": True, "tokenizer_note": "Every turn records its actual prompt IDs. Later turns are templated from the complete prior response text, including reasoning.",
                **{k: getattr(self, k) for k in keys}}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        temp = path.with_suffix(".json.tmp")
        write_private_text(temp, json.dumps(self.payload(), ensure_ascii=False, allow_nan=False))
        temp.replace(path)
        return path

    def export(self):
        """Gradio only serves returned files from permitted temporary locations."""
        return self.save(Path(tempfile.mkdtemp(prefix="chatlab-maze-")))

    def request_pause(self):
        self.pause_requested = True

    def request_stop(self, save_dir=None):
        with self.lock:
            if self.replay_only or self.phase in TERMINAL:
                return
            self.stop_requested = True
            if self.busy:
                return  # The active stream owns cleanup and persistence.
            self.phase, self.detail = "stopped", "Stopped by you. This is not scored as model abandonment."
            abandon_closure(self)
            if save_dir:
                try:
                    self.save(save_dir)
                except OSError as exc:
                    self.warn_autosave(str(exc))

    def warn_autosave(self, error):
        logger.warning("Autosave of run %s failed: %s", self.run_id, error)
        self.detail += (f" Autosave failed: {error}. Latest changes remain in memory. "
                        "Use Export run JSON to download them, and check the run directory or free disk space.")

    def request_interruption(self):
        if self.phase in TERMINAL or self.replay_only:
            raise ValueError("Start a new episode to request an interruption. This episode is finished or is a saved replay.")
        if self.interrupted:
            raise ValueError("This episode already contains its interruption. Start another run to compare settings.")
        if not self.config.get("interruption_text", "").strip():
            raise ValueError("Choose interruption text before starting this episode.")
        self.interrupt_next, self.manual_intervention = True, True

    def request_closure(self, cell):
        """Queue one cell to be walled off before the next generated response.

        Checked here against the map and the position as they stand, so a cell
        that cannot be closed is refused where it was asked for. It is checked
        again when it lands, because the character moves in between.

        One at a time, as an interruption is. A second request would replace a
        queued cell that the reader has already been told will close, and the
        one it replaced would leave no mark anywhere: never applied, so not in
        map_updates, and never refused by the map, so not among the closures
        this run reports dropping.

        The whole check and the assignment are under the episode's lock, unlike
        the interruption's. The page registers the close button off the queue,
        so two clicks run as two callbacks at once, and an interruption is a
        flag both of them can set to the same True while a queued cell is a
        cell: read and written apart, the second click's test of an empty queue
        would pass against the first click's and one of them would go missing.
        """
        with self.lock:
            if not self.map_changes:
                raise ValueError("This episode's map is fixed. Start an episode with a changing map to close cells during a run.")
            if self.phase in TERMINAL or self.replay_only:
                raise ValueError("Start a new episode to change the map. This episode is finished or is a saved replay.")
            if self.close_next:
                raise ValueError(f"Row {self.close_next[0]}, column {self.close_next[1]} is already queued to close "
                                 "before the next response. Let it land before queueing another.")
            check_closure(self.current_maze, self.position, cell)
            self.close_next, self.manual_intervention = tuple(cell), True


def abandon_closure(episode):
    """Record a queued closure the run will never reach. The caller holds the lock.

    A queued cell is applied by the next response, so an episode that ends
    without one leaves the reader told a closure would happen and the run
    showing no sign that anything was asked. That is the same silence
    dropped_closures was added to break, so it is broken the same way.
    """
    if not episode.close_next:
        return
    cell, episode.close_next = tuple(episode.close_next), ()
    episode.dropped_closures.append(dict(
        before_turn=len(episode.turns), cell=list(cell),
        reason="The episode ended before the closure could land."))
    logger.warning("Run %s ended %s with the closure at %s still queued",
                   episode.run_id, episode.phase, cell)


def apply_closure(episode):
    """Wall off the queued cell, if there is one, and record what the map became.

    The character has moved since the closure was queued, so the same rules are
    asked again here. One that has become impossible is dropped and recorded as
    dropped: the run went on under a map the reader asked to change and which
    did not change, and anything scoring the run needs to know that rather than
    read an unchanged map as an unchanged intention.

    Taking the cell, reading the map and writing the change are one operation,
    because emptying the queue ahead of recording the closure would leave a
    moment when request_closure sees a free queue beside a map that has not
    changed yet, and takes the same cell again. The run would then record
    closing a wall, which is a closure no map ever allowed and which the reader
    of that run would refuse.
    """
    with episode.lock:
        if not episode.close_next:
            return
        cell, episode.close_next = tuple(episode.close_next), ()
        boundary = len(episode.turns)
        try:
            changed = check_closure(episode.current_maze, episode.position, cell)
        except ValueError as exc:
            episode.dropped_closures.append(dict(before_turn=boundary, cell=list(cell), reason=str(exc)))
            episode.detail = f"The queued closure at row {cell[0]}, column {cell[1]} was dropped. {exc}"
            logger.warning("Run %s dropped the closure at %s before response %s: %s",
                           episode.run_id, cell, boundary + 1, exc)
            return
        episode.config.setdefault("map_updates", []).append(
            dict(before_turn=boundary, position=list(episode.position),
                 closed_cell=list(cell), grid=list(changed.grid)))
        episode.detail = f"The map changed: row {cell[0]}, column {cell[1]} is now a wall."
        logger.info("Run %s closed %s before response %s, leaving %s moves to the destination",
                    episode.run_id, cell, boundary + 1, len(changed.route(episode.position)) - 1)


def context_messages(episode, index):
    """The messages a response was given, or is about to be given at ``index`` -1.

    The history grows by exactly two messages - the assistant's response and
    the simulator's reply - for each response that attempted a call, and by
    none for one that ended without attempting anything, so a response's own
    prompt is the history up to the calls the responses before it made. The
    initial block is the setup plus one such pair per supplied move.
    """
    supplied = sum(event["source"] == "supplied" for event in episode.events)
    attempts = sum("event" in turn for turn in episode.turns[:max(index, 0)])
    return episode.messages[:2 + 2 * (supplied + attempts)]


def interrupted_prefix(episode, manager):
    if episode.interrupted or not episode.config.get("interruption_text"):
        return []
    if not episode.interrupt_next and episode.moves < episode.config["interrupt_after"]:
        return []
    text = episode.config["interruption_text"]
    if any(mark in text for mark in ("<tool_call", "</tool_call", "<|im_", "<|endoftext|>", "<think>", "</think>", "```", "~~~")):
        raise ValueError("Interruption text cannot supply tool syntax, conversation boundary tokens, reasoning delimiters or code fences.")
    # forced_ids inserts into the actual next response, including inside an
    # already-open reasoning block. Unlike answer_prefill it adds no </think>.
    ids = manager.encode(text)
    count = int(episode.config["prefix_tokens"])
    return list(ids if count == 0 else ids[:count])


def forkable_without_evidence(episode, manager):
    """Whether this episode may be forked where the run records nothing to check.

    Only a live episode of this session qualifies: its load identifier was
    assigned by this process, so it names the load in memory now. An uploaded
    replay never qualifies, because load_count restarts at zero in each
    process, and the first load of a repository in one session answers to the
    same name as the first load in the next.
    """
    return (not episode.replay_only and manager.load_id is not None
            and manager.load_id == episode.load_id)


def visible_token_ids(metrics, literal_prefill_tokens, hidden_ids):
    """The IDs a stretch of recorded metrics was decoded into text through.

    The runtime streams each response through IncrementalDecoder, which drops
    a hidden special rather than decoding it, so those IDs appear among the
    turn's metrics and never in its text. Reader-supplied prefill is the
    exception the runtime makes: replay forces those tokens visible, including
    special-token spellings.
    """
    return [metric["token_id"] for index, metric in enumerate(metrics)
            if index < literal_prefill_tokens or metric["token_id"] not in hidden_ids]


def literal_prefill_of(turn):
    """How many leading tokens of this response replay forces visible."""
    return turn.get("literal_prefill_tokens", turn["forced_prefix_tokens"])


def verify_recorded_text(episode, turn_index, manager):
    """Refuse a fork whose stored IDs no longer decode to the text they recorded.

    The same repository ID can be re-downloaded at a revision whose tokenizer or
    vocabulary changed, so a matching model_id is not on its own evidence that
    replaying stored IDs reproduces the original run. Every response records the
    text its own IDs decoded to at generation time, so the loaded tokenizer can
    be checked against the run itself, with no fingerprint that existing exports
    never carried.

    Each response is compared as a whole rather than a token at a time, because
    decoding is not piecewise. A revision that moves an ID from the word-boundary
    piece "▁world" to "world" leaves it decoding alone to "world" either way,
    while the sequence after "Hello" reads "Hello world" under one vocabulary and
    "Helloworld" under the other. Decoding the response as a whole also puts a
    byte-level piece among the neighbours that complete its character, so a
    fragment that names no ID by itself is still pinned down by the text it
    makes with them.

    Every earlier response is verified in full, because the fork rebuilds each of
    them from its stored IDs. The edited response is verified in full too, past
    the edited token as well as before it: boundary semantics only mean anything
    in context, so a vocabulary that has moved anywhere in that response is
    evidence the tokenizer is not the one that produced the run.

    A load identifier is not evidence of anything here, because load_count
    restarts at zero in each process, so the first load of a repository in one
    session and its first load in the next both answer to the same name.
    """
    hidden = manager.hidden_token_ids
    for turn in episode.turns[:turn_index + 1]:
        try:
            current = manager.decode(visible_token_ids(turn["metrics"], literal_prefill_of(turn), hidden))
        except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError("The loaded model cannot decode this run's token IDs, so its tokenizer is not the "
                             f"one that produced the run ({exc}). This happens when the same model ID has been "
                             "re-downloaded at a different revision.") from exc
        if current != turn["text"]:
            raise ValueError("The loaded weights tokenize differently from the ones that produced this run, so "
                             "its tokens cannot be replayed. This happens when the same model ID has been "
                             "re-downloaded at a different revision; load that snapshot to fork this run.")


def verify_recorded_candidate(episode, manager):
    """Refuse a recorded alternative on a run this session did not produce.

    The chosen alternative is the one ID the fork replays that the run never
    generated, so no response text stands behind it, and nothing an export
    records says how the model that offered it spelled that ID. build_metric
    records an alternative by decoding it alone, and SentencePiece reads the
    word-boundary space off the first token of whatever it decodes, so "▁world"
    and "world" both record "world". A later vocabulary spelling that ID either
    way reproduces the recording exactly, while the branch after a retained
    "Hello" reads "Hello world" under one and "Helloworld" under the other. The
    recording is the same in both directions, so no comparison against the
    loaded vocabulary can establish which one offered the alternative, and the
    session that offered it is the only place it can be applied: its load
    identifier names the load in memory now.

    This costs nothing, because Replacement text expresses the same branch and
    is checked more strictly. ModelManager.encode_replacement validates typed
    text in place against the retained tokens, against exactly this ambiguity,
    and picks whichever ID spells it correctly in that position under the
    loaded vocabulary. Someone who wants the alternative "world" types "world"
    and gets a correctly spelled branch.
    """
    if not forkable_without_evidence(episode, manager):
        raise ValueError("A recorded alternative can only be applied in the session that offered it. The run "
                         "records how the alternative decodes on its own, which cannot say how the model that "
                         "offered it spelled that token after your retained tokens. Type the text you want in "
                         "Replacement text instead, which is checked in place against those tokens.")


def verify_visible_candidate(candidate_id, manager, stop_ids):
    """Refuse an alternative the loaded model never shows in a response.

    The replacement reaches the runtime as forced_ids past the literal prefill,
    where IncrementalDecoder.push drops a hidden special instead of decoding it.
    Choosing one would leave the response text exactly as it was while the token
    still entered the model's context, so the branch the panel advertised never
    appears and the edit reads as a no-op that quietly changed the run.

    A hidden stop token is the exception, because the runtime cuts a forced
    sequence at its first stop token past the literal prefill: the response ends
    there, which is a visible outcome even though the token itself never shows.

    This holds whatever the run's provenance, so it is asked before the
    provenance question: a hidden alternative is no more usable in the session
    that offered it than on an uploaded run.
    """
    if candidate_id in manager.hidden_token_ids and candidate_id not in stop_ids:
        raise ValueError("The loaded model does not show that token in a response, so choosing it would leave "
                         "the text unchanged while still feeding the token to the model. Type the branch you "
                         "want in Replacement text instead.")


def stop_deciding_id(turn):
    """The token whose membership in the stop set decides this turn's outcome.

    finish_turn reads the last sampled token, and for an edited response whose
    replacement ended it with nothing sampled afterwards, the last token of the
    response.
    """
    sampled = turn["metrics"][turn["forced_prefix_tokens"]:]
    if sampled:
        return sampled[-1]["token_id"]
    if turn.get("token_edit") and turn["metrics"]:
        return turn["metrics"][-1]["token_id"]
    return None


def verify_recorded_stops(episode, turn_index, kept, literal_prefill_tokens, stop_ids):
    """Refuse a fork whose replayed tokens no longer behave as they did under the stop set.

    Each earlier response is replayed through finish_turn against the stop set
    of the load in memory now. An ID that decodes to the same text can still
    have stopped being configured as a stop token, and then a response that
    ended naturally is read as a length failure: its tool call is never parsed,
    so its messages, event and maze position never reach the forked episode and
    regeneration starts from the wrong state.

    The retained prefix of the edited response is checked too. It reaches the
    runtime as forced_ids, which cuts the forced sequence at the first stop
    token past the literal prefill, so an ID this load newly treats as a stop
    token ends the response inside the prefix and the selected token is never
    reached. kept stops before the edited token, so the response's own closing
    stop token is not part of it and any stop ID found there is one this load
    added.
    """
    for turn in episode.turns[:turn_index]:
        last = stop_deciding_id(turn)
        if (last is not None and last in stop_ids) != (turn.get("finish_reason") == "stop"):
            raise ValueError("The loaded model's stop tokens differ from the ones that produced this run, so "
                             "its earlier responses cannot be reconstructed. This happens when the same model "
                             "ID has been re-downloaded at a different revision; load that snapshot to fork "
                             "this run.")
    if any(metric["token_id"] in stop_ids for metric in kept[literal_prefill_tokens:]):
        raise ValueError("The loaded model treats one of the tokens kept before your edit as a stop token, "
                         "so it would end this response inside the retained prefix and never reach the token "
                         "you selected. This happens when the same model ID has been re-downloaded at a "
                         "different revision; load that snapshot to fork this run.")


def fork_token_edit(episode, turn_index, token_index, replacement, manager, *, candidate_id=None):
    """Fork before one response; replay exact earlier IDs plus a replacement.

    token_index addresses the full response metric list, including any supplied
    prefix. Only generated tokens are editable. The original remains untouched,
    including a run uploaded for replay: forking it rebuilds the maze, history
    and counters into a new live episode rather than reopening the saved one.
    """
    with episode.lock:
        if episode.busy:
            raise ValueError("Pause or stop the episode before editing tokens.")
        # The fork replays stored token IDs, so the tokenizer has to match. The
        # model ID is the cheap gate; a later load of the same ID is allowed,
        # which is what lets an uploaded run be forked at all, but only after
        # the run's own recorded text and stop outcomes confirm that this load
        # tokenizes it the same way.
        if episode.model_id and manager.model_id != episode.model_id:
            raise ValueError(f"This run was generated by {episode.model_id}. "
                             "Load that model before editing its tokens.")
        if not isinstance(turn_index, int) or not 0 <= turn_index < len(episode.turns):
            raise ValueError("Select a response and token to edit.")
        original = episode.turns[turn_index]
        metrics = original["metrics"]
        if (not isinstance(token_index, int)
                or not original["forced_prefix_tokens"] <= token_index < len(metrics)):
            raise ValueError("Select a model-generated token to edit.")
        kept = metrics[:token_index]
        kept_ids = [m["token_id"] for m in kept]
        stop_ids = manager.stop_token_ids
        literal_prefill_tokens = literal_prefill_of(original)
        verify_recorded_text(episode, turn_index, manager)
        verify_recorded_stops(episode, turn_index, kept, literal_prefill_tokens, stop_ids)
        if candidate_id is None:
            replacement_ids = manager.encode_replacement(
                kept_ids, replacement, literal_prefill_tokens=literal_prefill_tokens,
            )
        else:
            # The alternative has to be one the run recorded for this token,
            # which is a correctness check on the selection rather than a
            # question about the tokenizer, so it holds whatever the provenance.
            if not any(c["token_id"] == candidate_id
                       for c in metrics[token_index].get("top_candidates", [])):
                raise ValueError("Choose an alternative for the selected token.")
            verify_visible_candidate(candidate_id, manager, stop_ids)
            verify_recorded_candidate(episode, manager)
            replacement_ids = [candidate_id]
        if not replacement_ids:
            raise ValueError("Enter replacement text or choose a token alternative.")
        if any(t in stop_ids for t in replacement_ids[:-1]):
            raise ValueError("A stop token can only appear at the end of the replacement.")
        # The fork rebuilds the run one response at a time, so the closures it
        # keeps are replayed at their own boundaries below rather than being in
        # force from the start. Closures after the edited response are left
        # behind with the responses that followed them.
        carried, config = [], copy.deepcopy(episode.config)
        if episode.map_changes:
            carried = [u for u in config.get("map_updates", []) if u["before_turn"] <= turn_index]
            config["map_updates"] = []
        result = Episode(episode.maze, config)
        # The fork continues under the weights in memory now, not the ones that
        # produced the original; token_edit keeps the original stamp.
        result.model_id, result.load_id = manager.model_id, manager.load_id
        result.manual_intervention = True
        # Closures the map refused are carried on the same rule as the ones it
        # accepted. The fork keeps the responses that were generated after a
        # failed intervention, so a fork reporting none would say those
        # responses ran under a map nobody had asked to change.
        result.dropped_closures = copy.deepcopy(
            [d for d in episode.dropped_closures if d["before_turn"] <= turn_index])
        # Rebuild history and recovery counters through the same simulator path
        # used during generation, excluding the edited response and its future.
        for i, previous in enumerate(episode.turns[:turn_index]):
            if carried:
                result.config["map_updates"].extend(u for u in carried if u["before_turn"] == i)
            turn = copy.deepcopy(previous)
            result.turns.append(turn)
            if episode.interrupted and episode.intervention_turn == i:
                result.interrupted = True
                result.intervention_turn = i
                result.intervention_tokens = result.sampled_tokens
                result.intervention_attempts = result.tool_attempts
            result.phase = "running"
            finish_turn(result, turn, stop_ids, result.config["per_turn_tokens"])
        if carried:
            result.config["map_updates"].extend(u for u in carried if u["before_turn"] == turn_index)
        prefix = kept_ids + replacement_ids
        result.token_edit = dict(parent_run_id=episode.run_id, turn=turn_index,
                                 token_index=token_index, original_token_id=metrics[token_index]["token_id"],
                                 replacement_ids=replacement_ids,
                                 replacement_text=replacement if candidate_id is None else manager.decode(replacement_ids),
                                 parent_model_id=episode.model_id, parent_load_id=episode.load_id,
                                 parent_replay=episode.replay_only, created_at=time.time())
        result.pending_edit = dict(forced_ids=prefix,
                                   literal_prefill_tokens=literal_prefill_tokens,
                                   interruption_here=bool(episode.interrupted and episode.intervention_turn == turn_index))
        result.phase, result.detail = "paused", "Token edit prepared. Regeneration will replace this response and its later moves in a new run."
        return result


def finish_turn(episode, turn, stop_ids, max_tokens):
    sampled = turn["metrics"][turn["forced_prefix_tokens"]:]
    episode.sampled_tokens += len(sampled)
    turn["sampled_tokens"] = len(sampled)
    turn["tokens_cumulative"] = episode.sampled_tokens
    natural_stop = bool(sampled and sampled[-1]["token_id"] in stop_ids)
    if turn.get("token_edit") and not sampled and turn["metrics"]:
        natural_stop = turn["metrics"][-1]["token_id"] in stop_ids
    if episode.stop_requested:
        episode.phase, episode.detail = "stopped", "Stopped by you. Partial tokens were retained; no partial action executed."
        turn["finish_reason"] = "user_stopped"
        return
    if not natural_stop:
        episode.phase, episode.detail = "budget", "The response reached its generation limit. No unfinished action executed."
        turn["finish_reason"] = "length" if len(sampled) >= max_tokens else "incomplete_stream"
        return
    turn["finish_reason"] = "stop"
    text = turn["text"]
    if turn.get("reasoning_prefilled"):
        text = "<think>" + text
    assistant_content = text
    # Tool-looking text inside reasoning is not an external action.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    args, error = parse_call(text)
    if args is None and error is None:
        episode.phase, episode.detail = "abandoned", "The model ended its response without making a movement call."
        turn["outcome"] = "no_call"
        return
    episode.tool_attempts += 1
    if error:
        event = {"accepted": False, "before": list(episode.position), "after": list(episode.position),
                 "error": error, "arrived": False, "progress": False}
    else:
        event = apply_call(episode.current_maze, episode.position, args, goal_mode=episode.config["goal_mode"])
    event.update(source="model", turn=len(episode.turns) - 1)
    episode.events.append(event)
    turn["event"] = event
    if event["accepted"]:
        episode.position = tuple(event["after"])
        if episode.interrupted and episode.resumed is None:
            episode.resumed = True
            episode.first_move_progress = event["progress"]
            episode.latency = episode.sampled_tokens - episode.intervention_tokens
        episode.detail = f"Moved {event['direction']} to row {episode.position[0]}, column {episode.position[1]}."
    else:
        episode.detail = "Rejected call: " + event["error"].replace("_", " ") + ". The position did not change."
    episode.messages.extend([
        {"role": "assistant", "content": assistant_content},
        {"role": "tool", "content": json.dumps(episode.model_state(event["error"]), separators=(",", ":"))},
    ])
    if event["arrived"]:
        episode.phase, episode.detail = "arrived", "The simulator confirmed arrival at the destination."
    elif (episode.interrupted and episode.resumed is None
          and (episode.sampled_tokens - episode.intervention_tokens >= episode.config["recovery_tokens"]
               or episode.tool_attempts - episode.intervention_attempts >= episode.config["recovery_attempts"])):
        episode.phase, episode.detail = "budget", "No accepted move within the recovery window."
    elif episode.sampled_tokens >= episode.config["token_budget"] or episode.tool_attempts >= episode.config["attempt_budget"]:
        episode.phase, episode.detail = "budget", "The episode reached its token or action limit."


def stream_episode(episode, models, *, single_step=False, save_dir=None):
    with episode.lock:
        if episode.busy:
            raise ValueError("This episode is already generating. Pause it before changing the run.")
        if episode.phase in TERMINAL or episode.replay_only:
            raise ValueError("Start a new episode to run again. This episode is finished or is a saved replay. Use Play or Next to inspect its recorded responses.")
        manager = models.open_session()
        episode.busy = True
        episode.pause_requested = episode.stop_requested = False
        episode.phase = "running"
        episode.model_id = episode.model_id or manager.model_id
        episode.load_id = episode.load_id or manager.load_id
    logger.info("Run %s generating with %s: %s responses so far, %s sampled tokens, %s moves, %s",
                episode.run_id, episode.model_id, len(episode.turns), episode.sampled_tokens,
                episode.moves, "one response" if single_step else "until it ends")
    turn = None
    autosave_error = None

    def record(turn):
        """One line for a response, wherever that response was finalized.

        A turn is completed on three paths: the ordinary one, a stop caught
        between its opening frame and generation, and the cleanup that closes
        an unfinished turn after a failure or a viewer hanging up. The outcome
        line below counts all three as responses, so recording only the first
        left a reader the two cases this file is opened for - a stop and a
        crash - with nothing between the run starting and its summary.

        The duration is set here rather than read, because the two abnormal
        paths never had one, and a turn saved without it reads as a response
        that took no time rather than one nobody timed.
        """
        turn.setdefault("seconds", time.time() - turn["started_at"])
        logger.info("Run %s response %s: %s after %s sampled tokens in %.1fs. %s",
                    episode.run_id, len(episode.turns), turn["finish_reason"],
                    turn.get("sampled_tokens", 0), turn["seconds"], episode.detail)

    def autosave():
        nonlocal autosave_error
        if not save_dir or autosave_error is not None:
            return
        try:
            episode.save(save_dir)
        except OSError as exc:
            # Storage is separate from the model outcome. Stop between turns,
            # retain the in-memory run, and never retry this failure in cleanup.
            autosave_error = str(exc)
            episode.pause_requested = True

    try:
        while episode.phase == "running":
            if episode.stop_requested:
                episode.phase, episode.detail = "stopped", "Stopped by you before the next response."
                break
            if episode.pause_requested:
                episode.phase = "paused"
                episode.detail += " Paused before the next response."
                break
            if manager.load_id != episode.load_id:
                raise ValueError("The model changed during this episode. Start a new episode with the selected model.")
            # Before the response is generated, so its call is judged against
            # the map as changed. The response's own prompt still shows the map
            # it was given, because the history is already written: the model
            # meets the change in the simulator's reply to whatever it does
            # next, which carries the current grid whether the call was
            # accepted or refused.
            apply_closure(episode)
            edit = episode.pending_edit
            forced = edit["forced_ids"] if edit else interrupted_prefix(episode, manager)
            inserts_interruption = edit["interruption_here"] if edit else bool(forced)
            limit = min(episode.config["per_turn_tokens"], episode.config["token_budget"] - episode.sampled_tokens)
            window = episode.config["recovery_tokens"]
            if inserts_interruption:
                limit = min(limit, window)
            elif episode.interrupted and episode.resumed is None:
                limit = min(limit, window - (episode.sampled_tokens - episode.intervention_tokens))
            if limit <= 0:
                episode.phase, episode.detail = "budget", "The sampled-token budget is exhausted."
                break
            turn = {"text": "", "metrics": [], "prompt_ids": [], "forced_prefix_tokens": 0,
                    "prefix_ids": [], "prefix_text": "",
                    "planned_prefix_ids": forced, "planned_prefix_text": manager.decode(forced),
                    "position_before": list(episode.position), "started_at": time.time(), "finish_reason": None}
            turn["literal_prefill_tokens"] = edit["literal_prefill_tokens"] if edit else len(forced)
            if edit:
                turn["token_edit"] = copy.deepcopy(episode.token_edit)
            episode.turns.append(turn)
            episode.pending_edit = None
            yield episode
            if episode.stop_requested:
                finish_turn(episode, turn, set(), limit)
                record(turn)
                break
            stop_ids = manager.stop_token_ids
            generator = manager.generate(
                episode.messages, temperature=episode.config["temperature"], top_p=1., top_k=0,
                max_new_tokens=limit, seed=episode.config["sampling_seed"] + 100003 * (len(episode.turns) - 1),
                analyze_prompt=False, tools=TOOLS, forced_ids=forced, literal_prefill_tokens=turn["literal_prefill_tokens"],
            )
            try:
                for update in generator:
                    turn.update(text=update.text, metrics=copy.deepcopy(update.metrics), prompt_ids=list(update.prompt_ids),
                                forced_prefix_tokens=update.forced_prefix_tokens, reasoning_prefilled=update.reasoning_prefilled,
                                load_id=update.load_id, model_id=update.model_id)
                    # The runtime emits prefix metrics only after prefill has
                    # consumed them. An opening frame or a failed model call
                    # alone is not evidence that an interruption was inserted.
                    if forced and update.forced_prefix_tokens and update.metrics:
                        turn.update(prefix_ids=forced, prefix_text=manager.decode(forced))
                    if inserts_interruption and not episode.interrupted and update.forced_prefix_tokens and update.metrics:
                        episode.interrupted = True
                        episode.intervention_turn = len(episode.turns) - 1
                        episode.intervention_tokens = episode.sampled_tokens
                        episode.intervention_attempts = episode.tool_attempts
                        episode.detail = "Interruption inserted. Watching for a real movement call."
                    yield episode
                    if episode.stop_requested:
                        break
            finally:
                generator.close()
            turn["seconds"] = time.time() - turn["started_at"]
            finish_turn(episode, turn, stop_ids, limit)
            record(turn)
            if episode.phase in TERMINAL and episode.interrupted and episode.resumed is None:
                episode.resumed = False if episode.phase not in ("stopped", "error") else None
                episode.first_move_progress = False if episode.resumed is False else None
            autosave()
            yield episode
            if episode.phase in TERMINAL:
                break
            if episode.stop_requested:
                continue
            if single_step or episode.pause_requested:
                episode.phase = "paused"
                episode.detail += " Paused before the next response."
                break
            time.sleep(.25)
    except GeneratorExit:
        if episode.phase == "running":
            episode.phase, episode.detail = "stopped", "Viewer stopped streaming. The partial response was retained."
        raise
    except Exception as exc:
        # The panel says this too, but the panel is gone by the time anyone
        # asks, and a failure inside generation is what a log read after a
        # memory kill is looking for.
        logger.exception("Run %s failed while generating", episode.run_id)
        episode.phase, episode.detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if turn is not None and turn.get("finish_reason") is None:
            count = max(0, len(turn["metrics"]) - turn["forced_prefix_tokens"])
            episode.sampled_tokens += count
            turn.update(sampled_tokens=count, tokens_cumulative=episode.sampled_tokens, finish_reason=episode.phase)
            # Only a turn no other path finalized reaches here, so this cannot
            # write a second line for a response already recorded.
            record(turn)
        with episode.lock:
            episode.busy = False
            # Before the autosave, so the file records the closure this run
            # will now never reach rather than a queue it emptied silently.
            if episode.phase in TERMINAL:
                abandon_closure(episode)
            manager.close()
            autosave()
            if autosave_error is not None:
                episode.warn_autosave(autosave_error)
        logger.info("Run %s %s after %s responses and %s sampled tokens, %s moves: %s",
                    episode.run_id, episode.phase, len(episode.turns), episode.sampled_tokens,
                    episode.moves, episode.detail)
    yield episode


def from_payload(data):
    if not isinstance(data, dict) or data.get("format") not in (FORMAT, CHANGING_FORMAT):
        raise ValueError("Choose a ChatLab maze run JSON file.")
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    if not isinstance(data.get("maze"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("The run is missing its map or its configuration.")
    changing = data["format"] == CHANGING_FORMAT
    # A fixed-map run carrying closures would replay as the map it started
    # from, which is not the map its responses were answering.
    if not changing and (data["config"].get("map_updates") or data.get("dropped_closures")
                         or data.get("close_next")):
        raise ValueError(f"A run whose map changes has to be recorded as {CHANGING_FORMAT}.")
    maze = load_maze(data["maze"]) if changing else Maze.from_dict(data["maze"])
    result = Episode(maze, data["config"])
    allowed = result.payload().keys() - {"format", "maze", "config", "exploratory", "tokenizer_note"}
    for key in allowed:
        if key in data:
            setattr(result, key, data[key])
    if changing:
        updates = result.config.get("map_updates", [])
        if not isinstance(updates, list):
            raise ValueError("A run's map changes must be a list.")
        validate_updates(maze, updates, result.turns, result.events)
        # The closures a run reports dropping are read as provenance by Run
        # details and carried into every fork, so they are checked like the
        # ones it reports taking rather than taken as written.
        validate_drops(maze, result.dropped_closures, updates, result.turns)
        # A run that has ended clears its queue on the way out, so a finished
        # one still waiting to close a cell is a state no run reaches.
        if result.close_next and result.phase in TERMINAL:
            raise ValueError("A run that has ended cannot still be waiting to close a cell.")
        validate_pending(maze, result.close_next, updates)
    # Reconstruct the visible path from real transitions, never trust claimed positions.
    position = maze.start
    for event in result.events:
        if tuple(event["before"]) != position:
            raise ValueError("The saved path contains a position mismatch.")
        if event["accepted"]:
            mode = result.config["goal_mode"]
            current = maze_at_turn(maze, result.config.get("map_updates", []), event.get("turn", -1)) if changing else maze
            actual = apply_call(current, position, {"maze_id": current.tool_id(mode), "direction": event["direction"]}, goal_mode=mode)
            if not actual["accepted"] or actual["after"] != event["after"]:
                raise ValueError("The saved path contains an invalid transition.")
            position = tuple(event["after"])
        elif tuple(event["after"]) != position:
            raise ValueError("A rejected action changed the saved position.")
    if tuple(result.position) != position:
        raise ValueError("The saved final position does not match its path.")
    result.position = position
    result.replay_only = True
    return result
