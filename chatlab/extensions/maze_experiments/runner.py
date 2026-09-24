"""Single-episode controller: token generation is separate from authoritative maze movement."""
from __future__ import annotations

import copy
import json
import logging
import re
import threading
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .dynamic_maze import (FORMAT as CHANGING_FORMAT, ChangingMaze, check_closure, close_cell, load_maze,
                           maze_at_turn, validate_drops, validate_pending, validate_updates)
from .inserts import CHANNELS, FORMAT as INSERT_FORMAT, check_insert, render_insert
from .maze import SYSTEM, Maze, TOOLS, apply_call, default_instruction, initial_history, parse_call
from chatlab.extension_api import normalize_steering, write_private_text

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


def reachable_before_arriving(maze, origin):
    """The cells a character at ``origin`` can walk to without arriving on the way.

    Arriving at the destination ends the run, so a cell whose every route
    passes through the destination is one the run can never reach, however
    open the map around it. The destination itself is left out for the same
    reason.
    """
    origin = tuple(origin)
    found, todo = {origin}, deque([origin])
    while todo:
        for cell in maze.neighbors(todo.popleft()).values():
            if cell not in found and cell != maze.goal:
                found.add(cell)
                todo.append(cell)
    return found


def checked_cell(value, maze, label):
    """One cell of ``maze`` the run can reach before arriving, as a list, or None."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError(f"The {label} names one cell as a row and a column.")
    if not maze.open(value):
        raise ValueError(f"The {label} must be an open cell inside the maze.")
    if tuple(value) == maze.goal:
        raise ValueError(f"The {label} cannot be the destination, because arriving there ends the run.")
    if tuple(value) not in reachable_before_arriving(maze, maze.start):
        raise ValueError(f"The {label} has to be reachable from the start without passing through the "
                         "destination, because arriving there ends the run.")
    return list(value)


def check_checkpoint(config, maze):
    """Validate a run's waypoint and steering trigger, normalizing them in place.

    A waypoint is a cell the run asks the model to pass through on its way.
    Steering adds an activation vector to the responses it covers: it starts
    with the first response generated once its trigger holds - the character
    standing on ``steer_when["cell"]``, or ``steer_when["moves"]`` accepted
    moves made, supplied ones included, as the interruption counts them - and
    covers ``steer_responses`` responses from there, 0 meaning every response
    to the end of the run. It starts once: a character that leaves the cell and
    comes back is not steered again. A run carries its vector whole rather
    than a reference into this machine's vector store, so an export reads
    back anywhere.
    """
    waypoint = checked_cell(config.get("waypoint"), maze, "waypoint")
    if waypoint is not None and tuple(waypoint) == maze.start:
        raise ValueError("The waypoint cannot be the start, because the run would begin having reached it.")
    if "waypoint" in config:
        config["waypoint"] = waypoint
    if config.get("steering") is None:
        if config.get("steer_when") is not None or config.get("steer_responses"):
            raise ValueError("A steering trigger needs a steering vector.")
        return
    vector = normalize_steering(config["steering"])
    if "vector" not in vector:
        raise ValueError("A maze run carries its steering vector whole. Import the vector file itself.")
    when = config.get("steer_when")
    if not isinstance(when, dict) or len(when) != 1 or set(when) - {"cell", "moves"}:
        raise ValueError("Say when steering starts: at a cell, or after a number of accepted moves.")
    if "cell" in when:
        # checked_cell reads None as no cell, which is right for an optional
        # waypoint and wrong here: a trigger at no cell would never fire, and
        # the run would read as steered while nothing steered it.
        if when["cell"] is None:
            raise ValueError("Name the cell steering starts at.")
        when = {"cell": checked_cell(when["cell"], maze, "steering cell")}
    elif type(when["moves"]) is not int or not 0 <= when["moves"] <= 255:
        raise ValueError("Steering after moves needs a whole number of moves from 0 to 255.")
    responses = config.get("steer_responses", 0)
    if type(responses) is not int or not 0 <= responses <= 256:
        raise ValueError("Steered responses must be a whole number from 0 to 256; 0 steers to the end of the run.")
    config.update(steering=vector, steer_when=dict(when), steer_responses=responses)


def steering_active(config):
    vector = config.get("steering")
    return bool(vector and vector.get("enabled", True) and vector.get("strength", 1) != 0)


def steered_at(config, start, index, position, moves):
    """Whether response ``index`` is steered, given the run as it stands before it.

    ``start`` is the first steered response so far, or None. Written as one
    function over the recorded quantities so generation, a fork rebuilding its
    earlier responses, and a saved run being read back all decide it the same way.
    """
    if not steering_active(config):
        return False
    if start is None:
        when = config["steer_when"]
        return list(position) == when["cell"] if "cell" in when else moves >= when["moves"]
    count = config["steer_responses"]
    return count == 0 or index - start < count


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
    # A message queued to go into the context before the next response. Not
    # written to the run: one still waiting when the episode ends is dropped.
    insert_next: dict | None = None
    pause_requested: bool = False
    stop_requested: bool = False
    # Why the last write of this run failed, or None once it is on disk. Kept
    # apart from the detail, which is prose for a reader, so a caller that has
    # to know whether the file exists does not have to read it back.
    autosave_error: str | None = None
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
            setattr(result, key, threading.RLock() if key == "lock" else copy.deepcopy(value, memo))
        if result.phase == "ready" and not result.turns:
            result.run_id = uuid4().hex
            result.created_at = time.time()
        return result

    def __post_init__(self):
        # Reentrant, because a run is written down on two paths: one that
        # already holds the lock, as stopping and autosaving do, and one that
        # does not, as the export button does. Both have to read the run as it
        # stands at one moment rather than field by field.
        self.lock = threading.RLock()
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
        check_checkpoint(self.config, self.maze)
        supplied = int(self.config.get("supplied_moves", 3))
        self.messages, self.events, self.position = initial_history(
            self.maze, supplied, goal_mode=self.config["goal_mode"], goal_hint=self.config["goal_hint"],
            system=self.config["system_prompt"], instruction=self.config["instruction"],
            waypoint=self.config.get("waypoint"))
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
                                       goal_hint=self.config["goal_hint"], waypoint=self.config.get("waypoint"),
                                       waypoint_reached=self.waypoint_turn is not None)

    @property
    def waypoint_turn(self):
        """The response whose accepted move reached the waypoint, -1 for a supplied move, or None.

        Read off the path rather than kept, so it cannot disagree with the
        moves a saved run is checked against.
        """
        waypoint = self.config.get("waypoint")
        if waypoint is None:
            return None
        for event in self.events:
            if event["accepted"] and list(event["after"]) == list(waypoint):
                return event.get("turn", -1) if event.get("source") == "model" else -1
        return None

    @property
    def steer_turn(self):
        """The first steered response, or None while steering has not started."""
        return next((i for i, turn in enumerate(self.turns) if turn.get("steered")), None)

    def steers_next(self):
        """Whether the response about to be generated is steered."""
        return steered_at(self.config, self.steer_turn, len(self.turns), self.position, self.moves)

    @property
    def moves(self):
        return sum(e["accepted"] for e in self.events)

    def payload(self):
        keys = ("run_id", "phase", "detail", "messages", "events", "turns", "position", "model_id", "load_id",
                "sampled_tokens", "tool_attempts", "supplied_moves", "interrupted", "intervention_turn",
                "intervention_tokens", "intervention_attempts", "resumed", "first_move_progress", "latency",
                "manual_intervention", "created_at", "token_edit", "pending_edit", "dropped_closures",
                "close_next")
        # One reading, not one per field. Queueing a closure marks the run as
        # intervened in and fills its queue together, and the close button runs
        # off Gradio's queue, so a snapshot taken field by field could catch the
        # two apart and write a run carrying an intervention while saying none
        # was made.
        with self.lock:
            # A run carrying an insertion is written under a format an older
            # ChatLab refuses, rather than one it would read with every later
            # response's context shifted.
            kind = INSERT_FORMAT if self.config.get("context_inserts") else CHANGING_FORMAT if self.map_changes else FORMAT
            return {"format": kind, "maze": self.maze.to_dict(), "config": self.config,
                    "exploratory": True, "tokenizer_note": "Every turn records its actual prompt IDs. Later turns are templated from the complete prior response text, including reasoning.",
                    **{k: getattr(self, k) for k in keys}}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        temp = path.with_suffix(".json.tmp")
        # Rendered under the lock, not merely read under it: the reading hands
        # back the run's own lists, so a stop landing between the reading and
        # the rendering would write a closure as pending and dropped at once.
        # The file itself is written outside, where only the disk is slow.
        with self.lock:
            text = json.dumps(self.payload(), ensure_ascii=False, allow_nan=False)
        write_private_text(temp, text)
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
            abandon_insert(self)
            if save_dir:
                try:
                    self.save(save_dir)
                    self.autosave_error = None
                except OSError as exc:
                    self.warn_autosave(str(exc))

    def warn_autosave(self, error):
        logger.warning("Autosave of run %s failed: %s", self.run_id, error)
        self.autosave_error = error
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
            # One closure to a response. A fork of the response after a closure
            # starts holding that closure at the boundary it is about to
            # regenerate, and a run that wrote two there could not be read back.
            boundary = len(self.turns)
            if any(record["before_turn"] == boundary
                   for record in (*self.config.get("map_updates", ()), *self.dropped_closures)):
                raise ValueError("The map already changed before this response. Generate it before closing another cell.")
            check_checkpoint_closure(self, cell, check_closure(self.current_maze, self.position, cell))
            self.close_next, self.manual_intervention = tuple(cell), True

    def request_insert(self, channel, text, sender=None, advised_direction=None):
        """Queue one message to go into the context before the next generated response.

        The rules are the closure's. One is queued at a time, and a second
        request is refused naming the first, which the reader has already been
        told will land. One lands at a boundary, so a fork carrying an
        insertion at the boundary it is about to regenerate refuses another.
        The check and the assignment are one operation under the lock, because
        the button runs off Gradio's queue and two clicks arrive at once.
        """
        with self.lock:
            if self.phase in TERMINAL or self.replay_only:
                raise ValueError("Start a new episode to insert a message. This episode is finished or is a saved replay.")
            if self.insert_next:
                queued = self.insert_next
                raise ValueError(f"{describe_insert(queued)} is already queued before the next response. "
                                 "Let it land before queueing another.")
            boundary = len(self.turns)
            if any(record["before_turn"] == boundary for record in self.config.get("context_inserts", ())):
                raise ValueError("A message was already inserted before this response. Generate it before inserting another.")
            insert = check_insert(dict(channel=channel, text=text, sender=sender or None,
                                       advised_direction=advised_direction or None))
            # A response being generated now will be answered by the
            # simulator before this lands, so only an idle run is asked.
            if not self.busy:
                render_insert(self.messages, insert)
            self.insert_next, self.manual_intervention = insert, True


def describe_insert(insert):
    """An insertion named for a reader: its channel, its sender, and how it begins."""
    text = insert["text"] if len(insert["text"]) <= 40 else insert["text"][:39] + "…"
    sender = f" from {insert['sender']}" if insert.get("sender") else ""
    return f"The {CHANNELS[insert['channel']].lower()}{sender} “{text}”"


def land_insert(episode, insert):
    """Write one insertion into the history and the record together. The caller holds the lock."""
    episode.messages = render_insert(episode.messages, insert)
    episode.config.setdefault("context_inserts", []).append(insert)


def apply_insert(episode, manager=None):
    """Put the queued message into the context, if one is queued, and record where it landed.

    Called with a response about to be appended. Returns the history as it
    stood before, so the stream can withdraw the message if that response
    never reaches the model: a message is only kept on the record once a
    prompt holding it has been fed, as an interruption is only marked once
    its tokens have been. One that can no longer be rendered is dropped with
    a note in the detail rather than recorded, and None comes back, as is
    one ``manager``'s tokenizer reads a special token out of: templates
    tokenize message text with the specials parsed, so such a message would
    end or open a turn wherever it sits, whatever spelling the fixed list in
    inserts.py missed.
    """
    with episode.lock:
        if not episode.insert_next:
            return None
        before = episode.messages
        queued, episode.insert_next = episode.insert_next, None
        insert = dict(before_turn=len(episode.turns), channel=queued["channel"], text=queued["text"],
                      sender=queued["sender"], position=list(episode.position),
                      advised_direction=queued["advised_direction"])
        try:
            if manager is not None and any(set(manager.encode(value)) & manager.hidden_token_ids
                                           for value in (insert["text"], insert["sender"] or "")):
                raise ValueError("The loaded model reads part of the text or the sender as one of its special tokens.")
            land_insert(episode, insert)
        except ValueError as exc:
            episode.detail = f"{describe_insert(insert)} was dropped. {exc}"
            logger.warning("Run %s dropped the message queued before response %s: %s",
                           episode.run_id, insert["before_turn"] + 1, exc)
            return None
        episode.detail = f"{describe_insert(insert)} went into the context before this response."
        logger.info("Run %s inserted a %s before response %s at %s", episode.run_id, insert["channel"],
                    insert["before_turn"] + 1, insert["position"])
        return before


def withdraw_insert(episode, before):
    """Take back the message the last apply_insert landed, its response never having been generated.

    A stop at the opening frame, or a model call that fails before its first
    update, leaves a response the model never read a prompt for, so a record
    saying that response received the message would describe an
    intervention nobody made.
    """
    with episode.lock:
        withdrawn = episode.config["context_inserts"].pop()
        if not episode.config["context_inserts"]:
            del episode.config["context_inserts"]
        episode.messages = before
    logger.warning("Run %s withdrew the %s before response %s: that response was never generated",
                   episode.run_id, withdrawn["channel"], withdrawn["before_turn"] + 1)


def abandon_insert(episode):
    """Drop a queued message the run will never reach. The caller holds the lock."""
    if episode.insert_next:
        logger.warning("Run %s ended %s with a %s still queued; it was dropped", episode.run_id,
                       episode.phase, episode.insert_next["channel"])
        episode.insert_next = None


def check_checkpoint_closure(episode, cell, changed, boundary=None, position=None):
    """Refuse a closure that takes away the waypoint or the steering cell.

    Neither is ever closed, so the board and the saved run always show the
    cell the run was set up around. Until the character has reached one, a
    closure that walls the character off from it is refused as well, since
    the run could no longer do what it was set up to test. A route left only
    through the destination counts as walled off, because arriving ends the
    run before the character gets there.

    ``boundary`` and ``position`` default to the run as it stands, which is
    where a live closure lands. A saved run's closures are asked the same
    question at the boundary and position each one records, so a file cannot
    carry a closure the run itself would have refused.
    """
    config = episode.config
    boundary = len(episode.turns) if boundary is None else boundary
    position = episode.position if position is None else position
    reached, steered = episode.waypoint_turn, episode.steer_turn
    checkpoints = []
    if config.get("waypoint") is not None:
        checkpoints.append(("waypoint", config["waypoint"], reached is None or reached >= boundary))
    when = config.get("steer_when") or {}
    if "cell" in when:
        checkpoints.append(("steering cell", when["cell"],
                            steering_active(config) and (steered is None or steered >= boundary)))
    for label, point, _ in checkpoints:
        if tuple(point) == tuple(cell):
            raise ValueError(f"The {label} is never closed.")
    reachable = reachable_before_arriving(changed, position)
    for label, point, pending in checkpoints:
        if pending and tuple(point) not in reachable:
            raise ValueError(f"Closing this cell would cut the character off from the {label}.")


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
            check_checkpoint_closure(episode, cell, changed)
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
    initial block is the setup plus one such pair per supplied move. A user
    message inserted at or before a response's boundary adds one more; a note
    or a teammate message is written into a reply already counted.

    Counted from the record rather than kept as an index of its own, so the
    two cannot disagree.
    """
    supplied = sum(event["source"] == "supplied" for event in episode.events)
    attempts = sum("event" in turn for turn in episode.turns[:max(index, 0)])
    users = sum(insert["channel"] == "user" and insert["before_turn"] <= index
                for insert in episode.config.get("context_inserts", ()))
    return episode.messages[:2 + 2 * (supplied + attempts) + users]


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


def verify_carried_inserts(episode, turn_index, manager):
    """Refuse a fork that would carry a message the loaded model cannot vouch for.

    An uploaded run is checked against its recorded prompts only where the
    model that recorded them was loaded at upload, and against special tokens
    only by the fixed list in inserts.py. A fork has that model in memory, so
    the messages it keeps are asked both questions a live run asks: whether
    this tokenizer reads a special token out of one, and whether every kept
    response from the first message on, the edited one included, recorded the
    prompt its history becomes under this template.
    """
    inserts = [i for i in episode.config.get("context_inserts", ()) if i["before_turn"] <= turn_index]
    if not inserts:
        return
    hidden = manager.hidden_token_ids
    for insert in inserts:
        if any(set(manager.encode(value)) & hidden for value in (insert["text"], insert.get("sender") or "")):
            raise ValueError(f"The loaded model reads a special token out of the message inserted before response "
                             f"{insert['before_turn'] + 1}, so it cannot be carried into a fork.")
    for index in range(inserts[0]["before_turn"], turn_index + 1):
        turn = episode.turns[index]
        if not turn.get("prompt_ids") or (turn.get("model_id") or episode.model_id) != manager.model_id:
            continue
        context = context_messages(episode, index)
        try:
            prompt = manager.decode(turn["prompt_ids"])
        except (IndexError, KeyError, OverflowError, TypeError, ValueError):
            prompt = None
        try:
            templated = manager.prompt_text(context, TOOLS)
        except Exception:
            templated = None
        if prompt is None or not (prompt == templated if templated is not None
                                  else prompt_holds(prompt, context, episode.messages[len(context):])):
            raise ValueError(f"Response {index + 1}'s recorded prompt is not the history the run records for it "
                             "with its inserted messages, so those messages cannot be carried into a fork.")


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
        verify_carried_inserts(episode, turn_index, manager)
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
        # Insertions follow the same rule, and one at exactly the edited
        # boundary stays: the edited response was generated after it.
        inserts = [i for i in config.pop("context_inserts", []) if i["before_turn"] <= turn_index]
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
            for insert in inserts:
                if insert["before_turn"] == i:
                    land_insert(result, insert)
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
        for insert in inserts:
            if insert["before_turn"] == turn_index:
                land_insert(result, insert)
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


def assistant_content(turn):
    """A response as the history carries it, with the reasoning a template opened for it restored."""
    return ("<think>" if turn.get("reasoning_prefilled") else "") + turn["text"]


def reply_messages(episode, turn, event):
    """The response and the simulator's reply to its call, as the next prompt carries them.

    ``episode`` stands where the call left it. Generation and the check of a
    saved run's history both write the pair through here.
    """
    return [{"role": "assistant", "content": assistant_content(turn)},
            {"role": "tool", "content": json.dumps(episode.model_state(event["error"]), separators=(",", ":"))}]


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
    text = assistant_content(turn)
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
    episode.messages.extend(reply_messages(episode, turn, event))
    if event["arrived"]:
        episode.phase, episode.detail = "arrived", "The simulator confirmed arrival at the destination."
    elif (episode.interrupted and episode.resumed is None
          and (episode.sampled_tokens - episode.intervention_tokens >= episode.config["recovery_tokens"]
               or episode.tool_attempts - episode.intervention_attempts >= episode.config["recovery_attempts"])):
        episode.phase, episode.detail = "budget", "No accepted move within the recovery window."
    elif episode.sampled_tokens >= episode.config["token_budget"] or episode.tool_attempts >= episode.config["attempt_budget"]:
        episode.phase, episode.detail = "budget", "The episode reached its token or action limit."


def stream_episode(episode, models, *, single_step=False, save_dir=None, session=None):
    """Generate the episode's responses, yielding it after each change.

    ``session`` is a model session the caller already holds and keeps: a batch
    of trials holds one for every episode it runs, so nothing else can take
    the model or load another between two trials. Without one, the episode
    opens its own session and closes it when it stops.
    """
    with episode.lock:
        if episode.busy:
            raise ValueError("This episode is already generating. Pause it before changing the run.")
        if episode.phase in TERMINAL or episode.replay_only:
            raise ValueError("Start a new episode to run again. This episode is finished or is a saved replay. Use Play or Next to inspect its recorded responses.")
        manager = session or models.open_session()
        # Steering can start several responses in, so a vector this load
        # cannot take is refused before the run starts rather than there.
        if steering_active(episode.config):
            try:
                manager.check_steering(episode.config["steering"])
            except BaseException:
                if session is None:
                    manager.close()
                raise
        episode.busy = True
        episode.pause_requested = episode.stop_requested = False
        episode.autosave_error = None
        episode.phase = "running"
        episode.model_id = episode.model_id or manager.model_id
        episode.load_id = episode.load_id or manager.load_id
    logger.info("Run %s generating with %s: %s responses so far, %s sampled tokens, %s moves, %s",
                episode.run_id, episode.model_id, len(episode.turns), episode.sampled_tokens,
                episode.moves, "one response" if single_step else "until it ends")
    turn = None
    inserted = None
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
        logger.info("Run %s response %s%s: %s after %s sampled tokens in %.1fs. %s",
                    episode.run_id, len(episode.turns), ", steered" if turn.get("steered") else "",
                    turn["finish_reason"], turn.get("sampled_tokens", 0), turn["seconds"], episode.detail)

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
            # After the budget is known to allow a response, so an insertion is
            # only ever recorded with the response that read it.
            inserted = apply_insert(episode, manager)
            turn = {"text": "", "metrics": [], "prompt_ids": [], "forced_prefix_tokens": 0,
                    "prefix_ids": [], "prefix_text": "",
                    "planned_prefix_ids": forced, "planned_prefix_text": manager.decode(forced),
                    "position_before": list(episode.position), "started_at": time.time(), "finish_reason": None}
            turn["literal_prefill_tokens"] = edit["literal_prefill_tokens"] if edit else len(forced)
            steered = episode.steers_next()
            if episode.config.get("steering") is not None:
                # Unmarked until generation is entered below. The flag says the
                # vector touched this response, and a Stop taken at the opening
                # frame ends the run before the vector is ever installed.
                turn["steered"] = False
                if steered and episode.steer_turn is None:
                    episode.detail = "Steering starts with this response."
            if edit:
                turn["token_edit"] = copy.deepcopy(episode.token_edit)
            episode.turns.append(turn)
            episode.pending_edit = None
            yield episode
            if episode.stop_requested:
                if inserted is not None:
                    withdraw_insert(episode, inserted)
                    inserted = None
                finish_turn(episode, turn, set(), limit)
                record(turn)
                break
            if steered:
                turn["steered"] = True
            stop_ids = manager.stop_token_ids
            generator = manager.generate(
                episode.messages, temperature=episode.config["temperature"], top_p=1., top_k=0,
                max_new_tokens=limit, seed=episode.config["sampling_seed"] + 100003 * (len(episode.turns) - 1),
                analyze_prompt=False, tools=TOOLS, forced_ids=forced, literal_prefill_tokens=turn["literal_prefill_tokens"],
                steering=episode.config["steering"] if steered else None,
            )
            try:
                for update in generator:
                    turn.update(text=update.text, metrics=copy.deepcopy(update.metrics), prompt_ids=list(update.prompt_ids),
                                forced_prefix_tokens=update.forced_prefix_tokens, reasoning_prefilled=update.reasoning_prefilled,
                                load_id=update.load_id, model_id=update.model_id)
                    # The prompt holding the message has been fed.
                    if update.prompt_ids:
                        inserted = None
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
            if inserted is not None:
                withdraw_insert(episode, inserted)
                inserted = None
            turn["seconds"] = time.time() - turn["started_at"]
            finish_turn(episode, turn, stop_ids, limit)
            record(turn)
            if episode.phase in TERMINAL and episode.interrupted and episode.resumed is None:
                episode.resumed = False if episode.phase not in ("stopped", "error") else None
                episode.first_move_progress = False if episode.resumed is False else None
            # Before this autosave rather than only in the cleanup below. A
            # response that ends the episode can be the one a closure was
            # queued during, and the file written here is the whole record if
            # the process is killed at the yield: a finished run still holding
            # a queue is one this file's own reader refuses.
            if episode.phase in TERMINAL:
                with episode.lock:
                    abandon_closure(episode)
                    abandon_insert(episode)
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
        # A model call that failed before feeding its prompt.
        if inserted is not None:
            withdraw_insert(episode, inserted)
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
                abandon_insert(episode)
            if session is None:
                manager.close()
            autosave()
            if autosave_error is not None:
                episode.warn_autosave(autosave_error)
        logger.info("Run %s %s after %s responses and %s sampled tokens, %s moves: %s",
                    episode.run_id, episode.phase, len(episode.turns), episode.sampled_tokens,
                    episode.moves, episode.detail)
    yield episode


def validate_steering(episode):
    """Refuse a saved run whose steered responses are not the ones its trigger picks.

    Each response's flag is read as provenance - which responses the vector
    touched - so it is decided again from the recorded path, and a run that
    marks a response the trigger would have left alone, or leaves unmarked one
    it would have steered, is refused rather than scored.

    One response the trigger picks may be unmarked: a run stopped at the
    opening frame of a response, before generation, never installed the
    vector. That response carries no metrics and a stop as its outcome, and
    is the run's last, because a stopped run cannot continue.
    """
    config = episode.config
    if config.get("steering") is None:
        if any("steered" in turn for turn in episode.turns):
            raise ValueError("A run without a steering vector cannot mark responses as steered.")
        return
    supplied = [e for e in episode.events if e["source"] == "supplied"]
    position = tuple(supplied[-1]["after"]) if supplied else episode.maze.start
    moves, start = len(supplied), None
    by_turn = {e["turn"]: e for e in episode.events if e["source"] == "model"}
    for index, turn in enumerate(episode.turns):
        expected = steered_at(config, start, index, position, moves)
        never_generated = (index == len(episode.turns) - 1 and not turn.get("metrics")
                           and turn.get("finish_reason") in ("user_stopped", "stopped"))
        if turn.get("steered", False) is not expected and not (expected and never_generated):
            raise ValueError(f"Response {index + 1} is recorded as {'steered' if turn.get('steered') else 'unsteered'}, "
                             "which is not what the run's steering trigger decides at that point in its path.")
        if expected and start is None:
            start = index
        event = by_turn.get(index)
        if event and event["accepted"]:
            position, moves = tuple(event["after"]), moves + 1


def validate_checkpoint_closures(episode):
    """Refuse a saved run whose map changes a live run would have refused.

    Each closure is checked by check_checkpoint_closure, at the boundary and
    position it records, against the map it produced. A run with a waypoint
    and no vector is checked too, since the waypoint rule does not depend on
    steering. Runs after validate_updates and validate_steering, so the
    closures, the path and the steered flags it reads from are ones the file
    has already been held to.
    """
    maze = episode.maze
    for update in episode.config.get("map_updates", ()):
        maze = close_cell(maze, update["closed_cell"])
        try:
            check_checkpoint_closure(episode, update["closed_cell"], maze, update["before_turn"], update["position"])
        except ValueError as exc:
            raise ValueError(f"The map change before response {update['before_turn'] + 1} is one the run "
                             f"could not have made. {exc}") from None


OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def insert_outcome(episode, insert):
    """What the run did after one insertion, read off its path and its map and never off the file.

    The advice is judged on the map as the response after it found it, from
    the cell the character stood on: whether the advised step is open, and
    whether it is on a shortest route to the destination. The move is the
    first accepted model move from that response on, and it followed the
    advice, went against it (the opposite direction), or took another.
    """
    boundary, advised = insert["before_turn"], insert.get("advised_direction")
    move = next((e for e in episode.events
                 if e["source"] == "model" and e["accepted"] and e["turn"] >= boundary), None)
    outcome = dict(advised=advised, move=move, legal=None, shortest=None, followed=None)
    if advised is None:
        return outcome
    maze = maze_at_turn(episode.maze, episode.config.get("map_updates", ()), boundary)
    position = tuple(insert["position"])
    step = maze.neighbors(position).get(advised)
    distances = maze.distances(maze.goal)
    outcome.update(legal=step is not None,
                   shortest=step is not None and distances.get(step) == distances[position] - 1)
    if move is not None:
        outcome["followed"] = ("followed" if move["direction"] == advised
                               else "against" if move["direction"] == OPPOSITE[advised] else "other")
    return outcome


def path_position(episode, boundary):
    """Where the run's own transitions put the character before response ``boundary``."""
    accepted = [e for e in episode.events if e["accepted"] and e.get("turn", -1) < boundary]
    return list(accepted[-1]["after"]) if accepted else list(episode.maze.start)


def validate_inserts(episode, read_prompt=None):
    """Refuse a saved run whose inserted messages are not the ones its history holds.

    Each insertion has to name a response the run reaches, one to a boundary
    and in order, meet its channel's rules, and record where the character's
    path had it standing. The history is then written again from the setup,
    the responses, the simulator's replies and the recorded insertions, the
    way generation wrote it, and has to be the saved ``messages`` exactly: a
    file whose record says one thing and whose history another would be read
    as the first and replayed as the second.

    The response each insertion landed before has to record the prompt it
    was fed, since a run only keeps a message once a prompt holding it has
    been. Every response from the first insertion on that records a prompt is
    then read. ``read_prompt(episode, turn, context)`` answers for a prompt with
    two readings, or None when the model that recorded it is not the one
    loaded: the recorded IDs decoded, and ``context`` - the messages the
    record gives that response - put through the same model's template, or
    None where the template cannot render them. The two have to be the same
    text, the whole prompt and not only the parts the record names, so a
    message nobody recorded cannot sit anywhere in it. Where the template
    gives no reading, the decoded prompt is held to :func:`prompt_holds`
    instead. Without a reader, a run uploads with no tokenizer at all.
    """
    inserts = episode.config.get("context_inserts")
    if not isinstance(inserts, list) or not inserts:
        raise ValueError(f"A {INSERT_FORMAT} run carries its inserted messages as a non-empty list.")
    previous = -1
    for insert in inserts:
        boundary = insert.get("before_turn") if isinstance(insert, dict) else None
        if type(boundary) is not int:
            raise ValueError("Each inserted message names the response it landed before by its index.")
        if not 0 <= boundary < len(episode.turns):
            raise ValueError(f"The message inserted before response {boundary + 1} names a response the run "
                             "never reached.")
        if boundary <= previous:
            raise ValueError(f"The message inserted before response {boundary + 1} is out of order or shares its "
                             "response with another. One message lands before a response, in order.")
        previous = boundary
        try:
            check_insert(insert)
        except ValueError as exc:
            raise ValueError(f"The message inserted before response {boundary + 1} is refused. {exc}") from None
        if insert.get("position") != path_position(episode, boundary):
            raise ValueError(f"The message inserted before response {boundary + 1} records a position the "
                             "run's path does not reach there.")
        if not episode.turns[boundary].get("prompt_ids"):
            raise ValueError(f"Response {boundary + 1} records no prompt, so nothing says the model read the "
                             "message inserted before it.")
    if not episode.manual_intervention:
        raise ValueError("A run carrying an inserted message cannot report that nobody intervened in it.")
    validate_history(episode)
    if read_prompt is None:
        return
    # Every response from the first message on, not only the one each landed
    # before: a message stays in every later context, and a later prompt
    # without it would credit its response to an intervention it never read.
    for index in range(inserts[0]["before_turn"], len(episode.turns)):
        turn = episode.turns[index]
        if not turn.get("prompt_ids"):
            continue
        context = context_messages(episode, index)
        reading = read_prompt(episode, turn, context)
        if reading is None:
            continue
        prompt, templated = reading
        if not (prompt == templated if templated is not None
                else prompt_holds(prompt, context, episode.messages[len(context):])):
            raise ValueError(f"Response {index + 1}'s recorded prompt is not the history the run records "
                             "for it, with the messages inserted up to that response.")


def prompt_holds(prompt, context, following):
    """Whether a decoded prompt carries a response's context and nothing after it.

    The fallback for a template that gives no reading, so weaker than the
    comparison: text between the messages it finds is not read. Read without
    the template: every user and simulator message has to appear
    in the prompt as written and in order, and the next one the history holds
    after this context must not. Templates write those two roles verbatim,
    where some rewrite an earlier response's reasoning, so they are what can be
    found. Finding the messages in order pins an insertion to its own
    boundary: the same note given twice, or a user message whose words
    already sit in the system prompt, is only found where the order puts it.
    """
    position = 0
    for message in context:
        if message["role"] in ("user", "tool"):
            found = prompt.find(message["content"], position)
            if found < 0:
                return False
            position = found + len(message["content"])
    later = next((m["content"] for m in following if m["role"] in ("user", "tool")), None)
    return later is None or later not in prompt[position:]


def validate_history(episode):
    """Write a saved run's history again from its record and compare it with the saved one.

    Walks the responses in order on a fresh episode of the same scenario,
    landing each closure and each insertion at its own boundary and adding the
    pair each call left, and names the first response whose context disagrees.
    """
    config = copy.deepcopy(episode.config)
    config.pop("context_inserts")
    config["map_updates"] = []
    rebuilt = Episode(episode.maze, config)
    updates = episode.config.get("map_updates", [])
    by_boundary = {insert["before_turn"]: insert for insert in episode.config["context_inserts"]}
    by_turn = {e["turn"]: e for e in episode.events if e["source"] == "model"}
    saved = episode.messages
    if not isinstance(saved, list):
        raise ValueError("A run's messages must be a list.")
    for index, turn in enumerate(episode.turns):
        if ("event" in turn) != (index in by_turn):
            raise ValueError(f"Response {index + 1} disagrees with the run's path about whether it made a call.")
        rebuilt.config["map_updates"].extend(u for u in updates if u["before_turn"] == index)
        if index in by_boundary:
            try:
                rebuilt.messages = render_insert(rebuilt.messages, by_boundary[index])
            except ValueError as exc:
                raise ValueError(f"The message inserted before response {index + 1} cannot be placed. {exc}") from None
        if rebuilt.messages != saved[:len(rebuilt.messages)]:
            raise ValueError(f"The saved messages disagree with the run's record at response {index + 1}: its "
                             "context is not the history and the inserted messages the run records.")
        event = by_turn.get(index)
        if event is not None:
            rebuilt.events.append(event)
            rebuilt.position = tuple(event["after"])
            rebuilt.messages.extend(reply_messages(rebuilt, turn, event))
    if rebuilt.messages != saved:
        raise ValueError("The saved messages disagree with the run's record after its last response.")


def from_payload(data, read_prompt=None):
    """A saved run, checked against itself, ready to replay.

    ``read_prompt`` is handed to :func:`validate_inserts` for a run carrying
    inserted messages.
    """
    if not isinstance(data, dict) or data.get("format") not in (FORMAT, CHANGING_FORMAT, INSERT_FORMAT):
        raise ValueError("Choose a ChatLab maze run JSON file.")
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    if not isinstance(data.get("maze"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("The run is missing its map or its configuration.")
    inserted = data["format"] == INSERT_FORMAT
    # An insertion shifts the context of every response after it, so a run
    # carrying one is only read under the format that says so.
    if not inserted and data["config"].get("context_inserts"):
        raise ValueError(f"A run carrying inserted messages has to be recorded as {INSERT_FORMAT}.")
    # The format says whether a run carries insertions; whether its map
    # changes is read off the map itself, which a changing one records with
    # its environment identifier.
    changing = data["format"] == CHANGING_FORMAT or (inserted and "environment_id" in data["maze"])
    # A fixed-map run carrying closures would replay as the map it started
    # from, which is not the map its responses were answering.
    if not changing and (data["config"].get("map_updates") or data.get("dropped_closures")
                         or data.get("close_next")):
        raise ValueError(f"A run whose map changes has to be recorded as {CHANGING_FORMAT}." if not inserted else
                         "A run whose map changes has to record the changing map, with its environment identifier.")
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
        validate_pending(maze, result.close_next, updates, result.dropped_closures, result.turns)
        # Every closure begins as a request from the reader, and requesting one
        # marks the run. A file carrying a closure while reporting an untouched
        # run would be read as a clean control by anything scoring it.
        if (updates or result.dropped_closures or result.close_next) and not result.manual_intervention:
            raise ValueError("A run carrying a closure cannot report that nobody intervened in it.")
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
    validate_steering(result)
    validate_checkpoint_closures(result)
    result.position = position
    if inserted:
        validate_inserts(result, read_prompt)
    result.replay_only = True
    return result
