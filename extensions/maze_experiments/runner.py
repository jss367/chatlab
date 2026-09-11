"""Single-episode controller: token generation is separate from authoritative maze movement."""
from __future__ import annotations

import copy
import json
import re
import threading
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .maze import Maze, TOOLS, apply_call, initial_history, parse_call
from extension_api import write_private_text

FORMAT = "chatlab-maze-run-1"
TERMINAL = {"arrived", "abandoned", "budget", "stopped", "error"}


@dataclass
class Episode:
    maze: Maze
    config: dict
    run_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "ready"
    detail: str = "Ready. Run the episode or generate one response at a time."
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
    pause_requested: bool = False
    stop_requested: bool = False
    busy: bool = False
    replay_only: bool = False
    token_edit: dict | None = None
    pending_edit: dict | None = None
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
        supplied = int(self.config.get("supplied_moves", 3))
        self.messages, self.events, self.position = initial_history(
            self.maze, supplied, goal_mode=self.config["goal_mode"], goal_hint=self.config["goal_hint"])
        self.supplied_moves = supplied

    def model_state(self, error=None):
        return self.maze.state(self.position, error, goal_mode=self.config["goal_mode"], goal_hint=self.config["goal_hint"])

    @property
    def moves(self):
        return sum(e["accepted"] for e in self.events)

    def payload(self):
        keys = ("run_id", "phase", "detail", "messages", "events", "turns", "position", "model_id", "load_id",
                "sampled_tokens", "tool_attempts", "supplied_moves", "interrupted", "intervention_turn",
                "intervention_tokens", "intervention_attempts", "resumed", "first_move_progress", "latency",
                "manual_intervention", "created_at", "token_edit", "pending_edit")
        return {"format": FORMAT, "maze": self.maze.to_dict(), "config": self.config,
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
            if save_dir:
                try:
                    self.save(save_dir)
                except OSError as exc:
                    self.warn_autosave(str(exc))

    def warn_autosave(self, error):
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


def identifies_its_token(text):
    """Whether recorded text pins down the ID it came from.

    A byte-level piece of a character decodes on its own to the replacement
    character, and a token carrying no characters decodes to nothing. Either
    way the recorded text is the same for any ID of that shape, so a later
    vocabulary can move the ID to other bytes and still produce a decode that
    looks identical. Text recorded as part of a response needs no such test,
    because the characters its neighbours complete name the bytes it carried;
    a token recorded on its own, with nothing around it, does.
    """
    return bool(text) and "�" not in text


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


def visible_token_ids(turn, hidden_ids):
    """The IDs this response's recorded text was decoded from.

    The runtime streams each response through IncrementalDecoder, which drops
    a hidden special rather than decoding it, so those IDs appear among the
    turn's metrics and never in its text. Reader-supplied prefill is the
    exception the runtime makes: replay forces those tokens visible, including
    special-token spellings.
    """
    literal_prefill_tokens = turn.get("literal_prefill_tokens", turn["forced_prefix_tokens"])
    return [metric["token_id"] for index, metric in enumerate(turn["metrics"])
            if index < literal_prefill_tokens or metric["token_id"] not in hidden_ids]


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
            current = manager.decode(visible_token_ids(turn, hidden))
        except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError("The loaded model cannot decode this run's token IDs, so its tokenizer is not the "
                             f"one that produced the run ({exc}). This happens when the same model ID has been "
                             "re-downloaded at a different revision.") from exc
        if current != turn["text"]:
            raise ValueError("The loaded weights tokenize differently from the ones that produced this run, so "
                             "its tokens cannot be replayed. This happens when the same model ID has been "
                             "re-downloaded at a different revision; load that snapshot to fork this run.")


def verify_recorded_candidate(episode, candidate, manager):
    """Refuse a fork onto an alternative the loaded model decodes differently.

    The chosen alternative is the one ID the fork replays that the run never
    generated, so no response text stands behind it and it is the one ID that
    has to be checked on its own. A revised vocabulary can leave every recorded
    response decoding as it always did and still move this one, and the panel's
    offer of "'x' · token 120" would then put some other text into the response.
    Recorded with nothing around it, its text also has to name an ID before the
    comparison means anything, which is where the run itself can run out of
    evidence and the session that offered the alternative is the only place
    that does not need any.
    """
    try:
        current = manager.decode([candidate["token_id"]])
    except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("The loaded model cannot decode the token ID of this alternative, so its tokenizer "
                         f"is not the one that offered it ({exc}). This happens when the same model ID has "
                         "been re-downloaded at a different revision.") from exc
    recorded = candidate.get("text")
    if not identifies_its_token(current) or not identifies_its_token(recorded):
        # An alternative that decodes to no characters was recorded under the
        # vocabulary's own name for it, which names no particular ID either.
        if not forkable_without_evidence(episode, manager):
            raise ValueError("This alternative recorded a byte fragment rather than text, so there is "
                             "nothing to confirm that the loaded tokenizer still maps it to the same bytes. "
                             "It can only be applied by the session that offered it.")
        return
    if current != recorded:
        raise ValueError("The loaded model decodes this alternative differently from the model that offered "
                         "it, so it cannot be applied. This happens when the same model ID has been "
                         "re-downloaded at a different revision; load that snapshot to fork this run.")


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
        literal_prefill_tokens = original.get("literal_prefill_tokens", original["forced_prefix_tokens"])
        verify_recorded_text(episode, turn_index, manager)
        verify_recorded_stops(episode, turn_index, kept, literal_prefill_tokens, stop_ids)
        if candidate_id is None:
            replacement_ids = manager.encode_replacement(
                kept_ids, replacement, literal_prefill_tokens=literal_prefill_tokens,
            )
        else:
            candidate = next((c for c in metrics[token_index].get("top_candidates", [])
                              if c["token_id"] == candidate_id), None)
            if candidate is None:
                raise ValueError("Choose an alternative for the selected token.")
            verify_recorded_candidate(episode, candidate, manager)
            replacement_ids = [candidate_id]
        if not replacement_ids:
            raise ValueError("Enter replacement text or choose a token alternative.")
        if any(t in stop_ids for t in replacement_ids[:-1]):
            raise ValueError("A stop token can only appear at the end of the replacement.")
        result = Episode(episode.maze, episode.config)
        # The fork continues under the weights in memory now, not the ones that
        # produced the original; token_edit keeps the original stamp.
        result.model_id, result.load_id = manager.model_id, manager.load_id
        result.manual_intervention = True
        # Rebuild history and recovery counters through the same simulator path
        # used during generation, excluding the edited response and its future.
        for i, previous in enumerate(episode.turns[:turn_index]):
            turn = copy.deepcopy(previous)
            result.turns.append(turn)
            if episode.interrupted and episode.intervention_turn == i:
                result.interrupted = True
                result.intervention_turn = i
                result.intervention_tokens = result.sampled_tokens
                result.intervention_attempts = result.tool_attempts
            result.phase = "running"
            finish_turn(result, turn, stop_ids, result.config["per_turn_tokens"])
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
    import re
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
        event = apply_call(episode.maze, episode.position, args, goal_mode=episode.config["goal_mode"])
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
          and (episode.sampled_tokens - episode.intervention_tokens >= 1024
               or episode.tool_attempts - episode.intervention_attempts >= 4)):
        episode.phase, episode.detail = "budget", "No accepted move within the recovery window."
    elif episode.sampled_tokens >= episode.config["token_budget"] or episode.tool_attempts >= episode.config["attempt_budget"]:
        episode.phase, episode.detail = "budget", "The episode reached its token or action limit."


def stream_episode(episode, models, *, single_step=False, save_dir=None):
    with episode.lock:
        if episode.busy:
            raise ValueError("This episode is already generating. Pause it before changing the run.")
        if episode.phase in TERMINAL or episode.replay_only:
            raise ValueError("Start a new episode to run again. This episode is finished or is a saved replay.")
        manager = models.open_session()
        episode.busy = True
        episode.pause_requested = episode.stop_requested = False
        episode.phase = "running"
        episode.model_id = episode.model_id or manager.model_id
        episode.load_id = episode.load_id or manager.load_id
    turn = None
    autosave_error = None

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
            edit = episode.pending_edit
            forced = edit["forced_ids"] if edit else interrupted_prefix(episode, manager)
            inserts_interruption = edit["interruption_here"] if edit else bool(forced)
            limit = min(episode.config["per_turn_tokens"], episode.config["token_budget"] - episode.sampled_tokens)
            if inserts_interruption:
                limit = min(limit, 1024)
            elif episode.interrupted and episode.resumed is None:
                limit = min(limit, 1024 - (episode.sampled_tokens - episode.intervention_tokens))
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
        episode.phase, episode.detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if turn is not None and turn.get("finish_reason") is None:
            count = max(0, len(turn["metrics"]) - turn["forced_prefix_tokens"])
            episode.sampled_tokens += count
            turn.update(sampled_tokens=count, tokens_cumulative=episode.sampled_tokens, finish_reason=episode.phase)
        with episode.lock:
            episode.busy = False
            manager.close()
            autosave()
            if autosave_error is not None:
                episode.warn_autosave(autosave_error)
    yield episode


def from_payload(data):
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Choose a ChatLab maze run JSON file.")
    if not re.fullmatch(r"[a-f0-9]{32}", str(data.get("run_id", ""))):
        raise ValueError("Invalid run identifier.")
    maze = Maze.from_dict(data["maze"])
    result = Episode(maze, data["config"])
    allowed = result.payload().keys() - {"format", "maze", "config", "exploratory", "tokenizer_note"}
    for key in allowed:
        if key in data:
            setattr(result, key, data[key])
    # Reconstruct the visible path from real transitions, never trust claimed positions.
    position = maze.start
    for event in result.events:
        if tuple(event["before"]) != position:
            raise ValueError("The saved path contains a position mismatch.")
        if event["accepted"]:
            mode = result.config["goal_mode"]
            actual = apply_call(maze, position, {"maze_id": maze.tool_id(mode), "direction": event["direction"]}, goal_mode=mode)
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
