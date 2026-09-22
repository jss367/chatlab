"""Running every trial in a file, one after another, under one model session.

A trial file is a whole experiment's inputs. This runs it: each trial becomes
a fresh episode, generated to its end with no one at the controls, saved as an
ordinary run file, and summarized in one row of a table written beside them.
The session is held from the first trial to the last, so Chat cannot answer
and no other model can be loaded between two trials, and every row describes
the same weights.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path

from .runner import stream_episode
from .trials import prepare_trial
from extension_api import write_private_text

logger = logging.getLogger(__name__)

FORMAT = "chatlab-maze-batch-1"
SUMMARY_NAME = "summary.csv"
MANIFEST_NAME = "batch.json"
COLUMNS = ("trial_id", "label", "run_id", "outcome", "model_moves", "supplied_moves", "responses",
           "sampled_tokens", "tool_attempts", "rejected_calls", "interrupted", "recovered", "recovery_latency",
           "first_move_progress", "waypoint_reached", "steered_responses", "seconds", "model_id", "run_file",
           "detail")
# How often a running batch reports progress while a response streams. The
# episode yields on every token, and the pane only needs to keep up with a
# reader glancing at it.
PROGRESS_SECONDS = .5


class BatchControl:
    """What the Stop button reaches while a batch runs, one per browser session."""

    def __init__(self):
        self.running = False
        self.stop_requested = False
        self.episode = None

    def request_stop(self):
        """End the trial running now, recorded as stopped, and run no more."""
        self.stop_requested = True
        episode = self.episode
        if episode is not None:
            episode.request_stop()


def batch_directory(root, title):
    """A new directory for one batch, named for when it started and its collection."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "trials"
    directory = Path(root) / "batches" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    suffix, candidate = 1, directory
    while candidate.exists():
        suffix += 1
        candidate = directory.with_name(f"{directory.name}-{suffix}")
    candidate.mkdir(parents=True)
    return candidate


def summarize(item, episode, seconds, error=None):
    """One row of the summary table for a trial, whether or not it ran."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(trial_id=item["id"], label=item["label"])
    if episode is None:
        row.update(outcome="refused", detail=str(error))
        return row
    config = episode.config
    model_events = [e for e in episode.events if e["source"] == "model"]
    waypoint_turn = episode.waypoint_turn
    row.update(
        run_id=episode.run_id, outcome="refused" if error else episode.phase,
        model_moves=episode.moves - episode.supplied_moves, supplied_moves=episode.supplied_moves,
        responses=len(episode.turns), sampled_tokens=episode.sampled_tokens,
        tool_attempts=episode.tool_attempts, rejected_calls=sum(not e["accepted"] for e in model_events),
        interrupted=episode.interrupted,
        recovered="" if episode.resumed is None else episode.resumed,
        recovery_latency="" if episode.latency is None else episode.latency,
        first_move_progress="" if episode.first_move_progress is None else episode.first_move_progress,
        waypoint_reached="" if config.get("waypoint") is None else waypoint_turn is not None,
        steered_responses=sum(bool(turn.get("steered")) for turn in episode.turns),
        seconds=round(seconds, 1), model_id=episode.model_id or "",
        run_file="" if error else f"runs/{episode.run_id}.json",
        detail=str(error) if error else episode.detail)
    return row


def summary_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_summary(directory, manifest, rows):
    """Write the table and the manifest from scratch, after every trial.

    A stopped batch is still an experiment, so both are on disk from the first
    trial rather than at the end. Rewriting them costs a few kilobytes a trial,
    which is nothing beside a response.
    """
    write_private_text(directory / SUMMARY_NAME, summary_csv(rows), newline="")
    write_private_text(directory / MANIFEST_NAME,
                       json.dumps(dict(manifest, results=rows), ensure_ascii=False, indent=1) + "\n")


def downloads(directory):
    """Copies of the table and the manifest where the interface is allowed to serve them."""
    staged = Path(tempfile.mkdtemp(prefix="chatlab-maze-batch-"))
    found = []
    for name in (SUMMARY_NAME, MANIFEST_NAME):
        if (Path(directory) / name).exists():
            found.append(str(shutil.copy(Path(directory) / name, staged / name)))
    return found


def run_trials(data, models, root, control, *, source=""):
    """Run every trial in ``data`` in order, yielding progress as it goes.

    Each yield is ``(done, total, rows, directory, current)``: the trials
    finished, how many there are, their summary rows, where the batch is
    written, and the episode generating now or None. A trial the loaded model
    refuses, such as one whose steering vector was made for another model, is
    recorded as refused and the batch moves on. A stop ends the trial running
    now and runs no more.
    """
    session = models.open_session()
    try:
        directory = batch_directory(root, data["title"])
    except BaseException:
        session.close()
        raise
    control.running, control.stop_requested, control.episode = True, False, None
    items = data["trials"]
    runs = directory / "runs"
    manifest = dict(format=FORMAT, title=data["title"], source=source, file_sha256=data["file_sha256"],
                    model_id=session.model_id, load_id=session.load_id,
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), finished_at=None,
                    status="running", total=len(items))
    rows = []
    logger.info("Batch %s: running %s trials from %s with %s", directory.name, len(items),
                data["title"], session.model_id)
    try:
        write_summary(directory, manifest, rows)
        yield 0, len(items), rows, directory, None
        for item in items:
            if control.stop_requested:
                break
            started, episode, error = time.time(), None, None
            try:
                episode = prepare_trial(data, item["id"])
                reported = 0.
                with closing(stream_episode(episode, models, save_dir=runs, session=session)) as frames:
                    for current in frames:
                        # Handed to the Stop button only once it is generating.
                        # Stopping an episode that has not started ends it on
                        # the spot, and the stream would then refuse it as
                        # finished; a stop that lands before this is caught
                        # by the flag instead.
                        control.episode = current
                        if control.stop_requested:
                            current.request_stop()
                        if time.monotonic() - reported >= PROGRESS_SECONDS:
                            reported = time.monotonic()
                            yield len(rows), len(items), rows, directory, current
            except ValueError as exc:
                error = exc
                logger.warning("Batch %s: trial %r refused: %s", directory.name, item["id"], exc)
            finally:
                # In the finally, so a batch closed mid-trial - its window gone,
                # say - still lists the trial whose run it saved as stopped.
                control.episode = None
                rows.append(summarize(item, episode, time.time() - started, error))
            logger.info("Batch %s: trial %s of %s, %r, %s", directory.name, len(rows), len(items),
                        item["id"], rows[-1]["outcome"])
            write_summary(directory, manifest, rows)
            yield len(rows), len(items), rows, directory, None
        manifest["status"] = "stopped" if control.stop_requested else "finished"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        control.running, control.episode = False, None
        session.close()
        try:
            write_summary(directory, manifest, rows)
        except OSError as exc:
            logger.warning("Batch %s: could not write its summary: %s", directory.name, exc)
        logger.info("Batch %s %s after %s of %s trials", directory.name, manifest["status"], len(rows), len(items))
