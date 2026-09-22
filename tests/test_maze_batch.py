import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr

from extension_api import SteeringError, TokenInspector
from extensions.maze_experiments import runner
from extensions.maze_experiments.batch import MANIFEST_NAME, SUMMARY_NAME, BatchControl, downloads, run_trials
from extensions.maze_experiments.maze import call_text
from extensions.maze_experiments.page import build_page
from extensions.maze_experiments.runner import from_payload
from extensions.maze_experiments.trials import FORMAT, read_trials
from test_maze import CONFIG, MAZE, Manager
from test_maze_steering import VECTOR

MOVE = call_text(MAZE.maze_id, "east")
ARRIVE = (MOVE, list(MOVE.encode()) + [0])


class CountingManager(Manager):
    """The maze fixture, counting claims and refusing one steering strength."""

    def __init__(self, replies):
        super().__init__(replies)
        self.claims = 0

    def claim_generation(self):
        held = super().claim_generation()
        self.claims += held is None
        return held

    def check_steering(self, value):
        if value["strength"] == 99:
            raise SteeringError("This vector was made for another model.")


def trial(identifier, **config):
    return dict(id=identifier, label=f"Trial {identifier}", maze=MAZE.to_dict(), openness=.7,
                config=CONFIG | dict(interruption_text="", goal_mode="coordinates", goal_hint="") | config)


class BatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # The pause between two responses is for a reader watching the board.
        patcher = mock.patch.object(runner.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def trials(self, *items, **extra):
        path = self.root / "trials.json"
        path.write_text(json.dumps(dict(format=FORMAT, title="Pilot sweep", trials=list(items), **extra)))
        return read_trials(path)

    def test_every_trial_runs_under_one_session_and_is_saved_and_summarized(self):
        data = self.trials(trial("a"), trial("b", sampling_seed=7))
        manager = CountingManager([ARRIVE] * 4)
        control = BatchControl()
        frames = []
        for frame in run_trials(data, manager, self.root, control, source="trials.json"):
            frames.append(frame)
            # Nothing else can take the model or load another between trials.
            self.assertTrue(manager.busy)
        self.assertFalse(manager.busy)
        self.assertEqual(manager.claims, 1)
        self.assertFalse(control.running)
        done, total, rows, directory, current = frames[-1]
        self.assertEqual((done, total, current), (2, 2, None))
        self.assertEqual([row["trial_id"] for row in rows], ["a", "b"])
        self.assertEqual({row["outcome"] for row in rows}, {"arrived"})
        self.assertEqual(rows[0]["model_moves"], 2)
        self.assertEqual(directory.parent, self.root / "batches")
        with open(directory / SUMMARY_NAME, newline="") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual([row["trial_id"] for row in table], ["a", "b"])
        manifest = json.loads((directory / MANIFEST_NAME).read_text())
        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(manifest["file_sha256"], data["file_sha256"])
        self.assertEqual((manifest["model_id"], manifest["source"]), ("test/model", "trials.json"))
        # Each run is an ordinary saved run, stamped with its trial.
        for row in rows:
            replay = from_payload(json.loads((directory / row["run_file"]).read_text()))
            self.assertEqual(replay.config["trial"]["id"], row["trial_id"])
            self.assertEqual(replay.run_id, row["run_id"])
        self.assertEqual(manager.calls[2][1]["seed"], 7)
        self.assertEqual({Path(path).name for path in downloads(directory)}, {SUMMARY_NAME, MANIFEST_NAME})

    def test_a_trial_the_model_refuses_is_recorded_and_the_batch_goes_on(self):
        steered = dict(steering=dict(VECTOR, strength=99), steer_when={"moves": 0}, steer_responses=0)
        data = self.trials(trial("refused", **steered), trial("after"))
        rows = list(run_trials(data, CountingManager([ARRIVE] * 2), self.root, BatchControl()))[-1][2]
        self.assertEqual([row["outcome"] for row in rows], ["refused", "arrived"])
        self.assertIn("another model", rows[0]["detail"])
        self.assertEqual(rows[0]["run_file"], "")

    def test_stop_ends_the_running_trial_as_stopped_and_starts_no_more(self):
        data = self.trials(trial("a"), trial("b"))
        manager = CountingManager([ARRIVE] * 4)
        control = BatchControl()
        for _done, _total, rows, directory, current in run_trials(data, manager, self.root, control):
            if current is not None:
                control.request_stop()
        self.assertEqual([row["outcome"] for row in rows], ["stopped"])
        self.assertEqual(json.loads((directory / MANIFEST_NAME).read_text())["status"], "stopped")
        self.assertFalse(manager.busy)

    def test_a_batch_closed_mid_trial_releases_the_model_and_lists_that_trial(self):
        data = self.trials(trial("a"), trial("b"))
        manager = CountingManager([ARRIVE] * 4)
        frames = run_trials(data, manager, self.root, BatchControl())
        for _done, _total, _rows, directory, current in frames:
            if current is not None:
                break
        frames.close()
        self.assertFalse(manager.busy)
        manifest = json.loads((directory / MANIFEST_NAME).read_text())
        self.assertEqual(manifest["status"], "stopped")
        self.assertEqual([row["outcome"] for row in manifest["results"]], ["stopped"])

    def test_a_busy_model_refuses_the_batch_before_anything_is_written(self):
        manager = CountingManager([])
        manager.busy = True
        with self.assertRaisesRegex(ValueError, "busy"):
            next(run_trials(self.trials(trial("a")), manager, self.root, BatchControl()))
        self.assertFalse((self.root / "batches").exists())

    def test_the_page_runs_the_loaded_file_on_its_own_queue(self):
        data = self.trials(trial("a"))
        manager = CountingManager([ARRIVE] * 2)
        context = SimpleNamespace(tokens=TokenInspector(), models=manager, data_dir=self.root,
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}
        batch = callbacks["run_batch"]
        # A batch runs for hours, so it cannot hold the view's queue.
        self.assertNotEqual(batch.concurrency_id, callbacks["load_trial"].concurrency_id)
        with self.assertRaisesRegex(gr.Error, "trial definitions file"):
            list(batch.fn(None, BatchControl(), None))
        frames = list(batch.fn(data, BatchControl(), str(self.root / "trials.json")))
        self.assertTrue(all(len(frame) == len(batch.outputs) for frame in frames))
        status, table, files, run_button, stop_button = frames[-1]
        self.assertIn("Finished 1 trial", status)
        self.assertIn("1 arrived", status)
        self.assertEqual(table["value"], [["Trial a", "arrived", 2, 2 * len(ARRIVE[1]), "", ""]])
        self.assertEqual(len(files["value"]), 2)
        self.assertTrue(run_button["visible"])
        self.assertFalse(stop_button["visible"])


if __name__ == "__main__":
    unittest.main()
