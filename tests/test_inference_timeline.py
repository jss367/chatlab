import copy
import unittest
from unittest import mock

from chatlab import charts
from chatlab import experiment_runs as runs
import settings_sandbox
from compare_support import metric, run
from fakes import lens_manager
from chatlab.ui import inference_timeline as timeline, runtime


class TimelineTests(unittest.TestCase):
    def setUp(self):
        settings_sandbox.start()
        self.addCleanup(settings_sandbox.stop)
        self.manager = lens_manager([0, 1, 2, 3])
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        original = run([metric(1, 0, 1), metric(2, 1, 2), metric(3, 2, 3)])
        original.update(context_ids=[3], session_id=runs.SESSION_ID, load_id=self.manager.load_id)
        self.document = runs.save(original, "Timeline test")

    def capture(self, **overrides):
        options = dict(document=self.document, expected=self.document["id"], first=2, last=4,
                       stride=1, mode="Logit", pin="", keep_attention=True)
        options.update(overrides)
        return timeline.capture(**options)

    def test_capture_and_offline_replay_synchronize_exact_positions(self):
        result = list(self.capture())[-1][0]
        self.assertFalse(self.manager.busy)
        self.assertEqual(len(result["timeline_inspections"]), 3)
        self.manager.unload()
        result = runs.read(result["id"])
        for number in (2, 3, 4):
            frame = timeline.frame(result, number, "Logit", 0)
            self.assertIn(f"position {number} / 4", frame[-1])
            self.assertIn("recomputed after the run", frame[-1])
            self.assertNotEqual(frame[4], timeline.EMPTY_LENS)
            self.assertNotEqual(frame[5], charts.EMPTY_ATTENTION)
            self.assertEqual(frame[1][number - 1][1], "Selected")
        missing = timeline.frame(result, 1, "Logit", 0)
        self.assertEqual(missing[4:6], (timeline.EMPTY_LENS, charts.EMPTY_ATTENTION))
        self.assertIn("not retained", missing[2])

    def test_cancel_keeps_completed_readouts_and_releases_model(self):
        stream = self.capture()
        next(stream)
        next(stream)
        self.assertTrue(self.manager.busy)
        stream.close()
        self.assertFalse(self.manager.busy)
        self.assertEqual(len(runs.read(self.document["id"])["timeline_inspections"]), 1)

    def test_reload_session_change_and_changed_picker_are_refused(self):
        for updates in ({"session_id": "old-session"}, {"load_id": "old-load"}):
            document = runs.save(self.document["run"] | updates)
            frames = list(self.capture(document=document, expected=document["id"]))
            self.assertIn("original model load", frames[-1][1])
            self.assertNotIn("timeline_inspections", runs.read(document["id"]))
            self.assertFalse(self.manager.busy)
        self.assertIn("wait for it to open", list(self.capture(expected="other"))[-1][1])

    def test_capture_bounds_and_first_logit_position(self):
        for first, last, stride in ((0, 4, 1), (2, 9, 1), (4, 2, 1), (2, 4, -1), (1, 4, 1)):
            with self.subTest(first=first, last=last, stride=stride):
                frames = list(self.capture(first=first, last=last, stride=stride))
                self.assertIn("Could not capture timeline", frames[-1][1])
                self.assertFalse(self.manager.busy)
        document = runs.save(self.document["run"] | {"metrics": [metric(i, 0, 1) for i in range(40)]})
        frames = list(self.capture(document=document, expected=document["id"], last=40))
        self.assertIn("at most 32", frames[-1][1])

    def test_attention_is_optional_and_missing_lens_does_not_reuse_other_mode(self):
        result = list(self.capture(first=2, last=2, keep_attention=False))[-1][0]
        insight = timeline.readout(result, 1, "Logit")
        self.assertEqual(insight["attention"], [])
        self.assertEqual(insight["tokens"], [])
        frame = timeline.frame(result, 2, "Jacobian", 0)
        self.assertEqual(frame[4:6], (timeline.EMPTY_LENS, charts.EMPTY_ATTENTION))
        self.assertIn("Import a matching Jacobian lens", list(self.capture(mode="Jacobian"))[-1][1])

    def test_playback_clamps_stops_and_resets_on_run_switch(self):
        args = (self.document, self.document["id"])
        frame = timeline.move(*args, 3, "Logit", 0, playing=True, delta=1)
        self.assertEqual(frame[0]["value"], 4)
        self.assertFalse(frame[-2])
        self.assertFalse(frame[-1]["active"])
        self.assertEqual(timeline.move(*args, 1, "Logit", 0, delta=-1)[0]["value"], 1)
        opened = timeline.opened(*args, "old-run", 4, "Logit", 0)
        self.assertEqual(opened[1]["value"], 1)
        self.assertFalse(opened[-2])
        self.assertEqual(timeline.move(self.document, "other", 3, "Logit", 0)[4], timeline.EMPTY_LENS)

    def test_prompt_metrics_survive_trace_save_without_borrowing_other_tokens(self):
        prompt = metric(1, 3, 5) | {"segment": "prompt"}
        trace = {"tokens": self.document["run"]["metrics"], "prompt_tokens": [prompt],
                 "run_context": {"context_ids": [3]}}
        result = runs.save(runs.from_trace(trace))
        prompt["token_id"] = 99
        self.assertEqual(timeline.tokens(result)[0]["token_id"], 3)
        self.assertEqual(timeline.tokens(result)[0]["surprise_bits"], 5)
        result["run"]["prompt_metrics"][0]["token_id"] = 9
        self.assertFalse(timeline.tokens(result)[0]["scored"])

    def test_replacing_capture_merges_bookmarks_and_checks_position(self):
        result = list(self.capture(first=2, last=2))[-1][0]
        runs.bookmark(result["id"], 0, "Keep this finding")
        result = list(self.capture(first=2, last=2))[-1][0]
        self.assertEqual(result["bookmarks"], {"0": "Keep this finding"})
        self.assertEqual(len(result["timeline_inspections"]), 1)
        insight = copy.deepcopy(timeline.readout(result, 1, "Logit"))
        insight["token_id"] = 999
        with self.assertRaises(ValueError):
            runs.save_timeline_inspection(result["id"], 1, insight)
        self.assertEqual(runs.read(result["id"]), result)

    def test_saved_legacy_inspection_maps_response_index_to_absolute_position(self):
        result = list(self.capture(first=2, last=2))[-1][0]
        insight = timeline.readout(result, 1, "Logit")
        result["timeline_inspections"] = {}
        result["inspections"] = [{"token_index": 0, "insight": insight}]
        self.assertEqual(timeline.readout(result, 1, "Logit"), insight)
        self.assertIsNone(timeline.readout(result, 0, "Logit"))

    def test_storage_limits_leave_previous_readouts_intact(self):
        result = list(self.capture(first=2, last=2))[-1][0]
        insight = timeline.readout(result, 1, "Logit")
        full = copy.deepcopy(result)
        full["timeline_inspections"].update({f"logit:{i}": {} for i in range(3, 66)})
        runs._write(full)
        # Replacing an existing readout remains possible at the count limit.
        runs.save_timeline_inspection(full["id"], 1, insight)
        with self.assertRaisesRegex(ValueError, "64 timeline readouts"):
            runs.save_timeline_inspection(full["id"], 2, insight | {"index": 2, "token_id": 1})
        with self.assertRaisesRegex(ValueError, "32 MB"):
            runs.save_timeline_inspection(full["id"], 1, insight | {"oversized": "x" * (32 * 1024 * 1024)})
        self.assertEqual(runs.read(full["id"])["timeline_inspections"]["logit:1"]["insight"], insight)
