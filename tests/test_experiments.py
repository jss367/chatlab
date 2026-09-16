import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import compare
import experiment_runs as runs
import settings_sandbox
from model_runtime import LoadedModel
from test_compare import metric, run
from test_streaming import EOS_ID, loaded_manager
from ui import experiment_compare, experiments, runtime


class SavedRunsTests(unittest.TestCase):
    def setUp(self):
        settings_sandbox.start()
        self.addCleanup(settings_sandbox.stop)
        self.run = run([metric(1, 0, 2), metric(2, 1, 9), metric(3, 2, 0, scored=False)])

    def test_round_trip_keeps_metrics_alignment_and_configuration(self):
        self.run.update(tokenizer="vocab", token_ends=[2, 3, 5], decoded="hello", context_ids=[4])
        item = runs.save(self.run, "Precision comparison")
        self.run["metrics"][0]["surprise_bits"] = 100
        loaded = runs.read(item["id"])
        self.assertEqual(loaded["run"]["metrics"][0]["surprise_bits"], 2)
        self.assertEqual(loaded["run"]["token_ends"], [2, 3, 5])
        self.assertEqual(loaded["run"]["settings"]["seed"], 1)
        self.assertEqual(runs._path(item["id"]).stat().st_mode & 0o777, 0o600)

    def test_concurrent_bookmarks_merge_and_search_notes(self):
        item = runs.save(self.run, "Baseline")
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda index: runs.bookmark(item["id"], index, f"finding {index}"), [0, 1]))
        self.assertEqual(len(runs.read(item["id"])["bookmarks"]), 2)
        self.assertEqual(runs.search("FINDING 1")[0]["id"], item["id"])
        runs.bookmark(item["id"], 1, "", remove=True)
        self.assertEqual(runs.search("finding 1"), [])

    def test_failed_replace_leaves_previous_document(self):
        item = runs.save(self.run)
        with mock.patch.object(runs.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                runs.bookmark(item["id"], 0, "new")
        self.assertEqual(runs.read(item["id"])["bookmarks"], {})
        self.assertEqual(list(runs.directory().glob("*.tmp")), [])

    def test_refresh_cannot_clear_a_selection_made_while_it_was_running(self):
        item = runs.save(self.run)
        update = experiments.refresh_choices()
        self.assertNotIn("value", update)
        self.assertEqual(update["choices"][0][1], item["id"])
        self.assertEqual(experiments.choices("no match", item["id"])["value"], None)

    def test_invalid_ids_and_corrupt_files_do_not_break_library(self):
        item = runs.save(self.run)
        runs._path("a" * 32).write_text("broken")
        self.assertEqual(len(runs.search()), 1)
        for identifier in ("../conversations", "", None):
            with self.assertRaises(ValueError):
                runs.read(identifier)
        self.assertEqual(runs.read(item["id"])["id"], item["id"])

    def test_offline_open_navigation_and_bookmark_selection(self):
        item = runs.save(self.run)
        opened = experiments.open_run(item["id"], "Raw rank")
        self.assertEqual(opened[0]["id"], item["id"])
        first = experiments.navigate(opened[0], None, "Highest surprise")
        self.assertEqual(first[0], 1)
        self.assertEqual(first[5]["value"][1][1], "Selected token")
        second = experiments.navigate(opened[0], first[0], "Highest surprise")
        self.assertEqual(second[0], 0)
        self.assertEqual(experiments.navigate(opened[0], second[0], "Highest surprise")[0], 1)
        bookmarked = runs.bookmark(item["id"], 0, "Keep this")
        self.assertEqual(experiments.navigate(bookmarked, None, "Bookmarks")[3], "Keep this")
        self.assertIsNone(experiments.select_token(item, 99)[0])

    def test_close_alternatives_sort_smallest_margin_first(self):
        self.run["metrics"][1]["top1_margin"] = 0.001
        self.assertEqual(runs.ranked_tokens(self.run, "Closest alternatives"), [1, 0])

    def test_difference_navigation_uses_only_comparable_spans(self):
        left = run([metric(1, 0, 1), metric(2, 1, 2), metric(3, 2, 90)])
        right = run([metric(1, 0, 2), metric(2, 1, 9), metric(3, 5, 0)])
        held = {"reading": compare.reading(left, right)}
        position, detail = experiment_compare.next_difference(held, None)
        self.assertIn("gap: **7.00 bits**", detail)
        next_position, detail = experiment_compare.next_difference(held, position)
        self.assertIn("gap: **1.00 bits**", detail)
        self.assertEqual(experiment_compare.next_difference(held, next_position)[0], position)

    def test_inspection_rejects_another_response_with_the_same_prompt(self):
        self.run.update(context_ids=[2], metrics_generation=4, session_id=runs.SESSION_ID)
        item = runs.save(self.run)
        target = {"strip": "response", "index": 0, "generation": 4}
        insight = {"saved_target": target, "saved_session": runs.SESSION_ID, "layers": [], "attention": []}
        context = (4, [2], self.run["load_id"])
        saved = runs.save_inspection(item["id"], insight, target, context)
        self.assertEqual(len(saved["inspections"]), 1)
        with self.assertRaises(ValueError):
            runs.save_inspection(item["id"], insight, dict(target, generation=5), (5, [2], self.run["load_id"]))
        with mock.patch.object(runs, "SESSION_ID", "another-session"):
            with self.assertRaises(ValueError):
                runs.save_inspection(item["id"], insight, target, context)

    def test_chat_trace_keeps_its_own_provenance_after_a_model_switch(self):
        trace = {"tokens": self.run["metrics"], "model_id": "original/model", "response": "hello",
                 "messages": [{"role": "system", "content": "Saved system prompt"}],
                 "run_context": {"load_id": "original/model#1", "context_ids": [2, 3],
                                 "precision": "4-bit", "tokenizer": "original-vocab"}}
        captured = runs.from_trace(trace, (99, [8], "replacement/model#2"))
        self.assertEqual(captured["load_id"], "original/model#1")
        self.assertEqual(captured["context_ids"], [2, 3])
        self.assertEqual(captured["precision"], "4-bit")
        self.assertEqual(captured["settings"]["system_prompt"], "Saved system prompt")


class AutomatedComparisonTests(unittest.TestCase):
    def setUp(self):
        settings_sandbox.start()
        self.addCleanup(settings_sandbox.stop)
        self.manager = loaded_manager([0, 1, EOS_ID])
        self.manager._loaded = LoadedModel("fake/model", "CPU", "full", self.manager.load_id)
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        cache = mock.patch.object(experiment_compare, "cache_status", return_value=SimpleNamespace(
            present=True, missing_files=[], unsupported=False, kind="text"))
        cache.start()
        self.addCleanup(cache.stop)
        self.conditions = ["", "Current", 0.0, 12, False, 1, "", "Current", 0.0, 13, False, 1]
        self.shared = [compare.REPLY, "Hello", "", False, "", "", 0.0, 1.0, 0, 0.0,
                       8, 1, False, "default", None, False, 1.0, 0]

    def test_pair_runs_sequentially_and_saves_both_conditions(self):
        frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
        self.assertIn("Comparison complete", frames[-1][2])
        stored = runs.search()
        self.assertEqual(len(stored), 2)
        self.assertEqual({item["run"]["settings"]["seed"] for item in stored}, {12, 13})
        self.assertFalse(self.manager.busy)
        self.assertTrue(any(isinstance(frame[0], dict) and frame[0].get("metrics") for frame in frames))
        self.assertTrue(any(isinstance(frame[1], dict) and frame[1].get("metrics") for frame in frames))

    def test_cleared_comparison_seeds_use_the_normal_zero_fallback(self):
        for seed_a, seed_b in ((None, 13), (12, None), (None, None)):
            with self.subTest(seed_a=seed_a, seed_b=seed_b):
                conditions = list(self.conditions)
                conditions[3], conditions[9] = seed_a, seed_b
                previous = {item["id"] for item in runs.search()}
                frames = list(experiment_compare.run_pair(*conditions, *self.shared))
                self.assertIn("Comparison complete", frames[-1][2])
                saved = [item for item in runs.search() if item["id"] not in previous]
                self.assertEqual(len(saved), 2)
                self.assertEqual(
                    {item["run"]["slot"]: item["run"]["settings"]["seed"] for item in saved},
                    {"A": seed_a or 0, "B": seed_b or 0},
                )
                self.assertIsNone(self.manager.occupant)
                self.assertFalse(experiment_compare._PAIR_LOCK.locked())

    def test_cancel_releases_generation_and_pair_lock(self):
        generator = experiment_compare.run_pair(*self.conditions, *self.shared)
        next(generator)
        next(generator)
        self.assertTrue(self.manager.busy)
        generator.close()
        self.assertFalse(self.manager.busy)
        self.assertFalse(experiment_compare._PAIR_LOCK.locked())
        self.assertEqual(runs.search(), [])

    def test_second_failure_preserves_first_saved_result(self):
        original = experiment_compare.fill_slot
        def filling(side, *args, **kwargs):
            if side == "B":
                raise RuntimeError("second failed")
            yield from original(side, *args, **kwargs)
        with mock.patch.object(experiment_compare, "fill_slot", filling):
            frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
        self.assertIn("second failed", frames[-1][2])
        self.assertEqual(len(runs.search()), 1)
        self.assertFalse(self.manager.busy)

    def test_missing_steering_refuses_before_any_generation(self):
        self.conditions[10] = True
        frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
        self.assertIn("Import a steering vector", frames[-1][2])
        self.assertEqual(runs.search(), [])

    def test_model_loads_are_sequential_and_blank_b_uses_original_model(self):
        self.conditions[0] = "other/model"
        loaded = []
        def loading(model, path, precision, kind):
            self.assertFalse(self.manager.busy)
            loaded.append(model)
            self.manager.model_id = model
            self.manager.load_count += 1
            self.manager._loaded = LoadedModel(model, "CPU", precision, self.manager.load_id)
            yield "Loading…"
        with mock.patch.object(self.manager, "find_cached", return_value="/fake/cache"), \
                mock.patch.object(experiment_compare, "stream_load", loading):
            frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
        self.assertEqual(loaded, ["other/model", "fake/model"])
        self.assertIn("Comparison complete", frames[-1][2])
        self.assertEqual({item["run"]["model_id"] for item in runs.search()}, {"other/model", "fake/model"})
        self.assertIsNone(self.manager.occupant)

    def test_stopping_after_a_completed_frame_already_has_a_saved_run(self):
        generator = experiment_compare.run_pair(*self.conditions, *self.shared)
        for frame in generator:
            if isinstance(frame[0], dict) and frame[0].get("metrics"):
                self.assertEqual(len(runs.search()), 1)
                generator.close()
                break
        self.assertFalse(self.manager.busy)
        self.assertEqual(len(runs.search()), 1)

    def test_mlx_uses_its_stored_precision(self):
        self.manager._loaded = self.manager._loaded._replace(precision="4-bit")
        with mock.patch.object(experiment_compare, "cache_status", return_value=SimpleNamespace(
                present=True, missing_files=[], unsupported=False, kind="mlx")):
            frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
            self.assertIn("Comparison complete", frames[-1][2])
            self.conditions[1] = "full"
            frames = list(experiment_compare.run_pair(*self.conditions, *self.shared))
            self.assertIn("stored in their repository", frames[-1][2])

    def test_model_swap_before_claim_is_refused(self):
        from ui.compare import fill_slot
        frames = list(fill_slot("A", *self.shared, expected_load_id="old/model#1"))
        self.assertIn("model changed", frames[-1][1])
        self.assertFalse(self.manager.busy)

    def test_saved_rerun_preserves_messages_and_creates_new_document(self):
        original = run([metric(1, 0, 1)])
        original["messages"] = [{"role": "user", "content": "Earlier question"},
                                {"role": "assistant", "content": "Earlier answer"},
                                {"role": "user", "content": "Follow-up"}]
        item = runs.save(original, "Conversation experiment")
        frames = list(experiments.rerun_saved(item))
        self.assertIn("Rerun saved", frames[-1][1])
        stored = runs.search("Rerun:")[0]
        self.assertEqual(stored["run"]["messages"], original["messages"])
        self.assertNotEqual(stored["id"], item["id"])
        self.assertEqual(runs.read(item["id"])["run"], original)
        self.assertFalse(self.manager.busy)

    def test_rerun_preserves_literal_prefix_provenance(self):
        from ui.compare import _write_reply
        original = list(_write_reply("Hello", "", "Hello", 0, 1, 0, 0, 8, 1, False,
                                    "default", None, self.manager.loaded_model()))[-1][0]
        original["settings"]["forced_prefix_tokens"] = 1
        item = runs.save(original, "Literal prefix")
        frames = list(experiments.rerun_saved(item))
        self.assertIn("Rerun saved", frames[-1][1])
        result = runs.search("Rerun:")[0]["run"]
        self.assertEqual(result["metrics"][0]["token_id"], original["metrics"][0]["token_id"])
        self.assertEqual(result["settings"]["forced_prefix_tokens"], 1)
        self.assertEqual(result["text"], original["text"])

    def test_rerun_cancellation_keeps_original_and_releases_model(self):
        item = runs.save(run([metric(1, 0, 1)]))
        generator = experiments.rerun_saved(item)
        next(generator)
        next(generator)
        generator.close()
        self.assertFalse(self.manager.busy)
        self.assertEqual(len(runs.search()), 1)

    def test_saved_documents_are_plain_json(self):
        list(experiment_compare.run_pair(*self.conditions, *self.shared))
        for path in runs.directory().glob("*.json"):
            self.assertIsInstance(json.loads(path.read_text())["run"]["metrics"], list)
