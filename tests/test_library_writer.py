"""Persistence barriers, bounded coalescing, and concurrent tab merge semantics."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from chatlab import library
from chatlab.conversation import copy_forks, drop_branch, make_turn, new_forks, put_branch, put_branch_archived, put_branch_sampling
from chatlab.library_writer import LibraryWriter


class LibraryWriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "conversations.json"
        self.writer = LibraryWriter(interval=60)
        self.addCleanup(self.writer.close)

    def snapshot(self, text):
        forks = new_forks()
        put_branch(forks, "Main", [make_turn("assistant", text)])
        return forks

    def test_coalesces_and_snapshots_before_producer_mutates(self):
        with mock.patch.object(library, "write", wraps=library.write) as write:
            receipts = [self.writer.submit("tab", self.snapshot(str(i)), self.path) for i in range(100)]
            final = self.snapshot("last")
            receipt = self.writer.submit("tab", final, self.path)
            final["branches"]["Main"][0]["content"] = "mutated"
            self.assertTrue(self.writer.flush(receipt))
            self.assertEqual(write.call_count, 1)
            self.assertTrue(all(r.done.is_set() and r.success for r in receipts))
        self.assertEqual(library.read(self.path)["branches"]["Main"][0]["content"], "last")

    def test_flush_waits_for_inflight_and_latest_save_without_blocking_submit(self):
        entered, release = threading.Event(), threading.Event()
        original = library.write

        def blocked(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        self.addCleanup(release.set)
        with mock.patch.object(library, "write", blocked):
            first = self.writer.submit("tab", self.snapshot("first"), self.path)
            waiter = threading.Thread(target=self.writer.flush, args=(first,))
            waiter.start()
            self.assertTrue(entered.wait(2))
            last = self.writer.submit("tab", self.snapshot("last"), self.path)
            self.assertFalse(first.done.is_set())
            self.assertFalse(last.done.is_set())
            release.set()
            self.assertTrue(self.writer.flush(last))
            waiter.join(2)
            self.assertFalse(waiter.is_alive())
        self.assertEqual(library.read(self.path)["branches"]["Main"][0]["content"], "last")

    def test_other_tab_selection_sampling_archive_and_deletion_survive(self):
        initial = self.snapshot("old")
        put_branch(initial, "Chat 1", [make_turn("user", "other")])
        library.write(initial, self.path)
        streaming = copy_forks(initial)
        put_branch(streaming, "Chat 1", [make_turn("assistant", "stream")])
        receipt = self.writer.submit("generation", streaming, self.path)
        other = copy_forks(initial)
        put_branch_sampling(other, "Chat 1", {"temperature": 0.2})
        put_branch_archived(other, "Chat 1", True)
        library.write(other, self.path)
        self.assertTrue(self.writer.flush(receipt))
        saved = library.read(self.path)
        self.assertEqual(saved["active"], "Main")
        self.assertEqual(saved["branches"]["Chat 1"][0]["content"], "stream")
        self.assertEqual(saved["sampling"]["Chat 1"], {"temperature": 0.2})
        self.assertTrue(saved["archived"]["Chat 1"])
        receipt = self.writer.submit("generation", streaming, self.path)
        drop_branch(saved, "Chat 1")
        library.write(saved, self.path)
        self.assertTrue(self.writer.flush(receipt))
        self.assertNotIn("Chat 1", library.read(self.path)["branches"])

    def test_pending_tabs_merge_instead_of_replacing_each_other(self):
        first = self.snapshot("one")
        second = new_forks()
        put_branch(second, "Chat 1", [make_turn("user", "two")])
        self.writer.submit("tab1", first, self.path)
        receipt = self.writer.submit("tab2", second, self.path)
        self.assertTrue(self.writer.flush(receipt))
        saved = library.read(self.path)
        self.assertEqual(saved["branches"]["Main"][0]["content"], "one")
        self.assertEqual(saved["branches"]["Chat 1"][0]["content"], "two")

    def test_shutdown_drains_and_rejects_later_frames(self):
        receipt = self.writer.submit("tab", self.snapshot("kept"), self.path)
        self.assertTrue(self.writer.close())
        self.assertTrue(receipt.success)
        rejected = self.writer.submit("tab", self.snapshot("late"), self.path)
        self.assertTrue(rejected.done.is_set())
        self.assertFalse(rejected.success)
        self.assertEqual(library.read(self.path)["branches"]["Main"][0]["content"], "kept")

    def test_failure_releases_barrier_and_writer_can_save_next_frame(self):
        with mock.patch.object(library, "write", side_effect=OSError("full disk")):
            receipt = self.writer.submit("tab", self.snapshot("lost"), self.path)
            self.assertFalse(self.writer.flush(receipt))
        receipt = self.writer.submit("tab", self.snapshot("recovered"), self.path)
        self.assertTrue(self.writer.flush(receipt))

    def test_flush_of_completed_receipt_does_not_disable_next_coalescing_window(self):
        receipt = self.writer.submit("tab", self.snapshot("first"), self.path)
        self.assertTrue(self.writer.flush(receipt))
        self.assertTrue(self.writer.flush(receipt))
        second = self.writer.submit("tab", self.snapshot("second"), self.path)
        self.assertFalse(second.done.wait(0.03))
        self.assertTrue(self.writer.flush(second))

    def test_periodic_save_does_not_need_a_flush(self):
        writer = LibraryWriter(interval=0.03)
        self.addCleanup(writer.close)
        first = writer.submit("tab", self.snapshot("first"), self.path)
        deadline = time.monotonic() + 2
        i = 0
        while not first.done.is_set() and time.monotonic() < deadline:
            latest = writer.submit("tab", self.snapshot(str(i)), self.path)
            i += 1
            first.done.wait(0.002)
        # A debounce that restarts the deadline on each frame never reaches
        # this barrier while frames keep arriving.
        self.assertTrue(first.done.is_set())
        self.assertTrue(latest.done.wait(2))
        self.assertTrue(latest.success)
