"""The Models page: My Models and Model search, and the chat page's model switcher
and badge that read the same cache."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

from chatlab import app
from chatlab.ui import models_page, runtime
from chatlab import model_runtime
from chatlab import device_memory
from chatlab import hub_search
from chatlab import model_cache
from chatlab import progress_bars
from chatlab import settings
from chatlab.hub_search import HubModel
from chatlab.model_cache import (
    IMAGE_KIND,
    MODEL_WEIGHTS,
    TEXT_KIND,
    CacheStatus,
    format_count,
    list_cached_models,
    remove_cached_model,
    sort_cached_models,
)
from chatlab.model_runtime import ModelManager
from chatlab.progress_bars import DownloadProgress

from models_support import COMMIT, OLMO, cached, painted, picked
import settings_sandbox


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def roomy(
    test, total_gb=48, available_gb=40, backend="mps", dtype="float16", held_gb=0
):
    """Judge fit against a fixed machine, not the one running the tests.

    Availability moves from one second to the next and the tests must not,
    so every fit verdict under test is read from a profile like this one.
    ``held_gb`` is what a loaded model is holding on the device, which a load
    gives back before it checks whether the next model fits.
    """

    profile = device_memory.DeviceProfile(
        backend=backend,
        dtype=dtype,
        total=total_gb * 1024**3,
        available=available_gb * 1024**3,
        pool="this machine",
        held=held_gb * 1024**3,
    )
    original = models_page.device_profile
    models_page.device_profile = lambda torch=None: profile
    test.addCleanup(lambda: setattr(models_page, "device_profile", original))
    return profile


def lay_out(root: str, model_id: str, files: dict[str, bytes]) -> Path:
    """A model folder the way ``huggingface_hub`` keeps one: blobs plus symlinks."""

    folder = Path(root) / f"models--{model_id.replace('/', '--')}"
    blobs = folder / "blobs"
    blobs.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(COMMIT)
    snapshot = folder / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    for index, (name, content) in enumerate(files.items()):
        blob = blobs / f"blob{index}"
        blob.write_bytes(content)
        (snapshot / name).symlink_to(blob)
    return folder


class ModelActionTests(unittest.TestCase):
    def test_download_status_follows_worker_then_returns_to_local_files(self):
        progress = DownloadProgress()
        with (
            mock.patch.dict(runtime.MANAGER.active_downloads, {"org/model": progress}),
            mock.patch.object(models_page, "cache_status", return_value=CacheStatus()) as cached,
        ):
            detail, _, _, load = models_page.refresh_model_actions("org/model", None)
            self.assertIn("**Downloading**", detail)
            self.assertIn("Asking Hugging Face", detail)
            self.assertFalse(load["visible"])
            cached.assert_not_called()

            with mock.patch.object(progress, "snapshot", return_value=progress_bars.DownloadSnapshot(
                files_done=1, files_total=3, bytes_done=500_000_000, bytes_total=1_000_000_000,
            )):
                detail, *_ = models_page.refresh_model_actions("org/old-id", "org/model")
            self.assertIn("50%", detail)
            self.assertIn("500 MB of 1.0 GB", detail)

            # Changing selection must not show another model's download.
            detail, *_ = models_page.refresh_model_actions("org/other", None)
            self.assertIn("Not downloaded", detail)

            del runtime.MANAGER.active_downloads["org/model"]
            for status, expected in (
                (CacheStatus(cached_bytes=100), "**Downloaded**"),
                (CacheStatus(cached_bytes=100, missing_files=(MODEL_WEIGHTS,)), "Download incomplete"),
            ):
                cached.return_value = status
                detail, *_ = models_page.refresh_model_actions("org/model", None)
                self.assertIn(expected, detail)

    def test_the_timers_refresh_scans_the_cache_only_when_something_moved(self):
        progress = DownloadProgress()
        with (
            mock.patch.object(runtime.MANAGER, "active_downloads", {}) as downloads,
            mock.patch.object(models_page, "cache_status", return_value=CacheStatus()) as cached,
        ):
            # An unstamped tab is painted; an idle one that already matches
            # the cache's revision is not, and never reaches the disk.
            detail, *_, stamp = models_page.refresh_stale_model_actions(
                "org/model", None, None, None, None
            )
            self.assertIn("Not downloaded", detail)
            self.assertEqual(cached.call_count, 1)

            for _ in range(3):
                painted = models_page.refresh_stale_model_actions(
                    "org/model", None, None, None, stamp
                )
                self.assertTrue(all(models_page.gr.skip() == value for value in painted[:4]))
                self.assertEqual(painted[-1], stamp)
            self.assertEqual(cached.call_count, 1)

            # A download in flight repaints every tick, without a scan.
            downloads["org/model"] = progress
            runtime.MANAGER.note_cache_change()
            for _ in range(3):
                detail, *_, stamp = models_page.refresh_stale_model_actions(
                    "org/model", None, None, None, stamp
                )
                self.assertIn("**Downloading**", detail)
            self.assertEqual(cached.call_count, 1)

            # Finishing moves the revision, which buys exactly one scan.
            del downloads["org/model"]
            runtime.MANAGER.note_cache_change()
            cached.return_value = CacheStatus(cached_bytes=100)
            detail, *_, stamp = models_page.refresh_stale_model_actions(
                "org/model", None, None, None, stamp
            )
            self.assertIn("**Downloaded**", detail)
            self.assertEqual(cached.call_count, 2)
            models_page.refresh_stale_model_actions("org/model", None, None, None, stamp)
            self.assertEqual(cached.call_count, 2)

    def test_downloaded_selection_offers_local_loading(self):
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus(cached_bytes=100)) as status:
            detail, download_load, download, load = models_page.refresh_model_actions(
                "org/old-id", "org/downloaded"
            )
        status.assert_called_once_with("org/downloaded")
        self.assertIn("Downloaded", detail)
        self.assertFalse(download_load["visible"])
        self.assertFalse(download["visible"])
        self.assertTrue(load["visible"])
        self.assertEqual(load["variant"], "primary")

    def test_a_downloaded_model_is_named_by_kind_with_the_page_that_drives_it(self):
        # The panel is where a reader asks what they can do with the model in
        # front of them, and the two kinds are driven from different pages.
        for kind, page in ((TEXT_KIND, "**Chat**"), (IMAGE_KIND, "**Images**")):
            with self.subTest(kind=kind), mock.patch.object(
                models_page,
                "cache_status",
                return_value=CacheStatus(cached_bytes=100, kind=kind),
            ):
                detail, _, _, load = models_page.refresh_model_actions("org/model", None)

            self.assertIn("Downloaded", detail)
            self.assertIn(page, detail)
            self.assertTrue(load["visible"])

    def test_a_loaded_image_model_still_says_where_it_is_used(self):
        with (
            mock.patch.object(
                models_page,
                "cache_status",
                return_value=CacheStatus(cached_bytes=100, kind=IMAGE_KIND),
            ),
            mock.patch.object(runtime.MANAGER, "model_id", "org/model"),
        ):
            detail, _, _, _ = models_page.refresh_model_actions("org/model", None)

        self.assertIn("Loaded now", detail)
        self.assertIn("**Images**", detail)

    def test_missing_and_partial_models_keep_download_actions(self):
        for status, expected in (
            (CacheStatus(), "Not downloaded"),
            (CacheStatus(cached_bytes=100, missing_files=(MODEL_WEIGHTS,)), "Download incomplete"),
        ):
            with self.subTest(cached=status), mock.patch.object(models_page, "cache_status", return_value=status):
                detail, download_load, download, load = models_page.refresh_model_actions("org/model", None)
            self.assertIn(expected, detail)
            self.assertTrue(download_load["visible"])
            self.assertTrue(download["visible"])
            self.assertFalse(load["visible"])

    def test_unsupported_download_does_not_offer_to_fetch_the_same_files(self):
        with mock.patch.object(
            models_page, "cache_status", return_value=CacheStatus(cached_bytes=100, kind="")
        ):
            detail, *buttons = models_page.refresh_model_actions("org/model", None)
        self.assertIn("Unsupported", detail)
        self.assertTrue(all(not button["visible"] for button in buttons))

    def test_loaded_model_can_be_reloaded_with_new_precision(self):
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus(cached_bytes=100)), mock.patch.object(runtime.MANAGER, "model_id", "org/model"):
            detail, _, _, load = models_page.refresh_model_actions("org/model", None)
        self.assertIn("Loaded now", detail)
        self.assertTrue(load["visible"])

    def test_cache_read_failure_preserves_local_load_for_error_reporting(self):
        with mock.patch.object(models_page, "cache_status", side_effect=OSError("offline disk")):
            detail, _, _, load = models_page.refresh_model_actions("org/model", None)
        self.assertIn("Could not check", detail)
        self.assertTrue(load["visible"])

    def test_invalid_model_id_explains_format_and_hides_actions(self):
        for model_id in ("foo", "org/", "../escape"):
            with self.subTest(model_id=model_id):
                detail, *buttons = models_page.refresh_model_actions(model_id, None)
                self.assertIn("organization/model-name", detail)
                self.assertNotIn("Could not check", detail)
                self.assertTrue(all(not button["visible"] for button in buttons))


class FormatCountTests(unittest.TestCase):
    def test_counts_read_like_the_hub_pages(self):
        for count, text in [
            (0, "0"),
            (999, "999"),
            (1500, "1.5K"),
            (45_000, "45K"),
            (1_484_916_736, "1.5B"),
            (7_298_011_136, "7.3B"),
        ]:
            with self.subTest(count=count):
                self.assertEqual(format_count(count), text)


class CachedModelListTests(unittest.TestCase):
    """What the cache scan reports for the folders a cache can hold."""

    CONFIG = json.dumps(
        {"architectures": ["Olmo3ForCausalLM"], "dtype": "bfloat16"}
    ).encode()

    def test_an_absent_cache_lists_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(list_cached_models(Path(root) / "missing"), [])
            self.assertEqual(list_cached_models(Path(root)), [])

    def test_a_cache_that_cannot_be_read_lists_nothing(self):
        # ``is_dir`` can pass while the enumeration itself is refused.
        with tempfile.TemporaryDirectory() as root:
            lay_out(root, OLMO, {"config.json": b"{}", "model.safetensors": b"x"})
            with mock.patch.object(
                Path, "iterdir", side_effect=PermissionError(13, "Permission denied")
            ):
                self.assertEqual(list_cached_models(Path(root)), [])

    def test_a_folder_that_fails_partway_leaves_the_others_listed(self):
        with tempfile.TemporaryDirectory() as root:
            lay_out(root, OLMO, {"config.json": b"{}", "model.safetensors": b"x"})
            broken = lay_out(root, "org/broken", {"config.json": b"{}"})
            newest_write = model_cache._newest_write

            def refuse_one(folder, snapshot):
                if folder == broken:
                    raise OSError(5, "Input/output error")
                return newest_write(folder, snapshot)

            with mock.patch.object(model_cache, "_newest_write", refuse_one):
                listed = [entry.model_id for entry in list_cached_models(Path(root))]

        self.assertEqual(listed, [OLMO])

    def test_a_complete_model_is_listed_with_its_details(self):
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(
                root,
                OLMO,
                {"config.json": self.CONFIG, "model.safetensors": b"x" * 100},
            )
            (entry,) = list_cached_models(Path(root))

        self.assertEqual(entry.model_id, OLMO)
        self.assertTrue(entry.status.complete)
        # The size is the whole folder, refs/main included.
        self.assertEqual(entry.size_bytes, 100 + len(self.CONFIG) + len(COMMIT))
        self.assertEqual(entry.files, 2)
        self.assertEqual(entry.commit, COMMIT)
        self.assertEqual(entry.architecture, "Olmo3ForCausalLM")
        self.assertEqual(entry.dtype, "bfloat16")
        self.assertIsNotNone(entry.updated)
        self.assertEqual(entry.path, folder)

    def test_a_cut_off_download_is_listed_as_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": self.CONFIG})
            (folder / "blobs" / "shard.incomplete").write_bytes(b"y" * 40)
            (entry,) = list_cached_models(Path(root))

        self.assertEqual(entry.status.missing_files, (MODEL_WEIGHTS,))
        self.assertEqual(entry.status.partial_files, 1)
        self.assertEqual(entry.size_bytes, len(self.CONFIG) + 40 + len(COMMIT))
        # The config is on disk, so what the model is can still be said.
        self.assertEqual(entry.architecture, "Olmo3ForCausalLM")

    def test_old_revisions_count_toward_the_listed_size(self):
        # Without symlinks each snapshot holds its own files; the list says
        # what the folder takes, which is what removing it frees.
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": b"{}", "model.safetensors": b"x"})
            old = folder / "snapshots" / ("0" * 40)
            old.mkdir()
            (old / "model.safetensors").write_bytes(b"y" * 500)
            (entry,) = list_cached_models(Path(root))

        self.assertEqual(entry.size_bytes, 3 + 500 + len(COMMIT))
        self.assertEqual(entry.status.total_bytes, 3)

    def test_folders_that_are_not_models_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            lay_out(root, OLMO, {"config.json": b"{}", "model.safetensors": b"x"})
            (Path(root) / "datasets--allenai--dolma").mkdir()
            (Path(root) / "models--nonsense").mkdir()
            (Path(root) / "models--org--empty").mkdir()
            (Path(root) / "stray.txt").write_text("")
            listed = [entry.model_id for entry in list_cached_models(Path(root))]

        self.assertEqual(listed, [OLMO])

    def test_a_config_that_is_not_json_leaves_the_architecture_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            lay_out(root, OLMO, {"config.json": b"{", "model.safetensors": b"x"})
            (entry,) = list_cached_models(Path(root))

        self.assertIsNone(entry.architecture)
        self.assertIsNone(entry.dtype)

    def test_a_config_of_the_wrong_shape_leaves_the_architecture_unknown(self):
        # Another repo's config.json can hold anything: an object where the
        # list of architectures belongs, or no object at all. Neither may
        # take the whole list down with it.
        shapes = {
            "object architectures": b'{"architectures": {"name": "X"}, "dtype": 1}',
            "string architectures": b'{"architectures": "Olmo3ForCausalLM"}',
            "not an object": b'["Olmo3ForCausalLM"]',
        }
        for name, config in shapes.items():
            with self.subTest(shape=name), tempfile.TemporaryDirectory() as root:
                lay_out(root, OLMO, {"config.json": config, "model.safetensors": b"x"})
                lay_out(root, "org/other", {"config.json": self.CONFIG})
                entries = {e.model_id: e for e in list_cached_models(Path(root))}

                self.assertEqual(set(entries), {OLMO, "org/other"})
                self.assertIsNone(entries[OLMO].architecture)
                self.assertIsNone(entries[OLMO].dtype)
                self.assertEqual(entries["org/other"].architecture, "Olmo3ForCausalLM")

    def test_the_newest_download_comes_first(self):
        with tempfile.TemporaryDirectory() as root:
            older = lay_out(root, "org/older", {"config.json": b"{}"})
            lay_out(root, "org/newer", {"config.json": b"{}"})
            for blob in (older / "blobs").iterdir():
                os.utime(blob, (1_000_000, 1_000_000))
            listed = [entry.model_id for entry in list_cached_models(Path(root))]

        self.assertEqual(listed, ["org/newer", "org/older"])


class RemoveCachedModelTests(unittest.TestCase):
    """Removing a model deletes its folder and nothing else."""

    def test_the_folder_is_removed_and_the_size_reported(self):
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": b"{}", "model.safetensors": b"x" * 99})
            (folder / "blobs" / "shard.incomplete").write_bytes(b"y" * 10)
            other = lay_out(root, "org/other", {"config.json": b"{}"})

            freed = remove_cached_model(OLMO, Path(root))

            self.assertFalse(folder.exists())
            self.assertTrue(other.is_dir())
        # The 40-byte refs/main file goes too, so it counts.
        self.assertEqual(freed, 2 + 99 + 10 + len(COMMIT))

    def test_every_revision_counts_toward_the_space_freed(self):
        # Without symlinks the hub keeps each revision's files in its own
        # snapshot folder, and deleting the repo folder takes them all.
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": b"{}"})
            old = folder / "snapshots" / ("0" * 40)
            old.mkdir()
            (old / "model.safetensors").write_bytes(b"x" * 500)
            (folder / "snapshots" / COMMIT / "model.safetensors").write_bytes(b"y" * 70)

            self.assertEqual(
                remove_cached_model(OLMO, Path(root)), 2 + 500 + 70 + len(COMMIT)
            )

    def test_the_hubs_lock_folder_is_left_for_other_processes(self):
        # Released lock files are harmless; a deleted one that another
        # process was waiting on would let two writers in.
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": b"{}"})
            lock = Path(root) / ".locks" / folder.name
            lock.mkdir(parents=True)
            (lock / "blob.lock").write_text("")

            remove_cached_model(OLMO, Path(root))

            self.assertFalse(folder.exists())
            self.assertTrue((lock / "blob.lock").is_file())

    def test_a_lock_held_by_another_process_refuses_the_removal(self):
        with tempfile.TemporaryDirectory() as root:
            folder = lay_out(root, OLMO, {"config.json": b"{}"})
            lock_dir = Path(root) / ".locks" / folder.name
            lock_dir.mkdir(parents=True)
            # filelock is reentrant within a process, so the hold has to come
            # from outside it, as the hub's would.
            script = (
                "import time\nfrom filelock import FileLock\n"
                f"lock = FileLock({str(lock_dir / 'blob.lock')!r})\nlock.acquire()\n"
                "print('held', flush=True)\ntime.sleep(30)"
            )
            other = subprocess.Popen(
                [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
            )
            try:
                self.assertEqual(other.stdout.readline().strip(), "held")
                with self.assertRaises(model_cache.ModelDownloading) as caught:
                    remove_cached_model(OLMO, Path(root))
                self.assertIn("another process", str(caught.exception))
                self.assertTrue(folder.is_dir())
            finally:
                other.kill()
                other.wait()

            # Once the other process is gone the same lock file is no bar.
            remove_cached_model(OLMO, Path(root))
            self.assertFalse(folder.exists())

    def test_a_model_that_is_not_on_disk_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                remove_cached_model(OLMO, Path(root))

    def test_a_malformed_id_is_refused_before_anything_is_touched(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                remove_cached_model("../../etc", Path(root))
            self.assertEqual(list(Path(root).iterdir()), [])


class ManagerRemoveTests(unittest.TestCase):
    """The manager deletes a model only when nothing of its own is using it."""

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.folder = lay_out(
            self.root.name, OLMO, {"config.json": b"{}", "model.safetensors": b"x" * 10}
        )
        self.manager = ModelManager()

    def test_an_idle_model_is_removed_and_the_locks_released(self):
        freed = self.manager.remove(OLMO, Path(self.root.name))

        self.assertFalse(self.folder.exists())
        self.assertEqual(freed, 12 + len(COMMIT))  # files plus the refs/main entry
        self.assertFalse(self.manager._lock.locked())
        self.assertFalse(self.manager._downloads_lock.locked())

    def test_a_removal_notes_the_change_to_the_cache(self):
        # What another tab's switcher reads to learn that the model it is
        # still offering has gone; see cache_revision.
        self.manager.remove(OLMO, Path(self.root.name))

        self.assertEqual(self.manager.cache_revision, 1)

    def test_a_refused_removal_leaves_the_cache_revision_alone(self):
        self.manager.model_id = OLMO
        with self.assertRaises(model_cache.ModelLoaded):
            self.manager.remove(OLMO, Path(self.root.name))

        self.assertEqual(self.manager.cache_revision, 0)

    def test_the_loaded_model_is_refused(self):
        self.manager.model_id = OLMO
        with self.assertRaises(model_cache.ModelLoaded):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())
        self.assertFalse(self.manager._lock.locked())

    def test_a_model_being_downloaded_is_refused(self):
        self.manager.active_downloads[OLMO] = DownloadProgress()
        with self.assertRaises(model_cache.ModelDownloading):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())
        self.assertFalse(self.manager._downloads_lock.locked())

    def test_a_busy_manager_is_refused_without_waiting(self):
        self.manager._lock.acquire()
        self.addCleanup(self.manager._lock.release)
        with self.assertRaises(model_cache.ModelBusy):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())

    def test_every_refusal_is_a_model_in_use(self):
        for error in (model_cache.ModelLoaded, model_cache.ModelDownloading, model_cache.ModelBusy):
            self.assertTrue(issubclass(error, model_cache.ModelInUse))

    def test_a_load_claimed_on_another_thread_is_refused(self):
        # The load's own thread has not reached the model lock yet, so the
        # lock is free and would let the deletion through.
        _model_id, claim = self.manager.reserve_load(OLMO)
        with self.assertRaises(model_cache.ModelBusy):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())
        self.assertFalse(self.manager._lock.locked())
        self.manager.release_load(claim)
        self.manager.remove(OLMO, Path(self.root.name))
        self.assertFalse(self.folder.exists(), "removed once the claim is gone")

    def test_a_malformed_id_is_refused(self):
        with self.assertRaises(ValueError):
            self.manager.remove("nonsense", Path(self.root.name))


class LoadingIdTests(unittest.TestCase):
    """The manager names the model it is loading for as long as the load runs,
    and keeps the loads that overlap apart from each other."""

    def test_the_loading_id_is_set_during_the_load_and_cleared_after(self):
        manager = ModelManager()
        seen = []

        def fake_load(
            model_id, local_path, torch, progress=None, precision="full", kind=TEXT_KIND
        ):
            seen.append((manager.loading_id, manager._lock.locked()))
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            self.assertEqual(manager.load(OLMO, Path("/snap")), "CPU")

        self.assertEqual(seen, [(OLMO, True)])
        self.assertIsNone(manager.loading_id)

    def test_a_load_waiting_for_the_lock_is_already_named(self):
        # A generation holds the lock for as long as its reply takes; the
        # load queued behind it must count as under way from the click.

        manager = ModelManager()
        manager._lock.acquire()
        entered = threading.Event()

        def fake_load(
            model_id, local_path, torch, progress=None, precision="full", kind=TEXT_KIND
        ):
            entered.set()
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            worker = threading.Thread(target=manager.load, args=(OLMO, Path("/snap")))
            worker.start()
            try:
                for _ in range(200):
                    if manager.loading_id == OLMO:
                        break
                    time.sleep(0.005)
                self.assertEqual(manager.loading_id, OLMO)
                self.assertFalse(entered.is_set())
            finally:
                manager._lock.release()
                worker.join(timeout=5)

        self.assertTrue(entered.is_set())
        self.assertIsNone(manager.loading_id)

    def test_a_failed_load_clears_the_loading_id(self):
        manager = ModelManager()

        def fail(
            model_id, local_path, torch, progress=None, precision="full", kind=TEXT_KIND
        ):
            raise RuntimeError("gpu fell over")

        with mock.patch.object(manager, "_load_locked", fail):
            with self.assertRaises(RuntimeError):
                manager.load(OLMO, Path("/snap"))

        self.assertIsNone(manager.loading_id)
        self.assertFalse(manager._lock.locked())

    def test_a_claim_names_the_load_before_it_starts(self):
        # A load that runs on its own thread is under way from the click:
        # the worker names it only once it reaches load(), and a redownload
        # or a removal arriving in between must find it already claimed.
        manager = ModelManager()

        checked_id, _claim = manager.reserve_load(" allenai/Olmo-3-7B-Think ")

        self.assertEqual(checked_id, OLMO)
        self.assertEqual(manager.loading_id, OLMO)
        self.assertTrue(manager.is_loading(OLMO))
        self.assertFalse(manager.is_loading("org/other"))

    def test_two_overlapping_loads_keep_their_own_claims(self):
        manager = ModelManager()
        _first_id, first = manager.reserve_load(OLMO)
        _second_id, second = manager.reserve_load("org/other")

        manager.release_load(first)

        self.assertFalse(manager.is_loading(OLMO))
        self.assertTrue(manager.is_loading("org/other"), "the other load stands")
        manager.release_load(second)
        self.assertIsNone(manager.loading_id)
        manager.release_load(second)
        self.assertIsNone(manager.loading_id, "releasing twice is harmless")

    def test_an_exclusive_claim_is_refused_while_another_load_stands(self):
        # For the callers whose promise is "one load at a time": reading
        # loading_id and then claiming is two steps, and a streaming handler
        # yields between them.
        manager = ModelManager()

        claimed, held = manager.claim_exclusive_load(" allenai/Olmo-3-7B-Think ")

        self.assertIsNone(held)
        checked_id, claim = claimed
        self.assertEqual(checked_id, OLMO)
        self.assertEqual(
            manager.claim_exclusive_load("org/other"),
            (None, model_runtime.LOADING),
        )
        self.assertEqual(
            manager.claim_exclusive_load(OLMO),
            (None, model_runtime.LOADING),
            "not even the same one",
        )
        manager.release_load(claim)
        self.assertIsNotNone(manager.claim_exclusive_load("org/other").claim)

    def test_an_exclusive_claim_is_refused_while_a_load_reads_weights(self):
        manager = ModelManager()
        with manager._reading_weights(OLMO):
            self.assertEqual(
                manager.claim_exclusive_load("org/other"),
                (None, model_runtime.LOADING),
            )

        self.assertIsNotNone(manager.claim_exclusive_load("org/other").claim)

    def test_an_exclusive_claim_is_refused_while_a_reply_is_running(self):
        # The other half of the switcher's promise. A load admitted beside a
        # generation does not run beside it - it waits on the model lock and
        # then unloads the model that was producing the tokens.
        manager = ModelManager()
        self.assertTrue(manager.reserve_generation())

        self.assertEqual(
            manager.claim_exclusive_load(OLMO), (None, model_runtime.GENERATING)
        )

        manager.release_generation()
        self.assertIsNotNone(manager.claim_exclusive_load(OLMO).claim)

    def test_a_generation_is_refused_while_a_load_is_claimed(self):
        # The mirror image, and the reason the switcher can stop asking
        # whether anything is generating: a reply that started after the
        # load was claimed would wait out the load and answer from whatever
        # it brought in.
        manager = ModelManager()
        _checked_id, claim = manager.claim_exclusive_load(OLMO).claim

        self.assertFalse(manager.reserve_generation())

        manager.release_load(claim)
        self.assertTrue(manager.reserve_generation())
        manager.release_generation()

    def test_a_generation_is_refused_while_an_ordinary_load_is_claimed(self):
        # Not only the exclusive ones: the Models page's buttons claim
        # through reserve_load, and a reply must not slip past those either.
        manager = ModelManager()
        _checked_id, claim = manager.reserve_load(OLMO)

        self.assertFalse(manager.reserve_generation())

        manager.release_load(claim)
        self.assertTrue(manager.reserve_generation())
        manager.release_generation()

    def test_a_generation_is_refused_while_a_load_reads_weights(self):
        manager = ModelManager()
        with manager._reading_weights(OLMO):
            self.assertFalse(manager.reserve_generation())

        self.assertTrue(manager.reserve_generation())
        manager.release_generation()

    def test_a_refused_claim_names_what_has_the_model(self):
        # The refusal and the reason are decided in one step, because every
        # caller that turns a refusal into words - the API, the batch, Score
        # text, Inspect layers, the image page, an extension - would
        # otherwise read a load that had ended in between and say the wrong
        # one of "a reply is running" and "a model is loading".
        manager = ModelManager()
        _checked_id, claim = manager.reserve_load(OLMO)

        self.assertEqual(manager.claim_generation(), model_runtime.LOADING)
        self.assertEqual(manager.occupant, model_runtime.LOADING)

        manager.release_load(claim)
        self.assertIsNone(manager.claim_generation(), "the slot was free")
        self.assertEqual(manager.occupant, model_runtime.GENERATING)
        self.assertEqual(manager.claim_generation(), model_runtime.GENERATING)
        manager.release_generation()
        self.assertIsNone(manager.occupant)

    def test_a_load_reading_weights_is_named_as_a_load(self):
        manager = ModelManager()
        with manager._reading_weights(OLMO):
            self.assertEqual(manager.claim_generation(), model_runtime.LOADING)
            self.assertEqual(manager.occupant, model_runtime.LOADING)

    def test_a_refused_claim_does_not_take_the_slot(self):
        # Naming the reason must not leave the slot taken on the way out: a
        # non-blocking acquire that is never released wedges the chat page.
        manager = ModelManager()
        _checked_id, claim = manager.reserve_load(OLMO)

        self.assertEqual(manager.claim_generation(), model_runtime.LOADING)

        manager.release_load(claim)
        self.assertFalse(manager.busy)
        self.assertTrue(manager.reserve_generation())
        manager.release_generation()

    def test_a_refused_generation_does_not_take_the_slot(self):
        # A non-blocking acquire that is never reached cannot be released,
        # and a slot left taken would wedge the chat page for good.
        manager = ModelManager()
        _checked_id, claim = manager.reserve_load(OLMO)

        self.assertFalse(manager.reserve_generation())

        self.assertFalse(manager.busy)
        manager.release_load(claim)

    def test_only_one_of_a_crowd_of_loads_and_replies_is_admitted(self):
        # Every path a reader can start work on the model by, raced against
        # each other on real threads: one winner, whichever it is, and the
        # rest told no. The two reservations answer under the same lock, so
        # there is no interleaving in which both say yes.
        manager = ModelManager()
        start = threading.Barrier(8)
        won: list[str] = []
        lock = threading.Lock()

        def switch():
            start.wait()
            claimed = manager.claim_exclusive_load("org/one").claim
            if claimed is not None:
                with lock:
                    won.append("load")

        def reply():
            start.wait()
            if manager.reserve_generation():
                with lock:
                    won.append("reply")

        threads = [
            threading.Thread(target=switch if turn % 2 else reply) for turn in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(won), 1, f"admitted {won}")

    def test_a_load_does_not_clear_a_claim_it_did_not_take(self):
        # One load finishing used to leave the other looking idle while it
        # waited for the lock, which is the window a removal needs.
        from unittest import mock

        manager = ModelManager()
        _model_id, waiting = manager.reserve_load(OLMO)

        with mock.patch.object(
            manager, "_load_locked", lambda *args, **kwargs: "CPU"
        ):
            manager.load("org/other", Path("/snap"))

        self.assertTrue(manager.is_loading(OLMO), "still claimed by its own worker")
        manager.release_load(waiting)
        self.assertIsNone(manager.loading_id)

    def test_a_malformed_id_is_refused_before_the_lock(self):
        manager = ModelManager()
        with self.assertRaises(ValueError):
            manager.load("nonsense", Path("/snap"))
        self.assertIsNone(manager.loading_id)

    def test_one_load_finishing_does_not_cancel_another_still_running(self):
        # "Load cached" and "Download and load" are separate Gradio events,
        # so they get a worker each and can both be inside load() at once,
        # the second waiting on the model lock. The load that finishes first
        # must leave the other's name in place: otherwise the chat badge
        # goes back to "No model loaded" halfway through a load, and offers
        # the reader a button to start one more.

        manager = ModelManager()
        second = "org/second"
        in_first = threading.Event()
        in_second = threading.Event()
        let_first_finish = threading.Event()
        let_second_finish = threading.Event()

        def fake_load(
            model_id, local_path, torch, progress=None, precision="full", kind=TEXT_KIND
        ):
            if model_id == OLMO:
                in_first.set()
                let_first_finish.wait(5)
            else:
                in_second.set()
                let_second_finish.wait(5)
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            first_worker = threading.Thread(
                target=manager.load, args=(OLMO, Path("/snap"))
            )
            first_worker.start()
            try:
                self.assertTrue(in_first.wait(5))
                second_worker = threading.Thread(
                    target=manager.load, args=(second, Path("/snap"))
                )
                second_worker.start()
                try:
                    for _ in range(200):
                        if manager.is_loading(second):
                            break
                        time.sleep(0.005)
                    # Both count as under way, and the one that got there
                    # first is the one the badge names.
                    self.assertTrue(manager.is_loading(second))
                    self.assertTrue(manager.is_loading(OLMO))
                    self.assertEqual(manager.loading_id, OLMO)
                    self.assertFalse(in_second.is_set())

                    let_first_finish.set()
                    first_worker.join(timeout=5)
                    self.assertTrue(in_second.wait(5))
                    # The finished load took only its own name away.
                    self.assertEqual(manager.loading_id, second)
                    self.assertFalse(manager.is_loading(OLMO))
                finally:
                    let_second_finish.set()
                    second_worker.join(timeout=5)
            finally:
                let_first_finish.set()
                first_worker.join(timeout=5)

        self.assertIsNone(manager.loading_id)
        self.assertFalse(manager.is_loading(second))
        self.assertFalse(manager._lock.locked())

    def test_the_load_holding_the_lock_is_named_whatever_order_they_claimed_in(self):
        # Loads claim themselves before they wait for the model lock, so
        # claim order is arrival order, and arrival order is not promised to
        # be the order the lock is handed out in: a thread can be set aside
        # between the two steps. The badge has to name the load that is
        # really reading weights, not the one that claimed last.

        manager = ModelManager()
        reading = threading.Event()
        let_it_finish = threading.Event()

        def fake_load(
            model_id, local_path, torch, progress=None, precision="full", kind=TEXT_KIND
        ):
            reading.set()
            let_it_finish.wait(5)
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            worker = threading.Thread(target=manager.load, args=(OLMO, Path("/snap")))
            worker.start()
            try:
                self.assertTrue(reading.wait(5))
                # A later click lands its claim while the first load reads.
                _later_id, later = manager.reserve_load("org/second")
                self.assertEqual(manager.loading_id, OLMO)
                # Both are claimed, so neither model's files may be touched.
                self.assertTrue(manager.is_loading(OLMO))
                self.assertTrue(manager.is_loading("org/second"))
            finally:
                let_it_finish.set()
                worker.join(timeout=5)

        # With the lock free, the load waiting for it is named again, so the
        # badge does not fall back to "No model loaded" while it runs.
        self.assertEqual(manager.loading_id, "org/second")
        manager.release_load(later)
        self.assertIsNone(manager.loading_id)
        self.assertFalse(manager._lock.locked())


PARTIAL = cached(
    "org/partial",
    status=CacheStatus(
        cached_bytes=100, partial_files=1, partial_bytes=50, missing_files=(MODEL_WEIGHTS,)
    ),
    architecture=None,
    dtype=None,
)

UNSUPPORTED = cached(
    # A CTranslate2 export: whole on disk and not a model of either kind.
    # Not a diffusers pipeline, which is now one of the kinds that load.
    "org/olmo-ct2",
    status=CacheStatus(cached_bytes=5_500_000_000, kind=""),
    architecture=None,
    dtype=None,
)

PIPELINE = cached(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    status=CacheStatus(cached_bytes=5_500_000_000, kind=IMAGE_KIND),
    architecture="StableDiffusionPipeline",
    dtype=None,
)


class SortCachedModelsTests(unittest.TestCase):
    OLD_SMALL = cached("zeta/old-small", updated=1.0, status=CacheStatus(cached_bytes=10))
    NEW_LARGE = cached("Alpha/new-large", updated=3.0, status=CacheStatus(cached_bytes=300))
    MID = cached("mid/model", updated=2.0, status=CacheStatus(cached_bytes=20))
    MODELS = [OLD_SMALL, NEW_LARGE, MID]

    def ids(self, order):
        return [entry.model_id for entry in sort_cached_models(self.MODELS, order)]

    def test_each_order_lists_the_models_its_way(self):
        self.assertEqual(
            self.ids("Newest first"), ["Alpha/new-large", "mid/model", "zeta/old-small"]
        )
        self.assertEqual(self.ids("Name"), ["Alpha/new-large", "mid/model", "zeta/old-small"])
        self.assertEqual(
            self.ids("Largest first"), ["Alpha/new-large", "mid/model", "zeta/old-small"]
        )
        self.assertEqual(
            self.ids("Smallest first"), ["zeta/old-small", "mid/model", "Alpha/new-large"]
        )

    def test_name_order_ignores_case(self):
        models = [cached("b/one"), cached("A/two"), cached("a/one")]
        self.assertEqual(
            [e.model_id for e in sort_cached_models(models, "Name")],
            ["a/one", "A/two", "b/one"],
        )

    def test_an_unknown_order_falls_back_to_newest_first(self):
        self.assertEqual(self.ids(None), self.ids("Newest first"))
        self.assertEqual(self.ids("sideways"), self.ids("Newest first"))

    def test_the_input_is_left_alone(self):
        sort_cached_models(self.MODELS, "Name")
        self.assertEqual(self.MODELS, [self.OLD_SMALL, self.NEW_LARGE, self.MID])


class MyModelsPaneTests(unittest.TestCase):
    """What My Models lists, and what choosing a model does."""

    def setUp(self):
        self.entries = [cached(OLMO), PARTIAL]
        self.manager = ModelManager()
        originals = (runtime.MANAGER, models_page.list_cached_models, models_page.cache_root)
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list(self.entries)
        models_page.cache_root = lambda: Path("/cache")
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(models_page, "cache_root", originals[2])
        )

    def test_every_cached_model_is_listed_with_its_size(self):
        radio, detail, summary = app.refresh_my_models(None)

        self.assertEqual(
            radio["choices"],
            [
                (f"{OLMO} · 15.0 GB", OLMO),
                ("org/partial · 150 B · incomplete", "org/partial"),
            ],
        )
        self.assertIsNone(radio["value"])
        self.assertEqual(detail, app.NO_CACHED_MODEL_SELECTED)
        self.assertIn("2 models", summary)
        self.assertIn("15.0 GB", summary)
        self.assertIn("/cache", summary)

    def test_the_list_follows_the_chosen_sort_order(self):
        self.entries = [cached("b/big"), PARTIAL, cached("a/small", status=CacheStatus(cached_bytes=1))]

        by_name, _, _ = app.refresh_my_models(None, "Name")
        by_size, _, _ = app.refresh_my_models(None, "Largest first")
        smallest, _, _ = app.refresh_my_models(None, "Smallest first")

        self.assertEqual([v for _, v in by_name["choices"]], ["a/small", "b/big", "org/partial"])
        self.assertEqual([v for _, v in by_size["choices"]], ["b/big", "org/partial", "a/small"])
        self.assertEqual([v for _, v in smallest["choices"]], ["a/small", "org/partial", "b/big"])

    def test_the_kind_filter_narrows_the_list_to_one_kind(self):
        self.entries = [cached(OLMO), PIPELINE, MLX]

        every, _, all_summary = app.refresh_my_models(None, "Name", None, app.ALL_KINDS)
        images, _, image_summary = app.refresh_my_models(
            None, "Name", None, model_cache.IMAGE_KIND
        )
        texts, _, _ = app.refresh_my_models(None, "Name", None, model_cache.TEXT_KIND)
        mlx, _, _ = app.refresh_my_models(None, "Name", None, model_cache.MLX_KIND)

        self.assertEqual(len(every["choices"]), 3)
        self.assertEqual([v for _, v in images["choices"]], [PIPELINE.model_id])
        self.assertEqual([v for _, v in texts["choices"]], [OLMO])
        self.assertEqual([v for _, v in mlx["choices"]], [MLX.model_id])
        # The count and the size stay about the cache; only the lead says
        # what is on screen, because the disk figure is about the folder.
        self.assertNotIn("Showing", all_summary)
        self.assertIn("3 models", all_summary)
        self.assertIn("Showing 1 image", image_summary)
        self.assertIn("3 models", image_summary)

    def test_a_kind_with_nothing_downloaded_says_where_to_find_one(self):
        # What a reader who pressed Choose an image model with no pipeline in
        # the cache sees: the reason the list is empty, and the way out.
        self.entries = [cached(OLMO)]

        radio, detail, summary = app.refresh_my_models(
            None, "Name", None, model_cache.IMAGE_KIND
        )

        self.assertEqual(radio["choices"], [])
        self.assertEqual(detail, "")
        self.assertIn("No image models", summary)
        self.assertIn("**Discover models**", summary)

    def test_a_filter_that_hides_the_selected_row_drops_the_selection(self):
        self.entries = [cached(OLMO), PIPELINE]

        radio, detail, _ = app.refresh_my_models(
            OLMO, "Name", None, model_cache.IMAGE_KIND
        )

        self.assertIsNone(radio["value"])
        self.assertEqual(detail, app.NO_CACHED_MODEL_SELECTED)

    def test_a_model_of_no_kind_is_listed_under_all_kinds_only(self):
        # An unsupported snapshot wears no kind, so no kind claims it; the
        # unfiltered list still has to show it, since it is on disk.
        self.entries = [UNSUPPORTED]

        every, _, _ = app.refresh_my_models(None, "Name", None, app.ALL_KINDS)
        texts, _, _ = app.refresh_my_models(None, "Name", None, model_cache.TEXT_KIND)

        self.assertEqual([v for _, v in every["choices"]], [UNSUPPORTED.model_id])
        self.assertEqual(texts["choices"], [])

    def test_the_name_filter_keeps_ids_holding_every_word_in_any_case(self):
        qwen_small = cached("Qwen/Qwen3-0.6B")
        qwen_large = cached("mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.entries = [cached(OLMO), qwen_small, qwen_large]

        qwen, _, summary = app.refresh_my_models(None, "Name", None, app.ALL_KINDS, "qwen")
        seven, _, _ = app.refresh_my_models(None, "Name", None, app.ALL_KINDS, " QWEN  7b ")
        blank, _, blank_summary = app.refresh_my_models(None, "Name", None, app.ALL_KINDS, "  ")

        self.assertEqual(
            [v for _, v in qwen["choices"]], [qwen_large.model_id, qwen_small.model_id]
        )
        self.assertEqual([v for _, v in seven["choices"]], [qwen_large.model_id])
        self.assertEqual(len(blank["choices"]), 3)
        self.assertIn("Showing 2 matching `qwen`", summary)
        self.assertIn("3 models", summary)
        self.assertNotIn("Showing", blank_summary)

    def test_the_name_filter_narrows_within_the_chosen_kind(self):
        self.entries = [cached("Qwen/Qwen3-0.6B"), PIPELINE, MLX]

        radio, _, summary = app.refresh_my_models(
            None, "Name", None, model_cache.IMAGE_KIND, PIPELINE.model_id.split("/")[-1]
        )
        _, _, empty = app.refresh_my_models(
            None, "Name", None, model_cache.IMAGE_KIND, "qwen"
        )

        self.assertEqual([v for _, v in radio["choices"]], [PIPELINE.model_id])
        self.assertIn("Showing 1 image matching", summary)
        self.assertIn("No image models matching `qwen`", empty)
        self.assertIn("**Discover models**", empty)

    def test_a_name_that_hides_the_selected_row_drops_the_selection(self):
        self.entries = [cached(OLMO), cached("Qwen/Qwen3-0.6B")]

        radio, detail, _ = app.refresh_my_models(OLMO, "Name", None, app.ALL_KINDS, "qwen")

        self.assertIsNone(radio["value"])
        self.assertEqual(detail, app.NO_CACHED_MODEL_SELECTED)

    def test_the_unfiltered_list_is_what_no_kind_at_all_gives(self):
        # demo.load and the tests that predate the filter pass no kind.
        self.entries = [cached(OLMO), PIPELINE]

        default, _, _ = app.refresh_my_models(None, "Name")
        every, _, _ = app.refresh_my_models(None, "Name", None, app.ALL_KINDS)

        self.assertEqual(default["choices"], every["choices"])

    def test_an_incomplete_label_ends_in_the_word_the_stylesheet_looks_for(self):
        # The CSS tints options whose label carries "· incomplete", the only
        # hook Gradio's Radio gives a stylesheet.
        radio, _, _ = app.refresh_my_models(None)
        labels = dict((v, k) for k, v in radio["choices"])
        self.assertIn("· incomplete", labels["org/partial"])
        self.assertNotIn("incomplete", labels[OLMO])
        self.assertIn('[data-testid*="· incomplete"]', app.CSS)

    def test_a_whole_repo_of_another_kind_is_listed_as_unsupported(self):
        self.entries = [UNSUPPORTED]

        radio, _, _ = app.refresh_my_models(None)
        _, detail = app.select_my_model(UNSUPPORTED.model_id)

        self.assertEqual(
            radio["choices"][0][0], f"{UNSUPPORTED.model_id} · 5.5 GB · unsupported"
        )
        self.assertIn("Unsupported", detail)
        self.assertIn("not a model ChatLab loads", detail)
        self.assertNotIn("Incomplete", detail)
        self.assertNotIn("Download and load", detail)

    def test_an_image_pipeline_is_listed_as_one_and_points_at_the_images_page(self):
        self.entries = [PIPELINE]

        radio, _, _ = app.refresh_my_models(None)
        _, detail = app.select_my_model(PIPELINE.model_id)

        self.assertEqual(
            radio["choices"][0][0], f"{PIPELINE.model_id} · 5.5 GB · image"
        )
        self.assertIn("Ready to load", detail)
        self.assertIn("**Images** page", detail)
        self.assertIn("**Kind:** image model", detail)
        self.assertNotIn("Unsupported", detail)

    def test_an_image_row_carries_its_kind_and_its_fit_verdict(self):
        # #45 put a fit verdict on every row and this branch put a kind on
        # the image ones; a row has to say both.
        from chatlab.device_memory import FITS, Fit

        label = models_page.cached_model_label(PIPELINE, Fit(FITS))

        self.assertIn("· image", label)
        self.assertIn("· fits", label)
        self.assertLess(label.index("· image"), label.index("· fits"))

    def test_an_image_pipeline_is_judged_at_the_precision_it_is_measured_at(self):
        # A pipeline is sized whole whatever the radio says - the Metal
        # quantizer is Transformers' own and the load clears the choice for
        # one - so carrying the bits into the verdict would put a quantized
        # label on a full-size figure.
        profile = device_memory.DeviceProfile(
            backend="mps",
            dtype="float16",
            total=48 * 1024**3,
            available=40 * 1024**3,
        )
        measured = []

        def estimate(snapshot, dtype, bits, kind):
            measured.append((kind, bits))
            return 5 * 1024**3

        with mock.patch.object(models_page, "estimate_snapshot_bytes", estimate):
            with mock.patch.object(models_page, "snapshot_folder", lambda path: path):
                pipeline = models_page.cached_fit(PIPELINE, "4-bit", profile)
                text = models_page.cached_fit(cached("org/text"), "4-bit", profile)

        self.assertEqual(measured, [(IMAGE_KIND, None), (TEXT_KIND, 4)])
        self.assertIn("of full 16-bit weights", pipeline.note)
        self.assertIn("of 4-bit weights", text.note)

    def test_a_text_model_is_not_flagged_with_a_kind_in_the_list(self):
        # Text models are the majority and the default: a word on every row
        # to distinguish the exception would put one on every row.
        radio, _, _ = app.refresh_my_models(None)
        labels = dict((value, label) for label, value in radio["choices"])

        self.assertNotIn("· image", labels[OLMO])
        self.assertNotIn("· text", labels[OLMO])

    def test_a_refresh_keeps_the_selection(self):
        radio, detail, _ = app.refresh_my_models("org/partial")

        self.assertEqual(radio["value"], "org/partial")
        self.assertIn("Incomplete", detail)
        self.assertIn("the model weights are missing", detail)
        self.assertIn("Download and load", detail)

    def test_a_refresh_falls_back_to_the_loaded_model(self):
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS)"

        radio, detail, _ = app.refresh_my_models("gone/model")

        self.assertEqual(radio["value"], OLMO)
        self.assertEqual(radio["choices"][0][0], f"{OLMO} · 15.0 GB · loaded")
        self.assertIn("Loaded now", detail)
        self.assertIn("Apple Metal (MPS)", detail)

    def test_an_empty_cache_says_so(self):
        self.entries = []

        radio, detail, summary = app.refresh_my_models(OLMO)

        self.assertEqual(radio["choices"], [])
        self.assertIsNone(radio["value"])
        self.assertEqual(detail, "")
        self.assertIn("No models", summary)
        self.assertIn("Discover models", summary)

    def test_refresh_does_not_replace_an_uncached_typed_id_with_the_loaded_model(self):
        self.manager.model_id = OLMO
        radio, _, _ = app.refresh_my_models(None, "Name", model_id="org/not-downloaded")
        self.assertIsNone(radio["value"])
        self.assertEqual(app.chosen_model("org/not-downloaded", radio["value"]), "org/not-downloaded")

    def test_refresh_keeps_a_picked_row_ahead_of_a_stale_textbox(self):
        radio, _, _ = app.refresh_my_models("org/partial", "Name", model_id=OLMO)
        self.assertEqual(radio["value"], "org/partial")

    def test_choosing_a_model_fills_the_id_box_and_describes_it(self):
        box, detail = app.select_my_model(OLMO)

        self.assertEqual(box["value"], OLMO)
        self.assertIn("Ready to load", detail)
        self.assertIn("15.0 GB cached", detail)
        self.assertIn("12 in the current snapshot", detail)
        self.assertIn("Olmo3ForCausalLM (bfloat16)", detail)
        self.assertIn(COMMIT[:7], detail)
        self.assertIn("2023-11-1", detail)  # the fixed ``updated`` stamp, any zone
        self.assertIn("/cache/models--allenai--Olmo-3-7B-Think", detail)

    def test_a_model_that_left_the_cache_is_reported(self):
        box, detail = app.select_my_model("gone/model")

        self.assertEqual(box, gr.skip())
        self.assertIn("no longer in the cache", detail)

    def test_choosing_nothing_leaves_the_id_box_alone(self):
        self.assertEqual(
            app.select_my_model(None), (gr.skip(), app.NO_CACHED_MODEL_SELECTED)
        )


class ModelFitTests(unittest.TestCase):
    """Whether a model would load, said before the button is pressed."""

    GB = 1024**3
    CONFIG = json.dumps({"architectures": ["Olmo3ForCausalLM"], "dtype": "float16"})

    def cache(self, sizes: dict[str, int]) -> Path:
        """A cache holding one model per entry, each weighing what it says.

        Sparse files: the fit check reads sizes, never contents, and the
        alternative is writing tens of gigabytes to a temporary directory.
        """

        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        for model_id, size in sizes.items():
            folder = lay_out(
                root,
                model_id,
                {"config.json": self.CONFIG.encode(), "model.safetensors": b""},
            )
            weights = folder / "blobs" / "blob1"
            with weights.open("r+b") as handle:
                handle.truncate(size)
        return root

    def setUp(self):
        self.manager = ModelManager()
        self.profile = roomy(self)
        root = self.cache({OLMO: 15 * self.GB, "org/huge": 200 * self.GB})
        originals = (runtime.MANAGER, models_page.list_cached_models, models_page.cache_root)
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list_cached_models(root)
        models_page.cache_root = lambda: root
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(models_page, "cache_root", originals[2])
        )

    def labels(self, precision=None):
        radio, _, _ = app.refresh_my_models(None, "Name", precision)
        return dict((value, label) for label, value in radio["choices"])

    def test_each_model_is_listed_with_the_verdict_a_load_would_get(self):
        labels = self.labels()

        self.assertIn("· fits", labels[OLMO])
        self.assertIn("· won't fit", labels["org/huge"])

    def test_a_quantized_precision_shrinks_what_has_to_fit(self):
        # 15 GB of half-precision weights against a machine with 18 GB free:
        # too big whole once the 4 GB reserve is counted, comfortable at four
        # bits. This is what makes the weight precision radio the first thing
        # to try.
        roomy(self, total_gb=24, available_gb=18)

        self.assertIn("· tight", self.labels("full")[OLMO])
        self.assertIn("· fits", self.labels("4-bit")[OLMO])

    def test_a_quantized_precision_is_ignored_where_it_would_not_apply(self):
        # 8-bit and 4-bit weights need Apple Metal; a load elsewhere reads
        # the checkpoint whole whatever the radio says, so the verdict does
        # too rather than promising room the load will not find.
        roomy(self, total_gb=24, available_gb=18, backend="cpu", dtype="float16")

        self.assertIn("· tight", self.labels("4-bit")[OLMO])

    def test_a_device_not_read_yet_is_judged_at_full_weights(self):
        # Of the two ways to be wrong for the few seconds before the device
        # is read, saying a model is tight when 4-bit would have fitted costs
        # a reader nothing; saying it fits when the load will refuse it is
        # the disagreement these verdicts exist to prevent. The list is
        # repainted once the device is known - see the next test.
        roomy(self, total_gb=24, available_gb=18, backend=None, dtype=None)

        self.assertIn("· tight", self.labels("4-bit")[OLMO])

    def test_the_verdicts_are_repainted_once_the_device_is_read(self):
        # The page is painted before the background import finishes, so the
        # first verdicts are given without knowing the device. The badge's
        # timer corrects them once, and then leaves the list alone.
        roomy(self, total_gb=24, available_gb=18, backend=None, dtype=None)
        original = models_page.imported_torch
        models_page.imported_torch = lambda: None
        self.addCleanup(lambda: setattr(models_page, "imported_torch", original))

        # Nothing to correct yet: torch is still importing.
        self.assertEqual(
            app.refresh_after_device(False, None, "Name", "4-bit"), (gr.skip(),) * 7
        )

        models_page.imported_torch = lambda: object()
        radio, _detail, _summary, _results, _search_detail, _selected, known = (
            app.refresh_after_device(False, None, "Name", "4-bit")
        )

        self.assertTrue(known)
        self.assertIn("· tight", dict((v, k) for k, v in radio["choices"])[OLMO])
        # And once it has run, it never runs again.
        self.assertEqual(
            app.refresh_after_device(True, None, "Name", "4-bit"), (gr.skip(),) * 7
        )

    def test_a_search_run_before_the_device_was_read_is_repainted_too(self):
        held = {
            "org/small": hub_search.HubModel(
                model_id="org/small", parameters=1_000_000_000
            )
        }
        roomy(self, total_gb=24, available_gb=18, backend=None, dtype=None)
        original = models_page.imported_torch
        models_page.imported_torch = lambda: object()
        self.addCleanup(lambda: setattr(models_page, "imported_torch", original))

        _radio, _detail, _summary, results, _search, _selected, known = (
            app.refresh_after_device(
                False, None, "Name", "full", app.ALL_KINDS, None, None, None, held
            )
        )

        self.assertTrue(known)
        self.assertEqual(cells(results, "Fit"), ["fits"])

    def test_the_selected_model_says_what_the_verdict_rests_on(self):
        _box, detail = app.select_my_model(OLMO, "full")

        self.assertIn("Memory", detail)
        self.assertIn("15.0 GB of full 16-bit weights", detail)
        self.assertIn("40.0 GB", detail)

    def test_the_verdict_names_the_precision_it_was_measured_at(self):
        # The figure moves several-fold with the radio, so a note that left
        # the precision out would look as though it had changed by itself.
        _box, detail = app.select_my_model(OLMO, "4-bit")

        self.assertIn("of 4-bit weights", detail)
        self.assertNotIn("15.0 GB", detail)

    def test_a_precision_this_device_ignores_is_not_claimed_in_the_verdict(self):
        # A quantized choice is honoured on Apple Metal alone, and the load
        # clears it everywhere else. Naming what the load will really do is
        # how a reader on a graphics card learns their choice changed nothing.
        roomy(self, backend="cuda", dtype="bfloat16")

        _box, detail = app.select_my_model(OLMO, "4-bit")

        self.assertIn("15.0 GB of full 16-bit weights", detail)

    def test_a_replacement_is_judged_after_the_loaded_model_is_given_back(self):
        # A load unloads first and only then checks whether the next model
        # fits, so the weights on the device now are not in the way of the
        # model that would replace them. Without giving them back, a full
        # machine marks every alternative tight and the button loads them
        # anyway.
        roomy(self, total_gb=24, available_gb=2)
        self.assertIn("· tight", self.labels()[OLMO])

        roomy(self, total_gb=24, available_gb=2, held_gb=18)
        self.assertIn("· fits", self.labels()[OLMO])

    def test_a_model_on_the_cpu_is_given_back_by_the_loads_own_estimate(self):
        # Host memory keeps no allocator figure, and a CPU model's weights
        # are anonymous memory, which the availability estimate deliberately
        # does not count. The load's own estimate stands in, or every
        # alternative would stay tight on a machine that would load them.
        # The dtype is left at half precision so the estimate under test is
        # the reclaim, not the CPU's own float32 conversion.
        roomy(self, total_gb=24, available_gb=2, backend="cpu", dtype="float16")
        self.manager.model_id = "org/other"

        self.assertIn("· tight", self.labels()[OLMO])

        self.manager.loaded_bytes = 18 * self.GB

        self.assertIn("· fits", self.labels()[OLMO])

    def test_the_model_in_memory_is_not_judged_again(self):
        # It fits: it is there. Judging it against what is left free would
        # call the loaded model tight.
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS)"
        roomy(self, total_gb=24, available_gb=2)

        label = self.labels()[OLMO]
        self.assertIn("· loaded", label)
        self.assertNotIn("tight", label)

    def test_the_verdicts_are_the_words_the_stylesheet_looks_for(self):
        # The CSS tints options by their label text, the only hook Gradio's
        # Radio gives a stylesheet, so the words and the selectors have to
        # stay spelled the same. "unsupported" is greyed by the same rule as
        # "won't fit": both mean the row will not load.
        for verdict in ("· won't fit", "· tight", "· unsupported"):
            self.assertIn(f'[data-testid*="{verdict}"]', app.CSS)

    def test_a_row_that_fits_is_left_untinted(self):
        # Most of the list fits, so tinting it would leave nothing to stand
        # out; only the rows that need a second look are coloured.
        self.assertNotIn('[data-testid*="· fits"]', app.CSS)

    def test_a_reload_at_another_precision_is_judged_again(self):
        # Load cached on the model in memory is how a new precision is
        # applied, so a reader who has moved that radio is asking about a
        # load that has not happened - and a model that fits at four bits
        # may not fit whole.
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS), 4-bit weights"
        self.manager.precision = "4-bit"
        roomy(self, total_gb=24, available_gb=18)

        # The precision it is loaded at: nothing to ask.
        self.assertNotIn("tight", self.labels("4-bit")[OLMO])
        # Whole weights would not fit beside the 4 GB reserve.
        self.assertIn("· tight", self.labels("full")[OLMO])

    def test_a_reload_where_precision_cannot_apply_is_not_judged_again(self):
        # Off Metal the radio changes nothing about the load, so moving it
        # does not turn the loaded model into a question.
        self.manager.model_id = OLMO
        self.manager.precision = "full"
        roomy(self, total_gb=24, available_gb=2, backend="cpu", dtype="float16")

        self.assertNotIn("tight", self.labels("4-bit")[OLMO])

    def test_an_incomplete_model_has_no_size_to_judge(self):
        models_page.list_cached_models = lambda: [PARTIAL]

        label = self.labels()["org/partial"]
        self.assertIn("· incomplete", label)
        for verdict in ("fits", "tight"):
            self.assertNotIn(verdict, label)


class ManageMyModelsTests(unittest.TestCase):
    """Redownloading and removing a model from My Models."""

    def setUp(self):
        self.entries = [cached(OLMO), PARTIAL]
        self.manager = ModelManager()
        self.removed = []
        originals = (
            runtime.MANAGER,
            models_page.list_cached_models,
            model_cache.remove_cached_model,
            models_page.download_model,
        )
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list(self.entries)
        # The manager deletes through the module-level function, so that is
        # what stands in: the manager's own checks stay real.
        model_cache.remove_cached_model = self.remove
        models_page.download_model = self.download
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(model_cache, "remove_cached_model", originals[2])
            or setattr(models_page, "download_model", originals[3])
        )

    def remove(self, model_id, cache_dir=None):
        self.removed.append(model_id)
        return PARTIAL.size_bytes

    def download(self, model_id, hf_token):
        yield f"downloading {model_id} with {hf_token!r}"

    def test_redownload_resumes_the_selected_model(self):
        frames = list(app.redownload_my_model("org/partial", "tok"))
        self.assertEqual(frames, ["downloading org/partial with 'tok'"])

    def test_the_loaded_model_is_not_redownloaded_under_itself(self):
        # A newer revision on disk with the old weights in memory would be
        # listed as loaded: the label goes by ID alone.
        self.manager.model_id = OLMO

        (card,) = list(app.redownload_my_model(OLMO, ""))

        self.assertIn("Model in use", card)
        self.assertIn("Unload", card)
        self.assertFalse(card.startswith("downloading"), card)  # the fake never ran

    def test_a_model_being_loaded_is_not_redownloaded_under_itself(self):
        # model_id is empty for the whole of a load, so the manager names
        # the model it is bringing in separately.
        self.manager.reserve_load("org/partial")

        (card,) = list(app.redownload_my_model("org/partial", ""))

        self.assertIn("Model in use", card)
        self.assertIn("being loaded", card)
        self.assertFalse(card.startswith("downloading"), card)

    def test_a_model_waiting_behind_another_load_is_not_redownloaded_either(self):
        # Two loads can run at once, and only one of them is the one the
        # badge names. The one waiting its turn is still going to read its
        # own files, so a redownload of it has to be refused as well.
        self.manager.reserve_load(OLMO)
        self.manager.reserve_load("org/partial")

        (card,) = list(app.redownload_my_model("org/partial", ""))

        self.assertIn("Model in use", card)
        self.assertIn("being loaded", card)
        self.assertFalse(card.startswith("downloading"), card)

    def test_redownload_with_nothing_selected_says_so(self):
        (card,) = list(app.redownload_my_model(None, ""))
        self.assertIn("Nothing to redownload", card)
        self.assertIn("Select a model", card)

    def test_asking_to_remove_shows_the_question_with_the_size(self):
        status, confirm, question, pending = app.ask_remove_my_model("org/partial")

        self.assertEqual(status, gr.skip())
        self.assertTrue(confirm["visible"])
        self.assertIn("org/partial", question)
        self.assertIn("150 B", question)
        self.assertIn("cannot be undone", question)
        self.assertEqual(pending, "org/partial")

    def test_asking_with_nothing_selected_is_refused(self):
        status, confirm, question, pending = app.ask_remove_my_model(None)

        self.assertIn("Nothing to remove", status)
        self.assertFalse(confirm["visible"])
        self.assertEqual(question, "")
        self.assertIsNone(pending)

    def test_the_loaded_model_cannot_be_removed(self):
        self.manager.model_id = OLMO

        status, confirm, _, pending = app.ask_remove_my_model(OLMO)
        self.assertIn("Unload", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

        status, confirm, pending = app.remove_my_model(OLMO)
        self.assertIn("Unload", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)
        self.assertEqual(self.removed, [])

    def test_a_model_being_downloaded_cannot_be_removed(self):
        self.manager.active_downloads["org/partial"] = DownloadProgress()

        status, confirm, _, _ = app.ask_remove_my_model("org/partial")
        self.assertIn("Still downloading", status)
        self.assertFalse(confirm["visible"])

        status, _, _ = app.remove_my_model("org/partial")
        self.assertIn("Still downloading", status)
        self.assertEqual(self.removed, [])

    def test_a_model_busy_in_memory_cannot_be_removed(self):
        # A load holds the model lock until ``from_pretrained`` returns, and
        # ``model_id`` is only assigned after: the lock is the real guard.
        self.manager._lock.acquire()
        self.addCleanup(self.manager._lock.release)

        status, confirm, _ = app.remove_my_model("org/partial")

        self.assertIn("Model busy", status)
        self.assertIn("idle", status)
        self.assertFalse(confirm["visible"])
        self.assertEqual(self.removed, [])

    def test_a_refused_removal_says_why_in_the_log(self):
        # The card explains itself and then goes. Without these lines the
        # trail held "Removal confirmed" and no account of why the files
        # are still there.
        self.manager._lock.acquire()
        self.addCleanup(self.manager._lock.release)

        with self.assertLogs("chatlab.ui.models_page", level="INFO") as logged:
            app.remove_my_model("org/partial")

        self.assertIn("Removal confirmed for org/partial", logged.output[0])
        self.assertIn("refused: the manager is busy", logged.output[1])

    def test_a_model_that_left_the_cache_is_reported_without_a_question(self):
        status, confirm, _, pending = app.ask_remove_my_model("gone/model")

        self.assertIn("no longer in the cache", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_confirming_removes_the_model_and_reports_the_space_freed(self):
        status, confirm, pending = app.remove_my_model("org/partial")

        self.assertEqual(self.removed, ["org/partial"])
        self.assertIn("Model removed", status)
        self.assertIn("org/partial", status)
        self.assertIn("150 B", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_confirming_with_no_pending_model_removes_nothing(self):
        # The question was withdrawn (another model chosen, or Cancel) before
        # the click landed: nothing is pending, so nothing is deleted.
        status, confirm, pending = app.remove_my_model(None)

        self.assertEqual(self.removed, [])
        self.assertIn("Nothing to remove", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_withdrawing_the_question_forgets_the_model(self):
        confirm, pending = app.hide_remove_confirm()
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_a_removal_that_fails_is_reported(self):
        def refuse(model_id, cache_dir=None):
            raise PermissionError(13, "Permission denied: <blobs>")

        model_cache.remove_cached_model = refuse

        status, confirm, pending = app.remove_my_model("org/partial")

        self.assertIn("Could not remove model", status)
        self.assertIn("Permission denied: &lt;blobs&gt;", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_a_model_gone_before_confirming_is_reported(self):
        def gone(model_id, cache_dir=None):
            raise FileNotFoundError(model_id)

        model_cache.remove_cached_model = gone

        status, _, _ = app.remove_my_model("org/partial")
        self.assertIn("no longer in the cache", status)


INSTRUCT = HubModel(
    model_id="allenai/Olmo-3-7B-Instruct",
    parameters=7_298_011_136,
    downloads=281_405,
    likes=143,
    pipeline_tag="text-generation",
    library="transformers",
    last_modified="2026-06-25",
    license="apache-2.0",
)
GATED = HubModel(model_id="meta-llama/Llama-3.1-8B", gated="manual")


def cells(table, column):
    """One column of a search table update, top to bottom, as the numbers underneath."""

    return list(table["value"].data[column])


class ModelSearchPaneTests(unittest.TestCase):
    """What Model search lists, and what choosing a result does."""

    def setUp(self):
        self.results = [INSTRUCT, GATED]
        self.queries = []
        roomy(self)
        original_search, original_status = models_page.search_hub_models, models_page.cache_status
        models_page.search_hub_models = self.search
        models_page.cache_status = lambda model_id: CacheStatus()
        self.addCleanup(
            lambda: setattr(models_page, "search_hub_models", original_search)
            or setattr(models_page, "cache_status", original_status)
        )

    def search(self, query, hf_token, kind=TEXT_KIND, order="Popular", limit=100):
        self.queries.append((query, hf_token, kind))
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results)

    def test_results_are_tabled_with_size_and_popularity(self):
        table, detail, state, selected = app.search_models("  olmo 3 ", "tok")

        self.assertEqual(self.queries, [("olmo 3", "tok", TEXT_KIND)])
        sent = painted(table)
        self.assertEqual(
            sent["headers"], ["Model", "Params", "Fit", "Downloads", "Likes", "Updated"]
        )
        # The numbers underneath are numbers, so the browser sorts them as
        # such; a result whose parameter count the hub did not give has no
        # size to judge, so it is listed without a verdict.
        self.assertEqual(
            sent["data"],
            [
                ["allenai/Olmo-3-7B-Instruct", 7_298_011_136, "fits", 281_405, 143, "2026-06-25"],
                ["meta-llama/Llama-3.1-8B", None, "", None, None, None],
            ],
        )
        # And they are shown the way the hub shows them.
        self.assertEqual(
            sent["metadata"]["display_value"],
            [
                ["allenai/Olmo-3-7B-Instruct", "7.3B", "fits", "281K", "143", "2026-06-25"],
                ["meta-llama/Llama-3.1-8B", "—", "", "—", "—", "—"],
            ],
        )
        self.assertIsNone(selected)
        self.assertIn("2 results", detail)
        self.assertEqual(set(state), {INSTRUCT.model_id, GATED.model_id})
        # The columns share the width by weight, so the Model column cannot
        # grow to its longest ID and push the last columns out of view.
        self.assertEqual(table["column_widths"], ["38%", "11%", "10%", "14%", "10%", "17%"])

    def test_rows_are_tinted_by_fit(self):
        # Room on the machine for the 7B, but not free right now: tight.
        roomy(self, total_gb=48, available_gb=10, backend="mps", dtype="float16")
        self.results = [
            INSTRUCT,
            HubModel(model_id="org/huge", parameters=500_000_000_000),
            HubModel(model_id="org/small", parameters=100_000_000),
        ]
        table, _, _, _ = app.search_models("", "")

        self.assertEqual(cells(table, "Fit"), ["tight", "won't fit", "fits"])
        styling = painted(table)["metadata"]["styling"]
        self.assertEqual({style for style in styling[0]}, {"color: var(--fit-tight)"})
        self.assertEqual(
            {style for style in styling[1]}, {"color: var(--body-text-color-subdued)"}
        )
        self.assertEqual({style for style in styling[2]}, {""})

    def test_an_empty_query_browses_popular_models(self):
        table, detail, state, _ = app.search_models("   ", "")
        self.assertEqual(self.queries, [("", "", TEXT_KIND)])
        self.assertEqual(len(cells(table, "Model")), 2)
        self.assertIn("Most downloaded first", detail)
        self.assertEqual(len(state), 2)

    def test_recommended_starters_work_offline(self):
        self.results = ConnectionError("offline")
        # Gradio sends None for an untouched textbox on initial page load. An
        # empty query is the one view answered without reaching the Hub.
        table, detail, state, selected = app.search_models(None, "", order="Recommended")
        self.assertEqual(self.queries, [])
        self.assertEqual(len(state), 3)
        self.assertIsNone(selected)
        self.assertIn("offline", detail)
        # Starters carry a download size and no popularity, so their table
        # has that column and not the hub's; the note sits under the name.
        sent = painted(table)
        self.assertEqual(sent["headers"], ["Model", "Params", "Download size", "Fit"])
        self.assertEqual(table["column_widths"], ["51%", "15%", "20%", "14%"])
        self.assertEqual(sent["metadata"]["display_value"][1][2], "1.5 GB")
        self.assertEqual(
            sent["data"][1][0], "Qwen/Qwen3-0.6B\nCompact reasoning — try a thinking model with modest memory needs."
        )
        # A click reports the whole cell, and the note is not part of the ID.
        box, description, chosen = app.select_search_result(
            state, None, picked(sent["data"][1][0])
        )
        self.assertEqual(box["value"], "Qwen/Qwen3-0.6B")
        self.assertEqual(chosen, "Qwen/Qwen3-0.6B")
        self.assertIn("Compact reasoning", description)
        self.assertIn("Full download", description)
        self.assertIn("not this download", description)

    def test_filtering_uses_candidates_beyond_the_first_twenty(self):
        self.results = [
            HubModel(model_id=f"org/huge-{i}", parameters=500_000_000_000)
            for i in range(25)
        ] + [INSTRUCT, GATED]
        table, detail, state, _ = app.search_models("", "", fits_only=True)
        self.assertEqual(cells(table, "Model"), [INSTRUCT.model_id])
        self.assertEqual(len(state), 27)
        self.assertIn("unknown sizes are hidden", detail)
        table, _, _ = models_page.refresh_search_results(None, state, fits_only=False)
        self.assertEqual(len(cells(table, "Model")), 20)
        self.assertEqual(len(self.queries), 1)

    def test_filtered_selection_clears_and_returns_after_precision_change(self):
        roomy(self, total_gb=16, available_gb=10, backend="mps", dtype="float16")
        state = {INSTRUCT.model_id: INSTRUCT, GATED.model_id: GATED}
        table, detail, selected = models_page.refresh_search_results(
            INSTRUCT.model_id, state, "full", True
        )
        self.assertEqual(cells(table, "Model"), [])
        self.assertIsNone(selected)
        self.assertIn("No estimated fits", detail)
        table, _, selected = models_page.refresh_search_results(None, state, "4-bit", True)
        self.assertEqual(cells(table, "Model"), [INSTRUCT.model_id])
        self.assertIsNone(selected)

    def test_a_selection_survives_a_filter_that_keeps_it(self):
        state = {INSTRUCT.model_id: INSTRUCT, GATED.model_id: GATED}
        table, detail, selected = models_page.refresh_search_results(
            INSTRUCT.model_id, state, "full", True
        )
        self.assertEqual(cells(table, "Model"), [INSTRUCT.model_id])
        self.assertEqual(selected, INSTRUCT.model_id)
        self.assertIn("https://huggingface.co/allenai/Olmo-3-7B-Instruct", detail)

    def test_recommended_query_searches_the_hub_under_the_starters(self):
        # A starter matching the query used to end the search there, which
        # hid the rest of the Hub behind a three-model list.
        _, detail, state, _ = app.search_models("olmo", "tok", order="Recommended")
        self.assertEqual(self.queries, [("olmo", "tok", TEXT_KIND)])
        self.assertEqual(
            list(state),
            ["allenai/Olmo-3-7B-Think", INSTRUCT.model_id, GATED.model_id],
        )
        self.assertIn("Starters first, then Hugging Face", detail)

    def test_a_starter_the_hub_also_returns_is_listed_once_with_both_halves(self):
        # The catalog has the note and the download estimate; the search has
        # the popularity and the date. The row keeps all of it.
        self.results = [
            HubModel(
                model_id="allenai/Olmo-3-7B-Think",
                parameters=7_298_011_136,
                downloads=94_210,
                likes=712,
                last_modified="2026-07-02",
            ),
            INSTRUCT,
        ]
        table, _, state, _ = app.search_models("olmo", "", order="Recommended")
        self.assertEqual(list(state), ["allenai/Olmo-3-7B-Think", INSTRUCT.model_id])
        merged = state["allenai/Olmo-3-7B-Think"]
        self.assertIn("ChatLab", merged.summary)
        self.assertEqual(merged.download_bytes, 14_605_886_999)
        self.assertEqual((merged.downloads, merged.likes), (94_210, 712))
        self.assertEqual(merged.last_modified, "2026-07-02")
        sent = painted(table)
        self.assertEqual(
            sent["headers"],
            ["Model", "Params", "Download size", "Fit", "Downloads", "Likes", "Updated"],
        )
        self.assertEqual(sent["metadata"]["display_value"][0][2], "14.6 GB")
        self.assertEqual(sent["metadata"]["display_value"][0][4], "94K")

    def test_a_starter_the_hub_leaves_blank_keeps_the_catalogs_own_facts(self):
        # A repository with no safetensors index has no parameter count in the
        # search, and the hub drops a licence as readily.
        self.results = [HubModel(model_id="allenai/Olmo-3-7B-Think", downloads=12)]
        _, _, state, _ = app.search_models("olmo", "", order="Recommended")
        merged = state["allenai/Olmo-3-7B-Think"]
        self.assertEqual(merged.parameters, 7_298_011_136)
        self.assertEqual(merged.license, "apache-2.0")
        self.assertEqual(merged.downloads, 12)

    def test_recommended_falls_back_to_starters_when_the_hub_is_unreachable(self):
        self.results = ConnectionError("offline")
        table, detail, state, selected = app.search_models("olmo", "", order="Recommended")
        self.assertEqual(self.queries, [("olmo", "", TEXT_KIND)])
        self.assertEqual(list(state), ["allenai/Olmo-3-7B-Think"])
        self.assertIsNone(selected)
        self.assertIn("Starters only", detail)
        self.assertIn("offline", detail)
        self.assertEqual(len(cells(table, "Model")), 1)

    def test_recommended_query_with_no_starter_match_searches_the_hub(self):
        _, detail, state, _ = app.search_models("gemma", "tok", order="Recommended")
        self.assertEqual(self.queries, [("gemma", "tok", TEXT_KIND)])
        self.assertEqual(len(state), 2)
        self.assertIn("No starters matched", detail)
        self.assertIn("most downloaded first", detail)
        self.assertNotIn("Curated starters", detail)

    def test_recommended_fallthrough_failure_points_back_to_starters(self):
        self.results = ConnectionError("offline")
        table, detail, state, selected = app.search_models("gemma", "", order="Recommended")
        self.assertEqual(cells(table, "Model"), [])
        self.assertIsNone(selected)
        self.assertIn("Search failed", detail)
        self.assertIn("Clear the search to see offline starters", detail)
        self.assertEqual(state, {})

    def test_image_recommendations_are_separate_and_do_not_guess_memory(self):
        table, _, state, _ = app.search_models("", "", kind=IMAGE_KIND, order="Recommended")
        self.assertEqual(list(state), ["stabilityai/sd-turbo"])
        self.assertEqual(models_page.results_kind(state), IMAGE_KIND)
        # No parameter count, so no Params column either.
        self.assertNotIn("Params", painted(table)["headers"])
        table, detail, _ = models_page.refresh_search_results(None, state, "4-bit", True)
        self.assertEqual(cells(table, "Model"), [])
        self.assertIn("unknown sizes are hidden", detail)

    def test_a_failed_search_is_reported(self):
        self.results = ConnectionError("hub <unreachable>")

        table, detail, state, selected = app.search_models("olmo", "")

        self.assertEqual(cells(table, "Model"), [])
        self.assertIn("Search failed", detail)
        self.assertIn("hub &lt;unreachable&gt;", detail)
        self.assertEqual(state, {})
        self.assertIsNone(selected)

    def test_a_failed_search_leaves_its_traceback_in_the_log(self):
        # The handler is broad enough to catch a mistake in the search as
        # well as an unreachable Hub, and the card already carries the
        # message, so a line without the stack would only repeat it.
        self.results = ConnectionError("hub unreachable")

        with self.assertLogs("chatlab.ui.models_page", level="WARNING") as logged:
            app.search_models("olmo", "")

        self.assertIn("Hub search for 'olmo' failed", logged.output[0])
        self.assertIn("Traceback", logged.output[0])
        self.assertIn("hub unreachable", logged.output[0])

    def test_no_matches_is_said_plainly(self):
        self.results = []

        table, detail, _, _ = app.search_models("zzzz", "")

        self.assertEqual(cells(table, "Model"), [])
        self.assertIn("No language models matched", detail)
        self.assertIn("zzzz", detail)

    def test_choosing_a_result_fills_the_id_box_and_describes_it(self):
        _, _, state, _ = app.search_models("olmo", "")

        # The click lands on a row of the table as the browser has sorted it,
        # so the row is known by its first cell, not its position.
        box, detail, selected = app.select_search_result(
            state, None, picked(INSTRUCT.model_id, 7_298_011_136, "fits")
        )

        self.assertEqual(box["value"], INSTRUCT.model_id)
        self.assertEqual(selected, INSTRUCT.model_id)
        self.assertIn("https://huggingface.co/allenai/Olmo-3-7B-Instruct", detail)
        self.assertIn("7.3B", detail)
        self.assertIn("281K downloads", detail)
        self.assertIn("143 likes", detail)
        self.assertIn("apache-2.0", detail)
        self.assertIn("2026-06-25", detail)
        self.assertNotIn("Gated", detail)
        self.assertIn("Download and load", detail)

    def test_a_gated_result_says_a_token_is_needed(self):
        _, _, state, _ = app.search_models("llama", "")

        _, detail, _ = app.select_search_result(state, None, picked(GATED.model_id))

        self.assertIn("Gated", detail)
        self.assertIn("token", detail)

    def test_a_result_already_on_disk_says_so(self):
        models_page.cache_status = lambda model_id: CacheStatus(cached_bytes=15_000_000_000)
        _, _, state, _ = app.search_models("olmo", "")

        _, detail, _ = app.select_search_result(state, None, picked(INSTRUCT.model_id))

        self.assertIn("Already cached", detail)
        self.assertIn("15.0 GB cached", detail)
        self.assertIn("Load cached", detail)
        self.assertNotIn("Download and load", detail)

    def test_a_cached_result_of_another_kind_is_not_called_partly_cached(self):
        models_page.cache_status = lambda model_id: CacheStatus(
            cached_bytes=5_500_000_000, kind=""
        )
        _, _, state, _ = app.search_models("olmo", "")

        _, detail, _ = app.select_search_result(state, None, picked(INSTRUCT.model_id))

        self.assertIn("Already cached", detail)
        self.assertIn("not a model ChatLab can load", detail)
        self.assertNotIn("Partly cached", detail)
        self.assertNotIn("Download and load", detail)

    def test_a_partly_downloaded_result_says_so(self):
        models_page.cache_status = lambda model_id: CacheStatus(
            cached_bytes=100, missing_files=(MODEL_WEIGHTS,)
        )
        _, _, state, _ = app.search_models("olmo", "")

        _, detail, _ = app.select_search_result(state, None, picked(INSTRUCT.model_id))

        self.assertIn("Partly cached", detail)

    def test_an_unreadable_cache_leaves_the_result_uncached(self):
        def refuse(model_id):
            raise PermissionError(13, "Permission denied")

        models_page.cache_status = refuse
        _, _, state, _ = app.search_models("olmo", "")

        box, detail, _ = app.select_search_result(state, None, picked(INSTRUCT.model_id))

        self.assertEqual(box["value"], INSTRUCT.model_id)
        self.assertNotIn("cached", detail)
        self.assertIn("Download and load", detail)

    def test_choosing_nothing_leaves_the_id_box_alone(self):
        self.assertEqual(
            app.select_search_result({}, None, picked(None)),
            (gr.skip(), app.NO_RESULT_SELECTED, None),
        )
        self.assertEqual(
            app.select_search_result({}, None, picked("stale/pick")),
            (gr.skip(), app.NO_RESULT_SELECTED, None),
        )


class ModelSwitchTests(unittest.TestCase):
    """The chat page's switcher: the downloaded models a load would take now."""

    GB = 1024**3
    CONFIG = ModelFitTests.CONFIG
    cache = ModelFitTests.cache

    def setUp(self):
        self.manager = ModelManager()
        roomy(self)
        root = self.cache({OLMO: 15 * self.GB, "org/huge": 200 * self.GB, "org/small": self.GB})
        self.extra = [PARTIAL, UNSUPPORTED, PIPELINE]
        originals = (runtime.MANAGER, models_page.list_cached_models, models_page.cache_root)
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list_cached_models(root) + list(self.extra)
        models_page.cache_root = lambda: root
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(models_page, "cache_root", originals[2])
        )

    def load(self, model_id=OLMO):
        self.manager.model = object()
        self.manager.tokenizer = object()
        self.manager.model_id = model_id
        self.manager.device_name = "Apple Metal (MPS)"

    def painted(self, precision=None):
        """A draw of the switcher, without the stamp beside it."""

        update, stamp = app.refresh_model_switch(precision)
        self.assertEqual(stamp.revision, self.manager.cache_revision)
        return update

    def stamp(self, precision=None):
        """The stamp a tab holds after a draw of the switcher."""

        _update, stamp = app.refresh_model_switch(precision)
        return stamp

    def aged(self, stamp):
        """The same stamp, with its fit reading dated past SWITCH_FIT_SECONDS.

        Which is the tick on which the timer reads the machine's memory again
        rather than skipping on the two attribute checks alone.
        """

        return stamp._replace(checked=stamp.checked - app.SWITCH_FIT_SECONDS - 1)

    def test_only_whole_supported_text_models_that_fit_are_offered(self):
        # The partial download, the CTranslate2 export, the image pipeline
        # and the 200 GB model are all left out: a pick is a load, and each
        # of those loads would be refused.
        self.assertEqual(
            sorted(app.switch_choices()),
            [(OLMO, OLMO), ("org/small", "org/small")],
        )

    def test_an_mlx_conversion_is_offered_beside_the_transformers_models(self):
        # It answers on the Chat page like any other language model, so the
        # one control for choosing which model is answering has to offer it.
        self.extra.append(MLX)

        self.assertIn((MLX.model_id, MLX.model_id), app.switch_choices())

    def test_an_mlx_model_without_mlx_lm_is_left_out_as_unsupported(self):
        # Off Apple silicon the cache scan marks the repo unsupported, and
        # that filter is what keeps it out - not the kind.
        self.extra.append(
            cached(MLX.model_id, status=CacheStatus(cached_bytes=2_300_000_000, kind=""))
        )

        self.assertNotIn((MLX.model_id, MLX.model_id), app.switch_choices())

    def test_an_mlx_model_in_memory_is_chosen_and_left_alone_by_the_timer(self):
        # The switcher's value is whatever is in memory unless it is an image
        # pipeline, so an MLX model missing from the choices would be a value
        # the dropdown does not offer - and every tick would call the
        # switcher stale and repaint it under the reader.
        self.extra.append(MLX)
        self.load(MLX.model_id)
        self.manager.kind = model_cache.MLX_KIND

        update = self.painted()
        self.assertEqual(update["value"], MLX.model_id)
        self.assertIn(MLX.model_id, [value for _, value in update["choices"]])
        self.assertEqual(
            app.refresh_stale_model_switch(MLX.model_id, self.stamp()),
            (gr.skip(), gr.skip()),
        )

    def test_nothing_loaded_leaves_nothing_chosen(self):
        update = self.painted()

        self.assertIsNone(update["value"])
        self.assertTrue(update["visible"])
        self.assertEqual(sorted(value for _, value in update["choices"]), [OLMO, "org/small"])

    def test_the_model_in_memory_is_the_one_chosen(self):
        self.load()

        self.assertEqual(self.painted()["value"], OLMO)

    def test_a_load_under_way_is_named_as_the_choice(self):
        # As the badge does: memory is emptied for the whole of a load, and a
        # switcher showing nothing would invite a second load on top.
        self.manager.reserve_load("org/small")

        self.assertEqual(self.painted()["value"], "org/small")

    def test_a_precision_the_machine_cannot_hold_leaves_the_model_out(self):
        # 15 GB of half-precision weights against 18 GB free is tight whole
        # and comfortable at four bits, so the radio decides what is offered.
        roomy(self, total_gb=24, available_gb=18)

        self.assertNotIn((OLMO, OLMO), app.switch_choices("full"))
        self.assertIn((OLMO, OLMO), app.switch_choices("4-bit"))

    def test_the_model_in_memory_stays_offered_whether_or_not_it_fits(self):
        roomy(self, total_gb=24, available_gb=18)
        self.load()

        self.assertIn((OLMO, OLMO), app.switch_choices("full"))

    def test_an_empty_cache_hides_the_switcher(self):
        models_page.list_cached_models = lambda: [PARTIAL, PIPELINE]

        self.assertFalse(self.painted()["visible"])

    def test_the_timer_leaves_a_switcher_that_agrees_with_memory_alone(self):
        # A repaint would close the list under a reader who just opened it,
        # and costs a cache scan; nothing is redrawn until the answer changes.
        idle = (gr.skip(), gr.skip())
        stamp = self.stamp()
        self.assertEqual(app.refresh_stale_model_switch(None, stamp), idle)
        self.load()
        self.assertEqual(app.refresh_stale_model_switch(OLMO, stamp), idle)

    def test_an_image_model_in_memory_leaves_an_empty_switcher_alone(self):
        # An image model is never a choice, so the switcher rightly shows
        # nothing; a timer that compared it with the model ID would rescan
        # the cache every two seconds for as long as the pipeline stayed.
        self.manager.pipeline = object()
        self.manager.kind = IMAGE_KIND
        self.manager.model_id = "org/pipe"
        self.manager.device_name = "Apple Metal (MPS)"

        self.assertEqual(
            app.refresh_stale_model_switch(None, self.stamp()),
            (gr.skip(), gr.skip()),
        )
        self.assertIsNone(self.painted()["value"])

    def test_the_timer_repaints_a_switcher_another_tab_made_stale(self):
        stamp = self.stamp()
        self.load()

        update, drawn = app.refresh_stale_model_switch(None, stamp)

        self.assertEqual(update["value"], OLMO)
        self.assertIn((OLMO, OLMO), update["choices"])
        self.assertEqual(drawn.revision, self.manager.cache_revision)

    def test_the_timer_repaints_after_another_tab_changed_the_cache(self):
        # A download or a removal in another tab leaves the model in memory
        # alone, so the value on show is still right and only the list is
        # wrong: without the revision this tab would skip for ever and never
        # offer the new model, or go on offering the deleted one.
        self.load()
        stamp = self.stamp()
        self.assertEqual(
            app.refresh_stale_model_switch(OLMO, stamp), (gr.skip(), gr.skip())
        )

        self.manager.note_cache_change()
        update, drawn = app.refresh_stale_model_switch(OLMO, stamp)

        self.assertIn((OLMO, OLMO), update["choices"])
        self.assertEqual(drawn.revision, self.manager.cache_revision)
        self.assertEqual(
            app.refresh_stale_model_switch(OLMO, drawn),
            (gr.skip(), gr.skip()),
            "and settles again once this tab has caught up",
        )

    def test_a_tab_that_has_not_drawn_the_switcher_yet_is_painted(self):
        # The State starts empty, which no stamp ever equals.
        update, stamp = app.refresh_stale_model_switch(None, None)

        self.assertTrue(update["visible"])
        self.assertEqual(stamp.revision, self.manager.cache_revision)

    def test_picking_the_model_already_in_memory_does_nothing(self):
        self.load()

        nothing = [(gr.skip(), gr.skip(), gr.skip())]
        self.assertEqual(list(app.switch_model(OLMO)), nothing)
        self.assertEqual(list(app.switch_model(None)), nothing)

    def test_a_pick_during_a_reply_is_refused_and_put_back(self):
        self.load()
        self.assertTrue(self.manager.reserve_generation())
        self.addCleanup(self.manager.release_generation)
        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/small"))

        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip(), gr.skip())])
        alarm.assert_called_once()
        self.assertIn(app.SWITCH_BUSY, alarm.call_args.args)

    def test_a_pick_during_another_load_is_refused_and_put_back(self):
        self.manager.reserve_load(OLMO)
        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/small"))

        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip(), gr.skip())])
        self.assertIn(app.SWITCH_LOADING, alarm.call_args.args)

    def test_a_pick_refused_by_a_load_that_then_ends_still_names_the_load(self):
        # The refusal and its reason are one answer. Asking what has the
        # model a second time, after the reservation came back empty, races
        # the load finishing: nothing holds it by then, so the reader is
        # told a response is running and to press a Stop button that is not
        # on the page, over a switch that no reply ever touched.
        self.load()
        _checked_id, claim = self.manager.reserve_load(OLMO)
        real = self.manager.claim_exclusive_load

        def then_the_load_ends(model_id):
            answer = real(model_id)
            self.manager.release_load(claim)
            return answer

        with mock.patch.object(
            self.manager, "claim_exclusive_load", then_the_load_ends
        ), mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/small"))

        self.assertIsNone(self.manager.occupant, "the load ended in between")
        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip(), gr.skip())])
        self.assertIn(app.SWITCH_LOADING, alarm.call_args.args)
        self.assertNotIn(app.SWITCH_BUSY, alarm.call_args.args)

    def test_a_load_refused_by_a_load_that_then_ends_still_names_the_load(self):
        # The same race on the Models page's own card, which is worded from
        # the same answer.
        _checked_id, claim = self.manager.reserve_load(OLMO)
        real = self.manager.claim_exclusive_load

        def then_the_load_ends(model_id):
            answer = real(model_id)
            self.manager.release_load(claim)
            return answer

        with mock.patch.object(
            self.manager, "claim_exclusive_load", then_the_load_ends
        ):
            frames = list(models_page.load_cached_model("org/small"))

        self.assertIsNone(self.manager.occupant, "the load ended in between")
        self.assertEqual(len(frames), 1, "refused before any other card")
        self.assertIn(models_page.LOAD_WHILE_LOADING, frames[0])
        self.assertNotIn(models_page.LOAD_WHILE_GENERATING, frames[0])

    def test_a_pick_loads_from_the_cache_at_the_chosen_precision(self):
        self.load()
        cards = iter(["loading card", "ready card"])
        with mock.patch.object(
            models_page, "load_cached_model", side_effect=lambda *args: cards
        ) as load:
            frames = list(app.switch_model("org/small", "4-bit"))

        # The claim it took is handed down rather than left to be taken
        # again: load_cached_model claims the load itself for the Models
        # page's buttons, and a second claim here would refuse this one.
        load.assert_called_once_with("org/small", None, "4-bit", mock.ANY)
        # The switcher itself is left to the rescan that follows; the cards
        # go to the Models page, as Load cached's do, and the badge beside
        # the switcher is repainted with each of them and once more at the
        # end, so the reader sees the load where they are looking.
        self.assertEqual([frame[:2] for frame in frames[:-1]], [
            (gr.skip(), "loading card"),
            (gr.skip(), "ready card"),
        ])
        for frame in frames:
            self.assertIn("model-badge", frame[2])
        self.assertEqual(frames[-1][:2], (gr.skip(), gr.skip()))
        self.assertIsNone(self.manager.loading_id, "the claim is given back")

    def test_a_pick_takes_the_load_before_it_yields_its_first_card(self):
        # Two tabs picking at once must not both get through. The load claims
        # nothing until stream_load, a cache scan and several cards later, and
        # Gradio resumes a yielded handler only after the browser has the
        # frame, so a check-then-load would leave a round-trip-wide window in
        # which both picks find the manager idle and both fill memory in turn.
        self.load()
        during = {}

        def cards(*args):
            during["claimed"] = self.manager.loading_id
            with mock.patch.object(models_page, "alarm") as alarm:
                during["second"] = list(app.switch_model("org/small"))
            during["told"] = alarm.call_args.args
            yield "card"

        with mock.patch.object(models_page, "load_cached_model", side_effect=cards):
            frames = list(app.switch_model("org/small"))

        self.assertEqual(during["claimed"], "org/small", "claimed before any card")
        self.assertEqual(
            during["second"], [(gr.update(value=OLMO), gr.skip(), gr.skip())]
        )
        self.assertIn(app.SWITCH_LOADING, during["told"])
        self.assertEqual([frame[:2] for frame in frames], [
            (gr.skip(), "card"),
            (gr.skip(), gr.skip()),
        ])
        self.assertIsNone(self.manager.loading_id, "the claim is given back")

    def test_a_pick_gives_the_load_back_when_the_cards_stop_early(self):
        # "Not cached" and the other refusals return without loading; the
        # claim must not outlive them, or the switcher jams for good.
        self.load()

        def refusal(*args):
            yield "not cached card"

        with mock.patch.object(models_page, "load_cached_model", side_effect=refusal):
            list(app.switch_model("org/small"))

        self.assertIsNone(self.manager.loading_id)

    def test_a_pick_of_a_malformed_id_is_refused_with_a_card(self):
        self.load()

        frames = list(app.switch_model("nonsense"))

        self.assertEqual(frames[0][0], gr.update(value=OLMO))
        self.assertIn("Could not load cached model", frames[0][1])
        self.assertIsNone(self.manager.loading_id)

    def test_the_models_page_cannot_load_while_a_pick_holds_the_load(self):
        # The claim the switcher takes has to turn away the other buttons,
        # not only another pick: Load cached reaching stream_load beside it
        # would fill memory twice over and leave whichever load finished
        # last in it, which is not the one the reader chose.
        self.load()
        during = {}
        real = models_page.load_cached_model

        def cards(*_args):
            during["frames"] = list(real("org/huge"))
            yield "card"

        with mock.patch.object(models_page, "load_cached_model", side_effect=cards):
            list(app.switch_model("org/small"))

        self.assertEqual(len(during["frames"]), 1, "refused before any other card")
        self.assertIn("Cannot load now", during["frames"][0])
        self.assertIn(models_page.LOAD_WHILE_LOADING, during["frames"][0])

    def test_a_reply_cannot_start_while_a_pick_holds_the_load(self):
        # The generation slot and the load claim used to be unrelated, so a
        # reply starting after the switcher's busy check was admitted and
        # then ran on whatever the switch had just loaded.
        self.load()
        during = {}

        def cards(*_args):
            during["reserved"] = self.manager.reserve_generation()
            yield "card"

        with mock.patch.object(models_page, "load_cached_model", side_effect=cards):
            list(app.switch_model("org/small"))

        self.assertFalse(during["reserved"], "a reply was admitted beside the load")
        self.assertTrue(self.manager.reserve_generation(), "and can start after it")
        self.manager.release_generation()

    def test_a_model_being_redownloaded_is_not_offered(self):
        # Redownload leaves the cache entry complete for the whole fetch, so
        # nothing but active_downloads says the pick would be refused.
        progress, reserved = self.manager.reserve_download("org/small")
        self.assertTrue(reserved)

        self.assertNotIn(("org/small", "org/small"), app.switch_choices())
        self.assertIn((OLMO, OLMO), app.switch_choices())

        self.manager.release_download("org/small", progress)
        self.assertIn(("org/small", "org/small"), app.switch_choices())

    def test_the_model_in_memory_stays_offered_while_it_is_redownloaded(self):
        # It is what the switcher has to show as chosen, and picking it is a
        # no-op; dropping it would blank the dropdown instead.
        self.load("org/small")
        self.manager.reserve_download("org/small")

        self.assertIn(("org/small", "org/small"), app.switch_choices())
        self.assertEqual(self.painted()["value"], "org/small")

    def test_the_timer_repaints_when_a_download_starts(self):
        # Filtering a download out is only worth anything if the tab that
        # did not start it hears about it: the revision moves at both ends
        # of a download, not only when it finishes.
        self.load()
        stamp = self.stamp()
        self.assertEqual(
            app.refresh_stale_model_switch(OLMO, stamp), (gr.skip(), gr.skip())
        )

        progress, _reserved = self.manager.reserve_download("org/small")
        update, drawn = app.refresh_stale_model_switch(OLMO, stamp)

        self.assertNotIn(("org/small", "org/small"), update["choices"])
        self.assertEqual(drawn.revision, self.manager.cache_revision)

        self.manager.release_download("org/small", progress)
        update, _drawn = app.refresh_stale_model_switch(OLMO, drawn)

        self.assertIn(("org/small", "org/small"), update["choices"])

    def test_memory_taken_by_another_process_withdraws_a_model(self):
        # Fit is the one input to the list that nothing in ChatLab moves.
        # Another process taking several gigabytes leaves the model in memory
        # and the cache exactly as they were, so the two attribute checks
        # skip for ever and the dropdown goes on offering a load that would
        # now be refused - and a refused load has already unloaded the model
        # the reader was talking to.
        self.load("org/small")
        stamp = self.stamp()
        self.assertIn(OLMO, stamp.offered)
        roomy(self, total_gb=48, available_gb=6)

        self.assertEqual(
            app.refresh_stale_model_switch("org/small", stamp),
            (gr.skip(), gr.skip()),
            "not on every tick: this reading costs a cache scan",
        )
        update, drawn = app.refresh_stale_model_switch("org/small", self.aged(stamp))

        self.assertNotIn((OLMO, OLMO), update["choices"])
        self.assertNotIn(OLMO, drawn.offered)

    def test_memory_given_back_puts_a_model_on_offer_again(self):
        roomy(self, total_gb=48, available_gb=6)
        self.load("org/small")
        stamp = self.stamp()
        self.assertNotIn(OLMO, stamp.offered)
        roomy(self, total_gb=48, available_gb=40)

        update, drawn = app.refresh_stale_model_switch("org/small", self.aged(stamp))

        self.assertIn((OLMO, OLMO), update["choices"])
        self.assertIn(OLMO, drawn.offered)

    def test_a_re_read_that_says_the_same_thing_leaves_the_list_alone(self):
        # The beat is how often the fit is read, not how often the dropdown
        # is redrawn: a list that comes out the same is left exactly as it
        # is, open or closed, and only its stamp moves on.
        self.load()
        stamp = self.aged(self.stamp())

        update, drawn = app.refresh_stale_model_switch(OLMO, stamp)

        self.assertEqual(update, gr.skip())
        self.assertEqual(drawn.offered, stamp.offered)
        self.assertGreater(drawn.checked, stamp.checked)
        self.assertEqual(
            app.refresh_stale_model_switch(OLMO, drawn),
            (gr.skip(), gr.skip()),
            "and the next tick is two attribute reads again",
        )

    def test_a_switch_that_comes_to_nothing_is_announced_to_the_reader(self):
        # The load's cards go to the Models page, which is not the page the
        # pick was made on. Without a toast the reader watching the chat page
        # sees the badge fall back to "No model loaded" and nothing anywhere
        # saying why.
        self.load()
        card = models_page.status_card(
            "Not cached", "Nothing for `org/small` is in the cache.", "error"
        )

        with mock.patch.object(
            models_page, "load_cached_model", side_effect=lambda *args: iter([card])
        ):
            with mock.patch.object(models_page, "alarm") as alarm:
                frames = list(app.switch_model("org/small"))

        self.assertEqual([frame[:2] for frame in frames], [
            (gr.skip(), card),
            (gr.skip(), gr.skip()),
        ])
        alarm.assert_called_once_with(
            "Not cached", "Nothing for `org/small` is in the cache."
        )

    def test_a_model_removed_between_the_draw_and_the_pick_says_so(self):
        # The race the filter cannot close, end to end: the list is drawn,
        # the model goes, and the pick lands on a cache without it.
        self.load()

        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/gone"))

        self.assertIn("Not cached", frames[-2][1])
        alarm.assert_called_once()
        self.assertEqual(alarm.call_args.args[0], "Not cached")
        self.assertIsNone(self.manager.loading_id, "the claim is still given back")

    def test_a_failure_that_has_already_spoken_is_not_told_twice(self):
        # failure_card raises its own toast, so the switcher covers only the
        # endings that never wrote anything but a card.
        self.load()

        with mock.patch.object(gr, "Warning") as warning:
            card = models_page.failure_card(
                "Could not load cached model", "It did not fit."
            )
            with mock.patch.object(
                models_page, "load_cached_model", side_effect=lambda *args: iter([card])
            ):
                list(app.switch_model("org/small"))

        warning.assert_called_once()

    def test_a_switch_that_works_says_nothing_extra(self):
        self.load()
        card = models_page.status_card(
            "Model ready", "`org/small` is loaded on **CPU**.", "success"
        )

        with mock.patch.object(
            models_page, "load_cached_model", side_effect=lambda *args: iter([card])
        ):
            with mock.patch.object(models_page, "alarm") as alarm:
                list(app.switch_model("org/small"))

        alarm.assert_not_called()


class ModelBadgeTests(unittest.TestCase):
    """The chat page's badge: the model in memory, or the lack of one."""

    def setUp(self):
        self.manager = ModelManager()
        original = runtime.MANAGER
        runtime.MANAGER = self.manager
        self.addCleanup(lambda: setattr(runtime, "MANAGER", original))

    def load(self):
        self.manager.model = object()
        self.manager.tokenizer = object()
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS)"

    def test_a_loaded_model_is_named_with_its_device(self):
        self.load()
        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="ready"', badge)
        self.assertIn(OLMO, badge)
        self.assertIn("Apple Metal (MPS)", badge)
        # Nothing to go to the Models page for, and nothing to offer.
        self.assertFalse(offer["visible"])

    def test_no_model_says_so_and_offers_the_way_to_the_models_page(self):
        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="empty"', badge)
        self.assertIn(app.NO_MODEL_BADGE, badge)
        self.assertTrue(offer["visible"])

    def test_downloads_leave_the_setup_links_available(self):
        # These links only navigate, including while the default is fetching.
        for model_id in (settings.DEFAULT_MODEL_ID, "org/something-else"):
            with self.subTest(model_id=model_id):
                self.manager.active_downloads[model_id] = object()
                _badge, offer = app.refresh_model_badge()
                self.assertTrue(offer["visible"])
                self.assertNotIn("interactive", offer)

    def test_a_download_alone_does_not_claim_a_model_is_coming(self):
        # Download only ends by telling the reader to load the cached model;
        # nothing is read into memory. A badge saying "loading" would promise
        # a model that never arrives and then fall back to "No model loaded".
        self.manager.active_downloads[settings.DEFAULT_MODEL_ID] = object()

        badge, _offer = app.refresh_model_badge()

        self.assertIn('data-state="empty"', badge)
        self.assertIn(app.NO_MODEL_BADGE, badge)

    def test_a_load_under_way_names_the_model_coming_in(self):
        # model_id is cleared for the whole of a load, so the badge reads
        # loading_id and reports the minutes in between as a load.
        self.manager.reserve_load(OLMO)

        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn(f"Loading {OLMO}", badge)
        self.assertFalse(offer["visible"])

    def test_a_second_load_finishing_leaves_the_first_one_showing(self):
        # Two loads can be under way at once, and the one that finishes
        # first must not give back the other's claim: the badge would then
        # say nothing was loaded in the middle of a load.
        self.manager.reserve_load(OLMO)
        _second_id, second = self.manager.reserve_load("org/second")
        self.manager.release_load(second)

        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn(f"Loading {OLMO}", badge)
        self.assertFalse(offer["visible"])

    def test_a_load_waiting_its_turn_leaves_the_answering_model_named(self):
        # A load counts itself as under way before it waits for the model
        # lock, so asking for a second model mid-reply queues it behind the
        # generation. The model still producing the tokens is the one the
        # badge is for, so it keeps the name until the load empties memory.
        self.load()
        self.manager.reserve_load("org/second")

        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="ready"', badge)
        self.assertIn(OLMO, badge)
        self.assertNotIn("Loading", badge)

    def test_a_load_that_has_emptied_memory_names_the_model_coming_in(self):
        # Once the queued load wins the lock it unloads first, and from then
        # on there is nothing in memory to name.
        self.load()
        self.manager.reserve_load("org/second")
        self.manager.model = None
        self.manager.tokenizer = None
        self.manager.model_id = None
        self.manager.device_name = None

        badge, offer = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn("Loading org/second", badge)

    def progress_for(self, model_id, steps_done=0, steps_total=0):
        """Claim a load of ``model_id`` and publish a bar that far along."""

        from chatlab.progress_bars import LoadProgress

        _checked_id, claim = self.manager.reserve_load(model_id)
        progress = LoadProgress()
        self.manager.note_load_progress(claim, progress)
        if steps_total:
            progress.bar_class()(desc="Loading weights", total=steps_total).update(
                steps_done
            )
        return claim, progress

    def test_a_load_in_progress_fills_a_bar_in_the_badge(self):
        # The reader who picked a model in the switcher beside the badge is
        # watching this, not the Models page where the load's card goes.
        self.progress_for(OLMO, steps_done=3, steps_total=4)

        badge, _offer = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn(f"Loading {OLMO}… 75%", badge)
        self.assertIn('data-progress="known"', badge)
        self.assertIn("width: 75%", badge)

    def test_a_load_that_has_not_begun_reporting_shows_no_figure(self):
        # A load queued behind a reply, or still opening the snapshot, has
        # nothing to report, and a bar pinned at zero through it would read
        # as a load that has stalled.
        self.progress_for(OLMO)

        badge, _offer = app.refresh_model_badge()

        self.assertIn('data-progress="unknown"', badge)
        self.assertNotIn("%", badge)

    def test_a_load_started_elsewhere_is_measured_on_the_chat_page(self):
        # The cards a load streams reach only the handler that started it,
        # so the manager is where every other tab, and every other page,
        # reads how far it has come.
        self.progress_for(OLMO, steps_done=1, steps_total=2)

        images, _button = app.refresh_image_badge()

        self.assertIn("50%", images)

    def test_the_bar_follows_the_load_the_badge_names(self):
        # Two claims can stand at once, and the one holding the model lock is
        # the one being named; showing the other's figures beside that name
        # would describe two loads as one.
        self.progress_for(OLMO, steps_done=1, steps_total=4)
        self.progress_for("org/second", steps_done=3, steps_total=4)
        with self.manager._reading_weights(OLMO):
            badge, _offer = app.refresh_model_badge()

        self.assertIn(f"Loading {OLMO}… 25%", badge)

    def test_a_finished_load_leaves_no_bar_behind(self):
        claim, _progress = self.progress_for(OLMO, steps_done=4, steps_total=4)
        self.manager.release_load(claim)
        self.load()

        badge, _offer = app.refresh_model_badge()

        self.assertIn('data-state="ready"', badge)
        self.assertNotIn("model-badge-bar", badge)
        self.assertIsNone(self.manager.loading_progress())

    def test_the_model_id_is_escaped(self):
        self.load()
        self.manager.model_id = "org/<script>"

        self.assertIn("&lt;script&gt;", app.loaded_model_badge())

    def load_pipeline(self):
        self.manager.pipeline = object()
        self.manager.kind = IMAGE_KIND
        self.manager.model_id = "org/pipe"
        self.manager.device_name = "Apple Metal (MPS)"

    def test_an_image_model_is_ready_on_images_and_the_wrong_kind_on_chat(self):
        # Something is in memory, so neither page shows the empty state; the
        # Chat page saying "no model loaded" would send a reader off to load
        # a second model on top of the one filling the machine.
        self.load_pipeline()

        chat, offer = app.refresh_model_badge()
        images, image_button = app.refresh_image_badge()

        self.assertIn('data-state="ready"', images)
        self.assertIn("org/pipe", images)
        self.assertIn("Apple Metal (MPS)", images)
        self.assertFalse(image_button["visible"])

        self.assertIn('data-state="other"', chat)
        self.assertIn("image model, not used here", chat)
        # From the Chat page there is still a model to go and load.
        self.assertTrue(offer["visible"])

    def test_a_text_model_is_the_wrong_kind_on_the_images_page(self):
        self.load()

        images, image_button = app.refresh_image_badge()

        self.assertIn('data-state="other"', images)
        self.assertIn("text model, not used here", images)
        self.assertTrue(image_button["visible"])

    def test_neither_page_offers_a_setup_link_while_a_load_is_pending(self):
        self.manager.reserve_load(OLMO)

        images, image_button = app.refresh_image_badge()

        self.assertIn('data-state="loading"', images)
        self.assertFalse(image_button["visible"])

    def test_nothing_loaded_reads_the_same_on_both_pages(self):
        images, image_button = app.refresh_image_badge()

        self.assertIn('data-state="empty"', images)
        self.assertIn(app.NO_MODEL_BADGE, images)
        self.assertTrue(image_button["visible"])


MLX = cached(
    "mlx-community/Qwen3-4B-4bit",
    status=CacheStatus(cached_bytes=2_300_000_000, kind=model_cache.MLX_KIND),
    architecture="Qwen3ForCausalLM",
    dtype="4-bit MLX",
)


class MlxModelsPaneTests(unittest.TestCase):
    """An MLX conversion in My Models and in Discover."""

    def setUp(self):
        self.entries = [cached(OLMO), MLX]
        self.manager = ModelManager()
        originals = (runtime.MANAGER, models_page.list_cached_models, models_page.cache_root)
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list(self.entries)
        models_page.cache_root = lambda: Path("/cache")
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(models_page, "cache_root", originals[2])
        )

    def test_an_mlx_model_is_listed_as_one_and_points_at_the_chat_page(self):
        radio, _, _ = app.refresh_my_models(None)
        _, detail = app.select_my_model(MLX.model_id)

        labels = dict((value, label) for label, value in radio["choices"])
        self.assertIn("· MLX", labels[MLX.model_id])
        self.assertNotIn("· MLX", labels[OLMO])
        self.assertIn("Ready to load", detail)
        self.assertIn("**Chat** page", detail)
        self.assertIn("**Kind:** MLX text model", detail)
        self.assertIn("4-bit MLX", detail)
        self.assertNotIn("Unsupported", detail)

    def test_an_mlx_row_carries_its_kind_before_its_fit_verdict(self):
        from chatlab.device_memory import FITS, Fit

        label = models_page.cached_model_label(MLX, Fit(FITS))

        self.assertIn("· MLX", label)
        self.assertIn("· fits", label)
        self.assertLess(label.index("· MLX"), label.index("· fits"))

    def test_moving_the_precision_radio_does_not_rejudge_a_loaded_mlx_model(self):
        # Load cached at a new precision is how a Transformers model is
        # requantized; an MLX model loads at its own width whatever the radio
        # says, so the loaded one has nothing to be judged again for.
        from chatlab.device_memory import DeviceProfile

        self.manager.model_id = MLX.model_id
        self.manager.precision = "4-bit"
        profile = DeviceProfile(backend="mps", dtype="float16", total=10**11, available=10**11)

        self.assertIsNone(models_page.cached_fit(MLX, "full", profile))
        self.assertIsNone(models_page.cached_fit(MLX, "8-bit", profile))

    def test_mlx_recommendations_are_their_own_list_and_judged_at_their_width(self):
        _, _, state, _ = app.search_models(
            "", "", kind=model_cache.MLX_KIND, order="Recommended"
        )

        self.assertEqual(
            list(state),
            [
                "mlx-community/Qwen3-0.6B-4bit",
                "mlx-community/Qwen3-4B-4bit",
                "mlx-community/Olmo-3-7B-Think-4bit",
            ],
        )
        self.assertEqual(models_page.results_kind(state), model_cache.MLX_KIND)
        _, detail, _ = models_page.select_search_result(
            state, "full", picked("mlx-community/Qwen3-4B-4bit")
        )
        self.assertIn("quantized already", detail)
        self.assertNotIn("Choosing 4-bit or 8-bit", detail)

    def test_an_mlx_result_is_sized_from_the_width_in_its_name(self):
        from chatlab.device_memory import DeviceProfile
        from chatlab.hub_search import HubModel
        from chatlab.model_cache import estimate_parameter_bytes

        profile = DeviceProfile(backend="mps", dtype="float16", total=10**11, available=10**11)
        four_bit = HubModel(
            model_id="mlx-community/Some-7B-4bit", parameters=7_000_000_000, kind=model_cache.MLX_KIND
        )
        unnamed = HubModel(
            model_id="mlx-community/Some-7B", parameters=7_000_000_000, kind=model_cache.MLX_KIND
        )

        # The radio says full; the name says 4 bits, and the name wins.
        packed = models_page.hub_fit(four_bit, "full", profile, model_cache.MLX_KIND)
        whole = models_page.hub_fit(unnamed, "4-bit", profile, model_cache.MLX_KIND)

        self.assertEqual(packed.estimated, estimate_parameter_bytes(7_000_000_000, "float16", 4))
        self.assertEqual(whole.estimated, estimate_parameter_bytes(7_000_000_000, "float16", None))
        # And the note names the width the figure was measured at, which for
        # the packed one is the name's and not the radio's.
        self.assertIn("of 4-bit weights", packed.note)
        self.assertIn("of full 16-bit weights", whole.note)

    def test_a_cached_mlx_verdict_names_the_width_it_was_converted_to(self):
        # The radio is at full and the repo is at four bits. The figure is
        # of the packed file already on disk, so calling it full 16-bit
        # weights would misread it by four times.
        GB = 1024**3
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        config = json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "quantization": {"group_size": 64, "bits": 4},
            }
        )
        folder = lay_out(
            root, MLX.model_id, {"config.json": config.encode(), "model.safetensors": b""}
        )
        with (folder / "blobs" / "blob1").open("r+b") as weights:
            weights.truncate(2 * GB)
        entry = cached(
            MLX.model_id,
            status=CacheStatus(cached_bytes=2 * GB, kind=model_cache.MLX_KIND),
            path=folder,
        )
        profile = device_memory.DeviceProfile(
            backend="mps", dtype="float16", total=48 * GB, available=40 * GB
        )

        fit = models_page.cached_fit(entry, "full", profile)

        self.assertIn("2.0 GB of 4-bit weights", fit.note)

    @contextlib.contextmanager
    def capped_mac(self):
        """20 GB of weights per entry, on a 48 GB Mac behind a 16 GB PyTorch cap.

        30 GB of that machine is free, so the MLX conversion fits it and the
        same bytes as a Transformers checkpoint do not fit the cap.
        """

        GB = 1024**3
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        config = json.dumps({"architectures": ["Qwen3ForCausalLM"], "dtype": "float16"})
        entries = []
        for model_id, kind in ((MLX.model_id, model_cache.MLX_KIND), (OLMO, TEXT_KIND)):
            folder = lay_out(
                root, model_id, {"config.json": config.encode(), "model.safetensors": b""}
            )
            with (folder / "blobs" / "blob1").open("r+b") as weights:
                weights.truncate(20 * GB)
            entries.append(
                cached(model_id, status=CacheStatus(cached_bytes=20 * GB, kind=kind), path=folder)
            )
        self.entries = entries
        capped = device_memory.DeviceProfile(
            backend="mps",
            dtype="float16",
            total=16 * GB,
            available=16 * GB,
            ceiling=16 * GB,
            pool="Metal on this machine",
        )
        with mock.patch.object(models_page, "device_profile", lambda torch=None: capped):
            with mock.patch.object(
                device_memory, "system_memory", lambda: (48 * GB, 30 * GB)
            ):
                yield

    def test_an_mlx_model_is_judged_against_the_machine_not_the_metal_cap(self):
        # Load lets the MLX conversion through, because mlx-lm is not under
        # that cap, so the list has to say fits.
        with self.capped_mac():
            radio, _, _ = app.refresh_my_models(None, "Name")

        labels = dict((value, label) for label, value in radio["choices"])
        self.assertIn("· fits", labels[MLX.model_id])
        self.assertIn("· won't fit", labels[OLMO])

    def test_selecting_an_mlx_row_repeats_the_verdict_the_list_gave_it(self):
        # The details are recomputed from scratch on selection, so they have
        # to take the pool for the row's own kind too. Judged against the
        # PyTorch cap instead, a row the list calls a fit would describe
        # itself as unfit the moment it was clicked.
        with self.capped_mac():
            radio, _, _ = app.refresh_my_models(None, "Name")
            _, mlx_detail = app.select_my_model(MLX.model_id)
            _, text_detail = app.select_my_model(OLMO)

        labels = dict((value, label) for label, value in radio["choices"])
        self.assertIn("· fits", labels[MLX.model_id])
        self.assertIn("inside the 30.0 GB ChatLab estimates free", mlx_detail)
        self.assertNotIn("Metal on this machine", mlx_detail)
        # The Transformers checkpoint is still held to the cap, in both places.
        self.assertIn("· won't fit", labels[OLMO])
        self.assertIn("more than the 16.0 GB Metal on this machine has", text_detail)


if __name__ == "__main__":
    unittest.main()
