"""The Models page: My Models, Model search, and where the settings live."""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import app
from ui import models_page, runtime
import model_runtime
import settings
from model_runtime import (
    IMAGE_KIND,
    MODEL_WEIGHTS,
    TEXT_KIND,
    CachedModel,
    CacheStatus,
    DownloadProgress,
    HubModel,
    ModelManager,
    format_count,
    list_cached_models,
    remove_cached_model,
    sort_cached_models,
)

import settings_sandbox

COMMIT = "d97e442d7cc678210054dbcc9b440894d62c89a4"
OLMO = "allenai/Olmo-3-7B-Think"


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

    profile = model_runtime.DeviceProfile(
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
        for cached, expected in (
            (CacheStatus(), "Not downloaded"),
            (CacheStatus(cached_bytes=100, missing_files=(MODEL_WEIGHTS,)), "Download incomplete"),
        ):
            with self.subTest(cached=cached), mock.patch.object(models_page, "cache_status", return_value=cached):
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
            newest_write = model_runtime._newest_write

            def refuse_one(folder, snapshot):
                if folder == broken:
                    raise OSError(5, "Input/output error")
                return newest_write(folder, snapshot)

            with mock.patch.object(model_runtime, "_newest_write", refuse_one):
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
                with self.assertRaises(model_runtime.ModelDownloading) as caught:
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
        with self.assertRaises(model_runtime.ModelLoaded):
            self.manager.remove(OLMO, Path(self.root.name))

        self.assertEqual(self.manager.cache_revision, 0)

    def test_the_loaded_model_is_refused(self):
        self.manager.model_id = OLMO
        with self.assertRaises(model_runtime.ModelLoaded):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())
        self.assertFalse(self.manager._lock.locked())

    def test_a_model_being_downloaded_is_refused(self):
        self.manager.active_downloads[OLMO] = DownloadProgress()
        with self.assertRaises(model_runtime.ModelDownloading):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())
        self.assertFalse(self.manager._downloads_lock.locked())

    def test_a_busy_manager_is_refused_without_waiting(self):
        self.manager._lock.acquire()
        self.addCleanup(self.manager._lock.release)
        with self.assertRaises(model_runtime.ModelBusy):
            self.manager.remove(OLMO, Path(self.root.name))
        self.assertTrue(self.folder.is_dir())

    def test_every_refusal_is_a_model_in_use(self):
        for error in (model_runtime.ModelLoaded, model_runtime.ModelDownloading, model_runtime.ModelBusy):
            self.assertTrue(issubclass(error, model_runtime.ModelInUse))

    def test_a_load_claimed_on_another_thread_is_refused(self):
        # The load's own thread has not reached the model lock yet, so the
        # lock is free and would let the deletion through.
        _model_id, claim = self.manager.reserve_load(OLMO)
        with self.assertRaises(model_runtime.ModelBusy):
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


def cached(model_id: str, **overrides) -> CachedModel:
    fields = dict(
        model_id=model_id,
        status=CacheStatus(cached_bytes=15_000_000_000),
        files=12,
        commit=COMMIT,
        updated=1_700_000_000.0,
        architecture="Olmo3ForCausalLM",
        dtype="bfloat16",
        path=Path("/cache") / f"models--{model_id.replace('/', '--')}",
    )
    fields.update(overrides)
    return CachedModel(**fields)


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
        from model_runtime import FITS, Fit

        label = models_page.cached_model_label(PIPELINE, Fit(FITS))

        self.assertIn("· image", label)
        self.assertIn("· fits", label)
        self.assertLess(label.index("· image"), label.index("· fits"))

    def test_an_image_pipeline_is_judged_at_the_precision_it_is_measured_at(self):
        # A pipeline is sized whole whatever the radio says - the Metal
        # quantizer is Transformers' own and the load clears the choice for
        # one - so carrying the bits into the verdict would put a quantized
        # label on a full-size figure.
        profile = model_runtime.DeviceProfile(
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
            "org/small": model_runtime.HubModel(
                model_id="org/small", parameters=1_000_000_000
            )
        }
        roomy(self, total_gb=24, available_gb=18, backend=None, dtype=None)
        original = models_page.imported_torch
        models_page.imported_torch = lambda: object()
        self.addCleanup(lambda: setattr(models_page, "imported_torch", original))

        _radio, _detail, _summary, results, _search, _selected, known = (
            app.refresh_after_device(False, None, "Name", "full", None, None, held)
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
        # stay spelled the same.
        for verdict in ("· won't fit", "· tight"):
            self.assertIn(f'[data-testid*="{verdict}"]', app.CSS)

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
            model_runtime.remove_cached_model,
            models_page.download_model,
        )
        runtime.MANAGER = self.manager
        models_page.list_cached_models = lambda: list(self.entries)
        # The manager deletes through the module-level function, so that is
        # what stands in: the manager's own checks stay real.
        model_runtime.remove_cached_model = self.remove
        models_page.download_model = self.download
        self.addCleanup(
            lambda: setattr(runtime, "MANAGER", originals[0])
            or setattr(models_page, "list_cached_models", originals[1])
            or setattr(model_runtime, "remove_cached_model", originals[2])
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

        model_runtime.remove_cached_model = refuse

        status, confirm, pending = app.remove_my_model("org/partial")

        self.assertIn("Could not remove model", status)
        self.assertIn("Permission denied: &lt;blobs&gt;", status)
        self.assertFalse(confirm["visible"])
        self.assertIsNone(pending)

    def test_a_model_gone_before_confirming_is_reported(self):
        def gone(model_id, cache_dir=None):
            raise FileNotFoundError(model_id)

        model_runtime.remove_cached_model = gone

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


def painted(table) -> dict:
    """What the browser is sent for a search table: headers, data, and metadata."""

    return gr.Dataframe(interactive=False).postprocess(table["value"]).model_dump()


def picked(model_id: str | None, *rest) -> gr.SelectData:
    """A click on a row whose first cell is ``model_id``; None for no row at all."""

    row = [model_id, *rest] if model_id else None
    return gr.SelectData(None, {"index": [0, 0], "value": model_id, "row_value": row})


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
        # Gradio sends None for an untouched textbox on initial page load.
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

    def test_recommended_query_matching_a_starter_stays_offline(self):
        self.results = ConnectionError("offline")
        _, detail, state, _ = app.search_models("qwen", "", order="Recommended")
        self.assertEqual(self.queries, [])
        self.assertEqual(list(state), ["Qwen/Qwen3-0.6B"])
        self.assertIn("Curated starters", detail)

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
        self.manager.kind = model_runtime.MLX_KIND

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

        self.assertEqual(list(app.switch_model(OLMO)), [(gr.skip(), gr.skip())])
        self.assertEqual(list(app.switch_model(None)), [(gr.skip(), gr.skip())])

    def test_a_pick_during_a_reply_is_refused_and_put_back(self):
        self.load()
        self.assertTrue(self.manager.reserve_generation())
        self.addCleanup(self.manager.release_generation)
        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/small"))

        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip())])
        alarm.assert_called_once()
        self.assertIn(app.SWITCH_BUSY, alarm.call_args.args)

    def test_a_pick_during_another_load_is_refused_and_put_back(self):
        self.manager.reserve_load(OLMO)
        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/small"))

        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip())])
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
        self.assertEqual(frames, [(gr.update(value=OLMO), gr.skip())])
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
        # go to the Models page, as Load cached's do.
        self.assertEqual(frames, [(gr.skip(), "loading card"), (gr.skip(), "ready card")])
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
        self.assertEqual(during["second"], [(gr.update(value=OLMO), gr.skip())])
        self.assertIn(app.SWITCH_LOADING, during["told"])
        self.assertEqual(frames, [(gr.skip(), "card")])
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

        self.assertEqual(frames, [(gr.skip(), card)])
        alarm.assert_called_once_with(
            "Not cached", "Nothing for `org/small` is in the cache."
        )

    def test_a_model_removed_between_the_draw_and_the_pick_says_so(self):
        # The race the filter cannot close, end to end: the list is drawn,
        # the model goes, and the pick lands on a cache without it.
        self.load()

        with mock.patch.object(models_page, "alarm") as alarm:
            frames = list(app.switch_model("org/gone"))

        self.assertIn("Not cached", frames[-1][1])
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


class PageLayoutTests(unittest.TestCase):
    """The nav picks a page: the model controls sit on Models, the settings on
    Settings, and the conversation on Chat."""

    ON_MODELS_PAGE = [
        "Hugging Face model ID",
        "Hugging Face token (optional)",
        "Downloaded models",
        "Sort by",
        "Search Hugging Face",
        "Search results",
    ]
    ON_SETTINGS_PAGE = [
        "System prompt",
        "Send previous reasoning back to the model",
        "Measure prompt tokens",
        "Enter sends the message",
        "Context limit (tokens)",
    ]
    # Everything about drawing a picture is on Images, including its own
    # sliders: the Chat page's sampling controls say nothing to a diffusion
    # model, and these say nothing to a language one.
    ON_IMAGES_PAGE = [
        "Prompt",
        "Negative prompt",
        "Picture",
        "Denoising steps",
        "Guidance scale",
        "Size",
        "Seed",
        "Randomize seed",
        "Record cross-attention",
        "Step",
        "Prompt tokens — click one for its map",
    ]
    # Sampling sits with the conversation, not behind the nav: these are what
    # a reader moves between one retry and the next.
    ON_CHAT_PAGE = [
        "Conversation",
        "Message",
        "Color tokens by",
        "Text to score",
        "Temperature",
        "Top-p",
        "Top-k (0 disables)",
        "Maximum new tokens",
        "Random seed",
        "🎲 New seed each response",
    ]

    def setUp(self):
        self.demo = app.build_app()

    def by_id(self, elem_id):
        return next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "elem_id", None) == elem_id
        )

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def within(self, block, container) -> bool:
        parent = getattr(block, "parent", None)
        while parent is not None:
            if parent is container:
                return True
            parent = getattr(parent, "parent", None)
        return False

    def test_each_control_sits_on_its_page(self):
        for page, labels in [
            ("models-page", self.ON_MODELS_PAGE),
            ("settings-page", self.ON_SETTINGS_PAGE),
            ("chat-page", self.ON_CHAT_PAGE),
            ("images-page", self.ON_IMAGES_PAGE),
        ]:
            container = self.by_id(page)
            for label in labels:
                with self.subTest(page=page, label=label):
                    self.assertTrue(self.within(self.labelled(label), container))

    def test_the_nav_offers_every_page_and_starts_on_chat(self):
        nav = self.by_id("nav")
        self.assertIsInstance(nav, gr.Radio)
        self.assertEqual(
            [value for _, value in nav.choices], ["Chat", "Images", "Models", "Settings"]
        )
        self.assertEqual(nav.value, "Chat")
        self.assertTrue(self.within(nav, self.by_id("nav-pane")))

    def test_the_shell_spans_the_whole_window(self):
        # Gradio otherwise caps the page at one of a handful of widths and
        # centers it, leaving empty room down each side on a wide screen.
        self.assertTrue(self.demo.fill_width)

    def test_the_nav_pane_is_thin(self):
        # Wide enough for the longest page name at the tile's small type,
        # and no wider: the pane is a signpost, not a sidebar.
        self.assertLessEqual(app.NAV_PANE_WIDTH, 96)
        self.assertEqual(self.by_id("nav-pane").min_width, app.NAV_PANE_WIDTH)

    def test_each_nav_tile_shows_an_icon_above_the_page_name(self):
        for page in app.PAGES:
            with self.subTest(page=page):
                tile = f'#nav label[data-testid="{page}-radio-label"]'
                # The empty alternative text keeps the icon out of what a
                # screen reader reads; the label's own text stands for the
                # tile, and is now printed under the icon rather than hidden.
                self.assertIn(
                    f'{tile}::before {{ content: "{app.NAV_ICONS[page]}" / ""; }}',
                    app.CSS,
                )
        self.assertIn("#nav label span { font-size:", app.CSS)

    def test_a_compact_window_stacks_the_images_panes_too(self):
        # Its two panes want about 620px between them, so in a narrow window
        # the readings would sit off the side of a row that neither wraps
        # nor scrolls sideways.
        compact = app.CSS[app.CSS.index("@media (max-width: 850px)") :]
        compact = compact[: compact.index("\n}")]

        self.assertIn("#images-columns", compact)
        self.assertIn("#images-workspace", compact)
        self.assertIn("#image-inspector", compact)

    def test_each_readings_pane_has_a_handle_on_its_seam(self):
        # The handle is a flex item between the workspace and the pane, so
        # the seam it sits on is the edge the reader drags.
        for pane_id, handle_id in [
            ("inspector-pane", "inspector-resizer"),
            ("image-inspector", "image-inspector-resizer"),
        ]:
            with self.subTest(pane=pane_id):
                pane = self.by_id(pane_id)
                handle = self.by_id(handle_id)
                self.assertIs(handle.parent, pane.parent)
                self.assertIn(f'data-pane="{pane_id}"', handle.value)
                self.assertIn(f'data-property="--{pane_id}-width"', handle.value)
                self.assertIn(f'data-store="chatlab.{pane_id}-width"', handle.value)
                # A width the reader chose is written to that property, so
                # every rule that sizes the pane has to read it - including
                # the narrower window's, which sets a smaller default.
                for rule in [
                    line
                    for line in app.CSS.splitlines()
                    if "flex" in line and f"--{pane_id}-width" in line
                ]:
                    self.assertIn(f"var(--{pane_id}-width,", rule)
                self.assertEqual(
                    app.CSS.count(f"var(--{pane_id}-width,"), 2, pane_id
                )
                # And the script that writes it knows the pane by the same name.
                self.assertIn(f"'{pane_id}'", app.RESIZE_JS)

    def test_the_handle_keeps_touch_gestures_off_its_strip(self):
        # A touch device wider than the stacking breakpoint still drags the
        # handle, and a browser that reads that drag as a pan or a zoom takes
        # the pointer back mid-resize, which leaves the pane part-moved.
        # Refusing the pointerdown does not stop it; only this does.
        rule = app.CSS[app.CSS.index(".pane-resizer {") :]
        rule = rule[: rule.index("}")]

        self.assertIn("touch-action: none", rule)

    def test_the_stacked_layout_drops_the_handles(self):
        # Under 850px the panes are rows, one above the other, where a width
        # would mean a height and a sideways drag would mean nothing.
        compact = app.CSS[app.CSS.index("@media (max-width: 850px)") :]
        compact = compact[: compact.index("\n}")]

        self.assertIn("#inspector-resizer, #image-inspector-resizer", compact)
        self.assertIn("display: none", compact)

    def test_a_saved_width_is_fitted_to_the_room_the_pane_has(self):
        # A width chosen on a wide window has to be cut down when the window
        # narrows, and the figure to cut it to is the row the pane sits in
        # rather than the window itself: the Chat row gives up space to the
        # conversations pane and the Images row does not. Watching the rows
        # covers a page that was away while the window changed as well, since
        # the row it is built into reports its size the moment it has one.
        # What that watching is worth when a page comes and goes is run
        # through in PaneResizeScriptTests.
        self.assertIn("new ResizeObserver", app.RESIZE_JS)
        self.assertIn("rows.observe(row)", app.RESIZE_JS)
        self.assertIn("window.addEventListener('resize'", app.RESIZE_JS)
        # Where the panes become rows a width would mean a height, so the
        # script stops fitting at the same width the stylesheet stops
        # reading the property.
        stacked = "(max-width: 850px)"
        self.assertIn(f"@media {stacked}", app.CSS)
        self.assertIn(f"matchMedia('{stacked}')", app.RESIZE_JS)

    def test_the_nav_names_are_on_screen_rather_than_a_hover_away(self):
        # Four pages is not a number worth hiding. Nothing clips the name
        # out of sight, and no tooltip stands in for it.
        self.assertNotIn("clip-path: inset(50%)", app.CSS)
        self.assertNotIn("#nav label::after", app.CSS)
        self.assertNotIn(":hover::after", app.CSS)

    def test_only_the_chat_page_starts_visible(self):
        self.assertTrue(self.by_id("chat-page").visible)
        self.assertTrue(self.by_id("conversation-pane").visible)
        self.assertFalse(self.by_id("images-page").visible)
        self.assertFalse(self.by_id("models-page").visible)
        self.assertFalse(self.by_id("settings-page").visible)

    def test_picking_a_page_shows_it_alone(self):
        (listener,) = self.listeners("show_page")
        self.assertEqual(listener.targets, [(self.by_id("nav")._id, "change")])
        self.assertEqual(
            listener.outputs,
            [
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("images-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )
        shown = lambda page: [update["visible"] for update in app.show_page(page)]
        # The conversations pane comes and goes with Chat.
        self.assertEqual(shown("Chat"), [True, True, False, False, False])
        self.assertEqual(shown("Images"), [False, False, True, False, False])
        self.assertEqual(shown("Models"), [False, False, False, True, False])
        self.assertEqual(shown("Settings"), [False, False, False, False, True])

    def listeners(self, name):
        return [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        ]

    def follows(self, listener, name) -> bool:
        """Whether a handler called ``name`` runs, sooner or later, after ``listener``."""

        after: dict = {}
        for dependency in self.demo.config["dependencies"]:
            after.setdefault(dependency["trigger_after"], []).append(dependency["id"])
        pending, seen = [listener._id], set()
        while pending:
            for dependency_id in after.get(pending.pop(), []):
                if dependency_id in seen:
                    continue
                seen.add(dependency_id)
                if getattr(self.demo.fns[dependency_id].fn, "__name__", None) == name:
                    return True
                pending.append(dependency_id)
        return False

    def cancelled_by(self, trigger) -> set:
        """Event indices cancelled by anything bound to ``trigger``.

        Gradio records a listener's ``cancels`` against the target rather
        than the handler, so this reads every function on that target.
        """

        return {
            index
            for fn in self.demo.fns.values()
            if fn.targets == [trigger]
            for index in fn.cancels
        }

    def test_the_badge_sits_above_the_chat_page_tabs(self):
        chat_page = self.by_id("chat-page")
        badge = self.by_id("model-badge")
        switch = self.by_id("model-switch")
        self.assertTrue(self.within(badge, chat_page))
        self.assertTrue(self.within(switch, chat_page))
        self.assertIsInstance(switch, gr.Dropdown)
        # Above the tabs, so Score text names the model as well as Chat.
        tabs = next(
            block for block in self.demo.blocks.values() if isinstance(block, gr.Tabs)
        )
        self.assertFalse(self.within(badge, tabs))

    def test_every_change_to_what_is_in_memory_repaints_the_badge(self):
        # Download-and-load, load cached and unload change what is in memory;
        # the page load draws the badge first, switching pages catches a load
        # that started while the chat page was out of sight, and the timer
        # catches one another tab started.
        # The four that change memory (the switcher included), the download
        # that only changes what is on disk, redownload and a confirmed
        # removal, plus the page load, the nav and the timer.
        self.assertEqual(len(self.listeners("refresh_model_badge")), 10)

    def test_the_timer_also_un_sticks_the_scored_token_count(self):
        # A count asked for during a reply gives up and says so, and that
        # message does not correct itself when the reply ends. Rather than
        # ask every path out of a generation to remember, the badge's timer
        # carries the recovery - guarded so the ordinary tick costs nothing.
        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        (recovery,) = self.listeners("recover_score_budget")

        self.assertEqual(recovery.targets, [(timers[0]._id, "tick")])
        self.assertEqual(recovery.inputs[0], self.by_id("score-budget"))
        self.assertEqual(recovery.outputs[0], self.by_id("score-budget"))
        # The count and the load it was counted against travel together.
        self.assertIsInstance(recovery.inputs[1], gr.State)
        self.assertEqual(recovery.outputs, recovery.inputs[:2])

    def test_everything_that_writes_the_count_shares_one_queue(self):
        # Gradio's concurrency limit is per event, not across events, so
        # without a shared id the timer's recovery and a keystroke's count
        # can overlap - and they contend for the same model lock, so one of
        # them loses it and publishes the "not mid-response" message. The
        # loser finishing last would leave a count that does not describe the
        # box, which is the one thing this line exists to rule out.
        budget = self.by_id("score-budget")
        writers = [fn for fn in self.demo.fns.values() if budget in fn.outputs]

        self.assertGreater(len(writers), 1)
        self.assertEqual(
            {fn.concurrency_id for fn in writers}, {app.SCORE_BUDGET_QUEUE}
        )

    def test_the_badge_asks_again_on_a_timer(self):
        # The manager is one object for the whole process, but a handler's
        # updates only reach the tab that ran it. Without the timer a second
        # tab would name a model that another tab has since swapped out.
        timers = [
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Timer) and block.value == app.BADGE_REFRESH_SECONDS
        ]
        self.assertEqual([timer.value for timer in timers], [app.BADGE_REFRESH_SECONDS])
        self.assertLessEqual(app.BADGE_REFRESH_SECONDS, 5)
        ticks = [
            listener
            for listener in self.listeners("refresh_model_badge")
            if listener.targets == [(timers[0]._id, "tick")]
        ]
        self.assertEqual(len(ticks), 1)
        # Nobody asked for this one, so it does not put a pending shimmer on
        # the badge every couple of seconds.
        self.assertEqual(ticks[0].show_progress, "hidden")

    def test_the_badge_buttons_send_the_nav_to_the_models_page(self):
        # One on Images: its badge says a model it can use is missing, and
        # offers the way to load one. The Chat page's badge has the switcher
        # and the default-model button instead.
        panes = [
            self.by_id("nav"),
            self.by_id("conversation-pane"),
            self.by_id("chat-page"),
            self.by_id("images-page"),
            self.by_id("models-page"),
            self.by_id("settings-page"),
        ]
        buttons = {"image-load-model"}
        found = set()
        for listener in self.listeners("go_to_models"):
            ((block_id, event),) = listener.targets
            self.assertEqual(event, "click")
            found.add(self.demo.blocks[block_id].elem_id)
            # The button switches the pages itself: a Radio set by a handler
            # reports no change, so the nav's own handler would not run.
            self.assertEqual(listener.outputs, panes)
        self.assertEqual(found, buttons)

        page, *updates = app.go_to_models()
        self.assertEqual(page, "Models")
        self.assertEqual(
            [update["visible"] for update in updates],
            [False, False, False, True, False],
        )

    def test_every_model_change_rescans_the_cache(self):
        # Download, download-and-load, load cached, unload, redownload,
        # confirmed removal, the refresh button, a new sort order, a new
        # weight precision, the page load and a pick in the chat page's
        # switcher each rescan. Selecting the default only navigates.
        self.assertEqual(len(self.listeners("refresh_my_models")), 11)

    def test_model_actions_follow_selections_and_cache_refreshes(self):
        listeners = self.listeners("refresh_model_actions")
        radio = self.labelled("Downloaded models")
        model_id = self.labelled("Hugging Face model ID")
        token = self.labelled("Hugging Face token (optional)")
        for control in (radio, model_id, token):
            self.assertTrue(any(fn.targets == [(control._id, "change")] for fn in listeners))
        for fn in listeners:
            self.assertEqual(fn.inputs[:2], [model_id, radio])
            self.assertIsInstance(fn.inputs[2], gr.State)
            self.assertEqual(fn.inputs[3:], [token])
            self.assertEqual(fn.outputs[0], self.by_id("model-availability"))
            self.assertEqual(
                [button.value for button in fn.outputs[1:]],
                ["Download and load", "Download only", "Load cached"],
            )
        refresh_ids = {fn._id for fn in self.listeners("refresh_my_models")}
        chained = [
            dependency for dependency in self.demo.config["dependencies"]
            if dependency["id"] in {fn._id for fn in listeners}
            and dependency["trigger_after"] in refresh_ids
        ]
        # All seven mutations (the switcher included), manual refresh, and
        # startup refresh the controls even when the radio's selected value
        # stays the same.
        self.assertEqual(len(chained), 9)

    def test_explicit_repository_checks_make_one_request_after_leaving_the_id_field(self):
        model_id = self.labelled("Hugging Face model ID")
        (button,) = [
            block for block in self.demo.blocks.values()
            if isinstance(block, gr.Button) and block.value == "Check model"
        ]
        checks = self.listeners("check_model_repository")
        info = mock.Mock(
            tags=[], config={}, library_name="transformers", siblings=[],
            private=False, gated=False,
        )
        for events, expected in (
            ([(model_id._id, "blur"), (button._id, "click")], 1),
            ([(model_id._id, "submit"), (model_id._id, "blur")], 1),
            ([(model_id._id, "blur")], 0),
        ):
            with self.subTest(events=events), mock.patch(
                "huggingface_hub.HfApi.model_info", return_value=info
            ) as request:
                # Dispatch the actual registered dependencies in browser event
                # order: clicking Check first blurs the focused model ID field.
                for event in events:
                    for listener in checks:
                        if event in listener.targets:
                            states = list(listener.fn("org/model", ""))
                            self.assertEqual(states[-1]["status"], "found")
                self.assertEqual(request.call_count, expected)

    def test_repository_precision_refreshes_for_cached_selections_and_rescans(self):
        views = self.listeners("repository_view")
        selected = self.labelled("Downloaded models")
        self.assertTrue(any(fn.targets == [(selected._id, "change")] for fn in views))
        self.assertTrue(any(event == "load" for fn in views for _, event in fn.targets))
        for fn in views:
            self.assertEqual(fn.inputs[-1], selected)
            self.assertEqual(fn.inputs[0], self.labelled("Hugging Face model ID"))
            self.assertEqual(fn.inputs[2], self.labelled("Hugging Face token (optional)"))
        action_ids = {fn._id for fn in self.listeners("refresh_model_actions")}
        chained = [
            dependency for dependency in self.demo.config["dependencies"]
            if dependency["id"] in {fn._id for fn in views}
            and dependency["trigger_after"] in action_ids
        ]
        self.assertEqual(len(chained), 9)

    def test_every_load_reads_the_my_models_selection(self):
        # The ID box lags a row selection by a server round trip, so a button
        # clicked in that window would act on the box's previous contents -
        # the 15 GB default. Each load takes the radio as well and prefers it.
        radio = self.labelled("Downloaded models")
        for name in ("load_cached_model", "download_model", "download_and_load_model"):
            (fn,) = self.listeners(name)
            self.assertIn(radio, fn.inputs, name)

    def test_download_then_load_keeps_the_typed_model_when_another_model_is_loaded(self):
        manager = ModelManager()
        manager.model_id = OLMO
        entries = [cached(OLMO)]
        typed_id = "org/new-model"
        (download,) = self.listeners("download_model")
        (load,) = self.listeners("load_cached_model")
        dependency = next(
            item for item in self.demo.config["dependencies"]
            if item["trigger_after"] == download._id
        )
        refresh = self.demo.fns[dependency["id"]]
        self.assertEqual(refresh.fn, models_page.refresh_my_models)
        self.assertEqual(refresh.inputs[-1], self.labelled("Hugging Face model ID"))

        def fetch(model_id, token):
            entries.append(cached(model_id))
            yield "download progress"
            return Path("/cache/new-model")

        def read_weights(*args):
            yield "load progress"
            return "CPU"

        with (
            mock.patch.object(runtime, "MANAGER", manager),
            mock.patch.object(models_page, "list_cached_models", side_effect=lambda: list(entries)),
            mock.patch.object(models_page, "cache_status", side_effect=lambda model_id: next((entry.status for entry in entries if entry.model_id == model_id), CacheStatus())),
            mock.patch.object(models_page, "stream_download", side_effect=fetch),
            mock.patch.object(manager, "find_cached", return_value=Path("/cache/new-model")),
            mock.patch.object(models_page, "stream_load", side_effect=read_weights) as stream_load,
        ):
            selected = models_page.clear_my_model_selection()[0]["value"]
            cards = list(download.fn(typed_id, "", selected))
            self.assertIn("Download complete", cards[-1])
            self.assertEqual(manager.model_id, OLMO)
            radio, _, _ = refresh.fn(selected, "Name", model_id=typed_id)
            detail, _, _, load_button = models_page.refresh_model_actions(typed_id, radio["value"])
            self.assertEqual(radio["value"], typed_id)
            self.assertIn("Ready to load", detail)
            self.assertTrue(load_button["visible"])
            list(load.fn(typed_id, radio["value"]))
            self.assertEqual(stream_load.call_args.args[0], typed_id)

    def test_a_picked_row_outranks_the_id_box(self):
        # A click's inputs are snapshotted in the browser, and a row reaches
        # the box only through a server round trip, so the box a button
        # carries can still hold the 15 GB default while the radio is
        # current. The radio therefore wins whenever there is one.
        self.assertEqual(
            app.chosen_model(settings.DEFAULT_MODEL_ID, "org/picked"), "org/picked"
        )
        self.assertEqual(app.chosen_model("", "org/picked"), "org/picked")
        # With no row picked the typed ID is all there is.
        self.assertEqual(app.chosen_model("  org/typed  ", None), "org/typed")
        self.assertEqual(app.chosen_model("", None), "")

    def test_naming_a_model_another_way_withdraws_the_selection(self):
        # Typing an ID or picking a search result names its own model, so the
        # highlighted row cannot outrank it.
        listeners = self.listeners("clear_my_model_selection")
        self.assertEqual(len(listeners), 2)
        radio = self.labelled("Downloaded models")
        for fn in listeners:
            self.assertIn(radio, fn.outputs)

    def test_removal_asks_before_deleting(self):
        # The Remove button only opens the question; deleting is the
        # confirm button's job. Cancelling withdraws it, and so does naming
        # another model, whether by choosing a row or by typing an ID.
        (ask,) = self.listeners("ask_remove_my_model")
        (remove,) = self.listeners("remove_my_model")
        buttons = {
            self.demo.blocks[block_id].value: fn
            for fn in (ask, remove)
            for block_id, _ in fn.targets
        }
        self.assertIs(buttons["🗑️ Remove"], ask)
        self.assertIs(buttons["Remove from disk"], remove)
        self.assertEqual(len(self.listeners("hide_remove_confirm")), 3)

    def test_the_confirm_button_deletes_the_model_the_question_named(self):
        # The confirm handler reads the stored pending ID, not the radio, so
        # a selection moved after the question opened cannot redirect it.
        (ask,) = self.listeners("ask_remove_my_model")
        (remove,) = self.listeners("remove_my_model")
        radio = self.labelled("Downloaded models")
        (pending,) = remove.inputs
        self.assertIsInstance(pending, gr.State)
        self.assertIsNot(pending, radio)
        self.assertIn(pending, ask.outputs)
        self.assertIn(pending, remove.outputs)

    def test_clear_asks_before_it_takes_every_conversation(self):
        # Clear reaches past the conversation on screen: it deletes every
        # other one too, and nothing brings them back. The button only opens
        # the question; the confirm button clears once a running job stops.
        (ask,) = self.listeners("ask_clear_chat")
        (clear,) = self.listeners("clear_chat")
        cancel = next(
            fn
            for fn in self.listeners("hide_clear_confirm")
            if self.demo.blocks[fn.targets[0][0]].value == "Cancel"
        )
        buttons = {
            self.demo.blocks[block_id].value: fn
            for fn in (ask, clear, cancel)
            for block_id, _ in fn.targets
        }
        self.assertIs(buttons["🗑️ Clear all"], ask)
        self.assertIs(buttons["Clear everything"], clear)
        self.assertIs(buttons["Cancel"], cancel)
        # Cancelling is recorded against the target rather than the handler,
        # so it is read the way ClearCancelsGenerationTests reads it.
        self.assertFalse(self.cancelled_by(ask.targets[0]))
        self.assertFalse(self.cancelled_by(clear.targets[0]))

    def test_changing_the_conversations_withdraws_the_clear_question(self):
        # The question names how many conversations it would take, counted
        # when it was asked. Left open across a New or a Fork it would
        # promise less than "Clear everything" would take - and that promise
        # is the whole reason the question exists.
        withdrawals = self.listeners("hide_clear_confirm")
        triggered_by = {fn.targets[0][0] for fn in withdrawals}
        buttons = {
            self.demo.blocks[block_id].value
            for block_id in triggered_by
            if isinstance(self.demo.blocks[block_id], gr.Button)
        }

        self.assertEqual(buttons, {"Cancel", "➕ New", "🌿 Fork", "🗑️ Delete"})
        # Switching conversations counts too, and it is the list itself.
        self.assertIn(self.by_id("conversation-list")._id, triggered_by)
        for fn in withdrawals:
            self.assertEqual(fn.outputs, [self.by_id("clear-confirm")])

    def test_the_clear_button_is_named_for_everything_it_takes(self):
        # "Clear" alone reads as emptying the chat on screen, which is what
        # Delete does. This one takes the lot.
        (ask,) = self.listeners("ask_clear_chat")
        ((block_id, _),) = ask.targets

        self.assertEqual(self.demo.blocks[block_id].value, "🗑️ Clear all")

    def test_the_offer_sits_beside_the_badge_that_says_it_is_needed(self):
        # The badge names the missing model; the offer is what to do about
        # it, and both belong where the reader already is.
        offer = self.by_id("default-model")

        self.assertTrue(self.within(offer, self.by_id("chat-page")))
        self.assertTrue(self.within(offer, self.by_id("model-bar")))
        (setup,) = self.listeners("select_default_model")
        self.assertEqual(setup.targets, [(offer._id, "click")])
        self.assertEqual(offer.value, "Set up the default model")
        # No chained handler may turn this navigation back into automatic I/O.
        self.assertFalse(any(
            dependency["trigger_after"] == setup._id
            for dependency in self.demo.config["dependencies"]
        ))
        # It switches the pages itself, for the reason go_to_models gives: a
        # Radio set by a handler reports no change, so setting the nav alone
        # would tick Models and leave the chat page on screen.
        self.assertEqual(
            setup.outputs,
            [
                self.labelled("Hugging Face model ID"),
                # A row picked earlier outranks the ID box, so it goes.
                self.labelled("Downloaded models"),
                self.by_id("my-model-detail"),
                # The search selection is a State beside the table, so it is
                # found through the handler that writes it.
                self.listeners("select_search_result")[0].outputs[2],
                self.listeners("select_search_result")[0].outputs[1],
                self.by_id("model-status"),
                self.listeners("hide_remove_confirm")[0].outputs[0],
                self.listeners("hide_remove_confirm")[0].outputs[1],
                self.by_id("nav"),
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("images-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )

    def test_the_offer_is_published_wherever_the_badge_is(self):
        # Setup links share the badge's visibility decision in every tab.
        listeners = self.listeners("refresh_model_badge")
        self.assertTrue(listeners)
        for listener in listeners:
            self.assertEqual(
                listener.outputs,
                [self.by_id("model-badge"), self.by_id("default-model")],
            )

    def test_the_switcher_is_drawn_when_the_badge_is_and_after_every_rescan(self):
        # Arriving, opening the page, and the timer - which asks first whether
        # the switcher still names what is in memory, so an open list is not
        # closed under the reader every couple of seconds.
        switch = self.by_id("model-switch")
        precision = self.labelled("Weight precision")
        listeners = self.listeners("refresh_model_switch")
        triggers = {listener.targets[0] for listener in listeners}
        self.assertIn((self.by_id("nav")._id, "change"), triggers)
        self.assertIn((self.demo._id, "load"), triggers)
        # Every draw hands back the cache revision it read, kept per tab, so
        # the timer can tell an idle list from one another tab left stale.
        revision = listeners[0].outputs[1]
        self.assertIsInstance(revision, gr.State)
        for listener in listeners:
            self.assertEqual(listener.inputs, [precision])
            self.assertEqual(listener.outputs, [switch, revision])
        # Every load, unload and download repaints it: the cache and memory
        # are what it offers.
        for name in ("load_cached_model", "download_and_load_model", "unload_model",
                     "download_model", "switch_model"):
            with self.subTest(handler=name):
                action = self.listeners(name)[0]
                self.assertTrue(self.follows(action, "refresh_model_switch"))

        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        ticks = self.listeners("refresh_stale_model_switch")
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].targets, [(timers[0]._id, "tick")])
        self.assertEqual(ticks[0].inputs, [switch, revision, precision])
        self.assertEqual(ticks[0].outputs, [switch, revision])
        self.assertEqual(ticks[0].show_progress, "hidden")

    def test_a_pick_in_the_switcher_loads_at_the_chosen_precision(self):
        switch = self.by_id("model-switch")
        listeners = self.listeners("switch_model")
        self.assertEqual(len(listeners), 1)
        self.assertEqual(listeners[0].targets, [(switch._id, "input")])
        self.assertEqual(listeners[0].inputs, [switch, self.labelled("Weight precision")])
        self.assertEqual(listeners[0].outputs, [switch, self.by_id("model-status")])
        # And is followed by the same rescan as Load cached: the badge, the
        # token count and the hardware panel all change with the model.
        for name in ("refresh_my_models", "refresh_model_badge", "refresh_hardware"):
            with self.subTest(handler=name):
                self.assertTrue(self.follows(listeners[0], name))

    def test_the_images_badge_is_refreshed_on_the_same_three_occasions(self):
        # Arriving at the page, opening it, and the timer that tells a tab
        # which did not start a load about it.
        listeners = self.listeners("refresh_image_badge")
        outputs = [self.by_id("image-model-badge"), self.by_id("image-load-model")]
        for listener in listeners:
            self.assertEqual(listener.outputs, outputs)
        self.assertEqual(len(listeners), 3)

        triggers = {listener.targets[0] for listener in listeners}
        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        self.assertIn((self.by_id("nav")._id, "change"), triggers)
        self.assertIn((timers[0]._id, "tick"), triggers)
        # The timer's own refresh does not put a pending shimmer on the badge
        # every couple of seconds; nobody asked it anything.
        ticks = [
            listener
            for listener in listeners
            if listener.targets == [(timers[0]._id, "tick")]
        ]
        self.assertEqual([listener.show_progress for listener in ticks], ["hidden"])

    def test_stop_drawing_does_not_cancel_the_generator_that_publishes_the_run(self):
        # The pipeline runs on its own thread and would keep running with the
        # generator gone, taking every recorded step with it. So Stop sets an
        # event the run checks between steps, and the generator itself
        # publishes the stopped run.
        (stop,) = self.listeners("stop_drawing")
        ((block_id, event),) = stop.targets

        self.assertEqual(event, "click")
        self.assertEqual(self.demo.blocks[block_id].elem_id, "stop-drawing")
        self.assertEqual(stop.cancels, [])
        (draw,) = self.listeners("draw")
        self.assertNotIn(draw._id, self.cancelled_by((block_id, event)))

    def test_moving_the_step_repaints_the_frame_the_shading_and_the_map(self):
        # Attention moves between steps as much as the picture does, so these
        # cannot be allowed to disagree about which step is on screen.
        (select,) = self.listeners("select_step")

        self.assertEqual(select.targets, [(self.by_id("image-step")._id, "release")])
        self.assertEqual(
            select.outputs,
            [
                self.by_id("image-trajectory"),
                self.by_id("image-prompt-strip"),
                self.by_id("image-attention-note"),
                self.by_id("image-attention"),
            ],
        )
        self.assertIs(select.inputs[1], self.by_id("image-step"))

    def test_clicking_a_prompt_token_is_remembered_before_the_map_is_drawn(self):
        # The click's index has to land in the state the map reads, so the
        # map follows the step slider afterwards without another click.
        (remember,) = self.listeners("remember_token")
        (token_state,) = remember.outputs
        (paint,) = self.listeners("select_token")

        self.assertEqual(
            remember.targets, [(self.by_id("image-prompt-strip")._id, "select")]
        )
        self.assertEqual(paint.inputs[1], token_state)
        self.assertEqual(paint.outputs, [self.by_id("image-attention")])
        self.assertEqual(paint.trigger_after, remember._id)
    def test_the_prompt_upload_takes_every_file_the_parser_reads(self):
        # The parser reads anything that is not JSON as blank-line separated
        # text, and the README says so, so a filter that only offered .txt
        # would hide the .md and extensionless prompt sets it handles.
        upload = next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == "\U0001f4c2 Load prompts"
        )

        self.assertIn("text", upload.file_types)
        self.assertIn(".json", upload.file_types)
        self.assertIn(".jsonl", upload.file_types)

    def test_escape_is_wired_to_the_stop_button_by_its_id(self):
        # The shortcut presses the button rather than reaching past it, so
        # whatever Stop does, Escape does. It needs the id to find it.
        stops = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "value", None) == "Stop"
        ]

        # One stops a reply, one a batch of prompts, one a picture being
        # drawn. No more than one can be in the page: they contend for the
        # same generation slot and the losers refuse.
        self.assertEqual(
            {stop.elem_id for stop in stops},
            {"stop-button", "stop-batch-button", "stop-drawing"},
        )
        self.assertIn("#stop-button", app.SHORTCUT_JS)
        self.assertIn("#stop-batch-button", app.SHORTCUT_JS)
        self.assertIn("#stop-drawing", app.SHORTCUT_JS)
        # Whether the button is in the document is the whole test. Gradio
        # leaves a component whose visible is false out of the page, so its
        # presence is the generation state itself. Testing whether it can be
        # *seen* would drop the key on the Score text tab, where the button
        # is still in the page with a hidden ancestor - the moment a reader
        # is most likely to reach for it, being away from the button.
        self.assertNotIn("offsetParent", app.SHORTCUT_JS)
        self.assertNotIn("offsetWidth", app.SHORTCUT_JS)
        self.assertNotIn("getBoundingClientRect", app.SHORTCUT_JS)
        self.assertNotIn("checkVisibility", app.SHORTCUT_JS)
        self.assertTrue(
            any(fn.js == app.SHORTCUT_JS for fn in self.demo.fns.values()),
            "nothing attaches the keyboard shortcut on load",
        )

    def test_the_sampling_accordion_starts_showing_the_saved_values(self):
        # The summary is only worth having if it is right before anything is
        # touched, which means the label and the sliders read one set of
        # numbers - and that set is the saved settings, not a second copy of
        # the defaults that could drift from them.
        accordion = next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Accordion)
            and (block.label or "").startswith("Sampling")
        )
        saved = settings.load()

        self.assertEqual(
            accordion.label,
            app.sampling_label(
                saved.temperature, saved.top_p, saved.top_k, saved.max_new_tokens
            ),
        )
        for label, value in [
            ("Temperature", saved.temperature),
            ("Top-p", saved.top_p),
            ("Top-k (0 disables)", saved.top_k),
            ("Maximum new tokens", saved.max_new_tokens),
        ]:
            with self.subTest(control=label):
                self.assertEqual(self.labelled(label).value, value)
                self.assertTrue(self.within(self.labelled(label), accordion))
        # The response length cannot outrun the context limit.
        self.assertEqual(
            self.labelled("Maximum new tokens").maximum, saved.prefill_token_limit
        )

    def test_the_sampling_summary_follows_every_slider(self):
        # A slider fires continuously while it is dragged; the label only has
        # to be right once it is let go.
        sliders = [
            self.labelled(label)
            for label in ("Temperature", "Top-p", "Top-k (0 disables)", "Maximum new tokens")
        ]
        listeners = self.listeners("update_sampling_label")
        # A page load's target has no block, so look the ids up by hand.
        by_id = {slider._id: slider for slider in sliders}
        moved = [fn for fn in listeners if fn.targets[0][0] in by_id]

        self.assertEqual([by_id[fn.targets[0][0]] for fn in moved], sliders)
        for fn in listeners:
            self.assertEqual(fn.inputs, sliders)
        # The others are the paths that move a slider without anyone touching
        # it: the settings file read back on load, the context limit
        # committed, which can pull the response length down with it, and the
        # six that change which conversation is on screen - forking, starting
        # one, switching, deleting, clearing, and the page load that brings
        # the saved conversations back - each of which brings that
        # conversation's own sampling onto the sliders.
        self.assertEqual(len(listeners) - len(moved), 9)

    def test_everything_that_writes_the_summary_shares_one_queue(self):
        # always_last coalesces each slider's own requests; across four
        # listeners Gradio orders nothing. Each handler reads all four values
        # as they were when its request was sent, so two sliders moved in
        # quick succession can finish out of order and leave the label
        # describing the older pair - and only the next change rewrites it.
        accordion = next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Accordion)
            and (block.label or "").startswith("Sampling")
        )
        writers = [fn for fn in self.demo.fns.values() if accordion in fn.outputs]

        self.assertGreater(len(writers), 1)
        self.assertEqual(
            {fn.concurrency_id for fn in writers}, {app.SAMPLING_LABEL_QUEUE}
        )
        # And it is its own queue, not shared with the token count, which
        # costs an encoding and would make the label wait behind it.
        self.assertNotEqual(app.SAMPLING_LABEL_QUEUE, app.SCORE_BUDGET_QUEUE)

    def test_the_sampling_summary_follows_a_slider_moved_by_keyboard(self):
        # Gradio dispatches release from pointerup alone, so a slider moved
        # with the arrow keys - which is how it is moved without a mouse -
        # changes its value and never reports a release. Listening for
        # release would leave the summary describing the old settings for
        # anyone not using a pointer.
        sliders = {
            self.labelled(label)._id
            for label in ("Temperature", "Top-p", "Top-k (0 disables)", "Maximum new tokens")
        }

        for fn in self.listeners("update_sampling_label"):
            block_id, event = fn.targets[0]
            if block_id in sliders:
                with self.subTest(slider=self.demo.blocks[block_id].label):
                    self.assertEqual(event, "change")
                    # A drag fires change on every step, so they coalesce.
                    self.assertEqual(fn.trigger_mode, "always_last")

    def test_the_scored_token_count_follows_every_box_that_feeds_it(self):
        # The count has to match what would actually be scored, so a change
        # to the context or the chat-template box moves it too.
        boxes = [
            self.labelled("Context (optional)"),
            self.labelled("Text to score"),
            self.labelled("Treat the context as a chat message"),
        ]
        listeners = self.listeners("score_token_count")
        typed = [fn for fn in listeners if fn.trigger_mode == "always_last"]

        self.assertEqual([self.demo.blocks[fn.targets[0][0]] for fn in typed], boxes)
        for fn in listeners:
            self.assertEqual(fn.inputs, boxes)
        # A different tokenizer counts the same passage differently and a
        # different model has its own limit, so every handler that changes
        # what is loaded recomputes the count rather than leaving the old
        # model's answer under the box.
        # Four of the rescans change neither: the refresh button, a new sort
        # order, a new weight precision, and the page load.
        self.assertEqual(
            len(listeners) - len(typed), len(self.listeners("refresh_my_models")) - 4
        )

    def test_choosing_a_model_writes_the_id_box(self):
        box = self.labelled("Hugging Face model ID")
        for name in ("select_my_model", "select_search_result"):
            with self.subTest(handler=name):
                (listener,) = self.listeners(name)
                self.assertIs(listener.outputs[0], box)


# Enough of a page for RESIZE_JS to run against: two rows of the shape the
# layout builds, and stand-ins for the browser it talks to. Nothing here
# lays anything out, so every element is told its own width, and the frames
# and the mutations are delivered by the checks rather than by a clock.
RESIZE_PAGE = """
'use strict';
const assert = require('node:assert');

class Style {
  constructor() { this.props = {}; }
  setProperty(name, value) { this.props[name] = value; }
  removeProperty(name) { delete this.props[name]; }
}

class Element {
  constructor(id, width) {
    this.id = id || '';
    this.width = width || 0;
    this.children = [];
    this.parentElement = null;
    this.dataset = {};
    this.style = new Style();
    this.names = new Set();
    this.attributes = {};
    this.pointer = null;
    this.classList = {
      add: (name) => this.names.add(name),
      remove: (name) => this.names.delete(name),
      contains: (name) => this.names.has(name),
    };
  }
  get clientWidth() { return this.width; }
  getBoundingClientRect() { return { width: this.width }; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }
  append(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  remove() {
    const siblings = this.parentElement.children;
    siblings.splice(siblings.indexOf(this), 1);
    this.parentElement = null;
  }
  closest(selector) {
    if (selector === '.pane-resizer' && this.names.has('pane-resizer')) { return this; }
    return this.parentElement ? this.parentElement.closest(selector) : null;
  }
  setPointerCapture(pointer) { this.pointer = pointer; }
  releasePointerCapture(pointer) {
    if (this.pointer === pointer) { this.pointer = null; }
  }
  hasPointerCapture(pointer) { return this.pointer === pointer; }
}

const find = (node, id) => {
  if (node.id === id) { return node; }
  for (const child of node.children) {
    const found = find(child, id);
    if (found) { return found; }
  }
  return null;
};

// The one selector the script asks the document for.
const gather = (node, name, found) => {
  if (node.names.has(name)) { found.push(node); }
  for (const child of node.children) { gather(child, name, found); }
  return found;
};

let watchers = [];
class MutationObserver {
  constructor(react) { this.react = react; }
  observe() { watchers.push(this.react); }
  disconnect() { watchers = watchers.filter((react) => react !== this.react); }
}
const mutated = () => { for (const react of watchers.slice()) { react(); } };

const resizers = [];
class ResizeObserver {
  constructor(react) {
    this.react = react;
    this.targets = new Set();
    resizers.push(this);
  }
  observe(target) {
    if (this.targets.has(target)) { return; }
    this.targets.add(target);
    this.react();
  }
  unobserve(target) { this.targets.delete(target); }
  disconnect() { this.targets.clear(); }
}

let frames = [];
const requestAnimationFrame = (frame) => frames.push(frame);
const paint = () => {
  const due = frames;
  frames = [];
  for (const frame of due) { frame(); }
};

const kept = new Map();
const localStorage = {
  getItem: (key) => (kept.has(key) ? kept.get(key) : null),
  setItem: (key, value) => kept.set(key, value),
  removeItem: (key) => kept.delete(key),
};

const documentElement = new Element('html', 1200);
const body = documentElement.append(new Element('body', 1200));
const heard = { document: {}, window: {} };
const document = {
  documentElement,
  body,
  getElementById: (id) => find(documentElement, id),
  querySelectorAll: (selector) => {
    assert.strictEqual(selector, '.pane-resizer', 'the page answers one selector');
    return gather(documentElement, 'pane-resizer', []);
  },
  addEventListener: (type, fn) => {
    (heard.document[type] = heard.document[type] || []).push(fn);
  },
};
const window = {
  innerWidth: 1200,
  matchMedia: () => ({ matches: window.innerWidth <= 850 }),
  addEventListener: (type, fn) => {
    (heard.window[type] = heard.window[type] || []).push(fn);
  },
};
const fire = (where, type, event) => {
  for (const fn of (heard[where][type] || []).slice()) { fn(event); }
};
const shell = body.append(new Element('shell', 1200));

// A handle of the kind pane_handle() writes, carrying the same attributes.
const seam = (pane) => {
  const handle = new Element('', 6);
  handle.names.add('pane-resizer');
  handle.dataset.pane = pane;
  handle.dataset.property = '--' + pane + '-width';
  handle.dataset.store = 'chatlab.' + pane + '-width';
  return handle;
};

// The Chat row gives up space to the conversations pane beside it, and the
// Images row has only its handle to pay for.
const chatRow = (width) => {
  const row = shell.append(new Element('chat-columns', width));
  row.append(new Element('conversation-pane', 260));
  row.append(new Element('chat-workspace', width - 580));
  row.append(seam('inspector-pane'));
  row.append(new Element('inspector-pane', 314));
  return row;
};

const imagesRow = (width) => {
  const row = shell.append(new Element('images-columns', width));
  row.append(new Element('images-workspace', width - 320));
  row.append(seam('image-inspector'));
  row.append(new Element('image-inspector', 314));
  return row;
};
"""


class PaneResizeScriptTests(unittest.TestCase):
    """RESIZE_JS itself, run over a stand-in page.

    The script is the one part of the resizing that no Python call can
    reach, so these run the real string in node and let its own assertions
    report. A machine without node skips them.
    """

    def check(self, checks: str):
        script = f"{RESIZE_PAGE}\nconst start = {app.RESIZE_JS};\n{checks}"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resize.js"
            path.write_text(script)
            result = subprocess.run(
                ["node", str(path)], capture_output=True, text=True, timeout=60
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_page_built_again_comes_back_to_a_fitted_width(self):
        # The nav takes a page out of the document when it turns away and
        # Gradio builds it afresh on the way back, in a row nothing has
        # measured yet. A width fitted to the window while the page was away
        # is a guess, and the page returning is the moment to correct it.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const [rows] = resizers;
assert.ok(rows.targets.has(chat), 'the row on screen is watched from the start');

// The Images page is built the first time the reader opens it.
const first = imagesRow(1200);
mutated();
paint();
assert.ok(rows.targets.has(first), 'a row that has just arrived is watched');

// The nav turns away, taking that page out of the document, and the window
// is dragged narrower while it is gone. With no row left to measure, the
// width the reader chose is cut to what the window alone suggests.
kept.set('chatlab.image-inspector-width', '900');
first.remove();
mutated();
assert.ok(
  !rows.targets.has(first),
  'the row of a page that has been taken away is let go at once'
);
window.innerWidth = 1000;
fire('window', 'resize', {});
paint();
assert.strictEqual(documentElement.style.props['--image-inspector-width'], '380px');

// The nav turns back and the page is built again. Its row has more room
// than the window alone suggested, and the pane is given it.
const second = imagesRow(1000);
mutated();
paint();
assert.ok(rows.targets.has(second), 'the row a rebuilt page comes back in is watched');
assert.ok(!rows.targets.has(first), 'and the row it left is let go');
assert.strictEqual(documentElement.style.props['--image-inspector-width'], '634px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_drag_that_ends_outside_the_window_still_ends(self):
        # A button let go beyond the edge of the window is a release the
        # page never hears, so the pointer is captured for the length of the
        # drag and a pointer that comes back with nothing held ends it.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
const pane = document.getElementById('inspector-pane');
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
assert.ok(handle.hasPointerCapture(7), 'the handle keeps the pointer for the drag');
assert.ok(body.classList.contains('pane-dragging'));

// The pane is on the right of its handle, so dragging left widens it.
fire('window', 'pointermove', { buttons: 1, clientX: 760 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');

// The reader let go out beyond the edge of the window and brought the
// pointer back with the button up.
fire('window', 'pointermove', { buttons: 0, clientX: 600 });
assert.ok(!body.classList.contains('pane-dragging'), 'the page stops being dragged');
assert.ok(!handle.hasPointerCapture(7), 'and the handle gives the pointer back');
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), String(pane.width));

// So moving the pointer over the page again leaves the pane where it was.
fire('window', 'pointermove', { buttons: 1, clientX: 400 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_pane_wider_than_a_drag_allows_is_reported_where_it_is(self):
        # Between the width that stacks the panes and the width that gives
        # them their full share, the stylesheet's smaller default can be
        # more than a drag would leave the workspace. The separator says
        # where the pane is rather than where it would be allowed, and the
        # key asking for it to be pushed out does not pull it in.
        self.check(
            """
// The row keeps 260 for the conversations pane and 6 for the handle, so a
// drag would allow 294 of the 654 left, and the pane is already at 314.
const chat = chatRow(920);
start();
paint();

const handle = chat.children[2];
assert.strictEqual(handle.getAttribute('aria-valuenow'), '314');
assert.strictEqual(handle.getAttribute('aria-valuemax'), '314', 'the range holds it');
assert.strictEqual(handle.getAttribute('aria-valuetext'), '314 pixels');

// ArrowLeft asks for a wider pane. There is no room to widen it, so it
// stays where it is rather than being cut to what a drag would allow.
fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], undefined);
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), undefined);
assert.strictEqual(handle.getAttribute('aria-valuenow'), '314');

// ArrowRight asks for a narrower one, which there is room for.
fire('document', 'keydown', {
  target: handle, key: 'ArrowRight', preventDefault: () => {},
});
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '294px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_key_that_moves_nothing_keeps_the_width_the_reader_chose(self):
        # A pane squeezed by a narrow window is already at its maximum, so
        # the key asking for it to be wider moves nothing. Writing that
        # squeezed width down as the reader's choice would lose the wider
        # one they picked when there was room for it.
        self.check(
            """
const chat = chatRow(1000);
kept.set('chatlab.inspector-pane-width', '520');
start();
paint();

// The row of 1000 keeps 260 for the conversations pane and 6 for the
// handle, so the 520 the reader chose is cut to 374 while the window is
// this narrow. The choice itself is untouched.
const handle = chat.children[2];
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '374px');
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), '520');

// The stand-in page does not lay itself out, so the pane is told what the
// width just written would have made it.
document.getElementById('inspector-pane').width = 374;

fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(
  kept.get('chatlab.inspector-pane-width'), '520',
  'a key with nowhere to go leaves the choice alone'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_click_on_a_handle_chooses_nothing(self):
        # Clicking a handle is how it takes the focus the arrow keys need,
        # and a reader who has chosen nothing has still chosen nothing. A
        # click that pinned the width on screen would take the pane out of
        # the stylesheet's hands, and one made while a narrow window was
        # squeezing the pane would write that squeeze over the wider width
        # the reader picked when there was room for it.
        self.check(
            """
const chat = chatRow(1200);
kept.set('chatlab.inspector-pane-width', '520');
start();
paint();

const handle = chat.children[2];
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointerup', { pointerId: 7 });
assert.strictEqual(
  kept.get('chatlab.inspector-pane-width'), '520', 'the choice is left alone'
);

// A drag that moves the pane is a choice, and is kept.
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 8,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 700, pointerId: 8 });
fire('window', 'pointerup', { pointerId: 8 });
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), String(pane().width));

function pane() { return document.getElementById('inspector-pane'); }
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_only_the_pointer_that_started_a_drag_can_move_or_end_it(self):
        # A second finger on a touch screen reports moves and a release of
        # its own. Neither belongs to the drag the first finger started.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 760, pointerId: 7 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');

// A second finger lands on the same strip, moves and then lifts.
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 300, pointerId: 9,
  preventDefault: () => {},
});
assert.ok(handle.hasPointerCapture(7), 'the first pointer keeps the drag');
fire('window', 'pointermove', { buttons: 1, clientX: 300, pointerId: 9 });
assert.strictEqual(
  documentElement.style.props['--inspector-pane-width'], '354px',
  'the pane does not jump to a pointer that is not dragging it'
);
fire('window', 'pointerup', { pointerId: 9 });
assert.ok(body.classList.contains('pane-dragging'), 'the drag is still going');
assert.ok(handle.hasPointerCapture(7), 'and the first pointer is still held');

// The finger that started the drag lifts, and it ends.
fire('window', 'pointerup', { pointerId: 7 });
assert.ok(!body.classList.contains('pane-dragging'));
assert.ok(!handle.hasPointerCapture(7));
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_separator_carries_the_position_it_has_put_the_pane_in(self):
        # Focusing a separator is meant to tell a screen reader how the room
        # has been divided, and nothing else on the page can say. So every
        # write of a width says it again, and a pane with no width on screen
        # to speak of says nothing at all.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
const pane = document.getElementById('inspector-pane');

// The row of 1200 gives 260 to the conversations pane and 6 to the handle,
// leaving 934 for the pane and its workspace to divide, of which the
// workspace keeps at least 360.
assert.strictEqual(handle.getAttribute('aria-valuemin'), '240');
assert.strictEqual(handle.getAttribute('aria-valuemax'), '574');
// Nobody has dragged anything yet, so the figure is the width the
// stylesheet gave the pane.
assert.strictEqual(handle.getAttribute('aria-valuenow'), String(pane.width));
assert.strictEqual(handle.getAttribute('aria-valuetext'), pane.width + ' pixels');

fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 760 });
fire('window', 'pointerup', {});
assert.strictEqual(handle.getAttribute('aria-valuenow'), '354');
assert.strictEqual(handle.getAttribute('aria-valuetext'), '354 pixels');

// An arrow key steps the same figure along.
fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(handle.getAttribute('aria-valuenow'), '330');

// A double-click hands the pane back to the stylesheet, whose width only
// the layout knows, so the separator reports what it measures next.
fire('document', 'dblclick', { target: handle, preventDefault: () => {} });
paint();
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], undefined);
assert.strictEqual(handle.getAttribute('aria-valuenow'), String(pane.width));

// The handle of a page the nav is not showing measures nothing, and a
// position invented for it would describe a layout that never happened.
const images = imagesRow(0);
mutated();
paint();
assert.strictEqual(images.children[1].getAttribute('aria-valuenow'), null);
"""
        )


class HardwarePanelTests(unittest.TestCase):
    """What the Settings page says about the machine."""

    GB = 1024**3

    def setUp(self):
        self.manager = ModelManager()
        original = runtime.MANAGER
        runtime.MANAGER = self.manager
        self.addCleanup(lambda: setattr(runtime, "MANAGER", original))

    def card(self, **profile):
        return app.hardware_card(model_runtime.DeviceProfile(**profile))

    def test_a_device_not_read_yet_shows_the_memory_and_says_to_wait(self):
        card = self.card(total=48 * self.GB, available=40 * self.GB)

        self.assertIn("not read yet", card)
        self.assertIn("48.0 GB in total", card)
        self.assertIn("40.0 GB", card)
        self.assertIn(app.HARDWARE_UNREAD, card)

    def test_a_metal_machine_reports_the_cap_and_where_it_comes_from(self):
        card = self.card(
            backend="mps",
            dtype="float16",
            total=24 * self.GB,
            available=20 * self.GB,
            ceiling=24 * self.GB,
            recommended=36 * self.GB,
            fraction=24 / 36,
        )

        self.assertIn("Apple Metal (MPS)", card)
        self.assertIn("loaded as float16", card)
        self.assertIn("8-bit and 4-bit", card)
        self.assertIn("24.0 GB, 0.67 of the 36.0 GB Metal recommends", card)
        self.assertIn("mps_memory_fraction", card)
        # The reserve the fit check keeps back is part of the same story.
        self.assertIn("4.0 GB kept beside the weights", card)

    def test_a_machine_without_metal_says_a_quantized_choice_is_ignored(self):
        card = self.card(
            backend="cpu", dtype="float32", total=16 * self.GB, available=8 * self.GB
        )

        self.assertIn("CPU", card)
        self.assertIn("loaded as float32", card)
        self.assertIn("needs Apple Metal", card)
        self.assertNotIn("Metal cap", card)

    def test_the_panel_names_the_model_in_memory(self):
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS), 4-bit weights"
        self.manager.precision = "4-bit"

        card = self.card(backend="mps", dtype="float16", total=48 * self.GB)

        self.assertIn(OLMO, card)
        self.assertIn("4-bit weights", card)

    def test_an_empty_runtime_points_at_the_models_page(self):
        card = self.card(backend="mps", dtype="float16", total=48 * self.GB)

        self.assertIn("none", card)
        self.assertIn("Models page", card)

    def test_a_machine_that_reports_no_memory_says_unknown_rather_than_zero(self):
        card = self.card(backend="cpu", dtype="float32")

        self.assertIn("unknown in total", card)
        self.assertNotIn("0.0 GB in total", card)


class SavedSettingsTests(unittest.TestCase):
    """The settings file the interface opens with and writes back to."""

    def setUp(self):
        self.path = settings.settings_path()
        self.addCleanup(self.forget)

    def forget(self):
        self.path.unlink(missing_ok=True)
        settings.load()

    def build_with(self, **values):
        """The interface as it comes up with ``values`` already saved."""

        settings.write(settings.sanitize(values), path=self.path)
        settings.load()
        self.demo = app.build_app()
        return self.demo

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def listeners(self, name):
        return [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        ]

    def test_every_control_starts_from_the_saved_file(self):
        self.build_with(
            model_id="org/other-model",
            system_prompt="Be brief.",
            assistant_prefill="Well,",
            keep_reasoning=True,
            temperature=0.25,
            top_p=0.5,
            top_k=7,
            max_new_tokens=64,
            seed=99,
            randomize_seed=False,
            analyze_prompt=False,
            color_scale="Surprise",
            prefill_token_limit=2048,
        )

        for label, value in [
            ("Hugging Face model ID", "org/other-model"),
            ("System prompt", "Be brief."),
            ("Assistant prefill (optional)", "Well,"),
            ("Send previous reasoning back to the model", True),
            ("Temperature", 0.25),
            ("Top-p", 0.5),
            ("Top-k (0 disables)", 7),
            ("Maximum new tokens", 64),
            ("Random seed", 99),
            ("🎲 New seed each response", False),
            ("Measure prompt tokens", False),
            ("Color tokens by", "Surprise"),
            ("Context limit (tokens)", 2048),
        ]:
            with self.subTest(label=label):
                self.assertEqual(self.labelled(label).value, value)

    def test_the_message_box_keys_start_from_the_saved_file(self):
        self.build_with(enter_sends=False)

        self.assertFalse(self.labelled("Enter sends the message").value)
        self.assertEqual(
            self.labelled("Message").placeholder,
            app.message_box_settings(enter_sends=False)["placeholder"],
        )

    def test_the_response_length_cannot_exceed_the_context_limit(self):
        self.build_with(prefill_token_limit=2048)

        self.assertEqual(self.labelled("Maximum new tokens").maximum, 2048)

    def test_a_missing_file_leaves_every_control_at_its_default(self):
        self.path.unlink(missing_ok=True)
        settings.load()
        self.demo = app.build_app()

        self.assertEqual(
            self.labelled("Temperature").value, settings.DEFAULTS.temperature
        )
        self.assertEqual(
            self.labelled("Hugging Face model ID").value, settings.DEFAULT_MODEL_ID
        )

    def test_the_file_is_there_to_edit_after_one_launch(self):
        self.path.unlink(missing_ok=True)
        settings.load()

        app.build_app()

        self.assertTrue(self.path.is_file())

    def saving_listeners(self):
        """Every handler that writes the whole set, however it was reached."""

        return self.listeners("remember_settings") + self.listeners(
            "remember_committed_seed"
        )

    def test_changing_any_setting_saves_them_all(self):
        self.build_with()
        saved = self.saving_listeners()
        triggers = {
            self.demo.blocks[block_id]: event
            for fn in saved
            for block_id, event in fn.targets
        }

        for label in [
            "System prompt",
            "Send previous reasoning back to the model",
            "Assistant prefill (optional)",
            "Temperature",
            "Top-p",
            "Top-k (0 disables)",
            "Maximum new tokens",
            "Random seed",
            "🎲 New seed each response",
            "Measure prompt tokens",
            "Color tokens by",
            "Thinking mode",
            "Enter sends the message",
            "Hugging Face model ID",
        ]:
            with self.subTest(label=label):
                self.assertIn(self.labelled(label), triggers)
        # Each one publishes the whole set, in the order the names are in.
        for fn in saved:
            self.assertEqual(len(fn.inputs), len(app.PERSISTED_SETTING_NAMES))

    def test_the_seed_is_saved_when_it_is_committed_and_not_when_it_is_written(self):
        # A finished response leaves the seed that produced it in the box, and
        # saving that would overwrite the seed the reader chose.
        self.build_with()
        events = {}
        for fn in self.saving_listeners():
            for block_id, event in fn.targets:
                events.setdefault(self.demo.blocks[block_id], set()).add(event)

        self.assertEqual(events[self.labelled("Random seed")], {"blur", "submit"})
        # The four sampling controls are saved on input rather than change:
        # switching conversations sets them, and a save from that would put
        # the sampling of the conversation being looked at into the file
        # every unpinned conversation answers with.
        for label in ("Temperature", "Top-p", "Top-k (0 disables)", "Maximum new tokens"):
            self.assertEqual(events[self.labelled(label)], {"input"}, label)
        self.assertEqual(events[self.labelled("Measure prompt tokens")], {"change"})
        # And only the seed box's own events are allowed to write it down.
        for fn in self.listeners("remember_committed_seed"):
            self.assertEqual(
                {self.demo.blocks[block_id] for block_id, _ in fn.targets},
                {self.labelled("Random seed")},
            )

    def test_the_hugging_face_token_is_not_among_the_settings_saved(self):
        self.build_with()
        token_box = self.labelled("Hugging Face token (optional)")

        for fn in self.saving_listeners():
            self.assertNotIn(token_box, fn.inputs)
            self.assertNotIn(token_box, [self.demo.blocks[i] for i, _ in fn.targets])

    def test_the_settings_page_says_where_the_file_is(self):
        self.build_with()
        page = next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "elem_id", None) == "settings-page"
        )
        notes = [
            block.value
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Markdown)
            and getattr(block, "value", None)
            and str(self.path) in str(block.value)
        ]

        self.assertTrue(notes, f"{self.path} is not named on {page.elem_id}")
        self.assertIn("mps_memory_fraction", notes[0])

    def test_saving_a_setting_writes_the_file(self):
        self.build_with()
        values = dict(zip(app.PERSISTED_SETTING_NAMES, [None] * len(app.PERSISTED_SETTING_NAMES)))
        values.update(settings.current().to_mapping())
        values["temperature"] = 0.1
        app.remember_settings(
            *(values[name] for name in app.PERSISTED_SETTING_NAMES)
        )

        self.assertEqual(settings.current().temperature, 0.1)
        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)

    def test_a_pinned_model_is_not_saved_when_something_else_changes(self):
        """``OLMO_MODEL_ID`` names a model for one run, not for every run."""

        with mock.patch.dict(
            os.environ, {"OLMO_MODEL_ID": "org/pinned"}, clear=False
        ):
            self.build_with(model_id="org/saved-model")
            box = self.labelled("Hugging Face model ID")
            self.assertEqual(box.value, "org/pinned")

            values = settings.current().to_mapping() | {
                "enter_sends": settings.current().enter_sends,
                "model_id": box.value,
                "temperature": 0.1,
            }
            app.remember_settings(
                *(values[name] for name in app.PERSISTED_SETTING_NAMES)
            )

        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)
        self.assertEqual(settings.current().model_id, "org/saved-model")
        self.assertEqual(json.loads(self.path.read_text())["model_id"], "org/saved-model")

    def test_a_model_typed_over_a_pinned_one_is_saved(self):
        with mock.patch.dict(
            os.environ, {"OLMO_MODEL_ID": "org/pinned"}, clear=False
        ):
            self.build_with(model_id="org/saved-model")
            values = settings.current().to_mapping() | {
                "enter_sends": settings.current().enter_sends,
                "model_id": "org/typed",
            }
            app.remember_settings(
                *(values[name] for name in app.PERSISTED_SETTING_NAMES)
            )

        self.assertEqual(settings.current().model_id, "org/typed")

    def test_a_generated_seed_is_not_saved_when_something_else_changes(self):
        """A response leaves its own seed in the box; that is not a choice."""

        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {
            "seed": 1234567,  # what a finished response put in the box
            "temperature": 0.1,
        }
        app.remember_settings(*(values[name] for name in app.PERSISTED_SETTING_NAMES))

        self.assertEqual(json.loads(self.path.read_text())["temperature"], 0.1)
        self.assertEqual(settings.current().seed, 99)
        self.assertEqual(json.loads(self.path.read_text())["seed"], 99)

    def test_committing_the_seed_box_saves_what_is_in_it(self):
        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {"seed": 7}
        app.remember_committed_seed(
            *(values[name] for name in app.PERSISTED_SETTING_NAMES)
        )

        self.assertEqual(settings.current().seed, 7)

    def test_a_locked_seed_is_saved_by_any_control(self):
        # With randomization off the box is the reader's alone, and turning it
        # off is how one keeps the seed a response has just used.
        self.build_with(seed=99, randomize_seed=True)
        values = settings.current().to_mapping() | {
            "seed": 1234567,
            "randomize_seed": False,
        }
        app.remember_settings(*(values[name] for name in app.PERSISTED_SETTING_NAMES))

        self.assertEqual(settings.current().seed, 1234567)

    def test_lowering_the_context_limit_pulls_the_response_length_under_it(self):
        self.build_with(prefill_token_limit=8192, max_new_tokens=4096)

        limit, length, _forks = app.remember_prefill_limit(1024, 4096)

        self.assertEqual(limit["value"], 1024)
        self.assertEqual(length["maximum"], 1024)
        self.assertEqual(length["value"], 1024)
        self.assertEqual(settings.current().max_new_tokens, 1024)
        self.assertEqual(json.loads(self.path.read_text())["prefill_token_limit"], 1024)

    def test_a_page_load_puts_the_saved_settings_back_into_the_controls(self):
        self.build_with(temperature=0.4, max_new_tokens=64, prefill_token_limit=2048)
        (restore,) = self.listeners("restore_settings")

        self.assertEqual(restore.targets, [(self.demo._id, "load")])
        self.assertEqual(
            restore.outputs,
            [
                *(
                    self.labelled(label)
                    for label in [
                        "System prompt",
                        "Send previous reasoning back to the model",
                        "Assistant prefill (optional)",
                        "Temperature",
                        "Top-p",
                        "Top-k (0 disables)",
                        "Maximum new tokens",
                        "Random seed",
                        "🎲 New seed each response",
                        "Measure prompt tokens",
                        "Color tokens by",
                        "Thinking mode",
                        "Enter sends the message",
                        "Hugging Face model ID",
                        "Weight precision",
                    ]
                ),
                self.labelled("Context limit (tokens)"),
            ],
        )
        updates = app.restore_settings()
        self.assertEqual(len(updates), len(app.PERSISTED_SETTING_NAMES) + 1)
        published = dict(zip(app.PERSISTED_SETTING_NAMES, updates))
        self.assertEqual(published["temperature"]["value"], 0.4)
        self.assertEqual(published["max_new_tokens"]["value"], 64)
        # The response-length ceiling comes back with it.
        self.assertEqual(published["max_new_tokens"]["maximum"], 2048)
        self.assertEqual(updates[-1]["value"], 2048)

    def test_a_page_load_reads_the_file_again_so_a_hand_edit_takes_effect(self):
        self.build_with(temperature=0.4)
        self.path.write_text(
            json.dumps(settings.current().to_mapping() | {"temperature": 1.1}),
            encoding="utf-8",
        )

        published = dict(zip(app.PERSISTED_SETTING_NAMES, app.restore_settings()))

        self.assertEqual(published["temperature"]["value"], 1.1)
        self.assertEqual(settings.current().temperature, 1.1)

    def test_a_context_limit_outside_its_range_is_pulled_back_into_it(self):
        self.build_with()

        limit, _length, _forks = app.remember_prefill_limit(2, 512)

        self.assertEqual(limit["value"], settings.PREFILL_TOKEN_LIMIT_RANGE[0])


if __name__ == "__main__":
    unittest.main()


MLX = cached(
    "mlx-community/Qwen3-4B-4bit",
    status=CacheStatus(cached_bytes=2_300_000_000, kind=model_runtime.MLX_KIND),
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
        from model_runtime import FITS, Fit

        label = models_page.cached_model_label(MLX, Fit(FITS))

        self.assertIn("· MLX", label)
        self.assertIn("· fits", label)
        self.assertLess(label.index("· MLX"), label.index("· fits"))

    def test_moving_the_precision_radio_does_not_rejudge_a_loaded_mlx_model(self):
        # Load cached at a new precision is how a Transformers model is
        # requantized; an MLX model loads at its own width whatever the radio
        # says, so the loaded one has nothing to be judged again for.
        from model_runtime import DeviceProfile

        self.manager.model_id = MLX.model_id
        self.manager.precision = "4-bit"
        profile = DeviceProfile(backend="mps", dtype="float16", total=10**11, available=10**11)

        self.assertIsNone(models_page.cached_fit(MLX, "full", profile))
        self.assertIsNone(models_page.cached_fit(MLX, "8-bit", profile))

    def test_mlx_recommendations_are_their_own_list_and_judged_at_their_width(self):
        _, _, state, _ = app.search_models(
            "", "", kind=model_runtime.MLX_KIND, order="Recommended"
        )

        self.assertEqual(
            list(state),
            [
                "mlx-community/Qwen3-0.6B-4bit",
                "mlx-community/Qwen3-4B-4bit",
                "mlx-community/Olmo-3-7B-Think-4bit",
            ],
        )
        self.assertEqual(models_page.results_kind(state), model_runtime.MLX_KIND)
        _, detail, _ = models_page.select_search_result(
            state, "full", picked("mlx-community/Qwen3-4B-4bit")
        )
        self.assertIn("quantized already", detail)
        self.assertNotIn("Choosing 4-bit or 8-bit", detail)

    def test_an_mlx_result_is_sized_from_the_width_in_its_name(self):
        from model_runtime import DeviceProfile, HubModel, estimate_parameter_bytes

        profile = DeviceProfile(backend="mps", dtype="float16", total=10**11, available=10**11)
        four_bit = HubModel(
            model_id="mlx-community/Some-7B-4bit", parameters=7_000_000_000, kind=model_runtime.MLX_KIND
        )
        unnamed = HubModel(
            model_id="mlx-community/Some-7B", parameters=7_000_000_000, kind=model_runtime.MLX_KIND
        )

        # The radio says full; the name says 4 bits, and the name wins.
        packed = models_page.hub_fit(four_bit, "full", profile, model_runtime.MLX_KIND)
        whole = models_page.hub_fit(unnamed, "4-bit", profile, model_runtime.MLX_KIND)

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
            status=CacheStatus(cached_bytes=2 * GB, kind=model_runtime.MLX_KIND),
            path=folder,
        )
        profile = model_runtime.DeviceProfile(
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
        for model_id, kind in ((MLX.model_id, model_runtime.MLX_KIND), (OLMO, TEXT_KIND)):
            folder = lay_out(
                root, model_id, {"config.json": config.encode(), "model.safetensors": b""}
            )
            with (folder / "blobs" / "blob1").open("r+b") as weights:
                weights.truncate(20 * GB)
            entries.append(
                cached(model_id, status=CacheStatus(cached_bytes=20 * GB, kind=kind), path=folder)
            )
        self.entries = entries
        capped = model_runtime.DeviceProfile(
            backend="mps",
            dtype="float16",
            total=16 * GB,
            available=16 * GB,
            ceiling=16 * GB,
            pool="Metal on this machine",
        )
        with mock.patch.object(models_page, "device_profile", lambda torch=None: capped):
            with mock.patch.object(
                model_runtime, "system_memory", lambda: (48 * GB, 30 * GB)
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
