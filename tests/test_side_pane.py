"""The Models page: My Models, Model search, and where the settings live."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import app
import model_runtime
import settings
from model_runtime import (
    MODEL_WEIGHTS,
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

        def fake_load(model_id, local_path, torch, progress=None):
            seen.append((manager.loading_id, manager._lock.locked()))
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            self.assertEqual(manager.load(OLMO, Path("/snap")), "CPU")

        self.assertEqual(seen, [(OLMO, True)])
        self.assertIsNone(manager.loading_id)

    def test_a_load_waiting_for_the_lock_is_already_named(self):
        # A generation holds the lock for as long as its reply takes; the
        # load queued behind it must count as under way from the click.
        import threading

        manager = ModelManager()
        manager._lock.acquire()
        entered = threading.Event()

        def fake_load(model_id, local_path, torch, progress=None):
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

        def fail(model_id, local_path, torch, progress=None):
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
        import threading

        manager = ModelManager()
        second = "org/second"
        in_first = threading.Event()
        in_second = threading.Event()
        let_first_finish = threading.Event()
        let_second_finish = threading.Event()

        def fake_load(model_id, local_path, torch, progress=None):
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
        import threading

        manager = ModelManager()
        reading = threading.Event()
        let_it_finish = threading.Event()

        def fake_load(model_id, local_path, torch, progress=None):
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
    "runwayml/stable-diffusion-v1-5",
    status=CacheStatus(cached_bytes=5_500_000_000, unsupported=True),
    architecture=None,
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
        originals = (app.MANAGER, app.list_cached_models, app.cache_root)
        app.MANAGER = self.manager
        app.list_cached_models = lambda: list(self.entries)
        app.cache_root = lambda: Path("/cache")
        self.addCleanup(
            lambda: setattr(app, "MANAGER", originals[0])
            or setattr(app, "list_cached_models", originals[1])
            or setattr(app, "cache_root", originals[2])
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
        self.assertIn("not a Transformers language model", detail)
        self.assertNotIn("Incomplete", detail)
        self.assertNotIn("Download and load", detail)

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
        self.assertIn("Model search", summary)

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


class ManageMyModelsTests(unittest.TestCase):
    """Redownloading and removing a model from My Models."""

    def setUp(self):
        self.entries = [cached(OLMO), PARTIAL]
        self.manager = ModelManager()
        self.removed = []
        originals = (
            app.MANAGER,
            app.list_cached_models,
            model_runtime.remove_cached_model,
            app.download_model,
        )
        app.MANAGER = self.manager
        app.list_cached_models = lambda: list(self.entries)
        # The manager deletes through the module-level function, so that is
        # what stands in: the manager's own checks stay real.
        model_runtime.remove_cached_model = self.remove
        app.download_model = self.download
        self.addCleanup(
            lambda: setattr(app, "MANAGER", originals[0])
            or setattr(app, "list_cached_models", originals[1])
            or setattr(model_runtime, "remove_cached_model", originals[2])
            or setattr(app, "download_model", originals[3])
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


class ModelSearchPaneTests(unittest.TestCase):
    """What Model search lists, and what choosing a result does."""

    def setUp(self):
        self.results = [INSTRUCT, GATED]
        self.queries = []
        original_search, original_status = app.search_hub_models, app.cache_status
        app.search_hub_models = self.search
        app.cache_status = lambda model_id: CacheStatus()
        self.addCleanup(
            lambda: setattr(app, "search_hub_models", original_search)
            or setattr(app, "cache_status", original_status)
        )

    def search(self, query, hf_token):
        self.queries.append((query, hf_token))
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results)

    def test_results_are_listed_with_size_and_popularity(self):
        radio, detail, state = app.search_models("  olmo 3 ", "tok")

        self.assertEqual(self.queries, [("olmo 3", "tok")])
        self.assertEqual(
            radio["choices"],
            [
                (
                    "allenai/Olmo-3-7B-Instruct · 7.3B params · 281K downloads",
                    "allenai/Olmo-3-7B-Instruct",
                ),
                ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
            ],
        )
        self.assertIsNone(radio["value"])
        self.assertIn("2 results", detail)
        self.assertEqual(set(state), {INSTRUCT.model_id, GATED.model_id})

    def test_an_empty_query_does_not_go_online(self):
        radio, detail, state = app.search_models("   ", "")

        self.assertEqual(self.queries, [])
        self.assertEqual(radio["choices"], [])
        self.assertEqual(detail, app.SEARCH_HINT)
        self.assertEqual(state, {})

    def test_a_failed_search_is_reported(self):
        self.results = ConnectionError("hub <unreachable>")

        radio, detail, state = app.search_models("olmo", "")

        self.assertEqual(radio["choices"], [])
        self.assertIn("Search failed", detail)
        self.assertIn("hub &lt;unreachable&gt;", detail)
        self.assertEqual(state, {})

    def test_no_matches_is_said_plainly(self):
        self.results = []

        radio, detail, _ = app.search_models("zzzz", "")

        self.assertEqual(radio["choices"], [])
        self.assertIn("No text-generation models matched", detail)
        self.assertIn("zzzz", detail)

    def test_choosing_a_result_fills_the_id_box_and_describes_it(self):
        _, _, state = app.search_models("olmo", "")

        box, detail = app.select_search_result(INSTRUCT.model_id, state)

        self.assertEqual(box["value"], INSTRUCT.model_id)
        self.assertIn("https://huggingface.co/allenai/Olmo-3-7B-Instruct", detail)
        self.assertIn("7.3B", detail)
        self.assertIn("281K downloads", detail)
        self.assertIn("143 likes", detail)
        self.assertIn("apache-2.0", detail)
        self.assertIn("2026-06-25", detail)
        self.assertNotIn("Gated", detail)
        self.assertIn("Download and load", detail)

    def test_a_gated_result_says_a_token_is_needed(self):
        _, _, state = app.search_models("llama", "")

        _, detail = app.select_search_result(GATED.model_id, state)

        self.assertIn("Gated", detail)
        self.assertIn("token", detail)

    def test_a_result_already_on_disk_says_so(self):
        app.cache_status = lambda model_id: CacheStatus(cached_bytes=15_000_000_000)
        _, _, state = app.search_models("olmo", "")

        _, detail = app.select_search_result(INSTRUCT.model_id, state)

        self.assertIn("Already cached", detail)
        self.assertIn("15.0 GB cached", detail)

    def test_a_cached_result_of_another_kind_is_not_called_partly_cached(self):
        app.cache_status = lambda model_id: CacheStatus(
            cached_bytes=5_500_000_000, unsupported=True
        )
        _, _, state = app.search_models("olmo", "")

        _, detail = app.select_search_result(INSTRUCT.model_id, state)

        self.assertIn("Already cached", detail)
        self.assertIn("not a model ChatLab can load", detail)
        self.assertNotIn("Partly cached", detail)
        self.assertNotIn("Download and load", detail)

    def test_a_partly_downloaded_result_says_so(self):
        app.cache_status = lambda model_id: CacheStatus(
            cached_bytes=100, missing_files=(MODEL_WEIGHTS,)
        )
        _, _, state = app.search_models("olmo", "")

        _, detail = app.select_search_result(INSTRUCT.model_id, state)

        self.assertIn("Partly cached", detail)

    def test_an_unreadable_cache_leaves_the_result_uncached(self):
        def refuse(model_id):
            raise PermissionError(13, "Permission denied")

        app.cache_status = refuse
        _, _, state = app.search_models("olmo", "")

        box, detail = app.select_search_result(INSTRUCT.model_id, state)

        self.assertEqual(box["value"], INSTRUCT.model_id)
        self.assertNotIn("cached", detail)
        self.assertIn("Download and load", detail)

    def test_choosing_nothing_leaves_the_id_box_alone(self):
        self.assertEqual(
            app.select_search_result(None, {}), (gr.skip(), app.NO_RESULT_SELECTED)
        )
        self.assertEqual(
            app.select_search_result("stale/pick", {}),
            (gr.skip(), app.NO_RESULT_SELECTED),
        )


class ModelBadgeTests(unittest.TestCase):
    """The chat page's badge: the model in memory, or the lack of one."""

    def setUp(self):
        self.manager = ModelManager()
        original = app.MANAGER
        app.MANAGER = self.manager
        self.addCleanup(lambda: setattr(app, "MANAGER", original))

    def load(self):
        self.manager.model = object()
        self.manager.tokenizer = object()
        self.manager.model_id = OLMO
        self.manager.device_name = "Apple Metal (MPS)"

    def test_a_loaded_model_is_named_with_its_device(self):
        self.load()
        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="ready"', badge)
        self.assertIn(OLMO, badge)
        self.assertIn("Apple Metal (MPS)", badge)
        # Nothing to go to the Models page for, and nothing to offer.
        self.assertFalse(button["visible"])
        self.assertFalse(offer["visible"])

    def test_no_model_says_so_and_offers_the_way_to_the_models_page(self):
        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="empty"', badge)
        self.assertIn(app.NO_MODEL_BADGE, badge)
        self.assertTrue(button["visible"])
        # And the offer beside it says which way it would go.
        self.assertTrue(offer["visible"])
        self.assertEqual(offer["value"], app.default_model_offer())

    def test_a_load_under_way_names_the_model_coming_in(self):
        # model_id is cleared for the whole of a load, so the badge reads
        # loading_id and reports the minutes in between as a load.
        self.manager.reserve_load(OLMO)

        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn(f"Loading {OLMO}", badge)
        self.assertFalse(button["visible"])
        self.assertFalse(offer["visible"])

    def test_a_second_load_finishing_leaves_the_first_one_showing(self):
        # Two loads can be under way at once, and the one that finishes
        # first must not give back the other's claim: the badge would then
        # say nothing was loaded in the middle of a load.
        self.manager.reserve_load(OLMO)
        _second_id, second = self.manager.reserve_load("org/second")
        self.manager.release_load(second)

        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn(f"Loading {OLMO}", badge)
        self.assertFalse(button["visible"])
        self.assertFalse(offer["visible"])

    def test_a_load_waiting_its_turn_leaves_the_answering_model_named(self):
        # A load counts itself as under way before it waits for the model
        # lock, so asking for a second model mid-reply queues it behind the
        # generation. The model still producing the tokens is the one the
        # badge is for, so it keeps the name until the load empties memory.
        self.load()
        self.manager.reserve_load("org/second")

        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="ready"', badge)
        self.assertIn(OLMO, badge)
        self.assertNotIn("Loading", badge)
        self.assertFalse(button["visible"])

    def test_a_load_that_has_emptied_memory_names_the_model_coming_in(self):
        # Once the queued load wins the lock it unloads first, and from then
        # on there is nothing in memory to name.
        self.load()
        self.manager.reserve_load("org/second")
        self.manager.model = None
        self.manager.tokenizer = None
        self.manager.model_id = None
        self.manager.device_name = None

        badge, offer, button = app.refresh_model_badge()

        self.assertIn('data-state="loading"', badge)
        self.assertIn("Loading org/second", badge)
        self.assertFalse(button["visible"])

    def test_the_model_id_is_escaped(self):
        self.load()
        self.manager.model_id = "org/<script>"

        self.assertIn("&lt;script&gt;", app.loaded_model_badge())


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
        ]:
            container = self.by_id(page)
            for label in labels:
                with self.subTest(page=page, label=label):
                    self.assertTrue(self.within(self.labelled(label), container))

    def test_the_nav_offers_the_three_pages_and_starts_on_chat(self):
        nav = self.by_id("nav")
        self.assertIsInstance(nav, gr.Radio)
        self.assertEqual([value for _, value in nav.choices], ["Chat", "Models", "Settings"])
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

    def test_the_nav_names_are_on_screen_rather_than_a_hover_away(self):
        # Three pages is not a number worth hiding. Nothing clips the name
        # out of sight, and no tooltip stands in for it.
        self.assertNotIn("clip-path: inset(50%)", app.CSS)
        self.assertNotIn("#nav label::after", app.CSS)
        self.assertNotIn(":hover::after", app.CSS)

    def test_only_the_chat_page_starts_visible(self):
        self.assertTrue(self.by_id("chat-page").visible)
        self.assertTrue(self.by_id("conversation-pane").visible)
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
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )
        shown = lambda page: [update["visible"] for update in app.show_page(page)]
        # The conversations pane comes and goes with Chat.
        self.assertEqual(shown("Chat"), [True, True, False, False])
        self.assertEqual(shown("Models"), [False, False, True, False])
        self.assertEqual(shown("Settings"), [False, False, False, True])

    def listeners(self, name):
        return [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        ]

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
        button = self.by_id("load-model")
        self.assertTrue(self.within(badge, chat_page))
        self.assertTrue(self.within(button, chat_page))
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
        # The three that change memory, the download that only changes what
        # is on disk, redownload, a confirmed removal and the offer's own
        # chain, plus the page load, the nav and the timer.
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
        self.assertEqual(recovery.outputs, [self.by_id("score-budget")])

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
            if isinstance(block, gr.Timer)
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

    def test_the_badge_button_sends_the_nav_to_the_models_page(self):
        (listener,) = self.listeners("go_to_models")
        self.assertEqual(listener.targets, [(self.by_id("load-model")._id, "click")])
        # The button switches the pages itself: a Radio set by a handler
        # reports no change, so the nav's own handler would not run.
        self.assertEqual(
            listener.outputs,
            [
                self.by_id("nav"),
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )
        page, *panes = app.go_to_models()
        self.assertEqual(page, "Models")
        self.assertEqual(
            [update["visible"] for update in panes], [False, False, True, False]
        )

    def test_every_model_change_rescans_the_cache(self):
        # Download, download-and-load, load cached, unload, redownload,
        # confirmed removal, the banner's own setup of the default model,
        # the refresh button, a new sort order, and the page load: each ends
        # in a rescan.
        self.assertEqual(len(self.listeners("refresh_my_models")), 10)

    def test_removal_asks_before_deleting(self):
        # The Remove button only opens the question; deleting is the
        # confirm button's job, and choosing another model withdraws it.
        (ask,) = self.listeners("ask_remove_my_model")
        (remove,) = self.listeners("remove_my_model")
        buttons = {
            self.demo.blocks[block_id].value: fn
            for fn in (ask, remove)
            for block_id, _ in fn.targets
        }
        self.assertIs(buttons["🗑️ Remove"], ask)
        self.assertIs(buttons["Remove from disk"], remove)
        self.assertEqual(len(self.listeners("hide_remove_confirm")), 2)

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
        # the question; the confirm button is the one that clears, and so the
        # only one that cancels the running generators.
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
        self.assertTrue(self.cancelled_by(clear.targets[0]))

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
        (setup,) = self.listeners("start_default_model")
        self.assertEqual(setup.targets, [(offer._id, "click")])
        # It switches the pages itself, for the reason go_to_models gives: a
        # Radio set by a handler reports no change, so setting the nav alone
        # would tick Models and leave the chat page on screen.
        self.assertEqual(
            setup.outputs,
            [
                self.labelled("Hugging Face model ID"),
                self.by_id("nav"),
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )

    def test_the_offer_is_published_wherever_the_badge_is(self):
        # What the offer would do depends on what is on disk, which a
        # download changes. It rides the badge's own refresh, so the timer
        # that keeps the badge honest in every open tab keeps the offer
        # honest too.
        listeners = self.listeners("refresh_model_badge")
        self.assertTrue(listeners)
        for listener in listeners:
            self.assertEqual(
                listener.outputs,
                [
                    self.by_id("model-badge"),
                    self.by_id("default-model"),
                    self.by_id("load-model"),
                ],
            )

    def test_escape_is_wired_to_the_stop_button_by_its_id(self):
        # The shortcut presses the button rather than reaching past it, so
        # whatever Stop does, Escape does. It needs the id to find it.
        stop = next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "value", None) == "Stop"
        )

        self.assertEqual(stop.elem_id, "stop-button")
        self.assertIn("#stop-button", app.SHORTCUT_JS)
        self.assertIn("offsetParent", app.SHORTCUT_JS)
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

    def test_the_sampling_summary_follows_the_sliders_on_release(self):
        # A slider fires continuously while it is dragged; the label only has
        # to be right once it is let go.
        sliders = [
            self.labelled(label)
            for label in ("Temperature", "Top-p", "Top-k (0 disables)", "Maximum new tokens")
        ]
        listeners = self.listeners("update_sampling_label")
        released = [fn for fn in listeners if fn.targets[0][1] == "release"]

        self.assertEqual(
            [self.demo.blocks[fn.targets[0][0]] for fn in released], sliders
        )
        for fn in listeners:
            self.assertEqual(fn.inputs, sliders)
        # The other three are the paths that move a slider without anyone
        # touching it: the settings file read back on load, and the context
        # limit committed, which can pull the response length down with it.
        self.assertEqual(len(listeners) - len(released), 3)

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
        self.assertEqual(
            len(listeners) - len(typed), len(self.listeners("refresh_my_models")) - 3
        )

    def test_choosing_a_model_writes_the_id_box(self):
        box = self.labelled("Hugging Face model ID")
        for name in ("select_my_model", "select_search_result"):
            with self.subTest(handler=name):
                (listener,) = self.listeners(name)
                self.assertIs(listener.outputs[0], box)


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
        self.assertEqual(events[self.labelled("Temperature")], {"change"})
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
        values = dict(zip(app.PERSISTED_SETTING_NAMES, [None] * 13))
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

        limit, length = app.remember_prefill_limit(1024, 4096)

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
                        "Enter sends the message",
                        "Hugging Face model ID",
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

        limit, _length = app.remember_prefill_limit(2, 512)

        self.assertEqual(limit["value"], settings.PREFILL_TOKEN_LIMIT_RANGE[0])


if __name__ == "__main__":
    unittest.main()
