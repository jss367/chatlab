"""The Models page: My Models, Model search, and where the settings live."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import app
import model_runtime
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

COMMIT = "d97e442d7cc678210054dbcc9b440894d62c89a4"
OLMO = "allenai/Olmo-3-7B-Think"


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
        self.assertEqual(entry.size_bytes, 100 + len(self.CONFIG))
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
        self.assertEqual(entry.size_bytes, len(self.CONFIG) + 40)
        # The config is on disk, so what the model is can still be said.
        self.assertEqual(entry.architecture, "Olmo3ForCausalLM")

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

    def test_a_malformed_id_is_refused(self):
        with self.assertRaises(ValueError):
            self.manager.remove("nonsense", Path(self.root.name))


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
        "Temperature",
        "Top-p",
        "Top-k (0 disables)",
        "Maximum new tokens",
        "Random seed",
        "🎲 New seed each response",
        "Measure prompt tokens",
        "Enter sends the message",
    ]
    ON_CHAT_PAGE = ["Conversation", "Message", "Color tokens by", "Text to score"]

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

    def test_the_nav_pane_is_thin(self):
        self.assertLessEqual(app.NAV_PANE_WIDTH, 100)
        self.assertEqual(self.by_id("nav-pane").min_width, app.NAV_PANE_WIDTH)

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

    def test_every_model_change_rescans_the_cache(self):
        # Download, download-and-load, load cached, unload, redownload,
        # confirmed removal, the refresh button, a new sort order, and the
        # page load: each ends in a rescan.
        self.assertEqual(len(self.listeners("refresh_my_models")), 9)

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

    def test_choosing_a_model_writes_the_id_box(self):
        box = self.labelled("Hugging Face model ID")
        for name in ("select_my_model", "select_search_result"):
            with self.subTest(handler=name):
                (listener,) = self.listeners(name)
                self.assertIs(listener.outputs[0], box)


if __name__ == "__main__":
    unittest.main()
