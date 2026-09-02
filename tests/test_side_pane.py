"""The side pane: My Models, Model search, and where the settings live."""

import json
import os
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
    HubModel,
    ModelManager,
    format_count,
    list_cached_models,
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

    def test_the_newest_download_comes_first(self):
        with tempfile.TemporaryDirectory() as root:
            older = lay_out(root, "org/older", {"config.json": b"{}"})
            lay_out(root, "org/newer", {"config.json": b"{}"})
            for blob in (older / "blobs").iterdir():
                os.utime(blob, (1_000_000, 1_000_000))
            listed = [entry.model_id for entry in list_cached_models(Path(root))]

        self.assertEqual(listed, ["org/newer", "org/older"])


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


class SidePaneLayoutTests(unittest.TestCase):
    """The settings and model controls sit in the pane; the conversation does not."""

    IN_PANE = [
        "Hugging Face model ID",
        "Hugging Face token (optional)",
        "Downloaded models",
        "Search Hugging Face",
        "Search results",
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
    ON_PAGE = ["Conversation", "Message", "Color tokens by", "Text to score"]

    def setUp(self):
        self.demo = app.build_app()
        (self.pane,) = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Sidebar)
        ]

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def in_pane(self, block) -> bool:
        parent = getattr(block, "parent", None)
        while parent is not None:
            if parent is self.pane:
                return True
            parent = getattr(parent, "parent", None)
        return False

    def test_the_pane_holds_the_model_controls_and_settings(self):
        for label in self.IN_PANE:
            with self.subTest(label=label):
                self.assertTrue(self.in_pane(self.labelled(label)))

    def test_the_conversation_stays_on_the_page(self):
        for label in self.ON_PAGE:
            with self.subTest(label=label):
                self.assertFalse(self.in_pane(self.labelled(label)))

    def test_the_pane_is_thin(self):
        self.assertEqual(self.pane.width, app.SIDE_PANE_WIDTH)
        self.assertLessEqual(app.SIDE_PANE_WIDTH, 360)

    def listeners(self, name):
        return [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        ]

    def test_every_model_change_rescans_the_cache(self):
        # Download, download-and-load, load cached, unload, the refresh
        # button, and the page load: each ends in a rescan.
        self.assertEqual(len(self.listeners("refresh_my_models")), 6)

    def test_choosing_a_model_writes_the_id_box(self):
        box = self.labelled("Hugging Face model ID")
        for name in ("select_my_model", "select_search_result"):
            with self.subTest(handler=name):
                (listener,) = self.listeners(name)
                self.assertIs(listener.outputs[0], box)


if __name__ == "__main__":
    unittest.main()
