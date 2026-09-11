"""Repository checks distinguish existence, access, and local availability."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from model_runtime import CacheStatus, cache_status, list_cached_models
from ui import model_repository as repository, models_page


class RepositoryTests(unittest.TestCase):
    def check(self, info=None, error=None, model_id="org/model", token="", config=None, config_error=None):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "huggingface_hub.HfApi.model_info", return_value=info, side_effect=error
        ) as call:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config if config is not None else {
                "model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64},
            }))
            with mock.patch(
                "huggingface_hub.hf_hub_download", return_value=str(config_path),
                side_effect=config_error,
            ) as download:
                states = list(repository.check_model_repository(model_id, token))
            self.config_download = download
        return states, call

    def info(self, **kwargs):
        return SimpleNamespace(**{
            "tags": ["mlx"], "config": {"model_type": "qwen3_5"},
            "library_name": "mlx", "private": False, "gated": False,
            "sha": "a" * 40,
            "siblings": [
                SimpleNamespace(rfilename="model.safetensors", size=17_000_000_000),
                SimpleNamespace(rfilename="config.json", size=200),
            ],
            **kwargs,
        })

    def test_mlx_lookup_checks_metadata_and_shows_fixed_precision(self):
        with mock.patch("mlx_runtime.mlx_supports", return_value=True):
            states, call = self.check(self.info(), model_id="org/model-4bit", token=" secret ")
        call.assert_called_once_with("org/model-4bit", token="secret", files_metadata=True, timeout=10)
        self.assertEqual(states[0]["status"], "checking")
        result = states[-1]
        self.assertEqual(result["download_bytes"], 17_000_000_200)
        self.config_download.assert_called_once_with(
            "org/model-4bit", "config.json", revision="a" * 40, token="secret", etag_timeout=10,
            cache_dir=mock.ANY,
        )
        detail, precision = repository.repository_view("org/model-4bit", result, "secret")
        self.assertIn("Repository found", detail)
        self.assertIn("4-bit", detail)
        self.assertFalse(precision["visible"])
        self.assertNotIn("secret", str(states))

    def test_config_inspection_leaves_the_model_cache_and_inventory_untouched(self):
        from huggingface_hub import constants

        for contents in ('{"model_type":"qwen3_5","quantization":{"bits":4}}', "invalid json"):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as default_cache:
                download_caches = []

                def download(repo_id, filename, *, revision, token, etag_timeout, cache_dir=None):
                    # Reproduce the Hub's cache writes, including its fallback
                    # to the user's model cache when no isolated cache is given.
                    root = Path(cache_dir or constants.HF_HUB_CACHE)
                    download_caches.append(root)
                    model = root / ("models--" + repo_id.replace("/", "--"))
                    blob = model / "blobs" / "config-blob"
                    blob.parent.mkdir(parents=True)
                    blob.write_text(contents)
                    config = model / "snapshots" / revision / filename
                    config.parent.mkdir(parents=True)
                    config.symlink_to(blob)
                    return str(config)

                with mock.patch.object(constants, "HF_HUB_CACHE", default_cache), mock.patch(
                    "huggingface_hub.HfApi.model_info", return_value=self.info()
                ), mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
                    states = list(repository.check_model_repository("org/model", "secret"))
                    self.assertEqual(states[-1]["status"], "found")
                    self.assertFalse(cache_status("org/model").present)
                    self.assertEqual(list_cached_models(), [])
                    self.assertEqual(list(Path(default_cache).iterdir()), [])
                self.assertEqual(len(download_caches), 1)
                self.assertNotEqual(download_caches[0], Path(default_cache))
                self.assertFalse(download_caches[0].exists())

    def test_unquantized_mlx_tags_or_names_do_not_hide_precision(self):
        for library, tags in (("mlx", []), ("transformers", ["mlx"])):
            with self.subTest(library=library):
                states, _ = self.check(
                    self.info(library_name=library, tags=tags), model_id="org/model-4bit",
                    config={"model_type": "qwen3_5", "torch_dtype": "bfloat16"},
                )
                detail, precision = repository.repository_view("org/model-4bit", states[-1])
                self.assertTrue(precision["visible"])
                self.assertEqual(states[-1]["format"], "Transformers")
                self.assertIsNone(states[-1]["bits"])
                self.assertNotIn("Precision is fixed", detail)

    def test_config_detects_quantized_mlx_without_tags_or_a_bit_width_in_the_name(self):
        states, _ = self.check(self.info(tags=[], library_name="transformers"), config={
            "model_type": "qwen3_5", "quantization_config": {"bits": 8, "group_size": 64},
        })
        detail, precision = repository.repository_view("org/model", states[-1])
        self.assertIn("MLX · 8-bit weights", detail)
        self.assertFalse(precision["visible"])

    def test_failed_config_lookup_preserves_metadata_and_only_denies_proven_access_failures(self):
        request = httpx.Request("GET", "https://huggingface.co/org/model/resolve/main/config.json")
        for error, denied in (
            (httpx.ConnectError("offline"), False),
            (HfHubHTTPError("missing config", response=httpx.Response(404, request=request)), False),
            (GatedRepoError("gated", response=httpx.Response(403, request=request)), True),
            (HfHubHTTPError("invalid token", response=httpx.Response(401, request=request)), True),
        ):
            with self.subTest(error=type(error).__name__):
                states, _ = self.check(self.info(gated="auto"), config_error=error)
                self.assertEqual(states[-1]["status"], "found")
                self.assertEqual(states[-1]["access_restricted"], denied)
                detail, precision = repository.repository_view("org/model", states[-1])
                self.assertTrue(precision["visible"])
                self.assertIn("Repository found", detail)
                self.assertEqual("Access required" in detail, denied)
                self.assertIn("Configuration could not be verified", detail)
                for cached_status in (CacheStatus(), CacheStatus(cached_bytes=100)):
                    with mock.patch.object(models_page, "cache_status", return_value=cached_status):
                        _, load, download, cached = models_page.refresh_model_actions(
                            "org/model", None, states[-1]
                        )
                    self.assertEqual(load["interactive"], not denied)
                    self.assertEqual(download["interactive"], not denied)
                    self.assertEqual(cached["visible"], cached_status.complete)

    def test_gated_repository_with_configuration_access_allows_download(self):
        states, _ = self.check(self.info(gated="auto"))
        detail, _ = repository.repository_view("org/model", states[-1])
        self.assertIn("Access to the configuration was verified", detail)
        self.assertNotIn("Access required", detail)
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
            _, load, download, _ = models_page.refresh_model_actions("org/model", None, states[-1])
        self.assertTrue(load["interactive"])
        self.assertTrue(download["interactive"])

    def test_cached_quantization_controls_precision_offline_and_follows_selected_model(self):
        from huggingface_hub import constants

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            constants, "HF_HUB_CACHE", directory
        ), mock.patch("huggingface_hub.HfApi.model_info") as online_check:
            for name, config in (
                ("quantized", {"model_type": "qwen3_5", "quantization": {"bits": 4}}),
                ("unquantized", {"model_type": "qwen3_5", "torch_dtype": "bfloat16"}),
            ):
                folder = Path(directory) / f"models--org--{name}"
                snapshot = folder / "snapshots" / ("a" * 40)
                snapshot.mkdir(parents=True)
                (folder / "refs").mkdir()
                (folder / "refs" / "main").write_text("a" * 40)
                (snapshot / "config.json").write_text(json.dumps(config))
                (snapshot / "model.safetensors").write_bytes(b"weights")
            for result in (
                None,
                {"model_id": "org/other", "token_scope": repository.token_scope(None), "mlx": True},
                {"model_id": "org/quantized", "token_scope": repository.token_scope("old-token"), "mlx": True},
                {"model_id": "org/quantized", "token_scope": repository.token_scope(None), "status": "error", "detail": "Offline"},
            ):
                with self.subTest(result=result):
                    detail, precision = repository.repository_view(
                        "org/unquantized", result, selected="org/quantized"
                    )
                    self.assertFalse(precision["visible"])
                    self.assertIn("Cached checkpoint", detail)
                    self.assertIn("4-bit weights", detail)
                    self.assertNotIn("Repository found", detail)
            for selected in ("org/unquantized", "org/missing"):
                detail, precision = repository.repository_view("org/quantized", None, selected=selected)
                self.assertTrue(precision["visible"])
                self.assertNotIn("Cached checkpoint", detail)
            online_check.assert_not_called()

    def test_a_late_old_token_response_cannot_restore_the_result_for_the_same_id(self):
        request = httpx.Request("GET", "https://huggingface.co/api/models/org/model")
        missing = RepositoryNotFoundError("hidden", response=httpx.Response(404, request=request))
        for error in (None, missing):
            with self.subTest(error=error):
                old_states, _ = self.check(self.info(), token="old-token", error=error)
                old = old_states[-1]
                # The old request finishes after the user has changed credentials.
                detail, precision = repository.repository_view("org/model", old, "new-token")
                self.assertEqual(detail, repository.UNCHECKED)
                self.assertTrue(precision["visible"])
                with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
                    _, load, download, _ = models_page.refresh_model_actions(
                        "org/model", None, old, "new-token"
                    )
                self.assertTrue(load["interactive"])
                self.assertTrue(download["interactive"])
        states, _ = self.check(self.info(), token="new-token")
        detail, precision = repository.repository_view("org/model", states[-1], " new-token ")
        self.assertIn("Repository found", detail)
        self.assertFalse(precision["visible"])
        self.assertNotIn("new-token", str(states))

    def test_a_late_response_cannot_verify_or_disable_an_edited_id(self):
        old = {"model_id": "org/old", "token_scope": repository.token_scope(None), "status": "missing", "detail": "Repository not found"}
        detail, precision = repository.repository_view("org/new", old)
        self.assertEqual(detail, repository.UNCHECKED)
        self.assertTrue(precision["visible"])
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
            _, download, _, _ = models_page.refresh_model_actions("org/new", None, old)
        self.assertTrue(download["interactive"])

    def test_missing_private_gated_and_network_failure_are_distinct(self):
        request = httpx.Request("GET", "https://huggingface.co/api/models/org/model")
        for error, status, wording in (
            (RepositoryNotFoundError("hidden", response=httpx.Response(404, request=request)), "missing", "not found or private"),
            (GatedRepoError("gated", response=httpx.Response(403, request=request)), "restricted", "Access required"),
            (httpx.ConnectError("connection failed"), "error", "Could not reach"),
        ):
            with self.subTest(status=status):
                states, _ = self.check(error=error)
                self.assertEqual(states[-1]["status"], status)
                self.assertIn(wording, states[-1]["detail"])

    def test_invalid_id_does_not_make_a_request(self):
        states, call = self.check(model_id="invalid")
        call.assert_not_called()
        self.assertEqual(states[-1]["status"], "invalid")

    def test_cached_loading_survives_failed_repository_checks(self):
        for status in ("missing", "restricted", "error", "checking"):
            with self.subTest(status=status), mock.patch.object(
                models_page, "cache_status", return_value=CacheStatus(cached_bytes=100)
            ):
                _, _, _, cached = models_page.refresh_model_actions(
                    "org/model", None, {"model_id": "org/model", "token_scope": repository.token_scope(None), "status": status}
                )
                self.assertTrue(cached["visible"])

    def test_missing_repository_disables_download_until_rechecked_or_edited(self):
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
            _, load, download, _ = models_page.refresh_model_actions(
                "org/model", None, {"model_id": "org/model", "token_scope": repository.token_scope(None), "status": "missing"}
            )
        self.assertFalse(load["interactive"])
        self.assertFalse(download["interactive"])

    def test_incomplete_file_sizes_do_not_claim_a_download_total(self):
        states, _ = self.check(self.info(
            tags=[], library_name="transformers",
            siblings=[SimpleNamespace(rfilename="model.safetensors", size=None)],
        ))
        detail, _ = repository.repository_view("org/model", states[-1])
        self.assertIn("Download size unavailable", detail)

    def test_gguf_repository_exists_but_is_not_loadable(self):
        states, _ = self.check(self.info(
            tags=["gguf"], library_name=None,
            siblings=[SimpleNamespace(rfilename="model.gguf", size=100)],
        ))
        self.assertEqual(states[-1]["status"], "found")
        self.assertTrue(states[-1]["unsupported"])
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
            _, load, _, _ = models_page.refresh_model_actions("org/model", None, states[-1])
        self.assertFalse(load["interactive"])
