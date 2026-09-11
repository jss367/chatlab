"""Repository checks distinguish existence, access, and local availability."""

import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from model_runtime import CacheStatus
from ui import model_repository as repository, models_page


class RepositoryTests(unittest.TestCase):
    def check(self, info=None, error=None, model_id="org/model", token=""):
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=info, side_effect=error) as call:
            states = list(repository.check_model_repository(model_id, token))
        return states, call

    def info(self, **kwargs):
        return SimpleNamespace(**{
            "tags": ["mlx"], "config": {"model_type": "qwen3_5"},
            "library_name": "mlx", "private": False, "gated": False,
            "siblings": [SimpleNamespace(rfilename="model.safetensors", size=17_000_000_000)],
            **kwargs,
        })

    def test_mlx_lookup_checks_metadata_and_shows_fixed_precision(self):
        with mock.patch("mlx_runtime.mlx_supports", return_value=True):
            states, call = self.check(self.info(), model_id="org/model-4bit", token=" secret ")
        call.assert_called_once_with("org/model-4bit", token="secret", files_metadata=True, timeout=10)
        self.assertEqual(states[0]["status"], "checking")
        result = states[-1]
        self.assertEqual(result["download_bytes"], 17_000_000_000)
        detail, precision = repository.repository_view("org/model-4bit", result)
        self.assertIn("Repository found", detail)
        self.assertIn("4-bit", detail)
        self.assertFalse(precision["visible"])
        self.assertNotIn("secret", str(states))

    def test_a_late_response_cannot_verify_or_disable_an_edited_id(self):
        old = {"model_id": "org/old", "status": "missing", "detail": "Repository not found"}
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
                    "org/model", None, {"model_id": "org/model", "status": status}
                )
                self.assertTrue(cached["visible"])

    def test_missing_repository_disables_download_until_rechecked_or_edited(self):
        with mock.patch.object(models_page, "cache_status", return_value=CacheStatus()):
            _, load, download, _ = models_page.refresh_model_actions(
                "org/model", None, {"model_id": "org/model", "status": "missing"}
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
