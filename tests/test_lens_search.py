"""Finding published Jacobian lenses on the Hub for the loaded model."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from huggingface_hub.errors import RepositoryNotFoundError

from chatlab import jacobian_lens


def _file(name, size):
    return SimpleNamespace(rfilename=name, size=size)


class FakeHub:
    """Answers ``list_models`` and ``model_info`` from fixed repositories."""

    def __init__(self, repositories, tags=None):
        self.repositories = repositories
        self.tags = tags or {}
        self.searches = []

    def list_models(self, filter, limit, search=None):
        self.searches.append((filter, search))
        base_filter, *tags = [filter] if isinstance(filter, str) else filter
        base = base_filter.removeprefix("base_model:")
        return [
            SimpleNamespace(id=name) for name, repo in self.repositories.items()
            if repo["base"] == base
            and (search is None or search in name.lower())
            and all(tag in repo.get("tags", []) for tag in tags)
        ]

    def model_info(self, repository, files_metadata=False):
        if repository in self.tags:
            return SimpleNamespace(tags=self.tags[repository])
        if repository not in self.repositories:
            raise RepositoryNotFoundError(repository, response=mock.Mock(status_code=404, headers={}))
        repo = self.repositories[repository]
        return SimpleNamespace(
            siblings=repo["files"], downloads=repo.get("downloads", 0),
            last_modified=datetime(2026, 8, 16, tzinfo=timezone.utc), tags=repo.get("tags", []),
        )


HUB = {
    "eyes-ml/Qwen3.8-27B_jacobian-lens": {
        "base": "Qwen/Qwen3.8-27B", "downloads": 12,
        "files": [_file("README.md", 5921), _file("Qwen3.8-27B_jacobian_lens.pt", 3_303_033_664)],
    },
    "gghfez/Qwen3.8-27B-jacobian-lens": {
        "base": "Qwen/Qwen3.8-27B", "downloads": 3,
        "files": [_file("Qwen3.8-27B_jlens.pt", 3_303_033_048)],
    },
    # Says "lens" but publishes no .pt file.
    "someone/jacobian-lens-notes": {
        "base": "Qwen/Qwen3.8-27B", "files": [_file("model.safetensors", 10)],
    },
    "someone/huge-lens": {
        "base": "Qwen/Qwen3.8-27B", "files": [_file("lens.pt", jacobian_lens.MAX_FILE_BYTES + 1)],
    },
    # A repository of lenses for several models tags each as its base.
    "lab/rs-lenses": {
        "base": "Qwen/Qwen3.8-27B",
        "files": [_file("qwen3.8-27b/r-lens/lens.pt", 100), _file("gemma-4-12b/r-lens/lens.pt", 100)],
    },
    # Named for neither "lens" nor "jacobian"; only its tags say what it is.
    "lab/qwen-readouts": {
        "base": "Qwen/Qwen3.8-27B", "tags": ["jacobian-lens"], "files": [_file("readout.pt", 50)],
    },
    # Says "lens" and holds a .pt file, but is not a lens of a kind ChatLab reads.
    "someone/cot-oracle-lens": {
        "base": "Qwen/Qwen3.8-27B", "files": [_file("training_state.pt", 10)],
    },
    "other/Qwen3-8B-jacobian-lens": {
        "base": "Qwen/Qwen3-8B", "files": [_file("lens.pt", 100)],
    },
}


class FindLensesTests(unittest.TestCase):
    def test_lists_pt_files_for_the_base_model_most_downloaded_first(self):
        hub = FakeHub(HUB)
        found = jacobian_lens.find_lenses("Qwen/Qwen3.8-27B", api=hub)
        self.assertEqual(
            [(lens.repository, lens.filename) for lens in found],
            [
                ("eyes-ml/Qwen3.8-27B_jacobian-lens", "Qwen3.8-27B_jacobian_lens.pt"),
                ("gghfez/Qwen3.8-27B-jacobian-lens", "Qwen3.8-27B_jlens.pt"),
                ("lab/qwen-readouts", "readout.pt"),
                ("lab/rs-lenses", "qwen3.8-27b/r-lens/lens.pt"),
            ],
        )
        self.assertEqual(found[0].fitted_model_id, "Qwen/Qwen3.8-27B")
        self.assertEqual(found[0].updated, "2026-08-16")
        self.assertEqual(
            {search for _, search in hub.searches} - {None}, set(jacobian_lens.LENS_SEARCH_TERMS),
        )

    def test_a_quantized_checkpoint_names_its_source(self):
        hub = FakeHub(HUB, tags={"Qwen/Qwen3.8-27B-FP8": [
            "base_model:Qwen/Qwen3.8-27B", "base_model:quantized:Qwen/Qwen3.8-27B",
        ]})
        self.assertEqual(jacobian_lens.quantized_from("Qwen/Qwen3.8-27B-FP8", api=hub), "Qwen/Qwen3.8-27B")

    def test_a_finetune_or_unknown_model_has_no_source(self):
        hub = FakeHub(HUB, tags={"me/tuned": ["base_model:finetune:Qwen/Qwen3-8B"]})
        self.assertIsNone(jacobian_lens.quantized_from("me/tuned", api=hub))
        self.assertIsNone(jacobian_lens.quantized_from("missing/model", api=hub))
        self.assertIsNone(jacobian_lens.quantized_from("/Users/me/models/local", api=hub))

    def test_a_hub_failure_is_reported(self):
        hub = mock.Mock()
        hub.list_models.side_effect = OSError("offline")
        with self.assertRaisesRegex(ValueError, "Could not search Hugging Face"):
            jacobian_lens.find_lenses("Qwen/Qwen3.8-27B", api=hub)


class FindLensesPageTests(unittest.TestCase):
    def _find(self, model_id, hub):
        from chatlab.ui import inspection, runtime

        manager = SimpleNamespace(loaded=True, model_id=model_id)
        with mock.patch.object(runtime, "MANAGER", manager), mock.patch(
            "huggingface_hub.HfApi", return_value=hub,
        ):
            return inspection.find_lenses()

    def test_a_quantized_load_lists_its_sources_lenses_and_says_what_to_load(self):
        hub = FakeHub(HUB, tags={"Qwen/Qwen3.8-27B-FP8": ["base_model:quantized:Qwen/Qwen3.8-27B"]})
        choices, note = self._find("Qwen/Qwen3.8-27B-FP8", hub)
        self.assertTrue(choices["visible"])
        self.assertEqual(len(choices["choices"]), 4)
        self.assertIn("quantized version of", note)
        self.assertIn("load `Qwen/Qwen3.8-27B` in full precision", note)

        from chatlab.ui import inspection

        self.assertEqual(
            inspection.pick_found_lens(choices["choices"][0][1]),
            ("eyes-ml/Qwen3.8-27B_jacobian-lens", "Qwen3.8-27B_jacobian_lens.pt", "Qwen/Qwen3.8-27B"),
        )

    def test_no_lens_found_says_so(self):
        choices, note = self._find("nobody/model", FakeHub(HUB))
        self.assertFalse(choices["visible"])
        self.assertIn("No lens on Hugging Face", note)

    def test_nothing_loaded(self):
        from chatlab.ui import inspection, runtime

        with mock.patch.object(runtime, "MANAGER", SimpleNamespace(loaded=False, model_id=None)):
            choices, note = inspection.find_lenses()
        self.assertIn("Load a model first", note)


if __name__ == "__main__":
    unittest.main()
