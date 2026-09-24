"""LoRA adapters: recognized in the cache, measured by their base, merged on load."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import settings_sandbox
import tiny_tokenizer

import adapters
import model_runtime
import model_cache
import model_loading
from model_cache import (
    BASE_MODEL,
    MLX_KIND,
    TEXT_KIND,
    cache_status,
    estimate_snapshot_bytes,
)

ADAPTER = "someone/persona-lora"
BASE = "org/base-model"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def lora_config(**overrides) -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE,
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        **overrides,
    }


def snapshot(root: Path, model_id: str, files: dict[str, bytes]) -> Path:
    """A cache folder for ``model_id`` laid out as ``huggingface_hub`` leaves it."""

    folder = root / f"models--{model_id.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(COMMIT)
    path = folder / "snapshots" / COMMIT
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).write_bytes(content)
    return path


def other_snapshot(
    root: Path, model_id: str, commit: str, files: dict[str, bytes], ref: str | None = None
) -> Path:
    """A second snapshot of ``model_id``, at ``commit``, beside the one ``main`` names.

    With ``ref`` it is a branch or tag the cache has a ref file for; without,
    it is a commit fetched by hash, which the hub files under its own name.
    """

    folder = root / f"models--{model_id.replace('/', '--')}"
    if ref is not None:
        (folder / "refs" / ref).parent.mkdir(parents=True, exist_ok=True)
        (folder / "refs" / ref).write_text(commit)
    path = folder / "snapshots" / commit
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).write_bytes(content)
    return path


PINNED = "fedcba9876543210fedcba9876543210fedcba98"


def adapter_files(**overrides) -> dict[str, bytes]:
    return {
        "adapter_config.json": json.dumps(lora_config(**overrides)).encode(),
        "adapter_model.safetensors": b"x" * 10,
    }


BASE_FILES = {
    "config.json": b'{"model_type": "llama", "torch_dtype": "float16"}',
    "model.safetensors": b"x" * 1000,
}


class AdapterConfigTests(unittest.TestCase):
    def test_an_unsloth_4bit_base_is_swapped_for_the_weights_it_was_made_from(self):
        for named, loaded in (
            ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "unsloth/Qwen2.5-7B-Instruct"),
            ("unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", "unsloth/Llama-3.2-3B-Instruct"),
            ("unsloth/Qwen2.5-0.5B-Instruct", "unsloth/Qwen2.5-0.5B-Instruct"),
            ("other/model-bnb-4bit", "other/model-bnb-4bit"),
        ):
            with self.subTest(named=named):
                config = lora_config(base_model_name_or_path=named)
                self.assertEqual(adapters.base_model_id(config), loaded)

    def test_a_pinned_base_revision_is_kept_except_across_the_unsloth_swap(self):
        self.assertEqual(adapters.base_revision(lora_config(revision=PINNED)), PINNED)
        self.assertEqual(adapters.base_revision(lora_config(revision=" v1.0 ")), "v1.0")
        for unpinned in (None, "", "  ", 3):
            with self.subTest(revision=unpinned):
                self.assertIsNone(adapters.base_revision(lora_config(revision=unpinned)))
        self.assertIsNone(adapters.base_revision(lora_config()))
        # The pin names a commit of the 4-bit repo, which the full-precision
        # one it is swapped for does not have.
        swapped = lora_config(
            base_model_name_or_path="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", revision=PINNED
        )
        self.assertIsNone(adapters.base_revision(swapped))

    def test_the_adapters_chatlab_cannot_merge_are_explained(self):
        cases = {
            "PREFIX_TUNING": lora_config(peft_type="PREFIX_TUNING"),
            "SEQ_CLS": lora_config(task_type="SEQ_CLS"),
            "base_model_name_or_path": lora_config(base_model_name_or_path=None),
            "/content/model": lora_config(base_model_name_or_path="/content/model"),
        }
        for expected, config in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, adapters.adapter_problem(config))
        self.assertIsNone(adapters.adapter_problem(lora_config()))
        self.assertIn("itself", adapters.adapter_problem(lora_config(), BASE))


class AdapterCacheStatusTests(unittest.TestCase):
    def test_an_adapter_without_its_base_names_the_base_as_missing(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, adapter_files())
            status = cache_status(ADAPTER, Path(root))

        self.assertEqual(status.kind, TEXT_KIND)
        self.assertEqual(status.base_model, BASE)
        self.assertEqual(status.missing_files, (BASE_MODEL,))
        self.assertFalse(status.complete)

    def test_an_adapter_with_its_base_on_disk_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, adapter_files())
            snapshot(Path(root), BASE, BASE_FILES)
            status = cache_status(ADAPTER, Path(root))

        self.assertTrue(status.complete)
        self.assertEqual(status.kind, TEXT_KIND)
        self.assertEqual(status.base_model, BASE)

    def test_an_adapter_short_of_its_weights_says_so(self):
        with tempfile.TemporaryDirectory() as root:
            files = adapter_files()
            del files["adapter_model.safetensors"]
            snapshot(Path(root), ADAPTER, files)
            snapshot(Path(root), BASE, BASE_FILES)
            status = cache_status(ADAPTER, Path(root))

        self.assertEqual(status.missing_files, ("adapter_model.safetensors",))

    def test_an_adapter_cut_off_before_its_config_is_incomplete(self):
        # The weights landed first; alone they would read as another
        # framework's and hide the download that finishes them.
        with tempfile.TemporaryDirectory() as root:
            files = adapter_files()
            del files["adapter_config.json"]
            snapshot(Path(root), ADAPTER, files)
            status = cache_status(ADAPTER, Path(root))

        self.assertEqual(status.kind, TEXT_KIND)
        self.assertEqual(status.missing_files, ("adapter_config.json",))
        self.assertFalse(status.unsupported)
        self.assertTrue(status.present)

    def test_a_pinned_base_is_judged_at_its_revision_not_at_main(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = snapshot(Path(root), ADAPTER, adapter_files(revision=PINNED))
            # main has moved on: the base is whole there, but not at the pin.
            snapshot(Path(root), BASE, BASE_FILES)
            before = cache_status(ADAPTER, Path(root))
            self.assertIsNone(model_cache.adapter_base_snapshot(adapter))
            pinned = other_snapshot(Path(root), BASE, PINNED, BASE_FILES)
            after = cache_status(ADAPTER, Path(root))
            found = model_cache.adapter_base_snapshot(adapter)

        self.assertEqual(before.missing_files, (BASE_MODEL,))
        self.assertTrue(after.complete)
        self.assertEqual(found, pinned)

    def test_a_base_pinned_to_a_branch_follows_that_branchs_ref(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = snapshot(Path(root), ADAPTER, adapter_files(revision="v1"))
            snapshot(Path(root), BASE, BASE_FILES)
            pinned = other_snapshot(Path(root), BASE, PINNED, BASE_FILES, ref="v1")
            found = model_cache.adapter_base_snapshot(adapter)

        self.assertEqual(found, pinned)

    def test_an_adapter_chatlab_cannot_merge_is_unsupported_with_its_reason(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, adapter_files(peft_type="PROMPT_TUNING"))
            status = cache_status(ADAPTER, Path(root))

        self.assertTrue(status.unsupported)
        self.assertIn("PROMPT_TUNING", status.unsupported_reason)

    def test_an_adapter_on_an_mlx_base_is_unsupported(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, adapter_files())
            snapshot(Path(root), BASE, BASE_FILES)
            with (
                mock.patch(
                    "model_cache.is_mlx_snapshot",
                    side_effect=lambda path: BASE.replace("/", "--") in str(path),
                ),
                mock.patch("model_cache.mlx_available", return_value=True),
            ):
                self.assertEqual(cache_status(BASE, Path(root)).kind, MLX_KIND)
                status = cache_status(ADAPTER, Path(root))

        self.assertTrue(status.unsupported)
        self.assertIn("MLX", status.unsupported_reason)

    def test_adapters_that_name_each_other_are_refused_without_recursing(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, adapter_files())
            snapshot(Path(root), BASE, adapter_files(base_model_name_or_path=ADAPTER))
            status = cache_status(ADAPTER, Path(root))

        self.assertTrue(status.unsupported)
        self.assertIn("itself an adapter", status.unsupported_reason)

    def test_a_repo_with_a_whole_checkpoint_beside_its_adapter_config_is_the_model(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot(Path(root), ADAPTER, {**adapter_files(), **BASE_FILES})
            status = cache_status(ADAPTER, Path(root))

        self.assertTrue(status.complete)
        self.assertIsNone(status.base_model)

    def test_an_adapter_is_measured_as_its_base_at_full_precision(self):
        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), ADAPTER, adapter_files())
            self.assertIsNone(estimate_snapshot_bytes(path, "float16", 4, TEXT_KIND))
            snapshot(Path(root), BASE, BASE_FILES)
            estimated = estimate_snapshot_bytes(path, "float16", 4, TEXT_KIND)

        self.assertEqual(estimated, 1000)


class AdapterLoadTests(unittest.TestCase):
    """A real tiny Llama and a real LoRA, merged by the loader ChatLab runs."""

    def build(
        self, root: Path, extra_token: str | None = None,
        padding: int = 0, trained_rows: int | None = None,
    ) -> tuple[Path, object]:
        """``padding`` pads the base's embeddings past the tokenizer, and
        ``trained_rows`` resizes them before training and saves them whole
        through ``modules_to_save``, the way a run that added tokens does."""

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        tokenizer = tiny_tokenizer.build()
        torch.manual_seed(0)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer) + padding, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64,
        ))
        base_path = snapshot(root, BASE, {})
        base.save_pretrained(base_path)
        tokenizer.save_pretrained(base_path)
        saved = {}
        if trained_rows is not None:
            base.resize_token_embeddings(trained_rows)
            saved = {"modules_to_save": ["embed_tokens", "lm_head"]}
        # Nonzero B matrices, so the merge visibly changes the weights.
        lora = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", init_lora_weights=False, **saved,
        ))
        ids = torch.tensor([tokenizer.encode("the cat sat on the mat")])
        with torch.no_grad():
            expected = lora(ids).logits
        adapter_path = snapshot(root, ADAPTER, {})
        lora.save_pretrained(adapter_path)
        config = json.loads((adapter_path / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = BASE
        (adapter_path / "adapter_config.json").write_text(json.dumps(config))
        if extra_token is not None:
            from transformers import AutoTokenizer

            grown = AutoTokenizer.from_pretrained(base_path)
            grown.add_tokens([extra_token])
            grown.save_pretrained(adapter_path)
        return adapter_path, (ids, expected)

    def test_the_adapter_is_merged_into_a_plain_model(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            path, (ids, expected) = self.build(Path(root))
            model, tokenizer, pipeline, device = model_loading._read_text_model(
                path, torch, "cpu", torch.float32, None, "full"
            )
            with torch.no_grad():
                logits = model(ids).logits

        self.assertEqual(type(model).__name__, "LlamaForCausalLM")
        self.assertFalse(any("lora" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(logits, expected)
        self.assertIsNone(pipeline)
        self.assertEqual(device, "CPU")

    def test_a_new_adapter_commit_over_the_same_base_is_a_new_revision(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            first, _ = self.build(Path(root))
            second = other_snapshot(
                Path(root), ADAPTER, PINNED,
                {path.name: path.read_bytes() for path in first.iterdir()},
            )
            managers = []
            for path in (first, second):
                manager = model_runtime.ModelManager()
                manager.model, manager.tokenizer, _pipeline, _device = (
                    model_loading._read_text_model(path, torch, "cpu", torch.float32, None, "full")
                )
                manager.model_id, manager.precision = ADAPTER, "full"
                managers.append(manager)
            old, new = managers
            self.assertEqual(old.model_revision(), f"{COMMIT}+{COMMIT}")
            self.assertEqual(new.model_revision(), f"{COMMIT}+{PINNED}")

            # A lens fitted to the first commit's weights is refused over the second.
            lens = Path(root) / "lens.pt"
            torch.save({
                "J": {0: torch.eye(16)}, "source_layers": [0], "n_prompts": 1,
                "d_model": 16, "model_revision": old.model_revision(),
            }, lens)
            self.assertEqual(old.import_jacobian_lens(str(lens), ADAPTER)["model_revision"], f"{COMMIT}+{COMMIT}")
            with self.assertRaisesRegex(ValueError, "revision"):
                new.import_jacobian_lens(str(lens), ADAPTER)

    def test_an_adapter_that_ships_a_grown_tokenizer_gets_embeddings_to_match(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            path, _ = self.build(Path(root), extra_token="<persona>")
            model, tokenizer, _pipeline, _device = model_loading._read_text_model(
                path, torch, "cpu", torch.float32, None, "full"
            )

        self.assertIn("<persona>", tokenizer.get_vocab())
        self.assertEqual(model.get_input_embeddings().weight.shape[0], len(tokenizer))

    def test_saved_embeddings_set_the_base_size_in_either_direction(self):
        import torch

        tokenizer = tiny_tokenizer.build()
        # Down: a base padded past its tokenizer, trained at the tokenizer's
        # length. Up: a tokenizer grown past the base's rows.
        for padding, rows in ((16, len(tokenizer)), (0, len(tokenizer) + 8)):
            with self.subTest(padding=padding, rows=rows), tempfile.TemporaryDirectory() as root:
                path, (ids, expected) = self.build(Path(root), padding=padding, trained_rows=rows)
                model, _tokenizer, _pipeline, _device = model_loading._read_text_model(
                    path, torch, "cpu", torch.float32, None, "full"
                )
                with torch.no_grad():
                    logits = model(ids).logits

                self.assertEqual(model.get_input_embeddings().weight.shape[0], rows)
                torch.testing.assert_close(logits, expected)

    def test_a_quantized_choice_loads_the_adapter_at_full_precision(self):
        import torch

        manager = model_runtime.ModelManager()
        seen = {}

        def read(path, torch_, backend, dtype, bits, precision):
            seen.update(bits=bits, precision=precision)
            return mock.Mock(), mock.Mock(), None, "Apple Metal (MPS)"

        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), ADAPTER, adapter_files())
            snapshot(Path(root), BASE, BASE_FILES)
            with (
                mock.patch("device_memory.detect_backend", return_value="mps"),
                mock.patch.object(manager, "_unload_locked"),
                mock.patch.object(manager, "_cap_mps_memory", return_value=None),
                mock.patch.object(manager, "_check_memory", return_value=(None, None)) as check,
                mock.patch("device_memory.allocated_bytes", return_value=None),
                mock.patch("device_memory.reserved_bytes", return_value=None),
                mock.patch("model_loading._read_text_model", side_effect=read),
                self.assertLogs("model_loading", level=logging.INFO) as logs,
            ):
                manager._load_locked(ADAPTER, path, torch, precision="4-bit")

        self.assertEqual(seen, {"bits": None, "precision": "full"})
        self.assertIsNone(check.call_args.kwargs["bits"])
        self.assertEqual(manager.precision, "full")
        self.assertTrue(any("merges into full-precision" in line for line in logs.output))
        self.assertTrue(any(f"into {BASE}" in line for line in logs.output))

    def test_a_load_without_the_base_on_disk_says_which_base(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), ADAPTER, adapter_files())
            with self.assertRaises(RuntimeError) as caught:
                model_loading._read_text_model(path, torch, "cpu", torch.float32, None, "full")

        self.assertIn(BASE, str(caught.exception))
        self.assertIn("Download and load", str(caught.exception))


class AdapterDownloadTests(unittest.TestCase):
    def test_downloading_an_adapter_fetches_its_base_next(self):
        from ui import models_page

        fetched = []

        def fake_download(model_id, hf_token, revision=None):
            fetched.append((model_id, hf_token))
            yield f"card for {model_id}"
            return self.paths[model_id]

        with tempfile.TemporaryDirectory() as root:
            self.paths = {
                ADAPTER: snapshot(Path(root), ADAPTER, adapter_files()),
                BASE: snapshot(Path(root), BASE, BASE_FILES),
            }
            with (
                mock.patch.object(models_page, "stream_download", side_effect=fake_download),
                mock.patch.object(models_page, "cache_status", return_value=model_cache.CacheStatus()),
            ):
                stream = models_page.stream_download_with_base(ADAPTER, "token")
                cards = []
                try:
                    while True:
                        cards.append(next(stream))
                except StopIteration as done:
                    path, note = done.value

        self.assertEqual(fetched, [(ADAPTER, "token"), (BASE, "token")])
        self.assertEqual(path, self.paths[ADAPTER])
        self.assertIn(BASE, note)
        self.assertTrue(any("trained on" in card for card in cards))

    def test_a_pinned_base_is_fetched_and_measured_at_its_revision(self):
        from ui import models_page

        fetched = []

        def fake_download(model_id, hf_token, revision=None):
            fetched.append((model_id, revision))
            yield "card"
            return path

        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), ADAPTER, adapter_files(revision=PINNED))
            with (
                mock.patch.object(models_page, "stream_download", side_effect=fake_download),
                mock.patch.object(
                    models_page, "cache_status", return_value=model_cache.CacheStatus()
                ) as status,
            ):
                list(models_page.stream_download_with_base(ADAPTER, ""))

        self.assertEqual(fetched, [(ADAPTER, None), (BASE, PINNED)])
        for call in status.call_args_list:
            self.assertEqual(call, mock.call(BASE, revision=PINNED))

    def test_the_manager_hands_the_revision_to_the_hub(self):
        manager = model_runtime.ModelManager()
        with mock.patch("huggingface_hub.snapshot_download", return_value="/cache/x") as fetch:
            manager.download(BASE, revision=PINNED)

        self.assertEqual(fetch.call_args.kwargs["repo_id"], BASE)
        self.assertEqual(fetch.call_args.kwargs["revision"], PINNED)

    def test_a_model_that_is_not_an_adapter_fetches_nothing_more(self):
        from ui import models_page

        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), BASE, BASE_FILES)

            def fake_download(model_id, hf_token, revision=None):
                yield "card"
                return path

            with mock.patch.object(models_page, "stream_download", side_effect=fake_download) as download:
                stream = models_page.stream_download_with_base(BASE, "")
                try:
                    while True:
                        next(stream)
                except StopIteration as done:
                    result = done.value

        self.assertEqual(result, (path, ""))
        download.assert_called_once()

    def test_a_missing_base_is_named_among_the_missing_files(self):
        from ui import models_page

        status = model_cache.CacheStatus(
            cached_bytes=10, missing_files=(BASE_MODEL,), base_model=BASE
        )
        self.assertEqual(
            models_page.describe_missing(status), f"the base model `{BASE}` is missing"
        )


if __name__ == "__main__":
    unittest.main()
