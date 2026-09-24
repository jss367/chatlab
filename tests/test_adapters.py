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
from model_runtime import (
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
                    "model_runtime.is_mlx_snapshot",
                    side_effect=lambda path: BASE.replace("/", "--") in str(path),
                ),
                mock.patch("model_runtime.mlx_available", return_value=True),
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

    def build(self, root: Path, extra_token: str | None = None) -> tuple[Path, object]:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        tokenizer = tiny_tokenizer.build()
        torch.manual_seed(0)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64,
        ))
        base_path = snapshot(root, BASE, {})
        base.save_pretrained(base_path)
        tokenizer.save_pretrained(base_path)
        # Nonzero B matrices, so the merge visibly changes the weights.
        lora = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", init_lora_weights=False,
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
            model, tokenizer, pipeline, device = model_runtime._read_text_model(
                path, torch, "cpu", torch.float32, None, "full"
            )
            with torch.no_grad():
                logits = model(ids).logits

        self.assertEqual(type(model).__name__, "LlamaForCausalLM")
        self.assertFalse(any("lora" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(logits, expected)
        self.assertIsNone(pipeline)
        self.assertEqual(device, "CPU")

    def test_an_adapter_that_ships_a_grown_tokenizer_gets_embeddings_to_match(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            path, _ = self.build(Path(root), extra_token="<persona>")
            model, tokenizer, _pipeline, _device = model_runtime._read_text_model(
                path, torch, "cpu", torch.float32, None, "full"
            )

        self.assertIn("<persona>", tokenizer.get_vocab())
        self.assertEqual(model.get_input_embeddings().weight.shape[0], len(tokenizer))

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
                mock.patch("model_runtime.detect_backend", return_value="mps"),
                mock.patch.object(manager, "_unload_locked"),
                mock.patch.object(manager, "_cap_mps_memory", return_value=None),
                mock.patch.object(manager, "_check_memory", return_value=(None, None)) as check,
                mock.patch("model_runtime.allocated_bytes", return_value=None),
                mock.patch("model_runtime.reserved_bytes", return_value=None),
                mock.patch("model_runtime._read_text_model", side_effect=read),
                self.assertLogs("model_runtime", level=logging.INFO) as logs,
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
                model_runtime._read_text_model(path, torch, "cpu", torch.float32, None, "full")

        self.assertIn(BASE, str(caught.exception))
        self.assertIn("Download and load", str(caught.exception))


class AdapterDownloadTests(unittest.TestCase):
    def test_downloading_an_adapter_fetches_its_base_next(self):
        from ui import models_page

        fetched = []

        def fake_download(model_id, hf_token):
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
                mock.patch.object(models_page, "cache_status", return_value=model_runtime.CacheStatus()),
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

    def test_a_model_that_is_not_an_adapter_fetches_nothing_more(self):
        from ui import models_page

        with tempfile.TemporaryDirectory() as root:
            path = snapshot(Path(root), BASE, BASE_FILES)

            def fake_download(model_id, hf_token):
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

        status = model_runtime.CacheStatus(
            cached_bytes=10, missing_files=(BASE_MODEL,), base_model=BASE
        )
        self.assertEqual(
            models_page.describe_missing(status), f"the base model `{BASE}` is missing"
        )


if __name__ == "__main__":
    unittest.main()
