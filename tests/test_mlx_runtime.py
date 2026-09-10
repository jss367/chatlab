"""The MLX backend, run against a tiny real mlx-lm model where mlx is installed.

The pure helpers - what an MLX config looks like, what a repository name
says about its width - run everywhere. The engine tests build a two-layer
Llama with random weights through mlx-lm's own classes, so they exercise the
real forward pass, the real cache and the real attention kernel, and are
skipped where mlx cannot be imported (anything but Apple silicon).
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import mlx_runtime
from mlx_runtime import (
    MlxEngine,
    bits_from_name,
    mlx_quantization,
    mlx_supports,
    precision_label,
    read_mlx_config,
    read_stop_ids,
)

try:
    import mlx.core as mx
    from mlx_lm.models import gpt2, llama
except ImportError:  # pragma: no cover - the engine tests skip themselves
    mx = None
    gpt2 = None
    llama = None

needs_mlx = unittest.skipIf(mx is None, "mlx and mlx-lm are installed on Apple silicon only")


class ConfigTests(unittest.TestCase):
    def test_a_converted_config_names_its_width(self):
        config = {"model_type": "qwen2", "quantization": {"group_size": 64, "bits": 4}}

        self.assertEqual(mlx_quantization(config), {"group_size": 64, "bits": 4})
        self.assertEqual(precision_label(config), "4-bit")

    def test_the_newer_spelling_is_read_too(self):
        config = {"model_type": "llama", "quantization_config": {"group_size": 64, "bits": 8}}

        self.assertEqual(mlx_quantization(config)["bits"], 8)
        self.assertEqual(precision_label(config), "8-bit")

    def test_a_transformers_quantizer_is_not_mistaken_for_mlx(self):
        # bitsandbytes, GPTQ and AWQ all write quant_method; MLX never does.
        config = {
            "model_type": "llama",
            "quantization_config": {"quant_method": "bitsandbytes", "bits": 4},
        }

        self.assertIsNone(mlx_quantization(config))
        self.assertEqual(precision_label(config), "full")

    def test_a_plain_config_is_not_mlx(self):
        self.assertIsNone(mlx_quantization({"model_type": "llama"}))
        self.assertIsNone(mlx_quantization(None))
        self.assertIsNone(mlx_quantization({"quantization": {"bits": True}}))
        self.assertIsNone(mlx_quantization({"quantization": {"bits": 0}}))

    def test_a_snapshot_is_read_from_its_config_file(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            self.assertIsNone(read_mlx_config(snapshot))
            (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
            self.assertIsNone(read_mlx_config(snapshot))
            (snapshot / "config.json").write_text(
                json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
            )
            self.assertEqual(read_mlx_config(snapshot)["model_type"], "llama")
            (snapshot / "config.json").write_text("not json")
            self.assertIsNone(read_mlx_config(snapshot))
            (snapshot / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
            # No model_type: not a model config at all.
            self.assertIsNone(read_mlx_config(snapshot))


class StopTokenTests(unittest.TestCase):
    """Stop tokens need no mlx: they are read from the checkpoint's JSON files."""

    def write(self, snapshot: Path, name: str, value) -> None:
        (snapshot / name).write_text(value if isinstance(value, str) else json.dumps(value))

    def test_stop_tokens_come_from_the_config(self):
        self.assertEqual(MlxEngine(object(), {"eos_token_id": 2}).eos_token_ids(), {2})
        self.assertEqual(MlxEngine(object(), {"eos_token_id": [2, 7]}).eos_token_ids(), {2, 7})
        self.assertEqual(MlxEngine(object(), {}).eos_token_ids(), set())
        self.assertEqual(MlxEngine(object(), {"eos_token_id": True}).eos_token_ids(), set())
        self.assertEqual(MlxEngine(object(), {"eos_token_id": [2, True, "x"]}).eos_token_ids(), {2})

    def test_the_generation_config_adds_its_end_of_turn_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            self.write(snapshot, "config.json", {"model_type": "llama", "eos_token_id": 2})
            self.write(snapshot, "generation_config.json", {"eos_token_id": [2, 7]})
            self.assertEqual(read_stop_ids(snapshot), {2, 7})
            engine = MlxEngine.from_snapshot(object(), snapshot)
            self.assertEqual(engine.eos_token_ids(), {2, 7})
            self.assertEqual(engine.config["model_type"], "llama")

    def test_a_missing_generation_config_leaves_the_configs_own_token(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            self.write(snapshot, "config.json", {"model_type": "llama", "eos_token_id": 2})
            self.assertEqual(read_stop_ids(snapshot), {2})
            self.assertEqual(MlxEngine.from_snapshot(object(), snapshot).eos_token_ids(), {2})

    def test_a_generation_config_alone_is_enough(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            self.write(snapshot, "generation_config.json", {"eos_token_id": 7})
            self.assertEqual(read_stop_ids(snapshot), {7})

    def test_junk_in_either_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            self.write(snapshot, "config.json", {"model_type": "llama", "eos_token_id": 2})
            self.write(snapshot, "generation_config.json", "not json")
            self.assertEqual(read_stop_ids(snapshot), {2})
            self.write(snapshot, "generation_config.json", [1, 2])
            self.assertEqual(read_stop_ids(snapshot), {2})
            self.write(snapshot, "generation_config.json", {"eos_token_id": "<|eot_id|>"})
            self.assertEqual(read_stop_ids(snapshot), {2})
            self.write(snapshot, "generation_config.json", {"eos_token_id": [7, True, None, "x"]})
            self.assertEqual(read_stop_ids(snapshot), {2, 7})
            self.assertEqual(read_stop_ids(snapshot / "missing"), set())

    def test_the_engines_set_is_a_copy(self):
        engine = MlxEngine(object(), {"eos_token_id": 2})
        engine.eos_token_ids().add(9)
        self.assertEqual(engine.eos_token_ids(), {2})


class BitsFromNameTests(unittest.TestCase):
    def test_the_width_is_read_off_the_repository_name(self):
        self.assertEqual(bits_from_name("mlx-community/Qwen3-4B-4bit"), 4)
        self.assertEqual(bits_from_name("mlx-community/Llama-3.2-3B-Instruct-8bit"), 8)
        self.assertEqual(bits_from_name("mlx-community/gemma-3-4b-it-3bit"), 3)
        self.assertEqual(bits_from_name("mlx-community/Qwen3-8B-4bit-DWQ"), 4)
        self.assertEqual(bits_from_name("org/Model-4Bit"), 4)

    def test_a_name_that_says_nothing_or_says_full_is_none(self):
        self.assertIsNone(bits_from_name("mlx-community/Llama-3.2-1B-Instruct-bf16"))
        self.assertIsNone(bits_from_name("mlx-community/Qwen3-0.6B"))
        self.assertIsNone(bits_from_name("allenai/Olmo-3-7B-Think"))
        # A digit in the organisation is not a width.
        self.assertIsNone(bits_from_name("4bit-lab/plain-model"))
        self.assertIsNone(bits_from_name("org/model-16bit"))


class SupportsTests(unittest.TestCase):
    """Whether mlx-lm implements an architecture: the remapping table, then a module lookup."""

    def test_nothing_is_supported_where_mlx_is_not_installed(self):
        with mock.patch("mlx_runtime.mlx_available", return_value=False):
            self.assertFalse(mlx_supports("llama"))

    def test_the_lookup_goes_through_the_remapping_table_then_the_module_path(self):
        # The same reasoning mlx_lm.utils._get_classes follows, without mlx:
        # the table and the module path are both stood in for.
        utils = types.ModuleType("mlx_lm.utils")
        utils.MODEL_REMAPPING = {"mistral": "llama"}
        package = types.ModuleType("mlx_lm")
        package.utils = utils
        modules = {"mlx_lm.models.llama"}
        asked = []

        def find_spec(name, *args):
            asked.append(name)
            return object() if name in modules else None

        with (
            mock.patch("mlx_runtime.mlx_available", return_value=True),
            mock.patch.dict(sys.modules, {"mlx_lm": package, "mlx_lm.utils": utils}),
            mock.patch("importlib.util.find_spec", side_effect=find_spec),
        ):
            self.assertTrue(mlx_supports("llama"))
            self.assertTrue(mlx_supports("mistral"))
            self.assertFalse(mlx_supports("no-such-architecture"))
            self.assertFalse(mlx_supports(None))
            self.assertFalse(mlx_supports(""))

        # mistral was looked up as llama, the type it is remapped to.
        self.assertEqual(asked, ["mlx_lm.models.llama"] * 2 + ["mlx_lm.models.no-such-architecture"])

    @needs_mlx
    def test_the_installed_mlx_lm_answers_for_itself(self):
        self.assertTrue(mlx_supports("llama"))
        self.assertTrue(mlx_supports("gpt2"))
        # Remapped by mlx-lm: mistral runs as llama.
        self.assertTrue(mlx_supports("mistral"))
        self.assertFalse(mlx_supports("no-such-architecture"))


VOCAB = 32
HIDDEN = 16
LAYERS = 2
HEADS = 2


def tiny_llama(tie_word_embeddings: bool = True, seed: int = 0, vocab: int = VOCAB):
    """A two-layer Llama with random weights, small enough to run in a test."""

    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        intermediate_size=32,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    mx.random.seed(seed)
    model = llama.Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def reference_logits(model, ids: list[int]) -> np.ndarray:
    """Every position's logits from one uncached pass, as float32 numpy."""

    logits = model(mx.array([ids]))
    mx.eval(logits)
    return np.array(logits[0].astype(mx.float32))


@needs_mlx
class ForwardTests(unittest.TestCase):
    def test_a_chunked_cached_pass_matches_one_uncached_pass(self):
        model = tiny_llama()
        engine = MlxEngine(model, {"eos_token_id": 2})
        ids = [3, 5, 7, 11, 13, 17, 19]
        expected = reference_logits(model, ids)

        first, cache = engine.forward(ids[:4], None, 0)
        second, cache = engine.forward(ids[4:6], cache, 4)
        third, cache = engine.forward(ids[6:], cache, 6)

        np.testing.assert_allclose(first.row(3), expected[3], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(second.row(1), expected[5], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(third.row(-1), expected[6], rtol=1e-2, atol=1e-2)
        self.assertEqual(cache[0].offset, len(ids))

    def test_a_cache_that_disagrees_with_the_caller_is_refused(self):
        engine = MlxEngine(tiny_llama())
        _, cache = engine.forward([1, 2, 3], None, 0)

        with self.assertRaises(RuntimeError):
            engine.forward([4], cache, 2)


@needs_mlx
class CacheTests(unittest.TestCase):
    def test_a_cache_is_cut_back_in_place(self):
        model = tiny_llama()
        engine = MlxEngine(model)
        ids = [3, 5, 7, 11, 13]
        _, cache = engine.forward(ids, None, 0)

        self.assertTrue(engine.can_crop(cache, len(ids)))
        engine.crop(cache, 2)
        self.assertEqual(cache[0].offset, 3)

        # Continuing from the cropped cache gives what a fresh pass gives.
        logits, cache = engine.forward([11], cache, 3)
        expected = reference_logits(model, ids[:4])
        np.testing.assert_allclose(logits.row(-1), expected[3], rtol=1e-2, atol=1e-2)

    def test_an_unknown_cache_cannot_be_cropped(self):
        self.assertFalse(MlxEngine.can_crop(None, 0))
        self.assertFalse(MlxEngine.can_crop([object()], 3))
        self.assertFalse(MlxEngine.can_crop([], 0))


@needs_mlx
class LensTests(unittest.TestCase):
    def inspect(self, model, ids: list[int]):
        engine = MlxEngine(model)
        _, cache = engine.forward(ids[:-1], None, 0)
        return engine, engine.inspect_step(ids[-1], cache, len(ids) - 1)

    def test_the_final_row_is_the_model_output_and_the_stack_is_read_through(self):
        model = tiny_llama()
        ids = [3, 5, 7, 11]
        _, reading = self.inspect(model, ids)
        expected = reference_logits(model, ids)[-1]

        np.testing.assert_allclose(reading.final_logits, expected, rtol=1e-2, atol=1e-2)
        # Embeddings plus one state per layer, the way Transformers counts.
        self.assertEqual(reading.layer_count, LAYERS + 1)
        self.assertEqual(len(reading.layer_logits), LAYERS)
        for logits in reading.layer_logits:
            self.assertEqual(logits.shape, (VOCAB,))
            self.assertTrue(np.isfinite(logits).all())
        # The cache now covers the predicting token too.
        self.assertEqual(reading.cache[0].offset, len(ids))

    def test_reading_the_last_state_through_the_head_reproduces_the_output(self):
        # The check inspect_step makes before it trusts the intermediate rows,
        # made explicitly: norm, then head, equals what the model emitted.
        model = tiny_llama()
        engine = MlxEngine(model)
        ids = [3, 5, 7, 11]
        expected = reference_logits(model, ids)[-1]
        recorder = mlx_runtime._Recorder()
        with engine._recording(recorder):
            mx.eval(model(mx.array([ids])))
        replayed = engine.read_head(engine.final_norm()(recorder.hidden[-1][:, -1:, :]))[0, -1]

        np.testing.assert_allclose(
            np.array(replayed.astype(mx.float32)), expected, rtol=1e-2, atol=1e-2
        )
        # The layers were put back as they were.
        self.assertIsInstance(model.model.layers[0], llama.TransformerBlock)

    def test_an_untied_head_is_read_through_lm_head(self):
        model = tiny_llama(tie_word_embeddings=False)
        ids = [3, 5, 7]
        _, reading = self.inspect(model, ids)

        self.assertEqual(len(reading.layer_logits), LAYERS)
        np.testing.assert_allclose(
            reading.final_logits, reference_logits(model, ids)[-1], rtol=1e-2, atol=1e-2
        )

    def test_a_head_transform_the_config_does_not_declare_withholds_the_rows(self):
        model = tiny_llama()
        engine = MlxEngine(model)
        original = engine.read_head

        # Stand in for an architecture whose head does something this reader
        # has not been told about: the replayed logits will not match.
        engine.read_head = lambda vector: original(vector) * 3.0
        _, cache = engine.forward([3, 5, 7], None, 0)
        reading = engine.inspect_step(11, cache, 3)

        self.assertEqual(reading.layer_logits, [])
        self.assertEqual(reading.layer_count, LAYERS + 1)
        np.testing.assert_allclose(
            reading.final_logits, reference_logits(model, [3, 5, 7, 11])[-1], rtol=1e-2, atol=1e-2
        )

    def test_a_declared_softcap_is_applied_when_reading_the_head(self):
        engine = MlxEngine(tiny_llama(), {"final_logit_softcapping": 30.0})
        vector = mx.ones((1, 1, HIDDEN)) * 100
        capped = np.array(engine.read_head(vector).astype(mx.float32))

        self.assertLessEqual(float(np.abs(capped).max()), 30.0)

    def test_the_attention_strip_has_one_row_per_layer_over_every_key(self):
        model = tiny_llama()
        ids = [3, 5, 7, 11, 13]
        _, reading = self.inspect(model, ids)

        self.assertEqual(len(reading.attention), LAYERS)
        for row in reading.attention:
            self.assertEqual(len(row), len(ids))
            self.assertAlmostEqual(sum(row), 1.0, places=3)
            self.assertTrue(all(value >= 0 for value in row))

    def test_the_recorded_attention_matches_what_the_kernel_computed(self):
        # The recording recomputes softmax(q k / sqrt(d)) beside the fused
        # kernel; check it against the same arithmetic done by hand on the
        # layer's own projections.
        model = tiny_llama()
        ids = [3, 5, 7, 11]
        engine, reading = self.inspect(model, ids)

        block = model.model.layers[0]
        attention = block.self_attn
        hidden = model.model.embed_tokens(mx.array([ids]))
        normed = block.input_layernorm(hidden)
        queries = attention.q_proj(normed).reshape(1, len(ids), HEADS, -1).transpose(0, 2, 1, 3)
        keys = attention.k_proj(normed).reshape(1, len(ids), 1, -1).transpose(0, 2, 1, 3)
        queries = attention.rope(queries)
        keys = attention.rope(keys)
        keys = mx.repeat(keys, HEADS, axis=1)
        scores = (queries[:, :, -1:, :] * attention.scale) @ keys.transpose(0, 1, 3, 2)
        expected = mx.softmax(scores.astype(mx.float32), axis=-1)[0, :, -1, :].mean(axis=0)
        mx.eval(expected)

        np.testing.assert_allclose(
            reading.attention[0], np.array(expected), rtol=1e-3, atol=1e-4
        )

    def test_the_model_is_restored_after_a_failed_inspection(self):
        model = tiny_llama()
        engine = MlxEngine(model)
        layers_before = model.model.layers
        kernel_before = llama.scaled_dot_product_attention

        class Broken:
            offset = 0

        with self.assertRaises(Exception):
            engine.inspect_step(3, [Broken(), Broken()], 0)

        self.assertIs(model.model.layers, layers_before)
        self.assertIs(llama.scaled_dot_product_attention, kernel_before)


def tiny_gpt2(seed: int = 0):
    """A two-layer GPT-2 with random weights: the family that keeps its stack under ``h``."""

    args = gpt2.ModelArgs(
        model_type="gpt2",
        n_ctx=64,
        n_embd=HIDDEN,
        n_head=HEADS,
        n_layer=LAYERS,
        n_positions=64,
        layer_norm_epsilon=1e-5,
        vocab_size=VOCAB,
    )
    mx.random.seed(seed)
    model = gpt2.Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


@needs_mlx
class Gpt2LensTests(unittest.TestCase):
    """mlx-lm's GPT-2 keeps its blocks under ``model.h``; the outer ``layers`` is a property."""

    def test_the_stack_is_found_where_the_forward_pass_reads_it(self):
        model = tiny_gpt2()
        engine = MlxEngine(model)

        owner, attribute = engine._layer_stack()
        self.assertIs(owner, model.model)
        self.assertEqual(attribute, "h")
        self.assertIs(engine.final_norm(), model.model.ln_f)

    def test_every_layer_is_read_through_and_the_model_is_restored(self):
        model = tiny_gpt2()
        ids = [3, 5, 7, 11, 13]
        engine = MlxEngine(model)
        stack_before = model.model.h
        kernel_before = gpt2.scaled_dot_product_attention

        _, cache = engine.forward(ids[:-1], None, 0)
        reading = engine.inspect_step(ids[-1], cache, len(ids) - 1)

        np.testing.assert_allclose(
            reading.final_logits, reference_logits(model, ids)[-1], rtol=1e-2, atol=1e-2
        )
        self.assertEqual(reading.layer_count, LAYERS + 1)
        self.assertEqual(len(reading.layer_logits), LAYERS)
        for logits in reading.layer_logits:
            self.assertEqual(logits.shape, (VOCAB,))
            self.assertTrue(np.isfinite(logits).all())
        self.assertEqual(len(reading.attention), LAYERS)
        for row in reading.attention:
            self.assertEqual(len(row), len(ids))
            self.assertAlmostEqual(sum(row), 1.0, places=3)
        # The stack was put back under h, as the same list, and the kernel too.
        self.assertIs(model.model.h, stack_before)
        self.assertIsInstance(model.model.h[0], gpt2.TransformerBlock)
        self.assertIs(gpt2.scaled_dot_product_attention, kernel_before)


@needs_mlx
class ManagerTests(unittest.TestCase):
    """The manager driving an MLX engine end to end, tokenizer faked."""

    def manager(self):
        from model_runtime import ModelManager
        from test_streaming import FakeTokenizer

        model = tiny_llama()
        manager = ModelManager()
        manager.model = model
        manager.engine = MlxEngine(model, {"eos_token_id": 99})
        manager.tokenizer = FakeTokenizer(tuple(f"t{i}" for i in range(VOCAB)), 1)
        manager.model_id = "mlx-community/tiny-4bit"
        manager.kind = "mlx"
        return manager

    def test_a_response_streams_with_metrics_for_every_token(self):
        manager = self.manager()
        updates = list(
            manager.generate(
                [{"role": "user", "content": "t3 t5"}],
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=4,
                seed=0,
            )
        )

        self.assertTrue(updates)
        last = updates[-1]
        self.assertGreaterEqual(len(last.metrics), 1)
        self.assertLessEqual(len(last.metrics), 4)
        for metric in last.metrics:
            self.assertIn("raw_rank", metric)
            self.assertIn("entropy_bits", metric)
            self.assertEqual(metric["segment"], "response")
        self.assertTrue(last.prompt_metrics)
        # The config's stop token joined the tokenizer's.
        self.assertIn(99, manager._stop_token_ids())
        self.assertIn(1, manager._stop_token_ids())

    def test_a_token_can_be_inspected_and_the_cache_reused(self):
        manager = self.manager()
        ids = [3, 5, 7, 11, 13]

        first = manager.inspect(ids, 3, context_count=2)
        self.assertEqual(len(first.layers), LAYERS + 1)
        self.assertEqual(first.layers[-1]["layer"], LAYERS)
        self.assertEqual(len(first.attention), LAYERS)
        self.assertEqual(len(first.attention[0]), 3)
        self.assertEqual([token["segment"] for token in first.tokens], ["prompt", "prompt", "response"])
        kept = manager._inspect_cache
        self.assertEqual(kept[1], ids[:3])

        # One further along reuses and extends the kept cache; one earlier
        # cuts it back. Both must describe the same distribution a fresh
        # pass would.
        later = manager.inspect(ids, 4)
        self.assertEqual(manager._inspect_cache[1], ids[:4])
        earlier = manager.inspect(ids, 2)
        self.assertEqual(manager._inspect_cache[1], ids[:2])

        self.assertAlmostEqual(
            later.layers[-1]["probability"],
            self.manager().inspect(ids, 4).layers[-1]["probability"],
            places=3,
        )
        self.assertAlmostEqual(
            earlier.layers[-1]["probability"],
            self.manager().inspect(ids, 2).layers[-1]["probability"],
            places=3,
        )

    def test_scoring_text_measures_every_token_after_the_first(self):
        # A real (tiny) tokenizer, so the context and the text are split
        # the way Score text splits them; the model's vocabulary is sized
        # to match.
        import tiny_tokenizer

        tokenizer = tiny_tokenizer.build()
        manager = self.manager()
        model = tiny_llama(vocab=len(tokenizer))
        manager.model = model
        manager.engine = MlxEngine(model, {})
        manager.tokenizer = tokenizer

        scored = manager.score_text("the cat sat", context="on the mat")

        self.assertEqual(len(scored.context_metrics), 4)
        self.assertEqual(len(scored.metrics), 3)
        self.assertFalse(scored.context_metrics[0]["scored"])
        self.assertTrue(all(metric["scored"] for metric in scored.context_metrics[1:]))
        self.assertTrue(all(metric["scored"] for metric in scored.metrics))
        self.assertTrue(all("raw_rank" in metric for metric in scored.metrics))
        self.assertEqual({metric["segment"] for metric in scored.metrics}, {"response"})
        self.assertTrue(scored.seam_verified)


if __name__ == "__main__":
    unittest.main()
