"""The key-value cache view, read from small real Transformers and MLX decoders."""

import threading
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

import charts
import kv_cache
import model_runtime
from kv_cache import CacheLayer
from model_runtime import ModelChanged, ModelManager
from test_mlx_runtime import needs_mlx, tiny_llama
from tiny_tokenizer import build
from ui import inspection, runtime

try:
    import mlx.core as mx
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache
except ImportError:  # pragma: no cover - the MLX tests skip themselves
    mx = None


def tiny_manager(config_type=LlamaConfig, model_type=LlamaForCausalLM, **extra):
    torch.manual_seed(7)
    manager = ModelManager()
    manager.tokenizer = build()
    config = config_type(
        vocab_size=len(manager.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=0, pad_token_id=0, **extra,
    )
    manager.model = model_type(config).eval()
    manager.model_id = "test/tiny-decoder"
    manager.precision = "full"
    return manager


def norms(tensor) -> np.ndarray:
    return np.linalg.norm(np.asarray(tensor, dtype=np.float32), axis=-1)


class ReadLayerTests(unittest.TestCase):
    def test_norms_are_per_head_and_position(self):
        keys = np.array([[[3.0, 4.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 2.0]]])
        values = np.array([[[6.0, 8.0], [0.0, 0.0]], [[0.0, 5.0], [1.0, 0.0]]])
        reading = kv_cache.read_layer(CacheLayer(keys, values, [4, 5]))

        self.assertEqual(reading["heads"], 2)
        self.assertEqual(reading["dim"], 2)
        self.assertEqual(reading["positions"], [4, 5])
        np.testing.assert_allclose(reading["key_norm"], [[5.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(reading["value_norm"], [[10.0, 0.0], [5.0, 1.0]])

    def test_similarity_leaves_out_the_mean_key_every_score_shares(self):
        # The same three directions, with and without an offset far larger
        # than any of them. The offset moves every attention score equally,
        # so the similarity must not see it.
        base = np.array([[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.1]]])
        plain = kv_cache.read_layer(CacheLayer(base, base, [0, 1, 2]))
        shifted = kv_cache.read_layer(CacheLayer(base + [50.0, 50.0], base, [0, 1, 2]))

        np.testing.assert_allclose(plain["key_similarity"], shifted["key_similarity"], atol=1e-5)
        similarity = plain["key_similarity"][0]
        self.assertAlmostEqual(similarity[-1], 1.0, places=5)
        self.assertGreater(similarity[0], 0.9)
        self.assertLess(similarity[1], -0.9)

    def test_a_single_position_has_nothing_to_compare_and_reads_zero(self):
        keys = np.ones((1, 1, 4))
        reading = kv_cache.read_layer(CacheLayer(keys, keys, [0]))
        self.assertEqual(reading["key_similarity"], [[0.0]])

    def test_a_long_layer_keeps_its_latest_positions(self):
        keys = np.arange(12, dtype=float).reshape(1, 6, 2)
        with mock.patch.object(kv_cache, "MAX_POSITIONS", 3):
            reading = kv_cache.read_layer(CacheLayer(keys, keys, list(range(6))))
        self.assertEqual(reading["positions"], [3, 4, 5])
        self.assertEqual(reading["held"], 6)
        self.assertEqual(len(reading["key_norm"][0]), 3)

    def test_a_given_mean_key_is_the_one_taken_out(self):
        keys = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        mean = np.array([[1.0, 1.0]])
        reading = kv_cache.read_layer(CacheLayer(keys, keys, [8, 9], key_mean=mean, held=10))
        # Centered on (1, 1), the two keys are (0, -1) and (-1, 0): at right angles.
        self.assertAlmostEqual(reading["key_similarity"][0][0], 0.0, places=5)
        self.assertEqual(reading["held"], 10)


class TorchCacheTests(unittest.TestCase):
    def setUp(self):
        self.manager = tiny_manager()
        self.ids = self.manager.tokenizer.encode("the cat sat on the mat")
        self.index = len(self.ids) - 1

    def inspect(self, index=None):
        index = self.index if index is None else index
        insight = self.manager.inspect(self.ids, index, load_id=self.manager.load_id)
        return [token["token_id"] for token in insight.to_dict()["tokens"]]

    def test_the_view_matches_the_keys_and_values_the_model_computes(self):
        held = self.inspect()
        view = self.manager.read_kv_cache(held, 2, load_id=self.manager.load_id)

        with torch.inference_mode():
            reference = self.manager.model(
                torch.tensor([self.ids[: self.index]]), use_cache=True
            ).past_key_values.layers[1]
        reading = view["reading"]
        self.assertEqual(view["layer"], 2)
        self.assertEqual(view["backend"], "torch")
        self.assertEqual(reading["positions"], list(range(self.index)))
        np.testing.assert_allclose(reading["key_norm"], norms(reference.keys[0]), rtol=1e-4)
        np.testing.assert_allclose(reading["value_norm"], norms(reference.values[0]), rtol=1e-4)

    def test_a_long_layer_copies_only_its_latest_positions_but_centers_on_all(self):
        held = self.inspect()
        with torch.inference_mode():
            reference = self.manager.model(
                torch.tensor([self.ids[: self.index]]), use_cache=True
            ).past_key_values.layers[1].keys[0].numpy()
        with mock.patch.object(kv_cache, "MAX_POSITIONS", 3):
            layer = self.manager._engine().cache_layer(
                self.manager._inspect_cache[2], 1, self.index
            )
            reading = self.manager.read_kv_cache(held, 2)["reading"]

        self.assertEqual(layer.keys.shape[1], 3)
        self.assertEqual(layer.values.shape[1], 3)
        self.assertEqual(layer.held, self.index)
        np.testing.assert_allclose(layer.key_mean, reference.mean(axis=1), rtol=1e-5, atol=1e-6)
        self.assertEqual(reading["positions"], list(range(self.index - 3, self.index)))
        self.assertEqual(reading["held"], self.index)
        centered = reference[:, -3:] - reference.mean(axis=1, keepdims=True)
        expected = np.einsum("hpd,hd->hp", centered, centered[:, -1]) / (
            np.linalg.norm(centered, axis=-1) * np.linalg.norm(centered[:, -1], axis=-1)[:, None]
        )
        np.testing.assert_allclose(reading["key_similarity"], expected, rtol=1e-4, atol=1e-5)

    def test_the_summary_counts_every_layer(self):
        summary = self.manager.read_kv_cache(self.inspect(), 1)["summary"]
        self.assertEqual(summary["tokens"], self.index)
        self.assertEqual(summary["layers"], 2)
        self.assertEqual(summary["attention_layers"], 2)
        self.assertEqual(summary["heads"], [2])
        self.assertEqual(summary["dims"], [8])
        self.assertEqual(summary["dtypes"], ["float32"])
        self.assertEqual(summary["positions"], [self.index, self.index])
        # Two layers, keys and values, two heads of eight float32 numbers per token.
        self.assertEqual(summary["nbytes"], 2 * 2 * 2 * self.index * 8 * 4)

    def test_the_layer_is_kept_inside_the_stack(self):
        held = self.inspect()
        self.assertEqual(self.manager.read_kv_cache(held, 99)["layer"], 2)
        self.assertEqual(self.manager.read_kv_cache(held, 0)["layer"], 1)

    def test_a_sliding_window_layer_numbers_its_latest_positions(self):
        self.manager = tiny_manager(
            Qwen3Config, Qwen3ForCausalLM,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=3, use_sliding_window=True,
        )
        held = self.inspect()
        full = self.manager.read_kv_cache(held, 1)
        sliding = self.manager.read_kv_cache(held, 2)

        self.assertEqual(full["reading"]["positions"], list(range(self.index)))
        # Transformers keeps the last ``sliding_window - 1`` keys.
        self.assertEqual(sliding["reading"]["positions"], [self.index - 2, self.index - 1])
        self.assertEqual(sliding["summary"]["positions"], [self.index, 2])

    def test_a_later_inspection_leaves_the_earlier_readout_without_its_cache(self):
        earlier = self.inspect()
        self.inspect(self.index - 2)
        with self.assertRaises(kv_cache.CacheGone):
            self.manager.read_kv_cache(earlier, 1)

    def test_a_reply_takes_the_cache_back(self):
        held = self.inspect()
        self.manager._drop_inspect_cache()
        with self.assertRaises(kv_cache.CacheGone):
            self.manager.read_kv_cache(held, 1)

    def test_a_reload_is_refused_by_name(self):
        held = self.inspect()
        with self.assertRaises(ModelChanged):
            self.manager.read_kv_cache(held, 1, load_id="test/tiny-decoder#9")

    def test_a_busy_model_is_not_waited_on(self):
        held = self.inspect()
        started, finish = threading.Event(), threading.Event()

        def hold():
            with self.manager._lock:
                started.set()
                finish.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        self.addCleanup(holder.join)
        self.addCleanup(finish.set)
        started.wait(5)
        with mock.patch.object(model_runtime, "KV_CACHE_WAIT", 0.05):
            with self.assertRaises(kv_cache.CacheBusy):
                self.manager.read_kv_cache(held, 1)


@needs_mlx
class MlxCacheTests(unittest.TestCase):
    def test_the_view_matches_the_mlx_cache(self):
        from mlx_runtime import MlxEngine
        from test_streaming import FakeTokenizer

        model = tiny_llama()
        manager = ModelManager()
        manager.model = model
        manager.engine = MlxEngine(model, {})
        manager.tokenizer = FakeTokenizer(tuple(f"t{i}" for i in range(32)), 1)
        manager.model_id = "mlx-community/tiny-4bit"
        manager.kind = "mlx"
        ids = [3, 5, 7, 9, 11, 2]
        insight = manager.inspect(ids, 5, load_id=manager.load_id).to_dict()
        view = manager.read_kv_cache([token["token_id"] for token in insight["tokens"]], 2)

        cache = make_prompt_cache(model)
        model(mx.array([ids[:5]]), cache=cache)
        keys, values = cache[1].state
        self.assertEqual(view["backend"], "mlx")
        self.assertEqual(view["reading"]["positions"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            view["reading"]["key_norm"], norms(np.array(keys[0].astype(mx.float32))), rtol=1e-4
        )
        np.testing.assert_allclose(
            view["reading"]["value_norm"], norms(np.array(values[0].astype(mx.float32))), rtol=1e-4
        )
        self.assertEqual(view["summary"]["tokens"], 5)

    def test_a_rotating_cache_that_has_wrapped_is_read_in_order(self):
        from mlx_runtime import MlxEngine

        cache = RotatingKVCache(max_size=4)
        for position in range(6):
            step = mx.full((1, 1, 1, 2), float(position + 1))
            cache.update_and_fetch(step, step)
        layer = MlxEngine(object(), {}).cache_layer([cache], 0, 6)

        self.assertEqual(layer.positions, [2, 3, 4, 5])
        np.testing.assert_allclose(layer.keys[0, :, 0], [3.0, 4.0, 5.0, 6.0])

    def test_a_long_layer_copies_only_its_latest_positions_but_centers_on_all(self):
        from mlx_lm.models.cache import KVCache
        from mlx_runtime import MlxEngine

        cache = KVCache()
        for position in range(6):
            step = mx.full((1, 1, 1, 2), float(position + 1))
            cache.update_and_fetch(step, step)
        with mock.patch.object(kv_cache, "MAX_POSITIONS", 3):
            layer = MlxEngine(object(), {}).cache_layer([cache], 0, 6)

        self.assertEqual(layer.positions, [3, 4, 5])
        self.assertEqual(layer.held, 6)
        np.testing.assert_allclose(layer.keys[0, :, 0], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(layer.values[0, :, 0], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(layer.key_mean, [[3.5, 3.5]])

    def test_a_rotating_cache_keeps_its_numbering_when_cut_short(self):
        from mlx_runtime import MlxEngine

        cache = RotatingKVCache(max_size=4, keep=1)
        for position in range(6):
            step = mx.full((1, 1, 1, 2), float(position + 1))
            cache.update_and_fetch(step, step)
        engine = MlxEngine(object(), {})
        whole = engine.cache_layer([cache], 0, 6)
        with mock.patch.object(kv_cache, "MAX_POSITIONS", 2):
            cut = engine.cache_layer([cache], 0, 6)

        self.assertEqual(whole.positions, [0, 3, 4, 5])
        self.assertEqual(cut.positions, [4, 5])
        self.assertEqual(cut.held, 4)
        np.testing.assert_allclose(cut.keys[0, :, 0], [5.0, 6.0])
        np.testing.assert_allclose(cut.key_mean, whole.keys.mean(axis=1))

    def test_a_quantized_cache_is_not_read(self):
        from mlx_lm.models.cache import QuantizedKVCache
        from mlx_runtime import MlxEngine

        cache = QuantizedKVCache(group_size=32, bits=4)
        step = mx.ones((1, 1, 1, 32))
        cache.update_and_fetch(step, step)
        self.assertEqual(MlxEngine(object(), {}).cache_shapes([cache]), [None])


class ChartTests(unittest.TestCase):
    def view(self, *, held=3, tokens=3, positions=None):
        keys = np.arange(held * 4, dtype=float).reshape(2, held, 2) + 1
        reading = kv_cache.read_layer(
            CacheLayer(keys, keys, positions or list(range(tokens - held, tokens)))
        )
        shapes = [kv_cache.LayerShape(2, tokens, 2, "bfloat16", 3 << 20),
                  kv_cache.LayerShape(2, held, 2, "bfloat16", 1 << 20)]
        return {"summary": kv_cache.summarize(shapes, tokens), "layer": 2, "reading": reading}

    def tokens(self, count):
        return [
            {"index": i, "token_id": i, "text": f"t{i}", "segment": "prompt" if i < 2 else "response"}
            for i in range(count)
        ]

    def test_the_summary_names_the_shape_and_the_memory(self):
        text = charts.kv_cache_summary(self.view(held=4, tokens=4)["summary"])
        self.assertIn("covers 4 tokens across 2 layers", text)
        self.assertIn("2 key-value heads of 2 dimensions", text)
        self.assertIn("bfloat16, 4.0 MB in all", text)
        self.assertNotIn("sliding window", text)

    def test_a_sliding_layer_is_called_out(self):
        page = charts.kv_cache_grid(self.view(held=2, tokens=5), self.tokens(5), "Key norm")
        self.assertIn("Layers with a sliding window keep only their latest 2.", page)
        self.assertIn("Layer 2 has a sliding window and holds only the latest 2 tokens.", page)
        # Positions 4 and 5 by their numbers from 1, and nothing earlier.
        self.assertIn("<th>4</th>", page)
        self.assertNotIn("<th>3</th>", page)

    def test_the_query_row_and_the_metric_are_marked(self):
        page = charts.kv_cache_grid(self.view(), self.tokens(3), "Key similarity")
        self.assertEqual(page.count('class="kv-query'), 1)
        self.assertIn("+1.00", page)
        self.assertIn("mean key", page)
        self.assertIn("head 2", page)

    def test_a_layer_without_keys_says_so(self):
        view = self.view() | {"reading": None}
        self.assertIn("Layer 2 stores no keys", charts.kv_cache_grid(view, [], "Key norm"))


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.manager = tiny_manager()
        self.ids = self.manager.tokenizer.encode("the cat sat on the mat")
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)

    def insight(self):
        insight = self.manager.inspect(self.ids, 4, load_id=self.manager.load_id).to_dict()
        return insight | {"load_id": self.manager.load_id}

    def test_a_readout_brings_its_cache_view_and_the_slider_range(self):
        page, slider = inspection.render_kv_cache(self.insight(), 1, "Value norm")
        self.assertIn("Key-value cache, layer 1", page)
        self.assertIn("Value norm", page)
        self.assertEqual(slider["maximum"], 2)
        self.assertEqual(slider["value"], 1)

    def test_a_cleared_readout_clears_the_view(self):
        page, _slider = inspection.render_kv_cache(None, 1, "Key norm")
        self.assertEqual(page, charts.EMPTY_KV_CACHE)

    def test_a_released_cache_asks_for_another_inspection(self):
        insight = self.insight()
        self.manager._drop_inspect_cache()
        page, _slider = inspection.render_kv_cache(insight, 1, "Key norm")
        self.assertIn("press Inspect layers again", page)

    def test_the_jacobian_lens_points_back_to_the_logit_lens(self):
        page, _slider = inspection.render_kv_cache({"kind": "jacobian"}, 1, "Key norm")
        self.assertIn("Select the Logit lens", page)


if __name__ == "__main__":
    unittest.main()
