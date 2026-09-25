import unittest
from types import SimpleNamespace

import numpy as np
import torch

from chatlab.model_inspection import TokenInsight
from chatlab.model_runtime import ModelManager
from chatlab.text_generation import ModelChanged
from fakes import FakeCache, lens_manager



class InspectTests(unittest.TestCase):
    def test_the_lens_shows_where_the_model_changed_its_mind(self):
        manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
        insight = manager.inspect([0, 1, 2, 3, 4], 2, context_count=2)

        self.assertEqual(insight.token_id, 2)
        self.assertEqual(insight.token_text, "!")
        self.assertEqual(len(insight.layers), 5)  # embeddings + 4 layers
        self.assertEqual([row["layer"] for row in insight.layers], [0, 1, 2, 3, 4])
        self.assertEqual(
            [row["rank"] == 1 for row in insight.layers],
            [False, False, True, True, True],
        )
        self.assertEqual(insight.layers[0]["top_id"], 3)
        self.assertEqual(insight.layers[-1]["top_id"], 2)
        self.assertEqual(insight.decided_at, 2)
        self.assertAlmostEqual(insight.layers[-1]["probability"], 1.0, places=5)

    def test_the_final_norm_is_found_one_level_down_in_the_decoder(self):
        # OPT keeps its norm at ``base_model.decoder.final_layer_norm``.
        manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
        norm = torch.nn.Identity()
        manager.model.base_model = SimpleNamespace(
            decoder=SimpleNamespace(final_layer_norm=norm)
        )
        self.assertIs(manager._final_norm(), norm)
        insight = manager.inspect([0, 1, 2, 3, 4], 2)
        self.assertEqual(len(insight.layers), 5)
        self.assertEqual(insight.decided_at, 2)

    def test_a_model_without_a_final_norm_shows_only_its_output(self):
        # Intermediate rows read without the norm would be wrong, not approximate.
        manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
        manager.model.base_model = SimpleNamespace(decoder=SimpleNamespace())
        self.assertIsNone(manager._final_norm())
        insight = manager.inspect([0, 1, 2, 3, 4], 2)
        self.assertEqual([row["layer"] for row in insight.layers], [4])
        self.assertEqual(insight.layers[0]["rank"], 1)
        self.assertEqual(insight.decided_at, 4)
        self.assertEqual(len(insight.attention), 4)

    def test_a_soft_capped_head_is_read_the_way_the_model_reads_it(self):
        # Gemma 2 and 3 squash logits with tanh before the softmax. A one-hot
        # +-20 hidden state capped at 2 gives the top token 1/(1 + 8e^-4).
        manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
        manager.model.config.final_logit_softcapping = 2.0
        insight = manager.inspect([0, 1, 2, 3, 4], 2)
        self.assertEqual(len(insight.layers), 5)
        expected = 1 / (1 + 8 * np.exp(-4.0))
        for row in insight.layers:
            self.assertAlmostEqual(row["top_probability"], expected, places=3)
        self.assertEqual(insight.decided_at, 2)

    def test_scaled_heads_are_read_the_way_the_model_reads_them(self):
        for name, value in (("logits_scaling", 4.0), ("logit_scale", 0.25)):
            with self.subTest(name=name):
                # Granite divides, Cohere multiplies; both leave +-5 logits.
                manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
                setattr(manager.model.config, name, value)
                insight = manager.inspect([0, 1, 2, 3, 4], 2)
                self.assertEqual(len(insight.layers), 5)
                expected = 1 / (1 + 8 * np.exp(-10.0))
                for row in insight.layers:
                    self.assertAlmostEqual(row["top_probability"], expected, places=4)

    def test_a_head_transform_the_lens_cannot_replicate_shows_only_the_output(self):
        manager = lens_manager([1, 2, 3, 4, 5], decide_layer=2, early=3)
        manager.model.undeclared_scale = 0.1
        insight = manager.inspect([0, 1, 2, 3, 4], 2)
        self.assertEqual([row["layer"] for row in insight.layers], [4])
        self.assertEqual(insight.layers[0]["rank"], 1)
        # The output row is still the model's own, transform included.
        self.assertAlmostEqual(
            insight.layers[0]["top_probability"], 1 / (1 + 8 * np.exp(-4.0)), places=3
        )

    def test_a_token_never_chosen_has_no_deciding_layer(self):
        manager = lens_manager([1, 2, 3], decide_layer=1)
        insight = manager.inspect([0, 1, 7], 2)
        self.assertIsNone(insight.decided_at)
        self.assertGreater(insight.layers[-1]["rank"], 1)

    def test_attention_has_one_row_per_layer_and_one_column_per_visible_token(self):
        manager = lens_manager([1, 2, 3, 4, 5], layers=3, focus=1)
        insight = manager.inspect([0, 1, 2, 3, 4], 4, context_count=1)

        self.assertEqual(len(insight.attention), 3)
        for row in insight.attention:
            self.assertEqual(len(row), 4)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            self.assertAlmostEqual(row[0], 0.6, places=5)
            self.assertAlmostEqual(row[1], 0.4, places=5)
        self.assertEqual([token["segment"] for token in insight.tokens], ["prompt", "response", "response", "response"])
        self.assertEqual([token["token_id"] for token in insight.tokens], [0, 1, 2, 3])
        self.assertEqual(insight.tokens[1]["text"], " world")

    def test_a_sliding_window_layer_is_aligned_on_the_keys_it_saw(self):
        manager = lens_manager([1, 2, 3, 4, 5], layers=3, focus=1)
        manager.model.sliding_layers = {1: 2}
        insight = manager.inspect([0, 1, 2, 3, 4], 4)

        self.assertEqual([len(row) for row in insight.attention], [4, 4, 4])
        # The window covered the last two keys, so the first two got nothing.
        self.assertEqual(insight.attention[1], [0.0, 0.0, 0.5, 0.5])
        self.assertAlmostEqual(insight.attention[0][0], 0.6, places=5)

    def test_attention_is_read_with_eager_kernels_and_switched_back(self):
        manager = lens_manager([1, 2, 3])
        manager.inspect([0, 1, 2, 3], 2)
        self.assertEqual(manager.model.attn_calls, ["eager", "sdpa"])
        self.assertEqual(manager.model.config._attn_implementation, "sdpa")

    def test_a_model_without_attention_weights_still_gives_the_lens(self):
        manager = lens_manager([1, 2, 3])
        manager.model.return_attentions = False
        insight = manager.inspect([0, 1, 2, 3], 2)
        self.assertEqual(insight.attention, [])
        self.assertEqual(len(insight.layers), 5)

    def test_the_second_token_needs_no_cache_warmup(self):
        manager = lens_manager([1, 2, 3])
        insight = manager.inspect([0, 1, 2, 3], 1)
        self.assertEqual(len(insight.tokens), 1)
        self.assertEqual(len(insight.attention[0]), 1)

    def test_the_first_token_and_out_of_range_positions_are_refused(self):
        manager = lens_manager([1, 2, 3])
        with self.assertRaises(ValueError):
            manager.inspect([0, 1, 2], 0)
        with self.assertRaises(ValueError):
            manager.inspect([0, 1, 2], 3)

    def test_tokens_from_another_load_are_refused_under_the_lock(self):
        manager = lens_manager([1, 2, 3])
        insight = manager.inspect([0, 1, 2], 1, load_id=manager.load_id)
        self.assertEqual(insight.token_id, 1)
        with self.assertRaises(ModelChanged):
            manager.inspect([0, 1, 2], 1, load_id="fake/lens#7")
        # Callers that do not care pass nothing and are not checked.
        manager.inspect([0, 1, 2], 1)

    def test_the_load_id_tells_one_load_of_a_model_from_the_next(self):
        manager = lens_manager([1, 2, 3])
        self.assertEqual(manager.load_id, "fake/lens#0")
        manager.load_count += 1
        self.assertEqual(manager.load_id, "fake/lens#1")
        self.assertIsNone(ModelManager().load_id)

    def test_an_unloaded_manager_is_refused(self):
        with self.assertRaises(RuntimeError):
            ModelManager().inspect([0, 1], 1)

    def test_the_next_click_reuses_the_cache_the_last_one_built(self):
        manager = lens_manager([1, 2, 3, 4, 5, 6, 7])
        manager.model.caching = True
        ids = [0, 1, 2, 3, 4, 5, 6]

        first = manager.inspect(ids, 3)
        self.assertEqual(manager.model.fed, [2, 1])
        cache = manager._inspect_cache
        self.assertEqual(cache[1], ids[:3])
        self.assertEqual(cache[2].length, 3)

        manager.model.fed.clear()
        second = manager.inspect(ids, 5)
        # Only the two tokens between the clicks, then the predicting one.
        self.assertEqual(manager.model.fed, [1, 1])
        self.assertEqual(manager._inspect_cache[1], ids[:5])
        self.assertIs(manager._inspect_cache[2], cache[2])
        self.assertEqual(second.tokens[:3], first.tokens[:3])

    def test_clicking_the_same_token_twice_feeds_nothing_new(self):
        manager = lens_manager([1, 2, 3, 4, 5])
        manager.model.caching = True
        ids = [0, 1, 2, 3, 4]
        manager.inspect(ids, 4)
        manager.model.fed.clear()

        manager.inspect(ids, 4)

        self.assertEqual(manager.model.fed, [1])
        # The cache held the four tokens the first click read and wrote; the
        # second needs the three before the predicting one, so one comes off.
        self.assertEqual(manager._inspect_cache[2].crops, [-1])
        self.assertEqual(manager._inspect_cache[2].length, 4)

    def test_an_earlier_click_crops_the_cache_instead_of_rebuilding_it(self):
        manager = lens_manager([1, 2, 3, 4, 5, 6, 7])
        manager.model.caching = True
        ids = [0, 1, 2, 3, 4, 5, 6]
        manager.inspect(ids, 6)
        cache = manager._inspect_cache[2]
        manager.model.fed.clear()

        manager.inspect(ids, 2)

        self.assertEqual(manager.model.fed, [1])
        # Six tokens held, one needed before the predicting token: five off,
        # then the predicting token is read into the cache.
        self.assertEqual(cache.crops, [-5])
        self.assertEqual(manager._inspect_cache[1], ids[:2])
        self.assertEqual(cache.length, 2)

    def test_an_earlier_click_rebuilds_a_sliding_cache_that_reached_its_window(self):
        # A hybrid model: one full layer and one with a window of 4. After six
        # tokens the sliding layer has let the first ones go, so cutting back
        # to two would leave a hole; the cache is rebuilt instead.
        manager = lens_manager([1, 2, 3, 4, 5, 6, 7])
        manager.model.caching = True
        manager.model.sliding_windows = [None, 4]
        ids = [0, 1, 2, 3, 4, 5, 6]
        manager.inspect(ids, 6)
        first = manager._inspect_cache[2]
        manager.model.fed.clear()

        manager.inspect(ids, 3)

        self.assertEqual(manager.model.fed, [2, 1])
        self.assertEqual(first.crops, [])
        self.assertIsNot(manager._inspect_cache[2], first)

    def test_an_earlier_click_rebuilds_a_cache_with_recurrent_layers(self):
        # Qwen3.5, LFM2 and Mamba keep a running state in some layers, which
        # cannot be cut back, and their crop raises rather than doing it.
        manager = lens_manager([1, 2, 3, 4, 5, 6, 7])
        manager.model.caching = True
        ids = [0, 1, 2, 3, 4, 5, 6]
        manager.inspect(ids, 6)
        first = manager._inspect_cache[2]
        first.is_croppable = False

        def refuse(_tokens):
            raise RuntimeError("crop was called, but the layer does not track past states")

        first.crop = refuse
        manager.model.fed.clear()

        manager.inspect(ids, 3)

        self.assertEqual(manager.model.fed, [2, 1])
        self.assertIsNot(manager._inspect_cache[2], first)

    def test_an_earlier_click_still_crops_a_sliding_cache_within_its_window(self):
        manager = lens_manager([1, 2, 3, 4, 5, 6, 7])
        manager.model.caching = True
        manager.model.sliding_windows = [None, 64]
        ids = [0, 1, 2, 3, 4, 5, 6]
        manager.inspect(ids, 6)
        cache = manager._inspect_cache[2]
        manager.model.fed.clear()

        manager.inspect(ids, 3)

        self.assertEqual(manager.model.fed, [1])
        self.assertEqual(cache.crops, [-4])

    def test_a_sliding_layer_with_an_unknown_window_is_never_cropped(self):
        from chatlab.torch_engine import _cache_can_crop

        unknown = FakeCache([None])
        unknown.is_sliding = [True]
        self.assertFalse(_cache_can_crop(unknown, 2))
        self.assertTrue(_cache_can_crop(FakeCache([None, None]), 500))
        self.assertTrue(_cache_can_crop(FakeCache([8]), 7))
        self.assertFalse(_cache_can_crop(FakeCache([8]), 8))
        self.assertFalse(_cache_can_crop(object(), 1))

    def test_a_different_sequence_rebuilds_the_cache(self):
        manager = lens_manager([1, 2, 3, 4, 5])
        manager.model.caching = True
        manager.inspect([0, 1, 2, 3, 4], 4)
        manager.model.fed.clear()

        manager.inspect([0, 2, 2, 3, 4], 4)

        self.assertEqual(manager.model.fed, [3, 1])

    def test_a_cache_from_another_load_is_not_trusted(self):
        manager = lens_manager([1, 2, 3, 4, 5])
        manager.model.caching = True
        manager.inspect([0, 1, 2, 3, 4], 4)
        manager.load_count += 1
        manager.model.fed.clear()

        manager.inspect([0, 1, 2, 3, 4], 4)

        self.assertEqual(manager.model.fed, [3, 1])

    def test_a_model_without_a_cache_is_inspected_as_before(self):
        manager = lens_manager([1, 2, 3, 4, 5])
        manager.inspect([0, 1, 2, 3, 4], 4)
        self.assertIsNone(manager._inspect_cache)
        manager.model.fed.clear()

        manager.inspect([0, 1, 2, 3, 4], 4)

        self.assertEqual(manager.model.fed, [3, 1])

    def test_unloading_forgets_the_cache(self):
        manager = lens_manager([1, 2, 3, 4, 5])
        manager.model.caching = True
        manager.inspect([0, 1, 2, 3, 4], 4)

        manager.unload()

        self.assertIsNone(manager._inspect_cache)

    def test_to_dict_copies_every_row(self):
        manager = lens_manager([1, 2, 3])
        insight = manager.inspect([0, 1, 2, 3], 2, context_count=1)
        payload = insight.to_dict()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["index"], 2)
        self.assertEqual(len(payload["layers"]), len(insight.layers))
        self.assertIsNot(payload["layers"][0], insight.layers[0])
        self.assertEqual(payload["decided_at"], insight.decided_at)
        self.assertIsInstance(TokenInsight(**{**payload}), TokenInsight)


if __name__ == "__main__":
    unittest.main()
