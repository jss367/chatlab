import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model_runtime import ModelManager, TokenInsight

from test_streaming import EOS_ID, PIECES, FakeTokenizer


class FakeConfig:
    def __init__(self, layers: int):
        self.num_hidden_layers = layers
        self._attn_implementation = "sdpa"


class FakeLensModel(torch.nn.Module):
    """A model whose layers change their mind partway up the stack.

    Position ``k`` of the sequence predicts ``script[k]``, so a sequence is
    self-consistent when it is one leading token followed by the script. The
    residual stream below ``decide_layer`` points at ``early`` instead, so the
    logit lens has something to show. The hidden size equals the vocab
    size and the head is the identity, so a one-hot hidden state is its own
    logit vector. Attention puts most weight on the first key (the sink) and
    the rest on ``focus``.
    """

    def __init__(self, script, *, layers=4, heads=2, decide_layer=2, early=3, focus=1):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.script = script
        self.vocab = len(PIECES)
        self.layers = layers
        self.heads = heads
        self.decide_layer = decide_layer
        self.early = early
        self.focus = focus
        self.config = FakeConfig(layers)
        self.generation_config = SimpleNamespace(eos_token_id=EOS_ID)
        self.head = torch.nn.Linear(self.vocab, self.vocab, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(self.vocab))
        self.base_model = SimpleNamespace(norm=torch.nn.Identity())
        self.attn_calls: list[str] = []
        self.return_attentions = True

    def set_attn_implementation(self, name: str) -> None:
        self.attn_calls.append(name)
        self.config._attn_implementation = name

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        length = int(input_ids.shape[-1])
        keys = int(attention_mask.shape[-1])
        first = keys - length
        targets = [
            self.script[(first + offset) % len(self.script)] for offset in range(length)
        ]

        def one_hot(indices):
            state = torch.full((1, length, self.vocab), -20.0)
            for offset, index in enumerate(indices):
                state[0, offset, index] = 20.0
            return state

        hidden = tuple(
            one_hot([self.early] * length if layer < self.decide_layer else targets)
            for layer in range(self.layers + 1)
        )
        logits = self.head(hidden[-1])
        attentions = None
        if output_attentions and self.return_attentions:
            weights = torch.full((1, self.heads, length, keys), 0.0)
            weights[..., 0] = 0.6
            if keys > 1:
                weights[..., min(self.focus, keys - 1)] += 0.4
            else:
                weights[..., 0] += 0.4
            attentions = tuple(weights.clone() for _ in range(self.layers))
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            hidden_states=hidden if output_hidden_states else None,
            attentions=attentions,
        )


def lens_manager(script, **options) -> ModelManager:
    manager = ModelManager()
    manager.tokenizer = FakeTokenizer()
    manager.model = FakeLensModel(script, **options)
    manager.model_id = "fake/lens"
    return manager


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

    def test_an_unloaded_manager_is_refused(self):
        with self.assertRaises(RuntimeError):
            ModelManager().inspect([0, 1], 1)

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
