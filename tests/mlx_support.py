"""A tiny real mlx-lm model, for the tests that run the MLX backend.

mlx only installs on Apple silicon, so the import is allowed to fail and the
tests that need it are marked with ``needs_mlx`` to skip themselves elsewhere.
"""

import unittest

try:
    import mlx.core as mx
    from mlx_lm.models import gpt2, llama
except ImportError:  # pragma: no cover - the engine tests skip themselves
    mx = None
    gpt2 = None
    llama = None

needs_mlx = unittest.skipIf(mx is None, "mlx and mlx-lm are installed on Apple silicon only")


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
