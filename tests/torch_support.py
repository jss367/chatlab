"""A small real Transformers decoder, loaded into a ModelManager, for the tests to build.

Kept here so the test modules share it without importing each other.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from chatlab.model_runtime import ModelManager
from tiny_tokenizer import build


def tiny_manager(config_type=LlamaConfig, model_type=LlamaForCausalLM, *, seed=7, layers=2, heads=4, **extra):
    torch.manual_seed(seed)
    manager = ModelManager()
    manager.tokenizer = build()
    config = config_type(
        vocab_size=len(manager.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=0, pad_token_id=0, **extra,
    )
    manager.model = model_type(config).eval()
    manager.model_id = "test/tiny-decoder"
    manager.precision = "full"
    return manager
