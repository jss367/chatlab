"""Exercise the desktop bundle's lazily imported Metal dependencies."""

from __future__ import annotations

import importlib
from tempfile import TemporaryDirectory


def smoke_test_metal() -> None:
    """Check imports everywhere, and load both weight precisions on an MPS Mac.

    The checkpoint is synthesized locally; no model download is needed. The
    first forward pass may download the small Metal kernel from Hugging Face,
    just as a user's first quantized generation does.
    """

    import torch
    import transformers

    # Use the same lazy registry as from_pretrained. Dynamic imports also keep
    # this smoke test from accidentally fixing the recipe's hidden imports.
    quantizers = importlib.import_module("transformers.quantizers")
    metal = importlib.import_module("transformers.integrations.metal_quantization")
    hub = importlib.import_module("transformers.integrations.hub_kernels")
    kernels = importlib.import_module("kernels")
    if not transformers.utils.is_kernels_available() or not callable(kernels.get_kernel):
        raise RuntimeError("The desktop bundle is missing compatible kernels or its metadata.")
    if not callable(hub.get_kernel):
        raise RuntimeError("The desktop bundle is missing the Transformers Hub kernel loader.")
    configs = [transformers.MetalConfig(bits=bits, group_size=64) for bits in (4, 8)]
    for config in configs:
        quantizer = quantizers.AutoHfQuantizer.from_config(config, pre_quantized=False)
        if quantizer.__class__.__name__ != "MetalHfQuantizer":
            raise RuntimeError(f"Metal configuration selected an unexpected quantizer: {quantizer}")
    print("ChatLab Metal dependency and quantizer checks passed")

    if not torch.backends.mps.is_available():
        print("SKIP: Metal quantized load and forward pass require an available MPS GPU")
        return

    with TemporaryDirectory(prefix="chatlab-metal-smoke-") as checkpoint:
        config = transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        original = transformers.AutoModelForCausalLM.from_config(config)
        original.save_pretrained(checkpoint)
        del original
        for quantization_config in configs:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint,
                local_files_only=True,
                dtype=torch.float16,
                device_map="mps",
                quantization_config=quantization_config,
            ).eval()
            try:
                packed = [module for module in model.modules() if isinstance(module, metal.MetalLinear)]
                if not packed or any(module.weight.dtype != torch.uint32 for module in packed):
                    raise RuntimeError("Metal smoke model did not load packed quantized weights.")
                with torch.inference_mode():
                    logits = model(torch.tensor([[1, 2, 3]], device="mps")).logits
                if logits.shape != (1, 3, 128) or not torch.isfinite(logits).all().item():
                    raise RuntimeError("Metal smoke model produced invalid logits.")
                print(f"ChatLab {quantization_config.bits}-bit Metal load and forward pass passed")
            finally:
                del model
                torch.mps.empty_cache()
