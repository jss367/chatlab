"""Exercise the dependencies the desktop bundle only imports lazily.

Nothing here is about whether the code is right. It is about whether
PyInstaller put it in the bundle: the Metal quantizer, every diffusers
pipeline class and every mlx-lm architecture are reached by name at run time,
so a missing hidden import in ``ChatLab.spec`` shows up as a failure on a
user's Mac and nowhere else.
"""

from __future__ import annotations

import importlib
from tempfile import TemporaryDirectory


# The pipeline and component classes a downloaded image repo names in its
# ``model_index.json``. ChatLab's own code imports none of them: they are
# looked up by name from that file, which is exactly the kind of import a
# bundle can be built without. These are the ones the common Stable Diffusion
# and SDXL repos ask for.
PIPELINE_CLASSES = (
    ("diffusers", "DiffusionPipeline"),
    ("diffusers", "StableDiffusionPipeline"),
    ("diffusers", "StableDiffusionXLPipeline"),
    ("diffusers", "UNet2DConditionModel"),
    ("diffusers", "AutoencoderKL"),
    ("diffusers", "PNDMScheduler"),
    ("diffusers", "EulerDiscreteScheduler"),
    ("diffusers", "DDIMScheduler"),
    ("transformers", "CLIPTextModel"),
    ("transformers", "CLIPTextModelWithProjection"),
    ("transformers", "CLIPTokenizer"),
)


def smoke_test_pipelines() -> None:
    """Check the bundle can build a diffusers pipeline by class name.

    ``DiffusionPipeline.from_pretrained`` reads ``model_index.json`` and
    resolves each component by looking its class up on the library the file
    names, so a bundle missing any of them loads no image model at all. The
    lookup is made the same way here, and the recording processor is built
    against the attention class it will wrap.
    """

    diffusers = importlib.import_module("diffusers")
    transformers = importlib.import_module("transformers")
    libraries = {"diffusers": diffusers, "transformers": transformers}
    missing = [
        f"{library}.{name}"
        for library, name in PIPELINE_CLASSES
        if getattr(libraries[library], name, None) is None
    ]
    if missing:
        raise RuntimeError(
            "The desktop bundle is missing pipeline classes a downloaded image "
            f"model would ask for: {', '.join(missing)}."
        )
    attention = importlib.import_module("diffusers.models.attention_processor")
    if not callable(getattr(attention.Attention, "get_attention_scores", None)):
        raise RuntimeError(
            "The desktop bundle's diffusers cannot report attention "
            "probabilities, so the Images page could map no prompt token."
        )
    print("ChatLab diffusers pipeline class checks passed")


def smoke_test_mlx() -> None:
    """Check the bundle can build and run an mlx-lm model, on Apple silicon.

    mlx-lm resolves each architecture by importing ``mlx_lm.models.<type>``
    from the ``model_type`` in a downloaded config, so a bundle that left the
    model modules or MLX's Metal library out fails on a user's Mac and
    nowhere else. Built from random weights; no download is needed. Skipped
    where mlx is not installed, which is every machine but an Apple silicon
    Mac.
    """

    import mlx_runtime

    if not mlx_runtime.mlx_available():
        print("SKIP: MLX models need mlx and mlx-lm, which install on Apple silicon only")
        return

    mx = importlib.import_module("mlx.core")
    llama = importlib.import_module("mlx_lm.models.llama")
    importlib.import_module("mlx_lm.utils")
    cache = importlib.import_module("mlx_lm.models.cache")
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=32,
    )
    model = llama.Model(args)
    model.eval()
    mx.eval(model.parameters())
    engine = mlx_runtime.MlxEngine(model, {"model_type": "llama"})
    logits, kv = engine.forward([1, 2, 3], None, 0)
    row = logits.row(-1)
    if row.shape != (128,) or not all(map(lambda value: value == value, row)):
        raise RuntimeError("MLX smoke model produced invalid logits.")
    reading = engine.inspect_step(4, kv, 3)
    if len(reading.layer_logits) != 2 or len(reading.attention) != 2:
        raise RuntimeError(
            "MLX smoke model could not be read layer by layer: "
            f"{len(reading.layer_logits)} lens rows, {len(reading.attention)} attention rows."
        )
    if not isinstance(cache.make_prompt_cache(model)[0], cache.KVCache):
        raise RuntimeError("MLX smoke model did not build a key-value cache.")
    print("ChatLab MLX load, forward pass and lens checks passed")


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
