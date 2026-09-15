"""Read imported Jacobian lenses without fitting or changing the loaded model.

File interoperability and readout convention follow anthropics/jacobian-lens:
https://github.com/anthropics/jacobian-lens (Apache-2.0).
Source layer zero is the output of decoder block zero, before the final norm.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_MODELS = {"llama", "qwen2", "qwen3"}
MAX_FILE_BYTES = 2 * 1024**3


def model_layout(engine):
    """Keep the initial support boundary explicit; other layouts need validation."""
    if engine.backend != "torch":
        raise ValueError("Jacobian inspection needs a Transformers model; MLX is not supported yet.")
    model = engine.model
    if getattr(model.config, "model_type", None) not in SUPPORTED_MODELS:
        raise ValueError("Jacobian inspection currently supports Llama, Qwen2, and Qwen3 text models.")
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("Load full-precision weights for Jacobian inspection; quantized models are not validated yet.")
    decoder = getattr(model, "model", None)
    blocks = getattr(decoder, "layers", None)
    norm = getattr(decoder, "norm", None)
    if blocks is None or norm is None or len(blocks) != model.config.num_hidden_layers:
        raise ValueError("This model's decoder layout is not supported by the Jacobian inspector.")
    return blocks, norm


@dataclass
class FittedLens:
    matrices: dict
    width: int
    n_prompts: int
    name: str

    @classmethod
    def load(cls, path, engine, model_id: str, fitted_model_id: str):
        import torch

        blocks, _ = model_layout(engine)
        if fitted_model_id.strip() != model_id:
            raise ValueError("The fitted model ID must exactly match the loaded model ID.")
        path = Path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError("Choose a saved lens.pt file no larger than 2 GiB.")
        # mmap keeps the whole set of dense matrices off the accelerator and
        # avoids eagerly copying the file into RAM. Never load arbitrary pickle.
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(data, dict) or not isinstance(data.get("J"), dict) or not data["J"]:
            raise ValueError("Expected a saved Jacobian lens with a nonempty J dictionary.")
        declared_model = data.get("model_id")
        if declared_model is not None and declared_model != model_id:
            raise ValueError("The lens file names a different model.")
        revision = data.get("model_revision")
        if revision is not None and revision != getattr(engine.model.config, "_commit_hash", None):
            raise ValueError("The lens file's model revision does not match the loaded checkpoint.")
        width = data.get("d_model")
        if type(width) is not int or width != engine.model.config.hidden_size:
            raise ValueError("The lens residual width does not match the loaded model.")
        count = data.get("n_prompts")
        if type(count) is not int or count <= 0:
            raise ValueError("The lens must record a positive n_prompts.")
        matrices = data["J"]
        if any(type(layer) is not int or not 0 <= layer < len(blocks) for layer in matrices):
            raise ValueError("The lens contains invalid decoder layer indices.")
        if data.get("source_layers") != sorted(matrices):
            raise ValueError("The lens source_layers must match its sorted matrix indices.")
        total = 0
        for matrix in matrices.values():
            if (
                not isinstance(matrix, torch.Tensor)
                or matrix.layout != torch.strided
                or matrix.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or tuple(matrix.shape) != (width, width)
            ):
                raise ValueError("Every lens matrix must be a dense floating-point d_model × d_model tensor.")
            total += matrix.numel() * matrix.element_size()
            if total > MAX_FILE_BYTES:
                raise ValueError("The lens matrices exceed the 2 GiB limit.")
            if not torch.isfinite(matrix).all():
                raise ValueError("The lens contains non-finite matrix entries.")
        return cls(matrices, width, count, path.name)

    @contextmanager
    def record(self, engine):
        """Capture only the selected position, including the raw final block."""
        blocks, _ = model_layout(engine)
        states, handles = {}, []

        def capture(layer):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                states[layer] = value[0, -1].detach().float().cpu().clone()
            return hook

        try:
            for layer in sorted(set(self.matrices) | {len(blocks) - 1}):
                handles.append(blocks[layer].register_forward_hook(capture(layer)))
            yield states
        finally:
            for handle in handles:
                handle.remove()

    def read(self, engine, states, actual_logits, decode, pinned_id=None):
        import torch

        blocks, norm = model_layout(engine)
        norm_weight = next(norm.parameters())
        head_weight = engine.model.get_output_embeddings().weight

        def unembed(vector):
            vector = norm(vector.to(device=norm_weight.device, dtype=norm_weight.dtype))
            return engine.read_head(vector.to(device=head_weight.device, dtype=head_weight.dtype)).float().cpu()

        if set(states) != set(self.matrices) | {len(blocks) - 1}:
            raise ValueError("The model did not return all requested decoder activations.")
        actual = torch.as_tensor(actual_logits, dtype=torch.float32)
        replayed = unembed(states[len(blocks) - 1])
        if not torch.allclose(replayed, actual, rtol=1e-2, atol=1e-2):
            raise ValueError("The final-layer readout does not reproduce this model's output; Jacobian results were withheld.")
        if pinned_id is not None and not 0 <= pinned_id < actual.numel():
            raise ValueError("The pinned token is outside this model's output vocabulary.")
        rows = []
        for layer, matrix in sorted(self.matrices.items()):
            # The reference transports in float32, then uses the model's norm
            # and head dtype. CPU matvec bounds accelerator memory per click.
            scores = unembed(matrix.float() @ states[layer])
            if not torch.isfinite(scores).all():
                raise ValueError("The Jacobian readout produced non-finite scores.")
            values, ids = scores.topk(min(5, scores.numel()))
            row = {
                "layer": layer,
                "candidates": [
                    {"token_id": int(token), "text": decode(int(token)), "score": float(score)}
                    for token, score in zip(ids, values)
                ],
            }
            if pinned_id is not None:
                row["rank"] = int((scores > scores[pinned_id]).sum()) + 1
                row["score"] = float(scores[pinned_id])
            rows.append(row)
        return rows


@dataclass
class JacobianInsight:
    payload: dict

    def to_dict(self):
        return self.payload
