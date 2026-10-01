"""Showing a vision-language model the pictures in a conversation.

A vision model reads a picture as a run of placeholder tokens in its prompt -
``<|image_pad|>`` in Qwen, ``<image_soft_token>`` in Gemma - which its
processor writes out at the length the picture needs, and whose embeddings
the model swaps for what its vision encoder made of the pixels. Everything
past that swap is an ordinary language model, which is what lets every token
measurement, the logit lens, the attention view and branching work on a
conversation with pictures in it.

ChatLab cannot hand the model ``pixel_values`` and let it do the swap,
though, because ChatLab never feeds a prompt whole. A prefill goes in chunks
(:data:`text_generation.PREFILL_CHUNK_SIZE`), an inspection feeds only what
the last one's cache did not cover, and a response feeds one token at a
time, while the model's own path expects every picture's placeholders and
pixels in one call. So the encoder is run once per picture set
(:func:`encode_images`) and the result is laid out against the token
sequence (:class:`MediaLayout`): each forward call is given the encoder rows
for the placeholders it feeds, through the ``mm_encoder_outputs`` argument
Transformers models accept for exactly this, along with whatever positions
the architecture needs to know where it is. Qwen's multimodal RoPE is the
case that needs them: a picture's tokens are numbered by row and column
rather than one after another, every token after it is shifted back by how
much shorter that numbering is, and a chunk on its own cannot know any of
that.

Heavy imports (torch, transformers) are made where they are used.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A processor's settings, under the names processors save them. One of them is
# what says a repository ships what it takes to turn pixels into model input.
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")

# What a refusal suggests loading instead: small enough for most Macs, and the
# vision model this was tested against.
VISION_SUGGESTION = "Qwen/Qwen3.5-4B"

# Why a model in memory cannot read pictures, for the message that refuses one.
NO_VISION = "is a text-only model"
MLX_VISION = "runs through MLX, which ChatLab only runs as a text model"


class ImagesUnsupported(ValueError):
    """The model in memory cannot be shown pictures."""


def _read_config(local_path: Path) -> dict:
    try:
        value = json.loads((Path(local_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def checkpoint_has_vision(local_path: Path) -> bool:
    """Whether the config at ``local_path`` describes a vision tower at all."""

    return isinstance(_read_config(local_path).get("vision_config"), dict)


def checkpoint_reads_images(local_path: Path) -> bool:
    """Whether the checkpoint at ``local_path`` is a vision model ChatLab can show pictures.

    Three things have to hold: the config describes a vision tower, which a
    text-only conversion of a multimodal family does not; Transformers has
    an image-text-to-text class for its ``model_type``; and the repository
    ships a processor, without which there is no way to turn a picture into
    the tensors that class takes.
    """

    if not checkpoint_has_vision(local_path):
        return False
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError:
        return False
    if _read_config(local_path).get("model_type") not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return False
    return any((Path(local_path) / name).is_file() for name in PROCESSOR_FILES)


def model_class(local_path: Path):
    """The auto class a text checkpoint is read with.

    A vision model is read whole, encoder and all, through
    ``AutoModelForImageTextToText``. For some families that is what
    ``AutoModelForCausalLM`` would have read anyway (Gemma 3 and 4); for
    others (Qwen3.5) the causal-LM class is the language model alone and the
    vision weights would be left on disk.
    """

    if checkpoint_reads_images(local_path):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def image_token_id(model) -> int | None:
    """The placeholder token a picture's encoder rows replace, or ``None``."""

    config = getattr(model, "config", None)
    for name in ("image_token_id", "image_token_index"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _base(model):
    return getattr(model, "model", None) or getattr(model, "base_model", model)


def _uses_mrope(model) -> bool:
    rope_index = getattr(_base(model), "get_rope_index", None)
    if rope_index is None:
        return False
    parameters = inspect.signature(rope_index).parameters
    return "mm_token_type_ids" in parameters and "image_grid_thw" in parameters


def read_processor(local_path: Path, model) -> tuple[Any, str | None]:
    """The processor that prepares pictures for ``model``, and why there is none.

    ``None`` with no reason for a text model, which is the ordinary case and
    nothing to explain. A vision model whose processor will not load is still
    a working text model, so the load goes ahead and the reason is kept for
    the message that refuses a picture: usually a missing package.
    """

    if model is None or image_token_id(model) is None:
        return None, None
    if not callable(getattr(model, "get_image_features", None)):
        return None, None
    if getattr(_base(model), "get_rope_index", None) is not None and not _uses_mrope(model):
        return None, "numbers its picture tokens in a way ChatLab does not lay out yet"
    if not checkpoint_reads_images(local_path):
        return None, "ships no image processor"
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    except Exception as error:  # noqa: BLE001 - any failure leaves a text model
        logger.warning("Could not read the image processor for %s: %s", local_path, error)
        return None, f"has an image processor that would not load ({str(error).strip().splitlines()[0]})"
    if getattr(processor, "image_processor", None) is None:
        return None, "ships no image processor"
    return processor, None


def message_images(messages: Sequence[dict]) -> list[str]:
    """Every picture ``messages`` carry, in the order the prompt places them."""

    return [name for message in messages for name in message.get("images") or []]


def template_messages(messages: Sequence[dict]) -> list[dict]:
    """``messages`` with each picture written into its content the way chat templates read it.

    A message with pictures becomes a list of parts, pictures first: that is
    the order the model cards ask for, and the one a reader means by pasting
    a screenshot and then asking about it. A message without pictures keeps
    its plain string, so a conversation that has none renders exactly as it
    did before.
    """

    rendered = []
    for message in messages:
        names = message.get("images") or []
        entry = {key: value for key, value in message.items() if key != "images"}
        if names:
            parts = [{"type": "image"} for _ in names]
            if message.get("content"):
                parts.append({"type": "text", "text": message["content"]})
            entry["content"] = parts
        rendered.append(entry)
    return rendered


def expand_prompt(processor, rendered: str, images: Sequence[Any]) -> list[int]:
    """Token ids for a rendered chat prompt, each picture's placeholder written out at full length.

    The template writes one placeholder per picture; the processor knows how
    many tokens each picture really takes and writes that many. Special
    tokens are not added, because the template has put them in already.
    """

    encoded = processor(
        text=[rendered], images=list(images), return_tensors="pt", add_special_tokens=False
    )
    return [int(value) for value in encoded["input_ids"][0].tolist()]


@dataclass
class ImageFeatures:
    """What a vision encoder made of one conversation's pictures.

    ``rows`` holds one row per placeholder token, every picture's in order,
    on the model's device. ``sequence`` records whether the model hands its
    encoder output around as a tuple of per-picture tensors (Qwen, LLaVA,
    Mistral) or as one tensor (Gemma 3, PaliGemma), because its forward
    reads ``mm_encoder_outputs`` back in the form it wrote it. ``grid`` is
    each picture's temporal, height and width patch count, for the models
    whose positions follow the picture's shape.
    """

    rows: Any
    sequence: bool
    grid: Any = None

    @property
    def count(self) -> int:
        return int(self.rows.shape[0])


def encode_images(model, processor, images: Sequence[Any]) -> ImageFeatures:
    """Run the vision encoder over ``images`` once, for every forward call that needs them."""

    import torch

    prepared = processor.image_processor(images=list(images), return_tensors="pt")
    accepted = inspect.signature(model.get_image_features).parameters
    parameter = next(iter(model.parameters()))
    arguments = {}
    for key, value in prepared.items():
        if key not in accepted:
            continue
        if isinstance(value, torch.Tensor):
            value = value.to(parameter.device)
            if value.is_floating_point():
                value = value.to(parameter.dtype)
        arguments[key] = value
    output = model.get_image_features(**arguments, return_dict=True)
    pooled = getattr(output, "pooler_output", output)
    sequence = isinstance(pooled, (list, tuple))
    parts = list(pooled) if sequence else [pooled]
    rows = torch.cat([part.reshape(-1, part.shape[-1]) for part in parts], dim=0)
    grid = prepared.get("image_grid_thw")
    return ImageFeatures(rows=rows.detach(), sequence=sequence, grid=grid)


class MediaLayout:
    """One token sequence's pictures, laid out for whichever slice of it is fed next.

    Built for the sequence a prompt (or an inspected transcript) is, with
    that sequence's encoder output. Every forward call over any part of it -
    a prefill chunk, an inspection, one sampled token after the prompt -
    asks :meth:`forward_arguments` for what to add to the model call.
    Positions past the end of ``token_ids`` are text the sequence has grown
    by since, a response being sampled.
    """

    def __init__(self, model, token_ids: Sequence[int], features: ImageFeatures) -> None:
        import torch

        self.token_id = image_token_id(model)
        if self.token_id is None:
            raise ImagesUnsupported("This model has no image token.")
        self.features = features
        self.length = len(token_ids)
        ids = torch.tensor([list(token_ids)], dtype=torch.long)
        placed = ids[0] == self.token_id
        found = int(placed.sum())
        if found != features.count:
            raise ValueError(
                f"The prompt has room for {found:,} picture tokens but its pictures "
                f"make {features.count:,}. An edited prompt must keep each "
                "picture's tokens as they were."
            )
        # Where each encoder row goes: the k-th placeholder takes row k.
        self.ordinal = torch.cumsum(placed.long(), dim=0) - placed.long()
        self.placed = placed
        self.device = features.rows.device
        forward = inspect.signature(model.forward).parameters
        self.positions = None
        self.delta = 0
        if _uses_mrope(model):
            type_ids = placed.long().unsqueeze(0)
            positions, delta = _base(model).get_rope_index(
                ids, mm_token_type_ids=type_ids, image_grid_thw=features.grid
            )
            text_row = torch.arange(self.length, dtype=positions.dtype).view(1, 1, -1)
            self.positions = torch.cat([text_row, positions.cpu()], dim=0)
            self.delta = int(delta.reshape(-1)[0])
        # Gemma lets a picture's tokens attend to one another in both
        # directions, and works out which tokens those are from a type-id
        # tensor the processor would have handed it beside the ids. The mask
        # reads it by absolute position, so it is given for everything up to
        # the end of each call, cache included.
        self.type_name = None
        if self.positions is None:
            for name in ("mm_token_type_ids", "token_type_ids"):
                if name in forward:
                    self.type_name = name
                    break

    @property
    def joined(self) -> bool:
        """Whether a picture's tokens attend to one another in both directions.

        Such a picture has to be fed in one call: tokens fed before the rest
        of it arrived would have attended to less of it than the model
        expects, and a cache built that way differs from the one a whole
        prompt leaves.
        """

        return self.type_name is not None

    def is_placeholder(self, position: int) -> bool:
        return 0 <= position < self.length and bool(self.placed[position])

    def run_start(self, position: int) -> int:
        """Where the picture holding ``position`` begins, for a picture fed whole.

        ``position`` itself when it is not inside one, or when the model reads
        pictures causally and a picture can be split like any other run of
        tokens.
        """

        if not self.joined:
            return position
        while position > 0 and self.is_placeholder(position - 1) and self.is_placeholder(position):
            position -= 1
        return position

    def chunk_end(self, end: int) -> int:
        """Move a chunk's end past the picture it would otherwise cut, when that matters."""

        if not self.joined:
            return end
        while self.is_placeholder(end) and self.is_placeholder(end - 1):
            end += 1
        return end

    def forward_arguments(self, start: int, count: int) -> dict:
        """What a model call feeding positions ``start`` to ``start + count`` adds to its arguments."""

        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        end = start + count
        arguments: dict[str, Any] = {}
        inside = slice(min(start, self.length), min(end, self.length))
        placed = self.placed[inside]
        if bool(placed.any()):
            first = int(self.ordinal[inside][placed][0])
            rows = self.features.rows[first : first + int(placed.sum())]
            pooled = (rows,) if self.features.sequence else rows
            arguments["mm_encoder_outputs"] = {
                "image": BaseModelOutputWithPooling(pooler_output=pooled)
            }
        if self.positions is not None:
            known = self.positions[:, :, inside]
            beyond = max(0, end - max(start, self.length))
            if beyond:
                after = torch.arange(max(start, self.length), end, dtype=known.dtype)
                tail = torch.stack([after] + [after + self.delta] * 3).view(4, 1, -1)
                known = torch.cat([known, tail], dim=2)
            arguments["position_ids"] = known.to(self.device)
        if self.type_name is not None:
            types = torch.zeros((1, end), dtype=torch.long)
            covered = min(end, self.length)
            types[0, :covered] = self.placed[:covered].long()
            arguments[self.type_name] = types.to(self.device)
        return arguments
