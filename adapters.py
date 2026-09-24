"""LoRA adapters: a small set of weights trained on top of a text model.

A PEFT adapter repository holds no model of its own. ``adapter_config.json``
names the checkpoint it was trained on (``base_model_name_or_path``) and
``adapter_model.safetensors`` holds a pair of low-rank matrices for each
layer it changes. ChatLab reads the base checkpoint the way it reads any text
model, adds the adapter, and merges it: every changed weight becomes
``W + B @ A * scale``, and what is left in memory is a plain Transformers
model. That is what keeps the logit lens, steering, inspection and patching
working unchanged, since each of them walks the model's own modules and an
unmerged PEFT wrapper would put another layer of names in front of them.

Merging needs the full-precision weights, because a packed 4-bit or 8-bit
matrix has nothing to add the product to, so an adapter always loads at full
precision. Nothing here knows about the cache; :mod:`model_runtime` finds the
base checkpoint on disk and calls in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ADAPTER_CONFIG = "adapter_config.json"
# What ``PeftModel.from_pretrained`` reads, in the order it looks for them.
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

# The adapter types that merge into the weights they were trained on. Prompt
# and prefix tuning add virtual tokens rather than changing any weight, so
# there is nothing to merge, and the other weight-changing types are left
# out until one has been loaded and checked here.
MERGEABLE_TYPES = frozenset({"LORA"})
# ``task_type`` is optional in an adapter config; when it is set it has to be
# the one ChatLab runs. A classifier head on a sequence model is not a chat.
CAUSAL_TASKS = frozenset({"CAUSAL_LM"})

# The same form model_runtime.MODEL_ID_PATTERN checks, repeated so this module
# imports nothing of the runtime's.
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Unsloth trains on its own bitsandbytes 4-bit copies of the popular models
# and records that copy as the base, but bitsandbytes does not run on Apple
# silicon and a packed matrix cannot take a merge. Unsloth publishes the
# full-precision weights each copy was quantized from under the same name
# without the suffix, and an adapter trained on one fits the other.
UNSLOTH_4BIT = re.compile(r"^(unsloth/.+?)(?:-unsloth)?-bnb-4bit$")


def read_adapter_config(snapshot: Path | None) -> dict[str, Any] | None:
    """The adapter config in ``snapshot``, or ``None`` where there is none to read."""

    if snapshot is None:
        return None
    try:
        config = json.loads((Path(snapshot) / ADAPTER_CONFIG).read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def is_adapter(snapshot: Path | None) -> bool:
    """Whether ``snapshot`` is an adapter repository rather than a model."""

    return read_adapter_config(snapshot) is not None


def trained_base(config: dict[str, Any]) -> str | None:
    """The base the adapter's config names, as written, or ``None`` for none."""

    base = config.get("base_model_name_or_path")
    return base.strip() if isinstance(base, str) and base.strip() else None


def base_model_id(config: dict[str, Any]) -> str | None:
    """The Hub ID of the checkpoint ChatLab loads the adapter onto, or ``None``.

    The base the config names, except that an Unsloth 4-bit copy is swapped
    for the full-precision repository it was made from; see
    :data:`UNSLOTH_4BIT`. ``None`` when the config names no Hub repository,
    which is what a local path left by the training machine looks like.
    """

    base = trained_base(config)
    if base is None or not HUB_ID.fullmatch(base):
        return None
    unquantized = UNSLOTH_4BIT.fullmatch(base)
    return unquantized.group(1) if unquantized else base


def base_revision(config: dict[str, Any]) -> str | None:
    """The commit, branch or tag of the base the adapter pins, or ``None``.

    PEFT's ``revision`` records which state of the base repository the
    adapter was trained on, and the Hub's default branch can have moved since:
    merged into other weights, the adapter would load without complaint and
    say something else. ``None`` leaves it to the default branch, which is
    what a config that pins nothing means. A pin on an Unsloth 4-bit copy is
    dropped with the swap in :func:`base_model_id`, because it names a commit
    in the 4-bit repository's history, which the full-precision one does not
    share.
    """

    revision = config.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        return None
    base = trained_base(config)
    if base is not None and UNSLOTH_4BIT.fullmatch(base):
        return None
    return revision.strip()


def adapter_problem(config: dict[str, Any], adapter_id: str | None = None) -> str | None:
    """Why ChatLab cannot load this adapter, or ``None`` when it can.

    Worded to follow "This adapter", so the Models page can put it after the
    adapter's name.
    """

    peft_type = str(config.get("peft_type") or "").upper()
    if peft_type not in MERGEABLE_TYPES:
        named = f"a {peft_type} adapter" if peft_type else "an adapter of no stated type"
        return f"is {named}; ChatLab loads LoRA adapters, which merge into the model's weights."
    task = config.get("task_type")
    if task is not None and str(task).upper() not in CAUSAL_TASKS:
        return f"was trained for {task}, not for generating text."
    base = trained_base(config)
    if base is None:
        return "does not name the model it was trained on (`base_model_name_or_path`)."
    loadable = base_model_id(config)
    if loadable is None:
        return (
            f"names its base model as `{base}`, a path on the machine it was "
            "trained on rather than a Hugging Face ID."
        )
    if adapter_id is not None and loadable == adapter_id:
        return "names itself as its own base model."
    return None


def missing_adapter_files(snapshot: Path) -> tuple[str, ...]:
    """The files an adapter snapshot still needs before it can be merged."""

    return () if has_adapter_weights(snapshot) else (ADAPTER_WEIGHTS[0],)


def has_adapter_weights(snapshot: Path) -> bool:
    """Whether the adapter's weights sit at the root, config or no config."""

    return any((snapshot / name).is_file() for name in ADAPTER_WEIGHTS)


def has_tokenizer(snapshot: Path) -> bool:
    """Whether the adapter ships a tokenizer of its own.

    One trained with added tokens, or with a chat template the base lacks,
    saves its tokenizer beside the weights, and that tokenizer is the one the
    adapter's embeddings are indexed by.
    """

    return (snapshot / "tokenizer_config.json").is_file()


def merge_adapter(model, adapter_path: Path, vocabulary: int | None = None):
    """``model`` with the adapter at ``adapter_path`` merged into its weights.

    ``vocabulary`` is the size of the tokenizer the adapter was trained
    with, when it shipped one. A tokenizer that grew during training grew
    the embeddings with it, and the adapter's saved embedding rows fit only
    a matrix that has grown to match, so the base is resized first. Only
    ever upwards: many checkpoints pad the matrix past the tokenizer's
    length, and cutting those rows off would break the model.
    """

    from peft import PeftModel

    rows = model.get_input_embeddings().weight.shape[0]
    if vocabulary is not None and vocabulary > rows:
        model.resize_token_embeddings(vocabulary)
    wrapped = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    return wrapped.merge_and_unload()
