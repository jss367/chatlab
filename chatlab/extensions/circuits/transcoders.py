"""Per-layer transcoders: which models have them, fetching them, and running them.

A transcoder stands in for one decoder block's MLP. It reads what the MLP
reads, writes a sparse set of feature activations, and decodes them into an
approximation of what the MLP wrote. Attribution needs one for every layer.

The sets here are the ones circuit-tracer publishes on the Hub, in its layout:
one ``layer_<n>.safetensors`` per layer, each holding ``W_enc``, ``b_enc``,
``W_dec``, ``b_dec`` and, for a JumpReLU set, ``activation_function.threshold``.
Beside the weights sits a ``features`` directory with each feature's
top-activating examples and top output tokens, which is read one feature at a
time by byte range rather than downloaded whole.
"""

from __future__ import annotations

import gc
import gzip
import json
import logging
import struct
import threading
from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscoderSpec:
    """One published transcoder set and the model it was trained on.

    ``model_ids`` lists every Hub ID the set reads correctly: the repository it
    was trained on and any mirror with the same weights. ``input`` and
    ``output`` name where in a decoder block the set reads and writes - see
    :mod:`chatlab.extensions.circuits.architecture` for what each means.
    """

    key: str
    title: str
    model_ids: tuple[str, ...]
    repo: str
    subfolder: str
    layers: int
    width: int
    d_model: int
    weights_gb: float
    input: str = "mlp_input"
    output: str = "mlp_output"

    def path(self, name):
        return f"{self.subfolder}/{name}" if self.subfolder else name


CATALOGUE = (
    TranscoderSpec(
        "gemma-3-270m-it", "Gemma Scope 2 · 16k features per layer",
        ("google/gemma-3-270m-it",), "mwhanna/gemma-scope-2-270m-it",
        "transcoder_all/width_16k_l0_big", layers=18, width=16384, d_model=640, weights_gb=1.5,
    ),
    TranscoderSpec(
        "gemma-3-1b-it", "Gemma Scope 2 · 16k features per layer",
        ("google/gemma-3-1b-it",), "mwhanna/gemma-scope-2-1b-it",
        "transcoder_all/width_16k_l0_big", layers=26, width=16384, d_model=1152, weights_gb=3.9,
    ),
    TranscoderSpec(
        "gemma-2-2b", "Gemma Scope · 16k features per layer",
        ("google/gemma-2-2b",), "mwhanna/gemma-scope-transcoders", "",
        layers=26, width=16384, d_model=2304, weights_gb=7.9,
    ),
    TranscoderSpec(
        "qwen3-0.6b", "Qwen3 transcoders, low L0 · 164k features per layer",
        ("Qwen/Qwen3-0.6B",), "mwhanna/qwen3-0.6b-transcoders-lowl0", "",
        layers=28, width=163840, d_model=1024, weights_gb=18.8,
    ),
)


def spec_for(model_id):
    """The transcoder set for a Hub model ID, or ``None`` when none is published."""
    if not model_id:
        return None
    wanted = model_id.strip().lower()
    for spec in CATALOGUE:
        if wanted in (m.lower() for m in spec.model_ids):
            return spec
    return None


def supported_models():
    return [model_id for spec in CATALOGUE for model_id in spec.model_ids]


@dataclass
class Transcoders:
    """A whole set, on the model's device, encoding and decoding per layer.

    Weights are kept in float32 unless they were published in a narrower type,
    as a JumpReLU threshold compared in bfloat16 switches features near it on
    and off at random.
    """

    spec: TranscoderSpec
    w_enc: list
    b_enc: list
    w_dec: list
    b_dec: list
    threshold: list
    device: object = None
    revision: str | None = None
    load_id: str = field(default_factory=lambda: uuid4().hex)
    _unembedded: dict = field(default_factory=dict)

    @property
    def layers(self):
        return len(self.w_enc)

    def pre_activations(self, layer, x):
        """Encoder pre-activations for inputs ``x`` (..., d_model), in float32."""
        weights = self.w_enc[layer]
        return (x.to(weights.dtype) @ weights.T).float() + self.b_enc[layer].float()

    def activate(self, layer, pre):
        threshold = self.threshold[layer]
        if threshold is None:
            return pre.clamp(min=0)
        return pre * (pre > threshold.float())

    def activate_one(self, layer, feature, pre):
        threshold = self.threshold[layer]
        if threshold is None:
            return pre.clamp(min=0)
        return pre * (pre > threshold[feature].float())

    def encode(self, layer, x):
        return self.activate(layer, self.pre_activations(layer, x))

    def decode(self, layer, acts):
        weights = self.w_dec[layer]
        return (acts.to(weights.dtype) @ weights).float() + self.b_dec[layer].float()

    def decoder_rows(self, layer, features):
        return self.w_dec[layer][features].float()

    def encoder_rows(self, layer, features):
        return self.w_enc[layer][features].float()


_LOADED = {}
_LOAD_LOCK = threading.Lock()


def downloaded(spec):
    """Whether every layer's weights are already in the Hub cache."""
    from huggingface_hub import try_to_load_from_cache

    for layer in range(spec.layers):
        found = try_to_load_from_cache(spec.repo, spec.path(f"layer_{layer}.safetensors"))
        if not isinstance(found, str):
            return False
    return True


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        from .attribution import Cancelled
        raise Cancelled()


def snapshot_revision(path):
    for parent in Path(path).parents:
        if parent.parent.name == "snapshots" and len(parent.name) == 40:
            if all(c in "0123456789abcdef" for c in parent.name):
                return parent.name
    raise ValueError("The transcoder download did not identify an immutable Hub snapshot.")


def download(spec, progress=None, cancelled=None, revision=None):
    """Fetch every layer's weights into the Hub cache, one file at a time."""
    from huggingface_hub import hf_hub_download

    paths = []
    for layer in range(spec.layers):
        _check_cancelled(cancelled)
        if progress is not None:
            progress(layer, spec.layers)
        path = hf_hub_download(spec.repo, spec.path(f"layer_{layer}.safetensors"), revision=revision)
        _check_cancelled(cancelled)
        resolved = snapshot_revision(path)
        if revision is not None and resolved != revision:
            raise ValueError("The transcoder download returned another snapshot.")
        revision = resolved
        paths.append(path)
    if progress is not None:
        progress(spec.layers, spec.layers)
    return paths


def loaded(spec, device, revision=None):
    """The set already in memory for this device, or ``None``."""
    with _LOAD_LOCK:
        held = _LOADED.get(spec.key)
    return held if (held is not None and str(held.device) == str(device)
                    and (revision is None or held.revision == revision)) else None


def load(spec, device, progress=None, cancelled=None, revision=None):
    """Read the whole set onto ``device``, replacing any other set in memory.

    Only one set is held at a time: the sets are gigabytes each, and the one
    worth keeping is the one for the model that is loaded.
    """
    import torch
    from safetensors.torch import load_file

    _check_cancelled(cancelled)
    held = loaded(spec, device, revision=revision)
    if held is not None:
        return held
    unload()
    paths = download(spec, progress, cancelled, revision=revision)
    parts = {"w_enc": [], "b_enc": [], "w_dec": [], "b_dec": [], "threshold": []}
    tensors, w_enc, w_dec, threshold = {}, None, None, None
    try:
        for path in paths:
            _check_cancelled(cancelled)
            tensors = load_file(path)
            _check_cancelled(cancelled)
            missing = {"W_enc", "b_enc", "W_dec", "b_dec"} - tensors.keys()
            if missing:
                raise ValueError(f"{Path(path).name} has no {', '.join(sorted(missing))}.")
            if "W_skip" in tensors:
                raise ValueError("Transcoders with a skip connection are not supported.")

            def place(tensor):
                _check_cancelled(cancelled)
                dtype = torch.float32 if tensor.dtype in (torch.float32, torch.float64) else tensor.dtype
                return tensor.to(device=device, dtype=dtype)

            w_enc, w_dec = tensors["W_enc"], tensors["W_dec"]
            if w_enc.shape != (spec.width, spec.d_model) or w_dec.shape != (spec.width, spec.d_model):
                raise ValueError(f"{Path(path).name} does not have the shape {spec.title} was published with.")
            parts["w_enc"].append(place(w_enc))
            parts["b_enc"].append(place(tensors["b_enc"]))
            parts["w_dec"].append(place(w_dec))
            parts["b_dec"].append(place(tensors["b_dec"]))
            threshold = tensors.get("activation_function.threshold")
            parts["threshold"].append(None if threshold is None else place(threshold))
            tensors = {}
        _check_cancelled(cancelled)
        held = Transcoders(spec, device=device, revision=snapshot_revision(paths[0]), **parts)
        with _LOAD_LOCK:
            _LOADED[spec.key] = held
        logger.info("Loaded transcoders %s onto %s", spec.key, device)
        return held
    except BaseException:
        # A canceled or failed partial load must not keep its device blocks.
        parts.clear()
        tensors.clear()
        w_enc = w_dec = threshold = None
        gc.collect()
        from chatlab.model_loading import LoadingMixin
        LoadingMixin._release_device_cache()
        raise


def in_memory(spec):
    with _LOAD_LOCK:
        return spec.key in _LOADED


def unload():
    with _LOAD_LOCK:
        had_loaded = bool(_LOADED)
        _LOADED.clear()
    # Drop all cache-owned references before returning allocator blocks.
    gc.collect()
    from chatlab.model_loading import LoadingMixin
    LoadingMixin._release_device_cache()
    return had_loaded


class FeatureRecords:
    """Each feature's examples and top tokens, fetched by byte range and cached.

    ``features/index.json.gz`` lists, per layer, the file holding that layer's
    records and the byte offset of each record in it. A record is a 4-byte
    little-endian length followed by that many bytes of gzipped JSON. Records
    fetched once are kept under ``cache_dir`` so a graph reopened offline still
    shows them.
    """

    def __init__(self, spec, cache_dir, revision=None):
        self.spec = spec
        self._cache_root = Path(cache_dir) / spec.key
        self.revision = revision
        self.cache_dir = self._cache_root / revision if revision else self._cache_root
        self._index = None
        self._lock = threading.Lock()

    def _url(self, name):
        from huggingface_hub import hf_hub_url

        return hf_hub_url(self.spec.repo, self.spec.path(f"features/{name}"), revision=self.revision)

    def _read_index(self):
        from huggingface_hub import hf_hub_download

        with self._lock:
            if self._index is None:
                path = hf_hub_download(self.spec.repo, self.spec.path("features/index.json.gz"), revision=self.revision)
                resolved = snapshot_revision(path)
                if self.revision is not None and self.revision != resolved:
                    raise OSError("The feature index belongs to another transcoder snapshot.")
                self.revision = resolved
                self.cache_dir = self._cache_root / resolved
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    self._index = json.load(handle)
            return self._index

    def get(self, layer, feature):
        """The record for one feature, or raise ``OSError`` when it cannot be read."""
        if self.revision is None:
            self._read_index()
        cached = self.cache_dir / str(layer) / f"{feature}.json"
        if cached.exists():
            try:
                return json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        index = self._read_index()
        entry = index.get(str(layer))
        if not isinstance(entry, dict) or not 0 <= feature < len(entry["offsets"]) - 1:
            raise OSError(f"No record for feature {feature} in layer {layer}.")
        start, end = entry["offsets"][feature], entry["offsets"][feature + 1]
        if end <= start:
            raise OSError(f"Feature {feature} in layer {layer} has no recorded examples.")
        record = parse_record(_fetch_range(self._url(entry["filename"]), start, end))
        record["_transcoder_revision"] = self.revision
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(record), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not cache feature record %s/%s: %s", layer, feature, exc)
        return record


def parse_record(raw):
    if len(raw) < 4:
        raise OSError("A feature record was cut short.")
    (length,) = struct.unpack("<I", raw[:4])
    try:
        return json.loads(gzip.decompress(raw[4:4 + length]))
    except (OSError, ValueError, EOFError) as exc:
        raise OSError("A feature record could not be read.") from exc


def _fetch_range(url, start, end):
    from huggingface_hub import get_session
    from huggingface_hub.utils import build_hf_headers

    headers = build_hf_headers()
    headers["Range"] = f"bytes={start}-{end - 1}"
    try:
        response = get_session().get(url, headers=headers, timeout=30, follow_redirects=True)
    except Exception as exc:
        raise OSError(f"Could not reach the Hub for feature examples: {exc}") from exc
    if response.status_code not in (200, 206):
        raise OSError(f"The Hub answered {response.status_code} for feature examples.")
    data = response.content
    if response.status_code == 200:
        data = data[start:end]
    return data
