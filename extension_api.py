"""Supported services for trusted ChatLab extensions (API version 1).

Extensions own their UI and domain logic. Only this host adapter knows about
ModelManager internals, shared UI helpers, or the application's singleton.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path

from trace_export import write_private_text

API_VERSION = 1
__all__ = ["API_VERSION", "ExtensionContext", "ModelService", "GenerationSession", "TokenInspector", "write_private_text"]


class ModelService:
    def __init__(self, manager_provider):
        self._provider = manager_provider

    @property
    def loaded(self):
        return self._provider().loaded

    def open_session(self):
        """Reserve the shared model until close; fail rather than queue behind Chat."""
        manager = self._provider()
        if not manager.loaded:
            raise ValueError("Load a model on the Models page before running an extension.")
        if not manager.reserve_generation():
            raise ValueError("The model is busy in another view. Wait for that response to finish.")
        try:
            return GenerationSession(manager)
        except BaseException:
            manager.release_generation()
            raise


class GenerationSession:
    """A pinned model lease. Use on one worker, close every stream, then close the lease.

    cancel() may be called by another thread; it stops at the next generation
    update. Closing a stream never executes domain actions. Returned updates
    own their metric lists, so later tokens cannot mutate an earlier frame.
    """
    def __init__(self, manager):
        self._manager = manager
        self.model_id, self.load_id = manager.model_id, manager.load_id
        self._closed = False
        self._generating = False
        self._cancelled = threading.Event()

    def _check(self):
        if self._closed:
            raise ValueError("The model session is closed.")
        if self._manager.load_id != self.load_id:
            raise ValueError("The loaded model changed. Start a new episode.")

    def encode(self, text):
        self._check()
        return list(self._manager.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids):
        self._check()
        return self._manager.tokenizer.decode(ids, skip_special_tokens=False)

    @property
    def stop_token_ids(self):
        self._check()
        return set(self._manager._stop_token_ids())

    def generate(self, messages, *, temperature, top_p, top_k, max_new_tokens, seed,
                 tools=None, forced_ids=(), literal_prefill_tokens=0, analyze_prompt=False):
        self._check()
        if self._generating:
            raise ValueError("This model session is already streaming.")
        if self._cancelled.is_set():
            return
        self._generating = True
        generator = None
        try:
            generator = self._manager.generate(
                messages, temperature=temperature, top_p=top_p, top_k=top_k,
                max_new_tokens=max_new_tokens, seed=seed, tools=tools,
                forced_ids=forced_ids, literal_prefill_tokens=literal_prefill_tokens,
                analyze_prompt=analyze_prompt, load_id=self.load_id,
            )
            for update in generator:
                if self._cancelled.is_set():
                    break
                yield copy.deepcopy(update)
                if self._cancelled.is_set():
                    break
        finally:
            if generator is not None:
                generator.close()
            self._generating = False

    def cancel(self):
        self._cancelled.set()

    def close(self):
        if self._generating:
            raise ValueError("Close the generation iterator before releasing its model session.")
        if not self._closed:
            self._closed = True
            self._manager.release_generation()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()


class TokenInspector:
    """ChatLab's shared token rendering, without an extension importing UI internals."""
    @property
    def color_map(self):
        from token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE
        return dict(COLOR_SCALES[DEFAULT_COLOR_SCALE].color_map)

    def strip(self, metrics):
        from token_metrics import DEFAULT_COLOR_SCALE
        from ui.panel import strip_value
        return strip_value(metrics, DEFAULT_COLOR_SCALE)

    def describe(self, metric):
        from ui.panel import describe_token
        return describe_token(metric)


@dataclass(frozen=True)
class ExtensionContext:
    models: ModelService
    tokens: TokenInspector
    data_dir: Path
    navigation: object  # The host's Gradio navigation component; emit "Models" to open it.
    api_version: int = API_VERSION
