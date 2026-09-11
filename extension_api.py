"""Supported services for trusted ChatLab extensions (API version 1).

Extensions own their UI and domain logic. Only this host adapter knows about
ModelManager internals, shared UI helpers, or the application's singleton.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from model_runtime import LOADING
from trace_export import write_private_text

API_VERSION = 1
__all__ = ["API_VERSION", "ExtensionContext", "ModelService", "GenerationSession", "TokenInspector", "TokenSelections", "NavigationService", "write_private_text"]


class ModelService:
    def __init__(self, manager_provider):
        self._provider = manager_provider

    @property
    def loaded(self):
        return self._provider().loaded

    def open_session(self):
        """Reserve the shared model until close; fail rather than queue behind Chat.

        A load turns the session away as a running reply does, and says so in
        its own words: there is no response to wait for while weights are
        being read, and the model the extension checked for is on its way out.

        The slot is claimed before memory is looked at, because a load empties
        it before it reads the new weights: an extension asking in that window
        would be told to load a model on the Models page, which is the page
        already loading one. The claim is also what keeps the answer good -
        no load can start while it is held, so the model the session pins
        cannot be unloaded between the check and the first token.
        """
        manager = self._provider()
        held = manager.claim_generation()
        if held == LOADING:
            raise ValueError("A model is loading. Wait for it to finish, then try again.")
        if held is not None:
            raise ValueError("The model is busy in another view. Wait for that response to finish.")
        try:
            if not manager.loaded:
                raise ValueError("Load a model on the Models page before running an extension.")
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

    def encode_replacement(self, kept_ids, text, *, literal_prefill_tokens=0):
        """Encode typed text after exact retained IDs using this pinned model."""
        self._check()
        return self._manager.encode_replacement(
            kept_ids, text, literal_prefill_tokens=literal_prefill_tokens, load_id=self.load_id,
        )

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

    def selections(self):
        """Create an independent selection controller for one extension view."""
        return TokenSelections(self)


class TokenSelections:
    """Date token snapshots against live, per-session server state.

    Store new_session as a gr.State callable and forget as its delete_callback.
    Only the stable session ID and stamped metrics travel through Gradio inputs;
    the current view/stamp remains here, outside event input snapshots.
    """
    def __init__(self, inspector):
        self._inspector = inspector
        self._sessions = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session():
        return uuid4().hex

    def forget(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)

    def view(self, session_id, view_id, metrics):
        """Stamp a strip; changed is true when its detail panel must be cleared."""
        with self._lock:
            previous = self._sessions.get(session_id)
            changed = previous is None or previous[0] != view_id
            stamp = uuid4().hex if changed else previous[1]
            self._sessions[session_id] = (view_id, stamp)
        return (stamp, metrics), changed

    def inspect(self, session_id, payload, event):
        import gradio as gr
        stamp, metrics = payload
        def current():
            with self._lock:
                active = self._sessions.get(session_id)
                return active is not None and active[1] == stamp
        if not current():
            return gr.skip(), gr.skip()
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        if not isinstance(index, int) or not 0 <= index < len(metrics):
            return "Select a token in the current response.", []
        result = self._inspector.describe(metrics[index])
        # Formatting may overlap a stream update or replay switch.
        return result if current() else (gr.skip(), gr.skip())

    def resolve(self, session_id, payload, index):
        """Resolve an actionable selection only while its view is current."""
        stamp, metrics = payload
        with self._lock:
            active = self._sessions.get(session_id)
            if (active is None or active[1] != stamp or not isinstance(index, int)
                    or not 0 <= index < len(metrics)):
                raise ValueError("Select a token in the current response again.")
            return active[0], index, copy.deepcopy(metrics[index])


class NavigationService:
    """Register navigation actions while building an extension's page.

    The host wires both its navigation selection and page visibility after all
    pages exist. Extensions never need references to the host's UI components.
    """
    def __init__(self, register_models_button):
        self._register_models_button = register_models_button

    def open_models(self, button):
        """Make this button open model loading when clicked."""
        self._register_models_button(button)


@dataclass(frozen=True)
class ExtensionContext:
    models: ModelService
    tokens: TokenInspector
    data_dir: Path
    navigation: NavigationService
    api_version: int = API_VERSION
