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
from ui.icons import icon_classes

API_VERSION = 1
__all__ = ["API_VERSION", "ExtensionContext", "ModelService", "GenerationSession", "TokenInspector", "TokenMenu", "TokenSelections", "NavigationService", "write_private_text", "icon_classes"]


class ModelService:
    def __init__(self, manager_provider):
        self._provider = manager_provider

    @property
    def loaded(self):
        return self._provider().loaded

    def decode(self, ids):
        """Read recorded token IDs back as text without reserving the model.

        A view showing what a model was given must not queue behind the
        response it is showing, nor keep a load from starting, so this takes
        no claim. What it reads is framed by the load instead, as :meth:`_read`
        describes. Returns the text with the load that spelled it, or
        ``(None, None)`` when nothing is loaded or a load moved underneath the
        reading.
        """
        return self._read(lambda manager, tokenizer: tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        ))

    def prompt_text(self, messages, tools=None):
        """The prompt these messages would become, as the loaded model would be given it.

        Built through the same template path generation uses, so a reader
        seeing it here is seeing the tool schemas, turn markers and generation
        prompt that would actually be sent, rather than a description of them.
        Returned with its load, or ``(None, None)`` as :meth:`decode`.
        """
        def render(manager, tokenizer):
            ids, _ = manager._prompt_token_ids(messages, tools)
            return tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
        return self._read(render)

    def _read(self, reading):
        """Read the loaded model, or answer that no one load made the reading.

        A load publishes its weights field by field and the snapshot naming
        them last, so ``loaded`` turns true while the identifiers still name
        the load before it - a window wide enough on MLX to decode inside,
        because the engine is built in it. The snapshot is the reading that
        moves as one, and an unload empties it before a load begins, so a read
        framed by two equal snapshots that name a load was made under that
        load and no other: an unfinished load is nameless, and a load count
        only rises, so no cycle of loads can put the frame back the way it
        was. Returns the text with that load, or ``(None, None)``.
        """
        manager = self._provider()
        published = manager.loaded_model()
        tokenizer = manager.tokenizer
        if published.load_id is None or tokenizer is None:
            return None, None
        text = reading(manager, tokenizer)
        return (text, published.load_id) if manager.loaded_model() == published else (None, None)

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
        """Decode exactly as the runtime does when it records per-token text.

        clean_up_tokenization_spaces is not left to the tokenizer's own default,
        which some repositories set true: that rewrites spacing around
        punctuation, so the same IDs would decode one way into a recorded metric
        and another way here, and a caller comparing the two would see a
        difference the vocabulary does not have.
        """
        self._check()
        return self._manager.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )

    @property
    def stop_token_ids(self):
        self._check()
        return set(self._manager._stop_token_ids())

    @property
    def hidden_token_ids(self):
        """Special token IDs that never reach a recorded response text.

        The streaming decoder drops these rather than decoding them, so a
        caller checking recorded text against a fresh decode of the same IDs
        has to leave out exactly this set. Reader-supplied prefill is the
        exception the runtime makes: replay forces those tokens visible,
        special-token spellings included.
        """
        self._check()
        return set(self._manager.hidden_token_ids())

    def generate(self, messages, *, temperature, top_p, top_k, max_new_tokens, seed,
                 skip_top_below=0.0, tools=None, forced_ids=(),
                 literal_prefill_tokens=0, analyze_prompt=False):
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
                skip_top_below=skip_top_below, max_new_tokens=max_new_tokens,
                seed=seed, tools=tools,
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

    def menu(self, strip_id):
        """The right-click token menu for the strip with this element id."""
        return TokenMenu(strip_id)


class TokenMenu:
    """ChatLab's right-click token menu, attached to one extension strip.

    The script the host loads is shared: it opens on any strip built with
    ``strip_classes`` and talks to the three hidden components ``bridges()``
    returns, which are named after that strip. What the menu offers is the
    extension's to say. Answer the strip's select event with ``offer`` or
    ``refuse``, passing whatever identifies the token in your own view; the
    same value comes back in the action the reader clicks, alongside its
    ``kind`` - ``"candidate"`` with an ``index`` into the alternatives you
    offered, ``"text"`` with the reader's own text, or one of your ``actions``.
    A selection travelling through the browser is a claim, not a fact: resolve
    it again before acting on it, exactly as with a click.
    """

    def __init__(self, strip_id):
        self._strip_id = strip_id

    @property
    def strip_classes(self):
        from ui.token_menu import MENU_STRIP_CLASS
        return [MENU_STRIP_CLASS]

    def bridges(self):
        """Create the request, response and action components, in that order.

        The request holds the identifier of the right-click being answered,
        the response the answer, and the action what the reader chose. Read
        the request in the strip's select callback and act on the action's
        ``input`` event. All three are hidden.
        """
        import gradio as gr
        from ui.token_menu import MENU_BRIDGE_CLASS, menu_bridge_ids
        request, response, action = menu_bridge_ids(self._strip_id)
        return (gr.Textbox(elem_id=request, elem_classes=[MENU_BRIDGE_CLASS]),
                gr.HTML(elem_id=response, elem_classes=[MENU_BRIDGE_CLASS]),
                gr.Textbox(elem_id=action, elem_classes=[MENU_BRIDGE_CLASS]))

    def offer(self, request_id, selection, *, text, candidates, verb, label, submit, actions=()):
        from ui.token_menu import menu_markup
        return menu_markup({
            "request": request_id, "selection": selection, "error": "", "text": text,
            "candidates": [{"token_id": c["token_id"], "text": c["text"],
                            "probability": c["probability"]} for c in candidates],
            "verb": verb, "label": label, "submit": submit, "actions": list(actions),
        })

    def refuse(self, request_id, message):
        """Open the menu on a message instead of a branch."""
        from ui.token_menu import menu_markup
        return menu_markup({"request": request_id, "selection": None, "error": message})


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

    def open_models(self, button, model_id=None):
        """Make this button open model loading when clicked.

        ``model_id`` is an optional component holding the ID of the model the
        extension wants. When it holds one at the click, Models opens with
        that ID already in its box; when it is empty, or is not a model ID at
        all, the page opens with the box as the reader left it. Filling the
        box is the whole of it: reading weights is still an explicit click on
        the Models page, whoever named the model.
        """
        self._register_models_button(button, model_id)


@dataclass(frozen=True)
class ExtensionContext:
    models: ModelService
    tokens: TokenInspector
    data_dir: Path
    navigation: NavigationService
    api_version: int = API_VERSION
