# Optional extensions for ChatLab

Extensions add specialized pages while sharing ChatLab's model runtime and token inspection. Bundled extensions include **Maze experiments**, **OS-Harm results**, **Computer-use safety benchmark** and **Hangman**. Fresh installations start with all extensions disabled.

## Enable or disable an extension

Open **Settings → Extensions**, check or uncheck an extension, and restart ChatLab. The choice saves immediately, survives browser refreshes, and takes effect when the server/app next starts. Refreshing the browser alone does not restart the server. Current pages and running trials remain available until that restart.

In the macOS app, **Restart ChatLab** appears beside that note while the saved choice differs from the pages on screen. It asks first, because restarting unloads the model and stops anything running; answering **Restart now** closes the window and opens a fresh copy. A ChatLab served to a browser by `python app.py` has no window to reopen, so it shows the note without the button and the server is restarted by hand.

When enabled, **Maze**, **OS-Harm**, **Safety** or **Hangman** appears in the sidebar. When disabled, an extension's Python module and stylesheet are not loaded, its page and callbacks are not registered, and existing saved results remain on disk. The rest of ChatLab works without it. Import or API-version failures appear in Settings and do not prevent the core app from starting.

This version provides **bundled, optional modules**. It does not yet install external packages. The explicit catalogue and versioned service boundary give us a place to add external distribution later. Extensions are trusted Python code running in ChatLab's process, not sandboxed programs.

For importing computer-use safety evaluations, comparing recorded judgments,
and replaying screenshots in **OS-Harm**, see the [results viewer guide](OS_HARM_RESULTS.md).

The **Computer-use safety benchmark** extension adds **Safety** to the sidebar. It implements case imports, text-only local action evaluation by label probability or free-text judgment, the blocking threshold curve, token inspection, external prediction scoring and execution-result review for OSGuard. Its interchange format and limitations are documented in [the computer-use safety guide](COMPUTER_USE_SAFETY.md). Its code lives in `extensions/osguard/` and uses the same host services as Maze.

The **Hangman** extension has the loaded model host a game of hangman. You guess, and each reply is shown token by token and checked against the replies before it. Any token can be branched from. See [the hangman guide](HANGMAN.md). Its code lives in `extensions/hangman/`.

## Ownership

| ChatLab owns | Each extension owns |
| --- | --- |
| Model loading and exclusive generation access | Domain rules and authoritative state |
| Tokenization, tool templates, exact prefix insertion and output streaming | Prompts, tool schemas and action parsing |
| Token probabilities and inspection components | Interruption schedules and outcome scoring |
| Cancellation and model-session lifecycle | Page layout, controls, run format and replay |
| Navigation, enable preferences and module loading | Data written beneath its assigned data directory |

ChatLab’s HTTP API continues to serve outside clients. In-process extensions use `extension_api.py` to access the loaded model directly; both interfaces respect the same model-generation reservation.

The Maze implementation lives entirely in `extensions/maze_experiments/`. Core layout code knows only the extension contract. The catalogue in `extensions/registry.py` contains its descriptive metadata and import path; it does not import the implementation until enabled.

## Extension contract: API version 1

A catalogue entry declares an identifier, title, description, sidebar label, icon, module path and required API version. Extension tiles sit below Chat, Images and Models and above Settings, under a hairline that separates them from the pages that ship with the app, so enabling or removing one never moves a built-in page. The icon is a name in `ui/icons.py`, not a glyph: the host masks that drawing onto the extension's sidebar tile so it is stroked at the same weight as the pages around it, and a name this build does not have falls back to the default. `extension_api.icon_classes(name)` gives the same treatment to an extension's own buttons. The module exports:

```python
CSS = "..."  # Scope selectors to this extension's page.

def build_page(context):
    # Build Gradio components and register their callbacks here.
    # Called once during app construction, only when enabled.
    ...
```

The host passes `extension_api.ExtensionContext`:

- `models`: the model service; obtain exclusive access with `open_session()`, or read the loaded model without claiming it with `decode(ids)`, `prompt_text(messages, tools)` and `loaded_model_id()`.
- `tokens`: the shared `TokenInspector`, with `color_map`, `strip(metrics)`, `describe(metric)`, `selections()` for dated token selection and `menu(strip_id)` for the right-click token menu.
- `data_dir`: an extension-specific directory. Create it only when writing data.
- `navigation`: the host navigation service; call `context.navigation.open_models(button)` during page construction to make a Gradio button open model loading. The host updates the sidebar selection and all page visibility together. Pass a second component holding a model ID - `open_models(button, wanted_model)` - and the Models page also opens with that ID in its box, which is how an extension points at the model its own view needs. The host validates the ID and opens the page with the box untouched when it is empty or is not a model ID. Loading is still an explicit click on the Models page.
- `api_version`: the version supplied by this host.

### Generating

For an interactive token strip, create one `selections = context.tokens.selections()` controller per view. Store its session ID using `gr.State(value=selections.new_session, delete_callback=selections.forget)`. `selections.view(session_id, response_identity, metrics)` returns a stamped metrics payload and a flag telling the UI to clear its selected-token details when the response changes. Keep the identity stable while appending tokens to that response. Pass the stamped payload to `selections.inspect(session_id, payload, event)` in the strip's selection callback: it discards delayed clicks from replaced responses. Current stamps stay in the controller, outside Gradio's event input snapshots, and are isolated from other browser sessions, extension views and core Chat.

For actions such as token editing, `selections.resolve(session_id, payload, index)` returns the current response identity, token index, and a copied metric. It raises `ValueError` for stale selections or invalid indices. Resolve again when applying an edit and verify that the response belongs to the episode being changed.

### The right-click token menu

`context.tokens.menu(strip_id)` attaches ChatLab's token context menu to the strip with that element id. Build the strip with `elem_classes=menu.strip_classes`, and create the three hidden components the menu talks through with `menu.bridges()`, which returns the request, response and action components in that order. Answer the strip's `select` event with `menu.offer(request_id, selection, text=..., candidates=..., verb=..., label=..., submit=..., actions=())`, taking the request component as an input, or with `menu.refuse(request_id, message)` when the token has nothing to offer. Candidates are metric `top_candidates` entries; the menu shows each one's text and probability. `selection` is the extension's own: whatever identifies the token in your view comes back untouched in the action.

Act on the action component's `input` event. The action names a `kind`: `"candidate"` with an `index` into the alternatives offered, `"text"` with the reader's own text, or one of the extra `actions` named in the offer. What the browser sends back is a claim rather than a fact, so resolve the selection through `selections.resolve` again and check the token identity before acting on it, exactly as for a click. Naming an alternative by its position in the menu, and reading its token ID from the resolved metric, keeps that check in one place.

```python
with context.models.open_session() as session:
    prefix_ids = session.encode("A supplied response prefix")
    stream = session.generate(
        messages,
        temperature=0.7, top_p=1.0, top_k=0, skip_top_below=0.0,
        max_new_tokens=1024, seed=42,
        tools=tool_schemas,
        forced_ids=prefix_ids,
        literal_prefill_tokens=len(prefix_ids),
    )
    try:
        for update in stream:
            # update.text, update.metrics, update.prompt_ids,
            # update.forced_prefix_tokens, update.reasoning_prefilled,
            # update.model_id and update.load_id retain ChatLab's semantics.
            ...
    finally:
        stream.close()
```

`open_session()` fails clearly if no model is loaded or another page owns generation. The session pins the model/load identifiers and retains ownership across multiple responses until closed. Token encoding/decoding and `stop_token_ids` are exposed on the session; extensions do not access the model manager or tokenizer directly. Each streamed update owns its token-metric lists.

To show what a model was given rather than to generate, `context.models.decode(ids)` and `context.models.prompt_text(messages, tools)` read the loaded model without reserving it, so a view of a prompt never queues behind the response it is describing or holds up a load. `prompt_text` renders through the same template path generation uses, tool schemas and generation prompt included. Both return the text with the load identifier that spelled it, or `(None, None)` when no model is loaded or a load landed while they were reading, which is how a caller tells a reading made under the recording load from one made under a later one. `context.models.loaded_model_id()` answers what is in memory as it is asked, and frames no text: use it to say what a reader would unload beside text no load produced, never to explain text a load did produce, because the load that answered an earlier reading may already be gone. It is `None` while a load is under way, as a nameless load is no answer, and `None` for an image pipeline, which is published under its own ID like any load and has no tokenizer to spell anything with.

To steer a response, pass a `chatlab-steering-1` vector as `session.generate(..., steering=vector)`; the runtime adds it at the vector's layer for that response alone, prompt included, and removes it before the stream ends. `extension_api.read_steering_vector(path)` reads a vector file the way Chat imports one, and `normalize_steering(value)` validates one read from anywhere else; both raise `SteeringError`, a `ValueError`. A run that only turns steering on partway through should call `session.check_steering(vector)` before its first response: it asks the questions `generate` would - a PyTorch load, the vector's model, its layer and its width - without installing anything.

For typed token replacements, use `session.encode_replacement(kept_ids, text, literal_prefill_tokens=0)` to encode in the retained tokens' decoder context. It preserves the retained IDs and checks that the visible continuation matches the text exactly, including boundary spaces; pass the original literal-prefill count when retaining supplied text.

Close a generation iterator before closing its session, including on errors or interrupted UI streams. `session.cancel()` may be called from another thread to stop at the next generation update; cancellation remains set for that session. A stopped stream does not execute domain actions. The extension decides which completed responses count as valid actions and which partial results to retain.

### Storage and compatibility

By default, extension data lives under `$XDG_DATA_HOME/chatlab/extensions/<identifier>` (or `~/.local/share/chatlab/extensions/<identifier>`). `CHATLAB_EXTENSIONS_DATA_PATH` overrides the parent directory. Maze also preserves the existing `CHATLAB_MAZE_RUNS_PATH` override. Previously exported `chatlab-maze-run-1` traces import unchanged; legacy files in `~/.local/share/chatlab/maze_runs` are left in place and can be uploaded for replay. A run whose map changes while it is running is written as `chatlab-maze-run-2` and uploads alongside them. A team run, several agents in one maze, is written as `chatlab-maze-team-1` and loads in the Maze page's **Team** tab.

The desktop packaging specification includes the bundled extension modules even though they are imported lazily. Adding a bundled extension means registering its manifest and implementing the contract; it does not require adding domain-specific code to the core UI.

## Checks

The automated suite covers default-disabled startup without importing Maze, preference persistence and restart behavior, independent page registration, import/API/builder failure handling, model ownership, cancellation, stream cleanup and snapshot isolation. The Maze controller tests exercise the shared model service as well as movement, interruption accounting and replay validation.
