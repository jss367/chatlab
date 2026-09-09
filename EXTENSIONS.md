# Optional extensions for ChatLab

Extensions add specialized pages while sharing ChatLab's model runtime and token inspection. The first bundled extension is **Maze experiments**. Fresh installations start with all extensions disabled.

## Enable or disable an extension

Open **Settings → Extensions**, check or uncheck **Maze experiments**, and restart ChatLab. The choice saves immediately, survives browser refreshes, and takes effect when the server/app next starts. Refreshing the browser alone does not restart the server. Current pages and running trials remain available until that restart.

When enabled, **Maze** appears in the sidebar. When disabled, its Python module and stylesheet are not loaded, its page and callbacks are not registered, and existing saved trials remain on disk. The rest of ChatLab works without it. Import or API-version failures appear in Settings and do not prevent the core app from starting.

This first version provides **bundled, optional modules**. It does not yet install external packages. The explicit catalogue and versioned service boundary give us a place to add external distribution later, after another extension exercises the interface. Extensions are trusted Python code running in ChatLab's process, not sandboxed programs.

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

A catalogue entry declares an identifier, title, description, sidebar label, icon, module path and required API version. The module exports:

```python
CSS = "..."  # Scope selectors to this extension's page.

def build_page(context):
    # Build Gradio components and register their callbacks here.
    # Called once during app construction, only when enabled.
    ...
```

The host passes `extension_api.ExtensionContext`:

- `models`: the model service; obtain exclusive access with `open_session()`.
- `tokens`: the shared `TokenInspector`, with `color_map`, `strip(metrics)`, `describe(metric)` and `selections()` for dated token selection.
- `data_dir`: an extension-specific directory. Create it only when writing data.
- `navigation`: the host navigation service; call `context.navigation.open_models(button)` during page construction to make a Gradio button open model loading. The host updates the sidebar selection and all page visibility together.
- `api_version`: the version supplied by this host.

### Generating

For an interactive token strip, create one `selections = context.tokens.selections()` controller per view. Store its session ID using `gr.State(value=selections.new_session, delete_callback=selections.forget)`. `selections.view(session_id, response_identity, metrics)` returns a stamped metrics payload and a flag telling the UI to clear its selected-token details when the response changes. Keep the identity stable while appending tokens to that response. Pass the stamped payload to `selections.inspect(session_id, payload, event)` in the strip's selection callback: it discards delayed clicks from replaced responses. Current stamps stay in the controller, outside Gradio's event input snapshots, and are isolated from other browser sessions, extension views and core Chat.

```python
with context.models.open_session() as session:
    prefix_ids = session.encode("A supplied response prefix")
    stream = session.generate(
        messages,
        temperature=0.7, top_p=1.0, top_k=0,
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

Close a generation iterator before closing its session, including on errors or interrupted UI streams. `session.cancel()` may be called from another thread to stop at the next generation update; cancellation remains set for that session. A stopped stream does not execute domain actions. The extension decides which completed responses count as valid actions and which partial results to retain.

### Storage and compatibility

By default, extension data lives under `$XDG_DATA_HOME/chatlab/extensions/<identifier>` (or `~/.local/share/chatlab/extensions/<identifier>`). `CHATLAB_EXTENSIONS_DATA_PATH` overrides the parent directory. Maze also preserves the existing `CHATLAB_MAZE_RUNS_PATH` override. Previously exported `chatlab-maze-run-1` traces import unchanged; legacy files in `~/.local/share/chatlab/maze_runs` are left in place and can be uploaded for replay.

The desktop packaging specification includes the bundled extension modules even though they are imported lazily. Adding a bundled extension means registering its manifest and implementing the contract; it does not require adding domain-specific code to the core UI.

## Checks

The automated suite covers default-disabled startup without importing Maze, preference persistence and restart behavior, independent page registration, import/API/builder failure handling, model ownership, cancellation, stream cleanup and snapshot isolation. The Maze controller tests exercise the shared model service as well as movement, interruption accounting and replay validation.
