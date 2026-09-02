# Chatlab

A local chat interface that shows what happened under the hood for every token, generated or not. Tokens are colored by whichever measurement you pick, and clicking one shows its probability, sampling probability, surprise, entropy, and the alternatives the model preferred.

## What it includes

- Hugging Face model download and cache controls
- A chat interface that collapses OLMo reasoning blocks into an expandable section
- Live token-by-token generation with a **Stop** button
- Exact raw vocabulary rank for each generated token
- Raw and post-sampling probabilities
- Distribution entropy and the top-1 margin behind every token
- Four color scales: raw rank, surprise, entropy, and sampling shift
- Prompt tokens scored in the same pass that warms the cache
- A **Score text** tab for measuring text the model did not write
- Perplexity, mean surprise, and a surprise trace for each response
- Full metric-trace export as JSON or CSV
- A system prompt, plus temperature, top-p, top-k, seed, and response-length controls
- Retry, edit, and undo for any turn, and saving or loading a whole conversation
- Enter sends a message and Shift+Enter starts a new line, with a setting to swap them
- Branching a response from any token into one of the alternatives the model considered
- Forking the conversation so the same transcript can be taken in several directions
- A logit lens showing what every layer would have predicted for a token, and where it was decided
- An attention view showing which earlier tokens the model looked at when predicting it
- Apple Metal, NVIDIA CUDA, and CPU loading

The default model is [`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think). Its full weights require a download of roughly 15 GB. Other Hugging Face causal language models with built-in Transformers support can also work.

## Run it

### As a macOS application

Build the native desktop application on an Apple Silicon Mac:

```bash
./scripts/build_macos_app.sh
```

The finished application is `dist/ChatLab.app`. Open it directly or drag it to
your Applications folder. ChatLab opens in its own native window and stops its
local server when you quit. The app bundle contains Python and its runtime
dependencies, so it does not need a separate Python installation.

The app checks GitHub Releases for a newer version when it starts, and
**Help → Check for Updates…** does the same on demand. Accepting an update
downloads the new bundle, swaps it into place, and relaunches ChatLab.

Model weights are intentionally not included in the application. Models are
downloaded on first use and remain in the standard Hugging Face cache, which
keeps the app bundle manageable and lets terminal and desktop launches reuse the
same downloads.

### In Conductor

Create a workspace for this repository. Its setup script creates the Python environment and installs the dependencies. Use the **Chatlab** action to start the app on the workspace's assigned port.

### From a terminal

On macOS or Linux:

```bash
./run.sh
```

The first run creates an isolated Python environment and installs the dependencies. The app then opens in your browser. Paste a Hugging Face model ID into **Model setup** and choose **Download and load**.

To install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Model files use the standard Hugging Face cache. By default this is under `~/.cache/huggingface`; setting `HF_HOME` before starting the app changes that location. A Hugging Face token is only needed for private or gated models, and the app does not save the token.

## Working with a conversation

- **Stop** cancels the running generation and keeps whatever was produced so far.
- **Retry** regenerates the last reply. Because **🎲 New seed each response** is on by default, a retry actually explores a different sample; turn it off to lock the seed and reproduce a response exactly. The seed field always shows the seed that produced the response on screen.
- Hovering a message in the transcript gives per-message retry, edit, and undo. Editing one of your messages truncates the conversation there and generates a new reply; editing a reply just corrects it in place. **↩️ Undo last** removes the last exchange and puts your message back in the input box.
- **💾 Save conversation** writes a JSON file containing every turn, its reasoning block, and the system prompt. **📂 Load conversation** restores it.

## Branching from a token

Every response token comes with the alternatives the model ranked highest. Branching lets you take one of them instead and see where the model goes from there.

1. Click a token in **Response tokens**.
2. Click a row in **Most likely alternatives**. The detail panel confirms what the branch will do.
3. Press **🌱 Branch from token**.

The response is kept up to the token before the one you clicked, the alternative is put in its place, and the model continues from there under the current sampling settings. The branched response replaces the one on screen, so **Retry** and **Undo** work on it as usual. Choosing the token the model already picked resamples the rest of the response from that point, which is a way to see how much of what followed was chance.

The replayed tokens are still measured against the model's own distribution, so a token the model would never have chosen shows its real rank and surprise. **Maximum new tokens** counts the tokens sampled after the branch point, so a branch made late in a long response still has room to finish. The JSON export records how many tokens were replayed as `forced_prefix_tokens`.

Only a chat response can be branched. Prompt tokens and text measured in the **Score text** tab have no conversation to continue.

## Forking the conversation

**🌿 Fork** copies the conversation into a new fork and switches to it, so you can ask something different without losing the original. The **Conversation fork** dropdown moves between forks, and **Delete fork** removes the one on screen.

Click a message before pressing Fork to fork at that point. Forking at a reply keeps the conversation through that reply, ready for a different next question. Forking at one of your own messages keeps what came before it and puts the message back in the input box so it can be reworded, the same shape **Undo** gives.

Each fork has its own transcript, but the token panel describes only the response on screen: switching forks clears it until the next response. **💾 Save conversation** writes the fork on screen, and **🗑️ Clear** removes every fork.

## Layers and attention

The token panel says how likely a token was. **Layers and attention** says how the model got there.

1. Click a token in **Response tokens** or **Prompt and context tokens**.
2. Open **Layers and attention** and press **🔬 Inspect layers**.

The model is run again over everything before the token, one extra pass. That costs a few seconds on a 7B model with a long context, which is why it is a button rather than something that happens on every click.

**Logit lens.** The residual stream after each layer is read through the model's final norm and unembedding, as though the network had stopped there. The chart traces the probability of the chosen token from the embeddings to the output; the faint line is whatever each layer liked best. The table under it names that preferred token per layer, with the chosen token's rank and the distribution's entropy, and the caption says from which layer the chosen token stayed the first choice. The last row is the model's real output and matches the numbers in the token panel. Readings from early layers are approximate: the lens assumes every layer writes in the same basis the output reads, which is roughly true late in the stack and less so early. The final norm is looked up under the names the common architectures use, on the base model and one level down (OPT keeps it inside its decoder); a model whose norm cannot be found shows only the output row, since readings taken without it would be wrong rather than approximate. Heads that post-process their logits (Gemma's soft-capping, Granite's and Cohere's scaling) are replicated, and the reading of the final layer is checked against the model's real output before any intermediate row is shown; a mismatch also falls back to the output row alone.

**Attention.** The prediction for a token is made at the position *before* it, so that earlier token is the query, drawn with a dashed outline. Every token it could see is shaded by how much attention it received, averaged over heads, and the strongest are listed underneath. The **Attention layer** slider picks one layer or, at 0, the mean of all of them; moving it repaints from the stored readout without another pass. A layer with a sliding window (most of OLMo 3's, and Mistral's) sees only the most recent tokens, so the ones it could not see are shown with no weight. The first token of a sequence almost always takes a large share regardless of content (the attention sink), so shading is scaled to the strongest token after it and the sink's share is stated in words.

Attention weights need the model's eager attention kernel, which is switched on for the inspection step only and switched back afterwards. A model that cannot return them still gets the logit lens. Only the first token of a sequence has nothing to show: nothing came before it.

## Reasoning blocks

Text the model wraps in `<think>` tags is pulled out of the reply and shown as a collapsible **Reasoning** section, so the answer stays readable while the trace stays available.

By default that reasoning is **not** sent back to the model on the next turn. Think models are trained to produce a fresh reasoning block each time, so replaying old ones spends context and tends to degrade the next answer. Enable **Send previous reasoning back to the model** under *System prompt and reasoning* if you want the older behavior.

## Reading the visualization

- **Raw rank** is the generated token's position in the model's unmodified distribution. Rank 1 was the model's first choice.
- **Raw model probability** is calculated before temperature or filtering.
- **Actual sampling probability** includes temperature, top-k, and top-p.
- **Surprise** is `-log2(probability)`. Larger values are less expected.
- **Distribution entropy** is the width of the whole distribution the model chose from, in bits. Surprise says how unexpected the choice was; entropy says how undecided the model was before making it.
- **Top-1 margin** is the probability gap between the model's first and second choice.
- **Sampling shift** is `log2(sampling probability / raw probability)`: how far your temperature, top-k, and top-p settings moved that token away from the raw model.
- **Probability mass above it** is the combined raw probability of every token ranked above the generated token.

**Color tokens by** repaints the strip without regenerating anything. Rank, surprise, and entropy are magnitudes and share one light-to-dark blue ramp; sampling shift is a diverging red-to-blue scale around no change. Quantized model weights can slightly change logits, probabilities, and ranks.

Under each response are its headline numbers — perplexity, mean surprise, the share of tokens the model ranked first, mean entropy — and a trace of surprise across the response, so a stretch where the model lost the thread is visible at a glance. Long responses are grouped into bins, with the range inside each bin shaded.

After a response finishes, open **Export full metric trace** under the conversation and use **Download JSON** or **Download CSV**. JSON preserves the complete trace, including conversation context, generation settings, and nested alternatives. CSV contains one row per generated token, repeats the generation metadata, and expands every recorded alternative into numbered columns.

## Prompt tokens and scoring text

Every prompt token is measured against the distribution the model held one step earlier, during the same pass that fills the key-value cache, so it costs nothing extra to see how predictable your own prompt was. They appear under **Prompt and context tokens**; the first token has nothing before it, so it is left unscored. Turn the measurement off in **Sampling, analysis, and input controls** if you do not want it, and note that only the most recent 1,024 tokens of a very long prompt are scored.

The **Score text** tab measures text the model did not generate. Paste it, optionally give it context first, and one forward pass reports the same numbers for every token — useful for comparing two prompts, checking how memorized a passage is, or evaluating a response that came from somewhere else. Scoring is capped at 4,096 tokens per run.

## Releasing a new version

1. Set the new number in `version.py` and merge it to `main`.
2. Tag that commit and push the tag:

   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```

The `Release macOS app` workflow builds `ChatLab.app` on an Apple Silicon
runner, smoke-tests it, and attaches `ChatLab-macos-arm64.zip` and its
`.sha256` checksum to a GitHub Release for that tag. It fails if the tag
disagrees with `version.py` or the zip exceeds GitHub's 2 GB asset limit.
Installed apps offer the release the next time they start; the updater
verifies the download against the published checksum and confirms the
unpacked bundle is ChatLab at the release's version before installing it.

The app is not code-signed or notarized, so the checksum only protects
against a corrupted or tampered download in transit. Signing releases with a
Developer ID is the step that would let clients verify who built them.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The application deliberately leaves `trust_remote_code` disabled. Models that require executing custom repository code will not load unless their architecture is supported directly by Transformers.
