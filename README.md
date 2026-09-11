# ChatLab

A local interface for models that shows what happened under the hood while they
worked. For a language model that means every token, generated or not: tokens
are colored by whichever measurement you pick, and clicking one shows its
probability, sampling probability, surprise, entropy, and the alternatives the
model preferred. For a diffusion model it means every denoising step: the
trajectory to scrub through, how hard the prompt pulled against what the model
would have drawn from noise alone, and which pixels each word of the prompt
drove.

## What it includes

- Hugging Face model download and cache controls
- A Models page listing the models already downloaded, with recommended starters and browsing of popular, trending, and new Hugging Face models
- Every model in both lists marked *fits*, *tight* or *won't fit* against the memory this machine has free, at the weight precision chosen
- A badge above the chat naming the model in memory and the device it runs on, or saying that none is loaded
- A chat interface that collapses OLMo reasoning blocks into an expandable section
- Live token-by-token generation with a **Stop** button
- Exact raw vocabulary rank for each generated token
- Raw and post-sampling probabilities
- Distribution entropy and the top-1 margin behind every token
- Four color scales: raw rank, surprise, entropy, and sampling shift
- Prompt tokens scored in the same pass that warms the cache
- A **Score text** tab for measuring text the model did not write
- A **Prompts** tab that runs a list of prompts, each in a conversation of its own, and writes one trace per prompt plus a table of every token
- Perplexity, mean surprise, and a surprise trace for each response
- Full metric-trace export as JSON or CSV
- An OpenAI-compatible HTTP API on the same port, so the measurements can be scripted
- A system prompt, plus temperature, top-p, top-k, seed, and response-length controls
- Every setting saved to one JSON file you can edit by hand or share between machines
- Temperature, top-p, top-k and response length kept per conversation, so two forks can be compared at different settings
- Per-conversation activation steering: import a vector, choose its layer and strength, and compare forks with steering enabled or disabled
- Optional assistant prefill text that the model must continue from
- Retry, edit, and undo for any turn, and saving or loading a whole conversation
- A conversations pane listing every chat, tagged with the model that answered and the conversation's size in tokens
- A draggable seam between the transcript and the panel beside it, remembered between sessions
- Every conversation kept between sessions in one JSON file, so a reload or a restart brings the pane back as it was
- Enter sends a message and Shift+Enter starts a new line, with a setting to swap them, and Escape stops a response, or a run of prompts, from anywhere on the Chat page
- Right-click a token to regenerate from it, choose an alternative, or type a custom replacement and continue the response
- Branching a response from any token into one of the alternatives the model considered, or into text you type yourself
- Forking the conversation so the same transcript can be taken in several directions, and starting new ones beside it
- A logit lens showing what every layer would have predicted for a token, and where it was decided
- An attention view showing which earlier tokens the model looked at when predicting it
- A hardware panel naming the device, the memory ChatLab judges a load against, the Metal cap, and what the process is holding
- Apple Metal, NVIDIA CUDA, and CPU loading
- 8-bit and 4-bit weights on Apple Metal, so a 7B model fits a 16 GB Mac
- MLX-quantized models from `mlx-community` on Apple silicon, run through mlx-lm with every token measurement, the logit lens and the attention view intact
- An **Images** page that draws with a diffusion model and reads the drawing back: one frame per denoising step, the guidance pull and the latent movement per step, and a cross-attention map per prompt token

The default model is [`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think). Its full weights require a download of roughly 15 GB. Other Hugging Face causal language models with built-in Transformers support can also work, and so can diffusers text-to-image pipelines; see [Images](#images).

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

Create a workspace for this repository. Its setup script creates the Python environment and installs the dependencies. Use the **ChatLab** action to start the app on the workspace's assigned port.

### From a terminal

On macOS or Linux:

```bash
./run.sh
```

The first run creates an isolated Python environment and installs the dependencies. The app then opens in your browser. Open **Models** in the pane at the far left, paste a Hugging Face model ID into the **Model** box and choose **Download and load**, or pick one from **My Models** or **Discover models** there.

To install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Model files use the standard Hugging Face cache. ChatLab lists complete downloads and resumable partial downloads under **My Models** on the Models page. By default the cache is under `~/.cache/huggingface`; setting `HF_HOME` before starting the app changes that location. A Hugging Face token is only needed for private or gated models, and the app does not save the token.

On Apple silicon the requirements also bring in [mlx-lm](https://github.com/ml-explore/mlx-lm), which is what runs the MLX models described next; it is skipped everywhere else, and the rest of the app does not need it.

### MLX models

A repository quantized with `mlx_lm.convert`, which is what `mlx-community` publishes, keeps its weights in safetensors files under the names Transformers uses, but packed the way MLX packs them: a 4-bit matrix plus per-group scales and biases per linear layer, which `AutoModelForCausalLM` cannot read. ChatLab recognises one by the `quantization` block in its `config.json`, lists it under **My Models** marked *MLX*, and loads it through mlx-lm rather than through Transformers. It runs on the GPU through Metal at the width it was converted to, so a 7B model at four bits takes about 4 GB and the **Weight precision** radio does not apply; the badge above the chat says *Apple Metal (MLX), 4-bit weights*.

Everything the Chat page measures survives the change of runtime. The model returns logits for every position, so the ranks, probabilities, surprise, entropy, alternatives and branching are read exactly as they are from a Transformers model, and **Score text** and **Prompts** work the same way. The logit lens records the residual stream between the decoder layers for the one step an inspection takes, reads each state through the model's final norm and output head, and checks that the last state read that way reproduces the model's own output before it trusts the intermediate rows, as it does for Transformers. The attention view recomputes the weights beside the fused attention kernel for the inspected query, so it shows one row per layer wherever the architecture attends the ordinary way; a model whose layers do not (state-space layers, quantized caches) shows the lens alone and says so. An unquantized MLX conversion (`-bf16`, `-fp16`) is a Transformers checkpoint under another name and loads as one.

The OpenAI-compatible API lists MLX models beside the Transformers ones and answers from them the same way. GGUF files are not loaded: llama.cpp exposes no hidden states, so the logit lens could not be read, and Transformers can only dequantize them into full weights, which would lose the memory the packing buys.

### Memory

A model has to fit in memory with room to spare: the weights, the key-value
cache that grows with every token of a conversation, the app itself, and the
rest of the system all share it, and on Apple silicon the GPU draws from the
same pool. Before reading any weights, ChatLab estimates the loaded size from
the checkpoint and requires a 4 GB safety reserve beside the weights. Every
estimate names the precision it was measured at - full weights at the device's
own dtype, or the 4- or 8-bit width a quantized load would pack the linear
layers into - because the same checkpoint is several times smaller at four
bits, and because a quantized choice is honoured on Apple Metal alone, so a
refusal on a graphics card is about full weights however the radio is set. An
MLX model is named at the width it was converted to rather than the width the
radio asks for, for the same reason the radio does not apply to it.
On macOS, the available-memory estimate includes reclaimable file cache when
the system reports normal memory pressure. When pressure is elevated or cannot
be read, it uses a stricter estimate: free, speculative and purgeable pages,
plus only the part of the inactive queue known to be file-backed. Anonymous
and compressed memory are not counted as available. Existing swap usage alone
does not block a load. These are estimates intended to reduce heavy paging,
not guarantees that a load will fit; a refusal states ChatLab's safety budget
rather than claiming the machine has run out of free RAM.
On CUDA the weights fill the graphics cards first and the rest is placed in
the machine's own memory, so that check is made against the two together. On
Apple Metal it also caps what PyTorch may allocate at half the machine's
memory, so a conversation that outgrows that half ends with an out-of-memory
message rather than a frozen Mac. Half rather than everything available
because Metal's own recommendation is most of the machine — 37.4 GB of a 48 GB
Mac — and a process that size leaves the window server, the browser and the
editor paging to disk. Move that cap with `mps_memory_fraction` in the
settings file, or with `CHATLAB_MPS_MEMORY_FRACTION` for one run; either names
a fraction of Metal's recommendation, so on that 48 GB Mac the default works
out to about `0.64` and `1.0` restores the old ceiling. The environment wins
over the file, and PyTorch's own `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, if set,
leaves the allocator alone. The cache a response used is handed back when it
finishes, so the process returns to the model's own size between requests.

Each load and each response is recorded in the log with the model, the
weight precision, the estimate, what the device ended up holding and the
estimated memory available beforehand, which is what makes a memory failure
readable after the fact.

When the weights will not fit, **Weight precision** on the Models page is the
first thing to try. At 4 bits the default 7B model's linear layers shrink from
about 14 GB to under 4 GB, and the whole model, embeddings included, to
roughly 5 GB, which is what lets it run on a 16 GB Mac. The refusal above is
made against the quantized size, so a checkpoint the machine could not hold
whole is let through when the quantized weights fit.

What a conversation costs is mostly its key-value cache, which grows with
every token and is held for as long as the answer runs. How fast it grows is
the model's own doing: an architecture with grouped-query attention, which
most current models have, spends a fraction of what one with a key-value head
per query head does at the same size. Lowering **Context limit (tokens)** on
the Settings page is the direct way to bound it.

## The pages

A pane at the far left switches between four pages, each tile an icon above the page's name. **Chat** is the conversation, with the conversations pane beside it and the inspector to its right: whatever token you last clicked, and what the model was doing when it produced it. **Images** is the same shape with a picture where the transcript goes. **Models** is everything about which model is running. **Settings**, at the bottom of the pane, is how every reply is prompted and measured.

The seam between a transcript and the panel beside it is a handle: drag it to give either one the other's room, or use the arrow keys once it has focus. Double-click it to go back to the width it started at. Each page remembers its own width between sessions. In a window under 850px wide the panel sits under the transcript instead, and the handle goes away with the seam it sat on.

One model is in memory at a time whichever kind it is, because the two share the device and, on Apple silicon, the machine's memory. So loading an image model unloads a text one and the other way round, and each page's badge says whether what is in memory is a model it can use: a model of the other kind is named and greyed rather than reported as nothing loaded, which would send you off to load a second one on top of it.

A badge above the tabs names the model that would answer. Beside it, a dropdown lists the downloaded text models that would load on this machine at the chosen weight precision, MLX conversions among them, each at the width it was converted to; picking one loads it in place of the model in memory, and the badge follows the load. A pick during a reply is refused. Models that are still downloading, would not fit, or are not yet on disk are not offered: those go through the Models page. Until a model is loaded, **Set up the default model** opens Models with the default selected. Selecting the default does not start a download or replace a loaded model. On Models, choose **Load cached** to use local files without a network check, or **Download and load** to fetch and load the model. A full default-model download is about 15 GB; the setup guidance states this before you start. Progress and any load errors appear on the Models page.

### Models

- **Currently loaded** names the model in memory and holds **Unload**. **Choose a model** holds the Hugging Face model ID and **Check model**: pressing Enter or clicking the button checks repository metadata and the small configuration file without downloading weights. Configuration previews require a known download size of at most 1 MiB. The preview distinguishes a found repository, a missing or private one, access restrictions, and connection failures. A found repository shows its format and total file size when available; existence does not guarantee loading compatibility. Editing the ID or access token withdraws the previous result; an older request cannot verify the new selection. The optional token is under **Access token**. The local-file status and download/load buttons sit below, while progress and past results are labelled **Latest model action**. **Full (16-bit)** loads the checkpoint as it is. **8-bit** and **4-bit** quantize the linear layers on the way in, on Apple Metal only: the weights take about a half or a quarter of the memory, generation runs on fused Metal kernels fetched from the Hub the first time, and the embeddings and output head are left in half precision so the logit lens still reads through the real head. Accuracy drops a little, most at 4 bits; the token measurements describe the quantized model, which is the one answering. Another device loads full weights whatever is chosen, and says so in the log. The choice is saved and applies to the next load. A checkpoint whose configuration confirms MLX quantization hides the precision choice: its weights were packed when the repo was converted, and it loads at that width, which the badge names. Cached checkpoints follow the same rule while offline, using their local configuration. The card follows a download file by file and byte by byte, and then the load in the same shape: how many of the weights have been read, how much of the model is on the device, the speed, and how long is left. Reading 15 GB of cached weights into memory takes half a minute or so, and the card says so rather than sitting still.
- **My Models** lists every model in the Hugging Face cache with its size on disk, and says whether it would load: *fits* is a model there is room for now, *tight* one that fits the machine but not what is free at the moment, and *won't fit* one that is larger than the machine can hold whatever is free. The verdict is the same check that refuses a load, made before the button is pressed and against the **Weight precision** chosen, so switching to 4-bit repaints both lists and shows what that buys. An image pipeline is judged as one: its size is summed over its component folders, each from its own stored dtype, and on an NVIDIA card it has to fit both the card and the machine, because it is read into host memory before it moves onto the card. For the first few seconds after the app starts, before it has finished reading the device, the verdicts are given as though the weights were loaded whole: too generous a verdict would send a reader to a button that then refuses them, and the lists are repainted as soon as the device is known. A model whose files are incomplete has no size to judge yet, and the model already in memory is not judged again - unless the weight precision has been moved since it was loaded, since **Load cached** on it is how a new precision is applied and a model that fits at four bits may not fit whole. Every verdict is for a model that would replace whatever is loaded, and a load frees the old weights before it checks whether the new ones fit, so what the device is holding now is counted as available. Two figures answer that, and the larger is the one used: what the device's own allocator reports, and what the last load estimated its weights would take. Neither is enough alone - host memory keeps no allocator figure at all, and a model spread over the graphics cards and the machine is only counted on the cards by one while the other covers the whole of it. **Sort by** orders the list newest download first, by name, or by size in either direction. A model short of files is marked *incomplete* and tinted amber, a diffusers pipeline *image*, an MLX conversion *MLX*, one that is whole but not a kind ChatLab loads (a CTranslate2 or ONNX export, a GGUF file, a folder of SAE weights) *unsupported*, and the one in memory *loaded*. An MLX model is judged at the size of its files, since they are read onto the device as they are. Selecting one shows its file count, architecture and weight type from its `config.json`, revision, when it was last downloaded, and its folder, and puts its ID in the model box ready for **Load cached**. **Redownload** fetches whatever the selected model still lacks, resuming partial files rather than starting over; on a complete model it checks the Hub for updated files. **Remove** deletes the selected model's folder from the cache after a confirmation; a model that is loaded or still downloading has to be unloaded or finished first. The list rescans after every download, load, unload, and removal, and **Refresh** rescans it by hand.
- **Discover models** opens with an offline shortlist of recommended starters, each with a reason to try it and an estimate of the full repository download. Choose **Popular**, **Trending**, or **New** to browse Hugging Face without knowing a model name; **New** means newest repositories, not recently updated ones. Enter a model name or organization and press **Search / refresh** to narrow the current view. In **Recommended**, this filters the starter catalog first and searches the Hub, most downloaded first, when no starter matches. **Text models** finds language models with Transformers support, **Image models** finds text-to-image diffusers pipelines, and on Apple silicon **MLX models** finds language models quantized for MLX whose architecture mlx-lm implements. Results still go through ChatLab's runtime compatibility checks, excluding GGUF-only repositories, MLX conversions from the Transformers list, and text architectures that need a different loader. Select a result to see its memory estimate, access requirements, license and popularity when available, and put its ID in the model box ready for **Download and load**. Browsing and selecting never download or load weights.
- **Fits this computer** shows only models estimated to fit the current memory budget at the chosen weight precision. Tight, too-large, and unknown sizes are hidden. Changing precision or turning the filter off reuses the fetched candidates without another Hub request. Each Hub query examines at most 400 repositories, retains up to 100 compatible candidates, and displays up to 20 results after filtering; an empty filtered list means no estimated fits in those candidates, not that the Hub has none. Narrow the search to look further. Memory estimates use the Hub's parameter counts and are not guarantees; image pipelines often lack a complete count and therefore have no fit estimate. Full download estimates in the starter catalog include alternate file formats because ChatLab downloads the entire repository. Quantization reduces loaded memory, not download size; live Hub results whose download size is unavailable link to the repository files instead.

### Settings

The system prompt, assistant prefill, and reasoning options; the analysis and input controls described below; the context limit under **Memory**; and a **Hardware** panel. Settings apply to the next reply on any conversation.

**Hardware** is what the memory guard reads when it decides whether a model fits: the device a load would use and the precision it would read weights as, the machine's memory and how much of it ChatLab estimates is available within its own limits, the safety reserve it keeps beside the weights, the Metal cap and the share of Metal's recommendation it comes to, what the device allocator is holding for this process, and the model in memory. It is read when the page opens, when the Settings page is opened, after every load and unload, and whenever **↻ Refresh** is pressed - not on a timer, since reading it costs a subprocess. The same figures go to the log with every load and every reply, which is what makes a memory failure readable after the fact; the panel is how to look before one.

The sampling controls are **not** here. Temperature, top-p, top-k, the response length and the seed are what gets moved between one retry and the next, so they sit under the message box on the Chat page, in a **Sampling** section that wears its own values: the summary reads without opening it.

Four of them belong to the conversation rather than to the app: temperature, top-p, top-k and the response length are kept per conversation, so one fork can sit at temperature 0 while another beside it sits at 1.2, and switching between them brings each one's sliders back. A fork answers the way the conversation it was forked from does, and forking pins both sides to that: a conversation carrying no sampling of its own follows the settings file, and the first slider moved on either side would rewrite the file and move the other with it, which is the one thing the fork was for. A new conversation starts from the settings file and is pinned to what it said at the time; the file is also where a control moved on any conversation is written, so a new conversation begins from the values last used. The seed and **New seed each response** are not per conversation: a finished reply leaves the seed it used in that box, so a seed kept per conversation would record the app's dice rather than a choice.

Every setting is saved as you change it, and read back the next time the app
starts. They live in one file:

```
~/.config/chatlab/settings.json
```

`XDG_CONFIG_HOME` moves the directory and `CHATLAB_SETTINGS_PATH` names the
file outright, so the file can be symlinked out of a dotfiles repository and
shared between machines. It is plain JSON, written with one key per setting,
and safe to edit by hand: a value out of range is pulled back into it and one
of the wrong type falls back to its default, with a line in the log to say so.
Keys the running version does not recognize are left where they are, so one
file can be shared by two machines on different versions. The Hugging Face
token is not among the settings, because a file meant to be committed is the
wrong place for a secret.

`prefill_token_limit` is **Context limit (tokens)** on the page, and it is
the one to lower when a model runs out of memory: it caps the tokens one
prompt may carry and, with it, the ceiling on the response length. Every
reply is measured against it, a branch and an ordinary answer alike, and a
conversation that has grown past it is refused before anything is allocated
rather than run until the machine gives out. The file
holds three keys whose controls are not on this page: `model_id`, the model
the Models page opens with, which `OLMO_MODEL_ID` still overrides for one run,
`weight_precision`, the Models page's **Weight precision** choice (`full`,
`8-bit` or `4-bit`, ignored by an MLX model), and `mps_memory_fraction`, the
Apple Metal cap described under [Memory](#memory), which is read when a
Transformers model is loaded; MLX has an allocator of its own, and an MLX load
is judged against the machine's free memory alone. The Images
page's own controls are saved beside them under the `image_` keys.

## Images

The Images page draws a picture with a diffusion model and reads back what the
model did while it drew. Load a diffusers text-to-image pipeline on the Models
page — the model list marks one *image*, and **Discover models** finds them under
**Image models** — then type a prompt and press **Draw**. A pipeline that
wants more than a prompt (img2img, inpainting, upscaling, video, audio) is
marked *unsupported* rather than offered, because this page has only a prompt
to give it, and one too old to report its denoising steps is refused before it
draws rather than producing a run with no trajectory and a Stop button that
does nothing. **Stop** ends the run
after the step it is on and keeps the trajectory it had recorded; there is no
finished picture, because the pipeline never reached its decode.

**Drawing settings** holds the denoising step count, the guidance scale, the
size, the seed, and **Record cross-attention**. The negative prompt sits with
the prompt rather than in that accordion, because it changes the readings as
well as the picture: it is what the unconditional half of every step is
prompted with, and the guidance pull is measured against it.

Weight precision does not apply here. The Metal quantizer is Transformers'
own, and a pipeline is several models of which only some are Transformers
ones, so an image load takes its weights whole and says so in the log rather
than quantizing part of it and reporting a precision that held for the text
encoder alone.

Three readings sit beside the picture, and none of them needs the pipeline to
be rewritten — whichever pipeline the repo ships is the one that runs, prompt
encoding and scheduler and all.

**Denoising trajectory** is one frame per step, with a slider to scrub
through. The frames are a linear projection of the latent's four channels onto
red, green and blue, not a full decode: running the VAE on every step would
about double the wait, and layout and colour are what a frame is for. The
finished picture beside it is the pipeline's own decode. The frames say so, so
a disagreement over detail is not a puzzle.

**Guidance and movement** charts two numbers per step on one axis, both of them
a length divided by a length. The *guidance pull* is how far the prompt moved
that step's prediction away from the unconditional one, as a fraction of the
unconditional prediction's own size: 0.2 means the prompt pulled by a fifth of
what the model would have drawn from noise alone. It is the closest thing a
diffusion model has to surprise, and it is read straight out of the denoiser's
own output, whose batch is the unconditional and conditional predictions side
by side. With guidance at 1 or below there is no second prediction and nothing
was pulling, so the series is absent rather than flat. The *latent movement* is
how far each step moved the latent relative to where it already was; it falls
as a picture settles, and the tiles name the step from which every step moved
less than a tenth of the largest move — the step the composition was decided
and the rest became detail.

**Prompt attention** shades every prompt token by its share of the picture's
cross-attention, against the strongest token in the prompt, and clicking one
lays its map over the picture: brighter is more of that cell's attention.
Averaged over heads and over every cross-attention layer that could be read.
The pipeline's own attention kernel never builds the probability matrix, so
recording it means computing the queries, keys and softmax a second time
alongside — which is the one reading that costs real time, and why
**Record cross-attention** can be turned off for the pipeline's own speed
while the trajectory and the guidance trace still arrive.

The line under the strip reports what the *padding* took, which is usually
most of it. CLIP pads every prompt to a fixed length and Stable Diffusion
passes no attention mask, so those positions past the end of your prompt are
attended to like any other; without that line a token's share would be a share
of something unnamed and every number on the page would look mysteriously
small.

The step slider moves the frame, the shading and the map together. Attention
moves between steps as much as the picture does, so a strip left on the run's
average beside a moved frame would be quietly wrong.

## Working with a conversation

- **Stop** cancels the running generation and keeps whatever was produced so far.
- **Retry** regenerates the last reply. Because **🎲 New seed each response** is on by default, a retry actually explores a different sample; turn it off to lock the seed and reproduce a response exactly. The seed field always shows the seed that produced the response on screen.
- Hovering a message in the transcript gives per-message retry, edit, and undo. Editing one of your messages truncates the conversation there and generates a new reply; editing a reply just corrects it in place. **↩️ Undo last** removes the last exchange and puts your message back in the input box.
- **💾 Save conversation** writes a JSON file containing every turn, its reasoning block, and the system prompt, along with the model and token counts behind each reply. **📂 Load conversation** restores it.

### Token view

The transcript has two views of the same conversation, and **Token view** above
it switches between them. Off, the chatbot renders each reply the way you would
read it: markdown, code blocks, a collapsed reasoning block. On, the same
messages are written out as the tokens the model actually emitted — spaces as
`␠`, newlines as `↵`, special tokens named — each painted by whichever scale
**Color tokens by** is set to.

Click any token there and the inspector on the right describes it: its rank,
its probabilities, its surprise, and the alternatives the model ranked highest.
That is the same click that starts a branch, and the same click **🔬 Inspect
layers** reads. Click your own message to open **Edit your message** below the
transcript. **Save and regenerate** replaces everything after that message with
a new reply; **Cancel** discards the edit. You stay in token view throughout.
Clicking a heading or a message you typed also clears the inspector, since
neither has a distribution behind it.

Replies carry their own measurements, so the whole conversation is painted, not
just the newest reply. A message you typed, a reply you edited by hand, and a
conversation restored from the saved library have no measurements to paint and
appear as plain text.

### Assistant prefill

Enter text in **Assistant prefill (optional)** to force every new reply to begin
with those words. ChatLab measures the prefilled tokens against the model's own
distribution, then resumes sampling after them. **Maximum new tokens** counts
only the tokens sampled after the prefill, so the prefix does not reduce the
requested continuation length.

For a reasoning model whose chat template already opens a `<think>` block,
ChatLab closes that block before replaying the prefill. The supplied text
therefore appears as the visible answer rather than hidden reasoning. Clear the
field to return to ordinary generation. JSON metric exports record the supplied
text as `assistant_prefill` and the replayed token count as
`forced_prefix_tokens`.

### Steering a conversation

Open **Chat → Conversation tools → Steering vector** and choose **Import vector**.
The JSON file contains a direction previously extracted from the model. ChatLab
imports one vector per conversation; it does not extract vectors from examples.
This is the file shape (the three-entry vector below is only an illustration):

```json
{
  "format": "chatlab-steering-1",
  "model_id": "organization/model-name",
  "layer": 12,
  "vector": [0.12, -0.04, 0.09],
  "strength": 1.0,
  "enabled": true
}
```

Use the loaded model's exact ID and a full vector with one finite number per
hidden dimension. Layers start at **0**. `format`, `strength`, and `enabled` are
optional; their defaults are the values shown above. Files must be at most
4 MiB, with at most 65,536 vector entries. ChatLab checks the model ID, decoder
architecture, target layer, and vector width before applying steering. The file
should come from the same checkpoint and use the same layer-output convention;
the ID and shape checks cannot establish that a direction has the intended meaning.

**Enable steering**, **Steering strength**, and **Target layer** control subsequent
Chat responses, retries, and token branches. Strength ranges from -100 to 100;
zero disables the addition and negative values reverse the direction. The
operation is `layer_output + strength × vector`, applied to every token position
during both prompt processing and response generation. Vectors are not normalized
automatically, so useful strength values depend on how the vector was created.
Model weights are unchanged, and the addition is removed when a response finishes,
fails, or is stopped.

Steering needs a PyTorch model: it adds its vector through a forward hook on a
decoder block, and an MLX checkpoint has none. With an MLX model loaded, an
enabled vector is refused with a message saying so and the previous response is
left in place; load the same model's unquantized Transformers version to steer
it. A vector that is switched off, or at strength zero, adds nothing and runs on
either backend.

Forks inherit their parent's vector and controls; **New conversation** starts
without a vector. ChatLab stores each vector once in `conversations-vectors/`
beside its conversation library; responses and settings keep small references,
so streaming does not copy or rewrite the vector for every token. **Save
conversation** embeds each referenced vector once for transfer to another machine,
and JSON trace exports include the response's full vector. Neither depends on the
original upload. **Inspect layers** replays the vector and strength recorded for
that response even after the controls change. Its probabilities describe the
steered model before sampling filters.

To reclaim vector files left behind after removing vectors or deleting
conversations, first quit ChatLab and close any other running ChatLab servers,
then run this from the source checkout:

```bash
.venv/bin/python -m steering --cleanup-unused
```

The command keeps vectors referenced by any saved response or conversation
setting, including inactive forks and disabled steering. It uses the same
`CHATLAB_LIBRARY_PATH` / `XDG_DATA_HOME` location as the app and refuses cleanup
if the saved library is unreadable or invalid. Cleanup runs only when explicitly
requested; keeping ChatLab closed protects references still held by live sessions.

## Branching from a token

Every response token comes with the alternatives the model ranked highest. Branching lets you take one of them instead and see where the model goes from there.

1. Tick **Token view** and click the token in the conversation.
2. Click a row in **Most likely alternatives**. The detail panel confirms what the branch will do.
3. Press **🌱 Branch from token**.

The reply is kept up to the token before the one you clicked, the alternative is put in its place, and the model continues from there under the current sampling settings. The branch replaces the reply it was taken from and every message after it, so it is a different continuation of the conversation rather than an edit buried in the middle of one; **Retry** and **Undo** then work on it as usual. Choosing the token the model already picked resamples the rest of the reply from that point, which is a way to see how much of what followed was chance.

Any reply in the conversation can be branched, not only the newest one, because each carries the tokens it was made of. What it cannot outlive is the model: a reply's token IDs belong to the tokenizer that produced them, so loading another model - or reloading the same one - leaves the older replies readable but unbranchable, and the detail panel says so when you click one.

The replayed tokens are still measured against the model's own distribution, so a token the model would never have chosen shows its real rank and surprise. **Maximum new tokens** counts the tokens sampled after the branch point, so a branch made late in a long response still has room to finish. The JSON export records how many tokens were replayed as `forced_prefix_tokens`.

To explore one token at a time, choose an alternative and press **Next token** below the message box. This applies the selected token and samples just one additional token. Each subsequent click extends the latest reply by one token, without selecting its last token again. It uses the current sampling and steering settings and leaves **Maximum new tokens** unchanged. A reply that has reached its stop token cannot advance; choose an earlier alternative to start another branch. You can also use **Next token** on a reply paused with **Stop** or the token limit. Each step replays the reply's prefix through the model, so longer replies take longer to advance.

### Branching with your own text

The alternatives table only offers what the model ranked highly. To put anything else at a token position, click the token, type the replacement in **Or type your own replacement**, and press **✏️ Branch with text**. The typed text is spliced in exactly as written where the clicked token was, and the model continues from there. Type the space yourself if the word needs one: the text is checked in place, after the tokens that are kept, so it reads the same whether the tokenizer keeps the word-boundary space inside the token (as BPE does) or drops it from the start of what it decodes (as SentencePiece does). It can be one word or a whole sentence. Text the tokenizer cannot reproduce exactly at that position is refused rather than approximated. The prompt and replayed response prefix together are capped at 8,192 tokens, or at the model's shorter positional limit; an oversized branch is refused without replacing the response on screen.

Only a reply the model wrote can be branched. Prompt tokens, text measured in the **Score text** tab, and a message typed or edited by hand have no measured tokens to continue from. Editing a reply also takes the measurements off it and off every reply after it: those were produced from a transcript the edit replaced.

## The conversations pane

On the Chat page, the pane beside the nav lists every conversation. Each entry shows the conversation's name and the start of its first message, then the model that answered and how many tokens the conversation has come to. Click an entry to switch to it.

The token count is the size of the conversation as the model last saw it: every token in the prompt behind the latest reply - system prompt, transcript, and chat template - plus every token of the reply, reasoning included. It updates as a reply streams, so a reply that is stopped part way shows how far it got. The count belongs to the reply the model generated, and a conversation loaded from an older file, or whose only replies were typed in by hand, says so instead of showing a number. A conversation answered by more than one model names each of them, most recent first.

**➕ New** puts the conversation on screen away and starts an empty one. **🌿 Fork** copies the conversation into a new fork and switches to it, so you can ask something different without losing the original. **🗑️ Delete** removes the conversation on screen and returns to the main one, which cannot be deleted; **🗑️ Clear all** under the chat empties it and removes every other conversation with it, which is why it asks first and names how many it would take.

Click a message before pressing Fork to fork at that point. Forking at a reply keeps the conversation through that reply, ready for a different next question. Forking at one of your own messages keeps what came before it and puts the message back in the input box so it can be reworded, the same shape **Undo** gives.

Each conversation carries its own token view, since the measurements live on the replies themselves; what switching conversations does clear is the prompt strip, the charts and the export, which describe one reply at a time. **💾 Save conversation** writes the conversation on screen.

Every conversation in the pane is kept between sessions, its own sampling
included, and the two are kept apart when two windows disagree: a window
that moves a slider does not thereby claim a transcript it may be a reply
behind on, and a newer transcript does not undo a slider moved in the other
window. The whole pane - the active conversation and every other branch -
is written to one file as it changes, a streaming reply included, and read back when the page loads, so a
browser reload, a restart or a crash brings it back where it was. A reply
that was still streaming when the page went away is kept as far as it got.
Two windows on the same file - two tabs, or a reload beside the tab it
replaced - do not write over each other: each save is merged into the file
one conversation at a time, a conversation only the file knows stays, and
where both windows have one the more recent change wins. Deleting a
conversation in one window removes it from the file even while the other
still shows it. There is no live sync between windows: each shows the pane as
it was when it loaded, and a reload brings it up to date. The file is:

```
~/.local/share/chatlab/conversations.json
```

`XDG_DATA_HOME` moves the directory and `CHATLAB_LIBRARY_PATH` names the file
outright, the same two knobs the settings file answers to. It is written whole
and swapped into place, so a crash mid-write leaves the previous copy rather
than half of a new one. The token measurements are not in it: the file is rewritten on
every streaming frame, and a few hundred numbers per token would make that a
multi-megabyte write per token. A restored conversation therefore comes back
as plain text in the token view, and measured again from its next reply on. **💾 Save conversation** is still
the way to hand one conversation to someone else, and **📂 Load conversation**
brings such a file in.

## Layers and attention

The token panel says how likely a token was. **Layers and attention** says how the model got there.

1. Click a token in the conversation's **Token view**, or in **Prompt and context tokens**.
2. Open **Layers and attention** and press **🔬 Inspect layers**.

The readout is offered for the reply on screen and the prompt behind it. An earlier reply's prompt is no longer on screen to rebuild the pass from, so clicking one of its tokens reports its numbers without offering the layers.

The model is run again over everything before the token, one extra pass. That costs a few seconds on a 7B model with a long context, which is why it is a button rather than something that happens on every click. The key-value cache that pass builds is kept for the next inspection: clicking through the tokens of one response feeds the model only the tokens between one click and the next, so the second and later inspections of a response take a fraction of a second. The cache is given back the moment a response or a scoring pass starts, so it never competes with a reply for memory.

**Logit lens.** The residual stream after each layer is read through the model's final norm and unembedding, as though the network had stopped there. The chart traces the probability of the chosen token from the embeddings to the output; the faint line is whatever each layer liked best. The table under it names that preferred token per layer, with the chosen token's rank and the distribution's entropy, and the caption says from which layer the chosen token stayed the first choice. The last row is the model's real output and matches the numbers in the token panel. Readings from early layers are approximate: the lens assumes every layer writes in the same basis the output reads, which is roughly true late in the stack and less so early. The final norm is looked up under the names the common architectures use, on the base model and one level down (OPT keeps it inside its decoder); a model whose norm cannot be found shows only the output row, since readings taken without it would be wrong rather than approximate. Heads that post-process their logits (Gemma's soft-capping, Granite's and Cohere's scaling) are replicated, and the reading of the final layer is checked against the model's real output before any intermediate row is shown; a mismatch also falls back to the output row alone.

**Attention.** The prediction for a token is made at the position *before* it, so that earlier token is the query, drawn with a dashed outline. Every token it could see is shaded by how much attention it received, averaged over heads, and the strongest are listed underneath. The **Attention layer** slider picks one layer or, at 0, the mean of all of them; moving it repaints from the stored readout without another pass. A layer with a sliding window (most of OLMo 3's, and Mistral's) sees only the most recent tokens, so the ones it could not see are shown with no weight. The first token of a sequence almost always takes a large share regardless of content (the attention sink), so shading is scaled to the strongest token after it and the sink's share is stated in words.

Attention weights need the model's eager attention kernel, which is switched on for the inspection step only and switched back afterwards. A model that cannot return them still gets the logit lens. Only the first token of a sequence has nothing to show: nothing came before it.

## Reasoning blocks

Text the model wraps in `<think>` tags is pulled out of the reply and shown as a collapsible **Reasoning** section, so the answer stays readable while the trace stays available.

For supported Qwen3 models, **Thinking mode** under *Prompting* offers
**Model default**, **On**, and **Off**. This controls the
model's native thinking mode for the next chat reply or retry, on both PyTorch
and MLX. The control appears only when the loaded Qwen3 checkpoint's chat
template supports switching; thinking-only and instruct-only variants do not
get a switch. The choice is saved between sessions and recorded with each
reply and in its JSON/CSV token trace. Token branches keep the original reply's
mode so their replay uses the same template. An assistant prefill still starts
directly in the answer, even with thinking enabled. Batch prompts and the local
API continue to use the model's default mode.

By default that reasoning is **not** sent back to the model on the next turn. Think models are trained to produce a fresh reasoning block each time, so replaying old ones spends context and tends to degrade the next answer. Enable **Send previous reasoning back to the model** under *Prompting* if you want the older behavior.

## Reading the visualization

- **Raw rank** is the generated token's position in the model's unmodified distribution. Rank 1 was the model's first choice.
- **Raw model probability** is calculated before temperature or filtering.
- **Actual sampling probability** includes temperature, top-k, and top-p.
- **Surprise** is `-log2(probability)`. Larger values are less expected.
- **Distribution entropy** is the width of the whole distribution the model chose from, in bits. Surprise says how unexpected the choice was; entropy says how undecided the model was before making it.
- **Top-1 margin** is the probability gap between the model's first and second choice.
- **Sampling shift** is `log2(sampling probability / raw probability)`: how far your temperature, top-k, and top-p settings moved that token away from the raw model.
- **Probability mass above it** is the combined raw probability of every token ranked above the generated token.

Each of these names carries its own sentence in the token detail panel, so hovering one — or reaching it with a screen reader — says what the number is without leaving the page.

**Color tokens by** repaints the conversation without regenerating anything. Rank, surprise, and entropy are magnitudes and share one light-to-dark blue ramp; sampling shift is a diverging red-to-blue scale around no change. Quantized model weights can slightly change logits, probabilities, and ranks.

Under each response are its headline numbers — perplexity, mean surprise, the share of tokens the model ranked first, mean entropy — and a trace of surprise across the response, so a stretch where the model lost the thread is visible at a glance. Long responses are grouped into bins, with the range inside each bin shaded.

After a response finishes, open **Export full metric trace** under the conversation and use **Download JSON** or **Download CSV**. JSON preserves the complete trace, including conversation context, generation settings, and nested alternatives. CSV contains one row per generated token, repeats the generation metadata, and expands every recorded alternative into numbered columns.

## Prompt tokens and scoring text

Every prompt token is measured against the distribution the model held one step earlier, during the same pass that fills the key-value cache, so it costs nothing extra to see how predictable your own prompt was. They appear under **Prompt and context tokens**; the first token has nothing before it, so it is left unscored. Turn the measurement off in **Sampling, analysis, and input controls** if you do not want it, and note that only the most recent 1,024 tokens of a very long prompt are scored.

The **Score text** tab measures text the model did not generate. Paste it, optionally give it context first, and one forward pass reports the same numbers for every token — useful for comparing two prompts, checking how memorized a passage is, or evaluating a response that came from somewhere else. Scoring is capped at 4,096 tokens per run, or at the model's shorter positional limit. A line under the box counts what is in it against that cap as it is typed, using the same encoding the check itself uses, so a passage too large to score says so before the press rather than after it.

## Running a list of prompts

The **Prompts** tab runs an experiment rather than a conversation. Write the prompts into the box with a blank line between them, so a prompt can run to several lines, or press **Load prompts** for a file: `.jsonl` is one prompt per line (a plain string, or an object with a `prompt`, `text`, or `content` field), `.json` is a list of them, and any text file is read on the same blank-line rule. A loaded file is added to what is already in the box, and the box is what runs, so the set can be edited first.

A loaded prompt with a blank line inside it is the one thing the box cannot show whole, since that is how the box separates one prompt from the next. The run uses the file's own prompts while the box still holds what loading them wrote, so a dataset entry of several paragraphs is answered as the one prompt it is; the status line says so when a file contains one. Editing the box hands the reading back to it, blank lines and all.

Each prompt is answered in a conversation of its own. Nothing carries over from the prompt before it: the model sees the system prompt from **Settings**, the prompt, and nothing else. Sampling comes from the controls under the message box, so a batch is measured exactly as a reply typed by hand would be. With **New seed each response** off, every prompt runs on the seed in the box and the run reproduces; with it on, each prompt gets its own seed, and the row and the trace both record which.

The results table gives one row per prompt — an excerpt of the prompt and the answer, the token count, perplexity, mean surprise, and the seed. Below it are the files: `prompt-001.json` and its siblings, each a full trace in the same schema **Download JSON** writes for a single response, and `prompts.csv`, one row per generated token across the whole run with a `prompt_index` column naming the prompt each row came from and a `stopped` column saying whether that answer was cut short. Both are written as the run goes, so **Stop** — or Escape — leaves every prompt that produced tokens downloadable, the one it was in the middle of included; that one's trace says `stopped` in its sampling, since the model may have had more to say.

A prompt that fails does not end the run. Its row says what went wrong, the rest of the set still runs, and the status line counts the failures at the end. A model swapped in from another tab does end it: the run is pinned to the model it started on, so the rest of the prompts are refused rather than answered by other weights and reported in the same table.

## The local API

Everything ChatLab does to the model in memory is addressable from a script.
The API is served on the same port as the interface, so if the app is at
`http://127.0.0.1:7860` then its API is at `http://127.0.0.1:7860/v1`, and any
OpenAI client can be pointed at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7860/v1", api_key="not-needed")
answer = client.chat.completions.create(
    model="allenai/Olmo-3-7B-Think",
    messages=[{"role": "user", "content": "Name three cities."}],
    logprobs=True,
    top_logprobs=5,
)
```

`GET /v1/chatlab/status` is the call to make first: it names the model in
memory, says whether a response is already running and whether a load is under
way - two different reasons a request is turned away, and only one of them
ends by itself - and reports the memory
figures the hardware panel shows. `busy` and `loading` come from one reading of
what has the model, so they are never both true and never disagree with the 409
a request sent at that moment would get; a load is named ahead of a response.
`GET /v1/models` lists every complete model
in the cache and marks the loaded one.

`POST /v1/chat/completions` answers a conversation. It takes `messages`,
`temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stream`, `logprobs`
and `top_logprobs`, and anything left out takes the value the app is set to,
so a script and the interface answer alike unless the script says otherwise. A
value outside what ChatLab allows is refused with the nearest it would take
rather than clamped in silence. Reasoning arrives as `reasoning_content`
beside the answer's `content`, in the response and in the stream, so nothing
has to strip `<think>` markers. A trailing assistant message is the assistant
prefill: the reply must begin with that text, and its tokens are measured as
replayed rather than sampled.

With `logprobs`, every token carries its own `logprob`, its bytes, and the
alternatives `top_logprobs` asked for, and beside them, under `chatlab`, the same
measurements the token panel shows: raw rank, raw and sampling probability,
surprise, entropy, the top-1 margin, the probability mass above it, and the
sampling shift. `chatlab` names the seed, the device and the weight precision that answered
- in the closing event of a stream as much as in a whole response, since by
the time a stream ends another load may have replaced them.
`chatlab.summary` comes with every response whether or not
the tokens do - perplexity, mean surprise, the share the model ranked first -
and `prompt_logprobs: true` adds the prompt's own tokens under
`chatlab.prompt_tokens`, in the response or in the stream's closing event. The token that ended the response is measured and
counted like any other, even though it is not part of the text. A token
holding part of a character - a byte-level tokenizer splits one over several
- reports `bytes` as null rather than the bytes of the replacement character
it decodes to on its own; the text itself is assembled from the tokens
together and is unaffected.

`POST /v1/chatlab/score` is the **Score text** tab: give it `text` and
optionally `context`, and it measures every token in one forward pass. It
names the device and the weight precision that measured them, as a
completion does. It is
the endpoint for running a corpus past a model rather than a passage at a
time.

The API answers for the model already loaded and never loads one: a load takes
minutes, replaces what is in memory, and can be refused for want of it, so it
stays on the Models page where it can be watched. A request naming another
model is refused by name, and a request that passes that check is bound to
the load it was checked against: a load that lands before the first token is
refused rather than answered by weights the request did not name. Only one generation runs at a time, as in the
interface, and a second request is told the model is busy rather than queued
behind an answer thousands of tokens long. A request that arrives while the
Models page is loading something is refused too - one load and one generation
exclude each other - and it is told that rather than told a response is
running: the error type is `model_loading` instead of `model_busy`. Each generation runs on one thread
of its own and its frames cross to the response through a queue, so a
streaming answer is never resumed on a different worker; a client that stops
reading is noticed within a minute, and the model is handed back rather than
held by a response nobody is listening to; a client that comes back after
that is told the response was given up on rather than handed the tokens
that did arrive as though they were the whole answer. There is no
authentication, and
there is none on the interface either: both are served on the loopback address
and anything that can reach one can already do everything the other can.

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

The application deliberately leaves `trust_remote_code` disabled. Models that require executing custom repository code will not load unless their architecture is supported directly by Transformers or, for an image model, by diffusers.

The image tests need no pipeline weights. `tests/fake_pipeline.py` is a
denoising loop small enough to run on a laptop's CPU whose cross-attention
module is diffusers' own `Attention`, so the recording processor is exercised
against the class it will meet rather than a mock of it.
The application deliberately leaves `trust_remote_code` disabled. Models that require executing custom repository code will not load unless their architecture is supported directly by Transformers.

## Optional extensions

Specialized tools can be enabled under **Settings → Extensions** and take effect after restarting ChatLab. **Maze experiments** adds an interactive navigation workbench with interruptions, token inspection and saved-run replay. It is bundled and disabled by default. See [the extension guide](EXTENSIONS.md) and [the Maze workbench guide](MAZE_WORKBENCH.md).
