# Chatlab

A local chat interface that shows what happened under the hood for every generated token. Tokens are colored by their rank in the model's original next-token distribution. Clicking a token shows its probability, sampling probability, surprise, and the alternatives the model preferred.

## What it includes

- Hugging Face model download and cache controls
- A chat interface with OLMo reasoning blocks
- Live token-by-token generation
- Exact raw vocabulary rank for each generated token
- Raw and post-sampling probabilities
- Temperature, top-p, top-k, seed, and response-length controls
- Apple Metal, NVIDIA CUDA, and CPU loading

The default model is [`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think). Its full weights require a download of roughly 15 GB. Other Hugging Face causal language models with built-in Transformers support can also work.

## Run it

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

## Reading the visualization

- **Raw rank** is the generated token's position in the model's unmodified distribution. Rank 1 was the model's first choice.
- **Raw model probability** is calculated before temperature or filtering.
- **Actual sampling probability** includes temperature, top-k, and top-p.
- **Surprise** is `-log2(probability)`. Larger values are less expected.
- **Probability mass above it** is the combined raw probability of every token ranked above the generated token.

The color scale uses raw rank: rank 1, ranks 2–5, ranks 6–20, ranks 21–100, and ranks 101 or lower. Quantized model weights can slightly change logits, probabilities, and ranks.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The application deliberately leaves `trust_remote_code` disabled. Models that require executing custom repository code will not load unless their architecture is supported directly by Transformers.
