# Linear probes

The **Linear probes** extension fits a logistic regression to the loaded model's residual stream at every layer, from two sets of examples you write. It reports how well each layer's probe does on examples it was not trained on. Then it reads any reply or passage with the probe and colors every token by the probe's probability.

Enable it under **Settings → Extensions** and restart ChatLab. **Probes** appears in the sidebar. Probes need a PyTorch load. An MLX checkpoint has nowhere to put the forward hooks the readings are taken through, so it is refused.

## Training a probe

Load a model on the Models page. On the Probes page, name the two sides under **Looking for** and **Against**, and write 2 to 64 examples of each, one per line. **Train probe** reads every example through the model once and keeps each decoder block's output, pooled to one vector per block. These are the same readings Chat's **Extract a vector** takes, from the same hook, so block *n* is the tensor a steering vector at layer *n* is added to.

**How examples are read** holds four settings:

- **Read each example as a user message** puts each example in a user turn followed by the generation prompt, so the last token is where the model would answer from. Unticked, the example is read as a passage, starting with whatever marker the tokenizer opens one with.
- **The examples are pairs** says that line *n* of each side is the same example with the property changed: "Paris is the capital of France." against "Paris is the capital of Italy.". Each pair is then held out together. Leave it unticked for unpaired examples. When paired examples are split across folds, a held-out example's twin sits in training with the opposite label. The twin is the nearest thing to it there, so held-out accuracy falls below chance however well the probe generalizes. When every layer reads below chance, the summary suggests this setting.
- **Pool each example by** the last token, or the mean over every token.
- **L2 strength** is the penalty on the weights. Each feature is standardized before fitting, so the penalty treats every direction alike. The default, 1, is scikit-learn's default `C=1`.

The table lists every layer with three numbers:

- **Held-out accuracy**: the examples are dealt into up to five folds, with both sides in every fold. The probe is refitted with each fold left out and scored on that fold. Choose a layer by this number. The fold assignment is seeded, so training the same examples twice gives the same table.
- **Held-out loss**: the mean log loss of those held-out predictions. It breaks ties between layers with equal accuracy.
- **Training accuracy**: the probe fitted on every example, scored on those same examples. A residual stream has more dimensions than you have examples, so this is nearly always 100%. The gap between it and held-out accuracy shows how much the probe overfits.

The best layer is the one with the highest held-out accuracy. With a few dozen examples these numbers are noisy. Read the curve across layers rather than one layer's figure.

## Reading with a probe

Choose **Generate a reply** to send a message to the loaded model and read its reply, or **Read text** to read a passage as written. The reply streams into **Reply**. When it finishes, the whole sequence is run through the model once more, and every block's output at every position is read along that block's probe. One reading covers at most 4,096 tokens, prompt included, or the model's window if that is shorter. A longer passage is refused, and a reply is limited to 4,096 tokens. A stopped reply is still read, up to where it stopped. A reply that ran into the model's window ends on a token the model sampled but never read, so that token is left out of the reading.

The strip colors each token by the probe's probability at the layer on the slider. The slider starts at the best layer. Five buckets run from blue (under 10%, the **Against** side) through neutral (30–70%) to red (over 90%, the **Looking for** side). Moving the slider repaints the strip without running the model again. Click a token to see its probability at every layer. **Every layer** shows a grid of every layer against every token. Hover a cell to see its token.

**Show the prompt's tokens too** adds the templated prompt to the strip in front of the reply. **Read the text as a user message** wraps a passage in a user turn. It starts out matching the probe's own setting, so a passage is read the way its examples were.

The probe was trained on one position of each example, the last token or the mean. Mid-sentence tokens are a different distribution from that position, so read colors there with care. The position a probe was trained on is usually where its readings are cleanest: the final period of a sentence, or the generation prompt when the examples were read as user messages.

Reading needs the model the probe was trained on. A probe for another model is refused before anything runs, and so is a probe trained on another revision of the same model ID when both revisions are known.

## Saved probes

Every trained probe is saved as a `chatlab-probe-1` JSON file under the extension's data directory, by default `~/.local/share/chatlab/extensions/probes/`. **Saved probes** lists them by name, model and date. Opening one shows its table and refills the training form with its examples and settings, so you can change them and train again. A new training is always saved as a new probe. **This probe** downloads the current probe. **Import a probe file** opens one written elsewhere and saves a copy.

A probe file holds:

| Field | Meaning |
| --- | --- |
| `format` | `chatlab-probe-1` |
| `id` | 32 hexadecimal characters |
| `name`, `model_id` | the probe's name and the model it was trained on |
| `model_revision` | the checkpoint revision it was trained on, or `null` when the load recorded none |
| `positive_label`, `negative_label` | the **Looking for** and **Against** names |
| `pool`, `chat_template`, `paired`, `l2`, `folds` | how the examples were read and fitted |
| `examples` | `{"positive": [...], "negative": [...]}`, as written |
| `layers` | one entry per decoder block: `layer`, `weights`, `bias`, `train_accuracy`, `heldout_accuracy`, `heldout_loss` |
| `best_layer` | the layer the slider starts at |
| `created` | a Unix time between 1970 and 3000 |

The weights are in the raw activation basis, with the standardization folded in, so the probability at block *n* is `sigmoid(weights · x + bias)` for that block's output `x`. A probe for a 7B model is about 2.5 MB.
