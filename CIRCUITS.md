# Circuit tracing

The **Circuit tracing** extension adds **Circuits** to the sidebar. It builds an attribution graph for the token after a prompt: the transcoder features that carried the model to that token, joined by their direct effects on one another. You can then group features and test the groups by ablating or boosting them in the real model. **Neurons and heads** inspects native model components without transcoders; **Validate feature** tests a graph feature with strength sweeps, source-activation patches, and downstream restoration.

The method follows Ameisen et al., [*Circuit Tracing*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (2025), with per-layer transcoders, and makes the same choices as [circuit-tracer](https://github.com/safety-research/circuit-tracer) where the paper leaves room. The code lives in `chatlab/extensions/circuits/`.

## Inspecting native neurons and attention heads

Enable **Circuit tracing** under **Extensions**, restart ChatLab, and open **Circuits → Neurons and heads**. This mode needs no transcoder downloads. It supports the Llama, Qwen 2, Qwen 3, OLMo 3, Gemma 2, and Gemma 3 text architectures, with full-precision Transformers weights on one device. MLX and quantized loads are refused.

Choose **MLP neuron** or **Attention head**, then enter its layer and index, both counted from zero. A native neuron is one channel after the MLP's gate and up projections are multiplied, immediately before its down projection. A head is its full output vector immediately before the attention output projection. These indices identify actual model components; they are separate from the learned transcoder features in **Trace** and **Features**.

Write a user message and, optionally, the reply so far. List the output tokens whose next-token probabilities you want to measure, one per line. Leading spaces matter; each line must encode to exactly one token. **Edited token offset from end** selects where the component is changed: 0 is the last prompt token, 1 is the token before it. The baseline activation strip shows each token and its offset when hovered.

**Inspect and test** runs an ordinary baseline forward pass and a separate pass for each **Scale factor**. Zero ablates the component, one is a no-op control, and two doubles its activation at the selected token. Up to eleven distinct factors, including zero and one, are accepted between -10 and 10. Negative factors reverse the activation and can leave the range of naturally occurring activations. Attention and normalization run normally in every intervention.

The results show:

- baseline activations across the prompt: signed scalar activations for neurons, vector norms for heads;
- the component's effective activation under every intervention;
- each output token's probability, its multiple of baseline, and its change in log probability;
- a compact summary of the measured ablation effect;
- a vocabulary projection heuristic for the neuron's output direction per unit activation, or the head's baseline output at the selected position. This omits downstream layers and normalization, so it is not a measured causal effect.

For MLP neurons, **Find nearby neurons at the edited token** also draws a local neighborhood for the first destination prefix. It ranks candidates in the immediately preceding layer by activation times the local gradient into the selected activation, then ablates the top six individually and reports their measured effect on that activation. It shows neurons in the immediately following layer whose activations change most when the selection is ablated. This search covers those adjacent layers at the selected token; it does not enumerate every neuron, earlier token, or attention path. At a boundary layer, only the available side is shown. These extra partial forwards count toward the run's pass limit. Head inspection uses the output and restoration tests without this neuron search.

**Additional destination replies** tests the same component on further prefixes with the same system and user message. Separate replies with lines holding only `---`. Each prefix gets its own results instead of being averaged with the others. The offset is applied from each prefix's end; choose prefixes with comparable token positions when testing a common mechanism.

### Patching and testing a proposed pathway

Under **Patch from a contrasting prompt**, enable copying the source activation and provide the source user message, reply so far, and token offset. The source uses the destination's system prompt and template setting. ChatLab records a source baseline, then copies only the selected neuron's scalar activation or head's output vector into the destination at its edited token. The source and destination may have different lengths; their explicit offsets define the alignment. This is an activation replacement, not an added steering direction or a replacement of the entire residual stream.

Under **Test a proposed pathway**, choose a component in a later layer. ChatLab first measures the upstream ablation, then runs it again while restoring the later component to its original destination baseline activation. Its table shows whether upstream ablation changed the later activation and whether restoring it rescues the output probability. All other computation runs normally. A rescue supports investigating that pathway, but does not by itself establish a unique circuit or prove that a human-assigned interpretation is correct. Head interventions inspect or change head outputs; they do not explain how a head chooses its attention pattern.

Each prompt is limited to 512 tokens. A run accepts up to 256 destination prefixes and 1,024 model passes, checked before intervention forwards begin. **Stop** interrupts between decoder blocks or passes and removes the temporary hooks. Inspection results are offered as JSON and saved under the extension's data directory in `inspections/`. They record exact source and destination token IDs, model revision/load, component indices, offsets, factors, activations, and per-prefix measurements.

## Models

A graph needs a transcoder for every layer of the model, trained on that model. Graph tracing works with the models whose transcoders circuit-tracer publishes on the Hub:

| Model | Transcoders | Weights |
| --- | --- | --- |
| `google/gemma-3-270m-it` | Gemma Scope 2, 16k features per layer | 1.5 GB |
| `google/gemma-3-1b-it` | Gemma Scope 2, 16k features per layer | 3.9 GB |
| `google/gemma-2-2b` | Gemma Scope, 16k features per layer | 7.9 GB |
| `Qwen/Qwen3-0.6B` | Qwen3 low-L0 transcoders, 164k features per layer | 18.8 GB |

Load the model at full precision through Transformers. MLX loads and 8-bit or 4-bit weights are refused: they have no gradients to take. The transcoders download on the first trace (or **Load transcoders**) and stay in memory beside the model until **Unload transcoders** is pressed or another model's set replaces them. ChatLab's memory check before a load does not count them. `google/gemma-3-1b-it` is the default for a reason: with its transcoders it holds about 6.5 GB.

Each feature's top-activating examples, top output tokens and activation frequency are fetched from the Hub one feature at a time, by byte range, when you click it. They are cached under the extension's data directory.

## Tracing a graph

Write a user message and, optionally, the start of the reply. The graph explains the token after that reply. **Plain text, no chat template** sends the user message alone, after the beginning-of-sequence token, which is what a base model such as `google/gemma-2-2b` expects.

**Explain** picks the graph's output nodes:

- **The likeliest next tokens**: one node per token, the likeliest first, until they cover 95% of the probability or there are ten.
- **Chosen tokens**: one node per token you list.
- **Pivot tokens against others**: a single node for log P(pivot) − log P(other), where each side's probability is the sum over its tokens. This is the "pivot – other" node: what pushes the model toward *Wait*, *But* or *Hmm* and away from *So* or *The*.

The publisher configurations identify each base model by repository name but do not report the exact model checkpoint used for training. ChatLab therefore marks **checkpoint compatibility as unverified**, rather than treating a matching repository name as proof. Traces record the actual loaded model revision and immutable transcoder revision. The recorded reconstruction-error terms and frozen-forward check use the actual loaded model, and interventions measure that model directly; published feature examples remain descriptions of the publisher's training data, whose calibration for a different checkpoint is unverified. If a publisher supplies an exact training revision, the catalogue can record it and ChatLab refuses other or unknown model revisions for that set.

List tokens one per line. Leading spaces matter, since most vocabularies treat " Wait" and "Wait" as different tokens. Type `\n` for a line break. A line that is not exactly one token is refused, with its token count.

### What the trace does

1. The prompt runs once. Every transcoder reads its layer's MLP input, and the active features and the transcoder's error at each position are fixed. Features at the first position (the beginning-of-sequence token) are left out and counted as error, as circuit-tracer does.
2. The prompt runs again, batched, with every MLP output replaced by the features' decoded sum plus that error, every RMSNorm scale held at its value, and every attention pattern held at its value. The values are unchanged, but the model is now linear in the token embeddings, the features and the errors.
3. Each output node, and then each feature in order of its influence on the output, takes one backward pass. A source's edge into a target is the source's value times the target's gradient with respect to it: its direct contribution. The edges into a target, plus the transcoder bias terms, add up to the target's value exactly. The tests check this on Gemma 2, Gemma 3 and Qwen 3.
4. Features stop being traced after **Features to trace** (default 400). The graph is then pruned to the fewest features holding 80% of the influence on the output, and the fewest edges holding 98% of what remains. Both shares are under **Graph size**.

On an M-series Mac, a 400-feature graph of a 30-token prompt on `google/gemma-3-1b-it` takes about 15 seconds. Edge rows are held on the CPU, so a prompt with very many active features is refused with an estimate rather than run out of memory. Prompts are limited to 512 tokens.

## Reading the graph

Layers run from the bottom up and prompt positions from left to right. The graph opens on the last position, where the output nodes are.

- A **circle** is a feature. Its size is its influence on the output. Blue features promote the output, red ones suppress it, and gray ones have a small total effect. A feature's total effect counts its paths through other features as well as its direct edge.
- A **square** is a prompt token's embedding. The token is named on the axis below it.
- A **diamond** is transcoder error at one layer and position: what the transcoder failed to reconstruct. These are hidden until **Show transcoder error nodes** is checked. The note under the graph says how much of the influence they carry. On the Gemma Scope 2 transcoders this is often 70% or more, and a graph should be read as a partial account.
- Blue edges excite and red edges inhibit. Only the strongest edges into each node are drawn, scaled within that node. Hover a node to see only its edges.

**Features shown** sets how many of the most influential features are drawn. Click a feature to read it on the right: its activation, total effect on the output, graph influence, activation frequency, the tokens its decoder promotes and suppresses, and its top-activating examples from the transcoder training data. A feature is labelled with the two tokens it promotes most until you give it a name of your own with **Rename**.

**Ablate this feature** sets the feature to zero at its own position in the real model, with nothing frozen, and reports how each output node's log-probability moves. For a contrast node, it reports the change in log-odds.

### Focused neighborhoods and feature validation

Select a node and open **Neighborhood** to see its strongest incoming contributors and downstream effects around a central box. Click a neighboring node to follow the circuit. The number on each edge is its signed direct contribution in the frozen attribution model. Only edges retained by graph pruning are available; these values are distinct from measured ablation effects.

For a selected feature, **Validate feature** provides the same strength sweeps, contrasting-prompt activation patches, per-prefix results, and later-component restoration described above. It starts at the feature's traced token offset; selecting a later feature to restore also fills in its traced offset. You can change either offset explicitly. These interventions subtract or replace the feature's decoded contribution while preserving the transcoder reconstruction error, and later features read the changed model state. They use the graph's model and immutable transcoder revision, with the same compatibility checks as group interventions. Validation downloads identify the originating graph and transcoder revision.

## Groups and interventions

Shift-click (or Command-click) features to select several, name the selection, and press **Group selected**. A feature belongs to one group at a time.

The **Groups** tab draws each group as a box joined to the prompt tokens, the transcoder error and the other groups by the sum of the edges between their members. **Run interventions** then measures each group in the real model:

- **ablate**: every member's activation set to zero;
- **boost**: every member's activation multiplied by the **Boost factor** (2 by default).

A feature's activation is read from its layer's input on the same forward pass, so a change in an early group reaches later features through the model itself. The transcoder's error is left alone. By default each member is changed only at the position the graph found it, counted back from the end of the prompt. **Change features at every position** changes them wherever they fire.

**Pivot tokens** and **Alternatives** are the tokens measured. P(pivot) is the pivot tokens' summed probability at the last position. Leave the alternatives blank to use the model's eight likeliest other tokens. Each box shows P(pivot) after ablating and after boosting, as a multiple of its baseline. Its border is blue when boosting raises P(pivot) by more than 5% and red when it lowers it by more than 5%. Beside the boxes, each token's baseline probability is drawn as a bar. Choose a group to see how ablating and boosting it moves each token, as a multiplier on a log scale.

**More replies so far** takes further replies, separated by lines holding only `---`. Each follows the same system prompt and user message. Probabilities are averaged over all of them, and the traced prompt is included unless unchecked. This is how a group found in one graph is tested on many prefixes, for example 60 reasoning traces cut at a paragraph break. Each group's card says how many prefixes it was active in.

A measurement run accepts at most 32 nonempty groups and 1,024 model passes, calculated as the number of prefixes times (1 + twice the number of groups). Reduce groups or prefixes if the limit is exceeded; the limit is checked before reserving the model.

## Browsing features

The **Features** tab lists every feature of a transcoder set, twenty at a time. No model has to be loaded to browse. Choose the **Transcoders**, a **Layer** and the **First feature**, then press **Show**; **Previous** and **Next** turn the page. Each row gives the feature's index, the tokens it fires on, the tokens its decoder promotes, and how often it is active. *Fires on* counts, over the feature's top-activating examples in the transcoder's training data, which token each example peaked on, most common first. Records are fetched from the Hub as the graph's feature card fetches them, and cached the same way, so a page that was shown once opens offline.

Click a row to read the feature on the right: its highest recorded activation, activation frequency, the tokens it fires on and writes, and its top-activating contexts.

**Steer by this feature** puts the feature's decoder row on the current Chat conversation as its steering vector and opens Chat. The vector is added to the residual stream after the feature's layer, at every position, which is the feature switched on everywhere. The row is scaled to the feature's highest recorded activation, so **Strength** counts multiples of that: 1 adds what the feature writes at its strongest. The default of 3 moves `google/gemma-3-1b-it` toward the feature without breaking its sentences; at 8, the features tried while writing this turned its replies to noise. A negative strength steers away from the feature. The strength, the layer and the switch are then the Chat page's own controls, under **Conversation tools → Steering**, and the vector is saved with the conversation like an imported one.

Steering reads one layer's transcoder file, not the whole set, and downloads it first if it is not cached. The vector names the model the transcoders were trained on, so steering refuses to run until that model is loaded through Transformers.

## Saved graphs

Every graph is saved when it is traced, and again whenever a feature is renamed, a group changes or interventions are run. Graphs are written as `chatlab-attribution-graph-1` JSON under the extension's data directory, in `graphs/`. Open one from **Saved graphs**, or upload a graph file. The current graph is offered as a download.

Measuring a saved graph needs the model it was traced on.

## Limits

- Only per-layer transcoders are supported. Cross-layer transcoders, and transcoders with a skip connection, are refused.
- Attention is part of the frozen model and is not explained: an edge from a token three positions back says the token's information arrived, not which head carried it.
- A picture in the prompt is not supported; the prompt is text only.
- Graph sizes and timings scale with active features. A long reply on a model with many features per token can exceed the 6 GB edge budget, and is refused before tracing starts.
