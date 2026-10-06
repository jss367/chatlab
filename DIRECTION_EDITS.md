# Direction edits

The **Direction edits** extension edits the loaded model's residual stream along chosen directions during a forward pass. It shows how far the edit carries through later blocks and what it does to a word's probability read through a Jacobian lens. It is made for walking one passage through an experiment like this:

1. Inject a concept vector at one block over a range of tokens.
2. Erase a "detector" direction at one block, or at a range of blocks.
3. Watch whether later blocks rebuild the detector's signal.
4. Measure whether a target word's lens probability under different instructions changes.

Enable it under **Settings → Extensions** and restart ChatLab. **Edits** appears in the sidebar. The edits are forward hooks on the decoder blocks, so the extension needs a PyTorch load at full precision. An MLX checkpoint has nowhere to put the hooks, and 8-bit and 4-bit loads are refused as well, so load the model's unquantized Transformers version.

## The passage and its conditions

Write one **Passage**, then up to six **Conditions**, each a name and a prefix read before the passage: for example a prefix that does not mention the word, "The word orange is on today's list.", "Do not think about orange." and "Focus on orange.". A row with no name and no prefix is skipped. An empty prefix is allowed and reads the passage alone.

The prefix and the passage are tokenized apart and joined, so the passage is the same tokens under every condition. End a prefix with the space or line break that should come before the passage. Token numbers count from 1 at the first passage token and include both ends, so "tokens 1–64" are the passage's first 64 tokens whatever the prefix. Blocks count from 0, as everywhere in ChatLab: block *n* is the output a steering vector at layer *n* is added to.

## Directions

**Directions** opens one direction per block, in either of two formats:

- a `chatlab-directions-1` file, described below;
- a `chatlab-probe-1` file from the **Probes** page, whose layer weights are taken as each block's direction.

Every direction is scaled to unit length, so a coordinate along it is in the residual stream's own units. A file made for another model is refused before anything runs, and so is one made for another revision of the same model ID when both revisions are known. A direction file may leave blocks out; those blocks are blank in the heatmaps and cannot be edited.

## Injection

**Injection** is optional. Open a `chatlab-steering-1` vector, the kind Chat's **Steering vector** imports and **Extract a vector** writes, and choose a **Strength** and a token range. The vector is added as strength × vector at the vector's own block over those passage tokens only. Chat's steering adds at every position; this does not. The strength starts at the file's own. A vector for another model is refused.

## The edit

Choose the blocks to edit, one or a range, and the passage tokens to edit at each. With *d̂* the unit direction at a block and *x* the block's output at a token:

- **Erase to reference** first reads the same tokens with nothing injected and records each edited block's coordinate *c*<sub>ref</sub> = *x* · *d̂* at every edited token. In the edited pass it sets *x* ← *x* + (*c*<sub>ref</sub> − *x* · *d̂*) *d̂*, which puts the coordinate back where it was without the injection and leaves everything at right angles to *d̂* alone.
- **Clamp** sets the coordinate to a fixed value: *x* ← *x* + (*v* − *x* · *d̂*) *d̂*.
- **Add** moves it by a fixed amount: *x* ← *x* + α *d̂*. With no injection, this switches a detector on in a run that never had one.

When an injection and an edit land on the same block, the injection is added first.

**Random control** makes a second edited pass in which each edit is replaced by one of the same size per token along a random unit direction at right angles to *d̂*. Each block's random direction is drawn from the **Seed** and the block number, so the same seed gives the same control. The control shows what editing the residual stream by that much does at all, whatever the direction, so it can be subtracted from the real edit's effect.

## What one run reads

**Run** holds the model for the whole run, so no other view's work lands between passes, and makes these passes for each condition:

1. a reference pass, with nothing injected or edited;
2. an injected pass;
3. an injected and edited pass;
4. with **Random control**, an injected pass with the random edit.

With no injection, the injected pass is the reference. Each pass is read along every block's direction at every token. When a target word is given, every pass but the reference is also read through the lens, the reference too when nothing is injected, and each lens reading is one more pass with the same edits. One condition with every option on is seven forward passes. **Stop** stops at the next block of the pass under way, and every hook is removed however the run ends. A condition, prefix and passage together may be at most 4,096 tokens, or the model's window if that is shorter.

### Coordinate heatmap

One grid per pass, blocks down and passage tokens across, for the condition chosen above them. Each block's row is colored on its own scale, shared by every pass, from that block's 5th percentile (cool) to its 95th (warm): the residual stream grows with depth, so one scale for every block would leave the early blocks blank, and the first token's coordinate is often far from the rest, so a row scaled to its extremes would paint every other token one color. Edited blocks are outlined. The marks above the grid show which tokens were injected, edited and read. Hover a cell to see its token and value. The grids show the first 512 passage tokens; the download keeps them all.

### Recovery by block

For every block and condition:

recovery = mean(edited − reference) / mean(injected − reference)

with both means taken over the edited tokens. 0 means the block's coordinate is back at the reference; 1 means the injected shift is all there. The means are taken before dividing, because a token the injection barely moved would otherwise dominate the ratio. A block the injection did not move, such as any block before the injection, shows a dash. Edited blocks are marked. With **Random control** each condition has a second column, the same ratio for the random pass, which should stay near 1.

After an edit at one block, the recovery at later blocks, which were not edited, is the number to read: it says how much of the signal the model rebuilt from what the edit left. The summary above the heatmaps gives each condition's recovery at the last block and the highest recovery after the edit.

### Lens table

Give a **Target word**, a token window and a block range under **Lens readout**. The table has a row per condition and a column per edit setting: no edit (the injected pass), the edit, and the random control. Each cell is the target's log probability through the Jacobian lens, averaged over every token in the window and every block in the range.

The lens is the one imported for this model in Chat's **Layers** view, or the one last imported there for this model, which is brought back as Chat brings it back. Every block in the range must be one the lens was fitted at. As in Chat, the final block's lens readout is checked against the model's own output in every pass, and the readings are withheld if they disagree. Include the leading space a word has mid-sentence. A word that spans several tokens is read as the mean of their unembeddings, and the note above the table says so.

**Differences**, one per line, subtract one condition's row from another's, for example:

```
suppression = neutral - ignore
focus = focus - neutral
```

Each difference is shown for every edit setting, with its change against no edit as a percentage.

## Saved results

**This result** downloads the run as a `chatlab-direction-edits-1` JSON file: every input, including the directions and the vector, the passage's token IDs and their text, each condition's prefix token IDs, and every number on the page. **Open a saved result** shows one again without a model.

## The direction file

| Field | Meaning |
| --- | --- |
| `format` | `chatlab-directions-1` |
| `name` | what the directions are, at most 200 characters |
| `model_id` | the model they were made for |
| `model_revision` | the checkpoint revision they were made for, or `null` when it is not known |
| `precision` | `full`, `8-bit` or `4-bit`, or `null` |
| `directions` | `[{"layer": n, "vector": [...]}, ...]`: at most one per block, all the same width, none of zero length |

The vectors need not be unit length; they are scaled on reading.
