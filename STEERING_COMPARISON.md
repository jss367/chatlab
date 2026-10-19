# Steering comparison

Enable **Steering comparison** under **Extensions**, restart ChatLab, then open **Steering** in the sidebar. This view compares what steering changes in a Jacobian lens's vocabulary readout across multiple examples. It needs a full-precision Transformers model and a matching fitted lens imported in **Chat → Layers and attention → Jacobian**.

## Collect a comparison

1. Enter up to 64 examples in the table. Give each a decision-point name (up to four groups) and a prefix ending at the token immediately before the decision you want to study. The final prefix token is processed and read, with no later text fed to the model. Prefixes are literal text, not automatically chat-templated: include the model's template when needed. Each prefix must fit the model's window and the 4,096-token limit.
2. Open up to six exported `chatlab-steering-1` vector files. Their filenames label the vector selector. Each vector retains its injection block and saved strength. **Strength multiplier** multiplies those strengths, and can be negative to test the opposite direction. Read blocks must be at or after every vector's injection block.
3. Choose up to eight fitted decoder blocks, separated by commas, using zero-based indices.
4. Press **Collect comparison**. **Stop** cancels at the next block or scoring chunk and releases the model. A stopped run leaves the previous completed result visible.

Every example is read under no steering, the selected vector, and one seeded random vector of equal norm at the same injection block. The real and random vectors are added at every prefix position, as Chat steering does. For the computed-change comparison, there are also direct-addition passes at each read block. One vector and example take `3 + 2 × number of blocks` forward passes. The examples are identical under every intervention; this measures changes on fixed prefixes, not changes in which text steering would generate.

The optional **Returned to task** column accepts 1, 0, or empty for unknown. It records an outcome you supply about the example. The displayed rate uses only labeled examples at the selected decision point; it is the same supplied example set for every vector, not a newly measured recovery rate or an improvement over random steering. The tool does not generate or classify replies.

## Read the results

Select a vector, decision point, decoder block and change:

- **Total** compares the steered prefix against the unsteered prefix.
- **Computed** compares against the unsteered prefix with the same vector added directly at the read block. This removes the immediate addition from the comparison and asks what processing between the injection and read blocks changes. At the injection block itself this comparison has no change. The random effect is measured against its own random direct-addition pass.

For every readout, the entire output vocabulary receives an exact rank, `1 + number of tokens with strictly higher scores`. Equal scores share a rank. Final-block replay must reproduce the model's actual output before readings are accepted. Vocabulary scoring is chunked; each request is capped at eight million ranks, and decision-point groups are processed separately to bound working memory.

The table reports:

- **Rank improvement (t)**: the paired t statistic of `log(comparison rank / steered rank)` across examples. Positive means moving toward rank 1. Fewer than two examples or zero sample variance gives a dash, rather than an infinite or invented statistic.
- **Rank change**: the geometric mean of the same ratio. ×3 means three times closer to the top, ×0.5 means twice as far away. The block trace shows this factor at the other captured blocks.
- **Top 50 / Top 10**: the share of examples whose rank falls within that cutoff, separately for steered and comparison readouts.
- **Random t**: the equivalent paired log-rank statistic for the matched-norm random intervention.

Sort by t, top-50 entry gain (steered share minus comparison share), or geometric rank factor. Filter by token substrings, separated by commas, and by the minimum top-50 share on either side. **Specific only** keeps real t > 2 and more than 2 above random t. This is a browsing heuristic, not a multiple-comparison-corrected significance test. One random direction does not estimate the distribution over random directions.

Moved-up and moved-down tokens appear in separate tables, up to 100 rows each. All vocabulary tokens are ranked, but only tokens reaching the top 50 at least once in a real, comparison or random readout are retained for browsing. A zero top-50 floor therefore does not include tokens that never approached the top in any readout.

## Save and reopen

**Download comparison** writes a `chatlab-steering-comparison-1` JSON result containing model and lens metadata, vector inputs, example prefixes, seed, multiplier, and aggregate statistics for every retained token, block, group and change mode. **Open saved comparison** restores its selectors, tables and examples without loading a model. Prefixes and vectors are included in the download, so share it as you would the original inputs. Imported results are limited to 128 MiB and checked before display. Individual example vocabulary ranks and raw activations are not retained.
