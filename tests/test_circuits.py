"""Circuit tracing: the frozen replacement model, attribution, interventions and the page's pieces."""
import contextlib
import gzip
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from chatlab.extension_api import ModelService
from chatlab.extensions.circuits import (architecture, attribution, browser, interventions, render, transcoders,
                                        workbench)
from chatlab.extensions.registry import load_enabled
from fakes import FakeManager

WIDTH = 48


def tiny_model(kind):
    torch.manual_seed(0)
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    if kind == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        model = Gemma3ForCausalLM(Gemma3TextConfig(**common, sliding_window=4,
                                                   layer_types=["sliding_attention", "full_attention",
                                                                "sliding_attention"]))
    elif kind == "gemma2":
        from transformers import Gemma2Config, Gemma2ForCausalLM
        model = Gemma2ForCausalLM(Gemma2Config(**common, sliding_window=4, attn_logit_softcapping=20.0,
                                               final_logit_softcapping=15.0))
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        model = Qwen3ForCausalLM(Qwen3Config(**common))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                # Gemma stores its scale as an offset from one; a random one
                # makes a wrong convention in the frozen norm show.
                parameter.copy_(torch.randn_like(parameter) * 0.3 + (0.0 if kind != "qwen3" else 1.0))
    return model.eval()


def tiny_transcoders(blocks, *, bias=True, threshold=0.05):
    generator = torch.Generator().manual_seed(1)
    d = blocks.embed.weight.shape[1]
    layers = len(blocks.layers)

    def rand(*shape, scale):
        return torch.randn(*shape, generator=generator) * scale

    spec = transcoders.TranscoderSpec("tiny", "tiny", ("test/tiny",), "test/tiny", "", layers, WIDTH, d, 0.0)
    return transcoders.Transcoders(
        spec,
        w_enc=[rand(WIDTH, d, scale=0.4) for _ in range(layers)],
        b_enc=[rand(WIDTH, scale=0.05) for _ in range(layers)],
        w_dec=[rand(WIDTH, d, scale=0.2) for _ in range(layers)],
        b_dec=[rand(d, scale=0.1 if bias else 0.0) for _ in range(layers)],
        threshold=[torch.full((WIDTH,), threshold) if threshold is not None else None for _ in range(layers)],
        device=torch.device("cpu"),
    )


IDS = [2, 17, 5, 33, 9, 41, 12]


class FrozenModelTests(unittest.TestCase):
    def test_freezing_changes_no_values_and_is_undone(self):
        for kind in ("gemma3", "gemma2", "qwen3"):
            with self.subTest(kind=kind):
                model = tiny_model(kind)
                ids = torch.tensor([IDS])
                with torch.no_grad():
                    plain = model(ids).logits
                    implementation = model.config._attn_implementation
                    with architecture.frozen(model):
                        frozen = model(ids).logits
                    after = model(ids).logits
                torch.testing.assert_close(frozen, plain, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(after, plain)
                self.assertEqual(model.config._attn_implementation, implementation)
                norms = [m for m in model.modules() if type(m).__name__.endswith("RMSNorm")]
                self.assertTrue(norms)
                self.assertFalse(any("forward" in vars(m) for m in norms))

    def test_frozen_model_is_linear_in_its_input(self):
        # With the MLPs written out, what is left is attention and norms.
        # Frozen, that is linear without a bias, so the gradient dotted with
        # the input gives back the output; unfrozen, it does not.
        model = tiny_model("qwen3")
        blocks = architecture.blocks(model)
        embeddings = blocks.embed(torch.tensor([IDS])).detach().requires_grad_(True)
        with contextlib.ExitStack() as stack:
            for layer in range(len(blocks.layers)):
                stack.enter_context(architecture.replaced_output(blocks.mlp_output(layer),
                                                                 lambda x, *_a, **_k: torch.zeros_like(x)))
            plain = blocks.inner(inputs_embeds=embeddings).last_hidden_state[0, -1].sum()
            grad, = torch.autograd.grad(plain, embeddings)
            self.assertGreater(abs(float((grad * embeddings.detach()).sum() - plain.detach())), 0.1)
            with architecture.frozen(model):
                output = blocks.inner(inputs_embeds=embeddings).last_hidden_state[0, -1].sum()
                grad, = torch.autograd.grad(output, embeddings)
        torch.testing.assert_close((grad * embeddings).sum(), output, rtol=1e-4, atol=1e-4)

    def test_unsupported_models_are_refused(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gpt2"))
        with self.assertRaisesRegex(ValueError, "does not support gpt2"):
            architecture.blocks(model)


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.decode = lambda token: f"<{token}>"

    def frozen(self, kind, **options):
        model = tiny_model(kind)
        blocks = architecture.blocks(model)
        held = tiny_transcoders(blocks, **options)
        recording = attribution.record(blocks, held, IDS)
        return blocks, held, recording

    def test_recording_splits_each_mlp_into_features_and_error(self):
        blocks, held, recording = self.frozen("gemma3")
        self.assertGreater(len(recording.activation), 0)
        self.assertFalse(bool((recording.feature_position == 0).any()), "the first token's features are left out")
        for layer in range(len(blocks.layers)):
            start, end = recording.layer_slices[layer]
            acts = torch.zeros(len(IDS), WIDTH)
            acts[recording.feature_position[start:end], recording.feature_index[start:end]] = \
                recording.activation[start:end]
            rebuilt = held.decode(layer, acts) + recording.errors[layer]
            torch.testing.assert_close(rebuilt, recording.outputs[layer].float(), rtol=1e-5, atol=1e-5)

    def test_edges_into_a_logit_add_up_to_it(self):
        for kind in ("gemma3", "gemma2", "qwen3"):
            with self.subTest(kind=kind):
                blocks, held, recording = self.frozen(kind)
                targets = attribution.choose_targets(recording, self.decode)
                directions = attribution.logit_directions(blocks, recording, targets)
                graph = attribution.FrozenGraph(blocks, held, recording, batch_size=4)
                try:
                    vectors = [("vector", d) for d in directions[:4]]
                    rows = graph.rows(vectors)
                    biases = graph.bias_terms(vectors)
                finally:
                    graph.close()
                values = directions[:4] @ recording.final[-1]
                torch.testing.assert_close(rows.sum(1) + biases, values, rtol=1e-4, atol=1e-4)

    def test_edges_into_a_feature_add_up_to_its_activation(self):
        for kind in ("gemma3", "qwen3"):
            with self.subTest(kind=kind):
                blocks, held, recording = self.frozen(kind, bias=False)
                late = torch.nonzero(recording.feature_layer >= 1, as_tuple=True)[0][:6].tolist()
                self.assertTrue(late)
                graph = attribution.FrozenGraph(blocks, held, recording, batch_size=8)
                try:
                    rows = graph.rows([("feature", k) for k in late])
                finally:
                    graph.close()
                for row, k in zip(rows, late):
                    layer, feature = int(recording.feature_layer[k]), int(recording.feature_index[k])
                    expected = recording.activation[k] - held.b_enc[layer][feature]
                    torch.testing.assert_close(row.sum(), expected, rtol=1e-4, atol=1e-4)
                    # Nothing at or after the feature's own layer can feed it.
                    later = (recording.feature_layer >= layer).nonzero(as_tuple=True)[0]
                    self.assertEqual(float(row[later].abs().sum()), 0.0)

    def test_contrast_reads_the_log_odds_gradient(self):
        blocks, held, recording = self.frozen("qwen3")
        targets = attribution.choose_targets(recording, self.decode, contrast={"positive": [3, 4], "negative": [5]})
        self.assertEqual(len(targets), 1)
        direction = attribution.logit_directions(blocks, recording, targets)[0]
        hidden = recording.final[-1].clone().requires_grad_(True)
        logits = blocks.unembed(hidden.to(blocks.dtype)).float()
        log_probs = torch.log_softmax(logits, -1)
        odds = torch.logsumexp(log_probs[[3, 4]], 0) - log_probs[5]
        grad, = torch.autograd.grad(odds, hidden)
        torch.testing.assert_close(direction, grad, rtol=1e-4, atol=1e-5)

    def test_rare_contrast_sides_do_not_underflow_the_gradient_or_tracing_weight(self):
        blocks, held, recording = self.frozen("qwen3")
        with torch.no_grad():
            blocks.unembed.weight.mul_(1000)
        recording = attribution.record(blocks, held, IDS)
        rare = recording.logits.argsort()[:4].tolist()
        contrast = {"positive": rare[:2], "negative": rare[2:]}
        targets = attribution.choose_targets(recording, self.decode, contrast=contrast)
        self.assertEqual(targets[0]["probability"], 0.0)
        direction = attribution.logit_directions(blocks, recording, targets)[0]
        hidden = recording.final[-1].clone().requires_grad_(True)
        logits = blocks.unembed(hidden.to(blocks.dtype)).float()
        odds = torch.logsumexp(logits[rare[:2]].double(), 0) - torch.logsumexp(logits[rare[2:]].double(), 0)
        expected, = torch.autograd.grad(odds, hidden)
        torch.testing.assert_close(direction, expected, rtol=1e-4, atol=1e-5)
        graph = attribution.attribute(blocks, held, IDS, self.decode, contrast=contrast,
                                      settings=attribution.Settings(max_feature_nodes=16, batch_size=8))
        self.assertGreater(sum(node["influence"] for node in graph["nodes"]), 0)

    def test_gemma2_saturated_contrast_matches_the_softcapped_log_odds_gradient(self):
        blocks, held, recording = self.frozen("gemma2")
        with torch.no_grad():
            blocks.unembed.weight.mul_(100)
        recording = attribution.record(blocks, held, IDS)
        targets = attribution.choose_targets(recording, self.decode, contrast={"positive": [3, 4], "negative": [5]})
        direction = attribution.logit_directions(blocks, recording, targets)[0]
        hidden = recording.final[-1].clone().requires_grad_(True)
        raw = blocks.unembed(hidden.to(blocks.dtype)).float()
        self.assertGreater(float(raw.detach().abs().max()), blocks.final_softcap)
        logits = torch.tanh(raw / blocks.final_softcap) * blocks.final_softcap
        odds = torch.logsumexp(logits[[3, 4]], 0) - logits[5]
        grad, = torch.autograd.grad(odds, hidden)
        torch.testing.assert_close(direction, grad, rtol=1e-4, atol=1e-5)

    def test_saturated_token_direction_matches_centered_softcapped_logits(self):
        blocks, held, _ = self.frozen("gemma2")
        with torch.no_grad():
            blocks.unembed.weight.mul_(100)
        recording = attribution.record(blocks, held, IDS)
        targets = attribution.choose_targets(recording, self.decode, token_ids=[3])
        hidden = recording.final[-1].clone().requires_grad_(True)
        raw = blocks.unembed(hidden.to(blocks.dtype)).float()
        capped = torch.tanh(raw / blocks.final_softcap) * blocks.final_softcap
        grad, = torch.autograd.grad(capped[3] - capped.mean(), hidden)
        direction = attribution.logit_directions(blocks, recording, targets)[0]
        torch.testing.assert_close(direction, grad, rtol=1e-4, atol=1e-5)

    def test_large_batches_are_refused_before_device_construction(self):
        blocks = architecture.blocks(tiny_model("gemma3"))
        held = tiny_transcoders(blocks)
        recording = attribution.record(blocks, held, IDS)
        small = attribution.frozen_allocation_bytes(blocks, recording, 1)
        large = attribution.frozen_allocation_bytes(blocks, recording, 256)
        self.assertGreater(large, small * 100)
        with mock.patch.object(attribution, "MAX_ROW_BYTES", large - 1), \
                mock.patch.object(attribution, "FrozenGraph") as frozen:
            with self.assertRaisesRegex(ValueError, "smaller batch"):
                attribution.attribute(blocks, held, IDS, self.decode,
                                      settings=attribution.Settings(batch_size=256))
            frozen.assert_not_called()

    def test_memory_limit_counts_both_resident_edge_matrices(self):
        model = tiny_model("gemma3")
        blocks = architecture.blocks(model)
        held = tiny_transcoders(blocks)
        recording = attribution.record(blocks, held, IDS)
        targets = attribution.choose_targets(recording, self.decode)
        graph = attribution.FrozenGraph(blocks, held, recording, 8)
        single_matrix = (16 + len(targets)) * graph.columns * 4
        graph.close()
        # One matrix fits this limit, but the signed and normalized pair does not.
        with mock.patch.object(attribution, "MAX_ROW_BYTES", single_matrix + 1):
            with self.assertRaisesRegex(ValueError, "would need"):
                attribution.attribute(blocks, held, IDS, self.decode,
                                      settings=attribution.Settings(max_feature_nodes=16, batch_size=8))

    def test_memory_limit_reserves_pruning_before_tracing(self):
        model = tiny_model("gemma3")
        blocks = architecture.blocks(model)
        held = tiny_transcoders(blocks)
        recording = attribution.record(blocks, held, IDS)
        targets = attribution.choose_targets(recording, self.decode)
        graph = attribution.FrozenGraph(blocks, held, recording, 8)
        edge_pair = 2 * (16 + len(targets)) * graph.columns * 4
        graph.close()
        with mock.patch.object(attribution, "MAX_ROW_BYTES", edge_pair + 1), \
                mock.patch.object(attribution.FrozenGraph, "rows") as rows:
            with self.assertRaisesRegex(ValueError, "would need"):
                attribution.attribute(blocks, held, IDS, self.decode,
                                      settings=attribution.Settings(max_feature_nodes=16, batch_size=8))
            rows.assert_not_called()

    def test_memory_refusal_happens_before_frozen_graph_construction(self):
        blocks = architecture.blocks(tiny_model("gemma3"))
        held = tiny_transcoders(blocks)
        with mock.patch.object(attribution, "MAX_ROW_BYTES", 1), \
                mock.patch.object(attribution, "FrozenGraph") as frozen:
            with self.assertRaisesRegex(ValueError, "would need"):
                attribution.attribute(blocks, held, IDS, self.decode)
            frozen.assert_not_called()

    def test_frozen_construction_cancels_decoder_gathers_and_restores_model(self):
        model = tiny_model("gemma3")
        blocks = architecture.blocks(model)
        held = tiny_transcoders(blocks)
        recording = attribution.record(blocks, held, IDS)
        stopped = False
        rows = held.decoder_rows
        def first(*args):
            nonlocal stopped
            result = rows(*args)
            stopped = True
            return result
        with mock.patch.object(held, "decoder_rows", side_effect=first) as called:
            with self.assertRaises(attribution.Cancelled):
                attribution.FrozenGraph(blocks, held, recording, 4, cancelled=lambda: stopped)
            self.assertEqual(called.call_count, 1)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))
        self.assertFalse(any("forward" in vars(m) for m in model.modules()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_feature_edges_stop_between_decoder_chunks(self):
        blocks = architecture.blocks(tiny_model("gemma3"))
        held = tiny_transcoders(blocks)
        recording = attribution.record(blocks, held, IDS)
        stopped = False
        graph = attribution.FrozenGraph(blocks, held, recording, 4, cancelled=lambda: stopped)
        class Decoders:
            def __getitem__(self, key):
                nonlocal stopped
                stopped = True
                return original[key]
        layer = next(i for i, (start, end) in enumerate(recording.layer_slices) if end - start > 1)
        original = graph.decoders[layer]
        graph.decoders[layer] = Decoders()
        start, end = recording.layer_slices[layer]
        try:
            with mock.patch.object(attribution, "CHUNK_BYTES", 1):
                with self.assertRaises(attribution.Cancelled):
                    graph._feature_edges(torch.ones(1, graph.n, original.shape[-1]), layer, start, end)
        finally:
            graph.close()

    def test_recording_cancels_between_transcoder_layers(self):
        blocks = architecture.blocks(tiny_model("gemma3"))
        held = tiny_transcoders(blocks)
        stopped = False
        encode = held.encode

        def first(*args):
            nonlocal stopped
            value = encode(*args)
            stopped = True
            return value

        with mock.patch.object(held, "encode", side_effect=first) as called:
            with self.assertRaises(attribution.Cancelled):
                attribution.record(blocks, held, IDS, cancelled=lambda: stopped)
            self.assertEqual(called.call_count, 1)

    def test_contrast_sides_are_token_sets(self):
        blocks, _, recording = self.frozen("qwen3")
        unique = attribution.choose_targets(recording, self.decode, contrast={"positive": [3, 4], "negative": [5]})
        repeated = attribution.choose_targets(recording, self.decode,
                                              contrast={"positive": [3, 3, 4], "negative": [5, 5]})
        self.assertEqual(repeated, unique)
        torch.testing.assert_close(attribution.logit_directions(blocks, recording, repeated),
                                   attribution.logit_directions(blocks, recording, unique))

    def test_excess_chosen_targets_are_refused_before_recording(self):
        blocks = architecture.blocks(tiny_model("gemma3"))
        with mock.patch.object(attribution, "record") as record:
            with self.assertRaisesRegex(ValueError, "4,096"):
                attribution.attribute(blocks, None, IDS, self.decode, token_ids=list(range(4097)))
            record.assert_not_called()

    def test_contrast_needs_two_distinct_sides(self):
        _, _, recording = self.frozen("qwen3")
        with self.assertRaisesRegex(ValueError, "both sides"):
            attribution.choose_targets(recording, self.decode, contrast={"positive": [3], "negative": [3]})

    def test_a_whole_graph_is_consistent_and_saveable(self):
        model = tiny_model("gemma3")
        blocks = architecture.blocks(model)
        held = tiny_transcoders(blocks)
        stages = []
        graph = attribution.attribute(blocks, held, IDS, self.decode,
                                      settings=attribution.Settings(max_feature_nodes=16, batch_size=8),
                                      progress=lambda *args: stages.append(args[0]))
        json.dumps(graph)
        self.assertIn("Tracing features", stages)
        check = graph["targets_check"]
        for value, edges, bias in zip(check["values"], check["edge_sums"], check["bias_terms"]):
            self.assertAlmostEqual(edges + bias, value, places=3)
        ids = {n["id"] for n in graph["nodes"]}
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in graph["edges"]))
        stats = graph["stats"]
        self.assertLessEqual(stats["kept_features"], stats["traced_features"])
        self.assertLessEqual(stats["traced_features"], 16)
        self.assertTrue(0 <= stats["error_share"] <= 1)
        self.assertTrue(any(n["kind"] == "target" for n in graph["nodes"]))

    def test_cancelling_stops_between_batches(self):
        model = tiny_model("qwen3")
        blocks = architecture.blocks(model)
        with self.assertRaises(attribution.Cancelled):
            attribution.attribute(blocks, tiny_transcoders(blocks), IDS, self.decode,
                                  settings=attribution.Settings(max_feature_nodes=16, batch_size=4),
                                  cancelled=lambda: True)
        self.assertNotIn("forward", vars(blocks.mlp_output(0)), "the frozen model is undone on the way out")

    def test_stop_interrupts_target_batches_before_another_backward_pass(self):
        blocks = architecture.blocks(tiny_model("qwen3"))
        held = tiny_transcoders(blocks)
        stopped = False
        rows = attribution.FrozenGraph.rows

        def first(graph, targets):
            nonlocal stopped
            value = rows(graph, targets)
            stopped = True
            return value

        with mock.patch.object(attribution.FrozenGraph, "rows", first):
            with self.assertRaises(attribution.Cancelled):
                attribution.attribute(blocks, held, IDS, self.decode, token_ids=[3, 4],
                                      settings=attribution.Settings(batch_size=1), cancelled=lambda: stopped)
        self.assertNotIn("forward", vars(blocks.mlp_output(0)))

    def test_effect_passes_through_features_per_unit_of_activation(self):
        # source -> feature (activation 2, edge 4) -> target (edge 6), and source -> target (edge 1).
        adjacency = torch.zeros(3, 3)
        adjacency[1, 0] = 4.0
        adjacency[2, 1] = 6.0
        adjacency[2, 0] = 1.0
        weights = torch.tensor([0.0, 0.0, 1.0])
        effect = attribution._effect(adjacency, weights, torch.tensor([1]), torch.tensor([1.0, 2.0, 1.0]))
        self.assertAlmostEqual(float(effect[1]), 6.0)
        self.assertAlmostEqual(float(effect[0]), 1.0 + 4.0 / 2.0 * 6.0)

    def test_settings_are_checked(self):
        with self.assertRaisesRegex(ValueError, "Feature nodes"):
            attribution.Settings(max_feature_nodes=2).check()
        with self.assertRaisesRegex(ValueError, "Edge threshold"):
            attribution.Settings(edge_threshold=0).check()


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model("gemma3")
        self.blocks = architecture.blocks(self.model)
        self.held = tiny_transcoders(self.blocks)
        self.recording = attribution.record(self.blocks, self.held, IDS)

    def test_scaling_by_one_changes_nothing(self):
        k = 0
        layer, feature = int(self.recording.feature_layer[k]), int(self.recording.feature_index[k])
        plain = interventions.run(self.blocks, self.held, IDS, (), range(64))
        same = interventions.run(self.blocks, self.held, IDS, [(layer, feature, 1.0, None)], range(64))
        torch.testing.assert_close(torch.tensor(same["probabilities"]), torch.tensor(plain["probabilities"]))

    def test_boosting_in_the_last_layer_adds_its_decoder_row(self):
        last = len(self.blocks.layers) - 1
        start, end = self.recording.layer_slices[last]
        at_end = [k for k in range(start, end) if int(self.recording.feature_position[k]) == len(IDS) - 1]
        self.assertTrue(at_end)
        k = at_end[0]
        feature, activation = int(self.recording.feature_index[k]), float(self.recording.activation[k])
        boosted = interventions.run(self.blocks, self.held, IDS, [(last, feature, 3.0, [0])])
        added = 2.0 * activation * self.held.w_dec[last][feature]

        def hook(_module, _args, output):
            changed = output.clone()
            changed[0, -1] += added.to(output.dtype)
            return changed

        handle = self.blocks.mlp_output(last).register_forward_hook(hook)
        try:
            with torch.no_grad():
                expected = torch.log_softmax(self.model(torch.tensor([IDS])).logits[0, -1].float(), -1)
        finally:
            handle.remove()
        torch.testing.assert_close(boosted["log_probs"], expected, rtol=1e-4, atol=1e-4)
        self.assertEqual(boosted["active"], 1)

    def test_every_position_group_members_are_changed_once(self):
        member = (int(self.recording.feature_layer[0]), int(self.recording.feature_index[0]), 0)
        unique = interventions.group_effects(self.blocks, self.held, [IDS], {"g": [member]},
                                             pivot=[3], alternatives=[5], every_position=True)
        repeated = interventions.group_effects(self.blocks, self.held, [IDS],
                                               {"g": [member, (*member[:2], 1)]},
                                               pivot=[3], alternatives=[5], every_position=True)
        self.assertEqual(unique, repeated)

    def test_group_effects_average_over_prefixes(self):
        k = 0
        member = (int(self.recording.feature_layer[k]), int(self.recording.feature_index[k]),
                  len(IDS) - 1 - int(self.recording.feature_position[k]))
        result = interventions.group_effects(self.blocks, self.held, [IDS, IDS[:5]], {"g": [member]},
                                             pivot=[3, 4], alternatives=[5, 3], boost=2.0)
        self.assertEqual(result["prefixes"], 2)
        self.assertEqual(result["alternatives"], [5])
        baseline = result["baseline"]
        self.assertAlmostEqual(baseline["pivot"], baseline["tokens"][3] + baseline["tokens"][4])
        for kind in ("ablate", "boost"):
            self.assertTrue(all(0 <= p <= 1 for p in result["groups"]["g"][kind]["tokens"].values()))
        self.assertGreaterEqual(result["groups"]["g"]["active_prefixes"], 1)

    def test_group_effects_check_their_inputs(self):
        with self.assertRaisesRegex(ValueError, "pivot"):
            interventions.group_effects(self.blocks, self.held, [IDS], {"g": []}, pivot=[], alternatives=[])
        with self.assertRaisesRegex(ValueError, "group"):
            interventions.group_effects(self.blocks, self.held, [IDS], {}, pivot=[1], alternatives=[])


def small_graph():
    model = tiny_model("qwen3")
    blocks = architecture.blocks(model)
    graph = attribution.attribute(blocks, tiny_transcoders(blocks), IDS, lambda t: f"tok{t}",
                                  settings=attribution.Settings(max_feature_nodes=16, batch_size=8))
    graph["tokens"][2] = "<script>alert(1)</script>"
    graph.update(id="abc123", model_id="test/tiny", transcoders="tiny", labels={}, groups={}, effects=None,
                 prompt={"system": "", "user": "hi", "prefix": "", "raw": False})
    return graph


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = small_graph()

    def test_graph_view_escapes_text_and_limits_features(self):
        page = render.graph_view(self.graph, nodes_shown=3, show_errors=False)
        self.assertNotIn("<script>alert", page)
        self.assertLessEqual(page.count('class="cg-node feature'), 3)
        self.assertNotIn('class="cg-node error', page)
        with_errors = render.graph_view(self.graph, nodes_shown=3, show_errors=True)
        if any(n["kind"] == "error" for n in self.graph["nodes"]):
            self.assertIn('class="cg-node error', with_errors)

    def test_selected_and_named_features_are_drawn_so(self):
        feature = next(n for n in self.graph["nodes"] if n["kind"] == "feature")
        page = render.graph_view(self.graph, nodes_shown=1, selected=[feature["id"]],
                                 labels={feature["id"]: "my <name>"}, groups={"g": [feature["id"]]})
        self.assertIn("cg-node feature sel grouped", page)
        self.assertIn("my &lt;name&gt;", page)

    def test_group_flow_paths_are_bounded_even_with_equal_weights(self):
        graph = dict(nodes=[dict(id=str(i), kind="feature", layer=i, pos=0, feature=i) for i in range(12)],
                     edges=[dict(source=str(i), target=str(j), weight=1.) for i in range(12) for j in range(i + 1, 12)])
        groups = {str(i): [str(i)] for i in range(12)}
        with mock.patch.object(render, "MAX_GROUP_FLOWS", 5):
            page = render.group_view(graph, groups)
        self.assertEqual(page.count('<path class="cg-edge'), 5)

    def test_group_view_and_card_show_measured_effects(self):
        members = [n["id"] for n in self.graph["nodes"] if n["kind"] == "feature"][:2]
        effects = {"prefixes": 2, "boost": 2.0, "every_position": False, "pivot": [3], "alternatives": [5],
                   "baseline": {"tokens": {"3": 0.2, "5": 0.1}, "pivot": 0.2},
                   "groups": {"g": {"ablate": {"tokens": {"3": 0.1, "5": 0.1}, "pivot": 0.1},
                                    "boost": {"tokens": {"3": 0.4, "5": 0.05}, "pivot": 0.4},
                                    "active_prefixes": 2}}}
        view = render.group_view(self.graph, {"g": members}, effects, decode=lambda t: f"tok{t}")
        self.assertIn("ablate ×0.50 · boost ×2.00", view)
        self.assertIn('class="cg-group promotes"', view)
        card = render.group_card("g", members, self.graph, effects, decode=lambda t: f"tok{t}")
        self.assertIn("×2.00 when boosted", card)
        self.assertIn("×0.50", card)
        self.assertIn("cg-empty", render.group_view(self.graph, {}))

    def test_visible_prompt_sources_use_node_kind_instead_of_id_prefix(self):
        graph = dict(self.graph, nodes=[dict(n) for n in self.graph["nodes"]], edges=[dict(e) for e in self.graph["edges"]])
        mapping = {n["id"]: "imported-" + n["id"] for n in graph["nodes"] if n["kind"] == "embedding"}
        for node in graph["nodes"]:
            node["id"] = mapping.get(node["id"], node["id"])
        for edge in graph["edges"]:
            for key in ("source", "target"):
                edge[key] = mapping.get(edge[key], edge[key])
        before = render.visible(self.graph, 40, False)
        self.assertGreater(sum(n["kind"] == "embedding" for n in before), 0)
        after = render.visible(graph, 40, False)
        self.assertEqual(sum(n["kind"] == "embedding" for n in before), sum(n["kind"] == "embedding" for n in after))

    def test_group_names_cannot_collide_with_synthetic_flow_buckets(self):
        members = [n["id"] for n in self.graph["nodes"] if n["kind"] == "feature"]
        normal = render.group_view(self.graph, {"group": members})
        for name in ("@prompt", "@error", "@target"):
            view = render.group_view(self.graph, {name: members})
            self.assertEqual(view.count('class="cg-edge '), normal.count('class="cg-edge '))
            self.assertIn(f'data-group="{name}"', view)

    def test_feature_card_marks_the_strongest_token(self):
        feature = next(n for n in self.graph["nodes"] if n["kind"] == "feature")
        record = {"activation_frequency": 0.001, "top_logits": [" a"], "bottom_logits": [" b"],
                  "examples_quantiles": [{"examples": [{"tokens": ["x", "<y>", "z"],
                                                        "tokens_acts_list": [0.0, 3.0, 1.0]}]}]}
        card = render.feature_card(feature, record, tokens=self.graph["tokens"])
        self.assertIn('class="peak"', card)
        self.assertIn("&lt;y&gt;", card)
        self.assertIn("0.100%", card)
        self.assertIn("cg-empty", render.feature_card(None))


class TranscoderLifecycleTests(unittest.TestCase):
    def test_unload_drops_references_before_releasing_device_caches(self):
        import weakref
        held = tiny_transcoders(architecture.blocks(tiny_model("qwen3")))
        reference = weakref.ref(held)
        transcoders._LOADED[held.spec.key] = held
        del held
        with mock.patch("chatlab.model_loading.LoadingMixin._release_device_cache") as release:
            release.side_effect = lambda: self.assertIsNone(reference())
            self.assertTrue(transcoders.unload())
            release.assert_called_once()
        self.assertFalse(transcoders._LOADED)

    def test_cancellation_stops_download_before_the_next_layer(self):
        spec = transcoders.spec_for("google/gemma-3-1b-it")
        cancelled = threading.Event()
        def fetch(*args, **kwargs):
            cancelled.set()
            return "layer_0.safetensors"
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=fetch) as download:
            with self.assertRaises(attribution.Cancelled):
                transcoders.download(spec, cancelled=cancelled.is_set)
            download.assert_called_once()

    def test_all_layers_download_from_one_immutable_snapshot(self):
        spec = transcoders.TranscoderSpec("tiny", "Tiny", ("test/tiny",), "test/tiny", "", 2, 3, 2, 0.0)
        sha = "a" * 40
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda repo, name, **kw:
                        f"/tmp/models/snapshots/{sha}/{name}") as fetch:
            paths = transcoders.download(spec)
        self.assertEqual(fetch.call_args_list[0].kwargs["revision"], None)
        self.assertEqual(fetch.call_args_list[1].kwargs["revision"], sha)
        self.assertEqual(transcoders.snapshot_revision(paths[-1]), sha)
        held = tiny_transcoders(architecture.blocks(tiny_model("qwen3")))
        held.revision = sha
        workbench.Workbench._same_transcoders({"transcoder_revision": sha}, held)
        with self.assertRaisesRegex(ValueError, "another transcoder revision"):
            workbench.Workbench._same_transcoders({"transcoder_revision": "b" * 40}, held)

    def test_cancellation_stops_device_loading_before_another_layer(self):
        spec = transcoders.TranscoderSpec("tiny", "Tiny", ("test/tiny",), "test/tiny", "", 2, 3, 2, 0.0)
        cancelled = threading.Event()
        def read(path):
            cancelled.set()
            return {}
        with mock.patch.object(transcoders, "download", return_value=["layer0", "layer1"]), \
                mock.patch("safetensors.torch.load_file", side_effect=read) as load:
            with self.assertRaises(attribution.Cancelled):
                transcoders.load(spec, "cpu", cancelled=cancelled.is_set)
            load.assert_called_once_with("layer0")
        self.assertFalse(transcoders.in_memory(spec))

class WorkbenchTests(unittest.TestCase):
    def test_token_lists_keep_leading_spaces_and_read_escapes(self):
        self.assertEqual(workbench.parse_tokens(" Wait\nOkay\n\n\\n\\n\r\n"), [" Wait", "Okay", "\n\n"])
        self.assertEqual(workbench.parse_prefixes("one\n---\ntwo\nlines\n --- \n\n"), ["one", "two\nlines"])

    def test_plain_text_intervention_prefixes_are_appended_before_encoding(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        session = SimpleNamespace(encode=lambda text: [ord(c) for c in text])
        model = SimpleNamespace(config=SimpleNamespace(bos_token_id=1))
        prompt = dict(raw=True, user="prompt", prefix=" first reply", system="")
        ids = bench.prompt_ids(session, model, prompt)
        self.assertEqual(ids, [1] + [ord(c) for c in "prompt first reply"])
        other = bench.prompt_ids(session, model, {**prompt, "prefix": " second reply"})
        self.assertNotEqual(other, ids)

    def test_invalid_target_and_intervention_tokens_fail_before_model_loading(self):
        session = SimpleNamespace(model_revision=None, encode=lambda text: [1, 2] if text == "multi" else [int(text)])
        models = SimpleNamespace(open_session=lambda: contextlib.nullcontext(session))
        bench = workbench.Workbench(models, tempfile.gettempdir())
        explains = [dict(mode="tokens", tokens=[]), dict(mode="tokens", tokens=["multi"]),
                    dict(mode="tokens", tokens=[str(i) for i in range(attribution.MAX_CHOSEN_TARGETS + 1)]),
                    dict(mode="contrast", tokens=["1"], others=[]),
                    dict(mode="contrast", tokens=["1"], others=["1"]),
                    dict(mode="contrast", tokens=[str(i) for i in range(attribution.MAX_CHOSEN_TARGETS)],
                         others=[str(attribution.MAX_CHOSEN_TARGETS)])]
        with mock.patch.object(bench, "encoded_prompt", return_value=IDS), \
                mock.patch.object(bench, "_model", side_effect=AssertionError("expensive model path")), \
                mock.patch.object(bench, "_same_model"):
            for explain in explains:
                with self.subTest(mode=explain["mode"], count=len(explain["tokens"])), self.assertRaises(ValueError):
                    bench.trace({}, explain, attribution.Settings(), lambda *args: None, lambda: False)
            for pivot in ([], ["multi"], [str(i) for i in range(4097)]):
                with self.subTest(pivot=pivot), self.assertRaises(ValueError):
                    bench.group_effects({"ids": IDS, "groups": {"group": []}}, pivot, [], [], True,
                                        2.0, False, lambda *args: None, lambda: False)

    def test_saved_menu_uses_bounded_metadata_and_does_not_parse_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            bench = workbench.Workbench(SimpleNamespace(), directory)
            graph = small_graph()
            path = bench.save(graph)
            path.write_text("not parsed by the menu")
            self.assertEqual(bench.saved(), [(workbench.describe(graph), str(path))])
            legacy = path.parent / "legacy.json"
            legacy.write_text("not parsed either")
            self.assertEqual(len(bench.saved()), 2)

    def test_automatic_alternatives_search_beyond_excluded_top_twenty(self):
        self.assertEqual(workbench.automatic_alternatives(-torch.arange(64).float(), list(range(20))), list(range(20, 28)))

    def test_imported_identifiers_and_group_names_are_nonempty_and_bounded(self):
        graph = small_graph()
        member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for identifier in ("", "x" * 129):
                nodes = [dict(node, id=identifier) if node["id"] == member else node
                         for node in graph["nodes"]]
                path.write_text(json.dumps(graph | {"nodes": nodes}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            for name in ("", "x" * 61):
                path.write_text(json.dumps(graph | {"groups": {name: [member]}}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_imported_explanation_strings_are_bounded(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for key in ("tokens", "alternatives"):
                path.write_text(json.dumps(graph | {"explain": {key: ["x" * 4097]}}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_duplicate_imported_edges_are_rejected(self):
        graph = small_graph()
        self.assertTrue(graph["edges"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps(graph | {"edges": graph["edges"] + [graph["edges"][0]]}))
            with self.assertRaisesRegex(ValueError, "not a valid"):
                workbench.load_graph(path)

    def test_automatic_alternatives_follow_averaged_measured_prefixes(self):
        prefixes = [[7, 8], [7, 9]]
        distributions = [torch.tensor([0.9, 0.09, 0.01]), torch.tensor([0.001, 0.001, 0.998])]
        with mock.patch.object(interventions, "run", side_effect=[{"log_probs": p.log()} for p in distributions]) as run:
            selected = workbench.measured_alternatives("blocks", "held", prefixes, [1], lambda: False)
        self.assertEqual(selected, [2, 0])
        self.assertEqual([c.args[2] for c in run.call_args_list], prefixes)

    def test_imported_prompts_and_snapshot_revisions_are_bounded(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for key in ("system", "user", "prefix"):
                prompt = graph["prompt"] | {key: "x" * 32769}
                path.write_text(json.dumps(graph | {"prompt": prompt}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            for revision in (1, "main", "../snapshot", "g" * 40, "a" * 41):
                path.write_text(json.dumps(graph | {"transcoder_revision": revision}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            path.write_text(json.dumps(graph | {"transcoder_revision": "a" * 40}))
            self.assertEqual(workbench.load_graph(path)["transcoder_revision"], "a" * 40)
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        with mock.patch.object(bench, "_model", side_effect=AssertionError("must not tokenize")):
            with self.assertRaisesRegex(ValueError, "32768 characters"):
                bench.encoded_prompt(SimpleNamespace(), graph["prompt"] | {"user": "x" * 32769})

    def test_feature_aliases_cannot_duplicate_imported_coordinates(self):
        graph = small_graph()
        feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps(graph | {"nodes": graph["nodes"] + [dict(feature, id="alias")]}))
            with self.assertRaisesRegex(ValueError, "not a valid"):
                workbench.load_graph(path)

    def test_duplicate_imported_group_memberships_are_rejected(self):
        graph = small_graph()
        member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for groups in ({"one": [member, member]}, {"one": [member], "two": [member]}):
                path.write_text(json.dumps(graph | {"groups": groups}))
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_pruning_and_its_iterative_passes_honor_cancellation(self):
        with self.assertRaises(attribution.Cancelled):
            attribution._prune(None, None, None, None, None, None, None, lambda: True)
        matrix = torch.eye(3)
        weights, rows = torch.ones(3), torch.arange(3)
        for function, args in ((attribution._influence, (matrix, weights, rows)),
                               (attribution._influence_square, (matrix, weights, rows)),
                               (attribution._effect, (matrix, weights, rows, weights))):
            with self.subTest(function=function.__name__), self.assertRaises(attribution.Cancelled):
                function(*args, cancelled=lambda: True)

    def test_range_download_rejects_full_files_before_consuming_and_bounds_stream(self):
        response = mock.MagicMock(status_code=200)
        stream = mock.MagicMock()
        stream.__enter__.return_value = response
        client = mock.Mock(stream=mock.Mock(return_value=stream))
        with mock.patch("huggingface_hub.get_session", return_value=client):
            with self.assertRaisesRegex(OSError, "range response"):
                transcoders._fetch_range("https://example.test/features", 10, 14)
            response.iter_bytes.assert_not_called()
            response.status_code = 206
            response.iter_bytes.return_value = iter([b"ab", b"cd"])
            self.assertEqual(transcoders._fetch_range("https://example.test/features", 10, 14), b"abcd")
            response.iter_bytes.return_value = iter([b"abcde"])
            with self.assertRaisesRegex(OSError, "exceeded"):
                transcoders._fetch_range("https://example.test/features", 10, 14)
            response.iter_bytes.return_value = iter([b"a"])
            with self.assertRaisesRegex(OSError, "cut short"):
                transcoders._fetch_range("https://example.test/features", 10, 14)

    def test_long_prompts_are_rejected_before_transcoder_loading(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        session = SimpleNamespace(prompt_ids=lambda messages: [1] * (attribution.MAX_PREFIX + 1))
        with self.assertRaisesRegex(ValueError, "prompt tokens"):
            bench.encoded_prompt(session, dict(raw=False, user="long", system="", prefix=""))

    def test_feature_descriptions_check_cancellation_before_decoder_work(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        held = SimpleNamespace(decoder_rows=mock.Mock())
        blocks = SimpleNamespace(unembed=SimpleNamespace(weight=torch.empty(64, 2)))
        with self.assertRaises(attribution.Cancelled):
            bench._describe({"nodes": [{"kind": "feature"}]}, blocks, held, str, lambda: True)
        held.decoder_rows.assert_not_called()

    def test_reply_prefix_uses_the_retained_prompt_context(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        session = SimpleNamespace(prompt_ids=mock.Mock(return_value=[1, 2]),
                                  encode_replacement=mock.Mock(return_value=[3]),
                                  encode=mock.Mock(side_effect=AssertionError("standalone encoding")))
        self.assertEqual(bench.prompt_ids(session, None, dict(raw=False, user="hello", system="", prefix=" world")), [1, 2, 3])
        session.encode_replacement.assert_called_once_with([1, 2], " world")

    def test_templated_prefix_preparation_does_not_hold_the_model_lock(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        session = SimpleNamespace(prompt_ids=lambda messages: [1, 2], encode_replacement=lambda ids, text: [3])
        with mock.patch.object(bench, "_model", side_effect=AssertionError("model lock taken")):
            self.assertEqual(bench.encoded_prompt(session, dict(raw=False, user="hello", system="", prefix=" world")), [1, 2, 3])

    def test_imported_contrast_sides_cannot_double_count_a_token(self):
        graph = small_graph()
        target = next(n for n in graph["nodes"] if n["kind"] == "target")
        node = {**target, "target_kind": "contrast", "positive": [1, 1, 2], "negative": [3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps({**graph, "nodes": [node], "edges": []}))
            with self.assertRaisesRegex(ValueError, "not a valid"):
                workbench.load_graph(path)

    def test_imported_metadata_and_measurement_token_bounds(self):
        graph = small_graph()
        feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for change in ({"stats": "bad"}, {"stats": {"active_features": "bad"}},
                           {"explain": "bad"}, {"explain": {"tokens": [1]}}, {"prompt": "bad"}, {"prompt": {}},
                           {"tokens": ["&" * 4097] * len(graph["tokens"])},
                           {"nodes": [{**feature, "promotes": [1]}]}, {"nodes": [{**feature, "suppresses": "bad"}]},
                           {"nodes": [{**graph["nodes"][-1], "text": "&" * 4097}]}):
                path.write_text(json.dumps(graph | change))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
        blocks = SimpleNamespace(embed=SimpleNamespace(weight=torch.empty(64, 2)),
                                 unembed=SimpleNamespace(weight=torch.empty(64, 2)),
                                 layers=[None] * graph["layers"])
        workbench.Workbench._measurement_ids(graph, blocks)
        for changed in ({**graph, "ids": [64] * len(graph["ids"])},
                        {**graph, "nodes": [{"kind": "target", "target_kind": "token", "token_id": 64}]}):
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                workbench.Workbench._measurement_ids(changed, blocks)

    def test_malformed_saved_labels_and_effects_are_rejected(self):
        graph = small_graph()
        feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            effects = dict(prefixes=1, boost=2.0, every_position=False, pivot=[1], alternatives=[2],
                           baseline=dict(tokens={"1": "0.2", "2": 0.1}, pivot="0.2"), groups={},
                           token_text={"1": "hello", "2": "world"})
            path.write_text(json.dumps(graph | {"effects": effects}))
            loaded = workbench.load_graph(path)
            self.assertEqual(loaded["effects"]["baseline"]["pivot"], 0.2)
            render.group_view(loaded, {}, loaded["effects"])
            for change in ({"labels": {feature["id"]: 1}}, {"labels": {"missing": "name"}},
                           {"effects": "bad"}, {"effects": {"groups": [], "baseline": "bad"}}):
                path.write_text(json.dumps(graph | change))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_imported_effects_require_every_declared_token(self):
        for where in ("baseline", "ablate", "boost"):
            for missing in ("1", "2", "all"):
                summary = lambda: dict(tokens={"1": .2, "2": .1}, pivot=.2)
                effects = dict(prefixes=1, boost=2., every_position=False, pivot=[1], alternatives=[2],
                               baseline=summary(), groups={"g": dict(active_prefixes=1,
                                                                     ablate=summary(), boost=summary())})
                value = effects["baseline"] if where == "baseline" else effects["groups"]["g"][where]
                if missing == "all":
                    value["tokens"].clear()
                else:
                    del value["tokens"][missing]
                with self.subTest(where=where, missing=missing), self.assertRaises(ValueError):
                    workbench._validate_effects(effects, {"g": []})

    def test_interventions_require_the_traced_weight_snapshot(self):
        session = SimpleNamespace(model_id="test/tiny", model_revision="a" * 40, load_id="test/tiny#2")
        graph = dict(model_id=session.model_id, model_revision="a" * 40)
        workbench.Workbench._same_model(graph, session)
        with self.assertRaisesRegex(ValueError, "another model revision"):
            workbench.Workbench._same_model({**graph, "model_revision": "b" * 40}, session)
        unknown = dict(model_id=session.model_id, model_revision=None,
                       load_id=session.load_id, process_id=workbench.PROCESS_ID)
        workbench.Workbench._same_model(unknown, session)
        for change in (dict(load_id="test/tiny#1"), dict(process_id="previous process"), dict(process_id=None)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "another model load"):
                workbench.Workbench._same_model(unknown | change, session)

    def test_uploaded_graph_dimensions_are_bounded_before_rendering(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for change in (dict(groups={"bad": 1}), dict(groups={"bad": ["missing"]}),
                           dict(groups={"bad": [graph["nodes"][-1]["id"]]}),
                           dict(created=1e300), dict(created=float("nan")), dict(created="yesterday"),
                           dict(layers=10 ** 12), dict(layers=0), dict(layers="3"),
                           dict(tokens=["x"] * 513), dict(ids=[1]),
                           dict(nodes=[{**graph["nodes"][0], "position": 999}]),
                           dict(nodes=[{**graph["nodes"][0], "layer": 999}]),
                           dict(nodes=graph["nodes"] * 2),
                           dict(nodes=[{**graph["nodes"][-1], "id": f"target:{i}"} for i in range(4097)]),
                           dict(edges=[graph["edges"][0]] * (len(graph["nodes"]) ** 2 + 1))):
                with self.subTest(change=change):
                    path.write_text(json.dumps(graph | change))
                    with self.assertRaisesRegex(ValueError, "not a valid"):
                        workbench.load_graph(path)

    def test_feature_indices_are_bound_to_the_transcoder_width(self):
        graph = small_graph()
        feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for index in (-1, graph["transcoder_width"], "3"):
                bad = dict(graph, nodes=[{**feature, "feature": index}])
                path.write_text(json.dumps(bad))
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_saved_graphs_round_trip_and_bad_files_are_refused(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            bench = workbench.Workbench(SimpleNamespace(), directory)
            path = bench.save(graph)
            self.assertEqual(workbench.load_graph(path)["nodes"], graph["nodes"])
            self.assertEqual(len(bench.saved()), 1)
            strings = dict(graph, nodes=[{**node, "influence": str(node["influence"]), "effect": str(node["effect"])}
                                         for node in graph["nodes"]], edges=[{**edge, "weight": str(edge["weight"])} for edge in graph["edges"]])
            path.write_text(json.dumps(strings))
            loaded = workbench.load_graph(path)
            self.assertTrue(all(isinstance(edge["weight"], float) for edge in loaded["edges"]))
            render.graph_view(loaded)
            # Chosen-token traces can legitimately contain more than the ten default targets.
            chosen = dict(graph, nodes=[{**graph["nodes"][-1], "id": f"target:{i}"} for i in range(20)], edges=[])
            path.write_text(json.dumps(chosen))
            self.assertEqual(len(workbench.load_graph(path)["nodes"]), 20)
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"format": "something else"}))
            with self.assertRaisesRegex(ValueError, "not a saved"):
                workbench.load_graph(bad)
            broken = dict(graph, edges=[{"source": "nowhere", "target": "target:0", "weight": 1.0}])
            bad.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "not a valid"):
                workbench.load_graph(bad)

    def test_uploaded_graph_ids_cannot_escape_the_graphs_directory(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            bench = workbench.Workbench(SimpleNamespace(), directory)
            path = Path(directory) / "upload.json"
            for unsafe in ("../../other/file", "/tmp/overwrite", "../", "", None, ["abc"]):
                bad = dict(graph, id=unsafe)
                with self.subTest(id=unsafe):
                    path.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError, "not a valid"):
                        workbench.load_graph(path)
                    with self.assertRaisesRegex(ValueError, "safe filename"):
                        bench.save(bad)
            self.assertFalse(bench.graphs_dir().exists())

    def test_graph_mutations_cannot_save_over_a_newer_publication(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                handlers = handlers_by_name(demo)
                for name in ("group_selected", "rename_node", "delete_group"):
                    fn = handlers[name]
                    cell = dict(zip(fn.__code__.co_freevars, fn.__closure__))["save"]
                    save = cell.cell_contents
                    graph = small_graph()
                    member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
                    graph["groups"] = {"old": [member]}
                    save(graph, "view")
                    newest = {**graph, "labels": {member: "newer"}}
                    def concurrent(graph, owner, checked=False):
                        save(newest, owner)
                        return save(graph, owner, checked=checked)
                    cell.cell_contents = concurrent
                    try:
                        args = {"group_selected": (graph, [member], "new", 40, False, "view"),
                                "rename_node": (graph, member, "stale", [member], 40, False, "view"),
                                "delete_group": (graph, "old", [member], 40, False, "view")}[name]
                        result = fn(*args)
                    finally:
                        cell.cell_contents = save
                    with self.subTest(handler=name):
                        self.assertTrue(all(item == gr.skip() for item in result))
                        stored = workbench.load_graph(Path(directory) / "graphs" / f"{graph['id']}.json")
                        self.assertEqual(stored["labels"][member], "newer")
            finally:
                demo.close()

    def test_regrouping_clears_measured_effects(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                fn = handlers_by_name(demo)["group_selected"]
                graph = small_graph()
                members = [n["id"] for n in graph["nodes"] if n["kind"] == "feature"][:2]
                graph.update(groups={"old": members}, effects={"groups": {"old": {"stale": True}}})
                changed = fn(graph, members[:1], "new", 40, False, "view")[0]
                self.assertIsNone(changed["effects"])
                self.assertEqual(changed["groups"]["new"], members[:1])
                saved = workbench.load_graph(Path(directory) / "graphs" / f"{graph['id']}.json")
                self.assertIsNone(saved["effects"])
            finally:
                demo.close()

    def test_load_transcoders_cancellation_is_a_stopped_callback(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                def stopped(*args):
                    raise attribution.Cancelled()
                    yield
                with mock.patch.object(workbench.Workbench, "background", stopped):
                    frames = list(handlers_by_name(demo)["load_now"]("view"))
                self.assertEqual(frames[-1][0], "Stopped.")
                graph = small_graph()
                feature = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
                with mock.patch.object(workbench.Workbench, "background", stopped):
                    self.assertEqual(handlers_by_name(demo)["ablate_focused"]("view", graph, feature), gr.skip())
            finally:
                demo.close()

    def test_feature_details_completed_after_opening_another_graph_are_discarded(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                handlers = handlers_by_name(demo)
                old = small_graph()
                new = dict(old, id="new-graph")
                path = Path(directory) / "new.json"
                path.write_text(json.dumps(new))
                feature = next(n["id"] for n in old["nodes"] if n["kind"] == "feature")
                def fetched(*args):
                    handlers["open_path"](path, 40, False, "view")
                    return None
                with mock.patch.object(workbench.Workbench, "record", side_effect=fetched):
                    result = handlers["picked"](old, json.dumps(dict(selected=[feature], focus=feature)), "view")
                self.assertEqual(result, (gr.skip(),) * 5)
            finally:
                demo.close()

    def test_interventions_completed_after_opening_another_graph_are_discarded(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                handlers = handlers_by_name(demo)
                old = small_graph()
                old["groups"] = {"group": [next(n["id"] for n in old["nodes"] if n["kind"] == "feature")]}
                new = dict(old, id="new-graph", groups={})
                path = Path(directory) / "new.json"
                path.write_text(json.dumps(new))

                def completed(self, session_id, work):
                    handlers["open_path"](path, 40, False, session_id)
                    yield "result", {"groups": {}, "prefixes": 1}

                with mock.patch.object(workbench.Workbench, "background", completed):
                    frames = list(handlers["run_interventions"]("view", old, "", "", "", True, 2, False, "group"))
                self.assertEqual(frames[-1], (gr.skip(),) * 6)
                ticket = handlers["begin_trace"]("view")
                handlers["open_path"](path, 40, False, "view")
                with mock.patch.object(workbench.Workbench, "background", side_effect=AssertionError("stale trace work")):
                    stale_trace = list(handlers["run_trace"]("view", "", "test", "", False,
                                       "The likeliest next tokens", "", "", 16, .8, .98, 8, 40, False, ticket))
                self.assertTrue(all(value == gr.skip() for value in stale_trace[-1]))
                feature_id = old["groups"]["group"][0]
                self.assertTrue(all(value == gr.skip() for value in handlers["rename_node"](old, feature_id, "stale", [], 40, False, "view")))
                self.assertTrue(all(value == gr.skip() for value in handlers["group_selected"](old, [feature_id], "stale", 40, False, "view")))
                self.assertTrue(all(value == gr.skip() for value in handlers["delete_group"](old, "group", [], 40, False, "view")))
                self.assertFalse((Path(directory) / "graphs" / f"{old['id']}.json").exists())
                with mock.patch.object(workbench.Workbench, "background", side_effect=AssertionError("stale work")):
                    stale = list(handlers["run_interventions"]("view", old, "", "", "", True, 2, False, "group"))
                self.assertEqual(stale, [(gr.skip(),) * 6])
                with mock.patch.object(workbench.Workbench, "background", completed):
                    trace_frames = list(handlers["run_trace"]("view", "", "test", "", False,
                                      "The likeliest next tokens", "", "", 16, .8, .98, 8, 40, False))
                self.assertTrue(all(value == gr.skip() for value in trace_frames[-1]))
                # A newer graph may also open during rendering, after the result was saved.
                original = render.group_view
                def switched(*args, **kwargs):
                    if args[2] is not None:
                        handlers["open_path"](path, 40, False, "view")
                        return "old rendering"
                    return original(*args, **kwargs)
                def done(*args):
                    yield "result", {"groups": {}, "prefixes": 1}
                with mock.patch.object(workbench.Workbench, "background", done), mock.patch.object(render, "group_view", switched):
                    late_frames = list(handlers["run_interventions"]("view", old, "", "", "", True, 2, False, "group"))
                self.assertEqual(late_frames[-1], (gr.skip(),) * 6)

                features = [n["id"] for n in old["nodes"] if n["kind"] == "feature"]
                def changed_focus(*args):
                    handlers["picked"](old, json.dumps({"selected": [features[1]], "focus": features[1]}), "focus-view")
                    yield "result", {"deltas": []}
                with mock.patch.object(workbench.Workbench, "background", changed_focus):
                    self.assertEqual(handlers["ablate_focused"]("focus-view", old, features[0]), gr.skip())
                feature = next(n["id"] for n in old["nodes"] if n["kind"] == "feature")
                with mock.patch.object(workbench.Workbench, "background", completed):
                    self.assertEqual(handlers["ablate_focused"]("view", old, feature), gr.skip())
            finally:
                demo.close()

    def test_mutations_refresh_and_reuse_the_current_views_download(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                handlers = handlers_by_name(demo)
                graph = small_graph()
                members = [n["id"] for n in graph["nodes"] if n["kind"] == "feature"][:2]
                # Imported IDs need not use the publisher's f: prefix.
                old_id = members[0]
                members[0] = "imported-feature"
                for node in graph["nodes"]:
                    if node["id"] == old_id:
                        node["id"] = members[0]
                for edge in graph["edges"]:
                    for key in ("source", "target"):
                        if edge[key] == old_id:
                            edge[key] = members[0]
                grouped = handlers["group_selected"](graph, members, "group", 40, False, "view")
                offered = Path(grouped[-1])
                self.assertEqual(json.loads(offered.read_text())["groups"], {"group": members})
                renamed = handlers["rename_node"](grouped[0], members[0], "renamed", members, 40, False, "view")
                self.assertEqual(Path(renamed[-1]), offered)
                self.assertEqual(json.loads(offered.read_text())["labels"][members[0]], "renamed")
                deleted = handlers["delete_group"](renamed[0], "group", members, 40, False, "view")
                self.assertEqual(Path(deleted[-1]), offered)
                self.assertEqual(json.loads(offered.read_text())["groups"], {})
                self.assertEqual([p.name for p in offered.parent.iterdir()], ["circuit.json"])
                # Another view gets its own owner and copy.
                other = handlers["group_selected"](graph, members, "other", 40, False, "other")
                self.assertNotEqual(Path(other[-1]).parent, offered.parent)
                owner = next(component for component in demo.blocks.values()
                             if isinstance(component, gr.State)
                             and getattr(component.delete_callback, "__name__", None) == "forget")
                owner.delete_callback("view")
                self.assertFalse(offered.parent.exists())
                self.assertTrue(Path(other[-1]).exists())
            finally:
                demo.close()

    def test_background_work_reports_progress_and_errors(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())

        def work(progress, cancelled):
            progress("Step", 1, 2)
            return "result"

        items = list(bench.background("owner", work))
        self.assertEqual(items, [("progress", "Step", 1, 2), ("done", "result")])

        def fails(progress, cancelled):
            raise ValueError("broken")

        with self.assertRaisesRegex(ValueError, "broken"):
            list(bench.background("owner", fails))

    def test_stop_before_worker_registration_prevents_work(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        work = mock.Mock(return_value="result")
        bench.cancel("owner")
        with self.assertRaises(attribution.Cancelled):
            list(bench.background("owner", work))
        work.assert_not_called()
        self.assertEqual(list(bench.background("owner", work)), [("done", "result")])

    def test_one_run_per_view(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        release = threading.Event()
        first = bench.background("owner", lambda progress, cancelled: release.wait(5))
        next_item = threading.Thread(target=lambda: list(first))
        next_item.start()
        while "owner" not in bench._sessions:
            pass
        with self.assertRaisesRegex(ValueError, "already running"):
            list(bench.background("owner", lambda progress, cancelled: None))
        release.set()
        next_item.join()


class TranscoderTests(unittest.TestCase):
    def test_training_checkpoint_provenance_is_explicit_and_known_mismatches_are_refused(self):
        from dataclasses import replace
        spec = transcoders.spec_for("google/gemma-3-1b-it")
        self.assertIsNone(spec.training_model_revision)
        self.assertFalse(transcoders.check_model_revision(spec, "a" * 40))
        documented = replace(spec, training_model_revision="a" * 40)
        self.assertTrue(transcoders.check_model_revision(documented, "a" * 40))
        for loaded in (None, "b" * 40):
            with self.assertRaisesRegex(ValueError, "exact checkpoint"):
                transcoders.check_model_revision(documented, loaded)

    def test_catalogue_lookup(self):
        self.assertEqual(transcoders.spec_for("Google/Gemma-3-1B-IT").key, "gemma-3-1b-it")
        self.assertIsNone(transcoders.spec_for("allenai/Olmo-3-7B-Think"))
        self.assertIsNone(transcoders.spec_for(None))

    def test_feature_records_are_read_by_range_and_cached(self):
        record = {"index": 1, "top_logits": [" x"]}
        packed = gzip.compress(json.dumps(record).encode())
        raw = struct.pack("<I", len(packed)) + packed
        self.assertEqual(transcoders.parse_record(raw), record)
        spec = transcoders.spec_for("google/gemma-3-1b-it")
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "snapshots" / ("a" * 40) / "index.json.gz"
            index.parent.mkdir(parents=True)
            index.write_bytes(gzip.compress(json.dumps(
                {"version": "1.0", "3": {"filename": "layer_3.bin", "offsets": [0, 10, 10 + len(raw)]}}).encode()))
            records = transcoders.FeatureRecords(spec, Path(directory) / "cache")
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(index)), \
                    mock.patch.object(transcoders, "_fetch_range", return_value=raw) as fetch:
                self.assertEqual(records.get(3, 1), record | {"_transcoder_revision": "a" * 40})
                self.assertEqual(fetch.call_args.args[1:], (10, 10 + len(raw)))
                self.assertIn("/resolve/" + "a" * 40 + "/", fetch.call_args.args[0])
                self.assertEqual(records.get(3, 1), record | {"_transcoder_revision": "a" * 40})
                self.assertEqual(fetch.call_count, 1)
                with self.assertRaises(OSError):
                    records.get(3, 7)


class BrowserTests(unittest.TestCase):
    record = {"act_max": 40.0, "activation_frequency": 0.002, "top_logits": [" Paris", "<b>"],
              "bottom_logits": [" x"],
              "examples_quantiles": [{"examples": [
                  {"tokens": ["in", " France", "."], "tokens_acts_list": [0.0, 9.0, 1.0]},
                  {"tokens": [" France", " is"], "tokens_acts_list": [5.0, 0.0]},
                  {"tokens": ["<y>", "z"], "tokens_acts_list": [2.0, 0.0]},
                  {"tokens": ["bad"], "tokens_acts_list": []},
              ]}, {"examples": [{"tokens": ["later"], "tokens_acts_list": [1.0]}]}]}

    def test_top_tokens_count_each_top_examples_peak(self):
        self.assertEqual(render.top_tokens(self.record), [(" France", 2), ("<y>", 1)])
        self.assertEqual(render.top_tokens({}), [])

    def test_the_list_shows_each_feature_and_marks_the_chosen_one(self):
        rows = [(100, self.record, None), (101, None, "The Hub answered 404.")]
        html = render.feature_list(5, 100, 16384, rows, selected=100)
        self.assertIn("features 100–101 of 16,384", html)
        self.assertIn('<tr data-feature="100" class="sel">', html)
        self.assertIn("×2", html)
        self.assertIn("&lt;b&gt;", html)
        self.assertIn("0.200%", html)
        self.assertIn("The Hub answered 404.", html)
        detail = render.feature_detail(5, 100, self.record)
        self.assertIn("feature 100", detail)
        self.assertIn("40", detail)
        self.assertIn('class="peak"', detail)
        self.assertIn("unavailable", render.feature_detail(5, 101, None, "gone"))
        self.assertIn("cg-empty", render.feature_detail())

    def test_pages_stay_inside_the_layer_and_failed_records_are_kept(self):
        spec = transcoders.spec_for("google/gemma-3-1b-it")
        self.assertEqual(browser.page_start(spec, -5), 0)
        self.assertEqual(browser.page_start(spec, 10 ** 9), spec.width - browser.PAGE_SIZE)

        class Records:
            def get(self, layer, feature):
                if feature == 2:
                    raise OSError("no record")
                return {"layer": layer, "index": feature}

        rows = browser.fetch_page(Records(), 3, 0, count=4)
        self.assertEqual([r[0] for r in rows], [0, 1, 2, 3])
        self.assertEqual(rows[2], (2, None, "no record"))
        self.assertEqual(rows[3][1], {"layer": 3, "index": 3})

    def test_showing_another_page_clears_the_selected_feature_and_card(self):
        import gradio as gr
        from functools import partial
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda *args, **kwargs: None))
        records = SimpleNamespace(get=lambda layer, feature: self.record)
        bench = SimpleNamespace(records=lambda spec: records)
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            fn = next(listener.fn for listener in demo.fns.values()
                      if isinstance(listener.fn, partial) and listener.fn.func.__name__ == "list_page")
            selected = {"set": browser.DEFAULT_SET, "layer": 3, "feature": 2}
            begin = next(listener.fn for listener in demo.fns.values()
                         if getattr(listener.fn, "__name__", None) == "begin_page")
            raw_fn = fn
            fn = lambda key, layer, start, selected: raw_fn(key, layer, start, selected, "view", begin("view")[0])
            same = fn(browser.DEFAULT_SET, 3, 0, selected)
            self.assertEqual(same[3]["feature"], 2)
            for key, layer, start in ((browser.DEFAULT_SET, 4, 0),
                                      (browser.DEFAULT_SET, 3, browser.PAGE_SIZE),
                                      ("gemma-2-2b", 3, 0)):
                cleared = fn(key, layer, start, selected)
                self.assertIsNone(cleared[3])
                self.assertIn("cg-empty", cleared[4])
        finally:
            demo.close()

    def test_a_click_from_the_previous_page_is_refused(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda *args, **kwargs: None))
        records = mock.Mock()
        records.get.return_value = self.record
        bench = SimpleNamespace(records=lambda spec: records)
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            show = next(listener.fn for listener in demo.fns.values()
                        if isinstance(listener.fn, partial) and listener.fn.func.__name__ == "list_page")
            begin = handlers_by_name(demo)["begin_page"]
            raw_show = show
            show = lambda key, layer, start, selected: raw_show(key, layer, start, selected, "view", begin("view")[0])
            raw_pick = handlers_by_name(demo)["picked"]
            pick = lambda page, raw: raw_pick(page, raw, "view",
                                             handlers_by_name(demo)["begin_pick"](page, raw, "view")[0])
            old = show(browser.DEFAULT_SET, 3, 0, None)[2]
            new = show(browser.DEFAULT_SET, 4, 0, None)[2]
            records.get.reset_mock()
            self.assertEqual(pick(new, json.dumps(dict(feature=2, page_id=old["stamp"]))), (gr.skip(),) * 3)
            self.assertEqual(pick(new, json.dumps(dict(feature=99, page_id=new["stamp"]))), (gr.skip(),) * 3)
            records.get.assert_not_called()
            card, selected, _ = pick(new, json.dumps(dict(feature=2, page_id=new["stamp"])))
            self.assertEqual((selected["layer"], selected["feature"]), (4, 2))
            self.assertIn("feature 2", card)
            html = show(browser.DEFAULT_SET, 4, 0, None)[0]
            self.assertIn('data-page="', html)
        finally:
            demo.close()

    def test_failed_feature_range_retains_the_decoder_snapshot(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        vectors = []
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda button, fn, inputs, prepare=None:
                                                             vectors.append(prepare or fn)))
        records = SimpleNamespace(get=mock.Mock(return_value=self.record), revision="b" * 40)
        bench = SimpleNamespace(records=lambda spec: records)
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            callbacks = handlers_by_name(demo)
            show = next(listener.fn for listener in demo.fns.values()
                        if isinstance(listener.fn, partial) and listener.fn.func.__name__ == "list_page")
            page = show(browser.DEFAULT_SET, 3, 0, None, "view", callbacks["begin_page"]("view")[0])[2]
            raw = json.dumps(dict(feature=2, page_id=page["stamp"]))
            selected = callbacks["picked"](page, raw, "view", callbacks["begin_pick"](page, raw, "view")[0])[1]
            records.get.side_effect = OSError("range unavailable")
            with mock.patch.object(transcoders, "decoder_row", return_value=[0.5, -1.0]) as row:
                vectors[0](selected, 1, "view")
            row.assert_called_once_with(browser.spec_named(browser.DEFAULT_SET), 3, 2, revision="b" * 40)
        finally:
            demo.close()

    def test_steering_is_refused_as_soon_as_page_loading_begins(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        vectors = []
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda button, fn, inputs, prepare=None: vectors.append(prepare or fn)))
        bench = SimpleNamespace(records=lambda spec: SimpleNamespace(get=lambda *args: self.record))
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            begin = handlers_by_name(demo)["begin_page"]
            show = next(listener.fn for listener in demo.fns.values()
                        if isinstance(listener.fn, partial) and listener.fn.func.__name__ == "list_page")
            stamp = begin("view")[0]
            page = show(browser.DEFAULT_SET, 3, 0, None, "view", stamp)[2]
            selected = {**page, "feature": 2}
            next_stamp, cleared, card, disabled = begin("view")
            self.assertIsNone(cleared)
            self.assertFalse(disabled["interactive"])
            with mock.patch.object(browser, "feature_vector") as build:
                with self.assertRaisesRegex(ValueError, "Wait for the feature page"):
                    vectors[0](selected, 3, "view")
                build.assert_not_called()
            stale = show(browser.DEFAULT_SET, 3, 0, None, "view", stamp)
            self.assertEqual(stale, (gr.skip(),) * 6)
            self.assertNotEqual(stamp, next_stamp)
            # Changing controls while the old set is still fetching invalidates its request.
            switched = handlers_by_name(demo)["choose_set"]("gemma-2-2b", "view")
            self.assertFalse(switched[-1]["interactive"])
            self.assertEqual(show(browser.DEFAULT_SET, 3, 0, None, "view", next_stamp), (gr.skip(),) * 6)
        finally:
            demo.close()

    def test_steering_is_refused_when_another_row_is_picked(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        vectors = []
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda button, fn, inputs, prepare=None: vectors.append(prepare or fn)))
        bench = SimpleNamespace(records=lambda spec: SimpleNamespace(get=lambda *args: self.record))
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            handlers = handlers_by_name(demo)
            begin = handlers["begin_page"]
            show = next(listener.fn for listener in demo.fns.values()
                        if isinstance(listener.fn, partial) and listener.fn.func.__name__ == "list_page")
            page = show(browser.DEFAULT_SET, 3, 0, None, "view", begin("view")[0])[2]
            raw = lambda feature: json.dumps(dict(feature=feature, page_id=page["stamp"]))
            stamp = handlers["begin_pick"](page, raw(1), "view")[0]
            selected = handlers["picked"](page, raw(1), "view", stamp)[1]
            changed = handlers["begin_pick"](page, raw(2), "view")
            self.assertIsNone(changed[1])
            self.assertFalse(changed[2]["interactive"])
            with self.assertRaisesRegex(ValueError, "Wait for"):
                vectors[0](selected, 3, "view")
            selected = handlers["picked"](page, raw(2), "view", changed[0])[1]

            def switched_during_download(*args):
                handlers["begin_pick"](page, raw(3), "view")
                return {"vector": [1.0]}

            with mock.patch.object(browser, "feature_vector", side_effect=switched_during_download):
                with self.assertRaisesRegex(ValueError, "changed"):
                    vectors[0](selected, 3, "view")
        finally:
            demo.close()

    def test_the_vector_is_the_decoder_row_at_the_peak_activation(self):
        spec = transcoders.spec_for("google/gemma-3-1b-it")
        with mock.patch.object(transcoders, "decoder_row", return_value=[0.5, -1.0]) as row:
            vector = browser.feature_vector(spec, 7, 11, self.record, 3.0, "Google/Gemma-3-1B-IT")
            row.assert_called_once_with(spec, 7, 11)
            self.assertEqual(vector["vector"], [20.0, -40.0])
            self.assertEqual((vector["layer"], vector["strength"], vector["enabled"]), (7, 3.0, True))
            self.assertEqual(vector["model_id"], "Google/Gemma-3-1B-IT")
            inactive = browser.feature_vector(spec, 7, 11, {"act_max": 0}, 3.0, "Google/Gemma-3-1B-IT")
            self.assertEqual(inactive["vector"], [0.0, 0.0])
            browser.feature_vector(spec, 7, 11, self.record | {"_transcoder_revision": "a" * 40}, 3.0, None)
            row.assert_called_with(spec, 7, 11, revision="a" * 40)
            # No record: the row as published. Another model loaded: the set's own.
            vector = browser.feature_vector(spec, 7, 11, None, 1.0, "Qwen/Qwen3-0.6B")
            self.assertEqual(vector["vector"], [0.5, -1.0])
            self.assertEqual(vector["model_id"], "google/gemma-3-1b-it")

    def test_one_decoder_row_is_read_from_the_layer_file(self):
        from safetensors.torch import save_file

        spec = transcoders.TranscoderSpec("tiny", "Tiny", ("org/tiny",), "org/tiny-tc", "", layers=2, width=3,
                                          d_model=2, weights_gb=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer_1.safetensors"
            save_file({"W_dec": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}, str(path))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
                self.assertEqual(transcoders.decoder_row(spec, 1, 2), [5.0, 6.0])
                self.assertEqual(download.call_args.args, ("org/tiny-tc", "layer_1.safetensors"))
                with self.assertRaisesRegex(ValueError, "features 0–2"):
                    transcoders.decoder_row(spec, 1, 3)
                with self.assertRaisesRegex(ValueError, "layers 0–1"):
                    transcoders.decoder_row(spec, 2, 0)
            save_file({"W_dec": torch.zeros(3, 5)}, str(path))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                with self.assertRaisesRegex(ValueError, "shape"):
                    transcoders.decoder_row(spec, 1, 0)


class ModelAccessTests(unittest.TestCase):
    def manager(self, backend="torch", precision="full"):
        manager = FakeManager()
        manager._lock = threading.Lock()
        manager.model = object()
        manager.precision = precision
        manager._engine = lambda: SimpleNamespace(backend=backend)
        manager._release_device_cache = mock.Mock()
        return manager

    def test_the_model_is_held_under_the_lock(self):
        manager = self.manager()
        with ModelService(lambda: manager).open_session() as session:
            with session.transformers_model() as model:
                self.assertIs(model, manager.model)
                self.assertTrue(manager._lock.locked())
            self.assertFalse(manager._lock.locked())
        manager._release_device_cache.assert_called_once()

    def test_trace_reads_revision_before_taking_the_model_lock(self):
        manager = self.manager()
        manager.model = tiny_model("qwen3")
        manager.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))
        def revision():
            self.assertFalse(manager._lock.locked(), "revision lookup must not recursively acquire the model lock")
            return "a" * 40
        manager.model_revision = revision
        models = SimpleNamespace(open_session=ModelService(lambda: manager).open_session)
        bench = workbench.Workbench(models, tempfile.gettempdir())
        blocks = architecture.blocks(manager.model)
        held = tiny_transcoders(blocks)
        with mock.patch.object(bench, "_held", return_value=(blocks, held, held.spec)), \
                mock.patch.object(bench, "encoded_prompt", return_value=IDS), \
                mock.patch.object(bench, "_describe"):
            graph = bench.trace({}, {"mode": "top"}, attribution.Settings(max_feature_nodes=16, batch_size=8),
                                lambda *args: None, lambda: False)
            self.assertEqual(graph["model_revision"], "a" * 40)
            self.assertIsNone(graph["training_model_revision"])
            self.assertEqual(graph["checkpoint_compatibility"], "unverified")
            feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
            self.assertIn("deltas", bench.ablate(graph, feature, lambda *args: None, lambda: False))

    def test_revision_can_be_read_while_the_model_lock_is_held(self):
        from chatlab.extension_api import ModelService
        manager = self.manager()
        with mock.patch.object(manager, "model_revision", return_value="snapshot", create=True) as revision:
            with ModelService(lambda: manager).open_session() as session:
                with session.transformers_model():
                    self.assertEqual(session.model_revision, "snapshot")
            revision.assert_called_once()

    def test_mlx_and_quantized_loads_are_refused(self):
        for manager, message in ((self.manager(backend="mlx"), "MLX"), (self.manager(precision="4-bit"), "full")):
            with ModelService(lambda: manager).open_session() as session:
                with self.assertRaisesRegex(ValueError, message):
                    with session.transformers_model():
                        pass

    def test_a_changed_load_is_refused(self):
        manager = self.manager()
        with ModelService(lambda: manager).open_session() as session:
            manager.load_id = "second"
            with self.assertRaisesRegex(ValueError, "changed"):
                with session.transformers_model():
                    pass


class RegistryTests(unittest.TestCase):
    def test_circuits_loads_with_its_script(self):
        loaded, errors = load_enabled({"circuits"})
        self.assertEqual(errors, [])
        self.assertEqual([e.spec.id for e in loaded], ["circuits"])
        self.assertIn("circuits-pick", loaded[0].js)
        self.assertIn("circuits-feature-pick", loaded[0].js)


if __name__ == "__main__":
    unittest.main()
