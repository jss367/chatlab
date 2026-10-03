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

    def test_sharded_and_offloaded_maps_are_refused_before_transcoder_load(self):
        model = tiny_model("gemma3")
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        for mapping in ({"embed": 0, "layer": 1}, {"embed": "cpu", "layer": "disk"},
                        {"embed": "cpu", "layer": "cuda:0"}):
            model.hf_device_map = mapping
            with self.subTest(mapping=mapping), mock.patch.object(transcoders, "load") as load:
                with self.assertRaisesRegex(ValueError, "sharded|offloaded"):
                    bench._held(model, lambda *_args: None)
                load.assert_not_called()
        model.hf_device_map = {"": "cpu"}
        self.assertEqual(architecture.blocks(model).device, torch.device("cpu"))

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

    def test_half_precision_decoder_edges_and_error_reconstruct_the_same_output(self):
        for dtype in (torch.float16, torch.bfloat16):
            model = tiny_model("gemma3")
            blocks = architecture.blocks(model)
            held = tiny_transcoders(blocks)
            held.w_dec = [weight.to(dtype) for weight in held.w_dec]
            recording = attribution.record(blocks, held, IDS)
            for layer, (start, end) in enumerate(recording.layer_slices):
                acts = torch.zeros(len(IDS), WIDTH)
                acts[recording.feature_position[start:end], recording.feature_index[start:end]] = recording.activation[start:end]
                reconstruction = acts @ held.decoder_rows(layer, torch.arange(WIDTH)) + held.b_dec[layer].float()
                with self.subTest(dtype=dtype, layer=layer):
                    torch.testing.assert_close(reconstruction + recording.errors[layer], recording.outputs[layer].float(),
                                               rtol=1e-5, atol=1e-5)
            targets = attribution.choose_targets(recording, self.decode)
            directions = attribution.logit_directions(blocks, recording, targets)[:4]
            graph = attribution.FrozenGraph(blocks, held, recording, batch_size=4)
            try:
                vectors = [("vector", direction) for direction in directions]
                rows, biases = graph.rows(vectors), graph.bias_terms(vectors)
            finally:
                graph.close()
            torch.testing.assert_close(rows.sum(1) + biases, directions @ recording.final[-1], rtol=1e-4, atol=1e-4)

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

    def test_half_precision_encoders_preserve_feature_activation_completeness(self):
        for kind in ("gemma3", "qwen3"):
            for dtype in (torch.float16, torch.bfloat16):
                blocks = architecture.blocks(tiny_model(kind))
                held = tiny_transcoders(blocks, bias=False)
                held.w_enc = [weight.to(dtype) for weight in held.w_enc]
                recording = attribution.record(blocks, held, IDS)
                late = torch.nonzero(recording.feature_layer >= 1, as_tuple=True)[0][:6].tolist()
                self.assertTrue(late)
                graph = attribution.FrozenGraph(blocks, held, recording, batch_size=8)
                try:
                    rows = graph.rows([("feature", k) for k in late])
                finally:
                    graph.close()
                with self.subTest(kind=kind, dtype=dtype):
                    for row, k in zip(rows, late):
                        layer, feature = int(recording.feature_layer[k]), int(recording.feature_index[k])
                        expected = recording.activation[k] - held.b_enc[layer][feature]
                        torch.testing.assert_close(row.sum(), expected, rtol=1e-4, atol=1e-4)

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

    def test_intervention_encoder_matches_trace_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            self.held.w_enc[0] = self.held.w_enc[0].to(dtype)
            expected = []
            def record_input(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                expected.append(self.held.pre_activations(0, value[0])[:, 0])
            handle = self.blocks.mlp_input(0).register_forward_hook(record_input)
            try:
                with mock.patch.object(self.held, "activate_one", wraps=self.held.activate_one) as activate:
                    interventions.run(self.blocks, self.held, IDS, [(0, 0, 0., None)])
                with self.subTest(dtype=dtype):
                    torch.testing.assert_close(activate.call_args.args[2], expected[0], rtol=1e-6, atol=1e-6)
            finally:
                handle.remove()

    def test_rare_intervention_probabilities_do_not_underflow_in_float32(self):
        from dataclasses import replace
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            logits = torch.full((1, 64), -120., device=device)
            logits[..., 1] = 0.
            blocks = replace(self.blocks, unembed=lambda _hidden: logits, final_softcap=None)
            result = interventions.run(blocks, self.held, IDS, tokens=[0, 1])
            with self.subTest(device=device):
                self.assertGreater(result["probabilities"][0], 0.)
                expected = torch.tensor(-120., dtype=torch.float64).exp().item()
                self.assertAlmostEqual(result["probabilities"][0] / expected, 1.)

    def test_duplicate_measurement_tokens_are_saved_once_in_first_seen_order(self):
        member = (int(self.recording.feature_layer[0]), int(self.recording.feature_index[0]), 0)
        result = interventions.group_effects(self.blocks, self.held, [IDS], {"g": [member]},
                                              pivot=[3, 5, 3], alternatives=[7, 3, 7, 9, 5])
        expected = interventions.group_effects(self.blocks, self.held, [IDS], {"g": [member]},
                                                pivot=[3, 5], alternatives=[7, 9])
        self.assertEqual(result, expected)
        self.assertEqual(result["pivot"], [3, 5])
        self.assertEqual(result["alternatives"], [7, 9])
        self.assertEqual(list(result["baseline"]["tokens"]), [3, 5, 7, 9])

    def test_automatic_alternatives_reuse_each_prefix_baseline(self):
        member = (int(self.recording.feature_layer[0]), int(self.recording.feature_index[0]), 0)
        prefixes = [IDS, IDS[:5]]
        updates = []
        with mock.patch.object(interventions, "run", wraps=interventions.run) as passes:
            result = interventions.group_effects(self.blocks, self.held, prefixes, {"g": [member]},
                                                  pivot=[3], alternatives=[], auto_alternatives=True,
                                                  progress=lambda done, total: updates.append((done, total)))
            self.assertEqual(passes.call_count, 6)
        self.assertEqual(updates, [(i, 6) for i in range(1, 7)])
        self.assertNotIn(3, result["alternatives"])
        explicit = interventions.group_effects(self.blocks, self.held, prefixes, {"g": [member]},
                                                pivot=[3], alternatives=result["alternatives"])
        self.assertEqual(result, explicit)

    def test_stopped_intervention_interrupts_layers_and_restores_hooks(self):
        stopped = False
        def stop(*_args):
            nonlocal stopped
            stopped = True
        handle = self.blocks.layers[0].register_forward_hook(stop)
        try:
            later = self.blocks.layers[1]
            with mock.patch.object(later, "forward", wraps=later.forward) as forward:
                with self.assertRaises(attribution.Cancelled):
                    interventions.run(self.blocks, self.held, IDS, cancelled=lambda: stopped)
                forward.assert_not_called()
        finally:
            handle.remove()
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))
        stopped = False
        self.assertIn("log_probs", interventions.run(self.blocks, self.held, IDS, cancelled=lambda: stopped))

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

    def test_target_rendering_is_bounded_without_dropping_saved_targets(self):
        graph = dict(self.graph)
        target = next(n for n in graph["nodes"] if n["kind"] == "target")
        targets = [dict(target, id=f"target-{i}", token_id=i, text="<&" * 2048) for i in range(4096)]
        graph["nodes"] = [n for n in graph["nodes"] if n["kind"] != "target"] + targets
        view = render.graph_view(graph, selected=[n["id"] for n in targets])
        self.assertEqual(view.count('cg-node target'), 64)
        self.assertIn('Showing 64 of 4,096 targets.', view)
        self.assertLess(len(view), 250000)
        self.assertEqual(sum(n["kind"] == "target" for n in graph["nodes"]), 4096)

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

    def test_all_measured_token_bars_fit_inside_group_svg(self):
        import re
        members = [n["id"] for n in self.graph["nodes"] if n["kind"] == "feature"][:1]
        for pivot, alternatives in ((list(range(20)), []), ([], list(range(30))),
                                    (list(range(20)), list(range(20, 40)))):
            effects = dict(pivot=pivot, alternatives=alternatives, groups={}, prefixes=1,
                           baseline=dict(tokens={str(t): .01 for t in pivot + alternatives}))
            view = render.group_view(self.graph, {"g": members}, effects)
            height = float(re.search(r'viewBox="0 0 [^ ]+ ([^"]+)"', view)[1])
            positions = [float(y) for y in re.findall(r'<text[^>]+ y="([^"]+)"', view)]
            self.assertLess(max(positions) + 12, height)
            self.assertEqual(view.count('<rect class="bar '), len(pivot) + len(alternatives))

    def test_imported_group_card_bounds_feature_chips_and_labels(self):
        nodes = [dict(id=str(i), kind="feature", layer=0, feature=i, pos=0) for i in range(4096)]
        graph = dict(nodes=nodes)
        members = [n["id"] for n in nodes]
        labels = {m: '<&"' * 4096 for m in members}
        card = render.group_card("group", members, graph, labels=labels)
        self.assertEqual(card.count('<span title='), 64)
        self.assertIn('Showing 64 of 4,096 features.', card)
        self.assertLess(len(card), 65000)
        self.assertNotIn('<&"', card)
        self.assertIn('…', card)

    def test_group_box_display_is_bounded_without_mutating_saved_groups(self):
        import re
        feature = next(n for n in self.graph["nodes"] if n["kind"] == "feature")
        nodes = [dict(feature, id=f"feature{i}", layer=0, feature=i) for i in range(4096)]
        graph = dict(self.graph, nodes=nodes, edges=[])
        groups = {f"group{i}": [n["id"]] for i, n in enumerate(nodes)}
        view = render.group_view(graph, groups)
        self.assertEqual(view.count('<g class="cg-group '), 64)
        self.assertIn("Showing 64 of 4,096 groups.", view)
        height = float(re.search(r'viewBox="0 0 [^ ]+ ([^"]+)"', view)[1])
        self.assertLess(height, 6000)
        self.assertLess(len(view), 50000)
        self.assertEqual(len(groups), 4096)
        self.assertEqual(len(graph["nodes"]), 4096)

    def test_large_intervention_display_keeps_all_measurements_but_bounds_dom(self):
        import re
        members = [n["id"] for n in self.graph["nodes"] if n["kind"] == "feature"][:1]
        ids = list(range(8192))
        tokens = {str(token): .0001 for token in ids}
        result = dict(tokens=tokens, pivot=.4096)
        effects = dict(pivot=ids[:4096], alternatives=ids[4096:], prefixes=1, boost=2.,
                       baseline=result, groups={"g": dict(ablate=result, boost=result, active_prefixes=1)})
        view = render.group_view(self.graph, {"g": members}, effects)
        card = render.group_card("g", members, self.graph, effects)
        self.assertEqual(view.count('<rect class="bar '), 128)
        self.assertEqual(card.count('<div class="cg-mult">'), 256)
        self.assertEqual(view.count('Showing 64 of 4,096 tokens'), 2)
        self.assertEqual(card.count('Showing 128 of 8,192 measured tokens.'), 2)
        height = float(re.search(r'viewBox="0 0 [^ ]+ ([^"]+)"', view)[1])
        self.assertLess(height, 3100)
        self.assertEqual(len(effects["baseline"]["tokens"]), 8192)
        self.assertEqual(len(effects["pivot"]), 4096)

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

    def test_intervention_group_and_forward_budgets_precede_model_reservation(self):
        opened = mock.Mock(side_effect=AssertionError("model session acquired"))
        bench = workbench.Workbench(SimpleNamespace(open_session=opened), tempfile.gettempdir())
        for groups, prefixes in ((33, 1), (32, 16), (2, 256)):
            graph = {"groups": {str(i): ["feature"] for i in range(groups)}}
            with self.subTest(groups=groups, prefixes=prefixes), self.assertRaisesRegex(ValueError, "groups|passes"):
                bench.group_effects(graph, ["1"], [], ["prefix"] * (prefixes - 1), True, 2, False,
                                    lambda *_: None, lambda: False)
        opened.assert_not_called()
        interventions.check_workload(256, 1)
        interventions.check_workload(15, 32)
        with mock.patch.object(interventions, "run", side_effect=AssertionError("model forward")) as run:
            with self.assertRaisesRegex(ValueError, "32 nonempty groups"):
                interventions.group_effects(None, None, [[1]], {str(i): [] for i in range(33)}, [1], [])
            run.assert_not_called()

    def test_intervention_raw_bounds_precede_session_and_encoding(self):
        opened = mock.Mock(side_effect=AssertionError("model session acquired"))
        bench = workbench.Workbench(SimpleNamespace(open_session=opened), tempfile.gettempdir())
        for pivot, alternatives, prefixes in ((["1"] * 4097, [], []), (["1"], ["2"] * 4097, []),
                                               (["1"], [], ["x"] * 256), (["x" * 32769], [], []),
                                               (["1"], [], ["x" * 32769])):
            with self.subTest(pivot=len(pivot), alternatives=len(alternatives), prefixes=len(prefixes)), \
                    self.assertRaisesRegex(ValueError, "at most"):
                bench.group_effects({}, pivot, alternatives, prefixes, True, 2., False,
                                    lambda *args: None, lambda: False)
        opened.assert_not_called()

    def test_cancellation_interrupts_intervention_prefix_encoding(self):
        stopped = [False]
        session = SimpleNamespace(encode=lambda text: [1])
        bench = workbench.Workbench(SimpleNamespace(open_session=lambda: contextlib.nullcontext(session)),
                                   tempfile.gettempdir())
        def encoded(*args):
            stopped[0] = True
            return IDS
        with mock.patch.object(bench, "_same_model"), \
                mock.patch.object(bench, "encoded_prompt", side_effect=encoded) as encoding, \
                mock.patch.object(bench, "_model", side_effect=AssertionError("model loaded")):
            with self.assertRaises(attribution.Cancelled):
                bench.group_effects({"groups": {"g": []}}, ["1"], [], ["first", "second"], False,
                                    2., False, lambda *args: None, lambda: stopped[0])
            self.assertEqual(encoding.call_count, 1)

    def test_reusing_group_name_preserves_its_previous_members(self):
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
                group = handlers_by_name(demo)["group_selected"]
                graph = small_graph()
                members = [n["id"] for n in graph["nodes"] if n["kind"] == "feature"][:2]
                first = group(graph, members[:1], "named", 40, False, "view")[0]
                second = group(first, members[1:], "named", 40, False, "view")[0]
                self.assertEqual(second["groups"]["named"], members)
                self.assertEqual(first["groups"]["named"], members[:1])
                repeated = group(second, members[:1], "named", 40, False, "view")[0]
                self.assertEqual(repeated["groups"]["named"], members)
                # A gap in automatic names must not merge into another group.
                graph = small_graph()
                graph["groups"] = {"group 2": members[:1]}
                fresh = group(graph, members[1:], "", 40, False, "fresh")[0]
                self.assertEqual(fresh["groups"], {"group 2": members[:1], "group 1": members[1:]})
            finally:
                demo.close()

    def test_explanation_entry_bounds_precede_session_and_honor_cancellation(self):
        opened = mock.Mock(side_effect=AssertionError("model session acquired"))
        bench = workbench.Workbench(SimpleNamespace(open_session=opened), tempfile.gettempdir())
        for key in ("tokens", "others"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "4096"):
                bench.trace({}, {"mode": "contrast", "tokens": ["1"], "others": ["2"], key: ["x" * 32769]},
                            attribution.Settings(), lambda *args: None, lambda: False)
        opened.assert_not_called()
        stopped = [False]
        def encode(text):
            stopped[0] = True
            return [1]
        bench.models = SimpleNamespace(open_session=lambda: contextlib.nullcontext(
            SimpleNamespace(model_revision=None, encode=encode)))
        with mock.patch.object(bench, "encoded_prompt", return_value=IDS), \
                mock.patch.object(bench, "_model", side_effect=AssertionError("model loaded")):
            with self.assertRaises(attribution.Cancelled):
                bench.trace({}, {"mode": "tokens", "tokens": ["1", "2"]}, attribution.Settings(),
                            lambda *args: None, lambda: stopped[0])

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

    def test_imported_contrast_others_text_is_bounded_and_preserved(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for others in ("malformed", [1], ["x"] * 4097, ["x" * 4097]):
                path.write_text(json.dumps(graph | {"explain": dict(mode="contrast", tokens=[" yes"], others=others)}))
                with self.subTest(others_type=type(others).__name__), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            explain = dict(mode="contrast", tokens=[" yes"], others=[" no"])
            path.write_text(json.dumps(graph | {"explain": explain}))
            self.assertEqual(workbench.load_graph(path)["explain"], explain)

    def test_imported_feature_counts_influence_and_target_positions_are_physical(self):
        graph = small_graph()
        kept = sum(n["kind"] == "feature" for n in graph["nodes"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for changes in (dict(stats=graph["stats"] | {"kept_features": kept + 1}),
                            dict(stats=dict(kept_features=kept, traced_features=kept - 1, active_features=kept)),
                            dict(stats=dict(kept_features=kept, traced_features=kept + 1, active_features=kept)),
                            dict(nodes=[dict(n, influence=-.1) for n in graph["nodes"]]),
                            dict(nodes=[dict(n, position=0) if n["kind"] == "target" else n for n in graph["nodes"]])):
                path.write_text(json.dumps(graph | changes))
                with self.subTest(fields=list(changes)), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            path.write_text(json.dumps(graph | {"stats": {}}))
            self.assertEqual(workbench.load_graph(path)["stats"], dict(error_share=0., kept_features=kept, traced_features=kept, active_features=kept))
            path.write_text(json.dumps(graph | {"nodes": [dict(n, effect=-1.) for n in graph["nodes"]]}))
            self.assertTrue(all(n["effect"] == -1. for n in workbench.load_graph(path)["nodes"]))

    def test_imported_checkpoint_claims_are_normalized_against_the_catalogue(self):
        from dataclasses import replace
        graph = small_graph()
        claims = dict(training_model_revision="f" * 40, checkpoint_compatibility="verified")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps(graph | claims))
            loaded = workbench.load_graph(path)
            self.assertIsNone(loaded["training_model_revision"])
            self.assertEqual(loaded["checkpoint_compatibility"], "unverified")
            spec = replace(transcoders.CATALOGUE[0], key="tiny", model_ids=("test/tiny",), training_model_revision="a" * 40)
            with mock.patch.object(transcoders, "CATALOGUE", (spec,)):
                for revision, model, expected in (("a" * 40, "test/tiny", "verified"),
                                                   ("b" * 40, "test/tiny", "unverified"),
                                                   (None, "test/tiny", "unverified"),
                                                   ("a" * 40, "other/model", "unverified")):
                    path.write_text(json.dumps(graph | claims | dict(model_revision=revision, model_id=model)))
                    loaded = workbench.load_graph(path)
                    self.assertEqual(loaded["training_model_revision"], "a" * 40)
                    self.assertEqual(loaded["checkpoint_compatibility"], expected)

    def test_imported_targets_cannot_alias_the_same_objective(self):
        graph = small_graph()
        target = next(n for n in graph["nodes"] if n["kind"] == "target")
        token = dict(target, target_kind="token", token_id=1)
        contrast = dict(target, target_kind="contrast", positive=[1, 2], negative=[3, 4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for original, alias in ((token, dict(token, id="alias")),
                                     (contrast, dict(contrast, id="alias", positive=[2, 1], negative=[4, 3]))):
                path.write_text(json.dumps(graph | {"nodes": [original, alias], "edges": [], "stats": {}}))
                with self.subTest(kind=original["target_kind"]), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            path.write_text(json.dumps(graph | {"nodes": [token, dict(token, id="distinct", token_id=2)], "edges": [], "stats": {}}))
            self.assertEqual(len(workbench.load_graph(path)["nodes"]), 2)

    def test_imported_distinct_token_targets_share_one_probability_budget(self):
        graph = small_graph()
        target = next(n for n in graph["nodes"] if n["kind"] == "target")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for a, b, valid in ((.9, .9, False), (.6, .4000005, True), (.6, .400002, False)):
                nodes = [dict(target, id="a", target_kind="token", token_id=1, probability=a),
                         dict(target, id="b", target_kind="token", token_id=2, probability=b)]
                path.write_text(json.dumps(graph | {"nodes": nodes, "edges": [], "stats": {}}))
                with self.subTest(probabilities=(a, b)):
                    if valid:
                        self.assertEqual(len(workbench.load_graph(path)["nodes"]), 2)
                    else:
                        with self.assertRaisesRegex(ValueError, "not a valid"):
                            workbench.load_graph(path)

    def test_feature_aliases_cannot_duplicate_imported_coordinates(self):
        graph = small_graph()
        feature = next(n for n in graph["nodes"] if n["kind"] == "feature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps(graph | {"nodes": graph["nodes"] + [dict(feature, id="alias")]}))
            with self.assertRaisesRegex(ValueError, "not a valid"):
                workbench.load_graph(path)

    def test_residual_aliases_cannot_duplicate_imported_coordinates(self):
        graph = small_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            for kind in ("embedding", "error"):
                node = next(n for n in graph["nodes"] if n["kind"] == kind)
                # Keep just two nodes of this kind so cardinality limits do
                # not reject the alias before semantic-coordinate validation.
                nodes = [n for n in graph["nodes"] if n["kind"] != kind] + [node, dict(node, id="alias")]
                identifiers = {n["id"] for n in nodes}
                edges = [e for e in graph["edges"] if e["source"] in identifiers and e["target"] in identifiers]
                path.write_text(json.dumps(graph | {"nodes": nodes, "edges": edges}))
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "not a valid"):
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
            for boost in (-1., 100.1):
                path.write_text(json.dumps(graph | {"effects": effects | {"boost": boost}}))
                with self.subTest(boost=boost), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
            for boost in (0., 100.):
                path.write_text(json.dumps(graph | {"effects": effects | {"boost": boost}}))
                self.assertEqual(workbench.load_graph(path)["effects"]["boost"], boost)
            for change in ({"labels": {feature["id"]: 1}}, {"labels": {"missing": "name"}},
                           {"effects": "bad"}, {"effects": {"groups": [], "baseline": "bad"}}):
                path.write_text(json.dumps(graph | change))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)

    def test_oversized_group_map_is_rejected_before_effect_validation(self):
        graph = small_graph()
        graph["groups"] = {f"g{i}": [] for i in range(4097)}
        graph["effects"] = {"groups": {name: {} for name in graph["groups"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            path.write_text(json.dumps(graph))
            with mock.patch.object(workbench, "_validate_effects", side_effect=AssertionError("effects traversed")) as validate:
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    workbench.load_graph(path)
                validate.assert_not_called()

    def test_imported_effect_token_lists_must_be_unique_and_disjoint(self):
        for pivot, alternatives in (([1, 1], [2]), ([1], [2, 2]), ([1], [1, 2])):
            summary = dict(tokens={"1": .2, "2": .1}, pivot=.2)
            effects = dict(prefixes=1, boost=2., every_position=False, pivot=pivot, alternatives=alternatives,
                           baseline=summary, groups={"g": dict(active_prefixes=1, ablate=summary, boost=summary)})
            with self.subTest(pivot=pivot, alternatives=alternatives), self.assertRaises(ValueError):
                workbench._validate_effects(effects, {"g": []})

    def test_imported_measured_effects_need_a_pivot(self):
        summary = dict(tokens={"2": .1}, pivot=0.)
        effects = dict(prefixes=1, boost=2., every_position=False, pivot=[], alternatives=[2],
                       baseline=summary, groups={"g": dict(active_prefixes=1, ablate=summary, boost=summary)})
        with self.assertRaises(ValueError):
            workbench._validate_effects(effects, {"g": []})

    def test_imported_token_probability_sets_are_consistent_distributions(self):
        for where in ("baseline", "ablate", "boost"):
            def summary():
                return dict(tokens={"1": .2, "2": .1}, pivot=.2)
            effects = dict(prefixes=1, boost=2., every_position=False, pivot=[1], alternatives=[2],
                           baseline=summary(), groups={"g": dict(active_prefixes=1,
                                                                 ablate=summary(), boost=summary())})
            value = effects["baseline"] if where == "baseline" else effects["groups"]["g"][where]
            value.update(tokens={"1": .6, "2": .6}, pivot=.6)
            with self.subTest(where=where), self.assertRaises(ValueError):
                workbench._validate_effects(effects, {"g": []})
            value.update(tokens={"1": .5, "2": .5000005}, pivot=.5)
            workbench._validate_effects(effects, {"g": []})
            value.update(tokens={1: .3, "1": .3, "2": .1}, pivot=.6)
            with self.subTest(aliased=where), self.assertRaises(ValueError):
                workbench._validate_effects(effects, {"g": []})

    def test_imported_pivot_totals_must_match_the_token_probabilities(self):
        for where in ("baseline", "ablate", "boost"):
            summary = lambda: dict(tokens={"1": .2, "2": .1}, pivot=.2)
            effects = dict(prefixes=1, boost=2., every_position=False, pivot=[1], alternatives=[2],
                           baseline=summary(), groups={"g": dict(active_prefixes=1,
                                                                 ablate=summary(), boost=summary())})
            value = effects["baseline"] if where == "baseline" else effects["groups"]["g"][where]
            value["pivot"] = .9
            with self.subTest(where=where), self.assertRaises(ValueError):
                workbench._validate_effects(effects, {"g": []})
            value["pivot"] = .2 * (1 + 1e-8)
            workbench._validate_effects(effects, {"g": []})

    def test_focused_ablation_stops_before_its_second_pass(self):
        stopped = False
        graph = small_graph()
        node = next(n for n in graph["nodes"] if n["kind"] == "feature")
        session = SimpleNamespace(model_revision=None)
        bench = workbench.Workbench(SimpleNamespace(open_session=lambda: contextlib.nullcontext(session)),
                                    tempfile.gettempdir())
        blocks = architecture.blocks(tiny_model("gemma3"))
        held = tiny_transcoders(blocks)
        def baseline(*_args, **_kwargs):
            nonlocal stopped
            stopped = True
            return {"log_probs": torch.zeros(64)}
        with mock.patch.object(bench, "_same_model"), \
                mock.patch.object(bench, "_same_transcoders"), \
                mock.patch.object(bench, "_model", return_value=contextlib.nullcontext(None)), \
                mock.patch.object(bench, "_held", return_value=(blocks, held, None)), \
                mock.patch.object(interventions, "run", side_effect=baseline) as passes:
            with self.assertRaises(attribution.Cancelled):
                bench.ablate(graph, node, lambda *_args: None, lambda: stopped)
            self.assertEqual(passes.call_count, 1)

    def test_inactive_explanation_text_cannot_exceed_saved_schema_bounds(self):
        models = SimpleNamespace(open_session=mock.Mock(side_effect=AssertionError("model acquired")))
        bench = workbench.Workbench(models, tempfile.gettempdir())
        for field in ("tokens", "others"):
            explain = dict(mode="top", tokens=[], others=[])
            explain[field] = ["x" * 4097]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "4096 characters"):
                bench.trace({}, explain, attribution.Settings(), lambda *_: None, lambda: False)
        models.open_session.assert_not_called()

    def test_duplicate_trace_entries_are_bounded_before_tokenization(self):
        models = SimpleNamespace(open_session=mock.Mock(side_effect=AssertionError("must not tokenize")))
        bench = workbench.Workbench(models, tempfile.gettempdir())
        for field in ("tokens", "others"):
            explain = dict(mode="tokens", tokens=["same"], others=[])
            explain[field] = ["same"] * 4097
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "4,096 explanation entries"):
                bench.trace({}, explain, attribution.Settings(), lambda *_args: None, lambda: False)
        models.open_session.assert_not_called()

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
            chosen = dict(graph, nodes=[{**graph["nodes"][-1], "id": f"target:{i}", "token_id": i} for i in range(20)], edges=[], stats={})
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

    def test_ablation_cannot_stamp_old_graph_with_a_newly_opened_version(self):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits import page
        from ui_support import handlers_by_name
        with tempfile.TemporaryDirectory() as directory:
            context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                       NavigationService(lambda *args: None, lambda *args: None))
            with gr.Blocks() as demo:
                page.build_page(context)
            try:
                handlers = handlers_by_name(demo)
                old = small_graph()
                member = next(n["id"] for n in old["nodes"] if n["kind"] == "feature")
                old = handlers["group_selected"](old, [member], "old", 40, False, "view")[0]
                newer = small_graph()
                newer["id"] = "newer"
                path = Path(directory) / "newer.json"
                path.write_text(json.dumps(newer))
                fn = handlers["ablate_focused"]
                cell = dict(zip(fn.__code__.co_freevars, fn.__closure__))["node_of"]
                original = cell.cell_contents
                opened = False
                def opening(graph, node):
                    nonlocal opened
                    if not opened:
                        opened = True
                        handlers["open_path"](path, 40, False, "view")
                    return original(graph, node)
                def measured(*args):
                    yield ("done", None)
                cell.cell_contents = opening
                try:
                    with mock.patch.object(workbench.Workbench, "background", measured):
                        result = handlers["ablate_focused"]("view", old, member)
                finally:
                    cell.cell_contents = original
                self.assertTrue(opened)
                self.assertEqual(result, gr.skip())
            finally:
                demo.close()

    def test_stale_focus_and_redraw_do_not_start_or_repaint_old_work(self):
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
                graph = handlers["group_selected"](graph, members, "group", 40, False, "view")[0]
                with mock.patch.object(workbench.Workbench, "record", return_value=None):
                    handlers["picked"](graph, json.dumps(dict(selected=[members[1]], focus=members[1])), "view")
                with mock.patch.object(workbench.Workbench, "background") as work:
                    self.assertEqual(handlers["ablate_focused"]("view", graph, members[0]), gr.skip())
                work.assert_not_called()
                newer = small_graph()
                newer["id"] = "newer"
                path = Path(directory) / "newer.json"
                path.write_text(json.dumps(newer))
                original = render.graph_view
                opened = False
                def rendering(*args, **kwargs):
                    nonlocal opened
                    if not opened:
                        opened = True
                        handlers["open_path"](path, 40, False, "view")
                    return original(*args, **kwargs)
                with mock.patch.object(render, "graph_view", side_effect=rendering):
                    self.assertEqual(handlers["redraw"](graph, members, 40, False, "view"), gr.skip())
                with mock.patch.object(render, "graph_view") as drawn:
                    self.assertEqual(handlers["redraw"](graph, members, 40, False, "view"), gr.skip())
                drawn.assert_not_called()
            finally:
                demo.close()

    def test_group_selection_does_not_repaint_replaced_graphs(self):
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
                original = render.group_card
                for name in ("choose_group", "group_clicked"):
                    owner = "view-" + name
                    graph = small_graph()
                    member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
                    graph = handlers["group_selected"](graph, [member], "old", 40, False, owner)[0]
                    newer = small_graph()
                    newer["id"] = "newer"
                    path = Path(directory) / "newer.json"
                    path.write_text(json.dumps(newer))
                    opened = False
                    def rendering(*args, **kwargs):
                        nonlocal opened
                        if not opened:
                            opened = True
                            handlers["open_path"](path, 40, False, owner)
                        return original(*args, **kwargs)
                    args = (graph, "old" if name == "choose_group" else json.dumps({"name": "old"}), owner)
                    expected = gr.skip() if name == "choose_group" else (gr.skip(), gr.skip())
                    with mock.patch.object(render, "group_card", side_effect=rendering):
                        self.assertEqual(handlers[name](*args), expected)
                    self.assertTrue(opened)
                    with mock.patch.object(render, "group_card") as drawn:
                        self.assertEqual(handlers[name](*args), expected)
                    drawn.assert_not_called()
            finally:
                demo.close()

    def test_mutation_outputs_rendered_after_open_are_discarded(self):
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
                original = render.graph_view
                for name in ("group_selected", "rename_node", "delete_group"):
                    owner = "view-" + name
                    graph = small_graph()
                    member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
                    graph = handlers["group_selected"](graph, [member], "old", 40, False, owner)[0]
                    newer = small_graph()
                    newer["id"] = "newer"
                    path = Path(directory) / "newer.json"
                    path.write_text(json.dumps(newer))
                    opened = False
                    def rendering(*args, **kwargs):
                        nonlocal opened
                        if not opened:
                            opened = True
                            handlers["open_path"](path, 40, False, owner)
                        return original(*args, **kwargs)
                    args = {"group_selected": (graph, [member], "new", 40, False, owner),
                            "rename_node": (graph, member, "renamed", [member], 40, False, owner),
                            "delete_group": (graph, "old", [member], 40, False, owner)}[name]
                    with mock.patch.object(render, "graph_view", side_effect=rendering):
                        result = handlers[name](*args)
                    with self.subTest(handler=name):
                        self.assertTrue(opened)
                        self.assertTrue(all(item == gr.skip() for item in result))
            finally:
                demo.close()

    def test_pending_focus_blocks_old_ablation_before_details_arrive(self):
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
                graph = handlers["group_selected"](graph, members, "group", 40, False, "view")[0]
                with mock.patch.object(workbench.Workbench, "record", return_value=None):
                    handlers["picked"](graph, json.dumps(dict(selected=[members[0]], focus=members[0])), "view")
                def fetched(*_args):
                    self.assertEqual(handlers["ablate_focused"]("view", graph, members[0]), gr.skip())
                    return None
                with mock.patch.object(workbench.Workbench, "record", side_effect=fetched), \
                        mock.patch.object(workbench.Workbench, "background") as work:
                    result = handlers["picked"](graph, json.dumps(dict(selected=[members[1]], focus=members[1])), "view")
                    self.assertEqual(result[1], members[1])
                    work.assert_not_called()
            finally:
                demo.close()

    def test_new_trace_request_cancels_the_active_model_work(self):
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
                original = handlers["begin_trace"]("view")
                entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
                def running(_bench, _prompt, _explain, _settings, _progress, stop):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("new trace did not cancel")
                    if stop():
                        cancelled.set()
                        raise attribution.Cancelled()
                    raise AssertionError("active cancellation flag was not set")
                real_cancel = workbench.Workbench.cancel
                def cancel(bench, owner):
                    real_cancel(bench, owner)
                    release.set()
                with mock.patch.object(workbench.Workbench, "trace", autospec=True, side_effect=running), \
                        mock.patch.object(workbench.Workbench, "cancel", autospec=True, side_effect=cancel):
                    frames = []
                    thread = threading.Thread(target=lambda: frames.extend(handlers["run_trace"](
                        "view", "", "hi", "", False, "The likeliest next tokens", "", "", 40, .8, .98, 8, 40, False, original)))
                    thread.start()
                    self.assertTrue(entered.wait(5))
                    newer = handlers["begin_trace"]("view")
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                    self.assertNotEqual(original, newer)
                    self.assertTrue(cancelled.is_set())
                    self.assertTrue(all(value == gr.skip() for frame in frames for value in frame))
            finally:
                demo.close()

    def test_trace_ticket_is_rechecked_after_background_session_registration(self):
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
                original = handlers["begin_trace"]("view")
                real_background = workbench.Workbench.background
                def delayed_registration(bench, owner, work):
                    handlers["begin_trace"](owner)
                    yield from real_background(bench, owner, work)
                with mock.patch.object(workbench.Workbench, "background", delayed_registration), \
                        mock.patch.object(workbench.Workbench, "trace", side_effect=AssertionError("obsolete model run")) as trace:
                    frames = list(handlers["run_trace"]("view", "", "hi", "", False, "The likeliest next tokens",
                                                        "", "", 40, .8, .98, 8, 40, False, original))
                trace.assert_not_called()
                self.assertTrue(all(value == gr.skip() for frame in frames for value in frame))
            finally:
                demo.close()

    def test_opened_graph_cancels_active_work_and_stages_normalized_metadata(self):
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
                cells = dict(zip(handlers["begin_trace"].__code__.co_freevars, handlers["begin_trace"].__closure__))
                bench = cells["bench"].cell_contents
                flag = threading.Event()
                bench._sessions["view"] = {"cancelled": flag}
                path = Path(directory) / "upload.json"
                graph = small_graph() | dict(checkpoint_compatibility="verified", training_model_revision="f" * 40)
                graph["prompt"]["user"] = "東京"
                path.write_text(json.dumps(graph))
                opened = handlers["open_path"](path, 40, False, "view")
                self.assertTrue(flag.is_set())
                self.assertEqual(opened[0]["checkpoint_compatibility"], "unverified")
                staging_cell = dict(zip(handlers["open_path"].__code__.co_freevars, handlers["open_path"].__closure__))["staged"]
                staged_cells = dict(zip(staging_cell.cell_contents.__code__.co_freevars, staging_cell.cell_contents.__closure__))
                download = Path(staged_cells["staging"].cell_contents["view"].name) / "circuit.json"
                self.assertIn("東京", download.read_text())
                offered = json.loads(download.read_text())
                self.assertEqual(offered["checkpoint_compatibility"], "unverified")
                self.assertIsNone(offered["training_model_revision"])
                self.assertEqual(json.loads(path.read_text())["checkpoint_compatibility"], "verified")
                bench._sessions.clear()
            finally:
                demo.close()

    def test_stop_cancels_all_submitted_gradio_jobs(self):
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
                jobs = {fn._id for fn in demo.fns.values() if fn.name in
                        ("load_now", "run_trace", "ablate_focused", "run_interventions")}
                cancellations = [set(dependency["cancels"]) for dependency in demo.config["dependencies"]
                                 if dependency.get("cancels")]
                self.assertEqual(len(jobs), 4)
                self.assertEqual(cancellations, [jobs, jobs])
                handlers = handlers_by_name(demo)
                request = handlers["begin_trace"]("view")
                handlers["stop_now"]("view")
                with mock.patch.object(workbench.Workbench, "background") as work:
                    list(handlers["run_trace"]("view", "", "hi", "", False, "", False, False,
                                               40, .8, .98, 8, 40, False, request))
                work.assert_not_called()
            finally:
                demo.close()

    def test_graph_mutations_cancel_invalidated_model_work(self):
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
                cells = dict(zip(handlers["begin_trace"].__code__.co_freevars, handlers["begin_trace"].__closure__))
                bench = cells["bench"].cell_contents
                for name in ("group_selected", "rename_node", "delete_group"):
                    fn = handlers[name]
                    save = dict(zip(fn.__code__.co_freevars, fn.__closure__))["save"].cell_contents
                    graph = small_graph()
                    member = next(n["id"] for n in graph["nodes"] if n["kind"] == "feature")
                    graph["groups"] = {"old": [member]}
                    save(graph, "view")
                    flag = threading.Event()
                    bench._sessions["view"] = {"cancelled": flag}
                    args = {"group_selected": (graph, [member], "new", 40, False, "view"),
                            "rename_node": (graph, member, "renamed", [member], 40, False, "view"),
                            "delete_group": (graph, "old", [member], 40, False, "view")}[name]
                    with mock.patch.object(workbench.Workbench, "record", return_value=None):
                        fn(*args)
                    with self.subTest(mutation=name):
                        self.assertTrue(flag.is_set())
                    bench._sessions.clear()
            finally:
                demo.close()

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

    def test_idle_stop_does_not_cancel_future_work(self):
        bench = workbench.Workbench(SimpleNamespace(), tempfile.gettempdir())
        bench.cancel("owner")
        work = mock.Mock(return_value="result")
        self.assertEqual(list(bench.background("owner", work)), [("done", "result")])
        work.assert_called_once()

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

    def test_page_reservations_obey_click_order_before_queueing(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda *args, **kwargs: None))
        records = mock.Mock()
        records.get.return_value = self.record
        with gr.Blocks() as demo:
            browser.build_browser(context, SimpleNamespace(records=lambda spec: records))
        try:
            handlers = handlers_by_name(demo)
            refresh = handlers["begin_refresh"]
            entered, release = threading.Event(), threading.Event()
            old = []
            def delayed_stamp():
                if threading.current_thread().name == "old-page":
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("new click did not reserve")
                return SimpleNamespace(hex=threading.current_thread().name)
            with mock.patch.object(browser, "uuid4", side_effect=delayed_stamp):
                thread = threading.Thread(target=lambda: old.append(refresh("view", None, 1)), name="old-page")
                thread.start()
                self.assertTrue(entered.wait(5))
                newest = refresh("view", None, 2)
                release.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
            self.assertEqual(old[0][1:], (gr.skip(),) * 3)
            callbacks = {fn.fn.keywords["step"]: fn.fn for fn in demo.fns.values()
                         if isinstance(fn.fn, partial) and fn.fn.func.__name__ == "list_page"}
            stale = callbacks[1](browser.DEFAULT_SET, 3, 20, None, "view", old[0][0])
            self.assertEqual(stale, (gr.skip(),) * 6)
            records.get.assert_not_called()
            current = callbacks[-1](browser.DEFAULT_SET, 3, 20, None, "view", newest[0])
            self.assertEqual(current[1], 0)
            self.assertEqual(handlers["begin_page"]("view", 1)[1:], (gr.skip(),) * 3)
            self.assertEqual(handlers["choose_set"]("gemma-2-2b", "view", 1), (gr.skip(),) * 6)
            event = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", None) == "begin_refresh")
            self.assertIsInstance(event.outputs[0], gr.Textbox)
        finally:
            demo.close()

    def test_refresh_captures_selection_before_state_is_cleared(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda *args, **kwargs: None))
        bench = SimpleNamespace(records=lambda spec: SimpleNamespace(get=lambda layer, feature: self.record))
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            handlers = handlers_by_name(demo)
            refresh = handlers["begin_refresh"]
            selected = {"set": browser.DEFAULT_SET, "layer": 3, "feature": 2}
            for step in (0, -1, 1):
                callback = next(fn.fn for fn in demo.fns.values() if isinstance(fn.fn, partial)
                                and fn.fn.func.__name__ == "list_page" and fn.fn.keywords["step"] == step)
                # Previous clamps at zero. Next preserves the selection when
                # clamped at the last page; use that page's first feature.
                start = 0 if step != 1 else browser.page_start(browser.spec_named(browser.DEFAULT_SET), 10**9)
                chosen = dict(selected, feature=start + 2)
                stamp, cleared, detail, _ = refresh("view", chosen)
                self.assertIsNone(cleared)
                self.assertEqual(detail, gr.skip())
                frame = callback(browser.DEFAULT_SET, 3, start, cleared, "view", stamp)
                self.assertEqual(frame[3]["feature"], chosen["feature"])
                self.assertEqual(frame[4], gr.skip())
                moved = handlers["begin_page"]("view")[0]
                changed = callback(browser.DEFAULT_SET, 4, start, None, "view", moved)
                self.assertIsNone(changed[3])
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
            self.assertEqual(pick(new, json.dumps(dict(feature=2, page_id=old["stamp"], nonce=1))), (gr.skip(),) * 3)
            self.assertEqual(pick(new, json.dumps(dict(feature=99, page_id=new["stamp"], nonce=2))), (gr.skip(),) * 3)
            records.get.assert_not_called()
            card, selected, _ = pick(new, json.dumps(dict(feature=2, page_id=new["stamp"], nonce=3)))
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
            raw = json.dumps(dict(feature=2, page_id=page["stamp"], nonce=1))
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

    def test_late_old_row_reservation_cannot_replace_newer_click(self):
        import gradio as gr
        from functools import partial
        from ui_support import handlers_by_name
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        vectors = []
        context = SimpleNamespace(models=SimpleNamespace(loaded_model_id=lambda: None),
                                  navigation=SimpleNamespace(steer_chat=lambda button, fn, inputs, prepare=None: vectors.append(prepare or fn)))
        bench = SimpleNamespace(records=lambda spec: SimpleNamespace(get=lambda *args: self.record))
        with gr.Blocks() as demo:
            browser.build_browser(context, bench)
        try:
            handlers = handlers_by_name(demo)
            show = next(fn.fn for fn in demo.fns.values() if isinstance(fn.fn, partial) and fn.fn.func.__name__ == "list_page")
            page = show(browser.DEFAULT_SET, 3, 0, None, "view", handlers["begin_page"]("view")[0])[2]
            older = json.dumps(dict(feature=1, page_id=page["stamp"], nonce=1))
            newer = json.dumps(dict(feature=2, page_id=page["stamp"], nonce=2))
            entered, release = Event(), Event()
            parse = json.loads
            def delayed(raw):
                if raw == older:
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("old selection was not released")
                return parse(raw)
            with mock.patch.object(browser.json, "loads", side_effect=delayed), ThreadPoolExecutor(max_workers=2) as pool:
                old = pool.submit(handlers["begin_pick"], page, older, "view")
                self.assertTrue(entered.wait(1))
                try:
                    ticket = handlers["begin_pick"](page, newer, "view")[0]
                finally:
                    release.set()
                self.assertEqual(old.result(timeout=2), (gr.skip(),) * 3)
            self.assertEqual(handlers["picked"](page, older, "view", ticket), (gr.skip(),) * 3)
            selected = handlers["picked"](page, newer, "view", ticket)[1]
            self.assertEqual(selected["feature"], 2)
            listener = next(fn for fn in demo.fns.values() if fn.name == "begin_pick")
            self.assertIsInstance(listener.outputs[0], gr.Textbox)
            self.assertFalse(listener.queue)
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
            raw = lambda feature: json.dumps(dict(feature=feature, page_id=page["stamp"], nonce=feature))
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
