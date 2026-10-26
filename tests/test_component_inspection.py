"""Numerical checks of native channels, head edits and causal restoration."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr
import torch

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.circuits import architecture, attribution, interventions, render, transcoders, validation
from chatlab.extensions.circuits.feature_validation import FeatureModel
from chatlab.extensions.circuits.inspection_page import build_inspection, inspection_result
from chatlab.extensions.circuits.native import NativeModel
from chatlab.extensions.circuits.workbench import Workbench
from fakes import FakeManager
from ui_support import handlers_by_name

IDS = [2, 17, 5, 33, 9]


def model_for(kind):
    import transformers

    torch.manual_seed(15)
    prefix = {"gemma2": "Gemma2", "gemma3": "Gemma3Text", "qwen3": "Qwen3",
              "qwen2": "Qwen2", "llama": "Llama", "olmo3": "Olmo3"}[kind]
    common = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=64,
                  eos_token_id=1, bos_token_id=2, pad_token_id=0)
    if kind == "gemma3":
        common.update(layer_types=["full_attention", "full_attention"])
    config = getattr(transformers, prefix + "Config")(**common)
    cls = "Gemma3ForCausalLM" if kind == "gemma3" else prefix + "ForCausalLM"
    return getattr(transformers, cls)(config).eval()


def held_for(blocks):
    torch.manual_seed(25)
    width, d, layers = 12, blocks.embed.weight.shape[1], len(blocks.layers)
    spec = transcoders.TranscoderSpec("inspection", "Tiny", ("test/model",), "test/tiny", "", layers, width, d, 0.)
    return transcoders.Transcoders(spec, w_enc=[torch.randn(width, d) for _ in range(layers)],
                                  b_enc=[torch.ones(width) for _ in range(layers)],
                                  w_dec=[torch.randn(width, d) / 5 for _ in range(layers)],
                                  b_dec=[torch.zeros(d) for _ in range(layers)],
                                  threshold=[None] * layers, device=torch.device("cpu"))


class ToyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.down_proj(x)


class ToyModel(torch.nn.Module):
    """Two neurons in series: restoring the second must rescue the output."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", num_attention_heads=1)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in self.model.layers:
            layer.mlp = ToyMLP()
        self.embed = torch.nn.Embedding(4, 2)
        self.output = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor([[0., 1.], [1., 1.], [2., 1.], [3., 1.]]))
            self.output.weight.copy_(torch.tensor([[1., 0.], [0., 0.], [-1., 0.], [0., 1.]]))
        self.eval()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.output

    def forward(self, input_ids, use_cache=False, logits_to_keep=0):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer.mlp(x)
        return SimpleNamespace(logits=self.output(x[:, -logits_to_keep:] if logits_to_keep else x))


class NativeInspectionTests(unittest.TestCase):
    def test_zeroing_matches_manual_neuron_and_head_hooks_on_supported_models(self):
        for kind in ("gemma2", "gemma3", "qwen3", "qwen2", "llama", "olmo3"):
            model = model_for(kind)
            backend = NativeModel(model)
            for component_kind in ("neuron", "head"):
                with self.subTest(model=kind, component=component_kind):
                    component = (component_kind, 0, 2)
                    result = backend.forward(IDS, [(component, len(IDS)-1, "scale", 0.)], [component])
                    module = model.model.layers[0].mlp.down_proj if component_kind == "neuron" else model.model.layers[0].self_attn.o_proj
                    span = slice(2, 3) if component_kind == "neuron" else slice(8, 12)

                    def zero(_module, args):
                        x = args[0].clone()
                        x[0, -1, span] = 0
                        return (x, *args[1:])

                    handle = module.register_forward_pre_hook(zero)
                    try:
                        with torch.no_grad():
                            expected = model(torch.tensor([IDS])).logits[0, -1].float().log_softmax(-1)
                    finally:
                        handle.remove()
                    torch.testing.assert_close(result["log_probs"], expected)
                    self.assertTrue(bool((result["activations"][component][-1] == 0).all()))
                    self.assertFalse(module._forward_pre_hooks)

    def test_neuron_capture_is_after_the_gate_and_scale_one_is_a_no_op(self):
        model = model_for("qwen3")
        backend = NativeModel(model)
        component = ("neuron", 0, 7)
        captured = []
        mlp = model.model.layers[0].mlp

        def read(_module, args):
            x = args[0]
            captured.append((mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))[0, :, 7].detach())

        handle = mlp.register_forward_pre_hook(read)
        try:
            result = backend.forward(IDS, observe=[component])
        finally:
            handle.remove()
        torch.testing.assert_close(result["activations"][component][:, 0], captured[0])
        control = backend.forward(IDS, [(component, len(IDS)-1, "scale", 1.)], [component])
        torch.testing.assert_close(control["log_probs"], result["log_probs"], atol=0, rtol=0)

    def test_patching_and_restoring_later_component_rescue_known_causal_chain(self):
        backend = NativeModel(ToyModel())
        progress = []
        result = validation.experiment(backend.forward, [[0, 1], [0, 2]], ("neuron", 0, 0), [0, 2], [0, 1, 2],
                                       donor=[0, 0, 3], donor_offset=0, hub=("neuron", 1, 0),
                                       progress=lambda *args: progress.append(args))
        for prefix in result["prefixes"]:
            rows = prefix["conditions"]
            zero = next(r for r in rows if r.get("factor") == 0)
            patch, rescue = rows[-2:]
            self.assertEqual(zero["hub_activation"], [0.])
            self.assertEqual(patch["activation"], [3.])
            self.assertEqual(rescue["hub_activation"], rows[0]["hub_activation"])
            self.assertEqual(rescue["probabilities"], rows[0]["probabilities"])
            self.assertLess(zero["probabilities"][0], rows[0]["probabilities"][0])
            self.assertEqual(prefix["activations"][0], [0.])
        self.assertEqual(result["model_passes"], progress[-1][1])
        self.assertEqual(progress[-1][1], progress[-1][2])
        # An explicit source offset, not destination length, selects the donor.
        other = validation.experiment(backend.forward, [[0, 1]], ("neuron", 0, 0), [0], [0, 1],
                                      donor=[3, 2, 0], donor_offset=1)
        self.assertEqual(other["prefixes"][0]["conditions"][-1]["activation"], [2.])

    def test_hooks_are_removed_after_failure_or_mid_forward_cancellation(self):
        model = model_for("qwen3")
        backend = NativeModel(model)
        component = ("neuron", 0, 0)
        with mock.patch.object(model.model.layers[1].mlp, "forward", side_effect=RuntimeError("broken")):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                backend.forward(IDS, observe=[component])
        cancelled = [False]
        handle = model.model.layers[0].register_forward_hook(lambda *_: cancelled.__setitem__(0, True))
        try:
            with self.assertRaises(attribution.Cancelled):
                backend.forward(IDS, observe=[component], cancelled=lambda: cancelled[0])
        finally:
            handle.remove()
        self.assertFalse(any(m._forward_pre_hooks or m._forward_hooks for m in model.modules()))
        with torch.no_grad():
            expected = model(torch.tensor([IDS])).logits[0, -1].log_softmax(-1)
        torch.testing.assert_close(backend.forward(IDS)["log_probs"], expected)

    def test_native_neighborhood_measures_the_known_chain_and_restores_grad_mode(self):
        model = ToyModel()
        backend = NativeModel(model)
        upstream = backend.neighborhood([0, 1], ("neuron", 1, 0), 1)
        first = upstream["incoming"][0]
        self.assertEqual((first["layer"], first["index"]), (0, 0))
        self.assertEqual(first["score"], 1.)
        self.assertEqual(first["activation_delta"], -1.)
        downstream = backend.neighborhood([0, 1], ("neuron", 0, 0), 1)
        self.assertEqual(downstream["outgoing"][0]["index"], 0)
        self.assertEqual(downstream["outgoing"][0]["activation_delta"], -1.)
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(any(m._forward_pre_hooks or m._forward_hooks for m in model.modules()))

        # Loaded models can freeze parameters; the local gradient still starts
        # at the chosen upstream activation and must not change that setting.
        model.requires_grad_(False)
        with torch.no_grad():
            result = backend.neighborhood([0, 1], ("neuron", 1, 0), 1)
            self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(result["incoming"][0]["activation_delta"], -1.)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_native_neighborhood_scores_match_individual_real_model_ablations(self):
        model = model_for("qwen3")
        backend = NativeModel(model)
        component = ("neuron", 1, 7)
        baseline = backend.forward(IDS, observe=[component])["activations"][component][-1, 0]
        result = backend.neighborhood(IDS, component, len(IDS)-1, limit=3)
        for node in result["incoming"]:
            changed = backend.forward(IDS, [(("neuron", 0, node["index"]), len(IDS)-1, "scale", 0.)], [component])
            delta = float(changed["activations"][component][-1, 0] - baseline)
            self.assertAlmostEqual(node["activation_delta"], delta, places=7)
        self.assertFalse(any(m._forward_pre_hooks or m._forward_hooks for m in model.modules()))
        with self.assertRaises(attribution.Cancelled):
            backend.neighborhood(IDS, component, len(IDS)-1, cancelled=lambda: True)
        self.assertFalse(any(m._forward_pre_hooks or m._forward_hooks for m in model.modules()))

    def test_invalid_indices_and_experiment_limits_fail_before_forwards(self):
        backend = NativeModel(model_for("qwen3"))
        for component in (("neuron", -1, 0), ("head", 0, 4), ("neuron", 0, 32), ("feature", 0, 0)):
            with self.subTest(component=component), self.assertRaises(ValueError):
                backend.site(component)
        forward = mock.Mock(side_effect=AssertionError("forward"))
        for kwargs in (dict(prefixes=[IDS] * 256, scales=list(range(11))), dict(offset=10),
                       dict(hub=("neuron", 0, 1)), dict(targets=[-1]), dict(donor=IDS, donor_offset=20)):
            settings = dict(prefixes=[IDS], component=("neuron", 0, 0), targets=[0], scales=[0, 1])
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validation.experiment(forward, **settings)
        forward.assert_not_called()
        for text in ("nan", "inf", "", "11", "a"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validation.factors(text)


class FeatureValidationTests(unittest.TestCase):
    def test_feature_ablation_matches_existing_real_model_interventions(self):
        for kind in ("gemma2", "gemma3", "qwen3"):
            with self.subTest(model=kind):
                blocks = architecture.blocks(model_for(kind))
                held = held_for(blocks)
                backend = FeatureModel(blocks, held)
                component = ("feature", 0, 3)
                expected = interventions.run(blocks, held, IDS, [(0, 3, 0., [0])])["log_probs"]
                result = backend.forward(IDS, [(component, len(IDS)-1, "scale", 0.)], [component])
                torch.testing.assert_close(result["log_probs"], expected)
                self.assertEqual(result["activations"][component][-1].tolist(), [0.])
                baseline = interventions.run(blocks, held, IDS)["log_probs"]
                control = backend.forward(IDS, [(component, len(IDS)-1, "scale", 1.)], [component])
                torch.testing.assert_close(control["log_probs"], baseline, rtol=0, atol=0)

    def test_feature_replacement_and_rescue_use_effective_activations(self):
        blocks = architecture.blocks(model_for("qwen3"))
        backend = FeatureModel(blocks, held_for(blocks))
        result = validation.experiment(backend.forward, [IDS], ("feature", 0, 3), [4], [0, 1, 2],
                                       donor=IDS[:-1], hub=("feature", 1, 4))
        rows = result["prefixes"][0]["conditions"]
        self.assertEqual(rows[-2]["activation"], result["source"]["activation"])
        self.assertEqual(rows[-1]["activation"], [0.])
        self.assertEqual(rows[-1]["hub_activation"], rows[0]["hub_activation"])
        self.assertFalse(any(m._forward_hooks for m in blocks.model.modules()))


class InspectionIntegrationTests(unittest.TestCase):
    def bench(self, directory):
        manager = FakeManager()
        manager._lock = threading.Lock()
        manager.model = model_for("qwen3")
        manager.precision = "full"
        manager._engine = lambda: SimpleNamespace(backend="torch")
        manager._release_device_cache = mock.Mock()
        def revision():
            self.assertFalse(manager._lock.locked())
            return "a" * 40
        manager.model_revision = revision
        return manager, Workbench(ModelService(lambda: manager), directory)

    def test_workbench_pins_model_records_provenance_and_needs_no_transcoders(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, bench = self.bench(directory)
            prompt = dict(system="", user="12", prefix="", raw=True)
            with mock.patch.object(transcoders, "load", side_effect=AssertionError("transcoders")):
                result = bench.inspect_components(prompt, ("head", 0, 1), ["3"], "0,1,2", 0,
                                                   ["4"], {**prompt, "user": "11"}, 1,
                                                   ("neuron", 1, 1), 0, lambda *_: None, lambda: False)
            self.assertEqual(result["model_revision"], "a" * 40)
            self.assertEqual(result["source_ids"][-2:], [49, 49])
            self.assertEqual(len(result["prefixes"]), 2)
            self.assertFalse(manager.busy)
            self.assertFalse(manager._lock.locked())
            json.dumps(result, allow_nan=False)
            result["target_texts"] = ["<script>bad()</script>"]
            html = inspection_result(result)
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("Destination prefix 2", html)
            self.assertIn("Measured component ablation", html)

    def test_ui_runs_native_inspection_downloads_and_stop_invalidates_queued_work(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, bench = self.bench(directory)
            context = ExtensionContext(bench.models, TokenInspector(), Path(directory),
                                       NavigationService(lambda *_: None, lambda *_: None))
            with gr.Blocks() as demo:
                build_inspection(context, bench)
            try:
                handlers = handlers_by_name(demo)
                turn = handlers["begin_inspection"]("view")
                args = ("neuron", 0, 1, "", "12", "", True, "neuron", 1, 2,
                        0, "3", "0,1,2", "", False, "", "", 0, True, 0, True)
                frames = list(handlers["run_inspection"]("view", turn, *args))
                path = Path(frames[-1][2])
                saved = json.loads(path.read_text())
                self.assertEqual(saved["component"], ["neuron", 0, 1])
                self.assertTrue(saved["neighborhood"]["outgoing"])
                self.assertIn("Native neuron neighborhood", frames[-1][1])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(len(list((Path(directory) / "inspections").glob("*.json"))), 1)
                turn = handlers["begin_inspection"]("view")
                handlers["stop_inspection"]("view")
                with mock.patch.object(bench, "inspect_components") as run:
                    list(handlers["run_inspection"]("view", turn, *args))
                    run.assert_not_called()
            finally:
                demo.close()

    def test_feature_view_uses_selected_node_and_rejects_stale_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, bench = self.bench(directory)
            context = ExtensionContext(bench.models, TokenInspector(), Path(directory),
                                       NavigationService(lambda *_: None, lambda *_: None))
            with gr.Blocks() as demo:
                graph_state, focus = gr.State(None), gr.State(None)
                build_inspection(context, bench, graph_state, focus)
            try:
                handlers = handlers_by_name(demo)
                graph = dict(id="graph", model_id="other/model", prompt=dict(raw=True, system="", user="12", prefix=""),
                             nodes=[dict(id="f:0", kind="feature", layer=0, feature=3)])
                turn = handlers["begin_inspection"]("view")
                with self.assertRaisesRegex(gr.Error, "load that model"):
                    list(handlers["run_inspection"]("view", turn, graph, "f:0", None,
                         0, "3", "0,1", "", False, "", "", 0, False, 0))
                self.assertFalse(manager.busy)
                blocks = architecture.blocks(manager.model)
                held = held_for(blocks)
                graph.update(model_id=manager.model_id, model_revision="a" * 40, layers=2, ids=[2, 49, 50],
                             transcoder_load_id=held.load_id, transcoder_revision=held.revision,
                             transcoders=held.spec.key, nodes=[
                                 dict(id="f:0", kind="feature", layer=0, feature=3, position=1),
                                 dict(id="f:1", kind="feature", layer=1, feature=4, position=2)])
                choices = handlers["inspection_choices"](graph, "f:0")
                self.assertEqual(choices[2], 1)
                self.assertEqual(choices[0]["choices"][0][1], "f:1")
                self.assertEqual(handlers["inspection_hub_position"](graph, "f:1"), 0)
                turn = handlers["begin_inspection"]("view")
                with mock.patch.object(bench, "_held", return_value=(blocks, held, held.spec)):
                    frames = list(handlers["run_inspection"]("view", turn, graph, "f:0", "f:1",
                                  1, "3", "0,1,2", "", True, "11", "", 0, True, 0))
                saved = json.loads(Path(frames[-1][2]).read_text())
                self.assertEqual(saved["component"], ["feature", 0, 3])
                self.assertEqual(saved["graph_id"], "graph")
                self.assertEqual(saved["prefixes"][0]["position"], 1)
                self.assertEqual(saved["prefixes"][0]["conditions"][-1]["activation"], [0.])
                turn = handlers["begin_inspection"]("view")
                handlers["invalidate_inspection"]("view")
                with mock.patch.object(bench, "inspect_components") as run:
                    list(handlers["run_inspection"]("view", turn))
                    run.assert_not_called()
            finally:
                demo.close()

    def test_neighborhood_keeps_both_edge_directions_and_escapes_labels(self):
        graph = dict(tokens=["x"], labels={"hub": "<hub>"}, nodes=[
            dict(id="source", kind="embedding", position=0, layer=-1, text="x", effect=1., influence=1.),
            dict(id="hub", kind="feature", position=0, layer=0, feature=2, effect=1., influence=1., activation=1.),
            dict(id="output", kind="target", position=0, layer=1, text=" Wait", probability=.5)],
            edges=[dict(source="source", target="hub", weight=1.5), dict(source="hub", target="output", weight=-.2)])
        view = render.neighborhood_view(graph, "hub")
        self.assertIn("&lt;hub&gt;", view)
        self.assertIn("+1.5", view)
        self.assertIn("-0.2", view)
        self.assertIn('data-node="source"', view)
        self.assertIn('data-node="output"', view)
        self.assertNotIn("<hub>", view)


if __name__ == "__main__":
    unittest.main()
