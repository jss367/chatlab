"""Direct MLP interventions through generation, replay, and conversation storage."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gradio as gr
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from chatlab import compare, conversation, library, steering
from chatlab.ui import runtime
from chatlab.ui import steering as controls
from chatlab.ui.conversations import fork_conversation, new_conversation
import settings_sandbox
from test_steering import manager, MESSAGES, SAMPLING, vector
from ui_support import listener_named


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def neuron(**changes):
    return steering.normalize(dict(dict(format=steering.NEURON_FORMAT, model_id="test/tiny",
                                        layer=0, neuron=3, operation="add", strength=2), **changes))


class NeuronRuntimeTests(unittest.TestCase):
    def test_edits_exact_channel_for_prompt_and_cached_decode_without_mutating_inputs(self):
        for operation, strength in (("add", -2), ("scale", 0), ("scale", -3), ("set", 0), ("set", 4)):
            with self.subTest(operation=operation, strength=strength):
                held = manager()
                projection = held.model.model.layers[0].mlp.down_proj
                weights = {name: value.clone() for name, value in held.model.state_dict().items()}
                # A pre-hook already installed by an inspector must see the edited inputs.
                seen = []
                observer = projection.register_forward_pre_hook(lambda _module, args: seen.append(args[0].clone()))
                try:
                    for count in (4, 1):
                        hidden = torch.randn(2, count, projection.in_features)
                        original = hidden.clone()
                        expected = hidden.clone()
                        if operation == "add":
                            expected[..., 3] += strength
                        elif operation == "scale":
                            expected[..., 3] *= strength
                        else:
                            expected[..., 3] = strength
                        with steering.applied(held.model, held.model_id, neuron(operation=operation, strength=strength)):
                            actual = projection(hidden)
                        torch.testing.assert_close(seen[-1], expected)
                        torch.testing.assert_close(actual, torch.nn.functional.linear(expected, projection.weight))
                        torch.testing.assert_close(hidden, original, rtol=0, atol=0)
                    for name, value in held.model.state_dict().items():
                        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                finally:
                    observer.remove()
                self.assertFalse(projection._forward_pre_hooks)

    def test_add_matches_down_projection_column_on_llama(self):
        held = manager()
        ids = torch.tensor([[0, 1, 2]])
        column = held.model.model.layers[0].mlp.down_proj.weight[:, 3].detach().tolist()
        direction = steering.vector_from(held.model_id, 0, column, strength=2)
        with torch.inference_mode():
            plain = held.model(ids).logits
            with steering.applied(held.model, held.model_id, direction):
                expected = held.model(ids).logits
            with steering.applied(held.model, held.model_id, neuron()):
                actual = held.model(ids).logits
            torch.testing.assert_close(actual, expected)
            self.assertFalse(torch.allclose(actual, plain))
            torch.testing.assert_close(held.model(ids).logits, plain, rtol=0, atol=0)

    def test_qwen35_hybrid_decoder_and_nested_language_model(self):
        config = Qwen3_5TextConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        )
        model = Qwen3_5ForCausalLM(config).eval()
        ids = torch.tensor([[0, 1, 2]])
        with torch.inference_mode():
            plain = model(ids).logits
            for layer in (0, 1):
                value = neuron(layer=layer, operation="set", strength=5)
                with steering.applied(model, "test/tiny", value):
                    result = model(ids, use_cache=True)
                    changed = result.logits
                    decoded = model(torch.tensor([[3]]), past_key_values=result.past_key_values, use_cache=True)
                    self.assertTrue(torch.isfinite(decoded.logits).all())
                self.assertFalse(torch.allclose(plain, changed))
                torch.testing.assert_close(model(ids).logits, plain, rtol=0, atol=0)
            # The multimodal wrapper uses this same stack at model.language_model.layers.
            wrapper = torch.nn.Module()
            wrapper.model = torch.nn.Module()
            wrapper.model.language_model = model.model
            self.assertIs(steering.validate_model(wrapper, "test/tiny", neuron()), model.model.layers[0].mlp.down_proj)

    def test_zero_ablation_stays_active_and_replay_matches_generation(self):
        held = manager()
        for operation in ("scale", "set"):
            value = neuron(operation=operation, strength=0)
            self.assertTrue(steering.active(value))
            update = list(held.generate(MESSAGES, **SAMPLING, forced_ids=(1, 2), steering=value))[-1]
            ids = list(update.prompt_ids) + [row["token_id"] for row in update.metrics]
            replay = held.inspect(ids, len(update.prompt_ids) + 1, steering=value)
            self.assertAlmostEqual(replay.layers[-1]["probability"], update.metrics[1]["raw_probability"], places=6)
            self.assertIsNone(held._inspect_cache)
            self.assertFalse(held.model.model.layers[0].mlp.down_proj._forward_pre_hooks)
        for value in (neuron(strength=0), neuron(operation="scale", strength=1), neuron(enabled=False)):
            self.assertFalse(steering.active(value))
            # Identity interventions do not need a compatible backend or model.
            with steering.applied(None, None, value):
                pass

    def test_cancel_and_error_remove_hooks_and_release_generation(self):
        held = manager()
        projection = held.model.model.layers[0].mlp.down_proj
        stream = held.generate(MESSAGES, **SAMPLING, steering=neuron())
        next(stream)
        self.assertTrue(projection._forward_pre_hooks)
        stream.close()
        self.assertFalse(projection._forward_pre_hooks)
        self.assertFalse(held.busy)
        with mock.patch.object(projection, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                list(held.generate(MESSAGES, **SAMPLING, steering=neuron()))
        self.assertFalse(projection._forward_pre_hooks)
        self.assertFalse(held.busy)

    def test_overflow_fails_without_leaking_hook(self):
        held = manager()
        projection = held.model.model.layers[0].mlp.down_proj.half()
        hidden = torch.full((1, 2, 16), 60000, dtype=torch.float16)
        with self.assertRaisesRegex(steering.SteeringError, "overflows"):
            with steering.applied(held.model, held.model_id, neuron(operation="scale", strength=100)):
                projection(hidden)
        self.assertFalse(projection._forward_pre_hooks)

    def test_validation_refuses_invalid_targets_models_and_backends(self):
        held = manager()
        for changes, message in (({"layer": 2}, "2 layers"), ({"neuron": 16}, "16 neurons"),
                                 ({"model_id": "other/model"}, "requires")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                held.check_steering(neuron(**changes))
        with mock.patch.object(held, "_engine", return_value=SimpleNamespace(backend="mlx")):
            with self.assertRaisesRegex(ValueError, "MLX"):
                held.check_steering(neuron(operation="set", strength=0))
        held.model.is_quantized = True
        with self.assertRaisesRegex(ValueError, "unquantized"):
            held.check_steering(neuron())
        with self.assertRaisesRegex(ValueError, "dense MLP"):
            manager("gpt2").check_steering(neuron())
        for changes in ({"neuron": -1}, {"neuron": True}, {"neuron": 1.5}, {"layer": -1},
                        {"operation": "unknown"}, {"strength": float("nan")}, {"strength": 101}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                neuron(**changes)


class NeuronConversationTests(unittest.TestCase):
    def test_conversation_and_library_roundtrip_with_mixed_vector_and_neuron_history(self):
        value = neuron(operation="set", strength=0)
        reference = steering.compact(vector())
        turns = [dict(conversation.make_turn("assistant", "Hello"), steering=item) for item in (reference, value)]
        payload = conversation.to_json(turns, steering=value)
        self.assertEqual(len(json.loads(payload)["steering_vectors"]), 1)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(steering, "asset_directory", return_value=Path(directory)):
            loaded, _ = conversation.from_json(payload)
            self.assertEqual(loaded, turns)
            self.assertEqual(steering.expand(loaded[1]["steering"]), value)
            self.assertEqual(steering.expand(loaded[0]["steering"]), vector())
        forks = controls.store(conversation.new_forks(), value)
        forked = fork_conversation(turns, forks, None)["forks"]
        restored = library.parse(library.dump(forked))
        self.assertEqual(controls.steering_updates(restored)[0], value)
        self.assertIsNone(controls.steering_updates(new_conversation(turns, restored)["forks"])[0])
        self.assertEqual(steering.export_assets([value]), {})
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(library, "library_path", return_value=Path(directory) / "library.json"):
            library.write(restored)
            reference = steering.compact(vector(strength=3))
            # A mixed library retains the vector from its history and needs
            # no asset for the neuron's inline settings.
            self.assertEqual(steering.cleanup_unused_assets(), (0, 0))
            self.assertEqual(steering.expand(reference), vector(strength=3))

    def test_chat_trace_and_save_capture_visible_neuron_settings(self):
        from chatlab.trace_export import trace_to_json
        from chatlab.ui.conversations import save_conversation
        from chatlab.ui.generation import chat
        from conversation_support import FIXED

        held = manager()
        with mock.patch.object(runtime, "MANAGER", held):
            frame = list(chat("Hello", [], **dict(FIXED, assistant_prefill="Hello"),
                              steering=neuron(), steering_enabled=True,
                              steering_strength=0, steering_layer=1))[-1]
        expected = neuron(strength=0, layer=1)
        self.assertEqual(frame["turns"][-1]["steering"], expected)
        exported = json.loads(trace_to_json(frame["trace"]))
        self.assertEqual(exported["sampling"]["steering"], expected)
        saved, _ = save_conversation(frame["turns"], "", neuron(operation="set"), True, 0, 1)
        self.addCleanup(Path(saved["value"]).unlink)
        payload = json.loads(Path(saved["value"]).read_text())
        self.assertEqual(payload["steering"], neuron(operation="set", strength=0, layer=1))
        self.assertNotIn("steering_vectors", payload)

    def test_apply_checks_model_and_preserves_existing_settings_on_failure(self):
        held = manager()
        forks = controls.store(conversation.new_forks(), vector())
        before = conversation.copy_forks(forks)
        with mock.patch.object(runtime, "MANAGER", held):
            for args, message in (((0, 16, "add", 0, held.load_id), "16 neurons"),
                                  ((0, 3, "add", 1, "old-load"), "changed"),
                                  ((0, 3.5, "add", 1, held.load_id), "integer")):
                with self.assertRaisesRegex(gr.Error, message):
                    controls.use_neuron(forks, *args)
                self.assertFalse(held.busy)
                self.assertEqual(forks, before)
            held.claim_generation()
            try:
                with self.assertRaisesRegex(gr.Error, "Wait"):
                    controls.use_neuron(forks, 0, 3, "add", 1, held.load_id)
                self.assertTrue(held.busy)
            finally:
                held.release_generation()
            updated = controls.use_neuron(forks, 0, 3, "scale", 0, held.load_id)
        self.assertEqual(updated[1], neuron(operation="scale", strength=0))
        self.assertIn("Enabled", updated[-1])
        self.assertIn("neuron 3", updated[-1])
        self.assertIn("0 silences", updated[3]["info"])
        summary = compare.configuration({"kind": compare.REPLY, "settings": {"steering": updated[1]}})
        self.assertIn("on, layer 0", summary["Steering"])
        self.assertIn("neuron 3, scale", summary["Steering"])

    def test_ui_receipt_captures_target_and_load_and_discards_stale_edits(self):
        from chatlab.ui.layout import build_app

        demo = build_app()
        capture = listener_named(demo, "capture_use_neuron")
        apply = listener_named(demo, "use_neuron")
        held = manager()
        forks = conversation.new_forks()
        with mock.patch.object(runtime, "MANAGER", held):
            ticket = capture.fn(forks, 0, 3, "set", 0)
            result = apply.fn(forks, ticket)
            self.assertEqual(result[1], neuron(operation="set", strength=0))
            ticket = capture.fn(forks, 0, 3, "add", 1)
            held.load_count += 1
            with self.assertRaisesRegex(gr.Error, "changed"):
                apply.fn(forks, ticket)
            ticket = capture.fn(forks, 0, 3, "add", 1)
            controls.reserve_steering(forks, [])
            self.assertEqual(apply.fn(forks, ticket), (gr.skip(),) * len(apply.outputs))


if __name__ == "__main__":
    unittest.main()
