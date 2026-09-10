import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

import conversation
import library
import settings_sandbox
import steering
from model_runtime import ModelManager
from test_streaming import FakeTokenizer, PIECES, EOS_ID
from ui import runtime
from ui import steering as controls
from ui.conversations import fork_conversation, new_conversation, save_conversation


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def vector(**changes):
    return steering.normalize(dict(
        model_id="test/tiny", layer=0,
        vector=[1.0, -2.0, 3.0, -4.0, 0.0, 1.0, -1.0, 0.0],
        **changes,
    ))


def manager(architecture="llama"):
    torch.manual_seed(42)
    if architecture == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(PIECES), n_embd=8, n_layer=2, n_head=2,
            n_positions=64, eos_token_id=EOS_ID, bos_token_id=0, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
        ))
    else:
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(PIECES), hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, eos_token_id=EOS_ID,
        ))
    model.eval()
    model.set_attn_implementation("eager")
    result = ModelManager()
    result.model = model
    result.model_id = "test/tiny"
    result.tokenizer = FakeTokenizer()
    return result


SAMPLING = dict(temperature=0, top_p=1, top_k=0, max_new_tokens=3, seed=42)
MESSAGES = [{"role": "user", "content": "Hello"}]


class VectorTests(unittest.TestCase):
    def test_file_roundtrip_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vector.json"
            path.write_text(json.dumps(vector()))
            self.assertEqual(steering.read_vector(path), vector())
            for bad in (None, [], {"vector": []}, dict(vector(), vector=[float("nan")]),
                        dict(vector(), layer=True), dict(vector(), strength=float("inf")),
                        dict(vector(), vector=[True]), dict(vector(), enabled="yes")):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        steering.read_vector(path)
            path.write_bytes(b" " * (steering.MAX_FILE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "4 MiB"):
                steering.read_vector(path)

    def test_model_layer_width_and_architecture_checks(self):
        model = manager().model
        for bad, message in (
            (dict(vector(), model_id="other/model"), "requires"),
            (dict(vector(), layer=2), "2 layers"),
            (dict(vector(), vector=[1]), "8 entries"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                with steering.applied(model, "test/tiny", bad):
                    pass
        with self.assertRaisesRegex(ValueError, "architecture"):
            steering.decoder_layers(torch.nn.Linear(8, 8))
        self.assertFalse(model.model.layers[0]._forward_hooks)


class RuntimeTests(unittest.TestCase):
    def test_real_decoder_outputs_change_and_weights_do_not(self):
        for architecture in ("llama", "gpt2"):
            with self.subTest(architecture=architecture):
                model = manager(architecture).model
                ids = torch.tensor([[0, 1, 2]])
                weights = {name: value.clone() for name, value in model.state_dict().items()}
                with torch.inference_mode():
                    baseline_output = model(ids, output_hidden_states=True)
                    baseline = baseline_output.logits
                    with steering.applied(model, "test/tiny", vector()):
                        changed_output = model(ids, output_hidden_states=True)
                        changed = changed_output.logits
                    self.assertFalse(torch.allclose(baseline, changed))
                    # Capture hooks installed by a previous inspection must
                    # observe the modified residual, not the original output.
                    delta = changed_output.hidden_states[1] - baseline_output.hidden_states[1]
                    torch.testing.assert_close(delta, torch.tensor(vector()["vector"]).expand_as(delta))
                    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
                    for disabled in (vector(enabled=False), vector(strength=0)):
                        with steering.applied(model, "test/tiny", disabled):
                            torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_cleanup_on_cancel_error_and_cross_thread_resume(self):
        from concurrent.futures import ThreadPoolExecutor

        held = manager()
        block = steering.decoder_layers(held.model)[0]
        stream = held.generate(MESSAGES, **SAMPLING, steering=vector())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(next, stream).result()
        self.assertTrue(block._forward_hooks)
        stream.close()
        self.assertFalse(block._forward_hooks)
        self.assertFalse(held.busy)
        with mock.patch.object(block, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                list(held.generate(MESSAGES, **SAMPLING, steering=vector()))
        self.assertFalse(block._forward_hooks)
        self.assertFalse(held.busy)
        self.assertTrue(list(held.generate(MESSAGES, **SAMPLING)))

    def test_inspection_matches_steered_generation_and_does_not_reuse_other_cache(self):
        held = manager()
        update = list(held.generate(MESSAGES, **SAMPLING, forced_ids=(1, 2), steering=vector()))[-1]
        ids = list(update.prompt_ids) + [row["token_id"] for row in update.metrics]
        index = len(update.prompt_ids) + 1
        plain = held.inspect(ids, index).layers[-1]["probability"]
        hooks = dict(steering.decoder_layers(held.model)[0]._forward_hooks)
        self.assertIsNotNone(held._inspect_cache)
        steered = held.inspect(ids, index, steering=vector()).layers[-1]["probability"]
        self.assertAlmostEqual(steered, update.metrics[1]["raw_probability"], places=6)
        self.assertNotAlmostEqual(steered, plain, places=5)
        self.assertIsNone(held._inspect_cache)
        self.assertAlmostEqual(held.inspect(ids, index).layers[-1]["probability"], plain, places=7)
        self.assertEqual(dict(steering.decoder_layers(held.model)[0]._forward_hooks), hooks)

    def test_negative_strength_reverses_the_addition(self):
        held = manager()
        block = steering.decoder_layers(held.model)[0]
        hidden = torch.zeros(1, 2, 8)
        # Exercise the tuple form used by older Transformers releases too.
        with mock.patch.object(block, "forward", return_value=(hidden, "cache")):
            with steering.applied(held.model, held.model_id, vector(strength=-2)):
                changed, cache = block(hidden)
        torch.testing.assert_close(changed, -2 * torch.tensor(vector()["vector"]).expand_as(hidden))
        self.assertEqual(cache, "cache")
        self.assertEqual(hidden.count_nonzero().item(), 0)


class ConversationTests(unittest.TestCase):
    def test_fork_switch_new_chat_and_library_roundtrip(self):
        forks = controls.store(conversation.new_forks(), vector())
        turns = [conversation.make_turn("user", "Hello")]
        forked = fork_conversation(turns, forks, None)[3]
        self.assertEqual(controls.steering_updates(forked)[0], vector())
        changed, state, _ = controls.remember_steering(forked, vector(), True, -2, 1)
        self.assertEqual(state["strength"], -2)
        changed["active"] = conversation.MAIN_BRANCH
        self.assertEqual(controls.steering_updates(changed)[0], vector())
        restored = library.parse(library.dump(changed))
        self.assertEqual(controls.steering_updates(restored)[0], vector())
        fresh = new_conversation(turns, restored)[3]
        self.assertIsNone(controls.steering_updates(fresh)[0])
        copied = conversation.copy_forks(forks)
        copied["sampling"][conversation.MAIN_BRANCH]["steering"]["vector"][0] = 99
        self.assertEqual(controls.steering_updates(forks)[0], vector())

    def test_portable_save_load_keeps_current_and_per_response_vectors(self):
        turns = [dict(conversation.make_turn("assistant", "Hello"), steering=vector(strength=-2))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.json"
            path.write_text(conversation.to_json(turns, steering=vector()))
            loaded = controls.load_with_steering(str(path), [], "Raw rank", conversation.new_forks())
            self.assertEqual(loaded[1], turns)
            self.assertEqual(loaded[-5], vector())
            self.assertEqual(controls.steering_updates(loaded[-6])[0], vector())
            path.write_text(conversation.to_json(turns))
            loaded = controls.load_with_steering(str(path), turns, "Raw rank", loaded[-6])
            self.assertIsNone(loaded[-5])
            path.write_text(json.dumps({"format": conversation.SAVE_FORMAT, "turns": [], "steering": {}}))
            refused = controls.load_with_steering(str(path), turns, "Raw rank", loaded[-6])
            self.assertEqual(conversation.turn_entries(refused[1]), conversation.turn_entries(turns))
            self.assertEqual(refused[-5], {"__type__": "update"})
        saved, _ = save_conversation(turns, "", vector())
        self.addCleanup(Path(saved["value"]).unlink)
        self.assertEqual(json.loads(Path(saved["value"]).read_text())["steering"], vector())

    def test_ui_records_response_vector_and_inspects_that_snapshot(self):
        from test_app_flow import FIXED, TURNS, TRACE, CONTEXT_IDS, METRICS, PROMPT_METRICS
        from ui.generation import chat
        from ui.inspection import inspect_layers

        with mock.patch.object(runtime, "MANAGER", manager()):
            frames = list(chat("Hello", [], **FIXED, steering=vector()))
            frame = frames[-1]
            self.assertEqual(frame[TURNS][-1]["steering"], vector())
            self.assertEqual(frame[TRACE]["sampling"]["steering"], vector())
            generation, rows = frame[METRICS]
            context = next(row[CONTEXT_IDS] for row in frames if isinstance(row[CONTEXT_IDS], tuple) and len(row[CONTEXT_IDS]) == 4)
            prompt_metrics = next(row[PROMPT_METRICS] for row in frames[1:] if isinstance(row[PROMPT_METRICS], tuple))
            self.assertEqual(context[3], vector())
            target = dict(generation=generation, strip="response", index=0)
            result = list(inspect_layers(target, frame[METRICS], prompt_metrics, context, 0))[0]
            np.testing.assert_allclose(result[3]["layers"][-1]["probability"], rows[0]["raw_probability"], rtol=1e-5)
            copied = conversation.copy_turns(frame[TURNS])
            copied[-1]["steering"]["vector"][0] = 999
            self.assertEqual(frame[TURNS][-1]["steering"], vector())
