import json
import os
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
from model_runtime import ModelChanged, ModelManager
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


def branch_selection(turns, index=0, position=1):
    """Name one token of the reply at ``position``, as the token view does."""

    metric = conversation.turn_tokens(turns[position])[index]
    return {
        "source": "turn",
        "turn": position,
        "index": index,
        "at_generation": turns[position].get("metrics_generation"),
        "at_token_id": int(metric["token_id"]),
    }


def branch_pick(turns, index=0, position=1):
    """A resample of that token: the same selection, plus the row chosen."""

    metric = conversation.turn_tokens(turns[position])[index]
    return dict(
        branch_selection(turns, index, position),
        position=index + 1,
        token_id=int(metric["token_id"]),
        original_id=int(metric["token_id"]),
        text=metric["text"],
        original=metric["text"],
    )


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
    def test_offline_cleanup_keeps_response_and_inactive_disabled_references(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {library.LIBRARY_PATH_ENV: str(Path(directory) / "conversations.json")}):
            response = steering.compact(vector())
            disabled = steering.compact(dict(vector(), vector=[2.0] * 8, enabled=False))
            unused = steering.compact(dict(vector(), vector=[3.0] * 8))
            forks = conversation.new_forks()
            conversation.put_branch(forks, conversation.MAIN_BRANCH, [dict(conversation.make_turn("assistant", "Hello"), steering=response)])
            conversation.put_branch(forks, "Inactive", [])
            conversation.put_branch_sampling(forks, "Inactive", {"steering": disabled})
            library.write(forks)
            unrelated = steering.asset_directory() / "notes.json"
            unrelated.write_text("keep")
            unused_path = steering.asset_directory() / f"{unused['vector_id']}.json"
            unused_bytes = unused_path.stat().st_size
            self.assertEqual(steering.cleanup_unused_assets(), (1, unused_bytes))
            self.assertFalse(unused_path.exists())
            self.assertEqual(steering.expand(response), vector())
            self.assertEqual(steering.expand(disabled), dict(vector(), vector=[2.0] * 8, enabled=False))
            self.assertEqual(unrelated.read_text(), "keep")

    def test_offline_cleanup_refuses_invalid_library_and_handles_empty_library(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {library.LIBRARY_PATH_ENV: str(Path(directory) / "conversations.json")}):
            reference = steering.compact(vector())
            asset = steering.asset_directory() / f"{reference['vector_id']}.json"
            for invalid in ("not json", '{"format":"future-format"}', '{"format":"chatlab-library-1","branches":null}'):
                library.library_path().write_text(invalid)
                with self.assertRaises(ValueError):
                    steering.cleanup_unused_assets()
                self.assertTrue(asset.exists())
            library.library_path().unlink()
            self.assertEqual(steering.cleanup_unused_assets()[0], 1)
            self.assertFalse(asset.exists())

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
    def test_load_stamp_is_checked_before_vector_compatibility(self):
        held = manager()
        previous_load = held.load_id
        held.model_id = "other/model"
        with self.assertRaises(ModelChanged):
            list(held.generate(MESSAGES, **SAMPLING, steering=vector(), load_id=previous_load))
        with self.assertRaises(ModelChanged):
            held.inspect([0, 1], 1, steering=vector(), load_id=previous_load)
        self.assertFalse(steering.decoder_layers(held.model)[0]._forward_hooks)
        self.assertFalse(held.busy)

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
    def test_incompatible_steering_preserves_response_on_retry_edit_and_branches(self):
        import gradio as gr
        from test_app_flow import FIXED, TURNS, STATUS
        from ui.generation import chat, retry_last, edit_message, branch_from, branch_with_text

        for route in ("retry", "edit", "branch", "typed_branch"):
            for invalid in (dict(vector(), model_id="other/model"), dict(vector(), layer=99), dict(vector(), vector=[1.0])):
                with self.subTest(route=route, invalid=invalid):
                    held = manager()
                    settings = dict(FIXED, assistant_prefill="Hello")
                    with mock.patch.object(runtime, "MANAGER", held):
                        before = list(chat("Hello", [], **settings, steering=vector()))[-1]
                        current = (*settings.values(), steering.compact(invalid), True, 1, invalid["layer"])
                        if route == "retry":
                            stream = retry_last("", before[TURNS], *current)
                        elif route == "edit":
                            event = gr.EditData(None, dict(index=0, previous_value="Hello", value="Changed question"))
                            stream = edit_message(event, "", before[TURNS], *current)
                        elif route == "branch":
                            stream = branch_from(branch_pick(before[TURNS]), "", before[TURNS], *current)
                        else:
                            stream = branch_with_text(branch_selection(before[TURNS]), "Hello", "", before[TURNS], *current)
                        result = list(stream)[-1]
                        self.assertEqual(result[TURNS], before[TURNS])
                        self.assertTrue("Steering failed" in result[STATUS] or "Could not branch" in result[STATUS])
                        self.assertFalse(held.busy)
                        self.assertFalse(steering.decoder_layers(held.model)[0]._forward_hooks)

    def test_incompatible_steering_keeps_new_user_message_without_empty_reply(self):
        from test_app_flow import FIXED, TURNS, STATUS
        from ui.generation import chat

        with mock.patch.object(runtime, "MANAGER", manager()):
            result = list(chat("Hello", [], **FIXED, steering=dict(vector(), layer=99)))[-1]
            self.assertEqual(result[TURNS], [conversation.make_turn("user", "Hello")])
            self.assertIn("Steering failed", result[STATUS])
            self.assertFalse(runtime.MANAGER.busy)

    def test_queued_pane_actions_read_steering_from_latest_gradio_session_state(self):
        import asyncio
        import gradio as gr
        from ui.layout import build_app

        demo = build_app().queue(default_concurrency_limit=1)
        remember = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "remember_steering")

        async def run_pane_action(action):
            pane = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == action)
            self.assertEqual(pane.concurrency_id, remember.concurrency_id)
            demo._queue.create_event_queue_for_fn(remember)
            demo._queue.create_event_queue_for_fn(pane)
            self.assertEqual(demo._queue.event_queue_per_concurrency_id[pane.concurrency_id].concurrency_limit, 1)
            session = gr.blocks.SessionState(demo)
            forks = conversation.new_forks()
            source = "Other" if action == "switch_fork" else conversation.MAIN_BRANCH
            if source == "Other":
                conversation.put_branch(forks, source, [])
                forks["active"] = source
            old = steering.compact(vector())
            forks = controls.store(forks, old)
            session[remember.inputs[0]._id] = forks
            session[remember.inputs[1]._id] = old
            # Construct the later request *before* persistence finishes, with
            # stale state in its payload. Gradio must ignore client-supplied
            # State values and read the updated server session at execution.
            queued = [block.value for block in pane.inputs]
            for index, block in enumerate(pane.inputs):
                if block._id == remember.inputs[0]._id:
                    queued[index] = conversation.copy_forks(forks)
            if action == "switch_fork":
                queued[0] = conversation.MAIN_BRANCH
            await demo.process_api(remember, [forks, old, False, -2, 1], state=session)
            expected = dict(old, enabled=False, strength=-2.0, layer=1)
            await demo.process_api(pane, queued, state=session)
            result = session[remember.inputs[0]._id]
            self.assertEqual(conversation.branch_sampling(result, source)["steering"], expected)
            if action == "fork_conversation":
                self.assertEqual(conversation.branch_sampling(result, result["active"])["steering"], expected)

        for action in ("switch_fork", "fork_conversation"):
            with self.subTest(action=action):
                asyncio.run(run_pane_action(action))

    def test_model_reload_during_steered_branch_restores_original_response(self):
        from test_app_flow import FIXED, TURNS, STATUS
        from ui.generation import chat, branch_from, branch_with_text, BRANCH_MODEL_CHANGED

        for route in ("branch", "typed_branch"):
            with self.subTest(route=route):
                held = manager()
                settings = dict(FIXED, assistant_prefill="Hello")
                with mock.patch.object(runtime, "MANAGER", held):
                    before = list(chat("Hello", [], **settings, steering=vector()))[-1]
                    current = (*settings.values(), steering.compact(vector()), True, 1, 0)
                    if route == "branch":
                        stream = branch_from(branch_pick(before[TURNS]), "", before[TURNS], *current)
                    else:
                        stream = branch_with_text(branch_selection(before[TURNS]), "Hello", "", before[TURNS], *current)
                    generate = held.generate

                    def reloaded_before_generation(*args, **kwargs):
                        held.model_id = "other/model"
                        return generate(*args, **kwargs)

                    with mock.patch.object(held, "generate", side_effect=reloaded_before_generation):
                        result = list(stream)[-1]
                    self.assertEqual(result[STATUS], BRANCH_MODEL_CHANGED)
                    self.assertEqual(result[TURNS], before[TURNS])
                    self.assertFalse(steering.decoder_layers(held.model)[0]._forward_hooks)
                    self.assertFalse(held.busy)

    def test_large_vector_is_stored_once_and_not_copied_into_streamed_turns(self):
        from test_app_flow import FIXED, TURNS, TRACE
        from test_streaming import loaded_manager
        from ui.generation import chat
        from trace_export import trace_to_json

        large = dict(vector(), model_id="fake/model", vector=[i / steering.MAX_WIDTH for i in range(steering.MAX_WIDTH)])
        reference = steering.compact(large)
        path = steering.asset_directory() / f"{reference['vector_id']}.json"
        original_stamp = path.stat().st_mtime_ns
        self.assertGreater(path.stat().st_size, 500_000)
        turns = []
        for _ in range(10):
            turns.extend([conversation.make_turn("user", "Hi"), dict(conversation.make_turn("assistant", "Hello"), steering=reference)])
        forks = controls.store(conversation.new_forks(), reference)
        held = loaded_manager([0, 1, EOS_ID])
        original_generate = held.generate

        def generate(messages, **kwargs):
            # This test measures transport/storage cost, independent of model
            # width; the inference tests cover applying actual vector values.
            self.assertEqual(kwargs.pop("steering"), reference)
            yield from original_generate(messages, **kwargs)

        with mock.patch.object(runtime, "MANAGER", held), mock.patch.object(held, "generate", side_effect=generate):
            frames = list(chat("Hi", turns, **FIXED, steering=reference))
        for frame in frames:
            self.assertLess(len(json.dumps(frame[TURNS])), 8_000)
            for turn in frame[TURNS]:
                if turn.get("steering"):
                    self.assertNotIn("vector", turn["steering"])
            seen = library.as_seen(forks, frame[TURNS])
            self.assertLess(len(library.dump(seen)), 12_000)
            self.assertIsNotNone(library.write(seen))
            self.assertEqual(path.stat().st_mtime_ns, original_stamp)
        self.assertNotIn("vector", frames[-1][TRACE]["sampling"]["steering"])
        exported = json.loads(trace_to_json(frames[-1][TRACE]))
        self.assertEqual(exported["sampling"]["steering"], large)

    def test_portable_conversation_deduplicates_and_restores_into_empty_storage(self):
        reference = steering.compact(vector())
        changed = dict(reference, strength=-2, layer=1)
        turns = [dict(conversation.make_turn("assistant", "Hello"), steering=item) for item in (reference, changed, reference)]
        payload = conversation.to_json(turns, steering=changed)
        data = json.loads(payload)
        self.assertEqual(len(data["steering_vectors"]), 1)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(steering, "asset_directory", return_value=Path(directory)):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                steering.expand(reference)
            loaded, _ = conversation.from_json(payload)
            self.assertEqual(loaded, turns)
            self.assertEqual(steering.expand(changed), dict(vector(), strength=-2, layer=1))
            assets = list(Path(directory).glob("*.json"))
            self.assertEqual(len(assets), 1)
            # Reading the compact library after restart needs no embedded copy.
            restored = library.parse(library.dump(controls.store(conversation.new_forks(), changed)))
            self.assertEqual(steering.expand(controls.steering_updates(restored)[0]), dict(vector(), strength=-2, layer=1))
            assets[0].write_text("damaged")
            with self.assertRaisesRegex(ValueError, "integrity"):
                steering.expand(reference)
            conversation.from_json(payload)
            self.assertEqual(steering.expand(reference), vector())

    def test_portable_conversation_rejects_missing_or_changed_vector_data(self):
        reference = steering.compact(vector())
        data = json.loads(conversation.to_json([dict(conversation.make_turn("assistant", "Hello"), steering=reference)]))
        missing = dict(data, steering_vectors={})
        with self.assertRaisesRegex(ValueError, "missing"):
            conversation.from_json(json.dumps(missing))
        data["steering_vectors"][reference["vector_id"]]["vector"][0] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            conversation.from_json(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "identifier"):
            steering.normalize(dict(reference, vector_id="../../not-a-vector"))

    def test_generation_paths_use_visible_controls_before_persistence_catches_up(self):
        from test_app_flow import FIXED, TURNS, TRACE
        from ui.generation import chat, retry_last, branch_from, branch_with_text

        for route in ("send", "retry", "branch", "typed_branch"):
            for overrides in ((False, 1, 0), (True, 1, 0), (True, -2, 0), (True, 1, 1)):
                with self.subTest(route=route, overrides=overrides):
                    held = manager()
                    stale = vector(enabled=overrides != (True, 1, 0))
                    original = steering.normalize(stale)
                    expected = dict(stale, enabled=overrides[0], strength=overrides[1], layer=overrides[2])
                    with mock.patch.object(runtime, "MANAGER", held):
                        # Keep a visible prefix even if the tiny random model
                        # samples EOS immediately with steering turned off.
                        settings = dict(FIXED, assistant_prefill="Hello")
                        before = list(chat("Hello", [], **settings, steering=stale))[-1]
                        current = (*settings.values(), stale, *overrides)
                        if route == "send":
                            stream = chat("Hello", [], *current)
                        elif route == "retry":
                            stream = retry_last("", before[TURNS], *current)
                        elif route == "branch":
                            stream = branch_from(branch_pick(before[TURNS]), "", before[TURNS], *current)
                        else:
                            stream = branch_with_text(branch_selection(before[TURNS]), "Hello", "", before[TURNS], *current)
                        with mock.patch.object(held, "generate", wraps=held.generate) as generate:
                            result = list(stream)[-1]
                        self.assertEqual(steering.expand(generate.call_args.kwargs["steering"]), expected)
                        self.assertEqual(steering.expand(result[TURNS][-1]["steering"]), expected)
                        self.assertEqual(steering.expand(result[TRACE]["sampling"]["steering"]), expected)
                        self.assertEqual(stale, original)

    def test_generation_and_save_listeners_capture_all_visible_steering_controls(self):
        from ui.layout import build_app

        demo = build_app()
        listeners = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") in (
            "chat", "retry_last", "retry_message", "edit_message", "branch_from", "branch_with_text", "save_conversation",
        )]
        self.assertEqual(len(listeners), 8)  # Send and Enter each call chat.
        for listener in listeners:
            inputs = [item for item in listener.inputs if item.label is not None]
            if getattr(listener.fn, "__name__", "") != "save_conversation":
                self.assertEqual(inputs[-1].label, "Thinking mode")
                inputs = inputs[:-1]
            self.assertEqual([item.label for item in inputs[-3:]], [
                "Enable steering", "Steering strength", "Target layer (starting at 0)",
            ])

    def test_save_uses_visible_controls_before_persistence_catches_up(self):
        turns = [conversation.make_turn("user", "Hello")]
        stale = vector()
        saved, _ = save_conversation(turns, "", stale, False, -2, 1)
        self.addCleanup(Path(saved["value"]).unlink)
        self.assertEqual(steering.expand(json.loads(Path(saved["value"]).read_text())["steering"]), dict(stale, enabled=False, strength=-2, layer=1))
        self.assertEqual(stale, vector())

    def test_fork_switch_new_chat_and_library_roundtrip(self):
        forks = controls.store(conversation.new_forks(), vector())
        turns = [conversation.make_turn("user", "Hello")]
        forked = fork_conversation(turns, forks, None)[3]
        self.assertEqual(steering.expand(controls.steering_updates(forked)[0]), vector())
        changed, state, _ = controls.remember_steering(forked, vector(), True, -2, 1)
        self.assertEqual(state["strength"], -2)
        changed["active"] = conversation.MAIN_BRANCH
        self.assertEqual(steering.expand(controls.steering_updates(changed)[0]), vector())
        restored = library.parse(library.dump(changed))
        self.assertEqual(steering.expand(controls.steering_updates(restored)[0]), vector())
        fresh = new_conversation(turns, restored)[3]
        self.assertIsNone(controls.steering_updates(fresh)[0])
        copied = conversation.copy_forks(forks)
        copied["sampling"][conversation.MAIN_BRANCH]["steering"]["layer"] = 99
        self.assertEqual(steering.expand(controls.steering_updates(forks)[0]), vector())

    def test_portable_save_load_keeps_current_and_per_response_vectors(self):
        turns = [dict(conversation.make_turn("assistant", "Hello"), steering=vector(strength=-2))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.json"
            path.write_text(conversation.to_json(turns, steering=vector()))
            loaded = controls.load_with_steering(str(path), [], "Raw rank", conversation.new_forks())
            self.assertEqual(conversation.turn_entries(loaded[1]), conversation.turn_entries(turns))
            self.assertEqual(steering.expand(loaded[-5]), vector())
            self.assertEqual(steering.expand(controls.steering_updates(loaded[-6])[0]), vector())
            path.write_text(conversation.to_json(turns))
            loaded = controls.load_with_steering(str(path), turns, "Raw rank", loaded[-6])
            self.assertIsNone(loaded[-5])
            path.write_text(json.dumps({"format": conversation.SAVE_FORMAT, "turns": [], "steering": {}}))
            refused = controls.load_with_steering(str(path), turns, "Raw rank", loaded[-6])
            self.assertEqual(conversation.turn_entries(refused[1]), conversation.turn_entries(turns))
            self.assertEqual(refused[-5], {"__type__": "update"})
        saved, _ = save_conversation(turns, "", vector())
        self.addCleanup(Path(saved["value"]).unlink)
        self.assertEqual(steering.expand(json.loads(Path(saved["value"]).read_text())["steering"]), vector())

    def test_ui_records_response_vector_and_inspects_that_snapshot(self):
        from test_app_flow import FIXED, TURNS, TRACE, CONTEXT_IDS, METRICS, PROMPT_METRICS
        from ui.generation import chat
        from ui.inspection import inspect_layers

        with mock.patch.object(runtime, "MANAGER", manager()):
            frames = list(chat("Hello", [], **FIXED, steering=vector()))
            frame = frames[-1]
            self.assertEqual(steering.expand(frame[TURNS][-1]["steering"]), vector())
            self.assertEqual(steering.expand(frame[TRACE]["sampling"]["steering"]), vector())
            generation, rows = frame[METRICS]
            context = next(row[CONTEXT_IDS] for row in frames if isinstance(row[CONTEXT_IDS], tuple) and len(row[CONTEXT_IDS]) == 4)
            prompt_metrics = next(row[PROMPT_METRICS] for row in frames[1:] if isinstance(row[PROMPT_METRICS], tuple))
            self.assertEqual(steering.expand(context[3]), vector())
            target = dict(generation=generation, strip="response", index=0)
            result = list(inspect_layers(target, frame[METRICS], prompt_metrics, context, 0))[0]
            np.testing.assert_allclose(result[3]["layers"][-1]["probability"], rows[0]["raw_probability"], rtol=1e-5)
            copied = conversation.copy_turns(frame[TURNS])
            copied[-1]["steering"]["layer"] = 999
            self.assertEqual(steering.expand(frame[TURNS][-1]["steering"]), vector())
