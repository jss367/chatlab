"""Direction edits on a small real decoder, with synthetic directions, vector and lens."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from chatlab import steering
from chatlab.extension_api import (ExtensionContext, ModelService, NavigationService, ProjectionCancelled,
                                   TokenInspector)
from chatlab.extensions.direction_edits import experiment, files
from chatlab.extensions.direction_edits.page import build_page
from chatlab.extensions.probes import probe as probes
from chatlab.extensions.registry import load_enabled
from chatlab.model_runtime import ModelManager
from tiny_tokenizer import build
from ui_support import handlers_by_name

MODEL = "test/tiny-decoder"
WIDTH, BLOCKS = 16, 4
PASSAGE = "the cat sat on the mat and the dog sat on the log"
LENS_BLOCKS = (1, 2, 3)


def tiny_manager():
    torch.manual_seed(0)
    manager = ModelManager()
    manager.tokenizer = build()
    config = LlamaConfig(
        vocab_size=len(manager.tokenizer), hidden_size=WIDTH, intermediate_size=32,
        num_hidden_layers=BLOCKS, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    manager.model = LlamaForCausalLM(config).eval()
    manager.model_id = MODEL
    manager.precision = "full"
    return manager


def unit_rows(seed=1):
    rows = np.random.default_rng(seed).normal(size=(BLOCKS, WIDTH))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def direction_set(**changes):
    rows = np.random.default_rng(1).normal(size=(BLOCKS, WIDTH)) * 3
    value = {"format": files.DIRECTIONS_FORMAT, "name": "detector", "model_id": MODEL, "model_revision": None,
             "precision": "full", "directions": [{"layer": layer, "vector": row.tolist()}
                                                 for layer, row in enumerate(rows)]}
    return files.normalize_directions(value | changes)


def vector(layer=0, model_id=MODEL):
    values = np.random.default_rng(2).normal(size=WIDTH)
    return {"format": "chatlab-steering-1", "model_id": model_id, "layer": layer, "vector": values.tolist(),
            "strength": 1.0, "enabled": True}


def inputs(**changes):
    value = {
        "passage": PASSAGE,
        "conditions": [{"name": "neutral", "prefix": ""}, {"name": "focus", "prefix": "the dog\n"}],
        "injection": {"vector": vector(), "strength": 4.0, "tokens": [2, 6]},
        "directions": direction_set(),
        "edit": {"mode": experiment.ERASE, "value": 0.0, "blocks": [1, 1], "tokens": [3, 5],
                 "random_control": False, "seed": 7},
        "readout": None,
        "differences": [],
    }
    return value | changes


def run(session, value):
    steps = experiment.run(session, value)
    lines = []
    while True:
        try:
            lines.append(next(steps))
        except StopIteration as done:
            return done.value, lines


def hooks_left(manager):
    return [block for block in steering.decoder_layers(manager.model)
            if block._forward_hooks or block._forward_pre_hooks]


class EditDeviceTests(unittest.TestCase):
    def test_injection_refuses_scaled_and_cast_overflow(self):
        hidden = torch.zeros(1, 3, WIDTH, dtype=torch.float16)
        for entry, strength in ((1e5, 1.0), (1.0, 1e5), (1e308, 1e308)):
            with self.subTest(entry=entry, strength=strength):
                injection = {"vector": {"vector": [entry] * WIDTH}, "strength": strength, "tokens": [1, 3]}
                hooks = experiment.Hooks(injection, None, None, 0)
                with self.assertRaisesRegex(ValueError, "overflows.*activation precision"):
                    hooks._inject(hidden)
                self.assertTrue(torch.equal(hidden, torch.zeros_like(hidden)))
        # A large vector with a small, representable product must remain usable.
        hooks = experiment.Hooks({"vector": {"vector": [1e5] * WIDTH}, "strength": 1e-3,
                                  "tokens": [2, 2]}, None, None, 0)
        changed = hooks._inject(hidden)
        torch.testing.assert_close(changed[0, 1], torch.full((WIDTH,), 100.0, dtype=torch.float16))
        self.assertTrue(torch.equal(changed[0, [0, 2]], hidden[0, [0, 2]]))

    def test_edits_refuse_values_that_overflow_the_activations(self):
        hidden = torch.zeros(1, 3, WIDTH, dtype=torch.float16)
        unit = unit_rows()
        for mode in (experiment.CLAMP, experiment.ADD):
            edit = inputs()["edit"] | {"mode": mode, "tokens": [1, 3]}
            with self.subTest(mode=mode):
                hooks = experiment.Hooks(None, edit | {"value": 1e7}, unit, 0)
                with self.assertRaisesRegex(ValueError, "overflows.*activation precision"):
                    hooks.edited(1, None, {})(hidden)
                hooks = experiment.Hooks(None, edit | {"value": 10.0}, unit, 0)
                changed = hooks.edited(1, None, {})(hidden)
                self.assertTrue(torch.isfinite(changed).all())
        # The random control moves by the recorded sizes, so it is held to the same limit.
        hooks = experiment.Hooks(None, inputs()["edit"] | {"tokens": [1, 3]}, unit, 0)
        with self.assertRaisesRegex(ValueError, "overflows.*activation precision"):
            hooks.control(1, np.full(3, 1e7), unit[2])(hidden)
        # Activations already infinite before the edit are not the edit's doing.
        broken = hidden.clone()
        broken[0, 0, 0] = float("inf")
        changed = experiment.Hooks(None, inputs()["edit"] | {"mode": experiment.ADD, "value": 1.0,
                                                             "tokens": [1, 3]}, unit, 0).edited(1, None, {})(broken)
        self.assertTrue(torch.isfinite(changed[0, 1:]).all())

    def test_recording_moves_sizes_to_cpu_before_widening(self):
        # Emulate a device that supports the float32 edit but refuses float64,
        # so this regression runs on CPU-only CI as well as Apple hardware.
        class Float32DeviceSizes:
            def __init__(self, values):
                self.values = values

            def to(self, *, device, dtype):
                self.assert_dtype(dtype)
                return self

            def assert_dtype(self, dtype):
                if dtype != torch.float32:
                    raise TypeError("Device does not support float64")

            def double(self):
                raise TypeError("Device does not support float64")

            def cpu(self):
                return self.values.cpu()

            def __getitem__(self, index):
                return self.values[index]

        hidden = torch.zeros(1, 3, WIDTH)
        unit = unit_rows().astype(np.float32)
        edit = inputs()["edit"] | {"tokens": [1, 3]}
        hooks = experiment.Hooks(None, edit, unit, 0)
        sizes = torch.tensor([1.0, -2.0, 3.0])
        recorded = {}
        changed = hooks._move(hidden, 1, lambda _coordinate, _count: Float32DeviceSizes(sizes),
                              unit[1], recorded)
        np.testing.assert_array_equal(recorded[1], sizes.numpy())
        self.assertEqual(recorded[1].dtype, np.float64)
        torch.testing.assert_close(changed[0], sizes[:, None] * torch.tensor(unit[1]))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_every_edit_records_sizes_on_mps(self):
        unit = unit_rows().astype(np.float32)
        hidden = torch.zeros(1, 3, WIDTH, device="mps")
        for mode in experiment.MODES:
            with self.subTest(mode=mode):
                edit = inputs()["edit"] | {"mode": mode, "value": 2.0, "tokens": [1, 3]}
                hooks = experiment.Hooks(None, edit, unit, 0)
                recorded = {}
                changed = hooks.edited(1, np.ones(3), recorded)(hidden)
                self.assertEqual(changed.device.type, "mps")
                self.assertEqual(recorded[1].dtype, np.float64)
                expected = 1.0 if mode == experiment.ERASE else 2.0
                np.testing.assert_allclose(recorded[1], expected)
                control = hooks.control(1, recorded[1], experiment.random_direction(7, 1, unit[1]))(hidden)
                torch.testing.assert_close(control.cpu().norm(dim=-1), changed.cpu().norm(dim=-1))


class LensFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.manager = tiny_manager()
        self.service = ModelService(lambda: self.manager)
        torch.manual_seed(3)
        self.matrices = {layer: torch.randn(WIDTH, WIDTH) for layer in LENS_BLOCKS}

    def import_lens(self):
        path = self.directory / "lens.pt"
        torch.save({"J": self.matrices, "source_layers": list(LENS_BLOCKS), "n_prompts": 10, "d_model": WIDTH}, path)
        return self.manager.import_jacobian_lens(str(path), MODEL)

    def expected_log_probs(self, ids, targets, blocks, positions, edits=None):
        """The lens readout computed by hand from captured block outputs."""
        layers = steering.decoder_layers(self.manager.model)
        captured, handles = {}, []
        for layer, edit in (edits or {}).items():
            def edit_hook(_module, _inputs, output, edit=edit):
                return (edit(output[0]), *output[1:]) if isinstance(output, tuple) else edit(output)
            handles.append(layers[layer].register_forward_hook(edit_hook))
        for layer in blocks:
            def capture(_module, _inputs, output, layer=layer):
                captured[layer] = (output[0] if isinstance(output, tuple) else output)[0].detach().clone()
            handles.append(layers[layer].register_forward_hook(capture))
        try:
            with torch.no_grad():
                self.manager.model(torch.tensor([ids]), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        answer = np.empty((len(blocks), len(positions), len(targets)))
        model = self.manager.model
        with torch.no_grad():
            for row, layer in enumerate(blocks):
                logits = model.lm_head(model.model.norm(captured[layer][positions] @ self.matrices[layer].T))
                log_probs = torch.log_softmax(logits.double(), dim=-1)
                normalizer = torch.logsumexp(logits.double(), dim=-1)
                for column, group in enumerate(targets):
                    group = [group] if isinstance(group, int) else group
                    answer[row, :, column] = (logits[:, group].double().mean(dim=-1) - normalizer).numpy()
                    if len(group) == 1:
                        np.testing.assert_allclose(answer[row, :, column], log_probs[:, group[0]].numpy())
        return answer


class LensAccessorTests(LensFixture):
    def test_log_probabilities_match_the_lens_read_by_hand(self):
        self.import_lens()
        ids = self.manager.tokenizer.encode(PASSAGE)
        targets = [ids[2], [ids[3], ids[4]]]
        with self.service.open_session() as session:
            self.assertEqual(session.jacobian_lens(), {"name": "lens.pt", "n_prompts": 10, "layers": [1, 2, 3]})
            got = session.lens_log_probs(ids, targets, [1, 3], [0, 4, 6])
        np.testing.assert_allclose(got, self.expected_log_probs(ids, targets, [1, 3], [0, 4, 6]), atol=1e-5)
        self.assertEqual(hooks_left(self.manager), [])

    def test_the_lens_reads_through_the_callers_hooks_inside_the_held_model(self):
        self.import_lens()
        ids = self.manager.tokenizer.encode(PASSAGE)
        shift = torch.randn(WIDTH)

        def edit(hidden):
            changed = hidden.clone()
            changed[0, 2:5] += shift
            return changed

        with self.service.open_session() as session:
            plain = session.lens_log_probs(ids, [ids[1]], [2], [1, 3, 5])
            with session.transformers_model():
                with session.block_hooks({1: edit}):
                    edited = session.lens_log_probs(ids, [ids[1]], [2], [1, 3, 5])
                    self.assertTrue(self.manager._lock.locked())
            self.assertFalse(self.manager._lock.locked())
        np.testing.assert_allclose(edited, self.expected_log_probs(ids, [ids[1]], [2], [1, 3, 5], {1: edit}), atol=1e-5)
        # Position 1 comes before the edit, so only the later positions move.
        np.testing.assert_allclose(edited[:, 0], plain[:, 0], atol=1e-6)
        self.assertFalse(np.allclose(edited[:, 1:], plain[:, 1:]))
        self.assertEqual(hooks_left(self.manager), [])

    def test_missing_lens_unfitted_blocks_and_bad_targets_are_refused(self):
        ids = self.manager.tokenizer.encode(PASSAGE)
        with self.service.open_session() as session:
            self.assertIsNone(session.jacobian_lens())
            with self.assertRaisesRegex(ValueError, "Import a Jacobian lens"):
                session.lens_log_probs(ids, [ids[0]], [1], [0])
        self.import_lens()
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "no matrix for block 0"):
                session.lens_log_probs(ids, [ids[0]], [0, 1], [0])
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                session.lens_log_probs(ids, [10**7], [1], [0])
            with self.assertRaisesRegex(ValueError, "positions between"):
                session.lens_log_probs(ids, [ids[0]], [1], [len(ids)])
        self.assertEqual(hooks_left(self.manager), [])

    def test_a_remembered_lens_is_recalled_for_the_session(self):
        record = {"path": str(self.directory / "lens.pt"), "fitted_model_id": MODEL}
        self.import_lens()
        self.manager._jacobian_lens = None
        with mock.patch("chatlab.jacobian_lens.remembered", return_value=record):
            with self.service.open_session() as session:
                self.assertEqual(session.jacobian_lens()["layers"], [1, 2, 3])


class HookTests(LensFixture):
    def test_hooks_are_gone_after_an_exception_in_the_body_or_a_hook(self):
        ids = self.manager.tokenizer.encode(PASSAGE)
        unit = unit_rows()
        with self.service.open_session() as session:
            with self.assertRaisesRegex(RuntimeError, "body"):
                with session.transformers_model():
                    with session.block_hooks({1: lambda hidden: hidden * 2, 2: lambda hidden: None}):
                        self.assertEqual(len(hooks_left(self.manager)), 2)
                        raise RuntimeError("body")
            self.assertEqual(hooks_left(self.manager), [])

            def broken(_hidden):
                raise ValueError("hook failed")

            with self.assertRaisesRegex(ValueError, "hook failed"):
                with session.transformers_model():
                    with session.block_hooks({2: broken}):
                        session.project_layers(ids, unit)
            self.assertEqual(hooks_left(self.manager), [])
            # Hooks still entered when the model is let go are removed with it.
            left_open = session.block_hooks({0: lambda hidden: hidden})
            with session.transformers_model():
                left_open.__enter__()
                self.assertEqual(len(hooks_left(self.manager)), 1)
            self.assertEqual(hooks_left(self.manager), [])
            left_open.__exit__(None, None, None)
            self.assertFalse(self.manager._lock.locked())
            self.assertEqual(session.project_layers(ids, unit).shape, (BLOCKS, len(ids)))

    def test_hooks_need_the_held_model_and_lock_takers_are_refused_inside_it(self):
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "inside session.transformers_model"):
                with session.block_hooks({0: lambda hidden: hidden}):
                    pass
            with session.transformers_model():
                with self.assertRaisesRegex(ValueError, "outside"):
                    session.read_examples(["the cat"])
                with self.assertRaisesRegex(ValueError, "outside"):
                    session.jacobian_lens()
                with self.assertRaisesRegex(ValueError, "Hook blocks 0–3"):
                    with session.block_hooks({4: lambda hidden: hidden}):
                        pass
                with self.assertRaisesRegex(ValueError, "shaped like"):
                    with session.block_hooks({0: lambda hidden: hidden[:, :1]}):
                        session.project_layers(self.manager.tokenizer.encode("the cat"), unit_rows())
                session.check_projection(unit_rows())
            self.assertEqual(hooks_left(self.manager), [])

    def test_a_stopped_run_lets_go_of_the_model_and_its_hooks(self):
        with self.service.open_session() as session:
            calls = []

            def stop_on_the_third_pass(_module, _inputs, _output):
                calls.append(1)
                if len(calls) == 3:
                    session.cancel()

            # Block 3 runs once per pass; the third pass is the edited one.
            handle = steering.decoder_layers(self.manager.model)[3].register_forward_hook(stop_on_the_third_pass)
            try:
                with self.assertRaises(ProjectionCancelled):
                    run(session, inputs())
            finally:
                handle.remove()
            self.assertEqual(hooks_left(self.manager), [])
            self.assertFalse(self.manager._lock.locked())


class ExperimentTests(LensFixture):
    def capture(self, layer):
        """Every pass's output of one block, after the experiment's own hooks."""
        seen = []
        handle = steering.decoder_layers(self.manager.model)[layer].register_forward_hook(
            lambda _module, _inputs, output: seen.append(
                (output[0] if isinstance(output, tuple) else output)[0].detach().clone()))
        self.addCleanup(handle.remove)
        return seen

    def test_erase_sets_the_edited_coordinate_to_the_reference(self):
        with self.service.open_session() as session:
            result, lines = run(session, inputs(edit=inputs()["edit"] | {"blocks": [1, 2]}))
        self.assertEqual(len(lines), 6)
        for condition in result["conditions"]:
            passes = condition["coordinates"]
            for block in (1, 2):
                np.testing.assert_allclose(passes["edited"][block][2:5], passes["reference"][block][2:5], atol=1e-5)
                # The injection moved them first, so there was something to erase.
                self.assertGreater(np.abs(np.subtract(passes["injected"][block][2:5],
                                                      passes["reference"][block][2:5])).max(), 1e-2)
            self.assertAlmostEqual(condition["recovery"]["edited"][1], 0.0, places=5)
        self.assertEqual(result["edited_blocks"], [1, 2])
        self.assertEqual(hooks_left(self.manager), [])

    def test_untouched_positions_and_blocks_are_bit_identical_to_the_injected_pass(self):
        with self.service.open_session() as session:
            result, _ = run(session, inputs())
        for condition in result["conditions"]:
            passes = condition["coordinates"]
            # Before the first edited block, everything.
            self.assertEqual(passes["edited"][0], passes["injected"][0])
            # At it, every token outside the edit's 3–5.
            edited, injected = passes["edited"][1], passes["injected"][1]
            self.assertEqual(edited[:2] + edited[5:], injected[:2] + injected[5:])
            # Later blocks see the edit through attention, but not before it.
            for block in (2, 3):
                self.assertEqual(passes["edited"][block][:2], passes["injected"][block][:2])

    def test_injection_touches_only_its_tokens(self):
        seen = self.capture(0)
        value = inputs(edit=inputs()["edit"] | {"blocks": [2, 2]})
        with self.service.open_session() as session:
            run(session, value)
        added = torch.tensor(vector()["vector"]) * 4.0
        for condition, offset in zip(range(2), (0, len(self.manager.tokenizer.encode("the dog\n")))):
            reference, injected = seen[3 * condition], seen[3 * condition + 1]
            inside = slice(offset + 1, offset + 6)
            torch.testing.assert_close(injected[inside] - reference[inside], added.expand(5, -1), atol=1e-5, rtol=0)
            outside = torch.ones(len(reference), dtype=torch.bool)
            outside[inside] = False
            self.assertTrue(torch.equal(injected[outside], reference[outside]))

    def test_random_control_makes_an_edit_of_the_same_size_per_token(self):
        seen = self.capture(1)
        value = inputs(edit=inputs()["edit"] | {"random_control": True})
        with self.service.open_session() as session:
            result, lines = run(session, value)
        self.assertEqual(len(lines), 8)
        unit = files.matrix(value["directions"], BLOCKS)[0][1]
        for condition, offset in zip(range(2), (0, len(self.manager.tokenizer.encode("the dog\n")))):
            injected, edited, randomized = seen[4 * condition + 1:4 * condition + 4]
            rows = slice(offset + 2, offset + 5)
            real, control = edited[rows] - injected[rows], randomized[rows] - injected[rows]
            torch.testing.assert_close(control.norm(dim=-1), real.norm(dim=-1), atol=1e-5, rtol=1e-5)
            self.assertGreater(float(real.norm(dim=-1).min()), 1e-3)
            # The control is at right angles to the direction, so it leaves the coordinate alone.
            torch.testing.assert_close(control @ torch.tensor(unit), torch.zeros(3), atol=1e-5, rtol=0)
            self.assertTrue(torch.equal(randomized[:offset + 2], injected[:offset + 2]))
            self.assertIsNotNone(result["conditions"][condition]["recovery"]["random"][2])
        again = experiment.random_direction(7, 1, unit)
        np.testing.assert_allclose(again, experiment.random_direction(7, 1, unit))
        self.assertFalse(np.allclose(again, experiment.random_direction(8, 1, unit)))

    def test_clamp_and_add_move_the_coordinate_where_they_say(self):
        with self.service.open_session() as session:
            clamped, _ = run(session, inputs(edit=inputs()["edit"] | {"mode": experiment.CLAMP, "value": 2.5}))
            added, _ = run(session, inputs(injection=None,
                                           edit=inputs()["edit"] | {"mode": experiment.ADD, "value": 1.5}))
        for condition in clamped["conditions"]:
            np.testing.assert_allclose(condition["coordinates"]["edited"][1][2:5], [2.5] * 3, atol=1e-5)
        for condition in added["conditions"]:
            passes = condition["coordinates"]
            self.assertIs(passes["injected"], passes["reference"])
            np.testing.assert_allclose(np.subtract(passes["edited"][1][2:5], passes["reference"][1][2:5]),
                                       [1.5] * 3, atol=1e-5)
            # With nothing injected there is no shift to recover.
            self.assertEqual(set(condition["recovery"]["edited"]), {None})

    def test_the_lens_table_and_differences(self):
        self.import_lens()
        value = inputs(readout={"word": " sat", "tokens": [1, 8], "blocks": [2, 3]},
                       differences=experiment.parse_differences("focus = focus - neutral"),
                       edit=inputs()["edit"] | {"random_control": True})
        with self.service.open_session() as session:
            result, _ = run(session, value)
            ids = session.example_ids("the dog\n") + session.encode(PASSAGE)
            offset = len(session.example_ids("the dog\n"))
            target = session.encode(" sat")
        self.assertEqual(result["target"]["token_ids"], target)
        focus = result["conditions"][1]
        injection = vector()["vector"]

        def inject(hidden):
            changed = hidden.clone()
            changed[0, offset + 1:offset + 6] += 4.0 * torch.tensor(injection)
            return changed

        expected = self.expected_log_probs(ids, [target], [2, 3], list(range(offset, offset + 8)), {0: inject})
        self.assertAlmostEqual(focus["lens"]["no_edit"], float(expected.mean()), places=5)
        self.assertEqual({key for key, item in focus["lens"].items() if item is None}, set())
        difference = result["differences"][0]
        self.assertAlmostEqual(difference["edit"], focus["lens"]["edit"] - result["conditions"][0]["lens"]["edit"])
        self.assertAlmostEqual(difference["edit_change"],
                               100 * (difference["edit"] - difference["no_edit"]) / abs(difference["no_edit"]))

    def test_the_download_round_trips(self):
        self.import_lens()
        value = inputs(readout={"word": " sat", "tokens": [1, 8], "blocks": [1, 3]},
                       differences=experiment.parse_differences("focus = focus - neutral"))
        with self.service.open_session() as session:
            result, _ = run(session, value)
        path = self.directory / "result.json"
        path.write_text(files.dumps(result))
        read = files.read_result(path)
        self.assertEqual(read, json.loads(json.dumps(result)))
        self.assertEqual(read["passage_ids"], self.manager.tokenizer.encode(PASSAGE))
        self.assertEqual(read["inputs"]["directions"], value["directions"])


class RefusalTests(LensFixture):
    def test_files_for_another_model_or_revision_are_refused(self):
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "made for other/model"):
                run(session, inputs(directions=direction_set(model_id="other/model")))
            with self.assertRaisesRegex(ValueError, "other/model"):
                run(session, inputs(injection=inputs()["injection"] | {"vector": vector(model_id="other/model")}))
        with mock.patch.object(self.manager, "model_revision", return_value="a" * 40):
            with self.service.open_session() as session:
                with self.assertRaisesRegex(ValueError, "revision bbbbbbbbbbbb"):
                    run(session, inputs(directions=direction_set(model_revision="b" * 40)))
                # A file that does not record its revision is taken at its word.
                run(session, inputs(directions=direction_set(model_revision=None)))
        self.assertEqual(hooks_left(self.manager), [])

    def test_a_probe_file_reads_as_one_unit_direction_per_block(self):
        layers, folds = probes.train(np.random.default_rng(0).normal(size=(6, BLOCKS, WIDTH)),
                                     np.random.default_rng(1).normal(size=(6, BLOCKS, WIDTH)))
        probe = probes.build(name="probe", model_id="other/model", positive_label="Yes", negative_label="No",
                             positive_examples=[f"a{i}" for i in range(6)],
                             negative_examples=[f"b{i}" for i in range(6)], pool="last", chat_template=False,
                             l2=1.0, layers=layers, folds=folds)
        path = self.directory / "probe.json"
        path.write_text(probes.dumps(probe))
        directions = files.read_directions(path)
        self.assertEqual([item["layer"] for item in directions["directions"]], list(range(BLOCKS)))
        for item, layer in zip(directions["directions"], probe["layers"]):
            np.testing.assert_allclose(item["vector"], np.divide(layer["weights"], np.linalg.norm(layer["weights"])))
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "made for other/model"):
                run(session, inputs(directions=directions))

    def test_bad_direction_files_and_settings_are_refused(self):
        base = {"format": files.DIRECTIONS_FORMAT, "name": "d", "model_id": MODEL,
                "directions": [{"layer": 0, "vector": [1.0] * WIDTH}]}
        for change, message in (
            ({"directions": [{"layer": 0, "vector": [0.0] * WIDTH}]}, "no length"),
            ({"directions": [{"layer": 0, "vector": [1.0]}, {"layer": 1, "vector": [1.0, 2.0]}]}, "same width"),
            ({"directions": [{"layer": 0, "vector": [1.0]}, {"layer": 0, "vector": [1.0]}]}, "more than one"),
            ({"format": "other"}, "Expected"),
            ({"precision": "2-bit"}, "precision"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                files.normalize_directions(base | change)
        with self.service.open_session() as session:
            for value, message in (
                (inputs(directions=files.normalize_directions(base)), "none at block 1"),
                (inputs(edit=inputs()["edit"] | {"tokens": [3, 99]}), "reach token 99"),
                (inputs(edit=inputs()["edit"] | {"blocks": [1, 4]}), "blocks are 0–3"),
                (inputs(readout={"word": " sat", "tokens": [1, 2], "blocks": [1, 1]}), "Import a Jacobian lens"),
                (inputs(differences=[{"name": "x", "minuend": "neutral", "subtrahend": "nobody"}]), "nobody"),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    run(session, value)
            with mock.patch.object(experiment, "READ_LIMIT", 8):
                with self.assertRaisesRegex(ValueError, "above the 8"):
                    run(session, inputs())
        self.import_lens()
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "no matrix for block 0"):
                run(session, inputs(readout={"word": " sat", "tokens": [1, 2], "blocks": [0, 1]}))
        with self.assertRaisesRegex(ValueError, "suppression = neutral - ignore"):
            experiment.parse_differences("suppression neutral - ignore")

    def test_mlx_and_quantized_loads_are_refused(self):
        with mock.patch.object(self.manager, "_engine", return_value=mock.Mock(backend="mlx")):
            with self.service.open_session() as session:
                with self.assertRaisesRegex(ValueError, "MLX"):
                    run(session, inputs())
        self.manager.precision = "4-bit"
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, "full-precision"):
                run(session, inputs())


class PageTests(LensFixture):
    def setUp(self):
        super().setUp()
        context = ExtensionContext(self.service, TokenInspector(), self.directory / "data",
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        self.fn = handlers_by_name(demo)

    def test_the_extension_is_catalogued_and_loads(self):
        loaded, errors = load_enabled({"direction_edits"})
        self.assertEqual(errors, [])
        self.assertEqual(loaded[0].spec.page_label, "Edits")

    def test_a_run_shows_its_readings_and_its_download_opens_again(self):
        self.import_lens()
        directions_path = self.directory / "directions.json"
        directions_path.write_text(json.dumps(direction_set()))
        directions, note = self.fn["open_directions"](str(directions_path))
        self.assertIn("4 unit directions", note)
        vector_path = self.directory / "vector.json"
        vector_path.write_text(json.dumps(vector()))
        loaded_vector, _, strength = self.fn["open_vector"](str(vector_path))
        self.assertEqual(strength, 1.0)
        controls = [PASSAGE, directions, loaded_vector, 4.0, 2, 6, experiment.ERASE, 0.0, 1, 1, 3, 5, True, 7,
                    " sat", 1, 8, 1, 3, "focus = focus - neutral"]
        rows = ["neutral", "", "focus", "the dog\n"] + ["", ""] * 4
        frames = list(self.fn["run_experiment"]("owner", *controls, *rows))
        self.assertEqual(len(frames), 9)
        status, result, headline, picker, heat, recovery, lens_note, lens, diffs, download = frames[-1]
        self.assertEqual(status, "Done.")
        self.assertEqual([c["name"] for c in result["conditions"]], ["neutral", "focus"])
        self.assertIn("block 3", headline)
        self.assertIn("Injected and edited", heat)
        self.assertIn("1 (edited)", recovery)
        self.assertIn("one token", lens_note)
        self.assertIn("Random control", lens)
        self.assertIn("focus: focus − neutral", diffs)
        self.assertFalse(self.manager._lock.locked())
        self.assertEqual(hooks_left(self.manager), [])
        reopened = self.fn["open_result"](download, "owner")
        self.assertEqual(reopened[1], json.loads(Path(download).read_text()))
        self.assertEqual(reopened[4], heat)
        self.assertIn("focus", self.fn["pick"](result, 1))

    def test_controls_are_checked_before_the_model_is_claimed(self):
        controls = [PASSAGE, None, None, 1.0, 1, 2, experiment.ERASE, 0.0, 1, 1, 1, 2, False, 0, "", 1, 2, 1, 1, ""]
        with self.assertRaisesRegex(gr.Error, "Open a direction file"):
            list(self.fn["run_experiment"]("owner", *controls, "neutral", "", *[""] * 10))
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_malformed_results_are_rejected_before_rendering(self):
        self.import_lens()
        value = inputs(edit=inputs()["edit"] | {"random_control": True},
                       readout={"word": " sat", "tokens": [1, 8], "blocks": [1, 3]},
                       differences=experiment.parse_differences("focus = focus - neutral"))
        with self.service.open_session() as session:
            result, _ = run(session, value)
        missing = (
            ("model",), ("model", "model_id"), ("edited_blocks",), ("target",), ("lens",),
            ("inputs", "edit"), ("inputs", "edit", "random_control"), ("inputs", "injection"),
            ("inputs", "readout"), ("target", "tokens"), ("target", "token_ids"),
            ("lens", "name"), ("lens", "n_prompts"), ("conditions", 0, "lens"),
            ("conditions", 0, "lens", "edit"), ("conditions", 0, "recovery", "random"),
            ("conditions", 0, "coordinates", "random"), ("differences", 0, "edit_change"),
            ("differences", 0, "random_change"), ("differences", 0, "no_edit"),
        )
        invalid = (
            (("created",), 10**400),
            (("inputs", "conditions"), [None]), (("inputs", "edit", "blocks"), [0, 10**10]),
            (("edited_blocks",), []), (("edited_blocks",), [True]),
            (("passage_tokens", 0), None), (("conditions", 0, "coordinates", "reference"), []),
            (("conditions", 0, "coordinates", "edited", 1), None),
            (("conditions", 0, "coordinates", "reference", 0, 0), float("nan")),
            (("conditions", 0, "coordinates", "reference", 0, 0), 1e308),
            (("conditions", 0, "recovery", "random"), []),
            (("conditions", 1, "recovery", "edited"), []),
            (("conditions", 0, "lens", "edit"), float("inf")),
            (("differences", 0, "edit_change"), "bad"), (("target", "tokens", 0), None),
            (("target",), None), (("lens",), None),
        )
        path = self.directory / "malformed.json"
        for keys, replacement in [(keys, ...) for keys in missing] + list(invalid):
            with self.subTest(keys=keys, replacement=replacement):
                bad = copy.deepcopy(result)
                parent = bad
                for key in keys[:-1]:
                    parent = parent[key]
                if replacement is ...:
                    del parent[keys[-1]]
                else:
                    parent[keys[-1]] = replacement
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(gr.Error, "could not be opened"):
                    self.fn["open_result"](str(path), "owner")

    def test_sparse_results_without_a_lens_reopen(self):
        directions = direction_set()
        directions["directions"] = directions["directions"][1:3]
        for random in (False, True):
            with self.subTest(random=random):
                value = inputs(directions=directions, edit=inputs()["edit"] | {"random_control": random})
                with self.service.open_session() as session:
                    result, _ = run(session, value)
                if not random:
                    result["model"]["precision"] = None  # Also accepted by transformers_model().
                path = self.directory / "sparse.json"
                path.write_text(files.dumps(result))
                reopened = self.fn["open_result"](str(path), "owner")
                self.assertEqual(reopened[1], json.loads(path.read_text()))
                self.assertEqual(reopened[6:8], ("", ""))


if __name__ == "__main__":
    unittest.main()
