"""Rank statistics, real-model collection, persistence and view wiring."""
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
import torch

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, ProjectionCancelled, TokenInspector
from chatlab.extensions.steering_comparison import analysis, experiment, page
from chatlab import jacobian_lens, steering
from torch_support import tiny_manager


class LoadingTests(unittest.TestCase):
    def test_enabling_comparison_does_not_import_other_extension_pages(self):
        command = """
import sys
from chatlab.extensions.registry import load_enabled
loaded, errors = load_enabled(['steering_comparison'])
assert len(loaded) == 1 and not errors, errors
assert not any(name.startswith(('chatlab.extensions.direction_edits', 'chatlab.extensions.probes')) for name in sys.modules)
"""
        subprocess.run([sys.executable, '-c', command], check=True, capture_output=True, text=True)


class StatisticsTests(unittest.TestCase):
    def test_paired_t_factor_and_top_k_shares(self):
        aggregate = analysis.Aggregate()
        aggregate.add([[2, 100]], [[8, 20]], [[4, 30]])
        aggregate.add([[4, 80]], [[8, 40]], [[8, 60]])
        rows = aggregate.rows(0)
        self.assertAlmostEqual(rows[0]['factor'], np.sqrt(8))
        self.assertAlmostEqual(aggregate.traces()['0'][0], np.sqrt(8))
        self.assertAlmostEqual(rows[0]['t'], 3)
        self.assertEqual(rows[0]['top50'], 1)
        self.assertEqual(rows[1]['top50'], 0)
        self.assertEqual(rows[1]['base50'], 1)
        self.assertEqual(rows[1]['top10'], 0)
        rows[0]['token'], rows[1]['token'] = 'Helping', '<tag>'
        self.assertEqual(analysis.select(rows, 'help')[0]['id'], 0)
        self.assertEqual(analysis.select(rows, 'missing'), [])
        self.assertEqual(len(analysis.select(rows, floor=1)), 2)

    def test_undefined_t_and_direct_random_baseline(self):
        aggregate = analysis.Aggregate()
        aggregate.add([[1]], [[4]], [[8]], [[16]])
        self.assertIsNone(aggregate.rows(0)[0]['t'])
        aggregate.add([[1]], [[4]], [[4]], [[16]])
        row = aggregate.rows(0)[0]
        self.assertIsNone(row['t'])
        self.assertAlmostEqual(row['random_t'], 3)
        self.assertEqual(row['random50'], 1)

    def test_random_direct_baseline_retains_candidates_for_zero_floor_browsing(self):
        aggregate = analysis.Aggregate()
        aggregate.add([[80, 100]], [[90, 120]], [[100, 100]], [[10, 80]])
        aggregate.add([[80, 100]], [[90, 120]], [[100, 100]], [[20, 80]])
        rows = aggregate.rows(0)
        self.assertEqual([row['id'] for row in rows], [0])
        row = rows[0]
        self.assertAlmostEqual(row['factor'], 90 / 80)
        self.assertAlmostEqual(row['random_t'], np.log(.1 * .2) / np.log(2))
        self.assertEqual((row['top50'], row['base50'], row['random50']), (0, 0, 0))
        self.assertEqual(list(aggregate.traces()), ['0'])
        self.assertAlmostEqual(aggregate.traces()['0'][0], 90 / 80)
        row['token'] = 'random baseline only'
        self.assertEqual(analysis.select(rows, floor=0), rows)

    def test_random_direct_baseline_requires_matching_positive_finite_rank_arrays(self):
        for baseline in ([[1, 2]], [1], [[0]], [[-1]], [[np.nan]], [[np.inf]]):
            with self.subTest(baseline=baseline):
                aggregate = analysis.Aggregate()
                with self.assertRaisesRegex(ValueError, 'Rank arrays'):
                    aggregate.add([[80]], [[90]], [[100]], baseline)
                self.assertEqual(aggregate.n, 0)
                self.assertIsNone(aggregate.total)

    def test_candidates_and_rejected_inputs(self):
        aggregate = analysis.Aggregate()
        aggregate.add([[80, 5]], [[90, 60]], [[70, 100]])
        self.assertEqual([r['id'] for r in aggregate.rows(0)], [1])
        for rows in ([], [['point', 'hello', 'yes']], [['point', 'hello', 2]]):
            with self.assertRaises(ValueError):
                experiment.examples(rows)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.manager = tiny_manager(layers=3, heads=2)
        self.service = ModelService(lambda: self.manager)
        torch.manual_seed(8)
        self.path = self.directory / 'lens.pt'
        torch.save({'J': {b: torch.randn(16, 16) for b in (0, 1, 2)},
                    'source_layers': [0, 1, 2], 'n_prompts': 10, 'd_model': 16}, self.path)
        self.manager.import_jacobian_lens(str(self.path), self.manager.model_id)
        self.vector = self.directory / 'compassionate.json'
        self.vector.write_text(json.dumps({'format': steering.FORMAT, 'model_id': self.manager.model_id,
                                          'layer': 0, 'strength': 1, 'vector': [0.1] * 16}))

    def collect(self):
        with self.service.open_session() as session:
            steps = experiment.run(session, [['return', 'the cat', 1], ['return', 'the mat', 0],
                                             ['end', 'the cat sat', '']], [str(self.vector)], '0,2')
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    return done.value

    def test_exact_ranks_match_brute_force_including_ties(self):
        ids = self.manager.tokenizer.encode('the cat sat')
        score = jacobian_lens.unembed
        def tied(engine, layout, values):
            logits = score(engine, layout, values)
            if len(values) == 2:
                logits[:, 2] = logits[:, 1]
            return logits
        with self.service.open_session() as session:
            with mock.patch.object(jacobian_lens, 'unembed', side_effect=tied):
                got = session.lens_ranks(ids, [0, 2], [0, 1])
            with mock.patch.object(jacobian_lens, 'unembed', side_effect=tied):
                probabilities = session.lens_log_probs(ids, list(range(len(self.manager.tokenizer))), [0, 2], [0, 1])
        expected = (probabilities[..., None, :] > probabilities[..., :, None]).sum(axis=-1) + 1
        np.testing.assert_array_equal(got, expected)
        self.assertTrue(np.array_equal(got[:, :, 1], got[:, :, 2]))
        self.assertEqual(got.dtype, np.int32)

    def test_scoring_cancellation_and_output_mismatch_remove_hooks(self):
        original = jacobian_lens.unembed
        ids = self.manager.tokenizer.encode('the cat sat')
        with self.service.open_session() as session:
            def cancel_after_score(*args):
                output = original(*args)
                session.cancel()
                return output
            with mock.patch.object(jacobian_lens, 'unembed', side_effect=cancel_after_score):
                with self.assertRaises(ProjectionCancelled):
                    session.lens_ranks(ids, [0, 2], [1])
        with self.service.open_session() as session:
            with mock.patch.object(jacobian_lens, 'unembed', side_effect=lambda *args: original(*args) + 1):
                with self.assertRaisesRegex(ValueError, 'withheld'):
                    session.lens_ranks(ids, [0], [1])
        for block in steering.decoder_layers(self.manager.model):
            self.assertFalse(block._forward_hooks)
            self.assertFalse(block._forward_pre_hooks)

    def test_rank_allocation_bound_before_forward(self):
        self.manager.model.lm_head = torch.nn.Linear(16, 128000, bias=False)
        with self.service.open_session() as session:
            with self.assertRaisesRegex(ValueError, '8 million'):
                session.lens_ranks([1] * 32, [0, 1, 2], list(range(32)))
        for block in steering.decoder_layers(self.manager.model):
            self.assertFalse(block._forward_hooks)

    def test_collection_round_trip_direct_baseline_and_html_escape(self):
        result = self.collect()
        path = self.directory / 'result.json'
        path.write_text(json.dumps(result, allow_nan=False))
        self.assertEqual(experiment.read_result(path), result)
        computed = result['vectors'][0]['points']['return']['Computed']['blocks']['0']
        # At the injection block direct addition is exactly the actual intervention.
        self.assertTrue(all(abs(row['factor'] - 1) < 1e-10 for row in computed))
        total = result['vectors'][0]['points']['return']['Total']['blocks']['2']
        self.assertTrue(any(abs(row['factor'] - 1) > 1e-6 for row in total))
        total[0]['token'] = '<script>alert(1)</script>'
        summary, rendered, examples = page.render(result, 0, 'return', 2, 'Total', 'Rank improvement (t)', 0, '', False)
        self.assertIn('2 paired examples', summary)
        self.assertNotIn('<script>', rendered)
        self.assertEqual(len(examples), 2)
        for block in steering.decoder_layers(self.manager.model):
            self.assertFalse(block._forward_hooks)
            self.assertFalse(block._forward_pre_hooks)
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_cancel_removes_hooks_and_releases_reservation(self):
        with self.service.open_session() as session:
            session.cancel()
            with self.assertRaises(ProjectionCancelled):
                session.lens_ranks([1, 2], [0], [1])
        for block in steering.decoder_layers(self.manager.model):
            self.assertFalse(block._forward_hooks)
        self.assertIsNone(self.manager.claim_generation())
        self.manager.release_generation()

    def test_page_constructs_and_saved_data_is_validated(self):
        context = ExtensionContext(self.service, TokenInspector(), self.directory / 'data', NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            page.build_page(context)
        self.addCleanup(demo.close)
        callbacks = {fn.fn.__name__ for fn in demo.fns.values() if fn.fn}
        self.assertTrue({'collect', 'opened', 'render', 'cancel'} <= callbacks)
        result = self.collect()
        result['vectors'][0]['points']['return']['Total']['blocks']['0'][0]['top50'] = 2
        path = self.directory / 'bad.json'
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'shares'):
            experiment.read_result(path)


if __name__ == '__main__':
    unittest.main()
