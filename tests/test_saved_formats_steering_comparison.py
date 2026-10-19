"""The first released steering-comparison download stays readable offline.

Fixture written through 072f85c's publication path; never regenerate it.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import gradio as gr

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.steering_comparison import experiment, page
from ui_support import handlers_by_name

FIXTURE = Path(__file__).parent / 'fixtures/saved/steering_comparison/paired-ranks.json'


class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.path = self.directory / 'comparison.json'
        shutil.copyfile(FIXTURE, self.path)

    def assert_facts(self, result):
        self.assertEqual(result['format'], 'chatlab-steering-comparison-1')
        self.assertEqual(result['model_id'], 'test/tiny-decoder')
        self.assertEqual(result['blocks'], [0, 2])
        self.assertEqual(result['seed'], 7)
        self.assertEqual([v['name'] for v in result['vectors']], ['compassionate', 'opposite sign'])
        self.assertEqual([e['recovered'] for e in result['examples']], [1, 0, None])
        group = result['vectors'][0]['points']['Return to task']['Total']
        self.assertEqual(group['n'], 2)
        row = next(r for r in group['blocks']['2'] if r['token'] == 'cat')
        self.assertEqual(row['id'], 279)
        self.assertAlmostEqual(row['factor'], 5.29150262212918)
        self.assertAlmostEqual(row['t'], 5.954450503386643)
        self.assertEqual((row['top50'], row['base50'], row['top10'], row['base10']), (1, 1, 1, 1))
        self.assertEqual(len(group['traces']['279']), 2)
        summary, rendered, examples = page.render(result, 0, 'Return to task', 2, 'Total', 'Rank improvement (t)', 50, 'cat', False)
        self.assertIn('2 paired examples', summary)
        self.assertIn('50% (2 labeled examples)', summary)
        self.assertIn('×5.29', rendered)
        self.assertIn('Rank change across blocks', rendered)
        self.assertEqual(len(examples), 2)
        _, computed, _ = page.render(result, 0, 'Return to task', 0, 'Computed', 'Rank improvement (t)', 50, '', False)
        self.assertIn('No matching tokens moved', computed)

    def test_resaved_by_todays_page_it_reads_the_same(self):
        result = experiment.read_result(self.path)
        self.assert_facts(result)
        context = ExtensionContext(ModelService(lambda: None), TokenInspector(), self.directory,
                                   NavigationService(lambda *args: None))
        with gr.Blocks() as demo:
            page.build_page(context)
        self.addCleanup(demo.close)
        opened = handlers_by_name(demo)['opened'](str(self.path), 'fixture')
        self.assertEqual(opened[0], result)
        self.assertEqual(opened[1]['value'], 0)
        self.assertEqual(opened[2]['choices'], ['Return to task', 'End of turn'])
        self.assertEqual(opened[-1], 'Comparison ready.')
        self.assert_facts(experiment.read_result(opened[-2]))

    def test_saved_examples_use_the_normalized_values_in_the_view(self):
        original = json.loads(FIXTURE.read_text())
        for field in ('recovered', 'point'):
            with self.subTest(field=field):
                result = json.loads(json.dumps(original))
                for example in result['examples']:
                    if field == 'recovered' and example[field] is not None:
                        example[field] = str(example[field])
                    elif field == 'point':
                        example[field] = f"  {example[field]}  "
                self.path.write_text(json.dumps(result))
                loaded = experiment.read_result(self.path)
                self.assertEqual(loaded['examples'], original['examples'])
                self.assert_facts(loaded)
                self.path.write_text(json.dumps(loaded))
                self.assert_facts(experiment.read_result(self.path))


if __name__ == '__main__':
    unittest.main()
