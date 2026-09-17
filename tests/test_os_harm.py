"""OS-Harm schema compatibility, honest denominators and recorded-step replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import gradio as gr
from PIL import Image

from extensions.os_harm import page
from extensions.os_harm import results
from extensions.os_harm.results import CATEGORIES, UNKNOWN, filtered, import_results, summaries


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'
        self.task_dir = self.results / 'pyautogui' / 'screenshot_a11y_tree' / 'test-model' / 'chrome' / 'task-a'
        self.task_dir.mkdir(parents=True)
        self.log = {'params': {'model': 'test-model', 'action_space': 'pyautogui', 'observation_type': 'screenshot_a11y_tree'},
                    'task': {'instruction': 'Synthetic task: open a blank document.', 'jailbreak': False, 'injection': None},
                    'steps': [{'screenshot_file': 'step_0.png', 'response': 'Open the document.',
                               'actions': ['FIRST', 'SECOND'], 'a11y_tree': 'Document menu'}]}
        self.write('better_log.json', self.log)
        Image.new('RGB', (8, 8), 'red').save(self.task_dir / 'step_0.png')
        Image.new('RGB', (8, 8), 'blue').save(self.task_dir / 'step_1.png')
        self.write('judgment/test-judge/aer/v3/judgment.json',
                   {'safety': False, 'success': True, 'reasoning': 'Synthetic violation.', 'violation_step': 0})
        self.judge = 'test-judge/aer/v3'

    def write(self, name, value, directory=None):
        path = (directory or self.task_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def load(self, **kwargs):
        return import_results(str(self.results), **kwargs)[0]

    def trajectory(self, entries):
        (self.task_dir / 'traj.jsonl').write_text('\n'.join(json.dumps(e) for e in entries))

    def test_upstream_log_and_both_judge_layouts(self):
        self.write('judgment/human/reviewer.json', {'safety': True, 'success': False})
        task = self.load()[0]
        self.assertEqual(task.model, 'test-model')
        self.assertEqual(task.task_id, 'task-a')
        self.assertEqual(task.category, UNKNOWN)
        self.assertEqual(set(task.judgments), {self.judge, 'human/reviewer'})
        self.assertFalse(task.judgment(self.judge)['safety'])
        self.assertTrue(task.judgment('human/reviewer')['safety'])
        self.assertEqual(task.judgment('absent'), {})

    def test_manifest_categories_and_recorded_injection(self):
        definitions = self.root / 'evaluation_examples'
        for filename in ('test_misuse.json', 'test_injection.json', 'test_misbehavior.json'):
            self.write(filename, {'chrome': ['task-a']} if filename == 'test_misbehavior.json' else {}, definitions)
        self.assertEqual(self.load(definitions=str(self.root))[0].category, CATEGORIES[2])
        self.log['task']['injection'] = {'type': 'website', 'goal_id': 'synthetic'}
        self.write('better_log.json', self.log)
        self.assertEqual(self.load()[0].category, CATEGORIES[1])
        self.assertEqual(self.load(category=CATEGORIES[0])[0].category, CATEGORIES[0])

    def test_missing_outcomes_are_not_safe_or_failed(self):
        self.write('better_log.json', self.log, self.task_dir.parent / 'unjudged')
        partial = self.task_dir.parent / 'partial'
        self.write('better_log.json', self.log, partial)
        self.write('judgment/test-judge/aer/v3/judgment.json', {'safety': True, 'success': None}, partial)
        tasks = self.load()
        row, = summaries(tasks, self.judge)
        self.assertEqual((row['tasks'], row['safety_count'], row['unsafe'], row['success_count'], row['completed']), (3, 2, 1, 1, 1))
        self.assertEqual(len(filtered(tasks, self.judge, safety='Unjudged')), 1)
        self.assertIn('50.0% · 1/2', page.dashboard(tasks, self.judge))
        self.assertIn('100.0% · 1/1', page.dashboard(tasks, self.judge))
        self.assertIn('Unjudged', page.dashboard(tasks, 'absent'))

    def test_bad_judgment_and_bad_task_do_not_hide_valid_results(self):
        self.write('judgment/wrong/aer/v3/judgment.json', {'safety': 'false', 'success': True})
        self.write('better_log.json', {'steps': 'bad'}, self.task_dir.parent / 'broken')
        tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(set(tasks[0].judgments), {self.judge})
        self.assertEqual(len(warnings), 2)

    def test_refresh_replaces_source_and_preserves_other_sources(self):
        tasks = self.load(label='First')
        other = self.root / 'second' / 'chrome' / 'task-a'
        self.write('better_log.json', self.log, other)
        tasks += import_results(str(self.root / 'second'), label='Second')[0]
        loaded, judge, note = page.load_source(tasks, str(self.results), 'Updated', 'Automatic', '', self.judge)
        self.assertEqual(len(loaded), 2)
        self.assertEqual({t.label for t in loaded}, {'Second', 'Updated'})
        self.assertEqual(judge['value'], self.judge)
        self.assertEqual(len(summaries(loaded, self.judge)), 2)
        self.assertIn('2 total', note)

    def test_external_judgment_directory_does_not_drop_or_grade_task(self):
        judgment_dir = self.task_dir / 'judgment'
        outside = self.root / 'outside-judgments'
        judgment_dir.rename(outside)
        judgment_dir.symlink_to(outside, target_is_directory=True)
        self.trajectory([{'step_num': 1}])
        tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].judgments, {})
        self.assertEqual(tasks[0].trajectory, [{'step_num': 1}])
        self.assertIn('judgment directory could not be read', warnings[0])
        self.assertIn('leaves the task directory', warnings[0])
        self.assertEqual(summaries(tasks, self.judge)[0]['safety_count'], 0)
        self.assertEqual(page.replay(tasks, tasks[0].key, 0, self.judge)[0].getpixel((0, 0)), (255, 0, 0))

    def test_judgment_scan_failure_keeps_task_as_unjudged(self):
        rglob = Path.rglob

        def unreadable_judgments(path, *args, **kwargs):
            if path.name == 'judgment':
                raise PermissionError('Judgment directory is not readable')
            return rglob(path, *args, **kwargs)

        with mock.patch.object(Path, 'rglob', unreadable_judgments):
            tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].judgments, {})
        self.assertIn('Judgment directory is not readable', warnings[0])
        self.assertEqual(summaries(tasks, self.judge)[0]['safety_count'], 0)

    def test_before_action_and_final_state_and_zero_based_violation(self):
        self.trajectory([{'step_num': 1, 'action': 'FIRST', 'screenshot_file': 'step_1.png'},
                         {'step_num': 1, 'action': 'SECOND', 'screenshot_file': 'step_1.png'}])
        tasks = self.load()
        image, status, response, actions, tree, events = page.replay(tasks, tasks[0].key, 0, self.judge)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertIn('before the action', status)
        self.assertIn('First violation', status)
        self.assertEqual(len(events), 2)
        self.assertEqual(actions, 'FIRST\n\nSECOND')
        self.assertEqual(tree, 'Document menu')
        self.assertEqual(response, 'Open the document.')
        self.assertEqual(page.jump_to_violation(tasks, tasks[0].key, self.judge), 0)
        self.assertTrue(page.details(tasks, tasks[0].key, self.judge)[-1]['interactive'])
        final = page.replay(tasks, tasks[0].key, 1, self.judge)
        self.assertEqual(final[0].getpixel((0, 0)), (0, 0, 255))
        self.assertIn('Final recorded state', final[1])
        self.assertEqual(final[2:5], ('', '', ''))

    def test_missing_corrupt_and_escaping_screenshots_do_not_read_arbitrary_files(self):
        outside = self.root / 'outside.png'
        Image.new('RGB', (8, 8), 'green').save(outside)
        (self.task_dir / 'link.png').symlink_to(outside)
        (self.task_dir / 'corrupt.png').write_text('not an image')
        task = self.load()[0]
        for name in ('missing.png', 'link.png', '../../outside.png', str(outside), 'https://example.org/image.png', 'corrupt.png'):
            with self.subTest(name=name):
                image, warning = page.screenshot(task, name)
                self.assertIsNone(image)
                self.assertIn('unavailable', warning)

    def test_partial_trajectory_preserves_prior_steps_and_execution_errors(self):
        self.trajectory([{'step_num': 1, 'screenshot_file': 'step_1.png'}, {'Error': 'Stopped'}])
        with (self.task_dir / 'traj.jsonl').open('a') as file:
            file.write('\n{"incomplete":')
        tasks = self.load()
        self.assertEqual(len(tasks[0].trajectory), 2)
        self.assertIn('invalid', tasks[0].warnings[0])
        self.assertIn('Stopped', page.details(tasks, tasks[0].key, self.judge)[1])
        self.assertEqual(page.replay(tasks, tasks[0].key, 1, self.judge)[0].getpixel((0, 0)), (0, 0, 255))

    def test_empty_log_and_out_of_range_judge_do_not_invent_replay(self):
        self.log['steps'] = []
        self.write('better_log.json', self.log)
        tasks = self.load()
        self.assertFalse(page.details(tasks, tasks[0].key, self.judge)[-1]['interactive'])
        self.assertEqual(page.jump_to_violation(tasks, tasks[0].key, self.judge), gr.skip())
        self.assertIn('outside', page.details(tasks, tasks[0].key, self.judge)[1])
        self.assertIsNotNone(page.replay(tasks, tasks[0].key, 0, self.judge)[0])

    def test_invalid_utf8_trajectory_keeps_task_judgments_and_observations(self):
        (self.task_dir / 'traj.jsonl').write_bytes(b'{"action": "\xff"}\n')
        tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].trajectory, [])
        self.assertIn('traj.jsonl could not be read', warnings[0])
        self.assertIn('utf-8', warnings[0])
        self.assertEqual(summaries(tasks, self.judge)[0]['unsafe'], 1)
        self.assertEqual(page.replay(tasks, tasks[0].key, 0, self.judge)[0].getpixel((0, 0)), (255, 0, 0))
        self.assertIn('traj.jsonl could not be read', page.details(tasks, tasks[0].key, self.judge)[1])

    def test_trajectory_read_failure_keeps_valid_task(self):
        self.trajectory([{'step_num': 1}])
        open_file = Path.open

        def unreadable_trajectory(path, *args, **kwargs):
            if path.name == 'traj.jsonl':
                raise PermissionError('Trajectory is not readable')
            return open_file(path, *args, **kwargs)

        with mock.patch.object(Path, 'open', unreadable_trajectory):
            tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].trajectory, [])
        self.assertIn('Trajectory is not readable', warnings[0])
        self.assertEqual(summaries(tasks, self.judge)[0]['unsafe'], 1)

    def test_trajectory_line_limit_bounds_records_and_blank_lines(self):
        for line in ('{}\n', '\n'):
            with self.subTest(line=line):
                (self.task_dir / 'traj.jsonl').write_text(line * 6)
                with mock.patch.object(results, 'MAX_TRAJECTORY_LINES', 5):
                    tasks, warnings = import_results(str(self.results))
                self.assertEqual(len(tasks[0].trajectory), 5 if line.strip() else 0)
                self.assertTrue(any('exceeds 5 lines' in warning for warning in warnings))
                self.assertEqual(summaries(tasks, self.judge)[0]['unsafe'], 1)

    def test_many_invalid_lines_have_bounded_warnings_and_keep_valid_records(self):
        (self.task_dir / 'traj.jsonl').write_text('x\n' * 9 + '{"step_num": 1}\n')
        with mock.patch.object(results, 'MAX_TRAJECTORY_WARNINGS', 2):
            tasks, warnings = import_results(str(self.results))
        self.assertEqual(tasks[0].trajectory, [{'step_num': 1}])
        self.assertEqual(len(warnings), 3)
        self.assertIn('7 additional invalid-line warnings omitted', warnings[-1])

    def test_trajectory_exact_line_limit_does_not_claim_truncation(self):
        (self.task_dir / 'traj.jsonl').write_text('{}\n' * 5)
        with mock.patch.object(results, 'MAX_TRAJECTORY_LINES', 5):
            tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks[0].trajectory), 5)
        self.assertEqual(warnings, [])

    def test_html_is_escaped_and_judgments_do_not_execute_actions(self):
        self.log['params']['model'] = '<img src=x onerror=alert(1)>'
        self.log['steps'][0]['actions'] = ['raise RuntimeError("must never run")']
        self.write('better_log.json', self.log)
        tasks = self.load(label='<script>alert(1)</script>')
        chart = page.dashboard(tasks, self.judge)
        self.assertNotIn('<img', chart)
        self.assertNotIn('<script>', chart)
        self.assertIn('&lt;img', chart)
        self.assertIn('must never run', page.replay(tasks, tasks[0].key, 0, self.judge)[3])

    def test_failed_import_is_clear_and_no_files_are_written(self):
        before = set(self.root.rglob('*'))
        for folder in ('', str(self.root / 'absent'), str(self.root / 'empty')):
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                import_results(folder)
        self.load()
        self.assertEqual(set(self.root.rglob('*')), before)
        with self.assertRaises(gr.Error):
            page.load_source([], '', '', 'Automatic', '', None)

    def test_filtering_and_selection_clear_the_previous_task(self):
        tasks = self.load()
        self.assertEqual(len(filtered(tasks, self.judge, query='BLANK')), 1)
        chart, table, choice = page.browse(tasks, self.judge, 'All categories', 'Safe', '')
        self.assertEqual(table, [])
        self.assertIsNone(choice['value'])
        self.assertIn('No tasks', chart)
        self.assertEqual(page.details(tasks, None, self.judge)[:3], ('', '', {}))
        self.assertIsNone(page.replay(tasks, None, 0, self.judge)[0])


class ExtensionIntegrationTests(unittest.TestCase):
    def test_extension_builds_with_host_contract_and_no_loaded_model(self):
        import app
        from extensions.registry import load_enabled
        enabled, errors = load_enabled(['os_harm'])
        self.assertFalse(errors)
        with mock.patch('ui.layout.load_enabled', return_value=(enabled, [])):
            demo = app.build_app()
        try:
            nav = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'nav')
            self.assertIn(('OS-Harm', 'OS-Harm'), nav.choices)
            self.assertTrue(any(getattr(b, 'elem_id', None) == 'os-harm-page' for b in demo.blocks.values()))
            self.assertTrue(any(fn.fn is page.load_source for fn in demo.fns.values()))
        finally:
            demo.close()

    def test_disabled_startup_does_not_import_viewer(self):
        import settings
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c',
                "import app,sys; demo=app.build_app(); assert not any(n.startswith('extensions.os_harm') for n in tuple(sys.modules)); demo.close()"],
                env=os.environ | {settings.SETTINGS_PATH_ENV: str(Path(directory) / 'settings.json'),
                                  'GRADIO_ANALYTICS_ENABLED': 'False'},
                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
