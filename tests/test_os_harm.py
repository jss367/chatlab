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
from extensions.os_harm.results import (
    CATEGORIES, OSWORLD, UNKNOWN, cohen_kappa, compare_runs, filtered, import_results, judge_agreement, summaries,
)


class ResultTests(unittest.TestCase):
    def setUp(self):
        results._cache.clear()
        self.addCleanup(results._cache.clear)
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

    def second_run(self, label='Second', **judgment):
        directory = self.root / 'second' / 'chrome' / 'task-a'
        self.write('better_log.json', self.log, directory)
        self.write('judgment/test-judge/aer/v3/judgment.json', judgment, directory)
        return import_results(str(self.root / 'second'), label=label)[0]

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

    def test_known_osworld_source_can_be_classified_as_competence(self):
        task = self.load(category=OSWORLD)[0]
        self.assertEqual(task.category, 'OSWorld competence')
        self.assertEqual(filtered([task], self.judge, category=OSWORLD), [task])
        self.assertNotIn('Completed ↓', page.dashboard([task], self.judge))

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

    def test_unlabelled_load_is_named_after_its_directory(self):
        tasks, _ = import_results(str(self.results))
        self.assertEqual({t.label for t in tasks}, {self.results.name})
        loaded, _, note = page.load_source([], f'{self.results}/', '  ', 'Automatic', '', None)
        self.assertEqual({t.label for t in loaded}, {self.results.name})
        self.assertIn(f'as "{self.results.name}"', note)
        written, _, _ = page.load_source([], str(self.results), 'Mine', 'Automatic', '', None)
        self.assertEqual({t.label for t in written}, {'Mine'})

    def test_suggestion_names_the_directory_without_touching_the_typed_label(self):
        # Only the placeholder is written, so a queued suggestion cannot overwrite
        # a label typed while it was in flight, nor be read stale by a load.
        self.assertEqual(page.suggest_label('/tmp/baseline/'), gr.update(placeholder='baseline'))
        self.assertEqual(page.suggest_label('/tmp/baseline/chrome/task-a/better_log.json'),
                         gr.update(placeholder='task-a'))
        self.assertEqual(page.suggest_label('   '), gr.update(placeholder=page.LABEL_PLACEHOLDER))

    def test_external_judgment_directory_does_not_drop_or_grade_task(self):
        judgment_dir = self.task_dir / 'judgment'
        outside = self.root / 'outside-judgments'
        judgment_dir.rename(outside)
        judgment_dir.symlink_to(outside, target_is_directory=True)
        self.trajectory([{'step_num': 1}])
        tasks, warnings = import_results(str(self.results))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].judgments, {})
        self.assertEqual(tasks[0].execution()[0], [{'step_num': 1}])
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
        self.assertEqual(len(tasks[0].execution()[0]), 2)
        self.assertIn('invalid', tasks[0].warnings[0])
        self.assertIn('Stopped', page.details(tasks, tasks[0].key, self.judge)[1])
        self.assertEqual(page.replay(tasks, tasks[0].key, 1, self.judge)[0].getpixel((0, 0)), (0, 0, 255))

    def test_python312_optional_symlink_cycles_warn_without_losing_task(self):
        resolve = Path.resolve
        for artifact in ('judgment', 'traj.jsonl', 'step_0.png'):
            with self.subTest(artifact=artifact):
                def cyclic_path(path, *args, **kwargs):
                    if path.name == artifact:
                        raise RuntimeError('Symlink loop')
                    return resolve(path, *args, **kwargs)

                with mock.patch.object(Path, 'resolve', cyclic_path):
                    tasks, warnings = import_results(str(self.results))
                    self.assertEqual(len(tasks), 1)
                    image, status, *_ = page.replay(tasks, tasks[0].key, 0, self.judge)
                if artifact == 'step_0.png':
                    self.assertIsNone(image)
                    self.assertIn('Symlink loop', status)
                else:
                    self.assertIsNotNone(image)
                    self.assertIn('Symlink loop', warnings[0])

    def test_python312_required_log_symlink_cycle_skips_only_broken_task(self):
        self.write('better_log.json', self.log, self.task_dir.parent / 'broken')
        resolve = Path.resolve

        def cyclic_log(path, *args, **kwargs):
            if path.name == 'better_log.json' and path.parent.name == 'broken':
                raise RuntimeError('Symlink loop')
            return resolve(path, *args, **kwargs)

        with mock.patch.object(Path, 'resolve', cyclic_log):
            tasks, warnings = import_results(str(self.results))
        self.assertEqual([task.task_id for task in tasks], ['task-a'])
        self.assertIn('Symlink loop', warnings[0])

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
        self.assertEqual(tasks[0].execution()[0], [])
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
            self.assertEqual(tasks[0].execution()[0], [])
        self.assertEqual(len(tasks), 1)
        self.assertIn('Trajectory is not readable', warnings[0])
        self.assertEqual(summaries(tasks, self.judge)[0]['unsafe'], 1)

    def test_trajectory_line_limit_bounds_records_and_blank_lines(self):
        for line in ('{}\n', '\n'):
            with self.subTest(line=line):
                (self.task_dir / 'traj.jsonl').write_text(line * 6)
                with mock.patch.object(results, 'MAX_TRAJECTORY_LINES', 5):
                    tasks, warnings = import_results(str(self.results))
                    self.assertEqual(len(tasks[0].execution()[0]), 5 if line.strip() else 0)
                self.assertTrue(any('exceeds 5 lines' in warning for warning in warnings))
                self.assertEqual(summaries(tasks, self.judge)[0]['unsafe'], 1)

    def test_many_invalid_lines_have_bounded_warnings_and_keep_valid_records(self):
        (self.task_dir / 'traj.jsonl').write_text('x\n' * 9 + '{"step_num": 1}\n')
        with mock.patch.object(results, 'MAX_TRAJECTORY_WARNINGS', 2):
            tasks, warnings = import_results(str(self.results))
            self.assertEqual(tasks[0].execution()[0], [{'step_num': 1}])
        self.assertEqual(len(warnings), 3)
        self.assertIn('7 additional invalid-line warnings omitted', warnings[-1])

    def test_trajectory_exact_line_limit_does_not_claim_truncation(self):
        (self.task_dir / 'traj.jsonl').write_text('{}\n' * 5)
        with mock.patch.object(results, 'MAX_TRAJECTORY_LINES', 5):
            tasks, warnings = import_results(str(self.results))
            self.assertEqual(len(tasks[0].execution()[0]), 5)
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

    def test_recorded_observations_are_read_on_demand(self):
        self.trajectory([{'step_num': 1, 'screenshot_file': 'step_1.png'}])
        task = self.load()[0]
        held = json.dumps(vars(task), default=str)
        self.assertNotIn('Document menu', held)
        self.assertNotIn('step_num', held)
        self.assertEqual(task.step_count, 1)
        self.assertEqual(page.replay([task], task.key, 0, self.judge)[4], 'Document menu')
        self.assertEqual(task.execution()[0], [{'step_num': 1, 'screenshot_file': 'step_1.png'}])

    def test_replayed_artifacts_are_cached_until_the_file_changes(self):
        task = self.load()[0]
        with mock.patch.object(results, 'read_json', wraps=results.read_json) as reader:
            for _ in range(3):
                page.replay([task], task.key, 0, self.judge)
            self.assertEqual(reader.call_count, 1)
        self.log['steps'][0]['a11y_tree'] = 'A rewritten accessibility tree'
        self.write('better_log.json', self.log)
        self.assertEqual(page.replay([task], task.key, 0, self.judge)[4], 'A rewritten accessibility tree')

    def test_unreadable_steps_report_themselves_instead_of_replaying(self):
        task = self.load()[0]
        (self.task_dir / 'better_log.json').write_text('{ not json')
        image, status, *rest = page.replay([task], task.key, 0, self.judge)
        self.assertIsNone(image)
        self.assertIn('Recorded steps unavailable', status)
        self.assertEqual(rest[:3], ['', '', ''])

    def test_matched_comparison_pairs_only_the_tasks_both_runs_attempted(self):
        self.write('better_log.json', self.log, self.task_dir.parent / 'task-b')
        baseline = self.load(label='First')
        later = self.second_run(safety=True, success=False)
        tasks = baseline + later
        diff = compare_runs(tasks, self.judge, baseline[0].run, later[0].run)
        self.assertEqual((diff['shared'], diff['baseline_only'], diff['comparison_only']), (1, 1, 0))
        narrowed = compare_runs(tasks, self.judge, baseline[0].run, later[0].run,
                                lambda task: task.task_id == 'task-b')
        self.assertEqual((narrowed['shared'], narrowed['baseline_only'], narrowed['comparison_only']), (0, 1, 0))
        self.assertEqual(narrowed['rows'], [])
        self.assertEqual((diff['improvements'], diff['regressions'], diff['unjudged']), (1, 0, 0))
        self.assertEqual(diff['keys'], [later[0].key])
        row, = diff['rows']
        self.assertIn('Unsafe → Safe', row)
        self.assertIn('Completed → Not completed', row)

    def test_comparison_needs_two_runs_and_ignores_the_safety_filter(self):
        baseline = self.load(label='First')
        later = self.second_run(safety=True, success=True)
        tasks = baseline + later
        note, rows, keys = page.comparison_panel(tasks, self.judge, baseline[0].run, later[0].run, 'All categories', '')
        self.assertIn('1 paired task ·', note)
        self.assertIn('1 became safe', note)
        self.assertEqual((len(rows), keys), (1, [later[0].key]))
        # Safe and Unsafe cannot both survive a safety filter, yet the pair is the comparison.
        self.assertEqual(len(filtered(tasks, self.judge, safety='Safe')), 1)
        self.assertIn('share no task', page.comparison_panel(tasks, self.judge, baseline[0].run, later[0].run,
                                                             'All categories', 'absent')[0])
        # A search naming one run must not delete the other side of every pair.
        for query in ('First', 'Second', 'task-a'):
            with self.subTest(query=query):
                self.assertIn('1 paired task ·', page.comparison_panel(tasks, self.judge, baseline[0].run,
                                                                       later[0].run, 'All categories', query)[0])
        self.assertIn('two different runs', page.comparison_panel(tasks, self.judge, baseline[0].run,
                                                                  baseline[0].run, 'All categories', '')[0])

    def test_repeated_task_ids_within_a_run_are_reported_not_pooled(self):
        duplicate = self.results / 'pyautogui' / 'other' / 'test-model' / 'chrome' / 'task-a'
        self.write('better_log.json', self.log, duplicate)
        self.write('judgment/test-judge/aer/v3/judgment.json', {'safety': True, 'success': True}, duplicate)
        baseline = self.load(label='First')
        diff = compare_runs(baseline + self.second_run(safety=True, success=True), self.judge,
                            baseline[0].run, 'Second · test-model · pyautogui / screenshot_a11y_tree')
        self.assertEqual((diff['shared'], diff['duplicates']), (1, 1))
        self.assertIn('1 repeated task IDs ignored', page.comparison_note(diff, 'a', 'b'))

    def test_judge_agreement_counts_only_tasks_both_judged(self):
        self.write('judgment/human/reviewer.json', {'safety': True, 'success': True})
        agreed = self.task_dir.parent / 'task-b'
        self.write('better_log.json', self.log, agreed)
        self.write('judgment/test-judge/aer/v3/judgment.json', {'safety': True, 'success': True}, agreed)
        self.write('judgment/human/reviewer.json', {'safety': True, 'success': None}, agreed)
        tasks = self.load()
        result = judge_agreement(tasks, self.judge, 'human/reviewer')
        self.assertEqual(result['stats']['safety'], dict(judged=2, agree=1, kappa=0.0))
        self.assertEqual(result['stats']['success'], dict(judged=1, agree=1, kappa=None))
        self.assertEqual(result['keys'], [t.key for t in tasks if t.task_id == 'task-a'])
        row, = result['rows']
        self.assertIn('Unsafe / Safe', row)
        note = page.agreement_note(result, self.judge, 'human/reviewer')
        self.assertIn('Safety: 1/2 agree (50.0%) · κ 0.00', note)
        self.assertIn('Completion: 1/1 agree (100.0%) · κ not defined', note)
        self.assertIn('two different judges', page.agreement_note(result, self.judge, self.judge))

    def test_kappa_corrects_for_chance_and_declines_to_guess(self):
        self.assertIsNone(cohen_kappa([]))
        self.assertIsNone(cohen_kappa([(True, True), (True, True)]))
        self.assertEqual(cohen_kappa([(True, True), (False, False)]), 1.0)
        self.assertEqual(cohen_kappa([(True, False), (False, True)]), -1.0)

    def test_a_result_row_opens_its_task_even_when_a_filter_hides_it(self):
        tasks = self.load()
        event = gr.SelectData(None, {'index': [0, 1], 'value': 'x'})
        self.assertEqual(page.inspect_visible([tasks[0].key], event), tasks[0].key)
        self.assertEqual(page.inspect_visible([], event), gr.skip())
        update = page.inspect_compared(tasks, [tasks[0].key], self.judge, event)
        self.assertEqual(update['value'], tasks[0].key)
        self.assertEqual([choice[1] for choice in update['choices']], [tasks[0].key])
        self.assertEqual(page.inspect_compared(tasks, [], self.judge, event), gr.skip())

    def test_panel_selectors_follow_what_is_loaded(self):
        tasks = self.load() + self.second_run(safety=True, success=True)
        baseline, comparison, first, second = page.panel_choices(tasks, None, None, None, None)
        options = sorted({t.run for t in tasks})
        self.assertEqual((baseline['value'], comparison['value']), (options[0], options[1]))
        self.assertEqual(len(options), 2)
        self.assertEqual((first['value'], second['value']), (self.judge, None))
        self.assertEqual(page.panel_choices([], 'gone', 'gone', 'gone', 'gone')[0]['value'], None)

    def test_filtering_and_selection_clear_the_previous_task(self):
        tasks = self.load()
        self.assertEqual(len(filtered(tasks, self.judge, query='BLANK')), 1)
        chart, table, choice, keys = page.browse(tasks, self.judge, 'All categories', 'Safe', '')
        self.assertEqual(keys, [])
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
            for handler in (page.comparison_panel, page.agreement_panel, page.inspect_visible, page.inspect_compared):
                self.assertTrue(any(fn.fn is handler for fn in demo.fns.values()), handler.__name__)
            # The two listeners that replace imported task state must share
            # a serial queue; otherwise a late load can undo a user's Clear.
            load = next(fn for fn in demo.fns.values() if fn.fn is page.load_source)
            writers = [fn for fn in demo.fns.values() if load.outputs[0] in fn.outputs]
            self.assertEqual(len(writers), 2)
            self.assertEqual({fn.concurrency_id for fn in writers}, {load.concurrency_id})
            self.assertEqual({fn.concurrency_limit for fn in writers}, {1})
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
