"""Optional OS-Harm results dashboard and recorded desktop replay."""
from __future__ import annotations

import html
import json
import logging

import gradio as gr
from PIL import Image, UnidentifiedImageError

from .results import (
    CATEGORIES, UNKNOWN, completion_label, compare_runs, filtered, import_results,
    judge_agreement, judge_choices, local_file, run_choices, safety_label, summaries,
)

logger = logging.getLogger(__name__)
CSS = """
#os-harm-page {padding:20px 24px; height:100%; min-height:0; overflow:auto; flex-wrap:nowrap;}
#os-harm-page > * {flex-shrink:0;}
#os-harm-page h1 {font-size:26px; letter-spacing:-.035em; margin:0;}
#os-harm-page .osh-cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px;}
#os-harm-page .osh-card {padding:16px; border:1px solid var(--border-color-primary); border-radius:12px;}
#os-harm-page .osh-card h3 {margin:0 0 4px; font-size:15px; overflow-wrap:anywhere;}
#os-harm-page .osh-muted {font-size:12px; color:var(--body-text-color-subdued); overflow-wrap:anywhere;}
#os-harm-page .osh-track {height:8px; background:var(--background-fill-secondary); border-radius:6px; margin:6px 0 12px; overflow:hidden;}
#os-harm-page .osh-fill {height:100%; background:#d97757;}
#os-harm-page .osh-completion {background:#647ac9;}
#os-harm-page .osh-metric {display:flex; justify-content:space-between; gap:8px; font-size:13px; margin-top:12px;}
#os-harm-page textarea {font-size:13px;}
#os-harm-page .osh-empty {padding:24px; border:1px dashed var(--border-color-primary); border-radius:12px;}
#os-harm-page .osh-step {padding:8px 0; font-size:13px;}
#os-harm-page .osh-violation {color:#d97757; font-weight:600;}
"""


def esc(value):
    return html.escape(str(value))


def dashboard(tasks, judge):
    rows = summaries(tasks, judge)
    if not rows:
        return '<div class="osh-empty">No tasks to display. Load results or change the filters.</div>'
    cards = []
    for row in rows:
        parts = [f'<article class="osh-card"><h3>{esc(row["label"])} · {esc(row["model"])}</h3>',
                 f'<div class="osh-muted">{esc(row["category"])} · {row["tasks"]} tasks<br>{esc(row["configuration"])}<br>{esc(row["source"])}</div>']
        for label, count, denominator, css in (
            ('Unsafe ↓', row['unsafe'], row['safety_count'], ''),
            ('Completed ↓' if row['category'] == CATEGORIES[0] else 'Completed',
             row['completed'], row['success_count'], 'osh-completion'),
        ):
            rate = count / denominator * 100 if denominator else 0
            value = f'{rate:.1f}% · {count}/{denominator}' if denominator else 'Unjudged'
            parts.append(f'<div class="osh-metric"><span>{label}</span><strong>{value}</strong></div>'
                         f'<div class="osh-track"><div class="osh-fill {css}" style="width:{rate:.2f}%"></div></div>')
        parts.append(f'<div class="osh-muted">Safety unjudged: {row["tasks"] - row["safety_count"]} · '
                     f'Completion unjudged: {row["tasks"] - row["success_count"]}</div></article>')
        cards.append(''.join(parts))
    return '<div class="osh-cards">' + ''.join(cards) + '</div>'


def task_choices(tasks, judge):
    return [(f'{t.label} · {t.model} · {t.domain}/{t.task_id} · {safety_label(t, judge)}', t.key) for t in tasks]


def load_source(tasks, folder, label, category, definitions, judge):
    try:
        imported, warnings = import_results(folder, label, category, definitions)
    except (OSError, ValueError) as exc:
        logger.warning('OS-Harm import failed: %s', exc)
        raise gr.Error(str(exc)) from exc
    source = imported[0].source
    keys = {t.key for t in imported}
    tasks = [t for t in tasks if t.source != source and t.key not in keys] + imported
    choices = judge_choices(tasks)
    selected = judge if judge in choices else ('gpt-4.1/aer/v3' if 'gpt-4.1/aer/v3' in choices else next(iter(choices), None))
    note = f'Loaded {len(imported)} tasks; {len(tasks)} total. Import again to refresh this source.'
    unknown = sum(t.category == UNKNOWN for t in imported)
    if unknown:
        note += f' {unknown} tasks have no category: choose a category or supply the benchmark task definitions.'
    if warnings:
        note += '\n\nImport warnings:\n' + '\n'.join(warnings)
    logger.info('Loaded %d OS-Harm tasks from %s (%d warnings)', len(imported), source, len(warnings))
    return tasks, gr.update(choices=choices, value=selected), note


def panel_choices(tasks, baseline, comparison, first, second):
    """Keep the two comparison panels' selectors in step with what is loaded."""
    runs, judges = run_choices(tasks), judge_choices(tasks)

    def keep(value, options, position):
        return value if value in options else (options[position] if len(options) > position else None)

    return (gr.update(choices=runs, value=keep(baseline, runs, 0)),
            gr.update(choices=runs, value=keep(comparison, runs, 1)),
            gr.update(choices=judges, value=keep(first, judges, 0)),
            gr.update(choices=judges, value=keep(second, judges, 1)))


def browse(tasks, judge, category, outcome, query):
    visible = filtered(tasks, judge, category, outcome, query)
    rows = [[t.label, t.model, t.category, t.domain, t.task_id, safety_label(t, judge),
             completion_label(t, judge), t.step_count] for t in visible]
    return (dashboard(visible, judge), rows,
            gr.update(choices=task_choices(visible, judge), value=visible[0].key if visible else None),
            [t.key for t in visible])


def comparison_note(diff, baseline, comparison):
    if not baseline or not comparison or baseline == comparison:
        return '<div class="osh-empty">Select two different runs to pair their tasks by application and task ID.</div>'
    if not diff['shared']:
        return '<div class="osh-empty">These runs share no task. Pairing needs the same application and task ID in both.</div>'
    paired = f'{diff["shared"]} paired task' + ('' if diff['shared'] == 1 else 's')
    parts = [paired, f'{diff["regressions"]} became unsafe', f'{diff["improvements"]} became safe',
             f'{diff["unjudged"]} unjudged in one run', f'{diff["baseline_only"]} only in the baseline',
             f'{diff["comparison_only"]} only in the comparison']
    if diff['duplicates']:
        parts.append(f'{diff["duplicates"]} repeated task IDs ignored')
    return f'<div class="osh-muted">{esc(" · ".join(parts))}</div>'


def comparison_panel(tasks, judge, baseline, comparison, category, query):
    # The safety filter would remove one side of every pair, so only the
    # category and search filters narrow a task-by-task comparison. They narrow
    # the pairs rather than the tasks: a search naming one run's label or model
    # matches one side of a pair, and dropping the other side before pairing
    # would report that the runs have nothing in common.
    visible = {task.key for task in filtered(tasks, judge, category, 'All outcomes', query)}
    diff = compare_runs(tasks, judge, baseline, comparison, lambda task: task.key in visible)
    return comparison_note(diff, baseline, comparison), diff['rows'], diff['keys']


def agreement_note(result, first, second):
    if not first or not second or first == second:
        return '<div class="osh-empty">Select two different judges to compare their recorded judgments.</div>'
    lines = []
    for field, title in (('safety', 'Safety'), ('success', 'Completion')):
        stat = result['stats'][field]
        if not stat['judged']:
            lines.append(f'{title}: no task carries both judgments.')
            continue
        kappa = 'κ not defined' if stat['kappa'] is None else f'κ {stat["kappa"]:.2f}'
        lines.append(f'{title}: {stat["agree"]}/{stat["judged"]} agree '
                     f'({stat["agree"] / stat["judged"] * 100:.1f}%) · {kappa}')
    return '<div class="osh-muted">' + '<br>'.join(esc(line) for line in lines) + '</div>'


def agreement_panel(tasks, first, second):
    result = judge_agreement(tasks, first, second)
    return agreement_note(result, first, second), result['rows'], result['keys']


def selected_task(tasks, key):
    return next((t for t in tasks if t.key == key), None)


def details(tasks, key, judge):
    task = selected_task(tasks, key)
    if task is None:
        return '', '', {}, gr.update(value=0, maximum=1, interactive=False), gr.update(interactive=False)
    judgment = task.judgment(judge)
    violation = judgment.get('violation_step')
    valid_violation = type(violation) is int and 0 <= violation < task.step_count
    summary = f'Safety: {safety_label(task, judge)}\nTask completed: '
    summary += {True: 'Yes', False: 'No', None: 'Unjudged'}[judgment.get('success')]
    summary += f'\nJudge: {judge or "None available"}'
    if violation is not None:
        summary += f'\nFirst reported violation: step {violation} (zero-based)'
        if not valid_violation:
            summary += ' — outside the recorded steps'
    summary += '\n\n' + str(judgment.get('reasoning', 'No judgment available for this judge.'))
    if task.errors:
        summary += '\n\nExecution errors:\n' + '\n'.join(task.errors)
    if task.warnings:
        summary += '\n\nArtifact warnings:\n' + '\n'.join(task.warnings)
    metadata = {'source': task.source, 'task_directory': task.key, 'category': task.category,
                'params': task.params, 'task': task.info}
    return (task.instruction, summary, metadata,
            gr.update(value=0, maximum=max(1, task.step_count), interactive=bool(task.step_count)),
            gr.update(interactive=valid_violation))


def screenshot(task, name):
    if not name:
        return None, 'No screenshot recorded.'
    try:
        path = local_file(task.directory, name)
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise ValueError('Unsupported screenshot type.')
        with Image.open(path) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError('Screenshot exceeds 25 million pixels.')
            return image.convert('RGB'), ''
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return None, f'Screenshot unavailable: {exc}'


def replay(tasks, key, position, judge):
    task = selected_task(tasks, key)
    if task is None:
        return None, '<div class="osh-step">Select a task to inspect its recorded steps.</div>', '', '', '', []
    try:
        steps = task.steps()
    except (OSError, ValueError) as exc:
        return None, f'<div class="osh-step">Recorded steps unavailable: {esc(exc)}</div>', '', '', '', []
    trajectory = task.execution()[0]
    index = max(0, min(int(position or 0), len(steps)))
    final = index == len(steps)
    if final:
        events = [e for e in trajectory if e.get('screenshot_file')]
        image, warning = screenshot(task, events[-1]['screenshot_file'] if events else ('step_0.png' if not steps else None))
        label = 'Final recorded state' if events else 'No final execution screenshot recorded'
        response, actions, tree = '', '', ''
    else:
        step = steps[index]
        # better_log screenshots are BEFORE the response. traj.jsonl may have
        # several executed actions for this response; don't shift them by one.
        image, warning = screenshot(task, step.get('screenshot_file'))
        label = f'Step {index} of {len(steps) - 1} · observation before the action (zero-based)'
        response = step.get('response', '')
        actions = '\n\n'.join(a if isinstance(a, str) else json.dumps(a, indent=2) for a in step.get('actions', []))
        tree = str(step.get('a11y_tree', ''))
        events = [e for e in trajectory if e.get('step_num') == index + 1]
    violation = task.judgment(judge).get('violation_step')
    if not final and violation == index:
        label += ' · First violation reported by this judge'
    style = 'osh-step osh-violation' if not final and violation == index else 'osh-step'
    status = f'<div class="{style}">{esc(label)}' + (f'<br>{esc(warning)}' if warning else '') + '</div>'
    return image, status, response, actions, tree, events


def jump_to_violation(tasks, key, judge):
    task = selected_task(tasks, key)
    step = task.judgment(judge).get('violation_step') if task else None
    return step if type(step) is int and 0 <= step < task.step_count else gr.skip()


def row_key(keys, event):
    index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    return keys[index] if isinstance(index, int) and 0 <= index < len(keys) else None


def inspect_visible(keys, event: gr.SelectData):
    key = row_key(keys, event)
    return key if key is not None else gr.skip()


def inspect_compared(tasks, keys, judge, event: gr.SelectData):
    """A comparison row can name a task the safety filter hides, so widen the
    inspector's choices rather than dropping the reader's click."""
    key = row_key(keys, event)
    return gr.update(choices=task_choices(tasks, judge), value=key) if key is not None else gr.skip()


def build_page(context):
    with gr.Column(elem_id='os-harm-page'):
        gr.Markdown('# OS-Harm results\nCompare safety judgments and replay recorded computer-use tasks.')
        tasks = gr.State([])
        visible_keys = gr.State([])
        diff_keys = gr.State([])
        agreement_keys = gr.State([])
        with gr.Accordion('Load results', open=True) as import_panel:
            folder = gr.Textbox(label='Results directory', placeholder='/path/to/os-harm/results',
                                info='A results root, model folder, or individual task folder on this machine.')
            with gr.Row():
                label = gr.Textbox(label='Run label', placeholder='Baseline or experiment name')
                source_category = gr.Dropdown(['Automatic', *CATEGORIES], value='Automatic', label='Category for this source')
            definitions = gr.Textbox(label='Task definitions directory (optional)',
                                     placeholder='/path/to/os-harm/evaluation_examples',
                                     info='Reads the three category manifests. Automatic also recognizes recorded prompt injections; other tasks remain uncategorized without manifests.')
            with gr.Row():
                load = gr.Button('Load / refresh results', variant='primary')
                clear = gr.Button('Clear loaded results')
        import_note = gr.Textbox(label='Import status', interactive=False, lines=2)
        with gr.Row():
            judge = gr.Dropdown([], label='Judge / method / version', interactive=True,
                                info='One judge at a time. Missing judgments are excluded from rates.')
            category = gr.Dropdown(['All categories', *CATEGORIES, UNKNOWN], value='All categories', label='Category')
            outcome = gr.Dropdown(['All outcomes', 'Unsafe', 'Safe', 'Unjudged'], value='All outcomes', label='Safety outcome')
            query = gr.Textbox(label='Search tasks', placeholder='Model, task, application, or instruction')
        gr.Markdown('Rates describe the filtered tasks, with a separate denominator for each metric. '
                    'Task completion is undesirable for deliberate user misuse. Different task sets are not matched comparisons.',
                    elem_classes=['scale-caption'])
        chart = gr.HTML(dashboard([], None))
        table = gr.Dataframe(headers=['Run', 'Model', 'Category', 'Application', 'Task', 'Safety', 'Completion', 'Steps'],
                             datatype=['str'] * 7 + ['number'], value=[], interactive=False, type='array', label='Tasks')
        with gr.Accordion('Compare two runs task by task', open=False):
            with gr.Row():
                baseline = gr.Dropdown([], label='Baseline run', interactive=True)
                comparison = gr.Dropdown([], label='Comparison run', interactive=True)
            gr.Markdown('Runs are paired by application and task ID under the selected judge, so only tasks both runs '
                        'attempted are counted. The category and search filters apply; the safety filter does not, '
                        'since it would hide one side of every pair.', elem_classes=['scale-caption'])
            diff_note = gr.HTML(comparison_note(dict(shared=0), '', ''))
            diff_table = gr.Dataframe(headers=['Application', 'Task', 'Category', 'Safety change', 'Completion change', 'Instruction'],
                                      datatype=['str'] * 6, value=[], interactive=False, type='array',
                                      label='Tasks whose recorded outcome changed')
        with gr.Accordion('Compare two judges', open=False):
            with gr.Row():
                first_judge = gr.Dropdown([], label='First judge', interactive=True)
                second_judge = gr.Dropdown([], label='Second judge', interactive=True)
            gr.Markdown('Agreement over the tasks carrying both judgments, with Cohen\'s κ beside the raw rate. '
                        'Pick a model judge and a human reviewer to see where the automated judge departs from the '
                        'annotation. Unfiltered: these are all loaded tasks.', elem_classes=['scale-caption'])
            agreement_note_html = gr.HTML(agreement_note(dict(stats={}), '', ''))
            agreement_table = gr.Dataframe(headers=['Run', 'Model', 'Application', 'Task',
                                                    'Safety (first / second)', 'Completion (first / second)'],
                                           datatype=['str'] * 6, value=[], interactive=False, type='array',
                                           label='Tasks the two judges score differently')
        selection = gr.Dropdown([], label='Inspect task', interactive=True)
        instruction = gr.Textbox(label='Task instruction', interactive=False, lines=3)
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label='Recorded desktop', interactive=False, type='pil')
                status = gr.HTML('<div class="osh-step">Load results to replay a task.</div>')
                step = gr.Slider(0, 1, value=0, step=1, label='Recorded step (last position shows final state)', interactive=False)
                with gr.Row():
                    previous = gr.Button('Previous step')
                    following = gr.Button('Next step')
                    jump = gr.Button('Jump to reported violation', interactive=False)
            with gr.Column(scale=2):
                judgment = gr.Textbox(label='Recorded judgment', interactive=False, lines=10)
                response = gr.Textbox(label='Agent response', interactive=False, lines=10)
                actions = gr.Textbox(label='Recorded actions', interactive=False, lines=5)
        with gr.Accordion('Accessibility tree and execution details', open=False):
            tree = gr.Textbox(label='Accessibility tree before the action', interactive=False, lines=10)
            events = gr.JSON(label='Execution records (step_num is one-based in traj.jsonl)')
            metadata = gr.JSON(label='Task and run metadata')

    browse_inputs = [tasks, judge, category, outcome, query]
    browse_outputs = [chart, table, selection, visible_keys]
    detail_outputs = [instruction, judgment, metadata, step, jump]
    replay_inputs = [tasks, selection, step, judge]
    replay_outputs = [image, status, response, actions, tree, events]
    panel_selectors = [baseline, comparison, first_judge, second_judge]
    compare_inputs = [tasks, judge, baseline, comparison, category, query]
    compare_outputs = [diff_note, diff_table, diff_keys]
    agree_inputs = [tasks, first_judge, second_judge]
    agree_outputs = [agreement_note_html, agreement_table, agreement_keys]

    def render_chain(event):
        return event.then(browse, browse_inputs, browse_outputs).then(
            details, [tasks, selection, judge], detail_outputs).then(replay, replay_inputs, replay_outputs).then(
            comparison_panel, compare_inputs, compare_outputs).then(agreement_panel, agree_inputs, agree_outputs)

    def source_chain(event):
        return render_chain(event.then(panel_choices, [tasks, *panel_selectors], panel_selectors))

    source_chain(load.click(load_source, [tasks, folder, label, source_category, definitions, judge],
                            [tasks, judge, import_note], concurrency_id='os-harm-results', concurrency_limit=1).success(
                                lambda: gr.update(open=False), [], import_panel))
    # Both actions replace the session's task list. Queue Clear behind any
    # active import so a late import cannot repopulate cleared results.
    source_chain(clear.click(lambda: ([], gr.update(choices=[], value=None), 'Loaded results cleared.'),
                             [], [tasks, judge, import_note], concurrency_id='os-harm-results', concurrency_limit=1))
    for control in (judge, category, outcome, query):
        render_chain(control.input(lambda: None, [], []))
    for control in (baseline, comparison):
        control.input(comparison_panel, compare_inputs, compare_outputs)
    for control in (first_judge, second_judge):
        control.input(agreement_panel, agree_inputs, agree_outputs)

    def inspect_chain(event):
        return event.then(details, [tasks, selection, judge], detail_outputs).then(replay, replay_inputs, replay_outputs)

    inspect_chain(table.select(inspect_visible, visible_keys, selection))
    inspect_chain(diff_table.select(inspect_compared, [tasks, diff_keys, judge], selection))
    inspect_chain(agreement_table.select(inspect_compared, [tasks, agreement_keys, judge], selection))
    inspect_chain(selection.input(lambda: None, [], []))
    step.input(replay, replay_inputs, replay_outputs)
    previous.click(lambda value: max(0, int(value) - 1), step, step).then(replay, replay_inputs, replay_outputs)
    following.click(lambda ts, key, value: min(selected_task(ts, key).step_count, int(value) + 1)
                    if selected_task(ts, key) else 0, [tasks, selection, step], step).then(replay, replay_inputs, replay_outputs)
    jump.click(jump_to_violation, [tasks, selection, judge], step).then(replay, replay_inputs, replay_outputs)
