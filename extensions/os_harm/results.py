"""Read OS-Harm artifacts without importing or executing the benchmark harness."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

CATEGORIES = ('Deliberate user misuse', 'Prompt injection attacks', 'Model misbehavior')
MANIFESTS = dict(zip(('test_misuse.json', 'test_injection.json', 'test_misbehavior.json'), CATEGORIES))
UNKNOWN = 'Unknown category'
MAX_JSON_BYTES = 32 * 1024 * 1024


def local_file(directory: Path, name: str) -> Path:
    """Artifacts may only refer to files inside their own task directory."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Expected a relative artifact filename.')
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Artifact path leaves the task directory.')
    return path


def read_json(path: Path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('JSON artifact exceeds the 32 MB limit.')
    return json.loads(path.read_text(encoding='utf-8'))


@dataclass
class Task:
    directory: Path
    source: str
    label: str
    domain: str
    task_id: str
    category: str
    log: dict
    judgments: dict
    trajectory: list
    warnings: list

    @property
    def key(self):
        return str(self.directory)

    @property
    def model(self):
        return str(self.log['params'].get('model', 'Unknown model'))

    @property
    def configuration(self):
        params = self.log['params']
        return ' / '.join(str(params.get(k, 'Unknown')) for k in ('action_space', 'observation_type'))

    def judgment(self, judge):
        return self.judgments.get(judge, {})


def category_index(folder: str) -> dict:
    if not folder.strip():
        return {}
    root = Path(folder).expanduser().resolve()
    if (root / 'evaluation_examples').is_dir():
        root /= 'evaluation_examples'
    if not root.is_dir():
        raise ValueError('The task definitions directory does not exist.')
    index = {}
    for filename, category in MANIFESTS.items():
        data = read_json(root / filename)
        if not isinstance(data, dict):
            raise ValueError(f'{filename}: expected application names mapped to task ID lists.')
        for domain, ids in data.items():
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise ValueError(f'{filename}: expected a list of task IDs for {domain}.')
            for task_id in ids:
                key = (domain, task_id)
                if key in index and index[key] != category:
                    raise ValueError(f'A task appears in multiple category manifests: {domain}/{task_id}')
                index[key] = category
    return index


def validate_log(data):
    if not isinstance(data, dict) or not all(isinstance(data.get(k), dict) for k in ('params', 'task')):
        raise ValueError('Expected an OS-Harm better_log.json with params, task and steps.')
    steps = data.get('steps')
    if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
        raise ValueError('The steps field must be a list of objects.')
    if not isinstance(data['task'].get('instruction'), str):
        raise ValueError('The task instruction must be text.')
    for step in steps:
        if not isinstance(step.get('response', ''), str):
            raise ValueError('Each step response must be text.')
        if not isinstance(step.get('actions', []), list):
            raise ValueError('Each step actions field must be a list.')
    return data


def validate_judgment(data):
    if not isinstance(data, dict):
        raise ValueError('Expected a judgment object.')
    for key in ('safety', 'success'):
        if data.get(key) is not None and type(data[key]) is not bool:
            raise ValueError(f'{key} must be a boolean or null.')
    step = data.get('violation_step')
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError('violation_step must be a nonnegative integer or null.')
    return data


def load_task(directory, root, label, category, index):
    warnings, judgments, trajectory = [], {}, []
    log = validate_log(read_json(local_file(directory, 'better_log.json')))
    domain, task_id = directory.parent.name, directory.name
    if category == 'Automatic':
        if log['task'].get('injection'):
            category = CATEGORIES[1]
        else:
            category = index.get((domain, task_id.split('__inject__')[0]), UNKNOWN)
    judge_dir = local_file(directory, 'judgment')
    if judge_dir.is_dir():
        for path in sorted(judge_dir.rglob('*.json')):
            relative = path.relative_to(judge_dir)
            if path.name != 'judgment.json' and relative.parts[0] != 'human':
                continue
            try:
                path = local_file(directory, str(path.relative_to(directory)))
                key = relative.with_suffix('').as_posix()
                if key.endswith('/judgment'):
                    key = key[:-len('/judgment')]
                judgments[key] = validate_judgment(read_json(path))
            except (OSError, ValueError) as exc:
                warnings.append(f'{relative}: {exc}')
    try:
        traj_path = local_file(directory, 'traj.jsonl')
        if traj_path.exists():
            if traj_path.stat().st_size > MAX_JSON_BYTES:
                warnings.append('traj.jsonl exceeds 32 MB; execution details were skipped.')
            else:
                for line_no, line in enumerate(traj_path.read_text(encoding='utf-8').splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if not isinstance(entry, dict):
                            raise ValueError('Expected an object.')
                        trajectory.append(entry)
                    except ValueError:
                        warnings.append(f'traj.jsonl line {line_no} is invalid; other steps remain available.')
    except (OSError, ValueError) as exc:
        # Execution records are optional; a decoding or filesystem failure
        # must not remove a valid task and its judgments from the comparison.
        warnings.append(f'traj.jsonl could not be read; execution details were skipped: {exc}')
    return Task(directory, str(root), label, domain, task_id, category, log, judgments, trajectory, warnings)


def import_results(folder: str, label='', category='Automatic', definitions=''):
    if not folder.strip():
        raise ValueError('Enter an OS-Harm results directory.')
    root = Path(folder).expanduser().resolve()
    if root.is_file() and root.name == 'better_log.json':
        root = root.parent
    if not root.is_dir():
        raise ValueError('The results directory does not exist.')
    if category not in ('Automatic', *CATEGORIES):
        raise ValueError('Select a benchmark category or Automatic.')
    index = category_index(definitions)
    paths = sorted(root.rglob('better_log.json'))
    if not paths:
        raise ValueError('No better_log.json files found. Select a results root or a task directory.')
    tasks, warnings = [], []
    for path in paths:
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError('Result symlink leaves the selected directory.')
            task = load_task(path.parent, root, label.strip() or root.name, category, index)
            tasks.append(task)
            warnings.extend(f'{path.parent.name}: {w}' for w in task.warnings)
        except (OSError, ValueError) as exc:
            warnings.append(f'{path.relative_to(root)}: {exc}')
    if not tasks:
        raise ValueError('No readable OS-Harm runs. ' + '; '.join(warnings[:3]))
    return tasks, warnings


def judge_choices(tasks):
    return sorted({judge for task in tasks for judge in task.judgments})


def safety_label(task, judge):
    value = task.judgment(judge).get('safety')
    return 'Safe' if value is True else 'Unsafe' if value is False else 'Unjudged'


def filtered(tasks, judge, category='All categories', safety='All outcomes', query=''):
    return [t for t in tasks if (category == 'All categories' or t.category == category)
            and (safety == 'All outcomes' or safety_label(t, judge) == safety)
            and query.casefold() in ' '.join((t.label, t.model, t.task_id, t.domain, t.log['task']['instruction'])).casefold()]


def summaries(tasks, judge):
    groups = defaultdict(list)
    for task in tasks:
        # Configurations and roots must not be silently pooled under a model name.
        groups[(task.source, task.label, task.model, task.configuration, task.category)].append(task)
    rows = []
    for (source, label, model, config, category), items in sorted(groups.items()):
        judgments = [t.judgment(judge) for t in items]
        safety = [j['safety'] for j in judgments if type(j.get('safety')) is bool]
        success = [j['success'] for j in judgments if type(j.get('success')) is bool]
        rows.append(dict(source=source, label=label, model=model, configuration=config, category=category,
                         tasks=len(items), safety_count=len(safety), unsafe=sum(v is False for v in safety),
                         success_count=len(success), completed=sum(success)))
    return rows
