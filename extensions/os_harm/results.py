"""Read OS-Harm artifacts without importing or executing the benchmark harness."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import threading

CATEGORIES = ('Deliberate user misuse', 'Prompt injection attacks', 'Model misbehavior')
MANIFESTS = dict(zip(('test_misuse.json', 'test_injection.json', 'test_misbehavior.json'), CATEGORIES))
UNKNOWN = 'Unknown category'
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TRAJECTORY_LINES = 10_000
MAX_TRAJECTORY_WARNINGS = 20
MAX_CACHED_ARTIFACTS = 8

_cache = OrderedDict()
_cache_lock = threading.Lock()


def cached(kind, path, read):
    """Hold a few tasks' bulky artifacts, keyed by file identity.

    Recorded steps and execution records dwarf everything else a result
    directory holds, so loaded tasks keep only their summary and read these
    back when a reader actually replays them.
    """
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    key = (kind, str(path))
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None and entry[0] == stamp:
            _cache.move_to_end(key)
            return entry[1]
    value = read()
    with _cache_lock:
        _cache[key] = (stamp, value)
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHED_ARTIFACTS:
            _cache.popitem(last=False)
    return value


def resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except RuntimeError as exc:
        # Python 3.12 reports symlink cycles as RuntimeError, while newer
        # pathlib versions use OSError. Keep artifact failures on one boundary.
        raise ValueError(f'Could not resolve {path}: {exc}') from exc


def local_file(directory: Path, name: str) -> Path:
    """Artifacts may only refer to files inside their own task directory."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Expected a relative artifact filename.')
    path = resolve_path(directory / name)
    if not path.is_relative_to(resolve_path(directory)):
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
    params: dict
    info: dict
    step_count: int
    judgments: dict
    errors: list
    warnings: list

    @property
    def key(self):
        return str(self.directory)

    @property
    def model(self):
        return str(self.params.get('model', 'Unknown model'))

    @property
    def configuration(self):
        return ' / '.join(str(self.params.get(k, 'Unknown')) for k in ('action_space', 'observation_type'))

    @property
    def run(self):
        """One label, model and configuration: what a task-by-task diff pairs."""
        return ' · '.join((self.label, self.model, self.configuration))

    @property
    def instruction(self):
        return str(self.info.get('instruction', ''))

    def judgment(self, judge):
        return self.judgments.get(judge, {})

    def steps(self):
        def read():
            return validate_log(read_json(local_file(self.directory, 'better_log.json')))['steps']

        return cached('steps', self.directory / 'better_log.json', read)

    def execution(self):
        """Returns the execution records and any warnings from reading them."""
        return cached('trajectory', self.directory / 'traj.jsonl', lambda: read_trajectory(self.directory))


def category_index(folder: str) -> dict:
    if not folder.strip():
        return {}
    root = resolve_path(Path(folder).expanduser())
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


def read_trajectory(directory):
    entries, warnings = [], []
    try:
        traj_path = local_file(directory, 'traj.jsonl')
        if traj_path.exists():
            if traj_path.stat().st_size > MAX_JSON_BYTES:
                warnings.append('traj.jsonl exceeds 32 MB; execution details were skipped.')
            else:
                total_bytes = invalid_lines = 0
                with traj_path.open('rb') as stream:
                    for line_no in range(1, MAX_TRAJECTORY_LINES + 2):
                        # Bound allocation even if the file grows after stat(),
                        # and avoid materializing millions of split lines.
                        raw_line = stream.readline(MAX_JSON_BYTES - total_bytes + 1)
                        if not raw_line:
                            break
                        if line_no > MAX_TRAJECTORY_LINES:
                            warnings.append(f'traj.jsonl exceeds {MAX_TRAJECTORY_LINES:,} lines; remaining execution details were skipped.')
                            break
                        total_bytes += len(raw_line)
                        if total_bytes > MAX_JSON_BYTES:
                            warnings.append('traj.jsonl exceeds 32 MB; remaining execution details were skipped.')
                            break
                        line = raw_line.decode('utf-8')
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if not isinstance(entry, dict):
                                raise ValueError('Expected an object.')
                            entries.append(entry)
                        except ValueError:
                            invalid_lines += 1
                            if invalid_lines <= MAX_TRAJECTORY_WARNINGS:
                                warnings.append(f'traj.jsonl line {line_no} is invalid; other steps remain available.')
                if invalid_lines > MAX_TRAJECTORY_WARNINGS:
                    warnings.append(f'traj.jsonl: {invalid_lines - MAX_TRAJECTORY_WARNINGS:,} additional invalid-line warnings omitted.')
    except (OSError, ValueError) as exc:
        # Execution records are optional; a decoding or filesystem failure
        # must not remove a valid task and its judgments from the comparison.
        warnings.append(f'traj.jsonl could not be read; execution details were skipped: {exc}')
    return entries, warnings


def load_task(directory, root, label, category, index):
    warnings, judgments = [], {}
    log = validate_log(read_json(local_file(directory, 'better_log.json')))
    domain, task_id = directory.parent.name, directory.name
    if category == 'Automatic':
        if log['task'].get('injection'):
            category = CATEGORIES[1]
        else:
            category = index.get((domain, task_id.split('__inject__')[0]), UNKNOWN)
    try:
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
    except (OSError, ValueError) as exc:
        warnings.append(f'judgment directory could not be read; unavailable judgments were skipped: {exc}')
    # Read the execution records for their warnings and reported errors, then
    # let them go: a replayed task reads its own artifacts back on demand.
    entries, traj_warnings = read_trajectory(directory)
    warnings.extend(traj_warnings)
    errors = [str(entry['Error']) for entry in entries if 'Error' in entry]
    if len(errors) > MAX_TRAJECTORY_WARNINGS:
        errors = errors[:MAX_TRAJECTORY_WARNINGS]
        errors.append('Further execution errors are listed in the execution records.')
    return Task(directory, str(root), label, domain, task_id, category, log['params'], log['task'],
                len(log['steps']), judgments, errors, warnings)


def import_results(folder: str, label='', category='Automatic', definitions=''):
    if not folder.strip():
        raise ValueError('Enter an OS-Harm results directory.')
    root = resolve_path(Path(folder).expanduser())
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
            if not resolve_path(path).is_relative_to(root):
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


def run_choices(tasks):
    return sorted({task.run for task in tasks})


def safety_label(task, judge):
    value = task.judgment(judge).get('safety')
    return 'Safe' if value is True else 'Unsafe' if value is False else 'Unjudged'


def completion_label(task, judge):
    value = task.judgment(judge).get('success')
    return 'Completed' if value is True else 'Not completed' if value is False else 'Unjudged'


def filtered(tasks, judge, category='All categories', safety='All outcomes', query=''):
    return [t for t in tasks if (category == 'All categories' or t.category == category)
            and (safety == 'All outcomes' or safety_label(t, judge) == safety)
            and query.casefold() in ' '.join((t.label, t.model, t.task_id, t.domain, t.instruction)).casefold()]


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


def by_task(tasks, run):
    """Index one run's tasks by application and task ID."""
    index, duplicates = {}, 0
    for task in sorted(tasks, key=lambda t: t.key):
        if task.run != run:
            continue
        if (task.domain, task.task_id) in index:
            duplicates += 1
        else:
            index[(task.domain, task.task_id)] = task
    return index, duplicates


EMPTY_COMPARISON = dict(rows=[], keys=[], shared=0, baseline_only=0, comparison_only=0,
                        duplicates=0, regressions=0, improvements=0, unjudged=0)


def compare_runs(tasks, judge, baseline, comparison, match=None):
    """Pair two runs by application and task ID and report where they differ.

    Only tasks both runs attempted are paired; a rate over different task sets
    would compare the task lists as much as the models. Pairing happens before
    `match` narrows the result, so a search naming one run's label or model
    keeps that run's pairs instead of dropping the other side of each one.
    """
    if not baseline or not comparison or baseline == comparison:
        return dict(EMPTY_COMPARISON)
    before_tasks, before_duplicates = by_task(tasks, baseline)
    after_tasks, after_duplicates = by_task(tasks, comparison)
    wanted = (lambda *pair: True) if match is None else (lambda *pair: any(match(t) for t in pair))
    paired = before_tasks.keys() & after_tasks.keys()
    shared = sorted(k for k in paired if wanted(before_tasks[k], after_tasks[k]))
    rows, keys, regressions, improvements, unjudged = [], [], 0, 0, 0
    for key in shared:
        before, after = before_tasks[key], after_tasks[key]
        safety = (safety_label(before, judge), safety_label(after, judge))
        completion = (completion_label(before, judge), completion_label(after, judge))
        if 'Unjudged' in safety:
            unjudged += 1
        elif safety == ('Safe', 'Unsafe'):
            regressions += 1
        elif safety == ('Unsafe', 'Safe'):
            improvements += 1
        if safety[0] != safety[1] or completion[0] != completion[1]:
            rows.append([after.domain, after.task_id, after.category,
                         ' → '.join(safety), ' → '.join(completion), after.instruction])
            keys.append(after.key)
    return dict(rows=rows, keys=keys, shared=len(shared),
                baseline_only=sum(wanted(task) for key, task in before_tasks.items() if key not in paired),
                comparison_only=sum(wanted(task) for key, task in after_tasks.items() if key not in paired),
                duplicates=before_duplicates + after_duplicates,
                regressions=regressions, improvements=improvements, unjudged=unjudged)


def cohen_kappa(pairs):
    """Chance-corrected agreement, or None when one label leaves no chance to correct for."""
    total = len(pairs)
    if not total:
        return None
    observed = sum(a == b for a, b in pairs) / total
    expected = sum((sum(a is value for a, _ in pairs) / total) * (sum(b is value for _, b in pairs) / total)
                   for value in (True, False))
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def judge_agreement(tasks, first, second):
    """Compare two judges, or a judge and a human reviewer, on the tasks both judged."""
    fields = ('safety', 'success')
    pairs = {field: [] for field in fields}
    rows, keys = [], []
    if not first or not second or first == second:
        return dict(stats={field: dict(judged=0, agree=0, kappa=None) for field in fields}, rows=[], keys=[])
    for task in sorted(tasks, key=lambda t: t.key):
        one, two = task.judgment(first), task.judgment(second)
        differs = False
        for field in fields:
            a, b = one.get(field), two.get(field)
            if type(a) is bool and type(b) is bool:
                pairs[field].append((a, b))
                differs = differs or a != b
        if differs:
            rows.append([task.label, task.model, task.domain, task.task_id,
                         ' / '.join((safety_label(task, first), safety_label(task, second))),
                         ' / '.join((completion_label(task, first), completion_label(task, second)))])
            keys.append(task.key)
    stats = {field: dict(judged=len(values), agree=sum(a == b for a, b in values), kappa=cohen_kappa(values))
             for field, values in pairs.items()}
    return dict(stats=stats, rows=rows, keys=keys)
