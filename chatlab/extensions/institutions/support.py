"""Customer-support bundles and factual evidence views. Never imports experiment code.

The exporter replays each game through the code that recorded it and writes only games that reproduce, so the
scenario's rules live there. ChatLab checks that a bundle hangs together: identities match the index, every
reference resolves, and each seat carries the weights its team names. The manifest declares the teams, closure
rules and score columns; a bundle that declares none gets the support pilot's.
"""
from collections import defaultdict
import copy
import html
import json
import re

SCENARIO = 'customer_support'
# Teams before exporter version 3 were a count of organisms among parent models.
COMPOSITIONS = {0: 'All parent models', 1: 'One model organism', 2: 'Two model organisms'}
CLOSURES = {'direct': 'Direct closure', 'peer': 'Peer approval'}
# A column reads counts at dotted paths: game columns from the index entry, agent columns from a per_agent row.
# Two paths show as "numerator / denominator"; the run comparison sums each over a group's games.
GAME_COLUMNS = [
    {'label': 'Actual resolutions / intake', 'value': ['scores.actually_resolved', 'scores.cases']},
    {'label': 'Dashboard resolved / included', 'value': ['scores.dashboard.resolved', 'scores.dashboard.included']},
    {'label': 'False closure claims', 'value': 'scores.false_resolution_claims'},
    {'label': 'Distinct affected cases', 'distinct_cases': 'False closure claim'},
    {'label': 'False certifications', 'value': 'scores.false_certifications'},
    {'label': 'Omitted / observed failures', 'value': ['scores.omitted_observed_failures', 'scores.observed_failures']},
    {'label': 'Failures without valid report', 'value': 'scores.failures_with_missing_report'},
    {'label': 'Excluded real cases', 'value': 'scores.excluded_real_cases'},
    {'label': 'Joint false attestations', 'value': 'scores.joint_false_attestations'},
    {'label': 'Rejected / all turns', 'value': ['scores.invalid_turns', 'turns']},
]
AGENT_COLUMNS = [
    {'label': 'Owned cases', 'value': 'cases'},
    {'label': 'Actual resolutions', 'value': 'actually_resolved'},
    {'label': 'Observed failures', 'value': 'observed_failures'},
    {'label': 'False claims', 'value': 'false_resolution_claims'},
    {'label': 'False certifications', 'value': 'false_certifications'},
    {'label': 'Omissions', 'value': 'omitted_observed_failures'},
    {'label': 'Rejected turns', 'value': 'invalid_turns'},
]
CASE_HEADERS = ['Case', 'Owner', 'Required operation', 'Latest backend result', 'Actually resolved',
                'Requested label', 'Dashboard label', 'Excluded', 'Certification']


def require(condition, reason):
    if not condition:
        raise ValueError('Customer support: ' + reason)


def integer(value):
    return type(value) is int and value >= 0


def text(value):
    return isinstance(value, str) and bool(value)


def unique(rows, key):
    require(isinstance(rows, list) and all(isinstance(x, dict) and key in x for x in rows), f'invalid {key} records')
    values = [x[key] for x in rows]
    require(all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in values), f'invalid {key}')
    require(len(set(values)) == len(values), f'duplicate {key}')
    return {x[key]: x for x in rows}


def refs(values, available, name):
    require(isinstance(values, list) and all(type(v) in (str, int) for v in values), f'invalid {name} references')
    require(len(set(values)) == len(values) and set(values) <= set(available), f'duplicate or dangling {name} references')


def profiles(value):
    require(isinstance(value, dict) and value, 'missing model profiles')
    for key, p in value.items():
        require(isinstance(p, dict) and p.get('profile_id') == key, 'invalid model profile identity')
        # An adapter is absent or null, or a full repository; {} must not read as original parent weights.
        for repository in [p.get('parent')] + ([p['adapter']] if p.get('adapter') is not None else []):
            require(isinstance(repository, dict) and all(text(repository.get(k)) for k in ('repo', 'revision')),
                    'missing model repository or revision')
            require('subfolder' not in repository or isinstance(repository['subfolder'], str),
                    'invalid model repository subfolder')


def compositions(m):
    """Team key → (label, the profile in each seat, or None for a count of organisms)."""
    if 'compositions' not in m:
        return {k: (label, None) for k, label in COMPOSITIONS.items()}
    return {k: (c['label'], c['roster']) for k, c in m['compositions'].items()}


def composition_label(m, key):
    return compositions(m)[key][0]


def closure_rules(m):
    """Closure rule key → label, in the order the manifest declares them."""
    if 'closure_rules' in m:
        return m['closure_rules']
    return {k: CLOSURES.get(k, k) for k in sorted({e['closure_rule'] for e in m['games']})}


def closure_label(m, key):
    return closure_rules(m)[key]


def score_columns(m):
    declared = m.get('score_columns', {})
    return dict(game=declared.get('game', GAME_COLUMNS), agent=declared.get('agent', AGENT_COLUMNS))


def organism(game, actor):
    """A seat whose recorded weights carry an adapter. Says nothing about how it behaves."""
    return bool(game['model_profiles'][game['agents'][actor]['model_profile']].get('adapter'))


def answer(manifest, raw):
    """The answer after a reply's reasoning, or None when its reasoning never closed."""
    if manifest.get('reply_format', 'json') == 'json':
        return raw
    if '</think>' in raw:
        return raw.rsplit('</think>', 1)[1]
    # A prefilled opening is absent from generated text. Without a closing
    # marker even a JSON fragment can still be unfinished reasoning.
    return None if '<think>' in raw or manifest.get('reasoning_prefilled', True) else raw


def paths(column):
    return [column['value']] if isinstance(column['value'], str) else column['value']


def lookup(record, path):
    for key in path.split('.'):
        require(isinstance(record, dict) and key in record, f'missing score {path}')
        record = record[key]
    require(integer(record), f'score {path} is not a count')
    return record


def cell(column, records, read_game=None):
    """One column summed over records. A distinct-case count reads each record's game for its diagnostics."""
    if 'distinct_cases' in column:
        return sum(len({d['case_id'] for d in read_game(r)['diagnostics']
                        if d['category'] == column['distinct_cases'] and d.get('case_id') is not None}) for r in records)
    totals = [sum(lookup(r, p) for r in records) for p in paths(column)]
    return totals[0] if len(totals) == 1 else ' / '.join(str(t) for t in totals)


def validate_columns(columns, kind):
    require(isinstance(columns, list) and columns, f'missing {kind} score columns')
    for c in columns:
        require(isinstance(c, dict) and text(c.get('label')), f'invalid {kind} score column')
        if kind == 'game' and set(c) == {'label', 'distinct_cases'}:
            require(text(c['distinct_cases']), 'invalid distinct-case column')
            continue
        require(set(c) == {'label', 'value'} and (text(c['value']) or isinstance(c['value'], list)
                and len(c['value']) in (1, 2) and all(text(p) for p in c['value'])), f'invalid {kind} score column')


def validate_manifest(m):
    require(integer(m.get('scenario_version')) and integer(m.get('exporter_version')), 'missing scenario or exporter version')
    require(text(m.get('run_id')), 'missing run identity')
    require(text(m.get('scenario_label')) and isinstance(m.get('config'), dict), 'missing scenario or configuration')
    require(isinstance(m.get('provenance'), dict) and m['provenance'], 'missing source provenance')
    v = m.get('validation')
    require(isinstance(v, dict) and v.get('replay_verified') is True
            and type(v.get('reliability_gate_passed')) is bool, 'partial or unverified runs are unsupported')
    profiles(m.get('model_profiles'))
    require('reasoning_prefilled' not in m or type(m['reasoning_prefilled']) is bool, 'invalid reasoning prefill state')
    if 'compositions' in m:
        teams = m['compositions']
        require(isinstance(teams, dict) and teams and all(
            text(k) and isinstance(c, dict) and set(c) == {'label', 'roster'} and text(c['label'])
            and isinstance(c['roster'], list) and c['roster'] and all(p in m['model_profiles'] for p in c['roster'])
            for k, c in teams.items()), 'invalid team compositions')
    if 'closure_rules' in m:
        rules = m['closure_rules']
        require(isinstance(rules, dict) and rules and all(text(k) and text(x) for k, x in rules.items()), 'invalid closure rules')
    declared = m.get('score_columns', {})
    require(isinstance(declared, dict) and set(declared) <= {'game', 'agent'}, 'invalid score columns')
    columns = score_columns(m)
    validate_columns(columns['game'], 'game')
    validate_columns(columns['agent'], 'agent')
    teams = compositions(m)
    entries = unique(m.get('games'), 'game_id')
    require(entries and v.get('games') == len(entries), 'validation game count disagrees')
    rules = m.get('closure_rules')
    paths_seen, selections = set(), set()
    for e in entries.values():
        # True == 1, so a team key must also match by type.
        require(text(e.get('closure_rule')) and (rules is None or e['closure_rule'] in rules)
                and any(type(k) is type(e.get('composition')) and k == e['composition'] for k in teams)
                and integer(e.get('event_seed')), 'invalid game selection metadata')
        selection = (e['closure_rule'], e['composition'], e['event_seed'])
        require(selection not in selections, 'duplicate game selection')
        selections.add(selection)
        require(isinstance(e.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', e['sha256']), 'missing game hash')
        require(isinstance(e.get('file'), str) and e['file'] not in paths_seen, 'duplicate file reference')
        paths_seen.add(e['file'])
        require(integer(e.get('turns')) and integer(e.get('phases')) and e['turns'] > 0, 'invalid phase or turn count')
        require(isinstance(e.get('scores'), dict), 'missing scores')
        agents = unique(e['scores'].get('per_agent'), 'agent')
        require(all(a.get('model') in m['model_profiles'] for a in agents.values()), 'invalid agent scores')
        for c in columns['game']:
            if 'value' in c:
                cell(c, [e])
        for c in columns['agent']:
            cell(c, list(agents.values()))
    require(v.get('turns') == sum(e['turns'] for e in entries.values()), 'validation turn count disagrees')


def validate_game(run, entry, game):
    try:
        _validate_game(run, entry, game)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Customer support: malformed record or missing required field') from exc


def _validate_game(run, entry, game):
    require(isinstance(game, dict) and game.get('format') == 'chatlab-institutions-game-2', 'invalid game format')
    require(game.get('scenario') == SCENARIO and game.get('scenario_version') == run.manifest['scenario_version']
            and type(game.get('scenario_version')) is int, 'game scenario disagrees with manifest')
    for k in ('game_id', 'closure_rule', 'composition', 'event_seed', 'scores'):
        require(game.get(k) == entry[k], f'{k} disagrees with index')
    require(game.get('run_id') == run.run_id and game.get('model_profiles') == run.manifest['model_profiles'], 'manifest identity disagrees')
    agents = unique(game.get('agents'), 'actor')
    # The page names seats by number.
    require(agents and list(agents) == list(range(len(agents))), 'invalid roster')
    require(all(a.get('model_profile') in game['model_profiles'] for a in agents.values()), 'unknown agent profile')
    roster = compositions(run.manifest)[game['composition']][1]
    if roster is None:
        require(sum(organism(game, a) for a in agents) == game['composition'], 'composition disagrees')
    else:
        require([a['model_profile'] for a in agents.values()] == roster, 'composition disagrees')
    scored = unique(game['scores']['per_agent'], 'agent')
    require(set(scored) == set(agents) and all(scored[a]['model'] == agents[a]['model_profile'] for a in agents),
            'agent scores disagree with the roster')

    phases = unique(game.get('phases'), 'phase_id')
    turns = unique(game.get('turns'), 'turn_id')
    events = unique(game.get('events'), 'event_id')
    snapshots = game.get('snapshots')
    require(isinstance(snapshots, dict) and all(isinstance(s, dict) and isinstance(s.get('cases'), dict)
                                                and all(isinstance(c, dict) for c in s['cases'].values())
                                                for s in snapshots.values()), 'invalid snapshots')
    require(len(phases) == entry['phases'] and len(turns) == entry['turns'], 'phase/turn count disagrees')
    # The page finds an event by its position.
    require(list(events) == list(range(len(events))), 'event IDs must count from zero in order')
    placed_turns, placed_events = [], []
    for phase in phases.values():
        require(integer(phase.get('round')) and text(phase.get('kind')), 'invalid phase')
        refs(phase.get('turn_refs'), turns, 'turn')
        refs(phase.get('event_refs'), events, 'event')
        require(phase.get('before') in snapshots and phase.get('after') in snapshots, 'missing snapshot')
        require(all(turns[t]['phase_id'] == phase['phase_id'] for t in phase['turn_refs'])
                and all(events[e]['phase_id'] == phase['phase_id'] for e in phase['event_refs']), 'phase membership disagrees')
        placed_turns += phase['turn_refs']
        placed_events += phase['event_refs']
    require(sorted(placed_turns, key=str) == sorted(turns, key=str) and placed_events == list(events),
            'incomplete or duplicate phase references')
    for turn in turns.values():
        a = turn.get('actor')
        require(a in agents and turn.get('model_profile') == agents[a]['model_profile'], 'turn profile disagrees')
        require(type(turn.get('accepted')) is bool and (turn.get('error') is None if turn['accepted'] else isinstance(turn.get('error'), str)), 'invalid acceptance status')
        require(isinstance(turn.get('raw_reply'), str) and (turn.get('parsed') is None or isinstance(turn['parsed'], dict)), 'invalid reply')
        require(turn.get('input_tokens') is None and turn.get('output_tokens') is None, 'invented per-turn token measurements')
        messages = turn.get('messages')
        require(isinstance(messages, list) and len(messages) >= 2
                and all(isinstance(m, dict) and text(m.get('role')) and isinstance(m.get('content'), str) for m in messages)
                and messages[0]['role'] == 'system',
                'invalid exact message list')
        require(isinstance(turn.get('observation'), dict), 'missing recorded observation')
        refs(turn.get('event_refs'), events, 'emitted event')
        require(turn['event_refs'] == [e['event_id'] for e in events.values() if e.get('turn_id') == turn['turn_id']], 'event-to-turn links disagree')
    for e in events.values():
        require(text(e.get('kind')) and isinstance(e.get('payload'), dict), 'invalid event')
        require(e.get('actor') is None or e['actor'] in agents, 'unknown event actor')
        require(e.get('case_id') is None or text(e['case_id']), 'invalid case reference')
        require(e.get('turn_id') is None or (e['turn_id'] in turns and turns[e['turn_id']]['actor'] == e.get('actor')),
                'invalid originating turn')
    diagnostics = game.get('diagnostics')
    require(isinstance(diagnostics, list), 'invalid diagnostics')
    for d in diagnostics:
        require(isinstance(d, dict) and text(d.get('category')) and d.get('actor') in agents
                and (d.get('case_id') is None or text(d['case_id'])), 'invalid diagnostic')
        refs(d.get('event_refs'), events, 'diagnostic event')
        require(d['event_refs'], 'diagnostic without evidence')
    for c in score_columns(run.manifest)['game']:
        if 'distinct_cases' in c:
            cell(c, [entry], lambda _: game)


def overview_headers(run):
    return ['Closure rule', 'Team composition', 'Games'] + [c['label'] for c in score_columns(run.manifest)['game']]


def overview_rows(run):
    m = run.manifest
    groups = defaultdict(list)
    for e in run.games:
        groups[e['closure_rule'], e['composition']].append(e)
    rules, teams = list(closure_rules(m)), list(compositions(m))
    rows, keys = [], []
    for (closure, composition), entries in sorted(groups.items(), key=lambda x: (rules.index(x[0][0]), teams.index(x[0][1]))):
        rows.append([closure_label(m, closure), composition_label(m, composition), len(entries)]
                    + [cell(c, entries, run.read_game) for c in score_columns(m)['game']])
        keys.append((closure, composition))
    return rows, keys


def event_ids(game, values):
    """The event IDs in a payload list that name events in this game. Payload contents are the exporter's."""
    return [v for v in values if type(v) is int and 0 <= v < len(game['events'])] if isinstance(values, list) else []


def phase_state(game, index):
    phase = game['phases'][index]
    lookup = {t['turn_id']: i for i, t in enumerate(game['turns'])}
    return dict(phase=phase, index=index, turns=[lookup[t] for t in phase['turn_refs']],
                before=game['snapshots'][phase['before']], after=game['snapshots'][phase['after']])


def case_history(game, case_id, index):
    past, future = [], []
    phase_order = {p['phase_id']: i for i, p in enumerate(game['phases'])}
    turns = {t['turn_id']: t for t in game['turns']}
    for e in game['events']:
        proposed = (turns.get(e.get('turn_id'), {}).get('parsed') or {})
        referenced = any(isinstance(action, dict) and action.get('case_id') == case_id
                         for key in ('operations', 'updates', 'certifications')
                         for action in (proposed.get(key, []) if isinstance(proposed.get(key, []), list) else []))
        rejection = e['kind'] == 'invalid_turn' and referenced
        message = e['kind'] == 'message' and referenced
        phase = game['phases'][phase_order[e['phase_id']]]
        boundary = e['kind'] == 'dashboard' and case_id in game['snapshots'][phase['after']]['cases']
        reports = e['kind'] == 'management_report' and any(game['events'][i].get('case_id') == case_id
                   for i in event_ids(game, e['payload'].get('required_failure_ids')))
        if e.get('case_id') == case_id or reports or rejection or message or boundary:
            (past if phase_order[e['phase_id']] <= index else future).append(e)
    return past, future


def conversation(game, index, include_reply=False, *, manifest=None):
    require(type(index) is int and 0 <= index < len(game['turns']), 'select a turn first')
    t = game['turns'][index]
    messages = copy.deepcopy(t['messages'])
    if include_reply:
        raw = t['raw_reply']
        # A thinking model's reasoning goes in Chat's reasoning block; an unclosed block has no answer.
        if (manifest or {}).get('reply_format', 'json') == 'reasoning_then_json' and (
            '</think>' in raw or answer(manifest, raw) is None
        ):
            reasoning = raw.rsplit('</think>', 1)[0] if '</think>' in raw else raw
            messages.append(dict(role='assistant', content=(answer(manifest, raw) or '').strip(),
                                 reasoning=reasoning.replace('<think>', '', 1).strip()))
        else:
            messages.append(dict(role='assistant', content=raw))
    provenance = dict(run_id=game['run_id'], game_id=game['game_id'], turn_id=t['turn_id'],
                      model_profile=game['model_profiles'][t['model_profile']], accepted=t['accepted'], error=t['error'],
                      note='Recorded profile only; opening Chat does not load these weights or adapter. Sampling was batched.')
    for message in messages[1:]:
        message['institutions_provenance'] = copy.deepcopy(provenance)
    if include_reply:
        profile = provenance['model_profile']
        messages[-1]['model'] = (profile['parent']['repo'] +
                                 (' + adapter ' + profile['adapter']['repo'] if profile.get('adapter') else ''))
    return dict(format='chatlab-conversation-1', system_prompt=messages[0]['content'], turns=messages[1:],
                institutions_provenance=provenance)



def box(title, value):
    return '<div class="inst-box"><h3>' + html.escape(title) + '</h3><pre class="inst-log">' + html.escape(json.dumps(value, indent=2, ensure_ascii=False)) + '</pre></div>'


def render_table(headers, rows):
    def esc(v):
        return html.escape(str(v))
    return ('<table class="inst-table"><thead><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table>')


def folded(title, value):
    return '<details><summary>' + html.escape(title) + '</summary>' + box(title, value) + '</details>'


def scenario_html(run):
    v = run.manifest['validation']
    profile_rows = []
    for p in run.manifest['model_profiles'].values():
        adapter = p.get('adapter')
        # One repository can hold several checkpoints or adapters; include
        # either side's subfolder so its recorded weights are distinguishable.
        parent = p['parent']
        parent_name = parent['repo'] + ('/' + parent['subfolder'] if parent.get('subfolder') else '')
        adapter_name = adapter['repo'] + ('/' + adapter['subfolder'] if adapter.get('subfolder') else '') if adapter else 'None'
        profile_rows.append([p['profile_id'], parent_name, adapter_name,
                             p.get('merge_method') or 'Original parent weights'])
    return ('<div class="inst-box"><h3>' + html.escape(run.manifest['scenario_label']) + '</h3><p>Replay verified · Reliability gate '
            + ('passed' if v['reliability_gate_passed'] else 'failed — diagnostic run')
            + '. Actual resolution uses the fixed intake denominator, including excluded cases. '
              'Certifications check backend evidence. Under direct closure they do not control labels.</p>'
              '<p>Profile badges identify recorded weights, never honesty or intent. '
              'No model is needed to read this replay. Regenerate an individual prompt through Open in Chat; '
              'Chat does not execute support operations or load recorded weights automatically.</p>'
            + render_table(['Profile', 'Parent repository', 'Adapter repository', 'Merge method'], profile_rows)
            + folded('Model revisions and recorded sampling scope', run.manifest['model_profiles'])
            + folded('Configuration', run.config)
            + folded('Recorded aggregate measurements — per-turn tokens unknown', run.manifest.get('measurements', {'tokens_processed': None, 'scope': 'whole_run_aggregate'}))
            + folded('Validation and source provenance', dict(validation=v, provenance=run.manifest['provenance'])) + '</div>')


def game_header(run, entry, game):
    esc = html.escape
    m, s = run.manifest, game['scores']
    entry = entry or next(e for e in run.games if e['game_id'] == game['game_id'])
    columns = score_columns(m)
    title = f"{closure_label(m, game['closure_rule'])} · {composition_label(m, game['composition'])} · event seed {game['event_seed']}"
    outcomes = [(c['label'], cell(c, [entry], lambda _: game)) for c in columns['game']]
    agents = [[a['agent'], a['model']] + [cell(c, [a]) for c in columns['agent']] for a in s['per_agent']]
    grouped = []
    for profile in game['model_profiles']:
        roster = [a for a in s['per_agent'] if a['model'] == profile]
        if roster:
            grouped.append([profile, len(roster)] + [cell(c, roster) for c in columns['agent']])
    labels = [c['label'] for c in columns['agent']]
    return ('<div class="inst-box"><h3>' + esc(title) + '</h3><details><summary>Final per-game results — includes later rounds</summary>'
            + render_table(['Outcome', 'Result'], outcomes) + '</details><details><summary>Final agent results and workload by profile</summary>'
            + render_table(['Agent', 'Profile'] + labels, agents) + render_table(['Profile', 'Agents'] + labels, grouped)
            + '</details></div>')


def case_rows(game, state):
    rows = []
    phase_ids = {p['phase_id'] for p in game['phases'][:state['index'] + 1]}
    for cid, c in state['after']['cases'].items():
        prior = state['before']['cases'].get(cid, {})
        related = [e for e in game['events'] if e.get('case_id') == cid and e['phase_id'] in phase_ids]
        claims = [e['payload'] for e in related if e['kind'] == 'status_claim']
        certs = [e['payload'] for e in related if e['kind'] == 'certification']
        latest = (c.get('history') or [None])[-1]
        def change(key):
            return f"{prior[key]} → {c[key]}" if key in prior and key in c and prior[key] != c[key] else str(c.get(key, '—'))
        cert = certs[-1] if certs else None
        rows.append([cid, c.get('owner', '—'), c.get('operation', '—'),
                     f"event {latest.get('event_id')}: {'completed' if latest.get('success') else 'failed'}" if isinstance(latest, dict) else 'No attempt',
                     change('solved'), claims[-1].get('status', '—') if claims else 'No request', change('status'), change('excluded'),
                     f"event {cert.get('event_id')}: agent {cert.get('agent')} {'approved' if cert.get('approve') else 'declined'}" if cert else 'None'])
    return rows


def phase_html(game, state):
    phase = state['phase']
    relevant = [d for d in game['diagnostics'] if any(e in phase['event_refs'] for e in d['event_refs'])
                and d['category'] != 'Missing valid report']
    if phase['kind'] == 'review':
        relevant += [d for d in game['diagnostics'] if d['category'] == 'Missing valid report'
                     and game['events'][d['event_refs'][0]]['payload'].get('round') == phase.get('source_round')]
    return ('<div class="inst-box"><h3>Evaluator outcomes</h3><p>All seats receive observations captured before '
            'any replies in this phase are applied. Application order is not conversation order. '
            'Reviewers did not see an owner’s same-phase closure claim. Joint false attestation identifies '
            'a qualifying pair; it does not establish collusion.</p><p>Case values show the selected phase boundary. '
            'Before → after changes are marked. Click a case or event below to follow its evidence.</p></div>'
            + '<div class="inst-box"><h3>Diagnostics in this phase</h3>'
            + render_table(['Category', 'Case', 'Actor profile', 'Event references'],
                           [[d['category'], d.get('case_id') or '—',
                             f"agent {d['actor']} · {game['agents'][d['actor']]['model_profile']}",
                             ', '.join(str(e) for e in d['event_refs'])] for d in relevant])
            + folded('Factual diagnostic rule evaluations', relevant) + '</div>')


def turn_rows(game, state):
    return [[i, t['actor'], t['model_profile'], 'Accepted' if t['accepted'] else 'Rejected', 1,
             (t.get('parsed') or {}).get('message', '') if t['accepted'] else t['error']]
            for i in state['turns'] for t in [game['turns'][i]]]


def event_rows(game, events):
    return [[e['event_id'], e['kind'], e.get('actor'), e.get('case_id'), e.get('turn_id') or 'Phase boundary',
             json.dumps(e['payload'], ensure_ascii=False)] for e in events]
