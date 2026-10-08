"""Bounded telemetry for the synthetic browser fixture, never the app."""

import functools
import itertools
import json
import time

WATCHED = {'chat', 'new_conversation', 'poll', 'switch_fork', 'restore_conversations'}


def brief(value):
    if isinstance(value, dict):
        if 'branches' in value:
            return {'active': value.get('active'), 'branches': list(value['branches'])}
        if value.get('__type__') == 'update':
            return {'update': brief(value['value'])} if 'value' in value else {'skip': True}
    if isinstance(value, list):
        return {'count': len(value), 'text': str(value)[:240]}
    return str(value)[:240]


def instrument(demo, emit=print):
    sequence = itertools.count(1)
    for index, fn in demo.fns.items():
        if fn.name not in WATCHED:
            continue
        original = fn.fn
        names = {component: name for name, component in demo.conversation_outputs.items()}

        def wrap(original=original, fn=fn, index=index):
            @functools.wraps(original)
            def traced(*args, **kwargs):
                call = next(sequence)
                def record(phase, **fields):
                    emit('NAV_SERVER ' + json.dumps({
                        'call': call, 'fn_index': index, 'handler': fn.name,
                        'phase': phase, 'time_ns': time.time_ns(), **fields,
                    }), flush=True)
                record('start', state=[brief(a) for a in args if isinstance(a, dict) and 'branches' in a])
                try:
                    result = original(*args, **kwargs)
                except Exception as exc:
                    record('error', error=type(exc).__name__)
                    raise
                outputs = {
                    names[c]: brief(v) for c, v in zip(fn.outputs, result, strict=True)
                    if names.get(c) in {'chatbot', 'forks', 'conversation_list', 'status'}
                }
                record('end', outputs=outputs)
                return result
            return traced
        fn.fn = wrap()


# Observe a clone of the fixture's queue stream. The app consumes the original.
# Bound the number and size of records; do not store session IDs or request data.
BROWSER_DIAGNOSTICS = r"""
(() => {
  const events = [];
  const note = (kind, fields = {}) => {
    events.push({kind, time_ms: Date.now(), ...fields});
    if (events.length > 160) events.shift();
  };
  window.__navigationDiagnostics = events;
  const realFetch = window.fetch;
  window.fetch = async (...args) => {
    const response = await realFetch(...args);
    const url = String(args[0]?.url || args[0]);
    if (!url.includes('/queue/data')) return response;
    const reader = response.clone().body?.getReader();
    if (!reader) return response;
    (async () => {
      const decoder = new TextDecoder();
      let pending = '';
      try {
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          pending += decoder.decode(value, {stream: true});
          let newline;
          while ((newline = pending.indexOf('\n')) >= 0) {
            const line = pending.slice(0, newline); pending = pending.slice(newline + 1);
            if (!line.startsWith('data:')) continue;
            try {
              const data = JSON.parse(line.slice(5));
              if (!['process_completed', 'process_starts', 'unexpected_error'].includes(data.msg)) continue;
              note('queue', {msg: data.msg, event_id: data.event_id, success: data.success,
                output: JSON.stringify(data.output?.data ?? []).slice(0, 500)});
            } catch (_) { /* Ignore unrelated stream lines. */ }
          }
          if (pending.length > 65536) { note('parser_overflow'); break; }
        }
      } catch (error) { note('queue_read_error', {error: String(error).slice(0, 200)}); }
      finally { reader.cancel().catch(() => {}); }
    })();
    return response;
  };
  const observe = () => {
    let previous = '';
    const sample = () => {
      const transcript = document.querySelector('#conversation')?.innerText?.slice(0, 300) || '';
      const rows = [...document.querySelectorAll('#conversation-list input[type=radio]')];
      const state = {transcript, rows: rows.length, selected: rows.findIndex(row => row.checked)};
      const key = JSON.stringify(state);
      if (key !== previous) { previous = key; note('dom', state); }
    };
    new MutationObserver(sample).observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['checked']});
    sample();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', observe);
  else observe();
})();
"""
