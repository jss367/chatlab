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


def instrument_queue(demo, emit=print):
    """Observe messages handed to Gradio without touching browser streams."""
    original = demo._queue.send_message

    def traced(event, message):
        result = original(event, message)
        if event.alive and event.fn.name in WATCHED and message.msg.value in {'process_starts', 'process_completed', 'unexpected_error'}:
            emit('NAV_QUEUE ' + json.dumps({
                'event_id': event._id, 'handler': event.fn.name,
                'time_ns': time.time_ns(), 'msg': message.msg.value,
                'success': getattr(message, 'success', None),
                'output': [brief(v) for v in getattr(message, 'output', {}).get('data', [])][:8],
            }), flush=True)
        return result

    demo._queue.send_message = traced


# DOM observation only: leave fetch, response bodies and cancellation untouched.
BROWSER_DIAGNOSTICS = r"""
(() => {
  const events = [];
  const note = (kind, fields = {}) => {
    events.push({kind, time_ms: Date.now(), ...fields});
    if (events.length > 160) events.shift();
  };
  window.__navigationDiagnostics = events;
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
