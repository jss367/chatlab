"""Exercise fixture telemetry without a server, browser, or native inference."""
import json
import asyncio
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock
import gradio as gr
from gradio.server_messages import ProcessCompletedMessage

from browser.navigation_diagnostics import BROWSER_DIAGNOSTICS, brief, instrument, instrument_queue


class NavigationDiagnosticsTests(unittest.TestCase):
    def test_handler_keeps_output_identity_and_reports_branch_and_transcript(self):
        chatbot, forks = object(), object()
        result = ([], {'active': 'Chat 1', 'branches': {'Main': [], 'Chat 1': []}})
        fn = SimpleNamespace(name='new_conversation', fn=lambda *_: result, outputs=[chatbot, forks])
        demo = SimpleNamespace(fns={7: fn}, conversation_outputs={'chatbot': chatbot, 'forks': forks})
        emit = mock.Mock()
        instrument(demo, emit)
        self.assertIs(fn.fn({'active': 'Main', 'branches': {'Main': []}}), result)
        records = [json.loads(call.args[0].removeprefix('NAV_SERVER ')) for call in emit.call_args_list]
        self.assertEqual([r['phase'] for r in records], ['start', 'end'])
        self.assertEqual(records[0]['call'], records[1]['call'])
        self.assertEqual(records[1]['fn_index'], 7)
        self.assertEqual(records[1]['outputs']['chatbot']['count'], 0)
        self.assertEqual(records[1]['outputs']['forks']['active'], 'Chat 1')

    def test_exception_is_reported_and_propagates(self):
        def fail():
            raise ValueError('synthetic')
        fn = SimpleNamespace(name='poll', fn=fail, outputs=[])
        emit = mock.Mock()
        instrument(SimpleNamespace(fns={2: fn}, conversation_outputs={}), emit)
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            fn.fn()
        self.assertEqual(json.loads(emit.call_args.args[0].removeprefix('NAV_SERVER '))['error'], 'ValueError')

    def test_skips_and_large_text_are_compact(self):
        self.assertEqual(brief({'__type__': 'update'}), {'skip': True})
        self.assertEqual(len(brief(['x' * 1000])['text']), 240)

    def test_queue_observer_preserves_message_delivery_and_reports_event_id(self):
        original = mock.Mock(return_value='delivered')
        queue = SimpleNamespace(send_message=original)
        emit = mock.Mock()
        instrument_queue(SimpleNamespace(_queue=queue), emit)
        event = SimpleNamespace(alive=True, _id='synthetic-event', fn=SimpleNamespace(name='new_conversation'))
        message = SimpleNamespace(msg=SimpleNamespace(value='process_completed'), success=True, output={'data': [[]]})
        self.assertEqual(queue.send_message(event, message), 'delivered')
        original.assert_called_once_with(event, message)
        record = json.loads(emit.call_args.args[0].removeprefix('NAV_QUEUE '))
        self.assertEqual(record['event_id'], 'synthetic-event')
        self.assertEqual(record['output'][0]['count'], 0)

    def test_dom_observer_preserves_fetch_and_bounds_history(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node is unavailable for fixture script verification')
        harness = r"""
const assert = require('assert');
let sample, text = 'Original conversation';
const fetch = () => {throw new Error('DOM telemetry must never fetch');};
global.window = {fetch};
global.document = {readyState: 'complete', body: {},
  querySelector: () => ({innerText: text}), querySelectorAll: () => [{checked: true}, {checked: false}]};
global.MutationObserver = class {constructor(fn) {sample = fn;} observe() {}};
"""
        checks = r"""
assert.strictEqual(window.fetch, fetch);
for (let i=0; i<200; i++) {text = String(i); sample();}
assert.strictEqual(window.__navigationDiagnostics.length, 160);
assert.strictEqual(window.__navigationDiagnostics.at(-1).selected, 0);
assert.strictEqual(window.__navigationDiagnostics.at(-1).transcript, '199');
"""
        subprocess.run([node, '-e', harness + BROWSER_DIAGNOSTICS + checks], check=True, capture_output=True, text=True)

    def test_real_gradio_queue_message_is_enqueued_unchanged(self):
        demo = gr.Blocks().queue()
        messages = asyncio.Queue()
        demo._queue.pending_messages_per_session['synthetic'] = messages
        event = SimpleNamespace(alive=True, _id='synthetic-event', session_hash='synthetic',
                                fn=SimpleNamespace(name='new_conversation'))
        message = ProcessCompletedMessage(output={'data': [[]]}, success=True)
        emit = mock.Mock()
        instrument_queue(demo, emit)
        demo._queue.send_message(event, message)
        self.assertIs(messages.get_nowait(), message)
        self.assertEqual(message.event_id, 'synthetic-event')
        self.assertEqual(json.loads(emit.call_args.args[0].removeprefix('NAV_QUEUE '))['success'], True)
