"""Exercise fixture telemetry without a server, browser, or native inference."""
import json
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from browser.navigation_diagnostics import BROWSER_DIAGNOSTICS, brief, instrument


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

    def test_queue_parser_handles_chunk_boundaries_and_bounds_dom_history(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node is unavailable for fixture script verification')
        harness = r"""
const assert = require('assert');
let sample, text = 'Original conversation';
global.window = {};
global.document = {readyState: 'complete', body: {},
  querySelector: () => ({innerText: text}), querySelectorAll: () => [{checked: true}, {checked: false}]};
global.MutationObserver = class {constructor(fn) {sample = fn;} observe() {}};
const chunks = ['data: {"msg":"process_sta', 'rts","event_id":"e1"}\n',
 'data: {"msg":"process_completed","event_id":"e1","success":true,"output":{"data":[[]]}}\n'];
let offset = 0;
const original = {clone: () => ({body: {getReader: () => ({
 read: async () => offset < chunks.length ? {value: new TextEncoder().encode(chunks[offset++]), done: false} : {done: true},
 cancel: async () => {}
})}})};
window.fetch = async () => original;
"""
        checks = r"""
(async () => {
 assert.strictEqual(await window.fetch('/gradio_api/queue/data?session_hash=hidden'), original);
 await new Promise(resolve => setImmediate(resolve));
 const queues = window.__navigationDiagnostics.filter(x => x.kind === 'queue');
 assert.strictEqual(queues.length, 2);
 assert.strictEqual(queues[1].event_id, 'e1');
 assert.strictEqual(queues[1].output, '[[]]');
 assert(!JSON.stringify(queues).includes('hidden'));
 for (let i=0; i<200; i++) {text = String(i); sample();}
 assert.strictEqual(window.__navigationDiagnostics.length, 160);
 assert.strictEqual(window.__navigationDiagnostics.at(-1).selected, 0);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
        subprocess.run([node, '-e', harness + BROWSER_DIAGNOSTICS + checks], check=True, capture_output=True, text=True)
