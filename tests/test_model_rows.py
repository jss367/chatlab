"""Confirmation behavior of the real model-row script in a browser stand-in."""

import shutil
import subprocess
import unittest

from chatlab.ui.model_rows import MODEL_ROWS_JS, REMOVE_CONFIRM_MS


@unittest.skipUnless(shutil.which("node"), "needs node to run the page script")
class ModelRowsTests(unittest.TestCase):
    def test_confirmation_is_visible_and_only_a_second_press_deletes(self):
        script = r"""
const assert = require('node:assert/strict');
global.window = {};
const timers = new Map();
let nextTimer = 0;
global.setTimeout = (callback, delay) => {
  timers.set(++nextTimer, {callback, delay});
  return nextTimer;
};
global.clearTimeout = id => timers.delete(id);
global.MutationObserver = class { observe() {} };
global.HTMLInputElement = class {
  set value(value) { this.sent = JSON.parse(value); }
};
const bridge = new HTMLInputElement();
bridge.tagName = 'INPUT';
let dispatched = 0;
bridge.dispatchEvent = () => dispatched++;
let click;
global.document = {
  body: {},
  getElementById: () => ({querySelectorAll: () => []}),
  querySelector: () => bridge,
  addEventListener: (event, handler) => { click = handler; },
};
const classes = new Set();
const button = {
  dataset: {action: 'remove', name: 'org/model'},
  textContent: '',
  classList: {
    add: value => classes.add(value),
    remove: value => classes.delete(value),
    contains: value => classes.has(value),
  },
  setAttribute: () => {},
};
const press = () => click({
  target: {closest: () => button},
  preventDefault() {}, stopPropagation() {},
});
START();
press();
assert.match(button.textContent, /Press again to remove/);
assert.equal(dispatched, 0);
assert.equal(timers.get(button.confirmTimer).delay, CONFIRM_MS);
press();
assert.equal(dispatched, 1);
assert.equal(bridge.sent.name, 'org/model');
assert.equal(bridge.sent.action, 'remove');
assert.equal(button.textContent, '');
assert.equal(timers.size, 0);
// Expiring the confirmation resets it without sending a deletion.
press();
timers.get(button.confirmTimer).callback();
assert.equal(button.textContent, '');
assert.equal(classes.has('armed'), false);
assert.equal(dispatched, 1);
press();
assert.match(button.textContent, /Press again/);
assert.equal(dispatched, 1);
"""
        script = script.replace("START()", f"({MODEL_ROWS_JS})()")
        script = script.replace("CONFIRM_MS", str(REMOVE_CONFIRM_MS))
        result = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=20
        )
        self.assertEqual(result.returncode, 0, result.stderr)
