"""Dragging a table column wider, and what the drag must not disturb."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import app


# Enough of a Gradio dataframe for COLUMN_JS to run against: the two nested
# wraps it writes between, a header row, and stand-ins for the browser it
# talks to. Nothing here lays anything out, so every header is told where
# its edges are.
TABLE_PAGE = """
'use strict';
const assert = require('node:assert');

class Style {
  constructor() { this.props = {}; }
  setProperty(name, value) { this.props[name] = value; }
  removeProperty(name) { delete this.props[name]; }
  getPropertyValue(name) { return this.props[name] || ''; }
}

class Element {
  constructor(tag, classes) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.style = new Style();
    this.names = new Set(classes || []);
    this.left = 0;
    this.right = 0;
    this.pointer = null;
    this.classList = {
      add: (name) => this.names.add(name),
      remove: (name) => this.names.delete(name),
      contains: (name) => this.names.has(name),
    };
  }
  getBoundingClientRect() {
    return { left: this.left, right: this.right, width: this.right - this.left };
  }
  append(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  // The two selectors the script asks for: the wrap a header's width is
  // written to, and the header itself.
  matches(selector) {
    if (selector === '.table-wrap') { return this.names.has('table-wrap'); }
    if (selector === 'thead th') {
      if (this.tagName !== 'TH') { return false; }
      for (let up = this.parentElement; up; up = up.parentElement) {
        if (up.tagName === 'THEAD') { return true; }
      }
      return false;
    }
    throw new Error('unexpected selector: ' + selector);
  }
  closest(selector) {
    if (this.matches(selector)) { return this; }
    return this.parentElement ? this.parentElement.closest(selector) : null;
  }
  setPointerCapture(pointer) { this.pointer = pointer; }
  releasePointerCapture(pointer) {
    if (this.pointer === pointer) { this.pointer = null; }
  }
  hasPointerCapture(pointer) { return this.pointer === pointer; }
}

const body = new Element('body');
body.names.add('body');
const heard = { document: [], window: [] };
const document = {
  body,
  addEventListener: (type, fn) => heard.document.push([type, fn]),
};
const window = {
  addEventListener: (type, fn) => heard.window.push([type, fn]),
};

// A real event runs up from the header it was aimed at to the document and
// on to the window, so both sets of listeners hear it.
const fire = (type, event) => {
  event.defaulted = false;
  event.stopped = false;
  event.preventDefault = () => { event.defaulted = true; };
  event.stopPropagation = () => { event.stopped = true; };
  for (const [name, fn] of heard.document.concat(heard.window)) {
    if (name === type) { fn(event); }
  }
  return event;
};

const press = (th, x, extra) =>
  fire('pointerdown', Object.assign(
    { target: th, clientX: x, button: 0, pointerId: 1, pointerType: 'mouse' },
    extra || {}
  ));
const drag = (x) =>
  fire('pointermove', { target: body, clientX: x, pointerId: 1, buttons: 1 });
const release = () => fire('pointerup', { target: body, pointerId: 1 });
const hover = (target, x) =>
  fire('pointermove', { target: target, clientX: x, pointerId: 1, buttons: 0 });

// The shape Gradio builds: a wrap holding the table it measures with, and a
// second wrap inside it holding the table on screen. Widths are written to
// the outer one by Gradio and have to be covered on the inner one.
const dataframe = (widths, rowNumbers) => {
  const outer = body.append(new Element('div', ['table-wrap']));
  const inner = outer.append(new Element('div', ['table-wrap']));
  const row = inner.append(new Element('table'))
    .append(new Element('thead'))
    .append(new Element('tr'));
  const columns = [];
  let edge = 0;
  if (rowNumbers) {
    const number = row.append(new Element('th', ['row-number']));
    number.left = 0;
    number.right = 40;
    edge = 40;
  }
  for (const width of widths) {
    const th = row.append(new Element('th'));
    th.left = edge;
    th.right = edge + width;
    edge = th.right;
    columns.push(th);
  }
  widths.forEach((width, i) => {
    outer.style.setProperty('--cell-width-' + i, width + 'px');
  });
  return { outer: outer, inner: inner, columns: columns };
};
"""


class ColumnResizeScriptTests(unittest.TestCase):
    """COLUMN_JS itself, run over a stand-in table.

    The script is the one part of the resizing that no Python call can
    reach, so these run the real string in node and let its own assertions
    report. A machine without node skips them.
    """

    def check(self, checks: str):
        script = f"{TABLE_PAGE}\nconst start = {app.COLUMN_JS};\n{checks}"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "columns.js"
            path.write_text(script)
            result = subprocess.run(
                ["node", str(path)], capture_output=True, text=True, timeout=60
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_dragged_column_widens_without_touching_gradios_own_widths(self):
        # The point of the whole thing: the text Gradio clipped comes back.
        # The width has to land on the inner wrap, because Gradio rewrites
        # the outer one every time the rows change - which on a filtered
        # table is with every keystroke - and a width the reader chose has
        # to outlive that.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
drag(420);
release();

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '300px',
  'the column the reader dragged is 100px wider'
);
assert.strictEqual(
  table.outer.style.getPropertyValue('--cell-width-1'), '200px',
  "Gradio's own measurement is left where it was"
);
assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-0'), '',
  'a column nobody dragged goes on being measured'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_seam_belongs_to_the_column_on_its_left(self):
        # One line divides two columns, and a pointer a few pixels either
        # side of it is at the same seam. Whichever header it lands on, the
        # column that moves is the one whose edge the line is.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

// Just inside the third header, which is the same line as the second
// header's right edge.
press(table.columns[2], 322);
drag(372);
release();

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '250px',
  'the column on the left of the seam is the one that moved'
);
assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-2'), '',
  'the column on the right was not resized'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_press_away_from_a_seam_is_left_to_the_table(self):
        # A header is a control in its own right: clicking it sorts the
        # table. Only the strip at its edge is a seam, and a press anywhere
        # else has to reach Gradio untouched.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

const event = press(table.columns[1], 220);
assert.strictEqual(event.defaulted, false, 'the press is not taken over');
drag(320);
assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '',
  'nothing moved'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_click_that_ends_a_drag_does_not_sort_the_table(self):
        # A release over a header is also a click on it, and a click on a
        # header sorts. Re-sorting the rows every time the reader let go of
        # a seam would be a table that reshuffles itself whenever it is
        # read.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
drag(420);
release();
const ended = fire('click', { target: table.columns[1] });
assert.strictEqual(ended.stopped, true, 'the drag\\'s own click is dropped');

// And only that one: the next click is a real click.
const next = fire('click', { target: table.columns[1] });
assert.strictEqual(next.stopped, false, 'a later click sorts as it always did');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_press_that_moved_nothing_leaves_the_click_alone(self):
        # A press on the seam that goes nowhere is a click on the header,
        # and the reader who meant to sort should get their sort.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
release();
const event = fire('click', { target: table.columns[1] });

assert.strictEqual(event.stopped, false, 'a still press is still a click');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_double_clicking_a_seam_gives_the_column_back_its_measured_width(self):
        # The way out. The width to fall back to is the one Gradio measured,
        # which only Gradio can say, so the column is uncovered rather than
        # handed a figure this script would have to guess at.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
drag(420);
release();
fire('dblclick', { target: table.columns[1], clientX: 320 });

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '',
  "the column is back on Gradio's measurement"
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_column_cannot_be_dragged_out_of_reach(self):
        # A column dragged to nothing would take its seam with it, and the
        # reader would have no edge left to drag it back by.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
drag(-400);
release();

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '48px',
  'the column stops at a width there is still something to grab'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_row_number_column_is_not_one_of_the_numbered_columns(self):
        # Gradio numbers --cell-width-0 from the first column of data and
        # gives the row numbers a width of their own, so counting headers
        # from the left would name the column next door.
        self.check(
            """
start();
const table = dataframe([120, 200, 90], true);

press(table.columns[0], 160);
drag(210);
release();

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-0'), '170px',
  'the first column of data is column zero'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_seam_under_the_pointer_shows_itself(self):
        # Nothing on a table says which of its edges can be dragged.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

hover(table.columns[1], 318);
assert.strictEqual(
  table.columns[1].names.has('column-seam'), true, 'the edge marks itself'
);

hover(table.columns[1], 220);
assert.strictEqual(
  table.columns[1].names.has('column-seam'), false,
  'and stops when the pointer moves off it'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_release_the_page_never_heard_ends_the_drag(self):
        # A pointer let go beyond the edge of the window leaves a column
        # following a pointer with nothing held down.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
drag(420);
fire('pointermove', { target: body, clientX: 500, pointerId: 1, buttons: 0 });
fire('pointermove', { target: body, clientX: 900, pointerId: 1, buttons: 1 });

assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '300px',
  'the column stayed where the drag left it'
);
assert.strictEqual(
  body.names.has('column-dragging'), false, 'and the page is done dragging'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_second_pointer_does_not_take_over_a_drag(self):
        # A drag belongs to the pointer that began it until that pointer
        # ends it.
        self.check(
            """
start();
const table = dataframe([120, 200, 90]);

press(table.columns[1], 320);
fire('pointermove', { target: body, clientX: 900, pointerId: 2, buttons: 1 });
assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '',
  'the other pointer is not dragging anything'
);

fire('pointerup', { target: body, pointerId: 2 });
drag(420);
assert.strictEqual(
  table.inner.style.getPropertyValue('--cell-width-1'), '300px',
  'and did not end the drag the first one started'
);
"""
        )


class ColumnResizeTests(unittest.TestCase):
    """That the script and its seam reach the pages with the tables on."""

    def test_the_script_runs_on_every_page(self):
        demo = app.build_app()

        self.assertTrue(any(fn.js == app.COLUMN_JS for fn in demo.fns.values()))

    def test_the_seam_is_drawn_where_the_script_marks_it(self):
        self.assertIn("th.column-seam", app.CSS)
        self.assertIn("cursor: col-resize", app.CSS)
        self.assertIn("'column-seam'", app.COLUMN_JS)
        # The line is inset rather than a border: a border would widen the
        # header and change the width the drag is measuring.
        seam = app.CSS[app.CSS.index("th.column-seam {") :]
        self.assertIn("box-shadow: inset", seam[: seam.index("}")])

    def test_the_cursor_holds_for_the_whole_drag(self):
        # A drag that leaves the header behind still has the cursor over
        # whatever it crosses, and a page that keeps selecting text under
        # the pointer selects the whole table on the way past.
        self.assertIn("body.column-dragging", app.CSS)
        self.assertIn("user-select: none", app.CSS)
        self.assertIn("'column-dragging'", app.COLUMN_JS)


if __name__ == "__main__":
    unittest.main()
