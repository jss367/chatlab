"""Run the row script with inert DOM objects and no browser."""
import json
import shutil
import subprocess
import unittest

from chatlab.ui.common import FAMILY_OPEN_CLASS_PREFIX
from chatlab.ui.conversation_rows import CONVERSATION_ROWS_JS


@unittest.skipUnless(shutil.which("node"), "needs node for inert row script")
class ConversationRowsTests(unittest.TestCase):
    def test_single_active_fork_uses_view_state_and_unread_forks_stay_indented(self):
        script = r"""
const assert = require('node:assert/strict');
global.window = {};
const classList = () => {
  const values = new Set();
  return {contains:x=>values.has(x), add:x=>values.add(x),
    toggle:(x,on)=>on?values.add(x):values.delete(x)};
};
const element = () => ({dataset:{},classList:classList(),attributes:{},
  setAttribute(key,value){this.attributes[key]=value;},
  append(){}, remove(){ if(this.parent)this.parent.children=this.parent.children.filter(x=>x!==this); }});
function row(name,text) {
  const label=element(); label.children=[]; label.input={value:name}; label.span={textContent:text};
  label.append=function(child){child.parent=this;this.children.push(child);};
  label.querySelector=function(selector){
    if(selector==='input[type=radio]')return this.input;
    if(selector.includes('span:not'))return this.span;
    if(selector.includes('.conversation-family'))return this.children.find(c=>c.className.includes('conversation-family'));
    if(selector.includes('.conversation-actions'))return this.children.find(c=>c.className==='conversation-actions');
  };
  return label;
}
const head=row('Main','Main\n110 tokens · 1 fork');
const fork=row('Fork 1','↳ Fork 1 · 101 tokens');
const list={classList:classList(),querySelectorAll:()=>[head,fork]};
let sync, click;
global.MutationObserver=class {constructor(fn){sync=fn;}observe(){}};
global.requestAnimationFrame=fn=>fn();
global.HTMLInputElement=class {set value(text){this.sent=JSON.parse(text);}};
const bridge=new HTMLInputElement();bridge.tagName='INPUT';bridge.dispatchEvent=()=>{};
global.document={body:{},getElementById:()=>list,querySelector:()=>bridge,
  createElement:()=>element(),addEventListener:(name,fn)=>{click=fn;}};
START();
const toggle=()=>head.querySelector(':scope > .conversation-family');
const press=()=>click({target:{closest:selector=>selector.includes('conversation-family')?toggle():null},
  preventDefault(){},stopPropagation(){}});
assert.equal(toggle().attributes['aria-expanded'],'false');
press(); assert.equal(bridge.sent.open,true);
list.classList.add(__PREFIX__+Buffer.from('Main','utf8').toString('hex'));sync();
assert.equal(toggle().attributes['aria-expanded'],'true');
press();assert.equal(bridge.sent.open,false);
fork.span.textContent='● ↳ Fork 1 · 101 tokens';sync();
assert.equal(fork.classList.contains('conversation-fork'),true);
assert.equal(fork.classList.contains('conversation-head'),false);
"""
        script = script.replace("START()", f"({CONVERSATION_ROWS_JS})()")
        script = script.replace("__PREFIX__", json.dumps(FAMILY_OPEN_CLASS_PREFIX))
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
