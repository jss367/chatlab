"""Generate conversation/01-first-890d94a.json.

Format: chatlab-conversation-1, first era: a system prompt and turns of role,
content and reasoning, nothing else.
Generating commit: 890d94a "Add stop, reasoning blocks, system prompt, seed,
and conversation editing", which introduced the Save/Load conversation file.

Commands used:

    git worktree add --detach /tmp/chatlab-at-890d94a 890d94a
    cd /tmp/chatlab-at-890d94a && PYTHONPATH=. \
        /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/conversation/generators/01_first_890d94a.py \
        <repo>/tests/fixtures/saved/conversation/01-first-890d94a.json

Written as that commit's ``save_conversation`` wrote it:
``path.write_text(to_json(turns, system_prompt=...), encoding="utf-8")``.
"""

import sys
from pathlib import Path

from conversation import SAVE_FORMAT, to_json  # top-level module at this commit

assert SAVE_FORMAT == "chatlab-conversation-1"
target = Path(sys.argv[1]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)

turns = [
    {"role": "user", "content": "Explain recursion in one sentence.", "reasoning": ""},
    {"role": "assistant", "content": "Recursion is when a function solves a problem by calling itself on a smaller piece of it.",
     "reasoning": "Keep it to one sentence.\nMention the base idea of smaller subproblems."},
    {"role": "user", "content": "Give an example in Python, with <code> tags.", "reasoning": ""},
    {"role": "assistant", "content": "<code>def f(n): return 1 if n == 0 else n * f(n - 1)</code>", "reasoning": ""},
    # The reader wrote a follow-up and saved before the model answered.
    {"role": "user", "content": "Now in Haskell — with “quotes”.", "reasoning": ""},
]
system_prompt = "You are a patient teacher.\nKeep answers short."

target.write_text(to_json(turns, system_prompt=system_prompt), encoding="utf-8")
print(target)
