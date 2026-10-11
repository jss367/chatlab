"""Write a saved contrast-mode Circuits attribution graph, and its menu label, with the released writer.

Format marker: chatlab-attribution-graph-1 (graphs/<id>.json) plus the menu label beside it (graphs/.metadata/<id>.json)
Era: released (same writer as graph-tokens-released; see that generator). This file explains pivot tokens against
others (contrast mode, one contrast target) from a raw prompt without a chat template, and measures its groups with
automatically chosen alternatives, boosting at every position.
Generating commit: f01bf05 "Merge pull request #177 from jss367/claude/attribution-graphs".

Commands:
    git worktree add --detach /tmp/chatlab-at-f01bf05 f01bf05
    cd /tmp/chatlab-at-f01bf05 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/circuits/generators/graph-contrast-released.py \
        <repo>/tests/fixtures/saved/circuits/graph-contrast-released
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("graph_tokens_released",
                                              Path(__file__).with_name("graph-tokens-released.py"))
released = importlib.util.module_from_spec(spec)
spec.loader.exec_module(released)

if __name__ == "__main__":
    released.main(sys.argv[1], "contrast")
