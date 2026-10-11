"""Writes judgment-run-exported.json: judgment-run.json reopened on the workbench and exported again.

Format marker: chatlab-computer-safety-run-1
Era: free-text judgments only, when an export recorded where its judgments came from as one flat
``imported_provenance`` object (later versions keep a list, newest first, and still read the flat object).
Generating commit: 5e35942 Add OSGuard computer-use safety benchmark workbench (#134)

Commands used:
    git worktree add --detach /tmp/chatlab-at-5e35942 5e35942
    cd /tmp/chatlab-at-5e35942 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/computer_safety/generators/judgment-run-exported.py \
        <repo>/tests/fixtures/saved/computer_safety/judgment-run.json \
        <repo>/tests/fixtures/saved/computer_safety/judgment-run-exported.json

The input is judgment-run.json, itself written at this commit. That commit's own page handlers open it
(load_cases, the "Import cases / saved run" button) and export it (export_run, "Export current results"); the
exported file is copied byte for byte. Opening drops each prediction's token metrics, marks the run
"imported", and records the run's model, mode, sampling, creation time and fingerprint as its provenance.
"""
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

sys.path.insert(1, "tests")

import gradio as gr  # noqa: E402

from extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector  # noqa: E402
from extensions.osguard import page as page_module  # noqa: E402
import test_computer_safety  # noqa: E402
from test_computer_safety import JudgmentManager  # noqa: E402


def main(source, output):
    page_module.uuid4 = lambda: uuid.UUID(int=0x5e35942 + 1)
    test_computer_safety.setUpModule()
    try:
        with tempfile.TemporaryDirectory() as directory:
            manager = JudgmentManager()
            context = ExtensionContext(ModelService(lambda: manager), TokenInspector(),
                                       Path(directory) / "extension", NavigationService(lambda *args: None))
            with gr.Blocks() as demo:
                page_module.build_page(context)
            functions = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
            loaded = functions["load_cases"](source, "owner")
            exported = functions["export_run"](loaded[1])
            demo.close()
            shutil.copyfile(exported, output)
    finally:
        test_computer_safety.tearDownModule()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
