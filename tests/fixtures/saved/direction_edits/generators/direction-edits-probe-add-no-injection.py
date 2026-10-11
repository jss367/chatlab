"""Write direction-edits-probe-add-no-injection.json, a result downloaded from the Edits page of 390252b.

Format marker: chatlab-direction-edits-1
Era: the only one; the format arrived with the Direction edits extension.
Generating commit: 390252b "Add a Direction edits extension (#196)".

Commands:
    git worktree add --detach /tmp/chatlab-at-390252b 390252b
    cd /tmp/chatlab-at-390252b && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

The sparse branches of the result: the page's own ``open_directions`` opens
the committed probe ../../probes/probe-era3-precision.json as the directions
(a probe file as the source), nothing is injected, the edit adds 2.5 along
the direction at blocks 2-3, the random control is on, and there is no lens
readout (target and lens null) and no differences. ``run_experiment`` runs
two conditions on that commit's tiny test decoder (its tests'
``tiny_manager``). The bytes are the download the page staged, copied
unchanged. The creation time and the load's revision are pinned; the
revision matches the probe's, as a load of those weights would.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.direction_edits import experiment
from chatlab.extensions.direction_edits.page import build_page

assert Path(experiment.__file__).resolve().is_relative_to(Path.cwd().resolve()), experiment.__file__
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_direction_edits import tiny_manager  # noqa: E402
from ui_support import handlers_by_name  # noqa: E402

SAVED = Path(__file__).resolve().parent.parent.parent
REVISION = "fedcba9876543210fedcba9876543210fedcba98"
experiment.time = SimpleNamespace(time=lambda: 1791464400.125)

work = Path(tempfile.mkdtemp(prefix="chatlab-direction-edits-fixture-"))
manager = tiny_manager()
manager.model_revision = lambda: REVISION
context = ExtensionContext(ModelService(lambda: manager), TokenInspector(), work / "data",
                           NavigationService(lambda *args: None))
with gr.Blocks() as demo:
    build_page(context)
fn = handlers_by_name(demo)
directions, _ = fn["open_directions"](str(SAVED / "probes" / "probe-era3-precision.json"))
passage = "a quick brown fox jumps over the lazy dog"
controls = [passage, directions, None, 1.0, 1, 64, experiment.ADD, 2.5, 2, 3, 4, 9, True, 11,
            "", 1, 64, 0, 0, ""]
rows = ["plain", "", "greeting", "hello world, "] + ["", ""] * 4
frames = list(fn["run_experiment"]("owner", *controls, *rows))
status, result, *_, download = frames[-1]
assert status == "Done.", status
shutil.copyfile(download, sys.argv[1])
demo.close()
print(sys.argv[1])
