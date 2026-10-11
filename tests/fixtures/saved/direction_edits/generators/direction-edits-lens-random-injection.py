"""Write direction-edits-lens-random-injection.json, a result downloaded from the Edits page of 390252b.

Format marker: chatlab-direction-edits-1
Era: the only one; the format arrived with the Direction edits extension.
Generating commit: 390252b "Add a Direction edits extension (#196)".

Commands:
    git worktree add --detach /tmp/chatlab-at-390252b 390252b
    cd /tmp/chatlab-at-390252b && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

Every option on: the page's own handlers open ../directions-sparse-unnormalized.json
(blocks 0-2, block 3 blank) and ../steering-tiny-decoder-layer0.json beside
this generator, and ``run_experiment`` runs three conditions on that commit's
tiny test decoder (its tests' ``tiny_manager``, with the Jacobian lens its
LensFixture imports) with an injection, an erase at block 1, the random
control, a lens readout and two differences. The bytes are the download the
page staged, copied unchanged. The creation time and the load's revision are
pinned.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

from chatlab.extension_api import ExtensionContext, ModelService, NavigationService, TokenInspector
from chatlab.extensions.direction_edits import experiment
from chatlab.extensions.direction_edits.page import build_page

assert Path(experiment.__file__).resolve().is_relative_to(Path.cwd().resolve()), experiment.__file__
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_direction_edits import BLOCKS, LENS_BLOCKS, MODEL, WIDTH, tiny_manager  # noqa: E402
from ui_support import handlers_by_name  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
REVISION = "fedcba9876543210fedcba9876543210fedcba98"
experiment.time = SimpleNamespace(time=lambda: 1791460800.5)

work = Path(tempfile.mkdtemp(prefix="chatlab-direction-edits-fixture-"))
manager = tiny_manager()
manager.model_revision = lambda: REVISION
torch.manual_seed(3)
matrices = {layer: torch.randn(WIDTH, WIDTH) for layer in LENS_BLOCKS}
torch.save({"J": matrices, "source_layers": list(LENS_BLOCKS), "n_prompts": 10, "d_model": WIDTH}, work / "lens.pt")
manager.import_jacobian_lens(str(work / "lens.pt"), MODEL)
assert BLOCKS == 4

context = ExtensionContext(ModelService(lambda: manager), TokenInspector(), work / "data",
                           NavigationService(lambda *args: None))
with gr.Blocks() as demo:
    build_page(context)
fn = handlers_by_name(demo)
directions, _ = fn["open_directions"](str(HERE / "directions-sparse-unnormalized.json"))
vector, _, _ = fn["open_vector"](str(HERE / "steering-tiny-decoder-layer0.json"))
passage = "the cat sat on the mat and the dog sat on the log"
controls = [passage, directions, vector, 80.0, 2, 6, experiment.ERASE, 0.0, 1, 1, 3, 5, True, 7,
            " sat", 1, 8, 1, 3, "suppression = neutral - ignore\nfocus = focus - neutral"]
rows = ["neutral", "", "ignore", "Do not think about the dog.\n", "focus", "the dog\n"] + ["", ""] * 3
frames = list(fn["run_experiment"]("owner", *controls, *rows))
status, result, *_, download = frames[-1]
assert status == "Done.", status
shutil.copyfile(download, sys.argv[1])
demo.close()
print(sys.argv[1])
