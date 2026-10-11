"""Write steering-tiny-decoder-layer0.json, the steering vector the Direction edits fixtures inject.

Format marker: chatlab-steering-1 (as an input to Direction edits)
Era: the steering vector file as Chat's Extract a vector wrote it when
Direction edits arrived.
Generating commit: 390252b "Add a Direction edits extension (#196)".

Commands:
    git worktree add --detach /tmp/chatlab-at-390252b 390252b
    cd /tmp/chatlab-at-390252b && PYTHONPATH=. <python> <abs path to this generator> <abs output path>

Chat's own extraction runs on that commit's tiny test decoder (the
``tiny_manager`` of its tests/test_direction_edits.py, ``test/tiny-decoder``,
4 blocks of 16): ``chatlab.ui.steering.extract_vector`` reads the examples,
and ``download_extracted`` writes block 0's direction, whose bytes are copied
here unchanged.
"""
import shutil
import sys
from pathlib import Path

from chatlab.ui import runtime, steering as steering_ui

assert Path(steering_ui.__file__).resolve().is_relative_to(Path.cwd().resolve()), steering_ui.__file__
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_direction_edits import tiny_manager  # noqa: E402

runtime.MANAGER = tiny_manager()
extraction = steering_ui.extract_vector("the dog sat on the log\nthe lazy dog\na dog",
                                        "the cat sat on the mat\nhello world\ntokens, merges and bytes",
                                        False, "mean")[0]
assert extraction and extraction["model_id"] == "test/tiny-decoder", extraction
written = steering_ui.download_extracted(extraction, 0)
shutil.copyfile(written, sys.argv[1])
print(sys.argv[1])
