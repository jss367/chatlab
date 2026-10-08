"""Real Gradio server with only the model replaced by a deterministic CPU fake."""

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

# Set these before importing ChatLab: never read or write the user's files.
folder = Path(sys.argv[1])
os.environ["CHATLAB_SETTINGS_PATH"] = str(folder / "settings.json")
os.environ["CHATLAB_LIBRARY_PATH"] = str(folder / "conversations.json")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from chatlab import app, text_generation  # noqa: E402
from chatlab.ui import runtime  # noqa: E402
from fakes import loaded_manager, FakeModel, PIECES, EOS_ID  # noqa: E402
from navigation_diagnostics import instrument, instrument_queue  # noqa: E402


class SlowModel(FakeModel):
    def forward(self, *args, **kwargs):
        # Long enough to observe streaming and act before natural completion.
        time.sleep(0.15)
        return super().forward(*args, **kwargs)


runtime.MANAGER = loaded_manager([0, 1] * 20 + [EOS_ID])
runtime.MANAGER.model = SlowModel([0, 1] * 20 + [EOS_ID], len(PIECES), EOS_ID)
text_generation.STREAM_BATCH_TOKENS = 1

demo = app.build_app().queue(default_concurrency_limit=1)
instrument(demo)
instrument_queue(demo)
_, url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)
# Publish only after the whole URL is written; the parent polls for existence.
ready = folder / "ready.tmp"
ready.write_text(url)
ready.replace(folder / "ready")
demo.block_thread()
