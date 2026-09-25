"""``python -m chatlab``: serve ChatLab to a browser from a checkout."""

import os

from chatlab import api, branding, logs
from chatlab.app import build_app, current_manager
from chatlab.device_memory import watch_memory

# The same rules the desktop app runs under, so a problem reproduced from
# a checkout is recorded the way it was recorded on the machine that hit
# it. CHATLAB_LOG_LEVEL=debug turns the detail up without a code change.
logs.log_environment(logs.configure())
watch_memory()
conductor_port = os.environ.get("CONDUCTOR_PORT")
demo = build_app().queue(default_concurrency_limit=1)
# Gradio builds its FastAPI application inside launch(), so the API is
# added once the server is up rather than before it. prevent_thread_lock
# is what makes that possible: the thread is held afterwards instead.
demo.launch(
    inbrowser=conductor_port is None,
    server_port=int(conductor_port) if conductor_port else None,
    prevent_thread_lock=True,
    favicon_path=branding.favicon_path(),
)
api.attach(demo.app, current_manager)
demo.block_thread()
