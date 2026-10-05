"""``python -m chatlab``: serve ChatLab to a browser from a checkout."""

import argparse
import logging
import os

parser = argparse.ArgumentParser(prog="python -m chatlab", description="Serve ChatLab to a browser.")
parser.add_argument(
    "--remote",
    action="store_true",
    help=(
        "run as the far end of the desktop app's remote session: serve on this host's loopback "
        "address, open no browser, print the address once it is up, and stop when standard input closes"
    ),
)
args = parser.parse_args()

if args.remote:
    # Arm lifetime control before cold imports, page building, or launch.
    from chatlab import remote

    def stop() -> None:
        # Exits from the watching thread without waiting on the others: a
        # generation or a load in flight belongs to a window that is gone.
        logging.shutdown()
        os._exit(0)

    remote.exit_when_stdin_closes(stop)

from chatlab import api, branding, logs  # noqa: E402 - remote EOF must be watched before cold imports
from chatlab.app import build_app, current_manager  # noqa: E402
from chatlab.device_memory import watch_memory  # noqa: E402


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
_, local_url, _ = demo.launch(
    inbrowser=conductor_port is None and not args.remote,
    server_port=int(conductor_port) if conductor_port else None,
    # A remote server is reached through an SSH forward to its loopback
    # address, never over the network, whatever GRADIO_SERVER_NAME says.
    server_name="127.0.0.1" if args.remote else None,
    prevent_thread_lock=True,
    favicon_path=branding.favicon_path(),
)
api.attach(demo.app, current_manager)
if args.remote:

    print(f"{remote.READY_MARKER}{local_url}", flush=True)
demo.block_thread()
