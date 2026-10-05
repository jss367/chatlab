"""Stands in for ``python -m chatlab --remote`` in the remote session tests.

It answers the status route the desktop app polls, prints the ready line
the real server prints, and stops when its input closes, which is all a
remote session asks of the far end. ``FAKE_REMOTE_PIDFILE`` is where it
writes its process ID, so a test can check it stopped or stop it itself.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chatlab import api, remote


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"path": self.path}).encode()
        self.send_response(200 if self.path in ("/", f"{api.API_PREFIX}/chatlab/status") else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    pidfile = os.environ.get("FAKE_REMOTE_PIDFILE")
    if pidfile:
        with open(pidfile, "w") as handle:
            handle.write(str(os.getpid()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print("fake ChatLab starting", flush=True)
    if os.environ.get("FAKE_REMOTE_FAIL"):
        print(os.environ["FAKE_REMOTE_FAIL"], file=sys.stderr, flush=True)
        sys.exit(1)
    remote.exit_when_stdin_closes(lambda: os._exit(0))
    print(f"{remote.READY_MARKER}http://127.0.0.1:{server.server_address[1]}/", flush=True)
    server.serve_forever()
