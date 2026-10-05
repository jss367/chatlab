"""Run ChatLab on another machine over SSH and show it in this window.

The desktop window is a web view of a Gradio server, so the server can run
anywhere the window can reach. A remote session starts ``python -m chatlab
--remote`` in a checkout on another host, forwards a local port to it, and
hands the window the forwarded address.

Two SSH connections do the work. The first runs the server and reads the
address it prints once it is up; the second carries the port forward. The
server watches its standard input and exits when it closes, which happens
when the first connection drops, this side ends it, or this process dies, so
a GPU is not left holding a model nobody can reach. The forward ends the
same way, so it never outlives the app holding the local port.

SSH runs in batch mode: keys, the agent and ``~/.ssh/config`` decide how it
signs in, and a host that would ask for a password or a new host key is
refused with SSH's own message. Connecting once from Terminal settles both.
"""

from __future__ import annotations

import collections
import logging
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener



logger = logging.getLogger(__name__)

DEFAULT_DIRECTORY = "~/chatlab"
# The line ``python -m chatlab --remote`` prints when the server is up. What
# follows it is the address the server is listening on, on the remote host.
READY_MARKER = "CHATLAB_READY "
# A first start on a cold machine imports torch and builds the whole page
# before it prints the marker, which can take a minute or two.
START_TIMEOUT = 300.0
TUNNEL_TIMEOUT = 30.0
STOP_TIMEOUT = 5.0
# How many lines of the server's output are kept for an error message.
TAIL_LINES = 12
# What the forward's connection runs on the host: nothing but a wait for its
# input to close.
TUNNEL_COMMAND = "cat > /dev/null"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "RemoteCommand=none",
    "-o", "ForkAfterAuthentication=no",
    # Session lifetime is owned here, rather than an existing user SSH master.
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ControlPersist=no",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
)


class RemoteError(Exception):
    """A remote session could not start, or was stopped while starting."""


@dataclass(frozen=True)
class RemoteTarget:
    """An SSH host and the ChatLab checkout on it, written ``host:path``.

    The host is anything ``ssh`` accepts, including an alias from
    ``~/.ssh/config``. An IPv6 literal would need its colons, so give one an
    alias instead.
    """

    host: str
    directory: str = DEFAULT_DIRECTORY

    @classmethod
    def parse(cls, text: str) -> RemoteTarget:
        host, _, directory = text.strip().partition(":")
        host = host.strip()
        directory = directory.strip() or DEFAULT_DIRECTORY
        if not host:
            raise RemoteError("Enter an SSH host, such as gpu-box or me@gpu-box:~/chatlab.")
        # A leading dash would reach ssh as an option rather than a host.
        if host.startswith("-") or any(character.isspace() for character in host):
            raise RemoteError(f"{host!r} is not an SSH host.")
        return cls(host, directory)

    def __str__(self) -> str:
        return f"{self.host}:{self.directory}"


def remote_command(directory: str) -> str:
    """The shell command that starts the server in ``directory`` on the host."""

    return f"cd {_shell_path(directory)} && exec .venv/bin/python -m chatlab --remote"


def _shell_path(path: str) -> str:
    # Quoted, except for a leading ~, which the remote shell has to expand.
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def load_target(path: Path) -> RemoteTarget | None:
    """The target last connected to, kept so the next prompt offers it."""

    try:
        return RemoteTarget.parse(path.read_text(encoding="utf-8"))
    except (OSError, RemoteError):
        return None


def save_target(path: Path, target: RemoteTarget) -> None:
    try:
        path.write_text(f"{target}\n", encoding="utf-8")
    except OSError as error:
        logger.warning("Could not remember the remote host: %s", error)


class RemoteSession:
    """One remote server and the forward that reaches it.

    ``start`` blocks until the window can load the forwarded address and
    returns it. ``close`` ends both connections and may be called from any
    thread, including while ``start`` is still waiting, which stops it.
    ``on_lost`` is called with a reason if either connection ends after a
    successful start without ``close`` having been asked for.
    """

    def __init__(
        self,
        target: RemoteTarget,
        *,
        local_port: int,
        on_lost: Callable[[RemoteSession, str], None] | None = None,
        ssh: str = "ssh",
        start_timeout: float = START_TIMEOUT,
        tunnel_timeout: float = TUNNEL_TIMEOUT,
    ) -> None:
        self.target = target
        self.local_port = local_port
        self.on_lost = on_lost
        self.ssh = ssh
        self.start_timeout = start_timeout
        self.tunnel_timeout = tunnel_timeout
        self.url: str | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._server_drained = threading.Event()
        self._remote_address: str | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._tunnel_tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._tunnel_drained = threading.Event()
        self._server: subprocess.Popen | None = None
        self._tunnel: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._tunnel_reader: threading.Thread | None = None
        self._probe = build_opener(ProxyHandler({}))

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def start(self) -> str:
        """Start the server, forward a port to it, and return the local URL."""

        try:
            remote_address = self._start_server()
            self._start_tunnel(remote_address)
            url = f"http://127.0.0.1:{self.local_port}/"
            self._wait_for(url)
            with self._lock:
                if self.closed:
                    raise RemoteError("The connection was cancelled.")
                self.url = url
        except BaseException:
            self.close()
            raise
        for process, name in ((self._server, "server"), (self._tunnel, "port forward")):
            threading.Thread(target=self._watch, args=(process, name), daemon=True).start()
        logger.info("Connected to ChatLab on %s at %s", self.target, url)
        return url

    def _spawn(self, name: str, arguments: list[str], **options) -> subprocess.Popen:
        # Stored under the lock that ``close`` takes, so a close that lands
        # mid-start either stops the spawn or finds the process to stop.
        with self._lock:
            if self.closed:
                raise RemoteError("The connection was cancelled.")
            try:
                # Its own session, so a signal meant for the app is not also
                # delivered to SSH, and closing is this class's decision.
                process = subprocess.Popen([self.ssh, *arguments], start_new_session=True, **options)
            except OSError as error:
                raise RemoteError(f"Could not run ssh: {error}") from error
            setattr(self, name, process)
            return process

    def _start_server(self) -> str:
        command = remote_command(self.target.directory)
        logger.info("Starting ChatLab on %s: %s", self.target.host, command)
        server = self._spawn(
            "_server",
            ["-T", *SSH_OPTIONS, "-o", "ClearAllForwardings=yes",
             "-o", "StdinNull=no", "-o", "SessionType=default", self.target.host, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Drained for as long as the server runs, since a full pipe would
        # stall it the next time it wrote, and logged here because this log
        # is the only place a reader of this app will look.
        with self._lock:
            if self.closed:
                raise RemoteError("The connection was cancelled.")
            self._reader = threading.Thread(target=self._read_server, args=(server,), daemon=True)
            self._reader.start()
        deadline = time.monotonic() + self.start_timeout
        while not self._ready.wait(0.1):
            if self.closed:
                raise RemoteError("The connection was cancelled.")
            if server.poll() is not None:
                raise RemoteError(self._failure(f"ChatLab on {self.target.host} stopped before it was ready."))
            if time.monotonic() > deadline:
                raise RemoteError(
                    self._failure(
                        f"ChatLab on {self.target.host} did not report that it was ready within "
                        f"{self.start_timeout:.0f} seconds."
                    )
                )
        return self._remote_address

    def _read_server(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                if line.startswith(READY_MARKER) and not self._ready.is_set():
                    self._remote_address = line[len(READY_MARKER):].strip()
                    self._ready.set()
                    continue
                with self._lock:
                    self._tail.append(line)
                logger.info("[%s] %s", self.target.host, line)
        finally:
            self._server_drained.set()

    def _start_tunnel(self, remote_address: str) -> None:
        try:
            parts = urlsplit(remote_address)
            port = parts.port
            host = parts.hostname or "127.0.0.1"
        except ValueError as error:
            raise RemoteError(f"ChatLab on {self.target.host} reported an invalid address: {remote_address}") from error
        if port is None:
            raise RemoteError(f"ChatLab on {self.target.host} reported an address without a port: {remote_address}")
        forward = f"127.0.0.1:{self.local_port}:{host}:{port}"
        logger.info("Forwarding %s to %s on %s", self.local_port, remote_address, self.target.host)
        # The forward runs a command that reads its input to the end, rather
        # than -N, so it ends the way the server does when this process dies
        # without closing it: the pipe closes, ``cat`` exits, and SSH with it.
        # An -N forward would be orphaned holding the local port.
        tunnel = self._spawn(
            "_tunnel",
            [
                "-T", *SSH_OPTIONS, "-o", "ClearAllForwardings=no",
                "-o", "StdinNull=no", "-o", "SessionType=default",
                "-o", "ExitOnForwardFailure=yes", "-L", forward,
                self.target.host, TUNNEL_COMMAND,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._lock:
            if self.closed:
                raise RemoteError("The connection was cancelled.")
            self._tunnel_reader = threading.Thread(target=self._read_tunnel, args=(tunnel,), daemon=True)
            self._tunnel_reader.start()

    def _read_tunnel(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stderr:
                line = line.rstrip("\r\n")
                with self._lock:
                    self._tunnel_tail.append(line)
                logger.info("[%s port forward] %s", self.target.host, line)
        finally:
            self._tunnel_drained.set()

    def _tunnel_failure(self, summary: str) -> str:
        # Once SSH exits, let the reader collect its last diagnostic lines.
        if self._tunnel_reader is not None:
            self._tunnel_drained.wait(STOP_TIMEOUT)
        with self._lock:
            tail = "\n".join(self._tunnel_tail).strip()
        return f"{summary}\n\n{tail}" if tail else summary

    def _wait_for(self, url: str) -> None:
        """Poll the API's status route through the forward until it answers."""

        from chatlab import api

        status_url = f"{url.rstrip('/')}{api.API_PREFIX}/chatlab/status"
        deadline = time.monotonic() + self.tunnel_timeout
        while True:
            if self.closed:
                raise RemoteError("The connection was cancelled.")
            if self._tunnel.poll() is not None:
                raise RemoteError(self._tunnel_failure(f"The port forward to {self.target.host} failed."))
            if self._server.poll() is not None:
                raise RemoteError(self._failure(f"ChatLab on {self.target.host} stopped."))
            try:
                with self._probe.open(status_url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (URLError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RemoteError(f"ChatLab on {self.target.host} could not be reached through the port forward.")
            time.sleep(0.25)

    def _failure(self, summary: str) -> str:
        if self._reader is not None and self._server.poll() is not None:
            self._server_drained.wait(STOP_TIMEOUT)
        with self._lock:
            lines = tuple(self._tail)
        tail = "\n".join(line for line in lines if line.strip())
        return f"{summary}\n\n{tail}" if tail else summary

    def _watch(self, process: subprocess.Popen, name: str) -> None:
        process.wait()
        summary = f"The {name} connection to {self.target.host} ended."
        reason = self._failure(summary) if process is self._server else self._tunnel_failure(summary)
        # Only the watcher whose close is the one that ends the session
        # reports it; the other connection ending behind it is that close.
        if self._close() and self.on_lost is not None:
            logger.warning("%s", reason)
            self.on_lost(self, reason)

    def close(self) -> None:
        """End both connections; the server exits when its input closes."""

        self._close()

    def _close(self) -> bool:
        """Close once; concurrent callers wait until the owned processes are gone."""
        with self._lock:
            already_closed = self.closed
            if not already_closed:
                self._closed.set()
                server, tunnel = self._server, self._tunnel
        if already_closed:
            self._shutdown.wait()
            return False
        try:
            self._stop_processes(server, tunnel)
        finally:
            self._shutdown.set()
        return True

    def _stop_processes(self, server, tunnel) -> None:
        logger.info("Disconnecting from %s", self.target.host)
        for process in (server, tunnel):
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        for process in (tunnel, server):
            if process is None:
                continue
            if process is server:
                # Given a moment to exit on its own, which it does once the
                # remote end has seen its input close and stopped.
                try:
                    process.wait(STOP_TIMEOUT)
                    continue
                except subprocess.TimeoutExpired:
                    pass
            process.terminate()
            try:
                process.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        # The reader stops at the end of the server's output, which has come
        # now that SSH has exited; the pipes are closed once it has.
        if server is not None:
            reader = self._reader
            if reader is not None:
                reader.join(STOP_TIMEOUT)
            if reader is None or not reader.is_alive():
                server.stdout.close()
        if tunnel is not None:
            reader = self._tunnel_reader
            if reader is not None:
                reader.join(STOP_TIMEOUT)
            if reader is None or not reader.is_alive():
                tunnel.stderr.close()


def exit_when_stdin_closes(stop: Callable[[], None], stream=None) -> threading.Thread:
    """Call ``stop`` once standard input reaches its end.

    The server half of a remote session. The desktop app holds the other end
    of this input open through SSH for as long as the session lasts, so the
    end of it means the app disconnected, quit, or lost the connection.
    """

    def watch() -> None:
        source = stream if stream is not None else sys.stdin.buffer
        try:
            while source.read(4096):
                pass
        except (OSError, ValueError):
            pass
        logger.info("Standard input closed; stopping the remote server")
        stop()

    thread = threading.Thread(target=watch, name="chatlab-stdin-watch", daemon=True)
    thread.start()
    return thread
