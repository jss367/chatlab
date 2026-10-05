"""Tests for running ChatLab on another machine over SSH.

``fake_ssh.py`` stands in for ``ssh``: it runs the remote command here and
forwards ports itself, so a session goes through every step it takes
against a real host. ``fake_remote_chatlab.py`` stands in for the server
at the far end, except in the one test that starts the real one.
"""

from __future__ import annotations

import io
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from urllib.request import urlopen

from chatlab import api, library, logs, remote, settings
from chatlab.desktop_launcher import RemoteConnection, WINDOW_TITLE, find_available_port


TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _gone(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class TargetTests(unittest.TestCase):
    def test_a_host_alone_runs_the_checkout_in_the_home_directory(self):
        self.assertEqual(remote.RemoteTarget.parse(" gpu-box \n"), remote.RemoteTarget("gpu-box", "~/chatlab"))

    def test_a_path_after_a_colon_names_the_checkout(self):
        target = remote.RemoteTarget.parse("me@gpu-box:/srv/chat lab")

        self.assertEqual(target, remote.RemoteTarget("me@gpu-box", "/srv/chat lab"))
        self.assertEqual(remote.RemoteTarget.parse(str(target)), target)

    def test_nothing_entered_is_refused(self):
        for text in ("", "   ", ":~/chatlab"):
            with self.subTest(text=text), self.assertRaises(remote.RemoteError):
                remote.RemoteTarget.parse(text)

    def test_a_host_that_ssh_would_read_as_an_option_is_refused(self):
        for text in ("-oProxyCommand=touch /tmp/x", "gpu box"):
            with self.subTest(text=text), self.assertRaises(remote.RemoteError):
                remote.RemoteTarget.parse(text)


class CommandTests(unittest.TestCase):
    def test_the_home_directory_is_left_for_the_remote_shell_to_expand(self):
        self.assertEqual(
            remote.remote_command("~/chatlab"), "cd ~/chatlab && exec .venv/bin/python -m chatlab --remote"
        )
        self.assertEqual(remote.remote_command("~"), "cd ~ && exec .venv/bin/python -m chatlab --remote")

    def test_the_rest_of_the_path_is_quoted(self):
        self.assertTrue(remote.remote_command("~/my lab").startswith("cd ~/'my lab' && "))
        self.assertTrue(remote.remote_command("/srv/a;b").startswith("cd '/srv/a;b' && "))


class SSHConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH is unavailable")
    def test_server_suppresses_inherited_forwards_and_tunnel_owns_them_once(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            config.write_text("Host fixture-gpu\n  HostName 127.0.0.1\n  LocalForward 18080 127.0.0.1:8080\n  DynamicForward 18081\n  ControlMaster auto\n  ControlPath /tmp/chatlab-fixture-master\n  ControlPersist 60\n  RemoteCommand tmux attach\n  ClearAllForwardings yes\n  StdinNull yes\n  ForkAfterAuthentication yes\n  SessionType none\n")
            session = remote.RemoteSession(remote.RemoteTarget("fixture-gpu"), local_port=18082)
            captured = []
            def spawn(name, arguments, **options):
                captured.append(arguments)
                raise remote.RemoteError("capture only")
            with mock.patch.object(session, "_spawn", side_effect=spawn):
                with self.assertRaises(remote.RemoteError):
                    session._start_server()
                with self.assertRaises(remote.RemoteError):
                    session._start_tunnel("http://127.0.0.1:8123/")
            # -G only expands this controlled fixture config: it makes no connection.
            server = subprocess.check_output(["ssh", "-G", "-F", str(config), *captured[0]], text=True).splitlines()
            tunnel = subprocess.check_output(["ssh", "-G", "-F", str(config), *captured[1]], text=True).splitlines()
            self.assertFalse(any(line.startswith(("localforward ", "dynamicforward ")) for line in server))
            self.assertIn("clearallforwardings yes", server)
            self.assertIn("stdinnull no", server)
            self.assertIn("sessiontype default", server)
            self.assertIn("clearallforwardings no", tunnel)
            self.assertIn("sessiontype none", tunnel)
            self.assertEqual(len([line for line in tunnel if line.startswith("localforward ")]), 2)
            self.assertEqual(len([line for line in tunnel if line.startswith("dynamicforward ")]), 1)
            self.assertTrue(any("18082" in line and "8123" in line for line in tunnel if line.startswith("localforward ")))
            for lines in (server, tunnel):
                self.assertIn("controlmaster false", lines)
                self.assertIn("controlpersist no", lines)
                self.assertIn("forkafterauthentication no", lines)
                self.assertFalse(any(line.startswith("remotecommand ") for line in lines))


class SavedTargetTests(unittest.TestCase):
    def test_the_last_target_is_offered_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remote-host"
            self.assertIsNone(remote.load_target(path))

            remote.save_target(path, remote.RemoteTarget("gpu-box", "/srv/chatlab"))

            self.assertEqual(remote.load_target(path), remote.RemoteTarget("gpu-box", "/srv/chatlab"))


class StdinWatchTests(unittest.TestCase):
    def test_the_end_of_input_stops_the_server(self):
        read_end, write_end = os.pipe()
        stopped = threading.Event()
        with os.fdopen(read_end, "rb") as stream:
            remote.exit_when_stdin_closes(stopped.set, stream)
            os.write(write_end, b"anything the app sends is ignored\n")
            self.assertFalse(stopped.wait(0.2))

            os.close(write_end)

            self.assertTrue(stopped.wait(5))


class SessionConcurrencyTests(unittest.TestCase):
    def test_concurrent_close_waits_for_owned_transport_cleanup(self):
        session = remote.RemoteSession(remote.RemoteTarget("gpu-box"), local_port=18082)
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def wait(timeout=None):
            entered.set()
            if not release.wait(3):
                raise AssertionError("cleanup was not released")
        tunnel = mock.Mock(stderr=io.StringIO(), wait=wait)
        session._tunnel = tunnel
        first = threading.Thread(target=session.close)
        second = threading.Thread(target=lambda: (session.close(), done.set()))
        first.start()
        try:
            self.assertTrue(entered.wait(1))
            second.start()
            self.assertFalse(done.wait(0.1))
        finally:
            release.set()
            first.join(3)
            second.join(3)
        self.assertTrue(done.is_set())
        tunnel.terminate.assert_called_once()
        self.assertTrue(tunnel.stderr.closed)

    def test_failure_formats_a_snapshot_before_other_lines_arrive(self):
        session = remote.RemoteSession(remote.RemoteTarget("gpu-box"), local_port=18082)
        class Line(str):
            def strip(self):
                with session._lock:
                    session._tail.append("later output")
                return super().strip()
        session._tail.append(Line("original output"))
        self.assertEqual(session._failure("Stopped"), "Stopped\n\noriginal output")

    def test_exit_failure_waits_for_buffered_server_diagnostics(self):
        session = remote.RemoteSession(remote.RemoteTarget("gpu-box"), local_port=18082)
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def lines():
            entered.set()
            if not release.wait(3):
                raise AssertionError("diagnostics were not released")
            yield "Permission denied (publickey).\n"
        session._server = mock.Mock(stdout=lines(), poll=lambda: 255)
        session._reader = threading.Thread(target=session._read_server, args=(session._server,))
        messages = []
        report = threading.Thread(target=lambda: (messages.append(session._failure("Stopped")), done.set()))
        session._reader.start()
        try:
            self.assertTrue(entered.wait(1))
            report.start()
            self.assertFalse(done.wait(0.1))
        finally:
            release.set()
            session._reader.join(3)
            report.join(3)
        self.assertTrue(done.is_set())
        self.assertEqual(messages, ["Stopped\n\nPermission denied (publickey)."])

    def test_invalid_ready_ports_are_reported_as_remote_errors(self):
        session = remote.RemoteSession(remote.RemoteTarget("gpu-box"), local_port=18082)
        for address in ("http://127.0.0.1:bad/", "http://127.0.0.1:65536/", "http://[broken:123/"):
            with self.subTest(address=address), mock.patch.object(session, "_spawn") as spawn:
                with self.assertRaisesRegex(remote.RemoteError, "reported an invalid address"):
                    session._start_tunnel(address)
                spawn.assert_not_called()


class SessionTests(unittest.TestCase):
    """A session against the fake host, from the first connection to the last."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.ssh = _executable(self.root / "ssh", f'exec "{sys.executable}" "{TESTS / "fake_ssh.py"}" "$@"')
        self.checkout = self.root / "chatlab"
        _executable(
            self.checkout / ".venv" / "bin" / "python",
            f'exec "{sys.executable}" "{TESTS / "fake_remote_chatlab.py"}"',
        )
        self.pidfile = self.root / "server.pid"
        self._environment = {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "FAKE_REMOTE_PIDFILE": os.environ.get("FAKE_REMOTE_PIDFILE"),
        }
        os.environ["PYTHONPATH"] = str(REPO)
        os.environ["FAKE_REMOTE_PIDFILE"] = str(self.pidfile)
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
        for name in ("FAKE_SSH_REFUSE", "FAKE_REMOTE_FAIL"):
            os.environ.pop(name, None)
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._directory.cleanup()

    def session(self, directory=None, **options) -> remote.RemoteSession:
        target = remote.RemoteTarget("gpu-box", str(directory or self.checkout))
        session = remote.RemoteSession(target, local_port=find_available_port(), ssh=str(self.ssh), **options)
        self.sessions.append(session)
        return session

    def server_pid(self) -> int:
        return int(self.pidfile.read_text())

    def test_the_window_reaches_the_remote_server_through_the_forward(self):
        session = self.session()

        url = session.start()

        self.assertEqual(url, f"http://127.0.0.1:{session.local_port}/")
        with urlopen(f"{url.rstrip('/')}{api.API_PREFIX}/chatlab/status", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["path"], f"{api.API_PREFIX}/chatlab/status")

    def test_readiness_uses_the_tunnel_even_with_environment_http_proxies(self):
        with mock.patch.dict(os.environ, {
            "HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
            "NO_PROXY": "", "no_proxy": "",
        }), mock.patch("urllib.request._opener", None), mock.patch("urllib.request.proxy_bypass", return_value=False):
            session = self.session(tunnel_timeout=2)
            try:
                self.assertEqual(session.start(), f"http://127.0.0.1:{session.local_port}/")
            finally:
                session.close()

    def test_disconnecting_stops_the_remote_server(self):
        session = self.session()
        session.start()
        pid = self.server_pid()

        session.close()

        self.assertTrue(_gone(pid), "the remote server outlived its session")
        with self.assertRaises(OSError):
            urlopen(f"http://127.0.0.1:{session.local_port}/", timeout=2)

    def test_verbose_tunnel_diagnostics_do_not_fill_the_pipe_or_block_forwarding(self):
        session = self.session(tunnel_timeout=3)
        with mock.patch.dict(os.environ, {"FAKE_SSH_VERBOSE_TUNNEL": "1"}):
            try:
                url = session.start()
                with urlopen(f"{url.rstrip('/')}{api.API_PREFIX}/chatlab/status", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                deadline = time.monotonic() + 3
                while not session._tunnel_tail or "diagnostic 1023:" not in session._tunnel_tail[-1]:
                    if time.monotonic() > deadline:
                        self.fail("the tunnel diagnostics were not drained")
                    time.sleep(0.01)
                self.assertEqual(len(session._tunnel_tail), remote.TAIL_LINES)
                self.assertIn("diagnostic 1023:", session._tunnel_tail[-1])
            finally:
                session.close()
        self.assertFalse(session._tunnel_reader.is_alive())

    def test_a_server_that_dies_is_reported_once_and_ends_the_session(self):
        lost = []
        reported = threading.Event()

        def on_lost(session, reason):
            lost.append(reason)
            reported.set()

        session = self.session(on_lost=on_lost)
        session.start()

        os.kill(self.server_pid(), signal.SIGKILL)

        self.assertTrue(reported.wait(15))
        time.sleep(0.5)
        self.assertEqual(len(lost), 1)
        self.assertIn("gpu-box", lost[0])
        self.assertTrue(session.closed)

    def test_a_checkout_that_is_not_there_says_what_the_shell_said(self):
        session = self.session(directory=self.root / "missing")

        with self.assertRaises(remote.RemoteError) as caught:
            session.start()

        self.assertIn("stopped before it was ready", str(caught.exception))
        self.assertIn("missing", str(caught.exception))

    def test_a_server_that_fails_to_start_passes_on_its_output(self):
        os.environ["FAKE_REMOTE_FAIL"] = "CUDA error: no kernel image is available"
        session = self.session()

        with self.assertRaises(remote.RemoteError) as caught:
            session.start()

        self.assertIn("CUDA error: no kernel image is available", str(caught.exception))

    def test_a_host_ssh_cannot_sign_in_to_passes_on_ssh_s_message(self):
        os.environ["FAKE_SSH_REFUSE"] = "Host key verification failed."
        session = self.session()

        with self.assertRaises(remote.RemoteError) as caught:
            session.start()

        self.assertIn("Host key verification failed.", str(caught.exception))

    def test_a_forward_that_cannot_listen_fails_and_stops_the_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            target = remote.RemoteTarget("gpu-box", str(self.checkout))
            session = remote.RemoteSession(target, local_port=held.getsockname()[1], ssh=str(self.ssh))
            self.sessions.append(session)

            with self.assertRaises(remote.RemoteError) as caught:
                session.start()

        self.assertIn("port forward", str(caught.exception))
        self.assertIn("Could not request local forwarding", str(caught.exception))
        self.assertTrue(_gone(self.server_pid()))

    def test_a_server_that_never_reports_ready_times_out(self):
        session = self.session(start_timeout=1)
        _executable(
            self.checkout / ".venv" / "bin" / "python",
            f'echo $$ > "{self.pidfile}"; exec cat > /dev/null',
        )

        with self.assertRaises(remote.RemoteError) as caught:
            session.start()

        self.assertIn("did not report that it was ready", str(caught.exception))
        self.assertTrue(_gone(self.server_pid()))

    def test_closing_while_starting_stops_the_start(self):
        _executable(
            self.checkout / ".venv" / "bin" / "python",
            f'echo $$ > "{self.pidfile}"; exec cat > /dev/null',
        )
        session = self.session()
        failures = []
        thread = threading.Thread(target=lambda: self._start_expecting_failure(session, failures))
        thread.start()
        deadline = time.monotonic() + 10
        while not self.pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)

        session.close()
        thread.join(10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["The connection was cancelled."])
        self.assertTrue(_gone(self.server_pid()))

    @staticmethod
    def _start_expecting_failure(session, failures):
        try:
            session.start()
        except remote.RemoteError as error:
            failures.append(str(error))


class RealServerTests(unittest.TestCase):
    """``python -m chatlab --remote``: the far end the desktop app expects."""

    def test_eof_stops_a_remote_server_during_cold_build_or_launch(self):
        script = """
import runpy, sys, threading, types
stage = sys.argv[1]
def hang():
    print("CHATLAB_COLD_START", flush=True)
    threading.Event().wait()
class Demo:
    def queue(self, **kwargs):
        return self
    def launch(self, **kwargs):
        hang()
app = types.ModuleType("chatlab.app")
app.build_app = hang if stage == "build" else Demo
app.current_manager = lambda: None
sys.modules["chatlab.app"] = app
sys.argv = ["chatlab", "--remote"]
runpy.run_module("chatlab", run_name="__main__")
"""
        for stage in ("build", "launch"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                environment = dict(os.environ, PYTHONPATH=str(REPO), **{
                    settings.SETTINGS_PATH_ENV: str(Path(directory) / "settings.json"),
                    library.LIBRARY_PATH_ENV: str(Path(directory) / "conversations.json"),
                    logs.LOG_PATH_ENV: str(Path(directory) / "ChatLab.log"),
                })
                server = subprocess.Popen(
                    [sys.executable, "-c", script, stage], cwd=directory, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
                try:
                    self.assertEqual(server.stdout.readline().strip(), "CHATLAB_COLD_START")
                    self.assertIsNone(server.poll())
                    server.stdin.close()
                    self.assertEqual(server.wait(5), 0)
                finally:
                    if server.poll() is None:
                        server.kill()
                        server.wait()
                    server.stdin.close()
                    server.stdout.close()

    def test_the_server_reports_its_address_and_stops_when_its_input_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(
                os.environ,
                PYTHONPATH=str(REPO),
                GRADIO_SERVER_NAME="0.0.0.0",
                **{
                    settings.SETTINGS_PATH_ENV: str(Path(directory) / "settings.json"),
                    library.LIBRARY_PATH_ENV: str(Path(directory) / "conversations.json"),
                    logs.LOG_PATH_ENV: str(Path(directory) / "ChatLab.log"),
                },
            )
            environment.pop("CONDUCTOR_PORT", None)
            server = subprocess.Popen(
                [sys.executable, "-m", "chatlab", "--remote"],
                cwd=directory,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                address = None
                for line in server.stdout:
                    if line.startswith(remote.READY_MARKER):
                        address = line[len(remote.READY_MARKER):].strip()
                        break
                self.assertIsNotNone(address, "the server exited without reporting its address")
                # Loopback, whatever GRADIO_SERVER_NAME asks for.
                self.assertTrue(address.startswith("http://127.0.0.1:"), address)
                with urlopen(f"{address.rstrip('/')}{api.API_PREFIX}/chatlab/status", timeout=15) as response:
                    self.assertEqual(response.status, 200)

                server.stdin.close()

                self.assertEqual(server.wait(30), 0)
            finally:
                if server.poll() is None:
                    server.kill()
                    server.wait()
                server.stdout.close()


class FakeWindow:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def evaluate_js(self, script):
        self.calls.append(("evaluate_js", script))
        return self.answer

    def load_url(self, url):
        self.calls.append(("load_url", url))

    def set_title(self, title):
        self.calls.append(("set_title", title))

    def create_confirmation_dialog(self, title, message):
        self.calls.append(("dialog", message))
        return True

    def named(self, name):
        return [arguments for call, *arguments in self.calls if call == name]


class FakeSession:
    def __init__(self, target, *, local_port, on_lost, start=None):
        self.target = target
        self.local_port = local_port
        self.on_lost = on_lost
        self._start = start
        self.closed = False

    def start(self):
        if self._start is not None:
            return self._start(self)
        return f"http://127.0.0.1:{self.local_port}/"

    def close(self):
        self.closed = True


class RemoteConnectionTests(unittest.TestCase):
    """The Remote menu: what the window shows as a session starts and ends."""

    LOCAL = "http://127.0.0.1:47890/"

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.saved = Path(self._directory.name) / "remote-host"
        self.sessions = []

    def tearDown(self):
        self._directory.cleanup()

    def connection(self, window, start=None):
        def factory(target, **options):
            session = FakeSession(target, start=start, **options)
            self.sessions.append(session)
            return session

        return RemoteConnection(window, self.LOCAL, self.saved, session_factory=factory)

    def test_connecting_shows_the_remote_server_and_remembers_the_host(self):
        window = FakeWindow("gpu-box:/srv/chatlab")
        connection = self.connection(window)

        connection.connect()

        session = self.sessions[0]
        self.assertEqual(session.target, remote.RemoteTarget("gpu-box", "/srv/chatlab"))
        self.assertEqual(window.named("load_url"), [[f"http://127.0.0.1:{session.local_port}/"]])
        self.assertEqual(window.named("set_title")[-1], [f"{WINDOW_TITLE} — gpu-box"])
        self.assertEqual(remote.load_target(self.saved), session.target)

    def test_the_prompt_offers_the_last_host(self):
        remote.save_target(self.saved, remote.RemoteTarget("gpu-box", "/srv/chatlab"))
        window = FakeWindow(None)

        self.connection(window).connect()

        self.assertIn('"gpu-box:/srv/chatlab"', window.named("evaluate_js")[0][0])
        self.assertEqual(self.sessions, [])

    def test_a_host_that_is_not_one_is_explained(self):
        window = FakeWindow("-oProxyCommand=x")

        self.connection(window).connect()

        self.assertEqual(self.sessions, [])
        self.assertEqual(len(window.named("dialog")), 1)

    def test_a_session_that_fails_to_start_says_why_and_leaves_the_window_here(self):
        def fail(session):
            raise remote.RemoteError("Host key verification failed.")

        window = FakeWindow("gpu-box")
        connection = self.connection(window, start=fail)

        connection.connect()

        self.assertIsNone(connection.session)
        self.assertEqual(window.named("load_url"), [])
        self.assertEqual(window.named("set_title")[-1], [WINDOW_TITLE])
        self.assertIn("Host key verification failed.", window.named("dialog")[0][0])
        self.assertIsNone(remote.load_target(self.saved))

    def test_disconnecting_while_connecting_is_not_reported_as_a_failure(self):
        def disconnected(session):
            connection.disconnect()
            raise remote.RemoteError("The connection was cancelled.")

        window = FakeWindow("gpu-box")
        connection = self.connection(window, start=disconnected)

        connection.connect()

        self.assertTrue(self.sessions[0].closed)
        self.assertEqual(window.named("dialog"), [])
        self.assertEqual(window.named("load_url"), [[self.LOCAL]])

    def test_disconnecting_goes_back_to_this_mac(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()

        connection.disconnect()

        self.assertTrue(self.sessions[0].closed)
        self.assertEqual(window.named("load_url")[-1], [self.LOCAL])
        self.assertEqual(window.named("set_title")[-1], [WINDOW_TITLE])

    def test_a_lost_connection_goes_back_to_this_mac_and_says_so(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()
        session = self.sessions[0]

        session.on_lost(session, "The server connection to gpu-box ended.")

        self.assertIsNone(connection.session)
        self.assertEqual(window.named("load_url")[-1], [self.LOCAL])
        self.assertIn("The server connection to gpu-box ended.", window.named("dialog")[0][0])

    def test_a_second_connect_while_connected_is_refused(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()

        connection.connect()

        self.assertEqual(len(self.sessions), 1)
        self.assertIn("already connected to gpu-box", window.named("dialog")[0][0])

    def test_disconnect_with_nothing_connected_says_so(self):
        window = FakeWindow()

        self.connection(window).disconnect()

        self.assertEqual(window.named("load_url"), [])
        self.assertEqual(len(window.named("dialog")), 1)

    def test_old_disconnect_cannot_replace_a_reconnected_window(self):
        window = FakeWindow("old-gpu")
        connection = self.connection(window)
        connection.connect()
        old = self.sessions[0]
        entered, release = threading.Event(), threading.Event()
        def close():
            entered.set()
            if not release.wait(3):
                raise AssertionError("old close was not released")
            old.closed = True
        old.close = close
        thread = threading.Thread(target=connection.disconnect)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            window.answer = "new-gpu"
            connection.connect()
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIs(connection.session, self.sessions[1])
        self.assertNotEqual(window.named("load_url")[-1], [self.LOCAL])
        self.assertEqual(window.named("set_title")[-1], [f"{WINDOW_TITLE} — new-gpu"])

    def test_old_loss_callback_cannot_replace_a_reconnected_window(self):
        window = FakeWindow("old-gpu")
        connection = self.connection(window)
        connection.connect()
        old = self.sessions[0]
        entered, release = threading.Event(), threading.Event()
        show = connection._show_local
        def delayed(generation):
            entered.set()
            if not release.wait(3):
                raise AssertionError("old transition was not released")
            return show(generation)
        with mock.patch.object(connection, "_show_local", side_effect=delayed):
            thread = threading.Thread(target=lambda: old.on_lost(old, "Old server ended"))
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                window.answer = "new-gpu"
                connection.connect()
            finally:
                release.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIs(connection.session, self.sessions[1])
        self.assertEqual(window.named("set_title")[-1], [f"{WINDOW_TITLE} — new-gpu"])
        self.assertEqual(window.named("dialog"), [])

    def test_quitting_during_the_host_prompt_cannot_start_a_session(self):
        window = FakeWindow("gpu-box")
        start = mock.Mock(return_value="http://127.0.0.1:47891/")
        connection = self.connection(window, start=start)
        entered, release = threading.Event(), threading.Event()
        def answer(script):
            entered.set()
            if not release.wait(3):
                raise AssertionError("prompt was not released")
            return "gpu-box"
        window.evaluate_js = answer
        thread = threading.Thread(target=connection.connect)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            connection.close()
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        start.assert_not_called()
        self.assertIsNone(connection.session)
        self.assertEqual(window.named("load_url"), [])

    def test_cancelled_start_cannot_overwrite_the_new_saved_host(self):
        entered, release = threading.Event(), threading.Event()
        def start(session):
            if session.target.host == "old-gpu":
                entered.set()
                if not release.wait(3):
                    raise AssertionError("old start was not released")
            return f"http://127.0.0.1:{session.local_port}/"
        window = FakeWindow("old-gpu")
        connection = self.connection(window, start=start)
        thread = threading.Thread(target=connection.connect)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            connection.disconnect()
            window.answer = "new-gpu"
            connection.connect()
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(remote.load_target(self.saved).host, "new-gpu")
        self.assertIs(connection.session, self.sessions[1])
        self.assertEqual(window.named("set_title")[-1], [f"{WINDOW_TITLE} — new-gpu"])

    def test_menu_worker_returns_while_startup_waits_and_disconnect_remains_available(self):
        entered, release = threading.Event(), threading.Event()
        def start(session):
            entered.set()
            if not release.wait(3):
                raise AssertionError("startup was not released")
            return f"http://127.0.0.1:{session.local_port}/"
        window = FakeWindow("gpu-box")
        connection = self.connection(window, start=start)
        worker = connection.connect_in_background()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(worker.is_alive())
            stopping = connection.disconnect_in_background()
            stopping.join(1)
            self.assertFalse(stopping.is_alive())
            self.assertIsNone(connection.session)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(window.named("load_url")[-1], [self.LOCAL])

    def test_quit_waits_for_a_disconnect_worker_to_finish_cleanup(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()
        session = self.sessions[0]
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def close():
            entered.set()
            if not release.wait(3):
                raise AssertionError("cleanup was not released")
            session.closed = True
        session.close = close
        worker = connection.disconnect_in_background()
        quitting = threading.Thread(target=lambda: (connection.close(), done.set()))
        try:
            self.assertTrue(entered.wait(1))
            self.assertIsNone(connection.session)
            quitting.start()
            self.assertFalse(done.wait(0.1))
        finally:
            release.set()
            worker.join(3)
            quitting.join(3)
        self.assertTrue(done.is_set())
        self.assertTrue(session.closed)
        self.assertFalse(connection._disconnect_workers)

    def test_quitting_ends_the_session(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()

        connection.close()

        self.assertTrue(self.sessions[0].closed)


if __name__ == "__main__":
    unittest.main()
