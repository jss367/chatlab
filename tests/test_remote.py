"""Tests for running ChatLab on another machine over SSH.

``fake_ssh.py`` stands in for ``ssh``: it runs the remote command here and
forwards ports itself, so a session goes through every step it takes
against a real host. ``fake_remote_chatlab.py`` stands in for the server
at the far end, except in the one test that starts the real one.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
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

    def test_disconnecting_stops_the_remote_server(self):
        session = self.session()
        session.start()
        pid = self.server_pid()

        session.close()

        self.assertTrue(_gone(pid), "the remote server outlived its session")
        with self.assertRaises(OSError):
            urlopen(f"http://127.0.0.1:{session.local_port}/", timeout=2)

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

    def test_quitting_ends_the_session(self):
        window = FakeWindow("gpu-box")
        connection = self.connection(window)
        connection.connect()

        connection.close()

        self.assertTrue(self.sessions[0].closed)


if __name__ == "__main__":
    unittest.main()
