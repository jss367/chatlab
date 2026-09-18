"""Tests for the desktop app's local-server lifecycle."""

from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.request import urlopen

import desktop
import desktop_launcher
import logs
import model_runtime
import updater
from desktop_launcher import (
    DESKTOP_PORT,
    LOOPBACK_ADDRESS,
    find_available_port,
    port_is_free,
    start_local_server,
)


class DesktopLauncherTests(unittest.TestCase):
    def test_available_port_uses_loopback(self):
        port = find_available_port()

        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_local_server_responds_and_can_close(self):
        demo, local_url = start_local_server()
        try:
            self.assertTrue(local_url.startswith(f"http://{LOOPBACK_ADDRESS}:"))
            with urlopen(local_url, timeout=15) as response:
                self.assertEqual(response.status, 200)
        finally:
            demo.close(verbose=False)


class DesktopPortTests(unittest.TestCase):
    """The window is served on one port, so the browser keeps what it stores.

    A pane width is kept under the origin the page came from, and the port
    is part of that origin, so a port chosen afresh each launch would lose
    the width each launch.
    """

    def test_the_port_is_outside_the_range_macos_hands_out(self):
        # So a window that had to fall back to any free port is never given
        # this one, and never serves a second instance from the first one's
        # origin.
        self.assertLess(DESKTOP_PORT, 49152)
        self.assertGreater(DESKTOP_PORT, 1024)

    def test_a_free_port_is_free_and_a_held_one_is_not(self):
        port = find_available_port()
        self.assertTrue(port_is_free(port))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind((LOOPBACK_ADDRESS, port))
            self.assertFalse(port_is_free(port))

    def test_a_window_gives_up_the_usual_port_rather_than_refusing_to_open(self):
        # A second instance, or anything else holding the port, is no reason
        # not to open a window; it is only a reason for that window to be a
        # different origin, which is the old behavior.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            try:
                held.bind((LOOPBACK_ADDRESS, DESKTOP_PORT))
            except OSError:
                self.skipTest("something on this machine already holds the port")
            self.assertFalse(port_is_free(DESKTOP_PORT))


if __name__ == "__main__":
    unittest.main()


class LaunchRecordTests(unittest.TestCase):
    """The log the launcher opens with, and the watch it leaves running.

    The desktop app has no terminal, so the file it writes is the whole
    account of a session. Both halves are wired here rather than in ``logs``
    itself, so both are worth a test that they are still wired.
    """

    def test_the_launcher_uses_the_shared_rules_rather_than_its_own(self):
        with mock.patch.object(logs, "configure", return_value=Path("/tmp/ChatLab.log")) as configure:
            self.assertEqual(desktop_launcher.configure_logging(), Path("/tmp/ChatLab.log"))
        configure.assert_called_once_with()

    def test_starting_up_records_the_environment_before_anything_else_runs(self):
        order = []
        with mock.patch.object(logs, "configure", side_effect=lambda: order.append("configure")), \
                mock.patch.object(logs, "log_environment", side_effect=lambda target: order.append("record")), \
                mock.patch.object(desktop_launcher, "run_desktop", side_effect=lambda: order.append("run") or 0):
            self.assertEqual(desktop_launcher.main([]), 0)
        self.assertEqual(order, ["configure", "record", "run"])

    def test_a_launch_that_fails_is_recorded_rather_than_swallowed(self):
        with mock.patch.object(logs, "configure", return_value=None), \
                mock.patch.object(logs, "log_environment"), \
                mock.patch.object(desktop_launcher, "run_desktop", side_effect=RuntimeError("no window")):
            with self.assertLogs(level="ERROR") as caught:
                with self.assertRaises(RuntimeError):
                    desktop_launcher.main([])
        self.assertIn("ChatLab failed to start", "\n".join(caught.output))


class RestartOfferTests(unittest.TestCase):
    """The restart Settings offers is the window's, and only the app has one.

    A run from a checkout has no bundle to reopen, so it registers nothing
    and the button stays off the page; see the extension tests for what the
    page does with the answer.
    """

    def setUp(self):
        self.addCleanup(desktop.offer_restart, None)

    def open_window(self, bundle):
        """Run the launcher against a fake window, restarting once it is up."""

        window = mock.MagicMock()
        seen = {}

        def started(**_kwargs):
            seen["offered"] = desktop.restart_offered()
            seen["restarted"] = desktop.restart()

        webview = SimpleNamespace(create_window=mock.Mock(return_value=window), start=started)
        menu = SimpleNamespace(Menu=mock.Mock(), MenuAction=mock.Mock())
        with tempfile.TemporaryDirectory() as support, \
                mock.patch.dict(sys.modules, {"webview": webview, "webview.menu": menu}), \
                mock.patch.object(desktop_launcher, "app_support_directory", return_value=Path(support)), \
                mock.patch.object(desktop_launcher, "start_local_server", return_value=(mock.Mock(), "http://127.0.0.1:47890/")), \
                mock.patch.object(model_runtime, "watch_memory"), \
                mock.patch.object(updater, "running_app_bundle", return_value=bundle), \
                mock.patch.object(updater, "remove_previous_bundles"), \
                mock.patch.object(updater, "remove_stale_work_dirs"), \
                mock.patch.object(updater, "relaunch") as relaunch:
            self.assertEqual(desktop_launcher.run_desktop(), 0)
        return seen, window, relaunch

    def test_the_app_reopens_itself_and_closes_the_window_behind_it(self):
        bundle = Path("/Applications/ChatLab.app")
        seen, window, relaunch = self.open_window(bundle)

        self.assertTrue(seen["offered"])
        self.assertTrue(seen["restarted"])
        relaunch.assert_called_once_with(bundle)
        window.destroy.assert_called_once_with()

    def test_a_window_the_user_already_closed_costs_the_restart_nothing(self):
        window = mock.MagicMock()
        window.destroy.side_effect = RuntimeError("window is gone")
        with mock.patch.object(desktop_launcher, "updater") as patched:
            patched.running_app_bundle.return_value = Path("/Applications/ChatLab.app")
            with mock.patch.dict(sys.modules, {
                "webview": SimpleNamespace(create_window=mock.Mock(return_value=window), start=lambda **_: None),
                "webview.menu": SimpleNamespace(Menu=mock.Mock(), MenuAction=mock.Mock()),
            }), tempfile.TemporaryDirectory() as support, \
                    mock.patch.object(desktop_launcher, "app_support_directory", return_value=Path(support)), \
                    mock.patch.object(desktop_launcher, "start_local_server", return_value=(mock.Mock(), "http://127.0.0.1:47890/")), \
                    mock.patch.object(model_runtime, "watch_memory"):
                self.assertEqual(desktop_launcher.run_desktop(), 0)
            self.assertTrue(desktop.restart())
        patched.relaunch.assert_called_once_with(Path("/Applications/ChatLab.app"))

    def test_a_run_from_a_checkout_offers_nothing_to_restart(self):
        seen, _, relaunch = self.open_window(None)

        self.assertFalse(seen["offered"])
        self.assertFalse(seen["restarted"])
        relaunch.assert_not_called()
