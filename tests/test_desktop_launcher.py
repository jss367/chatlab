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
import device_memory
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

    def a_release(self):
        """A release newer than this one, for the tests that install it."""

        return updater.ReleaseInfo(
            version="9.9.9",
            asset_name="ChatLab.zip",
            asset_url="https://example.invalid/ChatLab.zip",
            asset_size=None,
            release_url="https://example.invalid/release",
            checksum_url=None,
        )

    def open_window(self, bundle, prepare=None, restarts=1, relaunch_effect=None):
        """Run the launcher against a fake window, restarting once it is up.

        ``prepare`` is handed the live :class:`UpdateFlow` just before the
        restart, for the tests that need an update already under way.
        ``restarts`` presses the button more than once, for the tests about a
        second confirmation arriving before the first has taken the app away,
        and ``relaunch_effect`` is ``updater.relaunch``'s side effect, for the
        ones where reopening the app fails.
        """

        window = mock.MagicMock()
        seen = {}
        flows = []
        build_flow = desktop_launcher.UpdateFlow

        def make_flow(*args, **kwargs):
            flow = build_flow(*args, **kwargs)
            flows.append(flow)
            return flow

        def started(**_kwargs):
            if prepare is not None:
                prepare(flows[-1])
            seen["offered"] = desktop.restart_offered()
            seen["answers"] = [desktop.restart() for _ in range(restarts)]
            seen["declined"] = seen["answers"][0]

        self.enterContext(mock.patch.object(desktop_launcher, "UpdateFlow", make_flow))
        webview = SimpleNamespace(create_window=mock.Mock(return_value=window), start=started)
        menu = SimpleNamespace(Menu=mock.Mock(), MenuAction=mock.Mock())
        with tempfile.TemporaryDirectory() as support, \
                mock.patch.dict(sys.modules, {"webview": webview, "webview.menu": menu}), \
                mock.patch.object(desktop_launcher, "app_support_directory", return_value=Path(support)), \
                mock.patch.object(desktop_launcher, "start_local_server", return_value=(mock.Mock(), "http://127.0.0.1:47890/")), \
                mock.patch.object(device_memory, "watch_memory"), \
                mock.patch.object(updater, "running_app_bundle", return_value=bundle), \
                mock.patch.object(updater, "remove_previous_bundles"), \
                mock.patch.object(updater, "remove_stale_work_dirs"), \
                mock.patch.object(updater, "relaunch") as relaunch:
            relaunch.side_effect = relaunch_effect
            self.assertEqual(desktop_launcher.run_desktop(), 0)
        return seen, window, relaunch

    def test_the_app_reopens_itself_and_closes_the_window_behind_it(self):
        bundle = Path("/Applications/ChatLab.app")
        seen, window, relaunch = self.open_window(bundle)

        self.assertTrue(seen["offered"])
        self.assertIsNone(seen["declined"])
        relaunch.assert_called_once_with(bundle)
        window.destroy.assert_called_once_with()

    def test_a_restart_asked_for_mid_update_waits_for_the_update_to_finish(self):
        """The swap owns the bundle, and the update reopens the app itself.

        Starting a second copy over a bundle being replaced would leave the
        current window open anyway - ``_on_closing`` refuses the close - so the
        restart stands down and says why.
        """

        seen, window, relaunch = self.open_window(
            Path("/Applications/ChatLab.app"), prepare=lambda flow: flow.swapping.set()
        )

        self.assertTrue(seen["offered"])
        self.assertIn("installing an update", seen["declined"])
        relaunch.assert_not_called()
        window.destroy.assert_not_called()

    def test_a_restart_asked_for_while_the_update_relaunches_stands_down(self):
        """The stretch between the swap ending and the relaunch is the update's.

        ``_offer`` clears ``swapping`` before it relaunches and closes the
        window, so a restart confirmed in that gap would otherwise start a
        second copy of the one the update is already bringing up. The update's
        own close still has to go through, or the replaced window would be
        left sitting there.
        """

        flows = []

        def mid_handoff(flow):
            flows.append(flow)
            self.assertTrue(flow._begin_swap())
            flow._end_swap(relaunching=True)

        seen, window, relaunch = self.open_window(
            Path("/Applications/ChatLab.app"), prepare=mid_handoff
        )

        self.assertTrue(seen["offered"])
        self.assertIn("already restarting", seen["declined"])
        relaunch.assert_not_called()
        window.destroy.assert_not_called()
        self.assertTrue(flows[0]._on_closing())

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
                    mock.patch.object(device_memory, "watch_memory"):
                self.assertEqual(desktop_launcher.run_desktop(), 0)
            self.assertIsNone(desktop.restart())
        patched.relaunch.assert_called_once_with(Path("/Applications/ChatLab.app"))

    def test_an_update_that_fails_hands_the_restart_back(self):
        """Only a finished swap owns the restart; a failed one gives it up.

        The flag that keeps Settings' restart out through the relaunch is set
        on the success path alone, so somebody whose update fell over can
        still restart into the choice they just saved.
        """

        flow = desktop_launcher.UpdateFlow(mock.MagicMock(), Path("/Applications/ChatLab.app"))
        with mock.patch.object(updater, "install_update", side_effect=updater.UpdateError("disk full")), \
                mock.patch.object(updater, "relaunch") as relaunch:
            with self.assertLogs(level="ERROR"):
                flow._offer(self.a_release())

        relaunch.assert_not_called()
        self.assertFalse(flow.relaunching.is_set())
        self.assertIsNone(flow.may_restart())

    def test_a_second_restart_confirmation_does_not_open_a_second_copy(self):
        """Confirming twice before the app goes away is still one restart.

        A double-click, or a second tab left open on the window's fixed port,
        gets two confirmations through before the server dies. The first takes
        the claim the update flow keeps for whoever is bringing a copy up, so
        the second is refused and says a restart is already under way.
        """

        bundle = Path("/Applications/ChatLab.app")
        seen, window, relaunch = self.open_window(bundle, restarts=2)

        self.assertIsNone(seen["answers"][0])
        self.assertIn("already restarting", seen["answers"][1])
        relaunch.assert_called_once_with(bundle)
        window.destroy.assert_called_once_with()

    def test_a_restart_that_cannot_reopen_the_app_leaves_it_where_it_is(self):
        """``open`` can fail, and a window closed after it would take the app.

        The claim goes back when it does, so the reader who presses the button
        again gets a real attempt rather than a refusal, and the window they
        are still looking at is the one the second attempt closes.
        """

        bundle = Path("/Applications/ChatLab.app")
        with self.assertLogs(level="ERROR"):
            seen, window, relaunch = self.open_window(
                bundle, restarts=2, relaunch_effect=[OSError("no such file"), None]
            )

        self.assertEqual(seen["answers"][0], desktop_launcher.RESTART_FAILED)
        self.assertIsNone(seen["answers"][1])
        self.assertEqual(relaunch.call_count, 2)
        window.destroy.assert_called_once_with()

    def test_an_update_that_cannot_reopen_the_app_hands_the_restart_back(self):
        """The bundle is replaced; only the reopening fell over.

        Destroying the window would leave the reader with nothing on screen
        and nothing coming, so it stays open, and the restart it still offers
        has to work - the update is not relaunching any more.
        """

        flow = desktop_launcher.UpdateFlow(mock.MagicMock(), Path("/Applications/ChatLab.app"))
        with mock.patch.object(updater, "install_update"), \
                mock.patch.object(updater, "relaunch", side_effect=OSError("no such file")):
            with self.assertLogs(level="ERROR"):
                flow._offer(self.a_release())

        flow.window.destroy.assert_not_called()
        self.assertFalse(flow.relaunching.is_set())
        self.assertIsNone(flow.may_restart())

    def test_a_restart_that_failed_does_not_cancel_the_next_update(self):
        """The stop a restart posts belongs to that restart, not to the app.

        Claiming a restart cancels whatever is downloading, so nothing slips
        into the swap behind a window that is about to go. When the relaunch
        then falls over the window stays, and the update flow has to keep
        working: an update asked for afterwards downloads and swaps rather
        than being cancelled the moment it starts.
        """

        flow = desktop_launcher.UpdateFlow(mock.MagicMock(), Path("/Applications/ChatLab.app"))
        self.assertIsNone(flow.may_restart())
        stopped = flow.cancel
        flow.release_restart()

        seen = {}

        def install(release, bundle, progress=None, begin_swap=None, cancelled=None):
            seen["cancelled"] = cancelled()
            seen["swapped"] = begin_swap()

        with mock.patch.object(updater, "install_update", side_effect=install), \
                mock.patch.object(updater, "relaunch") as relaunch:
            flow._offer(self.a_release())

        self.assertFalse(seen["cancelled"])
        self.assertTrue(seen["swapped"])
        self.assertTrue(stopped.is_set())
        relaunch.assert_called_once_with(flow.bundle)

    def test_the_update_a_failed_restart_stopped_stays_stopped(self):
        """A worker told to stop is not un-told by the restart giving up.

        The restart arrives mid-download and hands its claim back when the
        relaunch fails. The download it cancelled reads the event it was
        cancelled through, so it still reports cancelled and is still refused
        the swap - the fresh event is for the next update, not for this one.
        """

        flow = desktop_launcher.UpdateFlow(mock.MagicMock(), Path("/Applications/ChatLab.app"))
        seen = {}

        def install(release, bundle, progress=None, begin_swap=None, cancelled=None):
            self.assertIsNone(flow.may_restart())
            flow.release_restart()
            seen["cancelled"] = cancelled()
            seen["swapped"] = begin_swap()
            raise updater.UpdateCancelled("Update cancelled before installation.")

        with mock.patch.object(updater, "install_update", side_effect=install), \
                mock.patch.object(updater, "relaunch") as relaunch:
            with self.assertLogs(level="INFO"):
                flow._offer(self.a_release())

        self.assertTrue(seen["cancelled"])
        self.assertFalse(seen["swapped"])
        self.assertFalse(flow.swapping.is_set())
        relaunch.assert_not_called()
        flow.window.destroy.assert_not_called()

    def test_a_run_from_a_checkout_offers_nothing_to_restart(self):
        seen, _, relaunch = self.open_window(None)

        self.assertFalse(seen["offered"])
        self.assertEqual(seen["declined"], desktop.NO_RESTART_AVAILABLE)
        relaunch.assert_not_called()
