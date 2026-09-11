"""Tests for the desktop app's local-server lifecycle."""

from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.request import urlopen

import desktop_launcher
from desktop_launcher import (
    LOOPBACK_ADDRESS,
    find_available_port,
    remember_port,
    remembered_port,
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


class RememberedPortTests(unittest.TestCase):
    """The window keeps its port so the browser keeps what it stores by origin.

    Pane widths are kept per origin, and the port is part of the origin, so
    a window served somewhere new each launch would forget them each launch.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patch = mock.patch.object(
            desktop_launcher, "app_support_directory", lambda: Path(self.directory.name)
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_nothing_is_remembered_before_a_first_launch(self):
        self.assertIsNone(remembered_port())

    def test_the_port_a_launch_used_is_offered_to_the_next_one(self):
        port = find_available_port()

        remember_port(port)

        self.assertEqual(remembered_port(), port)

    def test_a_port_something_else_holds_is_given_up(self):
        port = find_available_port()
        remember_port(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind((LOOPBACK_ADDRESS, port))
            self.assertIsNone(remembered_port())

    def test_a_record_that_is_not_a_port_is_ignored(self):
        for written in ["", "not a port", "0", "70000"]:
            with self.subTest(written=written):
                (Path(self.directory.name) / "port").write_text(written)

                self.assertIsNone(remembered_port())


if __name__ == "__main__":
    unittest.main()
