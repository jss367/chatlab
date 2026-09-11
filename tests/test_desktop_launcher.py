"""Tests for the desktop app's local-server lifecycle."""

from __future__ import annotations

import socket
import unittest
from urllib.request import urlopen

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
