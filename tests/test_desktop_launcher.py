"""Tests for the desktop app's local-server lifecycle."""

from __future__ import annotations

import unittest
from urllib.request import urlopen

from desktop_launcher import LOOPBACK_ADDRESS, find_available_port, start_local_server


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


if __name__ == "__main__":
    unittest.main()
