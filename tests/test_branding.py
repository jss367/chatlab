"""Tests for the app's mark: the icon macOS draws and the browser tab's."""

from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

import branding


REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "ChatLab.spec"


class FaviconTests(unittest.TestCase):
    def test_the_favicon_ships_with_the_checkout(self):
        self.assertEqual(branding.favicon_path(), str(branding.FAVICON))

    def test_a_missing_favicon_costs_the_picture_and_not_the_launch(self):
        original = branding.FAVICON
        branding.FAVICON = REPO / "assets" / "no-such-icon.png"
        try:
            self.assertIsNone(branding.favicon_path())
        finally:
            branding.FAVICON = original

    def test_the_favicon_corners_are_transparent(self):
        """The artwork is a tile on black; the black is not part of the mark."""

        icon = Image.open(branding.FAVICON).convert("RGBA")
        self.assertEqual(icon.getpixel((0, 0))[3], 0)
        self.assertEqual(icon.getpixel((icon.width - 1, icon.height - 1))[3], 0)
        self.assertEqual(icon.getpixel((icon.width // 2, icon.height // 2))[3], 255)


class BundleIconTests(unittest.TestCase):
    def test_the_bundle_is_built_with_the_icns(self):
        spec = SPEC.read_text()
        self.assertRegex(spec, r"icon=ICON,")
        self.assertTrue((REPO / "assets" / "ChatLab.icns").is_file())

    def test_the_bundle_carries_the_favicon_as_data(self):
        """A frozen app reads the tab's picture out of the bundle, not a checkout."""

        self.assertIn('"assets", "icon.png"', SPEC.read_text())


class LaunchTests(unittest.TestCase):
    """Both ways in - the window and ``python app.py`` - draw the same tab."""

    def test_both_launch_calls_pass_the_favicon(self):
        for module in ("app.py", "desktop_launcher.py"):
            with self.subTest(module=module):
                source = (REPO / module).read_text()
                self.assertIn("favicon_path=branding.favicon_path()", source)


if __name__ == "__main__":
    unittest.main()
