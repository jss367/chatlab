"""Tests for the desktop self-updater's pure logic."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import updater


class VersionTests(unittest.TestCase):
    def test_parse_accepts_prefixed_and_bare(self):
        self.assertEqual(updater.parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(updater.parse_version("0.2"), (0, 2))
        self.assertIsNone(updater.parse_version("nightly"))
        self.assertIsNone(updater.parse_version("v1.2-rc1"))

    def test_is_newer_pads_short_versions(self):
        self.assertTrue(updater.is_newer("v0.3.0", "0.2.0"))
        self.assertTrue(updater.is_newer("1.0", "0.9.9"))
        self.assertFalse(updater.is_newer("v0.2", "0.2.0"))
        self.assertFalse(updater.is_newer("v0.1.9", "0.2.0"))
        self.assertFalse(updater.is_newer("garbage", "0.2.0"))


class ReleaseSelectionTests(unittest.TestCase):
    payload = {
        "tag_name": "v0.3.0",
        "html_url": "https://github.com/jss367/chatlab/releases/tag/v0.3.0",
        "assets": [
            {"name": "ChatLab-macos-x86_64.zip", "browser_download_url": "https://x/intel.zip", "size": 1},
            {"name": "ChatLab-macos-arm64.zip", "browser_download_url": "https://x/arm.zip", "size": 2},
        ],
    }

    def test_expected_asset_name(self):
        self.assertEqual(updater.expected_asset_name("arm64"), "ChatLab-macos-arm64.zip")
        self.assertEqual(updater.expected_asset_name("x86_64"), "ChatLab-macos-x86_64.zip")

    def test_selects_matching_asset(self):
        release = updater.select_release(self.payload, "ChatLab-macos-arm64.zip")
        self.assertEqual(release.version, "0.3.0")
        self.assertEqual(release.asset_url, "https://x/arm.zip")
        self.assertEqual(release.asset_size, 2)

    def test_missing_asset_or_bad_tag_gives_none(self):
        self.assertIsNone(updater.select_release(self.payload, "ChatLab-macos-riscv.zip"))
        self.assertIsNone(updater.select_release({"tag_name": "latest", "assets": []}, "x"))

    def test_check_for_update_skips_same_or_older(self):
        with mock.patch.object(updater, "fetch_latest_release", return_value=self.payload), mock.patch.object(
            updater, "expected_asset_name", return_value="ChatLab-macos-arm64.zip"
        ):
            self.assertIsNone(updater.check_for_update("0.3.0"))
            self.assertIsNone(updater.check_for_update("1.0.0"))
            self.assertEqual(updater.check_for_update("0.2.0").version, "0.3.0")


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def make_bundle(self, name: str, marker: str) -> Path:
        bundle = self.root / name
        (bundle / "Contents" / "MacOS").mkdir(parents=True)
        (bundle / "Contents" / "MacOS" / "ChatLab").write_text(marker)
        return bundle

    def test_running_app_bundle_from_executable(self):
        bundle = self.make_bundle("ChatLab.app", "old")
        self.assertEqual(updater.running_app_bundle(str(bundle / "Contents" / "MacOS" / "ChatLab")), bundle.resolve())
        self.assertIsNone(updater.running_app_bundle("/usr/bin/python3"))

    def test_swap_bundle_parks_old_and_cleanup_removes_it(self):
        current = self.make_bundle("ChatLab.app", "old")
        replacement = self.make_bundle("staging/ChatLab.app", "new")

        parked = updater.swap_bundle(current, replacement)

        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "new")
        self.assertTrue(parked.name.startswith("ChatLab.app.previous-"))
        self.assertEqual((parked / "Contents" / "MacOS" / "ChatLab").read_text(), "old")

        updater.remove_previous_bundles(current)
        self.assertFalse(parked.exists())
        self.assertTrue(current.exists())

    def test_swap_bundle_restores_old_after_partial_copy(self):
        current = self.make_bundle("ChatLab.app", "old")
        replacement = self.make_bundle("staging/ChatLab.app", "new")

        def partial_move(src, dst):
            Path(dst).mkdir()
            (Path(dst) / "half-written").write_text("x")
            raise OSError("No space left on device")

        with mock.patch.object(updater.shutil, "move", side_effect=partial_move):
            with self.assertRaises(updater.UpdateError):
                updater.swap_bundle(current, replacement)

        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "old")
        self.assertFalse((current / "half-written").exists())
        self.assertEqual(list(self.root.glob("ChatLab.app.previous-*")), [])

    @unittest.skipUnless(shutil.which("ditto"), "ditto is macOS-only")
    def test_extract_bundle_keeps_symlinks(self):
        bundle = self.make_bundle("ChatLab.app", "bin")
        (bundle / "Contents" / "link").symlink_to("MacOS/ChatLab")
        archive = self.root / "ChatLab-macos-arm64.zip"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(bundle), str(archive)], check=True)

        extracted = updater.extract_bundle(archive, self.root / "out")

        self.assertEqual(extracted.name, "ChatLab.app")
        self.assertTrue((extracted / "Contents" / "link").is_symlink())

    def test_install_update_wires_the_steps(self):
        current = self.make_bundle("ChatLab.app", "old")
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", None, "u")
        work = self.root / "work"

        def fake_download(rel, dest, progress):
            dest.mkdir(parents=True, exist_ok=True)
            progress(5, 10)
            return dest / rel.asset_name

        def fake_extract(archive, dest):
            return self.make_bundle("work/unpacked/ChatLab.app", "new")

        seen = []
        with mock.patch.object(updater, "download_asset", side_effect=fake_download), mock.patch.object(
            updater, "extract_bundle", side_effect=fake_extract
        ):
            updater.install_update(release, current, work_dir=work, progress=lambda r, t: seen.append((r, t)))

        self.assertEqual(seen, [(5, 10)])
        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "new")
        self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
