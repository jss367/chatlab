"""Tests for the desktop self-updater's pure logic."""

from __future__ import annotations

import hashlib
import plistlib
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
            {"name": "ChatLab-macos-arm64.zip.sha256", "browser_download_url": "https://x/arm.zip.sha256", "size": 65},
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
        self.assertEqual(release.checksum_url, "https://x/arm.zip.sha256")
        self.assertIsNone(updater.select_release(self.payload, "ChatLab-macos-x86_64.zip").checksum_url)

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

    def make_bundle(self, name: str, marker: str, version: str = "0.3.0", identifier: str | None = None) -> Path:
        bundle = self.root / name
        (bundle / "Contents" / "MacOS").mkdir(parents=True)
        (bundle / "Contents" / "MacOS" / "ChatLab").write_text(marker)
        with (bundle / "Contents" / "Info.plist").open("wb") as handle:
            plistlib.dump(
                {
                    "CFBundleIdentifier": identifier or updater.BUNDLE_IDENTIFIER,
                    "CFBundleShortVersionString": version,
                },
                handle,
            )
        return bundle

    RELEASE = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", None, "u")

    def test_verify_bundle_accepts_matching_and_rejects_others(self):
        updater.verify_bundle(self.make_bundle("ok.app", "x"), self.RELEASE)
        with self.assertRaisesRegex(updater.UpdateError, "identifies itself"):
            updater.verify_bundle(self.make_bundle("other.app", "x", identifier="com.evil.app"), self.RELEASE)
        with self.assertRaisesRegex(updater.UpdateError, "version 0.2.0"):
            updater.verify_bundle(self.make_bundle("old.app", "x", version="0.2.0"), self.RELEASE)
        broken = self.make_bundle("broken.app", "x")
        (broken / "Contents" / "Info.plist").unlink()
        with self.assertRaisesRegex(updater.UpdateError, "Info.plist"):
            updater.verify_bundle(broken, self.RELEASE)

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

        manual = self.make_bundle("ChatLab.app.previous-manual", "keep")
        versioned = self.make_bundle("ChatLab.app.previous-0.1.0", "keep")

        updater.remove_previous_bundles(current)
        self.assertFalse(parked.exists())
        self.assertTrue(current.exists())
        self.assertTrue(manual.exists())
        self.assertTrue(versioned.exists())

    def test_remove_stale_work_dirs_sweeps_beside_app_and_tempdir(self):
        current = self.make_bundle("ChatLab.app", "old")
        beside = self.root / "chatlab-update-abc"; beside.mkdir(); (beside / "big.zip").write_text("x")
        tmp_root = self.root / "tmp"; tmp_root.mkdir()
        in_tmp = tmp_root / "chatlab-update-def"; in_tmp.mkdir()
        unrelated = self.root / "chatlab-updates.txt"; unrelated.write_text("keep")
        with mock.patch.object(updater.tempfile, "gettempdir", return_value=str(tmp_root)):
            updater.remove_stale_work_dirs(current)
        self.assertFalse(beside.exists())
        self.assertFalse(in_tmp.exists())
        self.assertTrue(unrelated.exists())
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

    @unittest.skipUnless(shutil.which("ditto"), "ditto is macOS-only")
    def test_extract_bundle_can_be_cancelled(self):
        bundle = self.make_bundle("ChatLab.app", "bin")
        archive = self.root / "ChatLab-macos-arm64.zip"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(bundle), str(archive)], check=True)
        real_popen = updater.subprocess.Popen
        slow = lambda args, **kw: real_popen(["sleep", "30"], **kw)  # noqa: E731
        with mock.patch.object(updater.subprocess, "Popen", side_effect=slow):
            with self.assertRaises(updater.UpdateCancelled):
                updater.extract_bundle(archive, self.root / "out", cancelled=lambda: True)

    @staticmethod
    def fake_response(chunks: list[bytes]):
        chunks = iter(chunks + [b""])
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": "30"}
        response.read.side_effect = lambda n: next(chunks)
        return response

    def test_download_stops_when_cancelled(self):
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", 30, "u")
        response = self.fake_response([b"a" * 10, b"b" * 10, b"c" * 10])
        seen = []
        with mock.patch.object(updater, "fetch_checksum", return_value="0" * 64), mock.patch.object(
            updater, "urlopen", return_value=response
        ):
            with self.assertRaises(updater.UpdateCancelled):
                updater.download_asset(
                    release, self.root / "dl", progress=lambda r, t: seen.append(r), cancelled=lambda: len(seen) >= 2
                )
        self.assertEqual(seen, [10, 20])
        self.assertFalse((self.root / "dl" / release.asset_name).exists())

    def test_download_verifies_checksum(self):
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", 30, "u")
        body = b"a" * 10 + b"b" * 10 + b"c" * 10
        good = hashlib.sha256(body).hexdigest()
        with mock.patch.object(updater, "fetch_checksum", return_value=good), mock.patch.object(
            updater, "urlopen", return_value=self.fake_response([body[:10], body[10:20], body[20:]])
        ):
            target = updater.download_asset(release, self.root / "dl")
        self.assertEqual(target.read_bytes(), body)
        with mock.patch.object(updater, "fetch_checksum", return_value="f" * 64), mock.patch.object(
            updater, "urlopen", return_value=self.fake_response([body])
        ):
            with self.assertRaisesRegex(updater.UpdateError, "checksum"):
                updater.download_asset(release, self.root / "dl2")
        self.assertFalse((self.root / "dl2" / release.asset_name).exists())

    def test_fetch_checksum_requires_published_digest(self):
        with self.assertRaisesRegex(updater.UpdateError, "does not publish a checksum"):
            updater.fetch_checksum(self.RELEASE)
        release = updater.ReleaseInfo("0.3.0", "a.zip", "u", None, "r", checksum_url="https://x/a.zip.sha256")
        response = mock.MagicMock(); response.__enter__.return_value = response
        response.read.return_value = (" " + "ab" * 32 + "  a.zip\n").encode()
        with mock.patch.object(updater, "urlopen", return_value=response):
            self.assertEqual(updater.fetch_checksum(release), "ab" * 32)
        response.read.return_value = b"not a digest"
        with mock.patch.object(updater, "urlopen", return_value=response):
            with self.assertRaisesRegex(updater.UpdateError, "not a SHA-256"):
                updater.fetch_checksum(release)

    def test_install_update_cancelled_after_extract_leaves_bundle_alone(self):
        current = self.make_bundle("ChatLab.app", "old")
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", None, "u")
        work = self.root / "work"
        with mock.patch.object(updater, "download_asset", side_effect=lambda *a, **k: work / "x.zip"), mock.patch.object(
            updater, "extract_bundle", side_effect=lambda *a, **k: self.make_bundle("work/unpacked/ChatLab.app", "new")
        ):
            with self.assertRaises(updater.UpdateCancelled):
                updater.install_update(release, current, work_dir=work, cancelled=lambda: True)
        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "old")
        self.assertFalse(work.exists())

    def test_install_update_respects_begin_swap_veto(self):
        current = self.make_bundle("ChatLab.app", "old")
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", None, "u")
        work = self.root / "work"
        with mock.patch.object(updater, "download_asset", side_effect=lambda *a, **k: work / "x.zip"), mock.patch.object(
            updater, "extract_bundle", side_effect=lambda *a, **k: self.make_bundle("work/unpacked/ChatLab.app", "new")
        ):
            with self.assertRaises(updater.UpdateCancelled):
                updater.install_update(release, current, work_dir=work, begin_swap=lambda: False)
        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "old")

    def test_install_update_wires_the_steps(self):
        current = self.make_bundle("ChatLab.app", "old")
        release = updater.ReleaseInfo("0.3.0", "ChatLab-macos-arm64.zip", "https://x/arm.zip", None, "u")
        work = self.root / "work"

        def fake_download(rel, dest, progress, cancelled=None):
            dest.mkdir(parents=True, exist_ok=True)
            progress(5, 10)
            return dest / rel.asset_name

        def fake_extract(archive, dest, cancelled=None):
            return self.make_bundle("work/unpacked/ChatLab.app", "new")

        seen = []
        with mock.patch.object(updater, "download_asset", side_effect=fake_download), mock.patch.object(
            updater, "extract_bundle", side_effect=fake_extract
        ):
            updater.install_update(
                release,
                current,
                work_dir=work,
                progress=lambda r, t: seen.append((r, t)),
                begin_swap=lambda: seen.append("swap") or True,
            )

        self.assertEqual(seen, [(5, 10), "swap"])
        self.assertEqual((current / "Contents" / "MacOS" / "ChatLab").read_text(), "new")
        self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
