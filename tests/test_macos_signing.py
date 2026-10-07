"""Exercise signing selection without installing dependencies or accessing keys."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_macos_app.sh"
IDENTITY = "Developer ID Application: Julius Simonelli (3CJQ95F6MT)"
FINGERPRINT = "A" * 40
OTHER_FINGERPRINT = "B" * 40


class MacOSSigningTest(unittest.TestCase):
    def run_build(self, identities, override=None, signing_status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            log = root / "calls"
            for name, body in {
                "security": 'printf "%s\\n" "$TEST_IDENTITIES"',
                "python": 'echo python >> "$TEST_LOG"',
                "pyinstaller": 'echo pyinstaller >> "$TEST_LOG"',
                "codesign": (
                    'echo "codesign $*" >> "$TEST_LOG"\n'
                    'exit "$TEST_SIGNING_STATUS"'
                ),
            }.items():
                stub = bin_dir / name
                stub.write_text("#!/bin/sh\n" + body + "\n")
                stub.chmod(0o755)
            env = dict(os.environ)
            env.pop("CHATLAB_CODESIGN_IDENTITY", None)
            env.update(
                PATH=str(bin_dir) + os.pathsep + env["PATH"],
                CHATLAB_DESKTOP_VENV=str(root),
                TEST_IDENTITIES=identities,
                TEST_LOG=str(log),
                TEST_SIGNING_STATUS=str(signing_status),
            )
            if override is not None:
                env["CHATLAB_CODESIGN_IDENTITY"] = override
            result = subprocess.run(
                ["/bin/sh", str(BUILD_SCRIPT)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, log.read_text() if log.exists() else ""

    def test_default_signs_with_developer_id_and_verifies(self):
        # security lists a trusted identity twice; that is not ambiguity.
        listing = (
            f'  1) {FINGERPRINT} "{IDENTITY}"\n'
            f'  Valid identities only\n  1) {FINGERPRINT} "{IDENTITY}"'
        )
        result, calls = self.run_build(listing)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--sign {FINGERPRINT}", calls)
        self.assertIn("codesign --verify --strict", calls)
        self.assertLess(calls.index("pyinstaller"), calls.index("codesign"))

    def test_missing_default_stops_before_building(self):
        result, calls = self.run_build(
            f'  1) {OTHER_FINGERPRINT} "ChatLab Local"'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No code signing identity", result.stderr)
        self.assertEqual(calls, "")

    def test_ambiguous_default_stops_before_building(self):
        result, calls = self.run_build(
            f'  1) {FINGERPRINT} "{IDENTITY}"\n'
            f'  2) {OTHER_FINGERPRINT} "{IDENTITY}"'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Several code signing identities", result.stderr)
        self.assertEqual(calls, "")

    def test_fingerprint_override_resolves_duplicate_names(self):
        result, calls = self.run_build(
            f'  1) {FINGERPRINT} "{IDENTITY}"\n'
            f'  2) {OTHER_FINGERPRINT} "{IDENTITY}"',
            override=OTHER_FINGERPRINT.lower(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--sign {OTHER_FINGERPRINT}", calls)

    def test_name_override_is_supported(self):
        result, calls = self.run_build(
            f'  1) {OTHER_FINGERPRINT} "Local Developer"',
            override="Local Developer",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--sign {OTHER_FINGERPRINT}", calls)

    def test_missing_override_does_not_fall_back_to_default(self):
        result, calls = self.run_build(
            f'  1) {FINGERPRINT} "{IDENTITY}"', override="Missing certificate"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, "")

    def test_signing_failure_fails_the_build(self):
        result, calls = self.run_build(
            f'  1) {FINGERPRINT} "{IDENTITY}"', signing_status=1
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("codesign --force", calls)
        self.assertNotIn("Built ", result.stdout)
