"""The release script, run end to end against a scratch repository.

``tests/release_harness.sh`` stands a bare repository up in a temporary
directory, stubs out ``gh``, the bundle build and the test run, and drives
``scripts/release.sh`` through a normal release, a resumed one, a lost push
race and the checks that refuse to start. It needs the BSD ``sed``,
``plutil`` and ``stat`` the script itself uses, so it runs on macOS only.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "release_harness.sh"


class ReleaseHarnessIsolationTests(unittest.TestCase):
    def run_failed_setup(self, mktemp_script, git_script=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name, script in (("mktemp", mktemp_script), ("git", git_script)):
                if script is not None:
                    path = tools / name
                    path.write_text("#!/bin/sh\n" + script)
                    path.chmod(0o755)
            sentinel = root / "keep.txt"
            sentinel.write_text("original")
            result = subprocess.run(
                ["/bin/bash", str(HARNESS)],
                cwd=root,
                env=os.environ | {
                    "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                    "RELHARNESS_TEST_ROOT": str(root),
                },
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertFalse((root / "chatlab").exists())
            self.assertFalse((root / "scripts").exists())
            self.assertFalse((root / ".git").exists())
            return result

    def test_failed_temporary_directory_creation_never_touches_the_callers_checkout(self):
        result = self.run_failed_setup("exit 1\n")
        self.assertIn("Could not create", result.stderr)

    def test_failed_scratch_clone_never_runs_in_the_callers_checkout(self):
        result = self.run_failed_setup(
            'mkdir -p "$RELHARNESS_TEST_ROOT/scratch"\n'
            'printf "%s\\n" "$RELHARNESS_TEST_ROOT/scratch"\n',
            'if [ "$1" = clone ]; then exit 1; fi\nexit 0\n',
        )
        self.assertNotIn("dirty tree aborts", result.stdout)


@unittest.skipUnless(sys.platform == "darwin", "the release script is macOS only")
class ReleaseScriptTest(unittest.TestCase):
    def test_harness_passes(self):
        result = subprocess.run(
            ["/bin/bash", str(HARNESS)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
