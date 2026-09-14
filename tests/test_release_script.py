"""The release script, run end to end against a scratch repository.

``tests/release_harness.sh`` stands a bare repository up in a temporary
directory, stubs out ``gh``, the bundle build and the test run, and drives
``scripts/release.sh`` through a normal release, a resumed one, a lost push
race and the checks that refuse to start. It needs the BSD ``sed``,
``plutil`` and ``stat`` the script itself uses, so it runs on macOS only.
"""

import subprocess
import sys
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "release_harness.sh"


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
