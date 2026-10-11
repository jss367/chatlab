"""Every file format ChatLab reads back has a frozen file under tests/fixtures that a test reads.

A format marker ("chatlab-<name>-<n>") that appears in chatlab/ and is read back needs at least one committed
fixture carrying it, so a new format version, or a change that stops an old one reading, fails a test. See
tests/fixtures/saved/README.md for how to add one.
"""
import gzip
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures"
MARKER = re.compile(r"""["'](chatlab-[a-z-]+-\d+)["']""")

# Formats ChatLab writes for download and never reads back.
WRITE_ONLY = {
    "chatlab-activation-patching-2",
    "chatlab-hangman-batch-1",
    "chatlab-maze-batch-1",
}


def markers_in_source():
    return {m for path in (ROOT / "chatlab").rglob("*.py") for m in MARKER.findall(path.read_text())}


def formats_in(value):
    if isinstance(value, dict):
        if isinstance(value.get("format"), str):
            yield value["format"]
        for item in value.values():
            yield from formats_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from formats_in(item)


def formats_in_fixtures():
    found = set()
    for path in FIXTURES.rglob("*"):
        if path.name.endswith(".json.gz"):
            found.update(formats_in(json.loads(gzip.decompress(path.read_bytes()))))
        elif path.suffix == ".json":
            found.update(formats_in(json.loads(path.read_text())))
    return found


class SavedFormatCoverageTests(unittest.TestCase):
    def test_every_format_read_back_has_a_frozen_fixture(self):
        missing = sorted(markers_in_source() - WRITE_ONLY - formats_in_fixtures())
        self.assertEqual(missing, [], "Add a frozen fixture for each format under tests/fixtures/saved "
                                      "(see its README), or list it in WRITE_ONLY if nothing reads it back.")

    def test_every_write_only_format_still_exists(self):
        self.assertEqual(sorted(WRITE_ONLY - markers_in_source()), [])


if __name__ == "__main__":
    unittest.main()
