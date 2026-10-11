"""Steering vector files (chatlab-steering-1) as earlier versions accepted and wrote them.

Each file under tests/fixtures/saved/steering is frozen (see its generator).
They are imported through Chat's "Import vector" handler, ``import_vector``,
and the vector it stores on the conversation is checked number by number.
"""

from pathlib import Path
import unittest
from unittest import mock

from chatlab import steering
from chatlab.conversation import new_forks
from chatlab.ui.steering import download_extracted, import_vector

from saved_formats_support import SAVED, sandbox

STEERING = SAVED / "steering"


class SavedVectorCase(unittest.TestCase):
    fixture = ""

    def setUp(self):
        self.box = sandbox(self)
        self.path = self.box.copy(STEERING / self.fixture)
        forks, value, enabled, strength, layer, status = import_vector(str(self.path), new_forks())
        self.forks, self.value, self.status = forks, value, status
        self.controls = (enabled["value"], strength["value"], layer["value"])
        self.vector = steering.expand(self.value)

    def assert_round_trip(self):
        """Today's download writes the same vector, which imports to the same reading."""

        layer = self.vector["layer"]
        extraction = {
            "model_id": self.vector["model_id"],
            "layers": [[0.0] * len(self.vector["vector"]) for _ in range(layer + 2)],
            "stats": [{} for _ in range(layer + 2)],
        }
        extraction["layers"][layer] = self.vector["vector"]
        directory = self.box.root / "download"
        directory.mkdir()
        # Isolate only the output directory; exercise the real layer selection and writer.
        with mock.patch("chatlab.ui.steering.tempfile.mkdtemp", return_value=str(directory)):
            target = Path(download_extracted(extraction, layer))
        self.assertEqual(target.parent, directory)
        again = steering.read_vector(str(target))
        self.assertEqual(again, self.vector)
        _, value, *_ = import_vector(str(target), new_forks())
        self.assertEqual(value, self.value)


class HandWrittenTests(SavedVectorCase):
    """As 392ad4d's README described: format, strength and enabled left out."""

    fixture = "01-hand-written-392ad4d.json"

    def test_defaults_and_values(self):
        self.assertEqual(self.vector, {
            "format": "chatlab-steering-1", "model_id": "allenai/Olmo-3-7B-Think", "layer": 0,
            "strength": 1.0, "enabled": True, "vector": [1.0, -2.0, 0.5, 0.0, 3.25],
        })
        self.assertEqual(self.controls, (True, 1.0, 0))
        self.assertEqual(self.value["width"], 5)
        self.assertEqual(self.forks["sampling"]["Main"]["steering"], self.value)
        self.assertTrue(self.status.startswith("Enabled · allenai/Olmo-3-7B-Think · layer 0 · 5 entries"))

    def test_round_trip(self):
        self.assert_round_trip()


class ExtractedTests(SavedVectorCase):
    """be75ba8 (#102): an extracted direction as the Steering panel downloads it."""

    fixture = "02-extracted-be75ba8.json"

    def test_values(self):
        self.assertEqual(self.vector["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(self.vector["layer"], 1)
        self.assertEqual(self.vector["vector"], [2.75, -1.75, -0.08333333333333334, 0.6666666666666666])
        self.assertEqual(self.controls, (True, 1.0, 1))

    def test_today_extracts_the_same_direction(self):
        # The file is the layer-1 difference in means of the generator's
        # examples; today's extraction of those examples gives the same row.
        positive = [
            [[1.0, 0.0, 0.5, 0.0], [2.0, -1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]],
            [[1.5, 0.5, 0.5, 0.0], [2.5, -0.5, 0.5, 0.5], [0.5, 0.0, 1.0, 1.5]],
            [[0.5, 0.0, 1.0, 0.5], [1.5, -1.5, 0.0, 1.0], [0.0, 0.5, 1.5, 1.0]],
        ]
        negative = [
            [[0.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5]],
            [[0.5, 1.0, 0.5, 0.5], [-0.5, 0.5, 0.5, 0.0], [0.5, 0.0, 0.5, 0.5]],
        ]
        directions, _ = steering.contrast_directions(positive, negative)
        self.assertEqual(steering.vector_from("Qwen/Qwen3-0.6B", 1, directions[1]), self.vector)

    def test_round_trip(self):
        self.assert_round_trip()


if __name__ == "__main__":
    unittest.main()
