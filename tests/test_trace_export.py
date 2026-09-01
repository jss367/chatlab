import csv
import io
import json
import unittest

from trace_export import build_trace, trace_to_csv, trace_to_json


def sample_metrics():
    return [
        {
            "position": 1,
            "token_id": 42,
            "text": " hello,\nworld",
            "display_text": " hello,↵\nworld",
            "category": "Top 5",
            "raw_rank": 2,
            "raw_probability": 0.25,
            "sampling_probability": 0.4,
            "surprise_bits": 2.0,
            "probability_mass_above": 0.5,
            "top_candidates": [
                {"token_id": 7, "text": "héllo", "probability": 0.5},
                {"token_id": 42, "text": " hello,\nworld", "probability": 0.25},
            ],
        }
    ]


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        self.trace = build_trace(
            model_id="example/model",
            messages=[{"role": "user", "content": "Say hello"}],
            response="hello,\nworld",
            sampling={
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 100,
                "seed": 42,
            },
            metrics=sample_metrics(),
            generated_at="2026-08-31T12:00:00+00:00",
        )

    def test_json_preserves_context_and_nested_metrics(self):
        result = json.loads(trace_to_json(self.trace))

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["messages"][0]["content"], "Say hello")
        self.assertEqual(result["token_count"], 1)
        self.assertEqual(result["tokens"][0]["top_candidates"][0]["text"], "héllo")

    def test_csv_has_one_row_per_token_and_all_candidates(self):
        rows = list(csv.DictReader(io.StringIO(trace_to_csv(self.trace))))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_id"], "example/model")
        self.assertEqual(rows[0]["text"], " hello,\nworld")
        self.assertEqual(rows[0]["candidate_1_text"], "héllo")
        self.assertEqual(rows[0]["candidate_2_token_id"], "42")
        self.assertEqual(rows[0]["temperature"], "0.8")

    def test_empty_trace_csv_still_has_a_header(self):
        empty = self.trace | {"tokens": [], "token_count": 0}
        rows = list(csv.reader(io.StringIO(trace_to_csv(empty))))

        self.assertEqual(len(rows), 1)
        self.assertIn("position", rows[0])


if __name__ == "__main__":
    unittest.main()
