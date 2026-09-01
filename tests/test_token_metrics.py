import unittest

import numpy as np

from token_metrics import (
    build_metric,
    normalize_log_probabilities,
    rank_category,
    sampling_probabilities,
)


class TokenMetricTests(unittest.TestCase):
    def test_log_probabilities_are_normalized(self):
        result = normalize_log_probabilities(np.array([3.0, 2.0, -4.0]))
        self.assertAlmostEqual(float(np.exp(result).sum()), 1.0)

    def test_negative_infinity_can_represent_impossible_token(self):
        raw = normalize_log_probabilities(np.array([2.0, 1.0, -np.inf]))
        result = sampling_probabilities(raw, temperature=1, top_p=1, top_k=0)
        self.assertEqual(result[2], 0)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_greedy_distribution_selects_top_token(self):
        raw = normalize_log_probabilities(np.array([1.0, 5.0, 2.0]))
        result = sampling_probabilities(raw, temperature=0, top_p=0.5, top_k=1)
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 0.0]))

    def test_top_k_filters_lower_ranked_tokens(self):
        raw = normalize_log_probabilities(np.array([4.0, 3.0, 2.0, 1.0]))
        result = sampling_probabilities(raw, temperature=1, top_p=1, top_k=2)
        self.assertGreater(result[0], 0)
        self.assertGreater(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 0)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_top_p_keeps_crossing_token(self):
        raw = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        result = sampling_probabilities(raw, temperature=1, top_p=0.6, top_k=0)
        self.assertGreater(result[0], 0)
        self.assertGreater(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 0)

    def test_metric_reports_chosen_token_rank(self):
        raw = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        sampled = sampling_probabilities(raw, temperature=1, top_p=1, top_k=0)
        metric = build_metric(
            position=1,
            token_id=2,
            token_text=" third",
            fallback_text="third",
            raw_log_probabilities=raw,
            sampled_probabilities=sampled,
            decode_token=lambda token_id: str(token_id),
        )
        self.assertEqual(metric.raw_rank, 3)
        self.assertAlmostEqual(metric.raw_probability, 0.15)
        self.assertEqual(metric.category, "Top 5")

    def test_rank_categories_cover_boundaries(self):
        self.assertEqual(rank_category(1), "Top choice")
        self.assertEqual(rank_category(5), "Top 5")
        self.assertEqual(rank_category(20), "Top 20")
        self.assertEqual(rank_category(100), "Rank 21–100")
        self.assertEqual(rank_category(101), "Rank 101+")


if __name__ == "__main__":
    unittest.main()
