import unittest

import numpy as np

from token_metrics import (
    COLOR_SCALES,
    UNSCORED_LABEL,
    build_metric,
    category_for,
    entropy_bits,
    normalize_log_probabilities,
    rank_category,
    sampling_probabilities,
    summarize,
    top_margin,
    UNSCORED_BEYOND_LIMIT,
    UNSCORED_FIRST_TOKEN,
    unscored_metric,
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

    def test_metric_measures_the_distribution_it_chose_from(self):
        raw = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        sampled = sampling_probabilities(raw, temperature=1, top_p=1, top_k=2)
        metric = build_metric(
            position=1,
            token_id=0,
            token_text=" first",
            fallback_text="first",
            raw_log_probabilities=raw,
            sampled_probabilities=sampled,
            decode_token=str,
        )
        self.assertAlmostEqual(metric.entropy_bits, entropy_bits(raw))
        self.assertAlmostEqual(metric.top1_margin, 0.2)
        # Top-k dropped the tail, so the survivors were pushed upwards.
        self.assertGreater(metric.sampling_shift_bits, 0)
        self.assertEqual(metric.segment, "response")
        self.assertTrue(metric.scored)


class DistributionShapeTests(unittest.TestCase):
    def test_uniform_distribution_has_log2_entropy(self):
        raw = normalize_log_probabilities(np.zeros(8))
        self.assertAlmostEqual(entropy_bits(raw), 3.0)

    def test_impossible_tokens_do_not_break_entropy(self):
        raw = normalize_log_probabilities(np.array([0.0, 0.0, -np.inf]))
        self.assertAlmostEqual(entropy_bits(raw), 1.0)

    def test_margin_is_the_gap_between_the_first_two_choices(self):
        self.assertAlmostEqual(top_margin(np.array([0.5, 0.3, 0.2])), 0.2)

    def test_single_token_vocabulary_has_no_gap(self):
        self.assertAlmostEqual(top_margin(np.array([1.0])), 1.0)


class ColorScaleTests(unittest.TestCase):
    def sample(self, **overrides) -> dict:
        metric = {
            "raw_rank": 1,
            "surprise_bits": 0.2,
            "entropy_bits": 0.2,
            "sampling_shift_bits": 0.0,
            "scored": True,
        }
        return metric | overrides

    def test_every_scale_labels_a_token(self):
        for name, scale in COLOR_SCALES.items():
            with self.subTest(scale=name):
                label = category_for(self.sample(), name)
                self.assertIn(label, scale.labels)
                self.assertIn(label, scale.color_map)

    def test_buckets_split_at_their_edges(self):
        surprise = COLOR_SCALES["Surprise"]
        self.assertEqual(surprise.bucket(0.99), "Under 1 bit")
        self.assertEqual(surprise.bucket(1.0), "1–3 bits")
        self.assertEqual(surprise.bucket(99.0), "Over 10 bits")

    def test_sampling_shift_diverges_around_zero(self):
        shift = COLOR_SCALES["Sampling shift"]
        self.assertEqual(shift.bucket(-4.0), "Strongly suppressed")
        self.assertEqual(shift.bucket(0.0), "Unchanged")
        self.assertEqual(shift.bucket(4.0), "Strongly boosted")

    def test_unpredicted_tokens_are_labelled_separately(self):
        metric = unscored_metric(
            position=1, token_id=7, token_text="<s>", fallback_text="<s>"
        ).to_dict()
        self.assertEqual(metric["segment"], "prompt")
        self.assertFalse(metric["scored"])
        for name in COLOR_SCALES:
            with self.subTest(scale=name):
                self.assertEqual(category_for(metric, name), UNSCORED_LABEL)

    def test_unpredicted_tokens_record_why_they_were_not_scored(self):
        first = unscored_metric(
            position=1, token_id=7, token_text="<s>", fallback_text="<s>"
        ).to_dict()
        capped = unscored_metric(
            position=4,
            token_id=9,
            token_text=" the",
            fallback_text=" the",
            reason=UNSCORED_BEYOND_LIMIT,
        ).to_dict()

        self.assertEqual(first["unscored_reason"], UNSCORED_FIRST_TOKEN)
        self.assertEqual(capped["unscored_reason"], UNSCORED_BEYOND_LIMIT)


class SummaryTests(unittest.TestCase):
    def metrics(self) -> list[dict]:
        return [
            {"position": 1, "surprise_bits": 1.0, "entropy_bits": 2.0, "raw_rank": 1, "scored": True},
            {"position": 2, "surprise_bits": 3.0, "entropy_bits": 4.0, "raw_rank": 9, "scored": True},
            {"position": 3, "surprise_bits": 0.0, "entropy_bits": 0.0, "raw_rank": 0, "scored": False},
        ]

    def test_summary_ignores_unscored_tokens(self):
        summary = summarize(self.metrics())
        self.assertEqual(summary["token_count"], 2)
        self.assertAlmostEqual(summary["mean_surprise_bits"], 2.0)
        self.assertAlmostEqual(summary["perplexity"], 4.0)
        self.assertAlmostEqual(summary["total_surprise_bits"], 4.0)
        self.assertAlmostEqual(summary["mean_entropy_bits"], 3.0)
        self.assertAlmostEqual(summary["top1_share"], 0.5)
        self.assertEqual(summary["peak_position"], 2)

    def test_empty_summary_is_safe_to_render(self):
        self.assertEqual(summarize([])["token_count"], 0)
        self.assertEqual(summarize([])["perplexity"], 0.0)


if __name__ == "__main__":
    unittest.main()
