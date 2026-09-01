import unittest

import charts


def metrics(count: int, scored: bool = True) -> list[dict]:
    return [
        {"position": index + 1, "surprise_bits": float(index % 5), "scored": scored}
        for index in range(count)
    ]


class SurpriseChartTests(unittest.TestCase):
    def test_a_single_token_has_nothing_to_trace(self):
        self.assertEqual(charts.surprise_chart(metrics(1)), charts.EMPTY_CHART)
        self.assertEqual(charts.surprise_chart([]), charts.EMPTY_CHART)

    def test_unscored_tokens_are_left_out(self):
        self.assertEqual(charts.surprise_chart(metrics(30, scored=False)), charts.EMPTY_CHART)

    def test_short_responses_draw_one_point_per_token(self):
        svg = charts.surprise_chart(metrics(12))
        self.assertEqual(svg.count("<title>"), 12)
        self.assertIn("token 12", svg)
        self.assertNotIn("grouped into", svg)

    def test_the_peak_label_points_at_the_peak_token_not_the_bin_end(self):
        spiked = metrics(500)
        spiked[6]["surprise_bits"] = 9.0
        svg = charts.surprise_chart(spiked)
        self.assertIn("peak 9.0 bits at token 7", svg)

    def test_long_responses_are_binned_and_say_so(self):
        svg = charts.surprise_chart(metrics(5000))
        self.assertLessEqual(svg.count("<title>"), charts.MAX_BINS)
        self.assertIn("grouped into", svg)
        self.assertIn("5,000 tokens", svg)

    def test_titles_are_escaped(self):
        svg = charts.surprise_chart(metrics(4), title="Bits & pieces")
        self.assertIn("Bits &amp; pieces", svg)
        self.assertNotIn("Bits & pieces", svg)


class SummaryTileTests(unittest.TestCase):
    def test_empty_summary_renders_a_placeholder(self):
        self.assertIn("No scored tokens", charts.summary_tiles({}))

    def test_tiles_report_the_headline_numbers(self):
        summary = {
            "token_count": 4,
            "perplexity": 12.5,
            "mean_surprise_bits": 3.64,
            "median_surprise_bits": 3.0,
            "total_surprise_bits": 14.6,
            "mean_entropy_bits": 4.2,
            "top1_share": 0.5,
            "top5_share": 0.75,
            "peak_surprise_bits": 9.0,
            "peak_position": 3,
        }
        html = charts.summary_tiles(summary)
        self.assertIn("12.5", html)
        self.assertIn("50%", html)
        self.assertIn("perplexity", html)


if __name__ == "__main__":
    unittest.main()
