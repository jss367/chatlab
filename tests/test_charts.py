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

    def test_the_left_axis_names_the_first_scored_token(self):
        # A tokenizer that adds no BOS leaves the sequence's first token
        # unscored when no context precedes it, so the leftmost plotted point
        # is token 2 and the axis has to say so.
        run = metrics(12)
        run[0]["scored"] = False
        svg = charts.surprise_chart(run)

        self.assertIn("token 2</text>", svg)
        self.assertNotIn("token 1</text>", svg)
        self.assertEqual(svg.count("<title>"), 11)

    def test_the_left_axis_groups_thousands(self):
        run = metrics(5000)
        for metric in run[:1500]:
            metric["scored"] = False
        svg = charts.surprise_chart(run)

        self.assertIn("token 1,501</text>", svg)

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


def insight(layers=4, tokens=5, decided_at=2, attention=True):
    # ``decided_at=None`` is a token the model never put first.
    threshold = layers + 1 if decided_at is None else decided_at
    rows = [
        {
            "layer": layer,
            "probability": 0.1 if layer < threshold else 0.9,
            "rank": 3 if layer < threshold else 1,
            "entropy_bits": 2.0,
            "top_id": 7 if layer < threshold else 3,
            "top_text": " cat" if layer < threshold else " dog",
            "top_probability": 0.5,
        }
        for layer in range(layers + 1)
    ]
    weights = []
    if attention:
        for layer in range(layers):
            row = [0.1] * tokens
            row[0] = 0.5
            row[min(layer + 1, tokens - 1)] += 0.3
            weights.append([value / sum(row) for value in row])
    return {
        "index": tokens,
        "token_id": 3,
        "token_text": " dog",
        "layers": rows,
        "tokens": [
            {"index": i, "token_id": i, "text": f"t{i}" if i else "<s>", "fallback": "", "segment": "prompt" if i < 2 else "response"}
            for i in range(tokens)
        ],
        "attention": weights,
        "decided_at": decided_at,
    }


class LogitLensChartTests(unittest.TestCase):
    def test_an_empty_insight_shows_the_hint(self):
        self.assertEqual(charts.logit_lens_chart({}), charts.EMPTY_LENS)

    def test_one_table_row_per_reading_and_the_deciding_layer_is_named(self):
        svg = charts.logit_lens_chart(insight(layers=4, decided_at=2))
        self.assertEqual(svg.count("<tr"), 1 + 5)
        self.assertIn("first choice from layer 2", svg)
        self.assertIn("embeddings", svg)
        self.assertIn("output (layer 4)", svg)
        self.assertIn("&#x27; cat&#x27;", svg)

    def test_a_token_never_chosen_says_so(self):
        svg = charts.logit_lens_chart(insight(decided_at=None))
        self.assertIn("never the first choice", svg)


class AttentionStripTests(unittest.TestCase):
    def test_a_span_per_visible_token_plus_the_predicted_one(self):
        html = charts.attention_strip(insight(tokens=5), 0)
        self.assertEqual(html.count('class="attn-token'), 6)
        self.assertEqual(html.count("attn-query"), 1)
        self.assertEqual(html.count("attn-predicted"), 1)
        self.assertIn("mean of all layers", html)
        self.assertIn("The first token takes 42%", html)

    def test_a_single_layer_can_be_picked(self):
        html = charts.attention_strip(insight(tokens=5, layers=4), 3)
        self.assertIn("layer 3", html)
        self.assertNotIn("mean of all layers", html)

    def test_weights_average_over_layers_when_no_layer_is_picked(self):
        data = insight(tokens=5, layers=4)
        mean = charts.attention_weights(data, 0)
        self.assertEqual(len(mean), 5)
        self.assertAlmostEqual(sum(mean), 1.0)
        self.assertEqual(charts.attention_weights(data, 2), data["attention"][1])
        self.assertEqual(charts.attention_weights(data, 99), mean)

    def test_a_model_without_weights_is_explained(self):
        html = charts.attention_strip(insight(attention=False), 0)
        self.assertIn("did not return attention", html)
        self.assertEqual(charts.attention_strip({}, 0), charts.EMPTY_ATTENTION)
