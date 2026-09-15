import math
import re
import unittest

import numpy as np

from ui.styles import CSS

from token_metrics import (
    COLOR_SCALES,
    DIVERGING_FILLS,
    SEQUENTIAL_FILLS,
    UNSCORED_FILL,
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


class SkipTopChoiceTests(unittest.TestCase):
    """The first choice can be refused where the model was not sure of it."""

    # The model is torn: 0.4 against 0.35 against the rest.
    UNSURE = np.log(np.array([0.4, 0.35, 0.15, 0.1]))
    # The model has all but decided.
    SURE = np.log(np.array([0.9, 0.06, 0.03, 0.01]))
    # Decided enough to clear a threshold of 0.6, but not by much.
    MOSTLY_SURE = np.log(np.array([0.7, 0.1, 0.1, 0.1]))

    def test_zero_leaves_the_distribution_alone(self):
        np.testing.assert_array_almost_equal(
            sampling_probabilities(
                self.UNSURE, temperature=1, top_p=1, top_k=0, skip_top_below=0.0
            ),
            sampling_probabilities(self.UNSURE, temperature=1, top_p=1, top_k=0),
        )

    def test_an_unsure_first_choice_is_refused(self):
        result = sampling_probabilities(
            self.UNSURE, temperature=1, top_p=1, top_k=0, skip_top_below=0.6
        )

        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        # The rest keep their proportions; only the missing mass is shared out.
        self.assertAlmostEqual(result[1] / result[2], 0.35 / 0.15)

    def test_a_confident_first_choice_stands(self):
        # This is what keeps the text together: the model is sure of the rest
        # of a word it has started, of a closing bracket, of the space after
        # a comma, and those positions sample as they always did.
        result = sampling_probabilities(
            self.SURE, temperature=1, top_p=1, top_k=0, skip_top_below=0.6
        )

        self.assertAlmostEqual(result[0], 0.9)

    def test_the_threshold_is_read_before_temperature_reshapes_anything(self):
        # A high temperature flattens a distribution and a low one sharpens
        # it, so a threshold read from the reshaped one would mean something
        # different at every setting of the other slider. Both of these
        # decide the other way once temperature has had its way with them:
        # 0.7 falls to 0.47 at 1.95, and 0.4 climbs to 0.62 at 0.25.
        kept = sampling_probabilities(
            self.MOSTLY_SURE, temperature=1.95, top_p=1, top_k=0, skip_top_below=0.6
        )
        skipped = sampling_probabilities(
            self.UNSURE, temperature=0.25, top_p=1, top_k=0, skip_top_below=0.6
        )

        self.assertGreater(kept[0], 0)
        self.assertEqual(skipped[0], 0)

    def test_a_zero_temperature_takes_the_second_choice_instead(self):
        # Deterministic, and off the greedy path: the same prompt gives the
        # same reply every time, and it is not the reply greedy decoding
        # would have written.
        result = sampling_probabilities(
            self.UNSURE, temperature=0, top_p=1, top_k=0, skip_top_below=0.6
        )

        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 0.0, 0.0]))

    def test_the_skip_runs_before_top_k_so_k_candidates_remain(self):
        # A reader who asked for two candidates gets two of them - the second
        # and third choices - rather than one survivor of the first two.
        result = sampling_probabilities(
            self.UNSURE, temperature=1, top_p=1, top_k=2, skip_top_below=0.6
        )

        self.assertEqual(result[0], 0)
        self.assertGreater(result[1], 0)
        self.assertGreater(result[2], 0)
        self.assertEqual(result[3], 0)

    def test_a_first_choice_with_nothing_behind_it_is_kept(self):
        # Skipping it would leave nothing to sample at all.
        one = normalize_log_probabilities(np.array([0.0]))
        np.testing.assert_array_equal(
            sampling_probabilities(one, temperature=1, top_p=1, top_k=0, skip_top_below=1.0),
            np.array([1.0]),
        )
        only_possible = normalize_log_probabilities(np.array([0.0, -np.inf]))
        np.testing.assert_array_equal(
            sampling_probabilities(
                only_possible, temperature=1, top_p=1, top_k=0, skip_top_below=1.0
            ),
            np.array([1.0, 0.0]),
        )

    def test_one_refuses_a_first_choice_the_model_left_any_doubt_about(self):
        # The top of the slider, where all that saves a first choice is the
        # model having been certain of it.
        result = sampling_probabilities(
            self.SURE, temperature=1, top_p=1, top_k=0, skip_top_below=1.0
        )

        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_a_certain_first_choice_survives_even_the_top_of_the_slider(self):
        # Nothing was in doubt, so there was nothing for the skip to resolve.
        certain = normalize_log_probabilities(np.array([20.0, -20.0, -20.0]))
        result = sampling_probabilities(
            certain, temperature=1, top_p=1, top_k=0, skip_top_below=1.0
        )

        self.assertAlmostEqual(result[0], 1.0)


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


# The strip paints these fills behind dark body text, and a reader has to tell
# two of them apart while they sit side by side in arbitrary order. Both of
# those are arithmetic, so they are checked rather than eyeballed. The ink is
# #0b0b0b, pinned in ui/styles.py.
STRIP_INK = "#0b0b0b"
MIN_INK_CONTRAST = 4.5
MIN_SEPARATION = 8.0

# Machado, Oliveira and Fernandes (2009) at full severity, the simulation the
# separation floor is calibrated against.
COLOR_VISION = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}


def _linear(fill: str) -> tuple[float, float, float]:
    channels = [int(fill[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    return tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    )


def _contrast(fill: str, other: str) -> float:
    weights = (0.2126, 0.7152, 0.0722)
    luminances = sorted(
        sum(weight * channel for weight, channel in zip(weights, _linear(color)))
        for color in (fill, other)
    )
    return (luminances[1] + 0.05) / (luminances[0] + 0.05)


def _oklab(linear: tuple[float, float, float]) -> tuple[float, float, float]:
    red, green, blue = linear
    long = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
    medium = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (
        1 / 3
    )
    short = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (1 / 3)
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def _separation(fill: str, other: str, vision: str = "") -> float:
    def seen(color: str) -> tuple[float, float, float]:
        linear = _linear(color)
        if not vision:
            return _oklab(linear)
        matrix = COLOR_VISION[vision]
        return _oklab(
            tuple(
                min(1.0, max(0.0, sum(w * c for w, c in zip(row, linear))))
                for row in matrix
            )
        )

    return 100 * math.dist(seen(fill), seen(other))


class PaletteTests(unittest.TestCase):
    def test_every_fill_keeps_the_strip_text_readable(self):
        fills = {*SEQUENTIAL_FILLS, *DIVERGING_FILLS, UNSCORED_FILL}
        for fill in sorted(fills):
            with self.subTest(fill=fill):
                self.assertGreaterEqual(
                    _contrast(fill, STRIP_INK), MIN_INK_CONTRAST, fill
                )

    def test_rank_fills_stay_apart_in_any_pairing(self):
        # Any two buckets can end up neighbours, because tokens arrive in the
        # order the model wrote them, so every pair is checked and not just
        # the adjacent ones.
        fills = [*SEQUENTIAL_FILLS, UNSCORED_FILL]
        for index, fill in enumerate(fills):
            for other in fills[index + 1 :]:
                for vision in ("", *COLOR_VISION):
                    with self.subTest(pair=(fill, other), vision=vision or "normal"):
                        self.assertGreaterEqual(
                            _separation(fill, other, vision), MIN_SEPARATION
                        )

class StripInkTests(unittest.TestCase):
    def test_dark_ink_is_pinned_for_every_strip(self):
        # The fills are only readable under dark text, and extensions mount
        # their own strips off the same palette, so the rule that pins the ink
        # must not be scoped to the ids the app happens to ship today.
        stylesheet = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
        rules = [
            rule.partition("{")
            for rule in stylesheet.split("}")
            if ".textspan.hl" in rule.split("{")[0]
        ]
        self.assertTrue(rules, "no rule pins ink on highlighted spans")
        for selector, _, body in rules:
            with self.subTest(selector=selector.strip()):
                self.assertNotIn("#", selector)
                self.assertIn(STRIP_INK, body)


if __name__ == "__main__":
    unittest.main()
