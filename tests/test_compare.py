import json
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import charts
import compare
import settings_sandbox
from model_runtime import LoadedModel, ModelChanged
from test_streaming import EOS_ID, loaded_manager
from ui import compare as controls
from ui import runtime


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def metric(position, token_id, surprise, *, top="a", top_id=1, scored=True, entropy=1.0, rank=1):
    """One token's measurements, in the shape the runtime publishes them."""

    return {
        "position": position,
        "token_id": token_id,
        "text": f"t{token_id}",
        "display_text": f"t{token_id}",
        "category": "Top choice",
        "raw_rank": rank,
        "raw_probability": 0.5,
        "sampling_probability": 0.5,
        "surprise_bits": surprise,
        "probability_mass_above": 0.0,
        "entropy_bits": entropy,
        "top1_margin": 0.1,
        "sampling_shift_bits": 0.0,
        "top_candidates": [{"token_id": top_id, "text": top, "probability": 0.5}],
        "scored": scored,
        "segment": "response",
        "unscored_reason": "",
    }


def run(metrics, *, kind=compare.REPLY, model_id="fake/model", **settings):
    return {
        "kind": kind,
        "model_id": model_id,
        "load_id": f"{model_id}#1",
        "device_name": "CPU",
        "precision": "float32",
        "prompt": "hello",
        "text": "hello",
        "metrics": metrics,
        "settings": {
            "system_prompt": "",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 8,
            "seed": 1,
            "assistant_prefill": "",
            "thinking_mode": "default",
            "steering": None,
        } | settings,
        "seconds": 0.1,
    }


class ReadingTests(unittest.TestCase):
    def test_alignment_stops_at_the_first_different_token(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0), metric(3, 7, 1.0)]
        right = [metric(1, 5, 1.0), metric(2, 9, 1.0), metric(3, 7, 1.0)]
        # The third token matches again, but its context no longer does.
        self.assertEqual(len(compare.align(left, right)), 1)
        self.assertEqual(len(compare.align(left, left)), 3)
        self.assertEqual(compare.align(left, []), [])
        # Within one vocabulary every span is one token against one.
        self.assertTrue(all(span["one_to_one"] for span in compare.align(left, left)))

    def test_span_gaps_and_categories_follow_the_surprise_difference(self):
        left = [metric(1, 5, 1.0, top="a"), metric(2, 6, 2.0, top="a")]
        right = [metric(1, 5, 1.2, top="a"), metric(2, 6, 9.0, top="b")]
        spans = compare.align(left, right)
        self.assertAlmostEqual(spans[0]["surprise_bits"], 0.2)
        self.assertAlmostEqual(spans[1]["surprise_bits"], 7.0)
        self.assertEqual(compare.gap_category(spans[0]["surprise_bits"]), compare.GAP_LABELS[0])
        self.assertEqual(compare.gap_category(spans[1]["surprise_bits"]), compare.GAP_LABELS[-1])
        self.assertEqual(spans[1]["left_top"], "a")
        self.assertEqual(spans[1]["right_top"], "b")

    def test_an_unscored_token_gets_no_gap_and_no_color(self):
        left = [metric(1, 5, 0.0, scored=False), metric(2, 6, 1.0)]
        right = [metric(1, 5, 0.0, scored=False), metric(2, 6, 1.0)]
        spans = compare.align(left, right)
        self.assertFalse(spans[0]["scored"])
        self.assertIsNone(spans[0]["left_surprise"])
        painted = compare.strip(left, spans, "left")
        self.assertEqual(painted[0][1], "Not predicted")
        self.assertEqual(painted[1][1], compare.GAP_LABELS[0])

    def test_tokens_after_the_split_are_painted_as_the_split(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [metric(1, 5, 1.0), metric(2, 9, 1.0)]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["spans"], 1)
        self.assertFalse(reading["complete"])
        painted = compare.strip(left, reading["readings"], "left")
        self.assertEqual([label for _text, label in painted][1], compare.SPLIT_LABEL)
        self.assertIn("parted", compare.headline(reading, run(left), run(right)))

    def test_one_passage_under_two_contexts_is_lined_up_on_its_text(self):
        # The context is encoded with the passage, so a different framing can
        # pull characters into the seam token: the same fixed text comes back
        # as a different set of tokens under one vocabulary.
        def piece(position, token_id, text, surprise=1.0):
            return dict(metric(position, token_id, surprise), text=text, display_text=text)

        framed = [piece(1, 1, "hel", 2.0), piece(2, 2, "lo", 3.0)]
        plain = [piece(1, 7, "hello", 4.0)]
        reading = compare.reading(
            dict(run(framed, kind=compare.MEASUREMENT), prompt="one context",
                 decoded="hello", token_ends=[3, 5]),
            dict(run(plain, kind=compare.MEASUREMENT), prompt="another",
                 decoded="hello", token_ends=[5]),
        )
        # Same vocabulary, so the choice comparison keeps its ID rule...
        self.assertFalse(reading["cross_model"])
        # ...but the alignment is over the characters, not the tokens.
        self.assertEqual(reading["spans"], 1)
        self.assertTrue(reading["complete"])
        self.assertAlmostEqual(reading["readings"][0]["left_surprise"], 5.0)
        self.assertTrue(reading["recut"])
        self.assertIn("cut the passage", " ".join(reading["caveats"]))

    def test_equal_token_counts_over_different_boundaries_are_not_a_match(self):
        # "a" + "bc" against "ab" + "c": two tokens each, one span, and
        # neither the tokens nor their boundaries agree.
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "a"), piece(2, 2, "bc")]
        right = [piece(1, 7, "ab"), piece(2, 8, "c")]
        reading = compare.reading(
            dict(run(left, model_id="a/model"), decoded="abc", token_ends=[1, 3]),
            dict(run(right, model_id="b/model"), decoded="abc", token_ends=[2, 3]),
        )
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["left_shared"], reading["right_shared"])
        self.assertFalse(reading["token_for_token"])
        headline = compare.headline(reading, None, None)
        self.assertNotIn("same 2 tokens", headline)
        self.assertIn("comparable span", headline)
        # And a real one-for-one match still says so.
        same = [piece(1, 1, "a"), piece(2, 2, "bc")]
        matched = compare.reading(run(left), run(same))
        self.assertTrue(matched["token_for_token"])
        self.assertIn("same 2 tokens", compare.headline(matched, None, None))

    def test_whitespace_inputs_are_drawn_rather_than_collapsed(self):
        # The row exists to say the two runs were given different text; two
        # cells reading "—" would conceal exactly that.
        spaces = dict(run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT), text="    ")
        tab = dict(run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT), text="\t")
        rows = compare.configuration_rows(spaces, tab)
        self.assertEqual([row[0] for row in rows], ["Measured text"])
        self.assertNotEqual(rows[0][1], rows[0][2])
        self.assertEqual(rows[0][1], "␠␠␠␠")
        self.assertEqual(rows[0][2], "⇥")
        # Line breaks survive inside ordinary prose; runs of spaces do not.
        self.assertEqual(compare.cell("a\nb"), "a↵b")
        self.assertEqual(compare.cell("a  b"), "a b")
        self.assertEqual(compare.cell(""), "—")

    def test_two_replies_from_one_vocabulary_still_match_on_token_ids(self):
        # Matching on IDs is what keeps a token that merely decodes alike
        # from being subtracted across a divergence that already happened.
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "a"), piece(2, 2, "b")]
        right = [piece(1, 1, "a"), piece(2, 9, "b")]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["spans"], 1)
        self.assertFalse(reading["recut"])

    def test_a_measurement_pair_is_compared_to_the_last_token(self):
        left = [metric(index, index, 1.0) for index in range(1, 5)]
        right = [metric(index, index, 3.0) for index in range(1, 5)]
        reading = compare.reading(
            run(left, kind=compare.MEASUREMENT), run(right, kind=compare.MEASUREMENT)
        )
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["compared"], 4)
        self.assertAlmostEqual(reading["mean_gap_bits"], 2.0)
        self.assertIn("same 4 tokens", compare.headline(reading, None, None))

    def test_top_choice_changes_are_counted_over_the_shared_tokens_only(self):
        left = [metric(1, 5, 1.0, top="a", top_id=1), metric(2, 6, 1.0, top="a", top_id=1)]
        right = [metric(1, 5, 1.0, top="b", top_id=2), metric(2, 7, 1.0, top="z", top_id=3)]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["compared"], 1)
        self.assertEqual(reading["top_choice_changed"], 1)

    def test_two_models_are_lined_up_on_text_not_on_token_ids(self):
        # The same two IDs stand for unrelated text in two vocabularies, so
        # matching on them would subtract one model's reading from another's.
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [dict(metric(1, 5, 4.0), text="other", display_text="other"),
                 metric(2, 6, 1.0)]
        reading = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertTrue(reading["cross_model"])
        self.assertEqual(reading["shared"], 0)
        self.assertIn("different models", compare.headline(reading, None, None))
        # Matching IDs in one model still count, and the caveat stays away.
        # One vocabulary gives one ID one text, so the fixture does too.
        echo = [metric(1, 5, 4.0), metric(2, 6, 1.0)]
        same = compare.reading(run(left), run(echo))
        self.assertFalse(same["cross_model"])
        self.assertEqual(same["shared"], 2)
        self.assertNotIn("different models", compare.headline(same, None, None))

    def test_matching_ids_that_decoded_differently_do_not_count_as_shared(self):
        # Two checkpoints can share every token and still register their
        # markers differently, and a run hides its special tokens when it
        # decodes — so one ID can be characters in one run and nothing in
        # the other. The fingerprint is told to look for that, and this is
        # the backstop for whatever it was not told about.
        left = [metric(1, 5, 1.0)]
        right = [dict(metric(1, 5, 1.0), text="", display_text="")]
        reading = compare.reading(
            dict(run(left), tokenizer="same"), dict(run(right), tokenizer="same")
        )
        self.assertFalse(reading["cross_model"])
        self.assertEqual(reading["spans"], 0)

    def test_two_tokenizers_that_cut_a_passage_differently_still_line_up(self):
        def piece(position, token_id, text, surprise=1.0):
            return dict(metric(position, token_id, surprise), text=text, display_text=text)

        # "hel" + "lo" against "hello": the same characters, cut differently.
        left = [piece(1, 1, "hel", 2.0), piece(2, 2, "lo", 3.0), piece(3, 3, "!", 1.0)]
        right = [piece(1, 9, "hello", 4.0), piece(2, 8, "!", 1.5)]
        reading = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertTrue(reading["cross_model"])
        self.assertEqual(reading["spans"], 2)
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["left_shared"], 3)
        self.assertEqual(reading["right_shared"], 2)
        spans = reading["readings"]
        self.assertEqual(spans[0]["text"], "hello")
        self.assertFalse(spans[0]["one_to_one"])
        # Total bits over the same characters is the question both models
        # were asked; 2 + 3 against 4.
        self.assertAlmostEqual(spans[0]["left_surprise"], 5.0)
        self.assertAlmostEqual(spans[0]["right_surprise"], 4.0)
        self.assertAlmostEqual(spans[0]["surprise_bits"], 1.0)
        # A span of several tokens against one has no first choices to pair.
        self.assertEqual(spans[0]["left_top"], "")
        self.assertTrue(spans[1]["one_to_one"])
        # Both of A's tokens for that span take the span's color.
        painted = compare.strip(left, spans, "left")
        self.assertEqual(painted[0][1], painted[1][1])
        self.assertEqual(len(compare.strip(right, spans, "right")), 2)

    def test_a_character_split_across_tokens_does_not_look_like_a_split(self):
        # A byte-level tokenizer decodes each half of a character as the
        # replacement character, so the standalone decodes say the two runs
        # never agreed. The recorded offsets say they read the same thing.
        def half(position, token_id, end):
            return dict(
                metric(position, token_id, 1.0), text="\ufffd", display_text="\ufffd"
            ), end

        a_one, a_end_one = half(1, 1, 0)
        a_two, a_end_two = half(2, 2, 1)
        left = [a_one, a_two]
        right = [dict(metric(1, 9, 2.0), text="é", display_text="é")]
        left_run = dict(
            run(left, ), decoded="é", token_ends=[0, 1],
        )
        right_run = dict(
            run(right, model_id="other/model"), decoded="é", token_ends=[1],
        )
        reading = compare.reading(left_run, right_run)
        self.assertEqual(reading["spans"], 1)
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["readings"][0]["text"], "é")
        # Without the recording there is nothing but the standalone decodes,
        # and the two runs do look like two different passages.
        bare = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertEqual(bare["spans"], 0)

    def test_token_ends_prefer_the_recording_over_standalone_decodes(self):
        metrics = [
            dict(metric(1, 1, 1.0), text="\ufffd"),
            dict(metric(2, 2, 1.0), text="\ufffd"),
        ]
        self.assertEqual(compare.token_ends(metrics, [0, 1]), [0, 1])
        # A recording that does not describe these tokens is not used.
        self.assertEqual(compare.token_ends(metrics, [0]), [1, 2])
        self.assertEqual(compare.token_ends(metrics, None), [1, 2])

    def test_a_zero_width_token_does_not_close_an_empty_span(self):
        # The first byte of a split character decodes to nothing, so both
        # sides can stand level without having covered anything. Closing
        # there would compare those tokens separately and call the two runs
        # agreed before the bytes that follow decode differently.
        def piece(position, token_id, surprise=1.0):
            return metric(position, token_id, surprise)

        left = [piece(1, 1, 2.0), piece(2, 2, 3.0), piece(3, 3, 1.0)]
        right = [piece(1, 9, 4.0), piece(2, 8, 1.0)]
        reading = compare.reading(
            # A's first token covers nothing; both runs reach "é" together.
            dict(run(left), decoded="é!", token_ends=[0, 1, 2]),
            dict(run(right, model_id="other/model"), decoded="é!", token_ends=[1, 2]),
        )
        self.assertEqual(reading["spans"], 2)
        spans = reading["readings"]
        self.assertEqual(spans[0]["text"], "é")
        self.assertEqual(spans[0]["left_range"], (0, 2))
        self.assertAlmostEqual(spans[0]["left_surprise"], 5.0)
        self.assertAlmostEqual(spans[0]["right_surprise"], 4.0)
        self.assertTrue(reading["complete"])

    def test_trailing_tokens_that_decode_to_nothing_stay_in_their_span(self):
        # A hidden stop token covers no characters; it belongs to the span it
        # trails rather than to a divergence that never happened.
        left = [metric(1, 1, 1.0), metric(2, 2, 1.0)]
        right = [metric(1, 9, 2.0), metric(2, 8, 1.0)]
        reading = compare.reading(
            dict(run(left), decoded="hi", token_ends=[2, 2]),
            dict(run(right, model_id="other/model"), decoded="hi", token_ends=[2, 2]),
        )
        # They pair off as a span of their own rather than falling past the
        # end of the alignment, which is what matters: a stop token painted
        # as "after the split" would report a divergence that never happened.
        self.assertEqual(reading["spans"], 2)
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["readings"][1]["text"], "")
        self.assertAlmostEqual(reading["readings"][1]["surprise_bits"], 0.0)
        painted = compare.strip(left, reading["readings"], "left")
        self.assertNotIn(compare.SPLIT_LABEL, [label for _text, label in painted])

    def test_a_reply_that_ended_and_one_that_ran_on_still_share_their_text(self):
        # A samples its end marker, B hits the token limit having written
        # exactly the same words. The marker covers no characters, so the two
        # replies did not part — one of them simply stopped.
        def piece(position, token_id, text, surprise=1.0):
            return dict(metric(position, token_id, surprise), text=text, display_text=text)

        stopped = [piece(1, 1, "hello"), piece(2, 99, "")]
        ran_on = [piece(1, 5, "hello")]
        reading = compare.reading(
            dict(run(stopped, model_id="a/model"), decoded="hello", token_ends=[5, 5]),
            dict(run(ran_on, model_id="b/model"), decoded="hello", token_ends=[5]),
        )
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["left_shared"], 2)
        self.assertEqual(reading["right_shared"], 1)
        # One span of shared text, plus a marker that is one run's alone.
        self.assertEqual(reading["spans"], 1)
        self.assertNotIn("parted", compare.headline(reading, None, None))
        # The leftover token is neither a divergence nor a comparison.
        painted = compare.strip(stopped, reading["readings"], "left")
        self.assertEqual(painted[1][1], compare.TRAILING_LABEL)
        self.assertNotEqual(painted[0][1], compare.SPLIT_LABEL)
        # It contributes nothing to the gap figures either.
        self.assertEqual(reading["compared"], 1)

    def test_a_real_divergence_is_still_a_divergence(self):
        # The sweep only runs when the walk ended by exhaustion; tokens after
        # a genuine parting belong to the split, whatever they decode to.
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "hello"), piece(2, 99, "")]
        right = [piece(1, 5, "world")]
        reading = compare.reading(
            dict(run(left, model_id="a/model"), decoded="hello", token_ends=[5, 5]),
            dict(run(right, model_id="b/model"), decoded="world", token_ends=[5]),
        )
        self.assertEqual(reading["spans"], 0)
        self.assertFalse(reading["complete"])
        self.assertEqual(
            compare.strip(left, reading["readings"], "left")[1][1], compare.SPLIT_LABEL
        )

    def test_text_that_genuinely_parts_ends_the_alignment(self):
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "hel"), piece(2, 2, "lo"), piece(3, 3, " there")]
        right = [piece(1, 9, "hello"), piece(2, 8, " world")]
        reading = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertEqual(reading["spans"], 1)
        self.assertFalse(reading["complete"])
        self.assertEqual(compare.strip(left, reading["readings"], "left")[2][1],
                         compare.SPLIT_LABEL)

    def test_only_one_to_one_spans_are_in_the_top_choice_denominator(self):
        def piece(position, token_id, text, surprise=1.0, top="a"):
            return dict(
                metric(position, token_id, surprise, top=top),
                text=text, display_text=text,
            )

        # A 2-to-1 span, then one comparable token whose choice changed.
        # Across two vocabularies the decoded choice is what can be compared.
        left = [piece(1, 1, "hel"), piece(2, 2, "lo"), piece(3, 3, "!", top="a")]
        right = [piece(1, 9, "hello"), piece(2, 8, "!", top="z")]
        reading = compare.reading(
            dict(run(left), decoded="hello!", token_ends=[3, 5, 6]),
            dict(run(right, model_id="other/model"), decoded="hello!", token_ends=[5, 6]),
        )
        self.assertEqual(reading["compared"], 2)
        # Only the second span had two first choices to put side by side.
        self.assertEqual(reading["choices_compared"], 1)
        self.assertEqual(reading["top_choice_changed"], 1)
        self.assertIn("1 of the 1 span", compare.headline(reading, None, None))
        self.assertIn("100%", charts.comparison_tiles(reading))

    def test_no_pairable_span_leaves_the_percentage_unstated(self):
        def piece(position, token_id, text):
            return dict(metric(position, token_id, 1.0), text=text, display_text=text)

        left = [piece(1, 1, "hel"), piece(2, 2, "lo")]
        right = [piece(1, 9, "hello")]
        reading = compare.reading(
            dict(run(left), decoded="hello", token_ends=[3, 5]),
            dict(run(right, model_id="other/model"), decoded="hello", token_ends=[5]),
        )
        self.assertEqual(reading["choices_compared"], 0)
        self.assertIn("no first choices", compare.headline(reading, None, None))
        self.assertIn("—", charts.comparison_tiles(reading))

    def test_an_empty_decoded_choice_is_a_choice_like_any_other(self):
        # A valid special token can decode to nothing, and "nothing" is a
        # different answer from "something" whichever slot it lands in.
        def piece(top, top_id):
            return dict(metric(1, 5, 1.0, top=top, top_id=top_id), text="x", display_text="x")

        blank, spoken = piece("", 11), piece("word", 12)
        for left, right in ((blank, spoken), (spoken, blank)):
            with self.subTest(left=left["top_candidates"][0]["text"]):
                reading = compare.reading(
                    run([left]), run([right], model_id="other/model")
                )
                self.assertEqual(reading["choices_compared"], 1)
                self.assertEqual(reading["top_choice_changed"], 1)

    def test_two_models_that_both_chose_to_stop_have_not_changed_their_minds(self):
        # A token that decodes to nothing is shown under its vocabulary
        # label, and those labels differ between models. Comparing the labels
        # would call two models that both chose to stop a change of mind.
        def stopped(label):
            held = metric(1, 5, 1.0)
            held["top_candidates"] = [
                {"token_id": 7, "text": label, "probability": 0.9, "raw_text": ""}
            ]
            return dict(held, text="x", display_text="x")

        reading = compare.reading(
            run([stopped("<|endoftext|>")]),
            run([stopped("</s>")], model_id="other/model"),
        )
        self.assertEqual(reading["choices_compared"], 1)
        self.assertEqual(reading["top_choice_changed"], 0)
        # Without a recorded raw decode there is only the label to go on.
        def labelled(label):
            held = metric(1, 5, 1.0)
            held["top_candidates"] = [{"token_id": 7, "text": label, "probability": 0.9}]
            return dict(held, text="x", display_text="x")

        older = compare.reading(
            run([labelled("<|endoftext|>")]),
            run([labelled("</s>")], model_id="other/model"),
        )
        self.assertEqual(older["top_choice_changed"], 1)

    def test_a_candidate_holding_half_a_character_is_not_compared(self):
        # Two different half-characters both decode to the replacement
        # character, so counting them as the same choice would be luck.
        def choosing(raw):
            held = metric(1, 5, 1.0)
            held["top_candidates"] = [
                {"token_id": 7, "text": "?", "probability": 0.9, "raw_text": raw}
            ]
            return dict(held, text="x", display_text="x")

        across = compare.reading(
            run([choosing("\ufffd")]),
            run([choosing("\ufffd")], model_id="other/model"),
        )
        self.assertEqual(across["compared"], 1)
        self.assertEqual(across["choices_compared"], 0)
        # Within one vocabulary the token ID still answers the question.
        within = compare.reading(run([choosing("\ufffd")]), run([choosing("\ufffd")]))
        self.assertEqual(within["choices_compared"], 1)

    def test_stopping_is_a_different_choice_from_any_other_silence(self):
        # Every hidden token writes nothing, so the text cannot tell a model
        # that wanted to stop from one that wanted a padding marker and
        # would have carried on.
        def choosing(label, stops):
            held = metric(1, 5, 1.0)
            held["top_candidates"] = [
                {"token_id": 7, "text": label, "probability": 0.9,
                 "raw_text": "", "stops": stops}
            ]
            return dict(held, text="x", display_text="x")

        ending = compare.reading(
            run([choosing("<|endoftext|>", True)]),
            run([choosing("</s>", True)], model_id="other/model"),
        )
        self.assertEqual(ending["choices_compared"], 1)
        self.assertEqual(ending["top_choice_changed"], 0)
        parting = compare.reading(
            run([choosing("<|endoftext|>", True)]),
            run([choosing("<pad>", False)], model_id="other/model"),
        )
        self.assertEqual(parting["choices_compared"], 1)
        self.assertEqual(parting["top_choice_changed"], 1)

    def test_a_span_with_a_missing_candidate_is_counted_in_neither(self):
        bare = dict(metric(1, 5, 1.0), top_candidates=[])
        reading = compare.reading(run([bare]), run([metric(1, 5, 1.0)]))
        self.assertEqual(reading["compared"], 1)
        self.assertEqual(reading["choices_compared"], 0)
        self.assertEqual(reading["top_choice_changed"], 0)

    def test_a_measurement_records_no_system_prompt_to_differ_over(self):
        # score_text has nowhere to put a system message, so recording one
        # would have the table report a difference neither run saw.
        left = run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, use_chat_template=False)
        right = run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, use_chat_template=True)
        self.assertNotIn("System prompt", compare.configuration(left))
        self.assertIn("System prompt", compare.configuration(run([metric(1, 5, 1.0)])))
        self.assertEqual(
            [row[0] for row in compare.configuration_rows(left, right)],
            ["Context read as"],
        )

    def test_a_same_model_top_choice_change_is_read_from_the_token_id(self):
        # Two vocabulary entries can decode to the same characters, and two
        # special tokens to nothing at all, so text cannot answer this.
        left = [metric(1, 5, 1.0, top="", top_id=11), metric(2, 6, 1.0, top="a", top_id=1)]
        right = [metric(1, 5, 1.0, top="", top_id=12), metric(2, 6, 1.0, top="a", top_id=1)]
        reading = compare.reading(run(left), run(right))
        self.assertEqual(reading["compared"], 2)
        self.assertEqual(reading["top_choice_changed"], 1)
        # Across two vocabularies an ID means nothing, so the text decides.
        across = compare.reading(run(left), run(right, model_id="other/model"))
        self.assertTrue(across["cross_model"])
        self.assertEqual(across["top_choice_changed"], 0)

    def test_a_token_with_no_candidates_is_not_counted_as_a_change(self):
        bare = dict(metric(1, 5, 1.0), top_candidates=[])
        reading = compare.reading(run([bare]), run([metric(1, 5, 1.0)]))
        self.assertEqual(reading["top_choice_changed"], 0)

    def test_the_framing_label_says_what_the_pass_really_did(self):
        def measured(**settings):
            return dict(
                run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, **settings),
                prompt="some context",
            )

        asked = measured(use_chat_template=True)
        self.assertEqual(compare.context_framing(asked), "a chat message")
        # A model with no chat template reads plain text however the box is
        # ticked, and the table must not show that tick as a difference.
        fell_back = measured(use_chat_template=True, chat_template_missing=True)
        self.assertIn("no chat template", compare.context_framing(fell_back))
        # An empty context has nothing to frame.
        self.assertIn(
            "no context",
            compare.context_framing(dict(measured(use_chat_template=True), prompt="")),
        )
        self.assertEqual(compare.context_framing(measured()), "plain text")
        rows = compare.configuration_rows(asked, fell_back)
        self.assertIn("Context read as", [row[0] for row in rows])

    def test_an_unverified_seam_is_carried_into_the_comparison(self):
        # The Score text tab warns here; a comparison that dropped the warning
        # would present two guessed boundaries as an exact difference.
        guessed = dict(
            run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT, seam_verified=False),
            prompt="some context",
        )
        exact = dict(
            run([metric(1, 5, 2.0)], kind=compare.MEASUREMENT, seam_verified=True),
            prompt="some context",
        )
        reading = compare.reading(guessed, exact)
        self.assertTrue(any("Slot A" in note for note in reading["caveats"]))
        self.assertFalse(any("Slot B" in note for note in reading["caveats"]))
        self.assertIn("could not confirm", compare.headline(reading, guessed, exact))
        # And it reaches the export beside the numbers it qualifies.
        document = compare.export(guessed, exact, reading)
        self.assertTrue(document["caveats"])
        # A reply has no seam to be unsure about.
        clean = compare.reading(run([metric(1, 5, 1.0)]), run([metric(1, 5, 2.0)]))
        self.assertEqual(clean["caveats"], [])

    def test_one_model_id_with_two_vocabularies_is_lined_up_on_text(self):
        # A repository fetched again can come back with a different
        # vocabulary, and the ID cannot tell the reader that.
        left = dict(run([metric(1, 5, 1.0)]), tokenizer="aaaa")
        right = dict(run([metric(1, 5, 2.0)]), tokenizer="bbbb")
        self.assertFalse(compare.same_vocabulary(left, right))
        reading = compare.reading(left, right)
        self.assertTrue(reading["cross_model"])
        self.assertIn("same model", " ".join(reading["caveats"]))
        # The same vocabulary under one ID still matches on token IDs, which
        # is what two loads at different precisions are.
        same = dict(right, tokenizer="aaaa")
        self.assertTrue(compare.same_vocabulary(left, same))
        self.assertFalse(compare.reading(left, same)["cross_model"])
        # Runs recorded before the fingerprint existed fall back to the ID.
        self.assertTrue(compare.same_vocabulary(run([]), run([])))
        self.assertFalse(
            compare.same_vocabulary(run([]), run([], model_id="other/model"))
        )

    def test_two_readings_of_one_repository_are_named_in_the_table(self):
        # Same ID, same device, same settings, different snapshot: a table
        # reporting no difference would hand the reader a gap it says
        # nothing about.
        left = run([metric(1, 5, 1.0)])
        right = dict(run([metric(1, 5, 2.0)]), load_id="fake/model#2")
        rows = compare.configuration_rows(left, right)
        self.assertEqual([row[0] for row in rows], ["Weights load"])
        self.assertEqual(rows[0][2], "fake/model#2")

    def test_the_thinking_row_says_what_the_model_did(self):
        asked = run([metric(1, 5, 1.0)], thinking_mode="on")
        self.assertEqual(compare.configuration(asked)["Thinking mode"], "on")
        # A checkpoint that cannot switch reports none, whatever was asked.
        ignored = run(
            [metric(1, 5, 1.0)], thinking_mode=None, requested_thinking_mode="on"
        )
        self.assertIn("cannot switch", compare.configuration(ignored)["Thinking mode"])
        # The two must not read alike: one of them ignored the setting.
        rows = compare.configuration_rows(asked, ignored)
        self.assertIn("Thinking mode", [row[0] for row in rows])
        plain = run([metric(1, 5, 1.0)], thinking_mode=None)
        self.assertEqual(
            compare.configuration(plain)["Thinking mode"], "model default"
        )

    def test_configuration_rows_name_only_what_differed(self):
        left = run([metric(1, 5, 1.0)], seed=1)
        right = run([metric(1, 5, 1.0)], seed=2, temperature=0.7)
        rows = compare.configuration_rows(left, right)
        self.assertEqual({row[0] for row in rows}, {"Seed", "Temperature"})
        self.assertEqual(
            len(compare.configuration_rows(left, right, differences_only=False)),
            len(compare.configuration(left)),
        )

    def test_the_inputs_each_run_was_given_are_compared_too(self):
        # A box edited between filling A and filling B changes the variable
        # without touching a control.
        left = run([metric(1, 5, 1.0)])
        right = dict(run([metric(1, 5, 1.0)]), prompt="a different question")
        rows = compare.configuration_rows(left, right)
        self.assertEqual([row[0] for row in rows], ["Prompt"])
        self.assertEqual(rows[0][2], "a different question")
        # Worst for a measurement: two contexts leave the passage lining up
        # token for token, so every position looks comparable.
        measured = run([metric(1, 5, 1.0)], kind=compare.MEASUREMENT)
        framed = dict(measured, prompt="framed differently")
        self.assertEqual(
            [row[0] for row in compare.configuration_rows(measured, framed)], ["Context"]
        )
        passage = dict(measured, text="another passage")
        self.assertEqual(
            [row[0] for row in compare.configuration_rows(measured, passage)],
            ["Measured text"],
        )

    def test_long_inputs_are_compared_whole_and_drawn_short(self):
        body = "word " * 80
        left = dict(run([metric(1, 5, 1.0)]), prompt=body + "one")
        right = dict(run([metric(1, 5, 1.0)]), prompt=body + "two")
        rows = compare.configuration_rows(left, right)
        # They agree far past what a cell shows, and still differ.
        self.assertEqual([row[0] for row in rows], ["Prompt"])
        self.assertTrue(rows[0][1].endswith("…"))
        self.assertEqual(len(rows[0][1]), compare.CELL_LENGTH)
        # The line break is drawn rather than collapsed; see the whitespace
        # test above for why.
        self.assertEqual(compare.cell("  a\n b  "), "a↵ b")
        self.assertEqual(compare.cell(""), "—")

    def test_a_steered_run_says_so_in_its_configuration(self):
        steered = run(
            [metric(1, 5, 1.0)],
            steering={"layer": 7, "strength": 2.0, "enabled": True, "vector_id": "ab" * 32},
        )
        reading = compare.configuration(steered)
        self.assertIn("layer 7", reading["Steering"])
        self.assertIn("strength 2", reading["Steering"])
        self.assertEqual(compare.configuration(run([metric(1, 5, 1.0)]))["Steering"], "off")

    def test_divergence_rows_are_ordered_widest_first(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0), metric(3, 7, 1.0)]
        right = [metric(1, 5, 1.5), metric(2, 6, 5.0), metric(3, 7, 1.1)]
        spans = compare.align(left, right)
        rows = compare.divergence_rows(spans)
        self.assertEqual([row[0] for row in rows], [2, 1, 3])
        self.assertEqual(rows[0][4], 4.0)
        self.assertEqual(len(compare.divergence_rows(spans, limit=1)), 1)

    def test_export_holds_both_runs_and_every_shared_token(self):
        left = [metric(1, 5, 1.0), metric(2, 6, 1.0)]
        right = [metric(1, 5, 2.0), metric(2, 6, 1.0)]
        left_run, right_run = run(left), run(right)
        reading = compare.reading(left_run, right_run)
        document = compare.export(left_run, right_run, reading)
        self.assertEqual(document["a"]["model_id"], "fake/model")
        self.assertEqual(len(document["aligned_spans"]), 2)
        self.assertEqual(document["aligned_spans"][0]["gap_bits"], 1.0)
        # Serializable as it stands: the download writes exactly this.
        json.dumps(document)

    def test_the_export_writes_the_steering_vector_out_in_full(self):
        import steering

        held = steering.compact(steering.normalize({
            "model_id": "fake/model", "layer": 1, "vector": [1.0, -2.0, 3.0],
        }))
        self.assertNotIn("vector", held)
        left = run([metric(1, 5, 1.0)], steering=held)
        right = run([metric(1, 5, 2.0)])
        document = compare.export(left, right, compare.reading(left, right))
        # The reference resolves against one machine only; the file has to
        # carry the numbers themselves.
        self.assertEqual(document["a"]["settings"]["steering"]["vector"], [1.0, -2.0, 3.0])
        self.assertIsNone(document["b"]["settings"]["steering"])
        json.dumps(document)

    def test_an_unresolvable_vector_leaves_its_reference_rather_than_failing(self):
        import steering

        held = steering.compact(steering.normalize({
            "model_id": "fake/model", "layer": 1, "vector": [1.0, -2.0, 3.0],
        }))
        (steering.asset_directory() / f"{held['vector_id']}.json").unlink()
        left = run([metric(1, 5, 1.0)], steering=held)
        right = run([metric(1, 5, 2.0)])
        document = compare.export(left, right, compare.reading(left, right))
        self.assertEqual(document["a"]["settings"]["steering"]["vector_id"], held["vector_id"])

    def test_an_empty_pair_reads_as_empty_everywhere(self):
        self.assertEqual(compare.reading(None, run([metric(1, 5, 1.0)])), {})
        self.assertIn("Fill both slots", compare.headline({}, None, None))
        self.assertIn("viz-empty", charts.comparison_tiles({}))
        self.assertIn("Empty", compare.describe(None, "A"))


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime.MANAGER
        manager = loaded_manager([0, 1, EOS_ID])
        manager._loaded = LoadedModel("fake/model", "CPU", "float32", manager.load_id)
        runtime.MANAGER = manager
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def fill(self, slot, mode=compare.REPLY, prompt="Hello", measured=""):
        frames = list(controls.fill_slot(
            slot, mode, prompt, measured, False, "", "", 0.0, 1.0, 0, 8, 1, False,
            "default", None, False, 1.0, 0,
        ))
        return frames[-1]

    def test_half_a_character_establishes_no_boundary(self):
        # Decoding the first byte of a two-byte character gives one
        # replacement character, so counting what the decode *shows* would
        # put a boundary inside the character — and a tokenizer that spent
        # one token on the whole of it would be lined up against the first
        # byte alone, with the rest called a divergence.
        class SplitCharacter:
            chat_template = None
            all_special_ids = []
            pieces = (b"\xc3", b"\xa9", b"x")

            def decode(self, ids, skip_special_tokens=False, **kwargs):
                raw = b"".join(self.pieces[int(index)] for index in ids)
                return raw.decode("utf-8", "replace")

            def convert_ids_to_tokens(self, index):
                return f"<{index}>"

        runtime.MANAGER.tokenizer = SplitCharacter()
        metrics = [metric(1, 0, 1.0), metric(2, 1, 1.0), metric(3, 2, 1.0)]
        decoded, ends = controls._decoded_spans(metrics)
        self.assertEqual(decoded, "éx")
        # The first token settles nothing; the second completes the character.
        self.assertEqual(ends, [0, 1, 2])
        # Which is what lets the two halves line up against one whole token.
        whole = [dict(metric(1, 9, 1.0), text="é"), dict(metric(2, 8, 1.0), text="x")]
        reading = compare.reading(
            dict(run(metrics, model_id="a/model"), decoded=decoded, token_ends=ends),
            dict(run(whole, model_id="b/model"), decoded="éx", token_ends=[1, 2]),
        )
        self.assertTrue(reading["complete"])
        self.assertEqual(reading["spans"], 2)

    def test_the_context_is_decoded_through_but_kept_out_of_the_text(self):
        # The token at the seam is assigned whole to the passage, and what
        # comes before a token changes what it decodes to, so the decode has
        # to start at the context and then cut the context back off.
        metrics = [dict(metric(1, 1, 1.0), text=" world")]
        decoded, ends = controls._decoded_spans(metrics, context_ids=[0])
        self.assertEqual(decoded, " world")
        self.assertEqual(ends, [len(" world")])
        # Without a context the offsets are the passage's own from the start.
        self.assertEqual(controls._decoded_spans(metrics), (" world", [6]))

    def test_the_seam_comes_from_the_passage_not_the_context_tokens(self):
        # A token straddling the seam belongs to the passage, so it is not
        # among context_ids at all: decoding those alone stops short of the
        # characters that token carries.
        # Token 0 is "Hello" and token 1 is " world". Both are in the
        # measured list with no context tokens behind them, which is what a
        # seam falling inside the first measured token looks like.
        straddling = [metric(1, 0, 1.0), metric(2, 1, 1.0)]
        decoded, ends = controls._decoded_spans(
            straddling, context_ids=[], expected=" world"
        )
        self.assertEqual(decoded, " world")
        self.assertEqual(ends, [0, len(" world")])
        # A tokenizer that does not round-trip its input falls back to the
        # context's own decode rather than slicing at the wrong place.
        decoded, _ends = controls._decoded_spans(
            straddling, context_ids=[], expected="nothing like it"
        )
        self.assertEqual(decoded, "Hello world")

    def test_a_hidden_stop_token_stays_out_of_the_recorded_text(self):
        # Generation decodes without these, so a decode that kept them would
        # disagree with the reply's own text — and two models spelling their
        # stop token differently would part at their last character.
        from test_streaming import EOS_ID

        metrics = [metric(1, 0, 1.0), metric(2, EOS_ID, 1.0)]
        decoded, ends = controls._decoded_spans(metrics)
        self.assertEqual(decoded, "Hello")
        self.assertEqual(ends, [len("Hello"), len("Hello")])

    def test_decoding_gives_up_quietly_with_no_tokenizer(self):
        runtime.MANAGER.tokenizer = None
        self.assertEqual(controls._decoded_spans([metric(1, 1, 1.0)]), ("", []))

    def test_a_reply_records_where_each_token_ends_in_its_own_text(self):
        held, _status, *_buttons = self.fill("A")
        self.assertEqual(len(held["token_ends"]), len(held["metrics"]))
        self.assertEqual(held["token_ends"][-1], len(held["decoded"]))

    def test_a_whitespace_passage_is_measured_rather_than_refused(self):
        # How expected a paragraph break was is a real question, and
        # score_text accepts whitespace for exactly that reason.
        held, status, *_buttons = self.fill(
            "A", mode=compare.MEASUREMENT, prompt="", measured="\n\n"
        )
        self.assertNotEqual(held, gr.skip())
        self.assertIn("Slot A filled", status)
        # A genuinely empty box is still refused.
        held, status, *_buttons = self.fill(
            "A", mode=compare.MEASUREMENT, prompt="", measured=""
        )
        self.assertEqual(held, gr.skip())
        self.assertEqual(status, controls.COMPARE_NO_TEXT)

    def test_a_measured_special_token_stays_in_the_recorded_passage(self):
        # Every measured token is the reader's own; score_text scores a
        # literal end marker as written, so the decode must keep it.
        from test_streaming import EOS_ID

        metrics = [metric(1, 0, 1.0), metric(2, EOS_ID, 1.0)]
        kept, ends = controls._decoded_spans(metrics, literal_prefix=len(metrics))
        self.assertEqual(kept, "Hello<eos>")
        self.assertEqual(ends[-1], len(kept))
        # A sampled stop token is still hidden.
        hidden, _ends = controls._decoded_spans(metrics)
        self.assertEqual(hidden, "Hello")

    def test_a_measurement_run_keeps_no_system_prompt_in_its_settings(self):
        held, _status, *_buttons = self.fill(
            "A", mode=compare.MEASUREMENT, prompt="", measured="Hello world"
        )
        self.assertEqual(held["kind"], compare.MEASUREMENT)
        self.assertNotIn("system_prompt", held["settings"])

    def test_a_reply_fills_its_slot_with_the_settings_it_ran_under(self):
        held, status, *_buttons = self.fill("A")
        self.assertEqual(held["kind"], compare.REPLY)
        self.assertEqual(held["model_id"], "fake/model")
        self.assertEqual(held["settings"]["seed"], 1)
        self.assertEqual(held["slot"], "A")
        self.assertTrue(held["metrics"])
        self.assertIn("Slot A filled", status)

    def test_an_empty_prompt_leaves_the_slot_alone_and_frees_the_model(self):
        held, status, *_buttons = self.fill("A", prompt="  ")
        self.assertEqual(held, gr.skip())
        self.assertEqual(status, controls.COMPARE_NO_PROMPT)
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_real_run_records_a_stop_marker_as_decoding_to_nothing(self):
        # The comparison below only works if production metrics agree that a
        # stop marker adds no characters. Decoded on its own it comes back as
        # the marker's name, which is what made the earlier fix a no-op.
        import numpy as np
        from test_streaming import EOS_ID, PIECES

        probabilities = np.full(len(PIECES), 0.01)
        probabilities[EOS_ID] = 0.9
        log_probabilities = np.log(probabilities / probabilities.sum())
        described = runtime.MANAGER._describe_token(
            position=1,
            token_id=0,
            raw_log_probabilities=log_probabilities,
            sampled_probabilities=np.exp(log_probabilities),
            segment="response",
        )
        chosen = described["top_candidates"][0]
        self.assertEqual(chosen["token_id"], EOS_ID)
        self.assertEqual(chosen["raw_text"], "")
        # The table still shows the marker under its name.
        self.assertEqual(chosen["text"], PIECES[EOS_ID])
        # And an ordinary token is untouched.
        self.assertEqual(
            [c["raw_text"] for c in described["top_candidates"] if c["token_id"] == 1],
            [PIECES[1]],
        )

    def test_a_run_records_the_vocabulary_it_was_measured_with(self):
        held, _status, *_buttons = self.fill("A")
        self.assertTrue(held["tokenizer"])
        other, _status, *_buttons = self.fill("B")
        # The same tokenizer fingerprints the same way, whatever else moved.
        self.assertEqual(held["tokenizer"], other["tokenizer"])

    def test_two_repositories_sharing_a_vocabulary_fingerprint_alike(self):
        # A fine-tune and the model it was tuned from ship one tokenizer, and
        # separating them would throw away the token-ID comparison that
        # sharing a vocabulary earns.
        runtime.MANAGER.tokenizer.get_vocab = lambda: {"a": 0, "b": 1}
        base = controls._tokenizer_identity()
        runtime.MANAGER.model_id = "someone/fine-tune"
        self.assertEqual(controls._tokenizer_identity(), base)

    def test_registering_a_marker_differently_changes_the_fingerprint(self):
        # One mapping, two decoders: the same ID is characters in one run and
        # nothing in the other, so the two do not share text.
        runtime.MANAGER.tokenizer.get_vocab = lambda: {"a": 0, "</s>": 1}
        runtime.MANAGER.tokenizer.all_special_ids = []
        plain = controls._tokenizer_identity()
        runtime.MANAGER.tokenizer.all_special_ids = [1]
        self.assertNotEqual(controls._tokenizer_identity(), plain)
        runtime.MANAGER.tokenizer.all_special_ids = []
        self.assertEqual(controls._tokenizer_identity(), plain)
        runtime.MANAGER.tokenizer.eos_token_id = 999
        self.assertNotEqual(controls._tokenizer_identity(), plain)

    def test_a_token_added_to_the_vocabulary_changes_the_fingerprint(self):
        # The likeliest way a refreshed repository differs is a token added
        # at the end, which shifts nothing a sampled encoding would cover.
        vocabulary = {"a": 0, "b": 1}
        runtime.MANAGER.tokenizer.get_vocab = lambda: dict(vocabulary)
        before = controls._tokenizer_identity()
        vocabulary["<|extra|>"] = 2
        after = controls._tokenizer_identity()
        self.assertTrue(before and after)
        self.assertNotEqual(before, after)
        # The mapping is the whole reading, so the repository name is not in
        # it: what separates two runs is what their vocabularies say.
        runtime.MANAGER.model_id = "other/model"
        self.assertEqual(controls._tokenizer_identity(), after)

    def test_an_unreadable_vocabulary_still_separates_two_repositories(self):
        def refuse():
            raise RuntimeError("no vocabulary here")

        runtime.MANAGER.tokenizer.get_vocab = refuse
        first = controls._tokenizer_identity()
        runtime.MANAGER.model_id = "other/model"
        self.assertTrue(first)
        # Nothing was established, so the ID goes in rather than a guess
        # being allowed to pass two repositories off as one.
        self.assertNotEqual(controls._tokenizer_identity(), first)
        # And a guess never collides with a reading of the real mapping.
        runtime.MANAGER.tokenizer.get_vocab = lambda: {"a": 0}
        self.assertNotEqual(controls._tokenizer_identity(), first)

    def test_a_busy_model_refuses_without_touching_the_slot_or_the_buttons(self):
        self.assertTrue(runtime.MANAGER.reserve_generation())
        try:
            held, status, *buttons = self.fill("B")
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(held, gr.skip())
        self.assertEqual(status, controls.COMPARE_BUSY)
        # The run that holds the slot owns the buttons. Publishing the idle
        # ones here would re-enable Run and hide Stop under a live run.
        self.assertEqual(buttons, [gr.skip()] * 3)

    def test_a_refusal_that_never_took_the_buttons_leaves_them_alone(self):
        for kwargs, expected in (
            ({"prompt": "  "}, controls.COMPARE_NO_PROMPT),
            ({"mode": compare.MEASUREMENT, "prompt": "", "measured": ""},
             controls.COMPARE_NO_TEXT),
        ):
            with self.subTest(expected=expected):
                held, status, *buttons = self.fill("A", **kwargs)
                self.assertEqual(held, gr.skip())
                self.assertEqual(status, expected)
                self.assertEqual(buttons, [gr.skip()] * 3)

    def test_a_failure_after_the_run_started_gives_the_buttons_back(self):
        with mock.patch.object(
            runtime.MANAGER, "generate", side_effect=ModelChanged("gone")
        ):
            _held, _status, run_a, run_b, stop = self.fill("A")
        self.assertTrue(run_a["interactive"])
        self.assertTrue(run_b["interactive"])
        self.assertFalse(stop["visible"])

    def test_a_model_change_mid_run_leaves_the_slot_as_it_was(self):
        with mock.patch.object(
            runtime.MANAGER, "generate", side_effect=ModelChanged("gone")
        ):
            held, status, *_buttons = self.fill("A")
        self.assertEqual(held, gr.skip())
        self.assertIn("model changed", status)
        self.assertFalse(runtime.MANAGER.busy)

    def test_rendering_both_slots_draws_the_strips_and_the_tables(self):
        left, right = run([metric(1, 5, 1.0)]), run([metric(1, 5, 4.0)])
        (a_heading, b_heading, a_strip, b_strip, tiles, chart, headline,
         settings_table, rows_table, export_state) = controls.render(left, right)
        self.assertIn("fake/model", a_heading)
        self.assertIn("fake/model", b_heading)
        self.assertEqual(a_strip["color_map"], compare.GAP_COLORS)
        self.assertEqual(len(a_strip["value"]), 1)
        self.assertEqual(len(b_strip["value"]), 1)
        self.assertIn("viz-tiles", tiles)
        self.assertIn("same 1 token", headline)
        self.assertEqual(export_state["reading"]["shared"], 1)
        self.assertEqual(settings_table["value"], [])
        self.assertEqual(rows_table["value"][0][4], 3.0)

    def test_clearing_empties_both_slots_and_puts_the_buttons_back(self):
        left, right, status, run_a, run_b, stop, *drawn = controls.clear_slots()
        self.assertIsNone(left)
        self.assertIsNone(right)
        self.assertEqual(status, controls.COMPARE_EMPTY)
        # Clearing cancels a run in flight, so that run never reaches its own
        # final frame and the buttons are published from here instead.
        self.assertTrue(run_a["interactive"])
        self.assertTrue(run_b["interactive"])
        self.assertFalse(stop["visible"])
        self.assertIn("Empty", drawn[0])
        self.assertEqual(drawn[-1], {"left": None, "right": None, "reading": {}})

    def test_the_download_writes_the_document_only_when_both_slots_are_full(self):
        self.assertIsNone(controls.download_comparison(None))
        self.assertIsNone(controls.download_comparison({"left": run([]), "right": None}))
        left, right = run([metric(1, 5, 1.0)]), run([metric(1, 5, 2.0)])
        held = {"left": left, "right": right, "reading": compare.reading(left, right)}
        path = Path(controls.download_comparison(held))
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        document = json.loads(path.read_text())
        self.assertEqual(document["b"]["model_id"], "fake/model")
        self.assertEqual(document["aligned_spans"][0]["gap_bits"], 1.0)

    def test_the_mode_switch_renames_the_prompt_box_and_shows_the_passage(self):
        prompt, text, template = controls.mode_controls(compare.MEASUREMENT)
        self.assertEqual(prompt["label"], "Context (optional)")
        self.assertTrue(text["visible"])
        self.assertTrue(template["visible"])
        prompt, text, template = controls.mode_controls(compare.REPLY)
        self.assertEqual(prompt["label"], "Prompt for both runs")
        self.assertFalse(text["visible"])


if __name__ == "__main__":
    unittest.main()
