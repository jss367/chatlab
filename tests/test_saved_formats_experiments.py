"""Saved experiments (schema_version 1) as earlier versions wrote them.

Each document under tests/fixtures/saved/experiments was written by the commit
its directory is named after (see the generator) and is never regenerated.
The Experiments tab reads them with no model loaded, re-deriving what it shows
from the stored measurements: the colour buckets of every token, the order
"Find tokens" walks them in, the token detail, bookmarks, the saved layer
inspections and the timeline. These tests pin all of that, then save the
documents again with today's writer and check the same reading comes back.
"""

import unittest

from chatlab import experiment_runs as runs
from chatlab.token_metrics import category_for
from chatlab.ui import experiments, inference_timeline

from saved_formats_support import SAVED, sandbox

EXPERIMENTS = SAVED / "experiments"
REPLY = "0123456789abcdef0123456789abcdef"
MEASUREMENT = "fedcba9876543210fedcba9876543210"


class SavedExperimentsCase(unittest.TestCase):
    identifier = ""

    def setUp(self):
        self.box = sandbox(self)
        for era in ("01-reply-42396fe", "02-measurement-timeline-cf1acea"):
            for path in (EXPERIMENTS / era / "experiments").glob("*.json"):
                self.box.copy(path, f"experiments/{path.name}")
        self.opened = experiments.open_run(self.identifier, "Raw rank")
        self.document = self.opened[0]
        self.assertIsNotNone(self.document, self.opened[2])
        self.run = self.document["run"]

    def buckets(self, scale, document=None):
        return [category_for(metric, scale) for metric in (document or self.document)["run"]["metrics"]]

    def reading(self, document):
        """What the tab derives from a document, for comparing a re-saved copy."""

        run = document["run"]
        return {
            "title": document["title"], "text": run["text"], "kind": run["kind"],
            "settings": run["settings"], "bookmarks": document.get("bookmarks", {}),
            "buckets": {scale: self.buckets(scale, document) for scale in ("Raw rank", "Surprise", "Entropy")},
            "surprise": runs.ranked_tokens(run, "Highest surprise"),
            "closest": runs.ranked_tokens(run, "Closest alternatives"),
            "timeline": [(token["display_text"], token.get("scored", True))
                         for token in inference_timeline.tokens(document)],
            "readouts": [index for index in range(len(inference_timeline.tokens(document)))
                         if inference_timeline.readout(document, index, "Logit")],
        }

    def assert_resave_keeps_reading(self):
        before = self.reading(self.document)
        # A bookmark is the tab's own rewrite of the whole document.
        rewritten = runs.bookmark(self.identifier, 0, "Re-saved today")
        after = self.reading(runs.read(self.identifier))
        self.assertEqual(after.pop("bookmarks"), dict(before.pop("bookmarks"), **{"0": "Re-saved today"}))
        self.assertEqual(after, before)
        self.assertEqual(rewritten["inspections"], self.document["inspections"])
        self.assertEqual(rewritten.get("timeline_inspections"), self.document.get("timeline_inspections"))
        # And saved afresh as a new experiment, as a rerun or Compare would.
        copy = runs.save(self.run, "Copy")
        again = self.reading(runs.read(copy["id"]))
        self.assertEqual(again["buckets"], before["buckets"])
        self.assertEqual((again["surprise"], again["closest"]), (before["surprise"], before["closest"]))
        self.assertEqual(again["settings"], before["settings"])


class LibraryListingTests(unittest.TestCase):
    def test_both_eras_are_listed_newest_first(self):
        box = sandbox(self)
        for path in EXPERIMENTS.glob("*/experiments/*.json"):
            box.copy(path, f"experiments/{path.name}")
        self.assertEqual(experiments.choices()["choices"], [
            ("Cat passage, measured · test/tiny-decoder · 2026-09-22T12:00:00+00:00", MEASUREMENT),
            ("Rhyme with a tiny model · test/tiny-decoder · 2026-09-16T12:00:00+00:00", REPLY),
        ])
        self.assertEqual([item["id"] for item in runs.search("unexpected word")], [REPLY])
        self.assertEqual([item["id"] for item in runs.search("first passage")], [MEASUREMENT])


class ReplyEraTests(SavedExperimentsCase):
    """42396fe (#126): a chat reply saved from its trace, bookmarks, an attached inspection."""

    identifier = REPLY

    def test_opened(self):
        self.assertEqual(self.opened[2], "Opened **Rhyme with a tiny model** · 8 tokens.")
        self.assertEqual(self.opened[3], " log and the cat sat on the mat")
        configuration = self.opened[4]
        self.assertEqual(configuration["kind"], "reply")
        self.assertEqual(configuration["prompt"], "the dog sat on the")
        self.assertEqual(configuration["settings"], {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40, "skip_top_below": 0.0, "max_new_tokens": 16,
            "seed": 1234, "thinking_mode": "default", "system_prompt": "Finish the rhyme.",
        })
        self.assertNotIn("metrics", configuration)
        self.assertEqual(self.opened[6], [[2, " and", "Unexpected word here"], [5, " sat", ""]])
        self.assertEqual(self.opened[11]["choices"], [("Token 3 · inspection 1", 0)])
        self.assertEqual(self.document["app_version"], "0.22.0")

    def test_token_buckets(self):
        self.assertEqual([m["display_text"] for m in self.run["metrics"]],
                         [" log", " and", " the", " cat", " sat", " on", " the", " mat"])
        self.assertEqual([m["raw_rank"] for m in self.run["metrics"]], [13, 162, 237, 216, 44, 139, 33, 174])
        self.assertEqual(self.buckets("Raw rank"), [
            "Top 20", "Rank 101+", "Rank 101+", "Rank 101+", "Rank 21–100", "Rank 101+", "Rank 21–100", "Rank 101+"])
        self.assertEqual(self.buckets("Surprise"), [
            "6–10 bits", "Over 10 bits", "Over 10 bits", "Over 10 bits", "6–10 bits", "Over 10 bits",
            "6–10 bits", "Over 10 bits"])
        self.assertEqual(self.buckets("Entropy"), [
            "Wide (3–5)", "Very wide (5+)", "Wide (3–5)", "Wide (3–5)", "Wide (3–5)", "Wide (3–5)",
            "Open (1.5–3)", "Wide (3–5)"])
        self.assertEqual(set(self.buckets("Sampling shift")), {"Unchanged"})
        strip = experiments.token_strip(self.document, "Surprise", 1)["value"]
        self.assertEqual(strip[1], (" and", "Selected token"))
        self.assertEqual(strip[0], (" log", "6–10 bits"))

    def test_find_tokens(self):
        self.assertEqual(runs.ranked_tokens(self.run, "Highest surprise"), [2, 3, 7, 1, 5, 4, 6, 0])
        self.assertEqual(runs.ranked_tokens(self.run, "Closest alternatives"), [7, 4, 3, 1, 0, 2, 5, 6])
        self.assertEqual(experiments.navigate(self.document, None, "Highest surprise")[0], 2)
        self.assertEqual(experiments.navigate(self.document, 2, "Highest surprise")[0], 3)
        bookmarked = experiments.navigate(self.document, None, "Bookmarks")
        self.assertEqual((bookmarked[0], bookmarked[3]), (1, "Unexpected word here"))
        self.assertEqual(experiments.navigate(self.document, 1, "Bookmarks")[0], 4)

    def test_token_detail(self):
        index, detail, candidates, note, number, _ = experiments.select_token(self.document, 1)
        self.assertEqual((index, note, number), (1, "Unexpected word here", 2))
        self.assertIn("### Token 2: `' and'`", detail)
        self.assertIn("Raw rank</abbr>:** 162", detail)
        self.assertIn("0.01071%", detail)
        self.assertEqual(candidates[0][:2], [38, "'F'"])
        self.assertAlmostEqual(candidates[0][2], 0.23385002705300698, places=12)
        self.assertEqual(len(candidates), 8)

    def test_saved_inspection(self):
        record = self.document["inspections"][0]
        self.assertEqual(record["token_index"], 2)
        self.assertEqual([(row["layer"], row["rank"], row["top_text"]) for row in record["insight"]["layers"]],
                         [(0, 232, "tes"), (1, 234, "tes"), (2, 237, "tes")])
        lens, attention = experiments.inspection_view(self.document, 0, 1)
        self.assertIn("Logit lens for <code>&#x27; the&#x27;</code>", lens)
        self.assertIn("rank 237 at the end", lens)
        self.assertIn("Attention while predicting <code>&#x27; the&#x27;</code>", attention)
        self.assertIn("layer 1, averaged over heads", attention)

    def test_timeline_falls_back_on_the_attached_inspection(self):
        tokens = inference_timeline.tokens(self.document)
        self.assertEqual([t["display_text"] for t in tokens[:5]],
                         ["[ID 258]", "[ID 275]", "[ID 272]", "[ID 274]", "[ID 262]"])
        self.assertFalse(any(t.get("scored", True) for t in tokens[:5]))
        self.assertEqual(len(tokens), 13)
        readouts = [i for i in range(13) if inference_timeline.readout(self.document, i, "Logit")]
        # Response token 3 sits after the five context tokens.
        self.assertEqual(readouts, [7])
        frame = inference_timeline.frame(self.document, 8, "Logit", 0)
        self.assertTrue(frame[6].startswith("Response / passage · position 8 / 13"))
        self.assertIn("Layers: recomputed after the run", frame[6])
        self.assertIn("rank 237 at the end", frame[4])

    def test_resave(self):
        self.assert_resave_keeps_reading()


class MeasurementTimelineEraTests(SavedExperimentsCase):
    """cf1acea (#155): a measured passage with recomputed timeline readouts."""

    identifier = MEASUREMENT

    def test_opened(self):
        self.assertEqual(self.opened[2], "Opened **Cat passage\\, measured** · 6 tokens.")
        self.assertEqual(self.opened[3], " the cat sat on the mat")
        self.assertEqual(self.opened[4]["run_id"], "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a")
        self.assertEqual(self.opened[4]["settings"], {
            "use_chat_template": False, "seam_verified": True, "chat_template_missing": False, "steering": None})
        self.assertEqual(self.opened[6], [[1, " the", "First passage token"]])
        self.assertEqual(self.opened[11]["choices"], [])
        self.assertEqual(self.document["app_version"], "0.30.0")

    def test_token_buckets_and_order(self):
        self.assertEqual([m["raw_rank"] for m in self.run["metrics"]], [63, 244, 61, 128, 34, 211])
        self.assertEqual(self.buckets("Raw rank"), [
            "Rank 21–100", "Rank 101+", "Rank 21–100", "Rank 101+", "Rank 21–100", "Rank 101+"])
        self.assertEqual(self.buckets("Surprise"), [
            "Over 10 bits", "Over 10 bits", "Over 10 bits", "Over 10 bits", "6–10 bits", "Over 10 bits"])
        self.assertEqual(self.buckets("Entropy"), [
            "Wide (3–5)", "Wide (3–5)", "Wide (3–5)", "Wide (3–5)", "Open (1.5–3)", "Wide (3–5)"])
        self.assertEqual(runs.ranked_tokens(self.run, "Highest surprise"), [1, 5, 3, 2, 0, 4])
        self.assertEqual(runs.ranked_tokens(self.run, "Closest alternatives"), [5, 2, 1, 3, 0, 4])
        _, detail, candidates, *_ = experiments.select_token(self.document, 1)
        self.assertIn("Raw rank</abbr>:** 244", detail)
        self.assertEqual(candidates[0][0], 185)
        self.assertAlmostEqual(candidates[0][2], 0.2532704971334142, places=12)

    def test_timeline(self):
        tokens = inference_timeline.tokens(self.document)
        self.assertEqual(len(tokens), 14)
        self.assertEqual([t["display_text"] for t in tokens[:8]],
                         ["[ID 48]", "[ID 65]", "[ID 83]", "[ID 83]", "[ID 65]", "[ID 71]", "[ID 69]", "[ID 26]"])
        self.assertEqual([t["display_text"] for t in tokens[8:]], [" the", " cat", " sat", " on", " the", " mat"])
        self.assertEqual(sorted(self.document["timeline_inspections"]), ["logit:1", "logit:3", "logit:5"])
        readouts = [i for i in range(14) if inference_timeline.readout(self.document, i, "Logit")]
        self.assertEqual(readouts, [1, 3, 5])
        for number, text in ((2, "a"), (4, "s"), (6, "g")):
            frame = inference_timeline.frame(self.document, number, "Logit", 0)
            self.assertTrue(frame[6].startswith(f"Prompt / context · position {number} / 14"))
            self.assertIn("Layers: recomputed after the run", frame[6])
            self.assertIn(f"Logit lens for <code>&#x27;{text}&#x27;</code>", frame[4])
            self.assertIn("averaged over heads", frame[5])
        empty = inference_timeline.frame(self.document, 9, "Logit", 0)
        self.assertIn("No saved layer readout at this position.", empty[6])
        record = self.document["timeline_inspections"]["logit:3"]
        self.assertEqual((record["source"], record["recorded_at"]), ("recomputed", "2026-09-22T12:00:00+00:00"))
        self.assertEqual((record["insight"]["index"], record["insight"]["token_id"]), (3, 83))
        self.assertEqual(len(record["insight"]["layers"]), 3)

    def test_resave(self):
        self.assert_resave_keeps_reading()


if __name__ == "__main__":
    unittest.main()
