import time
import unittest
from pathlib import Path

import numpy as np

import app
from model_runtime import PROMPT_SCORE_LIMIT, ScoredText
from token_metrics import UNSCORED_BEYOND_LIMIT, build_metric, unscored_metric


class Selection:
    """The one attribute ``inspect_token`` reads off a Gradio select event."""

    def __init__(self, index: int):
        self.index = index


class InspectTokenTests(unittest.TestCase):
    def inspect(self, metric: dict):
        # The state pairs the metrics with the stamp of the strip they were
        # drawn for, and inspect_token() drops a click that misses it.
        return app.inspect_token(app.stamped([metric]), Selection(0))

    def test_the_opening_token_is_explained_as_unpredicted(self):
        detail, rows = self.inspect(
            unscored_metric(
                position=1, token_id=7, token_text="<s>", fallback_text="<s>"
            ).to_dict()
        )

        self.assertIn("Nothing came before this token", detail)
        self.assertEqual(rows, [])

    def test_a_token_beyond_the_window_is_explained_as_skipped(self):
        # It had plenty of predecessors; it was dropped by the prompt cap, and
        # calling it unpredicted would be a false explanation.
        detail, rows = self.inspect(
            unscored_metric(
                position=9,
                token_id=7,
                token_text=" the",
                fallback_text=" the",
                reason=UNSCORED_BEYOND_LIMIT,
            ).to_dict()
        )

        self.assertNotIn("Nothing came before", detail)
        self.assertIn(f"{PROMPT_SCORE_LIMIT:,}", detail)
        self.assertIn("skipped", detail)
        self.assertEqual(rows, [])


class StubManager:
    """A loaded manager that returns one scored token and nothing else."""

    loaded = True

    def __init__(self, seam_verified: bool = True, chat_template_missing: bool = False):
        self.seam_verified = seam_verified
        self.chat_template_missing = chat_template_missing

    def score_text(self, text, *, context="", use_chat_template=False):
        log_probs = np.log(np.array([0.75, 0.25]))
        metric = build_metric(
            position=1,
            token_id=0,
            token_text="a",
            fallback_text="a",
            raw_log_probabilities=log_probs,
            sampled_probabilities=np.exp(log_probs),
            decode_token=str,
        ).to_dict()
        return ScoredText(
            context_metrics=[],
            metrics=[metric],
            seam_verified=self.seam_verified,
            chat_template_missing=self.chat_template_missing,
        )


class ScoreStatusTests(unittest.TestCase):
    """What the Score text status line promises about the numbers above it."""

    def status(self, seam_verified: bool = True, chat_template_missing: bool = False) -> str:
        original = app.MANAGER
        app.MANAGER = StubManager(seam_verified, chat_template_missing)
        try:
            return app.score_text("foo", "bar", False, app.DEFAULT_COLOR_SCALE)[7]
        finally:
            app.MANAGER = original

    def test_a_verified_seam_is_reported_without_a_caveat(self):
        status = self.status(True)

        self.assertIn("Scored 1 tokens", status)
        self.assertNotIn("Approximate", status)
        self.assertNotIn(app.TEMPLATE_CAVEAT, status)

    def test_an_unverifiable_seam_is_called_out(self):
        # The split had to encode the context and the text apart, so the first
        # scored token may not be the one the passage produces. Showing the
        # numbers is still better than refusing to score, but showing them as
        # exact is not.
        status = self.status(False)

        self.assertIn("Scored 1 tokens", status)
        self.assertIn(app.SEAM_CAVEAT, status)
        self.assertNotIn(app.TEMPLATE_CAVEAT, status)

    def test_a_missing_chat_template_is_called_out(self):
        # The reader ticked a box the model could not honour. The numbers are
        # exact for what was scored, so they are shown; what they are exact
        # about is the part that has to be said out loud.
        status = self.status(chat_template_missing=True)

        self.assertIn("Scored 1 tokens", status)
        self.assertIn(app.TEMPLATE_CAVEAT, status)
        self.assertNotIn(app.SEAM_CAVEAT, status)

    def test_both_caveats_read_as_two_sentences(self):
        # A tokenizer can fail both ways at once, and the line has to stay
        # readable: the count and perplexity first, then which passage was
        # scored, then how sure its first token is.
        status = self.status(seam_verified=False, chat_template_missing=True)

        self.assertTrue(status.startswith("Scored 1 tokens. Perplexity"))
        self.assertTrue(status.endswith(f"{app.TEMPLATE_CAVEAT} {app.SEAM_CAVEAT}"))


if __name__ == "__main__":
    unittest.main()


class DownloadCardTests(unittest.TestCase):
    """What the model panel says while a download runs, and after."""

    def setUp(self):
        self.original = app.MANAGER
        self.original_poll = app.DOWNLOAD_POLL_SECONDS
        app.DOWNLOAD_POLL_SECONDS = 0.01
        self.addCleanup(setattr, app, "MANAGER", self.original)
        self.addCleanup(setattr, app, "DOWNLOAD_POLL_SECONDS", self.original_poll)

    def test_bytes_are_shown_in_decimal_units(self):
        self.assertEqual(app.format_bytes(512), "512 B")
        self.assertEqual(app.format_bytes(1_500_000), "1.5 MB")
        self.assertEqual(app.format_bytes(14_600_000_000), "14.6 GB")

    def test_durations_read_as_estimates(self):
        self.assertEqual(app.describe_duration(30), "under a minute")
        self.assertEqual(app.describe_duration(60), "about 1 minute")
        self.assertEqual(app.describe_duration(4 * 60 + 20), "about 4 minutes")
        self.assertEqual(app.describe_duration(3600 + 10 * 60), "about 1 hour 10 minutes")

    def test_the_detail_shows_progress_speed_and_time_left(self):
        snap = app.DownloadSnapshot(
            files_done=14, files_total=17, bytes_done=4_000_000_000, bytes_total=16_000_000_000
        )

        detail = app.download_detail("org/model", snap, rate=50_000_000)

        self.assertIn("25%", detail)
        self.assertIn("4.0 GB of 16.0 GB", detail)
        self.assertIn("14 of 17 files", detail)
        self.assertIn("50.0 MB/s", detail)
        self.assertIn("about 4 minutes left", detail)
        self.assertIn("█", detail)

    def test_the_detail_explains_the_wait_before_the_file_list_arrives(self):
        detail = app.download_detail("org/model", app.DownloadSnapshot(), rate=None)

        self.assertIn("Asking Hugging Face", detail)
        self.assertNotIn("%", detail)

    def test_the_rate_meter_measures_from_the_first_bytes_not_from_zero(self):
        clock = iter([0.0, 1.0, 2.0, 4.0])
        meter = app.RateMeter(window=100.0, clock=lambda: next(clock))

        self.assertIsNone(meter.rate(0))
        # A resumed download credits everything already on disk at once.
        self.assertIsNone(meter.rate(5_000_000_000))
        self.assertIsNone(meter.rate(5_000_000_000), "no bytes moved yet")
        self.assertAlmostEqual(meter.rate(5_000_000_300), 100.0)

    def test_the_download_card_updates_until_the_download_ends(self):
        class Manager:
            active_downloads = {}

            def download(self, model_id, token, progress):
                bar = progress.bar_class()
                files = bar(desc="Fetching 2 files", total=2)
                rebuild = bar(desc="Reconstructing", total=0, unit="B")
                rebuild.total = 100
                for _ in range(2):
                    time.sleep(0.03)
                    rebuild.update(50)
                    files.update(1)
                return Path("/cache/snap")

        app.MANAGER = Manager()

        frames = list(app.download_model("org/model", ""))

        self.assertIn("Downloading model", frames[0])
        self.assertIn("Asking Hugging Face", frames[0])
        self.assertTrue(
            any("% " in frame or "%\n" in frame for frame in frames[1:-1]),
            frames,
        )
        self.assertIn("Download complete", frames[-1])
        self.assertIn("/cache/snap", frames[-1])
        self.assertIn("Load cached", frames[-1])

    def test_a_failed_download_is_reported_on_the_card(self):
        class Manager:
            active_downloads = {}

            def download(self, model_id, token, progress):
                raise OSError("no network")

        app.MANAGER = Manager()

        frames = list(app.download_model("org/model", ""))

        self.assertIn("Download failed", frames[-1])
        self.assertIn("no network", frames[-1])

    def test_a_second_request_follows_the_download_already_running(self):
        progress = app.DownloadProgress()
        rebuild = progress.bar_class()(desc="Reconstructing", total=1000, unit="B")
        rebuild.update(250)
        progress.bar_class()(desc="Fetching 1 files", total=1)
        calls = []

        class Manager:
            active_downloads = {"org/model": progress}

            def download(self, model_id, token, progress):
                calls.append(model_id)
                return Path("/cache/snap")

        app.MANAGER = Manager()
        frames = []
        for frame in app.download_model("org/model", ""):
            frames.append(frame)
            if len(frames) == 2:
                # The other handler's download finishes.
                app.MANAGER.active_downloads.clear()

        self.assertIn("25%", frames[0])
        self.assertEqual(calls, ["org/model"], "one quick pass over the cached files")
        self.assertIn("Download complete", frames[-1])

    def test_load_cached_while_downloading_points_at_the_running_download(self):
        progress = app.DownloadProgress()
        rebuild = progress.bar_class()(desc="Reconstructing", total=16_000_000_000, unit="B")
        rebuild.update(4_000_000_000)

        class Manager:
            active_downloads = {"org/model": progress}

        app.MANAGER = Manager()

        frames = list(app.load_cached_model("org/model"))

        self.assertEqual(len(frames), 1)
        self.assertIn("Still downloading", frames[0])
        self.assertIn("4.0 GB of 16.0 GB", frames[0])
        self.assertIn("Download and load", frames[0])

    def test_load_cached_on_a_partial_snapshot_says_how_to_finish_it(self):
        class Manager:
            active_downloads = {}

            def find_cached(self, model_id):
                raise app.IncompleteSnapshotError(
                    "The cached snapshot for 'org/model' (revision 'main', commit abc) is "
                    "incomplete: 3 file(s) are missing (model-00001-of-00003.safetensors, "
                    "model-00002-of-00003.safetensors, model-00003-of-00003.safetensors). "
                    "Outgoing traffic is disabled ('local_files_only=True'). Re-run the "
                    "download with network access to complete the snapshot.",
                    snapshot_path="/cache/snapshots/abc",
                )

        app.MANAGER = Manager()

        frames = list(app.load_cached_model("org/model"))

        self.assertIn("Download unfinished", frames[-1])
        self.assertIn("3 files still missing", frames[-1])
        self.assertIn("model-00003-of-00003.safetensors", frames[-1])
        self.assertIn("Download and load", frames[-1])
        self.assertNotIn("local_files_only", frames[-1])
