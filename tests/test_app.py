import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app
from model_runtime import (
    MODEL_WEIGHTS,
    PROMPT_SCORE_LIMIT,
    CacheStatus,
    DownloadProgress,
    ModelManager,
    ScoredText,
)
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
    model_id = "stub/model"
    load_id = "stub/model#1"

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


class FakeDownloads(ModelManager):
    """A manager that keeps the real download bookkeeping around a fake fetch."""

    def fetch(self, model_id, token, progress):
        raise NotImplementedError

    def download(self, model_id, token=None, progress=None):
        progress = progress or DownloadProgress()
        try:
            with self._downloads_lock:
                self.active_downloads.setdefault(model_id, progress)
            return self.fetch(model_id, token, progress)
        finally:
            with self._downloads_lock:
                if self.active_downloads.get(model_id) is progress:
                    del self.active_downloads[model_id]


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
        self.assertEqual(app.describe_duration(4), "a few seconds")
        self.assertEqual(app.describe_duration(32), "about 30 seconds")
        self.assertEqual(app.describe_duration(57), "about 1 minute")
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
        # The Hub's answer arrives only once the first card has been shown.
        file_list_arrives = threading.Event()

        class Manager(FakeDownloads):
            def fetch(self, model_id, token, progress):
                file_list_arrives.wait(5)
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

        cards = app.download_model("org/model", "")
        first = next(cards)
        waiting = next(cards)
        file_list_arrives.set()
        frames = [first, waiting, *cards]

        self.assertIn("Downloading model", frames[0])
        self.assertTrue(any("Asking Hugging Face" in frame for frame in frames[1:]))
        self.assertTrue(
            any("% " in frame or "%\n" in frame for frame in frames[1:-1]),
            frames,
        )
        self.assertIn("Download complete", frames[-1])
        self.assertIn("/cache/snap", frames[-1])
        self.assertIn("Load cached", frames[-1])

    def test_a_failed_download_is_reported_on_the_card(self):
        class Manager(FakeDownloads):
            def fetch(self, model_id, token, progress):
                raise OSError("no network")

        app.MANAGER = Manager()

        frames = list(app.download_model("org/model", ""))

        self.assertIn("Download failed", frames[-1])
        self.assertIn("no network", frames[-1])
        self.assertEqual(app.MANAGER.active_downloads, {})

    def test_a_worker_that_cannot_start_releases_its_reservation(self):
        app.MANAGER = ModelManager()

        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("no threads")
        ):
            with self.assertRaisesRegex(RuntimeError, "no threads"):
                list(app.stream_download("org/model", ""))

        self.assertEqual(app.MANAGER.active_downloads, {})

    def test_a_second_request_follows_the_download_already_running(self):
        progress = DownloadProgress()
        rebuild = progress.bar_class()(desc="Reconstructing", total=1000, unit="B")
        rebuild.update(250)
        progress.bar_class()(desc="Fetching 1 files", total=1)
        calls = []

        class Manager(FakeDownloads):
            def fetch(self, model_id, token, progress):
                calls.append(model_id)
                return Path("/cache/snap")

        app.MANAGER = Manager()
        app.MANAGER.active_downloads["org/model"] = progress
        frames = []
        for frame in app.download_model("org/model", ""):
            frames.append(frame)
            if len(frames) == 2:
                # The other handler's download finishes.
                app.MANAGER.active_downloads.clear()

        self.assertIn("25%", frames[1])
        self.assertEqual(calls, ["org/model"], "one quick pass over the cached files")
        self.assertIn("Download complete", frames[-1])

    def test_two_handlers_starting_together_share_one_download(self):
        # Download and "Download and load" clicked in the same instant: both
        # handlers reach stream_download before either worker has started.
        started = threading.Event()
        finish = threading.Event()
        calls = []

        class Manager(FakeDownloads):
            def fetch(self, model_id, token, progress):
                calls.append(model_id)
                started.set()
                finish.wait(5)
                return Path("/cache/snap")

            def load(self, model_id, local_path, progress=None):
                return "cpu"

        app.MANAGER = Manager()
        first = app.download_model("org/model", "")
        second = app.download_and_load_model("org/model", "")

        self.assertIn("Downloading model", next(first))
        self.assertIn("Downloading model", next(first))
        self.assertTrue(started.wait(5))
        self.assertIn("Downloading model", next(second))
        self.assertIn("Downloading model", next(second))
        self.assertEqual(calls, ["org/model"], "the second handler follows the first")

        finish.set()
        self.assertIn("Download complete", list(first)[-1])
        # The follower's own pass over the now-cached files is its second call.
        self.assertIn("Model ready", list(second)[-1])
        self.assertEqual(calls, ["org/model", "org/model"])

    def test_the_card_refuses_a_malformed_model_id_before_downloading(self):
        class Manager(FakeDownloads):
            def fetch(self, model_id, token, progress):
                raise AssertionError("should not be reached")

        app.MANAGER = Manager()

        frames = list(app.download_model("not a model id", ""))

        self.assertIn("Download failed", frames[-1])
        self.assertIn("organization/model-name", frames[-1])
        self.assertEqual(app.MANAGER.active_downloads, {})

    def test_load_cached_while_downloading_points_at_the_running_download(self):
        progress = DownloadProgress()
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
                    "/cache/snapshots/abc",
                )

        app.MANAGER = Manager()

        with mock.patch.object(
            app, "cache_status", return_value=CacheStatus(cached_bytes=1)
        ):
            frames = list(app.load_cached_model("org/model"))

        self.assertIn("Download unfinished", frames[-1])
        self.assertIn("3 files still missing", frames[-1])
        self.assertIn("model-00003-of-00003.safetensors", frames[-1])
        self.assertIn("Download and load", frames[-1])
        self.assertNotIn("local_files_only", frames[-1])


class DownloadManager(FakeDownloads):
    """Stands in for the real manager: downloads succeed without a network."""

    def __init__(self):
        super().__init__()
        self.downloads = 0

    def fetch(self, model_id, hf_token, progress):
        self.downloads += 1
        return Path("/cache/models--allenai--Olmo-3-7B-Think/snapshots/abc")

    def load(self, model_id, path, progress=None):
        return "mps"


class LoadCardTests(unittest.TestCase):
    """What the model panel says while a cached model is read into memory."""

    MODEL = "allenai/Olmo-3-7B-Think"

    class Manager:
        """A manager whose load does whatever the test hands it."""

        active_downloads: dict = {}

        def __init__(self, work):
            self._work = work

        def find_cached(self, model_id):
            return Path("/cache/models--allenai--Olmo-3-7B-Think/snapshots/abc")

        def load(self, model_id, path, progress=None):
            return self._work(progress)

    def setUp(self):
        self.addCleanup(setattr, app, "MANAGER", app.MANAGER)
        self.addCleanup(setattr, app, "cache_status", app.cache_status)
        self.addCleanup(setattr, app, "LOAD_POLL_SECONDS", app.LOAD_POLL_SECONDS)
        app.MANAGER = self.Manager(lambda progress: "CPU")
        app.cache_status = lambda model_id: CacheStatus(cached_bytes=14_600_000_000)
        app.LOAD_POLL_SECONDS = 0.01

    def test_the_first_frame_says_the_weights_are_being_read_not_fetched(self):
        # The old card named the size and the folder and then sat there, which
        # read like a download that had stalled.
        detail = app.load_detail(
            self.MODEL,
            app.LoadSnapshot(bytes_total=14_600_000_000),
            rate=None,
            remaining=None,
        )

        self.assertIn("14.6 GB of weights", detail)
        self.assertIn("Nothing is being downloaded", detail)
        self.assertNotIn("%", detail)

    def test_the_card_counts_the_weights_read_before_any_reach_the_device(self):
        detail = app.load_detail(
            self.MODEL,
            app.LoadSnapshot(
                bytes_done=0, bytes_total=14_600_000_000, steps_done=178, steps_total=356
            ),
            rate=None,
            remaining=90.0,
        )

        self.assertIn("25%", detail)
        self.assertIn("178 of 356 parts read", detail)
        self.assertIn("about 2 minutes left", detail)
        self.assertIn("\u2588", detail)

    def test_the_card_counts_bytes_once_they_are_on_the_device(self):
        detail = app.load_detail(
            self.MODEL,
            app.LoadSnapshot(
                bytes_done=7_300_000_000,
                bytes_total=14_600_000_000,
                steps_done=356,
                steps_total=356,
            ),
            rate=500_000_000,
            remaining=15.0,
        )

        self.assertIn("75%", detail)
        self.assertIn("7.3 GB of 14.6 GB on the device", detail)
        self.assertIn("about 15 seconds left", detail)
        self.assertIn("500 MB/s", detail)

    def test_a_card_that_is_still_loading_stops_short_of_finished(self):
        # The allocator holds the last byte a little before the loader is
        # done with the model, and a full bar over a wait that goes on reads
        # as a hang.
        detail = app.load_detail(
            self.MODEL,
            app.LoadSnapshot(
                bytes_done=14_600_000_000,
                bytes_total=14_600_000_000,
                steps_done=356,
                steps_total=356,
            ),
            rate=None,
            remaining=2.0,
        )

        self.assertIn("99%", detail)
        self.assertNotIn("100%", detail)

    def test_the_pace_is_measured_from_the_first_progress_not_from_the_start(self):
        # The seconds before the first weight are setup, and counting them as
        # slow progress would put the first estimate minutes out.
        clock = iter([0.0, 10.0, 10.5, 12.0, 14.0])
        pace = app.Pace(clock=lambda: next(clock))

        self.assertIsNone(pace.remaining(0.0), "nothing has moved yet")
        self.assertIsNone(pace.remaining(0.25), "the baseline reading")
        self.assertIsNone(pace.remaining(0.30), "too soon to tell")
        # A quarter of the load in the two seconds since the baseline, so
        # the half that is left reads as four seconds, not the ten that
        # counting the setup as progress would have implied.
        self.assertAlmostEqual(pace.remaining(0.50), 4.0)
        # Two seconds later and no further on: a stall lengthens the estimate
        # rather than freezing it.
        self.assertAlmostEqual(pace.remaining(0.50), 8.0)

    def test_the_load_card_updates_until_the_load_ends(self):
        def work(progress):
            bar = progress.bar_class()(desc="Loading weights", total=4)
            held = [0]
            progress.measure_bytes(1000, lambda: held[0])
            for _ in range(4):
                time.sleep(0.02)
                bar.update(1)
            for byte_count in (500, 1000):
                time.sleep(0.02)
                held[0] = byte_count
            return "Apple Metal (MPS)"

        app.MANAGER = self.Manager(work)

        frames = list(app.load_cached_model(self.MODEL))

        self.assertIn("Finding cached model", frames[0])
        self.assertTrue(
            any("parts read" in frame for frame in frames), frames
        )
        self.assertTrue(
            any("on the device" in frame for frame in frames), frames
        )
        self.assertIn("Model ready", frames[-1])
        self.assertIn("Apple Metal (MPS)", frames[-1])
        self.assertIn("seconds", frames[-1])

    def test_a_failed_load_is_reported_on_the_card(self):
        def work(progress):
            raise RuntimeError("Metal ran out of memory")

        app.MANAGER = self.Manager(work)

        frames = list(app.load_cached_model(self.MODEL))

        self.assertIn("Could not load cached model", frames[-1])
        self.assertIn("Metal ran out of memory", frames[-1])

    def test_a_load_that_ends_before_the_first_frame_still_reports_ready(self):
        app.MANAGER = self.Manager(lambda progress: "CPU")

        frames = list(app.load_cached_model(self.MODEL))

        self.assertIn("Model ready", frames[-1])


class DownloadStatusTests(unittest.TestCase):
    """What the model card says about files that were already on disk.

    The download itself never fetches anything twice; the point of these
    cards is that the reader can tell that from the screen.
    """

    MODEL = "allenai/Olmo-3-7B-Think"

    def run_handler(self, handler, statuses: list[CacheStatus], *args) -> list[str]:
        remaining = list(statuses)
        original_manager, original_status = app.MANAGER, app.cache_status
        app.MANAGER = DownloadManager()
        app.cache_status = lambda model_id: remaining.pop(0)
        try:
            return list(handler(self.MODEL, *args))
        finally:
            app.MANAGER, app.cache_status = original_manager, original_status

    def test_a_first_download_is_announced_as_a_full_one(self):
        cards = self.run_handler(
            app.download_model,
            [CacheStatus(), CacheStatus(cached_bytes=15_000_000_000)],
            "",
        )

        self.assertIn("Nothing is cached yet", cards[0])
        self.assertIn("Fetched 15.0 GB", cards[-1])
        self.assertIn("Load cached", cards[-1])

    def test_a_cut_off_download_is_announced_as_resumed(self):
        cards = self.run_handler(
            app.download_model,
            [
                CacheStatus(
                    cached_bytes=10,
                    partial_files=3,
                    partial_bytes=3_000_000_000,
                    missing_files=(MODEL_WEIGHTS,),
                ),
                CacheStatus(cached_bytes=15_000_000_010),
            ],
            "",
        )

        self.assertIn("Resuming download", cards[0])
        self.assertIn("3 files (3.0 GB) partly downloaded", cards[0])
        self.assertNotIn("weight files (", cards[0])
        self.assertIn("Fetched the remaining 12.0 GB", cards[-1])

    def test_a_stray_partial_blob_beside_a_complete_cache_is_not_a_resume(self):
        """A leftover from another revision changes nothing about ``main``."""

        cached = CacheStatus(cached_bytes=15_000_000_000, partial_files=1, partial_bytes=5)
        cards = self.run_handler(app.download_model, [cached, cached], "")

        self.assertIn("already in the Hugging Face cache", cards[0])
        self.assertNotIn("Resuming", cards[0])

    def test_a_complete_cache_reports_that_nothing_was_fetched(self):
        cached = CacheStatus(cached_bytes=15_000_000_000)
        cards = self.run_handler(app.download_model, [cached, cached], "")

        self.assertIn("already in the Hugging Face cache", cards[0])
        self.assertIn("nothing new was fetched", cards[-1])

    def test_download_and_load_carries_the_same_wording(self):
        cached = CacheStatus(cached_bytes=15_000_000_000)
        cards = self.run_handler(app.download_and_load_model, [cached, cached], "")

        self.assertIn("already in the Hugging Face cache", cards[0])
        self.assertIn("nothing new was fetched", cards[1])
        self.assertIn("Model ready", cards[-1])

    def test_load_cached_refuses_a_cut_off_download(self):
        cards = self.run_handler(
            app.load_cached_model,
            [
                CacheStatus(
                    cached_bytes=10,
                    partial_files=1,
                    partial_bytes=5,
                    missing_files=("model-00002-of-00002.safetensors",),
                )
            ],
        )

        self.assertIn("Download incomplete", cards[-1])
        self.assertIn("1 file (5 B) partly downloaded", cards[-1])
        self.assertIn("`model-00002-of-00002.safetensors` is missing", cards[-1])
        self.assertIn("Download and load", cards[-1])

    def test_load_cached_loads_a_complete_snapshot_beside_a_stray_partial_blob(self):
        """Only what the ``main`` snapshot lacks can refuse a load; a partial
        blob left by another revision, or by a file the model never reads,
        used to be mistaken for a cut-off download."""

        snapshot = "/cache/models--allenai--Olmo-3-7B-Think/snapshots/abc"
        with mock.patch("huggingface_hub.snapshot_download", return_value=snapshot):
            cards = self.run_handler(
                app.load_cached_model,
                [CacheStatus(cached_bytes=15_000_000_000, partial_files=1, partial_bytes=5)],
            )

        self.assertNotIn("Download incomplete", "".join(cards))
        self.assertIn("Model ready", cards[-1])

    def test_load_cached_refuses_a_snapshot_without_weights(self):
        """Config and tokenizer alone used to sail through to a shard-missing traceback."""

        cards = self.run_handler(
            app.load_cached_model,
            [CacheStatus(cached_bytes=2_000_000, missing_files=(MODEL_WEIGHTS,))],
        )

        self.assertIn("Download incomplete", cards[-1])
        self.assertIn("2.0 MB cached", cards[-1])
        self.assertIn("the model weights are missing", cards[-1])
        self.assertIn("Download and load", cards[-1])

    def test_load_cached_refuses_a_repo_of_another_kind(self):
        cards = self.run_handler(
            app.load_cached_model,
            [CacheStatus(cached_bytes=5_500_000_000, unsupported=True)],
        )

        self.assertIn("Unsupported model", cards[-1])
        self.assertIn("5.5 GB cached", cards[-1])
        self.assertIn("not a Transformers language model", cards[-1])
        self.assertNotIn("Download incomplete", cards[-1])

    def test_a_download_stopped_between_shards_is_announced_as_resumed(self):
        shards = tuple(f"model-0000{i}-of-00006.safetensors" for i in range(2, 7))
        before = CacheStatus(cached_bytes=3_000_000_000, missing_files=shards)
        cards = self.run_handler(
            app.download_model, [before, CacheStatus(cached_bytes=15_000_000_000)], ""
        )

        self.assertIn("Resuming download", cards[0])
        self.assertIn(
            "`model-00002-of-00006.safetensors`, `model-00003-of-00006.safetensors` "
            "and 3 more weight files are missing",
            cards[0],
        )
        self.assertNotIn("already in the Hugging Face cache", cards[0])
        self.assertIn("Fetched the remaining 12.0 GB", cards[-1])

    def test_a_single_missing_shard_is_named(self):
        before = CacheStatus(cached_bytes=10, missing_files=("model.safetensors",))
        cards = self.run_handler(app.load_cached_model, [before])

        self.assertIn("`model.safetensors` is missing", cards[-1])

    def test_load_cached_explains_an_absent_model(self):
        cards = self.run_handler(app.load_cached_model, [CacheStatus()])

        self.assertIn("Not cached", cards[-1])

    def test_an_invalid_id_fails_before_anything_is_measured(self):
        original = app.MANAGER
        app.MANAGER = DownloadManager()
        try:
            cards = list(app.download_model("not-a-model-id", ""))
        finally:
            app.MANAGER = original

        self.assertEqual(len(cards), 1)
        self.assertIn("Download failed", cards[0])
        self.assertIn("organization/model-name", cards[0])


if __name__ == "__main__":
    unittest.main()
