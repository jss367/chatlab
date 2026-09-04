import html
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np

import app
import settings
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
        # The generation slot, modelled the way ModelManager does it: a plain
        # Lock, taken without blocking, so a caller that loses reports instead
        # of queueing.
        self._generating = threading.Lock()

    def reserve_generation(self) -> bool:
        return self._generating.acquire(blocking=False)

    def release_generation(self) -> None:
        self._generating.release()

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
            frames = list(app.score_text("foo", "bar", False, app.DEFAULT_COLOR_SCALE))
            return frames[-1][7]
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


class ScoreWhileGeneratingTests(unittest.TestCase):
    """Score text refuses while a reply is streaming, rather than waiting on it.

    MANAGER.score_text() takes the model lock, which a generation holds across
    every one of its yields, so a pass started mid-reply would not run until the
    reply ended. Without the slot the button simply stops responding for as long
    as the response takes, which the reader cannot tell from a hang.
    """

    def setUp(self):
        self.manager = StubManager()
        original = app.MANAGER
        app.MANAGER = self.manager
        self.addCleanup(setattr, app, "MANAGER", original)

    def score(self):
        return list(app.score_text("foo", "bar", False, app.DEFAULT_COLOR_SCALE))[-1]

    def test_a_reserved_slot_is_refused_with_a_reason(self):
        self.assertTrue(self.manager.reserve_generation())
        try:
            result = self.score()
        finally:
            self.manager.release_generation()

        self.assertEqual(result[7], app.SCORE_BUSY)

    def test_a_refusal_touches_nothing_but_the_status(self):
        # The strips still describe the response that is streaming, and the
        # stamp on them still has to match the clicks it is collecting.
        before = app._metrics_generation
        self.assertTrue(self.manager.reserve_generation())
        try:
            result = self.score()
        finally:
            self.manager.release_generation()

        self.assertEqual(app._metrics_generation, before, "no stamp was minted")
        for index, value in enumerate(result):
            if index != 7:
                self.assertEqual(value, gr.skip(), f"output {index}")

    def test_the_slot_is_given_back_after_a_successful_pass(self):
        self.score()

        self.assertTrue(
            self.manager.reserve_generation(), "score_text() kept the slot"
        )
        self.manager.release_generation()

    def test_the_slot_is_given_back_after_a_failed_pass(self):
        self.manager.score_text = mock.Mock(side_effect=RuntimeError("no room"))

        result = self.score()

        self.assertIn("no room", result[7])
        self.assertTrue(
            self.manager.reserve_generation(), "a failure kept the slot"
        )
        self.manager.release_generation()

    def test_the_slot_is_held_until_the_scored_strips_are_delivered(self):
        # Gradio resumes the generator only once the browser has this frame.
        # Giving the slot back any earlier would let a Send mint a newer stamp
        # and publish its opening frame first, leaving these strips on screen
        # under a stamp the app has already moved past.
        frames = app.score_text("foo", "bar", False, app.DEFAULT_COLOR_SCALE)
        first = next(frames)

        self.assertNotEqual(first[7], app.SCORE_BUSY)
        self.assertFalse(
            self.manager.reserve_generation(), "the slot was given back early"
        )
        self.assertEqual(list(frames), [])
        self.assertTrue(self.manager.reserve_generation())
        self.manager.release_generation()


class FailureReportTests(unittest.TestCase):
    """A failure has to reach a reader who is not looking at the status line."""

    def test_a_failure_pops_up_and_stays_on_the_line(self):
        with mock.patch.object(app.gr, "Warning") as toast:
            line = app.failure_status("Generation failed", "out of memory")

        self.assertIn("out of memory", line)
        self.assertIn('class="failure"', line)
        # The toast carries the cause, not just the fact that something went
        # wrong, and it waits to be closed rather than fading on its own.
        toast.assert_called_once_with(
            "out of memory", title="Generation failed", duration=None
        )

    def test_an_angle_bracket_survives_both_the_line_and_the_toast(self):
        # Error messages quote what the reader typed and what the runtime
        # printed, and neither is markup. The toast writes its message into
        # the page as markup, so an unescaped "cannot read <pad>" would show
        # up with the word missing.
        with mock.patch.object(app.gr, "Warning") as toast:
            line = app.failure_status("Could not score that text", "no <pad> here")

        self.assertIn("no &lt;pad&gt; here", line)
        self.assertNotIn("<pad>", line)
        self.assertEqual(toast.call_args.args[0], "no &lt;pad&gt; here")

    def test_a_failure_card_pops_up_and_is_tinted(self):
        with mock.patch.object(app.gr, "Warning") as toast:
            card = app.failure_card("Download failed", "no such repo")

        self.assertEqual(card, app.status_card("Download failed", "no such repo", "error"))
        self.assertIn('class="failure-text"', card)
        toast.assert_called_once_with(
            "no such repo", title="Download failed", duration=None
        )

    def test_only_a_failing_card_is_tinted(self):
        for tone in ("neutral", "working", "success"):
            with self.subTest(tone=tone):
                self.assertNotIn(
                    "failure-text", app.status_card("Model ready", "loaded", tone)
                )

    def test_a_card_detail_keeps_its_markdown(self):
        # The heading carries the tint so the detail stays plain markdown -
        # a card that spelled out a file name in backticks still renders it.
        card = app.failure_card("Download unfinished", "`config.json` is missing.")

        self.assertIn("`config.json` is missing.", card)


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

    def test_a_slow_rate_is_rounded_to_whole_bytes(self):
        # The meter measures in float bytes per second, and format_bytes()
        # prints anything under 1 KB verbatim, so an unrounded rate used to
        # read "812.3456789 B/s".
        snap = app.DownloadSnapshot(
            files_done=1, files_total=2, bytes_done=100, bytes_total=2_000
        )

        detail = app.download_detail("org/model", snap, rate=812.3456789)

        self.assertIn("812 B/s", detail)
        self.assertNotIn("812.3", detail)

    def test_a_crawling_download_is_never_reported_as_stopped(self):
        # The meter has no rate at all for a download with nothing moving, so
        # this clause only runs while bytes arrive. Rounding a byte every few
        # seconds down to "0 B/s" would contradict the time left beside it.
        snap = app.DownloadSnapshot(
            files_done=1, files_total=2, bytes_done=100, bytes_total=2_000
        )

        detail = app.download_detail("org/model", snap, rate=0.25)

        self.assertIn("1 B/s", detail)
        self.assertNotIn("0 B/s", detail)

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

    class Manager(ModelManager):
        """A manager whose load does whatever the test hands it.

        The real claim bookkeeping is kept, so the card's own reservation is
        the one under test.
        """

        def __init__(self, work):
            super().__init__()
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

    def test_the_load_is_claimed_before_its_worker_starts(self):
        # Between the click and the worker's first instruction the manager
        # would otherwise look idle, and a removal or a redownload arriving
        # in that window could move the snapshot out from under the load.
        order = []

        class Manager(self.Manager):
            def reserve_load(self, model_id):
                order.append(("claimed", model_id))
                return super().reserve_load(model_id)

            def load(self, model_id, path, progress=None):
                order.append(("loaded", self.loading_id))
                return "CPU"

        app.MANAGER = Manager(lambda progress: "CPU")

        list(app.load_cached_model(self.MODEL))

        self.assertEqual(order, [("claimed", self.MODEL), ("loaded", self.MODEL)])
        self.assertIsNone(app.MANAGER.loading_id, "given back when the load ends")

    def test_a_worker_that_cannot_start_gives_the_claim_back(self):
        app.MANAGER = ModelManager()

        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("no threads")
        ):
            with self.assertRaisesRegex(RuntimeError, "no threads"):
                list(app.stream_load(self.MODEL, Path("/cache/snap")))

        self.assertIsNone(app.MANAGER.loading_id)

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


class MetricGlossaryTests(unittest.TestCase):
    """Every measurement in the detail panel says what it is on hover.

    The names mean nothing on first reading, and the README that explains
    them is not where the reader is. Attaching the sentence to the name puts
    the answer where the question is - including for a screen reader, which
    reads an ``abbr`` title out.
    """

    def scored(self) -> dict:
        log_probs = np.log(np.array([0.75, 0.25]))
        return build_metric(
            position=1,
            token_id=0,
            token_text="a",
            fallback_text="a",
            raw_log_probabilities=log_probs,
            sampled_probabilities=np.exp(log_probs),
            decode_token=str,
        ).to_dict()

    def test_each_measurement_carries_its_meaning(self):
        detail, _rows = app.describe_token(self.scored())

        for name, meaning in app.METRIC_GLOSSARY.items():
            with self.subTest(metric=name):
                self.assertIn(f">{name}</abbr>", detail)
                # Escaped once, into an attribute: an apostrophe in a
                # sentence must not be able to close it.
                self.assertIn(html.escape(meaning, quote=True), detail)

    def test_a_meaning_with_a_quote_in_it_stays_inside_the_attribute(self):
        # The sentences are written by hand today, but they are strings in a
        # dict; a quotation mark in one would otherwise end the attribute
        # early and spill the rest into the markup.
        with mock.patch.dict(
            app.METRIC_GLOSSARY, {"Surprise": 'the "surprise" of it'}, clear=False
        ):
            self.assertEqual(
                app.metric_term("Surprise"),
                '<abbr title="the &quot;surprise&quot; of it">Surprise</abbr>',
            )

    def test_an_unscored_token_says_why_instead_of_listing_measurements(self):
        detail, _rows = app.describe_token(
            unscored_metric(
                position=1, token_id=7, token_text="<s>", fallback_text="<s>"
            ).to_dict()
        )

        self.assertNotIn("<abbr", detail)


class HintTests(unittest.TestCase):
    """The panels' explanations fold to one line."""

    def test_a_hint_is_a_disclosure_that_starts_closed(self):
        folded = app.hint("What branching does", "It keeps the response.")

        self.assertTrue(folded.startswith('<details class="hint">'))
        self.assertNotIn("open", folded[: folded.index(">") + 1])
        self.assertIn("<summary>What branching does</summary>", folded)
        self.assertIn("It keeps the response.", folded)


class SamplingSummaryTests(unittest.TestCase):
    """The sampling accordion wears its own values, so it reads without opening."""

    def test_the_summary_names_every_knob_behind_it(self):
        summary = app.sampling_label(0.8, 0.95, 50, 1024)

        self.assertIn("temperature 0.8", summary)
        self.assertIn("top-p 0.95", summary)
        self.assertIn("top-k 50", summary)
        self.assertIn("1,024 new tokens", summary)

    def test_a_disabled_top_k_is_left_out_rather_than_shown_as_zero(self):
        # "top-k 0" reads as a setting of zero, which is the opposite of what
        # it means: zero is the control switched off.
        summary = app.sampling_label(1.0, 1.0, 0, 256)

        self.assertNotIn("top-k", summary)
        self.assertIn("temperature 1", summary)

    def test_the_summary_is_an_accordion_label_update(self):
        self.assertEqual(
            app.update_sampling_label(0.8, 0.95, 50, 1024),
            {"label": app.sampling_label(0.8, 0.95, 50, 1024), "__type__": "update"},
        )


class ScoreBudgetTests(unittest.TestCase):
    """The count under the Score text box, which turns a refusal into a number."""

    class Counting:
        loaded = True

        def __init__(self, answer, load_id="stub/model#1"):
            self.answer = answer
            self.load_id = load_id
            self.asked = []

        def count_score_tokens(self, text, *, context="", use_chat_template=False):
            self.asked.append((text, context, use_chat_template))
            return self.answer

    def budget(self, manager, context="", text="some text", template=False) -> str:
        original = app.MANAGER
        app.MANAGER = manager
        try:
            shown, _load_id = app.score_token_count(context, text, template)
            return shown
        finally:
            app.MANAGER = original

    def test_the_count_travels_with_the_load_it_was_counted_against(self):
        # A tokenizer belongs to the weights in memory, so a number counted
        # under one load says nothing about the next.
        original = app.MANAGER
        app.MANAGER = self.Counting((12, 4096), load_id="stub/model#7")
        try:
            self.assertEqual(
                app.score_token_count("", "some text", False),
                ("12 of 4,096 tokens.", "stub/model#7"),
            )
        finally:
            app.MANAGER = original

    def test_an_empty_box_is_not_counted_at_all(self):
        counting = self.Counting((0, 4096))

        self.assertEqual(self.budget(counting, text=""), app.SCORE_COUNT_HINT)
        self.assertEqual(counting.asked, [])

    def test_a_passage_within_the_limit_reads_as_a_fraction_of_it(self):
        self.assertEqual(self.budget(self.Counting((1200, 4096))), "1,200 of 4,096 tokens.")

    def test_a_passage_over_the_limit_says_so_before_the_press(self):
        budget = self.budget(self.Counting((5000, 4096)))

        self.assertIn("failure-text", budget)
        self.assertIn("5,000 tokens, above the 4,096", budget)
        self.assertIn("smaller pieces", budget)

    def test_a_count_that_cannot_be_had_says_so_rather_than_guessing(self):
        # No model, or one mid-response. A wrong number would be worse than
        # none: the whole point of the line is that it matches the check.
        self.assertEqual(self.budget(self.Counting(None)), app.SCORE_COUNT_UNKNOWN)

    def test_the_count_is_asked_for_exactly_what_would_be_scored(self):
        counting = self.Counting((10, 4096))

        self.budget(counting, context="before", text="passage", template=True)

        self.assertEqual(counting.asked, [("passage", "before", True)])


class ScoreBudgetRecoveryTests(unittest.TestCase):
    """The count that gave up has to come back on its own.

    A count asked for during a reply cannot have the model lock and says so.
    Nothing about that message corrects itself: the reply ends, the model
    goes idle, and the box still reads "not mid-response" until something is
    typed into it. The badge's timer is what un-sticks it.
    """

    class Counting:
        loaded = True

        def __init__(self, answer, load_id="stub/model#1"):
            self.answer = answer
            self.load_id = load_id
            self.asked = 0

        def count_score_tokens(self, text, *, context="", use_chat_template=False):
            self.asked += 1
            return self.answer

    def recover(self, manager, shown, counted_load="stub/model#1"):
        original = app.MANAGER
        app.MANAGER = manager
        try:
            return app.recover_score_budget(
                shown, counted_load, "", "some text", False
            )
        finally:
            app.MANAGER = original

    def test_a_count_that_is_stuck_is_recomputed(self):
        counting = self.Counting((12, 4096))

        recovered, load_id = self.recover(counting, app.SCORE_COUNT_UNKNOWN)

        self.assertEqual(recovered, "12 of 4,096 tokens.")
        self.assertEqual(load_id, "stub/model#1")
        self.assertEqual(counting.asked, 1)

    def test_a_model_swapped_out_from_another_tab_is_recounted(self):
        # The handlers that recompute on a load or an unload only reach the
        # tab that asked. This tab would otherwise go on advertising a number
        # counted under the old tokenizer, against the old context limit,
        # under the new model's badge.
        counting = self.Counting((12, 512), load_id="other/model#1")

        recovered, load_id = self.recover(
            counting, "99 of 4,096 tokens.", counted_load="stub/model#1"
        )

        self.assertEqual(recovered, "12 of 512 tokens.")
        self.assertEqual(load_id, "other/model#1")

    def test_an_unloaded_model_takes_the_count_down(self):
        counting = self.Counting(None, load_id=None)

        recovered, load_id = self.recover(
            counting, "99 of 4,096 tokens.", counted_load="stub/model#1"
        )

        self.assertEqual(recovered, app.SCORE_COUNT_UNKNOWN)
        self.assertIsNone(load_id)

    def test_a_count_that_is_fine_is_not_asked_for_again(self):
        # This runs every couple of seconds, so the ordinary case must not
        # pay for an encoding.
        counting = self.Counting((12, 4096))

        for shown in ("12 of 4,096 tokens.", app.SCORE_COUNT_HINT, ""):
            with self.subTest(shown=shown):
                self.assertEqual(
                    self.recover(counting, shown), (gr.skip(), gr.skip())
                )
        self.assertEqual(counting.asked, 0)

    def test_a_count_that_is_still_stuck_publishes_nothing(self):
        # Still mid-response, or still no model. Rewriting the same message
        # every tick would put a change event on the wire for nothing.
        counting = self.Counting(None)

        self.assertEqual(
            self.recover(
                counting, app.SCORE_COUNT_UNKNOWN, counted_load=counting.load_id
            ),
            (gr.skip(), gr.skip()),
        )

    def test_recovery_reads_the_boxes_it_would_score(self):
        # The recomputed count has to describe what is in the boxes now, not
        # what was there when the count gave up.
        (listener,) = [
            fn
            for fn in app.build_app().fns.values()
            if getattr(fn.fn, "__name__", None) == "recover_score_budget"
        ]

        self.assertEqual(len(listener.inputs), 5)
        self.assertEqual(listener.outputs, listener.inputs[:2])
        self.assertEqual(listener.show_progress, "hidden")


class ClearConfirmationTests(unittest.TestCase):
    """Clear deletes every conversation, so it asks first.

    It sits one button away from Undo and looks the same, and nothing brings
    the conversations back. Removing a model from disk already asks; this is
    the same question about the same kind of loss.
    """

    def forks(self, *names) -> dict:
        branches = {"Main": [], **{name: [] for name in names}}
        return {"active": "Main", "branches": branches}

    def turns(self) -> list[dict]:
        return [{"role": "user", "content": "hello"}]

    def test_an_empty_app_has_nothing_to_clear_and_asks_nothing(self):
        status, panel, question = app.ask_clear_chat([], self.forks())

        self.assertEqual(status, app.NOTHING_TO_CLEAR)
        self.assertEqual(panel, {"visible": False, "__type__": "update"})
        self.assertEqual(question, "")

    def test_the_question_counts_the_other_conversations_it_would_take(self):
        _status, panel, question = app.ask_clear_chat(
            self.turns(), self.forks("Fork 1", "Fork 2")
        )

        self.assertEqual(panel, {"visible": True, "__type__": "update"})
        self.assertIn("2 others", question)
        self.assertIn("cannot be undone", question)
        # And points at the control that takes only this one.
        self.assertIn("Delete", question)

    def test_one_other_conversation_is_named_in_the_singular(self):
        _status, _panel, question = app.ask_clear_chat(
            self.turns(), self.forks("Fork 1")
        )

        self.assertIn("1 other?", question)

    def test_a_lone_conversation_is_described_without_a_count(self):
        _status, _panel, question = app.ask_clear_chat(self.turns(), self.forks())

        self.assertIn("the conversation on screen?", question)
        self.assertNotIn("other", question)

    def test_asking_never_clears_anything(self):
        # The question is the whole of what the Clear button does. Nothing in
        # this handler touches the conversation, so a reader who opens it and
        # walks away still has everything.
        turns = self.turns()
        forks = self.forks("Fork 1")

        app.ask_clear_chat(turns, forks)

        self.assertEqual(turns, self.turns())
        self.assertEqual(forks, self.forks("Fork 1"))

    def test_confirming_clears_and_closes_the_question(self):
        result = app.clear_chat()

        self.assertEqual(result[-1], {"visible": False, "__type__": "update"})
        self.assertEqual(result[0], [])

    def test_cancelling_only_closes_the_question(self):
        self.assertEqual(app.hide_clear_confirm(), {"visible": False, "__type__": "update"})


class CachedLoadOutcomeTests(unittest.TestCase):
    """load_cached_model says whether it loaded, rather than leaving it to be
    inferred from what happens to be in memory afterwards."""

    def outcome(self, **cache):
        """Run the generator to exhaustion and return what it reported."""

        with mock.patch.object(app, "cache_status", return_value=CacheStatus(**cache)):
            generator = app.load_cached_model("org/model")
            try:
                while True:
                    next(generator)
            except StopIteration as stop:
                return stop.value

    def test_every_refusal_reports_that_nothing_was_loaded(self):
        for name, cache in [
            ("nothing cached", {}),
            ("part of it cached", {"cached_bytes": 1, "missing_files": ("x",)}),
            ("not a transformers model", {"cached_bytes": 1, "unsupported": True}),
        ]:
            with self.subTest(cache=name):
                self.assertIs(self.outcome(**cache), False)

    def test_a_load_that_raises_reports_that_nothing_was_loaded(self):
        with mock.patch.object(app.MANAGER, "find_cached", side_effect=OSError("no")):
            self.assertIs(self.outcome(cached_bytes=1), False)

    def test_a_load_that_works_reports_that_it_did(self):
        def loaded(model_id, path):
            yield "loading card"
            return "Apple Metal (MPS)"

        with mock.patch.object(app.MANAGER, "find_cached", return_value=Path("/tmp/x")):
            with mock.patch.object(app, "stream_load", side_effect=loaded):
                self.assertIs(self.outcome(cached_bytes=1), True)


class DefaultModelOfferTests(unittest.TestCase):
    """The offer beside the chat page's badge, and the route it takes.

    Until a model is loaded nothing on the chat page does anything. The badge
    says so; this is what to do about it, and it has to say which of the two
    things it would do before it is pressed.
    """

    def test_a_cached_default_offers_a_load(self):
        with mock.patch.object(app, "default_model_cached", return_value=True):
            offer = app.default_model_offer()

        self.assertIn("Load", offer)
        self.assertNotIn(app.DEFAULT_MODEL_DOWNLOAD, offer)

    def test_an_uncached_default_offers_a_download_and_says_how_large(self):
        # 15 GB is not a thing to find out about halfway through.
        with mock.patch.object(app, "default_model_cached", return_value=False):
            offer = app.default_model_offer()

        self.assertIn("Download", offer)
        self.assertIn(app.DEFAULT_MODEL_DOWNLOAD, offer)

    def run_setup(self, *, cached: bool, cached_load_works: bool = True):
        """Drive setup_default_model with both halves stubbed as generators.

        The cached half reports its own outcome, which is the whole point:
        setup_default_model must read that rather than ask the manager what
        is loaded now. So the stub returns it the way the real generator
        does, and no manager is involved at all.
        """

        def cached_half(*args, **kwargs):
            yield "cached card"
            return cached_load_works

        def download_half(*args, **kwargs):
            yield "download card"
            return True

        with mock.patch.object(app, "default_model_cached", return_value=cached):
            with mock.patch.object(
                app, "load_cached_model", side_effect=cached_half
            ) as load:
                with mock.patch.object(
                    app, "download_and_load_model", side_effect=download_half
                ) as fetch:
                    self.cards = list(app.setup_default_model("token"))
        return load, fetch

    def test_a_cached_default_is_loaded_without_touching_the_network(self):
        # The offer has just promised an immediate local load. Going through
        # the download path would reach snapshot_download for a Hub update
        # check: a stall at best, and a failure with no network at all.
        load, fetch = self.run_setup(cached=True)

        load.assert_called_once_with(settings.DEFAULT_MODEL_ID)
        fetch.assert_not_called()

    def test_the_load_is_believed_over_whatever_is_loaded_afterwards(self):
        # A load started in another tab can replace MANAGER.model_id between
        # the cached load's last frame and this handler resuming. Asking the
        # manager then would call a successful load a failure, download the
        # default again, and put it back over the model that other tab
        # deliberately chose. Nothing here reads the manager at all.
        class Swapped:
            model_id = "org/chosen-in-another-tab"
            load_id = "org/chosen-in-another-tab#1"

        original = app.MANAGER
        app.MANAGER = Swapped()
        try:
            _load, fetch = self.run_setup(cached=True)
        finally:
            app.MANAGER = original

        fetch.assert_not_called()

    def test_a_cache_that_would_not_load_falls_through_to_the_download(self):
        # missing_files checks the config and the weights alone, so a
        # download cut off before the tokenizer looks complete and fails in
        # from_pretrained. From this button that would be a dead end: the
        # reader asked for a working model and the rest is one fetch away.
        load, fetch = self.run_setup(cached=True, cached_load_works=False)

        load.assert_called_once_with(settings.DEFAULT_MODEL_ID)
        fetch.assert_called_once_with(settings.DEFAULT_MODEL_ID, "token")

    def test_the_fallback_says_what_it_is_doing(self):
        # A second progress card appearing unexplained would read as the
        # download the button promised was not needed.
        self.run_setup(cached=True, cached_load_works=False)

        self.assertTrue(any("Finishing the download" in card for card in self.cards))

    def test_an_uncached_default_is_downloaded(self):
        load, fetch = self.run_setup(cached=False)

        fetch.assert_called_once_with(settings.DEFAULT_MODEL_ID, "token")
        load.assert_not_called()

    def test_the_offer_cannot_be_redirected_by_the_id_box(self):
        # The chain writes the default into the ID box and then loads it, but
        # it takes the ID from settings rather than reading the box back, so a
        # box edited in between cannot send it elsewhere.
        _load, fetch = self.run_setup(cached=False)

        self.assertEqual(fetch.call_args.args[0], settings.DEFAULT_MODEL_ID)

    def test_pressing_the_offer_opens_the_page_that_reports_on_it(self):
        # The download's progress and any failure land on the Models page, so
        # the press goes there rather than leaving the reader on a chat page
        # that looks like nothing happened.
        model_id, page, *panes = app.start_default_model()

        self.assertEqual(model_id, settings.DEFAULT_MODEL_ID)
        self.assertEqual(page, app.MODELS_PAGE)
        self.assertEqual(
            [update["visible"] for update in panes], [False, False, True, False]
        )

    def test_a_cache_that_cannot_be_read_is_treated_as_no_cache(self):
        # The offer is a courtesy; an unreadable cache should make it offer
        # the download rather than raise on the chat page's first paint.
        for error in (OSError("nope"), ValueError("nope")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(app, "cache_status", side_effect=error):
                    self.assertFalse(app.default_model_cached())

    def test_a_partly_downloaded_default_is_not_offered_as_loadable(self):
        for status, cached in [
            (CacheStatus(cached_bytes=1, missing_files=("x",)), False),
            (CacheStatus(cached_bytes=1, unsupported=True), False),
            (CacheStatus(), False),
            (CacheStatus(cached_bytes=1), True),
        ]:
            with self.subTest(status=status):
                with mock.patch.object(app, "cache_status", return_value=status):
                    self.assertEqual(app.default_model_cached(), cached)
