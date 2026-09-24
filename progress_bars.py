"""Live progress for a download or a load, read from the libraries' tqdm bars.

Neither ``snapshot_download`` nor ``from_pretrained`` offers a callback.
Both draw tqdm bars, so each is handed a silent bar class that records its
counts where the page, polling from another thread, can read them.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import NamedTuple


class DownloadSnapshot(NamedTuple):
    """One reading of a download: how many files and bytes are in, out of how many.

    ``bytes_total`` covers only files that need fetching; a file already in the
    cache finishes without ever reporting a size, so it counts in ``files_*``
    alone.
    """

    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0

    @property
    def started(self) -> bool:
        """Whether the Hub has answered with the file list yet."""

        return self.files_total > 0

    @property
    def fraction(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        return min(1.0, self.bytes_done / self.bytes_total)


class LoadSnapshot(NamedTuple):
    """One reading of a load: how much of the model is in memory yet.

    Two measures of the same weights: ``steps_*`` is the loader's own count
    of the ones it has read, ``bytes_done`` what the device allocator holds.
    :attr:`fraction` averages the two, because each covers half of a load
    onto Metal - the weights are read into host memory and copied across
    afterwards - and on a graphics card, where one pass does both at once,
    the average of two measures of that pass is still that pass.

    A load into host memory has no bytes to report, there being no allocator
    to ask, and counts in steps alone. ``bytes_total`` comes from the files,
    so it runs a little ahead of what a loaded model really occupies (tied
    weights are stored twice and loaded once), which the step count covers.
    """

    bytes_done: int = 0
    bytes_total: int = 0
    steps_done: int = 0
    steps_total: int = 0

    @property
    def started(self) -> bool:
        """Whether the loader has begun placing weights."""

        return self.bytes_done > 0 or self.steps_total > 0

    @property
    def counts_bytes(self) -> bool:
        """Whether this reading can be given in bytes."""

        return self.bytes_total > 0 and self.bytes_done > 0

    @property
    def fraction(self) -> float:
        measures = []
        if self.bytes_total > 0:
            measures.append(min(1.0, self.bytes_done / self.bytes_total))
        if self.steps_total > 0:
            measures.append(min(1.0, self.steps_done / self.steps_total))
        if not measures:
            return 0.0
        return sum(measures) / len(measures)


def _recording_bar_class(base: type, progress) -> type:
    """A silent tqdm subclass whose counts ``progress`` can read at any time.

    Neither the downloader nor the loader offers a callback: both report
    through tqdm, so both are watched by handing them a bar class that keeps
    its numbers where another thread can find them. ``progress`` supplies the
    ``_lock`` and the ``_register`` that receives each bar as it is built.

    A bar built this way answers ``n`` and ``total`` and nothing else. tqdm
    leaves a disabled bar half-initialized - no ``desc``, no rate - so a
    reader may only count.
    """

    class RecordingBar(base):
        def __init__(self, *args, **kwargs) -> None:
            self.counts_bytes = kwargs.get("unit") == "B"
            # A disabled bar never writes to the terminal, and tqdm's own
            # update() drops the count on the floor for one, so it is kept
            # here instead.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            progress._register(self)

        def update(self, n=1) -> None:
            if n:
                with progress._lock:
                    self.n += n

        def __iter__(self):
            # tqdm's own __iter__ skips counting for a disabled bar, and bars
            # that are consumed by iteration rather than advanced by update()
            # are on both paths: huggingface_hub before 1.25 hands the file
            # bar to tqdm's thread_map (which before tqdm 4.70 counted by
            # iterating), and the loader walks its bar over the weights.
            for item in self.iterable:
                yield item
                self.update(1)

    return RecordingBar


class DownloadProgress:
    """Live totals for one snapshot download, safe to read from another thread.

    ``snapshot_download`` reports through tqdm rather than callbacks: one bar
    over files advances as each finishes, and two byte bars (network transfer
    and bytes reconstructed on disk) grow their ``total`` as each file learns
    its size. :meth:`bar_class` gives it a silent tqdm that records those
    numbers here instead of drawing them.

    The byte bars belong to the snapshot, not to its files: since
    huggingface_hub 1.1 every per-file download feeds them through an internal
    aggregating stand-in, so the ``tqdm_class`` handed to ``snapshot_download``
    only ever sees the file bar and those snapshot-wide byte bars (one of them
    before 1.23, which added the transfer bar). That is why byte totals below
    are read as a maximum across byte bars rather than a sum: they are two
    views of the same bytes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: list = []

    def bar_class(self) -> type:
        """A tqdm class for ``snapshot_download(tqdm_class=...)`` that reports here."""

        from huggingface_hub.utils import tqdm as hub_tqdm

        return _recording_bar_class(hub_tqdm, self)

    def _register(self, bar) -> None:
        with self._lock:
            self._bars.append(bar)

    def snapshot(self) -> DownloadSnapshot:
        with self._lock:
            bars = list(self._bars)
            files_done = files_total = 0
            bytes_done = bytes_total = 0
            for bar in bars:
                count = int(bar.n or 0)
                total = int(bar.total or 0)
                if bar.counts_bytes:
                    # Transfer and reconstruction count the same bytes from
                    # the two ends of the pipe. Whichever is further along is
                    # the truer picture: a resumed file's on-disk bytes are
                    # credited to reconstruction only, and network bytes lead
                    # the disk for the rest.
                    bytes_done = max(bytes_done, count)
                    bytes_total = max(bytes_total, total)
                else:
                    files_done += count
                    files_total += total
        return DownloadSnapshot(files_done, files_total, bytes_done, bytes_total)


# The loader's progress bar over the weights, by transformers version:
# 5.x walks one bar over the parameters in core_model_loading, 4.x one over
# the checkpoint shards through the logging module's tqdm. Both are module
# attributes bound at import time, so both are replaced for a load.
#
# 4.x builds its bar only for a checkpoint of several shards, so a model kept
# in a single weight file draws none at all. Such a load is watched through
# the allocator alone, which is why LoadProgress.begin marks the start of a
# load rather than leaving the first bar to stand for it.
LOADER_BAR_ATTRIBUTES = (
    ("transformers.core_model_loading", "tqdm"),
    ("transformers.utils.logging", "tqdm"),
)


class LoadProgress:
    """Live progress for one load, safe to read from another thread.

    ``from_pretrained`` blocks until the last weight is in and offers no
    callback, so a load is watched from two sides. :meth:`watch` puts a silent
    bar in place of the loader's own, which counts the weights it has placed;
    :meth:`measure_bytes` reads the device allocator, which counts the bytes
    those weights take. :class:`LoadSnapshot` says why both are needed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: list = []
        self._total_bytes = 0
        self._sampler: Callable[[], int | None] | None = None
        self._baseline = 0
        self._started = False

    def measure_bytes(
        self, total_bytes: int | None, sampler: Callable[[], int | None]
    ) -> None:
        """Count bytes towards ``total_bytes``, reading them from ``sampler``.

        Call this once the device is settled and the model it replaces is
        gone: whatever the sampler reports now is the baseline, so memory in
        use for something else is not credited to this load. A ``total_bytes``
        of ``None`` (a snapshot whose weights could not be measured) leaves
        the load counted in steps alone.
        """

        baseline = sampler() or 0
        with self._lock:
            self._sampler = sampler
            self._total_bytes = max(0, int(total_bytes or 0))
            self._baseline = baseline

    def bar_class(self) -> type:
        """A tqdm class the loader can build its bar from that reports here."""

        from tqdm.auto import tqdm

        return _recording_bar_class(tqdm, self)

    def begin(self) -> None:
        """Note that the loader now has the weights and has started reading.

        The allocator is only worth reading once that is true: before it, a
        reading is whatever the device was already holding rather than this
        load's progress. The loader's first bar cannot stand in for this -
        transformers 4.x builds one only for a checkpoint of several shards -
        so the start of a load is recorded on its own.
        """

        with self._lock:
            self._started = True

    @contextlib.contextmanager
    def watch(self) -> Iterator[None]:
        """Report the loader's own progress bar here for the duration.

        The bars are module attributes, so this is the whole process's tqdm
        for as long as the load runs. A load holds the model lock, and nothing
        else in the app draws a transformers bar, so there is nobody else to
        surprise.
        """

        import importlib

        bar_class = self.bar_class()
        patched: list[tuple[object, str, object]] = []
        for module_name, attribute in LOADER_BAR_ATTRIBUTES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            original = getattr(module, attribute, None)
            if original is None:
                continue
            setattr(module, attribute, bar_class)
            patched.append((module, attribute, original))
        self.begin()
        try:
            yield
        finally:
            for module, attribute, original in patched:
                setattr(module, attribute, original)

    def _register(self, bar) -> None:
        with self._lock:
            self._bars.append(bar)
            # A bar is the loader saying it has begun, for a caller that
            # watched the load without going through watch().
            self._started = True

    def snapshot(self) -> LoadSnapshot:
        with self._lock:
            sampler = self._sampler
            baseline, total = self._baseline, self._total_bytes
            steps_done = sum(int(bar.n or 0) for bar in self._bars)
            steps_total = sum(int(bar.total or 0) for bar in self._bars)
            # Bytes are counted from the moment the loader starts reading,
            # and not before: between the baseline and the first weight the
            # allocator has nothing to say about this load. A single-file
            # checkpoint under transformers 4.x never builds a bar, so the
            # allocator is all such a load has to report with.
            if not self._started:
                sampler = None
        # Sampled outside the lock: the allocator holds a lock of its own that
        # the loading thread has for much of a load.
        bytes_done = max(0, (sampler() or 0) - baseline) if sampler else 0
        return LoadSnapshot(bytes_done, total, steps_done, steps_total)
