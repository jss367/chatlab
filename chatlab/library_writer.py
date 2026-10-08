"""Coalesced streaming saves, with all disk work on one dedicated thread.

The first pending update starts a 500 ms deadline; later frames do not move
it. Each producer has one pending snapshot (and at most one in flight).
Normal completion/cancellation flush the final snapshot. Shutdown drains the
queue and rejects subsequent frames without waiting for model inference.

An abrupt process crash can lose 500 ms plus write/scheduling time of streamed
text under healthy storage. Stalled or failed storage cannot offer that bound.
Atomic replacement protects the previous file, not against power loss: the
existing library writer does not fsync. Explicit flushes report write failure.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from chatlab import library
from chatlab.conversation import copy_forks

logger = logging.getLogger(__name__)
SAVE_INTERVAL = 0.5


@dataclass
class SaveReceipt:
    done: threading.Event = field(default_factory=threading.Event)
    success: bool = False


class LibraryWriter:
    def __init__(self, interval=SAVE_INTERVAL):
        self.interval = interval
        self.condition = threading.Condition()
        self.pending = {}
        self.deadline = None
        self.closing = False
        self.failed = False
        self.worker = threading.Thread(target=self._work, name="chatlab-library-writer", daemon=True)
        self.worker.start()

    def submit(self, source, forks, path=None):
        # Capture the destination now: environment changes must not redirect
        # a queued save. Callers supply only the branch they own.
        target = Path(path or library.library_path()).absolute()
        snapshot = copy_forks(forks)
        with self.condition:
            if self.closing:
                receipt = SaveReceipt()
                receipt.done.set()
                return receipt
            key = (target, source)
            receipt = self.pending[key][1] if key in self.pending else SaveReceipt()
            self.pending[key] = (snapshot, receipt)
            if self.deadline is None:
                self.deadline = time.monotonic() + self.interval
            self.condition.notify_all()
            return receipt

    def flush(self, receipt):
        """Force a queued receipt to disk and wait, including an in-flight write."""
        with self.condition:
            if self.pending:
                self.deadline = time.monotonic()
                self.condition.notify_all()
        receipt.done.wait()
        return receipt.success

    def close(self):
        """Drain accepted frames; safe to call repeatedly or concurrently."""
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.worker.join()
        return not self.failed

    def _work(self):
        while True:
            with self.condition:
                while not self.pending:
                    if self.closing:
                        return
                    self.condition.wait()
                delay = (self.deadline or 0) - time.monotonic()
                if delay > 0 and not self.closing:
                    self.condition.wait(delay)
                    continue
                batch, self.pending = self.pending, {}
                self.deadline = None
            for (target, _), (snapshot, receipt) in batch.items():
                try:
                    receipt.success = library.write(snapshot, target, preserve_active=True, preserve_order=True) is not None
                except Exception:
                    logger.exception("Background conversation save failed for %s", target)
                if not receipt.success:
                    self.failed = True
                    logger.warning("Background conversation save failed for %s", target)
                receipt.done.set()


_WRITER = None
_LOCK = threading.Lock()
_CLOSED = False


def submit(source, forks, path=None):
    global _WRITER
    with _LOCK:
        if _CLOSED:
            receipt = SaveReceipt()
            receipt.done.set()
            return receipt
        if _WRITER is None:
            _WRITER = LibraryWriter()
        return _WRITER.submit(source, forks, path)


def flush(receipt):
    with _LOCK:
        writer = _WRITER
    return writer.flush(receipt) if writer else False


def shutdown():
    global _CLOSED
    with _LOCK:
        _CLOSED = True
        writer = _WRITER
    if writer:
        return writer.close()
    return True


atexit.register(shutdown)
