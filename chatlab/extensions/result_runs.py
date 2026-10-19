"""Session-scoped result publication and cancellation shared by measurement views.

Subclasses implement _stage(owner, result) to prepare their downloadable artifact.
"""
import logging
import threading
from uuid import uuid4

logger = logging.getLogger(__name__)


class ResultRuns:
    """One run per view, a way to stop it from another event, and whether its result may still be shown.

    Each view has a turn that moves when a run starts or a saved result is
    opened. A run publishes only while the turn it took is still the view's.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._next_turn = 0
        self._turns = {}
        self._sessions = {}
        self._downloads = {}

    @staticmethod
    def new_owner():
        return uuid4().hex

    def start(self, owner):
        with self._lock:
            self._next_turn += 1
            self._turns[owner] = self._next_turn
            active = self._sessions.pop(owner, None)
            if active is not None:
                active[1].cancel()
            return self._turns[owner]

    def attach(self, owner, turn, session):
        with self._lock:
            if self._turns.get(owner) != turn:
                session.cancel()
            else:
                self._sessions[owner] = (turn, session)

    def finish(self, owner, turn):
        with self._lock:
            if self._sessions.get(owner, (None,))[0] == turn:
                self._sessions.pop(owner)

    def cancel(self, owner):
        with self._lock:
            self._next_turn += 1
            self._turns[owner] = self._next_turn
            active = self._sessions.pop(owner, None)
            if active is not None:
                active[1].cancel()

    def live(self, owner, turn):
        with self._lock:
            return self._turns.get(owner) == turn

    def publish(self, owner, turn, result, render):
        """Stage and construct a response only while this turn is current.

        Rendering stays inside the same critical section as the turn check
        and file replacement. The publishing Gradio events also share a
        queue, so their responses cannot overtake each other after return.
        """
        with self._lock:
            if self._turns.get(owner) != turn:
                return None
            try:
                path = self._stage(owner, result)
            except OSError as exc:
                logger.warning("Could not write the measurement result: %s", exc)
                path = None
            return render(path)

    def _stage(self, owner, result):
        raise NotImplementedError("Provide a result download writer.")

    def forget(self, owner):
        with self._lock:
            active = self._sessions.pop(owner, None)
            if active is not None:
                active[1].cancel()
            self._turns.pop(owner, None)
            directory = self._downloads.pop(owner, None)
        if directory is not None:
            directory.cleanup()

