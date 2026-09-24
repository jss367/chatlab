"""The one model in memory, and who may use it.

:class:`ModelManager` owns the single loaded model: it downloads, loads,
generates, scores and inspects, and it decides which of those may run at a
time. The work itself lives in the modules its mixins come from -
:mod:`model_loading`, :mod:`text_generation` and :mod:`model_inspection` -
over the cache in :mod:`model_cache`, the device in :mod:`device_memory`, and
the tokenizer handling in :mod:`tokenization`. What stays here is the state
they share and the claims and locks that keep a generation, a load and a
download from stepping on each other.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

from chatlab import device_memory
from chatlab import model_cache
from chatlab import steering as steering_vectors
from chatlab.device_memory import first_line, memory_note, reraise_out_of_memory
from chatlab.model_cache import (
    IMAGE_KIND,
    ModelBusy,
    ModelDownloading,
    ModelLoaded,
    format_bytes,
    validate_model_id,
)
from chatlab.model_inspection import InspectionMixin
from chatlab.model_loading import LoadedModel, LoadingMixin
from chatlab.progress_bars import DownloadProgress, LoadProgress, LoadSnapshot
from chatlab.text_generation import GenerationMixin
from chatlab.torch_engine import TorchEngine

logger = logging.getLogger(__name__)


# What has the model, when something does. One generation and one load are
# refused for each other as well as for themselves, so a refusal that only
# says "busy" is wrong half the time: a reader told to wait for a response
# that is not running, or to press a Stop button that is not there, looks for
# something that is not on the page. Every refusal a reader sees is worded
# from one of these; see ModelManager.claim_generation.
GENERATING = "generating"
LOADING = "loading"


class ExclusiveLoad(NamedTuple):
    """What :meth:`ModelManager.claim_exclusive_load` answers with.

    Exactly one side is filled in. ``claim`` is what :meth:`reserve_load`
    returns - the checked ID and the claim number that gives it back - and
    ``held`` is ``None`` beside it. A refusal is the other way round: no
    claim, and :data:`LOADING` or :data:`GENERATING` naming what has the
    model, decided in the same step as the refusal so that a load ending an
    instant later cannot change the answer on screen.
    """

    claim: tuple[str, int] | None
    held: str | None


class ModelManager(LoadingMixin, GenerationMixin, InspectionMixin):
    """Own the single in-memory model used by the local application."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        # :meth:`hidden_token_ids` for the load named beside it; see
        # :meth:`_hidden_ids`.
        self._hidden_ids_cache: tuple[str | None, frozenset[int]] = (None, frozenset())
        # :meth:`_stop_token_ids` for the load named beside it.
        self._stopping_ids_cache: tuple[str | None, frozenset[int]] = (None, frozenset())
        # The diffusers pipeline, when the model in memory is an image model.
        # It stands apart from ``model`` rather than sharing the slot because
        # everything that generates text asks :attr:`loaded`, and a pipeline
        # answering that question yes would be fed tokens.
        self.pipeline = None
        # How the model in memory is run: an :class:`mlx_runtime.MlxEngine`
        # for an MLX checkpoint, and ``None`` for a Transformers model, which
        # :meth:`_engine` wraps in a :class:`TorchEngine` on demand. Left
        # unset for a Transformers model so that anything that puts a model
        # in ``model`` by hand - the tests do - is run the way it always was.
        self.engine = None
        self.kind: str | None = None
        self.model_id: str | None = None
        self.local_path: Path | None = None
        self.device_name: str | None = None
        self.precision: str | None = None
        # What the load estimated the weights would take. Host memory keeps
        # no allocator figure, so this is what a fit verdict gives back when
        # judging a model that would replace this one.
        self.loaded_bytes: int | None = None
        # Counts successful loads, so state produced under one set of weights
        # can be told from state produced under the next even when both came
        # from the same repository ID (a re-download at a newer revision).
        self.load_count = 0
        # A fitted lens belongs to one load, including its exact checkpoint.
        # The opaque import ID prevents another browser tab using its replacement.
        self._jacobian_lens = None
        # What is in memory, kept as one value beside the fields above so it
        # can be read without straddling a load; see :meth:`loaded_model`.
        self._loaded = LoadedModel()
        self._loaded_lock = threading.Lock()
        # Loads claimed and not finished yet, by claim number. ``model_id``
        # is cleared for the whole of a load and set only once the weights
        # are in, so on its own it says nothing about the minutes in
        # between; anything that must not touch a model's files while they
        # are being read (a redownload, say) asks these as well.
        #
        # More than one claim can stand at once. The model lock serializes
        # the loads themselves, but a second click lands its claim while the
        # first load is still running, and a claim numbered this way is only
        # ever given back by the load that took it.
        self._load_claims: dict[int, str] = {}
        self._next_claim = 0
        # The live progress of each claimed load that has one, by the same
        # claim number, so a page that did not start the load can still show
        # how far it has come. The card streaming beside a load reaches only
        # the tab that asked for it; the chat page's badge is on a timer and
        # has nothing but the manager to read.
        self._load_progress: dict[int, LoadProgress] = {}
        # Guards the claims, so that taking one and reading them are each a
        # single step for handlers running on several workers.
        self._claims_lock = threading.Lock()
        # The one load that has the model lock in hand and is reading
        # weights right now, as opposed to claimed and queued behind
        # something. Kept apart from the claims because the order loads are
        # claimed in need not be the order they win the lock in, and the
        # badge on the chat page names the load that is really reading.
        self._active_load: str | None = None
        # Counts the changes this process has made to what a scan of the
        # Hugging Face cache would find: a download that started or finished,
        # a model deleted. A download that is running counts because a model
        # whose folder is being written cannot be loaded, so it is not
        # something to offer. Read by anything drawn from such a scan and too
        # dear to redo on a timer - the chat page's model switcher - so a tab
        # that did not make the change can tell "nothing has moved" from
        # "rescan" in an attribute read. A change made outside ChatLab does
        # not move it, which is the bargain My Models already makes with its
        # Refresh button.
        self.cache_revision = 0
        self._cache_revision_lock = threading.Lock()
        # Downloads under way right now, by model ID, so a second request for
        # the same model can follow the first instead of racing it for the
        # same files.
        self.active_downloads: dict[str, DownloadProgress] = {}
        # Guards active_downloads, so that "is anyone fetching this?" and "then
        # I am" happen as one step: two handlers asking at the same instant
        # must come away with one download between them, not one each.
        self._downloads_lock = threading.Lock()
        # A generation holds this lock across streaming yields. Gradio may run
        # the next step (or close the generator) on a different worker thread,
        # so this cannot be an RLock: its owner check would reject that second
        # thread's release and leave the model permanently wedged. A plain Lock
        # still excludes loads, unloads, scoring, and inspection, but permits
        # the worker that resumes the stream to release it.
        self._lock = threading.Lock()
        # A separate flag records "a generation is running right now" without
        # making callers contend for the model lock just to ask.
        #
        # A plain Lock, deliberately: it is acquired and released by whichever
        # worker thread happens to be running the generator at the time, and
        # Gradio is free to resume a streaming handler on a different thread
        # than the one that started it. An RLock, or any owner-checked
        # primitive, would refuse the release from that second thread.
        self._generating = threading.Lock()
        # What a response's record needs and cannot read afterwards: the
        # model and prompt size of the run under way, so a run that raises
        # before it publishes anything still has both, and what the device
        # held at the end, read under the lock above for the reason given
        # where it is set. One slot each is enough: that lock admits one
        # generation at a time.
        self._run_note: tuple[str | None, int] | None = None
        self._run_device_bytes: int | None = None
        # The key-value cache the last inspection left behind, with the load
        # it belongs to and the tokens it covers. See _inspect_cache_for().
        self._inspect_cache: tuple[str, list[int], Any] | None = None
        # How Stop reaches the image run that is drawing right now, and only
        # that one. It belongs to the run holding the generation slot rather
        # than to the page, because button visibility is per browser tab: a
        # second tab can press Draw while the first is still drawing, and a
        # token the page owned would be cleared by that second attempt even
        # though it is refused, losing the first tab's cancellation.
        self._image_cancel: threading.Event | None = None

    def loaded_model(self) -> LoadedModel:
        """What is in memory: the model, its device, its precision, its load.

        One reading of the four, so a caller reporting them together cannot
        straddle a load and describe one model with another's device. Empty
        fields where nothing is loaded.
        """

        with self._loaded_lock:
            return self._loaded

    @property
    def loaded(self) -> bool:
        """Whether a text model is in memory, ready to be fed tokens."""

        return self.model is not None and self.tokenizer is not None

    @property
    def image_loaded(self) -> bool:
        """Whether an image pipeline is in memory, ready to be given a prompt."""

        return self.pipeline is not None

    @property
    def in_memory(self) -> bool:
        """Whether a model of either kind is in memory."""

        return self.loaded or self.image_loaded

    @property
    def load_id(self) -> str | None:
        """Identify the weights in memory: the model ID plus which load this is.

        Two loads of the same repository ID can hold different snapshots, so
        anything that must be read back by the model that produced it is
        stamped with this rather than the ID alone. ``None`` when nothing is
        loaded, of either kind: an image run's readings are stamped with this
        too, so a picture's maps can be told from the next load's.
        """

        if not self.in_memory:
            return None
        return f"{self.model_id}#{self.load_count}"

    @property
    def busy(self) -> bool:
        """True while the generation slot is reserved.

        Never blocks, so it can only ever be an early exit: a caller that is
        about to generate has to take the slot with reserve_generation()
        rather than act on this answer.
        """

        return self._generating.locked()

    def _occupant_locked(self) -> str | None:
        """What has the model, answered with :attr:`_claims_lock` already held.

        The one place the question is decided, so the property that only
        labels and the two claims that refuse cannot name different things.
        Both can be true at once - :meth:`reserve_load` does not exclude a
        reply already streaming - and the load wins, because it is what the
        reply is about to lose the model to, and because a load is the one a
        reader can only wait out rather than stop.
        """

        if self._load_claims or self._active_load is not None:
            return LOADING
        if self._generating.locked():
            return GENERATING
        return None

    @property
    def occupant(self) -> str | None:
        """What has the model right now: :data:`LOADING`, :data:`GENERATING`, or nothing.

        The same question :meth:`claim_generation` settles, asked without
        taking anything, so it is only ever an early exit or a label - a
        caller about to generate takes the slot and reads the answer the
        claim gives back, and a caller about to load reads the one
        :meth:`claim_exclusive_load` gives back.
        """

        with self._claims_lock:
            return self._occupant_locked()

    def claim_generation(self) -> str | None:
        """Claim the right to run a generation, or name what has the model.

        ``None`` when the slot is now the caller's. Otherwise
        :data:`LOADING` or :data:`GENERATING`, decided in the same step as
        the refusal, which is the point of returning it here rather than
        leaving each caller to read :attr:`occupant` afterwards: by then the
        load can have finished and the refusal on screen would name the
        wrong thing. Every refusal a reader sees is worded from this, and
        "wait for the response to finish" is a lie when no response is
        running.

        Everything :meth:`reserve_generation` says about reserving applies
        here; that method is this one with the reason thrown away.
        :meth:`claim_exclusive_load` is the same bargain on the load side.
        """

        with self._claims_lock:
            held = self._occupant_locked()
            if held is not None:
                return held
            # Cannot fail: every acquire of the slot is made under this lock
            # and nothing holds it, so no other thread can be between the
            # two lines. Releasing happens outside the lock, but that only
            # ever frees the slot.
            self._generating.acquire(blocking=False)
            return None

    def reserve_generation(self) -> bool:
        """Claim the right to run a generation, or report that it is taken.

        Never blocks: a caller that loses the race must refuse, not queue.
        Queuing is what corrupts the conversation - a handler that waited would
        resume holding the inputs Gradio captured when its click was queued,
        and write that stale snapshot over everything the running generation
        produced in the meantime.

        The caller must reserve *before* publishing its first frame and release
        in a ``finally``. Checking :attr:`busy` and then generating is not the
        same thing: those two steps are separated by a yield, and Gradio does
        not resume a streaming handler until the browser has been sent the
        frame, so the window between them is a network round trip wide.

        A claimed load refuses this too, and under the same lock the load was
        claimed with, so the two decisions cannot both say yes. A generation
        admitted while a load is under way does not run on the model the
        reader was looking at: it waits on the model lock behind the load and
        then answers from whatever the load brought in. Refusing it is the
        only answer that keeps the reply and the badge agreeing. The
        :meth:`claim_exclusive_load` side of the same rule is what stops a
        load starting while a reply is streaming.

        A caller that has to tell the reader why it refused wants
        :meth:`claim_generation`, which is this with the reason kept.
        """

        return self.claim_generation() is None

    def release_generation(self) -> None:
        """Give the generation slot back. Pairs with a successful reservation."""

        self._generating.release()

    def download(
        self,
        model_id: str,
        hf_token: str | None = None,
        progress: DownloadProgress | None = None,
        revision: str | None = None,
    ) -> Path:
        """Fetch ``model_id`` into the Hugging Face cache and return its snapshot.

        Blocks until the last byte; ``progress`` is how a caller on another
        thread watches it happen. Files already cached are skipped, and a
        partial file left by an interrupted download is resumed. ``revision``
        is the branch, tag or commit to fetch, the default branch when
        ``None``; an adapter's pinned base is the one caller that sets it.

        ``progress`` is listed in :attr:`active_downloads` for as long as this
        runs. A caller that already listed it through :meth:`reserve_download`
        keeps that entry; one that did not gets it added here, unless another
        download of the same model is already listed, which is left alone.
        """

        checked_id = validate_model_id(model_id)
        progress = progress or DownloadProgress()
        token = hf_token.strip() if hf_token and hf_token.strip() else None
        started = time.monotonic()
        # A download is the longest thing the app does and the one most likely
        # to be interrupted, and until now it left no trace at all: a cache
        # holding a partial 14 GB repo looked the same in the log as one that
        # was never asked for.
        logger.info(
            "Downloading %s%s from the Hub%s",
            checked_id,
            f" at revision {revision}" if revision else "",
            " with an access token" if token else "",
        )
        try:
            self._list_download(checked_id, progress)
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=checked_id,
                revision=revision,
                token=token,
                tqdm_class=progress.bar_class(),
            )
        except Exception as error:
            reached = progress.snapshot()
            logger.warning(
                "Download of %s stopped after %.1fs at %s of %s files, %s of %s: %s",
                checked_id,
                time.monotonic() - started,
                reached.files_done,
                reached.files_total,
                format_bytes(reached.bytes_done),
                format_bytes(reached.bytes_total),
                first_line(error),
            )
            raise
        finally:
            # The end of the download is noted by release_download, which
            # covers both halves of what changed: the files that landed, and
            # the model becoming loadable again now that nothing is writing
            # its folder.
            self.release_download(checked_id, progress)
        landed = progress.snapshot()
        logger.info(
            "Downloaded %s in %.1fs: %s files, %s, cached at %s",
            checked_id,
            time.monotonic() - started,
            landed.files_total or landed.files_done,
            format_bytes(landed.bytes_done),
            path,
        )
        return Path(path)

    def note_cache_change(self) -> None:
        """Record that what a scan of the cache would find has changed.

        A download that started or ended, a model deleted: anything that
        changes which models a reader can be offered. See
        :attr:`cache_revision`.
        """

        with self._cache_revision_lock:
            self.cache_revision += 1

    def _list_download(
        self, checked_id: str, progress: DownloadProgress
    ) -> DownloadProgress | None:
        """List ``progress`` as ``checked_id``'s download unless one already is.

        Returns the download that was already listed, or ``None`` when this
        call is the one that listed ``progress`` - which is also when the
        cache revision moves: a model being written is a model that cannot be
        loaded, so the lists drawn from a cache scan have to hear about it.
        """

        with self._downloads_lock:
            running = self.active_downloads.get(checked_id)
            if running is not None:
                return running
            self.active_downloads[checked_id] = progress
        self.note_cache_change()
        return None

    def reserve_download(self, model_id: str) -> tuple[DownloadProgress, bool]:
        """Claim ``model_id`` for a new download, or point at the one running.

        Returns ``(progress, reserved)``. When ``reserved`` is true the caller
        owns the download: it must pass ``progress`` to :meth:`download`, whose
        ``finally`` removes the entry. When false, another caller is fetching
        the model and ``progress`` is theirs to watch.

        The lookup and the reservation are one atomic step. Checking
        :attr:`active_downloads` and then starting a worker is not: the worker
        registers itself only once it reaches :meth:`download`, and two
        handlers (say **Download** and **Download and load**) clicked together
        would both find the table empty in that gap and fetch the same files
        twice.
        """

        checked_id = validate_model_id(model_id)
        progress = DownloadProgress()
        running = self._list_download(checked_id, progress)
        if running is not None:
            return running, False
        return progress, True

    def downloading_ids(self) -> frozenset[str]:
        """Every model being downloaded right now, read as one step.

        For a caller filtering a list: iterating :attr:`active_downloads`
        itself would race a download starting or ending, which in CPython is
        a "dictionary changed size during iteration" in the middle of
        drawing a page.
        """

        with self._downloads_lock:
            return frozenset(self.active_downloads)

    def release_download(self, model_id: str, progress: DownloadProgress) -> None:
        """Remove ``progress`` only when it still owns ``model_id``'s entry."""

        checked_id = validate_model_id(model_id)
        with self._downloads_lock:
            removed = self.active_downloads.get(checked_id) is progress
            if removed:
                del self.active_downloads[checked_id]
        if removed:
            self.note_cache_change()

    @property
    def loading_id(self) -> str | None:
        """The model a load is bringing in right now, or ``None``.

        The load that has the model lock in hand, when one has: it is the
        one really reading weights, whatever order the claims came in. That
        matters to the badge on the chat page, which names this load while
        memory stands empty. Failing that the most recently claimed load,
        which covers the moment between a load being asked for and it taking
        the lock. Ask :meth:`is_loading` about a particular model rather than
        comparing against this.
        """

        with self._claims_lock:
            if self._active_load is not None:
                return self._active_load
            return next(reversed(self._load_claims.values()), None)

    def is_loading(self, model_id: str) -> bool:
        """Whether a load of ``model_id`` is claimed and not finished."""

        checked_id = validate_model_id(model_id)
        with self._claims_lock:
            return checked_id in self._load_claims.values()

    def reserve_load(self, model_id: str) -> tuple[str, int]:
        """Claim a load of ``model_id``: its checked ID and the claim number.

        Claimed before waiting for the model lock, not after: a load queued
        behind a long generation is a load under way for the whole wait. A
        caller that runs :meth:`load` on another thread claims it here first,
        because the worker names the load only once it reaches ``load`` and
        takes the lock later still, and in that window a removal or a
        redownload would find the manager idle and move the snapshot out from
        under the load. This is what :meth:`reserve_download` is for
        downloads.

        The claim number is what gives the claim back, so two overlapping
        loads cannot clear each other's. Whoever takes one must release it:
        ``load`` releases its own, and a caller that claimed a load for
        another thread releases that claim when the thread is done with it.
        """

        checked_id = validate_model_id(model_id)
        with self._claims_lock:
            self._next_claim += 1
            self._load_claims[self._next_claim] = checked_id
            return checked_id, self._next_claim

    def claim_exclusive_load(self, model_id: str) -> ExclusiveLoad:
        """Claim a load of ``model_id``, or name what has the model instead.

        An :class:`ExclusiveLoad` holding what :meth:`reserve_load` returns,
        or, when another load is claimed or a generation is running and the
        caller must refuse rather than queue, holding :data:`LOADING` or
        :data:`GENERATING` instead. Why the reason comes back from here is
        what :meth:`claim_generation` says: a caller that refused and then
        read :attr:`occupant` to find out why has asked twice, and a load
        that ended in between answers the second question with the other
        reason - so a load turned away by a load would tell the reader a
        response was running and to press a Stop button that is not on the
        page. This is the gate every load a reader asks for goes
        through, so "no other load is claimed" really means no other load:
        once this returns a claim, the ordinary :meth:`reserve_load` that the
        load itself makes is the only one that can appear, and the next
        reader to reach this gate sees that claim and is turned away.

        Refusing rather than queuing is the point. Reading :attr:`loading_id`
        or :attr:`busy` and then loading is two steps, and a handler that
        yields between them - as a streaming one must, to show its first card
        - leaves a window a whole browser round trip wide for a second load
        to be claimed. Both halves of the question are answered here under
        :attr:`_claims_lock`, and :meth:`reserve_generation` answers the
        mirror image under the same lock, so a load and a reply can never
        both be admitted: one of them sees the other.

        The claim stands from here, so the caller owns it and must give it
        back with :meth:`release_load` in a ``finally``; the load it goes on
        to start takes its own claim and releases that one itself.
        """

        checked_id = validate_model_id(model_id)
        with self._claims_lock:
            held = self._occupant_locked()
            if held is not None:
                return ExclusiveLoad(None, held)
            self._next_claim += 1
            self._load_claims[self._next_claim] = checked_id
            return ExclusiveLoad((checked_id, self._next_claim), None)

    def release_load(self, claim: int) -> None:
        """Give back one claim, leaving any other load's standing."""

        with self._claims_lock:
            self._load_claims.pop(claim, None)
            self._load_progress.pop(claim, None)

    def note_load_progress(self, claim: int, progress: LoadProgress) -> None:
        """Publish ``claim``'s progress, for pages that did not start the load.

        The reader who picks a model in the chat page's switcher watches the
        badge beside it, and the badge is redrawn by a timer that has only
        this manager to read: the cards a load streams go back to the one
        handler that started it. Registering the progress object here is what
        lets any tab, on any page, say how far the load has come.

        Kept by claim number, as the claim itself is, so
        :meth:`release_load` clears both together and an overlapping load
        cannot drop another's readings.
        """

        with self._claims_lock:
            self._load_progress[claim] = progress

    def loading_progress(self) -> LoadSnapshot | None:
        """How far the load :attr:`loading_id` names has come, or ``None``.

        The load reading weights right now when one is, matching
        :attr:`loading_id` claim for claim, so the figure the badge shows
        belongs to the model it names. A load that has been claimed but has
        not reached :func:`stream_load` yet has published nothing and is
        reported as ``None``; so is a load whose reader is on the page that
        started it, for whom there is nothing to answer.
        """

        with self._claims_lock:
            active = self._active_load
            found = None
            for claim, model_id in reversed(self._load_claims.items()):
                progress = self._load_progress.get(claim)
                if progress is None:
                    continue
                if active is None or model_id == active:
                    found = progress
                    break
        # Snapshotting reads the device allocator, which takes a lock the
        # loading thread holds for much of a load, so it happens with this
        # manager's own lock let go.
        return None if found is None else found.snapshot()

    def remove(self, model_id: str, cache_dir: Path | None = None) -> int:
        """Delete ``model_id``'s cache folder, unless the manager is using it.

        Removal is serialized with everything else that touches the files.
        The model lock is taken for the whole deletion, so a load that is
        still reading the folder (``model_id`` is assigned only once
        ``from_pretrained`` returns) cannot have its files pulled from under
        it, and no load can start until the folder is gone. The downloads
        lock is held too, so a download cannot be reserved for the model
        between the check and the deletion. Looking at ``model_id`` and
        ``active_downloads`` first and deleting afterwards would leave exactly
        that gap: the interface runs its handlers on several workers.

        A load claimed through :meth:`reserve_load` refuses the removal even
        before it reaches the lock, because a load that runs on its own thread
        is under way from the click, not from its first weight.

        The model lock is never waited for. It is held across a whole
        generation, and a handler blocked on it would tie up a worker for as
        long as the reply takes, so a busy manager raises :class:`ModelBusy`
        and the reader tries again when the model is idle.
        """

        checked_id = validate_model_id(model_id)
        if not self._lock.acquire(blocking=False):
            raise ModelBusy(
                f"{checked_id} cannot be removed while a model is loading, "
                "generating, scoring, or being inspected."
            )
        try:
            if self.model_id == checked_id:
                raise ModelLoaded(f"{checked_id} is loaded in memory.")
            claimed = self.loading_id
            if claimed is not None:
                # A load claimed on another thread that has not reached the
                # lock yet. The lock alone would let this deletion through in
                # the window between the claim and the first weight.
                raise ModelBusy(
                    f"{checked_id} cannot be removed while {claimed} is being loaded."
                )
            with self._downloads_lock:
                if checked_id in self.active_downloads:
                    raise ModelDownloading(f"{checked_id} is being downloaded.")
                freed = model_cache.remove_cached_model(checked_id, cache_dir)
                self.note_cache_change()
                return freed
        finally:
            self._lock.release()

    def _engine(self):
        """What runs the model in memory; see :attr:`engine`."""

        return self.engine if self.engine is not None else TorchEngine(self.model)

    def _steering(self, steering: dict | None):
        """Install a steering vector for one run; see :mod:`steering`.

        Steering adds its vector through a ``torch.nn.Module`` forward hook on
        a decoder block, which only the Transformers backend has: an mlx-lm
        model is not a torch module, so the hook has nowhere to go. Refuse
        here, naming the backend, rather than letting the block search fail
        with an architecture complaint the reader cannot act on. An inactive
        vector - none imported, the switch off, or a strength of zero - runs
        on either backend, because nothing is added.
        """

        engine = self._engine()
        if getattr(engine, "backend", "torch") != "torch" and steering_vectors.active(steering):
            raise steering_vectors.SteeringError(
                "Steering is not supported for MLX models. Load the model's "
                "unquantized Transformers version to steer it, or turn steering off."
            )
        return steering_vectors.applied(self.model, self.model_id, steering)

    def check_steering(self, steering: dict | None) -> None:
        """Refuse a vector the loaded model cannot take, without installing it.

        The same questions :meth:`_steering` asks when a response starts - the
        backend, the vector's model, its layer and its width - asked ahead of
        time, for a caller that only turns steering on partway through a run
        and would otherwise learn at that response that it never could. No
        hook is installed, so this needs no model lock.
        """

        if not steering_vectors.active(steering):
            return
        if getattr(self._engine(), "backend", "torch") != "torch":
            raise steering_vectors.SteeringError(
                "Steering is not supported for MLX models. Load the model's "
                "unquantized Transformers version to steer it, or turn steering off."
            )
        steering_vectors.validate_model(self.model, self.model_id, steering_vectors.expand(steering))

    def start_image_run(self) -> threading.Event:
        """Claim the generation slot for an image run and publish its cancel token.

        For a caller that has to hold both *before* it publishes anything,
        which a streaming handler does: Gradio does not resume it until the
        browser has been sent its first frame, so a run reserved after that
        frame leaves a network round trip in which the page shows a Stop
        button over nothing reserved, ``stop_image_run`` reports that nothing
        is drawing, and a load arriving in between can replace the pipeline
        the page checked. The Chat page reserves before its first frame for
        the same reason; see :meth:`reserve_generation`.

        The caller must release with :meth:`finish_image_run` in a
        ``finally``. :meth:`generate_image` picks up a run started this way
        rather than starting a second one.

        A load holds this off as well as a generation, and the refusal says
        which: an image page told to wait for a run to finish while a model
        is loading sends the reader looking for a Stop button nothing is
        under.
        """

        held = self.claim_generation()
        if held == LOADING:
            raise ModelBusy("A model is loading. Wait for it to finish.")
        if held is not None:
            raise ModelBusy("The model is busy. Wait for the current run to finish.")
        cancel = threading.Event()
        self._image_cancel = cancel
        return cancel

    def finish_image_run(self) -> None:
        """Give back what :meth:`start_image_run` took. Pairs with it."""

        self._image_cancel = None
        self.release_generation()

    def stop_image_run(self) -> bool:
        """Ask the image run that is drawing to stop, and say whether one was.

        The token belongs to the run holding the generation slot, so a Stop
        pressed while nothing is drawing is a no-op rather than something a
        later run inherits, and a second tab's refused Draw cannot clear the
        first tab's cancellation.
        """

        cancel = self._image_cancel
        if cancel is None:
            return False
        cancel.set()
        return True

    def generate_image(self, request, *, on_step=None, cancel=None, expected_load_id=None):
        """Draw ``request`` with the pipeline in memory, and report what happened.

        Blocks until the picture is finished; ``on_step`` is called with each
        step's readings from this thread, which is how a caller watching from
        another one shows the trajectory arriving. ``request`` is an
        :class:`image_runtime.ImageRequest` and the answer an
        :class:`image_runtime.ImageRun`.

        Holds the same two things a text generation holds: the generation
        slot, so two runs cannot start at once, and the model lock, so a load
        or an unload cannot pull the pipeline out from under one. Which means
        a reply and a picture exclude each other, as they must - there is one
        model in memory and one device under it.

        The cancel token is made here rather than handed in, and published
        only once the slot is held: see :meth:`stop_image_run`. So a caller
        that loses the race for the slot never touches the running run's
        token, and the token is gone again the moment the run ends.

        A caller that had to reserve before it could publish anything has
        already done both through :meth:`start_image_run` and hands its
        token back as ``cancel``; this then runs on that reservation and
        gives it up when the run ends. Handed in rather than guessed at: a
        slot that is already taken is either that caller's own or another
        run's, and the two must not be confused.

        The run is stamped with the load that drew it, so a maps-and-steps
        readout can be told apart from one the next load produced.

        ``expected_load_id`` pins a word-removal comparison to its original
        model load and requires a pipeline that accepts a seeded generator.
        Both checks happen under the model lock before inference begins.
        """

        from chatlab import image_runtime

        if cancel is None:
            cancel = self.start_image_run()
        started = time.monotonic()
        run = None
        try:
            with self._lock:
                if self.pipeline is None:
                    raise RuntimeError("No image model is loaded.")
                if expected_load_id is not None:
                    if self.load_id != expected_load_id:
                        raise image_runtime.Unwatchable(
                            "The loaded model changed. Draw a new original before testing a word."
                        )
                    if "generator" not in image_runtime._call_arguments(
                        self.pipeline, request, None, None
                    ):
                        raise image_runtime.Unwatchable(
                            "This pipeline cannot accept a fixed seed, so word comparisons "
                            "are unavailable."
                        )
                run = image_runtime.run(
                    self.pipeline,
                    request,
                    cancel=cancel,
                    on_step=on_step,
                    model_id=self.model_id,
                    load_id=self.load_id,
                )
                return run
        except (RuntimeError, MemoryError) as error:
            # A picture is the largest single allocation the app makes, so
            # running out of memory is the failure worth naming; everything
            # else is the pipeline's own error, passed through.
            reraise_out_of_memory(error, IMAGE_KIND)
            raise
        finally:
            # The run's own end, whoever reserved it: a streaming caller
            # cannot release the slot itself, because the pipeline is on
            # this thread and would still be drawing after that caller's
            # generator was closed.
            self.finish_image_run()
            self._log_image_run(run, request, time.monotonic() - started)
            # The largest thing a run allocates is the pipeline's own
            # activations, and the previews and maps it leaves behind are the
            # caller's now; give the allocator's blocks back rather than hold
            # them until the next run.
            self._release_device_cache()

    def _log_image_run(self, run, request, seconds: float) -> None:
        """Record what a picture cost, whatever its outcome. One line per run."""

        try:
            logger.info(
                "Drew %s steps of %s at %sx%s for %s in %.1fs: guidance %s, "
                "%s held on the device%s",
                0 if run is None else run.steps_done,
                request.steps,
                request.width,
                request.height,
                self.model_id or "no model",
                seconds,
                request.guidance_scale,
                memory_note(device_memory.reserved_bytes()),
                "" if run is not None and not run.stopped else ", stopped",
            )
        except Exception:  # noqa: BLE001 - a log line must not break a run
            logger.debug("Could not record the image run", exc_info=True)
