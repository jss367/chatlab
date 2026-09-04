"""The reader's settings, kept in one JSON file between sessions.

Everything on the Settings page, the model to load, and the two memory limits
live in one file the reader owns. It is written whenever a control changes, so
there is nothing to press, and it is plain JSON at a predictable path, so it
can be edited by hand or symlinked out of a dotfiles repository to give
several machines one set of choices.

The Hugging Face token is deliberately absent. It is a secret, and a file
meant to be committed and synced is the wrong place for one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from token_metrics import COLOR_SCALES, DEFAULT_COLOR_SCALE

logger = logging.getLogger(__name__)

# Where the file goes when nothing says otherwise. The XDG variable is honored
# on every platform: a reader who has set it has said where configuration
# belongs, and macOS's own Application Support directory is a poor place for a
# file meant to be symlinked out of a repository.
SETTINGS_PATH_ENV = "CHATLAB_SETTINGS_PATH"
XDG_CONFIG_ENV = "XDG_CONFIG_HOME"
SETTINGS_DIRECTORY = "chatlab"
SETTINGS_FILENAME = "settings.json"

DEFAULT_MODEL_ID = "allenai/Olmo-3-7B-Think"
# ``OLMO_MODEL_ID`` predates this file and still wins, so a launcher that
# pins a model keeps working.
MODEL_ID_ENV = "OLMO_MODEL_ID"

# Refuse a generation prefix large enough to make replay itself an
# unexpectedly expensive operation, and to keep one conversation's key-value
# cache inside the machine. The response-length control tops out at whatever
# this is, so a typed branch cannot paste an unbounded second response around
# that control even when the model advertises a much larger context window.
DEFAULT_PREFILL_TOKEN_LIMIT = 8192
# Below the floor a single ordinary question would not fit; above the ceiling
# the key-value cache outgrows any machine this runs on.
PREFILL_TOKEN_LIMIT_RANGE = (256, 131072)

# The share of Metal's recommended working set PyTorch may allocate before it
# raises rather than letting macOS page the machine into a freeze. Its own
# default is 1.7, well past physical memory. A null in the file means the
# model runtime's default of 1.0; see model_runtime.mps_memory_fraction.
MPS_MEMORY_FRACTION_RANGE = (0.1, 2.0)

TEMPERATURE_RANGE = (0.0, 2.0)
TOP_P_RANGE = (0.05, 1.0)
TOP_K_RANGE = (0, 200)
# NumPy's default generator rejects a negative seed but takes any
# non-negative one, however large; see app.resolve_seed. So the seed has a
# floor rather than a range: the point of locking a seed is to reproduce a
# response, and pulling a saved 3_000_000_000 down to a ceiling the app only
# uses when it picks a seed itself (app.SEED_LIMIT) would reproduce a
# different one.
SEED_FLOOR = 0


def _clamped_float(value: Any, bounds: tuple[float, float], default: float) -> float:
    low, high = bounds
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    if number != number:  # NaN compares false against every bound.
        return default
    return min(max(number, low), high)


def _clamped_int(value: Any, bounds: tuple[int, int], default: int) -> int:
    low, high = bounds
    try:
        number = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return min(max(number, low), high)


def _floored_int(value: Any, low: int, default: int) -> int:
    try:
        number = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return max(number, low)


def _text(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _flag(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


@dataclass(frozen=True)
class Settings:
    """One reader's choices, with the defaults the interface starts from.

    Every default matches the value its control was built with before this
    file existed, so a machine with no settings file behaves exactly as the
    app always did.
    """

    model_id: str = DEFAULT_MODEL_ID
    system_prompt: str = ""
    assistant_prefill: str = ""
    keep_reasoning: bool = False
    enter_sends: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 1024
    seed: int = 42
    randomize_seed: bool = True
    analyze_prompt: bool = True
    color_scale: str = DEFAULT_COLOR_SCALE
    prefill_token_limit: int = DEFAULT_PREFILL_TOKEN_LIMIT
    mps_memory_fraction: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        """The object as the JSON file spells it."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


DEFAULTS = Settings()
_FIELD_NAMES = frozenset(field.name for field in fields(Settings))


def sanitize(values: Mapping[str, Any]) -> Settings:
    """The settings ``values`` describes, with anything unusable replaced.

    A hand-edited file is expected to contain mistakes, and a settings file is
    never a good enough reason to refuse to start: a value out of range is
    pulled to the nearest end of it, and one of the wrong shape entirely falls
    back to its default.
    """

    prefill = _clamped_int(
        values.get("prefill_token_limit", DEFAULTS.prefill_token_limit),
        PREFILL_TOKEN_LIMIT_RANGE,
        DEFAULTS.prefill_token_limit,
    )
    fraction = values.get("mps_memory_fraction", DEFAULTS.mps_memory_fraction)
    scale = _text(values.get("color_scale", DEFAULTS.color_scale), DEFAULTS.color_scale)
    return Settings(
        model_id=_text(values.get("model_id", DEFAULTS.model_id), DEFAULTS.model_id)
        or DEFAULTS.model_id,
        system_prompt=_text(
            values.get("system_prompt", DEFAULTS.system_prompt), DEFAULTS.system_prompt
        ),
        assistant_prefill=_text(
            values.get("assistant_prefill", DEFAULTS.assistant_prefill),
            DEFAULTS.assistant_prefill,
        ),
        keep_reasoning=_flag(
            values.get("keep_reasoning", DEFAULTS.keep_reasoning),
            DEFAULTS.keep_reasoning,
        ),
        enter_sends=_flag(
            values.get("enter_sends", DEFAULTS.enter_sends), DEFAULTS.enter_sends
        ),
        temperature=_clamped_float(
            values.get("temperature", DEFAULTS.temperature),
            TEMPERATURE_RANGE,
            DEFAULTS.temperature,
        ),
        top_p=_clamped_float(
            values.get("top_p", DEFAULTS.top_p), TOP_P_RANGE, DEFAULTS.top_p
        ),
        top_k=_clamped_int(
            values.get("top_k", DEFAULTS.top_k), TOP_K_RANGE, DEFAULTS.top_k
        ),
        # The response-length control shares the prefix cap, so a saved length
        # follows a cap the reader has since lowered.
        max_new_tokens=_clamped_int(
            values.get("max_new_tokens", DEFAULTS.max_new_tokens),
            (1, prefill),
            min(DEFAULTS.max_new_tokens, prefill),
        ),
        seed=_floored_int(
            values.get("seed", DEFAULTS.seed), SEED_FLOOR, DEFAULTS.seed
        ),
        randomize_seed=_flag(
            values.get("randomize_seed", DEFAULTS.randomize_seed),
            DEFAULTS.randomize_seed,
        ),
        analyze_prompt=_flag(
            values.get("analyze_prompt", DEFAULTS.analyze_prompt),
            DEFAULTS.analyze_prompt,
        ),
        color_scale=scale if scale in COLOR_SCALES else DEFAULTS.color_scale,
        prefill_token_limit=prefill,
        mps_memory_fraction=(
            None
            if fraction is None
            else _clamped_float(
                fraction, MPS_MEMORY_FRACTION_RANGE, DEFAULTS.mps_memory_fraction or 1.0
            )
        ),
    )


def settings_path() -> Path:
    """Where the settings file is read from and written to."""

    chosen = os.environ.get(SETTINGS_PATH_ENV)
    if chosen:
        return Path(chosen).expanduser()
    configured = os.environ.get(XDG_CONFIG_ENV)
    root = Path(configured).expanduser() if configured else Path.home() / ".config"
    return root / SETTINGS_DIRECTORY / SETTINGS_FILENAME


def read(path: Path | None = None) -> tuple[Settings, dict[str, Any]]:
    """The settings on disk, and the keys in the file this version has no use for.

    Unknown keys are handed back so :func:`write` can put them where it found
    them. Two machines sharing one file need not run the same version of the
    app, and the newer one's settings must survive a save by the older.
    """

    target = path or settings_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULTS, {}
    except OSError as error:
        logger.warning("Could not read settings from %s: %s", target, error)
        return DEFAULTS, {}
    try:
        loaded = json.loads(raw)
    except ValueError as error:
        logger.warning("Ignoring unreadable settings in %s: %s", target, error)
        return DEFAULTS, {}
    if not isinstance(loaded, dict):
        logger.warning("Ignoring settings in %s: expected a JSON object.", target)
        return DEFAULTS, {}
    unknown = {key: value for key, value in loaded.items() if key not in _FIELD_NAMES}
    return sanitize(loaded), unknown


def write(
    chosen: Settings,
    unknown: Mapping[str, Any] | None = None,
    path: Path | None = None,
) -> Path | None:
    """Save ``settings``, or return ``None`` when the file cannot be written.

    The write is atomic: settings are rewritten on every change, and a reader
    whose disk fills or whose home directory is read-only should lose the new
    value rather than the file. Failing to save is reported to the log and
    nowhere else, because it must not interrupt a conversation.
    """

    target = path or settings_path()
    payload = dict(unknown or {}) | chosen.to_mapping()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        # A settings file symlinked out of a dotfiles repository is one of the
        # arrangements this file is for, and os.replace onto the link would
        # put a plain file where the link was and leave the repository's copy
        # behind. Following the link first writes where the reader actually
        # keeps the file, and leaves the link itself alone.
        destination = target.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the destination: os.replace is only atomic within
        # one filesystem, and the temporary directory is often another.
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        )
        try:
            with handle as stream:
                stream.write(text)
            os.replace(handle.name, destination)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(handle.name)
            raise
    except OSError as error:
        logger.warning("Could not save settings to %s: %s", target, error)
        return None
    return target


_lock = threading.Lock()
_current: Settings | None = None
_unknown: dict[str, Any] = {}


def current() -> Settings:
    """The settings this process is running under, read from disk once."""

    with _lock:
        if _current is not None:
            return _current
    return load()


def load() -> Settings:
    """Re-read the settings file and make it what the process runs under."""

    global _current, _unknown
    loaded, unknown = read()
    with _lock:
        _current = loaded
        _unknown = dict(unknown)
    return loaded


def ensure_file(path: Path | None = None) -> Path | None:
    """Write the settings file if it is not there yet.

    A file that exists from the first launch is one the reader can open, edit,
    or symlink out of a repository without having to change a control first to
    bring it into being.
    """

    target = path or settings_path()
    if target.exists():
        return target
    return write(current(), dict(_unknown), target)


def update(**values: Any) -> Settings:
    """Change some settings, save them, and return the whole set.

    Every value goes through :func:`sanitize`, so a caller may pass whatever
    a control gave it. A change that changes nothing is not written: controls
    report their value on every frame, and most frames repeat it.
    """

    global _current
    with _lock:
        base = _current if _current is not None else DEFAULTS
        unknown = dict(_unknown)
    merged = sanitize(base.to_mapping() | values)
    if merged == base:
        return merged
    with _lock:
        _current = merged
    write(merged, unknown)
    return merged


@contextlib.contextmanager
def override(**values: Any):
    """Run with different settings, then put the old ones back.

    For tests and for anything that must not touch the reader's file.
    """

    global _current, _unknown
    with _lock:
        previous, previous_unknown = _current, dict(_unknown)
        base = previous if previous is not None else DEFAULTS
        _current = sanitize(base.to_mapping() | values)
    try:
        yield _current
    finally:
        with _lock:
            _current, _unknown = previous, previous_unknown


def model_id_at_startup(chosen: Settings | None = None) -> str:
    """The model the interface offers when it opens.

    ``OLMO_MODEL_ID`` still names a model for one run, which is what a
    launcher or a shell alias uses it for, so it wins over the saved choice.
    """

    saved = chosen if chosen is not None else current()
    return os.environ.get(MODEL_ID_ENV) or saved.model_id


def model_id_to_save(shown: str, chosen: Settings | None = None) -> str:
    """The model to write down while the box on the Models page shows ``shown``.

    ``OLMO_MODEL_ID`` pins a model for one run, and the box shows what it
    pins. Every control publishes the box's contents whenever anything
    changes, so without this a nudge of the temperature would write the
    one-run model down as the reader's own choice and leave it selected long
    after the variable was gone. A reader who types a different model has
    chosen it, and that is saved like any other setting.
    """

    override = os.environ.get(MODEL_ID_ENV)
    if not override or shown != override:
        return shown
    saved = chosen if chosen is not None else current()
    return saved.model_id


def seed_to_save(shown: Any, randomizing: bool, chosen: Settings | None = None) -> Any:
    """The seed to write down while the box on the Settings page shows ``shown``.

    The seed box is the one control the app writes to itself: a finished
    response leaves the seed that produced it there. While randomization is
    on, that number is the app's rather than the reader's, and every control
    publishes it whenever anything changes, so without this a nudge of the
    temperature would save a seed nobody chose over the one the reader did.
    Committing the seed box itself is an explicit choice and is saved, and so
    is turning randomization off, which is how a reader keeps the seed a
    response has just used.
    """

    if not randomizing:
        return shown
    saved = chosen if chosen is not None else current()
    return saved.seed
