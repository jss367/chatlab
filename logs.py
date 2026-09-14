"""Where ChatLab's log goes, how much of it is kept, and how loud it is.

One module, because there are two ways in - the desktop launcher and
``python app.py`` - and each used to configure logging for itself. A rule
added to one silently did not hold for the other, and the launcher's file had
no size limit at all, which is a poor arrangement for the one record that
outlives a crash.

What is written here is read after the fact, usually after a memory kill that
took the process down without warning. That shapes every choice below: the
file is capped but deep, the level can be raised without a rebuild, and the
libraries that log a line per HTTP request are held down so ChatLab's own
lines are findable.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

from version import __version__


APP_NAME = "ChatLab"

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Named the way settings.SETTINGS_PATH_ENV is, so a reader looking for what
# the environment controls finds all of it by grepping for _ENV.
LOG_PATH_ENV = "CHATLAB_LOG_PATH"
LOG_LEVEL_ENV = "CHATLAB_LOG_LEVEL"

# Two megabytes a file, five files behind it. A capped file is not optional:
# this lives in a directory nobody opens, and an uncapped one grows until a
# reader notices, which is never. Ten megabytes of plain text is on the order
# of a hundred thousand lines, so days of heavy use survive a rotation.
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 5

# Libraries that log a line per HTTP request, per file lock, per font lookup.
# At INFO their chatter is most of the file and none of it is about ChatLab.
# At DEBUG the reader has asked for everything and gets it.
NOISY_LOGGERS = (
    "asyncio",
    "filelock",
    "fsspec",
    "gradio",
    "h11",
    "hpack",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "matplotlib",
    "multipart",
    "PIL",
    "urllib3",
    "uvicorn.access",
    "watchfiles",
)

# Read for the startup record without importing any of them: package metadata
# is a file on disk, so naming torch here costs nothing, while importing it
# would cost several seconds at the moment the app is trying to open a window.
# These are the ones that allocate, quantize, or decide where a tensor lands,
# which is to say the ones a memory report has to name.
RECORDED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "kernels",
    "mlx",
    "mlx-lm",
    "diffusers",
    "huggingface-hub",
    "gradio",
)

# The attribute marking a handler as this module's, so a second call to
# :func:`configure` replaces its own handlers rather than adding a second copy
# of each and writing every line twice.
_MARK = "chatlab_handler"


def log_directory() -> Path:
    """The folder holding the log file.

    ``~/Library/Logs/ChatLab`` on macOS: that is where Console.app looks, and
    where a reader told to "send the log" will go. Elsewhere the XDG state
    directory, so a Linux checkout does not drop files loose in ``$HOME``.
    """

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    state = os.environ.get("XDG_STATE_HOME")
    root = Path(state).expanduser() if state and state.strip() else Path.home() / ".local" / "state"
    return root / "chatlab"


def log_path() -> Path:
    """The file being written to. ``CHATLAB_LOG_PATH`` names another."""

    override = os.environ.get(LOG_PATH_ENV)
    if override and override.strip():
        return Path(override.strip()).expanduser()
    return log_directory() / f"{APP_NAME}.log"


# The open lock file whose flock says this process owns the plain log name.
# Held for the life of the process and never closed on purpose: the kernel
# drops it when the process ends, however it ends, which is the case that
# matters here.
_claim = None


def claim(target: Path) -> Path:
    """``target`` if this process can have it to itself, otherwise a name only it uses.

    A rotating handler renames files as it rolls over, and two processes
    rolling over the same file scramble and drop records - exactly the
    records this log exists to keep. Two ChatLab processes at once is not
    hypothetical: ``desktop_launcher.start_local_server`` falls back to a
    free port rather than refusing a second instance, and a checkout running
    beside the installed app is an ordinary afternoon.

    So the first process to ask takes the plain name and the next writes
    beside it under its own process id. The lock is on a file of its own
    rather than on the log, because a rollover closes and reopens the log and
    a lock closed with it would not be a lock.
    """

    global _claim
    if _claim is not None:
        # Already ours. Asking again with a second descriptor would be
        # refused by the kernel, which counts flocks per open file rather
        # than per process, and a second configure() would rename the log
        # out from under the first.
        return target
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX only, which is where ChatLab runs
        return target
    try:
        handle = open(target.parent / f"{target.name}.lock", "w")
    except OSError:
        # No lock file, so no claim to make. One process writing an
        # unprotected log is the situation this had before.
        return target
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return target.with_name(f"{target.stem}-{os.getpid()}{target.suffix}")
    _claim = handle
    return target


def level() -> int:
    """The level to record at, from ``CHATLAB_LOG_LEVEL``; INFO by default.

    Takes a name (``debug``, ``WARNING``) or a number. Anything unreadable
    falls back to INFO rather than to silence: a typo in an environment
    variable should not be the reason a crash left no trace.
    """

    raw = (os.environ.get(LOG_LEVEL_ENV) or "").strip()
    if not raw:
        return logging.INFO
    named = logging.getLevelName(raw.upper())
    if isinstance(named, int):
        return named
    try:
        return int(raw)
    except ValueError:
        return logging.INFO


def configure(to_file: bool = True) -> Path | None:
    """Install ChatLab's handlers on the root logger; return the file being written.

    ``None`` means nothing is going to disk, either because ``to_file`` said
    so or because the file could not be opened - a read-only home costs the
    file, not the launch. Calling this twice is safe: the handlers it owns are
    replaced, not stacked.
    """

    root = logging.getLogger()
    for handler in [item for item in root.handlers if getattr(item, _MARK, False)]:
        root.removeHandler(handler)
        handler.close()
    chosen = level()
    root.setLevel(chosen)
    formatter = logging.Formatter(LOG_FORMAT)
    # The console first, so that a file that cannot be opened has somewhere to
    # say so. Under a windowed bundle there is no stderr to attach to and this
    # is skipped; the file is the whole log there.
    if getattr(sys, "stderr", None) is not None:
        _install(root, logging.StreamHandler(sys.stderr), formatter, chosen)
    if not to_file:
        quiet_noisy_loggers(chosen)
        return None
    target = log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = claim(target)
        handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
    except OSError as error:
        logging.getLogger(__name__).warning("Could not open the log file %s: %s", target, error)
        return None
    _install(root, handler, formatter, chosen)
    quiet_noisy_loggers(chosen)
    return target


def _install(root: logging.Logger, handler: logging.Handler, formatter, chosen: int) -> None:
    handler.setFormatter(formatter)
    handler.setLevel(chosen)
    setattr(handler, _MARK, True)
    root.addHandler(handler)


def quiet_noisy_loggers(chosen: int) -> None:
    """Hold the libraries' own chatter at warnings, unless debug was asked for."""

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.NOTSET if chosen <= logging.DEBUG else logging.WARNING
        )


def package_versions() -> str:
    """``torch 2.14.0, transformers 5.17.1, kernels absent`` for the record below."""

    from importlib.metadata import PackageNotFoundError, version

    read = []
    for name in RECORDED_PACKAGES:
        try:
            read.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            read.append(f"{name} absent")
        except Exception:  # noqa: BLE001 - a version that cannot be read is not a crash
            read.append(f"{name} unreadable")
    return ", ".join(read)


def machine_note() -> str:
    """The machine's own memory, which is the pool every backend here draws on."""

    from model_runtime import memory_note, system_memory

    total, available = system_memory()
    return f"{memory_note(total)} of memory, {memory_note(available)} available"


def settings_note() -> str:
    """The saved settings a memory report has to be read against.

    Not the whole file: the ones that decide how large a load is allowed to
    be and how long a reply may run, which are what a reader is trying to
    account for when they open the log after a kill.
    """

    import settings

    try:
        saved = settings.load()
    except Exception as error:  # noqa: BLE001 - the record must not stop the launch
        return f"unreadable ({error})"
    fraction = saved.mps_memory_fraction
    return (
        f"{settings.settings_path()}: weight precision {saved.weight_precision}, "
        f"max_new_tokens {saved.max_new_tokens}, mps_memory_fraction "
        f"{'unset' if fraction is None else f'{fraction:g}'}, model {saved.model_id}"
    )


def log_environment(target: Path | None = None) -> None:
    """Record what this build is running on, once, at startup.

    The first thing anyone wants from a log read after a crash, and the thing
    it never used to hold: which build, which Python, which machine, which
    versions of the libraries that do the allocating, and what the settings
    were. Without it a memory kill reported a week later cannot even be
    matched to a release. The device itself is not here - torch has not
    finished importing this early - and is recorded by
    :func:`model_runtime.warm_device` as soon as it can be read.
    """

    logger = logging.getLogger(__name__)
    build = "packaged app" if getattr(sys, "frozen", False) else f"checkout at {Path.cwd()}"
    logger.info("ChatLab %s starting: %s", __version__, build)
    logger.info("Platform: %s, Python %s at %s", platform.platform(), platform.python_version(), sys.executable)
    logger.info("Machine: %s", machine_note())
    logger.info("Packages: %s", package_versions())
    logger.info("Settings: %s", settings_note())
    if target is not None:
        logger.info(
            "Logging at %s to %s, %s per file and %s kept",
            logging.getLevelName(level()),
            target,
            f"{MAX_BYTES // 1024**2} MB",
            BACKUP_COUNT,
        )
