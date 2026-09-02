"""Self-update support for the packaged ChatLab macOS application.

The desktop app checks the GitHub Releases feed for a newer tag, downloads the
``.zip`` asset for this platform, swaps the bundle on disk, and relaunches.
The pieces that touch the network or the filesystem are kept behind small
functions so they can be tested and so the launcher only has to orchestrate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from version import BUNDLE_IDENTIFIER, __version__

logger = logging.getLogger(__name__)

GITHUB_REPO = "jss367/chatlab"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases"
LATEST_RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
USER_AGENT = f"ChatLab/{__version__} (+{RELEASES_PAGE_URL})"
PREVIOUS_BUNDLE_MARKER = ".previous-"
WORK_DIR_PREFIX = "chatlab-update-"
WORK_DIR_OWNER_FILE = "owner.pid"
REQUEST_TIMEOUT_SECONDS = 15
DOWNLOAD_CHUNK_BYTES = 1 << 20

_VERSION_PATTERN = re.compile(r"^v?(\d+(?:\.\d+)*)$")

ProgressCallback = Callable[[int, int | None], None]
CancelCheck = Callable[[], bool]


class UpdateError(RuntimeError):
    """Raised when an update cannot be checked, downloaded, or installed."""


class UpdateCancelled(UpdateError):
    """Raised when the caller asked to stop before the bundle swap began."""


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    asset_name: str
    asset_url: str
    asset_size: int | None
    release_url: str
    checksum_url: str | None = None


def parse_version(text: str) -> tuple[int, ...] | None:
    """Turn ``v1.2.3`` or ``1.2.3`` into a comparable tuple, else ``None``."""

    match = _VERSION_PATTERN.match(text.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(candidate: str, current: str = __version__) -> bool:
    """Return whether ``candidate`` is a strictly newer release than ``current``."""

    candidate_parts = parse_version(candidate)
    current_parts = parse_version(current)
    if candidate_parts is None or current_parts is None:
        return False
    width = max(len(candidate_parts), len(current_parts))
    pad = lambda parts: parts + (0,) * (width - len(parts))  # noqa: E731
    return pad(candidate_parts) > pad(current_parts)


def expected_asset_name(machine: str | None = None) -> str:
    """Name of the release asset built for this machine."""

    machine = machine or platform.machine()
    arch = "arm64" if machine in ("arm64", "aarch64") else machine
    return f"ChatLab-macos-{arch}.zip"


def select_release(payload: dict, asset_name: str) -> ReleaseInfo | None:
    """Pick the asset for this platform out of a GitHub release payload."""

    tag = str(payload.get("tag_name", ""))
    if parse_version(tag) is None:
        return None
    assets = {
        asset.get("name"): asset
        for asset in payload.get("assets", [])
        if asset.get("browser_download_url")
    }
    asset = assets.get(asset_name)
    if asset is None:
        return None
    checksum = assets.get(f"{asset_name}.sha256")
    return ReleaseInfo(
        version=tag.lstrip("v"),
        asset_name=asset_name,
        asset_url=asset["browser_download_url"],
        asset_size=asset.get("size"),
        release_url=payload.get("html_url") or RELEASES_PAGE_URL,
        checksum_url=checksum["browser_download_url"] if checksum else None,
    )


def fetch_latest_release(api_url: str = LATEST_RELEASE_API_URL) -> dict:
    request = Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except Exception as error:  # noqa: BLE001 - surfaced to the caller as one type
        raise UpdateError(f"Could not reach GitHub to check for updates: {error}") from error


def check_for_update(current_version: str = __version__) -> ReleaseInfo | None:
    """Return the newer release for this platform, or ``None`` when up to date."""

    release = select_release(fetch_latest_release(), expected_asset_name())
    if release is None or not is_newer(release.version, current_version):
        return None
    return release


def running_app_bundle(executable: str | None = None) -> Path | None:
    """Locate the ``.app`` that contains the frozen executable, if any."""

    if executable is None:
        if not getattr(sys, "frozen", False):
            return None
        executable = sys.executable
    path = Path(executable).resolve()
    # ChatLab.app/Contents/MacOS/ChatLab -> ChatLab.app
    if len(path.parents) < 3:
        return None
    bundle = path.parents[2]
    if bundle.suffix != ".app" or path.parents[1].name != "Contents":
        return None
    return bundle


def fetch_checksum(release: ReleaseInfo) -> str:
    """Return the published SHA-256 hex digest for the release asset."""

    if release.checksum_url is None:
        raise UpdateError(
            f"Release {release.version} does not publish a checksum for {release.asset_name}."
        )
    request = Request(release.checksum_url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            text = response.read(1024).decode("ascii", "replace")
    except Exception as error:  # noqa: BLE001
        raise UpdateError(f"Could not fetch the release checksum: {error}") from error
    digest = text.split()[0].lower() if text.split() else ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UpdateError("The release checksum file is not a SHA-256 digest.")
    return digest


def download_asset(
    release: ReleaseInfo,
    destination_dir: Path,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
) -> Path:
    """Stream the release zip to ``destination_dir``, verify it, and return its path.

    The file is hashed as it arrives and must match the checksum the release
    publishes beside it. ``cancelled`` is polled between chunks; when it
    returns True the partial file is removed and :class:`UpdateCancelled` is
    raised.
    """

    expected_digest = fetch_checksum(release)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / release.asset_name
    request = Request(release.asset_url, headers={"User-Agent": USER_AGENT})
    received = 0
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response, target.open("wb") as out:
            length_header = response.headers.get("Content-Length")
            total = int(length_header) if length_header else release.asset_size
            while True:
                if cancelled is not None and cancelled():
                    raise UpdateCancelled("Update cancelled during download.")
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(received, total)
    except UpdateCancelled:
        target.unlink(missing_ok=True)
        raise
    except Exception as error:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise UpdateError(f"Download failed: {error}") from error
    if release.asset_size is not None and received != release.asset_size:
        target.unlink(missing_ok=True)
        raise UpdateError(
            f"Download was {received} bytes but the release lists {release.asset_size}."
        )
    if digest.hexdigest() != expected_digest:
        target.unlink(missing_ok=True)
        raise UpdateError("The downloaded update does not match the published checksum.")
    return target


def extract_bundle(
    archive: Path,
    destination_dir: Path,
    cancelled: CancelCheck | None = None,
) -> Path:
    """Unzip with ``ditto`` (which keeps symlinks and permissions) and return the ``.app``.

    ``cancelled`` is polled while ``ditto`` runs; when it returns True the
    child is terminated and :class:`UpdateCancelled` is raised.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["ditto", "-x", "-k", str(archive), str(destination_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while True:
            try:
                _, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    process.terminate()
                    process.wait()
                    raise UpdateCancelled("Update cancelled during extraction.")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    if process.returncode != 0:
        raise UpdateError(f"Could not unpack the update: {(stderr or '').strip()}")
    bundles = sorted(destination_dir.glob("*.app"))
    if len(bundles) != 1:
        raise UpdateError(f"Expected one .app in the update, found {len(bundles)}.")
    return bundles[0]


def verify_bundle(bundle: Path, release: ReleaseInfo) -> None:
    """Refuse a bundle whose Info.plist is not ChatLab at the release's version."""

    plist_path = bundle / "Contents" / "Info.plist"
    try:
        with plist_path.open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as error:
        raise UpdateError(f"The update has no readable Info.plist: {error}") from error
    identifier = info.get("CFBundleIdentifier")
    if identifier != BUNDLE_IDENTIFIER:
        raise UpdateError(f"The update identifies itself as {identifier!r}, not ChatLab.")
    version = str(info.get("CFBundleShortVersionString", ""))
    if parse_version(version) != parse_version(release.version):
        raise UpdateError(
            f"The update is version {version or 'unknown'} but the release is {release.version}."
        )
    if not (bundle / "Contents" / "MacOS" / "ChatLab").is_file():
        raise UpdateError("The update is missing its ChatLab executable.")


def swap_bundle(current: Path, replacement: Path) -> Path:
    """Move ``replacement`` into ``current``'s place; return where the old bundle went."""

    parked = current.with_name(f"{current.name}{PREVIOUS_BUNDLE_MARKER}{int(time.time())}")
    try:
        current.rename(parked)
    except OSError as error:
        raise UpdateError(
            f"Could not replace {current}. Move ChatLab somewhere you can write to, "
            f"or download the update from {RELEASES_PAGE_URL}. ({error})"
        ) from error
    try:
        shutil.move(str(replacement), str(current))
    except OSError as error:
        # A cross-volume move copies instead of renaming, so a failure can leave
        # a partial bundle at ``current``; clear it before putting the old one back.
        shutil.rmtree(current, ignore_errors=True)
        parked.rename(current)
        raise UpdateError(f"Could not install the update: {error}") from error
    return parked


def is_parked_bundle(path: Path, bundle: Path) -> bool:
    """Whether ``path`` has the exact ``<bundle>.previous-<unix timestamp>`` shape."""

    prefix = f"{bundle.name}{PREVIOUS_BUNDLE_MARKER}"
    return path.name.startswith(prefix) and path.name[len(prefix):].isdigit()


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_work_dir(work_dir: Path) -> None:
    """Mark ``work_dir`` as owned by this process so sweeps leave it alone."""

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / WORK_DIR_OWNER_FILE).write_text(str(os.getpid()))


def is_abandoned_work_dir(path: Path) -> bool:
    """Whether ``path`` is an updater staging dir whose owning process is gone.

    Directories without an owner marker are not touched: they are either
    unrelated or were created a moment ago and not yet claimed.
    """

    if not path.is_dir() or not path.name.startswith(WORK_DIR_PREFIX):
        return False
    try:
        pid = int((path / WORK_DIR_OWNER_FILE).read_text().strip())
    except (OSError, ValueError):
        return False
    return pid != os.getpid() and not _process_alive(pid)


def remove_stale_work_dirs(bundle: Path) -> None:
    """Delete staging directories an interrupted update left behind.

    ``install_update`` stages beside the app when it can and in the system
    temporary directory otherwise, so both are swept. Only directories whose
    recorded owner process is no longer running are removed, so a second
    ChatLab instance mid-update keeps its files.
    """

    for parent in {bundle.parent, Path(tempfile.gettempdir())}:
        for candidate in parent.glob(f"{WORK_DIR_PREFIX}*"):
            if is_abandoned_work_dir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)


def remove_previous_bundles(bundle: Path) -> None:
    """Delete bundles a prior update parked beside the running app.

    Only names ending in the timestamp ``swap_bundle`` writes are touched, so a
    hand-made ``ChatLab.app.previous-manual`` next to the app is left alone.
    """

    for stale in bundle.parent.glob(f"{bundle.name}{PREVIOUS_BUNDLE_MARKER}*"):
        if is_parked_bundle(stale, bundle):
            shutil.rmtree(stale, ignore_errors=True)


def relaunch(bundle: Path) -> None:
    subprocess.Popen(["open", "-n", str(bundle)])


def install_update(
    release: ReleaseInfo,
    bundle: Path,
    work_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    begin_swap: Callable[[], bool] | None = None,
    cancelled: CancelCheck | None = None,
) -> None:
    """Download, unpack, and swap in ``release``; the caller quits and relaunches.

    ``cancelled`` is polled during the download and extraction and raises
    :class:`UpdateCancelled`. The downloaded archive must match the release's
    published SHA-256 and the unpacked bundle must identify itself as ChatLab
    at the release's version. ``begin_swap`` is called once the new bundle is
    unpacked and verified; it must atomically decide whether to proceed (returning True and
    holding off shutdown for the few seconds the swap takes) or report that
    the update was cancelled (returning False). Nothing is checked after it.
    """

    if work_dir is None:
        try:
            work_dir = Path(tempfile.mkdtemp(prefix=WORK_DIR_PREFIX, dir=bundle.parent))
        except OSError:
            work_dir = Path(tempfile.mkdtemp(prefix=WORK_DIR_PREFIX))
    try:
        claim_work_dir(work_dir)
        archive = download_asset(release, work_dir, progress, cancelled)
        replacement = extract_bundle(archive, work_dir / "unpacked", cancelled)
        verify_bundle(replacement, release)
        if begin_swap is not None:
            if not begin_swap():
                raise UpdateCancelled("Update cancelled before installation.")
        elif cancelled is not None and cancelled():
            raise UpdateCancelled("Update cancelled before installation.")
        parked = swap_bundle(bundle, replacement)
        logger.info("Installed ChatLab %s over %s (previous bundle at %s)", release.version, bundle, parked)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
