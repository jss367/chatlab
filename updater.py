"""Self-update support for the packaged ChatLab macOS application.

The desktop app checks the GitHub Releases feed for a newer tag, downloads the
``.zip`` asset for this platform, swaps the bundle on disk, and relaunches.
The pieces that touch the network or the filesystem are kept behind small
functions so they can be tested and so the launcher only has to orchestrate.
"""

from __future__ import annotations

import json
import logging
import platform
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

from version import __version__

logger = logging.getLogger(__name__)

GITHUB_REPO = "jss367/chatlab"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases"
LATEST_RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
USER_AGENT = f"ChatLab/{__version__} (+{RELEASES_PAGE_URL})"
PREVIOUS_BUNDLE_MARKER = ".previous-"
REQUEST_TIMEOUT_SECONDS = 15
DOWNLOAD_CHUNK_BYTES = 1 << 20

_VERSION_PATTERN = re.compile(r"^v?(\d+(?:\.\d+)*)$")

ProgressCallback = Callable[[int, int | None], None]


class UpdateError(RuntimeError):
    """Raised when an update cannot be checked, downloaded, or installed."""


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    asset_name: str
    asset_url: str
    asset_size: int | None
    release_url: str


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
    for asset in payload.get("assets", []):
        if asset.get("name") == asset_name and asset.get("browser_download_url"):
            return ReleaseInfo(
                version=tag.lstrip("v"),
                asset_name=asset_name,
                asset_url=asset["browser_download_url"],
                asset_size=asset.get("size"),
                release_url=payload.get("html_url") or RELEASES_PAGE_URL,
            )
    return None


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


def download_asset(
    release: ReleaseInfo,
    destination_dir: Path,
    progress: ProgressCallback | None = None,
) -> Path:
    """Stream the release zip to ``destination_dir`` and return its path."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / release.asset_name
    request = Request(release.asset_url, headers={"User-Agent": USER_AGENT})
    received = 0
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response, target.open("wb") as out:
            length_header = response.headers.get("Content-Length")
            total = int(length_header) if length_header else release.asset_size
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(received, total)
    except Exception as error:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise UpdateError(f"Download failed: {error}") from error
    if release.asset_size is not None and received != release.asset_size:
        target.unlink(missing_ok=True)
        raise UpdateError(
            f"Download was {received} bytes but the release lists {release.asset_size}."
        )
    return target


def extract_bundle(archive: Path, destination_dir: Path) -> Path:
    """Unzip with ``ditto`` (which keeps symlinks and permissions) and return the ``.app``."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ditto", "-x", "-k", str(archive), str(destination_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UpdateError(f"Could not unpack the update: {result.stderr.strip()}")
    bundles = sorted(destination_dir.glob("*.app"))
    if len(bundles) != 1:
        raise UpdateError(f"Expected one .app in the update, found {len(bundles)}.")
    return bundles[0]


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


def remove_previous_bundles(bundle: Path) -> None:
    """Delete bundles a prior update parked beside the running app."""

    for stale in bundle.parent.glob(f"{bundle.name}{PREVIOUS_BUNDLE_MARKER}*"):
        shutil.rmtree(stale, ignore_errors=True)


def relaunch(bundle: Path) -> None:
    subprocess.Popen(["open", "-n", str(bundle)])


def install_update(
    release: ReleaseInfo,
    bundle: Path,
    work_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    before_swap: Callable[[], None] | None = None,
) -> None:
    """Download, unpack, and swap in ``release``; the caller quits and relaunches.

    ``before_swap`` runs once the new bundle is unpacked and about to replace the
    old one, so the caller can hold off shutdown for the few seconds it takes.
    """

    if work_dir is None:
        try:
            work_dir = Path(tempfile.mkdtemp(prefix="chatlab-update-", dir=bundle.parent))
        except OSError:
            work_dir = Path(tempfile.mkdtemp(prefix="chatlab-update-"))
    try:
        archive = download_asset(release, work_dir, progress)
        replacement = extract_bundle(archive, work_dir / "unpacked")
        if before_swap is not None:
            before_swap()
        parked = swap_bundle(bundle, replacement)
        logger.info("Installed ChatLab %s over %s (previous bundle at %s)", release.version, bundle, parked)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
