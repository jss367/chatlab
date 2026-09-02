"""Native macOS launcher for the ChatLab Gradio application."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.request import urlopen

import updater
from app import build_app
from version import __version__


APP_NAME = "ChatLab"
WINDOW_TITLE = "ChatLab"
LOOPBACK_ADDRESS = "127.0.0.1"


def app_support_directory() -> Path:
    """Return the per-user directory used for logs and WebKit storage."""

    return Path.home() / "Library" / "Application Support" / APP_NAME


def configure_logging() -> Path:
    """Write desktop-launch errors somewhere accessible without a terminal."""

    log_directory = Path.home() / "Library" / "Logs" / APP_NAME
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "ChatLab.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return log_path


def find_available_port() -> int:
    """Ask macOS for an unused loopback TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_ADDRESS, 0))
        return int(listener.getsockname()[1])


def start_local_server():
    """Start Gradio in the background and return its app and local URL."""

    port = find_available_port()
    demo = build_app().queue(default_concurrency_limit=1)
    try:
        _, local_url, _ = demo.launch(
            inbrowser=False,
            prevent_thread_lock=True,
            quiet=True,
            server_name=LOOPBACK_ADDRESS,
            server_port=port,
            show_api=False,
        )
    except Exception:
        demo.close(verbose=False)
        raise
    return demo, local_url


def smoke_test() -> int:
    """Verify a packaged executable can start and serve the application."""

    demo, local_url = start_local_server()
    try:
        with urlopen(local_url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"ChatLab returned HTTP {response.status}.")
        print(f"ChatLab desktop smoke test passed at {local_url}")
        return 0
    finally:
        demo.close(verbose=False)


class UpdateFlow:
    """Check GitHub Releases and, with the user's consent, replace the app."""

    def __init__(self, window, bundle: Path | None) -> None:
        self.window = window
        self.bundle = bundle
        self._lock = threading.Lock()
        # ``swapping`` and ``cancel`` only change under ``_phase_lock`` so a quit
        # and the start of the swap cannot both win.
        self._phase_lock = threading.Lock()
        self.swapping = threading.Event()
        self.cancel = threading.Event()
        window.events.closing += self._on_closing

    def _on_closing(self) -> bool:
        """Quit cancels a download in flight but waits out the bundle swap.

        Returning False from a closing handler makes pywebview keep the window
        open, so the few seconds between parking the old bundle and moving the
        new one in cannot be interrupted.
        """

        with self._phase_lock:
            if self.swapping.is_set():
                return False
            self.cancel.set()
            return True

    def _begin_swap(self) -> bool:
        """Enter the protected swap phase unless a quit already cancelled us."""

        with self._phase_lock:
            if self.cancel.is_set():
                return False
            self.swapping.set()
        self._window_call("set_title", f"{WINDOW_TITLE} — installing update…")
        return True

    def check_in_background(self, *, interactive: bool) -> threading.Thread:
        """Run ``check`` on a daemon thread so a quit can abandon it.

        Everything before the swap (release lookup, checksum fetch, download,
        extraction) is safe to drop mid-flight. The swap itself is protected by
        ``_on_closing`` refusing to close and ``wait_for_swap`` joining on exit.
        """

        worker = threading.Thread(
            target=self.check, kwargs={"interactive": interactive}, daemon=True
        )
        self._worker = worker
        worker.start()
        return worker

    def wait_for_swap(self, timeout: float = 300) -> None:
        """Called on the way out: forbid new swaps, then wait for one in progress."""

        with self._phase_lock:
            self.cancel.set()
            swapping = self.swapping.is_set()
        worker = getattr(self, "_worker", None)
        if swapping and worker is not None and worker.is_alive():
            logging.info("Waiting for the update swap to finish before exiting")
            worker.join(timeout)

    def _window_call(self, method: str, *args):
        """Call a window method, tolerating a window the user already closed."""

        try:
            return getattr(self.window, method)(*args)
        except Exception as error:  # noqa: BLE001 - window is gone; log and carry on
            logging.info("Window call %s skipped: %s", method, error)
            return None

    def check(self, *, interactive: bool) -> None:
        """Look for a newer release; ``interactive`` reports "up to date" too."""

        if self.bundle is None:
            logging.info("Not running from an app bundle; skipping update check")
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            release = updater.check_for_update()
        except updater.UpdateError as error:
            logging.warning("%s", error)
            if interactive and self._window_call(
                "create_confirmation_dialog", "ChatLab", f"{error}\n\nOpen the releases page?"
            ):
                webbrowser.open(updater.RELEASES_PAGE_URL)
            return
        else:
            if release is None:
                logging.info("ChatLab %s is up to date", __version__)
                if interactive:
                    self._window_call(
                        "create_confirmation_dialog",
                        "ChatLab",
                        f"ChatLab {__version__} is the latest version.",
                    )
                return
            self._offer(release)
        finally:
            self._lock.release()

    def _offer(self, release: updater.ReleaseInfo) -> None:
        accepted = self._window_call(
            "create_confirmation_dialog",
            "Update available",
            f"ChatLab {release.version} is available (you have {__version__}).\n\n"
            "Download and install it now? ChatLab will restart when it finishes.",
        )
        if not accepted:
            return
        try:
            updater.install_update(
                release,
                self.bundle,
                progress=self._report_progress,
                begin_swap=self._begin_swap,
                cancelled=self.cancel.is_set,
            )
        except updater.UpdateCancelled as error:
            logging.info("%s", error)
            return
        except updater.UpdateError as error:
            logging.error("Update failed: %s", error)
            self._window_call("set_title", WINDOW_TITLE)
            self._window_call("create_confirmation_dialog", "Update failed", str(error))
            return
        finally:
            self.swapping.clear()
        logging.info("Relaunching ChatLab %s", release.version)
        updater.relaunch(self.bundle)
        self._window_call("destroy")

    def _report_progress(self, received: int, total: int | None) -> None:
        if total:
            self._window_call("set_title", f"{WINDOW_TITLE} — downloading update {received * 100 // total}%")
        else:
            self._window_call("set_title", f"{WINDOW_TITLE} — downloading update ({received >> 20} MB)")


def run_desktop() -> int:
    """Open ChatLab in a native WebKit window until the user quits."""

    import webview
    from webview.menu import Menu, MenuAction

    support_directory = app_support_directory()
    support_directory.mkdir(parents=True, exist_ok=True)
    bundle = updater.running_app_bundle()
    demo, local_url = start_local_server()
    logging.info("Started ChatLab %s at %s", __version__, local_url)
    flow: UpdateFlow | None = None

    try:
        window = webview.create_window(
            WINDOW_TITLE,
            local_url,
            width=1440,
            height=960,
            min_size=(960, 680),
            background_color="#f8fafc",
            text_select=True,
            zoomable=True,
        )
        flow = UpdateFlow(window, bundle)

        def after_startup() -> None:
            # Runs once the native window is up, so a release that fails to
            # start still has the previous bundle parked beside it.
            if bundle is not None:
                updater.remove_previous_bundles(bundle)
            flow.check_in_background(interactive=False)

        webview.start(
            func=after_startup,
            gui="cocoa",
            private_mode=False,
            storage_path=str(support_directory / "WebKit"),
            menu=[
                Menu(
                    "Help",
                    [
                        MenuAction(
                            "Check for Updates…",
                            lambda: flow.check_in_background(interactive=True),
                        ),
                        MenuAction("ChatLab Releases", lambda: webbrowser.open(updater.RELEASES_PAGE_URL)),
                    ],
                )
            ],
        )
    finally:
        if flow is not None:
            flow.wait_for_swap()
        logging.info("Stopping ChatLab")
        demo.close(verbose=False)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the ChatLab macOS app.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="start the local server, verify it responds, and exit",
    )
    parser.add_argument("--version", action="version", version=f"ChatLab {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    configure_logging()
    args = parse_args(argv)
    try:
        return smoke_test() if args.smoke_test else run_desktop()
    except Exception:
        logging.exception("ChatLab failed to start")
        raise


if __name__ == "__main__":
    sys.exit(main())
