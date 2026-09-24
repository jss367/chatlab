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

from chatlab import api
from chatlab import branding
from chatlab import desktop
from chatlab import logs
from chatlab import device_memory
from chatlab import updater
from chatlab.app import build_app
from chatlab.desktop_smoke import smoke_test_metal, smoke_test_mlx, smoke_test_pipelines
from chatlab.version import __version__


logger = logging.getLogger(__name__)

APP_NAME = "ChatLab"
WINDOW_TITLE = "ChatLab"
LOOPBACK_ADDRESS = "127.0.0.1"
# The window is served on the same port every launch. Anything the browser
# keeps per origin - the width of the pane beside the transcript is kept
# that way - is kept per port with it, so a port chosen afresh each launch
# would be a new origin each launch, and those choices would be gone every
# time the app opened. The number is arbitrary beyond being outside the
# range macOS hands out for connections of its own, so a window that has to
# fall back to any free port cannot be handed this one by accident.
DESKTOP_PORT = 47890

# What Settings is told when the restart it asked for is not its to make,
# and when the one it was granted could not be started after all.
RESTART_DURING_UPDATE = (
    "ChatLab is installing an update. It restarts itself when the update "
    "finishes, and the saved choice applies then."
)
RESTART_ALREADY_UNDER_WAY = (
    "ChatLab is already restarting. The saved choice applies when the new window opens."
)
RESTART_FAILED = "ChatLab could not reopen itself. Quit and open it again to apply this change."


def app_support_directory() -> Path:
    """Return the per-user directory used for logs and WebKit storage."""

    return Path.home() / "Library" / "Application Support" / APP_NAME


def configure_logging() -> Path | None:
    """Put the log somewhere reachable without a terminal, and say where.

    The rules live in :mod:`logs` so that ``python -m chatlab`` gets the same
    ones. ``None`` means nothing could be opened for writing, which costs the
    file and not the launch.
    """

    return logs.configure()


def find_available_port() -> int:
    """Ask macOS for an unused loopback TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_ADDRESS, 0))
        return int(listener.getsockname()[1])


def port_is_free(port: int) -> bool:
    """Whether the window can be served on ``port`` right now."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((LOOPBACK_ADDRESS, port))
    except OSError:
        return False
    return True


def start_local_server():
    """Start Gradio in the background and return its app and local URL."""

    port = DESKTOP_PORT if port_is_free(DESKTOP_PORT) else find_available_port()
    demo = build_app().queue(default_concurrency_limit=1)
    try:
        _, local_url, _ = demo.launch(
            inbrowser=False,
            prevent_thread_lock=True,
            quiet=True,
            server_name=LOOPBACK_ADDRESS,
            server_port=port,
            show_api=False,
            favicon_path=branding.favicon_path(),
        )
        # The local API shares the window's port. Gradio builds the
        # application inside launch(), so this is the first moment there is
        # one to add routes to.
        api.attach(demo.app)
    except Exception:
        demo.close(verbose=False)
        raise
    return demo, local_url


def smoke_test() -> int:
    """Verify packaged Metal, MLX and diffusers support, and the local server."""

    smoke_test_metal()
    smoke_test_mlx()
    smoke_test_pipelines()
    demo, local_url = start_local_server()
    try:
        with urlopen(local_url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"ChatLab returned HTTP {response.status}.")
        # The API is served from the same port, so a build that mounts it
        # wrongly - Gradio's own routes shadowing it, say - fails here rather
        # than in front of somebody's script.
        status_url = f"{local_url.rstrip('/')}{api.API_PREFIX}/chatlab/status"
        with urlopen(status_url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"The ChatLab API returned HTTP {response.status}.")
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
        # ``swapping``, ``relaunching`` and ``cancel`` only change under
        # ``_phase_lock``, so a quit and the start of the swap cannot both win
        # and the swap hands the restart on without letting go of it first.
        self._phase_lock = threading.Lock()
        self.swapping = threading.Event()
        # Held from the moment a restart is claimed until the copy it starts
        # is on its way up: by the swap, which hands it to the update's own
        # relaunch and close, and by a manual restart for its own. Whoever
        # holds it, nobody else starts a second copy behind them.
        self.relaunching = threading.Event()
        # The cancellation the update running now answers to. An attempt takes
        # the event it started with and keeps reading that one, so a worker
        # told to stop stays stopped even after ``release_restart`` puts a
        # fresh event here for whatever comes next.
        self.cancel = threading.Event()
        window.events.closing += self._on_closing

    def _claim_close(self, *, manual: bool) -> str | None:
        """Claim the right to close: ``None`` grants it, a sentence refuses.

        The steps are one decision and share ``_phase_lock``, so a close either
        stops the update before the swap or finds the swap already under way.
        Everything that ends the window asks this first, and says whose close
        it is. A ``manual`` one - Settings asking for a restart - stands down
        while the bundle is being replaced and for the stretch after it while
        the update is relaunching, since a copy is on its way up; and when it
        is granted it takes that same claim for itself, so a second
        confirmation - a double-click, or a second tab left open on the fixed
        port - is refused rather than opening a second copy. The closes that
        finish a relaunch, the update's own and the one a granted restart
        makes itself, are not manual and go through.
        """

        with self._phase_lock:
            if self.swapping.is_set():
                return RESTART_DURING_UPDATE
            if manual:
                if self.relaunching.is_set():
                    return RESTART_ALREADY_UNDER_WAY
                self.relaunching.set()
            self.cancel.set()
            return None

    def may_restart(self) -> str | None:
        """Answer Settings asking to close this window and open a fresh copy."""

        return self._claim_close(manual=True)

    def release_restart(self) -> None:
        """Hand back a claimed restart whose relaunch never happened.

        ``updater.relaunch`` spawns a process and can fail like any other
        spawn. Nothing is coming up when it does, so the claim taken to start
        it goes back rather than standing forever, and the next restart -
        the reader's next try, or the one after the next saved choice - is
        free to take it.

        The claim also set ``cancel``, which is what keeps a download already
        in flight from entering the swap behind a departing window. That stop
        stands for the update it stopped, which goes on reading the event it
        was told through; the flow gets a fresh one, so a restart that never
        happened does not cancel every update for the life of the process.
        """

        with self._phase_lock:
            self.relaunching.clear()
            self.cancel = threading.Event()

    def _on_closing(self) -> bool:
        """Quit cancels a download in flight but waits out the bundle swap.

        Returning False from a closing handler makes pywebview keep the window
        open, so the few seconds between parking the old bundle and moving the
        new one in cannot be interrupted.
        """

        return self._claim_close(manual=False) is None

    def _begin_swap(self, cancel: threading.Event | None = None) -> bool:
        """Enter the protected swap phase unless a quit already cancelled us.

        ``cancel`` is the event the attempt asking started with. It is checked
        alongside the flow's current one, so an attempt that was told to stop
        cannot swap on the strength of the fresh event a later restart left
        behind, and nothing swaps while a close is being granted.
        """

        with self._phase_lock:
            if self.cancel.is_set() or (cancel is not None and cancel.is_set()):
                return False
            self.swapping.set()
        self._window_call("set_title", f"{WINDOW_TITLE} — installing update…")
        return True

    def _end_swap(self, *, relaunching: bool) -> None:
        """Leave the swap phase, handing the restart on where one is coming.

        Both flags move under a single ``_phase_lock``, so there is no moment
        in which neither the swap nor the relaunch owns the restart and a
        manual one could slip through and start a second copy. A cancelled or
        failed update sets nothing and leaves the manual restart available.
        """

        with self._phase_lock:
            if relaunching:
                self.relaunching.set()
            self.swapping.clear()

    def check_in_background(self, *, interactive: bool) -> threading.Thread:
        """Run ``check`` on a daemon thread so a quit can abandon it.

        Everything before the swap (release lookup, checksum fetch, download,
        extraction) is safe to drop mid-flight. The swap itself is protected by
        ``_on_closing`` refusing to close and ``wait_for_swap`` joining on exit.
        """

        with self._phase_lock:
            current = getattr(self, "_worker", None)
            if current is not None and current.is_alive():
                busy = True
            else:
                busy = False
                current = threading.Thread(
                    target=self.check, kwargs={"interactive": interactive}, daemon=True
                )
                self._worker = current
                current.start()
        if busy and interactive:
            self._window_call(
                "create_confirmation_dialog", "ChatLab", "An update check or download is already running."
            )
        return current

    def wait_for_swap(self, timeout: float = 300, grace: float = 3.0) -> None:
        """Called on the way out: forbid new swaps, then wait for the worker.

        A worker in the swap is waited for in full. Any other worker gets
        ``grace`` seconds to notice the cancel, stop ``ditto``, and delete its
        staging directory; one stalled in a network read is abandoned and its
        directory is swept on the next launch by ``remove_stale_work_dirs``.
        """

        with self._phase_lock:
            self.cancel.set()
            swapping = self.swapping.is_set()
        worker = getattr(self, "_worker", None)
        if worker is None or not worker.is_alive():
            return
        if swapping:
            logging.info("Waiting for the update swap to finish before exiting")
            worker.join(timeout)
        else:
            worker.join(grace)
            if worker.is_alive():
                logging.info("Abandoning a stalled update worker; staging is swept on next launch")

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
        relaunching = False
        # The cancellation this attempt answers to, taken once and read
        # everywhere below, so a restart that gives its claim back mid-download
        # cannot leave half of this update reading one event and half another.
        with self._phase_lock:
            cancel = self.cancel
        try:
            updater.install_update(
                release,
                self.bundle,
                progress=self._report_progress,
                begin_swap=lambda: self._begin_swap(cancel),
                cancelled=cancel.is_set,
            )
        except updater.UpdateCancelled as error:
            logging.info("%s", error)
            return
        except updater.UpdateError as error:
            logging.error("Update failed: %s", error)
            self._window_call("set_title", WINDOW_TITLE)
            self._window_call("create_confirmation_dialog", "Update failed", str(error))
            return
        else:
            relaunching = True
        finally:
            self._end_swap(relaunching=relaunching)
        logging.info("Relaunching ChatLab %s", release.version)
        try:
            updater.relaunch(self.bundle)
        except OSError as error:
            # The update is installed; only the reopening fell over. Closing
            # the window now would take the app away with nothing coming to
            # replace it, so it stays open, says so, and gives the restart
            # back for the reader to try again.
            logging.error("Could not reopen ChatLab after the update: %s", error)
            self.release_restart()
            self._window_call("set_title", WINDOW_TITLE)
            self._window_call(
                "create_confirmation_dialog",
                "Update installed",
                f"ChatLab {release.version} is installed, but it could not be reopened "
                f"({error}).\n\nQuit and open ChatLab again to run it.",
            )
            return
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
    window = None
    flow: UpdateFlow | None = None

    def restart() -> str | None:
        """Quit and open a fresh copy, which Settings asks for after a change.

        A restart is a close, so it asks the update flow to claim one, and
        stands down with the reason the flow gives for as long as somebody
        else owns it: mid-swap the bundle is being replaced, and after that a
        copy is already coming up, whether the update is bringing it or an
        earlier press of this same button is. Otherwise it is ordered the way
        the update flow orders it - the replacement is started first and the
        window closed behind it, so what is on screen is replaced rather than
        vanishing ahead of a launch that may not come.
        """

        if flow is not None:
            declined = flow.may_restart()
            if declined is not None:
                logging.info("Restart declined: %s", declined)
                return declined
        logging.info("Restarting ChatLab %s at the reader's request", __version__)
        try:
            updater.relaunch(bundle)
        except OSError as error:
            # Nothing is coming up, so the window stays where it is and the
            # claim goes back; the reader can press the button again.
            logging.error("Could not reopen ChatLab: %s", error)
            if flow is not None:
                flow.release_restart()
            return RESTART_FAILED
        try:
            window.destroy()
        except Exception as error:  # noqa: BLE001 - window is gone; log and carry on
            logging.info("Window close skipped: %s", error)
        return None

    # A run that is not from a bundle has nothing to reopen, so Settings
    # leaves the button off the page rather than quitting into nothing.
    if bundle is not None:
        desktop.offer_restart(restart)
    demo, local_url = start_local_server()
    logging.info("Started ChatLab %s at %s", __version__, local_url)
    # Only once the window is really opening. A smoke test runs for seconds
    # and has nothing to watch; this records the run-up to a memory kill,
    # which is a thing that happens to sessions, not to checks.
    device_memory.watch_memory()

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
                updater.remove_stale_work_dirs(bundle)
            flow.check_in_background(interactive=False)

        webview.start(
            func=after_startup,
            gui="cocoa",
            private_mode=False,
            storage_path=str(support_directory / "WebKit"),
            menu=[
                # "__app__" is pywebview's name for the ChatLab menu, where macOS
                # apps keep Check for Updates — just under About.
                Menu(
                    "__app__",
                    [
                        MenuAction(
                            "Check for Updates…",
                            lambda: flow.check_in_background(interactive=True),
                        ),
                    ],
                ),
                Menu(
                    "Help",
                    [
                        MenuAction("ChatLab Releases", lambda: webbrowser.open(updater.RELEASES_PAGE_URL)),
                    ],
                ),
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
        help="check Metal support (including a tiny model on MPS), verify the server, and exit",
    )
    parser.add_argument("--version", action="version", version=f"ChatLab {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    logs.log_environment(configure_logging())
    args = parse_args(argv)
    try:
        return smoke_test() if args.smoke_test else run_desktop()
    except Exception:
        logging.exception("ChatLab failed to start")
        raise


if __name__ == "__main__":
    sys.exit(main())
