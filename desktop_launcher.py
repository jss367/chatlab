"""Native macOS launcher for the ChatLab Gradio application."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import socket
import sys
from pathlib import Path
from urllib.request import urlopen

from app import build_app


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


def run_desktop() -> int:
    """Open ChatLab in a native WebKit window until the user quits."""

    import webview

    support_directory = app_support_directory()
    support_directory.mkdir(parents=True, exist_ok=True)
    demo, local_url = start_local_server()
    logging.info("Started ChatLab at %s", local_url)

    try:
        webview.create_window(
            WINDOW_TITLE,
            local_url,
            width=1440,
            height=960,
            min_size=(960, 680),
            background_color="#f8fafc",
            text_select=True,
            zoomable=True,
        )
        webview.start(
            gui="cocoa",
            private_mode=False,
            storage_path=str(support_directory / "WebKit"),
        )
    finally:
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
