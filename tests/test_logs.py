"""The log: where it goes, how much is kept, how loud it is, and what it opens with."""

import logging
import logging.handlers
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chatlab import logs
import settings_sandbox


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def _release_claim() -> None:
    """Drop the lock this process holds, so each test claims from nothing."""

    if logs._claim is not None:
        logs._claim.close()
        logs._claim = None


class _Handlers:
    """Put the root logger back the way it was, whatever a test does to it."""

    def __enter__(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level
        root.handlers = []
        return root

    def __exit__(self, *_):
        root = logging.getLogger()
        for handler in root.handlers:
            handler.close()
        root.handlers = self._handlers
        root.setLevel(self._level)
        return False


class PathTests(unittest.TestCase):
    def test_macos_writes_where_console_looks(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(logs.log_path(), Path.home() / "Library" / "Logs" / "ChatLab" / "ChatLab.log")

    def test_other_platforms_use_the_state_directory(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(logs.log_path(), Path.home() / ".local" / "state" / "chatlab" / "ChatLab.log")

    def test_the_state_variable_moves_it(self):
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": "/tmp/state"}, clear=True), mock.patch.object(
            sys, "platform", "linux"
        ):
            self.assertEqual(logs.log_path(), Path("/tmp/state/chatlab/ChatLab.log"))

    def test_the_log_path_variable_names_the_file_itself(self):
        with mock.patch.dict(os.environ, {logs.LOG_PATH_ENV: "~/somewhere.log"}, clear=True):
            self.assertEqual(logs.log_path(), Path.home() / "somewhere.log")


class LevelTests(unittest.TestCase):
    def test_the_default_is_info(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(logs.level(), logging.INFO)

    def test_a_name_is_read_in_any_case(self):
        with mock.patch.dict(os.environ, {logs.LOG_LEVEL_ENV: "debug"}, clear=True):
            self.assertEqual(logs.level(), logging.DEBUG)

    def test_a_number_is_read_too(self):
        with mock.patch.dict(os.environ, {logs.LOG_LEVEL_ENV: "25"}, clear=True):
            self.assertEqual(logs.level(), 25)

    def test_nonsense_falls_back_to_info_rather_than_to_silence(self):
        with mock.patch.dict(os.environ, {logs.LOG_LEVEL_ENV: "loud"}, clear=True):
            self.assertEqual(logs.level(), logging.INFO)


class ClaimTests(unittest.TestCase):
    """One process owns the plain log name; a second writes beside it.

    A rotating handler renames files as it rolls over, and two processes
    doing that to the same file lose records. Two are possible: the desktop
    launcher takes a free port rather than refusing a second instance, and a
    checkout runs beside the installed app.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "ChatLab.log"
        self.addCleanup(_release_claim)
        _release_claim()

    def test_the_first_claim_takes_the_plain_name(self):
        self.assertEqual(logs.claim(self.path), self.path)

    def test_a_second_process_writes_under_its_own_name(self):
        # A real second process, because flock is only guaranteed to refuse
        # another process: macOS lets one process lock the same file twice
        # through two descriptors, so a claim faked in this one would prove
        # nothing about the case this guards.
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys;from pathlib import Path;from chatlab import logs;"
                "print(logs.claim(Path(sys.argv[1])),flush=True);sys.stdin.read()",
                str(self.path),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )


        def done() -> None:
            child.kill()
            child.wait()
            child.stdout.close()
            child.stdin.close()

        self.addCleanup(done)
        self.assertEqual(child.stdout.readline().strip(), str(self.path), "the child took the plain name")

        self.assertEqual(logs.claim(self.path), self.path.with_name(f"ChatLab-{os.getpid()}.log"))

    def test_a_lock_the_kernel_refuses_moves_this_process_aside(self):
        with mock.patch("fcntl.flock", side_effect=OSError("held elsewhere")):
            self.assertEqual(logs.claim(self.path), self.path.with_name(f"ChatLab-{os.getpid()}.log"))
        self.assertIsNone(logs._claim, "a refused claim is not recorded as held")

    def test_asking_twice_in_one_process_does_not_rename_its_own_log(self):
        self.assertEqual(logs.claim(self.path), self.path)
        self.assertEqual(logs.claim(self.path), self.path, "configure() is safe to call twice")

    def test_a_lock_that_cannot_be_written_leaves_the_name_alone(self):
        with mock.patch("builtins.open", side_effect=OSError("read-only")):
            self.assertEqual(logs.claim(self.path), self.path)


class ConfigureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "logs" / "ChatLab.log"
        self.addCleanup(_release_claim)
        _release_claim()

    def test_the_file_is_rotated_rather_than_left_to_grow(self):
        with _Handlers() as root, mock.patch.dict(
            os.environ, {logs.LOG_PATH_ENV: str(self.path)}, clear=True
        ):
            self.assertEqual(logs.configure(), self.path)
            files = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].maxBytes, logs.MAX_BYTES)
            self.assertEqual(files[0].backupCount, logs.BACKUP_COUNT)

    def test_the_directory_is_made_and_the_lines_land_in_the_file(self):
        with _Handlers(), mock.patch.dict(os.environ, {logs.LOG_PATH_ENV: str(self.path)}, clear=True):
            logs.configure()
            logging.getLogger("chatlab.test").info("a line worth keeping")
            logging.shutdown()
        self.assertIn("a line worth keeping", self.path.read_text())

    def test_configuring_twice_does_not_write_every_line_twice(self):
        with _Handlers(), mock.patch.dict(os.environ, {logs.LOG_PATH_ENV: str(self.path)}, clear=True):
            logs.configure()
            logs.configure()
            logging.getLogger("chatlab.test").info("said once")
            for handler in logging.getLogger().handlers:
                handler.flush()
            self.assertEqual(self.path.read_text().count("said once"), 1)

    def test_a_file_that_cannot_be_opened_costs_the_file_and_not_the_launch(self):
        with _Handlers() as root, mock.patch.dict(
            os.environ, {logs.LOG_PATH_ENV: str(self.path)}, clear=True
        ), mock.patch.object(Path, "mkdir", side_effect=OSError("read-only")):
            self.assertIsNone(logs.configure())
            self.assertTrue(root.handlers, "the console log should survive a file that will not open")

    def test_nothing_goes_to_disk_when_the_caller_says_not_to(self):
        with _Handlers(), mock.patch.dict(os.environ, {logs.LOG_PATH_ENV: str(self.path)}, clear=True):
            self.assertIsNone(logs.configure(to_file=False))
            self.assertFalse(self.path.exists())

    def test_the_level_comes_from_the_environment(self):
        with _Handlers() as root, mock.patch.dict(
            os.environ, {logs.LOG_PATH_ENV: str(self.path), logs.LOG_LEVEL_ENV: "warning"}, clear=True
        ):
            logs.configure()
            self.assertEqual(root.level, logging.WARNING)
            self.assertTrue(all(handler.level == logging.WARNING for handler in root.handlers))


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.noisy = logging.getLogger(logs.NOISY_LOGGERS[0])
        previous = self.noisy.level
        self.addCleanup(self.noisy.setLevel, previous)

    def test_the_libraries_are_held_at_warnings(self):
        logs.quiet_noisy_loggers(logging.INFO)
        self.assertEqual(self.noisy.level, logging.WARNING)

    def test_debug_lets_everything_through(self):
        logs.quiet_noisy_loggers(logging.INFO)
        logs.quiet_noisy_loggers(logging.DEBUG)
        self.assertEqual(self.noisy.level, logging.NOTSET)


class EnvironmentRecordTests(unittest.TestCase):
    def record(self, target=None) -> str:
        with self.assertLogs(logs.__name__, level="INFO") as caught:
            logs.log_environment(target)
        return "\n".join(caught.output)

    def test_it_names_the_build_the_machine_and_the_packages(self):
        written = self.record()
        self.assertIn("ChatLab", written)
        self.assertIn("Python", written)
        self.assertIn("of memory", written)
        self.assertIn("torch", written)
        self.assertIn("weight precision", written)

    def test_it_says_where_the_log_is_and_how_much_is_kept(self):
        written = self.record(Path("/tmp/ChatLab.log"))
        self.assertIn("/tmp/ChatLab.log", written)
        self.assertIn(str(logs.BACKUP_COUNT), written)

    def test_it_distinguishes_a_packaged_app_from_a_checkout(self):
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertIn("packaged app", self.record())
        self.assertIn("checkout at", self.record())

    def test_a_missing_package_is_recorded_as_absent_rather_than_omitted(self):
        with mock.patch.object(logs, "RECORDED_PACKAGES", ("not-a-real-distribution",)):
            self.assertIn("not-a-real-distribution absent", logs.package_versions())

    def test_settings_that_cannot_be_read_do_not_stop_the_record(self):
        with mock.patch("chatlab.settings.load", side_effect=ValueError("bad json")):
            self.assertIn("unreadable", logs.settings_note())


if __name__ == "__main__":
    unittest.main()
