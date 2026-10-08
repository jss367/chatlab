"""Rendered UI regressions. Run separately from the handler-only unittest suite."""

import os
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from playwright.sync_api import expect, sync_playwright
from navigation_diagnostics import BROWSER_DIAGNOSTICS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("CHATLAB_BROWSER_ARTIFACTS", ROOT / ".context/browser-tests"))


class BrowserFlows:
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        folder = Path(self.folder.name)
        self.log = (folder / "server.log").open("w+")
        self.addCleanup(self.log.close)
        self.server = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("server.py")), str(folder)],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
        )
        self.addCleanup(self.stop_server)
        deadline = time.monotonic() + 90
        while not (folder / "ready").exists():
            if self.server.poll() is not None or time.monotonic() > deadline:
                self.log.seek(0)
                self.fail("Browser fixture failed to start:\n" + self.log.read())
            time.sleep(0.1)
        self.url = (folder / "ready").read_text()
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        self.browser = getattr(self.playwright, self.engine).launch()
        self.addCleanup(self.browser.close)
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.context.add_init_script(BROWSER_DIAGNOSTICS)
        self.addCleanup(self.context.close)
        self.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        self.page = self.context.new_page()
        self.queue_requests = []
        self.page.on('requestfinished', self.record_queue_request)
        self.errors = []
        self.error_events = []
        self.page.on("pageerror", lambda error: self.record_page_error('original', error))
        self.page.set_default_timeout(15000)
        # Assertions have a separate timeout; queued restoration on CI can
        # outlast their five-second default even when page actions succeed.
        expect.set_options(timeout=15000)
        self.page.goto(self.url)
        expect(self.page.locator("#conversation-list input[type=radio]")).to_have_count(1)
        expect(self.page.get_by_role("button", name="Send", exact=True)).to_be_visible()

    def stop_server(self):
        self.server.terminate()
        try:
            self.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=10)

    def record_queue_request(self, request):
        if '/queue/join' not in request.url:
            return
        try:
            response = request.response()
            self.queue_requests.append({
                'time_ms': time.time_ns() // 1_000_000,
                'fn_index': request.post_data_json.get('fn_index'),
                'event_id': response.json().get('event_id') if response else None,
            })
            self.queue_requests[:] = self.queue_requests[-100:]
        except Exception as error:
            self.queue_requests.append({'error': type(error).__name__})

    def record_page_error(self, context, error):
        # Retain every error in the original assertion; timestamps add evidence.
        self.errors.append(str(error))
        self.error_events.append({'context': context, 'time_ms': time.time_ns() // 1_000_000,
                                  'error': str(error)})

    def tearDown(self):
        # Retain a trace, screenshot and server log even when an assertion fails.
        # Print before artifact operations, so diagnosis does not require download.
        try:
            events = self.page.evaluate('window.__navigationDiagnostics || []')
            print('NAV_BROWSER ' + json.dumps(events), flush=True)
            print('NAV_REQUESTS ' + json.dumps(self.queue_requests), flush=True)
            print('NAV_ERRORS ' + json.dumps(self.error_events), flush=True)
        except Exception as error:
            print('NAV_DIAGNOSTIC_ERROR ' + type(error).__name__, flush=True)
        finally:
            self.log.flush()
            lines = Path(self.folder.name, 'server.log').read_text().splitlines()
            records = [line for line in lines if line.startswith('NAV_SERVER ')]
            # Preserve navigation boundaries even after many timer callbacks.
            anchors = [i for i, line in enumerate(records) if json.loads(line[11:])['handler'] != 'poll'][-40:]
            selected = sorted(set(anchors + list(range(max(0, len(records) - 80), len(records)))))
            for i in selected:
                line = records[i]
                print(line, flush=True)
            queue_records = [line for line in lines if line.startswith('NAV_QUEUE ')]
            anchors = [i for i, line in enumerate(queue_records) if json.loads(line[10:])['handler'] != 'poll'][-40:]
            selected = sorted(set(anchors + list(range(max(0, len(queue_records) - 80), len(queue_records)))))
            for i in selected:
                print(queue_records[i], flush=True)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        name = f"{self.engine}-{self._testMethodName}"
        self.page.screenshot(path=str(ARTIFACTS / f"{name}.png"), full_page=True)
        self.context.tracing.stop(path=str(ARTIFACTS / f"{name}.zip"))
        self.log.flush()
        (ARTIFACTS / f"{name}.log").write_text(Path(self.folder.name, "server.log").read_text())
        self.assertEqual(self.errors, [], "Uncaught browser JavaScript errors")

    def send(self, text):
        self.page.locator("#message-input textarea").fill(text)
        self.page.get_by_role("button", name="Send", exact=True).click()
        expect(self.page.locator("#stop-button")).to_be_visible()
        expect(self.page.locator("#conversation")).to_contain_text("Hello")

    def finished(self):
        expect(self.page.locator("#stop-button")).to_be_hidden(timeout=20000)
        expect(self.page.get_by_role("button", name="Send", exact=True)).to_be_visible()

    def test_extensions_navigation_and_saved_choice(self):
        self.page.locator('#nav label').filter(has_text="Extensions").click()
        expect(self.page.locator("#extensions-page")).to_be_visible()
        expect(self.page.locator("#conversation-pane")).to_be_hidden()
        expect(self.page.locator("#chat-page")).to_be_hidden()
        expect(self.page.locator("#settings-page")).to_be_hidden()
        choice = self.page.locator("#enabled-extensions input[type=checkbox]").first
        choice.check()
        expect(self.page.locator("#extensions-status")).to_contain_text("Restart ChatLab")
        self.page.locator('#nav label').filter(has_text="Settings").click()
        expect(self.page.locator("#extensions-page")).to_be_hidden()
        self.page.get_by_role("button", name="Manage extensions", exact=True).click()
        expect(self.page.locator("#extensions-page")).to_be_visible()
        expect(self.page.locator("#settings-page")).to_be_hidden()
        expect(choice).to_be_checked()
        self.page.reload()
        self.page.locator('#nav label').filter(has_text="Extensions").click()
        expect(choice).to_be_checked()
        expect(self.page.locator("#extensions-status")).to_contain_text("Restart ChatLab")
        self.page.locator('#nav label').filter(has_text="Models").click()
        expect(self.page.locator("#extensions-page")).to_be_hidden()
        expect(self.page.locator("#models-page")).to_be_visible()
        self.page.locator('#nav label').filter(has_text="Extensions").click()
        expect(self.page.locator("#extensions-page")).to_be_visible()
        expect(self.page.locator("#models-page")).to_be_hidden()

    def test_send_and_stop_keeps_partial_reply(self):
        self.send("Stop this response")
        expect(self.page.locator("#message-input textarea")).to_have_value("")
        self.page.locator("#stop-button").click()
        self.finished()
        transcript = self.page.locator("#conversation")
        expect(transcript).to_contain_text("Stop this response")
        expect(transcript).to_contain_text("Hello")
        stopped = transcript.inner_text()
        self.assertLess(stopped.count("Hello"), 20, "Stop must interrupt the scripted reply")
        reply = re.search(r"Hello(?: worldHello)*(?: world)?", stopped).group()
        self.page.reload()
        expect(self.page.locator("#conversation")).to_contain_text("Stop this response")
        expect(self.page.locator("#conversation").get_by_text(reply, exact=True)).to_be_visible()

    def test_switch_during_generation_keeps_reply_in_original(self):
        self.send("Original conversation")
        self.page.get_by_role("button", name="New", exact=True).click()
        expect(self.page.locator("#conversation-list input[type=radio]")).to_have_count(2)
        expect(self.page.locator("#conversation")).not_to_contain_text("Original conversation")
        expect(self.page.locator("#conversation-list")).to_contain_text("Generating")
        self.page.locator("#message-input textarea").fill("Draft in second conversation")
        self.finished()
        expect(self.page.locator("#conversation")).not_to_contain_text("Hello")
        expect(self.page.locator("#message-input textarea")).to_have_value("Draft in second conversation")
        self.page.locator("#conversation-list label").first.click()
        expect(self.page.locator("#conversation")).to_contain_text("Original conversation")
        expect(self.page.locator("#conversation")).to_contain_text("Hello world" * 20)
        expect(self.page.locator("#conversation-list")).not_to_contain_text("Generating")

    def test_token_replacement_creates_branch(self):
        self.send("Replace a reply token")
        self.finished()
        self.page.get_by_text("Token view", exact=True).click()
        token = self.page.locator("#token-strip").get_by_text("Hello", exact=True).first
        token.click(button="right")
        menu = self.page.locator("#token-context-menu")
        expect(menu).to_contain_text("Continue from")
        menu.locator("textarea").fill("How")
        menu.get_by_role("button", name="Continue with this text", exact=True).click()
        expect(self.page.locator("#stop-button")).to_be_visible()
        self.finished()
        expect(self.page.locator("#conversation-list input[type=radio]")).to_have_count(2)
        expect(self.page.locator("#token-strip")).to_contain_text("How")
        self.page.get_by_text("Rendered view", exact=True).click()
        expect(self.page.locator("#conversation")).to_contain_text("How")
        self.page.locator("#conversation-list label").first.click()
        expect(self.page.locator("#conversation")).not_to_contain_text("How")
        expect(self.page.locator("#conversation")).to_contain_text("Hello world" * 20)

    def test_fork_and_reload_saved_history(self):
        self.send("Saved original")
        self.finished()
        self.page.get_by_role("button", name="Fork", exact=True).click()
        expect(self.page.locator("#conversation-list input[type=radio]")).to_have_count(2)
        expect(self.page.locator("#conversation")).to_contain_text("Saved original")
        self.send("Only on the fork")
        self.finished()
        # A new context proves restoration does not depend on in-page state.
        restored = self.context.browser.new_context()
        restored.add_init_script(BROWSER_DIAGNOSTICS)
        self.addCleanup(restored.close)
        page = restored.new_page()
        page.on('requestfinished', self.record_queue_request)
        # Cover the fresh context as well as the original page on failures.
        page.on('pageerror', lambda error: self.record_page_error('restored', error))
        self.addCleanup(lambda: print('NAV_RESTORED ' + json.dumps(page.evaluate('window.__navigationDiagnostics || []')), flush=True))
        page.goto(self.url)
        expect(page.locator("#conversation-list input[type=radio]")).to_have_count(2)
        expect(page.locator("#conversation")).to_contain_text("Only on the fork")
        page.locator("#conversation-list label").first.click()
        expect(page.locator("#conversation")).to_contain_text("Saved original")
        expect(page.locator("#conversation")).not_to_contain_text("Only on the fork")
        page.reload()
        expect(page.locator("#conversation")).to_contain_text("Saved original")
        expect(page.locator("#conversation")).not_to_contain_text("Only on the fork")


# No silent skip: requesting an engine without its binary must fail CI.
for engine in os.environ.get("CHATLAB_TEST_BROWSERS", "chromium,webkit").split(","):
    if engine not in {"chromium", "webkit"}:
        raise ValueError(f"Unsupported browser: {engine}")
    globals()[f"{engine.title()}Flows"] = type(
        f"{engine.title()}Flows", (BrowserFlows, unittest.TestCase), {"engine": engine}
    )
