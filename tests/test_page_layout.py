"""The page around every page: the nav, the shell, the panes and how they resize,
and which controls each page holds."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

from chatlab import app
from chatlab.ui import common, icons, models_page, runtime
from chatlab import settings
from chatlab.model_cache import CacheStatus
from chatlab.model_runtime import ModelManager

from models_support import OLMO, cached
import settings_sandbox
from ui_support import listeners_named


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class PageLayoutTests(unittest.TestCase):
    """The nav picks a page: the model controls sit on Models, the settings on
    Settings, and the conversation on Chat."""

    ON_MODELS_PAGE = [
        "Hugging Face model ID",
        "Hugging Face token (optional)",
        "Downloaded models",
        "Sort by",
        "Search Hugging Face",
        "Search results",
    ]
    ON_SETTINGS_PAGE = [
        "System prompt",
        "Send previous reasoning back to the model",
        "Measure prompt tokens",
        "Enter sends the message",
        "Context limit (tokens)",
    ]
    # Everything about drawing a picture is on Images, including its own
    # sliders: the Chat page's sampling controls say nothing to a diffusion
    # model, and these say nothing to a language one.
    ON_IMAGES_PAGE = [
        "Prompt",
        "Negative prompt",
        "Picture",
        "Denoising steps",
        "Guidance scale",
        "Size",
        "Seed",
        "Randomize seed",
        "Record cross-attention",
        "Step",
        "Prompt tokens — click one for its map",
    ]
    # Sampling sits with the conversation, not behind the nav: these are what
    # a reader moves between one retry and the next.
    ON_CHAT_PAGE = [
        "Conversation",
        "Message",
        "Color tokens by",
        "Text to score",
        "Temperature",
        "Top-p",
        "Top-k (0 disables)",
        "Maximum new tokens",
        "Random seed",
        "New seed each response",
    ]

    def setUp(self):
        self.demo = app.build_app()

    def by_id(self, elem_id):
        return next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "elem_id", None) == elem_id
        )

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def within(self, block, container) -> bool:
        parent = getattr(block, "parent", None)
        while parent is not None:
            if parent is container:
                return True
            parent = getattr(parent, "parent", None)
        return False

    def test_each_control_sits_on_its_page(self):
        for page, labels in [
            ("models-page", self.ON_MODELS_PAGE),
            ("settings-page", self.ON_SETTINGS_PAGE),
            ("chat-page", self.ON_CHAT_PAGE),
            ("images-page", self.ON_IMAGES_PAGE),
        ]:
            container = self.by_id(page)
            for label in labels:
                with self.subTest(page=page, label=label):
                    self.assertTrue(self.within(self.labelled(label), container))

    def test_the_nav_offers_every_page_and_starts_on_chat(self):
        nav = self.by_id("nav")
        self.assertIsInstance(nav, gr.Radio)
        self.assertEqual(
            [value for _, value in nav.choices], ["Chat", "Images", "Models", "Settings"]
        )
        self.assertEqual(nav.value, "Chat")
        self.assertTrue(self.within(nav, self.by_id("nav-pane")))

    def test_the_shell_spans_the_whole_window(self):
        # Gradio otherwise caps the page at one of a handful of widths and
        # centers it, leaving empty room down each side on a wide screen.
        self.assertTrue(self.demo.fill_width)

    def test_the_nav_pane_is_thin(self):
        # Wide enough for the longest page name at the tile's small type,
        # and no wider: the pane is a signpost, not a sidebar.
        self.assertLessEqual(app.NAV_PANE_WIDTH, 96)
        self.assertEqual(self.by_id("nav-pane").min_width, app.NAV_PANE_WIDTH)

    def test_each_nav_tile_shows_an_icon_above_the_page_name(self):
        for page in app.PAGES:
            with self.subTest(page=page):
                tile = f'#nav label[data-testid="{page}-radio-label"]'
                # The icon is a mask over the tile rather than a glyph in it,
                # so it generates no text for a screen reader to read out and
                # the label's own name - printed under it - stands for the
                # tile on its own.
                self.assertIn(
                    icons.mask_rule(f"{tile}::before", app.NAV_ICONS[page]),
                    app.CSS,
                )
        self.assertIn("#nav label span { font-size:", app.CSS)

    def test_no_nav_tile_is_drawn_with_an_emoji(self):
        # Emoji are a different typeface per glyph: the weights, the colours
        # and the optical sizes never agreed, and one of them arrived as an
        # empty box on a machine without the font.
        for page, icon in app.NAV_ICONS.items():
            with self.subTest(page=page):
                self.assertIn(icon, icons.ICONS)

    def test_a_compact_window_stacks_the_images_panes_too(self):
        # Its two panes want about 620px between them, so in a narrow window
        # the readings would sit off the side of a row that neither wraps
        # nor scrolls sideways.
        compact = app.CSS[app.CSS.index("@media (max-width: 850px)") :]
        compact = compact[: compact.index("\n}")]

        self.assertIn("#images-columns", compact)
        self.assertIn("#images-workspace", compact)
        self.assertIn("#image-inspector", compact)

    def test_each_readings_pane_has_a_handle_on_its_seam(self):
        # The handle is a flex item between the workspace and the pane, so
        # the seam it sits on is the edge the reader drags.
        for pane_id, handle_id in [
            ("inspector-pane", "inspector-resizer"),
            ("image-inspector", "image-inspector-resizer"),
        ]:
            with self.subTest(pane=pane_id):
                pane = self.by_id(pane_id)
                handle = self.by_id(handle_id)
                self.assertIs(handle.parent, pane.parent)
                self.assertIn(f'data-pane="{pane_id}"', handle.value)
                self.assertIn(f'data-property="--{pane_id}-width"', handle.value)
                self.assertIn(f'data-store="chatlab.{pane_id}-width"', handle.value)
                # A width the reader chose is written to that property, so
                # every rule that sizes the pane has to read it - including
                # the narrower window's, which sets a smaller default.
                for rule in [
                    line
                    for line in app.CSS.splitlines()
                    if "flex" in line and f"--{pane_id}-width" in line
                ]:
                    self.assertIn(f"var(--{pane_id}-width,", rule)
                self.assertEqual(
                    app.CSS.count(f"var(--{pane_id}-width,"), 2, pane_id
                )
                # And the script that writes it knows the pane by the same name.
                self.assertIn(f"'{pane_id}'", app.RESIZE_JS)

    def test_a_box_that_only_shows_text_is_read_only_rather_than_dead(self):
        # Gradio draws a non-interactive textbox as a disabled textarea, and a
        # browser hands a disabled control no wheel and no caret, so text
        # longer than the box - the maze workbench's full response, the prompt
        # behind it - had nothing past its first screenful that could be
        # reached. Read-only refuses the same edits and gives the text back.
        self.assertTrue(
            any(fn.js == app.READ_ONLY_TEXT_JS for fn in self.demo.fns.values())
        )
        self.assertIn('textarea[data-testid="textbox"]', app.READ_ONLY_TEXT_JS)
        self.assertIn("box.disabled = false;", app.READ_ONLY_TEXT_JS)
        self.assertIn("box.readOnly = true;", app.READ_ONLY_TEXT_JS)
        # A page is built when the reader first opens it and an extension's
        # page later still, so the boxes are met as they arrive rather than
        # counted once at load.
        self.assertIn("new MutationObserver", app.READ_ONLY_TEXT_JS)
        self.assertIn("subtree: true", app.READ_ONLY_TEXT_JS)
        self.assertIn("attributeFilter: ['disabled']", app.READ_ONLY_TEXT_JS)

    def test_the_handle_keeps_touch_gestures_off_its_strip(self):
        # A touch device wider than the stacking breakpoint still drags the
        # handle, and a browser that reads that drag as a pan or a zoom takes
        # the pointer back mid-resize, which leaves the pane part-moved.
        # Refusing the pointerdown does not stop it; only this does.
        rule = app.CSS[app.CSS.index(".pane-resizer {") :]
        rule = rule[: rule.index("}")]

        self.assertIn("touch-action: none", rule)

    def test_the_stacked_layout_drops_the_handles(self):
        # Under 850px the panes are rows, one above the other, where a width
        # would mean a height and a sideways drag would mean nothing.
        compact = app.CSS[app.CSS.index("@media (max-width: 850px)") :]
        compact = compact[: compact.index("\n}")]

        self.assertIn("#inspector-resizer, #image-inspector-resizer", compact)
        self.assertIn("display: none", compact)

    def test_a_saved_width_is_fitted_to_the_room_the_pane_has(self):
        # A width chosen on a wide window has to be cut down when the window
        # narrows, and the figure to cut it to is the row the pane sits in
        # rather than the window itself: the Chat row gives up space to the
        # conversations pane and the Images row does not. Watching the rows
        # covers a page that was away while the window changed as well, since
        # the row it is built into reports its size the moment it has one.
        # What that watching is worth when a page comes and goes is run
        # through in PaneResizeScriptTests.
        self.assertIn("new ResizeObserver", app.RESIZE_JS)
        self.assertIn("rows.observe(row)", app.RESIZE_JS)
        self.assertIn("window.addEventListener('resize'", app.RESIZE_JS)
        # Where the panes become rows a width would mean a height, so the
        # script stops fitting at the same width the stylesheet stops
        # reading the property.
        stacked = "(max-width: 850px)"
        self.assertIn(f"@media {stacked}", app.CSS)
        self.assertIn(f"matchMedia('{stacked}')", app.RESIZE_JS)

    def test_the_shell_is_not_pushed_off_the_bottom_of_the_window(self):
        # The shell is a window tall, so anything that takes room above it
        # hangs the same distance off the bottom - and what falls off is the
        # tile the nav pins to its own bottom edge, Settings. The token
        # menu's bridge controls are hidden, but Gradio wraps each in a form
        # that is not, and a shown wrapper is still a flex item earning a gap
        # in the column it shares with the shell.
        self.assertIn(".form:has(> .token-menu-bridge) { display: none", app.CSS)
        self.assertIn("#shell {\n  height: 100dvh;", app.CSS)

    def test_every_icon_in_the_interface_comes_from_the_one_set(self):
        # Emoji are a different typeface per glyph, so a row of them agreed on
        # neither weight nor colour nor optical size, and some machines drew a
        # box instead. Every mark the interface draws is now one stroke set at
        # one weight, masked in the colour it lands in.
        emoji = re.compile("[\U0001F300-\U0001FAFF\u2190-\u27BF\uFE0F]")
        labelled = [
            block
            for block in self.demo.blocks.values()
            if isinstance(getattr(block, "value", None), str)
            and isinstance(block, gr.Button)
        ]
        self.assertTrue(labelled)
        for block in labelled:
            with self.subTest(label=block.value):
                self.assertIsNone(emoji.search(block.value))

    def test_an_icon_follows_the_colour_of_whatever_it_sits_in(self):
        # A mask is painted in the element's own colour, so one drawing
        # serves a quiet button, a primary one, dark mode and every theme.
        # An image would hold whatever colour it was exported at.
        self.assertIn(f".{icons.ICON_CLASS}::before", app.CSS)
        self.assertIn("background-color: currentColor;", app.CSS)
        for name in icons.ICONS:
            with self.subTest(icon=name):
                self.assertIn(icons.mask_rule(f".icon-{name}::before", name), app.CSS)

    def test_the_message_box_and_its_controls_are_one_composer(self):
        # The border belongs to the pair, so the row reads as part of the box
        # rather than as four loose buttons under it, and Send is moved to the
        # end of that row where the eye leaves the text it just typed.
        composer = self.by_id("composer")
        # Gradio puts a form of its own around a lone textbox, so the box is a
        # grandchild of the column it was written into.
        inside = [
            block
            for child in composer.children
            for block in (child, *getattr(child, "children", ()))
        ]
        self.assertIn(self.by_id("message-input"), inside)
        self.assertIn(self.by_id("chat-actions"), inside)
        self.assertIn("#composer {", app.CSS)
        self.assertIn("#chat-actions button.primary, #chat-actions #stop-button {", app.CSS)
        self.assertIn("order: 2; margin-left: auto;", app.CSS)

    def test_send_is_written_before_the_buttons_it_is_drawn_after(self):
        # Keyboard order is the written order, so Send stays first there; only
        # the drawing moves.
        actions = self.by_id("chat-actions")
        labels = [
            child.value for child in actions.children if isinstance(child, gr.Button)
        ]
        self.assertEqual(labels[0], "Send")
        self.assertEqual(labels[-3:], ["Retry", "Next token", "Undo last"])

    def test_the_conversation_being_read_is_tinted_rather_than_filled(self):
        # A filled block of the primary colour was the loudest thing in a pane
        # of two or three conversations, and said far more than "this is the
        # one you are in".
        rules = app.CSS[app.CSS.index("#conversation-list label.selected {") :]
        rules = rules[: rules.index("\n}")]

        self.assertIn("var(--primary-50)", rules)
        self.assertNotIn("var(--button-primary-background-fill)", rules)
        self.assertIn(
            "#conversation-list label.selected span {\n  color: var(--body-text-color)",
            app.CSS,
        )

    def test_a_model_wears_its_verdict_on_its_edge_not_across_its_name(self):
        # A row turned amber from end to end because its last word was
        # "tight", which read as a warning about the name rather than about
        # the memory.
        self.assertNotIn('.model-list label[data-testid*="· tight"] span', app.CSS)
        self.assertIn(
            '.model-list label[data-testid*="· tight"]::before,', app.CSS
        )
        self.assertIn("background: var(--fit-tight);", app.CSS)

    def test_a_reply_is_not_drawn_inside_a_box(self):
        # It arrived inside a bordered card holding a bordered reasoning box
        # holding the text: three edges deep for one answer. Space separates
        # the turns now, and your own message keeps the only bubble.
        self.assertIn("#conversation .message.bot {", app.CSS)
        self.assertIn("#conversation .message.user {", app.CSS)
        bot = app.CSS[app.CSS.index("#conversation .message.bot {") :]
        bot = bot[: bot.index("\n}")]
        self.assertIn("background: transparent !important;", bot)
        self.assertIn("border-color: transparent !important;", bot)

    def test_figures_meant_to_be_compared_are_set_in_one_digit_width(self):
        # Proportional digits are drawn at the width each digit wants, so a
        # column of token counts arrives ragged and a count ticking up during
        # a response jitters under the eye.
        self.assertIn("font-variant-numeric: tabular-nums;", app.CSS)
        for selector in ("#generation-status", "#conversation-list label span"):
            with self.subTest(selector=selector):
                rules = app.CSS[app.CSS.index(f"{selector} {{") :]
                self.assertIn("tabular-nums", rules[: rules.index("\n}")])

    def test_the_nav_names_are_on_screen_rather_than_a_hover_away(self):
        # Four pages is not a number worth hiding. Nothing clips the name
        # out of sight, and no tooltip stands in for it.
        self.assertNotIn("clip-path: inset(50%)", app.CSS)
        self.assertNotIn("#nav label::after", app.CSS)
        self.assertNotIn(":hover::after", app.CSS)

    def test_only_the_chat_page_starts_visible(self):
        self.assertTrue(self.by_id("chat-page").visible)
        self.assertTrue(self.by_id("conversation-pane").visible)
        self.assertFalse(self.by_id("images-page").visible)
        self.assertFalse(self.by_id("models-page").visible)
        self.assertFalse(self.by_id("settings-page").visible)

    def test_picking_a_page_shows_it_alone(self):
        (listener,) = listeners_named(self.demo, "show_page")
        self.assertEqual(listener.targets, [(self.by_id("nav")._id, "change")])
        self.assertEqual(
            listener.outputs,
            [
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("images-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )
        shown = lambda page: [update["visible"] for update in app.show_page(page)]
        # The conversations pane comes and goes with Chat.
        self.assertEqual(shown("Chat"), [True, True, False, False, False])
        self.assertEqual(shown("Images"), [False, False, True, False, False])
        self.assertEqual(shown("Models"), [False, False, False, True, False])
        self.assertEqual(shown("Settings"), [False, False, False, False, True])

    def follows(self, listener, name) -> bool:
        """Whether a handler called ``name`` runs, sooner or later, after ``listener``."""

        after: dict = {}
        for dependency in self.demo.config["dependencies"]:
            after.setdefault(dependency["trigger_after"], []).append(dependency["id"])
        pending, seen = [listener._id], set()
        while pending:
            for dependency_id in after.get(pending.pop(), []):
                if dependency_id in seen:
                    continue
                seen.add(dependency_id)
                if getattr(self.demo.fns[dependency_id].fn, "__name__", None) == name:
                    return True
                pending.append(dependency_id)
        return False

    def cancelled_by(self, trigger) -> set:
        """Event indices cancelled by anything bound to ``trigger``.

        Gradio records a listener's ``cancels`` against the target rather
        than the handler, so this reads every function on that target.
        """

        return {
            index
            for fn in self.demo.fns.values()
            if fn.targets == [trigger]
            for index in fn.cancels
        }

    def test_the_badge_sits_above_the_chat_page_tabs(self):
        chat_page = self.by_id("chat-page")
        badge = self.by_id("model-badge")
        switch = self.by_id("model-switch")
        self.assertTrue(self.within(badge, chat_page))
        self.assertTrue(self.within(switch, chat_page))
        self.assertIsInstance(switch, gr.Dropdown)
        # Above the tabs, so Score text names the model as well as Chat.
        tabs = next(
            block for block in self.demo.blocks.values() if isinstance(block, gr.Tabs)
        )
        self.assertFalse(self.within(badge, tabs))

    def test_every_change_to_what_is_in_memory_repaints_the_badge(self):
        # Download-and-load, load cached and unload change what is in memory;
        # the page load draws the badge first, switching pages catches a load
        # that started while the chat page was out of sight, and the timer
        # catches one another tab started.
        # The four that change memory (the switcher included), the download
        # that only changes what is on disk, redownload and a confirmed
        # removal, plus the page load, the nav and the timer.
        self.assertEqual(len(listeners_named(self.demo, "refresh_model_badge")), 10)

    def test_the_timer_also_un_sticks_the_scored_token_count(self):
        # A count asked for during a reply gives up and says so, and that
        # message does not correct itself when the reply ends. Rather than
        # ask every path out of a generation to remember, the badge's timer
        # carries the recovery - guarded so the ordinary tick costs nothing.
        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        (recovery,) = listeners_named(self.demo, "recover_score_budget")

        self.assertEqual(recovery.targets, [(timers[0]._id, "tick")])
        self.assertEqual(recovery.inputs[0], self.by_id("score-budget"))
        self.assertEqual(recovery.outputs[0], self.by_id("score-budget"))
        # The count and the load it was counted against travel together.
        self.assertIsInstance(recovery.inputs[1], gr.State)
        self.assertEqual(recovery.outputs, recovery.inputs[:2])

    def test_everything_that_writes_the_count_shares_one_queue(self):
        # Gradio's concurrency limit is per event, not across events, so
        # without a shared id the timer's recovery and a keystroke's count
        # can overlap - and they contend for the same model lock, so one of
        # them loses it and publishes the "not mid-response" message. The
        # loser finishing last would leave a count that does not describe the
        # box, which is the one thing this line exists to rule out.
        budget = self.by_id("score-budget")
        writers = [fn for fn in self.demo.fns.values() if budget in fn.outputs]

        self.assertGreater(len(writers), 1)
        self.assertEqual(
            {fn.concurrency_id for fn in writers}, {app.SCORE_BUDGET_QUEUE}
        )

    def test_the_badge_asks_again_on_a_timer(self):
        # The manager is one object for the whole process, but a handler's
        # updates only reach the tab that ran it. Without the timer a second
        # tab would name a model that another tab has since swapped out.
        timers = [
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Timer) and block.value == app.BADGE_REFRESH_SECONDS
        ]
        self.assertEqual([timer.value for timer in timers], [app.BADGE_REFRESH_SECONDS])
        self.assertLessEqual(app.BADGE_REFRESH_SECONDS, 5)
        ticks = [
            listener
            for listener in listeners_named(self.demo, "refresh_model_badge")
            if listener.targets == [(timers[0]._id, "tick")]
        ]
        self.assertEqual(len(ticks), 1)
        # Nobody asked for this one, so it does not put a pending shimmer on
        # the badge every couple of seconds.
        self.assertEqual(ticks[0].show_progress, "hidden")

    def test_no_timer_fades_the_text_it_repaints(self):
        # Gradio dims every output of a running event to a fifth of its
        # opacity and brings it back when the event lands, and
        # show_progress="hidden" only takes away the spinner, not the fade.
        # Which components are dimmed is show_progress_on, and it defaults to
        # all of them, so a handler on a timer fades its outputs in and out on
        # every tick - four times a second, on the conversation pane, which
        # reads as the status line and the token inspector blinking. An empty
        # list dims nothing, which is what a repaint nobody asked for should do.
        timers = {
            block._id
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Timer)
        }
        ticks = [
            fn
            for fn in self.demo.fns.values()
            if any(target in timers for target, event in fn.targets if event == "tick")
        ]
        self.assertTrue(ticks)
        for fn in ticks:
            with self.subTest(handler=getattr(fn.fn, "__name__", fn)):
                self.assertEqual(fn.show_progress_on, [])

    def test_the_inspector_does_not_blink_while_a_reply_streams(self):
        # The same fade, from the other direction. Streaming writes the
        # response metrics on every frame, and the handler that empties the
        # inspector rides that state, so during a reply it fires several times
        # a second. It skips its outputs once there is nothing left to clear,
        # but Gradio marks them pending either way, which puts a spinner and a
        # queue counter over the panel that is already saying to wait.
        resets = listeners_named(self.demo, "reset_inspection")
        self.assertTrue(resets)
        for fn in resets:
            with self.subTest(handler=fn):
                self.assertEqual(fn.show_progress, "hidden")
                self.assertEqual(fn.show_progress_on, [])

    def test_the_badge_buttons_send_the_nav_to_the_models_page(self):
        # One on Images: its badge says a model it can use is missing, and
        # offers the way to load one. The Chat page's badge has the switcher
        # and the default-model button instead.
        panes = [
            self.by_id("nav"),
            self.by_id("conversation-pane"),
            self.by_id("chat-page"),
            self.by_id("images-page"),
            self.by_id("models-page"),
            self.by_id("settings-page"),
        ]
        (listener,) = listeners_named(self.demo, "go_to_image_models")
        ((block_id, event),) = listener.targets
        self.assertEqual(event, "click")
        self.assertEqual(self.demo.blocks[block_id].elem_id, "image-load-model")
        # The button switches the pages itself: a Radio set by a handler
        # reports no change, so the nav's own handler would not run.
        self.assertEqual(listener.outputs[: len(panes)], panes)

        page, *updates = app.go_to_models()
        self.assertEqual(page, "Models")
        self.assertEqual(
            [update["visible"] for update in updates],
            [False, False, False, True, False],
        )

    def test_the_images_button_scopes_both_model_lists_to_image_models(self):
        # The list it lands on is the whole cache, where the kind is a word
        # mid-row and the majority of rows do not carry it at all. Both kind
        # controls are set on the way in, and both lists repainted from them.
        (listener,) = listeners_named(self.demo, "go_to_image_models")
        kind_filter = self.labelled("Kind")
        search_kind = self.by_id("search-kind")
        self.assertIn(kind_filter, listener.inputs)
        self.assertIn(search_kind, listener.inputs)
        self.assertIn(kind_filter, listener.outputs)
        self.assertIn(search_kind, listener.outputs)
        # A name left in the filter box would hide the image models the
        # button is there to show, so the box is cleared on the way in.
        self.assertIn(self.by_id("my-models-filter"), listener.outputs)
        self.assertIn(self.labelled("Downloaded models"), listener.outputs)
        self.assertIn(self.by_id("model-search-results"), listener.outputs)

    def test_every_model_change_rescans_the_cache(self):
        # Download, download-and-load, load cached, unload, redownload,
        # confirmed removal, the refresh button, a new sort order, a new kind
        # filter, a typed name, a new weight precision, the page load and a
        # pick in the chat page's switcher each rescan. Selecting the default
        # only navigates, and the Images page's button rescans inside
        # go_to_image_models.
        self.assertEqual(len(listeners_named(self.demo, "refresh_my_models")), 13)

    def test_model_actions_follow_selections_and_cache_refreshes(self):
        listeners = listeners_named(self.demo, "refresh_model_actions")
        radio = self.labelled("Downloaded models")
        model_id = self.labelled("Hugging Face model ID")
        token = self.labelled("Hugging Face token (optional)")
        for control in (radio, model_id, token):
            self.assertTrue(any(fn.targets == [(control._id, "change")] for fn in listeners))
        for fn in listeners:
            self.assertEqual(fn.inputs[:2], [model_id, radio])
            self.assertIsInstance(fn.inputs[2], gr.State)
            self.assertEqual(fn.inputs[3:], [token])
            self.assertEqual(fn.outputs[0], self.by_id("model-availability"))
            self.assertEqual(
                [button.value for button in fn.outputs[1:]],
                ["Download and load", "Download only", "Load cached"],
            )
        refresh_ids = {fn._id for fn in listeners_named(self.demo, "refresh_my_models")}
        chained = [
            dependency for dependency in self.demo.config["dependencies"]
            if dependency["id"] in {fn._id for fn in listeners}
            and dependency["trigger_after"] in refresh_ids
        ]
        # All seven mutations (the switcher included), manual refresh, and
        # startup refresh the controls even when the radio's selected value
        # stays the same.
        self.assertEqual(len(chained), 9)

    def test_the_timer_refreshes_the_actions_through_the_gated_handler(self):
        # The timer ticks in every open session for the life of the app, so
        # it goes through the stamped refresh rather than the scanning one.
        (fn,) = listeners_named(self.demo, "refresh_stale_model_actions")
        ((block_id, event),) = fn.targets
        self.assertEqual(event, "tick")
        self.assertIsInstance(self.demo.blocks[block_id], gr.Timer)
        self.assertEqual(fn.inputs[:2], [
            self.labelled("Hugging Face model ID"), self.labelled("Downloaded models")
        ])
        self.assertIsInstance(fn.inputs[-1], gr.State)
        self.assertIs(fn.inputs[-1], fn.outputs[-1])
        self.assertEqual(fn.outputs[0], self.by_id("model-availability"))
        self.assertEqual(
            [button.value for button in fn.outputs[1:-1]],
            ["Download and load", "Download only", "Load cached"],
        )
        self.assertFalse(any(
            event == "tick" for other in listeners_named(self.demo, "refresh_model_actions")
            for _, event in other.targets
        ))

    def test_model_progress_is_always_open_in_the_card_above_download_buttons(self):
        status = self.by_id("model-status")
        card = self.by_id("model-availability").parent
        self.assertTrue(self.within(status, card))
        parent = status.parent
        while parent is not card:
            self.assertNotIsInstance(parent, gr.Accordion)
            parent = parent.parent
        activity = status.parent
        for name in ("download_model", "download_and_load_model", "load_cached_model"):
            listener = listeners_named(self.demo, name)[0]
            button = self.demo.blocks[listener.targets[0][0]]
            self.assertTrue(self.within(button, card))
            self.assertLess(card.children.index(activity), card.children.index(button.parent))

    def test_explicit_repository_checks_make_one_request_after_leaving_the_id_field(self):
        model_id = self.labelled("Hugging Face model ID")
        (button,) = [
            block for block in self.demo.blocks.values()
            if isinstance(block, gr.Button) and block.value == "Check model"
        ]
        checks = listeners_named(self.demo, "check_model_repository")
        info = mock.Mock(
            tags=[], config={}, library_name="transformers", siblings=[],
            private=False, gated=False,
        )
        for events, expected in (
            ([(model_id._id, "blur"), (button._id, "click")], 1),
            ([(model_id._id, "submit"), (model_id._id, "blur")], 1),
            ([(model_id._id, "blur")], 0),
        ):
            with self.subTest(events=events), mock.patch(
                "huggingface_hub.HfApi.model_info", return_value=info
            ) as request:
                # Dispatch the actual registered dependencies in browser event
                # order: clicking Check first blurs the focused model ID field.
                for event in events:
                    for listener in checks:
                        if event in listener.targets:
                            states = list(listener.fn("org/model", ""))
                            self.assertEqual(states[-1]["status"], "found")
                self.assertEqual(request.call_count, expected)

    def test_repository_precision_refreshes_for_cached_selections_and_rescans(self):
        views = listeners_named(self.demo, "repository_view")
        selected = self.labelled("Downloaded models")
        self.assertTrue(any(fn.targets == [(selected._id, "change")] for fn in views))
        self.assertTrue(any(event == "load" for fn in views for _, event in fn.targets))
        for fn in views:
            self.assertEqual(fn.inputs[-1], selected)
            self.assertEqual(fn.inputs[0], self.labelled("Hugging Face model ID"))
            self.assertEqual(fn.inputs[2], self.labelled("Hugging Face token (optional)"))
        action_ids = {fn._id for fn in listeners_named(self.demo, "refresh_model_actions")}
        chained = [
            dependency for dependency in self.demo.config["dependencies"]
            if dependency["id"] in {fn._id for fn in views}
            and dependency["trigger_after"] in action_ids
        ]
        self.assertEqual(len(chained), 9)

    def test_every_load_reads_the_my_models_selection(self):
        # The ID box lags a row selection by a server round trip, so a button
        # clicked in that window would act on the box's previous contents -
        # the 15 GB default. Each load takes the radio as well and prefers it.
        radio = self.labelled("Downloaded models")
        for name in ("load_cached_model", "download_model", "download_and_load_model"):
            (fn,) = listeners_named(self.demo, name)
            self.assertIn(radio, fn.inputs, name)

    def test_download_then_load_keeps_the_typed_model_when_another_model_is_loaded(self):
        manager = ModelManager()
        manager.model_id = OLMO
        entries = [cached(OLMO)]
        typed_id = "org/new-model"
        (download,) = listeners_named(self.demo, "download_model")
        (load,) = listeners_named(self.demo, "load_cached_model")
        dependency = next(
            item for item in self.demo.config["dependencies"]
            if item["trigger_after"] == download._id
        )
        refresh = self.demo.fns[dependency["id"]]
        self.assertEqual(refresh.fn, models_page.refresh_my_models)
        self.assertEqual(refresh.inputs[-1], self.labelled("Hugging Face model ID"))

        def fetch(model_id, token):
            entries.append(cached(model_id))
            yield "download progress"
            return Path("/cache/new-model")

        def read_weights(*args):
            yield "load progress"
            return "CPU"

        with (
            mock.patch.object(runtime, "MANAGER", manager),
            mock.patch.object(models_page, "list_cached_models", side_effect=lambda: list(entries)),
            mock.patch.object(models_page, "cache_status", side_effect=lambda model_id: next((entry.status for entry in entries if entry.model_id == model_id), CacheStatus())),
            mock.patch.object(models_page, "stream_download", side_effect=fetch),
            mock.patch.object(manager, "find_cached", return_value=Path("/cache/new-model")),
            mock.patch.object(models_page, "stream_load", side_effect=read_weights) as stream_load,
        ):
            selected = models_page.clear_my_model_selection()[0]["value"]
            cards = list(download.fn(typed_id, "", selected))
            self.assertIn("Download complete", cards[-1])
            self.assertEqual(manager.model_id, OLMO)
            radio, _, _ = refresh.fn(selected, "Name", model_id=typed_id)
            detail, _, _, load_button = models_page.refresh_model_actions(typed_id, radio["value"])
            self.assertEqual(radio["value"], typed_id)
            self.assertIn("Ready to load", detail)
            self.assertTrue(load_button["visible"])
            list(load.fn(typed_id, radio["value"]))
            self.assertEqual(stream_load.call_args.args[0], typed_id)

    def test_a_picked_row_outranks_the_id_box(self):
        # A click's inputs are snapshotted in the browser, and a row reaches
        # the box only through a server round trip, so the box a button
        # carries can still hold the 15 GB default while the radio is
        # current. The radio therefore wins whenever there is one.
        self.assertEqual(
            app.chosen_model(settings.DEFAULT_MODEL_ID, "org/picked"), "org/picked"
        )
        self.assertEqual(app.chosen_model("", "org/picked"), "org/picked")
        # With no row picked the typed ID is all there is.
        self.assertEqual(app.chosen_model("  org/typed  ", None), "org/typed")
        self.assertEqual(app.chosen_model("", None), "")

    def test_naming_a_model_another_way_withdraws_the_selection(self):
        # Typing an ID or picking a search result names its own model, so the
        # highlighted row cannot outrank it.
        listeners = listeners_named(self.demo, "clear_my_model_selection")
        self.assertEqual(len(listeners), 2)
        radio = self.labelled("Downloaded models")
        for fn in listeners:
            self.assertIn(radio, fn.outputs)

    def test_removal_asks_before_deleting(self):
        # The Remove button only opens the question; deleting is the
        # confirm button's job. Cancelling withdraws it, and so does naming
        # another model, whether by choosing a row or by typing an ID.
        (ask,) = listeners_named(self.demo, "ask_remove_my_model")
        (remove,) = listeners_named(self.demo, "remove_my_model")
        buttons = {
            self.demo.blocks[block_id].value: fn
            for fn in (ask, remove)
            for block_id, _ in fn.targets
        }
        self.assertIs(buttons["Remove"], ask)
        self.assertIs(buttons["Remove from disk"], remove)
        self.assertEqual(len(listeners_named(self.demo, "hide_remove_confirm")), 3)

    def test_the_confirm_button_deletes_the_model_the_question_named(self):
        # The confirm handler reads the stored pending ID, not the radio, so
        # a selection moved after the question opened cannot redirect it.
        (ask,) = listeners_named(self.demo, "ask_remove_my_model")
        (remove,) = listeners_named(self.demo, "remove_my_model")
        radio = self.labelled("Downloaded models")
        (pending,) = remove.inputs
        self.assertIsInstance(pending, gr.State)
        self.assertIsNot(pending, radio)
        self.assertIn(pending, ask.outputs)
        self.assertIn(pending, remove.outputs)

    def test_clear_asks_before_it_takes_every_conversation(self):
        # Clear reaches past the conversation on screen: it deletes every
        # other one too, and nothing brings them back. The button only opens
        # the question; the confirm button clears once a running job stops.
        (ask,) = listeners_named(self.demo, "ask_clear_chat")
        (clear,) = listeners_named(self.demo, "clear_chat")
        cancel = next(
            fn
            for fn in listeners_named(self.demo, "hide_clear_confirm")
            if self.demo.blocks[fn.targets[0][0]].value == "Cancel"
        )
        buttons = {
            self.demo.blocks[block_id].value: fn
            for fn in (ask, clear, cancel)
            for block_id, _ in fn.targets
        }
        self.assertIs(buttons["Clear all"], ask)
        self.assertIs(buttons["Clear everything"], clear)
        self.assertIs(buttons["Cancel"], cancel)
        # Cancelling is recorded against the target rather than the handler,
        # so it is read the way ClearCancelsGenerationTests reads it.
        self.assertFalse(self.cancelled_by(ask.targets[0]))
        self.assertFalse(self.cancelled_by(clear.targets[0]))

    def test_changing_the_conversations_withdraws_the_clear_question(self):
        # The question names how many conversations it would take, counted
        # when it was asked. Left open across a New or a Fork it would
        # promise less than "Clear everything" would take - and that promise
        # is the whole reason the question exists.
        withdrawals = listeners_named(self.demo, "hide_clear_confirm")
        triggered_by = {fn.targets[0][0] for fn in withdrawals}
        buttons = {
            self.demo.blocks[block_id].value
            for block_id in triggered_by
            if isinstance(self.demo.blocks[block_id], gr.Button)
        }

        self.assertEqual(buttons, {"Cancel", "New", "Fork", "Delete"})
        # Switching conversations counts too, and it is the list itself.
        self.assertIn(self.by_id("conversation-list")._id, triggered_by)
        for fn in withdrawals:
            self.assertEqual(fn.outputs, [self.by_id("clear-confirm")])

    def test_the_clear_button_is_named_for_everything_it_takes(self):
        # "Clear" alone reads as emptying the chat on screen, which is what
        # Delete does. This one takes the lot.
        (ask,) = listeners_named(self.demo, "ask_clear_chat")
        ((block_id, _),) = ask.targets

        self.assertEqual(self.demo.blocks[block_id].value, "Clear all")

    def test_clear_stands_under_the_list_of_what_it_takes(self):
        # Under the message box it sat among Retry, Undo and Send, all of
        # which act on the one conversation on screen, and read as another
        # of them. It takes every conversation, so it belongs under the list
        # of them, beside New, Fork and Delete.
        (ask,) = listeners_named(self.demo, "ask_clear_chat")
        ((block_id, _),) = ask.targets
        pane = self.by_id("conversation-pane")

        self.assertTrue(self.within(self.demo.blocks[block_id], pane))
        self.assertTrue(self.within(self.by_id("clear-confirm"), pane))

    def test_the_offer_sits_beside_the_badge_that_says_it_is_needed(self):
        # The badge names the missing model; the offer is what to do about
        # it, and both belong where the reader already is.
        offer = self.by_id("default-model")

        self.assertTrue(self.within(offer, self.by_id("chat-page")))
        self.assertTrue(self.within(offer, self.by_id("model-bar")))
        (setup,) = listeners_named(self.demo, "select_default_model")
        self.assertEqual(setup.targets, [(offer._id, "click")])
        self.assertEqual(offer.value, "Set up the default model")
        # No chained handler may turn this navigation back into automatic I/O.
        self.assertFalse(any(
            dependency["trigger_after"] == setup._id
            for dependency in self.demo.config["dependencies"]
        ))
        # It switches the pages itself, for the reason go_to_models gives: a
        # Radio set by a handler reports no change, so setting the nav alone
        # would tick Models and leave the chat page on screen.
        self.assertEqual(
            setup.outputs,
            [
                self.labelled("Hugging Face model ID"),
                # A row picked earlier outranks the ID box, so it goes.
                self.labelled("Downloaded models"),
                self.by_id("my-model-detail"),
                # The search selection is a State beside the table, so it is
                # found through the handler that writes it.
                listeners_named(self.demo, "select_search_result")[0].outputs[2],
                listeners_named(self.demo, "select_search_result")[0].outputs[1],
                self.by_id("model-status"),
                listeners_named(self.demo, "hide_remove_confirm")[0].outputs[0],
                listeners_named(self.demo, "hide_remove_confirm")[0].outputs[1],
                self.by_id("nav"),
                self.by_id("conversation-pane"),
                self.by_id("chat-page"),
                self.by_id("images-page"),
                self.by_id("models-page"),
                self.by_id("settings-page"),
            ],
        )

    def test_the_offer_is_published_wherever_the_badge_is(self):
        # Setup links share the badge's visibility decision in every tab.
        listeners = listeners_named(self.demo, "refresh_model_badge")
        self.assertTrue(listeners)
        for listener in listeners:
            self.assertEqual(
                listener.outputs,
                [self.by_id("model-badge"), self.by_id("default-model")],
            )

    def test_the_switcher_is_drawn_when_the_badge_is_and_after_every_rescan(self):
        # Arriving, opening the page, and the timer - which asks first whether
        # the switcher still names what is in memory, so an open list is not
        # closed under the reader every couple of seconds.
        switch = self.by_id("model-switch")
        precision = self.labelled("Weight precision")
        listeners = listeners_named(self.demo, "refresh_model_switch")
        triggers = {listener.targets[0] for listener in listeners}
        self.assertIn((self.by_id("nav")._id, "change"), triggers)
        self.assertIn((self.demo._id, "load"), triggers)
        # Every draw hands back the cache revision it read, kept per tab, so
        # the timer can tell an idle list from one another tab left stale.
        revision = listeners[0].outputs[1]
        self.assertIsInstance(revision, gr.State)
        for listener in listeners:
            self.assertEqual(listener.inputs, [precision])
            self.assertEqual(listener.outputs, [switch, revision])
        # Every load, unload and download repaints it: the cache and memory
        # are what it offers.
        for name in ("load_cached_model", "download_and_load_model", "unload_model",
                     "download_model", "switch_model"):
            with self.subTest(handler=name):
                action = listeners_named(self.demo, name)[0]
                self.assertTrue(self.follows(action, "refresh_model_switch"))

        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        ticks = listeners_named(self.demo, "refresh_stale_model_switch")
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].targets, [(timers[0]._id, "tick")])
        self.assertEqual(ticks[0].inputs, [switch, revision, precision])
        self.assertEqual(ticks[0].outputs, [switch, revision])
        self.assertEqual(ticks[0].show_progress, "hidden")

    def test_a_pick_in_the_switcher_loads_at_the_chosen_precision(self):
        switch = self.by_id("model-switch")
        listeners = listeners_named(self.demo, "switch_model")
        self.assertEqual(len(listeners), 1)
        self.assertEqual(listeners[0].targets, [(switch._id, "input")])
        self.assertEqual(listeners[0].inputs, [switch, self.labelled("Weight precision")])
        self.assertEqual(
            listeners[0].outputs,
            [switch, self.by_id("model-status"), self.by_id("model-badge")],
        )
        # And is followed by the same rescan as Load cached: the badge, the
        # token count and the hardware panel all change with the model.
        for name in ("refresh_my_models", "refresh_model_badge", "refresh_hardware"):
            with self.subTest(handler=name):
                self.assertTrue(self.follows(listeners[0], name))

    def test_the_images_badge_is_refreshed_on_the_same_three_occasions(self):
        # Arriving at the page, opening it, and the timer that tells a tab
        # which did not start a load about it.
        listeners = listeners_named(self.demo, "refresh_image_badge")
        outputs = [self.by_id("image-model-badge"), self.by_id("image-load-model")]
        for listener in listeners:
            self.assertEqual(listener.outputs, outputs)
        self.assertEqual(len(listeners), 3)

        triggers = {listener.targets[0] for listener in listeners}
        timers = [
            block for block in self.demo.blocks.values() if isinstance(block, gr.Timer)
        ]
        self.assertIn((self.by_id("nav")._id, "change"), triggers)
        self.assertIn((timers[0]._id, "tick"), triggers)
        # The timer's own refresh does not put a pending shimmer on the badge
        # every couple of seconds; nobody asked it anything.
        ticks = [
            listener
            for listener in listeners
            if listener.targets == [(timers[0]._id, "tick")]
        ]
        self.assertEqual([listener.show_progress for listener in ticks], ["hidden"])

    def test_stop_drawing_does_not_cancel_the_generator_that_publishes_the_run(self):
        # The pipeline runs on its own thread and would keep running with the
        # generator gone, taking every recorded step with it. So Stop sets an
        # event the run checks between steps, and the generator itself
        # publishes the stopped run.
        (stop,) = [listener for listener in listeners_named(self.demo, "stop_drawing")
                   if listener.targets == [(self.by_id("stop-drawing")._id, "click")]]
        ((block_id, event),) = stop.targets

        self.assertEqual(event, "click")
        self.assertEqual(self.demo.blocks[block_id].elem_id, "stop-drawing")
        self.assertEqual(stop.cancels, [])
        (draw,) = listeners_named(self.demo, "draw")
        self.assertNotIn(draw._id, self.cancelled_by((block_id, event)))

    def test_moving_the_step_repaints_the_frame_the_shading_and_the_map(self):
        # Attention moves between steps as much as the picture does, so these
        # cannot be allowed to disagree about which step is on screen.
        (select,) = listeners_named(self.demo, "select_step")

        self.assertEqual(select.targets, [(self.by_id("image-step")._id, "release")])
        self.assertEqual(
            select.outputs,
            [
                self.by_id("image-trajectory"),
                self.by_id("image-prompt-strip"),
                self.by_id("image-attention-note"),
                self.by_id("image-attention"),
            ],
        )
        self.assertIs(select.inputs[1], self.by_id("image-step"))

    def test_clicking_a_prompt_token_is_remembered_before_the_map_is_drawn(self):
        # The click's index has to land in the state the map reads, so the
        # map follows the step slider afterwards without another click.
        (remember,) = [listener for listener in listeners_named(self.demo, "remember_token")
                       if listener.targets == [(self.by_id("image-prompt-strip")._id, "select")]]
        (token_state,) = remember.outputs
        (paint,) = listeners_named(self.demo, "select_token")

        self.assertEqual(
            remember.targets, [(self.by_id("image-prompt-strip")._id, "select")]
        )
        self.assertEqual(paint.inputs[1], token_state)
        self.assertEqual(paint.outputs, [self.by_id("image-attention")])
        self.assertEqual(paint.trigger_after, remember._id)
    def test_the_prompt_upload_takes_every_file_the_parser_reads(self):
        # The parser reads anything that is not JSON as blank-line separated
        # text, and the README says so, so a filter that only offered .txt
        # would hide the .md and extensionless prompt sets it handles.
        upload = next(
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == "Load prompts"
        )

        self.assertIn("text", upload.file_types)
        self.assertIn(".json", upload.file_types)
        self.assertIn(".jsonl", upload.file_types)

    def test_escape_is_wired_to_the_stop_button_by_its_id(self):
        # The shortcut presses the button rather than reaching past it, so
        # whatever Stop does, Escape does. It needs the id to find it.
        stops = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "value", None) in ("Stop", common.STOP_LABEL)
        ]

        # One stops a reply, one a batch of prompts, one a comparison slot
        # being filled, one an activation experiment, one a picture being drawn.
        # No more than one can be in the page: they contend for the same
        # generation slot and the losers refuse.
        self.assertEqual(
            {stop.elem_id for stop in stops},
            {"stop-button", "stop-batch-button", "stop-compare", "stop-patching", "stop-drawing"},
        )
        self.assertIn("#stop-button", app.SHORTCUT_JS)
        self.assertIn("#stop-batch-button", app.SHORTCUT_JS)
        self.assertIn("#stop-compare", app.SHORTCUT_JS)
        self.assertIn("#stop-patching", app.SHORTCUT_JS)
        self.assertIn("#stop-drawing", app.SHORTCUT_JS)
        # Whether the button is in the document is the whole test. Gradio
        # leaves a component whose visible is false out of the page, so its
        # presence is the generation state itself. Testing whether it can be
        # *seen* would drop the key on the Score text tab, where the button
        # is still in the page with a hidden ancestor - the moment a reader
        # is most likely to reach for it, being away from the button.
        self.assertNotIn("offsetParent", app.SHORTCUT_JS)
        self.assertNotIn("offsetWidth", app.SHORTCUT_JS)
        self.assertNotIn("getBoundingClientRect", app.SHORTCUT_JS)
        self.assertNotIn("checkVisibility", app.SHORTCUT_JS)
        self.assertTrue(
            any(fn.js == app.SHORTCUT_JS for fn in self.demo.fns.values()),
            "nothing attaches the keyboard shortcut on load",
        )

    def test_the_sampling_accordion_starts_showing_the_saved_values(self):
        # The summary is only worth having if it is right before anything is
        # touched, which means the label and the sliders read one set of
        # numbers - and that set is the saved settings, not a second copy of
        # the defaults that could drift from them.
        accordion = next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Accordion)
            and (block.label or "").startswith("Sampling")
        )
        saved = settings.load()

        self.assertEqual(
            accordion.label,
            app.sampling_label(
                saved.temperature,
                saved.top_p,
                saved.top_k,
                saved.skip_top_below,
                saved.max_new_tokens,
            ),
        )
        for label, value in [
            ("Temperature", saved.temperature),
            ("Top-p", saved.top_p),
            ("Top-k (0 disables)", saved.top_k),
            ("Skip top choice below (0 disables)", saved.skip_top_below),
            ("Maximum new tokens", saved.max_new_tokens),
        ]:
            with self.subTest(control=label):
                self.assertEqual(self.labelled(label).value, value)
                self.assertTrue(self.within(self.labelled(label), accordion))
        # The response length cannot outrun the context limit.
        self.assertEqual(
            self.labelled("Maximum new tokens").maximum, saved.prefill_token_limit
        )

    def test_the_sampling_summary_follows_every_slider(self):
        # A slider fires continuously while it is dragged; the label only has
        # to be right once it is let go.
        sliders = [
            self.labelled(label)
            for label in (
                "Temperature",
                "Top-p",
                "Top-k (0 disables)",
                "Skip top choice below (0 disables)",
                "Maximum new tokens",
            )
        ]
        listeners = listeners_named(self.demo, "update_sampling_label")
        # A page load's target has no block, so look the ids up by hand.
        by_id = {slider._id: slider for slider in sliders}
        moved = [fn for fn in listeners if fn.targets[0][0] in by_id]

        self.assertEqual([by_id[fn.targets[0][0]] for fn in moved], sliders)
        for fn in listeners:
            self.assertEqual(fn.inputs, sliders)
        # The others are the paths that move a slider without a hand on it:
        # the settings file read back on load, the context limit committed,
        # which can pull the response length down with it, the five resets,
        # and the six that change which conversation is on screen - forking,
        # starting one, switching, deleting, clearing, and the page load that
        # brings the saved conversations back - each of which brings that
        # conversation's own sampling onto the sliders.
        self.assertEqual(len(listeners) - len(moved), 14)

    def test_everything_that_writes_the_summary_shares_one_queue(self):
        # always_last coalesces each slider's own requests; across four
        # listeners Gradio orders nothing. Each handler reads all four values
        # as they were when its request was sent, so two sliders moved in
        # quick succession can finish out of order and leave the label
        # describing the older pair - and only the next change rewrites it.
        accordion = next(
            block
            for block in self.demo.blocks.values()
            if isinstance(block, gr.Accordion)
            and (block.label or "").startswith("Sampling")
        )
        writers = [fn for fn in self.demo.fns.values() if accordion in fn.outputs]

        self.assertGreater(len(writers), 1)
        self.assertEqual(
            {fn.concurrency_id for fn in writers}, {app.SAMPLING_LABEL_QUEUE}
        )
        # And it is its own queue, not shared with the token count, which
        # costs an encoding and would make the label wait behind it.
        self.assertNotEqual(app.SAMPLING_LABEL_QUEUE, app.SCORE_BUDGET_QUEUE)

    def test_the_sampling_summary_follows_a_slider_moved_by_keyboard(self):
        # Gradio dispatches release from pointerup alone, so a slider moved
        # with the arrow keys - which is how it is moved without a mouse -
        # changes its value and never reports a release. Listening for
        # release would leave the summary describing the old settings for
        # anyone not using a pointer.
        sliders = {
            self.labelled(label)._id
            for label in (
                "Temperature",
                "Top-p",
                "Top-k (0 disables)",
                "Skip top choice below (0 disables)",
                "Maximum new tokens",
            )
        }

        for fn in listeners_named(self.demo, "update_sampling_label"):
            block_id, event = fn.targets[0]
            if block_id in sliders:
                with self.subTest(slider=self.demo.blocks[block_id].label):
                    self.assertEqual(event, "change")
                    # A drag fires change on every step, so they coalesce.
                    self.assertEqual(fn.trigger_mode, "always_last")

    def test_the_scored_token_count_follows_every_box_that_feeds_it(self):
        # The count has to match what would actually be scored, so a change
        # to the context or the chat-template box moves it too.
        boxes = [
            self.labelled("Context (optional)"),
            self.labelled("Text to score"),
            self.labelled("Treat the context as a chat message"),
        ]
        listeners = listeners_named(self.demo, "score_token_count")
        typed = [fn for fn in listeners if fn.trigger_mode == "always_last"]

        self.assertEqual([self.demo.blocks[fn.targets[0][0]] for fn in typed], boxes)
        for fn in listeners:
            self.assertEqual(fn.inputs, boxes)
        # A different tokenizer counts the same passage differently and a
        # different model has its own limit, so every handler that changes
        # what is loaded recomputes the count rather than leaving the old
        # model's answer under the box.
        # Six of the rescans change neither: the refresh button, a new sort
        # order, a new kind filter, a typed name, a new weight precision, and
        # the page load.
        self.assertEqual(
            len(listeners) - len(typed), len(listeners_named(self.demo, "refresh_my_models")) - 6
        )

    def test_choosing_a_model_writes_the_id_box(self):
        box = self.labelled("Hugging Face model ID")
        for name in ("select_my_model", "select_search_result"):
            with self.subTest(handler=name):
                (listener,) = listeners_named(self.demo, name)
                self.assertIs(listener.outputs[0], box)


# Enough of a page for RESIZE_JS to run against: two rows of the shape the
# layout builds, and stand-ins for the browser it talks to. Nothing here
# lays anything out, so every element is told its own width, and the frames
# and the mutations are delivered by the checks rather than by a clock.
RESIZE_PAGE = """
'use strict';
const assert = require('node:assert');

class Style {
  constructor() { this.props = {}; }
  setProperty(name, value) { this.props[name] = value; }
  removeProperty(name) { delete this.props[name]; }
}

class Element {
  constructor(id, width) {
    this.id = id || '';
    this.width = width || 0;
    this.children = [];
    this.parentElement = null;
    this.dataset = {};
    this.style = new Style();
    this.names = new Set();
    this.attributes = {};
    this.pointer = null;
    this.classList = {
      add: (name) => this.names.add(name),
      remove: (name) => this.names.delete(name),
      contains: (name) => this.names.has(name),
    };
  }
  get clientWidth() { return this.width; }
  getBoundingClientRect() { return { width: this.width }; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }
  append(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  remove() {
    const siblings = this.parentElement.children;
    siblings.splice(siblings.indexOf(this), 1);
    this.parentElement = null;
  }
  closest(selector) {
    if (selector === '.pane-resizer' && this.names.has('pane-resizer')) { return this; }
    return this.parentElement ? this.parentElement.closest(selector) : null;
  }
  setPointerCapture(pointer) { this.pointer = pointer; }
  releasePointerCapture(pointer) {
    if (this.pointer === pointer) { this.pointer = null; }
  }
  hasPointerCapture(pointer) { return this.pointer === pointer; }
}

const find = (node, id) => {
  if (node.id === id) { return node; }
  for (const child of node.children) {
    const found = find(child, id);
    if (found) { return found; }
  }
  return null;
};

// The one selector the script asks the document for.
const gather = (node, name, found) => {
  if (node.names.has(name)) { found.push(node); }
  for (const child of node.children) { gather(child, name, found); }
  return found;
};

let watchers = [];
class MutationObserver {
  constructor(react) { this.react = react; }
  observe() { watchers.push(this.react); }
  disconnect() { watchers = watchers.filter((react) => react !== this.react); }
}
const mutated = () => { for (const react of watchers.slice()) { react(); } };

const resizers = [];
class ResizeObserver {
  constructor(react) {
    this.react = react;
    this.targets = new Set();
    resizers.push(this);
  }
  observe(target) {
    if (this.targets.has(target)) { return; }
    this.targets.add(target);
    this.react();
  }
  unobserve(target) { this.targets.delete(target); }
  disconnect() { this.targets.clear(); }
}

let frames = [];
const requestAnimationFrame = (frame) => frames.push(frame);
const paint = () => {
  const due = frames;
  frames = [];
  for (const frame of due) { frame(); }
};

const kept = new Map();
const localStorage = {
  getItem: (key) => (kept.has(key) ? kept.get(key) : null),
  setItem: (key, value) => kept.set(key, value),
  removeItem: (key) => kept.delete(key),
};

const documentElement = new Element('html', 1200);
const body = documentElement.append(new Element('body', 1200));
const heard = { document: {}, window: {} };
const document = {
  documentElement,
  body,
  getElementById: (id) => find(documentElement, id),
  querySelectorAll: (selector) => {
    assert.strictEqual(selector, '.pane-resizer', 'the page answers one selector');
    return gather(documentElement, 'pane-resizer', []);
  },
  addEventListener: (type, fn) => {
    (heard.document[type] = heard.document[type] || []).push(fn);
  },
};
const window = {
  innerWidth: 1200,
  matchMedia: () => ({ matches: window.innerWidth <= 850 }),
  addEventListener: (type, fn) => {
    (heard.window[type] = heard.window[type] || []).push(fn);
  },
};
const fire = (where, type, event) => {
  for (const fn of (heard[where][type] || []).slice()) { fn(event); }
};
const shell = body.append(new Element('shell', 1200));

// A handle of the kind pane_handle() writes, carrying the same attributes.
const seam = (pane) => {
  const handle = new Element('', 6);
  handle.names.add('pane-resizer');
  handle.dataset.pane = pane;
  handle.dataset.property = '--' + pane + '-width';
  handle.dataset.store = 'chatlab.' + pane + '-width';
  return handle;
};

// The Chat row gives up space to the conversations pane beside it, and the
// Images row has only its handle to pay for.
const chatRow = (width) => {
  const row = shell.append(new Element('chat-columns', width));
  row.append(new Element('conversation-pane', 260));
  row.append(new Element('chat-workspace', width - 580));
  row.append(seam('inspector-pane'));
  row.append(new Element('inspector-pane', 314));
  return row;
};

const imagesRow = (width) => {
  const row = shell.append(new Element('images-columns', width));
  row.append(new Element('images-workspace', width - 320));
  row.append(seam('image-inspector'));
  row.append(new Element('image-inspector', 314));
  return row;
};
"""


class PaneResizeScriptTests(unittest.TestCase):
    """RESIZE_JS itself, run over a stand-in page.

    The script is the one part of the resizing that no Python call can
    reach, so these run the real string in node and let its own assertions
    report. A machine without node skips them.
    """

    def check(self, checks: str):
        script = f"{RESIZE_PAGE}\nconst start = {app.RESIZE_JS};\n{checks}"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resize.js"
            path.write_text(script)
            result = subprocess.run(
                ["node", str(path)], capture_output=True, text=True, timeout=60
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_page_built_again_comes_back_to_a_fitted_width(self):
        # The nav takes a page out of the document when it turns away and
        # Gradio builds it afresh on the way back, in a row nothing has
        # measured yet. A width fitted to the window while the page was away
        # is a guess, and the page returning is the moment to correct it.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const [rows] = resizers;
assert.ok(rows.targets.has(chat), 'the row on screen is watched from the start');

// The Images page is built the first time the reader opens it.
const first = imagesRow(1200);
mutated();
paint();
assert.ok(rows.targets.has(first), 'a row that has just arrived is watched');

// The nav turns away, taking that page out of the document, and the window
// is dragged narrower while it is gone. With no row left to measure, the
// width the reader chose is cut to what the window alone suggests.
kept.set('chatlab.image-inspector-width', '900');
first.remove();
mutated();
assert.ok(
  !rows.targets.has(first),
  'the row of a page that has been taken away is let go at once'
);
window.innerWidth = 1000;
fire('window', 'resize', {});
paint();
assert.strictEqual(documentElement.style.props['--image-inspector-width'], '380px');

// The nav turns back and the page is built again. Its row has more room
// than the window alone suggested, and the pane is given it.
const second = imagesRow(1000);
mutated();
paint();
assert.ok(rows.targets.has(second), 'the row a rebuilt page comes back in is watched');
assert.ok(!rows.targets.has(first), 'and the row it left is let go');
assert.strictEqual(documentElement.style.props['--image-inspector-width'], '634px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_drag_that_ends_outside_the_window_still_ends(self):
        # A button let go beyond the edge of the window is a release the
        # page never hears, so the pointer is captured for the length of the
        # drag and a pointer that comes back with nothing held ends it.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
const pane = document.getElementById('inspector-pane');
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
assert.ok(handle.hasPointerCapture(7), 'the handle keeps the pointer for the drag');
assert.ok(body.classList.contains('pane-dragging'));

// The pane is on the right of its handle, so dragging left widens it.
fire('window', 'pointermove', { buttons: 1, clientX: 760 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');

// The reader let go out beyond the edge of the window and brought the
// pointer back with the button up.
fire('window', 'pointermove', { buttons: 0, clientX: 600 });
assert.ok(!body.classList.contains('pane-dragging'), 'the page stops being dragged');
assert.ok(!handle.hasPointerCapture(7), 'and the handle gives the pointer back');
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), String(pane.width));

// So moving the pointer over the page again leaves the pane where it was.
fire('window', 'pointermove', { buttons: 1, clientX: 400 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_pane_wider_than_a_drag_allows_is_reported_where_it_is(self):
        # Between the width that stacks the panes and the width that gives
        # them their full share, the stylesheet's smaller default can be
        # more than a drag would leave the workspace. The separator says
        # where the pane is rather than where it would be allowed, and the
        # key asking for it to be pushed out does not pull it in.
        self.check(
            """
// The row keeps 260 for the conversations pane and 6 for the handle, so a
// drag would allow 294 of the 654 left, and the pane is already at 314.
const chat = chatRow(920);
start();
paint();

const handle = chat.children[2];
assert.strictEqual(handle.getAttribute('aria-valuenow'), '314');
assert.strictEqual(handle.getAttribute('aria-valuemax'), '314', 'the range holds it');
assert.strictEqual(handle.getAttribute('aria-valuetext'), '314 pixels');

// ArrowLeft asks for a wider pane. There is no room to widen it, so it
// stays where it is rather than being cut to what a drag would allow.
fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], undefined);
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), undefined);
assert.strictEqual(handle.getAttribute('aria-valuenow'), '314');

// ArrowRight asks for a narrower one, which there is room for.
fire('document', 'keydown', {
  target: handle, key: 'ArrowRight', preventDefault: () => {},
});
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '294px');
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_key_that_moves_nothing_keeps_the_width_the_reader_chose(self):
        # A pane squeezed by a narrow window is already at its maximum, so
        # the key asking for it to be wider moves nothing. Writing that
        # squeezed width down as the reader's choice would lose the wider
        # one they picked when there was room for it.
        self.check(
            """
const chat = chatRow(1000);
kept.set('chatlab.inspector-pane-width', '520');
start();
paint();

// The row of 1000 keeps 260 for the conversations pane and 6 for the
// handle, so the 520 the reader chose is cut to 374 while the window is
// this narrow. The choice itself is untouched.
const handle = chat.children[2];
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '374px');
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), '520');

// The stand-in page does not lay itself out, so the pane is told what the
// width just written would have made it.
document.getElementById('inspector-pane').width = 374;

fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(
  kept.get('chatlab.inspector-pane-width'), '520',
  'a key with nowhere to go leaves the choice alone'
);
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_a_click_on_a_handle_chooses_nothing(self):
        # Clicking a handle is how it takes the focus the arrow keys need,
        # and a reader who has chosen nothing has still chosen nothing. A
        # click that pinned the width on screen would take the pane out of
        # the stylesheet's hands, and one made while a narrow window was
        # squeezing the pane would write that squeeze over the wider width
        # the reader picked when there was room for it.
        self.check(
            """
const chat = chatRow(1200);
kept.set('chatlab.inspector-pane-width', '520');
start();
paint();

const handle = chat.children[2];
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointerup', { pointerId: 7 });
assert.strictEqual(
  kept.get('chatlab.inspector-pane-width'), '520', 'the choice is left alone'
);

// A drag that moves the pane is a choice, and is kept.
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 8,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 700, pointerId: 8 });
fire('window', 'pointerup', { pointerId: 8 });
assert.strictEqual(kept.get('chatlab.inspector-pane-width'), String(pane().width));

function pane() { return document.getElementById('inspector-pane'); }
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_only_the_pointer_that_started_a_drag_can_move_or_end_it(self):
        # A second finger on a touch screen reports moves and a release of
        # its own. Neither belongs to the drag the first finger started.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 760, pointerId: 7 });
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], '354px');

// A second finger lands on the same strip, moves and then lifts.
fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 300, pointerId: 9,
  preventDefault: () => {},
});
assert.ok(handle.hasPointerCapture(7), 'the first pointer keeps the drag');
fire('window', 'pointermove', { buttons: 1, clientX: 300, pointerId: 9 });
assert.strictEqual(
  documentElement.style.props['--inspector-pane-width'], '354px',
  'the pane does not jump to a pointer that is not dragging it'
);
fire('window', 'pointerup', { pointerId: 9 });
assert.ok(body.classList.contains('pane-dragging'), 'the drag is still going');
assert.ok(handle.hasPointerCapture(7), 'and the first pointer is still held');

// The finger that started the drag lifts, and it ends.
fire('window', 'pointerup', { pointerId: 7 });
assert.ok(!body.classList.contains('pane-dragging'));
assert.ok(!handle.hasPointerCapture(7));
"""
        )

    @unittest.skipUnless(shutil.which("node"), "needs node to run the script")
    def test_the_separator_carries_the_position_it_has_put_the_pane_in(self):
        # Focusing a separator is meant to tell a screen reader how the room
        # has been divided, and nothing else on the page can say. So every
        # write of a width says it again, and a pane with no width on screen
        # to speak of says nothing at all.
        self.check(
            """
const chat = chatRow(1200);
start();
paint();

const handle = chat.children[2];
const pane = document.getElementById('inspector-pane');

// The row of 1200 gives 260 to the conversations pane and 6 to the handle,
// leaving 934 for the pane and its workspace to divide, of which the
// workspace keeps at least 360.
assert.strictEqual(handle.getAttribute('aria-valuemin'), '240');
assert.strictEqual(handle.getAttribute('aria-valuemax'), '574');
// Nobody has dragged anything yet, so the figure is the width the
// stylesheet gave the pane.
assert.strictEqual(handle.getAttribute('aria-valuenow'), String(pane.width));
assert.strictEqual(handle.getAttribute('aria-valuetext'), pane.width + ' pixels');

fire('document', 'pointerdown', {
  target: handle, button: 0, buttons: 1, clientX: 800, pointerId: 7,
  preventDefault: () => {},
});
fire('window', 'pointermove', { buttons: 1, clientX: 760 });
fire('window', 'pointerup', {});
assert.strictEqual(handle.getAttribute('aria-valuenow'), '354');
assert.strictEqual(handle.getAttribute('aria-valuetext'), '354 pixels');

// An arrow key steps the same figure along.
fire('document', 'keydown', {
  target: handle, key: 'ArrowLeft', preventDefault: () => {},
});
assert.strictEqual(handle.getAttribute('aria-valuenow'), '330');

// A double-click hands the pane back to the stylesheet, whose width only
// the layout knows, so the separator reports what it measures next.
fire('document', 'dblclick', { target: handle, preventDefault: () => {} });
paint();
assert.strictEqual(documentElement.style.props['--inspector-pane-width'], undefined);
assert.strictEqual(handle.getAttribute('aria-valuenow'), String(pane.width));

// The handle of a page the nav is not showing measures nothing, and a
// position invented for it would describe a layout that never happened.
const images = imagesRow(0);
mutated();
paint();
assert.strictEqual(images.children[1].getAttribute('aria-valuenow'), null);
"""
        )


if __name__ == "__main__":
    unittest.main()
