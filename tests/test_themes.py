"""The color themes: the ramps they are built from, and the CSS they become."""

import re
import unittest

import app
import settings
import settings_sandbox
import themes


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


HEX = re.compile(r"^#[0-9a-f]{6}$")


def _relative_luminance(color: str) -> float:
    """WCAG's luminance for a ``#rrggbb`` color."""

    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(one: str, other: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(one), _relative_luminance(other)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


class RampTests(unittest.TestCase):
    """Every theme has to be a pair of ramps Gradio can read as its own."""

    def test_every_ramp_has_one_color_per_step(self):
        for name, theme in themes.THEMES.items():
            for ramp in ("primary", "neutral"):
                with self.subTest(theme=name, ramp=ramp):
                    self.assertEqual(len(getattr(theme, ramp)), len(themes.STEPS))

    def test_every_color_is_a_six_digit_hex(self):
        for name, theme in themes.THEMES.items():
            for ramp in ("primary", "neutral"):
                for step, color in theme.ramp(ramp).items():
                    with self.subTest(theme=name, ramp=ramp, step=step):
                        self.assertRegex(color, HEX)
            with self.subTest(theme=name, ramp="paper"):
                self.assertRegex(theme.paper, HEX)

    def test_every_ramp_runs_light_to_dark(self):
        # Both modes read the ramp by position: light mode draws its text from
        # step 800 on paper, dark mode draws step 100 on step 950. A ramp that
        # doubled back would leave one of the two unreadable.
        for name, theme in themes.THEMES.items():
            for ramp in ("primary", "neutral"):
                with self.subTest(theme=name, ramp=ramp):
                    luminances = [
                        _relative_luminance(color) for color in getattr(theme, ramp)
                    ]
                    self.assertEqual(luminances, sorted(luminances, reverse=True))

    def test_a_primary_button_carries_its_white_text_at_contrast(self):
        # The primary button is step 600 under white text, as is the selected
        # conversation in the side pane. This is the bar the warmer themes are
        # tuned against: a gold button would never have cleared it.
        for name, theme in themes.THEMES.items():
            with self.subTest(theme=name):
                self.assertGreaterEqual(
                    _contrast(theme.ramp("primary")[themes.BUTTON_STEP], "#ffffff"), 4.5
                )

    def test_body_text_carries_against_the_paper_it_is_drawn_on(self):
        for name, theme in themes.THEMES.items():
            neutral = theme.ramp("neutral")
            with self.subTest(theme=name, mode="light"):
                self.assertGreaterEqual(_contrast(neutral[800], theme.paper), 4.5)
            with self.subTest(theme=name, mode="dark"):
                self.assertGreaterEqual(
                    _contrast(neutral[100], neutral[themes.DARKEST_STEP]), 4.5
                )

    def test_the_default_theme_is_the_look_the_app_already_had(self):
        # Graphite is indigo on zinc over white, which is what the Gradio
        # theme in ui.styles is built from. An install that never opens the
        # Settings page must not change appearance.
        theme = themes.THEMES[themes.DEFAULT_THEME]
        self.assertEqual(theme.ramp("primary")[500], "#6366f1")
        self.assertEqual(theme.ramp("neutral")[500], "#71717a")
        self.assertEqual(theme.paper, "#ffffff")

    def test_the_choices_pair_a_label_with_each_name(self):
        self.assertEqual(
            themes.THEME_CHOICES,
            [(theme.label, name) for name, theme in themes.THEMES.items()],
        )
        self.assertIn(themes.DEFAULT_THEME, themes.THEME_NAMES)

    def test_three_of_them_are_drawn_from_the_logo(self):
        for name in ("aurora", "nebula", "ember"):
            with self.subTest(theme=name):
                self.assertIn(name, themes.THEMES)


class ResolveTests(unittest.TestCase):
    def test_a_known_name_gives_its_theme(self):
        self.assertIs(themes.resolve("ember"), themes.THEMES["ember"])

    def test_an_unknown_name_falls_back_to_the_default(self):
        for name in ("sepia", "", None):
            with self.subTest(name=name):
                self.assertIs(
                    themes.resolve(name), themes.THEMES[themes.DEFAULT_THEME]
                )

    def test_the_caption_is_the_chosen_theme_s_own(self):
        self.assertEqual(themes.caption("lagoon"), themes.THEMES["lagoon"].caption)


class StylesheetTests(unittest.TestCase):
    def test_it_writes_both_ramps_over_the_built_in_ones(self):
        css = themes.stylesheet("aurora")
        theme = themes.THEMES["aurora"]
        for ramp in ("primary", "neutral"):
            for step, color in theme.ramp(ramp).items():
                with self.subTest(ramp=ramp, step=step):
                    self.assertIn(f"--{ramp}-{step}: {color} !important;", css)

    def test_the_ramps_reach_the_dark_body_as_well_as_the_page(self):
        # Gradio writes its own copy of every variable under ``.dark``, and a
        # value set on that element beats one inherited from the page however
        # important the inherited one is. So the ramps have to name both.
        css = themes.stylesheet("nebula")
        self.assertIn(":root, :root body.dark {", css)

    def test_the_paper_is_light_mode_s_alone(self):
        # Dark mode's surfaces already read from the neutral ramp, so writing
        # the paper onto the dark body would flatten it into the light one.
        css = themes.stylesheet("ember")
        page = css.split(":root {")[1]
        self.assertIn(f"--body-background-fill: {themes.THEMES['ember'].paper}", page)
        self.assertNotIn("body.dark", page)

    def test_an_unknown_theme_still_produces_the_default_s_stylesheet(self):
        self.assertEqual(
            themes.stylesheet("sepia"), themes.stylesheet(themes.DEFAULT_THEME)
        )

    def test_the_tag_wraps_the_stylesheet_in_a_style_element(self):
        tag = themes.style_tag("moss")
        self.assertTrue(tag.startswith("<style>"))
        self.assertTrue(tag.endswith("</style>"))
        self.assertIn(themes.stylesheet("moss"), tag)


class SettingTests(unittest.TestCase):
    """The chosen theme is one of the settings that outlive the session."""

    def test_a_new_install_starts_on_the_default(self):
        self.assertEqual(settings.DEFAULTS.theme, themes.DEFAULT_THEME)

    def test_a_named_theme_is_kept(self):
        self.assertEqual(settings.sanitize({"theme": "aurora"}).theme, "aurora")

    def test_a_theme_this_version_has_never_heard_of_falls_back(self):
        # A settings file shared between machines may name a theme added
        # since, and starting is never worth refusing over one.
        for value in ("sepia", 7, None, ["aurora"]):
            with self.subTest(value=value):
                self.assertEqual(
                    settings.sanitize({"theme": value}).theme, themes.DEFAULT_THEME
                )


class ThemeControlTests(unittest.TestCase):
    """The dropdown on the Settings page, and the stylesheet it drives."""

    @classmethod
    def setUpClass(cls):
        settings.write(settings.sanitize({"theme": "ember"}))
        settings.load()
        cls.demo = app.build_app()

    @classmethod
    def tearDownClass(cls):
        settings.settings_path().unlink(missing_ok=True)
        settings.load()

    def block(self, elem_id):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "elem_id", None) == elem_id
        ]
        self.assertEqual(len(matches), 1, elem_id)
        return matches[0]

    def labelled(self, label):
        matches = [
            block
            for block in self.demo.blocks.values()
            if getattr(block, "label", None) == label
        ]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def listeners(self, name):
        return [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == name
        ]

    def test_the_dropdown_offers_every_theme_and_starts_on_the_saved_one(self):
        dropdown = self.labelled("Color theme")
        self.assertEqual(
            [value for _label, value in dropdown.choices], list(themes.THEME_NAMES)
        )
        self.assertEqual(dropdown.value, "ember")

    def test_the_page_comes_up_already_painted_in_the_saved_theme(self):
        # Built with the stylesheet rather than given one afterwards: nothing
        # should be drawn in the built-in colors and then repainted.
        self.assertEqual(
            self.block("theme-style").value, themes.style_tag("ember")
        )

    def test_choosing_a_theme_repaints_and_saves_it(self):
        dropdown = self.labelled("Color theme")
        painting = self.listeners("apply_theme")
        self.assertTrue(
            any(dropdown in fn.inputs for fn in painting),
            "the dropdown drives no repaint",
        )
        saving = self.listeners("remember_settings")
        triggering = [fn for fn in saving if dropdown in fn.inputs]
        self.assertTrue(triggering, "the dropdown is not one of the saved settings")
        self.assertIn(
            dropdown,
            [self.demo.blocks[block_id] for fn in saving for block_id, _ in fn.targets],
        )

    def test_saving_and_repainting_keep_the_same_last_pick(self):
        # A reader trying the themes on picks one while the one before it is
        # still in flight. If only one of the two listeners kept the last
        # pick, the page would be painted in one theme and the file would hold
        # another, and a reload would undo a choice that was there on screen.
        dropdown = self.labelled("Color theme")
        listening = [
            fn
            for fn in self.demo.fns.values()
            if dropdown in fn.inputs
            and dropdown
            in [
                self.demo.blocks[block_id]
                for block_id, _ in fn.targets
                if block_id is not None
            ]
        ]
        self.assertEqual(
            sorted(fn.fn.__name__ for fn in listening),
            ["apply_theme", "remember_settings"],
        )
        for fn in listening:
            with self.subTest(handler=fn.fn.__name__):
                self.assertEqual(fn.trigger_mode, "always_last")

    def test_a_reload_paints_the_theme_the_file_now_names(self):
        # restore_settings re-reads the file, so the stylesheet has to follow
        # the dropdown on the way back rather than staying as it was built.
        restoring = self.listeners("restore_settings")
        self.assertEqual(len(restoring), 1)
        self.assertIn(self.labelled("Color theme"), restoring[0].outputs)
        self.assertIn(
            self.block("theme-style"),
            [output for fn in self.listeners("apply_theme") for output in fn.outputs],
        )

    def test_the_handler_gives_back_the_stylesheet_and_the_caption(self):
        style, caption = app.apply_theme("lagoon")
        self.assertEqual(style["value"], themes.style_tag("lagoon"))
        self.assertEqual(caption["value"], themes.THEMES["lagoon"].caption)


if __name__ == "__main__":
    unittest.main()
