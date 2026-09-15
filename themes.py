"""The color themes the Settings page offers.

Gradio fixes a theme when the interface is built, so a reader who picked
another one would have to reload the page to see it. Every color the
interface draws with is a CSS variable, though, and all of them lead back to
two eleven-step ramps: ``--primary-*``, which the buttons, the accents and
the focus rings come from, and ``--neutral-*``, which every surface, border
and line of text comes from. A theme here is those two ramps and the one
literal the ramps do not cover - the paper the light mode is drawn on - so
switching is a stylesheet swapped on the page rather than a restart.

Three of them are the app's own mark read as a palette: the speech bubble's
electric indigo (Aurora), the violet layers behind it (Nebula), and the gold
nodes threading through them (Ember).

Whether a theme is drawn light or dark is the other half of the page's look
and a choice of its own, since every theme has both. It is a class on the
body rather than a stylesheet; see APPEARANCE_JS.
"""

from __future__ import annotations

from dataclasses import dataclass


# The eleven steps every Gradio ramp has, lightest first. The names are the
# variable suffixes, so the order here is the order the values are written in.
STEPS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)

# Two steps carry more weight than the rest and are what each ramp below is
# tuned around. 600 is the primary button's fill, drawn under white text, so
# it is dark enough to read at 4.5:1 - the reason the warmer themes bend away
# from their own mid-tones there rather than running a gold button. 950 is
# the dark mode's paper, so it is where a theme's tint is most visible.
BUTTON_STEP = 600
DARKEST_STEP = 950


@dataclass(frozen=True)
class Theme:
    """One palette: what it is called, and the colors it draws with."""

    label: str
    caption: str
    # Lightest to darkest, one per entry in STEPS.
    primary: tuple[str, ...]
    neutral: tuple[str, ...]
    # Light mode's paper. The dark mode's is the neutral ramp's last step, so
    # it is not named twice.
    paper: str

    def ramp(self, name: str) -> dict[int, str]:
        """The named ramp as step number against color."""

        return dict(zip(STEPS, getattr(self, name), strict=True))


THEMES: dict[str, Theme] = {
    "graphite": Theme(
        label="Graphite",
        caption=(
            "Indigo on a neutral gray. ChatLab's original look, and what it "
            "starts from."
        ),
        primary=(
            "#eef2ff", "#e0e7ff", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1",
            "#4f46e5", "#4338ca", "#3730a3", "#312e81", "#2b2c5e",
        ),
        neutral=(
            "#fafafa", "#f4f4f5", "#e4e4e7", "#d4d4d8", "#bbbbc2", "#71717a",
            "#52525b", "#3f3f46", "#27272a", "#18181b", "#0f0f11",
        ),
        paper="#ffffff",
    ),
    "aurora": Theme(
        label="Aurora",
        caption=(
            "The logo's speech bubble: electric indigo over the blue-black "
            "the mark sits on."
        ),
        primary=(
            "#eef3ff", "#dde6ff", "#c2d1ff", "#9bb1ff", "#6f86fb", "#4f5bf0",
            "#3b3fdc", "#3231b4", "#2b2c90", "#272873", "#191a4a",
        ),
        neutral=(
            "#f6f7fc", "#ecedf6", "#d9dcec", "#b6bcd6", "#858cb4", "#5d6486",
            "#474d69", "#363b52", "#252940", "#12152e", "#080a20",
        ),
        paper="#fbfcff",
    ),
    "nebula": Theme(
        label="Nebula",
        caption=(
            "The violet layers behind the bubble, on a plum-tinted gray. The "
            "warmest of the three drawn from the mark."
        ),
        primary=(
            "#faf5ff", "#f3e8ff", "#e9d5ff", "#d8b4fe", "#c084fc", "#a855f7",
            "#8b28d8", "#7620b8", "#621d96", "#4f1a78", "#33104e",
        ),
        neutral=(
            "#faf8fb", "#f2eff5", "#e5e0ea", "#c9c1d2", "#a096ae", "#6f6580",
            "#574e66", "#433b50", "#2f293b", "#1e1927", "#110d18",
        ),
        paper="#fdfbff",
    ),
    "ember": Theme(
        label="Ember",
        caption=(
            "The gold nodes threading through the layers, over warm gray. "
            "Its buttons darken to burnt amber so their text stays readable."
        ),
        primary=(
            "#fffbeb", "#fef3c7", "#fde68a", "#fcd34d", "#fbbf24", "#d97706",
            "#b45309", "#92400e", "#78350f", "#633008", "#431f05",
        ),
        neutral=(
            "#fafaf9", "#f5f5f4", "#e7e5e4", "#d6d3d1", "#c0bab2", "#78716c",
            "#57534e", "#44403c", "#292524", "#1c1917", "#100e0c",
        ),
        paper="#fffdf8",
    ),
    "lagoon": Theme(
        label="Lagoon",
        caption="The cyan highlight along the bubble's edge, over cool slate.",
        primary=(
            "#ecfeff", "#cffafe", "#a5f3fc", "#67e8f9", "#22d3ee", "#0891b2",
            "#0e7490", "#155e75", "#164e63", "#123f51", "#082f3d",
        ),
        neutral=(
            "#f8fafc", "#f1f5f9", "#e2e8f0", "#cbd5e1", "#b0bccb", "#64748b",
            "#475569", "#334155", "#1e293b", "#0f172a", "#080d18",
        ),
        paper="#f9fdfe",
    ),
    "moss": Theme(
        label="Moss",
        caption=(
            "Deep green on a gray with the same cast. Nothing to do with the "
            "logo - the quiet one."
        ),
        primary=(
            "#ecfdf5", "#d1fae5", "#a7f3d0", "#6ee7b7", "#34d399", "#059669",
            "#047857", "#065f46", "#064e3b", "#053e30", "#022c22",
        ),
        neutral=(
            "#f7faf8", "#eef3f0", "#dde6e1", "#c0cec6", "#9aa9a1", "#667a70",
            "#4f6158", "#3c4a43", "#29342e", "#19211d", "#0b110e",
        ),
        paper="#fbfdfc",
    ),
}

# What a new install, and a settings file that names a theme this version has
# never heard of, both come up in: the look ChatLab had before it had themes.
DEFAULT_THEME = "graphite"

THEME_NAMES = tuple(THEMES)
# Label against name, which is what a Dropdown's choices are.
THEME_CHOICES = [(theme.label, name) for name, theme in THEMES.items()]

# Whether a theme is drawn light or dark is a separate choice from which
# theme it is: every one of them has both. Gradio decides it once at startup
# from the system setting, so these are the two overrides plus the setting
# itself.
APPEARANCES = {
    "system": "Follow system",
    "light": "Light",
    "dark": "Dark",
}
DEFAULT_APPEARANCE = "system"
APPEARANCE_NAMES = tuple(APPEARANCES)
APPEARANCE_CHOICES = [(label, name) for name, label in APPEARANCES.items()]

# Light or dark is a class on the body: Gradio writes every dark-mode value
# under ``.dark``, and the app's own stylesheet follows it. So the choice is
# that class added or removed, which is a repaint rather than a reload.
#
# The listener is what makes "Follow system" a standing arrangement rather
# than a reading taken once. Gradio registers one of its own at startup that
# does the same thing, and a forced light page would otherwise turn dark the
# moment the system did; this one is added after it and so has the last word.
# The choice is parked on the window because the listener outlives the call
# that installed it and has to know what the reader has picked since.
APPEARANCE_JS = """
(mode) => {
  const query = window.matchMedia('(prefers-color-scheme: dark)');
  const paint = () => {
    const chosen = window.__chatlabAppearance;
    const dark = chosen === 'dark' || (chosen !== 'light' && query.matches);
    document.body.classList.toggle('dark', dark);
  };
  window.__chatlabAppearance = mode;
  if (!window.__chatlabAppearanceWatched) {
    window.__chatlabAppearanceWatched = true;
    query.addEventListener('change', paint);
  }
  paint();
}
"""


def resolve(name: str | None) -> Theme:
    """The named theme, or the default where the name is not one of them."""

    return THEMES.get(name or "", THEMES[DEFAULT_THEME])


def caption(name: str | None) -> str:
    """The chosen theme's one-line description, for the control's caption."""

    return resolve(name).caption


def stylesheet(name: str | None) -> str:
    """The CSS that repaints the interface in the named theme.

    Every declaration is ``!important`` because Gradio has already written
    the built-in theme's ramps into the document and this has to win wherever
    it lands in the cascade. The ramps are given to both the page and the
    dark-mode body: Gradio writes its own copy of every variable under
    ``.dark``, and a value set on that element beats one inherited from the
    page however important the inherited one is.

    The paper is light mode's alone. Dark mode's surfaces are already written
    in terms of the neutral ramp - body from step 950, blocks from 900 - so
    they follow from the ramp above without being named here, and overriding
    them on the page would flatten dark mode into the light one's paper.
    """

    theme = resolve(name)
    ramps = "\n".join(
        f"  --{ramp}-{step}: {color} !important;"
        for ramp in ("primary", "neutral")
        for step, color in theme.ramp(ramp).items()
    )
    paper = "\n".join(
        f"  --{variable}: {theme.paper} !important;"
        for variable in (
            "body-background-fill",
            "background-fill-primary",
            "block-background-fill",
            "input-background-fill",
            "button-secondary-background-fill",
        )
    )
    return f":root, :root body.dark {{\n{ramps}\n}}\n:root {{\n{paper}\n}}"


def style_tag(name: str | None) -> str:
    """The stylesheet as an element to drop into the page.

    A ``<style>`` written into the document this way is parsed and applied
    the moment it lands, which is what lets a theme change without a reload.
    """

    return f"<style>\n{stylesheet(name)}\n</style>"
