"""The interface's icons: one stroke set, drawn in whatever colour it lands in.

Every icon here is an outline on the same 24x24 grid at the same 2px stroke,
which is what makes a row of them read as one set. The app drew them as emoji
before this, and emoji are a different typeface per glyph: the weights, the
colours and the optical sizes never agreed, and a few of them arrived as a
box on a machine without the font.

They are painted as CSS masks rather than dropped in as pictures. A mask
takes the colour of the element it sits on, so an icon in a button follows
that button's text through hover, selection, dark mode and a theme change
with no second copy of the file; an ``<img>`` would hold whatever colour it
was exported at. The mask is a data URL, so no icon is a network request and
none of them can be missing from a build.

The drawings are Lucide's (ISC), redrawn here as the path data alone so the
set is a dictionary rather than a dependency.
"""

from __future__ import annotations

from urllib.parse import quote


# What every drawing below is stroked with. The colour is immaterial - a mask
# reads a pixel's alpha and throws its colour away - but a stroke needs one.
_SVG_ATTRS = (
    'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#000" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"'
)


# Name against the marks that draw it. The names are Lucide's, so an icon
# wanted later can be looked up there and pasted in.
ICONS: dict[str, str] = {
    "message-square": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 '
        '2 2z"/>'
    ),
    "route": (
        '<circle cx="6" cy="19" r="3"/>'
        '<path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/>'
        '<circle cx="18" cy="5" r="3"/>'
    ),
    "image": (
        '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>'
        '<circle cx="9" cy="9" r="2"/>'
        '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'
    ),
    "box": (
        '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 '
        '8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/>'
        '<path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>'
    ),
    "settings": (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 '
        '2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 '
        '2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 '
        '0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 '
        '2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 '
        '1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 '
        '0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15'
        '-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 '
        '0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "git-branch": (
        '<line x1="6" x2="6" y1="3" y2="15"/>'
        '<circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
        '<path d="M18 9a9 9 0 0 1-9 9"/>'
    ),
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>'
        '<path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" x2="10" y1="11" y2="17"/>'
        '<line x1="14" x2="14" y1="11" y2="17"/>'
    ),
    "rotate-ccw": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
    ),
    "refresh": (
        '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
        '<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>'
        '<path d="M16 16h5v5"/>'
    ),
    "undo": (
        '<path d="M9 14 4 9l5-5"/>'
        '<path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H10"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>'
    ),
    "folder-open": (
        '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 '
        '6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 '
        '1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>'
    ),
    "pencil": (
        '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 '
        '0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 '
        '.83-.497z"/><path d="m15 5 4 4"/>'
    ),
    "layers": (
        '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 '
        '3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>'
        '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>'
        '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
    ),
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 '
        '0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "chevron-left": '<path d="m15 18-6-6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "play": '<path d="M6 3.5 20 12 6 20.5Z"/>',
    "pause": (
        '<rect x="6" y="4" width="4" height="16" rx="1"/>'
        '<rect x="14" y="4" width="4" height="16" rx="1"/>'
    ),
}


def svg(name: str) -> str:
    """The named icon as a standalone SVG document."""

    return f"<svg {_SVG_ATTRS}>{ICONS[name]}</svg>"


def data_url(name: str) -> str:
    """The named icon as a URL a stylesheet can mask with.

    Everything is percent-encoded rather than only the characters that must
    be: an unencoded ``#`` would end the URL at the stroke colour, and the
    rest of the drawing would be read as a fragment.
    """

    return f"data:image/svg+xml,{quote(svg(name), safe='')}"


def mask(name: str) -> str:
    """The declarations that paint the named icon over an element.

    The unprefixed property is written last so a browser that understands
    both ends up with the standard one.
    """

    url = f'url("{data_url(name)}")'
    return (
        f"-webkit-mask-image: {url}; mask-image: {url}; "
        "-webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; "
        "-webkit-mask-position: center; mask-position: center; "
        "-webkit-mask-size: contain; mask-size: contain;"
    )


def mask_rule(selector: str, name: str) -> str:
    """One rule that gives ``selector`` the named icon as its mask."""

    return f"{selector} {{ {mask(name)} }}"


# The class a control wears to say it carries an icon, and the class that
# says which one. Both are put on the same element: the first sizes the box
# and colours it, the second is only the drawing.
ICON_CLASS = "icon-btn"


# Put on a control whose icon belongs after its label rather than in front
# of it, which is what a "next" reads as.
TRAILING_CLASS = "icon-trailing"


def icon_classes(name: str, *, trailing: bool = False) -> list[str]:
    """The classes that draw ``name`` beside a control's label."""

    classes = [ICON_CLASS, f"icon-{name}"]
    if trailing:
        classes.append(TRAILING_CLASS)
    return classes


# Every icon's mask, one rule each, for whatever wears ``icon-<name>``. The
# box those masks are painted in is set once in the stylesheet; this is only
# which drawing goes in it.
ICON_CSS = "\n".join(
    mask_rule(f".icon-{name}::before", name) for name in ICONS
)
