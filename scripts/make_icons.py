"""Regenerate the app icon and favicon from ``assets/chatlab-logo.png``.

The artwork arrives as an icon-shaped tile drawn on a black square. macOS
wants the tile alone on a transparent canvas, at the proportions the rest of
the Dock uses, so this crops the tile out, rounds its corners back to the
radius the artwork was drawn with, and sets it in the 1024-pixel canvas
Apple's icon grid asks for. Both outputs are committed; this script exists so
a new drawing can replace them without anyone having to remember the numbers.

Usage: .venv/bin/python scripts/make_icons.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


ASSETS = Path(__file__).resolve().parent.parent / "assets"
SOURCE = ASSETS / "chatlab-logo.png"
ICNS = ASSETS / "ChatLab.icns"
FAVICON = ASSETS / "icon.png"

# Where the tile sits inside the drawing, measured off the black surround.
TILE_BOX = (123, 89, 1128, 1086)
# The corner radius the tile was drawn with, as a fraction of its own width.
# Rounder than Apple's own rounded rectangle, so trimming to Apple's radius
# would leave a wedge of the black surround at each corner.
CORNER_FRACTION = 280 / 1004

# Apple's icon grid: an 824-pixel tile centred in a 1024-pixel canvas, the
# empty margin left for the shadow every other Dock icon casts.
CANVAS = 1024
TILE = 824
FAVICON_SIZE = 256
# The mask is drawn large and shrunk so its curve is smooth rather than
# stepped; anti-aliased shape drawing is not something Pillow offers.
SUPERSAMPLE = 4


def rounded_tile(source: Image.Image) -> Image.Image:
    """Return the tile at ``TILE`` pixels square with its corners cut away."""

    tile = source.crop(TILE_BOX).resize((TILE, TILE), Image.LANCZOS).convert("RGBA")
    big = TILE * SUPERSAMPLE
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, big - 1, big - 1), radius=round(CORNER_FRACTION * big), fill=255
    )
    tile.putalpha(mask.resize((TILE, TILE), Image.LANCZOS))
    return tile


def icon_canvas(tile: Image.Image) -> Image.Image:
    """Set the tile in the transparent canvas at Apple's proportions."""

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    offset = (CANVAS - TILE) // 2
    canvas.paste(tile, (offset, offset), tile)
    return canvas


def write_icns(icon: Image.Image, destination: Path) -> None:
    """Build a .icns from the 1024-pixel icon with ``iconutil``."""

    with tempfile.TemporaryDirectory() as work:
        iconset = Path(work) / "ChatLab.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            icon.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(
            ["iconutil", "--convert", "icns", "--output", str(destination), str(iconset)], check=True
        )


def main() -> None:
    tile = rounded_tile(Image.open(SOURCE))
    write_icns(icon_canvas(tile), ICNS)
    # The browser tab wants the tile filling the frame: a 16-pixel favicon has
    # no room to spend on the margin the Dock needs. It is fetched on every
    # page load, so it is kept to the largest size a tab or a bookmark bar
    # will ever draw rather than to the icon's own.
    tile.resize((FAVICON_SIZE, FAVICON_SIZE), Image.LANCZOS).save(FAVICON, optimize=True)
    print(f"Wrote {ICNS} and {FAVICON}")


if __name__ == "__main__":
    main()
