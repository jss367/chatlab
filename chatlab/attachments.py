"""The pictures a reader pastes into a message, and where they are kept.

A picture is stored once, under the SHA-256 of its bytes, in a directory
beside the conversations file::

    ~/.local/share/chatlab/conversations-images/<sha256>.png

A user turn names its pictures by those file names, in the order they were
attached::

    {"role": "user", "content": "What is this?", "images": ["<sha256>.png"]}

So a turn stays a few dozen bytes however large the screenshot was, the
conversations file is not rewritten with megabytes of pixels on every
streaming frame, and the same picture pasted twice is one file. The bytes are
kept as they arrived when they are PNG, JPEG or WebP, and anything else
Pillow can read is written out as PNG. Nothing here scales a picture down:
what a model is shown is decided when it is read (:func:`open_for_model`),
and the stored copy stays the one the reader pasted.

A saved conversation file carries its pictures with it, base64 inside the
JSON, the way it carries its steering vectors (:func:`export_images`,
:func:`import_images`), so a conversation opened on another machine still has
them.

Nothing here imports torch: the conversation code reads it on every load.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

# The largest picture a message takes, in bytes as pasted. A full-resolution
# screenshot of a 6K display is about 30 MB as PNG; anything past this is
# more likely a mistake than a picture.
MAX_IMAGE_BYTES = 40 * 1024 * 1024
# The most pictures one message carries. Each is hundreds to thousands of
# prompt tokens once a model reads it.
MAX_IMAGES_PER_MESSAGE = 8
# What a vision model is shown at most, in pixels: a picture larger than
# this is scaled down, keeping its shape, before its processor sees it. The
# processors cap pictures themselves, but Qwen's cap is 16 megapixels, which
# turns a Retina screenshot into fifteen thousand prompt tokens. A megapixel
# is about a thousand tokens there and still reads small text.
MODEL_PIXEL_BUDGET = 1024 * 1024

# Kept as they are; every other format Pillow reads is converted to PNG.
KEPT_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
IMAGE_NAME = re.compile(r"[0-9a-f]{64}\.(?:png|jpg|webp)")
DIRECTORY_SUFFIX = "-images"


class AttachmentError(ValueError):
    """A picture cannot be stored, found, or read."""


def image_directory() -> Path:
    """Where the pictures live: beside the conversations file."""

    # Lazy to avoid the conversation -> attachments -> library import cycle.
    from chatlab.library import library_path

    target = library_path()
    return target.with_name(f"{target.stem}{DIRECTORY_SUFFIX}")


def is_image_name(value) -> bool:
    return isinstance(value, str) and IMAGE_NAME.fullmatch(value) is not None


def image_names(value) -> list[str]:
    """A turn's ``images`` as a checked list of names, or ``ValueError``."""

    if value is None:
        return []
    if not isinstance(value, list) or not all(is_image_name(name) for name in value):
        raise ValueError("Turn images must be a list of stored picture names.")
    if len(value) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(f"A message holds at most {MAX_IMAGES_PER_MESSAGE} pictures.")
    return list(value)


def image_path(name: str) -> Path:
    if not is_image_name(name):
        raise AttachmentError("That is not the name of a stored picture.")
    return image_directory() / name


def _write(name: str, data: bytes) -> None:
    directory = image_directory()
    path = directory / name
    if path.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".{name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def store_image(data: bytes) -> str:
    """Keep ``data`` as a picture and return the name a turn refers to it by."""

    from PIL import Image, UnidentifiedImageError

    if not data:
        raise AttachmentError("That picture is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise AttachmentError(
            f"That picture is {len(data) / 1024**2:.0f} MB; the most a message "
            f"takes is {MAX_IMAGE_BYTES // 1024**2} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.load()
            if image_format not in KEPT_FORMATS:
                converted = io.BytesIO()
                image.convert("RGBA" if "A" in image.getbands() else "RGB").save(
                    converted, format="PNG"
                )
                data, image_format = converted.getvalue(), "PNG"
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise AttachmentError("That file is not a picture ChatLab can read.") from error
    name = f"{hashlib.sha256(data).hexdigest()}.{KEPT_FORMATS[image_format]}"
    _write(name, data)
    return name


def store_file(path: str | Path) -> str:
    """:func:`store_image` for a file on disk, such as a browser upload."""

    path = Path(path)
    try:
        size = path.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise AttachmentError(
                f"That picture is {size / 1024**2:.0f} MB; the most a message "
                f"takes is {MAX_IMAGE_BYTES // 1024**2} MB."
            )
        data = path.read_bytes()
    except OSError as error:
        raise AttachmentError("That picture could not be read.") from error
    return store_image(data)


def read_bytes(name: str) -> bytes:
    """A stored picture's bytes, checked against the name they are kept under."""

    path = image_path(name)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AttachmentError(
            "A picture in this conversation is no longer on this machine."
        ) from error
    if hashlib.sha256(data).hexdigest() != name.split(".", 1)[0]:
        raise AttachmentError("A stored picture failed its integrity check.")
    return data


def open_for_model(name: str, budget: int = MODEL_PIXEL_BUDGET):
    """A stored picture as an RGB ``PIL.Image``, scaled down to ``budget`` pixels.

    Turned upright first, because a phone photo stores its orientation beside
    the pixels and a model shown the raw pixels sees it on its side.
    """

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(read_bytes(name))) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            # A transparent screenshot region is white to the reader, not black.
            backdrop = Image.new("RGB", image.size, "white")
            backdrop.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
            image = backdrop
        else:
            image = image.convert("RGB")
    width, height = image.size
    if width * height > budget:
        scale = (budget / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def size_of(name: str) -> tuple[int, int] | None:
    """A stored picture's width and height, or ``None`` when it cannot be read."""

    from PIL import Image

    try:
        with Image.open(image_path(name)) as image:
            return image.size
    except (OSError, AttachmentError, ValueError):
        return None


# How long an unreferenced picture is kept before a startup sweep removes it.
# A picture is stored the moment it is pasted, before any conversation names
# it, and another window can be holding one in its message box; a day leaves
# every draft in progress alone.
UNREFERENCED_GRACE_SECONDS = 24 * 60 * 60


def prune_unreferenced(sources: Iterable[Path], grace: float = UNREFERENCED_GRACE_SECONDS) -> int:
    """Remove pictures nothing in ``sources`` names any more; return how many.

    A picture taken off the message box, or left behind by a deleted
    conversation, would otherwise be kept forever, at up to
    :data:`MAX_IMAGE_BYTES` each. ``sources`` are the files that can name one
    (the conversations file, saved experiments), searched as text for
    picture names rather than parsed, so a file of any shape keeps what it
    mentions. A source that exists but cannot be read stops the sweep:
    deleting against a partial view of what is referenced would delete what
    it could not see.
    """

    import time

    referenced: set[str] = set()
    for source in sources:
        try:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            continue
        except OSError:
            return 0
        referenced.update(IMAGE_NAME.findall(text))
    directory = image_directory()
    if not directory.is_dir():
        return 0
    cutoff = time.time() - grace
    removed = 0
    for path in directory.iterdir():
        if not is_image_name(path.name) or path.name in referenced:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def names_in(turns: Iterable[dict] | None) -> list[str]:
    """Every picture ``turns`` refer to, once each, in the order they appear."""

    seen: dict[str, None] = {}
    for turn in turns or []:
        for name in turn.get("images") or []:
            seen.setdefault(name, None)
    return list(seen)


def export_images(names: Iterable[str]) -> dict[str, str]:
    """Each picture once, base64, for a conversation file that travels."""

    return {name: base64.b64encode(read_bytes(name)).decode("ascii") for name in dict.fromkeys(names)}


def import_images(names: Iterable[str], assets) -> None:
    """Install the pictures a conversation file carries, checking each one.

    A picture already on this machine is left alone. One that is neither here
    nor in the file refuses the import, since the turn would then point at
    nothing.
    """

    assets = assets if isinstance(assets, dict) else {}
    for name in dict.fromkeys(names):
        if image_path(name).is_file():
            continue
        encoded = assets.get(name)
        if not isinstance(encoded, str):
            raise AttachmentError("The conversation is missing an embedded picture.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise AttachmentError("An embedded picture is not valid base64.") from error
        if hashlib.sha256(data).hexdigest() != name.split(".", 1)[0]:
            raise AttachmentError("An embedded picture does not match its name.")
        _write(name, data)
