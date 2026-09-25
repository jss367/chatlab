"""Helpers for the tests that read the built Gradio page.

A test reaches a handler the way Gradio does, through the listener
``demo.fns`` holds for it, so it gets the inputs, outputs and queue the page
really wired rather than the bare function. Listeners are found by the name
of the function they call, because many handlers are closures the page
builds, with no module attribute to compare against.

The stylesheet is read the same way: as rules, selector to declarations,
rather than as text, so an assertion about what a rule says holds however
the rule happens to be laid out.
"""

import re
from dataclasses import dataclass
from functools import lru_cache


def _name(listener) -> str | None:
    # A listener that only runs JavaScript has no function; a partial has no name.
    return getattr(listener.fn, "__name__", None)


def listeners_named(demo, name: str) -> list:
    """Every listener on ``demo`` whose function is called ``name``, in wiring order."""

    return [listener for listener in demo.fns.values() if _name(listener) == name]


def listener_named(demo, name: str):
    """The first listener on ``demo`` whose function is called ``name``."""

    for found in demo.fns.values():
        if _name(found) == name:
            return found
    raise LookupError(f"no listener calls {name!r}")


def listeners_by_name(demo) -> dict:
    """Each named listener on ``demo``, keyed by its function's name.

    Where two listeners call functions of one name, the later-wired one is
    kept, as a dict built over ``demo.fns`` keeps it.
    """

    return {_name(found): found for found in demo.fns.values() if _name(found)}


def handlers_by_name(demo) -> dict:
    """The functions behind :func:`listeners_by_name`, to call directly."""

    return {name: found.fn for name, found in listeners_by_name(demo).items()}


@dataclass(frozen=True)
class CssRule:
    """One rule as written: the at-rule it sits in, its selectors, its declarations.

    ``media`` is the condition of the ``@media`` block around the rule, such
    as ``"(max-width: 850px)"``, or None at the top level; any other block
    at-rule (``@keyframes x``) is kept whole. ``declarations`` are in written
    order, so a property given twice appears twice.
    """

    media: str | None
    selectors: tuple[str, ...]
    declarations: tuple[tuple[str, str], ...]


def _top_level(text: str, separators: str):
    """Split ``text`` at ``separators`` outside quotes, brackets and parentheses.

    Selectors carry quoted attribute values (``[data-testid*="· won't fit"]``)
    and values carry ``url("data:...")``, either of which can hold the
    characters a naive split would cut at.
    """

    parts, start, depth, quote = [], 0, 0, None
    for index, character in enumerate(text):
        if quote:
            if character == quote and text[index - 1] != "\\":
                quote = None
        elif character in "\"'":
            quote = character
        elif character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character in separators and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def normalize_selector(selector: str) -> str:
    """One spelling of a selector: single spaces, and one either side of a combinator."""

    pieces, quote, depth = [], None, 0
    for index, character in enumerate(selector):
        if quote:
            if character == quote and selector[index - 1] != "\\":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character in ">+~" and depth == 0:
            pieces.append(f" {character} ")
            continue
        pieces.append(character)
    return " ".join("".join(pieces).split())


def normalize_media(condition: str) -> str:
    """One spelling of a media condition: ``(max-width: 850px)``, however it was spaced."""

    condition = " ".join(condition.split())
    condition = re.sub(r"\s*:\s*", ": ", condition)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", condition))


def _normalize_value(value: str) -> str:
    # Commas get one space after them outside quoted strings, so
    # "clamp(1px,2vw)" and "clamp(1px, 2vw)" read alike and a data URL does not.
    pieces = re.split(r"""("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""", value)
    value = "".join(
        piece if index % 2 else re.sub(r"\s*,\s*", ", ", piece)
        for index, piece in enumerate(pieces)
    )
    value = " ".join(value.split())
    # "none!important" and "none ! important" are the same declaration.
    return re.sub(r"\s*!\s*important$", " !important", value, flags=re.I)


def css_declarations(text: str) -> dict[str, str]:
    """The declarations in a block's body, ``"a: 1; b: 2"``, as a dict.

    Property names are lowercased except custom properties, which CSS treats
    as case-sensitive. A property given twice keeps its later value, as the
    cascade would.
    """

    return dict(_declaration_pairs(text))


def _declaration_pairs(text: str):
    for declaration in _top_level(text, ";"):
        name, colon, value = declaration.partition(":")
        name = name.strip()
        if not colon or not name:
            continue
        yield (name if name.startswith("--") else name.lower()), _normalize_value(value)


def _find_close(css: str, start: int) -> int:
    """The index of the brace closing the block that opens just before ``start``."""

    depth, quote = 1, None
    for index in range(start, len(css)):
        character = css[index]
        if quote:
            if character == quote and css[index - 1] != "\\":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced braces in the stylesheet")


def _parse(css: str, media: str | None, rules: list) -> None:
    position = 0
    while True:
        opening = _top_level_brace(css, position)
        if opening is None:
            return
        closing = _find_close(css, opening + 1)
        # A statement at-rule (@import ...;) ends before the prelude starts.
        prelude = _top_level(css[position:opening], ";")[-1].strip()
        body = css[opening + 1 : closing]
        if prelude.startswith("@media"):
            condition = normalize_media(prelude.removeprefix("@media"))
            _parse(body, condition if media is None else f"{media} and {condition}", rules)
        elif prelude.startswith("@"):
            _parse(body, " ".join(prelude.split()), rules)
        else:
            rules.append(CssRule(
                media,
                tuple(normalize_selector(s) for s in _top_level(prelude, ",") if s.strip()),
                tuple(_declaration_pairs(body)),
            ))
        position = closing + 1


def _top_level_brace(css: str, start: int) -> int | None:
    quote = None
    for index in range(start, len(css)):
        character = css[index]
        if quote:
            if character == quote and css[index - 1] != "\\":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "{":
            return index
    return None


@lru_cache(maxsize=16)
def css_rule_list(css: str) -> tuple[CssRule, ...]:
    """Every rule in ``css``, in written order, with comments stripped.

    Cached: the app's stylesheet is some eighty kilobytes, and a test that
    checks every icon asks for it once per icon.
    """

    rules: list[CssRule] = []
    _parse(re.sub(r"/\*.*?\*/", "", css, flags=re.S), None, rules)
    return tuple(rules)


@lru_cache(maxsize=16)
def css_rules(css: str) -> dict[tuple[str | None, str], dict[str, str]]:
    """What ``css`` declares for each selector, keyed by ``(media, selector)``.

    A selector list gives each of its selectors the whole block, and rules
    for the same selector under the same condition are merged in written
    order, later values winning, which is how the cascade reads them.
    """

    merged: dict[tuple[str | None, str], dict[str, str]] = {}
    for rule in css_rule_list(css):
        for selector in rule.selectors:
            merged.setdefault((rule.media, selector), {}).update(rule.declarations)
    return merged


def css_rule(css: str, selector: str, media: str | None = None) -> dict[str, str]:
    """The declarations ``css`` gives ``selector`` under ``media``; empty if none.

    Empty rather than an error so that an assertion on a missing rule fails
    by showing what it expected against nothing.
    """

    media = None if media is None else normalize_media(media)
    return dict(css_rules(css).get((media, normalize_selector(selector)), {}))


def css_selectors(css: str) -> set[str]:
    """Every selector ``css`` writes a rule for, under any condition."""

    return {selector for rule in css_rule_list(css) for selector in rule.selectors}


def css_media(css: str) -> set[str]:
    """The conditions of the ``@media`` blocks in ``css``."""

    return {
        rule.media
        for rule in css_rule_list(css)
        if rule.media is not None and not rule.media.startswith("@")
    }
