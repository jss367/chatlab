"""Reading a list of prompts, and writing what running them produced.

A batch is a plain list of prompts run one after another, each in a
conversation of its own. Nothing here talks to a model or to the interface:
this module turns text or a file into prompts, and turns finished traces into
files on disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trace_export import trace_to_json, traces_to_csv, write_private_text


# The keys a JSON prompt may carry, in the order they are looked for. A file
# written for another tool usually names the field one of these three, and a
# bare string is accepted as well, so a list of prompts needs no wrapping.
PROMPT_KEYS = ("prompt", "text", "content")


# The combined table lives beside the per-prompt traces under a directory of
# the run's own, so its name does not have to be unique.
BATCH_CSV_NAME = "prompts.csv"


def parse_prompts(text: str) -> list[str]:
    """Split written text into prompts on blank lines.

    A prompt worth measuring is often several lines - a passage, then the
    question about it - so one prompt per line would make most of them
    unwritable. A blank line is the separator a reader can see in the box.
    """

    blocks = re.split(r"\n[ \t]*\n", text or "")
    return [block.strip() for block in blocks if block.strip()]


def prompts_to_text(prompts: list[str]) -> str:
    """The box's contents for ``prompts``, ready for :func:`parse_prompts`."""

    return "\n\n".join(prompt.strip() for prompt in prompts if prompt.strip())


def _prompt_from(entry, where: str) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in PROMPT_KEYS:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(
            f"{where} has no prompt: expected one of "
            f"{', '.join(PROMPT_KEYS)}, or a plain string."
        )
    raise ValueError(f"{where} is a {type(entry).__name__}, not a prompt.")


def parse_prompt_file(path) -> list[str]:
    """Prompts from a file, read the way its extension says to read it.

    ``.jsonl`` is one prompt per line, ``.json`` is a list of them, and
    anything else is text with a blank line between prompts. The three cover
    what a prompt set arrives as; a file that is none of them fails by name
    and line, because a set silently read as one long prompt would run once
    and look like a model problem.
    """

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        prompts = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {number} is not JSON: {error.msg}.") from error
            prompt = _prompt_from(entry, f"Line {number}")
            if prompt:
                prompts.append(prompt)
        return prompts

    if suffix == ".json":
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Not JSON: {error.msg}.") from error
        if isinstance(loaded, dict):
            loaded = loaded.get("prompts")
        if not isinstance(loaded, list):
            raise ValueError(
                "Expected a list of prompts, or an object with a prompts list."
            )
        prompts = [
            _prompt_from(entry, f"Item {number}")
            for number, entry in enumerate(loaded, start=1)
        ]
        return [prompt for prompt in prompts if prompt]

    return parse_prompts(raw)


def write_batch_trace(trace: dict, directory: Path, index: int) -> str:
    """Write one prompt's trace, named for its place in the run."""

    path = Path(directory) / f"prompt-{index:03d}.json"
    write_private_text(path, trace_to_json(trace))
    return str(path)


def write_batch_csv(traces: list[dict], directory: Path) -> str:
    """Rewrite the table covering every prompt run so far.

    Rewritten after each prompt rather than once at the end, because a run
    that is stopped half way through is still a run: the table on disk always
    describes the prompts that have finished.
    """

    path = Path(directory) / BATCH_CSV_NAME
    write_private_text(path, traces_to_csv(traces), newline="")
    return str(path)
