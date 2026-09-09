"""Reading a list of prompts, and writing what running them produced.

A batch is a plain list of prompts run one after another, each in a
conversation of its own. Nothing here talks to a model or to the interface:
this module turns text or a file into prompts, and turns finished traces into
files on disk.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from trace_export import (
    append_private_text,
    candidate_width,
    trace_to_json,
    traces_to_csv,
    write_private_text,
)


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

    A blank line written on Windows carries a carriage return, and a file
    holding those would otherwise be read as one long prompt: the split
    takes them either way. What is inside a prompt is left as the file
    wrote it, since that is what the model is being asked.
    """

    blocks = re.split(r"\r?\n[ \t]*\r?\n", text or "")
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


def batch_trace_path(directory: Path, index: int) -> Path:
    """Where the trace for the prompt at ``index`` is written."""

    return Path(directory) / f"prompt-{index:03d}.json"


def read_batch_trace(directory: Path, index: int) -> dict:
    """Read back a trace this run wrote."""

    return json.loads(batch_trace_path(directory, index).read_text(encoding="utf-8"))


def write_batch_trace(trace: dict, directory: Path, index: int) -> str:
    """Write one prompt's trace, named for its place in the run.

    ``index`` is the prompt's position in the box, not its position among the
    traces: a prompt that failed leaves a gap in the numbering rather than
    letting the next one take its name.
    """

    path = batch_trace_path(directory, index)
    write_private_text(path, trace_to_json(trace))
    return str(path)


def write_batch_csv(
    traces: list[dict], directory: Path, indexes: Sequence[int] | None = None
) -> str:
    """Write the table covering every prompt given, from scratch.

    ``indexes`` names the prompt each trace answered, so the numbers in the
    table match the trace file names even when a prompt in between failed.

    A run adds to its table one prompt at a time; see :class:`BatchTable`.
    """

    path = Path(directory) / BATCH_CSV_NAME
    write_private_text(path, traces_to_csv(traces, indexes), newline="")
    return str(path)


class BatchTable:
    """The table for one run, extended as each prompt finishes.

    The table is on disk from the first prompt onwards rather than written at
    the end, because a run that is stopped half way through is still a run and
    what it measured should be downloadable. Each prompt adds its own rows
    instead of the whole table being written again: rewriting would serialize
    every token of every earlier prompt once per prompt, which on a
    hundred-prompt batch is the first prompt written a hundred times, and all
    of it between two generations while the model sits idle.

    The exception is the candidate columns. Their number is the header's, and
    a prompt whose tokens carry more alternatives than the header holds cannot
    be appended without the rows disagreeing with it, so that prompt widens
    the table and the file is written again. Every token a run measures
    carries the same number of alternatives, so this is close to unreachable
    in practice, and the rewrite reads the traces back from the run's own
    files rather than keeping them. What a batch measures is the largest
    thing it produces - thousands of tokens a prompt, each with its
    alternatives - and holding all of it until the run ends would take
    hundreds of megabytes from the machine the model is running on.

    The prompt numbers are kept, since they are what the rows are written
    under and cost nothing.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / BATCH_CSV_NAME
        self.indexes: list[int] = []
        self.width = 0

    def add(self, trace: dict, index: int) -> str:
        """Put one prompt's tokens in the table, and return where it is.

        The prompt's own trace file must already be written: a rewrite reads
        every prompt back from those files, this one included.
        """

        self.indexes.append(int(index))
        width = candidate_width([trace])
        if width > self.width or len(self.indexes) == 1:
            self.width = max(width, self.width)
            traces = [read_batch_trace(self.directory, number) for number in self.indexes]
            write_private_text(
                self.path,
                traces_to_csv(traces, self.indexes, width=self.width),
                newline="",
            )
        else:
            append_private_text(
                self.path,
                traces_to_csv([trace], [index], width=self.width, header=False),
                newline="",
            )
        return str(self.path)
