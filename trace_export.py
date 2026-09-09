"""Serialization helpers for generated-token metric traces."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
TOKEN_COLUMNS = [
    "position",
    "token_id",
    "text",
    "display_text",
    "category",
    "raw_rank",
    "raw_probability",
    "sampling_probability",
    "surprise_bits",
    "probability_mass_above",
]


def build_trace(
    *,
    model_id: str | None,
    messages: list[dict],
    response: str,
    sampling: dict,
    metrics: list[dict],
    generated_at: str | None = None,
) -> dict:
    """Build a self-contained trace for one generated response."""

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "messages": messages,
        "response": response,
        "sampling": sampling,
        "token_count": len(metrics),
        "tokens": metrics,
    }


def trace_to_json(trace: dict) -> str:
    """Serialize a trace without escaping token text or conversation content."""

    return json.dumps(trace, ensure_ascii=False, indent=2) + "\n"


METADATA_COLUMNS = ["schema_version", "generated_at", "model_id"]
SAMPLING_COLUMNS = ["temperature", "top_p", "top_k", "max_new_tokens", "seed"]
CANDIDATE_FIELDS = ("token_id", "text", "probability")


def _token_rows(trace: dict):
    """One row per generated token, with the trace's own columns on each."""

    sampling = trace.get("sampling") or {}
    metadata = {column: trace.get(column) for column in METADATA_COLUMNS}
    generation_settings = {column: sampling.get(column) for column in SAMPLING_COLUMNS}
    for token in trace.get("tokens") or []:
        row = metadata | generation_settings
        row.update({column: token.get(column) for column in TOKEN_COLUMNS})
        for index, candidate in enumerate(token.get("top_candidates") or [], start=1):
            for field in CANDIDATE_FIELDS:
                row[f"candidate_{index}_{field}"] = candidate.get(field)
        yield row


def candidate_width(traces: Sequence[dict]) -> int:
    """How many alternatives the widest token in ``traces`` recorded.

    This is what decides the header, so a caller adding to a table it has
    already written asks for it before deciding whether the table can be
    appended to or has to be written again.
    """

    return max(
        (
            len(token.get("top_candidates") or [])
            for trace in traces
            for token in (trace.get("tokens") or [])
        ),
        default=0,
    )


def _rows_to_csv(
    traces: list[dict],
    *,
    indexes: Sequence[int] | None = None,
    width: int | None = None,
    header: bool = True,
) -> str:
    """Write every trace's tokens into one table.

    The candidate columns are as wide as the widest token in any of the
    traces, so several responses can share a header even though one of them
    ran with a smaller top-k than another. ``width`` fixes that number
    instead, which is how rows are written to join a table already on disk.

    ``indexes`` gives each trace the prompt number it belongs to, written into
    a prompt_index column. The numbers are the caller's because a batch can
    drop a trace it failed to produce, and the ones that follow must keep the
    positions they were asked in. ``None`` leaves the column out, which is what
    a single trace exported on its own wants.

    ``header`` writes the column names first. Rows appended to an existing
    table leave it out, and are the caller's to keep under the same width.
    """

    if indexes is not None and len(indexes) != len(traces):
        raise ValueError("Each trace needs exactly one prompt index.")

    candidate_count = candidate_width(traces) if width is None else width
    candidate_columns = [
        f"candidate_{index}_{field}"
        for index in range(1, candidate_count + 1)
        for field in CANDIDATE_FIELDS
    ]
    columns = (
        # A batch's rows carry the prompt they answered and whether that
        # answer was cut short, because the table is read on its own: a
        # reader who never opens the traces would otherwise take a stopped
        # answer for a whole one.
        (["prompt_index", "stopped"] if indexes is not None else [])
        + METADATA_COLUMNS
        + SAMPLING_COLUMNS
        + TOKEN_COLUMNS
        + candidate_columns
    )

    output = io.StringIO(newline="")
    # A token can carry more alternatives than the fixed width holds, and the
    # columns for them do not exist in the table being joined. Dropping them
    # keeps the row on the shape its header promises; the caller widens the
    # table instead when it wants them, which is what BatchTable does.
    writer = csv.DictWriter(
        output, fieldnames=columns, lineterminator="\n", extrasaction="ignore"
    )
    if header:
        writer.writeheader()
    numbers = list(indexes) if indexes is not None else [None] * len(traces)
    for number, trace in zip(numbers, traces):
        stopped = bool((trace.get("sampling") or {}).get("stopped"))
        for row in _token_rows(trace):
            writer.writerow(
                row
                if number is None
                else row | {"prompt_index": number, "stopped": stopped}
            )

    return output.getvalue()


def trace_to_csv(trace: dict) -> str:
    """Flatten a trace into one row per generated token.

    Generation metadata and sampling settings are repeated on every row so the
    CSV stays useful when traces from several runs are concatenated. Alternative
    candidates use numbered columns, preserving the complete candidate list.
    """

    return _rows_to_csv([trace])


def traces_to_csv(
    traces: list[dict],
    indexes: Sequence[int] | None = None,
    *,
    width: int | None = None,
    header: bool = True,
) -> str:
    """Flatten a whole batch into one table, numbered by the prompt it came from.

    A batch is read as a group - the same question asked twenty ways, one row
    per token of every answer - so the prompt each row belongs to has to be a
    column rather than a file name.

    ``indexes`` says which prompt each trace answered. Without it the traces
    are numbered in the order they are given, which is only the same thing
    when every prompt produced a trace.

    ``width`` and ``header`` are for writing rows that join a table already on
    disk: the columns are the ones that table has, and the names are not
    repeated part way down it.
    """

    traces = list(traces)
    if indexes is None:
        indexes = range(1, len(traces) + 1)
    return _rows_to_csv(traces, indexes=list(indexes), width=width, header=header)


def append_private_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Add ``text`` to the end of ``path``, keeping it owner-only.

    The mode is given to ``os.open()`` for the case where the file is not
    there yet, and settled on the descriptor either way, so a file created by
    an append is as private as one written whole; see write_private_text() for
    why the mode cannot wait until afterwards.
    """

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "a", encoding="utf-8", newline=newline)
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        handle.write(text)


def write_private_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """Write ``text`` to ``path`` as a file only its owner can read.

    ``Path.write_text()`` creates the file with whatever the process umask
    allows - usually 0644 - and puts every byte of it on disk before a
    following ``chmod`` can narrow it. Exports and saved transcripts land in
    shared directories, so another account on the machine can open the file
    during that window and read it. Creating the file 0600 and settling its
    mode on the descriptor, before anything is written into it, closes the
    window. The mode is set explicitly rather than left to ``os.open()``
    because the umask can only take bits away from the mode it is given, so a
    strict one would otherwise leave the owner unable to read their own file.
    """

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline=newline)
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        handle.write(text)


def write_trace_export(trace: dict, file_format: str) -> str | None:
    """Write a browser-downloadable export and return its path."""

    if not trace or not trace.get("tokens"):
        return None
    serializers = {"json": trace_to_json, "csv": trace_to_csv}
    try:
        serialize = serializers[file_format]
    except KeyError as error:
        raise ValueError(f"Unsupported trace export format: {file_format}") from error

    export_dir = Path(tempfile.mkdtemp(prefix="chatlab-"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"olmo-metric-trace-{timestamp}-{uuid.uuid4().hex[:8]}.{file_format}"
    path = export_dir / filename
    write_private_text(path, serialize(trace), newline="")
    return str(path)
