"""Serialization helpers for generated-token metric traces."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import uuid
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


def trace_to_csv(trace: dict) -> str:
    """Flatten a trace into one row per generated token.

    Generation metadata and sampling settings are repeated on every row so the
    CSV stays useful when traces from several runs are concatenated. Alternative
    candidates use numbered columns, preserving the complete candidate list.
    """

    tokens = trace.get("tokens") or []
    sampling = trace.get("sampling") or {}
    candidate_count = max(
        (len(token.get("top_candidates") or []) for token in tokens), default=0
    )
    metadata_columns = ["schema_version", "generated_at", "model_id"]
    sampling_columns = [
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "seed",
    ]
    candidate_columns = [
        f"candidate_{index}_{field}"
        for index in range(1, candidate_count + 1)
        for field in ("token_id", "text", "probability")
    ]
    columns = metadata_columns + sampling_columns + TOKEN_COLUMNS + candidate_columns

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()

    metadata = {column: trace.get(column) for column in metadata_columns}
    generation_settings = {column: sampling.get(column) for column in sampling_columns}
    for token in tokens:
        row = metadata | generation_settings
        row.update({column: token.get(column) for column in TOKEN_COLUMNS})
        for index, candidate in enumerate(token.get("top_candidates") or [], start=1):
            for field in ("token_id", "text", "probability"):
                row[f"candidate_{index}_{field}"] = candidate.get(field)
        writer.writerow(row)

    return output.getvalue()


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
