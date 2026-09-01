"""Serialization helpers for generated-token metric traces."""

from __future__ import annotations

import csv
import io
import json
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


def write_trace_export(trace: dict, file_format: str) -> str | None:
    """Write a browser-downloadable export and return its path."""

    if not trace or not trace.get("tokens"):
        return None
    serializers = {"json": trace_to_json, "csv": trace_to_csv}
    try:
        serialize = serializers[file_format]
    except KeyError as error:
        raise ValueError(f"Unsupported trace export format: {file_format}") from error

    export_dir = Path(tempfile.gettempdir()) / "olmo-token-explorer"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"olmo-metric-trace-{timestamp}-{uuid.uuid4().hex[:8]}.{file_format}"
    path = export_dir / filename
    path.write_text(serialize(trace), encoding="utf-8", newline="")
    return str(path)
