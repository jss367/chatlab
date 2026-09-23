"""Durable, independently inspectable runs and token annotations."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import compare
import library
from trace_export import write_private_text
from version import __version__

_LOCK = threading.RLock()
SESSION_ID = uuid4().hex


def directory() -> Path:
    return library.library_path().parent / "experiments"


def _path(identifier: str) -> Path:
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Select a saved experiment first.")
    return directory() / f"{identifier}.json"


def _write(document: dict) -> None:
    path = _path(document["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        write_private_text(staged, json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def save(run: dict, title: str = "") -> dict:
    if not run or not run.get("metrics"):
        raise ValueError("Generate or measure a run before saving it.")
    recorded = copy.deepcopy(run)
    recorded["settings"] = compare._portable_settings(recorded.get("settings") or {})
    document = {
        "schema_version": 1,
        "id": uuid4().hex,
        "title": title.strip() or " ".join((run.get("prompt") or run.get("text") or "Saved run").split())[:80],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": __version__,
        "run": recorded,
        "bookmarks": {},
        "inspections": [],
    }
    with _LOCK:
        _write(document)
    return document


def read(identifier: str) -> dict:
    document = json.loads(_path(identifier).read_text())
    if document.get("schema_version") != 1 or document.get("id") != identifier:
        raise ValueError("This experiment file has an unsupported format.")
    if not isinstance(document.get("run", {}).get("metrics"), list):
        raise ValueError("This experiment has no token measurements.")
    return document


def search(query: str = "") -> list[dict]:
    found = []
    query = (query or "").casefold()
    for path in directory().glob("*.json"):
        try:
            item = read(path.stem)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        run = item["run"]
        searchable = " ".join(str(value) for value in (
            item["title"], run.get("model_id", ""), run.get("prompt", ""),
            run.get("text", ""), *item.get("bookmarks", {}).values(),
        ))
        if query in searchable.casefold():
            found.append(item)
    return sorted(found, key=lambda item: (item["created_at"], item["id"]), reverse=True)


def bookmark(identifier: str, index: int, note: str, *, remove=False) -> dict:
    with _LOCK:
        document = read(identifier)
        if not 0 <= index < len(document["run"]["metrics"]):
            raise ValueError("Select a token in this experiment first.")
        marks = document.setdefault("bookmarks", {})
        if remove:
            marks.pop(str(index), None)
        else:
            marks[str(index)] = note or ""
        _write(document)
        return document


def save_inspection(identifier: str, insight: dict, target: dict, context) -> dict:
    """Only attach an inspection whose exact input belongs to this run."""
    with _LOCK:
        document = read(identifier)
        run = document["run"]
        if not insight or not target or not context or len(context) < 3:
            raise ValueError("Inspect a token in the original run first.")
        if (target.get("strip") != "response" or target.get("generation") != context[0]
                or run.get("session_id") != SESSION_ID
                or insight.get("saved_session") != SESSION_ID
                or target.get("generation") != run.get("metrics_generation")
                or insight.get("saved_target") != target
                or context[2] != run.get("load_id")
                or list(context[1]) != run.get("context_ids")):
            raise ValueError("This inspection does not belong to the saved response.")
        index = int(target["index"])
        if not 0 <= index < len(run["metrics"]):
            raise ValueError("This token is not in the saved response.")
        snapshot = copy.deepcopy(insight)
        snapshot.pop("inspection_controls", None)
        document.setdefault("inspections", []).append({"token_index": index, "insight": snapshot})
        _write(document)
        return document


def from_trace(trace: dict, context=None) -> dict:
    if not trace or not trace.get("tokens"):
        raise ValueError("Finish a chat response before saving it.")
    messages = trace.get("messages") or []
    sampling = copy.deepcopy(trace.get("sampling") or {})
    sampling["system_prompt"] = next((m["content"] for m in messages if m["role"] == "system"), "")
    recorded = trace.get("run_context") or {}
    return {
        "kind": compare.REPLY,
        "model_id": trace.get("model_id"),
        "load_id": recorded.get("load_id", context[2] if context and len(context) > 2 else None),
        "context_ids": recorded.get("context_ids", list(context[1]) if context and len(context) > 1 else []),
        "metrics_generation": recorded.get("metrics_generation", context[0] if context else None),
        "tokenizer": recorded.get("tokenizer"),
        "session_id": recorded.get("session_id"),
        "decoded": recorded.get("decoded", trace.get("response", "")),
        "token_ends": recorded.get("token_ends", []),
        "prompt": next((m["content"] for m in reversed(messages) if m["role"] == "user"), ""),
        "messages": messages,
        "text": trace.get("response", ""),
        "metrics": trace["tokens"],
        "prompt_metrics": copy.deepcopy(trace.get("prompt_tokens") or []),
        "settings": sampling,
        "device_name": recorded.get("device_name", sampling.get("device_name")),
        "precision": recorded.get("precision", sampling.get("precision")),
        "generated_at": trace.get("generated_at"),
    }


def save_timeline_inspection(identifier: str, position: int, insight: dict) -> dict:
    """Store bounded, recomputed readouts; never retain raw activation tensors."""
    with _LOCK:
        document = read(identifier)
        run = document["run"]
        ids = run.get("context_ids", []) + [m["token_id"] for m in run["metrics"]]
        if (not 0 <= position < len(ids) or insight.get("index") != position
                or insight.get("token_id") != ids[position]):
            raise ValueError("The inspection does not match this timeline position.")
        records = document.setdefault("timeline_inspections", {})
        key = f"{insight.get('kind', 'logit')}:{position}"
        if key not in records and len(records) >= 64:
            raise ValueError("This experiment already has 64 timeline readouts. Save a new experiment to capture more.")
        records[key] = {
            "source": "recomputed", "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "insight": copy.deepcopy(insight),
        }
        if len(json.dumps(records).encode("utf-8")) > 32 * 1024 * 1024:
            raise ValueError("Timeline readouts would exceed 32 MB. Capture without attention or use a shorter context.")
        _write(document)
        return document


def ranked_tokens(run: dict, mode: str) -> list[int]:
    key = "top1_margin" if mode == "Closest alternatives" else "surprise_bits"
    candidates = []
    for index, metric in enumerate((run or {}).get("metrics", [])):
        value = metric.get(key)
        if metric.get("scored", True) and isinstance(value, (float, int)) and math.isfinite(value):
            candidates.append((float(value), index))
    candidates.sort(key=lambda pair: (pair[0] if key == "top1_margin" else -pair[0], pair[1]))
    return [index for _, index in candidates]
