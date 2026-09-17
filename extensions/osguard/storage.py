"""Private, atomic checkpoints; create the extension directory only on write."""
import json
from uuid import uuid4

from extension_api import write_private_text


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid4().hex}.tmp")
    try:
        write_private_text(temporary, json.dumps(value, ensure_ascii=False, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
