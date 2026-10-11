"""Generate the settings/*.json fixtures.

Format: settings.json (no format marker; every field the writing version
knows is written, sorted, and keys it does not know are carried through).
One file per era, each written by that era's own ``settings.update``, the
function every control calls, after ``settings.load``:

  - 01-first-3db3f4e.json: 3db3f4e "Keep every setting in one JSON file
    between sessions (#31)", the commit that introduced the file: sampling,
    prompts, seed, colour scale, prefill limit and MPS memory fraction only.
  - 02-themes-88ec738.json: 88ec738 "Let the reader choose light or dark, not
    only the system (#105)": adds thinking mode, writing suggestions, weight
    precision, image settings, enabled extensions, theme and appearance.
  - 03-current-cfd9fbd.json: cfd9fbd "Release ChatLab 0.40.0" (settings writer
    last changed in e7f7827, #162): adds skip_top_below; the file it saved
    over already held a key from a newer version (``future_setting``), which
    the writer carries through.

Commands used:

    git worktree add --detach /tmp/chatlab-at-3db3f4e 3db3f4e
    cd /tmp/chatlab-at-3db3f4e && PYTHONPATH=. <python> \
        <repo>/tests/fixtures/saved/settings/generators/settings_file.py \
        <repo>/tests/fixtures/saved/settings/01-first-3db3f4e.json
    git worktree add --detach /tmp/chatlab-at-88ec738 88ec738
    cd /tmp/chatlab-at-88ec738 && PYTHONPATH=. <python> \
        <repo>/tests/fixtures/saved/settings/generators/settings_file.py \
        <repo>/tests/fixtures/saved/settings/02-themes-88ec738.json
    # from a checkout of cfd9fbd:
    PYTHONPATH=. <python> tests/fixtures/saved/settings/generators/settings_file.py \
        --newer-key tests/fixtures/saved/settings/03-current-cfd9fbd.json

with <python> = /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python.
The values chosen are the same for every era, filtered to the fields that
era's ``Settings`` has; the extension IDs are the ones that existed then.
"""

import json
import os
import sys
from dataclasses import fields
from pathlib import Path

newer_key = "--newer-key" in sys.argv
target = Path([arg for arg in sys.argv[1:] if not arg.startswith("--")][0]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    raise SystemExit(f"{target} exists; fixtures are never regenerated.")
if newer_key:
    target.write_text(json.dumps({"future_setting": {"nested": [1, 2], "on": True}}), encoding="utf-8")
os.environ["CHATLAB_SETTINGS_PATH"] = str(target)

try:
    from chatlab import settings
    from chatlab.extensions.registry import CATALOGUE
    extension_ids = {spec.id if hasattr(spec, "id") else spec[0] for spec in CATALOGUE}
except ImportError:
    import settings  # top-level module before e7f7827
    try:
        from extensions.registry import CATALOGUE
        extension_ids = {spec.id if hasattr(spec, "id") else spec[0] for spec in CATALOGUE}
    except ImportError:
        extension_ids = set()

CHOSEN = {
    "model_id": "Qwen/Qwen3-0.6B",
    "system_prompt": "Be brief.\nUse British spelling.",
    "assistant_prefill": "Answer:",
    "keep_reasoning": True,
    "enter_sends": False,
    "temperature": 0.35,
    "top_p": 0.85,
    "top_k": 12,
    "skip_top_below": 0.1,
    "max_new_tokens": 600,
    "seed": 2024,
    "randomize_seed": False,
    "analyze_prompt": False,
    "color_scale": "Surprise",
    "prefill_token_limit": 4096,
    "mps_memory_fraction": 1.5,
    "thinking_mode": "off",
    "writing_suggestions": False,
    "theme": "ember",
    "appearance": "dark",
    "weight_precision": "4-bit",
    "image_negative_prompt": "blurry",
    "image_steps": 20,
    "image_guidance": 5.0,
    "image_size": 768,
    "image_seed": 7,
    "image_randomize_seed": False,
    "image_record_attention": False,
    "enabled_extensions": [
        name for name in ("maze_experiments", "hangman", "probes", "circuits")
        if name in extension_ids
    ],
}

known = {field.name for field in fields(settings.Settings)}
settings.load()
saved = settings.update(**{key: value for key, value in CHOSEN.items() if key in known})
for key in known & CHOSEN.keys():
    value = getattr(saved, key)
    expected = CHOSEN[key]
    assert (list(value) if isinstance(value, tuple) else value) == expected, (key, value, expected)
print(target)
