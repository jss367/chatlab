"""Generate the saved experiments under tests/fixtures/saved/experiments/.

Format: an Experiments document, ``schema_version: 1``, one JSON file per
experiment named by its 32-hex id in the ``experiments`` directory beside the
conversations file.

  - 01-reply-42396fe/experiments/<id>.json - 42396fe "Add saved experiments,
    automatic comparisons, and token bookmarks (#126)", the commit that
    introduced the format: a chat reply saved through the Experiments tab's
    own ``save_run`` from the reply's trace, two bookmarks added with its
    ``annotate``, and a logit-lens inspection attached with its
    ``attach_inspection`` (the "older, manually attached inspections" today's
    timeline still falls back on).
  - 02-measurement-timeline-cf1acea/experiments/<id>.json - cf1acea "Add saved
    inference timeline replay and selective layer capture (#155)": a fixed
    passage measured through Compare's ``_measure_text`` (now stamped with
    ``run_id`` and ``session_id``), saved with ``experiment_runs.save``, a
    bookmark, and ``timeline_inspections`` written by the Timeline's own
    ``capture`` handler (positions 2, 4, 6, attention kept).

Every number comes from a real forward pass of a tiny random Llama decoder
(2 layers, 16 wide, seed 7, output head scaled by 40) over the tests' tiny byte-level BPE tokenizer of
that commit, run through that commit's ``ModelManager``.

Commands used:

    git worktree add --detach /tmp/chatlab-at-42396fe 42396fe
    cd /tmp/chatlab-at-42396fe && PYTHONPATH=.:tests <python> \
        <repo>/tests/fixtures/saved/experiments/generators/saved_experiments.py reply \
        <repo>/tests/fixtures/saved/experiments/01-reply-42396fe
    git worktree add --detach /tmp/chatlab-at-cf1acea cf1acea
    cd /tmp/chatlab-at-cf1acea && PYTHONPATH=.:tests <python> \
        <repo>/tests/fixtures/saved/experiments/generators/saved_experiments.py timeline \
        <repo>/tests/fixtures/saved/experiments/02-measurement-timeline-cf1acea

with <python> = /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python.
The ids, the creation times and the session id are pinned so the bytes are
reproducible; everything else is the writer's.
"""

import datetime as real_datetime
import os
import sys
import uuid
from pathlib import Path

mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
if out.exists():
    raise SystemExit(f"{out} exists; fixtures are never regenerated.")
os.environ["CHATLAB_LIBRARY_PATH"] = str(out / "conversations.json")

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import experiment_runs as runs  # noqa: E402  (top-level modules at these commits)
import model_runtime  # noqa: E402
import tiny_tokenizer  # noqa: E402
from ui import runtime  # noqa: E402

# The first uuid4 is the experiment's id; later ones only name staging files.
IDS = iter(["0123456789abcdef0123456789abcdef" if mode == "reply" else "fedcba9876543210fedcba9876543210"])


class PinnedUUID:
    def __init__(self, value):
        self.hex = value


class PinnedDatetime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return real_datetime.datetime(2026, 9, 16 if mode == "reply" else 22, 12, 0, 0, tzinfo=tz)


runs.uuid4 = lambda: PinnedUUID(next(IDS, None) or uuid.uuid4().hex)
runs.datetime = PinnedDatetime
runs.SESSION_ID = "5e55105e55105e55105e55105e551055"


def tiny_manager():
    torch.manual_seed(7)
    manager = model_runtime.ModelManager()
    manager.tokenizer = tiny_tokenizer.build()
    config = LlamaConfig(
        vocab_size=len(manager.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    manager.model = LlamaForCausalLM(config).eval()
    # Sharpen the output head so the distributions are peaked enough for
    # surprise and entropy to fall in different buckets token to token.
    with torch.no_grad():
        manager.model.lm_head.weight.mul_(40)
    manager.model_id = "test/tiny-decoder"
    manager.precision = "full"
    manager.load_count = 1
    manager._loaded = model_runtime.LoadedModel("test/tiny-decoder", "CPU", "full", "test/tiny-decoder#1")
    return manager


manager = runtime.MANAGER = tiny_manager()
published = manager.loaded_model()

if mode == "reply":
    from trace_export import build_trace
    from ui import experiments
    from ui.compare import _decoded_spans, _tokenizer_identity

    prompt, response = "the dog sat on the", " log and the cat sat on the mat"
    scored = manager.score_text(response, context=prompt, load_id=published.load_id)
    metrics = list(scored.metrics)
    context_ids = list(scored.context_ids)
    generation = 3
    decoded, ends = _decoded_spans(metrics, context_ids, response, 0)
    trace = build_trace(
        model_id=published.model_id,
        messages=[{"role": "system", "content": "Finish the rhyme."}, {"role": "user", "content": prompt}],
        response=response,
        sampling={"temperature": 0.7, "top_p": 0.9, "top_k": 40, "skip_top_below": 0.0,
                  "max_new_tokens": 16, "seed": 1234, "thinking_mode": "default"},
        metrics=metrics, generated_at="2026-09-16T11:59:00+00:00",
    )
    trace["run_context"] = {
        "session_id": runs.SESSION_ID, "load_id": published.load_id, "metrics_generation": generation,
        "context_ids": context_ids, "tokenizer": _tokenizer_identity(),
        "device_name": published.device_name, "precision": published.precision,
        "decoded": decoded, "token_ends": ends,
    }
    context = (generation, context_ids, published.load_id)
    picker, status = experiments.save_run("Rhyme with a tiny model", "Latest chat response",
                                          trace, context, None, None)
    assert status == "", status
    identifier = picker["value"]
    document = runs.read(identifier)
    for index, note in ((1, "Unexpected word here"), (4, "")):
        document, _, status = experiments.annotate(document, index, note, identifier)
        assert status == "Bookmark saved.", status
    position = 2
    insight = manager.inspect(context_ids + [m["token_id"] for m in metrics], len(context_ids) + position,
                              context_count=len(context_ids), load_id=published.load_id).to_dict()
    target = {"strip": "response", "generation": generation, "index": position}
    insight["saved_target"] = dict(target)
    insight["saved_session"] = runs.SESSION_ID
    document, _, status = experiments.attach_inspection(document, insight, target, context, identifier)
    assert status == "Inspection saved.", status
else:
    from ui import inference_timeline
    from ui.compare import _measure_text

    run = _measure_text("Passage:", " the cat sat on the mat", False, None, published)
    run["run_id"] = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
    item = runs.save(run, "Cat passage, measured")
    runs.bookmark(item["id"], 0, "First passage token")
    *_, (document, status) = inference_timeline.capture(item, item["id"], 2, 6, 2, "Logit", "", True)
    assert status.startswith("Saved 3 / 3"), status

print(out)
