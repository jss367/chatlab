"""Write a saved Circuits attribution graph, and its menu label, with the writer every release has had.

Format marker: chatlab-attribution-graph-1 (graphs/<id>.json) plus the menu label beside it (graphs/.metadata/<id>.json)
Era: released. PR #177's merge introduced this writer, and Workbench.trace, Workbench.save, group_effects and
load_graph have not changed since; every ChatLab release with Circuits (v0.33.0 on) writes this form. It records the
model and transcoder revisions, the tracing process and loads, checkpoint provenance, and the page's view version.
This file explains chosen tokens (tokens mode) with a chat-template prompt and a reply prefix;
graph-contrast-released.py calls main() here for the contrast-mode file.
Generating commit: f01bf05 "Merge pull request #177 from jss367/claude/attribution-graphs".

Commands:
    git worktree add --detach /tmp/chatlab-at-f01bf05 f01bf05
    cd /tmp/chatlab-at-f01bf05 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/circuits/generators/graph-tokens-released.py \
        <repo>/tests/fixtures/saved/circuits/graph-tokens-released

The output argument is a ChatLab Circuits data directory; that commit's Workbench.save writes graphs/<id>.json and
graphs/.metadata/<id>.json in it. The model is that commit's tests' tiny Gemma 3 with its tiny transcoders, standing in
for gemma-3-270m-it's catalogue entry. Graph id, time, process id, loads, revisions and tokenizer are pinned, and the
page's edits (name a feature, group features, measure groups) are replayed as the page makes them, so the bytes are the
same on every run.
"""
import contextlib
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TREE = Path.cwd()
sys.path[:0] = [str(TREE), str(TREE / "tests")]
os.environ["TZ"] = "UTC"
time.tzset()

from chatlab.extensions.circuits import architecture, attribution, transcoders, workbench  # noqa: E402
from test_circuits import tiny_model, tiny_transcoders  # noqa: E402

MODEL_ID = "google/gemma-3-270m-it"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "3d5a4f1e2c7b9a8d6e0f1c2b3a4d5e6f7a8b9c0d"
SPECIALS = ["<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>", "user", "model", "\n"]
WORDS = ["Answer", " in", " one", " word.", "What", " language", " do", " people", " speak", " Paris?",
         " The", " answer", " is", " French", " English", " German", " Spanish", " Italian", " Paris", " London",
         " Berlin", " Madrid", " Rome", " France", " city", " capital", " of", " the", " country", " whose",
         " Answer:", " It", " a", " language.", " They", " say", " Bonjour", " Hello", ".", ",", "!", "?",
         " yes", " no", " maybe", " word", " one.", " two", " three", " four", " five", " six", " seven",
         " eight", " nine", " ten"]
VOCAB = SPECIALS + WORDS
assert len(VOCAB) <= 64

RUNS = {
    "tokens": dict(
        graph_id="a41c7d02e9b84f6a9c3e5d7f1b2a6c80", created=1759484700.0,  # 2025-10-03 09:45 UTC
        prompt={"system": "Answer in one word.", "user": "What language do people speak in Paris?",
                "prefix": " The answer is", "raw": False},
        explain={"mode": "tokens", "tokens": [" French", " English", " German"], "others": []},
        settings=(24, 8, 0.8, 0.9), label="French over English",
        measure=([" French"], [" English", " German"], [" Answer:"], True, 2.0, False)),
    "contrast": dict(
        graph_id="c0ffee5e1d2b4a6f8e9d7c3b5a1f2e40", created=1759571100.0,  # 2025-10-04 09:45 UTC
        prompt={"system": "", "user": "What language do people speak in Paris? The answer is", "prefix": "", "raw": True},
        explain={"mode": "contrast", "tokens": [" French", " France"], "others": [" English", " German", " Spanish"]},
        settings=(20, 4, 0.75, 0.85), label="Paris → French",
        measure=([" French"], [], [" It is", " They say"], False, 3.5, True)),
}


class Session:
    """The GenerationSession surface this commit's Workbench uses, over a word-level tokenizer."""

    model_id = MODEL_ID
    load_id = "load-0002"
    model_revision = MODEL_REVISION

    def __init__(self, model):
        self.model = model

    def transformers_model(self):
        return contextlib.nullcontext(self.model)

    def encode(self, text):
        return [VOCAB.index(piece) for piece in re.findall(r"\n|\s?[^\s]+", text)]

    def encode_replacement(self, kept_ids, text, **options):
        return self.encode(text)

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def prompt_ids(self, messages):
        ids = [VOCAB.index("<bos>")]
        system = "".join(m["content"] + "\n\n" for m in messages if m["role"] == "system")
        for message in messages:
            if message["role"] == "user":
                ids += [3, 5, 7] + self.encode(system + message["content"]) + [4, 7]
        return ids + [3, 6, 7]


def main(out, mode="tokens"):
    run = RUNS[mode]
    model = tiny_model("gemma3")
    blocks = architecture.blocks(model)
    held = tiny_transcoders(blocks)
    held.revision = TRANSCODER_REVISION
    held.load_id = "transcoders-0002"
    session = Session(model)
    models = SimpleNamespace(open_session=lambda: contextlib.nullcontext(session),
                             loaded_model_id=lambda: MODEL_ID)
    bench = workbench.Workbench(models, out)
    versions = iter(f"{mode}-view-{i}" for i in range(1, 10))

    def save(graph):
        # The page stamps a fresh view version on the graph before every save.
        graph["_view_version"] = next(versions)
        return bench.save(graph)

    with mock.patch.object(transcoders, "loaded", return_value=held), \
            mock.patch.object(workbench, "uuid4", return_value=SimpleNamespace(hex=run["graph_id"])), \
            mock.patch.object(workbench, "PROCESS_ID", "process-0002"), \
            mock.patch.object(workbench.time, "time", return_value=run["created"]):
        graph = bench.trace(run["prompt"], run["explain"], attribution.Settings(*run["settings"]),
                            lambda *args: None, lambda: False)
        save(graph)
        features = [n for n in graph["nodes"] if n["kind"] == "feature"]
        graph = {**graph, "labels": {features[1]["id"]: run["label"]}}
        save(graph)
        graph = {**graph, "groups": {"city": [n["id"] for n in features[:2]]}, "effects": None}
        save(graph)
        graph = {**graph, "groups": {"city": [n["id"] for n in features[:2]],
                                     "language": [n["id"] for n in features[2:5]]}, "effects": None}
        save(graph)
        effects = bench.group_effects(graph, *run["measure"], lambda *args: None, lambda: False)
        graph = {**graph, "effects": effects}
        print(save(graph))


if __name__ == "__main__":
    main(sys.argv[1])
