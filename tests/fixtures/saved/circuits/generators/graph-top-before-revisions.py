"""Write a saved Circuits attribution graph with the first writer the extension had.

Format marker: chatlab-attribution-graph-1
Era: the first Circuits writer, before graphs recorded model or transcoder revisions, the process that traced them,
checkpoint provenance, or a menu label beside the file. Explains "the likeliest next tokens" (top mode). This writer
lived only on the attribution-graphs branch (PR #177) and was never in a release; every released ChatLab writes the
form in graph-tokens-released and graph-contrast-released.
Generating commit: 90bcb9f "Add a Circuit tracing extension: attribution graphs over transcoders, with group
interventions" (first commit of PR #177).

Commands:
    git worktree add --detach /tmp/chatlab-at-90bcb9f 90bcb9f
    cd /tmp/chatlab-at-90bcb9f && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/circuits/generators/graph-top-before-revisions.py \
        <repo>/tests/fixtures/saved/circuits/graph-top-before-revisions

The output argument is a ChatLab Circuits data directory; that commit's Workbench.save writes graphs/<id>.json in it.
The model is that commit's tests' tiny Gemma 3 with its tiny transcoders, standing in for gemma-3-270m-it's catalogue
entry; the graph id, creation time and tokenizer are pinned so the bytes are the same on every run.
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

GRAPH_ID = "5f0c2a9e7b1d4c3a8e6f9b2d1a0c7e45"
CREATED = 1759427700.0  # 2025-10-02 17:55 UTC
MODEL_ID = "google/gemma-3-270m-it"
SPECIALS = ["<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>", "user", "model", "\n"]
WORDS = ["Answer", " in", " one", " word.", "What", " language", " do", " people", " speak", " Paris?",
         " The", " answer", " is", " French", " English", " German", " Spanish", " Italian", " Paris", " London",
         " Berlin", " Madrid", " Rome", " France", " city", " capital", " of", " the", " country", " whose",
         " Answer:", " It", " a", " language.", " They", " say", " Bonjour", " Hello", ".", ",", "!", "?",
         " yes", " no", " maybe", " word", " one.", " two", " three", " four", " five", " six", " seven",
         " eight", " nine", " ten"]
VOCAB = SPECIALS + WORDS
assert len(VOCAB) <= 64


class Session:
    """The GenerationSession surface this commit's Workbench uses, over a word-level tokenizer."""

    model_id = MODEL_ID
    load_id = "load-0001"

    def __init__(self, model):
        self.model = model

    def transformers_model(self):
        return contextlib.nullcontext(self.model)

    def encode(self, text):
        return [VOCAB.index(piece) for piece in re.findall(r"\n|\s?[^\s]+", text)]

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def prompt_ids(self, messages):
        ids = [VOCAB.index("<bos>")]
        system = "".join(m["content"] + "\n\n" for m in messages if m["role"] == "system")
        for message in messages:
            if message["role"] == "user":
                ids += [3, 5, 7] + self.encode(system + message["content"]) + [4, 7]
        return ids + [3, 6, 7]


def main(out):
    model = tiny_model("gemma3")
    blocks = architecture.blocks(model)
    held = tiny_transcoders(blocks)
    session = Session(model)
    models = SimpleNamespace(open_session=lambda: contextlib.nullcontext(session),
                             loaded_model_id=lambda: MODEL_ID)
    bench = workbench.Workbench(models, out)
    prompt = {"system": "Answer in one word.", "user": "What language do people speak in Paris?",
              "prefix": "", "raw": False}
    explain = {"mode": "top", "tokens": [], "others": []}
    settings = attribution.Settings(20, 8, 0.8, 0.9)
    with mock.patch.object(transcoders, "loaded", return_value=held), \
            mock.patch.object(workbench, "uuid4", return_value=SimpleNamespace(hex=GRAPH_ID)), \
            mock.patch.object(workbench.time, "time", return_value=CREATED):
        graph = bench.trace(prompt, explain, settings, lambda *args: None, lambda: False)
        bench.save(graph)
        # What the page does next: name a feature, group features, measure the groups.
        features = [n for n in graph["nodes"] if n["kind"] == "feature"]
        graph = {**graph, "labels": {features[0]["id"]: "French answer"}}
        bench.save(graph)
        graph = {**graph, "groups": {"city": [n["id"] for n in features[:2]],
                                     "language": [n["id"] for n in features[2:5]]}, "effects": None}
        bench.save(graph)
        effects = bench.group_effects(graph, [" French"], [" English", " German"], [" The answer is"], True, 2.0,
                                      False, lambda *args: None, lambda: False)
        graph = {**graph, "effects": effects}
        print(bench.save(graph))


if __name__ == "__main__":
    main(sys.argv[1])
