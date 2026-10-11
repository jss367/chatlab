"""Saved Circuits attribution graphs written by earlier ChatLab code keep opening and reading the same way.

The files under tests/fixtures/saved/circuits were written once, by the commit each generator names, and are never
regenerated. Each directory is a Circuits data directory: graphs/<id>.json, plus graphs/.metadata/<id>.json (the menu
label) where that era's writer made one. The tests copy one into a temporary data directory, open it the way the
Circuits page does, and check what it reads: nodes, edges, prompt, explanation, pruning, names, groups, measured
effects, provenance, and what the page draws from them.
"""
import json
import re
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from chatlab.extensions.circuits import render, workbench

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "saved" / "circuits"
MODEL_ID = "google/gemma-3-270m-it"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "3d5a4f1e2c7b9a8d6e0f1c2b3a4d5e6f7a8b9c0d"

CHAT_TOKENS = ["<bos>", "<start_of_turn>", "user", "\n", "Answer", " in", " one", " word.", "\n", "\n", "What",
               " language", " do", " people", " speak", " in", " Paris?", "<end_of_turn>", "\n", "<start_of_turn>",
               "model", "\n"]
CHAT_IDS = [2, 3, 5, 7, 8, 9, 10, 11, 7, 7, 12, 13, 14, 15, 16, 9, 17, 4, 7, 3, 6, 7]

# What each frozen file must read as. Floats are compared exactly: the reader must hand back the numbers written.
EXPECTED = {
    "graph-top-before-revisions": dict(
        id="5f0c2a9e7b1d4c3a8e6f9b2d1a0c7e45", created=1759427700.0,
        tokens=CHAT_TOKENS, ids=CHAT_IDS,
        kinds={"feature": 15, "error": 26, "embedding": 19, "target": 10}, edges=509,
        features=["f:1:21:16", "f:0:21:40", "f:2:21:1", "f:0:21:10", "f:2:21:23", "f:2:21:21", "f:2:21:46",
                  "f:2:21:30", "f:0:21:23", "f:1:21:24", "f:0:21:39", "f:0:21:32", "f:1:21:10", "f:0:21:0",
                  "f:0:21:37"],
        feature=dict(id="f:1:21:16", layer=1, position=21, feature=16, activation=9.11589527130127,
                     influence=0.04999746382236481, effect=0.03227006271481514,
                     promotes=[" The", " of", " Italian", " answer", "model", " French"],
                     suppresses=[" say", " seven", " speak", "?", " city", " They"]),
        embedding=dict(id="t:1", position=1, token_id=3, text="<start_of_turn>"),
        targets=[("target:0", 35, " the", 0.02064310573041439), ("target:1", 6, "model", 0.020272783935070038),
                 ("target:2", 4, "<end_of_turn>", 0.01951015368103981),
                 ("target:3", 38, " Answer:", 0.019405607134103775),
                 ("target:4", 11, " word.", 0.018408408388495445), ("target:5", 61, " eight", 0.018402058631181717),
                 ("target:6", 17, " Paris?", 0.018383601680397987), ("target:7", 8, "Answer", 0.018372617661952972),
                 ("target:8", 39, " It", 0.018274841830134392), ("target:9", 50, " yes", 0.01821550913155079)],
        weights={("f:0:21:10", "f:1:21:16"): 1.1948518753051758, ("e:0:21", "f:1:21:16"): -6.15252685546875,
                 ("f:0:21:40", "target:8"): -0.13320331275463104},
        stats={"active_features": 1418, "traced_features": 20, "kept_features": 15,
               "error_share": 0.5206051468849182},
        settings={"max_feature_nodes": 20, "batch_size": 8, "node_threshold": 0.8, "edge_threshold": 0.9},
        prompt={"system": "Answer in one word.", "user": "What language do people speak in Paris?", "prefix": "",
                "raw": False},
        explain={"mode": "top", "tokens": [], "others": []},
        labels={"f:1:21:16": "French answer"},
        groups={"city": ["f:1:21:16", "f:0:21:40"], "language": ["f:2:21:1", "f:0:21:10", "f:2:21:23"]},
        effects=dict(prefixes=2, boost=2.0, every_position=False, pivot=[21], alternatives=[22, 23],
                     baseline=0.014518617186695337,
                     groups={"city": (0.013483137357980013, 0.015896529890596867, 2),
                             "language": (0.015056292526423931, 0.014484770130366087, 2)},
                     token_text={"21": " French", "22": " English", "23": " German"}),
        menu_label="Saved graph · 5f0c2a9e7b1d4c3a8e6f9b2d1a0c7e45",
        note="15 features survived pruning, of 20 traced and 1,418 active. Transcoder error carries 52% of the "
             "influence.",
        drawn_edge=("f:0:21:40", "target:8", "-0.133"),
        group_lines=["ablate ×0.93 · boost ×1.09", "ablate ×1.04 · boost ×1.00"],
        city_card="P(pivot) ×0.93 when ablated, ×1.09 when boosted ×2. Active in 2 of 2 prefixes.",
        pivot_text=" the",
    ),
    "graph-tokens-released": dict(
        id="a41c7d02e9b84f6a9c3e5d7f1b2a6c80", created=1759484700.0,
        tokens=CHAT_TOKENS + [" The", " answer", " is"], ids=CHAT_IDS + [18, 19, 20],
        kinds={"feature": 17, "error": 28, "embedding": 20, "target": 3}, edges=220,
        features=["f:2:24:46", "f:1:24:38", "f:0:24:33", "f:0:24:10", "f:0:24:23", "f:0:24:40", "f:0:24:14",
                  "f:1:24:30", "f:0:24:42", "f:2:24:33", "f:2:24:30", "f:1:24:16", "f:0:24:32", "f:1:24:45",
                  "f:0:22:20", "f:0:24:46", "f:0:24:11"],
        feature=dict(id="f:1:24:38", layer=1, position=24, feature=38, activation=2.6318681240081787,
                     influence=0.04070909693837166, effect=0.0017553707584738731,
                     promotes=[" maybe", " in", " German", " seven", " say", "!"],
                     suppresses=[" French", " one", ".", "What", " whose", " English"]),
        embedding=dict(id="t:2", position=2, token_id=5, text="user"),
        targets=[("target:0", 21, " French", 0.01446889340877533), ("target:1", 22, " English", 0.015329545363783836),
                 ("target:2", 23, " German", 0.01761634275317192)],
        weights={("f:0:24:33", "f:2:24:46"): 1.4757674932479858, ("e:0:24", "f:1:24:16"): -7.463561534881592,
                 ("t:22", "f:0:22:20"): 3.7670092582702637},
        stats={"active_features": 1607, "traced_features": 24, "kept_features": 17,
               "error_share": 0.5088100433349609},
        settings={"max_feature_nodes": 24, "batch_size": 8, "node_threshold": 0.8, "edge_threshold": 0.9},
        prompt={"system": "Answer in one word.", "user": "What language do people speak in Paris?",
                "prefix": " The answer is", "raw": False},
        explain={"mode": "tokens", "tokens": [" French", " English", " German"], "others": []},
        labels={"f:1:24:38": "French over English"},
        groups={"city": ["f:2:24:46", "f:1:24:38"], "language": ["f:0:24:33", "f:0:24:10", "f:0:24:23"]},
        effects=dict(prefixes=2, boost=2.0, every_position=False, pivot=[21], alternatives=[22, 23],
                     baseline=0.014432411977045444,
                     groups={"city": (0.01476789733876224, 0.01433584405047064, 2),
                             "language": (0.016949093330665806, 0.013234369242982331, 2)},
                     token_text={"21": " French", "22": " English", "23": " German"}),
        menu_label="Oct 03 09:45 · google/gemma-3-270m-it · The answer is",
        note="17 features survived pruning, of 24 traced and 1,607 active. Transcoder error carries 51% of the "
             "influence.",
        drawn_edge=("t:22", "f:0:22:20", "+3.77"),
        group_lines=["ablate ×1.02 · boost ×0.99", "ablate ×1.17 · boost ×0.92"],
        city_card="P(pivot) ×1.02 when ablated, ×0.99 when boosted ×2. Active in 2 of 2 prefixes.",
        pivot_text=" French",
    ),
    "graph-contrast-released": dict(
        id="c0ffee5e1d2b4a6f8e9d7c3b5a1f2e40", created=1759571100.0,
        tokens=["<bos>", "What", " language", " do", " people", " speak", " in", " Paris?", " The", " answer", " is"],
        ids=[2, 12, 13, 14, 15, 16, 9, 17, 18, 19, 20],
        kinds={"feature": 12, "error": 12, "embedding": 10, "target": 1}, edges=55,
        features=["f:1:10:45", "f:1:10:8", "f:1:10:38", "f:0:10:23", "f:0:10:19", "f:0:10:33", "f:0:10:42",
                  "f:2:10:46", "f:2:10:18", "f:0:10:47", "f:1:10:5", "f:0:10:46"],
        feature=dict(id="f:1:10:8", layer=1, position=10, feature=8, activation=2.796417474746704,
                     influence=0.04858746379613876, effect=-0.04815720394253731,
                     promotes=["<eos>", " Spanish", " nine", " six", "\n", " English"],
                     suppresses=[" in", "Answer", " They", " yes", " French", " people"]),
        embedding=dict(id="t:1", position=1, token_id=12, text="What"),
        targets=[("target:0", None, " French /  France vs other", 0.03245224431157112)],
        weights={("e:0:10", "f:1:10:45"): 1.5058774948120117, ("e:0:10", "f:1:10:8"): -2.9698076248168945,
                 ("t:10", "f:0:10:46"): 2.3502917289733887},
        stats={"active_features": 674, "traced_features": 20, "kept_features": 12, "error_share": 0.438120037317276},
        settings={"max_feature_nodes": 20, "batch_size": 4, "node_threshold": 0.75, "edge_threshold": 0.85},
        prompt={"system": "", "user": "What language do people speak in Paris? The answer is", "prefix": "",
                "raw": True},
        explain={"mode": "contrast", "tokens": [" French", " France"], "others": [" English", " German", " Spanish"]},
        labels={"f:1:10:8": "Paris → French"},
        groups={"city": ["f:1:10:45", "f:1:10:8"], "language": ["f:1:10:38", "f:0:10:23", "f:0:10:19"]},
        effects=dict(prefixes=2, boost=3.5, every_position=True, pivot=[21],
                     alternatives=[20, 55, 3, 42, 13, 57, 26, 45], baseline=0.01627714309026629,
                     groups={"city": (0.01596916825047362, 0.015808062429930755, 2),
                             "language": (0.020366385657827737, 0.01130433180399586, 2)},
                     token_text={"21": " French", "20": " is", "55": " two", "3": "<start_of_turn>", "42": " They",
                                 "13": " language", "57": " four", "26": " Paris", "45": " Hello"}),
        menu_label="Oct 04 09:45 · google/gemma-3-270m-it ·  do people speak in Paris? The answer is",
        note="12 features survived pruning, of 20 traced and 674 active. Transcoder error carries 44% of the "
             "influence.",
        drawn_edge=("t:10", "f:0:10:46", "+2.35"),
        group_lines=["ablate ×0.98 · boost ×0.97", "ablate ×1.25 · boost ×0.69"],
        city_card="P(pivot) ×0.98 when ablated, ×0.97 when boosted ×3.5. Active in 2 of 2 prefixes.",
        pivot_text=" French\n France",
    ),
}


def data_dir_with(name, directory):
    """A writable copy of a frozen Circuits data directory, and the path of its one saved graph."""
    shutil.copytree(FIXTURES / name, directory, dirs_exist_ok=True)
    graphs = sorted((Path(directory) / "graphs").glob("*.json"))
    assert len(graphs) == 1, graphs
    return graphs[0]


class SavedGraphFacts:
    """Assertions shared by every era: what a graph file reads as once ChatLab has opened it."""

    name = None

    @property
    def expected(self):
        return EXPECTED[self.name]

    def assert_reads_as_written(self, graph):
        want = self.expected
        self.assertEqual(graph["format"], "chatlab-attribution-graph-1")
        self.assertEqual(graph["id"], want["id"])
        self.assertEqual(graph["created"], want["created"])
        self.assertEqual(graph["model_id"], MODEL_ID)
        self.assertEqual(graph["transcoders"], "gemma-3-270m-it")
        self.assertEqual(graph["layers"], 3)
        self.assertEqual(graph["tokens"], want["tokens"])
        self.assertEqual(graph["ids"], want["ids"])
        nodes = {n["id"]: n for n in graph["nodes"]}
        self.assertEqual(dict(Counter(n["kind"] for n in graph["nodes"])), want["kinds"])
        self.assertEqual([n["id"] for n in graph["nodes"] if n["kind"] == "feature"], want["features"])
        feature = nodes[want["feature"]["id"]]
        for key, value in want["feature"].items():
            self.assertEqual(feature[key], value, key)
        self.assertEqual(feature["kind"], "feature")
        embedding = nodes[want["embedding"]["id"]]
        self.assertEqual((embedding["kind"], embedding["layer"]), ("embedding", -1))
        for key, value in want["embedding"].items():
            self.assertEqual(embedding[key], value, key)
        targets = [n for n in graph["nodes"] if n["kind"] == "target"]
        self.assertEqual([(n["id"], n.get("token_id"), n["text"], n["probability"]) for n in targets], want["targets"])
        last = len(want["tokens"]) - 1
        self.assertTrue(all(n["layer"] == 3 and n["position"] == last for n in targets))
        errors = [n for n in graph["nodes"] if n["kind"] == "error"]
        self.assertTrue(all(0 <= n["layer"] < 3 and "feature" not in n for n in errors))
        self.assertEqual(len(graph["edges"]), want["edges"])
        weights = {(e["source"], e["target"]): e["weight"] for e in graph["edges"]}
        for pair, weight in want["weights"].items():
            self.assertEqual(weights[pair], weight, pair)
        self.assertEqual(graph["stats"], want["stats"])
        self.assertEqual(graph["settings"], want["settings"])
        self.assertEqual(graph["prompt"], want["prompt"])
        self.assertEqual(graph["explain"], want["explain"])
        self.assertEqual(graph["labels"], want["labels"])
        self.assertEqual(graph["groups"], want["groups"])
        effects, measured = graph["effects"], want["effects"]
        for key in ("prefixes", "boost", "every_position", "pivot", "alternatives", "token_text"):
            self.assertEqual(effects[key], measured[key], key)
        self.assertEqual(effects["baseline"]["pivot"], measured["baseline"])
        self.assertEqual(set(effects["baseline"]["tokens"]), {str(t) for t in measured["pivot"] + measured["alternatives"]})
        self.assertEqual({name: (g["ablate"]["pivot"], g["boost"]["pivot"], g["active_prefixes"])
                          for name, g in effects["groups"].items()}, measured["groups"])
        # The catalogue states no training checkpoint for this set, so the reader calls every graph unverified.
        self.assertIsNone(graph["training_model_revision"])
        self.assertEqual(graph["checkpoint_compatibility"], "unverified")

    def open_in_page(self, directory, path, show_errors=False):
        import gradio as gr
        from chatlab.extension_api import ExtensionContext, NavigationService, TokenInspector
        from chatlab.extensions.circuits.page import build_page
        from ui_support import handlers_by_name

        context = ExtensionContext(SimpleNamespace(loaded_model_id=lambda: None), TokenInspector(), Path(directory),
                                   NavigationService(lambda *args: None, lambda *args: None))
        with gr.Blocks() as demo:
            build_page(context)
        try:
            return handlers_by_name(demo)["open_path"](str(path), 40, show_errors, "view")
        finally:
            demo.close()

    def test_load_graph_reads_what_was_written(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_reads_as_written(workbench.load_graph(data_dir_with(self.name, directory)))

    def test_the_page_opens_and_draws_it(self):
        want = self.expected
        with tempfile.TemporaryDirectory() as directory:
            path = data_dir_with(self.name, directory)
            (graph, selection, focus, view, card, note, groups_view, group_pick, group_card, download, saved,
             pivot) = self.open_in_page(directory, path)
            self.assert_reads_as_written(graph)
            self.assertEqual((selection, focus, note), ([], None, ""))
            for node_id in want["features"]:
                self.assertIn(f'data-node="{node_id}"', view)
            for target, *_ in want["targets"]:
                self.assertIn(f'data-node="{target}"', view)
            self.assertNotIn('data-node="e:', view)
            self.assertIn(f'cg-note">{want["note"]}', view)
            source, target, weight = want["drawn_edge"]
            self.assertRegex(view, rf'data-s="{source}" data-t="{target}"[^>]*><title>{re.escape(weight)}</title>')
            for label in want["labels"].values():
                self.assertIn(render.html.escape(render._short(label)), view)
            for line in want["group_lines"]:
                self.assertIn(line, groups_view)
            self.assertEqual(group_pick["choices"], list(want["groups"]))
            self.assertEqual(group_pick["value"], "city")
            self.assertIn(want["city_card"], group_card)
            self.assertEqual(pivot, want["pivot_text"])
            self.assertEqual([label for label, _ in saved["choices"]], [want["menu_label"]])
            self.assertIn("Click a node", card)
            # The download box offers the opened graph, and uploading that copy reads the same.
            self.assert_reads_as_written(workbench.load_graph(download))
            with_errors = self.open_in_page(directory, path, show_errors=True)[3]
            self.assertIn('data-node="e:', with_errors)

    def test_feature_card_shows_the_named_feature(self):
        want = self.expected
        with tempfile.TemporaryDirectory() as directory:
            graph = workbench.load_graph(data_dir_with(self.name, directory))
        node = next(n for n in graph["nodes"] if n["id"] == want["feature"]["id"])
        card = render.feature_card(node, None, graph["labels"], tokens=graph["tokens"])
        label = want["labels"][node["id"]]
        self.assertIn(f'{render.html.escape(label)} <span class="cg-muted">— layer {node["layer"]} · '
                      f'feature {node["feature"]}', card)
        self.assertIn(f'{node["activation"]:.4g}', card)
        for token in want["feature"]["promotes"]:
            self.assertIn(render.html.escape(render.token_text(token)), card)

    def test_the_saved_menu_lists_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = data_dir_with(self.name, directory)
            listed = workbench.Workbench(SimpleNamespace(), directory).saved()
        self.assertEqual(listed, [(self.expected["menu_label"], str(path))])

    def test_resaving_with_todays_writer_keeps_what_it_reads_as(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            graph = workbench.load_graph(data_dir_with(self.name, directory))
            bench = workbench.Workbench(SimpleNamespace(), other)
            path = bench.save(graph)
            self.assertEqual(path.name, f"{self.expected['id']}.json")
            self.assert_reads_as_written(workbench.load_graph(path))
            [(label, listed)] = bench.saved()
        self.assertEqual(listed, str(path))
        tail = (self.expected["prompt"]["prefix"] or self.expected["prompt"]["user"]).strip()[-40:]
        self.assertTrue(label.endswith(f" · {MODEL_ID} · {tail}"), label)


class FirstWriterGraphTests(SavedGraphFacts, unittest.TestCase):
    """Written by 90bcb9f, before graphs recorded revisions, the tracing process, or a menu label."""

    name = "graph-top-before-revisions"

    def raw(self):
        return json.loads(next((FIXTURES / self.name / "graphs").glob("*.json")).read_text())

    def test_the_file_predates_provenance_and_width(self):
        raw = self.raw()
        for key in ("model_revision", "transcoder_revision", "process_id", "transcoder_load_id",
                    "training_model_revision", "checkpoint_compatibility", "transcoder_width", "_view_version"):
            self.assertNotIn(key, raw)
        self.assertFalse((FIXTURES / self.name / "graphs" / ".metadata").exists())

    def test_missing_provenance_reads_as_unpinned(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = workbench.load_graph(data_dir_with(self.name, directory))
        self.assertNotIn("transcoder_revision", graph)
        self.assertNotIn("model_revision", graph)
        self.assertEqual(graph["load_id"], "load-0001")
        # Nothing ties this graph to a load ChatLab can prove is the same, so it refuses to measure it.
        session = SimpleNamespace(model_id=MODEL_ID, model_revision=MODEL_REVISION, load_id="load-0001")
        with self.assertRaisesRegex(ValueError, "another model load"):
            workbench.Workbench._same_model(graph, session)
        with self.assertRaisesRegex(ValueError, "another transcoder load"):
            workbench.Workbench._same_transcoders(graph, SimpleNamespace(revision=TRANSCODER_REVISION,
                                                                         load_id="transcoders-0001"))


class ReleasedTokensGraphTests(SavedGraphFacts, unittest.TestCase):
    """The released writer (f01bf05, PR #177's merge), explaining chosen tokens."""

    name = "graph-tokens-released"

    def test_recorded_provenance_pins_measurement_to_the_same_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = workbench.load_graph(data_dir_with(self.name, directory))
        self.assertEqual(graph["model_revision"], MODEL_REVISION)
        self.assertEqual(graph["transcoder_revision"], TRANSCODER_REVISION)
        self.assertEqual((graph["process_id"], graph["load_id"], graph["transcoder_load_id"]),
                         ("process-0002", "load-0002", "transcoders-0002"))
        self.assertEqual(graph["transcoder_width"], 48)
        # A later process with the same model revision and transcoder snapshot may measure it.
        later = SimpleNamespace(model_id=MODEL_ID, model_revision=MODEL_REVISION, load_id="another-load")
        workbench.Workbench._same_model(graph, later)
        workbench.Workbench._same_transcoders(graph, SimpleNamespace(revision=TRANSCODER_REVISION, load_id="x"))
        with self.assertRaisesRegex(ValueError, "another model revision"):
            workbench.Workbench._same_model(graph, SimpleNamespace(model_id=MODEL_ID, model_revision="0" * 40,
                                                                   load_id="load-0002"))
        with self.assertRaisesRegex(ValueError, "another transcoder revision"):
            workbench.Workbench._same_transcoders(graph, SimpleNamespace(revision="0" * 40, load_id="transcoders-0002"))
        with self.assertRaisesRegex(ValueError, "traced on google/gemma-3-270m-it"):
            workbench.Workbench._same_model(graph, SimpleNamespace(model_id="google/gemma-3-1b-it",
                                                                   model_revision=MODEL_REVISION, load_id="x"))

    def test_the_menu_label_comes_from_its_metadata_file(self):
        metadata = next((FIXTURES / self.name / "graphs" / ".metadata").glob("*.json"))
        self.assertEqual(json.loads(metadata.read_text()), {"label": self.expected["menu_label"]})


class ReleasedContrastGraphTests(SavedGraphFacts, unittest.TestCase):
    """The released writer explaining pivot tokens against others, from a raw prompt."""

    name = "graph-contrast-released"

    def test_the_contrast_target_keeps_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = workbench.load_graph(data_dir_with(self.name, directory))
        [target] = [n for n in graph["nodes"] if n["kind"] == "target"]
        self.assertEqual(target["target_kind"], "contrast")
        self.assertEqual((target["positive"], target["negative"]), ([21, 31], [22, 23, 24]))
        self.assertEqual(target["negative_probability"], 0.04382709413766861)
        self.assertEqual(graph["effects"]["every_position"], True)


if __name__ == "__main__":
    unittest.main()
