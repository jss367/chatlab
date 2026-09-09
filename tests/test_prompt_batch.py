import csv
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

import gradio as gr

import app
import settings_sandbox
from prompt_batch import (
    BatchTable,
    parse_prompt_file,
    parse_prompts,
    prompts_to_text,
    write_batch_csv,
    write_batch_trace,
)
from trace_export import build_trace, traces_to_csv
from ui import runtime

from test_streaming import EOS_ID, PIECES, loaded_manager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


# The batch handlers publish app.BATCH_OUTPUT_NAMES, in that order.
STATUS, RESULTS, RUN, STOP, FILES, DIRECTORY = range(len(app.BATCH_OUTPUT_NAMES))

# The results table's columns, by name.
(
    INDEX,
    PROMPT,
    RESPONSE,
    TOKENS,
    PERPLEXITY,
    MEAN_SURPRISE,
    SEED,
) = range(len(app.BATCH_HEADERS))

# The sampling half of run_prompts()' arguments: greedy, short, and with the
# seed pinned so a row's numbers are the same on every run.
SAMPLING = (0.0, 1.0, 0, 8, 42, False)


def trace_of(update) -> dict:
    """The trace written for one prompt, read back off disk."""

    paths = [Path(path) for path in update["value"]]
    return json.loads(paths[0].read_text(encoding="utf-8"))


class ParsePromptsTests(unittest.TestCase):
    def test_a_blank_line_separates_prompts(self):
        prompts = parse_prompts("first prompt\n\nsecond prompt")

        self.assertEqual(prompts, ["first prompt", "second prompt"])

    def test_a_prompt_may_run_to_several_lines(self):
        prompts = parse_prompts("a passage\nand its question\n\nanother")

        self.assertEqual(prompts, ["a passage\nand its question", "another"])

    def test_whitespace_only_lines_separate_and_are_dropped(self):
        prompts = parse_prompts("  first \n \t \n\n\nsecond\n")

        self.assertEqual(prompts, ["first", "second"])

    def test_nothing_written_is_no_prompts(self):
        self.assertEqual(parse_prompts("   \n\n  "), [])
        self.assertEqual(parse_prompts(""), [])

    def test_the_box_text_round_trips(self):
        prompts = ["one", "two\nwith a second line"]

        self.assertEqual(parse_prompts(prompts_to_text(prompts)), prompts)


class ParsePromptFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="chatlab-test-"))
        self.addCleanup(shutil.rmtree, self.directory)

    def write(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_jsonl_takes_one_prompt_per_line(self):
        path = self.write(
            "prompts.jsonl",
            '{"prompt": "first"}\n\n{"text": "second"}\n"third"\n',
        )

        self.assertEqual(parse_prompt_file(path), ["first", "second", "third"])

    def test_a_jsonl_line_that_is_not_json_names_its_line(self):
        path = self.write("prompts.jsonl", '{"prompt": "fine"}\nnot json\n')

        with self.assertRaises(ValueError) as caught:
            parse_prompt_file(path)

        self.assertIn("Line 2", str(caught.exception))

    def test_a_jsonl_object_without_a_prompt_says_which_keys_it_wanted(self):
        path = self.write("prompts.jsonl", '{"question": "first"}\n')

        with self.assertRaises(ValueError) as caught:
            parse_prompt_file(path)

        self.assertIn("Line 1", str(caught.exception))
        self.assertIn("prompt", str(caught.exception))

    def test_json_takes_a_list_or_an_object_holding_one(self):
        listed = self.write("prompts.json", '["first", {"content": "second"}]')
        wrapped = self.write("wrapped.json", '{"prompts": ["first", "second"]}')

        self.assertEqual(parse_prompt_file(listed), ["first", "second"])
        self.assertEqual(parse_prompt_file(wrapped), ["first", "second"])

    def test_json_that_is_not_a_list_of_prompts_is_refused(self):
        path = self.write("prompts.json", '{"model": "olmo"}')

        with self.assertRaises(ValueError) as caught:
            parse_prompt_file(path)

        self.assertIn("list of prompts", str(caught.exception))

    def test_any_other_extension_is_read_as_text(self):
        path = self.write("prompts.txt", "first\n\nsecond over\ntwo lines\n")

        self.assertEqual(parse_prompt_file(path), ["first", "second over\ntwo lines"])


def sample_trace(*, seed: int = 1, candidates: int = 1, response: str = "hello"):
    return build_trace(
        model_id="example/model",
        messages=[{"role": "user", "content": "Say hello"}],
        response=response,
        sampling={
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 100,
            "seed": seed,
        },
        metrics=[
            {
                "position": 1,
                "token_id": 42,
                "text": response,
                "display_text": response,
                "category": "Top 5",
                "raw_rank": 2,
                "raw_probability": 0.25,
                "sampling_probability": 0.4,
                "surprise_bits": 2.0,
                "probability_mass_above": 0.5,
                "top_candidates": [
                    {"token_id": index, "text": "x", "probability": 0.5}
                    for index in range(candidates)
                ],
            }
        ],
        generated_at="2026-08-31T12:00:00+00:00",
    )


class BatchExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="chatlab-test-"))
        self.addCleanup(shutil.rmtree, self.directory)
        # A permissive umask is what makes the file mode visible: under it a
        # plain write would leave the traces world-readable.
        self.addCleanup(os.umask, os.umask(0))

    def test_one_trace_per_prompt_is_named_for_its_place_in_the_run(self):
        first = Path(write_batch_trace(sample_trace(), self.directory, 1))
        tenth = Path(write_batch_trace(sample_trace(), self.directory, 10))

        self.assertEqual(first.name, "prompt-001.json")
        self.assertEqual(tenth.name, "prompt-010.json")
        self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)

    def test_the_table_numbers_every_row_with_the_prompt_it_came_from(self):
        traces = [sample_trace(seed=1), sample_trace(seed=2)]

        path = Path(write_batch_csv(traces, self.directory))
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

        self.assertEqual([row["prompt_index"] for row in rows], ["1", "2"])
        self.assertEqual([row["seed"] for row in rows], ["1", "2"])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_the_table_numbers_rows_by_the_prompt_the_caller_names(self):
        # The traces a batch collects are only the prompts that worked, so the
        # run says which prompt each one answered rather than letting the list
        # position speak for it.
        traces = [sample_trace(seed=1), sample_trace(seed=2)]

        path = Path(write_batch_csv(traces, self.directory, [2, 5]))
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

        self.assertEqual([row["prompt_index"] for row in rows], ["2", "5"])

    def test_a_trace_without_a_number_is_refused(self):
        # Numbering by count is the bug this parameter exists to prevent, so a
        # caller that passes the wrong number of indexes is told, not guessed
        # at.
        with self.assertRaises(ValueError):
            traces_to_csv([sample_trace(), sample_trace()], [1])

    def test_the_table_says_which_answers_were_cut_short(self):
        # The table is read on its own, so a reader who never opens the
        # traces would otherwise take a stopped answer for a whole one.
        whole = sample_trace(seed=1)
        stopped = sample_trace(seed=2)
        stopped["sampling"]["stopped"] = True

        rows = list(csv.DictReader(io.StringIO(traces_to_csv([whole, stopped]))))

        self.assertEqual([row["stopped"] for row in rows], ["False", "True"])

    def test_prompts_with_different_candidate_counts_share_one_header(self):
        # Top-k can be changed between runs, and a shorter candidate list must
        # not truncate the columns of the longer one it is written beside.
        table = traces_to_csv([sample_trace(candidates=1), sample_trace(candidates=3)])
        reader = csv.DictReader(io.StringIO(table))
        rows = list(reader)

        self.assertIn("candidate_3_text", reader.fieldnames)
        self.assertEqual(rows[0]["candidate_3_text"], "")
        self.assertEqual(rows[1]["candidate_3_text"], "x")

    def test_a_finished_prompt_survives_the_next_ones_rewrite(self):
        # The table is rewritten after every prompt so a stopped run still has
        # one. The rows already in it have to come back unchanged.
        first = Path(write_batch_csv([sample_trace(seed=1)], self.directory))
        after = Path(write_batch_csv([sample_trace(seed=1), sample_trace(seed=2)], self.directory))
        rows = list(csv.DictReader(io.StringIO(after.read_text(encoding="utf-8"))))

        self.assertEqual(first, after)
        self.assertEqual([row["seed"] for row in rows], ["1", "2"])




class ResolvePromptsTests(unittest.TestCase):
    """Which prompts a run uses: the file's own, or the box's."""

    def test_a_loaded_prompt_keeps_its_blank_lines(self):
        # A dataset entry of two paragraphs reads as two prompts once it is
        # in the box. Running it as two would answer half an entry at a time
        # and file the measurements under prompts nobody wrote.
        loaded = ["a passage\n\nand its second paragraph", "another"]
        text = app.prompts_to_text(loaded)

        self.assertEqual(app.resolve_prompts(text, loaded), loaded)
        self.assertEqual(len(app.parse_prompts(text)), 3)

    def test_an_edited_box_is_read_as_it_is_written(self):
        loaded = ["first", "second"]
        edited = app.prompts_to_text(loaded) + "\n\nthird"

        self.assertEqual(app.resolve_prompts(edited, loaded), ["first", "second", "third"])

    def test_a_box_nothing_was_loaded_into_is_read_as_written(self):
        self.assertEqual(app.resolve_prompts("one\n\ntwo", []), ["one", "two"])
        self.assertEqual(app.resolve_prompts("one\n\ntwo", None), ["one", "two"])

    def test_loading_says_when_the_box_cannot_show_a_prompt_whole(self):
        directory = Path(tempfile.mkdtemp(prefix="chatlab-test-"))
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "prompts.jsonl"
        path.write_text('{"prompt": "one\\n\\ntwo"}\n', encoding="utf-8")

        _text, status, prompts = app.load_prompt_file(str(path), "", [])

        self.assertEqual(prompts, ["one\n\ntwo"])
        self.assertIn(app.PARAGRAPH_NOTE, status)

    def test_a_run_answers_a_loaded_paragraph_prompt_once(self):
        original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([0, 1, EOS_ID], PIECES, EOS_ID)
        self.addCleanup(setattr, runtime, "MANAGER", original)
        loaded = ["a passage\n\nand its question"]

        frames = list(
            app.run_prompts(
                app.prompts_to_text(loaded), loaded, "", "", *SAMPLING
            )
        )

        rows = frames[-1][RESULTS]["value"]
        self.assertEqual(len(rows), 1)


class BatchCsvTests(unittest.TestCase):
    """The CSV a run adds to as each prompt finishes."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="chatlab-test-"))
        self.addCleanup(shutil.rmtree, self.directory)
        self.addCleanup(os.umask, os.umask(0))

    def rows(self, path):
        return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))))

    def test_each_prompt_adds_its_rows_to_the_one_table(self):
        table = BatchTable(self.directory)

        table.add(sample_trace(seed=1), 1)
        path = table.add(sample_trace(seed=2), 3)
        rows = self.rows(path)

        self.assertEqual([row["prompt_index"] for row in rows], ["1", "3"])
        self.assertEqual([row["seed"] for row in rows], ["1", "2"])

    def test_the_column_names_are_written_once(self):
        # An appended prompt that repeated the header would put a row of
        # column names half way down the table, which every reader would
        # then have to know to skip.
        table = BatchTable(self.directory)
        table.add(sample_trace(seed=1), 1)
        path = table.add(sample_trace(seed=2), 2)
        lines = Path(path).read_text(encoding="utf-8").splitlines()

        self.assertEqual(sum(line.startswith("prompt_index") for line in lines), 1)

    def test_a_prompt_is_not_written_again_by_the_next_one(self):
        # The point of appending: a hundred-prompt run used to serialize the
        # first prompt a hundred times, between generations, holding the model.
        table = BatchTable(self.directory)
        table.add(sample_trace(seed=1), 1)
        first = Path(table.path).read_text(encoding="utf-8")

        table.add(sample_trace(seed=2), 2)
        after = Path(table.path).read_text(encoding="utf-8")

        self.assertTrue(after.startswith(first))

    def test_a_wider_candidate_list_rewrites_the_table(self):
        # The header names the candidate columns, so a prompt carrying more
        # alternatives than it holds cannot simply be appended: the whole
        # table is written again under the wider header, and the rows already
        # in it keep their values.
        table = BatchTable(self.directory)
        table.add(sample_trace(seed=1, candidates=1), 1)
        path = table.add(sample_trace(seed=2, candidates=3), 2)
        rows = self.rows(path)

        self.assertEqual([row["prompt_index"] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["candidate_3_text"], "")
        self.assertEqual(rows[1]["candidate_3_text"], "x")

    def test_a_narrower_prompt_after_a_wide_one_keeps_the_columns(self):
        table = BatchTable(self.directory)
        table.add(sample_trace(seed=1, candidates=3), 1)
        path = table.add(sample_trace(seed=2, candidates=1), 2)
        rows = self.rows(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["candidate_1_text"], "x")
        self.assertEqual(rows[1]["candidate_3_text"], "")

    def test_the_table_is_owner_only_from_the_first_prompt(self):
        table = BatchTable(self.directory)
        path = Path(table.add(sample_trace(), 1))

        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        table.add(sample_trace(), 2)

        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class RunPromptsTests(unittest.TestCase):
    """What a batch publishes, and what it leaves on disk."""

    def setUp(self):
        # Keep the exports out of the shared upload folder this machine's
        # other Gradio apps write into.
        self.uploads = tempfile.TemporaryDirectory(prefix="chatlab-uploads-")
        self.addCleanup(self.uploads.cleanup)
        previous = os.environ.get("GRADIO_TEMP_DIR")
        os.environ["GRADIO_TEMP_DIR"] = self.uploads.name
        if previous is None:
            self.addCleanup(os.environ.pop, "GRADIO_TEMP_DIR", None)
        else:
            self.addCleanup(os.environ.__setitem__, "GRADIO_TEMP_DIR", previous)

        self.original = runtime.MANAGER
        # "Hello world" and then the end of the response, on repeat, so every
        # prompt in a batch is answered the same way.
        runtime.MANAGER = loaded_manager([0, 1, EOS_ID], PIECES, EOS_ID)
        self.addCleanup(setattr, runtime, "MANAGER", self.original)

    def run_batch(self, prompts_text, system_prompt="", prefill="", loaded=()):
        frames = list(
            app.run_prompts(
                prompts_text, list(loaded), system_prompt, prefill, *SAMPLING
            )
        )
        self.assertTrue(frames)
        for frame in frames:
            self.assertEqual(len(frame), len(app.BATCH_OUTPUT_NAMES))
        return frames

    def test_every_prompt_gets_a_row_and_a_trace_file(self):
        final = self.run_batch("first prompt\n\nsecond prompt")[-1]
        rows = final[RESULTS]["value"]
        names = [Path(path).name for path in final[FILES]["value"]]

        self.assertEqual([row[INDEX] for row in rows], [1, 2])
        self.assertEqual(names, ["prompt-001.json", "prompt-002.json", "prompts.csv"])
        self.assertIn("Ran 2 of 2 prompts", final[STATUS])
        self.assertTrue(final[FILES]["visible"])

    def test_a_row_carries_the_answer_and_its_measurements(self):
        final = self.run_batch("say hello")[-1]
        row = final[RESULTS]["value"][0]

        self.assertEqual(row[RESPONSE], "Hello world")
        # Two words and the end-of-response token, which is measured too.
        self.assertEqual(row[TOKENS], 3)
        self.assertEqual(row[SEED], 42)
        self.assertGreater(row[PERPLEXITY], 0)

    def test_each_prompt_runs_in_a_conversation_of_its_own(self):
        # The point of a batch: the second prompt must not see the first, or
        # its measurements would describe a context no row mentions.
        final = self.run_batch("first prompt\n\nsecond prompt")[-1]
        traces = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in final[FILES]["value"][:2]
        ]

        for trace, prompt in zip(traces, ["first prompt", "second prompt"]):
            self.assertEqual(trace["messages"], [{"role": "user", "content": prompt}])

    def test_the_system_prompt_leads_every_conversation(self):
        final = self.run_batch("first\n\nsecond", system_prompt="Be terse.")[-1]
        trace = trace_of(final[FILES])

        self.assertEqual(trace["messages"][0], {"role": "system", "content": "Be terse."})

    def test_the_trace_records_the_sampling_the_row_was_measured_under(self):
        final = self.run_batch("say hello")[-1]
        trace = trace_of(final[FILES])

        self.assertEqual(trace["sampling"]["seed"], 42)
        self.assertEqual(trace["sampling"]["max_new_tokens"], 8)
        self.assertEqual(trace["token_count"], 3)
        self.assertNotIn("assistant_prefill", trace["sampling"])

    def test_a_prefill_is_recorded_with_the_response_it_shaped(self):
        # "Hello" is a token this tokenizer knows, so the prefill is replayed
        # rather than refused, and the trace has to say the answer began with
        # text the model did not choose.
        final = self.run_batch("say hello", prefill="Hello")[-1]
        trace = trace_of(final[FILES])

        self.assertEqual(trace["sampling"]["assistant_prefill"], "Hello")

    def test_the_replayed_prefix_is_counted_in_the_trace(self):
        # The prefill's tokens are measured like sampled ones, so a trace
        # that did not say how many were replayed would have every analysis
        # read them as the model's own choices.
        final = self.run_batch("say hello", prefill="Hello")[-1]
        trace = trace_of(final[FILES])

        self.assertEqual(trace["sampling"]["forced_prefix_tokens"], 1)

    def test_a_response_with_no_prefill_counts_no_replayed_tokens(self):
        final = self.run_batch("say hello")[-1]
        trace = trace_of(final[FILES])

        self.assertNotIn("forced_prefix_tokens", trace["sampling"])

    def test_the_buttons_swap_for_the_run_and_back_again(self):
        frames = self.run_batch("say hello")

        self.assertEqual(frames[0][RUN], gr.update(visible=False))
        self.assertEqual(frames[0][STOP], gr.update(visible=True))
        self.assertEqual(frames[-1][RUN], gr.update(visible=True))
        self.assertEqual(frames[-1][STOP], gr.update(visible=False))

    def test_progress_is_reported_prompt_by_prompt(self):
        statuses = [frame[STATUS] for frame in self.run_batch("first\n\nsecond")]

        self.assertTrue(any("Prompt 1 of 2" in status for status in statuses))
        self.assertTrue(any("Prompt 2 of 2" in status for status in statuses))

    def test_a_cancelled_prompt_keeps_the_tokens_it_produced(self):
        # Gradio throws GeneratorExit into whichever yield the run is parked
        # on, and the lines that write the trace come after the loop. Without
        # a write on the way out, a Stop landing there took the answer with
        # it - the one thing the files written as the run goes prevent.
        run = app.run_prompts("first\n\nsecond", [], "", "", *SAMPLING)
        opening = next(run)
        directory = Path(opening[DIRECTORY])
        next(run)
        run.close()

        traces = sorted(directory.glob("prompt-*.json"))
        self.assertTrue(traces)
        self.assertTrue((directory / "prompts.csv").exists())

    def test_a_failed_prompt_is_not_exported_as_a_stopped_one(self):
        # A failed response is not a response to export, which is what the
        # chat does for a single reply. Exporting the tokens it managed
        # would put a half answer in the table under a row saying it failed.
        manager = runtime.MANAGER
        original = manager.generate

        def fail_after_a_token(*args, **kwargs):
            yield from ()
            raise RuntimeError("out of memory")

        def failing(*args, **kwargs):
            manager.generate = original
            return fail_after_a_token()

        manager.generate = failing
        frames = self.run_batch("first\n\nsecond")
        directory = Path(frames[0][DIRECTORY])

        written = sorted(path.name for path in directory.glob("prompt-*.json"))
        self.assertEqual(written, ["prompt-002.json"])
        self.assertIn("Failed", frames[-1][RESULTS]["value"][0][RESPONSE])

    def test_a_stopped_prompt_says_it_was_stopped(self):
        # The tokens are exact but may not be the whole answer, and a trace
        # read as a finished response would put a truncated answer in an
        # experiment beside whole ones.
        run = app.run_prompts("say hello", [], "", "", *SAMPLING)
        opening = next(run)
        directory = Path(opening[DIRECTORY])
        run.close()

        stopped = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("prompt-*.json"))
        ]
        for trace in stopped:
            self.assertTrue(trace["sampling"].get("stopped"))

    def test_a_finished_prompt_is_not_marked_stopped(self):
        final = self.run_batch("say hello")[-1]
        trace = trace_of(final[FILES])

        self.assertNotIn("stopped", trace["sampling"])

    def test_stopping_publishes_what_the_run_wrote(self):
        run = app.run_prompts("first\n\nsecond", [], "", "", *SAMPLING)
        opening = next(run)
        directory = opening[DIRECTORY]
        next(run)
        run.close()

        _status, _rows, _run, _stop, files, _directory = app.stop_batch(directory)

        names = [Path(path).name for path in files["value"]]
        self.assertIn("prompt-001.json", names)
        self.assertIn("prompts.csv", names)
        self.assertTrue(files["visible"])

    def test_stopping_without_a_directory_leaves_the_files_alone(self):
        _status, _rows, _run, _stop, files, _directory = app.stop_batch(None)

        self.assertEqual(files, gr.skip())

    def test_a_finished_prompt_is_recorded_before_anything_can_cancel_it(self):
        # Stop closes the generator at whichever yield it is parked on. A
        # progress line published after the last update would sit between a
        # prompt finishing and its row and trace being written, and a Stop
        # landing there took a fully generated answer with it. The frame
        # after the opening one now already carries the finished prompt.
        run = app.run_prompts("say hello", [], "", "", *SAMPLING)
        self.addCleanup(run.close)
        next(run)
        frame = next(run)

        self.assertEqual(len(frame[RESULTS]["value"]), 1)
        self.assertEqual(len(frame[FILES]["value"]), 2)

    def test_the_files_grow_as_the_run_does(self):
        # A run stopped half way through keeps what it produced, which is only
        # true if each finished prompt is published as it lands.
        frames = self.run_batch("first\n\nsecond")
        published = [
            len(frame[FILES]["value"])
            for frame in frames
            if isinstance(frame[FILES], dict) and frame[FILES].get("value")
        ]

        # One trace and the table, then two traces and the table.
        self.assertEqual(published[0], 2)
        self.assertEqual(published[-1], 3)

    def test_a_prompt_that_fails_does_not_end_the_run(self):
        manager = runtime.MANAGER
        original = manager.generate
        calls = {"count": 0}

        def fail_the_first(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("out of memory")
            return original(*args, **kwargs)

        manager.generate = fail_the_first
        final = self.run_batch("first\n\nsecond")[-1]
        rows = final[RESULTS]["value"]

        self.assertIn("Failed: out of memory", rows[0][RESPONSE])
        self.assertEqual(rows[1][TOKENS], 3)
        self.assertIn("1 of 2 prompts failed", final[STATUS])

    def test_a_failed_prompt_does_not_hand_its_number_to_the_next_one(self):
        # A trace numbered by how many succeeded would file the second
        # prompt's measurements as the first prompt's, which is the one
        # reading of the run that is worse than losing it.
        manager = runtime.MANAGER
        original = manager.generate
        calls = {"count": 0}

        def fail_the_first(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("out of memory")
            return original(*args, **kwargs)

        manager.generate = fail_the_first
        final = self.run_batch("first\n\nsecond")[-1]
        paths = [Path(path) for path in final[FILES]["value"]]
        table = [path for path in paths if path.name == "prompts.csv"][0]
        rows = list(csv.DictReader(io.StringIO(table.read_text(encoding="utf-8"))))

        self.assertEqual([path.name for path in paths[:-1]], ["prompt-002.json"])
        self.assertEqual({row["prompt_index"] for row in rows}, {"2"})

    def test_a_batch_stops_rather_than_finish_on_weights_it_did_not_start_on(self):
        # A load started from another browser tab waits on the model lock and
        # can take it between two prompts. Finishing the batch there would put
        # two models' measurements in one table under one heading.
        manager = runtime.MANAGER
        original = manager.generate
        calls = {"count": 0}

        def reload_before_the_second(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] > 1:
                # What another tab's load leaves behind: the same manager,
                # holding a different load of the weights.
                manager.load_count += 1
            return original(*args, **kwargs)

        manager.generate = reload_before_the_second
        final = self.run_batch("first\n\nsecond\n\nthird")[-1]
        rows = final[RESULTS]["value"]
        names = [Path(path).name for path in final[FILES]["value"]]

        self.assertEqual([row[INDEX] for row in rows], [1])
        self.assertEqual(names, ["prompt-001.json", "prompts.csv"])
        self.assertIn("Ran 1 of 3 prompts", final[STATUS])
        self.assertIn(app.BATCH_MODEL_CHANGED, final[STATUS])
        self.assertIn("Prompt 2 onwards did not run", final[STATUS])

    def test_an_unloaded_model_is_reported_before_anything_runs(self):
        runtime.MANAGER = self.original.__class__()
        frames = self.run_batch("say hello")

        self.assertEqual(frames[-1][STATUS], app.BATCH_NO_MODEL)
        self.assertEqual(frames[-1][RESULTS], gr.skip())

    def test_an_empty_box_is_reported_rather_than_run(self):
        frames = self.run_batch("   \n\n  ")

        self.assertEqual(frames[-1][STATUS], app.BATCH_NO_PROMPTS)

    def test_a_batch_refuses_while_a_reply_is_being_written(self):
        self.assertTrue(runtime.MANAGER.reserve_generation())
        self.addCleanup(runtime.MANAGER.release_generation)

        frames = self.run_batch("say hello")

        self.assertEqual(frames[-1][STATUS], app.BATCH_BUSY)

    def test_the_generation_slot_comes_back_when_the_run_is_cancelled(self):
        # Gradio closes the generator where it stood. A slot left reserved
        # there would refuse every reply for the rest of the session.
        run = app.run_prompts("first\n\nsecond", [], "", "", *SAMPLING)
        next(run)
        run.close()

        self.assertTrue(runtime.MANAGER.reserve_generation())
        runtime.MANAGER.release_generation()

    def test_chat_refuses_while_a_batch_holds_the_model(self):
        run = app.run_prompts("first\n\nsecond", [], "", "", *SAMPLING)
        self.addCleanup(run.close)
        next(run)

        refusal = list(app.chat("hi", [], "", False, "", 0.0, 1.0, 0, 8, 42, False))[-1]

        self.assertIn(app.BUSY_STATUS, refusal[5])


class BatchTableTests(unittest.TestCase):
    def test_an_excerpt_is_one_line_and_stops_at_the_limit(self):
        excerpt = app.excerpt("a prompt\nwith a second line " + "x" * 200)

        self.assertNotIn("\n", excerpt)
        self.assertLessEqual(len(excerpt), app.EXCERPT_LENGTH)
        self.assertTrue(excerpt.endswith("…"))

    def test_a_short_prompt_is_shown_whole(self):
        self.assertEqual(app.excerpt("say hello"), "say hello")

    def test_a_loaded_paragraph_prompt_is_counted_once(self):
        # The count is the only number on screen before the press, so it has
        # to be the number the run will take, not the number of blocks the
        # box happens to show.
        loaded = ["one\n\ntwo"]

        self.assertEqual(app.count_prompts(app.prompts_to_text(loaded), loaded), "1 prompt.")

    def test_the_count_follows_the_box(self):
        self.assertEqual(app.count_prompts(""), app.PROMPT_COUNT_HINT)
        self.assertEqual(app.count_prompts("one"), "1 prompt.")
        self.assertEqual(app.count_prompts("one\n\ntwo"), "2 prompts.")


class LoadPromptFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="chatlab-test-"))
        self.addCleanup(shutil.rmtree, self.directory)

    def write(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_file_is_added_to_what_is_already_in_the_box(self):
        path = self.write("prompts.txt", "third\n\nfourth")

        text, status, prompts = app.load_prompt_file(str(path), "first\n\nsecond", [])

        self.assertEqual(parse_prompts(text), ["first", "second", "third", "fourth"])
        self.assertIn("Loaded 2 prompts", status)

    def test_an_unreadable_file_leaves_the_box_alone(self):
        path = self.write("prompts.jsonl", "not json\n")

        text, status, prompts = app.load_prompt_file(str(path), "first", [])

        self.assertEqual(text, gr.skip())
        self.assertIn("Could not read that file", status)

    def test_a_file_with_no_prompts_says_so(self):
        path = self.write("prompts.txt", "\n\n   \n")

        text, status, prompts = app.load_prompt_file(str(path), "first", [])

        self.assertEqual(text, gr.skip())
        self.assertIn("No prompts", status)


class StopBatchTests(unittest.TestCase):
    def test_stopping_gives_the_buttons_back_and_keeps_the_rows(self):
        status, results, run, stop, files, directory = app.stop_batch()

        self.assertIn("Stopped", status)
        self.assertEqual(results, gr.skip())
        self.assertEqual(files, gr.skip())
        self.assertEqual(run, gr.update(visible=True))
        self.assertEqual(stop, gr.update(visible=False))


if __name__ == "__main__":
    unittest.main()
