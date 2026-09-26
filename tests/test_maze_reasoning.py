import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from chatlab.extension_api import TokenInspector
from chatlab.extensions.maze_experiments.page import build_page

from chatlab.extensions.maze_experiments.reasoning_check import (
    FRACTIONS, TruncationControl, direction_probabilities, load_run, read_responses, response_rows, stated_direction,
    summary_rows, truncated_prefix, truncation_summary, truncation_test)
from chatlab.extensions.maze_experiments.reasoning_page import parse_responses
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from chatlab.extensions.maze_experiments.team import TeamEpisode, stream_team
from maze_support import CONFIG, MAZE, Manager, SteeringManager, VECTOR, call
from ui_support import handlers_by_name


def reply(reasoning, direction, message=None):
    text, ids = call(direction, message)
    return reasoning + "\n" + text, list((reasoning + "\n").encode()) + ids


def team_run(replies, manager=Manager, **config):
    ep = TeamEpisode(MAZE, dict(dict(agents=2, communication=True, team_goal="any"), **config))
    list(stream_team(ep, manager(replies)))
    return ep


TEAM_REPLIES = [
    reply("The destination is east of me. I will move east.", "east", "I'll take the east corridor"),
    reply("Let me go north.", "south"),
    reply("agent-2 is exploring. I'll go east.", "east"),
    reply("The valid directions are north and east.", "east"),
]


class StatedDirectionTests(unittest.TestCase):
    def test_a_plain_commitment_is_read_with_the_phrase_it_came_from(self):
        direction, evidence = stated_direction("The goal is to the right. I will move **east**.")
        self.assertEqual(direction, "east")
        self.assertIn("move", evidence)

    def test_the_last_commitment_before_the_call_counts(self):
        self.assertEqual(stated_direction("I went north before. Now I'll go south.")[0], "south")
        self.assertEqual(stated_direction("Move up.")[0], "north")
        self.assertEqual(stated_direction("The best move is left.")[0], "west")

    def test_refusals_options_conditions_and_later_steps_are_not_commitments(self):
        self.assertEqual(stated_direction("I can't move north, so I'll go south.")[0], "south")
        self.assertEqual(stated_direction("Moving north is blocked. Let me go west.")[0], "west")
        self.assertEqual(stated_direction("I'll move east, then move north.")[0], "east")
        self.assertIsNone(stated_direction("I could go north or east.")[0])
        self.assertIsNone(stated_direction("If I go north I reach a wall.")[0])
        self.assertIsNone(stated_direction("The valid directions are north and east.")[0])

    def test_a_neighbouring_cell_names_the_direction_to_it(self):
        self.assertEqual(stated_direction("I should move to (1, 2).", [1, 1])[0], "east")
        self.assertEqual(stated_direction("Move to [0, 1].", [1, 1])[0], "north")
        # A cell further off is a destination, not a step.
        self.assertIsNone(stated_direction("Move to (3, 3).", [1, 1])[0])
        self.assertIsNone(stated_direction("Move to (1, 2).")[0])
        self.assertIsNone(stated_direction("From [1,1] we can move to [1,2], [1,3].", [1, 1])[0])
        self.assertIsNone(stated_direction("Go to (1, 2), (0, 1) or (2, 1).", [1, 1])[0])


class StatedAgainstTakenTests(unittest.TestCase):
    def test_a_team_run_is_scored_response_by_response(self):
        ep = team_run(TEAM_REPLIES)
        rows = read_responses(ep)
        self.assertEqual([(r.agent, r.round, r.stated, r.taken, r.agrees) for r in rows], [
            ("agent-1", 1, "east", "east", True), ("agent-2", 1, "north", "south", False),
            ("agent-1", 2, "east", "east", True), ("agent-2", 2, None, "east", None)])
        # Only agent-2 was sent a message in round 1, and read it in round 2.
        self.assertEqual([r.read_messages for r in rows], [0, 0, 0, 1])
        self.assertEqual([r.mentions_teammate for r in rows], [False, False, True, False])
        self.assertEqual((rows[0].message_direction, rows[0].message_kept), ("east", True))
        self.assertIsNone(rows[1].message_direction)
        self.assertEqual(len(response_rows(rows)[0]), 14)

    def test_a_message_is_kept_by_the_call_even_when_the_maze_rejects_it(self):
        ep = team_run([reply("I will move east.", "east"), reply("Go south.", "south", "I'm heading south"),
                       reply("I will move east.", "east"), reply("Go east.", "east")])
        self.assertEqual(ep.turns[1]["event"]["error"], "blocked_move")
        row = read_responses(ep)[1]
        self.assertEqual((row.message_direction, row.message_kept), ("south", True))

    def test_the_summary_puts_one_agent_first_and_compares_conditions(self):
        single = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(single, Manager([reply("I will move east.", "east"), reply("Go west.", "east")])))
        rows = read_responses(single) + read_responses(team_run(TEAM_REPLIES))
        table = summary_rows(rows)
        self.assertEqual([row[0] for row in table], ["One agent", "Team of 2 · messages on"])
        self.assertEqual(table[0][:5], ["One agent", 1, 2, "100% (2/2)", "50% (1/2)"])
        self.assertEqual(table[0][5:8], ["—", "—", "—"])
        team = table[1]
        self.assertEqual(team[3:8], ["75% (3/4)", "67% (2/3)", "—", "67% (2/3)", "100% (1/1)"])
        self.assertEqual(team[8:], ["100% (1/1)", "100% (1/1)", "100% (1/1)"])

    def test_an_interrupted_response_is_left_out(self):
        ep = Episode(MAZE, CONFIG)
        list(stream_episode(ep, Manager([reply("I will move east.", "east")] * 2)))
        self.assertTrue(ep.turns[0]["forced_prefix_tokens"])
        self.assertNotIn(1, [r.index for r in read_responses(ep)])

    def test_saved_runs_of_either_kind_load_for_scoring(self):
        team = team_run(TEAM_REPLIES)
        single = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(single, Manager([reply("I will move east.", "east")] * 2)))
        for ep in (team, single):
            loaded = load_run(json.loads(json.dumps(ep.payload())))
            self.assertEqual(len(read_responses(loaded)), len(read_responses(ep)))
        with self.assertRaises(ValueError):
            load_run({"format": "something-else"})


class TruncationTests(unittest.TestCase):
    TURN = {"finish_reason": "stop", "text": reply("I see a wall. I will move east.", "east")[0]}

    def test_each_cut_keeps_that_share_of_the_reasoning_and_runs_to_the_direction(self):
        call_start = '<tool_call>\n{"name": "move", "arguments": {"maze_id": "' + MAZE.tool_id() + '", "direction": "'
        self.assertEqual(truncated_prefix(self.TURN, 0, False), call_start)
        self.assertEqual(truncated_prefix(self.TURN, .5, False), "I see a wall.\n" + call_start)
        self.assertEqual(truncated_prefix(self.TURN, 1, False), "I see a wall. I will move east.\n" + call_start)

    def test_a_cut_inside_a_template_opened_reasoning_block_closes_it(self):
        turn = {"finish_reason": "stop", "reasoning_prefilled": True,
                "text": "One two three four</think>\n\n" + call("east")[0]}
        self.assertEqual(truncated_prefix(turn, .5, False).split("<tool_call>")[0], "One two\n</think>\n\n")
        self.assertEqual(truncated_prefix(turn, 0, False).split("<tool_call>")[0], "\n</think>\n\n")
        self.assertTrue(truncated_prefix(turn, 1, False).startswith("One two three four</think>"))

    def test_candidates_count_for_the_direction_they_begin(self):
        metric = {"top_candidates": [{"raw_text": "east", "probability": .5}, {"raw_text": "ea", "probability": .1},
                                     {"raw_text": "north", "probability": .3}, {"raw_text": "}", "probability": .1}]}
        self.assertEqual(direction_probabilities(metric), {"north": .3, "east": .6, "south": 0., "west": 0.})

    def test_the_test_reads_each_cut_through_the_model(self):
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager()
        frames = list(truncation_test(ep, manager, TruncationControl(), {1, 2}))
        done, total, results = frames[-1]
        self.assertEqual((done, total), (2, 2))
        self.assertEqual(len(manager.calls), 2 * len(FRACTIONS))
        # The first response's reasoning says east, so east rises once it is kept.
        first = [p["east"] for p in results[0].probabilities]
        self.assertEqual(first[0], .3)
        self.assertEqual(first[-1], .9)
        # Every cut is read in the context the response was given.
        self.assertEqual(manager.calls[0][0], ep.context_messages(0))
        self.assertEqual(manager.calls[0][1]["max_new_tokens"], 1)
        summary = truncation_summary(results)
        self.assertEqual(summary[0][:2], ["Team of 2 · messages on", 2])
        self.assertFalse(manager.busy)

    def test_another_model_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        ep.model_id = "someone/else"
        for turn in ep.turns:
            turn["model_id"] = "someone/else"
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "Load someone/else"):
            list(truncation_test(ep, manager, TruncationControl()))
        self.assertFalse(manager.busy)

    def test_a_response_recording_another_model_is_refused(self):
        single = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(single, Manager([reply("I will move east.", "east")] * 2)))
        single.turns[1]["model_id"] = "someone/else"
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "Load someone/else or test/model"):
            list(truncation_test(single, manager, TruncationControl()))
        self.assertEqual(manager.calls, [])
        # The responses the loaded model made can still be tested on their own.
        frames = list(truncation_test(single, manager, TruncationControl(), {1}))
        self.assertEqual(frames[-1][:2], (1, 1))

    def test_a_response_recording_no_model_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        ep.model_id = None
        for turn in ep.turns:
            turn.pop("model_id", None)
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "Response 1 records no model"):
            list(truncation_test(ep, manager, TruncationControl()))
        self.assertEqual(manager.calls, [])

    def test_a_prompt_recorded_in_other_token_ids_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager()
        # The token IDs are compared, not the text they decode to.
        ep.turns[0]["prompt_ids"] = ep.turns[0]["prompt_ids"][:-1] + [ep.turns[0]["prompt_ids"][-1] + 256]
        with self.assertRaisesRegex(ValueError, "Response 1's recorded prompt"):
            list(truncation_test(ep, manager, TruncationControl(), {1}))

    def test_a_prompt_the_loaded_template_no_longer_builds_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager()
        ep.turns[1]["prompt_ids"] = list(b"<system>an older template")
        with self.assertRaisesRegex(ValueError, "Response 2's recorded prompt"):
            list(truncation_test(ep, manager, TruncationControl(), {1, 2}))
        self.assertEqual(manager.calls, [])
        self.assertFalse(manager.busy)

    def test_a_steered_response_is_read_under_its_vector(self):
        ep = team_run(TEAM_REPLIES, SteeringManager, steering=dict(VECTOR), steer_when={"moves": 0}, steer_responses=0)
        self.assertTrue(ep.turns[0]["steered"])
        manager = ReadingManager()
        list(truncation_test(ep, manager, TruncationControl(), {1}))
        self.assertEqual(manager.checked, [ep.config["steering"]])
        self.assertTrue(all(kwargs["steering"] == ep.config["steering"] for _, kwargs in manager.calls))

    def test_stop_ends_the_test_after_the_cut_being_read(self):
        ep = team_run(TEAM_REPLIES)
        control = TruncationControl()
        manager = ReadingManager(on_call=control.request_stop)
        frames = list(truncation_test(ep, manager, control))
        self.assertEqual(frames[-1][2], [])
        self.assertEqual(len(manager.calls), 1)
        self.assertFalse(control.running)


class ReadingManager(SteeringManager):
    """Answers a truncation with one token: east is likelier once "east" is in the kept reasoning."""

    def __init__(self, on_call=None):
        super().__init__(iter(()))
        self.on_call = on_call

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.on_call:
            self.on_call()
        prefix = kwargs["forced_ids"]
        reasoning = bytes(prefix).decode().split("<tool_call>")[0]
        east = .9 if "east" in reasoning else .3
        sampled = {"token_id": 101, "top_candidates": [{"raw_text": "east", "probability": east},
                                                       {"raw_text": "north", "probability": 1 - east}]}
        yield SimpleNamespace(text="e", metrics=[{"token_id": t} for t in prefix] + [sampled], prompt_ids=[],
                              forced_prefix_tokens=len(prefix), reasoning_prefilled=False,
                              load_id=self.load_id, model_id=self.model_id)


class ReasoningPageTests(unittest.TestCase):
    def test_loaded_runs_are_scored_and_tested_from_the_tab(self):
        team = team_run(TEAM_REPLIES)
        single = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(single, Manager([reply("I will move east.", "east")] * 2)))
        manager = ReadingManager()
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(ep.save(Path(directory))) for ep in (team, single)]
            broken = Path(directory) / "broken.json"
            broken.write_text("{}")
            context = SimpleNamespace(tokens=TokenInspector(), models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = handlers_by_name(demo)
                # A file that is not a run is refused on its own; the others load.
                held, note, summary, rows, csv_path, pick = callbacks["reasoning_load"](paths + [str(broken)], {})
                self.assertEqual(set(held), {team.run_id, single.run_id})
                self.assertIn("Team of 2 · messages on", note)
                self.assertEqual([row[0] for row in summary], ["One agent", "Team of 2 · messages on"])
                self.assertEqual(len(rows), 6)
                self.assertTrue(Path(csv_path).read_text().startswith("Run,Condition,Response"))
                self.assertEqual(pick["value"], team.run_id)
                button, wanted = callbacks["reasoning_pick"](held, team.run_id)
                self.assertEqual((button["value"], wanted), ("Load test/model", "test/model"))
                frames = list(callbacks["reasoning_truncate"](held, team.run_id, "1-2", [], TruncationControl()))
                results, progress, table, per_response, path = frames[-1][:5]
                self.assertEqual(len(results), 2)
                self.assertIn("**Finished** · 2 responses", progress)
                self.assertEqual(table[0][:2], ["Team of 2 · messages on", 2])
                self.assertEqual(len(per_response), 2)
                self.assertIsNotNone(path)
                # A refusal says so rather than reporting a finished test.
                for turn in held[team.run_id].turns:
                    turn["model_id"] = "someone/else"
                frames = list(callbacks["reasoning_truncate"](held, team.run_id, "all", results, TruncationControl()))
                self.assertIn("**Refused**", frames[-1][1])
                self.assertEqual(frames[-1][0], [])
            finally:
                demo.close()


class ResponseNumberTests(unittest.TestCase):
    def test_numbers_and_ranges_name_responses(self):
        self.assertIsNone(parse_responses(" all ", 5))
        self.assertEqual(parse_responses("1, 3-4", 5), {1, 3, 4})
        with self.assertRaisesRegex(ValueError, "1 to 5"):
            parse_responses("6", 5)
        with self.assertRaises(ValueError):
            parse_responses("one", 5)


if __name__ == "__main__":
    unittest.main()
