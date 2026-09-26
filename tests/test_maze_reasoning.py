import copy
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
    recorded_spans, summary_rows, truncated_ids, truncation_summary, truncation_test)
from chatlab.extensions.maze_experiments.reasoning_page import parse_responses
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.team import TeamEpisode, stream_team
from maze_support import CONFIG, MAZE, Manager, SteeringManager, VECTOR, call, scored
from ui_support import handlers_by_name


def reply(reasoning, direction, message=None):
    text, ids = call(direction, message)
    return reasoning + "\n" + text, list((reasoning + "\n").encode()) + ids


def team_run(replies, manager=Manager, **config):
    ep = TeamEpisode(MAZE, dict(dict(agents=2, communication=True, team_goal="any"), **config))
    generating = manager(replies)
    # Metrics that spell each token, as the runtime records them.
    generating.generate = scored(generating.generate)
    list(stream_team(ep, generating))
    return ep


def recorded(text, prefilled=False, offset=0):
    """A finished response whose recorded tokens are its bytes, each ID moved by ``offset``."""
    metrics = [{"token_id": byte + offset, "text": chr(byte)} for byte in text.encode()]
    return {"finish_reason": "stop", "text": text, "reasoning_prefilled": prefilled,
            "metrics": metrics + [{"token_id": 0, "text": "<stop>"}]}


def spelled(ids):
    """Bytes decoded as the runtime decodes a prefix: a character cut short reads as a replacement."""
    return bytes(i % 1000 for i in ids if i < 2000).decode(errors="replace")


def encode(kept, text):
    return list(text.encode())


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

    def test_a_malformed_call_uses_up_the_window_a_message_is_kept_in(self):
        # agent-1 says west, then makes a malformed call and a call north, so
        # the west it calls in round 4 falls outside its three calls.
        maze = Maze((".....",) * 5, (1, 2), (4, 4))

        def move(reasoning, direction, message=None):
            text = reply(reasoning, direction, message)[0].replace(MAZE.tool_id(), maze.tool_id())
            return text, list(text.encode()) + [0]
        broken = "<tool_call>\n{not json}\n</tool_call>"
        replies = [move("I will move east.", "east", "I'll take the west side"), move("Go south.", "south"),
                   (broken, list(broken.encode()) + [0]), move("Go north.", "north"),
                   move("Go north.", "north"), move("Go north.", "north"),
                   move("Go west.", "west"), move("Go west.", "west")]
        ep = TeamEpisode(maze, dict(agents=2, communication=True, team_goal="any", round_limit=4))
        list(stream_team(ep, Manager(replies)))
        self.assertEqual(ep.turns[2]["event"]["error"], "invalid_json")
        first = read_responses(ep)[0]
        self.assertEqual(first.message_direction, "west")
        self.assertFalse(first.message_kept)

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

    def test_runs_sharing_a_short_label_stay_two_runs(self):
        import dataclasses
        rows = read_responses(team_run(TEAM_REPLIES))
        twin = [dataclasses.replace(r, run_id=r.run_id[:8] + "f" * 24) for r in rows]
        self.assertEqual(rows[0].run, twin[0].run)
        self.assertEqual(summary_rows(rows + twin)[0][1], 2)

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
    TURN = recorded(reply("I see a wall. I will move east.", "east")[0])
    CALL = '<tool_call>\n{"name": "move", "arguments": {"maze_id": "' + MAZE.tool_id() + '", "direction": "'

    def test_each_cut_keeps_that_share_of_the_reasoning_and_runs_to_the_direction(self):
        self.assertEqual(spelled(truncated_ids(self.TURN, 0, False, encode)), self.CALL)
        self.assertEqual(spelled(truncated_ids(self.TURN, .5, False, encode)), "I see a wall.\n" + self.CALL)
        self.assertEqual(spelled(truncated_ids(self.TURN, 1, False, encode)),
                         "I see a wall. I will move east.\n" + self.CALL)

    def test_a_cut_replays_the_recorded_tokens_and_encodes_only_what_it_inserts(self):
        # Recorded IDs the tokenizer would never produce: every kept token is
        # the response's own, and only the break before the call is encoded.
        turn = recorded(reply("I see a wall. I will move east.", "east")[0], offset=1000)
        whole = truncated_ids(turn, 1, False, encode)
        self.assertTrue(all(i >= 1000 for i in whole))
        half = truncated_ids(turn, .5, False, encode)
        kept = len("I see a wall.")
        self.assertTrue(all(i >= 1000 for i in half[:kept] + half[kept + 1:]))
        self.assertEqual(half[kept], ord("\n"))
        # The break is encoded after the tokens it follows.
        seen = []
        truncated_ids(turn, .5, False, lambda kept_ids, text: seen.append((kept_ids, text)) or encode(kept_ids, text))
        self.assertEqual(seen, [(half[:kept], "\n")])

    def test_a_character_split_across_tokens_is_not_cut_inside(self):
        # A byte-level tokenizer spells "→" in three tokens, and the prefixes
        # ending inside it read as a replacement character.
        turn = recorded(reply("Go → then east.", "east")[0])
        for metric in turn["metrics"][:-1]:
            metric["text"] = "?"
        spans = recorded_spans(turn, spelled)
        self.assertEqual([end is None for end, _ in spans[3:6]], [True, True, False])
        self.assertEqual(spelled(truncated_ids(turn, 1, False, encode, spans)), "Go → then east.\n" + self.CALL)
        self.assertEqual(spelled(truncated_ids(turn, .5, False, encode, spans)), "Go →\n" + self.CALL)

    def test_a_hidden_special_is_replayed_but_not_read(self):
        # The runtime records a hidden special among the metrics and leaves it
        # out of the text; the cut still feeds it.
        turn = recorded(reply("I see a wall. I will move east.", "east")[0])
        turn["metrics"].insert(3, {"token_id": 2000, "text": ""})
        spans = recorded_spans(turn, spelled, hidden={2000})
        self.assertIsNotNone(spans)
        self.assertIn(2000, truncated_ids(turn, 1, False, encode, spans))

    def test_a_hidden_special_just_before_the_call_is_fed_once(self):
        turn = recorded(reply("I will move east.", "east")[0])
        at = [chr(m["token_id"]) for m in turn["metrics"]].index("<")
        turn["metrics"].insert(at, {"token_id": 2000, "text": ""})
        spans = recorded_spans(turn, spelled, hidden={2000})
        recorded_ids = [m["token_id"] for m in turn["metrics"][:-1]]
        whole = truncated_ids(turn, 1, False, encode, spans)
        self.assertEqual(whole.count(2000), 1)
        self.assertEqual(whole, recorded_ids[:len(whole)])
        self.assertEqual(truncated_ids(turn, 0, False, encode, spans).count(2000), 1)

    def test_a_run_whose_token_labels_do_not_add_up_is_read_through_the_tokenizer(self):
        ep = team_run(TEAM_REPLIES)
        ep.turns[1]["metrics"][0]["text"] = "?"
        manager = ReadingManager()
        frames = list(truncation_test(ep, manager, TruncationControl(), {1, 2}))
        self.assertEqual(frames[-1][:2], (2, 2))

    def test_token_boundaries_come_from_the_tokenizer_when_labels_do_not_add_up(self):
        # A tokenizer that drops a word-boundary space when a token is read
        # alone records labels that do not concatenate to the response.
        turn = recorded(reply("I see a wall. I will move east.", "east")[0])
        for metric in turn["metrics"][:-1]:
            metric["text"] = metric["text"].strip() or "_"
        self.assertIsNone(recorded_spans(turn))
        spans = recorded_spans(turn, spelled)
        self.assertEqual(spelled(truncated_ids(turn, .5, False, encode, spans)), "I see a wall.\n" + self.CALL)
        self.assertEqual(spelled(truncated_ids(turn, 1, False, encode, spans)),
                         "I see a wall. I will move east.\n" + self.CALL)

    def test_tokens_that_do_not_spell_the_response_are_refused(self):
        turn = recorded(reply("I will move east.", "east")[0])
        turn["text"] = "Something else. " + turn["text"]
        with self.assertRaisesRegex(ValueError, "do not spell its text"):
            truncated_ids(turn, .5, False, encode)

    def test_a_cut_inside_a_template_opened_reasoning_block_closes_it(self):
        turn = recorded("One two three four</think>\n\n" + call("east")[0], prefilled=True)
        self.assertEqual(spelled(truncated_ids(turn, .5, False, encode)).split("<tool_call>")[0],
                         "One two\n</think>\n\n")
        self.assertEqual(spelled(truncated_ids(turn, 0, False, encode)).split("<tool_call>")[0], "\n</think>\n\n")
        self.assertTrue(spelled(truncated_ids(turn, 1, False, encode)).startswith("One two three four</think>"))

    def test_candidates_count_for_the_direction_they_begin(self):
        metric = {"top_candidates": [{"raw_text": "east", "probability": .5}, {"raw_text": "ea", "probability": .1},
                                     {"raw_text": "north", "probability": .3}, {"raw_text": "}", "probability": .1}]}
        # A direction outside the recorded alternatives is unknown, not impossible.
        self.assertEqual(direction_probabilities(metric), {"north": .3, "east": .6})
        # A token that spells the direction and closes the value chooses it;
        # one that runs on into another word does not.
        closing = {"top_candidates": [{"raw_text": 'west"', "probability": .4},
                                      {"raw_text": "western", "probability": .2}]}
        self.assertEqual(direction_probabilities(closing), {"west": .4})

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

    def test_a_direction_the_alternatives_leave_out_is_read_on_its_own(self):
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager(omit={"west"}, forced_probability=.04)
        frames = list(truncation_test(ep, manager, TruncationControl(), {1}))
        probabilities = frames[-1][2][0].probabilities
        self.assertEqual([p["west"] for p in probabilities], [.04] * len(FRACTIONS))
        self.assertEqual(list(probabilities[0]), ["north", "east", "south", "west"])
        # One pass per cut, and one more for the direction left out.
        self.assertEqual(len(manager.calls), 2 * len(FRACTIONS))
        read = [kwargs["forced_ids"] for _, kwargs in manager.calls[1::2]]
        self.assertTrue(all(ids[-1] == ord("w") for ids in read))
        # Encoded after the cut, in the decoder context it continues.
        cut = manager.calls[0][1]["forced_ids"]
        self.assertIn((cut, "west"), manager.replacements)

    def test_a_candidate_that_would_write_a_space_counts_for_no_direction(self):
        # Its standalone text reads "east", but after the cut it writes " east",
        # a value the call rejects.
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager(spaced=.1)
        frames = list(truncation_test(ep, manager, TruncationControl(), {1}))
        self.assertAlmostEqual(frames[-1][2][0].probabilities[0]["east"], .2)

    def test_a_test_started_before_the_runs_changed_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        control = TruncationControl()
        before, after = {ep.run_id: ep}, {ep.run_id: ep}
        control.publish(after)
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "changed after this test was started"):
            list(truncation_test(ep, manager, control, {1}, runs=before))
        self.assertIsNone(control.holder)
        self.assertEqual(len(list(truncation_test(ep, manager, control, {1}, runs=after))), 2)

    def test_tokens_the_loaded_tokenizer_reads_differently_are_refused(self):
        ep = team_run(TEAM_REPLIES)
        manager = ReadingManager()
        # The recorded text still matches, but the loaded vocabulary spells
        # one of the IDs as something else.
        first = ep.turns[0]["metrics"][0]["token_id"]
        spell = manager.spell
        manager.tokenizer.decode = lambda ids, **kw: spell([1000 if i == first else i for i in ids])
        with self.assertRaisesRegex(ValueError, "Response 1's recorded tokens read differently"):
            list(truncation_test(ep, manager, TruncationControl(), {1}))
        self.assertEqual(manager.calls, [])

    def test_a_response_recording_no_prompt_is_refused(self):
        ep = team_run(TEAM_REPLIES)
        ep.turns[1]["prompt_ids"] = []
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "Response 2 records no prompt"):
            list(truncation_test(ep, manager, TruncationControl(), {1, 2}))
        self.assertEqual(manager.calls, [])

    def test_a_test_is_refused_while_the_runs_are_changing(self):
        # An upload holding the claim refuses a test that starts meanwhile,
        # and the refusal leaves the upload's claim in place.
        ep = team_run(TEAM_REPLIES)
        control = TruncationControl()
        self.assertTrue(control.claim("load"))
        manager = ReadingManager()
        with self.assertRaisesRegex(ValueError, "being changed"):
            list(truncation_test(ep, manager, control))
        self.assertEqual(control.holder, "load")
        self.assertFalse(manager.busy)
        control.release("load")
        list(truncation_test(ep, manager, control, {1}))
        self.assertIsNone(control.holder)
        # Each browser session gets its own claim.
        control.claim("test")
        self.assertIsNone(copy.deepcopy(control).holder)

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
        generating = Manager([reply("I will move east.", "east")] * 2)
        generating.generate = scored(generating.generate)
        list(stream_episode(single, generating))
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
    """Answers a truncation with one token: east is likelier once "east" is in the kept reasoning.

    ``omit`` leaves directions out of the recorded alternatives. A forced
    token is measured at ``forced_probability``, as the runtime measures one.
    """

    # Whole-word tokens past the byte range, so a candidate spells its word.
    WORDS = {1000: "east", 1001: "north", 1002: "south", 1003: "west", 1004: " east"}

    def __init__(self, on_call=None, omit=(), forced_probability=.02, spaced=0.):
        super().__init__(iter(()))
        self.on_call = on_call
        self.omit = set(omit)
        self.forced_probability = forced_probability
        self.spaced = spaced
        self.replacements = []
        self.tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()), decode=self.spell)

    def spell(self, ids, **kwargs):
        text, run = "", []
        for token in list(ids) + [None]:
            if token is not None and token < 256:
                run.append(token)
                continue
            text += bytes(run).decode()
            run = []
            if token is not None:
                text += self.WORDS[token]
        return text

    def encode_replacement(self, kept_ids, text, **kwargs):
        self.replacements.append((list(kept_ids), text))
        return super().encode_replacement(kept_ids, text, **kwargs)

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.on_call:
            self.on_call()
        prefix = kwargs["forced_ids"]
        reasoning = bytes(prefix).decode().split("<tool_call>")[0]
        east = .9 if "east" in reasoning else .3
        candidates = [{"token_id": 1000, "raw_text": "east", "probability": east - self.spaced},
                      {"token_id": 1001, "raw_text": "north", "probability": 1 - east},
                      {"token_id": 1002, "raw_text": "south", "probability": 0.},
                      {"token_id": 1003, "raw_text": "west", "probability": 0.},
                      {"token_id": 1004, "raw_text": "east", "probability": self.spaced}]
        sampled = {"token_id": 101, "top_candidates": [c for c in candidates if c["raw_text"] not in self.omit]}
        forced = [{"token_id": t, "raw_probability": self.forced_probability} for t in prefix]
        yield SimpleNamespace(text="e", metrics=forced + [sampled], prompt_ids=[],
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
                loaded = callbacks["reasoning_load"](paths + [str(broken)], {}, [], TruncationControl())
                held, note, summary, rows, csv_path, pick = loaded[:6]
                self.assertEqual(set(held), {team.run_id, single.run_id})
                self.assertIn("Team of 2 · messages on", note)
                self.assertEqual([row[0] for row in summary], ["One agent", "Team of 2 · messages on"])
                self.assertEqual(len(rows), 6)
                self.assertTrue(Path(csv_path).read_text().startswith("Run,Condition,Response"))
                self.assertEqual(pick["value"], team.run_id)
                button, wanted = callbacks["reasoning_pick"](held, team.run_id)
                self.assertEqual((button["value"], wanted), ("Load test/model", "test/model"))
                control = TruncationControl()
                frames = []
                for frame in callbacks["reasoning_truncate"](held, team.run_id, "1-2", [], control):
                    # The claim is held through every update, the last included.
                    self.assertEqual(control.holder, "test")
                    frames.append(frame)
                self.assertIsNone(control.holder)
                busy = TruncationControl()
                busy.claim("load")
                with self.assertRaisesRegex(gr.Error, "being changed"):
                    list(callbacks["reasoning_truncate"](held, team.run_id, "1-2", [], busy))
                self.assertEqual(busy.holder, "load")
                results, progress, table, per_response, path = frames[-1][:5]
                self.assertEqual(len(results), 2)
                self.assertIn("**Finished** · 2 responses", progress)
                self.assertEqual(table[0][:2], ["Team of 2 · messages on", 2])
                self.assertEqual(len(per_response), 2)
                self.assertIsNotNone(path)
                # Loading the run again replaces it, and what was read from the
                # file it replaces goes with it.
                reloaded = callbacks["reasoning_load"]([paths[0]], held, results, TruncationControl())
                self.assertEqual(reloaded[6:9], ([], [], []))
                self.assertIsNone(reloaded[9])
                other = callbacks["reasoning_load"]([paths[1]], held, results, TruncationControl())
                # Loading another run leaves the results alone.
                self.assertEqual(other[6:], (gr.skip(),) * 4)
                # Nothing loads while a test is reading, since the test would
                # hand back results from the file being replaced.
                busy = TruncationControl()
                busy.claim("test")
                with self.assertRaisesRegex(gr.Error, "Stop the truncation test"):
                    callbacks["reasoning_load"]([paths[0]], held, results, busy)
                # A refusal says so rather than reporting a finished test.
                for turn in held[team.run_id].turns:
                    turn["model_id"] = "someone/else"
                frames = list(callbacks["reasoning_truncate"](held, team.run_id, "all", results, TruncationControl()))
                self.assertIn("**Refused**", frames[-1][1])
                self.assertEqual(frames[-1][0], [])
                # Any other failure still gives the Run button back.
                for turn in held[team.run_id].turns:
                    turn["model_id"] = "test/model"

                def fail():
                    raise RuntimeError("out of memory")
                manager.on_call = fail
                frames = list(callbacks["reasoning_truncate"](held, team.run_id, "all", [], TruncationControl()))
                self.assertIn("**Failed**", frames[-1][1])
                self.assertEqual((frames[-1][5]["visible"], frames[-1][6]["visible"]), (True, False))
                self.assertFalse(manager.busy)
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
