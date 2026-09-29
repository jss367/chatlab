"""Forking a team at one token of one agent's response."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import gradio as gr

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.page import export_run
from chatlab.extensions.maze_experiments.reasoning_check import read_responses
from chatlab.extensions.maze_experiments.runner import (Episode, context_messages, fork_token_edit, from_payload,
                                                        stream_episode)
from maze_support import Manager, call

CORRIDOR = Maze((".....", "#####", "#####", "#####", "#####"), (0, 0), (0, 4))
ID = CORRIDOR.tool_id()


def step(direction):
    # Recorded as its own bytes, so the fork can check the IDs spell the text.
    text = "\n" + call(direction, maze_id=ID)[0]
    return text, list(text.encode()) + [0]


def team(**config):
    return Episode(CORRIDOR, dict(dict(agents=2, communication=False, team_goal="all"), **config))


def saved(ep):
    return json.loads(json.dumps(ep.payload()))


def fork(ep, turn, token, text, manager):
    with manager.open_session() as session:
        return fork_token_edit(ep, turn, token, text, session)


def two_rounds(**config):
    ep = team(**config)
    manager = Manager([step("east")] * 4)
    list(stream_episode(ep, manager, single_step=True))
    ep.request_insert("tool_note", "Nearly there.", index=1)
    list(stream_episode(ep, manager, single_step=True))
    return ep


class TeamForkTests(unittest.TestCase):
    def test_a_fork_keeps_the_teammates_its_round_already_asked_and_finishes_the_round(self):
        ep = two_rounds()
        parent = copy.deepcopy(saved(ep))
        manager = Manager([step("east"), step("east"), step("east"), step("east")])
        forked = fork(ep, 3, 0, "x", manager)
        self.assertEqual(saved(ep), parent)
        self.assertEqual((forked.rounds, len(forked.turns), forked.phase), (1, 3, "paused"))
        self.assertEqual([a["position"] for a in forked.agents], [(0, 1), (0, 1)])
        self.assertEqual(len(forked.open_round["actions"]), 1)
        # The message that went in before the edited response went in again.
        self.assertEqual(forked.config["context_inserts"], ep.config["context_inserts"])
        self.assertEqual(json.loads(forked.agents[1]["messages"][-1]["content"])["note"], "Nearly there.")
        list(stream_episode(forked, manager, single_step=True))
        # Only agent-2 was asked again, opening with the kept tokens and the replacement.
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(manager.calls[0][1]["forced_ids"], [ord("x")])
        self.assertEqual(manager.calls[0][0], context_messages(forked, 3))
        self.assertEqual(forked.turns[2], {key: value for key, value in ep.turns[2].items()})
        self.assertEqual(forked.turns[3]["token_edit"], forked.token_edit)
        self.assertEqual(forked.token_edit["turn"], 3)
        self.assertEqual(forked.rounds, 2)
        self.assertEqual([a["position"] for a in forked.agents], [(0, 2), (0, 2)])
        self.assertTrue(forked.manual_intervention)
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))
        # The edited response's reasoning is partly the reader's, so the reasoning check leaves it out.
        self.assertEqual([row.index for row in read_responses(forked)], [1, 2, 3])
        refused = {
            "a forgotten edit": (lambda c: c.update(token_edit=None), "names the edit"),
            "another edit": (lambda c: c["token_edit"].update(turn=2), "not one its edited response records"),
            "a replacement it did not plan": (lambda c: [t["token_edit"].update(replacement_ids=[9])
                                                         for t in c["turns"] if "token_edit" in t]
                                              + [c["token_edit"].update(replacement_ids=[9])], "plan the tokens"),
        }
        for name, (change, message) in refused.items():
            copy_ = saved(forked)
            change(copy_)
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                from_payload(copy_)

    def test_a_fork_at_the_first_agent_of_a_round_asks_the_whole_round_again(self):
        ep = two_rounds()
        manager = Manager([step("east"), step("east")])
        forked = fork(ep, 2, 0, "x", manager)
        self.assertIsNone(forked.open_round)
        self.assertEqual((forked.rounds, len(forked.turns)), (1, 2))
        list(stream_episode(forked, manager, single_step=True))
        self.assertEqual([kwargs["forced_ids"] for _, kwargs in manager.calls], [[ord("x")], []])
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))

    def test_a_fork_does_not_carry_an_interruption_asked_for_after_its_edit(self):
        ep = team(interruption_text="Distracted", interrupt_after=9, prefix_tokens=2)
        manager = Manager([step("east")] * 6)
        list(stream_episode(ep, manager, single_step=True))
        list(stream_episode(ep, manager, single_step=True))
        ep.request_interruption(1)
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.agents[1]["intervention_turn"], 5)
        # Forked in round 1, before anyone asked: agent-2 is not interrupted early.
        forked = fork(ep, 1, 0, "x", Manager([]))
        self.assertFalse(forked.agents[1]["interrupt_next"])
        regenerating = Manager([step("east")])
        list(stream_episode(forked, regenerating, single_step=True))
        self.assertEqual(regenerating.calls[0][1]["forced_ids"], [ord("x")])
        self.assertFalse(forked.agents[1]["interrupted"])
        # Forked after the interruption, the request travels with the fork.
        later = fork(ep, 5, 2, "x", Manager([]))
        self.assertTrue(later.agents[1]["interrupt_next"])

    def test_a_fork_that_fails_before_it_regenerates_saves_a_run_that_reads_back(self):
        ep = two_rounds()
        forked = fork(ep, 3, 0, "x", Manager([]))
        # Another load is in memory by the time regeneration starts.
        moved = Manager([step("east")])
        moved.load_id = "test/model#2"
        with tempfile.TemporaryDirectory() as directory:
            list(stream_episode(forked, moved, save_dir=Path(directory)))
            written = json.loads((Path(directory) / f"{forked.run_id}.json").read_text())
        self.assertEqual(forked.phase, "error")
        self.assertEqual(from_payload(written).phase, "error")

    def test_a_team_run_saved_as_team_1_can_be_forked(self):
        # A chatlab-maze-team-1 file's responses record no prefix fields.
        ep = team()
        list(stream_episode(ep, Manager([step("east")] * 2), single_step=True))
        older = saved(ep)
        older["format"] = "chatlab-maze-team-1"
        for key in ("supplied_moves", "manual_intervention", "dropped_closures", "close_next", "token_edit"):
            older.pop(key, None)
        for key in ("interruption_text", "recovery_tokens", "recovery_attempts"):
            older["config"].pop(key, None)
        older["agents"] = [{key: agent[key] for key in ("name", "position", "status", "messages")}
                           for agent in older["agents"]]
        for turn in older["turns"]:
            for key in ("forced_prefix_tokens", "prefix_ids", "prefix_text", "planned_prefix_ids",
                        "planned_prefix_text", "literal_prefill_tokens"):
                del turn[key]
        replay = from_payload(older)
        forked = fork(replay, 1, 0, "x", Manager([]))
        self.assertEqual((forked.rounds, len(forked.turns)), (0, 1))

    def test_a_fork_of_an_interrupted_response_keeps_its_interruption(self):
        ep = team(interruption_text="Distracted", interrupt_after=1, prefix_tokens=2, interrupt_agents=[1])
        manager = Manager([step("east")] * 4)
        list(stream_episode(ep, manager, single_step=True))
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.agents[1]["intervention_turn"], 3)
        with self.assertRaisesRegex(ValueError, "model-generated token"):
            fork(ep, 3, 1, "x", manager)
        forked = fork(ep, 3, 2, "x", Manager([step("east")]))
        self.assertEqual(forked.pending_edit["literal_prefill_tokens"], 2)
        self.assertTrue(forked.pending_edit["interruption_here"])
        self.assertFalse(forked.agents[1]["interrupted"])
        regenerating = Manager([step("east")])
        list(stream_episode(forked, regenerating, single_step=True))
        self.assertEqual(regenerating.calls[0][1]["forced_ids"], [68, 105, ord("x")])
        self.assertEqual((forked.agents[1]["interrupted"], forked.agents[1]["intervention_turn"]), (True, 3))
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))

    def test_a_stopped_fork_discards_the_round_it_left_open(self):
        ep = two_rounds()
        forked = fork(ep, 3, 0, "x", Manager([]))
        stream = stream_episode(forked, Manager([step("east")]))
        next(stream)
        forked.request_stop()
        list(stream)
        self.assertEqual(forked.phase, "stopped")
        self.assertEqual([t.get("outcome") for t in forked.turns[2:]], ["not_applied", "not_applied"])
        self.assertEqual([a["position"] for a in forked.agents], [(0, 1), (0, 1)])
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))


    def test_a_prepared_fork_stopped_before_it_regenerates_saves_a_run_that_reads_back(self):
        # Forked at agent-2's round-2 response: agent-1's is kept in an open
        # round, and the message agent-2 read before it is landed again.
        ep = two_rounds()
        forked = fork(ep, 3, 0, "x", Manager([]))
        self.assertIsNotNone(forked.open_round)
        self.assertIsNotNone(forked.edit_insert)
        with self.assertRaisesRegex(gr.Error, "Generate the edited response before exporting"):
            export_run(forked, Path(tempfile.mkdtemp()))
        with tempfile.TemporaryDirectory() as directory:
            forked.request_stop(Path(directory))
            written = json.loads((Path(directory) / f"{forked.run_id}.json").read_text())
        self.assertEqual((forked.open_round, forked.edit_insert), (None, None))
        self.assertEqual(forked.turns[2]["outcome"], "not_applied")
        self.assertNotIn("context_inserts", forked.config)
        self.assertEqual(from_payload(written).phase, "stopped")


if __name__ == "__main__":
    unittest.main()
