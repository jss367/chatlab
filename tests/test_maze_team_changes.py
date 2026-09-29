"""A team whose map changes under it, and whose agents are sent messages partway through."""
import json
import unittest

from chatlab.extensions.maze_experiments.dynamic_maze import changing
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.page import prompt_reading
from chatlab.extensions.maze_experiments.runner import Episode, context_messages, from_payload, stream_episode
from maze_support import Manager, call

ROOM = changing(Maze(("...", "...", "..."), (0, 0), (0, 2)))
CORRIDOR = Maze((".....", "#####", "#####", "#####", "#####"), (0, 0), (0, 4))


def step(direction, maze=ROOM, message=None):
    return call(direction, message, maze_id=maze.tool_id())[0], [8, 0]


def team(maze=ROOM, **config):
    return Episode(maze, dict(dict(agents=2, communication=False, team_goal="all"), **config))


def saved(ep):
    return json.loads(json.dumps(ep.payload()))


def last_reply(manager, index):
    return json.loads(manager.calls[index][0][-1]["content"])


def altered(ep, change):
    copy = saved(ep)
    change(copy)
    return copy


class TeamClosureTests(unittest.TestCase):
    def closed_run(self):
        ep = team()
        manager = Manager([step("south"), step("south"), step("east"), step("east"), step("east"), step("east")])
        list(stream_episode(ep, manager, single_step=True))
        ep.request_closure((1, 1))
        list(stream_episode(ep, manager, single_step=True))
        return ep, manager

    def test_a_closure_lands_before_the_next_round_for_every_agent(self):
        ep, manager = self.closed_run()
        self.assertTrue(ep.manual_intervention)
        self.assertEqual(ep.config["map_updates"], [dict(before_round=1, positions=[[1, 0], [1, 0]], closed_cell=[1, 1],
                                                         grid=["...", ".#.", "..."])])
        # Both agents walked into the new wall in round 2, having been shown the
        # map as it was, and each reply to that round shows the map they met.
        self.assertEqual([e["error"] for e in ep.events[-2:]], ["blocked_move", "blocked_move"])
        self.assertEqual(last_reply(manager, 3)["grid"], ["...", "...", "..."])
        for agent in ep.agents:
            self.assertEqual(json.loads(agent["messages"][-1]["content"])["grid"], ["...", ".#.", "..."])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_a_closure_is_refused_where_it_would_strand_or_crush_an_agent(self):
        ep = team()
        list(stream_episode(ep, Manager([step("south"), step("east")]), single_step=True))
        for cell, message in (((1, 0), "agent-1: The cell this agent is standing in"),
                              ((0, 1), "agent-2: The cell this agent is standing in"),
                              ((0, 0), "start and the destination")):
            with self.subTest(cell), self.assertRaisesRegex(ValueError, message):
                ep.request_closure(cell)
        ep.request_closure((2, 2))
        with self.assertRaisesRegex(ValueError, "already queued to close before the next round"):
            ep.request_closure((2, 1))
        with self.assertRaisesRegex(ValueError, "fixed"):
            team(maze=CORRIDOR).request_closure((0, 1))
        # The waypoint is never closed, and no agent still on its way to it is
        # walled off from it: past (1, 0), the only way on from the start is
        # through the destination.
        ring = changing(Maze(("...", ".#.", "..."), (0, 0), (0, 2)))
        guarded = team(maze=ring, waypoint=[2, 1])
        with self.assertRaisesRegex(ValueError, "The waypoint is never closed"):
            guarded.request_closure((2, 1))
        with self.assertRaisesRegex(ValueError, "cut agent-1 off from the waypoint"):
            guarded.request_closure((1, 0))

    def test_a_queued_closure_an_agent_has_walked_onto_is_dropped_and_recorded(self):
        ep = team()
        list(stream_episode(ep, Manager([step("south"), step("east")]), single_step=True))
        # Queued as the round began; by the time it lands agent-1 stands on it.
        ep.close_next, ep.manual_intervention = (1, 0), True
        list(stream_episode(ep, Manager([step("east"), step("east")]), single_step=True))
        self.assertNotIn("map_updates", ep.config)
        self.assertEqual(ep.dropped_closures[0]["before_round"], 1)
        self.assertIn("agent-1", ep.dropped_closures[0]["reason"])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_a_stop_records_the_closure_it_left_queued(self):
        ep = team()
        list(stream_episode(ep, Manager([step("south"), step("east")]), single_step=True))
        ep.request_closure((2, 2))
        ep.request_stop()
        self.assertEqual(ep.close_next, ())
        self.assertEqual(ep.dropped_closures, [dict(before_round=1, cell=[2, 2],
                                                    reason="The episode ended before the closure could land.")])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_a_saved_closure_has_to_be_one_the_run_could_have_made(self):
        ep, _ = self.closed_run()
        refused = {
            "a moved agent": (lambda c: c["config"]["map_updates"][0].update(positions=[[1, 0], [2, 0]]),
                              "positions the team's paths never reached"),
            "another map": (lambda c: c["config"]["map_updates"][0].update(grid=["...", "...", ".#."]),
                            "map its own closure does not produce"),
            "a round it never reached": (lambda c: c["config"]["map_updates"][0].update(before_round=5),
                                         "round it never reached"),
            "a wall where an agent stands": (
                lambda c: c["config"]["map_updates"][0].update(closed_cell=[1, 0], grid=["...", "#..", "..."]),
                "could not have made"),
            "nobody intervening": (lambda c: c.update(manual_intervention=False), "nobody intervened"),
            "a second closure before the same round": (lambda c: c["config"]["map_updates"].append(dict(
                c["config"]["map_updates"][0], closed_cell=[2, 2], grid=[".#.", "...", "..#"])), "two map changes"),
            "a fixed map": (lambda c: c["maze"].pop("environment_id"), "changing map"),
            "a finished run still waiting": (lambda c: c.update(phase="arrived", close_next=[2, 2]),
                                             "outcome other than|cannot still be waiting"),
        }
        for name, (change, message) in refused.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                from_payload(altered(ep, change))
        older = altered(ep, lambda c: c.update(format="chatlab-maze-team-1"))
        with self.assertRaisesRegex(ValueError, "has to be recorded as chatlab-maze-team-2"):
            from_payload(older)


class TeamInsertTests(unittest.TestCase):
    def test_a_message_reaches_only_the_agent_it_was_sent_to(self):
        ep = team(maze=CORRIDOR, communication=True)
        manager = Manager([step("east", CORRIDOR)] * 8)
        list(stream_episode(ep, manager, single_step=True))
        ep.request_insert("tool_note", "The exit is east.", index=1, advised_direction="east")
        self.assertIsNone(ep.agents[0]["insert_next"])
        list(stream_episode(ep, manager, single_step=True))
        self.assertNotIn("note", last_reply(manager, 2))
        self.assertEqual(last_reply(manager, 3)["note"], "The exit is east.")
        self.assertEqual(ep.config["context_inserts"], [dict(
            before_round=1, agent=1, channel="tool_note", text="The exit is east.", sender=None, position=[0, 1],
            advised_direction="east")])
        # A teammate message joins the messages the team already sends each other.
        ep.request_insert("teammate", "Keep going.", sender="coach", index=0)
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(last_reply(manager, 4)["messages"], [{"from": "coach", "text": "Keep going."}])
        # A user message is a turn of its own, and the next response is given it.
        ep.request_insert("user", "Hurry.", index=1)
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(manager.calls[7][0][-1], {"role": "user", "content": "Hurry."})
        for index in range(len(ep.turns)):
            self.assertEqual(manager.calls[index][0], context_messages(ep, index))
        self.assertEqual(ep.phase, "arrived")
        read = lambda run, turn, messages: prompt_reading(run, turn, messages, manager)
        self.assertEqual(saved(from_payload(saved(ep), read_prompt=read)), saved(ep))
        with self.assertRaisesRegex(ValueError, "already queued for agent-1|finished"):
            ep.request_insert("user", "Again.", index=0)

        refused = {
            "a moved position": (lambda c: c["config"]["context_inserts"][0].update(position=[0, 0]),
                                 "position its path does not reach"),
            "another agent": (lambda c: c["config"]["context_inserts"][0].update(agent=0), "agents do not match"),
            "other words": (lambda c: c["config"]["context_inserts"][0].update(text="The exit is west."),
                            "agents do not match"),
            "an unread message": (lambda c: c["turns"][3].update(prompt_ids=[]), "records no prompt"),
            "a later response that read it unrecorded": (lambda c: c["turns"][5].update(prompt_ids=[]),
                                                          "records no prompt"),
            "a response never asked for": (lambda c: c["config"]["context_inserts"][0].update(before_round=9),
                                           "never recorded"),
            "two to one response": (lambda c: c["config"]["context_inserts"].append(
                dict(c["config"]["context_inserts"][0])), "one to an agent a round"),
            "nobody intervening": (lambda c: c.update(manual_intervention=False), "nobody intervened"),
        }
        for name, (change, message) in refused.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                from_payload(altered(ep, change))

    def test_a_message_queued_mid_round_waits_for_the_next_round(self):
        # Every response in a round answers the state the round began with, so
        # a message for agent-2 queued while agent-1 is answering waits.
        ep = team(maze=CORRIDOR)
        manager = Manager([step("east", CORRIDOR)] * 4)
        stream = stream_episode(ep, manager, single_step=True)
        next(stream)
        ep.request_insert("tool_note", "Late news.", index=1)
        list(stream)
        self.assertNotIn("Late news.", json.dumps(manager.calls[1][0]))
        self.assertNotIn("context_inserts", ep.config)
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(last_reply(manager, 3)["note"], "Late news.")
        self.assertEqual(ep.config["context_inserts"][0]["before_round"], 1)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_prompts_are_read_under_the_runs_model_when_responses_name_none(self):
        ep = team(maze=CORRIDOR)
        manager = Manager([step("east", CORRIDOR)] * 4)
        list(stream_episode(ep, manager, single_step=True))
        ep.request_insert("tool_note", "The exit is east.", index=1)
        list(stream_episode(ep, manager, single_step=True))
        read = lambda run, turn, messages: prompt_reading(run, turn, messages, manager)
        unnamed = saved(ep)
        for turn in unnamed["turns"]:
            turn.pop("model_id")
        self.assertEqual(saved(from_payload(unnamed, read_prompt=read)), saved(from_payload(unnamed)))
        # A prompt that left the message out is still caught.
        unnamed["turns"][3]["prompt_ids"] = unnamed["turns"][1]["prompt_ids"]
        with self.assertRaisesRegex(ValueError, "recorded prompt in round 2"):
            from_payload(unnamed, read_prompt=read)

    def test_a_message_queued_as_a_round_opens_waits_for_the_next_round(self):
        # Queued from another callback once the round has begun but before its
        # first response is appended, a message still waits: every response in
        # the round answers the state it began with.
        ep = team(maze=CORRIDOR)
        ep.busy = ep.round_open = True
        ep.request_insert("tool_note", "Late news.", index=1)
        self.assertEqual(ep.agents[1]["insert_next"]["for_boundary"], 1)
        # Queued between rounds, one lands in the next round.
        ep = team(maze=CORRIDOR)
        manager = Manager([step("east", CORRIDOR)] * 4)
        list(stream_episode(ep, manager, single_step=True))
        self.assertFalse(ep.round_open)
        ep.request_insert("tool_note", "Between rounds.", index=1)
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.config["context_inserts"][0]["before_round"], 1)

    def test_a_message_whose_response_is_never_generated_is_withdrawn(self):
        ep = team(maze=CORRIDOR)
        manager = Manager([step("east", CORRIDOR)] * 4)
        list(stream_episode(ep, manager, single_step=True))
        ep.request_insert("user", "Stop here.", index=0)
        stream = stream_episode(ep, manager, single_step=True)
        next(stream)
        ep.request_stop()
        list(stream)
        self.assertNotIn("context_inserts", ep.config)
        self.assertEqual(ep.agents[0]["messages"], context_messages(ep, 2))
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_a_stopped_agent_is_sent_nothing(self):
        ep = team(maze=CORRIDOR, team_goal="all")
        list(stream_episode(ep, Manager([("I give up.", list(b"I give up.") + [0]), step("east", CORRIDOR)]),
                            single_step=True))
        with self.assertRaisesRegex(ValueError, "agent-1 has stopped moving"):
            ep.request_insert("user", "Come back.", index=0)


if __name__ == "__main__":
    unittest.main()
