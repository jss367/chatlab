"""A team given what a run of one agent could always be given: supplied moves, a
waypoint, an interruption with its recovery window, and a limit on its calls."""
import json
import unittest

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, context_messages, from_payload, stream_episode
from maze_support import Manager, call

CORRIDOR = Maze((".....", "#####", "#####", "#####", "#####"), (0, 0), (0, 4))
ID = CORRIDOR.tool_id()


def step(direction, message=None):
    # A leading newline, so the call still opens a line after an interruption
    # prefix, and two sampled tokens, the second the stop token.
    return "\n" + call(direction, message, maze_id=ID)[0], [8, 0]


def say(text):
    return text, list(text.encode()) + [0]


def team(**config):
    return Episode(CORRIDOR, dict(dict(agents=2, communication=False, team_goal="all"), **config))


def saved(ep):
    return json.loads(json.dumps(ep.payload()))


def reply_state(manager, call_index):
    return json.loads(manager.calls[call_index][0][-1]["content"])


class TeamSuppliedMovesTests(unittest.TestCase):
    def test_every_agent_starts_after_the_same_supplied_moves_told_as_itself(self):
        ep = team(supplied_moves=2, communication=True)
        self.assertEqual([agent["position"] for agent in ep.agents], [(0, 2), (0, 2)])
        self.assertEqual(ep.supplied_moves, 2)
        self.assertEqual([(e["agent"], e["source"]) for e in ep.events],
                         [(0, "supplied"), (0, "supplied"), (1, "supplied"), (1, "supplied")])
        history = ep.agents[1]["messages"]
        self.assertEqual(len(history), 6)
        self.assertIn("You are agent-2", history[1]["content"])
        last = json.loads(history[-1]["content"])
        self.assertEqual((last["agent"], last["current"], last["messages"]), ("agent-2", [0, 2], []))
        manager = Manager([step("east"), step("east"), step("east"), step("east")])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.rounds, 2)
        # Each response was given its agent's own history, supplied moves included.
        self.assertEqual(manager.calls[1][0], context_messages(ep, 1))
        self.assertEqual(len(context_messages(ep, 2)), 8)
        replay = from_payload(saved(ep))
        self.assertEqual(saved(replay), saved(ep))
        forged = saved(ep)
        forged["supplied_moves"] = 0
        with self.assertRaisesRegex(ValueError, "supplied-move count"):
            from_payload(forged)


class TeamWaypointTests(unittest.TestCase):
    def test_each_agent_is_told_whether_it_has_passed_the_waypoint_itself(self):
        ep = team(waypoint=[0, 1])
        manager = Manager([step("east"), step("west"), step("east"), step("east")])
        list(stream_episode(ep, manager, single_step=True))
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.waypoint_turn_of(0), 0)
        self.assertEqual(ep.waypoint_turn_of(1), 3)
        first, second = reply_state(manager, 2), reply_state(manager, 3)
        self.assertEqual((first["waypoint"], first["waypoint_reached"]), ([0, 1], True))
        self.assertEqual((second["current"], second["waypoint_reached"]), ([0, 0], False))
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))


class TeamInterruptionTests(unittest.TestCase):
    def test_only_the_agents_the_run_interrupts_get_the_prefix_and_each_recovers_on_its_own(self):
        ep = team(interruption_text="Distracted", interrupt_after=1, prefix_tokens=2, interrupt_agents=[1])
        manager = Manager([step("east"), step("east"), step("east"), step("east"),
                           step("east"), step("east"), step("east"), step("east")])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        # agent-2 reached its trigger after its first move; agent-1 is never interrupted.
        self.assertEqual([kwargs["forced_ids"] for _, kwargs in manager.calls][:4], [[], [], [], [68, 105]])
        first, second = ep.agents
        self.assertFalse(first["interrupted"])
        self.assertEqual((second["interrupted"], second["intervention_turn"], second["resumed"]), (True, 3, True))
        self.assertEqual((second["intervention_tokens"], second["intervention_attempts"], second["latency"]), (2, 1, 2))
        # The interrupted response opens with the prefix, which is not counted as sampled.
        self.assertEqual(ep.turns[3]["prefix_ids"], [68, 105])
        self.assertEqual(ep.turns[3]["sampled_tokens"], 2)
        self.assertEqual(ep.turns[3]["text"][:2], "Di")
        self.assertFalse(ep.manual_intervention)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

        def altered(change):
            copy = saved(ep)
            change(copy)
            return copy

        def moved_to_agent_one(copy):
            for key in ("forced_prefix_tokens", "prefix_ids", "planned_prefix_ids", "literal_prefill_tokens"):
                copy["turns"][2][key], copy["turns"][3][key] = copy["turns"][3][key], copy["turns"][2][key]

        refused = {
            "a skipped interruption": (lambda c: c["turns"][3].update(
                forced_prefix_tokens=0, prefix_ids=[], planned_prefix_ids=[], literal_prefill_tokens=0),
                "due its interruption"),
            "an interruption on an agent the run leaves alone": (moved_to_agent_one, "could not have given it"),
            "a prefix other than the tokens it opens with": (
                lambda c: c["turns"][3].update(prefix_ids=[1, 2], planned_prefix_ids=[1, 2]), "not the prefix"),
            "a forged recovery": (lambda c: c["agents"][1].update(latency=1), "agents do not match"),
            "a forged interruption count": (lambda c: c["agents"][0].update(interrupted=True), "agents do not match"),
            "a queued interruption nobody asked for": (
                lambda c: c["agents"][0].update(interrupt_next=True), "nobody asked|anyone asked"),
        }
        for name, (change, message) in refused.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                from_payload(altered(change))

    def test_an_agent_that_does_not_come_back_in_its_window_stops_and_its_teammate_carries_on(self):
        blocked = step("north")
        ep = team(interruption_text="Distracted", interrupt_after=0, prefix_tokens=2, interrupt_agents=[0],
                  recovery_attempts=1)
        manager = Manager([blocked, step("east"), step("east"), step("east"), step("east")])
        list(stream_episode(ep, manager))
        self.assertEqual([agent["status"] for agent in ep.agents], ["not_recovered", "arrived"])
        self.assertEqual((ep.agents[0]["resumed"], ep.agents[0]["first_move_progress"]), (False, False))
        # agent-1 was out after round 1, so every later round asked agent-2 alone.
        self.assertEqual([turn["agent"] for turn in ep.turns], [0, 1, 1, 1, 1])
        # Every agent that did not arrive was stopped by a limit of the run's.
        self.assertEqual(ep.phase, "budget")
        self.assertIn("agent-1 not recovered", ep.detail)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_each_interrupted_response_is_capped_by_what_is_left_of_its_agents_window(self):
        ep = team(interruption_text="Distracted", interrupt_after=0, prefix_tokens=2, recovery_tokens=50,
                  per_turn_tokens=100)
        manager = Manager([step("north"), step("north"), step("east"), step("east")])
        list(stream_episode(ep, manager, single_step=True))
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual([kwargs["max_new_tokens"] for _, kwargs in manager.calls], [50, 50, 48, 48])
        self.assertEqual([agent["resumed"] for agent in ep.agents], [True, True])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        # A response that sampled past its window is one the run could not have produced.
        greedy = saved(ep)
        greedy["config"]["recovery_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "more tokens than its round allowed"):
            from_payload(greedy)

    def test_a_reader_can_interrupt_one_agent_before_its_trigger(self):
        ep = team(interruption_text="Distracted", interrupt_after=9, prefix_tokens=2)
        with self.assertRaisesRegex(ValueError, "not one of the agents"):
            team(interruption_text="Distracted", interrupt_agents=[0]).request_interruption(1)
        ep.request_interruption(1)
        self.assertTrue(ep.manual_intervention)
        with self.assertRaisesRegex(ValueError, "already been interrupted"):
            manager = Manager([step("east"), step("east")])
            list(stream_episode(ep, manager, single_step=True))
            ep.request_interruption(1)
        self.assertEqual([kwargs["forced_ids"] for _, kwargs in manager.calls], [[], [68, 105]])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        # Without the record of someone asking, an early interruption is one the run could not have given.
        unasked = saved(ep)
        unasked["manual_intervention"] = False
        unasked["agents"][1]["interrupt_next"] = False
        with self.assertRaisesRegex(ValueError, "could not have given it"):
            from_payload(unasked)
        # Nor can the request be credited to another agent than the one interrupted.
        moved = saved(ep)
        moved["agents"][0]["interrupt_next"], moved["agents"][1]["interrupt_next"] = True, False
        with self.assertRaisesRegex(ValueError, "could not have given it"):
            from_payload(moved)

    def test_a_response_the_recovery_window_cuts_off_leaves_its_agent_not_recovered(self):
        ep = team(interruption_text="Distracted", interrupt_after=0, prefix_tokens=2, interrupt_agents=[0],
                  recovery_tokens=1)
        # agent-1's interrupted response runs into the one token its window allows.
        manager = Manager([("\n", [8]), step("east"), step("east"), step("east"), step("east")])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.turns[0]["outcome"], "cut_off")
        self.assertEqual([agent["status"] for agent in ep.agents], ["not_recovered", "arrived"])
        self.assertEqual(ep.phase, "budget")
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_blank_interruption_text_interrupts_nobody(self):
        ep = team(interruption_text="   ")
        manager = Manager([step("east")] * 8)
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        self.assertTrue(all(kwargs["forced_ids"] == [] for _, kwargs in manager.calls))

    def test_a_team_stops_an_agent_that_has_made_every_call_the_run_allows(self):
        ep = team(attempt_budget=1, team_goal="any")
        manager = Manager([step("east"), step("north"), step("east")])
        list(stream_episode(ep, manager))
        self.assertEqual([agent["status"] for agent in ep.agents], ["out_of_calls", "out_of_calls"])
        self.assertEqual(ep.phase, "budget")
        self.assertEqual(len(manager.calls), 2)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))


if __name__ == "__main__":
    unittest.main()
