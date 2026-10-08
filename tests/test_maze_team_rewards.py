"""Team runs with exits and rewards: several exits, responses after arriving, a taste of the vector, and laps."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from chatlab.extension_api import TokenInspector
from chatlab.extensions.maze_experiments.page import build_page

from chatlab.extensions.maze_experiments.maze import Maze, generate_paired, parting_cell
from chatlab.extensions.maze_experiments.reasoning_check import read_responses
from chatlab.extensions.maze_experiments.runner import context_messages, fork_token_edit, from_payload, stream_episode
from chatlab.extensions.maze_experiments.team import REWARD_DEFAULTS, REWARD_FORMAT
from chatlab.extensions.maze_experiments.team_views import (mail_text, positions_after, response_line, reward_status,
                                                            statuses_after, team_board, team_history_rows, team_status)
from maze_support import Manager, SteeringManager, VECTOR, call, saved, say, scenario, scored, team_episode
from ui_support import listeners_by_name

# The start in the middle of a corridor, exit A at its west end and exit B at its east end.
HALL = Maze((".....", "#####", "#####", "#####", "#####"), (0, 2), (0, 0))
HALL_ID = HALL.tool_id()
REWARD = dict(communication=True, team_goal="all", exits=[[0, 4]], reward_exit="B", arrival_responses=1,
              taste=True, laps=2, steering=VECTOR, taste_strength=1.0)


def move(direction, message=None):
    return call(direction, message, maze_id=HALL_ID)


# Round 0 is the taste. In lap 1 agent-1 walks west to exit A and agent-2 east
# to exit B, each writing once after arriving. In lap 2 both walk east to B.
SCRIPT = [say("I feel calm."), say("I feel bright."),
          move("west"), move("east"),
          move("west"), move("east", "heading east"),
          say("Exit A was quiet."), say("Exit B felt wonderful. Come east."),
          move("east"), move("east"),
          move("east"), move("east"),
          say("East it is."), say("Wonderful again.")]


def run(script=SCRIPT, **changes):
    ep = team_episode(HALL, REWARD | changes)
    manager = SteeringManager(list(script))
    manager.generate = scored(manager.generate)
    list(stream_episode(ep, manager))
    return ep, manager


class RewardRunTests(unittest.TestCase):
    def test_two_laps_with_taste_and_a_rewarded_exit(self):
        ep, manager = run()
        self.assertEqual(ep.phase, "arrived", ep.detail)
        self.assertEqual(ep.lap_rounds, [4])
        self.assertEqual([(t["round"], t["agent"], t["kind"]) for t in ep.turns][:4],
                         [(0, 0, "taste"), (0, 1, "taste"), (1, 0, "move"), (1, 1, "move")])
        self.assertEqual([e.get("exit") for e in ep.events if e["arrived"]], ["A", "B", "B", "B"])
        # The taste at its own strength, then only the responses after arriving at B.
        steered = [(t["round"], t["agent"]) for t in ep.turns if t["steered"]]
        self.assertEqual(steered, [(0, 0), (0, 1), (3, 1), (6, 0), (6, 1)])
        strengths = [kw["steering"] and kw["steering"]["strength"] for _, kw in manager.calls]
        self.assertEqual(strengths[:2], [1.0, 1.0])
        self.assertEqual(strengths[7], VECTOR["strength"])
        self.assertIsNone(strengths[6])
        # The taste comes first, and the task follows it.
        first = ep.agents[0]["messages"]
        self.assertEqual(first[1]["content"], ep.config["taste_prompt"])
        self.assertEqual(first[2], {"role": "assistant", "content": "I feel calm."})
        self.assertIn('"exits":{"A":[0,0],"B":[0,4]}', first[3]["content"])
        self.assertIn("every agent has reached an exit", first[3]["content"])
        # A message after arriving waits for agent-1 until lap 2 begins.
        lap_start = next(m for m in first if m["role"] == "user" and m["content"].startswith("Lap 2 of 2"))
        state = json.loads(lap_start["content"].split("\n", 1)[1])
        self.assertEqual(state["messages"], [{"from": "agent-2", "text": "Exit B felt wonderful. Come east."}])
        self.assertEqual(state["current"], [0, 2])
        self.assertEqual([m.get("after_arrival", False) for m in ep.mail], [False, True, True, True, True])
        self.assertEqual(ep.agents[0]["inbox"], [{"from": "agent-2", "text": "Wonderful again."}])
        # Every response's history is the one replay rebuilds.
        payload = saved(ep)
        self.assertEqual(payload["format"], REWARD_FORMAT)
        self.assertEqual(saved(from_payload(payload)), payload)
        self.assertIn("Exit B", team_board(ep))
        self.assertIn("Laps", team_status(ep))
        rows = [row for _, row in team_history_rows(ep)]
        self.assertTrue(any(row[4].startswith("Taste") for row in rows))
        self.assertTrue(any("Exit B felt wonderful" in row[5] for row in rows))

    def test_words_alone_steer_nothing_and_replay(self):
        ep, manager = run(steering=None, taste_strength=None)
        self.assertEqual(ep.phase, "arrived")
        self.assertTrue(all("steered" not in t for t in ep.turns))
        self.assertTrue(all(kw["steering"] is None for _, kw in manager.calls))
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_reward_at_the_one_destination_with_no_extra_exit(self):
        script = [move("west"), move("west"), move("west"), move("west"), say("Done."), say("Done too.")]
        ep, _ = run(script, exits=[], reward_exit="A", taste=False, laps=1)
        self.assertEqual(ep.phase, "arrived")
        self.assertIn('"destination":[0,0]', ep.agents[0]["messages"][1]["content"])
        self.assertEqual(ep.agents[0]["messages"][-2]["content"],
                         "You reached the destination. " + ep.config["arrival_prompt"])
        self.assertEqual([t["steered"] for t in ep.turns], [False] * 4 + [True, True])
        from_payload(saved(ep))

    def test_cut_off_response_after_arriving_sends_what_it_wrote_and_keeps_the_agent(self):
        script = list(SCRIPT)
        script[7] = ("x" * 100, list(b"x" * 100))
        ep, _ = run(script)
        self.assertEqual(ep.phase, "arrived")
        turn = next(t for t in ep.turns if t["kind"] == "arrival" and t["agent"] == 1)
        self.assertNotEqual(turn["finish_reason"], "stop")
        sent = next(m for m in ep.mail if m["sender"] == "agent-2" and m.get("after_arrival"))
        self.assertTrue(sent["text"])
        self.assertEqual(set(sent["text"]), {"x"})
        self.assertEqual(ep.agents[1]["status"], "arrived")
        from_payload(saved(ep))

    def test_a_call_written_after_arriving_is_taken_out_of_the_message(self):
        script = list(SCRIPT)
        script[6] = say("Exit A was quiet.\n" + call("west", maze_id=HALL_ID)[0])
        script[7] = say("Go east. <tool_call>\n<function=move>")
        ep, _ = run(script)
        sent = {m["sender"]: m["text"] for m in ep.mail if m.get("after_arrival") and m["round"] == 3}
        self.assertEqual(sent, {"agent-1": "Exit A was quiet.", "agent-2": "Go east."})
        self.assertNotIn("tool_call", json.dumps(ep.mail))
        from_payload(saved(ep))

    def test_a_tag_cut_in_half_is_taken_out_of_the_message(self):
        from chatlab.extensions.maze_experiments.runner import spoken_text
        for tail in ("<tool_call", "<tool_c", "<", "</tool_call", "</tool", "</"):
            self.assertEqual(spoken_text("Go east. " + tail), "Go east.", tail)
        self.assertEqual(spoken_text('Go east. <tool_call\n{"name": "move"'), "Go east.")
        self.assertEqual(spoken_text("Go east.</tool_call"), "Go east.")

    def test_the_default_taste_prompt_names_no_vector(self):
        ep, _ = run()
        self.assertNotIn("vector", ep.config["taste_prompt"].lower())
        self.assertNotIn("activation", ep.config["taste_prompt"].lower())

    def test_long_message_after_arriving_is_cut_to_the_limit(self):
        script = list(SCRIPT)
        script[7] = say("y" * 290)
        ep, _ = run(script, per_turn_tokens=400)
        self.assertEqual(ep.mail[2]["text"], "y" * 280)
        from_payload(saved(ep))

    def test_round_limit_mid_lap_ends_the_run(self):
        ep, _ = run(round_limit=5)
        self.assertEqual((ep.phase, ep.rounds), ("budget", 5))
        from_payload(saved(ep))

    def test_a_taste_at_its_own_strength_is_steered_when_the_reward_strength_is_zero(self):
        ep, manager = run(steering=dict(VECTOR, strength=0.0), taste_strength=1.0)
        self.assertEqual([(t["round"], t["agent"]) for t in ep.turns if t["steered"]], [(0, 0), (0, 1)])
        self.assertEqual([kw["steering"]["strength"] for _, kw in manager.calls[:2]], [1.0, 1.0])
        self.assertEqual(manager.checked, [dict(VECTOR, strength=0.0)])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        ep, manager = run(steering=dict(VECTOR, enabled=False), taste_strength=1.0)
        self.assertFalse(any(t["steered"] for t in ep.turns))
        self.assertEqual(manager.checked, [])

    def test_a_vector_with_no_reward_exit_and_no_steered_taste_is_not_checked(self):
        # The vector steers only after arriving at the reward exit, so with none
        # and no taste at a strength of its own, a load that refuses it can run.
        for changes in (dict(taste=False), dict(taste_strength=0.0)):
            with self.subTest(**changes):
                script = SCRIPT if changes.get("taste", True) else SCRIPT[2:]
                ep = team_episode(HALL, REWARD | dict(reward_exit=None) | changes)
                manager = SteeringManager(list(script), refuse="This load cannot steer.")
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, "arrived", ep.detail)
                self.assertEqual(manager.checked, [])
                self.assertFalse(any(t["steered"] for t in ep.turns))
                self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        ep = team_episode(HALL, REWARD | dict(reward_exit=None))
        manager = SteeringManager(list(SCRIPT), refuse="This load cannot steer.")
        with self.assertRaisesRegex(Exception, "cannot steer"):
            list(stream_episode(ep, manager))
        self.assertEqual(ep.turns, [])

    def test_run_details_name_the_responses_the_run_steers(self):
        def details(**changes):
            return team_status(run(**changes)[0]).split("**Exits and rewards:** ", 1)[1].split("\n", 1)[0]
        self.assertIn("1 steered response after arriving", details())
        self.assertIn("taste at strength 1", details())
        # A vector at strength 0 steers the taste at its own strength and nothing after arriving.
        line = details(steering=dict(VECTOR, strength=0.0), taste_strength=1.0)
        self.assertIn("1 unsteered response after arriving", line)
        self.assertIn("taste at strength 1", line)
        # A vector switched off steers neither, and neither does a taste at strength 0.
        line = details(steering=dict(VECTOR, enabled=False), taste_strength=1.0)
        self.assertIn("1 unsteered response after arriving", line)
        self.assertIn("taste, unsteered", line)
        line = details(taste_strength=0.0)
        self.assertIn("1 steered response after arriving", line)
        self.assertIn("taste, unsteered", line)
        self.assertIn("1 unsteered, no vector response after arriving", details(steering=None, taste_strength=None))

    def test_a_lap_is_not_begun_on_the_last_round(self):
        # Lap 1 ends with round 4 (the taste, two moves, one message each).
        ep, _ = run(round_limit=4)
        self.assertEqual((ep.phase, ep.rounds, ep.lap_rounds), ("budget", 4, []))
        self.assertIn("round limit after lap 1 of 2", ep.detail)
        self.assertEqual([a["position"] for a in ep.agents], [(0, 0), (0, 4)])
        self.assertEqual([a["status"] for a in ep.agents], ["arrived", "arrived"])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_every_response_joins_the_history_in_the_form_the_run_names(self):
        # Moves, the taste and the responses after arriving, each with a reasoning block.
        script = [say("<think>Weighing it.</think>" + text) for text, _ in SCRIPT]
        for form in ("reasoning_content", "content"):
            with self.subTest(form):
                ep, _ = run(script, reasoning_history=form)
                self.assertEqual(ep.phase, "arrived", ep.detail)
                responses = [m for m in ep.agents[1]["messages"] if m["role"] == "assistant"]
                self.assertEqual(len(responses), 7)
                if form == "reasoning_content":
                    self.assertEqual(responses[0], {"role": "assistant", "reasoning_content": "Weighing it.",
                                                    "content": "I feel bright."})
                    self.assertTrue(responses[1]["content"].startswith("<tool_call>"))
                    self.assertEqual(responses[3]["content"], "Exit B felt wonderful. Come east.")
                    self.assertTrue(all(m["reasoning_content"] == "Weighing it." for m in responses))
                else:
                    self.assertTrue(all(set(m) == {"role", "content"}
                                        and m["content"].startswith("<think>Weighing it.</think>") for m in responses))
                self.assertEqual([m["text"] for m in ep.mail[1:3]],
                                 ["Exit A was quiet.", "Exit B felt wonderful. Come east."])
                self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_agents_that_arrive_on_their_last_call_begin_no_lap(self):
        ep, manager = run(SCRIPT[:8] + [move("east")] * 4, attempt_budget=2)
        self.assertEqual((ep.phase, ep.rounds, ep.lap_rounds), ("budget", 4, []))
        self.assertIn("none has the tokens or calls for lap 2 of 2", ep.detail)
        self.assertEqual(len(manager.calls), 8)
        self.assertEqual([a["status"] for a in ep.agents], ["arrived", "arrived"])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_an_agent_that_arrives_on_its_last_token_sits_out_the_next_lap(self):
        script = list(SCRIPT[:8])
        script[7] = say("B" * 250)
        # agent-2 arrives and writes on exactly the last of its tokens; agent-1 has some left.
        budget = sum(len(ids) for _, ids in script[1::2])
        ep, manager = run(script + [move("east"), move("east"), say("Done.")], agent_token_budget=budget,
                          per_turn_tokens=400)
        self.assertEqual(ep.lap_rounds, [4])
        self.assertEqual(ep.agent_tokens()[1], budget)
        self.assertEqual(ep.agents[1]["status"], "out_of_tokens")
        self.assertEqual(ep.agents[1]["position"], (0, 4))
        # It stays at the exit its last lap ended at, and records that exit.
        self.assertEqual(ep.agents[1]["exit"], "B")
        self.assertFalse(any(t["agent"] == 1 for t in ep.turns[8:]))
        self.assertFalse(any(m["role"] == "user" and m["content"].startswith("Lap 2")
                             for m in ep.agents[1]["messages"]))
        self.assertEqual((ep.agents[0]["status"], ep.agents[0]["exit"]), ("arrived", "B"))
        self.assertEqual(ep.phase, "budget", ep.detail)
        self.assertEqual(len(manager.calls), 11)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_configurations_no_run_could_use_are_refused(self):
        cases = {
            "at least one": dict(arrival_responses=0),
            "one of the run's exits": dict(reward_exit="C"),
            "through another exit": dict(exits=[[0, 1]]),
            "apart from the start": dict(exits=[[0, 2]]),
            "Every agent arrives": dict(team_goal="any"),
            "a waypoint": dict(waypoint=[0, 3]),
            "an interruption": dict(interruption_text="Wait"),
            "supplied moves": dict(supplied_moves=1),
            "no other steering trigger": dict(steer_when={"moves": 1}),
            "between 1 and 32": dict(laps=0),
            "taste strength needs": dict(steering=None, taste_strength=1.0),
            "tool syntax": dict(arrival_prompt="<tool_call>"),
        }
        for message, changes in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                team_episode(HALL, REWARD | changes)
        from chatlab.extensions.maze_experiments.runner import Episode
        with self.assertRaisesRegex(ValueError, "a team's"):
            Episode(HALL, dict(exits=[[0, 4]], supplied_moves=0))

    def test_files_its_responses_could_not_have_made_are_refused(self):
        ep, _ = run()
        good = saved(ep)

        def forged(change):
            data = copy.deepcopy(good)
            change(data)
            return data

        cases = {
            "steered flag": lambda d: d["turns"][7].update(steered=False),
            "kind of response": lambda d: d["turns"][0].update(kind="move"),
            "history": lambda d: d["turns"][2].update(history_length=2),
            "messages do not match": lambda d: d["mail"][1].update(text="Exit A felt wonderful."),
            "lap starts": lambda d: d.update(lap_rounds=[5]),
            "agents do not match": lambda d: d["agents"][0].update(inbox=[]),
            "recorded as chatlab-maze-team-3": lambda d: d.update(format="chatlab-maze-team-2"),
        }
        for message, change in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                from_payload(forged(change))
        plain = forged(lambda d: [d["config"].pop(key) for key in list(d["config"]) if key in REWARD_DEFAULTS])
        with self.assertRaisesRegex(ValueError, "recorded as chatlab-maze-team-3"):
            from_payload(plain)

    def test_every_prompt_is_read_against_the_history_it_records(self):
        ep, _ = run()
        data = saved(ep)
        seen = []

        def read_prompt(episode, turn, context):
            seen.append(len(context))
            return json.dumps(context), json.dumps(context)

        from_payload(data, read_prompt=read_prompt)
        self.assertEqual(len(seen), len(ep.turns))
        self.assertEqual(seen[:2], [2, 2])

        def wrong(episode, turn, context):
            return "a prompt", json.dumps(context)

        with self.assertRaisesRegex(ValueError, "recorded prompt"):
            from_payload(data, read_prompt=wrong)


    def test_round_by_round_matches_one_pass(self):
        whole, _ = run()
        ep = team_episode(HALL, REWARD)
        manager = SteeringManager(list(SCRIPT))
        manager.generate = scored(manager.generate)
        while ep.phase not in ("arrived", "budget", "abandoned"):
            list(stream_episode(ep, manager, single_step=True))
            self.assertEqual(saved(from_payload(saved(ep)))["agents"], saved(ep)["agents"])
        self.assertEqual([t["text"] for t in ep.turns], [t["text"] for t in whole.turns])
        self.assertEqual(ep.mail, whole.mail)
        self.assertEqual(ep.lap_rounds, whole.lap_rounds)

    def test_views_follow_the_laps(self):
        ep, _ = run()
        # After round 3 lap 2 has begun, so both agents are back at the start.
        self.assertEqual(positions_after(ep, 2), [(0, 0), (0, 4)])
        self.assertEqual(positions_after(ep, 3), [(0, 2), (0, 2)])
        self.assertEqual(positions_after(ep, 4), [(0, 3), (0, 3)])
        self.assertEqual(statuses_after(ep, 2), ["arrived", "arrived"])
        self.assertEqual(statuses_after(ep, 4), ["active", "active"])
        self.assertIn("Lap 2 of 2", team_board(ep, 4))
        self.assertIn("Lap 1 of 2", team_board(ep, 1))
        self.assertIn("Reward exit B", team_board(ep))
        rows = [row for _, row in team_history_rows(ep)]
        self.assertIn(["Round 5", "All", "(0, 2)", "—", "Lap 2 begins", "—"], rows)
        self.assertIn("After arriving at B · steered", [row[4] for row in rows])
        self.assertIn("After arriving", response_line(ep, 7))
        self.assertIn("after arriving", mail_text(ep))
        self.assertIn("exits A → B", team_status(ep))
        # The responses written in place of a move have no call for the reasoning check to read.
        self.assertTrue(all(ep.turns[row.index - 1].get("kind", "move") == "move" for row in read_responses(ep)))

    def test_a_fork_of_a_response_after_arriving_regenerates_it_and_continues(self):
        ep, _ = run()
        manager = SteeringManager([say("Bright, truly."), move("east"), move("east"), move("east"), move("east"),
                                   say("Again."), say("Again too.")])
        manager.generate = scored(manager.generate)
        with manager.open_session() as session:
            forked = fork_token_edit(ep, 7, 5, "B", session)
        self.assertEqual(forked.turns, ep.turns[:7])
        list(stream_episode(forked, manager))
        self.assertEqual(forked.phase, "arrived")
        self.assertTrue(forked.turns[7]["steered"])
        self.assertEqual(manager.calls[0][1]["steering"], VECTOR)
        self.assertEqual(manager.calls[0][0], context_messages(forked, 7))
        self.assertTrue(forked.mail[2]["text"].startswith("Exit B"))
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))

    def test_a_fork_of_the_taste_keeps_its_strength(self):
        ep, _ = run()
        manager = SteeringManager([say("ner."), *SCRIPT[2:]])
        manager.generate = scored(manager.generate)
        with manager.open_session() as session:
            forked = fork_token_edit(ep, 1, 0, "Z", session)
        list(stream_episode(forked, manager))
        self.assertEqual(manager.calls[0][1]["steering"]["strength"], 1.0)
        self.assertEqual(forked.agents[1]["messages"][2]["content"], "Zner.")
        self.assertEqual(saved(from_payload(saved(forked))), saved(forked))


# A longer corridor: exit A two moves west of the start, exit B four moves east.
LONG = Maze((".......",) + ("#######",) * 6, (0, 2), (0, 0))
LONG_ID = LONG.tool_id()
TEAM_REWARD = dict(communication=True, team_goal="all", exits=[[0, 6]], reward_exit="A", arrival_responses=1,
                   laps=1, steering=VECTOR, team_reward=2)


def long_move(direction):
    return call(direction, maze_id=LONG_ID)


# agent-1 reaches A in round 2 and writes in round 3 while agent-2 walks on to B.
LONG_SCRIPT = [long_move("west"), long_move("east"),
               long_move("west"), long_move("east"),
               say("A was lovely."), long_move("east"),
               long_move("east"),
               say("B was plain.")]


def run_long(**changes):
    ep = team_episode(LONG, TEAM_REWARD | changes)
    manager = SteeringManager(list(LONG_SCRIPT))
    manager.generate = scored(manager.generate)
    list(stream_episode(ep, manager))
    return ep, manager


class TeamRewardTests(unittest.TestCase):
    def test_an_arrival_at_the_reward_exit_steers_the_teammates_still_moving(self):
        ep, manager = run_long()
        self.assertEqual(ep.phase, "arrived", ep.detail)
        steered = [(t["round"], t["agent"], t["kind"]) for t in ep.turns if t["steered"]]
        # Its own response after arriving, then agent-2's next two moves. Its
        # arrival at B, unrewarded, is not.
        self.assertEqual(steered, [(2, 0, "arrival"), (2, 1, "move"), (3, 1, "move")])
        strengths = [kw["steering"] and kw["steering"]["strength"] for _, kw in manager.calls]
        self.assertEqual(strengths, [None] * 4 + [VECTOR["strength"]] * 3 + [None])
        self.assertEqual(ep.agents[1]["reward_left"], 0)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        forged = saved(ep)
        forged["turns"][5]["steered"] = False
        with self.assertRaisesRegex(ValueError, "steered flag"):
            from_payload(forged)

    def test_a_team_reward_of_one_steers_one_move(self):
        ep, _ = run_long(team_reward=1)
        self.assertEqual([(t["round"], t["agent"]) for t in ep.turns if t["steered"]], [(2, 0), (2, 1)])

    def test_without_a_team_reward_only_the_arriving_agent_is_steered(self):
        ep, _ = run_long(team_reward=0)
        self.assertEqual([(t["round"], t["agent"]) for t in ep.turns if t["steered"]], [(2, 0)])
        self.assertNotIn("reward_left", ep.agents[0])

    def test_a_team_reward_with_the_vector_off_steers_nothing(self):
        ep, manager = run_long(steering=dict(VECTOR, enabled=False))
        self.assertFalse(any(t["steered"] for t in ep.turns))
        self.assertIn("team reward of 2 responses, unsteered", reward_status(ep))

    def test_run_details_give_route_lengths_where_routes_part_and_norms(self):
        ep, _ = run_long(taste=True, taste_strength=-2.0)
        details = reward_status(ep)
        self.assertIn("Exits A (0, 0) at 2 moves, B (0, 6) at 4 moves", details)
        self.assertIn("routes part at (0, 2), 0 moves in", details)
        norm = (1 + 4 + .25) ** .5
        self.assertIn(f"after arriving there · norm {norm * 4:.3g}", details)
        self.assertIn(f"taste at strength -2 · norm {norm * 2:.3g}", details)
        self.assertIn("also steers the next 2 responses of every teammate still moving", details)

    def test_single_exit_reward_details_include_its_route_length(self):
        for settings in ({"arrival_responses": 1}, {"taste": True}, {"laps": 2}):
            with self.subTest(settings=settings):
                ep = team_episode(LONG, dict(team_goal="all", **settings))
                self.assertIn("Destination (0, 0) at 2 moves", reward_status(ep))

    def test_refused_team_rewards_and_paired_exits(self):
        refused = {"Name one": dict(reward_exit=None),
                   "team_reward must be an integer": dict(team_reward=9),
                   "same route length": dict(paired_exits=True),
                   "paired_exits is either true or false": dict(paired_exits="yes")}
        for message, change in refused.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                team_episode(LONG, TEAM_REWARD | change)
        self.assertTrue(team_episode(HALL, REWARD | dict(paired_exits=True)).config["paired_exits"])


class PairedExitTests(unittest.TestCase):
    def test_both_exits_are_drawn_at_the_route_length(self):
        for seed in range(12):
            maze, second = generate_paired(7, seed, 6, .7)
            distances = maze.distances(maze.start)
            self.assertEqual((distances[maze.goal], distances[second]), (6, 6))
            self.assertNotIn(second, (maze.goal, maze.start))
            self.assertEqual(generate_paired(7, seed, 6, .7), (maze, second))
            fork, steps = parting_cell(maze, maze.goal, second)
            self.assertTrue(0 <= steps < 6)
            self.assertEqual(distances[fork], steps)
        with self.assertRaisesRegex(ValueError, "two cells at that route length"):
            generate_paired(3, 1, 8, .5)

    def test_the_parting_cell_is_the_last_one_both_routes_share(self):
        fork = Maze((".....", "#.#.#", "#...#", "#.###", "#.###"), (4, 1), (0, 0))
        self.assertEqual(parting_cell(fork, (0, 0), (0, 4)), ((2, 1), 2))
        self.assertEqual(parting_cell(LONG, (0, 0), (0, 6)), ((0, 2), 0))


class RewardPageTests(unittest.TestCase):
    def test_paired_exits_cannot_silently_drop_a_required_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=TokenInspector(), models=Manager([]), data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                prepare = listeners_by_name(demo)["prepare_episode"]
                ep = team_episode(HALL, {})
                for settings in (dict(size=3, seed=0, distance=1, openness=.7),
                                 dict(size=5, seed=7, distance=4, openness=.7)):
                    with self.subTest(settings=settings), self.assertRaisesRegex(gr.Error, "required checkpoint"):
                        prepare.fn(ep, False, "s", None, *scenario(agents=2, team_goal="all", required=True,
                                                                    paired_exits=True, **settings))
                new = prepare.fn(ep, False, "s", None,
                                 *scenario(agents=2, team_goal="all", required=True, distance=4))[0]
                self.assertIsNotNone(new.config["required_checkpoint"])
            finally:
                demo.close()

    def test_prepare_export_reload_and_rebuild_a_reward_run(self):
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=TokenInspector(), models=Manager([]), data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = listeners_by_name(demo)
                prepare, load = callbacks["prepare_episode"], callbacks["load"]
                ep = team_episode(HALL, {})
                base = dict(agents=3, team_goal="all", supplied=0, interruption_text="", vector=VECTOR, strength=6.,
                            layer=2)
                plain = prepare.fn(ep, False, "s", None, *scenario(**base))[0]
                maze = plain.maze
                spare = next((r, c) for r in range(maze.size) for c in range(maze.size)
                             if maze.open((r, c)) and (r, c) not in (maze.start, maze.goal)
                             and len(maze.neighbors((r, c))) == 1)
                rewards = dict(exits=f"{spare[0]}, {spare[1]}", reward_exit="B", arrival_responses=2, taste=True,
                               taste_strength=2.5, laps=3, lap_prompt="Choose again.")
                prepared = prepare.fn(ep, False, "s", None, *scenario(**base, **rewards, steer="reward"))
                self.assertEqual(len(prepared), len(prepare.outputs))
                new = prepared[0]
                self.assertTrue(new.rewarded)
                self.assertEqual(new.maze, maze)
                self.assertEqual(new.config["exits"], [list(spare)])
                self.assertEqual((new.config["reward_exit"], new.config["laps"], new.config["taste_strength"]),
                                 ("B", 3, 2.5))
                self.assertEqual(new.config["steering"], dict(VECTOR, strength=6., layer=2))
                self.assertNotIn("steer_when", new.config)
                loaded = load.fn(str(new.export()), ep, False, "s", None)
                self.assertEqual(len(loaded), len(load.outputs))
                filled = {block._id: value.get("value") if isinstance(value, dict) and value.get("__type__") == "update"
                          else value for block, value in zip(load.outputs, loaded)}
                settings = [filled[block._id] for block in prepare.inputs[4:]]
                rebuilt = prepare.fn(ep, False, "s", None, *settings)[0]
                self.assertEqual(rebuilt.config, new.config)
                refused = {"taste and reward exit needs": scenario(**base, steer="reward"),
                           "no other steering trigger": scenario(**base, **rewards, steer="moves"),
                           "an interruption": scenario(**dict(base, interruption_text="Wait"), **rewards)}
                for message, values in refused.items():
                    with self.subTest(message), self.assertRaisesRegex(gr.Error, message):
                        prepare.fn(ep, False, "s", None, *values)
            finally:
                demo.close()


    def test_a_drawn_exit_b_and_a_team_reward_reload_into_the_same_run(self):
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=TokenInspector(), models=Manager([]), data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = listeners_by_name(demo)
                prepare, load = callbacks["prepare_episode"], callbacks["load"]
                ep = team_episode(HALL, {})
                settings = dict(agents=2, team_goal="all", distance=4, vector=VECTOR, strength=3., layer=1,
                                steer="reward", reward_exit="A", arrival_responses=1, paired_exits=True,
                                team_reward=2)
                new = prepare.fn(ep, False, "s", None, *scenario(**settings))[0]
                maze, second = generate_paired(5, 7, 4, .7)
                self.assertEqual(new.maze, maze)
                self.assertEqual(new.config["exits"], [list(second)])
                self.assertEqual((new.config["paired_exits"], new.config["team_reward"]), (True, 2))
                loaded = load.fn(str(new.export()), ep, False, "s", None)
                filled = {block._id: value.get("value") if isinstance(value, dict) and value.get("__type__") == "update"
                          else value for block, value in zip(load.outputs, loaded)}
                rebuilt = prepare.fn(ep, False, "s", None, *[filled[block._id] for block in prepare.inputs[4:]])[0]
                self.assertEqual(rebuilt.config, new.config)
                self.assertEqual(rebuilt.maze, new.maze)
                with self.assertRaisesRegex(gr.Error, "Leave Extra exits blank"):
                    prepare.fn(ep, False, "s", None, *scenario(**settings, exits="0, 1"))
            finally:
                demo.close()

if __name__ == "__main__":
    unittest.main()
