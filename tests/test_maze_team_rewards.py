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

from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.reasoning_check import read_responses
from chatlab.extensions.maze_experiments.runner import context_messages, fork_token_edit, from_payload, stream_episode
from chatlab.extensions.maze_experiments.team import REWARD_FORMAT
from chatlab.extensions.maze_experiments.team_views import (mail_text, positions_after, response_line, statuses_after,
                                                            team_board, team_history_rows, team_status)
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

    def test_cut_off_response_after_arriving_sends_nothing_and_keeps_the_agent(self):
        script = list(SCRIPT)
        script[7] = ("x" * 100, list(b"x" * 100))
        ep, _ = run(script)
        self.assertEqual(ep.phase, "arrived")
        self.assertNotIn("x" * 10, json.dumps(ep.mail))
        self.assertEqual(ep.agents[1]["status"], "arrived")
        from_payload(saved(ep))

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
        plain = forged(lambda d: [d["config"].pop(key) for key in list(d["config"])
                                  if key in ("exits", "reward_exit", "arrival_responses", "arrival_prompt", "taste",
                                             "taste_prompt", "taste_strength", "laps", "lap_prompt")])
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


class RewardPageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
