import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from chatlab.extension_api import TokenInspector
from chatlab.extensions.maze_experiments.maze import Maze, parse_call
from chatlab.extensions.maze_experiments.page import build_page
from chatlab.extensions.maze_experiments.team import (MAX_AGENTS, MESSAGE_LIMIT, TeamEpisode, format_agents,
                                                      from_payload, parse_agents, stream_team, team_tools)
from chatlab.extensions.maze_experiments.team_page import (agent_color, mail_text, response_view, team_board,
                                                           team_status, team_timeline)
from maze_support import call, Manager, MAZE, MAZE_ID, scored
from ui_support import listeners_by_name

# No walls, so a team placed on its start shares one cell.
OPEN = Maze(("...", "...", "..."), (1, 1), (0, 2))


def say(text):
    return text, list(text.encode()) + [0]


def team(maze=MAZE, **config):
    return TeamEpisode(maze, dict(dict(agents=2, communication=True, team_goal="any"), **config))


class TeamParseTests(unittest.TestCase):
    def test_a_message_is_admitted_only_where_the_run_allows_one(self):
        text = call("east", "hi")[0]
        self.assertEqual(parse_call(text), (None, "invalid_arguments"))
        self.assertEqual(parse_call(text, message_limit=MESSAGE_LIMIT)[0]["message"], "hi")
        self.assertEqual(parse_call(call("east")[0], message_limit=MESSAGE_LIMIT)[0],
                         {"maze_id": MAZE_ID, "direction": "east"})
        self.assertEqual(parse_call(call("east", "x" * (MESSAGE_LIMIT + 1))[0], message_limit=MESSAGE_LIMIT),
                         (None, "message_too_long"))
        self.assertEqual(parse_call(call("east", 7)[0], message_limit=MESSAGE_LIMIT), (None, "invalid_arguments"))

    def test_the_tool_declares_a_message_only_when_agents_can_talk(self):
        self.assertNotIn("message", team_tools(False)[0]["function"]["parameters"]["properties"])
        self.assertIn("message", team_tools(True)[0]["function"]["parameters"]["properties"])


class TeamEpisodeTests(unittest.TestCase):
    def test_each_agent_is_told_who_it_is_and_whether_it_can_talk(self):
        on, off = team(), team(communication=False, agents=3)
        first = on.agents[0]["messages"][1]["content"]
        self.assertIn("You are agent-1, one of 2 agents", first)
        self.assertIn("Your teammate is agent-2", first)
        self.assertIn("message argument", first)
        self.assertIn('"messages":[]', first)
        third = off.agents[2]["messages"][1]["content"]
        self.assertIn("Your teammates are agent-1 and agent-2", third)
        self.assertIn("You cannot communicate with your teammates.", third)
        self.assertNotIn('"messages"', third)
        # Agents never see where their teammates stand, whether or not they talk.
        self.assertNotIn("positions", first + third)

    def test_a_round_applies_every_move_together_and_delivers_messages_next_round(self):
        manager = Manager([call("east", "I am heading east"), call("south"), call("east"), call("north")])
        ep = team()
        list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.rounds, 2)
        self.assertEqual([a["status"] for a in ep.agents], ["arrived", "active"])
        # agent-2 was asked in round 1 against the start, not against the cell
        # agent-1 had just been told it would move to.
        round_one = manager.calls[1][0]
        self.assertEqual(json.loads(round_one[-1]["content"].rsplit("\n", 1)[1])["current"], [0, 0])
        # The message reached agent-2 in its reply for round 1, which it read in round 2.
        reply = json.loads(manager.calls[3][0][-1]["content"])
        self.assertEqual(reply["messages"], [{"from": "agent-1", "text": "I am heading east"}])
        self.assertEqual(reply["error"], "blocked_move")
        own = json.loads(manager.calls[2][0][-1]["content"])
        self.assertEqual(own["messages"], [])
        self.assertEqual(ep.mail, [dict(round=0, sender="agent-1", text="I am heading east", to=["agent-2"])])
        self.assertEqual(ep.tools[0]["function"]["parameters"]["properties"]["message"]["maxLength"], MESSAGE_LIMIT)

    def test_with_communication_off_a_message_is_a_rejected_call(self):
        manager = Manager([call("east", "hello"), call("east"), call("east"), call("east")])
        ep = team(communication=False)
        list(stream_team(ep, manager))
        self.assertEqual(ep.events[0]["error"], "invalid_arguments")
        self.assertEqual(ep.events[0]["after"], [0, 0])
        self.assertEqual(ep.mail, [])
        self.assertNotIn("messages", json.loads(manager.calls[2][0][-1]["content"]))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.detail, "agent-2 reached the destination in round 2.")

    def test_every_agent_goal_waits_for_the_last_and_stops_asking_those_that_arrived(self):
        manager = Manager([call("east"), call("south"), call("east"), call("north"),
                           call("east"), call("east")])
        ep = team(team_goal="all")
        list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.rounds, 4)
        self.assertEqual([turn["agent"] for turn in ep.turns], [0, 1, 0, 1, 1, 1])
        self.assertEqual(ep.detail, "Every agent reached the destination by round 4.")

    def test_an_agent_that_stops_calling_drops_out_while_its_teammate_continues(self):
        manager = Manager([say("I give up."), call("east"), call("east")])
        ep = team()
        list(stream_team(ep, manager))
        self.assertEqual(ep.agents[0]["status"], "abandoned")
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual([turn["agent"] for turn in ep.turns], [0, 1, 1])
        self.assertEqual(team_timeline(ep)[1][4], "No call · stopped")
        # The board says how each agent stood at the round it shows, not how the run ended.
        self.assertIn("agent-1 · active", team_board(ep, -1))
        self.assertIn("agent-1 · abandoned", team_board(ep, 0))
        self.assertIn("agent-2 · arrived", team_board(ep))

    def test_a_team_with_nobody_moving_has_abandoned(self):
        manager = Manager([say("No."), say("Nor I.")])
        ep = team()
        list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "abandoned")
        self.assertIn("agent-1 abandoned, agent-2 abandoned", ep.detail)

    def test_the_round_limit_ends_the_run(self):
        manager = Manager([call("west"), call("west")])
        ep = team(round_limit=1)
        list(stream_team(ep, manager))
        self.assertEqual((ep.phase, ep.detail), ("budget", "The team reached its round limit."))

    def test_each_agent_is_capped_by_what_it_has_left(self):
        chatty = call("south", "the south is walled off")
        manager = Manager([chatty, call("south"), call("west"), call("west")])
        ep = team(agent_token_budget=500, per_turn_tokens=1000, round_limit=2)
        list(stream_team(ep, manager))
        self.assertEqual([kwargs["max_new_tokens"] for _, kwargs in manager.calls],
                         [500, 500, 500 - len(chatty[1]), 500 - len(call("south")[1])])
        self.assertEqual(ep.agent_tokens(), [len(chatty[1]) + len(call("west")[1]), len(call("south")[1] + call("west")[1])])
        self.assertIn("at most 500 per agent", team_status(ep))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        self.assertEqual(json.loads(json.dumps(replay.payload())), json.loads(json.dumps(ep.payload())))
        # The chatty agent's second response only fit what it had left before
        # the limit was lowered.
        tighter = json.loads(json.dumps(ep.payload()))
        tighter["config"]["agent_token_budget"] = len(chatty[1]) + 10
        with self.assertRaisesRegex(ValueError, "more tokens than its round allowed"):
            from_payload(tighter)

    def test_an_agent_that_spends_its_limit_stops_while_its_teammate_continues(self):
        spent = len(call("south")[1])
        # west is a letter shorter than south, which leaves agent-2 one token.
        manager = Manager([call("south"), call("west"), call("west")])
        ep = team(agent_token_budget=spent, per_turn_tokens=1000)
        list(stream_team(ep, manager))
        self.assertEqual([kwargs["max_new_tokens"] for _, kwargs in manager.calls], [spent, spent, 1])
        self.assertEqual([turn["agent"] for turn in ep.turns], [0, 1, 1])
        self.assertIn("agent-1 · out of tokens", team_board(ep, 0))
        self.assertIn("agent-2 · active", team_board(ep, 0))
        self.assertEqual(ep.phase, "budget")
        self.assertEqual(ep.detail, "No agent is still moving: agent-1 out of tokens, agent-2 out of tokens.")

    def test_a_team_out_of_tokens_beside_one_that_gave_up_has_abandoned(self):
        spent = len(call("south")[1])
        ep = team(agent_token_budget=spent)
        list(stream_team(ep, Manager([call("south"), say("I give up.")])))
        self.assertEqual(ep.phase, "abandoned")

    def test_a_response_cut_off_at_the_last_of_its_limit_leaves_its_agent_out_of_tokens(self):
        # Without its stop token each reply fills the cap its agent's limit set.
        text, ids = call("south")
        cut = (text, ids[:-1])
        ep = team(agent_token_budget=len(ids) - 1, per_turn_tokens=1000)
        list(stream_team(ep, Manager([cut, cut])))
        self.assertEqual([turn["finish_reason"] for turn in ep.turns], ["length", "length"])
        self.assertEqual([agent["status"] for agent in ep.agents], ["out_of_tokens", "out_of_tokens"])
        self.assertEqual(ep.phase, "budget")
        self.assertIn("agent-1 · out of tokens", team_board(ep, 0))
        self.assertEqual([row[4] for row in team_timeline(ep)[1:]], ["Out of tokens · stopped"] * 2)
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        self.assertEqual(json.loads(json.dumps(replay.payload())), json.loads(json.dumps(ep.payload())))
        # A cut-off with tokens still to spend is only cut off.
        ep = team(agent_token_budget=1000, per_turn_tokens=len(ids) - 1)
        list(stream_team(ep, Manager([cut, cut])))
        self.assertEqual((ep.phase, ep.agents[0]["status"]), ("abandoned", "cut_off"))
        self.assertEqual(team_timeline(ep)[1][4], "Cut off · stopped")

    def test_a_run_saved_under_one_limit_for_the_team_still_splits_it_evenly(self):
        manager = Manager([call("south"), call("south")])
        ep = team(token_budget=30, per_turn_tokens=100)
        list(stream_team(ep, manager))
        # Each reply is far longer than 15 tokens, so asking in turn would have
        # left the second agent nothing.
        self.assertEqual([kwargs["max_new_tokens"] for _, kwargs in manager.calls], [15, 15])

    def test_a_team_budget_too_small_for_every_agent_runs_no_round(self):
        manager = Manager([])
        ep = team(token_budget=1)
        list(stream_team(ep, manager))
        self.assertEqual(manager.calls, [])
        self.assertEqual((ep.phase, ep.rounds), ("budget", 0))
        self.assertEqual(from_payload(json.loads(json.dumps(ep.payload()))).phase, "budget")

    def test_a_run_saved_under_one_limit_for_the_team_replays_under_it(self):
        ep = team(token_budget=1000)
        list(stream_team(ep, Manager([call("east"), call("south"), call("east"), call("north")])))
        self.assertNotIn("agent_token_budget", ep.config)
        self.assertIn("of 1,000 sampled tokens", team_status(ep))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        self.assertEqual(json.loads(json.dumps(replay.payload())), json.loads(json.dumps(ep.payload())))
        with self.assertRaisesRegex(ValueError, "not both"):
            team(token_budget=1000, agent_token_budget=500)

    def test_a_failure_mid_round_marks_every_answered_response_not_applied(self):
        # The fixture runs out of replies on agent-2, which fails its response.
        manager = Manager([call("east")])
        ep = team()
        list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "error")
        self.assertEqual(ep.events, [])
        self.assertEqual([turn["outcome"] for turn in ep.turns], ["not_applied", "not_applied"])
        self.assertEqual([row[4] for row in team_timeline(ep)[1:]], ["Not applied", "Not applied"])

    def test_a_discarded_round_takes_nobody_out_of_the_team(self):
        # agent-1 makes no call, then agent-2's response fails, so the round
        # never resolves and agent-1 never leaves.
        manager = Manager([say("I give up.")])
        ep = team()
        list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "error")
        self.assertEqual([turn["outcome"] for turn in ep.turns], ["not_applied", "not_applied"])
        self.assertEqual([agent["status"] for agent in ep.agents], ["active", "active"])
        self.assertEqual(ep.turns[0]["text"], "I give up.")
        self.assertIn("agent-1 · active", team_board(ep))

    def test_runs_ended_mid_round_read_back_as_they_were_saved(self):
        stopped = team()
        for frame in stream_team(stopped, Manager([call("east"), call("east")])):
            if len(frame.turns) == 2 and frame.turns[0].get("finish_reason") == "stop":
                stopped.request_stop()
        failed = team()
        list(stream_team(failed, Manager([say("I give up.")])))
        paused = team()
        list(stream_team(paused, Manager([call("south"), call("south")]), single_step=True))
        for ep in (stopped, failed, paused):
            with self.subTest(ep.phase):
                replay = from_payload(json.loads(json.dumps(ep.payload())))
                self.assertEqual(replay.phase, ep.phase)
                self.assertEqual(json.loads(json.dumps(replay.payload())), json.loads(json.dumps(ep.payload())))

    def test_every_agent_samples_under_its_own_seed(self):
        manager = Manager([call("south"), call("south"), call("east"), call("east")])
        list(stream_team(team(sampling_seed=5, round_limit=2), manager))
        self.assertEqual(len({kwargs["seed"] for _, kwargs in manager.calls}), 4)

    def test_next_runs_one_round(self):
        manager = Manager([call("east")] * 4)
        ep = team()
        list(stream_team(ep, manager, single_step=True))
        self.assertEqual((ep.phase, ep.rounds, len(ep.turns)), ("paused", 1, 2))
        list(stream_team(ep, manager, single_step=True))
        self.assertEqual((ep.phase, ep.rounds), ("arrived", 2))

    def test_stop_during_a_round_applies_none_of_it(self):
        manager = Manager([call("east"), call("east")])
        ep = team()
        stream = stream_team(ep, manager)
        for frame in stream:
            if len(frame.turns) == 2 and frame.turns[0].get("finish_reason") == "stop":
                ep.request_stop()
        self.assertEqual(ep.phase, "stopped")
        self.assertEqual(ep.rounds, 0)
        self.assertEqual(ep.events, [])
        self.assertEqual([a["position"] for a in ep.agents], [MAZE.start, MAZE.start])
        self.assertEqual(ep.turns[0]["outcome"], "not_applied")
        self.assertEqual(len(ep.agents[0]["messages"]), 2)

    def test_a_saved_run_replays_and_a_tampered_one_is_refused(self):
        manager = Manager([call("east", "east is open"), call("south"), call("east"), call("north")])
        ep = team()
        with tempfile.TemporaryDirectory() as directory:
            list(stream_team(ep, manager, save_dir=Path(directory)))
            saved = json.loads((Path(directory) / f"{ep.run_id}.json").read_text())
        self.assertEqual(saved["format"], "chatlab-maze-team-1")
        replay = from_payload(saved)
        self.assertTrue(replay.replay_only)
        self.assertEqual([a["position"] for a in replay.agents], [(0, 2), (0, 0)])
        self.assertIn("east is open", mail_text(replay))
        with self.assertRaisesRegex(ValueError, "Start a new team episode"):
            list(stream_team(replay, manager))
        def altered(change):
            copy = json.loads(json.dumps(saved))
            change(copy)
            return copy

        def forged(change):
            # A forgery rewrites a move in both places the run records it.
            def both(copy):
                for event in (copy["events"][0], copy["turns"][copy["events"][0]["turn"]]["event"]):
                    change(event)
            return altered(both)

        refused = {
            "a moved final position": (altered(lambda c: c["agents"][1].update(position=[0, 1])), "agents do not match"),
            "a teleport": (forged(lambda e: e.update(after=[2, 2])), "responses do not match"),
            "a forged arrival": (forged(lambda e: e.update(arrived=True)), "responses do not match"),
            "forged progress": (forged(lambda e: e.update(progress=not e["progress"]))
                                , "responses do not match"),
            # The response still says east, so another legal direction is a
            # move that response never asked for.
            "a direction the response never asked for": (
                forged(lambda e: e.update(direction="south", after=[0, 0], accepted=False)), "responses do not match"),
            "an error the simulator never gave": (
                forged(lambda e: e.update(accepted=False, error="already_arrived", after=[0, 0])),
                "responses do not match"),
            "a move detached from its response": (
                altered(lambda c: c["events"][0].update(after=[0, 0])), "moves do not match"),
            "a message nobody sent": (forged(lambda e: e.update(message="a different plan")), "responses do not match"),
            "missing mail": (altered(lambda c: c.update(mail=[])), "messages do not match"),
            "a claimed status": (altered(lambda c: c["agents"][1].update(status="arrived")), "agents do not match"),
            "a forged outcome": (altered(lambda c: c.update(phase="budget")), "outcome other than"),
            "a response longer than its cap": (
                altered(lambda c: c["config"].update(per_turn_tokens=10)), "more tokens than its round allowed"),
            "a moved starting position": (
                altered(lambda c: c["turns"][1].update(position_before=[2, 2])), "starting position"),
            "a stop with no tokens": (altered(lambda c: [t.update(metrics=[]) for t in c["turns"]]),
                                      "finish reason its tokens"),
            "a cut-off short of its cap": (altered(lambda c: c["turns"][0].update(finish_reason="length")),
                                           "finish reason its tokens"),
            "another model": (altered(lambda c: c.update(model_id="other/model")), "names a model other"),
            "another load": (altered(lambda c: c.update(load_id="test/model#9")), "names a model other"),
            "a zeroed token count": (altered(lambda c: c.update(sampled_tokens=0)), "sampled-token count does not"),
            "a zeroed call count": (altered(lambda c: c.update(tool_attempts=0)), "call count does not"),
            "more rounds than the limit": (altered(lambda c: c.update(rounds=99)), "within its round limit"),
            # Walking south instead leaves agent-1 where its later responses
            # say it was not, and never reaches the arrival the run reports.
            "an edited response": (altered(lambda c: c["turns"][0].update(text=call("south", "east is open")[0])),
                                   "starting position|outcome other than"),
        }
        for name, (payload, message) in refused.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                from_payload(payload)
        prompt = json.loads(json.dumps(saved))
        prompt["agents"][0]["messages"][1]["content"] = "Something else."
        with self.assertRaisesRegex(ValueError, "agents do not match"):
            from_payload(prompt)

    def test_the_board_draws_every_agent_and_fans_out_a_shared_cell(self):
        ep = team(maze=OPEN, agents=3)
        drawn = team_board(ep)
        self.assertEqual(drawn.count('font-weight="700">1<'), 1)
        self.assertEqual(drawn.count('r="13"'), 3)
        self.assertIn("agent-3 · active", drawn)

    def test_messages_are_shown_as_text_rather_than_markdown(self):
        ep = team()
        ep.mail = [dict(round=0, sender="agent-1", text="**loud** [link](x) it's", to=["agent-2"])]
        shown = mail_text(ep)
        self.assertIn("it's", shown)
        self.assertNotIn("**loud**", shown)
        self.assertIn("\\*\\*loud\\*\\*", shown)

    def test_config_refuses_a_team_of_one_or_more_than_the_cap(self):
        for count in (1, MAX_AGENTS + 1):
            with self.subTest(count), self.assertRaisesRegex(ValueError, "2 to 100 agents"):
                team(agents=count)

    def test_a_hundred_agents_play_a_round_and_fit_on_the_board(self):
        ep = team(maze=OPEN, agents=MAX_AGENTS)
        self.assertIn("Your teammates are agent-2, agent-3", ep.agents[0]["messages"][1]["content"])
        self.assertIn("agent-99 and agent-100", ep.agents[0]["messages"][1]["content"])
        drawn = team_board(ep)
        # Every agent starts on one cell, drawn as one marker with their count.
        self.assertEqual(drawn.count("×100<"), 1)
        self.assertNotIn('font-weight="700">1<', drawn)
        self.assertIn("agent-1, agent-2", drawn)
        self.assertIn("agent-100 · active", drawn)
        manager = Manager([call("east", maze_id=OPEN.tool_id())] * MAX_AGENTS)
        manager.generate = scored(manager.generate)
        list(stream_team(ep, manager, single_step=True))
        self.assertEqual((ep.rounds, ep.moves), (1, MAX_AGENTS))
        self.assertEqual(len({kwargs["seed"] for _, kwargs in manager.calls}), MAX_AGENTS)
        # Each path is drawn off the row's centre line, never out of its cell.
        centre = 28 + 1.5 * 56
        rows = [float(y) for y in re.findall(r'<line x1="[^"]+" y1="([^"]+)"', team_board(ep))]
        self.assertEqual(len(rows), MAX_AGENTS)
        self.assertTrue(all(abs(y - centre) <= 20 for y in rows))
        ep.selected_turn = MAX_AGENTS - 1
        self.assertIn(f"{len(ep.turns[-1]['prompt_ids']):,} prompt tokens", response_view(ep)[0])

    def test_a_small_group_on_one_cell_is_still_fanned_out(self):
        ep = team(maze=OPEN, agents=5)
        ep.events.append(dict(accepted=True, round=0, agent=4, before=[1, 1], after=[1, 2]))
        drawn = team_board(ep, 0)
        self.assertEqual(drawn.count('r="13"'), 4)
        self.assertEqual(drawn.count('r="16"'), 1)
        self.assertNotIn("×", drawn)

    def test_every_agent_has_its_own_colour_and_none_is_the_destinations_green(self):
        colors = [agent_color(k) for k in range(MAX_AGENTS)]
        self.assertEqual(len(set(colors)), MAX_AGENTS)
        hues = [float(c[4:].split()[0]) for c in colors if c.startswith("hsl")]
        self.assertTrue(all(not 95 <= hue <= 175 for hue in hues))

    def test_agents_to_steer_are_read_as_numbers_and_ranges(self):
        self.assertEqual(parse_agents("1, 3, 5-8", 10), [0, 2, 4, 5, 6, 7])
        self.assertEqual(format_agents([0, 2, 4, 5, 6, 7], 10), "1, 3, 5-8")
        self.assertEqual(parse_agents(format_agents([9, 0], 10), 10), [0, 9])
        for every in ("", " all ", "ALL", "1-10", "1-5, 6-10"):
            with self.subTest(every):
                self.assertIsNone(parse_agents(every, 10))
        self.assertEqual(format_agents(None, 10), "all")
        for bad in ("0", "11", "3-1", "x", "1,,2", "1 2"):
            with self.subTest(bad), self.assertRaises(ValueError):
                parse_agents(bad, 10)


class TeamPageTests(unittest.TestCase):
    def test_the_team_tab_prepares_runs_and_replays(self):
        manager = Manager([])
        manager.generate = scored(manager.generate)
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=TokenInspector(), models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = listeners_by_name(demo)
                prepare = callbacks["team_prepare_episode"]
                values = (2, True, "any", 3, 1, 2, .9, "coordinates", "", "Be brief.", "Reach the star.",
                          .7, 1, 200, 1000, 10)
                prepared = prepare.fn(team(), False, *values)
                self.assertEqual(len(prepared), len(prepare.outputs))
                ep = prepared[0]
                self.assertEqual(ep.config["agents"], 2)
                self.assertIn("Reach the star.", ep.agents[1]["messages"][1]["content"])
                maze_id = ep.maze.tool_id()
                manager.replies = iter([call("north", "go", maze_id), call("north", maze_id=maze_id)])
                play = callbacks["team_step_forward"]
                frames = list(play.fn(ep, False))
                self.assertTrue(all(len(frame) == len(play.outputs) for frame in frames))
                self.assertEqual(ep.rounds, 1)
                self.assertTrue((Path(directory) / f"{ep.run_id}.json").exists())
                load = callbacks["team_load"]
                loaded = load.fn(str(ep.export()), team(), False)
                self.assertEqual(len(loaded), len(load.outputs))
                # The per-agent limit field comes back as it was set.
                self.assertEqual(loaded[25], 1000)
                legacy = team(token_budget=3000)
                self.assertEqual(load.fn(str(legacy.export()), team(), False)[25], 1500)
                self.assertTrue(loaded[0].replay_only)
                select = callbacks["team_select_history"]
                shown = select.fn(loaded[0], False, SimpleNamespace(index=[1, 0]))
                self.assertIn("agent-1", shown[4])
            finally:
                demo.close()


if __name__ == "__main__":
    unittest.main()
