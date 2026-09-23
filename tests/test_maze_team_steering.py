"""Team activation interventions, mandatory passage, and replay provenance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from extension_api import SteeringError, TokenInspector
from extensions.maze_experiments.maze import Maze, generate, unavoidable_cells
from extensions.maze_experiments.page import build_page
from extensions.maze_experiments.team import TeamEpisode, from_payload, stream_team
from extensions.maze_experiments.team_page import response_view, team_board, team_status, team_timeline
from test_maze import Manager, scored
from test_maze_steering import SteeringManager, VECTOR
from test_maze_team import call

CORRIDOR = Maze((".....", "#####", "#####", "#####", "#####"), (0, 0), (0, 4))
ROOM = Maze(("...", "...", "..."), (0, 0), (0, 2))
CONFIG = dict(team_goal="all", steering=VECTOR, steer_when={"cell": [0, 1]},
              required_checkpoint=[0, 1], steer_responses=2)
# Agent 2 reaches the checkpoint one round later. Agent 1 returns to it after
# its two steered responses have expired; this must not trigger another dose.
WALK = ["east", "west", "east", "east", "west", "east", "west", "east",
        "east", "east", "east", "east", "east"]


def replies(walk):
    return [call(direction, maze_id=CORRIDOR.tool_id()) for direction in walk]


def saved(ep):
    return json.loads(json.dumps(ep.payload()))


def run(walk=WALK, **changes):
    ep = TeamEpisode(CORRIDOR, CONFIG | changes)
    manager = SteeringManager(replies(walk))
    manager.generate = scored(manager.generate)
    list(stream_team(ep, manager))
    return ep, manager


class TeamSteeringTests(unittest.TestCase):
    def test_each_agent_starts_and_expires_independently_without_retriggering(self):
        ep, manager = run()
        expected = [False, False, True, False, True, True, False, True, False, False, False, False, False]
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual([t["steered"] for t in ep.turns], expected)
        self.assertEqual([kw["steering"] for _, kw in manager.calls], [VECTOR if flag else None for flag in expected])
        self.assertEqual(manager.checked, [VECTOR])
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))
        self.assertIn("agent-1", team_status(ep))
        self.assertIn("first in round 2", team_status(ep))
        self.assertIn("first in round 3", team_status(ep))
        self.assertIn("steered", team_timeline(ep)[3][4])
        ep.selected_turn = 2
        self.assertIn("Steered", response_view(ep)[0])
        self.assertIn("Required checkpoint", team_board(ep))
        self.assertIn("Steering cell", team_board(ep))
        # No intervention cue is added to either agent's actual prompt.
        self.assertNotIn("checkpoint", ep.agents[0]["messages"][1]["content"])
        self.assertNotIn("steering", ep.agents[0]["messages"][1]["content"])

    def test_only_the_selected_agent_is_steered_and_zero_means_until_the_end(self):
        ep, manager = run(steer_agents=[1], steer_responses=0)
        flags = [t["steered"] for t in ep.turns]
        self.assertEqual(flags, [t["agent"] == 1 and t["round"] >= 2 for t in ep.turns])
        self.assertEqual([kw["steering"] is not None for _, kw in manager.calls], flags)
        self.assertEqual(saved(from_payload(saved(ep))), saved(ep))

    def test_move_trigger_counts_each_agents_accepted_moves(self):
        ep, _ = run(steer_when={"moves": 2}, steer_responses=1)
        self.assertEqual([(t["agent"], t["round"]) for t in ep.turns if t["steered"]], [(0, 2), (1, 3)])
        from_payload(saved(ep))

    def test_pause_resume_preserves_each_agents_remaining_duration(self):
        ep = TeamEpisode(CORRIDOR, CONFIG)
        manager = SteeringManager(replies(WALK))
        list(stream_team(ep, manager, single_step=True))
        list(stream_team(ep, manager, single_step=True))
        self.assertEqual([t["steered"] for t in ep.turns], [False, False, True, False])
        list(stream_team(ep, manager))
        expected, _ = run()
        self.assertEqual([t["steered"] for t in ep.turns], [t["steered"] for t in expected.turns])
        from_payload(saved(ep))

    def test_unreached_trigger_disabled_and_zero_strength_do_not_steer(self):
        cases = [dict(steer_when={"moves": 200}), dict(steering=dict(VECTOR, enabled=False)),
                 dict(steering=dict(VECTOR, strength=0))]
        for changes in cases:
            with self.subTest(changes):
                ep, manager = run(**changes)
                self.assertFalse(any(t["steered"] for t in ep.turns))
                self.assertTrue(all(kw["steering"] is None for _, kw in manager.calls))
                if "steering" in changes:
                    self.assertEqual(manager.checked, [])
                from_payload(saved(ep))

    def test_incompatible_vector_refused_before_any_response_and_session_released(self):
        ep = TeamEpisode(CORRIDOR, CONFIG)
        manager = SteeringManager([], refuse="wrong model")
        with self.assertRaisesRegex(SteeringError, "wrong model"):
            list(stream_team(ep, manager))
        self.assertEqual(ep.phase, "ready")
        self.assertEqual(ep.turns, [])
        self.assertFalse(manager.busy)
        self.assertFalse(ep.busy)

    def test_stop_before_generation_does_not_mark_the_opening_frame_steered(self):
        ep = TeamEpisode(CORRIDOR, CONFIG | dict(steer_when={"moves": 0}))
        manager = SteeringManager([])
        for frame in stream_team(ep, manager):
            if frame.turns:
                ep.request_stop()
        self.assertEqual(manager.calls, [])
        self.assertFalse(ep.turns[0]["steered"])
        from_payload(saved(ep))

    def test_interrupted_steered_round_keeps_provenance_without_applying_moves(self):
        for ending in ("stop", "failure", "closed"):
            with self.subTest(ending):
                ep = TeamEpisode(CORRIDOR, CONFIG | dict(steer_when={"moves": 0}))
                manager = SteeringManager(replies(["east"]))
                stream = stream_team(ep, manager)
                for frame in stream:
                    if len(frame.turns) == 2:
                        if ending == "stop":
                            ep.request_stop()
                        elif ending == "closed":
                            stream.close()
                            break
                self.assertEqual(ep.events, [])
                self.assertTrue(ep.turns[0]["steered"])
                self.assertEqual(ep.turns[0]["outcome"], "not_applied")
                self.assertFalse(manager.busy)
                from_payload(saved(ep))

    def test_forged_steering_flags_targets_and_duration_are_refused(self):
        ep, _ = run()
        payload = saved(ep)
        changes = [lambda p: p["turns"][0].update(steered=True),
                   lambda p: p["turns"][2].update(steered=False),
                   lambda p: p["turns"][2].pop("steered"),
                   lambda p: p["turns"][2].update(steered=1),
                   lambda p: p["config"].update(steer_agents=[1]),
                   lambda p: p["config"].update(steer_responses=1),
                   lambda p: p["config"].update(steer_when={"cell": [0, 2]})]
        for change in changes:
            forged = copy.deepcopy(payload)
            change(forged)
            with self.assertRaisesRegex(ValueError, "steered flag"):
                from_payload(forged)

    def test_unsteered_baseline_preserves_checkpoint_and_has_no_vector_marks(self):
        ep = TeamEpisode(CORRIDOR, dict(team_goal="all", required_checkpoint=[0, 1]))
        list(stream_team(ep, Manager(replies(["east"] * 8))))
        self.assertEqual(ep.phase, "arrived")
        self.assertTrue(all("steered" not in t for t in ep.turns))
        from_payload(saved(ep))
        forged = saved(ep)
        forged["turns"][0]["steered"] = True
        with self.assertRaisesRegex(ValueError, "without a steering vector"):
            from_payload(forged)

    def test_invalid_targets_vectors_triggers_and_bypassable_cells_are_refused(self):
        cases = [dict(steer_agents=[2]), dict(steer_agents=[]), dict(steer_agents=[0, 0]),
                 dict(steer_agents=[True]), dict(steer_agents="all"), dict(steer_responses=-1),
                 dict(steering=dict(VECTOR, vector=[])), dict(steer_when={"cell": None}),
                 dict(steer_when={"cell": [0, 4]}), dict(steer_when={"moves": -1}),
                 dict(required_checkpoint=[0, 0]), dict(required_checkpoint=[0, 4])]
        for changes in cases:
            with self.subTest(changes), self.assertRaises(ValueError):
                TeamEpisode(CORRIDOR, CONFIG | changes)
        with self.assertRaisesRegex(ValueError, "every route"):
            TeamEpisode(ROOM, CONFIG)


class RequiredCheckpointTests(unittest.TestCase):
    def test_only_cells_that_disconnect_start_from_goal_are_candidates(self):
        self.assertEqual(unavoidable_cells(CORRIDOR), [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(unavoidable_cells(ROOM), [])

    def test_generation_is_deterministic_and_retains_requested_distance(self):
        for seed in (1, 7, 20260911):
            maze = generate(5, seed, 8, .7, require_checkpoint=True)
            self.assertEqual(maze, generate(5, seed, 8, .7, require_checkpoint=True))
            self.assertEqual(len(maze.route()) - 1, 8)
            cells = unavoidable_cells(maze)
            self.assertTrue(cells)
            self.assertNotIn(maze.start, cells)
            self.assertNotIn(maze.goal, cells)
            TeamEpisode(maze, dict(required_checkpoint=list(cells[0])))
        with self.assertRaisesRegex(ValueError, "at least two"):
            generate(3, 1, 1, require_checkpoint=True)


class TeamSteeringPageTests(unittest.TestCase):
    def test_import_prepare_export_reload_and_matched_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=TokenInspector(), models=Manager([]), data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}
                vector_path = Path(directory) / "vector.json"
                vector_path.write_text(json.dumps(VECTOR))
                imported = callbacks["team_import_vector"].fn(str(vector_path))
                self.assertEqual(imported[0], VECTOR)
                prepare = callbacks["team_prepare_episode"]
                values = (2, True, "all", 5, 7, 8, .7, "coordinates", "", "Be brief.", "Deliver the solution.",
                          .7, 1, 200, 4000, 24)
                steering = [True, VECTOR, 6., 2, "cell", "", 3, 2, "0"]
                ep = TeamEpisode(CORRIDOR, {})
                prepared = prepare.fn(ep, False, *values, *steering)
                self.assertEqual(len(prepared), len(prepare.outputs))
                new = prepared[0]
                self.assertEqual(new.config["steer_agents"], [0])
                self.assertEqual(new.config["steer_when"]["cell"], new.config["required_checkpoint"])
                self.assertEqual(new.config["steering"], dict(VECTOR, strength=6., layer=2))
                baseline = steering.copy()
                baseline[4] = "off"
                plain = prepare.fn(ep, False, *values, *baseline)[0]
                self.assertEqual(new.maze, plain.maze)
                self.assertEqual(new.config["required_checkpoint"], plain.config["required_checkpoint"])
                self.assertNotIn("steering", plain.config)
                load = callbacks["team_load"]
                loaded = load.fn(str(new.export()), ep, False)
                self.assertEqual(len(loaded), len(load.outputs))
                # Fill controls exactly as the upload does, then create the same condition again.
                settings = [v.get("value") if isinstance(v, dict) and v.get("__type__") == "update" else v
                            for v in loaded[11:-1]]
                rebuilt = prepare.fn(ep, False, *settings)[0]
                self.assertEqual(rebuilt.config, new.config)
                self.assertEqual(rebuilt.maze, new.maze)
                for bad in ([True, None, 1., 0, "cell", "", 3, 1, "all"],
                            [False, VECTOR, 1., 0, "cell", "", 3, 1, "all"],
                            [True, VECTOR, 1., 0, "cell", "", 3, 1, "3"]):
                    with self.assertRaises(gr.Error):
                        prepare.fn(ep, False, *values, *bad)
            finally:
                demo.close()


if __name__ == "__main__":
    unittest.main()
