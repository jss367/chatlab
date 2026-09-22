import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from extensions.maze_experiments.dynamic_maze import changing
from extensions.maze_experiments.maze import Maze, call_text, generate
from extensions.maze_experiments.page import board, build_page, cell_text, parse_cell, status, timeline
from extensions.maze_experiments.runner import Episode, fork_token_edit, from_payload, stream_episode
from extensions.maze_experiments.trials import FORMAT as TRIALS_FORMAT, prepare_trial, read_trials
from extension_api import ModelService, SteeringError, TokenInspector

from test_maze import CONFIG, NO_CHECKPOINT, Manager

# An open room: from the start in the corner the character can walk down,
# across and back up to the destination at the end of the top row.
ROOM = Maze(("....", "....", "....", "...."), (0, 0), (0, 3))
# A top-right pocket reached only along the top row, so sealing (0, 2) cuts
# the pocket off while the start keeps its own way to the destination.
POCKET = Maze(("....", ".###", "....", "...."), (2, 0), (3, 3))
VECTOR = {"format": "chatlab-steering-1", "model_id": "test/model", "layer": 3,
          "vector": [1.0, -2.0, 0.5], "strength": 4.0, "enabled": True}
BASE = CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 4000, "attempt_budget": 20}
# Down, across, the waypoint at (1, 1), then across and up to the destination.
WALK = ["south", "east", "east", "north", "east"]


def reply(maze, direction):
    text = call_text(maze.tool_id(), direction)
    return text, list(text.encode()) + [0]


class SteeringManager(Manager):
    """The maze fixture manager, answering the steering pre-flight."""

    def __init__(self, replies, refuse=None):
        super().__init__(replies)
        self.refuse = refuse
        self.checked = []

    def check_steering(self, value):
        self.checked.append(copy.deepcopy(value))
        if self.refuse:
            raise SteeringError(self.refuse)


def checkpoint(**changes):
    return BASE | dict(waypoint=[1, 1], steering=VECTOR, steer_when={"cell": [1, 1]}, steer_responses=2) | changes


def run(config, walk=WALK, maze=ROOM):
    manager = SteeringManager([reply(maze, d) for d in walk])
    episode = Episode(maze, config)
    list(stream_episode(episode, manager))
    return episode, manager


def steering_sent(manager):
    return [kwargs["steering"] for _, kwargs in manager.calls]


class WaypointTests(unittest.TestCase):
    def test_the_state_shows_the_waypoint_and_when_it_is_reached(self):
        episode, _ = run(BASE | {"waypoint": [1, 1]})
        initial = json.loads(episode.messages[1]["content"].split("\n", 1)[1])
        self.assertEqual((initial["waypoint"], initial["waypoint_reached"]), ([1, 1], False))
        replies = [json.loads(m["content"]) for m in episode.messages if m["role"] == "tool"]
        self.assertEqual([r["waypoint_reached"] for r in replies], [False, True, True, True, True])
        self.assertEqual((episode.phase, episode.waypoint_turn), ("arrived", 1))
        self.assertIn("**Waypoint** (1, 1): reached in response 2", status(episode))
        self.assertIn("⚑", board(episode))

    def test_a_run_that_arrives_without_the_waypoint_missed_it(self):
        episode, _ = run(BASE | {"waypoint": [3, 3]}, walk=["east", "east", "east"])
        self.assertEqual((episode.phase, episode.waypoint_turn), ("arrived", None))
        self.assertIn("(3, 3): missed", status(episode))

    def test_a_supplied_move_can_reach_the_waypoint(self):
        episode = Episode(ROOM, BASE | {"waypoint": [0, 1], "supplied_moves": 1})
        self.assertEqual(episode.waypoint_turn, -1)
        self.assertTrue(json.loads(episode.messages[-1]["content"])["waypoint_reached"])

    def test_no_waypoint_leaves_the_state_as_it_was(self):
        episode, _ = run(BASE, walk=["east", "east", "east"])
        self.assertNotIn("waypoint", episode.messages[1]["content"])
        self.assertNotIn("waypoint", episode.config)
        self.assertNotIn("Waypoint", status(episode))

    def test_bad_waypoints_are_refused(self):
        for cell, message in (([0, 3], "destination"), ([0, 0], "start"), ([9, 9], "open cell"),
                              ([1], "row and a column"), ("1,1", "row and a column")):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, message):
                Episode(ROOM, BASE | {"waypoint": cell})
        with self.assertRaisesRegex(ValueError, "open cell"):
            Episode(POCKET, BASE | {"waypoint": [1, 1]})


class SteeringTests(unittest.TestCase):
    def test_steering_starts_at_the_cell_and_covers_the_responses_asked_for(self):
        episode, manager = run(checkpoint())
        self.assertEqual(episode.phase, "arrived")
        self.assertEqual(manager.checked, [episode.config["steering"]])
        self.assertEqual(steering_sent(manager), [None, None, VECTOR, VECTOR, None])
        self.assertEqual([t["steered"] for t in episode.turns], [False, False, True, True, False])
        self.assertEqual(episode.steer_turn, 2)
        line = status(episode)
        self.assertIn("**Steering:** layer 3, strength 4, at (1, 1), for 2 responses", line)
        self.assertIn("started in response 3 · 2 steered · 2 accepted moves under steering, 2 toward the destination", line)
        self.assertEqual([row[3] for row in timeline(episode)][3:5], ["Accepted · steered"] * 2)
        self.assertIn('stroke="#7c3aed" stroke-width="3"', board(episode))

    def test_steering_after_moves_can_last_to_the_end(self):
        episode, manager = run(checkpoint(steer_when={"moves": 1}, steer_responses=0))
        self.assertEqual(steering_sent(manager), [None, VECTOR, VECTOR, VECTOR, VECTOR])
        self.assertIn("after 1 accepted move, to the end of the run", status(episode))

    def test_steering_starts_once(self):
        # The character steps onto the cell, off it, and back: only the first
        # visit starts steering, and the second is past its one response.
        walk = ["south", "east", "west", "east", "north", "east", "east"]
        episode, manager = run(checkpoint(steer_responses=1), walk=walk)
        self.assertEqual(steering_sent(manager), [None, None, VECTOR, None, None, None, None])

    def test_a_trigger_the_run_never_meets_never_steers(self):
        episode, manager = run(checkpoint(steer_when={"cell": [3, 3]}), walk=["east", "east", "east"])
        self.assertEqual(steering_sent(manager), [None] * 3)
        self.assertIn("never started", status(episode))

    def test_a_vector_the_model_cannot_take_is_refused_before_the_run(self):
        manager = SteeringManager([], refuse="This vector requires other/model")
        episode = Episode(ROOM, checkpoint())
        with self.assertRaisesRegex(ValueError, "requires other/model"):
            list(stream_episode(episode, manager))
        self.assertEqual((episode.turns, episode.busy, manager.busy), ([], False, False))

    def test_a_disabled_vector_is_not_checked_or_sent(self):
        episode, manager = run(checkpoint(steering=dict(VECTOR, strength=0)))
        self.assertEqual(manager.checked, [])
        self.assertEqual(steering_sent(manager), [None] * 5)

    def test_bad_steering_settings_are_refused(self):
        for changes, message in (
                (dict(steer_when=None), "Say when steering starts"),
                (dict(steer_when={"cell": [1, 1], "moves": 2}), "Say when steering starts"),
                (dict(steer_when={"moves": -1}), "0 to 255"),
                (dict(steer_when={"cell": [0, 3]}), "destination"),
                (dict(steer_when={"cell": None}), "Name the cell steering starts at"),
                (dict(steer_responses=-1), "0 to 256"),
                (dict(steering=dict(VECTOR, vector=[])), "Vector must be"),
                (dict(steering={k: v for k, v in VECTOR.items() if k != "vector"} | {
                    "format": "chatlab-steering-reference-1", "vector_id": "0" * 64, "width": 3}), "whole"),
                (dict(steering=None), "needs a steering vector")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                Episode(ROOM, checkpoint(**changes))

    def test_a_saved_run_reads_back_and_a_forged_flag_is_refused(self):
        episode, _ = run(checkpoint())
        replay = from_payload(json.loads(json.dumps(episode.payload())))
        self.assertEqual(replay.config["steering"], VECTOR)
        self.assertEqual(replay.steer_turn, 2)
        for index, flag in ((1, True), (4, True), (2, False)):
            with self.subTest(response=index + 1):
                forged = json.loads(json.dumps(episode.payload()))
                forged["turns"][index]["steered"] = flag
                with self.assertRaisesRegex(ValueError, f"Response {index + 1} is recorded"):
                    from_payload(forged)
        plain, _ = run(BASE, walk=["east", "east", "east"])
        forged = json.loads(json.dumps(plain.payload()))
        forged["turns"][0]["steered"] = True
        with self.assertRaisesRegex(ValueError, "without a steering vector"):
            from_payload(forged)

    def test_a_fork_of_a_steered_response_is_steered_again(self):
        episode, _ = run(checkpoint())
        # The edit keeps the response's first token, so the regenerated
        # response is the rest of the same call.
        rest, _ = reply(ROOM, "east")
        manager = SteeringManager([(rest[1:], list(rest[1:].encode()) + [0]),
                                   reply(ROOM, "north"), reply(ROOM, "east")])
        with ModelService(lambda: manager).open_session() as session:
            fork = fork_token_edit(episode, 2, 0, "<", session)
        self.assertEqual([t["steered"] for t in fork.turns], [False, False])
        list(stream_episode(fork, manager))
        self.assertEqual(steering_sent(manager), [VECTOR, VECTOR, None])
        self.assertEqual(fork.steer_turn, 2)
        from_payload(json.loads(json.dumps(fork.payload())))

    def test_a_changing_map_keeps_its_checkpoints(self):
        maze = changing(POCKET)
        episode = Episode(maze, BASE | dict(waypoint=[0, 3], steering=VECTOR, steer_when={"cell": [0, 1]},
                                            steer_responses=1))
        for cell, message in (((0, 3), "waypoint is never closed"), ((0, 1), "steering cell is never closed"),
                              ((0, 2), "cut the character off from the waypoint")):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, message):
                episode.request_closure(cell)
        episode.request_closure((3, 0))
        forged = json.loads(json.dumps(episode.payload()))
        forged["config"]["map_updates"] = [dict(before_turn=0, position=[2, 0], closed_cell=[0, 3],
                                                grid=["....", ".###", "....", "...."])]
        with self.assertRaises(ValueError):
            from_payload(forged)


class TrialTests(unittest.TestCase):
    def write(self, trials, **extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trials.json"
        path.write_text(json.dumps({"format": TRIALS_FORMAT, "title": "Penguins", "trials": trials, **extra}))
        return path

    def trial(self, **config):
        keys = ("supplied_moves", "interrupt_after", "interruption_text", "prefix_tokens", "temperature",
                "sampling_seed", "per_turn_tokens", "token_budget", "attempt_budget", "goal_mode", "goal_hint")
        base = {key: BASE.get(key, "") for key in keys} | {"goal_mode": "coordinates", "goal_hint": ""}
        return {"id": "t1", "label": "Room · penguins", "maze": ROOM.to_dict() | {"seed": 0}, "openness": .7,
                "config": base | config}

    def test_a_trial_names_a_vector_from_the_file_and_overrides_its_strength(self):
        path = self.write([self.trial(waypoint=[1, 1], steering={"vector": "penguins", "strength": 8},
                                      steer_when={"cell": [1, 1]}, steer_responses=3)],
                          vectors={"penguins": VECTOR})
        episode = prepare_trial(read_trials(path), "t1", Episode(ROOM, BASE))
        self.assertEqual(episode.config["steering"], dict(VECTOR, strength=8.0))
        self.assertEqual((episode.config["waypoint"], episode.config["steer_responses"]), ([1, 1], 3))

    def test_an_inline_vector_is_read_too(self):
        path = self.write([self.trial(steering=VECTOR, steer_when={"moves": 2}, steer_responses=0)])
        self.assertEqual(prepare_trial(read_trials(path), "t1", Episode(ROOM, BASE)).config["steering"], VECTOR)

    def test_bad_trial_checkpoints_are_refused(self):
        for trial, extra, message in (
                (self.trial(steering={"vector": "missing"}, steer_when={"moves": 1}), {}, "does not hold"),
                (self.trial(steer_when={"moves": 1}), {}, "needs a steering vector"),
                (self.trial(waypoint=[0, 3]), {}, "destination")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, f"Trial 't1': .*{message}"):
                read_trials(self.write([trial], **extra))


class PageTests(unittest.TestCase):
    def build(self, manager=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        context = SimpleNamespace(tokens=TokenInspector(), models=manager, data_dir=self.directory,
                                  navigation=SimpleNamespace(open_models=lambda button, wanted=None: None))
        with gr.Blocks() as demo:
            build_page(context)
        self.addCleanup(demo.close)
        return {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}

    def test_parse_cell_reads_the_ways_a_cell_is_typed(self):
        for text in ("3, 3", "3 3", "(3,3)", " [3, 3] "):
            self.assertEqual(parse_cell(text, "waypoint"), [3, 3])
        self.assertIsNone(parse_cell("  ", "waypoint"))
        for text in ("3", "a, b", "1, 2, 3"):
            with self.assertRaisesRegex(ValueError, "row and a column"):
                parse_cell(text, "waypoint")

    def test_the_pane_imports_a_vector_starts_a_steered_episode_and_fills_from_a_run(self):
        callbacks = self.build()
        path = self.directory / "penguins.json"
        path.write_text(json.dumps(VECTOR))
        vector, note, strength, layer = callbacks["import_vector"].fn(str(path))
        self.assertEqual((vector, strength, layer), (VECTOR, 4.0, 3))
        self.assertIn("test/model · 3 entries · layer 3", note)
        self.assertEqual(callbacks["import_vector"].fn(None)[0], None)
        prepare = callbacks["prepare_episode"]
        values = (5, 20260911, 10, .7, 0, 3, "", 8, .7, 1, 200, 4000, 20, 1024, 4,
                  "coordinates", "", "sys", "Pass through the waypoint.", False)
        episode = Episode(ROOM, BASE)
        drawn = generate(*values[:4])
        open_cell = next([r, c] for r in range(5) for c in range(5)
                         if drawn.open((r, c)) and (r, c) not in (drawn.start, drawn.goal))
        # A blank steering cell starts steering at the waypoint.
        new = prepare.fn(episode, False, "s", None, *values, cell_text(open_cell), vector, 6, 1, "cell", "", 3, 2)[0]
        self.assertEqual(new.config["waypoint"], open_cell)
        self.assertEqual(new.config["steer_when"], {"cell": open_cell})
        self.assertEqual(new.config["steering"], dict(VECTOR, strength=6.0, layer=1))
        new = prepare.fn(episode, False, "s", None, *values, "", vector, 6, 1, "moves", "", 3, 0)[0]
        self.assertEqual((new.config["steer_when"], new.config["steer_responses"]), ({"moves": 3}, 0))
        self.assertNotIn("waypoint", new.config)
        plain = prepare.fn(episode, False, "s", None, *values, *NO_CHECKPOINT)[0]
        self.assertNotIn("steering", plain.config)
        for steer, message in (((None, 1, 0, "cell", "", 3, 1), "Import a steering vector"),
                                    ((vector, 1, 0, "cell", "", 3, 1), "set a waypoint"),
                                    ((vector, 1, 0, "off", "", 3, 1), None)):
            with self.subTest(message=message):
                if message is None:
                    prepare.fn(episode, False, "s", None, *values, "", *steer)
                    continue
                with self.assertRaisesRegex(gr.Error, message):
                    prepare.fn(episode, False, "s", None, *values, "", *steer)
        ran, _ = run(checkpoint())
        loaded = callbacks["load"].fn(str(ran.export()), episode, False, "s", None)
        outputs = callbacks["load"].outputs
        self.assertEqual(len(loaded), len(outputs))
        start = 16
        self.assertEqual(loaded[start:start + 8], ("1, 1", VECTOR, 4.0, 3, "cell", "1, 1", 3, 2))
        self.assertIn("layer 3", loaded[start + 8])


if __name__ == "__main__":
    unittest.main()
