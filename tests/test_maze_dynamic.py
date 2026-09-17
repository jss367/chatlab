import copy
import json
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from extensions.maze_experiments.dynamic_maze import (FORMAT, ChangingMaze, changing, check_closure,
                                                      close_cell, environment_id, load_maze)
from extensions.maze_experiments.maze import GOAL_MODES, Maze, call_text
from extensions.maze_experiments.page import board, build_page, scenario_values, status
from extensions.maze_experiments.runner import Episode, fork_token_edit, from_payload, stream_episode
from extension_api import TokenInspector

from test_maze import CONFIG, Manager

# A short corridor and one long way round, so a single closure can send the
# character the long way without cutting it off, and a second one strands it.
GRID = ("....",
        ".##.",
        ".##.",
        "....")
OPEN = Maze(GRID, (0, 0), (0, 3))
# A maze whose top row is a pocket: sealing it leaves the start its own route
# to the destination while cutting off a character standing inside.
POCKET = Maze(("....", "###.", "....", "...."), (2, 0), (3, 3))
CHANGING_CONFIG = CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 2000,
                            "openness": .7}


def reply(maze, direction):
    """One response that calls move, as the fixture manager hands it back."""
    text = call_text(maze.tool_id(), direction)
    return text, list(text.encode()) + [0]


class ChangingMapTests(unittest.TestCase):
    def episode_with_a_closure(self):
        """A run that closes the cell ahead of the character, which then walks into it."""
        maze = changing(OPEN)
        manager = Manager([reply(maze, "east"), reply(maze, "east"), reply(maze, "west")])
        episode = Episode(maze, CHANGING_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        episode.request_closure((0, 2))
        list(stream_episode(episode, manager, single_step=True))
        list(stream_episode(episode, manager, single_step=True))
        return episode, manager

    def test_the_tool_identifier_survives_a_closure_in_every_goal_mode(self):
        maze = changing(OPEN)
        closed = close_cell(maze, (0, 2))
        for mode in GOAL_MODES:
            with self.subTest(goal_mode=mode):
                self.assertEqual(closed.tool_id(mode), maze.tool_id(mode))
                self.assertEqual(closed.state((0, 0), goal_mode=mode, goal_hint="north")["maze_id"],
                                 maze.environment_id)
        # Which is the reason this maze exists: a fixed one renames itself when
        # its grid changes, under a concealed destination as well as a disclosed
        # one, and the model's next call would come back addressed to the wrong
        # maze rather than blocked by the new wall.
        renamed = Maze(closed.grid, OPEN.start, OPEN.goal)
        self.assertNotEqual(renamed.maze_id, OPEN.maze_id)
        self.assertNotEqual(renamed.tool_id("hidden"), OPEN.tool_id("hidden"))
        # The identifier names no destination, so concealing the goal conceals
        # as much here as it does on a fixed maze.
        self.assertEqual(maze.environment_id, Maze(GRID, OPEN.start, (3, 3)).tool_id("hidden"))
        self.assertEqual(environment_id(OPEN), maze.environment_id)

    def test_a_closure_has_to_leave_a_maze_the_run_could_have_begun_with(self):
        maze = changing(OPEN)
        for cell, message in (((0, 0), "start and the destination"), ((0, 3), "start and the destination"),
                              ((1, 1), "open cell inside the maze"), ((4, 0), "open cell inside the maze")):
            with self.subTest(cell=cell):
                with self.assertRaisesRegex(ValueError, message):
                    close_cell(maze, cell)
        for cell in ((0, 1, 2), ("0", 1), 3):
            with self.subTest(cell=cell):
                with self.assertRaisesRegex(ValueError, "one cell as a row and a column"):
                    close_cell(maze, cell)
        # Closing the short corridor sends the character the long way round;
        # closing the long way round as well strands the start.
        narrowed = close_cell(maze, (0, 2))
        with self.assertRaisesRegex(ValueError, "no route from the start"):
            close_cell(narrowed, (3, 0))
        # The character's own cell is never walled in, and a closure that only
        # cuts the character off is refused even where the start still has a
        # route of its own.
        with self.assertRaisesRegex(ValueError, "standing in cannot be closed"):
            check_closure(maze, (0, 1), (0, 1))
        with self.assertRaisesRegex(ValueError, "cut the character off"):
            check_closure(changing(POCKET), (0, 0), (1, 3))

    def test_a_closure_lands_between_responses_and_the_reply_carries_the_new_map(self):
        episode, manager = self.episode_with_a_closure()
        closed = close_cell(episode.maze, (0, 2))
        self.assertEqual(episode.config["map_updates"],
                         [dict(before_turn=1, position=[0, 1], closed_cell=[0, 2], grid=list(closed.grid))])
        # The response that met the wall was generated under the map it had
        # already been given, so its own prompt still shows the cell open.
        self.assertEqual(json.loads(manager.calls[1][0][-1]["content"])["grid"], list(GRID))
        # It meets the change in the simulator's reply, which refuses the move
        # as blocked rather than as addressed to some other maze, and carries
        # the current grid under the identifier the model has had all along.
        self.assertEqual(episode.events[1]["error"], "blocked_move")
        self.assertEqual(episode.position, (0, 0))
        state = json.loads(episode.messages[-3]["content"])
        self.assertEqual(state["grid"], list(closed.grid))
        self.assertEqual(state["maze_id"], episode.maze.environment_id)
        self.assertTrue(episode.manual_intervention)

    def test_the_run_replays_each_response_against_the_map_it_acted_on(self):
        episode, _ = self.episode_with_a_closure()
        payload = json.loads(json.dumps(episode.payload()))
        self.assertEqual(payload["format"], FORMAT)
        self.assertEqual(payload["maze"]["environment_id"], episode.maze.environment_id)
        self.assertEqual(payload["maze"]["grid"], list(GRID))
        replay = from_payload(payload)
        self.assertEqual(replay.position, episode.position)
        self.assertEqual(replay.config["map_updates"], episode.config["map_updates"])
        self.assertEqual(replay.current_maze.grid, episode.current_maze.grid)
        self.assertEqual(replay.current_maze.tool_id(), episode.maze.environment_id)
        self.assertTrue(replay.replay_only)
        self.assertEqual(board(replay, None), board(episode, None))

    def test_an_accepted_move_is_replayed_against_the_version_it_was_made_on(self):
        episode, _ = self.episode_with_a_closure()
        payload = json.loads(json.dumps(episode.payload()))
        # A file whose closure lands before the move that walks through the
        # cell: the recorded transition is one the map it claims never allowed.
        closed = close_cell(episode.maze, (0, 1))
        payload["config"]["map_updates"] = [dict(before_turn=0, position=[0, 0], closed_cell=[0, 1],
                                                 grid=list(closed.grid))]
        with self.assertRaisesRegex(ValueError, "invalid transition"):
            from_payload(payload)

    def test_a_replay_is_refused_when_its_record_of_the_map_does_not_hold(self):
        episode, _ = self.episode_with_a_closure()
        payload = json.loads(json.dumps(episode.payload()))
        for key, value, message in (("grid", ["...."] * 4, "closure does not produce"),
                                    ("position", [1, 0], "position the run's path never reached"),
                                    ("closed_cell", [0, 0], "start and the destination"),
                                    ("closed_cell", [0, 1], "standing in cannot be closed"),
                                    ("before_turn", 99, "one response boundary, in order")):
            with self.subTest(field=key, value=value):
                broken = copy.deepcopy(payload)
                broken["config"]["map_updates"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    from_payload(broken)
        # Two closures at one boundary is a timeline no run could have produced.
        doubled = copy.deepcopy(payload)
        doubled["config"]["map_updates"] *= 2
        with self.assertRaisesRegex(ValueError, "one response boundary, in order"):
            from_payload(doubled)
        # And the map the run starts from has to be the one its identifier names.
        for key, value in (("environment_id", "maze-000000000000"), ("grid", ["...."] * 4)):
            with self.subTest(maze=key):
                broken = copy.deepcopy(payload)
                broken["maze"][key] = value
                with self.assertRaises(ValueError):
                    from_payload(broken)

    def test_a_replay_is_refused_when_its_record_of_a_dropped_closure_cannot_hold(self):
        maze = changing(OPEN)
        manager = Manager([reply(maze, "east"), reply(maze, "east"), reply(maze, "west")])
        episode = Episode(maze, CHANGING_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        frames = stream_episode(episode, manager, single_step=True)
        next(frames)
        episode.request_closure((0, 2))
        list(frames)
        list(stream_episode(episode, manager, single_step=True))
        payload = json.loads(json.dumps(episode.payload()))
        self.assertEqual(len(payload["dropped_closures"]), 1)
        self.assertEqual(from_payload(payload).dropped_closures, episode.dropped_closures)
        for value, message in (("garbage", "must be a list"),
                               (["nope"], "must be an object"),
                               ([dict(before_turn=99, cell=[0, 1], reason="x")], "free response boundary"),
                               ([dict(before_turn=-1, cell=[0, 1], reason="x")], "free response boundary"),
                               ([dict(cell=[0, 1], reason="x")], "free response boundary"),
                               ([dict(before_turn=0, cell=[0, 1], reason="x")] * 2, "free response boundary"),
                               ([dict(before_turn=0, cell="0,1", reason="x")], "row and a column"),
                               ([dict(before_turn=0, cell=[99, 99], reason="x")], "open cell inside the maze"),
                               ([dict(before_turn=0, cell=[1, 1], reason="x")], "open cell inside the maze"),
                               ([dict(before_turn=0, cell=[0, 0], reason="x")], "start and the destination"),
                               ([dict(before_turn=0, cell=[0, 1], reason=" ")], "records why it was dropped")):
            with self.subTest(dropped=value):
                broken = copy.deepcopy(payload)
                broken["dropped_closures"] = value
                with self.assertRaisesRegex(ValueError, message):
                    from_payload(broken)
        # One closure queues at a time, so no boundary holds a drop and a
        # closure the map took.
        clash = copy.deepcopy(payload)
        clash["dropped_closures"] = [dict(before_turn=1, cell=[0, 1], reason="x")]
        clash["config"]["map_updates"] = [dict(
            before_turn=1, position=[0, 1], closed_cell=[0, 2],
            grid=list(close_cell(maze, (0, 2)).grid))]
        with self.assertRaisesRegex(ValueError, "free response boundary"):
            from_payload(clash)

    def test_a_fixed_map_run_neither_carries_closures_nor_makes_them(self):
        episode, _ = self.episode_with_a_closure()
        payload = json.loads(json.dumps(episode.payload()))
        payload["format"] = "chatlab-maze-run-1"
        with self.assertRaisesRegex(ValueError, "chatlab-maze-run-2"):
            from_payload(payload)
        fixed = Episode(OPEN, CHANGING_CONFIG)
        self.assertEqual(fixed.payload()["format"], "chatlab-maze-run-1")
        self.assertNotIn("environment_id", fixed.payload()["maze"])
        self.assertNotIn("map_updates", fixed.payload()["config"])
        self.assertEqual(fixed.payload()["dropped_closures"], [])
        claiming = json.loads(json.dumps(fixed.payload()))
        claiming["dropped_closures"] = [dict(before_turn=0, cell=[0, 1], reason="x")]
        with self.assertRaisesRegex(ValueError, "chatlab-maze-run-2"):
            from_payload(claiming)
        with self.assertRaisesRegex(ValueError, "map is fixed"):
            fixed.request_closure((0, 1))
        self.assertEqual(from_payload(json.loads(json.dumps(fixed.payload()))).current_maze, OPEN)

    def test_a_queued_closure_that_cannot_happen_by_then_is_dropped_and_recorded(self):
        maze = changing(OPEN)
        manager = Manager([reply(maze, "east"), reply(maze, "east"), reply(maze, "west")])
        episode = Episode(maze, CHANGING_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        # Queued while the next response is already generating, which is the
        # one moment the character can move between the request and the change.
        frames = stream_episode(episode, manager, single_step=True)
        next(frames)
        episode.request_closure((0, 2))
        list(frames)
        self.assertEqual(episode.position, (0, 2))
        # The autosave written here carries a cell the character has just moved
        # onto. That is a closure about to be dropped, not a file no run could
        # have written, so the pending cell is read without asking the position.
        waiting = json.loads(json.dumps(episode.payload()))
        self.assertEqual(waiting["close_next"], [0, 2])
        self.assertEqual(from_payload(waiting).close_next, [0, 2])
        list(stream_episode(episode, manager, single_step=True))
        self.assertEqual(episode.config.get("map_updates", []), [])
        self.assertEqual(episode.dropped_closures,
                         [dict(before_turn=2, cell=[0, 2],
                               reason="The cell the character is standing in cannot be closed.")])
        self.assertIn("1 closure dropped", status(episode))
        self.assertEqual(from_payload(json.loads(json.dumps(episode.payload()))).dropped_closures,
                         episode.dropped_closures)

    def test_only_one_closure_queues_at_a_time(self):
        maze = changing(OPEN)
        episode = Episode(maze, CHANGING_CONFIG)
        episode.request_closure((0, 1))
        # The reader has been told (0, 1) will close, so a second request is
        # refused rather than replacing it with a cell nothing would record.
        with self.assertRaisesRegex(ValueError, r"Row 0, column 1 is already queued"):
            episode.request_closure((0, 2))
        self.assertEqual(episode.close_next, (0, 1))
        self.assertEqual(episode.dropped_closures, [])
        # Once it has landed, the next one queues as normal.
        manager = Manager([reply(maze, "east"), reply(maze, "south")])
        list(stream_episode(episode, manager, single_step=True))
        self.assertEqual(episode.close_next, ())
        self.assertEqual(len(episode.config["map_updates"]), 1)
        episode.request_closure((0, 2))
        self.assertEqual(episode.close_next, (0, 2))

    def test_queueing_a_closure_is_one_operation_under_the_lock(self):
        # The page registers the close button off the queue, so two clicks run
        # as two callbacks at once and the test of an empty queue has to be
        # part of the same operation as filling it.
        episode = Episode(changing(OPEN), CHANGING_CONFIG)
        episode.lock.acquire()
        finished = []
        waiting = threading.Thread(target=lambda: (episode.request_closure((0, 1)), finished.append(True)))
        waiting.start()
        waiting.join(.3)
        self.assertEqual(finished, [])
        self.assertEqual(episode.close_next, ())
        episode.lock.release()
        waiting.join(2)
        self.assertEqual(finished, [True])
        self.assertEqual(episode.close_next, (0, 1))

    def test_a_run_saved_with_a_closure_queued_says_so_and_ending_drops_it(self):
        maze = changing(OPEN)
        manager = Manager([reply(maze, "east")])
        episode = Episode(maze, CHANGING_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        episode.request_closure((0, 2))
        # Paused with a cell queued: the run can still reach it, so the export
        # carries it rather than reporting a closure nobody asked for.
        self.assertEqual(json.loads(json.dumps(episode.payload()))["close_next"], [0, 2])
        self.assertEqual(from_payload(json.loads(json.dumps(episode.payload()))).close_next, [0, 2])
        self.assertEqual(episode.dropped_closures, [])
        # A pending cell has to be one the run could have queued, and no run
        # that has ended is still waiting to close one.
        for value, message in (([0], "row and a column"), ([99, 99], "open cell inside the maze"),
                               ([0, 3], "start and the destination")):
            with self.subTest(pending=value):
                broken = json.loads(json.dumps(episode.payload()))
                broken["close_next"] = value
                with self.assertRaisesRegex(ValueError, message):
                    from_payload(broken)
        ended = json.loads(json.dumps(episode.payload()))
        ended["phase"] = "arrived"
        with self.assertRaisesRegex(ValueError, "still be waiting to close"):
            from_payload(ended)
        fixed = json.loads(json.dumps(Episode(OPEN, CHANGING_CONFIG).payload()))
        fixed["close_next"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "chatlab-maze-run-2"):
            from_payload(fixed)
        # Stopping ends the run before any response can apply it, so the cell
        # the reader was told would close is recorded as one that never did.
        episode.request_stop()
        self.assertEqual(episode.close_next, ())
        self.assertEqual(episode.dropped_closures,
                         [dict(before_turn=1, cell=[0, 2],
                               reason="The episode ended before the closure could land.")])
        self.assertIn("1 closure dropped", status(episode))

    def test_a_closure_queued_as_the_episode_ends_is_dropped_by_the_stream(self):
        maze = changing(OPEN)
        # A response with no call in it ends the episode as abandonment, so the
        # cell queued while it was generating has no response left to land in.
        manager = Manager([("no call here", list(b"no call here") + [0])])
        episode = Episode(maze, CHANGING_CONFIG)
        frames = stream_episode(episode, manager)
        next(frames)
        episode.request_closure((0, 1))
        list(frames)
        self.assertEqual(episode.phase, "abandoned")
        self.assertEqual(episode.config.get("map_updates", []), [])
        self.assertEqual(episode.close_next, ())
        self.assertEqual(episode.dropped_closures,
                         [dict(before_turn=1, cell=[0, 1],
                               reason="The episode ended before the closure could land.")])

    def test_a_fork_carries_the_closures_the_map_refused_as_well_as_the_ones_it_took(self):
        maze = changing(OPEN)
        manager = Manager([reply(maze, "east"), reply(maze, "east"), reply(maze, "west")])
        episode = Episode(maze, CHANGING_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        frames = stream_episode(episode, manager, single_step=True)
        next(frames)
        episode.request_closure((0, 2))
        list(frames)
        list(stream_episode(episode, manager, single_step=True))
        self.assertEqual(len(episode.dropped_closures), 1)
        # The fork keeps the responses generated after the failed intervention,
        # so it has to keep the record that one was attempted and refused.
        original = episode.turns[2]["text"]
        index = original.index("west")
        with manager.open_session() as session:
            forked = fork_token_edit(episode, 2, index, "east", session)
        self.assertEqual(forked.dropped_closures, episode.dropped_closures)
        self.assertIsNot(forked.dropped_closures[0], episode.dropped_closures[0])
        self.assertIn("1 closure dropped", status(forked))
        # A fork before the attempt reports no attempt, because it kept no
        # response that ran after one.
        with manager.open_session() as session:
            earlier = fork_token_edit(episode, 1, episode.turns[1]["text"].index("east"), "west", session)
        self.assertEqual(earlier.dropped_closures, [])

    def test_a_fork_replays_the_closures_that_preceded_the_edited_response(self):
        episode, manager = self.episode_with_a_closure()
        original = episode.turns[2]["text"]
        index = original.index("west")
        with manager.open_session() as session:
            forked = fork_token_edit(episode, 2, index, "east", session)
        self.assertEqual(forked.config["map_updates"], episode.config["map_updates"])
        self.assertEqual(forked.current_maze.grid, episode.current_maze.grid)
        self.assertEqual(len(forked.turns), 2)
        self.assertEqual(forked.position, (0, 1))
        # The closure landed before the response being edited, so the branch is
        # judged against the map as changed: east is the wall it made.
        suffix = original[index + len("west"):]
        manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
        list(stream_episode(forked, manager, single_step=True))
        self.assertEqual(forked.events[-1]["error"], "blocked_move")
        self.assertEqual(forked.position, (0, 1))
        self.assertEqual(episode.config["map_updates"][0]["before_turn"], 1)

    def test_a_fork_before_a_closure_leaves_that_closure_behind(self):
        episode, manager = self.episode_with_a_closure()
        original = episode.turns[0]["text"]
        index = original.index("east")
        with manager.open_session() as session:
            forked = fork_token_edit(episode, 0, index, "south", session)
        self.assertEqual(forked.config["map_updates"], [])
        self.assertEqual(forked.current_maze.grid, OPEN.grid)
        self.assertEqual(len(forked.turns), 0)
        suffix = original[index + len("east"):]
        manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
        list(stream_episode(forked, manager, single_step=True))
        self.assertEqual(forked.position, (1, 0))
        # The parent keeps its own map and its own record of the change.
        self.assertEqual(episode.current_maze.grid, close_cell(episode.maze, (0, 2)).grid)

    def test_the_board_shows_each_response_under_the_map_it_acted_on(self):
        episode, _ = self.episode_with_a_closure()
        closed_fill = 'fill="#78350f"'
        self.assertNotIn(closed_fill, board(episode, -1))
        self.assertNotIn(closed_fill, board(episode, 0))
        self.assertEqual(board(episode, 1).count(closed_fill), 1)
        self.assertEqual(board(episode, None).count(closed_fill), 1)
        self.assertIn("Closed during the run", board(episode, None))
        self.assertNotIn("Closed during the run", board(Episode(OPEN, CHANGING_CONFIG)))
        self.assertIn("1 cell closed", status(episode))
        self.assertNotIn("**Map:**", status(Episode(OPEN, CHANGING_CONFIG)))

    def test_the_pane_offers_the_change_to_changing_runs_and_refuses_bad_cells(self):
        episode, _ = self.episode_with_a_closure()
        inspector = TokenInspector()
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=inspector, models=Manager([]), data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                close = {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}["close_map_cell"]
                for row, column, message in ((0, 0, "standing in"), (0, 3, "start and the destination"),
                                             (3, 0, "no route from the start"), (None, 1, "Enter the row")):
                    with self.subTest(cell=(row, column)):
                        with self.assertRaisesRegex(gr.Error, message):
                            close.fn(episode, row, column)
                self.assertEqual(episode.close_next, ())
                self.assertIn("Closing (0, 1)", close.fn(episode, 0, 1)[1])
                self.assertEqual(episode.close_next, (0, 1))
            finally:
                demo.close()
        # An uploaded run fills the changing-map checkbox from the run itself.
        self.assertIs(scenario_values(episode)[-2], True)
        self.assertIs(scenario_values(Episode(OPEN, CHANGING_CONFIG))[-2], False)

    def test_the_identifier_is_derived_rather_than_taken_from_the_file(self):
        maze = changing(OPEN)
        self.assertEqual(load_maze(maze.to_dict()), maze)
        self.assertIsInstance(load_maze(maze.to_dict()), ChangingMaze)
        with self.assertRaisesRegex(ValueError, "does not belong to the map"):
            load_maze(dict(maze.to_dict(), environment_id="maze-000000000000"))
        with self.assertRaisesRegex(ValueError, "needs an environment identifier"):
            ChangingMaze(OPEN.grid, OPEN.start, OPEN.goal, 0, "")


if __name__ == "__main__":
    unittest.main()
