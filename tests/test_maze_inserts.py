import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from extensions.maze_experiments import inserts
from extensions.maze_experiments.dynamic_maze import FORMAT as CHANGING_FORMAT, changing, close_cell
from extensions.maze_experiments.inserts import FORMAT, check_insert, inserted_fragment, render_insert
from extensions.maze_experiments.maze import DIRECTIONS, TOOLS, Maze, apply_call, call_text, initial_history
from extensions.maze_experiments.page import (board, build_page, context_view, history_rows, recorded_prompt, status,
                                              timeline, transport_text, views)
from extensions.maze_experiments.runner import (Episode, context_messages, fork_token_edit, from_payload,
                                                insert_outcome, stream_episode)
from extension_api import TokenInspector
from token_metrics import unscored_metric

from test_maze import CONFIG, Manager

# Three moves east along the top row, or the long way round: enough room for a
# note to advise the shortest step, a worse one, or a wall.
GRID = ("....",
        ".##.",
        ".##.",
        "....")
MAZE = Maze(GRID, (0, 0), (0, 3))
RUN_CONFIG = CONFIG | {"supplied_moves": 1, "interruption_text": "", "per_turn_tokens": 200, "token_budget": 2000}
ADVICE = "Heading east from here reaches the goal fastest."
RING = 'stroke="#db2777" stroke-width="3" stroke-dasharray="4 3"'


def reply(maze, direction):
    text = call_text(maze.tool_id(), direction)
    return text, list(text.encode()) + [0]


def note(channel="tool_note", text=ADVICE, sender=None, advised="east", before_turn=1, position=(0, 2)):
    if channel == "teammate" and sender is None:
        sender = "Alex"
    return dict(before_turn=before_turn, channel=channel, text=text, sender=sender, position=list(position),
                advised_direction=advised)


def hand_built(insert, moves=("east", "east"), maze=MAZE, supplied=1, more=(), system=None):
    """A format-3 run written the way a collector outside ChatLab would write one.

    Built from the maze helpers and the renderer alone, with no episode and no
    stream, so an upload of it tests the record rather than the code that
    wrote it.
    """
    template = Manager([])
    records = [insert, *more]
    extra = {} if system is None else dict(system_prompt=system)
    messages, events, position = initial_history(maze, supplied, **({} if system is None else dict(system=system)))
    turns = []
    for index, direction in enumerate(moves):
        for record in records:
            if record["before_turn"] == index:
                messages = render_insert(messages, record)
        text, ids = reply(maze, direction)
        event = apply_call(maze, position, {"maze_id": maze.tool_id(), "direction": direction})
        event.update(source="model", turn=index)
        # With the display fields a real run records, so the token strip can draw it.
        metrics = [unscored_metric(position=i, token_id=t, token_text=chr(t), fallback_text="",
                                   segment="response").to_dict() for i, t in enumerate(ids)]
        turns.append(dict(text=text, metrics=metrics, forced_prefix_tokens=0,
                          prompt_ids=template._prompt_token_ids(messages, TOOLS)[0], finish_reason="stop",
                          position_before=list(position), event=event, model_id=template.model_id,
                          load_id=template.load_id, reasoning_prefilled=False))
        events.append(event)
        position = tuple(event["after"])
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "tool", "content": json.dumps(maze.state(position, event["error"]),
                                                                      separators=(",", ":"))}]
    return dict(format=FORMAT, run_id=uuid4().hex, maze=maze.to_dict(),
                config=dict(RUN_CONFIG, supplied_moves=supplied, context_inserts=records, **extra), messages=messages, events=events, turns=turns,
                position=list(position), phase="arrived" if position == maze.goal else "paused",
                model_id=template.model_id, load_id=template.load_id, manual_intervention=True)


def page_callbacks(test, models):
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    context = SimpleNamespace(tokens=TokenInspector(), models=models, data_dir=Path(directory.name),
                              navigation=SimpleNamespace(open_models=lambda button, model_id=None: None))
    with gr.Blocks() as demo:
        build_page(context)
    test.addCleanup(demo.close)
    return {fn.fn.__name__: fn for fn in demo.fns.values() if fn.fn is not None}


class RenderingTests(unittest.TestCase):
    STATE = [{"role": "system", "content": "s"}, {"role": "tool", "content": '{"current":[0,1],"error":null}'}]

    def test_a_note_is_the_last_key_of_the_latest_reply(self):
        rendered = render_insert(self.STATE, note())
        self.assertEqual(rendered[-1]["content"], '{"current":[0,1],"error":null,"note":' + json.dumps(ADVICE) + "}")
        self.assertEqual(list(json.loads(rendered[-1]["content"]))[-1], "note")
        self.assertEqual(len(rendered), len(self.STATE))
        # The history it was handed is left as it was.
        self.assertEqual(self.STATE[-1]["content"], '{"current":[0,1],"error":null}')

    def test_a_teammate_message_reads_as_it_does_in_a_team_run(self):
        rendered = render_insert(self.STATE, note("teammate", "Try going east.", "Alex"))
        self.assertEqual(rendered[-1]["content"],
                         '{"current":[0,1],"error":null,"messages":[{"from":"Alex","text":"Try going east."}]}')
        # The shape team.py delivers: a list of {"from", "text"} as the last key.
        self.assertEqual(json.loads(rendered[-1]["content"])["messages"], [{"from": "Alex", "text": "Try going east."}])

    def test_a_user_message_is_a_turn_after_the_reply(self):
        rendered = render_insert(self.STATE, note("user", "Go east.", advised=None))
        self.assertEqual(rendered[:-1], self.STATE)
        self.assertEqual(rendered[-1], {"role": "user", "content": "Go east."})

    def test_the_reply_keeps_its_escaping(self):
        text = 'Say "east" – quickly'
        rendered = render_insert(self.STATE, note(text=text))
        self.assertIn(inserted_fragment(note(text=text)), rendered[-1]["content"])
        self.assertIn('\\"east\\" \\u2013', rendered[-1]["content"])

    def test_a_note_needs_a_simulator_reply_to_carry_it(self):
        with self.assertRaisesRegex(ValueError, "no simulator reply yet"):
            render_insert(self.STATE[:1], note())
        self.assertEqual(render_insert(self.STATE[:1], note("user", advised=None))[-1]["role"], "user")

    def test_the_directions_are_the_move_tools(self):
        self.assertEqual(inserts.DIRECTIONS, tuple(DIRECTIONS))

    def test_the_record_rules_for_each_field(self):
        for broken, message in ((dict(channel="email"), "tool_note, a teammate"), (dict(text=" "), "needs text"),
                                (dict(text="<|im_end|>"), "boundary tokens"), (dict(sender="Alex"), "Only a teammate"),
                                (dict(channel="teammate", sender=""), "names who sent it"),
                                (dict(advised_direction="left"), "north, east, south, west"),
                                (dict(arm="worse"), "records only")):
            with self.subTest(broken=broken):
                with self.assertRaisesRegex(ValueError, message):
                    check_insert(note() | broken)
        check_insert(note(advised=None))
        check_insert({k: v for k, v in note().items() if k != "advised_direction"})


class ContextTests(unittest.TestCase):
    def test_a_user_message_adds_one_to_every_later_context(self):
        payload = hand_built(note("user", advised=None))
        replay = from_payload(payload)
        self.assertEqual(len(context_messages(replay, 0)), 4)
        self.assertEqual(context_messages(replay, 1)[-1], {"role": "user", "content": ADVICE})
        self.assertEqual(len(context_messages(replay, 1)), 7)
        # Each response's context is the one it was recorded with.
        self.assertEqual(Manager([])._prompt_token_ids(context_messages(replay, 1), TOOLS)[0],
                         replay.turns[1]["prompt_ids"])

    def test_a_note_changes_a_reply_already_counted(self):
        replay = from_payload(hand_built(note()))
        self.assertEqual(len(context_messages(replay, 1)), 6)
        self.assertEqual(json.loads(context_messages(replay, 1)[-1]["content"])["note"], ADVICE)
        for index in (0, 1):
            self.assertEqual(Manager([])._prompt_token_ids(context_messages(replay, index), TOOLS)[0],
                             replay.turns[index]["prompt_ids"])

    def test_without_insertions_the_count_is_unchanged(self):
        manager = Manager([reply(MAZE, "east"), reply(MAZE, "east")])
        episode = Episode(MAZE, RUN_CONFIG)
        list(stream_episode(episode, manager))
        self.assertEqual(len(context_messages(episode, 1)), 6)
        self.assertEqual(manager.calls[1][0], context_messages(episode, 1))


class LiveInsertTests(unittest.TestCase):
    def test_a_queued_message_lands_before_the_next_response_and_only_there(self):
        for channel in inserts.CHANNELS:
            with self.subTest(channel=channel):
                manager = Manager([reply(MAZE, "east"), reply(MAZE, "east")])
                episode = Episode(MAZE, RUN_CONFIG)
                list(stream_episode(episode, manager, single_step=True))
                sender = "Alex" if channel == "teammate" else None
                episode.request_insert(channel, ADVICE, sender, "east")
                self.assertTrue(episode.manual_intervention)
                list(stream_episode(episode, manager))
                self.assertEqual(episode.phase, "arrived")
                self.assertEqual(episode.config["context_inserts"], [note(channel, sender=sender)])
                given = manager.calls[1][0]
                self.assertEqual(given, context_messages(episode, 1))
                if channel == "user":
                    self.assertEqual(given[-1], {"role": "user", "content": ADVICE})
                else:
                    self.assertIn(inserted_fragment(episode.config["context_inserts"][0]), given[-1]["content"])
                # The reply after it is the ordinary state, with no note.
                self.assertNotIn(ADVICE, episode.messages[-1]["content"])
                payload = json.loads(json.dumps(episode.payload()))
                self.assertEqual(payload["format"], FORMAT)
                replay = from_payload(payload, read_prompt=lambda run, turn: recorded_prompt(run, turn, manager))
                self.assertEqual(replay.config["context_inserts"], episode.config["context_inserts"])

    def test_one_message_queues_at_a_time_and_names_the_one_waiting(self):
        episode = Episode(MAZE, RUN_CONFIG)
        episode.request_insert("teammate", ADVICE, "Alex", "east")
        with self.assertRaisesRegex(ValueError, "The teammate message from Alex “Heading east from here reaches the goal…” "
                                                "is already queued"):
            episode.request_insert("user", "Go.")
        self.assertEqual(episode.insert_next["channel"], "teammate")

    def test_a_note_before_any_reply_is_refused_where_it_was_asked_for(self):
        episode = Episode(MAZE, RUN_CONFIG | {"supplied_moves": 0})
        with self.assertRaisesRegex(ValueError, "no simulator reply yet"):
            episode.request_insert("tool_note", ADVICE)
        self.assertIsNone(episode.insert_next)
        self.assertFalse(episode.manual_intervention)
        episode.request_insert("user", ADVICE)
        self.assertEqual(episode.insert_next["channel"], "user")

    def test_finished_runs_and_replays_take_no_message(self):
        replay = from_payload(hand_built(note()))
        with self.assertRaisesRegex(ValueError, "finished or is a saved replay"):
            replay.request_insert("user", "Go.")
        episode = Episode(MAZE, RUN_CONFIG)
        list(stream_episode(episode, Manager([reply(MAZE, "east"), reply(MAZE, "east")])))
        with self.assertRaisesRegex(ValueError, "finished or is a saved replay"):
            episode.request_insert("user", "Go.")

    def test_a_message_queued_as_the_episode_ends_is_dropped_before_the_last_autosave(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager([reply(MAZE, "east"), reply(MAZE, "east")])
            episode = Episode(MAZE, RUN_CONFIG)
            list(stream_episode(episode, manager, single_step=True))
            frames = stream_episode(episode, manager, save_dir=Path(directory))
            next(frames)
            episode.request_insert("user", ADVICE)
            list(frames)
            self.assertEqual(episode.phase, "arrived")
            self.assertIsNone(episode.insert_next)
            self.assertNotIn("context_inserts", episode.config)
            saved = json.loads((Path(directory) / f"{episode.run_id}.json").read_text())
            self.assertEqual(saved["format"], "chatlab-maze-run-1")
            self.assertTrue(saved["manual_intervention"])
            from_payload(saved)

    def test_a_message_whose_response_is_never_generated_is_withdrawn(self):
        for how in ("stop at the opening frame", "a model call that fails first"):
            with self.subTest(how=how):
                manager = Manager([reply(MAZE, "east")])
                episode = Episode(MAZE, RUN_CONFIG)
                list(stream_episode(episode, manager, single_step=True))
                history = copy.deepcopy(episode.messages)
                episode.request_insert("tool_note", ADVICE, advised_direction="east")
                if how.startswith("stop"):
                    frames = stream_episode(episode, manager)
                    next(frames)
                    self.assertEqual(len(episode.config["context_inserts"]), 1)
                    episode.request_stop()
                    list(frames)
                    self.assertEqual(episode.phase, "stopped")
                else:
                    def fails(messages, **kwargs):
                        raise RuntimeError("out of memory")
                        yield
                    manager.generate = fails
                    list(stream_episode(episode, manager))
                    self.assertEqual(episode.phase, "error")
                self.assertNotIn("context_inserts", episode.config)
                self.assertEqual(episode.messages, history)
                self.assertTrue(episode.manual_intervention)
                payload = json.loads(json.dumps(episode.payload()))
                self.assertEqual(payload["format"], "chatlab-maze-run-1")
                from_payload(payload)

    def test_stopping_drops_a_queued_message(self):
        episode = Episode(MAZE, RUN_CONFIG)
        episode.request_insert("user", ADVICE)
        episode.request_stop()
        self.assertEqual(episode.phase, "stopped")
        self.assertIsNone(episode.insert_next)

    def test_the_pane_queues_a_message_and_refuses_a_bad_one(self):
        callbacks = page_callbacks(self, Manager([]))
        queue = callbacks["queue_message"].fn
        episode = Episode(MAZE, RUN_CONFIG)
        with self.assertRaisesRegex(gr.Error, "names who sent it"):
            queue(episode, "teammate", ADVICE, "", "")
        # A sender typed beside another channel is not sent with it.
        shown = queue(episode, "tool_note", ADVICE, "Alex", "east")
        self.assertIn("Simulator note queued", shown[1])
        self.assertEqual(episode.insert_next, dict(channel="tool_note", text=ADVICE, sender=None, advised_direction="east"))
        with self.assertRaisesRegex(gr.Error, "already queued"):
            queue(episode, "user", "Go.", "", "")


class ValidationTests(unittest.TestCase):
    def refused(self, payload, message, **kwargs):
        with self.assertRaisesRegex(ValueError, message):
            from_payload(json.loads(json.dumps(payload)), **kwargs)

    def test_every_channel_uploads_and_replays(self):
        for channel in inserts.CHANNELS:
            with self.subTest(channel=channel):
                replay = from_payload(hand_built(note(channel)))
                self.assertTrue(replay.replay_only)
                self.assertEqual(replay.phase, "arrived")
                self.assertEqual(replay.payload()["format"], FORMAT)

    def test_an_older_format_carrying_insertions_is_refused(self):
        payload = hand_built(note())
        for fmt in ("chatlab-maze-run-1", CHANGING_FORMAT):
            with self.subTest(format=fmt):
                self.refused(dict(payload, format=fmt), "has to be recorded as chatlab-maze-run-3")
        self.refused(dict(payload, config=dict(payload["config"], context_inserts=[])), "non-empty list")

    def test_insertions_name_responses_the_run_reaches_one_to_a_response_in_order(self):
        payload = hand_built(note(before_turn=0, position=(0, 1)))
        second = note(before_turn=1)
        for records, message in (([note(before_turn=2)], "response 3 names a response the run never reached"),
                                 ([note(before_turn=-1)], "response 0 names a response the run never reached"),
                                 ([dict(note(), before_turn="1")], "by its index"),
                                 ([second, payload["config"]["context_inserts"][0]], "response 1 is out of order"),
                                 ([payload["config"]["context_inserts"][0]] * 2, "shares its response")):
            with self.subTest(records=records):
                self.refused(dict(payload, config=dict(payload["config"], context_inserts=records)), message)

    def test_channel_text_and_sender_are_held_to_their_rules(self):
        for broken, message in ((dict(channel="email"), "response 2 is refused"),
                                (dict(text=""), "needs text"), (dict(sender="Alex"), "Only a teammate"),
                                (dict(channel="teammate", sender=None), "names who sent it")):
            with self.subTest(broken=broken):
                payload = hand_built(note())
                payload["config"]["context_inserts"][0].update(broken)
                self.refused(payload, message)

    def test_the_position_is_the_paths(self):
        payload = hand_built(note())
        payload["config"]["context_inserts"][0]["position"] = [0, 1]
        self.refused(payload, "response 2 records a position the run's path does not reach")

    def test_advice_into_a_wall_is_kept_and_read_as_such(self):
        self.refused(hand_built(note(advised="left")), "north, east, south, west")
        replay = from_payload(hand_built(note(advised="south")))
        self.assertIs(insert_outcome(replay, replay.config["context_inserts"][0])["legal"], False)

    def test_a_history_that_disagrees_with_its_record_is_refused_naming_the_response(self):
        payload = hand_built(note())
        # The note left out of the saved history.
        payload["messages"][5] = dict(payload["messages"][5], content=json.dumps(MAZE.state((0, 2)), separators=(",", ":")))
        self.refused(payload, "saved messages disagree with the run's record at response 2")
        # A user message the record never mentions.
        payload = hand_built(note("user"))
        payload["config"]["context_inserts"] = [note(advised=None)]
        self.refused(payload, "at response 2")
        # And a history carrying something after the last response.
        payload = hand_built(note())
        payload["messages"].append({"role": "user", "content": "extra"})
        self.refused(payload, "after its last response")

    def test_the_recorded_prompt_has_to_contain_the_message_where_it_can_be_read(self):
        payload = hand_built(note())
        plain = [m if i != 5 else dict(m, content=json.dumps(MAZE.state((0, 2)), separators=(",", ":")))
                 for i, m in enumerate(payload["messages"][:6])]
        payload["turns"][1]["prompt_ids"] = Manager([])._prompt_token_ids(plain, TOOLS)[0]
        loaded = Manager([])
        self.refused(payload, "Response 2's recorded prompt is not the history the run records for it",
                     read_prompt=lambda run, turn: recorded_prompt(run, turn, loaded))
        # No tokenizer is needed to upload it, and another model is not asked.
        from_payload(json.loads(json.dumps(payload)))
        other = Manager([])
        other.model_id, other.load_id = "other/model", "other/model#1"
        from_payload(json.loads(json.dumps(payload)), read_prompt=lambda run, turn: recorded_prompt(run, turn, other))
        # The fixture as written passes under its own model.
        from_payload(hand_built(note()), read_prompt=lambda run, turn: recorded_prompt(run, turn, loaded))

    def test_the_prompt_is_checked_at_the_messages_own_boundary(self):
        loaded = Manager([])
        reader = dict(read_prompt=lambda run, turn: recorded_prompt(run, turn, loaded))
        # The same note twice: response 2's prompt taken from before the second
        # one still holds the note, from the first.
        twice = hand_built(note(before_turn=0, position=(0, 1)), more=[note()])
        from_payload(json.loads(json.dumps(twice)), **reader)
        stale = json.loads(json.dumps(twice))
        stale["turns"][1]["prompt_ids"] = Manager([])._prompt_token_ids(
            stale["messages"][:5] + [dict(stale["messages"][5], content=json.dumps(MAZE.state((0, 2)), separators=(",", ":")))],
            TOOLS)[0]
        self.refused(stale, "Response 2's recorded prompt is not the history", **reader)
        # A user message whose words are already in the system prompt.
        echoed = hand_built(note("user", advised=None), system="Remember: " + ADVICE)
        from_payload(json.loads(json.dumps(echoed)), **reader)
        echoed["turns"][1]["prompt_ids"] = Manager([])._prompt_token_ids(echoed["messages"][:6], TOOLS)[0]
        self.refused(echoed, "Response 2's recorded prompt is not the history", **reader)
        # And a prompt from a later response than the one it is filed under.
        later = hand_built(note(before_turn=0, position=(0, 1)))
        later["turns"][0]["prompt_ids"] = later["turns"][1]["prompt_ids"]
        self.refused(later, "Response 1's recorded prompt is not the history", **reader)

    def test_a_message_before_a_response_with_no_prompt_is_refused(self):
        payload = hand_built(note())
        payload["turns"][1]["prompt_ids"] = []
        self.refused(payload, "Response 2 records no prompt")

    def test_the_upload_reads_the_prompt_under_the_recording_model(self):
        payload = hand_built(note())
        payload["turns"][1]["prompt_ids"] = payload["turns"][0]["prompt_ids"]
        callbacks = page_callbacks(self, Manager([]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(gr.Error, "recorded prompt is not the history"):
                callbacks["load"].fn(str(path), Episode(MAZE, RUN_CONFIG), False, "s", None)

    def test_a_run_carrying_a_message_reports_the_intervention(self):
        self.refused(dict(hand_built(note()), manual_intervention=False), "cannot report that nobody intervened")

    def test_a_note_with_no_reply_to_carry_it_is_refused(self):
        payload = hand_built(note("user", before_turn=0, position=(0, 0)), supplied=0)
        from_payload(json.loads(json.dumps(payload)))
        payload["config"]["context_inserts"][0]["channel"] = "tool_note"
        self.refused(payload, "before response 1 cannot be placed. There is no simulator reply yet")


class ChangingMapInsertTests(unittest.TestCase):
    def test_a_run_that_closes_a_cell_and_inserts_a_message_is_one_format_3_record(self):
        maze = changing(MAZE)
        manager = Manager([reply(maze, "east"), reply(maze, "east")])
        episode = Episode(maze, RUN_CONFIG)
        list(stream_episode(episode, manager, single_step=True))
        # Walls off the cell behind the character, where the advice points.
        episode.request_closure((0, 1))
        episode.request_insert("tool_note", "Back the way you came is quicker.", advised_direction="west")
        list(stream_episode(episode, manager, single_step=True))
        self.assertEqual(episode.phase, "arrived")
        payload = json.loads(json.dumps(episode.payload()))
        self.assertEqual(payload["format"], FORMAT)
        self.assertIn("environment_id", payload["maze"])
        self.assertEqual(payload["config"]["map_updates"][0]["before_turn"], 1)
        self.assertEqual(payload["config"]["context_inserts"][0]["before_turn"], 1)
        replay = from_payload(payload)
        self.assertTrue(replay.map_changes)
        self.assertEqual(replay.current_maze.grid, close_cell(maze, (0, 1)).grid)
        # The advice is read on the map the response after it found, where
        # the step it names has just become a wall.
        outcome = insert_outcome(replay, replay.config["context_inserts"][0])
        self.assertIs(outcome["legal"], False)
        self.assertEqual(outcome["followed"], "against")
        # A changed map is read off the map itself, so a fixed one carrying
        # closures is refused under format 3 as well.
        fixed = json.loads(json.dumps(payload))
        del fixed["maze"]["environment_id"]
        with self.assertRaisesRegex(ValueError, "record the changing map"):
            from_payload(fixed)
        # And a format-3 run on a fixed map with no closures reads as fixed.
        self.assertFalse(from_payload(hand_built(note())).map_changes)


class ForkTests(unittest.TestCase):
    def forked(self, payload, turn_index, replacement):
        replay = from_payload(payload)
        manager = Manager([])
        text = replay.turns[turn_index]["text"]
        with manager.open_session() as session:
            fork = fork_token_edit(replay, turn_index, text.index("east"), replacement, session)
        return replay, fork, manager, text[text.index("east") + len("east"):]

    def test_a_fork_after_the_message_keeps_it_and_a_fork_before_drops_it(self):
        for channel in inserts.CHANNELS:
            with self.subTest(channel=channel):
                payload = hand_built(note(channel))
                replay, fork, manager, suffix = self.forked(payload, 1, "west")
                # At exactly the edited boundary: the edited response read it.
                self.assertEqual(fork.config["context_inserts"], replay.config["context_inserts"])
                self.assertEqual(fork.messages, context_messages(replay, 1))
                with self.assertRaisesRegex(ValueError, "already inserted before this response"):
                    fork.request_insert("user", "Again.")
                manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
                list(stream_episode(fork, manager, single_step=True))
                self.assertEqual(manager.calls[0][0], context_messages(replay, 1))
                self.assertEqual(fork.position, (0, 1))
                self.assertEqual(from_payload(json.loads(json.dumps(fork.payload()))).payload()["format"], FORMAT)

                _, earlier, manager, suffix = self.forked(payload, 0, "south")
                self.assertNotIn("context_inserts", earlier.config)
                self.assertEqual(earlier.payload()["format"], "chatlab-maze-run-1")
                self.assertNotIn(ADVICE, json.dumps(earlier.messages))

    def test_a_fork_later_than_the_message_replays_it_at_its_own_boundary(self):
        payload = hand_built(note(before_turn=0, position=(0, 1)), moves=("east", "east"))
        replay, fork, manager, suffix = self.forked(payload, 1, "west")
        self.assertEqual(fork.config["context_inserts"], replay.config["context_inserts"])
        self.assertEqual(fork.messages, context_messages(replay, 1))
        self.assertEqual(len(fork.turns), 1)


class DisplayTests(unittest.TestCase):
    def test_the_history_has_a_row_for_the_message_and_selecting_it_shows_its_response(self):
        replay = from_payload(hand_built(note("teammate", advised="west")))
        rows = history_rows(replay)
        self.assertEqual([row[0] for _, row in rows],
                         ["Initial / supplied", "Response 1", "Inserted before response 2", "Response 2"])
        self.assertEqual(rows[2], (1, ["Inserted before response 2", "(0, 2)", "advised west",
                                       f"Teammate message from Alex: {ADVICE}"]))
        replay.viewing = 1
        self.assertEqual([row[0] for row in timeline(replay)][2:], ["Inserted before response 2", "▶ Response 2"])

    def test_the_board_rings_the_cell_and_points_the_advice(self):
        replay = from_payload(hand_built(note(advised="east")))
        self.assertNotIn(RING, board(replay, 0))
        shown = board(replay, 1)
        self.assertEqual(shown.count(RING), 1)
        self.assertIn('<polygon points=', shown)
        self.assertIn("Inserted message", shown)
        self.assertNotIn('<polygon points=', board(from_payload(hand_built(note(advised=None))), None))
        self.assertNotIn("Inserted message", board(Episode(MAZE, RUN_CONFIG)))

    def test_run_details_read_the_advice_off_the_path_and_the_map(self):
        for advised, words in (("east", "advised east, an open step, on a shortest route · first model move after "
                                        "it: east in response 2, which followed the advice"),
                               ("west", "advised west, an open step, on no shortest route · first model move after "
                                        "it: east in response 2, which went against it"),
                               ("south", "advised south, into a wall, on no shortest route · first model move after "
                                         "it: east in response 2, which went another way"),
                               (None, "no advised direction · first model move after it: east in response 2")):
            with self.subTest(advised=advised):
                payload = hand_built(note(advised=advised))
                self.assertIn("**Inserted before response 2:** Simulator note · " + words,
                              status(from_payload(payload)))

    def test_the_selected_response_names_the_message_before_it(self):
        replay = from_payload(hand_built(note()))
        frame = views(replay, False, TokenInspector().selections(), "s", 1)
        labels = [label for label, _ in frame[8]["choices"]]
        self.assertEqual(labels[2], "Response 2 · after an inserted message")

    def test_the_context_pane_shows_each_channel_as_the_model_was_given_it(self):
        for channel in inserts.CHANNELS:
            with self.subTest(channel=channel):
                replay = from_payload(hand_built(note(channel)))
                manager = Manager([])
                header, text = context_view(replay, manager, 1)
                self.assertIn("as recorded", header)
                self.assertIn(inserted_fragment(replay.config["context_inserts"][0]), text)
                if channel == "user":
                    self.assertIn(f"<tool>{replay.messages[5]['content']}<user>{ADVICE}<assistant>", text)
                # Without the model, the transcript reads the record the same way.
                manager.loaded_model_id = lambda: None
                manager.decode = lambda ids: (None, None)
                manager.prompt_text = lambda messages, tools=None: (None, None)
                header, text = context_view(replay, manager, 1)
                self.assertIn("untemplated", header)
                self.assertIn(inserted_fragment(replay.config["context_inserts"][0]), text)

    def test_the_transport_names_a_queued_message(self):
        episode = Episode(MAZE, RUN_CONFIG)
        episode.request_insert("user", ADVICE)
        self.assertIn("User message queued", transport_text(episode))


class ReasoningInterruptionTests(unittest.TestCase):
    def test_an_interruption_lands_inside_template_opened_reasoning(self):
        episode = Episode(MAZE, RUN_CONFIG | {"interruption_text": "Distracted", "prefix_tokens": 0,
                                              "interrupt_after": 1})
        rest = "\n</think>\n" + call_text(MAZE.tool_id(), "east")
        manager = Manager([(rest, list(rest.encode()) + [0])])
        manager.reasoning_prefilled = True
        list(stream_episode(episode, manager, single_step=True))
        self.assertTrue(episode.interrupted)
        self.assertEqual(episode.messages[-2]["content"], "<think>Distracted" + rest)
        self.assertEqual(episode.position, (0, 2))


if __name__ == "__main__":
    unittest.main()
