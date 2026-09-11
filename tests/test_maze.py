import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from extensions.maze_experiments.maze import Maze, apply_call, call_text, generate, parse_call
from extensions.maze_experiments.runner import Episode, TERMINAL, fork_token_edit, from_payload, stream_episode
from model_runtime import ModelManager
from extension_api import ModelService
from extension_api import TokenInspector
from token_metrics import unscored_metric
from extensions.maze_experiments.page import build_page, export_run, views
import gradio as gr

CONFIG = dict(supplied_moves=0, interrupt_after=0, interruption_text="Distracted", prefix_tokens=2,
              temperature=.7, sampling_seed=99, per_turn_tokens=100, token_budget=300, attempt_budget=10)
MAZE = Maze(("...", "##.", "..."), (0, 0), (0, 2))


class Manager:
    loaded = True
    model_id = "test/model"
    load_id = "test-load"
    reasoning_prefilled = False
    tokenizer = SimpleNamespace(encode=lambda s, **kw: list(s.encode()), decode=lambda ids, **kw: bytes(ids).decode())

    def __init__(self, replies):
        self.replies = iter(replies)
        self.busy = False
        self.calls = []

    def open_session(self):
        return ModelService(lambda: self).open_session()

    def reserve_generation(self):
        if self.busy:
            return False
        self.busy = True
        return True

    def release_generation(self):
        self.busy = False

    def _stop_token_ids(self):
        return {0}

    def hidden_token_ids(self):
        # The stop token is a special: generate() leaves it out of the response
        # text the way the runtime's decoder does, and records it in metrics.
        return {0}

    def encode_replacement(self, kept_ids, text, **kwargs):
        # This fixture encodes independent UTF-8 bytes; context-sensitive
        # behavior is exercised separately through the real runtime encoder.
        return list(text.encode())

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text, ids = next(self.replies)
        prefix = kwargs["forced_ids"]
        metrics = [{"token_id": t} for t in prefix + ids]
        yield SimpleNamespace(text=self.tokenizer.decode(prefix) + text, metrics=metrics, prompt_ids=[10, 20],
                              forced_prefix_tokens=len(prefix), reasoning_prefilled=self.reasoning_prefilled,
                              load_id=self.load_id, model_id=self.model_id)


def scored(generate):
    """Add the display metrics and alternatives the token strip and edit panel read."""
    def reply(*args, **kwargs):
        for frame in generate(*args, **kwargs):
            for position, metric in enumerate(frame.metrics):
                metric.update(unscored_metric(position=position, token_id=metric['token_id'],
                                              token_text=chr(metric['token_id']), fallback_text='',
                                              segment='response').to_dict())
                metric['top_candidates'] = [{'token_id': 120, 'text': 'x', 'probability': .1}]
            yield frame
    return reply


class MazeTests(unittest.TestCase):
    def test_typed_edit_preserves_sentencepiece_boundary_and_literal_prefix(self):
        from test_streaming import sentencepiece_manager, SP_HELLO, SP_SPACE_WORLD, SP_WORLD
        for text, expected_id in [('world', SP_WORLD), (' world', SP_SPACE_WORLD)]:
            with self.subTest(replacement=text):
                manager = sentencepiece_manager()
                ep = Episode(MAZE, CONFIG)
                ep.model_id, ep.load_id = manager.model_id, manager.load_id
                ep.turns = [dict(text='Hello world', forced_prefix_tokens=1, literal_prefill_tokens=1,
                                 metrics=[{'token_id': SP_HELLO}, {'token_id': SP_SPACE_WORLD}])]
                with ModelService(lambda: manager).open_session() as session:
                    with mock.patch.object(manager, 'encode_replacement', wraps=manager.encode_replacement) as encode:
                        edited = fork_token_edit(ep, 0, 1, text, session)
                    encode.assert_called_once_with([SP_HELLO], text, literal_prefill_tokens=1,
                                                   load_id=ep.load_id)
                    self.assertEqual(edited.pending_edit['forced_ids'], [SP_HELLO, expected_id])
                    self.assertEqual(session.decode(edited.pending_edit['forced_ids']), 'Hello' + text)
                    self.assertEqual(edited.token_edit['replacement_text'], text)

    def test_fork_refuses_a_later_load_that_moved_a_word_boundary(self):
        """Every ID still decodes alone to the characters it recorded; only the response changed.

        A SentencePiece revision that respells an ID from the word-boundary
        piece "▁world" to "world" leaves both vocabularies decoding it alone as
        "world", so a token at a time the run looks intact. After "Hello" the
        same ID reads " world" under one and "world" under the other, which is
        the retained prefix silently changing under the fork.
        """
        from test_streaming import SP_HELLO, SP_SPACE_WORLD, SP_WORLD, SP_PIECES, sentencepiece_manager
        recorded = ['Hello', 'world', '!']
        bang = SP_PIECES.index('!')
        revised_pieces = [piece if index != SP_SPACE_WORLD else 'world'
                          for index, piece in enumerate(SP_PIECES)]

        def uploaded():
            episode = Episode(MAZE, CONFIG)
            episode.model_id, episode.load_id = 'fake/model', 'session-1-load-1'
            episode.replay_only = True
            episode.turns = [dict(text='Hello world!', forced_prefix_tokens=0, literal_prefill_tokens=0,
                                  finish_reason='stop', metrics=[
                                      dict(token_id=token_id, text=text)
                                      for token_id, text in zip([SP_HELLO, SP_SPACE_WORLD, bang], recorded)])]
            return episode

        intact, revised = sentencepiece_manager(), sentencepiece_manager(revised_pieces)
        for manager in (intact, revised):
            self.assertEqual([manager.tokenizer.decode([token_id])
                              for token_id in (SP_HELLO, SP_SPACE_WORLD, bang)], recorded)
        self.assertEqual(intact.tokenizer.decode([SP_HELLO, SP_SPACE_WORLD]), 'Hello world')
        self.assertEqual(revised.tokenizer.decode([SP_HELLO, SP_SPACE_WORLD]), 'Helloworld')

        with ModelService(lambda: intact).open_session() as session:
            forked = fork_token_edit(uploaded(), 0, 2, 'world', session)
        self.assertEqual(forked.pending_edit['forced_ids'], [SP_HELLO, SP_SPACE_WORLD, SP_WORLD])

        # The moved ID is retained before the edit, and again past it, where the
        # fork replays nothing: the response it belongs to is evidence either way.
        for token_index in (2, 1):
            with self.subTest(token_index=token_index):
                with ModelService(lambda: revised).open_session() as session:
                    with self.assertRaisesRegex(ValueError, 'tokenize differently'):
                        fork_token_edit(uploaded(), 0, token_index, 'world', session)

    def test_edit_ui_callbacks_select_regenerate_archive_and_reject_stale_token(self):
        manager = Manager([('abc', [97, 98, 99, 0]), ('yz', [121, 122, 0])])
        manager.generate = scored(manager.generate)
        inspector = TokenInspector()
        selections = inspector.selections()
        inspector.selections = lambda: selections
        session_id = selections.new_session()
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        list(stream_episode(ep, manager))
        original = copy.deepcopy(ep.payload())
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(tokens=inspector, models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
                metrics = views(ep, False, selections, session_id)[7]
                selected = callbacks['select_token'].fn(ep, session_id, metrics, SimpleNamespace(index=1))
                self.assertEqual(selected[3], 'b')
                self.assertIn(("'x' · token 120", '120'), selected[4]['choices'])
                edit = callbacks['edit_token']
                frames = list(edit.fn(ep, False, session_id, metrics, selected[2], 'ignored', '120'))
                self.assertTrue(all(len(frame) == len(edit.outputs) for frame in frames))
                result = frames[-1][0]
                self.assertEqual(result.turns[0]['text'], 'axyz')
                self.assertEqual(result.phase, 'abandoned')
                self.assertEqual(json.loads((Path(directory) / f'{ep.run_id}.json').read_text()),
                                 json.loads(json.dumps(original)))
                self.assertTrue((Path(directory) / f'{result.run_id}.json').exists())
                self.assertEqual(ep.payload(), original)
                with self.assertRaisesRegex(gr.Error, 'current response'):
                    list(edit.fn(ep, False, session_id, metrics, selected[2], 'x', 'text'))
            finally:
                demo.close()

    def test_token_edit_rewinds_history_and_moves_and_preserves_original(self):
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        manager = Manager([(move, ids), (move, ids)])
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        before = copy.deepcopy(ep.payload())
        index = move.index('east')
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 1, index, "west", session)
        self.assertEqual(edited.position, (0, 1))
        self.assertEqual(len(edited.turns), 1)
        self.assertEqual(len(edited.events), 1)
        self.assertEqual(edited.tool_attempts, 1)
        self.assertEqual(edited.sampled_tokens, len(ids))
        self.assertNotEqual(edited.run_id, ep.run_id)
        self.assertEqual(edited.token_edit["parent_run_id"], ep.run_id)
        self.assertTrue(edited.manual_intervention)
        self.assertEqual(edited.pending_edit["forced_ids"], ids[:index] + list(b'west'))
        suffix = move[index + len('east'):]
        manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
        list(stream_episode(edited, manager, single_step=True))
        self.assertEqual(edited.phase, "paused")
        self.assertEqual(edited.position, MAZE.start)
        self.assertEqual(edited.events[-1]["direction"], "west")
        self.assertEqual(edited.messages[-2]["content"], move.replace('east', 'west'))
        self.assertEqual(manager.calls[-1][0], ep.messages[:4])
        self.assertEqual(manager.calls[-1][1]["literal_prefill_tokens"], 0)
        self.assertEqual(edited.sampled_tokens, len(ids) + len(suffix) + 1)
        self.assertEqual(ep.payload(), before)
        replay = from_payload(copy.deepcopy(edited.payload()))
        self.assertEqual(replay.token_edit, edited.token_edit)
        self.assertEqual(replay.position, MAZE.start)

    def test_token_edit_can_choose_exact_candidate_and_regenerate_stopped_response(self):
        manager = Manager([('abc', [97, 98, 99])])
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        list(stream_episode(ep, manager))
        ep.turns[0]["metrics"][1]["top_candidates"] = [{"token_id": 120, "text": "x"}]
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 0, 1, "ignored", session, candidate_id=120)
        self.assertEqual(edited.pending_edit["forced_ids"], [97, 120])
        manager.replies = iter([('yz', [121, 122, 0])])
        list(stream_episode(edited, manager))
        self.assertEqual(edited.turns[0]["text"], 'axyz')
        self.assertEqual(edited.phase, 'abandoned')
        self.assertFalse(edited.interrupted)
        self.assertEqual(edited.sampled_tokens, 3)

    def test_token_edit_restores_interruption_and_recovery_at_rewind_point(self):
        move = '\n' + call_text(MAZE.maze_id, "east")
        manager = Manager([(move, list(move.encode()) + [0])] * 2)
        ep = Episode(MAZE, CONFIG | {"per_turn_tokens": 200, "token_budget": 1000})
        list(stream_episode(ep, manager))
        with manager.open_session() as session:
            first = fork_token_edit(ep, 0, 2, '\n', session)
            second = fork_token_edit(ep, 1, 0, '\n', session)
        self.assertFalse(first.interrupted)
        self.assertTrue(first.pending_edit["interruption_here"])
        self.assertEqual(first.pending_edit["literal_prefill_tokens"], 2)
        self.assertTrue(second.interrupted)
        self.assertTrue(second.resumed)
        self.assertEqual(second.latency, ep.latency)
        self.assertEqual(second.intervention_turn, 0)
        manager.replies = iter([(move, list(move.encode()) + [0])])
        list(stream_episode(first, manager, single_step=True))
        self.assertTrue(first.interrupted)
        self.assertTrue(first.resumed)
        self.assertEqual(first.intervention_turn, 0)

    def test_token_edit_rejects_busy_replay_changed_model_and_invalid_selection(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('abc', [97, 98, 99, 0])])
        list(stream_episode(ep, manager))
        with manager.open_session() as session:
            for field, value, message in [('busy', True, 'Pause'),
                                          ('model_id', 'other/model', 'Load that model')]:
                previous = getattr(ep, field)
                setattr(ep, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    fork_token_edit(ep, 0, 2, 'x', session)
                setattr(ep, field, previous)
            for turn, token in [(-1, 2), (1, 2), (0, -1), (0, 0), (0, 99)]:
                with self.assertRaises(ValueError):
                    fork_token_edit(ep, turn, token, 'x', session)
            with self.assertRaisesRegex(ValueError, 'replacement text'):
                fork_token_edit(ep, 0, 2, '', session)
            with self.assertRaisesRegex(ValueError, 'alternative'):
                fork_token_edit(ep, 0, 2, '', session, candidate_id=123)

    def test_uploaded_replay_forks_into_a_live_run_under_a_later_load(self):
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        author = Manager([(move, ids)])
        author.generate = scored(author.generate)
        list(stream_episode(ep, author, single_step=True))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        self.assertTrue(replay.replay_only)

        # The same repository ID reloaded: a new load_id, the same tokenizer.
        reloaded = Manager([])
        reloaded.load_id = "test-load#2"
        index = move.index("east")
        with reloaded.open_session() as session:
            forked = fork_token_edit(replay, 0, index, "west", session)
        self.assertFalse(forked.replay_only)
        self.assertEqual((forked.load_id, forked.model_id), ("test-load#2", "test/model"))
        self.assertEqual(forked.token_edit["parent_run_id"], ep.run_id)
        self.assertEqual(forked.token_edit["parent_load_id"], "test-load")
        self.assertTrue(forked.token_edit["parent_replay"])
        self.assertEqual(forked.pending_edit["forced_ids"], ids[:index] + list(b"west"))
        self.assertEqual(forked.turns, [])
        self.assertEqual(forked.position, MAZE.start)

        suffix = move[index + len("east"):]
        reloaded.replies = iter([(suffix, list(suffix.encode()) + [0])])
        list(stream_episode(forked, reloaded, single_step=True))
        self.assertEqual(forked.phase, "paused")
        self.assertEqual(forked.turns[0]["text"], move.replace("east", "west"))
        self.assertEqual(forked.events[-1]["error"], "blocked_move")
        self.assertEqual(forked.position, MAZE.start)
        self.assertTrue(replay.replay_only)
        self.assertEqual(json.loads(json.dumps(replay.payload())), json.loads(json.dumps(ep.payload())))

    def test_fork_refuses_a_later_load_whose_tokenizer_decodes_differently(self):
        """The same model ID re-downloaded at another revision must not replay stored IDs."""
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        author = Manager([(move, ids)])
        author.generate = scored(author.generate)
        list(stream_episode(ep, author, single_step=True))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        index = move.index("east")

        # Same model ID, new load, a vocabulary that maps those IDs elsewhere.
        revised = Manager([])
        revised.load_id = "test-load#2"
        revised.tokenizer = SimpleNamespace(
            encode=lambda s, **kw: [(b + 1) % 128 for b in s.encode()],
            decode=lambda ids, **kw: bytes((i + 1) % 128 for i in ids).decode())
        with revised.open_session() as session:
            with self.assertRaises(ValueError) as caught:
                fork_token_edit(replay, 0, index, "west", session)
        self.assertIn("tokenize differently", str(caught.exception))

        # An ID beyond the new vocabulary raises rather than returning text.
        vocabulary = list(range(64))
        out_of_range = Manager([])
        out_of_range.load_id = "test-load#3"
        out_of_range.tokenizer = SimpleNamespace(
            encode=lambda s, **kw: list(s.encode()),
            decode=lambda ids, **kw: bytes(vocabulary[i] for i in ids).decode())
        with out_of_range.open_session() as session:
            with self.assertRaises(ValueError) as caught:
                fork_token_edit(replay, 0, index, "west", session)
        self.assertIn("cannot decode", str(caught.exception))

        # The comparison reads the text of the response as a whole, which every
        # export has always carried, so one predating per-token text is checked
        # like any other rather than refused for want of evidence.
        for metric in replay.turns[0]["metrics"]:
            metric.pop("text")
        with revised.open_session() as session:
            with self.assertRaises(ValueError) as caught:
                fork_token_edit(replay, 0, index, "west", session)
        self.assertIn("tokenize differently", str(caught.exception))
        reloaded = Manager([])
        reloaded.load_id = "test-load#4"
        with reloaded.open_session() as session:
            forked = fork_token_edit(replay, 0, index, "west", session)
        self.assertEqual(forked.pending_edit["forced_ids"], ids[:index] + list(b"west"))

        # A restart gives the first load of the same repository the load ID the
        # previous session's first load carried, which proves nothing about it.
        restarted = Manager([])
        restarted.load_id = replay.load_id
        restarted.tokenizer = revised.tokenizer
        with restarted.open_session() as session:
            with self.assertRaises(ValueError) as caught:
                fork_token_edit(replay, 0, index, "west", session)
        self.assertIn("tokenize differently", str(caught.exception))

    def test_fork_refuses_a_later_load_that_changed_an_earlier_response(self):
        """Earlier turns are rebuilt from their stored IDs, so they are verified too."""
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        author = Manager([(move, ids), (move, ids)])
        author.generate = scored(author.generate)
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        list(stream_episode(ep, author))
        self.assertEqual([turn["finish_reason"] for turn in ep.turns], ["stop", "stop"])
        index = move.index("east")

        # Every stored ID still decodes the same, but the ID the first response
        # ended on is no longer configured as a stop token, so replaying that
        # response would read it as a length failure and skip its movement.
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        restopped = Manager([])
        restopped.load_id = "test-load#2"
        restopped._stop_token_ids = lambda: {1}
        with restopped.open_session() as session:
            with self.assertRaisesRegex(ValueError, "stop tokens differ"):
                fork_token_edit(replay, 1, index, "west", session)

        # An earlier response's own tokens no longer decode to what it recorded,
        # here written into the run the way a changed vocabulary would read.
        drifted = from_payload(json.loads(json.dumps(ep.payload())))
        drifted.turns[0]["text"] = "¡" + drifted.turns[0]["text"]
        reloaded = Manager([])
        reloaded.load_id = "test-load#2"
        with reloaded.open_session() as session:
            with self.assertRaisesRegex(ValueError, "tokenize differently"):
                fork_token_edit(drifted, 1, index, "west", session)

        # Unchanged, the same later load forks the second response as before.
        intact = from_payload(json.loads(json.dumps(ep.payload())))
        with reloaded.open_session() as session:
            forked = fork_token_edit(intact, 1, index, "west", session)
        self.assertEqual(forked.position, (0, 1))
        self.assertEqual(len(forked.turns), 1)

    def test_fork_refuses_a_retained_prefix_the_loaded_model_would_stop_inside(self):
        """forced_ids is cut at its first stop token, so one inside the kept prefix ends the response early."""
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        author = Manager([(move, ids)])
        author.generate = scored(author.generate)
        list(stream_episode(ep, author, single_step=True))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        index = move.index("east")
        self.assertIn(10, ids[:index])

        # Every ID still decodes the same and the response still ends on a stop
        # token, but the newline inside the retained prefix is now one too.
        restopped = Manager([])
        restopped.load_id = "test-load#2"
        restopped._stop_token_ids = lambda: {0, 10}
        with restopped.open_session() as session:
            with self.assertRaisesRegex(ValueError, "retained prefix"):
                fork_token_edit(replay, 0, index, "west", session)

        # Editing before that token leaves it out of the prefix, so it is fine.
        with restopped.open_session() as session:
            forked = fork_token_edit(replay, 0, 5, "west", session)
        self.assertEqual(forked.pending_edit["forced_ids"], ids[:5] + list(b"west"))

    def test_fork_refuses_a_recorded_alternative_outside_the_session_that_offered_it(self):
        """The chosen candidate is the one replayed ID no recorded response stands behind."""
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        author = Manager([(move, ids)])
        author.generate = scored(author.generate)
        list(stream_episode(ep, author, single_step=True))
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        index = move.index("east")
        self.assertNotIn(120, ids)

        # Every recorded ID still decodes to the text it recorded, and the
        # alternative is refused anyway: the run records how it decodes alone,
        # which is the same recording whichever way the offering vocabulary
        # spelled it, so the loaded one has nothing to be checked against.
        reloaded = Manager([])
        reloaded.load_id = "test-load#2"
        with reloaded.open_session() as session:
            with self.assertRaisesRegex(ValueError, "Replacement text"):
                fork_token_edit(replay, 0, index, "ignored", session, candidate_id=120)

        # Typing the same text is encoded and checked after the retained
        # tokens, so the branch the alternative offered stays reachable.
        with reloaded.open_session() as session:
            typed = fork_token_edit(replay, 0, index, "x", session)
        self.assertEqual(typed.pending_edit["forced_ids"], ids[:index] + [120])

        # A live run whose weights have been reloaded since is refused too: the
        # load that offered the alternative is no longer the one in memory.
        author.load_id = "test-load#2"
        with author.open_session() as session:
            with self.assertRaisesRegex(ValueError, "Replacement text"):
                fork_token_edit(ep, 0, index, "ignored", session, candidate_id=120)

        # The session that offered it names the load in memory now, which is
        # the one place the evidence the run lacks is not needed.
        author.load_id = ep.load_id
        with author.open_session() as session:
            forked = fork_token_edit(ep, 0, index, "ignored", session, candidate_id=120)
        self.assertEqual(forked.pending_edit["forced_ids"], ids[:index] + [120])

        # An ID the run never offered for this token is a wrong selection
        # whatever produced the run, so it is refused on its own terms.
        with author.open_session() as session:
            with self.assertRaisesRegex(ValueError, "Choose an alternative"):
                fork_token_edit(ep, 0, index, "ignored", session, candidate_id=121)

    def test_fork_refuses_an_alternative_a_respelled_vocabulary_would_reproduce(self):
        """An alternative is recorded by decoding it alone, which drops a word boundary.

        SentencePiece reads the word-boundary space off the first token of
        whatever it decodes, so "▁world" and "world" both record "world" on
        their own. A vocabulary that respells the offered ID from one to the
        other reproduces every recording the run carries, including the
        alternative decoded where the fork will put it, while the branch reads
        "Hello world" under the vocabulary that offered it and "Helloworld"
        under the loaded one. Nothing recorded separates the two, so the
        alternative is applied only in the session that offered it.
        """
        from test_streaming import (sentencepiece_manager, SP_HELLO, SP_SPACE_WORLD, SP_WORLD,
                                    SP_PIECES)
        bang = SP_PIECES.index('!')
        respelled_pieces = [piece if index != SP_SPACE_WORLD else 'world'
                            for index, piece in enumerate(SP_PIECES)]

        def episode_for(*, replay, load_id):
            episode = Episode(MAZE, CONFIG)
            episode.model_id, episode.load_id = 'fake/model', load_id
            episode.replay_only = replay
            episode.turns = [dict(text='Hello!', forced_prefix_tokens=0, literal_prefill_tokens=0,
                                  finish_reason='stop', metrics=[
                                      dict(token_id=SP_HELLO, text='Hello'),
                                      dict(token_id=bang, text='!',
                                           top_candidates=[dict(token_id=SP_SPACE_WORLD, text='world')])])]
            return episode

        offering, respelled = sentencepiece_manager(), sentencepiece_manager(respelled_pieces)

        # The response the run records reads the same under both, and so does
        # the alternative on its own, which is all the run records of it.
        for manager in (offering, respelled):
            self.assertEqual(manager.tokenizer.decode([SP_HELLO, bang]), 'Hello!')
            self.assertEqual(manager.tokenizer.decode([SP_SPACE_WORLD]), 'world')

        # Decoding the alternative where the fork will put it does not separate
        # them either. Under the respelled vocabulary what it adds after the
        # retained "Hello" is what it decodes to alone, so that comparison
        # agrees exactly where the branch is wrong, and under the vocabulary
        # that offered it the two differ, where the branch is right.
        self.assertEqual(respelled.tokenizer.decode([SP_HELLO, SP_SPACE_WORLD]), 'Helloworld')
        self.assertEqual(offering.tokenizer.decode([SP_HELLO, SP_SPACE_WORLD]), 'Hello world')

        for manager in (offering, respelled):
            with ModelService(lambda: manager).open_session() as session:
                with self.assertRaisesRegex(ValueError, 'Replacement text'):
                    fork_token_edit(episode_for(replay=True, load_id='session-1-load-1'),
                                    0, 1, 'ignored', session, candidate_id=SP_SPACE_WORLD)

        with ModelService(lambda: offering).open_session() as session:
            # The session that offered the alternative applies it.
            live = episode_for(replay=False, load_id=offering.load_id)
            forked = fork_token_edit(live, 0, 1, 'ignored', session, candidate_id=SP_SPACE_WORLD)
            self.assertEqual(forked.pending_edit['forced_ids'], [SP_HELLO, SP_SPACE_WORLD])

            # The uploaded run reaches either branch by typing it, and gets the
            # ID that spells what was typed after the retained "Hello".
            for text, expected_id in [(' world', SP_SPACE_WORLD), ('world', SP_WORLD)]:
                with self.subTest(replacement=text):
                    forked = fork_token_edit(episode_for(replay=True, load_id='session-1-load-1'),
                                             0, 1, text, session)
                    self.assertEqual(forked.pending_edit['forced_ids'], [SP_HELLO, expected_id])
                    self.assertEqual(session.decode(forked.pending_edit['forced_ids']), 'Hello' + text)

    def test_fork_reads_an_uploaded_byte_fragment_in_the_characters_it_completes(self):
        """A piece of a multi-byte character records the same text whatever ID carries it.

        On its own it names no ID, so a token at a time there is nothing to
        check. Read as part of the response it belongs to, the character its
        neighbours complete says which bytes it carried.
        """
        move = "café\n" + call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        author = Manager([(move, ids)])
        author.generate = scored(author.generate)
        list(stream_episode(ep, author, single_step=True))
        index = move.index("east")

        # The two halves of "é" record the replacement character, as the runtime
        # writes them: neither identifies the ID it came from.
        replay = from_payload(json.loads(json.dumps(ep.payload())))
        halves = [position for position, byte in enumerate(ids) if byte > 127]
        self.assertEqual(halves, [3, 4])
        for position in halves:
            replay.turns[0]["metrics"][position]["text"] = "�"
        reloaded = Manager([])
        reloaded.load_id = "test-load#2"
        with reloaded.open_session() as session:
            forked = fork_token_edit(replay, 0, index, "west", session)
        self.assertEqual(forked.pending_edit["forced_ids"], ids[:index] + list(b"west"))

        # A later load that moved those bytes writes another character into the
        # retained prefix, which the response it belongs to still catches.
        swapped = {0xC3: 0xC4, 0xC4: 0xC3}
        moved = Manager([])
        moved.load_id = "test-load#3"
        moved.tokenizer = SimpleNamespace(
            encode=lambda s, **kw: list(s.encode()),
            decode=lambda ids, **kw: bytes(swapped.get(i, i) for i in ids).decode())
        self.assertEqual(moved.tokenizer.decode(ids[3:5]), "ĩ")
        with moved.open_session() as session:
            with self.assertRaisesRegex(ValueError, "tokenize differently"):
                fork_token_edit(replay, 0, index, "west", session)

        # The same holds past the edited token, where the fork replays nothing:
        # a vocabulary that moved anywhere in the response is evidence enough.
        ahead = from_payload(json.loads(json.dumps(ep.payload())))
        with moved.open_session() as session:
            with self.assertRaisesRegex(ValueError, "tokenize differently"):
                fork_token_edit(ahead, 0, 1, "west", session)

    def test_fork_reads_a_supplied_prefix_special_the_way_replay_forces_it(self):
        """Supplied prefix tokens are replayed visibly, specials included, so the text keeps them.

        Everywhere else a special is dropped before the text is decoded, which
        is what makes the two positions of the same ID read differently.
        """
        manager = Manager([])
        manager.load_id = "test-load#2"
        ep = Episode(MAZE, CONFIG)
        ep.model_id, ep.load_id = "test/model", "test-load"
        ep.replay_only = True
        ep.turns = [dict(text="a\x00b", forced_prefix_tokens=2, literal_prefill_tokens=2, finish_reason="stop",
                         metrics=[{"token_id": token_id} for token_id in (97, 0, 98, 0)])]
        with manager.open_session() as session:
            forked = fork_token_edit(ep, 0, 2, "c", session)
        self.assertEqual(forked.pending_edit["forced_ids"], [97, 0, 99])

    def test_replay_edit_leaves_a_newer_archive_of_the_same_run_alone(self):
        move = call_text(MAZE.maze_id, "east")
        ids = list(move.encode()) + [0]
        manager = Manager([(move, ids), (move, ids)])
        manager.generate = scored(manager.generate)
        inspector = TokenInspector()
        selections = inspector.selections()
        inspector.selections = lambda: selections
        session_id = selections.new_session()
        ep = Episode(MAZE, CONFIG | {"interruption_text": "", "per_turn_tokens": 200, "token_budget": 1000})
        with tempfile.TemporaryDirectory() as directory:
            list(stream_episode(ep, manager, single_step=True, save_dir=Path(directory)))
            snapshot = json.loads(json.dumps(ep.payload()))
            list(stream_episode(ep, manager, save_dir=Path(directory)))
            self.assertEqual(ep.phase, "arrived")
            archive = Path(directory) / f"{ep.run_id}.json"
            completed = archive.read_bytes()

            replay = from_payload(snapshot)
            manager.load_id = "test-load#2"
            index = move.index("east")
            suffix = move[index + len("east"):]
            manager.replies = iter([(suffix, list(suffix.encode()) + [0])])
            context = SimpleNamespace(tokens=inspector, models=manager, data_dir=Path(directory),
                                      navigation=SimpleNamespace(open_models=lambda button: None))
            with gr.Blocks() as demo:
                build_page(context)
            try:
                callbacks = {fn.fn.__name__: fn for fn in demo.fns.values()}
                metrics = views(replay, False, selections, session_id)[7]
                selected = callbacks["select_token"].fn(replay, session_id, metrics, SimpleNamespace(index=index))
                frames = list(callbacks["edit_token"].fn(replay, False, session_id, metrics,
                                                         selected[2], "west", "text"))
                forked = frames[-1][0]
            finally:
                demo.close()
            self.assertFalse(forked.replay_only)
            self.assertEqual(forked.events[-1]["error"], "blocked_move")
            self.assertEqual(archive.read_bytes(), completed)
            self.assertTrue((Path(directory) / f"{forked.run_id}.json").exists())

    def test_edit_to_stop_token_finishes_without_executing_partial_action(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([('abc', [97, 98, 99, 0])])
        list(stream_episode(ep, manager))
        # The stop token offered as an alternative, applied by the live episode
        # that offered it.
        ep.turns[0]["metrics"][1]["top_candidates"] = [{"token_id": 0, "text": "\x00"}]
        with manager.open_session() as session:
            edited = fork_token_edit(ep, 0, 1, '', session, candidate_id=0)
        manager.replies = iter([('', [])])  # Runtime returns after consuming a forced stop token.
        list(stream_episode(edited, manager))
        self.assertEqual(edited.phase, 'abandoned')
        self.assertEqual(edited.turns[0]["finish_reason"], 'stop')
        self.assertEqual(edited.tool_attempts, 0)

    def test_goal_modes_cover_prompts_errors_supplied_moves_arrival_and_replay(self):
        hint = "The destination is in the top row."
        for mode in ("coordinates", "hidden", "hint"):
            for supplied in (0, 1):
                with self.subTest(mode=mode, supplied=supplied):
                    ep = Episode(MAZE, CONFIG | dict(goal_mode=mode, goal_hint=hint,
                                                    supplied_moves=supplied, interruption_text=""))
                    maze_id = ep.model_state()["maze_id"]
                    replies = [call_text(maze_id, "south"), call_text("wrong", "east"),
                               '<tool_call>{}</tool_call>'] + [call_text(maze_id, "east")] * (2 - supplied)
                    manager = Manager([(reply, [8, 0]) for reply in replies])
                    with mock.patch('extensions.maze_experiments.runner.time.sleep'):
                        list(stream_episode(ep, manager))
                    self.assertEqual(ep.phase, "arrived")
                    self.assertEqual(ep.position, MAZE.goal)
                    self.assertEqual([e['error'] for e in ep.events if not e['accepted']],
                                     ['blocked_move', 'wrong_maze', 'invalid_tool_schema'])
                    # Check the actual messages passed to generation as well as the final reply.
                    for history in [ep.messages, *(call[0] for call in manager.calls)]:
                        for message in history:
                            if message['role'] not in ('user', 'tool'):
                                continue
                            content = message['content']
                            state = json.loads(content.split('\n', 1)[1] if message['role'] == 'user' else content)
                            self.assertEqual(state['grid'], list(MAZE.grid))
                            self.assertEqual(state['maze_id'], maze_id)
                            self.assertEqual('destination' in state, mode == 'coordinates')
                            self.assertEqual('goal_hint' in state, mode == 'hint')
                            self.assertNotIn('progress', state)
                            if mode == 'coordinates':
                                self.assertEqual(state['destination'], list(MAZE.goal))
                            elif mode == 'hint':
                                self.assertEqual(state['goal_hint'], hint)
                    self.assertTrue(json.loads(ep.messages[-1]['content'])['arrived'])
                    replay = from_payload(json.loads(json.dumps(ep.payload())))
                    self.assertEqual(replay.config['goal_mode'], mode)
                    self.assertEqual(replay.config['goal_hint'], hint)
                    self.assertEqual(replay.messages, ep.messages)
                    self.assertEqual(replay.position, MAZE.goal)
                    self.assertTrue(replay.replay_only)

    def test_concealed_state_does_not_encode_destination_in_identifier(self):
        other = Maze(MAZE.grid, MAZE.start, (2, 0))
        for mode in ('hidden', 'hint'):
            options = dict(goal_mode=mode, goal_hint='The destination is on an outer row.')
            self.assertEqual(MAZE.state(MAZE.start, **options), other.state(other.start, **options))
        self.assertNotEqual(MAZE.maze_id, other.maze_id)

    def test_goal_mode_validation_and_legacy_replay(self):
        for options in ({'goal_mode': 'unknown'}, {'goal_mode': 'hint'},
                        {'goal_mode': 'hint', 'goal_hint': '  '}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Episode(MAZE, CONFIG | options)
        old = Episode(MAZE, CONFIG).payload()
        old['config'].pop('goal_mode')
        old['config'].pop('goal_hint')
        replay = from_payload(json.loads(json.dumps(old)))
        self.assertEqual(replay.config['goal_mode'], 'coordinates')
        self.assertEqual(replay.model_state()['destination'], list(MAZE.goal))

    def test_new_response_reset_and_replay_clear_selection_but_streaming_preserves_it(self):
        selections = TokenInspector().selections()
        session = selections.new_session()
        ep = Episode(MAZE, CONFIG)
        metric = dict(token_id=1, display_text='a', text='a', scored=False)
        turn = dict(metrics=[metric], text='a', forced_prefix_tokens=0, finish_reason='stop')
        ep.turns = [turn]
        first = views(ep, False, selections, session)
        self.assertEqual(first[-2:], ('Select a model-generated token above.', []))
        ep.turns[0]['metrics'].append(metric)
        streamed = views(ep, False, selections, session)
        self.assertEqual(streamed[-2:], (gr.skip(), gr.skip()))
        self.assertEqual(first[7][0], streamed[7][0])
        ep.turns.append(copy.deepcopy(turn))
        second = views(ep, False, selections, session)
        replay = views(ep, False, selections, session, index=0)
        reset = views(Episode(MAZE, CONFIG), False, selections, session)
        for frame in (second, replay, reset):
            self.assertEqual(frame[-2:], ('Select a model-generated token above.', []))
        self.assertEqual(len({frame[7][0] for frame in (first, second, replay, reset)}), 4)

    def test_browser_sessions_have_independent_episode_state(self):
        original = Episode(MAZE, CONFIG)
        original.created_at = 100.
        with mock.patch('extensions.maze_experiments.runner.time.time', return_value=200.):
            duplicate = copy.deepcopy(original)
        duplicate.request_stop()
        self.assertEqual(original.phase, "ready")
        self.assertNotEqual(original.run_id, duplicate.run_id)
        self.assertIsNot(original.lock, duplicate.lock)
        self.assertIsNot(original.messages, duplicate.messages)
        self.assertEqual(original.created_at, 100.)
        self.assertEqual(duplicate.created_at, 200.)
        original.phase = 'paused'
        resumed = copy.deepcopy(original)
        self.assertEqual(resumed.created_at, original.created_at)
        self.assertEqual(resumed.run_id, original.run_id)

    def test_seed_and_distance(self):
        a, b = generate(), generate()
        self.assertEqual(a, b)
        self.assertEqual(len(a.route()), 11)
        for p, q in zip(a.route(), a.route()[1:]):
            self.assertIn(q, a.neighbors(p).values())

    def test_no_actions_from_quotes_or_fake_state(self):
        text = call_text(MAZE.maze_id, "east")
        for value in ("I moved east.", "```\n" + text + "\n```", "\n".join("> " + line for line in text.splitlines())):
            self.assertEqual(parse_call(value), (None, None))
        args, error = parse_call(text)
        self.assertIsNone(error)
        self.assertEqual(apply_call(MAZE, MAZE.start, args)["after"], [0, 1])
        self.assertFalse(apply_call(MAZE, MAZE.start, {"maze_id": MAZE.maze_id, "direction": "south"})["accepted"])
        self.assertFalse(apply_call(MAZE, MAZE.start, {"maze_id": "fake", "direction": "east"})["accepted"])

    def test_invalid_argument_types_are_rejected(self):
        for value in ([], {}, None, 7):
            text = '<tool_call>\n' + json.dumps({"name": "move", "arguments": {"maze_id": MAZE.maze_id, "direction": value}}) + '\n</tool_call>'
            self.assertEqual(parse_call(text)[1], "invalid_arguments")

    def test_supplied_reasoning_markers_are_rejected_before_generation(self):
        for marker in ('<think>', '</think>'):
            for prefilled in (False, True):
                with self.subTest(marker=marker, reasoning_prefilled=prefilled):
                    ep = Episode(MAZE, CONFIG | {'interruption_text': marker, 'prefix_tokens': 0})
                    manager = Manager([])
                    manager.reasoning_prefilled = prefilled
                    list(stream_episode(ep, manager))
                    self.assertEqual(ep.phase, 'error')
                    self.assertIn('reasoning delimiters', ep.detail)
                    self.assertEqual(manager.calls, [])
                    self.assertFalse(manager.busy)
                    self.assertFalse(ep.interrupted)
                    self.assertIsNone(ep.resumed)

    def test_supplied_code_fences_are_rejected_before_they_can_hide_generated_calls(self):
        for fence in ('```', '~~~'):
            with self.subTest(fence=fence):
                # The normal parser must still ignore model-generated fenced examples.
                self.assertEqual(parse_call(fence + '\n' + call_text(MAZE.maze_id, 'east')), (None, None))
                ep = Episode(MAZE, CONFIG | {'interruption_text': fence + '\nExample', 'prefix_tokens': 0})
                manager = Manager([])
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, 'error')
                self.assertIn('code fences', ep.detail)
                self.assertEqual(manager.calls, [])
                self.assertFalse(ep.interrupted)
                self.assertFalse(manager.busy)

    def test_ordinary_interruption_prose_with_inline_backticks_remains_allowed(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': 'Discuss `inline text` and ~one tilde~.', 'prefix_tokens': 0})
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.phase, 'paused')
        self.assertEqual(ep.position, (0, 1))
        self.assertTrue(ep.interrupted)
        self.assertTrue(ep.resumed)

    def test_interruption_rejects_finished_runs_and_replays_without_changing_provenance(self):
        for phase, replay in [(phase, False) for phase in TERMINAL] + [('ready', True), ('paused', True), ('running', True)]:
            with self.subTest(phase=phase, replay=replay):
                ep = Episode(MAZE, CONFIG)
                ep.phase, ep.replay_only = phase, replay
                before = copy.deepcopy(ep.payload())
                with self.assertRaisesRegex(ValueError, 'Start a new episode'):
                    ep.request_interruption()
                self.assertEqual(ep.payload(), before)
                self.assertFalse(ep.interrupt_next)
        for phase in ('ready', 'paused', 'running'):
            ep = Episode(MAZE, CONFIG)
            ep.phase = phase
            ep.request_interruption()
            self.assertTrue(ep.interrupt_next)
            self.assertTrue(ep.manual_intervention)

    def test_return_then_arrival_excludes_supplied_tokens(self):
        ep = Episode(MAZE, CONFIG)
        move = '\n' + call_text(MAZE.maze_id, "east")
        manager = Manager([(move, [8, 0]), (move, [8, 0])])
        with tempfile.TemporaryDirectory() as d:
            list(stream_episode(ep, manager, save_dir=Path(d)))
            replay = from_payload(json.loads(ep.save(Path(d)).read_text()))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.latency, 2)
        self.assertEqual(ep.sampled_tokens, 4)
        self.assertTrue(ep.first_move_progress)
        self.assertFalse(manager.busy)
        self.assertEqual(manager.calls[0][1]["forced_ids"], [68, 105])
        self.assertTrue(manager.calls[0][1]["tools"])
        self.assertEqual(replay.position, MAZE.goal)
        self.assertTrue(replay.replay_only)
        bad = ep.payload()
        bad["events"][0]["after"] = [2, 2]
        with self.assertRaises(ValueError):
            from_payload(bad)

    def test_abandonment_does_not_prompt_again(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([("let's discuss bicycles", [8, 0])])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "abandoned")
        self.assertFalse(ep.resumed)
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(ep.position, MAZE.start)

    def test_template_reasoning_is_restored_in_next_turn_history(self):
        for prefilled in (False, True):
            with self.subTest(reasoning_prefilled=prefilled):
                ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
                thought = "Consider this move:\n" + call_text(MAZE.maze_id, "south")
                suffix = thought + "\n</think>\n" + call_text(MAZE.maze_id, "east")
                raw = suffix if prefilled else "<think>" + suffix
                manager = Manager([(raw, [8, 0]), (raw, [8, 0])])
                manager.reasoning_prefilled = prefilled
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, "arrived")
                self.assertEqual(ep.tool_attempts, 2)
                self.assertEqual(manager.calls[1][0][-2], {
                    "role": "assistant", "content": "<think>" + suffix,
                })
                # Replay keeps actual emitted text; only templated history is reconstructed.
                self.assertEqual(ep.turns[0]["text"], raw)
                self.assertEqual(ep.payload()["messages"][-2]["content"], "<think>" + suffix)

    def test_unfinished_and_thought_calls_do_not_move(self):
        for text, ids, phase in ((call_text(MAZE.maze_id, "east"), [8], "budget"),
                                 ("<think>\n" + call_text(MAZE.maze_id, "east") + "\n</think>", [8, 0], "abandoned")):
            ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
            list(stream_episode(ep, Manager([(text, ids)])))
            self.assertEqual(ep.phase, phase)
            self.assertEqual(ep.position, MAZE.start)

    def test_step_then_stop(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([(call_text(MAZE.maze_id, "east"), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.phase, "paused")
        self.assertFalse(manager.busy)
        ep.request_stop()
        self.assertEqual(ep.phase, "stopped")

    def test_failed_autosave_pauses_or_preserves_outcome_and_yields_cleanup(self):
        for supplied, reply, phase in ((0, call_text(MAZE.maze_id, 'east'), 'paused'),
                                       (1, call_text(MAZE.maze_id, 'east'), 'arrived'),
                                       (0, 'I am done.', 'abandoned')):
            with self.subTest(phase=phase):
                ep = Episode(MAZE, CONFIG | {'supplied_moves': supplied})
                manager = Manager([('\n' + reply, [8, 0])])
                with mock.patch.object(ep, 'save', side_effect=PermissionError('Archive not writable')) as save:
                    frames = [(frame.phase, frame.busy, frame.detail) for frame in stream_episode(ep, manager, save_dir=Path('/unused'))]
                save.assert_called_once()
                self.assertEqual(frames[-1][:2], (phase, False))
                self.assertIn('Autosave failed:', frames[-1][2])
                self.assertFalse(manager.busy)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(ep.resumed, phase != 'abandoned')
                self.assertEqual(ep.latency, 2 if phase != 'abandoned' else None)

    def test_save_failure_during_final_cleanup_still_yields_a_final_frame(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        with mock.patch.object(ep, 'save', side_effect=[None, OSError('Disk full')]) as save:
            frames = [(frame.phase, frame.busy, frame.detail) for frame in stream_episode(ep, manager, single_step=True, save_dir=Path('/unused'))]
        self.assertEqual(save.call_count, 2)
        self.assertEqual(frames[-1][:2], ('paused', False))
        self.assertIn('Disk full', frames[-1][2])
        self.assertFalse(manager.busy)
        self.assertEqual(ep.sampled_tokens, 2)

    def test_export_download_survives_unwritable_archive(self):
        ep = Episode(MAZE, CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'not-a-directory'
            archive.write_text('blocked')
            with mock.patch('extensions.maze_experiments.page.gr.Warning') as warning:
                path = Path(export_run(ep, archive))
            warning.assert_called_once()
        try:
            self.assertEqual(json.loads(path.read_text())['run_id'], ep.run_id)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        finally:
            path.unlink()
            path.parent.rmdir()

    def test_exporting_old_replay_does_not_overwrite_newer_completed_archive(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        move = call_text(MAZE.maze_id, 'east')
        manager = Manager([(move, [8, 0]), (move, [8, 0])])
        with tempfile.TemporaryDirectory() as directory:
            list(stream_episode(ep, manager, single_step=True, save_dir=Path(directory)))
            old_snapshot = copy.deepcopy(ep.payload())
            self.assertEqual(old_snapshot['phase'], 'paused')
            list(stream_episode(ep, manager, save_dir=Path(directory)))
            self.assertEqual(ep.phase, 'arrived')
            archive = Path(directory) / f'{ep.run_id}.json'
            completed = archive.read_bytes()
            path = Path(export_run(from_payload(old_snapshot), Path(directory)))
            try:
                self.assertEqual(archive.read_bytes(), completed)
                downloaded = json.loads(path.read_text())
                self.assertEqual(downloaded['phase'], 'paused')
                self.assertEqual(downloaded['run_id'], ep.run_id)
                self.assertNotEqual(path, archive)
            finally:
                path.unlink()
                path.parent.rmdir()

    def test_resuming_with_another_model_preserves_original_provenance(self):
        ep = Episode(MAZE, CONFIG | {"interruption_text": ""})
        manager = Manager([(call_text(MAZE.maze_id, "east"), [8, 0])])
        list(stream_episode(ep, manager, single_step=True))
        manager.model_id, manager.load_id = "different/model", "different-load"
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "error")
        self.assertEqual(ep.model_id, "test/model")
        self.assertEqual(len(manager.calls), 1)
        self.assertFalse(manager.busy)

    def test_stop_during_stream_does_not_execute_partial_action(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, "east"), [8, 0])])
        gen = stream_episode(ep, manager)
        next(gen)
        next(gen)
        ep.request_stop()
        list(gen)
        self.assertEqual(ep.position, MAZE.start)
        self.assertEqual(ep.phase, "stopped")
        self.assertIsNone(ep.resumed)
        self.assertFalse(manager.busy)

    def test_stop_on_completed_frame_does_not_start_another_response(self):
        for single_step in (False, True):
            with self.subTest(single_step=single_step):
                ep = Episode(MAZE, CONFIG)
                manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                for frame in stream_episode(ep, manager, single_step=single_step):
                    if frame.turns and frame.turns[-1]['finish_reason'] == 'stop':
                        ep.request_stop()
                self.assertEqual(ep.phase, 'stopped')
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(len(ep.turns), 1)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertFalse(manager.busy)

    def test_stop_during_response_gap_does_not_generate_more_tokens(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        with mock.patch('extensions.maze_experiments.runner.time.sleep', side_effect=lambda _: ep.request_stop()):
            list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(len(ep.turns), 1)
        self.assertEqual(ep.sampled_tokens, 2)

    def test_pause_during_response_gap_does_not_start_another_response(self):
        for also_stop in (False, True):
            with self.subTest(also_stop=also_stop):
                ep = Episode(MAZE, CONFIG)
                manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                def request_during_gap(_):
                    ep.request_pause()
                    if also_stop:
                        ep.request_stop()
                with mock.patch('extensions.maze_experiments.runner.time.sleep', side_effect=request_during_gap):
                    list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, 'stopped' if also_stop else 'paused')
                self.assertEqual(len(manager.calls), 1)
                self.assertEqual(len(ep.turns), 1)
                self.assertEqual(ep.sampled_tokens, 2)
                self.assertFalse(manager.busy)

    def test_pause_during_active_response_still_completes_its_move(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
        stream = stream_episode(ep, manager)
        next(stream)
        ep.request_pause()
        list(stream)
        self.assertEqual(ep.phase, 'paused')
        self.assertEqual(ep.position, (0, 1))
        self.assertEqual(ep.turns[0]['finish_reason'], 'stop')
        self.assertEqual(ep.sampled_tokens, 2)

    def test_stop_on_opening_frame_never_invokes_generation(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([])
        stream = stream_episode(ep, manager)
        next(stream)
        ep.request_stop()
        list(stream)
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(manager.calls, [])
        self.assertEqual(ep.sampled_tokens, 0)
        self.assertEqual(ep.turns[0]['finish_reason'], 'user_stopped')
        self.assertFalse(manager.busy)
        self.assertFalse(ep.interrupted)
        self.assertIsNone(ep.intervention_turn)
        self.assertEqual(ep.turns[0]['forced_prefix_tokens'], 0)
        self.assertEqual(ep.turns[0]['prefix_ids'], [])
        self.assertEqual(ep.turns[0]['prefix_text'], '')
        self.assertTrue(ep.turns[0]['planned_prefix_ids'])
        selections = TokenInspector().selections()
        rendered = views(ep, False, selections, selections.new_session())
        self.assertIn('insertion has not been confirmed', rendered[4])

    def test_closing_completed_frame_preserves_terminal_archive(self):
        for supplied, text, ids, phase in ((1, call_text(MAZE.maze_id, 'east'), [8, 0], 'arrived'),
                                           (0, 'I am done.', [8, 0], 'abandoned'),
                                           (0, 'Unfinished', [8], 'budget')):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                ep = Episode(MAZE, CONFIG | {'supplied_moves': supplied, 'interruption_text': ''})
                manager = Manager([(text, ids)])
                stream = stream_episode(ep, manager, save_dir=Path(directory))
                for frame in stream:
                    if frame.phase == phase:
                        before = copy.deepcopy(frame.payload())
                        stream.close()
                        break
                self.assertEqual(ep.payload(), before)
                saved = json.loads((Path(directory) / f'{ep.run_id}.json').read_text())
                self.assertEqual(saved, json.loads(json.dumps(before)))
                self.assertFalse(manager.busy)

    def test_closing_active_stream_retains_partial_tokens_without_moving(self):
        ep = Episode(MAZE, CONFIG | {'interruption_text': ''})
        manager = Manager([(call_text(MAZE.maze_id, 'east'), [8, 0])])
        stream = stream_episode(ep, manager)
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(ep.sampled_tokens, 2)
        self.assertEqual(ep.position, MAZE.start)
        self.assertFalse(manager.busy)

    def test_failure_before_prefix_update_does_not_record_insertion(self):
        ep = Episode(MAZE, CONFIG)
        ep.request_interruption()
        manager = Manager([])
        with mock.patch.object(manager, 'generate', side_effect=RuntimeError('Prefill failed')):
            list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'error')
        self.assertFalse(ep.interrupted)
        self.assertIsNone(ep.intervention_turn)
        self.assertEqual(ep.intervention_tokens, 0)
        self.assertEqual(ep.intervention_attempts, 0)
        self.assertTrue(ep.manual_intervention)  # A request occurred, but insertion did not.
        self.assertEqual(ep.turns[0]['prefix_ids'], [])
        self.assertFalse(manager.busy)

    def test_prefix_only_update_records_consumption_even_if_sampling_fails(self):
        ep = Episode(MAZE, CONFIG | {'per_turn_tokens': 2048, 'token_budget': 8192})
        manager = Manager([])
        def generate(messages, **options):
            self.assertEqual(options['max_new_tokens'], 1024)
            prefix = options['forced_ids']
            yield SimpleNamespace(text=manager.tokenizer.decode(prefix), metrics=[{'token_id': t} for t in prefix],
                                  prompt_ids=[10], forced_prefix_tokens=len(prefix), reasoning_prefilled=False,
                                  load_id=manager.load_id, model_id=manager.model_id)
            raise RuntimeError('Sampling failed')
        manager.generate = generate
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'error')
        self.assertTrue(ep.interrupted)
        self.assertEqual(ep.intervention_turn, 0)
        self.assertEqual(ep.intervention_tokens, 0)
        self.assertEqual(ep.intervention_attempts, 0)
        self.assertEqual(ep.sampled_tokens, 0)
        self.assertIsNone(ep.resumed)
        self.assertEqual(ep.turns[0]['prefix_ids'], [68, 105])
        self.assertEqual(ep.turns[0]['prefix_text'], 'Di')
        self.assertFalse(manager.busy)

    def test_stop_at_startup_lock_handoff_is_not_cleared(self):
        ep = Episode(MAZE, CONFIG)
        manager = Manager([])
        class StopOnFirstUnlock:
            def __init__(self):
                self.lock = threading.Lock()
                self.released = False
            def __enter__(self):
                self.lock.acquire()
            def __exit__(self, *exc):
                self.lock.release()
                if not self.released:
                    self.released = True
                    ep.request_stop()
        ep.lock = StopOnFirstUnlock()
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, 'stopped')
        self.assertEqual(manager.calls, [])
        self.assertEqual(ep.turns, [])
        self.assertFalse(manager.busy)

    def test_idle_stops_persist_ready_and_paused_runs(self):
        for paused in (False, True):
            with self.subTest(paused=paused), tempfile.TemporaryDirectory() as directory:
                ep = Episode(MAZE, CONFIG)
                if paused:
                    manager = Manager([('\n' + call_text(MAZE.maze_id, 'east'), [8, 0])])
                    list(stream_episode(ep, manager, single_step=True, save_dir=Path(directory)))
                before = (ep.sampled_tokens, ep.tool_attempts, ep.resumed, ep.latency)
                ep.request_stop(Path(directory))
                saved = json.loads((Path(directory) / f'{ep.run_id}.json').read_text())
                self.assertEqual(saved['phase'], 'stopped')
                self.assertEqual((ep.sampled_tokens, ep.tool_attempts, ep.resumed, ep.latency), before)

    def test_idle_stop_does_not_rewrite_replays_or_finished_runs(self):
        ep = Episode(MAZE, CONFIG)
        ep.phase = 'paused'
        with tempfile.TemporaryDirectory() as directory:
            path = ep.save(Path(directory))
            original = path.read_bytes()
            replay = from_payload(json.loads(original))
            replay.request_stop(Path(directory))
            self.assertEqual(replay.phase, 'paused')
            self.assertFalse(replay.stop_requested)
            self.assertEqual(path.read_bytes(), original)
        ep.phase = 'arrived'
        with mock.patch.object(ep, 'save') as save:
            ep.request_stop(Path('/unused'))
        save.assert_not_called()
        self.assertEqual(ep.phase, 'arrived')

    def test_idle_stop_storage_failure_retains_stopped_state_and_export_guidance(self):
        ep = Episode(MAZE, CONFIG)
        with mock.patch.object(ep, 'save', side_effect=PermissionError('Archive not writable')) as save:
            ep.request_stop(Path('/unused'))
            ep.request_stop(Path('/unused'))
        save.assert_called_once()
        self.assertEqual(ep.phase, 'stopped')
        self.assertIn('Autosave failed:', ep.detail)
        self.assertIn('Export run JSON', ep.detail)

    def test_export_uses_a_private_temporary_file(self):
        ep = Episode(MAZE, CONFIG)
        path = ep.export()
        try:
            self.assertTrue(path.is_relative_to(Path(tempfile.gettempdir())))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["run_id"], ep.run_id)
        finally:
            path.unlink()
            path.parent.rmdir()

    def test_native_template_gets_tools(self):
        calls = []
        def template(messages, **kwargs):
            calls.append(kwargs)
            return [1, 2] if kwargs["tokenize"] else "tool prompt"
        manager = ModelManager()
        manager.tokenizer = SimpleNamespace(chat_template="native", apply_chat_template=template)
        ids, reasoning = manager._prompt_token_ids([], tools=[{"name": "move"}])
        self.assertEqual(ids, [1, 2])
        self.assertFalse(reasoning)
        self.assertEqual([c["tools"] for c in calls], [[{"name": "move"}]] * 2)


if __name__ == "__main__":
    unittest.main()
