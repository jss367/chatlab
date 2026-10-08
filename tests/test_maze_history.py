import ast
import copy
import json
import unittest
from pathlib import Path

import jinja2

from chatlab.extension_api import ModelService
from chatlab.extensions.maze_experiments import history
from chatlab.extensions.maze_experiments.history import assistant_message, check_messages, history_form, split_reasoning
from chatlab.extensions.maze_experiments.maze import call_text
from chatlab.extensions.maze_experiments.page import context_view, prompt_reading, transcript
from chatlab.extensions.maze_experiments.runner import (Episode, context_messages, fork_token_edit, from_payload,
                                                        read_run_file, stream_episode)

from maze_support import CONFIG, MAZE, Manager, team_episode

RUN = CONFIG | {"interruption_text": ""}
THOUGHT = "The goal is east of me."
CALL = call_text(MAZE.tool_id(), "east")
# What a model writes after a template that ends its prompt with "<think>\n".
REASONED = THOUGHT + "\n</think>\n\n" + CALL
# The assistant branch of Qwen3.8-27B's template (chat_template.jinja lines
# 111-117, with the rest of that branch's rendering) and its generation
# prompt: an earlier turn's reasoning block is built from reasoning_content
# alone, and content is written after it whatever it holds.
QWEN38 = jinja2.Environment().from_string(
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{%- set reasoning_content = '' %}"
    "{%- if message.reasoning_content is string %}{%- set reasoning_content = message.reasoning_content %}{%- endif %}"
    "{%- set reasoning_content = reasoning_content|trim %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + message.content|trim"
    " + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' }}")


def reasoned(text=REASONED):
    return text, list(text.encode()) + [0]


class ReasoningManager(Manager):
    """A model whose template opens every response's reasoning block."""
    reasoning_prefilled = True


class Qwen38Manager(ReasoningManager):
    def _prompt_token_ids(self, messages, tools=None):
        return list(QWEN38.render(messages=messages).encode()), True


class ContentOnlyManager(ReasoningManager):
    """OLMo-style template: only content contributes to the prompt."""
    def _prompt_token_ids(self, messages, tools=None):
        return list("".join(m["content"] for m in messages).encode()), True


def written_before_the_field(payload):
    """A run as a ChatLab predating the history form saved it."""
    payload = copy.deepcopy(payload)
    payload["config"].pop("reasoning_history")
    return payload


def saved(ep):
    return json.loads(json.dumps(ep.payload()))


def single_run(form, insert=False):
    """Two reasoned moves east, the history written in ``form``, a note inserted before the second if asked."""
    ep = Episode(MAZE, RUN | {"reasoning_history": form})
    manager = ReasoningManager([reasoned(), reasoned()])
    list(stream_episode(ep, manager, single_step=True))
    if insert:
        ep.request_insert("tool_note", "East reaches the goal.", advised_direction="east")
    list(stream_episode(ep, manager))
    assert ep.phase == "arrived", ep.detail
    return ep, manager


def team_run(form):
    ep = team_episode(MAZE, RUN | {"reasoning_history": form, "communication": False, "team_goal": "any",
                                   "per_turn_tokens": 1000, "token_budget": 4000})
    list(stream_episode(ep, ReasoningManager([reasoned()] * 4)))
    assert ep.phase == "arrived", ep.detail
    return ep


class SplittingTests(unittest.TestCase):
    def test_a_closed_block_splits_into_reasoning_and_answer(self):
        self.assertEqual(assistant_message("\n" + THOUGHT + "\n\n</think>\n\n  " + CALL, reasoning_prefilled=True),
                         {"role": "assistant", "reasoning_content": THOUGHT, "content": CALL})
        # A block the model opened itself splits the same way.
        self.assertEqual(assistant_message("<think>\n" + REASONED),
                         {"role": "assistant", "reasoning_content": THOUGHT, "content": CALL})
        # Only the newlines around the reasoning are trimmed, and only the start of the answer.
        self.assertEqual(split_reasoning("<think>\n  indented\n</think>answer \n"), ("  indented", "answer \n"))

    def test_reasoning_cut_off_before_it_closed_is_all_reasoning(self):
        self.assertEqual(assistant_message("Still weighing east\n", reasoning_prefilled=True),
                         {"role": "assistant", "reasoning_content": "Still weighing east", "content": ""})

    def test_a_response_that_opened_no_block_keeps_its_form(self):
        for text in (CALL, "East. <think>an aside</think>", ""):
            with self.subTest(text=text):
                self.assertIsNone(split_reasoning(text))
                self.assertEqual(assistant_message(text), {"role": "assistant", "content": text})

    def test_the_content_form_keeps_the_whole_response_with_its_opening_tag(self):
        self.assertEqual(assistant_message(REASONED, True, "content"),
                         {"role": "assistant", "content": "<think>" + REASONED})
        with self.assertRaisesRegex(ValueError, "reasoning history"):
            assistant_message(REASONED, True, "both")

    def test_a_run_naming_no_form_was_written_in_content(self):
        self.assertEqual(history_form({}), "content")
        self.assertEqual(history_form({"reasoning_history": "reasoning_content"}), "reasoning_content")
        with self.assertRaisesRegex(ValueError, "reasoning_history"):
            history_form({"reasoning_history": "both"})

    def test_messages_are_checked_against_the_form_named(self):
        split = assistant_message(REASONED, True)
        whole = assistant_message(REASONED, True, "content")
        check_messages([split, {"role": "tool", "content": "{}"}], "reasoning_content")
        check_messages([whole], "content")
        with self.assertRaisesRegex(ValueError, "keeps its reasoning in content"):
            check_messages([whole], "reasoning_content")
        with self.assertRaisesRegex(ValueError, "carries reasoning_content"):
            check_messages([split], "content")
        with self.assertRaisesRegex(ValueError, "Only a response"):
            check_messages([{"role": "tool", "content": "{}", "reasoning_content": ""}], "reasoning_content")

    def test_the_helper_needs_only_the_standard_library(self):
        # So a collector outside ChatLab can import it, or copy it.
        tree = ast.parse(Path(history.__file__).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {"__future__"})


class TemplateTests(unittest.TestCase):
    def test_fresh_trials_select_content_under_the_shared_batch_session(self):
        manager = ContentOnlyManager([reasoned()] * 4)
        session = manager.open_session()
        try:
            for _ in range(2):
                ep = Episode(MAZE, RUN)
                list(stream_episode(ep, manager, session=session))
                self.assertEqual(ep.phase, "arrived")
                self.assertEqual(ep.config["reasoning_history"], "content")
                self.assertIn("<think>" + THOUGHT, bytes(ep.turns[1]["prompt_ids"]).decode())
                self.assertTrue(manager.busy)
        finally:
            session.close()
        self.assertFalse(manager.busy)

    def test_content_only_template_retains_reasoning_on_later_solo_and_team_turns(self):
        for team in (False, True):
            with self.subTest(team=team):
                ep = (team_episode(MAZE, RUN | {"communication": False, "team_goal": "any",
                                               "per_turn_tokens": 1000, "token_budget": 4000})
                      if team else Episode(MAZE, RUN))
                manager = ContentOnlyManager([reasoned()] * 4)
                list(stream_episode(ep, manager))
                self.assertEqual(ep.phase, "arrived")
                self.assertEqual(ep.config["reasoning_history"], "content")
                later = ep.turns[2 if team else 1]
                self.assertIn("<think>" + THOUGHT, bytes(later["prompt_ids"]).decode())
                replay = from_payload(saved(ep))
                self.assertEqual(replay.config["reasoning_history"], "content")

    def test_template_selection_is_kept_after_pause(self):
        ep = Episode(MAZE, RUN)
        manager = ContentOnlyManager([reasoned(), reasoned()])
        list(stream_episode(ep, manager, single_step=True))
        self.assertEqual(ep.config["reasoning_history"], "content")
        self.assertFalse(ep.select_history)
        list(stream_episode(ep, manager))
        self.assertIn("<think>" + THOUGHT, bytes(ep.turns[1]["prompt_ids"]).decode())

    def test_an_explicit_recorded_form_is_not_reselected(self):
        ep = Episode(MAZE, RUN | {"reasoning_history": "reasoning_content"})
        list(stream_episode(ep, ContentOnlyManager([reasoned(), reasoned()])))
        self.assertEqual(ep.config["reasoning_history"], "reasoning_content")

    def test_qwen38_reads_one_reasoning_block_per_earlier_turn(self):
        ep = Episode(MAZE, RUN)
        manager = Qwen38Manager([reasoned(), reasoned()])
        list(stream_episode(ep, manager))
        self.assertEqual(ep.phase, "arrived")
        self.assertEqual(ep.config["reasoning_history"], "reasoning_content")
        prompt = bytes(ep.turns[1]["prompt_ids"]).decode()
        earlier = prompt.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
        self.assertEqual(earlier, "<think>\n" + THOUGHT + "\n</think>\n\n" + CALL)
        # The generation prompt opens the response's own block, and nothing else does.
        self.assertEqual(prompt.count("<think>"), 2)

    def test_the_content_form_gave_qwen38_an_empty_block_and_a_literal_one(self):
        ep = Episode(MAZE, RUN | {"reasoning_history": "content"})
        list(stream_episode(ep, Qwen38Manager([reasoned(), reasoned()])))
        earlier = bytes(ep.turns[1]["prompt_ids"]).decode().split("<|im_start|>assistant\n")[1]
        self.assertTrue(earlier.startswith("<think>\n\n</think>\n\n<think>" + THOUGHT))


class RoundTripTests(unittest.TestCase):
    def upload(self, path, manager):
        return from_payload(read_run_file(path),
                            read_prompt=lambda run, turn, context: prompt_reading(run, turn, context, manager))

    def test_a_new_run_round_trips_through_export_and_upload(self):
        for insert in (False, True):
            with self.subTest(insert=insert):
                ep, manager = single_run("reasoning_content", insert)
                self.assertEqual(ep.messages[-2], {"role": "assistant", "reasoning_content": THOUGHT, "content": CALL})
                replay = self.upload(ep.export(), manager)
                self.assertEqual(replay.config["reasoning_history"], "reasoning_content")
                self.assertEqual(replay.messages, ep.messages)
                self.assertIn("reasoning_content", saved(ep)["tokenizer_note"])

    def test_a_run_written_before_the_field_round_trips_as_it_did(self):
        for insert in (False, True):
            with self.subTest(insert=insert):
                ep, manager = single_run("content", insert)
                old = written_before_the_field(saved(ep))
                replay = from_payload(old, read_prompt=lambda run, turn, context: prompt_reading(run, turn, context,
                                                                                                  manager))
                self.assertEqual(replay.config["reasoning_history"], "content")
                self.assertEqual(replay.messages, old["messages"])
                self.assertEqual(replay.messages[-2], {"role": "assistant", "content": "<think>" + REASONED})
                # Exported again, it names the form it was read in, and reads back the same.
                again = self.upload(replay.export(), manager)
                self.assertEqual(again.messages, old["messages"])

    def test_a_team_round_trips_in_either_form(self):
        for form in ("reasoning_content", "content"):
            with self.subTest(form=form):
                ep = team_run(form)
                payload = saved(ep) if form == "reasoning_content" else written_before_the_field(saved(ep))
                replay = from_payload(payload)
                self.assertEqual(replay.config["reasoning_history"], form)
                for agent, recorded in zip(replay.agents, payload["agents"]):
                    self.assertEqual(agent["messages"], recorded["messages"])
                self.assertEqual(replay.agents[0]["messages"][-2],
                                 assistant_message(REASONED, True, form))

    def test_a_fork_keeps_the_form_of_the_run_it_forks(self):
        for form in ("reasoning_content", "content"):
            with self.subTest(form=form):
                ep, manager = single_run(form)
                payload = saved(ep) if form == "reasoning_content" else written_before_the_field(saved(ep))
                replay = from_payload(payload)
                with ModelService(lambda: manager).open_session() as session:
                    fork = fork_token_edit(replay, 1, 0, "I", session)
                self.assertEqual(fork.config["reasoning_history"], form)
                self.assertEqual(fork.messages, context_messages(replay, 1))
                self.assertEqual(fork.messages[-2], assistant_message(REASONED, True, form))


class ContextPaneTests(unittest.TestCase):
    def test_the_recorded_prompt_is_the_history_through_the_template_in_either_form(self):
        for form in ("reasoning_content", "content"):
            with self.subTest(form=form):
                ep, manager = single_run(form)
                header, text = context_view(ep, manager, 1)
                self.assertIn("as recorded", header)
                templated, _ = manager.prompt_text(context_messages(ep, 1), ep.tools)
                self.assertEqual(text, templated)

    def test_the_untemplated_transcript_shows_reasoning_kept_apart(self):
        ep, _ = single_run("reasoning_content")
        shown = transcript(context_messages(ep, 1), ep.tools)
        self.assertIn("[assistant reasoning]\n" + THOUGHT + "\n\n[assistant]\n" + CALL, shown)


class DeclarationTests(unittest.TestCase):
    def test_a_new_run_whose_messages_keep_reasoning_in_content_is_refused(self):
        payload = saved(single_run("reasoning_content")[0])
        payload["messages"][-2] = assistant_message(REASONED, True, "content")
        with self.assertRaisesRegex(ValueError, "keeps its reasoning in content"):
            from_payload(payload)

    def test_a_run_naming_no_form_whose_messages_carry_reasoning_content_is_refused(self):
        payload = saved(single_run("reasoning_content")[0])
        for declared in (None, "content"):
            with self.subTest(declared=declared):
                broken = written_before_the_field(payload)
                if declared:
                    broken["config"]["reasoning_history"] = declared
                with self.assertRaisesRegex(ValueError, "carries reasoning_content"):
                    from_payload(broken)

    def test_a_new_run_is_rebuilt_whole_and_refused_where_its_reasoning_differs(self):
        payload = saved(single_run("reasoning_content")[0])
        payload["messages"][-2]["reasoning_content"] = "Something else."
        with self.assertRaisesRegex(ValueError, "disagree"):
            from_payload(payload)

    def test_an_unknown_form_is_refused(self):
        payload = saved(single_run("reasoning_content")[0])
        payload["config"]["reasoning_history"] = "both"
        with self.assertRaisesRegex(ValueError, "reasoning_history"):
            from_payload(payload)

    def test_a_team_whose_messages_disagree_with_its_form_is_refused(self):
        new = saved(team_run("reasoning_content"))
        old = written_before_the_field(saved(team_run("content")))
        swapped_new, swapped_old = copy.deepcopy(new), copy.deepcopy(old)
        swapped_new["agents"][0]["messages"] = old["agents"][0]["messages"]
        swapped_old["agents"][0]["messages"] = new["agents"][0]["messages"]
        with self.assertRaisesRegex(ValueError, "keeps its reasoning in content"):
            from_payload(swapped_new)
        with self.assertRaisesRegex(ValueError, "carries reasoning_content"):
            from_payload(swapped_old)


if __name__ == "__main__":
    unittest.main()
