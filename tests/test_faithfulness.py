"""The Reasoning check: cutting, breaking and paraphrasing a chat reply's reasoning, then answering again."""

import unittest
from types import SimpleNamespace
from unittest import mock

import gradio as gr
import torch

import settings_sandbox
from chatlab import faithfulness as check
from chatlab.conversation import model_messages, split_reasoning
from chatlab.model_runtime import GENERATING, ModelManager
from chatlab.ui import reasoning_check, runtime
from fakes import ChatTemplateTokenizer

PIECES = ["prompt", "<think>", "</think>", "\n", "\n\n", "Two", " plus", " two", " is", " four", " five", ".",
          "Four", "Five", "Dunno", "Adding", " and", " gives", " Two", "<eos>"]
EOS = PIECES.index("<eos>")
# What the model writes after each ending of its reasoning so far.
REASONING = {"": "Two", "Two": " plus", " plus": " two", " two": " is", " is": " four", " four": ".", " five": ".",
             ".": "\n", "\n": "</think>"}


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class ReasoningModel(torch.nn.Module):
    """Reasons that two plus two is four, then answers from whatever its reasoning says.

    Each position predicts from the whole response so far, so a cut or a
    planted step changes what it answers: "Four" for reasoning that reaches
    four, "Five" for five, and "Dunno" for reasoning that reaches neither.
    """

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=EOS)
        self.seen = []

    def next_piece(self, text):
        if "</think>" not in text:
            ending = next((end for end in sorted(REASONING, key=len, reverse=True) if text.endswith(end)), "")
            return REASONING[ending]
        reasoning, _, answer = text.partition("</think>")
        if not answer:
            return "\n\n"
        if answer == "\n\n":
            return "Four" if " four" in reasoning or "four" in reasoning else "Five" if " five" in reasoning else "Dunno"
        return "<eos>" if answer.endswith(".") else "."

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True, **_kwargs):
        ids = input_ids[0].tolist()
        self.seen = self.seen[:attention_mask.shape[1] - len(ids)]
        logits = torch.full((1, len(ids), len(PIECES)), -20.0)
        for offset, token in enumerate(ids):
            self.seen.append(token)
            text = "".join(PIECES[i] for i in self.seen[1:])
            logits[0, offset, PIECES.index(self.next_piece(text))] = 20.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def manager():
    held = ModelManager()
    held.tokenizer = ChatTemplateTokenizer("\nassistant: <think>", pieces=PIECES, eos_id=EOS)
    held.model = ReasoningModel()
    held.model_id = "fake/reasoner"
    return held


def written(held, question="What is two plus two?"):
    """A conversation whose reply the model wrote, carrying what Chat records on a reply."""
    turns = [{"role": "user", "content": question, "reasoning": ""}]
    last = None
    for last in held.generate(model_messages(turns), temperature=0., top_p=1., top_k=0, max_new_tokens=40,
                              seed=0, analyze_prompt=False):
        pass
    reasoning, answer, _ = split_reasoning(last.text, reasoning_prefilled=last.reasoning_prefilled)
    turns.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tokens": list(last.metrics),
                  "load_id": last.load_id, "metrics_generation": 7, "ends_on_stop_token": last.ends_on_stop_token,
                  "generation_settings": {"system_prompt": "", "keep_reasoning": False}})
    return turns


def finished(steps):
    """Run a check generator to its end and return what it returns."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.manager = manager()
        self.turns = written(self.manager)
        self.reply = check.read_reply(self.turns, 1, self.manager)
        self.encode = lambda kept, text: self.manager.encode_replacement(kept, text, load_id=self.reply.load_id)

    def run_plan(self, plan, pattern=""):
        return finished(check.run_plan(self.manager, self.reply, plan, max_new_tokens=20, pattern=pattern,
                                       encode_after=self.encode))

    def test_the_reply_is_located_in_its_own_tokens(self):
        self.assertEqual(self.turns[1]["reasoning"], "Two plus two is four.")
        self.assertEqual(self.turns[1]["content"], "Four.")
        self.assertTrue(self.reply.prefilled)
        self.assertEqual(self.reply.reasoning, "Two plus two is four.")
        self.assertEqual(self.reply.answer, "Four.")
        self.assertEqual(self.reply.close, "\n</think>\n\n")
        self.assertEqual(self.reply.number, 1)

    def test_each_cut_keeps_its_share_of_the_words_and_closes_the_reasoning(self):
        spelled = [self.manager._decode_ids(check.cut_plan(self.reply, f, self.encode).forced_ids)
                   for f in check.FRACTIONS]
        self.assertEqual(spelled, ["\n</think>\n\n", "Two\n</think>\n\n", "Two plus\n</think>\n\n",
                                   "Two plus two is\n</think>\n\n", "Two plus two is four.\n</think>\n\n"])

    def test_a_cut_replays_the_replys_own_tokens(self):
        plan = check.cut_plan(self.reply, .5, self.encode)
        self.assertEqual(plan.forced_ids[:2], self.reply.ids[:2])

    def test_an_empty_completion_is_no_answer(self):
        plan = check.cut_plan(self.reply, 0, self.encode)

        def empty(*args, **kwargs):
            yield 0
            return SimpleNamespace(text="<think></think>   ")

        with mock.patch.object(check, "_generate", side_effect=empty):
            result = self.run_plan(plan)
        self.assertIsNone(result.answer)
        self.assertIsNone(result.same)
        self.assertEqual(check.result_rows([result])[0][4], "No answer")

    def test_reasoning_the_answer_needs_changes_it_when_cut(self):
        results = [self.run_plan(check.cut_plan(self.reply, f, self.encode)) for f in check.FRACTIONS]
        self.assertEqual([r.answer for r in results], ["Dunno."] * 4 + ["Four."])
        self.assertEqual([r.same for r in results], [False] * 4 + [True])
        # The original answer is all but certain after the whole reasoning and
        # all but impossible after any less of it.
        self.assertAlmostEqual(results[-1].perplexity, 1.0, places=3)
        self.assertTrue(all(r.perplexity > 1e6 for r in results[:-1]))
        self.assertEqual(results[2].reasoning, "Two plus")

    def test_an_unscorable_perplexity_keeps_the_generated_answer(self):
        plan = check.cut_plan(self.reply, 0, self.encode)
        prompt, _ = self.manager._prompt_token_ids(self.reply.messages, thinking_mode=self.reply.thinking_mode)
        # The intervention prefix fits, while prefix plus original answer does not.
        limit = len(prompt) + len(plan.forced_ids)
        with mock.patch("chatlab.tokenization.application_prefill_limit", return_value=limit):
            result = self.run_plan(plan)
        self.assertEqual(result.answer, "Dunno.")
        self.assertFalse(result.same)
        self.assertIsNone(result.perplexity)
        self.assertEqual(check.result_rows([result])[0][-1], "—")

    def test_a_planted_mistake_is_fed_and_the_model_reasons_on_from_it(self):
        plan = check.mistake_plan(self.reply, "Two plus two is five.", self.encode)
        self.assertTrue(plan.continues)
        self.assertEqual(self.manager._decode_ids(plan.forced_ids), "Two plus two is five.")
        result = self.run_plan(plan)
        self.assertEqual(result.answer, "Five.")
        self.assertFalse(result.same)
        self.assertEqual(result.reasoning, "Two plus two is five.")
        self.assertGreater(result.perplexity, 1e6)

    def test_a_paraphrase_that_keeps_the_step_keeps_the_answer(self):
        plan = check.paraphrase_plan(self.reply, "Adding two and two gives four.", self.encode)
        self.assertEqual(self.manager._decode_ids(plan.forced_ids), "Adding two and two gives four.\n</think>\n\n")
        result = self.run_plan(plan)
        self.assertEqual(result.answer, "Four.")
        self.assertTrue(result.same)
        self.assertAlmostEqual(result.perplexity, 1.0, places=3)

    def test_an_answer_pattern_compares_only_what_it_matches(self):
        self.assertEqual(check.answer_key("So the answer is 391.", r"answer is (\d+)"), "391")
        self.assertEqual(check.answer_key("12, then 15", r"\d+"), "15")
        self.assertIsNone(check.answer_key("No number", r"\d+"))
        self.assertIsNone(check.answer_key("word", r"(\d+)?[A-Za-z]+"))
        self.assertIsNone(check.answer_key("word", r"(\d+)|[A-Za-z]+"))
        unmatched = self.run_plan(check.cut_plan(self.reply, 0, self.encode), pattern=r"(\d+)?[A-Za-z]+")
        self.assertIsNone(unmatched.same)
        self.assertEqual(check.answer_key("  Four   IS it "), "four is it")
        result = self.run_plan(check.cut_plan(self.reply, 0, self.encode), pattern=r"\d+")
        self.assertIsNone(result.same)
        with self.assertRaisesRegex(ValueError, "not a regular expression"):
            check.check_pattern("(")

    def test_a_mistake_needs_a_change_that_writes_something(self):
        with self.assertRaisesRegex(ValueError, "Change a step"):
            check.mistake_plan(self.reply, "Two plus two is four.", self.encode)
        with self.assertRaisesRegex(ValueError, "only removes"):
            check.mistake_plan(self.reply, "Two plus two is.", self.encode)
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            check.mistake_plan(self.reply, "Two plus two</think> is four.", self.encode)
        with self.assertRaisesRegex(ValueError, "Write the paraphrase"):
            check.paraphrase_plan(self.reply, "  ", self.encode)

    def test_a_reply_from_another_load_is_refused(self):
        self.manager.load_count += 1
        with self.assertRaisesRegex(ValueError, "no longer loaded"):
            check.read_reply(self.turns, 1, self.manager)

    def test_a_reply_without_tokens_is_not_offered(self):
        self.assertEqual([p for _, p in check.reasoning_replies(self.turns)], [1])
        typed = [*self.turns, {"role": "user", "content": "And?"},
                 {"role": "assistant", "content": "Yes.", "reasoning": "Sure."}]
        self.assertEqual([p for _, p in check.reasoning_replies(typed)], [1])

    def test_tokens_that_no_longer_spell_the_reply_are_refused(self):
        self.turns[1]["content"] = "Four, edited."
        with self.assertRaisesRegex(ValueError, "do not spell"):
            check.read_reply(self.turns, 1, self.manager)

    def test_the_model_writes_a_paraphrase(self):
        text = finished(check.write_paraphrase(self.manager, self.reply, 40))
        # The fake reasons and answers whatever it is asked; what comes back
        # is the answer, never its reasoning.
        self.assertEqual(text, "Four.")


class StepTests(unittest.TestCase):
    """Where a planted mistake ends, read without a model."""

    def reply(self, reasoning):
        text = reasoning + "\n</think>\n\nOK."
        ends = list(range(1, len(text) + 1))
        layout = check.read_layout(text, True)
        return check.Reply(0, 1, list(range(len(text))), ends, text, True, *layout, [], "default", None, None, None)

    def planted(self, reasoning, edited):
        return check.mistake_plan(self.reply(reasoning), edited, lambda kept, text: [ord(c) for c in text]).reasoning

    def test_a_changed_number_keeps_the_rest_of_its_sentence(self):
        self.assertEqual(self.planted("17 times 23 is 391. So 391.", "17 times 24 is 391. So 391."),
                         "17 times 24 is 391.")

    def test_an_inserted_sentence_ends_where_it_does(self):
        self.assertEqual(self.planted("A is one. B is two.", "A is one. X is nine. B is two."), "A is one. X is nine.")

    def test_a_step_can_end_at_a_line(self):
        self.assertEqual(self.planted("First line\nSecond line", "First lime\nSecond line"), "First lime\n")

    def test_token_ends_fall_back_to_decoding_prefixes(self):
        # Standalone labels that drop a word-boundary space do not add up.
        metrics = [{"token_id": 0, "text": "A"}, {"token_id": 1, "text": "b"}, {"token_id": 2, "text": ""}]
        decode = lambda ids: "".join(["A", " b", ""][i] for i in ids)
        self.assertEqual(check.token_ends(metrics, decode, {2}), ("A b", [1, 3, 3]))


class TabTests(unittest.TestCase):
    def setUp(self):
        self.manager = manager()
        self.turns = written(self.manager)
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        _choices, self.picked, self.mistake, *_ = reasoning_check.refresh_replies(self.turns, None, None)

    def run_tab(self, kind, text="", picked=None):
        return list(reasoning_check.run_check(kind, self.turns, 1, picked or self.picked, "", 20, "", False, [],
                                              text=text))

    def test_the_newest_reply_is_picked_with_its_reasoning(self):
        update, picked, mistake, paraphrase, status = reasoning_check.refresh_replies(self.turns, None, None)
        self.assertEqual(update["value"], 1)
        self.assertEqual(mistake, "Two plus two is four.")
        self.assertEqual(picked, check.reply_identity(self.turns[1]))
        # Refreshed again while it is the same reply, the boxes are left alone.
        again = reasoning_check.refresh_replies(self.turns, 1, picked)
        self.assertTrue(all(isinstance(value, dict) and value.get("__type__") == "update" and len(value) == 1
                            for value in again[1:]))

    def test_the_reply_list_is_one_gradio_accepts(self):
        update = reasoning_check.refresh_replies(self.turns, None, None)[0]
        picker = gr.Dropdown(choices=update["choices"], value=update["value"])
        self.assertEqual(picker.preprocess(update["value"]), 1)

    def test_the_cut_test_adds_a_row_per_cut_and_gives_the_slot_back(self):
        frames = self.run_tab(check.CUT)
        results, table, status = frames[-1][:3]
        self.assertEqual(len(results), len(check.FRACTIONS))
        self.assertEqual([row[4] for row in table["value"]], ["No"] * 4 + ["Yes"])
        self.assertIn("Finished", status)
        self.assertIsNone(self.manager.occupant)

    def test_a_planted_mistake_row_says_the_answer_changed(self):
        results = self.run_tab(check.MISTAKE, "Two plus two is five.")[-1][0]
        self.assertEqual((results[0].kind, results[0].answer, results[0].same), (check.MISTAKE, "Five.", False))

    def test_a_reply_replaced_since_it_was_picked_is_refused(self):
        frames = self.run_tab(check.CUT, picked=("other",))
        self.assertIn("Pick it again", frames[-1][2])
        self.assertIsNone(self.manager.occupant)

    def test_a_busy_model_is_not_claimed(self):
        with mock.patch.object(self.manager, "claim_generation", return_value=GENERATING):
            frames = self.run_tab(check.CUT)
        self.assertEqual(frames[-1][2], reasoning_check.BUSY)

    def test_stop_is_visible_before_the_first_prefill(self):
        def expensive_run(*args, **kwargs):
            raise AssertionError("The first prefill ran before Stop was published")
            yield
        with mock.patch.object(check, "run_plan", side_effect=expensive_run):
            steps = reasoning_check.run_check(check.CUT, self.turns, 1, self.picked, "", 20, "", False, [])
            frame = next(steps)
            self.assertTrue(frame[-1]["visible"])
            steps.close()
        self.assertIsNone(self.manager.occupant)
        with mock.patch.object(check, "write_paraphrase", side_effect=expensive_run):
            steps = reasoning_check.paraphrase_reply(self.turns, 1, self.picked, 20, "", False)
            frame = next(steps)
            self.assertTrue(frame[-1]["visible"])
            steps.close()
        self.assertIsNone(self.manager.occupant)

    def test_detail_fences_are_longer_than_any_content_backtick_run(self):
        result = self.run_tab(check.MISTAKE, "Two plus two is five.")[-1][0][0]
        from dataclasses import replace
        for count in (3, 4, 5, 12):
            content = "before\n" + "`" * count + "\nafter"
            detail = check.result_detail(replace(result, reasoning=content, answer=content))
            fence = "`" * (count + 1)
            self.assertEqual(detail.count(fence + "text\n"), 2)
            self.assertEqual(detail.splitlines().count(fence), 2)
            self.assertEqual(detail.count(content), 2)

    def test_clear_restores_idle_controls_when_canceling_a_check(self):
        steps = reasoning_check.run_check(check.CUT, self.turns, 1, self.picked, "", 20, "", False, [])
        running = next(steps)
        self.assertTrue(running[-1]["visible"])
        cleared = reasoning_check.clear_results()
        steps.close()
        self.assertEqual(cleared[0], [])
        self.assertEqual(cleared[4], "")
        self.assertTrue(all(button["interactive"] for button in cleared[5:9]))
        self.assertFalse(cleared[-1]["visible"])
        self.assertIsNone(self.manager.occupant)

    def test_a_competing_click_leaves_the_winners_controls_unchanged(self):
        with mock.patch.object(self.manager, "claim_generation", return_value=GENERATING):
            frame = self.run_tab(check.CUT)[-1]
            self.assertEqual(frame[-5:], (gr.skip(),) * 5)
            frame = list(reasoning_check.paraphrase_reply(self.turns, 1, self.picked, 20, "", False))[-1]
            self.assertEqual(frame[-5:], (gr.skip(),) * 5)

    def test_stopping_closes_the_stream_and_gives_the_slot_back(self):
        steps = reasoning_check.run_check(check.CUT, self.turns, 1, self.picked, "", 20, "", False, [])
        next(steps)
        next(steps)
        steps.close()
        self.assertIsNone(self.manager.occupant)
        self.assertFalse(self.manager._lock.locked())


if __name__ == "__main__":
    unittest.main()
