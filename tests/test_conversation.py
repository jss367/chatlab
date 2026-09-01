import json
import unittest

from conversation import (
    REASONING_TITLE,
    SAVE_FORMAT,
    display_messages,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    split_reasoning,
    to_json,
    user_index_at_or_before,
)


class SplitReasoningTests(unittest.TestCase):
    def test_plain_text_has_no_reasoning(self):
        self.assertEqual(
            split_reasoning("Just an answer."), ("", "Just an answer.", True)
        )

    def test_extracts_a_complete_block(self):
        reasoning, answer, closed = split_reasoning(
            "<think>Weigh the options.</think>\n\nThe answer is 4."
        )
        self.assertEqual(reasoning, "Weigh the options.")
        self.assertEqual(answer, "The answer is 4.")
        self.assertTrue(closed)

    def test_open_block_is_reported_as_unclosed(self):
        reasoning, answer, closed = split_reasoning("<think>Still working")
        self.assertEqual(reasoning, "Still working")
        self.assertEqual(answer, "")
        self.assertFalse(closed)

    def test_handles_a_template_supplied_opening_tag(self):
        reasoning, answer, closed = split_reasoning(
            "Counting.</think>Four.", reasoning_prefilled=True
        )
        self.assertEqual(reasoning, "Counting.")
        self.assertEqual(answer, "Four.")
        self.assertTrue(closed)

    def test_a_lone_closing_tag_is_literal_without_the_prefilled_flag(self):
        # A model that writes about the marker instead of using it must keep
        # its whole answer: only the runtime can say the prompt prefilled one.
        text = "The marker </think> ends a reasoning block."
        reasoning, answer, closed = split_reasoning(text)
        self.assertEqual(reasoning, "")
        self.assertEqual(answer, text)
        self.assertTrue(closed)

    def test_collects_several_blocks(self):
        reasoning, answer, _ = split_reasoning("<think>one</think>A<think>two</think>B")
        self.assertEqual(reasoning, "one\n\ntwo")
        self.assertEqual(answer, "AB")

    def test_streaming_hides_a_half_written_tag(self):
        self.assertEqual(split_reasoning("Hello <th", streaming=True)[1], "Hello")
        self.assertEqual(split_reasoning("Hello <th", streaming=False)[1], "Hello <th")

    def test_a_lone_angle_bracket_survives_a_finished_response(self):
        self.assertEqual(split_reasoning("a < b")[1], "a < b")

    def test_prefilled_reasoning_stays_hidden_before_the_closing_tag(self):
        # An OLMo Think prompt ends with <think>, so the reasoning arrives with
        # no marker at all and must not be shown as an answer while it streams.
        reasoning, answer, closed = split_reasoning(
            "Let me add two and two",
            streaming=True,
            reasoning_prefilled=True,
        )
        self.assertEqual(reasoning, "Let me add two and two")
        self.assertEqual(answer, "")
        self.assertFalse(closed)

    def test_prefilled_reasoning_closes_when_the_tag_arrives(self):
        reasoning, answer, closed = split_reasoning(
            "Let me add two and two.</think>\n\nFour.",
            streaming=True,
            reasoning_prefilled=True,
        )
        self.assertEqual(reasoning, "Let me add two and two.")
        self.assertEqual(answer, "Four.")
        self.assertTrue(closed)

    def test_prefilled_reasoning_hides_a_half_written_closing_tag(self):
        reasoning, answer, closed = split_reasoning(
            "Counting.</thi", streaming=True, reasoning_prefilled=True
        )
        self.assertEqual(reasoning, "Counting.")
        self.assertEqual(answer, "")
        self.assertFalse(closed)

    def test_a_plain_prompt_never_turns_an_answer_into_reasoning(self):
        self.assertEqual(
            split_reasoning("Just an answer.", streaming=True),
            ("", "Just an answer.", True),
        )


class DisplayTests(unittest.TestCase):
    def test_reasoning_becomes_its_own_collapsible_message(self):
        turns = [
            make_turn("user", "hi"),
            make_turn("assistant", "Hello.", "Greet them."),
        ]
        messages, index_map = display_messages(turns)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["metadata"]["title"], REASONING_TITLE)
        self.assertEqual(messages[1]["metadata"]["status"], "done")
        self.assertEqual(messages[2]["content"], "Hello.")
        self.assertEqual(index_map, [(0, "content"), (1, "reasoning"), (1, "content")])

    def test_an_unfinished_block_is_marked_pending(self):
        turn = make_turn("assistant", "", "thinking")
        turn["reasoning_closed"] = False
        messages, _ = display_messages([turn])
        self.assertEqual(messages[0]["metadata"]["status"], "pending")

    def test_an_empty_reply_still_renders_a_bubble(self):
        messages, index_map = display_messages([make_turn("assistant", "")])
        self.assertEqual(messages, [{"role": "assistant", "content": ""}])
        self.assertEqual(index_map, [(0, "content")])

    def test_locate_maps_chatbot_indexes_onto_turns(self):
        turns = [
            make_turn("user", "hi"),
            make_turn("assistant", "Hello.", "Greet them."),
        ]
        self.assertEqual(locate(turns, 0), (0, "content"))
        self.assertEqual(locate(turns, 1), (1, "reasoning"))
        self.assertEqual(locate(turns, (2, 0)), (1, "content"))
        self.assertIsNone(locate(turns, 9))
        self.assertIsNone(locate(turns, None))


class HistoryLookupTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            make_turn("user", "one"),
            make_turn("assistant", "first"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]

    def test_last_user_index(self):
        self.assertEqual(last_user_index(self.turns), 2)
        self.assertIsNone(last_user_index([]))

    def test_user_index_at_or_before_walks_backwards(self):
        self.assertEqual(user_index_at_or_before(self.turns, 3), 2)
        self.assertEqual(user_index_at_or_before(self.turns, 1), 0)
        self.assertIsNone(user_index_at_or_before([make_turn("assistant", "x")], 0))


class ModelMessagesTests(unittest.TestCase):
    def test_reasoning_is_dropped_by_default(self):
        turns = [make_turn("assistant", "Hello.", "Greet them.")]
        self.assertEqual(
            model_messages(turns),
            [{"role": "assistant", "content": "Hello."}],
        )

    def test_reasoning_can_be_replayed_on_request(self):
        turns = [make_turn("assistant", "Hello.", "Greet them.")]
        content = model_messages(turns, include_reasoning=True)[0]["content"]
        self.assertIn("<think>", content)
        self.assertIn("Greet them.", content)
        self.assertTrue(content.endswith("Hello."))

    def test_system_prompt_leads_the_request(self):
        messages = model_messages(
            [make_turn("user", "hi")], system_prompt="  Be terse.  "
        )
        self.assertEqual(messages[0], {"role": "system", "content": "Be terse."})
        self.assertEqual(len(messages), 2)

    def test_blank_system_prompt_adds_nothing(self):
        self.assertEqual(
            len(model_messages([make_turn("user", "hi")], system_prompt="  ")), 1
        )

    def test_empty_turns_are_skipped(self):
        self.assertEqual(model_messages([make_turn("assistant", "")]), [])

    def test_a_reasoning_only_reply_keeps_its_slot(self):
        # Stopping a Think model mid-answer keeps a turn with reasoning but no
        # text; the request must still alternate user/assistant.
        turns = [make_turn("user", "hi"), make_turn("assistant", "", "Thinking…")]
        messages = model_messages(turns)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "")

    def test_a_reasoning_only_reply_never_yields_two_user_messages(self):
        turns = [
            make_turn("user", "one"),
            make_turn("assistant", "", "Thinking…"),
            make_turn("user", "two"),
        ]
        roles = [m["role"] for m in model_messages(turns, system_prompt="Be terse.")]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        for position, role in enumerate(roles[2:], start=2):
            self.assertNotEqual(role, roles[position - 1])

    def test_a_reasoning_only_reply_is_replayed_when_reasoning_is_kept(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "", "Thinking…")]
        messages = model_messages(turns, include_reasoning=True)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertIn("Thinking…", messages[1]["content"])


class SaveLoadTests(unittest.TestCase):
    def test_round_trip(self):
        turns = [
            make_turn("user", "hi"),
            make_turn("assistant", "Hello.", "Greet them."),
        ]
        restored, system_prompt = from_json(to_json(turns, system_prompt="Be terse."))
        self.assertEqual(restored, turns)
        self.assertEqual(system_prompt, "Be terse.")

    def test_streaming_only_keys_are_not_written(self):
        turn = make_turn("assistant", "Hello.", "Greet them.")
        turn["reasoning_closed"] = False
        payload = json.loads(to_json([turn]))
        self.assertEqual(payload["format"], SAVE_FORMAT)
        self.assertEqual(set(payload["turns"][0]), {"role", "content", "reasoning"})

    def test_rejects_files_from_elsewhere(self):
        for payload in (
            "not json",
            json.dumps({"turns": []}),
            json.dumps({"format": SAVE_FORMAT, "turns": "nope"}),
            json.dumps({"format": SAVE_FORMAT, "turns": [{"role": "root"}]}),
            json.dumps(
                {"format": SAVE_FORMAT, "turns": [{"role": "user", "content": 7}]}
            ),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    from_json(payload)


if __name__ == "__main__":
    unittest.main()
