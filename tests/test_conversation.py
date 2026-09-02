import json
import unittest

from conversation import (
    MAIN_BRANCH,
    REASONING_TITLE,
    SAVE_FORMAT,
    TITLE_LIMIT,
    branch_choices,
    branch_label,
    branch_title,
    copy_forks,
    describe_branch,
    display_messages,
    fork_at,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    new_forks,
    next_branch_name,
    next_fork_name,
    short_model_name,
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


class ForkTests(unittest.TestCase):
    def turns(self):
        return [
            make_turn("user", "one"),
            make_turn("assistant", "first", "thinking"),
            make_turn("user", "two"),
            make_turn("assistant", "second"),
        ]

    def test_a_fresh_set_of_forks_has_only_the_main_branch(self):
        forks = new_forks()
        self.assertEqual(forks["active"], MAIN_BRANCH)
        self.assertEqual(list(forks["branches"]), [MAIN_BRANCH])

    def test_fork_names_skip_the_ones_in_use(self):
        forks = new_forks()
        self.assertEqual(next_fork_name(forks), "Fork 1")
        forks["branches"]["Fork 1"] = []
        forks["branches"]["Fork 3"] = []
        self.assertEqual(next_fork_name(forks), "Fork 2")

    def test_each_prefix_numbers_its_own_branches(self):
        forks = new_forks()
        forks["branches"]["Fork 1"] = []
        self.assertEqual(next_branch_name(forks, "Chat"), "Chat 1")
        forks["branches"]["Chat 1"] = []
        self.assertEqual(next_branch_name(forks, "Chat"), "Chat 2")
        self.assertEqual(next_branch_name(forks, "Fork"), "Fork 2")

    def test_copying_forks_detaches_every_turn(self):
        forks = new_forks()
        forks["branches"][MAIN_BRANCH] = self.turns()
        copied = copy_forks(forks)
        copied["branches"][MAIN_BRANCH][0]["content"] = "changed"
        self.assertEqual(forks["branches"][MAIN_BRANCH][0]["content"], "one")

    def test_copying_nothing_gives_a_fresh_set(self):
        self.assertEqual(copy_forks(None), new_forks())

    def test_no_selection_copies_the_whole_conversation(self):
        turns = self.turns()
        forked, box = fork_at(turns, None)
        self.assertEqual(forked, turns)
        self.assertIsNone(box)
        forked[0]["content"] = "changed"
        self.assertEqual(turns[0]["content"], "one")

    def test_an_assistant_message_keeps_the_conversation_through_its_turn(self):
        # The reasoning block and the answer are one turn, so clicking either
        # forks at the same place.
        turns = self.turns()
        for part in ("reasoning", "content"):
            with self.subTest(part=part):
                forked, box = fork_at(turns, (1, part))
                self.assertEqual([t["content"] for t in forked], ["one", "first"])
                self.assertIsNone(box)

    def test_a_user_message_is_handed_back_for_rewording(self):
        forked, box = fork_at(self.turns(), (2, "content"))
        self.assertEqual([t["content"] for t in forked], ["one", "first"])
        self.assertEqual(box, "two")

    def test_an_index_past_the_end_copies_everything(self):
        turns = self.turns()
        forked, box = fork_at(turns, (9, "content"))
        self.assertEqual(forked, turns)
        self.assertIsNone(box)


def measured(content, model="allenai/Olmo-3-7B-Think", prompt=100, generated=20):
    """An assistant turn as a generation leaves it: tagged with its origin."""

    turn = make_turn("assistant", content)
    turn["model"] = model
    turn["prompt_tokens"] = prompt
    turn["generated_tokens"] = generated
    return turn


class ConversationListTests(unittest.TestCase):
    """What the pane beside the chat says about each conversation."""

    def test_the_short_model_name_drops_the_organization(self):
        self.assertEqual(short_model_name("allenai/Olmo-3-7B-Think"), "Olmo-3-7B-Think")
        self.assertEqual(short_model_name("gpt2"), "gpt2")
        self.assertEqual(short_model_name("org/model/"), "model")

    def test_the_title_is_the_first_user_message_on_one_line(self):
        turns = [make_turn("user", "  Tell me\nabout   whales "), measured("Sure.")]
        self.assertEqual(branch_title(turns), "Tell me about whales")

    def test_a_long_title_is_cut_with_an_ellipsis(self):
        title = branch_title([make_turn("user", "x" * (TITLE_LIMIT + 10))])
        self.assertEqual(len(title), TITLE_LIMIT)
        self.assertTrue(title.endswith("…"))
        self.assertEqual(branch_title([make_turn("user", "x" * TITLE_LIMIT)]), "x" * TITLE_LIMIT)

    def test_an_empty_conversation_has_no_title(self):
        self.assertEqual(branch_title([]), "")
        self.assertEqual(branch_title(None), "")
        self.assertEqual(branch_title([make_turn("assistant", "hello")]), "")

    def test_the_token_count_is_the_latest_measured_exchange(self):
        # The last prompt already holds everything before it, so the last
        # measured reply is the size of the whole conversation.
        turns = [
            make_turn("user", "one"),
            measured("first", prompt=10, generated=5),
            make_turn("user", "two"),
            measured("second", prompt=30, generated=7),
        ]
        self.assertEqual(describe_branch(turns)["tokens"], 37)

    def test_an_unmeasured_reply_has_no_count(self):
        turns = [make_turn("user", "one"), make_turn("assistant", "first")]
        summary = describe_branch(turns)
        self.assertIsNone(summary["tokens"])
        self.assertEqual(summary["models"], [])
        self.assertEqual(summary["replies"], 1)

    def test_models_are_listed_once_each_most_recent_first(self):
        turns = [
            make_turn("user", "one"),
            measured("a", model="org/alpha"),
            make_turn("user", "two"),
            measured("b", model="org/beta"),
            make_turn("user", "three"),
            measured("c", model="org/alpha"),
        ]
        self.assertEqual(describe_branch(turns)["models"], ["alpha", "beta"])

    def test_models_that_share_a_name_are_told_apart_by_their_full_ids(self):
        # org-a/model and org-b/model are different models; shortening both to
        # "model" would merge them into one entry. Only the colliding pair is
        # spelled out in full - the third model keeps its short name.
        turns = [
            make_turn("user", "one"),
            measured("a", model="org-a/model"),
            make_turn("user", "two"),
            measured("b", model="org-b/model"),
            make_turn("user", "three"),
            measured("c", model="org-c/other"),
            make_turn("user", "four"),
            measured("d", model="org-a/model"),
        ]
        self.assertEqual(
            describe_branch(turns)["models"], ["org-a/model", "other", "org-b/model"]
        )

    def test_the_label_of_an_empty_conversation(self):
        self.assertEqual(branch_label(MAIN_BRANCH, []), "Main\nNo messages yet")

    def test_the_label_before_the_first_reply(self):
        turns = [make_turn("user", "Tell me about whales")]
        self.assertEqual(
            branch_label("Fork 1", turns), "Fork 1 · Tell me about whales\nNo replies yet"
        )

    def test_the_label_of_a_measured_conversation(self):
        turns = [make_turn("user", "Tell me about whales"), measured("Sure.", prompt=1200, generated=345)]
        self.assertEqual(
            branch_label(MAIN_BRANCH, turns),
            "Main · Tell me about whales\nOlmo-3-7B-Think · 1,545 tokens",
        )

    def test_the_label_of_an_unrecorded_reply(self):
        turns = [make_turn("user", "hi"), make_turn("assistant", "hello")]
        self.assertEqual(branch_label(MAIN_BRANCH, turns), "Main · hi\nModel not recorded")

    def test_a_model_without_counts_is_still_named(self):
        turn = make_turn("assistant", "hello")
        turn["model"] = "org/alpha"
        self.assertEqual(
            branch_label(MAIN_BRANCH, [make_turn("user", "hi"), turn]), "Main · hi\nalpha"
        )

    def test_the_choices_read_the_active_branch_from_the_live_turns(self):
        # The active branch's stored entry is stale by design; the live turns
        # are what the conversation state holds.
        forks = new_forks()
        forks["branches"][MAIN_BRANCH] = [make_turn("user", "stale")]
        forks["branches"]["Fork 1"] = [make_turn("user", "other")]
        live = [make_turn("user", "fresh")]
        choices = branch_choices(forks, live)
        self.assertEqual([name for _label, name in choices], [MAIN_BRANCH, "Fork 1"])
        self.assertEqual(choices[0][0], "Main · fresh\nNo replies yet")
        self.assertEqual(choices[1][0], "Fork 1 · other\nNo replies yet")

    def test_choices_for_no_forks_at_all(self):
        self.assertEqual(branch_choices(None, None), [("Main\nNo messages yet", MAIN_BRANCH)])


class SaveLoadTests(unittest.TestCase):
    def test_a_reply_keeps_its_origin_through_a_save(self):
        turns = [make_turn("user", "hi"), measured("Hello.", prompt=12, generated=3)]
        restored, _ = from_json(to_json(turns))
        self.assertEqual(restored, turns)
        payload = json.loads(to_json(turns))
        self.assertEqual(
            set(payload["turns"][1]),
            {"role", "content", "reasoning", "model", "prompt_tokens", "generated_tokens"},
        )

    def test_a_malformed_origin_is_refused(self):
        for field, value in (
            ("model", 7),
            ("prompt_tokens", "12"),
            ("generated_tokens", True),
            ("generated_tokens", -1),
            ("prompt_tokens", 1.5),
        ):
            with self.subTest(field=field, value=value):
                payload = json.dumps(
                    {
                        "format": SAVE_FORMAT,
                        "turns": [{"role": "assistant", "content": "x", field: value}],
                    }
                )
                with self.assertRaises(ValueError):
                    from_json(payload)

    def test_a_flag_is_never_written_as_a_count(self):
        turn = make_turn("assistant", "x")
        turn["generated_tokens"] = True
        self.assertNotIn("generated_tokens", json.loads(to_json([turn]))["turns"][0])

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
