import json
import unittest

from chatlab.conversation import (
    ARCHIVED_VIEW,
    FAMILY_VIEW,
    MAIN_BRANCH,
    REASONING_TITLE,
    SAVE_FORMAT,
    TITLE_LIMIT,
    branch_archived,
    branch_choices,
    branch_label,
    branch_sampling,
    branch_stamp,
    branch_title,
    copy_forks,
    copy_turns,
    describe_branch,
    display_messages,
    drop_branch,
    forget_measurements,
    fork_at,
    from_json,
    last_user_index,
    locate,
    make_turn,
    model_messages,
    new_forks,
    next_branch_name,
    next_fork_name,
    put_branch,
    put_branch_archived,
    put_branch_sampling,
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


class CopyTurnsTests(unittest.TestCase):
    """What a snapshot of the turns duplicates, and what it deliberately shares."""

    def reply(self):
        return {
            "role": "assistant",
            "content": "hi",
            "reasoning": "",
            "steering": {"layer": 4, "scale": 0.5},
            "tokens": [{"token_id": 1, "top_candidates": [{"token_id": 2}]}],
            "load_id": "load-1",
        }

    def test_a_nested_value_is_copied(self):
        turns = [self.reply()]
        copied = copy_turns(turns)
        copied[0]["steering"]["layer"] = 999
        self.assertEqual(turns[0]["steering"]["layer"], 4)

    def test_the_measurements_are_shared_rather_than_duplicated(self):
        # Copying them would mean copying every measurement in the
        # conversation on every streaming frame, at a cost that grows with the
        # square of the reply's length. Nothing edits a metric after
        # build_metric writes it, so the copies can share them.
        turns = [self.reply()]
        copied = copy_turns(turns)
        self.assertIs(copied[0]["tokens"][0], turns[0]["tokens"][0])

    def test_the_list_of_measurements_is_still_its_own(self):
        # Shared metrics, but not a shared list: a turn that gains or loses
        # tokens must not change one that was copied from it.
        turns = [self.reply()]
        copied = copy_turns(turns)
        copied[0]["tokens"].append({"token_id": 9})
        self.assertEqual(len(turns[0]["tokens"]), 1)

    def test_a_turn_with_no_measurements_is_unchanged(self):
        turns = [make_turn("user", "hi")]
        copied = copy_turns(turns)
        self.assertEqual(copied, turns)
        self.assertIsNot(copied[0], turns[0])


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

    def test_names_the_file_has_spoken_for_are_stepped_over_too(self):
        forks = new_forks()
        self.assertEqual(next_branch_name(forks, "Chat", {"Chat 1", "Fork 1"}), "Chat 2")
        self.assertEqual(next_fork_name(forks, {"Chat 1", "Fork 1"}), "Fork 2")
        forks["branches"]["Chat 2"] = []
        self.assertEqual(next_branch_name(forks, "Chat", {"Chat 1"}), "Chat 3")

    def test_copying_forks_detaches_every_turn(self):
        forks = new_forks()
        forks["branches"][MAIN_BRANCH] = self.turns()
        copied = copy_forks(forks)
        copied["branches"][MAIN_BRANCH][0]["content"] = "changed"
        self.assertEqual(forks["branches"][MAIN_BRANCH][0]["content"], "one")

    def test_copying_nothing_gives_a_fresh_set(self):
        self.assertEqual(copy_forks(None), new_forks())

    def test_putting_a_branch_stamps_it_only_when_what_is_saved_changes(self):
        forks = new_forks()
        put_branch(forks, MAIN_BRANCH, self.turns())
        first = forks["updated"][MAIN_BRANCH]

        # The same turns again, with a measurement the file does not keep.
        again = self.turns()
        again[0]["surprise"] = 0.5
        put_branch(forks, MAIN_BRANCH, again)
        self.assertEqual(forks["updated"][MAIN_BRANCH], first)

        put_branch(forks, MAIN_BRANCH, self.turns()[:1])
        self.assertGreater(forks["updated"][MAIN_BRANCH], first)
        self.assertEqual(len(forks["branches"][MAIN_BRANCH]), 1)

    def test_a_new_branch_is_stamped_and_copied(self):
        forks = new_forks()
        turns = self.turns()
        put_branch(forks, "Fork 1", turns)
        turns[0]["content"] = "changed"
        self.assertEqual(forks["branches"]["Fork 1"][0]["content"], "one")
        self.assertIn("Fork 1", forks["updated"])

    def test_dropping_a_branch_leaves_the_time_it_went(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [])
        made = forks["updated"]["Fork 1"]
        drop_branch(forks, "Fork 1")
        self.assertEqual(list(forks["branches"]), [MAIN_BRANCH])
        self.assertGreater(forks["updated"]["Fork 1"], made)

    def test_the_stamp_orders_as_a_string(self):
        first = branch_stamp()
        second = branch_stamp()
        self.assertGreaterEqual(second, first)
        self.assertTrue(first.endswith("+00:00"))

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


class BranchSamplingTests(unittest.TestCase):
    """The sampling a conversation carries of its own."""

    SAMPLING = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "skip_top_below": 0.0,
        "max_new_tokens": 256,
    }

    def test_a_branch_carries_none_until_it_is_given_some(self):
        forks = new_forks()
        self.assertEqual(forks["sampling"], {})
        self.assertEqual(branch_sampling(forks, MAIN_BRANCH), {})
        self.assertEqual(branch_sampling(None, MAIN_BRANCH), {})

    def test_giving_a_branch_sampling_stamps_it_under_its_own_time(self):
        # Not the turns' stamp: two pages can have one conversation open, and
        # a page that moves a slider may be a reply behind the other. Sharing
        # one stamp would have it win the whole branch and take the newer
        # reply off the file with it.
        forks = new_forks()
        put_branch(forks, MAIN_BRANCH, [make_turn("user", "one")])
        turns_stamp = forks["updated"][MAIN_BRANCH]

        self.assertTrue(put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING))

        self.assertEqual(branch_sampling(forks, MAIN_BRANCH), self.SAMPLING)
        self.assertIn(MAIN_BRANCH, forks["sampling_updated"])
        self.assertEqual(forks["updated"][MAIN_BRANCH], turns_stamp)

    def test_the_same_sampling_again_changes_nothing(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING)
        stamp = forks["sampling_updated"][MAIN_BRANCH]

        self.assertFalse(put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING))
        self.assertEqual(forks["sampling_updated"][MAIN_BRANCH], stamp)

    def test_what_is_written_is_what_is_stored(self):
        # The caller decides: the interface passes the four this version
        # knows, and a fork passes those plus any key a newer version wrote
        # on the conversation it came from. The file layer is what checks
        # the types of the four - see library.sampling_entry.
        forks = new_forks()
        put_branch_sampling(
            forks, MAIN_BRANCH, self.SAMPLING | {"repetition_penalty": 1.15}
        )
        self.assertEqual(
            branch_sampling(forks, MAIN_BRANCH),
            self.SAMPLING | {"repetition_penalty": 1.15},
        )

    def test_a_key_a_newer_version_wrote_survives_a_slider_moved_here(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, {"repetition_penalty": 1.15})

        put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING)

        self.assertEqual(
            branch_sampling(forks, MAIN_BRANCH),
            self.SAMPLING | {"repetition_penalty": 1.15},
        )

    def test_reading_it_hands_back_a_copy(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING)
        held = branch_sampling(forks, MAIN_BRANCH)
        held["temperature"] = 1.9
        self.assertEqual(forks["sampling"][MAIN_BRANCH]["temperature"], 0.0)

    def test_copying_the_forks_copies_it(self):
        forks = new_forks()
        put_branch_sampling(forks, MAIN_BRANCH, self.SAMPLING)
        copied = copy_forks(forks)
        copied["sampling"][MAIN_BRANCH]["temperature"] = 1.9
        self.assertEqual(forks["sampling"][MAIN_BRANCH]["temperature"], 0.0)

    def test_a_deleted_branch_takes_its_sampling_with_it(self):
        forks = new_forks()
        put_branch(forks, "Fork 1", [])
        put_branch_sampling(forks, "Fork 1", self.SAMPLING)

        drop_branch(forks, "Fork 1")

        self.assertEqual(forks["sampling"], {})
        self.assertEqual(forks["sampling_updated"], {})
        self.assertEqual(branch_sampling(forks, "Fork 1"), {})


def measured(content, model="allenai/Olmo-3-7B-Think", prompt=100, generated=20):
    """An assistant turn as a generation leaves it: tagged with its origin."""

    turn = make_turn("assistant", content)
    turn["model"] = model
    turn["prompt_tokens"] = prompt
    turn["generated_tokens"] = generated
    return turn


class ArchiveTests(unittest.TestCase):
    def forks(self) -> dict:
        forks = new_forks()
        put_branch(forks, "Chat 1", [make_turn("user", "one")])
        put_branch(forks, "Chat 2", [make_turn("user", "two")])
        return forks

    def test_archiving_takes_a_conversation_out_of_the_list_and_into_the_archive(self):
        forks = self.forks()

        self.assertTrue(put_branch_archived(forks, "Chat 1", True))

        self.assertTrue(branch_archived(forks, "Chat 1"))
        self.assertEqual([name for _label, name in branch_choices(forks, [])], [MAIN_BRANCH, "Chat 2"])
        forks[ARCHIVED_VIEW] = True
        self.assertEqual([name for _label, name in branch_choices(forks, [])], ["Chat 1"])

    def test_archiving_keeps_the_conversation_and_stamps_only_the_archiving(self):
        forks = self.forks()
        updated = dict(forks["updated"])

        put_branch_archived(forks, "Chat 1", True)

        self.assertEqual(forks["branches"]["Chat 1"][0]["content"], "one")
        self.assertEqual(forks["updated"], updated)
        self.assertIn("Chat 1", forks["archived_updated"])

    def test_bringing_one_back_restores_it_to_the_list(self):
        forks = self.forks()
        put_branch_archived(forks, "Chat 1", True)

        self.assertTrue(put_branch_archived(forks, "Chat 1", False))

        self.assertFalse(branch_archived(forks, "Chat 1"))
        self.assertIn("Chat 1", [name for _label, name in branch_choices(forks, [])])

    def test_the_main_conversation_and_a_missing_one_are_never_archived(self):
        forks = self.forks()

        self.assertFalse(put_branch_archived(forks, MAIN_BRANCH, True))
        self.assertFalse(put_branch_archived(forks, "Chat 9", True))
        self.assertEqual((forks["archived"], forks["archived_updated"]), ({}, {}))

    def test_archiving_twice_changes_nothing_the_second_time(self):
        forks = self.forks()
        put_branch_archived(forks, "Chat 1", True)
        stamp = forks["archived_updated"]["Chat 1"]

        self.assertFalse(put_branch_archived(forks, "Chat 1", True))
        self.assertEqual(forks["archived_updated"]["Chat 1"], stamp)

    def test_the_view_and_the_archive_survive_a_copy(self):
        forks = self.forks()
        put_branch_archived(forks, "Chat 1", True)
        forks[ARCHIVED_VIEW] = True

        copied = copy_forks(forks)

        self.assertTrue(copied[ARCHIVED_VIEW])
        self.assertEqual(copied["archived"], {"Chat 1": True})
        copied["archived"].clear()
        self.assertTrue(branch_archived(forks, "Chat 1"))

    def test_deleting_an_archived_conversation_forgets_that_it_was(self):
        forks = self.forks()
        put_branch_archived(forks, "Chat 1", True)

        drop_branch(forks, "Chat 1")

        self.assertEqual((forks["archived"], forks["archived_updated"]), ({}, {}))


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

    def test_rewriting_a_reply_forgets_what_its_counts_measured(self):
        turns = [
            make_turn("user", "one"),
            measured("first", prompt=10, generated=5),
            make_turn("user", "two"),
            measured("second", prompt=30, generated=7),
            make_turn("user", "three"),
            measured("third", prompt=50, generated=9),
        ]
        result = forget_measurements(turns, 3)
        # The edited reply's counts measured text that is gone; its model stays.
        self.assertEqual(set(result[3]), {"role", "content", "reasoning", "model"})
        # A later reply's prompt held the old text; its own text did not change.
        self.assertNotIn("prompt_tokens", result[5])
        self.assertEqual(result[5]["generated_tokens"], 9)
        self.assertEqual(result[5]["model"], "allenai/Olmo-3-7B-Think")
        # Earlier replies are untouched, and the size falls back to the last.
        self.assertEqual(result[1], turns[1])
        self.assertEqual(describe_branch(result)["tokens"], 15)
        # The input was not mutated.
        self.assertEqual(turns[3]["prompt_tokens"], 30)
        self.assertEqual(turns[5]["prompt_tokens"], 50)

    def test_rewriting_a_reply_forgets_the_measurements_after_it_too(self):
        # A later reply's own text is untouched, so its token count is still a
        # true count. Its distributions are not: they were produced from a
        # transcript the edit replaced, and replaying its tokens onto the
        # edited conversation would force a reply the model never gave.
        tokens = [{"token_id": 1}]
        turns = [
            make_turn("user", "one"),
            dict(measured("first", prompt=10, generated=5), tokens=tokens, load_id="a"),
            make_turn("user", "two"),
            dict(measured("second", prompt=30, generated=7), tokens=tokens, load_id="a"),
        ]
        result = forget_measurements(turns, 1)
        for reply in (result[1], result[3]):
            self.assertNotIn("tokens", reply)
            self.assertNotIn("load_id", reply)
            self.assertNotIn("metrics_generation", reply)
        self.assertEqual(result[3]["generated_tokens"], 7)
        # The input was not mutated.
        self.assertEqual(turns[1]["tokens"], tokens)

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

    @staticmethod
    def family(count=3):
        """Main with ``count`` forks of it, each answered at its own length."""

        forks = new_forks()
        forks["branches"][MAIN_BRANCH] = [make_turn("user", "bored"), measured("Read.", generated=10)]
        for index in range(1, count + 1):
            name = f"Fork {index}"
            forks["branches"][name] = [make_turn("user", "bored"), measured("Walk.", generated=index)]
            forks["origins"][name] = {"parent": MAIN_BRANCH, "kind": "copy"}
        return forks

    def test_a_closed_family_shows_its_head_and_how_many_forks_it_has(self):
        forks = self.family()
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual(
            choices, [("Main · bored\nOlmo-3-7B-Think · 110 tokens · 3 forks", MAIN_BRANCH)]
        )

    def test_a_closed_family_still_shows_the_fork_on_screen(self):
        forks = self.family()
        forks["active"] = "Fork 2"
        choices = branch_choices(forks, forks["branches"]["Fork 2"])
        self.assertEqual([name for _label, name in choices], [MAIN_BRANCH, "Fork 2"])

    def test_an_open_family_lists_its_forks_on_one_line_each(self):
        # The title and model are the head's, so a fork's row leaves them out.
        forks = self.family()
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True}
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual(
            [label for label, _name in choices[1:]],
            ["↳ Fork 1 · 101 tokens", "↳ Fork 2 · 102 tokens", "↳ Fork 3 · 103 tokens"],
        )

    def test_a_fork_row_names_what_differs_from_its_head(self):
        forks = self.family(1)
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True}
        forks["branches"]["Fork 1"] = [make_turn("user", "tired"), measured("Nap.", model="org/beta")]
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual(choices[1][0], "↳ Fork 1 · tired · beta · 120 tokens")

    def test_a_fork_of_a_fork_joins_the_family_of_the_first(self):
        forks = self.family(1)
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True}
        forks["branches"]["Fork 2"] = [make_turn("user", "bored")]
        forks["origins"]["Fork 2"] = {"parent": "Fork 1", "kind": "copy"}
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual([name for _label, name in choices], [MAIN_BRANCH, "Fork 1", "Fork 2"])
        self.assertIn("2 forks", choices[0][0])

    def test_a_fork_whose_parent_is_gone_heads_its_own_family(self):
        forks = self.family(1)
        forks["origins"]["Fork 1"]["parent"] = "Fork 9"
        put_branch_archived(forks, MAIN_BRANCH, False)
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual([name for _label, name in choices], [MAIN_BRANCH, "Fork 1"])
        self.assertFalse(choices[1][0].startswith("↳"))

    def test_a_loop_of_parents_lists_each_branch_once(self):
        # Each origin is valid on its own, so a file can name A as B's parent
        # and B as A's; a fork of either still joins the one family.
        forks = self.family(2)
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True, "Fork 1": True, "Fork 2": True}
        forks["origins"]["Fork 1"]["parent"] = "Fork 2"
        forks["origins"]["Fork 2"]["parent"] = "Fork 1"
        forks["branches"]["Fork 3"] = [make_turn("user", "bored")]
        forks["origins"]["Fork 3"] = {"parent": "Fork 2", "kind": "copy"}
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH])
        self.assertEqual(
            [name for _label, name in choices], [MAIN_BRANCH, "Fork 1", "Fork 2", "Fork 3"]
        )
        self.assertIn("2 forks", choices[1][0])

    def test_a_hidden_fork_answering_is_named_on_its_head(self):
        forks = self.family()
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH], running="Fork 3")
        self.assertTrue(choices[0][0].endswith("3 forks · Fork 3 generating…"))
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True}
        choices = branch_choices(forks, forks["branches"][MAIN_BRANCH], running="Fork 3")
        self.assertTrue(choices[0][0].endswith("3 forks"))
        self.assertEqual(choices[3][0], "↳ Fork 3 · 103 tokens · Generating…")

    def test_the_family_view_survives_a_copy(self):
        forks = self.family()
        forks[FAMILY_VIEW] = {MAIN_BRANCH: True}
        self.assertEqual(copy_forks(forks)[FAMILY_VIEW], {MAIN_BRANCH: True})
        self.assertNotIn(FAMILY_VIEW, copy_forks(self.family()))


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

    def test_editing_an_earlier_reply_preserves_later_invisible_assistant_slots(self):
        paused = dict(make_turn("assistant", ""), token_step_paused=True)
        turns = [make_turn("user", "one"), make_turn("assistant", "first"),
                 make_turn("user", "two"), paused]
        edited = forget_measurements(turns, 1)
        self.assertEqual(model_messages(edited)[-1], {"role": "assistant", "content": ""})
        self.assertNotIn("token_step_paused", forget_measurements(turns, 3)[-1])

    def test_a_failed_reply_keeps_its_failure_in_the_file(self):
        failed = dict(make_turn("assistant", "Half"), error="There is no Stream(gpu, 1) in current thread.")
        turns = [make_turn("user", "one"), failed]
        restored, _ = from_json(to_json(turns))
        self.assertEqual(restored[1]["error"], failed["error"])
        messages, index_map = display_messages(restored)
        self.assertEqual(index_map, [(0, "content"), (1, "content"), (1, "error")])
        self.assertEqual(messages[1]["content"], "Half")
        self.assertIn("Generation failed: There is no Stream(gpu, 1)", messages[2]["content"])
        # The notice is the screen's; the model reads the reply's text alone.
        self.assertEqual(model_messages(restored)[-1], {"role": "assistant", "content": "Half"})
        with self.assertRaises(ValueError):
            from_json(json.dumps({"format": SAVE_FORMAT, "turns": [
                {"role": "assistant", "content": "", "error": 3}
            ]}))

    def test_a_failure_notice_is_escaped(self):
        empty = dict(make_turn("assistant", ""), error="no <pad> token")
        messages, index_map = display_messages([empty])
        # Nothing arrived, so the notice is the reply's only message.
        self.assertEqual(index_map, [(0, "error")])
        self.assertIn("no &lt;pad&gt; token", messages[0]["content"])

    def test_rejects_a_nonboolean_paused_marker(self):
        for value in ("false", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                from_json(json.dumps({"format": SAVE_FORMAT, "turns": [
                    {"role": "assistant", "content": "", "token_step_paused": value}
                ]}))

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
