"""Every file format ChatLab reads back has a frozen file under tests/fixtures that a test reads.

A format marker ("chatlab-<name>-<n>") that appears in chatlab/ and is read back needs at least one committed
fixture carrying it, so a new format version, or a change that stops an old one reading, fails a test. See
tests/fixtures/saved/README.md for how to add one.
"""
import gzip
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures"
MARKER = re.compile(r"""["'](chatlab-[a-z-]+-\d+)["']""")

# Formats ChatLab writes for download and never reads back.
WRITE_ONLY = {
    "chatlab-activation-patching-2",
    "chatlab-component-inspection-1",
    "chatlab-hangman-batch-1",
    "chatlab-maze-batch-1",
}

# Approved evidence and the discoverable compatibility test that opens it.
# Adding marker bytes to an unrelated fixture cannot satisfy this registry.
# When adding a format, review the reader assertions in its owner test too.
COVERAGE = {
    "chatlab-attribution-graph-1": (
        "saved/circuits/graph-tokens-released/graphs/a41c7d02e9b84f6a9c3e5d7f1b2a6c80.json",
        "test_saved_formats_circuits.ReleasedTokensGraphTests.test_resaving_with_todays_writer_keeps_what_it_reads_as"),
    "chatlab-computer-safety-run-1": (
        "saved/computer_safety/judgment-run.json",
        "test_saved_formats_computer_safety.JudgmentRunTests.test_resaved_by_todays_writer_it_reads_the_same"),
    "chatlab-conversation-1": (
        "saved/conversation/01-first-890d94a.json",
        "test_saved_formats_conversation.FirstEraTests.test_resave"),
    "chatlab-direction-edits-1": (
        "saved/direction_edits/direction-edits-lens-random-injection.json",
        "test_saved_formats_direction_edits.SavedResultTests.test_resaving_with_todays_writer_keeps_the_facts"),
    "chatlab-directions-1": (
        "saved/direction_edits/directions-sparse-unnormalized.json",
        "test_saved_formats_direction_edits.SavedInputTests.test_the_direction_file_reads_as_sorted_unit_directions"),
    "chatlab-hangman-1": (
        "saved/hangman/hand-played-reopened-branched.json",
        "test_saved_formats_hangman.HandPlayedGameTests.test_resaved_by_todays_writer_it_reads_the_same"),
    "chatlab-hangman-trials-1": (
        "saved/hangman/trials.json",
        "test_saved_formats_hangman.TrialFileTests.test_the_batch_pane_reads_every_trial_with_its_settings_filled_in"),
    "chatlab-institutions-game-1": (
        "saved/institutions/invoice-pilot-run-1/games/eval/democracy/red-seed0.json.gz",
        "test_saved_institutions.InvoicePilotTests.test_the_democracy_timeline_elects_pays_rejects_and_expels"),
    "chatlab-institutions-run-1": (
        "saved/institutions/invoice-pilot-run-1/manifest.json",
        "test_saved_institutions.InvoicePilotTests.test_the_run_lists_every_game_it_holds"),
    "chatlab-institutions-game-2": (
        "institutions/support-integrity-interface-v2-20261009T004644Z-full/games/s0-peer-organisms0.json.gz",
        "test_institutions_support.SupportTests.test_all_archived_games_and_independent_validation_status"),
    "chatlab-institutions-run-2": (
        "institutions/support-integrity-interface-v2-20261009T004644Z-full/manifest.json",
        "test_institutions_support.SupportTests.test_all_archived_games_and_independent_validation_status"),
    "chatlab-library-1": (
        "saved/library/01-first-02683f2/conversations.json",
        "test_saved_formats_library.FirstEraTests.test_resave"),
    "chatlab-maze-run-1": (
        "saved/maze/run-1-before-goal-mode.json",
        "test_saved_maze.SavedMazeRunTests.test_the_first_runs_read_with_the_goal_prompts_and_window_they_were_scored_under"),
    "chatlab-maze-run-2": (
        "saved/maze/run-2-closures.json",
        "test_saved_maze.SavedMazeRunTests.test_closures_drops_and_a_queued_closure_read_back"),
    "chatlab-maze-run-3": (
        "saved/maze/run-3-inserts.json",
        "test_saved_maze.SavedMazeRunTests.test_inserted_messages_read_with_their_outcomes"),
    "chatlab-maze-team-1": (
        "saved/maze/team-1-shared-budget.json",
        "test_saved_maze.SavedMazeRunTests.test_the_first_team_runs_read_under_one_limit_for_the_whole_team"),
    "chatlab-maze-team-2": (
        "saved/maze/team-2-current.json",
        "test_saved_maze.SavedMazeRunTests.test_a_current_team_run_reads_back"),
    "chatlab-maze-team-3": (
        "saved/maze/team-3-current.json",
        "test_saved_maze.SavedMazeRunTests.test_a_current_reward_run_reads_its_recorded_rule_laps_and_limits"),
    "chatlab-maze-trials-1": (
        "saved/maze/trials-1-original.json",
        "test_saved_maze.SavedMazeRunTests.test_a_trial_run_keeps_its_stamp_and_the_trial_file_still_reads"),
    "chatlab-probe-1": (
        "saved/probes/probe-era1-no-revision-below-chance.json",
        "test_saved_formats_probes.SavedProbeTests.test_resaving_with_todays_writer_keeps_the_facts"),
    "chatlab-steering-1": (
        "saved/steering/02-extracted-be75ba8.json",
        "test_saved_formats_steering.ExtractedTests.test_round_trip"),
    "chatlab-steering-comparison-1": (
        "saved/steering_comparison/paired-ranks.json",
        "test_saved_formats_steering_comparison.SavedComparisonTests.test_resaved_by_todays_page_it_reads_the_same"),
    "chatlab-steering-reference-1": (
        "saved/library/03-steering-392ad4d/conversations.json",
        "test_saved_formats_library.SteeringEraTests.test_resave"),
}


def markers_in_source():
    return {m for path in (ROOT / "chatlab").rglob("*.py") for m in MARKER.findall(path.read_text())}


def formats_in(value):
    if isinstance(value, dict):
        if isinstance(value.get("format"), str):
            yield value["format"]
        for item in value.values():
            yield from formats_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from formats_in(item)


def fixture_formats(path):
    raw = gzip.decompress(path.read_bytes()) if path.name.endswith(".json.gz") else path.read_bytes()
    return set(formats_in(json.loads(raw)))


def owner_exists(name):
    # Loading checks real unittest discovery, including inherited methods,
    # without executing the owner or any other tests in its module.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(name)
    return not loader.errors and suite.countTestCases() == 1 and next(iter(suite)).id() == name


def missing_coverage(markers, coverage):
    return sorted(markers - WRITE_ONLY - coverage.keys())


class SavedFormatCoverageTests(unittest.TestCase):
    def test_every_format_read_back_has_a_frozen_fixture(self):
        self.assertEqual(missing_coverage(markers_in_source(), COVERAGE), [],
                         "Register an approved frozen fixture and its compatibility test in COVERAGE, "
                         "or list the format in WRITE_ONLY if nothing reads it back.")

    def test_registered_fixtures_and_owner_tests_exist(self):
        self.assertEqual(sorted(COVERAGE.keys() - (markers_in_source() - WRITE_ONLY)), [])
        for marker, (relative, owner) in COVERAGE.items():
            with self.subTest(marker=marker, fixture=relative, owner=owner):
                path = (FIXTURES / relative).resolve()
                self.assertTrue(any(root in path.parents for root in
                                    (FIXTURES / "saved", FIXTURES / "institutions")))
                self.assertIn(marker, fixture_formats(path))
                self.assertTrue(owner_exists(owner), "Compatibility test removed or no longer discoverable")

    def test_unregistered_marker_cannot_satisfy_coverage(self):
        self.assertEqual(missing_coverage({"chatlab-unrelated-99"}, COVERAGE), ["chatlab-unrelated-99"])

    def test_removed_owner_cannot_satisfy_coverage(self):
        self.assertFalse(owner_exists("test_saved_formats_steering.ExtractedTests.test_removed_reader"))

    def test_every_write_only_format_still_exists(self):
        self.assertEqual(sorted(WRITE_ONLY - markers_in_source()), [])


if __name__ == "__main__":
    unittest.main()
