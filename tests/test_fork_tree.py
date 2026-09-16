import json
import unittest

import library
from conversation import copy_forks, drop_branch, make_turn, new_forks, put_branch
from fork_tree import comparison_html, tree_html, validate_origin
from ui.fork_tree import render_fork_tree, select_tree_branch


def example():
    forks = new_forks()
    first = [make_turn('user', 'Tell a story'), make_turn('assistant', 'The cat slept.', 'Think quietly')]
    second = [make_turn('user', 'Tell a story'), make_turn('assistant', 'The dog ran.', 'Think quickly')]
    first[-1]['generation_settings'] = {'temperature': 0.4, 'seed': 7}
    second[-1]['generation_settings'] = {'temperature': 0.8, 'seed': 7}
    put_branch(forks, 'Main', first)
    put_branch(forks, 'Fork 1', second)
    forks['origins']['Fork 1'] = {
        'parent': 'Main', 'kind': 'token', 'turn': 1, 'token': 2,
        'original': ' cat', 'replacement': ' dog', 'replacement_ids': [42],
    }
    return forks


class ForkTreeTests(unittest.TestCase):
    def test_history_and_reply_settings_survive_reload_without_token_metrics(self):
        forks = example()
        restored = library.parse(library.dump(forks))
        self.assertEqual(restored['origins'], forks['origins'])
        self.assertEqual(restored['branches']['Fork 1'][-1]['generation_settings'],
                         {'temperature': 0.8, 'seed': 7})
        self.assertIn('token 2', tree_html(restored, {}))

    def test_older_libraries_do_not_invent_ancestry(self):
        forks = example()
        forks.pop('origins')
        restored = library.parse(library.dump(forks))
        self.assertEqual(restored['origins'], {})
        self.assertIn('origin not recorded', tree_html(restored, {}))

    def test_merge_preserves_origin_when_an_older_writer_updates_a_branch(self):
        original = example()
        old_writer = copy_forks(original)
        old_writer.pop('origins')
        put_branch(old_writer, 'Fork 1', [make_turn('user', 'Edited')])
        merged = library.merge(old_writer, original)
        self.assertEqual(merged['origins'], original['origins'])
        self.assertEqual(merged['branches']['Fork 1'][0]['content'], 'Edited')

    def test_deleted_parent_keeps_child_and_origin_visible(self):
        forks = example()
        drop_branch(forks, 'Main')
        markup = tree_html(forks, {})
        self.assertIn('Parent was deleted', markup)
        self.assertIn('From Main', markup)
        self.assertIn('Fork 1', markup)

    def test_cycles_and_self_references_render_each_node_once(self):
        forks = example()
        forks['origins']['Main'] = {'parent': 'Fork 1', 'kind': 'copy'}
        markup = tree_html(forks, {})
        self.assertEqual(markup.count('<article'), 2)
        forks['origins']['Main']['parent'] = 'Main'
        self.assertEqual(tree_html(forks, {}).count('<article'), 2)

    def test_selecting_nodes_compares_settings_answers_and_reasoning(self):
        forks = example()
        turns = forks['branches']['Main']
        selected, _, _ = select_tree_branch(json.dumps({'slot': 'A', 'name': 'Main'}), turns, forks, {})
        selected, tree, comparison = select_tree_branch(json.dumps({'slot': 'B', 'name': 'Fork 1'}), turns, forks, selected)
        self.assertEqual(selected, {'A': 'Main', 'B': 'Fork 1'})
        self.assertEqual(tree.count('aria-pressed="true"'), 2)
        for text in ('Temperature', '0.4', '0.8', '<del>cat</del>', '<ins>dog</ins>', 'Reasoning', '1 shared opening message'):
            self.assertIn(text, comparison)

    def test_live_transcript_wins_over_stored_active_branch(self):
        forks = example()
        turns = [make_turn('assistant', 'Streaming answer')]
        _, comparison = render_fork_tree(turns, forks, {'A': 'Main', 'B': 'Fork 1'})
        self.assertIn('Streaming', comparison)
        self.assertNotIn('Streaming', str(forks))

    def test_same_missing_and_malformed_selections_are_safe(self):
        forks = example()
        self.assertIn('different branch', comparison_html(forks, {'A': 'Main', 'B': 'Main'}))
        self.assertIn('Select A and B', comparison_html(forks, {'A': 'gone', 'B': 'Main'}))
        for action in ('oops', '[]', '{"slot": [], "name": "Main"}', '{"slot": "A", "name": "gone"}'):
            selected, _, _ = select_tree_branch(action, [], forks, {})
            self.assertEqual(selected, {})

    def test_all_user_content_is_escaped_in_tree_and_comparison(self):
        forks = example()
        attack = '<img src=x onerror="alert(1)">'
        forks['origins']['Fork 1']['replacement'] = attack
        forks['branches']['Fork 1'][-1]['content'] = attack
        forks['sampling']['Fork 1'] = {'system_prompt': attack}
        for markup in (tree_html(forks, {}), comparison_html(forks, {'A': 'Main', 'B': 'Fork 1'})):
            self.assertNotIn('<img', markup)
            self.assertIn('&lt;img', markup)

    def test_invalid_origin_coordinates_are_rejected(self):
        for values in ({'turn': True}, {'token': -1}, {'replacement_ids': ['4']}, {'original': []}):
            with self.assertRaises(ValueError):
                validate_origin({'parent': 'Main', 'kind': 'token', **values})
