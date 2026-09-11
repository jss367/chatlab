"""Context-menu selections must stay attached to the response they describe."""

import html
import json
import unittest
from unittest import mock

import app
from ui import runtime, token_menu
from test_app_flow import (
    SETTINGS, THINK_EOS, THINK_PIECES, TURNS, METRICS, STATUS,
    token_span, metrics_of,
)
from test_streaming import loaded_manager
import settings_sandbox


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class TokenMenuTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(runtime, 'MANAGER', loaded_manager(
            [2, 3, THINK_EOS], THINK_PIECES, THINK_EOS,
        ))
        patch.start()
        self.addCleanup(patch.stop)
        self.frame = list(app.chat('hi', [], *SETTINGS))[-1]

    def payload(self, index=1, request='open-1'):
        markup = token_menu.token_menu_payload(
            self.frame[TURNS], self.frame[METRICS], request,
            token_span(self.frame[TURNS], index),
        )
        return json.loads(html.unescape(markup.split('data-token-menu="')[1].split('"')[0]))

    def test_menu_keeps_identity_and_escapes_token_text(self):
        self.frame[TURNS][-1]['tokens'][1]['text'] = '<img src=x onerror="bad()">'
        payload = self.payload(request='"<request>')
        self.assertEqual(payload['text'], '<img src=x onerror="bad()">')
        self.assertEqual(payload['request'], '"<request>')
        self.assertEqual(payload['selection']['index'], 1)
        self.assertTrue(payload['candidates'])

    def test_menu_candidate_replays_the_chosen_token(self):
        payload = self.payload()
        action = json.dumps(dict(kind='candidate', index=1, selection=payload['selection']))
        final = list(token_menu.branch_from_menu(action, '', self.frame[TURNS], *SETTINGS))[-1]
        self.assertEqual(metrics_of(final[METRICS])[1]['token_id'], payload['candidates'][1]['token_id'])
        self.assertIn('Branched at token 2', final[STATUS])

    def test_custom_text_is_passed_without_stripping_spaces_or_newlines(self):
        payload = self.payload()
        action = json.dumps(dict(kind='text', text=' Hello\n', selection=payload['selection']))
        with mock.patch.object(token_menu, 'branch_with_text', return_value=iter([])) as branch:
            list(token_menu.branch_from_menu(action, '', self.frame[TURNS], *SETTINGS))
        self.assertEqual(branch.call_args.args[1], ' Hello\n')

    def test_old_menu_cannot_branch_a_retried_response(self):
        payload = self.payload()
        newer = list(app.retry_last('', self.frame[TURNS], *SETTINGS))[-1]
        for action in (dict(kind='candidate', index=1), dict(kind='text', text='Hello')):
            with self.subTest(action=action):
                action['selection'] = payload['selection']
                final = list(token_menu.branch_from_menu(json.dumps(action), '', newer[TURNS], *SETTINGS))[-1]
                self.assertIn(app.BRANCH_UNAVAILABLE, final[STATUS])
                self.assertEqual(final[TURNS], newer[TURNS])

    def test_reloaded_model_disables_menu(self):
        runtime.MANAGER.load_count += 1
        payload = self.payload()
        self.assertIsNone(payload['selection'])
        self.assertEqual(payload['error'], app.BRANCH_MODEL_CHANGED)

    def test_invalid_action_is_a_refusal(self):
        for action in ('not json', '{}', 'null'):
            final = list(token_menu.branch_from_menu(action, '', self.frame[TURNS], *SETTINGS))[-1]
            self.assertIn(app.BRANCH_UNAVAILABLE, final[STATUS])
