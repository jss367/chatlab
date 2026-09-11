"""Context-menu selections must stay attached to the response they describe."""

import html
import json
import unittest
from unittest import mock

import app
import gradio as gr
from ui import runtime, token_menu
from test_app_flow import (
    FIXED, SETTINGS, THINK_EOS, THINK_PIECES, TURNS, METRICS, STATUS, TRACE,
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

    def regenerate(self, payload, settings=SETTINGS):
        action = json.dumps(dict(kind='regenerate', selection=payload['selection']))
        return list(token_menu.branch_from_menu(action, '', self.frame[TURNS], *settings))[-1]

    def test_regenerate_preserves_prefix_and_resamples_selected_token(self):
        payload = self.payload()
        original = self.frame[TURNS][-1]['tokens']
        # Make the model prefer Hello at every step, including the selected
        # position that originally held " world".
        runtime.MANAGER.model.script = [2]
        final = self.regenerate(payload)
        metrics = metrics_of(final[METRICS])
        self.assertEqual(metrics[0]['token_id'], original[0]['token_id'])
        self.assertEqual(metrics[1]['token_id'], 2)
        self.assertNotEqual(metrics[1]['token_id'], original[1]['token_id'])
        self.assertEqual(final[TRACE]['sampling']['forced_prefix_tokens'], 1)
        self.assertIn('Regenerating from token 2', final[STATUS])

    def test_regenerate_first_token_ignores_current_assistant_prefill(self):
        payload = self.payload(index=0)
        runtime.MANAGER.model.script = [3]
        settings = tuple((FIXED | {'assistant_prefill': 'Hello'}).values())
        final = self.regenerate(payload, settings)
        self.assertEqual(metrics_of(final[METRICS])[0]['token_id'], 3)
        self.assertFalse(final[TRACE]['sampling'].get('forced_prefix_tokens'))

    def test_regenerate_earlier_reply_replaces_following_turns(self):
        self.frame = list(app.chat('again', self.frame[TURNS], *SETTINGS))[-1]
        markup = token_menu.token_menu_payload(
            self.frame[TURNS], self.frame[METRICS], 'earlier',
            token_span(self.frame[TURNS], 1, turn=1),
        )
        payload = json.loads(html.unescape(markup.split('data-token-menu="')[1].split('"')[0]))
        final = self.regenerate(payload)
        self.assertEqual(len(final[TURNS]), 2)
        self.assertEqual(final[TURNS][0], self.frame[TURNS][0])
        self.assertIn('Regenerating from token 2', final[STATUS])

    def test_regenerate_is_refused_while_generation_is_busy(self):
        payload = self.payload()
        runtime.MANAGER.claim_generation()
        try:
            final = self.regenerate(payload)
        finally:
            runtime.MANAGER.release_generation()
        self.assertEqual(final[TURNS], gr.skip())
        self.assertEqual(final[STATUS], app.BUSY_STATUS)

    def test_regenerate_is_refused_after_model_reload(self):
        payload = self.payload()
        runtime.MANAGER.load_count += 1
        final = self.regenerate(payload)
        self.assertEqual(final[TURNS], self.frame[TURNS])
        self.assertIn(app.BRANCH_MODEL_CHANGED, final[STATUS])

    def test_old_menu_cannot_branch_a_retried_response(self):
        payload = self.payload()
        newer = list(app.retry_last('', self.frame[TURNS], *SETTINGS))[-1]
        for action in (dict(kind='candidate', index=1), dict(kind='text', text='Hello'), dict(kind='regenerate')):
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
