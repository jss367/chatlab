"""Native thinking modes change prompts and survive chat replay and export."""

import copy
import csv
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import app
import settings
import settings_sandbox
import tiny_tokenizer
from conversation import turn_entries, turns_from_entries
from model_runtime import ModelManager
from thinking import supports_thinking
from trace_export import trace_to_csv, trace_to_json
from ui import runtime
from ui.settings_page import refresh_thinking_mode
from test_app_flow import FIXED, SETTINGS, TURNS, TRACE, METRICS, click_token, cell
from test_streaming import loaded_manager


# The switching suffix from Qwen/Qwen3-0.6B's chat template. The off mode
# supplies an empty, closed think block; default/on leave generation alone.
TEMPLATE = """{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}
{% if add_generation_prompt %}assistant:
{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"""


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def manager_for_thinking():
    manager = loaded_manager([0])
    manager.tokenizer = copy.deepcopy(tiny_tokenizer.build())
    manager.tokenizer.chat_template = TEMPLATE
    token = manager.tokenizer.encode("hello")[0]
    manager.model.script = [token]
    manager.model.vocab_size = len(manager.tokenizer)
    manager.model.config = SimpleNamespace(model_type="qwen3")
    return manager


class ThinkingRuntimeTests(unittest.TestCase):
    def test_capability_needs_both_architecture_and_switchable_template(self):
        for model_type in ("qwen3", "qwen3_moe", "olmo3", "qwen3_next", None):
            for template in (TEMPLATE, "assistant: <think>", "assistant:", None):
                for config_field in ("config", "args"):
                    with self.subTest(model_type=model_type, template=template, backend=config_field):
                        model = SimpleNamespace(**{config_field: SimpleNamespace(model_type=model_type)})
                        tokenizer = SimpleNamespace(chat_template=template)
                        self.assertEqual(
                            supports_thinking(model, tokenizer),
                            model_type in ("qwen3", "qwen3_moe") and template == TEMPLATE,
                        )
        self.assertFalse(ModelManager().supports_thinking)

    def test_comments_fixed_variables_and_bad_templates_do_not_advertise_switching(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
        for template in ("{# enable_thinking #}", "{% set enable_thinking = true %}{{ enable_thinking }}", "{% broken %}"):
            self.assertFalse(supports_thinking(model, SimpleNamespace(chat_template=template)))

    def test_modes_render_and_tokenize_the_native_prompt_consistently(self):
        manager = manager_for_thinking()
        messages = [{"role": "user", "content": "hi"}]
        ids_by_mode = {}
        for mode, kwargs in (("default", {}), ("on", {"enable_thinking": True}), ("off", {"enable_thinking": False})):
            with self.subTest(mode=mode), mock.patch.object(manager.tokenizer, "apply_chat_template", wraps=manager.tokenizer.apply_chat_template) as apply:
                ids, prefilled = manager._prompt_token_ids(messages, thinking_mode=mode)
                calls = apply.call_args_list
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertEqual({k: v for k, v in call.kwargs.items() if k == "enable_thinking"}, kwargs)
                rendered = manager.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
                self.assertEqual(ids, manager.tokenizer.encode(rendered, add_special_tokens=False))
                self.assertFalse(prefilled)
                ids_by_mode[mode] = ids
        self.assertEqual(ids_by_mode["default"], ids_by_mode["on"])
        self.assertNotEqual(ids_by_mode["default"], ids_by_mode["off"])
        self.assertTrue(manager.tokenizer.decode(ids_by_mode["off"]).endswith("<think>\n\n</think>\n\n"))

    def test_generation_stamps_modes_and_uses_the_selected_prompt(self):
        for mode in ("default", "on", "off"):
            manager = manager_for_thinking()
            messages = [{"role": "user", "content": "hi"}]
            expected, _ = manager._prompt_token_ids(messages, thinking_mode=mode)
            updates = list(manager.generate(messages, temperature=0, top_p=1, top_k=0, max_new_tokens=2, seed=1, thinking_mode=mode))
            self.assertTrue(updates)
            for update in updates:
                self.assertEqual(update.thinking_mode, mode)
                self.assertEqual(update.prompt_ids, tuple(expected))

    def test_hidden_setting_is_ignored_for_an_unsupported_model(self):
        manager = manager_for_thinking()
        manager.model.config.model_type = "olmo3"
        with mock.patch.object(manager.tokenizer, "apply_chat_template", wraps=manager.tokenizer.apply_chat_template) as apply:
            update = list(manager.generate([{"role": "user", "content": "hi"}], temperature=0, top_p=1, top_k=0, max_new_tokens=1, seed=1, thinking_mode="off"))[-1]
        self.assertIsNone(update.thinking_mode)
        self.assertTrue(all("enable_thinking" not in call.kwargs for call in apply.call_args_list))

    def test_prefix_budget_uses_the_same_mode_as_generation(self):
        manager = manager_for_thinking()
        messages = [{"role": "user", "content": "hi"}]
        expected, _ = manager._prompt_token_ids(messages, thinking_mode="off")
        with mock.patch.object(manager, "_validate_generation_prefix_length") as validate:
            manager.validate_generation_prefix(messages, [0], max_new_tokens=2, thinking_mode="off")
        self.assertEqual(validate.call_args.args[0], expected)


class ThinkingChatTests(unittest.TestCase):
    def setUp(self):
        self.manager = manager_for_thinking()
        patch = mock.patch.object(runtime, "MANAGER", self.manager)
        patch.start()
        self.addCleanup(patch.stop)

    def reply(self, mode="off"):
        return list(app.chat("hi", [], **(FIXED | {"max_new_tokens": 3, "thinking_mode": mode})))[-1]

    def test_mode_survives_saved_conversation_and_json_and_csv_trace(self):
        result = self.reply()
        restored = turns_from_entries(turn_entries(result[TURNS]))
        self.assertEqual(restored[-1]["thinking_mode"], "off")
        self.assertEqual(json.loads(trace_to_json(result[TRACE]))["sampling"]["thinking_mode"], "off")
        rows = list(csv.DictReader(io.StringIO(trace_to_csv(result[TRACE]))))
        self.assertTrue(rows)
        self.assertTrue(all(row["thinking_mode"] == "off" for row in rows))

    def test_retry_uses_the_new_choice(self):
        result = self.reply("on")
        retry = list(app.regenerate_from(0, "", result[TURNS], **(FIXED | {"max_new_tokens": 2, "thinking_mode": "off"})))[-1]
        self.assertEqual(retry[TURNS][-1]["thinking_mode"], "off")
        self.assertEqual(retry[TRACE]["sampling"]["thinking_mode"], "off")

    def test_token_branches_keep_the_original_mode_when_control_changes(self):
        result = self.reply()
        selected = click_token(result, 1)
        metrics = result[METRICS]
        _, pick = app.choose_alternative(result[TURNS], metrics, app.empty_metrics(), selected, cell(0))
        self.assertIsNotNone(pick)
        controls = (*SETTINGS, None, None, None, None, "on")
        branch = list(app.branch_from(pick, "", result[TURNS], *controls))[-1]
        self.assertEqual(branch[TURNS][-1]["thinking_mode"], "off")
        self.assertEqual(branch[TRACE]["sampling"]["thinking_mode"], "off")
        with mock.patch.object(self.manager, "validate_generation_prefix", wraps=self.manager.validate_generation_prefix) as validate:
            typed = list(app.branch_with_text(selected, "hello", "", result[TURNS], *controls))[-1]
        self.assertEqual(validate.call_args.kwargs["thinking_mode"], "off")
        self.assertEqual(typed[TURNS][-1]["thinking_mode"], "off")

    def test_next_token_keeps_the_original_mode_for_validation_and_replay(self):
        result = self.reply("off")
        controls = (*SETTINGS, None, None, None, None, "on")
        with mock.patch.object(self.manager, "validate_generation_prefix", wraps=self.manager.validate_generation_prefix) as validate:
            stepped = list(app.next_token(None, "", result[TURNS], *controls))[-1]
        self.assertEqual(validate.call_args.kwargs["thinking_mode"], "off")
        self.assertEqual(stepped[TURNS][-1]["thinking_mode"], "off")
        self.assertEqual(stepped[TRACE]["sampling"]["thinking_mode"], "off")
        self.assertEqual(len(stepped[TURNS][-1]["tokens"]), len(result[TURNS][-1]["tokens"]) + 1)

    def test_visibility_follows_loaded_capability_without_resetting_choice(self):
        self.assertTrue(refresh_thinking_mode()["visible"])
        self.manager.tokenizer.chat_template = "assistant: <think>"
        self.assertFalse(refresh_thinking_mode()["visible"])
        self.assertNotIn("value", refresh_thinking_mode())

    def test_setting_survives_reload_and_invalid_values_fall_back(self):
        settings.update(thinking_mode="off")
        self.assertEqual(settings.load().thinking_mode, "off")
        for value in (None, True, [], "sometimes"):
            self.assertEqual(settings.sanitize({"thinking_mode": value}).thinking_mode, "default")

    def test_chat_and_persistence_listeners_read_the_control(self):
        demo = app.build_app()
        control = next(block for block in demo.blocks.values() if getattr(block, "label", None) == "Thinking mode")
        self.assertTrue(control.visible)
        for fn in demo.fns.values():
            if fn.fn in (app.chat, app.retry_last, app.branch_from, app.branch_with_text, app.next_token):
                self.assertIs(fn.inputs[-1], control)
            if fn.fn is app.remember_settings:
                self.assertIs(fn.inputs[app.PERSISTED_SETTING_NAMES.index("thinking_mode")], control)
