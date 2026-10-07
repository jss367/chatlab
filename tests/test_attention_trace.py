"""Attention across the reply: the one-pass trace and the view drawn from it."""

import base64
import json
import re
import shutil
import string
import subprocess
import unittest
from unittest import mock

import numpy as np
import torch

from chatlab.model_errors import ModelChanged
from chatlab import model_inspection
from chatlab.model_runtime import ModelManager
from chatlab.ui import attention_trace, runtime
from chatlab.ui.panel import new_metrics_generation, restore_chat_metrics_generation
from fakes import FakeTokenizer, SentencePieceTokenizer, lens_manager
from mlx_support import needs_mlx, tiny_llama
from torch_support import tiny_manager


def full_attention(manager, ids) -> np.ndarray:
    """Every layer's every head over the whole sequence, from one plain pass."""

    with torch.inference_mode():
        manager.model.set_attn_implementation("eager")
        try:
            outputs = manager.model(torch.tensor([ids]), output_attentions=True)
        finally:
            manager.model.set_attn_implementation("sdpa")
    return torch.stack([layer[0] for layer in outputs.attentions]).float().numpy()


def step_rows(trace) -> list[np.ndarray]:
    rows, start = [], 0
    for step in range(len(trace.tokens) - trace.context_count):
        width = trace.context_count + step
        rows.append(trace.rows[start : start + width].astype(np.float32))
        start += width
    assert start == len(trace.rows)
    return rows


class TorchTraceTests(unittest.TestCase):
    def setUp(self):
        self.manager = tiny_manager()
        self.ids = self.manager.tokenizer.encode("the cat sat on the mat and then the dog ran far away")
        self.context = 5

    def trace(self, **options):
        return self.manager.trace_attention(
            self.ids, self.context, load_id=self.manager.load_id, **options
        )

    def test_chunked_rows_match_one_pass_over_the_whole_sequence(self):
        with mock.patch.object(model_inspection, "ATTENTION_TRACE_CHUNK_TOKENS", 3):
            trace = self.trace()
        weights = full_attention(self.manager, self.ids)

        self.assertEqual((trace.layer_count, trace.head_count), weights.shape[:2])
        self.assertEqual(len(trace.heads), trace.layer_count * trace.head_count)
        self.assertEqual(len(trace.tokens), len(self.ids))
        for step, row in enumerate(step_rows(trace)):
            query = self.context - 1 + step
            expected = weights[:, :, query, : query + 1].mean(axis=(0, 1))
            np.testing.assert_allclose(row, expected, atol=2e-3)

    def test_head_shares_leave_out_the_first_token(self):
        with mock.patch.object(model_inspection, "RECENT_KEYS", 3):
            trace = self.trace()
        weights = full_attention(self.manager, self.ids)
        steps = len(self.ids) - self.context
        keys = np.zeros(trace.key_shares.shape)
        recent = np.zeros(trace.recent_shares.shape)
        active = np.zeros(trace.active.shape)
        for step in range(steps):
            query = self.context - 1 + step
            kept = weights[:, :, query, 1 : query + 1]
            keys[..., 1 : query + 1] += kept
            recent += kept[..., max(0, query - 3) :].sum(axis=-1)
            active += kept.sum(axis=-1)

        self.assertEqual(trace.key_shares.shape[-1], len(self.ids) - 1)
        np.testing.assert_allclose(trace.key_shares, keys / active[..., None], atol=1e-4)
        np.testing.assert_allclose(trace.recent_shares, recent / active, atol=1e-4)
        np.testing.assert_allclose(trace.active, active / steps, atol=1e-4)
        np.testing.assert_allclose(trace.key_shares.sum(axis=-1), 1.0, atol=1e-4)

    def test_chosen_heads_are_the_only_ones_averaged(self):
        trace = self.trace(heads=lambda layers, heads: [(1, 2), (0, 0)])
        weights = full_attention(self.manager, self.ids)

        self.assertEqual(trace.heads, ((1, 2), (0, 0)))
        for step, row in enumerate(step_rows(trace)):
            query = self.context - 1 + step
            expected = (weights[1, 2, query, : query + 1] + weights[0, 0, query, : query + 1]) / 2
            np.testing.assert_allclose(row, expected, atol=2e-3)

    def test_a_choice_the_model_does_not_have_is_refused(self):
        def choose(layers, heads):
            raise ValueError("no such head")

        with self.assertRaisesRegex(ValueError, "no such head"):
            self.trace(heads=choose)
        with self.assertRaisesRegex(ValueError, "at least one head"):
            self.trace(heads=lambda layers, heads: [])

    def test_a_sliding_window_layer_matches_the_full_pass_in_chunks(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        self.manager = tiny_manager(
            Qwen3Config, Qwen3ForCausalLM,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=3, use_sliding_window=True,
        )
        with mock.patch.object(model_inspection, "ATTENTION_TRACE_CHUNK_TOKENS", 4):
            trace = self.trace()
        weights = full_attention(self.manager, self.ids)
        for step, row in enumerate(step_rows(trace)):
            query = self.context - 1 + step
            expected = weights[:, :, query, : query + 1].mean(axis=(0, 1))
            np.testing.assert_allclose(row, expected, atol=2e-3)
        # The sliding layer gave nothing to keys outside its window.
        last = step_rows(self.manager.trace_attention(
            self.ids, self.context, heads=lambda layers, heads: [(1, 0)], load_id=self.manager.load_id,
        ))[-1]
        self.assertTrue(np.all(last[:-3] == 0))

    def test_the_cache_is_kept_for_the_next_inspection(self):
        self.trace()
        self.assertEqual(self.manager._inspect_cache[1], self.ids[:-1])
        insight = self.manager.inspect(self.ids, len(self.ids) - 1, load_id=self.manager.load_id)
        self.assertEqual(len(insight.attention), 2)

    def test_tokens_from_another_load_and_a_missing_reply_are_refused(self):
        with self.assertRaises(ModelChanged):
            self.manager.trace_attention(self.ids, self.context, load_id="another load")
        with self.assertRaisesRegex(ValueError, "no reply"):
            self.manager.trace_attention(self.ids, len(self.ids), load_id=self.manager.load_id)
        with self.assertRaisesRegex(RuntimeError, "load a model"):
            ModelManager().trace_attention(self.ids, self.context)


class FakeModelTraceTests(unittest.TestCase):
    def test_expanded_images_belong_to_their_earlier_or_current_user_message(self):
        class ImageTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=True, **kwargs):
                def body(message):
                    content = message["content"]
                    if isinstance(content, list):
                        return "".join("<image>" if part["type"] == "image" else part["text"]
                                       for part in content)
                    return content
                text = "".join(f"<|{message['role']}|>\n{body(message)}<|end|>\n"
                               for message in messages) + "<|assistant|>"
                return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

        class Processor:
            chat_template = "native vision template"

            def apply_chat_template(self, *args, **kwargs):
                return tokenizer.apply_chat_template(*args, **kwargs)

            def __call__(self, text, images, **kwargs):
                # A processor expands each picture into three encoder slots.
                rendered = text[0].replace("<image>", "<image>" * 3)
                return {"input_ids": torch.tensor([tokenizer(rendered, add_special_tokens=False)["input_ids"]])}

        pieces = ["<|user|>", "<|end|>", "<|assistant|>", "\n", "question", "reply", "<image>"]
        pieces += list(string.ascii_letters + string.digits + "_; ")
        tokenizer = ImageTokenizer(pieces)  # Its template lives on the processor.
        manager = lens_manager([1, 2, 3])
        manager.tokenizer, manager.processor = tokenizer, Processor()
        with (mock.patch.object(runtime, "MANAGER", manager),
              mock.patch("chatlab.attachments.open_for_model", return_value=object()),
              mock.patch.object(manager, "_image_layout", return_value=None)):
            for content in ("question", ""):
                with self.subTest(content=content):
                    messages = [
                        {"role": "user", "content": "question", "images": ["earlier.png"]},
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": content, "images": ["current.png"]},
                    ]
                    ids, _, _ = manager._response_prompt(
                        messages, tools=None, thinking_mode="default", prompt_override_ids=None,
                    )
                    image_positions = [index for index, token in enumerate(ids) if token == 6]
                    manager.model.focus = image_positions[3]
                    generation = new_metrics_generation()
                    _panel, state, *_ = list(attention_trace.trace_reply(
                        (generation, [{"token_id": 5}]),
                        (generation, ids, manager.load_id, None, ["earlier.png", "current.png"]),
                        messages + [{"role": "assistant", "content": "reply"}],
                        "", "all", "", "user message", 4,
                    ))[0]
                    self.assertEqual(len(image_positions), 6)
                    self.assertEqual([state["regions"][index] for index in image_positions],
                                     ["earlier turns"] * 3 + ["user message"] * 3)
                    self.assertGreater(attention_trace.rank_heads(state, "user message")[0][2], 0.5)

    def test_template_boundaries_win_over_duplicate_preamble_and_role_marker_text(self):
        class NativeTokenizer(FakeTokenizer):
            chat_template = "native"

            def apply_chat_template(self, messages, tokenize=True, **kwargs):
                text = "preamble user;" + "".join(
                    f"<|{message['role']}|>\n{message['content']}<|end|>\n" for message in messages
                ) + "<|assistant|>"
                return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

        pieces = ["<|user|>", "<|end|>", "<|assistant|>", "\n", "user", "reply", "preamble ", "user;"]
        pieces += list(string.ascii_letters + string.digits + "_; ")
        manager = lens_manager([1, 2, 3])
        manager.tokenizer = NativeTokenizer(pieces)
        with mock.patch.object(runtime, "MANAGER", manager):
            for content in ("user", "preamble user;"):
                with self.subTest(content=content):
                    messages = [{"role": "user", "content": content}]
                    ids, _, _ = manager._response_prompt(
                        messages, tools=None, thinking_mode="default", prompt_override_ids=None,
                    )
                    generation = new_metrics_generation()
                    _panel, state, *_ = list(attention_trace.trace_reply(
                        (generation, [{"token_id": 5}]), (generation, ids, manager.load_id),
                        messages + [{"role": "assistant", "content": "reply"}],
                        "", "all", "", "user message", 4,
                    ))[0]
                    count = len(manager.tokenizer(content, add_special_tokens=False)["input_ids"])
                    expected = ["template"] * len(ids) + ["earlier reply"]
                    expected[4:4 + count] = ["user message"] * count
                    self.assertEqual(state["regions"], expected)
                    _panel, remarked, *_ = attention_trace.remark(state, "", "user message", 4)
                    self.assertEqual(remarked["regions"], expected)

    def test_sentencepiece_regions_and_remarking_use_the_complete_prompt(self):
        manager = lens_manager([1, 2, 3])
        manager.tokenizer = SentencePieceTokenizer(["▁user:", "▁hello", "▁world", "▁reply"])
        trace = manager.trace_attention([0, 1, 2, 3], 3)
        self.assertEqual(trace.prompt_text, "user: hello world")
        messages = [{"role": "user", "content": "hello world"}]
        labels = attention_trace.token_regions(
            trace.tokens, 3, messages,
            prompt_text=trace.prompt_text, prompt_spans=trace.prompt_spans,
        )
        self.assertEqual(labels, ["template", "user message", "user message", "earlier reply"])
        state = {
            "tokens": trace.tokens, "context_count": 3, "messages": messages,
            "regions": labels, "prompt_text": trace.prompt_text, "prompt_spans": trace.prompt_spans,
            "rows": trace.rows, "key_shares": trace.key_shares, "recent_shares": trace.recent_shares,
            "active": trace.active, "heads_label": "all heads",
        }
        _panel, marked, _ranking, _region = attention_trace.remark(state, "world", "marked passage", 4)
        self.assertEqual(marked["regions"][2], "marked passage")
        self.assertAlmostEqual(attention_trace.rank_heads(state, "user message")[0][2], 1.0)

    def test_every_byte_of_a_split_character_belongs_to_its_message(self):
        class ByteTokenizer(FakeTokenizer):
            def decode(self, token_ids, **kwargs):
                pieces = [b"user: ", b"\xc3", b"\xa9", b"reply"]
                return b"".join(pieces[int(index)] for index in token_ids).decode("utf-8", errors="replace")

        manager = lens_manager([1, 2, 3])
        manager.tokenizer = ByteTokenizer(["user: ", "<0xC3>", "<0xA9>", "reply"])
        trace = manager.trace_attention([0, 1, 2, 3], 3)
        self.assertEqual(trace.prompt_text, "user: é")
        self.assertEqual(trace.prompt_spans[1:], ((6, 7), (6, 7)))
        labels = attention_trace.token_regions(
            trace.tokens, 3, [{"role": "user", "content": "é"}],
            prompt_text=trace.prompt_text, prompt_spans=trace.prompt_spans,
        )
        self.assertEqual(labels, ["template", "user message", "user message", "earlier reply"])

    def test_a_model_without_attention_weights_has_nothing_to_trace(self):
        manager = lens_manager([1, 2, 3])
        manager.model.return_attentions = False
        with self.assertRaisesRegex(ValueError, "does not report its attention"):
            manager.trace_attention([0, 1, 2, 3], 2)
        self.assertIsNone(manager._inspect_cache)


@needs_mlx
class MlxTraceTests(unittest.TestCase):
    def test_rows_match_the_single_token_inspection(self):
        from chatlab.mlx_runtime import MlxEngine
        from fakes import FakeTokenizer

        model = tiny_llama()
        manager = ModelManager()
        manager.model = model
        manager.engine = MlxEngine(model, {})
        manager.tokenizer = FakeTokenizer(tuple(f"t{i}" for i in range(32)), 1)
        manager.model_id = "mlx-community/tiny-4bit"
        manager.kind = "mlx"
        ids = [3, 5, 7, 9, 11, 2, 4, 6, 8]
        trace = manager.trace_attention(ids, 4, load_id=manager.load_id)

        self.assertEqual(trace.head_count, 2)
        for step, row in enumerate(step_rows(trace)):
            index = 4 + step
            insight = manager.inspect(ids, index, load_id=manager.load_id)
            np.testing.assert_allclose(row, np.mean(insight.attention, axis=0), atol=2e-3)


class HeadParsingTests(unittest.TestCase):
    def test_all_or_nothing_means_every_head(self):
        self.assertIsNone(attention_trace.parse_heads(""))
        self.assertIsNone(attention_trace.parse_heads(" all "))

    def test_layers_ranges_and_heads_count_from_one(self):
        choose = attention_trace.parse_heads("2.3, 1 l3.h1 2.3")
        self.assertEqual(
            choose(3, 4),
            [(1, 2), (0, 0), (0, 1), (0, 2), (0, 3), (2, 0)],
        )
        self.assertEqual(attention_trace.parse_heads("2-3")(3, 1), [(1, 0), (2, 0)])

    def test_malformed_and_missing_heads_are_refused(self):
        with self.assertRaisesRegex(ValueError, "is not a head"):
            attention_trace.parse_heads("layer two")
        with self.assertRaisesRegex(ValueError, "no layer 4"):
            attention_trace.parse_heads("4")(3, 4)
        with self.assertRaisesRegex(ValueError, "no head 5"):
            attention_trace.parse_heads("1.5")(3, 4)

    def test_the_label_names_a_few_heads(self):
        self.assertEqual(attention_trace.heads_label([(0, 0), (0, 1)], 1, 2), "all 2 heads")
        label = attention_trace.heads_label([(layer, 0) for layer in range(10)], 10, 4)
        self.assertEqual(label, "10 heads: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 and 2 more")


def tokens(*pieces):
    return [{"text": piece, "token_id": index} for index, piece in enumerate(pieces)]


class RegionTests(unittest.TestCase):
    def test_legacy_region_matching_skips_role_delimiters_and_transcript_headings(self):
        self.assertEqual(attention_trace.token_regions(
            tokens("<|user|>", "\n", "user", "<|end|>"), 4,
            [{"role": "user", "content": "user"}],
        ), ["template", "template", "user message", "template"])
        self.assertEqual(attention_trace.token_regions(
            tokens("User:", " ", "User"), 3, [{"role": "user", "content": "User"}],
        ), ["template", "template", "user message"])

    MESSAGES = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Add two fractions. Also Agincourt."},
    ]
    PIECES = (
        "<s>", "system:", " Be", " brief.", "\n", "user:", " Hi", " there", "\n",
        "assistant:", " Hello", "\n", "user:", " Add", " two", " fractions.", " Also",
        " Agincourt.", "\n", "assistant:", " Sure", " thing",
    )

    def test_each_message_is_found_in_turn_and_the_rest_is_template(self):
        labels = attention_trace.token_regions(tokens(*self.PIECES), 20, self.MESSAGES)
        T, S, E, U, R = "template", "system prompt", "earlier turns", "user message", "earlier reply"
        self.assertEqual(
            labels,
            [T, T, S, S, T, T, E, E, T, T, E, T, T, U, U, U, U, U, T, T, R, R],
        )

    def test_a_marked_passage_takes_its_tokens_from_any_region(self):
        labels = attention_trace.token_regions(
            tokens(*self.PIECES), 20, self.MESSAGES, marked=" Also Agincourt. "
        )
        self.assertEqual(labels[15:19], ["user message", "marked passage", "marked passage", "template"])

    def test_a_message_the_template_rewrote_stays_template(self):
        labels = attention_trace.token_regions(
            tokens("<s>", "user:", " hi"), 3, [{"role": "user", "content": "HELLO"}]
        )
        self.assertEqual(labels, ["template"] * 3)

    def test_the_messages_are_those_before_the_latest_reply(self):
        turns = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        messages = attention_trace._messages_before_reply(turns, "rules")
        self.assertEqual([message["content"] for message in messages], ["rules", "one", "two", "three"])

    def test_recorded_system_and_reasoning_settings_reproduce_the_prompt(self):
        turns = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two", "reasoning": "earlier thought"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four", "generation_settings": {
                "system_prompt": "original rules", "keep_reasoning": True,
            }},
        ]
        messages = attention_trace._messages_before_reply(turns, "edited rules")
        self.assertEqual(messages[0], {"role": "system", "content": "original rules"})
        self.assertEqual(messages[2]["content"], "<think>\nearlier thought\n</think>\ntwo")
        pieces = tokens("system:", " original rules", " user: one assistant: ",
                        "<think>\nearlier thought\n</think>\ntwo", " user: three", " reply")
        labels = attention_trace.token_regions(pieces, 5, messages)
        self.assertEqual(labels[1], "system prompt")
        self.assertEqual(labels[3], "earlier turns")

    def test_a_recorded_empty_system_prompt_does_not_use_the_edited_control(self):
        turns = [{"role": "user", "content": "one"}, {
            "role": "assistant", "content": "two", "generation_settings": {"system_prompt": ""},
        }]
        self.assertEqual(attention_trace._messages_before_reply(turns, "new rules"),
                         [{"role": "user", "content": "one"}])


def trace_state(regions, key_shares, recent_shares, rows=None, context_count=None, active=None):
    count = context_count if context_count is not None else regions.index("earlier reply")
    return {
        "context_count": count,
        "tokens": [{"text": f"t{index}"} for index in range(len(regions))],
        "messages": [],
        "regions": regions,
        "rows": rows if rows is not None else np.zeros(0, dtype=np.float16),
        "key_shares": np.asarray(key_shares, dtype=np.float32),
        "recent_shares": np.asarray(recent_shares, dtype=np.float32),
        "active": np.full(np.asarray(recent_shares).shape, 0.5) if active is None else np.asarray(active),
        "heads_label": "all 4 heads",
    }


class RankingTests(unittest.TestCase):
    def state(self):
        # Two layers of two heads over four keys: template, user, user, reply.
        shares = np.zeros((2, 2, 4))
        shares[0, 0] = [0.0, 0.1, 0.1, 0.8]
        shares[0, 1] = [0.0, 0.4, 0.4, 0.2]
        shares[1, 0] = [0.0, 0.3, 0.0, 0.7]
        shares[1, 1] = [0.0, 0.0, 0.0, 1.0]
        recent = [[0.9, 0.1], [0.5, 0.2]]
        return trace_state(
            ["template", "user message", "user message", "earlier reply", "earlier reply"],
            shares, recent,
        )

    def test_heads_are_ranked_by_their_share_of_a_region(self):
        ranked = attention_trace.rank_heads(self.state(), "user message", 3)
        self.assertEqual([(layer, head) for layer, head, _ in ranked], [(0, 1), (1, 0), (0, 0)])
        self.assertAlmostEqual(ranked[0][2], 0.8, places=5)

    def test_the_recent_context_is_ranked_from_its_own_reading(self):
        ranked = attention_trace.rank_heads(self.state(), attention_trace.RECENT, 2)
        self.assertEqual([(layer, head) for layer, head, _ in ranked], [(0, 0), (1, 0)])

    def test_a_head_that_rests_on_the_first_token_is_not_ranked(self):
        state = self.state()
        state["active"] = np.array([[0.5, 0.001], [0.5, 0.5]])
        ranked = attention_trace.rank_heads(state, "user message", 4)
        self.assertEqual([(layer, head) for layer, head, _ in ranked], [(1, 0), (0, 0), (1, 1)])
        state["active"] = np.zeros((2, 2))
        self.assertEqual(attention_trace.rank_heads(state, "user message", 4), [])
        self.assertIn("No head puts enough attention", attention_trace.render_ranking(state, "user message", 4))

    def test_a_region_the_reply_lacks_ranks_nothing(self):
        self.assertEqual(attention_trace.rank_heads(self.state(), "system prompt", 3), [])
        self.assertIn("no system prompt", attention_trace.render_ranking(self.state(), "system prompt", 3))

    def test_the_choices_are_the_regions_present(self):
        self.assertEqual(
            attention_trace.region_choices(self.state()),
            ["user message", "template", "earlier reply", attention_trace.RECENT],
        )

    def test_the_top_heads_are_written_for_the_heads_box(self):
        self.assertEqual(attention_trace.top_heads_text(self.state(), "user message", 2), "1.2, 2.1")


class RenderTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "needs node to run the page script")
    def test_recent_total_overlaps_the_original_regions_in_the_real_page_script(self):
        payload = {
            "context": 3, "recent": 16, "recentRegion": 6,
            "names": list(attention_trace.REGIONS), "regions": [4, 0, 2, 5],
            "tokens": ["sink", "user", "system", "reply"],
            "rows": base64.b64encode(np.array([0.5, 0.25, 0.25], dtype="<f2").tobytes()).decode(),
        }
        script = r"""
const assert = require('node:assert/strict');
const element = () => ({style:{},dataset:{},handlers:{},classList:{toggle(){}},
  offsetTop:0,offsetHeight:10,clientHeight:100,scrollTop:0,
  addEventListener(name,fn){this.handlers[name]=fn;},append(){}});
const spans = Array.from({length:4}, element);
const bars = Array.from({length:7}, (_,i) => {
  const bar=element(), fill=element(), value=element();
  bar.dataset.region=String(i); bar.fill=fill; bar.value=value;
  bar.querySelector=selector => selector==='.atr-fill' ? fill : value;
  return bar;
});
const selectors = ['.atr-text','canvas','.atr-tip','.atr-sink input','.atr-range',
 '.atr-step','.atr-caption','.atr-play','.atr-prev','.atr-next'];
const nodes=Object.fromEntries(selectors.map(selector=>[selector,element()]));
nodes['canvas'].clientWidth=0;
nodes['.atr-sink input'].checked=true;
const root=element(); root.isConnected=true;
root.querySelector=selector => selector==='script.atr-data' ? {textContent:JSON.stringify(payload)} : nodes[selector];
root.querySelectorAll=selector => selector==='.atr-tok' ? spans : bars;
global.window={devicePixelRatio:1};
global.document={body:{},querySelectorAll:()=>[root],createElement:()=>element()};
global.MutationObserver=class{observe(){}};
global.ResizeObserver=class{observe(){}};
"""
        script = "const payload=" + json.dumps(payload) + ";\n" + script
        script += "\nconst start=" + attention_trace.ATTENTION_TRACE_JS + ";\nstart();\n"
        script += r"""
assert.equal(bars[0].value.textContent,'50.0%');
assert.equal(bars[2].value.textContent,'50.0%');
assert.equal(bars[6].value.textContent,'100.0%');
spans[1].handlers.mouseenter();
assert.ok(spans[1].title.startsWith('user message'));
nodes['.atr-sink input'].checked=false;
nodes['.atr-sink input'].handlers.change();
assert.equal(bars[0].value.textContent,'25.0%');
assert.equal(bars[4].value.textContent,'50.0%');
assert.equal(bars[6].value.textContent,'100.0%');
"""
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_view_carries_every_row_for_the_page_script(self):
        rows = np.array([0.5, 0.5, 0.2, 0.3, 0.5], dtype=np.float16)
        state = trace_state(
            ["template", "user message", "earlier reply", "earlier reply"],
            np.zeros((1, 1, 3)), [[0.0]], rows=rows,
        )
        state["tokens"][1]["text"] = "</script><b>"
        page = attention_trace.render_trace(state)

        data = re.search(r'<script type="application/json" class="atr-data">(.*?)</script>', page, re.S)
        payload = json.loads(data.group(1))
        decoded = np.frombuffer(base64.b64decode(payload["rows"]), dtype="<f2")
        np.testing.assert_array_equal(decoded, rows)
        self.assertEqual(payload["context"], 2)
        self.assertEqual(payload["tokens"][1], "</script><b>")
        self.assertEqual(payload["regions"], [4, 0, 5, 5])
        self.assertIn("&lt;/script&gt;&lt;b&gt;", page)
        self.assertEqual(page.count("</script>"), 1)
        self.assertNotIn("<b>", page.split('class="atr-text">')[1])
        self.assertIn('max="1"', page)
        self.assertIn("Averaging all 4 heads", page)

    def test_nothing_is_drawn_without_a_trace(self):
        self.assertEqual(attention_trace.render_trace(None), attention_trace.EMPTY_TRACE)
        self.assertEqual(attention_trace.reset_trace(None), (mock.ANY,) * 4)
        self.assertEqual(
            attention_trace.reset_trace({"x": 1}),
            (attention_trace.EMPTY_TRACE, None, attention_trace.EMPTY_RANKING, attention_trace.TRACE_HINT),
        )


class TraceHandlerTests(unittest.TestCase):
    def setUp(self):
        self.manager = tiny_manager()
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.context_ids = self.manager.tokenizer.encode("user: add two fractions\nassistant:")
        reply = self.manager.tokenizer.encode(" first find a common denominator")
        self.generation = new_metrics_generation()
        self.metrics = (self.generation, [{"token_id": token} for token in reply])
        self.context = (self.generation, self.context_ids, self.manager.load_id)
        self.turns = [
            {"role": "user", "content": "add two fractions"},
            {"role": "assistant", "content": "first find a common denominator"},
        ]

    def run_trace(self, metrics=None, context=None, heads="all", marked="", region="user message"):
        return list(attention_trace.trace_reply(
            metrics or self.metrics, context or self.context, self.turns, "", heads, marked, region, 4,
        ))

    def test_a_trace_draws_the_view_and_ranks_heads(self):
        frames = self.run_trace()
        self.assertEqual(len(frames), 1)
        panel, state, ranking, region, status = frames[0]
        self.assertIn("atr-root", panel)
        self.assertEqual(state["context_count"], len(self.context_ids))
        self.assertIn("user message", state["regions"])
        self.assertIn("Heads that read the user message most", ranking)
        self.assertEqual(region["value"], "user message")
        self.assertIn("averaging all 8 heads", status)
        # The slot went back once the frame was out.
        self.assertFalse(self.manager.claim_generation())
        self.manager.release_generation()

    def test_marking_a_passage_redraws_without_a_pass(self):
        _panel, state, *_ = self.run_trace()[0]
        with mock.patch.object(self.manager, "trace_attention", side_effect=AssertionError):
            panel, marked_state, _ranking, region = attention_trace.remark(
                state, "two fractions", "marked passage", 4
            )
        self.assertIn("marked passage", marked_state["regions"])
        self.assertEqual(region["value"], "marked passage")
        self.assertIn("atr-root", panel)

    def test_a_chosen_head_set_is_traced(self):
        _panel, state, _ranking, _region, status = self.run_trace(heads="2.1, 1.4")[0]
        self.assertIn("2 heads: 2.1, 1.4", status)
        self.assertIn("2 heads: 2.1, 1.4", state["heads_label"])

    def test_refusals_say_why(self):
        skip = mock.ANY
        empty = (0, [])
        self.assertEqual(self.run_trace(metrics=empty)[0][-1], attention_trace.TRACE_NO_REPLY)
        restore_chat_metrics_generation(self.generation + 1000)
        self.assertEqual(self.run_trace()[0][-1], attention_trace.TRACE_GONE)
        restore_chat_metrics_generation(self.generation)
        self.assertIn("is not a head", self.run_trace(heads="layer two")[0][-1])
        moved = (self.generation, self.context_ids, "another load")
        self.assertEqual(self.run_trace(context=moved)[0][-1], attention_trace.TRACE_MODEL_CHANGED)
        with mock.patch.object(attention_trace, "TRACE_VALUE_LIMIT", 10):
            self.assertIn("too long to trace", self.run_trace()[0][-1])
        self.assertEqual(self.run_trace(metrics=empty)[0][:4], (skip,) * 4)

    def test_a_busy_model_is_not_traced(self):
        self.assertFalse(self.manager.claim_generation())
        try:
            self.assertEqual(self.run_trace()[0][-1], attention_trace.TRACE_BUSY)
        finally:
            self.manager.release_generation()

    def test_a_reply_replaced_while_the_frame_travelled_is_taken_down(self):
        frames = attention_trace.trace_reply(
            self.metrics, self.context, self.turns, "", "all", "", "user message", 4,
        )
        first = next(frames)
        self.assertIn("atr-root", first[0])
        new_metrics_generation()
        self.assertEqual(next(frames)[:3], (attention_trace.EMPTY_TRACE, None, attention_trace.EMPTY_RANKING))


class PromptEditTraceTests(unittest.TestCase):
    def setUp(self):
        import prompt_edit_support
        import settings_sandbox

        self.edits = prompt_edit_support
        settings_sandbox.start()
        self.addCleanup(settings_sandbox.stop)
        self.fixture = prompt_edit_support.PromptEditFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def trace(self, frame):
        # Generation's small fake emits the edited reply; the inspection fake
        # additionally exposes attention without changing its recorded ids.
        runtime.MANAGER.model = lens_manager([1, 2, 3]).model
        runtime.MANAGER.engine = None
        frames = list(attention_trace.trace_reply(
            frame["chat_metrics"], frame["context_ids"], frame["turns"], "", "all", "",
            "user message", 4,
        ))
        self.assertIn("atr-root", frames[0][0], frames[0][-1])
        return frames[0][1]

    def test_one_message_token_replaced_by_several_keeps_its_region_and_ranking(self):
        frame = self.fixture.replace_with_text("Hello world")
        state = self.trace(frame)
        at = self.edits.MESSAGE_AT
        self.assertEqual(state["regions"][at:at + 2], ["user message"] * 2)
        self.assertTrue(attention_trace.rank_heads(state, "user message"))
        _panel, marked, _ranking, _region = attention_trace.remark(state, "world", "marked passage", 4)
        self.assertEqual(marked["regions"][at:at + 2], ["user message", "marked passage"])
        _panel, cleared, _ranking, _region = attention_trace.remark(marked, "", "user message", 4)
        self.assertEqual(cleared["regions"][at:at + 2], ["user message"] * 2)

    def test_an_edited_template_token_does_not_acquire_the_message_region(self):
        at = self.edits.PROMPT_IDS.index(4)
        frame = self.fixture.replace_with_text("Hello", index=at)
        state = self.trace(frame)
        self.assertEqual(state["regions"][at], "template")
        self.assertEqual(state["regions"][self.edits.MESSAGE_AT], "user message")

    def test_repeated_edits_in_different_regions_preserve_both_origins(self):
        first = self.fixture.replace_with_text("Hello world")
        at = first["context_ids"][1].index(4)
        payload = self.fixture.payload(index=at, frame=first)
        second = self.fixture.edit(
            {"kind": "text", "text": "Hello", "selection": payload["selection"]},
            frame=first, turns=first["turns"],
        )
        state = self.trace(second)
        self.assertEqual(state["regions"][self.edits.MESSAGE_AT:self.edits.MESSAGE_AT + 2],
                         ["user message"] * 2)
        self.assertEqual(state["regions"][at], "template")


if __name__ == "__main__":
    unittest.main()
