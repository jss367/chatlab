import unittest
from types import SimpleNamespace

import numpy as np
import torch

import model_runtime
from model_runtime import IncrementalDecoder, ModelManager


PIECES = [
    "Hello",
    " world",
    "!",
    "\n",
    "How",
    " are",
    " you",
    "?",
    "<eos>",
]
EOS_ID = PIECES.index("<eos>")


class FakeTokenizer:
    """A whitespace-joining stand-in for a Hugging Face tokenizer."""

    chat_template = None

    def __init__(self, pieces=PIECES, eos_id=EOS_ID):
        self.pieces = pieces
        self.eos_token_id = eos_id
        self.all_special_ids = [eos_id]
        self.last_prompt = ""

    def __call__(self, text, return_tensors=None):
        self.last_prompt = text
        return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        return "".join(
            self.pieces[int(token_id)]
            for token_id in token_ids
            if not (skip_special_tokens and int(token_id) in self.all_special_ids)
        )

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[int(token_id)]


class FakeModel(torch.nn.Module):
    """Emits ``script`` one token at a time, whatever the sampler asks for."""

    def __init__(self, script, vocab_size=None, eos_id=EOS_ID):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.script = script
        self.vocab_size = vocab_size or len(PIECES)
        self.step = 0
        self.generation_config = SimpleNamespace(eos_token_id=eos_id)

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True
    ):
        logits = torch.full((1, 1, self.vocab_size), -20.0)
        logits[0, 0, self.script[self.step % len(self.script)]] = 20.0
        self.step += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


def loaded_manager(script, pieces=PIECES, eos_id=EOS_ID):
    manager = ModelManager()
    manager.tokenizer = FakeTokenizer(pieces, eos_id)
    manager.model = FakeModel(script, vocab_size=len(pieces), eos_id=eos_id)
    manager.model_id = "fake/model"
    return manager


class IncrementalDecoderTests(unittest.TestCase):
    def assert_matches_full_decode(self, token_ids, skip_ids=None):
        tokenizer = FakeTokenizer()
        decoder = IncrementalDecoder(tokenizer, skip_ids)
        kept = [i for i in token_ids if i not in (skip_ids or set())]
        for position, token_id in enumerate(token_ids, start=1):
            decoder.push(token_id)
            expected = tokenizer.decode(
                [i for i in token_ids[:position] if i not in (skip_ids or set())]
            )
            self.assertEqual(decoder.text, expected)
        self.assertEqual(decoder.text, tokenizer.decode(kept))

    def test_text_always_matches_a_full_decode(self):
        self.assert_matches_full_decode([0, 1, 2, 3, 4, 5, 6, 7])

    def test_skipped_ids_never_reach_the_text(self):
        self.assert_matches_full_decode([0, 1, EOS_ID, 2], skip_ids={EOS_ID})

    def test_a_long_unbroken_run_is_flushed(self):
        tokenizer = FakeTokenizer(pieces=["x"])
        decoder = IncrementalDecoder(tokenizer)
        count = model_runtime.DECODE_CACHE_LIMIT * 3
        for _ in range(count):
            decoder.push(0)
        self.assertEqual(decoder.text, "x" * count)


class GenerateStreamingTests(unittest.TestCase):
    def collect(self, manager, **kwargs):
        options = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 40,
            "seed": 1,
        }
        options.update(kwargs)
        updates = []
        for update in manager.generate([{"role": "user", "content": "hi"}], **options):
            updates.append((update.text, len(update.metrics)))
        return updates

    def test_updates_are_batched(self):
        manager = loaded_manager([0, 1, 2, 4, 5, 6, 7, 2])
        updates = self.collect(manager, max_new_tokens=40)
        self.assertEqual(updates[-1][1], 40)
        self.assertLessEqual(len(updates), 40 // model_runtime.STREAM_BATCH_TOKENS + 1)

    def test_the_final_update_holds_the_whole_response(self):
        manager = loaded_manager([0, 1, 2])
        text, count = self.collect(manager, max_new_tokens=6)[-1]
        self.assertEqual(text, "Hello world!Hello world!")
        self.assertEqual(count, 6)

    def test_stop_tokens_end_the_stream_and_stay_hidden(self):
        manager = loaded_manager([0, 1, EOS_ID])
        updates = self.collect(manager, max_new_tokens=40)
        self.assertEqual(updates[-1], ("Hello world", 3))

    def test_metrics_describe_every_token(self):
        manager = loaded_manager([0, 1])
        metrics = None
        for update in manager.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=4,
            seed=1,
        ):
            metrics = update.metrics
        self.assertEqual([metric["position"] for metric in metrics], [1, 2, 3, 4])
        self.assertTrue(all(metric["raw_rank"] == 1 for metric in metrics))
        self.assertTrue(all(np.isfinite(metric["surprise_bits"]) for metric in metrics))

    def test_a_system_prompt_reaches_the_prompt_text(self):
        manager = loaded_manager([0])
        manager._prompt_inputs(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertIn("Be terse.", manager.tokenizer.last_prompt)


class HiddenTokenTests(unittest.TestCase):
    def test_reasoning_markers_survive_special_token_filtering(self):
        pieces = ["<think>", "</think>", "<eos>", "ok"]
        tokenizer = FakeTokenizer(pieces=pieces)
        tokenizer.all_special_ids = [0, 1, 2]
        tokenizer.eos_token_id = 2
        manager = ModelManager()
        manager.tokenizer = tokenizer
        self.assertEqual(manager._hidden_token_ids(), {2})


if __name__ == "__main__":
    unittest.main()
