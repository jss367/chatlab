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


class Encoding(dict):
    """The subset of a Hugging Face ``BatchEncoding`` the prompt path uses."""

    @property
    def input_ids(self) -> list[int]:
        return self["input_ids"]


class FakeTokenizer:
    """A whitespace-joining stand-in for a Hugging Face tokenizer."""

    chat_template = None

    def __init__(self, pieces=PIECES, eos_id=EOS_ID):
        self.pieces = pieces
        self.eos_token_id = eos_id
        self.all_special_ids = [eos_id]
        self.last_prompt = ""

    def __call__(self, text, **_kwargs):
        self.last_prompt = text
        return Encoding(input_ids=[0])

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        return "".join(
            self.pieces[int(token_id)]
            for token_id in token_ids
            if not (skip_special_tokens and int(token_id) in self.all_special_ids)
        )

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[int(token_id)]


class BytePieceTokenizer:
    """A byte-level tokenizer, where one character can span several tokens.

    This is how GPT-2 style BPE behaves: decoding a token run that starts or
    ends inside a multi-byte character yields replacement characters.
    """

    chat_template = None

    def __init__(self, pieces: list[bytes]):
        self.pieces = pieces
        self.all_special_ids: list[int] = []

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        raw = b"".join(self.pieces[int(token_id)] for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, token_id):
        return repr(self.pieces[int(token_id)])


class SentencePieceTokenizer:
    """A SentencePiece stand-in: the word-boundary space is dropped at the start."""

    chat_template = None

    def __init__(self, pieces: list[str]):
        self.pieces = pieces
        self.all_special_ids: list[int] = []

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        text = "".join(self.pieces[int(token_id)] for token_id in token_ids)
        return text.replace("\u2581", " ").removeprefix(" ")

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
    def assert_matches_full_decode(self, token_ids, skip_ids=None, tokenizer=None):
        tokenizer = tokenizer or FakeTokenizer()
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

    def test_a_flush_does_not_split_a_multi_byte_character(self):
        """Emoji span several byte-level tokens, so a flush can land inside one."""

        emoji = "\U0001f4be"  # four UTF-8 bytes, one per token
        pieces = [b"x"] + [bytes([byte]) for byte in emoji.encode("utf-8")]
        tokenizer = BytePieceTokenizer(pieces)
        # Walk the emoji across every offset a flush could cut it at.
        for filler in range(model_runtime.DECODE_CACHE_LIMIT + 4):
            with self.subTest(filler=filler):
                run = [0] * filler + [1, 2, 3, 4]
                self.assert_matches_full_decode(
                    run * 4, tokenizer=BytePieceTokenizer(pieces)
                )
        # The fake really is context-sensitive: half an emoji does not decode.
        self.assertIn("\ufffd", tokenizer.decode([1, 2]))
        self.assertEqual(tokenizer.decode([1, 2, 3, 4]), emoji)

    def test_a_flush_keeps_the_word_boundary_space(self):
        """SentencePiece suppresses the leading space, so a flush must not restart."""

        words = [f"\u2581w{index}" for index in range(200)]
        tokenizer = SentencePieceTokenizer(words)
        self.assert_matches_full_decode(list(range(200)), tokenizer=tokenizer)

    def test_a_real_byte_level_tokenizer_round_trips(self):
        """The same invariant against GPT-2 BPE, when it is in the local cache."""

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        except (ImportError, OSError, ValueError) as error:  # pragma: no cover
            self.skipTest(f"gpt2 is not cached locally: {error}")

        text = "\U0001f3b2\U0001f9e0\U0001f501\u21a9\ufe0f\U0001f4be\U0001f4c2" * 8
        token_ids = tokenizer.encode(text)
        self.assertGreater(len(token_ids), model_runtime.DECODE_CACHE_LIMIT)

        decoder = IncrementalDecoder(tokenizer)
        for position, token_id in enumerate(token_ids, start=1):
            decoder.push(token_id)
            self.assertEqual(decoder.text, tokenizer.decode(token_ids[:position]))
        self.assertNotIn("\ufffd", decoder.text)

    def test_the_cache_stays_bounded(self):
        tokenizer = FakeTokenizer(pieces=["x"])
        decoder = IncrementalDecoder(tokenizer)
        ceiling = (
            model_runtime.DECODE_CACHE_LIMIT
            + model_runtime.DECODE_FLUSH_GRACE
            + model_runtime.DECODE_CONTEXT_TOKENS
        )
        for _ in range(model_runtime.DECODE_CACHE_LIMIT * 10):
            decoder.push(0)
            self.assertLessEqual(len(decoder._cache), ceiling)


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
        manager._prompt_token_ids(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertIn("Be terse.", manager.tokenizer.last_prompt)


class ChatTemplateTokenizer(FakeTokenizer):
    """A tokenizer whose chat template can pre-fill the opening <think> tag."""

    chat_template = "{{ messages }}"

    def __init__(self, suffix, pieces=PIECES, eos_id=EOS_ID):
        super().__init__(pieces, eos_id)
        self.suffix = suffix

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        rendered = (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages) + self.suffix
        )
        if not kwargs.get("tokenize", True):
            return rendered
        self.last_prompt = rendered
        return [0]


class PrefilledReasoningTests(unittest.TestCase):
    """Only the prompt can reveal that the opening <think> tag was supplied."""

    def manager(self, tokenizer):
        manager = ModelManager()
        manager.tokenizer = tokenizer
        manager.model = FakeModel([0], vocab_size=len(PIECES))
        return manager

    def test_a_template_ending_in_the_open_tag_is_detected(self):
        manager = self.manager(ChatTemplateTokenizer("\nassistant: <think>"))
        _ids, prefilled = manager._prompt_token_ids([{"role": "user", "content": "hi"}])
        self.assertTrue(prefilled)

    def test_a_trailing_newline_after_the_open_tag_still_counts(self):
        manager = self.manager(ChatTemplateTokenizer("\nassistant: <think>\n"))
        _ids, prefilled = manager._prompt_token_ids([{"role": "user", "content": "hi"}])
        self.assertTrue(prefilled)

    def test_an_ordinary_template_is_not_prefilled(self):
        manager = self.manager(ChatTemplateTokenizer("\nassistant: "))
        _ids, prefilled = manager._prompt_token_ids([{"role": "user", "content": "hi"}])
        self.assertFalse(prefilled)

    def test_the_transcript_fallback_is_not_prefilled(self):
        manager = self.manager(FakeTokenizer())
        _ids, prefilled = manager._prompt_token_ids([{"role": "user", "content": "hi"}])
        self.assertFalse(prefilled)

    def test_the_flag_rides_along_on_every_update(self):
        manager = self.manager(ChatTemplateTokenizer("\nassistant: <think>"))
        manager.model = FakeModel([0, EOS_ID], vocab_size=len(PIECES))
        updates = list(
            manager.generate(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=4,
                seed=1,
            )
        )
        self.assertTrue(updates)
        self.assertTrue(all(update.reasoning_prefilled for update in updates))


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
