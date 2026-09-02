import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

import model_runtime
from conversation import split_reasoning
from model_runtime import IncrementalDecoder, ModelChanged, ModelManager


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
        if _kwargs.get("add_special_tokens") is False:
            # Response-prefill tests need a small but real text-to-token path.
            # Match the supplied vocabulary greedily; ordinary prompt tests
            # keep using the single placeholder token below.
            remaining = text
            token_ids: list[int] = []
            pieces = sorted(
                (
                    (piece, index)
                    for index, piece in enumerate(self.pieces)
                    if piece
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            while remaining:
                match = next(
                    (
                        (piece, index)
                        for piece, index in pieces
                        if remaining.startswith(piece)
                    ),
                    None,
                )
                if match is None:
                    if "<unk>" in self.pieces:
                        # Real vocabularies have an unknown piece; anything it
                        # stands in for no longer decodes to what was typed.
                        token_ids.append(self.pieces.index("<unk>"))
                        remaining = remaining[1:]
                        continue
                    raise ValueError(f"No fake token for {remaining!r}")
                piece, index = match
                token_ids.append(index)
                remaining = remaining[len(piece) :]
            return Encoding(input_ids=token_ids)
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

    def __init__(self, pieces: list[bytes], eos_id: int | None = None):
        self.pieces = pieces
        self.eos_token_id = eos_id
        self.all_special_ids = [] if eos_id is None else [eos_id]

    def __call__(self, _text, **_kwargs):
        return Encoding(input_ids=[0])

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        raw = b"".join(self.pieces[int(token_id)] for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, token_id):
        return repr(self.pieces[int(token_id)])


class SentencePieceTokenizer(FakeTokenizer):
    """A SentencePiece stand-in: the word-boundary space is dropped at the start.

    Encoding prepends the dummy-prefix marker and matches pieces greedily, so
    a standalone word becomes its ``\u2581word`` piece just as SentencePiece
    makes it. Decoding turns markers back into spaces and drops the first, so
    that piece reads without a space on its own and with one after other
    tokens: ``decode(a + b)`` is not ``decode(a) + decode(b)``.
    """

    def __init__(self, pieces: list[str], eos_id: int | None = None):
        super().__init__(pieces, eos_id)
        self.all_special_ids = [] if eos_id is None else [eos_id]

    def __call__(self, text, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            text = "\u2581" + text.replace(" ", "\u2581")
        return super().__call__(text, **kwargs)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return text.replace("\u2581", " ").removeprefix(" ")


class FakeModel(torch.nn.Module):
    """Emits ``script`` one token at a time, whatever the sampler asks for.

    Every input position advances the script by one step and predicts the
    next scripted token, so a multi-token prefill chunk gets one distribution
    per position exactly as a real model would give it.
    """

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
        length = 1 if input_ids is None else int(input_ids.shape[-1])
        logits = torch.full((1, length, self.vocab_size), -20.0)
        for offset in range(length):
            logits[0, offset, self.script[self.step % len(self.script)]] = 20.0
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

    def test_stable_text_excludes_an_incomplete_character_suffix(self):
        pieces = [b"<think>", b"\xf0", b"\x9f", b"\x92", b"\xbe"]
        decoder = IncrementalDecoder(BytePieceTokenizer(pieces))
        decoder.push(0)
        decoder.push(1)

        self.assertEqual(decoder.text, "<think>\ufffd")
        self.assertEqual(decoder.stable_text, "<think>")

        for token_id in (2, 3, 4):
            decoder.push(token_id)
        self.assertEqual(decoder.text, "<think>\U0001f4be")
        self.assertEqual(decoder.stable_text, decoder.text)

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

    def test_a_forced_prefix_from_an_earlier_load_is_refused_before_any_token(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        stale = manager.load_id
        manager.load_count += 1
        stream = manager.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=4,
            seed=1,
            forced_ids=[0],
            load_id=stale,
        )
        with self.assertRaises(ModelChanged):
            next(stream)
        # The refusal released the slot it took, and the current load is
        # still accepted.
        self.assertFalse(manager.busy)
        self.assertTrue(self.collect(manager, load_id=manager.load_id))

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


class ForcedPrefixTests(unittest.TestCase):
    """A branched response replays kept tokens before it samples anything."""

    def updates(self, manager, forced, **kwargs):
        options = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 40,
            "seed": 1,
            "forced_ids": forced,
        }
        options.update(kwargs)
        # The metrics list on an update is the generator's own and keeps
        # growing, so each update is snapshotted as it arrives.
        return [
            replace(update, metrics=list(update.metrics))
            for update in manager.generate(
                [{"role": "user", "content": "hi"}], **options
            )
        ]

    def test_forced_tokens_come_first_and_are_measured_honestly(self):
        # The script predicts "Hello" then " world"; the reader kept "Hello"
        # and swapped " are" in for " world".
        manager = loaded_manager([0, 1, 2, EOS_ID])
        updates = self.updates(manager, [0, 5])

        first = updates[0]
        self.assertEqual(first.text, "Hello are")
        self.assertEqual([m["token_id"] for m in first.metrics], [0, 5])
        self.assertEqual(first.metrics[0]["raw_rank"], 1)
        # The swapped token was not what the model wanted, and its metric
        # says so instead of pretending it was sampled.
        self.assertGreater(first.metrics[1]["raw_rank"], 1)
        self.assertEqual([m["position"] for m in first.metrics], [1, 2])

        final = updates[-1]
        self.assertEqual(final.text, "Hello are!")
        self.assertEqual([m["token_id"] for m in final.metrics], [0, 5, 2, EOS_ID])
        self.assertEqual([m["position"] for m in final.metrics], [1, 2, 3, 4])

    def test_literal_replay_ranges_become_character_spans_and_metrics(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        (first, *_rest) = self.updates(
            manager,
            [0, 5],
            literal_text_ranges=((1, 2),),
        )

        self.assertEqual(
            first.literal_text_spans,
            ((len("Hello"), len("Hello are")),),
        )
        self.assertNotIn("literal_text", first.metrics[0])
        self.assertTrue(first.metrics[1]["literal_text"])

    def test_the_prefix_is_measured_under_the_sampling_settings(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        (first, *_rest) = self.updates(manager, [0, 5], temperature=0.0)
        # Greedy sampling gives the swapped token no sampling probability at
        # all, which is exactly what its sampling shift should report.
        self.assertEqual(first.metrics[1]["sampling_probability"], 0.0)
        self.assertEqual(first.metrics[0]["sampling_probability"], 1.0)

    def test_a_stop_token_inside_the_prefix_ends_the_response_there(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        updates = self.updates(manager, [0, EOS_ID, 1])
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].text, "Hello")
        self.assertEqual([m["token_id"] for m in updates[0].metrics], [0, EOS_ID])

    def test_the_token_budget_counts_only_sampled_tokens(self):
        manager = loaded_manager([0, 1, 2, 4, 5, 6])
        final = self.updates(manager, [0, 1], max_new_tokens=2)[-1]
        self.assertEqual(len(final.metrics), 4)

    def test_prompt_and_response_metrics_stay_apart(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        final = self.updates(manager, [0, 1])[-1]
        self.assertEqual([m["segment"] for m in final.prompt_metrics], ["prompt"])
        self.assertTrue(all(m["segment"] == "response" for m in final.metrics))

    def test_an_unmeasured_prompt_still_leaves_the_prefix_measured(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        final = self.updates(manager, [0, 1], analyze_prompt=False)[-1]
        self.assertEqual(final.prompt_metrics, [])
        self.assertEqual([m["token_id"] for m in final.metrics][:2], [0, 1])
        self.assertTrue(all(m["scored"] for m in final.metrics))

    def test_no_prefix_is_the_ordinary_stream(self):
        manager = loaded_manager([0, 1, EOS_ID])
        updates = self.updates(manager, [])
        self.assertEqual(updates[-1].text, "Hello world")
        self.assertEqual(len(updates[-1].metrics), 3)

    def test_answer_prefill_is_encoded_replayed_and_counted(self):
        manager = loaded_manager([0, 1, 2, EOS_ID])
        updates = self.updates(manager, [], answer_prefill="Hello world")

        self.assertEqual(updates[0].text, "Hello world")
        self.assertEqual(updates[0].forced_prefix_tokens, 2)
        self.assertEqual(updates[0].literal_prefill_text, "Hello world")
        self.assertEqual([m["token_id"] for m in updates[0].metrics], [0, 1])
        self.assertTrue(
            all(m["literal_prefill"] for m in updates[0].metrics[:2])
        )
        self.assertEqual(updates[-1].text, "Hello world!")

    def test_a_literal_eos_in_answer_prefill_is_visible_and_does_not_stop(self):
        manager = loaded_manager([0, EOS_ID, 1, 2, EOS_ID])
        updates = self.updates(
            manager, [], answer_prefill="Hello<eos> world", max_new_tokens=4
        )

        self.assertEqual(updates[0].text, "Hello<eos> world")
        self.assertEqual(updates[0].literal_prefill_text, "Hello<eos> world")
        self.assertEqual(updates[0].forced_prefix_tokens, 3)
        self.assertEqual(updates[-1].text, "Hello<eos> world!")

    def test_a_branch_inside_a_multi_token_character_uses_a_stable_literal_prefix(self):
        pieces = [
            b"prompt",
            b"<think>",
            b"\xf0",
            b"\x9f",
            b"\x92",
            b"\xbe",
            b" continued",
            b"<eos>",
        ]
        manager = ModelManager()
        manager.tokenizer = BytePieceTokenizer(pieces, eos_id=7)
        manager.model = FakeModel(
            [0, 0, 3, 4, 5, 6, 7], vocab_size=len(pieces), eos_id=7
        )
        manager.model_id = "fake/model"

        updates = self.updates(
            manager,
            [1, 2],
            literal_prefill_tokens=2,
            max_new_tokens=5,
        )

        self.assertEqual(updates[0].text, "<think>\ufffd")
        self.assertEqual(updates[0].literal_prefill_text, "<think>")
        self.assertEqual(updates[-1].text, "<think>\U0001f4be continued")

    def test_a_token_branch_and_text_prefill_are_mutually_exclusive(self):
        manager = loaded_manager([0, 1, EOS_ID])
        with self.assertRaisesRegex(ValueError, "cannot be applied together"):
            self.updates(manager, [0], answer_prefill="Hello")

    def test_a_prefill_that_the_tokenizer_normalizes_is_rejected(self):
        class NormalizingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                if kwargs.get("add_special_tokens") is False:
                    return Encoding(input_ids=[0])
                return super().__call__(text, **kwargs)

        manager = loaded_manager([0, EOS_ID])
        manager.tokenizer = NormalizingTokenizer()

        with self.assertRaisesRegex(ValueError, "represented exactly"):
            self.updates(manager, [], answer_prefill="  Hello")


SP_PIECES = ["\u2581Hello", "\u2581world", "world", "\u2581", "!", "<unk>", "<eos>"]
SP_HELLO, SP_SPACE_WORLD, SP_WORLD, SP_SPACE = 0, 1, 2, 3
SP_EOS = SP_PIECES.index("<eos>")


def sentencepiece_manager(pieces=SP_PIECES):
    manager = loaded_manager([SP_HELLO], pieces, pieces.index("<eos>"))
    manager.tokenizer = SentencePieceTokenizer(pieces, pieces.index("<eos>"))
    return manager


class ReplacementEncodingTests(unittest.TestCase):
    """Typed branch text must read, after the kept tokens, exactly as typed.

    A tokenizer that drops the word-boundary space from the first decoded
    token makes the standalone round-trip check meaningless: ``"world"``
    passes it and then gains a space once it follows ``"Hello"``, while the
    typed ``" world"`` passes it and then carries two.
    """

    def decoded(self, manager, kept, text):
        ids = manager.encode_replacement(kept, text)
        return ids, manager.tokenizer.decode(list(kept) + ids)

    def test_a_word_typed_without_a_space_does_not_gain_one(self):
        manager = sentencepiece_manager()
        ids, joined = self.decoded(manager, [SP_HELLO], "world")
        self.assertEqual(ids, [SP_WORLD])
        self.assertEqual(joined, "Helloworld")

    def test_a_typed_leading_space_appears_once(self):
        manager = sentencepiece_manager()
        ids, joined = self.decoded(manager, [SP_HELLO], " world")
        self.assertEqual(ids, [SP_SPACE_WORLD])
        self.assertEqual(joined, "Hello world")

    def test_the_first_response_token_has_no_context_to_read(self):
        manager = sentencepiece_manager()
        ids, joined = self.decoded(manager, [], "Hello")
        self.assertEqual(ids, [SP_HELLO])
        self.assertEqual(joined, "Hello")

    def test_text_no_encoding_places_exactly_is_refused(self):
        # Without a bare "world" piece nothing decodes to "Helloworld": alone
        # the word gains a space, in context it falls to the unknown piece.
        pieces = [piece for piece in SP_PIECES if piece != "world"]
        manager = sentencepiece_manager(pieces)
        with self.assertRaisesRegex(ValueError, "exactly at this position"):
            manager.encode_replacement([pieces.index("\u2581Hello")], "world")

    def test_a_joint_suffix_can_follow_noncanonical_kept_tokens(self):
        # The model sampled "Hello" as two pieces, while encoding the whole
        # text merges them into one. Its "world" suffix can still follow the
        # original kept ids exactly; those ids must remain unchanged.
        pieces = ["\u2581Hello", "\u2581Hel", "lo", "\u2581world", "world", "<unk>", "<eos>"]
        manager = sentencepiece_manager(pieces)
        kept = [pieces.index("\u2581Hel"), pieces.index("lo")]
        self.assertEqual(manager.tokenizer.decode(kept), "Hello")
        ids, joined = self.decoded(manager, kept, "world")
        self.assertEqual(ids, [pieces.index("world")])
        self.assertEqual(joined, "Helloworld")

    def test_a_joint_suffix_still_rejects_an_embedded_stop_token(self):
        pieces = [
            "\u2581Hello",
            "\u2581Hel",
            "lo",
            "\u2581<eos>",
            "<eos>",
            "Hello",
            "<unk>",
        ]
        manager = sentencepiece_manager(pieces)
        kept = [pieces.index("\u2581Hel"), pieces.index("lo")]
        with self.assertRaisesRegex(ValueError, "stop token before its end"):
            manager.encode_replacement(kept, "<eos>Hello")

    def test_a_joint_suffix_still_allows_a_terminal_stop_token(self):
        pieces = [
            "\u2581Hello",
            "\u2581Hel",
            "lo",
            "\u2581world",
            "world",
            "<eos>",
            "<unk>",
        ]
        manager = sentencepiece_manager(pieces)
        kept = [pieces.index("\u2581Hel"), pieces.index("lo")]
        self.assertEqual(
            manager.encode_replacement(kept, "world<eos>"),
            [pieces.index("world"), pieces.index("<eos>")],
        )

    def test_a_joint_suffix_still_rejects_a_hidden_special_token(self):
        pieces = [
            "\u2581Hello",
            "\u2581Hel",
            "lo",
            "\u2581<pad>",
            "<pad>",
            "<eos>",
            "<unk>",
        ]
        manager = sentencepiece_manager(pieces)
        manager.tokenizer.all_special_ids = [
            pieces.index("<pad>"),
            pieces.index("<eos>"),
        ]
        kept = [pieces.index("\u2581Hel"), pieces.index("lo")]
        with self.assertRaisesRegex(ValueError, "hidden special token"):
            manager.encode_replacement(kept, "<pad>")

    def test_a_long_rejected_replacement_decodes_only_boundary_candidates(self):
        class CountingNormalizingTokenizer:
            """One token per character, with ``z`` normalized to ``x``."""

            chat_template = None
            eos_token_id = None
            all_special_ids = []
            is_fast = False

            def __init__(self):
                self.decode_calls = 0
                self.decoded_ids = 0

            def __call__(self, text, **_kwargs):
                return Encoding(
                    input_ids=[0 if character == "a" else 1 for character in text]
                )

            def decode(self, token_ids, **_kwargs):
                self.decode_calls += 1
                self.decoded_ids += len(token_ids)
                return "".join(
                    "a" if int(token_id) == 0 else "x" for token_id in token_ids
                )

            def convert_ids_to_tokens(self, token_id):
                return ("a", "x")[int(token_id)]

        manager = loaded_manager([0], pieces=["a", "x"], eos_id=None)
        tokenizer = CountingNormalizingTokenizer()
        manager.tokenizer = tokenizer
        replacement = "a" * 4095 + "z"

        with self.assertRaisesRegex(ValueError, "exactly at this position"):
            manager.encode_replacement([0], replacement)

        # The old exhaustive suffix loop made 4,098 decodes and visited more
        # than eight million token ids here. Boundary discovery plus exact
        # validation stays linear in the pasted text instead.
        self.assertLessEqual(tokenizer.decode_calls, 20)
        self.assertLess(tokenizer.decoded_ids, len(replacement) * 6)

    def test_a_tokenizer_with_the_space_inside_the_token_is_unchanged(self):
        manager = loaded_manager([0])
        self.assertEqual(manager.encode_replacement([0], " world"), [1])
        self.assertEqual(manager.encode_replacement([0], "!\nHow"), [2, 3, 4])

    def test_empty_text_is_refused(self):
        manager = loaded_manager([0])
        with self.assertRaisesRegex(ValueError, "did not produce any tokens"):
            manager.encode_replacement([0], "")

    def test_an_unloaded_manager_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "load a model"):
            ModelManager().encode_replacement([0], "Hello")

    def test_a_hidden_non_stop_special_token_is_refused(self):
        pieces = ["Hello", "<pad>", "<eos>"]
        manager = loaded_manager([0], pieces, eos_id=2)
        manager.tokenizer.all_special_ids = [1, 2]
        with self.assertRaisesRegex(ValueError, "hidden special token"):
            manager.encode_replacement([], "<pad>")

    def test_a_stop_special_token_before_the_end_is_refused(self):
        manager = loaded_manager([0])
        with self.assertRaisesRegex(ValueError, "stop token before its end"):
            manager.encode_replacement([], "<eos>Hello")

    def test_a_terminal_stop_special_token_can_still_end_the_replacement(self):
        manager = loaded_manager([0])
        self.assertEqual(
            manager.encode_replacement([], "Hello<eos>"), [0, EOS_ID]
        )

    def test_kept_ids_from_an_earlier_load_are_refused_under_the_lock(self):
        manager = loaded_manager([0])
        stale = manager.load_id
        self.assertEqual(manager.encode_replacement([0], " world", load_id=stale), [1])
        # A same-ID reload is a new load: the kept ids belong to the old one.
        manager.load_count += 1
        with self.assertRaises(ModelChanged):
            manager.encode_replacement([0], " world", load_id=stale)
        self.assertEqual(
            manager.encode_replacement([0], " world", load_id=manager.load_id), [1]
        )


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

    def test_answer_prefill_closes_template_supplied_reasoning(self):
        pieces = ["prompt", "</think>\n\n", "The answer", " continues", "<eos>"]
        tokenizer = ChatTemplateTokenizer(
            "\nassistant: <think>", pieces=pieces, eos_id=4
        )
        manager = self.manager(tokenizer)
        manager.model = FakeModel([1, 2, 3, 4], vocab_size=len(pieces), eos_id=4)

        updates = list(
            manager.generate(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=4,
                seed=1,
                answer_prefill="The answer",
            )
        )

        self.assertEqual(updates[0].text, "</think>\n\nThe answer")
        self.assertEqual(updates[0].forced_prefix_tokens, 2)
        self.assertEqual(updates[0].literal_prefill_text, "</think>\n\nThe answer")
        self.assertTrue(
            all(m["literal_prefill"] for m in updates[0].metrics[:2])
        )
        self.assertEqual(updates[-1].text, "</think>\n\nThe answer continues")
        reasoning, answer, closed = split_reasoning(
            updates[-1].text, reasoning_prefilled=True
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "The answer continues")
        self.assertTrue(closed)

    def test_a_batch_encoding_from_the_template_yields_ids(self):
        # Transformers 5 returns a dict from apply_chat_template(tokenize=True).
        class DictTemplateTokenizer(ChatTemplateTokenizer):
            def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
                result = super().apply_chat_template(
                    messages, add_generation_prompt, **kwargs
                )
                if isinstance(result, str):
                    return result
                return {"input_ids": result, "attention_mask": [1] * len(result)}

        manager = self.manager(DictTemplateTokenizer("\nassistant: "))
        ids, _prefilled = manager._prompt_token_ids([{"role": "user", "content": "hi"}])
        self.assertEqual(ids, [0])

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
