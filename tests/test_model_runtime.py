import re
import unittest

from model_runtime import (
    ModelManager,
    encode_for_scoring,
    split_context_and_text,
    validate_model_id,
)


class ModelIdTests(unittest.TestCase):
    def test_accepts_hugging_face_model_id(self):
        self.assertEqual(
            validate_model_id(" allenai/Olmo-3-7B-Think "),
            "allenai/Olmo-3-7B-Think",
        )

    def test_rejects_local_and_incomplete_paths(self):
        for value in ("Olmo-3-7B-Think", "../model", "/tmp/model", "owner/model/extra"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_model_id(value)


class Encoding(dict):
    """The subset of a Hugging Face ``BatchEncoding`` the split helper uses."""

    @property
    def input_ids(self) -> list[int]:
        return self["input_ids"]


class FakeTokenizer:
    """Merges each run of non-space characters into one token, as BPE would.

    That merging is the whole point: it makes ``tokenize(a) + tokenize(b)``
    differ from ``tokenize(a + b)`` whenever the seam lands mid-run.
    """

    def __init__(
        self,
        *,
        is_fast: bool = True,
        trailing_specials: int = 0,
        chat_template: str | None = None,
    ):
        self.is_fast = is_fast
        self.trailing_specials = trailing_specials
        self.chat_template = chat_template
        self.vocab: dict[str, int] = {"<s>": 0, "</s>": 1, "<|user|>": 2, "<|assistant|>": 3}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if return_offsets_mapping and not self.is_fast:
            raise NotImplementedError("offset mapping needs a fast tokenizer")
        ids = [0] if add_special_tokens else []
        offsets = [(0, 0)] if add_special_tokens else []
        for match in re.finditer(r"\s+|\S+", text):
            ids.append(self._id(match.group()))
            offsets.append(match.span())
        if add_special_tokens:
            # A post-processor appends its closing specials after the text,
            # each carrying the empty span every special token carries.
            ids.extend([self.vocab["</s>"]] * self.trailing_specials)
            offsets.extend([(0, 0)] * self.trailing_specials)
        encoding = Encoding(input_ids=ids)
        if return_offsets_mapping:
            encoding["offset_mapping"] = offsets
        return encoding


    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [self.vocab["<|user|>"]]
        for message in messages:
            ids.extend(self._id(match.group()) for match in re.finditer(r"\s+|\S+", message["content"]))
        if add_generation_prompt:
            ids.append(self.vocab["<|assistant|>"])
        return ids


class ContextSplitTests(unittest.TestCase):
    def test_the_seam_is_tokenized_as_one_passage(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids = split_context_and_text(tokenizer, "foo", "bar")

        # "foobar" merges into a single token, so scoring the two halves
        # encoded apart would measure a sequence the passage never produces.
        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_clean_seam_keeps_every_context_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids = split_context_and_text(tokenizer, "foo ", "bar")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_an_empty_context_still_carries_the_special_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids = split_context_and_text(tokenizer, "", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_trailing_special_token_is_not_scored_as_text(self):
        tokenizer = FakeTokenizer(trailing_specials=1)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo ", "bar")

        # The appended EOS sits after the first text token, so the seam search
        # alone would leave it in the scored segment.
        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_trailing_special_token_is_dropped_without_a_context(self):
        tokenizer = FakeTokenizer(trailing_specials=1)
        context_ids, text_ids = split_context_and_text(tokenizer, "", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_every_trailing_empty_span_is_dropped(self):
        tokenizer = FakeTokenizer(trailing_specials=2)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_whitespace_only_context_keeps_its_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids = split_context_and_text(tokenizer, " ", "bar")

        # A lone space is what makes the text start with a leading-space token,
        # so it has to survive into the scored passage.
        self.assertEqual(context_ids, [0, tokenizer.vocab[" "]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_slow_tokenizers_fall_back_to_encoding_each_half(self):
        tokenizer = FakeTokenizer(is_fast=False)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0, tokenizer.vocab["foo"]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])


class ScoringEncodeTests(unittest.TestCase):
    def test_a_whitespace_only_context_is_scored_not_discarded(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids = encode_for_scoring(tokenizer, "bar", context=" ")

        self.assertEqual(context_ids, [0, tokenizer.vocab[" "]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_an_empty_context_is_unchanged(self):
        tokenizer = FakeTokenizer()

        self.assertEqual(
            encode_for_scoring(tokenizer, "bar", context=""),
            split_context_and_text(tokenizer, "", "bar"),
        )

    def test_a_real_context_uses_the_chat_template(self):
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        context_ids, text_ids = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertEqual(
            context_ids,
            [
                tokenizer.vocab["<|user|>"],
                tokenizer.vocab["hello"],
                tokenizer.vocab["<|assistant|>"],
            ],
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_whitespace_only_context_skips_the_chat_template(self):
        # Pure whitespace is not a turn worth wrapping in a user message, but
        # it still belongs in front of the text, so the plain path scores it.
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        context_ids, text_ids = encode_for_scoring(
            tokenizer, "bar", context=" ", use_chat_template=True
        )

        self.assertEqual(context_ids, [0, tokenizer.vocab[" "]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])


class ScoreTextGuardTests(unittest.TestCase):
    """What ``score_text`` refuses, decided before any tensor is built."""

    def manager(self) -> ModelManager:
        manager = ModelManager()
        manager.model = object()
        manager.tokenizer = FakeTokenizer()
        self.prefilled: list[int] = []

        def fake_prefill(token_ids, *, segments, positions, score_from, collect):
            self.prefilled = list(token_ids)
            return (
                [
                    {"segment": segment, "position": position, "scored": index >= score_from}
                    for index, (segment, position) in enumerate(zip(segments, positions))
                ],
                None,
                None,
            )

        manager._prefill = fake_prefill
        return manager

    def test_an_empty_box_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.manager().score_text("")

        self.assertIn("Enter some text", str(caught.exception))

    def test_whitespace_only_text_is_scored_not_rejected(self):
        # How expected a paragraph break was is a real question to put to the
        # model, and the tokenizer turns those newlines into ordinary tokens.
        manager = self.manager()
        result = manager.score_text("\n\n", context="the end.")

        self.assertEqual(
            self.prefilled[-1], manager.tokenizer.vocab["\n\n"]
        )
        self.assertEqual(len(result.metrics), 1)
        self.assertTrue(result.metrics[0]["scored"])

    def test_text_that_tokenizes_to_nothing_still_says_so(self):
        # The narrowed guard hands this case to the ``text_ids`` check, which
        # is the one that knows the tokenizer dropped the input.
        class DropsEverything(FakeTokenizer):
            def __call__(self, text, **kwargs):
                encoding = Encoding(input_ids=[])
                if kwargs.get("return_offsets_mapping"):
                    encoding["offset_mapping"] = []
                return encoding

        manager = self.manager()
        manager.tokenizer = DropsEverything()
        with self.assertRaises(ValueError) as caught:
            manager.score_text(" ")

        self.assertIn("did not produce any tokens", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
