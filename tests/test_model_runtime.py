import re
import unittest

from model_runtime import split_context_and_text, validate_model_id


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

    def __init__(self, *, is_fast: bool = True):
        self.is_fast = is_fast
        self.vocab: dict[str, int] = {"<s>": 0}

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
        encoding = Encoding(input_ids=ids)
        if return_offsets_mapping:
            encoding["offset_mapping"] = offsets
        return encoding


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

    def test_slow_tokenizers_fall_back_to_encoding_each_half(self):
        tokenizer = FakeTokenizer(is_fast=False)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0, tokenizer.vocab["foo"]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])


if __name__ == "__main__":
    unittest.main()
