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
        generation_prompt: str = "<|assistant|>",
    ):
        self.is_fast = is_fast
        self.trailing_specials = trailing_specials
        self.chat_template = chat_template
        self.generation_prompt = generation_prompt
        self.vocab: dict[str, int] = {"<s>": 0, "</s>": 1, "<|user|>": 2, "<|assistant|>": 3}
        self.all_special_ids = list(self.vocab.values())

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def decode(self, ids, skip_special_tokens=False, **kwargs) -> str:
        pieces = {index: piece for piece, index in self.vocab.items()}
        return "".join(
            pieces[int(index)]
            for index in ids
            if not (skip_special_tokens and int(index) in self.all_special_ids)
        )

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
        """Render the turn as text, the way a real template does.

        The generation prompt is the interesting part: it ends in whatever
        characters the template writes after its last special token, and by
        default that is nothing at all, so the marker abuts the reply and the
        two merge into one token.
        """

        rendered = "<|user|> " + " ".join(
            message["content"] for message in messages
        )
        if add_generation_prompt:
            rendered += " " + self.generation_prompt
        if not tokenize:
            return rendered
        return [
            int(value)
            for value in self(rendered, add_special_tokens=False).input_ids
        ]


class EatsTheLeadingSpace(FakeTokenizer):
    """Decodes the way SentencePiece does: the opening space is the marker.

    ``decode`` is then not the inverse of ``encode``, so a seam found by
    decoding a prefix of the sequence would sit one token off.
    """

    def decode(self, ids, skip_special_tokens=False, **kwargs) -> str:
        spoken = super().decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return spoken[1:] if spoken.startswith(" ") else spoken


class CutsCharactersInHalf:
    """One token per byte, as byte-level BPE does when it has no merge left.

    Decoding a run of bytes that ends mid-character says U+FFFD rather than
    the character, so a seam found by decoding can be stranded early even
    though the two halves still concatenate to the passage.
    """

    is_fast = False
    chat_template = None
    all_special_ids: list[int] = []

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if return_offsets_mapping:
            raise NotImplementedError("offset mapping needs a fast tokenizer")
        return Encoding(input_ids=list(text.encode()))

    def decode(self, ids, skip_special_tokens=False, **kwargs) -> str:
        return bytes(int(index) for index in ids).decode("utf-8", errors="replace")


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

    def test_slow_tokenizers_keep_the_joint_encoding(self):
        # No offsets to search, so the seam is found by decoding a growing
        # prefix back to text. The passage is still encoded once, so the
        # merged token is the one that gets scored.
        tokenizer = FakeTokenizer(is_fast=False)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_slow_tokenizer_drops_its_trailing_special_token(self):
        tokenizer = FakeTokenizer(is_fast=False, trailing_specials=1)
        context_ids, text_ids = split_context_and_text(tokenizer, "foo ", "bar")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_decode_that_does_not_round_trip_falls_back(self):
        # SentencePiece eats the space that opens a sequence, so decoding the
        # leading run of tokens no longer says what the context said. Rather
        # than move the seam by a token and score part of the context, the
        # split gives up and each half is encoded on its own.
        tokenizer = EatsTheLeadingSpace(is_fast=False)
        context_ids, text_ids = split_context_and_text(tokenizer, " foo", "bar")

        self.assertEqual(context_ids, [0, tokenizer.vocab[" "], tokenizer.vocab["foo"]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_seam_stranded_mid_character_falls_back(self):
        # The halves still concatenate to the passage here, so only the token
        # the split lands on gives the mistake away: it has to reach the seam,
        # and a run of bytes cut inside a character does not.
        tokenizer = CutsCharactersInHalf()
        context_ids, text_ids = split_context_and_text(tokenizer, "日本語", "です")

        self.assertEqual(context_ids, list("日本語".encode()))
        self.assertEqual(text_ids, list("です".encode()))

    def test_a_tokenizer_that_cannot_decode_falls_back(self):
        tokenizer = FakeTokenizer(is_fast=False)
        tokenizer.decode = None
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
        # The generation prompt is followed by a space here, so the seam
        # cannot merge and the halves come out exactly as the template
        # tokenizes them.
        tokenizer = FakeTokenizer(
            chat_template="{{ messages }}", generation_prompt="<|assistant|> "
        )
        context_ids, text_ids = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertEqual(
            context_ids,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}], add_generation_prompt=True
            ),
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_chat_template_seam_is_tokenized_as_one_passage(self):
        # Most templates end in ordinary characters after their last special
        # token, so the marker and the first reply token merge. Encoding the
        # rendered prompt and the reply apart would report ranks for a first
        # token the model never sees.
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        context_ids, text_ids = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertEqual(
            context_ids,
            [
                tokenizer.vocab["<|user|>"],
                tokenizer.vocab[" "],
                tokenizer.vocab["hello"],
                tokenizer.vocab[" "],
            ],
        )
        self.assertEqual(text_ids, [tokenizer.vocab["<|assistant|>bar"]])

    def test_a_slow_tokenizer_splits_the_chat_template_seam_too(self):
        # No offsets, so the seam is found by decoding; the specials the
        # template rendered have to survive that decode to round trip.
        tokenizer = FakeTokenizer(is_fast=False, chat_template="{{ messages }}")
        context_ids, text_ids = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertEqual(
            context_ids,
            [
                tokenizer.vocab["<|user|>"],
                tokenizer.vocab[" "],
                tokenizer.vocab["hello"],
                tokenizer.vocab[" "],
            ],
        )
        self.assertEqual(text_ids, [tokenizer.vocab["<|assistant|>bar"]])

    def test_the_chat_template_is_not_given_a_second_beginning_token(self):
        # The template renders its own opening special; asking the tokenizer
        # to add one as well would prepend a second <s> the model never sees.
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        context_ids, _ = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertNotIn(tokenizer.vocab["<s>"], context_ids)
        self.assertEqual(context_ids[0], tokenizer.vocab["<|user|>"])

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
