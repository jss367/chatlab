import re
import tempfile
import unittest
from pathlib import Path

from model_runtime import (
    MIN_MODEL_POSITION_LIMIT,
    SCORE_TOKEN_LIMIT,
    ModelManager,
    cache_status,
    encode_for_scoring,
    format_bytes,
    score_token_limit,
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


class CacheStatusTests(unittest.TestCase):
    """What the cache inspection reports for the three states a model can be in."""

    def folder(self, root: str) -> Path:
        blobs = Path(root) / "models--allenai--Olmo-3-7B-Think" / "blobs"
        blobs.mkdir(parents=True)
        return blobs

    def test_an_unknown_model_is_absent(self):
        with tempfile.TemporaryDirectory() as root:
            status = cache_status("allenai/Olmo-3-7B-Think", Path(root))

        self.assertFalse(status.present)
        self.assertEqual(status.total_bytes, 0)

    def test_partial_blobs_are_counted_apart_from_finished_ones(self):
        with tempfile.TemporaryDirectory() as root:
            blobs = self.folder(root)
            (blobs / "abc").write_bytes(b"x" * 10)
            (blobs / "def.1234.incomplete").write_bytes(b"x" * 100)
            (blobs / "ghi.5678.incomplete").write_bytes(b"x" * 200)
            status = cache_status("allenai/Olmo-3-7B-Think", Path(root))

        self.assertTrue(status.present)
        self.assertEqual(status.cached_bytes, 10)
        self.assertEqual(status.partial_files, 2)
        self.assertEqual(status.partial_bytes, 300)

    def test_a_finished_download_has_no_partials(self):
        with tempfile.TemporaryDirectory() as root:
            (self.folder(root) / "abc").write_bytes(b"x" * 10)
            status = cache_status("allenai/Olmo-3-7B-Think", Path(root))

        self.assertEqual(status.partial_files, 0)
        self.assertEqual(status.cached_bytes, 10)

    def test_an_invalid_id_is_rejected_before_the_disk_is_read(self):
        with self.assertRaises(ValueError):
            cache_status("../escape", Path("/nonexistent"))

    def test_byte_counts_read_like_a_download_dialog(self):
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(3_418_357_760), "3.4 GB")
        self.assertEqual(format_bytes(146_800_640), "147 MB")
        self.assertEqual(format_bytes(15_000_000_000), "15.0 GB")


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


class ProbesDifferently(FakeTokenizer):
    """Encodes the passage differently when the post-processor is switched off.

    A normalizer that only runs alongside the post-processor would look like
    this. The second encoding then proves nothing about where the trailing
    specials came from.
    """

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if not add_special_tokens:
            text += " unasked"
        return super().__call__(
            text,
            return_offsets_mapping=return_offsets_mapping,
            add_special_tokens=add_special_tokens,
        )


class RefusesTheProbe(FakeTokenizer):
    """Will not encode anything without its post-processor.

    There is no second encoding to compare against at all here.
    """

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if not add_special_tokens:
            raise ValueError("this tokenizer always adds its special tokens")
        return super().__call__(
            text,
            return_offsets_mapping=return_offsets_mapping,
            add_special_tokens=add_special_tokens,
        )


class SkipsADoubledCloser(FakeTokenizer):
    """Appends its closers only where the text does not already end in one.

    A post-processor that avoids writing ``</s></s>`` behaves this way, and
    it means the wrapping measured on an ordinary probe is more than such a
    passage carries. A count measured elsewhere says nothing about a passage
    that contradicts it, so the passage itself has to answer for its own
    trailing specials.
    """

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        doubled = text.endswith("</s>")
        keep = self.trailing_specials
        self.trailing_specials = 0 if doubled else keep
        try:
            return super().__call__(
                text,
                return_offsets_mapping=return_offsets_mapping,
                add_special_tokens=add_special_tokens,
            )
        finally:
            self.trailing_specials = keep


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
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo", "bar")

        # "foobar" merges into a single token, so scoring the two halves
        # encoded apart would measure a sequence the passage never produces.
        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_clean_seam_keeps_every_context_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "bar")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_an_empty_context_still_carries_the_special_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_trailing_special_token_is_not_scored_as_text(self):
        tokenizer = FakeTokenizer(trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "bar")

        # The appended EOS sits after the first text token, so the seam search
        # alone would leave it in the scored segment.
        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_a_trailing_special_token_is_dropped_without_a_context(self):
        tokenizer = FakeTokenizer(trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_every_trailing_empty_span_is_dropped(self):
        tokenizer = FakeTokenizer(trailing_specials=2)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_whitespace_only_context_keeps_its_token(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, " ", "bar")

        # A lone space is what makes the text start with a leading-space token,
        # so it has to survive into the scored passage.
        self.assertEqual(context_ids, [0, tokenizer.vocab[" "]])
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])

    def test_slow_tokenizers_keep_the_joint_encoding(self):
        # No offsets to search, so the seam is found by decoding a growing
        # prefix back to text. The passage is still encoded once, so the
        # merged token is the one that gets scored.
        tokenizer = FakeTokenizer(is_fast=False)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(context_ids, [0])
        self.assertEqual(text_ids, [tokenizer.vocab["foobar"]])

    def test_a_slow_tokenizer_drops_its_trailing_special_token(self):
        tokenizer = FakeTokenizer(is_fast=False, trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "bar")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["bar"]])


    def test_a_pasted_trailing_special_token_survives_a_slow_tokenizer(self):
        # Nothing was appended here: the reader pasted "</s>" at the end of
        # their own text. Dropping it by id membership would report ranks and
        # perplexity for a passage that stops one token early.
        tokenizer = FakeTokenizer(is_fast=False)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["</s>"]])

    def test_a_pasted_special_and_an_appended_one_are_told_apart(self):
        # The case that separates a real fix from a plausible one: the text
        # ends in a special the reader wrote *and* the post-processor appends
        # its own after it. Exactly one of the two is the reader's.
        tokenizer = FakeTokenizer(is_fast=False, trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["</s>"]])

    def test_an_all_special_passage_keeps_the_token_the_reader_wrote(self):
        # The passage is nothing but special tokens: the tokenizer opens with
        # <s>, the reader wrote </s>, and the post-processor closed with the
        # same </s>. Nothing about the passage itself can say which of the
        # two closers is the reader's — every reading of it explains the ids —
        # so the wrapping is measured on the tokenizer instead, and the
        # reader's token is the one that gets scored.
        tokenizer = FakeTokenizer(is_fast=False, trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "", "</s>")

        self.assertEqual(context_ids, [tokenizer.vocab["<s>"]])
        self.assertEqual(text_ids, [tokenizer.vocab["</s>"]])

    def test_a_wrapper_the_passage_contradicts_is_not_applied_to_it(self):
        # The probe says two closers are appended, but this passage ends in
        # one special token altogether, so the count measured elsewhere is
        # not subtracted here. The passage is asked instead, and it answers:
        # nothing was appended to it, and the </s> is the reader's.
        tokenizer = SkipsADoubledCloser(is_fast=False, trailing_specials=2)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["</s>"]])

    def test_the_offsets_path_keeps_a_pasted_trailing_special_token(self):
        # The fast path never had to guess, and still does not: the pasted
        # token carries a real span and the appended one carries (0, 0).
        tokenizer = FakeTokenizer(trailing_specials=1)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertEqual(
            context_ids, [0, tokenizer.vocab["foo"], tokenizer.vocab[" "]]
        )
        self.assertEqual(text_ids, [tokenizer.vocab["</s>"]])

    def test_a_probe_that_proves_nothing_drops_the_whole_trailing_run(self):
        # The second encoding does not line up with the joint one, so which
        # trailing specials the post-processor added is unknown. The pasted
        # token is lost, as it was before there was a probe at all, rather
        # than a closer nobody wrote being scored.
        tokenizer = ProbesDifferently(is_fast=False)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertNotIn(tokenizer.vocab["</s>"], context_ids + text_ids)

    def test_a_tokenizer_that_refuses_the_probe_drops_the_whole_run_too(self):
        tokenizer = RefusesTheProbe(is_fast=False)
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "foo ", "</s>")

        self.assertNotIn(tokenizer.vocab["</s>"], context_ids + text_ids)

    def test_a_decode_that_does_not_round_trip_still_cuts_one_encoding(self):
        # SentencePiece eats the space that opens a sequence, so decoding the
        # leading run of tokens no longer says what the context said. The
        # position cannot be confirmed, but the joint encoding is still the
        # sequence the passage produces, so it is cut rather than abandoned:
        # "foo" and "bar" merged, and that merged token is what gets scored.
        tokenizer = EatsTheLeadingSpace(is_fast=False)
        split = split_context_and_text(tokenizer, " foo", "bar")

        self.assertEqual(split.context_ids, [0, tokenizer.vocab[" "]])
        self.assertEqual(split.text_ids, [tokenizer.vocab["foobar"]])
        self.assertFalse(split.seam_verified)

    def test_a_seam_stranded_mid_character_falls_back(self):
        # The halves still concatenate to the passage here, so only the token
        # the split lands on gives the mistake away: it has to reach the seam,
        # and a run of bytes cut inside a character does not.
        tokenizer = CutsCharactersInHalf()
        context_ids, text_ids, *_ = split_context_and_text(tokenizer, "日本語", "です")

        self.assertEqual(context_ids, list("日本語".encode()))
        self.assertEqual(text_ids, list("です".encode()))

    def test_a_checked_seam_reports_itself_as_verified(self):
        # Both the offsets path and the decoding path cut one joint encoding,
        # so the ids are the passage's own and the caller has nothing to warn
        # about.
        for tokenizer in (FakeTokenizer(), FakeTokenizer(is_fast=False)):
            with self.subTest(is_fast=tokenizer.is_fast):
                split = split_context_and_text(tokenizer, "foo", "bar")

                self.assertTrue(split.seam_verified)

    def test_an_unverifiable_seam_reports_itself(self):
        # Encoding the halves apart is still better than refusing to score at
        # all — the numbers are off by at most the token spanning the seam —
        # but the caller has to be able to say the numbers are approximate.
        cases = (
            (EatsTheLeadingSpace(is_fast=False), " foo", "bar"),
            (CutsCharactersInHalf(), "日本語", "です"),
        )
        for tokenizer, context, text in cases:
            with self.subTest(tokenizer=type(tokenizer).__name__):
                split = split_context_and_text(tokenizer, context, text)

                self.assertFalse(split.seam_verified)

    def test_a_fallback_without_a_context_is_still_exact(self):
        # No context means no seam for a merge to cross: the text is encoded
        # exactly as the joint passage would encode it, so warning here would
        # be noise on every score.
        split = split_context_and_text(EatsTheLeadingSpace(is_fast=False), "", "bar")

        self.assertTrue(split.seam_verified)

    def test_a_tokenizer_that_cannot_decode_still_cuts_one_encoding(self):
        # Nothing here can say where the seam fell, so the cut is counted out
        # instead. The ids stay the joint encoding's own, which is the whole
        # point: "foobar" is the token the passage produces, and encoding the
        # halves apart would have scored a "bar" the model never sees.
        tokenizer = FakeTokenizer(is_fast=False)
        tokenizer.decode = None
        split = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(split.context_ids, [0])
        self.assertEqual(split.text_ids, [tokenizer.vocab["foobar"]])
        self.assertFalse(split.seam_verified)


def _gpt2_or_none():
    """The cached GPT-2 tokenizer, or ``None`` where it is not on this machine.

    The vocabulary is the point: a real one where the literal
    ``<|endoftext|>`` a reader might paste and the id a post-processor would
    append are the same token, which is the case the fake tokenizers can only
    assert into being.
    """

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:  # noqa: BLE001 - no tokenizer on this machine is fine
        return None


GPT2 = _gpt2_or_none()


class WrappedWithoutOffsets:
    """A real tokenizer given a post-processor and no offsets to search.

    Transformers no longer ships a genuine slow GPT-2 class, so the slow
    path is reached by refusing offsets rather than by asking for one. What
    the wrapper adds is a post-processor of the shape this whole question is
    about: an opening id and a closing id that happen to be the same token.
    """

    is_fast = False
    chat_template = None

    def __init__(self, inner, opening: list[int], closing: list[int]):
        self._inner = inner
        self.opening, self.closing = opening, closing
        self.all_special_ids = list(inner.all_special_ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if return_offsets_mapping:
            raise NotImplementedError("offset mapping needs a fast tokenizer")
        ids = [
            int(value)
            for value in self._inner(text, add_special_tokens=False).input_ids
        ]
        if add_special_tokens:
            ids = self.opening + ids + self.closing
        return Encoding(input_ids=ids)

    def decode(self, ids, **kwargs) -> str:
        return self._inner.decode(list(ids), **kwargs)


@unittest.skipIf(GPT2 is None, "the GPT-2 tokenizer is not cached on this machine")
class RealVocabularyTests(unittest.TestCase):
    """The all-special case with a real vocabulary behind it."""

    def tokenizer(self):
        closer = [int(GPT2.eos_token_id)]
        return WrappedWithoutOffsets(GPT2, closer, list(closer))

    def test_a_passage_of_nothing_but_specials_scores_the_pasted_one(self):
        # The reader pasted <|endoftext|> and nothing else, so the ids are
        # the opening id, their token, and the appended closer — all three
        # the same number. The scored token is theirs, and the closer the
        # post-processor wrote is not scored.
        tokenizer = self.tokenizer()
        eos = int(GPT2.eos_token_id)
        self.assertEqual(
            tokenizer("<|endoftext|>").input_ids, [eos, eos, eos]
        )

        context_ids, text_ids, *_ = split_context_and_text(
            tokenizer, "", "<|endoftext|>"
        )

        self.assertEqual(context_ids, [eos])
        self.assertEqual(text_ids, [eos])

    def test_an_ordinary_appended_closer_still_comes_off(self):
        tokenizer = self.tokenizer()
        eos = int(GPT2.eos_token_id)
        context_ids, text_ids, *_ = split_context_and_text(
            tokenizer, "the cat sat on the ", "mat"
        )

        self.assertEqual(context_ids[0], eos)
        self.assertNotIn(eos, text_ids)
        # The token that straddles the seam carries the context's trailing
        # space with it, and is scored as part of the text, as it is
        # everywhere else.
        self.assertEqual(GPT2.decode(text_ids), " mat")


class ScoringEncodeTests(unittest.TestCase):
    def test_a_whitespace_only_context_is_scored_not_discarded(self):
        tokenizer = FakeTokenizer()
        context_ids, text_ids, *_ = encode_for_scoring(tokenizer, "bar", context=" ")

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
        context_ids, text_ids, *_ = encode_for_scoring(
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
        context_ids, text_ids, *_ = encode_for_scoring(
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
        context_ids, text_ids, *_ = encode_for_scoring(
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
        context_ids, *_ = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertNotIn(tokenizer.vocab["<s>"], context_ids)
        self.assertEqual(context_ids[0], tokenizer.vocab["<|user|>"])

    def test_a_whitespace_only_context_is_wrapped_as_a_turn(self):
        # A turn of pure whitespace is still the turn the reader asked to
        # send. Dropping the template for it would quietly score the raw
        # characters instead: no role markers, no generation prompt, and no
        # caveat either, because the model has a template all along.
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        split = encode_for_scoring(
            tokenizer, "bar", context=" ", use_chat_template=True
        )

        self.assertEqual(split.context_ids[0], tokenizer.vocab["<|user|>"])
        self.assertNotIn(tokenizer.vocab["<s>"], split.context_ids)
        self.assertFalse(split.chat_template_missing)

    def test_an_empty_context_is_not_wrapped_as_a_turn(self):
        # An empty box holds no message for a template to render, so this is
        # the plain path by nature rather than a request that got dropped.
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")

        self.assertEqual(
            encode_for_scoring(tokenizer, "bar", context="", use_chat_template=True),
            split_context_and_text(tokenizer, "", "bar"),
        )

    def test_a_model_without_a_chat_template_still_scores_and_says_so(self):
        # GPT-2 and friends have no turn to wrap the context in. Refusing here
        # would drop a measurement the reader asked for to avoid numbers that
        # are not wrong, only differently framed, so the passage is scored
        # verbatim and the flag lets the caller name the framing.
        tokenizer = FakeTokenizer(chat_template=None)
        split = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertTrue(split.chat_template_missing)
        self.assertEqual(
            (split.context_ids, split.text_ids),
            split_context_and_text(tokenizer, "hello", "bar")[:2],
        )

    def test_a_template_that_was_applied_raises_no_caveat(self):
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        split = encode_for_scoring(
            tokenizer, "bar", context="hello", use_chat_template=True
        )

        self.assertFalse(split.chat_template_missing)

    def test_a_template_nobody_asked_for_raises_no_caveat(self):
        # Nothing was ignored when the box was never ticked, and a caveat on
        # every plain score would be noise.
        for context in ("hello", " ", ""):
            with self.subTest(context=context):
                split = encode_for_scoring(
                    FakeTokenizer(chat_template=None), "bar", context=context
                )

                self.assertFalse(split.chat_template_missing)

    def test_an_empty_context_raises_no_template_caveat(self):
        # There is no turn in an empty box for any model to wrap, so this is
        # not the missing-template case and saying it was would misdirect.
        split = encode_for_scoring(
            FakeTokenizer(chat_template=None),
            "bar",
            context="",
            use_chat_template=True,
        )

        self.assertFalse(split.chat_template_missing)

    def test_a_whitespace_turn_still_raises_the_template_caveat(self):
        # The turn was real and the model had nothing to wrap it in, which is
        # exactly what the caveat is for.
        split = encode_for_scoring(
            FakeTokenizer(chat_template=None),
            "bar",
            context=" ",
            use_chat_template=True,
        )

        self.assertTrue(split.chat_template_missing)


class Config:
    """The one thing ``score_token_limit`` reads off a loaded model."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)

    def get_text_config(self):
        return self


class Model:
    def __init__(self, config=None):
        self.config = config


class ScoreTokenLimitTests(unittest.TestCase):
    """The scoring cap, worked out without running a forward pass."""

    def test_a_short_context_window_caps_the_flat_limit(self):
        # GPT-2 has 1,024 position embeddings; feeding it more indexes off the
        # end of that table instead of raising anything a reader can act on.
        limit = score_token_limit(Model(Config(max_position_embeddings=1024)))

        self.assertEqual(limit, 1024)

    def test_a_long_context_window_leaves_the_flat_limit_alone(self):
        limit = score_token_limit(Model(Config(max_position_embeddings=131072)))

        self.assertEqual(limit, SCORE_TOKEN_LIMIT)

    def test_an_older_config_spelling_is_read_too(self):
        limit = score_token_limit(Model(Config(n_positions=512)))

        self.assertEqual(limit, 512)

    def test_mpt_style_configs_name_their_window_differently(self):
        # MPT and DBRX call it max_seq_len; nothing else in the config says
        # how long the window is, so missing this name means missing the cap.
        limit = score_token_limit(Model(Config(max_seq_len=2048)))

        self.assertEqual(limit, 2048)

    def test_a_config_that_does_not_say_keeps_the_flat_limit(self):
        # A model that carries no position table — or one whose config simply
        # does not name its window — would otherwise be blocked outright.
        for model in (Model(), Model(Config()), object()):
            with self.subTest(model=model):
                self.assertEqual(score_token_limit(model), SCORE_TOKEN_LIMIT)

    def test_an_absurd_window_keeps_the_flat_limit(self):
        # None of these can be a real window, and honouring them would refuse
        # every passage on a model that would have run fine.
        for value in (0, -1, "lots", None, True, MIN_MODEL_POSITION_LIMIT - 1):
            with self.subTest(value=value):
                model = Model(Config(max_position_embeddings=value))
                self.assertEqual(score_token_limit(model), SCORE_TOKEN_LIMIT)

    def test_a_multimodal_config_uses_its_language_window(self):
        # The scored tokens are laid out against the text tower, so its window
        # is the one that matters.
        text = Config(max_position_embeddings=2048)
        wrapper = Config(max_position_embeddings=131072)
        wrapper.get_text_config = lambda: text

        self.assertEqual(score_token_limit(Model(wrapper)), 2048)


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

    def test_an_unverifiable_seam_reaches_the_result(self):
        manager = self.manager()
        manager.tokenizer = EatsTheLeadingSpace(is_fast=False)

        self.assertFalse(manager.score_text("bar", context=" foo").seam_verified)
        self.assertTrue(self.manager().score_text("bar", context="foo ").seam_verified)

    def test_a_missing_chat_template_reaches_the_result(self):
        # The score still happens — the flag is what the interface turns into
        # a sentence about which passage the numbers describe.
        result = self.manager().score_text(
            "bar", context="hello", use_chat_template=True
        )

        self.assertTrue(result.chat_template_missing)
        self.assertEqual(len(result.metrics), 1)

        manager = self.manager()
        manager.tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        with_template = manager.score_text(
            "bar", context="hello", use_chat_template=True
        )

        self.assertFalse(with_template.chat_template_missing)

    def test_both_caveats_can_be_raised_by_one_score(self):
        # Nothing couples them: a tokenizer can lack a chat template and also
        # be unable to say where the context ends, and the reader is owed both
        # facts rather than whichever one the code checked first.
        manager = self.manager()
        manager.tokenizer = EatsTheLeadingSpace(is_fast=False, chat_template=None)
        result = manager.score_text("bar", context=" hello", use_chat_template=True)

        self.assertTrue(result.chat_template_missing)
        self.assertFalse(result.seam_verified)

    def test_a_passage_past_the_model_window_is_refused_by_name(self):
        # The flat cap is not the only ceiling: a model with a shorter
        # position table has to say so here, because saying it after the fact
        # means an IndexError from inside the forward pass.
        manager = self.manager()
        manager.model = Model(Config(max_position_embeddings=32))

        with self.assertRaises(ValueError) as caught:
            manager.score_text(" ".join(str(number) for number in range(64)))

        self.assertIn("32 positions", str(caught.exception))
        self.assertEqual(self.prefilled, [])

    def test_a_roomy_model_still_refuses_at_the_flat_limit(self):
        manager = self.manager()
        manager.model = Model(Config(max_position_embeddings=131072))

        with self.assertRaises(ValueError) as caught:
            manager.score_text(" ".join(str(number) for number in range(4096)))

        self.assertIn(f"{SCORE_TOKEN_LIMIT:,} token limit", str(caught.exception))
        self.assertEqual(self.prefilled, [])

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



class NeverRoundTrips(FakeTokenizer):
    """Declines both joint paths: no offsets, and a decode that says nothing true.

    That combination is the only way to reach a seam nobody can confirm, so it
    is the only way to test what the split makes of one.
    """

    def decode(self, ids, skip_special_tokens=False, **kwargs) -> str:
        return " nope"


class RefusesTheWholePassage(FakeTokenizer):
    """Encodes either half, but raises on the passage the two make together.

    The one case with no joint encoding to cut, and so the one case where the
    halves still have to be encoded apart.
    """

    bos_token_id = 0

    def __init__(self, passage: str, **kwargs):
        super().__init__(**kwargs)
        self.passage = passage

    def __call__(self, text, **kwargs):
        if text == self.passage:
            raise ValueError("this tokenizer will not encode that")
        return super().__call__(text, **kwargs)


class UnverifiedSeamTests(unittest.TestCase):
    """A seam nobody could confirm is still a cut in the passage's own ids.

    Encoding the two halves apart is what let a post-processor's closing
    ``</s>`` land between them, and what let a token merged across the seam be
    replaced by two the passage never produces. Cutting one joint encoding can
    do neither: only the boundary's position is a guess.
    """

    def test_an_unverified_split_is_still_the_joint_encoding(self):
        tokenizer = NeverRoundTrips(is_fast=False, trailing_specials=1)
        joint = [
            int(value) for value in tokenizer("The capital ofFrance").input_ids
        ]

        split = split_context_and_text(tokenizer, "The capital of", "France")

        # Every scored id comes out of that one encoding, closing special
        # aside, so no distribution is taken from a sequence the reader did
        # not write — including the token "of" and "France" merged into.
        self.assertFalse(split.seam_verified)
        self.assertEqual(split.context_ids + split.text_ids, joint[:-1])
        self.assertEqual(split.text_ids, [tokenizer.vocab["ofFrance"]])

    def test_an_empty_context_never_gains_a_closing_token(self):
        # A post-processor that answers "" with an opening *and* a closing
        # special would otherwise put that closer between the halves, and the
        # text would be scored as what follows the end of a passage.
        tokenizer = NeverRoundTrips(is_fast=False, trailing_specials=1)

        split = split_context_and_text(tokenizer, "", "France")

        self.assertEqual(split.context_ids, [tokenizer.vocab["<s>"]])
        self.assertEqual(split.text_ids, [tokenizer.vocab["France"]])
        self.assertTrue(split.seam_verified)

    def test_the_opening_special_survives_and_the_closing_one_does_not(self):
        tokenizer = EatsTheLeadingSpace(is_fast=False, trailing_specials=1)

        split = split_context_and_text(tokenizer, "The capital of", " France")

        self.assertFalse(split.seam_verified)
        self.assertEqual(split.context_ids[0], tokenizer.vocab["<s>"])
        self.assertNotIn(tokenizer.vocab["</s>"], split.context_ids)
        self.assertNotIn(tokenizer.vocab["</s>"], split.text_ids)

    def test_the_guess_leaves_the_text_something_to_score(self):
        # The context's tokens account for the whole passage when the seam
        # token merged across it, and scoring nothing at all would refuse the
        # passage outright. That token is the text's.
        tokenizer = NeverRoundTrips(is_fast=False)

        split = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(split.text_ids, [tokenizer.vocab["foobar"]])

    def test_a_passage_that_will_not_encode_uses_the_halves_apart(self):
        # No joint encoding exists to cut here. Neither half is
        # post-processed, so nothing can be appended into the seam; the
        # opening token the model expects is put in front by name.
        tokenizer = RefusesTheWholePassage("foobar", trailing_specials=1)

        split = split_context_and_text(tokenizer, "foo", "bar")

        self.assertEqual(
            split.context_ids, [tokenizer.vocab["<s>"], tokenizer.vocab["foo"]]
        )
        self.assertEqual(split.text_ids, [tokenizer.vocab["bar"]])
        self.assertFalse(split.seam_verified)


if __name__ == "__main__":
    unittest.main()
