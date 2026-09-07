"""A real byte-level BPE tokenizer the tests can build for themselves.

A few tests want a genuine fast tokenizer rather than a fake: one whose
byte-level merges split an emoji into several tokens, whose offsets come from
the Rust side, and whose special token is registered the way GPT-2's
``<|endoftext|>`` is. They used to reach for GPT-2 out of the Hugging Face
cache and skip when it was not there, which made the suite depend on what
one machine had happened to download. This trains an equivalent on a few
sentences in well under a second, so the tests run the same everywhere.
"""

from __future__ import annotations

from functools import lru_cache

EOS = "<|endoftext|>"

_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a quick brown fox jumps over the lazy dog",
    "hello world, hello again",
    "tokens, merges and bytes",
]


@lru_cache(maxsize=1)
def build():
    """A ``PreTrainedTokenizerFast`` with byte-level BPE and one special token.

    Cached: the tests share one instance, exactly as they shared GPT-2.
    """

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_CORPUS * 4, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        bos_token=EOS,
        unk_token=EOS,
    )
