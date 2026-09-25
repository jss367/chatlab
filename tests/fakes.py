"""Stand-ins for the text model runtime, shared by the test modules.

The fakes the tests drive the runtime with: tokenizers over a fixed
vocabulary, a model that emits a scripted reply, and managers built from
them. They live here rather than in the test module that first needed them
so that a test module never imports another one, and so any single module
still runs on its own (``python -m unittest test_app_flow`` from ``tests``).
"""

import copy
from types import SimpleNamespace

import torch

from chatlab.model_runtime import GENERATING, LOADING, ModelManager
import tiny_tokenizer


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


# "Hello" and " world" are the answer; the reasoning tags are their own tokens.
THINK_PIECES = ["<think>", "</think>", "Hello", " world", "<eos>"]
THINK_EOS = 4


class Encoding(dict):
    """The subset of a Hugging Face ``BatchEncoding`` the prompt and split paths use."""

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


SP_PIECES = ["\u2581Hello", "\u2581world", "world", "\u2581", "!", "<unk>", "<eos>"]
SP_HELLO, SP_SPACE_WORLD, SP_WORLD, SP_SPACE = 0, 1, 2, 3
SP_EOS = SP_PIECES.index("<eos>")


def sentencepiece_manager(pieces=SP_PIECES):
    manager = loaded_manager([SP_HELLO], pieces, pieces.index("<eos>"))
    manager.tokenizer = SentencePieceTokenizer(pieces, pieces.index("<eos>"))
    return manager


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


# A manager as the extension pages see it: it hands out the generation slot
# and streams three tokens, and counts what it was asked to do.
class FakeManager:
    loaded = True
    model_id = "test/model"
    load_id = "first"
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [ord(c) for c in text], decode=lambda ids, **kw: ''.join(map(chr, ids)))

    def __init__(self):
        self.busy = False
        # A load claimed but not finished: the weights on their way out are
        # still in memory, so the session passes its loaded check.
        self.loading = False
        self.releases = 0
        self.closed_streams = 0
        self.options = None

    def claim_generation(self):
        if self.loading:
            return LOADING
        if self.busy:
            return GENERATING
        self.busy = True
        return None

    def reserve_generation(self):
        return self.claim_generation() is None

    def release_generation(self):
        self.busy = False
        self.releases += 1

    def _stop_token_ids(self):
        return {0}

    def encode_replacement(self, kept_ids, text, *, literal_prefill_tokens=0, load_id=None):
        return [ord(character) for character in text]

    def generate(self, messages, **options):
        self.options = options
        metrics = []
        try:
            for value in (1, 2, 3):
                metrics.append({'token_id': value})
                yield SimpleNamespace(metrics=metrics)
        finally:
            self.closed_streams += 1


class FakeConfig:
    def __init__(self, layers: int):
        self.num_hidden_layers = layers
        self._attn_implementation = "sdpa"


class FakeCache:
    """A key-value cache that only remembers how many tokens it holds.

    ``crop`` follows ``DynamicCache.crop`` in Transformers 4.57 through 5.x:
    a negative count removes that many tokens, and a positive one is the
    older "length to keep" form, a no-op when the cache is already shorter.
    """

    def __init__(self, sliding_windows: list[int | None] = ()):
        self.length = 0
        self.crops: list[int] = []
        # One entry per layer, as DynamicCache exposes them: None for a
        # full-attention layer, the window for a sliding one.
        self.is_sliding = [window is not None for window in sliding_windows]
        self.layers = [
            SimpleNamespace(sliding_window=window) if window is not None else SimpleNamespace()
            for window in sliding_windows
        ]

    def crop(self, tokens_to_remove: int) -> None:
        self.crops.append(tokens_to_remove)
        if tokens_to_remove > 0:
            if tokens_to_remove >= self.length:
                return
            tokens_to_remove = self.length - tokens_to_remove
        self.length -= abs(tokens_to_remove)


class FakeLensModel(torch.nn.Module):
    """A model whose layers change their mind partway up the stack.

    Position ``k`` of the sequence predicts ``script[k]``, so a sequence is
    self-consistent when it is one leading token followed by the script. The
    residual stream below ``decide_layer`` points at ``early`` instead, so the
    logit lens has something to show. The hidden size equals the vocab
    size and the head is the identity, so a one-hot hidden state is its own
    logit vector. Attention puts most weight on the first key (the sink) and
    the rest on ``focus``.
    """

    def __init__(self, script, *, layers=4, heads=2, decide_layer=2, early=3, focus=1):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.script = script
        self.vocab = len(PIECES)
        self.layers = layers
        self.heads = heads
        self.decide_layer = decide_layer
        self.early = early
        self.focus = focus
        self.config = FakeConfig(layers)
        self.generation_config = SimpleNamespace(eos_token_id=EOS_ID)
        self.head = torch.nn.Linear(self.vocab, self.vocab, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(self.vocab))
        self.base_model = SimpleNamespace(norm=torch.nn.Identity())
        self.attn_calls: list[str] = []
        self.return_attentions = True
        # A transform the config does not declare, to stand in for an
        # architecture the lens does not know how to read.
        self.undeclared_scale: float | None = None
        # Layers with a sliding window return weights for their last few keys
        # only, the way a sliding-window cache does.
        self.sliding_layers: dict[int, int] = {}
        # With ``caching`` on, forward() returns a FakeCache that grows with
        # the tokens fed, the way a DynamicCache does, and records how many
        # tokens each call fed it.
        self.caching = False
        self.sliding_windows: list[int | None] = []
        self.fed: list[int] = []

    def set_attn_implementation(self, name: str) -> None:
        self.attn_calls.append(name)
        self.config._attn_implementation = name

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        length = int(input_ids.shape[-1])
        keys = int(attention_mask.shape[-1])
        first = keys - length
        self.fed.append(length)
        cache = None
        if self.caching:
            cache = (
                past_key_values
                if past_key_values is not None
                else FakeCache(self.sliding_windows)
            )
            assert cache.length == first, (cache.length, first)
            cache.length += length
        targets = [
            self.script[(first + offset) % len(self.script)] for offset in range(length)
        ]

        def one_hot(indices):
            state = torch.full((1, length, self.vocab), -20.0)
            for offset, index in enumerate(indices):
                state[0, offset, index] = 20.0
            return state

        hidden = tuple(
            one_hot([self.early] * length if layer < self.decide_layer else targets)
            for layer in range(self.layers + 1)
        )
        logits = self.head(hidden[-1])
        if self.config.__dict__.get("logit_scale"):
            logits = logits * self.config.logit_scale
        if self.config.__dict__.get("logits_scaling"):
            logits = logits / self.config.logits_scaling
        if self.config.__dict__.get("final_logit_softcapping"):
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        if self.undeclared_scale:
            logits = logits * self.undeclared_scale
        attentions = None
        if output_attentions and self.return_attentions:
            weights = torch.full((1, self.heads, length, keys), 0.0)
            weights[..., 0] = 0.6
            if keys > 1:
                weights[..., min(self.focus, keys - 1)] += 0.4
            else:
                weights[..., 0] += 0.4
            attentions = tuple(
                torch.full(
                    (1, self.heads, length, min(keys, self.sliding_layers[layer])),
                    1.0 / min(keys, self.sliding_layers[layer]),
                )
                if layer in self.sliding_layers
                else weights.clone()
                for layer in range(self.layers)
            )
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache,
            hidden_states=hidden if output_hidden_states else None,
            attentions=attentions,
        )


def lens_manager(script, **options) -> ModelManager:
    manager = ModelManager()
    manager.tokenizer = FakeTokenizer()
    manager.model = FakeLensModel(script, **options)
    manager.model_id = "fake/lens"
    return manager


# The switching suffix from Qwen/Qwen3-0.6B's chat template. The off mode
# supplies an empty, closed think block; default/on leave generation alone.
THINKING_TEMPLATE = """{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}
{% if add_generation_prompt %}assistant:
{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"""


def manager_for_thinking():
    manager = loaded_manager([0])
    manager.tokenizer = copy.deepcopy(tiny_tokenizer.build())
    manager.tokenizer.chat_template = THINKING_TEMPLATE
    token = manager.tokenizer.encode("hello")[0]
    manager.model.script = [token]
    manager.model.vocab_size = len(manager.tokenizer)
    manager.model.config = SimpleNamespace(model_type="qwen3")
    return manager
