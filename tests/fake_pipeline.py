"""A diffusers pipeline small enough to run in a test, shaped like a real one.

Only the parts :mod:`image_runtime` watches are real, and they are real
rather than mocked: the cross-attention module is diffusers' own
``Attention``, so the recording processor is exercised against the class it
will meet, and the denoiser is a module the forward hook attaches to and
calls with a batch of two the way a guided pipeline does.

What is faked is the schedule and the decode. There is no scheduler and no
VAE: the latents walk towards a fixed target and the picture is drawn from
the last of them, which is enough for the readings to have something to
report and fast enough to run on a laptop's CPU.
"""

from __future__ import annotations

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
from PIL import Image


LATENT_CHANNELS = 4
SIDE = 8
TOKENS = 77
CROSS_DIM = 16


class FakeUnet(torch.nn.Module):
    """A denoiser with one cross-attention module and a processor registry.

    ``attn_processors`` and ``set_attn_processor`` are the two names
    :func:`image_runtime._install_recorders` needs, spelled as
    ``UNet2DConditionModel`` spells them, so what the test installs is what
    the real loader would.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attn2 = Attention(
            query_dim=CROSS_DIM,
            cross_attention_dim=CROSS_DIM,
            heads=2,
            dim_head=8,
        )
        self._processors = {"blocks.0.attn2.processor": AttnProcessor()}
        self.calls = 0

    @property
    def attn_processors(self) -> dict:
        return dict(self._processors)

    def set_attn_processor(self, processors) -> None:
        self._processors = dict(processors)

    def ask_the_prompt(self, encoder_hidden_states) -> None:
        """Run the cross-attention the way a UNet block would, once per step."""

        batch = encoder_hidden_states.shape[0]
        hidden = torch.randn(batch, SIDE * SIDE, CROSS_DIM)
        processor = self._processors["blocks.0.attn2.processor"]
        processor(self.attn2, hidden, encoder_hidden_states)

    def forward(self, sample, encoder_hidden_states=None):
        """A prediction per row, with the conditional half pulling harder."""

        self.calls += 1
        prediction = sample * 0.1
        if sample.shape[0] > 1:
            # The conditional half disagrees with the unconditional one, so
            # the guidance reading has something to measure.
            prediction = prediction.clone()
            prediction[sample.shape[0] // 2 :] += 0.4
        return (prediction,)


class FakeTokenizer:
    """Enough of a CLIP tokenizer to name the tokens a map is keyed by."""

    def __init__(self, vocabulary: tuple[str, ...] = ()) -> None:
        self.vocabulary = vocabulary

    def __call__(self, text, truncation=True, **kwargs):
        words = text.split() if text else []
        pieces = ["<|startoftext|>"] + [f"{word}</w>" for word in words] + ["<|endoftext|>"]
        self.pieces = pieces
        return type("Encoding", (), {"input_ids": list(range(len(pieces)))})()

    def convert_ids_to_tokens(self, ids):
        return list(self.pieces)


class FakePipeline:
    """A denoising loop with the two hooks a real one offers and nothing else."""

    def __init__(
        self, *, steps_run: int | None = None, tokenizer=None, watcher=None
    ) -> None:
        self.unet = FakeUnet()
        self.tokenizer = FakeTokenizer() if tokenizer is None else tokenizer
        # What the scheduler would really run, which need not be what was
        # asked for: a real one can add a step of its own.
        self.steps_run = steps_run
        # Called after each step, from inside the loop: how a test reaches
        # into a run that is under way, to stop it or to make it wait.
        self.watcher = watcher
        self.device = "cpu"

    def to(self, device):
        """Move onto a device, as ``DiffusionPipeline.to`` does, and return self."""

        self.device = device
        return self

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=64,
        height=64,
        num_images_per_prompt=1,
        generator=None,
        output_type="pil",
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        guided = guidance_scale > 1
        batch = 2 if guided else 1
        steps = self.steps_run if self.steps_run is not None else num_inference_steps
        latents = torch.randn(1, LATENT_CHANNELS, SIDE, SIDE, generator=generator)
        target = torch.ones_like(latents)
        for step in range(steps):
            encoder_hidden_states = torch.randn(batch, TOKENS, CROSS_DIM)
            self.unet(latents.repeat(batch, 1, 1, 1), encoder_hidden_states)
            self.unet.ask_the_prompt(encoder_hidden_states)
            # Towards the target by a shrinking share, so the movement
            # readings fall the way a real run's do.
            latents = latents + (target - latents) * (0.6 / (step + 1))
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 1000 - step * 20, {"latents": latents})
            if self.watcher is not None:
                self.watcher(step)
        pixels = ((latents[0, :3].permute(1, 2, 0).clamp(0, 1)) * 255).byte().numpy()
        image = Image.fromarray(pixels).resize((width, height))
        return type("Output", (), {"images": [image]})()
