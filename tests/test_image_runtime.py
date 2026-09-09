"""Recognizing, sizing and running a diffusers pipeline."""

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy
import torch

import image_runtime
import model_runtime
import settings_sandbox
from fake_pipeline import FakePipeline, FakeTokenizer
from image_runtime import ImageRequest
from model_runtime import (
    IMAGE_KIND,
    MODEL_WEIGHTS,
    TEXT_KIND,
    ModelBusy,
    ModelManager,
    cache_status,
    is_pipeline,
    pipeline_class,
    pipeline_components,
    pipeline_missing_files,
    pipeline_variant,
    pipeline_weight_bytes,
    weight_bytes_for,
)


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# What a Stable Diffusion repo's index really looks like, trimmed: the
# component pairs, the pipeline class, and a component the repo shipped
# without, which is written as a pair of nulls rather than left out.
INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "_diffusers_version": "0.31.0",
    "requires_safety_checker": True,
    "safety_checker": [None, None],
    "scheduler": ["diffusers", "PNDMScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


class PipelineLayoutTests(unittest.TestCase):
    """Reading a diffusers snapshot from the outside, without loading it."""

    def snapshot(self, root: str, files: dict[str, bytes]) -> Path:
        """A cache folder holding one snapshot of ``files``, as the hub lays it out."""

        folder = Path(root) / f"models--{MODEL.replace('/', '--')}"
        (folder / "refs").mkdir(parents=True)
        (folder / "refs" / "main").write_text("abc")
        snapshot = folder / "snapshots" / "abc"
        for name, content in files.items():
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes(content)
        return snapshot

    def whole(self) -> dict[str, bytes]:
        return {
            "model_index.json": json.dumps(INDEX).encode(),
            "scheduler/scheduler_config.json": b"{}",
            "text_encoder/config.json": b"{}",
            "text_encoder/model.safetensors": b"e" * 400,
            "tokenizer/tokenizer_config.json": b"{}",
            "tokenizer/vocab.json": b"{}",
            "unet/config.json": b"{}",
            "unet/diffusion_pytorch_model.safetensors": b"u" * 1000,
            "vae/config.json": b"{}",
            "vae/diffusion_pytorch_model.safetensors": b"v" * 200,
        }

    def test_a_pipeline_is_recognized_by_its_index_and_read_from_it(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, self.whole())

            self.assertTrue(is_pipeline(snapshot))
            self.assertEqual(pipeline_class(snapshot), "StableDiffusionPipeline")
            # The nulls and the underscored keys are not components, and the
            # order is the index's own.
            self.assertEqual(
                pipeline_components(snapshot),
                ("scheduler", "text_encoder", "tokenizer", "unet", "vae"),
            )

    def test_a_whole_pipeline_is_an_image_model_ready_to_load(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, self.whole())
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.kind, IMAGE_KIND)
            self.assertFalse(status.unsupported)
            self.assertTrue(status.complete)
            self.assertEqual(status.missing_files, ())

    def test_a_component_folder_that_never_arrived_is_missing(self):
        files = {
            name: content
            for name, content in self.whole().items()
            if not name.startswith("vae/")
        }
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.kind, IMAGE_KIND)
            self.assertEqual(status.missing_files, ("vae/",))
            self.assertFalse(status.complete)

    def test_a_component_folder_without_its_weights_is_missing_them(self):
        files = self.whole()
        del files["unet/diffusion_pytorch_model.safetensors"]
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.missing_files, (f"unet/{MODEL_WEIGHTS}",))

    def test_a_folder_with_no_config_of_its_own_needs_no_weights(self):
        """The tokenizer and the scheduler keep a config under a name of
        their own and no weights at all, so absence there is not a gap."""

        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, self.whole())

            self.assertEqual(pipeline_missing_files(snapshot), ())
            self.assertEqual(
                sorted(entry.name for entry in (snapshot / "tokenizer").iterdir()),
                ["tokenizer_config.json", "vocab.json"],
            )

    def test_a_component_missing_shards_is_incomplete_rather_than_ready(self):
        """A download cut off after the first shard leaves a folder with
        *some* weights. Reading that as whole sends Load cached into
        diffusers to fail on the shards that never arrived."""

        files = self.whole()
        del files["unet/diffusion_pytorch_model.safetensors"]
        files["unet/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
            {
                "weight_map": {
                    "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            }
        ).encode()
        files["unet/diffusion_pytorch_model-00001-of-00002.safetensors"] = b"u" * 500
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertEqual(
                status.missing_files,
                ("unet/diffusion_pytorch_model-00002-of-00002.safetensors",),
            )
            self.assertFalse(status.complete)

    def test_a_component_with_every_shard_is_ready(self):
        files = self.whole()
        del files["unet/diffusion_pytorch_model.safetensors"]
        files["unet/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
            {
                "weight_map": {
                    "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            }
        ).encode()
        files["unet/diffusion_pytorch_model-00001-of-00002.safetensors"] = b"u" * 500
        files["unet/diffusion_pytorch_model-00002-of-00002.safetensors"] = b"u" * 500
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.missing_files, ())
            self.assertTrue(status.complete)

    def test_a_component_with_weights_but_no_config_is_missing_the_config(self):
        """diffusers cannot construct a unet or a VAE without its config, so
        a download cut off before it arrived is incomplete, not whole."""

        files = self.whole()
        del files["unet/config.json"]
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.missing_files, ("unet/config.json",))
            self.assertFalse(status.complete)

    def test_a_sharded_component_with_no_config_is_missing_it_too(self):
        files = self.whole()
        del files["unet/config.json"]
        del files["unet/diffusion_pytorch_model.safetensors"]
        files["unet/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
            {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00001.safetensors"}}
        ).encode()
        files["unet/diffusion_pytorch_model-00001-of-00001.safetensors"] = b"u" * 900
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)

            self.assertEqual(
                cache_status(MODEL, Path(root)).missing_files, ("unet/config.json",)
            )

    def test_a_folder_with_neither_a_config_nor_weights_is_still_no_gap(self):
        # The tokenizer and the scheduler keep a config of their own name and
        # no weights at all; absence of both is what they look like.
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, self.whole())
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.missing_files, ())
            self.assertTrue(status.complete)

    def test_an_index_that_cannot_be_read_counts_as_missing_weights(self):
        files = self.whole()
        del files["unet/diffusion_pytorch_model.safetensors"]
        files["unet/diffusion_pytorch_model.safetensors.index.json"] = b"not json"
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, files)

            self.assertEqual(
                cache_status(MODEL, Path(root)).missing_files,
                (f"unet/{MODEL_WEIGHTS}",),
            )

    def test_a_pipeline_is_sized_by_the_one_weight_set_it_will_load(self):
        """A repo often ships a half-precision set beside the full one, and
        ``from_pretrained`` reads one of them. Counting both would double
        every component and refuse loads that fit."""

        files = self.whole()
        files["unet/diffusion_pytorch_model.fp16.safetensors"] = b"h" * 500
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            # The plain safetensors set, not it plus the variant.
            self.assertEqual(pipeline_weight_bytes(snapshot), 400 + 1000 + 200)
            self.assertEqual(weight_bytes_for(snapshot, IMAGE_KIND), 1600)

    def test_a_variant_only_repo_is_loaded_by_asking_for_that_variant(self):
        """The halves have to agree: pipeline_missing_files already calls
        such a snapshot complete, so a load that did not name the variant
        would fail on weights diffusers never looked for."""

        files = self.whole()
        files["unet/diffusion_pytorch_model.fp16.safetensors"] = files.pop(
            "unet/diffusion_pytorch_model.safetensors"
        )
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            self.assertEqual(pipeline_variant(snapshot), "fp16")
            self.assertEqual(cache_status(MODEL, Path(root)).missing_files, ())

    def test_a_repo_with_plain_weights_asks_for_no_variant(self):
        # Which is what diffusers wants asked of it in the common case.
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(pipeline_variant(self.snapshot(root, self.whole())))

    def test_the_variant_named_is_one_every_bare_component_ships(self):
        """from_pretrained takes one variant for the whole pipeline and can
        only fall back to unsuffixed files, so naming a variant that some
        components lack would leave those with nothing to fall back to."""

        files = self.whole()
        for component in ("unet", "vae"):
            for variant in ("fp16", "bf16"):
                files[f"{component}/diffusion_pytorch_model.{variant}.safetensors"] = (
                    b"w" * 100
                )
            del files[f"{component}/diffusion_pytorch_model.safetensors"]
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            # Both ship both, so either would load; the choice is stable.
            self.assertEqual(pipeline_variant(snapshot), "bf16")
            self.assertFalse(model_runtime.pipeline_variant_missing(snapshot))
            self.assertEqual(cache_status(MODEL, Path(root)).missing_files, ())

    def test_components_sharing_no_variant_are_reported_incomplete(self):
        # An fp16 unet beside a bf16 vae, neither with a plain set: the files
        # are all there and there is still no load to make of them, so saying
        # so beats naming one variant and failing on the other component.
        files = self.whole()
        files["unet/diffusion_pytorch_model.fp16.safetensors"] = files.pop(
            "unet/diffusion_pytorch_model.safetensors"
        )
        files["vae/diffusion_pytorch_model.bf16.safetensors"] = files.pop(
            "vae/diffusion_pytorch_model.safetensors"
        )
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)
            status = cache_status(MODEL, Path(root))

            self.assertTrue(model_runtime.pipeline_variant_missing(snapshot))
            self.assertIsNone(pipeline_variant(snapshot))
            self.assertEqual(status.missing_files, (MODEL_WEIGHTS,))
            self.assertFalse(status.complete)

    def test_a_component_with_a_plain_set_beside_a_variant_needs_no_variant(self):
        # A repo that ships both loads from the plain set, so a component
        # that only has the variant does not force one on everybody.
        files = self.whole()
        files["unet/diffusion_pytorch_model.fp16.safetensors"] = b"w" * 100
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            self.assertIsNone(pipeline_variant(snapshot))
            self.assertFalse(model_runtime.pipeline_variant_missing(snapshot))

    def test_a_pipeline_shipping_only_a_variant_is_sized_by_that(self):
        files = self.whole()
        files["unet/diffusion_pytorch_model.fp16.safetensors"] = files.pop(
            "unet/diffusion_pytorch_model.safetensors"
        )
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            self.assertEqual(pipeline_weight_bytes(snapshot), 1600)

    def test_a_shard_number_is_not_read_as_a_variant(self):
        files = self.whole()
        del files["unet/diffusion_pytorch_model.safetensors"]
        files["unet/diffusion_pytorch_model-00001-of-00002.safetensors"] = b"u" * 600
        files["unet/diffusion_pytorch_model-00002-of-00002.safetensors"] = b"u" * 400
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            # Both shards are the one set and are summed.
            self.assertEqual(pipeline_weight_bytes(snapshot), 1600)

    def test_a_text_checkpoint_is_still_a_text_model(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root,
                {
                    "config.json": b'{"model_type": "olmo3"}',
                    "model.safetensors": b"w" * 100,
                },
            )
            status = cache_status(MODEL, Path(root))

            self.assertEqual(status.kind, TEXT_KIND)
            self.assertTrue(status.complete)

    def test_the_pipeline_class_stands_in_for_an_architecture(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, self.whole())
            from model_runtime import list_cached_models

            (entry,) = list_cached_models(Path(root))

            self.assertEqual(entry.architecture, "StableDiffusionPipeline")
            # A pipeline's components need not share a dtype and the index
            # names none, so none is claimed.
            self.assertIsNone(entry.dtype)

    def test_an_unreadable_index_is_a_pipeline_that_cannot_be_measured(self):
        files = self.whole()
        files["model_index.json"] = b"not json"
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(root, files)

            self.assertTrue(is_pipeline(snapshot))
            self.assertEqual(pipeline_components(snapshot), ())
            self.assertIsNone(pipeline_weight_bytes(snapshot))
            self.assertIsNone(pipeline_class(snapshot))


class PipelineDtypeTests(unittest.TestCase):
    """How big a pipeline's weights get once loaded, read from the files."""

    def pipeline(self, root: Path, components: dict[str, tuple[str, int]]) -> Path:
        """A snapshot whose components hold safetensors of the given dtype and size."""

        import torch
        from safetensors.torch import save_file

        index = {"_class_name": "StableDiffusionPipeline"}
        for name, (dtype, side) in components.items():
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / "config.json").write_text("{}")
            index[name] = ["diffusers", "UNet2DConditionModel"]
            save_file(
                {"w": torch.zeros(side, side, dtype=getattr(torch, dtype))},
                root / name / "diffusion_pytorch_model.safetensors",
            )
        (root / "model_index.json").write_text(json.dumps(index))
        return root

    def test_one_file_reports_the_dtype_out_of_its_own_header(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.pipeline(Path(root), {"unet": ("float16", 32)})
            weights = snapshot / "unet" / "diffusion_pytorch_model.safetensors"

            self.assertEqual(model_runtime.safetensors_dtype(weights), "float16")

    def test_each_component_is_measured_in_its_own_dtype(self):
        """A pipeline's components need not agree, and one dtype applied to
        the whole byte count mis-scales the ones that differ."""

        with tempfile.TemporaryDirectory() as root:
            snapshot = self.pipeline(
                Path(root),
                {"unet": ("float16", 200), "text_encoder": ("float32", 100)},
            )
            unet = model_runtime._loaded_variant_bytes(
                model_runtime._component_weights(snapshot / "unet")
            )
            encoder = model_runtime._loaded_variant_bytes(
                model_runtime._component_weights(snapshot / "text_encoder")
            )

            self.assertEqual(
                model_runtime.component_dtype(snapshot / "unet"), "float16"
            )
            self.assertEqual(
                model_runtime.component_dtype(snapshot / "text_encoder"), "float32"
            )
            # A float32 load doubles the fp16 unet and leaves the fp32
            # encoder alone; a single dtype over the sum would scale both.
            self.assertEqual(
                model_runtime.pipeline_loaded_bytes(snapshot, "float32"),
                unet * 2 + encoder,
            )
            # And a float16 load halves the encoder, leaving the unet alone.
            self.assertEqual(
                model_runtime.pipeline_loaded_bytes(snapshot, "float16"),
                unet + encoder // 2,
            )

    def test_a_file_that_is_not_safetensors_reports_no_dtype(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weights.safetensors"
            path.write_bytes(b"nowhere near a header")

            self.assertIsNone(model_runtime.safetensors_dtype(path))

    def test_a_header_claiming_an_absurd_length_is_refused(self):
        # A truncated or hostile file, not something to allocate for.
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weights.safetensors"
            path.write_bytes((2**60).to_bytes(8, "little") + b"{}")

            self.assertIsNone(model_runtime.safetensors_dtype(path))

    def test_a_half_precision_pipeline_is_doubled_for_a_float32_load(self):
        """The bug this closes: on the CPU the loader asks for float32, so an
        fp16-only pipeline occupies twice its on-disk size. Measured as
        though it matched, it passed the safety reserve and then exhausted
        the machine."""

        with tempfile.TemporaryDirectory() as root:
            snapshot = self.pipeline(Path(root), {"unet": ("float16", 200)})
            stored = model_runtime.pipeline_weight_bytes(snapshot)

            with mock.patch.object(
                model_runtime, "system_memory", return_value=(64 * 1024**3, 32 * 1024**3)
            ):
                as_float32, _ = ModelManager._check_memory(
                    "org/pipe", snapshot, "float32", "cpu", kind=IMAGE_KIND
                )
                as_float16, _ = ModelManager._check_memory(
                    "org/pipe", snapshot, "float16", "cpu", kind=IMAGE_KIND
                )

        self.assertAlmostEqual(as_float32 / stored, 2.0, places=1)
        self.assertAlmostEqual(as_float16 / stored, 1.0, places=1)

    def test_a_component_shipping_only_pickles_is_assumed_half_precision(self):
        # A pickle has no cheap header to read. Pipelines ship as half
        # precision, and erring that way refuses a load rather than letting a
        # float32 CPU load double past the estimate and exhaust the machine.
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.pipeline(Path(root), {"unet": ("float16", 200)})
            weights = snapshot / "unet" / "diffusion_pytorch_model.safetensors"
            stored = weights.stat().st_size
            weights.rename(snapshot / "unet" / "diffusion_pytorch_model.bin")

            self.assertIsNone(model_runtime.component_dtype(snapshot / "unet"))
            self.assertEqual(model_runtime.ASSUMED_PIPELINE_DTYPE, "float16")
            # Assumed fp16, so a float32 load is estimated at twice the file.
            self.assertEqual(
                model_runtime.pipeline_loaded_bytes(snapshot, "float32"), stored * 2
            )

    def test_a_pipeline_on_cuda_has_to_fit_the_card_and_the_machine(self):
        """It is staged in host memory and then moved onto the card whole,
        with nothing offloaded. The combined pool would pass one that fits
        host memory and then fail inside .to("cuda"); the card alone would
        pass one that exhausts the machine while from_pretrained stages it.
        So both pools have to hold it and the tighter one decides - and the
        card is the one it lands on, not every visible one summed."""

        import torch
        from safetensors.torch import save_file

        # Two cards: the sum an offloading text load may use, and the one
        # card a pipeline is moved onto.
        one_card = (8 * 1024**3, 6 * 1024**3)
        all_cards = (16 * 1024**3, 12 * 1024**3)
        host = (64 * 1024**3, 48 * 1024**3)
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.pipeline(Path(root), {"unet": ("float16", 64)})
            # A checkpoint at the root as well, so the text leg has weights
            # to measure and reaches the pool decision at all.
            save_file(
                {"w": torch.zeros(64, 64, dtype=torch.float16)},
                snapshot / "model.safetensors",
            )
            (snapshot / "config.json").write_text('{"model_type": "olmo3"}')
            with (
                mock.patch.object(model_runtime, "cuda_memory", return_value=all_cards),
                mock.patch.object(
                    model_runtime, "cuda_device_memory", return_value=one_card
                ),
                mock.patch.object(model_runtime, "system_memory", return_value=host),
                mock.patch.object(model_runtime, "check_memory_for_load") as checked,
            ):
                ModelManager._check_memory(
                    "org/pipe", snapshot, "float16", "cuda", kind=IMAGE_KIND
                )
                image_pool = checked.call_args.kwargs["pool"]
                image_total = checked.call_args.args[2]

                ModelManager._check_memory(
                    "org/model", snapshot, "float16", "cuda", kind=TEXT_KIND
                )
                text_total = checked.call_args.args[2]

        self.assertEqual(image_pool, "both this GPU and this machine")
        # The tighter of the two, which here is the one card it lands on -
        # not their sum, and not every visible card summed.
        self.assertEqual(image_total, min(one_card[0], host[0]))
        self.assertLess(image_total, one_card[0] + host[0])
        self.assertLess(image_total, all_cards[0])
        # A text model does still get the offload pool it really uses, over
        # every card, because device_map="auto" really does spread it.
        self.assertEqual(text_total, all_cards[0] + host[0])


class KindAwareFitTests(unittest.TestCase):
    """The fit verdicts beside the model lists have to know the two kinds.

    Both halves of a verdict depend on it: a pipeline has no checkpoint at
    its root to measure, and on CUDA it is staged in host memory before it
    moves onto the card. Without either, every verdict beside an image model
    would be a text-shaped guess, which is what these verdicts exist to
    prevent.
    """

    def test_the_estimate_routes_a_pipeline_to_its_own_sizing(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = PipelineDtypeTests().pipeline(
                Path(root), {"unet": ("float16", 64)}
            )

            as_image = model_runtime.estimate_snapshot_bytes(
                snapshot, "float16", None, IMAGE_KIND
            )
            as_text = model_runtime.estimate_snapshot_bytes(snapshot, "float16")
            summed = model_runtime.pipeline_loaded_bytes(snapshot, "float16")

        self.assertEqual(as_image, summed)
        # A pipeline has no checkpoint at its root, so the text reading has
        # nothing to measure and says so rather than guessing.
        self.assertIsNone(as_text)

    def test_a_quantized_choice_does_not_shrink_a_pipeline_estimate(self):
        # The Metal quantizer is Transformers' own and a load has already
        # cleared the choice, so honouring it here would report a size no
        # load will produce.
        with tempfile.TemporaryDirectory() as root:
            snapshot = PipelineDtypeTests().pipeline(
                Path(root), {"unet": ("float16", 64)}
            )

            whole = model_runtime.estimate_snapshot_bytes(
                snapshot, "float16", None, IMAGE_KIND
            )
            asked_for_4bit = model_runtime.estimate_snapshot_bytes(
                snapshot, "float16", 4, IMAGE_KIND
            )

        self.assertEqual(asked_for_4bit, whole)

    ONE_CARD = (8 * 1024**3, 6 * 1024**3)
    ALL_CARDS = (16 * 1024**3, 12 * 1024**3)
    HOST = (64 * 1024**3, 48 * 1024**3)

    def memory(self):
        return (
            mock.patch.object(model_runtime, "cuda_memory", return_value=self.ALL_CARDS),
            mock.patch.object(
                model_runtime, "cuda_device_memory", return_value=self.ONE_CARD
            ),
            mock.patch.object(model_runtime, "system_memory", return_value=self.HOST),
        )

    def profile(self, backend):
        cards, card, host = self.memory()
        with cards, card, host:
            total, available, pool = model_runtime.memory_pool(backend)
        return model_runtime.DeviceProfile(
            backend=backend, total=total, available=available, pool=pool
        )

    def test_a_pipeline_on_cuda_reads_the_tighter_of_the_two_pools(self):
        profile = self.profile("cuda")

        cards, card, host = self.memory()
        with cards, card, host:
            for_images = profile.for_kind(IMAGE_KIND)

        self.assertEqual(profile.total, self.ALL_CARDS[0] + self.HOST[0])
        self.assertEqual(for_images.total, min(self.ONE_CARD[0], self.HOST[0]))
        self.assertEqual(for_images.pool, "both this GPU and this machine")

    def test_a_text_model_and_every_other_backend_read_unchanged(self):
        cuda = self.profile("cuda")
        metal = self.profile("mps")

        self.assertIs(cuda.for_kind(TEXT_KIND), cuda)
        # Only CUDA stages in host memory before moving across, so Metal and
        # the CPU already answer for both.
        self.assertIs(metal.for_kind(IMAGE_KIND), metal)


class PreviewTests(unittest.TestCase):
    """Turning a latent into a frame a person can look at."""

    def test_a_frame_is_scaled_to_the_preview_size_keeping_its_shape(self):
        latent = torch.randn(4, 8, 16)
        image = image_runtime.latent_preview(latent, "sd", size=64)

        self.assertEqual(image.size, (64, 32))
        self.assertEqual(image.mode, "RGB")

    def test_a_latent_with_an_unmeasured_channel_count_still_draws(self):
        """A 16-channel latent has no factors here. Its first three channels
        are not red, green and blue, but they do move with the picture."""

        image = image_runtime.latent_preview(torch.randn(16, 8, 8), "sd", size=32)

        self.assertEqual(image.size, (32, 32))

    def test_a_flat_latent_does_not_divide_by_its_own_zero_spread(self):
        image = image_runtime.latent_preview(torch.zeros(4, 8, 8), "sd", size=16)

        self.assertEqual(image.size, (16, 16))

    def test_a_frame_is_written_as_jpeg_and_a_heat_map_as_png(self):
        """Every frame travels with the run, and PNG cannot compress noise;
        the maps stay PNG because they need their alpha channel."""

        frame = image_runtime.latent_preview(torch.randn(4, 8, 8), "sd", size=32)

        self.assertTrue(
            image_runtime.data_uri(frame, quality=80).startswith("data:image/jpeg;base64,")
        )
        self.assertTrue(image_runtime.data_uri(frame).startswith("data:image/png;base64,"))

    def test_a_heat_map_is_scaled_to_the_picture_and_keeps_its_alpha(self):
        import base64
        import io

        from PIL import Image

        weights = numpy.zeros((4, 4), dtype="float32")
        weights[1, 1] = 1.0
        uri = image_runtime.heat_overlay(weights, 32, 24)
        raw = base64.b64decode(uri.split(",", 1)[1])
        overlay = Image.open(io.BytesIO(raw))

        self.assertEqual(overlay.size, (32, 24))
        self.assertEqual(overlay.mode, "RGBA")
        # The strongest cell's own centre is nearly opaque; a corner the
        # map gave nothing to is nearly clear.
        self.assertGreater(overlay.getpixel((12, 9))[3], 180)
        self.assertLess(overlay.getpixel((30, 22))[3], 60)
        # One hue throughout: opacity is the only channel carrying a value.
        for channel, value in enumerate(image_runtime.HEAT_COLOR):
            self.assertAlmostEqual(overlay.getpixel((12, 9))[channel], value, delta=2)

    def test_an_empty_map_does_not_divide_by_its_own_zero_ceiling(self):
        uri = image_runtime.heat_overlay(numpy.zeros((4, 4), dtype="float32"), 8, 8)

        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_the_latent_family_comes_from_the_pipeline_class(self):
        class StableDiffusionXLPipeline:
            pass

        class StableDiffusionPipeline:
            pass

        self.assertEqual(image_runtime.latent_family(StableDiffusionXLPipeline()), "sdxl")
        self.assertEqual(image_runtime.latent_family(StableDiffusionPipeline()), "sd")


class GuidanceReaderTests(unittest.TestCase):
    """Reading the classifier-free guidance split out of a denoiser's output."""

    def reading(self, sample):
        reader = image_runtime.GuidanceReader()
        reader(None, None, (sample,))
        return reader.take()

    def test_a_guided_batch_gives_the_vector_the_prompt_added(self):
        uncond = torch.zeros(1, 1, 2, 2)
        cond = torch.full((1, 1, 2, 2), 3.0)
        cond_norm, uncond_norm, delta = self.reading(torch.cat([uncond, cond]))

        self.assertAlmostEqual(uncond_norm, 0.0, places=5)
        self.assertAlmostEqual(cond_norm, 6.0, places=5)
        self.assertAlmostEqual(delta, 6.0, places=5)

    def test_an_unguided_batch_has_no_second_prediction_to_compare(self):
        cond_norm, uncond_norm, delta = self.reading(torch.full((1, 1, 2, 2), 3.0))

        self.assertAlmostEqual(cond_norm, 6.0, places=5)
        self.assertIsNone(uncond_norm)
        self.assertIsNone(delta)

    def test_an_output_object_is_read_through_its_sample(self):
        reader = image_runtime.GuidanceReader()
        sample = torch.cat([torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 3.0)])
        reader(None, None, type("Output", (), {"sample": sample})())

        self.assertAlmostEqual(reader.take()[2], 6.0, places=5)

    def test_a_reading_is_cleared_when_it_is_taken(self):
        reader = image_runtime.GuidanceReader()
        reader(None, None, (torch.ones(2, 1, 2, 2),))
        reader.take()

        self.assertEqual(reader.take(), (None, None, None))

    def test_an_output_with_nothing_readable_in_it_is_ignored(self):
        reader = image_runtime.GuidanceReader()
        reader(None, None, None)

        self.assertEqual(reader.calls, 0)
        self.assertEqual(reader.take(), (None, None, None))


class AttentionReaderTests(unittest.TestCase):
    """Turning cross-attention probabilities into one map per prompt token."""

    def reader(self, tokens=3, size=4):
        return image_runtime.AttentionReader(tokens, size)

    def probabilities(self, queries=4, keys=5, heads=2, batch=2):
        """A softmaxed ``[batch * heads, queries, keys]``, as a module gives it."""

        return torch.softmax(torch.randn(batch * heads, queries, keys), dim=-1)

    def test_a_step_of_maps_sums_to_one_at_every_cell(self):
        reader = self.reader(tokens=3, size=4)
        probabilities = self.probabilities(queries=16, keys=8)
        reader._total += reader._grid(probabilities[2:], torch)
        reader._modules += 1
        reader.recorded = 1
        reader.flush()

        maps = reader.collected()
        # Three token rows plus the padding row; see PADDING_ROW.
        self.assertEqual(maps.shape, (1, 4, 4, 4))
        numpy.testing.assert_allclose(maps[0].sum(axis=0), numpy.ones((4, 4)), atol=1e-5)

    def test_the_keys_past_the_prompt_are_summed_into_the_padding_row(self):
        reader = self.reader(tokens=2, size=2)
        # Every query gives all its attention to key 4, which is padding.
        probabilities = torch.zeros(2, 4, 6)
        probabilities[:, :, 4] = 1.0
        grid = reader._grid(probabilities, torch)

        numpy.testing.assert_allclose(grid[:2], numpy.zeros((2, 2, 2)), atol=1e-6)
        numpy.testing.assert_allclose(grid[2], numpy.ones((2, 2)), atol=1e-6)

    def test_the_modules_own_qk_norms_are_applied(self):
        """A module with norm_q or norm_k normalizes its projections before
        it computes attention, and the processor this wraps does that. Maps
        computed without them look valid and describe different
        probabilities from the ones that drew the picture."""

        from diffusers.models.attention_processor import Attention

        reader = self.reader(tokens=3, size=4)
        attn = Attention(
            query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, qk_norm="rms_norm"
        )
        self.assertIsNotNone(attn.norm_q)

        hidden = torch.randn(1, 16, 8)
        encoder = torch.randn(1, 5, 8)
        with_norms = reader._probabilities(attn, hidden, encoder, torch)

        # The same module with its norms taken away gives different
        # probabilities, which is what makes skipping them a wrong answer
        # rather than a rounding difference.
        attn.norm_q = None
        attn.norm_k = None
        without = reader._probabilities(attn, hidden, encoder, torch)

        self.assertFalse(torch.allclose(with_norms, without, atol=1e-4))

    def test_a_norm_that_cannot_be_applied_reports_no_map(self):
        # Reported unsupported rather than guessed at: a map that describes
        # different probabilities is worse than no map, because the page says
        # the token drove those pixels.
        from diffusers.models.attention_processor import Attention

        reader = self.reader(tokens=3, size=4)
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)
        attn.norm_q = lambda tensor: (_ for _ in ()).throw(RuntimeError("wrong shape"))

        self.assertIsNone(
            reader._probabilities(attn, torch.randn(1, 16, 8), torch.randn(1, 5, 8), torch)
        )

    def test_a_query_grid_that_is_not_square_is_refused(self):
        reader = self.reader()

        with self.assertRaises(ValueError):
            reader._grid(self.probabilities(queries=5, batch=1), torch)

    def test_a_step_with_nothing_recorded_keeps_an_empty_map(self):
        reader = self.reader()
        reader.flush()

        self.assertEqual(len(reader.steps), 1)
        # collected() still answers None: no module was ever read, so there
        # is nothing to show rather than a run of zeroes to puzzle over.
        self.assertIsNone(reader.collected())

    def test_a_module_this_cannot_read_is_counted_and_left_out(self):
        reader = self.reader()
        reader.record(object(), torch.zeros(1, 4, 4), torch.zeros(1, 5, 4))

        self.assertEqual(reader.skipped, 1)
        self.assertEqual(reader.recorded, 0)
        self.assertEqual(reader.too_large, 0)

    def test_the_budget_is_on_the_matrix_rather_than_the_query_count(self):
        """How many queries is too many depends on the model: a UNet that
        downsamples before its first attention layer asks about a quarter of
        the pixels one that does not asks about, and a resolution cap tight
        enough for the second refuses every layer of the first."""

        from diffusers.models.attention_processor import Attention

        reader = self.reader(tokens=3, size=4)
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)
        queries = 4096

        # Well inside the budget: 1 x 2 heads x 4096 x 77 x 4 bytes is 2.5 MB.
        reader.record(attn, torch.randn(1, queries, 8), torch.randn(1, 77, 8))
        self.assertEqual(reader.recorded, 1)
        self.assertEqual(reader.too_large, 0)

        # The same query count with a key sequence long enough to blow it.
        keys = image_runtime.MAX_ATTENTION_BYTES // (2 * queries * 4) + 1
        reader.record(attn, torch.randn(1, queries, 8), torch.randn(1, keys, 8))
        self.assertEqual(reader.recorded, 1)
        self.assertEqual(reader.too_large, 1)


class RunTests(unittest.TestCase):
    """A whole run against a pipeline shaped like a real one."""

    def run_pipeline(self, pipeline=None, **overrides):
        request = ImageRequest(
            **{
                "prompt": "a red bicycle",
                "steps": 4,
                "guidance_scale": 7.5,
                "seed": 1,
                "width": 32,
                "height": 32,
                **overrides,
            }
        )
        return image_runtime.run(pipeline or FakePipeline(), request)

    def test_a_run_reports_one_reading_per_step_the_scheduler_really_ran(self):
        # Five steps for four asked for: a scheduler can add one of its own,
        # and the readings are what happened rather than what was requested.
        run = self.run_pipeline(FakePipeline(steps_run=5), steps=4)

        self.assertEqual(run.steps_done, 5)
        self.assertEqual([reading.step for reading in run.readings], [1, 2, 3, 4, 5])
        self.assertFalse(run.stopped)
        self.assertIsNotNone(run.image)
        self.assertEqual(run.image.size, (32, 32))

    def test_the_first_step_has_no_movement_to_report(self):
        run = self.run_pipeline()

        self.assertIsNone(run.readings[0].latent_change)
        self.assertTrue(all(r.latent_change is not None for r in run.readings[1:]))

    def test_a_guided_run_measures_the_pull_and_an_unguided_one_does_not(self):
        guided = self.run_pipeline()
        unguided = self.run_pipeline(guidance_scale=1.0)

        for reading in guided.readings:
            self.assertIsNotNone(reading.guidance_norm)
            self.assertIsNotNone(reading.guidance_share)
        for reading in unguided.readings:
            self.assertIsNone(reading.guidance_norm)
            self.assertIsNone(reading.guidance_share)
            self.assertIsNotNone(reading.cond_norm)

    def test_every_step_carries_a_frame(self):
        run = self.run_pipeline()

        for reading in run.readings:
            self.assertTrue(reading.preview.startswith("data:image/jpeg;base64,"))

    def test_the_prompt_tokens_key_the_maps_and_lose_the_word_marker(self):
        run = self.run_pipeline()

        self.assertEqual(
            [token["text"] for token in run.tokens],
            ["<|startoftext|>", "a", "red", "bicycle", "<|endoftext|>"],
        )
        self.assertTrue(run.tokens[0]["special"])
        self.assertFalse(run.tokens[1]["special"])
        self.assertEqual(
            run.attention.shape,
            (run.steps_done, len(run.tokens) + 1, image_runtime.MAP_SIZE, image_runtime.MAP_SIZE),
        )
        self.assertEqual(run.attention_note, "")

    def test_the_shares_over_the_prompt_and_the_padding_sum_to_one(self):
        run = self.run_pipeline()
        shares = image_runtime.token_shares(run)

        self.assertEqual(len(shares), len(run.tokens))
        self.assertAlmostEqual(
            sum(shares) + image_runtime.padding_share(run), 1.0, places=4
        )

    def test_a_step_can_be_asked_for_on_its_own_or_averaged(self):
        run = self.run_pipeline()

        averaged = image_runtime.step_maps(run, 0)
        first = image_runtime.step_maps(run, 1)
        numpy.testing.assert_allclose(averaged, run.attention.mean(axis=0))
        numpy.testing.assert_allclose(first, run.attention[0])
        # Out of range falls back to the average rather than raising.
        numpy.testing.assert_allclose(image_runtime.step_maps(run, 999), averaged)

    def test_one_token_map_can_be_read_and_the_padding_sits_past_it(self):
        run = self.run_pipeline()

        self.assertEqual(
            image_runtime.token_map(run, 1).shape,
            (image_runtime.MAP_SIZE, image_runtime.MAP_SIZE),
        )
        self.assertIsNotNone(image_runtime.token_map(run, len(run.tokens)))
        self.assertIsNone(image_runtime.token_map(run, len(run.tokens) + 1))
        self.assertIsNone(image_runtime.token_map(run, -1))

    def test_tokens_the_encoder_never_saw_are_dropped_from_the_strip(self):
        """A pipeline can encode with a shorter limit than its tokenizer's,
        and the tokens past it were never keyed on. Keeping them left rows of
        zeros standing in for words the model never read, with the padding
        row stranded past them where nothing looks."""

        pipeline = FakePipeline()
        # Six words plus the two special tokens, but the encoder keys on
        # four of them.
        prompt = "a red bicycle on a beach"
        pipeline.unet.ask_the_prompt = lambda hidden: pipeline.unet._processors[
            "blocks.0.attn2.processor"
        ](
            pipeline.unet.attn2,
            torch.randn(hidden.shape[0], 64, 16),
            torch.randn(hidden.shape[0], 4, 16),
        )

        run = self.run_pipeline(pipeline, prompt=prompt, steps=2)

        self.assertEqual(len(run.tokens), 4)
        self.assertEqual(
            [token["text"] for token in run.tokens],
            ["<|startoftext|>", "a", "red", "bicycle"],
        )
        # Four token rows and the padding row, which is the last one.
        self.assertEqual(run.attention.shape[1], 5)
        self.assertEqual(len(image_runtime.token_shares(run)), 4)
        self.assertIsNotNone(image_runtime.padding_share(run))
        self.assertAlmostEqual(
            sum(image_runtime.token_shares(run)) + image_runtime.padding_share(run),
            1.0,
            places=4,
        )
        # No row of zeros standing in for a word the model never read.
        for share in image_runtime.token_shares(run):
            self.assertGreater(share, 0.0)

    def test_a_prompt_within_the_key_count_keeps_every_token(self):
        run = self.run_pipeline(prompt="a red bicycle", steps=2)

        self.assertEqual(len(run.tokens), 5)
        self.assertEqual(run.attention.shape[1], 6)

    def test_turning_attention_off_keeps_the_rest_and_says_why(self):
        run = self.run_pipeline(record_attention=False)

        self.assertIsNone(run.attention)
        self.assertEqual(run.tokens, [])
        self.assertIn("not recorded", run.attention_note)
        self.assertEqual(run.steps_done, 4)
        self.assertTrue(all(r.guidance_share is not None for r in run.readings))

    def test_a_model_too_large_to_map_says_so_and_says_what_to_do(self):
        # Silently handing back no maps was the bug: the budget is there to
        # bound one allocation, not to refuse the reading.
        pipeline = FakePipeline()
        with mock.patch.object(image_runtime, "MAX_ATTENTION_BYTES", 8):
            run = self.run_pipeline(pipeline, steps=2)

        self.assertIsNone(run.attention)
        self.assertIn("larger than the", run.attention_note)
        self.assertIn("Draw a smaller picture", run.attention_note)
        # The picture itself is unaffected: the maps are read alongside.
        self.assertIsNotNone(run.image)
        self.assertEqual(run.steps_done, 2)

    def test_a_pipeline_whose_attention_was_never_reached_reports_no_tokens(self):
        """The tokens are only there to key the maps. A strip with nothing to
        shade it by and nothing behind a click is worse than the note."""

        pipeline = FakePipeline()
        pipeline.unet.ask_the_prompt = lambda encoder_hidden_states: None

        run = self.run_pipeline(pipeline, steps=2)

        self.assertIsNone(run.attention)
        self.assertEqual(run.tokens, [])
        self.assertIn("No cross-attention module", run.attention_note)
        self.assertEqual(run.steps_done, 2)

    def test_a_pipeline_with_no_tokenizer_has_no_tokens_to_map(self):
        pipeline = FakePipeline()
        pipeline.tokenizer = None

        run = self.run_pipeline(pipeline)

        self.assertIsNone(run.attention)
        self.assertIn("no tokenizer", run.attention_note)
        self.assertEqual(run.steps_done, 4)

    def test_a_stop_between_steps_keeps_the_trajectory_and_drops_the_picture(self):
        stop = threading.Event()
        pipeline = FakePipeline()
        request = ImageRequest(prompt="a red bicycle", steps=6, seed=1, width=32, height=32)

        def halt(reading):
            if reading.step == 2:
                stop.set()

        run = image_runtime.run(pipeline, request, cancel=stop, on_step=halt)

        self.assertTrue(run.stopped)
        self.assertEqual(run.steps_done, 2)
        self.assertIsNone(run.image)

    def test_the_pipeline_is_left_as_it_was_found(self):
        """The hook and the recording processors are the run's own, so a
        second run does not stack another layer on the first one's."""

        pipeline = FakePipeline()
        before = pipeline.unet.attn_processors

        self.run_pipeline(pipeline)

        after = pipeline.unet.attn_processors
        self.assertEqual(list(after), list(before))
        for name, processor in after.items():
            self.assertIs(processor, before[name])
        self.assertFalse(pipeline.unet._forward_hooks)

    def test_a_run_is_stamped_with_the_load_that_drew_it(self):
        run = image_runtime.run(
            FakePipeline(),
            ImageRequest(prompt="x", steps=2, seed=1, width=32, height=32),
            model_id="org/pipe",
            load_id="org/pipe#3",
        )

        self.assertEqual(run.model_id, "org/pipe")
        self.assertEqual(run.load_id, "org/pipe#3")
        self.assertGreater(run.seconds, 0)

    def test_only_the_arguments_the_pipeline_takes_are_passed(self):
        class Fixed(FakePipeline):
            def __call__(self, prompt=None, num_inference_steps=30, callback_on_step_end=None):
                for step in range(num_inference_steps):
                    callback_on_step_end(self, step, 10, {"latents": torch.randn(1, 4, 4, 4)})
                return type("Output", (), {"images": []})()

        run = self.run_pipeline(Fixed(), steps=2)

        self.assertEqual(run.steps_done, 2)
        self.assertIsNone(run.image)


class GuidanceSummaryTests(unittest.TestCase):
    """The headline numbers a finished run is described by."""

    def readings(self, shares, changes):
        return [
            image_runtime.StepReading(
                step=position + 1,
                timestep=100 - position,
                preview="",
                latent_change=change,
                uncond_norm=1.0 if share is not None else None,
                guidance_norm=share,
                guidance_share=share,
            )
            for position, (share, change) in enumerate(zip(shares, changes))
        ]

    def test_the_settled_step_is_where_the_movement_stopped_mattering(self):
        # Moves of 0.4, 0.5, 0.02 and 0.01: from step 4 on, nothing moved
        # more than a tenth of the largest move.
        summary = image_runtime.guidance_summary(
            self.readings([0.2] * 5, [None, 0.4, 0.5, 0.02, 0.01])
        )

        self.assertEqual(summary["step_count"], 5)
        self.assertEqual(summary["settled_step"], 4)
        self.assertAlmostEqual(summary["final_latent_change"], 0.01)

    def test_the_peak_pull_is_named_with_its_own_step(self):
        summary = image_runtime.guidance_summary(
            self.readings([0.1, 0.9, 0.2], [None, 0.5, 0.1])
        )

        self.assertAlmostEqual(summary["peak_guidance_share"], 0.9)
        self.assertEqual(summary["peak_guidance_step"], 2)
        self.assertAlmostEqual(summary["mean_guidance_share"], 0.4)

    def test_the_peak_pull_keeps_its_own_step_when_a_reading_is_missing(self):
        """A step whose guidance the hook could not read drops out of the
        pulls and not out of the readings, so an index into one is not an
        index into the other and the peak would be named at the wrong step."""

        readings = self.readings([None, 0.1, 0.9, 0.2], [None, 0.5, 0.4, 0.1])
        summary = image_runtime.guidance_summary(readings)

        # The 0.9 belongs to step 3, not to step 2 as a bare index would say.
        self.assertAlmostEqual(summary["peak_guidance_share"], 0.9)
        self.assertEqual(summary["peak_guidance_step"], 3)
        # And the mean is over the pulls that exist, not over every step.
        self.assertAlmostEqual(summary["mean_guidance_share"], (0.1 + 0.9 + 0.2) / 3)

    def test_an_unguided_run_is_summarized_without_a_pull(self):
        summary = image_runtime.guidance_summary(
            self.readings([None, None], [None, 0.3])
        )

        self.assertEqual(summary["step_count"], 2)
        self.assertNotIn("mean_guidance_share", summary)
        self.assertAlmostEqual(summary["final_latent_change"], 0.3)

    def test_no_readings_at_all_is_an_empty_summary(self):
        self.assertEqual(image_runtime.guidance_summary([]), {"step_count": 0})


class TorchSeedTests(unittest.TestCase):
    """The seed has to be one torch's generator will take.

    Named apart from the Images page's own SeedTests, which cover how the
    box and the randomize checkbox pick one; this covers the ceiling torch
    imposes on whatever they picked.
    """

    def test_a_seed_past_torch_maximum_is_pulled_into_range(self):
        # NumPy takes any non-negative integer however large, so the Chat
        # page's seed only floors. torch raises above its own maximum, which
        # would fail the draw outright rather than reproduce a different one.
        self.assertEqual(image_runtime.usable_seed(2**70), image_runtime.MAX_SEED)
        self.assertEqual(image_runtime.usable_seed(-1), 0)
        self.assertEqual(image_runtime.usable_seed(42), 42)
        self.assertEqual(image_runtime.usable_seed(None), 0)
        self.assertEqual(image_runtime.usable_seed(float("inf")), 0)

    def test_an_out_of_range_seed_still_draws(self):
        run = image_runtime.run(
            FakePipeline(),
            ImageRequest(prompt="x", steps=2, seed=2**70, width=32, height=32),
        )

        self.assertEqual(run.steps_done, 2)
        self.assertIsNotNone(run.image)


class ManagerImageRunTests(unittest.TestCase):
    """What the manager guards around an image run."""

    def loaded(self) -> ModelManager:
        manager = ModelManager()
        manager.pipeline = FakePipeline()
        manager.kind = IMAGE_KIND
        manager.model_id = "org/pipe"
        manager.device_name = "CPU"
        return manager

    def request(self, **overrides):
        return ImageRequest(
            **{"prompt": "a red bicycle", "steps": 2, "seed": 1, "width": 32, "height": 32, **overrides}
        )

    def test_a_pipeline_in_memory_is_loaded_without_being_a_text_model(self):
        manager = self.loaded()

        self.assertTrue(manager.image_loaded)
        self.assertFalse(manager.loaded)
        self.assertTrue(manager.in_memory)
        self.assertEqual(manager.load_id, "org/pipe#0")

    def test_a_run_holds_the_lock_and_the_slot_and_gives_both_back(self):
        manager = self.loaded()

        run = manager.generate_image(self.request())

        self.assertEqual(run.steps_done, 2)
        self.assertEqual(run.load_id, "org/pipe#0")
        self.assertFalse(manager.busy)
        self.assertFalse(manager._lock.locked())

    def test_a_second_run_is_refused_rather_than_queued(self):
        manager = self.loaded()
        self.assertTrue(manager.reserve_generation())
        self.addCleanup(manager.release_generation)

        with self.assertRaises(ModelBusy):
            manager.generate_image(self.request())

    def test_running_out_of_memory_advises_something_a_reader_can_see(self):
        # A conversation and a response length mean nothing to someone who
        # was drawing a picture.
        from model_runtime import OutOfMemoryError

        class Full(FakePipeline):
            def __call__(self, *args, **kwargs):
                raise RuntimeError("MPS backend out of memory")

        manager = self.loaded()
        manager.pipeline = Full()

        with self.assertRaises(OutOfMemoryError) as caught:
            manager.generate_image(self.request())

        message = str(caught.exception)
        self.assertIn("Draw a smaller picture", message)
        self.assertIn("lower the step count", message)
        self.assertNotIn("conversation", message)
        self.assertFalse(manager.busy)

    def test_a_run_without_a_pipeline_says_so_and_frees_the_slot(self):
        manager = ModelManager()

        with self.assertRaisesRegex(RuntimeError, "No image model is loaded"):
            manager.generate_image(self.request())

        self.assertFalse(manager.busy)
        self.assertFalse(manager._lock.locked())

    def test_unloading_clears_the_pipeline_and_the_kind(self):
        manager = self.loaded()

        manager.unload()

        self.assertIsNone(manager.pipeline)
        self.assertIsNone(manager.kind)
        self.assertFalse(manager.in_memory)
        self.assertIsNone(manager.load_id)

    def test_a_stop_keeps_the_step_it_was_pressed_during(self):
        """Stop promises to end the run after the step it is on. The
        pipeline calls the step-end callback after a step has run, so
        raising before the reading was taken threw that step's frame and
        maps away and reported the run one step shorter than it got."""

        manager = self.loaded()
        # The watcher runs after the callback, so the cancellation is seen
        # at the end of the *next* step - the one it was pressed during.
        manager.pipeline = FakePipeline(
            watcher=lambda step: manager.stop_image_run() if step == 0 else None
        )

        run = manager.generate_image(self.request(steps=6))

        self.assertTrue(run.stopped)
        self.assertEqual(run.steps_done, 2)
        self.assertEqual([reading.step for reading in run.readings], [1, 2])
        # And the maps were flushed for both, not just the first.
        self.assertEqual(len(run.attention), 2)

    def test_a_run_can_be_stopped_through_the_manager(self):
        manager = self.loaded()

        run = manager.generate_image(
            self.request(steps=6), on_step=lambda reading: manager.stop_image_run()
        )

        self.assertTrue(run.stopped)
        self.assertEqual(run.steps_done, 1)
        self.assertFalse(manager.busy)

    def test_stopping_with_nothing_drawing_says_so_and_leaves_no_token_behind(self):
        """A Stop pressed while nothing is running must not be inherited by
        the next run, and a refused Draw must not clear a running one's."""

        manager = self.loaded()

        self.assertFalse(manager.stop_image_run())
        # And the next run is unaffected by that press.
        run = manager.generate_image(self.request(steps=3))

        self.assertFalse(run.stopped)
        self.assertEqual(run.steps_done, 3)

    def test_the_token_belongs_to_the_run_holding_the_slot(self):
        # Two tabs: the first is drawing and has pressed Stop; the second
        # presses Draw and is refused. The refusal must not lose the first
        # tab's cancellation, which is what a page-owned token did.
        manager = self.loaded()
        refused: list = []

        def second_tab(reading):
            manager.stop_image_run()
            try:
                manager.generate_image(self.request(steps=2))
            except ModelBusy as error:
                refused.append(error)

        run = manager.generate_image(self.request(steps=6), on_step=second_tab)

        self.assertEqual(len(refused), 1)
        self.assertTrue(run.stopped)
        self.assertEqual(run.steps_done, 1)

    def test_the_token_is_gone_once_the_run_ends(self):
        manager = self.loaded()

        manager.generate_image(self.request(steps=2))

        self.assertIsNone(manager._image_cancel)
        self.assertFalse(manager.stop_image_run())


class PipelineLoaderTests(unittest.TestCase):
    """What the loader asks diffusers for, and what it refuses to ask.

    Driven through ``_load_locked`` with a stand-in torch, the way the
    quantized text-load tests are: patching the real ``torch.backends.mps``
    to claim a Metal machine leaves ``_release_device_cache`` calling
    ``torch.mps.empty_cache()`` on a runner that has no MPS backend, which
    raises. A namespace that says what the branch under test needs, and
    nothing else, does not depend on the host at all.
    """

    def load_pipeline(self, precision: str = "full", mps: bool = True):
        """Read a pipeline through the loader; return the manager and the kwargs."""

        manager = ModelManager()
        calls = []

        def from_pretrained(path, **kwargs):
            calls.append(kwargs)
            return FakePipeline()

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: mps)
            ),
            float16="torch.float16",
            float32="torch.float32",
        )
        fake_diffusers = types.SimpleNamespace(
            DiffusionPipeline=types.SimpleNamespace(from_pretrained=from_pretrained)
        )
        with (
            mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}),
            mock.patch.object(manager, "_cap_mps_memory", return_value=None),
            mock.patch.object(manager, "_check_memory", return_value=(None, None)) as check,
            mock.patch.object(manager, "_release_device_cache"),
            mock.patch("model_runtime.allocated_bytes", return_value=None),
        ):
            device = manager._load_locked(
                "org/pipe", Path("/snap"), fake_torch, precision=precision, kind=IMAGE_KIND
            )
        return manager, device, calls, check

    def test_a_quantized_precision_is_noted_and_not_applied_to_a_pipeline(self):
        """The Metal quantizer is Transformers' own, and a pipeline is
        several models of which only some are Transformers ones."""

        manager, device, calls, check = self.load_pipeline(precision="4-bit")

        self.assertEqual(manager.kind, IMAGE_KIND)
        self.assertEqual(manager.precision, "full")
        self.assertNotIn("quantization_config", calls[0])
        self.assertEqual(calls[0]["torch_dtype"], "torch.float16")
        self.assertTrue(calls[0]["local_files_only"])
        self.assertTrue(manager.image_loaded)
        self.assertFalse(manager.loaded)
        self.assertEqual(manager.pipeline.device, "mps")
        self.assertEqual(device, "Apple Metal (MPS)")
        # And the check was told which kind it was sizing.
        self.assertEqual(check.call_args.kwargs["kind"], IMAGE_KIND)
        self.assertIsNone(check.call_args.kwargs["bits"])

    def test_the_variant_a_repo_needs_reaches_diffusers(self):
        with mock.patch.object(model_runtime, "pipeline_variant", return_value="fp16"):
            _manager, _device, calls, _check = self.load_pipeline()

        self.assertEqual(calls[0]["variant"], "fp16")

    def test_a_plain_repo_is_not_given_a_variant_to_look_for(self):
        with mock.patch.object(model_runtime, "pipeline_variant", return_value=None):
            _manager, _device, calls, _check = self.load_pipeline()

        self.assertNotIn("variant", calls[0])

    def test_a_pipeline_on_the_cpu_is_read_as_float32_and_stays_there(self):
        manager, device, calls, _check = self.load_pipeline(mps=False)

        self.assertEqual(calls[0]["torch_dtype"], "torch.float32")
        self.assertEqual(device, "CPU")
        # Nothing to move onto, so .to() is never called.
        self.assertEqual(manager.pipeline.device, "cpu")


class TokenizerTests(unittest.TestCase):
    """Reading the prompt's tokens off a pipeline."""

    def test_a_tokenizer_that_will_not_answer_costs_the_maps_only(self):
        class Broken(FakeTokenizer):
            def __call__(self, text, truncation=True, **kwargs):
                raise ValueError("no vocabulary")

        pipeline = FakePipeline(tokenizer=Broken())

        self.assertEqual(image_runtime.prompt_tokens(pipeline, "a red bicycle"), [])


if __name__ == "__main__":
    unittest.main()
