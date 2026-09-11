"""The headless release runner must still catch missing Metal dependencies."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
from io import StringIO
import unittest
from unittest.mock import patch

import transformers

from desktop_smoke import smoke_test_metal, smoke_test_mlx, smoke_test_pipelines


@unittest.skipUnless(hasattr(transformers, "MetalConfig"), "Metal requires transformers>=5.3")
class DesktopMetalSmokeTests(unittest.TestCase):
    def test_headless_runner_checks_dependencies_then_reports_hardware_skip(self):
        output = StringIO()
        with patch("torch.backends.mps.is_available", return_value=False), redirect_stdout(output):
            smoke_test_metal()
        self.assertIn("dependency and quantizer checks passed", output.getvalue())
        self.assertIn("SKIP: Metal quantized load", output.getvalue())

    def test_headless_runner_fails_when_lazy_integration_is_missing(self):
        real_import = importlib.import_module

        def without_metal(name, *args, **kwargs):
            if name == "transformers.integrations.metal_quantization":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("desktop_smoke.importlib.import_module", side_effect=without_metal),
            self.assertRaisesRegex(ModuleNotFoundError, "metal_quantization"),
        ):
            smoke_test_metal()

    def test_headless_runner_fails_when_kernels_metadata_is_missing(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("transformers.utils.is_kernels_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "missing compatible kernels or its metadata"),
        ):
            smoke_test_metal()


class DesktopMlxSmokeTests(unittest.TestCase):
    """The same runner has to catch a bundle built without mlx-lm's models."""

    def test_the_runner_runs_a_model_or_says_why_it_cannot(self):
        output = StringIO()
        with redirect_stdout(output):
            smoke_test_mlx()

        self.assertRegex(
            output.getvalue(), "MLX load, forward pass and lens checks passed|SKIP: MLX models"
        )

    def test_the_runner_skips_without_mlx(self):
        output = StringIO()
        with (
            patch("mlx_runtime.mlx_available", return_value=False),
            redirect_stdout(output),
        ):
            smoke_test_mlx()

        self.assertIn("SKIP: MLX models", output.getvalue())


class DesktopPipelineSmokeTests(unittest.TestCase):
    """The same runner has to catch a bundle built without diffusers."""

    def test_the_runner_checks_every_class_an_image_repo_can_name(self):
        output = StringIO()
        with redirect_stdout(output):
            smoke_test_pipelines()

        self.assertIn("pipeline class checks passed", output.getvalue())

    def test_the_runner_names_the_classes_a_bundle_left_out(self):
        # These are reached by name out of model_index.json, so a bundle can
        # be built without them and nothing fails until a user loads a model.
        import diffusers

        with (
            patch.object(diffusers, "StableDiffusionXLPipeline", None),
            self.assertRaisesRegex(RuntimeError, "StableDiffusionXLPipeline"),
        ):
            smoke_test_pipelines()

    def test_the_runner_fails_when_attention_cannot_be_read(self):
        from diffusers.models.attention_processor import Attention

        with (
            patch.object(Attention, "get_attention_scores", None),
            self.assertRaisesRegex(RuntimeError, "cannot report attention"),
        ):
            smoke_test_pipelines()


if __name__ == "__main__":
    unittest.main()
