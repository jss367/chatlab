"""The headless release runner must still catch missing Metal dependencies."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
from io import StringIO
import unittest
from unittest.mock import patch

import transformers

from desktop_smoke import smoke_test_metal


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


if __name__ == "__main__":
    unittest.main()
