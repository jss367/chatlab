"""Both backends answer every question the engine interface names."""

import unittest

from chatlab.engine import Engine
from chatlab.mlx_runtime import MlxEngine
from chatlab.torch_engine import TorchEngine


class EngineInterfaceTests(unittest.TestCase):
    # Nothing type-checks the code, so this is what keeps the two engines
    # from drifting apart: a method added to one and to Engine but not the
    # other fails here rather than on the first model of that backend.
    # Neither constructor touches its model, so no weights are needed.

    def test_the_torch_engine_answers_the_interface(self):
        self.assertIsInstance(TorchEngine(None), Engine)

    def test_the_mlx_engine_answers_the_interface(self):
        self.assertIsInstance(MlxEngine(None, {}), Engine)


if __name__ == "__main__":
    unittest.main()
