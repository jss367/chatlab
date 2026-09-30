"""MLX work done from the threads the application really uses.

Since 0.31.2 every thread gets its own MLX stream, and an array built lazily
on one thread can only be evaluated on that thread; anywhere else MLX raises
``There is no Stream(gpu, N) in current thread``. ChatLab reads a checkpoint
on a ``chatlab-load`` thread, streams each reply from a new
``chatlab-conversation`` thread, and serves inspection from Gradio's worker
pool, whose threads resume a streaming handler in no fixed order.

The model here is a two-layer Llama saved to disk the way ``mlx_lm.convert``
saves one, with Llama 3 RoPE scaling: that keeps its frequencies in an
underscore attribute ``parameters()`` leaves out, so reading the checkpoint
with ``lazy=False`` leaves them unevaluated, exactly as OLMo 3's YaRN
attention does. Every thread below touches MLX before the model is read, so
none of them can inherit the load thread's stream number by accident.
"""

import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

import settings_sandbox
from mlx_support import HIDDEN, LAYERS, llama, mx, needs_mlx

LENS_SOURCE = "test/tiny-decoder"
MODEL_ID = "mlx-community/tiny-decoder-4bit"


def setUpModule():
    # The Jacobian lens test imports a lens, which is written down beside
    # the settings.
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


class Worker:
    """A long-lived thread that runs what it is handed, like a Gradio worker."""

    def __init__(self, name: str) -> None:
        self.tasks: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self.thread.start()
        # Claim this thread's own stream now, before anything is loaded.
        self.run(lambda: mx.eval(mx.ones((2,)) + 1))

    def _serve(self) -> None:
        while True:
            task, answer = self.tasks.get()
            if task is None:
                return
            try:
                answer.put((True, task()))
            except BaseException as error:  # noqa: BLE001 - handed back to the caller
                answer.put((False, error))

    def run(self, task):
        answer: queue.SimpleQueue = queue.SimpleQueue()
        self.tasks.put((task, answer))
        ok, value = answer.get(timeout=60)
        if not ok:
            raise value
        return value

    def stop(self) -> None:
        self.tasks.put((None, None))
        self.thread.join(timeout=10)


def on_new_thread(task, name: str):
    """Run ``task`` on a thread that ends afterwards, as a load or a reply does."""

    result: dict = {}

    def run():
        try:
            result["value"] = task()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            result["error"] = error

    thread = threading.Thread(target=run, name=name)
    thread.start()
    thread.join(timeout=60)
    if "error" in result:
        raise result["error"]
    return result["value"]


def write_snapshot(directory: Path, vocab: int) -> None:
    """A tiny Llama with Llama 3 RoPE scaling, saved as an MLX checkpoint."""

    from mlx.utils import tree_flatten

    config = {
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": vocab,
        "max_position_embeddings": 256,
        "tie_word_embeddings": True,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
        },
        "eos_token_id": 0,
    }
    mx.random.seed(3)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (directory / "config.json").write_text(json.dumps(config))


class MlxThreadTests(unittest.TestCase):
    """The hand-off itself, which needs no MLX."""

    def test_every_call_runs_on_the_same_thread_whoever_makes_it(self):
        from chatlab.mlx_runtime import MlxThread

        owner = MlxThread("test-mlx")
        seen = {owner.run(threading.get_ident)}
        seen.add(on_new_thread(lambda: owner.run(threading.get_ident), "caller"))
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen, {threading.get_ident()})

    def test_a_failure_reaches_the_caller_and_the_thread_carries_on(self):
        from chatlab.mlx_runtime import MlxThread

        owner = MlxThread("test-mlx")

        def fail():
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

        with self.assertRaisesRegex(RuntimeError, "no Stream"):
            owner.run(fail)
        self.assertEqual(owner.run(lambda: 2 + 2), 4)

    def test_a_call_made_from_the_thread_runs_in_place(self):
        # An engine method that calls another must not wait on itself.
        from chatlab.mlx_runtime import MlxThread

        owner = MlxThread("test-mlx")
        outer = owner.run(lambda: (threading.get_ident(), owner.run(threading.get_ident)))
        self.assertEqual(outer[0], outer[1])


@needs_mlx
class ThreadedMlxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tiny_tokenizer

        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)
        tokenizer = tiny_tokenizer.build()
        tokenizer.save_pretrained(cls.path)
        write_snapshot(cls.path, len(tokenizer))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.workers = [Worker("gradio-worker-1"), Worker("gradio-worker-2")]
        for worker in self.workers:
            self.addCleanup(worker.stop)
        self.manager = on_new_thread(self.load, "chatlab-load")

    def load(self):
        """What the loader does with an MLX checkpoint, minus the memory checks."""

        from chatlab import mlx_runtime
        from chatlab.model_runtime import ModelManager

        model, tokenizer, _ = mlx_runtime.read_mlx_model(self.path)
        manager = ModelManager()
        manager.model = model
        manager.tokenizer = tokenizer
        manager.engine = mlx_runtime.MlxEngine.from_snapshot(model, self.path)
        manager.model_id = MODEL_ID
        manager.kind = "mlx"
        manager.precision = "4-bit"
        manager.load_count = 1
        return manager

    def stream(self, max_new_tokens: int = 6):
        return self.manager.generate(
            [{"role": "user", "content": "the cat sat on the mat"}],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            seed=0,
        )

    def drain(self, stream, threads) -> list:
        """Resume ``stream`` once per step, taking the next thread each time."""

        updates = []
        for step in range(1000):
            worker = threads[step % len(threads)]
            update = worker.run(lambda: next(stream, None))
            if update is None:
                return updates
            updates.append(update)
        self.fail("The stream never ended.")

    def test_a_reply_streams_when_every_step_resumes_on_another_thread(self):
        updates = self.drain(self.stream(), self.workers)

        self.assertTrue(updates)
        self.assertTrue(updates[-1].metrics)
        self.assertTrue(updates[-1].prompt_metrics)
        self.assertFalse(self.manager.busy)

    def test_consecutive_replies_each_on_a_new_thread_agree(self):
        # The conversation pane gives every reply a thread of its own, which
        # ends with the reply. Greedy replies to one prompt must match.
        first = on_new_thread(lambda: list(self.stream()), "chatlab-conversation")
        second = on_new_thread(lambda: list(self.stream()), "chatlab-conversation")

        ids = [[metric["token_id"] for metric in run[-1].metrics] for run in (first, second)]
        self.assertTrue(ids[0])
        self.assertEqual(ids[0], ids[1])

    def test_a_stopped_reply_leaves_the_model_ready_for_the_next(self):
        stream = self.stream(max_new_tokens=50)
        self.workers[0].run(lambda: next(stream))
        # Stop: Gradio closes the handler from whichever worker is free.
        self.workers[1].run(stream.close)
        self.assertFalse(self.manager.busy)

        updates = self.drain(self.stream(), self.workers[::-1])
        self.assertTrue(updates[-1].metrics)

    def test_the_layer_inspector_and_the_cache_view_run_from_any_thread(self):
        ids = self.manager.tokenizer("the cat sat on the mat")["input_ids"]

        first = self.workers[0].run(lambda: self.manager.inspect(ids, 3, context_count=2))
        later = self.workers[1].run(lambda: self.manager.inspect(ids, 4))
        fresh = on_new_thread(lambda: self.load().inspect(ids, 4), "chatlab-load")
        cache = self.workers[0].run(lambda: self.manager.read_kv_cache(ids[:4], 1))

        self.assertEqual(len(first.layers), LAYERS + 1)
        self.assertEqual(len(first.attention), LAYERS)
        self.assertAlmostEqual(
            later.layers[-1]["probability"], fresh.layers[-1]["probability"], places=3
        )
        self.assertTrue(cache)

    def test_the_jacobian_lens_reads_from_any_thread(self):
        lens = self.path / "lens.pt"
        torch.manual_seed(7)
        torch.save({
            "J": {layer: torch.randn(HIDDEN, HIDDEN) for layer in range(LAYERS)},
            "source_layers": list(range(LAYERS)), "n_prompts": 50, "d_model": HIDDEN,
        }, lens)
        imported = self.workers[0].run(
            lambda: self.manager.import_jacobian_lens(str(lens), LENS_SOURCE)
        )
        ids = self.manager.tokenizer("the cat sat on the mat")["input_ids"]

        results = [
            worker.run(lambda: self.manager.inspect_jacobian(
                ids, 3, lens_id=imported["import_id"], load_id=self.manager.load_id,
            ).to_dict())
            for worker in self.workers
        ]

        self.assertEqual(results[0]["backend"], "mlx")
        self.assertEqual(len(results[0]["layers"]), LAYERS)
        for one, other in zip(results[0]["layers"], results[1]["layers"]):
            self.assertEqual(
                [c["token_id"] for c in one["candidates"]],
                [c["token_id"] for c in other["candidates"]],
            )
            np.testing.assert_allclose(
                [c["score"] for c in one["candidates"]],
                [c["score"] for c in other["candidates"]],
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
