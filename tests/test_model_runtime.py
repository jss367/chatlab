import json
import re
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import settings
import settings_sandbox
import tiny_tokenizer
import model_runtime
from model_runtime import (
    IMAGE_KIND,
    MIN_MODEL_POSITION_LIMIT,
    MODEL_WEIGHTS,
    SCORE_TOKEN_LIMIT,
    SEARCH_SCAN_LIMIT,
    ModelManager,
    cache_status,
    encode_for_scoring,
    format_bytes,
    generation_prefill_token_limit,
    list_cached_models,
    score_token_limit,
    search_hub_models,
    split_context_and_text,
    validate_model_id,
)


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


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
    """What the cache inspection reports for the states a model can be in."""

    MODEL = "allenai/Olmo-3-7B-Think"
    COMMIT = "d97e442d7cc678210054dbcc9b440894d62c89a4"
    CONFIG = b'{"model_type": "olmo3"}'

    def folder(self, root: str) -> Path:
        blobs = Path(root) / "models--allenai--Olmo-3-7B-Think" / "blobs"
        blobs.mkdir(parents=True)
        return blobs

    def snapshot(self, root: str, files: dict[str, bytes]) -> Path:
        """Lay files out the way ``huggingface_hub`` does: blobs plus symlinks."""

        blobs = self.folder(root)
        model = blobs.parent
        (model / "refs").mkdir()
        (model / "refs" / "main").write_text(self.COMMIT)
        snapshot = model / "snapshots" / self.COMMIT
        snapshot.mkdir(parents=True)
        for index, (name, content) in enumerate(files.items()):
            blob = blobs / f"blob{index}"
            blob.write_bytes(content)
            (snapshot / name).symlink_to(blob)
        return snapshot

    def shard_index(self, *shards: str) -> bytes:
        weight_map = {f"layer.{i}.weight": shard for i, shard in enumerate(shards)}
        return json.dumps({"metadata": {}, "weight_map": weight_map}).encode()

    def test_an_unknown_model_is_absent(self):
        with tempfile.TemporaryDirectory() as root:
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.present)
        self.assertFalse(status.complete)
        self.assertEqual(status.total_bytes, 0)

    def test_a_single_weights_file_and_config_make_a_loadable_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root, {"config.json": self.CONFIG, "model.safetensors": b"x" * 100}
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.complete)
        self.assertEqual(status.missing_files, ())
        self.assertEqual(status.cached_bytes, 100 + len(self.CONFIG))

    def test_every_shard_the_index_names_makes_a_loadable_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root,
                {
                    "config.json": self.CONFIG,
                    "model.safetensors.index.json": self.shard_index("a.st", "b.st"),
                    "a.st": b"x",
                    "b.st": b"x",
                },
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.complete)

    def test_config_and_tokenizer_alone_are_not_a_model(self):
        """Another tool's ``AutoTokenizer`` call leaves finished blobs but no weights."""

        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, {"config.json": self.CONFIG, "tokenizer.json": b"{}"})
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.present)
        self.assertFalse(status.complete)
        self.assertEqual(status.partial_files, 0)
        self.assertEqual(status.missing_files, (MODEL_WEIGHTS,))

    def test_shards_the_index_names_but_the_snapshot_lacks_are_reported(self):
        """A download stopped between shards leaves no ``.incomplete`` file."""

        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root,
                {
                    "config.json": self.CONFIG,
                    "model.safetensors.index.json": self.shard_index(
                        "model-00001-of-00003.safetensors",
                        "model-00002-of-00003.safetensors",
                        "model-00003-of-00003.safetensors",
                    ),
                    "model-00001-of-00003.safetensors": b"x" * 10,
                },
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.complete)
        self.assertEqual(
            status.missing_files,
            ("model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"),
        )

    def test_a_malformed_weight_index_is_incomplete(self):
        malformed_indexes = (
            b"[]",
            b'{"weight_map": {}}',
            b'{"weight_map": {"layer": null}}',
            b'{"weight_map": {"layer": 42}}',
        )
        for index in malformed_indexes:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as root:
                self.snapshot(
                    root,
                    {
                        "config.json": self.CONFIG,
                        "model.safetensors.index.json": index,
                    },
                )
                status = cache_status(self.MODEL, Path(root))

            self.assertFalse(status.complete)
            self.assertEqual(status.missing_files, (MODEL_WEIGHTS,))

    def test_a_link_whose_blob_was_deleted_does_not_count_as_weights(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(
                root, {"config.json": self.CONFIG, "model.safetensors": b"x" * 100}
            )
            (snapshot / "model.safetensors").resolve().unlink()
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.complete)
        self.assertEqual(status.missing_files, (MODEL_WEIGHTS,))

    def test_a_snapshot_of_plain_files_and_no_blobs_is_complete(self):
        """Where the cache sits on a filesystem without symlinks, the hub
        moves each finished file into the snapshot and leaves ``blobs/``
        empty or absent; the model is no less loadable for it."""

        for blobs_dir in (True, False):
            with self.subTest(blobs_dir=blobs_dir), tempfile.TemporaryDirectory() as root:
                model = Path(root) / "models--allenai--Olmo-3-7B-Think"
                if blobs_dir:
                    (model / "blobs").mkdir(parents=True)
                (model / "refs").mkdir(parents=True)
                (model / "refs" / "main").write_text(self.COMMIT)
                snapshot = model / "snapshots" / self.COMMIT
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_bytes(self.CONFIG)
                (snapshot / "model.safetensors").write_bytes(b"x" * 100)
                status = cache_status(self.MODEL, Path(root))

                self.assertTrue(status.present)
                self.assertTrue(status.complete)
                self.assertEqual(status.missing_files, ())
                self.assertEqual(status.cached_bytes, 100 + len(self.CONFIG))

    def test_blobs_without_a_resolvable_snapshot_lack_everything(self):
        with tempfile.TemporaryDirectory() as root:
            (self.folder(root) / "abc").write_bytes(b"x" * 10)
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.present)
        self.assertEqual(status.missing_files, ("config.json", MODEL_WEIGHTS))

    def test_a_stray_partial_blob_does_not_unmake_a_complete_snapshot(self):
        """The blob folder is shared by every revision, so a leftover from
        another one, or from a file the model never loads, says nothing about
        whether ``main`` can load."""

        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(
                root, {"config.json": self.CONFIG, "model.safetensors": b"x" * 100}
            )
            blobs = snapshot.parents[1] / "blobs"
            (blobs / "other.1234.incomplete").write_bytes(b"x" * 40)
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.complete)
        self.assertEqual(status.missing_files, ())
        self.assertEqual(status.partial_files, 1)
        self.assertEqual(status.partial_bytes, 40)

    def test_a_partial_blob_the_snapshot_needs_shows_up_as_missing(self):
        """The hub links a file into the snapshot only once it has finished,
        so the shard still downloading is caught by name, not by its blob."""

        with tempfile.TemporaryDirectory() as root:
            snapshot = self.snapshot(
                root,
                {
                    "config.json": self.CONFIG,
                    "model.safetensors.index.json": self.shard_index("a.st", "b.st"),
                    "a.st": b"x",
                },
            )
            blobs = snapshot.parents[1] / "blobs"
            (blobs / "bblob.incomplete").write_bytes(b"x" * 5)
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.complete)
        self.assertEqual(status.missing_files, ("b.st",))
        self.assertEqual(status.partial_files, 1)

    def test_safetensors_shards_are_judged_before_a_complete_bin_fallback(self):
        """``from_pretrained`` loads the safetensors checkpoint when one is
        there, so a finished ``pytorch_model.bin`` does not make up for a
        safetensors shard that never arrived."""

        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root,
                {
                    "config.json": self.CONFIG,
                    "pytorch_model.bin": b"x" * 100,
                    "model.safetensors.index.json": self.shard_index("a.st", "b.st"),
                    "a.st": b"x",
                },
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.complete)
        self.assertEqual(status.missing_files, ("b.st",))

    def test_a_bin_only_repo_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root, {"config.json": self.CONFIG, "pytorch_model.bin": b"x" * 100}
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertTrue(status.complete)
        self.assertEqual(status.missing_files, ())

    def snapshot_with_folders(self, root: str, files: dict[str, bytes]) -> Path:
        snapshot = self.snapshot(root, {k: v for k, v in files.items() if "/" not in k})
        for name, content in files.items():
            if "/" in name:
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / name).write_bytes(content)
        return snapshot

    def test_a_repo_of_another_kind_is_unsupported_rather_than_incomplete(self):
        """A CTranslate2 export, a folder of ONNX models, or an ONNX export
        that kept its Transformers ``config.json`` is whole on disk; it just
        is not something ChatLab loads. A diffusers pipeline is not here:
        that is one of the two kinds it does load, and has tests of its own."""

        layouts = {
            "ctranslate2": {"config.json": b'{"lang_ids": []}', "model.bin": b"x"},
            "onnx bundle": {"sam2-small/model.onnx": b"x" * 10},
            "sae weights": {"resid_post/width_16k/params.npz": b"x"},
            "onnx export at root": {"config.json": self.CONFIG, "model.onnx": b"x"},
            "onnx export in a folder": {"config.json": self.CONFIG, "onnx/model.onnx": b"x"},
        }
        for kind, files in layouts.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                self.snapshot_with_folders(root, files)
                status = cache_status(self.MODEL, Path(root))

                self.assertTrue(status.present)
                self.assertTrue(status.unsupported)
                self.assertFalse(status.complete)
                self.assertEqual(status.missing_files, ())

    def test_absence_alone_is_incomplete_not_unsupported(self):
        """A tokenizer another tool fetched, or a download cut off before the
        config arrived, has no weights of any kind: that is a gap, not a repo
        of another kind."""

        layouts = {
            "tokenizer only": {"tokenizer.json": b"{}", "tokenizer_config.json": b"{}"},
            "readme only": {"README.md": b"# model"},
            "config and tokenizer": {"config.json": self.CONFIG, "tokenizer.json": b"{}"},
        }
        for kind, files in layouts.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                self.snapshot(root, files)
                status = cache_status(self.MODEL, Path(root))

                self.assertFalse(status.unsupported)
                self.assertIn(MODEL_WEIGHTS, status.missing_files)

    def test_transformers_extras_do_not_make_a_missing_checkpoint_foreign(self):
        """Llama repos ship ``original/consolidated.00.pth``; a download that
        has that and the config but not the safetensors yet is incomplete.
        So is one that has fetched a shard but not the index."""

        layouts = {
            "original weights": {"config.json": self.CONFIG, "original/consolidated.00.pth": b"x"},
            "shard before index": {
                "config.json": self.CONFIG,
                "model-00001-of-00003.safetensors": b"x",
            },
            "trainer artifacts before config": {
                "training_args.bin": b"x",
                "optimizer.pt": b"x",
                "rng_state_0.pth": b"x",
            },
        }
        for kind, files in layouts.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                self.snapshot_with_folders(root, files)
                status = cache_status(self.MODEL, Path(root))

                self.assertFalse(status.unsupported)
                self.assertIn(MODEL_WEIGHTS, status.missing_files)

    def test_a_transformers_config_without_weights_is_still_incomplete(self):
        """``architectures`` alone marks a Transformers config too."""

        with tempfile.TemporaryDirectory() as root:
            self.snapshot(
                root, {"config.json": b'{"architectures": ["LlamaForCausalLM"]}'}
            )
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.unsupported)
        self.assertEqual(status.missing_files, (MODEL_WEIGHTS,))

    def test_weights_with_an_unreadable_config_are_judged_by_the_weights(self):
        with tempfile.TemporaryDirectory() as root:
            self.snapshot(root, {"config.json": b"not json", "model.safetensors": b"x"})
            status = cache_status(self.MODEL, Path(root))

        self.assertFalse(status.unsupported)
        self.assertTrue(status.complete)

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


class CachedModelsTests(unittest.TestCase):
    def make_cache_entry(
        self, root: str, model_id: str, *, complete: bool
    ) -> Path:
        folder = Path(root) / f"models--{model_id.replace('/', '--')}"
        blobs = folder / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "cached-blob").write_bytes(b"cached")
        if complete:
            commit = "abc123"
            (folder / "refs").mkdir()
            (folder / "refs" / "main").write_text(commit)
            snapshot = folder / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").symlink_to(blobs / "cached-blob")
            (snapshot / "model.safetensors").symlink_to(blobs / "cached-blob")
        return folder

    def test_lists_complete_and_resumable_model_caches(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_cache_entry(root, "org/complete--variant", complete=True)
            self.make_cache_entry(root, "org/partial", complete=False)
            inventory = {
                model.model_id: model for model in list_cached_models(Path(root))
            }

        # The double hyphen inside the name survives the folder-name split.
        self.assertEqual(set(inventory), {"org/complete--variant", "org/partial"})
        self.assertTrue(inventory["org/complete--variant"].status.complete)
        self.assertFalse(inventory["org/partial"].status.complete)

    def test_ignores_cache_entries_chatlab_cannot_select(self):
        with tempfile.TemporaryDirectory() as root:
            bare = Path(root) / "models--gpt2" / "blobs"
            bare.mkdir(parents=True)
            (bare / "cached-blob").write_bytes(b"cached")
            (Path(root) / "datasets--org--corpus").mkdir()

            self.assertEqual(list_cached_models(Path(root)), [])

    def test_a_missing_cache_has_an_empty_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "not-created"
            self.assertEqual(list_cached_models(missing), [])


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

    def test_a_byte_level_seam_follows_the_complete_context_character(self):
        # Intermediate prefixes end in U+FFFD while the context character is
        # incomplete. Once it is complete, the exact boundary must win over
        # equally provisional prefixes from the text's first character.
        tokenizer = CutsCharactersInHalf()
        split = split_context_and_text(tokenizer, "日本語", "です")

        self.assertEqual(split.context_ids, list("日本語".encode()))
        self.assertEqual(split.text_ids, list("です".encode()))
        self.assertTrue(split.seam_verified)

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
        cases = ((EatsTheLeadingSpace(is_fast=False), " foo", "bar"),)
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


# A real byte-level vocabulary is the point of the tests below: one where the
# literal ``<|endoftext|>`` a reader might paste and the id a post-processor
# would append are the same token, which is the case the fake tokenizers can
# only assert into being. It is trained on the spot rather than read out of
# the Hugging Face cache, so the tests do not depend on what one machine has
# downloaded.
REAL = tiny_tokenizer.build()


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


class RealVocabularyTests(unittest.TestCase):
    """The all-special case with a real vocabulary behind it."""

    def tokenizer(self):
        closer = [int(REAL.eos_token_id)]
        return WrappedWithoutOffsets(REAL, closer, list(closer))

    def test_a_passage_of_nothing_but_specials_scores_the_pasted_one(self):
        # The reader pasted <|endoftext|> and nothing else, so the ids are
        # the opening id, their token, and the appended closer — all three
        # the same number. The scored token is theirs, and the closer the
        # post-processor wrote is not scored.
        tokenizer = self.tokenizer()
        eos = int(REAL.eos_token_id)
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
        eos = int(REAL.eos_token_id)
        context_ids, text_ids, *_ = split_context_and_text(
            tokenizer, "the cat sat on the ", "mat"
        )

        self.assertEqual(context_ids[0], eos)
        self.assertNotIn(eos, text_ids)
        # The token that straddles the seam carries the context's trailing
        # space with it, and is scored as part of the text, as it is
        # everywhere else.
        self.assertEqual(REAL.decode(text_ids), " mat")


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


class GenerationPrefillTokenLimitTests(unittest.TestCase):
    """Generation replay has both a model window and an application ceiling."""

    def test_a_short_model_window_is_the_hard_limit(self):
        model = Model(Config(max_position_embeddings=1024))

        self.assertEqual(generation_prefill_token_limit(model), 1024)

    def test_a_roomy_model_still_has_the_application_limit(self):
        model = Model(Config(max_position_embeddings=131072))

        self.assertEqual(
            generation_prefill_token_limit(model),
            settings.DEFAULT_PREFILL_TOKEN_LIMIT,
        )

    def test_the_saved_context_limit_is_the_application_limit(self):
        model = Model(Config(max_position_embeddings=131072))

        with settings.override(prefill_token_limit=4096):
            self.assertEqual(generation_prefill_token_limit(model), 4096)


class ModelLockTests(unittest.TestCase):
    def test_a_stream_can_release_the_model_lock_from_another_worker(self):
        """Gradio is allowed to resume a streaming generator on another thread."""

        import threading

        manager = ModelManager()
        acquired = threading.Event()

        def begin_stream():
            manager._lock.acquire()
            acquired.set()

        first_worker = threading.Thread(target=begin_stream)
        first_worker.start()
        first_worker.join(timeout=1)

        self.assertTrue(acquired.is_set())
        self.assertFalse(first_worker.is_alive())
        # This raised "cannot release un-acquired lock" when the guard was an
        # RLock owned by first_worker.
        manager._lock.release()
        self.assertFalse(manager._lock.locked())


class ScoreTextGuardTests(unittest.TestCase):
    """What ``score_text`` refuses, decided before any tensor is built."""

    def manager(self) -> ModelManager:
        manager = ModelManager()
        manager.model = object()
        manager.tokenizer = FakeTokenizer()
        self.prefilled: list[int] = []

        def fake_prefill(token_ids, *, segments, positions, score_from, **_options):
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


class DownloadProgressTests(unittest.TestCase):
    """The silent tqdm hands ``snapshot_download`` reports what a reader needs."""

    def bars(self):
        # The three bars snapshot_download builds, with the arguments it uses.
        from model_runtime import DownloadProgress

        progress = DownloadProgress()
        cls = progress.bar_class()
        files = cls(desc="Fetching 3 files", total=3)
        transfer = cls(
            desc="Downloading bytes", total=0, initial=0, unit="B", unit_scale=True
        )
        rebuild = cls(
            desc="Reconstructing (incomplete total...)", total=0, unit="B", unit_scale=True
        )
        return progress, files, transfer, rebuild

    def test_nothing_is_started_before_the_file_list_arrives(self):
        from model_runtime import DownloadProgress

        snap = DownloadProgress().snapshot()

        self.assertFalse(snap.started)
        self.assertEqual(snap.fraction, 0.0)

    def test_files_and_bytes_are_read_from_the_bars(self):
        progress, files, transfer, rebuild = self.bars()
        # Each file grows both byte totals as its size becomes known.
        for bar in (transfer, rebuild):
            bar.total = (bar.total or 0) + 1000
        # A resumed file credits its on-disk bytes to reconstruction alone.
        rebuild.update(400)
        transfer.update(250)
        rebuild.update(250)
        files.update(1)

        snap = progress.snapshot()

        self.assertTrue(snap.started)
        self.assertEqual((snap.files_done, snap.files_total), (1, 3))
        self.assertEqual((snap.bytes_done, snap.bytes_total), (650, 1000))
        self.assertAlmostEqual(snap.fraction, 0.65)

    def test_the_bars_never_draw(self):
        _, files, transfer, rebuild = self.bars()
        for bar in (files, transfer, rebuild):
            self.assertTrue(bar.disable)
        # tqdm's context-manager protocol is what hf_thread_map drives.
        with files as entered:
            entered.update(1)
        self.assertEqual(files.n, 1)

    def test_iterating_a_bar_counts_files(self):
        # huggingface_hub before 1.25 hands the file bar to tqdm's thread_map,
        # which (before tqdm 4.70) advances it by iterating ``tqdm_class(iterable)``.
        # tqdm's disabled __iter__ would yield without counting.
        from model_runtime import DownloadProgress

        progress = DownloadProgress()
        cls = progress.bar_class()
        results = list(cls((name for name in ("a", "b", "c")), desc="Fetching 3 files", total=3))

        self.assertEqual(results, ["a", "b", "c"])
        snap = progress.snapshot()
        self.assertEqual((snap.files_done, snap.files_total), (3, 3))

    def test_iterating_a_sized_iterable_learns_its_total(self):
        from model_runtime import DownloadProgress

        progress = DownloadProgress()
        bar = progress.bar_class()(["x", "y"], desc="Fetching 2 files")
        consumed = list(bar)

        self.assertEqual(consumed, ["x", "y"])
        self.assertEqual((progress.snapshot().files_done, progress.snapshot().files_total), (2, 2))

    def test_the_manager_registers_the_download_while_it_runs(self):
        from unittest import mock

        manager = ModelManager()
        seen = {}

        def fake_snapshot_download(repo_id, token, tqdm_class):
            seen["active"] = dict(manager.active_downloads)
            tqdm_class(desc="Fetching 1 files", total=1).update(1)
            return "/cache/snapshots/abc"

        with mock.patch("huggingface_hub.snapshot_download", fake_snapshot_download):
            path = manager.download(" org/model ", " tok ")

        self.assertEqual(str(path), "/cache/snapshots/abc")
        self.assertEqual(list(seen["active"]), ["org/model"])
        self.assertEqual(seen["active"]["org/model"].snapshot().files_done, 1)
        self.assertEqual(manager.active_downloads, {}, "cleared when the download ends")

    def test_a_download_notes_its_start_and_its_finish(self):
        # What a tab that did not run the download reads to learn that its
        # list of cached models is out of date; see cache_revision. Both ends
        # count: a model whose files are being written cannot be loaded, so
        # it has to leave the lists while the download runs and come back
        # when it ends.
        from unittest import mock

        manager = ModelManager()
        seen = []

        def fetch(**kwargs):
            seen.append(manager.cache_revision)
            return "/cache/snapshots/abc"

        with mock.patch("huggingface_hub.snapshot_download", fetch):
            manager.download("org/model")

        self.assertEqual(seen, [1], "noted before the first byte")
        self.assertEqual(manager.cache_revision, 2)

    def test_a_download_that_failed_still_notes_that_it_ended(self):
        # The model is loadable again, which is a change the lists have to
        # hear about however the download went.
        from unittest import mock

        manager = ModelManager()

        def failing(**kwargs):
            raise OSError("offline")

        with mock.patch("huggingface_hub.snapshot_download", failing):
            with self.assertRaises(OSError):
                manager.download("org/model")

        self.assertEqual(manager.cache_revision, 2)

    def test_a_download_that_joins_a_running_one_notes_nothing(self):
        # The entry is already there, so nothing about what can be offered
        # has changed and no tab needs to redraw.
        manager = ModelManager()
        progress, reserved = manager.reserve_download("org/model")
        self.assertTrue(reserved)
        self.assertEqual(manager.cache_revision, 1)

        self.assertEqual(manager.reserve_download("org/model"), (progress, False))

        self.assertEqual(manager.cache_revision, 1)
        manager.release_download("org/model", progress)
        self.assertEqual(manager.cache_revision, 2)
        manager.release_download("org/model", progress)
        self.assertEqual(manager.cache_revision, 2, "releasing twice is harmless")

    def test_the_registration_is_cleared_when_the_download_fails(self):
        from unittest import mock

        manager = ModelManager()

        def failing(**kwargs):
            raise OSError("offline")

        with mock.patch("huggingface_hub.snapshot_download", failing):
            with self.assertRaises(OSError):
                manager.download("org/model")

        self.assertEqual(manager.active_downloads, {})

    def test_a_reservation_is_kept_by_the_download_and_cleared_after_it(self):
        from unittest import mock

        manager = ModelManager()
        progress, reserved = manager.reserve_download(" org/model ")
        self.assertTrue(reserved)
        self.assertIs(manager.active_downloads["org/model"], progress)
        seen = {}

        def fake_snapshot_download(repo_id, token, tqdm_class):
            seen["active"] = dict(manager.active_downloads)
            return "/cache/snapshots/abc"

        with mock.patch("huggingface_hub.snapshot_download", fake_snapshot_download):
            manager.download("org/model", None, progress)

        self.assertIs(seen["active"]["org/model"], progress, "not replaced")
        self.assertEqual(manager.active_downloads, {})

    def test_a_second_reservation_points_at_the_running_download(self):
        manager = ModelManager()
        first, reserved_first = manager.reserve_download("org/model")

        second, reserved_second = manager.reserve_download("org/model")

        self.assertTrue(reserved_first)
        self.assertFalse(reserved_second)
        self.assertIs(second, first)

    def test_simultaneous_reservations_yield_one_download(self):
        import threading

        manager = ModelManager()
        gate = threading.Barrier(8)
        results = []

        def race():
            gate.wait()
            results.append(manager.reserve_download("org/model"))

        threads = [threading.Thread(target=race) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sum(reserved for _, reserved in results), 1)
        self.assertEqual(len({id(progress) for progress, _ in results}), 1)

    def test_a_download_leaves_another_download_of_the_same_model_listed(self):
        from unittest import mock

        from model_runtime import DownloadProgress

        manager = ModelManager()
        running, _ = manager.reserve_download("org/model")

        with mock.patch("huggingface_hub.snapshot_download", lambda **kw: "/cache/x"):
            manager.download("org/model", None, DownloadProgress())

        self.assertIs(manager.active_downloads["org/model"], running)

    def test_a_malformed_model_id_is_refused_without_a_reservation(self):
        manager = ModelManager()

        with self.assertRaises(ValueError):
            manager.reserve_download("not a model id")

        self.assertEqual(manager.active_downloads, {})


class LoadProgressTests(unittest.TestCase):
    """What a load reports about itself while it runs."""

    def progress(self):
        from model_runtime import LoadProgress

        return LoadProgress()

    def test_nothing_is_started_before_the_loader_builds_its_bar(self):
        snap = self.progress().snapshot()

        self.assertFalse(snap.started)
        self.assertEqual(snap.fraction, 0.0)
        self.assertFalse(snap.counts_bytes)

    def test_the_weights_the_loader_has_read_are_counted(self):
        progress = self.progress()
        bar = progress.bar_class()(desc="Loading weights", total=4)
        bar.update(3)

        snap = progress.snapshot()

        self.assertTrue(snap.started)
        self.assertEqual((snap.steps_done, snap.steps_total), (3, 4))
        self.assertAlmostEqual(snap.fraction, 0.75)

    def test_a_bar_the_loader_walks_rather_than_advances_is_counted_too(self):
        progress = self.progress()
        bar = progress.bar_class()(["a", "b", "c", "d"], desc="Loading weights")

        self.assertEqual(list(bar), ["a", "b", "c", "d"])
        self.assertEqual(progress.snapshot().steps_done, 4)

    def test_bytes_are_counted_from_where_the_load_started(self):
        # Memory already held when the load begins is not this load's own.
        progress = self.progress()
        held = [400]
        progress.measure_bytes(1000, lambda: held[0])
        progress.bar_class()(desc="Loading weights", total=2)
        held[0] = 900

        snap = progress.snapshot()

        self.assertEqual((snap.bytes_done, snap.bytes_total), (500, 1000))
        self.assertTrue(snap.counts_bytes)

    def test_bytes_are_left_alone_until_the_load_begins(self):
        # Between the baseline and the first weight the allocator holds
        # whatever the device was already holding, not this load's progress.
        progress = self.progress()
        progress.measure_bytes(1000, lambda: 1000)

        snap = progress.snapshot()

        self.assertEqual(snap.bytes_done, 0)
        self.assertFalse(snap.started)

    def test_a_load_that_never_builds_a_bar_still_counts_bytes(self):
        # transformers 4.x builds its bar only for a checkpoint of several
        # shards, so a model kept in a single weight file draws none: the
        # load is watched through the allocator alone rather than sitting in
        # the pre-start state from beginning to end.
        progress = self.progress()
        held = [0]
        progress.measure_bytes(1000, lambda: held[0])
        with progress.watch():
            held[0] = 400
            snap = progress.snapshot()

        self.assertTrue(snap.started)
        self.assertEqual((snap.bytes_done, snap.bytes_total), (400, 1000))
        self.assertAlmostEqual(snap.fraction, 0.4)
        self.assertEqual(snap.steps_total, 0, "the loader built no bar")

    def test_a_snapshot_that_could_not_be_measured_leaves_the_bytes_out(self):
        progress = self.progress()
        progress.measure_bytes(None, lambda: 900)
        progress.bar_class()(desc="Loading weights", total=2).update(1)

        snap = progress.snapshot()

        self.assertEqual((snap.bytes_done, snap.bytes_total), (0, 0))
        self.assertAlmostEqual(snap.fraction, 0.5, msg="counted in steps alone")

    def test_the_fraction_averages_the_measures_there_are(self):
        from model_runtime import LoadSnapshot

        # Metal reads the weights into host memory and copies them across
        # afterwards, so each measure covers half the load.
        reading = LoadSnapshot(bytes_done=0, bytes_total=1000, steps_done=2, steps_total=4)
        moving = LoadSnapshot(bytes_done=500, bytes_total=1000, steps_done=4, steps_total=4)
        self.assertAlmostEqual(reading.fraction, 0.25)
        self.assertAlmostEqual(moving.fraction, 0.75)
        # One pass that does both at once is still that pass.
        together = LoadSnapshot(bytes_done=300, bytes_total=1000, steps_done=3, steps_total=10)
        self.assertAlmostEqual(together.fraction, 0.3)
        # And a byte total read off the files can overshoot what a loaded
        # model holds, which is what the step count is there to finish.
        overshot = LoadSnapshot(bytes_done=1200, bytes_total=1000, steps_done=4, steps_total=4)
        self.assertEqual(overshot.fraction, 1.0)

    def test_watching_lends_the_loader_a_bar_and_takes_it_back(self):
        import importlib

        from model_runtime import LOADER_BAR_ATTRIBUTES

        progress = self.progress()
        found = []
        for module_name, attribute in LOADER_BAR_ATTRIBUTES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, attribute):
                found.append((module, attribute, getattr(module, attribute)))

        self.assertTrue(found, "transformers draws its loading bar somewhere")
        with progress.watch():
            for module, attribute, original in found:
                self.assertIsNot(getattr(module, attribute), original)
        for module, attribute, original in found:
            self.assertIs(getattr(module, attribute), original)

    def test_the_loader_gets_its_bar_back_even_when_the_load_fails(self):
        import importlib

        from model_runtime import LOADER_BAR_ATTRIBUTES

        # Whichever of the two the installed transformers actually draws
        # through: 5.x moved its bar between modules more than once.
        for module_name, attribute in LOADER_BAR_ATTRIBUTES:
            module = importlib.import_module(module_name)
            if hasattr(module, attribute):
                break
        else:
            self.fail("transformers draws its loading bar somewhere")
        original = getattr(module, attribute)

        with self.assertRaises(RuntimeError):
            with self.progress().watch():
                raise RuntimeError("out of memory")

        self.assertIs(getattr(module, attribute), original)

    def test_the_manager_hands_a_load_the_progress_it_was_given(self):
        from unittest import mock

        from model_runtime import LoadProgress

        manager = ModelManager()
        progress = LoadProgress()
        seen = []

        def fake_load(
            model_id, local_path, torch, load_progress=None, precision="full", kind="text"
        ):
            seen.append(load_progress)
            return "CPU"

        with mock.patch.object(manager, "_load_locked", fake_load):
            manager.load("org/model", Path("/snap"), progress)

        self.assertEqual(seen, [progress])


class LoadingReportTests(unittest.TestCase):
    def test_conversion_memory_error_reaches_the_manager_and_log(self):
        import logging

        manager = model_runtime.ModelManager()
        source = logging.getLogger("transformers.modeling_utils")

        def fail(*args):
            source.warning(
                "Qwen LOAD REPORT\nweight | CONVERSION |\n"
                "RuntimeError: MPS backend out of memory (MPS allocated: 8.93 GiB)\n"
            )
            raise RuntimeError("Conversion failed. Look at the above report!")

        import torch

        with (
            mock.patch("model_runtime.detect_backend", return_value="mps"),
            mock.patch.object(manager, "_unload_locked"),
            mock.patch.object(manager, "_cap_mps_memory", return_value=None),
            mock.patch.object(manager, "_check_memory", return_value=(None, None)),
            mock.patch.object(manager, "_release_device_cache") as release,
            mock.patch("model_runtime.allocated_bytes", return_value=None),
            mock.patch("model_runtime.reserved_bytes", return_value=None),
            mock.patch("model_runtime._read_text_model", side_effect=fail),
            self.assertLogs("model_runtime", level="WARNING") as logs,
            self.assertRaises(model_runtime.OutOfMemoryError) as caught,
        ):
            manager._load_locked("org/model", Path("/snap"), torch, precision="4-bit")

        self.assertIn("did not fit in memory", str(caught.exception))
        self.assertIn("MPS backend out of memory", str(caught.exception))
        self.assertTrue(any("CONVERSION" in line for line in logs.output))
        release.assert_called_once()
        self.assertFalse(manager.loaded)

    def test_conversion_cause_is_visible_without_terminal_formatting(self):
        import logging

        source = logging.getLogger("transformers.modeling_utils")
        parent = logging.getLogger("transformers")
        before = list(parent.handlers)
        with self.assertRaisesRegex(RuntimeError, "ValueError: incompatible shape") as caught:
            with model_runtime._capture_loading_report():
                source.warning(
                    "\x1b[1mTiny LOAD REPORT\x1b[0m\nCONVERSION\n"
                    "ValueError: incompatible shape\n"
                )
                raise RuntimeError("See the above report!")
        self.assertNotIn("\x1b", str(caught.exception))
        self.assertEqual(parent.handlers, before)

    def test_other_threads_and_previous_loads_do_not_supply_a_report(self):
        import logging

        source = logging.getLogger("transformers.modeling_utils")
        original = RuntimeError("See the above report!")
        with model_runtime._capture_loading_report():
            source.warning("Earlier LOAD REPORT\nValueError: old failure")
        with self.assertRaises(RuntimeError) as caught:
            with model_runtime._capture_loading_report():
                worker = threading.Thread(target=lambda: source.warning(
                    "Other LOAD REPORT\nRuntimeError: out of memory"
                ))
                worker.start()
                worker.join()
                raise original
        self.assertIs(caught.exception, original)

    def test_unrelated_errors_are_preserved(self):
        import logging

        original = RuntimeError("Tokenizer failed")
        with self.assertRaises(RuntimeError) as caught:
            with model_runtime._capture_loading_report():
                logging.getLogger("transformers.modeling_utils").warning(
                    "Tiny LOAD REPORT\nUnexpected key: visual"
                )
                raise original
        self.assertIs(caught.exception, original)


class QuantizedLoadTests(unittest.TestCase):
    """What the loader asks transformers for, per device and precision."""

    def load_with(self, precision, mps: bool):
        from model_runtime import ModelManager

        manager = ModelManager()
        calls = []

        def from_pretrained(path, **kwargs):
            calls.append(kwargs)
            model = mock.MagicMock()
            model.to.return_value = model
            return model

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
            float16="torch.float16",
            float32="torch.float32",
        )
        transformers = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=from_pretrained),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            MetalConfig=lambda **kwargs: ("metal", kwargs),
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch.object(manager, "_cap_mps_memory", return_value=None),
            mock.patch.object(manager, "_check_memory", return_value=(None, None)) as check,
            mock.patch.object(manager, "_release_device_cache"),
            mock.patch("model_runtime.allocated_bytes", return_value=None),
        ):
            device = manager._load_locked("org/model", Path("/snap"), fake_torch, precision=precision)
        return manager, device, calls, check

    def test_a_quantized_load_on_metal_goes_through_the_metal_quantizer(self):
        manager, device, calls, check = self.load_with("4-bit", mps=True)

        self.assertEqual(device, "Apple Metal (MPS), 4-bit weights")
        self.assertEqual(manager.precision, "4-bit")
        (kwargs,) = calls
        self.assertEqual(kwargs["device_map"], "mps")
        self.assertEqual(kwargs["quantization_config"], ("metal", {"bits": 4, "group_size": 64}))
        self.assertEqual(check.call_args.kwargs["bits"], 4)

    def test_full_weights_on_metal_are_loaded_as_before(self):
        manager, device, calls, check = self.load_with("full", mps=True)

        self.assertEqual(device, "Apple Metal (MPS)")
        self.assertEqual(manager.precision, "full")
        (kwargs,) = calls
        self.assertNotIn("quantization_config", kwargs)
        self.assertNotIn("device_map", kwargs)
        self.assertIsNone(check.call_args.kwargs["bits"])

    def test_a_quantized_choice_off_metal_loads_full_weights_and_says_so(self):
        with self.assertLogs("model_runtime", level="INFO") as logs:
            manager, device, calls, check = self.load_with("8-bit", mps=False)

        self.assertEqual(device, "CPU")
        self.assertEqual(manager.precision, "full")
        (kwargs,) = calls
        self.assertNotIn("quantization_config", kwargs)
        self.assertIsNone(check.call_args.kwargs["bits"])
        self.assertTrue(any("need Apple Metal" in line for line in logs.output))

    def test_a_transformers_without_the_quantizer_is_explained(self):
        from model_runtime import ModelManager

        manager = ModelManager()
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
            float16="torch.float16",
            float32="torch.float32",
        )
        # A 4.57-era transformers: no MetalConfig to import.
        transformers = types.SimpleNamespace(
            __version__="4.57.1",
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: self.fail("must not reach the loader")
            ),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch.object(manager, "_cap_mps_memory", return_value=None),
            mock.patch.object(manager, "_check_memory", return_value=(None, None)),
            mock.patch.object(manager, "_release_device_cache"),
            mock.patch("model_runtime.allocated_bytes", return_value=None),
        ):
            with self.assertRaises(RuntimeError) as caught:
                manager._load_locked("org/model", Path("/snap"), fake_torch, precision="8-bit")
        self.assertIn("transformers 5.3 or newer", str(caught.exception))
        self.assertIn("4.57.1", str(caught.exception))
        self.assertFalse(manager.loaded)

    def test_a_missing_kernels_package_is_explained(self):
        from model_runtime import ModelManager

        manager = ModelManager()

        def from_pretrained(path, **kwargs):
            raise ImportError("Metal quantization requires kernels: `pip install kernels`")

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
            float16="torch.float16",
            float32="torch.float32",
        )
        transformers = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=from_pretrained),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            MetalConfig=lambda **kwargs: kwargs,
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch.object(manager, "_cap_mps_memory", return_value=None),
            mock.patch.object(manager, "_check_memory", return_value=(None, None)),
            mock.patch.object(manager, "_release_device_cache"),
            mock.patch("model_runtime.allocated_bytes", return_value=None),
        ):
            with self.assertRaises(RuntimeError) as caught:
                manager._load_locked("org/model", Path("/snap"), fake_torch, precision="4-bit")
        self.assertIn("pip install kernels", str(caught.exception))
        self.assertFalse(manager.loaded)


class AllocatedBytesTests(unittest.TestCase):
    """How far a load has got, read from the device's own allocator."""

    def test_the_graphics_cards_are_summed(self):
        from model_runtime import allocated_bytes

        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                device_count=lambda: 2,
                memory_allocated=lambda index: (index + 1) * 1000,
            )
        )

        self.assertEqual(allocated_bytes("cuda", torch), 3000)

    def test_metal_reports_what_it_holds(self):
        from model_runtime import allocated_bytes

        torch = types.SimpleNamespace(
            mps=types.SimpleNamespace(current_allocated_memory=lambda: 4096)
        )

        self.assertEqual(allocated_bytes("mps", torch), 4096)

    def test_host_memory_keeps_no_such_figure(self):
        from model_runtime import allocated_bytes

        self.assertIsNone(allocated_bytes("cpu", types.SimpleNamespace()))

    def test_a_device_that_will_not_answer_is_left_unmeasured(self):
        from model_runtime import allocated_bytes

        def refuse():
            raise RuntimeError("no metal device")

        torch = types.SimpleNamespace(
            mps=types.SimpleNamespace(current_allocated_memory=refuse)
        )

        self.assertIsNone(allocated_bytes("mps", torch))


class MemoryGuardTests(unittest.TestCase):
    """A model is refused before any weight is read when it cannot fit."""

    GB = 1024**3

    def _sparse_snapshot(self, name: str, size: int) -> Path:
        """A snapshot whose weights are as big as they claim without the bytes.

        The guard reads sizes, never contents, so a hole stands in for the
        tens of gigabytes a realistic checkpoint would otherwise write.
        """

        folder = self._snapshot({name: 1})
        with (folder / name).open("r+b") as handle:
            handle.truncate(size)
        return folder

    def _snapshot(self, files: dict[str, int], index: dict | None = None) -> Path:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        for name, size in files.items():
            (folder / name).write_bytes(b"\0" * size)
        if index is not None:
            (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        return folder

    def test_a_single_weights_file_is_measured_by_its_size(self):
        from model_runtime import snapshot_weight_bytes

        snapshot = self._snapshot({"model.safetensors": 3000, "config.json": 10})
        self.assertEqual(snapshot_weight_bytes(snapshot), 3000)

    def test_shards_are_summed_once_each_however_often_the_index_names_them(self):
        from model_runtime import snapshot_weight_bytes

        snapshot = self._snapshot(
            {"model-00001-of-00002.safetensors": 2000, "model-00002-of-00002.safetensors": 500},
            index={"weight_map": {"a": "model-00001-of-00002.safetensors",
                                  "b": "model-00001-of-00002.safetensors",
                                  "c": "model-00002-of-00002.safetensors"}},
        )
        self.assertEqual(snapshot_weight_bytes(snapshot), 2500)

    def test_a_snapshot_without_weights_cannot_be_measured(self):
        from model_runtime import snapshot_weight_bytes

        self.assertIsNone(snapshot_weight_bytes(self._snapshot({"config.json": 10})))

    def test_an_index_naming_a_missing_shard_cannot_be_measured(self):
        from model_runtime import snapshot_weight_bytes

        snapshot = self._snapshot({}, index={"weight_map": {"a": "missing.safetensors"}})
        self.assertIsNone(snapshot_weight_bytes(snapshot))

    def test_the_estimate_follows_the_dtype_conversion(self):
        from model_runtime import estimate_loaded_bytes

        self.assertEqual(estimate_loaded_bytes(1000, "bfloat16", "float16"), 1000)
        self.assertEqual(estimate_loaded_bytes(1000, "float32", "float16"), 500)
        self.assertEqual(estimate_loaded_bytes(1000, "bfloat16", "float32"), 2000)
        # Unknown on either side: assume the file's own size.
        self.assertEqual(estimate_loaded_bytes(1000, None, "float16"), 1000)
        self.assertEqual(estimate_loaded_bytes(1000, "int4", "float16"), 1000)

    def test_the_quantized_estimate_leaves_the_embeddings_whole(self):
        from model_runtime import estimate_quantized_bytes

        # 1000 half-precision parameters, 200 of them in the embeddings.
        # 4-bit: 200 x 2 bytes + 800 x (0.5 + 4/64) bytes.
        self.assertEqual(estimate_quantized_bytes(2000, "bfloat16", 4, 200), 400 + 450)
        # 8-bit: 200 x 2 + 800 x (1 + 4/64).
        self.assertEqual(estimate_quantized_bytes(2000, "bfloat16", 8, 200), 400 + 850)
        # A float32 checkpoint is halved on the way in first.
        self.assertEqual(estimate_quantized_bytes(4000, "float32", 4, 200), 400 + 450)
        # Unknown embeddings: everything is quantized.
        self.assertEqual(estimate_quantized_bytes(2000, "bfloat16", 4, None), 562)
        # Embeddings larger than the model itself cannot be: capped.
        self.assertEqual(estimate_quantized_bytes(2000, "bfloat16", 4, 5000), 2000)

    def test_the_embedding_size_is_read_from_the_config(self):
        from model_runtime import _embedding_params

        snapshot = self._snapshot({})
        (snapshot / "config.json").write_text(
            json.dumps({"vocab_size": 100, "hidden_size": 8, "tie_word_embeddings": False})
        )
        self.assertEqual(_embedding_params(snapshot), 1600)
        (snapshot / "config.json").write_text(
            json.dumps({"vocab_size": 100, "hidden_size": 8, "tie_word_embeddings": True})
        )
        self.assertEqual(_embedding_params(snapshot), 800)
        (snapshot / "config.json").write_text(json.dumps({"vocab_size": "many"}))
        self.assertIsNone(_embedding_params(snapshot))
        self.assertIsNone(_embedding_params(None))

    def test_the_width_is_read_under_the_names_other_architectures_use(self):
        from model_runtime import _embedding_params, _embedding_params_from

        # GPT-2 spells it n_embd, MPT d_model. Transformers resolves both
        # through its config classes when the file names the architecture.
        self.assertEqual(
            _embedding_params_from({"vocab_size": 100, "n_embd": 8, "tie_word_embeddings": True}), 800
        )
        self.assertEqual(_embedding_params_from({"vocab_size": 100, "d_model": 8}), 1600)
        self.assertIsNone(_embedding_params_from({"vocab_size": 100}))

        snapshot = self._snapshot({})
        (snapshot / "config.json").write_text(
            json.dumps({"model_type": "gpt2", "vocab_size": 100, "n_embd": 8})
        )
        # GPT-2 ties its embeddings by default, which the config class knows
        # and the raw file does not say.
        self.assertEqual(_embedding_params(snapshot), 800)
        # A file with no architecture at all still reads under the aliases.
        (snapshot / "config.json").write_text(json.dumps({"vocab_size": 100, "d_model": 8}))
        self.assertEqual(_embedding_params(snapshot), 1600)

    def test_multimodal_text_embeddings_are_not_estimated_as_quantized(self):
        from model_runtime import _embedding_params, estimate_snapshot_bytes

        snapshot = self._snapshot({})
        config = {
            "model_type": "qwen3_5",
            "text_config": {"vocab_size": 100, "hidden_size": 8, "tie_word_embeddings": False},
        }
        (snapshot / "config.json").write_text(json.dumps(config))
        (snapshot / "model.safetensors").write_bytes(b"\0" * 4000)
        # 1600 embedding/head parameters stay at two bytes; only the
        # remaining 400 parameters cost 0.5625 bytes at four bits.
        self.assertEqual(_embedding_params(snapshot), 1600)
        self.assertEqual(estimate_snapshot_bytes(snapshot, "float16", bits=4), 3425)
        # The raw config fallback must agree when the library cannot read it.
        with mock.patch("transformers.AutoConfig.from_pretrained", side_effect=ValueError):
            self.assertEqual(_embedding_params(snapshot), 1600)
        config["text_config"]["tie_word_embeddings"] = True
        (snapshot / "config.json").write_text(json.dumps(config))
        self.assertEqual(_embedding_params(snapshot), 800)

    def test_a_model_larger_than_the_machine_is_refused(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        with self.assertRaises(InsufficientMemoryError) as caught:
            check_memory_for_load("org/big", 54 * self.GB, 48 * self.GB, 40 * self.GB)
        self.assertIn("this machine has 48.0 GB in total", str(caught.exception))
        self.assertIn("54.0 GB", str(caught.exception))

    def test_a_model_that_fits_the_machine_but_not_right_now_is_refused(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        with self.assertRaises(InsufficientMemoryError) as caught:
            check_memory_for_load("org/mid", 30 * self.GB, 48 * self.GB, 20 * self.GB)
        self.assertIn("ChatLab estimates 20.0 GB available", str(caught.exception))
        self.assertIn("4.0 GB of safety reserve", str(caught.exception))
        self.assertNotIn("free right now", str(caught.exception))

    def test_headroom_is_kept_beside_the_weights(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        check_memory_for_load("org/ok", 10 * self.GB, 48 * self.GB, 15 * self.GB, headroom=4 * self.GB)
        with self.assertRaises(InsufficientMemoryError):
            check_memory_for_load("org/ok", 12 * self.GB, 48 * self.GB, 15 * self.GB, headroom=4 * self.GB)

    def test_unknown_memory_figures_let_the_load_through(self):
        from model_runtime import check_memory_for_load

        check_memory_for_load("org/any", 500 * self.GB, None, None)

    def test_the_message_names_the_pool_the_figures_came_from(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        with self.assertRaises(InsufficientMemoryError) as caught:
            check_memory_for_load("org/big", 30 * self.GB, 24 * self.GB, 24 * self.GB, pool="the GPU")
        self.assertIn("and the GPU has 24.0 GB in total", str(caught.exception))

    def test_the_message_names_the_precision_the_estimate_was_made_at(self):
        # The same figure is a refusal to a reader who chose four bits and a
        # fair reading to one who did not, so the number alone leaves them
        # unable to tell whether a smaller precision would lift the refusal.
        from model_runtime import (
            InsufficientMemoryError,
            check_memory_for_load,
            weights_note,
        )

        for bits, named in ((None, "for full 16-bit weights"), (4, "for 4-bit weights")):
            with self.subTest(bits=bits):
                for total, available in ((48, 20), (24, 24)):
                    with self.assertRaises(InsufficientMemoryError) as caught:
                        check_memory_for_load(
                            "org/big",
                            30 * self.GB,
                            total * self.GB,
                            available * self.GB,
                            weights=weights_note("float16", bits),
                        )
                    self.assertIn(named, str(caught.exception))

    def test_the_precision_note_is_the_width_the_weights_will_take(self):
        from model_runtime import weights_note

        self.assertEqual(weights_note("float16"), "full 16-bit weights")
        self.assertEqual(weights_note("bfloat16"), "full 16-bit weights")
        self.assertEqual(weights_note("float32"), "full 32-bit weights")
        self.assertEqual(weights_note("float16", 8), "8-bit weights")
        self.assertEqual(weights_note("float32", 4), "4-bit weights")
        # A device not read yet, or a dtype nothing is known about, names no
        # width rather than inventing one.
        self.assertEqual(weights_note(None), "full weights")
        self.assertEqual(weights_note("mystery"), "full weights")

    @staticmethod
    def _fake_torch(*figures, failing=False):
        class Cuda:
            @staticmethod
            def device_count():
                return len(figures)

            @staticmethod
            def mem_get_info(index):
                if failing:
                    raise RuntimeError("CUDA driver missing")
                return figures[index]

        return types.SimpleNamespace(cuda=Cuda)

    def test_cuda_memory_is_summed_across_devices(self):
        from model_runtime import cuda_memory

        torch = self._fake_torch((10 * self.GB, 24 * self.GB), (20 * self.GB, 24 * self.GB))
        self.assertEqual(cuda_memory(torch), (48 * self.GB, 30 * self.GB))

    def test_cuda_memory_is_unknown_without_a_usable_device(self):
        from model_runtime import cuda_memory

        self.assertEqual(cuda_memory(self._fake_torch()), (None, None))
        torch = self._fake_torch((1, 1), failing=True)
        self.assertEqual(cuda_memory(torch), (None, None))

    def _check_with(self, snapshot, backend, host, gpu, ceiling=None, bits=None):
        import model_runtime
        from model_runtime import ModelManager

        saved = model_runtime.system_memory, model_runtime.cuda_memory
        model_runtime.system_memory = lambda: host
        model_runtime.cuda_memory = lambda torch=None: gpu
        try:
            return ModelManager._check_memory(
                "org/model", snapshot, "float16", backend, ceiling=ceiling, bits=bits
            )
        finally:
            model_runtime.system_memory, model_runtime.cuda_memory = saved

    def test_the_manager_refuses_before_reading_any_weight(self):
        from model_runtime import InsufficientMemoryError

        snapshot = self._snapshot({"model.safetensors": 4096})
        (snapshot / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
        with self.assertRaises(InsufficientMemoryError):
            self._check_with(snapshot, "cpu", host=(2048, 2048), gpu=(None, None))

    def test_the_offload_pool_is_the_cards_plus_the_host(self):
        from model_runtime import offload_pool

        gpu, host = (8 * self.GB, 6 * self.GB), (32 * self.GB, 20 * self.GB)
        self.assertEqual(offload_pool(gpu, host), (40 * self.GB, 26 * self.GB))
        # A side that reports nothing is left out, not counted as empty.
        self.assertEqual(offload_pool(gpu, (None, None)), gpu)
        self.assertEqual(offload_pool((None, None), host), host)
        self.assertEqual(offload_pool((8 * self.GB, None), (None, 20 * self.GB)), (8 * self.GB, 20 * self.GB))
        self.assertEqual(offload_pool((None, None), (None, None)), (None, None))

    def test_a_cuda_model_larger_than_the_cards_may_spill_onto_the_host(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load, offload_pool

        # A 12 GB model on an 8 GB card: device_map="auto" puts the rest on
        # the CPU, and a 32 GB host has room for it.
        total, available = offload_pool((8 * self.GB, 8 * self.GB), (32 * self.GB, 28 * self.GB))
        check_memory_for_load("org/mid", 12 * self.GB, total, available, pool="the GPU plus this machine")
        # More than the two together can hold is still refused.
        with self.assertRaises(InsufficientMemoryError) as caught:
            check_memory_for_load("org/big", 44 * self.GB, total, available, pool="the GPU plus this machine")
        self.assertIn("the GPU plus this machine has 40.0 GB in total", str(caught.exception))

    def test_a_cuda_load_is_judged_by_the_cards_and_the_host_together(self):
        from model_runtime import InsufficientMemoryError

        # A few KB of weights plus the 4 GB of headroom: more than a 2 GB host
        # can hold alone, but comfortable once 8 GB of graphics memory joins it.
        snapshot = self._snapshot({"model.safetensors": 4096})
        (snapshot / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
        host = (2 * self.GB, 1 * self.GB)
        self._check_with(snapshot, "cuda", host=host, gpu=(8 * self.GB, 8 * self.GB))
        with self.assertRaises(InsufficientMemoryError) as caught:
            self._check_with(snapshot, "cuda", host=host, gpu=(1 * self.GB, 1 * self.GB))
        self.assertIn("the GPU plus this machine has 3.0 GB in total", str(caught.exception))
        # Without a host figure the cards stand alone, and the other way round.
        self._check_with(snapshot, "cuda", host=(None, None), gpu=(8 * self.GB, 8 * self.GB))
        with self.assertRaises(InsufficientMemoryError):
            self._check_with(snapshot, "cuda", host=host, gpu=(None, None))
        # Metal shares the machine's memory, so the host figures alone rule.
        with self.assertRaises(InsufficientMemoryError):
            self._check_with(snapshot, "mps", host=host, gpu=(8 * self.GB, 8 * self.GB))

    def test_the_metal_ceiling_is_part_of_the_fit_check(self):
        # A model can fit the machine and still not fit the allocator's half
        # of it. Refusing here saves reading the whole checkpoint off disk
        # only for .to("mps") to fail.
        from model_runtime import InsufficientMemoryError

        snapshot = self._sparse_snapshot("model.safetensors", 25 * self.GB)
        idle = (48 * self.GB, 40 * self.GB)

        # Without a ceiling an idle 48 GB machine takes it.
        estimated, _ = self._check_with(snapshot, "mps", host=idle, gpu=(None, None))
        self.assertEqual(estimated, 25 * self.GB)

        with self.assertRaises(InsufficientMemoryError) as refused:
            self._check_with(
                snapshot, "mps", host=idle, gpu=(None, None), ceiling=24 * self.GB
            )
        message = str(refused.exception)
        self.assertIn("Metal on this machine", message)
        self.assertIn("24.0 GB", message)

    def test_a_refusal_reports_the_precision_the_load_would_have_used(self):
        # The bits are cleared before the check on anything but Metal, so
        # what the card and the log name is what the load would really have
        # done - which is how a reader on a graphics card learns their 4-bit
        # choice did not shrink anything.
        from model_runtime import InsufficientMemoryError

        snapshot = self._sparse_snapshot("model.safetensors", 25 * self.GB)
        with self.assertRaises(InsufficientMemoryError) as caught:
            self._check_with(
                snapshot, "cpu", host=(32 * self.GB, 20 * self.GB), gpu=(None, None)
            )
        self.assertIn("for full 16-bit weights", str(caught.exception))
        # Quantized, the same checkpoint is a fraction of that, and it is the
        # smaller figure the message has to be about.
        with self.assertLogs("model_runtime", level="WARNING") as logged:
            with self.assertRaises(InsufficientMemoryError) as caught:
                self._check_with(
                    snapshot, "mps", host=(8 * self.GB, 8 * self.GB), gpu=(None, None), bits=4
                )
        self.assertIn("for 4-bit weights", str(caught.exception))
        self.assertIn("as 4-bit weights", "\n".join(logged.output))

    def test_a_refused_load_is_recorded(self):
        # The refusal is the outcome most worth explaining afterwards, and the
        # caller turns it into a status card the log never sees.
        from model_runtime import InsufficientMemoryError

        snapshot = self._sparse_snapshot("model.safetensors", 8 * self.GB)
        with self.assertLogs("model_runtime", level="WARNING") as logged:
            with self.assertRaises(InsufficientMemoryError):
                self._check_with(
                    snapshot, "cpu", host=(32 * self.GB, 2 * self.GB), gpu=(None, None)
                )
        (line,) = [entry for entry in logged.output if "Refused" in entry]
        self.assertIn("org/model", line)
        self.assertIn("8.0 GB estimated", line)
        self.assertIn("2.0 GB estimated available", line)

    def test_the_check_reports_the_memory_it_expects_the_weights_to_take(self):
        # The figure a load counts its own progress towards, so the two can
        # never disagree about how much there is to do.
        snapshot = self._snapshot({"model.safetensors": 4096})
        (snapshot / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))

        estimated, available = self._check_with(
            snapshot, "cpu", host=(32 * self.GB, 32 * self.GB), gpu=(None, None)
        )

        self.assertEqual(estimated, 4096, "bfloat16 weights loaded as float16")
        # The second figure is what the decision was judged against, which is
        # what the load then records.
        self.assertEqual(available, 32 * self.GB)
        self.assertEqual(
            self._check_with(
                self._snapshot({"config.json": 2}),
                "cpu",
                host=(32 * self.GB, 32 * self.GB),
                gpu=(None, None),
            ),
            (None, None),
            "an unmeasurable snapshot has no figures to report",
        )

    def test_an_unmeasurable_snapshot_is_left_to_the_loader(self):
        snapshot = self._snapshot({"config.json": 2})
        self._check_with(snapshot, "cpu", host=(1, 1), gpu=(1, 1))


class DarwinAvailableMemoryTests(unittest.TestCase):
    """Reclaim file cache at normal pressure, retaining a stricter fallback."""

    # A real reading from a 48 GB Mac deep in swap, where the inactive queue
    # is smaller than the machine's anonymous total and so could be all
    # dirty anonymous pages.
    VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   147787.
Pages active:                                 961343.
Pages inactive:                              1031471.
Pages speculative:                            149733.
Pages throttled:                                   0.
Pages wired down:                             277525.
Pages purgeable:                               73270.
File-backed pages:                            925706.
Anonymous pages:                             1440383.
"""
    DISJOINT = 147787 + 149733 + 73270

    def _available(self, output, pressure="2"):
        import model_runtime

        saved = model_runtime._run_quietly
        def run(command):
            if command == ["vm_stat"]:
                return output
            if command == ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]:
                return pressure
            raise AssertionError(f"Unexpected command: {command}")

        model_runtime._run_quietly = run
        try:
            return model_runtime._darwin_available_memory()
        finally:
            model_runtime._run_quietly = saved

    def _swap(self, label, value):
        return re.sub(rf"{label}:\s*\d+\.", f"{label}: {value}.", self.VM_STAT)

    def test_free_read_ahead_and_purgeable_pages_count_in_full(self):
        # The three that do not overlap each other. Purgeable pages sit
        # inside the active and inactive queues, but purgeable memory is
        # anonymous, so they never overlap the file-page credit below.
        self.assertEqual(self._available(self.VM_STAT), self.DISJOINT * 16384)

    def test_an_inactive_queue_that_could_be_all_anonymous_credits_nothing(self):
        # Reclaiming a dirty anonymous page means writing it to swap, which
        # is the freeze this figure exists to prevent. memory_pressure counts
        # the queue whole and reported 87% of this machine free while its
        # swap was nearly full.
        counted = self._available(self.VM_STAT)
        self.assertLess(counted, (self.DISJOINT + 1031471) * 16384)
        self.assertEqual(counted, self.DISJOINT * 16384)

    def test_the_part_of_the_queue_that_must_be_file_pages_is_credited(self):
        # The queue cannot hold more anonymous pages than the machine has, so
        # the excess is file pages and reclaimable without swap.
        machine = self._swap("Anonymous pages", 400000)
        self.assertEqual(
            self._available(machine), (self.DISJOINT + 1031471 - 400000) * 16384
        )

    def test_the_file_backed_total_is_not_used_as_a_bound(self):
        # It counts active file pages too, and vm_stat does not say how many,
        # so it cannot bound the inactive queue. Changing it changes nothing.
        self.assertEqual(
            self._available(self._swap("File-backed pages", 5)),
            self._available(self.VM_STAT),
        )

    def test_normal_pressure_credits_file_cache_without_counting_speculative_twice(self):
        self.assertEqual(
            self._available(self.VM_STAT, pressure="1\n"),
            (147787 + 73270 + 925706) * 16384,
        )

    def test_a_small_model_on_a_cache_heavy_mac_passes_only_at_normal_pressure(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        # The user's 48 GB Mac: the old estimate was only 2.1 GB, despite
        # 9.7 GB of pageable file-backed memory. No model is actually loaded.
        reading = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 3693.
Pages speculative: 52662.
Pages purgeable: 79907.
Pages inactive: 1113319.
Anonymous pages: 1680799.
File-backed pages: 638406.
Pages occupied by compressor: 514962.
Swapouts: 8624257.
"""
        gb = 1024**3
        check_memory_for_load(
            "org/small", int(2.8 * gb), 48 * gb,
            self._available(reading, pressure="1"),
        )
        for pressure in ("2", "4", "", "unknown", "0", "8"):
            with self.subTest(pressure=pressure):
                with self.assertRaises(InsufficientMemoryError):
                    check_memory_for_load(
                        "org/small", int(2.8 * gb), 48 * gb,
                        self._available(reading, pressure=pressure),
                    )

    def test_normal_pressure_still_refuses_a_model_that_exceeds_available_memory(self):
        from model_runtime import InsufficientMemoryError, check_memory_for_load

        with self.assertRaises(InsufficientMemoryError):
            check_memory_for_load(
                "org/large", 20 * 1024**3, 48 * 1024**3,
                self._available(self.VM_STAT, pressure="1"),
            )

    def test_missing_cache_counters_keep_the_conservative_estimate(self):
        for label in ("File-backed pages", "Pages speculative"):
            reading = re.sub(rf"{label}:.*\n", "", self.VM_STAT)
            with self.subTest(label=label):
                self.assertEqual(
                    self._available(reading, pressure="1"), self._available(reading)
                )

    def test_file_credit_cannot_subtract_from_the_baseline(self):
        self.assertEqual(
            self._available(self._swap("File-backed pages", 5), pressure="1"),
            self.DISJOINT * 16384,
        )

    def test_normal_pressure_does_not_count_the_inactive_file_floor_twice(self):
        reading = self._swap("Anonymous pages", 400000)
        self.assertEqual(
            self._available(reading, pressure="1"),
            (147787 + 73270 + 925706) * 16384,
        )

    def test_an_exhausted_machine_answers_zero_rather_than_unknown(self):
        # None means unmeasured, and an unmeasured machine is let through. A
        # machine with nothing left must not read as one of those.
        empty = re.sub(r"(Pages|File-backed pages|Anonymous pages)(.*?):\s*\d+\.", r"\1\2: 0.", self.VM_STAT)
        self.assertEqual(self._available(empty), 0)
        self.assertEqual(self._available(empty, pressure="1"), 0)

    def test_output_without_a_page_size_says_nothing(self):
        self.assertIsNone(self._available("nothing useful here"))

    def test_output_without_any_counters_says_nothing(self):
        self.assertIsNone(self._available("Mach Virtual Memory Statistics: (page size of 16384 bytes)"))


class FirstLineTests(unittest.TestCase):
    """A backend that raises with nothing to say must not take the run down."""

    def test_the_first_line_is_quoted_and_clipped(self):
        from model_runtime import first_line

        self.assertEqual(first_line(RuntimeError("boom\ndetail")), "boom")
        self.assertEqual(len(first_line(RuntimeError("x" * 500))), 200)

    def test_a_message_less_error_names_its_type_instead(self):
        # MemoryError() is the one that turns up under memory pressure, and
        # indexing the first line of an empty message is how a memory failure
        # becomes an IndexError somewhere else entirely.
        from model_runtime import first_line, out_of_memory_message

        self.assertEqual(first_line(MemoryError()), "no message from MemoryError")
        self.assertIn("MemoryError", out_of_memory_message(MemoryError()))

    def test_a_message_less_error_still_reaches_the_caller_as_one(self):
        from model_runtime import OutOfMemoryError, _reraise_out_of_memory

        with self.assertRaises(OutOfMemoryError):
            _reraise_out_of_memory(MemoryError())


class MetalCapTests(unittest.TestCase):
    def setUp(self):
        import os

        self.saved = {
            key: os.environ.pop(key, None)
            for key in ("CHATLAB_MPS_MEMORY_FRACTION", "PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        }

    def tearDown(self):
        import os

        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_the_default_caps_at_half_the_machine(self):
        from model_runtime import mps_memory_fraction

        # A 48 GB Mac whose Metal recommendation is 37.44 GiB: half the
        # machine is 24 GB, which is 0.64 of what Metal would allow.
        total, recommended = 48 * 1000**3, int(37.44 * 1024**3)
        fraction = mps_memory_fraction(recommended, total)
        self.assertAlmostEqual(recommended * fraction / 1000**3, 24.0, places=1)

    def test_the_default_never_exceeds_what_metal_offers(self):
        from model_runtime import mps_memory_fraction

        # A machine whose Metal recommendation is already below half of it.
        self.assertEqual(mps_memory_fraction(4 * 1024**3, 64 * 1024**3), 1.0)

    def test_unknown_figures_take_half_of_the_recommendation(self):
        from model_runtime import FALLBACK_MPS_MEMORY_FRACTION, mps_memory_fraction

        self.assertEqual(mps_memory_fraction(), FALLBACK_MPS_MEMORY_FRACTION)
        self.assertEqual(mps_memory_fraction(None, 48 * 1024**3), 0.5)
        self.assertEqual(mps_memory_fraction(37 * 1024**3, None), 0.5)

    def test_the_environment_overrides_the_default(self):
        import os

        from model_runtime import mps_memory_fraction

        os.environ["CHATLAB_MPS_MEMORY_FRACTION"] = "0.8"
        self.assertEqual(mps_memory_fraction(), 0.8)

    def test_the_settings_file_overrides_the_default(self):
        from model_runtime import mps_memory_fraction

        with settings.override(mps_memory_fraction=0.6):
            self.assertEqual(mps_memory_fraction(), 0.6)

    def test_the_environment_overrides_the_settings_file(self):
        import os

        from model_runtime import mps_memory_fraction

        os.environ["CHATLAB_MPS_MEMORY_FRACTION"] = "0.9"
        with settings.override(mps_memory_fraction=0.6):
            self.assertEqual(mps_memory_fraction(), 0.9)

    def test_a_value_pytorch_would_reject_leaves_the_allocator_alone(self):
        import os

        from model_runtime import mps_memory_fraction

        for raw in ("0", "-1", "2.5"):
            os.environ["CHATLAB_MPS_MEMORY_FRACTION"] = raw
            self.assertIsNone(mps_memory_fraction(), raw)
        os.environ["CHATLAB_MPS_MEMORY_FRACTION"] = "lots"
        self.assertEqual(mps_memory_fraction(), 0.5)

    def test_a_user_set_pytorch_watermark_stands(self):
        import os

        from model_runtime import mps_memory_fraction

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        self.assertIsNone(mps_memory_fraction())

    def test_the_cap_is_applied_through_torch(self):
        from types import SimpleNamespace

        from model_runtime import ModelManager

        import model_runtime

        applied = []
        recommended = 32 * 1024**3
        fake_torch = SimpleNamespace(
            mps=SimpleNamespace(
                set_per_process_memory_fraction=applied.append,
                recommended_max_memory=lambda: recommended,
            )
        )
        ceiling = ModelManager._cap_mps_memory(fake_torch)
        self.assertEqual(len(applied), 1)
        self.assertEqual(ceiling, int(recommended * applied[0]))
        # Never more than half the machine, whatever Metal recommends.
        total = model_runtime.system_memory()[0]
        if total:
            self.assertLessEqual(ceiling, total // 2 + 1)

    def test_a_torch_without_the_setter_is_tolerated(self):
        from types import SimpleNamespace

        from model_runtime import ModelManager

        ModelManager._cap_mps_memory(SimpleNamespace(mps=SimpleNamespace()))


class OutOfMemoryTests(unittest.TestCase):
    """A backend's out-of-memory failure reaches the caller as one readable error."""

    def _manager(self):
        from model_runtime import ModelManager

        manager = ModelManager()
        manager.model = object()
        manager.tokenizer = FakeTokenizer()
        manager.released = 0
        manager._release_device_cache = lambda torch=None: setattr(
            manager, "released", manager.released + 1
        )
        return manager

    def test_backend_messages_are_recognised(self):
        from model_runtime import OutOfMemoryError, is_out_of_memory_error

        self.assertTrue(is_out_of_memory_error(RuntimeError(
            "MPS backend out of memory (MPS allocated: 36.00 GB, other allocations: 1 KB, max allowed: 36.00 GB)."
        )))
        self.assertTrue(is_out_of_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertTrue(is_out_of_memory_error(MemoryError()))
        self.assertTrue(is_out_of_memory_error(OutOfMemoryError("x")))
        self.assertFalse(is_out_of_memory_error(RuntimeError("shape mismatch")))

    def test_a_generation_that_runs_out_of_memory_raises_one_error_and_frees_the_slot(self):
        from model_runtime import OutOfMemoryError

        manager = self._manager()

        def failing(*_args, **_kwargs):
            yield "first frame"
            raise RuntimeError("MPS backend out of memory (MPS allocated: 36 GB). Tried to allocate 1 GB")

        manager._generate = failing
        stream = manager.generate([], temperature=1, top_p=1, top_k=0, max_new_tokens=1, seed=0)
        self.assertEqual(next(stream), "first frame")
        with self.assertRaises(OutOfMemoryError) as caught:
            next(stream)
        self.assertIn("ran out of memory", str(caught.exception))
        self.assertIn("Shorten the conversation", str(caught.exception))
        self.assertFalse(manager._generating.locked())
        self.assertEqual(manager.released, 1)

    def test_other_generation_failures_pass_through_unchanged(self):
        manager = self._manager()

        def failing(*_args, **_kwargs):
            yield from ()
            raise RuntimeError("shape mismatch")

        manager._generate = failing
        with self.assertRaises(RuntimeError) as caught:
            list(manager.generate([], temperature=1, top_p=1, top_k=0, max_new_tokens=1, seed=0))
        self.assertEqual(str(caught.exception), "shape mismatch")
        self.assertEqual(manager.released, 1)

    def test_a_finished_generation_hands_its_cache_back(self):
        manager = self._manager()
        # The real stream is a generator, explicitly closed on every exit.
        manager._generate = lambda *a, **k: (frame for frame in ["only frame"])
        self.assertEqual(
            list(manager.generate([], temperature=1, top_p=1, top_k=0, max_new_tokens=1, seed=0)),
            ["only frame"],
        )
        self.assertEqual(manager.released, 1)

    def test_scoring_and_inspection_translate_and_release_too(self):
        from model_runtime import OutOfMemoryError

        manager = self._manager()

        def exploding_prefill(*_args, **_kwargs):
            raise RuntimeError("MPS backend out of memory")

        manager._prefill = exploding_prefill
        with self.assertRaises(OutOfMemoryError):
            manager.score_text("some text to score")
        self.assertEqual(manager.released, 1)
        manager.model = None
        with self.assertRaises(RuntimeError) as caught:
            manager.inspect([1, 2, 3], 2)
        self.assertNotIsInstance(caught.exception, OutOfMemoryError)
        self.assertEqual(manager.released, 2)


class CountScoreTokensTests(unittest.TestCase):
    """The live count under the Score text box.

    Scoring refuses a passage above the model's limit, and it does so after
    the paste, the press and the wait. This is the same arithmetic done while
    the passage is still being written, so the number on screen is the number
    that will be judged - which means it has to come from the encoding the
    check itself uses, not an estimate beside it.
    """

    def manager(self, **config) -> ModelManager:
        manager = ModelManager()
        manager.tokenizer = FakeTokenizer()
        manager.model = Model(Config(**config)) if config else Model(Config())
        return manager

    def test_a_passage_is_counted_against_the_flat_limit(self):
        manager = self.manager()

        count, limit = manager.count_score_tokens("one two three")

        self.assertEqual(limit, SCORE_TOKEN_LIMIT)
        # The opening special, three words, and the two spaces between them:
        # the passage's own encoding, which is what scoring will measure.
        self.assertEqual(count, 6)

    def test_the_context_is_counted_with_the_text_it_precedes(self):
        # Both halves go into the same pass, so both spend the same budget.
        manager = self.manager()

        with_context, _ = manager.count_score_tokens("two", context="one ")
        alone, _ = manager.count_score_tokens("two")

        self.assertGreater(with_context, alone)

    def test_a_short_context_window_lowers_the_limit_it_reports(self):
        manager = self.manager(max_position_embeddings=512)

        _count, limit = manager.count_score_tokens("one")

        self.assertEqual(limit, 512)

    def test_an_empty_box_counts_nothing_but_still_names_the_limit(self):
        manager = self.manager()

        self.assertEqual(manager.count_score_tokens(""), (0, SCORE_TOKEN_LIMIT))

    def test_nothing_loaded_has_no_answer_rather_than_a_wrong_one(self):
        self.assertIsNone(ModelManager().count_score_tokens("one two"))

    def test_the_encoding_does_not_run_under_the_model_lock(self):
        # The lock is held only to read the tokenizer, the limit and the load
        # they belong to. Encoding under it was the real hazard: a generation
        # claims its slot before it goes for the lock, so a keystroke landing
        # in that window could keep a reply waiting for as long as tokenizing
        # a large paste took.
        manager = self.manager()
        held = []

        class WatchesTheLock(FakeTokenizer):
            def __call__(self, text, **kwargs):
                held.append(manager._lock.locked())
                return super().__call__(text, **kwargs)

        manager.tokenizer = WatchesTheLock()

        self.assertIsNotNone(manager.count_score_tokens("one two"))

        self.assertTrue(held, "the tokenizer was never asked")
        self.assertFalse(any(held), "the lock was held while encoding")
        self.assertFalse(manager._lock.locked())

    def test_a_load_landing_mid_count_throws_the_count_away(self):
        # Encoding outside the lock means the weights can change under it,
        # and a number from the wrong tokenizer is worse than no number.
        manager = self.manager()

        class LoadsUnderneath(FakeTokenizer):
            def __call__(self, text, **kwargs):
                manager.load_count += 1
                return super().__call__(text, **kwargs)

        manager.tokenizer = LoadsUnderneath()

        self.assertIsNone(manager.count_score_tokens("one two"))

    def test_a_reserved_generation_is_not_talked_over(self):
        # A generation claims the slot before it goes for the model lock, so
        # in between the lock is free. Taking it then would tokenize however
        # much text has been pasted while an operation the interface already
        # reports as running waits behind a keystroke.
        manager = self.manager()
        self.assertTrue(manager.reserve_generation())
        try:
            self.assertFalse(manager._lock.locked(), "the lock is free here")
            self.assertIsNone(manager.count_score_tokens("one two"))
        finally:
            manager.release_generation()
        self.assertIsNotNone(manager.count_score_tokens("one two"))

    def test_a_busy_model_is_not_waited_on(self):
        # The count answers a box being typed into. Blocking on the model lock
        # would hang the keystroke behind a whole generation, so a held lock
        # means no answer this time and another chance on the next character.
        manager = self.manager()
        manager._lock.acquire()
        try:
            self.assertIsNone(manager.count_score_tokens("one two"))
        finally:
            manager._lock.release()
        # And the lock is left as it was found, so the next count works.
        self.assertIsNotNone(manager.count_score_tokens("one two"))

    def test_text_that_tokenizes_to_nothing_has_no_answer_either(self):
        # score_text refuses a passage that produces no tokens, so reporting
        # it as a confident zero would read as room to spare. Pressing Score
        # text explains the refusal properly; a half-typed passage is not yet
        # worth complaining about.
        class RefusesEverything(FakeTokenizer):
            def __call__(self, text, **kwargs):
                raise ValueError("this tokenizer will not encode that")

        manager = self.manager()
        manager.tokenizer = RefusesEverything()

        self.assertIsNone(manager.count_score_tokens("one two"))
        self.assertFalse(manager._lock.locked())


class DeviceProfileTests(unittest.TestCase):
    """Reading the device a load would use, before anything has loaded."""

    GB = 1024**3

    def test_the_backend_is_the_one_a_load_would_pick(self):
        from model_runtime import detect_backend

        both = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True)
            ),
        )
        self.assertEqual(detect_backend(both), "cuda")
        metal = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True)
            ),
        )
        self.assertEqual(detect_backend(metal), "mps")
        plain = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        )
        self.assertEqual(detect_backend(plain), "cpu")

    def test_a_torch_that_will_not_answer_is_the_cpu(self):
        from model_runtime import detect_backend

        def raises():
            raise RuntimeError("no driver")

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=raises))
        self.assertEqual(detect_backend(torch), "cpu")

    def test_each_backend_loads_the_dtype_it_can_use(self):
        from model_runtime import dtype_name, load_dtype

        torch = types.SimpleNamespace(
            bfloat16="torch.bfloat16",
            float16="torch.float16",
            float32="torch.float32",
            cuda=types.SimpleNamespace(is_bf16_supported=lambda: True),
        )
        self.assertEqual(dtype_name(load_dtype("cuda", torch)), "bfloat16")
        self.assertEqual(dtype_name(load_dtype("mps", torch)), "float16")
        self.assertEqual(dtype_name(load_dtype("cpu", torch)), "float32")
        torch.cuda = types.SimpleNamespace(is_bf16_supported=lambda: False)
        self.assertEqual(dtype_name(load_dtype("cuda", torch)), "float16")

    def _pool_with(self, backend, host, gpu, ceiling=None):
        import model_runtime

        saved = model_runtime.system_memory, model_runtime.cuda_memory
        model_runtime.system_memory = lambda: host
        model_runtime.cuda_memory = lambda torch=None: gpu
        try:
            return model_runtime.memory_pool(backend, ceiling)
        finally:
            model_runtime.system_memory, model_runtime.cuda_memory = saved

    def test_the_pool_is_the_one_the_load_check_judges_against(self):
        host, gpu = (48 * self.GB, 40 * self.GB), (24 * self.GB, 20 * self.GB)

        self.assertEqual(
            self._pool_with("cpu", host, gpu), (48 * self.GB, 40 * self.GB, "this machine")
        )
        self.assertEqual(
            self._pool_with("mps", host, gpu), (48 * self.GB, 40 * self.GB, "this machine")
        )
        self.assertEqual(
            self._pool_with("cuda", host, gpu),
            (72 * self.GB, 60 * self.GB, "the GPU plus this machine"),
        )

    def test_the_metal_ceiling_caps_the_pool_and_names_it(self):
        total, available, pool = self._pool_with(
            "mps", (48 * self.GB, 40 * self.GB), (None, None), ceiling=24 * self.GB
        )
        self.assertEqual((total, available), (24 * self.GB, 24 * self.GB))
        self.assertEqual(pool, "Metal on this machine")

    def test_a_machine_that_says_nothing_still_gives_the_ceiling(self):
        total, available, _ = self._pool_with(
            "mps", (None, None), (None, None), ceiling=24 * self.GB
        )
        self.assertEqual((total, available), (24 * self.GB, 24 * self.GB))

    def test_the_profile_answers_from_memory_alone_before_torch_is_imported(self):
        # The Models page is painted before anything has needed torch, and
        # the import takes seconds. Until it lands the device is unknown
        # rather than guessed at, and the figures are the machine's own.
        import model_runtime

        saved = model_runtime.system_memory
        model_runtime.system_memory = lambda: (48 * self.GB, 40 * self.GB)
        try:
            with mock.patch.object(model_runtime, "_torch_ready", threading.Event()):
                profile = model_runtime.device_profile()
        finally:
            model_runtime.system_memory = saved

        self.assertIsNone(profile.backend)
        self.assertIsNone(profile.dtype)
        self.assertFalse(profile.quantizes)
        self.assertEqual(profile.total, 48 * self.GB)
        self.assertEqual(profile.available, 40 * self.GB)
        self.assertIsNone(profile.ceiling)

    def test_the_profile_reads_the_metal_ceiling_when_torch_is_there(self):
        import model_runtime

        torch = types.SimpleNamespace(
            float16="torch.float16",
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True)
            ),
            mps=types.SimpleNamespace(recommended_max_memory=lambda: 36 * self.GB),
        )
        saved = model_runtime.system_memory
        model_runtime.system_memory = lambda: (48 * self.GB, 40 * self.GB)
        try:
            profile = model_runtime.device_profile(torch)
        finally:
            model_runtime.system_memory = saved

        self.assertEqual(profile.backend, "mps")
        self.assertEqual(profile.dtype, "float16")
        self.assertTrue(profile.quantizes)
        # Half the machine, expressed against Metal's recommendation.
        self.assertEqual(profile.ceiling, 24 * self.GB)
        self.assertEqual(profile.total, 24 * self.GB)
        self.assertEqual(profile.pool, "Metal on this machine")

    def test_the_ceiling_is_the_one_the_cap_would_set(self):
        # The fit verdict and the load's own check have to agree, so both
        # read the ceiling from one formula.
        import model_runtime

        applied = []
        torch = types.SimpleNamespace(
            mps=types.SimpleNamespace(
                recommended_max_memory=lambda: 36 * self.GB,
                set_per_process_memory_fraction=applied.append,
            )
        )
        saved = model_runtime.system_memory
        model_runtime.system_memory = lambda: (48 * self.GB, 40 * self.GB)
        try:
            self.assertEqual(
                ModelManager._cap_mps_memory(torch), model_runtime.mps_ceiling(torch)
            )
        finally:
            model_runtime.system_memory = saved
        self.assertEqual(applied, [24 / 36])

    def test_the_loaded_models_memory_is_given_back_for_a_replacement(self):
        from model_runtime import DeviceProfile

        profile = DeviceProfile(
            backend="mps",
            available=2 * self.GB,
            total=24 * self.GB,
            held=15 * self.GB,
        )

        self.assertEqual(profile.reclaimed().available, 17 * self.GB)
        # The total is the machine's own either way.
        self.assertEqual(profile.reclaimed().total, 24 * self.GB)
        # Nothing to give back, or no figure for it: unchanged.
        self.assertEqual(
            DeviceProfile(available=2 * self.GB).reclaimed().available, 2 * self.GB
        )
        self.assertIsNone(DeviceProfile(held=self.GB).reclaimed().available)

    def test_whichever_figure_is_larger_is_what_a_load_gives_back(self):
        # Neither is enough alone: a CUDA model spread over the cards and the
        # machine is only counted on the cards by the allocator, while the
        # load's estimate covers the whole of it; on Metal the allocator can
        # be the larger, a response's key-value cache being live tensors too.
        from model_runtime import DeviceProfile

        profile = DeviceProfile(available=self.GB, held=4 * self.GB)

        self.assertEqual(profile.reclaimed(10 * self.GB).available, 11 * self.GB)
        self.assertEqual(profile.reclaimed(2 * self.GB).available, 5 * self.GB)
        self.assertEqual(profile.reclaimed(None).available, 5 * self.GB)

    def test_the_profile_reads_what_the_device_is_holding(self):
        import model_runtime

        torch = types.SimpleNamespace(
            float32="torch.float32",
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        )
        saved = model_runtime.system_memory
        model_runtime.system_memory = lambda: (16 * self.GB, 8 * self.GB)
        try:
            # Host memory keeps no such figure, so there is nothing to give back.
            self.assertIsNone(model_runtime.device_profile(torch).held)
        finally:
            model_runtime.system_memory = saved

    def test_a_torch_that_is_still_importing_is_not_read(self):
        # Python puts a module in sys.modules before its body has run, so a
        # reader that went by presence alone could find torch without
        # torch.backends and either raise - at startup, where the hardware
        # panel is built - or quietly report the wrong device.
        import model_runtime

        half_built = types.ModuleType("torch")  # no cuda, no backends, no dtypes

        with mock.patch.dict(sys.modules, {"torch": half_built}):
            with mock.patch.object(model_runtime, "_torch_ready", threading.Event()):
                self.assertIsNone(model_runtime.imported_torch())
                self.assertIsNone(model_runtime.device_profile().backend)
            ready = threading.Event()
            ready.set()
            with mock.patch.object(model_runtime, "_torch_ready", ready):
                self.assertIs(model_runtime.imported_torch(), half_built)

    def test_the_import_thread_is_what_says_torch_may_be_read(self):
        import model_runtime

        with mock.patch.object(model_runtime, "_torch_ready", threading.Event()) as flag:
            model_runtime.warm_device()
            self.assertTrue(flag.wait(timeout=60))
        self.assertIsNotNone(model_runtime.imported_torch())

    def test_the_device_names_itself_the_way_a_loaded_model_does(self):
        from model_runtime import device_label

        self.assertEqual(device_label("mps"), "Apple Metal (MPS)")
        self.assertEqual(device_label("cpu"), "CPU")
        self.assertEqual(device_label(None), "not determined yet")
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(get_device_name=lambda index: "RTX 4090")
        )
        self.assertEqual(device_label("cuda", torch), "CUDA (RTX 4090)")
        self.assertEqual(device_label("cuda", types.SimpleNamespace()), "CUDA")


class FitTests(unittest.TestCase):
    """Saying beforehand what the load check would answer."""

    GB = 1024**3

    def fit(self, estimated, total, available):
        from model_runtime import fit_for

        return fit_for(estimated, total, available, headroom=4 * self.GB)

    def test_a_model_with_room_beside_it_fits(self):
        from model_runtime import FITS

        fit = self.fit(10 * self.GB, 48 * self.GB, 40 * self.GB)
        self.assertEqual(fit.state, FITS)
        self.assertTrue(fit.known)
        self.assertIn("10.0 GB", fit.note)

    def test_a_model_larger_than_the_machine_will_not_fit(self):
        from model_runtime import UNFIT

        fit = self.fit(60 * self.GB, 48 * self.GB, 40 * self.GB)
        self.assertEqual(fit.state, UNFIT)
        self.assertIn("48.0 GB", fit.note)
        self.assertIn("this machine", fit.note)

    def test_the_headroom_is_part_of_the_verdict(self):
        from model_runtime import FITS, UNFIT

        # 45 GB of weights fits a 48 GB machine only without the reserve.
        self.assertEqual(self.fit(45 * self.GB, 48 * self.GB, 48 * self.GB).state, UNFIT)
        self.assertEqual(self.fit(44 * self.GB, 48 * self.GB, 48 * self.GB).state, FITS)

    def test_a_model_the_machine_could_hold_but_has_no_room_for_now_is_tight(self):
        from model_runtime import TIGHT

        fit = self.fit(30 * self.GB, 48 * self.GB, 20 * self.GB)
        self.assertEqual(fit.state, TIGHT)
        self.assertTrue(fit.known)
        self.assertIn("20.0 GB", fit.note)
        self.assertIn("memory pressure", fit.note)

    def test_a_size_that_could_not_be_measured_is_not_guessed_at(self):
        from model_runtime import FIT_UNKNOWN

        fit = self.fit(None, 48 * self.GB, 40 * self.GB)
        self.assertEqual(fit.state, FIT_UNKNOWN)
        self.assertFalse(fit.known)
        self.assertIn("downloaded", fit.note)

    def test_a_machine_that_reports_nothing_gets_no_verdict(self):
        from model_runtime import FIT_UNKNOWN

        fit = self.fit(10 * self.GB, None, None)
        self.assertEqual(fit.state, FIT_UNKNOWN)
        self.assertIn("does not report its memory", fit.note)

    def test_the_verdict_names_the_precision_it_was_measured_at(self):
        # The panel and the refusal have to agree about the precision as well
        # as the figure: a note that left it out would look as though the
        # estimate had changed by itself when the radio moved.
        from model_runtime import fit_for, weights_note

        for bits, named in ((None, "of full 16-bit weights"), (4, "of 4-bit weights")):
            for estimated in (10, 30, 60):
                with self.subTest(bits=bits, gigabytes=estimated):
                    fit = fit_for(
                        estimated * self.GB,
                        48 * self.GB,
                        20 * self.GB,
                        weights=weights_note("float16", bits),
                    )
                    self.assertIn(named, fit.note)

    def test_model_fit_takes_the_precision_from_the_device_and_the_bits(self):
        from model_runtime import DeviceProfile, model_fit

        profile = DeviceProfile(
            backend="mps", dtype="float16", total=48 * self.GB, available=20 * self.GB
        )
        self.assertIn("of full 16-bit weights", model_fit(30 * self.GB, profile).note)
        self.assertIn("of 4-bit weights", model_fit(30 * self.GB, profile, 4).note)
        # Before torch is imported there is no dtype to name.
        unread = DeviceProfile(total=48 * self.GB, available=20 * self.GB)
        self.assertIn("of full weights", model_fit(30 * self.GB, unread).note)

    def test_a_verdict_agrees_with_the_refusal_it_predicts(self):
        from model_runtime import (
            FITS,
            InsufficientMemoryError,
            check_memory_for_load,
            fit_for,
        )

        for estimated in (10, 30, 60):
            with self.subTest(gigabytes=estimated):
                fit = fit_for(estimated * self.GB, 48 * self.GB, 20 * self.GB)
                try:
                    check_memory_for_load(
                        "org/model", estimated * self.GB, 48 * self.GB, 20 * self.GB
                    )
                except InsufficientMemoryError:
                    self.assertNotEqual(fit.state, FITS)
                else:
                    self.assertEqual(fit.state, FITS)


class EstimateTests(unittest.TestCase):
    """The two ways a model's loaded size is estimated before it loads."""

    def snapshot(self, weights: int, dtype: str | None = "bfloat16") -> Path:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        (folder / "model.safetensors").write_bytes(b"\0" * weights)
        config = {"architectures": ["OlmoForCausalLM"]}
        if dtype is not None:
            config["torch_dtype"] = dtype
        (folder / "config.json").write_text(json.dumps(config))
        return folder

    def test_a_snapshot_is_measured_by_the_weights_it_holds(self):
        from model_runtime import estimate_snapshot_bytes

        snapshot = self.snapshot(2000)
        self.assertEqual(estimate_snapshot_bytes(snapshot, "float16"), 2000)
        # A load onto the CPU converts half precision up on the way in.
        self.assertEqual(estimate_snapshot_bytes(snapshot, "float32"), 4000)

    def test_a_quantized_estimate_is_of_what_the_device_would_hold(self):
        from model_runtime import estimate_snapshot_bytes

        snapshot = self.snapshot(2000)
        whole = estimate_snapshot_bytes(snapshot, "float16")
        quantized = estimate_snapshot_bytes(snapshot, "float16", bits=4)
        self.assertLess(quantized, whole // 2)

    def test_a_snapshot_that_cannot_be_measured_says_so(self):
        from model_runtime import estimate_snapshot_bytes

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        (folder / "config.json").write_text("{}")
        self.assertIsNone(estimate_snapshot_bytes(folder, "float16"))

    def test_a_parameter_count_stands_in_for_a_model_not_yet_on_disk(self):
        from model_runtime import estimate_parameter_bytes

        # Half precision: two bytes a parameter.
        self.assertEqual(estimate_parameter_bytes(1_000_000, "float16"), 2_000_000)
        self.assertEqual(estimate_parameter_bytes(1_000_000, "float32"), 4_000_000)
        # Quantized, with no way to know the embeddings' share: four bits a
        # parameter plus the scale and bias each group of 64 shares.
        self.assertEqual(estimate_parameter_bytes(1_000_000, "float16", bits=4), 562_500)
def hub_result(model_id, pipeline_tag, **extra):
    """One entry the way ``list_models`` hands it over, config and tags and all.

    The config defaults to an architecture ``AutoModelForCausalLM`` loads, so
    a test that is about something else does not have to say so.
    """

    fields = {
        "id": model_id,
        "pipeline_tag": pipeline_tag,
        "config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        "downloads": 1000,
        "likes": 1,
        "library_name": "transformers",
        "last_modified": None,
        "safetensors": None,
        "gated": False,
        "tags": [],
    }
    fields.update(extra)
    return types.SimpleNamespace(**fields)


class HubSearchTests(unittest.TestCase):
    """Which of the hub's answers reach the Model search list."""

    def setUp(self):
        self.calls = []
        self.found = []

        def list_models(**kwargs):
            self.calls.append(kwargs)
            return list(self.found)

        api = mock.Mock()
        api.list_models.side_effect = list_models
        patched = mock.patch("huggingface_hub.HfApi", return_value=api)
        patched.start()
        self.addCleanup(patched.stop)

    def test_a_multimodal_model_is_listed_beside_a_plain_one(self):
        # google/gemma-4-E4B-it writes text like any other model here, and
        # Transformers loads it through AutoModelForCausalLM, but the hub
        # tags it for the pictures and sound it also reads. Keeping only
        # "text-generation" hid it and every model like it.
        self.found = [
            hub_result(
                "google/gemma-4-E4B-it",
                "any-to-any",
                config={
                    "model_type": "gemma4",
                    "architectures": ["Gemma4ForConditionalGeneration"],
                },
            ),
            hub_result("allenai/Olmo-3-7B-Think", "text-generation"),
        ]

        found = search_hub_models("gemma")

        self.assertEqual(
            [result.model_id for result in found],
            ["google/gemma-4-E4B-it", "allenai/Olmo-3-7B-Think"],
        )

    def test_an_image_search_asks_the_hub_for_diffusers_pipelines(self):
        # A different library and a different tag: a Transformers query and a
        # diffusers one are different searches, not one with a wider net.
        self.found = [
            hub_result("stable-diffusion-v1-5/stable-diffusion-v1-5", "text-to-image"),
            hub_result("allenai/Olmo-3-7B-Think", "text-generation"),
        ]

        found = search_hub_models("diffusion", kind=IMAGE_KIND)

        self.assertEqual(self.calls[-1]["filter"], "diffusers")
        self.assertEqual(
            [result.model_id for result in found],
            ["stable-diffusion-v1-5/stable-diffusion-v1-5"],
        )

    def test_an_image_search_makes_no_causal_lm_check(self):
        """A pipeline has no model_type in that map and is built from its
        model_index.json, so the library and the tag are what say it loads.
        Asking the map would reject every pipeline there is."""

        self.found = [hub_result("org/pipe", "text-to-image", config=None)]

        with mock.patch.object(model_runtime, "causal_lm_model_types") as auto_map:
            found = search_hub_models("pipe", kind=IMAGE_KIND)

        auto_map.assert_not_called()
        self.assertEqual([result.model_id for result in found], ["org/pipe"])

    def test_an_image_search_still_leaves_out_another_runtime(self):
        # The runtime check asks about the weight format rather than about
        # Transformers, so it reads a converted diffusion model the same way.
        self.found = [
            hub_result("mlx-community/sdxl", "text-to-image", tags=["mlx"]),
            hub_result("org/onnx-sd", "text-to-image", tags=["onnx"]),
            hub_result("org/real-sd", "text-to-image"),
        ]

        found = search_hub_models("sd", kind=IMAGE_KIND)

        self.assertEqual([result.model_id for result in found], ["org/real-sd"])

    def test_a_text_search_is_what_a_caller_gets_by_default(self):
        self.found = [hub_result("allenai/Olmo-3-7B-Think", "text-generation")]

        search_hub_models("olmo")

        self.assertEqual(self.calls[-1]["filter"], "transformers")

    def test_a_multimodal_model_needing_its_own_auto_class_is_left_out(self):
        # The pipeline tag says what a model does, not which auto class loads
        # it: BLIP-2 and LLaVA sit under "image-text-to-text" beside Gemma 4
        # but need Blip2ForConditionalGeneration and
        # LlavaForConditionalGeneration. Listing them would download the whole
        # checkpoint for a load that ends in an unsupported-config error.
        self.found = [
            hub_result(
                "Salesforce/blip2-opt-2.7b",
                "image-text-to-text",
                config={
                    "model_type": "blip-2",
                    "architectures": ["Blip2ForConditionalGeneration"],
                },
            ),
            hub_result(
                "llava-hf/llava-1.5-7b-hf",
                "image-text-to-text",
                config={
                    "model_type": "llava",
                    "architectures": ["LlavaForConditionalGeneration"],
                },
            ),
            hub_result(
                "google/gemma-4-E4B-it",
                "any-to-any",
                config={
                    "model_type": "gemma4",
                    "architectures": ["Gemma4ForConditionalGeneration"],
                },
            ),
        ]

        found = search_hub_models("multimodal")

        self.assertEqual([result.model_id for result in found], ["google/gemma-4-E4B-it"])

    def test_a_config_without_a_model_type_is_left_out(self):
        # AutoConfig resolves by model_type, not by the architectures a config
        # lists beside it, so a mapped class name there is not a promise the
        # load path can keep: it would place the repository nowhere and fail
        # after the whole snapshot had come down.
        self.found = [
            hub_result(
                "org/no-model-type",
                "text-generation",
                config={"architectures": ["LlamaForCausalLM"]},
            ),
            hub_result("org/no-config-at-all", "text-generation", config=None),
        ]

        self.assertEqual(search_hub_models("org"), [])

    def test_the_hub_is_not_asked_to_do_the_filtering(self):
        # The tags are checked here, so asking the hub for one pipeline tag
        # would throw the rest away before they could be. No limit is sent
        # either: the results are paged through until the list is full.
        search_hub_models("gemma", limit=5)

        (call,) = self.calls
        self.assertNotIn("pipeline_tag", call)
        self.assertNotIn("limit", call)
        self.assertEqual(call["filter"], "transformers")

    def test_a_model_that_writes_no_text_is_left_out(self):
        self.found = [
            hub_result("sentence-transformers/all-MiniLM-L6-v2", "feature-extraction"),
            hub_result("openai/whisper-large-v3", "automatic-speech-recognition"),
            hub_result("some-body/untagged-weights", None),
            hub_result("allenai/Olmo-3-7B-Think", "text-generation"),
        ]

        found = search_hub_models("olmo")

        self.assertEqual([result.model_id for result in found], ["allenai/Olmo-3-7B-Think"])

    def test_a_conversion_to_another_runtime_is_left_out(self):
        # An MLX conversion looks like the model it came from in every field
        # the search reads - same pipeline tag, same library, weights in
        # safetensors files - and lmstudio-community publishes one per
        # precision, so the four of them would crowd out google's own.
        self.found = [
            hub_result(
                "lmstudio-community/gemma-4-E4B-it-MLX-4bit",
                "any-to-any",
                tags=["transformers", "safetensors", "mlx"],
            ),
            hub_result(
                "google/gemma-4-E4B-it",
                "any-to-any",
                tags=["transformers", "safetensors"],
            ),
        ]

        found = search_hub_models("gemma")

        self.assertEqual([result.model_id for result in found], ["google/gemma-4-E4B-it"])

    def test_a_gguf_only_repository_is_left_out(self):
        # Downloading one would fetch the whole snapshot for a load that
        # cannot happen: AutoModelForCausalLM does not read GGUF, and
        # judge_snapshot calls the same files unsupported once they land.
        self.found = [
            hub_result(
                "unsloth/gemma-4-E4B-it-qat-GGUF",
                "any-to-any",
                tags=["transformers", "gguf"],
            ),
            hub_result(
                "google/gemma-4-E4B-it",
                "any-to-any",
                tags=["transformers", "safetensors"],
            ),
        ]

        found = search_hub_models("gemma")

        self.assertEqual([result.model_id for result in found], ["google/gemma-4-E4B-it"])

    def test_a_repository_shipping_both_formats_is_kept(self):
        # The GGUF files sit beside a Transformers checkpoint that loads, so
        # the repository is not a dead end. judge_snapshot agrees: a snapshot
        # is only unsupported where it holds no Transformers checkpoint. Both
        # of the formats in WEIGHT_FORMATS count, safetensors and the older
        # pytorch_model.bin alike.
        self.found = [
            hub_result(
                "org/model-with-a-gguf-folder",
                "text-generation",
                tags=["transformers", "safetensors", "gguf"],
            ),
            hub_result(
                "org/model-from-before-safetensors",
                "text-generation",
                tags=["transformers", "pytorch", "gguf"],
            ),
        ]

        found = search_hub_models("model")

        self.assertEqual(
            [result.model_id for result in found],
            ["org/model-with-a-gguf-folder", "org/model-from-before-safetensors"],
        )

    def test_a_repository_in_another_framework_is_left_out(self):
        # The hub files a TensorFlow or Flax checkpoint under its framework,
        # not under the .h5 or .msgpack it is written in, and _load_locked
        # passes neither from_tf nor from_flax. judge_snapshot already calls
        # both suffixes foreign once they are on disk.
        self.found = [
            hub_result(
                "org/tensorflow-only", "text-generation", tags=["transformers", "tf"]
            ),
            hub_result("org/flax-only", "text-generation", tags=["transformers", "jax"]),
            hub_result(
                "allenai/Olmo-3-7B-Think",
                "text-generation",
                tags=["transformers", "safetensors"],
            ),
        ]

        found = search_hub_models("model")

        self.assertEqual([result.model_id for result in found], ["allenai/Olmo-3-7B-Think"])

    def test_matches_below_the_rejected_ones_still_fill_the_list(self):
        # The hub sorts by downloads and the checks here run afterwards, so a
        # query whose most-downloaded matches are all embedding models must
        # not report that nothing matched: the reading goes on past them.
        self.found = [
            hub_result(f"org/embedder-{n}", "feature-extraction") for n in range(150)
        ] + [hub_result("allenai/Olmo-3-7B-Think", "text-generation")]

        found = search_hub_models("olmo", limit=3)

        self.assertEqual([result.model_id for result in found], ["allenai/Olmo-3-7B-Think"])

    def test_the_reading_stops_rather_than_paging_through_the_hub(self):
        # A query that matches nothing loadable would otherwise page to the
        # end of the hub for a list that stays empty.
        self.found = [
            hub_result(f"org/embedder-{n}", "feature-extraction")
            for n in range(SEARCH_SCAN_LIMIT + 50)
        ] + [hub_result("allenai/Olmo-3-7B-Think", "text-generation")]

        self.assertEqual(search_hub_models("olmo"), [])

    def test_the_list_stops_at_the_limit(self):
        # More matches than the pane shows: the extras were fetched to make
        # up for what the tag check drops, not to lengthen the list.
        self.found = [hub_result(f"org/model-{n}", "text-generation") for n in range(30)]

        found = search_hub_models("model", limit=3)

        self.assertEqual(
            [result.model_id for result in found],
            ["org/model-0", "org/model-1", "org/model-2"],
        )

    def test_empty_queries_browse_with_each_hub_order_and_keep_compatibility_checks(self):
        self.found = [
            hub_result("org/converted", "text-generation", tags=["mlx"]),
            hub_result("org/native", "text-generation"),
        ]
        for order, expected in [
            ("Popular", "downloads"), ("Trending", "trending_score"), ("New", "created_at")
        ]:
            with self.subTest(order=order):
                results = search_hub_models("   ", order=order)
                self.assertEqual([r.model_id for r in results], ["org/native"])
                self.assertIsNone(self.calls[-1]["search"])
                self.assertEqual(self.calls[-1]["sort"], expected)

    def test_invalid_order_does_not_go_online(self):
        with self.assertRaises(ValueError):
            search_hub_models("", order="wrong")
        self.assertEqual(self.calls, [])
