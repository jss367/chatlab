"""Pictures in a conversation: kept, saved, laid out for a vision model, and fed.

The layout tests build tiny vision models from random weights and compare a
prompt fed the way ChatLab feeds one - in chunks, from a cache, a sampled
token at a time - with the same prompt fed whole through Transformers' own
path, pixels and all. Agreement there is what lets every token measurement
stand on a conversation with a picture in it.
"""

import base64
import io
import json
import math
import unittest
from unittest import mock

import torch
from PIL import Image

import settings_sandbox
from chatlab import attachments, vision
from chatlab.conversation import (
    PICTURE_MARK,
    branch_title,
    display_messages,
    from_json,
    make_turn,
    model_messages,
    to_json,
    turn_entries,
    turns_from_entries,
)
from chatlab.model_runtime import ModelManager
from chatlab.token_metrics import UNSCORED_IMAGE
from chatlab.ui import pictures


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


def png_bytes(size=(40, 30), color="red", mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class AttachmentStoreTests(unittest.TestCase):
    def test_a_picture_is_kept_once_under_the_hash_of_its_bytes(self):
        data = png_bytes()
        name = attachments.store_image(data)
        self.assertRegex(name, r"^[0-9a-f]{64}\.png$")
        self.assertEqual(attachments.store_image(data), name)
        self.assertEqual(attachments.read_bytes(name), data)
        self.assertEqual(attachments.image_path(name).parent, attachments.image_directory())

    def test_a_format_that_is_not_kept_is_written_as_png(self):
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), "blue").save(buffer, format="BMP")
        name = attachments.store_image(buffer.getvalue())
        self.assertTrue(name.endswith(".png"))
        with Image.open(attachments.image_path(name)) as image:
            self.assertEqual(image.format, "PNG")

    def test_what_is_not_a_picture_is_refused(self):
        with self.assertRaises(attachments.AttachmentError):
            attachments.store_image(b"not a picture at all")
        with self.assertRaises(attachments.AttachmentError):
            attachments.store_image(b"")

    def test_a_picture_too_large_is_refused_before_it_is_read(self):
        with mock.patch.object(attachments, "MAX_IMAGE_BYTES", 10):
            with self.assertRaises(attachments.AttachmentError) as caught:
                attachments.store_image(png_bytes())
        self.assertIn("MB", str(caught.exception))

    def test_a_picture_with_too_many_pixels_is_refused_before_decoding(self):
        data = png_bytes(size=(200, 100))
        name = attachments.store_image(data)
        with mock.patch.object(attachments, "MAX_IMAGE_PIXELS", 100 * 100):
            with self.assertRaises(attachments.AttachmentError) as caught:
                attachments.store_image(png_bytes(size=(200, 101)))
            self.assertIn("megapixels", str(caught.exception))
            with self.assertRaises(attachments.AttachmentError):
                attachments.open_for_model(name)

    def test_a_tampered_picture_fails_its_integrity_check(self):
        name = attachments.store_image(png_bytes(color="green"))
        attachments.image_path(name).write_bytes(png_bytes(color="purple"))
        with self.assertRaises(attachments.AttachmentError):
            attachments.read_bytes(name)

    def test_a_model_is_shown_a_large_picture_scaled_to_the_budget(self):
        name = attachments.store_image(png_bytes(size=(400, 300)))
        image = attachments.open_for_model(name, budget=100 * 75)
        self.assertEqual(image.mode, "RGB")
        self.assertLessEqual(image.width * image.height, 100 * 75)
        self.assertAlmostEqual(image.width / image.height, 4 / 3, places=1)
        small = attachments.open_for_model(name)
        self.assertEqual(small.size, (400, 300))

    def test_a_transparent_screenshot_is_shown_on_white(self):
        name = attachments.store_image(png_bytes(color=(0, 0, 0, 0), mode="RGBA"))
        image = attachments.open_for_model(name)
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_turn_picture_names_are_checked(self):
        name = attachments.store_image(png_bytes(color="orange"))
        self.assertEqual(attachments.image_names([name]), [name])
        self.assertEqual(attachments.image_names(None), [])
        for bad in ("x.png", ["../etc/passwd"], [name.upper()], name):
            with self.assertRaises(ValueError):
                attachments.image_names(bad)
        with self.assertRaises(attachments.AttachmentError):
            attachments.image_path("../../secrets.png")

    def test_pictures_travel_inside_a_conversation_file(self):
        name = attachments.store_image(png_bytes(color="teal"))
        exported = attachments.export_images([name])
        attachments.image_path(name).unlink()
        attachments.import_images([name], exported)
        self.assertEqual(attachments.read_bytes(name), base64.b64decode(exported[name]))

    def test_an_import_refuses_a_missing_or_mismatched_picture(self):
        name = attachments.store_image(png_bytes(color="navy"))
        attachments.image_path(name).unlink()
        with self.assertRaises(attachments.AttachmentError):
            attachments.import_images([name], {})
        wrong = base64.b64encode(png_bytes(color="pink")).decode()
        with self.assertRaises(attachments.AttachmentError):
            attachments.import_images([name], {name: wrong})


class PruneTests(unittest.TestCase):
    def test_only_old_pictures_nothing_names_are_swept(self):
        import os
        import time

        kept = attachments.store_image(png_bytes(color=(1, 2, 3)))
        orphan = attachments.store_image(png_bytes(color=(4, 5, 6)))
        fresh = attachments.store_image(png_bytes(color=(7, 8, 9)))
        old = time.time() - 2 * attachments.UNREFERENCED_GRACE_SECONDS
        for name in (kept, orphan):
            os.utime(attachments.image_path(name), (old, old))
        source = attachments.image_directory().parent / "refs.json"
        source.write_text(json.dumps({"turns": [{"images": [kept]}]}))
        attachments.prune_unreferenced([source, source.with_name("absent.json")])
        self.assertTrue(attachments.image_path(kept).is_file())
        self.assertTrue(attachments.image_path(fresh).is_file())
        self.assertFalse(attachments.image_path(orphan).exists())

    def test_an_unreadable_source_stops_the_sweep(self):
        import os
        import time

        orphan = attachments.store_image(png_bytes(color=(10, 11, 12)))
        old = time.time() - 2 * attachments.UNREFERENCED_GRACE_SECONDS
        os.utime(attachments.image_path(orphan), (old, old))
        unreadable = attachments.image_directory().parent / "refs-dir.json"
        unreadable.mkdir(exist_ok=True)
        self.assertEqual(attachments.prune_unreferenced([unreadable]), 0)
        self.assertTrue(attachments.image_path(orphan).is_file())


class ConversationPictureTests(unittest.TestCase):
    def setUp(self):
        self.name = attachments.store_image(png_bytes(color="yellow"))
        self.turn = make_turn("user", "What is this?")
        self.turn["images"] = [self.name]

    def test_the_model_is_given_pictures_beside_the_text(self):
        messages = model_messages([self.turn], system_prompt="Be brief.")
        self.assertEqual(messages[1], {"role": "user", "content": "What is this?", "images": [self.name]})
        self.assertEqual(vision.message_images(messages), [self.name])

    def test_a_message_of_pictures_alone_still_reaches_the_model(self):
        self.turn["content"] = ""
        self.assertEqual(model_messages([self.turn])[0]["images"], [self.name])

    def test_each_picture_is_a_chatbot_message_ahead_of_the_text(self):
        messages, index = display_messages([self.turn])
        self.assertEqual(messages[0]["content"]["path"], str(attachments.image_path(self.name)))
        self.assertEqual(messages[1]["content"], "What is this?")
        self.assertEqual(index, [(0, "image"), (0, "content")])

    def test_a_missing_picture_is_said_rather_than_drawn_broken(self):
        missing = make_turn("user", "")
        missing["images"] = ["0" * 64 + ".png"]
        messages, _ = display_messages([missing])
        self.assertIn("no longer on this machine", messages[0]["content"])

    def test_pictures_survive_the_conversations_file_and_a_saved_file(self):
        self.assertEqual(turns_from_entries(turn_entries([self.turn]))[0]["images"], [self.name])
        payload = to_json([self.turn])
        self.assertIn(self.name, json.loads(payload)["images"])
        attachments.image_path(self.name).unlink()
        turns, _ = from_json(payload)
        self.assertEqual(turns[0]["images"], [self.name])
        self.assertTrue(attachments.image_path(self.name).is_file())

    def test_a_title_says_the_message_had_a_picture(self):
        self.assertTrue(branch_title([self.turn]).startswith(PICTURE_MARK))

    def test_the_composer_strip_has_one_thumbnail_per_picture(self):
        html = pictures.strip_html([self.name])
        self.assertEqual(html.count('class="picture-chip"'), 1)
        self.assertIn(f'data-name="{self.name}"', html)
        self.assertEqual(pictures.strip_html([]), "")

    def test_attaching_stores_files_and_removing_takes_one_off(self):
        upload = attachments.image_directory().parent / "upload.png"
        upload.write_bytes(png_bytes(color="lime"))
        names, strip, status = pictures.attach_pictures([str(upload)], [self.name])
        self.assertEqual(len(names), 2)
        self.assertEqual(strip.count("picture-chip"), 2)
        names, strip = pictures.remove_picture(f"{self.name}|123", names)
        self.assertNotIn(self.name, names)
        self.assertEqual(strip.count("picture-chip"), 1)

    def test_attaching_a_file_that_is_not_a_picture_says_so(self):
        upload = attachments.image_directory().parent / "notes.txt"
        upload.write_text("hello")
        names, _strip, status = pictures.attach_pictures([str(upload)], [])
        self.assertEqual(names, [])
        self.assertIn("not a picture", status)


# -- tiny vision models ------------------------------------------------------

IMAGE, START, END = 100, 101, 102


def tiny_qwen():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=["linear_attention", "full_attention"] * 2,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
            rope_parameters={
                "rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2],
                "mrope_interleaved": True, "partial_rotary_factor": 1.0,
            },
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
            patch_size=4, spatial_merge_size=2, temporal_patch_size=2,
            num_position_embeddings=64, in_channels=3,
        ),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END,
        video_token_id=103,
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


def tiny_gemma():
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    torch.manual_seed(1)
    return Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=dict(
            vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=16,
            layer_types=["sliding_attention", "full_attention"],
        ),
        vision_config=dict(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=4,
        ),
        mm_tokens_per_image=4, image_token_index=260, boi_token_index=261, eoi_token_index=262,
    )).eval()


def qwen_inputs(grids):
    """Pixels and a prompt for pictures of ``grids`` patches each, Qwen's way."""

    grid = torch.tensor(grids)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 4 * 4)
    ids = [1, 2, 3]
    for t, h, w in grids:
        ids += [START] + [IMAGE] * (t * h * w // 4) + [END]
    ids += list(range(4, 30))
    return ids, {"pixel_values": pixels, "image_grid_thw": grid}


class FakeProcessor:
    """A processor whose image half hands back fixed pixels, as a real one would compute them."""

    def __init__(self, prepared, ids=None):
        self.prepared = prepared
        self.ids = ids
        self.image_processor = lambda images, return_tensors: dict(prepared)
        self.calls = 0

    def __call__(self, text, images, return_tensors, add_special_tokens):
        self.calls += 1
        self.rendered = text[0]
        return {"input_ids": torch.tensor([self.ids])}


def fed_like_chatlab(model, ids, layout, chunk):
    """Logits for ``ids`` fed ``chunk`` tokens at a time from a growing cache."""

    cache, rows, start = None, [], 0
    while start < len(ids):
        end = min(layout.chunk_end(start + chunk), len(ids))
        out = model(
            input_ids=torch.tensor([ids[start:end]]),
            attention_mask=torch.ones(1, end, dtype=torch.long),
            past_key_values=cache, use_cache=True,
            **layout.forward_arguments(start, end - start),
        )
        cache = out.past_key_values
        rows.append(out.logits[0])
        start = end
    return torch.cat(rows)


class MediaLayoutTests(unittest.TestCase):
    def check(self, model, ids, prepared, native_kwargs, prompt_length):
        with torch.inference_mode():
            native = model(
                input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                **native_kwargs,
            ).logits[0]
            features = vision.encode_images(model, FakeProcessor(prepared), [None])
            # The layout covers the prompt; what follows it is text it grew by.
            layout = vision.MediaLayout(model, ids[:prompt_length], features)
            for chunk in (len(ids), 5, 3, 1):
                fed = fed_like_chatlab(model, ids, layout, chunk)
                self.assertTrue(
                    torch.allclose(fed, native, atol=1e-4),
                    f"chunk {chunk}: {(fed - native).abs().max().item()}",
                )

    def test_qwen_pictures_and_their_positions_match_the_models_own_path(self):
        model = tiny_qwen()
        ids, prepared = qwen_inputs([[1, 4, 6], [1, 4, 4]])
        types = torch.tensor([[int(token == IMAGE) for token in ids]])
        self.check(model, ids, prepared, dict(prepared, mm_token_type_ids=types), len(ids) - 8)

    def test_gemma_pictures_attend_both_ways_when_fed_whole(self):
        model = tiny_gemma()
        pixels = torch.randn(2, 3, 16, 16)
        ids = [2, 5, 6, 261] + [260] * 4 + [262, 7, 8, 261] + [260] * 4 + [262] + list(range(10, 30))
        types = torch.tensor([[int(token == 260) for token in ids]])
        self.check(model, ids, {"pixel_values": pixels},
                   {"pixel_values": pixels, "token_type_ids": types}, len(ids) - 5)

    def test_llava_pictures_match_the_models_own_path(self):
        from transformers import LlavaConfig, LlavaForConditionalGeneration

        torch.manual_seed(2)
        model = LlavaForConditionalGeneration(LlavaConfig(
            text_config=dict(
                model_type="llama", vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            ),
            vision_config=dict(
                model_type="clip_vision_model", hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=4,
                projection_dim=16,
            ),
            image_token_index=120, vision_feature_select_strategy="full",
        )).eval()
        pixels = torch.randn(1, 3, 16, 16)
        ids = [1, 2, 3] + [120] * 17 + list(range(4, 30))
        self.check(model, ids, {"pixel_values": pixels}, {"pixel_values": pixels}, len(ids) - 5)

    def test_a_prompt_whose_picture_tokens_were_edited_is_refused(self):
        model = tiny_qwen()
        ids, prepared = qwen_inputs([[1, 4, 4]])
        features = vision.encode_images(model, FakeProcessor(prepared), [None])
        edited = [token if token != IMAGE else 7 for token in ids]
        with self.assertRaises(ValueError) as caught:
            vision.MediaLayout(model, edited, features)
        self.assertIn("picture", str(caught.exception))

    def test_only_a_picture_read_both_ways_moves_a_chunk_boundary(self):
        model = tiny_qwen()
        ids, prepared = qwen_inputs([[1, 4, 4]])
        layout = vision.MediaLayout(model, ids, vision.encode_images(model, FakeProcessor(prepared), [None]))
        self.assertFalse(layout.joined)
        self.assertEqual(layout.chunk_end(5), 5)
        self.assertEqual(layout.run_start(6), 6)


class ChatTemplateTokenizer:
    """Just enough tokenizer for a prompt with pictures and a reply."""

    chat_template = "fake"

    def __init__(self, size=128):
        self.pieces = [f"<{index}>" for index in range(size)]
        self.eos_token_id = size - 1
        self.all_special_ids = [size - 1]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        self.template = messages
        return "rendered"

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return "".join(self.pieces[int(token)] for token in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[int(token_id)]


class VisionManagerTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_qwen()
        torch.manual_seed(3)
        self.ids, self.prepared = qwen_inputs([[1, 4, 4]])
        self.ids = self.ids[:12]
        self.manager = ModelManager()
        self.manager.model = self.model
        self.manager.tokenizer = ChatTemplateTokenizer()
        self.manager.processor = FakeProcessor(self.prepared, self.ids)
        self.manager.model_id = "test/vision"
        self.picture = attachments.store_image(png_bytes(color="white"))
        self.messages = [{"role": "user", "content": "What is it?", "images": [self.picture]}]

    def reply(self, **options):
        updates = list(self.manager.generate(
            self.messages, temperature=0, top_p=1, top_k=0, max_new_tokens=4, seed=0, **options,
        ))
        return updates[-1]

    def test_a_reply_reads_the_picture_and_measures_around_it(self):
        last = self.reply()
        self.assertEqual(list(last.prompt_ids), self.ids)
        template = self.manager.tokenizer.template[0]["content"]
        self.assertEqual(template, [{"type": "image"}, {"type": "text", "text": "What is it?"}])
        placeholders = [metric for metric in last.prompt_metrics if metric["token_id"] == IMAGE]
        self.assertEqual(len(placeholders), 4)
        self.assertTrue(all(metric["unscored_reason"] == UNSCORED_IMAGE for metric in placeholders))
        # The first reply token is measured against what the model really
        # predicted with the picture in front of it.
        with torch.inference_mode():
            types = torch.tensor([[int(token == IMAGE) for token in self.ids]])
            native = self.model(
                input_ids=torch.tensor([self.ids]), mm_token_type_ids=types, **self.prepared
            ).logits[0, -1].float().log_softmax(-1)
        first = last.metrics[0]
        self.assertEqual(first["token_id"], int(native.argmax()))
        self.assertAlmostEqual(first["raw_probability"], float(native.exp().max()), places=4)

    def test_an_inspection_sees_the_picture_and_the_encoder_runs_once(self):
        with mock.patch.object(vision, "encode_images", wraps=vision.encode_images) as encode:
            last = self.reply()
            ids = list(last.prompt_ids) + [metric["token_id"] for metric in last.metrics]
            insight = self.manager.inspect(
                ids, len(last.prompt_ids) + 1, context_count=len(last.prompt_ids),
                images=[self.picture],
            )
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(insight.layers[-1]["rank"], last.metrics[1]["raw_rank"])
        self.assertAlmostEqual(
            insight.layers[-1]["probability"], last.metrics[1]["raw_probability"], places=4
        )
        view = self.manager.read_kv_cache(ids[: len(last.prompt_ids) + 1], 1, images=[self.picture])
        self.assertIsNotNone(view["summary"])
        with self.assertRaises(Exception):
            self.manager.read_kv_cache(ids[: len(last.prompt_ids) + 1], 1)

    def test_a_picture_token_is_not_inspected(self):
        last = self.reply()
        with self.assertRaises(ValueError) as caught:
            self.manager.inspect(list(last.prompt_ids), self.ids.index(IMAGE) + 1, images=[self.picture])
        self.assertIn("picture's tokens", str(caught.exception))

    def test_inspecting_just_after_a_picture_read_both_ways_feeds_it_whole(self):
        model = tiny_gemma()
        pixels = torch.randn(1, 3, 16, 16)
        ids = [2, 5, 6, 261] + [260] * 4 + [262, 7, 8, 9]
        manager = ModelManager()
        manager.model = model
        manager.tokenizer = ChatTemplateTokenizer(300)
        manager.processor = FakeProcessor({"pixel_values": pixels}, ids)
        manager.model_id = "test/gemma"
        # The token after the picture's last placeholder is predicted from a
        # step that feeds that placeholder; the picture has to go in whole.
        index = ids.index(262)
        insight = manager.inspect(ids, index, images=[self.picture])
        with torch.inference_mode():
            types = torch.tensor([[int(token == 260) for token in ids]])
            native = model(
                input_ids=torch.tensor([ids]), pixel_values=pixels, token_type_ids=types
            ).logits[0, index - 1].float().softmax(-1)
        # Compared on a log scale: a random model's probabilities are all
        # near one in three hundred, and a picture fed in two pieces moves
        # them by less than a fixed number of decimal places can see.
        self.assertAlmostEqual(
            math.log(insight.layers[-1]["probability"]), math.log(float(native[ids[index]])), delta=1e-5
        )

    def test_a_text_model_refuses_pictures_and_says_what_to_load(self):
        self.manager.processor = None
        self.manager.vision_note = None
        self.assertFalse(self.manager.accepts_images)
        with self.assertRaises(vision.ImagesUnsupported) as caught:
            self.reply()
        message = str(caught.exception)
        self.assertIn("test/vision can't be shown pictures: it is a text-only model", message)
        self.assertIn(vision.VISION_SUGGESTION, message)


class SendTests(unittest.TestCase):
    """The composer's pictures go with the message, or stay put when they cannot."""

    def setUp(self):
        from conversation_support import FIXED
        from fakes import loaded_manager
        from chatlab.ui import runtime

        self.settings = FIXED
        self.manager = loaded_manager([0, 1])
        patcher = mock.patch.object(runtime, "MANAGER", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.picture = attachments.store_image(png_bytes(color="gray"))

    def send(self, text, images):
        from chatlab.ui.generation import chat

        return list(chat(text, [], **self.settings, images=images))

    def test_a_text_model_keeps_the_message_and_its_pictures_in_the_box(self):
        frames = self.send("What is this?", [self.picture])
        self.assertEqual(len(frames), 1)
        self.assertIn("can't be shown pictures", frames[0]["status"])
        self.assertEqual(frames[0]["prompt"], "What is this?")
        self.assertEqual(frames[0]["turns"], [])
        self.assertNotIn("attachments", dict(frames[0]))

    def test_a_text_model_refuses_a_follow_up_to_an_earlier_picture(self):
        from chatlab.ui.generation import chat, retry_last

        earlier = make_turn("user", "Look")
        earlier["images"] = [self.picture]
        history = [earlier, make_turn("assistant", "A gray square.")]
        frames = list(chat("And now?", history, **self.settings))
        self.assertEqual(len(frames), 1)
        self.assertIn("can't be shown pictures", frames[0]["status"])
        self.assertEqual(frames[0]["prompt"], "And now?")
        self.assertEqual(len(frames[0]["turns"]), 2)
        frames = list(retry_last("", history, *self.settings.values()))
        self.assertEqual(len(frames), 1)
        self.assertIn("can't be shown pictures", frames[0]["status"])
        self.assertEqual(frames[0]["turns"][1]["content"], "A gray square.")

    def test_a_sent_picture_joins_the_turn_and_leaves_the_box(self):
        self.manager.processor = object()
        original = self.manager.generate
        asked = []

        def generate(messages, **options):
            asked.append(messages)
            yield from original([{**m, "images": []} for m in messages], **options)

        self.manager.generate = generate
        frames = self.send("", [self.picture])
        self.assertEqual(asked[0][-1]["images"], [self.picture])
        self.assertEqual(frames[0]["attachments"], [])
        self.assertEqual(frames[0]["attachment_strip"], "")
        self.assertEqual(frames[-1]["turns"][0]["images"], [self.picture])
        # Only the opening frame writes the box.
        self.assertTrue(all("attachments" not in dict(frame) for frame in frames[1:]))


class ProcessorTests(unittest.TestCase):
    def test_a_transformers_without_encoder_output_reuse_is_refused_up_front(self):
        model = tiny_qwen()

        def forward(input_ids=None, pixel_values=None):
            raise AssertionError("not called")

        with mock.patch.object(model, "forward", forward):
            processor, note = vision.read_processor(attachments.image_directory(), model)
        self.assertIsNone(processor)
        self.assertIn("newer Transformers", note)


class CheckpointTests(unittest.TestCase):
    def write(self, directory, config, processor=True):
        (directory / "config.json").write_text(json.dumps(config))
        if processor:
            (directory / "preprocessor_config.json").write_text("{}")

    def test_a_vision_checkpoint_is_read_with_its_encoder(self):
        import tempfile
        from pathlib import Path

        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write(path, {"model_type": "qwen3_5", "vision_config": {}})
            self.assertTrue(vision.checkpoint_reads_images(path))
            self.assertIs(vision.model_class(path), AutoModelForImageTextToText)
            self.write(path, {"model_type": "qwen3_5", "vision_config": {}}, processor=False)
            (path / "preprocessor_config.json").unlink()
            self.assertIs(vision.model_class(path), AutoModelForCausalLM)
            self.write(path, {"model_type": "llama"})
            self.assertFalse(vision.checkpoint_reads_images(path))


if __name__ == "__main__":
    unittest.main()
