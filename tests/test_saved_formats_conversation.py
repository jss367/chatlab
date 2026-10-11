"""Saved conversation files (chatlab-conversation-1) as every earlier writer left them.

Each file under tests/fixtures/saved/conversation was written by the commit it
is named after (see its generator) and is never regenerated. These tests load
each one through Chat's "Load conversation" handler, ``load_with_steering``,
and pin what the reader takes from it. Each is also saved again with today's
``to_json`` and loaded once more, on a machine that has none of its vectors
or pictures yet, to check the same reading comes back.
"""

import hashlib
import json
import unittest

from chatlab import steering
from chatlab.conversation import from_json, new_forks, to_json
from chatlab.ui.conversations import open_conversation
from chatlab.ui.steering import load_with_steering

from saved_formats_support import SAVED, sandbox

CONVERSATION = SAVED / "conversation"


class SavedConversationCase(unittest.TestCase):
    fixture = ""

    def setUp(self):
        self.box = sandbox(self)
        self.path = self.box.copy(CONVERSATION / self.fixture)
        self.frame = load_with_steering(str(self.path), [], "Raw rank", new_forks())
        self.assertTrue(self.frame["status"].startswith("Loaded "), self.frame["status"])
        self.turns = self.frame["turns"]

    def texts(self):
        return [(turn["role"], turn["content"]) for turn in self.turns]

    def resaved(self):
        """Today's writer saves what was loaded; read back on a fresh machine."""

        payload = to_json(self.turns, system_prompt=self.frame["system_prompt"],
                          steering=self.frame["steering_state"])
        fresh = sandbox(self)
        fresh.use_library(fresh.root / "elsewhere" / "conversations.json")
        target = fresh.root / "resaved.json"
        target.write_text(payload, encoding="utf-8")
        frame = load_with_steering(str(target), [], "Raw rank", new_forks())
        self.assertTrue(frame["status"].startswith("Loaded "), frame["status"])
        return frame

    def assert_resave_keeps_reading(self):
        frame = self.resaved()
        self.assertEqual(frame["turns"], self.turns)
        self.assertEqual(frame["system_prompt"], self.frame["system_prompt"])
        self.assertEqual(frame["steering_state"], self.frame["steering_state"])
        self.assertEqual(len(frame["chatbot"]), len(self.frame["chatbot"]))
        return frame


class FirstEraTests(SavedConversationCase):
    """890d94a: system prompt and turns of role, content and reasoning."""

    fixture = "01-first-890d94a.json"

    def test_turns(self):
        self.assertEqual(self.frame["system_prompt"], "You are a patient teacher.\nKeep answers short.")
        self.assertEqual(self.texts(), [
            ("user", "Explain recursion in one sentence."),
            ("assistant", "Recursion is when a function solves a problem by calling itself on a smaller piece of it."),
            ("user", "Give an example in Python, with <code> tags."),
            ("assistant", "<code>def f(n): return 1 if n == 0 else n * f(n - 1)</code>"),
            ("user", "Now in Haskell — with “quotes”."),
        ])
        self.assertEqual(self.turns[1]["reasoning"],
                         "Keep it to one sentence.\nMention the base idea of smaller subproblems.")
        self.assertEqual(set(self.turns[1]), {"role", "content", "reasoning"})

    def test_screen(self):
        self.assertEqual(self.frame["status"], "Loaded 5 messages.")
        self.assertEqual(self.frame["chatbot"][1], {
            "role": "assistant", "content": "Keep it to one sentence.\nMention the base idea of smaller subproblems.",
            "metadata": {"title": "Reasoning", "status": "done"},
        })
        self.assertEqual(len(self.frame["chatbot"]), 6)
        self.assertIsNone(self.frame["steering_state"])
        self.assertEqual(self.frame["forks"]["sampling"]["Main"], {"steering": None})

    def test_opened_from_an_extension(self):
        # The same payload handed to Chat by an extension opens as a new conversation.
        frame = open_conversation(self.path.read_text(encoding="utf-8"), [], new_forks())
        self.assertEqual(frame["turns"], self.turns)
        self.assertEqual(frame["system_prompt"], "You are a patient teacher.\nKeep answers short.")
        self.assertIn("Press Retry to answer its last message.", frame["status"])

    def test_resave(self):
        self.assert_resave_keeps_reading()


class SteeringEraTests(SavedConversationCase):
    """392ad4d (#59): counts on replies, steering references and embedded vectors."""

    fixture = "02-steering-392ad4d.json"
    CONVERSATION_VECTOR = "56919d59f08928f25c65301428ec974a25338e1ff67a49b72ea39c674a795431"
    REPLY_VECTOR = "70d8f5737e1707883f247fbd02938ac1c68d2f92fa8348ff209ff7b62de7f637"

    def test_turns(self):
        self.assertEqual(self.frame["system_prompt"], "Be poetic.")
        self.assertEqual(self.texts(), [
            ("user", "Describe the sea."), ("assistant", "Grey, restless, endless."),
            ("user", "And the sky?"), ("assistant", "Bright and calm."),
        ])
        first, second = self.turns[1], self.turns[3]
        self.assertEqual((first["model"], first["prompt_tokens"], first["generated_tokens"]),
                         ("Qwen/Qwen3-0.6B", 25, 14))
        self.assertEqual((second["prompt_tokens"], second["generated_tokens"]), (52, 5))

    def test_steering(self):
        self.assertEqual(self.frame["steering_state"], {
            "format": "chatlab-steering-reference-1", "model_id": "Qwen/Qwen3-0.6B", "layer": 4,
            "strength": 8.0, "enabled": True, "vector_id": self.CONVERSATION_VECTOR, "width": 5,
        })
        self.assertEqual(self.frame["steering_enabled"]["value"], True)
        self.assertEqual(self.frame["steering_strength"]["value"], 8.0)
        self.assertEqual(self.frame["steering_layer"]["value"], 4)
        self.assertEqual(self.frame["forks"]["sampling"]["Main"]["steering"]["vector_id"], self.CONVERSATION_VECTOR)
        reply = self.turns[1]["steering"]
        self.assertEqual((reply["vector_id"], reply["strength"], reply["enabled"], reply["layer"]),
                         (self.REPLY_VECTOR, -1.5, False, 1))
        # The embedded vectors were installed and resolve to their numbers.
        self.assertEqual(steering.expand(self.frame["steering_state"])["vector"], [0.25, -0.75, 1.5, 0.0, -2.0])
        self.assertEqual(steering.expand(reply)["vector"], [1.0, 2.0, 3.0])

    def test_embedded_vectors_are_checked(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(set(data["steering_vectors"]), {self.CONVERSATION_VECTOR, self.REPLY_VECTOR})
        data["steering_vectors"][self.REPLY_VECTOR]["vector"][0] = 9.0
        fresh = sandbox(self)
        fresh.use_library(fresh.root / "other" / "conversations.json")
        with self.assertRaises(steering.SteeringError):
            from_json(json.dumps(data))

    def test_resave(self):
        frame = self.assert_resave_keeps_reading()
        self.assertEqual(steering.expand(frame["turns"][1]["steering"])["vector"], [1.0, 2.0, 3.0])


class PicturesEraTests(SavedConversationCase):
    """b69af3b (#176): embedded pictures, plus errors, pauses and settings snapshots."""

    fixture = "03-pictures-b69af3b.json"
    YELLOW = "d6324f93e3487f7727477e2c2ff108882e7b866daca4c0f256651a7dcd72faec.png"
    PURPLE = "3a5b525e0ad11a1888d32150c65d458fb89ff212987c293ed509423bed743d0f.png"

    def test_pictures_installed(self):
        self.assertEqual(self.turns[0]["images"], [self.YELLOW, self.PURPLE])
        self.assertEqual(self.turns[2]["images"], [self.PURPLE])
        for name in (self.YELLOW, self.PURPLE):
            stored = self.box.root / "conversations-images" / name
            self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), name.split(".")[0])
        pictures = [m["content"]["path"] for m in self.frame["chatbot"] if isinstance(m["content"], dict)]
        self.assertEqual([path.rsplit("/", 1)[-1] for path in pictures], [self.YELLOW, self.PURPLE, self.PURPLE])

    def test_reply_fields(self):
        answer, failed, paused = self.turns[1], self.turns[3], self.turns[5]
        self.assertEqual(answer["thinking_mode"], "default")
        self.assertEqual(answer["generation_settings"]["seed"], 9)
        self.assertEqual(failed["generation_settings"]["seed"], 10)
        self.assertEqual((failed["content"], failed["error"]),
                         ("The purple one", "Generation stopped: the device ran out of memory."))
        self.assertTrue(paused["token_step_paused"])
        self.assertEqual(paused["content"], "")
        self.assertIn("Generation failed: Generation stopped: the device ran out of memory.",
                      self.frame["chatbot"][7]["content"])
        self.assertEqual(self.frame["chatbot"][-1]["content"], "Paused before visible text.")

    def test_steering(self):
        self.assertEqual(self.frame["steering_state"]["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(steering.expand(self.frame["steering_state"])["vector"], [-0.5, 0.5, 0.25, -0.25])
        self.assertEqual(self.turns[1]["steering"], self.frame["steering_state"])

    def test_resave(self):
        frame = self.assert_resave_keeps_reading()
        self.assertEqual(frame["turns"][0]["images"], [self.YELLOW, self.PURPLE])


class CurrentEraTests(SavedConversationCase):
    """cfd9fbd: Institutions provenance on turns opened from a replay."""

    fixture = "04-current-cfd9fbd.json"

    def test_provenance(self):
        provenance = self.turns[0]["institutions_provenance"]
        self.assertEqual((provenance["run_id"], provenance["game_id"], provenance["turn_id"]),
                         ("support-run-42", "game-3", "turn-1"))
        self.assertFalse(provenance["accepted"])
        self.assertEqual(self.turns[1]["institutions_provenance"]["recorded_message_provenance"],
                         {"source": "replay"})
        self.assertEqual(self.frame["status"],
                         "Loaded 4 messages. Recorded parent: Qwen/Qwen3-4B @ 0123abcd. Recorded adapter: "
                         "example/support-lora/checkpoint-50 @ 4567ef01. Recorded weights are not loaded "
                         "automatically; loaded weight identity is unverified.")

    def test_turns(self):
        answer, failed = self.turns[1], self.turns[3]
        self.assertEqual(answer["content"], '{"action": "escalate", "reason": "damage"}')
        self.assertEqual((answer["prompt_tokens"], answer["generated_tokens"], answer["thinking_mode"]),
                         (812, 97, "on"))
        self.assertEqual(answer["generation_settings"]["assistant_prefill"], "{")
        self.assertEqual((failed["content"], failed["reasoning"], failed["error"]),
                         ("", "Checking the refund policy", "Generation stopped by the reader."))
        self.assertEqual(len(self.turns[0]["images"]), 1)

    def test_steering(self):
        self.assertFalse(self.frame["steering_enabled"]["value"])
        self.assertEqual(self.frame["steering_strength"]["value"], 12.5)
        self.assertEqual(steering.expand(self.frame["steering_state"])["vector"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_resave(self):
        self.assert_resave_keeps_reading()


if __name__ == "__main__":
    unittest.main()
