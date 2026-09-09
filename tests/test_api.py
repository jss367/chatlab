"""The local HTTP API: what it answers, and what it refuses."""

import json
import threading
from itertools import chain
import time
import unittest
from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api
import model_runtime
import settings
import settings_sandbox
from conversation import split_reasoning
from model_runtime import CachedModel, CacheStatus, GenerationUpdate, ScoredText
from ui import runtime

from test_streaming import loaded_manager


def setUpModule():
    settings_sandbox.start()


def tearDownModule():
    settings_sandbox.stop()


# "Hello", " world" and the reasoning markers, so a reply can be split into a
# reasoning block and an answer the way the chat splits it.
THINK_PIECES = ["<think>", "</think>", "Hello", " world", "!", "<eos>"]
THINK_EOS = THINK_PIECES.index("<eos>")


class Recorder:
    """A manager that answers nothing and remembers what it was asked.

    For the plumbing: which values reached the runtime, and whether the
    generation slot was given back. What a real generation produces is tested
    against the real ModelManager below.
    """

    def __init__(self, updates=None, raises=None):
        self.model_id = "fake/model"
        self.load_id = "fake/model#1"
        self.device_name = "CPU"
        self.precision = "full"
        self.busy = False
        self.calls = []
        self.updates = updates if updates is not None else [update("Hello")]
        self.raises = raises

    @property
    def loaded(self) -> bool:
        return self.model_id is not None

    def reserve_generation(self):
        if self.busy:
            return False
        self.busy = True
        return True

    def release_generation(self):
        self.busy = False

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.raises is not None:
            raise self.raises
        yield from self.updates


def metric(position, text, probability=0.5, rank=1, candidates=(("Hi", 0.3),)):
    return {
        "position": position,
        "token_id": position,
        "text": text,
        "display_text": text,
        "category": "Top choice",
        "raw_rank": rank,
        "raw_probability": probability,
        "sampling_probability": probability,
        "surprise_bits": 1.0,
        "probability_mass_above": 0.0,
        "entropy_bits": 2.0,
        "top1_margin": 0.2,
        "sampling_shift_bits": 0.0,
        "top_candidates": [
            {"token_id": 99, "text": text, "probability": probability},
            *(
                {"token_id": 100 + index, "text": name, "probability": value}
                for index, (name, value) in enumerate(candidates)
            ),
        ],
        "scored": True,
        "segment": "response",
        "unscored_reason": "",
    }


def update(text, metrics=None, prompt_ids=(1, 2, 3), **fields):
    return GenerationUpdate(
        text=text,
        metrics=metrics if metrics is not None else [metric(1, text)],
        load_id="fake/model#1",
        prompt_ids=tuple(prompt_ids),
        model_id="fake/model",
        **fields,
    )


class ApiTestCase(unittest.TestCase):
    """One client over the routes, with a manager the test owns."""

    manager_updates = None

    def setUp(self):
        self.manager = Recorder(self.manager_updates)
        self.use(self.manager)
        app = FastAPI()
        api.attach(app)
        self.client = TestClient(app)

    def use(self, manager):
        original = runtime.MANAGER
        runtime.MANAGER = manager
        self.addCleanup(setattr, runtime, "MANAGER", original)
        return manager

    def post(self, path="/v1/chat/completions", **body):
        return self.client.post(path, json=body)


class ModelListTests(ApiTestCase):
    def test_every_complete_model_is_listed_and_the_loaded_one_marked(self):
        entries = [
            CachedModel(
                model_id="fake/model",
                status=CacheStatus(cached_bytes=1000),
                updated=1700000000.0,
                architecture="OlmoForCausalLM",
            ),
            CachedModel(
                model_id="org/half",
                status=CacheStatus(cached_bytes=5, missing_files=("model weights",)),
            ),
        ]
        original = api.list_cached_models
        api.list_cached_models = lambda: entries
        self.addCleanup(setattr, api, "list_cached_models", original)

        body = self.client.get("/v1/models").json()

        self.assertEqual(body["object"], "list")
        self.assertEqual([entry["id"] for entry in body["data"]], ["fake/model"])
        listed = body["data"][0]
        self.assertEqual(listed["owned_by"], "fake")
        self.assertEqual(listed["created"], 1700000000)
        self.assertTrue(listed["chatlab"]["loaded"])
        self.assertEqual(listed["chatlab"]["size_bytes"], 1000)

    def test_the_status_says_what_would_answer_and_whether_it_can(self):
        body = self.client.get("/v1/chatlab/status").json()

        self.assertEqual(body["model"], "fake/model")
        self.assertEqual(body["device"], "CPU")
        self.assertEqual(body["precision"], "full")
        self.assertFalse(body["busy"])
        self.assertIn("pool", body["memory"])

    def test_the_status_answers_with_no_model_loaded(self):
        self.manager.model_id = None
        self.manager.device_name = None

        body = self.client.get("/v1/chatlab/status").json()

        self.assertIsNone(body["model"])


class RefusalTests(ApiTestCase):
    def test_a_request_with_no_model_loaded_is_refused_by_name(self):
        self.manager.model_id = None

        response = self.post(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(response.status_code, 409)
        body = response.json()
        self.assertEqual(body["error"]["type"], "model_not_loaded")
        self.assertIn("Models page", body["error"]["message"])

    def test_naming_another_model_is_refused_rather_than_answered(self):
        response = self.post(
            model="org/other", messages=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(response.status_code, 409)
        message = response.json()["error"]["message"]
        self.assertIn("org/other", message)
        self.assertIn("fake/model", message)
        self.assertEqual(self.manager.calls, [])

    def test_naming_the_loaded_model_is_answered(self):
        response = self.post(
            model="fake/model", messages=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(response.status_code, 200)

    def test_a_second_request_is_told_the_model_is_busy(self):
        self.manager.busy = True

        response = self.post(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["type"], "model_busy")

    def test_messages_must_be_a_conversation(self):
        for body, expected in [
            ({}, "messages must be a non-empty list."),
            ({"messages": []}, "messages must be a non-empty list."),
            ({"messages": ["hi"]}, "Every message must be an object."),
            (
                {"messages": [{"role": "tool", "content": "hi"}]},
                "Unsupported message role: 'tool'.",
            ),
            (
                {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
                "Message content must be a string.",
            ),
        ]:
            with self.subTest(body=body):
                response = self.client.post("/v1/chat/completions", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected, response.json()["error"]["message"])

    def test_a_prefill_needs_a_conversation_before_it(self):
        response = self.post(messages=[{"role": "assistant", "content": "Well,"}])

        self.assertEqual(response.status_code, 400)
        self.assertIn("needs a conversation", response.json()["error"]["message"])

    def test_a_value_outside_the_range_is_refused_with_the_nearest(self):
        response = self.post(
            messages=[{"role": "user", "content": "hi"}], temperature=40
        )

        self.assertEqual(response.status_code, 400)
        message = response.json()["error"]["message"]
        self.assertIn("temperature of 40", message)
        self.assertIn(str(settings.TEMPERATURE_RANGE[1]), message)

    def test_a_response_longer_than_the_context_limit_is_refused(self):
        response = self.post(
            messages=[{"role": "user", "content": "hi"}], max_tokens=10**7
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("max_new_tokens", response.json()["error"]["message"])

    def test_a_sampling_value_that_is_not_finite_is_refused(self):
        # JSON has NaN and Infinity. A NaN would slip past the range check,
        # every comparison against it being false, and the request would
        # quietly answer at the saved setting instead.
        for raw in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=raw):
                response = self.client.post(
                    "/v1/chat/completions",
                    content=(
                        '{"messages":[{"role":"user","content":"hi"}],'
                        f'"temperature":{raw}}}'
                    ),
                    headers={"content-type": "application/json"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("finite", response.json()["error"]["message"])
                self.assertEqual(self.manager.calls, [])

    def test_a_field_of_the_wrong_type_says_so(self):
        for body, expected in [
            ({"temperature": "hot"}, "temperature must be a number."),
            ({"stream": "yes"}, "stream must be true or false."),
            ({"logprobs": True, "top_logprobs": 99}, "top_logprobs must be"),
            ({"top_logprobs": 5}, "top_logprobs needs logprobs"),
            ({"seed": -1}, "seed must be a whole number"),
        ]:
            with self.subTest(body=body):
                response = self.post(
                    messages=[{"role": "user", "content": "hi"}], **body
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected, response.json()["error"]["message"])

    def test_a_refusal_on_the_way_to_the_first_token_is_an_http_error(self):
        # A prompt past the context limit is raised as the generator starts.
        # Streaming it inside a 200 would tell a client it had succeeded.
        self.use(Recorder(raises=ValueError("That conversation is 9,000 tokens")))

        for streaming in (False, True):
            with self.subTest(stream=streaming):
                response = self.post(
                    messages=[{"role": "user", "content": "hi"}], stream=streaming
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("9,000 tokens", response.json()["error"]["message"])
                self.assertFalse(runtime.MANAGER.busy)

    def test_a_model_that_will_not_fit_is_its_own_status(self):
        self.use(Recorder(raises=model_runtime.OutOfMemoryError("no room")))

        response = self.post(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(response.status_code, 507)
        self.assertEqual(response.json()["error"]["type"], "insufficient_memory")

    def test_the_slot_is_given_back_after_every_outcome(self):
        self.post(messages=[{"role": "user", "content": "hi"}])
        self.assertFalse(self.manager.busy)
        self.post(messages=[])
        self.assertFalse(self.manager.busy)


class RequestPlumbingTests(ApiTestCase):
    def test_a_field_left_out_takes_the_saved_setting(self):
        with settings.override(temperature=0.3, top_p=0.7, top_k=13, max_new_tokens=77):
            self.post(messages=[{"role": "user", "content": "hi"}])

        call = self.manager.calls[0]
        self.assertEqual(call["temperature"], 0.3)
        self.assertEqual(call["top_p"], 0.7)
        self.assertEqual(call["top_k"], 13)
        self.assertEqual(call["max_new_tokens"], 77)

    def test_what_the_request_names_wins(self):
        with settings.override(temperature=0.3):
            self.post(
                messages=[{"role": "user", "content": "hi"}],
                temperature=1.25,
                top_p=0.5,
                top_k=7,
                max_tokens=64,
                seed=99,
            )

        call = self.manager.calls[0]
        self.assertEqual(call["temperature"], 1.25)
        self.assertEqual(call["top_p"], 0.5)
        self.assertEqual(call["top_k"], 7)
        self.assertEqual(call["max_new_tokens"], 64)
        self.assertEqual(call["seed"], 99)

    def test_openais_own_name_for_the_response_length_is_taken_too(self):
        self.post(
            messages=[{"role": "user", "content": "hi"}], max_completion_tokens=32
        )
        self.assertEqual(self.manager.calls[0]["max_new_tokens"], 32)

    def test_a_locked_seed_is_used_and_reported(self):
        with settings.override(seed=1234, randomize_seed=False):
            body = self.post(messages=[{"role": "user", "content": "hi"}]).json()

        self.assertEqual(self.manager.calls[0]["seed"], 1234)
        self.assertEqual(body["chatlab"]["seed"], 1234)

    def test_the_seed_that_answered_is_always_reported(self):
        with settings.override(randomize_seed=True):
            body = self.post(messages=[{"role": "user", "content": "hi"}]).json()

        self.assertEqual(body["chatlab"]["seed"], self.manager.calls[0]["seed"])

    def test_every_turn_reaches_the_model(self):
        messages = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]

        self.post(messages=messages)

        self.assertEqual(self.manager.calls[0]["messages"], messages)
        self.assertEqual(self.manager.calls[0]["answer_prefill"], "")

    def test_a_trailing_assistant_message_is_the_prefill(self):
        self.post(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Well, actually"},
            ]
        )

        call = self.manager.calls[0]
        self.assertEqual(call["answer_prefill"], "Well, actually")
        self.assertEqual([turn["role"] for turn in call["messages"]], ["user"])

    def test_the_generation_is_bound_to_the_load_that_was_checked(self):
        # A load from the Models page can take the model lock between the
        # check and the first token. The runtime compares this under the lock
        # and refuses, rather than answering from the new weights while the
        # response names the old ones.
        self.post(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(self.manager.calls[0]["load_id"], "fake/model#1")

    def test_the_model_checked_and_the_load_bound_are_one_reading(self):
        # Two reads could straddle a load: the check would pass for the model
        # named and the request would bind to - and answer from - the next
        # one. The load ID names the model it is, so both come from it.
        self.manager.model_id = "fake/model"
        self.manager.load_id = "fake/model#7"

        body = self.post(
            model="fake/model", messages=[{"role": "user", "content": "hi"}]
        ).json()

        self.assertEqual(self.manager.calls[0]["load_id"], "fake/model#7")
        self.assertEqual(body["model"], "fake/model")

    def test_a_request_for_a_model_the_load_id_does_not_name_is_refused(self):
        self.manager.model_id = "fake/model"
        self.manager.load_id = "fake/other#2"

        response = self.post(
            model="fake/model", messages=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("fake/other", response.json()["error"]["message"])

    def test_a_load_that_landed_in_between_is_refused(self):
        self.use(
            Recorder(raises=model_runtime.ModelChanged("the model has changed"))
        )

        response = self.post(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["type"], "model_not_loaded")
        self.assertFalse(runtime.MANAGER.busy)

    def test_the_model_reported_is_the_one_that_answered(self):
        # Read under the model lock by the runtime, not from the manager
        # afterwards.
        self.manager.updates = [update("Hello")]
        self.manager.updates[0] = replace(
            self.manager.updates[0], model_id="fake/other"
        )

        body = self.post(messages=[{"role": "user", "content": "hi"}]).json()

        self.assertEqual(body["model"], "fake/other")

    def test_the_prompt_is_measured_only_when_it_is_asked_for(self):
        self.post(messages=[{"role": "user", "content": "hi"}])
        self.assertFalse(self.manager.calls[0]["analyze_prompt"])

        self.post(messages=[{"role": "user", "content": "hi"}], prompt_logprobs=True)
        self.assertTrue(self.manager.calls[1]["analyze_prompt"])


class CompletionShapeTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.manager.updates = [
            update(
                "<think>Hello</think> world",
                metrics=[metric(1, "Hello"), metric(2, " world", 0.25, rank=3)],
                prompt_ids=(1, 2, 3, 4),
            )
        ]

    def answer(self, **body):
        response = self.post(
            messages=[{"role": "user", "content": "hi"}], **body
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_the_reply_is_split_into_an_answer_and_its_reasoning(self):
        body = self.answer()

        message = body["choices"][0]["message"]
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "world")
        self.assertEqual(message["reasoning_content"], "Hello")

    def test_reasoning_markers_the_caller_supplied_stay_in_the_answer(self):
        # A prefill is the caller's own text, not syntax: a reply told to
        # begin "<think>quoted</think>" keeps that at the start of the answer
        # rather than having it read as a reasoning block.
        prefill = "<think>quoted</think>"
        self.manager.updates = [
            replace(
                update(f"{prefill} and on", metrics=[metric(1, prefill)]),
                literal_prefill_text=prefill,
            )
        ]

        body = self.answer()

        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], f"{prefill} and on")
        self.assertNotIn("reasoning_content", message)

    def test_a_reply_with_no_reasoning_carries_no_reasoning_field(self):
        self.manager.updates = [update("Hello")]

        self.assertNotIn("reasoning_content", self.answer()["choices"][0]["message"])

    def test_the_shape_is_the_one_an_openai_client_reads(self):
        body = self.answer()

        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "fake/model")
        self.assertEqual(body["choices"][0]["index"], 0)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            body["usage"],
            {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        )

    def test_a_response_that_ran_into_its_ceiling_says_length(self):
        body = self.answer(max_tokens=2)

        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_a_stop_token_on_the_last_allowed_position_is_still_a_stop(self):
        # The count cannot tell the two apart, and a client told a finished
        # answer was truncated would continue it.
        self.manager.updates = [
            replace(self.manager.updates[0], ends_on_stop_token=True)
        ]

        body = self.answer(max_tokens=2)

        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_the_measurements_are_left_out_until_they_are_asked_for(self):
        self.assertIsNone(self.answer()["choices"][0]["logprobs"])

    def test_every_token_carries_its_measurements(self):
        body = self.answer(logprobs=True)

        tokens = body["choices"][0]["logprobs"]["content"]
        self.assertEqual([entry["token"] for entry in tokens], ["Hello", " world"])
        self.assertAlmostEqual(tokens[0]["logprob"], -0.6931471805599453)
        self.assertEqual(tokens[0]["bytes"], list(b"Hello"))
        # ChatLab's own numbers ride along under its own key.
        chatlab = tokens[1]["chatlab"]
        self.assertEqual(chatlab["raw_rank"], 3)
        self.assertEqual(chatlab["raw_probability"], 0.25)
        self.assertEqual(chatlab["entropy_bits"], 2.0)
        self.assertEqual(chatlab["top1_margin"], 0.2)
        self.assertEqual(chatlab["token_id"], 2)
        # No alternatives unless they are asked for by name.
        self.assertEqual(tokens[0]["top_logprobs"], [])

    def test_the_alternatives_come_when_they_are_asked_for(self):
        body = self.answer(logprobs=True, top_logprobs=2)

        first = body["choices"][0]["logprobs"]["content"][0]
        self.assertEqual([entry["token"] for entry in first["top_logprobs"]], ["Hello", "Hi"])
        self.assertAlmostEqual(first["top_logprobs"][1]["logprob"], -1.2039728043259361)

    def test_an_impossible_token_is_floored_rather_than_infinite(self):
        # JSON has no negative infinity every client can read.
        self.manager.updates = [update("Hello", metrics=[metric(1, "Hello", 0.0)])]

        body = self.answer(logprobs=True)

        self.assertEqual(
            body["choices"][0]["logprobs"]["content"][0]["logprob"], api.LOGPROB_FLOOR
        )

    def test_the_headline_numbers_come_with_every_answer(self):
        body = self.answer()

        summary = body["chatlab"]["summary"]
        self.assertEqual(summary["token_count"], 2)
        self.assertEqual(summary["mean_surprise_bits"], 1.0)
        self.assertEqual(body["chatlab"]["device"], "CPU")
        self.assertEqual(body["chatlab"]["precision"], "full")

    def test_the_device_reported_is_the_one_that_answered(self):
        # Read with the load the request is bound to, not after the
        # generation: by then the model lock is free and a queued load can
        # have replaced the device and the precision both.
        class Swaps(Recorder):
            def generate(self, messages, **kwargs):
                self.device_name = "Apple Metal (MPS), 4-bit weights"
                self.precision = "4-bit"
                return super().generate(messages, **kwargs)

        self.use(Swaps(self.manager.updates))

        body = self.answer()

        self.assertEqual(body["chatlab"]["device"], "CPU")
        self.assertEqual(body["chatlab"]["precision"], "full")

    def test_the_prompt_measurements_come_when_they_are_asked_for(self):
        self.manager.updates = [
            update("Hello", prompt_ids=(1, 2)),
        ]
        self.manager.updates[0].prompt_metrics.append(metric(1, "hi"))

        body = self.answer(prompt_logprobs=True, logprobs=True)

        self.assertEqual(
            [entry["token"] for entry in body["chatlab"]["prompt_tokens"]], ["hi"]
        )

    def test_the_body_is_valid_json_a_client_can_read(self):
        # No NaN or Infinity, whatever the measurements hold.
        self.manager.updates = [update("Hello", metrics=[metric(1, "Hello", 0.0)])]
        response = self.post(
            messages=[{"role": "user", "content": "hi"}], logprobs=True
        )

        json.loads(response.text)  # strict: raises on NaN and Infinity
        self.assertNotIn("Infinity", response.text)


class StreamingTests(ApiTestCase):
    def frames(self, **body):
        response = self.post(
            messages=[{"role": "user", "content": "hi"}], stream=True, **body
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [
            line[len("data: ") :]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(event) for event in events[:-1]]

    def test_the_deltas_add_up_to_the_answer(self):
        self.manager.updates = [
            update("Hel", metrics=[metric(1, "Hel")]),
            update("Hello", metrics=[metric(1, "Hel"), metric(2, "lo")]),
            update("Hello world", metrics=[metric(1, "Hel"), metric(2, "lo"), metric(3, " world")]),
        ]

        frames = self.frames()

        self.assertEqual(frames[0]["choices"][0]["delta"], {"role": "assistant"})
        text = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        self.assertEqual(text, "Hello world")
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")

    def test_reasoning_streams_apart_from_the_answer(self):
        self.manager.updates = [
            update("<think>Hello", metrics=[metric(1, "Hello")]),
            update("<think>Hello</think> world", metrics=[metric(1, "Hello"), metric(2, " world")]),
        ]

        frames = self.frames()

        reasoning = "".join(
            frame["choices"][0]["delta"].get("reasoning_content", "")
            for frame in frames
        )
        answer = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        self.assertEqual(reasoning, "Hello")
        self.assertEqual(answer, "world")

    def test_the_last_frame_counts_the_tokens(self):
        frames = self.frames()

        last = frames[-1]
        self.assertEqual(last["usage"]["prompt_tokens"], 3)
        self.assertEqual(last["usage"]["completion_tokens"], 1)
        self.assertEqual(last["chatlab"]["summary"]["token_count"], 1)

    def test_each_frame_carries_only_its_new_tokens(self):
        self.manager.updates = [
            update("Hel", metrics=[metric(1, "Hel")]),
            update("Hello", metrics=[metric(1, "Hel"), metric(2, "lo")]),
        ]

        frames = self.frames(logprobs=True)

        measured = [
            [entry["token"] for entry in frame["choices"][0]["logprobs"]["content"]]
            for frame in frames
            if "logprobs" in frame["choices"][0]
        ]
        self.assertEqual(measured, [["Hel"], ["lo"]])

    def test_no_measurements_unless_they_are_asked_for(self):
        frames = self.frames()

        self.assertFalse(any("logprobs" in frame["choices"][0] for frame in frames))

    def test_a_marker_split_across_frames_never_leaks_into_the_answer(self):
        # A frame can end on the "<" of "</think>". A delta already sent
        # cannot be taken back, so the half marker is withheld until the rest
        # of it arrives.
        self.manager.updates = [
            update("<think>Hello", metrics=[metric(1, "Hello")]),
            update("<think>Hello<", metrics=[metric(1, "Hello"), metric(2, "<")]),
            update(
                "<think>Hello</think> wor",
                metrics=[metric(1, "Hello"), metric(2, "<"), metric(3, " wor")],
            ),
            update(
                "<think>Hello</think> world",
                metrics=[
                    metric(1, "Hello"),
                    metric(2, "<"),
                    metric(3, " wor"),
                    metric(4, "ld"),
                ],
            ),
        ]

        frames = self.frames()

        answer = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        reasoning = "".join(
            frame["choices"][0]["delta"].get("reasoning_content", "")
            for frame in frames
        )
        self.assertEqual(answer, "world")
        self.assertEqual(reasoning, "Hello")

    def test_half_a_character_is_withheld_until_it_is_whole(self):
        # A byte-level tokenizer splits "e" over two tokens, and a frame can
        # land between them: the decoder shows the bytes so far as U+FFFD and
        # the next frame replaces it. A delta already sent cannot be
        # replaced, so it waits.
        self.manager.updates = [
            update("caf\ufffd", metrics=[metric(1, "caf")]),
            update("caf\u00e9", metrics=[metric(1, "caf"), metric(2, "\u00e9")]),
        ]

        frames = self.frames()

        answer = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        self.assertEqual(answer, "caf\u00e9")
        self.assertNotIn("\ufffd", answer)

    def test_a_character_still_half_arrived_at_the_end_is_released(self):
        # Whatever the last frame withheld is emitted by the closing frame,
        # so the stream still adds up to the text the response holds.
        self.manager.updates = [update("caf\ufffd", metrics=[metric(1, "caf")])]

        frames = self.frames()

        answer = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        self.assertEqual(answer, "caf\ufffd")

    def test_the_assembled_stream_is_the_answer_the_whole_response_gives(self):
        # Whatever the last frame withheld is released in the closing frame,
        # so a client that concatenates the deltas has what a client that
        # asked for the whole response would have read.
        final = "<think>Reasoning</think> The answer <think>again</think> ends"
        self.manager.updates = [
            update(final[:length], metrics=[metric(1, final[:length])])
            for length in (8, 20, 30, len(final) - 1, len(final))
        ]

        frames = self.frames()

        assembled = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        reasoning = "".join(
            frame["choices"][0]["delta"].get("reasoning_content", "")
            for frame in frames
        )
        whole_reasoning, whole_answer, _closed = split_reasoning(final)
        self.assertEqual(assembled, whole_answer)
        self.assertEqual(reasoning, whole_reasoning)

    def test_a_stream_that_paid_for_prompt_measurements_receives_them(self):
        # The prompt is measured during the same pass that warms the cache,
        # so a streaming caller that asked for it has already paid.
        self.manager.updates[0].prompt_metrics.append(metric(1, "hi"))

        frames = self.frames(prompt_logprobs=True, logprobs=True)

        self.assertEqual(
            [entry["token"] for entry in frames[-1]["chatlab"]["prompt_tokens"]],
            ["hi"],
        )

    def test_a_stream_that_did_not_ask_gets_no_prompt_measurements(self):
        self.manager.updates[0].prompt_metrics.append(metric(1, "hi"))

        frames = self.frames()

        self.assertNotIn("prompt_tokens", frames[-1]["chatlab"])

    def test_the_slot_is_given_back_when_the_stream_ends(self):
        self.frames()
        self.assertFalse(self.manager.busy)

    def test_a_failure_part_way_through_is_said_in_the_stream(self):
        class Fails:
            def __init__(self, manager):
                self.manager = manager

            def __iter__(self):
                yield update("Hello")
                raise RuntimeError("the gpu fell over")

        self.manager.updates = Fails(self.manager)

        frames = self.frames()

        self.assertEqual(frames[-1]["error"]["type"], "server_error")
        self.assertFalse(self.manager.busy)


class FramesTests(ApiTestCase):
    """The frames of one generation, produced on a thread of their own."""

    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, api, "ABANDONED_AFTER_SECONDS", api.ABANDONED_AFTER_SECONDS)
        self.addCleanup(setattr, api, "FRAME_WAIT_SECONDS", api.FRAME_WAIT_SECONDS)

    def test_every_frame_is_produced_on_one_thread(self):
        # torch.inference_mode is thread-local and Starlette advances a
        # streaming iterator through its thread pool without promising the
        # same worker twice, so a generation that hopped threads would run
        # its forward passes with gradients enabled.
        threads = []

        def generate(messages, **kwargs):
            for text in ("a", "ab", "abc"):
                threads.append(threading.get_ident())
                yield update(text)

        self.manager.generate = generate

        response = self.post(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(threads), 3)
        self.assertEqual(len(set(threads)), 1)
        self.assertNotIn(threading.get_ident(), threads)

    def test_a_frame_is_copied_before_it_crosses(self):
        # The metrics list on an update belongs to the generator, which goes
        # on appending to it: a frame read after the next one was produced
        # would otherwise carry the later frame's tokens.
        live = [metric(1, "one")]

        def generate(messages, **kwargs):
            yield update("one", metrics=live)
            live.append(metric(2, "two"))
            yield update("one two", metrics=live)

        produced = api.Frames(generate([]))

        first = produced.first()
        rest = list(produced.rest())

        self.assertEqual(len(first.metrics), 1)
        self.assertEqual(len(rest[0].metrics), 2)

    def test_the_slot_is_given_back_by_the_thread_that_generated(self):
        def generate(messages, **kwargs):
            yield update("one")

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        produced.first()
        list(produced.rest())

        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)
        self.assertFalse(self.manager.busy)

    def test_a_stream_nobody_reads_is_abandoned_and_the_model_given_back(self):
        # A streaming response has no other way to learn that the client has
        # gone. The generator is closed from the thread that owns it, so its
        # own cleanup runs.
        closed = threading.Event()

        def generate(messages, **kwargs):
            try:
                for index in range(50):
                    yield update("x" * (index + 1))
            except GeneratorExit:
                closed.set()
                raise

        api.ABANDONED_AFTER_SECONDS = 0.05

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        produced.first()  # and then nothing reads the rest

        self.assertTrue(closed.wait(timeout=5))
        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)
        self.assertFalse(self.manager.busy)

    def test_a_generation_that_ends_with_nobody_reading_gives_the_model_back(self):
        # The reader left with the buffer full and the generation then
        # finished: the write that says so has to give up like any other, or
        # this thread parks for good holding the generation slot and every
        # later reply - here and in the interface - is refused as busy.
        api.ABANDONED_AFTER_SECONDS = 0.05

        def generate(messages, **kwargs):
            for index in range(api.FRAME_BUFFER + 1):
                yield update("x" * (index + 1))

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        produced.first()  # and then nothing reads the rest

        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)
        self.assertFalse(self.manager.busy)

    def test_a_failure_with_nobody_reading_gives_the_model_back_too(self):
        api.ABANDONED_AFTER_SECONDS = 0.05
        api.FRAME_WAIT_SECONDS = 0.05

        def generate(messages, **kwargs):
            for index in range(api.FRAME_BUFFER + 1):
                yield update("x" * (index + 1))
            raise RuntimeError("the gpu fell over")

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        produced.first()

        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)
        self.assertFalse(self.manager.busy)

        # The failure itself could not cross a full buffer, so a reader that
        # comes back is told the response was given up on rather than handed
        # a clean end to a truncated answer.
        with self.assertRaises(api.ApiError) as caught:
            list(produced.rest())
        self.assertEqual(caught.exception.kind, "abandoned")

    def test_a_reader_that_comes_back_to_an_abandoned_generation_is_told(self):
        # Abandonment leaves a full buffer and no last item in it. A reader
        # that comes back - a client that stalled for a minute and then
        # resumed - drains what is there and is then told the response was
        # given up on, rather than waiting for a frame nobody is going to
        # write, or being handed a short answer as though it were whole.
        api.ABANDONED_AFTER_SECONDS = 0.05
        api.FRAME_WAIT_SECONDS = 0.05

        def generate(messages, **kwargs):
            for index in range(api.FRAME_BUFFER + 3):
                yield update("x" * (index + 1))

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        produced.first()
        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)

        drained = []
        with self.assertRaises(api.ApiError) as caught:
            for frame in produced.rest():
                drained.append(frame)

        self.assertLessEqual(len(drained), api.FRAME_BUFFER)
        self.assertEqual(caught.exception.kind, "abandoned")

    def test_an_abandoned_stream_ends_in_an_error_rather_than_a_clean_stop(self):
        # A client that reads a truncated answer must not read it as
        # complete: an experiment would take the short response for the
        # model's whole answer.
        api.ABANDONED_AFTER_SECONDS = 0.05
        api.FRAME_WAIT_SECONDS = 0.05

        def generate(messages, **kwargs):
            for index in range(api.FRAME_BUFFER + 3):
                yield update("x" * (index + 1))

        self.manager.reserve_generation()
        produced = api.Frames(generate([]))
        first = produced.first()
        for _ in range(200):
            if not self.manager.busy:
                break
            time.sleep(0.01)

        events = list(
            api.stream_completion(
                chain([first], produced.rest()),
                "chatcmpl-x",
                0,
                "fake/model",
                {"max_new_tokens": 64, "seed": 1},
                False,
                0,
            )
        )
        payloads = [
            json.loads(event[len("data: ") :])
            for event in events
            if event.startswith("data: ") and "[DONE]" not in event
        ]

        self.assertEqual(payloads[-1]["error"]["type"], "abandoned")
        self.assertTrue(events[-1].startswith("data: [DONE]"))
        # And no frame claimed the response had finished normally.
        self.assertFalse(
            [
                payload
                for payload in payloads
                if payload.get("choices", [{}])[0].get("finish_reason")
            ]
        )

    def test_a_generation_that_produces_nothing_says_so(self):
        def generate(messages, **kwargs):
            return
            yield  # pragma: no cover - never reached

        produced = api.Frames(generate([]))

        with self.assertRaises(api.ApiError) as caught:
            produced.first()
        self.assertEqual(caught.exception.status, 500)


class ScoreTests(ApiTestCase):
    def test_the_scored_tokens_and_the_headline_numbers_come_back(self):
        scored = ScoredText(
            context_metrics=[metric(1, "before")],
            metrics=[metric(1, "Hello"), metric(2, " world")],
            seam_verified=False,
            chat_template_missing=True,
        )
        self.manager.score_text = lambda text, **kwargs: scored

        body = self.client.post(
            "/v1/chatlab/score", json={"text": "Hello world", "logprobs": True}
        ).json()

        self.assertEqual(body["object"], "chatlab.score")
        self.assertEqual([entry["token"] for entry in body["tokens"]], ["Hello", " world"])
        self.assertEqual([entry["token"] for entry in body["context_tokens"]], ["before"])
        self.assertEqual(body["summary"]["token_count"], 2)
        self.assertFalse(body["seam_verified"])
        self.assertTrue(body["chat_template_missing"])

    def test_what_the_scorer_refuses_is_a_bad_request(self):
        def refuse(text, **kwargs):
            raise ValueError("That is 9,000 tokens, above the 4,096 token limit")

        self.manager.score_text = refuse

        response = self.client.post("/v1/chatlab/score", json={"text": "long"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("9,000 tokens", response.json()["error"]["message"])

    def test_a_context_that_is_not_text_is_refused_rather_than_dropped(self):
        # A falsey non-string would otherwise become an empty context and the
        # text would be measured against nothing at all, which is a different
        # question than the one asked.
        self.manager.score_text = lambda text, **kwargs: self.fail("scored anyway")

        for context in (0, False, [], {}):
            with self.subTest(context=context):
                response = self.client.post(
                    "/v1/chatlab/score", json={"text": "hi", "context": context}
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "context must be a string", response.json()["error"]["message"]
                )

    def test_an_absent_context_is_no_context(self):
        seen = {}

        def score(text, **kwargs):
            seen.update(kwargs)
            return ScoredText(context_metrics=[], metrics=[metric(1, "hi")])

        self.manager.score_text = score

        for body in ({"text": "hi"}, {"text": "hi", "context": None}):
            with self.subTest(body=body):
                response = self.client.post("/v1/chatlab/score", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(seen["context"], "")

    def test_the_text_is_required(self):
        response = self.client.post("/v1/chatlab/score", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("text must be a string", response.json()["error"]["message"])

    def test_scoring_takes_the_generation_slot_while_it_runs(self):
        # Scoring and generating take the same model lock. Without the slot,
        # a score would wait out a whole response instead of saying the model
        # was busy, and would hold the lock a later response was refused for.
        held = []

        def score(text, **kwargs):
            held.append(runtime.MANAGER.busy)
            return ScoredText(context_metrics=[], metrics=[metric(1, "hi")])

        self.manager.score_text = score

        response = self.client.post("/v1/chatlab/score", json={"text": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(held, [True])
        self.assertFalse(self.manager.busy)

    def test_scoring_during_a_response_is_told_the_model_is_busy(self):
        self.manager.busy = True
        self.manager.score_text = lambda text, **kwargs: self.fail("scored anyway")

        response = self.client.post("/v1/chatlab/score", json={"text": "hi"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["type"], "model_busy")

    def test_the_slot_is_given_back_after_a_refused_score(self):
        def refuse(text, **kwargs):
            raise ValueError("nothing to score")

        self.manager.score_text = refuse

        self.client.post("/v1/chatlab/score", json={"text": "x"})

        self.assertFalse(self.manager.busy)

    def test_scoring_is_bound_to_the_load_it_was_checked_against(self):
        # Reserving the slot does not hold off a load, and score_text waits
        # for the model lock, so the runtime is told which load the request
        # was checked against.
        seen = {}

        def score(text, **kwargs):
            seen.update(kwargs)
            return ScoredText(context_metrics=[], metrics=[metric(1, "hi")])

        self.manager.score_text = score

        self.client.post("/v1/chatlab/score", json={"text": "hi"})

        self.assertEqual(seen["load_id"], "fake/model#1")

    def test_a_score_whose_model_changed_underneath_it_is_refused(self):
        def changed(text, **kwargs):
            raise model_runtime.ModelChanged("the model in memory is another")

        self.manager.score_text = changed

        response = self.client.post("/v1/chatlab/score", json={"text": "hi"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["type"], "model_not_loaded")
        self.assertFalse(self.manager.busy)

    def test_scoring_needs_a_model_too(self):
        self.manager.model_id = None

        response = self.client.post("/v1/chatlab/score", json={"text": "hi"})

        self.assertEqual(response.status_code, 409)


class RealModelTests(unittest.TestCase):
    """One pass through the real runtime, so the shapes are not only fakes."""

    def setUp(self):
        original = runtime.MANAGER
        runtime.MANAGER = loaded_manager([2, 3, THINK_EOS], THINK_PIECES, THINK_EOS)
        self.addCleanup(setattr, runtime, "MANAGER", original)
        app = FastAPI()
        api.attach(app)
        self.client = TestClient(app)

    def test_a_real_generation_answers_with_its_measurements(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 2,
                "seed": 7,
                "max_tokens": 8,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello world")
        tokens = body["choices"][0]["logprobs"]["content"]
        # The token that ended the response is measured like any other, and
        # is counted, even though it is not part of the text.
        self.assertEqual(
            [entry["token"] for entry in tokens], ["Hello", " world", "<eos>"]
        )
        for entry in tokens:
            self.assertLessEqual(entry["logprob"], 0.0)
            self.assertEqual(len(entry["top_logprobs"]), 2)
            self.assertGreaterEqual(entry["chatlab"]["raw_rank"], 1)
        self.assertEqual(body["usage"]["completion_tokens"], len(tokens))
        self.assertEqual(body["chatlab"]["seed"], 7)
        self.assertFalse(runtime.MANAGER.busy)

    def test_a_real_generation_streams_the_same_answer(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            self.assertEqual(response.status_code, 200)
            events = [
                line[len("data: ") :]
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        self.assertEqual(events[-1], "[DONE]")
        frames = [json.loads(event) for event in events[:-1]]
        text = "".join(
            frame["choices"][0]["delta"].get("content", "") for frame in frames
        )
        self.assertEqual(text, "Hello world")
        self.assertFalse(runtime.MANAGER.busy)

    def test_the_prefill_the_last_message_asks_for_is_replayed(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "max_tokens": 8,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["choices"][0]["message"]["content"].startswith("Hello"))
        self.assertGreaterEqual(body["chatlab"]["replayed_tokens"], 1)


class MountTests(unittest.TestCase):
    def test_the_interfaces_own_routes_do_not_shadow_the_api(self):
        # Gradio builds its application with named routes and no catch-all,
        # which is what lets both share one port. A release that changed
        # would be caught here rather than in front of somebody's script.
        import gradio as gr
        from gradio.routes import App

        with gr.Blocks() as demo:
            gr.Markdown("ChatLab")
        app = App.create_app(demo)
        api.attach(app)

        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/models").status_code, 200)
            self.assertEqual(client.get("/v1/chatlab/status").status_code, 200)
            # And the interface is still served from the same application.
            self.assertEqual(client.get("/").status_code, 200)
