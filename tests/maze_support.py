"""The maze extension's test fixtures: a small maze and a scripted manager.

The manager replays one scripted reply per call over a UTF-8 byte
vocabulary, so a tool call the test writes is the text the runner reads back.
Kept here so the maze test modules share it without importing each other.
"""

import copy
import json
from types import SimpleNamespace

from chatlab.extension_api import ModelService, SteeringError
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.model_loading import LoadedModel
from chatlab.model_runtime import GENERATING
from chatlab.token_metrics import unscored_metric


CONFIG = dict(supplied_moves=0, interrupt_after=0, interruption_text="Distracted", prefix_tokens=2,
              temperature=.7, sampling_seed=99, per_turn_tokens=100, token_budget=300, attempt_budget=10)
MAZE = Maze(("...", "##.", "..."), (0, 0), (0, 2))
# The Waypoint & steering controls as a fresh pane has them: no waypoint, steering off.
NO_CHECKPOINT = ('', None, 1.0, 0, 'off', '', 3, 1)


class Manager:
    loaded = True
    model_id = "test/model"
    load_id = "test/model#1"
    reasoning_prefilled = False
    tokenizer = SimpleNamespace(encode=lambda s, **kw: list(s.encode()), decode=lambda ids, **kw: bytes(ids).decode())

    def __init__(self, replies):
        self.replies = iter(replies)
        self.busy = False
        self.calls = []

    def open_session(self):
        return ModelService(lambda: self).open_session()

    def loaded_model(self):
        # Published as one reading, the way a finished load publishes it.
        return LoadedModel(self.model_id, "test-device", "full", self.load_id)

    def decode(self, ids):
        return ModelService(lambda: self).decode(ids)

    def prompt_text(self, messages, tools=None):
        return ModelService(lambda: self).prompt_text(messages, tools)

    def loaded_model_id(self):
        return ModelService(lambda: self).loaded_model_id()

    def _prompt_token_ids(self, messages, tools=None):
        # Stands in for a chat template: the fixture's vocabulary is UTF-8
        # bytes, so a rendering the reader can read round-trips through decode.
        rendered = "".join([f"<tools>{json.dumps(tools)}</tools>" if tools else ""]
                           + [f"<{m['role']}>{m['content']}" for m in messages] + ["<assistant>"])
        return list(rendered.encode()), False

    def claim_generation(self):
        if self.busy:
            return GENERATING
        self.busy = True
        return None

    def reserve_generation(self):
        return self.claim_generation() is None

    def release_generation(self):
        self.busy = False

    def _stop_token_ids(self):
        return {0}

    def hidden_token_ids(self):
        # The stop token is a special: generate() leaves it out of the response
        # text the way the runtime's decoder does, and records it in metrics.
        return {0}

    def encode_replacement(self, kept_ids, text, **kwargs):
        # This fixture encodes independent UTF-8 bytes; context-sensitive
        # behavior is exercised separately through the real runtime encoder.
        return list(text.encode())

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text, ids = next(self.replies)
        prefix = kwargs["forced_ids"]
        metrics = [{"token_id": t} for t in prefix + ids]
        # Prompt IDs come from the template, as the runtime's do, so a reader
        # decoding them back sees the tools and history the turn was given.
        prompt_ids, _ = self._prompt_token_ids(messages, kwargs.get("tools"))
        yield SimpleNamespace(text=self.tokenizer.decode(prefix) + text, metrics=metrics, prompt_ids=prompt_ids,
                              forced_prefix_tokens=len(prefix), reasoning_prefilled=self.reasoning_prefilled,
                              load_id=self.load_id, model_id=self.model_id)


def scored(generate):
    """Add the display metrics and alternatives the token strip and edit panel read."""
    def reply(*args, **kwargs):
        for frame in generate(*args, **kwargs):
            for position, metric in enumerate(frame.metrics):
                metric.update(unscored_metric(position=position, token_id=metric['token_id'],
                                              token_text=chr(metric['token_id']), fallback_text='',
                                              segment='response').to_dict())
                metric['top_candidates'] = [{'token_id': 120, 'text': 'x', 'probability': .1}]
            yield frame
    return reply


VECTOR = {"format": "chatlab-steering-1", "model_id": "test/model", "layer": 3,
          "vector": [1.0, -2.0, 0.5], "strength": 4.0, "enabled": True}


class SteeringManager(Manager):
    """The maze fixture manager, answering the steering pre-flight."""

    def __init__(self, replies, refuse=None):
        super().__init__(replies)
        self.refuse = refuse
        self.checked = []

    def check_steering(self, value):
        self.checked.append(copy.deepcopy(value))
        if self.refuse:
            raise SteeringError(self.refuse)


MAZE_ID = MAZE.tool_id()


def call(direction, message=None, maze_id=MAZE_ID):
    args = {"maze_id": maze_id, "direction": direction}
    if message is not None:
        args["message"] = message
    text = "<tool_call>\n" + json.dumps({"name": "move", "arguments": args}) + "\n</tool_call>"
    return text, list(text.encode()) + [0]
