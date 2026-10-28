"""Generate the first neuron-steering fixture with its introducing change.

Generating revision: the neuron-steering introduction, committed with this
generator and fixture. Run from that revision's checkout (not a later writer):

PYTHONPATH=. .venv/bin/python tests/fixtures/saved/steering/generators/make_neuron_conversation.py /tmp/neuron-conversation.json

Copy the new result to steering/03-neuron-conversation.json once. Never replace
that frozen file when changing the reader or writer in a later revision.
"""

from pathlib import Path
import sys

from chatlab.conversation import make_turn, to_json
from chatlab.steering import NEURON_FORMAT, normalize


def main():
    turns = []
    for operation, strength, enabled in (("add", -2.5, True), ("scale", 0, True),
                                         ("set", 0, True), ("set", 4, False)):
        value = normalize(dict(format=NEURON_FORMAT, model_id="test/tiny", layer=0,
                               neuron=3, operation=operation, strength=strength, enabled=enabled))
        turns.append(make_turn("user", f"Try {operation} at {strength}."))
        turns.append(dict(make_turn("assistant", "Recorded neuron settings."), steering=value))
    current = normalize(dict(value, layer=1, operation="scale", strength=0, enabled=True))
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        stream.write(to_json(turns, system_prompt="Compare neuron interventions.", steering=current))


if __name__ == "__main__":
    main()
