"""The hangman tests' shared fixtures: finished games and a character-per-token model.

Kept here so the hangman test modules share them without importing each other.
"""

from types import SimpleNamespace

from chatlab.extensions.hangman.game import SYSTEM, finish_turn, new_game
from chatlab.model_runtime import GENERATING

STOP = 0


def game_of(*exchanges):
    """A finished game from (guess, reply) pairs."""
    game = new_game(SYSTEM)
    for guess, reply in exchanges:
        game["turns"].append(finish_turn(dict(guess=guess, text=reply, metrics=[], finish_reason="stop")))
    return game


class CharacterModel:
    """The model plumbing both fixtures share: one character per token, one generation at a time."""
    loaded = True
    model_id = "test/model"
    load_id = "first"
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))

    def claim_generation(self):
        if self.busy:
            return GENERATING
        self.busy = True
        return None

    def release_generation(self):
        self.busy = False

    def _stop_token_ids(self):
        return {STOP}

    def hidden_token_ids(self):
        return {STOP}
