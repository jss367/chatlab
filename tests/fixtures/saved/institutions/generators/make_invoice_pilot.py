"""Write the invoice-payment pilot bundle (chatlab-institutions-run-1, game-1) as ChatLab first read it.

Ran at 50db305 (#202, the commit that added the Institutions extension), with that commit's own test games and
bundle writer:

    git worktree add --detach /tmp/chatlab-at-50db305 50db305
    cd /tmp/chatlab-at-50db305
    PYTHONPATH=.:tests .venv/bin/python <this file> <repo>/tests/fixtures/saved/institutions

No model produced these games; they are the scripted democracy, vote and anarchy games from that commit's tests.
"""
import sys

from test_institutions import anarchy_game, democracy_game, vote_game, write_bundle

write_bundle(sys.argv[1], "invoice-pilot-run-1", [
    ("eval", democracy_game(), None),
    ("eval", vote_game(), None),
    ("eval", anarchy_game(), None),
    ("dev", anarchy_game("honest", 3, {"harm": 0.0, "usefulness": 1.0, "misaligned_expelled": 0,
                                       "honest_expelled": 0}), 1),
])
