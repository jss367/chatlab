"""Writes batch-game-with-probes.json: one game a hangman batch saved, with reveal-probe answers on its responses.

Format marker: chatlab-hangman-1
Era: batch-played games, which add ``trial`` to the game and ``probes`` to each response (load() reads every
probe's word again from its text).
Generating commit: 5c32174 Play Hangman trial files as batches, with an optional reveal probe (#157)

Commands used:
    git worktree add --detach /tmp/chatlab-at-5c32174 5c32174
    cd /tmp/chatlab-at-5c32174 && PYTHONPATH=. /Users/julius/conductor/workspaces/chatlab/chengdu-v1/.venv/bin/python \
        <repo>/tests/fixtures/saved/hangman/generators/batch-game-with-probes.py \
        <repo>/tests/fixtures/saved/hangman/batch-game-with-probes.json

The batch is played by that commit's own read_trials and run_trials against a subclass of its test suite's
``Host`` that holds no word, so each board comes from a script and each probe answer from a list. The
batch's own save() writes the game; the written file is copied byte for byte.

What the game holds: four boards (the fourth moves A off position 3 and shows an O nobody guessed), three
probes after each, and a closing "I give up" turn that reveals CRATE. The probe answers are readable
words that fit, a word that does not fit, an answer with no word in it, an answer cut off by the probe's
token limit on its Word: line, which reads as no word, and one cut off after its first line, which reads.

Ids, timestamps, the batch directory name and response times are pinned by replacing the game module's
uuid4 and now and the batch module's clock; everything written still comes from the old writer.
"""
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import uuid

sys.path.insert(1, "tests")

from extension_api import ModelService  # noqa: E402
from extensions.hangman import batch as batch_module, game as game_module  # noqa: E402
from extensions.hangman.batch import BatchControl, read_trials, run_trials  # noqa: E402
from test_hangman_batch import STOP, WORDS, Host  # noqa: E402


class ScriptedHost(Host):
    """The suite's word-less Host, answering probes with raw text and honoring the probe's token limit."""

    def reply(self, messages, prefill):
        if prefill:
            return self.names.pop(0)
        return super().reply(messages, prefill)

    def generate(self, messages, **options):
        self.calls.append(dict(messages=messages, **options))
        prefill = options.get("answer_prefill", "")
        sampled = [ord(c) for c in self.reply(messages, prefill)] + [STOP]
        if prefill and len(sampled) > options["max_new_tokens"]:
            sampled = sampled[:options["max_new_tokens"]]
        ids = [ord(c) for c in prefill] + sampled
        metrics = []
        for position, token in enumerate(ids, 1):
            metrics.append(dict(token_id=token, text=chr(token) if token else "",
                                display_text=chr(token) if token else "⏹", segment="response", position=position,
                                scored=False, top_candidates=[dict(token_id=ord("e"), text="e", probability=.25)]))
            yield SimpleNamespace(text="".join(chr(m["token_id"]) for m in metrics if m["token_id"]),
                                  metrics=[dict(m) for m in metrics], forced_prefix_tokens=len(prefill),
                                  reasoning_prefilled=False)


def main(output):
    ids = (uuid.UUID(int=0x5c32174 * 1000 + n) for n in itertools.count(1))
    game_module.uuid4 = lambda: next(ids)
    stamps = (f"2026-09-23T18:{minute:02d}:00+00:00" for minute in itertools.count())
    game_module.now = lambda: next(stamps)
    clock = itertools.count(1_758_675_600, 2)
    batch_module.time = SimpleNamespace(time=lambda: float(next(clock)), monotonic=lambda: 0.0,
                                        strftime=lambda pattern: "20260923-180000" if "%Y%m%d" in pattern
                                        else "2026-09-23T18:00:00-0700")
    batch_module.dictionary = game_module.dictionary = lambda: WORDS

    host = ScriptedHost(
        boards=["_ _ _ _ _", "_ _ _ _ E", "_ _ A _ E", "_ _ O T E"],
        names=[" crane", " slate", " crane",            # after response 1: all fit five blanks
               " crate", " stone", " ???",               # after response 2: ??? is no word
               " grape", " crate", " stonewall",         # after response 3: STONEWALL is cut off at 8 tokens
               " crate", " crate", " slate\nWord",       # after response 4: none fits the moved A; SLATE is
                                                         # cut off past its first line, so read
               "crate"])                                 # the answer to "I give up"
    trials = dict(format="chatlab-hangman-trials-1", title="Probe pilot",
                  defaults=dict(temperature=0.8, max_new_tokens=128),
                  trials=[dict(id="held-word", label="Fixed guesses, probed", seed=11, guesser=["e", "a", "t"],
                               probe=dict(samples=3, max_new_tokens=8, temperature=0.5))])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "trials.json"
        path.write_text(json.dumps(trials))
        data = read_trials(path)
        list(run_trials(data, ModelService(lambda: host), root, BatchControl(), source="trials.json"))
        manifest = json.loads((root / "batches" / "20260923-180000-probe-pilot" / "batch.json").read_text())
        assert manifest["status"] == "finished", manifest
        row, = manifest["results"]
        assert row["outcome"] == "unfinished", row
        shutil.copyfile(root / "batches" / "20260923-180000-probe-pilot" / row["game_file"], output)


if __name__ == "__main__":
    main(sys.argv[1])
