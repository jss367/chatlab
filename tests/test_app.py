import unittest

import numpy as np

import app
from model_runtime import PROMPT_SCORE_LIMIT, ScoredText
from token_metrics import UNSCORED_BEYOND_LIMIT, build_metric, unscored_metric


class Selection:
    """The one attribute ``inspect_token`` reads off a Gradio select event."""

    def __init__(self, index: int):
        self.index = index


class InspectTokenTests(unittest.TestCase):
    def inspect(self, metric: dict):
        # The state pairs the metrics with the stamp of the strip they were
        # drawn for, and inspect_token() drops a click that misses it.
        return app.inspect_token(app.stamped([metric]), Selection(0))

    def test_the_opening_token_is_explained_as_unpredicted(self):
        detail, rows = self.inspect(
            unscored_metric(
                position=1, token_id=7, token_text="<s>", fallback_text="<s>"
            ).to_dict()
        )

        self.assertIn("Nothing came before this token", detail)
        self.assertEqual(rows, [])

    def test_a_token_beyond_the_window_is_explained_as_skipped(self):
        # It had plenty of predecessors; it was dropped by the prompt cap, and
        # calling it unpredicted would be a false explanation.
        detail, rows = self.inspect(
            unscored_metric(
                position=9,
                token_id=7,
                token_text=" the",
                fallback_text=" the",
                reason=UNSCORED_BEYOND_LIMIT,
            ).to_dict()
        )

        self.assertNotIn("Nothing came before", detail)
        self.assertIn(f"{PROMPT_SCORE_LIMIT:,}", detail)
        self.assertIn("skipped", detail)
        self.assertEqual(rows, [])


class StubManager:
    """A loaded manager that returns one scored token and nothing else."""

    loaded = True

    def __init__(self, seam_verified: bool = True, chat_template_missing: bool = False):
        self.seam_verified = seam_verified
        self.chat_template_missing = chat_template_missing

    def score_text(self, text, *, context="", use_chat_template=False):
        log_probs = np.log(np.array([0.75, 0.25]))
        metric = build_metric(
            position=1,
            token_id=0,
            token_text="a",
            fallback_text="a",
            raw_log_probabilities=log_probs,
            sampled_probabilities=np.exp(log_probs),
            decode_token=str,
        ).to_dict()
        return ScoredText(
            context_metrics=[],
            metrics=[metric],
            seam_verified=self.seam_verified,
            chat_template_missing=self.chat_template_missing,
        )


class ScoreStatusTests(unittest.TestCase):
    """What the Score text status line promises about the numbers above it."""

    def status(self, seam_verified: bool = True, chat_template_missing: bool = False) -> str:
        original = app.MANAGER
        app.MANAGER = StubManager(seam_verified, chat_template_missing)
        try:
            return app.score_text("foo", "bar", False, app.DEFAULT_COLOR_SCALE)[7]
        finally:
            app.MANAGER = original

    def test_a_verified_seam_is_reported_without_a_caveat(self):
        status = self.status(True)

        self.assertIn("Scored 1 tokens", status)
        self.assertNotIn("Approximate", status)
        self.assertNotIn(app.TEMPLATE_CAVEAT, status)

    def test_an_unverifiable_seam_is_called_out(self):
        # The split had to encode the context and the text apart, so the first
        # scored token may not be the one the passage produces. Showing the
        # numbers is still better than refusing to score, but showing them as
        # exact is not.
        status = self.status(False)

        self.assertIn("Scored 1 tokens", status)
        self.assertIn(app.SEAM_CAVEAT, status)
        self.assertNotIn(app.TEMPLATE_CAVEAT, status)

    def test_a_missing_chat_template_is_called_out(self):
        # The reader ticked a box the model could not honour. The numbers are
        # exact for what was scored, so they are shown; what they are exact
        # about is the part that has to be said out loud.
        status = self.status(chat_template_missing=True)

        self.assertIn("Scored 1 tokens", status)
        self.assertIn(app.TEMPLATE_CAVEAT, status)
        self.assertNotIn(app.SEAM_CAVEAT, status)

    def test_both_caveats_read_as_two_sentences(self):
        # A tokenizer can fail both ways at once, and the line has to stay
        # readable: the count and perplexity first, then which passage was
        # scored, then how sure its first token is.
        status = self.status(seam_verified=False, chat_template_missing=True)

        self.assertTrue(status.startswith("Scored 1 tokens. Perplexity"))
        self.assertTrue(status.endswith(f"{app.TEMPLATE_CAVEAT} {app.SEAM_CAVEAT}"))


if __name__ == "__main__":
    unittest.main()
