import unittest

import app
from model_runtime import PROMPT_SCORE_LIMIT
from token_metrics import UNSCORED_BEYOND_LIMIT, unscored_metric


class Selection:
    """The one attribute ``inspect_token`` reads off a Gradio select event."""

    def __init__(self, index: int):
        self.index = index


class InspectTokenTests(unittest.TestCase):
    def inspect(self, metric: dict):
        return app.inspect_token([metric], Selection(0))

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


if __name__ == "__main__":
    unittest.main()
