from __future__ import annotations

import unittest

from locomo.sota import majority_vote, merge_contexts, plan_gap_fill, profile_defaults


class SotaHelpersTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        self.assertEqual(majority_vote(["Yes", "yes", "No"]), "Yes")
        self.assertEqual(majority_vote(["", "  "]), "")

    def test_profile_defaults(self) -> None:
        self.assertEqual(profile_defaults("product")["sc_samples"], 1)
        self.assertEqual(profile_defaults("sota")["sc_samples"], 5)
        self.assertTrue(profile_defaults("sota")["gap_fill"])

    def test_gap_fill_on_abstain(self) -> None:
        plan = plan_gap_fill(
            "Would Caroline pursue writing?",
            "Not mentioned in the conversation",
            {"text_context": "Caroline likes reading and wants to be a counselor"},
        )
        self.assertTrue(plan.needs_retry)

    def test_merge_contexts(self) -> None:
        merged = merge_contexts(
            {"source_passages": ["a"], "text_context": "a"},
            {"source_passages": ["b", "a"], "text_context": "b"},
        )
        self.assertEqual(merged["source_passages"], ["a", "b"])
        self.assertIn("b", merged["text_context"])


if __name__ == "__main__":
    unittest.main()
