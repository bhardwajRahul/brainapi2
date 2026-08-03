from __future__ import annotations

import unittest

from locomo.sota import (
    complete_education_fields,
    is_hard_abstain,
    is_soft_abstain,
    majority_vote,
    majority_vote_count,
    majority_vote_traits,
    merge_contexts,
    plan_gap_fill,
    profile_defaults,
    rank_image_cues,
    strip_leading_abstain,
    symbols_missing_from_answer,
)


class SotaHelpersTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        self.assertEqual(majority_vote(["Yes", "yes", "No"]), "Yes")
        self.assertEqual(majority_vote(["", "  "]), "")

    def test_majority_vote_demotes_hard_abstain(self) -> None:
        self.assertEqual(
            majority_vote(
                [
                    "Not mentioned in the conversation",
                    "Not mentioned in the conversation.",
                    "No; she refers to her husband and kids.",
                    "Not mentioned in the conversation",
                    "No, she does not self-identify as LGBTQ.",
                ]
            ),
            "No; she refers to her husband and kids.",
        )

    def test_majority_vote_prefers_richer_list(self) -> None:
        self.assertEqual(
            majority_vote(
                [
                    "Charlotte's Web",
                    "Charlotte's Web, Nothing is Impossible",
                ]
            ),
            "Charlotte's Web, Nothing is Impossible",
        )

    def test_is_hard_abstain(self) -> None:
        self.assertTrue(is_hard_abstain("Not mentioned in the conversation"))
        self.assertFalse(
            is_hard_abstain(
                "No evidence she is LGBTQ; she is a supportive ally."
            )
        )

    def test_is_soft_abstain_hybrid(self) -> None:
        self.assertTrue(
            is_soft_abstain(
                "Not mentioned in the conversation. Caroline wants counseling, "
                "not writing."
            )
        )

    def test_strip_leading_abstain(self) -> None:
        stripped = strip_leading_abstain(
            "Not mentioned in the conversation. Caroline wants to be a counselor, "
            "not a writer."
        )
        self.assertTrue(stripped.startswith("Caroline wants"))
        self.assertEqual(
            majority_vote(
                [
                    "Not mentioned in the conversation. No; she wants counseling.",
                    "Not mentioned in the conversation",
                ]
            ),
            "No; she wants counseling.",
        )

    def test_profile_defaults(self) -> None:
        import os

        saved = {
            k: os.environ.pop(k)
            for k in ("BENCH_SC_SAMPLES", "BENCH_SC_TEMPERATURE", "BENCH_GAP_FILL")
            if k in os.environ
        }
        try:
            self.assertEqual(profile_defaults("product")["sc_samples"], 1)
            self.assertEqual(profile_defaults("sota")["sc_samples"], 5)
            self.assertTrue(profile_defaults("sota")["gap_fill"])
        finally:
            os.environ.update(saved)

    def test_gap_fill_on_abstain(self) -> None:
        plan = plan_gap_fill(
            "Would Caroline pursue writing?",
            "Not mentioned in the conversation",
            {"text_context": "Caroline likes reading and wants to be a counselor"},
        )
        self.assertTrue(plan.needs_retry)

    def test_gap_fill_on_soft_abstain(self) -> None:
        plan = plan_gap_fill(
            "Would Caroline be considered religious?",
            "There is no evidence that Caroline practices any religion.",
            {"text_context": "grandmother necklace means love faith and strength"},
        )
        self.assertTrue(plan.needs_retry)

    def test_gap_fill_on_undercount(self) -> None:
        plan = plan_gap_fill(
            "How many times has Melanie gone to the beach in 2023?",
            "Melanie went to the beach once in 2023.",
            {"text_context": "beach camping beach recently"},
        )
        self.assertTrue(plan.needs_retry)
        self.assertIn("all occasions", plan.reformulated_query)

    def test_gap_fill_on_people_undercount(self) -> None:
        plan = plan_gap_fill(
            "How many children does Melanie have?",
            "2",
            {"text_context": "my son got into an accident daughter's birthday younger kids"},
        )
        self.assertTrue(plan.needs_retry)
        self.assertIn("named children", plan.reformulated_query)

    def test_merge_contexts(self) -> None:
        merged = merge_contexts(
            {"source_passages": ["a"], "text_context": "a", "image_cues": ["q1"]},
            {"source_passages": ["b", "a"], "text_context": "b", "image_cues": ["q2", "q1"]},
        )
        self.assertEqual(merged["source_passages"], ["a", "b"])
        self.assertIn("b", merged["text_context"])
        self.assertEqual(merged["image_cues"], ["q1", "q2"])

    def test_rank_image_cues_pins_symbols(self) -> None:
        ranked = rank_image_cues(
            "What symbols are important to Caroline?",
            [
                "x: [image query: pottery bowl flower]",
                "y: [image query: pendant transgender symbol]",
                "z: [image query: rainbow flag pride march]",
            ],
        )
        self.assertLess(ranked.index(next(c for c in ranked if c.startswith("y:"))), 2)
        self.assertEqual(ranked[-1].split(":", 1)[0].strip(), "x")

    def test_majority_vote_traits(self) -> None:
        voted = majority_vote_traits(
            [
                "Courageous, empathetic, kind, caring, strong",
                "thoughtful, authentic, driven",
                "She's so real and dedicated and thoughtful",
                "kind and driven",
                "authentic and thoughtful",
            ]
        )
        self.assertIsNotNone(voted)
        assert voted is not None
        low = voted.lower()
        self.assertIn("thoughtful", low)
        self.assertTrue("authentic" in low or "driven" in low)

    def test_majority_vote_count(self) -> None:
        self.assertEqual(majority_vote_count(["2", "3", "three", "2", "3"]), "3")

    def test_complete_education_fields(self) -> None:
        out = complete_education_fields(
            "What fields would Caroline be likely to pursue in her educaton?",
            "Counseling and mental health with a certification path",
        )
        self.assertIn("Psychology", out)
        self.assertIn("counseling", out.lower())

    def test_symbols_missing_from_answer(self) -> None:
        missing = symbols_missing_from_answer(
            "Rainbow flag, pride umbrella",
            [
                "Caroline (D4:1): [image query: pendant transgender symbol]",
                "Caroline (D8:17): [image query: rainbow flag pride march]",
            ],
        )
        self.assertIn("transgender symbol", missing)

    def test_gap_fill_on_image_symbols(self) -> None:
        plan = plan_gap_fill(
            "What symbols are important to Caroline?",
            "Rainbow flag",
            {
                "image_cues": [
                    "Caroline (D4:1): [image query: pendant transgender symbol]",
                ],
                "text_context": "rainbow flag mural",
            },
        )
        self.assertTrue(plan.needs_retry)
        self.assertIn("image-query", plan.reformulated_query)


if __name__ == "__main__":
    unittest.main()
