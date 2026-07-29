from __future__ import annotations

import unittest

from src.core.saving.topic_hyperedges import (
    cluster_sessions_into_topics,
    lexical_topic_score,
    rrf_fuse,
    select_topics_and_sessions,
    session_entity_sets,
)


class TopicHyperedgesTests(unittest.TestCase):
    def test_session_entity_sets_and_cluster(self) -> None:
        rows = [
            ("session_1", "e1", "pottery"),
            ("session_1", "e2", "Melanie"),
            ("session_2", "e3", "pottery workshop"),
            ("session_2", "e2", "Melanie"),
            ("session_3", "e4", "counseling"),
            ("session_3", "e5", "Caroline"),
        ]
        entities = session_entity_sets(rows)
        topics = cluster_sessions_into_topics(entities, merge_threshold=0.2)
        self.assertGreaterEqual(len(topics), 2)
        session_ids = {t.session_id for t in topics}
        self.assertEqual(session_ids, {"session_1", "session_2", "session_3"})
        coverage = len(session_ids) / 3
        self.assertGreaterEqual(coverage, 0.8)

    def test_select_topics_boosts_query_overlap(self) -> None:
        from src.core.saving.topic_hyperedges import TopicSession

        memberships = [
            TopicSession("topic:a", "pottery camping", "session_1"),
            TopicSession("topic:a", "pottery camping", "session_2"),
            TopicSession("topic:b", "counseling career", "session_3"),
        ]
        topics, sessions = select_topics_and_sessions(
            "What pottery workshops did Melanie attend?",
            memberships,
            seed_sessions=["session_1"],
            k_topics=2,
            k_sessions=5,
        )
        self.assertTrue(topics)
        self.assertIn("session_1", sessions)
        self.assertGreater(lexical_topic_score("pottery", "pottery camping"), 0.0)

    def test_rrf_fuse(self) -> None:
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
        self.assertEqual(fused[0], "a")
        self.assertIn("b", fused[:2])


if __name__ == "__main__":
    unittest.main()
