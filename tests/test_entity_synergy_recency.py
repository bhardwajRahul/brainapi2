import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.kg import Node
from src.core.search.entity_info import _recency_score


def _node(happened_at=None):
    return Node.model_construct(
        uuid="node-1",
        name="Dinner with Melanie",
        labels=["EVENT"],
        happened_at=happened_at,
    )


class RecencyScoreTests(unittest.TestCase):
    def test_missing_happened_at_is_neutral(self):
        self.assertEqual(_recency_score(_node()), 1.0)

    def test_stored_day_first_date_decays(self):
        score = _recency_score(_node("08/05/2023"))
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_locomo_timestamp_decays(self):
        self.assertEqual(
            _recency_score(_node("1:56 pm on 8 May, 2023")),
            _recency_score(_node("08/05/2023")),
        )

    def test_older_event_scores_lower_than_newer_event(self):
        older = _recency_score(_node("08/05/2013"))
        newer = _recency_score(_node("08/05/2023"))
        self.assertLess(older, newer)
        self.assertLess(newer, 1.0)

    def test_today_is_neutral(self):
        today = datetime.now().strftime("%d/%m/%Y")
        self.assertEqual(_recency_score(_node(today)), 1.0)

    def test_datetime_happened_at_is_supported(self):
        self.assertEqual(
            _recency_score(_node(datetime(2023, 5, 8))),
            _recency_score(_node("08/05/2023")),
        )

    def test_offset_aware_datetime_does_not_raise(self):
        aware = datetime(2023, 5, 8, tzinfo=timezone(timedelta(hours=2)))
        self.assertLess(_recency_score(_node(aware)), 1.0)

    def test_unparseable_happened_at_falls_back_to_neutral_and_logs(self):
        with patch("builtins.print") as printed:
            self.assertEqual(_recency_score(_node("sometime last summer")), 1.0)
        self.assertTrue(printed.called)
        self.assertIn("unparseable happened_at", printed.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
