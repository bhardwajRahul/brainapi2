import os
import unittest

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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


class SynergyMetricsTests(unittest.TestCase):
    def test_intra_list_diversity_and_coverage(self):
        from src.core.search.synergy_metrics import (
            intra_list_diversity,
            popularity_stratum,
            summarize_recommendation_list,
            type_coverage,
            unexpectedness_by_graph_distance,
        )

        labels = [["PRODUCT", "PHONE"], ["PRODUCT", "CASE"], ["CATEGORY"]]
        self.assertGreater(intra_list_diversity(labels), 0.0)
        self.assertEqual(type_coverage(labels), 4)
        self.assertGreater(unexpectedness_by_graph_distance([1, 2, 4]), 0.0)
        self.assertEqual(popularity_stratum(50), "head")
        self.assertEqual(popularity_stratum(1), "tail")
        summary = summarize_recommendation_list(labels, [1, 2, 3])
        self.assertEqual(summary["n"], 3)
        self.assertIn("intra_list_diversity", summary)


class RecommendMmrTests(unittest.TestCase):
    def test_mmr_prefers_type_diversity(self):
        from src.constants.kg import Node
        from src.core.search.recommend import _mmr_diversify

        a = Node.model_construct(uuid="a", name="A", labels=["PHONE"], properties={})
        b = Node.model_construct(uuid="b", name="B", labels=["PHONE"], properties={})
        c = Node.model_construct(uuid="c", name="C", labels=["CASE"], properties={})
        items = [
            (a, 1.0, [], "synergy"),
            (b, 0.99, [], "synergy"),
            (c, 0.5, [], "synergy"),
        ]
        out = _mmr_diversify(items, top_k=2, lambda_mult=0.5)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0].uuid, "a")
        self.assertEqual(out[1][0].uuid, "c")


if __name__ == "__main__":
    unittest.main()
