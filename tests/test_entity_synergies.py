import os
import unittest
from unittest.mock import MagicMock, patch

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
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


class PolarityMatchesTests(unittest.TestCase):
    def test_same_matching_and_neutral(self):
        from src.core.search.entity_sibilings import polarity_matches

        self.assertTrue(polarity_matches("positive", "positive", "same"))
        self.assertTrue(polarity_matches(None, None, "same"))
        self.assertTrue(polarity_matches("neutral", None, "same"))
        self.assertFalse(polarity_matches("positive", "negative", "same"))

    def test_opposite_pos_neg_only(self):
        from src.core.search.entity_sibilings import polarity_matches

        self.assertTrue(polarity_matches("positive", "negative", "opposite"))
        self.assertTrue(polarity_matches("negative", "positive", "opposite"))
        self.assertFalse(polarity_matches("positive", "positive", "opposite"))
        self.assertFalse(polarity_matches(None, "negative", "opposite"))
        self.assertFalse(polarity_matches("neutral", "positive", "opposite"))


class SynergyTopKAndValidityTests(unittest.TestCase):
    def test_predicate_validity(self):
        from src.constants.kg import Predicate
        from src.core.search.entity_sibilings import _predicate_currently_valid

        ok = Predicate.model_construct(
            uuid="p1", name="MADE", properties={}, deprecated=False
        )
        self.assertTrue(_predicate_currently_valid(ok))
        bad = Predicate.model_construct(
            uuid="p2",
            name="MADE",
            properties={"invalid_at": "2024-01-01"},
            deprecated=False,
        )
        self.assertFalse(_predicate_currently_valid(bad))
        dep = Predicate.model_construct(
            uuid="p3", name="MADE", properties={}, deprecated=True
        )
        self.assertFalse(_predicate_currently_valid(dep))

    def test_top_k_bounds_results(self):
        from src.constants.kg import EntitySynergy, Node
        from src.core.search import entity_sibilings as mod

        target = Node.model_construct(
            uuid="t1",
            name="Widget",
            labels=["PRODUCT"],
            polarity="positive",
            properties={"v_id": "tv"},
        )
        candidates = [
            EntitySynergy(
                node=Node.model_construct(
                    uuid=f"c{i}",
                    name=f"C{i}",
                    labels=["PRODUCT"],
                    polarity="positive",
                    properties={},
                ),
                connected_by=[target],
                association_score=float(10 - i),
            )
            for i in range(5)
        ]

        retriever = mod.EntitySinergyRetriever("recsys-test")

        embed = MagicMock()
        embed.embeddings = [0.1, 0.2, 0.3]
        vs = MagicMock()
        vs.metadata = {"uuid": "t1"}

        with (
            patch.object(mod, "embeddings_adapter") as emb,
            patch.object(mod, "vector_search") as vsearch,
            patch.object(mod, "graph_adapter") as graph,
            patch.object(mod, "vector_store_adapter"),
        ):
            emb.embed_text.return_value = embed
            vsearch.search_nodes.return_value = [vs]
            graph.get_by_uuid.return_value = target
            graph.get_neighbors.return_value = {"t1": []}

            def fake_retrieve(*args, **kwargs):
                top_k = kwargs.get("top_k", 50)
                ranked = sorted(
                    candidates, key=lambda x: x.association_score, reverse=True
                )
                return target, ranked[:top_k], [], []

            with patch.object(
                retriever, "retrieve_sibilings", side_effect=fake_retrieve
            ):
                _, syn, _, _ = retriever.retrieve_sibilings(
                    "Widget", "same", top_k=2
                )
        self.assertEqual(len(syn), 2)
        self.assertEqual(syn[0].node.uuid, "c0")
        self.assertEqual(syn[1].node.uuid, "c1")


if __name__ == "__main__":
    unittest.main()
