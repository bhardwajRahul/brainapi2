import os
import random
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.embeddings import Vector
from src.services.api.controllers import retrieve as retrieve_mod
from src.utils.vector_search import (
    ann_overfetch_k,
    stable_top_k_vectors,
    vector_rank_key,
)


class StableTopKVectorTests(unittest.TestCase):
    def test_equal_distance_uses_uuid_total_order(self):
        hits = [
            Vector(id="1", metadata={"uuid": "z-node"}, distance=0.1),
            Vector(id="2", metadata={"uuid": "a-node"}, distance=0.1),
            Vector(id="3", metadata={"uuid": "m-node"}, distance=0.05),
            Vector(id="4", metadata={"uuid": "b-node"}, distance=0.1),
        ]
        forward = stable_top_k_vectors(hits, 3)
        reverse = stable_top_k_vectors(list(reversed(hits)), 3)
        self.assertEqual(
            [v.metadata["uuid"] for v in forward],
            [v.metadata["uuid"] for v in reverse],
        )
        self.assertEqual(
            [v.metadata["uuid"] for v in forward],
            ["m-node", "a-node", "b-node"],
        )

    def test_boundary_k_stable_under_shuffled_equal_distance_pool(self):
        pool = [
            Vector(id=str(i), metadata={"uuid": f"u{i:02d}"}, distance=0.2)
            for i in range(12)
        ]
        expected = [f"u{i:02d}" for i in range(5)]
        for seed in range(30):
            rng = random.Random(seed)
            shuffled = list(pool)
            rng.shuffle(shuffled)
            got = [v.metadata["uuid"] for v in stable_top_k_vectors(shuffled, 5)]
            self.assertEqual(got, expected)

    def test_overfetch_is_at_least_4x(self):
        self.assertEqual(ann_overfetch_k(25), 100)
        self.assertEqual(ann_overfetch_k(5), 37)
        self.assertEqual(ann_overfetch_k(0), 0)

    def test_rank_key_falls_back_to_vector_id(self):
        v = Vector(id="fallback-id", metadata={}, distance=0.3)
        self.assertEqual(vector_rank_key(v), (0.3, "fallback-id"))


class SeedStabilizationTests(unittest.TestCase):
    def test_stabilize_seed_hits_dedupes_and_orders(self):
        seeds = [
            ("b", 0.2, "B"),
            ("a", 0.2, "A"),
            ("b", 0.1, "B-better"),
            ("c", 0.05, "C"),
            ("", 0.0, "drop"),
        ]
        out = retrieve_mod._stabilize_seed_hits(seeds)
        self.assertEqual(
            out,
            [("c", 0.05, "C"), ("b", 0.1, "B-better"), ("a", 0.2, "A")],
        )

    def test_quantize_distance_collapses_embed_float_noise(self):
        a = retrieve_mod._stabilize_seed_hits(
            [("u", 0.6657524665076711, "A"), ("v", 0.37019320459679717, "B")]
        )
        b = retrieve_mod._stabilize_seed_hits(
            [("v", 0.3702673417792153, "B"), ("u", 0.6657041188926127, "A")]
        )
        self.assertEqual([uid for uid, _, _ in a], [uid for uid, _, _ in b])
        self.assertEqual([d for _, d, _ in a], [d for _, d, _ in b])
        self.assertEqual(
            retrieve_mod._quantize_distance(0.6657524665076711),
            retrieve_mod._quantize_distance(0.6657041188926127),
        )
        self.assertEqual(
            retrieve_mod._quantize_distance(0.37019320459679717),
            retrieve_mod._quantize_distance(0.3702673417792153),
        )

    def test_equal_distance_concurrent_seed_paths_identical_uuid_lists(self):
        equal_hits = [
            Vector(
                id=f"n{i}",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": f"node-{chr(ord('z') - i)}", "name": f"N{i}"},
                distance=0.42,
            )
            for i in range(8)
        ]
        near = Vector(
            id="near",
            embeddings=[0.1, 0.2, 0.3],
            metadata={"uuid": "node-near", "name": "Near"},
            distance=0.01,
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )

        def run_once(order_seed: int):
            rng = random.Random(order_seed)
            node_hits = [near] + list(equal_hits)
            rng.shuffle(node_hits)
            rel_hits = [
                Vector(
                    id="r1",
                    embeddings=[0.1, 0.2, 0.3],
                    metadata={"node_ids": ["tail-b", "tail-a"]},
                    distance=0.42,
                ),
                Vector(
                    id="r0",
                    embeddings=[0.1, 0.2, 0.3],
                    metadata={"node_ids": ["tail-c", "tail-a"]},
                    distance=0.42,
                ),
            ]
            rng.shuffle(rel_hits)
            mock_vs = MagicMock()
            mock_vs.search_nodes.return_value = node_hits
            mock_vs.search_relationships.return_value = rel_hits
            with patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings):
                with patch.object(retrieve_mod, "vector_search", mock_vs):
                    return [
                        uid
                        for uid, _, _ in retrieve_mod._seed_nodes_for_text(
                            "query", "brain"
                        )
                    ]

        results = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(run_once, seed) for seed in range(24)]
            for fut in futs:
                results.append(fut.result())

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0][0], "node-near")
        self.assertEqual(
            results[0],
            sorted(results[0], key=lambda uid: (0.01 if uid == "node-near" else 0.42, uid)),
        )


class AdapterStableSearchTests(unittest.TestCase):
    def test_adapter_breaks_distance_ties_by_uuid(self):
        from src.adapters.embeddings import VectorStoreAdapter

        store = MagicMock()
        store.search_vectors.return_value = [
            Vector(id="1", metadata={"uuid": "z"}, distance=0.1),
            Vector(id="2", metadata={"uuid": "a"}, distance=0.1),
            Vector(id="3", metadata={"uuid": "m"}, distance=0.05),
        ]
        adapter = VectorStoreAdapter()
        adapter.add_client(store)
        results = adapter.search_vectors([1.0, 0.0], brain_id="b", store="nodes", k=2)
        self.assertEqual([v.metadata["uuid"] for v in results], ["m", "a"])


if __name__ == "__main__":
    unittest.main()
