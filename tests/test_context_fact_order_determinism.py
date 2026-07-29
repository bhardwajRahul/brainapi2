import asyncio
import os
import random
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

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

from src.constants.kg import Node, Predicate
from src.services.api.controllers import retrieve as retrieve_mod


def _node(uuid: str, name: str, *, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        properties={},
    )


def _pred(uuid: str, name: str, *, flow_key: str = ""):
    return Predicate(
        uuid=uuid,
        name=name,
        description=f"desc-{name}",
        flow_key=flow_key or None,
    )


def _candidate(
    *,
    score: float,
    r_uuid: str,
    r2_uuid: str,
    flow_key: str = "flow",
    n_uuid: str = "n",
    m_uuid: str = "m",
    b_uuid: str = "b",
    entity: str = "q",
):
    n = _node(n_uuid, n_uuid)
    r = _pred(r_uuid, "REL", flow_key=flow_key)
    m = _node(m_uuid, m_uuid, labels=["EVENT"])
    r2 = _pred(r2_uuid, "REL2", flow_key=flow_key)
    b = _node(b_uuid, b_uuid)
    return {
        "identified_entity": entity,
        "triple": (n, r, m, r2, b),
        "score": score,
        "key": (r_uuid, r2_uuid),
        "text": f"{n_uuid}|{r_uuid}|{m_uuid}|{r2_uuid}|{b_uuid}",
        "chunk_ids": [],
    }


def _ordered_ids(candidates):
    return [c["key"] for c in candidates]


class FactOrderDeterminismTests(unittest.TestCase):
    def test_equal_scores_use_stable_tiebreakers(self):
        a = _candidate(score=0.5, r_uuid="r-b", r2_uuid="r2-b", flow_key="f2")
        b = _candidate(score=0.5, r_uuid="r-a", r2_uuid="r2-a", flow_key="f1")
        c = _candidate(score=0.1, r_uuid="r-c", r2_uuid="r2-c", flow_key="f0")
        ranked_forward = retrieve_mod._dedupe_candidates([a, b, c])
        ranked_reverse = retrieve_mod._dedupe_candidates([c, b, a])
        self.assertEqual(_ordered_ids(ranked_forward), _ordered_ids(ranked_reverse))
        self.assertEqual(
            _ordered_ids(ranked_forward),
            [("r-c", "r2-c"), ("r-a", "r2-a"), ("r-b", "r2-b")],
        )

    def test_merge_variant_lists_is_order_stable_under_shuffled_arrival(self):
        v0 = [
            _candidate(score=0.2, r_uuid="r0", r2_uuid="x0", entity="v0"),
            _candidate(score=0.2, r_uuid="r1", r2_uuid="x1", entity="v0"),
        ]
        v1 = [
            _candidate(score=0.2, r_uuid="r2", r2_uuid="x2", entity="v1"),
            _candidate(score=0.2, r_uuid="r3", r2_uuid="x3", entity="v1"),
        ]
        expected = _ordered_ids(
            retrieve_mod._dedupe_candidates(
                retrieve_mod._merge_variant_candidate_lists([v0, v1])
            )
        )

        def race_once(seed: int):
            rng = random.Random(seed)
            buckets = [None, None]

            def worker(idx, payload):
                buckets[idx] = list(payload)
                rng.random()

            with ThreadPoolExecutor(max_workers=2) as pool:
                order = [0, 1]
                rng.shuffle(order)
                futs = [
                    pool.submit(worker, idx, [v0, v1][idx]) for idx in order
                ]
                for fut in futs:
                    fut.result()
            merged = retrieve_mod._merge_variant_candidate_lists(
                [buckets[0], buckets[1]]
            )
            return _ordered_ids(retrieve_mod._dedupe_candidates(merged))

        results = [race_once(seed) for seed in range(40)]
        self.assertTrue(all(r == expected for r in results))
        self.assertEqual(
            expected,
            [("r0", "x0"), ("r1", "x1"), ("r2", "x2"), ("r3", "x3")],
        )

    def test_nondeterministic_extend_diverges_without_ordered_merge(self):
        a = _candidate(score=0.2, r_uuid="r0", r2_uuid="x0", entity="v0")
        b = _candidate(score=0.2, r_uuid="r1", r2_uuid="x1", entity="v1")
        score_only_ab = [c["key"] for c in sorted([a, b], key=lambda c: c["score"])]
        score_only_ba = [c["key"] for c in sorted([b, a], key=lambda c: c["score"])]
        self.assertNotEqual(score_only_ab, score_only_ba)

        fixed = {
            tuple(
                _ordered_ids(
                    retrieve_mod._dedupe_candidates(
                        retrieve_mod._merge_variant_candidate_lists(lists)
                    )
                )
            )
            for lists in ([[a], [b]], [[b], [a]])
        }
        self.assertEqual(len(fixed), 1)
        self.assertEqual(next(iter(fixed)), (("r0", "x0"), ("r1", "x1")))


class ContextFactOrderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_identical_get_context_calls_same_triple_order(self):
        import contextlib
        from unittest.mock import MagicMock, patch

        from src.constants.embeddings import Vector
        from src.services.api.constants.requests import GetContextRequestBody
        from src.utils.nlp.ner import ExtractElementsResponse

        facts = []
        for i in range(6):
            facts.append(
                (
                    _node(f"n{i}", f"N{i}"),
                    _pred(f"r{i}", "REL", flow_key=f"f{i % 3}"),
                    _node(f"e{i}", f"E{i}", labels=["EVENT"]),
                    _pred(f"r2{i}", "REL2", flow_key=f"f{i % 3}"),
                    _node(f"b{i}", f"B{i}"),
                )
            )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n0",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "n0", "name": "N0"},
                distance=0.5,
            ),
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "n1", "name": "N1"},
                distance=0.5,
            ),
        ]
        mock_vs.search_relationships.return_value = []
        mock_vs.search_data.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = facts
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.get_text_chunks_by_ids.return_value = ([], None)
        mock_data.search.return_value = MagicMock(text_chunks=[])

        async def _once():
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings)
                )
                stack.enter_context(patch.object(retrieve_mod, "vector_search", mock_vs))
                stack.enter_context(
                    patch.object(retrieve_mod, "graph_adapter", mock_graph)
                )
                stack.enter_context(patch.object(retrieve_mod, "data_adapter", mock_data))
                stack.enter_context(
                    patch.object(
                        retrieve_mod._entity_extractor,
                        "extract_elements",
                        return_value=ExtractElementsResponse(
                            tokens=[
                                {"text": "Alice", "label": "PERSON"},
                                {"text": "Bob", "label": "PERSON"},
                            ],
                            noun_chunks=["Alice", "Bob"],
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(retrieve_mod, "_retrieve_passages", return_value=[])
                )
                return await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="Alice and Bob events",
                        brain_id="brain-a",
                        max_facts=4,
                        use_ppr=False,
                    )
                )

        first = await _once()
        second = await _once()
        first_ids = [(t.triple[1].uuid, t.triple[3].uuid) for t in first.triples]
        second_ids = [(t.triple[1].uuid, t.triple[3].uuid) for t in second.triples]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 4)

        parallel = await asyncio.gather(*[_once() for _ in range(12)])
        id_sets = [
            tuple((t.triple[1].uuid, t.triple[3].uuid) for t in resp.triples)
            for resp in parallel
        ]
        self.assertEqual(len(set(id_sets)), 1)
        self.assertEqual(id_sets[0], tuple(first_ids))


if __name__ == "__main__":
    unittest.main()
