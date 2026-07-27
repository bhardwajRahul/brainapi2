import os
import sys
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.lib.postgresql.graph_store import _BrainGraph
    from src.lib.postgresql.networkx_client import NetworkXGraphClient

    DEPS_AVAILABLE = True
except ImportError:
    _BrainGraph = None
    NetworkXGraphClient = None
    DEPS_AVAILABLE = False


class _FakeStore:
    def __init__(self, brain):
        self._brain = brain

    def ensure_database(self, brain_id):
        return None

    def get_brain(self, brain_id):
        return self._brain


def _build_brain():
    brain = _BrainGraph("test-brain")
    brain.graph.add_node("A", uuid="A", name="Alice", labels=["Person"])
    brain.graph.add_node("B", uuid="B", name="Meeting", labels=["Event"])
    brain.graph.add_node("C", uuid="C", name="Bob", labels=["Person"])
    brain.graph.add_node("D", uuid="D", name="Elsewhere", labels=["Event"])
    brain.graph.add_edge(
        "A", "B", key="r1", uuid="r1", rel_type="CAUSED", flow_key="flow1"
    )
    brain.graph.add_edge(
        "B", "C", key="r2", uuid="r2", rel_type="LED_TO", flow_key="flow1"
    )
    brain.graph.add_edge(
        "B", "D", key="r3", uuid="r3", rel_type="UNRELATED", flow_key="other"
    )
    return brain


@unittest.skipUnless(DEPS_AVAILABLE, "postgresql/networkx dependencies unavailable")
class NetworkXEventCentricTests(unittest.TestCase):
    def _client(self):
        client = object.__new__(NetworkXGraphClient)
        client._store = _FakeStore(_build_brain())
        return client

    def test_event_path_records_helper_exists(self):
        self.assertTrue(hasattr(NetworkXGraphClient, "_event_path_records"))
        self.assertTrue(hasattr(NetworkXGraphClient, "_node_from_brain"))
        self.assertTrue(hasattr(NetworkXGraphClient, "_predicate_from_edge"))

    def test_get_event_centric_neighbors_returns_matching_flow_paths(self):
        client = self._client()
        results = client.get_event_centric_neighbors(["A"], "test-brain")
        self.assertEqual(len(results), 1)
        n, r1, m, r2, b = results[0]
        self.assertEqual(n.uuid, "A")
        self.assertEqual(m.uuid, "B")
        self.assertEqual(b.uuid, "C")
        self.assertEqual(r1.name, "CAUSED")
        self.assertEqual(r2.name, "LED_TO")
        self.assertEqual(r1.flow_key, "flow1")
        self.assertEqual(r2.flow_key, "flow1")

    def test_get_event_centric_neighbors_empty_for_unknown_node(self):
        client = self._client()
        self.assertEqual(client.get_event_centric_neighbors(["Z"], "test-brain"), [])

    def test_get_event_centric_neighbors_empty_input(self):
        client = self._client()
        self.assertEqual(client.get_event_centric_neighbors([], "test-brain"), [])

    def test_get_nexts_by_flow_key_returns_next_triple(self):
        client = self._client()
        res = client.get_nexts_by_flow_key(
            [{"predicate_uuid": "r1", "flow_key": "flow1"}], "test-brain"
        )
        self.assertIn("r1", res)
        triples = res["r1"]
        self.assertEqual(len(triples), 1)
        m, r2, b = triples[0]
        self.assertEqual(m.uuid, "B")
        self.assertEqual(r2.name, "LED_TO")
        self.assertEqual(b.uuid, "C")


if __name__ == "__main__":
    unittest.main()
