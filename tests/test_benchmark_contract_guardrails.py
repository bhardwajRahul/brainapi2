import os
import unittest
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkContractGuardrailsTests(unittest.TestCase):
    """Ensure LoCoMo/BEAM/LME clients stay on frozen HTTP contracts."""

    def _client_sources(self) -> list[Path]:
        paths = []
        for suite in ("locomo", "beam", "longmemeval"):
            client = ROOT / "benchmarks" / suite / "client.py"
            if client.exists():
                paths.append(client)
        return paths

    def test_bench_clients_use_ingest_and_context_only(self):
        clients = self._client_sources()
        self.assertGreaterEqual(len(clients), 1)
        forbidden = (
            "/ingest/structured",
            "/retrieve/entity/synergies",
            "/retrieve/recommend",
            "/recsys/",
        )
        for path in clients:
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(
                    needle,
                    text,
                    msg=f"{path} must not call {needle} (benchmark isolation)",
                )

    def test_bench_clients_call_core_ingest(self):
        for path in self._client_sources():
            text = path.read_text(encoding="utf-8")
            self.assertTrue(
                "/ingest/" in text or "ingest" in text.lower(),
                msg=f"{path} should still target core ingest",
            )

    def test_search_client_scores_via_search_not_context(self):
        client = ROOT / "benchmarks" / "search" / "client.py"
        self.assertTrue(client.exists(), msg="benchmarks/search/client.py is required")
        text = client.read_text(encoding="utf-8")
        self.assertIn("/retrieve/search", text)
        self.assertIn("/ingest/", text)
        self.assertIn("skip_enrichment", text)
        self.assertNotIn(
            "/retrieve/context",
            text,
            msg="search harness must not score via /retrieve/context",
        )
        self.assertNotIn(
            "/retrieve/recommend",
            text,
            msg="search harness must not score via /retrieve/recommend",
        )
        for needle in ("locomoconv", "demorecsys"):
            self.assertNotIn(
                needle,
                text,
                msg=f"search client must not default to {needle}",
            )

    def test_search_config_defaults_to_searchbench(self):
        config = ROOT / "benchmarks" / "search" / "config.py"
        text = config.read_text(encoding="utf-8")
        self.assertIn('DEFAULT_BRAIN_ID = "searchbenchsmoke"', text)
        self.assertIn('REQUIRED_PREFIX = "searchbench"', text)


if __name__ == "__main__":
    unittest.main()
