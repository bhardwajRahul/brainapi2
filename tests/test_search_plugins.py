import os
import sys
import unittest
from pathlib import Path
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from fastapi import HTTPException

from src.constants.data import TextChunk
from src.core.search.hooks import (
    register_search_reranker,
    register_search_retriever,
    reset_search_plugins,
)
from src.services.api.constants.requests import SearchRequestBody

ROOT = Path(__file__).resolve().parents[1]


class SearchPluginHookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_search_plugins()

    def tearDown(self):
        reset_search_plugins()

    async def test_unknown_rerank_plugin_is_400(self):
        from src.services.api.controllers import search as search_mod

        with patch.object(search_mod.config, "search_enabled", True):
            with self.assertRaises(HTTPException) as ctx:
                await search_mod.search(
                    SearchRequestBody(
                        query="license",
                        rerank="plugin:missing-ce",
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unknown search rerank plugin", str(ctx.exception.detail))

    async def test_unknown_retriever_plugin_is_400(self):
        from src.services.api.controllers import search as search_mod

        with patch.object(search_mod.config, "search_enabled", True):
            with self.assertRaises(HTTPException) as ctx:
                await search_mod.search(
                    SearchRequestBody(
                        query="license",
                        channels=["plugin:splade"],
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unknown search retriever plugin", str(ctx.exception.detail))

    async def test_linear_rerank_is_400_not_silent(self):
        from src.services.api.controllers import search as search_mod

        with patch.object(search_mod.config, "search_enabled", True):
            with self.assertRaises(HTTPException) as ctx:
                await search_mod.search(
                    SearchRequestBody(query="license", rerank="linear")
                )
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_cross_encoder_reranks_top_hits(self):
        from src.services.api.controllers import search as search_mod

        def reverse_rerank(query, candidates, k):
            ordered = list(reversed(candidates))
            for index, item in enumerate(ordered):
                item["score"] = 10 - index
            return ordered[:k]

        register_search_reranker("cross-encoder", reverse_rerank)
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id="first", text="alpha"), 9.0),
            (TextChunk(id="second", text="beta"), 8.0),
            (TextChunk(id="third", text="gamma"), 7.0),
        ]
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=3,
                    rerank="plugin:cross-encoder",
                )
            )

        self.assertEqual([hit.id for hit in response.hits], ["third", "second", "first"])
        self.assertEqual(response.hits[0].scores.rerank, 10)
        mock_embeddings.embed_text.assert_not_called()

    def test_catalog_retrieve_and_rerank_caps(self):
        from src.core.search.hooks import (
            CATALOG_RERANK_MAX_K,
            RERANK_MAX_K,
            rerank_max_k_for_mode,
            retrieve_k_for_mode,
        )

        self.assertEqual(RERANK_MAX_K, 10)
        self.assertEqual(CATALOG_RERANK_MAX_K, 50)
        self.assertEqual(retrieve_k_for_mode("default", 10), 10)
        self.assertEqual(retrieve_k_for_mode("catalog", 10), 50)
        self.assertEqual(retrieve_k_for_mode("catalog", 50), 50)
        self.assertEqual(retrieve_k_for_mode("catalog", 200), 200)
        self.assertEqual(rerank_max_k_for_mode("default"), 10)
        self.assertEqual(rerank_max_k_for_mode("catalog"), 50)

    async def test_default_reranks_at_most_ten(self):
        from src.core.search.hooks import RERANK_MAX_K
        from src.services.api.controllers import search as search_mod

        seen: list[tuple[int, int]] = []

        def capture_rerank(query, candidates, k):
            seen.append((len(candidates), k))
            return list(candidates)[:k]

        register_search_reranker("cross-encoder", capture_rerank)
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id=f"h{index}", text=f"t{index}"), float(60 - index))
            for index in range(50)
        ]
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=50,
                    rerank="plugin:cross-encoder",
                )
            )

        self.assertEqual(RERANK_MAX_K, 10)
        self.assertEqual(seen, [(10, 10)])
        self.assertEqual(len(response.hits), 50)
        mock_data.search_bm25.assert_called_with("license", "default", limit=50)

    async def test_catalog_mode_reranks_fifty_when_k_fifty(self):
        from src.core.search.hooks import CATALOG_RERANK_MAX_K
        from src.services.api.controllers import search as search_mod

        seen: list[tuple[int, int]] = []

        def capture_rerank(query, candidates, k):
            seen.append((len(candidates), k))
            ordered = list(reversed(candidates))
            return ordered[:k]

        register_search_reranker("cross-encoder", capture_rerank)
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id=f"h{index}", text=f"t{index}"), float(60 - index))
            for index in range(50)
        ]
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=50,
                    mode="catalog",
                    rerank="plugin:cross-encoder",
                )
            )

        self.assertEqual(CATALOG_RERANK_MAX_K, 50)
        self.assertEqual(seen, [(50, 50)])
        self.assertEqual(len(response.hits), 50)
        self.assertEqual(response.hits[0].id, "h49")
        mock_data.search_bm25.assert_called_with("license", "default", limit=50)

    async def test_catalog_mode_retrieves_fifty_when_k_ten(self):
        from src.services.api.controllers import search as search_mod

        seen: list[tuple[int, int]] = []

        def capture_rerank(query, candidates, k):
            seen.append((len(candidates), k))
            return list(candidates)[:k]

        register_search_reranker("cross-encoder", capture_rerank)
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id=f"h{index}", text=f"t{index}"), float(60 - index))
            for index in range(50)
        ]
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=10,
                    mode="catalog",
                    rerank="plugin:cross-encoder",
                )
            )

        self.assertEqual(seen, [(50, 50)])
        self.assertEqual(len(response.hits), 10)
        mock_data.search_bm25.assert_called_with("license", "default", limit=50)

    async def test_splade_channel_fuses_without_silent_bm25_fallback(self):
        from src.services.api.controllers import search as search_mod

        def fake_splade(query, brain_id, k):
            return ["splade-hit"], {"splade-hit": 4.2}, {"splade-hit": "sparse hit"}

        register_search_retriever("splade", fake_splade)
        mock_data = MagicMock()
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=5,
                    channels=["plugin:splade"],
                )
            )

        self.assertEqual([hit.id for hit in response.hits], ["splade-hit"])
        self.assertEqual(response.hits[0].scores.plugin["splade"], 4.2)
        mock_data.search_bm25.assert_not_called()

    async def test_plugin_lists_fill_frozen_tail_not_head(self):
        from src.services.api.controllers import search as search_mod

        def fake_splade(query, brain_id, k):
            return (
                ["plugin-extra"],
                {"plugin-extra": 9.9},
                {"plugin-extra": "sparse extra"},
            )

        register_search_retriever("splade", fake_splade)
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id=f"h{index}", text=f"t{index}"), float(20 - index))
            for index in range(12)
        ]
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod.config, "search_literal_fill", False),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=15,
                    channels=["passages", "plugin:splade"],
                )
            )

        ids = [hit.id for hit in response.hits]
        self.assertEqual(ids[:10], [f"h{index}" for index in range(10)])
        self.assertIn("plugin-extra", ids[10:])
        self.assertNotIn("plugin-extra", ids[:10])

    def test_context_controller_does_not_import_search_hooks(self):
        retrieve = ROOT / "src" / "services" / "api" / "controllers" / "retrieve.py"
        text = retrieve.read_text(encoding="utf-8")
        self.assertNotIn("src.core.search.hooks", text)
        self.assertNotIn("src.core.search.graph_channels", text)
        self.assertNotIn("register_search_reranker", text)
        self.assertNotIn("plugin:cross-encoder", text)


class SearchPluginPackageTests(unittest.TestCase):
    def setUp(self):
        reset_search_plugins()
        self._extra_paths: list[str] = []
        self._extra_modules: list[str] = []

    def tearDown(self):
        reset_search_plugins()
        for name in self._extra_modules:
            sys.modules.pop(name, None)
        for path in self._extra_paths:
            if path in sys.path:
                sys.path.remove(path)

    def _load_plugin_dir(self, name: str):
        plugin_dir = str(ROOT / "plugins" / name)
        sys.path.insert(0, plugin_dir)
        self._extra_paths.append(plugin_dir)
        return plugin_dir

    def test_cross_encoder_plugin_reranks_with_injected_predict(self):
        self._load_plugin_dir("search-rerank")
        import rerank as ce

        self._extra_modules.extend(["rerank", "main", "routes"])
        ce.set_predict(lambda pairs: [float(len(text)) for _, text in pairs])
        ranked = ce.rerank(
            "q",
            [
                {"id": "short", "text": "ab", "score": 1},
                {"id": "long", "text": "abcdefghij", "score": 1},
            ],
            10,
        )
        self.assertEqual([row["id"] for row in ranked], ["long", "short"])
        from main import register as register_ce

        class Ctx:
            _app = None

            def register_search_reranker(self, name, fn):
                register_search_reranker(name, fn)

            def register_search_retriever(self, name, fn):
                register_search_retriever(name, fn)

        register_ce(Ctx())
        from src.core.search.hooks import get_search_reranker

        self.assertIsNotNone(get_search_reranker("cross-encoder"))

    def test_splade_index_and_retrieve_with_fake_encoder(self):
        self._load_plugin_dir("search-splade")
        import encode as splade_encode
        import index as splade_index

        self._extra_modules.extend(["encode", "index", "routes", "main"])
        splade_encode.set_encoder(
            lambda text: {token: 1.0 for token in text.split() if token}
        )
        splade_index.index_chunks(
            "searchbenchsmoke",
            [
                {"id": "a", "text": "alice license portland"},
                {"id": "b", "text": "piano thursday"},
            ],
        )
        ids, scores, texts = splade_index.retrieve(
            "alice license", "searchbenchsmoke", 2
        )
        self.assertEqual(ids[0], "a")
        self.assertGreater(scores["a"], scores.get("b", 0))
        self.assertIn("alice", texts["a"])

    def test_colbert_maxsim_retrieve_with_fake_encoder(self):
        self._load_plugin_dir("search-colbert")
        import encode as colbert_encode
        import index as colbert_index

        self._extra_modules.extend(["encode", "index", "routes", "main"])

        def fake_tokens(text: str):
            table = {
                "q": [[1.0, 0.0]],
                "match": [[1.0, 0.0]],
                "other": [[0.0, 1.0]],
            }
            return table.get(text, [[0.0, 0.0]])

        colbert_encode.set_encoder(fake_tokens)
        colbert_index.index_chunks(
            "searchbenchsmoke",
            [{"id": "hit", "text": "match"}, {"id": "miss", "text": "other"}],
        )
        ids, scores, _ = colbert_index.retrieve("q", "searchbenchsmoke", 2)
        self.assertEqual(ids[0], "hit")
        self.assertGreater(scores["hit"], scores["miss"])


if __name__ == "__main__":
    unittest.main()
