import os
import unittest
from unittest.mock import MagicMock

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

from src.constants.embeddings import Vector
from src.core.agents.scout_agent import ScoutEntity
from src.core.search.catalog_graph import node_embed_text
from src.core.saving.ingestion_manager import IngestionManager
from src.lib.postgresql.graph_store import PostgreSQLGraphStore
from src.services.api.constants.requests import SearchRequestBody


class CatalogGraphSearchUnitTests(unittest.TestCase):
    def test_search_request_body_stays_passages_without_catalog_fields(self):
        body = SearchRequestBody(query="modern sofas")
        self.assertEqual(body.channels, ["passages"])
        fields = set(SearchRequestBody.model_fields)
        self.assertIn("target", fields)
        for banned in ("product_id", "sku", "brand", "user_id"):
            self.assertNotIn(banned, fields)

    def test_node_embed_text_uses_search_text_only_when_set(self):
        catalog = ScoutEntity(
            type="ENTITY",
            name="sofa",
            uuid="sofa-1",
            properties={"search_text": "velvet sofa navy modern"},
        )
        text, cache_key = node_embed_text(catalog)
        self.assertEqual(text, "velvet sofa navy modern")
        self.assertEqual(cache_key, "uuid:sofa-1")
        memory = ScoutEntity(type="PERSON", name="Ada", uuid="p1")
        mem_text, mem_key = node_embed_text(memory)
        self.assertEqual(mem_text, "Ada")
        self.assertEqual(mem_key, "Ada")

    def test_process_node_vectors_embeds_search_text(self):
        embeddings = MagicMock()
        embeddings.embed_text.return_value = Vector(
            id="v1", embeddings=[0.1, 0.2], metadata={}
        )
        store = MagicMock()
        store.add_vectors.return_value = ["vid-1"]
        manager = IngestionManager(embeddings, store, MagicMock())
        node = ScoutEntity(
            type="ENTITY",
            name="sofa",
            uuid="sofa-1",
            properties={"search_text": "velvet sofa for small rooms"},
        )
        manager.process_node_vectors(node, "searchbenchsmoke")
        embeddings.embed_text.assert_called_once_with("velvet sofa for small rooms")
        meta = embeddings.embed_text.return_value.metadata
        self.assertEqual(meta["uuid"], "sofa-1")
        self.assertEqual(meta["name"], "sofa")
        memory = ScoutEntity(type="PERSON", name="Ada", uuid="p1")
        manager.process_node_vectors(memory, "searchbenchsmoke")
        embeddings.embed_text.assert_called_with("Ada")

    def test_node_search_ddl_is_english_and_gated(self):
        ddl = PostgreSQLGraphStore._SEARCH_DDL
        self.assertIn("kg_nodes", ddl)
        self.assertIn("search_tsv", ddl)
        self.assertIn("to_tsvector", ddl)
        self.assertIn("'english'", ddl)
        self.assertNotIn("'italian'", ddl)
        self.assertIn("data->>'search_text'", ddl)
        self.assertNotIn("concat_ws", ddl)
        self.assertIn("::regconfig", ddl)
        self.assertIn("||", ddl)


if __name__ == "__main__":
    unittest.main()
