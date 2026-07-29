import contextlib
import os
import sys
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


class ContextPathNoDossierTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_context_does_not_invoke_dossier_retrieval(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        self.assertFalse(hasattr(retrieve_mod, "EventSynergyRetriever"))
        self.assertFalse(hasattr(retrieve_mod, "_flatten_match_path"))
        self.assertFalse(hasattr(retrieve_mod, "_MAX_DOSSIER_ENTITIES"))

        fact = (
            _node("a", "Alice"),
            _pred("r1", "ATTENDED"),
            _node("e1", "Event1", labels=["EVENT"]),
            _pred("r2", "WITH"),
            _node("b", "Bob"),
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_vs.search_data.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [fact]
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.get_text_chunks_by_ids.return_value = ([], None)
        mock_data.search.return_value = MagicMock(text_chunks=[])

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings)
            )
            stack.enter_context(patch.object(retrieve_mod, "vector_search", mock_vs))
            stack.enter_context(patch.object(retrieve_mod, "graph_adapter", mock_graph))
            stack.enter_context(patch.object(retrieve_mod, "data_adapter", mock_data))
            stack.enter_context(
                patch.object(
                    retrieve_mod._entity_extractor,
                    "extract_elements",
                    return_value=ExtractElementsResponse(
                        tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
                        noun_chunks=[],
                    ),
                )
            )
            synergy = stack.enter_context(
                patch("src.core.search.entity_info.EventSynergyRetriever")
            )
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="What did Alice do with Bob?",
                    brain_id="brain-a",
                    max_facts=5,
                    max_passages=2,
                    profile_stages=True,
                )
            )

        synergy.assert_not_called()
        self.assertFalse(
            any(
                line.startswith("[dossier:")
                for line in response.text_context.splitlines()
            )
        )
        stage_names = {
            stage["stage"] for stage in (response.stage_timings or {}).get("stages", [])
        }
        self.assertFalse(any(name.startswith("dossiers") for name in stage_names))
        self.assertEqual(len(response.triples), 1)


if __name__ == "__main__":
    unittest.main()
