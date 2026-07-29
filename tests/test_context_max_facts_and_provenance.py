import contextlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError


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

from src.constants.data import TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, labels=None, source_chunk_ids=None):
    props = {}
    if source_chunk_ids:
        props["source_chunk_ids"] = list(source_chunk_ids)
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        properties=props,
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


def _fact(*, chunk_ids=None):
    return (
        _node("a", "Alice", source_chunk_ids=chunk_ids),
        _pred("r1", "ATTENDED"),
        _node("e1", "Event1", labels=["EVENT"], source_chunk_ids=chunk_ids),
        _pred("r2", "WITH"),
        _node("b", "Bob", source_chunk_ids=chunk_ids),
    )


@contextlib.contextmanager
def _stubbed_adapters(retrieve_mod, facts, *, chunks=None, passages=None, extra=()):
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
    mock_graph.get_event_centric_neighbors.return_value = facts
    mock_data = MagicMock()
    mock_data.get_last_text_chunks.return_value = []
    mock_data.get_last_structured_data.return_value = []
    mock_data.get_text_chunks_by_ids.return_value = (chunks or [], None)
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
                return_value=ExtractElementsResponse(tokens=[], noun_chunks=[]),
            )
        )
        if passages is not None:
            stack.enter_context(
                patch.object(retrieve_mod, "_retrieve_passages", return_value=passages)
            )
        for context in extra:
            stack.enter_context(context)
        yield mock_data


class MaxFactsZeroTests(unittest.IsolatedAsyncioTestCase):
    def test_negative_max_facts_rejected_at_schema(self):
        with self.assertRaises(ValidationError):
            GetContextRequestBody(text="Alice", brain_id="brain-a", max_facts=-1)

    async def test_max_facts_zero_returns_no_graph_facts(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            [_fact(chunk_ids=["chunk-1"]), _fact(chunk_ids=["chunk-2"])],
            passages=[("p1", 0.1, "Session id: session_9. Passage body.")],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice",
                    brain_id="brain-a",
                    max_facts=0,
                    max_passages=2,
                )
            )

        self.assertEqual(response.triples, [])
        self.assertIsNone(response.graph_session_ids)
        fact_lines = [
            line
            for line in response.text_context.splitlines()
            if line and not line.startswith("[passage]")
        ]
        self.assertEqual(fact_lines, [])
        self.assertTrue(
            any("Session id: session_9" in p for p in response.source_passages)
        )

    async def test_max_facts_one_keeps_a_single_fact(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            [_fact(chunk_ids=["chunk-1"]), _fact(chunk_ids=["chunk-2"])],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", max_facts=1
                )
            )

        self.assertEqual(len(response.triples), 1)


class GraphProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_triples_expose_chunk_and_session_provenance(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunks = [
            TextChunk(
                id="chunk-3",
                text="Session id: session_3. Melanie put the keys in the slipper.",
            )
        ]
        with _stubbed_adapters(
            retrieve_mod,
            [_fact(chunk_ids=["chunk-3"])],
            chunks=chunks,
            passages=[],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", max_facts=5, max_passages=2
                )
            )

        self.assertEqual(len(response.triples), 1)
        triple = response.triples[0]
        self.assertEqual(triple.source_chunk_ids, ["chunk-3"])
        self.assertEqual(triple.source_session_ids, ["session_3"])
        self.assertEqual(response.graph_session_ids, ["session_3"])
        self.assertTrue(
            any("(session_3)" in line for line in response.text_context.splitlines())
        )

    async def test_missing_chunk_provenance_omits_session_ids(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            [_fact(chunk_ids=["chunk-missing"])],
            chunks=[],
            passages=[],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", max_facts=5
                )
            )

        self.assertEqual(len(response.triples), 1)
        self.assertEqual(response.triples[0].source_chunk_ids, ["chunk-missing"])
        self.assertIsNone(response.triples[0].source_session_ids)
        self.assertIsNone(response.graph_session_ids)

    async def test_instrumentation_still_records_core_stages(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(
            retrieve_mod,
            [_fact(chunk_ids=["chunk-3"])],
            chunks=[
                TextChunk(
                    id="chunk-3",
                    text="Session id: session_3. Evidence.",
                )
            ],
            passages=[("p1", 0.2, "Session id: session_9. Passage.")],
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice",
                    brain_id="brain-a",
                    max_facts=2,
                    profile_stages=True,
                )
            )

        self.assertIsNotNone(response.stage_timings)
        stage_names = {
            stage["stage"] for stage in response.stage_timings.get("stages", [])
        }
        self.assertFalse(any(name.startswith("dossiers") for name in stage_names))
        self.assertTrue(any(name.startswith("facts") for name in stage_names))
        self.assertTrue(any(name.startswith("passages") for name in stage_names))


if __name__ == "__main__":
    unittest.main()
