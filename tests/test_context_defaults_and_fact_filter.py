import contextlib
import inspect
import os
import re
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
from src.core.search.fact_filter import FactFilterDecision
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, happened_at: str | None = None, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


def _facts(count: int):
    return [
        (
            _node("a", "Alice"),
            _pred(f"r{i}a", f"ATTENDED_{i}"),
            _node(
                f"e{i}",
                f"Event{i}",
                happened_at=f"2024-0{i + 1}-01",
                labels=["EVENT"],
            ),
            _pred(f"r{i}b", f"WITH_{i}"),
            _node(f"b{i}", f"Person{i}"),
        )
        for i in range(count)
    ]


@contextlib.contextmanager
def _stubbed_adapters(retrieve_mod, facts, extra=()):
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
    mock_graph = MagicMock()
    mock_graph.get_event_centric_neighbors.return_value = facts
    mock_data = MagicMock()
    mock_data.get_last_text_chunks.return_value = []
    mock_data.get_last_structured_data.return_value = []

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
        for context in extra:
            stack.enter_context(context)
        yield mock_graph


class ContextRequestDefaultsTests(unittest.IsolatedAsyncioTestCase):
    def test_controller_reads_no_declared_field_through_a_getattr_default(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        source = inspect.getsource(retrieve_mod.get_context)
        offenders = [
            field
            for field in GetContextRequestBody.model_fields
            if re.search(rf"getattr\(\s*request\s*,\s*[\"']{field}[\"']", source)
        ]
        self.assertEqual(offenders, [])

    def test_ppr_is_on_and_sufficiency_retry_is_off_by_default(self):
        defaults = {
            name: field.default
            for name, field in GetContextRequestBody.model_fields.items()
        }
        self.assertIs(defaults["use_ppr"], True)
        self.assertIs(defaults["sufficiency_retry"], False)

    async def test_default_request_runs_ppr(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        adjacency = MagicMock(return_value={})
        with _stubbed_adapters(
            retrieve_mod,
            _facts(3),
            extra=[
                patch.object(retrieve_mod, "_build_adjacency_from_seeds", adjacency)
            ],
        ):
            await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a")
            )

        adjacency.assert_called_once()

    async def test_sufficiency_retry_is_opt_in(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        for sufficiency_retry in (False, True):
            with self.subTest(sufficiency_retry=sufficiency_retry):
                insufficient = MagicMock(return_value=True)
                passages = MagicMock(return_value=[])
                with _stubbed_adapters(
                    retrieve_mod,
                    _facts(2),
                    extra=[
                        patch.object(
                            retrieve_mod, "_context_looks_insufficient", insufficient
                        ),
                        patch.object(retrieve_mod, "_retrieve_passages", passages),
                    ],
                ):
                    await retrieve_mod.get_context(
                        GetContextRequestBody(
                            text="Alice",
                            brain_id="brain-a",
                            sufficiency_retry=sufficiency_retry,
                        )
                    )

                self.assertEqual(insufficient.called, sufficiency_retry)


class ContextFactFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_path_without_an_adapter_keeps_every_ranked_fact(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(3)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", apply_fact_filter=True
                )
            )

        self.assertEqual(len(response.triples), 3)

    async def test_supplied_adapter_actually_filters(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        adapter = MagicMock()
        adapter.generate_structured.return_value = FactFilterDecision(keep_indices=[1])

        with _stubbed_adapters(retrieve_mod, _facts(3)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", apply_fact_filter=True
                ),
                fact_filter_adapter=adapter,
            )

        adapter.generate_structured.assert_called_once()
        self.assertEqual(len(response.triples), 1)

    async def test_adapter_is_ignored_when_the_caller_disables_filtering(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        adapter = MagicMock()
        adapter.generate_structured.return_value = FactFilterDecision(keep_indices=[1])

        with _stubbed_adapters(retrieve_mod, _facts(3)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Alice", brain_id="brain-a", apply_fact_filter=False
                ),
                fact_filter_adapter=adapter,
            )

        adapter.generate_structured.assert_not_called()
        self.assertEqual(len(response.triples), 3)


if __name__ == "__main__":
    unittest.main()
