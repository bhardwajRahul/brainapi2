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


def _immediate_chain_factory(persist_calls):
    class ImmediateChain:
        def __init__(self, *steps):
            self.steps = steps

        def apply_async(self, *args, **kwargs):
            for step in self.steps:
                task_name = getattr(step, "task", None) or getattr(step, "name", "")
                args_ = getattr(step, "args", ()) or ()
                if "process_architect_relationships" in str(task_name) and args_:
                    persist_calls.append(args_[0])
            return MagicMock(id="chain")

    return ImmediateChain



def _full_triple():
    return {
        "subject": {"name": "Alice", "type": "PERSON", "uuid": "sub-1"},
        "subj_event": {"name": "MADE", "uuid": "rel-1"},
        "event": {
            "name": "Purchase",
            "type": "EVENT",
            "uuid": "evt-1",
            "happened_at": "01/01/2024",
        },
        "event_obj": {"name": "TARGETED", "uuid": "rel-2", "amount": 3},
        "object": {"name": "Widget", "type": "PRODUCT", "uuid": "obj-1"},
    }


class TripleConversionTests(unittest.TestCase):
    def test_full_triple_produces_subject_event_and_event_object_edges(self):
        from src.core.agents.architect_agent import ingestion_triples_to_relationships
        from src.services.api.constants.requests import IngestionTripleSet

        triple = IngestionTripleSet(**_full_triple())
        relationships, entities = ingestion_triples_to_relationships([triple], [])

        self.assertEqual(len(relationships), 2)
        self.assertEqual(relationships[0].tail.uuid, "sub-1")
        self.assertEqual(relationships[0].name, "MADE")
        self.assertEqual(relationships[0].tip.uuid, "evt-1")
        self.assertEqual(relationships[0].tip.happened_at, "01/01/2024")
        self.assertEqual(relationships[1].tail.uuid, "evt-1")
        self.assertEqual(relationships[1].name, "TARGETED")
        self.assertEqual(relationships[1].tip.uuid, "obj-1")
        self.assertEqual(relationships[1].amount, 3)
        self.assertIn(("alice", "person"), entities)
        self.assertIn(("purchase", "event", "01/01/2024"), entities)


class StructuredTriplePersistenceTests(unittest.TestCase):
    def test_triple_only_persists_without_llm(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-struct-1"
        persist_calls = []
        saved_structured = []
        scout_called = []
        architect_called = []

        args = {
            "data": [_full_triple()],
            "brain_id": "tenant-a",
            "text": None,
            "mode": "deterministic",
        }

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "graph_adapter"),
            patch.object(ingestion_mod, "IngestionManager"),
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "ScoutAgent") as ScoutAgent,
            patch.object(ingestion_mod, "ArchitectAgent") as ArchitectAgent,
            patch.object(ingestion_mod, "KGAgent") as KGAgent,
            patch("celery.chain", _immediate_chain_factory(persist_calls)),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch.object(ingestion_mod, "finalize_ingestion_task"),
        ):
            data_adapter.save_structured_data.side_effect = (
                lambda data, brain_id="default": saved_structured.append(
                    {"data": data, "brain_id": brain_id}
                )
                or data
            )
            ScoutAgent.side_effect = lambda **kwargs: scout_called.append(True)
            ArchitectAgent.side_effect = lambda **kwargs: architect_called.append(True)
            KGAgent.side_effect = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("KGAgent must not be constructed in deterministic mode")
            )

            result = ingestion_mod.ingest_structured_data.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(result, "task-struct-1")
        self.assertEqual(len(persist_calls), 1)
        relationships = persist_calls[0]["relationships"]
        self.assertEqual(len(relationships), 2)
        self.assertEqual(relationships[0]["name"], "MADE")
        self.assertEqual(relationships[1]["name"], "TARGETED")
        self.assertEqual(
            relationships[0]["properties"].get("source"), "structured_deterministic"
        )
        self.assertEqual(persist_calls[0]["brain_id"], "tenant-a")
        self.assertEqual(len(saved_structured), 1)
        self.assertEqual(saved_structured[0]["brain_id"], "tenant-a")
        self.assertEqual(saved_structured[0]["data"].id, "task-struct-1")
        self.assertEqual(scout_called, [])
        self.assertEqual(architect_called, [])

    def test_deterministic_ignores_text_for_llm(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-struct-det-text"
        persist_calls = []
        scout_called = []

        args = {
            "data": [_full_triple()],
            "brain_id": "recsys-demo",
            "text": "should not trigger scout",
            "mode": "deterministic",
        }

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "graph_adapter"),
            patch.object(ingestion_mod, "IngestionManager"),
            patch.object(ingestion_mod, "data_adapter"),
            patch.object(ingestion_mod, "ScoutAgent") as ScoutAgent,
            patch.object(ingestion_mod, "ArchitectAgent"),
            patch.object(ingestion_mod, "KGAgent"),
            patch("celery.chain", _immediate_chain_factory(persist_calls)),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch.object(ingestion_mod, "finalize_ingestion_task"),
        ):
            ScoutAgent.side_effect = lambda **kwargs: scout_called.append(True)
            ingestion_mod.ingest_structured_data.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(scout_called, [])
        self.assertEqual(len(persist_calls), 1)

    def test_deterministic_fails_closed_on_unresolved_name_anchor(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-struct-anchor-fail"
        graph = MagicMock()
        graph.get_by_identification_params.return_value = None

        args = {
            "data": [_full_triple()],
            "brain_id": "recsys-demo",
            "mode": "deterministic",
            "anchor": {"name": "Alice", "type": "PERSON"},
        }

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "graph_adapter", graph),
            patch.object(ingestion_mod, "IngestionManager"),
            patch.object(ingestion_mod, "data_adapter"),
            patch.object(ingestion_mod, "KGAgent") as KGAgent,
            patch.object(ingestion_mod, "set_ingestion_task_status"),
        ):
            KGAgent.side_effect = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("KGAgent must not run in deterministic mode")
            )
            with self.assertRaises(ValueError) as ctx:
                ingestion_mod.ingest_structured_data.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )
        self.assertIn("deterministic mode", str(ctx.exception))

    def test_text_enrichment_does_not_repersist_submitted_triples(self):
        from src.core.agents.architect_agent import ArchitectAgentResponse
        from src.core.agents.scout_agent import ScoutAgentResponse
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-struct-2"
        persist_calls = []
        run_structured_kwargs = []

        args = {
            "data": [_full_triple()],
            "brain_id": "tenant-a",
            "text": "Alice bought a widget",
            "mode": "hybrid",
        }

        scout = MagicMock()
        scout.run_structured.return_value = ScoutAgentResponse(entities=[])
        architect = MagicMock()
        architect.session_id = "sess-enrich"
        architect.take_pending_relationships.return_value = []
        architect.run_structured.side_effect = (
            lambda **kwargs: run_structured_kwargs.append(kwargs)
            or ArchitectAgentResponse(new_nodes=[], relationships=[])
        )

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "graph_adapter"),
            patch.object(ingestion_mod, "IngestionManager"),
            patch.object(ingestion_mod, "data_adapter"),
            patch.object(ingestion_mod, "ScoutAgent", return_value=scout),
            patch.object(ingestion_mod, "ArchitectAgent", return_value=architect),
            patch("celery.chain", _immediate_chain_factory(persist_calls)),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch.object(ingestion_mod, "finalize_ingestion_task"),
        ):
            ingestion_mod.ingest_structured_data.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(len(persist_calls), 1)
        self.assertEqual(len(run_structured_kwargs), 1)
        self.assertFalse(run_structured_kwargs[0]["persist_submitted"])
        for rel in persist_calls[0]["relationships"]:
            self.assertEqual(rel["properties"].get("source"), "structured_deterministic")

    def test_resolved_mode_inference(self):
        from src.services.api.constants.requests import (
            IngestionStructuredRequestBody,
            IngestionTripleSet,
        )

        triple = IngestionTripleSet(**_full_triple())
        self.assertEqual(
            IngestionStructuredRequestBody(data=[triple]).resolved_mode(),
            "deterministic",
        )
        self.assertEqual(
            IngestionStructuredRequestBody(
                data=[triple], text="x"
            ).resolved_mode(),
            "hybrid",
        )
        self.assertEqual(
            IngestionStructuredRequestBody(
                data=[triple], text="x", mode="deterministic"
            ).resolved_mode(),
            "deterministic",
        )


if __name__ == "__main__":
    unittest.main()
