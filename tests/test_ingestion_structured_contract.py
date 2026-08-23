import os
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
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


def _minimal_triple():
    return {
        "event": {"name": "Purchase", "type": "EVENT"},
        "event_obj": {"name": "TARGETED"},
        "object": {"name": "Widget", "type": "PRODUCT"},
    }


class StructuredRequestValidationTests(unittest.TestCase):
    def test_triple_only_without_anchor_or_text_validates(self):
        from src.services.api.constants.requests import IngestionStructuredRequestBody

        body = IngestionStructuredRequestBody(data=[_minimal_triple()])
        self.assertIsNone(body.anchor)
        self.assertIsNone(body.text)

    def test_direct_has_triple_validates(self):
        from src.services.api.constants.requests import IngestionTripleSet

        triple = IngestionTripleSet(
            subject={"name": "bed", "type": "ENTITY", "uuid": "0"},
            subj_event={"name": "HAS", "uuid": "rel-1"},
            object={"name": "navy", "type": "ATTR", "uuid": "hub-navy"},
        )
        self.assertIsNone(triple.event)
        self.assertIsNone(triple.event_obj)
        prefers = IngestionTripleSet(
            subject={"name": "u01", "type": "USER", "uuid": "user:u01"},
            subj_event={"name": "PREFERS", "uuid": "rel-pref"},
            object={"name": "70s", "type": "ATTR", "uuid": "hub:attr:70s"},
        )
        self.assertIsNone(prefers.event)
        self.assertEqual(prefers.subj_event.name, "PREFERS")
        with self.assertRaises(ValidationError):
            IngestionTripleSet(
                subject={"name": "bed", "type": "ENTITY", "uuid": "0"},
                subj_event={"name": "HAS"},
                event={"name": "HAS", "type": "EVENT"},
                object={"name": "navy", "type": "ATTR"},
            )

    def test_string_anchor_is_rejected(self):
        from src.services.api.constants.requests import IngestionStructuredRequestBody

        with self.assertRaises(ValidationError):
            IngestionStructuredRequestBody(
                data=[_minimal_triple()],
                anchor="John",
            )

    def test_anchor_requires_uuid_or_name_and_type(self):
        from src.services.api.constants.requests import IngestionStructuredRequestBody

        with self.assertRaises(ValidationError):
            IngestionStructuredRequestBody(
                data=[_minimal_triple()],
                anchor={"name": "John"},
            )
        with self.assertRaises(ValidationError):
            IngestionStructuredRequestBody(
                data=[_minimal_triple()],
                anchor={"type": "PERSON"},
            )

        body = IngestionStructuredRequestBody(
            data=[_minimal_triple()],
            anchor={"uuid": "abc-123"},
        )
        self.assertEqual(body.anchor.uuid, "abc-123")

        body = IngestionStructuredRequestBody(
            data=[_minimal_triple()],
            anchor={"name": "John", "type": "PERSON"},
        )
        self.assertEqual(body.anchor.name, "John")


class StructuredAnchorRuntimeTests(unittest.TestCase):
    def test_missing_uuid_anchor_raises_and_marks_failed(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-anchor-miss"
        cache_writes = []

        def fake_cache_set(key, value, brain_id="default", expires_in=None):
            cache_writes.append({"key": key, "value": value, "brain_id": brain_id})
            return True

        args = {
            "data": [_minimal_triple()],
            "anchor": {"uuid": "missing-uuid"},
            "brain_id": "tenant-a",
        }

        with (
            patch.object(ingestion_mod, "cache_adapter") as cache_adapter,
            patch.object(ingestion_mod, "graph_adapter") as graph_adapter,
            patch.object(ingestion_mod, "IngestionManager"),
        ):
            cache_adapter.set.side_effect = fake_cache_set
            graph_adapter.get_by_uuid.return_value = None

            with self.assertRaises(ValueError) as ctx:
                ingestion_mod.ingest_structured_data.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )

        self.assertIn("missing-uuid", str(ctx.exception))
        failed = [w for w in cache_writes if '"status": "failed"' in w["value"]]
        self.assertTrue(failed)
        self.assertEqual(failed[-1]["brain_id"], "tenant-a")
        graph_adapter.get_by_uuid.assert_called_once_with(
            "missing-uuid", brain_id="tenant-a"
        )

    def test_name_type_anchor_lookup_passes_brain_id(self):
        from src.constants.kg import Node
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-anchor-name"
        captured = {}

        def fake_get_by_identification_params(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return Node(
                uuid="n1",
                name="John",
                labels=["PERSON"],
                description="",
                properties={},
            )

        args = {
            "data": [_minimal_triple()],
            "anchor": {"name": "John", "type": "PERSON"},
            "brain_id": "tenant-b",
        }

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "graph_adapter") as graph_adapter,
            patch.object(ingestion_mod, "IngestionManager"),
            patch.object(ingestion_mod, "data_adapter"),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch("celery.chain") as chain_mock,
        ):
            chain_mock.return_value.apply_async = MagicMock()
            graph_adapter.get_by_identification_params.side_effect = (
                fake_get_by_identification_params
            )
            result = ingestion_mod.ingest_structured_data.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(result, "task-anchor-name")
        self.assertEqual(captured["kwargs"].get("brain_id"), "tenant-b")


if __name__ == "__main__":
    unittest.main()
