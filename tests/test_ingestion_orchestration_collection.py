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


class IngestionOrchestrationCollectionTests(unittest.TestCase):
    def test_architect_queues_relationships_instead_of_celery_delay(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.core.agents.architect_agent import ArchitectAgent

        agent = ArchitectAgent.__new__(ArchitectAgent)
        agent.relationships_set = []
        agent.pending_persistence_batches = []
        agent.session_id = "session-1"

        rel = ArchitectAgentRelationship(
            uuid="rel-1",
            flow_key="flow-1",
            name="OWNS",
            tail=ArchitectAgentEntity(uuid="a", name="Alice", type="PERSON"),
            tip=ArchitectAgentEntity(uuid="b", name="Widget", type="PRODUCT"),
        )
        with patch(
            "src.workers.tasks.ingestion.process_architect_relationships"
        ) as mocked_task:
            agent.queue_relationships_for_persistence([rel])
            mocked_task.delay.assert_not_called()

        pending = agent.take_pending_relationships()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].uuid, "rel-1")
        self.assertEqual(agent.pending_persistence_batches, [])

    def test_enrich_kg_returns_complete_persistence_batch(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.core.saving.auto_kg import EnrichmentOrchestrationResult
        from src.core.saving import auto_kg

        rel = ArchitectAgentRelationship(
            uuid="rel-1",
            flow_key="flow-1",
            name="OWNS",
            description="owns",
            tail=ArchitectAgentEntity(uuid="a", name="Alice", type="PERSON"),
            tip=ArchitectAgentEntity(uuid="b", name="Widget", type="PRODUCT"),
        )

        scout = MagicMock()
        scout.run.return_value = MagicMock(entities=[])
        architect = MagicMock()
        architect.session_id = "sess-1"
        architect.take_pending_relationships.return_value = [rel]

        with patch.object(auto_kg, "ScoutAgent", return_value=scout), patch.object(
            auto_kg, "ArchitectAgent", return_value=architect
        ), patch.object(auto_kg, "IngestionManager", return_value=MagicMock()), patch.object(
            auto_kg.config, "pipeline_mode", "accurate"
        ), patch.object(
            auto_kg.config, "run_graph_consolidator", True
        ), patch(
            "src.workers.tasks.ingestion.consolidate_graph_async"
        ) as consolidate_task:
            result = auto_kg.enrich_kg_from_input("Alice owns a widget", brain_id="brain-a")
            consolidate_task.delay.assert_not_called()

        self.assertIsInstance(result, EnrichmentOrchestrationResult)
        self.assertEqual(result.session_id, "sess-1")
        self.assertEqual(result.brain_id, "brain-a")
        self.assertEqual(len(result.enrichment_relationships), 1)
        self.assertTrue(result.should_consolidate)

    def test_create_relationship_tool_does_not_publish_celery_tasks(self):
        from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
        from src.core.agents.architect_agent import ArchitectAgent

        agent = ArchitectAgent.__new__(ArchitectAgent)
        agent.relationships_set = []
        agent.pending_persistence_batches = []
        agent.session_id = "sess-1"

        output_rels = [
            ArchitectAgentRelationship(
                uuid="rel-1",
                flow_key="flow-1",
                name="OWNS",
                tail=ArchitectAgentEntity(uuid="a", name="Alice", type="PERSON"),
                tip=ArchitectAgentEntity(uuid="b", name="Widget", type="PRODUCT"),
            )
        ]

        with patch(
            "src.workers.tasks.ingestion.process_architect_relationships"
        ) as mocked_task:
            agent.queue_relationships_for_persistence(output_rels)
            mocked_task.delay.assert_not_called()

        self.assertEqual(len(agent.take_pending_relationships()), 1)

if __name__ == "__main__":
    unittest.main()
