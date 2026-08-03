import os

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "PIPELINE_MODE": "accurate",
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

from unittest.mock import MagicMock, patch

from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.scout_agent import ScoutEntity
from src.core.saving.architect_batch import (
    BatchEndpoint,
    BatchExtractResponse,
    BatchRelationship,
)
from src.core.saving.ingest_cost import IngestCostLedger


def test_run_batch_extract_queues_valid_without_escalate():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice hired Bob as a contractor."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Hiring", type="EVENT"),
        ScoutEntity(uuid="u3", name="Bob", type="PERSON"),
    ]
    payload = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="TARGETED",
                source_span="Alice hired Bob as a contractor",
            ),
        ]
    )

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler"
    ) as tooler:
        agent.agent = MagicMock()
        agent.agent.invoke.return_value = {
            "structured_response": payload,
            "messages": [],
        }
        ledger = IngestCostLedger()
        rels = agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=True,
            timeout=5,
        )
        tooler.assert_not_called()
        assert len(rels) == 2
        assert ledger.architect_units == 1
        assert ledger.architect_escalations == 0
        assert ledger.architect_schema_calls == 1


def test_run_batch_extract_repairs_then_escalates_on_failure():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice met Bob."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Bob", type="PERSON"),
    ]
    bad = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="PERSON", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice met Bob",
            )
        ]
    )

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler", return_value=[]
    ) as tooler:
        agent.agent = MagicMock()
        agent.agent.invoke.return_value = {
            "structured_response": bad,
            "messages": [],
        }
        ledger = IngestCostLedger()
        agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=True,
            timeout=5,
            max_retries=1,
        )
        assert agent.agent.invoke.call_count == 2
        tooler.assert_called_once()
        assert ledger.architect_escalations == 1
        assert ledger.escalate_rate == 1.0
        assert ledger.architect_repair_calls == 1


def test_run_batch_extract_keeps_partial_without_escalate():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice hired Bob as a contractor."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Hiring", type="EVENT"),
        ScoutEntity(uuid="u3", name="Bob", type="PERSON"),
    ]
    mixed = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="TARGETED",
                source_span="totally fabricated zebra claim",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="PERSON", type="PERSON"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice hired Bob",
            ),
        ]
    )

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler"
    ) as tooler:
        agent.agent = MagicMock()
        agent.agent.invoke.return_value = {
            "structured_response": mixed,
            "messages": [],
        }
        ledger = IngestCostLedger()
        rels = agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=True,
            timeout=5,
            max_retries=1,
        )
        assert agent.agent.invoke.call_count == 2
        tooler.assert_not_called()
        assert len(rels) == 1
        assert rels[0].name == "MADE"
        assert ledger.architect_escalations == 0
        assert ledger.escalate_rate == 0.0
        assert ledger.architect_repair_calls == 1


def test_run_batch_extract_prefers_primary_when_repair_worse():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    source = "Alice hired Bob as a contractor."
    entities = [
        ScoutEntity(uuid="u1", name="Alice", type="PERSON"),
        ScoutEntity(uuid="u2", name="Hiring", type="EVENT"),
        ScoutEntity(uuid="u3", name="Bob", type="PERSON"),
    ]
    primary = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="Alice", type="PERSON"),
                tip=BatchEndpoint(uuid="u2", name="Hiring", type="EVENT"),
                name="MADE",
                source_span="Alice hired Bob as a contractor",
            ),
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="PERSON", type="PERSON"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice hired Bob",
            ),
        ]
    )
    worse_repair = BatchExtractResponse(
        relationships=[
            BatchRelationship(
                tail=BatchEndpoint(uuid="u1", name="PERSON", type="PERSON"),
                tip=BatchEndpoint(uuid="u3", name="Bob", type="PERSON"),
                name="MET",
                source_span="Alice hired Bob",
            )
        ]
    )

    with patch.object(agent, "_get_agent"), patch.object(
        agent, "run_tooler"
    ) as tooler:
        agent.agent = MagicMock()
        agent.agent.invoke.side_effect = [
            {"structured_response": primary, "messages": []},
            {"structured_response": worse_repair, "messages": []},
        ]
        ledger = IngestCostLedger()
        rels = agent.run_batch_extract(
            source,
            entities,
            cost_ledger=ledger,
            escalate=True,
            timeout=5,
            max_retries=1,
        )
        tooler.assert_not_called()
        assert len(rels) == 1
        assert rels[0].name == "MADE"
        assert ledger.architect_escalations == 0
