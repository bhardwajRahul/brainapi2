from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

from src.constants.agents import (
    ArchitectAgentEntity,
    ArchitectAgentRelationship,
    AtomicJanitorAgentInputOutput,
    AtomicJanitorAgentWrongRelationship,
    _ArchitectAgentRelationship,
)
from src.core.agents.architect_agent import ArchitectAgent
from src.core.saving.ingest_cost import IngestCostLedger


def _rel(name="MET", grounded=True):
    props = {}
    if grounded:
        props["source_span"] = "Alice met Bob at the park"
    return ArchitectAgentRelationship(
        flow_key="fk",
        uuid="r1",
        name=name,
        description="Alice met Bob at the park" if grounded else "zebras invented ships",
        properties=props,
        tail=ArchitectAgentEntity(
            uuid="a", name="Alice", type="PERSON", description="", properties={}
        ),
        tip=ArchitectAgentEntity(
            uuid="b", name="Bob", type="PERSON", description="", properties={}
        ),
    )


def test_janitor_ok_status_keeps_batch():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    # Force ambiguous path: weak/missing span so triage sends to LLM.
    rel = _rel(grounded=False)
    rel.properties = {}
    rel.description = ""
    agent.pending_persistence_batches = [[rel]]
    ledger = IngestCostLedger()

    mock_janitor = MagicMock()
    mock_janitor.run_atomic_janitor.return_value = "OK"
    with patch(
        "src.core.agents.janitor_agent.JanitorAgent", return_value=mock_janitor
    ), patch("src.services.input.agents.llm_small_adapter"), patch(
        "src.services.input.agents.graph_adapter"
    ) as ga, patch(
        "src.services.input.agents.vector_store_adapter"
    ), patch(
        "src.services.input.agents.embeddings_adapter"
    ):
        ga.graphdb_description = "db"
        agent.run_batched_janitor(
            text="Alice met Bob at the park.",
            cost_ledger=ledger,
        )
    pending = agent.take_pending_relationships()
    assert len(pending) == 1
    assert ledger.janitor_ran == 1
    assert ledger.janitor_ambiguous == 1


def test_janitor_reject_status_vetoes_without_persist():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    rel = _rel(grounded=False)
    rel.properties = {}
    rel.description = ""
    agent.pending_persistence_batches = [[rel]]
    ledger = IngestCostLedger()

    mock_janitor = MagicMock()
    mock_janitor.run_atomic_janitor.return_value = AtomicJanitorAgentInputOutput(
        status="REJECT",
        veto_reasons=["fabricated_edge"],
    )
    with patch(
        "src.core.agents.janitor_agent.JanitorAgent", return_value=mock_janitor
    ), patch("src.services.input.agents.llm_small_adapter"), patch(
        "src.services.input.agents.graph_adapter"
    ) as ga, patch(
        "src.services.input.agents.vector_store_adapter"
    ), patch(
        "src.services.input.agents.embeddings_adapter"
    ):
        ga.graphdb_description = "db"
        agent.run_batched_janitor(
            text="Alice met Bob at the park.",
            cost_ledger=ledger,
        )
    assert agent.take_pending_relationships() == []
    assert "fabricated_edge" in ledger.janitor_drop_reasons


def test_janitor_parse_failure_does_not_approve():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    rel = _rel(grounded=False)
    rel.properties = {}
    rel.description = ""
    agent.pending_persistence_batches = [[rel]]
    ledger = IngestCostLedger()

    mock_janitor = MagicMock()
    mock_janitor.run_atomic_janitor.return_value = None
    with patch(
        "src.core.agents.janitor_agent.JanitorAgent", return_value=mock_janitor
    ), patch("src.services.input.agents.llm_small_adapter"), patch(
        "src.services.input.agents.graph_adapter"
    ) as ga, patch(
        "src.services.input.agents.vector_store_adapter"
    ), patch(
        "src.services.input.agents.embeddings_adapter"
    ):
        ga.graphdb_description = "db"
        agent.run_batched_janitor(
            text="Alice met Bob at the park.",
            cost_ledger=ledger,
        )
    assert agent.take_pending_relationships() == []
    assert "janitor_parse_failure" in ledger.janitor_drop_reasons


def test_janitor_wrong_relationship_dropped_fixed_kept():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    bad = _rel(grounded=False)
    bad.properties = {}
    bad.description = ""
    agent.pending_persistence_batches = [[bad]]
    ledger = IngestCostLedger()

    fixed = _ArchitectAgentRelationship(
        tip=bad.tip,
        tail=bad.tail,
        name="MET",
        description="Alice met Bob at the park",
        properties={"source_span": "Alice met Bob at the park"},
    )
    wrong = AtomicJanitorAgentWrongRelationship(
        relationship=_ArchitectAgentRelationship(
            tip=bad.tip,
            tail=bad.tail,
            name=bad.name,
            description=bad.description or "",
            properties={},
        ),
        reason="ungrounded",
        instructions="drop",
    )
    mock_janitor = MagicMock()
    mock_janitor.run_atomic_janitor.return_value = AtomicJanitorAgentInputOutput(
        status="ERROR",
        fixed_relationships=[fixed],
        wrong_relationships=[wrong],
    )
    with patch(
        "src.core.agents.janitor_agent.JanitorAgent", return_value=mock_janitor
    ), patch("src.services.input.agents.llm_small_adapter"), patch(
        "src.services.input.agents.graph_adapter"
    ) as ga, patch(
        "src.services.input.agents.vector_store_adapter"
    ), patch(
        "src.services.input.agents.embeddings_adapter"
    ):
        ga.graphdb_description = "db"
        agent.run_batched_janitor(
            text="Alice met Bob at the park.",
            cost_ledger=ledger,
        )
    pending = agent.take_pending_relationships()
    assert len(pending) == 1
    assert pending[0].name == "MET"
    assert "ungrounded" in ledger.janitor_drop_reasons


def test_deterministic_reject_skips_llm():
    agent = ArchitectAgent(
        llm_adapter=MagicMock(),
        cache_adapter=MagicMock(),
        kg=MagicMock(),
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        ingestion_manager=MagicMock(),
    )
    placeholder = ArchitectAgentRelationship(
        flow_key="fk",
        uuid="r1",
        name="MET",
        description="Alice met Bob",
        properties={"source_span": "Alice met Bob"},
        tail=ArchitectAgentEntity(
            uuid="a", name="PERSON", type="PERSON", description="", properties={}
        ),
        tip=ArchitectAgentEntity(
            uuid="b", name="Bob", type="PERSON", description="", properties={}
        ),
    )
    agent.pending_persistence_batches = [[placeholder]]
    ledger = IngestCostLedger()
    with patch("src.core.agents.janitor_agent.JanitorAgent") as jan_cls:
        agent.run_batched_janitor(
            text="Alice met Bob at the park.",
            cost_ledger=ledger,
        )
        jan_cls.assert_not_called()
    assert agent.take_pending_relationships() == []
    assert ledger.janitor_rejected == 1
    assert ledger.janitor_ran == 0
    assert "type_named_placeholder_endpoint" in ledger.janitor_drop_reasons
