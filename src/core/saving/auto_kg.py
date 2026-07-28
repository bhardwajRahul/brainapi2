"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import config
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutAgent
from src.core.agents.architect_agent import ArchitectAgent
from src.core.saving.ingestion_manager import IngestionManager
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)

import langsmith


@dataclass
class EnrichmentOrchestrationResult:
    session_id: Optional[str]
    ingestion_session_id: str
    brain_id: str
    enrichment_relationships: List[dict] = field(default_factory=list)
    should_consolidate: bool = False


def enrich_kg_from_input(
    input: str,
    targeting: Optional[Node] = None,
    brain_id: str = "default",
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    preferred_extraction_entities: Optional[List[str]] = None,
) -> EnrichmentOrchestrationResult:
    """
    Orchestrates enrichment of the knowledge graph from a free-text input.

    Runs the scout and architect agents to extract entities and relationships from the provided input
    and returns the collected persistence batch for the caller to write.
    """

    ingestion_session_id = str(uuid.uuid4())
    project_name = os.getenv("LANGSMITH_PROJECT", "brainapi")
    with langsmith.tracing_context(
        project_name=project_name,
        enabled=True,
        tags=["enrich_kg", "scout", "architect"],
        metadata={
            "ingestion_session_id": ingestion_session_id,
            "brain_id": brain_id,
            "flow": "enrich_kg_from_input",
            "source_chunk_id": source_chunk_id,
            "source_timestamp": source_timestamp,
        },
    ):
        return _enrich_kg_impl(
            input,
            targeting,
            brain_id,
            ingestion_session_id,
            source_chunk_id=source_chunk_id,
            source_timestamp=source_timestamp,
            preferred_extraction_entities=preferred_extraction_entities,
        )


def _enrich_kg_impl(
    input,
    targeting,
    brain_id: str,
    ingestion_session_id: str,
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    preferred_extraction_entities: Optional[List[str]] = None,
) -> EnrichmentOrchestrationResult:
    from src.core.saving.identity import stamp_provenance

    ingestion_manager = IngestionManager(
        embeddings_adapter, vector_store_adapter, graph_adapter
    )

    scout_agent = ScoutAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
    )
    architect_agent = ArchitectAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
        ingestion_manager=ingestion_manager,
    )

    mode = "coarse" if config.pipeline_mode == "lightweight" else "granular"
    print(f"[DEBUG (enrich_kg_from_input)]: {config.pipeline_mode} pipeline mode selected")

    entities = scout_agent.run(
        input,
        targeting=targeting,
        brain_id=brain_id,
        ingestion_session_id=ingestion_session_id,
        mode=("coarse" if mode == "coarse" else "granular"),
        reference_time=source_timestamp,
        preferred_extraction_entities=preferred_extraction_entities,
    )
    print("[DEBUG (initial_scout_entities)]: ", entities.entities)
    architect_agent.run_tooler(
        input,
        entities.entities,
        targeting=targeting,
        brain_id=brain_id,
        timeout=20000,
        ingestion_session_id=ingestion_session_id,
        mode=("coarse" if mode == "coarse" else "granular"),
    )

    pending = architect_agent.take_pending_relationships()
    enrichment_relationships = []
    for rel in pending:
        payload = rel.model_dump(mode="json")
        payload["properties"] = stamp_provenance(
            payload.get("properties"),
            source_chunk_id=source_chunk_id,
            source_timestamp=source_timestamp,
        )
        for endpoint_key in ("tail", "tip"):
            endpoint = payload.get(endpoint_key) or {}
            endpoint["properties"] = stamp_provenance(
                endpoint.get("properties"),
                source_chunk_id=source_chunk_id,
                source_timestamp=source_timestamp,
            )
            payload[endpoint_key] = endpoint
        enrichment_relationships.append(payload)
    should_consolidate = bool(
        config.pipeline_mode == "accurate"
        and config.run_graph_consolidator
        and enrichment_relationships
    )

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except ImportError:
        pass

    return EnrichmentOrchestrationResult(
        session_id=architect_agent.session_id,
        ingestion_session_id=ingestion_session_id,
        brain_id=brain_id,
        enrichment_relationships=enrichment_relationships,
        should_consolidate=should_consolidate,
    )
