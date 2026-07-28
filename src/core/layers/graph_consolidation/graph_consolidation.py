"""
File: /graph_consolidation.py
Project: graph_consolidation
Created Date: Saturday January 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List

from src.core.agents.architect_agent import ArchitectAgentRelationship
from src.core.agents.janitor_agent import (
    JanitorAgent,
)
from src.core.agents.kg_agent import KGAgent
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.core.instances import llm_small_adapter as llm_adapter

RELATIONSHIP_BATCH_SIZE = 20


def consolidate_graph(
    new_relationships: List[ArchitectAgentRelationship],
    brain_id: str = "default",
) -> None:
    """
    Consolidates and normalizes a collection of new knowledge-graph relationships across the graph.

    Processes the provided relationships in batches to perform macroscopic fixes such as name normalization, connection normalization, and deduplication across multiple relationships and graph areas.

    Parameters:
        brain_id (str): Identifier of the target knowledge graph/brain to consolidate into (defaults to "default").
    """

    batches = [
        new_relationships[i : i + RELATIONSHIP_BATCH_SIZE]
        for i in range(0, len(new_relationships), RELATIONSHIP_BATCH_SIZE)
    ]

    print(
        "[DEBUG (consolidate_graph)]: Total batches: ",
        len(batches),
    )

    for batch in batches:

        janitor_agent = JanitorAgent(
            llm_adapter=llm_adapter,
            kg=graph_adapter,
            vector_store=vector_store_adapter,
            embeddings=embeddings_adapter,
            database_desc=graph_adapter.graphdb_description,
        )

        tasks = janitor_agent.run_graph_consolidator(
            batch,
            brain_id=brain_id,
            timeout=300,
            max_retries=3,
        )

        print(
            "[DEBUG (consolidate_graph)]: Janitor analysis for batch: ",
            tasks,
        )

        kg_agent = KGAgent(
            llm_adapter=llm_small_adapter,
            cache_adapter=cache_adapter,
            kg=graph_adapter,
            vector_store=vector_store_adapter,
            embeddings=embeddings_adapter,
            database_desc=graph_adapter.graphdb_description,
        )
        for task in tasks:
            try:
                kg_agent.run_graph_consolidator_operator(
                    task, brain_id=brain_id, reuse_agent=True
                )
            except Exception as e:
                print(
                    "[DEBUG (consolidate_graph)]: Exception error for kg_agent.run_graph_consolidator_operator(task, brain_id=brain_id)"
                )
                print(e)
                continue
