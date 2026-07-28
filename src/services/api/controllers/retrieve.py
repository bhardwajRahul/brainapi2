"""
File: /retrieve.py
Created Date: Sunday October 26th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
import json
import threading
from typing import Any, Optional

from fastapi import HTTPException
from starlette.responses import JSONResponse

from src.constants.embeddings import Vector
from src.constants.kg import IdentificationParams, Node, Predicate
from src.core.search.entities import search_entities
from src.core.search.entity_info import EventSynergyRetriever, MatchPath
from src.core.search.fact_filter import filter_relevant_facts, reciprocal_rank_fusion
from src.core.search.relationships import search_relationships
from src.utils.vector_search import VectorSearchFacade
from src.services.api.constants.requests import (
    GetContextRequestBody,
    GetContextResponse,
    GetContextTriple,
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
)
from src.services.kg_agent.main import graph_adapter, kg_agent
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter
from src.utils.similarity.vectors import cosine_similarity
from src.utils.nlp.ner import _entity_extractor

vector_search = VectorSearchFacade(vector_store_adapter)

_MAX_NOUN_CHUNKS = 5
_MAX_DOSSIER_ENTITIES = 5
_DOSSIER_MAX_DEPTH = 3
_SEED_K = 20
_PASSAGE_K = 12


async def retrieve_data(
    text: str, limit: int, preferred_entities: str, brain_id: str = "default"
) -> RetrieveRequestResponse:
    """
    Retrieve data from the knowledge graph and data store.
    """
    if preferred_entities:
        preferred_entities = [
            e.strip() for e in preferred_entities.split(",") if e.strip()
        ]
    else:
        preferred_entities = []

    def _get_data():
        text_embeddings = embeddings_adapter.embed_text(text)

        data_vectors = vector_search.search_data(
            text_embeddings.embeddings,
            brain_id=brain_id,
            k=limit,
        )
        triple_vectors = vector_search.search_triplets(
            text_embeddings.embeddings,
            brain_id=brain_id,
            k=limit,
        )

        search_result = data_adapter.search(text, brain_id)

        ts_text_chunks = search_result.text_chunks
        ts_observations = search_result.observations

        v_text_chunks, v_observations = data_adapter.get_text_chunks_by_ids(
            [dv.metadata.get("resource_id") for dv in data_vectors], True, brain_id
        )

        node_ids = [
            node_id
            for tv in triple_vectors
            for node_id in tv.metadata.get("node_ids", [])
        ]
        nodes = graph_adapter.get_nodes_by_uuid(
            uuids=node_ids,
            brain_id=brain_id,
            with_relationships=True,
            relationships_depth=1,
            relationships_type=[
                tv.metadata.get("predicate")
                for tv in triple_vectors
                if tv.metadata.get("predicate", None)
            ],
            preferred_labels=preferred_entities or [],
        )
        return ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes

    ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes = (
        await asyncio.to_thread(_get_data)
    )

    return RetrieveRequestResponse(
        data=[*ts_text_chunks, *v_text_chunks],
        observations=[*ts_observations, *v_observations],
        relationships=nodes,
    )


async def retrieve_neighbors(
    uuid: Optional[str] = None,
    look_for: Optional[str] = None,
    identification_params: Optional[IdentificationParams] = None,
    limit: int = 10,
    brain_id: str = "default",
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighboring nodes related to a specified main node.

    If `uuid` is provided it is used to locate the main node; otherwise `identification_params` is used. Optionally filters first-degree neighbors by semantic similarity to `look_for`, expands those matches to similar nodes, and returns a deduplicated list of neighbor nodes (up to `limit`) with their relationship and the most common matching similar node.

    Parameters:
        uuid (Optional[str]): UUID of the main node to retrieve neighbors for. If omitted, `identification_params` must be provided.
        look_for (Optional[str]): Text used to filter first-degree neighbors by embedding similarity before expanding to similar nodes.
        identification_params (Optional[IdentificationParams]): Identification parameters used to find the main node when `uuid` is not provided.
        limit (int): Maximum number of neighbor results to include in the response.
        brain_id (str): Identifier of the brain / dataset to query.

    Returns:
        RetrieveNeighborsRequestResponse: Object containing:
          - count: total number of unique neighbors found,
          - main_node: the resolved main Node,
          - neighbors: list of RetrievedNeighborNode objects (neighbor, relationship, most_common) limited to `limit`.

    Raises:
        HTTPException: 404 if the main node cannot be found.
    """

    async def _get_neighbors():

        # ---------------------------------------------------------
        # ================= GETTING THE MAIN NODE =================
        # ---------------------------------------------------------
        def _get_node() -> Node:
            node = None
            if uuid:
                node = graph_adapter.get_by_uuid(uuid, brain_id)
            elif identification_params:
                node = graph_adapter.get_by_identification_params(
                    identification_params,
                    brain_id=brain_id,
                    entity_types=identification_params.entity_types,
                )
            if not node:
                raise HTTPException(status_code=404, detail="Entity not found")
            return node

        node = await asyncio.to_thread(_get_node)
        target_node_types = node.labels

        looking_for_v = embeddings_adapter.embed_text(look_for) if look_for else None

        # ---------------------------------------------------------
        # ===== Getting 1st degree neighbors of the main node =====
        # ---------------------------------------------------------
        def _get_fd_neighbors() -> (
            tuple[dict[str, list[tuple[Predicate, Node]]], list[str]]
        ):
            fd_neighbors = graph_adapter.get_neighbors(
                [node.uuid], limit=limit, brain_id=brain_id
            )
            fd_v_neighbors_ids = [
                fd[1].properties.get("v_id")
                for fd in fd_neighbors[node.uuid]
                if fd[1].properties.get("v_id") is not None
            ]
            if look_for:
                fd_v_neighbors_embeddings = vector_store_adapter.get_by_ids(
                    fd_v_neighbors_ids, brain_id=brain_id, store="nodes"
                )
                fd_v_neighbors_embeddings_map = {
                    v.id: v.embeddings
                    for v in fd_v_neighbors_embeddings
                    if (
                        cosine_similarity(looking_for_v.embeddings, v.embeddings) > 0.5
                        and v.id
                        and not v.id.replace(
                            "-", ""
                        ).isalpha()  # likely not a UUID if all numeric (may have hyphens for uuid standard)
                    )
                }
                fd_v_neighbors_ids = list(fd_v_neighbors_embeddings_map.keys())

            return fd_neighbors, fd_v_neighbors_ids

        fd_neighbors, fd_v_neighbors_ids = await asyncio.to_thread(_get_fd_neighbors)

        # ---------------------------------------------------------
        # === Getting nodes similar to the 1st degree neighbors ===
        # ---------------------------------------------------------
        fd_v_similar_node_futures = []
        for fd_v_neighbor_id in fd_v_neighbors_ids:
            fd_v_similar_node_futures.append(
                asyncio.to_thread(
                    vector_store_adapter.search_similar_by_ids,
                    [fd_v_neighbor_id],
                    brain_id,
                    "nodes",
                    0.5,
                    limit,
                )
            )
        fd_v_similar_nodes_results: list[dict[str, list[Vector]]] = (
            await asyncio.gather(*fd_v_similar_node_futures)
        )
        fd_similar_node_ids = [
            v.metadata.get("uuid")
            for result_dict in fd_v_similar_nodes_results
            for vectors in result_dict.values()
            for v in vectors
            if v.metadata.get("uuid") is not None
        ]
        fd_similar_nodes = await asyncio.to_thread(
            graph_adapter.get_by_uuids, fd_similar_node_ids, brain_id
        )
        fd_similar_nodes_by_uuid = {n.uuid: n for n in fd_similar_nodes}

        # ---------------------------------------------------------
        # === Getting neighbors of the 1st degree similar nodes ===
        # ---------------------------------------------------------
        def _get_fd_similar_node_neighbors() -> dict[str, list[tuple[Predicate, Node]]]:
            fd_similar_node_neighbors = graph_adapter.get_neighbors(
                fd_similar_node_ids,
                limit=limit,
                brain_id=brain_id,
                of_types=list(set(target_node_types)),
            )
            return fd_similar_node_neighbors

        fd_similar_node_neighbors = await asyncio.to_thread(
            _get_fd_similar_node_neighbors
        )

        seen_neighbor_uuids = set()
        unique_neighbors = []
        for source_uuid, neighbors_list in fd_similar_node_neighbors.items():
            for neighbor_tuple in neighbors_list:
                neighbor_uuid = neighbor_tuple[1].uuid
                if neighbor_uuid not in seen_neighbor_uuids:
                    seen_neighbor_uuids.add(neighbor_uuid)
                    unique_neighbors.append(
                        RetrievedNeighborNode(
                            neighbor=neighbor_tuple[1],
                            relationship=neighbor_tuple[0],
                            most_common=fd_similar_nodes_by_uuid.get(source_uuid),
                        )
                    )

        return RetrieveNeighborsRequestResponse(
            count=len(unique_neighbors),
            main_node=node,
            neighbors=unique_neighbors[:limit],
        )

    return await _get_neighbors()


async def retrieve_neighbors_ai_mode(
    identification_params: IdentificationParams,
    looking_for: Optional[list[str]],
    limit: int,
    brain_id: str = "default",
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """

    def _get_neighbors():
        node = graph_adapter.get_by_identification_params(
            identification_params,
            brain_id=brain_id,
            entity_types=identification_params.entity_types,
        )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        result = kg_agent.retrieve_neighbors(node, looking_for, limit, brain_id)

        ids = [neighbor.uuid for neighbor in result.neighbors]
        descriptions = [neighbor.description for neighbor in result.neighbors]

        nodes = graph_adapter.get_nodes_by_uuid(uuids=ids, brain_id=brain_id)
        paired = list(zip(nodes, descriptions))

        return RetrieveNeighborsRequestResponse(neighbors=paired)

    result = await asyncio.to_thread(_get_neighbors)

    return result


async def get_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[list[str]] = None,
    from_node_labels: Optional[list[str]] = None,
    to_node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
    brain_id: str = "default",
):
    """
    Retrieve relationships from the knowledge graph with optional filtering and pagination.

    Parameters:
        relationship_types (list[str], optional): Filter results to specific relationship types.
        from_node_labels (list[str], optional): Filter relationships originating from nodes with these labels.
        to_node_labels (list[str], optional): Filter relationships targeting nodes with these labels.
        query_text (str, optional): Text to search within relationship or node content.
        query_search_target (str, optional): Field to target for text search; commonly "all", "from", or "to".
        limit (int, optional): Maximum number of relationships to return.
        skip (int, optional): Number of relationships to skip (offset).
        brain_id (str, optional): Identifier of the brain/graph to query.

    Returns:
        JSONResponse: A response whose JSON content contains:
            - message: Confirmation string.
            - relationships: List of serialized relationship objects.
            - total: Total number of matching relationships.
    """
    relationships = await asyncio.to_thread(
        search_relationships,
        limit,
        skip,
        relationship_types,
        from_node_labels,
        to_node_labels,
        query_text,
        query_search_target,
        brain_id,
    )

    return JSONResponse(
        content={
            "message": "Relationships retrieved successfully",
            "relationships": [r.model_dump(mode="json") for r in relationships.results],
            "total": relationships.total,
        }
    )


async def get_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Retrieve entities from the knowledge graph with optional label and text filters.

    Parameters:
        limit (int): Maximum number of entities to return (pagination).
        skip (int): Number of entities to skip (pagination offset).
        node_labels (Optional[list[str]]): If provided, only return entities whose labels match any value in this list.
        query_text (Optional[str]): If provided, filter entities by matching text content.
        brain_id (str): Identifier of the knowledge graph/brain to query.

    Returns:
        JSONResponse: Object containing:
            - message (str): Informational message.
            - entities (list): Serialized entity objects.
            - total (int): Total number of matching entities.
    """
    entities = await asyncio.to_thread(
        search_entities, limit, skip, node_labels, query_text, brain_id
    )

    return JSONResponse(
        content={
            "message": "Entities retrieved successfully",
            "entities": [e.model_dump(mode="json") for e in entities.results],
            "total": entities.total,
        }
    )


def _format_kg_item(item: Any) -> str:
    name = getattr(item, "name", None) or ""
    description = getattr(item, "description", None) or ""
    happened_at = getattr(item, "happened_at", None) or ""
    base = f"{name}: {description}" if description else name
    if happened_at:
        return f"{base} @{happened_at}"
    return base


def _format_event_fact(n: Node, r: Predicate, m: Node, r2: Predicate, b: Node) -> str:
    return " | ".join(_format_kg_item(item) for item in (n, r, m, r2, b))


def _fact_key(r: Predicate, r2: Predicate) -> tuple[str, str]:
    return (getattr(r, "uuid", None) or "", getattr(r2, "uuid", None) or "")


def _is_currently_valid(predicate: Predicate) -> bool:
    props = getattr(predicate, "properties", None) or {}
    if props.get("invalid_at"):
        return False
    if getattr(predicate, "deprecated", False):
        return False
    return True


def _collect_query_variants(text: str, elements) -> list[str]:
    seen: set[str] = set()
    variants: list[str] = []

    def _add(value: str | None) -> None:
        if not value:
            return
        cleaned = value.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(cleaned)

    _add(text)
    for token in getattr(elements, "tokens", None) or []:
        if isinstance(token, dict):
            _add(token.get("text"))
    noun_chunks = getattr(elements, "noun_chunks", None) or []
    added_chunks = 0
    for chunk in noun_chunks:
        if added_chunks >= _MAX_NOUN_CHUNKS:
            break
        before = len(variants)
        _add(chunk if isinstance(chunk, str) else None)
        if len(variants) > before:
            added_chunks += 1
    return variants


def _seed_nodes_for_text(text: str, brain_id: str) -> list[tuple[str, float, str]]:
    """Return (node_uuid, distance, entity_name) seeds from nodes + relationships."""
    text_embeddings = embeddings_adapter.embed_text(text)
    embeddings = text_embeddings.embeddings
    seeds: list[tuple[str, float, str]] = []

    node_vectors = vector_search.search_nodes(
        embeddings,
        brain_id=brain_id,
        k=_SEED_K,
    )
    for vector in node_vectors:
        meta = vector.metadata or {}
        uuid = meta.get("uuid")
        if not uuid:
            continue
        distance = (
            vector.distance if vector.distance is not None else float("inf")
        )
        name = meta.get("name") or text
        seeds.append((uuid, float(distance), str(name)))

    rel_vectors = vector_search.search_relationships(
        embeddings,
        brain_id=brain_id,
        k=_SEED_K,
    )
    for vector in rel_vectors:
        meta = vector.metadata or {}
        distance = (
            vector.distance if vector.distance is not None else float("inf")
        )
        node_ids = meta.get("node_ids") or []
        for node_id in node_ids:
            if node_id:
                seeds.append((str(node_id), float(distance), text))

    return seeds


def _extract_source_chunk_ids(*items: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        props = getattr(item, "properties", None) or {}
        for value in props.get("source_chunk_ids") or []:
            chunk_id = str(value).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ids.append(chunk_id)
        single = props.get("source_chunk_id")
        if single:
            chunk_id = str(single).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ids.append(chunk_id)
    return ids


def _retrieve_passages(
    text: str, brain_id: str, *, limit: int
) -> list[tuple[str, float, str]]:
    """Return (chunk_id, score, text) ranked passages via vector + keyword fusion."""
    text_embeddings = embeddings_adapter.embed_text(text)
    vector_hits = vector_search.search_data(
        text_embeddings.embeddings,
        brain_id=brain_id,
        k=max(limit, _PASSAGE_K),
    )
    vector_ids: list[str] = []
    id_to_text: dict[str, str] = {}
    id_to_distance: dict[str, float] = {}
    for vector in vector_hits:
        meta = vector.metadata or {}
        resource_id = meta.get("resource_id") or getattr(vector, "id", None)
        if not resource_id:
            continue
        resource_id = str(resource_id)
        vector_ids.append(resource_id)
        id_to_distance[resource_id] = (
            float(vector.distance)
            if vector.distance is not None
            else float("inf")
        )

    keyword_ids: list[str] = []
    try:
        search_result = data_adapter.search(text, brain_id)
        for chunk in getattr(search_result, "text_chunks", None) or []:
            chunk_id = str(getattr(chunk, "id", "") or "")
            if not chunk_id:
                continue
            keyword_ids.append(chunk_id)
            id_to_text[chunk_id] = getattr(chunk, "text", "") or ""
    except Exception:
        pass

    fused = reciprocal_rank_fusion([vector_ids, keyword_ids])
    ranked_ids = [item for item, _ in fused[:limit]]
    missing = [cid for cid in ranked_ids if cid not in id_to_text]
    if missing:
        try:
            chunks, _ = data_adapter.get_text_chunks_by_ids(
                missing, False, brain_id
            )
            for chunk in chunks:
                id_to_text[str(chunk.id)] = chunk.text or ""
        except Exception:
            pass

    passages: list[tuple[str, float, str]] = []
    for chunk_id in ranked_ids:
        body = (id_to_text.get(chunk_id) or "").strip()
        if not body:
            continue
        score = id_to_distance.get(chunk_id, 1.0 / (1.0 + len(passages)))
        passages.append((chunk_id, float(score), body))
    return passages


def _flatten_match_path(path: MatchPath) -> list[str]:
    lines: list[str] = []
    current: MatchPath | None = path
    while current is not None:
        predicate, node = current.path
        if node is None and not getattr(predicate, "name", None):
            break
        parts = []
        if predicate and getattr(predicate, "name", None):
            parts.append(_format_kg_item(predicate))
        if node is not None:
            parts.append(_format_kg_item(node))
        if parts:
            lines.append(" -> ".join(parts))
        if current.children:
            current = current.children[0]
        else:
            current = None
    return lines


async def get_context(request: GetContextRequestBody) -> GetContextResponse:
    """
    Retrieve contextual information for a text.
    """

    elements = _entity_extractor.extract_elements(request.text)
    variants = _collect_query_variants(request.text, elements)
    historical_context: list[str] = []
    source_passages: list[str] = []
    candidate_lock = threading.Lock()
    candidates: list[dict[str, Any]] = []
    seed_hits: list[tuple[str, float, str]] = []
    passage_hits: list[tuple[str, float, str]] = []

    def _collect_facts_for_variant(text: str) -> None:
        seeds = _seed_nodes_for_text(text, request.brain_id)
        if not seeds:
            return
        with candidate_lock:
            seed_hits.extend(seeds)
        uuid_to_distance: dict[str, float] = {}
        for uuid, distance, _ in seeds:
            prev = uuid_to_distance.get(uuid)
            if prev is None or distance < prev:
                uuid_to_distance[uuid] = distance
        seed_uuids = list(uuid_to_distance.keys())
        neighbors = graph_adapter.get_event_centric_neighbors(
            seed_uuids, brain_id=request.brain_id
        )
        local: list[dict[str, Any]] = []
        for n, r, m, r2, b in neighbors:
            if not _is_currently_valid(r) or not _is_currently_valid(r2):
                continue
            distances = [
                uuid_to_distance[u]
                for u in (
                    getattr(n, "uuid", None),
                    getattr(m, "uuid", None),
                    getattr(b, "uuid", None),
                )
                if u in uuid_to_distance
            ]
            score = min(distances) if distances else float("inf")
            local.append(
                {
                    "identified_entity": text,
                    "triple": (n, r, m, r2, b),
                    "score": score,
                    "key": _fact_key(r, r2),
                    "text": _format_event_fact(n, r, m, r2, b),
                    "chunk_ids": _extract_source_chunk_ids(n, r, m, r2, b),
                }
            )
        if local:
            with candidate_lock:
                candidates.extend(local)

    def _collect_passages() -> None:
        nonlocal passage_hits
        passage_hits = _retrieve_passages(
            request.text,
            request.brain_id,
            limit=max(1, int(getattr(request, "max_passages", 8) or 8)),
        )

    async def _get_historical_context():
        nonlocal historical_context
        passages = await asyncio.to_thread(
            _retrieve_passages,
            request.text,
            request.brain_id,
            limit=max(request.historical_limit, _PASSAGE_K),
        )
        if passages:
            historical_context = [text for _, _, text in passages[: request.historical_limit]]
            return historical_context
        text_chunks, structured_data = await asyncio.gather(
            asyncio.to_thread(
                data_adapter.get_last_text_chunks,
                brain_id=request.brain_id,
                limit=request.historical_limit,
            ),
            asyncio.to_thread(
                data_adapter.get_last_structured_data,
                brain_id=request.brain_id,
                limit=request.historical_limit,
            ),
        )
        historical_context = [text_chunk.text for text_chunk in text_chunks] + [
            json.dumps(structured_data.data)
            for structured_data in structured_data
            if len(structured_data.data.items()) > 0
        ]
        return historical_context

    futures = [
        asyncio.to_thread(_collect_facts_for_variant, variant) for variant in variants
    ]
    futures.append(asyncio.to_thread(_collect_passages))
    futures.append(_get_historical_context())
    await asyncio.gather(*futures)

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["key"]
        existing = deduped.get(key)
        if existing is None or candidate["score"] < existing["score"]:
            deduped[key] = candidate

    ranked = sorted(deduped.values(), key=lambda c: c["score"])
    max_facts = max(1, int(getattr(request, "max_facts", 40) or 40))

    if getattr(request, "apply_fact_filter", True) and ranked:
        keep = filter_relevant_facts(
            request.text,
            [c["text"] for c in ranked],
            max_keep=max_facts,
        )
        curated = [ranked[i] for i in keep if 0 <= i < len(ranked)]
        if not curated:
            curated = ranked[:max_facts]
    else:
        curated = ranked[:max_facts]

    text_lines: list[str] = []
    triples: list[GetContextTriple] = []
    provenance_ids: list[str] = []
    for candidate in curated:
        n, r, m, r2, b = candidate["triple"]
        text_lines.append(candidate["text"])
        triples.append(
            GetContextTriple(
                identified_entity=candidate["identified_entity"],
                triple=(n, r, m, r2, b),
            )
        )
        provenance_ids.extend(candidate.get("chunk_ids") or [])

    for chunk_id, _, body in passage_hits:
        if body and body not in source_passages:
            source_passages.append(body)
        if chunk_id not in provenance_ids:
            provenance_ids.append(chunk_id)

    if provenance_ids:
        try:
            chunks, _ = await asyncio.to_thread(
                data_adapter.get_text_chunks_by_ids,
                provenance_ids[:40],
                False,
                request.brain_id,
            )
            for chunk in chunks:
                body = (chunk.text or "").strip()
                if body and body not in source_passages:
                    source_passages.append(body)
        except Exception:
            pass

    max_passages = max(1, int(getattr(request, "max_passages", 8) or 8))
    source_passages = source_passages[:max_passages]

    entity_scores: dict[str, float] = {}
    for _uuid, distance, entity_name in seed_hits:
        name = (entity_name or "").strip()
        if not name:
            continue
        if name.lower() == request.text.strip().lower():
            continue
        prev = entity_scores.get(name)
        if prev is None or distance < prev:
            entity_scores[name] = distance
    top_entities = [
        name
        for name, _ in sorted(entity_scores.items(), key=lambda item: item[1])[
            :_MAX_DOSSIER_ENTITIES
        ]
    ]

    dossier_lines: list[str] = []
    if top_entities:

        def _run_dossiers() -> list[str]:
            lines: list[str] = []
            retriever = EventSynergyRetriever(request.brain_id)
            for entity_name in top_entities:
                try:
                    path = retriever.retrieve_matches(
                        entity_name,
                        request.text,
                        max_depth=_DOSSIER_MAX_DEPTH,
                    )
                except Exception:
                    continue
                if path is None or path.target_node is None:
                    continue
                flattened = _flatten_match_path(path)
                if not flattened:
                    continue
                target_name = getattr(path.target_node, "name", None) or entity_name
                lines.append(f"[dossier:{target_name}] " + " | ".join(flattened))
            return lines

        dossier_lines = await asyncio.to_thread(_run_dossiers)

    passage_block = [f"[passage] {p}" for p in source_passages]
    text_context = "\n".join(passage_block + text_lines + dossier_lines)

    return GetContextResponse(
        text_context=text_context,
        triples=triples,
        historical_context=historical_context,
        source_passages=source_passages,
    )
