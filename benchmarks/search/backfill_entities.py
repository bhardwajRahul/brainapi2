from __future__ import annotations

from typing import Any, Callable

from search.config import FROZEN_STRUCTURED_BRAINS, validate_brain_id
from search.mapping import catalog_entity_backfill_rows, entity_uuid


def refuse_entity_backfill(brain_id: str) -> str:
    bid = validate_brain_id(brain_id)
    if bid in FROZEN_STRUCTURED_BRAINS:
        raise SystemExit(
            f"Refusing ENTITY text backfill on frozen brain {bid}. "
            "Use searchbenchwandsgraph only (never wipe frozen structured brains)."
        )
    return bid


def backfill_rows_from_docs(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return catalog_entity_backfill_rows(docs)


def apply_entity_text_backfill(
    *,
    brain_id: str,
    rows: list[dict[str, str]],
    graph: Any,
    embeddings: Any,
    vector_store: Any,
    is_item: Callable[[str, list[str] | None], bool] | None = None,
) -> dict[str, Any]:
    bid = refuse_entity_backfill(brain_id)
    if is_item is None:
        from src.core.search.graph_channels import is_item_entity as is_item
    from src.constants.embeddings import Vector
    from src.core.agents.scout_agent import ScoutEntity
    from src.core.search.catalog_graph import node_embed_text

    updated = 0
    missing = 0
    skipped = 0
    for row in rows:
        uuid = str(row.get("uuid") or entity_uuid(str(row.get("doc_id") or ""))).strip()
        blob = str(row.get("search_text") or "").strip()
        if not uuid or not blob:
            skipped += 1
            continue
        node = graph.get_by_uuid(uuid, bid)
        if node is None:
            missing += 1
            continue
        labels = list(getattr(node, "labels", None) or [])
        if not is_item(uuid, labels):
            skipped += 1
            continue
        props = dict(getattr(node, "properties", None) or {})
        props["search_text"] = blob
        graph.update_node(
            uuid,
            bid,
            new_properties={"search_text": blob},
        )
        scout = ScoutEntity(
            uuid=uuid,
            name=str(getattr(node, "name", "") or row.get("name") or uuid),
            type="ENTITY",
            properties=props,
        )
        text, _ = node_embed_text(scout)
        vector = embeddings.embed_text(text)
        if not isinstance(vector, Vector):
            skipped += 1
            continue
        existing = props.get("v_id")
        if existing:
            vector.id = str(existing)
        vector.metadata = {
            "labels": labels or ["ENTITY"],
            "name": scout.name,
            "uuid": uuid,
        }
        v_ids = vector_store.add_vectors([vector], store="nodes", brain_id=bid)
        if v_ids:
            graph.update_node(
                uuid,
                bid,
                new_properties={"search_text": blob, "v_id": v_ids[0]},
            )
        updated += 1
    return {
        "brain_id": bid,
        "n_rows": len(rows),
        "updated": updated,
        "missing": missing,
        "skipped": skipped,
    }
