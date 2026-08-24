from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.search.catalog_graph import (  # noqa: E402
    ATTR_LABEL,
    CLASS_LABEL,
    ENTITY_TYPE,
    EVENT_LABEL,
    FEATURE_KEY_CAP,
    HAS_EVENT_NAME,
    TYPE_LABEL,
    USER_TYPE,
    compose_search_text,
    doc_fields,
    doc_id_from_text,
    doc_to_triples,
    docs_to_triples,
    catalog_entity_backfill_rows,
    entity_search_text,
    entity_uuid,
    node_id_from_passage_text,
    event_uuid,
    format_happened_at,
    has_triple,
    is_static_has_triple,
    hub_uuid,
    interaction_to_triples,
    interactions_to_triples,
    load_interaction_rows,
    parse_feature_string,
    rel_uuid,
    sanitize_label,
    split_hierarchy,
    structured_ingest_body,
)

__all__ = [
    "ATTR_LABEL",
    "CLASS_LABEL",
    "ENTITY_TYPE",
    "EVENT_LABEL",
    "FEATURE_KEY_CAP",
    "HAS_EVENT_NAME",
    "TYPE_LABEL",
    "USER_TYPE",
    "compose_search_text",
    "doc_fields",
    "doc_id_from_text",
    "doc_to_triples",
    "docs_to_triples",
    "catalog_entity_backfill_rows",
    "entity_search_text",
    "entity_uuid",
    "node_id_from_passage_text",
    "event_uuid",
    "format_happened_at",
    "has_triple",
    "is_static_has_triple",
    "hub_uuid",
    "interaction_to_triples",
    "interactions_to_triples",
    "load_interaction_rows",
    "parse_feature_string",
    "rel_uuid",
    "sanitize_label",
    "split_hierarchy",
    "structured_ingest_body",
]
