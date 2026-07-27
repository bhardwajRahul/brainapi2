import hashlib
from typing import Optional

from src.utils.dates import normalize_date_string


def stable_uuid(*parts: Optional[str]) -> str:
    material = "|".join((part or "").strip().lower() for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def stable_node_id(
    name: Optional[str],
    entity_type: Optional[str],
    happened_at: Optional[str] = None,
    supplied_uuid: Optional[str] = None,
) -> str:
    if supplied_uuid and str(supplied_uuid).strip():
        return str(supplied_uuid).strip()
    normalized_type = (entity_type or "").strip().lower()
    parts = [(name or "").strip().lower(), normalized_type]
    if normalized_type == "event":
        parts.append(normalize_date_string(happened_at) or "")
    return stable_uuid("node", *parts)


def stable_relationship_id(
    tail_uuid: str,
    predicate: str,
    tip_uuid: str,
    flow_key: Optional[str] = None,
) -> str:
    parts = [
        "rel",
        tail_uuid or "",
        (predicate or "").strip().upper(),
        tip_uuid or "",
    ]
    if flow_key:
        parts.append(flow_key)
    return stable_uuid(*parts)


def stable_flow_key(
    event_uuid: Optional[str] = None,
    event_name: Optional[str] = None,
    event_type: Optional[str] = None,
    happened_at: Optional[str] = None,
    supplied: Optional[str] = None,
) -> str:
    if supplied and str(supplied).strip():
        return str(supplied).strip()
    if event_uuid and str(event_uuid).strip():
        return str(event_uuid).strip()
    return stable_uuid(
        "flow",
        event_name,
        event_type,
        normalize_date_string(happened_at) if happened_at else None,
    )
