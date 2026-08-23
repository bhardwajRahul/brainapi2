from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from src.core.search.recommend import behavior_weight

HAS_EVENT_NAME = "HAS"
ENTITY_TYPE = "ENTITY"
CLASS_LABEL = "CLASS"
TYPE_LABEL = "TYPE"
ATTR_LABEL = "ATTR"
EVENT_LABEL = "EVENT"
USER_TYPE = "USER"
OPTION_ALLOWLIST = frozenset(
    {"color", "material", "category", "brand", "size", "style"}
)
_BEHAVIOR_EVENT = {
    "view": "View",
    "viewed": "View",
    "click": "View",
    "clicked": "View",
    "cart": "AddToCart",
    "add_to_cart": "AddToCart",
    "added_to_cart": "AddToCart",
    "addtocart": "AddToCart",
    "purchase": "Purchase",
    "purchased": "Purchase",
    "buy": "Purchase",
    "bought": "Purchase",
    "favorite": "Favorite",
    "favourite": "Favorite",
    "favorites": "Favorite",
    "favourites": "Favorite",
    "wishlist": "Wishlist",
    "add_to_favorite": "Favorite",
    "addtofavorite": "Favorite",
    "add_to_favourite": "Favorite",
    "addtofavourite": "Favorite",
    "add_to_wishlist": "Wishlist",
    "addtowishlist": "Wishlist",
    "follow": "Follow",
    "flw": "Follow",
}
FEATURE_KEY_CAP = 16
FEATURE_KEY_PRIORITY = (
    "style",
    "mood",
    "color",
    "colour",
    "material",
    "shape",
    "finish",
    "fabric",
    "design",
    "woodtone",
)
FEATURE_SKIP_TOKENS = (
    "width",
    "height",
    "weight",
    "length",
    "depth",
    "capacity",
    "temperature",
    "rating",
    "price",
    "count",
    "warranty",
    "origin",
    "assembly",
    "thickness",
    "diameter",
    "clearance",
    "sku",
)
CONTINUOUS_PROPERTY_KEYS = (
    "price",
    "rating",
    "average_rating",
    "rating_count",
    "review_count",
    "width",
    "height",
    "depth",
    "weight",
    "length",
)
BOOLEAN_NOISE = {
    "",
    "yes",
    "no",
    "true",
    "false",
    "y",
    "n",
    "0",
    "1",
    "none",
    "n/a",
    "na",
    "unknown",
    "null",
}
_LABEL_RE = re.compile(r"[^A-Za-z0-9]+")
_TEXT_FIELD_RE = re.compile(
    r"^(Title|Brand|Color|Class|Category|Features|Locale|Hierarchy|Description|Price|Rating)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_HIER_SPLIT = re.compile(r"\s*[>/|]+\s*")


def entity_uuid(doc_id: str) -> str:
    return str(doc_id)


def hub_uuid(kind: str, value: str) -> str:
    return f"hub:{kind}:{value.strip().lower()}"


def event_uuid(owner: str, kind: str, seq: int) -> str:
    return f"evt:{owner}:{kind}:{seq}"


def rel_uuid(owner: str, kind: str, seq: int) -> str:
    return f"rel:{owner}:{kind}:{seq}"


def sanitize_label(raw: str) -> str:
    cleaned = _LABEL_RE.sub("_", (raw or "").strip().upper()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        return ATTR_LABEL
    return cleaned[:40]


_DOC_ID_RE = re.compile(r"DOCID\s+(\S+)")


def compose_search_text(*parts: str | None) -> str:
    chunks = [str(part).strip() for part in parts if str(part or "").strip()]
    return " ".join(chunks)


def doc_id_from_text(text: str | None) -> str | None:
    match = _DOC_ID_RE.search(str(text or ""))
    if not match:
        return None
    doc_id = match.group(1).strip().rstrip(".")
    return doc_id or None


def node_id_from_passage_text(text: str | None) -> str | None:
    doc_id = doc_id_from_text(text)
    if not doc_id:
        return None
    return entity_uuid(doc_id)


def entity_search_text(doc: dict[str, Any], fields: dict[str, Any] | None = None) -> str:
    raw = str(doc.get("text") or "").strip()
    if raw:
        return raw
    resolved = fields if fields is not None else doc_fields(doc)
    return compose_search_text(
        resolved.get("title"),
        resolved.get("class"),
        resolved.get("hierarchy"),
        resolved.get("brand"),
        resolved.get("color"),
        resolved.get("features"),
        resolved.get("description"),
    )


def node_embed_text(node_data: Any) -> tuple[str, str]:
    props = getattr(node_data, "properties", None) or {}
    if not isinstance(props, dict):
        props = {}
    search_text = str(props.get("search_text") or "").strip()
    name = str(getattr(node_data, "name", "") or "").strip()
    if search_text:
        uuid = str(getattr(node_data, "uuid", "") or "").strip()
        return search_text, f"uuid:{uuid}" if uuid else search_text
    return name, name


def split_hierarchy(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [item.strip() for item in _HIER_SPLIT.split(text) if item.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def _feature_key_rank(key: str) -> int:
    lowered = key.lower().replace(" ", "")
    for index, token in enumerate(FEATURE_KEY_PRIORITY):
        if token in lowered:
            return index
    return len(FEATURE_KEY_PRIORITY)


def _skip_feature_key(key: str) -> bool:
    lowered = key.lower().replace(" ", "")
    if not lowered:
        return True
    return any(token in lowered for token in FEATURE_SKIP_TOKENS)


def parse_feature_string(
    raw: str | None, *, cap: int = FEATURE_KEY_CAP
) -> list[tuple[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if ":" in text and not parts:
        parts = [text]
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if _skip_feature_key(key):
            continue
        if value.lower() in BOOLEAN_NOISE:
            continue
        item = (key, value)
        if item in seen:
            continue
        seen.add(item)
        pairs.append(item)
    pairs.sort(key=lambda item: (_feature_key_rank(item[0]), item[0].lower()))
    if cap <= 0:
        return pairs
    kept: list[tuple[str, str]] = []
    keys: set[str] = set()
    for key, value in pairs:
        key_id = key.lower()
        if key_id not in keys and len(keys) >= cap:
            continue
        keys.add(key_id)
        kept.append((key, value))
    return kept


def _fields_from_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = _TEXT_FIELD_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if value:
            out[key] = value
    return out


def doc_fields(doc: dict[str, Any]) -> dict[str, Any]:
    text_fields = _fields_from_text(str(doc.get("text") or ""))
    title = str(doc.get("title") or text_fields.get("title") or "").strip()
    class_name = str(
        doc.get("class")
        or doc.get("product_class")
        or text_fields.get("class")
        or ""
    ).strip()
    hierarchy = str(
        doc.get("hierarchy")
        or doc.get("category_hierarchy")
        or text_fields.get("hierarchy")
        or text_fields.get("category")
        or ""
    ).strip()
    if not class_name:
        class_name = str(text_fields.get("category") or "").strip()
    brand = str(doc.get("brand") or text_fields.get("brand") or "").strip()
    color = str(doc.get("color") or text_fields.get("color") or "").strip()
    locale = str(doc.get("locale") or text_fields.get("locale") or "").strip()
    features = str(doc.get("features") or text_fields.get("features") or "").strip()
    description = str(
        doc.get("description") or text_fields.get("description") or ""
    ).strip()
    dataset = str(doc.get("dataset") or "").strip()
    properties: dict[str, str] = {}
    for key in CONTINUOUS_PROPERTY_KEYS:
        value = str(doc.get(key) or text_fields.get(key) or "").strip()
        if value:
            properties[key] = value
    return {
        "title": title,
        "class": class_name,
        "hierarchy": hierarchy,
        "brand": brand,
        "color": color,
        "locale": locale,
        "features": features,
        "description": description,
        "dataset": dataset,
        "properties": properties,
    }


def _triple_pred_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("name") or "").strip().upper()
    return str(getattr(value, "name", "") or "").strip().upper()


def is_static_has_triple(triple: Any) -> bool:
    if isinstance(triple, dict):
        event = triple.get("event")
        subj_event = triple.get("subj_event")
        event_obj = triple.get("event_obj")
    else:
        event = getattr(triple, "event", None)
        subj_event = getattr(triple, "subj_event", None)
        event_obj = getattr(triple, "event_obj", None)
    happened = None
    event_name = ""
    if isinstance(event, dict):
        happened = event.get("happened_at")
        event_name = str(event.get("name") or "").strip().upper()
    elif event is not None:
        happened = getattr(event, "happened_at", None)
        event_name = str(getattr(event, "name", "") or "").strip().upper()
    if happened:
        return False
    pred = _triple_pred_name(subj_event) or HAS_EVENT_NAME
    if event is None:
        return pred == HAS_EVENT_NAME
    other = _triple_pred_name(event_obj) or HAS_EVENT_NAME
    return event_name == HAS_EVENT_NAME and pred == HAS_EVENT_NAME and other == HAS_EVENT_NAME


def has_triple(
    *,
    subject: dict[str, Any],
    object_node: dict[str, Any],
    seq: int,
    kind: str,
) -> dict[str, Any]:
    owner = str(subject.get("uuid") or subject.get("name") or "node")
    return {
        "subject": subject,
        "subj_event": {
            "name": HAS_EVENT_NAME,
            "uuid": rel_uuid(owner, f"{kind}_has", seq),
        },
        "object": object_node,
    }


def _entity_node(
    doc_id: str,
    name: str,
    *,
    description: str | None = None,
    properties: dict[str, Any] | None = None,
    search_text: str | None = None,
) -> dict[str, Any]:
    props = dict(properties or {})
    blob = str(search_text or "").strip() or compose_search_text(name, description)
    if blob:
        props["search_text"] = blob
    node: dict[str, Any] = {
        "name": name or doc_id,
        "type": ENTITY_TYPE,
        "uuid": entity_uuid(doc_id),
        "labels": [ENTITY_TYPE],
        "properties": props,
    }
    if description:
        node["description"] = description
    return node


def _hub_node(kind: str, value: str, extra_label: str | None = None) -> dict[str, Any]:
    labels = [kind]
    if extra_label:
        sanitized = sanitize_label(extra_label)
        if sanitized and sanitized not in labels:
            labels.append(sanitized)
    search_text = compose_search_text(extra_label or kind, value)
    return {
        "name": value,
        "type": kind,
        "uuid": hub_uuid(kind.lower(), value),
        "labels": labels,
        "description": search_text,
        "properties": {
            "search_text": search_text,
            "catalog_labels": labels,
        },
    }


def doc_to_triples(doc: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = str(doc.get("doc_id") or "").strip()
    if not doc_id:
        return []
    fields = doc_fields(doc)
    name = fields["title"] or doc_id
    subject = _entity_node(
        doc_id,
        name,
        description=fields["description"] or None,
        properties=dict(fields["properties"] or {}),
        search_text=entity_search_text(doc, fields),
    )
    triples: list[dict[str, Any]] = []
    seq = 0
    class_name = fields["class"]
    if class_name:
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(CLASS_LABEL, class_name),
                seq=seq,
                kind="class",
            )
        )
        seq += 1
    class_key = class_name.strip().lower()
    for part in split_hierarchy(fields["hierarchy"]):
        if part.strip().lower() == class_key:
            continue
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(TYPE_LABEL, part, extra_label="CATEGORY"),
                seq=seq,
                kind="hierarchy",
            )
        )
        seq += 1
    if fields["brand"]:
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(ATTR_LABEL, fields["brand"], extra_label="BRAND"),
                seq=seq,
                kind="brand",
            )
        )
        seq += 1
    if fields["color"]:
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(ATTR_LABEL, fields["color"], extra_label="COLOR"),
                seq=seq,
                kind="color",
            )
        )
        seq += 1
    if fields["locale"]:
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(TYPE_LABEL, fields["locale"], extra_label="LOCALE"),
                seq=seq,
                kind="locale",
            )
        )
        seq += 1
    for key, value in parse_feature_string(fields["features"]):
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(ATTR_LABEL, value, extra_label=key),
                seq=seq,
                kind=f"feat:{sanitize_label(key).lower()}",
            )
        )
        seq += 1

    if not triples:
        dataset = fields["dataset"] or "ITEM"
        triples.append(
            has_triple(
                subject=subject,
                object_node=_hub_node(TYPE_LABEL, dataset),
                seq=seq,
                kind="type",
            )
        )
    return triples


def docs_to_triples(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for doc in docs:
        triples.extend(doc_to_triples(doc))
    return triples


def catalog_entity_backfill_rows(docs: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue
        fields = doc_fields(doc)
        rows.append(
            {
                "uuid": entity_uuid(doc_id),
                "name": str(fields.get("title") or doc_id),
                "search_text": entity_search_text(doc, fields),
            }
        )
    return rows


def format_happened_at(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    raw = str(timestamp).strip()
    if not raw:
        return None
    from datetime import datetime

    if len(raw) == 10 and raw[2] == "/" and raw[5] == "/":
        return raw
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ):
        try:
            parsed = datetime.strptime(
                raw.replace("Z", "+0000") if fmt.endswith("%z") and raw.endswith("Z") else raw,
                fmt,
            )
            return parsed.strftime("%m/%d/%Y")
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%m/%d/%Y")
    except ValueError:
        return raw


def _interaction_event_name(behavior: str) -> str:
    key = (behavior or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _BEHAVIOR_EVENT:
        return _BEHAVIOR_EVENT[key]
    compact = key.replace("_", "")
    for alias, name in _BEHAVIOR_EVENT.items():
        if alias.replace("_", "") == compact:
            return name
    label = (behavior or "Interaction").strip() or "Interaction"
    return label[:1].upper() + label[1:]


def _option_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src in (row.get("options"), row.get("attributes")):
        if not isinstance(src, dict):
            continue
        for key, raw in src.items():
            facet = str(key).strip().lower()
            if facet not in OPTION_ALLOWLIST or raw is None:
                continue
            value = str(raw).strip()
            if not value:
                continue
            pair = (facet, value)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def prefers_triple(
    *,
    user_id: str,
    facet: str,
    value: str,
    seq: int,
    amount: float,
) -> dict[str, Any]:
    extra = "CATEGORY" if facet == "category" else facet.upper()
    kind = CLASS_LABEL if facet == "category" else ATTR_LABEL
    return {
        "subject": {
            "name": user_id,
            "type": USER_TYPE,
            "uuid": f"user:{user_id}",
            "labels": [USER_TYPE],
        },
        "subj_event": {
            "name": "PREFERS",
            "uuid": rel_uuid(user_id, f"prefers:{facet}:{value}", seq),
            "amount": amount,
            "properties": {"facet": facet, "value": value, "weight": amount},
        },
        "object": _hub_node(kind, value, extra_label=extra),
    }


def interaction_to_triples(
    row: dict[str, Any] | Sequence[dict[str, Any]], *, seq: int = 1
) -> list[dict[str, Any]]:
    if not isinstance(row, dict):
        return interactions_to_triples(row, seq_start=seq)
    user_id = str(row.get("user_id") or "").strip()
    item_id = str(row.get("item_id") or "").strip()
    if not user_id or not item_id:
        return []
    behavior = str(row.get("behavior") or "interaction").strip() or "interaction"
    event_name = _interaction_event_name(behavior)
    happened_at = format_happened_at(row.get("timestamp") or row.get("ts"))
    item_name = str(row.get("title") or row.get("item_name") or item_id)
    event_node: dict[str, Any] = {
        "name": event_name,
        "type": EVENT_LABEL,
        "uuid": event_uuid(user_id, f"{item_id}:{behavior}", seq),
        "labels": [EVENT_LABEL],
    }
    if happened_at:
        event_node["happened_at"] = happened_at
    user_node = {
        "name": user_id,
        "type": USER_TYPE,
        "uuid": f"user:{user_id}",
        "labels": [USER_TYPE],
    }
    triples = [
        {
            "subject": user_node,
            "subj_event": {
                "name": "MADE",
                "uuid": rel_uuid(user_id, f"made:{item_id}", seq),
            },
            "event": event_node,
            "event_obj": {
                "name": "TARGETED",
                "uuid": rel_uuid(user_id, f"tgt:{item_id}", seq),
            },
            "object": _entity_node(item_id, item_name),
        }
    ]
    weight = behavior_weight(event_name)
    for facet, value in _option_pairs(row):
        triples.append(
            prefers_triple(
                user_id=user_id,
                facet=facet,
                value=value,
                seq=seq,
                amount=weight,
            )
        )
    catalog_doc = {
        "doc_id": item_id,
        "title": item_name,
        "class": row.get("category") or row.get("class"),
        "brand": row.get("brand"),
        "color": row.get("color"),
        "features": row.get("features"),
        "dataset": "interaction",
    }
    existing = {
        str((t.get("event") or {}).get("uuid") or "")
        for t in triples
        if (t.get("event") or {}).get("uuid")
    }
    for triple in doc_to_triples(catalog_doc):
        event_id = str((triple.get("event") or {}).get("uuid") or "")
        if event_id and event_id in existing:
            continue
        triples.append(triple)
    return triples


def interactions_to_triples(
    rows: Sequence[dict[str, Any]],
    *,
    seq_start: int = 1,
) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        triples.extend(interaction_to_triples(row, seq=seq_start + i))
    return triples


def load_interaction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if not isinstance(obj, dict):
            continue
        if obj.get("user_id") and obj.get("item_id"):
            rows.append(obj)
    return rows


def structured_ingest_body(
    triples: Sequence[dict[str, Any]],
    *,
    brain_id: str,
    mode: str = "deterministic",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "brain_id": brain_id,
        "data": list(triples),
    }


__all__ = [
    "HAS_EVENT_NAME",
    "ENTITY_TYPE",
    "CLASS_LABEL",
    "TYPE_LABEL",
    "ATTR_LABEL",
    "EVENT_LABEL",
    "USER_TYPE",
    "FEATURE_KEY_CAP",
    "compose_search_text",
    "doc_fields",
    "doc_id_from_text",
    "doc_to_triples",
    "entity_search_text",
    "node_id_from_passage_text",
    "docs_to_triples",
    "catalog_entity_backfill_rows",
    "entity_uuid",
    "format_happened_at",
    "has_triple",
    "is_static_has_triple",
    "hub_uuid",
    "interaction_to_triples",
    "prefers_triple",
    "interactions_to_triples",
    "load_interaction_rows",
    "node_embed_text",
    "parse_feature_string",
    "sanitize_label",
    "split_hierarchy",
    "structured_ingest_body",
]
