from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


BEHAVIOR_MAP: dict[str, tuple[str, str]] = {
    "purchase": ("Purchase", "TARGETED"),
    "purchased": ("Purchase", "TARGETED"),
    "buy": ("Purchase", "TARGETED"),
    "bought": ("Purchase", "TARGETED"),
    "view": ("View", "TARGETED"),
    "viewed": ("View", "TARGETED"),
    "click": ("View", "TARGETED"),
    "clicked": ("View", "TARGETED"),
    "cart": ("AddToCart", "TARGETED"),
    "add_to_cart": ("AddToCart", "TARGETED"),
    "added_to_cart": ("AddToCart", "TARGETED"),
    "addtocart": ("AddToCart", "TARGETED"),
}


def user_uuid(user_id: str) -> str:
    return f"user:{user_id}"


def item_uuid(item_id: str) -> str:
    return f"item:{item_id}"


def event_uuid(user_id: str, item_id: str, seq: int) -> str:
    return f"evt:{user_id}:{item_id}:{seq}"


def rel_uuid(user_id: str, item_id: str, kind: str, seq: int) -> str:
    return f"rel:{user_id}:{item_id}:{kind}:{seq}"


def normalize_behavior(behavior: str) -> tuple[str, str]:
    key = (behavior or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in BEHAVIOR_MAP:
        return BEHAVIOR_MAP[key]
    label = (behavior or "Interaction").strip() or "Interaction"
    return (label[:1].upper() + label[1:], "TARGETED")


def format_happened_at(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    raw = timestamp.strip()
    if not raw:
        return None
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


def parse_timestamp_for_sort(timestamp: str | None) -> float:
    if not timestamp:
        return 0.0
    raw = timestamp.strip()
    formatted = format_happened_at(raw)
    if formatted and len(formatted) == 10 and formatted[2] == "/":
        try:
            return datetime.strptime(formatted, "%m/%d/%Y").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def interaction_to_triple(
    interaction: dict[str, Any],
    *,
    seq: int,
) -> dict[str, Any]:
    user_id = str(interaction["user_id"])
    item_id = str(interaction["item_id"])
    event_name, edge_name = normalize_behavior(str(interaction.get("behavior") or "purchase"))
    happened_at = format_happened_at(
        interaction.get("timestamp") or interaction.get("ts")
    )
    event_node: dict[str, Any] = {
        "name": event_name,
        "type": "EVENT",
        "uuid": event_uuid(user_id, item_id, seq),
    }
    if happened_at:
        event_node["happened_at"] = happened_at

    return {
        "subject": {
            "name": user_id,
            "type": "USER",
            "uuid": user_uuid(user_id),
        },
        "subj_event": {
            "name": "MADE",
            "uuid": rel_uuid(user_id, item_id, "made", seq),
        },
        "event": event_node,
        "event_obj": {
            "name": edge_name,
            "uuid": rel_uuid(user_id, item_id, "tgt", seq),
        },
        "object": {
            "name": item_id,
            "type": "PRODUCT",
            "uuid": item_uuid(item_id),
        },
    }


def catalog_triple(
    item_id: str,
    *,
    category: str | None = None,
    brand: str | None = None,
    color: str | None = None,
    material: str | None = None,
    seq: int = 0,
) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    facets = (
        ("category", category, "CATEGORY", "Categorized", "IN_CATEGORY", "cat"),
        ("brand", brand, "BRAND", "Branded", "OF_BRAND", "brand"),
        ("color", color, "COLOR", "HasColor", "OF_COLOR", "color"),
        ("material", material, "MATERIAL", "HasMaterial", "OF_MATERIAL", "mat"),
    )
    for i, (facet, value, label, event_name, edge_name, kind) in enumerate(facets):
        if not value:
            continue
        val = str(value)
        triples.append(
            {
                "subject": {
                    "name": item_id,
                    "type": "PRODUCT",
                    "uuid": item_uuid(item_id),
                },
                "subj_event": {
                    "name": "HAS",
                    "uuid": rel_uuid(item_id, val, f"{kind}_has", seq + i),
                },
                "event": {
                    "name": event_name,
                    "type": "EVENT",
                    "uuid": event_uuid(item_id, f"{kind}:{val}", seq + i),
                },
                "event_obj": {
                    "name": edge_name,
                    "uuid": rel_uuid(item_id, val, kind, seq + i),
                },
                "object": {
                    "name": val,
                    "type": label,
                    "uuid": f"attr:{facet}:{val.strip().lower()}",
                },
            }
        )
    return triples


def interactions_to_triples(
    interactions: Sequence[dict[str, Any]],
    *,
    include_catalog: bool = True,
    seq_start: int = 1,
) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    seen_catalog: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
    for i, row in enumerate(interactions):
        seq = seq_start + i
        triples.append(interaction_to_triple(row, seq=seq))
        if include_catalog:
            item_id = str(row["item_id"])
            category = row.get("category")
            brand = row.get("brand")
            color = row.get("color")
            material = row.get("material")
            key = (
                item_id,
                category,
                brand,
                color,
                material,
            )
            if key not in seen_catalog and (category or brand or color or material):
                seen_catalog.add(key)
                triples.extend(
                    catalog_triple(
                        item_id,
                        category=str(category) if category else None,
                        brand=str(brand) if brand else None,
                        color=str(color) if color else None,
                        material=str(material) if material else None,
                        seq=seq,
                    )
                )
    return triples


def structured_ingest_body(
    triples: Sequence[dict[str, Any]],
    *,
    brain_id: str = "demorecsys",
    mode: str = "deterministic",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "brain_id": brain_id,
        "data": list(triples),
    }


def group_interactions_by_user(
    interactions: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in interactions:
        uid = str(row["user_id"])
        by_user.setdefault(uid, []).append(row)
    for uid, rows in by_user.items():
        rows.sort(
            key=lambda r: (
                parse_timestamp_for_sort(r.get("timestamp") or r.get("ts")),
                str(r.get("item_id") or ""),
            )
        )
    return by_user


def leave_one_out_splits(
    interactions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for user_id, rows in group_interactions_by_user(interactions).items():
        if len(rows) < 2:
            continue
        train = rows[:-1]
        holdout = rows[-1]
        splits.append(
            {
                "user_id": user_id,
                "train": train,
                "holdout": holdout,
                "holdout_item_id": str(holdout["item_id"]),
                "holdout_item_uuid": item_uuid(str(holdout["item_id"])),
            }
        )
    return splits
