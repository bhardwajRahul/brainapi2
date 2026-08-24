from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from src.constants.kg import Node
from src.core.search.entity_sibilings import _predicate_currently_valid
from src.core.search.recommend import (
    _is_attr_node,
    _prefers_weight,
    behavior_weight,
    recency_decay,
)
from src.services.kg_agent.main import graph_adapter

SHORT_TERM_HALF_LIFE_DAYS = 14.0
ITEM_LABELS = frozenset({"ENTITY", "PRODUCT"})
PREF_PREDICATES = frozenset({"PREFERS", "PREFER", "LIKES"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
    }
)


def query_personalize_lambda(query: str) -> float:
    tokens = [
        token.lower()
        for token in _TOKEN_RE.findall(query or "")
        if token.lower() not in _STOP
    ]
    if not tokens:
        return 0.0
    if any(any(char.isdigit() for char in token) for token in tokens):
        return 0.0
    count = len(tokens)
    if count == 1:
        return 0.85
    if count == 2:
        return 0.5
    if count == 3:
        return 0.25
    return 0.1


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 0.0 for key in values}
    span = hi - lo
    return {key: (value - lo) / span for key, value in values.items()}


def blend_ranked(
    ids: List[str],
    retrieve_scores: Dict[str, float],
    pref_scores: Dict[str, float],
    lam: float,
) -> tuple[list[str], dict[str, float]]:
    ordered_ids = list(ids)
    if not ordered_ids:
        return [], {}
    lam = max(0.0, min(1.0, float(lam)))
    retrieve_out = {
        item_id: float(retrieve_scores.get(item_id, 0.0)) for item_id in ordered_ids
    }
    if lam <= 0.0 or not any(
        float(pref_scores.get(item_id, 0.0)) for item_id in ordered_ids
    ):
        return ordered_ids, retrieve_out
    retrieve_norm = _minmax(retrieve_out)
    pref_norm = _minmax(
        {item_id: float(pref_scores.get(item_id, 0.0)) for item_id in ordered_ids}
    )
    blended = {
        item_id: (1.0 - lam) * retrieve_norm[item_id] + lam * pref_norm[item_id]
        for item_id in ordered_ids
    }
    ranked = sorted(
        ordered_ids,
        key=lambda item_id: (-blended[item_id], ordered_ids.index(item_id)),
    )
    return ranked, blended


def _is_item_node(node: Node) -> bool:
    labels = {str(label).upper() for label in (node.labels or [])}
    return bool(labels & ITEM_LABELS)


def _neighbors(uuid: str, brain_id: str, of_types: Optional[list[str]] = None):
    try:
        payload = graph_adapter.get_neighbors(
            [uuid], of_types=of_types, brain_id=brain_id
        )
    except TypeError:
        payload = graph_adapter.get_neighbors([uuid], brain_id=brain_id)
    except Exception:
        return []
    return payload.get(uuid, []) if isinstance(payload, dict) else []


def _attr_nodes_from_item(item: Node, brain_id: str) -> List[Node]:
    attrs: List[Node] = []
    for pred, neighbor in _neighbors(item.uuid, brain_id):
        if not _predicate_currently_valid(pred):
            continue
        if _is_attr_node(neighbor):
            attrs.append(neighbor)
            continue
        if "EVENT" in {str(label).upper() for label in (neighbor.labels or [])}:
            for pred2, tip in _neighbors(neighbor.uuid, brain_id):
                if not _predicate_currently_valid(pred2):
                    continue
                if _is_attr_node(tip):
                    attrs.append(tip)
    return attrs


def resolve_user_node(target: str, brain_id: str) -> Optional[Node]:
    text = (target or "").strip()
    if not text:
        return None
    candidates = [text]
    if not text.startswith("user:"):
        candidates.append(f"user:{text}")
    try:
        nodes = graph_adapter.get_by_uuids(candidates, brain_id=brain_id) or []
    except Exception:
        nodes = []
    for node in nodes:
        if node is not None and getattr(node, "uuid", None):
            return node
    for uuid in candidates:
        try:
            node = graph_adapter.get_by_uuid(uuid, brain_id=brain_id)
        except Exception:
            node = None
        if node is not None and getattr(node, "uuid", None):
            return node
    return None


def user_pref_weights(
    seed: Node | None,
    brain_id: str,
    *,
    short_half_life_days: float = SHORT_TERM_HALF_LIFE_DAYS,
    behavior_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    if seed is None or not getattr(seed, "uuid", None):
        return {}
    for pred, neighbor in _neighbors(seed.uuid, brain_id):
        if not _predicate_currently_valid(pred):
            continue
        pred_name = (pred.name or "").upper()
        if pred_name in PREF_PREDICATES or "PREFER" in pred_name or "LIKE" in pred_name:
            if _is_attr_node(neighbor):
                scores[neighbor.uuid] += _prefers_weight(pred)
    for pred, event_node in _neighbors(seed.uuid, brain_id, of_types=["EVENT"]):
        if not _predicate_currently_valid(pred):
            continue
        if "EVENT" not in {str(label).upper() for label in (event_node.labels or [])}:
            continue
        weight = behavior_weight(event_node.name, behavior_weights)
        decay = recency_decay(event_node.happened_at, short_half_life_days)
        for pred2, tip in _neighbors(event_node.uuid, brain_id):
            if not _predicate_currently_valid(pred2):
                continue
            if tip.uuid == seed.uuid:
                continue
            if not _is_item_node(tip):
                continue
            contrib = float(weight) * float(decay)
            for attr in _attr_nodes_from_item(tip, brain_id):
                scores[attr.uuid] += contrib
    return {uuid: float(value) for uuid, value in scores.items() if value}


def score_nodes_for_user(
    node_ids: Iterable[str],
    prefs: Dict[str, float],
    brain_id: str,
) -> Dict[str, float]:
    if not prefs:
        return {str(node_id): 0.0 for node_id in node_ids if node_id}
    out: Dict[str, float] = {}
    for node_id in node_ids:
        key = str(node_id or "")
        if not key:
            continue
        total = 0.0
        try:
            node = graph_adapter.get_by_uuid(key, brain_id=brain_id)
        except Exception:
            node = None
        if node is None:
            out[key] = 0.0
            continue
        seen: set[str] = set()
        for attr in _attr_nodes_from_item(node, brain_id):
            if attr.uuid in seen:
                continue
            seen.add(attr.uuid)
            total += float(prefs.get(attr.uuid, 0.0))
        out[key] = total
    return out


def personalize_ranked_ids(
    *,
    query: str,
    ranked_ids: List[str],
    retrieve_scores: Dict[str, float],
    node_id_by_hit: Dict[str, Optional[str]],
    target: Optional[str],
    brain_id: str,
) -> tuple[list[str], dict[str, float]]:
    ids = list(ranked_ids)
    if not target or not ids:
        return ids, {}
    seed = resolve_user_node(target, brain_id)
    if seed is None:
        return ids, {}
    prefs = user_pref_weights(seed, brain_id)
    if not prefs:
        return ids, {}
    unique_nodes = []
    seen_nodes: set[str] = set()
    for hit_id in ids:
        node_id = node_id_by_hit.get(hit_id)
        if not node_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        unique_nodes.append(node_id)
    node_prefs = score_nodes_for_user(unique_nodes, prefs, brain_id)
    pref_by_hit = {
        hit_id: float(node_prefs.get(node_id_by_hit.get(hit_id) or "", 0.0))
        for hit_id in ids
    }
    lam = query_personalize_lambda(query)
    return blend_ranked(ids, retrieve_scores, pref_by_hit, lam)


__all__ = [
    "SHORT_TERM_HALF_LIFE_DAYS",
    "blend_ranked",
    "personalize_ranked_ids",
    "query_personalize_lambda",
    "resolve_user_node",
    "score_nodes_for_user",
    "user_pref_weights",
]
