from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Literal, Optional, Set, Tuple

from src.constants.kg import Node
from src.core.search.entity_sibilings import (
    DEFAULT_TOP_K,
    EntitySinergyRetriever,
    _predicate_currently_valid,
)
from src.utils.dates import parse_date_string, to_naive_utc
from src.services.kg_agent.main import graph_adapter

DEFAULT_BEHAVIOR_WEIGHTS: Dict[str, float] = {
    "view": 0.2,
    "click": 0.2,
    "addtocart": 0.5,
    "add_to_cart": 0.5,
    "cart": 0.5,
    "purchase": 1.0,
    "buy": 1.0,
    "purchased": 1.0,
    "bought": 1.0,
}

ATTR_LABELS = frozenset(
    {"CATEGORY", "BRAND", "COLOR", "MATERIAL", "ATTR", "ATTRIBUTE"}
)


def behavior_weight(
    event_name: str | None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    table = weights or DEFAULT_BEHAVIOR_WEIGHTS
    key = (event_name or "").strip().lower().replace("-", "").replace(" ", "")
    if key in table:
        return float(table[key])
    key2 = (event_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key2 in table:
        return float(table[key2])
    for alias, value in table.items():
        compact = alias.replace("-", "").replace("_", "").replace(" ", "")
        if compact == key:
            return float(value)
    return 1.0


def recency_decay(
    happened_at: str | datetime | None,
    half_life_days: float | None,
    *,
    now: datetime | None = None,
) -> float:
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    if not happened_at:
        return 1.0
    parsed = (
        parse_date_string(happened_at) if isinstance(happened_at, str) else happened_at
    )
    if not isinstance(parsed, datetime):
        return 1.0
    ref = now or datetime.now()
    age_days = max(0.0, (to_naive_utc(ref) - to_naive_utc(parsed)).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / float(half_life_days)))


def degree_dampen(degree: int, enabled: bool) -> float:
    if not enabled:
        return 1.0
    return 1.0 / math.log2(2.0 + max(0, int(degree)))


def attr_idf(degree: int, enabled: bool = True) -> float:
    if not enabled:
        return 1.0
    return 1.0 / math.log2(2.0 + max(0, int(degree)))


def _mmr_diversify(
    items: List[Tuple[Node, float, List[Node], str]],
    top_k: int,
    lambda_mult: float = 0.7,
) -> List[Tuple[Node, float, List[Node], str]]:
    if not items or top_k <= 0:
        return []
    selected: List[Tuple[Node, float, List[Node], str]] = []
    remaining = list(items)
    selected_types: set[str] = set()

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_val = float("-inf")
        for i, (node, score, _bridges, _kind) in enumerate(remaining):
            labels = {l.upper() for l in node.labels}
            overlap = len(labels & selected_types) / max(1, len(labels))
            mmr = lambda_mult * score - (1.0 - lambda_mult) * overlap
            if mmr > best_val:
                best_val = mmr
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_types.update(l.upper() for l in chosen[0].labels)
    return selected


def _node_degree(node: Node, brain_id: str) -> int:
    neighbors = graph_adapter.get_neighbors([node.uuid], brain_id=brain_id).get(
        node.uuid, []
    )
    return len(neighbors or [])


def _collect_seen_product_uuids(seed: Node, brain_id: str) -> Set[str]:
    seen: Set[str] = set()
    event_neighbors = graph_adapter.get_neighbors(
        [seed.uuid], of_types=["EVENT"], brain_id=brain_id
    ).get(seed.uuid, [])
    for pred, event_node in event_neighbors:
        if not _predicate_currently_valid(pred):
            continue
        outs = graph_adapter.get_neighbors(
            [event_node.uuid], brain_id=brain_id
        ).get(event_node.uuid, [])
        for pred2, tip in outs:
            if not _predicate_currently_valid(pred2):
                continue
            if tip.uuid == seed.uuid:
                continue
            if "PRODUCT" in {l.upper() for l in tip.labels}:
                seen.add(tip.uuid)
    return seen


def _asymmetric_event_targets(
    seed: Node,
    brain_id: str,
    *,
    direction: Literal["outbound", "inbound", "both"] = "outbound",
    behavior_weights: Optional[Dict[str, float]] = None,
    recency_half_life_days: Optional[float] = None,
    dampen_degree: bool = False,
) -> List[Tuple[Node, float, List[Node], str]]:
    results: List[Tuple[Node, float, List[Node], str]] = []
    if direction in ("outbound", "both"):
        event_neighbors = graph_adapter.get_neighbors(
            [seed.uuid], of_types=["EVENT"], brain_id=brain_id
        ).get(seed.uuid, [])
        for pred, event_node in event_neighbors:
            if not _predicate_currently_valid(pred):
                continue
            base = behavior_weight(event_node.name, behavior_weights)
            decay = recency_decay(event_node.happened_at, recency_half_life_days)
            outs = graph_adapter.get_neighbors(
                [event_node.uuid], brain_id=brain_id
            ).get(event_node.uuid, [])
            for pred2, tip in outs:
                if not _predicate_currently_valid(pred2):
                    continue
                if tip.uuid == seed.uuid:
                    continue
                if "EVENT" in tip.labels:
                    continue
                score = base * decay
                if dampen_degree:
                    score *= degree_dampen(_node_degree(tip, brain_id), True)
                results.append(
                    (tip, float(score), [event_node], "asymmetric_outbound")
                )
    if direction in ("inbound", "both"):
        inbound = graph_adapter.get_neighbors(
            [seed.uuid], brain_id=brain_id
        ).get(seed.uuid, [])
        for pred, neighbor in inbound:
            if not _predicate_currently_valid(pred):
                continue
            if "EVENT" not in neighbor.labels:
                continue
            base = 0.55 * behavior_weight(neighbor.name, behavior_weights)
            decay = recency_decay(neighbor.happened_at, recency_half_life_days)
            actors = graph_adapter.get_neighbors(
                [neighbor.uuid], brain_id=brain_id
            ).get(neighbor.uuid, [])
            for pred2, actor in actors:
                if not _predicate_currently_valid(pred2):
                    continue
                if actor.uuid == seed.uuid:
                    continue
                if "EVENT" in actor.labels:
                    continue
                score = base * decay
                if dampen_degree:
                    score *= degree_dampen(_node_degree(actor, brain_id), True)
                results.append(
                    (actor, float(score), [neighbor], "asymmetric_inbound")
                )
    return results


def _multi_interest_medoids(
    seed: Node,
    brain_id: str,
    max_clusters: int = 3,
) -> List[Node]:
    neighbors = graph_adapter.get_neighbors([seed.uuid], brain_id=brain_id).get(
        seed.uuid, []
    )
    by_label: Dict[str, List[Node]] = defaultdict(list)
    for pred, node in neighbors:
        if not _predicate_currently_valid(pred):
            continue
        if "EVENT" in node.labels:
            continue
        label = (node.labels[0] if node.labels else "UNKNOWN").upper()
        by_label[label].append(node)
    ranked_groups = sorted(by_label.values(), key=len, reverse=True)
    medoids: List[Node] = []
    for group in ranked_groups[:max_clusters]:
        if group:
            medoids.append(group[0])
    return medoids


def _prefers_weight(pred) -> float:
    props = pred.properties or {}
    raw = props.get("weight", props.get("properties", {}).get("weight") if isinstance(props.get("properties"), dict) else None)
    if raw is None and pred.amount is not None:
        raw = pred.amount
    try:
        return float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _is_attr_node(node: Node) -> bool:
    labels = {l.upper() for l in node.labels}
    if labels & ATTR_LABELS:
        return True
    return any(str(l).upper().startswith("ATTR") for l in node.labels)


def _products_for_attr(
    attr: Node,
    brain_id: str,
    *,
    seed_uuid: str,
) -> List[Node]:
    products: List[Node] = []
    neighbors = graph_adapter.get_neighbors([attr.uuid], brain_id=brain_id).get(
        attr.uuid, []
    )
    for pred, neighbor in neighbors:
        if not _predicate_currently_valid(pred):
            continue
        labels = {l.upper() for l in neighbor.labels}
        if "PRODUCT" in labels and neighbor.uuid != seed_uuid:
            products.append(neighbor)
            continue
        if "EVENT" not in labels:
            continue
        tips = graph_adapter.get_neighbors(
            [neighbor.uuid], brain_id=brain_id
        ).get(neighbor.uuid, [])
        for pred2, tip in tips:
            if not _predicate_currently_valid(pred2):
                continue
            if tip.uuid == seed_uuid:
                continue
            if "PRODUCT" in {l.upper() for l in tip.labels}:
                products.append(tip)
    return products


def _collaborative_user_item_targets(
    seed: Node,
    brain_id: str,
    *,
    behavior_weights: Optional[Dict[str, float]] = None,
    recency_half_life_days: Optional[float] = None,
    dampen_degree: bool = False,
) -> List[Tuple[Node, float, List[Node], str]]:
    """
    USER → EVENT → PRODUCT → (other EVENTs) → other USERs → their PRODUCTs.
    Produces unseen candidates for next-item RecSys (asymmetric alone only yields seen items).
    """
    results: List[Tuple[Node, float, List[Node], str]] = []
    seen = _collect_seen_product_uuids(seed, brain_id)
    if not seen:
        return results

    event_neighbors = graph_adapter.get_neighbors(
        [seed.uuid], of_types=["EVENT"], brain_id=brain_id
    ).get(seed.uuid, [])
    seed_products: List[Tuple[Node, float]] = []
    for pred, event_node in event_neighbors:
        if not _predicate_currently_valid(pred):
            continue
        w = behavior_weight(event_node.name, behavior_weights)
        w *= recency_decay(event_node.happened_at, recency_half_life_days)
        tips = graph_adapter.get_neighbors(
            [event_node.uuid], brain_id=brain_id
        ).get(event_node.uuid, [])
        for pred2, tip in tips:
            if not _predicate_currently_valid(pred2):
                continue
            if tip.uuid in seen and "PRODUCT" in {l.upper() for l in tip.labels}:
                seed_products.append((tip, w))

    peer_scores: Dict[str, float] = defaultdict(float)
    peer_nodes: Dict[str, Node] = {}
    for product, seed_w in seed_products:
        prod_neighbors = graph_adapter.get_neighbors(
            [product.uuid], brain_id=brain_id
        ).get(product.uuid, [])
        for pred, neighbor in prod_neighbors:
            if not _predicate_currently_valid(pred):
                continue
            if "EVENT" not in neighbor.labels:
                continue
            actors = graph_adapter.get_neighbors(
                [neighbor.uuid], brain_id=brain_id
            ).get(neighbor.uuid, [])
            for pred2, actor in actors:
                if not _predicate_currently_valid(pred2):
                    continue
                if actor.uuid == seed.uuid:
                    continue
                if "USER" not in {l.upper() for l in actor.labels}:
                    continue
                peer_scores[actor.uuid] += seed_w
                peer_nodes[actor.uuid] = actor

    for peer_uuid, peer_w in peer_scores.items():
        peer = peer_nodes[peer_uuid]
        peer_events = graph_adapter.get_neighbors(
            [peer.uuid], of_types=["EVENT"], brain_id=brain_id
        ).get(peer.uuid, [])
        for pred, event_node in peer_events:
            if not _predicate_currently_valid(pred):
                continue
            bw = behavior_weight(event_node.name, behavior_weights)
            bw *= recency_decay(event_node.happened_at, recency_half_life_days)
            tips = graph_adapter.get_neighbors(
                [event_node.uuid], brain_id=brain_id
            ).get(event_node.uuid, [])
            for pred2, tip in tips:
                if not _predicate_currently_valid(pred2):
                    continue
                if tip.uuid in seen or tip.uuid == seed.uuid:
                    continue
                if "PRODUCT" not in {l.upper() for l in tip.labels}:
                    continue
                score = float(peer_w * bw)
                if dampen_degree:
                    score *= degree_dampen(_node_degree(tip, brain_id), True)
                results.append(
                    (tip, score, [peer, event_node], "collaborative")
                )
    return results


def _attribute_pref_targets(
    seed: Node,
    brain_id: str,
    *,
    dampen_attr_degree: bool = True,
    behavior_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[Node, float, List[Node], str]]:
    results: List[Tuple[Node, float, List[Node], str]] = []
    pref_scores: Dict[str, float] = defaultdict(float)
    pref_nodes: Dict[str, Node] = {}

    neighbors = graph_adapter.get_neighbors([seed.uuid], brain_id=brain_id).get(
        seed.uuid, []
    )
    for pred, neighbor in neighbors:
        if not _predicate_currently_valid(pred):
            continue
        if not _is_attr_node(neighbor):
            continue
        pred_name = (pred.name or "").upper()
        if pred_name and pred_name not in {"PREFERS", "PREFER", "LIKES"}:
            if "PREFER" not in pred_name and "LIKE" not in pred_name:
                continue
        pref_scores[neighbor.uuid] += _prefers_weight(pred)
        pref_nodes[neighbor.uuid] = neighbor

    # Read-time soft prefs from USER → EVENT → PRODUCT → ATTR when PREFERS absent.
    if not pref_scores:
        event_neighbors = graph_adapter.get_neighbors(
            [seed.uuid], of_types=["EVENT"], brain_id=brain_id
        ).get(seed.uuid, [])
        for pred, event_node in event_neighbors:
            if not _predicate_currently_valid(pred):
                continue
            bw = behavior_weight(event_node.name, behavior_weights)
            tips = graph_adapter.get_neighbors(
                [event_node.uuid], brain_id=brain_id
            ).get(event_node.uuid, [])
            for pred2, tip in tips:
                if not _predicate_currently_valid(pred2):
                    continue
                if "PRODUCT" not in {l.upper() for l in tip.labels}:
                    continue
                prod_n = graph_adapter.get_neighbors(
                    [tip.uuid], brain_id=brain_id
                ).get(tip.uuid, [])
                for pred3, neigh in prod_n:
                    if not _predicate_currently_valid(pred3):
                        continue
                    attrs: List[Node] = []
                    if _is_attr_node(neigh):
                        attrs = [neigh]
                    elif "EVENT" in neigh.labels:
                        for pred4, tip2 in graph_adapter.get_neighbors(
                            [neigh.uuid], brain_id=brain_id
                        ).get(neigh.uuid, []):
                            if not _predicate_currently_valid(pred4):
                                continue
                            if _is_attr_node(tip2):
                                attrs.append(tip2)
                    for attr in attrs:
                        pref_scores[attr.uuid] += bw
                        pref_nodes[attr.uuid] = attr

    for attr_uuid, pref_w in pref_scores.items():
        attr = pref_nodes[attr_uuid]
        idf = attr_idf(_node_degree(attr, brain_id), dampen_attr_degree)
        for product in _products_for_attr(
            attr, brain_id, seed_uuid=seed.uuid
        ):
            results.append(
                (
                    product,
                    float(pref_w * idf),
                    [attr],
                    "attribute_pref",
                )
            )
    return results


class EntityRecommendRetriever:
    def __init__(self, brain_id: str = "default"):
        self.brain_id = brain_id
        self._synergies = EntitySinergyRetriever(brain_id)

    def recommend(
        self,
        target: str,
        *,
        polarity: Literal["same", "opposite"] = "same",
        top_k: int = DEFAULT_TOP_K,
        labels: Optional[List[str]] = None,
        include_asymmetric: bool = True,
        include_multi_interest: bool = True,
        include_attribute_pref: bool = False,
        diversify: bool = True,
        asymmetric_direction: Literal["outbound", "inbound", "both"] = "outbound",
        exclude_seen: bool = False,
        recency_half_life_days: Optional[float] = None,
        dampen_degree: bool = False,
        behavior_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[Node], List[dict]]:
        target_node, synergies, _anchors, _pa = self._synergies.retrieve_sibilings(
            target,
            polarity,
            do=False,
            pa=False,
            ppa=False,
            top_k=max(top_k * 3, top_k),
            labels=labels,
        )
        if target_node is None:
            return None, []

        seen_uuids: Set[str] = set()
        if exclude_seen:
            seen_uuids = _collect_seen_product_uuids(target_node, self.brain_id)

        pool: List[Tuple[Node, float, List[Node], str]] = []
        for syn in synergies:
            pool.append(
                (
                    syn.node,
                    float(syn.association_score),
                    list(syn.connected_by),
                    "synergy",
                )
            )

        if include_asymmetric:
            pool.extend(
                _asymmetric_event_targets(
                    target_node,
                    self.brain_id,
                    direction=asymmetric_direction,
                    behavior_weights=behavior_weights,
                    recency_half_life_days=recency_half_life_days,
                    dampen_degree=dampen_degree,
                )
            )
            if "USER" in {l.upper() for l in target_node.labels}:
                pool.extend(
                    _collaborative_user_item_targets(
                        target_node,
                        self.brain_id,
                        behavior_weights=behavior_weights,
                        recency_half_life_days=recency_half_life_days,
                        dampen_degree=dampen_degree,
                    )
                )

        if include_multi_interest:
            for medoid in _multi_interest_medoids(target_node, self.brain_id):
                if medoid.uuid == target_node.uuid:
                    continue
                _, medoid_syns, _, _ = self._synergies.retrieve_sibilings(
                    medoid.name,
                    polarity,
                    do=True,
                    top_k=max(5, top_k // 2),
                    labels=labels,
                )
                for syn in medoid_syns:
                    if syn.node.uuid == target_node.uuid:
                        continue
                    pool.append(
                        (
                            syn.node,
                            float(syn.association_score) * 0.85,
                            [medoid] + list(syn.connected_by),
                            "multi_interest",
                        )
                    )

        if include_attribute_pref:
            pool.extend(
                _attribute_pref_targets(
                    target_node,
                    self.brain_id,
                    dampen_attr_degree=True,
                    behavior_weights=behavior_weights,
                )
            )

        merged: Dict[str, Tuple[Node, float, List[Node], str]] = {}
        for node, score, bridges, kind in pool:
            if node.uuid == target_node.uuid:
                continue
            if exclude_seen and node.uuid in seen_uuids:
                continue
            if labels:
                label_set = {l.upper() for l in labels}
                if not set(l.upper() for l in node.labels).intersection(label_set):
                    continue
            if dampen_degree and kind in {"synergy", "multi_interest", "attribute_pref"}:
                score = float(score) * degree_dampen(
                    _node_degree(node, self.brain_id), True
                )
            prev = merged.get(node.uuid)
            if prev is None or score > prev[1]:
                merged[node.uuid] = (node, score, bridges, kind)
            elif prev is not None:
                bridge_ids = {b.uuid for b in prev[2]}
                extra = [b for b in bridges if b.uuid not in bridge_ids]
                merged[node.uuid] = (
                    prev[0],
                    prev[1],
                    prev[2] + extra,
                    prev[3],
                )

        ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        if diversify:
            ranked = _mmr_diversify(ranked, top_k)
        else:
            ranked = ranked[:top_k]

        recommendations = [
            {
                "node": node,
                "score": score,
                "connected_by": bridges,
                "channel": kind,
            }
            for node, score, bridges, kind in ranked
        ]
        return target_node, recommendations
