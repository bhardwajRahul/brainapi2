from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping, Optional, Sequence


_SPINE_ACTOR = ("MADE", "INITIATED", "PERFORMED", "EXPERIENCED", "COVERED")
_SPINE_TARGET = ("TARGETED", "AFFECTED", "RESULTED")
_CONTEXT_REL = ("OCCURRED", "WITHIN")


@dataclass(frozen=True)
class HubBridge:
    event_a: str
    event_b: str
    shared_entity: str
    shared_entity_name: str = ""
    weight: float = 1.0

    @property
    def pair(self) -> tuple[str, str]:
        return canonical_event_pair(self.event_a, self.event_b)


def canonical_event_pair(event_a: str, event_b: str) -> tuple[str, str]:
    a = str(event_a or "").strip()
    b = str(event_b or "").strip()
    if not a or not b or a == b:
        raise ValueError("hub bridge requires two distinct event uuids")
    return (a, b) if a < b else (b, a)


def is_event_labels(labels: Optional[Sequence[str]]) -> bool:
    return any(str(label).upper() == "EVENT" for label in (labels or ()))


def is_spine_rel(name: Optional[str]) -> bool:
    upper = (name or "").upper()
    return any(token in upper for token in _SPINE_ACTOR) or any(
        token in upper for token in _SPINE_TARGET
    )


def is_context_rel(name: Optional[str]) -> bool:
    upper = (name or "").upper()
    return any(token in upper for token in _CONTEXT_REL)


def entity_event_memberships(
    rows: Iterable[tuple[str, str, str, str]],
) -> dict[str, dict[str, str]]:
    """
    Build shared-entity → {event_uuid: entity_name} from spine rows.

    Each row is (event_uuid, entity_uuid, entity_name, rel_name).
    Context / OCCURRED_WITHIN legs are ignored so bridges stay actor/object mediated.
    """
    membership: dict[str, dict[str, str]] = {}
    for event_uuid, entity_uuid, entity_name, rel_name in rows:
        event_id = str(event_uuid or "").strip()
        entity_id = str(entity_uuid or "").strip()
        if not event_id or not entity_id or event_id == entity_id:
            continue
        if is_context_rel(rel_name):
            continue
        if not is_spine_rel(rel_name):
            continue
        bucket = membership.setdefault(entity_id, {})
        if event_id not in bucket:
            bucket[event_id] = str(entity_name or "")
    return membership


def bridges_from_memberships(
    membership: Mapping[str, Mapping[str, str]],
) -> list[HubBridge]:
    bridges: dict[tuple[str, str, str], HubBridge] = {}
    for entity_id, events in membership.items():
        event_ids = sorted(str(e) for e in events.keys() if e)
        if len(event_ids) < 2:
            continue
        entity_name = ""
        for name in events.values():
            if name:
                entity_name = str(name)
                break
        for i, left in enumerate(event_ids):
            for right in event_ids[i + 1 :]:
                a, b = canonical_event_pair(left, right)
                key = (a, b, entity_id)
                bridges[key] = HubBridge(
                    event_a=a,
                    event_b=b,
                    shared_entity=entity_id,
                    shared_entity_name=entity_name,
                    weight=1.0,
                )
    return sorted(
        bridges.values(),
        key=lambda br: (br.event_a, br.event_b, br.shared_entity),
    )


def _hub_session_set(
    hub: str,
    hub_sessions: Optional[Mapping[str, AbstractSet[str] | Sequence[str]]],
) -> set[str]:
    if not hub_sessions:
        return set()
    raw = hub_sessions.get(hub) or ()
    return {str(s) for s in raw if s}


def collect_bridge_neighbor_pool(
    seed_hubs: Sequence[str],
    bridges: Sequence[HubBridge],
    *,
    pool_cap: int,
) -> list[tuple[str, HubBridge]]:
    if pool_cap <= 0 or not seed_hubs or not bridges:
        return []
    seeds = [str(h) for h in seed_hubs if str(h or "").strip()]
    seed_set = set(seeds)
    if not seed_set:
        return []

    best: dict[str, tuple[tuple[float, str, str], HubBridge]] = {}
    for bridge in bridges:
        a, b = bridge.event_a, bridge.event_b
        neighbor = None
        if a in seed_set and b not in seed_set:
            neighbor = b
        elif b in seed_set and a not in seed_set:
            neighbor = a
        if neighbor is None:
            continue
        rank = (-float(bridge.weight), bridge.shared_entity, neighbor)
        prev = best.get(neighbor)
        if prev is None or rank < prev[0]:
            best[neighbor] = (rank, bridge)

    ranked = sorted(best.values(), key=lambda item: item[0])
    chosen = [(rank[2], bridge) for rank, bridge in ranked[:pool_cap]]
    chosen.sort(key=lambda item: (item[0], item[1].shared_entity, item[1].event_a))
    return chosen


def select_bridge_neighbors(
    seed_hubs: Sequence[str],
    bridges: Sequence[HubBridge],
    *,
    max_per_hub: int,
    seed_sessions: Optional[AbstractSet[str]] = None,
    hub_sessions: Optional[Mapping[str, AbstractSet[str] | Sequence[str]]] = None,
) -> list[tuple[str, HubBridge]]:
    """
    Expand seed hubs by at most one bridge hop.

    Returns deterministic (neighbor_event_uuid, bridge) pairs, unique by neighbor,
    preferring novel source sessions vs the seed/selected set, then higher weight,
    then lexicographic shared_entity / event ids.
    """
    if max_per_hub <= 0 or not seed_hubs or not bridges:
        return []
    seeds = [str(h) for h in seed_hubs if str(h or "").strip()]
    seed_set = set(seeds)
    if not seed_set:
        return []

    covered_sessions = {str(s) for s in (seed_sessions or ()) if s}
    by_hub: dict[str, list[tuple[str, HubBridge]]] = {h: [] for h in seeds}
    for bridge in bridges:
        a, b = bridge.event_a, bridge.event_b
        if a in seed_set and b not in seed_set:
            by_hub.setdefault(a, []).append((b, bridge))
        elif b in seed_set and a not in seed_set:
            by_hub.setdefault(b, []).append((a, bridge))

    chosen: list[tuple[str, HubBridge]] = []
    seen_neighbors: set[str] = set()
    for hub in sorted(by_hub.keys()):
        taken = 0
        remaining = list(by_hub[hub])
        while remaining and taken < max_per_hub:
            remaining.sort(
                key=lambda item: (
                    -len(_hub_session_set(item[0], hub_sessions) - covered_sessions),
                    -float(item[1].weight),
                    item[1].shared_entity,
                    item[0],
                )
            )
            neighbor, bridge = remaining.pop(0)
            if neighbor in seen_neighbors or neighbor in seed_set:
                continue
            chosen.append((neighbor, bridge))
            seen_neighbors.add(neighbor)
            covered_sessions.update(_hub_session_set(neighbor, hub_sessions))
            taken += 1
    chosen.sort(key=lambda item: (item[0], item[1].shared_entity, item[1].event_a))
    return chosen
