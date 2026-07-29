from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_JACCARD_MERGE = 0.28
_MAX_TOPIC_ENTITIES = 8


@dataclass(frozen=True)
class TopicSession:
    topic_id: str
    topic_label: str
    session_id: str
    weight: float = 1.0


def _norm_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def topic_id_for_label(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"topic:{digest}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def session_entity_sets(
    rows: Iterable[tuple[str, str, str]],
) -> dict[str, set[str]]:
    """
    Build session_id → entity-name tokens from (session_id, entity_uuid, entity_name).
    """
    out: dict[str, set[str]] = {}
    for session_id, _entity_uuid, entity_name in rows:
        sid = str(session_id or "").strip()
        if not sid:
            continue
        tokens = _norm_tokens(str(entity_name or ""))
        if not tokens:
            continue
        out.setdefault(sid, set()).update(tokens)
    return out


def cluster_sessions_into_topics(
    session_entities: Mapping[str, set[str]],
    *,
    merge_threshold: float = _JACCARD_MERGE,
) -> list[TopicSession]:
    """Greedy Jaccard clustering of sessions by shared entity tokens."""
    sessions = sorted(session_entities.keys())
    if not sessions:
        return []

    clusters: list[dict[str, object]] = []
    for sid in sessions:
        ents = set(session_entities.get(sid) or ())
        best_i = -1
        best_score = -1.0
        for i, cluster in enumerate(clusters):
            score = jaccard(ents, set(cluster["entities"]))  # type: ignore[arg-type]
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0 and best_score >= merge_threshold:
            cluster = clusters[best_i]
            members: list[str] = list(cluster["sessions"])  # type: ignore[arg-type]
            members.append(sid)
            cluster["sessions"] = members
            merged = set(cluster["entities"])  # type: ignore[arg-type]
            merged.update(ents)
            cluster["entities"] = merged
        else:
            clusters.append({"sessions": [sid], "entities": ents})

    results: list[TopicSession] = []
    for cluster in clusters:
        ents = sorted(set(cluster["entities"]))  # type: ignore[arg-type]
        label_tokens = ents[:_MAX_TOPIC_ENTITIES]
        label = ", ".join(label_tokens) if label_tokens else "general"
        tid = topic_id_for_label(label + "|" + ",".join(sorted(cluster["sessions"])))  # type: ignore[arg-type]
        for sid in sorted(cluster["sessions"]):  # type: ignore[arg-type]
            results.append(
                TopicSession(
                    topic_id=tid,
                    topic_label=label,
                    session_id=str(sid),
                    weight=1.0,
                )
            )
    return results


def lexical_topic_score(query: str, topic_label: str) -> float:
    q = _norm_tokens(query)
    t = _norm_tokens(topic_label)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def rrf_fuse(rank_lists: Sequence[Sequence[str]], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, item in enumerate(ranks):
            key = str(item)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def select_topics_and_sessions(
    query: str,
    memberships: Sequence[TopicSession],
    seed_sessions: Sequence[str],
    *,
    k_topics: int = 10,
    k_sessions: int = 10,
) -> tuple[list[dict[str, object]], list[str]]:
    """
    Coarse-to-fine: score topics lexically (+ seed-session boost), expand sessions.
    """
    if not memberships:
        return [], []

    by_topic: dict[str, dict[str, object]] = {}
    for row in memberships:
        bucket = by_topic.setdefault(
            row.topic_id,
            {
                "topic_id": row.topic_id,
                "label": row.topic_label,
                "sessions": set(),
            },
        )
        sessions = bucket["sessions"]
        assert isinstance(sessions, set)
        sessions.add(row.session_id)

    seed = {str(s) for s in seed_sessions if s}
    scored: list[tuple[float, str]] = []
    for topic_id, bucket in by_topic.items():
        label = str(bucket["label"])
        sessions = set(bucket["sessions"])  # type: ignore[arg-type]
        lex = lexical_topic_score(query, label)
        seed_hit = 1.0 if (sessions & seed) else 0.0
        score = 0.7 * lex + 0.3 * seed_hit
        scored.append((score, topic_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    top_topics = [tid for score, tid in scored[:k_topics] if score > 0 or seed]

    topic_payload: list[dict[str, object]] = []
    lexical_session_ranks: list[str] = []
    seed_ranks = [s for s in seed_sessions if s]
    for tid in top_topics:
        bucket = by_topic[tid]
        sessions = sorted(bucket["sessions"])  # type: ignore[arg-type]
        topic_payload.append(
            {
                "topic_id": tid,
                "label": bucket["label"],
                "sessions": sessions,
            }
        )
        for sid in sessions:
            if sid not in lexical_session_ranks:
                lexical_session_ranks.append(sid)

    fused = rrf_fuse([seed_ranks, lexical_session_ranks])
    return topic_payload, fused[:k_sessions]
