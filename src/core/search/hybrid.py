import re
from typing import Any, Optional

from src.core.search.fact_filter import reciprocal_rank_fusion

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_LITERAL_STOP = frozenset(
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
    }
)

_EXTRAS_SKIP = frozenset(
    {"resource_id", "uuid", "id", "inserted_at", "brain_version"}
)


def passage_snippet(text: str, max_len: int = 240) -> str:
    body = (text or "").strip()
    if len(body) <= max_len:
        return body
    return body[: max_len - 1].rstrip() + "…"


def dense_similarity(distance: float) -> float:
    return 1.0 - float(distance)


def extras_from_metadata(meta: Any) -> dict[str, str] | None:
    if not isinstance(meta, dict) or not meta:
        return None
    out: dict[str, str] = {}
    for key, value in meta.items():
        name = str(key or "").strip()
        if not name or name.lower() in _EXTRAS_SKIP:
            continue
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        text = str(value).strip()
        if not text:
            continue
        out[name] = text
    return out or None


def merge_hit_extras(*parts: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        for key, value in part.items():
            if value is None:
                continue
            merged[str(key)] = value
    return merged or None


def hit_matches_extras(
    have: dict[str, Any] | None,
    wanted: dict[str, str] | None,
) -> bool:
    if not wanted:
        return True
    lookup = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (have or {}).items()
        if value is not None
    }
    for key, value in wanted.items():
        if lookup.get(str(key).strip().lower()) != str(value).strip().lower():
            return False
    return True


def facet_counts_from_extras(
    extras_list: list[dict[str, Any] | None],
) -> dict[str, dict[str, int]] | None:
    buckets: dict[str, dict[str, int]] = {}
    for extras in extras_list:
        if not extras:
            continue
        for key, value in extras.items():
            name = str(key or "").strip()
            if not name or name.lower() in _EXTRAS_SKIP:
                continue
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            text = str(value).strip()
            if not text:
                continue
            inner = buckets.setdefault(name, {})
            inner[text] = inner.get(text, 0) + 1
    return buckets or None


def collect_dense_passages(
    vector_search: Any,
    query_vector: list[float],
    brain_id: str,
    k: int,
    extras_out: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[list[str], dict[str, float]]:
    if not query_vector or k <= 0:
        return [], {}
    hits = vector_search.search_data(
        query_vector, brain_id=brain_id, k=k
    )
    ids: list[str] = []
    distances: dict[str, float] = {}
    for vector in hits:
        meta = vector.metadata or {}
        resource_id = meta.get("resource_id") or getattr(vector, "id", None)
        if not resource_id:
            continue
        resource_id = str(resource_id)
        ids.append(resource_id)
        distances[resource_id] = (
            float(vector.distance) if vector.distance is not None else float("inf")
        )
        if extras_out is not None:
            parsed = extras_from_metadata(meta)
            if parsed:
                extras_out[resource_id] = parsed
    return ids, distances


def collect_bm25_passages(
    data_adapter: Any,
    query: str,
    brain_id: str,
    k: int,
    extras_out: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    if not query or k <= 0:
        return [], {}, {}
    ranked = data_adapter.search_bm25(query, brain_id, limit=k)
    ids: list[str] = []
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    for chunk, score in ranked or []:
        chunk_id = str(getattr(chunk, "id", "") or "")
        if not chunk_id:
            continue
        ids.append(chunk_id)
        scores[chunk_id] = float(score)
        texts[chunk_id] = getattr(chunk, "text", "") or ""
        if extras_out is not None:
            parsed = extras_from_metadata(getattr(chunk, "metadata", None))
            if parsed:
                extras_out[chunk_id] = parsed
    return ids, scores, texts


def collect_ilike_passages(
    data_adapter: Any,
    query: str,
    brain_id: str,
) -> tuple[list[str], dict[str, str]]:
    ids: list[str] = []
    texts: dict[str, str] = {}
    search_result = data_adapter.search(query, brain_id)
    for chunk in getattr(search_result, "text_chunks", None) or []:
        chunk_id = str(getattr(chunk, "id", "") or "")
        if not chunk_id:
            continue
        ids.append(chunk_id)
        texts[chunk_id] = getattr(chunk, "text", "") or ""
    return ids, texts


def query_tokens(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.lower()
        if len(token) < 3 or token in _LITERAL_STOP or token in seen:
            continue
        seen.add(token)
        found.append(token)
    return found


def literal_overlap_ids(
    query: str,
    texts_by_id: dict[str, str],
    *,
    k: int = 50,
) -> list[str]:
    qtoks = set(query_tokens(query))
    if not qtoks or k <= 0:
        return []
    scored: list[tuple[int, str]] = []
    for doc_id, body in texts_by_id.items():
        key = str(doc_id or "")
        if not key:
            continue
        overlap = len(qtoks & set(query_tokens(body or "")))
        if overlap <= 0:
            continue
        scored.append((overlap, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:k]]


def collect_literal_residual(
    data_adapter: Any,
    query: str,
    brain_id: str,
    k: int,
) -> tuple[list[str], dict[str, str]]:
    getter = getattr(data_adapter, "get_text_chunks", None)
    if getter is None or not query or k <= 0:
        return [], {}
    texts: dict[str, str] = {}
    for token in query_tokens(query):
        try:
            chunks, _total = getter(brain_id, limit=k, query_text=token)
        except TypeError:
            try:
                chunks, _total = getter(brain_id, k, 0, token)
            except Exception:
                continue
        except Exception:
            continue
        for chunk in chunks or []:
            chunk_id = str(getattr(chunk, "id", "") or "")
            if not chunk_id:
                continue
            texts[chunk_id] = getattr(chunk, "text", "") or ""
    return literal_overlap_ids(query, texts, k=k), texts


def frozen_head_merge(
    ranked_ids: list[str],
    extra_id_lists: Optional[list[list[str]]] = None,
    *,
    head_k: int = 10,
    k: int = 50,
    prefer_ids: Optional[set[str]] = None,
) -> list[str]:
    ranked = [str(item) for item in ranked_ids if item]
    head = ranked[: max(int(head_k), 0)]
    ranked_k = ranked[: max(int(k), 0)]
    passages_k = set(ranked_k)
    extras: list[str] = []
    seen_extra: set[str] = set()
    for extra in extra_id_lists or []:
        for hid in extra[: max(int(k), 0)]:
            key = str(hid)
            if not key or key in seen_extra or key in passages_k:
                continue
            if prefer_ids is not None and key not in prefer_ids:
                continue
            seen_extra.add(key)
            extras.append(key)
    rest = [item for item in ranked_k[len(head) :] if item not in set(head)]
    if prefer_ids is not None:
        fill = (
            extras
            + [item for item in rest if item in prefer_ids]
            + [item for item in rest if item not in prefer_ids]
        )
    else:
        fill = extras + rest
    tail: list[str] = []
    seen = set(head)
    limit = max(int(k), 0)
    for hid in fill:
        if hid in seen:
            continue
        seen.add(hid)
        tail.append(hid)
        if len(head) + len(tail) >= limit:
            break
    return (head + tail)[:limit]


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 1.0 for key in values}
    scale = hi - lo
    return {key: (value - lo) / scale for key, value in values.items()}


def convex_combination(
    dense_scores: dict[str, float],
    bm25_scores: dict[str, float],
    alpha: float,
) -> list[tuple[str, float]]:
    alpha = min(1.0, max(0.0, float(alpha)))
    dense_n = _minmax(dense_scores)
    bm25_n = _minmax(bm25_scores)
    ids = set(dense_n) | set(bm25_n)
    fused = {
        item: alpha * dense_n.get(item, 0.0) + (1.0 - alpha) * bm25_n.get(item, 0.0)
        for item in ids
    }
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def fuse_passage_lists(
    dense_ids: list[str],
    lexical_ids: list[str],
    *,
    fusion: str = "rrf",
    alpha: float = 0.5,
    dense_similarities: Optional[dict[str, float]] = None,
    bm25_scores: Optional[dict[str, float]] = None,
    extra_id_lists: Optional[list[list[str]]] = None,
) -> list[tuple[str, float]]:
    extra = [ids for ids in (extra_id_lists or []) if ids]
    lists = [ids for ids in (dense_ids, lexical_ids) if ids]
    if fusion == "cc" and (dense_ids or lexical_ids):
        core = convex_combination(
            dense_similarities or {},
            bm25_scores or {},
            alpha,
        )
        if not extra:
            return core
        return reciprocal_rank_fusion([[item for item, _ in core], *extra])
    lists.extend(extra)
    if not lists:
        return []
    if len(lists) == 1:
        return [(item, 1.0 / (rank + 1)) for rank, item in enumerate(lists[0])]
    return reciprocal_rank_fusion(lists)
