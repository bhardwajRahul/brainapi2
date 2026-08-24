from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def recall_at_k(
    ranked_ids: Sequence[str],
    gold: Iterable[str],
    k: int,
) -> float:
    gold_set = {str(item) for item in gold}
    if not gold_set:
        return 0.0
    retrieved = {str(item) for item in list(ranked_ids)[:k]}
    return len(retrieved & gold_set) / len(gold_set)


def mrr(ranked_ids: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = {str(item) for item in gold}
    if not gold_set:
        return 0.0
    for index, item in enumerate(ranked_ids, start=1):
        if str(item) in gold_set:
            return 1.0 / index
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    score = 0.0
    for index, rel in enumerate(list(relevances)[:k], start=1):
        score += float(rel) / math.log2(index + 1)
    return score


def ndcg_at_k(
    ranked_ids: Sequence[str],
    gold: Iterable[str],
    k: int,
    grades: dict[str, float] | None = None,
) -> float:
    if grades:
        gain = {
            str(doc_id): float(value)
            for doc_id, value in grades.items()
            if float(value) > 0
        }
        if not gain:
            return 0.0
        relevances = [gain.get(str(item), 0.0) for item in list(ranked_ids)[:k]]
        ideal = sorted(gain.values(), reverse=True)[:k]
        dcg = _dcg(relevances, k)
        idcg = _dcg(ideal, k)
        if idcg <= 0:
            return 0.0
        return dcg / idcg
    gold_set = {str(item) for item in gold}
    if not gold_set:
        return 0.0
    relevances = [1.0 if str(item) in gold_set else 0.0 for item in list(ranked_ids)[:k]]
    dcg = _dcg(relevances, k)
    ideal = [1.0] * min(len(gold_set), k)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return xs[int(rank)]
    return xs[low] * (high - rank) + xs[high] * (rank - low)


def _stage_wall_ms(stage_timings: dict[str, Any] | None, name: str) -> float | None:
    stages = (stage_timings or {}).get("stages") or []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if stage.get("stage") == name and stage.get("wall_ms") is not None:
            return float(stage["wall_ms"])
    return None


def retrieve_latency_ms(
    stage_timings: dict[str, Any] | None,
    client_wall_ms: float,
) -> tuple[float, float | None]:
    retrieve_ms = _stage_wall_ms(stage_timings, "search.retrieve")
    embed_ms = _stage_wall_ms(stage_timings, "embed.query")
    if retrieve_ms is not None:
        return retrieve_ms, embed_ms
    if embed_ms is not None:
        return max(0.0, float(client_wall_ms) - embed_ms), embed_ms
    return float(client_wall_ms), embed_ms


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_query_metrics(
    per_query: list[dict[str, Any]],
    *,
    ks: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in ks:
        key = f"recall@{k}"
        metrics[key] = mean([float(row["metrics"][key]) for row in per_query])
    ndcg_ks = tuple(dict.fromkeys((*ks, 10, 20)))
    for k in ndcg_ks:
        key = f"ndcg@{k}"
        values = [
            float(row["metrics"][key])
            for row in per_query
            if row.get("metrics", {}).get(key) is not None
        ]
        if values:
            metrics[key] = mean(values)
    ndcg_full = [
        float(row["metrics"]["ndcg"])
        for row in per_query
        if row.get("metrics", {}).get("ndcg") is not None
    ]
    if ndcg_full:
        metrics["ndcg"] = mean(ndcg_full)
    metrics["mrr"] = mean([float(row["metrics"]["mrr"]) for row in per_query])

    retrieve_ms = [float(row["retrieve_ms"]) for row in per_query]
    embed_ms = [
        float(row["embed_ms"])
        for row in per_query
        if row.get("embed_ms") is not None
    ]
    client_ms = [float(row["client_wall_ms"]) for row in per_query]
    metrics["p50_retrieve_ms"] = percentile(retrieve_ms, 50)
    metrics["p95_retrieve_ms"] = percentile(retrieve_ms, 95)
    metrics["p50_embed_ms"] = percentile(embed_ms, 50)
    metrics["p95_embed_ms"] = percentile(embed_ms, 95)
    metrics["p50_client_wall_ms"] = percentile(client_ms, 50)
    metrics["p95_client_wall_ms"] = percentile(client_ms, 95)

    slices: dict[str, list[dict[str, Any]]] = {}
    for row in per_query:
        slices.setdefault(str(row.get("slice") or "unspecified"), []).append(row)
    by_slice: dict[str, dict[str, Any]] = {}
    for name, rows in slices.items():
        by_slice[name] = {
            "n_queries": len(rows),
            "recall@10": mean([float(r["metrics"]["recall@10"]) for r in rows]),
            "ndcg@10": mean([float(r["metrics"]["ndcg@10"]) for r in rows]),
            "mrr": mean([float(r["metrics"]["mrr"]) for r in rows]),
        }
        for k in ks:
            recall_key = f"recall@{k}"
            if all(r.get("metrics", {}).get(recall_key) is not None for r in rows):
                by_slice[name][recall_key] = mean(
                    [float(r["metrics"][recall_key]) for r in rows]
                )
        for k in ndcg_ks:
            ndcg_key = f"ndcg@{k}"
            if any(r.get("metrics", {}).get(ndcg_key) is not None for r in rows):
                by_slice[name][ndcg_key] = mean(
                    [float(r["metrics"][ndcg_key]) for r in rows]
                )
        if any(r.get("metrics", {}).get("ndcg") is not None for r in rows):
            by_slice[name]["ndcg"] = mean([float(r["metrics"]["ndcg"]) for r in rows])
    metrics["by_slice"] = by_slice
    return metrics
