from __future__ import annotations

from typing import Any

from search.evaluate import candidate_pool_grades
from search.metrics import aggregate_query_metrics, mrr, ndcg_at_k, recall_at_k
from search.rank_pool import _plugin_rerank, doc_texts, score_ranked_pool
from search.dataset import split_corpus


def _first_stage_metrics(
    ranked: list[str],
    query_row: dict[str, Any],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    gold = {str(item) for item in (query_row.get("gold_doc_ids") or [])}
    grades = {
        str(doc_id): float(gain)
        for doc_id, gain in (query_row.get("gold_grades") or {}).items()
        if float(gain or 0) > 0
    }
    metrics = {f"recall@{cut}": recall_at_k(ranked, gold, cut) for cut in ks}
    ndcg_ks = tuple(dict.fromkeys((*ks, 10, 20)))
    for cut in ndcg_ks:
        metrics[f"ndcg@{cut}"] = ndcg_at_k(ranked, gold, cut, grades=grades or None)
    metrics["ndcg"] = ndcg_at_k(ranked, gold, max(len(ranked), 1), grades=grades or None)
    metrics["mrr"] = mrr(ranked, gold)
    return metrics


def run_ce_on_retrieved(
    eval_result: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    ks: tuple[int, ...] | None = None,
    brain_id: str | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    texts = doc_texts(docs)
    by_qid = {str(query.get("qid") or ""): query for query in queries}
    rerank, model = _plugin_rerank()
    source_ks = tuple(int(item) for item in (eval_result.get("ks") or (5, 10, 20)))
    ks = ks or source_ks
    rank_pool = bool(eval_result.get("rank_pool"))
    per_query: list[dict[str, Any]] = []
    missing_text = 0
    for row in eval_result.get("queries") or []:
        hit_ids = [str(item) for item in (row.get("hit_ids") or []) if item]
        qid = str(row.get("qid") or "")
        source = by_qid.get(qid) or {}
        candidates = []
        for doc_id in hit_ids:
            text = texts.get(doc_id) or ""
            if not text:
                missing_text += 1
            candidates.append({"id": doc_id, "text": text, "score": 0.0})
        ranked_rows = rerank(
            str(row.get("query") or source.get("query") or ""),
            candidates,
            len(candidates) or 1,
        )
        ranked = [str(item.get("id") or "") for item in ranked_rows if item.get("id")]
        seen = set(ranked)
        for doc_id in hit_ids:
            if doc_id not in seen:
                ranked.append(doc_id)
        if rank_pool and source:
            metrics = score_ranked_pool(ranked, source, ks=ks)
        else:
            metrics = _first_stage_metrics(ranked, row, ks=ks)
        per_query.append(
            {
                "qid": row.get("qid"),
                "query": row.get("query"),
                "slice": row.get("slice") or "unspecified",
                "gold_doc_ids": list(row.get("gold_doc_ids") or []),
                "gold_grades": row.get("gold_grades") or {},
                "candidate_doc_ids": row.get("candidate_doc_ids") or [],
                "pool_size": row.get("pool_size"),
                "pool_coverage": row.get("pool_coverage"),
                "missing_from_brain": row.get("missing_from_brain") or [],
                "hit_ids": ranked,
                "metrics": metrics,
                "retrieve_ms": float(row.get("retrieve_ms") or 0.0),
                "embed_ms": row.get("embed_ms"),
                "client_wall_ms": float(row.get("client_wall_ms") or 0.0),
                "n_hits": len(ranked),
            }
        )
    metrics = aggregate_query_metrics(per_query, ks=ks) if per_query else {}
    if rank_pool and per_query:
        coverages = [
            float(row["pool_coverage"])
            for row in per_query
            if row.get("pool_coverage") is not None
        ]
        if coverages:
            metrics["pool_coverage"] = sum(coverages) / len(coverages)
        metrics["missing_from_brain"] = sum(
            len(row.get("missing_from_brain") or []) for row in per_query
        ) / len(per_query)
    retrieve_k = int(eval_result.get("k") or 0)
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id or eval_result.get("brain_id"),
        "dataset": dataset_name,
        "fusion": eval_result.get("fusion") or "rrf",
        "rerank": "harness:cross-encoder",
        "channels": eval_result.get("channels") or ["passages"],
        "expand": eval_result.get("expand") or "none",
        "k": retrieve_k,
        "ks": list(ks),
        "n_docs": eval_result.get("n_docs") or len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": len(texts),
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": rank_pool,
        "rank_pool_ce": False,
        "ce_model": model,
        "ce_missing_text": missing_text,
        "source_run": eval_result.get("run_id"),
        "ingest": {
            "status": "completed",
            "n_docs": eval_result.get("n_docs") or len(docs),
            "tasks": [],
            "reused": True,
        },
        "graph_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "interaction_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "event_probe": None,
        "search_error": None,
        "protocol": (
            "harness CE over retrieved first-stage hits "
            f"(k={retrieve_k}, model={model}); not CE-on-pool; "
            "do not cite Reddy 0.857"
        ),
        "metrics": metrics,
        "queries": per_query,
    }
