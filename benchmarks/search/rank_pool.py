from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from search.dataset import split_corpus
from search.evaluate import candidate_pool_grades, candidate_pool_ids
from search.metrics import aggregate_query_metrics, mrr, ndcg_at_k, recall_at_k

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "search-rerank"


def _plugin_rerank():
    if str(PLUGIN_DIR) not in sys.path:
        sys.path.insert(0, str(PLUGIN_DIR))
    from rerank import model_name, rerank

    return rerank, model_name()


def doc_texts(docs: list[dict[str, Any]]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        texts[doc_id] = str(doc.get("text") or doc.get("title") or "")
    return texts


def score_ranked_pool(
    ranked: list[str],
    query: dict[str, Any],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    gold = {
        str(doc_id)
        for doc_id, gain in candidate_pool_grades(query).items()
        if float(gain) > 0
    }
    if not gold:
        gold = {str(item) for item in (query.get("gold_doc_ids") or [])}
    grades = candidate_pool_grades(query)
    metrics = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    ndcg_ks = tuple(dict.fromkeys((*ks, 10, 20)))
    for cut in ndcg_ks:
        metrics[f"ndcg@{cut}"] = ndcg_at_k(ranked, gold, cut, grades=grades)
    metrics["ndcg"] = ndcg_at_k(ranked, gold, max(len(ranked), 1), grades=grades)
    metrics["mrr"] = mrr(ranked, gold)
    return metrics


def run_ce_on_pool(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    ks: tuple[int, ...] = (5, 10, 20),
    brain_id: str | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    texts = doc_texts(docs)
    rerank, model = _plugin_rerank()
    per_query: list[dict[str, Any]] = []
    missing_text = 0
    for query in queries:
        pool = candidate_pool_ids(query)
        candidates = []
        for doc_id in pool:
            text = texts.get(doc_id) or ""
            if not text:
                missing_text += 1
            candidates.append({"id": doc_id, "text": text, "score": 0.0})
        ranked_rows = rerank(
            str(query.get("query") or ""),
            candidates,
            len(candidates) or 1,
        )
        ranked = [str(item.get("id") or "") for item in ranked_rows if item.get("id")]
        seen = set(ranked)
        for doc_id in pool:
            if doc_id not in seen:
                ranked.append(doc_id)
        metrics = score_ranked_pool(ranked, query, ks=ks)
        per_query.append(
            {
                "qid": query.get("qid"),
                "query": query.get("query"),
                "slice": query.get("slice") or "unspecified",
                "gold_doc_ids": list(query.get("gold_doc_ids") or []),
                "candidate_doc_ids": pool,
                "pool_size": len(pool),
                "pool_coverage": 1.0 if pool else None,
                "missing_from_brain": [],
                "hit_ids": ranked,
                "metrics": metrics,
                "retrieve_ms": 0.0,
                "embed_ms": None,
                "client_wall_ms": 0.0,
                "n_hits": len(ranked),
            }
        )
    metrics = aggregate_query_metrics(per_query, ks=ks) if per_query else {}
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id,
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "plugin:cross-encoder",
        "channels": ["rank-pool-ce"],
        "expand": "none",
        "k": max((len(row.get("hit_ids") or []) for row in per_query), default=0),
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": len(texts),
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": True,
        "rank_pool_ce": True,
        "ce_model": model,
        "ce_missing_text": missing_text,
        "ingest": {"status": "completed", "n_docs": len(docs), "tasks": [], "reused": True},
        "graph_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "interaction_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "event_probe": None,
        "search_error": None,
        "protocol": (
            "ranking-in-pool CE over labeled candidates; "
            f"model={model}; n is not Reddy ~4477 unless stated; "
            "cite 0.857 only against this CE-on-pool protocol"
        ),
        "metrics": metrics,
        "queries": per_query,
    }
