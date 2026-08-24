from __future__ import annotations

from typing import Any

from search.dataset import split_corpus
from search.metrics import aggregate_query_metrics, mrr, ndcg_at_k, recall_at_k
from search.rank_pool import doc_texts

DEFAULT_BASE = "sentence-transformers/all-MiniLM-L6-v2"


def score_ranked_first_stage(
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
    if grades:
        gold |= set(grades)
    metrics = {f"recall@{cut}": recall_at_k(ranked, gold, cut) for cut in ks}
    ndcg_ks = tuple(dict.fromkeys((*ks, 10, 20)))
    for cut in ndcg_ks:
        metrics[f"ndcg@{cut}"] = ndcg_at_k(ranked, gold, cut, grades=grades or None)
    metrics["ndcg"] = ndcg_at_k(ranked, gold, max(len(ranked), 1), grades=grades or None)
    metrics["mrr"] = mrr(ranked, gold)
    return metrics


def hard_negative_texts_from_eval(
    eval_result: dict[str, Any],
    texts: dict[str, str],
    queries: list[dict[str, Any]],
) -> list[str]:
    by_qid = {str(query.get("qid") or ""): query for query in queries}
    bank: list[str] = []
    seen: set[str] = set()
    for row in eval_result.get("queries") or []:
        qid = str(row.get("qid") or "")
        source = by_qid.get(qid) or row
        gold = {str(item) for item in (source.get("gold_doc_ids") or [])}
        grades = source.get("gold_grades") or {}
        if isinstance(grades, dict):
            gold |= {
                str(doc_id)
                for doc_id, gain in grades.items()
                if float(gain or 0) > 0
            }
        for doc_id in row.get("hit_ids") or []:
            key = str(doc_id)
            if key in gold or key in seen:
                continue
            text = texts.get(key) or ""
            if not text:
                continue
            seen.add(key)
            bank.append(text)
    return bank


def retrieve_dense(
    model: Any,
    queries: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    *,
    k: int,
    batch_size: int = 32,
) -> tuple[dict[str, list[str]], float]:
    import time

    import numpy as np

    texts = doc_texts(docs)
    doc_ids = [str(doc.get("doc_id") or "") for doc in docs if doc.get("doc_id")]
    doc_passages = [texts.get(doc_id) or doc_id for doc_id in doc_ids]
    start = time.perf_counter()
    doc_emb = model.encode(
        doc_passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_texts = [str(query.get("query") or "") for query in queries]
    q_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    scores = np.matmul(q_emb, doc_emb.T)
    ranked_by_qid: dict[str, list[str]] = {}
    cut = max(1, min(int(k), len(doc_ids)))
    for index, query in enumerate(queries):
        order = np.argsort(-scores[index])[:cut]
        qid = str(query.get("qid") or "")
        ranked_by_qid[qid] = [doc_ids[int(pos)] for pos in order]
    return ranked_by_qid, elapsed_ms


def evaluate_dense(
    ranked_by_qid: dict[str, list[str]],
    queries: list[dict[str, Any]],
    *,
    ks: tuple[int, ...],
    encode_ms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    per_q_ms = encode_ms / max(1, len(queries))
    for query in queries:
        qid = str(query.get("qid") or "")
        ranked = ranked_by_qid.get(qid) or []
        metrics = score_ranked_first_stage(ranked, query, ks=ks)
        per_query.append(
            {
                "qid": qid,
                "query": query.get("query"),
                "slice": query.get("slice") or "unspecified",
                "gold_doc_ids": list(query.get("gold_doc_ids") or []),
                "gold_grades": query.get("gold_grades") or {},
                "hit_ids": ranked,
                "metrics": metrics,
                "n_hits": len(ranked),
                "retrieve_ms": per_q_ms,
                "embed_ms": per_q_ms,
                "client_wall_ms": per_q_ms,
            }
        )
    metrics = aggregate_query_metrics(per_query, ks=ks) if per_query else {}
    return metrics, per_query


def run_local_dense(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    dataset_name: str,
    k: int = 50,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    brain_id: str | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    docs, queries = split_corpus(rows)
    model = SentenceTransformer(model_name)
    ranked_by_qid, encode_ms = retrieve_dense(model, queries, docs, k=k)
    metrics, per_query = evaluate_dense(
        ranked_by_qid, queries, ks=ks, encode_ms=encode_ms
    )
    return {
        "status": "ok",
        "brain_id": brain_id or "harness-local",
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "none",
        "channels": ["harness-dense"],
        "rank_pool": False,
        "skip_enrichment": True,
        "k": k,
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_docs_mapped": len(docs),
        "ce_model": model_name,
        "protocol": protocol
        or (
            "local dual-encoder over search_esci_74.jsonl; not BrainAPI embeddings; "
            "not Reddy 0.857"
        ),
        "metrics": metrics,
        "queries": per_query,
        "ingest": {"status": "completed", "n_docs": 0, "tasks": [], "reused": True},
    }
